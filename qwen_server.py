import modal

app = modal.App("qwen36-27b")

# ВНИМАНИЕ: официального AWQ-кванта от команды Qwen не существует.
# Qwen/Qwen3.6-27B-Instruct-AWQ -- несуществующий repo_id, скачивание падало бы 404.
# Базовая модель называется Qwen/Qwen3.6-27B (без "-Instruct", thinking/non-thinking
# встроены в один чекпоинт). AWQ доступен только как community-квант, например:
MODEL_REPO_ID = "QuantTrio/Qwen3.6-27B-AWQ"


def download_weights():
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=MODEL_REPO_ID, local_dir="/model")


image = (
    modal.Image.debian_slim()
    .pip_install(
        "vllm==0.30.0",  # актуальная стабильная на сентябрь 2026, содержит V1 AsyncLLM
        "huggingface_hub[hf_transfer,hf_xet]",
        "fastapi[standard]",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_XET_HIGH_PERFORMANCE": "1"})
    .run_function(download_weights)
)

MODEL_PATH = "/model"

# ТОП СКОРОСТЬ: L40S ощутимо быстрее A10G на 27B-классе (больше tensor cores,
# выше пропускная способность памяти) при сопоставимом классе цены.
# Откатитесь на "A10G", если важнее бюджет, а не топ-скорость.
GPU_TYPE = "L40S"


@app.cls(
    image=image,
    gpu=GPU_TYPE,
    scaledown_window=90,
    timeout=120,
    enable_memory_snapshot=True,
    # GPU-снапшоты на сентябрь 2026 всё ещё Alpha-фича и живут именно
    # под experimental_options -- отдельного gpu_snapshot=True на верхнем
    # уровне @app.cls не появилось.
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(max_inputs=4)
class Model:
    @modal.enter()
    async def start_engine(self):
        # AsyncLLM (V1-движок) -- единственный способ получить настоящий
        # token-by-token streaming в vLLM 0.29+. Синхронный класс `LLM`
        # (offline batch API) стриминг не поддерживает вообще -- он
        # возвращает список RequestOutput целиком после полного завершения.
        # `vllm.AsyncLLMEngine` в текущих версиях -- просто алиас на этот
        # же класс (V0 AsyncLLMEngine физически удалён), импортируем
        # напрямую как рекомендует актуальная документация vLLM.
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        engine_args = AsyncEngineArgs(
            model=MODEL_PATH,
            quantization="awq",
            max_model_len=8192,
            gpu_memory_utilization=0.90,
            enable_prefix_caching=True,   # пропускает recompute при повторяющемся system prompt
            enable_chunked_prefill=True,  # снижает TTFT при длинных промптах, не блокирует decode
        )
        self.engine = AsyncLLM.from_engine_args(engine_args)

        # Прогрев -- первый реальный запрос после старта контейнера иначе
        # платит за CUDA graph capture/JIT. Гоняем короткий forward pass
        # синхронно до того, как контейнер примет трафик.
        from vllm import SamplingParams
        from vllm.utils import random_uuid

        warmup_params = SamplingParams(max_tokens=8)
        async for _ in self.engine.generate(
            prompt="ping", sampling_params=warmup_params, request_id=random_uuid()
        ):
            pass

    @modal.method()
    async def _generate_once(self, prompt: dict) -> dict:
        """Нестримящий путь -- используется прогревом по расписанию."""
        from vllm import SamplingParams
        from vllm.utils import random_uuid

        params = SamplingParams(temperature=0.7, max_tokens=prompt.get("max_tokens", 350))
        text = ""
        async for output in self.engine.generate(
            prompt=prompt["text"], sampling_params=params, request_id=random_uuid()
        ):
            if output.outputs:
                text = output.outputs[0].text  # без DELTA -- накопленная строка
        return {"response": text}

    @modal.fastapi_endpoint(method="POST")
    async def generate(self, prompt: dict):
        from fastapi.responses import StreamingResponse, JSONResponse
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind
        from vllm.utils import random_uuid
        import json

        stream = prompt.get("stream", True)  # стрим по умолчанию -- это и есть "топ скорость"
        cors_headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        }

        if not stream:
            result = await self._generate_once.local(prompt)
            return JSONResponse(content=result, headers=cors_headers)

        params = SamplingParams(
            temperature=0.7,
            max_tokens=prompt.get("max_tokens", 350),
            output_kind=RequestOutputKind.DELTA,  # только новый кусочек на каждой итерации
        )
        request_id = random_uuid()

        async def sse_generator():
            async for output in self.engine.generate(
                prompt=prompt["text"], sampling_params=params, request_id=request_id
            ):
                for completion in output.outputs:
                    delta = completion.text
                    if delta:
                        yield f"data: {json.dumps({'delta': delta}, ensure_ascii=False)}\n\n"
                if output.finished:
                    break
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            sse_generator(),
            media_type="text/event-stream",
            headers={
                **cors_headers,
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # отключает буферизацию прокси перед клиентом
            },
        )

    @modal.fastapi_endpoint(method="OPTIONS")
    def generate_options(self):
        from fastapi import Response

        return Response(
            status_code=204,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
            },
        )


@app.function(schedule=modal.Cron("0 8,20 * * *"))
def warmup_ping():
    Model()._generate_once.remote({"text": "ping", "max_tokens": 1})
