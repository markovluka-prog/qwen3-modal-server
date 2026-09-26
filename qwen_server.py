import modal
from fastapi import Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

app = modal.App("qwen36-27b")

# HTTPBearer() создаётся один раз на уровне модуля -- Depends(auth_scheme)
# используется как значение по умолчанию параметра в сигнатуре эндпоинта,
# это стандартный FastAPI dependency-injection паттерн, подтверждённый
# официальным примером Modal для fastapi_endpoint.
auth_scheme = HTTPBearer()

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

# ВАЖНО: L40S/A100/H100 быстрее A10G, но дороже за час -- при вашей нагрузке
# (~1000 сообщений/день) L40S уже выходит за пределы бесплатных $30/мес
# Modal (расчёт: ~$32-130/мес против A10G ~$18-73/мес). Приоритет -- не терять
# большой бесплатный лимит, поэтому GPU зафиксирован на A10G. Дальше -- топ
# скорость исключительно программными средствами (все бесплатны).
GPU_TYPE = "A10G"

# max_inputs согласован с max_num_seqs ниже. Если оставить max_inputs=4 (как
# было раньше), continuous batching движка физически не увидит больше 4
# конкурентных запросов -- max_num_seqs становится мёртвым параметром.
# Стартовое значение для A10G 24GB + 27B AWQ (~16-18GB весов) -- дальше
# смотрите на "# GPU blocks"/"maximum concurrency" в логах запуска vLLM
# и поднимайте оба числа синхронно, если есть запас по VRAM.
MAX_NUM_SEQS = 48


# Bearer-токен для защиты эндпоинта -- иначе он открыт всем, у кого есть URL.
# Создать секрет один раз перед деплоем:
#   modal secret create qwen36-auth AUTH_TOKEN=<ваш-произвольный-токен>
AUTH_SECRET_NAME = "qwen36-auth"


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
    secrets=[modal.Secret.from_name(AUTH_SECRET_NAME)],
)
@modal.concurrent(max_inputs=MAX_NUM_SEQS)
class Model:
    @modal.enter()
    async def start_engine(self):
        # AsyncLLM (V1-движок) -- единственный способ получить настоящий
        # token-by-token streaming в vLLM 0.29+. Синхронный класс `LLM`
        # (offline batch API) стриминг не поддерживает вообще -- он
        # возвращает список RequestOutput целиком после полного завершения.
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        engine_args = AsyncEngineArgs(
            model=MODEL_PATH,
            quantization="awq",
            max_model_len=8192,
            gpu_memory_utilization=0.90,
            enable_prefix_caching=True,   # пропускает recompute при повторяющемся system prompt
            enable_chunked_prefill=True,  # снижает TTFT при длинных промптах, не блокирует decode
            # --- Continuous batching: подобраны под A10G 24GB + 27B AWQ. ---
            # Смотрите лог старта vLLM ("# GPU blocks", "maximum concurrency")
            # и поднимайте синхронно с MAX_NUM_SEQS выше, если есть запас VRAM.
            max_num_seqs=MAX_NUM_SEQS,
            max_num_batched_tokens=4096,  # компромисс TTFT/throughput при плотной VRAM
            # --- Quantized KV cache: освобождает VRAM под больший batch. ---
            # Официально задокументированный эффект: вдвое меньше KV cache ->
            # либо больше конкурентных запросов, либо длиннее контекст при
            # той же памяти. Честная оговорка: без scale-калибровки через
            # llm-compressor возможна лёгкая деградация точности, особенно
            # на длинных контекстах/математике -- проверьте на своих задачах,
            # откатите на "auto", если качество ответов заметно просядет.
            kv_cache_dtype="fp8",
            # --- CUDA graphs: НЕ отключаем ради снапшота. Устаревшее ---
            # предположение (eager нужен для совместимости с Modal GPU
            # snapshot) не подтвердилось: официальный блог Modal прямо
            # рекомендует делать warmup ДО снапшота именно чтобы CUDA graphs
            # попали в сохранённое состояние и не пересобирались при cold
            # start. enforce_eager здесь оставлен по умолчанию (False).
            compilation_config={"cudagraph_mode": "FULL_AND_PIECEWISE"},
            # --- Спекулятивное декодирование: n-gram, без риска по VRAM. ---
            # Draft-модель того же семейства не влезет рядом с уже занятыми
            # ~16-18GB весов на A10G 24GB -- n-gram не требует второй модели
            # вообще. Эффективность зависит от повторяемости контента (код,
            # структурированные ответы -- да; свободная проза -- под
            # вопросом) -- сравните throughput/TTFT до и после на реальном
            # трафике, это не гарантированный, а вероятностный выигрыш.
            speculative_config={
                "method": "ngram",
                "num_speculative_tokens": 4,
                "prompt_lookup_min": 2,
                "prompt_lookup_max": 5,
            },
        )
        self.engine = AsyncLLM.from_engine_args(engine_args)

        # Explicit warmup ПЕРЕД снапшотом -- без него CUDA graphs/torch.compile
        # артефакты не попадают в сохранённое состояние, и каждый cold start
        # платит за их пересборку заново, теряя весь смысл снапшота.
        from vllm import SamplingParams
        from vllm.utils import random_uuid

        warmup_params = SamplingParams(max_tokens=32)
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
    async def generate(
        self,
        prompt: dict,
        token: HTTPAuthorizationCredentials = Depends(auth_scheme),
    ):
        from fastapi import HTTPException, status
        from fastapi.responses import StreamingResponse, JSONResponse
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind
        from vllm.utils import random_uuid
        import json
        import os

        cors_headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
        }

        if token.credentials != os.environ["AUTH_TOKEN"]:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Неверный bearer-токен",
                headers={"WWW-Authenticate": "Bearer", **cors_headers},
            )

        stream = prompt.get("stream", True)  # стрим по умолчанию -- это и есть "топ скорость"

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
        # Preflight-запрос браузера НЕ несёт заголовок Authorization -- это
        # штатное поведение CORS, поэтому здесь авторизация не проверяется.
        # Именно "Authorization" в Allow-Headers ниже разрешает браузеру
        # отправить его на следующем реальном POST-запросе.
        from fastapi import Response

        return Response(
            status_code=204,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization",
            },
        )


@app.function(schedule=modal.Cron("0 8,20 * * *"))
def warmup_ping():
    Model()._generate_once.remote({"text": "ping", "max_tokens": 1})
