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
    # vLLM 0.9.0 устарел более чем на год к сентябрю 2026 и не поддержит Qwen3.6.
    # Берём актуальную ветку без жёсткого пина на конкретный патч.
    .pip_install(
        "vllm>=0.29,<0.31",
        "huggingface_hub[hf_transfer,hf_xet]",
        "fastapi[standard]",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_XET_HIGH_PERFORMANCE": "1"})
    .run_function(download_weights)
)

MODEL_PATH = "/model"

# Модель, которую собираем в fetch()-ответ на клиенте
ALLOWED_ORIGINS = ["*"]  # в проде сузьте до конкретного домена фронтенда


@app.cls(
    image=image,
    gpu="A10G",
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
    @modal.enter(snap=True)
    def load(self):
        from vllm import LLM

        self.llm = LLM(
            model=MODEL_PATH,
            quantization="awq",
            max_model_len=8192,
            gpu_memory_utilization=0.90,
            max_num_seqs=4,
            enable_prefix_caching=True,
            # enforce_eager убран: с GPU-снапшотами имеет смысл снапшотить
            # уже скомпилированные CUDA-графы, eager-режим только режет
            # throughput и не нужен для корректности снапшота.
        )
        # Прогрев форвард-пассом ДО снапшота -- Modal рекомендует переносить
        # инициализационную работу в фазу snap=True, чтобы она не повторялась
        # при каждом restore.
        from vllm import SamplingParams

        self.llm.generate(["ping"], SamplingParams(max_tokens=1))

    @modal.method()
    def _generate(self, prompt: dict) -> dict:
        from vllm import SamplingParams

        params = SamplingParams(
            temperature=0.7, max_tokens=prompt.get("max_tokens", 350)
        )
        out = self.llm.generate([prompt["text"]], params)
        return {"response": out[0].outputs[0].text}

    @modal.fastapi_endpoint(method="POST")
    def generate(self, prompt: dict):
        from fastapi import Response
        import json

        result = self._generate.local(prompt)
        # Modal НЕ добавляет CORS-заголовки автоматически -- без них
        # браузерный fetch() с другого origin будет заблокирован.
        return Response(
            content=json.dumps(result),
            media_type="application/json",
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
            },
        )

    @modal.fastapi_endpoint(method="OPTIONS")
    def generate_options(self):
        # Preflight-запрос браузера перед POST с Content-Type: application/json
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
    # Прогрев дергает бизнес-логику напрямую через @modal.method(),
    # а не HTTP fastapi_endpoint -- вызов .remote() на fastapi_endpoint
    # методе не проходит штатный путь ASGI/FastAPI-валидации и ненадёжен.
    Model()._generate.remote({"text": "ping", "max_tokens": 1})
