"""
Self-hosted Qwen3.6-27B на Beam.cloud serverless GPU.

Modal потребовал банковскую карту именно для доступа к GPU-функциям (A10G) --
даже в рамках бесплатного $30/мес тира. Проверяем, ведёт ли себя так же Beam.cloud.

ПРИМЕЧАНИЕ: готовая интеграция beta9.abstractions.integrations.vllm.VLLM
существует в пакете, но её внутренний импорт `from ...type import LLM_APP_KIND`
падает с ImportError -- этого имени нет в beta9.type в версии 0.1.268.
Это рассинхронизация внутри самого пакета Beam, не наша ошибка. Поэтому
здесь -- собственный минимальный ASGI-эндпоинт на чистом vLLM через
декоратор @asgi, без сломанной обёртки.

Установка:   pip install beam-client
Авторизация: beam config create default
Деплой:      beam deploy beam_server.py:qwen36
"""

from beam import Image, Volume, asgi

MODEL_REPO_ID = "QuantTrio/Qwen3.6-27B-AWQ"  # тот же community AWQ-квант, что и в Modal-варианте

# Кеш весов между холодными стартами -- Volume монтируется в контейнер
# и переживает редеплои, веса не перекачиваются заново на каждый деплой.
weights_cache = Volume(name="qwen36-weights", mount_path="./qwen36-weights")

image = (
    Image(python_version="python3.11")
    .add_python_packages(["vllm==0.8.4", "huggingface_hub[hf_transfer,hf_xet]", "fastapi"])
    .with_envs(["HF_HUB_ENABLE_HF_TRANSFER=1", "HF_XET_HIGH_PERFORMANCE=1"])
)


def init_llm():
    from vllm import LLM

    return LLM(
        model=MODEL_REPO_ID,
        download_dir=weights_cache.mount_path,
        quantization="awq",
        max_model_len=8192,  # без YaRN -- диалоги/код не теряют в качестве
        gpu_memory_utilization=0.90,
        enable_prefix_caching=True,  # экономит recompute при повторяющемся system prompt
    )


@asgi(
    name="qwen36-27b",
    cpu=4,
    memory="24Gi",
    gpu="A10G",  # 24GB VRAM -- как и в Modal-варианте
    image=image,
    volumes=[weights_cache],
    on_start=init_llm,  # выполняется один раз при поднятии контейнера, результат кешируется
    keep_warm_seconds=90,  # держим контейнер тёплым 90с после последнего запроса
)
def qwen36(context):
    from fastapi import FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    from vllm import SamplingParams

    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    llm = context.on_start_value  # результат init_llm(), общий на весь контейнер

    @app.post("/generate")
    async def generate(request: Request):
        body = await request.json()
        params = SamplingParams(temperature=0.7, max_tokens=body.get("max_tokens", 350))
        out = llm.generate([body["text"]], params)
        return {"response": out[0].outputs[0].text}

    return app


# Прогрев под реальные часы активности -- заведите отдельно, если понадобится:
#
#   from beam import schedule
#
#   @schedule(when="0 8,20 * * *")
#   def warmup_ping():
#       import requests
#       requests.post(
#           "https://<ваш-деплой>.beam.cloud/generate",
#           json={"text": "ping", "max_tokens": 1},
#       )
