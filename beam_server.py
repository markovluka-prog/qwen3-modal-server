"""
Self-hosted Qwen3.6-27B на Beam.cloud serverless GPU.

Modal потребовал банковскую карту именно для доступа к GPU-функциям (A10G) --
даже в рамках бесплатного $30/мес тира. Beam.cloud, по совокупности источников
(официальный FAQ + маркетинг Developer tier), не требует карту для GPU на
Developer-плане -- $30/мес кредитов, до 5 GPU-контейнеров одновременно.
Это НЕ подтверждено первичным заявлением вида "no card for GPU" в документации,
поэтому первый деплой -- это и есть эмпирическая проверка данного факта.

Используется готовая интеграция beam.integrations.VLLM -- она сама поднимает
OpenAI-совместимый эндпоинт поверх vLLM, без ручного FastAPI-кода.

Установка:   pip install beam-client
Авторизация: beam config create
Деплой:      beam deploy beam_server.py:qwen36
"""

from beam import Image, Volume
from beam.integrations import VLLM, VLLMArgs

MODEL_REPO_ID = "QuantTrio/Qwen3.6-27B-AWQ"  # тот же community AWQ-квант, что и в Modal-варианте

# Кеш весов между холодными стартами -- аналог "bake into image" в Modal,
# но проще: Beam Volume монтируется в контейнер и переживает редеплои.
weights_cache = Volume(name="qwen36-weights", mount_path="./qwen36-weights")

qwen36 = VLLM(
    name="qwen36-27b",
    cpu=4,
    memory="24Gi",
    gpu="A10G",  # 24GB VRAM -- как и в Modal-варианте
    keep_warm_seconds=90,  # аналог scaledown_window: держим контейнер тёплым 90с после запроса
    volumes=[weights_cache],
    image=(
        Image(python_version="python3.11")
        .add_python_packages(["vllm==0.29.0", "huggingface_hub[hf_transfer,hf_xet]"])
        .with_envs(["HF_HUB_ENABLE_HF_TRANSFER=1", "HF_XET_HIGH_PERFORMANCE=1"])
    ),
    vllm_args=VLLMArgs(
        model=MODEL_REPO_ID,
        served_model_name=[MODEL_REPO_ID],
        download_dir=weights_cache.mount_path,
        quantization="awq",
        max_model_len=8192,  # без YaRN -- диалоги/код не теряют в качестве
        gpu_memory_utilization=0.90,
        enable_prefix_caching=True,  # экономит recompute при повторяющемся system prompt
    ),
)

# Прогрев под реальные часы активности -- заведите отдельно, если понадобится:
#
#   from beam import schedule
#
#   @schedule(when="0 8,20 * * *")
#   def warmup_ping():
#       import requests
#       requests.post(
#           "https://<ваш-деплой>.beam.cloud/v1/chat/completions",
#           json={"model": MODEL_REPO_ID, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 1},
#       )
