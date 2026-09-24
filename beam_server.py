"""
Self-hosted Qwen3.6-27B на Beam.cloud serverless GPU.

Modal потребовал банковскую карту именно для доступа к GPU-функциям (A10G) --
даже в рамках бесплатного $30/мес тира. Проверяем, ведёт ли себя так же Beam.cloud.

Используется готовая интеграция beta9.abstractions.integrations.vllm.VLLM --
класс существует в реальном пакете, но НЕ реэкспортируется через публичные
фасады `beam.integrations` / `beta9.integrations` (там реэкспортирован только
MCPServer) -- поэтому импорт идёт напрямую из внутреннего модуля. Проверено
распаковкой реального .whl пакета beta9==0.1.268.

VLLM.__init__ сам добавляет `vllm==<vllm_version>` в образ (параметр
vllm_version, по умолчанию "0.8.4") -- поэтому НЕ указываем vllm вручную
в Image.add_python_packages, чтобы не конфликтовать с этим механизмом.

Установка:   pip install beam-client
Авторизация: beam config create default
Деплой:      beam deploy beam_server.py:qwen36
"""

from beam import Image, Volume
from beta9.abstractions.integrations.vllm import VLLM, VLLMArgs

MODEL_REPO_ID = "QuantTrio/Qwen3.6-27B-AWQ"  # тот же community AWQ-квант, что и в Modal-варианте

# Кеш весов между холодными стартами -- аналог "bake into image" в Modal,
# но проще: Beam Volume монтируется в контейнер и переживает редеплои.
weights_cache = Volume(name="qwen36-weights", mount_path="./qwen36-weights")

qwen36 = VLLM(
    name="qwen36-27b",
    cpu=4,
    memory="24Gi",
    gpu="A10G",  # 24GB VRAM -- как и в Modal-варианте
    keep_warm_seconds=90,  # держим контейнер тёплым 90с после запроса
    vllm_version="0.8.4",  # можно поднять, если нужна поддержка Qwen3.6 -- см. примечание ниже
    volumes=[weights_cache],
    image=Image(python_version="python3.11").with_envs(
        ["HF_HUB_ENABLE_HF_TRANSFER=1", "HF_XET_HIGH_PERFORMANCE=1"]
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

# ВАЖНО: если деплой упадёт на несовместимости vllm 0.8.4 с архитектурой
# Qwen3.6 (модель может требовать более новый vLLM) -- поднимите vllm_version
# на конструкторе VLLM выше (например "0.9.0" или новее) и повторите деплой.

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
