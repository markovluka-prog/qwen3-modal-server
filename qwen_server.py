"""
Self-hosted Qwen3.6-27B on Modal.com serverless GPU.

$0 при обычном паттерне использования (recurring $30/мес free credits),
без карты, без телефона, свой контейнер (не сторонний managed API модели).

Оптимизации:
- Веса запечены в Docker-образ на этапе сборки (не Volume) — качаются один раз навсегда.
- AWQ-квантование вместо GGUF — нативный mmap + packed-int4 CUDA-кернелы в vLLM.
- GPU/CPU memory snapshot — пропускает CUDA-инициализацию при повторных пробуждениях.
- Cron-прогрев 2 раза/сутки — почти устраняет ощутимый холодный старт в реальном использовании.
- max_model_len=8192 без YaRN — для диалогов/кода этого достаточно, без потери точности.

Деплой: modal deploy qwen_server.py
"""

import modal

app = modal.App("qwen36-27b")


def download_weights():
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="Qwen/Qwen3.6-27B-Instruct-AWQ",  # AWQ-сборка, не GGUF
        local_dir="/model",
    )


image = (
    modal.Image.debian_slim()
    .pip_install("vllm==0.9.0", "huggingface_hub[hf_transfer,hf_xet]", "fastapi[standard]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_XET_HIGH_PERFORMANCE": "1"})
    .run_function(download_weights)  # веса запекаются в image-слой при сборке
)

MODEL_PATH = "/model"


@app.cls(
    image=image,
    gpu="A10G",
    scaledown_window=90,  # короткое окно — экономит кредиты между сессиями
    timeout=120,
    enable_memory_snapshot=True,  # CPU snapshot: секунды вместо полной инициализации
    experimental_options={"enable_gpu_snapshot": True},  # пропускает CUDA-init/warmup
)
@modal.concurrent(max_inputs=4)
class Model:
    @modal.enter(snap=True)
    def load(self):
        from vllm import LLM

        self.llm = LLM(
            model=MODEL_PATH,
            quantization="awq",
            max_model_len=8192,  # без YaRN — короткие диалоги/код не теряют в качестве
            gpu_memory_utilization=0.90,
            max_num_seqs=4,
            enable_chunked_prefill=True,
            enable_prefix_caching=True,  # экономит recompute при повторяющемся system prompt
            enforce_eager=True,  # совместимость с GPU snapshot (CUDA graphs могут его ломать)
        )

    @modal.fastapi_endpoint(method="POST")
    def generate(self, prompt: dict):
        from vllm import SamplingParams

        params = SamplingParams(
            temperature=0.7,
            max_tokens=prompt.get("max_tokens", 350),  # держим короче — экономит free tier
        )
        out = self.llm.generate([prompt["text"]], params)
        return {"response": out[0].outputs[0].text}


# Держим воркер тёплым к ожидаемым пикам — 2 раза в сутки, ~$2/мес сверху.
# Подстройте время под свои реальные часы обращений.
@app.function(schedule=modal.Cron("0 8,20 * * *"))
def warmup_ping():
    Model().generate.remote({"text": "ping", "max_tokens": 1})
