"""
Бенчмарк реальной скорости задеплоенного qwen_server.py -- измеряет TTFT
(время до первого токена) и скорость генерации (токенов/сек) на настоящем
сервере, а не по теоретическим оценкам.

Токен и URL передаются через переменные окружения (не хардкодятся в файле,
чтобы не утекли при коммите в git):

    export QWEN_ENDPOINT_URL="https://<workspace>--qwen36-27b-model-generate.modal.run"
    export QWEN_AUTH_TOKEN="<тот же токен, что в modal secret create>"
    python3 benchmark.py

Опционально: --concurrency N для теста параллельной нагрузки (по умолчанию
тестирует single-stream, затем сравнивает с параллельными запросами).
"""

import argparse
import asyncio
import json
import os
import sys
import time

try:
    import httpx
except ImportError:
    print("Нужен httpx: pip install httpx", file=sys.stderr)
    sys.exit(1)


PROMPTS = [
    "Объясни разницу между MoE и dense моделями в двух абзацах.",
    "Напиши функцию на Python для быстрой сортировки.",
    "Что такое спекулятивное декодирование в LLM?",
]


async def single_request(client: httpx.AsyncClient, url: str, token: str, prompt: str, max_tokens: int = 350):
    """Один запрос со стримингом -- возвращает TTFT, полное время, число токенов (по дельтам)."""
    start = time.perf_counter()
    first_token_time = None
    delta_count = 0
    full_text = ""

    async with client.stream(
        "POST",
        url,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
        json={"text": prompt, "max_tokens": max_tokens, "stream": True},
        timeout=120.0,
    ) as resp:
        if resp.status_code == 401:
            raise RuntimeError("401 Unauthorized -- проверьте QWEN_AUTH_TOKEN")
        resp.raise_for_status()

        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                continue
            delta = data.get("delta", "")
            if delta:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                delta_count += 1
                full_text += delta

    end = time.perf_counter()
    ttft = (first_token_time - start) if first_token_time else None
    total_time = end - start
    decode_time = (end - first_token_time) if first_token_time else 0
    # delta_count -- число SSE-чанков, не точное число токенов (зависит от
    # того, как vLLM группирует дельты), но это разумная прокси-метрика --
    # для точного числа токенов нужен tokenizer на клиенте, что избыточно
    # для простого бенчмарка.
    tokens_per_sec = delta_count / decode_time if decode_time > 0 else 0

    return {
        "ttft": ttft,
        "total_time": total_time,
        "delta_count": delta_count,
        "tokens_per_sec": tokens_per_sec,
        "response_preview": full_text[:80] + ("..." if len(full_text) > 80 else ""),
    }


async def run_single_stream_test(url: str, token: str):
    print("=== Single-stream тест (по одному запросу за раз) ===\n")
    async with httpx.AsyncClient() as client:
        for i, prompt in enumerate(PROMPTS, 1):
            print(f"[{i}/{len(PROMPTS)}] Промпт: {prompt[:60]}...")
            try:
                result = await single_request(client, url, token, prompt)
            except Exception as e:
                print(f"  Ошибка: {e}\n")
                continue

            ttft_str = f"{result['ttft']:.3f}с" if result["ttft"] is not None else "н/д"
            print(f"  TTFT: {ttft_str}")
            print(f"  Полное время: {result['total_time']:.3f}с")
            print(f"  Скорость декода: {result['tokens_per_sec']:.1f} чанков/сек")
            print(f"  Ответ: {result['response_preview']}\n")


async def run_concurrency_test(url: str, token: str, concurrency: int):
    print(f"=== Тест параллельной нагрузки ({concurrency} одновременных запросов) ===\n")
    prompt = PROMPTS[0]

    async with httpx.AsyncClient() as client:
        start = time.perf_counter()
        tasks = [single_request(client, url, token, prompt) for _ in range(concurrency)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end = time.perf_counter()

    ok_results = [r for r in results if isinstance(r, dict)]
    errors = [r for r in results if not isinstance(r, dict)]

    if errors:
        print(f"Ошибок: {len(errors)}/{concurrency}")
        for e in errors[:3]:
            print(f"  {e}")

    if ok_results:
        avg_ttft = sum(r["ttft"] for r in ok_results if r["ttft"]) / len(ok_results)
        total_deltas = sum(r["delta_count"] for r in ok_results)
        wall_time = end - start
        aggregate_throughput = total_deltas / wall_time if wall_time > 0 else 0

        print(f"Успешных запросов: {len(ok_results)}/{concurrency}")
        print(f"Средний TTFT: {avg_ttft:.3f}с")
        print(f"Общее время выполнения (все параллельно): {wall_time:.3f}с")
        print(f"Суммарный throughput: {aggregate_throughput:.1f} чанков/сек (агрегировано по всем запросам)")


async def main():
    parser = argparse.ArgumentParser(description="Бенчмарк qwen_server.py")
    parser.add_argument("--concurrency", type=int, default=4, help="Число параллельных запросов для теста батча")
    parser.add_argument("--skip-single", action="store_true", help="Пропустить single-stream тест")
    parser.add_argument("--skip-concurrency", action="store_true", help="Пропустить тест параллельной нагрузки")
    args = parser.parse_args()

    url = os.environ.get("QWEN_ENDPOINT_URL")
    token = os.environ.get("QWEN_AUTH_TOKEN")

    if not url:
        print("Задайте QWEN_ENDPOINT_URL (URL вашего Modal-эндпоинта)", file=sys.stderr)
        sys.exit(1)
    if not token:
        print("Задайте QWEN_AUTH_TOKEN (тот же токен, что в modal secret create)", file=sys.stderr)
        sys.exit(1)

    if not args.skip_single:
        await run_single_stream_test(url, token)

    if not args.skip_concurrency:
        await run_concurrency_test(url, token, args.concurrency)


if __name__ == "__main__":
    asyncio.run(main())
