
import json
import csv
import time
import threading
import urllib.request
import psutil
from pathlib import Path

API = "http://localhost:11434/api/generate"

MODELS = [
    "qwen2.5:1.5b-instruct-q4_K_M",
    "qwen2.5:1.5b-instruct-q8_0"
]

PROMPTS = [
    "Explain artificial intelligence in three simple sentences.",
    "What is machine learning? Give two examples.",
    "Explain the difference between RAM and ROM.",
    "Write a Python function to check whether a number is prime.",
    "What are the advantages of cloud computing?",
    "Explain how a database index works.",
    "Describe the water cycle in simple terms.",
    "What is overfitting in machine learning?",
    "Explain the difference between supervised and unsupervised learning.",
    "Write five tips for improving cybersecurity."
]

def get_ollama_memory():
    total = 0
    for process in psutil.process_iter(["name", "memory_info"]):
        try:
            name = (process.info["name"] or "").lower()
            if "ollama" in name:
                total += process.info["memory_info"].rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total / (1024 ** 2)

def call_model(model, prompt):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "5m",
        "options": {
            "temperature": 0,
            "seed": 42,
            "num_predict": 128
        }
    }

    request = urllib.request.Request(
        API,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )

    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=600) as response:
        result = json.loads(response.read().decode("utf-8"))
    wall_time = time.perf_counter() - start

    eval_count = result.get("eval_count", 0)
    eval_duration = result.get("eval_duration", 0)
    tokens_per_second = (
        eval_count / (eval_duration / 1e9)
        if eval_duration else 0
    )

    return {
        "response": result.get("response", "").strip(),
        "tokens": eval_count,
        "tokens_per_second": round(tokens_per_second, 2),
        "wall_time_seconds": round(wall_time, 2),
        "total_duration_seconds": round(
            result.get("total_duration", 0) / 1e9, 2
        ),
        "prompt_tokens": result.get("prompt_eval_count", 0)
    }

all_results = []

for model in MODELS:
    print("\n" + "=" * 65)
    print("BENCHMARKING:", model)
    print("=" * 65)

    peak_memory = [get_ollama_memory()]
    monitoring = True

    def monitor_memory():
        while monitoring:
            peak_memory[0] = max(
                peak_memory[0], get_ollama_memory()
            )
            time.sleep(0.1)

    monitor = threading.Thread(target=monitor_memory)
    monitor.start()

    try:
        for index, prompt in enumerate(PROMPTS, start=1):
            print(f"\nPrompt {index}/10")
            print(prompt)

            metrics = call_model(model, prompt)

            row = {
                "model": model,
                "prompt_number": index,
                "prompt": prompt,
                **metrics
            }
            all_results.append(row)

            print("Tokens:", metrics["tokens"])
            print("Tokens/sec:", metrics["tokens_per_second"])
            print("Wall time:", metrics["wall_time_seconds"], "seconds")
            print("Response:", metrics["response"][:250])

    finally:
        monitoring = False
        monitor.join()

    print("\nPeak Ollama process memory:",
          round(peak_memory[0], 2), "MB")

    for row in all_results:
        if row["model"] == model:
            row["peak_ollama_memory_mb"] = round(peak_memory[0], 2)

    # Unload this model before loading the next one.
    unload_payload = {
        "model": model,
        "keep_alive": 0
    }
    unload_request = urllib.request.Request(
        API,
        data=json.dumps(unload_payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    try:
        urllib.request.urlopen(unload_request, timeout=120).read()
    except Exception as error:
        print("Unload warning:", error)

# Save detailed results.
output_dir = Path(__file__).resolve().parent
json_path = output_dir / "benchmark_results.json"
csv_path = output_dir / "benchmark_results.csv"

with open(json_path, "w", encoding="utf-8") as file:
    json.dump(all_results, file, indent=2, ensure_ascii=False)

with open(csv_path, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=all_results[0].keys()
    )
    writer.writeheader()
    writer.writerows(all_results)

print("\n" + "=" * 65)
print("BENCHMARK SUMMARY")
print("=" * 65)

for model in MODELS:
    rows = [r for r in all_results if r["model"] == model]
    average_speed = sum(
        r["tokens_per_second"] for r in rows
    ) / len(rows)
    average_time = sum(
        r["wall_time_seconds"] for r in rows
    ) / len(rows)
    memory = rows[0]["peak_ollama_memory_mb"]

    print("\nModel:", model)
    print("Average generation speed:", round(average_speed, 2), "tokens/sec")
    print("Average wall time:", round(average_time, 2), "seconds")
    print("Peak Ollama process memory:", memory, "MB")

print("\nSaved:", json_path)
print("Saved:", csv_path)
