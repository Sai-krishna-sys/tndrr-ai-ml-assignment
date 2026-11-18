
import json
import csv
from pathlib import Path

folder = Path(__file__).resolve().parent

memory_values = {
    "qwen2.5:1.5b-instruct-q4_K_M": 1112.48,
    "qwen2.5:1.5b-instruct-q8_0": 1739.20
}

json_file = folder / "benchmark_results.json"
csv_file = folder / "benchmark_results.csv"

with open(json_file, "r", encoding="utf-8") as f:
    results = json.load(f)

for row in results:
    model = row["model"]

    # Remove the earlier incomplete memory measurement.
    row.pop("peak_ollama_memory_mb", None)

    # Add corrected runner memory snapshot.
    row["runner_working_set_mb"] = memory_values[model]
    row["memory_measurement"] = (
        "llama-server process working-set snapshot after inference; "
        "not peak memory"
    )

with open(json_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

with open(csv_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("Benchmark results corrected successfully.")
print("Updated:", json_file)
print("Updated:", csv_file)
