# Task 3 - Local Model Optimisation

## Goal

Take a trained model and make it fast/small enough to run on a local machine: quantization,
ONNX export, (optional) pruning, and benchmark latency/throughput/memory before vs. after.

## Concepts

- **Quantization**: reduce weight precision (FP32 -> FP16/INT8/INT4) to cut memory and
  speed up compute; trade-off with accuracy.
- **ONNX export**: convert PyTorch model to ONNX for optimized runtime (ONNX Runtime).
- **Pruning**: remove low-magnitude weights (structured/unstructured) for sparsity.
- **Benchmarking**: measure latency, throughput, peak memory; compare quality metric.
- **Trade-off analysis**: accuracy vs. size vs. speed.

## Planned pipeline

1. Load base model (from Task 1 adapter merged, or a small pretrained model).
2. Quantize (dynamic/static/QLoRA-style) and export to ONNX.
3. Benchmark baseline vs. optimized (latency, memory, size).
4. Evaluate quality metric on a held-out set.
5. Report Pareto trade-offs.

## Directory layout

- `src/` - export, quantize, benchmark, eval modules.
- `models/` - exported/quantized artifacts.
- `benchmarks/` - saved benchmark JSON/CSV.
- `notebooks/` - exploration.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
source .venv/bin/activate # Linux/macOS
pip install -r requirements.txt
```

## Execution (to be implemented)

```bash
python src/export_onnx.py --model <path>
python src/quantize.py --model <path> --bits 8
python src/benchmark.py --baseline <path> --optimized <path>
```

## Evaluation

All latency/memory/accuracy numbers come from real runs on the target machine and are
saved to `benchmarks/`. No fabricated results.
