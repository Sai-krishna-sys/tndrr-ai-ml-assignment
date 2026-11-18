# Tndrr AI/ML Internship Assignment

## Project Overview

This repository contains three AI/ML tasks completed as part of the internship assignment.

## Task 1: LoRA Fine-Tuning

**Folder:** `task1-lora-finetuning/`

- Fine-tuned Qwen2.5-0.5B-Instruct using LoRA.
- Evaluated base and fine-tuned models on 104 synthetic test examples.
- Base model exact-match accuracy: 0%.
- Fine-tuned model exact-match accuracy: 100%.
- Includes adapter weights, evaluation results, and report.

**Report:** `task1-lora-finetuning/task1_report.md`

## Task 2: Retrieval-Augmented Generation (RAG)

**Folder:** `task2-rag-system/`

- Built a BM25 retrieval system from scratch.
- Used the NIST AI Risk Management Framework (AI RMF 1.0), a 48-page document.
- Compared 200-word and 400-word chunk configurations.
- Small chunks Recall@5: 80.00%.
- Large chunks Recall@5: 86.67%.
- Includes extracted document pages, chunks, evaluation questions, retrieval results, and source code.

**Report:** `task2-rag-system/task2_report.md`

## Task 3: Local Model Quantization and Benchmarking

**Folder:** `task3-local-optimisation/`

- Tested Qwen2.5-1.5B-Instruct using Q4_K_M and Q8_0 quantization.
- Benchmarked 10 prompts for each configuration.
- Q4 average speed: 18.53 tokens/second.
- Q8 average speed: 12.17 tokens/second.
- Conducted CPU-based inference because no dedicated NVIDIA GPU was available.

**Report:** `task3-local-optimisation/task3_report.md`

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- PEFT and LoRA
- Google Colab
- BM25
- Ollama
- Qwen models

## Important Evaluation Notes

The Task 1 evaluation uses synthetic held-out combinations, and Task 2 uses 15 manually prepared questions. Task 3 measurements were performed on CPU. These results should be interpreted within the limitations documented in each task report.

## Project Structure

- `task1-lora-finetuning/` — Fine-tuning implementation and results.
- `task2-rag-system/` — RAG implementation, document data, and results.
- `task3-local-optimisation/` — Quantization benchmarks and results.
- `report/` — Additional report materials.
