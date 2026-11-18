# Task 1 - LoRA Fine-Tuning of a Small Language Model

## Goal

Fine-tune a small open-source causal language model on a domain-specific dataset using
**LoRA (Low-Rank Adaptation)**, a parameter-efficient fine-tuning (PEFT) method.

## Why LoRA?

Full fine-tuning updates every weight of the model, which is expensive in memory and
compute. LoRA instead freezes the base model and injects small trainable rank-decomposition
matrices (A, B) into selected layers (usually attention projections). Only those small
matrices are trained, drastically reducing trainable parameters while keeping quality.

Key concepts:
- **Base model weights are frozen** (`requires_grad=False`).
- **Adapter matrices** A (d x r) and B (r x d) with r << d are trained.
- **Merge at inference**: adapter weights can be merged back into base weights.
- **Rank (r)** and **alpha** control capacity and scaling (`scale = alpha / r`).

## Planned pipeline

1. Load base model + tokenizer (Hugging Face `transformers`).
2. Prepare dataset (instruction / completion pairs), tokenize.
3. Configure LoRA via `peft.LoraConfig` (target modules, r, alpha, dropout).
4. Train with `transformers.Trainer` (or TRL SFTTrainer).
5. Save adapter (not full model).
6. Evaluate: perplexity / task metric on a held-out split.
7. (Optional) Merge adapter and export.

## Directory layout

- `src/` - Python modules (data, model, train, eval).
- `data/` - raw and processed datasets (not committed if large).
- `notebooks/` - exploratory notebooks.
- `outputs/` - checkpoints and saved adapters.

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## Execution (to be implemented)

```bash
python src/train.py --config configs/lora.yaml
python src/evaluate.py --adapter outputs/lora-adapter
```

## Evaluation

All metrics (perplexity, loss curves) are produced by actual training runs and stored in
`outputs/`. No numbers are fabricated.

## Notes for interview

Be ready to explain: why LoRA vs full FT, choice of rank, target modules, overfitting
signals, and how adapters are saved/loaded/merged.
