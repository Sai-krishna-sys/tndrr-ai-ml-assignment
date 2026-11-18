# Task 1: LoRA Fine-Tuning Report

## 1. Objective
Fine-tune a language model using Parameter-Efficient Fine-Tuning (LoRA) and compare its performance with the original base model.

## 2. Model and Environment
- Base model: Qwen2.5-0.5B-Instruct
- Parameters: Approximately 494 million
- Training environment: Google Colab
- GPU: NVIDIA Tesla T4
- Training precision: FP16
- Fine-tuning method: LoRA

## 3. Dataset
A synthetic structured-output dataset was generated with 768 examples:
- Training: 600 examples
- Validation: 96 examples
- Testing: 104 examples

The task was to convert user requests into JSON containing four fields: task, date, time, and category.

The test set used task-date-time combinations excluded from training and validation.

## 4. LoRA Configuration
- Rank (r): 8
- Alpha: 16
- Dropout: 0.05
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- Learning rate: 0.0002
- Epochs: 3
- Batch size: 4
- Gradient accumulation: 4

Trainable parameters: 4,399,104  
Total parameters: 498,431,872  
Trainable percentage: Approximately 0.88%.

## 5. Training Results

| Epoch | Training Loss | Validation Loss |
|---|---:|---:|
| 1 | 0.0357 | 0.0001 |
| 2 | 0.0001 | 0.0001 |
| 3 | 0.0000 | 0.0000 |

The training and validation losses decreased substantially over three epochs.

## 6. Evaluation Results

Both models were evaluated on the same 104 test examples using the same system prompt.

| Metric | Base Model | LoRA Fine-Tuned |
|---|---:|---:|
| Valid JSON | 100% | 100% |
| Exact-match accuracy | 0% | 100% |
| Correct examples | 0/104 | 104/104 |

The fine-tuned model learned to produce the expected structured JSON outputs on this test set.

## 7. Memory Estimates for a 7B Model

Approximate memory requirements for full fine-tuning with AdamW:
- Model weights (FP16): 14 GB
- Gradients (FP16): 14 GB
- Adam optimizer states (FP32): 56 GB
- FP32 master weights: 28 GB
- Total: Approximately 112 GB before activation memory.

LoRA reduces trainable parameters and optimizer memory. QLoRA additionally stores the base model in 4-bit quantization, reducing GPU memory requirements further. Actual requirements depend on batch size, sequence length, optimizer, and implementation.

## 8. Limitations and Failure Analysis

The dataset is synthetic and contains a limited set of task values, dates, times, and sentence templates. Although task-date-time combinations were separated across splits, individual values and templates were shared.

The 100% exact-match result applies only to these 104 synthetic held-out examples. It does not demonstrate 100% accuracy on independent real-world requests.

Further evaluation should use independently collected real-world examples, unseen task categories, and more varied language.

## 9. Conclusion

LoRA fine-tuning improved exact-match performance from 0% to 100% on the synthetic held-out test set while training approximately 0.88% of the model parameters. The experiment demonstrates the usefulness of parameter-efficient fine-tuning for structured-output tasks, subject to the evaluation limitations described above.
