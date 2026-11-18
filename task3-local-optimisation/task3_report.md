\# Task 3: Local Model Optimization and Quantization Benchmark



\## 1. Objective



The objective was to run a local language model using two quantization formats, compare inference speed and memory usage, evaluate responses to identical prompts, and explain how a model can serve multiple concurrent users on a 24 GB GPU.



\## 2. Hardware and Software Environment



| Component | Specification |

|---|---|

| CPU | AMD Ryzen 5 7520U |

| System RAM | 15.24 GB |

| GPU | AMD Radeon integrated graphics |

| Dedicated VRAM reported | 0.5 GB |

| Operating System | Windows |

| Python | 3.11.9 |

| Ollama | 0.40.16 |

| Model | Qwen2.5 1.5B Instruct |



The experiments were performed using CPU inference because the computer did not have a suitable dedicated GPU.



\## 3. Quantization Configurations



The same Qwen2.5 1.5B Instruct model was tested using two quantization formats:



\- Q4\_K\_M: 4-bit quantization.

\- Q8\_0: 8-bit quantization.



| Model | Quantization | Disk Size |

|---|---|---:|

| Qwen2.5 1.5B Instruct | Q4\_K\_M | 986 MB |

| Qwen2.5 1.5B Instruct | Q8\_0 | 1.6 GB |



Quantization reduces the precision used to represent model weights. Lower-bit quantization generally reduces memory requirements and can improve inference speed, although it may affect output quality.



\## 4. Benchmark Methodology



Both models were tested using the same 10 prompts covering artificial intelligence, machine learning, computer memory, Python programming, cloud computing, databases, cybersecurity and general knowledge.



The benchmark recorded:



\- Generated token count.

\- Generation speed in tokens per second.

\- Total wall-clock response time.

\- Model runner memory usage.



The benchmark used Ollama's local API with temperature set to 0 and a maximum generation limit of 128 tokens.



The generation speed was calculated using Ollama's evaluation statistics:



Generation speed = Generated tokens / Generation time in seconds.



\## 5. Benchmark Results



| Metric | Q4\_K\_M | Q8\_0 |

|---|---:|---:|

| Average generation speed | 18.53 tokens/sec | 12.17 tokens/sec |

| Average wall time | 9.58 seconds | 13.14 seconds |

| Disk size | 986 MB | 1.6 GB |

| Ollama model allocation | 1.2 GB | 1.8 GB |

| Runner working-set snapshot | 1,112.48 MB | 1,739.20 MB |



The Q4 model generated approximately 52.3% more tokens per second than Q8 on the test computer.



The Q4 runner working-set measurement was approximately 36.0% lower than the Q8 measurement.



These memory readings are individual llama-server process working-set snapshots after inference, not peak memory measurements or total system memory usage.



\## 6. Response Quality Comparison



Both quantization formats generated responses to the same 10 prompts.



The responses were generally relevant and addressed the requested topics. However, several generations reached the configured 128-token output limit, which caused some answers to be incomplete.



A formal human-rated quality assessment was not performed. Therefore, the experiment does not establish that one quantization format produces more accurate answers than the other.



Q8 retains more weight precision than Q4 in principle, but this experiment primarily measured inference speed and memory usage.



\## 7. KV Cache



KV cache stands for Key-Value cache. It stores previously calculated attention keys and values during autoregressive text generation.



Without KV caching, a model would repeatedly calculate attention information for earlier tokens. KV caching avoids much of this repeated computation and improves generation efficiency.



KV cache memory consumption depends on:



\- Number of transformer layers.

\- Number of key-value attention heads.

\- Head dimension.

\- Context length.

\- Number of concurrent sequences.

\- Cache precision.



Longer context lengths and more concurrent users increase KV cache memory requirements.



\## 8. Serving 20 Concurrent Users on a 24 GB GPU



A practical approach to serving 20 concurrent users includes:



1\. Use a quantized model to reduce model-weight memory.

2\. Use a serving engine such as vLLM that supports continuous batching and PagedAttention.

3\. Share model weights across requests instead of loading separate model copies.

4\. Manage KV cache efficiently for each active sequence.

5\. Set reasonable maximum context lengths.

6\. Queue requests when GPU capacity is insufficient.

7\. Monitor GPU memory, throughput and response latency.



\### Illustrative memory calculation



Assume a 7B model with:



\- 32 transformer layers.

\- 8 KV heads.

\- Head dimension of 128.

\- FP16 KV cache.

\- Context length of 4,096 tokens.



KV cache memory per token:



2 × 32 × 8 × 128 × 2 bytes = 131,072 bytes.



KV cache memory per sequence:



131,072 × 4,096 = 512 MiB.



For 20 sequences:



512 MiB × 20 = 10 GiB.



A 7B model using 4-bit quantization has approximately 3.5 GB of raw model weights. Actual runtime memory is higher because of quantization metadata, activations, CUDA overhead and other allocations.



This illustrative calculation suggests that a 24 GB GPU may support this workload under suitable conditions. Actual capacity must be validated with the selected model architecture, context length, serving engine and workload.



\## 9. Limitations



\- The experiment used CPU inference rather than dedicated GPU inference.

\- Only one model architecture and model size were tested.

\- The benchmark used 10 prompts.

\- Some responses were truncated by the 128-token output limit.

\- Memory readings were process working-set snapshots, not continuous peak measurements.

\- No formal human evaluation of answer quality was conducted.

\- The 20-user serving analysis is theoretical and was not tested on a physical 24 GB GPU.



\## 10. Conclusion



The experiment demonstrated the effect of quantization on local inference performance.



Q4\_K\_M achieved an average generation speed of 18.53 tokens per second, compared with 12.17 tokens per second for Q8\_0. It also required less measured runner memory.



For the tested CPU environment, Q4\_K\_M offered the better speed and memory trade-off. Q8\_0 may be preferable when preserving more weight precision is important and additional memory and inference time are acceptable.



The experiment also demonstrated the importance of KV cache management, quantization, continuous batching and memory planning when designing systems for concurrent inference.



