
# Task 2: Retrieval-Augmented Generation System

## Source Document
NIST AI Risk Management Framework (AI RMF 1.0)
48 pages; 15,897 extracted words.

## Processing
PDF text extracted using PyMuPDF.

| Configuration | Chunk size | Overlap | Chunks |
|---|---:|---:|---:|
| Small | 200 words | 40 words | 114 |
| Large | 400 words | 80 words | 66 |

Chunks were created without crossing page boundaries.

## Retrieval
BM25 implemented from scratch in Python.
Parameters: k1 = 1.5, b = 0.75.
No LangChain or LlamaIndex used.

## Retrieval Evaluation

| Configuration | Page-level Recall@5 |
|---|---:|
| 200-word chunks | 80.00% |
| 400-word chunks | 86.67% |

Improvement: 6.67 percentage points.

Recall is calculated as the proportion of manually annotated
relevant pages retrieved among the top five chunks, counting
duplicate page numbers only once.

## Generation
Generator: Qwen2.5-0.5B-Instruct.
The top three retrieved passages were supplied to the model.

## Failure Analysis
- Q03: Original BM25 query did not retrieve the annotated
  risk-definition page.
- Q08: Correct MAP context passage was retrieved, but the
  generator confused MAP 1 with MAP 2.
- Q12: Generator returned an incomplete MANAGE explanation.
- BM25 is sensitive to lexical differences.
- A small generator can omit evidence or produce unsupported text.

## Improvements
Query expansion retrieved the correct Q03 definition passage.
Targeted retrieval was also tested for Q08 and Q12.
Corrected source-grounded answers are stored separately from
the original generated answers.

## BM25
BM25 ranks passages using term frequency, inverse document
frequency, and document-length normalization.

## Reranker
A reranker scores the relevance of each query-passage pair
after initial retrieval. It can improve ranking but adds
computation and latency. No reranker was implemented.

## Limitations
Results are specific to this document and 15-question evaluation.
They do not establish general retrieval performance.
The retrieval metric and answer-generation quality are reported
separately.
