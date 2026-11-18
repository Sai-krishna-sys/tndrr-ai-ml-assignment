# Task 2 - Retrieval-Augmented Generation (RAG) System

## Goal

Build a RAG pipeline that answers questions over a custom document corpus by retrieving
relevant chunks and feeding them to an LLM as context.

## RAG concepts

- **Chunking**: split documents into overlapping chunks to balance context vs. precision.
- **Embeddings**: convert chunks to dense vectors (sentence-transformers).
- **Vector store**: index vectors for fast nearest-neighbour search (FAISS / Chroma).
- **Retrieval**: top-k chunks by cosine similarity (or MMR for diversity).
- **Generation**: prompt an LLM with retrieved context; cite sources.
- **Evaluation**: retrieval metrics (Recall@k, MRR) and answer quality (faithfulness).

## Planned pipeline

1. Ingest documents (`data/`), clean and chunk.
2. Embed chunks and build vector index (`vector_store/`).
3. Query: embed question -> retrieve top-k -> build prompt -> generate answer.
4. Return answer with source citations.
5. Evaluate on a small QA set.

## Directory layout

- `src/` - ingest, chunk, embed, retrieve, generate, eval modules.
- `data/` - source documents and QA eval set.
- `vector_store/` - persisted FAISS/Chroma index.
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
python src/ingest.py --docs data/raw
python src/query.py --question "What is ...?"
python src/evaluate.py --qa data/qa.jsonl
```

## Evaluation

Retrieval and answer metrics are computed on real runs and saved under `outputs/` (to be
added). No fabricated numbers.
