# ⚡ CompactBM25

> **A memory-efficient, bit-identical, drop-in replacement for `rank_bm25`.**
> Slash RAG index memory by **60% ~ 80%** while accelerating query speed by **10x~20x** with NumPy CSR/CSC sparse matrices.

[![PyPI](https://img.shields.io/pypi/v/compact-bm25?color=blue)](https://pypi.org/project/compact-bm25/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

---

## 🎯 The Problem

In modern LLM RAG pipelines, **Hybrid Search (Vector + BM25)** is the gold standard for retrieval accuracy. Almost everyone relies on `rank_bm25.BM25Okapi`.

However, as your knowledge base grows beyond 100,000 chunks:
- **Severe Memory Bloat**: `rank_bm25` stores term frequencies as Python `list[dict]`, easily consuming **1.5 GB ~ 3 GB of RAM** in production.
- **Painful Cold Starts**: Pickling/unpickling millions of nested Python dictionaries can take **over 70 seconds** upon container startup, often causing memory spikes and container OOM kills.

---

## 💡 The Solution: CompactBM25

**CompactBM25** completely reimagines BM25 storage using contiguous **NumPy flat arrays** and **CSR (Compressed Sparse Row) / CSC (Compressed Sparse Column)** offsets:

- 📉 **60% ~ 80% Memory Reduction**: Compact CSR representation slashes memory usage from gigabytes down to a couple hundred megabytes.
- 🎯 **Exact Bit-Identical Scores**: Matches `rank_bm25.BM25Okapi` Robertson IDF formula and epsilon flooring down to `1e-6` floating precision.
- ⚡ **Blazing Fast Cold Start & Query**: Instant memory mapping and zero dictionary deserialization overhead (0.1s load time). Vectorized scoring runs up to **20x faster** than pure Python loops.
- 🔄 **100% Drop-in Compatibility**: `bm25.get_scores(query)` and `bm25.get_top_n(query, docs)` work identically.

---

## 📊 Benchmark

Tested on a synthetic corpus of **5,000 documents** (50 words/doc, vocabulary size 2,000):

| Metric | `rank_bm25.BM25Okapi` | `CompactBM25` | Improvement |
| :--- | :--- | :--- | :--- |
| **Peak Memory** | 11.08 MB | **3.30 MB** | **📉 70.2% Saved** |
| **Query Latency (100 runs)** | 1.45 ms | **0.05 ms** | **⚡ 29x Faster** |
| **Score Difference** | Baseline | **0.00e+00** | **✅ Bit-Identical** |

*(On a 145,000-chunk production RAG dataset, steady-state RAM dropped from **1,547 MB down to 536 MB**)*.

---

## 🚀 Quick Start

### Installation

```bash
pip install compact-bm25
```

### Basic Usage (Drop-in Replacement)

```python
from compact_bm25 import CompactBM25

corpus = [
    ["deep", "learning", "model"],
    ["natural", "language", "processing"],
    ["deep", "neural", "network"],
    ["retrieval", "augmented", "generation"]
]

# 1. Build index from tokenized corpus
bm25 = CompactBM25.build(corpus)

# 2. Score documents against query
query = ["deep", "learning"]
scores = bm25.get_scores(query)
print("BM25 Scores:", scores)

# 3. Get Top-N documents directly
docs = ["Doc A", "Doc B", "Doc C", "Doc D"]
top_docs = bm25.get_top_n(query, docs, n=2)
print("Top 2 Matches:", top_docs)
```

### Persistence (Lightning Fast Serialization)

Because all internal buffers are flat contiguous NumPy arrays, serialization is trivial and ultra-compact:

```python
import pickle

# Save to disk (~10x smaller pickle than rank_bm25)
with open("bm25_index.pkl", "wb") as f:
    pickle.dump(bm25, f)

# Instant reload in production (<0.1s cold start)
with open("bm25_index.pkl", "rb") as f:
    bm25 = pickle.load(f)
```

---

## 🔬 How It Works

Instead of storing Python object trees (`dict[term -> count]`), **CompactBM25** packs document and term postings into parallel flat arrays:
1. `offsets [int64]`: CSR row pointers pointing to document boundaries.
2. `term_ids [int32]`: Flat array of integer term IDs.
3. `freqs [int8/int16/int32]`: Adaptively downcasted integer frequencies.
4. `term_offsets & inv_doc [CSC]`: Inverted posting lists allowing direct NumPy vectorized slice accumulation during queries.

---

## 🧪 Run Tests & Benchmarks

```bash
git clone https://github.com/wangkun5212-hue/compact-bm25.git
cd compact-bm25
pip install -r requirements.txt

# Run parity tests
pytest tests/ -v

# Run performance & memory benchmark
python benchmarks/benchmark_vs_rank_bm25.py
```

---

## 📄 License

MIT License. Free for commercial and research use.
