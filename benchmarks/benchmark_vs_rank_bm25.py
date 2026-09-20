#!/usr/bin/env python3
"""
Benchmark: CompactBM25 vs rank_bm25.BM25Okapi
Measures memory, indexing time, query latency, and numerical parity.
"""

import sys
import time
import tracemalloc
import numpy as np
from rank_bm25 import BM25Okapi
from compact_bm25 import CompactBM25

def generate_mock_corpus(num_docs=5000, doc_len=50, vocab_size=2000):
    words = [f"word_{i}" for i in range(vocab_size)]
    np.random.seed(42)
    corpus = []
    for _ in range(num_docs):
        doc = list(np.random.choice(words, size=doc_len))
        corpus.append(doc)
    return corpus

def main():
    num_docs = 5000
    print(f"Generating synthetic corpus: {num_docs} documents...")
    corpus = generate_mock_corpus(num_docs=num_docs)
    query = ["word_42", "word_100", "word_999", "nonexistent_term"]

    print("\n--- 1. Testing rank_bm25.BM25Okapi ---")
    tracemalloc.start()
    t0 = time.time()
    orig_bm25 = BM25Okapi(corpus)
    t_index_orig = time.time() - t0
    current, peak_orig = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    t0 = time.time()
    for _ in range(100):
        orig_scores = orig_bm25.get_scores(query)
    t_query_orig = (time.time() - t0) / 100 * 1000

    print(f"Indexing time: {t_index_orig:.3f}s")
    print(f"Peak memory:   {peak_orig / 1024 / 1024:.2f} MB")
    print(f"Query latency: {t_query_orig:.2f} ms")

    print("\n--- 2. Testing CompactBM25 ---")
    tracemalloc.start()
    t0 = time.time()
    compact_bm25 = CompactBM25.build(corpus, k1=orig_bm25.k1, b=orig_bm25.b, epsilon=orig_bm25.epsilon)
    t_index_compact = time.time() - t0
    current, peak_compact = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    t0 = time.time()
    for _ in range(100):
        compact_scores = compact_bm25.get_scores(query)
    t_query_compact = (time.time() - t0) / 100 * 1000

    print(f"Indexing time: {t_index_compact:.3f}s")
    print(f"Peak memory:   {peak_compact / 1024 / 1024:.2f} MB (Saved: {(1 - peak_compact/peak_orig)*100:.1f}%)")
    print(f"Query latency: {t_query_compact:.2f} ms")

    print("\n--- 3. Numerical Alignment Check ---")
    diff = np.abs(orig_scores - compact_scores)
    max_diff = np.max(diff)
    print(f"Max absolute score difference: {max_diff:.2e}")
    assert np.allclose(orig_scores, compact_scores, atol=1e-5), "Scores do not match!"
    print("✅ Numerical parity verified within the configured tolerance.")

if __name__ == "__main__":
    main()
