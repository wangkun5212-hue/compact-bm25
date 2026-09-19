#!/usr/bin/env python3
import pytest
import numpy as np
from rank_bm25 import BM25Okapi
from compact_bm25 import CompactBM25

def test_basic_scoring_and_top_n():
    corpus = [
        ["apple", "banana", "apple"],
        ["banana", "orange"],
        ["peach", "orange", "banana", "banana"],
        ["grape", "lemon"],
        ["strawberry", "cherry"]
    ]
    docs = ["Doc 1", "Doc 2", "Doc 3", "Doc 4", "Doc 5"]
    
    bm25 = CompactBM25.build(corpus)
    scores = bm25.get_scores(["apple"])
    assert len(scores) == 5
    assert scores[0] > 0.0        # Doc 1 has 2 apples
    assert scores[1] == 0.0       # Doc 2 has no apple

    top_docs = bm25.get_top_n(["apple"], docs, n=1)
    assert top_docs == ["Doc 1"]

def test_parity_with_rank_bm25():
    corpus = [
        ["the", "quick", "brown", "fox"],
        ["jumped", "over", "the", "lazy", "dog"],
        ["the", "fox", "and", "the", "hound"],
        ["quick", "brown", "dogs", "run", "fast"]
    ]
    query = ["quick", "fox", "unknown_token"]

    orig = BM25Okapi(corpus)
    compact = CompactBM25.build(corpus, k1=orig.k1, b=orig.b, epsilon=orig.epsilon)

    s_orig = orig.get_scores(query)
    s_compact = compact.get_scores(query)

    assert np.allclose(s_orig, s_compact, atol=1e-6)
