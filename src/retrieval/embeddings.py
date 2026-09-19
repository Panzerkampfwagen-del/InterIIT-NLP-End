"""Deterministic local embeddings (Phase 6, ADR-05).

Environment-conditioned decision: no embeddings endpoint is available in this
environment (Groq provides none; heavyweight local stacks are disproportionate
and non-deterministic). Hashed bag-of-ngrams -> fixed-dim L2-normalized vector:

- deterministic (same text => same vector - tests stay offline-deterministic);
- no network, no model download;
- adequate for the short-ticket/note corpus this system retrieves over;
- drop-in upgrade path: replace ``embed`` + bump ``EMBEDDING_MODEL_VERSION``.
"""
from __future__ import annotations

import hashlib
import math
import re

EMBEDDING_MODEL_VERSION = "hashed-ngram-v1"
EMBEDDING_DIM = 256

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens + bigrams (simple, deterministic)."""
    words = _TOKEN_RE.findall(text.lower())[:512]
    bigrams = [f"{a}_{b}" for a, b in zip(words, words[1:])]
    return words + bigrams


def embed(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    """Hashed bag-of-ngrams, tf-weighted, L2-normalized."""
    vec = [0.0] * dim
    tokens = tokenize(text)
    for tok in tokens:
        h = hashlib.md5(tok.encode()).digest()
        idx = int.from_bytes(h[:4], "little") % dim
        # signed weight from the hash to reduce collision bias
        sign = 1.0 if h[4] % 2 == 0 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-dim vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
