"""The worked examples from the Week 7 slides, computed with this module's own code."""
from __future__ import annotations

import numpy as np

from .models import SVD, cosine


def slide_7_cosine() -> dict:
    """Slides 6-8: Q = (1, 1), D1 = (5, 3) in a fruits/vegetables space."""
    q, d = np.array([1.0, 1.0]), np.array([5.0, 3.0])
    return {
        "dot": float(q @ d),                        # 8
        "q_norm": float(np.linalg.norm(q)),         # sqrt 2
        "d_norm": float(np.linalg.norm(d)),         # sqrt 34
        "cosine": cosine(q, d),                     # 0.97
    }


def slide_11_svd(k: int = 2) -> dict:
    """Slides 11-12: the 3 x 3 term-document matrix, decomposed and cut to k concepts."""
    A = np.array([[1.0, 1.0, 0.0],
                  [1.0, 0.0, 1.0],
                  [0.0, 1.0, 1.0]])
    svd = SVD(A)
    docs_k = (np.diag(svd.s[:k]) @ svd.Vt[:k]).T      # each document in k-concept space
    return {
        "A": A,
        "U": svd.U, "S": svd.s, "Vt": svd.Vt,
        "A_k": svd.reconstruct(k),
        "rebuilt_exactly": bool(np.allclose(svd.reconstruct(3), A)),
        "doc_similarity_terms": [[cosine(A[:, i], A[:, j]) for j in range(3)] for i in range(3)],
        "doc_similarity_concepts": [[cosine(docs_k[i], docs_k[j]) for j in range(3)] for i in range(3)],
    }
