"""Requirements 3 and 6: cosine similarity in three algebraic models.

Every model here is the same recipe with a different projection matrix P:

    query  q^ = P^T q          document  d^_j = P^T a_j
    score  cos(q^, d^_j) = q^ . d^_j / (|q^| |d^_j|)

    VSM   P = I        the terms themselves are the axes (exact words only)
    GVSM  P = T        T = A with each term row scaled to length 1, so P P^T = C,
                       the term-term correlation matrix (terms are no longer orthogonal)
    LSI   P = U_k      the k strongest concepts found by A = U S V^T (SVD)

Because q^ . d^_j = (P P^T q) . a_j, every score can be split exactly into one
contribution per document term. P P^T q is "the query as the model sees it": for
VSM it is the query itself; for GVSM and LSI it also gives weight to related terms
the user never typed. That split is how each result explains itself (requirement 6).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .matrix import DocumentTermMatrix
from .text import stem


def cosine(a, b) -> float:
    """cos(theta) = a.b / (|a| |b|); 0 when either vector is all zeros."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


def _safe(norms: np.ndarray) -> np.ndarray:
    out = norms.copy()
    out[out == 0] = 1.0
    return out


@dataclass
class Scored:
    scores: np.ndarray        # one cosine per document
    query_hat: np.ndarray     # q^ in the model's space
    expanded: np.ndarray      # P P^T q, back in term space (length m)
    scale: np.ndarray         # 1 / (|q^| |d^_j|) per document


class ProjectionModel:
    name = "base"
    label = "base"

    def __init__(self, dtm: DocumentTermMatrix, projection: np.ndarray | None):
        self.dtm = dtm
        self.P = projection                         # None means identity (plain VSM)
        A = dtm.A
        self.doc_hat = (A if self.P is None else self.P.T @ A).T      # n x r
        self.doc_norms = np.linalg.norm(self.doc_hat, axis=1)

    def project(self, q: np.ndarray) -> np.ndarray:
        return q if self.P is None else self.P.T @ q

    def score(self, q: np.ndarray) -> Scored:
        q_hat = self.project(q)
        q_norm = np.linalg.norm(q_hat)
        if q_norm == 0:
            zeros = np.zeros(len(self.doc_norms))
            return Scored(zeros, q_hat, np.zeros_like(q), zeros)
        scale = 1.0 / (q_norm * _safe(self.doc_norms))
        scores = (self.doc_hat @ q_hat) * scale
        expanded = q if self.P is None else self.P @ q_hat
        return Scored(scores, q_hat, expanded, scale)

    def term_contributions(self, scored: Scored, j: int) -> np.ndarray:
        """Contribution of every term to document j's score. Sums to the score."""
        return scored.expanded * self.dtm.A[:, j] * scored.scale[j]


class VectorSpaceModel(ProjectionModel):
    name, label = "vsm", "VSM · TF-IDF cosine (exact words)"

    def __init__(self, dtm):
        super().__init__(dtm, None)


class GeneralizedVSM(ProjectionModel):
    """Term vectors are not orthogonal: t_i . t_l = correlation of terms i and l.

    T has one row per term (that term's weights across the documents, length 1),
    so C = T T^T holds the cosine between every pair of terms. Two terms that
    occur in the same documents get a high correlation, and a query word then
    also counts for its correlated words.
    """
    name, label = "gvsm", "GVSM · term-correlation cosine"

    def __init__(self, dtm):
        rows = np.linalg.norm(dtm.A, axis=1, keepdims=True)
        T = dtm.A / _safe(rows)
        super().__init__(dtm, T)

    def correlation(self, word_a: str, word_b: str) -> float:
        """C[i, l] for two words (they are stemmed first, like every query word)."""
        i, l = self.dtm.index[stem(word_a)], self.dtm.index[stem(word_b)]
        return float(self.P[i] @ self.P[l])


class LSIModel(ProjectionModel):
    """Latent Semantic Indexing: A ~ U_k S_k V_k^T, compare in k concept dimensions."""
    name, label = "lsi", "LSI · cosine in concept space"

    def __init__(self, dtm, svd: "SVD", k: int):
        self.svd, self.k = svd, k
        super().__init__(dtm, svd.U[:, :k])

    def concept_contributions(self, scored: Scored, j: int) -> np.ndarray:
        """Contribution of every concept to document j's score. Sums to the score."""
        return scored.query_hat * self.doc_hat[j] * scored.scale[j]


class HybridModel:
    """alpha x LSI + (1 - alpha) x VSM: concepts for recall, exact words for precision."""
    name, label = "hybrid", "Hybrid · LSI + VSM"

    def __init__(self, lsi: LSIModel, vsm: VectorSpaceModel, alpha: float = 0.5):
        self.lsi, self.vsm, self.alpha = lsi, vsm, alpha
        self.dtm = lsi.dtm

    def score(self, q):
        a, b = self.lsi.score(q), self.vsm.score(q)
        return a, b, self.alpha * a.scores + (1 - self.alpha) * b.scores


class SVD:
    """A = U S V^T, computed once. Any k is then just a slice."""

    def __init__(self, A: np.ndarray):
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        # SVD signs are arbitrary; make each concept's largest loading positive so
        # the concepts read the same way on every run.
        for c in range(U.shape[1]):
            if U[np.argmax(np.abs(U[:, c])), c] < 0:
                U[:, c] *= -1
                Vt[c] *= -1
        self.U, self.s, self.Vt = U, s, Vt

    @property
    def energy(self) -> np.ndarray:
        """Cumulative share of sum(s^2) kept by the first 1..r concepts."""
        e = self.s ** 2
        return np.cumsum(e) / e.sum()

    def reconstruct(self, k: int) -> np.ndarray:
        return self.U[:, :k] @ np.diag(self.s[:k]) @ self.Vt[:k]
