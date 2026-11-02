"""Requirements 1 and 2: the document-term matrix and the query vector.

    A[i, j] = weight of term i in document j          (rows = terms, columns = documents)
    weight  = (1 + ln tf) x ln(N / df)                 TF-IDF, as slide 15 suggests
    every column is scaled to length 1                 so long documents do not dominate the SVD
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .text import raw_words, stem

FIELDS = ("title", "tags", "text")


def document_text(doc: dict) -> str:
    tags = doc.get("tags", [])
    return " ".join([doc.get("title", ""), " ".join(tags) if isinstance(tags, list) else str(tags), doc.get("text", "")])


def tf_weight(tf: float) -> float:
    return 1.0 + math.log(tf) if tf > 0 else 0.0


@dataclass
class QueryVector:
    text: str
    vector: np.ndarray                                   # length m, in term space
    weights: dict[str, float] = field(default_factory=dict)   # known term -> TF-IDF weight
    unknown: list[str] = field(default_factory=list)          # words the collection never uses

    @property
    def is_empty(self) -> bool:
        return not self.weights


class DocumentTermMatrix:
    def __init__(self, documents: list[dict]):
        self.documents = documents
        self.ids = [d["id"] for d in documents]
        surface: dict[str, Counter] = {}
        doc_counts: list[Counter] = []
        for doc in documents:
            counts = Counter()
            for word in raw_words(document_text(doc)):
                term = stem(word)
                counts[term] += 1
                surface.setdefault(term, Counter())[word] += 1
            doc_counts.append(counts)

        self.terms: list[str] = sorted(surface)
        self.index = {t: i for i, t in enumerate(self.terms)}
        # show "compressor", not the stem "compressor"; "cycle", not "cycl"
        self.display = {t: c.most_common(1)[0][0] for t, c in surface.items()}

        m, n = len(self.terms), len(documents)
        self.counts = np.zeros((m, n))
        for j, counts in enumerate(doc_counts):
            for term, tf in counts.items():
                self.counts[self.index[term], j] = tf
        self.df = (self.counts > 0).sum(axis=1)
        self.idf = np.log(n / self.df)

        tf = np.where(self.counts > 0, 1.0 + np.log(np.maximum(self.counts, 1)), 0.0)
        raw = tf * self.idf[:, None]
        norms = np.linalg.norm(raw, axis=0)
        norms[norms == 0] = 1.0
        self.A = raw / norms                              # the matrix every model works on

    @property
    def shape(self) -> tuple[int, int]:
        return self.A.shape

    def query_vector(self, text: str) -> QueryVector:
        """Requirement 2: put the query in the same term space as the documents."""
        known: Counter = Counter()
        unknown: list[str] = []
        for word in raw_words(text):
            term = stem(word)
            if term in self.index:
                known[term] += 1
            elif word not in unknown:
                unknown.append(word)
        vector = np.zeros(len(self.terms))
        weights = {}
        for term, tf in known.items():
            i = self.index[term]
            vector[i] = tf_weight(tf) * self.idf[i]
            weights[term] = float(vector[i])
        return QueryVector(text, vector, weights, unknown)

    def excerpt(self, terms: list[str], doc_indexes: list[int]) -> list[list[float]]:
        """Rows of A for some terms and documents, for showing the matrix in the interface."""
        rows = [self.index[t] for t in terms]
        return self.A[np.ix_(rows, doc_indexes)].round(3).tolist()
