"""Requirements 4 and 5: rank the documents and return the top results, with explanations."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .matrix import DocumentTermMatrix
from .models import SVD, GeneralizedVSM, HybridModel, LSIModel, VectorSpaceModel
from .text import word_spans

DATA = Path(__file__).resolve().parent.parent / "data"
MODELS = ("lsi", "hybrid", "gvsm", "vsm")
ENERGY_RULE = 0.5          # k = smallest number of concepts that keeps half of sum(s^2)
SHOWN = 1e-3               # contributions smaller than this are not listed
HIGHLIGHT = 10             # only the strongest contributing words are marked in the text


class QueryError(ValueError):
    """A query that cannot be turned into a usable vector. The message is for the user."""


def load_documents(path: Path | str = DATA / "knowledge_base.json") -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


class SemanticSearchEngine:
    def __init__(self, documents: list[dict], k: int | None = None, alpha: float = 0.5):
        self.documents = documents
        self.dtm = DocumentTermMatrix(documents)
        self.svd = SVD(self.dtm.A)
        self.max_k = len(self.svd.s)
        self.default_k = k or int(np.searchsorted(self.svd.energy, ENERGY_RULE) + 1)
        self.alpha = alpha
        self.vsm = VectorSpaceModel(self.dtm)
        self.gvsm = GeneralizedVSM(self.dtm)
        self._lsi: dict[int, LSIModel] = {}

    @classmethod
    def from_data_dir(cls, data_dir: Path | str = DATA, **kwargs) -> "SemanticSearchEngine":
        return cls(load_documents(Path(data_dir) / "knowledge_base.json"), **kwargs)

    # ------------------------------------------------------------------ models
    def lsi(self, k: int | None = None) -> LSIModel:
        k = min(max(int(k or self.default_k), 1), self.max_k)
        if k not in self._lsi:
            self._lsi[k] = LSIModel(self.dtm, self.svd, k)
        return self._lsi[k]

    def scores(self, query: str, model: str = "lsi", k: int | None = None) -> np.ndarray:
        """One similarity per document, in collection order. Used by the evaluation."""
        q = self.dtm.query_vector(query).vector
        if model == "vsm":
            return self.vsm.score(q).scores
        if model == "gvsm":
            return self.gvsm.score(q).scores
        if model == "lsi":
            return self.lsi(k).score(q).scores
        if model == "hybrid":
            return HybridModel(self.lsi(k), self.vsm, self.alpha).score(q)[2]
        raise QueryError(f"Unknown model '{model}'. Choose one of: {', '.join(MODELS)}.")

    def rank(self, query: str, model: str = "lsi", k: int | None = None) -> list[tuple[str, float]]:
        scores = self.scores(query, model, k)
        order = sorted(range(len(scores)), key=lambda j: (-scores[j], self.dtm.ids[j]))
        return [(self.dtm.ids[j], float(scores[j])) for j in order]

    # ------------------------------------------------------------------ search
    def search(self, query: str, model: str = "lsi", k: int | None = None,
               top: int = 10, min_score: float = 0.10) -> dict:
        if model not in MODELS:
            raise QueryError(f"Unknown model '{model}'. Choose one of: {', '.join(MODELS)}.")
        if not query.strip():
            raise QueryError("Type a few words describing what you are looking for.")
        qv = self.dtm.query_vector(query)
        if qv.is_empty:
            raise QueryError(
                "None of these words appear in the collection, so the query vector is all zeros. "
                "Try a different word for the same idea."
                if qv.unknown else "The query only has common words such as 'the' or 'on'. Add a content word.")
        if not np.any(qv.vector):
            raise QueryError("These words appear in every document, so they cannot tell documents apart.")

        lsi = self.lsi(k)
        vsm_scored = self.vsm.score(qv.vector)
        concept_model = None
        if model == "vsm":
            main, scores = vsm_scored, vsm_scored.scores
        elif model == "gvsm":
            main = self.gvsm.score(qv.vector)
            scores = main.scores
        elif model == "lsi":
            main = lsi.score(qv.vector)
            scores, concept_model = main.scores, lsi
        else:
            main, _, scores = HybridModel(lsi, self.vsm, self.alpha).score(qv.vector)
            concept_model = lsi

        order = sorted(range(len(scores)), key=lambda j: (-scores[j], self.dtm.ids[j]))
        kept = [j for j in order if scores[j] >= min_score]
        positive = int(np.sum(scores > 0))
        shown = kept[:top]

        results = []
        for rank, j in enumerate(shown, 1):
            contributions = self._term_contributions(model, main, vsm_scored, j, lsi)
            results.append({
                "rank": rank,
                "score": round(float(scores[j]), 4),
                "vsm_score": round(float(vsm_scored.scores[j]), 4),
                "found_by_meaning": bool(vsm_scored.scores[j] <= 0),
                "document": self.documents[j],
                "explanation": self._explain(model, main, vsm_scored, j, contributions, qv, concept_model, scores[j]),
                "highlights": self._highlights(self.documents[j], contributions, qv),
            })

        return {
            "query": query,
            "model": model,
            "model_label": {"lsi": lsi.label, "hybrid": HybridModel.label,
                            "gvsm": self.gvsm.label, "vsm": self.vsm.label}[model],
            "k": lsi.k,
            "energy": round(float(self.svd.energy[lsi.k - 1]), 3),
            "matrix_shape": list(self.dtm.shape),
            "query_vector": [
                {"term": t, "word": self.dtm.display[t], "weight": round(w, 3),
                 "idf": round(float(self.dtm.idf[self.dtm.index[t]]), 3)}
                for t, w in sorted(qv.weights.items(), key=lambda x: -x[1])
            ],
            "unknown": qv.unknown,
            "expanded_query": self._expanded(main, qv, model),
            "query_concepts": self._query_concepts(lsi, qv) if concept_model else [],
            "total_considered": len(self.documents),
            "total_positive": positive,
            "total_shown": len(results),
            "total_above_min": len(kept),
            "vsm_zero_hits": int(np.sum(vsm_scored.scores > 0)),
            "results": results,
            "matrix": self._matrix(qv, main, shown),
        }

    # ------------------------------------------------------------ explanation
    def _term_contributions(self, model, main, vsm_scored, j, lsi) -> np.ndarray:
        if model == "vsm":
            return self.vsm.term_contributions(main, j)
        if model == "gvsm":
            return self.gvsm.term_contributions(main, j)
        if model == "lsi":
            return lsi.term_contributions(main, j)
        a = self.alpha
        return a * lsi.term_contributions(main, j) + (1 - a) * self.vsm.term_contributions(vsm_scored, j)

    def _explain(self, model, main, vsm_scored, j, contributions, qv, concept_model, score) -> dict:
        typed = set(qv.weights)
        present = np.nonzero(self.dtm.A[:, j])[0]
        rows = sorted(present, key=lambda i: -contributions[i])
        words = [
            {
                "term": self.dtm.terms[i],
                "word": self.dtm.display[self.dtm.terms[i]],
                "typed": self.dtm.terms[i] in typed,
                "doc_weight": round(float(self.dtm.A[i, j]), 3),
                "query_weight": round(float(main.expanded[i]), 3),
                "contribution": round(float(contributions[i]), 4),
            }
            for i in rows[:8] if contributions[i] > SHOWN
        ]
        rest = float(contributions[present].sum()) - sum(w["contribution"] for w in words)

        if model == "hybrid":
            lsi_part = float(main.scores[j])
            vsm_part = float(vsm_scored.scores[j])
            formula = (f"{self.alpha:g} × LSI {lsi_part:.3f} + {1 - self.alpha:g} × VSM {vsm_part:.3f}"
                       f" = {score:.3f}")
        else:
            dot = float(main.expanded @ self.dtm.A[:, j])
            q_norm = float(np.linalg.norm(main.query_hat))
            d_norm = float(1.0 / (main.scale[j] * q_norm)) if main.scale[j] else 0.0
            space = {"vsm": "term space", "gvsm": "correlated term space", "lsi": f"{self._k(main)}-concept space"}[model]
            formula = f"q·d / (|q|·|d|) = {dot:.3f} / ({q_norm:.3f} × {d_norm:.3f}) = {score:.3f}   in {space}"

        out = {"formula": formula, "words": words, "other_words": round(rest, 4)}
        if concept_model is not None:
            share = self.alpha if model == "hybrid" else 1.0
            parts = concept_model.concept_contributions(main, j) * share
            top = np.argsort(-parts)[:4]
            out["concepts"] = [
                {"concept": int(c) + 1, "label": self._concept_label(concept_model, c, float(main.query_hat[c])),
                 "contribution": round(float(parts[c]), 4)}
                for c in top if parts[c] > SHOWN
            ]
        return out

    def _k(self, scored) -> int:
        return len(scored.query_hat)

    def _expanded(self, main, qv, model) -> list[dict]:
        """The query as the model sees it: the top weights of P P^T q in term space."""
        typed = set(qv.weights)
        weights = main.expanded
        top = np.argsort(-weights)[:12]
        peak = float(weights[top[0]]) or 1.0
        return [
            {"term": self.dtm.terms[i], "word": self.dtm.display[self.dtm.terms[i]],
             "typed": self.dtm.terms[i] in typed, "weight": round(float(weights[i]), 4),
             "relative": round(float(weights[i]) / peak, 3)}
            for i in top if weights[i] > 0
        ]

    def _query_concepts(self, lsi: LSIModel, qv) -> list[dict]:
        q_hat = lsi.project(qv.vector)
        norm = float(np.linalg.norm(q_hat)) or 1.0
        top = np.argsort(-np.abs(q_hat))[:5]
        return [
            {"concept": int(c) + 1, "weight": round(float(q_hat[c]) / norm, 3),
             "label": self._concept_label(lsi, c, float(q_hat[c]))}
            for c in top
        ]

    def _concept_label(self, lsi: LSIModel, c: int, direction: float = 1.0, size: int = 4) -> list[str]:
        column = lsi.svd.U[:, c] * (1 if direction >= 0 else -1)
        return [self.dtm.display[self.dtm.terms[i]] for i in np.argsort(-column)[:size]]

    def _highlights(self, doc: dict, contributions: np.ndarray, qv) -> dict:
        typed = set(qv.weights)
        strongest = [i for i in np.argsort(-contributions)[:HIGHLIGHT] if contributions[i] > SHOWN]
        marked = {self.dtm.terms[i] for i in strongest} | {t for t in typed if contributions[self.dtm.index[t]] > SHOWN}

        def segments(text: str):
            out, last = [], 0
            for start, end, term in word_spans(text):
                i = self.dtm.index.get(term)
                if i is None or term not in marked:
                    continue
                if start > last:
                    out.append({"text": text[last:start]})
                out.append({"text": text[start:end], "kind": "exact" if term in typed else "semantic",
                            "value": round(float(contributions[i]), 4)})
                last = end
            if last < len(text):
                out.append({"text": text[last:]})
            return out

        return {"title": segments(doc["title"]), "text": segments(doc["text"])}

    def _matrix(self, qv, main, shown: list[int]) -> dict:
        """A small window of the document-term matrix around this query (requirement 1)."""
        typed = sorted(qv.weights, key=lambda t: -qv.weights[t])
        docs = shown[:5]
        bridges = []
        if docs:
            present = np.nonzero(self.dtm.A[:, docs].sum(axis=1))[0]
            for i in sorted(present, key=lambda i: -main.expanded[i]):
                term = self.dtm.terms[i]
                if term not in qv.weights and main.expanded[i] > 0:
                    bridges.append(term)
                if len(bridges) == 4:
                    break
        rows = typed + bridges
        return {
            "terms": [{"term": t, "word": self.dtm.display[t], "typed": t in qv.weights} for t in rows],
            "docs": [self.dtm.ids[j] for j in docs],
            "values": self.dtm.excerpt(rows, docs) if docs and rows else [],
            "query": [round(qv.weights.get(t, 0.0), 3) for t in rows],
            "expanded": [round(float(main.expanded[self.dtm.index[t]]), 3) for t in rows],
        }

    # ---------------------------------------------------------------- concepts
    def concepts(self, k: int | None = None, size: int = 6) -> list[dict]:
        lsi = self.lsi(k)
        total = float((self.svd.s ** 2).sum())
        out = []
        for c in range(lsi.k):
            column = self.svd.U[:, c]
            docs = np.argsort(-np.abs(self.svd.Vt[c]))[:3]
            out.append({
                "concept": c + 1,
                "singular_value": round(float(self.svd.s[c]), 3),
                "energy": round(float(self.svd.s[c] ** 2) / total, 4),
                "positive": [self.dtm.display[self.dtm.terms[i]] for i in np.argsort(-column)[:size]],
                "negative": [self.dtm.display[self.dtm.terms[i]] for i in np.argsort(column)[:3]],
                "documents": [self.dtm.ids[j] for j in docs],
            })
        return out

    def related_terms(self, word: str, k: int | None = None, top: int = 8) -> list[dict]:
        """Terms closest to a word in concept space (rows of U_k S_k)."""
        qv = self.dtm.query_vector(word)
        if qv.is_empty:
            return []
        lsi = self.lsi(k)
        vectors = self.svd.U[:, :lsi.k] * self.svd.s[:lsi.k]
        target = vectors[self.dtm.index[next(iter(qv.weights))]]
        norms = np.linalg.norm(vectors, axis=1)
        sims = vectors @ target / (np.where(norms == 0, 1, norms) * (np.linalg.norm(target) or 1))
        order = [i for i in np.argsort(-sims) if self.dtm.terms[i] not in qv.weights][:top]
        return [{"word": self.dtm.display[self.dtm.terms[i]], "similarity": round(float(sims[i]), 3)} for i in order]
