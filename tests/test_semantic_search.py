import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from semantic_search import QueryError, SemanticSearchEngine, cosine  # noqa: E402
from semantic_search.lecture import slide_7_cosine, slide_11_svd  # noqa: E402
from semantic_search.text import stem, terms  # noqa: E402

ENGINE = SemanticSearchEngine.from_data_dir(ROOT / "data")
QUERIES = json.loads((ROOT / "data" / "queries.json").read_text(encoding="utf-8"))["queries"]


class LectureExamples(unittest.TestCase):
    def test_slide_7_cosine_is_0_97(self):
        c = slide_7_cosine()
        self.assertEqual(c["dot"], 8)
        self.assertAlmostEqual(c["q_norm"], math.sqrt(2))
        self.assertAlmostEqual(c["d_norm"], math.sqrt(34))
        self.assertAlmostEqual(c["cosine"], 8 / math.sqrt(68))
        self.assertEqual(round(c["cosine"], 2), 0.97)

    def test_slide_11_svd_rebuilds_the_matrix(self):
        s = slide_11_svd()
        self.assertTrue(s["rebuilt_exactly"])
        np.testing.assert_allclose(s["S"], [2, 1, 1], atol=1e-9)

    def test_cosine_edge_cases(self):
        self.assertEqual(cosine([1, 0], [0, 1]), 0.0)          # orthogonal: no relation
        self.assertAlmostEqual(cosine([2, 2], [1, 1]), 1.0)    # same direction
        self.assertAlmostEqual(cosine([1, 0], [-1, 0]), -1.0)  # opposite direction
        self.assertEqual(cosine([0, 0], [1, 1]), 0.0)          # empty vector


class DocumentTermMatrix(unittest.TestCase):
    def test_rows_are_terms_and_columns_are_documents(self):
        m, n = ENGINE.dtm.shape
        self.assertEqual(n, 40)
        self.assertEqual(m, len(ENGINE.dtm.terms))

    def test_columns_have_length_one(self):
        np.testing.assert_allclose(np.linalg.norm(ENGINE.dtm.A, axis=0), 1.0)

    def test_tf_idf_weight_of_a_rare_term(self):
        i = ENGINE.dtm.index["gasket"]
        self.assertEqual(ENGINE.dtm.df[i], 1)
        self.assertAlmostEqual(ENGINE.dtm.idf[i], math.log(40))

    def test_text_processing(self):
        self.assertEqual(stem("leakage"), stem("leaks"))
        self.assertEqual(terms("The compressor is running"), ["compressor", "run"])


class QueryVector(unittest.TestCase):
    def test_known_and_unknown_words(self):
        qv = ENGINE.dtm.query_vector("compressor wastage xyzzy")
        self.assertIn("compressor", qv.weights)
        self.assertIn("xyzzy", qv.unknown)
        self.assertEqual(len(qv.vector), ENGINE.dtm.shape[0])

    def test_query_weight_is_tf_idf(self):
        qv = ENGINE.dtm.query_vector("gasket gasket")
        self.assertAlmostEqual(qv.weights["gasket"], (1 + math.log(2)) * math.log(40))


class Models(unittest.TestCase):
    def test_k_follows_the_energy_rule(self):
        k = ENGINE.default_k
        self.assertGreaterEqual(ENGINE.svd.energy[k - 1], 0.5)
        self.assertLess(ENGINE.svd.energy[k - 2], 0.5)

    def test_cosines_are_between_minus_one_and_one(self):
        for model in ("vsm", "gvsm", "lsi", "hybrid"):
            s = ENGINE.scores("compressor left running overnight", model)
            self.assertTrue(np.all(s <= 1 + 1e-9) and np.all(s >= -1 - 1e-9), model)

    def test_vsm_only_scores_documents_sharing_a_word(self):
        s = ENGINE.scores("gasket", "vsm")
        self.assertEqual(int(np.sum(s > 0)), 1)

    def test_full_rank_lsi_ranks_like_vsm(self):
        query = "rotating machine mechanical wear"
        vsm = [d for d, s in ENGINE.rank(query, "vsm") if s > 0]
        lsi = [d for d, s in ENGINE.rank(query, "lsi", k=ENGINE.max_k)][: len(vsm)]
        self.assertEqual(vsm, lsi)

    def test_gvsm_correlation_matrix(self):
        self.assertAlmostEqual(ENGINE.gvsm.correlation("peak", "peak"), 1.0)
        self.assertAlmostEqual(ENGINE.gvsm.correlation("peak", "tariff"), ENGINE.gvsm.correlation("tariff", "peak"))
        self.assertGreater(ENGINE.gvsm.correlation("peak", "tariff"), ENGINE.gvsm.correlation("peak", "gasket"))

    def test_lsi_learns_related_words_without_a_thesaurus(self):
        near = [n["word"] for n in ENGINE.related_terms("peak")[:5]]
        self.assertIn("tariff", near)


class Explanations(unittest.TestCase):
    def test_word_contributions_add_up_to_the_score(self):
        for model in ("vsm", "gvsm", "lsi", "hybrid"):
            response = ENGINE.search("cold room warming up", model=model, top=3)
            for r in response["results"]:
                e = r["explanation"]
                total = sum(w["contribution"] for w in e["words"]) + e["other_words"]
                self.assertAlmostEqual(total, r["score"], places=2, msg=model)

    def test_concept_contributions_add_up_to_the_lsi_score(self):
        lsi = ENGINE.lsi()
        scored = lsi.score(ENGINE.dtm.query_vector("cold room warming up").vector)
        j = int(np.argmax(scored.scores))
        self.assertAlmostEqual(float(lsi.concept_contributions(scored, j).sum()), float(scored.scores[j]))

    def test_lsi_finds_documents_that_share_no_word(self):
        response = ENGINE.search("equipment left on after the factory closes", model="lsi", top=5)
        meaning = {r["document"]["id"] for r in response["results"] if r["found_by_meaning"]}
        self.assertTrue(meaning & {"KB02", "KB36", "KB09"})


class Search(unittest.TestCase):
    def test_results_are_ranked_and_limited(self):
        response = ENGINE.search("compressed air wastage", top=5)
        scores = [r["score"] for r in response["results"]]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertLessEqual(len(scores), 5)
        self.assertEqual([r["rank"] for r in response["results"]], list(range(1, len(scores) + 1)))

    def test_min_score_filters(self):
        response = ENGINE.search("compressed air wastage", min_score=0.5, top=40)
        self.assertTrue(all(r["score"] >= 0.5 for r in response["results"]))

    def test_friendly_errors(self):
        for query in ("", "xyzzy qwerty", "the on of"):
            with self.assertRaises(QueryError):
                ENGINE.search(query)
        with self.assertRaises(QueryError):
            ENGINE.search("compressor", model="bm25")

    def test_matrix_excerpt_matches_A(self):
        response = ENGINE.search("compressed air wastage")
        m = response["matrix"]
        first_doc = ENGINE.dtm.ids.index(m["docs"][0])
        first_term = ENGINE.dtm.index[m["terms"][0]["term"]]
        self.assertAlmostEqual(m["values"][0][0], round(float(ENGINE.dtm.A[first_term, first_doc]), 3))

    def test_judged_queries_reference_real_documents(self):
        ids = set(ENGINE.dtm.ids)
        for q in QUERIES:
            self.assertTrue(set(q["relevant"]) <= ids, q["id"])


class ConfusionMatrix(unittest.TestCase):
    def test_counts_add_up_and_match_precision(self):
        import evaluate
        ranked = ["KB01", "KB13", "KB08", "KB16", "KB27", "KB02"]
        relevant = ["KB01", "KB02", "KB08", "KB09"]
        c = evaluate.confusion(ranked, relevant, 40)
        self.assertEqual(c, {"tp": 2, "fp": 3, "fn": 2, "tn": 33})
        self.assertEqual(sum(c.values()), 40)
        self.assertAlmostEqual(c["tp"] / (c["tp"] + c["fp"]), evaluate.metrics(ranked, relevant)["precision@5"])
        self.assertAlmostEqual(c["tp"] / (c["tp"] + c["fn"]), evaluate.metrics(ranked, relevant)["recall@5"])


if __name__ == "__main__":
    unittest.main()
