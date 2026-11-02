# Code Guide

The model is kept separate from the interface. `semantic_search/` is plain Python plus numpy. `app.py` and `cli.py` only call `SemanticSearchEngine.search()` and display what it returns.

```text
documents ──► text.py ──► matrix.py: A (513 × 40) ──► models.py ──┐
                              │                    (SVD once)      │
query text ─► text.py ──► query_vector q ────────────────────────► engine.py ──► ranked top results + explanation
```

## `text.py` — words

- `raw_words(text)` lower-cases the text, keeps letters and digits, and removes stop words and pure numbers.
- `stem(word)` is the same light suffix stripper as Week 6 (*leakage → leak*, *running → run*).
- `terms(text)` combines the two: these are the index terms. `word_spans()` gives the same terms with their positions, for highlighting.

## `matrix.py` — requirements 1 and 2

`DocumentTermMatrix(documents)`:

| attribute | meaning |
|---|---|
| `terms`, `index` | the vocabulary (sorted) and term → row number |
| `counts` | raw term frequency, m × n |
| `df`, `idf` | document frequency and `ln(N / df)` |
| `A` | `(1 + ln tf) × idf`, each column scaled to length 1 — **the matrix every model uses** |
| `display` | the most common surface form of a stem (*tarif → tariff*), for showing to users |

`query_vector(text)` returns a `QueryVector`: the m-length vector, the known terms with their weights, and the unknown words.

## `models.py` — requirement 3

Every model is one recipe with a different projection matrix P:

```python
q_hat = P.T @ q                     # the query in the model's space
d_hat = P.T @ A[:, j]               # document j in the same space
score = q_hat @ d_hat / (norm(q_hat) * norm(d_hat))
```

| class | P | space |
|---|---|---|
| `VectorSpaceModel` | identity | 513 term axes, exact words only |
| `GeneralizedVSM` | T = A with every **row** scaled to length 1 | P Pᵀ = C, the term-term correlation matrix |
| `LSIModel` | U_k, the first k columns of U from the SVD | k concept axes |
| `HybridModel` | — | `α × LSI + (1 − α) × VSM`, α = 0.5 |

`SVD(A)` runs `numpy.linalg.svd` once. It fixes each concept's sign (largest loading positive) so that the labels are stable, and exposes `energy` (cumulative share of Σσ²) and `reconstruct(k)`.

The key identity for explanations is `q̂·d̂ = (P Pᵀ q)·d`. `Scored.expanded` stores `P Pᵀ q`, the query as the model reads it, so:

- `term_contributions(scored, j)` = `expanded × A[:, j] / (‖q̂‖‖d̂‖)`: one number per word, summing exactly to the score.
- `LSIModel.concept_contributions(scored, j)` = `q̂ × d̂ / (‖q̂‖‖d̂‖)`: one number per concept, summing exactly to the score.

The tests check both sums.

## `engine.py` — requirements 4, 5 and 6

`SemanticSearchEngine.from_data_dir()` builds the matrix and the SVD and chooses k:

```python
ENERGY_RULE = 0.5
default_k = smallest k with energy[k-1] >= 0.5        # = 14 for this collection
```

`search(query, model="lsi", k=None, top=10, min_score=0.10)`:

1. Builds the query vector. Raises `QueryError` with a readable message for an empty query, only unknown words, only stop words or an unknown model.
2. Scores all documents with the chosen model, and always with VSM too (for the "found by meaning" comparison).
3. Sorts by score (ties by id), keeps those ≥ `min_score` and returns the first `top`.
4. For each result: score, VSM score, `found_by_meaning` (VSM score 0), the formula line, the top 8 contributing words, the top concepts, and highlight segments (the 10 strongest words, marked *exact* or *semantic*).
5. For the whole query: the query vector, unknown words, the expanded query (top 12 of `P Pᵀ q`), the query's strongest concepts, counts, and a small window of A (`matrix`) for the interface.

There are also `scores()` and `rank()` (used by `evaluate.py`), `concepts(k)` (the concept list) and `related_terms(word)` (nearest words in concept space, rows of `U_k Σ_k`).

## `lecture.py`

`slide_7_cosine()` computes Q = (1, 1) and D = (5, 3) to get 0.970. `slide_11_svd()` decomposes the 3 × 3 slide matrix, checks that it is rebuilt exactly and returns the rank-2 approximation.

## `evaluate.py`

For each of the 12 queries in `data/queries.json`, it ranks with VSM, GVSM, LSI and the hybrid (a document is retrieved when its score is above 0). It computes P@5, R@5, F1@5, MRR, nDCG@5 and AP. It also sweeps k from 1 to 40 (sensitivity only), lists the relevant documents LSI found with no shared word, and prints the LSI neighbours of some domain words. It writes JSON, CSV, the console transcript and six charts to `results/`.

## `app.py` and `web/`

`app.py` is a standard-library HTTP server on port 8091:

- `GET /api/meta` returns the matrix shape, default and maximum k, the energy curve and the examples.
- `GET /api/concepts?k=14` returns the concept list for the sidebar.
- `GET /api/search?q=…&model=lsi|hybrid|gvsm|vsm&k=14&top=10&min=0.1` returns the full `search()` response.

`web/app.js` debounces typing, ignores out-of-date responses, renders cards and explanation tables with DOM methods (not raw HTML), keeps the query in the URL, and highlights the concepts the current query uses.
