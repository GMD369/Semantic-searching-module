# Week 7 Requirement Mapping

The case-study slide (18) lists six minimum requirements for IR students. The development-focus slide (15) lists the features a semantic ranking module should have. All of them are implemented.

## Minimum requirements (slide 18)

| # | Requirement | Implementation | How to demonstrate |
|---|---|---|---|
| 1 | **Create document-term matrix** | `matrix.py` builds A, **513 terms × 40 documents**. Each entry is `(1 + ln tf) × ln(N / df)` and each column is scaled to length 1. Rows are terms and columns are documents, as on slide 11. | Search anything and open **See the document-term matrix for this query** under the results. Also `results/document_term_matrix.png`. |
| 2 | **Convert query into vector** | `query_vector()` uses the same tokenizer, stemmer and idf, so the query lives in the same 513-dimensional space. Words the collection never uses are reported, not silently dropped. | The **Query vector q** row shows each term with its TF-IDF weight. In `equipment left on after the factory closes`, *factory* is struck out because no document uses it. |
| 3 | **Calculate cosine similarity** | `cosine()` and `ProjectionModel.score()` compute `q̂·d̂ / (‖q̂‖ ‖d̂‖)`. VSM works in term space, GVSM in correlated term space, LSI in k concept dimensions. | Open **Why this score?**. The formula line shows the dot product, both lengths and the result, e.g. `0.391 / (0.712 × 0.673) = 0.816`. |
| 4 | **Rank documents** | `search()` sorts all 40 documents by cosine (ties broken by id) and drops those below the minimum similarity. | The `#1, #2, …` ranks and score rings. Switch the model between VSM and LSI and watch the order change. |
| 5 | **Show top results** | Top 5 / 10 / 20 selector, minimum-similarity slider, highlighted words, CLI `--top`. | Sidebar → **Show top**. |
| 6 | **Explain how semantic matching improves retrieval** | Every score is split exactly into word contributions (typed vs related) and LSI concept contributions. Results with no shared word are flagged **Found by meaning**. The status bar shows how the model *reads* the query. The written explanation and the evaluation give the evidence. | `equipment left on after the factory closes`: three playbooks are marked "Found by meaning", and VSM finds only 5 documents. [SEMANTIC_MODEL_EXPLANATION.md](SEMANTIC_MODEL_EXPLANATION.md) |

## Development-focus features (slide 15)

| Feature | Implementation |
|---|---|
| Document-term matrix | `DocumentTermMatrix.A` |
| Vector representation (TF-IDF) | Documents are columns of A; the query is `query_vector()` |
| Cosine similarity | `models.cosine`, `ProjectionModel.score` |
| Semantic similarity | GVSM term correlations (`C = T Tᵀ`) and LSI concepts |
| Basic LSI idea | `SVD` class: `A = U Σ Vᵀ`, keep k, compare in concept space |
| Ranked search result | `engine.search` |
| Optional matrix reduction | **k slider** (1–40) in the interface; `results/k_sensitivity.png` |

## Demonstration path (about 5 minutes)

1. **The problem.** Choose **VSM** and type `equipment left on after the factory closes`. Only 5 documents share a word, and the overnight compressor incident is fifth with 0.05.
2. **Semantic matching.** Switch to **LSI**. The *Model reads it as* row now contains *shutdown, weekly, sunday, night, hours*, words you never typed. The humidification and compressor incidents come first, and three playbooks are marked **Found by meaning**.
3. **Why this score?** Open #1 (KB08). The formula line is the cosine from slide 7. The table shows which words carried the score, and *Shared concepts* shows concept 7 (*shutdown, overnight, sunday*) doing most of the work.
4. **The matrix.** Open **See the document-term matrix**. Typed terms have a q weight; the related rows have q = 0 but a positive weight in *q read as*. That is semantic matching in one table.
5. **Concepts and k.** Point at the concept list (columns of U). Drag **k** to 2: topics blur together. Drag it to 40: the ranking becomes plain VSM. Return it to 14.
6. **Another domain.** Click `shift heavy loads to cheaper hours`. The query never says *peak* or *tariff*, but LSI reads it that way.
7. **Honesty.** Click `dirty heat exchanger surfaces`. KB35 wins because it contains *heat exchanger* literally. This failure is reported in the README.
8. **Evidence.** Show `results/comparison_chart.png`, `results/per_query_ndcg.png` and `results/console_output.txt`.

## Lecture examples reproduced

`python evaluate.py` prints these, and `tests/` checks them:

| Slide | Calculation | Module output |
|---|---|---|
| 6–8 | Q = (1, 1), D₁ = (5, 3): `8 / (√2 × √34)` | 0.970 |
| 7 | 1 = same direction, 0 = orthogonal, −1 = opposite | `cosine()` tests |
| 11–12 | A = [[1,1,0],[1,0,1],[0,1,1]] | σ = 2, 1, 1; U Σ Vᵀ rebuilds A exactly; rank-2 approximation printed |
| 12 | keep the top k singular values | `SVD.reconstruct(k)`, `LSIModel(k)` |
| 13 | different words → one hidden concept | concept 2 = *peak, tariff, maximum, charge*; LSI neighbours of *peak*: tariff 1.00, maximum 1.00 |

## Clear FYP relevance

- The documents are what a VoltMize user looks up after an anomaly alert. Factories, zones, meters and faults come from the VoltMize simulator configuration.
- Operators describe symptoms, while reports describe causes. That gap is a vocabulary mismatch, the exact problem slides 4 and 10 describe.
- The concepts are interpretable energy topics (compressed-air leaks, peak tariff, phase imbalance, bearing wear, short cycling, power factor), so the model can be checked by a domain expert.
- **Integration path (not implemented here):** the engine needs only numpy. The SVD is computed once when the engine starts (well under a second for this collection). It could sit behind a factory-scoped VoltMize endpoint, and re-run the SVD when the knowledge base changes.
