# Semantic Ranking — Simple Explanation

## The idea in one sentence

Turn every document and the query into arrows (vectors) and rank documents by **how closely their arrow points in the same direction as the query's arrow**. LSI first squeezes thousands of word-directions into a few *meaning*-directions, so two documents about the same thing point the same way even when they use different words.

## Explained like the Feynman rubric

### 1. What goes in

A query, such as

```text
equipment left on after the factory closes
```

and the 40 VoltMize documents (incidents, playbooks, maintenance logs, recommendations).

### 2. What happens inside

**Step 1 — Build the document-term matrix A (requirement 1).** One row per term (513 of them after stemming and stop-word removal), one column per document (40). Each cell says how important that term is in that document:

```text
weight = (1 + ln tf) × ln(N / df)        tf = times in the document, df = documents containing it, N = 40
```

A word that appears in only one document (e.g. *gasket*, idf = ln 40 = 3.69) is a strong clue. A word in many documents (*compressor*) is a weak one. Every column is then scaled to length 1, so a long document does not win just by being long.

**Step 2 — Turn the query into a vector (requirement 2).** The same recipe gives the query a weight for each of the 513 terms. Most are 0. For our query:

| term | weight | note |
|---|---|---|
| equipment | 3.69 | used in 1 document |
| closes → *clos* | 3.69 | matches *closed* in KB08 after stemming |
| left | 2.30 | used in 4 documents |
| factory | — | **no document uses it**, so it has no dimension |

**Step 3a — Classical VSM: cosine in term space (requirement 3).**

```text
cos(q, d) = q·d / (‖q‖ ‖d‖)
```

This is exactly slide 7. The dot product only adds up terms that are non-zero in **both** vectors, so a document that shares no word with the query scores 0 no matter how relevant it is. For this query, only 5 of the 40 documents score above 0.

**Step 3b — LSI: cosine in concept space.** Singular Value Decomposition splits the matrix:

```text
A = U Σ Vᵀ        U: term → concept      Σ: concept strength      Vᵀ: concept → document
```

We keep only the **k = 14 strongest concepts**: the smallest k that keeps half of the matrix's energy Σσ². The concepts are not invented by us; they fall out of which words share documents:

| concept | strongest words |
|---|---|
| 2 | peak, tariff, maximum, charge, window, evening |
| 3 | leak, air, compressed, valve, drain |
| 4 | phase, heater, welding, imbalance, single |
| 5 | bearing, vibration, wear, spindle |
| 6 | short, switch, float, cycling, started |
| 8 | furnace, factor, harmonic, capacitor |

The query and every document are projected onto these 14 axes, and the cosine is taken there:

```text
q̂ = U_kᵀ q        d̂ = U_kᵀ d        score = q̂·d̂ / (‖q̂‖ ‖d̂‖)
```

The query words *left* and *closed* load on the concept that also carries *shutdown, overnight, Sunday*. A document full of those words now points the same way as the query, **even if it never says "left" or "closed"**.

**Step 3c — GVSM, the in-between model.** GVSM keeps the 513 term axes but stops pretending they are at right angles. The angle between two term axes comes from how often the terms occur in the same documents (the correlation matrix `C = T Tᵀ`). Then `score = qᵀ C d / √(qᵀCq · dᵀCd)`. It matches slide 16's formula `d_j = Σ wᵢⱼ gᵢ`: a document is a weighted sum of *generalized* term vectors gᵢ.

**Step 4 — Rank and show the top results (requirements 4 and 5).** Sort by score, drop anything under the minimum similarity (0.10 by default), and show the top 5, 10 or 20.

### 3. What comes out

| rank | document | LSI cosine | exact-word VSM |
|---|---|---|---|
| 1 | KB08 Humidification left on during weekly shutdown | 0.82 | 0.17 |
| 2 | KB01 Compressor C1 left running overnight | 0.71 | 0.05 (5th) |
| 3 | KB02 Playbook: compressor running after shift end | 0.61 | **0 — no shared word** |
| 4 | KB36 Playbook: investigating high night-time base load | 0.60 | **0 — no shared word** |
| 5 | KB09 Schedule the humidifier plant with the production calendar | 0.59 | **0 — no shared word** |

All five are judged relevant. Under VSM, the top 5 contained two irrelevant documents (KB13, KB16), and three relevant playbooks could not be found at all.

### 4. Why that result came first — the score taken apart

Because `q̂·d̂ = (U_k U_kᵀ q)·d`, the cosine can be split **exactly** into one share per document word. `U_k U_kᵀ q` is the query *as LSI reads it*: a weight on every vocabulary word, including words you never typed. For KB08:

```text
q·d / (|q|·|d|) = 0.391 / (0.712 × 0.673) = 0.816
```

| word in KB08 | typed? | adds to the score |
|---|---|---|
| shutdown | no — related | +0.080 |
| humidification | no — related | +0.079 |
| weekly | no — related | +0.055 |
| sunday | no — related | +0.054 |
| switch | no — related | +0.034 |
| … all other words | | +0.43 |

By concept: concept 7 (*shutdown, overnight, sunday*) gives +0.29, concept 1 (*compressor, peak, night*) +0.18, and so on. The interface draws exactly these numbers in **Why this score?**, so the explanation is the same arithmetic that produced the score.

## How semantic matching improves retrieval (requirement 6)

1. **It solves vocabulary mismatch (synonymy).** Users and report writers use different words for the same thing. LSI found three relevant playbooks for Q1 that share no word with the query. On 12 judged queries, MAP rose from 0.633 (VSM) to 0.735 (LSI) and nDCG@5 from 0.656 to 0.714.
2. **It needs no thesaurus.** Week 6 linked *overnight* and *night* with a hand-written synonym list. LSI learned that *peak* ≈ *tariff* ≈ *maximum demand*, *bearing* ≈ *vibration* ≈ *spindle* and *imbalance* ≈ *phase* from the documents alone.
3. **It reduces noise.** Dropping the weak concepts (k = 14 of 40) removes word choices that occur only once, which is the "reduce noise" on slide 15.
4. **It gives a better ordering, not just more results.** In Q8 (`cold room warming up`) and Q6 (`machine switching on and off too often`), LSI moved every relevant document into the top positions (nDCG@5 = 1.00).

## Where it does not help — honest limitations

- **Polysemy.** One word, two meanings. In Q5 (`cooling plant getting too hot`), *cooling* co-occurs with *pre-cooling* in the peak-tariff documents, so LSI reads the query as a tariff question. A single vector per word cannot separate senses.
- **Unknown words have no vector.** *Hot*, *dirty* and *factory* never occur in the collection, so they contribute nothing. Q11 (`motor running hot`) got worse than VSM for that reason.
- **Exact words can still mislead.** In Q10 (`dirty heat exchanger surfaces`), KB35 literally contains *heat exchanger* and wins under every model, although it is about heat recovery, not fouling.
- **Small collection.** Co-occurrence statistics from 40 short documents are thin. LSI's concepts are clearer in collections of thousands of documents.
- **The choice of k matters.** k = 2 gives nDCG@5 0.24 because topics merge; k = 40 reproduces VSM exactly (0.656). Anything from about 10 to 30 performs similarly here.
- **Negative cosines.** In concept space a cosine can be below 0, meaning the document points away from the query. Those documents are never shown.

## Lecture quiz (slide 19) answered with this module

1. **What is an algebraic IR model?** A model that represents documents and queries as vectors or matrices and computes relevance algebraically, R = F(Q, D), here as a cosine.
2. **What is the purpose of GVSM?** To drop VSM's assumption that terms are independent (orthogonal axes). Term vectors get angles from their correlations, so a query word also counts for the words it occurs with. In this module, `C = T Tᵀ`.
3. **What does cosine similarity measure?** The angle between the query and document vectors: `q·d / (‖q‖‖d‖)`. 1 means the same direction, 0 means unrelated, −1 means opposite. Slide 8's example gives 0.97 in `evaluate.py`.
4. **What is a term-document matrix?** Rows = terms, columns = documents, values = weights. Here it is 513 × 40 TF-IDF (`results/document_term_matrix.png`).
5. **Why is LSI useful?** It retrieves by meaning: different words that occur in the same contexts end up close together, so relevant documents are found even without the query's exact words (KB02, KB36, KB09 for Q1).
6. **What is the role of SVD in LSI?** It factorizes A = U Σ Vᵀ, orders the hidden concepts by strength σ, and lets us keep only the top k. That reduces dimensionality and noise and reveals the latent structure.
7. **How can LSI help in FYP or research paper retrieval?** In VoltMize an operator's description of a symptom finds the incident report and playbook written in an engineer's words. In an FYP repository, a search for "energy waste detection" would also find a project titled "anomaly alerts for idle machines".
