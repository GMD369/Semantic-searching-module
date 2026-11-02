"""Compare VSM, GVSM, LSI and the hybrid on the judged queries.

Writes everything to results/:
    <model>_results.json          per-query rankings and metrics for vsm, gvsm, lsi, hybrid
    comparison.csv                one row per query and model
    console_output.txt            exactly what this script prints
    comparison_chart.png          mean metrics per model
    per_query_ndcg.png            nDCG@5 per query, VSM vs LSI
    k_sensitivity.png             how LSI quality changes with the number of concepts k
    singular_values.png           the SVD spectrum and the energy rule that chose k
    concept_space.png             the 40 documents and a query in the first two concepts
    document_term_matrix.png      a window of the document-term matrix
    confusion_matrix.csv          TP / FP / FN / TN per query and model (top 5 = retrieved)
    comparison_table.md           every metric x every model, per-query tables, win/tie/loss vs VSM
    comparison_summary.csv        the same summary table as CSV (opens in Excel)
    confusion_matrices.png        the summed 2 x 2 confusion matrix for each model

Run:  python evaluate.py
"""
from __future__ import annotations

import csv
import io
import json
import math
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from semantic_search import MODELS, SemanticSearchEngine
from semantic_search.engine import ENERGY_RULE
from semantic_search.lecture import slide_7_cosine, slide_11_svd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
K = 5
METRICS = ["precision@5", "recall@5", "f1@5", "mrr", "ndcg@5", "average_precision"]
ORDER = ["vsm", "gvsm", "lsi", "hybrid"]
NAMES = {"vsm": "VSM (TF-IDF)", "gvsm": "GVSM", "lsi": "LSI", "hybrid": "Hybrid"}
# validated categorical palette (dataviz reference, slots 1-4, light surface)
COLORS = {"vsm": "#2a78d6", "gvsm": "#eb6834", "lsi": "#1baf7a", "hybrid": "#eda100"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"


def metrics(ranked: list[str], relevant: list[str]) -> dict[str, float]:
    rel = set(relevant)
    top = ranked[:K]
    hits = sum(doc in rel for doc in top)
    precision, recall = hits / K, hits / len(rel)
    f1 = 2 * precision * recall / (precision + recall) if hits else 0.0
    mrr = next((1 / (i + 1) for i, doc in enumerate(ranked) if doc in rel), 0.0)
    dcg = sum(1 / math.log2(i + 2) for i, doc in enumerate(top) if doc in rel)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(rel), K)))
    found, precisions = 0, []
    for i, doc in enumerate(ranked):
        if doc in rel:
            found += 1
            precisions.append(found / (i + 1))
    return {"precision@5": precision, "recall@5": recall, "f1@5": f1, "mrr": mrr,
            "ndcg@5": dcg / idcg, "average_precision": sum(precisions) / len(rel)}


def confusion(ranked: list[str], relevant: list[str], n_docs: int) -> dict[str, int]:
    """Confusion matrix at the cut-off: the top 5 count as "retrieved", the rest as "not retrieved".

                        relevant   not relevant
        retrieved          TP           FP
        not retrieved      FN           TN
    """
    top, rel = set(ranked[:K]), set(relevant)
    tp, fp, fn = len(top & rel), len(top - rel), len(rel - top)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": n_docs - tp - fp - fn}


def confusion_scores(c: dict[str, int]) -> dict[str, float]:
    """Micro-averaged scores computed from summed counts (every query-document pair counts once)."""
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "accuracy": (tp + tn) / (tp + fp + fn + tn), "specificity": tn / (tn + fp) if tn + fp else 0.0}


def retrieved(engine, query, model, k=None) -> list[tuple[str, float]]:
    """A document is retrieved when its similarity is above zero."""
    return [(d, s) for d, s in engine.rank(query, model, k) if s > 1e-9]


def run_model(engine, model, queries, k=None) -> dict:
    rows = []
    for q in queries:
        ranked = retrieved(engine, q["query"], model, k)
        ids = [d for d, _ in ranked]
        rows.append({
            "query_id": q["id"], "query": q["query"], "relevant": q["relevant"], "retrieved": len(ids),
            "ranking": [{"rank": i + 1, "document": d, "score": round(s, 4)} for i, (d, s) in enumerate(ranked[:15])],
            "metrics": {m: round(v, 4) for m, v in metrics(ids, q["relevant"]).items()},
            "confusion": confusion(ids, q["relevant"], len(engine.documents)),
        })
    mean = {m: round(sum(r["metrics"][m] for r in rows) / len(rows), 4) for m in METRICS}
    total = {key: sum(r["confusion"][key] for r in rows) for key in ("tp", "fp", "fn", "tn")}
    return {"model": model, "k_concepts": engine.lsi(k).k if model in ("lsi", "hybrid") else None,
            "cutoff": K, "queries": rows, "mean": mean,
            "confusion": total, "confusion_scores": {m: round(v, 4) for m, v in confusion_scores(total).items()}}


SUMMARY_ROWS = [
    # (label, where the number lives, higher is better)
    ("Precision@5", ("mean", "precision@5"), True),
    ("Recall@5", ("mean", "recall@5"), True),
    ("F1@5", ("mean", "f1@5"), True),
    ("MRR", ("mean", "mrr"), True),
    ("nDCG@5", ("mean", "ndcg@5"), True),
    ("MAP", ("mean", "average_precision"), True),
    ("TP (relevant in top 5)", ("confusion", "tp"), True),
    ("FP (irrelevant in top 5)", ("confusion", "fp"), False),
    ("FN (relevant missed)", ("confusion", "fn"), False),
    ("TN", ("confusion", "tn"), True),
    ("Precision (micro)", ("confusion_scores", "precision"), True),
    ("Recall (micro)", ("confusion_scores", "recall"), True),
    ("F1 (micro)", ("confusion_scores", "f1"), True),
    ("Accuracy", ("confusion_scores", "accuracy"), True),
]


def _fmt(v) -> str:
    return str(v) if isinstance(v, int) else f"{v:.3f}"


def comparison_table(results, queries) -> str:
    """Markdown: every metric x every model, best value in bold, plus LSI minus VSM; then per-query tables."""
    head = "| Metric | " + " | ".join(NAMES[m] for m in ORDER) + " | Best | LSI − VSM |"
    lines = ["# Evaluation comparison table", "",
             f"12 judged queries, cut-off {K}, LSI k = {results['lsi']['k_concepts']}. "
             "Generated by `python evaluate.py`; best value per row in **bold**.", "",
             "## 1. All metrics, all models", "", head, "|" + "---|" * (len(ORDER) + 3)]
    for label, (group, key), higher in SUMMARY_ROWS:
        values = {m: results[m][group][key] for m in ORDER}
        best = max(values.values()) if higher else min(values.values())
        winners = [NAMES[m] for m in ORDER if values[m] == best]
        cells = [f"**{_fmt(values[m])}**" if values[m] == best else _fmt(values[m]) for m in ORDER]
        diff = values["lsi"] - values["vsm"]
        diff_s = (f"{diff:+d}" if isinstance(diff, int) else f"{diff:+.3f}") if diff else "0"
        best_s = "all equal" if len(winners) == len(ORDER) else " / ".join(winners)
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {best_s} | {diff_s} |")
    lines += ["", "- **@5 metrics** are averaged per query (macro). **Micro** scores come from the summed "
              "confusion counts, so they can differ slightly (e.g. VSM recall 0.627 vs 0.615).",
              "- For FP and FN, lower is better.",
              "- Accuracy is high for every model because most query-document pairs are true negatives; "
              "it is listed for completeness, not used to compare the models.", ""]

    for title, key in (("nDCG@5", "ndcg@5"), ("Average precision", "average_precision")):
        lines += [f"## {2 if key == 'ndcg@5' else 3}. {title} per query", "",
                  "| Query | " + " | ".join(NAMES[m] for m in ORDER) + " | LSI − VSM |",
                  "|" + "---|" * (len(ORDER) + 2)]
        for i, q in enumerate(queries):
            values = {m: results[m]["queries"][i]["metrics"][key] for m in ORDER}
            best = max(values.values())
            cells = [f"**{values[m]:.2f}**" if values[m] == best and best > 0 else f"{values[m]:.2f}" for m in ORDER]
            diff = values["lsi"] - values["vsm"]
            lines.append(f"| {q['id']} {q['query']} | " + " | ".join(cells) + f" | {diff:+.2f} |")
        top = max(results[m]["mean"][key] for m in ORDER)
        means = [f"**{v:.3f}**" if v == top else f"{v:.3f}" for v in (results[m]["mean"][key] for m in ORDER)]
        lines.append("| **Mean** | " + " | ".join(means) + " | "
                     f"{results['lsi']['mean'][key] - results['vsm']['mean'][key]:+.3f} |")
        lines.append("")

    lines += ["## 4. Win / tie / loss against VSM (nDCG@5, per query)", "",
              "| Model | Better | Same | Worse |", "|---|---|---|---|"]
    for m in ("gvsm", "lsi", "hybrid"):
        diffs = [r["metrics"]["ndcg@5"] - v["metrics"]["ndcg@5"]
                 for r, v in zip(results[m]["queries"], results["vsm"]["queries"])]
        lines.append(f"| {NAMES[m]} | {sum(d > 1e-9 for d in diffs)} | {sum(abs(d) <= 1e-9 for d in diffs)} | "
                     f"{sum(d < -1e-9 for d in diffs)} |")
    return "\n".join(lines) + "\n"


def export_matrices(engine) -> Path:
    """Write every LSI matrix as a labelled CSV (opens in Excel) to results/matrices/.

        A          document-term matrix     513 terms x 40 documents   (TF-IDF, columns length 1)
        U_k        term-concept matrix      513 terms x k concepts
        Sigma_k    singular values          k values (and all 40 with energy)
        V_k^T      concept-document matrix  k concepts x 40 documents
        Sigma V^T  documents in concept space (what LSI compares with the query)
        A_k        rank-k approximation U_k Sigma_k V_k^T
    """
    out = RESULTS / "matrices"
    out.mkdir(exist_ok=True)
    dtm, svd, k = engine.dtm, engine.svd, engine.default_k
    words = [dtm.display[t] for t in dtm.terms]
    concepts = [f"concept_{c + 1}" for c in range(k)]

    def write(name, row_label, row_names, col_names, matrix):
        with (out / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([row_label] + list(col_names))
            for name_, row in zip(row_names, np.asarray(matrix)):
                w.writerow([name_] + [round(float(v), 4) for v in row])

    write("1_document_term_matrix_A.csv", "term", words, dtm.ids, dtm.A)
    write("1b_raw_term_counts.csv", "term", words, dtm.ids, dtm.counts)
    write(f"2_term_concept_U_k{k}.csv", "term", words, concepts, svd.U[:, :k])
    with (out / "3_singular_values_sigma.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["concept", "sigma", "sigma_squared", "cumulative_energy", f"kept_in_k{k}"])
        for c, (s, e) in enumerate(zip(svd.s, svd.energy)):
            w.writerow([c + 1, round(float(s), 4), round(float(s * s), 4), round(float(e), 4), c < k])
    write(f"4_concept_document_Vt_k{k}.csv", "concept", concepts, dtm.ids, svd.Vt[:k])
    write(f"5_document_concept_SigmaVt_k{k}.csv", "document", dtm.ids, concepts,
          (np.diag(svd.s[:k]) @ svd.Vt[:k]).T)
    write(f"6_rank_k{k}_approximation_A_k.csv", "term", words, dtm.ids, svd.reconstruct(k))
    with (out / f"7_concept_top_terms_k{k}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["concept", "sigma", "energy_share", "strongest_positive_terms", "strongest_negative_terms",
                    "strongest_documents"])
        for info in engine.concepts(k):
            w.writerow([info["concept"], info["singular_value"], info["energy"], ", ".join(info["positive"]),
                        ", ".join(info["negative"]), ", ".join(info["documents"])])
    return out


def print_matrices(engine):
    dtm, svd, k = engine.dtm, engine.svd, engine.default_k
    print("=" * 78)
    print("LSI MATRICES  (A = U S V^T, full CSVs in results/matrices/)")
    print("=" * 78)
    print(f"A    document-term matrix    {dtm.shape[0]} terms x {dtm.shape[1]} documents")
    print(f"U_k  term-concept matrix     {dtm.shape[0]} terms x {k} concepts")
    print(f"S_k  singular values         {k} of {len(svd.s)}: " + ", ".join(f"{s:.3f}" for s in svd.s[:k]))
    print(f"V_k^T concept-document       {k} concepts x {dtm.shape[1]} documents")
    print(f"U_k^T A = S_k V_k^T equal:   {np.allclose(svd.U[:, :k].T @ dtm.A, np.diag(svd.s[:k]) @ svd.Vt[:k])}")
    j = dtm.ids.index("KB08")
    print(f"KB08 in concept space (first 5 of {k}): "
          + ", ".join(f"{v:.3f}" for v in (np.diag(svd.s[:k]) @ svd.Vt[:k])[:5, j]))
    print()


def print_confusion(results, n_queries, n_docs):
    print("=" * 78)
    print(f"CONFUSION MATRICES AT CUT-OFF {K}  (summed over {n_queries} queries x {n_docs} documents "
          f"= {n_queries * n_docs} pairs)")
    print("=" * 78)
    print("Top 5 = retrieved, everything else = not retrieved.\n")
    for m in ORDER:
        c, s = results[m]["confusion"], results[m]["confusion_scores"]
        print(f"{NAMES[m]}")
        print(f"                     relevant   not relevant")
        print(f"   retrieved (top 5)  TP {c['tp']:>4}      FP {c['fp']:>4}")
        print(f"   not retrieved      FN {c['fn']:>4}      TN {c['tn']:>4}")
        print(f"   precision {s['precision']:.3f}  recall {s['recall']:.3f}  F1 {s['f1']:.3f}  "
              f"accuracy {s['accuracy']:.3f}  specificity {s['specificity']:.3f}\n")
    print("Accuracy and specificity look high for every model because most pairs are true negatives")
    print("(about 37 of 40 documents are irrelevant to each query). Compare models on TP, FN, precision and recall.")
    print()


def lecture_checks():
    print("=" * 78)
    print("LECTURE EXAMPLES REPRODUCED WITH THE MODULE'S OWN CODE")
    print("=" * 78)
    c = slide_7_cosine()
    print("Slides 6-8  Q = (1, 1), D1 = (5, 3)")
    print(f"            Q.D = {c['dot']:.0f}   |Q| = {c['q_norm']:.3f} (sqrt 2)   |D| = {c['d_norm']:.3f} (sqrt 34)")
    print(f"            cos = 8 / (sqrt 2 x sqrt 34) = {c['cosine']:.3f}   -> highly relevant")
    s = slide_11_svd()
    print("Slides 11-12  A = [[1,1,0],[1,0,1],[0,1,1]]  (terms x documents)")
    print(f"            singular values  = {np.round(s['S'], 3).tolist()}")
    print(f"            U S V^T rebuilds A exactly: {s['rebuilt_exactly']}")
    print("            rank-2 approximation A_2 =")
    for row in np.round(s["A_k"], 2):
        print("              ", row.tolist())
    sim_t = np.round(s["doc_similarity_terms"], 2)
    sim_c = np.round(s["doc_similarity_concepts"], 2)
    print(f"            cos(D1, D2) in term space = {sim_t[0][1]}, in 2-concept space = {sim_c[0][1]}")
    print()


def print_model(result, names):
    print("=" * 78)
    extra = f", k = {result['k_concepts']} concepts" if result["k_concepts"] else ""
    print(f"{names[result['model']].upper()}  (cut-off {K}{extra})")
    print("=" * 78)
    for row in result["queries"]:
        rel = set(row["relevant"])
        top = ", ".join(f"{r['document']}{'*' if r['document'] in rel else ''} {r['score']:.2f}"
                        for r in row["ranking"][:K]) or "(nothing retrieved)"
        m = row["metrics"]
        print(f"{row['query_id']:<4} {row['query']}")
        print(f"     top 5: {top}")
        print(f"     P@5 {m['precision@5']:.2f}  R@5 {m['recall@5']:.2f}  nDCG@5 {m['ndcg@5']:.2f}  AP {m['average_precision']:.2f}"
              f"  (retrieved {row['retrieved']})")
    print("     * = judged relevant")
    print()


def found_by_meaning(engine, queries, k=None) -> list[tuple[str, str]]:
    """Relevant documents LSI puts in its top 5 that share no word with the query."""
    out = []
    for q in queries:
        vsm = dict(engine.rank(q["query"], "vsm"))
        for d, _ in retrieved(engine, q["query"], "lsi", k)[:K]:
            if d in q["relevant"] and vsm[d] <= 0:
                out.append((q["id"], d))
    return out


# ----------------------------------------------------------------------- charts
def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def charts(engine, results, queries, sweep):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch  # noqa: F401  (rounded ends not needed for PNG)
    except ImportError:
        print("matplotlib is not installed; charts skipped.")
        return
    plt.rcParams.update({"font.family": "Segoe UI", "font.size": 10, "text.color": INK,
                         "axes.labelcolor": MUTED, "axes.titlesize": 12, "axes.titleweight": "bold"})

    # 1. mean metrics per model --------------------------------------------------
    labels = ["P@5", "R@5", "F1@5", "MRR", "nDCG@5", "MAP"]
    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=150)
    width = 0.19
    x = np.arange(len(METRICS))
    for i, model in enumerate(ORDER):
        values = [results[model]["mean"][m] for m in METRICS]
        bars = ax.bar(x + (i - 1.5) * width, values, width - 0.03, color=COLORS[model], label=NAMES[model])
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=7, color=MUTED)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("mean over 12 judged queries")
    ax.set_title(f"Semantic models vs exact-word VSM (cut-off {K}, LSI k = {engine.default_k})", loc="left")
    ax.legend(frameon=False, ncol=4, loc="upper left", fontsize=9)
    _style(ax)
    fig.tight_layout()
    fig.savefig(RESULTS / "comparison_chart.png")
    plt.close(fig)

    # 2. per-query nDCG, VSM vs LSI (dumbbell) ------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=150)
    ids = [q["id"] for q in queries]
    vsm = [r["metrics"]["ndcg@5"] for r in results["vsm"]["queries"]]
    lsi = [r["metrics"]["ndcg@5"] for r in results["lsi"]["queries"]]
    for i, (a, b) in enumerate(zip(vsm, lsi)):
        ax.plot([i, i], [a, b], color=GRID, linewidth=2, zorder=1)
    ax.scatter(range(len(ids)), vsm, s=60, color=COLORS["vsm"], label=NAMES["vsm"], zorder=2,
               edgecolor="white", linewidth=1.5)
    ax.scatter(range(len(ids)), lsi, s=60, color=COLORS["lsi"], label=NAMES["lsi"], zorder=3,
               edgecolor="white", linewidth=1.5)
    for i, (a, b) in enumerate(zip(vsm, lsi)):
        if abs(b - a) >= 0.05:
            ax.text(i + 0.12, (a + b) / 2, f"{b - a:+.2f}", fontsize=8, color=MUTED, va="center")
    ax.set_xticks(range(len(ids)), ids)
    ax.set_ylim(-0.05, 1.08)
    ax.set_ylabel("nDCG@5")
    ax.set_title("Per query: where concept matching helps, and where it does not", loc="left")
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    _style(ax)
    fig.tight_layout()
    fig.savefig(RESULTS / "per_query_ndcg.png")
    plt.close(fig)

    # 3. k sensitivity -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4.4), dpi=150)
    ks = [row["k"] for row in sweep]
    ax.plot(ks, [row["ndcg@5"] for row in sweep], color=COLORS["lsi"], linewidth=2, marker="o", markersize=4,
            label="LSI nDCG@5")
    ax.axhline(results["vsm"]["mean"]["ndcg@5"], color=COLORS["vsm"], linewidth=1.5, linestyle="--",
               label="VSM nDCG@5 (no concepts)")
    ax.axvline(engine.default_k, color=MUTED, linewidth=1, linestyle=":")
    ax.text(engine.default_k + 0.5, 0.08, f"k = {engine.default_k}\nchosen by the energy rule\nbefore any query was run",
            fontsize=8, color=MUTED)
    ax.set_xlabel("number of concepts k kept from the SVD")
    ax.set_ylabel("mean nDCG@5")
    ax.set_ylim(0, 1)
    ax.set_title("Too few concepts blur topics together; all 40 concepts give back plain VSM", loc="left")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    _style(ax)
    fig.tight_layout()
    fig.savefig(RESULTS / "k_sensitivity.png")
    plt.close(fig)

    # 4. singular values and energy (two panels, one scale each) ------------------
    s = engine.svd.s
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4), dpi=150)
    a1.bar(range(1, len(s) + 1), s, color=[COLORS["lsi"] if i < engine.default_k else "#b9d9cc" for i in range(len(s))],
           width=0.7)
    a1.set_title("Singular values σ (concept strength)", loc="left")
    a1.set_xlabel("concept")
    _style(a1)
    a2.plot(range(1, len(s) + 1), engine.svd.energy, color=COLORS["lsi"], linewidth=2)
    a2.axhline(ENERGY_RULE, color=MUTED, linewidth=1, linestyle=":")
    a2.axvline(engine.default_k, color=MUTED, linewidth=1, linestyle=":")
    a2.text(engine.default_k + 0.8, 0.3, f"k = {engine.default_k} keeps\n{engine.svd.energy[engine.default_k - 1]:.0%} of Σσ²",
            fontsize=8, color=MUTED)
    a2.set_ylim(0, 1.02)
    a2.set_title("Cumulative share of Σσ² kept", loc="left")
    a2.set_xlabel("k")
    _style(a2)
    fig.tight_layout()
    fig.savefig(RESULTS / "singular_values.png")
    plt.close(fig)

    # 5. concept space -------------------------------------------------------------
    groups = {
        "compressed air / compressors": {"KB01", "KB02", "KB03", "KB04", "KB05", "KB07", "KB26", "KB29"},
        "peak tariff / scheduling": {"KB14", "KB21", "KB40", "KB09", "KB32"},
        "refrigeration / cold rooms": {"KB16", "KB17", "KB18", "KB19", "KB20", "KB33"},
        "power quality / phases": {"KB11", "KB15", "KB22", "KB30", "KB31", "KB37"},
    }
    group_colors = list(COLORS.values())
    coords = (np.diag(engine.svd.s[:3]) @ engine.svd.Vt[:3]).T
    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=150)
    other = [j for j, d in enumerate(engine.dtm.ids) if not any(d in g for g in groups.values())]
    ax.scatter(coords[other, 1], coords[other, 2], s=36, color="#c9c8c1", label="other topics",
               edgecolor="white", linewidth=1)
    for (name, members), color in zip(groups.items(), group_colors):
        idx = [j for j, d in enumerate(engine.dtm.ids) if d in members]
        ax.scatter(coords[idx, 1], coords[idx, 2], s=48, color=color, label=name, edgecolor="white", linewidth=1.2)
    q = engine.lsi(3).project(engine.dtm.query_vector("shift heavy loads to cheaper hours").vector)
    q = q / np.linalg.norm(q) * np.linalg.norm(coords[:, 1:3], axis=1).max() * 0.9
    ax.annotate("", xy=(q[1], q[2]), xytext=(0, 0),
                arrowprops={"arrowstyle": "-|>", "color": INK, "linewidth": 1.5})
    ax.text(q[1], q[2], '  query: "shift heavy loads\n  to cheaper hours"', fontsize=8, color=INK, va="center")
    for j in (engine.dtm.ids.index("KB21"), engine.dtm.ids.index("KB40"), engine.dtm.ids.index("KB03")):
        ax.text(coords[j, 1], coords[j, 2], " " + engine.dtm.ids[j], fontsize=7.5, color=MUTED)
    ax.axhline(0, color=GRID, linewidth=1)
    ax.axvline(0, color=GRID, linewidth=1)
    ax.set_xlabel("concept 2  (" + ", ".join(engine.concepts()[1]["positive"][:3]) + " …)")
    ax.set_ylabel("concept 3  (" + ", ".join(engine.concepts()[2]["positive"][:3]) + " …)")
    ax.set_title("Documents placed by meaning: SVD concepts 2 and 3", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, length=0)
    fig.tight_layout()
    fig.savefig(RESULTS / "concept_space.png")
    plt.close(fig)

    # 6. document-term matrix window -------------------------------------------------
    rows = ["compressor", "leak", "air", "peak", "tariff", "refrigeration", "overheating", "phase", "imbalance", "bearing"]
    terms = [t for t in (engine.dtm.query_vector(w).weights for w in rows) for t in t]
    docs = ["KB01", "KB03", "KB05", "KB14", "KB17", "KB21", "KB22", "KB26", "KB33", "KB37"]
    cols = [engine.dtm.ids.index(d) for d in docs]
    values = np.array(engine.dtm.excerpt(terms, cols))
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", ["#fcfcfb", "#cde2fb", "#86b6ef", "#2a78d6", "#184f95"])
    ax.imshow(values, cmap=cmap, aspect="auto", vmin=0, vmax=max(0.5, values.max()))
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            ax.text(j, i, f"{v:.2f}" if v else "·", ha="center", va="center", fontsize=7.5,
                    color="white" if v > 0.3 else (INK if v else "#b5b4ad"))
    ax.set_xticks(range(len(docs)), docs, fontsize=8)
    ax.set_yticks(range(len(terms)), [engine.dtm.display[t] for t in terms], fontsize=8.5)
    ax.tick_params(length=0, colors=MUTED)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title(f"A window of the {engine.dtm.shape[0]} × {engine.dtm.shape[1]} document-term matrix "
                 "(TF-IDF, columns length 1)", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS / "document_term_matrix.png")
    plt.close(fig)

    # 7. LSI matrices: term-concept U_k (strongest terms) and document-concept S_k V_k^T --
    k = engine.default_k
    diverging = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#a9c8ee", "#f2f1ec", "#f5bd9d", "#eb6834"])
    U_k, docs_c = engine.svd.U[:, :k], (np.diag(engine.svd.s[:k]) @ engine.svd.Vt[:k]).T
    picked = []
    for c in range(k):                      # the 2 strongest terms of every concept
        for i in np.argsort(-np.abs(U_k[:, c]))[:2]:
            if i not in picked:
                picked.append(i)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 9), dpi=150, gridspec_kw={"width_ratios": [1, 1]})
    for ax, data, rows, title in (
            (a1, U_k[picked], [engine.dtm.display[engine.dtm.terms[i]] for i in picked],
             f"Term-concept matrix U_k (strongest terms, {k} concepts)"),
            (a2, docs_c, engine.dtm.ids, f"Document-concept matrix Σ_k V_kᵀ (40 documents × {k})")):
        lim = np.abs(data).max()
        ax.imshow(data, cmap=diverging, vmin=-lim, vmax=lim, aspect="auto")
        ax.set_xticks(range(k), [str(c + 1) for c in range(k)], fontsize=7.5)
        ax.set_yticks(range(len(rows)), rows, fontsize=6.5)
        ax.set_xlabel("concept  (blue = negative, orange = positive)", fontsize=8.5)
        ax.tick_params(length=0, colors=MUTED)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_title(title, loc="left", fontsize=11)
    fig.suptitle(f"A = U Σ Vᵀ: the two matrices LSI compares in (σ in results/singular_values.png)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(RESULTS / "lsi_matrices.png")
    plt.close(fig)

    # 8. confusion matrices, one 2 x 2 panel per model ---------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.6), dpi=150)
    from matplotlib.colors import ListedColormap
    # correct decisions (TP, TN) in green, mistakes (FP, FN) in orange; the numbers carry the comparison
    cell_colors = ListedColormap(["#d3f0e3", "#fbe1d3"])
    for ax, model in zip(axes, ORDER):
        c, s = results[model]["confusion"], results[model]["confusion_scores"]
        cells = np.array([[c["tp"], c["fp"]], [c["fn"], c["tn"]]])
        names = [["TP  correct", "FP  wrong"], ["FN  missed", "TN  correct"]]
        ax.imshow(np.array([[0, 1], [1, 0]]), cmap=cell_colors, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cells[i, j]}", ha="center", va="center", fontsize=16, color=INK,
                        fontweight="bold")
                ax.text(j, i + 0.28, names[i][j], ha="center", va="center", fontsize=7.5, color=MUTED)
        ax.axhline(0.5, color="white", linewidth=2)
        ax.axvline(0.5, color="white", linewidth=2)
        ax.set_xticks([0, 1], ["relevant", "not relevant"], fontsize=8)
        ax.set_yticks([0, 1], ["top 5", "not in top 5"], fontsize=8)
        ax.tick_params(length=0, colors=MUTED)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_title(NAMES[model], loc="left", fontsize=11)
        ax.set_xlabel(f"precision {s['precision']:.2f} · recall {s['recall']:.2f}", fontsize=8.5, color=MUTED)
    fig.suptitle(f"Confusion matrices at cut-off {K}, summed over {len(queries)} queries", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(RESULTS / "confusion_matrices.png")
    plt.close(fig)


def main():
    RESULTS.mkdir(exist_ok=True)
    engine = SemanticSearchEngine.from_data_dir()
    queries = json.loads((ROOT / "data" / "queries.json").read_text(encoding="utf-8"))["queries"]
    results = {m: run_model(engine, m, queries) for m in ORDER}
    sweep = []
    for k in range(1, engine.max_k + 1):
        r = run_model(engine, "lsi", queries, k)
        sweep.append({"k": k, **r["mean"]})

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        print(f"Collection: {len(engine.documents)} documents, {engine.dtm.shape[0]} index terms "
              f"-> document-term matrix A is {engine.dtm.shape[0]} x {engine.dtm.shape[1]}")
        print(f"LSI keeps k = {engine.default_k} concepts: the smallest k whose singular values keep "
              f"{ENERGY_RULE:.0%} of sum(s^2) (actual {engine.svd.energy[engine.default_k - 1]:.1%}).")
        print("Judged queries: 12, written before any model was run.\n")
        lecture_checks()
        print_matrices(engine)
        for m in ORDER:
            print_model(results[m], NAMES)

        print("=" * 78)
        print(f"MEAN OVER {len(queries)} QUERIES")
        print("=" * 78)
        print(f"{'metric':<18}" + "".join(f"{NAMES[m]:>14}" for m in ORDER))
        for metric in METRICS:
            print(f"{metric:<18}" + "".join(f"{results[m]['mean'][metric]:>14.3f}" for m in ORDER))
        print()
        print_confusion(results, len(queries), len(engine.documents))

        wins = {"better": [], "same": [], "worse": []}
        for a, b in zip(results["vsm"]["queries"], results["lsi"]["queries"]):
            d = b["metrics"]["ndcg@5"] - a["metrics"]["ndcg@5"]
            wins["better" if d > 1e-9 else "worse" if d < -1e-9 else "same"].append(a["query_id"])
        print("LSI vs VSM on nDCG@5, query by query")
        for key, ids in wins.items():
            print(f"  {key:<7} {len(ids):>2}  {', '.join(ids)}")
        meaning = found_by_meaning(engine, queries)
        print(f"\nRelevant documents in LSI's top 5 that share NO word with the query "
              f"(VSM score 0): {len(meaning)}")
        print("  " + ", ".join(f"{q}:{d}" for q, d in meaning))
        print("\nLSI nDCG@5 as k changes (sensitivity only; k was not tuned on these numbers)")
        print("  " + "  ".join(f"k={row['k']}:{row['ndcg@5']:.2f}" for row in sweep if row["k"] in
                               (1, 2, 4, 6, 8, 10, 12, engine.default_k, 20, 25, 30, 35, 40)))
        print("\nSemantic neighbours learned by LSI (no thesaurus was given)")
        for word in ("overnight", "leak", "peak", "overheating", "imbalance", "bearing"):
            near = engine.related_terms(word)[:5]
            print(f"  {word:<12} -> " + ", ".join(f"{n['word']} {n['similarity']:.2f}" for n in near))

    text = buffer.getvalue()
    print(text)
    (RESULTS / "console_output.txt").write_text(text, encoding="utf-8")
    for m in ORDER:
        (RESULTS / f"{m}_results.json").write_text(json.dumps(results[m], indent=2), encoding="utf-8")
    (RESULTS / "k_sweep.json").write_text(json.dumps(sweep, indent=2), encoding="utf-8")
    with (RESULTS / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "query", "model"] + METRICS)
        for m in ORDER:
            for row in results[m]["queries"]:
                writer.writerow([row["query_id"], row["query"], m] + [row["metrics"][x] for x in METRICS])
    with (RESULTS / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "query_id", "tp", "fp", "fn", "tn"])
        for m in ORDER:
            for row in results[m]["queries"]:
                writer.writerow([m, row["query_id"]] + [row["confusion"][x] for x in ("tp", "fp", "fn", "tn")])
            writer.writerow([m, "TOTAL"] + [results[m]["confusion"][x] for x in ("tp", "fp", "fn", "tn")])
    export_matrices(engine)
    (RESULTS / "comparison_table.md").write_text(comparison_table(results, queries), encoding="utf-8")
    with (RESULTS / "comparison_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric"] + ORDER)
        for label, (group, key), _ in SUMMARY_ROWS:
            writer.writerow([label] + [results[m][group][key] for m in ORDER])
    charts(engine, results, queries, sweep)
    print(f"Results written to {RESULTS}")


if __name__ == "__main__":
    main()
