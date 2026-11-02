"""Console search.

    python cli.py "equipment left on after the factory closes"
    python cli.py "cold room warming up" --model vsm --top 5
    python cli.py "shift heavy loads to cheaper hours" --explain
    python cli.py --concepts            list the concepts LSI found
    python cli.py                       interactive prompt
"""
from __future__ import annotations

import argparse
import sys

from semantic_search import MODELS, QueryError, SemanticSearchEngine

ENGINE = SemanticSearchEngine.from_data_dir()


def show(query: str, args) -> None:
    try:
        r = ENGINE.search(query, model=args.model, k=args.k, top=args.top, min_score=args.min)
    except QueryError as error:
        print(f"  ! {error}")
        return
    vector = ", ".join(f"{t['word']} {t['weight']:.2f}" for t in r["query_vector"])
    print(f"\n  {r['model_label']}" + (f"  (k = {r['k']})" if args.model in ("lsi", "hybrid") else ""))
    print(f"  query vector: {vector}" + (f"   unknown: {', '.join(r['unknown'])}" if r["unknown"] else ""))
    if args.model != "vsm":
        print("  read as:      " + ", ".join(t["word"] for t in r["expanded_query"][:10]))
    print(f"  {r['total_positive']} of {r['total_considered']} documents above 0; exact-word VSM finds {r['vsm_zero_hits']}\n")
    for res in r["results"]:
        doc = res["document"]
        flag = "  [found by meaning]" if res["found_by_meaning"] and args.model != "vsm" else ""
        print(f"  {res['rank']:>2}. {res['score']:.3f}  {doc['id']}  {doc['title']}{flag}")
        if args.explain:
            e = res["explanation"]
            print(f"        {e['formula']}")
            print("        words: " + ", ".join(
                f"{w['word']}{'' if w['typed'] else '*'} +{w['contribution']:.3f}" for w in e["words"][:6]))
            if e.get("concepts"):
                print("        concepts: " + "; ".join(
                    f"{c['concept']} ({', '.join(c['label'][:3])}) +{c['contribution']:.3f}" for c in e["concepts"]))
    if args.explain:
        print("\n  * = related word added by the model, not typed")
    print()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="VoltMize semantic ranking (Week 7)")
    parser.add_argument("query", nargs="*")
    parser.add_argument("--model", choices=MODELS, default="lsi")
    parser.add_argument("--k", type=int, default=None, help=f"LSI concepts (default {ENGINE.default_k})")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--min", type=float, default=0.10)
    parser.add_argument("--explain", action="store_true")
    parser.add_argument("--concepts", action="store_true")
    args = parser.parse_args()

    if args.concepts:
        for c in ENGINE.concepts(args.k):
            print(f"  {c['concept']:>2}  σ={c['singular_value']:.3f}  {', '.join(c['positive'])}"
                  f"   | opposite side: {', '.join(c['negative'])}")
        return
    if args.query:
        show(" ".join(args.query), args)
        return
    print("VoltMize semantic search. Type a query, or an empty line to quit.")
    for line in sys.stdin if not sys.stdin.isatty() else iter(lambda: input("search> "), ""):
        if not line.strip():
            break
        show(line.strip(), args)


if __name__ == "__main__":
    main()
