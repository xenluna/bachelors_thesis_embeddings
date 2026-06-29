"""Quantitative Evaluation der semantischen Produktsuche.

Erzeugt:
    evaluation/results/search_evaluation.csv
    evaluation/results/search_evaluation.json
    evaluation/results/search_evaluation_table.tex

Die Implementierung nutzt dieselbe Index-/Embedding-Logik wie der Prototyp
(ProductIndex + Embedder). Dadurch wird nicht ein separates Benchmark-System,
sondern die tatsaechliche Produktsuche der Arbeit evaluiert.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Sequence, Set

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vector_index import Embedder, ProductIndex, build_index  # noqa: E402

DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_ground_truth(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    queries = data.get("queries", [])
    if not queries:
        raise ValueError(f"No queries found in {path}")
    for item in queries:
        if "query" not in item or "relevant_product_ids" not in item:
            raise ValueError("Each ground-truth entry needs 'query' and 'relevant_product_ids'.")
        if not item["relevant_product_ids"]:
            raise ValueError(f"Ground-truth entry for {item['query']!r} has no relevant products.")
    return queries


def precision_at_k(ranked_ids: Sequence[str], relevant_ids: Set[str], k: int) -> float:
    top_k = ranked_ids[:k]
    if k <= 0:
        raise ValueError("k must be positive")
    return sum(pid in relevant_ids for pid in top_k) / k


def recall_at_k(ranked_ids: Sequence[str], relevant_ids: Set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top_k = ranked_ids[:k]
    return sum(pid in relevant_ids for pid in top_k) / len(relevant_ids)


def reciprocal_rank(ranked_ids: Sequence[str], relevant_ids: Set[str]) -> float:
    for rank, pid in enumerate(ranked_ids, start=1):
        if pid in relevant_ids:
            return 1.0 / rank
    return 0.0


def dcg_at_k(ranked_ids: Sequence[str], relevant_ids: Set[str], k: int) -> float:
    score = 0.0
    for rank, pid in enumerate(ranked_ids[:k], start=1):
        rel = 1.0 if pid in relevant_ids else 0.0
        score += rel / math.log2(rank + 1)
    return score


def ndcg_at_k(ranked_ids: Sequence[str], relevant_ids: Set[str], k: int) -> float:
    dcg = dcg_at_k(ranked_ids, relevant_ids, k)
    ideal_hits = min(len(relevant_ids), k)
    if ideal_hits == 0:
        return 0.0
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / ideal


def ensure_index(index_dir: Path, products_path: Path, model_name: str) -> None:
    faiss_file = index_dir / "faiss.index"
    payload_file = index_dir / "payloads.json"
    if faiss_file.exists() and payload_file.exists():
        return
    print(f"Index not found in {index_dir}. Building it from {products_path} ...")
    build_index(str(products_path), out_dir=str(index_dir), model_name=model_name)


def evaluate(
    ground_truth_path: Path,
    products_path: Path,
    index_dir: Path,
    model_name: str,
    k: int,
) -> Dict[str, Any]:
    ensure_index(index_dir=index_dir, products_path=products_path, model_name=model_name)

    gt_items = load_ground_truth(ground_truth_path)
    embedder = Embedder(model_name=model_name)
    pindex = ProductIndex.load(str(index_dir))

    per_query: List[Dict[str, Any]] = []
    for item in gt_items:
        query = item["query"]
        relevant_ids = {str(pid) for pid in item["relevant_product_ids"]}
        q_vec = embedder.embed_query(query)
        results = pindex.search(q_vec, k=k)
        ranked_ids = [str(payload.get("id")) for _, payload in results]
        ranked_titles = [str(payload.get("title", "")) for _, payload in results]

        row = {
            "query": query,
            "num_relevant": len(relevant_ids),
            "top_k_ids": ranked_ids,
            "top_k_titles": ranked_titles,
            f"precision@{k}": precision_at_k(ranked_ids, relevant_ids, k),
            f"recall@{k}": recall_at_k(ranked_ids, relevant_ids, k),
            "mrr": reciprocal_rank(ranked_ids, relevant_ids),
            f"ndcg@{k}": ndcg_at_k(ranked_ids, relevant_ids, k),
        }
        per_query.append(row)

    metric_keys = [f"precision@{k}", f"recall@{k}", "mrr", f"ndcg@{k}"]
    summary = {metric: mean(row[metric] for row in per_query) for metric in metric_keys}
    return {
        "model_name": model_name,
        "k": k,
        "num_queries": len(per_query),
        "per_query": per_query,
        "summary": summary,
    }


def write_csv(result: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    k = result["k"]
    fields = ["query", "num_relevant", "top_k_ids", "top_k_titles", f"precision@{k}", f"recall@{k}", "mrr", f"ndcg@{k}"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in result["per_query"]:
            writer.writerow({
                **row,
                "top_k_ids": "; ".join(row["top_k_ids"]),
                "top_k_titles": "; ".join(row["top_k_titles"]),
            })


def latex_escape(text: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "\\": r"\textbackslash{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def write_latex_table(result: Dict[str, Any], out_path: Path) -> None:
    k = result["k"]
    newline = r" \\"  # LaTeX row ending
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        rf"Query & Precision@{k} & Recall@{k} & MRR & nDCG@{k}" + newline,
        r"\midrule",
    ]
    for row in result["per_query"]:
        lines.append(
            f"{latex_escape(row['query'])} & "
            f"{row[f'precision@{k}']:.3f} & "
            f"{row[f'recall@{k}']:.3f} & "
            f"{row['mrr']:.3f} & "
            f"{row[f'ndcg@{k}']:.3f}" + newline
        )
    s = result["summary"]
    lines.extend([
        r"\midrule",
        f"Durchschnitt & {s[f'precision@{k}']:.3f} & {s[f'recall@{k}']:.3f} & {s['mrr']:.3f} & {s[f'ndcg@{k}']:.3f}" + newline,
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{Quantitative Evaluation der semantischen Produktsuche bei $k={k}$ auf zehn manuell annotierten Test-Queries.}}",
        r"\label{tab:search-evaluation}",
        r"\end{table}",
        "",
    ])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def print_result(result: Dict[str, Any]) -> None:
    k = result["k"]
    print(f"\nEvaluation finished: {result['num_queries']} queries, k={k}")
    for row in result["per_query"]:
        print(
            f"{row['query']:<45} "
            f"P@{k}={row[f'precision@{k}']:.3f} "
            f"R@{k}={row[f'recall@{k}']:.3f} "
            f"MRR={row['mrr']:.3f} "
            f"nDCG@{k}={row[f'ndcg@{k}']:.3f}"
        )
    s = result["summary"]
    print("\nAverages:")
    print(
        f"P@{k}={s[f'precision@{k}']:.3f}, "
        f"R@{k}={s[f'recall@{k}']:.3f}, "
        f"MRR={s['mrr']:.3f}, "
        f"nDCG@{k}={s[f'ndcg@{k}']:.3f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate semantic product search.")
    parser.add_argument("--ground-truth", type=Path, default=REPO_ROOT / "evaluation" / "data" / "ground_truth.json")
    parser.add_argument("--products", type=Path, default=REPO_ROOT / "data" / "mock_data_products.jsonl")
    parser.add_argument("--index-dir", type=Path, default=REPO_ROOT / "product_index")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "evaluation" / "results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate(
        ground_truth_path=args.ground_truth,
        products_path=args.products,
        index_dir=args.index_dir,
        model_name=args.model_name,
        k=args.k,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "search_evaluation.json"
    csv_path = args.out_dir / "search_evaluation.csv"
    tex_path = args.out_dir / "search_evaluation_table.tex"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(result, csv_path)
    write_latex_table(result, tex_path)
    print_result(result)
    print(f"\nWrote:\n- {json_path}\n- {csv_path}\n- {tex_path}")


if __name__ == "__main__":
    main()
