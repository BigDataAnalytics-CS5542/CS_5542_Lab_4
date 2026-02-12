"""Evaluation of hybrid RAG retrieval against a gold set of queries and evidence IDs.

Loads gold_queries.json (query_id, question, gold_evidence_ids), runs run_hybrid_query
for each, and computes P@5, R@10, plus missing_evidence and latency. Writes results to CSV.
"""
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Set, Optional

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
from backend.rag_pipeline import run_hybrid_query


GOLD_PATH = Path("data/data/gold_queries.json")
OUT_CSV = Path("data/data/eval_results.csv")


def precision_at_k(retrieved: List[str], gold: Set[str], k: int) -> float:
    """Precision at k: fraction of top-k retrieved IDs that are in the gold set.

    Args:
        retrieved: Ordered list of retrieved evidence IDs.
        gold: Set of relevant (gold) evidence IDs.
        k: Cutoff for top-k.

    Returns:
        Number in [0, 1]; 0.0 if top-k is empty.
    """
    top = retrieved[:k]
    if not top:
        return 0.0
    hits = sum(1 for eid in top if eid in gold)
    return hits / float(len(top))


def recall_at_k(retrieved: List[str], gold: Set[str], k: int) -> float:
    """Recall at k: fraction of gold IDs that appear in the top-k retrieved.

    Args:
        retrieved: Ordered list of retrieved evidence IDs.
        gold: Set of relevant (gold) evidence IDs.
        k: Cutoff for top-k.

    Returns:
        Number in [0, 1]; 0.0 if gold is empty.
    """
    if not gold:
        return 0.0
    top = retrieved[:k]
    hits = sum(1 for eid in top if eid in gold)
    return hits / float(len(gold))


def load_gold(path: Path = GOLD_PATH) -> List[Dict[str, Any]]:
    """Load the gold queries file (JSON array of query_id, question, gold_evidence_ids).

    Args:
        path: Path to gold_queries.json. Defaults to ``GOLD_PATH``.

    Returns:
        List of dicts, each with at least query_id, question, and optionally gold_evidence_ids.

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If file is not a JSON array.
    """
    if not path.exists():
        raise FileNotFoundError(f"Gold queries file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("gold_queries.json must be a JSON array")
    return data


def evaluate(
    alpha: float = 0.6,
    candidate_pool: int = 50,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """Run hybrid retrieval on each gold query and compute P@5, R@10, and metadata.

    For each gold query, calls run_hybrid_query, then computes precision-at-5 and recall-at-10
    using gold_evidence_ids. Aggregates latency, missing_evidence_behavior, and support_gate_pass.

    Args:
        alpha: Dense weight for hybrid fusion. Default 0.6.
        candidate_pool: Candidates per retriever before fusion. Default 50.
        top_k: Number of evidence items to retrieve (used for R@10 when top_k >= 10). Default 10.

    Returns:
        List of result dicts with query_id, question, alpha, candidate_pool, top_k, num_gold,
        p_at_5, r_at_10, missing_evidence, support_gate_pass, latency_ms, top1_evidence_id,
        unanswerable, reject_correct (True/False for unanswerable, None else), top1_correct_doc (answerable only).
    """
    gold_queries = load_gold()

    rows: List[Dict[str, Any]] = []
    for q in gold_queries:
        query_id = q["query_id"]
        question = q["question"]
        gold_ids = set(q.get("gold_evidence_ids", []))

        res = run_hybrid_query(
            question=question,
            top_k=top_k,
            alpha=alpha,
            candidate_pool=candidate_pool,
        )

        retrieved_ids = [e["evidence_id"] for e in res.get("evidence", [])]

        p5 = precision_at_k(retrieved_ids, gold_ids, k=5)
        r10 = recall_at_k(retrieved_ids, gold_ids, k=10)

        is_unanswerable = len(gold_ids) == 0
        missing = bool(res.get("missing_evidence_behavior"))
        reject_correct = missing if is_unanswerable else None

        top1_correct_doc = None
        if not is_unanswerable and gold_ids and retrieved_ids:
            expected_stem = list(gold_ids)[0].rsplit("_p", 1)[0] if "_p" in list(gold_ids)[0] else None
            top1_correct_doc = expected_stem and retrieved_ids[0].startswith(expected_stem + "_p")

        rows.append(
            {
                "query_id": query_id,
                "question": question,
                "alpha": alpha,
                "candidate_pool": candidate_pool,
                "top_k": top_k,
                "num_gold": len(gold_ids),
                "p_at_5": round(p5, 4),
                "r_at_10": round(r10, 4),
                "missing_evidence": missing,
                "support_gate_pass": bool(res.get("support_gate_pass")),
                "latency_ms": round(float(res.get("latency_ms", 0.0)), 2),
                "top1_evidence_id": (retrieved_ids[0] if retrieved_ids else ""),
                "unanswerable": is_unanswerable,
                "reject_correct": reject_correct,
                "top1_correct_doc": top1_correct_doc,
            }
        )

    return rows


def write_csv(rows: List[Dict[str, Any]], out_path: Path = OUT_CSV) -> None:
    """Write evaluation rows to a CSV file; fieldnames taken from the first row.

    Args:
        rows: List of result dicts from evaluate().
        out_path: Output CSV path. Defaults to ``OUT_CSV``. Parent directory is created if needed.

    Raises:
        ValueError: If rows is empty.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No evaluation rows to write")

    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def print_summary(rows: List[Dict[str, Any]]) -> None:
    """Print aggregate metrics (avg P@5, avg R@10 over answerable; reject accuracy for unanswerable)."""
    answerable = [r for r in rows if r["num_gold"] > 0]
    n_ans = len(answerable)
    if n_ans > 0:
        avg_p5 = sum(r["p_at_5"] for r in answerable) / n_ans
        avg_r10 = sum(r["r_at_10"] for r in answerable) / n_ans
    else:
        avg_p5 = avg_r10 = 0.0

    unans = [r for r in rows if r["unanswerable"]]
    if unans:
        reject_acc = sum(1 for r in unans if r["reject_correct"]) / len(unans)
    else:
        reject_acc = None

    top1_ok = [r for r in answerable if r.get("top1_correct_doc") is True]
    top1_ok_rate = len(top1_ok) / n_ans if n_ans else None

    print("\n=== Evaluation Summary ===")
    print(f"Queries: {len(rows)}  (answerable: {n_ans}, unanswerable: {len(unans)})")
    print(f"Avg P@5 (answerable only):  {avg_p5:.4f}")
    print(f"Avg R@10 (answerable only): {avg_r10:.4f}")
    if top1_ok_rate is not None:
        print(f"Top-1 from correct doc (answerable): {len(top1_ok)}/{n_ans} = {top1_ok_rate:.4f}")
    if unans:
        print(f"Reject accuracy (unanswerable): {reject_acc:.4f}  (Q5 handled separately)")
    print(f"CSV written to: {OUT_CSV}\n")

    for r in rows:
        extra = ""
        if r.get("top1_correct_doc") is not None:
            extra = f"  top1_ok={r['top1_correct_doc']}"
        if r["unanswerable"]:
            extra = f"  reject_correct={r['reject_correct']}"
        print(
            f"{r['query_id']}  P@5={r['p_at_5']:.4f}  R@10={r['r_at_10']:.4f}  "
            f"missing={r['missing_evidence']}  ms={r['latency_ms']:.2f}  top1={r['top1_evidence_id']}{extra}"
        )


if __name__ == "__main__":
    rows = evaluate(alpha=0.6, candidate_pool=50, top_k=10)
    write_csv(rows)
    print_summary(rows)
