"""
Compute evaluation metrics for all three conditions.

For each contract, we have:
    - `gold_clauses`: {category: [list of gold spans]} from CUAD.
    - `clauses` (from the system): list of predictions with text +
      classified_category.

Matching a predicted clause to a gold span is a design choice. Common
options:

    - EXACT MATCH: the predicted text equals a gold span. Rarely
      achieved in practice because of whitespace / punctuation.
    - CONTAINMENT: the predicted text is a substring of a gold span or
      vice versa.
    - FUZZY / JACCARD OVERLAP: token overlap above a threshold.

We use CONTAINMENT + FUZZY (token Jaccard >= 0.5) as "match", which is
lenient but honest: if the system's output overlaps substantially with
a gold annotation, we count it as a hit.

Metrics reported per condition:
    - Precision, Recall, F1 — overall and per category
    - Number of true positives, false positives, false negatives
    - A simple confusion table (counts by category)

Output:
    - results/evaluation.json  — machine-readable metrics
    - results/evaluation.md    — human-readable report for the paper

Usage:
    python experiments/evaluate.py
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import config


JACCARD_THRESHOLD = 0.5


def tokenise(text: str) -> set:
    """Very simple tokeniser: lowercased alphanumeric tokens."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def overlap_score(a: str, b: str) -> float:
    """Token-level Jaccard similarity between two strings."""
    ta, tb = tokenise(a), tokenise(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def is_match(pred_text: str, gold_text: str) -> bool:
    """
    Decide if a predicted clause matches a gold span.

    Accept if:
        - one is a substring of the other (after normalisation), OR
        - token Jaccard overlap >= JACCARD_THRESHOLD.
    """
    pred = re.sub(r"\s+", " ", pred_text.lower()).strip()
    gold = re.sub(r"\s+", " ", gold_text.lower()).strip()
    if not pred or not gold:
        return False
    if pred in gold or gold in pred:
        return True
    return overlap_score(pred, gold) >= JACCARD_THRESHOLD


def evaluate_review(review: dict) -> dict:
    """
    Count TP / FP / FN for one contract, per category.

    Logic:
        - For each category, take the list of gold spans.
        - For each predicted clause labelled with that category, check
          if it matches ANY unused gold span. If yes: TP (and that gold
          span is marked used). If no: FP.
        - Every gold span left unused at the end: FN.
    """
    gold = review.get("gold_clauses", {})
    preds = review.get("clauses", [])

    # Group predictions by their assigned category.
    preds_by_cat = defaultdict(list)
    for p in preds:
        cat = p.get("classified_category") or p.get("category") or ""
        preds_by_cat[cat].append(p.get("text", ""))

    per_category = {}
    for cat, gold_spans in gold.items():
        # Skip categories with no gold annotations — they don't
        # contribute to recall.
        if not gold_spans:
            per_category[cat] = {"tp": 0, "fp": 0, "fn": 0}
            continue

        used = [False] * len(gold_spans)
        tp = 0
        fp = 0
        for pred_text in preds_by_cat.get(cat, []):
            matched = False
            for i, gs in enumerate(gold_spans):
                if used[i]:
                    continue
                if is_match(pred_text, gs):
                    used[i] = True
                    matched = True
                    break
            if matched:
                tp += 1
            else:
                fp += 1

        fn = used.count(False)
        per_category[cat] = {"tp": tp, "fp": fp, "fn": fn}

    # Also count FPs in predicted categories that have no gold
    # annotations for this contract (hallucinated categories).
    gold_cats = set(gold.keys())
    for cat, pred_texts in preds_by_cat.items():
        if cat in gold_cats or cat == "OTHER" or cat == "":
            continue
        per_category.setdefault(cat, {"tp": 0, "fp": 0, "fn": 0})
        per_category[cat]["fp"] += len(pred_texts)

    return per_category


def aggregate(per_contract: list[dict]) -> dict:
    """
    Aggregate per-contract per-category counts into overall and
    per-category metrics.
    """
    totals = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for counts in per_contract:
        for cat, c in counts.items():
            totals[cat]["tp"] += c["tp"]
            totals[cat]["fp"] += c["fp"]
            totals[cat]["fn"] += c["fn"]

    def prf(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        return p, r, f

    per_category = {}
    sum_tp = sum_fp = sum_fn = 0
    for cat, c in totals.items():
        p, r, f = prf(c["tp"], c["fp"], c["fn"])
        per_category[cat] = {
            "precision": round(p, 3),
            "recall": round(r, 3),
            "f1": round(f, 3),
            **c,
        }
        sum_tp += c["tp"]; sum_fp += c["fp"]; sum_fn += c["fn"]

    micro_p, micro_r, micro_f = prf(sum_tp, sum_fp, sum_fn)

    # Macro average over categories that have any gold annotation.
    gold_cats = [cat for cat, m in per_category.items()
                 if m["tp"] + m["fn"] > 0]
    if gold_cats:
        macro_p = sum(per_category[c]["precision"] for c in gold_cats) / len(gold_cats)
        macro_r = sum(per_category[c]["recall"]    for c in gold_cats) / len(gold_cats)
        macro_f = sum(per_category[c]["f1"]        for c in gold_cats) / len(gold_cats)
    else:
        macro_p = macro_r = macro_f = 0.0

    return {
        "per_category": per_category,
        "micro": {
            "precision": round(micro_p, 3),
            "recall":    round(micro_r, 3),
            "f1":        round(micro_f, 3),
            "tp": sum_tp, "fp": sum_fp, "fn": sum_fn,
        },
        "macro": {
            "precision": round(macro_p, 3),
            "recall":    round(macro_r, 3),
            "f1":        round(macro_f, 3),
        },
    }


def evaluate_condition(path: Path) -> dict:
    """Load a condition's reviews and compute aggregated metrics."""
    if not path.exists():
        print(f"  [skip] {path.name} not found")
        return None

    with open(path, encoding="utf-8") as f:
        reviews = json.load(f)

    per_contract = [evaluate_review(r) for r in reviews]
    metrics = aggregate(per_contract)
    metrics["n_contracts"] = len(reviews)
    return metrics


def write_report(all_metrics: dict, out_path: Path):
    """
    Write a markdown report with per-condition summary tables.
    Paste this directly into the paper or use it as the basis for
    LaTeX tables.
    """
    lines = []
    lines.append("# Evaluation report\n")

    # --- Overall summary table across conditions ---
    lines.append("## Overall metrics across conditions\n")
    lines.append("| Condition | Contracts | Precision (micro) | Recall (micro) | F1 (micro) | F1 (macro) |")
    lines.append("|---|---|---|---|---|---|")
    labels = {
        "A": "A: Baseline (single prompt)",
        "B": "B: Orchestrator (no validation)",
        "C": "C: Full system (orchestrator + validation)",
    }
    for cid in ["A", "B", "C"]:
        m = all_metrics.get(cid)
        if m is None:
            continue
        lines.append(
            f"| {labels[cid]} | {m['n_contracts']} "
            f"| {m['micro']['precision']:.3f} "
            f"| {m['micro']['recall']:.3f} "
            f"| {m['micro']['f1']:.3f} "
            f"| {m['macro']['f1']:.3f} |"
        )
    lines.append("")

    # --- Per-category F1 for each condition ---
    lines.append("## Per-category F1\n")
    # Collect the union of categories seen anywhere.
    cats = set()
    for m in all_metrics.values():
        if m:
            cats.update(m["per_category"].keys())
    cats = sorted(c for c in cats if c and c != "OTHER")

    header = "| Category | " + " | ".join(
        f"F1 ({cid})" for cid in ["A", "B", "C"] if all_metrics.get(cid)
    ) + " |"
    sep = "|---" * (1 + sum(1 for cid in "ABC" if all_metrics.get(cid))) + "|"
    lines.append(header)
    lines.append(sep)

    for cat in cats:
        row = f"| {cat} |"
        for cid in ["A", "B", "C"]:
            m = all_metrics.get(cid)
            if m is None:
                continue
            entry = m["per_category"].get(cat)
            f1 = f"{entry['f1']:.3f}" if entry else "-"
            row += f" {f1} |"
        lines.append(row)
    lines.append("")

    # --- Per-condition detail ---
    for cid in ["A", "B", "C"]:
        m = all_metrics.get(cid)
        if m is None:
            continue
        lines.append(f"## Detail — Condition {cid}\n")
        lines.append("| Category | TP | FP | FN | Precision | Recall | F1 |")
        lines.append("|---|---|---|---|---|---|---|")
        for cat in cats:
            entry = m["per_category"].get(cat)
            if not entry:
                continue
            lines.append(
                f"| {cat} | {entry['tp']} | {entry['fp']} | {entry['fn']} "
                f"| {entry['precision']:.3f} | {entry['recall']:.3f} "
                f"| {entry['f1']:.3f} |"
            )
        lines.append("")

    out_path.write_text("\n".join(lines))
    print(f"  wrote markdown report to {out_path}")


def main():
    print("Evaluating all available condition outputs...")
    all_metrics = {}
    for cid, filename in [
        ("A", "reviews_A.json"),
        ("B", "reviews_B.json"),
        ("C", "reviews_C.json"),
    ]:
        path = config.RESULTS_DIR / filename
        m = evaluate_condition(path)
        if m is not None:
            all_metrics[cid] = m
            print(f"  {cid}: micro F1 = {m['micro']['f1']:.3f}, "
                  f"macro F1 = {m['macro']['f1']:.3f} "
                  f"(n = {m['n_contracts']})")

    # Save JSON and markdown report.
    json_path = config.RESULTS_DIR / "evaluation.json"
    md_path = config.RESULTS_DIR / "evaluation.md"
    with open(json_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"  wrote JSON metrics to {json_path}")
    write_report(all_metrics, md_path)


if __name__ == "__main__":
    main()
