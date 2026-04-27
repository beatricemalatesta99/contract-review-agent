"""
Run all three experimental conditions on the full sample of contracts.

Conditions:
    A: Baseline — single LLM call, no decomposition, no validation.
    B: Orchestrator — multi-agent orchestrator, no validation layer.
    C: Full system — orchestrator + rule-based validation layer.

Output: one JSON file per condition in results/, with the list of
reviews for all contracts. The evaluator reads these files.

We save incrementally after each contract so a crash halfway through
doesn't lose everything.

Usage:
    python experiments/run_all.py
    python experiments/run_all.py --conditions A       # just one condition
    python experiments/run_all.py --max 5              # quick test
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import config
from agents.baseline import BaselineAgent
from agents.orchestrator import Orchestrator
from validators.rules import RuleBasedValidator


def load_data():
    with open(config.CUAD_SAMPLE_PATH) as f:
        contracts = json.load(f)
    with open(config.CATEGORIES_PATH) as f:
        categories = json.load(f)["categories"]
    return contracts, categories


def run_condition_A(contracts, categories, out_path, max_contracts=None):
    """Baseline: single-shot LLM call per contract."""
    print("\n" + "=" * 60)
    print("CONDITION A: Baseline (single prompt, no validation)")
    print("=" * 60)

    agent = BaselineAgent(categories=categories)
    reviews = _run_loop(
        contracts=contracts,
        max_contracts=max_contracts,
        review_fn=lambda c: agent.review(c["text"], contract_id=c["contract_id"]),
        out_path=out_path,
    )
    return reviews


def run_condition_B(contracts, categories, out_path, max_contracts=None):
    """Orchestrator only, no validation layer."""
    print("\n" + "=" * 60)
    print("CONDITION B: Orchestrator (multi-agent, no validation)")
    print("=" * 60)

    orch = Orchestrator(categories=categories, use_planning=True)
    reviews = _run_loop(
        contracts=contracts,
        max_contracts=max_contracts,
        review_fn=lambda c: orch.review(c["text"], contract_id=c["contract_id"]),
        out_path=out_path,
    )
    return reviews


def run_condition_C(contracts, categories, out_path, max_contracts=None):
    """Full system: orchestrator + rule-based validation."""
    print("\n" + "=" * 60)
    print("CONDITION C: Full system (orchestrator + validation)")
    print("=" * 60)

    orch = Orchestrator(categories=categories, use_planning=True)
    validator = RuleBasedValidator()

    def review_fn(c):
        review = orch.review(c["text"], contract_id=c["contract_id"])
        review = validator.validate(review, c["text"])
        return review

    reviews = _run_loop(
        contracts=contracts,
        max_contracts=max_contracts,
        review_fn=review_fn,
        out_path=out_path,
    )
    return reviews


def _run_loop(contracts, max_contracts, review_fn, out_path):
    """
    Shared loop that runs review_fn over all contracts, saves
    incrementally, and handles per-contract failures gracefully.
    """
    if max_contracts is not None:
        contracts = contracts[:max_contracts]

    # Resume from previous partial run if the output file exists.
    already_done = {}
    if out_path.exists():
        with open(out_path) as f:
            prior = json.load(f)
        already_done = {r["contract_id"]: r for r in prior}
        print(f"  resuming: {len(already_done)} contracts already done")

    reviews = list(already_done.values())

    for i, contract in enumerate(contracts):
        cid = contract["contract_id"]
        if cid in already_done:
            continue

        t0 = time.time()
        try:
            review = review_fn(contract)
            # Attach gold clauses so the evaluator can use them later
            # without reloading the source data.
            review["gold_clauses"] = contract.get("gold_clauses", {})
            reviews.append(review)
        except Exception as e:
            print(f"  ✗ contract {cid} failed: {e}")
            reviews.append({
                "contract_id": cid,
                "error": str(e),
                "clauses": [],
                "gold_clauses": contract.get("gold_clauses", {}),
            })

        elapsed = time.time() - t0
        print(f"  [{i+1}/{len(contracts)}] {cid} done in {elapsed:.1f}s")

        # Save incrementally.
        with open(out_path, "w") as f:
            json.dump(reviews, f, indent=2)

    print(f"  saved {len(reviews)} reviews to {out_path}")
    return reviews


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conditions", default="A,B,C",
        help="comma-separated list of conditions to run (A, B, C)",
    )
    parser.add_argument(
        "--max", type=int, default=None,
        help="maximum number of contracts to run (for quick tests)",
    )
    args = parser.parse_args()

    conditions_to_run = [s.strip().upper() for s in args.conditions.split(",")]

    contracts, categories = load_data()
    print(f"Loaded {len(contracts)} contracts, {len(categories)} categories")
    if args.max:
        print(f"(running at most {args.max} contracts per condition)")

    if "A" in conditions_to_run:
        run_condition_A(
            contracts, categories,
            out_path=config.RESULTS_DIR / "reviews_A.json",
            max_contracts=args.max,
        )
    if "B" in conditions_to_run:
        run_condition_B(
            contracts, categories,
            out_path=config.RESULTS_DIR / "reviews_B.json",
            max_contracts=args.max,
        )
    if "C" in conditions_to_run:
        run_condition_C(
            contracts, categories,
            out_path=config.RESULTS_DIR / "reviews_C.json",
            max_contracts=args.max,
        )

    print("\nAll done. Next step: python experiments/evaluate.py")


if __name__ == "__main__":
    main()
