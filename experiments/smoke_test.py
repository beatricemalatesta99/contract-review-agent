"""
Smoke test: run one contract through the full orchestrator + validator
and print the output. Run this BEFORE launching the full experiment
sweep so you know the pipeline is end-to-end working.

Usage:
    python experiments/smoke_test.py
"""

import json
import sys
from pathlib import Path

# Allow `python experiments/smoke_test.py` from project root.
sys.path.append(str(Path(__file__).parent.parent))

import config
from agents.orchestrator import Orchestrator
from validators.rules import RuleBasedValidator


def main():
    # Load a single contract.
    with open(config.CUAD_SAMPLE_PATH) as f:
        contracts = json.load(f)

    with open(config.CATEGORIES_PATH) as f:
        categories = json.load(f)["categories"]

    contract = contracts[0]
    print(f"Testing on contract: {contract['title']}")
    print(f"  text length: {len(contract['text'])} chars")
    print(f"  gold categories present: {list(contract['gold_clauses'].keys())}")
    print()

    # Run orchestrator.
    orch = Orchestrator(categories=categories, use_planning=True)
    review = orch.review(contract["text"], contract_id=contract["contract_id"])

    # Run validator.
    validator = RuleBasedValidator()
    review = validator.validate(review, contract["text"])

    # Print summary.
    print()
    print("=" * 60)
    print(f"RESULTS for {contract['title']}")
    print("=" * 60)
    print(f"Total chunks: {review['n_chunks']}")
    print(f"Planned categories: {review['planned_categories']}")
    print(f"Validation: {review['validation']}")
    print(f"Final clauses: {len(review['clauses'])}")
    print()
    for i, c in enumerate(review["clauses"][:5]):
        print(f"--- Clause {i+1} ---")
        print(f"  Category:   {c.get('classified_category')}")
        print(f"  Risk:       {c.get('risk_level')}")
        print(f"  Confidence: {c.get('classifier_confidence')}")
        print(f"  Flags:      {c.get('validation_flags')}")
        print(f"  Text:       {c.get('text')[:200]}...")
        print()

    # Save full output for inspection.
    out_path = config.RESULTS_DIR / "smoke_test_output.json"
    with open(out_path, "w") as f:
        json.dump(review, f, indent=2)
    print(f"Full output saved to {out_path}")


if __name__ == "__main__":
    main()
