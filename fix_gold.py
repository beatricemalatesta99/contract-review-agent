"""
One-off fix: update the `gold_clauses` dict inside reviews_C.json
using the current CUAD mapping from data/prepare_cuad.py.
Does NOT re-run the agents. Keeps predictions intact, refreshes gold.
"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
import config

sys.path.append(str(config.DATA_DIR))
from prepare_cuad import download_cuad_json, reorganise_cuad


def main():
    print("Loading categories...")
    with open(config.CATEGORIES_PATH) as f:
        categories = json.load(f)["categories"]

    print("Re-parsing CUAD with current mapping...")
    raw = download_cuad_json()
    all_contracts = reorganise_cuad(raw, categories)
    gold_by_id = {c["contract_id"]: c["gold_clauses"] for c in all_contracts}

    print("Loading existing reviews...")
    reviews_path = config.RESULTS_DIR / "reviews_C.json"
    with open(reviews_path) as f:
        reviews = json.load(f)

    print(f"  {len(reviews)} reviews found")

    patched = 0
    for r in reviews:
        cid = r.get("contract_id")
        if cid in gold_by_id:
            r["gold_clauses"] = gold_by_id[cid]
            patched += 1

    print(f"  patched gold_clauses on {patched}/{len(reviews)} reviews")

    with open(reviews_path, "w", encoding="utf-8") as f:
        json.dump(reviews, f, indent=2, ensure_ascii=False)
    print(f"  wrote back to {reviews_path}")
    print("Now run: python experiments/evaluate.py")


if __name__ == "__main__":
    main()