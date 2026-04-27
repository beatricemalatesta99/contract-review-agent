"""
Remove predictions whose category is no longer in categories.json.
This happens when we drop a category post-hoc after running the agents,
and we don't want those orphan predictions to inflate false positives.
"""

import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
import config


def main():
    with open(config.CATEGORIES_PATH, encoding="utf-8") as f:
        valid_names = {c["name"] for c in json.load(f)["categories"]}
    valid_names.add("OTHER")  # keep OTHER predictions for completeness
    print(f"Valid categories: {sorted(valid_names)}")

    reviews_path = config.RESULTS_DIR / "reviews_C.json"
    with open(reviews_path, encoding="utf-8") as f:
        reviews = json.load(f)

    removed_total = 0
    for r in reviews:
        before = len(r["clauses"])
        r["clauses"] = [
            c for c in r["clauses"]
            if c.get("classified_category") in valid_names
        ]
        removed = before - len(r["clauses"])
        removed_total += removed
        if removed:
            print(f"  {r['contract_id'][:50]}: removed {removed} orphan predictions")

    print(f"\nTotal orphan predictions removed: {removed_total}")

    with open(reviews_path, "w", encoding="utf-8") as f:
        json.dump(reviews, f, indent=2, ensure_ascii=False)
    print(f"Cleaned reviews saved to {reviews_path}")
    print("Now run: python experiments/evaluate.py")


if __name__ == "__main__":
    main()