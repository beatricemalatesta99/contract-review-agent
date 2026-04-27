import json

with open("results/reviews_C.json", encoding="utf-8") as f:
    reviews = json.load(f)

print("=" * 70)
print("TUTTI I GOLD SPAN per 'Indemnification'")
print("=" * 70)

for r in reviews[:5]:
    golds = r["gold_clauses"].get("Indemnification", [])
    print(f"\n>>> {r['contract_id'][:60]}")
    print(f"    {len(golds)} gold span(s)")
    for i, g in enumerate(golds[:2]):
        print(f"    --- span {i} ---")
        print(f"    {g[:250]}")