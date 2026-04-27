import json

with open("results/reviews_C.json", encoding="utf-8") as f:
    reviews = json.load(f)

total_dropped = 0
total_flagged = 0
total_kept = 0

for r in reviews:
    v = r.get("validation", {})
    total_dropped += v.get("dropped", 0)
    total_flagged += v.get("flagged", 0)
    total_kept += v.get("kept", 0)

total = total_dropped + total_kept
print(f"Total candidates extracted: {total}")
print(f"Dropped by validator (not grounded): {total_dropped}")
print(f"Flagged (kept but with warnings): {total_flagged}")
print(f"Kept: {total_kept}")
print()
if total > 0:
    print(f"% dropped: {total_dropped/total*100:.1f}%")
    print(f"% flagged: {total_flagged/total*100:.1f}%")