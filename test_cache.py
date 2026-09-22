from tools.cache import lookup, store, similarity, clear, stats

clear()
FP = "testschema123"
BASE = "Why did revenue drop in Q3 2025?"
store(BASE, FP, {"answer": "South electronics"})

checks = [
    ("Why did revenue drop in Q3 2025?",           "exact"),
    ("why did revenue drop in q3 2025",            "exact"),
    ("What caused the revenue drop in Q3 2025?",   "fuzzy"),
    ("What drove the Q3 2025 decline in revenue?", "fuzzy"),
    ("Why did revenue drop in Q2 2024?",           "miss"),
    ("Why did revenue rise in Q3 2025?",           "miss"),
    ("Why did orders drop in Q3 2025?",            "miss"),
    ("Which payment type is most common?",         "miss"),
]

for q, expect in checks:
    payload, how = lookup(q, FP)
    got = how.split()[0] if how else "miss"
    flag = "OK " if got == expect else "!! "
    print(f"{flag}{got:6s} sim={similarity(BASE, q):.2f}  <- {q}")

print("\nwrong schema ->", lookup(BASE, "other")[1] or "miss")
print("stats:", stats())