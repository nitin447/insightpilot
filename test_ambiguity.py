from tools.warehouse import find_ambiguities

for q in [
    "How many orders were placed in each month of 2025?",
    "What is the total order count for Q3?",
    "How many customers do we have?",
    "What was the average time to delivery?",
    "Which region has the most revenue?",     # expect no match
]:
    hits = ", ".join(h["term"] for h in find_ambiguities(q)) or "-"
    print(f"{hits:<40} {q}")