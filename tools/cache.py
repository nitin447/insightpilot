"""
Two-layer result cache.

Exact: same question + same schema -> replay, zero API calls.
Fuzzy: reworded question -> matched by token overlap after synonym
       normalization, so demos and repeated questions cost nothing.

Keyed on a schema fingerprint, so uploading new data invalidates the cache
automatically instead of serving answers about the wrong tables.
"""
from __future__ import annotations
import os, json, re, time, hashlib

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         ".cache")
os.makedirs(CACHE_DIR, exist_ok=True)
INDEX = os.path.join(CACHE_DIR, "index.json")

SIMILARITY_THRESHOLD = 0.78
TTL_SECONDS = 7 * 24 * 3600

# "why" is NOT a stopword - it marks a diagnostic question
STOPWORDS = {
    "the", "a", "an", "of", "in", "for", "to", "and", "or", "is", "was", "were",
    "what", "which", "how", "did", "do", "does", "me", "show", "give",
    "tell", "our", "my", "on", "by", "with", "that", "this", "it", "much",
    "many", "please", "can", "you", "i", "we",
}

# phrasings that mean the same thing analytically
SYNONYMS = {
    "why": "cause", "caused": "cause", "causes": "cause", "causing": "cause",
    "reason": "cause", "reasons": "cause", "driver": "cause",
    "drivers": "cause", "drove": "cause", "explain": "cause",

    "drop": "drop", "dropped": "drop", "decline": "drop", "declined": "drop",
    "fall": "drop", "fell": "drop", "decrease": "drop", "decreased": "drop",
    "down": "drop",

    "rise": "rise", "rose": "rise", "increase": "rise", "increased": "rise",
    "grew": "rise", "growth": "rise", "spike": "rise", "spiked": "rise",

    "sales": "revenue", "turnover": "revenue", "earnings": "revenue",
    "customers": "customer", "orders": "order", "products": "product",
    "regions": "region", "categories": "category", "sellers": "seller",
}


def normalize(q: str) -> str:
    return re.sub(r"\s+", " ", q.lower().strip().rstrip("?."))


def tokens(q: str) -> set[str]:
    words = re.findall(r"[a-z0-9_]+", normalize(q))
    out = set()
    for w in words:
        if w in STOPWORDS:
            continue
        out.add(SYNONYMS.get(w, w))
    return out


def similarity(a: str, b: str) -> float:
    """Jaccard overlap, with numbers and dates required to match exactly."""
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0

    # a question about Q3 2025 is NOT the same as one about Q2 2024
    nums_a = {t for t in ta if any(ch.isdigit() for ch in t)}
    nums_b = {t for t in tb if any(ch.isdigit() for ch in t)}
    if nums_a != nums_b:
        return 0.0

    return len(ta & tb) / len(ta | tb)


def schema_fingerprint(schema_text: str) -> str:
    return hashlib.sha256(schema_text.encode("utf-8")).hexdigest()[:16]


def _load() -> list[dict]:
    if not os.path.exists(INDEX):
        return []
    try:
        with open(INDEX, encoding="utf-8") as f:
            entries = json.load(f)
    except Exception:
        return []
    now = time.time()
    return [e for e in entries if now - e.get("ts", 0) < TTL_SECONDS]


def _save(entries: list[dict]) -> None:
    with open(INDEX, "w", encoding="utf-8") as f:
        json.dump(entries[-500:], f, indent=1)


def lookup(question: str, schema_fp: str) -> tuple[dict | None, str]:
    """Return (payload, how) where how is 'exact', 'fuzzy ...' or ''."""
    entries = [e for e in _load() if e["schema"] == schema_fp]
    n = normalize(question)

    for e in entries:
        if e["norm"] == n:
            return e["payload"], "exact"

    best, score = None, 0.0
    for e in entries:
        s = similarity(question, e["question"])
        if s > score:
            best, score = e, s
    if best and score >= SIMILARITY_THRESHOLD:
        return best["payload"], f"fuzzy ({score:.2f}) of: {best['question']}"
    return None, ""


def store(question: str, schema_fp: str, payload: dict) -> None:
    entries = _load()
    n = normalize(question)
    entries = [e for e in entries
               if not (e["schema"] == schema_fp and e["norm"] == n)]
    entries.append({
        "question": question,
        "norm": n,
        "schema": schema_fp,
        "ts": time.time(),
        "payload": payload,
    })
    _save(entries)


def clear() -> int:
    n = len(_load())
    if os.path.exists(INDEX):
        os.remove(INDEX)
    return n


def stats() -> dict:
    entries = _load()
    return {"entries": len(entries),
            "schemas": len({e["schema"] for e in entries})}