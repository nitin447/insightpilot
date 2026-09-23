"""
Warehouse layer: the agent's primitives - schema context for prompts, a
guarded SQL runner, and the ambiguity lookup that drives human-in-the-loop.

Works against whichever dataset is active (tools/dataset.py): the built-in
demo with its hand-written semantic layer, or a dataset the user uploaded.
"""
from __future__ import annotations
import os, re, sys, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import duckdb, pandas as pd, yaml

from tools import dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")

DEMO_TABLES = ["customers", "sellers", "products", "orders",
               "order_items", "payments", "reviews"]

FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|copy|export|"
    r"install|load|pragma|call)\b",
    re.IGNORECASE,
)

_SCHEMA_CACHE: dict[str, str] = {}


# ------------------------------------------------------------------ build
def build(force: bool = False) -> str:
    """Ensure the active dataset's DuckDB file exists. Only the demo can be
    rebuilt from CSVs; uploads are created by tools/ingest.py."""
    ds = dataset.get_active()
    db_path = ds["db_path"]

    if os.path.exists(db_path) and not force:
        return db_path

    if ds["kind"] != "demo":
        raise FileNotFoundError(
            f"Dataset '{ds['name']}' has no database at {db_path}. "
            f"Re-run ingestion for it.")

    if os.path.exists(db_path):
        os.remove(db_path)
    con = duckdb.connect(db_path)
    for t in DEMO_TABLES:
        csv = os.path.join(RAW, f"{t}.csv").replace("\\", "/")
        con.execute(f"CREATE TABLE {t} AS SELECT * FROM read_csv_auto('{csv}')")
    con.close()
    _SCHEMA_CACHE.clear()
    return db_path


def connect(read_only: bool = True):
    build()
    return duckdb.connect(dataset.get_active()["db_path"], read_only=read_only)


def tables() -> list[str]:
    con = connect()
    try:
        return [r[0] for r in con.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema='main' ORDER BY table_name").fetchall()]
    finally:
        con.close()


# --------------------------------------------------------------- metadata
def load_semantics() -> dict | None:
    p = dataset.get_active().get("semantic_path", "")
    if p and os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return yaml.safe_load(f)
    return None


def load_profile() -> dict | None:
    p = dataset.get_active().get("profile_path", "")
    if p and os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return None


def save_semantics(sem: dict) -> str:
    p = dataset.get_active()["semantic_path"]
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(sem, f, sort_keys=False, allow_unicode=True)
    _SCHEMA_CACHE.clear()
    return p


# ------------------------------------------------------------- ambiguity
# phrasings that mean the same thing when matching ambiguous terms
_TERM_SYNONYMS = {
    "how many": "count", "number of": "count", "total number": "count",
    "how much": "amount", "sales": "revenue", "turnover": "revenue",
}

# words too generic to identify a term on their own
_WEAK_WORDS = {"count", "amount", "total", "average", "number", "value",
               "time", "rate", "sum", "mean"}


def _normalize_for_match(text: str) -> set[str]:
    t = " " + text.lower().strip() + " "
    for phrase, canon in _TERM_SYNONYMS.items():
        t = t.replace(phrase, canon)
    out = set()
    for w in re.findall(r"[a-z]+", t):
        if len(w) < 4:
            continue
        out.add(w)
        if w.endswith("s"):          # crude singular
            out.add(w[:-1])
    return out


def ambiguous_terms() -> list[dict]:
    """Business words in this dataset that need the user to disambiguate."""
    sem = load_semantics() or {}
    conv = sem.get("conventions") or {}
    return [t for t in (conv.get("ambiguous_terms") or []) if t.get("term")]


def find_ambiguities(question: str) -> list[dict]:
    """Which ambiguous terms does this question actually touch?

    A term only matches if its DISTINCTIVE word appears - sharing a generic
    word like 'count' is not enough, or every counting question would
    trigger every counting ambiguity.
    """
    qwords = _normalize_for_match(question)
    hits = []
    for t in ambiguous_terms():
        twords = _normalize_for_match(str(t["term"]))
        if not twords:
            continue
        strong = twords - _WEAK_WORDS
        if strong and not (strong & qwords):
            continue                      # distinctive word absent
        if len(qwords & twords) / len(twords) >= 0.5:
            hits.append(t)
    return hits


# --------------------------------------------------------- schema context
def _context_from_semantics(sem: dict) -> str:
    con = connect()
    parts = ["# TABLES"]
    for t in tables():
        cols = con.execute(f'PRAGMA table_info("{t}")').fetchdf()
        n = con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
        desc = (sem.get("tables") or {}).get(t, {}) or {}
        coldesc = desc.get("columns", {}) or {}
        line = ", ".join(
            f"{r['name']} {r['type']}"
            + (f" -- {coldesc[r['name']]}" if r["name"] in coldesc else "")
            for _, r in cols.iterrows()
        )
        parts.append(f"{t} ({n:,} rows; {desc.get('grain','')})\n  {line}")
    con.close()

    if sem.get("joins"):
        parts.append("\n# JOINS\n" + "\n".join("  " + str(j)
                                               for j in sem["joins"]))

    if sem.get("metrics"):
        parts.append("\n# METRIC DEFINITIONS (use these exactly)")
        for name, m in sem["metrics"].items():
            line = f"  {name} = {m['definition']}"
            if m.get("filter"):
                line += f"   [filter: {m['filter']}]"
            if m.get("note"):
                line += f"\n      note: {m['note']}"
            parts.append(line)

    if sem.get("dimensions"):
        parts.append("\n# DIMENSIONS")
        for name, ref in sem["dimensions"].items():
            parts.append(f"  {name} = {ref}")

    conv = sem.get("conventions", {}) or {}
    if conv:
        parts.append("\n# CONVENTIONS\n  currency: "
                     + str(conv.get("currency", "unspecified"))
                     + "\n  time column: "
                     + str(conv.get("time_column", "unspecified")))
    return "\n".join(parts)


def _context_from_profile(prof: dict) -> str:
    """Fallback for an uploaded dataset with no semantic layer yet."""
    parts = ["# TABLES"]
    for t in prof.get("tables", []):
        cols = []
        for c in t["columns"]:
            bits = [f"{c['name']} {c['dtype']}"]
            if c.get("is_unique"):
                bits.append("KEY")
            if c.get("null_pct"):
                bits.append(f"nulls={c['null_pct']}%")
            if c.get("samples"):
                bits.append("e.g. " + ", ".join(str(s) for s in c["samples"][:3]))
            cols.append("  ".join(bits))
        parts.append(f"{t['name']} ({t['rows']:,} rows)\n  " + "\n  ".join(cols))

    if prof.get("joins"):
        parts.append("\n# LIKELY JOINS (detected, not confirmed)")
        for j in prof["joins"]:
            parts.append(f"  {j['left']} = {j['right']}  ({j['confidence']})")

    parts.append("\n# NOTE\n  No metric definitions exist for this dataset yet."
                 "\n  Derive metrics from the columns above and state your"
                 "\n  assumptions. Do not invent columns.")
    return "\n".join(parts)


def _context_from_catalog() -> str:
    """Last resort: raw catalog, no metadata at all."""
    con = connect()
    parts = ["# TABLES"]
    for t in tables():
        cols = con.execute(f'PRAGMA table_info("{t}")').fetchdf()
        n = con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
        line = ", ".join(f"{r['name']} {r['type']}" for _, r in cols.iterrows())
        parts.append(f"{t} ({n:,} rows)\n  {line}")
    con.close()
    return "\n".join(parts)


def schema_context(force: bool = False) -> str:
    """Compact schema for the LLM prompt. Cached - this is called on every
    LLM call, and rebuilding it hits the database every time."""
    ds = dataset.get_active()
    key = ds["db_path"]
    if not force and key in _SCHEMA_CACHE:
        return _SCHEMA_CACHE[key]

    sem = load_semantics()
    if sem:
        ctx = _context_from_semantics(sem)
    else:
        prof = load_profile()
        ctx = _context_from_profile(prof) if prof else _context_from_catalog()

    _SCHEMA_CACHE[key] = ctx
    return ctx


# --------------------------------------------------------------- querying
def run_sql(sql: str, limit: int = 500) -> pd.DataFrame:
    """Execute read-only SQL. Raises ValueError on write attempts."""
    if FORBIDDEN.search(sql):
        raise ValueError("Only read-only SELECT queries are allowed.")
    if not re.match(r"^\s*(select|with)\b", sql, re.IGNORECASE):
        raise ValueError("Query must start with SELECT or WITH.")
    con = connect()
    try:
        df = con.execute(sql).fetchdf()
    finally:
        con.close()
    return df.head(limit)


def preview(df: pd.DataFrame, max_rows: int = 20) -> str:
    """Render a result back to the LLM compactly.

    float_format matters: pandas defaults to scientific notation for large
    values, which the model then copies into its answer, and which breaks
    the numeric grounding check downstream.
    """
    if df is None or df.empty:
        return "(0 rows)"
    head = df.head(max_rows).to_string(
        index=False, float_format=lambda x: f"{x:,.2f}")
    extra = f"\n... {len(df) - max_rows} more rows" if len(df) > max_rows else ""
    return f"{len(df)} rows x {len(df.columns)} cols\n{head}{extra}"


if __name__ == "__main__":
    ds = dataset.get_active()
    print(f"ACTIVE DATASET: {ds['name']} ({ds['kind']})")
    print(f"  db: {ds['db_path']}\n")
    print(schema_context())

    amb = ambiguous_terms()
    if amb:
        print("\n# AMBIGUOUS TERMS (agent will ask before answering)")
        for a in amb:
            print(f"  '{a['term']}': {a['ask']}")