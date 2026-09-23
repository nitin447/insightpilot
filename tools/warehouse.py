"""
Warehouse layer: the agent's primitives - schema context for prompts, a
guarded SQL runner, and the ambiguity lookup that drives human-in-the-loop.

Works against whichever dataset is active (tools/dataset.py): the built-in
demo with its hand-written semantic layer, or a dataset the user uploaded.

The schema context carries two things beyond names and types, both of which
exist because their absence produced confident wrong answers:

  COLUMN VALUES  - the complete value list of every low-cardinality text
                   column. Without it a model filters on words it got from
                   its own prompt vocabulary: `claim_status IN ('failed',
                   'unpaid')` matched 0 of 12,000 rows on a dataset whose
                   statuses are paid/pending/denied, and the empty result
                   was then reported as a finding.
  NUMERIC RANGES - min/max/mean per numeric column, so a metric claiming to
                   be a length of stay of 7.13 can be checked against a
                   column that runs 1.03 to 86.85 hours.
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

# Views are not tables. Without the table_type filter a view created for any
# reason appears in every prompt and duplicates all of its columns.
_BASE_TABLES = ("SELECT table_name FROM information_schema.tables "
                "WHERE table_schema='main' AND table_type='BASE TABLE' "
                "ORDER BY table_name")

TEXT_TYPES = ("VARCHAR", "CHAR", "TEXT", "STRING", "BOOLEAN")
NUMERIC_TYPES = ("TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT",
                 "FLOAT", "DOUBLE", "DECIMAL", "REAL")

MAX_DISTINCT = 25        # a column with more values than this is not a category
MAX_VALUE_COLS = 80      # ceiling on how many columns we interrogate
MAX_SHOWN = 12           # values printed before truncating the list

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
        return [r[0] for r in con.execute(_BASE_TABLES).fetchall()]
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


# ------------------------------------------------------- column facts
def _fmt_value(v) -> str:
    if v is True:
        return "true"
    if v is False:
        return "false"
    return str(v)


def column_values(max_distinct: int = MAX_DISTINCT,
                  max_cols: int = MAX_VALUE_COLS) -> dict[str, list[str]]:
    """{"table.column": [every value]} for low-cardinality text/boolean columns.

    This is the authoritative list of what a filter may compare against. It
    is deterministic on purpose: telling a model 'use a status value that
    actually occurs' does not work if the prompt never shows it which ones do.
    """
    out: dict[str, list[str]] = {}
    con = connect()
    try:
        examined = 0
        for t in [r[0] for r in con.execute(_BASE_TABLES).fetchall()]:
            info = con.execute(f'PRAGMA table_info("{t}")').fetchdf()
            for _, r in info.iterrows():
                if examined >= max_cols:
                    return out
                if not any(k in str(r["type"]).upper() for k in TEXT_TYPES):
                    continue
                examined += 1
                try:
                    rows = con.execute(
                        f'SELECT DISTINCT "{r["name"]}" FROM "{t}" '
                        f'WHERE "{r["name"]}" IS NOT NULL '
                        f'LIMIT {max_distinct + 1}').fetchall()
                except Exception:
                    continue
                if 0 < len(rows) <= max_distinct:
                    out[f'{t}.{r["name"]}'] = sorted(_fmt_value(x[0]) for x in rows)
    finally:
        con.close()
    return out


def numeric_ranges(max_cols: int = MAX_VALUE_COLS
                   ) -> dict[str, tuple[float, float, float]]:
    """{"table.column": (min, max, mean)} for every numeric column."""
    out: dict[str, tuple[float, float, float]] = {}
    con = connect()
    try:
        examined = 0
        for t in [r[0] for r in con.execute(_BASE_TABLES).fetchall()]:
            info = con.execute(f'PRAGMA table_info("{t}")').fetchdf()
            for _, r in info.iterrows():
                if examined >= max_cols:
                    return out
                if not any(k in str(r["type"]).upper() for k in NUMERIC_TYPES):
                    continue
                examined += 1
                try:
                    mn, mx, av = con.execute(
                        f'SELECT MIN("{r["name"]}"), MAX("{r["name"]}"), '
                        f'AVG("{r["name"]}") FROM "{t}"').fetchone()
                except Exception:
                    continue
                if mn is None or av is None:
                    continue
                out[f'{t}.{r["name"]}'] = (float(mn), float(mx), float(av))
    finally:
        con.close()
    return out


DATE_TYPES = ("TIMESTAMP", "DATE")
COMPLETE_SLACK_DAYS = 7     # a quarter whose last week is missing is still complete


def _quarter_start(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(ts.year, 3 * ((ts.month - 1) // 3) + 1, 1)


def date_ranges(max_cols: int = 20) -> dict[str, dict]:
    """{"table.column": {min, max, latest, previous}} for every date column.

    `latest` and `previous` are the last two COMPLETE quarters, as
    (label, start, end) with end exclusive. They exist because the model,
    told nothing about the data's dates, fell back on CURRENT_DATE - which
    returns the right quarters only on the day the dataset happens to end -
    or on no date filter at all, which returned every quarter, got
    truncated, and cut off exactly the two quarters the question was about.
    """
    out: dict[str, dict] = {}
    con = connect()
    try:
        examined = 0
        for t in [r[0] for r in con.execute(_BASE_TABLES).fetchall()]:
            info = con.execute(f'PRAGMA table_info("{t}")').fetchdf()
            for _, r in info.iterrows():
                if examined >= max_cols:
                    return out
                if not any(k in str(r["type"]).upper() for k in DATE_TYPES):
                    continue
                examined += 1
                try:
                    mn, mx = con.execute(
                        f'SELECT MIN("{r["name"]}"), MAX("{r["name"]}") '
                        f'FROM "{t}"').fetchone()
                except Exception:
                    continue
                if mn is None or mx is None:
                    continue
                mn, mx = pd.Timestamp(mn), pd.Timestamp(mx)
                q = _quarter_start(mx)
                q_end = q + pd.DateOffset(months=3)
                # Monthly data is dated the 1st of each month: an invoice for
                # June is stamped 1 June. Read by the day, that quarter looks
                # four weeks short and gets dropped - so the agent compares
                # Q4 with Q1 on data that runs to the end of Q2.
                monthly = con.execute(
                    f'SELECT COUNT(DISTINCT EXTRACT(day FROM "{r["name"]}")) '
                    f'FROM "{t}"').fetchone()[0] == 1
                done = (q_end - pd.DateOffset(months=1) if monthly
                        else q_end - pd.Timedelta(days=COMPLETE_SLACK_DAYS))
                if mx < done:
                    q_end, q = q, q - pd.DateOffset(months=3)   # last one is partial
                prev = q - pd.DateOffset(months=3)
                if prev < _quarter_start(mn):
                    prev = None

                def lab(d):
                    return f"Q{(d.month - 1) // 3 + 1} {d.year}"

                out[f'{t}.{r["name"]}'] = {
                    "min": mn.strftime("%Y-%m-%d"), "max": mx.strftime("%Y-%m-%d"),
                    "latest": (lab(q), q.strftime("%Y-%m-%d"),
                               q_end.strftime("%Y-%m-%d")),
                    "previous": None if prev is None else
                                (lab(prev), prev.strftime("%Y-%m-%d"),
                                 q.strftime("%Y-%m-%d")),
                }
    finally:
        con.close()
    return out


def _facts_section() -> str:
    parts: list[str] = []

    vals = column_values()
    if vals:
        parts.append("\n# COLUMN VALUES - the COMPLETE set for these columns."
                     "\n# A filter may only compare against a value listed here."
                     "\n# If a word you want is absent, this data does not "
                     "record it under that name.")
        for key, vs in vals.items():
            shown = vs[:MAX_SHOWN]
            line = " | ".join(shown)
            if len(vs) > MAX_SHOWN:
                line += f" | ... ({len(vs)} total)"
            parts.append(f"  {key} = {line}")

    rng = numeric_ranges()
    if rng:
        parts.append("\n# NUMERIC RANGES - min .. max (mean). Use these to "
                     "sanity-check units:\n# a figure far outside a column's "
                     "range is measuring something else.")
        for key, (mn, mx, av) in rng.items():
            parts.append(f"  {key}  {mn:,.2f} .. {mx:,.2f}  (mean {av:,.2f})")

    dr = date_ranges()
    if dr:
        parts.append("\n# DATE RANGES - what the data actually covers. Anchor every"
                     "\n# window to THESE dates. Never use CURRENT_DATE, NOW() or"
                     "\n# today: the data does not end today.")
        for key, d in dr.items():
            line = f"  {key}  {d['min']} .. {d['max']}"
            lq = d["latest"]
            line += f"  | latest complete quarter {lq[0]} = [{lq[1]}, {lq[2]})"
            if d["previous"]:
                pq = d["previous"]
                line += f", previous {pq[0]} = [{pq[1]}, {pq[2]})"
            parts.append(line)

    return "\n".join(parts)


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

    # Appended in every mode. A semantic layer describes what metrics MEAN;
    # it says nothing about which values a status column actually holds, and
    # that is where the filters go wrong.
    facts = _facts_section()
    if facts:
        ctx = ctx + "\n" + facts

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