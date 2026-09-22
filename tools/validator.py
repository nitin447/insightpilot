"""
Validate generated SQL before executing it.

DuckDB's EXPLAIN binds a query - resolving every table and column - without
running it. That turns a runtime failure into a cheap pre-flight check, and
lets us hand the model a precise, actionable correction.
"""
from __future__ import annotations
import re, difflib
import duckdb

FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|copy|export|"
    r"install|load|pragma|call)\b",
    re.IGNORECASE,
)

# identifiers quoted in DuckDB binder errors, e.g. Referenced column "foo"
_QUOTED = re.compile(r'"([^"]+)"')


def get_catalog(con: duckdb.DuckDBPyConnection) -> dict[str, list[str]]:
    """{table_name: [column, ...]} for everything in the database."""
    catalog = {}
    tables = [r[0] for r in con.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'main'"
    ).fetchall()]
    for t in tables:
        cols = [r[0] for r in con.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = ?", [t]
        ).fetchall()]
        catalog[t] = cols
    return catalog


def _suggest(bad: str, catalog: dict[str, list[str]], kind: str) -> str:
    """Nearest real table/column name, for the correction hint."""
    pool = list(catalog) if kind == "table" else \
        sorted({c for cols in catalog.values() for c in cols})
    close = difflib.get_close_matches(bad, pool, n=3, cutoff=0.5)
    if not close:
        return ""
    if kind == "column":
        located = []
        for c in close:
            owners = [t for t, cols in catalog.items() if c in cols]
            located.append(f"{c} (in {', '.join(owners[:2])})")
        return "  Did you mean: " + "; ".join(located)
    return "  Did you mean: " + ", ".join(close)


def validate_sql(sql: str, con: duckdb.DuckDBPyConnection) -> tuple[bool, str]:
    """
    Returns (ok, error_message). A False result carries a message written
    for the model to act on, not for a human to read.
    """
    if not sql or not sql.strip():
        return False, "Empty query."

    if FORBIDDEN.search(sql):
        return False, ("Only read-only SELECT queries are allowed. "
                       "Remove any INSERT/UPDATE/DELETE/DROP/CREATE.")

    if not re.match(r"^\s*(select|with)\b", sql.strip(), re.IGNORECASE):
        return False, "Query must start with SELECT or WITH."

    if sql.count("'") % 2 or sql.count('"') % 2:
        return False, "Unbalanced quotes in the query."

    try:
        con.execute(f"EXPLAIN {sql}")
        return True, ""
    except Exception as e:
        msg = str(e).split("\n")[0].strip()
        catalog = get_catalog(con)
        low = msg.lower()

        names = _QUOTED.findall(msg)
        bad = names[0] if names else ""

        if not bad:
            m = (re.search(r"with name\s+([A-Za-z_][\w.]*)", msg, re.I)
                 or re.search(r"(?:column|table|relation)\s+(?!with\b)"
                              r"([A-Za-z_][\w.]*)", msg, re.I))
            bad = m.group(1) if m else ""
        bad = bad.split(".")[-1]

        hint = ""
        if bad:
            if "table" in low or "relation" in low or "catalog" in low:
                hint = _suggest(bad, catalog, "table")
            elif "column" in low or "binder" in low:
                hint = _suggest(bad, catalog, "column")

        return False, f"{msg}{hint}"


def schema_brief(catalog: dict[str, list[str]]) -> str:
    """One-line-per-table schema, for re-grounding a failed query."""
    return "\n".join(f"  {t}({', '.join(cols)})" for t, cols in catalog.items())