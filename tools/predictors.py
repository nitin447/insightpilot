"""
Deterministic predictor sweep for "what predicts <event>?" questions.

Why this is code and not an LLM-written query: the question needs every
attribute tested, every numeric measure bucketed, durations computed from
pairs of timestamps, and the strongest attributes crossed with each other -
all in one pass, over all of the data. Asked to write that as one query, a
model dropped something different on every run: it filtered to one quarter
(1,839 of 12,000 rows) without being asked, never computed length of stay
(the real predictor), crossed the wrong pair of attributes, and produced a
table long enough to be truncated before the combinations were shown.

Testing every column is not judgement, it is arithmetic. So the SQL is
generated here, from the catalog, and the model's job shrinks to explaining a
complete, ranked table - which it does well.

Every result carries the SQL that produced it, so the UI can still show and
re-run each step. Rates are always percentages, and a bucket needs a minimum
number of rows before its rate can count as a finding.
"""
from __future__ import annotations
import os, re, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from tools.warehouse import connect, run_sql, preview, column_values
from tools.validator import validate_sql

NUMERIC = ("TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT",
           "FLOAT", "DOUBLE", "DECIMAL", "REAL")
DATES = ("TIMESTAMP", "DATE")
MAX_CATEGORIES = 25       # more distinct values than this is not a category
MIN_ROWS_FLOOR = 30       # a bucket smaller than this is noise
MIN_ROWS_SHARE = 0.01     # ... or smaller than 1% of the rows
TOP_FOR_COMBOS = 3        # strongest attributes crossed with each other
MAX_SWEEP_ROWS = 80       # rows of the ranked table shown to the model

SWEEP_STEPS = (
    "BASE RATE - how often the event happens across every row in scope",
    "ATTRIBUTE SWEEP - the event rate for every value of every attribute, "
    "every numeric measure split into quartiles, and every duration between "
    "two timestamps, ranked by how far apart the rates sit",
    "COMBINATIONS - the strongest attributes crossed with each other, "
    "because a real driver is often visible only in combination",
)

_IDENT = re.compile(r"^[A-Za-z_]\w*$")


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _lit(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


# ----------------------------------------------------------------- setup
def resolve_event(event: dict | None) -> tuple[str, str, str] | None:
    """(table, column, SQL boolean expression) for the event, or None.

    The event must be a real column. A boolean column is the event itself;
    any other column needs a value, and that value must actually occur in it.
    """
    if not isinstance(event, dict):
        return None
    ref = str(event.get("column", "")).strip()
    value = str(event.get("value", "") or "").strip()
    table, _, col = ref.rpartition(".")
    if not col:
        return None

    con = connect()
    try:
        tables = [r[0] for r in con.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema='main' AND table_type='BASE TABLE'").fetchall()]
        if not table:
            owners = [t for t in tables if col in _columns(con, t)]
            if len(owners) != 1:
                return None
            table = owners[0]
        if table not in tables:
            return None
        types = _columns(con, table)
    finally:
        con.close()
    if col not in types:
        return None

    if "BOOL" in types[col].upper() and not value:
        return table, col, f"COALESCE({_q(col)}, FALSE)"
    if not value:
        return None
    allowed = column_values().get(f"{table}.{col}", [])
    if value not in allowed:
        return None
    return table, col, f"({_q(col)} = {_lit(value)})"


def _columns(con, table: str) -> dict[str, str]:
    info = con.execute(f"PRAGMA table_info({_q(table)})").fetchdf()
    return {str(r["name"]): str(r["type"]) for _, r in info.iterrows()}


def _scope_ok(table: str, scope: str) -> bool:
    if not scope:
        return True
    con = connect()
    try:
        ok, _ = validate_sql(f"SELECT 1 FROM {_q(table)} WHERE {scope}", con)
        return ok
    finally:
        con.close()


# -------------------------------------------------------------- features
def _determined(con, table: str, expr: str, event_sql: str,
                scope_sql: str) -> bool:
    """True if this column is FIXED by the event rather than predicting it.

    A denied claim is paid zero by definition, so amount_paid 'predicts'
    denial perfectly - and means nothing. Test: on the event rows one value
    (zero, NULL, a single status) covers nearly all of them, while on the
    other rows it does not."""
    try:
        rows = con.execute(
            f"WITH x AS (SELECT CASE WHEN {event_sql} THEN 1 ELSE 0 END e, "
            f"{expr} v FROM {_q(table)} WHERE {scope_sql}), "
            f"c AS (SELECT e, v, COUNT(*) n FROM x GROUP BY e, v) "
            f"SELECT e, MAX(n) * 1.0 / SUM(n) FROM c GROUP BY e").fetchall()
    except Exception:
        return False
    share = {int(e): float(v) for e, v in rows}
    return share.get(1, 0) >= 0.95 and share.get(0, 1) < 0.80


def _features(table: str, event_col: str, scope_sql: str,
              event_sql: str = "TRUE") -> list[dict]:
    """Every testable attribute: {name, expr, kind, partition}.

    kind 'cat'      - the column's own values
    kind 'quartile' - a numeric measure split into four equal-sized buckets;
                      quartiles, not a median split, because a signal living
                      in one tail (the shortest stays) is diluted to nothing
                      when half the rows are averaged together
    partition       - for a duration, quartiles WITHIN the group that sets
                      its norm: a four-day stay is short for oncology and
                      long for paediatrics
    """
    con = connect()
    try:
        types = _columns(con, table)
        cats = {k.split(".", 1)[1]: v for k, v in column_values().items()
                if k.split(".", 1)[0] == table}
        feats: list[dict] = []

        for col, vals in cats.items():
            if col == event_col or not (2 <= len(vals) <= MAX_CATEGORIES):
                continue
            feats.append({"name": col, "expr": f"CAST({_q(col)} AS VARCHAR)",
                          "kind": "cat", "partition": ""})

        for col, typ in types.items():
            if col == event_col or not any(k in typ.upper() for k in NUMERIC):
                continue
            if col in cats:
                continue
            n_distinct = con.execute(
                f"SELECT COUNT(DISTINCT {_q(col)}) FROM {_q(table)}").fetchone()[0]
            if n_distinct < 8:
                continue
            feats.append({"name": col, "expr": _q(col),
                          "kind": "quartile", "partition": ""})

        # durations between every ordered pair of timestamps whose gap is
        # positive - the measures nobody stored, like length of stay
        dcols = [c for c, t in types.items() if any(k in t.upper() for k in DATES)]
        for a in dcols:
            for b in dcols:
                if a == b:
                    continue
                expr = f"DATE_DIFF('hour', {_q(a)}, {_q(b)}) / 24.0"
                med, cover = con.execute(
                    f"SELECT MEDIAN({expr}), AVG(CASE WHEN {expr} IS NULL THEN 0 ELSE 1 END) "
                    f"FROM {_q(table)} WHERE {scope_sql}").fetchone()
                if med is None or med <= 0 or (cover or 0) < 0.5:
                    continue
                name = f"days from {a} to {b}"
                feats.append({"name": name, "expr": expr,
                              "kind": "quartile", "partition": ""})
                group = _norm_group(con, table, expr, list(cats), scope_sql)
                if group:
                    feats.append({"name": f"{name}, vs its {group} norm",
                                  "expr": expr, "kind": "quartile",
                                  "partition": group})
        kept = []
        for f in feats:
            if _determined(con, table, f["expr"], event_sql, scope_sql):
                EXCLUDED.append(f["name"])
            else:
                kept.append(f)
        return kept
    finally:
        con.close()


EXCLUDED: list[str] = []


def _norm_group(con, table: str, expr: str, cats: list[str],
                scope_sql: str) -> str:
    """The category that best explains a measure's level (largest eta
    squared), if it explains a meaningful share. That category is the norm
    the measure should be judged against."""
    best, best_eta = "", 0.0
    total = con.execute(
        f"SELECT VAR_POP({expr}) FROM {_q(table)} WHERE {scope_sql}").fetchone()[0]
    if not total:
        return ""
    for c in cats:
        try:
            between = con.execute(
                f"WITH g AS (SELECT {_q(c)} k, AVG({expr}) m, COUNT({expr}) n "
                f"FROM {_q(table)} WHERE {scope_sql} GROUP BY 1), "
                f"o AS (SELECT AVG({expr}) m FROM {_q(table)} WHERE {scope_sql}) "
                f"SELECT SUM(g.n * POWER(g.m - o.m, 2)) / SUM(g.n) FROM g, o"
            ).fetchone()[0]
        except Exception:
            continue
        eta = (between or 0) / total
        if eta > best_eta:
            best, best_eta = c, eta
    return best if best_eta >= 0.05 else ""


def _bucket_sql(f: dict) -> str:
    """A SQL expression labelling each row with this feature's bucket."""
    if f["kind"] == "cat":
        return f["expr"]
    part = f"PARTITION BY {_q(f['partition'])}, " if f["partition"] else "PARTITION BY "
    # rowid breaks ties: rows with identical values would otherwise land in
    # quartiles arbitrarily, and the same question would return 15.58% on
    # one run and 15.52% on the next.
    tile = (f"NTILE(4) OVER ({part}({f['expr']}) IS NULL "
            f"ORDER BY {f['expr']}, rowid)")
    return (f"CASE WHEN {f['expr']} IS NULL THEN NULL ELSE "
            f"CASE {tile} WHEN 1 THEN 'Q1 lowest 25%' WHEN 2 THEN 'Q2' "
            f"WHEN 3 THEN 'Q3' ELSE 'Q4 highest 25%' END END")


# ------------------------------------------------------------------ SQL
def _labelled(table: str, event_sql: str, scope_sql: str,
              feats: list[dict]) -> str:
    cols = ",\n    ".join(f"{_bucket_sql(f)} AS f{i}" for i, f in enumerate(feats))
    return (f"labelled AS (\n  SELECT CASE WHEN {event_sql} THEN 1 ELSE 0 END AS ev,\n"
            f"    {cols}\n  FROM {_q(table)}\n  WHERE {scope_sql}\n)")


def sql_base(table: str, event_sql: str, scope_sql: str) -> str:
    return (f"SELECT COUNT(*) AS rows_in_scope,\n"
            f"  SUM(CASE WHEN {event_sql} THEN 1 ELSE 0 END) AS events,\n"
            f"  ROUND(100.0 * AVG(CASE WHEN {event_sql} THEN 1 ELSE 0 END), 2) "
            f"AS base_rate_pct\nFROM {_q(table)}\nWHERE {scope_sql}")


def sql_sweep(table: str, event_sql: str, scope_sql: str,
              feats: list[dict], min_n: int) -> str:
    blocks = [
        f"SELECT {_lit(f['name'])} AS attribute, f{i} AS bucket, COUNT(*) AS n,\n"
        f"    ROUND(100.0 * AVG(ev), 2) AS rate_pct\n"
        f"  FROM labelled WHERE f{i} IS NOT NULL GROUP BY f{i} HAVING COUNT(*) >= {min_n}"
        for i, f in enumerate(feats)]
    return (f"WITH {_labelled(table, event_sql, scope_sql, feats)},\n"
            f"cells AS (\n  " + "\n  UNION ALL\n  ".join(blocks) + "\n)\n"
            f"SELECT attribute, bucket, n, rate_pct,\n"
            f"  ROUND(MAX(rate_pct) OVER (PARTITION BY attribute)\n"
            f"      - MIN(rate_pct) OVER (PARTITION BY attribute), 2) AS spread_pts\n"
            f"FROM cells\nORDER BY spread_pts DESC, attribute, rate_pct DESC")


def sql_combos(table: str, event_sql: str, scope_sql: str,
               feats: list[dict], min_n: int) -> str:
    idx = range(len(feats))
    pairs = [(i, j) for i in idx for j in idx if i < j]
    blocks = [
        f"SELECT {_lit(feats[i]['name'] + '  x  ' + feats[j]['name'])} AS combination,\n"
        f"    f{i} || '  +  ' || f{j} AS cell, COUNT(*) AS n,\n"
        f"    ROUND(100.0 * AVG(ev), 2) AS rate_pct\n"
        f"  FROM labelled WHERE f{i} IS NOT NULL AND f{j} IS NOT NULL\n"
        f"  GROUP BY 2 HAVING COUNT(*) >= {min_n}"
        for i, j in pairs]
    return (f"WITH {_labelled(table, event_sql, scope_sql, feats)}\n"
            + "\nUNION ALL\n".join(blocks)
            + "\nORDER BY rate_pct DESC")


# ------------------------------------------------------------------ main
def _result(step: str, sql: str, df: pd.DataFrame, max_rows: int,
            note: str = "") -> dict:
    table = preview(df, max_rows)
    if note:
        table += "\n" + note
    return {"step": step, "sql": sql, "ok": True, "attempts": 1, "errors": [],
            "validator_catches": 0, "empty": df.empty, "warnings": [],
            "broken": False, "full": True, "table": table, "computed": True}


def flag_sweep(event: dict, scope: str = "") -> list[dict] | None:
    """The three results of a flag-event predictive investigation, or None
    if the event cannot be resolved - the caller then falls back to the
    model-written plan."""
    resolved = resolve_event(event)
    if not resolved:
        return None
    table, event_col, event_sql = resolved
    scope = scope if _scope_ok(table, scope) else ""
    scope_sql = f"({scope})" if scope else "TRUE"

    base = run_sql(sql_base(table, event_sql, scope_sql))
    rows = int(base.iloc[0]["rows_in_scope"] or 0)
    if rows == 0:
        return None
    min_n = max(MIN_ROWS_FLOOR, int(rows * MIN_ROWS_SHARE))

    EXCLUDED.clear()
    feats = _features(table, event_col, scope_sql, event_sql)
    if not feats:
        return None

    s_sql = sql_sweep(table, event_sql, scope_sql, feats, min_n)
    sweep = run_sql(s_sql, limit=100_000)

    # the strongest attributes, one per underlying column, go into the combos
    order = (sweep.groupby("attribute")["spread_pts"].max()
             .sort_values(ascending=False).index.tolist())
    picked, seen = [], set()
    for name in order:
        f = next(x for x in feats if x["name"] == name)
        root = f["expr"]
        if root in seen:
            continue
        seen.add(root)
        picked.append(f)
        if len(picked) == TOP_FOR_COMBOS:
            break
    c_sql = sql_combos(table, event_sql, scope_sql, picked, min_n)
    combos = run_sql(c_sql, limit=100_000)

    shown = sweep.head(MAX_SWEEP_ROWS)
    flat = sorted(set(sweep["attribute"]) - set(shown["attribute"]))
    note = (f"(complete: every attribute tested; each bucket has at least "
            f"{min_n} rows; ranked by spread_pts, the gap between an "
            f"attribute's highest and lowest rate)")
    if flat:
        note += "\nnot shown, spreads smaller than every row above: " + ", ".join(flat)
    if EXCLUDED:
        note += ("\nexcluded, because the event itself fixes their value (they "
                 "record the outcome, they do not predict it): "
                 + ", ".join(sorted(set(EXCLUDED))))
    base_note = (f"(scope: {scope or 'all rows'}; no date filter - every row in "
                 f"scope is included)")
    return [
        _result(SWEEP_STEPS[0], sql_base(table, event_sql, scope_sql), base, 5,
                base_note),
        _result(SWEEP_STEPS[1], s_sql, shown, MAX_SWEEP_ROWS, note),
        _result(SWEEP_STEPS[2], c_sql, combos.head(20), 20,
                f"(the {len(picked)} strongest attributes crossed pairwise; "
                f"each cell has at least {min_n} rows)"),
    ]