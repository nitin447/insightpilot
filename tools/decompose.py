"""
Deterministic decomposition for "why did <metric> change?" questions.

The model decides WHAT to measure - the metric as a per-row SQL expression,
its kind, the two periods, the scope. Code does everything that is
arithmetic:

  1. HEADLINE    the metric in each period
  2. BREAKDOWN   every categorical dimension - including ones reached through
                 a many-to-one join, like an account's region - with each
                 segment's contribution to the change, computed correctly
                 for the metric's kind
  3. CONTRAST    inside the segment that moved most versus everywhere else:
                 which other measure or category shifted there and nowhere
                 else - the candidate mechanism

Why code: left to the model, the breakdown step measured billing when the
question was about satisfaction, the contribution of an averaged metric was
read off an unweighted change, and a department carrying 88% of a drop was
reported as "uniform across all departments". None of those are judgement
calls. They are sums.

Contribution by kind:
  SUM / COUNT               segment change;  shares add to 100%
  AVERAGE / RATE / DURATION segment change x its share of the later
                            period's rows; the remainder is mix
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
MAX_CATEGORIES = 25
MAX_BREAKDOWN_ROWS = 80
MAX_CONTRAST_ROWS = 15
NEGLIGIBLE_PCT = 1.5    # a change smaller than this is flat: shares of it are noise
AVG_KINDS = ("AVERAGE", "RATE", "DURATION")

DECOMP_STEPS = (
    "HEADLINE - the metric in each of the two periods, computed in code",
    "BREAKDOWN - every dimension, each segment's contribution to the change, "
    "weighted correctly for the metric's kind, ranked by how concentrated "
    "the change is",
    "CONTRAST - inside the segment that moved most versus everywhere else: "
    "which other measures and categories shifted there and not elsewhere",
)

_WORD = re.compile(r"[A-Za-z_]\w*")


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _lit(v) -> str:
    return "'" + str(v).replace("'", "''") + "'"


def _columns(con, table: str) -> dict[str, str]:
    info = con.execute(f"PRAGMA table_info({_q(table)})").fetchdf()
    return {str(r["name"]): str(r["type"]) for _, r in info.iterrows()}


def _tables(con) -> list[str]:
    return [r[0] for r in con.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema='main' AND table_type='BASE TABLE'").fetchall()]


# ------------------------------------------------------------------ frame
class Frame:
    """The fact table, widened with every many-to-one lookup table.

    A joined column keeps its own name unless the fact table already has
    one like it, in which case it becomes table__column. That way a scope
    such as region = 'APAC' works whether region lives on the fact table or
    on the accounts table it points at."""

    def __init__(self, fact: str):
        self.fact = fact
        con = connect()
        try:
            self.types = {}
            fcols = _columns(con, fact)
            self.types.update(fcols)
            self.origin = {c: fact for c in fcols}
            selects = ["f.*"]
            joins = []
            for i, t in enumerate(t for t in _tables(con) if t != fact):
                tcols = _columns(con, t)
                for key in [k for k in tcols if k in fcols]:
                    n, d, nn = con.execute(
                        f"SELECT COUNT(*), COUNT(DISTINCT {_q(key)}), "
                        f"COUNT({_q(key)}) FROM {_q(t)}").fetchone()
                    if n and n == d == nn:              # key is unique there
                        alias = f"j{i}"
                        joins.append(f"LEFT JOIN {_q(t)} {alias} "
                                     f"ON f.{_q(key)} = {alias}.{_q(key)}")
                        for c, typ in tcols.items():
                            if c == key:
                                continue
                            name = c if c not in self.types else f"{t}__{c}"
                            if name in self.types:
                                continue
                            selects.append(f"{alias}.{_q(c)} AS {_q(name)}")
                            self.types[name] = typ
                            self.origin[name] = t
                        break
            self.sql = (f"SELECT {', '.join(selects)} FROM {_q(fact)} f "
                        + " ".join(joins))
        finally:
            con.close()

    def valid(self, where: str = "TRUE", select: str = "1") -> bool:
        con = connect()
        try:
            ok, _ = validate_sql(
                f"WITH w AS ({self.sql}) SELECT {select} FROM w WHERE {where}", con)
            return ok
        finally:
            con.close()


# -------------------------------------------------------------- building
def _metric_x(kind: str, row: str) -> tuple[str, str]:
    """(per-row value, aggregate) for the metric kind."""
    if kind == "COUNT":
        return "1", "SUM"
    if kind == "RATE":
        return f"CASE WHEN ({row}) THEN 100.0 ELSE 0.0 END", "AVG"
    if kind in ("AVERAGE", "DURATION"):
        return f"({row})", "AVG"
    return f"({row})", "SUM"


def _period_case(time_col: str, periods: list[dict]) -> str:
    whens = " ".join(
        f"WHEN {_q(time_col)} >= TIMESTAMP {_lit(p['start'])} AND "
        f"{_q(time_col)} < TIMESTAMP {_lit(p['end'])} THEN {i + 1}"
        for i, p in enumerate(periods))
    return f"CASE {whens} END"


def _base(frame: Frame, x: str, time_col: str, periods: list[dict],
          scope: str) -> str:
    where = f"({scope})" if scope else "TRUE"
    return (f"w AS ({frame.sql}),\n"
            f"base AS (\n  SELECT *, {_period_case(time_col, periods)} AS __pi, "
            f"{x} AS __x\n  FROM w WHERE {where}\n)")


def _dimensions(frame: Frame, exclude: set[str]) -> list[str]:
    vals = column_values()
    dims = []
    for name, typ in frame.types.items():
        if name in exclude:
            continue
        src = frame.origin.get(name, frame.fact)
        col = name.split("__", 1)[1] if "__" in name else name
        v = vals.get(f"{src}.{col}")
        if v and 2 <= len(v) <= MAX_CATEGORIES:
            dims.append(name)
    return dims


def _tautological(frame, x, time_col, periods, scope, dims) -> set[str]:
    """Dimensions where the whole metric sits in ONE value - e.g. only paid
    claims carry collected money, so claim_status = paid is always 100% of
    any revenue change. That is a definition, not a finding."""
    if not dims:
        return set()
    parts = ", ".join(
        f"COUNT(DISTINCT CASE WHEN __x <> 0 THEN CAST({_q(d)} AS VARCHAR) END) AS c{i}"
        for i, d in enumerate(dims))
    try:
        row = run_sql(f"WITH {_base(frame, x, time_col, periods, scope)}\n"
                      f"SELECT {parts} FROM base WHERE __pi IS NOT NULL").iloc[0]
    except Exception:
        return set()
    return {d for i, d in enumerate(dims) if int(row[f"c{i}"] or 0) <= 1}


def sql_headline(frame, x, agg, time_col, periods, scope) -> str:
    labels = " ".join(f"WHEN {i + 1} THEN {_lit(p['label'])}"
                      for i, p in enumerate(periods))
    return (f"WITH {_base(frame, x, time_col, periods, scope)}\n"
            f"SELECT CASE __pi {labels} END AS period,\n"
            f"  ROUND({agg}(__x), 2) AS value\n"
            f"FROM base WHERE __pi IS NOT NULL\nGROUP BY __pi ORDER BY __pi")


def sql_breakdown(frame, x, agg, kind, time_col, periods, scope, dims,
                  flat: bool = False) -> str:
    blocks = [
        f"SELECT {_lit(d)} AS dimension, "
        f"COALESCE(CAST({_q(d)} AS VARCHAR), '(missing)') AS segment, __pi,\n"
        f"    SUM(__x) AS sx, COUNT(__x) AS nx\n"
        f"  FROM base WHERE __pi IS NOT NULL GROUP BY 1, 2, 3"
        for d in dims]
    if agg == "AVG":
        size = "100.0 * n2 / NULLIF(cnt2, 0)"
        v1, v2 = "s1 / NULLIF(n1, 0)", "s2 / NULLIF(n2, 0)"
        total = "(tot2 / NULLIF(cnt2, 0) - tot1 / NULLIF(cnt1, 0))"
        contrib = f"(n2 * 1.0 / NULLIF(cnt2, 0)) * ({v2} - {v1})"
    else:
        size = "100.0 * COALESCE(s1, 0) / NULLIF(tot1, 0)"
        v1, v2 = "COALESCE(s1, 0)", "COALESCE(s2, 0)"
        total = "(COALESCE(tot2, 0) - COALESCE(tot1, 0))"
        contrib = f"({v2} - {v1})"
    return (
        f"WITH {_base(frame, x, time_col, periods, scope)},\n"
        f"g AS (\n  " + "\n  UNION ALL\n  ".join(blocks) + "\n),\n"
        f"s AS (SELECT dimension, segment,\n"
        f"    SUM(CASE WHEN __pi = 1 THEN sx END) s1, SUM(CASE WHEN __pi = 2 THEN sx END) s2,\n"
        f"    SUM(CASE WHEN __pi = 1 THEN nx END) n1, SUM(CASE WHEN __pi = 2 THEN nx END) n2\n"
        f"  FROM g GROUP BY 1, 2),\n"
        f"t AS (SELECT SUM(CASE WHEN __pi = 1 THEN __x END) tot1, SUM(CASE WHEN __pi = 2 THEN __x END) tot2,\n"
        f"    COUNT(CASE WHEN __pi = 1 THEN __x END) cnt1, COUNT(CASE WHEN __pi = 2 THEN __x END) cnt2\n"
        f"  FROM base),\n"
        f"r AS (SELECT dimension, segment,\n"
        f"    ROUND({v1}, 2) AS value_p1, ROUND({v2}, 2) AS value_p2,\n"
        f"    ROUND({v2} - {v1}, 2) AS change, n2 AS rows_p2,\n"
        f"    ROUND({contrib}, 4) AS contribution,\n"
        f"    ROUND(100.0 * {contrib} / NULLIF({total}, 0), 1) AS share_of_change_pct,\n"
        f"    ROUND({size}, 1) AS size_pct\n"
        f"  FROM s, t)\n"
        + (f"SELECT *, MAX(contribution) OVER (PARTITION BY dimension)\n"
           f"    - MIN(contribution) OVER (PARTITION BY dimension) AS concentration\n"
           f"FROM r\nORDER BY concentration DESC NULLS LAST, dimension, "
           f"contribution DESC NULLS LAST"
           if flat else
           # a segment holding 97% of the rows carries ~97% of any change
           # without being its cause - so rank by the share it carries
           # BEYOND its size
           f"SELECT *, MAX(share_of_change_pct - size_pct) OVER (PARTITION BY dimension)"
           f" AS concentration\n"
           f"FROM r\nORDER BY concentration DESC NULLS LAST, dimension, "
           f"share_of_change_pct - size_pct DESC NULLS LAST"))


def sql_contrast(frame, x, time_col, periods, scope, dim, segment,
                 dims, numerics) -> str:
    inside = f"(COALESCE(CAST({_q(dim)} AS VARCHAR), '(missing)') = {_lit(segment)})"
    num_blocks = [
        f"SELECT 'measure' AS kind, {_lit(c)} AS attribute,\n"
        f"    AVG(CASE WHEN __in AND __pi = 1 THEN {_q(c)} END) i1,\n"
        f"    AVG(CASE WHEN __in AND __pi = 2 THEN {_q(c)} END) i2,\n"
        f"    AVG(CASE WHEN NOT __in AND __pi = 1 THEN {_q(c)} END) o1,\n"
        f"    AVG(CASE WHEN NOT __in AND __pi = 2 THEN {_q(c)} END) o2,\n"
        f"    STDDEV_POP({_q(c)}) sd\n  FROM b2"
        for c in numerics]
    cat_blocks = []
    for d in dims:
        if d == dim:
            continue
        cat_blocks.append(
            f"SELECT 'share of rows' AS kind, {_lit(d)} || ' = ' || v AS attribute,\n"
            f"    100.0 * AVG(CASE WHEN __in AND __pi = 1 THEN hit END) i1,\n"
            f"    100.0 * AVG(CASE WHEN __in AND __pi = 2 THEN hit END) i2,\n"
            f"    100.0 * AVG(CASE WHEN NOT __in AND __pi = 1 THEN hit END) o1,\n"
            f"    100.0 * AVG(CASE WHEN NOT __in AND __pi = 2 THEN hit END) o2,\n"
            f"    100.0 * SQRT(AVG(hit) * (1 - AVG(hit))) sd\n"
            f"  FROM (SELECT b2.*, v, CASE WHEN CAST({_q(d)} AS VARCHAR) = v THEN 1.0 ELSE 0.0 END hit\n"
            f"        FROM b2, (SELECT DISTINCT CAST({_q(d)} AS VARCHAR) v FROM b2 "
            f"WHERE {_q(d)} IS NOT NULL) vals)\n"
            f"  GROUP BY v")
    return (
        f"WITH {_base(frame, x, time_col, periods, scope)},\n"
        f"b2 AS (SELECT *, {inside} AS __in FROM base WHERE __pi IS NOT NULL),\n"
        f"c AS (\n  " + "\n  UNION ALL\n  ".join(num_blocks + cat_blocks) + "\n)\n"
        f"SELECT kind, attribute,\n"
        f"  ROUND(i1, 2) AS inside_p1, ROUND(i2, 2) AS inside_p2,\n"
        f"  ROUND(o1, 2) AS outside_p1, ROUND(o2, 2) AS outside_p2,\n"
        f"  ROUND((i2 - i1) - (o2 - o1), 2) AS shift_vs_outside,\n"
        f"  ROUND(ABS((i2 - i1) - (o2 - o1)) / NULLIF(sd, 0), 2) AS strength\n"
        f"FROM c WHERE i1 IS NOT NULL AND i2 IS NOT NULL\n"
        f"ORDER BY strength DESC NULLS LAST\nLIMIT {MAX_CONTRAST_ROWS}")


# ------------------------------------------------------------------ main
def _result(step, sql, df, rows, note="") -> dict:
    table = preview(df, rows) + (("\n" + note) if note else "")
    return {"step": step, "sql": sql, "ok": True, "attempts": 1, "errors": [],
            "validator_catches": 0, "empty": df.empty, "warnings": [],
            "broken": False, "full": True, "table": table, "computed": True}


def resolve(metric_row: str, kind: str, time_column: str, periods: list[dict],
            scope: str) -> dict | None:
    """Check everything the model supplied before any of it runs."""
    if kind not in ("SUM", "COUNT", "AVERAGE", "RATE", "DURATION"):
        return None
    if len(periods) != 2 or not time_column:
        return None
    table, _, tcol = time_column.rpartition(".")
    if not table or not metric_row and kind != "COUNT":
        return None
    if ";" in (metric_row or "") or ";" in (scope or ""):
        return None
    con = connect()
    try:
        if table not in _tables(con):
            return None
    finally:
        con.close()
    frame = Frame(table)
    if tcol not in frame.types:
        return None
    x, agg = _metric_x(kind, metric_row or "1")
    if not frame.valid(select=f"{agg}({x})"):
        return None
    if scope and not frame.valid(where=scope):
        scope = ""
    return {"frame": frame, "x": x, "agg": agg, "kind": kind, "time_col": tcol,
            "periods": periods, "scope": scope, "row": metric_row or ""}


def decompose(metric_row: str, kind: str, time_column: str,
              periods: list[dict], scope: str = "") -> tuple[list[dict], dict] | None:
    """(results, summary) for the three computed steps, or None when the
    inputs cannot be resolved - the caller then falls back to the model's
    own plan."""
    r = resolve(metric_row, kind, time_column, periods, scope)
    if not r:
        return None
    f, x, agg, tc, per, sc = (r["frame"], r["x"], r["agg"], r["time_col"],
                              r["periods"], r["scope"])

    h_sql = sql_headline(f, x, agg, tc, per, sc)
    head = run_sql(h_sql)
    if len(head) < 2:
        return None
    counts = [int(v) for v in run_sql(
        f"WITH {_base(f, x, tc, per, sc)}\nSELECT COUNT(__x) AS n FROM base "
        f"WHERE __pi IS NOT NULL GROUP BY __pi ORDER BY __pi")["n"]]

    used = set(_WORD.findall(r["row"])) | {tc}
    scope_cols = set(_WORD.findall(sc))
    dims = _dimensions(f, exclude={tc} | scope_cols)
    tauto = _tautological(f, x, tc, per, sc, dims) if agg == "SUM" else set()
    dims = [d for d in dims if d not in tauto]

    v1, v2 = float(head.iloc[0]["value"]), float(head.iloc[1]["value"])
    total = v2 - v1
    flat = abs(total) <= abs(v1) * NEGLIGIBLE_PCT / 100.0

    b_sql = sql_breakdown(f, x, agg, kind, tc, per, sc, dims, flat)
    br = run_sql(b_sql, limit=100_000)
    # the segment that carries the change: largest share in the direction of
    # the change, in the most concentrated dimension
    top = br.dropna(subset=["share_of_change_pct"]).copy()
    top["__excess"] = top["share_of_change_pct"] - top["size_pct"].fillna(0)
    top = top.sort_values(["concentration", "__excess"], ascending=False)
    summary = {"total_change": round(total, 4), "kind": kind, "flat": flat,
               "top_dimension": "", "top_segment": "", "share_of_change_pct": 0.0,
               "size_pct": 0.0}
    if flat and not br.empty:
        # nothing to explain in the total, but the segment that moved most
        # still has a story - orthopaedics' stays lengthened while the
        # hospital's did not
        first = br.iloc[0]["dimension"]
        seg = br[br["dimension"] == first].dropna(subset=["contribution"])
        if not seg.empty:
            best = seg.loc[seg["contribution"].abs().idxmax()]
            summary.update(top_dimension=str(best["dimension"]),
                           top_segment=str(best["segment"]),
                           share_of_change_pct=0.0)
    if not top.empty and total != 0 and not flat:
        best = top.iloc[0]
        summary.update(top_dimension=str(best["dimension"]),
                       top_segment=str(best["segment"]),
                       share_of_change_pct=float(best["share_of_change_pct"]),
                       size_pct=float(best["size_pct"] or 0))

    results = [
        # the row counts go in the note, not the table: a second numeric
        # column in a two-row table is what the UI charts, and a line of row
        # counts rising under a question about a fall is worse than no chart
        _result(DECOMP_STEPS[0], h_sql, head, 5,
                f"(scope: {sc or 'all rows'}; metric kind {kind}; rows measured: "
                + ", ".join(f"{p['label']} {n:,}" for p, n in zip(per, counts))
                + ")"),
        _result(DECOMP_STEPS[1], b_sql, br.drop(columns=["concentration"]).head(
            MAX_BREAKDOWN_ROWS), MAX_BREAKDOWN_ROWS,
            f"(every dimension tested; contribution for a {kind} metric is "
            + ("the segment's own change" if agg == "SUM" else
               "the segment's change weighted by its share of the later "
               "period's rows; what is left over is mix")
            + ("; the TOTAL BARELY MOVED, so shares of it are meaningless: "
               "dimensions are ranked by how far their segments moved in "
               "OPPOSITE directions and cancelled out)" if flat else
               "; share_of_change_pct is contribution / total change, so a "
               "segment above 100% moved more than the whole while others "
               "moved the other way; size_pct is the segment's share of the "
               "rows or of the metric, and dimensions are ranked by the share "
               "of change a segment carries BEYOND its size)")
            + (("\nleft out, because the whole metric sits in one of their "
                "values by definition: " + ", ".join(sorted(tauto))) if tauto else "")),
    ]

    if summary["top_segment"]:
        numerics = [c for c, t in f.types.items()
                    if any(k in t.upper() for k in NUMERIC)
                    and c not in used and not c.endswith("_id")]
        # For a RATE the condition's own column IS the metric - "denied share
        # rose" restates "the denial rate rose". For a SUM the same kind of
        # column is the mechanism (failed payments), so it stays.
        cdims = dims + sorted(tauto)
        if kind == "RATE":
            cdims = [d for d in cdims if d not in used]
            # ...and so is any measure the condition fixes: a denied claim is
            # paid zero, so amount_paid 'shifts' wherever denials do
            from tools.predictors import _determined
            con = connect()
            try:
                numerics = [c for c in numerics if f.origin.get(c) != f.fact
                            or not _determined(con, f.fact, _q(c),
                                               f"({r['row']})", "TRUE")]
            finally:
                con.close()
        c_sql = sql_contrast(f, x, tc, per, sc, summary["top_dimension"],
                             summary["top_segment"], cdims, numerics)
        try:
            ct = run_sql(c_sql, limit=100_000)
            results.append(_result(
                DECOMP_STEPS[2], c_sql, ct, MAX_CONTRAST_ROWS,
                f"(inside = {summary['top_dimension']} = {summary['top_segment']}; "
                f"shift_vs_outside is how much more it moved there than "
                f"everywhere else; strength is that shift in standard "
                f"deviations, so measures and shares can be ranked together)"))
        except Exception:
            pass
    return results, summary