"""
Text-to-SQL with pre-execution validation and escalating self-healing.

Generated SQL is bound by DuckDB's EXPLAIN before it runs, so an invented
column becomes a cheap, precise correction instead of a runtime failure.

Retries escalate to a stronger model: asking the same weak model to fix its
own mistake tends to reproduce the mistake.

`notes` carries the planner's concept resolution - the expression a derived
metric must use, and which named concepts are absent from the schema. Without
it, a metric that has to be computed gets replaced by whichever column has a
similar name, and every check downstream passes.
"""
from __future__ import annotations
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass, field
import pandas as pd

from agent.llm import chat, extract_sql
from tools.warehouse import schema_context, run_sql, connect
from tools.validator import validate_sql, get_catalog, schema_brief

SYSTEM = """You are a senior analytics engineer writing DuckDB SQL.

Rules:
- Output ONLY a SQL query. No prose, no explanation.
- Use the METRIC DEFINITIONS exactly as given. Never invent your own.
- Metric names (revenue, aov, orders_count, ...) are DEFINITIONS, not
  columns. Expand them into the SQL expression shown. Never write
  `SELECT aov` or `GROUP BY revenue`.
- Use ONLY tables and columns that appear in the schema. Never invent a
  column name, and never assume a column exists because it would be useful.
- A filter may only compare against a value in the COLUMN VALUES list. If
  the word you want is not there, this data does not record it under that
  name - do not guess a synonym. A status filter that matches no rows
  returns an empty result, and an empty result reads downstream as a
  finding rather than as a broken query.
- If the step names a time period, the query MUST filter to it. A step that
  says "for Q1 2026 versus Q2 2026" and returns all history answers a
  different question, and nothing downstream can tell.
- If RESOLVED CONCEPTS gives an expression for something the step names,
  use that expression verbatim. It exists because the quantity is not a
  column, and substituting a similarly-named column answers a different
  question with entirely valid SQL.

Look for the column before you derive a proxy:
- Before computing a concept indirectly, search the schema for a column that
  already records it. A status column, or a date column that stays NULL
  until the event happens, IS the measurement. Counting rows that vanish
  between periods, or rows whose value hits zero, is a guess at the same
  thing and is usually wrong.
- A rate or share is COUNT(the thing) / COUNT(what it is a share of). If the
  expression you are writing can come out negative, it is not a rate - you
  are measuring net change, which moves the opposite way when the base is
  growing. Go back and find the column that marks the event.
- To measure money LOST to failed, unpaid, refunded, denied or cancelled
  rows, sum the GROSS or list amount for those rows, never the net or
  collected amount. The collected column is zero on exactly those rows by
  definition, so summing it always returns 0 - which looks like a finding
  that they cost nothing, and is the opposite of what happened.
- A duration is DATE_DIFF('hour'|'day'|..., start, end) between two
  timestamp columns, divided into the unit you want if DATE_DIFF's own unit
  isn't fine-grained enough - e.g. DATE_DIFF('hour', a, b) / 24.0 for days
  with fractional precision. DATE_DIFF returns a plain number in the unit
  you name.
  NEVER subtract two TIMESTAMPs directly with `end - start`. In DuckDB that
  produces an INTERVAL, not a number of seconds, and dividing an INTERVAL
  by a number FAILS with "No function matches /(INTERVAL, DECIMAL...)".
  This is the single most common mistake in this schema - always reach for
  DATE_DIFF, never for subtraction, whenever a duration is needed.
  Exclude rows where either end is NULL: an unfinished event is not a
  zero-length one.

A rate or average BROKEN DOWN BY a bucket must use GROUP BY that bucket -
never a window function whose OVER() clause is empty or lacks PARTITION BY.
An empty-OVER window aggregates across the ENTIRE result set and returns
the identical number copied into every row, which reads as "no variation"
when the real per-group numbers were never computed. If you are computing
several buckets with UNION ALL, each block is its own independent
SELECT ... GROUP BY - do not window across the union of them.

When a step asks for the rate of an event for a COMBINATION of two
conditions (e.g. "emergency admissions with a short stay"), build the
bucket with a CASE expression that tests both conditions in the WHERE or
the CASE, and GROUP BY that derived bucket. Do not test the two conditions
in separate queries and expect them combined later - the combination must
be computed as one bucket in the SQL itself.

Row limits - read this carefully, it is where these queries go wrong:
- If the step asks for a TOTAL or a headline figure, do NOT group by any
  dimension and do NOT apply a LIMIT. Return the whole figure.
- If the step asks for a breakdown by one dimension, return EVERY value of
  that dimension. Do not LIMIT it. A region or category list is short, and
  a total computed downstream from a truncated list is wrong.
- Apply a LIMIT only when the grouping is genuinely high-cardinality
  (customers, products, orders). Then compute the change as a column, ORDER
  BY that change ASC so the largest declines come first, and LIMIT 20.
  Be aware this deliberately discards the largest INCREASES, so such a
  result can never be summed into a total.
- Always alias aggregates with clear names.
- Use DATE_TRUNC('month'|'quarter', ...) for time grouping.
- Round money to 2 decimals.
- Never write INSERT/UPDATE/DELETE/DROP/CREATE.
"""


@dataclass
class SQLResult:
    question: str
    sql: str
    df: pd.DataFrame | None
    ok: bool
    attempts: int
    errors: list[str] = field(default_factory=list)
    caught_by_validator: int = 0


def generate_sql(question: str, schema: str, prior_sql: str = "",
                 prior_error: str = "", role: str = "sql",
                 notes: str = "") -> str:
    head = f"{schema}\n\n{notes}" if notes else schema
    if prior_error:
        prompt = f"""{head}

This query was REJECTED:
{prior_sql}

Reason:
{prior_error}

Fix it. Answer this question: {question}
Return only the corrected SQL."""
    else:
        prompt = f"{head}\n\nWrite one DuckDB SQL query to answer:\n{question}"
    return extract_sql(chat(prompt, system=SYSTEM, role=role))


def ask_sql(question: str, max_attempts: int = 3, verbose: bool = True,
            notes: str = "") -> SQLResult:
    """Generate SQL, validate it, run it, and self-correct on failure."""
    schema = schema_context()
    con = connect()
    sql, errors, caught = "", [], 0

    try:
        for attempt in range(1, max_attempts + 1):
            sql = generate_sql(
                question, schema,
                prior_sql=sql if errors else "",
                prior_error=errors[-1] if errors else "",
                role="sql" if attempt == 1 else "sql_hard",
                notes=notes,
            )
            if verbose:
                print(f"\n--- attempt {attempt} ---\n{sql}")

            ok, err = validate_sql(sql, con)
            if not ok:
                caught += 1
                errors.append(f"{err}\n\nActual schema:\n"
                              f"{schema_brief(get_catalog(con))}")
                if verbose:
                    print(f"  REJECTED (validator): {err}")
                continue

            try:
                df = run_sql(sql)
                return SQLResult(question, sql, df, True, attempt, errors, caught)
            except Exception as e:
                msg = str(e).split("\n")[0]
                errors.append(msg)
                if verbose:
                    print(f"  FAILED (runtime): {msg}")
    finally:
        con.close()

    return SQLResult(question, sql, None, False, max_attempts, errors, caught)


if __name__ == "__main__":
    from tools.warehouse import preview

    for q in [
        "What was total revenue by quarter?",
        "Which product category had the biggest revenue drop in Q3 2025 versus Q2 2025?",
        "What is the average review score for orders delivered late versus on time?",
    ]:
        print("\n" + "=" * 70)
        print("Q:", q)
        r = ask_sql(q)
        if r.ok:
            print(f"\nOK in {r.attempts} attempt(s), "
                  f"{r.caught_by_validator} caught pre-execution")
            print(preview(r.df, 10))
        else:
            print("FAILED after", r.attempts, "attempts:", r.errors)