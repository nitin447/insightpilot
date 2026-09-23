"""
Text-to-SQL with pre-execution validation and escalating self-healing.

Generated SQL is bound by DuckDB's EXPLAIN before it runs, so an invented
column becomes a cheap, precise correction instead of a runtime failure.

Retries escalate to a stronger model: asking the same weak model to fix its
own mistake tends to reproduce the mistake.
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
- When comparing two periods across segments, compute the change as a
  column and ORDER BY that change ASC (largest decline first), then
  LIMIT 15. Only the first rows are read downstream, so the biggest
  movers must appear at the top.
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
                 prior_error: str = "", role: str = "sql") -> str:
    if prior_error:
        prompt = f"""{schema}

This query was REJECTED:
{prior_sql}

Reason:
{prior_error}

Fix it. Answer this question: {question}
Return only the corrected SQL."""
    else:
        prompt = f"{schema}\n\nWrite one DuckDB SQL query to answer:\n{question}"
    return extract_sql(chat(prompt, system=SYSTEM, role=role))


def ask_sql(question: str, max_attempts: int = 3, verbose: bool = True) -> SQLResult:
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