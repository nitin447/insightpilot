"""
Infer a semantic layer for an uploaded dataset.

The LLM drafts business definitions from the column profile; every metric
is then VALIDATED against the real catalog, and anything referencing a
column that does not exist is dropped rather than shipped.

The draft is a proposal, not a fact. It is written to the dataset's
semantic.yml for the user to review and correct before it is trusted.
"""
from __future__ import annotations
import os, sys, json, re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from agent.llm import chat
from tools import dataset
from tools.warehouse import connect, load_profile, save_semantics, schema_context
from tools.validator import get_catalog, validate_sql

SYSTEM = """You are a senior analytics engineer defining a semantic layer.

You are given a profile of a database: tables, columns, types, null rates,
distinct counts, sample values, and detected joins.

Produce business definitions a non-technical user would expect. Be
conservative: it is better to define five metrics you are sure of than
fifteen you guessed at.

Rules:
- Use ONLY table and column names that appear in the profile. Never invent.
- Every metric definition must be a valid SQL expression referencing real
  columns, qualified as table.column.
- If a column looks like a status/state field, check its sample values: if
  some indicate a cancelled/failed/void record, EXCLUDE those in the filter
  of any revenue or count metric, and say so in the note.
- grain: one short sentence describing what one row of the table means.
- ambiguous_terms: business words a user might say that could map to more
  than one definition, with the question to ask them.
- Do not define a metric that needs a column the data does not have.

Return ONLY YAML, no prose, no code fences, in exactly this shape:

tables:
  <table_name>:
    grain: <one sentence>
    columns:
      <column_name>: <short description>
joins:
  - <left_table.col> = <right_table.col>
metrics:
  <metric_name>:
    definition: <SQL expression>
    filter: <SQL boolean, or omit>
    note: <one sentence, or omit>
dimensions:
  <dimension_name>: <table.column>
conventions:
  currency: <currency code or 'unknown'>
  time_column: <table.column used for time series>
  ambiguous_terms:
    - term: <word>
      ask: <question to put to the user>
"""


def _profile_text(prof: dict) -> str:
    out = []
    for t in prof.get("tables", []):
        out.append(f"TABLE {t['name']} ({t['rows']:,} rows)")
        for c in t["columns"]:
            bits = [f"  {c['name']}  {c['dtype']}"]
            if c.get("is_unique"):
                bits.append("KEY")
            bits.append(f"nulls={c['null_pct']}%")
            bits.append(f"distinct={c['distinct']:,}")
            if c.get("min") is not None:
                bits.append(f"range=[{c['min']} .. {c['max']}]")
            if c.get("samples"):
                bits.append("values: " + ", ".join(str(s) for s in c["samples"][:5]))
            out.append("  ".join(bits))
        out.append("")
    if prof.get("joins"):
        out.append("DETECTED JOINS")
        for j in prof["joins"]:
            out.append(f"  {j['left']} = {j['right']}  ({j['confidence']})")
    return "\n".join(out)


def _strip_fences(text: str) -> str:
    t = text.strip()
    if "```" in t:
        blocks = t.split("```")
        for b in blocks:
            b = b.strip()
            if b.lower().startswith("yaml"):
                return b[4:].strip()
        if len(blocks) > 1:
            return blocks[1].strip()
    return t


def validate_semantics(sem: dict) -> tuple[dict, list[str]]:
    """Drop anything that references a column the database does not have."""
    con = connect()
    try:
        catalog = get_catalog(con)
        real_tables = set(catalog)
        real_cols = {f"{t}.{c}" for t, cols in catalog.items() for c in cols}
        problems = []

        # tables / columns
        for tname in list(sem.get("tables", {})):
            if tname not in real_tables:
                problems.append(f"dropped unknown table '{tname}'")
                sem["tables"].pop(tname)
                continue
            cols = sem["tables"][tname].get("columns", {}) or {}
            for cname in list(cols):
                if cname not in catalog[tname]:
                    problems.append(f"dropped unknown column "
                                    f"'{tname}.{cname}'")
                    cols.pop(cname)

        # joins
        kept = []
        for j in sem.get("joins", []) or []:
            refs = re.findall(r"([A-Za-z_]\w*\.[A-Za-z_]\w*)", str(j))
            if refs and all(r in real_cols for r in refs):
                kept.append(j)
            else:
                problems.append(f"dropped invalid join '{j}'")
        sem["joins"] = kept

        # metrics - test each one actually runs
        good = {}
        for name, m in (sem.get("metrics") or {}).items():
            expr = str(m.get("definition", "")).strip()
            if not expr:
                continue
            probe_from = " , ".join(f'"{t}"' for t in real_tables)
            probe = f"SELECT {expr} FROM {probe_from} LIMIT 0"
            ok, err = validate_sql(probe, con)
            if ok:
                good[name] = m
            else:
                problems.append(f"dropped metric '{name}': {err[:80]}")
        sem["metrics"] = good

        # dimensions
        dims = {}
        for name, ref in (sem.get("dimensions") or {}).items():
            if str(ref) in real_cols:
                dims[name] = ref
            else:
                problems.append(f"dropped dimension '{name}' -> {ref}")
        sem["dimensions"] = dims

        return sem, problems
    finally:
        con.close()


def infer(write: bool = True, verbose: bool = True) -> dict:
    ds = dataset.get_active()
    prof = load_profile()
    if not prof:
        raise RuntimeError(
            f"Dataset '{ds['name']}' has no profile. Run tools/ingest.py first.")

    if verbose:
        print(f"Inferring semantic layer for '{ds['name']}'...")

    raw = chat(_profile_text(prof), system=SYSTEM, role="critic")
    try:
        sem = yaml.safe_load(_strip_fences(raw))
        if not isinstance(sem, dict):
            raise ValueError("model did not return a mapping")
    except Exception as e:
        raise RuntimeError(f"Could not parse inferred semantics: {e}\n\n{raw[:600]}")

    sem, problems = validate_semantics(sem)

    if verbose:
        n_m = len(sem.get("metrics", {}))
        n_d = len(sem.get("dimensions", {}))
        print(f"  {n_m} metrics, {n_d} dimensions, "
              f"{len(sem.get('joins', []))} joins kept")
        for p in problems:
            print(f"  ! {p}")

    if write:
        path = save_semantics(sem)
        if verbose:
            print(f"\nWritten to {path}")
            print("REVIEW THIS FILE before trusting the numbers - these are "
                  "the model's guesses at your business definitions.")
    return sem


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true",
                    help="print the resulting schema context")
    args = ap.parse_args()

    sem = infer()

    print("\n--- METRICS DEFINED ---")
    for name, m in (sem.get("metrics") or {}).items():
        line = f"  {name} = {m['definition']}"
        if m.get("filter"):
            line += f"  [filter: {m['filter']}]"
        print(line)
        if m.get("note"):
            print(f"      {m['note']}")

    amb = (sem.get("conventions") or {}).get("ambiguous_terms") or []
    if amb:
        print("\n--- NEEDS YOUR CONFIRMATION ---")
        for a in amb:
            print(f"  '{a.get('term')}': {a.get('ask')}")

    if args.show:
        print("\n--- SCHEMA CONTEXT THE AGENT WILL SEE ---")
        print(schema_context(force=True))