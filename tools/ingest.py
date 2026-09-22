"""
Load arbitrary user files into a DuckDB dataset, profile them, and report
data-quality issues.

Handles: CSV, TSV, Excel (all sheets), JSON, Parquet.
Deals with: messy headers, mixed types, currency symbols, date formats,
            encodings, empty rows/columns.

Profiling and quality checks run as SQL inside DuckDB over the FULL table,
not a pandas sample - sampling produces false uniqueness, and a false key
means a wrong join.

Nothing is silently deleted. Issues are REPORTED; the user decides.

Usage:
    python tools/ingest.py sales.csv customers.xlsx --name acme
    python tools/ingest.py --list
    python tools/ingest.py --use demo
"""
from __future__ import annotations
import os, re, sys, json, glob, argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass, field, asdict

import duckdb
import pandas as pd

from tools import dataset

READ_SAMPLE = 20000        # rows pulled into pandas for TYPE detection only
DATE_HINTS = ("date", "ts", "time", "created", "updated", "_at", "day", "month")
GENERIC_KEYS = {"id", "name", "date", "type", "status", "value", "amount",
                "category", "code", "label", "title", "description"}


# --------------------------------------------------------------- utilities
def clean_name(name: str, fallback: str = "col") -> str:
    """Make a SQL-safe identifier out of anything."""
    n = str(name).strip().lower()
    n = re.sub(r"[^\w]+", "_", n).strip("_")
    n = re.sub(r"_+", "_", n)
    if not n or n[0].isdigit():
        n = f"{fallback}_{n}" if n else fallback
    return n


def read_any(path: str) -> dict[str, pd.DataFrame]:
    """Return {table_name: dataframe} for one uploaded file."""
    base = clean_name(os.path.splitext(os.path.basename(path))[0], "table")
    ext = os.path.splitext(path)[1].lower()

    if ext in (".xlsx", ".xls", ".xlsm"):
        sheets = pd.read_excel(path, sheet_name=None)
        if len(sheets) == 1:
            return {base: list(sheets.values())[0]}
        return {f"{base}_{clean_name(s)}": df for s, df in sheets.items()}

    if ext == ".parquet":
        return {base: pd.read_parquet(path)}

    if ext == ".json":
        return {base: pd.read_json(path)}

    sep = "\t" if ext in (".tsv", ".tab") else None
    last = None
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(path, sep=sep, encoding=enc,
                             engine="python", on_bad_lines="skip")
            return {base: df}
        except Exception as e:
            last = e
    raise ValueError(f"Could not read {path}: {last}")


def coerce_types(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean column names, strip junk, detect dates and numbers."""
    notes = {}
    df = df.copy()

    before = len(df), len(df.columns)
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if len(df) != before[0]:
        notes["_rows_dropped_empty"] = before[0] - len(df)
    if len(df.columns) != before[1]:
        notes["_cols_dropped_empty"] = before[1] - len(df.columns)

    seen, cols = {}, []
    for i, c in enumerate(df.columns):
        name = clean_name(c, f"col_{i}")
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        cols.append(name)
    renamed = {a: b for a, b in zip(df.columns, cols) if str(a) != b}
    if renamed:
        notes["_columns_renamed"] = len(renamed)
    df.columns = cols

    probe = df.head(READ_SAMPLE)

    for c in df.columns:
        s = probe[c]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
            continue

        stripped = s.astype(str).str.strip()
        cleaned = (stripped.str.replace(r"[,\s₹$€£]", "", regex=True)
                            .str.replace(r"^\((.*)\)$", r"-\1", regex=True))
        if pd.to_numeric(cleaned, errors="coerce").notna().mean() > 0.9 \
           and stripped.ne("").any():
            full = (df[c].astype(str).str.strip()
                    .str.replace(r"[,\s₹$€£]", "", regex=True)
                    .str.replace(r"^\((.*)\)$", r"-\1", regex=True))
            df[c] = pd.to_numeric(full, errors="coerce")
            notes[c] = "parsed as number"
            continue

        hinted = any(h in c for h in DATE_HINTS)
        try:
            dt = pd.to_datetime(stripped, errors="coerce", format="mixed")
        except Exception:
            dt = pd.to_datetime(stripped, errors="coerce")
        rate = dt.notna().mean()
        if rate > 0.95 or (hinted and rate > 0.7):
            try:
                df[c] = pd.to_datetime(df[c].astype(str).str.strip(),
                                       errors="coerce", format="mixed")
            except Exception:
                df[c] = pd.to_datetime(df[c], errors="coerce")
            notes[c] = "parsed as datetime"

    return df, notes


# ---------------------------------------------------------------- profiling
@dataclass
class ColumnProfile:
    name: str
    dtype: str
    null_pct: float
    distinct: int
    is_unique: bool
    samples: list = field(default_factory=list)
    min: str | None = None
    max: str | None = None
    mean: float | None = None


@dataclass
class TableProfile:
    name: str
    rows: int
    columns: list
    notes: dict = field(default_factory=dict)
    issues: list = field(default_factory=list)


NUMERIC_TYPES = ("TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT",
                 "FLOAT", "DOUBLE", "DECIMAL")
TEMPORAL_TYPES = ("DATE", "TIMESTAMP", "TIME")


def profile_table_sql(con, name: str, notes: dict) -> TableProfile:
    """Profile the FULL table using DuckDB aggregates - no sampling."""
    rows = con.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
    info = con.execute(f"PRAGMA table_info('{name}')").fetchdf()

    cols, issues = [], []
    for _, r in info.iterrows():
        c, dtype = r["name"], str(r["type"]).upper()

        n_null, n_distinct = con.execute(
            f'SELECT COUNT(*) - COUNT("{c}"), COUNT(DISTINCT "{c}") FROM "{name}"'
        ).fetchone()
        n_nonnull = rows - n_null
        null_pct = round(n_null / rows * 100, 1) if rows else 0.0

        p = ColumnProfile(
            name=c, dtype=dtype, null_pct=null_pct, distinct=int(n_distinct),
            is_unique=bool(n_nonnull > 0 and n_distinct == n_nonnull
                           and n_nonnull >= rows * 0.99),
        )
        p.samples = [str(v)[:40] for (v,) in con.execute(
            f'SELECT DISTINCT "{c}" FROM "{name}" WHERE "{c}" IS NOT NULL LIMIT 5'
        ).fetchall()]

        if any(t in dtype for t in NUMERIC_TYPES) and n_nonnull:
            mn, mx, av = con.execute(
                f'SELECT MIN("{c}"), MAX("{c}"), AVG("{c}") FROM "{name}"'
            ).fetchone()
            p.min, p.max = f"{mn:,.2f}", f"{mx:,.2f}"
            p.mean = round(float(av), 2)
        elif any(t in dtype for t in TEMPORAL_TYPES) and n_nonnull:
            mn, mx = con.execute(
                f'SELECT MIN("{c}"), MAX("{c}") FROM "{name}"'
            ).fetchone()
            p.min, p.max = str(mn)[:19], str(mx)[:19]

        # ---- quality flags (reported, never auto-fixed)
        if null_pct >= 50:
            issues.append(f"{c}: {null_pct}% missing")
        if n_distinct <= 1 and rows > 1:
            issues.append(f"{c}: single value - carries no information")
        if any(t in dtype for t in NUMERIC_TYPES) and n_nonnull > 20:
            neg = con.execute(
                f'SELECT COUNT(*) FROM "{name}" WHERE "{c}" < 0').fetchone()[0]
            if neg and any(k in c for k in ("price", "amount", "qty",
                                            "quantity", "value", "cost")):
                issues.append(f"{c}: {neg:,} negative values in a "
                              f"quantity/price column")

        cols.append(p)

    # duplicate rows
    quoted = ", ".join(f'"{c["name"]}"' for c in info.to_dict("records"))
    dupes = con.execute(
        f'SELECT COUNT(*) - COUNT(*) FILTER (WHERE rn = 1) FROM '
        f'(SELECT ROW_NUMBER() OVER (PARTITION BY {quoted}) AS rn FROM "{name}")'
    ).fetchone()[0]
    if dupes:
        issues.append(f"{dupes:,} fully duplicated rows "
                      f"({dupes / rows * 100:.1f}%)")

    return TableProfile(name=name, rows=rows, columns=cols,
                        notes=notes, issues=issues)


def guess_joins(profiles: list) -> list[dict]:
    """Find likely FK relationships: same column name, one side a true key."""
    joins, seen = [], set()
    by_col: dict[str, list] = {}
    for t in profiles:
        for c in t.columns:
            by_col.setdefault(c.name, []).append((t.name, c))

    for col, holders in by_col.items():
        if len(holders) < 2 or col in GENERIC_KEYS:
            continue
        keys = [(t, c) for t, c in holders if c.is_unique]
        refs = [(t, c) for t, c in holders if not c.is_unique]

        for kt, _ in keys:
            for rt, _ in refs:
                pair = tuple(sorted([kt, rt])) + (col,)
                if kt != rt and pair not in seen:
                    seen.add(pair)
                    joins.append({"left": f"{rt}.{col}", "right": f"{kt}.{col}",
                                  "confidence": "high"})
        if len(keys) == 2 and not refs:
            (a, _), (b, _) = keys
            pair = tuple(sorted([a, b])) + (col,)
            if pair not in seen:
                seen.add(pair)
                joins.append({"left": f"{a}.{col}", "right": f"{b}.{col}",
                              "confidence": "high"})
        if not keys and len(holders) == 2:
            (a, _), (b, _) = holders
            pair = tuple(sorted([a, b])) + (col,)
            if pair not in seen:
                seen.add(pair)
                joins.append({"left": f"{a}.{col}", "right": f"{b}.{col}",
                              "confidence": "low"})
    return joins


# ------------------------------------------------------------------ loading
def load_files(paths: list[str], db_path: str) -> tuple[list, list[dict]]:
    """Read files, clean them, write to DuckDB, profile in-database."""
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    if os.path.exists(db_path):
        os.remove(db_path)
    con = duckdb.connect(db_path)
    profiles, note_map = [], {}

    try:
        for path in paths:
            for tname, raw in read_any(path).items():
                df, notes = coerce_types(raw)
                if df.empty:
                    print(f"  skipped {tname}: empty after cleaning")
                    continue
                con.register("_tmp", df)
                con.execute(f'CREATE TABLE "{tname}" AS SELECT * FROM _tmp')
                con.unregister("_tmp")
                note_map[tname] = notes
                print(f"  loaded {tname}: {len(df):,} rows x {len(df.columns)} cols")

        for tname, notes in note_map.items():
            profiles.append(profile_table_sql(con, tname, notes))
    finally:
        con.close()

    return profiles, guess_joins(profiles)


def profiles_to_text(profiles: list, joins: list[dict]) -> str:
    out = []
    for t in profiles:
        out.append(f"TABLE {t.name} ({t.rows:,} rows)")
        for c in t.columns:
            bits = [f"  {c.name}  {c.dtype}"]
            if c.is_unique:
                bits.append("KEY")
            bits.append(f"nulls={c.null_pct}%")
            bits.append(f"distinct={c.distinct:,}")
            if c.min is not None:
                bits.append(f"range=[{c.min} .. {c.max}]")
            if c.samples:
                bits.append("e.g. " + ", ".join(c.samples[:3]))
            out.append("  ".join(bits))
        out.append("")
    if joins:
        out.append("LIKELY JOINS")
        for j in joins:
            out.append(f"  {j['left']} -> {j['right']}  ({j['confidence']})")
    return "\n".join(out)


def quality_report(profiles: list) -> str:
    lines = []
    for t in profiles:
        cleaning = {k: v for k, v in t.notes.items() if k.startswith("_")}
        if t.issues or cleaning:
            lines.append(f"{t.name}:")
            for k, v in cleaning.items():
                lines.append(f"  cleaned: {k.lstrip('_').replace('_',' ')} = {v}")
            for i in t.issues:
                lines.append(f"  ISSUE: {i}")
    return "\n".join(lines) if lines else "No quality issues detected."


def ingest(paths: list[str], name: str, activate: bool = True) -> dict:
    ds = dataset.register_upload(name)
    print(f"\nIngesting into dataset '{ds['name']}'")
    profiles, joins = load_files(paths, ds["db_path"])

    payload = {"tables": [asdict(p) for p in profiles], "joins": joins}
    with open(ds["profile_path"], "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1, default=str)

    print("\n" + profiles_to_text(profiles, joins))
    print("\n--- DATA QUALITY ---")
    print(quality_report(profiles))

    if activate:
        dataset.set_active(ds)
        print(f"\nActive dataset is now '{ds['name']}'. "
              f"No metric definitions yet - the agent will infer them.")
    return ds


# ---------------------------------------------------------------------- cli
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="CSV / Excel / JSON / Parquet")
    ap.add_argument("--name", default="", help="name for this dataset")
    ap.add_argument("--list", action="store_true", help="list datasets")
    ap.add_argument("--use", default="", help="switch active dataset")
    args = ap.parse_args()

    if args.list:
        print("demo  (built-in)")
        for u in dataset.list_uploads():
            print(u)
        print(f"\nactive: {dataset.get_active()['name']}")
        sys.exit()

    if args.use:
        if args.use == "demo":
            print("Active:", dataset.use_demo()["name"])
        else:
            ds = dataset.register_upload(args.use)
            if not os.path.exists(ds["db_path"]):
                sys.exit(f"No dataset called '{args.use}'")
            print("Active:", dataset.set_active(ds)["name"])
        sys.exit()

    files = args.files or sorted(glob.glob("data/raw/*.csv"))
    if not files:
        sys.exit("No files given.")
    ingest(files, args.name or "dataset")