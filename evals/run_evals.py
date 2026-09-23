"""
Benchmark the agent against a single-shot SQL baseline.

--runs N repeats the benchmark and reports the spread plus per-question
pass rates. A single run of a multi-step LLM agent is a sample of size one;
the spread is part of the result, not noise to hide. For reproducible
numbers, pin a model: LLM_PIN_MODEL=groq:qwen/qwen3.8-27b

Ambiguity prompts are disabled (ask=False): a benchmark has no human
present, and every run must apply the same definitions.
"""
from __future__ import annotations
import sys, os, re, time, argparse, statistics
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml, pandas as pd
from tools.warehouse import run_sql
from tools import dataset
from agent.graph import investigate
from agent.sql_tool import ask_sql
from agent.llm import USAGE, reset_usage, budget_status

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = os.path.join(HERE, "questions.yml")
OUT = os.path.join(HERE, "results.csv")


def numbers_in(text: str) -> list[float]:
    """Every number in the answer, including scientific notation, plus
    M/K-scaled variants."""
    found = []
    for tok in re.findall(r"-?\d[\d,]*\.?\d*(?:[eE][+-]?\d+)?",
                          text.replace("\u2009", "")):
        try:
            found.append(float(tok.replace(",", "")))
        except ValueError:
            pass
    return found + [n * 1e6 for n in found] + [n * 1e3 for n in found]


def grade(case: dict, answer: str) -> tuple[bool, str]:
    check = case.get("check", "number")

    # any_of: the same answer can be phrased several correct ways
    if check == "any_of":
        opts = [str(o).lower() for o in case["any_of"]]
        return (any(o in answer.lower() for o in opts),
                f"expected one of {case['any_of']}")

    # facts: correctness is a SET of claims, not one value
    if check == "facts":
        missing = []
        for f in case["facts"]:
            if isinstance(f, (int, float)):
                target = float(f)
                if not any(abs(c - target) <= max(abs(target) * 0.03, 0.01)
                           for c in numbers_in(answer)):
                    missing.append(f"{target:,.2f}")
            elif str(f).lower() not in answer.lower():
                missing.append(str(f))
        return (not missing), ("all facts present" if not missing
                               else f"missing: {', '.join(missing)}")

    truth = run_sql(case["reference_sql"]).iloc[0, 0]

    if check == "number":
        target = float(truth)
        return (any(abs(c - target) <= max(abs(target) * 0.02, 0.01)
                    for c in numbers_in(answer)),
                f"expected ~{target:,.2f}")

    needle = str(case.get("contains", truth)).lower()
    return needle in answer.lower(), f"expected '{needle}'"


def run_once(cases: list[dict], mode: str, use_cache: bool) -> pd.DataFrame:
    rows = []
    for i, c in enumerate(cases, 1):
        print(f"\n[{i}/{len(cases)}] {c['id']}: {c['question'][:70]}")
        reset_usage()
        t0 = time.time()
        try:
            if mode == "agent":
                st = investigate(c["question"], use_cache=use_cache, ask=False)
                answer = st["answer"]
                retries = sum(r["attempts"] - 1 for r in st["results"])
                steps = len(st["results"])
            else:  # baseline: single-shot SQL, no planner/critic
                r = ask_sql(c["question"], verbose=False)
                answer = r.df.to_string(index=False) if r.ok else "FAILED"
                retries, steps = r.attempts - 1, 1
            ok, note = grade(c, answer)
            err = ""
        except Exception as e:
            ok, note, answer, retries, steps, err = (
                False, "crash", "", 0, 0, str(e)[:120])

        rows.append({
            "id": c["id"], "question": c["question"], "correct": ok,
            "note": note, "answer": answer[:500],
            "steps": steps, "self_corrections": retries,
            "seconds": round(time.time() - t0, 1),
            "llm_calls": USAGE["calls"],
            "tokens": USAGE["prompt_tokens"] + USAGE["output_tokens"],
            "error": err,
        })
        print(f"   {'PASS' if ok else 'FAIL'}  ({note})  {rows[-1]['seconds']}s")

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, mode: str) -> None:
    n = len(df)
    print("\n" + "=" * 60)
    print(f"MODE: {mode}")
    print(f"Accuracy:          {df.correct.sum()}/{n} = {df.correct.mean()*100:.1f}%")
    print(f"Avg steps:         {df.steps.mean():.2f}")
    print(f"Self-corrections:  {df.self_corrections.sum()} total")
    print(f"Avg latency:       {df.seconds.mean():.1f}s")
    print(f"Median latency:    {df.seconds.median():.1f}s")
    print(f"Avg tokens/query:  {df.tokens.mean():,.0f}")

    diag = df[df.id.str.startswith("d")]
    look = df[~df.id.str.startswith("d")]
    if len(diag) and len(look):
        print(f"\n  lookup/comparative: {look.correct.sum()}/{len(look)}")
        print(f"  diagnostic:         {diag.correct.sum()}/{len(diag)}")

    if (~df.correct).any():
        print("\nFailures:")
        for _, r in df[~df.correct].iterrows():
            print(f"  {r['id']}: {r['note']}  {r['error']}")


def main(mode: str, limit: int | None, only: str | None,
         use_cache: bool, runs: int) -> None:
    with open(CASES, encoding="utf-8") as f:
        cases = yaml.safe_load(f)

    if only:
        want = {s.strip() for s in only.split(",")}
        cases = [c for c in cases if c["id"] in want]
    if limit:
        cases = cases[:limit]

    ds = dataset.get_active()
    print(f"DATASET: {ds['name']} ({ds['kind']})  |  MODE: {mode}  "
          f"|  cache: {'on' if use_cache else 'off'}  |  runs: {runs}")
    print(f"MODELS: {budget_status()}")

    all_runs = []
    for r in range(runs):
        if runs > 1:
            print(f"\n{'#' * 60}\n# RUN {r + 1}/{runs}\n{'#' * 60}")
        df = run_once(cases, mode, use_cache)
        df["run"] = r + 1
        all_runs.append(df)
        summarize(df, mode)

    combined = pd.concat(all_runs, ignore_index=True)
    combined.to_csv(OUT, index=False)
    print(f"\nSaved -> {OUT}")

    if runs > 1:
        scores = [d.correct.mean() for d in all_runs]
        print("\n" + "=" * 60)
        print(f"ACROSS {runs} RUNS ({mode})")
        print(f"  accuracy: {statistics.mean(scores)*100:.1f}%  "
              f"(min {min(scores)*100:.0f}%, max {max(scores)*100:.0f}%, "
              f"spread {(max(scores)-min(scores))*100:.0f} pts)")
        if runs > 2:
            print(f"  std dev:  {statistics.stdev(scores)*100:.1f} pts")

        # FLAKY means the agent is unstable; always-fails means the
        # capability is missing. They need different fixes.
        print("\n  per-question pass rate:")
        rates = (combined.groupby("id")["correct"].agg(["sum", "count"])
                 .sort_values("sum"))
        for qid, row in rates.iterrows():
            passes, total = int(row["sum"]), int(row["count"])
            tag = ("always fails" if passes == 0
                   else "stable" if passes == total else "FLAKY")
            print(f"    {qid}: {passes}/{total}  {tag}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["agent", "baseline"], default="agent")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", type=str, default=None,
                    help="comma-separated ids, e.g. d01,d04")
    ap.add_argument("--no-cache", action="store_true",
                    help="bypass the semantic cache (use for real benchmarks)")
    ap.add_argument("--runs", type=int, default=1,
                    help="repeat the benchmark N times and report the spread")
    args = ap.parse_args()
    main(args.mode, args.limit, args.only,
         use_cache=not args.no_cache, runs=args.runs)