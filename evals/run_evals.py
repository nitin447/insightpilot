"""Benchmark the agent: accuracy, self-corrections, latency, token cost."""
from __future__ import annotations
import sys, os, re, time, argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml, pandas as pd
from tools.warehouse import run_sql
from agent.graph import investigate
from agent.sql_tool import ask_sql
from agent.llm import USAGE, reset_usage

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = os.path.join(HERE, "questions.yml")
OUT = os.path.join(HERE, "results.csv")


def numbers_in(text: str) -> list[float]:
    """Every number in the answer, plus M/K-scaled variants."""
    found = []
    for tok in re.findall(r"-?\d[\d,]*\.?\d*", text.replace("\u2009", "")):
        try:
            found.append(float(tok.replace(",", "")))
        except ValueError:
            pass
    scaled = [n * 1e6 for n in found] + [n * 1e3 for n in found]
    return found + scaled


def grade(case: dict, answer: str) -> tuple[bool, str]:
    # facts check: answer must contain every required element
    if case["check"] == "facts":
        missing = []
        for f in case["facts"]:
            if isinstance(f, (int, float)):
                target = float(f)
                if not any(abs(c - target) <= max(abs(target) * 0.03, 0.01)
                           for c in numbers_in(answer)):
                    missing.append(f"{target:,.2f}")
            else:
                if str(f).lower() not in answer.lower():
                    missing.append(str(f))
        return (not missing), ("all facts present" if not missing
                               else f"missing: {', '.join(missing)}")

    truth = run_sql(case["reference_sql"]).iloc[0, 0]

    if case["check"] == "number":
        target = float(truth)
        hit = any(abs(c - target) <= max(abs(target) * 0.02, 0.01)
                  for c in numbers_in(answer))
        return hit, f"expected ~{target:,.2f}"

    needle = str(case.get("contains", truth)).lower()
    return needle in answer.lower(), f"expected '{needle}'"


def main(mode: str, limit: int | None, only: str | None):
    with open(CASES, encoding="utf-8") as f:
        cases = yaml.safe_load(f)

    if only:
        want = {s.strip() for s in only.split(",")}
        cases = [c for c in cases if c["id"] in want]
    if limit:
        cases = cases[:limit]

    rows = []
    for i, c in enumerate(cases, 1):
        print(f"\n[{i}/{len(cases)}] {c['id']}: {c['question']}")
        reset_usage()
        t0 = time.time()
        try:
            if mode == "agent":
                st = investigate(c["question"])
                answer = st["answer"]
                retries = sum(r["attempts"] - 1 for r in st["results"])
                steps = len(st["results"])
            else:  # baseline: single-shot SQL, no planner/critic
                r = ask_sql(c["question"], verbose=False)
                answer = (r.df.to_string(index=False) if r.ok else "FAILED")
                retries, steps = r.attempts - 1, 1
            ok, note = grade(c, answer)
            err = ""
        except Exception as e:
            ok, note, answer, retries, steps, err = False, "crash", "", 0, 0, str(e)[:120]

        rows.append({
            "id": c["id"], "question": c["question"], "correct": ok, "note": note,
            "answer": answer[:500],
            "steps": steps, "self_corrections": retries,
            "seconds": round(time.time() - t0, 1),
            "llm_calls": USAGE["calls"],
            "tokens": USAGE["prompt_tokens"] + USAGE["output_tokens"],
            "error": err,
        })
        print(f"   {'PASS' if ok else 'FAIL'}  ({note})  {rows[-1]['seconds']}s")

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)

    n = len(df)
    print("\n" + "=" * 60)
    print(f"MODE: {mode}")
    print(f"Accuracy:          {df.correct.sum()}/{n} = {df.correct.mean()*100:.1f}%")
    print(f"Avg steps:         {df.steps.mean():.2f}")
    print(f"Self-corrections:  {df.self_corrections.sum()} total")
    print(f"Avg latency:       {df.seconds.mean():.1f}s")
    print(f"Avg tokens/query:  {df.tokens.mean():,.0f}")
    print(f"Saved -> {OUT}")
    if (~df.correct).any():
        print("\nFailures:")
        for _, r in df[~df.correct].iterrows():
            print(f"  {r['id']}: {r['note']}  {r['error']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["agent", "baseline"], default="agent")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", type=str, default=None,
                    help="comma-separated ids, e.g. d01,d04")
    args = ap.parse_args()
    main(args.mode, args.limit, args.only)