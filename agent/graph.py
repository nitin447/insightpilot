"""
LangGraph agent for business analytics.

Flow:  clarify -> cache -> planner -> executor -> critic -> synthesizer
                                         ^          |
                                         +----------+
                                  (one revision, DIAGNOSTIC only)

Human in the loop:
  the semantic layer records which business terms are ambiguous, and the
  agent asks BEFORE computing. The answer becomes part of the cache key.
  ask=False disables this for benchmarking.

Speed:
  classification merged into the planner; fast model + low reasoning effort
  for mechanical roles; steps run in parallel; semantic cache short-circuits
  the whole graph. Only successful runs are cached.

Accuracy:
  SQL validated before execution; the critic stops on a METRIC-AWARE
  checklist (a revenue decomposition does not explain a review-score
  change); every number in the answer is verified against the evidence.
"""
from __future__ import annotations
import sys, os, json, re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import TypedDict, Annotated
import operator
from concurrent.futures import ThreadPoolExecutor
from langgraph.graph import StateGraph, END

from agent.llm import chat
from agent.sql_tool import ask_sql
from tools.warehouse import schema_context, preview, find_ambiguities
from tools import cache as qcache

MAX_REVISIONS = 1
MAX_PARALLEL = 3
EVIDENCE_ROWS = 25          # rows of each result that reach a prompt

# which operational drivers actually explain which metric
DRIVER_MAP = """  revenue / orders / GMV -> order volume, average order value,
                            cancellation rate
  review scores          -> late delivery rate, actual vs promised delivery
                            time, cancellations
  freight / shipping cost-> product weight, category mix, shipping region
  cancellations          -> segment, seller tier, payment type"""


class AgentState(TypedDict):
    question: str
    clarifications: dict
    qtype: str
    plan: list[str]
    results: Annotated[list[dict], operator.add]
    checklist: dict
    critique: str
    verdict: str
    revisions: int
    answer: str
    grounding: dict


# ------------------------------------------------------------------ helpers
def is_sql(text: str) -> bool:
    return text.upper().lstrip().startswith(("SELECT", "WITH"))


def evidence_text(results: list[dict], with_sql: bool = False) -> str:
    """Compact evidence for a prompt - truncated, because tables are expensive."""
    blocks = []
    for r in results:
        table = r["table"]
        lines = table.splitlines()
        if len(lines) > EVIDENCE_ROWS + 2:
            table = "\n".join(lines[:EVIDENCE_ROWS + 2]) + "\n..."
        block = f"STEP: {r['step']}"
        if with_sql:
            block += f"\nSQL: {r['sql']}"
        blocks.append(block + f"\nRESULT:\n{table}")
    return "\n\n".join(blocks)


def numbers_in(text: str) -> list[float]:
    """Every number in the text, including scientific notation."""
    out = []
    for tok in re.findall(r"-?\d[\d,]*\.?\d*(?:[eE][+-]?\d+)?", text):
        try:
            out.append(float(tok.replace(",", "")))
        except ValueError:
            pass
    return out


# ------------------------------------------------- planner (+ classifier)
PLANNER_SYSTEM = f"""You are a lead data analyst planning an investigation.

First classify the question:
  LOOKUP     - asks for a specific value, ranking, count, or comparison
  DIAGNOSTIC - asks WHY something happened or what CAUSED a change

Then break it into steps, each answerable by ONE SQL query.
Steps must be INDEPENDENT - they run in parallel, so no step may depend on
another step's output.
Each step is a plain-English INSTRUCTION, never SQL. Write "Compare revenue
by region and category for Q2 vs Q3 2025", not a SELECT statement.

LOOKUP: return exactly ONE step. Do not pad it with extra breakdowns.

DIAGNOSTIC: return 2-3 steps.
  - One MUST break the change down by the two most relevant dimensions
    TOGETHER.
  - One MUST test the operational drivers that plausibly affect THIS
    METRIC. Pick them from this map:
{DRIVER_MAP}
    Do NOT apply a revenue decomposition (volume / value / cancellations)
    to a metric it does not explain, such as review scores.
  - Compare against the IMMEDIATELY PRECEDING period unless the question
    asks otherwise.

Return ONLY JSON, no prose:
{{"type": "LOOKUP" or "DIAGNOSTIC", "steps": ["step one", "step two"]}}
"""


def planner(state: AgentState) -> dict:
    prompt = f"{schema_context()}\n\nBusiness question: {state['question']}"

    clar = state.get("clarifications") or {}
    if clar:
        answers = "\n".join(f"  {k}: {v}" for k, v in clar.items())
        prompt += (f"\n\nThe user has clarified:\n{answers}\n"
                   f"Apply these exactly - they override any default "
                   f"interpretation.")

    raw = chat(prompt, system=PLANNER_SYSTEM, role="plan")
    if "```" in raw:
        raw = raw.split("```")[1].replace("json", "", 1).strip()
    try:
        parsed = json.loads(raw)
        qtype = "DIAGNOSTIC" if "DIAGNOSTIC" in str(
            parsed.get("type", "")).upper() else "LOOKUP"
        plan = [str(s) for s in parsed.get("steps", [])][:3]
    except Exception:
        qtype, plan = "LOOKUP", [state["question"]]

    plan = [s for s in plan if not is_sql(s)]
    if not plan:
        plan = [state["question"]]

    print(f"\n[TYPE] {qtype}")
    print(f"[PLAN] {len(plan)} step(s)")
    for i, s in enumerate(plan, 1):
        print(f"  {i}. {s}")
    return {"qtype": qtype, "plan": plan, "revisions": 0,
            "checklist": {"segment_localized": False,
                          "mechanism_identified": False}}


# --------------------------------------------------------------- executor
def _run_step(step: str, clar: dict) -> dict:
    question = step
    if clar:
        answers = "; ".join(f"{k}: {v}" for k, v in clar.items())
        question = f"{step}\n(User clarifications to respect: {answers})"
    r = ask_sql(question, verbose=False)
    return {
        "step": step, "sql": r.sql, "ok": r.ok, "attempts": r.attempts,
        "errors": r.errors, "validator_catches": r.caught_by_validator,
        "table": preview(r.df, 15) if r.ok else f"FAILED: {r.errors}",
    }


def executor(state: AgentState) -> dict:
    done = {r["step"] for r in state.get("results", [])}
    todo = [s for s in state["plan"] if s not in done]
    if not todo:
        return {"results": []}

    clar = state.get("clarifications") or {}
    print(f"\n[RUN] {len(todo)} step(s)")
    if len(todo) == 1:
        new = [_run_step(todo[0], clar)]
    else:
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
            new = list(pool.map(lambda s: _run_step(s, clar), todo))

    for r in new:
        mark = "ok" if r["ok"] else "FAILED"
        extra = (f", {r['validator_catches']} caught pre-exec"
                 if r["validator_catches"] else "")
        print(f"  {mark} ({r['attempts']} attempt(s){extra}) - {r['step'][:70]}")
    return {"results": new}


# ----------------------------------------------------------------- critic
CRITIC_SYSTEM = f"""You review an analytical investigation for completeness.

A metric CHANGE is fully explained only when BOTH are true:

  segment_localized     - the change is attributed to a specific SEGMENT
                          (a combination of two dimensions, e.g. region x
                          category), not just one large group. Naming a big
                          category is not enough - big categories move
                          simply because they are big.
                          If the evidence shows the change is UNIFORM across
                          segments, that is also localization: set this true
                          and note it is system-wide.

  mechanism_identified  - the change is linked to an operational driver
                          that plausibly affects THIS METRIC:
{DRIVER_MAP}
                          A revenue decomposition (volume / value /
                          cancellations) applied to a review-score question
                          does NOT count. Set this false if the obvious
                          driver for this metric was never tested.

Judge ONLY from the evidence shown. Do not assume work that isn't there.

Return ONLY JSON, no prose:
{{"segment_localized": true/false,
 "mechanism_identified": true/false,
 "next_step": "<one plain-English analytical step, or empty string if both are true>"}}

next_step must be plain English, never SQL, and must not repeat a step
already taken.
"""


def critic(state: AgentState) -> dict:
    if state["qtype"] == "LOOKUP":
        return {"verdict": "PASS", "critique": ""}
    if state["revisions"] >= MAX_REVISIONS:
        print("\n[CRITIC] PASS (revision budget spent)")
        return {"verdict": "PASS", "critique": "revision limit reached"}

    already = "\n".join(f"  - {r['step']}" for r in state["results"])
    prompt = (f"QUESTION: {state['question']}\n\n"
              f"STEPS TAKEN:\n{already}\n\n"
              f"EVIDENCE:\n{evidence_text(state['results'])}")
    raw = chat(prompt, system=CRITIC_SYSTEM, role="critic")

    if "```" in raw:
        raw = raw.split("```")[1].replace("json", "", 1).strip()
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = {"segment_localized": True, "mechanism_identified": True,
                  "next_step": ""}

    checklist = {
        "segment_localized": bool(parsed.get("segment_localized")),
        "mechanism_identified": bool(parsed.get("mechanism_identified")),
    }
    next_step = str(parsed.get("next_step", "")).strip()
    if is_sql(next_step):
        next_step = ""

    complete = all(checklist.values())
    print(f"\n[CRITIC] segment={checklist['segment_localized']} "
          f"mechanism={checklist['mechanism_identified']}"
          + ("" if complete else f" -> {next_step[:80]}"))

    if complete or not next_step:
        return {"verdict": "PASS", "critique": raw, "checklist": checklist}

    return {"verdict": "REVISE", "critique": raw, "checklist": checklist,
            "plan": state["plan"] + [next_step],
            "revisions": state["revisions"] + 1}


def route(state: AgentState) -> str:
    if state["qtype"] == "LOOKUP":
        return "synthesizer"
    return "executor" if state["verdict"] == "REVISE" else "synthesizer"


# ------------------------------------------------------------ synthesizer
SYNTH_SYSTEM = """You are a business analyst writing for an executive.

**Answer:** one or two sentences with the key numbers.
**Why:** the drivers found in the evidence, with numbers.
**Caveat:** what the data does NOT prove.

Rules:
- The **Answer** line states the exact figure asked for, before any detail,
  and says WHAT it moved from and to - not just the size of the change.
- Format money as Rs X.XX M for millions, or Rs X,XXX otherwise. NEVER use
  scientific notation (no 1.96e+07), and never print the same figure twice
  in two formats.
- Write magnitudes as POSITIVE numbers with a direction word: "fell by
  Rs 1.46 M", never "dropped by Rs -1,459,117.6".
- **Why** must name the specific SEGMENT (both dimensions) and the
  operational DRIVER, with numbers for each. If the change is uniform
  across segments, SAY it is system-wide rather than naming one segment.
- RANK the drivers. If one factor explains most of the change, say so.
  Mention a minor factor only to DISMISS it - e.g. "average order value was
  essentially flat (-0.7%), so this was a volume problem."
- Never imply two drivers are comparable when their magnitudes differ by
  more than 5x.
- Do not claim a driver caused the change unless the evidence tested that
  driver directly.
- Use ONLY numbers that appear in the evidence. Never estimate. Every
  figure you write will be verified.
- For a simple lookup, keep **Why** to one line and omit **Caveat**.
- Be concise. No filler.
"""


def check_grounding(answer: str, results: list[dict]) -> dict:
    """Verify every number in the answer appears in the evidence, or is a
    simple difference of two evidence numbers (a legitimate derivation)."""
    pool = numbers_in(" ".join(r["table"] for r in results))
    diffs = {round(abs(a - b), 2) for a in pool for b in pool if a != b}

    unverified = []
    for n in numbers_in(answer):
        if 1900 <= n <= 2100 and float(n).is_integer():
            continue                      # years
        if abs(n) < 10:
            continue                      # counts, small ratios, percentages
        hit = any(
            abs(n - p) <= max(abs(p) * 0.02, 0.01)
            or abs(n * 1e6 - p) <= max(abs(p) * 0.02, 0.01)
            or abs(n * 1e3 - p) <= max(abs(p) * 0.02, 0.01)
            or abs(n / 100 - p) <= max(abs(p) * 0.02, 0.0001)
            for p in pool
        )
        if not hit:
            hit = any(abs(abs(n) - d) <= max(d * 0.02, 0.01) for d in diffs)
        if not hit:
            unverified.append(n)
    return {"checked": True, "unverified": unverified, "clean": not unverified}

def synthesizer(state: AgentState) -> dict:
    ev = evidence_text(state["results"])
    prompt = f"QUESTION: {state['question']}\n\nEVIDENCE:\n{ev}"

    clar = state.get("clarifications") or {}
    if clar:
        answers = "; ".join(f"{k}: {v}" for k, v in clar.items())
        prompt += f"\n\nUSER CLARIFICATIONS APPLIED: {answers}"

    # the critic already located the segment and the mechanism - without
    # this the synthesizer re-derives them from raw tables, and often
    # settles for a vaguer, system-wide framing
    if state.get("critique"):
        prompt += f"\n\nREVIEWER FINDINGS (already verified):\n{state['critique']}"

    answer = chat(prompt, system=SYNTH_SYSTEM, role="synthesize")

    grounding = check_grounding(answer, state["results"])
    if not grounding["clean"]:
        bad = ", ".join(f"{n:,.2f}" for n in grounding["unverified"])
        print(f"\n[GROUNDING] unverified numbers: {bad} - regenerating")
        answer = chat(
            f"{prompt}\n\nYour previous answer contained numbers NOT present "
            f"in the evidence: {bad}\nRewrite it using only figures that "
            f"appear above. Drop any claim you cannot support.",
            system=SYNTH_SYSTEM, role="synthesize")
        grounding = check_grounding(answer, state["results"])
        grounding["regenerated"] = True

    return {"answer": answer, "grounding": grounding}


# ------------------------------------------------------------------ graph
def build_agent():
    g = StateGraph(AgentState)
    g.add_node("planner", planner)
    g.add_node("executor", executor)
    g.add_node("critic", critic)
    g.add_node("synthesizer", synthesizer)

    g.set_entry_point("planner")
    g.add_edge("planner", "executor")
    g.add_edge("executor", "critic")
    g.add_conditional_edges("critic", route,
                            {"executor": "executor", "synthesizer": "synthesizer"})
    g.add_edge("synthesizer", END)
    return g.compile()


def _empty_state(question: str, clarifications: dict, pending: list) -> dict:
    return {"question": question, "clarifications": clarifications,
            "needs_clarification": pending, "answer": "", "results": [],
            "qtype": "", "plan": [], "checklist": {}, "critique": "",
            "verdict": "", "revisions": 0, "grounding": {}, "cached": ""}


def investigate(question: str, use_cache: bool = True,
                clarifications: dict | None = None,
                interactive: bool = False,
                ask: bool = True) -> dict:
    """Run the agent.

    If the question touches a term the semantic layer marks as ambiguous and
    the user has not resolved it, ASK rather than guess. interactive=True
    asks at the console; otherwise the pending questions are returned for a
    UI to put to the user. ask=False skips this and uses dataset defaults -
    required for benchmarking, where no human is present.
    """
    clarifications = dict(clarifications or {})

    pending = ([a for a in find_ambiguities(question)
                if a["term"] not in clarifications] if ask else [])
    if pending:
        if interactive:
            for a in pending:
                print(f"\n[CLARIFY] {a['ask']}")
                reply = input("  > ").strip()
                if reply:
                    clarifications[a["term"]] = reply
                else:
                    print("  (no answer - using the dataset default)")
        else:
            return _empty_state(question, clarifications, pending)

    # clarifications change the answer, so they must change the cache key
    fp = qcache.schema_fingerprint(
        schema_context() + json.dumps(clarifications, sort_keys=True))

    if use_cache:
        hit, how = qcache.lookup(question, fp)
        if hit:
            print(f"\n[CACHE] {how}")
            return {**hit, "cached": how, "needs_clarification": []}

    state = build_agent().invoke({
        "question": question, "clarifications": clarifications,
        "qtype": "", "plan": [], "results": [],
        "checklist": {}, "critique": "", "verdict": "", "revisions": 0,
        "answer": "", "grounding": {},
    })
    state["cached"] = ""
    state["needs_clarification"] = []

    # never cache a run whose steps failed - one bad run would otherwise
    # poison every future ask of that question
    all_ok = bool(state.get("results")) and all(
        r.get("ok") for r in state["results"])

    if use_cache and state.get("answer") and all_ok:
        qcache.store(question, fp, {
            "question": state["question"], "qtype": state["qtype"],
            "plan": state["plan"], "results": state["results"],
            "checklist": state["checklist"], "critique": state["critique"],
            "verdict": state["verdict"], "revisions": state["revisions"],
            "answer": state["answer"], "grounding": state["grounding"],
            "clarifications": clarifications,
        })
    return state


if __name__ == "__main__":
    import time
    from agent.llm import USAGE
    from tools import dataset

    q = sys.argv[1] if len(sys.argv) > 1 else "Why did revenue drop in Q3 2025?"
    ds = dataset.get_active()
    print("=" * 70)
    print(f"DATASET: {ds['name']} ({ds['kind']})")
    print("QUESTION:", q)
    print("=" * 70)

    t0 = time.time()
    state = investigate(q, interactive=True)
    elapsed = time.time() - t0

    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(state["answer"])

    if state.get("clarifications"):
        print(f"\n[CLARIFIED] {state['clarifications']}")

    failed = [r for r in state.get("results", []) if not r.get("ok")]
    if failed:
        print(f"\n[WARNING] {len(failed)} step(s) failed - answer is "
              f"based on partial evidence")

    g = state.get("grounding", {})
    total_tok = USAGE["prompt_tokens"] + USAGE["output_tokens"]
    print(f"\n[STATS] {state.get('qtype')} | {len(state.get('results', []))} steps "
          f"| {USAGE['calls']} LLM calls | {total_tok:,} tokens | {elapsed:.1f}s")
    print(f"[MODELS] {dict(USAGE['by_model'])}")
    print(f"[ROLES]  {({k: v['calls'] for k, v in USAGE['by_role'].items()})}")
    print(f"[CHECKLIST] {state.get('checklist')}")
    print(f"[GROUNDING] {'clean' if g.get('clean') else g}")
    if state.get("cached"):
        print(f"[CACHE] served from cache: {state['cached']}")