"""
LangGraph agent for business analytics.

Flow:  clarify -> cache -> planner -> executor -> critic -> synthesizer
                                         ^          |
                                         +----------+
                                   (revision, non-LOOKUP only)

Three question shapes, because one template cannot serve them all:
  LOOKUP      a value, ranking or count            -> 1 step, no critic
  DIAGNOSTIC  why did X move between two periods   -> headline / decompose / driver
  PREDICTIVE  what comes before an event           -> two shapes, see below

CONCEPT RESOLUTION is the first thing the planner does, and it exists
because of the single worst failure this agent had: asked why length of
stay rose, it answered with discharge_delay_hours - correct SQL, real
numbers, clean grounding, and a completely different quantity. Nothing
downstream can catch that, because by then the wrong column IS the
evidence. So the planner must now declare, before writing a single step,
whether each concept the question names is FOUND, must be COMPUTED, or is
ABSENT from the schema entirely.

METRIC KIND decides arithmetic. A segment's contribution to a SUM is its
own change; its contribution to an AVERAGE is that change weighted by its
share of the rows. Conflating the two reported one department as 491% of a
satisfaction drop that it actually drove 87% of.

EVENT KIND decides the predictive shape. An event with a timestamp has a
timeline and a lead time. An event recorded as a flag on the same row has
neither - its predictors are the other attributes of that row, and forcing
a timeline onto it produces meaningless offsets and a false "no signal".
"""
from __future__ import annotations
import sys, os, json, re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import TypedDict, Annotated
import operator
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from langgraph.graph import StateGraph, END

from agent.llm import chat
from agent.sql_tool import ask_sql
from tools.warehouse import schema_context, preview, find_ambiguities, date_ranges
from tools import cache as qcache
from tools import predictors, decompose
from tools.warehouse import load_semantics

MAX_REVISIONS = 1
MAX_PARALLEL = 3
EVIDENCE_ROWS = 25          # rows of each result that reach a prompt
MAX_GROUND_ATTEMPTS = 2     # regeneration passes before the answer discloses
LOCALIZED_SHARE = 35.0      # floor: below this, nothing is localized
WEAK_LOCALIZATION = 70.0    # below this, spend a revision trying another dimension
MECHANISM_SCALE = 20.0      # % of the change a named driver must account for
RATE_HINTS = ("rate", "pct", "percent", "ratio", "share")
QTYPES = ("LOOKUP", "DIAGNOSTIC", "PREDICTIVE")
METRIC_KINDS = ("SUM", "COUNT", "AVERAGE", "RATE", "DURATION")
EVENT_KINDS = ("DATED", "FLAG", "NONE")

# Which operational drivers explain which KIND of metric.
#
# No status VALUES are enumerated here. An earlier version listed
# "failed, unpaid, refunded or cancelled" and the model filtered on those
# literal words against a schema whose statuses are paid/pending/denied -
# matching 0 of 12,000 rows, and the empty result was then read as a
# finding. The schema context now carries the real value list; this map
# points at it instead of guessing.
DRIVER_MAP = """  a MONEY metric (revenue, collections, billings, GMV)
      -> FIRST, whatever in THIS schema stops money being collected at all.
         Find the status column in COLUMN VALUES and read which of ITS
         values mean nothing was collected. Do not assume the words - they
         differ by dataset.
         MEASURE IT BY WHAT WAS FORGONE, NOT BY WHAT WAS COLLECTED. The
         collected column is zero on exactly those rows - that is what the
         status means - so summing it returns 0 and reads as proof they
         cost nothing, which is the precise opposite of the truth. Sum the
         gross or list amount for those rows instead.
         THEN volume (how many) and unit value (how much each).
         A discount trims an amount; an uncollected charge deletes it.
  a RATE or SCORE (satisfaction, rating, quality, conversion)
      -> the operational measure that feeds it: time to resolve, actual vs
         promised, defect or error counts, queue or category mix
  a DURATION (length of stay, time to resolve, cycle time, days to ship)
      -> the WAITING portion of it, if the schema records one separately -
         a delay, a queue time, a hold. Then mix: a category whose share of
         volume grew can move an average without anything getting slower.
         Do NOT reach for money drivers here. Billing status does not
         change how long something takes.
  a COST metric
      -> the physical driver (weight, distance, volume, mix) and the price
         per unit of it
  a COUNT of things or events
      -> segment mix, acquisition channel, and the preceding funnel step

Use ONLY drivers that exist as columns in THIS schema. If a driver in this
map has no column here, it does not apply - do not substitute a similar
word from another domain."""


class AgentState(TypedDict):
    question: str
    clarifications: dict
    qtype: str
    metric_kind: str
    event_kind: str
    concepts: list
    scope: str
    periods: list
    time_column: str
    event: dict
    metric_row: str
    decomp: dict
    plan: list[str]
    results: Annotated[list[dict], operator.add]
    checklist: dict
    findings: dict
    critique: str
    verdict: str
    revisions: int
    answer: str
    grounding: dict


# ------------------------------------------------------------------ helpers
def is_sql(text: str) -> bool:
    return text.upper().lstrip().startswith(("SELECT", "WITH"))


def sanity_warnings(df: pd.DataFrame | None) -> list[str]:
    """Catch results that are structurally impossible, before they become a
    confident sentence. A churn 'rate' of -1.76 is not a rounding error, it
    is the metric defined backwards - and without this nothing downstream
    ever notices."""
    out: list[str] = []
    if df is None or df.empty:
        return out
    for c in df.columns:
        name = str(c).lower()
        if not any(h in name for h in RATE_HINTS):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        s = df[c].dropna()
        if s.empty:
            continue
        if (s < 0).any():
            out.append(
                f"WARNING: column '{c}' contains negative values (min "
                f"{s.min():.4g}). A rate or share cannot be negative - this "
                f"metric is almost certainly defined backwards, or it is "
                f"measuring growth rather than loss. Do NOT quote it as a rate.")
        if ("pct" in name or "percent" in name) and s.max() > 100:
            out.append(f"WARNING: column '{c}' reaches {s.max():.4g} but is "
                       f"named as a percentage.")
        # A single-row headline always has one distinct value - that is not a
        # breakdown, so it needs at least three rows before 'no variation'
        # means anything.
        if len(s) >= 3 and s.nunique() == 1:
            out.append(
                f"BROKEN: column '{c}' holds the identical value "
                f"({s.iloc[0]:.4g}) in every single row of this breakdown. "
                f"A rate that does not vary by group was almost certainly "
                f"computed as a whole-table aggregate - a window function "
                f"with an empty OVER() clause, or a subquery missing its "
                f"GROUP BY - not grouped per bucket as the step asked. This "
                f"is NOT a finding that the real quantity is flat; the "
                f"measurement did not run per-group at all and must be "
                f"redone with an explicit GROUP BY.")
    return out


def _is_broken(warnings: list[str]) -> bool:
    """A warning severe enough that the step measured nothing usable -
    distinct from a milder warning (e.g. a rate over 100%) that is still
    worth a caveat but not worth throwing the evidence away over."""
    return any(w.startswith("BROKEN:") for w in warnings)


def evidence_text(results: list[dict], with_sql: bool = False) -> str:
    """Compact evidence for a prompt - truncated, because tables are expensive."""
    blocks = []
    for r in results:
        table = r["table"]
        lines = table.splitlines()
        if len(lines) > EVIDENCE_ROWS + 2 and not r.get("full"):
            table = ("\n".join(lines[:EVIDENCE_ROWS + 2])
                     + "\n... (TRUNCATED - this table is incomplete, so its "
                       "rows do NOT add up to a total)")
        block = f"STEP: {r['step']}"
        if with_sql:
            block += f"\nSQL: {r['sql']}"
        blocks.append(block + f"\nRESULT:\n{table}")
    return "\n\n".join(blocks)


def evidence_health(results: list[dict]) -> str:
    """A banner naming every measurement that did not actually happen.

    Without it a failed step and a step that found nothing look identical to
    a step that found zero - and 'found zero' reads like a discovery."""
    failed = [r for r in results if not r.get("ok")]
    empty = [r for r in results if r.get("ok") and r.get("empty")]
    warned = [r for r in results if r.get("warnings")]
    if not (failed or empty or warned):
        return ""

    lines = ["EVIDENCE HEALTH - read this before concluding anything:"]
    for r in failed:
        lines.append(f"  FAILED, produced no result: {r['step'][:90]}")
    for r in empty:
        lines.append(f"  RETURNED ZERO ROWS: {r['step'][:90]}")
    for r in warned:
        for w in r["warnings"]:
            lines.append(f"  {w}")
    lines.append(
        "A question this evidence could not answer must be reported as "
        "UNANSWERED. Do not turn a measurement you failed to take into a "
        "finding that there is nothing there.")
    return "\n".join(lines)


# A '-' directly after a digit is a range or a date separator, not a minus
# sign: "26.5-26.8" is two positive numbers, and reading the second as
# -26.8 makes a correctly-grounded figure look fabricated.
_NUM = re.compile(r"(?<![\d.])-?\d[\d,]*\.?\d*(?:[eE][+-]?\d+)?")


def numbers_in(text: str) -> list[float]:
    """Every number in the text, including scientific notation."""
    out = []
    for tok in _NUM.findall(text):
        try:
            out.append(float(tok.replace(",", "")))
        except ValueError:
            pass
    return out


def _skip_number(n: float) -> bool:
    """Years, step counts, month offsets and small ordinals carry no claim."""
    if 1900 <= n <= 2100 and float(n).is_integer():
        return True
    return float(n).is_integer() and abs(n) <= 12


def _json_from(raw: str) -> dict:
    if "```" in raw:
        raw = raw.split("```")[1].replace("json", "", 1).strip()
    try:
        return json.loads(raw)
    except Exception:
        start, end = raw.find("{"), raw.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(raw[start:end + 1])
            except Exception:
                pass
    return {}


def revision_budget(results: list[dict]) -> int:
    """Broken evidence buys one extra attempt. A step that failed or found
    nothing is precisely the case where another pass is worth paying for."""
    broken = any((not r.get("ok")) or r.get("empty") or r.get("broken")
                for r in results)
    return min(MAX_REVISIONS + (1 if broken else 0), 2)


def evidence_healthy(results: list[dict]) -> bool:
    return bool(results) and all(
        r.get("ok") and not r.get("empty") and not r.get("broken")
        for r in results)


# ------------------------------------------------------ concept resolution
def clean_concepts(raw) -> list[dict]:
    """Normalise whatever the planner returned. Every field optional, so a
    model that omits the block degrades to today's behaviour rather than
    crashing."""
    out = []
    if not isinstance(raw, list):
        return out
    for c in raw:
        if not isinstance(c, dict):
            continue
        asked = str(c.get("asked", "")).strip()
        if not asked:
            continue
        status = str(c.get("status", "")).strip().upper()
        if status not in ("FOUND", "COMPUTED", "ABSENT", "SUBSTITUTED"):
            status = "FOUND"
        out.append({
            "asked": asked, "status": status,
            "resolved": str(c.get("resolved", "")).strip(),
            "unit": str(c.get("unit", "")).strip(),
            "nearest": str(c.get("nearest", "")).strip(),
        })
    return out


def sql_notes(concepts: list[dict]) -> str:
    """What the SQL writer must be told before it reaches for a lookalike."""
    lines = []
    for c in concepts:
        if c["status"] == "COMPUTED" and c["resolved"]:
            unit = f"   [{c['unit']}]" if c["unit"] else ""
            lines.append(f'  "{c["asked"]}" = {c["resolved"]}{unit}')
        elif c["status"] == "ABSENT":
            near = f"; nearest column is {c['nearest']}" if c["nearest"] else ""
            lines.append(f'  "{c["asked"]}" is NOT in this schema{near}. '
                         f'Do not substitute it silently.')
    if not lines:
        return ""
    return ("RESOLVED CONCEPTS - use these expressions exactly:\n"
            + "\n".join(lines))


# Words that carry no identity on their own: "patient satisfaction" is about
# satisfaction, and "total revenue" is about revenue.
_GENERIC = {"patient", "patients", "customer", "customers", "client", "user",
            "users", "total", "average", "overall", "number", "count", "rate",
            "amount", "value", "data", "score", "level", "per", "the", "and",
            "account", "accounts", "monthly", "daily", "yearly"}


def _stems(text: str) -> set[str]:
    out = set()
    for w in re.findall(r"[a-z]+", str(text).lower()):
        if len(w) < 4 or w in _GENERIC:
            continue
        out.add(w[:-1] if w.endswith("s") and len(w) > 4 else w)
    return out


def _schema_names() -> dict[str, set[str]]:
    """Every column and defined metric, with the word stems of its name."""
    names: dict[str, set[str]] = {}
    try:
        from tools.warehouse import connect as _connect
        con = _connect()
        try:
            for (t,) in con.execute(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema='main' AND table_type='BASE TABLE'").fetchall():
                for r in con.execute(f'PRAGMA table_info("{t}")').fetchall():
                    names[str(r[1])] = _stems(str(r[1]).replace("_", " "))
        finally:
            con.close()
    except Exception:
        pass
    try:
        for m in ((load_semantics() or {}).get("metrics") or {}):
            names[str(m)] = _stems(str(m).replace("_", " "))
    except Exception:
        pass
    return names


def check_concepts(concepts: list[dict]) -> list[dict]:
    """Hold the planner's concept calls to the schema's actual names.

    Two opposite failures, both silent. Asked about "patient satisfaction",
    the planner called it ABSENT - beside a column named satisfaction_score.
    Asked about "insurance provider", it called payer_type FOUND - a category
    of payer, not a company - and the answer never said so. So:
      ABSENT, but a column's name contains the concept's word  -> FOUND
      FOUND as a column sharing no word with what was asked    -> SUBSTITUTED
    A substitute is not an error. It is disclosed, so the reader knows the
    question was answered through a stand-in.
    """
    names = _schema_names()
    if not names:
        return concepts
    out = []
    for c in concepts:
        c = dict(c)
        asked = _stems(c["asked"])
        if c["status"] == "ABSENT" and asked:
            hits = [n for n, st in names.items() if asked & st]
            if len(hits) == 1:
                c.update(status="FOUND", resolved=hits[0], nearest="")
        elif c["status"] == "FOUND" and asked:
            res = c["resolved"].split(".")[-1]
            if res in names and not (asked & names[res]):
                c.update(status="SUBSTITUTED", nearest=res)
        out.append(c)
    return out


def concepts_text(concepts: list[dict]) -> str:
    """What the writer must disclose."""
    notable = [c for c in concepts if c["status"] in ("COMPUTED", "ABSENT",
                                                     "SUBSTITUTED")]
    if not notable:
        return ""
    lines = ["CONCEPT RESOLUTION - the Answer must disclose every ABSENT item,"
             " and give the UNIT of every COMPUTED one:"]
    for c in notable:
        if c["status"] == "COMPUTED":
            unit = f"  [{c['unit']}]" if c["unit"] else ""
            lines.append(f'  "{c["asked"]}" was COMPUTED as '
                         f'{c["resolved"] or "an expression"}{unit} - it is '
                         f'not a stored column.')
        elif c["status"] == "SUBSTITUTED":
            lines.append(f'  "{c["asked"]}" has no field of that name; it was '
                         f'answered using {c["nearest"]}. Say so in the Answer, '
                         f'in plain words, so the reader can judge whether '
                         f'{c["nearest"]} is what they meant.')
        else:
            near = (f" The nearest available is {c['nearest']}, which is not "
                    f"the same thing.") if c["nearest"] else ""
            lines.append(f'  "{c["asked"]}" is ABSENT from this data.{near}')
    return "\n".join(lines)


# ------------------------------------------------- planner (+ classifier)
PLANNER_SYSTEM = f"""You are a lead data analyst planning an investigation.

STEP ONE - RESOLVE THE CONCEPTS, before writing any step.

For the metric the question asks about, and for every dimension it asks to
break down or filter by, decide which of these it is:

  FOUND    - a column or a defined metric holds it. Give its name.
  COMPUTED - the data implies it but no single column holds it. Give the
             SQL expression and the UNIT. A duration between two timestamps
             is the common case: "length of stay" is almost never a column.
             Write a duration ONLY as DATE_DIFF('hour', start, end) / 24.0
             (for days) or DATE_DIFF('<unit>', start, end) - NEVER as
             `end - start`. Subtracting timestamps in DuckDB yields an
             INTERVAL that cannot be divided by a number, and this
             expression is copied verbatim into every query that uses it.
  ABSENT   - this schema does not record it at all. Give the nearest
             column and say what that column actually is.

List EVERY concept the question names - the metric and every dimension -
including ones you resolve as FOUND, so each can be checked.
NEVER resolve a concept to a column that merely sounds similar. A column
named discharge_delay_hours is NOT length of stay - one is the wait to be
released, the other is the whole visit, and they differ by an order of
magnitude. A column named payer_type is NOT an insurance provider - one is
a category, the other is a company. When a lookalike is the only candidate,
the honest answer is ABSENT with that column as `nearest`, and the final
answer will disclose the substitution instead of hiding it.

Declare the KIND of the headline metric. It decides how a segment's
contribution is computed later, and getting it wrong misstates that
contribution by a factor of five:
  SUM      - revenue, totals, amounts added up
  COUNT    - a number of rows or entities
  AVERAGE  - a mean of a per-row value (a score, an amount per case)
  RATE     - a share or percentage of rows
  DURATION - an average time span

For a PREDICTIVE question, declare how the event is recorded:
  DATED - a timestamp column marks WHEN it happened, and the predictor is
          measured repeatedly over time. Offsets relative to that date are
          meaningful.
  FLAG  - a boolean or status on the SAME row marks it. There is no
          timeline. The predictors are the other attributes of that row.
  NONE  - not a predictive question.

SCOPE. If the question restricts itself to one segment - "in oncology",
"for enterprise accounts", "in APAC" - declare that restriction as a SQL
filter in "scope", using a value from COLUMN VALUES. Every query in this
investigation will be forced to apply it, including follow-up steps, so a
breakdown can never drift out to the whole business and have its numbers
reported as the segment's.

METRIC ROW. For a DIAGNOSTIC question, write the metric as a PER-ROW SQL
expression on the table that holds time_column - not an aggregate. The
aggregation comes from metric_kind:
  SUM      the amount on each row, e.g. amount_paid, or
           CASE WHEN payment_status = 'paid' THEN net_amount ELSE 0 END
  COUNT    leave empty - every row counts once
  AVERAGE  the value on each row, e.g. satisfaction_score
  DURATION the span on each row, e.g.
           DATE_DIFF('hour', admission_ts, discharge_ts) / 24.0
  RATE     a boolean CONDITION that marks the rows counted, e.g.
           claim_status = 'denied'
When you give it, the headline, the breakdown by every dimension with each
segment's weighted contribution, and the inside-versus-outside contrast are
all computed in code, and your steps are replaced. Still write the steps, in
case the expression cannot be used.

EVENT. For a PREDICTIVE question whose event_kind is FLAG, name the column
that records the event in "event" - a boolean column on its own, or a status
column plus the value (from COLUMN VALUES) that means it happened. When you
do, the attribute sweep is computed in code over every row, so your steps
for that case are replaced; still write them in case the event cannot be
resolved.

PERIODS. A DIAGNOSTIC question compares two periods. Declare them in
"periods" as literal dates - start inclusive, end exclusive, EARLIER period
first - with the date column in "time_column". If the question names the
periods, use those. If it does not, use the latest complete quarter and the
one before it, exactly as DATE RANGES lists them. Every query will be forced
to filter to these two periods, so a breakdown can never return all of
history, get truncated, and lose the very rows the question is about.
For LOOKUP and PREDICTIVE, leave "periods" empty unless the question names
a period.

STEP TWO - PLAN.

Break the question into steps, each answerable by ONE SQL query.
Steps must be INDEPENDENT - they run in parallel, so no step may depend on
another step's output.
Each step is a plain-English INSTRUCTION, never SQL.

A filter may only use a value that appears in COLUMN VALUES. If the word
you want is not there, this data does not record it under that name.

Time windows:
- If the question NAMES a period, every step carries that period, and each
  step states the exact periods inside itself. A breakdown over all history
  cannot be compared against a headline for one quarter, and that mismatch
  is invisible in the output - the numbers all look plausible.
- If the question names NO period and asks for a LEVEL or a ranking, use
  all the data and say so in the step. Do not invent a window.
- If the question names NO period but asks WHY something changed - rose,
  fell, dropped, improved - there must be two periods to compare. Use the
  latest complete quarter present in the data against the quarter before
  it, name both in EVERY step, and make the HEADLINE compare exactly those
  two. A single-period headline cannot show a change, and a headline over a
  different window from the breakdown cannot be compared with it.
- Anchor any relative window to the data, never to today - "the most recent
  month present in the data", never "the current month".

The question may assert something the data does not support - a drop that
was actually a rise, a spike that never happened. NEVER build the premise
into a step. Step 1 measures what the metric ACTUALLY did, so the premise
can be checked rather than assumed.

LOOKUP: exactly ONE step. Do not pad it with extra breakdowns.

DIAGNOSTIC: exactly THREE steps, in this order.

  1. HEADLINE - the metric itself for the period asked about and for the
     comparison period, with NO breakdown by any dimension and NO row limit.
     This is the figure the final answer quotes.

  2. DECOMPOSE - the same comparison, for THE SAME TWO PERIODS, broken down
     by ONE dimension: the one most likely to carry the change. Name both
     periods in the step. Ask for EVERY value of that dimension.
     Pick the dimension the CAUSE would live in, not the most obvious one.
     A billing problem lives in the payer or channel, not in the department
     that happened to treat the patient.

  3. DRIVER - the operational driver that plausibly moves THIS metric, FOR
     THE SAME TWO PERIODS and BROKEN DOWN BY THE SAME DIMENSION AS STEP 2.
     Work down the map for the metric's kind:
{DRIVER_MAP}

  - Compare against the IMMEDIATELY PRECEDING period unless asked otherwise.

PREDICTIVE with event_kind DATED: exactly THREE steps.

  1. BASE RATE - how many entities experienced the event, over what span,
     using the column that records it DIRECTLY.
  2. TIMELINE - for the entities that DID experience it, average the
     candidate metric by how many periods BEFORE the event each observation
     falls. Ask for the event period plus at least four leading up to it.
  3. BASELINE - the same metric for entities that did NOT experience the
     event, grouped by the SAME offsets, anchored to the most recent month
     in the data.

PREDICTIVE with event_kind FLAG: exactly THREE steps. There is no timeline
here - do not invent offsets.

  1. BASE RATE - the overall rate of the event across all rows, and the
     count. One number for everything else to be compared against.
  2. CATEGORICAL SWEEP - the event rate for EVERY value of EVERY
     low-cardinality attribute, in ONE query: UNION ALL a block per
     attribute, each selecting a literal attribute name, the value, the row
     count and the event rate. Ranking attributes against each other is the
     point. Do NOT pick one attribute and test only that - the strongest
     predictor is rarely the first one you would think of, and finding
     nothing in one column says nothing about the question.

     A real driver is frequently an INTERACTION that is invisible in any
     single attribute - each attribute alone looks like noise, but two of
     them TOGETHER separate the rows sharply. So this query must ALSO
     include, as further UNION ALL blocks, the two or three most plausible
     COMBINATIONS of attributes (an admission or acquisition channel
     crossed with a derived severity or duration bucket is the common
     shape) - each combination as its own labelled bucket, e.g.
     'emergency + short stay' vs 'everything else', with its own rate. Do
     not skip this because it costs extra blocks: testing attributes only
     one at a time is exactly how a real interaction gets missed and
     reported as "no signal".
  3. NUMERIC CONTRAST - the event rate for rows above versus below the
     median of each plausible numeric measure (a duration, an amount, a
     count), again as one query with a labelled attribute column. A measure
     that must be COMPUTED counts here - compute it.

  Every block in steps 2 and 3 MUST GROUP BY its own bucket explicitly.
  Never compute the rate with a window function that has an empty OVER()
  clause - that aggregates over the whole table and returns the identical
  number in every row, which is a broken measurement, not a flat result.

Return ONLY JSON, no prose:
{{"type": "LOOKUP" or "DIAGNOSTIC" or "PREDICTIVE",
 "metric": "<the headline metric in plain words>",
 "scope": "<SQL filter the question restricts everything to, e.g. department = 'oncology', or empty>",
 "time_column": "<table.column that dates each row, for DIAGNOSTIC, else empty>",
 "metric_row": "<the metric as a PER-ROW SQL expression on that table, for DIAGNOSTIC>",
 "event": {{"column": "<table.column that records the event, for PREDICTIVE FLAG>",
            "value": "<the value meaning it happened, or empty for a boolean column>"}},
 "periods": [{{"label": "Q1 2026", "start": "2026-01-01", "end": "2026-04-01"}},
             {{"label": "Q2 2026", "start": "2026-04-01", "end": "2026-07-01"}}],
 "metric_kind": "SUM" or "COUNT" or "AVERAGE" or "RATE" or "DURATION",
 "event_kind": "DATED" or "FLAG" or "NONE",
 "concepts": [{{"asked": "<the words the question used>",
               "status": "FOUND" or "COMPUTED" or "ABSENT",
               "resolved": "<column name or SQL expression>",
               "unit": "<days, hours, INR, %, or empty>",
               "nearest": "<nearest column if ABSENT>"}}],
 "steps": ["step one", "step two", "step three"]}}
"""


_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_IDENT = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)?")
_NAMES_PERIOD = re.compile(
    r"(?i)\b(q[1-4]|quarter|20\d\d|19\d\d|jan|feb|mar|apr|may|jun|jul|aug|"
    r"sep|oct|nov|dec|month|week|year|ytd|h[12])\b")


_CHANGE_WORDS = re.compile(
    r"(?i)\b(why|caus\w*|drove|driv\w*|explain\w*|chang\w*|fell|fall\w*|"
    r"drop\w*|rose|rise|rising|increas\w*|decreas\w*|declin\w*|grew|grow\w*|"
    r"jump\w*|spik\w*|dip\w*|surg\w*|plung\w*|slump\w*|improv\w*|"
    r"worsen\w*|went (?:up|down)|shift\w*|moved?)\b")


def clean_periods(raw) -> list[dict]:
    """At most two periods with literal ISO dates. Anything else is dropped -
    these strings go into every SQL prompt."""
    out = []
    if not isinstance(raw, list):
        return out
    for p in raw[:2]:
        if not isinstance(p, dict):
            continue
        a, b = str(p.get("start", "")).strip(), str(p.get("end", "")).strip()
        if not (_DATE.fullmatch(a) and _DATE.fullmatch(b)) or a >= b:
            continue
        label = re.sub(r"[^\w\s\-/]", "", str(p.get("label", "")))[:30] or f"{a}..{b}"
        out.append({"label": label, "start": a, "end": b})
    out.sort(key=lambda p: p["start"])
    return out


def default_periods(time_column: str = "") -> tuple[list[dict], str]:
    """Latest complete quarter and the one before it, read from the data."""
    try:
        dr = date_ranges()
    except Exception:
        return [], time_column
    if not dr:
        return [], time_column
    col = time_column if time_column in dr else next(iter(dr))
    d = dr[col]
    if not d.get("previous"):
        return [], col
    return ([{"label": d["previous"][0], "start": d["previous"][1], "end": d["previous"][2]},
             {"label": d["latest"][0], "start": d["latest"][1], "end": d["latest"][2]}], col)


def periods_note(periods: list[dict], time_column: str) -> str:
    if not periods:
        return ""
    col = time_column.split(".")[-1] if time_column else "the date column"
    lines = [f"PERIODS - EVERY query must restrict {col} to exactly these, and "
             f"label each row with its period:"]
    for p in periods:
        lines.append(f"  {p['label']}: {col} >= TIMESTAMP '{p['start']}' "
                     f"AND {col} < TIMESTAMP '{p['end']}'")
    lines.append("Return NO other periods. A query over all history returns "
                 "rows the question did not ask for, gets truncated, and can "
                 "lose exactly the periods it did ask for. Use these literal "
                 "dates - never CURRENT_DATE.")
    return "\n".join(lines)


def planner(state: AgentState) -> dict:
    prompt = f"{schema_context()}\n\nBusiness question: {state['question']}"

    clar = state.get("clarifications") or {}
    if clar:
        answers = "\n".join(f"  {k}: {v}" for k, v in clar.items())
        prompt += (f"\n\nThe user has clarified:\n{answers}\n"
                   f"Apply these exactly - they override any default "
                   f"interpretation.")

    raw = chat(prompt, system=PLANNER_SYSTEM, role="plan")
    parsed = _json_from(raw)

    if parsed:
        declared = str(parsed.get("type", "")).upper()
        qtype = next((t for t in QTYPES if t in declared), "LOOKUP")
        kind = str(parsed.get("metric_kind", "")).upper()
        metric_kind = next((k for k in METRIC_KINDS if k in kind), "SUM")
        ev = str(parsed.get("event_kind", "")).upper()
        event_kind = next((k for k in EVENT_KINDS if k in ev), "NONE")
        concepts = clean_concepts(parsed.get("concepts"))
        scope = str(parsed.get("scope", "") or "").strip()
        if is_sql(scope) or ";" in scope or len(scope) > 200:
            scope = ""
        periods = clean_periods(parsed.get("periods"))
        event = parsed.get("event") if isinstance(parsed.get("event"), dict) else {}
        metric_row = str(parsed.get("metric_row", "") or "").strip()[:400]
        time_column = str(parsed.get("time_column", "") or "").strip()
        if not _IDENT.fullmatch(time_column):
            time_column = ""
        plan = [str(s) for s in parsed.get("steps", [])][:3]
    else:
        periods, time_column, event, metric_row = [], "", {}, ""
        qtype, metric_kind, event_kind, concepts = "LOOKUP", "SUM", "NONE", []
        scope = ""
        plan = [state["question"]]

    # A predictive question with no declared event shape gets the flag
    # treatment: a timeline needs a date, and inventing one produces
    # meaningless offsets.
    if qtype == "PREDICTIVE" and event_kind == "NONE":
        event_kind = "FLAG"

    plan = [s for s in plan if not is_sql(s)]
    if not plan:
        plan = [state["question"]]

    print(f"\n[TYPE] {qtype}  metric={metric_kind}"
          + (f"  event={event_kind}" if qtype == "PREDICTIVE" else ""))
    for c in concepts:
        if c["status"] != "FOUND":
            print(f"  [CONCEPT] {c['asked']} -> {c['status']} "
                  f"{c['resolved'] or c['nearest']}")
    print(f"[PLAN] {len(plan)} step(s)")
    for i, s in enumerate(plan, 1):
        print(f"  {i}. {s}")

    blank = ({"signal_found": False, "baseline_compared": False}
             if qtype == "PREDICTIVE"
             else {"segment_localized": False, "mechanism_identified": False})
    # A change question needs two periods. If the model named none and the
    # question names none either, take them from the data itself rather
    # than let each step guess - or reach for CURRENT_DATE.
    if qtype == "DIAGNOSTIC" and not periods and not _NAMES_PERIOD.search(
            state["question"]):
        periods, time_column = default_periods(time_column)

    concepts = check_concepts(concepts)

    # "Which payer has the highest denial rate?" asks for a level, not a
    # change. Classified as DIAGNOSTIC it was answered for one quarter
    # (33.14%) instead of overall (9.22%). A question with no word of change
    # is a lookup, and gets no invented periods.
    if qtype == "DIAGNOSTIC" and not _CHANGE_WORDS.search(state["question"]):
        print("  [TYPE] no change word in the question -> LOOKUP")
        qtype = "LOOKUP"
        plan = plan[:1] or [state["question"]]
        if not _NAMES_PERIOD.search(state["question"]):
            periods, time_column = [], ""

    # A change question with a usable metric is decomposed in code.
    if qtype == "DIAGNOSTIC" and periods and time_column:
        try:
            ok = decompose.resolve(metric_row, metric_kind, time_column,
                                   periods, scope) is not None
        except Exception:
            ok = False
        if ok:
            plan = list(decompose.DECOMP_STEPS)
            print(f"  [METRIC] {metric_kind}: {metric_row or 'COUNT(*)'} "
                  f"-> decomposition computed in code")
        else:
            metric_row = ""

    # A flag event with a resolvable column is swept in code, not planned.
    if qtype == "PREDICTIVE" and event_kind == "FLAG" and event:
        try:
            resolved = predictors.resolve_event(event)
        except Exception:
            resolved = None
        if resolved:
            plan = list(predictors.SWEEP_STEPS)
            print(f"  [EVENT] {resolved[0]}.{resolved[1]} -> sweep computed in code")
        else:
            event = {}

    if scope:
        print(f"  [SCOPE] {scope}")
    if periods:
        print(f"  [PERIODS] {time_column}: "
              + " vs ".join(f"{p['label']} [{p['start']}, {p['end']})" for p in periods))
    return {"qtype": qtype, "metric_kind": metric_kind,
            "event_kind": event_kind, "concepts": concepts, "scope": scope,
            "periods": periods, "time_column": time_column, "event": event,
            "metric_row": metric_row, "decomp": {},
            "plan": plan, "revisions": 0, "findings": {}, "checklist": blank}


# --------------------------------------------------------------- executor
def _run_step(step: str, clar: dict, notes: str) -> dict:
    question = step
    if clar:
        answers = "; ".join(f"{k}: {v}" for k, v in clar.items())
        question = f"{step}\n(User clarifications to respect: {answers})"
    r = ask_sql(question, verbose=False, notes=notes)

    empty = bool(r.ok and (r.df is None or r.df.empty))
    warnings = sanity_warnings(r.df) if r.ok else []

    if not r.ok:
        table = f"FAILED: {r.errors}"
    elif empty:
        table = ("(0 rows - this step produced NO evidence. It did not find "
                 "that the quantity is zero; it found nothing at all.)")
    else:
        table = preview(r.df, EVIDENCE_ROWS)
        if warnings:
            table += "\n" + "\n".join(warnings)

    return {
        "step": step, "sql": r.sql, "ok": r.ok, "attempts": r.attempts,
        "errors": r.errors, "validator_catches": r.caught_by_validator,
        "empty": empty, "warnings": warnings, "broken": _is_broken(warnings),
        "table": table,
    }


def executor(state: AgentState) -> dict:
    done = {r["step"] for r in state.get("results", [])}
    todo = [s for s in state["plan"] if s not in done]
    if not todo:
        return {"results": []}

    decomp_steps = [s for s in todo if s in decompose.DECOMP_STEPS]
    if decomp_steps:
        print(f"\n[DECOMPOSE] computing headline, breakdown and contrast in code")
        try:
            got = decompose.decompose(state.get("metric_row") or "",
                                      state.get("metric_kind") or "SUM",
                                      state.get("time_column") or "",
                                      state.get("periods") or [],
                                      state.get("scope") or "")
        except Exception as e:
            got = None
            print(f"  decomposition failed: {str(e)[:120]}")
        if got:
            results, summary = got
            for r in results:
                print(f"  ok (computed) - {r['step'][:70]}")
            return {"results": results, "decomp": summary}
        return {"results": [{
            "step": s, "sql": "", "ok": False, "attempts": 1,
            "errors": ["the decomposition could not be computed"],
            "validator_catches": 0, "empty": False, "warnings": [],
            "broken": False, "table": "FAILED: the decomposition could not "
            "be computed"} for s in decomp_steps]}

    swept = [s for s in todo if s in predictors.SWEEP_STEPS]
    if swept:
        print(f"\n[SWEEP] computing predictor sweep in code")
        try:
            got = predictors.flag_sweep(state.get("event") or {},
                                        state.get("scope") or "")
        except Exception as e:
            got = None
            print(f"  sweep failed: {str(e)[:120]}")
        if got:
            for r in got:
                print(f"  ok (computed) - {r['step'][:70]}")
            return {"results": got}
        return {"results": [{
            "step": s, "sql": "", "ok": False, "attempts": 1,
            "errors": ["the predictor sweep could not be computed"],
            "validator_catches": 0, "empty": False, "warnings": [],
            "broken": False, "table": "FAILED: the predictor sweep could not "
            "be computed"} for s in swept]}

    clar = state.get("clarifications") or {}
    notes = sql_notes(state.get("concepts") or [])
    scope = state.get("scope") or ""
    if scope:
        notes = (f"SCOPE - EVERY query must filter to: {scope}\n"
                 f"The question is about that segment only. A query without "
                 f"this filter measures the whole business, and its numbers "
                 f"will be reported as if they were this segment's.\n\n"
                 + notes).strip()
    pnote = periods_note(state.get("periods") or [], state.get("time_column") or "")
    if pnote:
        notes = (pnote + "\n\n" + notes).strip()
    print(f"\n[RUN] {len(todo)} step(s)")
    if len(todo) == 1:
        new = [_run_step(todo[0], clar, notes)]
    else:
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
            new = list(pool.map(lambda s: _run_step(s, clar, notes), todo))

    for r in new:
        mark = "FAILED" if not r["ok"] else ("EMPTY" if r["empty"] else "ok")
        extra = (f", {r['validator_catches']} caught pre-exec"
                 if r["validator_catches"] else "")
        if r["warnings"]:
            extra += f", {len(r['warnings'])} sanity warning(s)"
        print(f"  {mark} ({r['attempts']} attempt(s){extra}) - {r['step'][:70]}")
    return {"results": new}


# ----------------------------------------------------------------- critic
_HEALTH_RULE = """
If the EVIDENCE HEALTH section reports a failed step, a step that returned
zero rows, or a metric that looks defined backwards, that measurement did
not happen. Set the affected check to false and make next_step REDO it -
preferring a column that records the event directly over any quantity
derived from side effects.

Return ONLY JSON, no prose. next_step must be plain English, never SQL, and
must not repeat a step already taken.
"""

# How a segment's contribution is computed depends entirely on the metric.
# Handed to the critic in the USER prompt so it cannot pick the wrong one.
KIND_RULE = {
    "SUM": (
        "METRIC KIND: SUM. A segment's contribution to the change IS its own "
        "change, so share_of_change_pct = segment change / total change x 100. "
        "mechanism_scale_pct is a genuine ratio here - compute it the same way."),
    "COUNT": (
        "METRIC KIND: COUNT. Treat it exactly as a SUM: a segment's own change "
        "IS its contribution, and mechanism_scale_pct is a genuine ratio."),
    "AVERAGE": (
        "METRIC KIND: AVERAGE. A segment's own change is NOT its contribution. "
        "Weight it by that segment's share of the rows: contribution = own "
        "change x share of volume. A department that fell 1.08 points while "
        "holding 17% of cases contributes about 0.19, not 1.08. CHECK that the "
        "contributions of all segments roughly ADD UP to the total change; if "
        "yours implies one segment moved the metric several times more than it "
        "actually moved, you used an unweighted change - redo it.\n"
        "mechanism_scale_pct cannot be a ratio for an averaged metric. Judge "
        "the driver by CONTRAST instead: report 100 if it moved sharply in the "
        "SAME segment that moved and stayed flat everywhere else, 0 if it moved "
        "everywhere or only outside that segment."),
}
KIND_RULE["RATE"] = KIND_RULE["AVERAGE"]
KIND_RULE["DURATION"] = KIND_RULE["AVERAGE"]

CRITIC_DIAGNOSTIC = f"""You review an analytical investigation for completeness.

You do NOT get to answer "is it localized?" with a bare yes. Report the
numbers and the caller decides:

  pattern               - "localized" if ONE segment carries most of the
                          change, "uniform" if it is spread evenly.
  top_segment           - the single segment with the largest contribution,
                          named exactly as the evidence names it. Empty only
                          if pattern is "uniform".
  share_of_change_pct   - that segment's contribution as a percentage of the
                          TOTAL change, computed by the rule in METRIC KIND
                          below. It can exceed 100 when other segments moved
                          the other way. Do not call a change uniform while
                          reporting a large share.
                          If the share is under 70% and two or more other
                          segments moved the same way, the change is NOT well
                          localized by this dimension - say so in next_step
                          and propose a DIFFERENT dimension to try. The cause
                          often lives in a dimension nobody thought to split
                          by: a billing problem shows up under payer, not
                          under the department that did the work.

  mechanism             - the operational driver the evidence links to the
                          change, in one clause, WITH its numbers expressed
                          as a CHANGE between the same two periods. A level
                          in one period is not comparable to a change:
                          "lost revenue was 14.28 M in Q2" cannot be weighed
                          against a 10.92 M drop, because some of that loss
                          existed in Q1 too. Quote it as "rose from X to Y, a
                          change of Z" and use Z as the magnitude.
  mechanism_scale_pct   - how much of the change that driver accounts for,
                          by the rule in METRIC KIND below. SIZE IS THE TEST:
                          a driver far smaller than the change does not
                          explain it, and naming it is worse than naming
                          nothing because it reads as an answer.
                          Beware a driver measured as ZERO: if the evidence
                          sums a COLLECTED amount for rows whose status means
                          nothing was collected, it is always zero. That is a
                          broken measurement, not a finding - set
                          mechanism_identified false and make next_step
                          measure the GROSS amount forgone instead.
  mechanism_identified  - true only if a driver that plausibly affects THIS
                          metric was actually tested AND is large enough:
{DRIVER_MAP}

Judge ONLY from the evidence shown. Do not assume work that isn't there.
If a table is marked TRUNCATED, do not treat its rows as a complete set.
If a breakdown covers a DIFFERENT window than the headline, it cannot be
compared to it - set the affected check false and redo it over the
headline's periods.
If the headline shows the metric did NOT move the way the question assumed -
it moved the opposite way, or barely moved at all - set both checks false:
there is no cause to find for a change that did not happen. Then:
  - if the breakdown step SUCCEEDED, next_step is EMPTY. The investigation
    is complete; the breakdown already shows which segments moved which way,
    and the answer will report them as context.
  - if the breakdown step FAILED or returned nothing, next_step REDOES the
    breakdown - not the headline. The answer still needs to say where the
    segments went, and without that measurement it cannot.
Never spend a revision re-measuring a headline that already succeeded.
next_step is ONE instruction for a query, or an empty string. Never write a
conclusion or a verdict in it - if you are done, it is empty. Any step you
propose must keep every restriction the question makes: a question about
oncology gets an oncology step, never a whole-hospital one. A step that
REDOES an earlier one measures the SAME metric as the step it replaces -
a redo of a satisfaction breakdown is a satisfaction breakdown, never a
billing one.
{_HEALTH_RULE}
{{"pattern": "localized" or "uniform",
 "top_segment": "<name or empty>",
 "share_of_change_pct": <number>,
 "mechanism": "<one clause with numbers, as a CHANGE>",
 "mechanism_scale_pct": <number>,
 "mechanism_identified": true/false,
 "next_step": "<one step, or empty string if the investigation is complete>"}}
"""

CRITIC_PREDICTIVE_DATED = f"""You review an investigation into whether an
early warning signal exists before a dated event.

  signal_found       - true ONLY if the candidate metric MOVES as the event
                       approaches. A genuinely flat line is a real finding of
                       "no signal", but only when the measurement succeeded.
                       A failed or empty step is not a flat line.
  signal             - the metric and how it moves, in one clause, WITH the
                       numbers at each offset. Quote the change between
                       PRE-EVENT periods only: the change between the last
                       pre-event period and the event period itself is not a
                       warning, because by then the event has happened and an
                       alert built on it fires too late to act on.
  lead_time          - how far ahead the movement BEGINS. NOT the span of the
                       table. Compare each period with the one before it; the
                       lead time starts at the first period whose change is
                       clearly larger than the drift before it. A series flat
                       for three periods that then falls 37% gives ONE period
                       of warning, not four. State the per-period changes.
  baseline_compared  - true only if the same metric was measured for entities
                       that did NOT experience the event.

Judge ONLY from the evidence shown.
{_HEALTH_RULE}
{{"signal_found": true/false,
 "signal": "<one clause with numbers at each offset>",
 "lead_time": "<e.g. about one month, with the per-period changes>",
 "baseline_compared": true/false,
 "next_step": "<one step, or empty string if complete>"}}
"""

CRITIC_PREDICTIVE_FLAG = f"""You review an investigation into what predicts
an event recorded as a FLAG on each row.

There is no timeline here and no lead time. The question is which attributes
separate the rows where the event happened from the rows where it did not.

  signal_found       - true if at least one SINGLE attribute, OR one of the
                       COMBINATION buckets (an attribute crossed with a
                       derived bucket, e.g. "emergency + short stay"), shows
                       a clear separation: its rate sits well above the base
                       rate and the complementary bucket well below. A
                       combination bucket counts fully as a predictor - it
                       is often the ONLY place the signal is visible, since
                       each attribute alone can look flat while the
                       combination is not. A spread of a point or two
                       around the base rate is noise.
  signal             - the strongest predictor in one clause, WITH the rate
                       at each end AND the base rate to compare against.
                       Name the attribute or combination, not just the
                       value. If a combination bucket is the strongest,
                       name BOTH conditions that define it, e.g. "emergency
                       admissions with a short stay: 25.2% vs 8.9% base".
  lead_time          - not applicable to a flag. Return "n/a".
  baseline_compared  - true only if SEVERAL attributes were tested and can be
                       ranked against each other. Testing ONE attribute and
                       finding nothing is not a finding about the question -
                       it is a finding about that one attribute, and the
                       answer must not generalise from it.

Judge ONLY from the evidence shown.
{_HEALTH_RULE}
{{"signal_found": true/false,
 "signal": "<strongest predictor, rates at each end, base rate>",
 "lead_time": "n/a",
 "baseline_compared": true/false,
 "next_step": "<one step, or empty string if complete>"}}
"""


def _critic_diagnostic(parsed: dict, healthy: bool, metric_kind: str,
                       revisions: int) -> tuple[dict, dict, str]:
    pattern = str(parsed.get("pattern", "")).strip().lower()
    segment = str(parsed.get("top_segment", "")).strip()
    try:
        share = abs(float(parsed.get("share_of_change_pct") or 0))
    except (TypeError, ValueError):
        share = 0.0
    mechanism = str(parsed.get("mechanism", "")).strip()
    try:
        mscale = abs(float(parsed.get("mechanism_scale_pct") or 0))
    except (TypeError, ValueError):
        mscale = 0.0

    # The decision is made here, not by the model. Letting it declare
    # "uniform" as a form of localization is how an answer calling a
    # 114%-APAC drop "system-wide" once passed this check.
    if pattern == "uniform":
        localized = share < LOCALIZED_SHARE
    else:
        localized = bool(segment) and share >= LOCALIZED_SHARE

    # A mediocre localization is worth one attempt at a better dimension.
    # Oncology at 57% passed this check while payer_type held the real
    # answer at 113%, because the floor is a minimum with nothing above it.
    weak = localized and share < WEAK_LOCALIZATION
    checklist = {
        "segment_localized": localized and healthy and not (weak and revisions == 0),
        "mechanism_identified": (bool(parsed.get("mechanism_identified"))
                                 and bool(mechanism) and healthy
                                 and mscale >= MECHANISM_SCALE),
    }
    findings = {"kind": "DIAGNOSTIC", "metric_kind": metric_kind,
                "pattern": "uniform" if (pattern == "uniform" and localized)
                           else "localized",
                "top_segment": segment, "share_of_change_pct": round(share, 1),
                "weak_localization": weak,
                "mechanism": mechanism, "mechanism_scale_pct": round(mscale, 1),
                "evidence_healthy": healthy}
    log = f"({segment or 'none'} @ {share:.0f}%{' WEAK' if weak else ''}, mech {mscale:.0f}%)"
    return checklist, findings, log


def _critic_predictive(parsed: dict, healthy: bool,
                       event_kind: str) -> tuple[dict, dict, str]:
    signal = str(parsed.get("signal", "")).strip()
    lead = str(parsed.get("lead_time", "")).strip()
    found = bool(parsed.get("signal_found")) and bool(signal)
    compared = bool(parsed.get("baseline_compared"))

    checklist = {
        "signal_found": found and healthy,
        "baseline_compared": compared and healthy,
    }
    findings = {"kind": "PREDICTIVE", "event_kind": event_kind,
                "signal_found": found, "signal": signal, "lead_time": lead,
                "baseline_compared": compared, "evidence_healthy": healthy}
    return checklist, findings, f"(signal={'yes' if found else 'no'}, {event_kind})"


# A critic that has decided the work is finished sometimes writes that
# verdict INTO next_step. Run as a step, it becomes a query with no clear
# instruction - which once dropped the segment filter and measured the
# whole hospital, and those numbers were then reported as oncology's.
_CONCLUSION = re.compile(
    r"(?i)(investigation is complete|no further (?:steps?|analysis|investigation)"
    r"|(?:is|are) not needed|nothing (?:more|further) to|premise .{0,40}not supported"
    r"|did not occur|cannot be found|no cause to find)")


def _is_conclusion(text: str) -> bool:
    return bool(text) and bool(_CONCLUSION.search(text))


def critic(state: AgentState) -> dict:
    if state["qtype"] == "LOOKUP":
        return {"verdict": "PASS", "critique": "", "findings": {}}

    budget = revision_budget(state["results"])
    if state["revisions"] >= budget:
        print(f"\n[CRITIC] PASS (revision budget {budget} spent)")
        return {"verdict": "PASS", "critique": "revision limit reached"}

    predictive = state["qtype"] == "PREDICTIVE"
    event_kind = state.get("event_kind") or "FLAG"
    metric_kind = state.get("metric_kind") or "SUM"

    if predictive:
        system = (CRITIC_PREDICTIVE_DATED if event_kind == "DATED"
                  else CRITIC_PREDICTIVE_FLAG)
        preamble = f"EVENT KIND: {event_kind}\n\n"
    else:
        system = CRITIC_DIAGNOSTIC
        preamble = KIND_RULE.get(metric_kind, KIND_RULE["SUM"]) + "\n\n"

    already = "\n".join(f"  - {r['step']}" for r in state["results"])
    prompt = (f"{preamble}QUESTION: {state['question']}\n\n"
              f"STEPS TAKEN:\n{already}\n\n"
              f"EVIDENCE:\n{evidence_text(state['results'])}")
    health = evidence_health(state["results"])
    if health:
        prompt += f"\n\n{health}"

    raw = chat(prompt, system=system, role="critic")
    parsed = _json_from(raw)
    healthy = evidence_healthy(state["results"])

    if predictive:
        checklist, findings, log = _critic_predictive(parsed, healthy, event_kind)
    else:
        dc = state.get("decomp") or {}
        if dc:
            # The segment and its share were computed. The model judges the
            # mechanism; it does not get to re-estimate the arithmetic.
            share, size = dc.get("share_of_change_pct", 0.0), dc.get("size_pct", 0.0)
            parsed["top_segment"] = (f"{dc['top_dimension']} = {dc['top_segment']}"
                                     if dc.get("top_segment") else "")
            parsed["share_of_change_pct"] = share
            parsed["pattern"] = ("localized" if dc.get("top_segment") and not dc.get("flat")
                                 and share >= LOCALIZED_SHARE and share - size >= 20
                                 else "uniform")
        checklist, findings, log = _critic_diagnostic(
            parsed, healthy, metric_kind,
            # every dimension was already tested: no revision to try another
            1 if dc else state["revisions"])
        if dc:
            findings.update(flat=bool(dc.get("flat")), size_pct=dc.get("size_pct", 0.0),
                            top_segment=parsed["top_segment"])
            if dc.get("flat"):
                checklist["segment_localized"] = False

    next_step = str(parsed.get("next_step", "")).strip()
    if is_sql(next_step) or _is_conclusion(next_step):
        next_step = ""
    # A computed sweep has already tested every attribute over every row.
    # A model-written follow-up can only test less, on a guessed slice.
    if state["results"] and all(r.get("computed") and r.get("ok")
                                for r in state["results"]):
        next_step = ""

    complete = all(checklist.values())
    print(f"\n[CRITIC:{state['qtype']}] {checklist} {log} "
          f"evidence={'ok' if healthy else 'BROKEN'}"
          + ("" if complete else f" -> {next_step[:80]}"))

    if complete or not next_step:
        return {"verdict": "PASS", "critique": raw,
                "checklist": checklist, "findings": findings}

    return {"verdict": "REVISE", "critique": raw, "checklist": checklist,
            "findings": findings, "plan": state["plan"] + [next_step],
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

WHAT WAS ASKED, AND WHAT WAS MEASURED
- If CONCEPT RESOLUTION lists anything ABSENT, say so in **Answer** before
  anything else, then give the nearest available breakdown explicitly
  labelled as a substitute. "The data has no campaign field; the closest is
  acquisition channel, and by channel inbound leads with..." is an answer.
  Reporting channels as though they were campaigns is not - the SQL is right
  and the data is real, but the question has been quietly changed.
- If a metric was COMPUTED rather than read from a column, give its UNIT in
  **Answer**: "average length of stay, computed from admission to discharge,
  was 4.87 days". A figure whose unit is wrong by a factor of twenty reads
  as perfectly normal otherwise.
- State the time period covered in **Answer** whenever the question did not
  name one. A figure without its window is not a figure.
- If the evidence contradicts the question's premise - it asks why something
  rose and the headline shows it fell or stayed flat - say so plainly in
  **Answer**, with both headline figures. A question can be wrong. Never
  restate a fall as "slowing decline", and never answer about a different
  metric that did move.
  Then, in **Why**, report the segments that DID move the way the question
  assumed, if a successful breakdown shows any, TOGETHER with the segments
  that offset them: "overall it did not rise; orthopaedics did (5.80 to 6.22
  days), but oncology fell (7.06 to 6.40) and the two cancelled out". This
  is context about where the movement is, labelled as segment-level. It is
  not a cause of the headline, because the headline did not move, so never
  present it as one.
- NEVER claim that no segment moved, that a change was "uniform", or that
  nothing drove it, unless a breakdown step SUCCEEDED and its numbers show
  it. If the breakdown failed, say the segment-level picture could not be
  measured. A universal negative with no numbers behind it cannot be
  checked, and it is exactly how an unmeasured breakdown turns into a false
  statement.

WHAT THE EVIDENCE SUPPORTS
- If CHECK STATUS shows every check false, the investigation established
  nothing. Say which question the evidence could not answer. Do NOT state
  that the thing does not exist - "no predictive signal", "no difference
  between the groups" - on the strength of one attribute tested, a query
  that failed, or a result of all zeros. That is not a finding, it is a
  measurement you do not have.
- If a step FAILED, returned zero rows, or returned the same value in every
  row, say so in **Answer** and build no conclusion on it.
- A driver that sums to exactly zero for failed, denied or refunded rows was
  measured with the wrong column. It is not evidence that those rows cost
  nothing.
- Do not name a driver as the cause when its magnitude is a small fraction
  of the change. If every driver tested is far too small, say the cause was
  not identified rather than promoting the largest.
- Do not claim a driver caused the change unless the evidence tested it.

SEGMENTS AND SHARES
- The **Answer** line quotes the figure from the HEADLINE step - the one
  measured with NO breakdown. A segment's own figure NEVER goes on the
  Answer line, however dramatic: "integrations CSAT fell to 2.49" answers a
  question nobody asked and overstates the business-level move. The segment
  belongs in **Why**.
- NEVER write "system-wide", "broad", "across the board" or "general" unless
  the reviewer's pattern is "uniform". Do not write "the drop was
  system-wide" and then name one region in the same sentence - that is a
  contradiction, and the region is the finding.
- A segment's share of the change belongs to the SEGMENT, not to the driver.
  If you state a share for the mechanism, it must be the mechanism's own.
- Report the segment at the level the reviewer named it. Do not narrow it to
  one sub-cell and present the sub-cell as the cause.
- A figure belongs inside a sentence about a segment ONLY if it was measured
  for that segment. If the driver evidence is business-wide, say so plainly.
- A driver explaining a CHANGE must itself be quoted as a change across the
  same two periods, never as a level in one.

COMPUTED DECOMPOSITION (a BREAKDOWN step computed in code)
- It tested every dimension. Lead **Why** with the reviewer's segment, its
  share_of_change_pct, and its size_pct: "orthopaedics carried 87.9% of the
  drop while holding 17.6% of the responses". Never call the change uniform
  or spread out when that share is far above that size.
- The CONTRAST step compares the segment with everywhere else. Its top rows
  are the candidate mechanism: quote inside_p1 -> inside_p2 against
  outside_p1 -> outside_p2, e.g. "discharge delay in orthopaedics rose from
  7.44 to 24.09 hours while it held at about 7.1 elsewhere". A shift that
  happened inside the segment and nowhere else is the strongest evidence of
  cause this data can give.
- When the reviewer says the TOTAL BARELY MOVED, lead with that, then name
  the segments that moved in opposite directions, with their figures.

PREDICTIVE ANSWERS
- If the evidence is a computed ATTRIBUTE SWEEP, it is complete: every
  attribute was tested over every row in scope. Lead with the attribute at
  the top of the ranking and its rate at each end, against the base rate,
  with the row counts. Then the strongest COMBINATION. Describe buckets in
  plain words: "Q1 lowest 25%" of "days from admission_ts to discharge_ts,
  vs its department norm" is "the shortest quarter of stays for their
  department". Say an attribute made little difference only if its
  spread_pts in the table is small.
- Write every rate as a percentage.
- Between two combinations with similar rates, lead with the one with more
  rows - it is the more reliable finding.
- For a DATED event, **Answer** says whether a signal exists and how much
  lead time it gives, with the numbers. Lead time is where the metric
  BREAKS, not how far back the table reaches.
- For a FLAG event there is no lead time. **Answer** names the strongest
  predictor with its rate at each end and the base rate, and says which
  other attributes were compared against it.

NUMBERS AND STYLE
- Format money as Rs X.XX M for millions, or Rs X,XXX otherwise. NEVER use
  scientific notation, and never print the same figure twice in two formats.
- Write magnitudes as POSITIVE numbers with a direction word: "fell by
  Rs 1.46 M", never "dropped by Rs -1,459,117.6".
- Check any percentage against the two numbers it comes from: a percentage
  change is (new - old) / old x 100. Every one will be recomputed.
- Use ONLY numbers that appear in the evidence. Never estimate.
- Every direction word must match its own numbers: "rose from 4.35 to 4.20"
  is wrong. Check each one before writing it, and never leave a
  self-correction ("wait", "actually") in the answer - write only the
  final version.
- RANK the drivers. Mention a minor factor only to DISMISS it.
- For a simple lookup, keep **Why** to one line and omit **Caveat**
  entirely. Never say a lookup "identified no causal drivers" - it did not
  look for any, and was not asked to.
- Be concise. No filler.
"""


def check_grounding(answer: str, results: list[dict]) -> dict:
    """Verify every number in the answer appears in the evidence, or is a
    legitimate derivation from it.

    Derivations are computed WITHIN a single result table, never across
    tables. Pooling every number from every step produced ~200 pairwise
    differences spread across the whole range, so almost any figure found a
    "match" - which is how a stated -14% decline passed while the true
    figure was -43.7%.
    """
    pools = [numbers_in(r["table"]) for r in results]
    pool = [n for p in pools for n in p]

    derived: set[float] = set()
    for p in pools:
        operands = [v for v in p if not _skip_number(v)]
        for a in operands:
            for b in operands:
                if a == b:
                    continue
                derived.add(round(abs(a - b), 4))                  # difference
                if b:
                    derived.add(round(abs((a - b) / b * 100), 4))  # % change

    # A segment's share of the change: its change (any table) over the
    # headline's change (the FIRST table only). Narrow on purpose - pooling
    # every table's differences once let a made-up -14% find a match - but
    # without it a correct "112.8% of the total change" is always flagged.
    if pools:
        head = [v for v in pools[0] if not _skip_number(v)]
        denoms = {round(abs(a - b), 4) for a in head for b in head if a != b}
        numers = {round(abs(a - b), 4) for p in pools
                  for a in p for b in p if a != b
                  and not _skip_number(a) and not _skip_number(b)}
        for d in denoms:
            if d:
                for n in numers:
                    derived.add(round(n / d * 100, 4))

    def tol(p: float) -> float:
        return max(abs(p) * 0.02, 0.005)

    def dtol(d: float) -> float:
        return max(abs(d) * 0.005, 0.01)   # a derivation is arithmetic: tight

    unverified = []
    for n in numbers_in(answer):
        if _skip_number(n):
            continue
        hit = any(
            abs(n - p) <= tol(p)
            or abs(n * 1e6 - p) <= tol(p)
            or abs(n * 1e3 - p) <= tol(p)
            or abs(n / 100 - p) <= tol(p)
            for p in pool
        )
        if not hit:
            # The same magnitude ladder as the pool check. Without it
            # "Rs 19.42 M" can never match a derived difference of
            # 19,415,045. No divide-by-100 here: percentage changes are
            # already in `derived` explicitly.
            scaled = (abs(n), abs(n) * 1e6, abs(n) * 1e3)
            hit = any(abs(c - d) <= dtol(d) for d in derived for c in scaled)
        if not hit:
            unverified.append(n)
    return {"checked": True, "unverified": unverified, "clean": not unverified}


# Claims that something does NOT exist. They carry no number, so the
# grounding check cannot touch them - and when the step that would have
# measured them failed, they are pure invention. "No segment drove an
# increase" was written about a breakdown that never ran, while four of
# seven departments had in fact risen.
_UNIVERSAL_NEGATIVE = re.compile(
    r"(?i)\b("
    r"no (?:single |one |clear |particular |specific )?"
    r"(?:segment|department|region|category|attribute|channel|group|"
    r"driver|cause|signal|predictor|difference|pattern)s?"
    r"|uniform(?:ly)?"
    r"|across the board"
    r"|system-wide"
    r"|nothing (?:drove|explains|caused|predicts)"
    r"|none of the (?:segments|departments|regions|categories|attributes)"
    r")\b")


def unsupported_negatives(answer: str, results: list[dict]) -> list[str]:
    """Universal negatives stated on evidence that did not fully run.

    Only fires when some step failed, came back empty, or was broken. On
    healthy evidence a negative is a real finding and the prompt rules
    govern it; on broken evidence it is a measurement that was never taken.
    """
    if evidence_healthy(results):
        return []
    return sorted({m.group(0).lower() for m in _UNIVERSAL_NEGATIVE.finditer(answer)})


_UP = r"(?:rose|risen|increased|grew|climbed|improved|went up|up)"
_DOWN = r"(?:fell|fallen|dropped|declined|decreased|slipped|worsened|went down|down)"
_MOVE = re.compile(
    rf"(?i)\b({_UP}|{_DOWN})\s+(?:by\s+[\d.,]+\s*\S*\s+)?from\s+"
    rf"(?:Rs\.?\s*)?(-?[\d][\d,]*\.?\d*)\s*%?\s*\S{{0,6}}\s+to\s+"
    rf"(?:Rs\.?\s*)?(-?[\d][\d,]*\.?\d*)")
# Any "4.33 to 3.25" pair, with or without "from" or a verb in front of it.
# "Orthopaedics (5.80 to 6.22 days)" states a direction as surely as
# "rose from 5.80 to 6.22" does.
_PAIR = re.compile(
    r"(?<![\w.])(?:Rs\.?\s*)?(-?\d[\d,]*\.?\d*)\s*(?:%|days?|hours?|h|M|K|pts|points)?"
    r"\s*(?:to|->|→)\s*(?:Rs\.?\s*)?(-?\d[\d,]*\.?\d*)(?![\d])")

_SPREAD_WORDS = re.compile(
    r"(?i)\b(uniform(?:ly)?|across (?:all|every)|across the board|system-wide|"
    r"in (?:all|every) (?:segment|department|region|categor|admission|type))")
_THINKING = re.compile(r"(?i)(\bwait\b\s*[:,]|\bactually,?\s+wait\b|\bhmm\b|"
                       r"\bcorrection:|\bI mean\b|\bscratch that\b)")


def prose_errors(answer: str) -> list[str]:
    """Wording that contradicts the answer's own numbers.

    Grounding checks that each number is real. It cannot see "rose from
    4.35 to 4.20", or "uniform" written beside a segment that moved the
    other way - every figure there IS in the evidence."""
    errs, dirs = [], set()
    for m in _MOVE.finditer(answer):
        word = m.group(1).lower()
        try:
            a = float(m.group(2).replace(",", ""))
            b = float(m.group(3).replace(",", ""))
        except ValueError:
            continue
        up = bool(re.fullmatch(_UP, word, re.I))
        if b == a:
            continue
        if up != (b > a):
            errs.append(f'"{m.group(0)}" - {b:g} is {"lower" if up else "higher"} '
                        f'than {a:g}, so the word "{word}" is the wrong direction')
    for m in _PAIR.finditer(answer):
        try:
            a = float(m.group(1).replace(",", ""))
            b = float(m.group(2).replace(",", ""))
        except ValueError:
            continue
        if a == b or _skip_number(a) or _skip_number(b):
            continue                     # "Q1 2026 to Q2 2026" is not a move
        dirs.add(b > a)
    if len(dirs) == 2:
        for m in _SPREAD_WORDS.finditer(answer):
            errs.append(f'"{m.group(0)}" - the answer itself reports segments '
                        f'moving in OPPOSITE directions, so the change was not '
                        f'spread evenly')
    for m in _THINKING.finditer(answer):
        errs.append(f'"{m.group(0).strip()}" - working-out text left in the answer')
    return errs


def _findings_text(findings: dict) -> str:
    if not findings:
        return ""
    lines = ["REVIEWER FINDINGS (verified against the evidence):"]
    if not findings.get("evidence_healthy", True):
        lines.append("  evidence is INCOMPLETE - at least one measurement "
                     "failed or returned nothing. Report what could not be "
                     "measured; do not conclude that nothing is there.")

    if findings.get("kind") == "PREDICTIVE":
        flag = findings.get("event_kind") == "FLAG"
        if findings.get("signal_found"):
            lines.append(f"  predictor: {findings.get('signal', '')}")
            if not flag and findings.get("lead_time"):
                lines.append(f"  lead time: {findings['lead_time']} - state "
                             f"this in the Answer, and do not widen it to the "
                             f"full span of the query.")
            if flag:
                lines.append("  this event is a FLAG: there is no lead time "
                             "and no timeline. Do not mention either.")
        else:
            lines.append("  no separation was detected. Say so ONLY if the "
                         "measurement succeeded AND several attributes were "
                         "compared.")
        if not findings.get("baseline_compared"):
            lines.append("  only a narrow comparison was made - other "
                         "attributes or a control group were NOT tested. Say "
                         "the question was not adequately tested rather than "
                         "implying nothing predicts it.")
        return "\n".join(lines)

    if findings.get("flat"):
        lines.append("  the TOTAL BARELY MOVED. Do not explain a change in it. "
                     "Report the segments that moved in opposite directions and "
                     "cancelled out, and, if the CONTRAST shows one, what changed "
                     f"inside the segment that moved most ({findings.get('top_segment', '')}).")
        return "\n".join(lines)
    if findings.get("pattern") == "uniform":
        lines.append("  pattern: UNIFORM - the change is spread across all "
                     "segments. Describing it as system-wide is correct.")
    else:
        lines.append("  pattern: LOCALIZED - the words 'system-wide', 'broad' "
                     "and 'across the board' are FORBIDDEN in this answer.")
        if findings.get("top_segment"):
            lines.append(
                f"  largest contributor: {findings['top_segment']} "
                f"({findings.get('share_of_change_pct', 0)}% of the total "
                f"change). Lead the Why with this segment. This share is the "
                f"SEGMENT's, not any driver's.")
        if findings.get("weak_localization"):
            lines.append("  this localization is WEAK - other segments moved "
                         "the same way. Say the change is concentrated here "
                         "but not confined to it.")
    if findings.get("mechanism"):
        scale = findings.get("mechanism_scale_pct", 0)
        if scale >= MECHANISM_SCALE:
            lines.append(f"  mechanism: {findings['mechanism']} "
                         f"(accounts for about {scale}% of the change)")
        else:
            lines.append(f"  mechanism NOT established - the only driver "
                         f"tested ({findings['mechanism']}) accounts for about "
                         f"{scale}% of the change, far too little to explain "
                         f"it. Say the cause was not identified; do not "
                         f"promote this driver to the cause.")
    return "\n".join(lines)


def synthesizer(state: AgentState) -> dict:
    ev = evidence_text(state["results"])
    prompt = f"QUESTION: {state['question']}\n\nEVIDENCE:\n{ev}"

    con = concepts_text(state.get("concepts") or [])
    if con:
        prompt += f"\n\n{con}"

    health = evidence_health(state["results"])
    if health:
        prompt += f"\n\n{health}"

    chk = state.get("checklist") or {}
    if chk:
        prompt += ("\n\nCHECK STATUS: "
                   + ", ".join(f"{k}={v}" for k, v in chk.items()))

    clar = state.get("clarifications") or {}
    if clar:
        answers = "; ".join(f"{k}: {v}" for k, v in clar.items())
        prompt += f"\n\nUSER CLARIFICATIONS APPLIED: {answers}"

    found = _findings_text(state.get("findings") or {})
    if found:
        prompt += f"\n\n{found}"
    elif state.get("critique"):
        prompt += f"\n\nREVIEWER NOTES:\n{state['critique']}"

    answer = chat(prompt, system=SYNTH_SYSTEM, role="synthesize")
    grounding = check_grounding(answer, state["results"])

    tries = 0
    while not grounding["clean"] and tries < MAX_GROUND_ATTEMPTS:
        tries += 1
        bad = ", ".join(f"{n:,.2f}" for n in grounding["unverified"])
        print(f"\n[GROUNDING] unverified: {bad} - regenerating ({tries})")
        answer = chat(
            f"{prompt}\n\nYour previous answer contained numbers NOT present "
            f"in the evidence and not derivable from it: {bad}\n"
            f"These usually come from adding up the rows of a breakdown "
            f"table, or from a percentage computed against the wrong base.\n"
            f"Rewrite using only figures that appear above, or arithmetic on "
            f"two figures from the SAME table. Recompute every percentage as "
            f"(new - old) / old x 100. If no step computed a figure, do not "
            f"state one.",
            system=SYNTH_SYSTEM, role="synthesize")
        grounding = check_grounding(answer, state["results"])

    grounding["regenerated"] = tries

    negatives = unsupported_negatives(answer, state["results"])
    if negatives:
        said = ", ".join(f'"{n}"' for n in negatives)
        print(f"\n[GROUNDING] unsupported negative(s) on broken evidence: {said}")
        answer = chat(
            f"{prompt}\n\nYour previous answer:\n{answer}\n\n"
            f"It claims {said}, but at least one measurement FAILED or "
            f"returned nothing - see EVIDENCE HEALTH. A claim that something "
            f"does not exist cannot rest on a measurement that did not run.\n"
            f"Rewrite it. Keep every figure that IS in the evidence. Where "
            f"the failed step would have answered a question, say that "
            f"question could not be measured, instead of stating a result.",
            system=SYNTH_SYSTEM, role="synthesize")
        grounding = {**check_grounding(answer, state["results"]),
                     "regenerated": tries + 1}
        negatives = unsupported_negatives(answer, state["results"])
        if negatives:
            answer += ("\n\n> **Unsupported:** this answer states "
                       + ", ".join(f'"{n}"' for n in negatives)
                       + " while at least one step failed to measure "
                         "anything. Treat those statements as unverified.")
    grounding["unsupported_negatives"] = negatives

    wording = prose_errors(answer)
    if wording:
        print(f"\n[GROUNDING] wording contradicts its own numbers: {wording}")
        answer = chat(
            f"{prompt}\n\nYour previous answer:\n{answer}\n\n"
            f"Its WORDS contradict its own numbers:\n"
            + "\n".join(f"  - {w}" for w in wording)
            + "\nRewrite it cleanly. Every number stays; fix each direction "
              "word to match its numbers, drop any claim that a change was "
              "uniform when segments moved in opposite directions, and remove "
              "any self-correction. Write only the final answer.",
            system=SYNTH_SYSTEM, role="synthesize")
        grounding = {**check_grounding(answer, state["results"]),
                     "regenerated": grounding.get("regenerated", 0) + 1,
                     "unsupported_negatives": negatives}
        wording = prose_errors(answer)
        if wording:
            answer += ("\n\n> **Wording check:** "
                       + "; ".join(wording)
                       + ". The figures are correct; read the direction from them.")
    grounding["wording_errors"] = wording

    if not grounding["clean"]:
        bad = ", ".join(f"{n:,.2f}" for n in grounding["unverified"])
        answer += (f"\n\n> **Unverified:** {bad} could not be matched to any "
                   f"query result. Treat these figures as unreliable - the "
                   f"rest of the answer is grounded in the evidence below.")

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


_BLANK = {"qtype": "", "metric_kind": "", "event_kind": "", "concepts": [],
          "scope": "", "periods": [], "time_column": "", "event": {},
          "metric_row": "", "decomp": {},
          "plan": [], "results": [], "checklist": {}, "findings": {},
          "critique": "", "verdict": "", "revisions": 0, "answer": "",
          "grounding": {}}


def _empty_state(question: str, clarifications: dict, pending: list) -> dict:
    return {**_BLANK, "question": question, "clarifications": clarifications,
            "needs_clarification": pending, "cached": ""}


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
            return {**_BLANK, **hit, "cached": how, "needs_clarification": []}

    state = build_agent().invoke({**_BLANK, "question": question,
                                  "clarifications": clarifications})
    state["cached"] = ""
    state["needs_clarification"] = []

    # Never cache a run built on broken evidence - one bad run would
    # otherwise poison every future ask of that question. A step that
    # returned nothing counts as broken, and so does an ungrounded answer.
    all_ok = evidence_healthy(state.get("results", []))
    grounded = state.get("grounding", {}).get("clean", False)

    if use_cache and state.get("answer") and all_ok and grounded:
        qcache.store(question, fp, {
            "question": state["question"], "qtype": state["qtype"],
            "metric_kind": state.get("metric_kind", ""),
            "event_kind": state.get("event_kind", ""),
            "concepts": state.get("concepts", []),
            "scope": state.get("scope", ""),
            "periods": state.get("periods", []),
            "time_column": state.get("time_column", ""),
            "plan": state["plan"], "results": state["results"],
            "checklist": state["checklist"], "findings": state.get("findings", {}),
            "critique": state["critique"],
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
    empty = [r for r in state.get("results", []) if r.get("empty")]
    if failed or empty:
        print(f"\n[WARNING] {len(failed)} step(s) failed, {len(empty)} "
              f"returned nothing - answer rests on partial evidence")

    g = state.get("grounding", {})
    total_tok = USAGE["prompt_tokens"] + USAGE["output_tokens"]
    print(f"\n[STATS] {state.get('qtype')}/{state.get('metric_kind')} "
          f"| {len(state.get('results', []))} steps "
          f"| {USAGE['calls']} LLM calls | {total_tok:,} tokens | {elapsed:.1f}s")
    print(f"[MODELS] {dict(USAGE['by_model'])}")
    print(f"[CONCEPTS] {state.get('concepts')}")
    print(f"[CHECKLIST] {state.get('checklist')}")
    print(f"[FINDINGS]  {state.get('findings')}")
    print(f"[GROUNDING] {'clean' if g.get('clean') else g}")
    if state.get("cached"):
        print(f"[CACHE] served from cache: {state['cached']}")