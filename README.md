# InsightPilot

An autonomous analytics agent. Ask a business question in plain English;
it plans the analysis, writes and runs SQL, checks its own work, and
returns a root-cause answer with the evidence attached.

Not "chat with your CSV". This investigates.

---

## What it does

**Lookup question** — one query, a few seconds:

> **Q:** Which payment type is used most often?
> **A:** UPI, with 11,443 payments.

**Diagnostic question** — the agent plans, drills down, and self-critiques:

> **Q:** Why did revenue drop in Q3 2025?
>
> **Answer:** Revenue fell by Rs 2.63 M, from Rs 19.60 M in Q2 2025 to
> Rs 16.97 M in Q3 2025.
>
> **Why:** The decline was driven primarily by a volume problem — order
> count fell by 353 (from 4,468 to 4,115). The largest single contributor
> was the South × Electronics segment, which fell by Rs 1.46 M (from
> Rs 3.20 M to Rs 1.74 M), accounting for over half the total drop.
>
> **Caveat:** The data shows correlation, not proof of causation.

It gets there by running four queries it chose itself, rejecting two
malformed ones before execution, and being sent back by its own critic
for a segment-level breakdown.

---

## Architecture

```
question
   │
   ├─ ambiguity check ──► asks the user if a term is undefined
   │                      ("revenue" — net of freight, or GMV?)
   ├─ semantic cache ───► exact + fuzzy match, zero API calls on a hit
   │
   ▼
 PLANNER ──► classifies LOOKUP vs DIAGNOSTIC, emits 1-3 independent steps
   │
   ▼
 EXECUTOR ─► steps run in parallel
   │         each step: generate SQL ─► VALIDATE ─► execute
   │                          ▲              │
   │                          └── retry with a STRONGER model
   ▼
 CRITIC ───► metric-aware checklist:
   │         · is the change localized to a segment?
   │         · is the mechanism identified?
   │         not both ─► send back to EXECUTOR with one more step
   ▼
 SYNTHESIZER ─► executive answer
   │
   └─ GROUNDING ─► every number verified against the evidence,
                   regenerate if the model invented one
```

### Design decisions

**Semantic layer.** Metric definitions live in YAML, not in the model's
head. `revenue = SUM(order_items.price) WHERE status <> 'canceled'` —
so the same question always returns the same number. Without this, an LLM
will define "revenue" differently on Tuesday than it did on Monday.

**Validation before execution.** DuckDB's `EXPLAIN` binds a query without
running it, turning an invented column into a precise, cheap correction
(`column "revenue" not found — did you mean "price"?`) instead of a
runtime failure.

**Escalating retries.** The first SQL attempt uses a small fast model.
Retries escalate to a larger one, because asking the same weak model to
fix its own mistake usually reproduces the mistake.

**Metric-aware critic.** "Mechanism identified" means something different
per metric. Revenue decomposes into volume / price / cancellations;
review scores decompose into delivery lateness. An earlier version applied
the revenue decomposition to a review-score question and confidently
blamed cancellations — which cannot affect reviews in this data, since
only delivered orders get reviewed.

**Numeric grounding.** After synthesis, every number in the answer is
checked against the evidence tables (allowing simple derived values such
as a difference of two figures). Unverified numbers trigger a regeneration.
A prompt instruction not to hallucinate is a request; this is a guarantee.

**Human in the loop.** The semantic layer records ambiguous business terms.
The agent asks *before* computing, and the answer becomes part of the
cache key, so "all orders" and "delivered only" never share a result.

---

## Benchmark

20 questions across four tiers, graded against reference SQL computed at
runtime. Baseline is single-shot text-to-SQL with no planning or critique.

| | Agent | Baseline |
|---|---|---|
| Overall accuracy | **86.7%** (80–95%, σ 7.6, 3 runs) | 70% (1 run) |
| Lookup / comparative | 13–15 / 16 | 13 / 16 |
| **Diagnostic (root cause)** | **10 / 12 trials** | **1 / 4** |
| Median latency | ~25s | ~10s |
| Tokens per query | ~11k | ~2.3k |

The agent matches the baseline on lookups and is substantially better on
root-cause questions — which is the only place the extra cost is justified.

### On variance

A single run of a multi-step LLM agent is a sample of size one. Across
three runs, accuracy ranged 80–95%. Tracing it: the spread came mostly
from **model fallback rotation** under free-tier rate limits — a different
model answered each run — rather than from sampling noise. Set
`LLM_PIN_MODEL` to fix one model for reproducible measurement.

Per-question pass rates separate **flaky** questions (agent instability)
from **always fails** (a missing capability). They need different fixes.

```bash
python evals/run_evals.py --mode agent --no-cache --runs 3
python evals/run_evals.py --mode baseline --no-cache
```

---

## Works on your own data

```bash
python tools/ingest.py sales.csv customers.xlsx --name acme
python tools/semantic.py
python agent/graph.py "Why did margin fall last quarter?"
```

Ingestion reads CSV / TSV / Excel (all sheets) / JSON / Parquet, fixes
column names, strips currency symbols, parses mixed date formats, handles
four encodings, and profiles every column **in DuckDB over the full table**
— not a pandas sample, because sampling produces false uniqueness and a
false key means a wrong join.

It then reports data-quality issues (duplicates, high-null columns,
constant columns, negative quantities) rather than silently "cleaning"
them. Dropping rows changes every number downstream; that is the user's
call, not the tool's.

`tools/semantic.py` drafts a semantic layer from the profile, **validates
every inferred metric against the real catalog**, discards the ones that
don't execute, and writes the rest for the user to review. The draft is a
proposal, not a fact.

---

## Setup

```bash
git clone <repo> && cd insightpilot
python -m venv venv && venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

`.env`:

```
GROQ_API_KEYS=key1,key2
GROQ_MODELS_FAST=openai/gpt-oss-20b
GROQ_MODELS_SMART=openai/gpt-oss-120b,qwen/qwen3.8-27b

GOOGLE_API_KEYS=key1,key2
MODELS_CHEAP=gemini-3.5-flash-lite,gemini-3.1-flash-lite
MODELS_SMART=gemini-3.5-flash,gemini-2.5-flash

LLM_RPM=25
LLM_TIMEOUT=45
LLM_MODE=live
```

Generate the demo dataset and ask a question:

```bash
python data/generate_data.py
python agent/graph.py "Why did revenue drop in Q3 2025?"
```

The demo data is seeded, so the numbers are identical on every machine.
It contains deliberately planted business problems — a segment-specific
revenue collapse, a freight cost spike, and a delivery failure that drags
review scores — so the agent has real causes to find and the benchmark
has verifiable ground truth.

---

## Known limitations

- **Run-to-run variance.** 80–95% across three runs. Pin a model for
  reproducibility; the variance is provider rotation, not sampling.
- **Answer phrasing is occasionally inconsistent.** One run described a
  6% change as "essentially flat" while correctly ranking it second.
- **Inferred semantic layers are guesses.** On uploaded data, accuracy
  depends on whether the LLM guessed your business definitions correctly.
  The demo dataset's hand-written layer is what the benchmark measures.
- **Large schemas are untested.** The full schema goes into every prompt;
  a 200-table database would need retrieval over the schema instead.
- **Free-tier rate limits dominate latency.** Most of the wall-clock time
  is waiting, not computing.

---

## Stack

DuckDB · LangGraph · Groq (gpt-oss, qwen) · Gemini · pandas · PyYAML

## Layout

```
agent/      llm.py (providers, routing, rotation)
            sql_tool.py (text-to-SQL, validation, self-healing)
            graph.py (the agent graph)
tools/      warehouse.py (schema context, guarded SQL, ambiguity)
            validator.py (pre-execution SQL binding)
            ingest.py (any file → profiled DuckDB dataset)
            semantic.py (infer + validate a semantic layer)
            cache.py (exact + fuzzy result cache)
            dataset.py (which dataset is active)
evals/      questions.yml, run_evals.py, results.csv
data/       generate_data.py (seeded demo dataset)
config/     semantic_layer.yml (demo metric definitions)
```