"""
Provider-agnostic LLM layer.

Supports Groq (fast open models) and Gemini, with:
  - role-based routing: fast models + low reasoning effort for mechanical
    steps, stronger models for judgement and prose
  - rotation across providers, keys and models when one is exhausted
  - sliding-window rate limiting per key
  - hard request timeout
  - usage tracking, and offline mock mode (LLM_MODE=mock)

Groq is tried first because latency dominates: this agent makes 7-13
sequential calls, so per-call speed matters more than raw model quality
for the mechanical steps.
"""
import os, time, random, threading, json, hashlib
from collections import defaultdict, deque
from dotenv import load_dotenv

load_dotenv()

MODE = os.getenv("LLM_MODE", "live").lower()
RPM = int(os.getenv("LLM_RPM", "25"))
REQUEST_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "60"))


def _split(name: str, default: str = "") -> list[str]:
    return [x.strip() for x in os.getenv(name, default).split(",") if x.strip()]


# ------------------------------------------------------------------ keys
GROQ_KEYS = _split("GROQ_API_KEYS", os.getenv("GROQ_API_KEY", ""))
GEMINI_KEYS = _split("GOOGLE_API_KEYS", os.getenv("GOOGLE_API_KEY", ""))

# ---------------------------------------------------------------- models
GROQ_FAST = _split("GROQ_MODELS_FAST", "openai/gpt-oss-20b")
GROQ_SMART = _split("GROQ_MODELS_SMART", "openai/gpt-oss-120b,qwen/qwen3.8-27b")
GEMINI_CHEAP = _split("MODELS_CHEAP", "gemini-3.5-flash-lite,gemini-3.1-flash-lite")
GEMINI_SMART = _split("MODELS_SMART", "gemini-3.5-flash,gemini-2.5-flash")


def _chain(*groups) -> list[tuple[str, str]]:
    """Flatten (provider, [models]) groups into an ordered candidate list."""
    out = []
    for provider, models in groups:
        for m in models:
            out.append((provider, m))
    return out


# Order matters: first entry is tried first.
ROLE_MODELS = {
    # mechanical and well-constrained -> fastest model wins
    "plan": _chain(("groq", GROQ_FAST), ("groq", GROQ_SMART),
                   ("gemini", GEMINI_CHEAP), ("gemini", GEMINI_SMART)),
    "sql":  _chain(("groq", GROQ_FAST), ("groq", GROQ_SMART),
                   ("gemini", GEMINI_CHEAP), ("gemini", GEMINI_SMART)),
    # judgement and user-facing prose -> stronger models
    "critic": _chain(("groq", GROQ_SMART), ("gemini", GEMINI_SMART),
                     ("groq", GROQ_FAST), ("gemini", GEMINI_CHEAP)),
    "synthesize": _chain(("groq", GROQ_SMART), ("gemini", GEMINI_SMART),
                         ("groq", GROQ_FAST), ("gemini", GEMINI_CHEAP)),
    "default": _chain(("groq", GROQ_FAST), ("gemini", GEMINI_CHEAP)),
}

# reasoning budget per role - "low" stops a reasoning model over-thinking
# a mechanical task, which is where most of the latency goes
EFFORT = {"plan": "low", "sql": "low", "critic": "medium",
          "synthesize": "medium", "default": "low"}

KEYS_BY_PROVIDER = {"groq": GROQ_KEYS, "gemini": GEMINI_KEYS}

if MODE == "live" and not (GROQ_KEYS or GEMINI_KEYS):
    raise RuntimeError("No API keys: set GROQ_API_KEYS or GOOGLE_API_KEYS in .env")

USAGE = {"calls": 0, "prompt_tokens": 0, "output_tokens": 0, "seconds": 0.0,
         "by_role": defaultdict(lambda: {"calls": 0, "tokens": 0}),
         "by_model": defaultdict(int)}

_dead: set[tuple[str, str, str]] = set()     # (provider, key, model)
_lock = threading.Lock()
_calls = defaultdict(deque)                  # key -> recent call timestamps
_rotate = [0]


class QuotaExhausted(RuntimeError):
    """Every provider/key/model combination is spent."""


def _throttle(key: str):
    """Sliding window: burst freely, wait only when the window is full."""
    with _lock:
        now = time.time()
        window = _calls[key]
        while window and now - window[0] > 60:
            window.popleft()
        if len(window) >= RPM:
            wait = 60 - (now - window[0]) + 0.2
            if wait > 0:
                print(f"  [rpm limit] waiting {wait:.0f}s", flush=True)
                time.sleep(wait)
                now = time.time()
                while window and now - window[0] > 60:
                    window.popleft()
        window.append(now)


def reset_usage():
    USAGE.update({"calls": 0, "prompt_tokens": 0, "output_tokens": 0,
                  "seconds": 0.0})
    USAGE["by_role"] = defaultdict(lambda: {"calls": 0, "tokens": 0})
    USAGE["by_model"] = defaultdict(int)


def _is_daily_quota(msg: str) -> bool:
    m = msg.lower()
    if any(d in m for d in ("perday", "per_day", "per day", "daily limit",
                            "requests_per_day", "tokens per day", "tpd")):
        return True
    if ("quota" in m or "429" in m) and not any(
            k in m for k in ("perminute", "per_minute", "per minute", "rpm")):
        return "retry" not in m
    return False


def _candidates(role: str):
    """Every (provider, key, model) still alive, in preference order."""
    for provider, model in ROLE_MODELS.get(role, ROLE_MODELS["default"]):
        keys = KEYS_BY_PROVIDER.get(provider, [])
        for i in range(len(keys)):
            key = keys[(_rotate[0] + i) % len(keys)]
            if (provider, key, model) not in _dead:
                yield provider, key, model


def budget_status() -> dict:
    return {"groq_keys": len(GROQ_KEYS), "gemini_keys": len(GEMINI_KEYS),
            "dead_combinations": len(_dead)}


# --------------------------------------------------------------- providers
_groq_clients: dict[str, object] = {}


def _call_groq(key: str, model: str, system: str, prompt: str,
               temperature: float, effort: str = "") -> tuple[str, int, int]:
    from groq import Groq
    if key not in _groq_clients:
        _groq_clients[key] = Groq(api_key=key, timeout=REQUEST_TIMEOUT)
    client = _groq_clients[key]

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    kwargs = {"model": model, "messages": messages, "temperature": temperature}
    if effort and "gpt-oss" in model:
        kwargs["reasoning_effort"] = effort

    resp = client.chat.completions.create(**kwargs)
    text = resp.choices[0].message.content or ""
    pt = getattr(resp.usage, "prompt_tokens", 0) or 0
    ct = getattr(resp.usage, "completion_tokens", 0) or 0
    return text.strip(), pt, ct


def _call_gemini(key: str, model: str, system: str, prompt: str,
                 temperature: float, effort: str = "") -> tuple[str, int, int]:
    import google.generativeai as genai
    genai.configure(api_key=key)
    m = genai.GenerativeModel(
        model,
        system_instruction=system or None,
        generation_config={"temperature": temperature},
    )
    resp = m.generate_content(prompt, request_options={"timeout": REQUEST_TIMEOUT})
    pt = ct = 0
    try:
        pt = resp.usage_metadata.prompt_token_count
        ct = resp.usage_metadata.candidates_token_count
    except Exception:
        pass
    return resp.text.strip(), pt, ct


PROVIDERS = {"groq": _call_groq, "gemini": _call_gemini}


# ------------------------------------------------------------- mock engine
MOCK_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        ".cache", "mock")
os.makedirs(MOCK_DIR, exist_ok=True)


def _mock_path(system: str, prompt: str) -> str:
    h = hashlib.sha256((system + "||" + prompt).encode()).hexdigest()[:20]
    return os.path.join(MOCK_DIR, f"{h}.json")


def _record(system: str, prompt: str, response: str) -> None:
    try:
        with open(_mock_path(system, prompt), "w", encoding="utf-8") as f:
            json.dump({"system": system[:200], "prompt": prompt[:400],
                       "response": response}, f)
    except Exception:
        pass


def _mock_response(system: str, prompt: str, role: str) -> str:
    p = _mock_path(system, prompt)
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)["response"]

    if role == "plan":
        q = prompt.lower()
        diag = any(w in q for w in ("why", "caused", "drove", "investigate",
                                    "explain", "spike", "fall", "fell"))
        return json.dumps({
            "type": "DIAGNOSTIC" if diag else "LOOKUP",
            "steps": [prompt.split("Business question:")[-1].strip()[:120]],
        })
    if role == "critic":
        return json.dumps({"segment_localized": True,
                           "mechanism_identified": True, "next_step": ""})
    if role == "sql":
        return "SELECT 1 AS mock_result"
    return ("**Answer:** [mock mode - no LLM called]\n"
            "**Why:** Set LLM_MODE=live in .env to run for real.")


# ------------------------------------------------------------------- chat
def chat(prompt: str, system: str = "", temperature: float = 0.0,
         role: str = "default", max_retries: int = 8) -> str:
    """One-shot completion. temperature=0 because analytics must be reproducible."""
    if MODE == "mock":
        USAGE["calls"] += 1
        USAGE["by_role"][role]["calls"] += 1
        return _mock_response(system, prompt, role)

    effort = EFFORT.get(role, "low")
    tried = 0

    for provider, key, model in _candidates(role):
        if tried >= max_retries:
            break
        tried += 1

        print(f"  [{role} -> {provider}:{model}]", flush=True)
        _throttle(key)
        t0 = time.time()
        try:
            text, pt, ct = PROVIDERS[provider](key, model, system, prompt,
                                               temperature, effort)
        except Exception as e:
            msg = str(e)
            low = msg.lower()

            if ("not found" in low or "404" in msg
                    or "does not exist" in low or "decommissioned" in low):
                with _lock:
                    _dead.add((provider, key, model))
                print(f"  [{model} unavailable] next", flush=True)
                continue

            if "deadline" in low or "timeout" in low or "timed out" in low:
                print(f"  [timeout after {REQUEST_TIMEOUT}s] next", flush=True)
                continue

            if "429" in msg or "quota" in low or "rate" in low or "exhausted" in low:
                if _is_daily_quota(msg):
                    with _lock:
                        _dead.add((provider, key, model))
                        _rotate[0] += 1
                    print(f"  [daily quota spent: {provider}:{model}] rotating",
                          flush=True)
                    continue
                delay = min(20, (2 ** tried) * 2) + random.uniform(0, 1)
                print(f"  [rate limited] waiting {delay:.0f}s", flush=True)
                time.sleep(delay)
                tried -= 1              # a wait is not a failed attempt
                continue

            if "authentication" in low or "api key" in low or "401" in msg:
                with _lock:
                    for _, mm in ROLE_MODELS.get(role, []):
                        _dead.add((provider, key, mm))
                print(f"  [bad {provider} key ...{key[-4:]}] skipping",
                      flush=True)
                continue
            raise

        USAGE["seconds"] += time.time() - t0
        USAGE["calls"] += 1
        USAGE["by_model"][f"{provider}:{model}"] += 1
        USAGE["prompt_tokens"] += pt
        USAGE["output_tokens"] += ct
        USAGE["by_role"][role]["calls"] += 1
        USAGE["by_role"][role]["tokens"] += pt + ct

        _record(system, prompt, text)
        return text

    raise QuotaExhausted(
        f"No available provider/key/model for role '{role}'. "
        f"{budget_status()}. Set LLM_MODE=mock in .env to work offline."
    )


def extract_sql(text: str) -> str:
    """Strip markdown fences and reasoning preambles the model may emit."""
    t = text.strip()
    if "```" in t:
        blocks = t.split("```")
        for b in blocks:
            b = b.strip()
            if b.lower().startswith("sql"):
                return b[3:].strip()
        if len(blocks) > 1:
            return blocks[1].strip()

    # reasoning models sometimes narrate before the query
    upper = t.upper()
    for kw in ("WITH ", "SELECT "):
        i = upper.find(kw)
        if i > 0:
            return t[i:].strip()
    return t