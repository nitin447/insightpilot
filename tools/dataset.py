"""
Which dataset is the agent currently pointed at?

A 'dataset' is a DuckDB file plus optional metadata:
  - semantic.yml : business metric definitions (hand-written or inferred)
  - profile.json : column profiles and detected joins from ingestion

The demo dataset keeps its original hand-written semantic layer; uploaded
datasets start with only a profile until a semantic layer is inferred.
"""
from __future__ import annotations
import os, json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, ".cache")
UPLOADS = os.path.join(ROOT, "data", "uploads")
ACTIVE_FILE = os.path.join(CACHE, "active_dataset.json")

os.makedirs(CACHE, exist_ok=True)
os.makedirs(UPLOADS, exist_ok=True)

DEMO = {
    "name": "demo_ecommerce",
    "kind": "demo",
    "db_path": os.path.join(ROOT, "data", "insightpilot.duckdb"),
    "semantic_path": os.path.join(ROOT, "config", "semantic_layer.yml"),
    "profile_path": "",
}


def get_active() -> dict:
    if os.path.exists(ACTIVE_FILE):
        try:
            with open(ACTIVE_FILE, encoding="utf-8") as f:
                ds = json.load(f)
            if os.path.exists(ds.get("db_path", "")):
                return ds
        except Exception:
            pass
    return DEMO


def set_active(ds: dict) -> dict:
    with open(ACTIVE_FILE, "w", encoding="utf-8") as f:
        json.dump(ds, f, indent=2)
    return ds


def use_demo() -> dict:
    return set_active(DEMO)


def register_upload(name: str) -> dict:
    """Paths for a newly uploaded dataset called `name`."""
    safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in name).strip("_")
    base = os.path.join(UPLOADS, safe)
    os.makedirs(base, exist_ok=True)
    return {
        "name": safe,
        "kind": "upload",
        "db_path": os.path.join(base, "data.duckdb"),
        "semantic_path": os.path.join(base, "semantic.yml"),
        "profile_path": os.path.join(base, "profile.json"),
    }


def list_uploads() -> list[str]:
    if not os.path.isdir(UPLOADS):
        return []
    return sorted(d for d in os.listdir(UPLOADS)
                  if os.path.exists(os.path.join(UPLOADS, d, "data.duckdb")))