"""Shared plumbing for the long-running study/pipeline scripts.

Extracted (CODEBASE_AUDIT.md step 1) from the four scripts that each re-implemented it:
compliance_robustness_sweep.py, reorient_variance_study.py, morph_pipeline_sweep.py,
compliance_dr_pipeline.py. Covers:

  - run/checkpoint lookup under results/rl (latest_run, final_ckpt, best_a_ckpt),
  - trainer-log parsing for object-height (iter_objheight),
  - per-item resumable JSON state (RecordStore), flat stage state (JsonState),
  - streamed human-readable reports (TxtReport) and DONE sentinels (Sentinel),
  - per-process Warp kernel caches (warp_cache_env) — a shared cache races and NaNs.

Studies stay unattended-robust: every put()/save() persists immediately, so a re-run
skips finished items.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import time
from pathlib import Path

# src/morphohand/studies/runlib.py -> repo root (the package is editable-installed from src/).
ROOT = Path(__file__).resolve().parents[3]
RESULTS_RL = ROOT / "results/rl"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- environments
def base_env(env: dict | None = None, *, mujoco_gl: str = "egl") -> dict:
    """Copy of `env` (default os.environ) with MUJOCO_GL defaulted for headless render."""
    e = dict(os.environ if env is None else env)
    e.setdefault("MUJOCO_GL", mujoco_gl)
    return e


def warp_cache_env(env: dict | None = None) -> dict:
    """Copy of `env` with a FRESH private WARP_CACHE_PATH for one Warp subprocess."""
    e = base_env(env)
    e["WARP_CACHE_PATH"] = tempfile.mkdtemp(prefix="warpcache_")
    return e


# --------------------------------------------------------------------------- run/ckpt lookup
def latest_run(glob: str, root: Path = RESULTS_RL) -> Path | None:
    """Most-recently-modified dir/file under `root` matching `glob` (None if no match)."""
    runs = sorted(root.glob(glob), key=lambda p: p.stat().st_mtime)
    return runs[-1] if runs else None


def final_ckpt(run: Path | None) -> Path | None:
    """Highest-iteration model_*.pt under `run`/tensorboard (None-tolerant)."""
    if run is None:
        return None
    cks = sorted((run / "tensorboard").glob("model_*.pt"),
                 key=lambda p: int(p.stem.split("_")[1]))
    return cks[-1] if cks else None


def iter_objheight(log_path: Path) -> dict[int, float]:
    """Parse a trainer log -> {iter: mean object_height}. For picking A's best lift ckpt."""
    if not log_path.exists():
        return {}
    text = log_path.read_text(errors="ignore")
    iters = [int(m.group(1)) for m in re.finditer(r"Learning iteration\s+(\d+)/", text)]
    chunks = re.split(r"Learning iteration\s+\d+/", text)
    oh: dict[int, float] = {}
    for chunk, it in zip(chunks[1:], iters):
        mm = re.search(r"lift_height/object_height:\s+([-\d.]+)", chunk)
        if mm:
            oh[it] = float(mm.group(1))
    return oh


def best_a_ckpt(run: Path, log_path: Path) -> tuple[Path | None, float | None]:
    """A's BEST-lifting saved checkpoint (robust to a late mid-training collapse — the final
    ckpt may be post-collapse). Returns (ckpt, best_objheight); falls back to the final ckpt
    (objheight None) when the log has no object-height data."""
    cks = {int(p.stem.split("_")[1]): p for p in (run / "tensorboard").glob("model_*.pt")}
    if not cks:
        return None, None
    oh = iter_objheight(log_path)
    if not oh:
        return final_ckpt(run), None
    best, best_oh = None, -1.0
    for n in sorted(cks):
        v = oh.get(n)
        if v is None:                       # ckpt iter has no logged oh -> nearest earlier iter
            earlier = [i for i in oh if i <= n]
            v = oh[max(earlier)] if earlier else -1.0
        if v > best_oh:
            best_oh, best = v, cks[n]
    return best, round(best_oh, 3)


# --------------------------------------------------------------------------- resumable state
class RecordStore:
    """Per-item resumable JSON: a list of dict records keyed by `key_field`, persisted on every
    put() so an interrupted study resumes exactly where it stopped (finished keys are skipped)."""

    def __init__(self, path: Path, key_field: str = "id"):
        self.path = path
        self.key_field = key_field
        self.records: dict[str, dict] = (
            {d[key_field]: d for d in json.loads(path.read_text())} if path.exists() else {}
        )

    def __contains__(self, key: str) -> bool:
        return key in self.records

    def __len__(self) -> int:
        return len(self.records)

    def get(self, key: str) -> dict | None:
        return self.records.get(key)

    def values(self) -> list[dict]:
        return list(self.records.values())

    def put(self, rec: dict) -> None:
        self.records[rec[self.key_field]] = rec
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(list(self.records.values()), indent=1))


class JsonState:
    """Flat resumable key->value state for staged pipelines; save() persists the whole dict."""

    def __init__(self, path: Path):
        self.path = path
        self.data: dict = json.loads(path.read_text()) if path.exists() else {}

    def get(self, key: str, default=None):
        return self.data.get(key, default)

    def update(self, **kv) -> None:
        self.data.update(kv)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=1))


class TxtReport:
    """Append-only human-readable report; `header` is written once when the file is created."""

    def __init__(self, path: Path, header: str):
        self.path = path
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(header)

    def line(self, s: str, echo: bool = True) -> None:
        if echo:
            print(s, flush=True)
        with self.path.open("a") as f:
            f.write(s + "\n")


class Sentinel:
    """A logs/*.DONE completion marker: clear() at study start, write() on completion."""

    def __init__(self, path: Path):
        self.path = path

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()

    def write(self, extra: str = "") -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.path.write_text(stamp + (f"  {extra}" if extra else "") + "\n")
