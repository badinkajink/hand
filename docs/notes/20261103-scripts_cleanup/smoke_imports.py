#!/usr/bin/env python3
"""Import every script that has a main guard, each in its own process, and record which imports fail (2026-11-03).

    python3 smoke_imports.py --root REPO --python PY --out rows.jsonl [--names a,b,...] [--layout moved|flat]

`moved` puts scripts/ on sys.path and imports _paths (the folders of 2026-11-03); `flat` puts scripts/ alone on
sys.path (the layout before). One row per module, fsynced as it is written: name, ok, seconds, last line of stderr.
Modules without `if __name__ == "__main__"` run their work at import and are left out.
"""
import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--python", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--names", default="")
    ap.add_argument("--layout", choices=("moved", "flat"), default="moved")
    ap.add_argument("--timeout", type=float, default=90)
    a = ap.parse_args()
    root = Path(a.root).resolve()
    scripts = root / "scripts"
    files = sorted(p for p in scripts.rglob("*.py") if "archive" not in p.relative_to(scripts).parts
                   and "__pycache__" not in p.parts and p.parent.joinpath("__init__.py").exists() is False
                   and p.name != "_paths.py")
    guarded = {p.stem: p for p in files if re.search(r"^if __name__ == ['\"]__main__['\"]", p.read_text(errors="replace"),
                                                     re.M)}
    names = [n for n in a.names.split(",") if n] or sorted(guarded)
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["name"] for l in open(a.out) if l.strip()}
    boot = "import sys; sys.path.insert(0, 'scripts'); import _paths; " if a.layout == "moved" else \
        "import sys; sys.path.insert(0, 'scripts'); "
    env = dict(os.environ, MUJOCO_GL="egl", PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(root / "src"))
    with open(a.out, "a") as fh:
        for n in names:
            if n in done:
                continue
            t0 = time.time()
            try:
                r = subprocess.run([a.python, "-c", boot + f"import importlib; importlib.import_module({n!r})"],
                                   cwd=root, env=env, capture_output=True, text=True, timeout=a.timeout)
                ok, err = r.returncode == 0, (r.stderr.strip().splitlines() or [""])[-1][:300]
            except subprocess.TimeoutExpired:
                ok, err = False, f"timeout {a.timeout:.0f} s"
            row = dict(name=n, ok=ok, s=round(time.time() - t0, 1), err=err,
                       file=str(guarded[n].relative_to(root)) if n in guarded else "")
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            os.fsync(fh.fileno())


if __name__ == "__main__":
    main()
