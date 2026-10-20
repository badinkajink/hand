#!/usr/bin/env python3
"""First-bench package of the v2 finger (2026-10-20): scripts/sim2real_fingertip_bench.py's `export` and `abort` on the 9.75 mm
longer finger of scripts/v2_finger_scenes.py, from its pinned-iCEM rows (scripts/v2_finger_cells.py), so that the plan of
the 10-16 readiness study (the block on the CAD finger, 16 mm cell) has its counterpart on the finger being built.

The cells module, the kinematics and the scenes are redirected as scripts/v2_finger_cells.py does; the plan's source row comes
from docs/experiments/20261019-v2_finger/cells/icem.jsonl; plans, trajectories and verification land in
docs/experiments/20261019-v2_finger/bench/ named 20261020-s2r_<variant>_...; the plan's metadata names the v2 scene. The
readout trace in the verification file uses the readiness study's four-magnet block row (its sensing.json), as the 10-16 plan's
did; the 11/22 tip's own bound is in docs/experiments/20261019-v2_tip_fe_checks/sensing.json.

    SCALE_HAND=cad_v1 ~/.claude/bin/resguard.sh run --mem 3G --cpu 200 -- env NUMPY_MADVISE_HUGEPAGE=0 .venv/bin/python \\
        scripts/v2_finger_bench.py export --variant ell11_g30_v2a --cell cylinder:16:0.625 --tau-cap 0.15 --release-pip-open 35
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import v2_finger_cells as VC  # noqa: E402  (redirects the cells module, the kinematics and the scenes)
import sim2real_fingertip_bench as SB  # noqa: E402

SB.DATE_TAG = "20261020"
SB.OUT = ROOT / "docs/experiments/20261019-v2_finger/bench"
READY_SCENE = "assets/mjcf/experimental/20261016-sim2real_fingertip_readiness/"
V2_SCENE = "assets/mjcf/experimental/20261019-v2_finger/"


def pick_source(variant, cell, seed):
    """sim2real_fingertip_bench.pick_source on the v2 finger's pinned-iCEM rows."""
    best = None
    for line in open(VC.SC.OUT / "icem.jsonl"):
        r = json.loads(line)
        if r.get("status") != "ok" or r["variant"] != variant:
            continue
        if f"{r['shape']}:{r['d_mm']:g}:{r['t']:g}" != cell or (seed is not None and r["seed"] != seed):
            continue
        key = (r["held"] and r["fingers_loaded_end"] == 3, r["held_turn_deg"])
        if best is None or key > best[0]:
            best = (key, r)
    if best is None:
        raise SystemExit(f"no pinned-iCEM row for {variant} on {cell}")
    return best[1]


SB.pick_source = pick_source

if __name__ == "__main__":
    SB.main()
    for pj in sorted(SB.OUT.glob("20261020-s2r_*_plan.json")):          # the plan names the scene it was planned in
        d = json.loads(pj.read_text())
        sc = d.get("meta", {}).get("scene", "")
        if sc.startswith(READY_SCENE):
            d["meta"]["scene"] = V2_SCENE + sc[len(READY_SCENE):]
            d["meta"]["source"] = "scripts/v2_finger_bench.py export (scripts/sim2real_fingertip_bench.py on the v2 finger)"
            pj.write_text(json.dumps(d, indent=1))
