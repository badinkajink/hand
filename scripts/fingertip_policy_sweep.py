#!/usr/bin/env python3
"""Run the shipped inline policies on each fingertip SHAPE, zero-shot.

    MUJOCO_GL=egl uv run --extra rl --extra gpu python scripts/fingertip_policy_sweep.py

The mechanics probe (`probe_fingertip_mechanics.py`) ranks shapes at a scripted grip, and that
grip is light: the best reachable open-loop pose loads the pads at ~0.3-3 N, while the deployed
a10->b33 handoff runs at ~20 N. Contact patch effects are load-dependent, so a ranking taken at
3 N is a hypothesis about the ranking at 20 N, not a measurement of it. This runs the actual
policies on each shape at the actual operating load.

WHAT THIS IS AND IS NOT. It is zero-shot: a10 and b33 were trained on the shipped `cap_cross`
tip and are not retrained here. So a shape that scores badly may be a bad shape OR a shape the
policy would adapt to given training -- this measures the transfer, which is the cheap question,
and is the screen for which shapes deserve the expensive one. It is deliberately NOT the basis
for a promotion decision on its own, because per-design policy draws in this program carry an sd
of 0.3-0.5 and a zero-shot transfer number is not a trainability number (the b33 zero-shot probe
already failed as a predictor of trainable reorientability once, in the A-selector search).

Reuses the robustness sweep's batched continuous-handoff evaluator so the numbers land on the
same scale as `SIM2REAL_ROBUSTNESS.txt` and the baseline row is directly comparable.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from morphohand.studies import runlib
from morphohand.studies.runlib import ROOT
from morphohand.studies.scene_mutate import SHIPPED_TIP, TIP_SHAPES

sys.path.insert(0, str(ROOT / "scripts"))
from sim2real_robustness_sweep import rate_point, scene_dir  # noqa: E402

JSON_P = ROOT / "docs/experiments/FINGERTIP_POLICY.json"
TXT_P = ROOT / "docs/experiments/FINGERTIP_POLICY.txt"

# Radius/half-length variants of the shipped shape, to separate "the shape changed" from "the
# pad got bigger" -- a fatter tip reaches further and grips differently for reasons that have
# nothing to do with its cross-section.
SIZE_VARIANTS = ((0.004, 0.006), (0.006, 0.006), (0.008, 0.006), (0.005, 0.010))


def points():
    for shape in TIP_SHAPES:
        yield f"shape:{shape}", dict(shape=shape)
    for r, h in SIZE_VARIANTS:
        yield f"size:{SHIPPED_TIP}_r{r * 1000:.0f}h{h * 1000:.0f}", dict(shape=SHIPPED_TIP, r=r, h=h)


def main() -> None:
    store = runlib.RecordStore(JSON_P, key_field="key")
    report = runlib.TxtReport(
        TXT_P, f"# fingertip shape x shipped policy (zero-shot) {time.strftime('%Y-%m-%d %H:%M')}\n"
               f"# a10->b33, n=32/point, 300 steps, continuous handoff. "
               f"'{SHIPPED_TIP}' r5 h6 IS the shipped tip = the baseline row.\n")
    work = ROOT / "logs" / "_fingertip_tmp"
    work.mkdir(parents=True, exist_ok=True)

    for key, spec in points():
        if key in store:
            print(f"[skip] {key}")
            continue
        t0 = time.time()
        try:
            scene = scene_dir(f"tip_{key.replace(':', '_')}",
                              lambda s, sp=spec: s.set_tip_shape(**sp))
            m = rate_point(scene, work, {})
        except Exception as ex:
            m = {"error": f"{type(ex).__name__}: {str(ex)[:160]}"}
        rec = {"key": key, **spec, **m, "secs": round(time.time() - t0)}
        store.put(rec)
        report.line(f"{key:34} hold {str(rec.get('hold_rate')):>6} "
                    f"reorient {str(rec.get('reorient_rate')):>6} "
                    f"peak {str(rec.get('peak_cos_mean')):>6} "
                    f"cos|held {str(rec.get('cos_p50_held')):>7} "
                    f"minz {str(rec.get('minz_p50')):>7} {rec.get('error', '')}")
        print(f"[{key}] {rec}")
    runlib.Sentinel(ROOT / "logs/FINGERTIP_POLICY.DONE").write()
    print(f"[fingertip] COMPLETE -> {TXT_P}")


if __name__ == "__main__":
    main()
