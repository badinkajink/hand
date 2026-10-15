#!/usr/bin/env python3
"""Cost of the curved fingertips on the hand (docs/handoff/20261015-curved_tips.md, step 3), by the protocol of U1 on the
pad-usability page (docs/experiments/20261014-pad_usability/: scripts/pad_usability_timing.py and
scripts/pad_usability_gpu_cost.py), with the tip forced in every scene it builds. Pads are builder v2 (each pad's
K = E a_i / h_i at its local depth); the reference is the as-built block printed face up (tcd_rs3_re3_p0.2) with the same
builder.

  cpu      pad_usability_timing on the real_v1 hand at the servo plant's three-finger grip: the scale study's cells
           cylinder:16:0.625 and cylinder:25:0.125 (welded palm at the cell's best HOM grasp, 1 ms, elliptic cone,
           impratio 100, Newton at 1e-10 / 100 / 50), models 'pads' and 'pt' (the tip as one convex mesh, point
           contact): per state (open, grip, grip_ref) the step time, contacts and constraint rows, the timer split, the
           held check, and the replanning time of 64 plans x 0.5 s on 1 and 3 threads. Cell scenes are cached per tip
           under logs/20261015-curved_tips/scale/<tip>/; rows cost/cpu_<tip>.jsonl
  rlscene  the servo-plant RL run's frozen scene (results/phase1/real_v1/20261011-sv1_u0308_b050_servo_tip) with the
           tip's 1 mm pads (make_pad_morphology_run.py, builder v2, pads on the tool only): a run directory
           results/phase1/real_v1/20261015-sv1_u0308_b050_servo_tip_<tip>pads1 per tip
  export   (GPU env) pad_usability_gpu_cost export of that run: the RL env's scripted grasp and lift at zero residual
           action, 64 worlds, state after 60 policy steps, under logs/20261015-curved_tips/gpu/; rows cost/gpu_export.jsonl
  time     (GPU env) pad_usability_gpu_cost time of that state over --nworld worlds (MuJoCo-Warp, the trainer's
           solver options, buffers 1.5 times the state's needs); rows cost/gpu_timing.jsonl

    taskset -c 17-19 .venv/bin/python scripts/curved_tips_cost.py cpu --tips tcd_cap6_u0.1
    .venv/bin/python scripts/curved_tips_cost.py rlscene --tips tcd_cap6_u0.1
    MUJOCO_GL=egl WARP_CACHE_PATH=$(mktemp -d) uv run --extra rl --extra gpu python scripts/curved_tips_cost.py export --tips tcd_cap6_u0.1
    MUJOCO_GL=egl WARP_CACHE_PATH=$(mktemp -d) uv run --extra rl --extra gpu python scripts/curved_tips_cost.py time --tip tcd_cap6_u0.1 --nworld 2048
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import curved_tips_common as C  # noqa: E402

COST = C.D / "cost"
BOX = ROOT / "results/phase1/real_v1/20261011-sv1_u0308_b050_servo_tip"
CELLS = ["cylinder:16:0.625", "cylinder:25:0.125"]


def run_dir(tip: str) -> Path:
    return ROOT / f"results/phase1/real_v1/20261015-sv1_u0308_b050_servo_tip_{tip}pads1"


_RT = None


def force_tip(tip: str):
    """Every scene built from here on gets `tip` (reorient_backends.replace_tips builds tcd keys, builder v2). The
    wrapper always calls the original replace_tips: wrapping the previous tip's wrapper built the first tip again."""
    import reorient_backends as RB
    global _RT
    if _RT is None:
        _RT = RB.replace_tips

    def rt(root, tip_, model, MU, out_dir, **kw):
        return _RT(root, tip, model, MU, out_dir, **kw)
    RB.replace_tips = rt


def cmd_cpu(a):
    import hand_object_scale_sim as S
    import pad_usability_timing as PT
    for tip in a.tips:
        force_tip(tip)
        S.SCENES = C.LOGS / "scale" / tip
        out = COST / f"cpu_{tip}.jsonl"
        orig_append = PT.append

        def append(path, row, tip=tip):
            row.update(tip=tip, study="20261015-curved_tips", script_wrapper="scripts/curved_tips_cost.py")
            orig_append(path, row)
        PT.append = append
        argv = sys.argv
        sys.argv = ["pad_usability_timing.py", "--plant", C.SERVO, "--cells", *CELLS, "--models", *a.models,
                    "--threads", *map(str, a.threads), "--out", str(out)]
        try:
            PT.main()
        finally:
            sys.argv = argv
            PT.append = orig_append


def cmd_rlscene(a):
    import make_pad_morphology_run as MPR
    for tip in a.tips:
        out = run_dir(tip)
        if (out / "frozen_scene.xml").exists():
            print(f"{tip}: {out.relative_to(ROOT)} exists", flush=True)
            continue
        argv = sys.argv
        sys.argv = ["make_pad_morphology_run.py", "--src", str(BOX), "--out", str(out), "--tip", tip, "--model", "pads"]
        try:
            MPR.main()
        finally:
            sys.argv = argv
        print(f"{tip}: {out.relative_to(ROOT)}", flush=True)


def _gpu_module():
    import pad_usability_gpu_cost as PG
    PG.COST = COST
    PG.LOGS = C.LOGS / "gpu"
    PG.LOGS.mkdir(parents=True, exist_ok=True)
    return PG


def cmd_export(a):
    PG = _gpu_module()
    for tip in a.tips:
        PG.RUNS[tip] = str(run_dir(tip).relative_to(ROOT))
    ns = argparse.Namespace(variants=a.tips, envs=64, steps=60, force=False)
    return PG.cmd_export(ns)


def cmd_time(a):
    PG = _gpu_module()
    PG.RUNS[a.tip] = str(run_dir(a.tip).relative_to(ROOT))
    ns = argparse.Namespace(variant=a.tip, nworld=a.nworld, rep=a.rep, blocks=a.blocks, train_buffers=False)
    return PG.cmd_time(ns)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("cpu")
    p.add_argument("--tips", nargs="+", required=True)
    p.add_argument("--models", nargs="+", default=["pads", "pt"])
    p.add_argument("--threads", nargs="+", type=int, default=[1, 3])
    for name in ("rlscene", "export"):
        q = sub.add_parser(name)
        q.add_argument("--tips", nargs="+", required=True)
    t = sub.add_parser("time")
    t.add_argument("--tip", required=True)
    t.add_argument("--nworld", type=int, default=2048)
    t.add_argument("--rep", type=int, default=0)
    t.add_argument("--blocks", type=int, default=20)
    a = ap.parse_args()
    return {"cpu": cmd_cpu, "rlscene": cmd_rlscene, "export": cmd_export, "time": cmd_time}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
