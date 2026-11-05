#!/usr/bin/env python3
"""Simulation cost of the fingertip-shape rerun's tips (docs/handoff/20261103-fingertip_shape_rerun.md, ledger step 4), by
the protocol of U1 on the pad-usability page (docs/experiments/20261014-pad_usability/) as scripts/curved_tips_cost.py ran
it on 10-15, with the pads at 2.5 MPa (builder v2, relaxation 0.04 s):

  cpu      scripts/pad_usability_timing.py on the v4b hand (SCALE_HAND=cad_v2, HOS_PLANT=servo) at the three-finger grip
           of the cells cylinder:16:0.625 and cylinder:25:0.125: welded palm at the grasp with the best median held
           HOM turn of the v4b tip at friction 0.7 (screen rows), friction 1, 1 ms, elliptic cone, impratio 100, Newton
           at 1e-10 / 100 / 50; models 'pads' and 'pt'; step time, contacts, constraint rows and the replanning time of
           64 plans x 0.5 s on 1 and 3 threads -> cost/cpu_<tip>.jsonl (cell scenes shared with the screen's friction-1.0
           cache, logs/<run>/scale/<tip>_mu1/)
  rlscene  the servo-plant RL run's frozen scene (results/phase1/real_v1/20261011-sv1_u0308_b050_servo_tip) with the tip's
           1 mm pads at 2.5 MPa (make_pad_morphology_run.py, pads on the tool only) -> logs/<run>/rl/<tip>/
  export   (GPU env) pad_usability_gpu_cost export of that run: the RL env's scripted grasp and lift at zero residual
           action, 64 worlds, state after 60 policy steps -> logs/<run>/gpu/, rows cost/gpu_export.jsonl
  time     (GPU env) pad_usability_gpu_cost time of that state over --nworld worlds (MuJoCo-Warp, the trainer's solver
           options, buffers 1.5 times the state's needs) -> cost/gpu_timing.jsonl

    SCALE_HAND=cad_v2 HOS_PLANT=servo taskset -c 17-19 .venv/bin/python scripts/fingertip_shape_rerun_cost.py cpu --tips <tip>
    .venv/bin/python scripts/fingertip_shape_rerun_cost.py rlscene --tips <tip>
    MUJOCO_GL=egl WARP_CACHE_PATH=$(mktemp -d) uv run --extra rl --extra gpu python scripts/fingertip_shape_rerun_cost.py \\
        export --tips <tip>
    MUJOCO_GL=egl WARP_CACHE_PATH=$(mktemp -d) uv run --extra rl --extra gpu python scripts/fingertip_shape_rerun_cost.py \\
        time --tip <tip> --nworld 2048
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import fingertip_shape_rerun_common as C  # noqa: E402

COST = C.D / "cost"
BOX = ROOT / "results/phase1/real_v1/20261011-sv1_u0308_b050_servo_tip"
CELLS = ["cylinder:16:0.625", "cylinder:25:0.125"]
SCRIPT = "scripts/fingertip_shape_rerun_cost.py"


def run_dir(tip: str) -> Path:
    return C.LOGS / "rl" / tip


def ref_rows() -> Path:
    return C.D / "screen/scale" / f"{C.REF}_mu0.7_{C.SERVO}/turn.jsonl"


def cmd_cpu(a):
    import fingertip_shape_rerun_rig as RR
    _, S = RR.scale_env()
    import pad_usability_timing as PT
    for tip in a.tips:
        S.TIP = tip
        S.MU = 1.0
        S.SCENES = C.LOGS / "scale" / f"{tip}_mu1"
        out = COST / f"cpu_{tip}.jsonl"
        orig_append = PT.append

        def append(path, row, tip=tip):
            row.update(tip=tip, E_Pa=C.E_PA, pad_tr_s=C.TR_S, study=f"{C.DATE}-fingertip_shape_rerun",
                       script_wrapper=SCRIPT)
            orig_append(path, row)
        PT.append = append
        argv = sys.argv
        sys.argv = ["pad_usability_timing.py", "--plant", C.SERVO, "--cells", *CELLS, "--models", *a.models,
                    "--threads", *map(str, a.threads), "--hom-rows", str(ref_rows()), "--out", str(out)]
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
        sys.argv = ["make_pad_morphology_run.py", "--src", str(BOX), "--out", str(out), "--tip", tip, "--model", "pads",
                    "--E", f"{C.E_PA:g}"]
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
