#!/usr/bin/env python3
"""Zero-shot transfer of the D6 servo-plant policies to the printable 85A fingertips (docs/handoff/
20261016-sim2real_fingertip_readiness.md, step 7).

The final policies of 2026-10-11 (results/rl/20261011-d6_servo_tpu27{mesh,pads1}_40M_s{0,1}, model_812.pt) trained on hand
D6 of real_v1 (results/phase1/real_v1/20261011-sv1_u0308_b050_servo_tip*): real_v1's capsule fingers on the servo plant,
the 24.5 g screwdriver, the TPU block tpu2.7 (17 x 14.8 x 22 mm, 2.7 mm fillets) as point contact or as builder-v1 pads at
10 MPa. Each finalist scene here is that run's frozen scene with only the fingertip changed: the variant's tip shape and its
1 mm pads (builder v2, the variant's modulus; scripts/tip_codesign_rig.py rlscene on the box-tip run). The policies'
observation and action schema, timing and the tool's contact class come from each run's own config (rl_contact_eval's
Evaluator), so only the fingertip differs from training.

  scenes   results/phase1/real_v1/20261016-sv1_u0308_b050_servo_tip_s2r_<variant>/ per variant
  eval     rl_contact_eval's transfer evaluation (64 deterministic rollouts of 5 s per policy and scene, the scripted grasp
           and lift, residual from step 58) on each scene, with the trained scene as the reference; rows in
           docs/experiments/20261016-sim2real_fingertip_readiness/policies/transfer.jsonl

    uv run --extra rl python scripts/sim2real_fingertip_policies.py scenes --variants block4_g30 cap_g30
    WARP_CACHE_PATH=$(mktemp -d) uv run --extra rl --extra gpu python scripts/sim2real_fingertip_policies.py eval \\
        --variants block4_g30 cap_g30
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

D = ROOT / "docs/experiments/20261016-sim2real_fingertip_readiness"
ASSETS = ROOT / "assets/mjcf/experimental/20261016-sim2real_fingertip_readiness"
MORPH = ROOT / "results/phase1/real_v1"
SRC = MORPH / "20261011-sv1_u0308_b050_servo_tip"                      # the box-tip run of the servo plant
POLICIES = ["20261011-d6_servo_tpu27mesh_40M_s0", "20261011-d6_servo_tpu27mesh_40M_s1",
            "20261011-d6_servo_tpu27pads1_40M_s0", "20261011-d6_servo_tpu27pads1_40M_s1"]


def run_dir(variant):
    return MORPH / f"20261016-sv1_u0308_b050_servo_tip_s2r_{variant}"


def cmd_scenes(a):
    import reorient_backends as RB
    import tip_codesign_rig as R
    for variant in a.variants:
        spec = json.loads((ASSETS / variant / "README.json").read_text())
        out = run_dir(variant)
        if (out / "frozen_scene.xml").exists() and not a.force:
            print(f"{out.relative_to(ROOT)}: exists", flush=True)
            continue
        ns = argparse.Namespace(E=float(spec["pad_E_Pa"]), tr=RB.PAD_TR, h="local", K="each", tip=spec["key"],
                                src=SRC, out_dir=out)
        R.cmd_rlscene(ns)
        meta = dict(variant=variant, tip=spec["key"], pad_E_Pa=spec["pad_E_Pa"], pad_builder="v2", src=str(SRC.relative_to(ROOT)),
                    note="the 2026-10-11 servo-plant D6 scene with only the fingertip changed",
                    script="scripts/sim2real_fingertip_policies.py scenes")
        (out / "s2r_variant.json").write_text(json.dumps(meta, indent=1))
        print(f"{out.relative_to(ROOT)}: {spec['key']} at {spec['pad_E_Pa'] / 1e6:.2f} MPa", flush=True)


def cmd_eval(a):
    import rl_contact_eval as E
    E.use_study("20261011")
    out = D / "policies"
    out.mkdir(parents=True, exist_ok=True)
    E.OUT8 = out                                                         # rows here, not in the 2026-10-11 folder
    scenes = {"tpu27pads1": E.ARMS["tpu27pads1"], "tpu27mesh": E.ARMS["tpu27mesh"]} if a.reference else {}
    for v in a.variants:
        scenes[f"s2r_{v}"] = run_dir(v)
    E.ARMS.update(scenes)
    for sc in scenes:
        ns = argparse.Namespace(scene=sc, n=a.n, steps=250, all=False, tags=a.policies, arms=None, seeds=None)
        E.cmd_transfer(ns)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scenes")
    s.add_argument("--variants", nargs="+", required=True)
    s.add_argument("--force", action="store_true")
    e = sub.add_parser("eval")
    e.add_argument("--variants", nargs="+", required=True)
    e.add_argument("--policies", nargs="+", default=POLICIES)
    e.add_argument("--n", type=int, default=64)
    e.add_argument("--reference", action="store_true", help="also the trained scenes (tpu27pads1, tpu27mesh)")
    a = ap.parse_args()
    {"scenes": cmd_scenes, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
