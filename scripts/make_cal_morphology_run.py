#!/usr/bin/env python3
"""A morphology-run directory for the RL trainer from a chain flat scene on the `cal` plant.

    python3 scripts/make_cal_morphology_run.py --hand sv1_u0308_b050 --squeeze 10 \
        --out results/phase1/real_v1/20260916-sv1_u0308_b050_cal

The chain's flat scene (`assets/mjcf/experimental/20260904-chain_bothsets/<hand>_sq<mm>__flat.xml`)
carries the fitter's grasp as the `open_ik` keyframe (qpos = open pose, ctrl = grip). This
script applies the measured plant (kp, kv, forcerange, frictionloss, masses), the Coulomb-like
contact model (elliptic cone, impratio 10, pad mu), moves the palm plate to its built height,
freezes nothing (the design scenes have no morphology joints) and writes what
`rl_train_cube.py --morphology-run` reads: frozen_scene.xml, summary.json and a
best_rollout.npz whose reference rollout is the Phase-1 evaluator's own settle-lift-hold on
this scene. The grip is the fitter's anchor, not a CEM result.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
NAMES = ("a_thumb_yaw", "a_thumb_mcp", "a_thumb_pip", "a_index_yaw", "a_index_mcp", "a_index_pip",
         "a_middle_yaw", "a_middle_mcp", "a_middle_pip")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hand", required=True)
    ap.add_argument("--squeeze", type=float, default=10.0)
    ap.add_argument("--scenes", type=Path, default=ROOT / "assets/mjcf/experimental/20260904-chain_bothsets")
    ap.add_argument("--kp", type=float, default=0.5)
    ap.add_argument("--kv", type=float, default=0.02)
    ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--impratio", type=float, default=10.0)
    ap.add_argument("--plate-mm", type=float, default=25.0)
    ap.add_argument("--lift-ramp-steps", type=int, default=80)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    import mujoco
    from morphohand.optimization.phase1_common import Phase1EvalConfig, Phase1GraspEvaluator
    from real_v1_plan_angle_sweep import plate_variant
    from real_v1_turn_probe import contact_variant
    flat = a.scenes / f"{a.hand}_sq{a.squeeze:g}__flat.xml"
    if not flat.exists():
        flat = a.scenes / f"{a.hand}__flat.xml"
    if not flat.exists():
        raise SystemExit(f"no flat scene for {a.hand} at squeeze {a.squeeze}: run real_v1_chain_hands.py first")
    a.out.mkdir(parents=True, exist_ok=True)
    plant = a.out / f"plant_kp{a.kp:g}_kv{a.kv:g}.xml"
    subprocess.run([sys.executable, str(ROOT / "scripts/apply_measured_plant.py"), "--scene", str(flat),
                    "--out", str(plant), "--kp", str(a.kp), "--kv", str(a.kv), "--forcerange", "0.35",
                    "--frictionloss", "0.0035"], check=True, capture_output=True)
    scene = plate_variant(contact_variant(plant, "elliptic", a.impratio, a.mu), a.plate_mm)
    frozen = a.out / "frozen_scene.xml"
    frozen.write_text(scene.read_text())
    m = mujoco.MjModel.from_xml_path(str(frozen))
    k = next(i for i in range(m.nkey) if m.key(i).name == "open_ik")
    grip = np.array([float(m.key_ctrl[k][mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)]) for n in NAMES])
    cfg = Phase1EvalConfig(lift_delta_z=0.10, lift_ramp_steps=a.lift_ramp_steps)
    ev = Phase1GraspEvaluator(frozen, keyframe="open_ik", cfg=cfg, backend="mujoco")
    score, metrics = ev.evaluate(grip)
    r = ev.rollout(grip)
    z = np.asarray(r["cube_z"]) * 1e3
    c = np.asarray(r["contacts"])
    np.savez(a.out / "best_rollout.npz", best_finger_ctrl=grip,
             **{kk: np.asarray(r[kk]) for kk in ("qpos", "qvel", "cube_z", "contacts")})
    summary = {"scene_xml": str(flat.resolve()), "frozen_scene_xml": str(frozen.resolve()), "keyframe": "open_ik",
               "backend": "mujoco", "best_score": float(score), "best_finger_ctrl": grip.tolist(),
               "best_metrics": {kk: float(v) for kk, v in metrics.items()},
               "eval_config": {"settle_steps": cfg.settle_steps, "lift_steps": cfg.lift_steps,
                               "hold_steps": cfg.hold_steps, "lift_delta_z": 0.10, "lift_ramp_steps": a.lift_ramp_steps},
               "plant": {"kp": a.kp, "kv": a.kv, "forcerange": 0.35, "frictionloss": 0.0035, "cone": "elliptic",
                         "impratio": a.impratio, "mu": a.mu, "plate_mm": a.plate_mm},
               "hand": a.hand, "squeeze_mm": a.squeeze, "grip": "chain fitter anchor (open_ik keyframe ctrl), no CEM"}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"{a.hand} sq{a.squeeze:g} on kp {a.kp} mu {a.mu}: evaluator lift peak {metrics.get('cube_lift', 0) * 1e3:.1f} mm, "
          f"end z {z[-1]:.1f} mm, contacts end {c[-1]:.0f}, persistence {metrics.get('contact_persistence', 0):.2f}  -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
