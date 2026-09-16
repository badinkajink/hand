#!/usr/bin/env python3
"""The bench maneuver on every deployed hand, per plant, against the hardware's own turn.

    python3 scripts/real_v1_bench_plants.py --out docs/experiments/20260916-turn_mechanism

Replays each hand's deployed plan (grip -> its _traj.csv -> hold, tool on the 100 mm post,
palm fixed: the maneuver the CB1 archive holds 251 runs of) on the shipped plant, the
corrected plant as calibrated (kv 0.6) and the corrected plant with the servo's speed
(kv 0.02), with the gate's spawn jitter, and puts the simulated turn beside the bench's
measured net turn on held trials (docs/experiments/20260902-cb1-log-archive, paper/figures/
ranking.json). The bench maneuver is the only one the hardware has run; a plant that does not
reproduce it per hand has no standing on the chain.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from real_v1_turn_mechanism import make_scene, replay  # noqa: E402

# D-number, plan file, bench hold rate and net turn on held trials (2026-09-02 archive).
HANDS = [
    ("D1", "sv1_w6689_b060", "20260902-residual-bench", "6/7", 51.9),
    ("D2", "sv1_w2360_b075", "20260902-residual-bench", "10/10", 48.2),
    ("D3", "sv1_u1364_b080", "20260902-residual-bench", "4/10", 42.1),
    ("D4", "g12_b095", "20260829-real_v1_deploy", "9/10", 52.8),
    ("D5", "sv1_u0060_b75", "20260830-real_v1-sobol128", "3/10", 69.6),
    ("D6", "sv1_u0308_b050", "20260829-real_v1_deploy", "2/10", 35.5),
    ("D7", "rv05_manual_b85", "20260902-residual-bench", "10/10", 33.3),
    ("D8", "sv1_w0099_b100", "20260829-real_v1_deploy", "4/10", 44.2),
]
PLANTS = {
    "shipped": dict(kp=30, forcerange=10, frictionloss=0, mass=False, kv=0.5),
    "corrected": dict(kp=0.5, forcerange=0.35, frictionloss=0.0035, mass=True, kv=0.6),
    "fast": dict(kp=0.5, forcerange=0.35, frictionloss=0.0035, mass=True, kv=0.02),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--jitter-xy", type=float, default=0.004)
    ap.add_argument("--jitter-yaw", type=float, default=0.09)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    scenes = a.out / "scenes"
    scenes.mkdir(exist_ok=True)
    rows = []
    for dn, tag, folder, hold, hw_deg in HANDS:
        plan_p = ROOT / "docs/experiments" / folder / "deploy" / f"{tag}_plan.json"
        plan = json.loads(plan_p.read_text())
        traj = plan_p.with_name(f"{tag}_traj.csv")
        base = Path(plan["meta"]["scene"])
        for pname, spec in PLANTS.items():
            scene = make_scene(base, scenes / f"{tag}__{pname}.xml", spec["kp"], spec["forcerange"],
                               spec["frictionloss"], mass=spec["mass"], kv=spec["kv"])
            turns, held, cos_end, f_end = [], 0, [], []
            for s in range(a.seeds):
                r = replay(scene, plan, traj if traj.exists() else None, post=True, gravity=True,
                           seed=1000 + s, jitter_xy=a.jitter_xy, jitter_yaw=a.jitter_yaw)
                turns.append(r["turn_deg"]); cos_end.append(r["cos_end"]); f_end.append(r["f_pad_end"])
                held += 0 if r["dropped"] else 1
            row = {"hand": dn, "tag": tag, "plant": pname, "hw_hold": hold, "hw_turn_deg": hw_deg,
                   "sim_held": held, "n": a.seeds,
                   "sim_turn_deg": round(float(np.mean(turns)), 1),
                   "sim_turn_sd": round(float(np.std(turns)), 1),
                   "sim_cos_end": round(float(np.mean(cos_end)), 3),
                   "sim_pad_N_end": round(float(np.mean(f_end)), 2)}
            rows.append(row)
            print(f"{dn} {tag:<16} {pname:<9} held {held}/{a.seeds}  turn {row['sim_turn_deg']:+6.1f} "
                  f"sd {row['sim_turn_sd']:4.1f}  cos {row['sim_cos_end']:+.3f}  pads {row['sim_pad_N_end']:.2f} N"
                  f"   | bench {hold} at {hw_deg} deg")
    (a.out / "bench_plants.json").write_text(json.dumps(rows, indent=1))
    # rank correlation of the simulated turn with the bench's turn, per plant
    from scipy.stats import spearmanr
    hw = [h[4] for h in HANDS]
    for pname in PLANTS:
        st = [r["sim_turn_deg"] for r in rows if r["plant"] == pname]
        rho = spearmanr(hw, st).correlation
        print(f"{pname:<9} spearman(sim turn, bench turn) = {rho:+.2f}")
    print(f"wrote {a.out / 'bench_plants.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
