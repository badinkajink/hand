#!/usr/bin/env python3
"""D1-D8 plan replays on the printable 85A fingertips (docs/handoff/20261016-sim2real_fingertip_readiness.md, step 5).

The trajectory protocol of scripts/curved_tips_traj.py (CT): each deployed hand D1-D8 replays its bench plan open loop on
the servo plant (kp0.72_kv0_fr0.18_fl0_dp0.0144), the 24.5 g screwdriver placed by reorient_backends.jitter(seed), friction
1.0, so every arm receives the same servo commands from the same start; every 20 ms the row records the tool's position and
orientation and each finger's summed normal force and friction-force vector on the tool. Per variant
(scripts/sim2real_fingertip_scenes.py) the tip is the variant's shape with pads at its modulus E:

  plan    MuJoCo, builder-v2 pads, 1 ms, impratio 100, Newton 100 iterations (the planner's and evaluation's numerics)
  rl      the same pads at the RL trainer's numerics: 2 ms, impratio 10, 10 iterations
  drake   Drake 1.57 hydroelastic, the tip's compliant convex at E, relaxation 0.01 s, SAP, 1 ms

Pad scenes are cached per variant (logs/20261016-sim2real_fingertip_readiness/replay_scenes/<variant>/), since the pads'
stiffness is baked into each scene. Rows: docs/experiments/20261016-sim2real_fingertip_readiness/replays/traces.jsonl,
one fsynced row per (variant, arm, hand, seed), resumable.

    PY=logs/20261001-hom_contact/venv/bin/python
    ~/.claude/bin/resguard.sh run --mem 3G --cpu 200 -- $PY scripts/sim2real_fingertip_replays.py --variants block4_g30 \\
        --arms plan rl --hands D1 --seeds 0
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import curved_tips_traj as CT  # noqa: E402
import pad_usability_backends as PUB  # noqa: E402
import reorient_backends as RB  # noqa: E402
import tip_codesign_rig as R  # noqa: E402

D = ROOT / "docs/experiments/20261016-sim2real_fingertip_readiness"
OUT = D / "replays" / "traces.jsonl"
ASSETS = ROOT / "assets/mjcf/experimental/20261016-sim2real_fingertip_readiness"
LOGS = ROOT / "logs/20261016-sim2real_fingertip_readiness"
SERVO = "kp0.72_kv0_fr0.18_fl0_dp0.0144"
SCRIPT = "scripts/sim2real_fingertip_replays.py"
ARMS = {k: CT.ARMS[k] for k in ("plan", "rl", "drake")}


def done(path):
    have = set()
    if path.exists():
        for line in open(path):
            try:
                r = json.loads(line)
                if r.get("status") in ("complete", "ejected"):
                    have.add((r["variant"], r["arm"], r["hand"], int(r["seed"])))
            except Exception:
                pass
    return have


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--arms", nargs="+", default=["plan", "rl"], choices=list(ARMS))
    ap.add_argument("--hands", nargs="+", default=list(RB.HANDS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    have = done(a.out)
    for variant in a.variants:
        spec = json.loads((ASSETS / variant / "README.json").read_text())
        key, E = spec["key"], float(spec["pad_E_Pa"])
        R.install(argparse.Namespace(E=E, pf=None, tr=RB.PAD_TR, tag=None, h="local", K="each"))
        PUB.SCN = LOGS / "replay_scenes" / variant
        for hand in a.hands:
            plan, traj = RB.load_plan(hand)
            for arm in a.arms:
                sp = ARMS[arm]
                todo = [s for s in a.seeds if (variant, arm, hand, s) not in have]
                if not todo:
                    continue
                gc.collect()
                db = None
                try:
                    if sp["sim"] == "mujoco":
                        scene, meta = PUB.build(hand, key, "pads", SERVO, sp["numerics"], sp["ir"], sp["mu"])
                    else:
                        scene, meta = PUB.drake_scene(hand, key, SERVO, sp["mu"])
                        RB.DT = 0.001
                        db = CT.Bench(scene, meta, E=E)
                except Exception as e:
                    for s in todo:
                        PUB.append_row(a.out, {"variant": variant, "arm": arm, "tip": key, "hand": hand, "seed": s,
                                               "status": "failed", "error": repr(e), "traceback": traceback.format_exc()[-1500:],
                                               "script": SCRIPT, "when": time.strftime("%Y-%m-%d %H:%M")})
                    print(f"{variant} {arm} {hand}: scene failed: {e!r}", flush=True)
                    continue
                for s in todo:
                    t0 = time.time()
                    row = {"variant": variant, "arm": arm, **sp, "tip": key, "E_Pa": E, "hand": hand, "seed": s,
                           "plant": SERVO, "jitter": RB.jitter(s), "scene": str(Path(scene).relative_to(ROOT))}
                    try:
                        res, ext = (CT.run_mujoco(scene, meta, plan, traj, s) if db is None else
                                    CT.run_drake(db, meta, plan, traj, s))
                        row.update(res)
                        row["trace"] = ext
                        row.update(n_pads=meta.get("n_pads"), pad_builder=meta.get("pad_builder"))
                    except Exception as e:
                        row.update(status="failed", error=repr(e), traceback=traceback.format_exc()[-1500:])
                    row.update(script=SCRIPT, git_rev=RB.git_rev(), wall_s=round(time.time() - t0, 2),
                               when=time.strftime("%Y-%m-%d %H:%M"))
                    PUB.append_row(a.out, row)
                    print(f"{variant:13s} {arm:5s} {hand} s{s}: {row.get('status')} turn {row.get('turn_end_deg', float('nan')):+.1f} "
                          f"held {row.get('held_end')} {row['wall_s']:.1f} s", flush=True)
                del db


if __name__ == "__main__":
    main()
