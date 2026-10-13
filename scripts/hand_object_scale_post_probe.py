#!/usr/bin/env python3
"""Traces of the HOM turn on chosen cells of the hand-object scale study, with and without the support post: per
10 ms tick the turn angle, the post's normal force on the object, the object's rise from its post-grip height and each
finger's normal force, at the grasp the landscape picks (largest median held turn in the given rows).

  .venv/bin/python scripts/hand_object_scale_post_probe.py --plant kp0.72_kv0_fr0.18_fl0_dp0.0144 \\
      --cells cylinder:16:0.625 cylinder:25:0.5 --seeds 0
Writes docs/experiments/20261013-scale_turn_controllers/post_traces.json (one entry per cell, protocol and seed).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import hand_object_scale_sim as S  # noqa: E402
import hom_turn3 as H  # noqa: E402
import hand_object_scale_mpc as M  # noqa: E402

OUT = M.OUT / "post_traces.json"


def best_cand(rows: Path, key) -> int:
    by = defaultdict(list)
    for line in rows.read_text().splitlines():
        r = json.loads(line)
        if r.get("status") == "ok" and r.get("model", "pads") == "pads" and (r["tag"], r["shape"], r["d_mm"]) == key:
            by[r["cand"]].append(r["held_turn_deg"])
    return max(sorted(by), key=lambda c: float(np.median(by[c])))


def main() -> int:
    import mujoco
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plant", default=None)
    ap.add_argument("--cells", nargs="+", required=True, help="shape:d_mm:t")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--rows", default=str(ROOT / "docs/experiments/20261011-bench_plan/servo_plant/turn.jsonl"),
                    help="turn rows (with the post) that pick each cell's grasp")
    a = ap.parse_args()
    if a.plant:
        S.PLANT = a.plant
    kp = S.RB.plant_args(S.PLANT)["kp"]
    out = json.loads(OUT.read_text()) if OUT.exists() else []
    for spec in a.cells:
        shape, dd, t = spec.split(":")
        dd, t = float(dd), float(t)
        cell, _, _ = next(c for c in S.cell_list("diag") if c[1] == shape and c[2] == dd and c[0].get("t") == t)
        ci = best_cand(Path(a.rows), (cell["tag"], shape, dd))
        plant, meta, geo, sc, gg = M.setup(cell, shape, dd, ci)
        mirror = S.Mirror(Path(sc["pt"][0]))
        S.set_palm(mirror.m, gg["palm"])
        post = plant.m.geom("post").id
        f6 = np.zeros(6)
        for post_off in (0.0, 1.0):
            for seed in a.seeds:
                log = []

                def cb(pl, tt, info):
                    m, d = pl.m, pl.d
                    fp = 0.0
                    for i in range(d.ncon):
                        g1, g2 = int(d.contact[i].geom[0]), int(d.contact[i].geom[1])
                        if (g1 == post and g2 in pl.tool_geoms) or (g2 == post and g1 in pl.tool_geoms):
                            mujoco.mj_contactForce(m, d, i, f6)
                            fp += float(f6[0])
                    s = pl.state()
                    log.append({"t": round(tt, 3), "phase": info["phase"], "th": round(info.get("th_deg", 0.0), 2),
                                "post_N": round(fp, 3), "z": float(s["p"][2]),
                                "F": [round(s["F"][f], 3) for f in S.FINGERS]})

                prm = dict(H.DEFAULTS, kp_model=kp, post_off=post_off)
                res = S.turn_rollout(plant, mirror, meta, seed, prm, frame_cb=cb)
                res.pop("trace", None)
                z0 = next(x["z"] for x in log if x["phase"] != "grip")
                for x in log:
                    x["rise_mm"] = round(1e3 * (x.pop("z") - z0), 3)
                out = [e for e in out if not (e["plant"] == S.PLANT and e["shape"] == shape and e["d_mm"] == dd
                                              and e["t"] == t and e["post_off"] == post_off and e["seed"] == seed)]
                out.append({"plant": S.PLANT, "tag": cell["tag"], "shape": shape, "d_mm": dd, "t": t, **geo,
                            "seed": seed, "post_off": post_off, "result": res, "trace": log})
                touch = [x for x in log if x["phase"] != "grip" and x["post_N"] > 0.01]
                print(f"{spec} c{ci} s{seed} post_off {post_off:g}: held {res['held']} turn {res['turn_end_deg']} "
                      f"frozen {res['frozen_deg']}; post touches in {len(touch)} ticks, max "
                      f"{max((x['post_N'] for x in touch), default=0):.2f} N", flush=True)
    OUT.write_text(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
