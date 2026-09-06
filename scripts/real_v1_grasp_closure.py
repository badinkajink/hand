#!/usr/bin/env python3
"""Where the pads sit on the shaft, measured at closure, for every deployed plan.

A cylinder held by pads that all touch ABOVE its equator is held by friction alone: the
contact normals point down and inward, their vertical resultant is downward, and the shaft is
driven out of the pinch rather than into it. Below the equator the normals point up and inward
and the geometry carries the load. `probe_real_v1_carry.fit_real_v1_pose` picks the palm height
for tip clearance and never scores this, so a plan can pass its bench screen -- where the shaft
stands on a 100 mm post that supplies the missing upward reaction -- and drop the same shaft in
the chain, where nothing is underneath it.

Reported per hand, at the fitted grip with the palm stationary for 0.8 s:

    z_rel     each pad contact's height above the shaft's axis, in units of the shaft radius;
              +1 is the top of the cylinder, -1 the bottom, 0 the equator
    n_below   pads contacting at or below the equator (z_rel <= 0)
    span_deg  azimuthal span of the pad contacts about the shaft axis; 180 deg is an opposed
              pinch, a small span is a one-sided push
    Fz_pad    vertical component of the summed pad contact force, N; positive supports the tool

    uv run --extra rl --extra arm python scripts/real_v1_grasp_closure.py --out <json>
"""
from __future__ import annotations

import argparse, json, math, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import real_v1_chain_hands as ch  # noqa: E402

OBJ = ch.BASE["obj"]


def one(h: dict, squeeze_mm: float, elevation_deg: float = 0.0) -> dict | None:
    import mujoco, numpy as np
    import probe_real_v1_carry as pc
    import fit_real_v1_pose as fp
    from morphohand.tools.keyframe_ik import FINGERS
    from real_v1_grasp_refit import flat_scene
    scene = flat_scene(h)
    built = pc._grip_from_fit(scene, h["straddle"], 0.0, squeeze_mm / 1000.0, OBJ,
                              h["depth"], h["thumb_axial"], elevation_deg)
    if built is None:
        return None
    m, open_qpos, grip, depth_mm = built
    d = mujoco.MjData(m)
    d.qpos[:] = open_qpos
    d.ctrl[:] = grip
    mujoco.mj_forward(m, d)
    z0 = float(d.body(OBJ).xpos[2])
    for _ in range(400):
        mujoco.mj_step(m, d)

    centre, r_obj, _ = fp._object_geometry(m, d, OBJ)
    axis = np.asarray(d.body(OBJ).xmat).reshape(3, 3)[:, 2]     # the shaft's own long axis
    bid = m.body(OBJ).id
    tips = {m.body(f"{f}_tip").id: f for f in FINGERS}
    obj_c = np.asarray(d.body(OBJ).xpos, float)
    # A frame on the shaft's cross-section: `up` is world +z projected off the axis, `side`
    # completes it. Azimuth is measured from `up`, so 0 deg is the top of the cylinder.
    up = np.array([0.0, 0.0, 1.0]) - axis * float(axis[2])
    n_up = float(np.linalg.norm(up))
    up = up / n_up if n_up > 1e-9 else np.array([1.0, 0.0, 0.0])
    side = np.cross(axis, up)

    pads, F, nonpad = [], np.zeros(3), 0
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if bid not in (b1, b2):
            continue
        other = b2 if b1 == bid else b1
        if other not in tips:
            nonpad += 1
            continue
        f6 = np.zeros(6)
        mujoco.mj_contactForce(m, d, i, f6)
        # mj_contactForce is in the contact frame; frame[0:3] is the normal, pointing from
        # geom1 to geom2. Fold it so the reported force is the one acting ON the shaft.
        R = np.asarray(c.frame, float).reshape(3, 3)
        f_w = R.T @ f6[:3]
        if b1 == bid:
            f_w = -f_w
        F += f_w
        rvec = np.asarray(c.pos, float) - obj_c
        rvec = rvec - axis * float(rvec @ axis)                 # into the cross-section
        pads.append({"finger": tips[other],
                     "z_rel": round(float(rvec @ up) / float(r_obj), 3),
                     "az_deg": round(math.degrees(math.atan2(float(rvec @ side),
                                                             float(rvec @ up))), 1),
                     "F_N": round(float(np.linalg.norm(f6[:3])), 2)})
    az = sorted(p["az_deg"] for p in pads)
    if len(az) >= 2:
        gaps = [az[i + 1] - az[i] for i in range(len(az) - 1)] + [az[0] + 360 - az[-1]]
        span = round(360.0 - max(gaps), 1)
    else:
        span = 0.0
    return {"tag": h["tag"], "squeeze_mm": squeeze_mm, "elevation_deg": elevation_deg,
            "nonpad": nonpad, "depth_mm": round(depth_mm, 2),
            "straddle_mm": round(h["straddle"] * 1000, 1),
            "dz_mm": round((float(d.body(OBJ).xpos[2]) - z0) * 1000, 2),
            "n_pads": len(pads), "n_below": sum(1 for p in pads if p["z_rel"] <= 0.0),
            "z_rel_mean": round(sum(p["z_rel"] for p in pads) / len(pads), 3) if pads else None,
            "z_rel_min": round(min((p["z_rel"] for p in pads), default=float("nan")), 3),
            "span_deg": span, "Fz_pad": round(float(F[2]), 3),
            "F_pad_N": round(sum(p["F_N"] for p in pads), 2), "pads": pads}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sets", default="A")
    ap.add_argument("--squeeze-mm", default="2.0")
    ap.add_argument("--hands", default=None, help="comma list of plan tags to keep")
    ap.add_argument("--straddles", default=None, help="comma list of straddles in mm")
    ap.add_argument("--depths", default=None, help="comma list of grip depths in mm")
    ap.add_argument("--elevations", default="0",
                    help="comma list of pad-ring elevations in deg about the shaft. 0 aims the "
                         "pad centres at the equator; negative aims below it, which is where "
                         "the wedge holds, at the cost of the pad's 2.0 mm floor clearance.")
    args = ap.parse_args()

    H = ch.hands(args.sets)
    if args.hands:
        keep = set(args.hands.split(","))
        H = [h for h in H if h["tag"] in keep]
    H = ch.variants(H, args.straddles, args.depths)
    rows = []
    for h in H:
        for sq in (float(v) for v in args.squeeze_mm.split(",")):
          for el in (float(v) for v in args.elevations.split(",")):
            r = one(h, sq, el)
            if r is not None:
                rows.append(r)
                print(f"  {r['tag']:20} sq{sq:<4g} el{el:<+6g} nonpad {r['nonpad']} "
                      f"pads {r['n_pads']} below {r['n_below']} "
                      f"z_rel {r['z_rel_mean']:+.3f} span {r['span_deg']:6.1f} "
                      f"Fz {r['Fz_pad']:+7.3f} dz {r['dz_mm']:+6.2f}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=1))
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
