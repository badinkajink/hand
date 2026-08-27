"""Build hands with the hardware's TWO flexing links set directly, and gate them on geometry.

The hand has two flexion joints (MCP, PIP), so it has two link lengths. The scene draws the MCP
link as two capsules with no joint between them, which is why the co-design "length" parameter
and `Scene.set_proximal_length` both move only part of it: m05's MCP link is 75.8-80.9 mm and the
"25 mm" hand's is 65 mm. Nothing has ever been trained near 40 mm. This builds those hands.

Shortening a finger is never one edit. Three things move together, and skipping any one produces
a hand that reads as "cannot grasp" for a reason that has nothing to do with its links:

  1. The links themselves (`Scene.set_link_lengths`, capsules and kinematics together), including
     the yaw link — the segment between the yaw joint and the MCP joint, which the scenes draw as
     zero and the hardware has at ~32 mm.
  2. The palm is FITTED, not just dropped. Dropping it by the reach removed keeps the tips in
     range vertically and leaves the hand off-centre over the shaft, which cost the first pass a
     spurious 2-finger verdict on 40x30 (thumb contact persistence 0.00, CEM score 2.33 against
     m05's 8.52). Short fingers converge over a smaller footprint, so px/py/pz are solved for.
  3. Each fingertip is IK'd to the WORLD position it holds in the source hand's grasp keyframe —
     never the same joint angles (the standing retarget rule).

Then the gate, which is the point of the script. A shorter finger holds the tool CLOSER to the
mounting plane, and the tool has to stand up in that gap: a 100 mm shaft rotated to vertical needs
~50 mm of headroom above the grip. That constraint binds long before grip quality does, and it is
pure geometry — knowable before a single GPU-hour. It has two outs, and the script reports both:
mount the MCP axis below the plane, or shorten the shaft (`--tool-length`, and the report gives
the longest shaft each hand could ever stand up).

Config spec is PROXxDIST[+YAWLINK[z]], all mm: `40x30` is a bare 40/30 finger, `40x30+32` puts a
32 mm yaw link along the yaw axis (the serial roll-then-pitch build), `40x30+32z` hangs the MCP
joint 32 mm below the mounting plane instead.

Run:
  MUJOCO_GL=egl uv run --extra rl python scripts/build_link_length_variants.py \
      --config 40x30+32 --config 30x30+32 --config 25x25+32 --fit-palm
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from morphohand.studies.scene_mutate import Scene  # noqa: E402
from morphohand.tools.keyframe_ik import (  # noqa: E402
    FINGERS, TIPS, actuator_ctrl_from_qpos, ik_finger, inject_keyframe, tip_targets,
)

SRC_SCENE = PROJECT_ROOT / "results/phase1/landscape/m05_ik_cem/frozen_scene.xml"
OUT_ROOT = PROJECT_ROOT / "assets/mjcf/experimental/20260827-linklen"
PAD_REACH = 0.011          # tip pad front surface past the distal link (6 mm half + 5 mm radius)
TOOL_HALF = 0.050          # screwdriver stand-in is 100 mm long
MOUNTS = {f: f"{f}_mount" for f in FINGERS}


def link_lengths(model, finger: str) -> tuple[float, float]:
    """(MCP->PIP, PIP->pad surface) in metres, read off the compiled model."""
    prox = float(model.body(f"{finger}_len_frame").pos[0] + model.body(f"{finger}_pip_frame").pos[0])
    dist = float(model.body(TIPS[finger]).pos[0]) + PAD_REACH
    return prox, dist


def reach_shell(model, data, finger: str, samples: int = 24) -> tuple[float, float]:
    """Min/max mount->tip distance over the MCP x PIP flexion envelope, at the current yaw."""
    saved = data.qpos.copy()
    mount = data.body(MOUNTS[finger]).xpos.copy()
    jm, jp = model.joint(FINGERS[finger][1]), model.joint(FINGERS[finger][2])
    dists = []
    for a in np.linspace(*model.jnt_range[jm.id], samples):
        for b in np.linspace(*model.jnt_range[jp.id], samples):
            data.qpos[model.jnt_qposadr[jm.id]] = a
            data.qpos[model.jnt_qposadr[jp.id]] = b
            mujoco.mj_forward(model, data)
            dists.append(float(np.linalg.norm(data.body(TIPS[finger]).xpos - mount)))
    data.qpos[:] = saved
    mujoco.mj_forward(model, data)
    return min(dists), max(dists)


def vertical_tool_check(model, data, obj_adr: int, tool_half: float,
                        lift: float = 0.10) -> dict:
    """Stand the shaft vertical at the grip centroid and look for palm/link penetration.

    Done at POST-LIFT height. The scripted lift raises `palm_pz` by `lift_delta_z` and the object
    rides up with it, so the grip's offset below the palm is what the reorient actually happens
    in; testing at pickup height only rediscovers the table.

    Geometry first (headroom above the grip), then MuJoCo's own contact list, because the palm
    plate is not the only thing up there — the proximal links close over the top of the grip too.
    The headroom also sets a HARD CEILING on held-cos: a 100 mm shaft tilted t off vertical rises
    tool_half*cos(t) above the grip, so cos(t) can never exceed grip_depth / tool_half — and
    inverting that gives the longest shaft the hand could ever stand upright.
    """
    saved = data.qpos.copy()
    pz = model.jnt_qposadr[model.joint("palm_pz").id]
    data.qpos[pz] += lift
    mujoco.mj_forward(model, data)

    tips = np.array([data.body(TIPS[f]).xpos for f in FINGERS])
    grip = tips.mean(axis=0)
    palm_z = float(data.body("palm_pose").xpos[2])

    data.qpos[obj_adr:obj_adr + 3] = grip
    data.qpos[obj_adr + 3:obj_adr + 7] = [1, 0, 0, 0]      # cylinder axis -> world +z
    mujoco.mj_forward(model, data)

    obj_geoms = {model.geom(g).id for g in range(model.ngeom)
                 if model.body(model.geom_bodyid[g]).name == "screwdriver_medium"}
    hits = []
    for c in range(data.ncon):
        con = data.contact[c]
        pair = {con.geom1, con.geom2}
        if pair & obj_geoms and con.dist < 0:
            other = (pair - obj_geoms).pop() if len(pair - obj_geoms) == 1 else None
            if other is None:
                continue
            hits.append((model.body(model.geom_bodyid[other]).name, float(con.dist)))
    data.qpos[:] = saved
    mujoco.mj_forward(model, data)

    worst = {}
    for name, dist in hits:
        worst[name] = min(dist, worst.get(name, 0.0))
    depth = palm_z - grip[2]
    return {
        "grip_z_mm": grip[2] * 1000,
        "grip_depth_below_palm_mm": depth * 1000,
        "headroom_mm": (depth - tool_half) * 1000,
        "held_cos_ceiling": min(1.0, depth / tool_half),
        "max_tool_length_mm": 2 * depth * 1000,
        "penetrations": {k: v * 1000 for k, v in sorted(worst.items(), key=lambda kv: kv[1])},
    }


def _ik_cost(m, d, base_qpos, tips_tgt, palm_xyz) -> float:
    """Sum of squared fingertip IK residuals at a candidate palm (px, py, pz)."""
    d.qpos[:] = base_qpos
    for name, v in zip(("palm_px", "palm_py", "palm_pz"), palm_xyz):
        d.qpos[m.jnt_qposadr[m.joint(name).id]] = v
    mujoco.mj_forward(m, d)
    return float(sum(ik_finger(m, d, f, tips_tgt[f]) ** 2 for f in FINGERS))


def fit_palm(m, d, base_qpos, tips_tgt, x0) -> np.ndarray:
    """Solve the palm's position so all three tips can reach their targets.

    A short finger converges over a smaller footprint than m05's, so inheriting m05's palm x/y
    leaves the hand off-centre over the shaft and the far finger simply never arrives. That is
    the same failure mode as transferring a keyframe in joint space, one level up: the fingers
    are retargeted but the thing they hang off is not.
    """
    from scipy.optimize import minimize
    res = minimize(lambda v: _ik_cost(m, d, base_qpos, tips_tgt, v), np.asarray(x0),
                   method="Nelder-Mead",
                   options={"xatol": 1e-5, "fatol": 1e-12, "maxiter": 600})
    return res.x


def build(proximal: float, distal: float, tag: str, *, yaw_link: float, yaw_axis: str,
          keyframe: str, src: Path, taper: float, tool_length: float | None,
          do_fit: bool) -> dict:
    tips_tgt, palm_vals, obj_qpos = tip_targets(str(src), keyframe)

    smodel = mujoco.MjModel.from_xml_path(str(src))
    src_reach = {f: sum(link_lengths(smodel, f)) for f in FINGERS}
    # Only links AFTER the first flexing joint can carry the tip downward; a yaw link along the
    # yaw axis adds reach in the flexion plane but no vertical travel.
    flexing = proximal + distal
    new_reach = flexing + (yaw_link if yaw_axis == "z" else 0.0)
    drop = float(np.mean(list(src_reach.values()))) - new_reach

    out_dir = OUT_ROOT / tag
    out = out_dir / "scene.xml"
    out_dir.mkdir(parents=True, exist_ok=True)
    sc = Scene(src).set_link_lengths(proximal, distal, taper=taper, pad_reach=PAD_REACH,
                                    yaw_link=yaw_link, yaw_link_axis=yaw_axis)
    if tool_length:
        sc = sc.set_tool_length(tool_length)
    sc.write(out)

    m = mujoco.MjModel.from_xml_path(str(out))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key(keyframe).id)
    for j, v in palm_vals.items():
        d.qpos[m.jnt_qposadr[m.joint(j).id]] = v
    d.qpos[m.jnt_qposadr[m.joint("palm_pz").id]] = palm_vals["palm_pz"] - drop
    obj_adr = m.jnt_qposadr[m.joint(m.body("screwdriver_medium").jntadr[0]).id]
    d.qpos[obj_adr:obj_adr + 7] = obj_qpos
    mujoco.mj_forward(m, d)

    palm_xyz = [palm_vals["palm_px"], palm_vals["palm_py"], palm_vals["palm_pz"] - drop]
    if do_fit:
        palm_xyz = fit_palm(m, d, d.qpos.copy(), tips_tgt, palm_xyz)
        d.qpos[:] = d.qpos  # fit leaves the last IK pose; re-seat cleanly below
        mujoco.mj_resetDataKeyframe(m, d, m.key(keyframe).id)
        for j, v in palm_vals.items():
            d.qpos[m.jnt_qposadr[m.joint(j).id]] = v
        d.qpos[obj_adr:obj_adr + 7] = obj_qpos
    for name, v in zip(("palm_px", "palm_py", "palm_pz"), palm_xyz):
        d.qpos[m.jnt_qposadr[m.joint(name).id]] = v
    mujoco.mj_forward(m, d)

    rec = {"tag": tag, "proximal_mm": proximal * 1000, "distal_mm": distal * 1000,
           "yaw_link_mm": yaw_link * 1000, "yaw_link_axis": yaw_axis,
           "flexing_reach_mm": flexing * 1000, "reach_mm": new_reach * 1000,
           "tool_length_mm": (tool_length or 2 * TOOL_HALF) * 1000,
           "palm_fitted": bool(do_fit),
           "palm_xyz_mm": [float(v) * 1000 for v in palm_xyz],
           "palm_z_mm": float(d.body("palm_pose").xpos[2]) * 1000, "fingers": {}}

    for f in FINGERS:
        lo, hi = reach_shell(m, d, f)
        need = float(np.linalg.norm(tips_tgt[f] - d.body(MOUNTS[f]).xpos))
        rec["fingers"][f] = {"shell_min_mm": lo * 1000, "shell_max_mm": hi * 1000,
                            "target_dist_mm": need * 1000,
                            "in_shell": bool(lo - 1e-4 <= need <= hi + 1e-4)}

    for f in FINGERS:
        rec["fingers"][f]["ik_residual_mm"] = ik_finger(m, d, f, tips_tgt[f]) * 1000

    rec.update(vertical_tool_check(m, d, obj_adr, (tool_length or 2 * TOOL_HALF) / 2))
    rec["scene"] = str(out.relative_to(PROJECT_ROOT))

    inject_keyframe(out, "open_ik", " ".join(f"{v:.6g}" for v in d.qpos),
                    " ".join(f"{v:.6g}" for v in actuator_ctrl_from_qpos(m, d)))
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", default=[],
                    help="PROXxDIST[+YAWLINK[z]] in mm, repeatable. 40x30 = bare 40/30 finger; "
                         "40x30+32 = 32 mm yaw link along the yaw axis; 40x30+32z = MCP joint "
                         "hung 32 mm below the mounting plane instead.")
    ap.add_argument("--tool-length", type=float, default=None,
                    help="Shorten the screwdriver stand-in to this many mm (default: leave at 100).")
    ap.add_argument("--fit-palm", action="store_true",
                    help="Solve palm px/py/pz for the tips instead of only dropping it in z.")
    ap.add_argument("--src", type=Path, default=SRC_SCENE)
    ap.add_argument("--keyframe", default="open_ik")
    ap.add_argument("--taper", type=float, default=0.4)
    ap.add_argument("--json", type=Path, default=PROJECT_ROOT / "docs/experiments/LINK_LENGTH_GATE.json")
    args = ap.parse_args()

    smodel = mujoco.MjModel.from_xml_path(str(args.src))
    print(f"[src] {args.src.relative_to(PROJECT_ROOT)} @ {args.keyframe}")
    for f in FINGERS:
        p, dd = link_lengths(smodel, f)
        print(f"    {f:7s} proximal {p*1000:6.1f}  distal {dd*1000:5.1f}  reach {(p+dd)*1000:6.1f} mm")

    recs = []
    for spec in args.config:
        body, _, yl = spec.partition("+")
        axis = "z" if yl.endswith("z") else "x"
        p_mm, _, d_mm = body.partition("x")
        tag = spec.replace("+", "_yl")
        if args.tool_length:
            tag += f"_t{args.tool_length:.0f}"
        rec = build(float(p_mm) / 1000, float(d_mm) / 1000, tag,
                    yaw_link=float(yl.rstrip("z") or 0) / 1000, yaw_axis=axis,
                    keyframe=args.keyframe, src=args.src, taper=args.taper,
                    tool_length=(args.tool_length / 1000 if args.tool_length else None),
                    do_fit=args.fit_palm)
        recs.append(rec)
        print(f"\n=== {spec}  proximal {rec['proximal_mm']:.0f} distal {rec['distal_mm']:.0f}"
              f"  yaw link {rec['yaw_link_mm']:.0f} along {rec['yaw_link_axis']}"
              f"  flexing reach {rec['flexing_reach_mm']:.0f} mm"
              f"  shaft {rec['tool_length_mm']:.0f} mm ===")
        print(f"    palm {'FITTED' if rec['palm_fitted'] else 'dropped'} to "
              f"({rec['palm_xyz_mm'][0]:+.1f}, {rec['palm_xyz_mm'][1]:+.1f}, "
              f"{rec['palm_xyz_mm'][2]:+.1f}) mm -> palm z {rec['palm_z_mm']:.1f} mm")
        for f, fr in rec["fingers"].items():
            print(f"    {f:7s} shell [{fr['shell_min_mm']:5.1f},{fr['shell_max_mm']:6.1f}] "
                  f"target {fr['target_dist_mm']:5.1f} {'OK ' if fr['in_shell'] else 'OUT'}"
                  f"   IK residual {fr['ik_residual_mm']:6.2f} mm")
        print(f"    grip sits {rec['grip_depth_below_palm_mm']:.1f} mm below the palm; "
              f"headroom {rec['headroom_mm']:+.1f} mm; held-cos CEILING "
              f"{rec['held_cos_ceiling']:.2f}; longest shaft it could stand up "
              f"{rec['max_tool_length_mm']:.0f} mm")
        if rec["penetrations"]:
            worst = ", ".join(f"{k} {v:.1f} mm" for k, v in list(rec["penetrations"].items())[:4])
            print(f"    VERTICAL SHAFT PENETRATES: {worst}")
        else:
            print("    vertical shaft is clear of the hand")

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(recs, indent=2))
    print(f"\n[write] {args.json.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
