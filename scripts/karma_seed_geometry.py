"""Where does KaRMA's chosen pinch actually touch a real_v1 finger?

KaRMA models every finger link as a capsule of one radius and lets contact land anywhere
on it -- the README says so ("contact can land anywhere on a finger, and we do not attempt
to emulate the real shape of the finger pads"). real_v1's fingers are not uniform
cylinders: they are 10.55 mm capsules with a pad sphere at the tip, and the pad is the
only surface with the high-friction contact parameters. So a pinch the metric calls closed
can be a pinch this hand cannot make.

For each completed run this rebuilds the hand in MuJoCo at KaRMA's own seed joint angles
with a sphere of KaRMA's own scaled radius at KaRMA's own seed centre, and measures, per
finger, the surface gap to the nearest point of every finger geom and to the pad sphere
specifically. Two numbers come out per finger:

  gap_nearest_mm   how far the object is from the finger at all. Positive means KaRMA's
                   contact does not close on the hardware geometry, and it should come out
                   near the 0.82 mm difference between KaRMA's scaled capsule radius and
                   the real 10.55 mm link.
  gap_pad_mm       how far the object is from the PAD. Large while gap_nearest is small
                   means the metric pinched with the shaft of a link, not the fingertip.

    python scripts/karma_seed_geometry.py --table .../karma_table.json --work <workdir> \\
        --out .../seed_geometry.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
FINGERS = ("thumb", "index", "middle")


def _seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / max(float(np.dot(ab, ab)), 1e-12), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def measure(mounts_mm: list[float], q: list[float], centre: list[float],
            sphere_r: float) -> dict:
    import mujoco
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from karma_replay_mujoco import hand_at_mounts, build_scene, KARMA_Q_ORDER

    xml = build_scene(hand_at_mounts(mounts_mm), sphere_r, np.array(centre), gravity=False)
    m = mujoco.MjModel.from_xml_string(xml, {})
    d = mujoco.MjData(m)
    for k, j in enumerate(KARMA_Q_ORDER):
        d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]] = q[k]
    mujoco.mj_forward(m, d)
    sp = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "karma_sphere")]

    out: dict = {}
    for g in range(m.ngeom):
        bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g])
        if not bn or bn == "palm_pose" or "karma_sphere" in bn:
            continue
        f = bn.split("_")[0]
        if f not in FINGERS:
            continue
        gt, pos, sz = m.geom_type[g], d.geom_xpos[g], m.geom_size[g]
        if gt == mujoco.mjtGeom.mjGEOM_CAPSULE:
            half = d.geom_xmat[g].reshape(3, 3)[:, 2] * sz[1]
            surf = _seg_dist(sp, pos - half, pos + half) - sz[0]
        elif gt == mujoco.mjtGeom.mjGEOM_SPHERE:
            surf = float(np.linalg.norm(sp - pos)) - sz[0]
        else:
            continue
        gap = (surf - sphere_r) * 1e3
        rec = out.setdefault(f, {"gap_nearest_mm": gap, "nearest_body": bn,
                                 "gap_pad_mm": None})
        if gap < rec["gap_nearest_mm"]:
            rec["gap_nearest_mm"], rec["nearest_body"] = gap, bn
        if bn.endswith("_tip"):
            rec["gap_pad_mm"] = gap
    for f in out:
        out[f]["gap_nearest_mm"] = round(out[f]["gap_nearest_mm"], 3)
        if out[f]["gap_pad_mm"] is not None:
            out[f]["gap_pad_mm"] = round(out[f]["gap_pad_mm"], 3)
        out[f]["on_pad"] = out[f]["nearest_body"].endswith("_tip")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--variant", default="scaled")
    ap.add_argument("--pair", default="thumb-index")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = [r for r in json.loads(args.table.read_text())
            if r["variant"] == args.variant and r["pair"] == args.pair]
    if args.limit:
        rows = rows[:args.limit]

    res = []
    for r in rows:
        rdir = args.work / f"{r['design']}_{r['pair'].replace('-', '_')}_{r['variant']}"
        y = yaml.safe_load((rdir / "current.yaml").read_text())
        seed, cfg = y["seed"], (rdir / "metric.yaml")
        mcfg = yaml.safe_load(cfg.read_text())
        # The scaled variant multiplies every nominal length by L_ref/l_ref_nominal.
        scale = (r["l_ref_mm"] * 1e-3) / float(mcfg.get("l_ref_nominal_m", 0.2))
        sphere_r = float(mcfg["sphere_radius_m"]) * scale
        try:
            g = measure(r["mounts_mm"], seed["q"], seed["centre_world"], sphere_r)
        except Exception as exc:
            res.append({"design": r["design"], "error": f"{type(exc).__name__}: {exc}"})
            continue
        contact = {c.split("_")[0] for c in (r["seed_contact"] or "").split("+") if c}
        res.append({"design": r["design"], "retained": r["retained"],
                    "sphere_radius_mm": round(sphere_r * 1e3, 3),
                    "contact_fingers": sorted(contact), "fingers": g})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))

    ok = [r for r in res if "error" not in r]
    near, pad, onpad = [], [], []
    for r in ok:
        for f in r["contact_fingers"]:
            if f in r["fingers"]:
                near.append(r["fingers"][f]["gap_nearest_mm"])
                onpad.append(r["fingers"][f]["on_pad"])
                if r["fingers"][f]["gap_pad_mm"] is not None:
                    pad.append(r["fingers"][f]["gap_pad_mm"])
    print(f"{len(ok)} designs, {len(near)} contact fingers -> {args.out}")
    if near:
        print(f"  gap to the nearest finger surface: median {np.median(near):+.3f} mm, "
              f"90th pct {np.percentile(near, 90):+.3f} mm")
        print(f"  gap to the PAD sphere:             median {np.median(pad):+.3f} mm, "
              f"90th pct {np.percentile(pad, 90):+.3f} mm")
        print(f"  contacts whose nearest surface IS the pad: "
              f"{100.0 * np.mean(onpad):.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
