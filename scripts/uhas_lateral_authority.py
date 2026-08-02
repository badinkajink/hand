#!/usr/bin/env python3
"""Measure what the `yaw` joint physically does to each fingertip.

The UHAS lateral (theta) action asks each finger to swing *sideways across the palm*.
Our fingers have no dedicated abduction joint -- `yaw` is a hinge about the finger's own
+x mount axis. Whether that is a usable abduction DOF is a question about GEOMETRY AT THE
OPERATING POSE, not about the axis at q=0, so measure it there.

For a pose and a finger we sweep `yaw` across its range and decompose the fingertip's
displacement in the palm frame:

  u_hat   in-palm-plane unit vector along the finger's reach (mcp origin -> tip)
  l_hat   in-palm-plane unit vector perpendicular to u_hat   <- ABDUCTION
  n_hat   palm normal                                        <- parasitic (lift/dip)

Reported per finger:
  lat_mm      peak-to-peak fingertip travel along l_hat (mm)      "lateral authority"
  lat_per_rad lateral travel per radian of yaw (mm/rad)
  eff         mean |v.l_hat| / |v|  over the sweep, in [0,1]      "abduction efficiency"
  oop_mm      peak-to-peak travel along n_hat (mm)                "out-of-plane parasite"

A true abduction joint scores eff ~ 1.0 and oop ~ 0. A pure roll scores lat ~ 0.

Usage:
  .venv/bin/python scripts/uhas_lateral_authority.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import mujoco
import numpy as np

FINGERS = ("thumb", "index", "middle")


def tip_body(model: mujoco.MjModel, finger: str) -> int:
    """Body id of the fingertip, tolerating naming across generated/base scenes."""
    for name in (f"{finger}_tip", f"{finger}_distal", f"{finger}_ft"):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            return bid
    raise KeyError(f"no fingertip body for {finger}")


def mcp_origin(model: mujoco.MjModel, data: mujoco.MjData, finger: str) -> np.ndarray:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{finger}_mcp_frame")
    if bid < 0:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{finger}_yaw_frame")
    return data.xpos[bid].copy()


def palm_frame(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "palm")
    if bid < 0:
        return np.zeros(3), np.eye(3)
    return data.xpos[bid].copy(), data.xmat[bid].reshape(3, 3).copy()


def sweep_finger(
    model: mujoco.MjModel,
    qpos0: np.ndarray,
    finger: str,
    n: int = 61,
) -> dict | None:
    """Sweep `<finger>_yaw` over its joint range; decompose tip motion in the palm frame."""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{finger}_yaw")
    if jid < 0:
        return None
    adr = model.jnt_qposadr[jid]
    lo, hi = model.jnt_range[jid]
    if not model.jnt_limited[jid]:
        lo, hi = -1.1, 1.1

    data = mujoco.MjData(model)
    tb = tip_body(model, finger)

    # Palm frame + reach direction are taken at the sweep's MIDPOINT so the decomposition
    # is not biased toward either end of the range.
    data.qpos[:] = qpos0
    data.qpos[adr] = 0.5 * (lo + hi)
    mujoco.mj_kinematics(model, data)
    _, palm_R = palm_frame(model, data)
    n_hat = palm_R[:, 2]
    tip_mid = data.xpos[tb].copy()
    reach = tip_mid - mcp_origin(model, data, finger)
    reach_ip = reach - np.dot(reach, n_hat) * n_hat
    if np.linalg.norm(reach_ip) < 1e-9:
        # Finger points straight out of the palm plane; any in-plane direction will do.
        reach_ip = palm_R[:, 0]
    u_hat = reach_ip / np.linalg.norm(reach_ip)
    l_hat = np.cross(n_hat, u_hat)

    def trace(angs: np.ndarray) -> np.ndarray:
        out = np.empty((len(angs), 3))
        for i, a in enumerate(angs):
            data.qpos[:] = qpos0
            data.qpos[adr] = a
            mujoco.mj_kinematics(model, data)
            out[i] = data.xpos[tb]
        return out

    angles = np.linspace(lo, hi, n)
    tips = trace(angles)

    # Small-signal: what an incremental policy command actually buys. Centred on the POSE's
    # own yaw value, where lateral motion is first-order and the out-of-plane excursion is
    # only second-order -- the full-range p2p numbers understate efficiency badly.
    yaw0 = float(qpos0[adr])
    d = 0.2
    loc = trace(np.linspace(yaw0 - d, yaw0 + d, 21))
    dloc = np.diff(loc, axis=0)
    dn = np.linalg.norm(dloc, axis=1)
    ok = dn > 1e-12
    loc_eff = float(np.mean(np.abs(dloc[ok] @ l_hat) / dn[ok])) if ok.any() else 0.0
    loc_lat = float(np.ptp(loc @ l_hat)) * 1e3
    loc_oop = float(np.ptp(loc @ n_hat)) * 1e3

    lat = tips @ l_hat
    rad = tips @ u_hat
    oop = tips @ n_hat

    step = np.diff(tips, axis=0)
    step_norm = np.linalg.norm(step, axis=1)
    live = step_norm > 1e-12
    eff = float(np.mean(np.abs(step[live] @ l_hat) / step_norm[live])) if live.any() else 0.0

    span = float(hi - lo)
    lat_mm = float(lat.max() - lat.min()) * 1e3
    return {
        "finger": finger,
        "range_rad": [float(lo), float(hi)],
        "lat_mm": lat_mm,
        "lat_per_rad": lat_mm / span if span else 0.0,
        "radial_mm": float(rad.max() - rad.min()) * 1e3,
        "oop_mm": float(oop.max() - oop.min()) * 1e3,
        "eff": eff,
        "arc_mm": float(np.sum(step_norm)) * 1e3,
        "loc_eff": loc_eff,
        "loc_lat_per_rad": loc_lat / (2 * d),
        "loc_oop_mm": loc_oop,
    }


def poses_for(model: mujoco.MjModel, want: list[str] | None) -> dict[str, np.ndarray]:
    """q=0 plus every keyframe in the model."""
    out = {"q=0": np.zeros(model.nq)}
    for k in range(model.nkey):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, k) or f"key{k}"
        out[name] = model.key_qpos[k].copy()
    if want:
        out = {k: v for k, v in out.items() if k in want}
    return out


def report(path: pathlib.Path, label: str, want: list[str] | None) -> dict:
    model = mujoco.MjModel.from_xml_path(str(path))
    res = {"hand": label, "xml": str(path), "poses": {}}
    for pose_name, qpos in poses_for(model, want).items():
        rows = [r for f in FINGERS if (r := sweep_finger(model, qpos, f))]
        if rows:
            res["poses"][pose_name] = rows
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", action="append", default=[], metavar="LABEL=PATH")
    ap.add_argument("--pose", action="append", default=None, help="restrict to these poses")
    ap.add_argument("--json", type=pathlib.Path)
    args = ap.parse_args()

    specs = args.xml or [
        "baseline=assets/mjcf/baseline/hand_morphology_actuated.xml",
        "perp=assets/mjcf/perp/perp_hand_morphology_actuated.xml",
        "m05=results/uhas/mjcf/hand_m05_tp0d0150p0d0050p0d0110_ip0d0040p0d0020p0d0120_mp0d0250p0d0240p0d0160.xml",
        "H06_04=results/uhas/mjcf/hand_H0604_tp0d0080n0d0220p0d0110_ip0d0110p0d0120p0d0120_mn0d0060n0d0070p0d0160.xml",
    ]

    out = []
    for spec in specs:
        label, _, p = spec.partition("=")
        path = pathlib.Path(p or label)
        if not path.exists():
            print(f"  ! missing {path}", file=sys.stderr)
            continue
        out.append(report(path, label, args.pose))

    hdr = (
        f"{'hand':9s} {'pose':10s} {'finger':7s} | {'lat_mm':>7s} {'mm/rad':>7s} {'oop_mm':>7s} {'eff':>5s}"
        f" | {'mm/rad':>7s} {'oop_mm':>7s} {'eff':>5s}"
    )
    print(f"{'':38s} |{'-- full range --':^31s}|{'-- local +/-0.2 rad --':^23s}")
    print(hdr)
    print("-" * len(hdr))
    for h in out:
        for pose, rows in h["poses"].items():
            for r in rows:
                print(
                    f"{h['hand']:9s} {pose:10s} {r['finger']:7s} | "
                    f"{r['lat_mm']:7.1f} {r['lat_per_rad']:7.1f} {r['oop_mm']:7.1f} {r['eff']:5.2f} | "
                    f"{r['loc_lat_per_rad']:7.1f} {r['loc_oop_mm']:7.1f} {r['loc_eff']:5.2f}"
                )

    if args.json:
        args.json.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
