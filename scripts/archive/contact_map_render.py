#!/usr/bin/env python3
"""Draw the contact map on the object's surface, unrolled, over a scripted grasp.

    uv run python scripts/contact_map_render.py \
        --scenes assets/mjcf/perp/scenes/scene_screwdriver_medium_perp.xml \
                 results/spherepack/perp_obj_e0.005.xml \
        --labels "capsule baseline" "object packed eps=0.005" \
        --out docs/experiments/20260803-sphere_contact/contact_map.png

THE POINT. Everything in this repo reads contact as three scalars -- one force per fingertip
body -- because the analytic colliders emit one contact per pair. That representation cannot
express WHERE on the tool a finger is riding, which is precisely the quantity a rolling
reorient is made of. This script asks the object where it is being touched, in ITS OWN surface
coordinates, and unrolls the cylinder into (theta, axial) so the answer is a picture.

Contact position is transformed into the object's local frame and mapped to the cylindrical
surface: axial s = local +Z (MuJoCo cylinders run along local Z), angle theta = atan2(y, x).
Both are frame-invariant -- as the screwdriver rolls, a finger that stays put in the WORLD
sweeps in theta, and that sweep is the roll, read directly off the contact set rather than
inferred from a quaternion.

Two panels per scene:
  top    - force-weighted contact map over the whole rollout, unrolled surface
  bottom - contact centroid theta vs time, which is the "contact trajectory" a Pollard-style
           tracking controller would be asked to follow. A representation that cannot draw
           this curve cannot be asked to track it.

Run it on a capsule scene and a packed scene side by side; the comparison IS the result.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

PHASES = ("open", "closed", "press")


def rollout(scene: Path, phases=PHASES, hold: int = 260, blend: int = 240):
    """Drive ctrl through the keyframe sequence and record every hand<->object contact."""
    m = mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m)
    obj_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "screwdriver_medium")
    if obj_bid < 0:
        raise SystemExit(f"{scene}: no body named screwdriver_medium")

    kids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, k) for k in phases]
    if any(k < 0 for k in kids):
        raise SystemExit(f"{scene}: missing one of keyframes {phases}")

    mujoco.mj_resetDataKeyframe(m, d, kids[0])
    ctrls = [m.key_ctrl[k].copy() for k in kids]

    # half-length of the object, for the axial axis -- read it off the model so a packed
    # scene (where the cylinder became spheres) still reports the true extent.
    zs = [m.geom_pos[g][2] for g in range(m.ngeom) if m.geom_bodyid[g] == obj_bid]
    half_len = float(max(np.abs(zs))) if len(zs) > 1 else float(
        m.geom_size[[g for g in range(m.ngeom) if m.geom_bodyid[g] == obj_bid][0]][1]
    )

    recs: list[tuple[int, float, float, float, str]] = []
    step = 0
    for a, b in zip(ctrls[:-1], ctrls[1:]):
        for i in range(blend + hold):
            t = min(1.0, i / max(blend, 1))
            d.ctrl[:] = a + t * (b - a)
            mujoco.mj_step(m, d)
            step += 1

            R = d.xmat[obj_bid].reshape(3, 3)
            p = d.xpos[obj_bid]
            f6 = np.zeros(6)
            for ci in range(d.ncon):
                c = d.contact[ci]
                b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
                if obj_bid not in (b1, b2):
                    continue
                other = b2 if b1 == obj_bid else b1
                nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, other)
                if nm is None:  # world/floor
                    continue
                loc = R.T @ (c.pos - p)
                mujoco.mj_contactForce(m, d, ci, f6)
                recs.append(
                    (step, float(np.arctan2(loc[1], loc[0])), float(loc[2]), abs(float(f6[0])), nm)
                )
    return recs, half_len, step


# Okabe-Ito: the standard colourblind-safe categorical set, assigned in fixed order so a
# finger keeps its colour across panels. Never cycled -- there are only ever three fingers.
FINGER_COLORS = {"thumb": "#0072B2", "index": "#E69F00", "middle": "#009E73"}


def _finger_of(body: str) -> str:
    return body.split("_")[0]


def draw(ax_map, ax_traj, recs, half_len, n_steps, label: str, vmax: float, legend: bool):
    if not recs:
        ax_map.text(0.5, 0.5, "no hand contacts", ha="center", va="center",
                    transform=ax_map.transAxes, color="#8a8a8a")
        return

    step, th, s, f, _ = (np.array(x) for x in zip(*recs))
    bodies = np.array([r[4] for r in recs])

    # force-weighted occupancy on the unrolled surface. Sequential single hue, light->dark:
    # this encodes magnitude, so it must not be a rainbow.
    nb_t, nb_s = 72, 48
    H, xe, ye = np.histogram2d(
        th, s, bins=[nb_t, nb_s],
        range=[[-np.pi, np.pi], [-half_len, half_len]], weights=f,
    )
    ax_map.pcolormesh(np.degrees(xe), np.array(ye) * 1000.0, H.T,
                      cmap="Blues", vmin=0.0, vmax=vmax, shading="auto")
    ax_map.set_title(f"{label}   ({len(recs)} contact-samples)", fontsize=9, pad=6)
    ax_map.set_ylabel("axial position (mm)", fontsize=8)
    ax_map.set_xlim(-180, 180)
    ax_map.set_xticks([-180, -90, 0, 90, 180])
    ax_map.tick_params(labelsize=7, length=2, color="#bbbbbb")
    for sp in ax_map.spines.values():
        sp.set_color("#dddddd")

    # Contact trajectory, PER FINGER. Aggregating across fingers is meaningless here: the
    # fingers sit on opposite sides of the barrel, so a circular mean over all of them lands
    # near the +/-180 seam and flips every step. What a tracking controller needs is where
    # EACH finger rides.
    #
    # theta is also UNWRAPPED per finger, because the quantity of interest is the roll: a
    # finger that keeps sweeping around the barrel should read as a monotonic drift, not as a
    # sawtooth at the seam. A flat line here means the contact is pinned -- no rolling.
    for finger, color in FINGER_COLORS.items():
        sel = np.array([_finger_of(b) == finger for b in bodies])
        if not sel.any():
            continue
        fs, ft, ff = step[sel], th[sel], f[sel]
        order = np.argsort(fs)
        fs, ft, ff = fs[order], ft[order], ff[order]
        uniq, idx = np.unique(fs, return_index=True)
        bounds = list(idx[1:]) + [len(fs)]
        cs, cx = [], []
        for st, a, b in zip(uniq, idx, bounds):
            w = ff[a:b]
            if w.sum() <= 1e-12:
                continue
            cs.append(st)
            cx.append(np.arctan2((np.sin(ft[a:b]) * w).sum(), (np.cos(ft[a:b]) * w).sum()))
        if not cs:
            continue
        ax_traj.plot(cs, np.degrees(np.unwrap(np.array(cx))), lw=2, color=color,
                     solid_capstyle="round", label=finger)

    if legend:
        ax_traj.legend(frameon=False, fontsize=7, loc="upper left", ncol=3)
    ax_traj.set_xlim(0, n_steps)
    ax_traj.set_xlabel("step", fontsize=8)
    ax_traj.set_ylabel("contact θ, unwrapped (deg)", fontsize=8)
    ax_traj.grid(True, lw=0.5, color="#eeeeee")
    ax_traj.set_axisbelow(True)
    ax_traj.tick_params(labelsize=7, length=2, color="#bbbbbb")
    for sp in ax_traj.spines.values():
        sp.set_color("#dddddd")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", type=Path, nargs="+", required=True)
    ap.add_argument("--labels", nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    labels = args.labels or [p.stem for p in args.scenes]
    if len(labels) != len(args.scenes):
        raise SystemExit("--labels must match --scenes in length")

    results = [rollout(p) for p in args.scenes]

    # Shared colour scale, or the panels cannot be compared at all. Ceiling is the 95th
    # percentile of the OCCUPIED cells, not the max: a couple of long-dwell cells otherwise
    # take the whole ramp and every real contact renders as near-white.
    def _ceiling(recs, hl) -> float:
        if not recs:
            return 0.0
        H = np.histogram2d(
            [r[1] for r in recs], [r[2] for r in recs], bins=[72, 48],
            range=[[-np.pi, np.pi], [-hl, hl]], weights=[r[3] for r in recs],
        )[0]
        occupied = H[H > 0]
        return float(np.percentile(occupied, 95)) if occupied.size else 0.0

    vmax = max(_ceiling(recs, hl) for recs, hl, _ in results) or 1.0

    n = len(args.scenes)
    fig, axes = plt.subplots(2, n, figsize=(5.4 * n, 6.2),
                             gridspec_kw={"height_ratios": [2.0, 1.0]}, squeeze=False)
    for i, ((recs, hl, ns), lab) in enumerate(zip(results, labels)):
        draw(axes[0][i], axes[1][i], recs, hl, ns, lab, vmax, legend=(i == 0))
    axes[0][0].set_xlabel("")
    fig.suptitle("Contact map on the object surface, unrolled  ·  force-weighted",
                 fontsize=11, y=0.98)
    fig.text(0.5, 0.925, "θ = angle around the barrel, axial = along it; "
                         "darker = more accumulated normal force",
             ha="center", fontsize=8, color="#6b6b6b")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")
    for (recs, _, _), lab in zip(results, labels):
        print(f"  {lab:<34} contact-samples={len(recs)}")


if __name__ == "__main__":
    main()
