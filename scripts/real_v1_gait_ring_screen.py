#!/usr/bin/env python3
"""Where each hand can hold a standing tool as a three-pad gait ring.

    python3 scripts/real_v1_gait_ring_screen.py --hands D1,D2,D3,D5,D6,D7 \
        --out docs/experiments/20260923-gait_ring/20260923-gait_ring_screen.json

The gait (probe_real_v1_gait, probe_real_v1_chain) plays a joint table solved for three pads on a
circle about the standing tool: grip radius r_obj + pad - squeeze and open radius + release_mm,
each swept through the stroke, and an approach radius at stroke 0 from which the pads close. The chain places that circle with the palm level,
the tool 4 mm ahead of the palm origin in x, and the pads at world azimuths 180 / +60 / -60 deg,
the tripod of the 2026-09-02 gait hands. On the D-hands that circle is 9-27 mm out of reach.

This screen searches, per hand, the tool's offset (dx, dy) in the palm frame, the ring's depth h
below the finger mounts and one azimuth per pad, for the ring the fingers reach over the whole
table, with the palm level. Reach is the fingertip IK residual (keyframe_ik.ik_finger); a
candidate is rejected when any finger link other than a pad penetrates the tool at any table
entry, when two pads sit closer than `--min-gap` deg, or when the three leave a gap wider than
`--max-gap` deg (no force closure). Kinematics only: no dynamics, no plant.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from morphohand.tools.keyframe_ik import FINGERS, TIPS, ik_finger  # noqa: E402

RUNS = {"D1": "20260920-sv1_w6689_b060_sq10_cal_tip", "D2": "20260920-sv1_w2360_b075_sq10_cal_tip",
        "D3": "20260920-sv1_u1364_b080_sq10_cal_tip", "D4": "20260920-g12_b095_sq10_cal_tip",
        "D5": "20260920-sv1_u0060_b75_sq10_cal_tip", "D6": "20260920-sv1_u0308_b050_cal_tip",
        "D7": "20260920-rv05_manual_b85_sq14_cal_tip"}
OBJ = "screwdriver_medium"
PAD_R = 0.01055


class Hand:
    def __init__(self, scene: Path):
        self.m = mujoco.MjModel.from_xml_path(str(scene))
        self.d = mujoco.MjData(self.m)
        mujoco.mj_resetDataKeyframe(self.m, self.d, self.m.key("open_ik").id)
        mujoco.mj_forward(self.m, self.d)
        self.palm = self.d.body("palm_pose").xpos.copy()
        assert np.allclose(self.d.body("palm_pose").xmat, np.eye(3).ravel(), atol=1e-6), "palm not level"
        g = [i for i in range(self.m.ngeom) if self.m.geom_bodyid[i] == self.m.body(OBJ).id][0]
        self.r_obj, self.half = float(self.m.geom_size[g, 0]), float(self.m.geom_size[g, 1])
        self.obj_adr = self.m.jnt_qposadr[self.m.body(OBJ).jntadr[0]]
        self.q0 = self.d.qpos.copy()
        self.tip_bodies = {self.m.body(TIPS[f]).id for f in FINGERS}
        self.hand_bodies = {i for i in range(self.m.nbody)
                            if any(k in self.m.body(i).name for k in ("thumb", "index", "middle"))}
        self.mount = {f: self.d.body(f"{f}_mount").xpos - self.palm for f in FINGERS}

    def stand(self, cx, cy, z_c):
        self.d.qpos[self.obj_adr:self.obj_adr + 7] = [cx, cy, z_c, 1, 0, 0, 0]

    def solve(self, f, pts, seed=None):
        """IK the finger through `pts` in order; residuals and joint rows."""
        adr = [self.m.jnt_qposadr[self.m.joint(j).id] for j in FINGERS[f]]
        if seed is not None:
            self.d.qpos[adr] = seed
        res, rows = [], []
        for p in pts:
            res.append(ik_finger(self.m, self.d, f, p, iters=150))
            rows.append(self.d.qpos[adr].copy())
        return np.array(res), rows

    def link_hits(self, f):
        """Depth (m) of the deepest contact between a non-pad link of finger f and the tool."""
        mujoco.mj_forward(self.m, self.d)
        worst = 0.0
        ob = self.m.body(OBJ).id
        for c in self.d.contact[:self.d.ncon]:
            b1, b2 = self.m.geom_bodyid[c.geom1], self.m.geom_bodyid[c.geom2]
            other = b2 if b1 == ob else (b1 if b2 == ob else None)
            if other is None or other in self.tip_bodies or other not in self.hand_bodies:
                continue
            if f not in self.m.body(other).name:
                continue
            worst = max(worst, -float(c.dist))
        return worst


def ring_cost(H: Hand, dx, dy, h, az, radii, phis, check_links=False):
    """Worst residual (m) of every pad over the table; with check_links, also the worst link depth."""
    c = H.palm + np.array([dx, dy, -h])
    worst, links, rows = 0.0, 0.0, {}
    for f in FINGERS:
        pts = [c + np.array([r * np.cos(az[f] + p), r * np.sin(az[f] + p), 0.0]) for r in radii for p in phis]
        H.d.qpos[:] = H.q0
        res, rw = H.solve(f, pts)
        worst = max(worst, float(res.max()))
        rows[f] = rw
        if check_links:
            H.stand(c[0], c[1], c[2])       # tool centred on the ring height: links near the ring are what hit
            for q in rw:
                H.d.qpos[[H.m.jnt_qposadr[H.m.joint(j).id] for j in FINGERS[f]]] = q
                links = max(links, H.link_hits(f))
    return worst, links, rows


def gaps_ok(az, min_gap, max_gap):
    a = np.sort(np.mod([az[f] for f in FINGERS], 2 * np.pi))
    g = np.diff(np.r_[a, a[0] + 2 * np.pi])
    return bool(g.min() >= np.radians(min_gap) and g.max() <= np.radians(max_gap)), np.degrees(g)


def screen(H: Hand, a) -> dict:
    r_grip = H.r_obj + PAD_R - a.squeeze / 1000
    r_open = r_grip + (a.squeeze + a.release) / 1000
    r_app = H.r_obj + PAD_R + a.approach / 1000
    stroke = np.radians(a.stroke)
    phis_fast = [0.0]
    # 1. coarse: per (dx, dy, h), each pad takes its own best azimuth at the grip radius
    azg = {f: np.radians(np.arange(-180, 180, 10)) for f in FINGERS}
    cand = []
    for dx in np.arange(-0.030, 0.0451, 0.010):
        for dy in np.arange(-0.060, 0.0401, 0.010):
            for h in np.arange(0.030, 0.0801, 0.010):
                c = H.palm + np.array([dx, dy, -h])
                best = {}
                for f in FINGERS:
                    H.d.qpos[:] = H.q0
                    rs = [H.solve(f, [c + np.array([r_grip * np.cos(t), r_grip * np.sin(t), 0])])[0][0] for t in azg[f]]
                    best[f] = (azg[f][int(np.argmin(rs))], float(np.min(rs)))
                az = {f: best[f][0] for f in FINGERS}
                ok, _ = gaps_ok(az, a.min_gap, a.max_gap)
                cand.append((max(b[1] for b in best.values()) + (0 if ok else 1.0), dx, dy, h, az))
    cand.sort(key=lambda t: t[0])
    # 2. refine the best: coordinate search on (dx, dy, h, az) against the whole table
    radii = [r_grip, r_open]
    phis = list(np.linspace(0.0, stroke, 4))

    def table_cost(dx, dy, h, az, check_links=False):
        w, l1, _ = ring_cost(H, dx, dy, h, az, radii, phis, check_links)
        wa, l2, _ = ring_cost(H, dx, dy, h, az, [r_app], [0.0], check_links)
        return w, wa, max(l1, l2)

    out = []
    for c0 in cand[:a.refine]:
        x = [c0[1], c0[2], c0[3], c0[4]["thumb"], c0[4]["index"], c0[4]["middle"]]

        def cost(x):
            az = dict(zip(FINGERS, x[3:]))
            ok, _ = gaps_ok(az, a.min_gap, a.max_gap)
            w, wa, _ = table_cost(x[0], x[1], x[2], az)
            return max(w, wa) + (0 if ok else 1.0)

        fx = cost(x)
        steps = [0.004, 0.004, 0.004, np.radians(8), np.radians(8), np.radians(8)]
        for _ in range(a.iters):
            improved = False
            for i in range(6):
                for sgn in (1, -1):
                    y = list(x); y[i] += sgn * steps[i]
                    fy = cost(y)
                    if fy < fx - 1e-6:
                        x, fx, improved = y, fy, True
            if not improved:
                steps = [s / 2 for s in steps]
                if steps[0] < 0.0005:
                    break
        az = dict(zip(FINGERS, x[3:]))
        w, wa, links = table_cost(x[0], x[1], x[2], az, check_links=True)
        ok, gaps = gaps_ok(az, a.min_gap, a.max_gap)
        out.append({"worst_mm": round(max(w, wa) * 1000, 2), "gait_mm": round(w * 1000, 2), "approach_mm": round(wa * 1000, 2),
                    "link_depth_mm": round(links * 1000, 2), "gaps_ok": ok,
                    "dx_mm": round(x[0] * 1000, 1), "dy_mm": round(x[1] * 1000, 1), "h_mm": round(x[2] * 1000, 1),
                    "az_deg": {f: round(float(np.degrees(np.arctan2(np.sin(v), np.cos(v)))), 1) for f, v in az.items()},
                    "gaps_deg": [round(float(g), 1) for g in gaps]})
    out.sort(key=lambda o: (not o["gaps_ok"], o["link_depth_mm"] > a.link_tol, o["worst_mm"]))
    # the chain's own ring for comparison: tool 4 mm ahead in x, az 180/+60/-60, best depth
    base = []
    for h in np.arange(0.030, 0.0801, 0.005):
        az0 = {"thumb": np.pi, "index": np.pi / 3, "middle": -np.pi / 3}
        w, wa, links = table_cost(0.004, 0.0, h, az0, check_links=True)
        base.append({"h_mm": round(h * 1000, 1), "worst_mm": round(max(w, wa) * 1000, 2), "gait_mm": round(w * 1000, 2),
                     "approach_mm": round(wa * 1000, 2), "link_depth_mm": round(links * 1000, 2)})
    return {"best": out[0], "candidates": out, "chain_default": min(base, key=lambda b: b["worst_mm"]),
            "mount_mm": {f: [round(float(v) * 1000, 1) for v in H.mount[f][:2]] for f in FINGERS},
            "radii_mm": [round(r * 1000, 2) for r in (r_grip, r_open, r_app)], "stroke_deg": a.stroke}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hands", default="D1,D2,D3,D5,D6,D7")
    ap.add_argument("--squeeze", type=float, default=2.0, help="mm inside the surface at the grip radius")
    ap.add_argument("--release", type=float, default=6.0, help="mm outside the surface at the open radius")
    ap.add_argument("--stroke", type=float, default=30.0, help="deg of azimuth each pad sweeps")
    ap.add_argument("--approach", type=float, default=12.0, help="mm between pad and tool surface at the approach radius")
    ap.add_argument("--min-gap", type=float, default=70.0)
    ap.add_argument("--max-gap", type=float, default=160.0)
    ap.add_argument("--link-tol", type=float, default=1.0, help="mm of link-tool penetration tolerated")
    ap.add_argument("--refine", type=int, default=6)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    res = {}
    for hand in a.hands.split(","):
        H = Hand(ROOT / "results/phase1/real_v1" / RUNS[hand] / "frozen_scene.xml")
        r = screen(H, a)
        res[hand] = r
        b, c = r["best"], r["chain_default"]
        print(f"{hand}: best {b['worst_mm']:5.2f} mm (gait {b['gait_mm']}, approach {b['approach_mm']}, links {b['link_depth_mm']:.1f}, gaps {b['gaps_deg']}) at dx {b['dx_mm']} "
              f"dy {b['dy_mm']} h {b['h_mm']} az {b['az_deg']} | chain's ring {c['worst_mm']:.1f} mm at h {c['h_mm']}", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"args": vars(a) | {"out": str(a.out)}, "hands": res}, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
