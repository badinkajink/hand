#!/usr/bin/env python3
"""How well a session's touch-offs fix the pad's place along the finger: the posterior standard deviation of the presses'
contact points from the registration fit's covariance (src/.../ur5_tip_calibration.py fit_registration: the tool on the
flange T_EM, the pad on the tool T_MP and the sphere's centre, 15 parameters with the program's priors), and how much
added touch-offs on the pad's tip end, where the normal has a part along the finger, would lower it.

The covariance is (J^T J)^-1 of the fit's residuals (touch-offs at REG_SIGMA's 0.03 mm, the homing pose, the priors),
evaluated at the session's fitted registration; it is also given scaled by the fit's own residual over 0.03 mm. A press's
contact point is the sphere's centre in the pad frame P (z_P along the finger, from the mount end) for the flange pose of
that press; its standard deviation along z_P and across the finger (y_P) is reported as a median over the plan's grid
presses. Candidate touch-offs are added greedily, each the one that lowers the median along z_P the most, at twist 0 and
the onset pose that the session's registration predicts. They are pad points that the sphere reaches without the finger's
other parts (scripts/ur5_tip_calibration_plan.py tip_reachable), either

  --nz 0.3        points whose normal has n_z of at least 0.3, touched at the sphere's top as the plan's touch-offs are, or
  --offset-mm 3   every reachable point with the sphere's top 3 mm beside the target along +-x_I or +-y_I (the plan's
                  "offset"), approached vertically until the pad first meets the sphere.

With --offset-mm every candidate must clear the margins of scripts/ur5_tip_calibration_check.py --clearance (mount 3.0,
finger 0.8, HEX-E 6.0 mm, plus --margin for a newly seated pad) with the session's registration, both as a fresh session
runs it (planned with the nominal tool) and as planned with the registration. On 2026-10-23 the first twelve, chosen
without that test, put the finger 0.8 to 1.2 mm into the sphere near the tip end and the mount within 0.1 mm of it.

Every vertical touch at the sphere's top has its contact normal along -z_I, so it sees no horizontal shift of the sphere's
centre: that position rests on the homing seating's prior (0.25 mm along the finger). On a curved part of the pad an
offset touch meets the sphere off its top, where the contact normal has a horizontal part.

    .venv/bin/python scripts/ur5_tip_calibration_touch_information.py docs/experiments/20261022-ur5_tip_calibration/lab/<session>
        [--add 8] [--nz 0.3 | --offset-mm 3] [--out touch_information.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src/morphohand/driver/reskin5x/host"))
import ur5_tip_calibration as C  # noqa: E402
import ur5_tip_calibration_plan as PL  # noqa: E402

SIG = C.REG_SIGMA


def session(folder):
    folder = Path(folder)
    plan = json.loads((folder / "plan.json").read_text())
    have = {a["id"] for a in plan["actions"]}
    for f in sorted(folder.glob("plan_resume_*.json")):
        for a in json.loads(f.read_text())["actions"]:
            if a["id"] not in have:
                plan["actions"].append(a)
                have.add(a["id"])
    reg = json.loads((folder / "registration.json").read_text())
    touches = json.loads((folder / "touches.json").read_text())
    return plan, reg, touches


class Info:
    """Residual rows of the registration fit, linearised at the session's fitted registration."""

    def __init__(self, geo, reg):
        self.geo = geo
        self.T_EM0, self.T_MP0 = np.array(reg["T_EM"], float), np.array(reg["T_MP"], float)
        self.c0 = np.array(reg["notes"]["register"]["c_B"], float)
        self.T_BI = np.array(reg["T_BI"], float)
        self.home = np.array(reg["T_BE_home"], float) if reg.get("T_BE_home") is not None else None
        self.sd_em = np.r_[np.radians(SIG["em_rot_deg"]), SIG["em_t_mm"]]
        self.sd_mp = np.r_[np.radians(SIG["mp_rot_deg"]), SIG["mp_t_mm"]]

    def unpack(self, x):
        return C.perturb(self.T_EM0, x[0:6]), C.perturb(self.T_MP0, x[6:12]), self.c0 + x[12:15]

    def touch_row(self, T_BE, kind="pad"):
        def f(x):
            T_EM, T_MP, c = self.unpack(x)
            if kind == "pad":
                q = C.apply(C.inv(T_BE @ T_EM @ T_MP), c)
                return np.array([(self.geo.surface_distance(q)[0] - self.geo.R) / SIG["pad_mm"]])
            q = C.apply(C.inv(T_BE @ T_EM), c)
            return np.array([(q @ self.geo.face_n - self.geo.face_d - self.geo.R) / SIG["face_mm"]])
        return jac(f)

    def prior_rows(self):
        rows = []
        if self.home is not None:
            R_BI = self.T_BI[:3, :3]
            T_MI_home = C.inv(self.geo.T_IM_home)

            def f(x):
                T_EM, T_MP, c = self.unpack(x)
                ch = C.apply(self.home @ T_EM @ T_MI_home, self.geo.c_I)
                return (R_BI.T @ (c - ch)) / np.array(SIG["home_mm"])
            rows.append(jac(f))
        P = np.zeros((12, 15))
        P[0:6, 0:6] = np.diag(1.0 / self.sd_em)
        P[6:12, 6:12] = np.diag(1.0 / self.sd_mp)
        rows.append(P)
        return np.vstack(rows)

    def contact_jac(self, T_BE):
        """d(sphere centre in P)/dx at the flange pose T_BE: (3, 15)."""
        def f(x):
            T_EM, T_MP, c = self.unpack(x)
            return C.apply(C.inv(T_BE @ T_EM @ T_MP), c)
        return jac(f)

    def T_BE_at(self, p, n, twist=0.0, standoff=0.0, offset=(0.0, 0.0)):
        T_IP = self.geo.T_IP(p, n, twist, standoff, offset)
        return self.T_BI @ T_IP @ C.inv(self.T_EM0 @ self.T_MP0)

    def onset(self, p, n, offset):
        """Flange pose where the pad first meets the sphere on a vertical approach with the target offset (I), the pad's
        contact point and the horizontal part of the contact normal; None if the pad does not meet it within 3 mm."""
        T_EP = self.T_EM0 @ self.T_MP0

        def gap(s):
            T_BE = self.T_BE_at(p, n, 0.0, s, offset)
            d, cp, nn = self.geo.surface_distance(C.apply(C.inv(T_BE @ T_EP), self.c0))
            return d - self.geo.R, T_BE, cp, nn
        lo, hi = -3.0, 3.0
        if gap(hi)[0] <= 0 or gap(lo)[0] >= 0:
            return None
        for _ in range(40):
            m = 0.5 * (lo + hi)
            lo, hi = (lo, m) if gap(m)[0] > 0 else (m, hi)
        _, T_BE, cp, nn = gap(hi)
        nI = (C.inv(self.T_BI) @ T_BE @ T_EP)[:3, :3] @ nn
        return T_BE, cp, float(np.hypot(nI[0], nI[1]))


def offset_action(geo, k, i, off, base=1.5, region=""):
    """A touch-off action aimed off the sphere's top (scripts/ur5_tip_calibration_offset_touches.py makes the same)."""
    p, n = geo.pts[i], geo.nrm[i]

    def gap(s_):
        return geo.surface_distance(C.apply(C.inv(geo.T_IP(p, n, 0.0, s_, off)), geo.c_I))[0] - geo.R
    lo, hi = -4.0, 4.0
    if gap(hi) <= 0 or gap(lo) >= 0:
        return None
    for _ in range(50):
        m = 0.5 * (lo + hi)
        lo, hi = (lo, m) if gap(m) > 0 else (m, hi)
    return dict(id=f"pad-off-{k:02d}", stage="pad", kind="touch", role="offset",
                target=dict(frame="P", p=[round(float(v), 4) for v in p], n=[round(float(v), 5) for v in n], index=int(i),
                            region=region),
                twist=0.0, offset=[float(v) for v in off], standoff=round(base + hi, 3), beyond_mm=round(1.2 - hi, 3),
                v=0.00015, stop_N=0.5, fit_N=0.4, onset_shift_mm=round(-hi, 3))


def clear_ok(folder, plan, actions, margin):
    """Ids of the touch-off actions that clear ur5_tip_calibration_check.py's margins plus margin (mm), as a fresh session
    runs them and as planned with the session's registration."""
    import tempfile
    import ur5_tip_calibration_check as CK
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(dict(plan, actions=actions), f, default=C.jsonable)
    ch = CK.Check(folder)
    ch.lines = []
    ch.registered_clearance(f.name, stages=("pad",))
    Path(f.name).unlink()
    need = dict(mount=3.0, finger=0.8, hex_e=6.0)
    per_r, per_a = ch.out["clearance_per_action"], ch.out["touch_as_run"]["per_action"]
    return {a["id"] for a in actions if all(min(per_r.get(a["id"], {}).get(g, -1e9), per_a.get(a["id"], {}).get(g, -1e9))
                                            >= need[g] + margin for g in need)}


def jac(f, h=1e-6):
    x0 = np.zeros(15)
    r0 = f(x0)
    J = np.empty((len(r0), 15))
    for j in range(15):
        dx = np.zeros(15)
        dx[j] = h
        J[:, j] = (f(x0 + dx) - r0) / h
    return J


def spread(cov, G):
    """Median over the presses of the contact's standard deviation along z_P and along y_P (mm)."""
    sz = np.sqrt(np.einsum("kj,jl,kl->k", G[:, 2, :], cov, G[:, 2, :]))
    sy = np.sqrt(np.einsum("kj,jl,kl->k", G[:, 1, :], cov, G[:, 1, :]))
    return float(np.median(sz)), float(np.median(sy))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("folder")
    ap.add_argument("--add", type=int, default=8)
    ap.add_argument("--nz", type=float, default=0.3)
    ap.add_argument("--offset-mm", type=float, default=0.0)
    ap.add_argument("--margin", type=float, default=0.3, help="mm beyond the checker's clearances (--offset-mm)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    plan, reg, touches = session(a.folder)
    geo = C.Geometry(plan)
    I = Info(geo, reg)
    rr = reg["notes"]["register"]
    scale = max(1.0, rr["rms_pad_mm"] / SIG["pad_mm"])
    rows = [I.touch_row(np.array(t["T_BE"], float), "pad" if t["kind"] == "pad" else "mount") for t in touches
            if t["kind"] in ("pad", "mount") and t.get("T_BE") is not None]
    J0 = np.vstack(rows + [I.prior_rows()])
    presses = [x for x in plan["actions"] if x.get("kind") == "press" and x.get("stage") == "grid"]
    G = np.array([I.contact_jac(I.T_BE_at(x["target"]["p"], x["target"]["n"], x.get("twist") or 0.0)) for x in presses])
    cov0 = np.linalg.inv(J0.T @ J0)
    z0, y0 = spread(cov0, G)
    nz_t = [t["target"]["n"][2] for t in touches if t["kind"] == "pad" and t.get("target")]
    print(f"{Path(a.folder).name}: {len(rows)} touch-offs (n_z up to {max(nz_t):.2f}), {len(presses)} grid presses; fit "
          f"{rr['rms_pad_mm'] * 1e3:.0f} um rms (x{scale:.2f} on the 0.03 mm of the fit's weights)")
    print(f"  contact point's sd at the presses (median): along the finger {z0:.3f} mm ({z0 * scale:.3f} scaled), across "
          f"{y0:.3f} mm ({y0 * scale:.3f})")
    # candidate touch-offs: (pad point, offset in I, row, horizontal part of the contact normal)
    reg_ = PL.regions(geo.nrm)
    if a.offset_mm > 0:
        pool = [i for i in range(len(geo.pts)) if reg_[i] and PL.tip_reachable(geo, geo.kind, i)[0]]
        o_ = a.offset_mm
        cands, acts = [], []
        for i in pool:
            for off in ((o_, 0.0), (-o_, 0.0), (0.0, o_), (0.0, -o_)):
                r_ = I.onset(geo.pts[i], geo.nrm[i], off)
                act = offset_action(geo, len(acts), i, off, region=str(reg_[i]))
                if r_ is not None and act is not None and np.linalg.norm(r_[1] - geo.pts[i]) <= 4.0:   # met near the target
                    cands.append((i, off, I.touch_row(r_[0]), r_[2], act["id"]))
                    acts.append(act)
        ok = clear_ok(a.folder, plan, acts, a.margin)
        n0 = len(cands)
        cands = [c for c in cands if c[4] in ok]
        print(f"  offset candidates ({o_:g} mm on {len(pool)} reachable points): {n0}, {len(cands)} clear the checker's "
              f"margins + {a.margin:g} mm; horizontal part of the contact normal median "
              f"{np.median([c[3] for c in cands]):.2f}", flush=True)
    else:
        pool = [int(i) for i in np.nonzero(geo.nrm[:, 2] >= a.nz)[0]]
        ok = [i for i in pool if PL.tip_reachable(geo, geo.kind, i)[0]]
        cands = [(i, (0.0, 0.0), I.touch_row(I.T_BE_at(geo.pts[i], geo.nrm[i])), 0.0, None) for i in ok]
        print(f"  candidates with n_z >= {a.nz:g}: {len(pool)}, reachable {len(ok)}", flush=True)
    J, chosen, steps = J0.copy(), [], []
    for _ in range(min(a.add, len(cands))):
        best = None
        for j, c in enumerate(cands):
            if j in chosen:
                continue
            Jt = np.vstack([J, c[2]])
            z, y = spread(np.linalg.inv(Jt.T @ Jt), G)
            if best is None or z < best[1]:
                best = (j, z, y)
        chosen.append(best[0])
        i, off, row, h, _ = cands[best[0]]
        J = np.vstack([J, row])
        p, n = geo.pts[i], geo.nrm[i]
        steps.append(dict(index=int(i), region=str(reg_[i]), p=p.round(3).tolist(), n=n.round(3).tolist(), offset_I=list(off),
                          normal_horizontal=h, sd_z_mm=best[1], sd_y_mm=best[2]))
        print(f"  + touch at p_P ({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f}) {reg_[i]}, n_z {n[2]:.2f}, offset {off}: along the "
              f"finger {best[1]:.3f} mm ({best[1] * scale:.3f} scaled), across {best[2]:.3f}", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(dict(folder=str(a.folder), n_touch=len(rows), n_press=len(presses),
                                               rms_pad_mm=rr["rms_pad_mm"], scale=scale, sd_z_mm=z0, sd_y_mm=y0,
                                               nz_min=a.nz, offset_mm=a.offset_mm, margin_mm=a.margin,
                                               n_candidates=len(cands),
                                               added=steps),
                                          indent=1) + "\n")
        print(f"-> {a.out}")


if __name__ == "__main__":
    main()
