#!/usr/bin/env python3
"""Ground truth of a probe-mode calibration session (the inverted rig, 2026-11-02 on: the tip held on its seat, the
sphere or the half rod on the HEX-E), in the holds format of scripts/ur5_tip_calibration_analysis.py, so that the board
GP scripts (ur5_tip_calibration_board_gp.py, ur5_tip_calibration_v3_data_curve.py) read it unchanged.

For every hold (a press's force level; a sweep's level, its offsets' holds and its slides' end holds; a twist's or a
roll's turns) the ground truth comes from the robot, the HEX-E and the session's registration (registration.json):

  probe     the flange's mean pose during the hold, the HEX-E's deflection under the measured wrench (the data sheet's
            compliance, as the tip-mode analysis takes it) and T_EM give the probe's frame M in the base; the fitted
            c_M (the sphere's centre, or a point of the rod's axis) and, for the rod, axis_M turned by the fit's turn
  pad       fixed on its seat: T_BP = T_BI @ T_AP from the registration (the program's Registration.T_BP)
  contact   sphere: the pad's surface point whose touching sphere's centre lies nearest c_P, depth R less the centre's
            distance; rod: the pad's point nearest the axis within half_len of c_P along it, depth R less that distance
  force     the HEX-E's change from its zero (the rests before and after the action, interpolated in time), through K:
            the force on the probe; the pad takes its opposite. No gravity term: a probe session has no gravity fit,
            and a press or a sweep keeps the probe's orientation from its rest (a twist's or a roll's turn changes the
            payload's moment by up to about 0.1 N, left in)
  torsion   the pad's torque about its normal at the contact point (the probe's, opposite)
  board     the held board's hold mean against its rests before and after, each chip corrected for its die temperature

--joint JSON (scripts/ur5_rod_joint_registration.py, 11-04): the pad pose T_AP, the rod's axis point c_M and its axis
of the joint rod fit's row for the session's seat replace the session's registration. The session's T_BI and T_EM must
equal those of the row's session (the night's second halves and extras ran on their first half's registration).

The rod's contact label stays the pad's point nearest the axis (11-04 06:2x, on the simulated rod session's 58 holds):
it stands 0.095 mm (median) from the simulator's centre of pressure, the penetration-weighted centroid 0.191 mm. Turning
the axis by 0.5 or 1 deg (across or about z_M) moves neither label's median; by 2.5 deg the nearest point moves 0.69 mm
and the centroid 0.41 mm, by 5 deg about 1 mm and 0.88 mm. The label's error is set by the registration's turn.

    .venv/bin/python scripts/ur5_probe_calibration_analysis.py SESSION_DIR [--plans PLAN ...] [--joint JSON] [--out DIR]

Checked on a simulated rod session (11-03, seat 2's half b cut to a reference press, an 8 N sweep, a twist and a roll;
58 holds): the contact point stands 0.095 mm (median) from the simulator's centre of pressure and the force 0.106 N from
its force. One point (face-692) stood 0.65 mm off across the rod's axis, from the registration: pass 1 sees little of the
rod's place across its axis, and the fit put c_M 0.88 mm from the simulator's.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ur5_tip_calibration_analysis as AN  # noqa: E402

C, SIM, W = AN.C, AN.SIM, AN.W
OUT = ROOT / "docs/experiments/20261030-inverted_rig_seat_block/data/holds"
KINDS = ("press", "sweep", "twist", "roll")
PLANS = []           # --plans: more program plans whose actions a resumed run took (a resume keeps the seed's plan.json)
JOINT = []           # --joint: a joint rod registration whose row for the seat replaces the session's


class ProbeSession(AN.Session):
    def __init__(self, sd):                             # the tip-mode Session's streams, without its gravity fit
        self.sd = Path(sd)
        self.excluded = set()
        self.d = AN.load(sd)
        have = {x["id"] for x in self.d["plan"]["actions"]}
        for f in PLANS:                                 # the night's plans (lab, 19:47 on 11-03: a resume does not rewrite
            for x in json.loads(Path(f).read_text())["actions"]:                # plan.json)
                if x["id"] not in have:
                    self.d["plan"]["actions"].append(x)
                    have.add(x["id"])
        self.geo = C.Geometry(self.d["plan"])
        if self.geo.probe is None:
            raise SystemExit(f"{sd}: not a probe-mode session (plan.geometry.probe)")
        if (self.sd / "held.bin").exists():             # a real run: the held tips' boards behind the mux, the seat's
            hb = np.fromfile(self.sd / "held.bin", C.HeldBoardStream.DTYPE)     # channel (the simulator writes the held
            hb = hb[hb["b"] == int(self.geo.seat["held_channel"])]              # board into board.bin)
            self.d["t_b"], self.d["r_b"] = hb["t"], hb["r"].reshape(-1, 5, 4).astype(float)
        self.joint = None
        if JOINT:
            J = json.loads(Path(JOINT[0]).read_text())
            seat = int(self.d["plan"]["geometry"]["seat"]["index"]) + 1
            jr = next(r for r in J["rows"] if r["seat"] == seat)
            src = next(x for x in J["sessions"] if x.rstrip("/").endswith(jr["session"]))
            rb = json.loads(((ROOT / src) / "registration.json").read_text())
            for k in ("T_BI", "T_EM"):
                if np.abs(np.asarray(rb[k], float) - np.asarray(self.d["reg"][k], float)).max() > 1e-6:
                    raise SystemExit(f"{sd}: its {k} is not that of {jr['session']}, the joint fit's seat {seat}")
            if "axis_M" not in jr["joint"]:
                raise SystemExit(f"{JOINT[0]}: no joint axis_M (rerun ur5_rod_joint_registration.py)")
            dP = C.pose_error(np.asarray(self.d["reg"]["T_AP"], float), np.asarray(jr["T_AP_joint"], float))
            self.d["reg"] = dict(self.d["reg"], T_AP=jr["T_AP_joint"], c_M=jr["joint"]["c_M"], axis_M=jr["joint"]["axis_M"])
            self.joint = dict(file=str(JOINT[0]), seat=seat, session=jr["session"],
                              pad_vs_session_mm_deg=[round(float(v), 4) for v in dP])
        self.reg = C.Registration(self.geo).load(self.d["reg"])
        self.T_BP = self.reg.T_BP()
        self.T_PB = C.inv(self.T_BP)
        self.c_M = np.asarray(self.reg.c_M, float)
        self.axis_M = None
        if self.geo.kind == "rod":                      # the fit's turn of the axis, when the registration has one
            ax = self.d["reg"].get("axis_M")
            self.axis_M = np.asarray(ax if ax is not None else self.geo.axis_M, float)
            self.axis_M /= np.linalg.norm(self.axis_M)
        self.Kinv, self.Ktinv = np.linalg.inv(self.reg.K), np.linalg.inv(self.reg.Kt)
        self.o_E = self.reg.T_ES[:3, 3]
        self.R_SE = self.reg.T_ES[:3, :3].T
        kap = W.load_thermal()
        self.kappa = kap if kap is not None else np.full(5, -0.0067)
        b, T = self.d["r_b"][:, :, 1:], self.d["r_b"][:, :, 0]
        c = np.cumsum(np.r_[np.zeros((1, 5)), T], axis=0)
        i0 = np.searchsorted(self.d["t_b"], self.d["t_b"] - 3.0)
        i1 = np.arange(1, len(T) + 1)
        Tm = (c[i1] - c[i0]) / (i1 - i0)[:, None]
        self.b = W.thermal_correct(b, Tm, self.kappa).reshape(-1, 15)
        self.T_chip = T
        self.ends = {e["id"]: e for e in self.d["ev"] if e.get("event") == "end"}
        self.acts = {a["id"]: a for a in self.d["plan"]["actions"]}
        g = self.geo
        Cd, Nd = W.subdivide(g.pts, g.nrm, g.tris, 3)
        self.Cd, self.Nd = Cd, Nd / np.linalg.norm(Nd, axis=1, keepdims=True)
        self.Sd = self.Cd + g.R * self.Nd

    def truth(self, t0, t1):
        """The simulator's contact during the hold (sim_truth.bin), without the tip-mode sphere centre."""
        tr = self.d["truth"]
        if tr is None:
            return None
        rows = tr[(tr[:, 0] >= t0) & (tr[:, 0] <= t1)]
        rows = rows[rows[:, 12] > 0]
        if not len(rows):
            return None
        n = rows[:, 7:10].mean(axis=0)
        return dict(f=rows[:, 1:4].mean(axis=0), cop=rows[:, 4:7].mean(axis=0), n=n / np.linalg.norm(n),
                    tn=float(rows[:, 10].mean()), kind=int(rows[0, 12]), c=np.full(3, np.nan))

    def contact(self, T_BM):
        """(contact point, outward normal, depth (mm), the rod's axis in P or zeros) for the probe at T_BM."""
        c_P = C.apply(self.T_PB @ T_BM, self.c_M)
        if self.geo.kind != "rod":
            d, _, _ = self.geo.surface_distance(c_P)
            _, n_c = self.contact_on_surface(c_P)
            return c_P - self.geo.R * n_c, n_c, float(self.geo.R - d), np.zeros(3), c_P
        a_P = (self.T_PB @ T_BM)[:3, :3] @ self.axis_M
        r = self.Cd - c_P
        y = r @ a_P
        m = np.abs(y) <= self.geo.half_len
        rp = r - y[:, None] * a_P[None]
        dist = np.where(m, np.linalg.norm(rp, axis=1), np.inf)
        k = int(np.argmin(dist))                        # the line's deepest point (a depth-weighted centre of the line
        return self.Cd[k], self.Nd[k], float(self.geo.R - dist[k]), a_P, c_P     # stood farther from the simulator's
        #                                                 centre of pressure: 0.165 against 0.095 mm, 11-03 19:5x)

    def ground_truth(self, rec, t0, t1, T_rest_pose):
        T_BE = self.pose(t0, t1)
        w = self.win(self.d["ft"][:, 0], self.d["ft"][:, 3:9], t0, t1)
        if T_BE is None or not len(w):
            return None
        w = w.mean(axis=0)
        tare, dz = self.zero(rec, 0.5 * (t0 + t1))
        f_E = self.Kinv @ (w[:3] - tare[:3])            # on the probe, from the pad
        t_E = self.Ktinv @ (w[3:] - tare[3:]) * 1e3     # N mm about the HEX-E's origin
        cw = SIM.HEX_COMPLIANCE                         # the probe deflected under it, in the HEX-E's frame
        dw = np.array(cw["r_rad_Nmm"]) * (self.R_SE @ t_E)
        dtr = np.array(cw["f_mm_N"]) * (self.R_SE @ f_E)
        T_ES, T_EM = self.reg.T_ES, self.reg.T_EM
        T_BM = T_BE @ T_ES @ C.Tm(C.exp_so3(dw), dtr) @ C.inv(T_ES) @ T_EM
        p_c, n_c, depth, a_P, c_P = self.contact(T_BM)
        p0, _, _, _, _ = self.contact(T_BE @ T_EM)
        R_PE = (self.T_PB @ T_BE)[:3, :3]
        f_P = -(R_PE @ f_E)                             # on the pad
        fn = float(-f_P @ n_c)
        p_E = C.apply(C.inv(T_BE) @ self.T_BP, p_c)
        tc_E = t_E - np.cross(p_E - self.o_E, f_E)      # the probe's torque about the contact
        tn = float(-(tc_E @ (R_PE.T @ n_c)))            # the pad's, about its normal
        return dict(T_BE=T_BE, c_P=c_P, p=p_c, n=n_c, depth=depth, f=f_P, f_n=fn, f_t=f_P + fn * n_c, tn=tn,
                    axis_P=a_P, deflection_mm=float(np.linalg.norm(p_c - p0)), ft_rest_drift_N=float(np.linalg.norm(dz[:3])))

    def holds(self):
        out = []
        for aid, e in self.ends.items():
            if e.get("status") != "ok" or aid not in self.acts:
                continue
            a, rec = self.acts[aid], e.get("rec") or {}
            if a["kind"] not in KINDS or a.get("stage") != "grid" or not rec.get("levels") or rec.get("T_rest") is None:
                continue
            T_rest = np.array(rec["T_rest"])
            win = [(lv["t"], "press" if a["kind"] != "sweep" else "sweep",
                    dict(level_N=lv["N"], turn_deg=0.0, dir_deg=None)) for lv in rec["levels"]]
            for o in rec.get("offsets", []):            # a sweep's offsets: tangential load inside the friction cone
                win.append((o["t"], "offset", dict(level_N=o["N"], turn_deg=0.0, dir_deg=o["az_deg"])))
            for s in rec.get("slides", []):             # its slides' end holds: the contact slid s mm
                win.append((s["t_hold"], "slide", dict(level_N=s["at_N"], turn_deg=0.0, dir_deg=s["az_deg"])))
            for mv in rec.get("moves", []):
                for tu in mv["turns"]:
                    win.append((tu["t"], a["kind"], dict(level_N=a.get("press_N"), turn_deg=tu["deg"],
                                                         dir_deg=mv.get("dir_deg"))))
            for (t0, t1), kind, meta in win:
                gt = self.ground_truth(rec, t0, t1, T_rest)
                if gt is None:
                    continue
                bb = self.win(self.d["t_b"], self.b, t0, t1)
                if not len(bb):
                    continue
                rest, drift = self.rest_board(rec, 0.5 * (t0 + t1))
                Tc = self.win(self.d["t_b"], self.T_chip, t0, t1)
                Tc0 = self.win(self.d["t_b"], self.T_chip, *rec["t_rest0"])
                tg = a["target"]
                out.append(dict(id=aid, stage=a["stage"], kind=kind, role=a.get("role"),
                                region=a.get("region") or tg.get("region"), target=np.array(tg["p"], float),
                                twist=a.get("twist", 0.0), t=0.5 * (t0 + t1), n_rec=len(bb), d_board=bb.mean(axis=0) - rest,
                                rest_drift_uT=drift, g_P=self.T_BP[:3, :3].T @ np.array([0.0, 0.0, -1.0]),
                                R_BE=gt["T_BE"][:3, :3].ravel(),
                                T_chip=Tc.mean(axis=0) if len(Tc) else np.full(5, np.nan),
                                dT_chip=(Tc.mean(axis=0) - Tc0.mean(axis=0)) if len(Tc) and len(Tc0) else np.full(5, np.nan),
                                dt_zero=0.5 * (t0 + t1) - float(np.mean(rec["t_rest0"])), est={}, truth=self.truth(t0, t1),
                                **meta, **gt))
        out.sort(key=lambda r: r["t"])
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("session")
    ap.add_argument("--out", default="", help="folder for holds.npz and summary.json (default data/holds/<session>)")
    ap.add_argument("--plans", nargs="*", default=[], help="program plans of the session's resumes (their targets)")
    ap.add_argument("--joint", default="", help="a joint rod registration (ur5_rod_joint_registration.py) in place of "
                    "the session's")
    a = ap.parse_args()
    PLANS[:] = a.plans
    JOINT[:] = [a.joint] if a.joint else []
    ses = ProbeSession(a.session)
    H = ses.holds()
    out = Path(a.out) if a.out else OUT / Path(a.session).name
    out.mkdir(parents=True, exist_ok=True)
    AN.save(H, out / "holds.npz")
    kinds = {}
    for h in H:
        kinds[h["kind"]] = kinds.get(h["kind"], 0) + 1
    fn = np.array([h["f_n"] for h in H])
    dep = np.array([h["depth"] for h in H])
    db = np.array([np.linalg.norm(h["d_board"]) for h in H])
    m = fn >= 1.0
    s = dict(session=str(a.session), tool=ses.geo.kind, registration=ses.joint or "the session's", holds=len(H), kinds=kinds,
             f_n_N=[round(float(np.percentile(fn, q)), 2) for q in (5, 50, 95)] if len(H) else None,
             depth_mm_at_1N_and_over=[round(float(np.percentile(dep[m], q)), 3) for q in (5, 50, 95)] if m.any() else None,
             board_uT_per_N_median=round(float(np.median(db[m] / fn[m])), 2) if m.any() else None)
    if H and H[0]["truth"] is not None:                 # a simulated session: the chain against the simulator's contact
        ok = [h for h in H if h["truth"] is not None]
        s["sim_check"] = dict(n=len(ok), cop_mm_median=round(float(np.median(
            [np.linalg.norm(h["p"] - h["truth"]["cop"]) for h in ok])), 3), force_N_median=round(float(np.median(
                [np.linalg.norm(h["f"] - h["truth"]["f"]) for h in ok])), 3))
    (out / "summary.json").write_text(json.dumps(s, indent=1) + "\n")
    print(json.dumps(s), "->", out)


if __name__ == "__main__":
    main()
