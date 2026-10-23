#!/usr/bin/env python3
"""Check of a calibration session on the UR5 (CB3) against Table 2 of the runbook
(docs/experiments/20261022-ur5_tip_calibration/README.md), stage by stage, from whatever the session folder has.

The small files (events.jsonl, registration.json, touches.json, pivots.json, session.json, plan.json) give: the stream
rates and noise of the preflight, the homing's spread and the indenter's placement, the gravity fit, each touch-off's
onset and its residual after the registration, the registration and the pivot check, the presses that ran to their
travel limit without reaching their force, the aborts and the time per stage. Two quantities come from them as well:
the HEX-E's zero over the session (each action's tare less the gravity fit's prediction at that orientation) and the
flange's position along the approach at the top level of each reference press (the seat's and the indenter's drift).
With ft.bin and robot.bin the check adds the HEX-E's noise in each rest, the force reached by the column pushes and,
in a dry run, the HEX-E's reading against the gravity fit along every travel (cables pulling as the tool turns).

With --clearance [PLAN] it adds the least clearance of the mount, the finger and the HEX-E to the indenter and the cell
at every press pose and every degree of every turn of the plan (the session's own plan by default), with the nominal
geometry that the builder checks and with the session's registration (fitted here from its touch-offs when the register
stage has not run). Once registered, the program puts the real pad on its targets and the mount stands where the real
geometry puts it: on 2026-10-22 sphere2's first 20 touch-offs moved the TCP 1.80 mm and 2.16 deg.

    .venv/bin/python scripts/ur5_tip_calibration_check.py <session folder> [--json out.json] [--clearance [PLAN]]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/morphohand/driver/reskin5x/host"))
import ur5_tip_calibration as C  # noqa: E402

PLAN = ROOT / "docs/experiments/20261022-ur5_tip_calibration/plan/20261022-plan_sphere.json"
NOISE_SHEET = np.array([0.035, 0.035, 0.15, 0.002, 0.002, 0.001])      # HEX-E data sheet, 1 sigma, N and N m
# 1 sigma per sample on the lab's HEX-E polled at 100 Hz, arm off, by the DAQ's filter code (lab/README.md, 2026-10-22)
NOISE_LAB = {0: [0.086, 0.059, 0.534, 0.0072, 0.0095, 0.0023], 2: [0.060, 0.050, 0.352, 0.0044, 0.0055, 0.0017],
             3: [0.043, 0.050, 0.213, 0.0028, 0.0038, 0.0010], 4: [0.034, 0.044, 0.131, 0.0016, 0.0021, 0.0006]}
EZ = np.array([0.0, 0.0, 1.0])


def read_events(sd):
    ev = []
    for line in open(sd / "events.jsonl"):
        try:
            ev.append(json.loads(line))
        except json.JSONDecodeError:                             # a line cut by a copy made mid-run
            pass
    return ev


def jload(p):
    return json.loads(p.read_text()) if p.exists() else None


def flag(ok):
    return "ok " if ok else "OFF"


class Check:
    def __init__(self, sd):
        self.sd = Path(sd)
        self.ev = read_events(self.sd)
        self.session = jload(self.sd / "session.json") or {}
        self.plan = jload(self.sd / "plan.json") or json.loads(PLAN.read_text())
        self.reg_d = jload(self.sd / "registration.json")
        self.touches = jload(self.sd / "touches.json") or []
        self.geo = C.Geometry(self.plan)
        self.reg = C.Registration(self.geo).load(self.reg_d) if self.reg_d else None
        self.notes = (self.reg_d or {}).get("notes", {})
        self.acts = {a["id"]: a for a in self.plan["actions"]}
        self.starts = {e["id"]: e for e in self.ev if e.get("event") == "start"}
        self.ends = [e for e in self.ev if e.get("event") == "end"]
        self.ok = {e["id"]: e for e in self.ends if e.get("status") == "ok"}
        self.dry = float((self.session.get("args") or {}).get("dry") or 0.0)
        self.ft = self.rb = None
        if (self.sd / "ft.bin").exists():
            self.ft = np.fromfile(self.sd / "ft.bin", "<f8")
            self.ft = self.ft[: len(self.ft) // 9 * 9].reshape(-1, 9)
        if (self.sd / "robot.bin").exists():
            self.rb = np.fromfile(self.sd / "robot.bin", "<f8")
            self.rb = self.rb[: len(self.rb) // 15 * 15].reshape(-1, 15)
        filt = (self.session.get("args") or {}).get("ft_filter")
        self.noise_ref = np.array(NOISE_LAB[filt]) if filt in NOISE_LAB else NOISE_SHEET
        self.noise_name = f"the lab's filter {filt}" if filt in NOISE_LAB else "the data sheet"
        self.out = {}
        self.lines = []

    def p(self, s=""):
        self.lines.append(s)

    # ------------------------------------------------------------------ the session as a whole
    def overview(self):
        a = self.session.get("args") or {}
        t = [e["t"] for e in self.ev if "t" in e]
        stages = {}
        for e in self.ends:
            st = (self.starts.get(e["id"]) or {}).get("stage") or (self.acts.get(e["id"]) or {}).get("stage") or e["id"]
            s = stages.setdefault(st, dict(ok=0, abort=0, t0=math.inf, t1=-math.inf))
            s["ok" if e.get("status") == "ok" else "abort"] += 1
            s["t0"] = min(s["t0"], (self.starts.get(e["id"]) or e)["t"])
            s["t1"] = max(s["t1"], e["t"])
        self.p(f"session {self.sd.name}: started {self.session.get('started')}, stages {a.get('stages')}, dry {self.dry:g} mm, "
               f"{(t[-1] - t[0]) / 60 if t else 0:.1f} min of events")
        for st, s in stages.items():
            self.p(f"  {st:10s} {s['ok']:3d} ok {s['abort']:2d} aborted  {(s['t1'] - s['t0']) / 60:5.1f} min")
        ab = [dict(id=e["id"], why=e.get("why")) for e in self.ends if e.get("status") != "ok"]
        for x in ab:
            self.p(f"  ABORT {x['id']}: {x['why']}")
        open_ = [i for i in self.starts if i not in {e['id'] for e in self.ends}]
        if open_:
            self.p(f"  started without an end (running or cut off): {open_}")
        reads = [e for e in self.ev if e.get("event") == "ft_reads"]
        if reads:
            r = reads[-1]
            self.p(f"  HEX-E reads: {r['n']} samples, {r['timeouts']} timed out, {r['late']} late, {r['repeats']} repeated, "
                   f"{len(r.get('gaps') or [])} gaps over 0.5 s [{flag(r['timeouts'] == 0 and r['late'] == 0 and not r.get('gaps'))}]")
        for g in [e for e in self.ev if e.get("event") == "ft_gap"]:
            self.p(f"  HEX-E gap in {g['id']}: {g['t1'] - g['t0']:.1f} s, counter {g['seq0']} -> {g['seq1']}, Fz changed by "
                   f"{g['dFz_N']:+.1f} N")
        self.out["stages"] = stages
        self.out["aborts"] = ab

    def preflight(self):
        e = self.ok.get("preflight")
        if not e:
            return
        r = e["rec"]
        self.p(f"preflight: HEX-E {r['ft_hz']:.0f} Hz, board {r['board_hz']:.0f} records/s, robot {r['robot_hz']:.0f} Hz "
               f"[{flag(r['ft_hz'] > 450 and r['board_hz'] > 120 and r['robot_hz'] > 120)}] (500, 140, 125)")
        if r.get("ft_std") is not None:
            s = np.array(r["ft_std"])
            ratio = s / self.noise_ref
            self.p(f"  HEX-E noise {np.round(s[:3], 4)} N, {np.round(s[3:], 5)} N m: {np.round(ratio, 2)} x {self.noise_name} "
                   f"[{flag(ratio.max() < 2.0)}]")
        if r.get("board_std_uT") is not None:
            b = np.array(r["board_std_uT"])
            self.p(f"  board noise per chip (x, y, z uT): median {np.round(np.median(b, axis=0), 2)}, max {b.max():.2f} "
                   f"(simulated 1.6, 1.6, 2.5)")
        if "ft_timeouts" in r:
            self.p(f"  HEX-E reads at preflight: {r['ft_timeouts']} timed out, {r['ft_late']} late")
        self.out["preflight"] = r

    def homing(self):
        sp = self.notes.get("homing_spread_mm_deg")
        if not sp:
            return
        sp = np.array(sp)
        self.p(f"homing: {len(sp)} seatings, spread {sp[:, 0].max():.3f} mm, {sp[:, 1].max():.3f} deg "
               f"[{flag(sp[:, 0].max() < 0.2 and sp[:, 1].max() < 0.2)}] (under 0.2 mm, 0.2 deg)")
        pl = self.notes.get("placement_offset_mm_deg")
        if pl and self.reg is not None and self.reg.T_BI is not None:
            T = self.reg.T_BI
            yaw = math.degrees(math.atan2(T[1, 0], T[0, 0]))
            dp = np.array(pl["d_mm"])
            self.p(f"  indenter at {np.round(T[:3, 3], 1)} mm, yaw {yaw:.1f} deg, tilt {pl['tilt']:.2f} deg; off the plan by "
                   f"{np.round(dp, 1)} mm ({np.linalg.norm(dp):.1f}), yaw {pl['d_yaw']:+.1f} deg "
                   f"[{flag(np.linalg.norm(dp) <= 20 and abs(pl['d_yaw']) <= 5 and pl['tilt'] <= 2)}]")
        self.out["homing"] = dict(spread=sp.tolist(), placement=pl)

    def gravity_records(self, stage):
        R, F, T, t, ids = [], [], [], [], []
        for a in self.plan["actions"]:
            if a["stage"] == stage and a["id"] in self.ok:
                r = self.ok[a["id"]]["rec"]
                R.append(r["R_BE"])
                F.append(r["ft"][:3])
                T.append(r["ft"][3:])
                t.append(float(np.mean(r["t"])))
                ids.append(a["id"])
        return np.array(R), np.array(F), np.array(T), np.array(t), ids

    @staticmethod
    def fit_with_drift(R, F, t):
        """The gravity fit with one more column, the record's time: f_S = A g_E + o + v (t - t_mean)."""
        gE = np.einsum("kji,j->ki", R, np.array([0.0, 0.0, -1.0]))
        X = np.c_[gE, np.ones(len(gE)), (t - t.mean()) / 60.0]
        sol, *_ = np.linalg.lstsq(X, F, rcond=None)
        A = sol[:3].T
        return dict(A=A, v_N_min=sol[4], weight_N=float(np.linalg.svd(A)[1].mean()),
                    rms_f_N=float(np.sqrt(np.mean((F - X @ sol) ** 2))))

    def gravity(self):
        for st in ("gravity_failed", "final_gravity_failed"):
            if st in self.notes:
                f = self.notes[st]
                self.p(f"{st.replace('_failed', '')}: FAILED the program's gate: weight {f['weight_N']:.3f} N, residual "
                       f"{f['rms_f_N']:.3f} N, {f['rms_t_Nm'] * 1e3:.2f} N mm")
        g = self.notes.get("gravity")
        if not g:
            return
        sv = np.array(g["sv"])
        self.p(f"gravity: {g['n']} orientations, weight {g['weight_N']:.3f} N ({g['mass_kg'] * 1e3:.0f} g) "
               f"[{flag(1.2 <= g['weight_N'] <= 2.0)}] (1.2 to 2.0 N), residual {g['rms_f_N']:.4f} N "
               f"[{flag(g['rms_f_N'] < 0.02)}], {g['rms_t_Nm'] * 1e3:.3f} N mm")
        self.p(f"  centre of mass {np.round(g['com_mm'], 1)} mm from the HEX-E's origin (simulated 0.3, 2.1, 16.9); sign "
               f"{g['sign']:+.0f}; HEX-E axes turned {g['z_turn_deg']:+.2f} deg about the tool axis, {g['turn_deg']:.2f} in all; "
               f"gains of its axes {np.round(sv / sv.mean(), 3)} of their mean; skew of K {g['skew_of_K']:.3f}")
        # the residual of each orientation, from the records, and the fit with a drift term
        R, F, T, t, ids = self.gravity_records("gravity")
        if len(R) >= 6:
            fit = C.fit_gravity(R, F, T)
            gE = np.einsum("kji,j->ki", R, np.array([0.0, 0.0, -1.0]))
            res = F - (gE @ fit["A"].T + fit["of"])
            worst = np.argsort(-np.linalg.norm(res, axis=1))[:3]
            self.p("  largest residuals: " + ", ".join(f"{ids[k]} {np.linalg.norm(res[k]) * 1e3:.1f} mN" for k in worst))
            fd = self.fit_with_drift(R, F, t)
            dA = np.linalg.norm(fd["A"] - fit["A"]) / np.linalg.norm(fit["A"])
            self.p(f"  with a drift term over the {(t[-1] - t[0]) / 60:.1f} min sweep: drift {np.round(fd['v_N_min'] * 1e3, 1)} mN/min, "
                   f"residual {fd['rms_f_N'] * 1e3:.1f} mN, weight {fd['weight_N']:.3f} N; the map A moves {dA * 100:.2f} %")
            self.out["gravity_drift"] = dict(v_mN_min=(fd["v_N_min"] * 1e3).tolist(), rms_mN=fd["rms_f_N"] * 1e3, dA=dA)
        fg = self.notes.get("final_gravity")
        if fg:
            dw = fg["weight_N"] - g["weight_N"]
            self.p(f"final gravity: weight {fg['weight_N']:.3f} N, {dw * 1e3:+.1f} mN from the first [{flag(abs(dw) < 0.01)}] "
                   f"(within 10 mN); centre of mass moved {np.linalg.norm(np.subtract(fg['com_mm'], g['com_mm'])):.2f} mm; "
                   f"axes turn {fg['z_turn_deg'] - g['z_turn_deg']:+.2f} deg")
        Rf, Ff, Tf, tf, idf = self.gravity_records("final_gravity")
        grav = (self.reg_d or {}).get("grav")
        if len(Rf) >= 2 and grav:
            # the changes between the final sweep's orientations against the first fit's map (offsets cancel)
            A = np.array(grav["A"])
            gE = np.einsum("kji,j->ki", Rf, np.array([0.0, 0.0, -1.0]))
            dF = Ff[1:] - Ff[0]
            dP = (gE[1:] - gE[0]) @ A.T
            s = float((dF.ravel() @ dP.ravel()) / (dP.ravel() @ dP.ravel()))
            rms = float(np.sqrt(np.mean((dF - dP) ** 2)))
            self.p(f"final sweep against the first fit: {len(Rf)} orientations; their changes are {s:.4f} x the predicted ones "
                   f"(1 = the same map), rms difference {rms * 1e3:.1f} mN over {(tf[-1] - tf[0]) / 60:.1f} min")
            self.out["final_vs_first"] = dict(scale=s, rms_mN=rms * 1e3)
        self.out["gravity"] = g

    # ------------------------------------------------------------------ touch-offs and the registration
    def approach(self):
        T = self.reg.T_BI if self.reg is not None and self.reg.T_BI is not None else None
        return None if T is None else -(T[:3, :3] @ EZ)

    def touch_residuals(self):
        n1 = self.notes.get("register")
        if not n1 or "T_EM" not in n1:
            return {}
        T_EM, T_MP, c = np.array(n1["T_EM"]), np.array(n1["T_MP"]), np.array(n1["c_B"])
        T_EP = T_EM @ T_MP
        out = {}
        for t in self.touches:
            T_BE = np.array(t["T_BE"])
            if t["kind"] == "pad":
                q = C.apply(C.inv(T_BE @ T_EP), c)
                out[t["id"]] = self.geo.surface_distance(q)[0] - self.geo.R
            else:
                q = C.apply(C.inv(T_BE @ T_EM), c)
                out[t["id"]] = float(q @ self.geo.face_n - self.geo.face_d - self.geo.R)
        return out

    def touchoffs(self):
        if self.dry:
            n = sum(1 for e in self.ends if (self.acts.get(e["id"]) or {}).get("stage") in ("mount", "pad") and e.get("status") == "ok")
            if n:
                self.p(f"touch-offs: {n} approaches in the dry run (no onsets by design)")
            return
        if not self.touches:
            return
        res = self.touch_residuals()
        rows = []
        for t in self.touches:
            a = self.acts.get(t["id"], {})
            rows.append(dict(id=t["id"], kind=t["kind"], region=a.get("target", {}).get("region"), d=t["s0"] - t["expected_mm"],
                             p=t["p"], rms_mN=t["rms_N"] * 1e3, n=t["n_contact"], resid=res.get(t["id"])))
        d = np.array([r["d"] for r in rows])
        self.p(f"touch-offs: {len(rows)} onsets, from the expected {d.min():+.3f} to {d.max():+.3f} mm, median {np.median(d):+.3f} "
               f"[{flag(np.abs(d).max() < 1.0)}] (within 1 mm; over 2 mm stop)")
        for kind in ("mount", "pad"):
            rr = [r for r in rows if r["kind"] == kind]
            if not rr:
                continue
            dd = np.array([r["d"] for r in rr])
            regions = sorted({r["region"] for r in rr if r["region"]})
            by = "; ".join(f"{g} {np.median([r['d'] for r in rr if r['region'] == g]):+.3f}" for g in regions)
            self.p(f"  {kind}: {len(rr)}, median {np.median(dd):+.3f} mm, spread {dd.std():.3f} mm" + (f" (by region: {by})" if by else "")
                   + f"; fit rms median {np.median([r['rms_mN'] for r in rr]):.1f} mN, power median {np.median([r['p'] for r in rr]):.2f}")
        if res:
            rs = sorted(rows, key=lambda r: -abs(r["resid"] or 0))[:5]
            self.p("  largest residuals after the registration: " + ", ".join(f"{r['id']} {r['resid'] * 1e3:+.0f} um" for r in rs))
        # the column pushes to 5 N: travel past the onset (the HEX-E's and the mount's compliance)
        dB = self.approach()
        for t in self.touches:
            a = self.acts.get(t["id"], {})
            if t["kind"] != "mount" or float(a.get("stop_N", 0.5)) < 2.0 or t["id"] not in self.ok or dB is None:
                continue
            rec = self.ok[t["id"]]["rec"]
            tr = float((np.array(rec["T_stop"])[:3, 3] - np.array(t["T_BE"])[:3, 3]) @ dB)
            fmax = self.peak_push(rec, dB)
            self.p(f"  {t['id']} pushed to {a.get('stop_N')} N: {tr:.3f} mm past the onset" +
                   (f", peak push {fmax:.2f} N" if fmax is not None else "") + " (data sheet 0.12 and 0.16 mm; simulated 0.16)")
        self.out["touches"] = rows

    def peak_push(self, rec, dB):
        if self.ft is None or self.reg is None:
            return None
        m = (self.ft[:, 0] >= rec["t_move0"]) & (self.ft[:, 0] <= rec["t_stop"] + 0.3)
        if not m.any():
            return None
        R_BE = np.array(rec["T_stop"])[:3, :3]
        tare = np.array(rec["tare"])[:3]
        f = [float(-(R_BE @ np.linalg.solve(self.reg.K, v[3:6] - tare)) @ dB) for v in self.ft[m]]
        return float(np.max(f))

    def register(self):
        for key in ("register", "register_pivot"):
            n = self.notes.get(key)
            if not n:
                continue
            tc = n["tcp_change_mm_deg"]
            sd = np.array(n.get("sd", [np.nan] * 15))
            self.p(f"{key}: pad touches {n['rms_pad_mm'] * 1e3:.1f} um rms ({n['n_pad']}), face {n['rms_face_mm'] * 1e3:.1f} um "
                   f"({n['n_face']}) [{flag(n['rms_pad_mm'] < 0.05)}] (under 50 um; over 100 um a touch slipped); TCP moved "
                   f"{tc[0]:.3f} mm, {tc[1]:.3f} deg (simulated 0.99 mm, 1.11 deg); sphere centre moved {n['centre_change_mm']:.3f} mm")
            self.p(f"  1-sigma of the fit: mount on flange {np.round(np.degrees(sd[0:3]), 3)} deg, {np.round(sd[3:6], 3)} mm; pad on "
                   f"mount {np.round(np.degrees(sd[6:9]), 3)} deg, {np.round(sd[9:12], 3)} mm; centre {np.round(sd[12:15], 3)} mm")
            if "T_EM" in n:
                T_EP = np.array(n["T_EM"]) @ np.array(n["T_MP"])
                self.p("  TCP of the pad frame (m, rotation vector): " + ", ".join(f"{v:.5f}" for v in C.T_to_ur(T_EP)))
            self.out[key] = {k: n[k] for k in ("rms_pad_mm", "rms_face_mm", "n_pad", "n_face", "tcp_change_mm_deg", "centre_change_mm")}
        pc = self.notes.get("pivot_check")
        if pc:
            self.p(f"pivot: board and registration apart by at most {pc['worst_mm']:.3f} mm [{flag(pc['worst_mm'] < 0.5)}] "
                   f"(under 0.5 mm; simulated 0.21)")
            for pv in pc["pivots"]:
                db, dr = np.array(pv["dp_board"]), np.array(pv.get("dp_registration", [np.nan, np.nan]))
                self.p(f"  {pv['id']}: board {np.round(db, 3)} mm, registration {np.round(dr, 3)} mm, shear {pv['shear_N']:.3f} N, "
                       f"fit residual {pv['resid_uT']:.1f} uT")
            self.out["pivot"] = dict(worst_mm=pc["worst_mm"])

    # ------------------------------------------------------------------ presses
    def presses(self):
        rows = []
        for e in self.ends:
            a = self.acts.get(e["id"])
            if not a or a["kind"] not in ("press", "slide", "roll", "twist") or e.get("status") != "ok":
                continue
            r = e["rec"]
            top = r["levels"][-1]["N"] if r.get("levels") else None
            want = max(a.get("levels") or [a.get("press_N") or a.get("stop_N") or 0])
            rows.append(dict(id=e["id"], stage=a["stage"], role=a.get("role"), kind=a["kind"], top=top, want=want,
                             short=bool(r.get("reached_max")), t=e["t"], rec=r))
        if not rows:
            return
        by = {}
        for r in rows:
            by.setdefault(r["stage"], []).append(r)
        for st, rr in by.items():
            if self.dry:
                self.p(f"{st}: {len(rr)} approaches ok (no contact by design)")
                continue
            short = [r for r in rr if r["short"] or (r["top"] is not None and r["top"] < r["want"] - 1e-6)]
            self.p(f"{st}: {len(rr)} actions ok; {len(short)} ended at the travel limit below their force"
                   + (": " + ", ".join(f"{r['id']} ({r['top'] or 0:g} of {r['want']:g} N)" for r in short[:8]) if short else ""))
        # the reference presses: position along the approach at the top level, over the session
        dB = self.approach()
        ref = [r for r in rows if r["role"] == "ref" and r["kind"] == "press" and r["stage"] in ("grid", "final")]
        if ref and dB is not None and not self.dry:
            s = np.array([np.array(r["rec"]["T_stop"])[:3, 3] @ dB for r in ref])
            tm = np.array([(r["t"] - ref[0]["t"]) / 60 for r in ref])
            self.p(f"reference presses: {len(ref)}; flange position at {ref[0]['top']:g} N along the approach moved "
                   f"{(s - s[0]).min() * 1e3:+.0f} to {(s - s[0]).max() * 1e3:+.0f} um from the first over {tm[-1]:.0f} min")
            self.out["reference_um"] = list(zip(tm.round(1).tolist(), ((s - s[0]) * 1e3).round(1).tolist()))
        self.out["presses"] = {st: len(rr) for st, rr in by.items()}

    # ------------------------------------------------------------------ the HEX-E's zero and noise
    def zero_drift(self):
        g = (self.reg_d or {}).get("grav")
        if not g or self.reg is None:
            return
        A, of = np.array(g["A"]), np.array(g["of"])
        Bt, ot = np.array(g["Bt"]), np.array(g["ot"])
        pts = []
        for e in self.ends:
            r = e.get("rec") or {}
            if e.get("status") != "ok" or "tare" not in r or r.get("T_stop") is None:
                continue
            R = np.array(r["T_stop"])[:3, :3]
            gE = R.T @ np.array([0.0, 0.0, -1.0])
            tare = np.array(r["tare"])
            pts.append((e["t"], tare[:3] - (A @ gE + of), (tare[3:] - (Bt @ gE + ot)) * 1e3))
        if len(pts) < 3:
            return
        t0 = pts[0][0]
        f = np.array([p[1] for p in pts])
        tq = np.array([p[2] for p in pts])
        tm = np.array([(p[0] - t0) / 60 for p in pts])
        X = np.c_[tm, np.ones_like(tm)]
        cf, *_ = np.linalg.lstsq(X, f, rcond=None)
        ct, *_ = np.linalg.lstsq(X, tq, rcond=None)
        self.p(f"HEX-E zero (each rest's tare less the gravity fit) over {tm[-1]:.0f} min, {len(pts)} rests: trend "
               f"{np.round(cf[0] * 1e3, 1)} mN/min, rms about it {np.round((f - X @ cf).std(axis=0) * 1e3, 1)} mN; torque trend "
               f"{np.round(ct[0], 2)} N mm/min, rms {np.round((tq - X @ ct).std(axis=0), 2)} N mm; last rest "
               f"{np.round(f[-1] * 1e3, 0)} mN")
        self.out["zero_drift"] = dict(t_min=tm.round(2).tolist(), f_mN=(f * 1e3).round(1).tolist(), t_Nmm=tq.round(2).tolist(),
                                      slope_mN_min=(cf[0] * 1e3).tolist())

    def rest_noise(self):
        if self.ft is None:
            return
        s = []
        for e in self.ends:
            r = e.get("rec") or {}
            if e.get("status") == "ok" and r.get("t_rest0"):
                m = (self.ft[:, 0] >= r["t_rest0"][0]) & (self.ft[:, 0] <= r["t_rest0"][1])
                if m.sum() > 20:                                  # a 0.5 s rest is 50 samples at 100 Hz
                    s.append(self.ft[m, 3:9].std(axis=0))
        if s:
            s = np.array(s)
            self.p(f"HEX-E noise in {len(s)} rests (median 1 sigma): {np.round(np.median(s[:, :3], axis=0), 4)} N, "
                   f"{np.round(np.median(s[:, 3:], axis=0), 5)} N m; {np.round(np.median(s, axis=0) / self.noise_ref, 2)} x "
                   f"{self.noise_name}")
        dt = np.diff(self.ft[:, 0])
        self.p(f"HEX-E stream: {len(self.ft)} samples over {(self.ft[-1, 0] - self.ft[0, 0]) / 60:.1f} min, median interval "
               f"{np.median(dt) * 1e3:.2f} ms, gaps over 20 ms {int((dt > 0.02).sum())}, longest {dt.max() * 1e3:.0f} ms")
        st, n = np.unique(self.ft[:, 2].astype(np.int64), return_counts=True)
        self.p("  status words: " + ", ".join(f"{hex(s_)} in {c / len(self.ft):.1%}" for s_, c in zip(st, n))
               + " (0x40000: the box's overload warning)")

    def travel_loads(self):
        """In a dry run: the HEX-E's reading less the gravity fit at the robot's orientation, along the whole run."""
        g = (self.reg_d or {}).get("grav")
        if self.ft is None or self.rb is None or not g or not self.dry:
            return
        A, of = np.array(g["A"]), np.array(g["of"])
        edges = np.arange(self.ft[0, 0], self.ft[-1, 0], 0.5)             # 0.5 s windows: the noise averages out
        k = np.searchsorted(self.ft[:, 0], edges)
        res, when = [], []
        for a_, b_ in zip(k[:-1], k[1:]):
            if b_ - a_ < 10:
                continue
            tc = 0.5 * (self.ft[a_, 0] + self.ft[b_ - 1, 0])
            i = int(np.clip(np.searchsorted(self.rb[:, 0], tc), 0, len(self.rb) - 1))
            gE = C.ur_to_T(self.rb[i, 8:14])[:3, :3].T @ np.array([0.0, 0.0, -1.0])
            res.append(self.ft[a_:b_, 3:6].mean(axis=0) - (A @ gE + of))
            when.append(tc)
        res = np.array(res)
        nrm = np.linalg.norm(res - np.median(res, axis=0), axis=1)
        worst = int(np.argmax(nrm))
        self.p(f"dry run: HEX-E force less the gravity fit, 0.5 s means along the run: median {np.median(nrm) * 1e3:.0f} mN, "
               f"99th percentile {np.percentile(nrm, 99) * 1e3:.0f} mN, max {nrm[worst] * 1e3:.0f} mN at "
               f"{(when[worst] - self.ft[0, 0]) / 60:.1f} min (a cable pulling shows as a load that follows the tool's turn)")
        self.out["travel_load_mN"] = dict(median=float(np.median(nrm) * 1e3), p99=float(np.percentile(nrm, 99) * 1e3))

    def registered_clearance(self, plan_path=None, stages=("pivot", "grid", "shear", "roll", "twist", "final"), step=1.0):
        """Least clearance of each tool group to the indenter, the cell and the other indenters over the plan's press
        poses (rest and 1 mm past contact) and its turns (every step deg), nominal against registered."""
        sys.path.insert(0, str(ROOT / "scripts"))
        import ur5_cb3_cell as CELL
        plan = json.loads(Path(plan_path).read_text()) if plan_path else self.plan
        ck = plan.get("checks", {})
        CELL.BOARD.update(ck.get("board") or {})
        CELL.OTHER[:] = [(np.array(o["T_BI"], float), o["kind"]) for o in ck.get("other", [])]
        need = dict(dict(mount=3.0, finger=0.8, hex_e=6.0), **{k: v for k, v in (ck.get("clearances_needed_mm") or {}).items()
                                                                if k in ("mount", "finger", "hex_e")})
        geo = C.Geometry(plan)
        reg = C.Registration(geo).load(self.reg_d)
        if reg.T_BE_home is None:
            self.p("clearance: no homing in this session")
            return
        T_BI_home = reg.T_BE_home @ geo.T_EM0 @ C.inv(geo.T_IM_home)
        if "register" in self.notes:
            src, (T_EM, T_MP, T_BI) = "the session's registration", (reg.T_EM, reg.T_MP, reg.T_BI)
        else:
            pads = [t for t in self.touches if t.get("kind") == "pad"]
            if len(pads) < 6:
                self.p(f"clearance: {len(pads)} pad touch-offs, too few for a registration")
                return
            r0 = C.Registration(geo).load(self.reg_d)
            r0.T_EM, r0.T_MP, r0.T_BI = geo.T_EM0.copy(), geo.T_MP0.copy(), T_BI_home
            f = C.fit_registration(geo, r0, pads)
            T_BI = C.Tm(T_BI_home[:3, :3], f["c_B"] - T_BI_home[:3, :3] @ geo.c_I)
            src, (T_EM, T_MP) = f"a fit to its {len(pads)} pad touch-offs", (f["T_EM"], f["T_MP"])
        G = dict(nominal=(T_BI_home, geo.T_EM0, geo.T_MP0), registered=(T_BI, T_EM, T_MP))
        TP = CELL.tool_points()
        groups = dict(mount=("adapter", "fork", "clamp_plate"), finger=("dip_link", "cover", "cap"), hex_e=("hex_e",))
        least = {lab: {g: (1e9, None) for g in groups} for lab in G}
        for a in plan["actions"]:
            if a["stage"] not in stages or a["kind"] not in ("press", "slide", "roll", "twist"):
                continue
            tg, st = a["target"], float(a.get("standoff", 1.5))
            for lab, (T_BI_, T_EM_, T_MP_) in G.items():
                T_EP = T_EM_ @ T_MP_
                T_IP = geo.T_IP(tg["p"], tg["n"], a.get("twist", 0.0), st, a.get("offset", (0.0, 0.0)))
                poses = [T_BI_ @ C.Tm(T_IP[:3, :3], T_IP[:3, 3] - EZ * dz) @ C.inv(T_EP) for dz in (0.0, st + 1.0)]
                if a["kind"] != "press":                        # the turns of run_turns, from the deep pose
                    c_B = C.apply(T_BI_, geo.c_I)
                    top_B = C.apply(T_BI_, geo.c_I + geo.R * EZ)
                    for mv in a["moves"]:
                        ang = math.radians(mv.get("dir_deg", 0.0))
                        u = T_BI_[:3, :3] @ np.cross(EZ, [math.cos(ang), math.sin(ang), 0.0])
                        centre, axis = ((c_B, u) if a["kind"] == "slide" else (top_B, u) if a["kind"] == "roll"
                                        else (top_B, T_BI_[:3, :3] @ EZ))
                        cur = 0.0
                        for dg in list(mv["degs"]) + [0.0]:
                            n = max(1, int(math.ceil(abs(dg - cur) / step)))
                            poses += [C.turn_about(poses[1], centre, axis, cur + (dg - cur) * j / n) for j in range(1, n + 1)]
                            cur = dg
                for T_BE in poses:
                    T_IM = C.inv(T_BI_) @ T_BE @ T_EM_
                    for g, names in groups.items():
                        for nm in names:
                            PI = C.apply(T_IM, TP[nm])
                            v = min(float(CELL.indenter_sdf(PI, plan["indenter"]).min()), float(CELL.cell_sdf(PI, T_BI_).min()),
                                    float(CELL.other_sdf(C.apply(T_BE @ T_EM_, TP[nm])).min()))
                            if v < least[lab][g][0]:
                                least[lab][g] = (v, a["id"])
        self.p(f"clearance over {', '.join(stages)} with {src} (the builder checks the nominal geometry):")
        for g in groups:
            (vn, an), (vr, ar) = least["nominal"][g], least["registered"][g]
            self.p(f"  {g:7s} least {vr:6.2f} mm registered ({ar}), {vn:6.2f} nominal ({an}); needs {need[g]:.1f} "
                   f"[{flag(vr >= need[g])}]")
        self.out["clearance_registered"] = {g: least["registered"][g] for g in groups}

    def run(self):
        self.overview()
        self.preflight()
        self.homing()
        self.gravity()
        self.touchoffs()
        self.register()
        self.presses()
        self.zero_drift()
        self.rest_noise()
        self.travel_loads()
        return "\n".join(self.lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("session")
    ap.add_argument("--json", default=None)
    ap.add_argument("--clearance", nargs="?", const="", default=None, help="the plan's clearances with the session's "
                    "registration (or a fit to its touch-offs); PLAN defaults to the session's own")
    a = ap.parse_args()
    ch = Check(a.session)
    print(ch.run())
    if a.clearance is not None:
        ch.lines = []
        ch.registered_clearance(a.clearance or None)
        print("\n".join(ch.lines))
    if a.json:
        Path(a.json).write_text(json.dumps(ch.out, indent=1, default=C.jsonable))


if __name__ == "__main__":
    main()
