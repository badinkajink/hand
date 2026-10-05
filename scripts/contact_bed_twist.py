#!/usr/bin/env python3
"""Contact-model comparison bed, task 2 (T2 twist): torque about the pinch axis until the pinched tool spins.

Rig and models as in docs/experiments/20261005-contact_bed/PROTOCOL.md (two real_v1 fingertip spheres on
force-controlled rails squeeze the real_v1 screwdriver, mu 1, gravity off). Per case (model, pinch N per
pad, physics step dt):

  ramp     settle 0.4 s at N, then a torque about world x (the pinch axis) on the tool ramped at
           tau_law(N) per second, tau_law(N) = 2 mu N c N^(1/4), c = 0.996e-3 m N^-1/4 (the elastic-
           foundation law fitted to Drake hydroelastic in the 10-01 study), until the spin passes
           30 deg/s and 80 ms beyond. Onset by the creep-law method of contact_bed_pull.py on the angular
           speed: below onset the tool creeps at w = k tau (k fitted between 30 and 80 % of the detection
           torque); onset is the last step on that law, every later step up to detection having
           w > 2 k tau + 0.6 deg/s.
  hold     fresh rig, settle, ramp at the same rate to 50 % of the onset torque, hold 1 s; creep is the
           slope of the rotation over the last 0.8 s.
  kinetic  hom_contact_rig.exp_torsion: the tool driven at 1 rad/s about the pinch axis by a stiff PD
           torque; the mean torque over 0.4-1.2 s is the two pads' sliding friction torque.

A case has a static phase (`stuck_before_onset`) when the spin speed at half the onset torque is under a tenth of
the frictionless spin-up speed at that time; without one (point contact, condim 3) the onset torque only measures
the detection lag. rbar = tau / (2 mu N) is the pressure-weighted friction arm per pad. Across N the ratio rbar(3 N) / rbar(0.5 N)
is 6^(1/4) = 1.565 for a Winkler (hydroelastic) law and 6^(1/3) = 1.817 for Hertz; it is written into each
N = 3 row once the N = 0.5 row of the same model and step exists. The N = 1 N, 1 ms case of each model is
filmed (ramp at 0.4x, then the 1 rad/s spin at 1x) to media/twist_<model>.mp4; `--tile` assembles
media/twist_models.mp4 and its JPEG poster.

    PY=logs/20261001-hom_contact/venv/bin/python
    $PY scripts/contact_bed_twist.py --models mj_pads1 --N 1 --dt 1         # one case
    $PY scripts/contact_bed_twist.py                                        # the protocol grid, then the tile
"""
from __future__ import annotations

import argparse
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import contact_bed_common as B  # noqa: E402

H = B.H
C_LAW = 0.996e-3                 # m N^-1/4
W_DETECT = math.radians(30.0)    # rad/s, gross-spin detection
W_FLOOR = math.radians(0.6)      # rad/s, onset margin above the creep law (same fraction of detection as T1)
T_SETTLE = 0.4
T_POST = 0.080
T_HOLD = 1.0
OMEGA_KIN = 1.0
EJECT_M = 0.005                  # tool centre displaced this far from its settled position: ejected
OUT = B.BED / "twist_slip.jsonl"
FILM_N, FILM_DT = 1.0, 1.0
FILM_VIEW = ((0.0, 0.0, 0.0), 0.13, 18.0, -14.0)


def tau_law(N):
    return 2 * H.MU * N * C_LAW * N ** 0.25


def settle(rig, N):
    rig.set_pad_force(N)
    rig.set_tool_wrench(np.zeros(3), np.zeros(3))
    rig.step(T_SETTLE)
    st = rig.tool_state()
    c = rig.contacts()
    return st, {"N_L": c["L"]["N"], "N_R": c["R"]["N"], "n_L": c["L"]["n"], "n_R": c["R"]["n"]}


def drive(rig, dt, st0, tau_of_t, stop, t_max, on_step=None):
    """Apply tau(t) about world x each step. Returns t, tau, theta (rad from the settled pose), w, step wall
    times and the largest displacement of the tool centre."""
    th0, p0 = st0["theta"], st0["pos"].copy()
    T, TAU, TH, W, WALL = [], [], [], [], []
    th_prev, dp_max = 0.0, 0.0
    n = int(round(t_max / dt))
    for k in range(n):
        tt = (k + 1) * dt
        tau = tau_of_t(tt)
        rig.set_tool_wrench(np.zeros(3), np.array([tau, 0.0, 0.0]))
        w0 = time.perf_counter()
        rig.step(dt)
        WALL.append(time.perf_counter() - w0)
        st = rig.tool_state()
        th = st["theta"] - th0
        w = (th - th_prev) / dt
        th_prev = th
        dp_max = max(dp_max, float(np.linalg.norm(st["pos"] - p0)))
        T.append(tt), TAU.append(tau), TH.append(th), W.append(w)
        if on_step is not None:
            on_step(rig, k, tt, tau, th)
        if not math.isfinite(th) or dp_max > EJECT_M or stop(tt, th, w):
            break
    return np.array(T), np.array(TAU), np.array(TH), np.array(W), np.array(WALL), dp_max


def smooth_at(T, V, t, half=0.010):
    m = np.abs(T - t) <= half
    return float(V[m].mean()) if m.any() else float("nan")


class TwistFilm:
    """Collects the filmed case's frames: the ramp every 10 ms (0.4x) and the driven spin every 40 ms (1x)."""

    def __init__(self, model, spec, dt):
        self.r = B.BedFilm(spec, 0.0, 480, 360, model=model, view=FILM_VIEW)
        self.dt = dt
        self.ramp, self.spin = [], []
        self.ramp_stride = max(1, int(round(0.010 / dt)))
        self.spin_stride = max(1, int(round(0.040 / dt)))
        self.spin_k0 = None

    def on_ramp(self, rig, k, tt, tau, th):
        if (k + 1) % self.ramp_stride == 0:
            self.ramp.append(self.r.frame(rig, "", speed=round(self.ramp_stride * self.dt * 25, 3),
                                          bottom=f"ramp   t {tt:5.3f} s   τ {tau * 1e3:5.3f} mN m   "
                                                 f"θ {math.degrees(th):+7.3f} deg"))

    def on_spin(self, rig, k):
        st = rig.tool_state()
        if self.spin_k0 is None:
            self.spin_k0 = st["theta"]
        if (k - 1) % self.spin_stride:
            return
        tau = float(rig.d.xfrc_applied[rig.tool, 3]) if rig.sim == "mujoco" else float(rig.tau_tool[0])
        self.spin.append(self.r.frame(rig, "", speed=round(self.spin_stride * self.dt * 25, 3),
                                      bottom=f"spin 1 rad/s  t {k * self.dt:4.2f} s  τ {tau * 1e3:5.3f} mN m  "
                                             f"θ {math.degrees(st['theta'] - self.spin_k0):+5.1f} deg"))

    def close(self):
        self.r.close()


def run_case(model, N, dt_ms, film=False):
    spec, chain = B.MODELS[model]
    dt = dt_ms * 1e-3
    t_wall = time.time()
    R = tau_law(N)
    row = {"task": "twist", "model": model, "rig_spec": spec, "chain_spec": chain, "N": N, "dt_ms": dt_ms,
           "rate_Nm_s": R, "tau_law_Nm": R, "c_law_m": C_LAW, "rbar_law_mm": C_LAW * N ** 0.25 * 1e3,
           "rbar_winkler_cyl_mm": H.winkler_law(N, 1e7)[1] * 1e3, "mu": H.MU, "I_tool_kgm2": H.I_TOOL_T,
           "gravity": False, "w_detect_deg_s": math.degrees(W_DETECT), "w_floor_deg_s": math.degrees(W_FLOOR)}
    fm = TwistFilm(model, spec, dt) if film else None
    status = "complete"

    # ---- ramp to gross spin
    rig = B.new_rig(spec, dt)
    st0, c0 = settle(rig, N)
    row["settle"] = c0
    row["info"] = getattr(rig, "info", {})
    if c0["n_L"] == 0 or c0["n_R"] == 0:
        status = "ejected"
    det = {}

    def stop(tt, th, w):
        if "t" not in det and w > W_DETECT:
            det["t"] = tt
        return "t" in det and tt > det["t"] + T_POST
    T, TAU, TH, W, WALL, dp = drive(rig, dt, st0, lambda tt: R * tt, stop, 3.0, fm.on_ramp if fm else None)
    row["dp_max_ramp_mm"] = dp * 1e3
    if dp > EJECT_M:
        status = "ejected"
    if "t" not in det:
        row.update(spun=False, tau_max_Nm=float(TAU[-1]), rot_end_deg=math.degrees(TH[-1]))
    else:
        td = det["t"]
        i_d = int(np.searchsorted(T, td - 1e-9))
        mc = (TAU >= 0.3 * TAU[i_d]) & (TAU <= 0.8 * TAU[i_d])
        k = float((W[mc] * TAU[mc]).sum() / max((TAU[mc] ** 2).sum(), 1e-30)) if mc.any() else 0.0
        i = i_d
        while i > 0 and W[i - 1] > 2 * max(k, 0.0) * TAU[i - 1] + W_FLOOR:
            i -= 1
        j = max(i - 1, 0)                         # last step on the creep law
        t0, tau_on, th_pre = float(T[j]), float(TAU[j]), float(TH[j])
        th50 = float(np.interp(t0 + 0.050, T, TH))
        mw = (T >= t0 + 0.005) & (T <= t0 + 0.060)
        if mw.sum() >= 4:
            alpha = 2 * np.polyfit(T[mw], TH[mw], 2)[0]
            tau_slide = float(TAU[mw].mean()) - H.I_TOOL_T * alpha
        else:
            tau_slide = float("nan")
        ww = W[(T > t0) & (T <= t0 + 0.050)]
        w_half = smooth_at(T, W, 0.5 * t0)
        w_free = R * (0.5 * t0) ** 2 / (2 * H.I_TOOL_T)    # spin-up speed with no friction torque at all
        row.update(spun=True, t_detect=td, tau_detect_Nm=R * td,
                   creep_k_deg_s_per_mNm=math.degrees(k) * 1e-3, t_onset=t0, tau_onset_Nm=tau_on,
                   rbar_onset_mm=tau_on / (2 * H.MU * N) * 1e3, tau_onset_over_law=tau_on / R,
                   rot_pre_deg=math.degrees(th_pre),
                   w_at_half_onset_deg_s=math.degrees(w_half),
                   w_at_090_onset_deg_s=math.degrees(smooth_at(T, W, 0.9 * t0)),
                   w_half_over_free=abs(w_half) / max(w_free, 1e-12),
                   stuck_before_onset=bool(abs(w_half) < 0.1 * w_free),
                   w_slip_mean50_deg_s=math.degrees((th50 - th_pre) / 0.050),
                   w_slip_mean50_coulomb_deg_s=math.degrees(R * 0.050 ** 2 / (6 * H.I_TOOL_T)),
                   w_slip_cv50=float(ww.std() / max(abs(ww.mean()), 1e-12)) if len(ww) > 2 else float("nan"),
                   rbar_slide_mm=tau_slide / (2 * H.MU * N) * 1e3)
    stride = max(1, int(round(0.005 / dt)))
    row["ramp_cols"] = ["t", "tau_mNm", "theta_deg", "w_deg_s"]
    row["ramp"] = [[round(float(a), 4), float(b * 1e3), float(math.degrees(c)), float(math.degrees(d))]
                   for a, b, c, d in zip(T[::stride], TAU[::stride], TH[::stride], W[::stride])]
    del rig

    # ---- hold at 50 % of onset
    if row.get("spun"):
        tau_h = 0.5 * row["tau_onset_Nm"]
        rig = B.new_rig(spec, dt)
        st0, _ = settle(rig, N)
        t_r = tau_h / R
        T, TAU, TH, W2, WALL2, dp = drive(rig, dt, st0, lambda tt: min(R * tt, tau_h), lambda *a: False,
                                          t_r + T_HOLD)
        mh = T >= t_r + 0.2
        slope = np.polyfit(T[mh], TH[mh], 1)[0] if mh.sum() > 3 else float("nan")
        th_hs = float(np.interp(t_r, T, TH))
        row.update(tau_hold_Nm=tau_h, creep_deg_s=math.degrees(slope), hold_rot_deg=math.degrees(TH[-1] - th_hs),
                   rot_at_hold_start_deg=math.degrees(th_hs), dp_max_hold_mm=dp * 1e3)
        if dp > EJECT_M:
            status = "ejected"
        stride = max(1, int(round(0.010 / dt)))
        row["hold_cols"] = ["t", "tau_mNm", "theta_deg"]
        row["hold"] = [[round(float(a), 4), float(b * 1e3), float(math.degrees(c))]
                       for a, b, c in zip(T[::stride], TAU[::stride], TH[::stride])]
        WALL = np.concatenate([WALL, WALL2])
        del rig
    row["us_per_step_median"] = float(np.median(WALL) * 1e6)
    row["us_per_step_mean"] = float(np.mean(WALL) * 1e6)

    # ---- kinetic torque at a driven 1 rad/s spin (hom_contact_rig.exp_torsion)
    H.DT = dt
    with B.StepProbe(fm.on_spin if fm else None) as P:
        kin = H.exp_torsion(spec, N, omega=OMEGA_KIN)
    row.update(omega_kin_rad_s=OMEGA_KIN, tau_kin_Nm=kin["tau_fric"], tau_kin_sd_Nm=kin["tau_sd"],
               rbar_kin_mm=kin["rbar_per_pad_mm"], kin_N_L=kin["N_L"], kin_N_R=kin["N_R"], kin_n_L=kin["n_L"],
               kin_n_R=kin["n_R"], us_per_step_median_kin=float(np.median(P.W) * 1e6) if P.W else None)
    if row.get("spun"):
        row["rbar_onset_over_kin"] = row["rbar_onset_mm"] / row["rbar_kin_mm"] if abs(row["rbar_kin_mm"]) > 1e-9 else None

    # ---- film
    row["film"] = None
    if fm is not None:
        rel = f"media/twist_{model}.mp4"
        B.write_h264(fm.ramp + fm.spin, B.BED / rel)
        row["film"] = rel
        row["film_segments"] = [len(fm.ramp), len(fm.spin)]
        fm.close()
    row["status"] = status
    row["script"] = "scripts/contact_bed_twist.py"
    row["git_rev"] = B.git_rev()
    row["wall_s"] = time.time() - t_wall
    row["when"] = time.strftime("%Y-%m-%d %H:%M")
    return row


def add_ratio(row, out):
    """rbar(3 N) / rbar(0.5 N) for the same model and step, written into the N = 3 row."""
    if row["N"] != 3.0:
        return
    lo = [r for r in B.read_rows(out) if r.get("model") == row["model"] and r.get("dt_ms") == row["dt_ms"]
          and r.get("N") == 0.5]
    if not lo:
        return
    lo = lo[-1]
    for key, f in (("rbar_onset_mm", "rbar_ratio_3_05_onset"), ("rbar_kin_mm", "rbar_ratio_3_05_kin")):
        if key == "rbar_onset_mm" and not (row.get("stuck_before_onset") and lo.get("stuck_before_onset")):
            continue                                       # no static phase: the onset torque is detection lag
        if row.get(key) and lo.get(key) and abs(lo[key]) > 1e-9:
            row[f] = row[key] / lo[key]
    row["rbar_ratio_winkler"] = 6 ** 0.25
    row["rbar_ratio_hertz"] = 6 ** (1 / 3)


def make_tile(out):
    rows = {r["model"]: r for r in B.read_rows(out) if r.get("film") and r.get("N") == FILM_N
            and r.get("dt_ms") == FILM_DT}
    models = [m for m in B.FILM_ORDER if m in rows and (B.BED / rows[m]["film"]).exists()]
    if not models:
        return None
    segs = []
    for m in models:
        fr = B.read_frames(B.BED / rows[m]["film"])
        n1 = rows[m]["film_segments"][0]
        segs.append((fr[:n1], fr[n1:]))
    n_r = max(len(a) for a, _ in segs)
    n_s = max(len(b) for _, b in segs)
    clips = []
    for a, b in segs:
        a = a + [a[-1]] * (n_r - len(a)) if a else []
        b = b + [b[-1]] * (n_s - len(b)) if b else []
        clips.append(a + b)
    tmp = []
    for m, c in zip(models, clips):
        p = B.MEDIA / f".tile_twist_{m}.mp4"
        B.write_h264(c, p, crf=18)
        tmp.append(p)
    n = B.tile(tmp, B.MEDIA / "twist_models.mp4", B.MEDIA / "twist_models.jpg", cols=3,
               poster_at=(n_r + 0.5 * n_s) / (n_r + n_s))
    for p in tmp:
        p.unlink()
    return models, n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--N", nargs="+", type=float, default=[0.5, 1.0, 3.0])
    ap.add_argument("--dt", nargs="+", type=float, help="physics step, ms (default: the protocol grid)")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--no-film", action="store_true")
    ap.add_argument("--tile", action="store_true", help="only assemble the tile from existing clips")
    a = ap.parse_args()
    if a.tile:
        print(make_tile(a.out))
        return
    have = B.done(a.out)
    for model, dt in B.grid(a.models, a.dt):
        for N in a.N:
            if (model, N, dt) in have:
                continue
            film = (not a.no_film) and N == FILM_N and dt == FILM_DT
            try:
                r = run_case(model, N, dt, film=film)
            except Exception as e:                          # a failed case is a row, not a lost run
                r = {"task": "twist", "model": model, "rig_spec": B.MODELS[model][0], "N": N, "dt_ms": dt,
                     "status": "failed", "error": repr(e), "traceback": traceback.format_exc()[-2000:],
                     "film": None, "script": "scripts/contact_bed_twist.py", "git_rev": B.git_rev()}
            add_ratio(r, a.out)
            H.append_row(a.out, r)
            keys = ("tau_onset_Nm", "rbar_onset_mm", "rot_pre_deg", "creep_deg_s", "rbar_kin_mm", "rbar_slide_mm",
                    "rbar_ratio_3_05_onset", "rbar_ratio_3_05_kin", "us_per_step_median", "wall_s")
            print(model, N, dt, r["status"], {k: round(r[k], 4) for k in keys if isinstance(r.get(k), float)},
                  flush=True)
    if not a.no_film:
        print("tile", make_tile(a.out))


if __name__ == "__main__":
    main()
