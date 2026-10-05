#!/usr/bin/env python3
"""Contact-model comparison bed, task 4: a pinched tool shaken along its own axis under gravity.

Rig: the two-pad pinch of scripts/hom_contact_rig.py (protocol in
docs/experiments/20261005-contact_bed/PROTOCOL.md). Gravity acts along -y, the tool axis, so the
tool's weight is carried by friction alone. The rig is built with gravity off and the weight is
applied as the body force -m g y on the tool; the pads ride on x rails, so gravity along y does no
work on them and the two are the same load. Per case (model, pinch N per pad, step dt, peak
acceleration a_pk):

  settle   0.4 s at N, no tangential load (the weight is compensated)
  weight   0.5 s with the weight on; the slope over its last 0.3 s is the gravity creep
  shake    2 s of the inertial force F(t) = m a_pk sin(w t) along y, f = 5 Hz (the load of shaking
           the pads with amplitude A = a_pk / w^2)

Rigid Coulomb slips when m (g + a_pk) > 2 mu N; its drift per cycle comes from a 1-D stick-slip
integration of the same load (`coulomb_shake`). Reported: drift per cycle (tool y at the cycle
starts, mean over the 10 cycles and over the last 5), total drift, peak displacement from the
unloaded settle position, peak-to-peak over the last cycle, drop (drift beyond 10 mm or a pad with
no contact), normal-force range and the wall time of the physics step.

    PY=logs/20261001-hom_contact/venv/bin/python
    $PY scripts/contact_bed_shake.py --models mj_pads1 --N 0.5 --a 2 --dt 1        # one case
    $PY scripts/contact_bed_shake.py                                              # grid + films
    $PY scripts/contact_bed_shake.py --probe-frame /tmp/shake_probe.png --models mj_pads1
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import contact_bed_common as CB  # noqa: E402
import contact_bed_pull as P  # noqa: E402

H = CB.H
OUT = CB.BED / "shake.jsonl"
F_HZ = 5.0
T_WEIGHT = 0.5
T_SHAKE = 2.0
N_LIST = [0.2, 0.5]
A_LIST = [0.5, 1.0, 2.0, 4.0]          # peak acceleration, g
DROP_MM = 10.0
FILM_CASE = (0.5, 2.0, 1.0)            # (N, a_pk in g, dt ms) filmed for every model
FILM_STRIDE_S = 0.010                  # one frame per 10 ms of simulation at 25 fps: 0.25x playback
SCRIPT = "scripts/contact_bed_shake.py"


def coulomb_shake(N, a_pk, f=F_HZ, n_cycles=int(T_SHAKE * F_HZ), m=H.M_TOOL, mu=H.MU, h=1e-5):
    """Rigid Coulomb, equal static and kinetic mu: the tool sticks while |F_ext| <= 2 mu N and slides
    against 2 mu N otherwise. F_ext = m (-g + a_pk sin(w t)). Returns the drift at each cycle start (m),
    the time of the first slip (s), the peak slip speed in the first cycle (m/s) and the time the tool
    centre passes 40 mm (s, or None)."""
    w = 2 * math.pi * f
    Fc = 2 * mu * N
    x = v = 0.0
    stuck = True
    starts = [0.0]
    t_first, v_pk1, t_left = None, 0.0, None
    n_per = int(round(1.0 / (f * h)))
    for c in range(n_cycles):
        for k in range(n_per):
            t = (c * n_per + k) * h
            Fe = m * (-H.G + a_pk * math.sin(w * t))
            if stuck:
                if abs(Fe) <= Fc:
                    continue
                stuck = False
                if t_first is None:
                    t_first = t
                s = math.copysign(1.0, Fe)
            else:
                s = math.copysign(1.0, v) if v != 0.0 else math.copysign(1.0, Fe)
            vn = v + h * (Fe - Fc * s) / m
            if v != 0.0 and vn * v <= 0.0:          # relative velocity reaches zero: stick if friction can hold
                vn = 0.0
                stuck = abs(Fe) <= Fc
            x += h * vn
            v = vn
            if c == 0:
                v_pk1 = max(v_pk1, abs(v))
            if t_left is None and abs(x) > 0.040:
                t_left = t
        starts.append(x)
    return np.array(starts), t_first, v_pk1, t_left


class ShakeFilm(CB.BedFilm):
    """BedFilm with the main view looking down the z axis so the tool axis (y, the gravity direction)
    is vertical in the frame, and the -x pad inset rotated so y points up there too."""

    def __init__(self, spec, model):
        super().__init__(spec, 0.0, w=480, h=360, model=model, view=((0.0, 0.0, 0.0), 0.085, 90.0, -89.0),
                         inset=150)

    def frame(self, rig, text, bottom=None, speed=None):
        self._sync(rig)
        m = self.m
        cons = self._mj_contacts(rig) if rig.sim == "mujoco" else []
        faces = self._drake_faces(rig) if (rig.sim == "drake" and self.sp_live["model"] == "hydro") else []
        if self.spheres and self.A_s is None:
            self.A_s = rig.info.get("area_per_sphere_mm2", 1.0) * 1e-6
        m.geom_rgba[:] = self.rgba0
        for g in self.pad_env:
            m.geom_rgba[g, 3] = 0.18
        if self.spheres:
            force = {}
            for s, pos, n, fN, name in cons:
                force[name] = force.get(name, 0.0) + fN
            for name, g in self.pad_sph.items():
                m.geom_rgba[g] = CB.heat(force[name] / self.A_s / CB.SCALE_PRESSURE) if force.get(name, 0.0) > 1e-5 \
                    else [0.80, 0.80, 0.82, 0.55]
        self.r.update_scene(self.d, self.cam)
        self._decorate(self.r.scene, rig, cons, faces, False)
        img = self.r.render().copy()
        if self.ri is not None:
            for g in self.tool_g:
                m.geom_rgba[g, 3] = 0.30
            for g in self.pad_env:
                m.geom_rgba[g, 3] = 0.10
            self.ri.update_scene(self.d, self.cami)
            self._decorate(self.ri.scene, rig, cons, faces, True)
            ins = np.ascontiguousarray(np.rot90(self.ri.render(), -1))   # +y up, as in the main view
            if self.point:
                cap = f"-x pad, 0-{CB.SCALE_POINT_N:g} N"
            elif self.spheres or faces:
                cap = f"-x pad, 0-{CB.SCALE_PRESSURE / 1e6:g} MPa"
            else:
                cap = "-x pad"
            ins = np.array(H.annotate(ins, cap, size=11))
            ins[[0, -1], :, :] = 90
            ins[:, [0, -1], :] = 90
            y0, x0 = 30, img.shape[1] - self.inset - 6
            img[y0:y0 + self.inset, x0:x0 + self.inset] = ins
        m.geom_rgba[:] = self.rgba0
        top = f"{self.model_name}: {CB.SHORT.get(self.model_name, H.label_of(self.spec))}"
        if speed is not None:
            top += f"   {speed:g}x"
        img = H.annotate(img, top, size=14)
        if bottom:
            img = H.annotate_bottom(img, bottom, size=13)
        return img


def _bad(rig):
    if rig.sim != "mujoco":
        return False
    import mujoco
    W = mujoco.mjtWarning
    return any(rig.d.warning[int(w)].number > 0 for w in (W.mjWARN_BADQACC, W.mjWARN_BADQVEL, W.mjWARN_BADQPOS))


def run_case(model, N, dt_ms, a_pk_g, film=None, rig_hook=None, probe_png=None):
    """One shake case. `rig_hook(rig)` is applied after the rig is built (task 6 uses it to set the
    noslip iterations and impratio). Returns the row."""
    spec = CB.MODELS[model][0]
    dt = dt_ms * 1e-3
    t_wall = time.time()
    a_pk = a_pk_g * H.G
    w = 2 * math.pi * F_HZ
    mg = H.M_TOOL * H.G
    row = {"task": "T4", "model": model, "rig_spec": spec, "N": N, "dt_ms": dt_ms, "a_pk_g": a_pk_g,
           "f_hz": F_HZ, "A_mm": a_pk / w ** 2 * 1e3, "T_weight_s": T_WEIGHT, "T_shake_s": T_SHAKE,
           "m_tool_kg": H.M_TOOL, "mu": H.MU,
           "gravity": "-y (tool axis), applied as the body force -m g on the tool; rig gravity off"}
    load_ratio = H.M_TOOL * (H.G + a_pk) / (2 * H.MU * N)
    cs, c_t1, c_v1, c_left = coulomb_shake(N, a_pk)
    row.update(coulomb_load_ratio=load_ratio, coulomb_slips=bool(load_ratio > 1.0),
               coulomb_drift_per_cycle_mm=float((cs[-1] - cs[0]) / (len(cs) - 1) * 1e3),
               coulomb_drift_first_cycle_mm=float((cs[1] - cs[0]) * 1e3),
               coulomb_drift_total_mm=float((cs[-1] - cs[0]) * 1e3),
               coulomb_t_first_slip_s=c_t1, coulomb_v_peak_first_cycle_mm_s=c_v1 * 1e3,
               coulomb_t_left_pinch_s=c_left,
               coulomb_drop=bool(abs(cs[-1] - cs[0]) * 1e3 > DROP_MM))

    rig = CB.new_rig(spec, dt)
    if rig_hook is not None:
        rig_hook(rig)
    p_set, a_set, c0 = P.settle(rig, N)
    row["settle"] = c0
    row["info"] = getattr(rig, "info", {})
    y_set = float(p_set[1])
    n_w = int(round(T_WEIGHT / dt))
    n_s = int(round(T_SHAKE / dt))
    per = int(round(1.0 / (F_HZ * dt)))
    stride_c = max(1, int(round(0.010 / dt)))
    stride_f = max(1, int(round(FILM_STRIDE_S / dt)))
    renderer = ShakeFilm(spec, model) if (film or probe_png) else None
    frames = []
    T, Y, Fy, Wt = [], [], [], []
    trace = []
    nsum_max, nsum_min = 0.0, math.inf
    status, bad_t = "complete", None
    lost_t = None
    nmin, nmax = [math.inf, math.inf], [0.0, 0.0]
    ncon = []
    for k in range(n_w + n_s):
        ts = (k - n_w) * dt                     # shake time at the start of this step
        f_y = -mg + (H.M_TOOL * a_pk * math.sin(w * ts) if k >= n_w else 0.0)
        rig.set_tool_wrench(np.array([0.0, f_y, 0.0]), np.zeros(3))
        w0 = time.perf_counter()
        rig.step(dt)
        Wt.append(time.perf_counter() - w0)
        st = rig.tool_state()
        y = float(st["pos"][1]) - y_set
        te = ts + dt
        T.append(te), Y.append(y), Fy.append(f_y)
        if not np.all(np.isfinite(st["pos"])) or _bad(rig):
            status, bad_t = "failed", te
            break
        if k % stride_c == 0 or k == n_w + n_s - 1:
            c = rig.contacts()
            nl, nr = c["L"]["N"], c["R"]["N"]
            trace.append([round(te, 4), y * 1e3, f_y, nl, nr, c["L"]["n"], c["R"]["n"]])
            if k >= n_w:
                nsum_max, nsum_min = max(nsum_max, nl + nr), min(nsum_min, nl + nr)
                nmin = [min(nmin[0], nl), min(nmin[1], nr)]
                nmax = [max(nmax[0], nl), max(nmax[1], nr)]
                ncon.append(c["L"]["n"] + c["R"]["n"])
            if (c["L"]["n"] == 0 or c["R"]["n"] == 0) and lost_t is None:
                lost_t = te
        if renderer is not None and (k % stride_f == 0) and (film or len(frames) == 0 and k >= n_w + per // 4):
            a_now = a_pk * math.sin(w * ts) / H.G if k >= n_w else 0.0
            phase = "shake" if k >= n_w else "weight on"
            c = rig.contacts()
            frames.append(renderer.frame(rig, "", bottom=(f"t {te:+5.2f} s {phase:9s} a {a_now:+4.1f} g  dy {y * 1e3:+7.3f} mm  "
                                                          f"N {c['L']['N']:.2f}/{c['R']['N']:.2f}"), speed=0.25))
            if probe_png and not film:
                from PIL import Image
                Image.fromarray(frames[0]).save(probe_png)
                renderer.close()
                return None
        if abs(y) > 0.040:                      # tool centre 40 mm along: it has left the pinch
            break
    T, Y = np.array(T), np.array(Y)
    row["us_per_step_median"] = float(np.median(Wt) * 1e6)
    row["us_per_step_mean"] = float(np.mean(Wt) * 1e6)
    row["n_steps"] = len(Wt)
    if status == "failed":
        row.update(status="failed", t_failed=bad_t)
    else:
        mw = (T > -0.3 + 1e-9) & (T <= 1e-9)
        row["creep_g_mm_s"] = float(np.polyfit(T[mw], Y[mw], 1)[0] * 1e3) if mw.sum() > 3 else None
        row["sag_g_mm"] = float(np.interp(0.0, T, Y) * 1e3)
        ms = T > 1e-9
        Ys, Ts = Y[ms], T[ms]
        y_shake0 = float(np.interp(0.0, T, Y))
        n_cyc = int(round(T_SHAKE * F_HZ))
        starts = [float(np.interp(j / F_HZ, T, Y)) if j / F_HZ <= Ts[-1] + 1e-12 else math.nan
                  for j in range(n_cyc + 1)]
        starts[0] = y_shake0
        done_c = [j for j in range(n_cyc + 1) if math.isfinite(starts[j])]
        nc = done_c[-1]
        cyc = np.diff(np.array(starts[:nc + 1]))
        row["n_cycles_completed"] = nc
        row["drift_per_cycle_mm"] = float((starts[nc] - starts[0]) / nc * 1e3) if nc >= 1 else None
        row["drift_per_cycle_last5_mm"] = float((starts[-1] - starts[5]) / 5 * 1e3) if nc == n_cyc else None
        row["drift_first_cycle_mm"] = float(cyc[0] * 1e3) if nc >= 1 else None
        row["drift_cycles_mm"] = [float(x * 1e3) for x in cyc]
        vs = np.gradient(Ys, Ts) if len(Ys) > 2 else np.zeros_like(Ys)
        m1 = Ts <= 1.0 / F_HZ
        row["v_peak_first_cycle_mm_s"] = float(np.abs(vs[m1]).max() * 1e3) if m1.any() else None
        slip = np.where(np.abs(vs) > 0.002)[0]
        row["t_first_slip_s"] = float(Ts[slip[0]]) if len(slip) else None   # tool speed first above 2 mm/s
        row["t_left_pinch_s"] = float(Ts[-1]) if abs(Ys[-1]) > 0.040 else None
        row["N_sum_max_over_2N"] = float(nsum_max / (2 * N))
        row["N_sum_min_over_2N"] = float(nsum_min / (2 * N)) if math.isfinite(nsum_min) else None
        row["drift_total_mm"] = float((Ys[-1] - y_shake0) * 1e3)
        row["peak_disp_mm"] = float(np.abs(Ys).max() * 1e3)               # from the unloaded settle position
        row["peak_rel_disp_mm"] = float(np.abs(Ys - y_shake0).max() * 1e3)  # from the shake start
        last = Ts >= Ts[-1] - 1.0 / F_HZ
        row["pp_last_cycle_mm"] = float((Ys[last].max() - Ys[last].min()) * 1e3)
        row["creep_g_per_cycle_mm"] = (row["creep_g_mm_s"] / F_HZ) if row["creep_g_mm_s"] is not None else None
        row["left_pinch"] = bool(abs(Ys[-1]) > 0.040)
        row["t_contact_lost"] = lost_t
        row["drop"] = bool(abs(row["drift_total_mm"]) > DROP_MM or lost_t is not None or row["left_pinch"])
        row["N_min"] = [float(x) for x in nmin]
        row["N_max"] = [float(x) for x in nmax]
        row["n_contacts_mean"] = float(np.mean(ncon)) if ncon else None
        row["status"] = "complete"
    row["trace_cols"] = ["t_shake", "y_mm", "F_y", "N_L", "N_R", "n_L", "n_R"]
    row["trace"] = trace
    row["film"] = None
    if renderer is not None and film:
        CB.write_h264(frames, film, fps=25, crf=27)
        row["film"] = str(Path(film).relative_to(CB.BED))
        row["film_playback"] = 0.25
        renderer.close()
    row["script"] = SCRIPT
    row["git_rev"] = CB.git_rev()
    row["wall_s"] = time.time() - t_wall
    row["when"] = time.strftime("%Y-%m-%d %H:%M")
    return row


def cases(models, Ns, As, dts):
    out = []
    for model in models:
        if model in CB.MAIN:
            for dt in dts:
                for N in Ns:
                    for a in As:
                        out.append((model, N, dt, a))
    for model, dt in CB.CONSISTENCY:          # pads consistency set at N 0.5 N, a_pk 2 g
        if FILM_CASE[0] not in Ns or FILM_CASE[1] not in As:
            continue
        if model in models and (model, FILM_CASE[0], dt, FILM_CASE[1]) not in out:
            out.append((model, FILM_CASE[0], dt, FILM_CASE[1]))
    return out


def tiles():
    clips = [CB.MEDIA / f"shake_{m}.mp4" for m in CB.FILM_ORDER if (CB.MEDIA / f"shake_{m}.mp4").exists()]
    if clips:
        n = CB.tile(clips, CB.MEDIA / "shake_models.mp4", CB.MEDIA / "shake_models_poster.jpg", cols=3, poster_at=0.62)
        print("tile", len(clips), "clips", n, "frames", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=CB.MAIN + ["mj_pads2", "mj_pads05"])
    ap.add_argument("--N", nargs="+", type=float, default=N_LIST)
    ap.add_argument("--a", nargs="+", type=float, default=A_LIST, help="peak acceleration, g")
    ap.add_argument("--dt", nargs="+", type=float, default=CB.MAIN_DT, help="physics step, ms")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--no-films", action="store_true")
    ap.add_argument("--probe-frame", help="save one frame of the film case of the first model here and exit")
    a = ap.parse_args()
    if a.probe_frame:
        run_case(a.models[0], FILM_CASE[0], FILM_CASE[2], FILM_CASE[1], probe_png=a.probe_frame)
        print("probe frame", a.probe_frame)
        return
    have = CB.done(a.out, key=("model", "N", "dt_ms", "a_pk_g"))
    for model, N, dt, acc in cases(a.models, a.N, a.a, a.dt):
        if (model, N, dt, acc) in have:
            continue
        film = None
        if not a.no_films and (N, acc, dt) == FILM_CASE:
            film = CB.MEDIA / f"shake_{model}.mp4"
        r = run_case(model, N, dt, acc, film=film)
        H.append_row(a.out, r)
        keys = ("coulomb_load_ratio", "coulomb_drift_first_cycle_mm", "drift_first_cycle_mm", "drift_per_cycle_mm",
                "drift_total_mm", "N_sum_max_over_2N",
                "peak_disp_mm", "pp_last_cycle_mm", "creep_g_mm_s", "drop", "us_per_step_median", "wall_s")
        print(model, N, dt, acc, r.get("status"), {k: (round(r[k], 5) if isinstance(r.get(k), float) else r.get(k))
                                                   for k in keys if k in r}, flush=True)
    if not a.no_films:
        tiles()


if __name__ == "__main__":
    main()
