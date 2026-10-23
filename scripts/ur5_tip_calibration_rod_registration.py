#!/usr/bin/env python3
"""Registration of a rod session with the rod as a cylinder (QUEUE.md R19, 2026-10-23).

The lab program's fit_registration treats every indenter as a sphere: a pad touch's residual is the pad's distance to one
point, minus R, and homing fixes the rod's axis. All touch-offs of the rod plan press the same point of the axis. A pad
that sits off its nominal place in the seat therefore meets the rod at other points along it, which a cylinder allows and
a sphere penalizes (by about s^2 / 2R, 0.16 mm at s = 2 mm). Here a touch's residual is the pad's distance to the rod's
axis line, minus R. The axis is homing's (the program's), or turns with the fitted mount, or has two more angles free.
The fit starts where the program's does: the plan's nominal mount and pad, the indenter from the homing pose.

    .venv/bin/python scripts/ur5_tip_calibration_rod_registration.py SESSION [--write-reg DIR]

Prints and writes SESSION-independent JSON (--json) with each fit's rms, residuals, the TCP's change from the nominal
and from the program's fit, the rod's centre and axis. --write-reg DIR copies the session's small files to DIR with
registration.json replaced by the chosen fit (--use), for scripts/ur5_tip_calibration_check.py DIR --clearance PLAN.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/morphohand/driver/reskin5x/host"))
import ur5_tip_calibration as C  # noqa: E402

SIG = dict(C.REG_SIGMA, axis_deg=2.0)


def line_distance(geo, q, d, t0, iters=40, tol=1e-6):
    """Signed distance (mm, outside positive) between the pad's surface and the line q + t d (pad frame), and the line
    parameter t of the closest pair, by alternating projection from t0."""
    t = float(t0)
    for _ in range(iters):
        _, cp, _ = geo.surface_distance(q + t * d)
        t1 = float((cp - q) @ d)
        done = abs(t1 - t) < tol
        t = t1
        if done:
            break
    sd, _, _ = geo.surface_distance(q + t * d)
    return sd, t


def start(geo, reg_d):
    """The program's starting point: nominal mount and pad, the indenter from the homing pose."""
    reg = C.Registration(geo).load(reg_d)
    reg.T_EM, reg.T_MP = geo.T_EM0.copy(), geo.T_MP0.copy()
    reg.T_BI = reg.T_BE_home @ geo.T_EM0 @ C.inv(geo.T_IM_home)
    return reg


def fit(geo, reg, touches, model="cylinder", axis="home", sig=SIG):
    """model "sphere": the program's residual (reproduces fit_registration); "cylinder": the distance to the axis line.
    axis "homing": the program's (fixed from homing); "home": turns with the fitted mount (homing seats the rod's block
    on the mount); "free": "home" with two more angles about the axis' normals, prior sigma sig["axis_deg"]."""
    T_BI0, home = reg.T_BI, reg.T_BE_home
    c0, R_BI = C.apply(T_BI0, geo.c_I), T_BI0[:3, :3]
    T_EM0, T_MP0 = reg.T_EM.copy(), reg.T_MP.copy()
    T_MI_home = C.inv(geo.T_IM_home)
    pad = [t for t in touches if t["kind"] == "pad"]
    sd_em = np.r_[np.radians(sig["em_rot_deg"]), sig["em_t_mm"]]
    sd_mp = np.r_[np.radians(sig["mp_rot_deg"]), sig["mp_t_mm"]]
    a_h = R_BI @ geo.rod_axis
    u1 = np.cross(a_h, [0.0, 0.0, 1.0])
    u1 /= np.linalg.norm(u1)
    u2 = np.cross(a_h, u1)
    nx = 17 if axis == "free" else 15
    tcache = {}

    def unpack(x):
        T_EM = C.perturb(T_EM0, x[0:6])
        T_MP = C.perturb(T_MP0, x[6:12])
        if axis == "homing":
            a = a_h
        else:
            a = (home @ T_EM @ T_MI_home)[:3, :3] @ geo.rod_axis
            if axis == "free":
                a = C.exp_so3(x[15] * u1 + x[16] * u2) @ a
        return T_EM, T_MP, c0 + x[12:15], a

    def resid(x, keep=False):
        T_EM, T_MP, c, a = unpack(x)
        T_EP = T_EM @ T_MP
        r = []
        for t in pad:
            T_BP = np.array(t["T_BE"]) @ T_EP
            q = C.apply(C.inv(T_BP), c)
            if model == "sphere":
                r.append((geo.surface_distance(q)[0] - geo.R) / sig["pad_mm"])
                continue
            d = T_BP[:3, :3].T @ a
            t0 = tcache.get(t["id"], float((np.array(t["target"]["p"]) - q) @ d))
            s, tt = line_distance(geo, q, d, t0)
            if keep:
                tcache[t["id"]] = tt
            r.append((s - geo.R) / sig["pad_mm"])
        ch = C.apply(home @ T_EM @ T_MI_home, geo.c_I)
        r.extend(list((R_BI.T @ (c - ch)) / np.array(sig["home_mm"])))
        r.extend(list(x[0:6] / sd_em))
        r.extend(list(x[6:12] / sd_mp))
        if axis == "free":
            r.extend(list(x[15:17] / math.radians(sig["axis_deg"])))
        return np.array(r)

    x, r, cov = C.lm(lambda x: resid(x), np.zeros(nx))
    r = resid(x, keep=True)
    T_EM, T_MP, c, a = unpack(x)
    n = len(pad)
    res_um = r[:n] * sig["pad_mm"] * 1e3
    T_EP0, T_EP1 = T_EM0 @ T_MP0, T_EM @ T_MP
    out = dict(model=model, axis=axis, rms_pad_mm=float(np.sqrt(np.mean((res_um * 1e-3) ** 2))), n_pad=n,
               residual_um={t["id"]: round(float(v), 1) for t, v in zip(pad, res_um)},
               tcp_change_mm_deg=C.pose_error(T_EP0, T_EP1), centre_change_mm=float(np.linalg.norm(c - c0)),
               axis_B=a.tolist(), axis_tilt_deg=math.degrees(math.asin(min(1.0, abs(float(a[2]))))),
               axis_turn_from_homing_deg=math.degrees(math.acos(float(np.clip(a @ a_h, -1, 1)))),
               T_EM=T_EM.tolist(), T_MP=T_MP.tolist(), c_B=c.tolist(), x=x.tolist(),
               sd=np.sqrt(np.clip(np.diag(cov), 0, None)).tolist(), chi2=float(r @ r))
    if model == "cylinder":
        out["contact_along_axis_mm"] = {k: round(v, 3) for k, v in tcache.items()}
    return out


def joint_fit(parts, axis="homing", sig=SIG):
    """One mount and pad (T_EM, T_MP) for several sessions with the same tool, each with its own indenter: parts is a list
    of (geo, reg, touches), reg at the program's starting point. A sphere's centre has 3 parameters; a rod's centre 3 and,
    with axis "free", 2 angles. Each session keeps its own homing prior. Returns the shared transforms and, per session,
    the centre, the axis and the residuals."""
    geo0, reg_0 = parts[0][0], parts[0][1]
    T_EM0, T_MP0 = reg_0.T_EM.copy(), reg_0.T_MP.copy()
    for g, r, _ in parts[1:]:
        if C.pose_error(T_EM0, r.T_EM)[0] > 1e-6 or C.pose_error(T_MP0, r.T_MP)[0] > 1e-6:
            raise SystemExit("the sessions' plans start from different nominal tools")
    sd_em = np.r_[np.radians(sig["em_rot_deg"]), sig["em_t_mm"]]
    sd_mp = np.r_[np.radians(sig["mp_rot_deg"]), sig["mp_t_mm"]]
    blocks, k = [], 12
    for g, r, t in parts:
        R_BI = r.T_BI[:3, :3]
        nb = 5 if (g.kind == "rod" and axis == "free") else 3
        a_h = R_BI @ g.rod_axis
        u1 = np.cross(a_h, [0.0, 0.0, 1.0])
        u1 /= np.linalg.norm(u1)
        blocks.append(dict(geo=g, reg=r, pad=[x for x in t if x["kind"] == "pad"], k=k, nb=nb, c0=C.apply(r.T_BI, g.c_I),
                           R_BI=R_BI, a_h=a_h, u1=u1, u2=np.cross(a_h, u1), tc={}))
        k += nb
    T_MI = [C.inv(b["geo"].T_IM_home) for b in blocks]

    def resid(x, keep=False):
        T_EM = C.perturb(T_EM0, x[0:6])
        T_MP = C.perturb(T_MP0, x[6:12])
        T_EP = T_EM @ T_MP
        r = []
        for b, TMI in zip(blocks, T_MI):
            g, k0 = b["geo"], b["k"]
            c = b["c0"] + x[k0:k0 + 3]
            a = b["a_h"] if b["nb"] == 3 else C.exp_so3(x[k0 + 3] * b["u1"] + x[k0 + 4] * b["u2"]) @ b["a_h"]
            for t in b["pad"]:
                T_BP = np.array(t["T_BE"]) @ T_EP
                q = C.apply(C.inv(T_BP), c)
                if g.kind == "sphere":
                    r.append((g.surface_distance(q)[0] - g.R) / sig["pad_mm"])
                    continue
                d = T_BP[:3, :3].T @ a
                s, tt = line_distance(g, q, d, b["tc"].get(t["id"], float((np.array(t["target"]["p"]) - q) @ d)))
                if keep:
                    b["tc"][t["id"]] = tt
                r.append((s - g.R) / sig["pad_mm"])
            ch = C.apply(b["reg"].T_BE_home @ T_EM @ TMI, g.c_I)
            r.extend(list((b["R_BI"].T @ (c - ch)) / np.array(sig["home_mm"])))
            if b["nb"] == 5:
                r.extend(list(x[k0 + 3:k0 + 5] / math.radians(sig["axis_deg"])))
        r.extend(list(x[0:6] / sd_em))
        r.extend(list(x[6:12] / sd_mp))
        return np.array(r)

    x, _, cov = C.lm(lambda x: resid(x), np.zeros(k))
    r = resid(x, keep=True)
    T_EM, T_MP = C.perturb(T_EM0, x[0:6]), C.perturb(T_MP0, x[6:12])
    out = dict(T_EM=T_EM.tolist(), T_MP=T_MP.tolist(), tcp_change_mm_deg=C.pose_error(T_EM0 @ T_MP0, T_EM @ T_MP),
               x=x.tolist(), sessions=[])
    i = 0
    for b in blocks:
        k0, n = b["k"], len(b["pad"])
        res_um = r[i:i + n] * sig["pad_mm"] * 1e3
        i += n + 3 + (2 if b["nb"] == 5 else 0)
        c = b["c0"] + x[k0:k0 + 3]
        a = b["a_h"] if b["nb"] == 3 else C.exp_so3(x[k0 + 3] * b["u1"] + x[k0 + 4] * b["u2"]) @ b["a_h"]
        out["sessions"].append(dict(kind=b["geo"].kind, n_pad=n, rms_pad_mm=float(np.sqrt(np.mean((res_um * 1e-3) ** 2))),
                                    residual_um={t["id"]: round(float(v), 1) for t, v in zip(b["pad"], res_um)},
                                    c_B=c.tolist(), axis_B=a.tolist(), centre_change_mm=float(np.linalg.norm(c - b["c0"])),
                                    axis_turn_from_homing_deg=math.degrees(math.acos(float(np.clip(a @ b["a_h"], -1, 1)))),
                                    axis_tilt_deg=math.degrees(math.asin(min(1.0, abs(float(a[2])))))))
    return out


def truth_errors(sd, geo, T_EM, T_MP, T_BI):
    """Against a simulated session's truth: the pad's surface points (median, max mm), the TCP (mm, deg) and, for a rod,
    the axis line (centre off the true axis, mm; angle, deg)."""
    p = Path(sd) / "sim_truth.json"
    if not p.exists():
        return None
    st = json.loads(p.read_text())
    Tt, Tr = np.array(st["T_EP"]), np.asarray(T_EM) @ np.asarray(T_MP)
    e = np.linalg.norm(C.apply(Tt, geo.pts) - C.apply(Tr, geo.pts), axis=1)
    out = dict(pad_median_mm=float(np.median(e)), pad_max_mm=float(e.max()), tcp_mm_deg=C.pose_error(Tt, Tr))
    if geo.kind == "rod":
        at, ar = np.array(st["T_BI"])[:3, :3] @ geo.rod_axis, np.asarray(T_BI)[:3, :3] @ geo.rod_axis
        dc = C.apply(np.asarray(T_BI), geo.c_I) - np.array(st["c_B"])
        out.update(axis_offset_mm=float(np.linalg.norm(dc - (dc @ at) * at)),
                   axis_angle_deg=math.degrees(math.acos(float(np.clip(at @ ar, -1, 1)))))
    return out


def T_BI_of(geo, reg0, f):
    """The indenter frame for a fit: the rod's axis as fitted (y_I along it), the centre at c_B."""
    R0 = reg0.T_BI[:3, :3]
    a0 = R0 @ geo.rod_axis
    R = C.rot_from_to(a0, f["axis_B"]) @ R0
    return C.Tm(R, np.array(f["c_B"]) - R @ geo.c_I)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("session")
    ap.add_argument("--json", default=None, help="write the fits here")
    ap.add_argument("--write-reg", default=None, help="copy the session's small files here with the --use fit's registration")
    ap.add_argument("--use", default="cylinder-home", help="fit for --write-reg: sphere-homing, cylinder-homing, cylinder-home, "
                                                          "cylinder-free, joint-homing, joint-free")
    ap.add_argument("--joint", default=None, help="sphere sessions with the same tool, comma-separated: fit one mount and pad "
                    "to all of them and this session")
    ap.add_argument("--only-joint", action="store_true", help="skip the single-session fits")
    ap.add_argument("--exclude", default="", help="touch-off ids of this session to leave out, comma-separated (rod1t: "
                    "pad-edge-01, which met the rod 0.66 mm before the modelled pad)")
    ap.add_argument("--exclude-joint", default="", help="touch-off ids of the --joint session to leave out")
    ap.add_argument("--prior-scale", type=float, default=1.0, help="multiply the mount's and pad's prior sigmas (the "
                    "program's REG_SIGMA) by this")
    a = ap.parse_args()
    if a.prior_scale != 1.0:
        for k in ("em_rot_deg", "em_t_mm", "mp_rot_deg", "mp_t_mm"):
            SIG[k] = tuple(v * a.prior_scale for v in C.REG_SIGMA[k])
        print(f"mount and pad priors x {a.prior_scale:g}: " + ", ".join(f"{k} {SIG[k]}" for k in ("em_rot_deg", "mp_rot_deg")))
    sd = Path(a.session)
    plan = json.loads((sd / "plan.json").read_text())
    reg_d = json.loads((sd / "registration.json").read_text())
    excl = {x for x in a.exclude.split(",") if x}
    touches = [t for t in json.loads((sd / "touches.json").read_text()) if t["id"] not in excl]
    if excl:
        print(f"left out of {sd.name}: {', '.join(sorted(excl))}")
    geo = C.Geometry(plan)
    prog = reg_d.get("notes", {}).get("register")
    T_EP_prog = np.array(prog["T_EM"]) @ np.array(prog["T_MP"]) if prog else None
    fits = {}
    if prog:
        te = truth_errors(sd, geo, prog["T_EM"], prog["T_MP"], reg_d["T_BI"])
        if te:
            print(f"program's fit against the truth: pad {te['pad_median_mm']:.3f} mm median, {te['pad_max_mm']:.3f} max; TCP "
                  f"{te['tcp_mm_deg'][0]:.3f} mm, {te['tcp_mm_deg'][1]:.2f} deg"
                  + (f"; axis {te['axis_offset_mm']:.3f} mm, {te['axis_angle_deg']:.2f} deg" if "axis_offset_mm" in te else ""))
    if a.joint:
        exj = {x for x in a.exclude_joint.split(",") if x}
        others = []
        for sj in (Path(x) for x in a.joint.split(",") if x):
            gj = C.Geometry(json.loads((sj / "plan.json").read_text()))
            rj = json.loads((sj / "registration.json").read_text())
            tj = [t for t in json.loads((sj / "touches.json").read_text()) if t["id"] not in exj]
            if exj & {t["id"] for t in json.loads((sj / "touches.json").read_text())}:
                print(f"left out of {sj.name}: {', '.join(sorted(exj))}")
            others.append((sj, gj, rj, tj))
        for axis in ("homing", "free"):
            f = joint_fit([(geo, start(geo, reg_d), touches)] + [(gj, start(gj, rj), tj) for _, gj, rj, tj in others], axis)
            s0 = f["sessions"][0]
            f["T_BI"] = T_BI_of(geo, start(geo, reg_d), dict(axis_B=s0["axis_B"], c_B=s0["c_B"])).tolist()
            f["truth"] = truth_errors(sd, geo, f["T_EM"], f["T_MP"], f["T_BI"])
            fits[f"joint-{axis}"] = dict(f, rms_pad_mm=s0["rms_pad_mm"], c_B=s0["c_B"], axis_B=s0["axis_B"])
            te = f["truth"]
            print(f"joint    axis {axis:6s}: {geo.kind} {s0['rms_pad_mm'] * 1e3:6.1f} um rms over {s0['n_pad']}, "
                  + ", ".join(f"{sj.name} {s1['rms_pad_mm'] * 1e3:6.1f} um over {s1['n_pad']}"
                              for (sj, *_), s1 in zip(others, f["sessions"][1:]))
                  + f"; TCP moved {f['tcp_change_mm_deg'][0]:.3f} mm, {f['tcp_change_mm_deg'][1]:.3f} deg"
                  + (f"; rod axis {s0['axis_turn_from_homing_deg']:.2f} deg from homing's, tilt {s0['axis_tilt_deg']:.2f} deg, "
                     f"centre moved {s0['centre_change_mm']:.3f} mm" if geo.kind == "rod" else "")
                  + (f"; from the program's fit {C.pose_error(T_EP_prog, np.array(f['T_EM']) @ np.array(f['T_MP']))[0]:.3f} mm, "
                     f"{C.pose_error(T_EP_prog, np.array(f['T_EM']) @ np.array(f['T_MP']))[1]:.3f} deg" if T_EP_prog is not None else "")
                  + (f"; against the truth: pad {te['pad_median_mm']:.3f} mm median, {te['pad_max_mm']:.3f} max, axis "
                     f"{te.get('axis_offset_mm', float('nan')):.3f} mm, {te.get('axis_angle_deg', float('nan')):.2f} deg" if te else ""),
                  flush=True)
    for model, axis in (() if a.only_joint else (("sphere", "homing"), ("cylinder", "homing"), ("cylinder", "home"),
                                                  ("cylinder", "free"))):
        reg0 = start(geo, reg_d)
        f = fit(geo, reg0, touches, model, axis)
        T_EP = np.array(f["T_EM"]) @ np.array(f["T_MP"])
        if T_EP_prog is not None:
            f["tcp_from_program_mm_deg"] = C.pose_error(T_EP_prog, T_EP)
        f["truth"] = truth_errors(sd, geo, f["T_EM"], f["T_MP"], T_BI_of(geo, start(geo, reg_d), f))
        if f["truth"]:
            te = f["truth"]
            print(f"   against the truth: pad {te['pad_median_mm']:.3f} mm median, {te['pad_max_mm']:.3f} max; axis "
                  f"{te.get('axis_offset_mm', float('nan')):.3f} mm, {te.get('axis_angle_deg', float('nan')):.2f} deg")
        fits[f"{model}-{axis}"] = f
        worst = sorted(f["residual_um"].items(), key=lambda kv: -abs(kv[1]))[:4]
        print(f"{model:8s} axis {axis:6s}: {f['rms_pad_mm'] * 1e3:6.1f} um rms over {f['n_pad']}; TCP moved "
              f"{f['tcp_change_mm_deg'][0]:.3f} mm, {f['tcp_change_mm_deg'][1]:.3f} deg"
              + (f"; from the program's fit {f['tcp_from_program_mm_deg'][0]:.3f} mm, {f['tcp_from_program_mm_deg'][1]:.3f} deg"
                 if T_EP_prog is not None else "")
              + f"; axis tilt {f['axis_tilt_deg']:.2f} deg, {f['axis_turn_from_homing_deg']:.2f} deg from homing's; worst "
              + ", ".join(f"{k} {v:+.0f}" for k, v in worst), flush=True)
    if prog:
        print(f"program  (registration.json): {prog['rms_pad_mm'] * 1e3:6.1f} um rms; TCP moved "
              f"{prog['tcp_change_mm_deg'][0]:.3f} mm, {prog['tcp_change_mm_deg'][1]:.3f} deg")
    if a.json:
        Path(a.json).write_text(json.dumps(dict(session=str(sd), fits=fits), indent=1) + "\n")
    if a.write_reg:
        f = fits[a.use]
        out = Path(a.write_reg)
        out.mkdir(parents=True, exist_ok=True)
        for p in sd.iterdir():
            if p.is_file() and p.suffix in (".json", ".jsonl", ".txt"):
                shutil.copy(p, out / p.name)
        reg0 = start(geo, reg_d)
        d = dict(reg_d)
        d["T_EM"], d["T_MP"] = f["T_EM"], f["T_MP"]
        d["T_BI"] = f["T_BI"] if "T_BI" in f else T_BI_of(geo, reg0, f).tolist()
        d["notes"] = dict(reg_d.get("notes", {}), register=dict(prog or {}, **{k: f[k] for k in ("rms_pad_mm", "tcp_change_mm_deg",
                                                                                              "T_EM", "T_MP", "c_B")}),
                          refit=dict(use=a.use, script=Path(__file__).name))
        (out / "registration.json").write_text(json.dumps(d, indent=1) + "\n")
        print(f"registration of {a.use} written to {out / 'registration.json'}")


if __name__ == "__main__":
    main()
