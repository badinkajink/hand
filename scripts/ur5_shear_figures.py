#!/usr/bin/env python3
"""Figures and the film of the page on shear, friction and impratio of the printed fingertip
(docs/experiments/20261026-shear_friction_impratio/), from the rows of scripts/ur5_shear_impratio.py (impratio under the
trainer's solver), scripts/ur5_shear_skin.py (the sprung skin on the UR5 slides) and scripts/ur5_shear_torsion.py (friction
torque). The slides' figure is drawn by scripts/ur5_shear_slides.py figure.

    .venv/bin/python scripts/ur5_shear_figures.py figures        # media/*.png and data/20261026-numbers.json
    MUJOCO_GL=egl .venv/bin/python scripts/ur5_shear_figures.py film
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
D = ROOT / "docs/experiments/20261026-shear_friction_impratio"
DATA, MEDIA = D / "data", D / "media"
BLUE, ORANGE, GREEN, GREY = "#2e5fa8", "#b06010", "#3a8a4a", "0.45"
IR_COL = {10.0: "#9bb7e0", 100.0: BLUE, 300.0: "#16305e", 1000.0: "#16305e"}


def rows(name):
    p = DATA / name
    return [json.loads(l) for l in open(p) if l.strip()] if p.exists() else []


def plt_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 13, "axes.titlesize": 13, "axes.labelsize": 13,
                         "legend.fontsize": 11.5, "xtick.labelsize": 12, "ytick.labelsize": 12,
                         "mathtext.fontset": "dejavuserif"})
    return plt


def save(fig, name):
    MEDIA.mkdir(parents=True, exist_ok=True)
    for ax in fig.axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(MEDIA / name, dpi=150, bbox_inches="tight", facecolor="white")
    import matplotlib.pyplot as plt
    plt.close(fig)
    print("wrote", (MEDIA / name).relative_to(ROOT), flush=True)


# ------------------------------------------------------------------------------------------------ numbers

def numbers():
    N = {}
    S = json.loads((DATA / "20261026-ur5_slides_summary.json").read_text())
    N["slides"] = S
    # creep in the UR5 rig
    N["creep"] = {f"{r['session']}_E{r['E_MPa']:g}_{r['numerics']}_ir{r['ir']:g}": dict(
        median_um_s=r["creep_mm_s"]["median"] * 1e3, p90_um_s=r["creep_mm_s"]["p90"] * 1e3, cap=r["cap_share"],
        us=r["us_per_step_median"], n=r["n"]) for r in rows("20261026-impratio_creep.jsonl")}
    # replays
    R = rows("20261026-impratio_replays.jsonl")
    rep = {}
    for st in ("d1d8", "icem"):
        for E in (10.0, 2.0):
            for ir in (10.0, 100.0, 300.0):
                X = [r for r in R if r["set"] == st and r["E_MPa"] == E and r["ir"] == ir]
                if not X:
                    continue
                if st == "d1d8":
                    held = [r for r in X if r.get("held_end")]
                    div = [r for r in X if r["status"] != "complete"]
                    early = [r for r in div if r.get("t_unstable_s") is not None and r["t_unstable_s"] < 2.0]
                    rep[f"{st}_E{E:g}_ir{ir:g}"] = dict(
                        n=len(X), held=len(held), diverged=len(div), diverged_before_2s=len(early),
                        dropped=len(X) - len(held) - len(div), tr_s=X[0]["tr_s"],
                        creep_um_s=float(np.median([r["creep_um_s"] for r in held])) if held else None,
                        F_hold_N={f: float(np.median([r["F_hold_N"][f] for r in held])) for f in ("thumb", "index", "middle")} if held else None,
                        turn_deg=float(np.median([r["turn_end_deg"] for r in held])) if held else None,
                        cap_share=float(np.mean([r.get("cap_share") or 0 for r in X])),
                        cap_share_max=float(np.max([r.get("cap_share") or 0 for r in X])),
                        us_held=float(np.median([r["us_per_step_median"] for r in held])) if held else None,
                        hands_held=sorted({r["plan"] for r in held}))
                else:
                    held = [r for r in X if r.get("held")]
                    rep[f"{st}_E{E:g}_ir{ir:g}"] = dict(
                        n=len(X), held=len(held), unstable=sum(r["status"] == "unstable" for r in X),
                        turn_deg=[r.get("turn_end_deg") for r in X], logged=[r["logged"]["turn_end_deg"] for r in X],
                        cap_share=float(np.mean([r.get("cap_share") or 0 for r in X])),
                        us=float(np.median([r["us_per_step_median"] for r in X if r.get("us_per_step_median")])))
    N["replays"] = rep
    # grasp check
    G = rows("20261026-impratio_mjw.jsonl")
    N["grasp"] = {f"{r['sim']}_{r.get('run', 'pads')}{'_slides' if r.get('skin_k') else ''}_E{r['E_MPa']:g}_ir{r['ir']:g}": dict(
        n=r["n"], held=r["held"], held_placements=r.get("held_placements"), nonfinite=r["nonfinite"],
        slip_um_s=r["slip_um_s"]["median"], cap=r.get("cap_share"), tc_s=r.get("tc_s"),
        us=r.get("us_per_step_median") or r.get("us_per_world_step"), nefc_max=r.get("nefc_max"))
        for r in G}
    # skin on the slides
    K = []
    for ir in (10, 100, 1000):
        K += rows(f"20261026-skin_slides_mu_ir{ir}.jsonl") or rows(f"20261026-skin_slides_ir{ir}.jsonl")
    sk = {}
    for model in ("pads", "skin"):
        for ir in (10.0, 100.0, 1000.0):
            for F in (1.0, 2.0):
                X = [r for r in K if r["model"] == model and r["ir"] == ir and r["press_N"] == F]
                if not X:
                    continue
                dirs = [x for r in X for x in r["directions"]]
                fits = [x["fit"] for x in dirs if x["fit"]]
                sk[f"{model}_ir{ir:g}_{F:g}N"] = dict(
                    slides=len(X), directions=len(dirs), rms_N=float(np.median([x["rms_vs_measured_N"] for x in dirs])),
                    give_um_N=float(np.median([f["give_um_N"] for f in fits])), mu=float(np.median([f["mu"] for f in fits])),
                    give_measured_um_N=float(np.median([x["measured"]["give_um_N"] for x in dirs])),
                    mu_measured=float(np.median([x["measured"]["mu"] for x in dirs])),
                    s_back_N=float(np.median([x["s_back_N"] for x in dirs])),
                    z_back_measured_N=float(np.median([x["measured"]["z_back_N"] for x in dirs if x["measured"]["z_back_N"] is not None])),
                    us=float(np.median([r["us_per_step_median"] for r in X])))
    N["skin_slides"] = sk
    # friction from the vector fit of each slide (ur5_shear_slides.py vector)
    V = rows("20261026-ur5_slides_vector_cb2t.jsonl")
    vec = {}
    for F in (1.0, 2.0):
        X = [v for v in V if v["press_N"] == F]
        mx = [x for v in X for x in v["max_ft_over_fn_by_dir"]]
        vec[f"{F:g}"] = dict(n=len(X), **{f"mu_{m}": dict(median=float(np.median([v["fits"][m]["mu"] for v in X])),
                                                        p25=float(np.percentile([v["fits"][m]["mu"] for v in X], 25)),
                                                        p75=float(np.percentile([v["fits"][m]["mu"] for v in X], 75)))
                                         for m in ("none", "const", "prop")},
                             b_over_fn=float(np.median([v["fits"]["const"]["b_over_fn"] for v in X])),
                             slip_share=float(np.median([v["fits"]["const"]["slip_share"] for v in X])),
                             give_um_N=float(np.median([v["fits"]["const"]["give_um_N"] for v in X])),
                             max_ft_over_fn=dict(median=float(np.median(mx)), p10=float(np.percentile(mx, 10)),
                                                 p90=float(np.percentile(mx, 90))))
    pairs = {}
    for v in V:
        pairs.setdefault(v["action"].rsplit("-", 1)[0], {})[v["press_N"]] = v
    rat, ang = [], []
    for p_ in pairs.values():
        if 1.0 in p_ and 2.0 in p_:
            b1, b2 = np.array(p_[1.0]["fits"]["const"]["b"]), np.array(p_[2.0]["fits"]["const"]["b"])
            rat.append(np.linalg.norm(b2) / max(np.linalg.norm(b1), 1e-9))
            ang.append(math.degrees(math.acos(np.clip(b1 @ b2 / np.linalg.norm(b1) / np.linalg.norm(b2), -1, 1))))
    S1 = [json.loads(l) for l in open(DATA / "20261026-ur5_slides_cb2t.jsonl")]
    st = [r["start"]["f_t_N"] / r["start"]["f_n_N"] for r in S1 if r["from_rest"]]
    vec["offset_pairs"] = dict(n=len(rat), ratio_median=float(np.median(rat)), ratio_p25=float(np.percentile(rat, 25)),
                               ratio_p75=float(np.percentile(rat, 75)), angle_median_deg=float(np.median(ang)))
    vec["press_ft_over_fn"] = float(np.median(st))
    N["vector"] = vec
    # torsion
    T = rows("20261026-torsion_pads.jsonl")
    tor = {}
    for F in (1.0, 2.5, 5.0):
        X = [r for r in T if r["F_N"] == F and r.get("tau_on_Nmm")]
        if X:
            tor[f"{F:g}"] = dict(n=len(X), mu=X[0]["mu"], tau_on_Nmm=[r["tau_on_Nmm"] for r in X], arm_mm=[r["arm_on_mm"] for r in X],
                                 hertz_Nmm=X[0]["hertz"]["tau_Nmm"], measured=X[0]["measured_cb2t"])
    N["torsion"] = tor
    (DATA / "20261026-numbers.json").write_text(json.dumps(N, indent=1) + "\n")
    return N


# ------------------------------------------------------------------------------------------------ figures

def fig_impratio(N):
    plt = plt_setup()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10.0, 10.0), gridspec_kw=dict(height_ratios=(1.25, 1.0)))
    ir = np.array([10.0, 100.0, 300.0])
    C = N["creep"]
    for key, lab, col, mk in (("sphere2_E0.9008", "UR5 rig, 5 mm sphere, 0.90 MPa", ORANGE, "o"),
                              ("rod1_E1.9974", "UR5 rig, 12.5 mm rod, 2.0 MPa", BLUE, "s"),
                              ("sphere2_E10", "UR5 rig, 5 mm sphere, 10 MPa", ORANGE, "^"),
                              ("rod1_E10", "UR5 rig, 12.5 mm rod, 10 MPa", BLUE, "v")):
        y = [C.get(f"{key}_rl_ir{x:g}", {}).get("median_um_s") for x in ir]
        if None in y:
            continue
        a1.plot(ir, y, "-" + mk, color=col, ms=7, lw=1.4, mfc="white" if "E10" in key else col, label=lab)
    R = N["replays"]
    for E, mk in ((10.0, "D"), (2.0, "d")):
        y = [R.get(f"d1d8_E{E:g}_ir{x:g}", {}).get("creep_um_s") for x in ir]
        a1.plot(ir, y, "--" + mk, color=GREEN, ms=7, lw=1.4, mfc="white" if E == 10.0 else GREEN,
                label=f"D1-D8 replays, held, {E:g} MPa")
    g = N["grasp"]
    y = [g.get(f"mjw_pads_E10_ir{x:g}", {}).get("slip_um_s") for x in (10.0, 100.0)]
    a1.plot([10.0, 100.0], y, ":x", color="black", ms=8, lw=1.4, label="Scripted grasp, MuJoCo-Warp, 10 MPa")
    a1.plot(ir, 300.0 * 10.0 / ir, "-", color="0.8", lw=4, zorder=0, label="Slope of 1/impratio")
    a1.set_xscale("log")
    a1.set_yscale("log")
    a1.set_xticks(ir)
    a1.set_xticklabels([f"{x:g}" for x in ir])
    a1.set_xlabel("Impratio")
    a1.set_ylabel("Creep (\N{MICRO SIGN}m/s)")
    a1.set_title("(a) Creep under the trainer's solver", loc="left")
    a1.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=11)
    # (b) outcomes of the 24 D1-D8 replays per setting
    labels, H, Dr, Dv = [], [], [], []
    for E in (10.0, 2.0):
        for x in ir:
            r = R.get(f"d1d8_E{E:g}_ir{x:g}")
            if not r:
                continue
            labels.append(f"{E:g} MPa\n{x:g}")
            H.append(r["held"])
            Dr.append(r["dropped"])
            Dv.append(r["diverged"])
    xs = np.arange(len(labels))
    a2.bar(xs, H, 0.7, color=BLUE, label="Held")
    a2.bar(xs, Dr, 0.7, bottom=H, color="0.8", label="Dropped")
    a2.bar(xs, Dv, 0.7, bottom=np.add(H, Dr), color=ORANGE, label="Diverged")
    for x_, h_, d_, v_ in zip(xs, H, Dr, Dv):
        a2.text(x_, h_ / 2, str(h_), ha="center", va="center", color="white", fontsize=12)
        if v_:
            a2.text(x_, h_ + d_ + v_ / 2, str(v_), ha="center", va="center", color="white", fontsize=12)
    a2.set_xticks(xs)
    a2.set_xticklabels(labels)
    a2.set_ylim(0, 26)
    a2.set_ylabel("D1-D8 replays (of 24)")
    a2.set_xlabel("Pad modulus and impratio")
    a2.set_title("(b) Outcomes of the D1-D8 replays", loc="left")
    a2.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=11)
    fig.tight_layout()
    save(fig, "20261026-impratio.png")


def fig_skin(N):
    plt = plt_setup()
    K = []
    for ir in (10, 100, 1000):
        K += rows(f"20261026-skin_slides_mu_ir{ir}.jsonl") or rows(f"20261026-skin_slides_ir{ir}.jsonl")
    if not K:
        return
    S = {(json.loads(l)["action"], json.loads(l)["dir_index"]): json.loads(l)
         for l in open(DATA / "20261026-ur5_slides_cb2t.jsonl")}
    Z = np.load(DATA / "20261026-ur5_slide_traces_cb2t.npz")
    meta = json.loads(str(Z["meta"]))
    aid, k = "slide-000-1N", 2
    i = next(j for j, m in enumerate(meta) if m["action"] == aid and m["dir_index"] == k)
    off = S[(aid, k)]["epp"]["s_off_N"]
    fig, axs = plt.subplots(1, 3, figsize=(10.0, 4.4), sharey=True)
    for ax, ir in zip(axs, (10.0, 100.0, 1000.0)):
        ax.plot(Z[f"d_{i}"] * 1e3, Z[f"s_{i}"] - off, "-", color="0.75", lw=1.0, label="Measured")
        for model, col, lab in (("pads", ORANGE, "1 mm pads"), ("skin", BLUE, "Pads on the sized skin")):
            r = next((r for r in K if r["action"] == aid and r["model"] == model and r["ir"] == ir), None)
            if r is None:
                continue
            tr = r["directions"][k]["trace"]
            ax.plot(np.array(tr["d"]), np.array(tr["s"]), "-", color=col, lw=1.5, label=lab)
        ax.axhline(0.0, color="0.88", lw=0.8, zorder=0)
        ax.set_title(f"({'abc'[[10.0, 100.0, 1000.0].index(ir)]}) Impratio {ir:g}", loc="left")
        ax.set_xticks([0, 100, 200])
        ax.set_xlabel("Travel (\N{MICRO SIGN}m)")
    axs[0].set_ylabel("Shear against the travel (N)")
    h, l_ = axs[0].get_legend_handles_labels()
    fig.legend(h, l_, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    save(fig, "20261026-skin_slides.png")


def fig_torsion(N):
    plt = plt_setup()
    T = N["torsion"]
    if not T:
        return
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    F = np.linspace(0.5, 5.5, 50)
    import ur5_shear_torsion as TO
    for mu_, ls in ((0.37, ":"), (0.44, "-")):
        ax.plot(F, [TO.hertz_torque_Nmm(f, mu=mu_)[0] for f in F], ls, color="black", lw=1.2,
                label="Hertz patch, \N{GREEK SMALL LETTER MU} " + f"{mu_:.2f}, 0.90 MPa")
    for k, (Fk, v) in enumerate(sorted(T.items(), key=lambda kv: float(kv[0]))):
        f = float(Fk)
        ax.plot([f] * len(v["tau_on_Nmm"]), v["tau_on_Nmm"], "o", color=BLUE, ms=7, label="1 mm pads, spin onset" if k == 0 else None)
        m = v["measured"]
        ax.errorbar([f + 0.08], [m["median_Nmm"]], yerr=[[m["median_Nmm"] - m["min_Nmm"]], [m["max_Nmm"] - m["median_Nmm"]]],
                    fmt="s", ms=7, color=ORANGE, capsize=3, label="HEX-E on cb2t, median and range" if k == 0 else None)
    ax.axhspan(-2.0, 2.0, color="0.9", zorder=0)
    ax.text(0.55, 1.75, "HEX-E torque noise, \N{PLUS-MINUS SIGN}1-2 N mm", fontsize=11, color="0.3", va="top")
    ax.axhline(0.0, color="0.7", lw=0.8)
    ax.set_xlabel("Normal force (N)")
    ax.set_ylabel("Friction torque about the normal (N mm)")
    ax.set_xlim(0.4, 5.6)
    ax.legend(loc="upper left", frameon=False, fontsize=11)
    fig.tight_layout()
    save(fig, "20261026-torsion.png")


# ------------------------------------------------------------------------------------------------ film

FILM_CASES = (("D1", 10.0), ("D1", 100.0), ("D2", 10.0), ("D2", 100.0))
FILM_W, FILM_H, FILM_FPS = 480, 360, 50


def render_replay(hand, ir, seed=0, E=10e6):
    """One D1-D8 replay at the trainer's numerics (ur5_shear_impratio.run_d1d8's loop), a frame every 20 ms of simulated
    time from a fixed camera on the tool; stops at a divergence (MuJoCo's reset) and returns the frames and the time."""
    import mujoco
    import fe_contact_calibration_sim as FS
    import reorient_backends as RB
    import ur5_shear_impratio as U
    key = FS.spec_of(U.TIP)["key"]
    arm = f"rl{ir:g}"
    FS.ARMS[arm] = dict(sim="mujoco", numerics="rl", ir=ir, mu="1.0")
    RB.PAD_TR = 0.02
    FS.LOGS = U.LOGS / "tr0.02"
    scene, meta = FS.d1d8_scene(U.TIP, key, arm, E, 0.001, None, hand)
    plan, traj = RB.load_plan(hand)
    m = mujoco.MjModel.from_xml_path(str(scene))
    U.set_rl(m, ir)
    d = mujoco.MjData(m)
    tb = m.body(RB.OBJ).id
    qa = m.jnt_qposadr[m.body_jntadr[tb]]
    tool7 = np.asarray(meta["tool7"], float)
    dx, dy, dyaw = RB.jitter(seed)
    d.qpos[qa:qa + 3] = tool7[:3] + [dx, dy, 0.0]
    d.qpos[qa + 3:qa + 7] = RB._yaw_quat(tool7[3:], dyaw)
    act = {}
    for f in RB.FINGERS:
        for j in RB.JOINTS:
            n = f"{f}_{j}"
            d.qpos[m.jnt_qposadr[m.joint(n).id]] = meta["q0"][n]
            act[n] = m.actuator(f"a_{n}").id
    mujoco.mj_forward(m, d)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = d.qpos[qa:qa + 3]
    cam.distance, cam.azimuth, cam.elevation = 0.22, 0.0, 0.0
    r = mujoco.Renderer(m, FILM_H, FILM_W)
    frames, t_bad = [], None
    k_frame = int(round(0.02 / m.opt.timestep))
    step = 0
    try:
        for t_end, tgt, _ in RB.schedule(plan, traj):
            for n_, a_ in act.items():
                d.ctrl[a_] = tgt[n_]
            while d.time < t_end - 1e-9:
                t_prev = float(d.time)
                mujoco.mj_step(m, d)
                step += 1
                if U.diverged(d):
                    t_bad = t_prev
                    break
                if step % k_frame == 0:
                    r.update_scene(d, cam)
                    frames.append((float(d.time), r.render().copy()))
            if t_bad is not None:
                break
    finally:
        r.close()
    return frames, t_bad, float(RB.schedule(plan, traj)[-1][0])


def label(img, text, sub):
    from PIL import Image, ImageDraw, ImageFont
    bar = 44
    im = Image.new("RGB", (img.shape[1], img.shape[0] + bar), "white")
    im.paste(Image.fromarray(img), (0, bar))
    dr = ImageDraw.Draw(im)
    try:
        f1 = ImageFont.truetype("DejaVuSerif.ttf", 17)
        f2 = ImageFont.truetype("DejaVuSerif.ttf", 14)
    except OSError:
        f1 = f2 = ImageFont.load_default()
    dr.text((8, 3), text, fill="black", font=f1)
    dr.text((8, 24), sub, fill=(70, 70, 70), font=f2)
    return np.asarray(im)


def cmd_film(a):
    import imageio.v2 as imageio
    runs = []
    for hand, ir in FILM_CASES:
        fr, t_bad, T = render_replay(hand, ir)
        runs.append((hand, ir, fr, t_bad, T))
        print(f"{hand} impratio {ir:g}: {len(fr)} frames, diverged at {t_bad}", flush=True)
    n = max(len(r[2]) for r in runs)
    T_all = max(r[4] for r in runs)
    n = int(round(T_all / 0.02))
    out = MEDIA / "20261026-impratio_replays.mp4"
    MEDIA.mkdir(parents=True, exist_ok=True)
    w = imageio.get_writer(out, fps=FILM_FPS, codec="libx264", quality=8, macro_block_size=8,
                           ffmpeg_params=["-pix_fmt", "yuv420p"])
    poster = None
    for k in range(n):
        tiles = []
        for hand, ir, fr, t_bad, T in runs:
            j = min(k, len(fr) - 1)
            t, img = fr[j]
            if t_bad is not None and k >= len(fr):
                sub = f"diverged at {t_bad:.2f} s (MuJoCo reset the state)"
            else:
                sub = f"t = {t:4.2f} s"
            tiles.append(label(img, f"{hand}, impratio {ir:g}, 10 MPa pads", sub))
        top = np.concatenate(tiles[:2], axis=1)
        bot = np.concatenate(tiles[2:], axis=1)
        frame = np.concatenate([top, bot], axis=0)
        w.append_data(frame)
        if k == int(round(2.4 / 0.02)):
            poster = frame
    w.close()
    imageio.imwrite(MEDIA / "20261026-impratio_replays.png", poster if poster is not None else frame)
    print("wrote", out.relative_to(ROOT), f"{out.stat().st_size / 1e6:.2f} MB", flush=True)


def cmd_figures(a):
    N = numbers()
    fig_impratio(N)
    fig_skin(N)
    fig_torsion(N)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("figures")
    sub.add_parser("film")
    a = ap.parse_args()
    {"figures": cmd_figures, "film": cmd_film}[a.cmd](a)


if __name__ == "__main__":
    main()
