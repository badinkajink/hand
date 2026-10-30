#!/usr/bin/env python3
"""The pin-push's middle finger with and without the re-anchor of hom.HomParams.reanchor, on the CPU (2026-10-30).

The code walkthrough's pin-push example (grid call 280: mounts 95 mm apart along palm x and 75 mm along palm y; 25 mm
cylinder from its foundational pose; seed 0) runs under the variants base and reanchor of hom_tasks.VARIANTS. Every
period keeps the middle finger's pad force, GCF distance, servo gap, joint angles, joint reference and servo target, and
the cylinder's angles. Keyframes at the instants that decide the middle finger's force draw the finger at its measured
angles, at its joint reference (orange) and at its servo target (cyan), from four sides.

    ~/.claude/bin/resguard.sh run --mem 3G --cpu 100 -- .venv/bin/python scripts/hom_reanchor_check.py cpu
Writes docs/experiments/20261029-hom_control_review/rows/20261030-reanchor_check.json and
media/20261030-reanchor_keyframes_<variant>.png, media/20261030-reanchor_trace.png.

    .venv/bin/python scripts/hom_reanchor_check.py grid
compares the GPU rows of the 25 mm pin-push grid with the re-anchor (rows/grid2d_v4b_reanchor, design page folder)
against those without it: the grid's validation worlds on seeds 3-5, both from the same code (port_check), and the
whole grid on seeds 0-2, whose rows without the re-anchor predate the CUDA-graph port. Writes
rows/grid2d_v4b_reanchor/summary.json and media/20261030-reanchor_grid.png.
"""
from __future__ import annotations

import os

os.environ.setdefault("SCALE_HAND", "cad_v2")
os.environ.setdefault("MUJOCO_GL", "egl")

import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import hom_code_trace as CT  # noqa: E402

T, S, H, HG, HR = CT.T, CT.S, CT.H, CT.HG, CT.HR
FINGERS = HG.FINGERS
VARIANTS = ("base", "reanchor")
VIEWS = ((340.0, 12.0), (90.0, -12.0), (225.0, 25.0), (45.0, 50.0))   # (azimuth, elevation)
GHOST = {"theta_ref": (0.95, 0.55, 0.0, 0.35), "q_cmd": (0.0, 0.75, 0.95, 0.35)}
OUT_ROWS = CT.ROWS / "20261030-reanchor_check.json"


class Snap:
    """Stands in for hom_tasks.Film: keeps the plant's qpos at every frame call (every fourth period)."""

    def __init__(self):
        self.qpos = []

    def frame(self, plant, title, line):
        self.qpos.append((float(plant.d.time), plant.d.qpos.copy()))
        return np.zeros((2, 2, 3), np.uint8)


def yaw_pitch(axis, psi):
    """The cylinder's angle about palm x (hom_tasks.pinch_angle) and about the vertical (deg), in the frame where the
    pinch runs along x (the toy's psi undone)."""
    Rz = np.array([[math.cos(psi), math.sin(psi), 0.0], [-math.sin(psi), math.cos(psi), 0.0], [0.0, 0.0, 1.0]])
    a = Rz @ np.asarray(axis, float)
    return T.pinch_angle(a), math.degrees(math.atan2(a[0], -a[1]))


def run(var):
    import mujoco
    task, d_mm, call, grasp = CT.EXAMPLES["pinpush"]
    xs, ys, g, mxy, setup = CT.setup_of(task, d_mm, call, grasp)
    scene = HG.Scene(T.SHAPE[task], d_mm)
    bm, q_open = scene.prepare(mxy, setup)
    plant = T.Plant(scene.pads_path, setup)
    m_pt = mujoco.MjModel.from_xml_path(str(scene.pt_path))
    for m in (plant.m, m_pt):
        for f in FINGERS:
            m.body_pos[m.body(f"{f}_mount").id] = bm[f]
    S.set_palm(m_pt, setup["palm"])
    fs, os_ = T.sr2_specs(setup)
    mirror = H.Mirror(m_pt, fs, os_)
    snap = Snap()
    CT.Recorder.log = []
    real = T.H.HomController
    T.H.HomController = CT.Recorder
    over, _noise = T.VARIANTS[var]
    try:
        res, trace, _ = T.rollout(task, setup, plant, mirror, q_open, 0, T.PRM, replace(H.HomParams(dt=T.DT_C), **over),
                                  snap)
    finally:
        T.H.HomController = real
    psi = float(setup.get("psi", 0.0))
    per = []
    for rec, row in zip(CT.Recorder.log, trace):
        i = rec["info"]["middle"]
        pitch, yaw = yaw_pitch(row["axis"], psi)
        per.append({"t": row["t"], "phase": row["phase"], "F": {f: row["F"][f] for f in FINGERS},
                    "dist_mm": round(1e3 * i["dist"], 3), "gap_mm": round(1e3 * i["gap"], 3),
                    "reanchored": bool(i.get("reanchored")), "theta": np.degrees(i["theta"]).round(3).tolist(),
                    "theta_ref": np.degrees(i["theta_ref"]).round(3).tolist(),
                    "q_cmd": np.degrees(i["q_cmd"]).round(3).tolist(), "pitch_deg": round(pitch, 3),
                    "yaw_deg": round(yaw, 3), "cmd_deg": row.get("theta_ref_deg"),
                    "off_gap": np.degrees(i["off_gap"]).round(3).tolist(),
                    "off_force": np.degrees(i["off_force"]).round(3).tolist()})
    log = [{"theta_ref": {f: r["info"][f]["theta_ref"].copy() for f in FINGERS},
            "q_cmd": {f: np.asarray(r["info"][f]["q_cmd"]).copy() for f in FINGERS}} for r in CT.Recorder.log]
    print(f"{var}: held {res.get('held')}, pinch-angle RMSE {res.get('theta_rmse_deg')} deg, middle F_mean "
          f"{res.get('F_mean', {}).get('middle')} N, re-anchored periods {sum(p['reanchored'] for p in per)}", flush=True)
    return {"res": res, "periods": per, "x_sep_mm": xs, "y_sep_mm": ys}, plant, snap, log


def instants(per):
    """The decisive instants of the base run: the forces switching on, the middle finger's servo gap first reaching the
    cut, the end of the squeeze, the up stroke and the down stroke."""
    cut = 1e3 * H.HomParams().gap_max
    t_cut = next((p["t"] for p in per if p["t"] > T.T_CLOSE and p["gap_mm"] >= cut - 1e-3), None)
    out = [("forces on", T.T_CLOSE + 0.02)]
    if t_cut is not None:
        out.append(("middle's gap at the cut", t_cut))
    out += [("end of squeeze", T.T_LIFT - 0.04), ("up stroke", 2.5), ("down stroke", 4.0)]
    return out


def ghost(scn, m, d, bodies, rgba):
    """Append the geoms of `bodies` at d's pose to the scene, coloured rgba (a see-through copy of a finger)."""
    import mujoco
    grp = m.geom_group.copy()
    m.geom_group[:] = 0
    m.geom_group[np.isin(m.geom_bodyid, bodies)] = 5
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[5] = 1
    n0 = scn.ngeom
    try:
        mujoco.mjv_addGeoms(m, d, opt, mujoco.MjvPerturb(), int(mujoco.mjtCatBit.mjCAT_ALL), scn)
    finally:
        m.geom_group[:] = grp
    for j in range(n0, scn.ngeom):
        scn.geoms[j].rgba[:] = rgba


def keyframes(var, plant, snap, log, per, times, out=None, w=400, h=330, views=VIEWS):
    """Rows of keyframes at the given (label, time) instants, each from the given views, with the middle finger's joint
    reference and servo target drawn as see-through copies; written to out when given, else returned."""
    import mujoco
    from PIL import Image
    m = plant.m
    for gid in range(m.ngeom):                     # finger links see-through, as in the walkthrough's films
        bn = m.body(m.geom_bodyid[gid]).name
        if any(bn.startswith(f + "_") for f in FINGERS) and "_pad" not in (m.geom(gid).name or ""):
            m.geom_rgba[gid, 3] = min(float(m.geom_rgba[gid, 3]), 0.45)
    m.geom_rgba[plant.post, 3] = 0.15
    mid = [b for b in range(m.nbody) if m.body(b).name.startswith("middle_") and m.body(b).name != "middle_mount"]
    ren = mujoco.Renderer(m, h, w, max_geom=20000)
    d, dg = mujoco.MjData(m), mujoco.MjData(m)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.0, 0.0, S.Z_OBJ + 0.005]
    cam.distance = 0.15
    rows = []
    try:
        for label, t in times:
            k = int(np.argmin([abs(tt - t) for tt, _ in snap.qpos]))
            tk, q = snap.qpos[k]
            i = max(0, min(int(round(tk / T.DT_C)) - 1, len(log) - 1))
            d.qpos[:] = q
            mujoco.mj_forward(m, d)
            ims = []
            for az, el in views:
                cam.azimuth, cam.elevation = az, el
                ren.update_scene(d, cam)
                for key, rgba in GHOST.items():
                    dg.qpos[:] = q
                    dg.qpos[plant.qadr["middle"]] = log[i][key]["middle"]
                    mujoco.mj_kinematics(m, dg)
                    ghost(ren.scene, m, dg, mid, rgba)
                ims.append(ren.render().copy())
            p = per[i]
            line = (f"{var}, t = {tk:.2f} s, {label}: middle pad {p['F']['middle']:.2f} N, gap {p['gap_mm']:.1f} mm, "
                    f"MCP {p['theta'][1]:+.1f} / ref {p['theta_ref'][1]:+.1f} / target {p['q_cmd'][1]:+.1f} deg, "
                    f"yaw {p['yaw_deg']:+.1f} deg")
            rows.append(HR.annotate(np.concatenate(ims, 1), line, size=15))
    finally:
        ren.close()
    if out is None:
        return rows
    Image.fromarray(np.concatenate(rows, 0)).save(out)
    print("wrote", out.relative_to(ROOT), flush=True)
    return rows


def trace_figure(data, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 13})
    fig, ax = plt.subplots(5, 1, figsize=(9.5, 13), sharex=True)
    col = {"base": "0.45", "reanchor": "tab:blue"}
    t_task = T.T_LIFT + 0.4
    for var in VARIANTS:
        per = data[var]["periods"]
        t = np.array([p["t"] for p in per])
        lab = {"base": "without the re-anchor", "reanchor": "with the re-anchor"}[var]
        ax[0].plot(t, [p["F"]["middle"] for p in per], color=col[var], lw=1.4, label=lab)
        ax[1].plot(t, [p["gap_mm"] for p in per], color=col[var], lw=1.4)
        ra = [p["t"] for p in per if p["reanchored"]]
        if ra:
            ax[1].plot(ra, [6.4] * len(ra), "v", color=col[var], ms=7, label="re-anchored")
        ax[2].plot(t, [p["theta"][1] for p in per], color=col[var], lw=1.6)
        ax[2].plot(t, [p["theta_ref"][1] for p in per], color=col[var], lw=1.0, ls="--")
        ax[3].plot(t, [p["yaw_deg"] - per[0]["yaw_deg"] for p in per], color=col[var], lw=1.4)
        k0 = next(k for k, p in enumerate(per) if p["t"] >= t_task - 1e-9)
        tt = [p["t"] for p in per[k0:]]
        ax[4].plot(tt, [p["pitch_deg"] - per[k0]["pitch_deg"] for p in per[k0:]], color=col[var], lw=1.4)
    per = data["base"]["periods"]
    cmd = [(p["t"], p["cmd_deg"]) for p in per if p["cmd_deg"] is not None and p["t"] >= t_task - 1e-9]
    ax[4].plot([c[0] for c in cmd], [c[1] for c in cmd], color="k", lw=0.9, label="command")
    ax[0].axhline(T.PRM.get("f_drive", 1.5), color="k", lw=0.8, ls="--")
    ax[1].axhline(1e3 * H.HomParams().gap_max, color="k", lw=0.8, ls=":")
    ax[0].set_ylabel("Middle pad\nforce (N)")
    ax[1].set_ylabel("Middle servo\ngap (mm)")
    ax[2].set_ylabel("Middle MCP\n(deg)")
    ax[3].set_ylabel("Cylinder yaw\n(deg)")
    ax[4].set_ylabel("Angle about the\npinch axis (deg)")
    ax[4].set_xlabel("Time (s)")
    ax[0].legend(loc="upper right", fontsize=11)
    ax[1].legend(loc="lower right", fontsize=11)
    ax[4].legend(loc="lower left", fontsize=11)
    ax[2].text(0.01, 0.9, "solid: measured; dashed: joint reference", transform=ax[2].transAxes, fontsize=11)
    for a in ax:
        for x in (T.T_CLOSE, T.T_LIFT, T.T_LIFT + 0.4):
            a.axvline(x, color="0.8", lw=0.8, zorder=0)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print("wrote", out.relative_to(ROOT), flush=True)


FPR = ROOT / "docs/experiments/20261029-foundational_pose_review/rows"
GRID_A, GRID_B = FPR / "grid2d_v4b", FPR / "grid2d_v4b_reanchor"


def load_rows(path):
    """{(call, grasp, seed): row} of a GPU row file."""
    out = {}
    for x in path.read_text().splitlines():
        if x.strip():
            r = json.loads(x)
            out[(int(r["call"]), int(r.get("grasp", 0)), int(r["seed"]))] = r
    return out


def compare(a, b):
    """Matched worlds without (a) and with (b) the re-anchor: held outcomes, scores, the middle finger's mean force over
    the task, the axis' largest change during the task, and the re-anchored periods."""
    import hom_landscape as L
    from scipy.stats import wilcoxon
    keys = sorted(set(a) & set(b))
    sa = np.array([L.rollout_score("pinpush", a[k]) for k in keys])
    sb = np.array([L.rollout_score("pinpush", b[k]) for k in keys])
    ha = np.array([bool(a[k].get("held")) for k in keys])
    hb = np.array([bool(b[k].get("held")) for k in keys])
    both = ha & hb
    fa = np.array([a[k]["F_mean"]["middle"] for k in keys])[both]
    fb = np.array([b[k]["F_mean"]["middle"] for k in keys])[both]
    oa = np.array([a[k].get("off_axis_max_deg") or 0.0 for k in keys])[both]
    ob = np.array([b[k].get("off_axis_max_deg") or 0.0 for k in keys])[both]
    nra = np.array([(b[k].get("reanchors") or {}).get("middle", 0) for k in keys])
    nro = np.array([sum((b[k].get("reanchors") or {}).get(f, 0) for f in ("thumb", "index")) for k in keys])
    d = sb - sa
    return {"worlds": len(keys), "held_a": float(ha.mean()), "held_b": float(hb.mean()), "same_held": float(np.mean(ha == hb)),
            "score_a_median": float(np.median(sa)), "score_b_median": float(np.median(sb)),
            "score_change_median": float(np.median(d)), "better": float(np.mean(d > 0.02)), "worse": float(np.mean(d < -0.02)),
            "wilcoxon_p": float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0,
            "held_both": int(both.sum()), "F_middle_a_median": float(np.median(fa)), "F_middle_b_median": float(np.median(fb)),
            "F_middle_b_over_1N": float(np.mean(fb > 1.0)), "off_axis_a_median": float(np.median(oa)),
            "off_axis_b_median": float(np.median(ob)), "reanchored_middle": float(np.mean(nra > 0)),
            "reanchors_middle_median": float(np.median(nra[nra > 0])) if np.any(nra > 0) else 0.0,
            "reanchored_pinch": float(np.mean(nro > 0)), "scores": [sa.round(4).tolist(), sb.round(4).tolist()],
            "F_middle": [fa.round(3).tolist(), fb.round(3).tolist()]}


def layout_best(rows):
    """Per grid call: the best grasp's score (max over grasps of the mean over seeds) and the foundational pose's."""
    import hom_landscape as L
    sc = {}
    for (c, g, _s), r in rows.items():
        sc.setdefault(c, {}).setdefault(g, []).append(L.rollout_score("pinpush", r))
    return ({c: max(float(np.mean(v)) for v in gs.values()) for c, gs in sc.items()},
            {c: float(np.mean(gs[0])) for c, gs in sc.items() if 0 in gs})


def cmd_grid():
    import hom_landscape as L
    import fp_review_figures as FR
    val = compare(load_rows(GRID_A / "port_check" / "val_pinpush_d25.jsonl"), load_rows(GRID_B / "val_pinpush_d25.jsonl"))
    A, B = load_rows(GRID_A / "pinpush_d25.jsonl"), load_rows(GRID_B / "pinpush_d25.jsonl")
    full = compare(A, B)
    ba, fa = layout_best(A)
    bb, fb = layout_best(B)
    cs = sorted(set(ba) & set(bb))
    poses = L.grid_poses("pinpush", 25.0)
    xy = {c: (poses[c][0], poses[c][1]) for c in cs}
    out = {"validation_s345_same_code": {k: v for k, v in val.items() if k not in ("scores", "F_middle")},
           "grid_s012": {k: v for k, v in full.items() if k not in ("scores", "F_middle")},
           "layouts": len(cs), "best16_median": [float(np.median([ba[c] for c in cs])), float(np.median([bb[c] for c in cs]))],
           "fp_median": [float(np.median([fa[c] for c in cs if c in fb])), float(np.median([fb[c] for c in cs if c in fa]))],
           "best16_spearman": FR.spearman(np.array([ba[c] for c in cs]), np.array([bb[c] for c in cs])),
           "best_layout": [list(xy[max(cs, key=ba.get)]), list(xy[max(cs, key=bb.get)])],
           "when": time.strftime("%Y-%m-%d %H:%M")}
    GRID_B.mkdir(parents=True, exist_ok=True)
    (GRID_B / "summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1), flush=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 13})
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.2))
    fa_, fb_ = val["F_middle"]
    ax[0].plot(fa_, fb_, ".", ms=3, color="tab:blue", alpha=0.5)
    m = max(max(fa_), max(fb_))
    ax[0].plot([0, m], [0, m], "k:", lw=1)
    ax[0].axhline(1.5, color="0.5", lw=0.8, ls="--")
    ax[0].set_xlabel("Middle pad force, without (N)")
    ax[0].set_ylabel("Middle pad force, with the re-anchor (N)")
    ax[0].set_title("Validation worlds held in both", fontsize=13)
    sa_, sb_ = val["scores"]
    h = ax[1].hist2d(sa_, sb_, bins=24, range=[[0, 1], [0, 1]], cmap="Greys", cmin=1)
    ax[1].plot([0, 1], [0, 1], "r:", lw=1)
    fig.colorbar(h[3], ax=ax[1], label="Worlds")
    ax[1].set_xlabel("Pin-push score, without")
    ax[1].set_ylabel("Pin-push score, with the re-anchor")
    ax[1].set_title("All validation worlds (0: dropped)", fontsize=13)
    xs = sorted({v[0] for v in xy.values()})
    ys = sorted({v[1] for v in xy.values()})
    Z = np.full((len(ys), len(xs)), np.nan)
    for c in cs:
        Z[ys.index(xy[c][1]), xs.index(xy[c][0])] = bb[c] - ba[c]
    im = ax[2].imshow(Z, origin="lower", cmap="RdBu", vmin=-0.5, vmax=0.5,
                      extent=[xs[0] - 2.5, xs[-1] + 2.5, ys[0] - 2.5, ys[-1] + 2.5], aspect="auto")
    fig.colorbar(im, ax=ax[2], label="Change of the best-of-16 score")
    ax[2].set_xlabel("Thumb-to-pair distance (mm)")
    ax[2].set_ylabel("Index-to-middle distance (mm)")
    ax[2].set_title("Grid, seeds 0-2", fontsize=13)
    fig.tight_layout()
    path = CT.MEDIA / "20261030-reanchor_grid.png"
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print("wrote", path.relative_to(ROOT), flush=True)
    return 0


def cmd_cpu():
    data, keep = {}, {}
    for var in VARIANTS:
        data[var], plant, snap, log = run(var)
        keep[var] = (plant, snap, log)
    times = instants(data["base"]["periods"])
    CT.MEDIA.mkdir(parents=True, exist_ok=True)
    for var in VARIANTS:
        plant, snap, log = keep[var]
        keyframes(var, plant, snap, log, data[var]["periods"], times,
                  CT.MEDIA / f"20261030-reanchor_keyframes_{var}.png")
    from PIL import Image
    rows = []
    for inst in [x for x in times if x[0] in ("end of squeeze", "up stroke")]:
        for var in VARIANTS:                   # the page's figure: both variants at the decisive instants, three views
            plant, snap, log = keep[var]
            rows += keyframes(var, plant, snap, log, data[var]["periods"], [inst], views=VIEWS[1:])
    path = CT.MEDIA / "20261030-reanchor_keyframes.jpg"
    Image.fromarray(np.concatenate(rows, 0)).convert("RGB").save(path, quality=88)
    print("wrote", path.relative_to(ROOT), flush=True)
    trace_figure(data, CT.MEDIA / "20261030-reanchor_trace.png")
    OUT_ROWS.write_text(json.dumps({"instants": times, "when": time.strftime("%Y-%m-%d %H:%M"), **data},
                                   default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    print("wrote", OUT_ROWS.relative_to(ROOT), flush=True)
    return 0


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    return {"cpu": cmd_cpu, "grid": cmd_grid}[cmd]()


if __name__ == "__main__":
    sys.exit(main())
