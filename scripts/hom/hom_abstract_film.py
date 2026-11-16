#!/usr/bin/env python3
"""A HOM rollout drawn on the abstract SR2 hand of the talk's diagrams (owner, 2026-11-11: "can we do one that's just an
abstracted form of the sr 2 hand, like the ones we've been using in the slide decks?"). The hand is the drawing of
codesign_tikz.hand(): the gantry travel boxes as pale planes, each finger as a grey polyline from its mount through the
yaw, MCP and PIP joints (white rings, purple edge) to its pad (grey dot), seen from the drawing's view (30 deg azimuth,
30 deg elevation). The joint positions, the object's pose, the pad forces and their centres of pressure come from one CPU
rollout under HOM (hom_tasks.rollout on the CPU plant, every contact), sampled every fourth control period, so the film
plays in real time at 25 frames per second.

  record  runs the rollout with a recorder in place of the MuJoCo film and writes
          docs/experiments/20261110-codesign_formulation/rows/20261111-abstract_<task>.json
  render  draws the frames and writes media/20261111-abstract_<task>.mp4 (and .jpg)

    HOS_PLANT=servo HOS_SCENES=logs/20261102-hom_servo_plant/scenes HOM_ROWS_TAG=_sv25 \\
        .venv/bin/python scripts/hom/hom_abstract_film.py record --task twist --d 40 --layout 105 70
    .venv/bin/python scripts/hom/hom_abstract_film.py render --task twist
    ... record --task twist --d 40 --layout 70 110 --tag 70_110            # a second layout (rows/20261115-...)
    .venv/bin/python scripts/hom/hom_abstract_film.py reconfig --task twist --tag 70_110   # A, reconfiguration, B

The ground-clear grasps run on the SR2 hand on 2026-11-16, paced to the hand's film (owner, 2026-11-16: "update the
abstract videos to match the ground-clearance trajectories we actually ran ... add pauses to sync up the motions with the
real stitched video"); the zoom then puts the drawing behind the hand's film by its extra length (owner: "i can live with
some de-sync on the one slide that has them side by side"):
    ... record --task twist --d 40 --layout 70 110 --grid grid2dgr --grasp 0 --pads --tag 70_110_gr_pads
    ... record --task twist --d 40 --layout 105 70 --grid grid2dgr --grasp 13 --pads --tag 105_70_gr_pads
    .venv/bin/python scripts/hom/hom_abstract_film.py timeline --clips DIR     # DIR: the two trial clips from corvus
    .venv/bin/python scripts/hom/hom_abstract_film.py reconfig --tag-a 70_110_gr_pads --tag 105_70_gr_pads --zoom-at 4.8 \\
        --sync docs/experiments/20261110-codesign_formulation/rows/20261116-abstract_sync_ground_clear.json \\
        --out-stem 20261116-abstract_twist_ground_clear       # also the drawing beside the hand's film (abstract_vs_hand)
"""
from __future__ import annotations

import os

os.environ.setdefault("SCALE_HAND", "cad_v2")

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path

OUT = ROOT / "docs/experiments/20261110-codesign_formulation"
CHAIN = ("mount", "yaw_frame", "mcp_frame", "len_frame", "pip_frame")     # bodies of each finger, base to PIP
NAME = {"twist": "Twist", "roll": "Roll", "press": "Press", "pinpush": "Pin-push"}
OBJ = {"twist": "sphere", "roll": "sphere", "press": "cylinder", "pinpush": "cylinder"}


class Recorder:
    """Takes the place of hom_films.Film in hom_tasks.rollout: at each filmed period it returns the geometry of the hand
    and the object in the palm frame (mm) instead of an image."""

    def __init__(self, plant, fingers, pads=False):
        m = plant.m
        self.plant, self.fingers, self.with_pads = plant, fingers, pads
        self.palm = m.body("palm_pose").id
        self.chain = {f: [m.body(f"{f}_{k}").id for k in CHAIN] for f in fingers}
        self.tip = {f: m.body(f"{f}_tip").id for f in fingers}
        self.pads = {f: np.flatnonzero((m.geom_bodyid == m.body(f"{f}_tip").id)
                                       & np.array(["_pad" in (m.geom(g).name or "") for g in range(m.ngeom)]))
                     for f in fingers}
        self.pad_of = {int(g): (f, k) for f in fingers for k, g in enumerate(self.pads[f])}
        self.obj_geoms = set(np.flatnonzero(m.geom_bodyid == plant.tb).tolist())

    def pads_local(self):
        """Each pad's centre in its tip body's frame (mm) and the pads' radius (mm)."""
        m = self.plant.m
        return ({f: (1e3 * m.geom_pos[self.pads[f]]).round(4).tolist() for f in self.fingers},
                float(1e3 * m.geom_size[self.pads[self.fingers[0]][0]][0]))

    def frame(self, plant, title, line):
        import mujoco
        d = plant.d
        Rp, op = d.xmat[self.palm].reshape(3, 3), d.xpos[self.palm]
        to = (lambda p: (1e3 * (Rp.T @ (np.asarray(p) - op))).round(3).tolist())
        out = {"t": round(float(d.time), 4),
               "chain": {f: [to(d.xpos[b]) for b in self.chain[f]] + [to(d.geom_xpos[self.pads[f]].mean(axis=0))]
                         for f in self.fingers},
               "obj_p": to(d.xpos[plant.tb]),
               "obj_R": (Rp.T @ d.xmat[plant.tb].reshape(3, 3)).round(6).tolist()}
        if self.with_pads:                         # every tip's pose and each pad's normal force on the object (N)
            out["tips"] = {f: {"p": to(d.xpos[self.tip[f]]),
                               "R": (Rp.T @ d.xmat[self.tip[f]].reshape(3, 3)).round(6).tolist()} for f in self.fingers}
            F = {f: {} for f in self.fingers}
            f6 = np.zeros(6)
            for i in range(d.ncon):
                c = d.contact[i]
                g1, g2 = int(c.geom1), int(c.geom2)
                pad = self.pad_of.get(g1) if g2 in self.obj_geoms else (self.pad_of.get(g2) if g1 in self.obj_geoms else None)
                if pad is None:
                    continue
                mujoco.mj_contactForce(plant.m, d, i, f6)
                if f6[0] > 1e-4:
                    F[pad[0]][str(pad[1])] = round(F[pad[0]].get(str(pad[1]), 0.0) + float(f6[0]), 4)
            out["pad_F"] = F
        return out

    def close(self):
        pass


def best_track_grasp(task, d_mm, call, rows_dir):
    """The layout's grasp with the highest mean tracking score over seeds 0-2 in the rerun grid."""
    import hom_landscape as L
    import hom_rows
    sc = {}
    for x in (L.ROWS / rows_dir / f"{task}_d{d_mm:g}.jsonl").read_text().splitlines():
        if x.strip():
            r = hom_rows.rescale_metrics(json.loads(x))
            if r.get("status", "ok") == "ok" and int(r["call"]) == call:
                sc.setdefault(int(r.get("grasp", 0)), []).append(L.track_score(task, r))
    return max(sc, key=lambda g: np.mean(sc[g])), {g: round(float(np.mean(v)), 3) for g, v in sc.items()}


def record(a):
    import mujoco
    import hom_films as HF
    import hom_gpu as HG
    import hom_landscape as L
    T, S, H = HG.T, HG.S, HG.H
    L.GRID_SRC = a.grid
    call = HF.call_at(a.task, a.d, *a.layout)
    grasp, by = (a.grasp, {}) if a.grasp is not None else best_track_grasp(a.task, a.d, call, a.rows)
    _key, mxy, setup = HF.find(a.task, f"{a.grid}_size", a.d, call, grasp)
    scene = HG.Scene(T.SHAPE[a.task], a.d)
    bm, q_open = scene.prepare(mxy, setup)
    plant = T.Plant(scene.pads_path, setup)
    m_pt = mujoco.MjModel.from_xml_path(str(scene.pt_path))
    for m in (plant.m, m_pt):
        for f in HG.FINGERS:
            m.body_pos[m.body(f"{f}_mount").id] = bm[f]
    S.set_palm(m_pt, setup["palm"])
    fs, os_ = T.sr2_specs(setup)
    mirror = H.Mirror(m_pt, fs, os_)
    rec = Recorder(plant, HG.FINGERS, pads=a.pads)
    over, noise = T.VARIANTS["base"]
    res, rows, frames = T.rollout(a.task, setup, plant, mirror, q_open, a.seed, T.PRM,
                                  replace(H.HomParams(dt=T.DT_C), **over), rec, noise, 0.0)
    keep = ("t", "phase", "w_dps", "w_ref_dps", "theta_ref_deg", "F", "cop_obj_mm", "axis")
    out = {"task": a.task, "d_mm": a.d, "layout": list(a.layout), "call": call, "grasp": int(grasp), "seed": a.seed,
           "grasp_track_scores": by, "plant": os.environ.get("HOS_PLANT", "work"), "r_mm": 1e3 * float(setup["r"]),
           "track": round(float(L.track_score(a.task, res)), 3), "earlier": round(float(L.rollout_score(a.task, res)), 3),
           "held": bool(res.get("held")), "track_err_rms_deg": res.get("track_err_rms_deg"),
           "track_ref_rms_deg": res.get("track_ref_rms_deg"), "contact_all_frac": res.get("contact_all_frac"),
           "dt_row": T.DT_C, "frame_every": 4, "frames": frames,
           **({"pads_L_mm": rec.pads_local()[0], "pad_r_mm": rec.pads_local()[1]} if a.pads else {}),
           "rows": [{k: r[k] for k in keep if k in r} for r in rows]}
    (OUT / "rows").mkdir(parents=True, exist_ok=True)
    path = OUT / "rows" / (f"20261111-abstract_{a.task}.json" if not a.tag else f"20261115-abstract_{a.task}_{a.tag}.json")
    path.write_text(json.dumps(out, separators=(",", ":")))
    print(f"{path.relative_to(ROOT)}: {len(frames)} frames, {len(rows)} periods, tracking {out['track']}, "
          f"earlier {out['earlier']}, held {out['held']}, grasp {grasp}", flush=True)
    return 0


# ------------------------------------------------------------------------------------------------------------ drawing
INK, GREY, PALE, NEW, ACH = "#1F2937", "#9CA3AF", "#F3F4F6", "#7C3AED", "#EA580C"
LINK = "#B0B5BF"                                   # TikZ grey!80


def view():
    import codesign_tikz as CT
    from sr2hand.geometry.rotations import rot_x, rot_z
    az, el = np.radians(CT.HAND_VIEW)
    return rot_x(el) @ rot_z(az)


def boxes():
    import fp_sr2_toy as FT
    out = []
    for i in range(3):
        (x0, y0), (x1, y1) = FT.LO6[2 * i:2 * i + 2], FT.HI6[2 * i:2 * i + 2]
        out.append(np.array([[x0, y0, 0.0], [x1, y0, 0.0], [x1, y1, 0.0], [x0, y1, 0.0]]))
    return out


def yaw_angles(rec):
    """Commanded and achieved rotation about the vertical (deg) at every period, integrated from the rows' rates."""
    h = rec["dt_row"]
    w = np.array([r["w_dps"][2] for r in rec["rows"]], float)
    wr = np.array([(r.get("w_ref_dps") or [0.0, 0.0, 0.0])[2] for r in rec["rows"]], float)
    return np.cumsum(wr) * h, np.cumsum(w) * h


def render(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyArrowPatch, Polygon
    plt.rcParams.update({"font.family": "Nimbus Sans", "mathtext.fontset": "cm"})
    rec = json.loads((OUT / "rows" / f"20261111-abstract_{a.task}.json").read_text())
    V = view()

    def uv(P):
        Q = np.asarray(P, float) @ V.T
        return Q[..., 0], Q[..., 2], Q[..., 1]          # right, up, depth (larger is farther)

    frames, rows = rec["frames"], rec["rows"]
    every = rec["frame_every"]
    r = rec["r_mm"]
    cmd, ach = yaw_angles(rec)
    t_rows = np.array([q["t"] for q in rows])
    R0 = np.array(frames[0]["obj_R"])                    # the object's axes at the start: the most vertical one is the
    kv = int(np.argmax(np.abs(R0[2])))                   # pole of the drawn equator, the most horizontal one carries the dot
    kh = min((k for k in range(3) if k != kv), key=lambda k: abs(R0[2, k]))
    k3 = 3 - kv - kh
    # fixed limits from every frame, so the drawing does not move
    pts = [p for fr in frames for ch in fr["chain"].values() for p in ch] + [fr["obj_p"] for fr in frames]
    U, Vv, _ = uv(np.array(pts + [p for b in boxes() for p in b]))
    lim = (U.min() - r - 12, U.max() + r + 12, Vv.min() - r - 12, Vv.max() + 12)
    tmp = Path(tempfile.mkdtemp(prefix="abstract_", dir=os.environ.get("TMPDIR", "/tmp")))
    for k, fr in enumerate(frames):
        i = min(k * every, len(rows) - 1)
        row = rows[i]
        fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
        ax = fig.add_axes([0.02, 0.04, 0.60, 0.80])
        ax.set_xlim(lim[0], lim[1])
        ax.set_ylim(lim[2], lim[3])
        ax.set_aspect("equal")
        ax.axis("off")
        mm = ax.transData.transform([[0, 0], [1, 0]])
        pt_per_mm = (mm[1, 0] - mm[0, 0]) * 72.0 / fig.dpi
        for b in boxes():
            u, v, _ = uv(b)
            ax.add_patch(Polygon(np.c_[u, v], closed=True, fc=PALE, ec=GREY, lw=0.6 * pt_per_mm, zorder=1))
        R = np.array(fr["obj_R"])
        c = np.array(fr["obj_p"])
        items = []
        for f, ch in fr["chain"].items():
            ch = np.array(ch)
            items.append((float(uv(ch)[2].mean()), "finger", f, ch))
        items.append((float(uv(c)[2]), "object", None, None))
        z = 2
        for _dep, kind, f, ch in sorted(items, key=lambda x: -x[0]):           # farthest first
            if kind == "finger":
                u, v, _ = uv(ch)
                ax.plot(u, v, color=LINK, lw=4.6 * pt_per_mm, solid_capstyle="round", solid_joinstyle="round", zorder=z)
                ax.add_patch(Circle((u[-1], v[-1]), 4.4, fc=LINK, ec="none", zorder=z + 0.1))
                for j in (1, 2, 4):                                          # yaw, MCP, PIP
                    ax.add_patch(Circle((u[j], v[j]), 3.6, fc="white", ec=NEW, lw=1.7 * pt_per_mm, zorder=z + 0.2))
                F = float(row["F"].get(f) or 0.0)
                cop = row["cop_obj_mm"].get(f)
                if cop is not None and F > 0.05:
                    p = c + R @ np.array(cop)
                    n = (c - p) / max(np.linalg.norm(c - p), 1e-9)
                    q0, q1 = uv(p - 7.0 * F * n), uv(p)
                    ax.add_patch(FancyArrowPatch((q0[0], q0[1]), (q1[0], q1[1]), arrowstyle="-|>", color=NEW,
                                                 lw=1.5 * pt_per_mm, mutation_scale=5.5 * pt_per_mm, zorder=z + 0.3))
            else:
                u0, v0, d0 = uv(c)
                ax.add_patch(Circle((u0, v0), r, fc="#E5E7EB", ec=GREY, lw=0.7 * pt_per_mm, zorder=z))
                tt = np.linspace(0, 2 * np.pi, 181)
                for e0, e1 in ((kh, k3), (kh, kv)):                          # the object's equator and one meridian
                    P = c + r * (np.outer(np.cos(tt), R[:, e0]) + np.outer(np.sin(tt), R[:, e1]))
                    u, v, dd = uv(P)
                    front = dd <= d0
                    for mask, sty, col in ((front, "-", INK), (~front, ":", GREY)):
                        uu, vv = np.where(mask, u, np.nan), np.where(mask, v, np.nan)
                        ax.plot(uu, vv, sty, color=col, lw=0.45 * pt_per_mm, zorder=z + 0.05)
                tip = c + r * R[:, kh]
                ut, vt, dt_ = uv(tip)
                ax.add_patch(Circle((ut, vt), 2.6, fc=ACH if dt_ <= d0 else "none", ec=ACH, lw=0.5 * pt_per_mm,
                                    zorder=z + 0.06))
            z += 1
        # the right panel: commanded against achieved rotation, the pad forces, the score
        ax2 = fig.add_axes([0.67, 0.50, 0.30, 0.30])
        ax2.plot(t_rows, cmd, "--", color=NEW, lw=2.4, label="commanded")
        ax2.plot(t_rows[:i + 1], ach[:i + 1], color=ACH, lw=2.8, label="object")
        ax2.axvline(t_rows[i], color=GREY, lw=1.0)
        ax2.set_xlim(t_rows[0], t_rows[-1])
        lo, hi = min(cmd.min(), ach.min()), max(cmd.max(), ach.max())
        ax2.set_ylim(lo - 0.1 * (hi - lo + 1), hi + 0.1 * (hi - lo + 1))
        ax2.set_xlabel("time (s)", fontsize=17)
        ax2.set_ylabel("turn about the vertical (deg)", fontsize=17)
        ax2.tick_params(labelsize=15)
        ax2.legend(fontsize=16, frameon=False, loc="upper left")
        for sp in ("top", "right"):
            ax2.spines[sp].set_visible(False)
        ax3 = fig.add_axes([0.67, 0.23, 0.30, 0.17])
        names = {"thumb": "thumb", "index": "index", "middle": "middle"}
        Fs = [float(row["F"].get(f) or 0.0) for f in names]
        ax3.barh(range(3), Fs, color=[NEW if v > 0.05 else GREY for v in Fs], height=0.55)
        ax3.set_yticks(range(3))
        ax3.set_yticklabels(list(names.values()), fontsize=16)
        ax3.invert_yaxis()
        ax3.set_xlim(0, 4.0)
        ax3.set_xlabel("pad force (N)", fontsize=17)
        ax3.tick_params(labelsize=15)
        for sp in ("top", "right"):
            ax3.spines[sp].set_visible(False)
        fig.text(0.02, 0.95, f"HOM on the SR2 hand: {NAME[rec['task']].lower()} of a {rec['d_mm']:g} mm {OBJ[rec['task']]}",
                 fontsize=30, color=INK, weight="bold", va="top")
        fig.text(0.02, 0.895, f"Layout ({rec['layout'][0]:.0f}, {rec['layout'][1]:.0f}) mm, CPU simulation, real time. "
                 f"Purple arrows: pad forces; orange dot: a point on the {OBJ[rec['task']]}.", fontsize=19, color="#4B5563",
                 va="top")
        fig.text(0.67, 0.86, f"t = {row['t']:.2f} s   {row['phase']}", fontsize=19, color=INK, va="top")
        fig.text(0.67, 0.075, f"Tracking score {rec['track']:.2f}: rotation followed "
                 f"{1 - rec['track_err_rms_deg'] / rec['track_ref_rms_deg']:.2f},\nall three pads loaded "
                 f"{100 * rec['contact_all_frac']:.0f} % of the task", fontsize=17, color="#4B5563", va="center",
                 linespacing=1.4)
        fig.savefig(tmp / f"f{k:05d}.png", facecolor="white")
        plt.close(fig)
    out = OUT / "media" / f"20261111-abstract_{rec['task']}.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "25", "-i", str(tmp / "f%05d.png"), "-c:v",
                    "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "slow", "-movflags", "+faststart",
                    str(out)], check=True)
    mid = sorted(tmp.glob("f*.png"))[int(0.6 * len(frames))]
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mid), "-q:v", "3", str(out.with_suffix(".jpg"))],
                   check=True)
    for p in tmp.glob("f*.png"):
        p.unlink()
    tmp.rmdir()
    print(f"{out.relative_to(ROOT)}: {len(frames)} frames", flush=True)
    return 0


MORPH = "#0F766E"                                  # teal: the morphology's degrees of freedom


def _ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def _camera(V, ctr, W):
    """Orthographic camera: rows of V are right, depth (away from the viewer) and up in the palm frame; ctr the 3-D point
    at the window's centre; W the half-width of the window (mm)."""
    return {"V": np.asarray(V, float), "ctr": np.asarray(ctr, float), "W": float(W)}


def frame_at(rec, kf):
    """The rollout's frame at the fractional index kf: positions interpolated linearly, rotations spherically, the pad
    forces those of the nearer frame. An integer kf returns the recorded frame."""
    fr = rec["frames"]
    k0 = int(np.floor(kf))
    u = float(kf) - k0
    if k0 >= len(fr) - 1 or u < 1e-6:
        return fr[min(k0, len(fr) - 1)]
    from scipy.spatial.transform import Rotation, Slerp
    p, q = fr[k0], fr[k0 + 1]

    def lin(x, y):
        return ((1 - u) * np.asarray(x, float) + u * np.asarray(y, float)).tolist()

    def sl(Ra, Rb):
        return Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([Ra, Rb])))([u]).as_matrix()[0].tolist()

    out = {"t": (1 - u) * p["t"] + u * q["t"], "chain": {f: lin(p["chain"][f], q["chain"][f]) for f in p["chain"]},
           "obj_p": lin(p["obj_p"], q["obj_p"]), "obj_R": sl(p["obj_R"], q["obj_R"])}
    if "tips" in p:
        out["tips"] = {f: {"p": lin(p["tips"][f]["p"], q["tips"][f]["p"]), "R": sl(p["tips"][f]["R"], q["tips"][f]["R"])}
                       for f in p["tips"]}
        out["pad_F"] = (p if u < 0.5 else q)["pad_F"]
    return out


def _look_side(V, n):
    """A view whose depth axis is perpendicular to the contact normal n and nearest the view V's: the contact seen
    side-on, the pads against the object's outline."""
    d0, u0 = V[1], V[2]
    dz = d0 - (d0 @ n) * n
    dz /= np.linalg.norm(dz)
    uz = u0 - (u0 @ dz) * dz
    uz /= np.linalg.norm(uz)
    rz = np.cross(dz, uz) if np.allclose(V[0], np.cross(V[1], V[2]), atol=1e-6) else np.cross(uz, dz)
    return np.array([rz, dz, uz])


def render_reconfig(a):
    """Rollout A, the hand reconfiguring to B's layout, rollout B (owner, 2026-11-15: "an alternative version of the
    abstract video that extends into a reconfiguration ... highlight the morphology DoFs during a morphology reconfig,
    and then we can show the updated morphology doing the same task ... it has control (fingers) and morphology (xy
    repositioning)"; then "zoom into a fingertip-sphere interaction, and show the sphere-packed surface"; on 11-16 "move
    the sphere pad zoom in to the controls after the morphology reconfiguration"). During B (--zoom-in A: during A) the
    camera turns and zooms onto one fingertip's contact, seen side-on, where its pad spheres press into the object,
    coloured by their normal force, with the rollout at a quarter of real time. Between the rollouts the object fades
    out, the fingers return to A's open pose, the mounts slide on their stages to B's layout (the travel boxes and each
    mount's x and y axes in teal), and B's object fades in. Subclips of the zoom and of the reconfiguration are written
    beside the film for the talk's dynamics and morphology sections."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import EllipseCollection
    from matplotlib.patches import Circle, FancyArrowPatch, Polygon
    from scipy.spatial.transform import Rotation, Slerp
    plt.rcParams.update({"font.family": "Nimbus Sans", "mathtext.fontset": "cm"})
    A = json.loads((OUT / "rows" / f"20261115-abstract_{a.task}_{a.tag_a}.json").read_text())
    B = json.loads((OUT / "rows" / f"20261115-abstract_{a.task}_{a.tag}.json").read_text())
    sync = json.loads(Path(a.sync).read_text()) if a.sync else None
    V0 = view()
    fps, r = (sync["fps"] if sync else 25), A["r_mm"]
    # ---- timeline: (segment, rollout frame index as float, blend w, zoom s)
    Zr = B if a.zoom_in == "B" else A                 # the rollout the zoom interrupts
    kz = int(np.argmin([abs(fr["t"] - a.zoom_at) for fr in Zr["frames"]]))
    slow = 0.25
    n_in, n_hold, n_out = int(1.8 * fps), int(3.2 * fps), int(1.6 * fps)

    def zoom_s(k):
        return (_ease((k + 1) / n_in) if k < n_in else
                (1.0 if k < n_in + n_hold else 1.0 - _ease((k - n_in - n_hold + 1) / n_out)))

    def rollout(seg, rec):
        """The rollout's frames, with the zoom from frame kz at a quarter of real time if it is the zoomed one; and the
        zoom's first index in them."""
        if rec is not Zr:
            return [(seg, float(k), 0.0, 0.0) for k in range(len(rec["frames"]))], None
        out = [(seg, float(k), 0.0, 0.0) for k in range(kz)]
        i_z, ka = len(out), float(kz)
        for k in range(n_in + n_hold + n_out):
            out.append(("Z", min(ka, len(rec["frames"]) - 1), 0.0, zoom_s(k)))
            ka += slow
        out += [(seg, float(k), 0.0, 0.0) for k in range(int(round(ka)), len(rec["frames"]))]
        return out, i_z

    held = {}                                         # sync: the frame each rollout last showed, held in pauses
    if sync is None:
        seq, i_za = rollout("A", A)
        i_zoom = i_za
        i_morph = len(seq)
        for k in range(int(0.8 * fps)):
            seq.append(("open", 0.0, _ease((k + 1) / (0.8 * fps)), 0.0))
        for k in range(int(0.6 * fps)):
            seq.append(("show", 0.0, _ease((k + 1) / (0.6 * fps)), 0.0))
        for k in range(int(2.8 * fps)):
            seq.append(("move", 0.0, _ease((k + 1) / (2.8 * fps)), 0.0))
        for k in range(int(0.8 * fps)):
            seq.append(("in", 0.0, _ease((k + 1) / (0.8 * fps)), 0.0))
        i_morph_end = len(seq)
        sb, i_zb = rollout("B", B)
        if i_zb is not None:
            i_zoom = i_morph_end + i_zb
        seq += sb
    else:
        # one frame per frame of the hand's film: its set-up steps pause the drawing, its reconfiguration is the
        # morphology segment, and each control frame shows the rollout at the controller time that frame shows on the
        # hand. The zoom plays its rollout at a quarter of real time and then skips the hand's frames it has covered,
        # so from there on the drawing runs behind the hand's film by the zoom's extra length.
        recs = {"A": A, "B": B}
        dtf = {k: v["dt_row"] * v["frame_every"] for k, v in recs.items()}
        seq, i_zoom, i_morph, i_morph_end = [], None, None, None
        for sg in sync["segments"]:
            n, kind, ro = sg["f1"] - sg["f0"], sg["kind"], sg["rollout"]
            if kind in ("open", "show", "move", "in"):
                i_morph = len(seq) if kind == "open" else i_morph
                seq += [(kind, 0.0, _ease((k + 1) / n), 0.0) for k in range(n)]
                i_morph_end = len(seq) if kind == "in" else i_morph_end
                continue
            rec = recs[ro]
            last = len(rec["frames"]) - 1
            if kind == "hold":
                seq += [(ro, held.get(ro, 0.0), 0.0, 0.0)] * n
                continue
            ts, j = sg["t"], 0
            while j < n:
                if rec is Zr and i_zoom is None and ts[j] >= a.zoom_at:
                    i_zoom, ka = len(seq), float(kz)
                    for k in range(n_in + n_hold + n_out):
                        seq.append(("Z", min(ka, last), 0.0, zoom_s(k)))
                        ka += slow / fps / dtf[ro]
                    held[ro] = min(ka, last)
                    t_res = rec["frames"][0]["t"] + ka * dtf[ro]
                    while j < n and ts[j] < t_res:
                        j += 1
                    continue
                held[ro] = min(max((ts[j] - rec["frames"][0]["t"]) / dtf[ro], 0.0), last)
                seq.append((ro, held[ro], 0.0, 0.0))
                j += 1
    chain = lambda fr: {f: np.array(c, float) for f, c in fr["chain"].items()}  # noqa: E731
    kA_end = held.get("A", len(A["frames"]) - 1)      # the frame A ends on: its last, or the last the hand's film showed
    A0, A1, B0 = chain(A["frames"][0]), chain(frame_at(A, kA_end)), chain(B["frames"][0])
    # ---- the full view's window, as a camera
    pts = [p for rec in (A, B) for fr in rec["frames"] for ch in fr["chain"].values() for p in ch]
    pts += [fr["obj_p"] for rec in (A, B) for fr in rec["frames"]] + [p for b in boxes() for p in b]
    Q = np.array(pts) @ V0.T
    lim = (Q[:, 0].min() - r - 14, Q[:, 0].max() + r + 14, Q[:, 2].min() - r - 12, Q[:, 2].max() + 16)
    aspect = (lim[3] - lim[2]) / (lim[1] - lim[0])
    full = _camera(V0, V0.T @ np.array([(lim[0] + lim[1]) / 2, 0.0, (lim[2] + lim[3]) / 2]), (lim[1] - lim[0]) / 2)
    # ---- the zoom's camera: the chosen finger's contact at the zoom frame, seen side-on
    fz = a.zoom_finger
    frz = Zr["frames"][kz]
    tz = frz["tips"][fz]
    padsL = np.array(Zr["pads_L_mm"][fz])
    rp = Zr["pad_r_mm"]
    Fz = frz["pad_F"][fz]
    P = np.array(tz["p"]) + padsL @ np.array(tz["R"]).T
    idx = np.array([int(i) for i in Fz])
    wts = np.array([Fz[str(i)] for i in idx])
    pc = (P[idx] * wts[:, None]).sum(0) / wts.sum()
    nrm = pc - np.array(frz["obj_p"])
    nrm /= np.linalg.norm(nrm)
    Vz = _look_side(V0, nrm)
    zoom = _camera(Vz, pc + 1.5 * nrm, a.zoom_halfwidth)
    slerp = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([V0, Vz])))
    track = []                                         # the zoomed finger's contact centroid and normal per frame
    last = (pc, nrm)
    for fr_ in Zr["frames"]:
        Fd = fr_.get("pad_F", {}).get(fz) or {}
        if Fd:
            t_ = fr_["tips"][fz]
            Pk = np.array(t_["p"]) + padsL @ np.array(t_["R"]).T
            ii = np.array([int(i) for i in Fd])
            ww = np.array([Fd[str(i)] for i in ii])
            c_ = (Pk[ii] * ww[:, None]).sum(0) / ww.sum()
            n_ = c_ - np.array(fr_["obj_p"])
            last = (c_, n_ / np.linalg.norm(n_))
        track.append(last)
    track_c = np.array([t[0] for t in track])
    track_n = np.array([t[1] for t in track])
    ker = np.ones(7) / 7.0                             # a 7-frame moving average, so the camera does not shake
    pad_ = lambda x: np.concatenate([np.repeat(x[:1], 3, 0), x, np.repeat(x[-1:], 3, 0)])  # noqa: E731
    track_c = np.stack([np.convolve(pad_(track_c)[:, j], ker, mode="valid") for j in range(3)], 1)
    track_n = np.stack([np.convolve(pad_(track_n)[:, j], ker, mode="valid") for j in range(3)], 1)
    track_n /= np.linalg.norm(track_n, axis=1, keepdims=True)

    def cam_at(sz, k=None):
        """The camera at zoom sz (0 the full view, 1 the zoom) on frame k of the zoomed rollout. The contact's place in
        the window moves straight from where it is in the full view to its place in the zoom, so it stays in sight while
        the window shrinks (a centre interpolated in space would leave it behind)."""
        if sz <= 0:
            return full
        tgt, nk = (pc, nrm) if k is None else (track_c[k], track_n[k])
        Vs = slerp([sz]).as_matrix()[0]
        Ws = float(np.exp((1 - sz) * np.log(full["W"]) + sz * np.log(zoom["W"])))
        q0 = V0 @ (tgt - full["ctr"]) / full["W"]       # (right, depth, up) over the half-width, in the full view
        q1 = Vz @ (-1.5 * nk) / zoom["W"]                # ... and in the zoom, centred 1.5 mm outside the contact
        q = (1 - sz) * q0 + sz * q1
        q[1] = 0.0
        return _camera(Vs, tgt - Ws * (Vs.T @ q), Ws)

    cmdA, achA = yaw_angles(A)
    cmdB, achB = yaw_angles(B)
    tmp = Path(tempfile.mkdtemp(prefix="abstract_rc_", dir=os.environ.get("TMPDIR", "/tmp")))
    L0, L1 = np.array(A["layout"], float), np.array(B["layout"], float)
    Fmax = 0.2
    for n, (seg, kf, w, sz) in enumerate(seq):
        kr = min(int(round(kf)), len(track_c) - 1)       # the nearest recorded frame of the zoomed rollout
        cam = cam_at(sz, kr if seg == "Z" else None)

        def uv(Pp):
            Qq = (np.asarray(Pp, float) - cam["ctr"]) @ cam["V"].T
            return Qq[..., 0], Qq[..., 2], Qq[..., 1]

        fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
        ax = fig.add_axes([0.02, 0.04, 0.60, 0.80])
        ax.set_xlim(-cam["W"], cam["W"])
        ax.set_ylim(-cam["W"] * aspect, cam["W"] * aspect)
        ax.set_aspect("equal")
        ax.axis("off")
        mm = ax.transData.transform([[0, 0], [1, 0]])
        ppm_true = (mm[1, 0] - mm[0, 0]) * 72.0 / fig.dpi
        ppm = ppm_true
        sw = full["W"] / cam["W"]                       # zoom factor: line widths drawn in mm would grow with it
        ppl = ppm / sw                                  # widths of the hand drawing stay as in the full view
        rec, row, obj, oalpha, forces = None, None, None, 1.0, True
        if seg in ("A", "B", "Z"):
            rec = A if seg == "A" else (B if seg == "B" else Zr)
            fr = frame_at(rec, kf)
            i = min(int(round(kf * rec["frame_every"])), len(rec["rows"]) - 1)
            row = rec["rows"][i]
            chains = chain(fr)
            obj = (np.array(fr["obj_p"]), np.array(fr["obj_R"]))
        else:
            forces = False
            fr = None
            if seg == "open":
                chains = {f: (1 - w) * A1[f] + w * A0[f] for f in A0}
                frl = frame_at(A, kA_end)
                obj, oalpha = (np.array(frl["obj_p"]), np.array(frl["obj_R"])), 1.0 - w
            elif seg == "show":
                chains = A0
            elif seg == "move":
                chains = {f: (1 - w) * A0[f] + w * B0[f] for f in A0}
            else:
                chains = B0
                frb = B["frames"][0]
                obj, oalpha = (np.array(frb["obj_p"]), np.array(frb["obj_R"])), w
        hand_a = 1.0 - _ease(min(1.0, sz / 0.6))        # the drawing fades as the zoom closes in
        pad_a = _ease((sz - 0.35) / 0.45) if sz > 0.35 else 0.0
        hl = {"A": 0.0, "B": 0.0, "Z": 0.0, "open": 0.0, "show": w, "move": 1.0, "in": 1.0 - w}[seg]
        for b in boxes():
            u, v, _ = uv(b)
            ax.add_patch(Polygon(np.c_[u, v], closed=True, fc=PALE, ec=MORPH if hl > 0 else GREY,
                                 lw=(0.6 + 1.6 * hl) * ppl, alpha=max(hand_a, 0.0), zorder=1))
            if hl > 0:
                ax.add_patch(Polygon(np.c_[u, v], closed=True, fc=MORPH, ec="none", alpha=0.10 * hl, zorder=1.05))
        if hl > 0:
            for f, ch in chains.items():
                m0 = ch[0]
                for e in (np.array([1.0, 0, 0]), np.array([0, 1.0, 0])):
                    q0, q1 = uv(m0 - 16 * e), uv(m0 + 16 * e)
                    ax.add_patch(FancyArrowPatch((q0[0], q0[1]), (q1[0], q1[1]), arrowstyle="<|-|>", color=MORPH,
                                                 lw=1.6 * ppl, mutation_scale=6.5 * ppl, alpha=hl, zorder=1.2))
                u0, v0, _ = uv(m0)
                ax.add_patch(Circle((u0, v0), 3.2, fc=MORPH, ec="white", lw=0.6 * ppl, alpha=hl, zorder=1.3))
        items = [(float(uv(ch)[2].mean()), "finger", f, ch) for f, ch in chains.items()]
        if obj is not None:
            items.append((float(uv(obj[0])[2]), "object", None, None))
        R0 = np.array((Zr if seg == "Z" else A if seg in ("A", "open", "show", "move") else B)["frames"][0]["obj_R"])
        kv = int(np.argmax(np.abs(R0[2])))
        kh = min((q for q in range(3) if q != kv), key=lambda q: abs(R0[2, q]))
        k3 = 3 - kv - kh
        z = 2
        for _dep, kind, f, ch in sorted(items, key=lambda x: -x[0]):
            if kind == "finger":
                if hand_a <= 0.01:
                    z += 1
                    continue
                u, v, _ = uv(ch)
                ax.plot(u, v, color=LINK, lw=4.6 * ppl, solid_capstyle="round", solid_joinstyle="round",
                        alpha=hand_a, zorder=z)
                ax.add_patch(Circle((u[-1], v[-1]), 4.4, fc=LINK, ec="none", alpha=hand_a, zorder=z + 0.1))
                for j in (1, 2, 4):
                    ax.add_patch(Circle((u[j], v[j]), 3.6, fc="white", ec=NEW, lw=1.7 * ppl, alpha=hand_a, zorder=z + 0.2))
                if forces and row is not None and obj is not None:
                    F = float(row["F"].get(f) or 0.0)
                    cop = row["cop_obj_mm"].get(f)
                    if cop is not None and F > 0.05:
                        c, R = obj
                        pp = c + R @ np.array(cop)
                        nn = (c - pp) / max(np.linalg.norm(c - pp), 1e-9)
                        q0, q1 = uv(pp - 7.0 * F * nn), uv(pp)
                        ax.add_patch(FancyArrowPatch((q0[0], q0[1]), (q1[0], q1[1]), arrowstyle="-|>", color=NEW,
                                                     lw=1.5 * ppl, mutation_scale=5.5 * ppl, alpha=hand_a, zorder=z + 0.3))
            else:
                c, R = obj
                u0, v0, d0 = uv(c)
                ax.add_patch(Circle((u0, v0), r, fc="#E5E7EB", ec=INK if pad_a > 0 else GREY,
                                    lw=(0.7 + 1.3 * pad_a) * ppl, alpha=oalpha, zorder=z))
                tt = np.linspace(0, 2 * np.pi, 361)
                for e0, e1 in ((kh, k3), (kh, kv)):
                    Pq = c + r * (np.outer(np.cos(tt), R[:, e0]) + np.outer(np.sin(tt), R[:, e1]))
                    u, v, dd = uv(Pq)
                    front = dd <= d0
                    for mask, sty, col in ((front, "-", INK), (~front, ":", GREY)):
                        ax.plot(np.where(mask, u, np.nan), np.where(mask, v, np.nan), sty, color=col,
                                lw=0.45 * ppl, alpha=oalpha * (1.0 - pad_a), zorder=z + 0.05)
                ut, vt, dt_ = uv(c + r * R[:, kh])
                ax.add_patch(Circle((ut, vt), 2.6, fc=ACH if dt_ <= d0 else "none", ec=ACH, lw=0.5 * ppl,
                                    alpha=oalpha, zorder=z + 0.06))
            z += 1
        if pad_a > 0 and seg == "Z":                    # the zoomed finger's pad spheres, over the drawing
            tip = fr["tips"][fz]
            Pp = np.array(tip["p"]) + padsL @ np.array(tip["R"]).T
            u, v, dd = uv(Pp)
            Fp = np.zeros(len(Pp))
            for i_, F_ in fr["pad_F"][fz].items():
                Fp[int(i_)] = F_
            d_cut = float(uv(track_c[kr])[2])           # a slice through the contact, one pad thick, seen side-on
            sl = np.abs(dd - d_cut) < a.slice_mm / 2
            ctx = np.flatnonzero(~sl)
            ax.add_collection(EllipseCollection(np.full(len(ctx), 2 * rp), np.full(len(ctx), 2 * rp), np.zeros(len(ctx)),
                                                units="xy", offsets=np.c_[u[ctx], v[ctx]], offset_transform=ax.transData,
                                                facecolors=(0.82, 0.84, 0.87, 0.05 * pad_a), edgecolors="none",
                                                zorder=39))
            order = np.flatnonzero(sl)
            order = order[np.argsort(-dd[order])]
            base = np.array([0.82, 0.84, 0.87])
            hot = np.array([0.486, 0.227, 0.929])
            tcol = np.clip(Fp[order] / Fmax, 0, 1)[:, None]
            mix = 0.35 + 0.65 * tcol                    # lilac at no force to the purple of the force arrows at Fmax
            cols = np.where(Fp[order, None] > 1e-4, mix * hot + (1 - mix) * np.array([0.93, 0.90, 0.99]), base)
            ec = np.where(Fp[order, None] > 1e-4, hot * 0.8, np.array([0.55, 0.57, 0.61]))
            ax.add_collection(EllipseCollection(np.full(len(order), 2 * rp), np.full(len(order), 2 * rp),
                                                np.zeros(len(order)), units="xy", offsets=np.c_[u[order], v[order]],
                                                offset_transform=ax.transData, facecolors=np.c_[cols, np.full(len(order), pad_a)],
                                                edgecolors=np.c_[ec, np.full(len(order), pad_a)],
                                                linewidths=0.4, zorder=40))
            nc, Ft = int((Fp > 1e-4).sum()), float(Fp.sum())
            if sz > 0.85:
                x0, y0 = -cam["W"] * 0.92, -cam["W"] * aspect * 0.88
                ax.plot([x0, x0 + 1.0], [y0, y0], color=INK, lw=3, zorder=50)
                ax.text(x0 + 0.5, y0 + 0.2, "1 mm", ha="center", va="bottom", fontsize=17, color=INK, zorder=50)
                fig.text(0.02, 0.862, f"A slice through the contact: {nc} of the {fz}'s pads touch the sphere, "
                         f"{Ft:.2f} N in all (darker: more force).", fontsize=17, color=ACH, va="top")
        after = seg in ("in", "B") or (seg == "Z" and Zr is B)
        lay = L0 + (_ease(w) if seg == "move" else (1.0 if after else 0.0)) * (L1 - L0)
        if seg in ("A", "B", "Z"):
            cmd, ach = (cmdA, achA) if rec is A else (cmdB, achB)
            t_rows = np.array([q["t"] for q in rec["rows"]])
            ax2 = fig.add_axes([0.67, 0.50, 0.30, 0.30])
            ax2.plot(t_rows, cmd, "--", color=NEW, lw=2.4, label="commanded")
            ax2.plot(t_rows[:i + 1], ach[:i + 1], color=ACH, lw=2.8, label="object")
            ax2.axvline(t_rows[i], color=GREY, lw=1.0)
            ax2.set_xlim(t_rows[0], t_rows[-1])
            lo = min(cmdA.min(), achA.min(), cmdB.min(), achB.min())
            hi = max(cmdA.max(), achA.max(), cmdB.max(), achB.max())
            ax2.set_ylim(lo - 0.1 * (hi - lo + 1), hi + 0.1 * (hi - lo + 1))
            ax2.set_xlabel("time (s)", fontsize=17)
            ax2.set_ylabel("turn about the vertical (deg)", fontsize=17)
            ax2.tick_params(labelsize=15)
            ax2.legend(fontsize=16, frameon=False, loc="upper left")
            for sp in ("top", "right"):
                ax2.spines[sp].set_visible(False)
            ax3 = fig.add_axes([0.67, 0.23, 0.30, 0.17])
            Fs = [float(row["F"].get(f) or 0.0) for f in ("thumb", "index", "middle")]
            ax3.barh(range(3), Fs, color=[NEW if v_ > 0.05 else GREY for v_ in Fs], height=0.55)
            ax3.set_yticks(range(3))
            ax3.set_yticklabels(["thumb", "index", "middle"], fontsize=16)
            ax3.invert_yaxis()
            ax3.set_xlim(0, 4.0)
            ax3.set_xlabel("pad force (N)", fontsize=17)
            ax3.tick_params(labelsize=15)
            for sp in ("top", "right"):
                ax3.spines[sp].set_visible(False)
            fig.text(0.67, 0.86, f"t = {row['t']:.2f} s   {row['phase']}", fontsize=19, color=INK, va="top")
            fig.text(0.67, 0.075, f"Tracking score {rec['track']:.2f}: rotation followed "
                     f"{1 - rec['track_err_rms_deg'] / rec['track_ref_rms_deg']:.2f},\nall three pads loaded "
                     f"{100 * rec['contact_all_frac']:.0f} % of the task", fontsize=17, color="#4B5563", va="center",
                     linespacing=1.4)
            head = ((f"Dynamics: the {fz}'s pad, 1,018 spheres of {2 * rp:g} mm", ACH) if seg == "Z" and sz > 0.35
                    else ("Control: HOM moves the nine finger joints", NEW))
        else:
            fig.text(0.67, 0.74, "Morphology", fontsize=30, color=MORPH, weight="bold", va="top")
            fig.text(0.67, 0.68, "Each finger's mount slides on two\nlead-screw stages, x and y:\nsix degrees of freedom "
                     "set\nbefore the task.", fontsize=21, color=INK, va="top", linespacing=1.35)
            fig.text(0.67, 0.42, "Control", fontsize=30, color=NEW, weight="bold", va="top")
            fig.text(0.67, 0.36, "Yaw, MCP and PIP of each finger:\nnine joints moved during the task.", fontsize=21,
                     color=INK, va="top", linespacing=1.35)
            head = ("Morphology: the stages move the finger mounts", MORPH)
        fig.text(0.02, 0.95, head[0], fontsize=30, color=head[1], weight="bold", va="top")
        fig.text(0.02, 0.895, f"Layout ({lay[0]:.0f}, {lay[1]:.0f}) mm: thumb to finger pair, and index to middle. "
                 f"{NAME[a.task]} of a {A['d_mm']:g} mm {OBJ[a.task]}, CPU simulation, "
                 + ("a quarter of real time." if seg == "Z" else ("paced to the hand's film." if sync else "real time.")),
                 fontsize=19, color="#4B5563", va="top")
        fig.savefig(tmp / f"f{n:05d}.png", facecolor="white")
        plt.close(fig)
    stem = a.out_stem or f"20261115-abstract_{a.task}"
    out = OUT / "media" / (f"{stem}.mp4" if a.out_stem else f"{stem}_reconfig.mp4")
    mv = [i_ for i_, s_ in enumerate(seq) if s_[0] == "move"]
    i_mid = mv[len(mv) // 2]                           # half way through the mounts' move

    def enc(first, count, path):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-start_number", str(first), "-i",
                        str(tmp / "f%05d.png"), "-frames:v", str(count), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "20", "-preset", "slow", "-movflags", "+faststart", str(path)], check=True)

    def still(i, path):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp / f"f{i:05d}.png"), "-q:v", "3", str(path)],
                       check=True)

    enc(0, len(seq), out)
    still(i_mid, out.with_suffix(".jpg"))
    zo = OUT / "media" / f"{stem}_pads_zoom.mp4"                                # the dynamics section's subclip
    enc(i_zoom, n_in + n_hold + n_out, zo)
    still(i_zoom + n_in + n_hold // 2, zo.with_suffix(".jpg"))
    mo = OUT / "media" / f"{stem}_morphology.mp4"                               # the morphology section's subclip
    enc(i_morph, i_morph_end - i_morph, mo)
    still(i_mid, mo.with_suffix(".jpg"))
    if sync:                                           # the drawing beside the hand's film, for the slide with both
        sbs = OUT / "media" / f"{stem.replace(f'abstract_{a.task}', 'abstract_vs_hand')}.mp4"
        dur = len(seq) / fps
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(out), "-i", str(ROOT / sync["film"]),
                        "-filter_complex", f"[0:v]scale=960:540:flags=lanczos,setsar=1[a];[1:v]tpad=stop_mode=clone:"
                        f"stop_duration={dur:.2f},setsar=1,format=yuv420p[b];[b][a]hstack=inputs=2,"   # hand left (owner)
                        f"trim=duration={dur:.3f}[v]", "-map", "[v]", "-r", str(fps), "-c:v", "libx264", "-pix_fmt",
                        "yuv420p", "-crf", "20", "-preset", "slow", "-movflags", "+faststart", str(sbs)], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{max(i_zoom - 10, 0) / fps:.3f}", "-i", str(sbs),
                        "-frames:v", "1", "-q:v", "3", str(sbs.with_suffix(".jpg"))], check=True)   # the second twist
        print(f"{sbs.relative_to(ROOT)}: the drawing beside {sync['film']}", flush=True)
    for q in tmp.glob("f*.png"):
        q.unlink()
    tmp.rmdir()
    print(f"{out.relative_to(ROOT)}: {len(seq)} frames ({len(seq) / fps:.1f} s); zoom {(n_in + n_hold + n_out) / fps:.1f} s "
          f"in {a.zoom_in} at t {Zr['frames'][kz]['t']:.2f} s on the {fz} (film {i_zoom / fps:.1f} s); morphology "
          f"{(i_morph_end - i_morph) / fps:.1f} s from {i_morph / fps:.1f} s", flush=True)
    return 0


# The ground-clear film of the SR2 hand (owner, 2026-11-16): its segments, read from its captions as film frames at
# 30 fps, and the rollout each control segment shows. film_session.py put the trials in at about 0.75-0.88 of real time
# (each clip's frame rate was measured over a span holding a 1.7 s camera gap), so a control segment's frames are mapped
# to the controller's clock through the trial clip they came from.
GC_FILM = "docs/experiments/20261115-hom_bench_trials/media/20261116-real_hand_twist_ground_clear.mp4"
GC_SEGMENTS = [(0, 23, "hold", "A", "recentre and pre-grasp at (70, 110) mm, 10x"),
               (23, 271, "control", "A", "control: HOM twist, ground-clear grasp at (70, 110) mm"),
               (271, 296, "hold", "A", "put down at (70, 110) mm, 10x"),
               (296, 313, "open", None, "fingers open at (70, 110) mm, 10x"),
               (313, 325, "show", None, "placement at (70, 110) mm, 10x"),
               (325, 412, "move", None, "reconfiguration to (105, 70) mm, 10x"),
               (412, 439, "in", None, "recentre and pre-grasp at (105, 70) mm, 10x"),
               (439, 729, "control", "B", "control: HOM twist, ground-clear grasp at (105, 70) mm"),
               (729, 809, "hold", "B", "put down, fingers open and straight, placement at (105, 70) mm, 10x")]
GC_TRIALS = {"A": ("20261116-125922", "trial_x70_y110_gr"), "B": ("20261116-130318", "trial_x105_y70_gr")}


def timeline(a):
    """The film's segments and, for each control segment, the controller time (s after HOM's start) shown in each of
    its frames: the film's frames are matched to the trial clip of camera 832412070344 by the best straight-line map
    from film frame to clip frame (the mean-removed correlation of 160 x 86 grey frames, summed over the segment), and
    the clip frame's camera time less the trial record's t0 is the controller time. The clips and their _frames.csv are
    corvus's logs/20261116-hom_trials/20261116-1258_session/ (--clips)."""
    import csv

    def grey(path):
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", "scale=160:90,format=gray", "-f", "rawvideo",
                              "-pix_fmt", "gray", "-"], capture_output=True, check=True).stdout
        g = np.frombuffer(raw, np.uint8).reshape(-1, 90, 160).astype(np.float32)[:, 4:, :]     # without the caption strip
        g = g - g.mean(axis=(1, 2), keepdims=True)
        return (g / (g.std(axis=(1, 2), keepdims=True) + 1e-6)).reshape(len(g), -1)

    film = grey(ROOT / GC_FILM)
    clips = Path(a.clips)
    runs = ROOT / "docs/experiments/20261115-hom_bench_trials/runs_corvus"
    segs = []
    for f0, f1, kind, rollout, cap in GC_SEGMENTS:
        s = {"f0": f0, "f1": f1, "kind": kind, "rollout": rollout, "caption": cap}
        if kind == "control":
            stamp, name = GC_TRIALS[rollout]
            base = clips / f"{stamp}_{name}_832412070344"
            clip = grey(base.with_suffix(".mp4"))
            with open(f"{base}_frames.csv") as fh:
                tc = np.array([float(r["t_camera_ms"]) for r in csv.DictReader(fh)])
            t0 = json.loads((runs / f"{stamp}_twist_corvus.jsonl").read_text().splitlines()[0])["t0_ms"]
            C = film[f0:f1] @ clip.T / film.shape[1]
            C -= C.mean(axis=1, keepdims=True)
            J = np.arange(f1 - f0)
            best = (-np.inf, 0.0, 0.0)
            for r in np.arange(0.70, 1.40, 0.002):
                for n0 in np.arange(-5.0, 70.0, 0.25):
                    n = np.round(n0 + J * r).astype(int)
                    ok = (n >= 0) & (n < len(clip))
                    if ok.mean() >= 0.98:
                        sc = float(C[J[ok], n[ok]].sum())
                        if sc > best[0]:
                            best = (sc, float(r), float(n0))
            _, r, n0 = best
            t = (np.interp(n0 + J * r, np.arange(len(tc)), tc) - t0) / 1000.0
            s.update({"trial": stamp, "clip_frames_per_film_frame": round(r, 4), "clip_frame0": n0,
                      "t": t.round(4).tolist()})
            print(f"{cap}: film {f0 / 30:.2f}-{f1 / 30:.2f} s shows controller time {t[0]:.2f} to {t[-1]:.2f} s "
                  f"({r * 30:.1f} clip frames per film second)", flush=True)
        segs.append(s)
    out = OUT / "rows" / "20261116-abstract_sync_ground_clear.json"
    out.write_text(json.dumps({"film": GC_FILM, "fps": 30, "frames": GC_SEGMENTS[-1][1], "segments": segs,
                               "rollouts": {"A": "70_110_gr_pads", "B": "105_70_gr_pads"}}, separators=(",", ":")))
    print(f"{out.relative_to(ROOT)}", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("record")
    r.add_argument("--task", default="twist")
    r.add_argument("--d", type=float, default=40.0)
    r.add_argument("--layout", type=float, nargs=2, default=(105.0, 70.0))
    r.add_argument("--grasp", type=int, default=None)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--rows", default="grid2dff_v4b_sv25r", help="grid rows that pick the grasp by the tracking score")
    r.add_argument("--grid", default="grid2dff", choices=("grid2dff", "grid2dgr"),
                   help="the toy grid of the grasp: grid2dgr holds the grasps clear of a ground at the sphere's lowest "
                        "point, as run on the SR2 hand on 2026-11-16 (lab/film_plan.py --grid)")
    r.add_argument("--tag", default="", help="a second rollout's name: rows/20261115-abstract_<task>_<tag>.json")
    r.add_argument("--pads", action="store_true", help="also each tip's pose and every pad's contact force per frame")
    d = sp.add_parser("render")
    d.add_argument("--task", default="twist")
    c = sp.add_parser("reconfig", help="rollout A (the 20261111 rows), the reconfiguration, rollout B (--tag)")
    c.add_argument("--task", default="twist")
    c.add_argument("--tag", default="70_110_pads", help="rollout B's rows (recorded with --pads to zoom in B)")
    c.add_argument("--tag-a", default="105_70_pads", help="rollout A's rows (recorded with --pads to zoom in A)")
    c.add_argument("--zoom-in", default="B", choices=("A", "B"), help="the rollout the zoom interrupts")
    c.add_argument("--zoom-at", type=float, default=4.6, help="that rollout's time (s) at which the zoom starts")
    c.add_argument("--zoom-finger", default="thumb")
    c.add_argument("--zoom-halfwidth", type=float, default=6.5, help="mm")
    c.add_argument("--slice-mm", type=float, default=0.8, help="thickness of the slice of pads drawn in the zoom")
    c.add_argument("--sync", default=None, help="a timeline of the `timeline` command: the film paced to the hand's film, "
                                                "30 frames per second, pauses where the hand's film shows set-up steps")
    c.add_argument("--out-stem", default=None, help="the films' name stem under media/ (default 20261115-abstract_<task>)")
    tl = sp.add_parser("timeline", help="the ground-clear film's segments and each control frame's controller time")
    tl.add_argument("--clips", required=True, help="a folder with the two trial clips of camera 832412070344 and their "
                                                    "_frames.csv, copied from corvus")
    a = ap.parse_args()
    return {"record": record, "render": render, "reconfig": render_reconfig, "timeline": timeline}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
