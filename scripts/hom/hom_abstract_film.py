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

    def __init__(self, plant, fingers):
        m = plant.m
        self.plant, self.fingers = plant, fingers
        self.palm = m.body("palm_pose").id
        self.chain = {f: [m.body(f"{f}_{k}").id for k in CHAIN] for f in fingers}
        self.pads = {f: np.flatnonzero((m.geom_bodyid == m.body(f"{f}_tip").id)
                                       & np.array(["_pad" in (m.geom(g).name or "") for g in range(m.ngeom)]))
                     for f in fingers}

    def frame(self, plant, title, line):
        d = plant.d
        Rp, op = d.xmat[self.palm].reshape(3, 3), d.xpos[self.palm]
        to = (lambda p: (1e3 * (Rp.T @ (np.asarray(p) - op))).round(3).tolist())
        return {"t": round(float(d.time), 4),
                "chain": {f: [to(d.xpos[b]) for b in self.chain[f]] + [to(d.geom_xpos[self.pads[f]].mean(axis=0))]
                          for f in self.fingers},
                "obj_p": to(d.xpos[plant.tb]),
                "obj_R": (Rp.T @ d.xmat[plant.tb].reshape(3, 3)).round(6).tolist()}

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
    L.GRID_SRC = "grid2dff"
    call = HF.call_at(a.task, a.d, *a.layout)
    grasp, by = (a.grasp, {}) if a.grasp is not None else best_track_grasp(a.task, a.d, call, a.rows)
    _key, mxy, setup = HF.find(a.task, "grid2dff_size", a.d, call, grasp)
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
    rec = Recorder(plant, HG.FINGERS)
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


def render_reconfig(a):
    """Rollout A, the hand reconfiguring to B's layout, rollout B (owner, 2026-11-15: "an alternative version of the
    abstract video that extends into a reconfiguration ... highlight the morphology DoFs during a morphology reconfig,
    and then we can show the updated morphology doing the same task ... it has control (fingers) and morphology (xy
    repositioning)"). Between the rollouts the object fades out, the fingers return to A's open pose, the mounts slide on
    their stages to B's layout (the travel boxes and each mount's x and y axes in teal), and B's object fades in."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyArrowPatch, Polygon
    plt.rcParams.update({"font.family": "Nimbus Sans", "mathtext.fontset": "cm"})
    A = json.loads((OUT / "rows" / f"20261111-abstract_{a.task}.json").read_text())
    B = json.loads((OUT / "rows" / f"20261115-abstract_{a.task}_{a.tag}.json").read_text())
    V = view()

    def uv(P):
        Q = np.asarray(P, float) @ V.T
        return Q[..., 0], Q[..., 2], Q[..., 1]

    fps, r = 25, A["r_mm"]
    seq = []                                           # (segment, rec, frame index, blend) per output frame
    for k in range(len(A["frames"])):
        seq.append(("A", k, 0.0))
    for k in range(int(0.8 * fps)):                    # object out, fingers back to A's open pose
        seq.append(("open", k, _ease((k + 1) / (0.8 * fps))))
    for k in range(int(0.6 * fps)):                    # hold: the stages light up
        seq.append(("show", k, _ease((k + 1) / (0.6 * fps))))
    for k in range(int(2.8 * fps)):                    # mounts slide to B's layout
        seq.append(("move", k, _ease((k + 1) / (2.8 * fps))))
    for k in range(int(0.8 * fps)):                    # B's object in
        seq.append(("in", k, _ease((k + 1) / (0.8 * fps))))
    for k in range(len(B["frames"])):
        seq.append(("B", k, 0.0))
    chain = lambda fr: {f: np.array(c, float) for f, c in fr["chain"].items()}  # noqa: E731
    A0, A1, B0 = chain(A["frames"][0]), chain(A["frames"][-1]), chain(B["frames"][0])
    pts = [p for rec in (A, B) for fr in rec["frames"] for ch in fr["chain"].values() for p in ch]
    pts += [fr["obj_p"] for rec in (A, B) for fr in rec["frames"]] + [p for b in boxes() for p in b]
    U, Vv, _ = uv(np.array(pts))
    lim = (U.min() - r - 14, U.max() + r + 14, Vv.min() - r - 12, Vv.max() + 16)
    cmdA, achA = yaw_angles(A)
    cmdB, achB = yaw_angles(B)
    tmp = Path(tempfile.mkdtemp(prefix="abstract_rc_", dir=os.environ.get("TMPDIR", "/tmp")))
    L0, L1 = np.array(A["layout"], float), np.array(B["layout"], float)
    for n, (seg, k, w) in enumerate(seq):
        fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
        ax = fig.add_axes([0.02, 0.04, 0.60, 0.80])
        ax.set_xlim(lim[0], lim[1])
        ax.set_ylim(lim[2], lim[3])
        ax.set_aspect("equal")
        ax.axis("off")
        mm = ax.transData.transform([[0, 0], [1, 0]])
        ppm = (mm[1, 0] - mm[0, 0]) * 72.0 / fig.dpi
        rec, row, obj, oalpha, forces = None, None, None, 1.0, True
        if seg in ("A", "B"):
            rec = A if seg == "A" else B
            fr = rec["frames"][k]
            row = rec["rows"][min(k * rec["frame_every"], len(rec["rows"]) - 1)]
            chains = chain(fr)
            obj = (np.array(fr["obj_p"]), np.array(fr["obj_R"]))
        else:
            forces = False
            if seg == "open":
                chains = {f: (1 - w) * A1[f] + w * A0[f] for f in A0}
                fr = A["frames"][-1]
                obj, oalpha = (np.array(fr["obj_p"]), np.array(fr["obj_R"])), 1.0 - w
            elif seg == "show":
                chains = A0
            elif seg == "move":
                chains = {f: (1 - w) * A0[f] + w * B0[f] for f in A0}
            else:
                chains = B0
                fr = B["frames"][0]
                obj, oalpha = (np.array(fr["obj_p"]), np.array(fr["obj_R"])), w
        hl = {"A": 0.0, "B": 0.0, "open": 0.0, "show": w, "move": 1.0, "in": 1.0 - w}[seg]   # morphology highlight
        for b in boxes():
            u, v, _ = uv(b)
            ax.add_patch(Polygon(np.c_[u, v], closed=True, fc=PALE, ec=MORPH if hl > 0 else GREY,
                                 lw=(0.6 + 1.6 * hl) * ppm, alpha=1.0, zorder=1))
            if hl > 0:
                ax.add_patch(Polygon(np.c_[u, v], closed=True, fc=MORPH, ec="none", alpha=0.10 * hl, zorder=1.05))
        if hl > 0:                                      # each mount's x and y stage axes
            for f, ch in chains.items():
                m0 = ch[0]
                for e in (np.array([1.0, 0, 0]), np.array([0, 1.0, 0])):
                    q0, q1 = uv(m0 - 16 * e), uv(m0 + 16 * e)
                    ax.add_patch(FancyArrowPatch((q0[0], q0[1]), (q1[0], q1[1]), arrowstyle="<|-|>", color=MORPH,
                                                 lw=1.6 * ppm, mutation_scale=6.5 * ppm, alpha=hl, zorder=1.2))
                u0, v0, _ = uv(m0)
                ax.add_patch(Circle((u0, v0), 3.2, fc=MORPH, ec="white", lw=0.6 * ppm, alpha=hl, zorder=1.3))
        items = [(float(uv(ch)[2].mean()), "finger", f, ch) for f, ch in chains.items()]
        if obj is not None:
            items.append((float(uv(obj[0])[2]), "object", None, None))
        R0 = np.array((A if seg in ("A", "open", "show", "move") else B)["frames"][0]["obj_R"])
        kv = int(np.argmax(np.abs(R0[2])))
        kh = min((q for q in range(3) if q != kv), key=lambda q: abs(R0[2, q]))
        k3 = 3 - kv - kh
        z = 2
        for _dep, kind, f, ch in sorted(items, key=lambda x: -x[0]):
            if kind == "finger":
                u, v, _ = uv(ch)
                ax.plot(u, v, color=LINK, lw=4.6 * ppm, solid_capstyle="round", solid_joinstyle="round", zorder=z)
                ax.add_patch(Circle((u[-1], v[-1]), 4.4, fc=LINK, ec="none", zorder=z + 0.1))
                for j in (1, 2, 4):
                    ax.add_patch(Circle((u[j], v[j]), 3.6, fc="white", ec=NEW, lw=1.7 * ppm, zorder=z + 0.2))
                if forces and row is not None:
                    F = float(row["F"].get(f) or 0.0)
                    cop = row["cop_obj_mm"].get(f)
                    if cop is not None and F > 0.05 and obj is not None:
                        c, R = obj
                        p = c + R @ np.array(cop)
                        nrm = (c - p) / max(np.linalg.norm(c - p), 1e-9)
                        q0, q1 = uv(p - 7.0 * F * nrm), uv(p)
                        ax.add_patch(FancyArrowPatch((q0[0], q0[1]), (q1[0], q1[1]), arrowstyle="-|>", color=NEW,
                                                     lw=1.5 * ppm, mutation_scale=5.5 * ppm, zorder=z + 0.3))
            else:
                c, R = obj
                u0, v0, d0 = uv(c)
                ax.add_patch(Circle((u0, v0), r, fc="#E5E7EB", ec=GREY, lw=0.7 * ppm, alpha=oalpha, zorder=z))
                tt = np.linspace(0, 2 * np.pi, 181)
                for e0, e1 in ((kh, k3), (kh, kv)):
                    P = c + r * (np.outer(np.cos(tt), R[:, e0]) + np.outer(np.sin(tt), R[:, e1]))
                    u, v, dd = uv(P)
                    front = dd <= d0
                    for mask, sty, col in ((front, "-", INK), (~front, ":", GREY)):
                        ax.plot(np.where(mask, u, np.nan), np.where(mask, v, np.nan), sty, color=col,
                                lw=0.45 * ppm, alpha=oalpha, zorder=z + 0.05)
                ut, vt, dt_ = uv(c + r * R[:, kh])
                ax.add_patch(Circle((ut, vt), 2.6, fc=ACH if dt_ <= d0 else "none", ec=ACH, lw=0.5 * ppm,
                                    alpha=oalpha, zorder=z + 0.06))
            z += 1
        lay = L0 + (_ease(w) if seg == "move" else (1.0 if seg in ("in", "B") else 0.0)) * (L1 - L0)
        if seg in ("A", "B"):
            cmd, ach = (cmdA, achA) if seg == "A" else (cmdB, achB)
            t_rows = np.array([q["t"] for q in rec["rows"]])
            i = min(k * rec["frame_every"], len(rec["rows"]) - 1)
            ax2 = fig.add_axes([0.67, 0.50, 0.30, 0.30])
            ax2.plot(t_rows, cmd, "--", color=NEW, lw=2.4, label="commanded")
            ax2.plot(t_rows[:i + 1], ach[:i + 1], color=ACH, lw=2.8, label="object")
            ax2.axvline(t_rows[i], color=GREY, lw=1.0)
            ax2.set_xlim(t_rows[0], t_rows[-1])
            lo, hi = min(cmdA.min(), achA.min(), cmdB.min(), achB.min()), max(cmdA.max(), achA.max(), cmdB.max(), achB.max())
            ax2.set_ylim(lo - 0.1 * (hi - lo + 1), hi + 0.1 * (hi - lo + 1))
            ax2.set_xlabel("time (s)", fontsize=17)
            ax2.set_ylabel("turn about the vertical (deg)", fontsize=17)
            ax2.tick_params(labelsize=15)
            ax2.legend(fontsize=16, frameon=False, loc="upper left")
            for sp in ("top", "right"):
                ax2.spines[sp].set_visible(False)
            ax3 = fig.add_axes([0.67, 0.23, 0.30, 0.17])
            Fs = [float(row["F"].get(f) or 0.0) for f in ("thumb", "index", "middle")]
            ax3.barh(range(3), Fs, color=[NEW if v > 0.05 else GREY for v in Fs], height=0.55)
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
            head = (f"Control: HOM moves the nine finger joints", NEW)
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
                 f"{NAME[a.task]} of a {A['d_mm']:g} mm {OBJ[a.task]}, CPU simulation, real time.", fontsize=19,
                 color="#4B5563", va="top")
        fig.savefig(tmp / f"f{n:05d}.png", facecolor="white")
        plt.close(fig)
    out = OUT / "media" / f"20261115-abstract_{a.task}_reconfig.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(tmp / "f%05d.png"), "-c:v",
                    "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "slow", "-movflags", "+faststart",
                    str(out)], check=True)
    pick = len(A["frames"]) + int(0.8 * fps) + int(0.6 * fps) + int(1.4 * fps)       # mid-move
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp / f"f{pick:05d}.png"), "-q:v", "3",
                    str(out.with_suffix(".jpg"))], check=True)
    for q in tmp.glob("f*.png"):
        q.unlink()
    tmp.rmdir()
    print(f"{out.relative_to(ROOT)}: {len(seq)} frames ({len(seq) / fps:.1f} s)", flush=True)
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
    r.add_argument("--tag", default="", help="a second rollout's name: rows/20261115-abstract_<task>_<tag>.json")
    d = sp.add_parser("render")
    d.add_argument("--task", default="twist")
    c = sp.add_parser("reconfig", help="rollout A (the 20261111 rows), the reconfiguration, rollout B (--tag)")
    c.add_argument("--task", default="twist")
    c.add_argument("--tag", default="70_110")
    a = ap.parse_args()
    return {"record": record, "render": render, "reconfig": render_reconfig}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
