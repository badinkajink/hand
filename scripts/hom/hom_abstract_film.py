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
    path = OUT / "rows" / f"20261111-abstract_{a.task}.json"
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
    d = sp.add_parser("render")
    d.add_argument("--task", default="twist")
    a = ap.parse_args()
    return {"record": record, "render": render}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
