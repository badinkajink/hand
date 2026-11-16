#!/usr/bin/env python3
"""Result page of 2026-11-15: the SR2 hand's gantry travel measured with AprilTags on the backs of the DIP links (the
owner's session with the corvus lab session, 16:53-17:13; owner: "move each stepper to its max extent, track with
apriltags").

    figs   the figures and the film into media/ from data/20261115-gantry_follow_fit.json
           (scripts/hardware/gantry_follow_fit.py) and the follow logs in logs/20261115-gantry_travel/follow/
    page   docs/experiments/20261115-gantry_travel/20261115-gantry_travel_dip_tags.html

    ~/.claude/bin/resguard.sh run --mem 2G -- .venv/bin/python scripts/hardware/gantry_travel_page.py figs|page
"""
from __future__ import annotations

import base64
import html
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "docs/experiments/20261115-gantry_travel"
DATA, MEDIA = D / "data", D / "media"
LOGS = ROOT / "logs/20261115-gantry_travel/follow"
PAGE = D / "20261115-gantry_travel_dip_tags.html"
TEMPLATE = ROOT / "scripts/ur5/ur5_tip_calibration_page.template.html"
DJ = DATA / "20261115-gantry_follow_fit.json"
SNS = ["832412070344", "836612071918"]
TAGS = {"thumb": 24, "index": 25, "middle": 26}
NAME = {v: k for k, v in TAGS.items()}
COL = {"832412070344": "tab:blue", "836612071918": "tab:orange", "joint": "k"}
ORDER = ["165655", "170337", "170427", "170506", "170735", "170626", "171108", "171309", "170813", "171230"]
STAGE = {("thumb", "y"): "J0", ("thumb", "x"): "J1", ("index", "y"): "J2", ("index", "x"): "J3",
         ("middle", "y"): "J4", ("middle", "x"): "J5"}     # frames.json: each x_stage moves along y_P, each y_stage along -x_P
SHORT = {"165655": "stalled: the fingertip caught the rail", "170506": "tag lost behind the thumb",
         "170813": "nothing moved", "171230": "nothing moved"}


def j(p):
    return json.loads(Path(p).read_text())


def moves():
    return {m["stamp"]: m for m in j(DJ)["moves"]}


def label(m):
    a, b = m["from"][m["axis"]], m["to"][m["axis"]]
    return f"{m['finger']} {m['axis']} ({STAGE[(m['finger'], m['axis'])]}): {a:+.0f} to {b:+.0f} mm"


# ------------------------------------------------------------------ figures
def fig_traces(M):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(5, 2, figsize=(12, 15), sharex=False)
    for ax, s in zip(axs.ravel(), ORDER):
        m = M[s]
        for sn, tr in m["track"].items():
            tr = np.array(tr)
            if len(tr):
                ax.plot(tr[:, 0], tr[:, 1], ".", ms=2.5, color=COL[sn], label=f"camera {sn[-4:]}")
        ax.axhline(m["cmd_mm"], color="0.4", ls="--", lw=1)
        ax.axhline(0, color="0.7", lw=0.8)
        jn = m["solves"].get("joint")
        txt = f"both cameras {jn['along_mm']:.2f} mm" if jn else \
            "; ".join(f"{k[-4:]} {v['along_mm']:.2f} mm" for k, v in m["solves"].items()) or "tag not seen after the move"
        ax.set_title(f"{label(m)}\n{txt}" + (f"; {SHORT[s]}" if s in SHORT else ""), fontsize=12)
        ax.set_ylim(-8, m["cmd_mm"] + 10)
        ax.grid(alpha=0.3)
        ax.set_ylabel("travel along the command (mm)", fontsize=11)
        ax.legend(loc="center right", fontsize=10, markerscale=4)
    for ax in axs[-1]:
        ax.set_xlabel("time from 1.5 s before the command (s)")
    fig.tight_layout()
    fig.savefig(MEDIA / "20261115-travel_traces.png", dpi=100)
    plt.close(fig)


def fig_turn(M):
    import matplotlib.pyplot as plt
    rows = [s for s in ORDER if M[s]["solves"] and M[s]["solves"].get(next(iter(M[s]["solves"])))["along_mm"] > 20]
    fig, ax = plt.subplots(figsize=(11, 0.6 * len(rows) + 1.5))
    for k, s in enumerate(rows):
        for key, v in M[s]["solves"].items():
            ax.plot(v["turn_deg"], k, "o" if key == "joint" else "s", color=COL[key], ms=9 if key == "joint" else 6,
                    label="both cameras" if key == "joint" else f"camera {key[-4:]} alone")
    ax.axvline(0, color="0.5", lw=1)
    ax.set_yticks(range(len(rows)), [label(M[s]) + (" (stalled)" if s == "165655" else "") for s in rows], fontsize=12)
    ax.invert_yaxis()
    ax.set_xlabel("turn about the palm normal z_P from the commanded axis (deg, counterclockwise positive)")
    ax.grid(axis="x", alpha=0.3)
    h, lab = ax.get_legend_handles_labels()
    seen = {}
    for hh, ll in zip(h, lab):
        seen.setdefault(ll, hh)
    ax.legend(seen.values(), seen.keys(), loc="lower left", fontsize=11)
    fig.tight_layout()
    fig.savefig(MEDIA / "20261115-axis_turn.png", dpi=100)
    plt.close(fig)


def j4_watch():
    """Tag 26's (and the other tags') position in P over the lab session's J4 homing run of 17:20 (watch log)."""
    from sr2hand.geometry.rotations import tf_inv
    from sr2hand.robot.tool_chain import SR2Hand
    path = LOGS / "20261115-172024_watch.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    T_PB = tf_inv(np.asarray(rows[0]["T_BE"], float) @ SR2Hand.load().T_E_P())
    fr = [r for r in rows if "sn" in r and r["sn"] == SNS[0]]
    t0 = min(r["t"] for r in fr)
    out = {}
    for tag in (24, 25, 26):
        f = [r for r in fr if r["tag"] == tag]
        if f:
            P = np.array([(T_PB @ np.asarray(r["T_BT"], float))[:3, 3] for r in f])
            out[tag] = (np.array([r["t"] for r in f]) - t0, P)
    return out


def fig_j4(W):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 4.2))
    t, P = W[26]
    ax.plot(t, P[:, 1], ".", ms=2.5, color="tab:green", label="middle's tag 26")
    ax.axhline(-86.6, color="tab:red", ls="--", lw=1)
    ax.text(t.max(), -86.0, "start of the home, y_P -86.6 mm (lab, from the session's reference)", ha="right", va="bottom",
            fontsize=12, color="tab:red")
    ax.set_xlabel(f"time from the start of the watch (s), camera {SNS[0]}")
    ax.set_ylabel("y_P (mm)")
    ax.set_ylim(-92, -40)
    ax.grid(alpha=0.3)
    ax.legend(loc="center right", fontsize=11, markerscale=4)
    fig.tight_layout()
    fig.savefig(MEDIA / "20261115-j4_home_watch.png", dpi=100)
    plt.close(fig)


def film(M, speed=8.0, fps=30):
    """Top view of P: each move's tags as the better camera saw them, the moving finger's reachable box drawn through
    its tag, at speed times real time."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    from matplotlib.patches import Rectangle
    from sr2hand.geometry.rotations import tf_inv
    from sr2hand.robot import sr2_layouts as SL
    from sr2hand.robot.tool_chain import SR2Hand
    T_EP = SR2Hand.load().T_E_P()
    boxes = SL.tables()
    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=100)
    writer = FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "27"])
    out = MEDIA / "20261115-gantry_tags_film.mp4"
    poster = None
    with writer.saving(fig, str(out), dpi=100):
        for s in [x for x in ORDER if x != "165655"] + ["165655"]:
            m = M[s]
            path = next(LOGS.glob(f"20261115-{s}_move.jsonl"))
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            T_PB = tf_inv(np.asarray(rows[0]["T_BE"], float) @ T_EP)
            fr = [r for r in rows if "sn" in r and "T_BT" in r]
            mv = next(r["move"] for r in rows if "move" in r)
            t0 = mv["t"] - 1.5
            t_end = max(r["t"] for r in fr)
            tag = m["tag"]
            cams = sorted({r["sn"] for r in fr if r["tag"] == tag}, key=lambda sn: -sum(r["sn"] == sn and r["tag"] == tag
                                                                                        for r in fr))
            if not cams:
                continue
            sn = cams[0]
            pts = {}
            for r in fr:
                if r["sn"] == sn or (r["tag"] != tag and r["sn"] == SNS[0]):
                    pts.setdefault(r["tag"], []).append((r["t"], (T_PB @ np.asarray(r["T_BT"], float))[:3, 3]))
            b = boxes[m["finger"]]
            start = np.array(SL.P_from_driver(m["finger"], (mv["from"]["x"], mv["from"]["y"])))
            p_tag0 = np.array(pts[tag][0][1][:2]) if tag in pts else start
            off = p_tag0 - start
            for tt in np.arange(t0, t_end, speed / fps):
                ax.clear()
                ax.set_xlim(-150, 150)
                ax.set_ylim(-120, 120)
                ax.set_aspect("equal")
                ax.grid(alpha=0.25)
                ax.set_xlabel("x_P (mm)")
                ax.set_ylabel("y_P (mm)")
                c, h = b["centre"] + off, b["half"]
                ax.add_patch(Rectangle(c - h, 2 * h[0], 2 * h[1], fill=False, ec="0.35", ls="--", lw=1.5))
                tgt = np.array(SL.P_from_driver(m["finger"], (mv["to"]["x"], mv["to"]["y"]))) + off
                ax.plot(*tgt, "x", color="tab:red", ms=14, mew=3)
                for tg, lst in pts.items():
                    seen = [p for t, p in lst if t <= tt]
                    if not seen:
                        continue
                    P = np.array(seen)
                    ax.plot(P[:, 0], P[:, 1], "-", color="0.6" if tg != tag else "tab:green", lw=1.2)
                    ax.plot(P[-1, 0], P[-1, 1], "o", color="tab:green" if tg == tag else "0.3", ms=14)
                    ax.annotate(f"{NAME[tg]} {tg}", P[-1, :2], xytext=(10, 8), textcoords="offset points", fontsize=13)
                jn = m["solves"].get("joint")
                meas = (f"measured {jn['along_mm']:.2f} mm with both cameras" if jn else
                        "; ".join(f"camera {k[-4:]}: {v['along_mm']:.2f} mm" for k, v in m["solves"].items()))
                ax.set_title(f"{label(m)}, {m['cmd_mm']:.0f} mm commanded; {meas or 'tag lost after the move'}\n"
                             f"t = {tt - mv['t']:+.1f} s from the command, {speed:.0f} x real time; dashed: the finger's "
                             f"travel box drawn through its tag; x: the target" + (f"; {m['note']}" if m["note"] else ""),
                             fontsize=12)
                fig.tight_layout()
                writer.grab_frame()
                if s == "170626" and poster is None and tt - mv["t"] > 20:
                    poster = MEDIA / "20261115-gantry_tags_film_poster.jpg"
                    fig.savefig(poster, dpi=100)
    plt.close(fig)
    print(out, f"{out.stat().st_size / 1e6:.1f} MB")


def figs():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 13, "font.family": "serif"})
    MEDIA.mkdir(parents=True, exist_ok=True)
    M = moves()
    fig_traces(M)
    fig_turn(M)
    fig_j4(j4_watch())
    film(M)
    print("figures written to", MEDIA)


# ------------------------------------------------------------------ page helpers
def esc(s):
    return html.escape(str(s), quote=True)


def img_uri(p, width=1100, quality=86):
    from PIL import Image
    im = Image.open(p).convert("RGB")
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    b = io.BytesIO()
    im.save(b, "JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def fig(n, p, caption, width=1100):
    return (f'<figure id="fig{n}"><img src="{img_uri(p, width)}" alt="{esc(caption.split(".")[0])}">'
            f"<figcaption>Figure {n}. {caption}</figcaption></figure>")


def film_html(n, mp4, poster, caption):
    v = "data:video/mp4;base64," + base64.b64encode(Path(mp4).read_bytes()).decode()
    return (f'<figure id="fig{n}"><video controls muted loop playsinline autoplay preload="metadata" '
            f'poster="{img_uri(poster, 1280)}" src="{v}" style="width:100%"></video>'
            f"<figcaption>Figure {n}. {caption}</figcaption></figure>")


def table(head, body, num=(), caption=None, n=None):
    cap = f"<caption>Table {n}. {caption}</caption>" if caption else ""
    th = "".join(f'<th{" class=n" if k in num else ""}>{h}</th>' for k, h in enumerate(head))
    tr = "".join("<tr>" + "".join(f'<td{" class=n" if k in num else ""}>{c}</td>' for k, c in enumerate(r)) + "</tr>"
                 for r in body)
    return f'<div class="tw"><table>{cap}<thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


def code(s):
    return f"<code>{esc(s)}</code>"


if __name__ == "__main__":
    from gantry_travel_page_text import page
    {"figs": figs, "page": page}[sys.argv[1]]()
