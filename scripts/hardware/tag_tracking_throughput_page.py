#!/usr/bin/env python3
"""Result page of 2026-11-13: throughput of the two-camera tag tracker on corvus and the controller link for sim2real
control (owner, 16:2x: "see how much throughput we can get on tag tracking ... get this ready for sim2real control. if
we need to setup policy streaming over ethernet, so be it").

    data   (corvus only: reads logs/20261113-tag_tracking_throughput/) writes
           docs/experiments/20261113-tag_tracking_throughput/data/20261113-tracking_throughput.json: each run's summary
           (tracker_bench.py), a sample of its frame ages, two still-block traces, the link's runs (link_summary.py)
           and the HOM timing (hom_timing.py)
    figs   the matplotlib figures into media/ from data/
    page   docs/experiments/20261113-tag_tracking_throughput/20261113-tag_tracking_throughput.html

    DOCS_REFRESH=0 /usr/bin/python3 scripts/hardware/tag_tracking_throughput_page.py data|figs|page
The film comes from docs/experiments/20261113-tag_tracking_throughput/lab/tracker_film.py.
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
D = ROOT / "docs/experiments/20261113-tag_tracking_throughput"
DATA, MEDIA, LAB = D / "data", D / "media", D / "lab"
LOGS = ROOT / "logs/20261113-tag_tracking_throughput"
PAGE = D / "20261113-tag_tracking_throughput.html"
TEMPLATE = ROOT / "scripts/ur5/ur5_tip_calibration_page.template.html"
DJ = DATA / "20261113-tracking_throughput.json"
SNS = ["832412070344", "836612071918"]
RUNS = [("1080p30_e166", "1920 × 1080, 30 fps, 16.6 ms"), ("1080p30_e50", "1920 × 1080, 30 fps, 5 ms"),
        ("720p30_e166", "1280 × 720, 30 fps, 16.6 ms"), ("540p60_e100", "960 × 540, 60 fps, 10 ms"),
        ("540p60_e50", "960 × 540, 60 fps, 5 ms"), ("1080p15_e166", "1920 × 1080, 15 fps, 16.6 ms"),
        ("1080p30_noroi", "1920 × 1080, 30 fps, whole frame"), ("540p60_noroi", "960 × 540, 60 fps, whole frame"),
        ("540p60_e100_long", "960 × 540, 60 fps, 10 ms, 10 min"),
        ("1080p30_withhom", "1920 × 1080, 30 fps, 16.6 ms, beside HOM")]
TRACES = ("1080p30_e166", "540p60_e100")
LINKS = [("link_local2", "50 Hz, corvus to itself"), ("link_ws", "50 Hz, corvus to the workstation"),
         ("link_ws100", "100 Hz, corvus to the workstation")]
CAL = ROOT / "docs/experiments/20261113-camera_arm_calibration/data/fits"


def j(p):
    return json.loads(Path(p).read_text())


def rows(p):
    return [json.loads(l) for l in open(p) if l.strip()]


# ------------------------------------------------------------------ data
def data():
    from sr2hand.geometry.rotations import log_so3, quat_to_mat
    DATA.mkdir(parents=True, exist_ok=True)
    out = {"runs": {}, "traces": {}, "links": {}, "hom": None}
    for name, label in RUNS:
        sf = LOGS / "m1" / f"{name}_summary.json"
        if not sf.exists():
            continue
        s = j(sf)
        R = rows(LOGS / "m1" / f"{name}_samples.jsonl")
        ok = [r for r in R if not r["flags"] & 1]
        age = np.array([r["t_publish"] - r["t_capture"] for r in ok])
        rng = np.random.default_rng(0)
        pick = np.sort(rng.choice(len(age), min(600, len(age)), replace=False)) if len(age) else []
        out["runs"][name] = dict(label=label, summary=s, age_sample=np.round(age[pick], 2).tolist())
        if name in TRACES:
            both = [r for r in ok if not r["flags"] & 2]
            t = np.array([r["t_capture"] for r in both]) / 1000.0
            P = np.array([r["p_BO"] for r in both])
            Q = np.array([r["q_BO"] for r in both])
            R0 = quat_to_mat(Q[len(Q) // 2])
            w = np.degrees(np.array([log_so3(R0.T @ quat_to_mat(q)) for q in Q]))
            k = max(1, len(t) // 600)
            out["traces"][name] = dict(t=np.round(t[::k] - t[0], 3).tolist(),
                                       dp=np.round((P - P.mean(0))[::k], 4).tolist(),
                                       dw=np.round((w - w.mean(0))[::k], 4).tolist())
    for name, label in LINKS:
        f = LOGS / f"{name}.jsonl"
        if f.exists():
            sf = LOGS / f"{name}_summary.json"
            R = rows(f)
            rtt = [r["rtt"] for r in R if r.get("rtt") is not None]
            out["links"][name] = dict(label=label, summary=j(sf) if sf.exists() else None,
                                      rtt_sample=np.round(rtt[::max(1, len(rtt) // 600)], 3).tolist())
    he = j(CAL / "20261113-handeye_tcp_fitted_lens_moved836.json")["cams"]
    pos = out["runs"].get("1080p30_e166", {}).get("summary", {}).get("static", {}).get("position_mm")
    if pos is not None:
        from sr2hand.geometry.rotations import tf_inv
        out["block_distance_mm"] = {sn: round(float(np.linalg.norm(tf_inv(np.array(he[sn]["T_CB"]))[:3, 3] - pos)), 1)
                                    for sn in SNS}
    hf = sorted(LOGS.glob("hom_timing*.json"))
    if hf:
        out["hom"] = {p.stem: j(p) for p in hf}
    DJ.write_text(json.dumps(out))
    print(DJ, f"{DJ.stat().st_size / 1e3:.0f} kB", sorted(out["runs"]), sorted(out["links"]))


# ------------------------------------------------------------------ figures
def figs():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 13, "font.family": "serif"})
    MEDIA.mkdir(parents=True, exist_ok=True)
    d = j(DJ)
    runs = [(n, d["runs"][n]) for n, _ in RUNS if n in d["runs"]]
    # frame age by mode
    fig, ax = plt.subplots(figsize=(11, 0.55 * len(runs) + 1.6))
    ages = [r["age_sample"] for _, r in runs]
    ax.boxplot(ages, vert=False, whis=(5, 95), showfliers=False, widths=0.6, medianprops=dict(color="k", lw=2))
    ax.set_yticks(range(1, len(runs) + 1), [r["label"] for _, r in runs])
    ax.invert_yaxis()
    ax.axvline(50, color="tab:red", ls="--", lw=1.5)
    ax.text(49, 0.995, "plan's target, 50 ms", color="tab:red", va="top", ha="right", fontsize=12,
            transform=ax.get_xaxis_transform())
    xr = max(max(a) for a in ages) + 4
    for k, (_, r) in enumerate(runs):
        c = r["summary"]["cpu"]["process_percent_of_one_core"]
        ax.text(xr, k + 1, f"{r['summary']['outputs']['rate_hz']:.1f} Hz, CPU {c:.0f} %", va="center", fontsize=12)
    ax.set_xlim(0, xr + 27)
    ax.set_xlabel("frame age when the pose is published (ms): box 25-75 %, whiskers 5-95 %")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(MEDIA / "20261113-frame_age_by_mode.png", dpi=110)
    plt.close(fig)
    # still-block traces
    tr = d["traces"]
    if tr:
        fig, ax = plt.subplots(len(tr), 2, figsize=(13, 2.9 * len(tr)), squeeze=False)
        for row, name in zip(ax, [n for n in TRACES if n in tr]):
            t, dp, dw = np.array(tr[name]["t"]), np.array(tr[name]["dp"]) * 1000, np.array(tr[name]["dw"])
            for k, (lab, col) in enumerate(zip("xyz", ("tab:red", "tab:green", "tab:blue"))):
                row[0].plot(t, dp[:, k], color=col, lw=0.8, label=lab)
                row[1].plot(t, dw[:, k] * 60, color=col, lw=0.8, label=lab)
            row[0].set_ylabel("position (µm)")
            row[1].set_ylabel("rotation (arcmin)")
            row[0].set_title(f"{d['runs'][name]['label']}: block position about its mean", fontsize=13)
            row[1].set_title("block rotation about its mean", fontsize=13)
            for a_ in row:
                a_.grid(alpha=0.3)
                a_.legend(loc="upper right", ncol=3, fontsize=11)
        for a_ in ax[-1]:
            a_.set_xlabel("time (s)")
        fig.tight_layout()
        fig.savefig(MEDIA / "20261113-still_block_traces.png", dpi=110)
        plt.close(fig)
    # latency budget
    fig, ax = plt.subplots(figsize=(11, 0.55 * len(runs) + 1.4))
    for k, (_, r) in enumerate(runs):
        s = r["summary"]
        det = np.mean([c["age_at_detection_ms"]["p50"] for c in s["per_camera"].values()])
        period = 1000.0 / float(s["mode"].split("@")[1])
        pub = s["outputs"]["age_at_publish_ms"]["p50"]
        dms = np.mean([max((c["det_ms"]["roi"] or {}).get("p50", 0), (c["det_ms"]["full"] or {}).get("p50", 0))
                       for c in s["per_camera"].values()])
        segs = [(det - dms, "tab:gray", "readout and transfer to corvus (about one frame period)"),
                (dms, "tab:green", "detection"), (pub - det, "tab:blue", "pairing wait and joint solve")]
        x = 0
        for w_, col, lab in segs:
            ax.barh(k, w_, left=x, color=col, label=lab if k == 0 else None)
            x += w_
    ax.set_yticks(range(len(runs)), [r["label"] for _, r in runs])
    ax.invert_yaxis()
    ax.set_xlabel("median frame age when the pose is published (ms)")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(MEDIA / "20261113-latency_budget.png", dpi=110)
    plt.close(fig)
    # link round trips
    if d["links"]:
        fig, ax = plt.subplots(figsize=(11, 3.6))
        for name, lk in d["links"].items():
            x = np.sort(lk["rtt_sample"])
            if len(x):
                ax.plot(x, np.linspace(0, 1, len(x)), lw=2, label=lk["label"])
        ax.set_xlabel("state sent to target applied (ms), with a 2 ms stand-in controller")
        ax.set_ylabel("share of periods")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=11)
        fig.tight_layout()
        fig.savefig(MEDIA / "20261113-link_round_trip.png", dpi=110)
        plt.close(fig)
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
            f'poster="{img_uri(poster, 1400)}" src="{v}" style="width:100%"></video>'
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
    from tag_tracking_throughput_page_text import page
    {"data": data, "figs": figs, "page": page}[sys.argv[1]]()
