#!/usr/bin/env python3
"""Page of the board's drift over the lab's calibration session sphere2 (2026-10-22): the zero over time, what a press
leaves, whether it can be learned, and the viewer's estimator against each zero. Reads the study's drift.json
(scripts/ur5_tip_calibration_drift.py) and writes
docs/experiments/20261022-ur5_tip_calibration/20261023-ur5_tip_board_drift.html.

    /usr/bin/python3 scripts/ur5_tip_calibration_drift_page.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import docs_index  # noqa: E402
import retro_style  # noqa: E402
import texsvg  # noqa: E402
import ur5_tip_calibration_page as P  # noqa: E402   (esc, link, code, fig, table, plt_setup)

D = ROOT / "docs/experiments/20261022-ur5_tip_calibration"
DATA = D / "data/20261023-board_drift/drift.json"
MEDIA = D / "media"
PAGE = D / "20261023-ur5_tip_board_drift.html"
TEMPLATE = ROOT / "scripts/ur5_tip_calibration_page.template.html"
TEX_CACHE = D / "texsvg_cache.json"
ZEROS = [("first", "first rest"), ("first+field", "first rest, field"), ("time+field", "line in time"),
         ("learned+field", "learned steps"), ("before", "rest before"), ("before+field", "rest before, field"),
         ("decay+field", "rest before, decay"), ("both", "both rests"), ("both+field", "both rests, field")]
COL = {"first": "#888888", "first+field": "#555555", "time+field": "#7a5c99", "learned+field": "#b06010",
       "before": "#304080", "before+field": "#7fa7d6", "decay+field": "#2a9d8f", "both": "#2a7a2a", "both+field": "#8fc98f"}


# ------------------------------------------------------------------------------------------------ figures
def fig_zero(R, out):
    plt = P.plt_setup()
    s = R["series"]
    t, z, raw = np.array(s["t_min"]), np.array(s["z_norm"]), np.array(s["raw_norm"])
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.plot(t, raw, ".", ms=3, color="0.65", label="Reading less the first rest")
    before = np.array([w == "before" for w in s["which"]])
    ax.plot(t[before], z[before], "o", ms=3.5, color="#304080", label="Less the field term: rest before an action")
    ax.plot(t[~before], z[~before], "^", ms=3.5, color="#b03030", label="Less the field term: rest after it")
    ax.plot(t, s["learned"], "-", lw=1.4, color="#b06010", label="Learned steps and drift")
    ax.plot(t, s["time_only"], "--", lw=1.4, color="#7a5c99", label="Drift in time alone")
    ax.axvline(R["t_split_min"], color="0.4", lw=1, ls=":")
    ax.text(R["t_split_min"] + 1, ax.get_ylim()[1] * 0.97, "tested from here", va="top", fontsize=12, color="0.3")
    ax.set_xlabel("Time from the session's first rest (min)")
    ax.set_ylabel("Change from the first rest,\nall 15 components (µT)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=11, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_steps(R, out):
    plt = P.plt_setup()
    rows = [r for r in R["step_rows"] if r["top_N"] is not None]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.0, 3.8), gridspec_kw=dict(width_ratios=(1.2, 1)))
    col = dict(face="#304080", edge="#2a7a2a", flank="#b03030")
    for reg in ("face", "edge", "flank"):
        rr = [r for r in rows if r["region"] == reg]
        if rr:
            a1.plot([r["top_N"] + np.random.default_rng(1).uniform(-0.06, 0.06) for r in rr], [r["step_uT"] for r in rr],
                    "o", ms=4, color=col[reg], label=reg.capitalize(), alpha=0.8)
    a1.set_xlabel("The action's top force (N)")
    a1.set_ylabel("Rest after less rest before (µT)")
    a1.legend(loc="upper center", fontsize=11)
    kk = [r for r in R["step_rows"] if r.get("keep") is not None]
    a2.plot([r["gap_s"] for r in kk], [r["keep"] for r in kk], "o", ms=4, color="0.3", alpha=0.7)
    a2.axhline(1.0, color="#b03030", lw=1, ls="--")
    a2.axhline(0.0, color="#304080", lw=1, ls="--")
    a2.set_ylim(-1.0, 2.0)
    a2.set_xlabel("Time to the next action's rest (s)")
    a2.set_ylabel("Share of the step kept")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_scores(R, out):
    from ur5_tip_calibration_drift_text import GROUPS, SHORT
    plt = P.plt_setup()
    keys = ["first", "time+field", "learned+field", "before", "decay+field", "both"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.0, 4.4))
    x = np.arange(len(GROUPS))
    for k in keys:
        a1.plot(x, [R["table"][g][k]["loc_med"] for g in GROUPS], "-o", ms=4, lw=1.4, color=COL[k], label=dict(ZEROS)[k])
        a2.plot(x, [R["bias"][g][k]["err"] for g in GROUPS], "-o", ms=4, lw=1.4, color=COL[k])
    for a in (a1, a2):
        a.set_xticks(x)
        a.set_xticklabels([SHORT[g] for g in GROUPS], rotation=30, ha="right", fontsize=11)
    a1.set_ylabel("Contact point's error (mm, median)")
    a2.set_ylabel("Normal force's error (N, median)")
    a2.axhline(0.0, color="0.4", lw=1)
    h, lab = a1.get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(h, lab, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=11, frameon=False)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ page
def build():
    from ur5_tip_calibration_drift_text import sections            # the prose, kept apart from the plotting
    R = json.loads(DATA.read_text())
    MEDIA.mkdir(parents=True, exist_ok=True)
    F = dict(zero=MEDIA / "20261023-drift_zero.png", steps=MEDIA / "20261023-drift_steps.png",
             scores=MEDIA / "20261023-drift_scores.png")
    fig_zero(R, F["zero"])
    fig_steps(R, F["steps"])
    fig_scores(R, F["scores"])
    S, lede, summary = sections(R, F, P, ZEROS)
    toc = "\n".join(f'<li><a href="#{sid}">{k}. {P.esc(hd)}</a></li>' for k, (sid, hd, _) in enumerate(S, 1))
    secs = "\n".join(f'<section id="{sid}"><h2>{k}. {P.esc(hd)}</h2>{body}</section>' for k, (sid, hd, body) in enumerate(S, 1))
    page = TEMPLATE.read_text()
    for k, v in {"TITLE": "Fingertip board drift",
                 "H1": "Drift of the fingertip board's zero over a calibration session on the UR5",
                 "EYEBROW": "Printed fingertips and force sensing &#183; 2026-10-23", "LEDE": lede,
                 "BYLINE": "Study and page by the lab computer's Claude Code session, on the workstation's analysis, for the owner.",
                 "FILM": "", "TOC": toc, "SECTIONS": secs, "FOOTER": str(PAGE.relative_to(ROOT)),
                 "TEXCSS": texsvg.CSS}.items():
        page = page.replace("{{" + k + "}}", v)
    items = [((m.group(1) if m.group(1) is not None else m.group(2)).strip(), m.group(1) is not None)
             for m in P.TEX_RE.finditer(page)]
    (D / "data/20261023-board_drift/formulas.json").write_text(json.dumps(items, indent=0))
    svgs = iter(texsvg.render(items, cache_path=TEX_CACHE))
    page = P.TEX_RE.sub(lambda m: next(svgs), page)
    page = retro_style.apply(page)
    page = docs_index.tag(page, topic="fingertip", summary=summary)
    PAGE.write_text(page)
    print(f"{PAGE.relative_to(ROOT)}: {PAGE.stat().st_size / 1e6:.2f} MB, {len(items)} formulas", flush=True)


if __name__ == "__main__":
    build()
