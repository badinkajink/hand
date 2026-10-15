#!/usr/bin/env python3
"""Scores, tables and figure of experiment U5 of the 2026-10-14 pad-usability study: the four servo-plant finals (TPU mesh
and 1 mm pads, seeds 0 and 1) on four plant scenes under four conditions, closed loop and as open-loop replays, and on
their own scene at impratio 100 and a 4 ms step.

Terms
  held          two fingertips each press at least 0.24 N and the tool is above 60 mm at 5 s (rl_contact_eval.metrics)
  held cosine   signed cosine of the tool axis with the vertical at 5 s, mean over held rollouts
  plant effect  the same final on the 1 mm pad plant minus on the TPU-mesh plant (held count)
  policy effect on one plant, a pad-trained final minus a mesh-trained final (four seed pairs)
  an effect counts when all four seed-level differences share the sign and each exceeds 10 of 128 (closed loop) or
  5 of 64 (replays); Fisher exact p on the pooled rollouts is reported beside it
  substitute plant  2 mm pads or condim 4 against the 1 mm pad plant: held count within 5 of 128 and held cosine within
                0.03, per closed-loop final and condition (16 pairs)

    .venv/bin/python scripts/pad_usability_rl_report.py
Reads docs/experiments/20261014-pad_usability/rl/{cross,numerics}.jsonl; writes rl/summary_U5.json and
rl/20261014-u5_cross.png.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import pad_usability_agree as AG  # noqa: E402

RL = ROOT / "docs/experiments/20261014-pad_usability/rl"
REL = "docs/experiments/20261014-pad_usability/rl/"
FINALS = [("20261011-d6_servo_tpu27mesh_40M_s0", "Mesh s0"), ("20261011-d6_servo_tpu27mesh_40M_s1", "Mesh s1"),
          ("20261011-d6_servo_tpu27pads1_40M_s0", "Pads s0"), ("20261011-d6_servo_tpu27pads1_40M_s1", "Pads s1")]
SCENES = [("tpu27mesh", "TPU-mesh plant"), ("tpu27pads1", "1 mm pad plant"), ("tpu27pads2", "2 mm pad plant"),
          ("tpu27meshc4", "Condim-4 plant")]
CONDS = [("nominal", "Nominal"), ("kp=0.36", "Servo gain 0.36"), ("fr=0.12", "Torque limit 0.12"),
         ("friction=0.5", "Friction 0.5")]


def fisher(a, n1, b, n2):
    from scipy.stats import fisher_exact
    return float(fisher_exact([[a, n1 - a], [b, n2 - b]])[1])


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(c - h) * n, float(c + h) * n


def main() -> int:
    R = {(r["tag"], r["scene"], r["mode"], r["condition"]): r for r in AG.jl(RL / "cross.jsonl") if r.get("status") == "ok"}
    N = [r for r in AG.jl(RL / "numerics.jsonl") if r.get("status") == "ok"]
    out = dict(id="U5", cells=[], effects=[], substitutes=[], numerics=[])
    for (tag, lab) in FINALS:
        for sc, sname in SCENES:
            for mode in ("closed", "open"):
                for c, cname in CONDS:
                    r = R.get((tag, sc, mode, c))
                    if r:
                        lo, hi = wilson(r["n_held"], r["n"])
                        out["cells"].append(dict(final=lab, tag=tag, scene=sc, mode=mode, condition=c, n=r["n"],
                                                 n_held=r["n_held"], wilson95=[lo, hi], held_cos=r["held_cos_mean"],
                                                 grip_N=r.get("grip_N")))
    thr = {"closed": 10, "open": 5}
    for mode in ("closed", "open"):
        for c, _ in CONDS:
            # plant effect: each final, pad plant minus mesh plant
            d = []
            for tag, lab in FINALS:
                a, b = R.get((tag, "tpu27pads1", mode, c)), R.get((tag, "tpu27mesh", mode, c))
                if a and b:
                    d.append((lab, a["n_held"] - b["n_held"], a["n_held"], b["n_held"], a["n"]))
            if len(d) == 4:
                same = all(x[1] > 0 for x in d) or all(x[1] < 0 for x in d)
                counts = same and all(abs(x[1]) > thr[mode] for x in d)
                n = d[0][4]
                p = fisher(sum(x[2] for x in d), 4 * n, sum(x[3] for x in d), 4 * n)
                out["effects"].append(dict(kind="plant", mode=mode, condition=c, diffs={x[0]: x[1] for x in d},
                                           counts=bool(counts), fisher_p=p))
            # policy effect on each plant: pad-trained minus mesh-trained, four seed pairs
            for sc, _ in SCENES:
                rows = {lab: R.get((tag, sc, mode, c)) for tag, lab in FINALS}
                if all(rows.values()):
                    pairs = [(pp, mm, rows[pp]["n_held"] - rows[mm]["n_held"]) for pp in ("Pads s0", "Pads s1")
                             for mm in ("Mesh s0", "Mesh s1")]
                    same = all(x[2] > 0 for x in pairs) or all(x[2] < 0 for x in pairs)
                    counts = same and all(abs(x[2]) > thr[mode] for x in pairs)
                    n = rows["Mesh s0"]["n"]
                    p = fisher(rows["Pads s0"]["n_held"] + rows["Pads s1"]["n_held"], 2 * n,
                               rows["Mesh s0"]["n_held"] + rows["Mesh s1"]["n_held"], 2 * n)
                    out["effects"].append(dict(kind="policy", scene=sc, mode=mode, condition=c,
                                               diffs={f"{a} - {b}": v for a, b, v in pairs}, counts=bool(counts),
                                               fisher_p=p))
    for sc in ("tpu27pads2", "tpu27meshc4"):
        ok = tot = 0
        for tag, lab in FINALS:
            for c, _ in CONDS:
                a, b = R.get((tag, sc, "closed", c)), R.get((tag, "tpu27pads1", "closed", c))
                if a and b:
                    tot += 1
                    dc = (None if a["held_cos_mean"] is None or b["held_cos_mean"] is None
                          else a["held_cos_mean"] - b["held_cos_mean"])
                    good = abs(a["n_held"] - b["n_held"]) <= 5 and dc is not None and abs(dc) <= 0.03
                    ok += good
                    out["substitutes"].append(dict(scene=sc, final=lab, condition=c, d_held=a["n_held"] - b["n_held"],
                                                   d_cos=dc, within=bool(good)))
        out[f"substitute_{sc}"] = f"{ok} of {tot}"
    nom = {(r["tag"]): r for r in N if r["numerics"] == "nominal"}
    for r in N:
        b = nom.get(r["tag"])
        if b is None:
            continue
        sl = lambda x: (np.nanmedian([v for v in (x.get("mech_slide_hold_path_um_s") or []) if v is not None])  # noqa: E731
                        if x.get("mech_slide_hold_path_um_s") else None)
        out["numerics"].append(dict(tag=r["tag"], numerics=r["numerics"], n_held=r["n_held"], held_cos=r["held_cos_mean"],
                                    d_held=r["n_held"] - b["n_held"],
                                    d_cos=None if r["held_cos_mean"] is None or b["held_cos_mean"] is None
                                    else r["held_cos_mean"] - b["held_cos_mean"],
                                    slide_hold_um_s=sl(r), slide_ratio=(sl(r) / sl(b)) if sl(r) and sl(b) else None,
                                    creep_mm_s=r.get("creep_mm_s_median"), impratio_read=r.get("impratio_read")))
    (RL / "summary_U5.json").write_text(json.dumps(out, indent=1) + "\n")
    fig(out, RL / "20261014-u5_cross.png")
    for e in out["effects"]:
        if e["counts"]:
            print("effect", e["kind"], e.get("scene", ""), e["mode"], e["condition"], e["diffs"], f"p={e['fisher_p']:.2g}")
    print({k: v for k, v in out.items() if k.startswith("substitute_")})
    for x in out["numerics"]:
        print(x)
    return 0


def fig(S, path):
    plt = AG._style()
    cols = {"Mesh s0": "#1baf7a", "Mesh s1": "#0e7a52", "Pads s0": "#2a78d6", "Pads s1": "#164e96"}
    mk = {"Mesh s0": "o", "Mesh s1": "s", "Pads s0": "^", "Pads s1": "D"}
    fig_, axs = plt.subplots(2, 4, figsize=(12.4, 6.6), dpi=110, sharey="row",
                             gridspec_kw=dict(hspace=0.42, wspace=0.12))
    for i, mode in enumerate(("closed", "open")):
        for j, (c, cname) in enumerate(CONDS):
            ax = axs[i, j]
            for lab in cols:
                pts = [(k, x["n_held"] / x["n"]) for k, (sc, _) in enumerate(SCENES) for x in S["cells"]
                       if x["final"] == lab and x["scene"] == sc and x["mode"] == mode and x["condition"] == c]
                if pts:
                    off = {"Mesh s0": -0.18, "Mesh s1": -0.06, "Pads s0": 0.06, "Pads s1": 0.18}[lab]
                    ax.plot([p[0] + off for p in pts], [100 * p[1] for p in pts], mk[lab], color=cols[lab], ms=7,
                            markeredgecolor="white", markeredgewidth=0.8, label=lab, linestyle="none")
            ax.set_xticks(range(len(SCENES)))
            ax.set_xticklabels(["Mesh", "1 mm\npads", "2 mm\npads", "Condim\n4"], fontsize=15)
            ax.set_ylim(-4, 104)
            ax.set_title(cname, fontsize=15)
            ax.grid(axis="y", color="#ddd", lw=0.8)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        axs[i, 0].set_ylabel(("Closed loop" if mode == "closed" else "Open-loop replay") + "\nheld (%)")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig_.legend(h, [x.replace("Mesh", "TPU-mesh policy,").replace("Pads", "1 mm pad policy,") for x in l], loc="lower center",
                ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01), fontsize=15)
    fig_.subplots_adjust(left=0.08, right=0.99, top=0.94, bottom=0.17)
    fig_.savefig(path)
    plt.close(fig_)


if __name__ == "__main__":
    sys.exit(main())
