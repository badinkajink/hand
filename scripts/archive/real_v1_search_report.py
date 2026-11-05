"""Read a `real_v1_design_search` pass and answer three questions about it.

1. WHICH DESIGNS REORIENT.  Per design, the best (grasp, pivot height, turn angle) cell it has,
   ranked on the repeated mean and gated on the cell keeping the shaft in most draws.
2. WHAT SCORES A DESIGN.  Every candidate cheap score the search recorded, ranked by Spearman
   against the outcome. A score is only useful if it can be computed WITHOUT the rollout, so the
   correlation is reported against the design's best-cell outcome, and separately as the AUC of a
   "does this design reorient at all" classifier -- which is the decision a design search actually
   makes.
3. HOW THEY TURN IT.  The style vector, labelled by a rule that reads off the contact trace:

     CARRY      pads ride the shaft, contact fixed         mean carry_frac > 0.55
     PIVOT      pads slide, the shaft spins in the grip    0.15 < carry_frac <= 0.55
     DRIVE      a pad outruns the rotation, over-travels   carry_frac <= 0.15 or any < -0.2

Usage:
    uv run python scripts/real_v1_search_report.py --rows docs/experiments/.../lin_pass \
        --out docs/experiments/.../REPORT.md --figs docs/experiments/.../figs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

FINGERS = ("thumb", "index", "middle")

# Candidate design scores, all computable from a settled grasp with no rollout. `sign` is the
# direction each is expected to point if it is the right score.
SCORES = {
    "extend_mm": ("radial reach left before the finger is straight", +1),
    "ceiling_deg": ("asin(extend / straddle), the fixed-contact bound", +1),
    "tau_cap_Nmm": ("mu * sum f_n * moment arm about the pinch axis", +1),
    "tau_pair_Nmm": ("the same, index and middle only", +1),
    "tau_thumb_Nmm": ("the same, thumb only", +1),
    "sweep_min_mm": ("worst finger's tangential authority in +-0.5 rad", +1),
    "sweep_sum_mm": ("the three fingers' tangential authority, summed", +1),
    "straddle_mm": ("half the index-middle contact separation", +1),
    "grip_N": ("total normal force at the grasp", -1),
    "grip_depth_mm": ("palm above the shaft's axis", -1),
    "depth_fit_mm": ("the fitter's own grip depth", -1),
}
GEOM = {"x_sep_mm": "thumb-to-pair mount separation",
        "y_sep_mm": "index-to-middle mount separation",
        "thumb_y_mm": "thumb mount offset along the shaft"}


def load(rows_dir: Path) -> list[dict]:
    rows = []
    for f in sorted(rows_dir.glob("rows_*.json")):
        rows.extend(json.load(f.open()))
    return [r for r in rows if "error" not in r]


def best_cell(row: dict):
    best = None
    for g in row.get("grasps", []):
        for c in g.get("cells", []):
            key = (c["kept"] * 2 >= c["n"], c["mean_cos"])
            if best is None or key > best[0]:
                best = (key, {**c, **{k: g[k] for k in
                                      ("straddle_mm", "thumb_axial_mm", "depth_req_mm",
                                       "squeeze_mm")},
                              "scores": g["scores"]})
    return None if best is None else best[1]


def restyle(rows: list[dict], lift=0.10, turn_steps=550, hold_steps=800) -> None:
    """Re-run each design's best cell with a contact trace and recompute its style in place.

    The sweep computes the style from the same rollout it scores, which is right, but the style
    function has been corrected since some passes were run (pad slip is now gated on the pad
    actually touching the shaft), and re-running one cell per design is a minute of CPU against
    45 for the whole sweep. It is also the honest way to add a trace-derived quantity after the
    fact -- the raw traces are not kept.
    """
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    import numpy as _np
    import probe_real_v1_carry as pc
    import real_v1_design_search as ds
    root = Path(__file__).resolve().parents[1]
    for r in rows:
        b = best_cell(r)
        if b is None:
            continue
        scene = root / r["scene"]
        built = pc._grip_from_fit(scene, b["straddle_mm"] / 1000.0, 0.0,
                                  b["squeeze_mm"] / 1000.0, ds.OBJ,
                                  None if b["depth_req_mm"] is None else b["depth_req_mm"] / 1000.0,
                                  b["thumb_axial_mm"] / 1000.0)
        if built is None:
            continue
        tr = []
        out = pc.carry(scene, lift, turn_steps, hold_steps, _np.radians(b["angle_deg"]), 0.0,
                       ds.OBJ, 0.5, False, straddle=b["straddle_mm"] / 1000.0,
                       label=r["design"], axis_k=b["axis_k"],
                       linear_anchor=(b["mode"] == "linear"), built=built, contact_trace=tr)
        if out is None:
            continue
        st = ds.style(tr, out)
        for g in r["grasps"]:
            for c in g.get("cells", []):
                if (c["axis_k"], c["mode"], c["angle_deg"]) == (b["axis_k"], b["mode"],
                                                                b["angle_deg"]) \
                        and g["straddle_mm"] == b["straddle_mm"] \
                        and g["thumb_axial_mm"] == b["thumb_axial_mm"] \
                        and g["depth_req_mm"] == b["depth_req_mm"]:
                    c["style"] = st


def label_style(st: dict) -> str:
    """How many pads are on the shaft while it turns, and where the shaft goes in the grip.

    An earlier version labelled on the pads' carry fraction (1 - pad travel / surface arc).
    That separates a true carry from a spin in principle and does not in practice on this hand:
    the pads come on and off the shaft and walk along it as well as around it, so 72 of 80
    designs landed in one bucket at roughly -1, i.e. "travelled about twice the arc". The
    contact count and the shaft's rise or fall in the grip do separate them, they are what the
    renders show, and they are what a hardware build cares about.
    """
    if not st or st.get("mean_contacts") is None:
        return "-"
    n = st["mean_contacts"]
    if st.get("obj_dz_mm", 0.0) > 15.0:
        return "PALM-PIN"
    if n >= 2.6:
        return "TRIPOD"
    if n >= 1.5:
        return "PINCH-ROLL"
    return "SINGLE"


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 6 or np.std(x[ok]) == 0:
        return float("nan"), float("nan")
    r = stats.spearmanr(x[ok], y[ok])
    return float(r.statistic), float(r.pvalue)


def auc(scores, labels) -> float:
    """P(score of a reorienting design > score of a non-reorienting one). 0.5 = no information."""
    s, l = np.asarray(scores, float), np.asarray(labels, bool)
    ok = np.isfinite(s)
    s, l = s[ok], l[ok]
    if l.sum() == 0 or (~l).sum() == 0:
        return float("nan")
    r = stats.rankdata(s)
    return float((r[l].sum() - l.sum() * (l.sum() + 1) / 2) / (l.sum() * (~l).sum()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--figs", type=Path, default=None)
    ap.add_argument("--restyle", action="store_true",
                    help="re-run each design's best cell to recompute its style vector")
    ap.add_argument("--bar", type=float, default=0.5,
                    help="mean held cos a design must clear to count as reorienting")
    args = ap.parse_args()

    rows = load(args.rows)
    if args.restyle:
        restyle(rows)
    table = []
    for r in rows:
        b = best_cell(r)
        if b is None:
            table.append({"design": r["design"], **{k: r.get(k) for k in GEOM},
                          "graspable": False, "mean_cos": 0.0, "kept": 0, "n": 0,
                          "style": "-", "reorients": False})
            continue
        st = b.get("style", {})
        table.append({
            "design": r["design"], **{k: r.get(k) for k in GEOM},
            "graspable": True,
            "mean_cos": b["mean_cos"], "sd_cos": b["sd_cos"],
            "kept": b["kept"], "n": b["n"],
            "axis_k": b["axis_k"], "angle_deg": b["angle_deg"], "mode": b["mode"],
            "straddle_mm": b["straddle_mm"], "thumb_axial_mm": b["thumb_axial_mm"],
            "depth_req_mm": b["depth_req_mm"],
            "final_z": b["final_z"], "contacts": b["contacts"], "force_N": b["force_N"],
            "style": label_style(st),
            "carry_frac": st.get("carry_frac"), "drive_share": st.get("drive_share"),
            "driver": st.get("driver"), "mean_contacts": st.get("mean_contacts"),
            "obj_dz_mm": st.get("obj_dz_mm"), "max_util": st.get("max_util"),
            "touch_frac": st.get("touch_frac"), "settle_frac": st.get("settle_frac"),
            "cos_turn_end": st.get("cos_turn_end"),
            "reorients": bool(b["kept"] * 2 >= b["n"] and b["mean_cos"] >= args.bar),
            **{k: b["scores"].get(k) for k in SCORES},
        })
    table.sort(key=lambda t: (-t["mean_cos"] if t["reorients"] else 1e9, -t["mean_cos"]))

    n_ok = sum(t["reorients"] for t in table)
    lines = [f"# real_v1 design search — {len(table)} hands",
             "",
             f"Held reorient (mean held cos >= {args.bar} over the cell's repeats, shaft kept in "
             f"most draws): **{n_ok} / {len(table)}**. "
             f"Graspable at all: {sum(t['graspable'] for t in table)}.",
             "",
             "## Every design, best cell",
             "",
             "| design | Xsep | Ysep | Ty | grasp sp/dep/thAx | k | ang | mean cos | sd | kept |"
             " z | style | driver |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for t in table:
        if not t["graspable"]:
            lines.append(f"| {t['design']} | {t['x_sep_mm']:.0f} | {t['y_sep_mm']:.0f} | "
                         f"{t['thumb_y_mm']:.0f} | — no grasp — | | | | | | | | |")
            continue
        lines.append(
            f"| {t['design']} | {t['x_sep_mm']:.0f} | {t['y_sep_mm']:.0f} | {t['thumb_y_mm']:.0f} "
            f"| {t['straddle_mm']:.0f}/{t['depth_req_mm'] or 'auto'}/{t['thumb_axial_mm']:.0f} "
            f"| {t['axis_k']:.2f} | {t['angle_deg']:.0f} | **{t['mean_cos']:+.3f}** "
            f"| {t['sd_cos']:.3f} | {t['kept']}/{t['n']} | {t['final_z']:.3f} "
            f"| {t['style']} | {t['driver'] or '-'} |")

    lines += ["", "## What scores a design", "",
              "Spearman is against the best cell's mean held cos over every design that has a "
              "grasp; AUC is P(a reorienting design scores above a non-reorienting one), so 0.5 "
              "is no information and 1.0 is a perfect separator. `n` is the number of designs "
              "the score is defined on.", "",
              "| score | what it is | rho | p | AUC | n |", "|---|---|---|---|---|---|"]
    have = [t for t in table if t["graspable"]]
    y = [t["mean_cos"] for t in have]
    lab = [t["reorients"] for t in have]
    stats_rows = []
    for k, (what, sign) in {**SCORES, **{g: (d, +1) for g, d in GEOM.items()}}.items():
        x = [t.get(k) if t.get(k) is not None else float("nan") for t in have]
        rho, p = spearman(x, y)
        a = auc(x, lab)
        stats_rows.append((k, what, rho, p, a, int(np.isfinite(np.asarray(x, float)).sum())))
    stats_rows.sort(key=lambda r: -abs(r[2]) if np.isfinite(r[2]) else 0)
    for k, what, rho, p, a, n in stats_rows:
        lines.append(f"| `{k}` | {what} | {rho:+.3f} | {p:.1e} | {a:.3f} | {n} |")

    styles = {}
    for t in have:
        styles.setdefault(t["style"], []).append(t)
    lines += ["", "## Styles", "",
              "| style | n | mean held cos | reorients | pads on the shaft (thumb/index/middle) |"
              " mean contacts | shaft dz | settles after the command | driver |",
              "|---|---|---|---|---|---|---|---|---|"]
    for name, group in sorted(styles.items()):
        tf = [g["touch_frac"] for g in group if g.get("touch_frac")]
        tfm = ("/".join(f"{np.mean([t[f] for t in tf]):.2f}" for f in FINGERS) if tf else "-")
        drivers = {}
        for g in group:
            drivers[g["driver"]] = drivers.get(g["driver"], 0) + 1
        sf = [g["settle_frac"] for g in group if g.get("settle_frac") is not None]
        lines.append(f"| {name} | {len(group)} | {np.mean([g['mean_cos'] for g in group]):+.3f} "
                     f"| {sum(g['reorients'] for g in group)}/{len(group)} | {tfm} "
                     f"| {np.mean([g['mean_contacts'] or 0 for g in group]):.2f} "
                     f"| {np.mean([g['obj_dz_mm'] or 0 for g in group]):+.1f} mm "
                     f"| {np.mean(sf) if sf else float('nan'):+.2f} "
                     f"| {', '.join(f'{k} x{v}' for k, v in sorted(drivers.items(), key=lambda kv: -kv[1]))} |")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:40]))
    print(f"\n-> {args.out}")
    if args.json_out:
        args.json_out.write_text(json.dumps(table, indent=1))

    if args.figs:
        _figures(table, stats_rows, args.figs, args.bar)
    return 0


def _figures(table, stats_rows, out: Path, bar: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out.mkdir(parents=True, exist_ok=True)
    have = [t for t in table if t["graspable"]]

    # 1. the landscape: mount separations vs outcome
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for a, (kx, lx) in zip(ax, (("x_sep_mm", "thumb-to-pair separation (mm)"),
                                ("y_sep_mm", "index-to-middle separation (mm)"))):
        a.scatter([t[kx] for t in have], [t["mean_cos"] for t in have],
                  c=["#2f8f6f" if t["reorients"] else "#b5504a" for t in have], s=26, alpha=.85)
        a.axhline(bar, color="#888", lw=.8, ls="--")
        a.set_xlabel(lx)
        a.set_ylabel("best cell, mean held cos")
        a.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(out / "landscape.png", dpi=140)
    plt.close(fig)

    # 2. the best two scores against the outcome
    top = [r for r in stats_rows if np.isfinite(r[2])][:2]
    if top:
        fig, ax = plt.subplots(1, len(top), figsize=(5.4 * len(top), 4.4), squeeze=False)
        for a, (k, what, rho, p, au, n) in zip(ax[0], top):
            a.scatter([t.get(k) for t in have], [t["mean_cos"] for t in have],
                      c=["#2f8f6f" if t["reorients"] else "#b5504a" for t in have], s=26)
            a.set_xlabel(f"{k}  ({what})")
            a.set_ylabel("best cell, mean held cos")
            a.set_title(f"rho {rho:+.2f}   AUC {au:.2f}")
            a.grid(alpha=.25)
        fig.tight_layout()
        fig.savefig(out / "scores.png", dpi=140)
        plt.close(fig)
    print(f"-> figures in {out}")


if __name__ == "__main__":
    raise SystemExit(main())
