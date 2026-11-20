#!/usr/bin/env python3
"""Page: what the fingertip boards read against the HEX-E, by the indenter's size (the UR5 calibrations) and by where a
contact lands on the pad (arm presses on the SR2 hand, 2026-11-19 evening). The owner, 16:3x: "the contact area vs force
sensing is really disconcerting. is tehre any way to estimate the size of the contact patch and thus get better
performance. can we try to learn that from our existing calibrations?"

Reads only committed files of docs/experiments/20261117-sr2_system_identification/data/s3/ (made by the study's
sim/20261119-*.py) and the calibration holds tables they cite. Writes
docs/experiments/20261117-sr2_system_identification/20261119-fingertip_board_force_by_contact_place.html. No formulas
(the page builds on corvus, which has no TeX).

    DOCS_REFRESH=0 /usr/bin/python3 scripts/fingertip/fingertip_board_force_place_page.py     (on corvus)
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import _paths  # noqa: E402,F401
import docs_index  # noqa: E402
import retro_style  # noqa: E402
import ur5_tip_calibration_page as P  # noqa: E402   (esc, fig, table, plt_setup)

D = ROOT / "docs/experiments/20261117-sr2_system_identification"
S3 = D / "data/s3"
PAGE = D / "20261119-fingertip_board_force_by_contact_place.html"
TEMPLATE = ROOT / "scripts/ur5/ur5_tip_calibration_page.template.html"
F_CAL = S3 / "20261119-board_patch_size_calibration.json"
F_ARM = S3 / "20261119-arm_press_boards.json"
F_PM = S3 / "20261119-board_force_pattern_match.json"
F_SUP = S3 / "20261119-board_patch_superposition.json"
F_TR = S3 / "20261119-arm_press_traces.json"
F_HAND18 = S3 / "20261119-board_field_per_newton_hand.json"
F_DW = S3 / "20261119-hexe_dead_weight.json"
CORR = ROOT / "docs/experiments/20261022-ur5_tip_calibration/data/20261105-viewer_inputs/correction"
RIG = ROOT / "docs/experiments/20261030-inverted_rig_seat_block/data/holds"
GROUPS = [("index", "PIP 0"), ("middle", "PIP 0"), ("thumb", "PIP 0"), ("index", "PIP 10"), ("index", "PIP 20")]
GEO_Z = {("index", "PIP 0"): 7.5, ("middle", "PIP 0"): 7.8, ("thumb", "PIP 0"): 11.6, ("index", "PIP 10"): 12.6,
         ("index", "PIP 20"): 15.0}       # the hand model's contact place (pad frame z, mm) at the press's joints, before load
COL = {"index": "#1f5fbf", "middle": "#b8481a", "thumb": "#2e8b57"}


def rel(p):
    import os
    return os.path.relpath(p, D)


def link(p, text):
    return f'<a href="{P.esc(rel(p))}">{text}</a>'


def code(s):
    return f"<code>{P.esc(s)}</code>"


def group_of(pr):
    pip = pr["indenter"].split("PIP")[-1].strip()
    return (pr["finger"], f"PIP {pip}")


def presses(arm):
    out = []
    for run in arm["runs"]:
        for pr in run["presses"]:
            if "g" in pr:
                out.append({**pr, "run": run["run"]})
    return out


def fig_calibration(C, out):
    plt = P.plt_setup()
    rows = []
    for tip, v in C.items():
        for k, r in v.items():
            if k in ("n_normal", "gp_trained_on_sphere_R5") or not isinstance(r, dict) or r.get("n", 0) < 10:
                continue
            rows.append((f"tip {tip.split()[0]}, {k}", r))
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=110, sharey=True)
    y = np.arange(len(rows))[::-1]
    for a, key, title in ((ax[0], "field_ratio", "Board field per newton"), (ax[1], "depth_ratio", "Indentation depth per newton")):
        for yi, (lab, r) in zip(y, rows):
            lo, md, hi = r[key]
            c = "#1f5fbf" if "rod" in lab else "#b8481a"
            a.plot([lo, hi], [yi, yi], color=c, lw=2.2)
            a.plot([md], [yi], "o", color=c, ms=8)
            a.text(hi + 0.03, yi, f"{md:.2f}", va="center", fontsize=12)
        a.axvline(1.0, color="0.4", lw=1, ls="--")
        a.set_xlim(0.4, 1.75)
        a.set_title(title)
        a.set_xlabel("ratio (median and quartiles)")
    ax[0].set_yticks(y)
    ax[0].set_yticklabels([f"{lab} (n {r['n']})" for lab, r in rows])
    fig.suptitle("Against the same tip's R 5 sphere at the same place and force", fontsize=13)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return rows


def fig_hand(R, out):
    plt = P.plt_setup()
    fig, ax = plt.subplots(figsize=(11.5, 5.0), dpi=110)
    for k, g in enumerate(GROUPS):
        rs = [r for r in R if group_of(r) == g]
        if not rs:
            continue
        xs = k + np.linspace(-0.12, 0.12, len(rs))
        ax.plot(xs, [r["field_per_N_uT"] for r in rs], "o", color=COL[g[0]], ms=9, label=None)
        for x_, r in zip(xs, rs):
            good = r["best_cos"] >= 0.9
            ax.plot([x_], [r["cal_field_per_N_uT"]], "_", color="k" if good else "0.65", ms=22, mew=2.2 if good else 1.4)
        cs = [r["best_cos"] for r in rs]
        ax.text(k, max(max(r["field_per_N_uT"] for r in rs), max(r["cal_field_per_N_uT"] for r in rs)) + 1.5,
                f"cosine {min(cs):.2f}-{max(cs):.2f}", ha="center", fontsize=12)
    ax.set_xticks(range(len(GROUPS)))
    ax.set_xticklabels([f"{f}\n{p}" for f, p in GROUPS])
    ax.set_ylabel("board field per HEX-E newton (uT/N)")
    ax.set_ylim(0, 37)
    ax.plot([], [], "o", color="0.3", ms=9, label="measured, each press (index blue, middle orange, thumb green)")
    ax.plot([], [], "_", color="k", ms=22, mew=2.2, label="the tip's calibration at the best-matching place (cosine 0.9 and over)")
    ax.plot([], [], "_", color="0.65", ms=22, mew=1.4, label="the same where no calibrated place matches (cosine under 0.9)")
    ax.legend(loc="lower right", frameon=True, framealpha=0.95, fontsize=11)
    ax.set_title("Arm presses at 0.2 mm/s to 2.5 N on the S3 block's plates")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_traces(T, out):
    plt = P.plt_setup()
    fig, ax = plt.subplots(figsize=(11.5, 5.0), dpi=110)
    style = {("index", "PIP 0"): ("#1f5fbf", "-"), ("index", "PIP 10"): ("#6a3d9a", "-"), ("index", "PIP 20"): ("#e31a1c", "-"),
             ("middle", "PIP 0"): ("#b8481a", ":"), ("thumb", "PIP 0"): ("#2e8b57", ":")}
    for tr in T["traces"]:
        g = (tr["finger"], "PIP " + tr["indenter"].split("PIP")[-1].strip())
        rows = np.array(tr["rows"])
        c, ls = style.get(g, ("0.5", "-"))
        ax.plot(rows[:, 1], rows[:, 3], ls, color=c, lw=1.6, label=f"{g[0]}, {g[1]}")
    ax.set_xlabel("HEX-E force along the press (N)")
    ax.set_ylabel("board change along its own field per newton (uT)")
    ax.legend(loc="upper left", frameon=False)
    ax.set_title("One press each, loading and unloading (0.1 s bins)")
    ax.axhline(0, color="0.6", lw=0.8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def cal_z(tip_sessions):
    zs = []
    for s in tip_sessions:
        z = np.load(s / "holds.npz", allow_pickle=True)
        p = z["p"]
        zs.append(p[np.isfinite(p).all(1), 2])
    return np.concatenate(zs)


def fig_pad(R, out):
    plt = P.plt_setup()
    fig, ax = plt.subplots(2, 1, figsize=(11.5, 6.6), dpi=110, gridspec_kw={"height_ratios": [2.2, 1]}, sharex=True)
    z = np.linspace(0, 21.41, 400)
    za = 10.86
    az = np.where(z < za, 22.0, 11.0)
    x = 8.5 * np.sqrt(np.clip(1 - ((z - za) / az) ** 2, 0, None))
    ax[0].fill_between(z, 0, x, color="#dddddd")
    ax[0].plot(z, x, "k", lw=1.5)
    ax[0].text(0.3, 1.0, "mount face\n(proximal)", fontsize=12)
    ax[0].text(18.2, 1.0, "distal end", fontsize=12)
    ax[0].plot([za], [8.5], "kv", ms=8)
    ax[0].text(za + 0.4, 7.6, "apex", ha="left", fontsize=12)
    for k, g in enumerate(GROUPS):
        rs = [r for r in R if group_of(r) == g]
        if not rs:
            continue
        zg = GEO_Z[g]
        yr = 9.2 + 0.7 * k
        c = COL[g[0]]
        ax[0].plot([zg], [yr], "o", color=c, mfc="white", ms=9, mew=2)
        zb = float(np.median([r["best_p"][2] for r in rs]))
        cs = float(np.median([r["best_cos"] for r in rs]))
        ax[0].plot([zb], [yr], "s" if cs > 0.9 else "x", color=c, ms=9, mew=2)
        ax[0].plot([zg, zb], [yr] * 2, color=c, lw=1, ls=":")
        ax[0].text(21.6, yr, f"{g[0]} {g[1]}", color=c, fontsize=11, va="center")
    ax[0].plot([], [], "o", color="0.3", mfc="white", mew=2, label="the hand model's contact place")
    ax[0].plot([], [], "s", color="0.3", label="best calibration pattern (cosine > 0.9)")
    ax[0].plot([], [], "x", color="0.3", mew=2, label="best calibration pattern (cosine < 0.9)")
    ax[0].legend(loc="lower center", frameon=False, fontsize=11)
    ax[0].set_ylabel("depth x (mm)")
    ax[0].set_ylim(0, 13.0)
    ax[0].set_xlim(-0.5, 26)
    ax[0].set_title("The v3 pad (one shape on all three fingers) from the side, in the calibrations' pad frame")
    z3s = cal_z([CORR / "20261030-spherev3id3"])
    z3r = cal_z([RIG / "20261104-145215-id3pts11"])
    b = np.arange(0, 22.5, 1.0)
    ax[1].hist(z3s, bins=b, color="#999999", label="tip 3, UR5 sphere holds")
    ax[1].hist(z3r, bins=b, color="#e31a1c", alpha=0.8, label="tip 3, inverted rig's rod presses")
    ax[1].set_xlabel("z along the pad (mm)")
    ax[1].set_ylabel("holds")
    ax[1].legend(loc="upper right", frameon=False, fontsize=11)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_estimators(PM, out):
    plt = P.plt_setup()
    fig, ax = plt.subplots(figsize=(11.5, 5.0), dpi=110)
    rows = PM["rows"]
    c = np.array([r["best_cos"] for r in rows])
    for key, lab, m, col in (("pattern_slope", "pattern-matched linear estimate", "o", "#1f5fbf"),
                             ("gp_slope", "the trials' GP without the own-pose term", "s", "#b8481a"),
                             ("gp_own_pose_slope", "the trials' GP with it (as in the trials)", "^", "#2e8b57")):
        ax.plot(c, [r[key] for r in rows], m, color=col, ms=8, label=lab, alpha=0.85)
    ax.axhline(1.0, color="0.4", lw=1, ls="--")
    ax.set_xlabel("best cosine of the press's field with any calibrated press of its tip")
    ax.set_ylabel("estimated force per HEX-E newton")
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper left", frameon=False)
    ax.set_title("Force estimates on the arm presses against the HEX-E")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def summary_rows(R, PM):
    pm = {(r["run"], r["rep"]): r for r in PM["rows"]}
    out = []
    for g in GROUPS:
        rs = [r for r in R if group_of(r) == g]
        if not rs:
            continue

        def rng(v, f="{:.2f}"):
            return f.format(min(v)) if f.format(min(v)) == f.format(max(v)) else f"{f.format(min(v))}-{f.format(max(v))}"
        pms = [pm[(r["run"], r["rep"])] for r in rs]
        out.append([f"{g[0]}, {g[1]}", str(len(rs)), rng([r["field_per_N_uT"] for r in rs], "{:.1f}"),
                    rng([r["best_cos"] for r in rs]), f"{GEO_Z[g]:.1f}",
                    rng([r["best_p"][2] for r in rs], "{:.1f}"),
                    rng([r["estimators"]["gp_own_pose"]["slope"] for r in rs]),
                    rng([p["pattern_slope"] for p in pms])])
    return out


def build():
    C = json.loads(F_CAL.read_text())
    A = json.loads(F_ARM.read_text())
    PM = json.loads(F_PM.read_text())
    T = json.loads(F_TR.read_text())
    DW = json.loads(F_DW.read_text()) if F_DW.exists() else None
    R = presses(A)
    tmp = Path(tempfile.mkdtemp())
    rows_cal = fig_calibration(C, tmp / "f1.png")
    fig_hand(R, tmp / "f2.png")
    fig_traces(T, tmp / "f3.png")
    fig_pad(R, tmp / "f4.png")
    fig_estimators(PM, tmp / "f5.png")
    fr = [r["field_ratio"][1] for _, r in rows_cal]
    dr = [r["depth_ratio"][1] for _, r in rows_cal]
    by = {g: [r for r in R if group_of(r) == g] for g in GROUPS}
    i0 = by[("index", "PIP 0")]
    i20 = by[("index", "PIP 20")]
    pm20 = [r for r in PM["rows"] if "PIP 20" in r["indenter"]]
    def rng(v, f="{:.2f}"):
        a, b = f.format(min(v)), f.format(max(v))
        return a if a == b else f"{a}-{b}"
    lede = (f"The fingertip boards measure force, not contact area. In the UR5 calibrations, an R 2 mm sphere and an R 12.5 mm "
            f"rod read {rng(fr)} times the R 5 sphere's field per newton at the same place. Their indentation per newton "
            f"changed {rng(dr)} times. On the SR2 hand, the reading depends on where a flat contact lands on the pad. "
            f"Pressed straight into a plate, the index pad gives {rng([r['field_per_N_uT'] for r in i0], '{:.1f}')} uT per "
            f"HEX-E newton, in a pattern no calibrated press shows. With its PIP bent 20 deg it gives "
            f"{rng([r['field_per_N_uT'] for r in i20], '{:.1f}')} uT/N, {rng([r['ratio'] for r in i20])} of its "
            f"calibration, and the pattern matches at cosine {min(r['best_cos'] for r in i20):.2f}. There the trials' GP "
            f"reads {rng([r['gp_own_pose_slope'] for r in pm20])} of the force; a linear estimate at the best-matching "
            f"calibrated place reads {rng([r['pattern_slope'] for r in pm20])}.")
    if DW is not None and DW.get("ratio_range"):
        lo, hi = DW["ratio_range"]
        lede += (f" The HEX-E itself read a dead weight at {lo:.2f}-{hi:.2f} of its true weight. If it read this low in October "
                 f"too, these ratios stand, and absolute HEX-E forces are about {1 / hi:.1f} times too small. If its scale "
                 f"changed since, the boards read about half their calibration even at PIP 20 (section 7).")
    summary = lede
    S = []
    S.append(("question", "The question and the short answer", """
<p>The owner, 2026-11-19 16:3x: &#8220;the contact area vs force sensing is really disconcerting. is there any way to
estimate the size of the contact patch and thus get better performance. can we try to learn that from our existing
calibrations?&#8221; The worry came from the presses of 11-18 and 11-19 on the owner&#8217;s 80/20 post. There the
boards&#8217; force estimate read 0.26-0.36 of the HEX-E with the pad flat on a plate, and 0.51-0.60 with the PIP
bent.</p>
<p>The existing calibrations answer the first half. The boards&#8217; field per newton does not change with the
indenter&#8217;s size (section 2). That is expected: the four magnets sit 7.5 mm under the apex, and the patches are
1-2 mm across. So the patch&#8217;s size cannot be read from the boards, and it needs no correction. New arm presses on
the hand show what does change the reading: where on the pad a flat contact lands (sections 3-5). Where the calibration
has data and the pattern matches, the boards read close to their calibration. A straight finger on a flat face lands
where no calibrated contact explains the field. Separately, the trials&#8217; estimator loses about a quarter of the
force that the boards do read (section 6).</p>"""))
    tb = P.table(["tip, indenter", "holds", "field per N (median, quartiles)", "depth per N (median)"],
                 [[lab, str(r["n"]), f"{r['field_ratio'][1]:.2f} ({r['field_ratio'][0]:.2f}-{r['field_ratio'][2]:.2f})",
                   f"{r['depth_ratio'][1]:.2f}"] for lab, r in rows_cal], num_cols=(1, 2, 3), n=1,
                 caption="Matched holds against the same tip's R 5 sphere: near-normal presses (tangential force under 0.15 of the normal, 0.5 N and over), within 0.75 mm and 30 % of force.")
    S.append(("calibration", "The calibrations already vary the contact's size", f"""
<p>The v3 tips were pressed by spheres of R 2 and R 5 mm and by a rod of R 12.5 mm. This was on the UR5 (tip mode, 10-28
to 10-30) and on the inverted rig (11-02 to 11-04), with the HEX-E as reference. Hertz puts the R 2 sphere&#8217;s patch
at about 0.6 of the R 5 sphere&#8217;s area, and the rod&#8217;s at 1.8-2.2 times it. Yet at the same place and force,
the field per newton stays within about 10 % of the R 5 sphere&#8217;s (Figure 1, Table 1). The depth per newton moves
as Hertz predicts. The field&#8217;s pattern differs little as well: cosine 0.96-0.996 with the sphere&#8217;s, against
0.995-0.999 between repeats. Script {code('sim/20261119-board_patch_size_calibration.py')}.</p>
{P.fig(1, tmp / 'f1.png', 'Field per newton (left) and indentation depth per newton (right) of the R 2 sphere (orange) and the R 12.5 rod (blue), against the R 5 sphere at the same place and force, per tip. Median and quartiles of the matched holds.')}
{tb}"""))
    S.append(("arm", "Arm presses on the hand", f"""
<p>A new tool, {code('lab/arm_press.py')}, has the arm push one pad straight into a plate of the post. The
finger&#8217;s servos hold their targets meanwhile. The arm moves at 1 mm/s to 1.5 mm short of contact, then at 0.2 mm/s
until the force along the press reaches 2.5 N. It stays 2 s and backs off at 0.2 mm/s; each set has three presses. The
HEX-E (100 Hz), the arm, the station and the boards record throughout. No joint is commanded, so the trials&#8217;
own-pose term stays near zero, unlike in the joint-driven presses of 11-18.</p>
<p>The joints still gave 0.6-1.5 deg in extension, with servo loads of 60-100. The fingertip thus yields about 0.45 N per
mm, and 2.5 N took 3-5 mm of arm travel past contact. A free-air run at the park showed no field change from the travel
itself. Single HEX-E samples scatter by 0.2 N, so the stops use 0.1 s means. The press side faces away from the cameras,
so there is no film. The sets: index, middle and thumb with the finger straight, then the index with its PIP bent 10 and
20 deg. The index and middle pressed the taped plate, the thumb the tag plate.</p>"""))
    S.append(("place", "Where the plate lands decides what the boards read", f"""
<p>For each press, a least-squares line of the board&#8217;s 15 components against the HEX-E&#8217;s force gives its
field per newton. Its direction is then compared, by cosine, with every calibrated press of the same tip (Figure 2,
Table 2). Straight fingers read 6-12 uT/N, in patterns no calibrated press shows (cosine 0.36-0.84). The index with its
PIP bent 10 and 20 deg reads 19 and 25 uT/N. Its pattern then matches the calibration at 0.96-0.99, and its size is 0.67
and 0.88-0.92 of the calibration there. Figure 3 shows one press of each kind: the straight index&#8217;s field rises a
third as fast, and less cleanly.</p>
<p>The joint-driven presses of 11-18 agree: their raw field per newton was 0.37-0.68 of the calibration at the GP&#8217;s
point. In their step back, the servo&#8217;s load fell by 200-400 at constant force, and the field moved only 1-3 uT. So
the motors&#8217; current matters little. The finger&#8217;s own motion matters more: 1.7 uT per degree of MCP in free
air. That is what the own-pose term should remove.</p>
{P.fig(2, tmp / 'f2.png', 'Field per HEX-E newton of each arm press (dots), and the same tip&#8217;s calibration at the place whose pattern matches best (bars), with the range of the best cosines.')}
{P.fig(3, tmp / 'f3.png', 'The board&#8217;s change along its own field-per-newton direction against the HEX-E&#8217;s force. One press each, loading and unloading.')}
{P.table(["press", "n", "uT/N", "best cosine", "model z (mm)", "pattern z (mm)", "trials' estimator", "pattern-matched"],
         summary_rows(R, PM), num_cols=(1, 2, 3, 4, 5, 6, 7), n=2,
         caption="The arm presses. z runs along the pad from its proximal mount face (apex 10.86, distal end 21.41). The model's z is where the plate's normal meets the pad at the press's joints before load. The last two columns are force per HEX-E newton.")}"""))
    S.append(("geometry", "Where on the pad: the hand model and the patterns disagree", f"""
<p><b>Corrected the same evening (19:xx).</b> The disagreement below is the normal-only pattern match's, not the boards'. The plates' friction leaned the force 15-26 deg, and a library of normal presses can only explain a leaning force by a more distal place. The trials' GP, which models shear, places the bent index's contacts 1.8-2.8 mm distal of the geometry at the joint readings, and within 1 mm of it with the DIP link 9.2 deg more flexed, as the DIP tags measured on 11-18. See <a href="../20261119-fingertip_sensing/20261119-magnetic_fingertip_contact_sensing_on_the_sr2_hand.html">the fingertip sensing page</a>, Section 9.</p>
<p>The hand model puts each contact where the plate&#8217;s normal meets the pad. For the straight index, that is 3.4 mm
proximal of the apex, on the pad&#8217;s flat proximal half. That half turns only 12 deg over 11 mm, so a few degrees of
lean move a flat contact by several mm. For PIP 10 and 20, the model puts the contact 1.7 and 4.1 mm distal of the apex.
The boards&#8217; patterns match the calibration best near z 19 for both, 4-6 mm more distal than the model
(Figure 4).</p>
<p>One more test sums the calibration&#8217;s point fields over a flat plate&#8217;s patch
({code('sim/20261119-board_patch_superposition.py')}), with a Winkler bed as in the simulator&#8217;s pads. It reproduces
the PIP 20 presses at cosine 0.99, with the patch near z 18-20, and 0.89-0.93 of their field. At the model&#8217;s place
it reaches only 0.86. No patch on the pad reproduces the straight presses: the cosine stays at or under 0.65. So either
the pad sits on the DIP link differently from the CAD, or the straight finger loads something the calibration never did.
The DIP link&#8217;s body, just proximal of the pad, is the first candidate. Tip 3&#8217;s UR5 sphere holds start at
z 4.3 mm; only its rig rod presses reach the proximal end.</p>
{P.fig(4, tmp / 'f4.png', 'Top: the pad&#8217;s profile, with each press set&#8217;s contact place from the hand model (open circles) and from the best-matching calibration pattern (squares above cosine 0.9, crosses below). Bottom: where tip 3&#8217;s calibration holds lie along the pad.')}"""))
    S.append(("estimator", "The estimator loses force the boards still read", f"""
<p>A simpler estimate takes the calibration point whose field per newton is most nearly parallel to the measured field.
The force is then the least-squares scale at that point ({code('sim/20261119-board_force_pattern_match.py')}). On the
PIP 20 presses it reads 0.83-0.87 of the HEX-E; the trials&#8217; GP reads 0.60-0.66 (Figure 5). The GP also adds
0.15-0.3 N at zero force. The best cosine tells when either can be trusted. At 0.99 the linear estimate is within
15 %. Below 0.7, both read about a quarter of the force.</p>
{P.fig(5, tmp / 'f5.png', 'Force per HEX-E newton estimated from each arm press&#8217;s field, against how well any calibrated press of its tip explains that field.')}"""))
    if DW is not None:
        S.append(("hexe", "The HEX-E's own scale", DW["html"]))
    S.append(("next", "Hypotheses to test", """
<ul>
<li><b>Sensing.</b> A contact whose field matches a calibrated press at cosine 0.95 or more should read within about
15 % with the linear estimate. Below 0.7 its force is unknown. Record the best cosine in the trials and compare grasps by
it.</li>
<li><b>Estimator.</b> Replace the GP&#8217;s force with the pattern-matched linear estimate, or retrain the GP so it
cannot shrink a matched field. Try it offline on the trials of 11-16 to 11-19 first, then live.</li>
<li><b>Own pose.</b> Under load the joints give 1-1.5 deg, and the own-pose term then removes contact signal. During a
contact, take that term at the commanded joints, or at the joints before contact.</li>
<li><b>Pad and DIP link.</b> The boards place bent-PIP contacts 4-6 mm more distal than the hand model. A small fixed
indenter at known places on the hand&#8217;s pad would show which is off. So would a measurement of where each pad sits
on its DIP link. Until then, contact places from the boards have that error.</li>
<li><b>Straight fingers on flat faces.</b> Press the straight index with a thin shim between the plate and the DIP
link&#8217;s body. If the field per newton then rises to the calibration&#8217;s, the body was sharing the load.</li>
<li><b>The HEX-E&#8217;s scale.</b> Rest a weighed mass on an upturned pad through a small ball, and compare the boards
with the calibration. That needs no HEX-E (section 7).</li>
</ul>"""))
    files = [F_CAL, F_ARM, F_PM, F_SUP, F_TR, F_HAND18] + ([F_DW] if DW is not None else [])
    S.append(("data", "Data and scripts", "<ul>" + "".join(f"<li>{link(f, P.esc(str(f.relative_to(D))))}</li>" for f in files)
              + "<li>Arm press records: " + code("data/s3/20261119-1*_arm_press.jsonl") + " (raw HEX-E, arm and station "
              "samples, 0.8-4.3 MB each, and the boards&#8217; splitter file stay in corvus&#8217;s "
              + code("logs/20261119-sr2_s3_press/") + " and " + code("logs/20261119-three_finger_mux/") + ")</li>"
              "<li>Lab notes: " + link(D / "NOTES.md", "NOTES.md") + ", section of 2026-11-19 16:36-17:45</li></ul>"))
    toc = "\n".join(f'<li><a href="#{sid}">{k}. {P.esc(hd)}</a></li>' for k, (sid, hd, _) in enumerate(S, 1))
    secs = "\n".join(f'<section id="{sid}"><h2>{k}. {P.esc(hd)}</h2>{body}</section>' for k, (sid, hd, body) in enumerate(S, 1))
    byline = ("UR5 calibrations of the v3 tips (10-28 to 11-04) and arm presses on the SR2 hand at corvus, 2026-11-19 "
              "16:55-17:30, with the owner&#8217;s go of 16:3x. Page by " + code("scripts/fingertip/fingertip_board_force_place_page.py") + ".")
    page = TEMPLATE.read_text()
    for k, v in {"TITLE": "Fingertip boards by contact place", "H1": "Fingertip board readings against the HEX-E by indenter size and by contact place on the SR2 hand's pads",
                 "EYEBROW": "Fingertip sensing &#183; 2026-11-19", "LEDE": lede, "BYLINE": byline, "FILM": "",
                 "TOC": toc, "SECTIONS": secs, "FOOTER": str(PAGE.relative_to(ROOT)), "TEXCSS": ""}.items():
        page = page.replace("{{" + k + "}}", v)
    page = retro_style.apply(page)
    page = docs_index.tag(page, topic="fingertip", summary=summary, date="2026-11-19")
    PAGE.write_text(page)
    print(f"{PAGE.relative_to(ROOT)}: {PAGE.stat().st_size / 1e6:.2f} MB", flush=True)


if __name__ == "__main__":
    build()
