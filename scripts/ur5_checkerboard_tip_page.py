#!/usr/bin/env python3
"""Page of the first checkerboard fingertip pressed on the UR5 (session cb1t, 2026-10-23 evening, QUEUE.md R22): the pole
pattern and the magnets' strength from the field at rest, the registration with the offset touch-offs, the pad's
compliance, the board's response, the tip's own predictors, and sphere2's predictors applied to it.

Reads the analysis outputs under docs/experiments/20261022-ur5_tip_calibration/data/ (paths below are relative to it):

    .venv/bin/python scripts/ur5_tip_calibration_analysis.py lab/20261023-203938-cb1t --gravity-from lab/20261022-195420-first2
    .venv/bin/python scripts/ur5_tip_calibration_fe.py data/20261023-203938-cb1t/holds.npz --plan lab/20261023-203938-cb1t/plan.json \
        --out data/20261023-203938-cb1t/fe.json
    .venv/bin/python scripts/ur5_tip_calibration_predict.py data/20261023-203938-cb1t
    .venv/bin/python scripts/ur5_tip_calibration_pads.py data/20261023-203938-cb1t/holds.npz --plan lab/20261023-203938-cb1t/plan.json \
        --fe-json data/20261023-203938-cb1t/fe.json --shear 1,10,100 --out data/20261023-203938-cb1t/pads.json
    .venv/bin/python scripts/ur5_tip_calibration_tip_transfer.py data/20261022-210852-sphere2 data/20261023-203938-cb1t
and, for the registration of pass 1 (git show 1b520414:.../lab/20261023-203938-cb1t/registration.json, kept as
data/20261023-203938-cb1t-pass1reg/registration_pass1.json), the analysis with --registration-from that file and
--out data/20261023-203938-cb1t-pass1reg, and the FE comparison on its holds. Sphere2's outputs are those of
scripts/ur5_tip_calibration_results_page.py. Writes the figures to media/ and the page
docs/experiments/20261022-ur5_tip_calibration/20261024-checkerboard_fingertip_on_the_ur5.html.

    .venv/bin/python scripts/ur5_checkerboard_tip_page.py
"""
from __future__ import annotations

import datetime
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src/morphohand/driver/reskin5x/host"))
import docs_index  # noqa: E402
import fingertip_wrench_model as W  # noqa: E402
import retro_style  # noqa: E402
import texsvg  # noqa: E402
import ur5_tip_calibration_page as P  # noqa: E402

D = ROOT / "docs/experiments/20261022-ur5_tip_calibration"
DATA, LAB, MEDIA = D / "data", D / "lab", D / "media"
PAGE = D / "20261024-checkerboard_fingertip_on_the_ur5.html"
TEMPLATE = ROOT / "scripts/ur5_tip_calibration_page.template.html"
TEX_CACHE = D / "texsvg_cache.json"
TEX_RE = re.compile(r"\$\$(.+?)\$\$|\$(.+?)\$", re.S)
CB, S2, S3 = "20261023-203938-cb1t", "20261022-210852-sphere2", "20261023-124052-sph3t"
PASS1 = DATA / f"{CB}-pass1reg"
MAIN_PAGE = D / "20261022-ur5_tip_ground_truth_and_predictors.html"
SECOND_PAGE = ROOT / "docs/experiments/20261023-second_fingertip_transfer/20261023-second_fingertip_transfer.html"
QUEUE = D / "QUEUE.md"
NOTES = LAB / "NOTES.md"
PLANNED = (1, -1, 1, -1, 1)          # the owner's "N S (bottom) / N / S N (top)" read from the palm, north toward the palm


def J(p):
    return json.loads(Path(p).read_text())


def load():
    R = {}
    for key, s in (("cb", CB), ("s2", S2)):
        sd = DATA / s
        R[key] = dict(S=J(sd / "summary.json"), cal=J(sd / "calib_rest.json"), FE=J(sd / "fe.json"), PADS=J(sd / "pads.json"),
                      PR=J(sd / "predict/predict.json"))
        z = np.load(sd / "holds.npz", allow_pickle=False)
        R[key]["H"] = {k: z[k] for k in z.files}
    R["TR"] = J(DATA / CB / "predict/transfer.json")
    R["SIM"] = J(DATA / S2 / "predict/magnet_transfer_sim_ADE.json")
    R["p1"] = dict(S=J(PASS1 / "summary.json"), FE=J(PASS1 / "fe.json"), reg=J(PASS1 / "registration_pass1.json"))
    R["reg"] = {k: J(LAB / s / "registration.json") for k, s in (("s2", S2), ("s3", S3), ("cb", CB))}
    R["reg"]["p1"] = R["p1"]["reg"]
    R["S3FE"] = J(DATA / f"{S3}-s2reg/fe.json")
    return R


# ------------------------------------------------------------------------------------------------ rest fits
def rest_fits(cal):
    """The field at rest fitted with the best patterns of the linear prescreen and with the planned checkerboard."""
    board = W.Calib(case=cal["case"], poles=tuple(cal["poles"]), s=cal["s"], theta0=cal["theta0"], perm=tuple(cal["perm"]))
    b = np.asarray(cal["b_rest_uT"], float)
    pats = [q[1] for q in W.prescreen_poles(b, W.fe_row(cal["case"]), board)[:5]]
    if PLANNED not in pats:
        pats.append(PLANNED)
    out = []
    for pl in pats:
        c = W.calibrate_frames_fixed(b, board, poles=pl)
        out.append(dict(poles=pl, rms_mT=c.rms_mT, k=c.k, d=(c.dx, c.dy, c.dz), g=c.g_uT))
    return sorted(out, key=lambda q: q["rms_mT"])


# ------------------------------------------------------------------------------------------------ figures
def fig_poles(R, fits, out):
    plt = P.plt_setup()
    row = W.fe_row(R["cb"]["cal"]["case"])
    yz = np.array([[m[2], m[3]] for m in row["magnets"]], float)
    fig = plt.figure(figsize=(11, 4.8))
    ax = [fig.add_axes([0.07, 0.14, 0.2, 0.74]), fig.add_axes([0.30, 0.14, 0.2, 0.74]),
          fig.add_axes([0.77, 0.14, 0.21, 0.74])]
    for a, (pl, title) in zip(ax[:2], ((PLANNED, "(a) Plan, palm view"), (tuple(R["cb"]["cal"]["poles"]), "(b) Fitted at rest"))):
        for j, (y, z) in enumerate(yz):
            col = "#c0392b" if pl[j] > 0 else "#1f5fbf"
            a.add_patch(plt.Circle((y, z), 1.5, color=col, alpha=0.85))
            a.text(y, z, "N" if pl[j] > 0 else "S", ha="center", va="center", color="white", fontsize=14, weight="bold")
        a.set_xlim(-5, 5), a.set_ylim(2.5, 18.5), a.set_aspect("equal"), a.set_xticks([-4, 0, 4])
        a.set_xlabel("y (mm)"), a.set_title(title)
    ax[0].set_ylabel("z, toward the tip end (mm)")
    lab = [("(" + ", ".join("+1" if v > 0 else "\u22121" for v in f["poles"]) + ")"
            + ("  plan (palm)" if tuple(f["poles"]) == PLANNED else ("  fitted" if i == 0 else ""))) for i, f in enumerate(fits)]
    y = np.arange(len(fits))
    ax[2].barh(y, [f["rms_mT"] for f in fits], color=["#2e8b57" if i == 0 else ("#c0392b" if tuple(f["poles"]) == PLANNED
                                                                               else "#7f7f7f") for i, f in enumerate(fits)])
    ax[2].set_yticks(y, lab, fontsize=12), ax[2].invert_yaxis()
    ax[2].set_xlabel("Residual at rest (mT rms)"), ax[2].set_title("(c) Residual of each pattern")
    fig.savefig(out, dpi=100)
    plt.close(fig)


def fig_levels(R, out):
    plt = P.plt_setup()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
    for key, name, col in (("s2", "sphere2's tip (poles all +1)", "#7f7f7f"), ("cb", "checkerboard tip (cb1t)", "#2e8b57")):
        bl = R[key]["S"]["estimator_by_level"]
        L = sorted(bl, key=float)
        x = [float(v) for v in L]
        ax[0].plot(x, [bl[v]["loc_median_mm"] for v in L], "o-", color=col, label=name)
        ax[1].plot(x, [bl[v]["rel_force_median"] for v in L], "o-", color=col, label=name)
    ax[0].set_ylabel("Location error, median (mm)"), ax[1].set_ylabel("Relative force error, median")
    for a, t in zip(ax, ("(a) Contact point", "(b) Normal force")):
        a.set_xlabel("Force level (N)"), a.set_title(t), a.grid(alpha=0.3)
    ax[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)


def fig_finetune(R, out):
    plt = P.plt_setup()
    TR, PRc = R["TR"], R["cb"]["PR"]
    fig, ax = plt.subplots(figsize=(11, 4.3))
    N = [r["N"] for r in TR["curve"]]
    loc = lambda r, part, m: r[part][m]["loc_mm"]["median"]                 # noqa: E731
    ax.axhline(TR["physical"]["heldout"]["loc_mm"]["median"], color="#2e8b57", ls=":", lw=2,
               label="Physical model from the rest fit, no presses")
    ax.plot(N, [loc(r, "A_and_B", "physical+gp") for r in TR["curve"]], "o-", color="#c0392b",
            label="Physical model + correction, sphere2 and cb1t")
    ax.plot(N, [loc(r, "B_only", "physical+gp") for r in TR["curve"]], "o--", color="#c0392b",
            label="Physical model + correction, cb1t alone")
    ax.plot(N, [loc(r, "A_and_B", "gp raw") for r in TR["curve"]], "s-", color="#1f5fbf",
            label="Gaussian process on the board, sphere2 and cb1t")
    ax.plot(N, [loc(r, "B_only", "gp raw") for r in TR["curve"]], "s--", color="#1f5fbf",
            label="Gaussian process on the board, cb1t alone")
    ax.plot([r["N"] for r in PRc["curve"]], [r["mlp:board"]["loc_mm"]["median"] for r in PRc["curve"]], "^--",
            color="#8e44ad", label="MLP ensemble on the board, cb1t alone")
    ax.set_xscale("log"), ax.set_xticks(N, [str(n) for n in N])
    ax.set_xlabel("cb1t's training presses N"), ax.set_ylabel("Location error, median (mm)")
    ax.grid(alpha=0.3), ax.legend(frameon=False, fontsize=11, ncol=2, loc="upper center")
    ax.set_ylim(0, 7.2)
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ text
def f2(x, n=2):
    return f"{x:.{n}f}"


def sections(R, fits, F):
    cb, s2, TR, SIM = R["cb"], R["s2"], R["TR"], R["SIM"]
    reg = {k: v["notes"]["register"] for k, v in R["reg"].items()}
    cB = {k: np.array(v["c_B"]) for k, v in reg.items()}
    sd1 = reg["p1"]["sd"][12:14]
    rc = cb["cal"]
    best, plan = fits[0], next(f for f in fits if tuple(f["poles"]) == PLANNED)
    nxt = fits[1]["rms_mT"]
    k_ratio = rc["k"] / s2["cal"]["k"]
    bmax = lambda c: max(np.linalg.norm(v) for v in c["b_rest_uT"]) / 1e3          # noqa: E731
    fe_c, fe_s, fe_1 = cb["FE"]["series"], s2["FE"]["series"], R["p1"]["FE"]["series"]
    s3s = R["S3FE"]["series"]
    bm2, bm1 = (np.array(R[k]["S"]["board_minus_registration_mm"]["mean"]) for k in ("cb", "p1"))
    bms2 = np.array(s2["S"]["board_minus_registration_mm"]["mean"])
    hi = lambda S: [S["estimator_by_level"][L]["loc_median_mm"] for L in S["estimator_by_level"] if float(L) >= 2]  # noqa: E731
    lo = lambda S: [S["estimator_by_level"][L]["loc_median_mm"] for L in S["estimator_by_level"] if float(L) <= 0.5]  # noqa: E731
    vw, vw2 = cb["PR"]["viewer"], s2["PR"]["viewer"]
    lastc, lasts = cb["PR"]["curve"][-1], s2["PR"]["curve"][-1]
    H = cb["H"]
    press = (H["kind"] == "press") & np.isfinite(H["f_n"])
    turn = np.isin(H["kind"], ["roll", "twist", "slide"])

    def resp(Hh, region):
        m = (Hh["kind"] == "press") & (Hh["region"] == region) & (Hh["level_N"] >= 1) & (Hh["level_N"] <= 2) & (Hh["f_n"] > 0.15)
        return float(np.median(np.linalg.norm(Hh["d_board"][m], axis=1) / Hh["f_n"][m]))

    rg = {r: (resp(H, r), resp(s2["H"], r)) for r in ("face", "edge", "flank")}
    span = lambda Hh: " to ".join(datetime.datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M")    # noqa: E731
                                  for t in (np.nanmin(Hh["t"]), np.nanmax(Hh["t"])))
    ftd = lambda Hh: (float(np.nanmedian(Hh["ft_rest_drift_N"])), float(np.nanmax(Hh["ft_rest_drift_N"])))  # noqa: E731
    ft_c, ft_s = ftd(H), ftd(s2["H"])
    S_ = []

    # ---- terms
    terms = [
        ("Press, hold and turn hold", "A press moves the pad onto the indenter, a sphere of radius 5 mm, along the normal at a "
         "target point of the pad. It stops at force levels from 0.25 to 5 N for about 1.5 s each. A hold is one level of "
         "a press. A turn hold is a level of a roll or a twist, where the tool turns while it presses."),
        ("Touch-off and registration", "A touch-off approaches the sphere slowly and records where the force begins. A plain "
         "touch-off aims at a point of the pad along its normal; an offset touch-off aims beside it, so that the sphere "
         "meets the pad on the side of its tip (" + F["code"]("scripts/ur5_tip_calibration_offset_touches.py") + "). The "
         "registration fits the pad's frame on the flange and the sphere's centre to the touch-offs, in mm and deg."),
        ("Ground truth", "The contact point $\\mathbf{p}$ and the normal $\\mathbf{n}$ follow from the flange's pose, the "
         "registration and the sphere's radius. The normal force $f_n$ is the HEX-E's force along $-\\mathbf{n}$, in N, "
         "with the tool's weight and the drift of the zero removed. The weight comes from the gravity fit of session "
         "first2 (2026-10-22)."),
        ("Pole pattern", "The direction of each magnet's moment, $+1$ or $-1$, in the model's order of the five magnets "
         "(two at each end of the pad and one at its centre). The magnets of sphere2's tip are all $+1$."),
        ("Strength $k$", "The magnets' magnetization over that of an N52 magnet of the model's size (3 mm diameter, 2 mm "
         "long) at the model's position."),
        ("Rest fit", "The model of the five chips' field at rest, fitted to the reading at rest. The chips' frames are "
         "kept from the board's calibration. The fit gives the strength $k$, the magnets' offsets $d_x, d_y, d_z$ (mm, $x$ "
         "along the pad's depth) and a uniform field. Its residual is the rms over the 15 components of the reading minus the model, in "
         "mT. Each of the 32 pole patterns gets its own fit."),
        ("Board change and response", "The board change is the mean field of the five chips over a hold minus the field at "
         "rest, 15 values in µT. The response is the length of the board change over $f_n$, in µT/N."),
        ("Physical model", "The contact viewer's surface estimator: the rest fit and the FE model's magnet motions map a "
         "board change to a contact point and a normal force. It uses no press. Fitted, $k$, the offsets and a compliance "
         "scale $c$ are refitted to training presses (" + F["code"]("scripts/ur5_tip_calibration_predict.py") + ")."),
        ("Learned predictors", "A Gaussian process or an MLP ensemble from the board change to the contact point and $f_n$. "
         "The correction of the physical model is a Gaussian process whose inputs are the physical estimate itself (point "
         "and force) and whose outputs are its corrections. The predictors on magnet motions take the five magnets' "
         "translations, solved from the board change through each tip's own field model."),
        ("Training presses $N$ and fine-tuning", "A predictor is trained on the holds of the first $N$ presses in time order "
         "and scored on the held-out presses, every fourth grid press. Fine-tuned, it is trained on all of sphere2's "
         "presses and the first $N$ of cb1t's."),
        ("Location error and force error", "The location error is the distance between an estimated and the true contact "
         "point, tangential to the pad, in mm. The force error is $|\\hat f_n - f_n|$ in N, and the relative force error "
         "$|\\hat f_n / f_n - 1|$. Each score is a median over holds."),
        ("FE scale $s$ and $\\Delta$", "The FE pad's compliance is multiplied by $s$ to fit the measured force. "
         "$\\Delta$ is the indentation beyond the FE pad at $s = 1$, fitted as $\\Delta = a + c\\,f_n$: $a$ in mm, the "
         "compliance in series $c$ in mm/N."),
        ("Sphere pads and $E_\\mathrm{meas}$", "The simulations' contact model, springs $K_i = E\\,a_i / h_i$ on the "
         "pad's 812 surface points; $E_\\mathrm{meas}$ in MPa fits their force to the HEX-E's by the median relative error."),
    ]
    S_.append(("terms", "Terms and metrics",
               '<dl class="terms">' + "".join(f"<dt>{t}</dt><dd>{d}</dd>" for t, d in terms) + "</dl>"))

    # ---- session
    n_turn = int(turn.sum())
    t1 = [["Tip", "board 9c:d0, poles all +1", "board d9:94, the first checkerboard"],
          ["Holds recorded", span(s2["H"]), span(H)],
          ["Touch-offs (plain, offset)", f"{reg['s2']['n_pad']}, 0", f"19, {reg['cb']['n_pad'] - 19}"],
          ["Registration residual (mm rms)", f2(reg["s2"]["rms_pad_mm"], 3), f2(reg["cb"]["rms_pad_mm"], 3)],
          ["Press holds (presses)", f"{int((s2['H']['kind'] == 'press').sum())} ({len(np.unique(s2['H']['id'][s2['H']['kind'] == 'press']))})",
           f"{int(press.sum())} ({len(np.unique(H['id'][press]))})"],
          ["Turn holds", f"{int(np.isin(s2['H']['kind'], ['roll', 'twist', 'slide']).sum())}", f"{n_turn}"],
          ["HEX-E zero between rests, median and largest (N)", f"{f2(ft_s[0], 3)}, {f2(ft_s[1], 3)}", f"{f2(ft_c[0], 3)}, {f2(ft_c[1], 3)}"]]
    body = ("<p>The robot ran mount v3's fork, the one of sphere2 and sph3t, with the first checkerboard tip on the "
            "workstation's board d9:94 (" + F["link"](NOTES, "lab notes") + ", sections on cb1t). The gravity sweep was "
            "left out at the owner's request. The first attempt aborted twice when the board's USB cable pulled on the "
            "tool, and the owner freed it.</p>"
            "<p>Pass 1 ran 19 plain touch-offs from 20:46 to 21:27. Pass 2 ran 12 offset touch-offs from 21:56 to 22:27. "
            "From 22:40 to 23:58 it ran 59 grid presses, six reference presses, 2 rolls, 6 twists and a final reference "
            "press, with no abort. Eight grid presses with joint 6 above +60 deg were left out. The new registration put "
            "six actions within the mount's 3.0 mm margin: grid-002, -007, -024, -025 and twist-1 at 1.0 and 2.5 N. The "
            "simulator's cell check put two more there (grid-027, -028), and all eight were dropped.</p>"
            f"<p>Between the two rests of a press the HEX-E's zero moved a median {ft_c[0]:.3f} N and at most "
            f"{ft_c[1]:.3f} N (Table 1). A cable pulling through a press would move it further. The reference press met the "
            f"sphere 1.552 to 1.631 mm past its rest, a drift of +0.054 mm over 77 min.</p>"
            + P.table(["", "sphere2", "cb1t"], t1, caption="The two UR5 sessions compared on this page.", n=1))
    S_.append(("session", "Session and tip", body))

    # ---- poles
    t2 = [["(" + ", ".join("+1" if v > 0 else "&#8722;1" for v in f["poles"]) + ")" + (" plan read from the palm" if tuple(f["poles"]) == PLANNED else ""),
           f2(f["rms_mT"], 3), f2(f["k"], 3), ", ".join(f2(v) for v in f["d"])] for f in fits]
    body = (f"<p>Fitted to the field at rest, the pattern (+1, &#8722;1, &#8722;1, &#8722;1, +1) leaves {best['rms_mT']:.3f} mT "
            f"rms (Table 2, Fig. 1). The next pattern leaves {nxt:.3f} mT. In the model, +1 is a north pole toward the palm, "
            f"as in all of sphere2's magnets. Seen from the palm with the tip end up, the bottom-left and top-right magnets "
            f"face the palm with north, and the centre, bottom-right and top-left magnets with south.</p>"
            f"<p>The plan was &#8220;N S&#8221; at the bottom, &#8220;N&#8221; in the middle and &#8220;S N&#8221; at the top. "
            f"Read from the palm, with the poles toward the palm, it is (+1, &#8722;1, +1, &#8722;1, +1). That pattern leaves "
            f"{plan['rms_mT']:.3f} mT, with its strength and offsets at the bounds of the fit, and it differs from the fitted "
            f"one at the centre magnet. Read from the board's side, with the poles toward the board, the plan is the fitted "
            f"pattern. The centre magnet is therefore reversed only if the plan was read from the palm.</p>"
            f"<p>The fitted strength is $k$ = {rc['k']:.3f}, {k_ratio:.2f} times the {s2['cal']['k']:.3f} of sphere2's "
            f"magnets. The fit also moves the magnets {rc['dx']:.2f} mm along the pad's depth, against {s2['cal']['dx']:.2f} "
            f"mm on sphere2's tip. The largest field at a chip is {bmax(rc):.1f} mT, against {bmax(s2['cal']):.1f} mT on "
            f"sphere2's tip. Next to a magnet of opposite pole, part of each magnet's field at the chips cancels.</p>"
            + P.table(["Pole pattern", "Residual (mT rms)", "Strength k", "Offsets d_x, d_y, d_z (mm)"], t2, num_cols=(1, 2),
                      caption="The field at rest of cb1t's tip fitted with the five best patterns of the linear prescreen "
                              "and the plan read from the palm (fingertip_wrench_model.calibrate_frames_fixed).", n=2)
            + F["fig"](1, MEDIA / "20261024-cb1t_pole_pattern.png",
                       "Pole pattern of cb1t's tip, seen from the palm with the tip end up; N (red) is a north pole toward "
                       "the palm, the direction of all of sphere2's magnets. (a) The plan read from the palm and (b) the "
                       "pattern fitted to the field at rest. (c) Residual of the rest fit for the five best patterns and the "
                       "plan read from the palm."))
    S_.append(("poles", "Pole pattern and magnet strength from the field at rest", body))

    # ---- registration
    d_cc = float(np.linalg.norm(cB["cb"] - cB["p1"]))
    t3 = [["Touch-offs", "19", f"{reg['cb']['n_pad']}"],
          ["Registration residual (mm rms)", f2(reg["p1"]["rms_pad_mm"], 3), f2(reg["cb"]["rms_pad_mm"], 3)],
          ["Residual of $\\Delta = a + c\\,f_n$ over the grid (µm rms)", f"{fe_1['rms_mm'] * 1e3:.0f}", f"{fe_c['rms_mm'] * 1e3:.0f}"],
          ["FE scale s; its median force error (%)", f"{R['p1']['FE']['s_best']:.2f}; {R['p1']['FE']['median_rel_err_best'] * 100:.1f}",
           f"{cb['FE']['s_best']:.2f}; {cb['FE']['median_rel_err_best'] * 100:.1f}"],
          ["Board's point minus the registered one at 3 N and over, mean length (mm)", f2(np.linalg.norm(bm1)), f2(np.linalg.norm(bm2))],
          ["Physical model's location error at 2 N and over (mm)", f"{min(hi(R['p1']['S'])):.2f} to {max(hi(R['p1']['S'])):.2f}",
           f"{min(hi(cb['S'])):.2f} to {max(hi(cb['S'])):.2f}"],
          ["Sphere's centre from sphere2's fit (mm)", f2(np.linalg.norm(cB["p1"] - cB["s2"])), f2(np.linalg.norm(cB["cb"] - cB["s2"]))]]
    body = (f"<p>With the 19 plain touch-offs of pass 1, the registration fitted at {reg['p1']['rms_pad_mm']:.3f} mm rms. "
            f"With the 12 offset touch-offs added, it fitted all 31 at {reg['cb']['rms_pad_mm']:.3f} mm rms. The pad's frame "
            f"turned {reg['cb']['tcp_change_mm_deg'][1]:.2f} deg and the sphere's centre moved {d_cc:.2f} mm. The lab's "
            f"simulator had predicted 0.89 deg and 0.27 mm. The plain touch-offs left the centre uncertain by "
            f"{min(sd1):.2f} to {max(sd1):.2f} mm sideways (one standard deviation), and the move is "
            f"{d_cc / max(sd1):.1f} times that.</p>"
            "<p>Three measurements that neither registration used were compared (Table 3). The grid presses were analysed "
            f"with each registration. With pass 2's, the indentation beyond the FE pad follows one line to within "
            f"{fe_c['rms_mm'] * 1e3:.0f} µm rms, against {fe_1['rms_mm'] * 1e3:.0f} µm with pass 1's. The FE pad's force "
            f"error falls from {R['p1']['FE']['median_rel_err_best'] * 100:.1f} to {cb['FE']['median_rel_err_best'] * 100:.1f} %. "
            f"The board's estimate of the contact point does not depend on the registration. At 3 N and over it lies a mean "
            f"{np.linalg.norm(bm2):.2f} mm from pass 2's contact points and {np.linalg.norm(bm1):.2f} mm from pass 1's. "
            "The other sections use pass 2's registration.</p>"
            f"<p>Pass 2 puts the sphere's centre {np.linalg.norm(cB['cb'] - cB['s2']):.2f} mm from sphere2's fit, and pass 1 "
            f"{np.linalg.norm(cB['p1'] - cB['s2']):.2f} mm. Sphere2's and sph3t's fits lie "
            f"{np.linalg.norm(cB['s3'] - cB['s2']):.2f} mm apart. Both were made with plain touch-offs only.</p>"
            + P.table(["", "Pass 1", "Pass 2"], t3, num_cols=(1, 2),
                      caption="Registrations of cb1t from the touch-offs of pass 1 and of both passes, and the grid "
                              "presses analysed with each.", n=3))
    S_.append(("registration", "Registration of the tool with the offset touch-offs", body))

    # ---- compliance
    t4 = [["FE scale s (median force error, %)", f"{s2['FE']['s_best']:.2f} ({s2['FE']['median_rel_err_best'] * 100:.1f})",
           f"{cb['FE']['s_best']:.2f} ({cb['FE']['median_rel_err_best'] * 100:.1f})"],
          ["$a$ (mm)", f2(fe_s["a_mm"], 3), f2(fe_c["a_mm"], 3)],
          ["$c$ (mm/N)", f2(fe_s["c_mm_N"], 3), f2(fe_c["c_mm_N"], 3)],
          ["Residual of $\\Delta$ (µm rms)", f"{fe_s['rms_mm'] * 1e3:.0f}", f"{fe_c['rms_mm'] * 1e3:.0f}"],
          ["$E_\\mathrm{meas}$ of the sphere pads, MPa (median force error, %)",
           f"{s2['PADS']['E_meas_MPa']:.2f} ({s2['PADS']['median_rel_err_meas'] * 100:.1f})",
           f"{cb['PADS']['E_meas_MPa']:.2f} ({cb['PADS']['median_rel_err_meas'] * 100:.1f})"]]
    body = (f"<p>The FE pad's compliance has to be multiplied by $s$ = {cb['FE']['s_best']:.2f} to give the measured "
            f"force on cb1t's pad (median error {cb['FE']['median_rel_err_best'] * 100:.1f} %). On sphere2's pad it was "
            f"{s2['FE']['s_best']:.2f} (Table 4).</p>"
            f"<p>The indentation beyond the FE pad is $\\Delta$ = {fe_c['a_mm']:.3f} mm + {fe_c['c_mm_N']:.3f} mm/N "
            f"$f_n$. On sphere2 it was {fe_s['a_mm']:.3f} mm + {fe_s['c_mm_N']:.3f} mm/N, and on sph3t {s3s['a_mm']:.3f} mm "
            f"+ {s3s['c_mm_N']:.3f} mm/N. On one fork, with two tips and over two nights, the compliance in series stays "
            f"within {abs(1 - fe_c['c_mm_N'] / max(fe_s['c_mm_N'], s3s['c_mm_N'])) * 100:.0f} %.</p>"
            f"<p>The sphere pads fit the HEX-E's force at $E_\\mathrm{{meas}}$ = {cb['PADS']['E_meas_MPa']:.2f} MPa, with "
            f"a median error of {cb['PADS']['median_rel_err_meas'] * 100:.1f} %. On sphere2's pad they needed "
            f"{s2['PADS']['E_meas_MPa']:.2f} MPa.</p>"
            + P.table(["", "sphere2", "cb1t"], t4, num_cols=(1, 2),
                      caption="The pads against the FE model and the sphere pads, over the press holds above 0.2 N.", n=4))
    S_.append(("compliance", "Pad compliance against the FE model", body))

    # ---- board and physical model
    pp = lastc["physical_params"]
    body = (f"<p>Per newton, the board changes about as much as on sphere2's tip. At 1 to 2 N the response is "
            f"{rg['face'][0]:.1f}, {rg['edge'][0]:.1f} and {rg['flank'][0]:.1f} µT/N on the face, edges and flanks. On "
            f"sphere2's tip it was {rg['face'][1]:.1f}, {rg['edge'][1]:.1f} and {rg['flank'][1]:.1f} µT/N. Between the two "
            f"rests of a press the board drifted a median {cb['S']['rest_drift_uT']['median']:.1f} µT, against "
            f"{s2['S']['rest_drift_uT']['median']:.1f} µT.</p>"
            f"<p>Built from the rest fit alone, the physical model places the held-out contacts a median "
            f"{vw['loc_mm']['median']:.2f} mm from the registered points, with a force error of "
            f"{vw['force_N']['median']:.3f} N. On sphere2's tip it gave {vw2['loc_mm']['median']:.2f} mm and "
            f"{vw2['force_N']['median']:.3f} N. Over all press holds at 2 N and over, its error is "
            f"{min(hi(cb['S'])):.2f} to {max(hi(cb['S'])):.2f} mm by level, against {min(hi(s2['S'])):.2f} to "
            f"{max(hi(s2['S'])):.2f} mm (Fig. 2). At 0.25 and 0.5 N it misses by {min(lo(cb['S'])):.1f} to "
            f"{max(lo(cb['S'])):.1f} mm, as on sphere2's tip. At 1 N it misses by "
            f"{cb['S']['estimator_by_level']['1']['loc_median_mm']:.2f} mm, against "
            f"{s2['S']['estimator_by_level']['1']['loc_median_mm']:.2f} mm on sphere2's tip.</p>"
            f"<p>Fitted to the training presses, the physical model moves $d_x$ from {rc['dx']:.2f} to {pp['dx']:.2f} mm, "
            f"with a compliance scale of {pp['c']:.2f}. Its location error grows to {lastc['physical']['loc_mm']['median']:.2f} mm, "
            f"and its force error is {lastc['physical']['force_N']['median']:.3f} N. On sphere2 the same fit took the error "
            f"from {vw2['loc_mm']['median']:.2f} to {lasts['physical']['loc_mm']['median']:.2f} mm.</p>"
            + F["fig"](2, MEDIA / "20261024-cb1t_physical_model_by_level.png",
                       "Physical model built from each tip's rest fit, over all press holds, by force level: (a) median "
                       "location error, (b) median relative force error."))
    S_.append(("physical", "Board response and the physical model", body))

    # ---- own predictors
    rows5 = []
    for key, name in (("physical", "Physical model, fitted"), ("gp:board", "Gaussian process on the board"),
                      ("mlp:board", "MLP ensemble on the board"), ("ridge:board", "Ridge regression on the board"),
                      ("physical+gp:board", "Physical model and a Gaussian process on its residual")):
        a_, b_ = lasts[key], lastc[key]
        rows5.append([name, f2(a_["loc_mm"]["median"]), f2(a_["force_N"]["median"], 3), f2(b_["loc_mm"]["median"]),
                      f2(b_["force_N"]["median"], 3)])
    rows5.insert(0, ["Physical model from the rest fit", f2(vw2["loc_mm"]["median"]), f2(vw2["force_N"]["median"], 3),
                     f2(vw["loc_mm"]["median"]), f2(vw["force_N"]["median"], 3)])
    tc, ts = cb["PR"]["turns"], s2["PR"]["turns"]
    body = (f"<p>Trained on {lastc['N']} presses ({lastc['holds']} holds), the Gaussian process on the board places "
            f"cb1t's held-out contacts within {lastc['gp:board']['loc_mm']['median']:.2f} mm and "
            f"{lastc['gp:board']['force_N']['median']:.3f} N (Table 5). The MLP ensemble reaches "
            f"{lastc['mlp:board']['loc_mm']['median']:.2f} mm and {lastc['mlp:board']['force_N']['median']:.3f} N. "
            f"On sphere2's tip, from {lasts['N']} presses, they reached {lasts['gp:board']['loc_mm']['median']:.2f} and "
            f"{lasts['mlp:board']['loc_mm']['median']:.2f} mm.</p>"
            f"<p>On the {tc['n_holds']} turn holds, with every press for training, the Gaussian process gives "
            f"{tc['gp:board']['loc_mm']['median']:.2f} mm and {tc['gp:board']['force_N']['median']:.3f} N. On sphere2's "
            f"{ts['n_holds']} it gave {ts['gp:board']['loc_mm']['median']:.2f} mm and {ts['gp:board']['force_N']['median']:.3f} N.</p>"
            + P.table(["Predictor", "sphere2, location (mm)", "sphere2, force (N)", "cb1t, location (mm)", "cb1t, force (N)"],
                      rows5, num_cols=(1, 2, 3, 4),
                      caption=f"Each tip's predictors trained on its own presses (sphere2 {lasts['N']}, cb1t {lastc['N']}) "
                              f"and scored on its held-out presses ({s2['PR']['n_test_holds']} and {cb['PR']['n_test_holds']} "
                              "holds). Medians.", n=5))
    S_.append(("own", "Predictors trained on the tip's own presses", body))

    # ---- transfer
    A_ = TR["A_only"]
    sc = lambda d, m: (d[m]["loc_mm"]["median"], d[m]["force_N"]["median"])       # noqa: E731
    cur = {r["N"]: r for r in TR["curve"]}
    simE, simD = SIM["scores"].get("E", {}), SIM["scores"].get("D", {})
    names = [("physical+gp", "Correction of the physical model", "physical+gp (trained on A)"),
             ("gp raw", "Gaussian process on the board", "gp raw (trained on A)"),
             ("gp motions", "Gaussian process on magnet motions", "gp on magnet motions (trained on A)"),
             ("mlp motions", "MLP ensemble on magnet motions", "mlp on magnet motions (trained on A)")]
    rows6 = [["Physical model from the rest fit, no presses", f2(TR["physical"]["heldout"]["loc_mm"]["median"]),
              f2(TR["physical"]["all"]["loc_mm"]["median"]), "&#8211;", "&#8211;", "&#8211;",
              f2(simE["physical"]["loc_mm"]) if simE else "", f2(simD["physical"]["loc_mm"]) if simD else ""]]
    Ns = [n for n in (10, 20) if n in cur] + [max(cur)]
    for key, name, sk in names:
        rows6.append([name, f2(sc(A_["heldout"], key)[0]), f2(sc(A_["all"], key)[0])]
                     + [f"{cur[n]['A_and_B'][key]['loc_mm']['median']:.2f} / {cur[n]['B_only'][key]['loc_mm']['median']:.2f}"
                        for n in Ns]
                     + [f2(simE[sk]["loc_mm"]) if simE and sk in simE else "", f2(simD[sk]["loc_mm"]) if simD and sk in simD else ""])
    pg = sc(A_["heldout"], "physical+gp")
    b10, b20 = cur.get(10), cur.get(20)
    mlpc = {r["N"]: r["mlp:board"]["loc_mm"]["median"] for r in cb["PR"]["curve"]}
    body = (f"<p>Trained on sphere2's {TR['n_A_holds']} press holds alone, no learned predictor places cb1t's contacts "
            f"closer than the physical model's {TR['physical']['heldout']['loc_mm']['median']:.2f} mm (Table 6). On the "
            f"held-out holds the Gaussian process on the board misses by {sc(A_['heldout'], 'gp raw')[0]:.2f} mm. The "
            f"predictors on magnet motions miss by {sc(A_['heldout'], 'gp motions')[0]:.2f} mm (Gaussian process) and "
            f"{sc(A_['heldout'], 'mlp motions')[0]:.2f} mm (MLP ensemble). Sphere2's correction of the physical model "
            f"takes cb1t's error from {TR['physical']['heldout']['loc_mm']['median']:.2f} to {pg[0]:.2f} mm.</p>"
            f"<p>The simulation of the pattern (+1, &#8722;1, +1, &#8722;1, +1) ({F['code']('scripts/fingertip_magnet_transfer_sim.py')}) "
            f"predicted the failures of the other three. It also predicted that the correction lowers the error, to "
            f"{simE['physical+gp (trained on A)']['loc_mm']:.2f} mm with today's magnets. In the simulation, sphere2's offset "
            f"between the board's point and the registered one was part of the pad. At 3 N and over that offset was "
            f"({', '.join(f2(v) for v in bms2)}) mm on sphere2 and is ({', '.join(f2(v) for v in bm2)}) mm on cb1t. A "
            "correction that removes sphere2's offset lengthens cb1t's.</p>"
            + (f"<p>Fine-tuned on sphere2's presses and cb1t's first 10 and 20, the correction of the physical model reaches "
               f"{b10['A_and_B']['physical+gp']['loc_mm']['median']:.2f} and {b20['A_and_B']['physical+gp']['loc_mm']['median']:.2f} mm "
               f"(Fig. 3). Trained on cb1t's presses alone, the MLP ensemble on the board reaches {mlpc.get(10, float('nan')):.2f} "
               f"and {mlpc.get(20, float('nan')):.2f} mm at the same $N$. With all {max(cur)} of cb1t's training presses, "
               f"cb1t's own presses alone give the lowest errors.</p>" if b10 and b20 else "")
            + P.table(["Predictor", "Trained on sphere2, cb1t held out (mm)", "Trained on sphere2, all of cb1t (mm)"]
                      + [f"N = {n}, fine-tuned / cb1t alone (mm)" for n in Ns]
                      + ["Simulated checkerboard, today's magnets (mm)", "Simulated checkerboard, N52 (mm)"],
                      rows6, num_cols=tuple(range(1, 6 + len(Ns) - 1)),
                      caption="Median location error on cb1t's held-out holds unless marked, for predictors trained on "
                              "sphere2's presses, fine-tuned with cb1t's first N, or trained on cb1t's alone; and the "
                              "simulation's prediction for the pattern (+1, &#8722;1, +1, &#8722;1, +1), which differs "
                              "from cb1t's at the centre magnet (sets E and D of "
                              "magnet_transfer_sim_ADE.json).", n=6)
            + F["fig"](3, MEDIA / "20261024-cb1t_fine_tuning.png",
                       "Median location error on cb1t's held-out holds against the number of cb1t's training presses: "
                       "trained on sphere2's presses and cb1t's (solid) and on cb1t's alone (dashed)."))
    S_.append(("transfer", "Transfer of sphere2's predictors and fine-tuning", body))

    # ---- limitations, future work
    body = (f"<p>One checkerboard tip was pressed. Whether its centre magnet follows the plan depends on the side from "
            f"which the plan's poles were named (Section 3). The held-out set has "
            f"{cb['PR']['n_test_presses']} presses ({cb['PR']['n_test_holds']} holds). The {n_turn} turn holds take the tool's "
            "weight from first2's gravity fit, made on 2026-10-22 with another tip and without the USB cable taped to the "
            f"tool. Along the pad's depth, the rest fit moves the magnets {rc['dx']:.2f} mm from the model's position, "
            f"and the fit to the presses {pp['dx']:.2f} mm. Part of the physical model's error is therefore in its "
            f"geometry.</p>")
    S_.append(("limits", "Limitations", body))
    fw = [
        "Check each magnet's pole at the palm face with a compass. The fit predicts north toward the palm at the "
        "bottom-left and top-right magnets, seen from the palm with the tip end up, and south at the other three. Any other "
        "result means that the chips' frames of the board's calibration are wrong. If the plan meant north at the centre "
        "seen from the palm, turn the centre magnet and fit the field at rest again. The pattern (+1, &#8722;1, +1, "
        "&#8722;1, +1) should then fit near 0.24 mT.",
        "Run the 12 offset touch-offs in every session (" + F["code"]("scripts/ur5_tip_calibration_offset_touches.py")
        + "). If sphere2's tip is pressed again with them, an offset between the board's point and the registered one "
        "below 0.8 mm, down from 1.4 mm, would place sphere2's offset in its registration.",
        "Fit the depth and spacing of the new pad's magnets to the rest and the presses together, in the physical model of "
        + F["code"]("scripts/ur5_tip_calibration_predict.py") + " with d_x free. If its error at 2 N and over falls below "
        "0.8 mm, the remaining error of the physical model is in its geometry.",
        "Rest the tool at the four orientations of the rolls, with joint 6 within &#177;30 deg, in the next session. A "
        "difference above 0.05 N between the HEX-E and first2's gravity fit would call for the tool's own fit before the "
        "turn holds' forces are used.",
        "Apply cb1t's correction of the physical model, trained on its 47 presses, to the next checkerboard tip after its "
        "rest fit. An error below 1.5 mm without presses would show a correction that passes between tips of one pattern.",
    ]
    S_.append(("future", "Future work", "<ol>" + "".join(f"<li>{x}</li>" for x in fw) + "</ol>"))

    lede = (f"<p>The first checkerboard fingertip was pressed on the UR5 on 2026-10-23 (session cb1t), at {int(press.sum())} "
            f"press holds and {n_turn} turn holds against the HEX-E. Of the 32 pole patterns, one fits its field at rest to "
            f"{best['rms_mT']:.2f} mT, and the next leaves {nxt:.2f} mT. It is the planned checkerboard if the plan's poles "
            f"were named from the board's side. Named from the palm, the plan has the centre magnet the other way round.</p>"
            f"<p>Built from that fit alone, the physical model places held-out contacts a median "
            f"{vw['loc_mm']['median']:.2f} mm from the registered points, against {vw2['loc_mm']['median']:.2f} mm on "
            f"sphere2's tip. At 2 N and over the error is {min(hi(cb['S'])):.1f} to {max(hi(cb['S'])):.1f} mm. Predictors "
            f"trained on sphere2's tip miss by {min(sc(A_['heldout'], k)[0] for k, _, _ in names):.1f} to "
            f"{max(sc(A_['heldout'], k)[0] for k, _, _ in names):.1f} mm. Trained on {lastc['N']} of cb1t's own presses, a "
            f"Gaussian process reaches {lastc['gp:board']['loc_mm']['median']:.2f} mm and an MLP ensemble "
            f"{lastc['mlp:board']['loc_mm']['median']:.2f} mm.</p>")
    summary = (f"On the first checkerboard fingertip (UR5 session cb1t, 2026-10-23), one pole pattern fits the field at rest: "
               f"the planned checkerboard if its poles were named from the board's side, or the plan with its centre magnet "
               f"reversed if they were named from the palm. Built from that fit alone, its physical model places held-out contacts "
               f"within {vw['loc_mm']['median']:.2f} mm (sphere2's tip {vw2['loc_mm']['median']:.2f} mm); sphere2's predictors "
               f"miss by {min(sc(A_['heldout'], k)[0] for k, _, _ in names):.1f} to "
               f"{max(sc(A_['heldout'], k)[0] for k, _, _ in names):.1f} mm, and {lastc['N']} of its own presses give "
               f"{lastc['mlp:board']['loc_mm']['median']:.2f} to {lastc['gp:board']['loc_mm']['median']:.2f} mm.")
    return S_, lede, summary


# ------------------------------------------------------------------------------------------------ the page
def build():
    R = load()
    fits = rest_fits(R["cb"]["cal"])
    MEDIA.mkdir(parents=True, exist_ok=True)
    fig_poles(R, fits, MEDIA / "20261024-cb1t_pole_pattern.png")
    fig_levels(R, MEDIA / "20261024-cb1t_physical_model_by_level.png")
    fig_finetune(R, MEDIA / "20261024-cb1t_fine_tuning.png")
    F = dict(fig=P.fig, table=P.table, link=P.link, code=P.code)
    S, lede, summary = sections(R, fits, F)
    toc = "\n".join(f'<li><a href="#{sid}">{k}. {P.esc(hd)}</a></li>' for k, (sid, hd, _) in enumerate(S, 1))
    secs = "\n".join(f'<section id="{sid}"><h2>{k}. {P.esc(hd)}</h2>{body}</section>' for k, (sid, hd, body) in enumerate(S, 1))
    built = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    byline = (f"Workstation analysis of the lab's run {P.esc(CB)}, built {built} from {P.code('data/' + CB + '/')}, "
              f"{P.code('data/' + PASS1.name + '/')} and {P.code('data/' + S2 + '/')} by "
              f"{P.code('scripts/ur5_checkerboard_tip_page.py')}. Sphere2's session: {P.link(MAIN_PAGE, 'ground truth and predictors')}; "
              f"the desk test of a second tip: {P.link(SECOND_PAGE, 'second fingertip')}; requests to the lab: "
              f"{P.link(QUEUE, 'QUEUE.md')}.")
    page = TEMPLATE.read_text()
    for k, v in {"TITLE": "Checkerboard fingertip on the UR5",
                 "H1": "Pole pattern, registration and contact predictors of the first checkerboard fingertip on the UR5",
                 "EYEBROW": "Printed fingertips and force sensing &#183; 2026-10-24", "LEDE": lede, "BYLINE": byline,
                 "FILM": "", "TOC": toc, "SECTIONS": secs, "FOOTER": str(PAGE.relative_to(ROOT)),
                 "TEXCSS": texsvg.CSS}.items():
        page = page.replace("{{" + k + "}}", v)
    items = [((m.group(1) if m.group(1) is not None else m.group(2)).strip(), m.group(1) is not None)
             for m in TEX_RE.finditer(page)]
    svgs = iter(texsvg.render(items, cache_path=TEX_CACHE))
    page = TEX_RE.sub(lambda m: next(svgs), page)
    page = retro_style.apply(page)
    page = docs_index.tag(page, topic="fingertip", summary=summary)
    PAGE.write_text(page)
    print(f"{PAGE.relative_to(ROOT)}: {PAGE.stat().st_size / 1e6:.2f} MB, {len(items)} formulas", flush=True)


if __name__ == "__main__":
    build()
