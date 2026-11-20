#!/usr/bin/env python3
"""Page: the state of contact sensing by the SR2 hand's magnetic fingertips, in one place (the owner, 2026-11-19 18:4x:
"all this calibration stuff can go in one document, we can discard stale information/calibration analysis that has
since been contradicted or made obsolete, so that i can get the freshest, most complete/thorough view/understanding of
the fingertip sensing without having to sift through a month of testing documentation"; "even <2mm contact sensing is
tolerable. ideally <1mm. what matters more is pose sensing smoothness, stability, throughput, and then: 1) design and
implementation of a pose filter in combination with closed loop control ... and then 2) force sensing"; "pose =
location, direction, and ideally orientation").

Reads committed data only: docs/experiments/20261117-sr2_system_identification/data/s3/20261119-*.json (made by that
study's sim/20261119-board_contact_place_rig.py, -arm_press_contact_place.py, -board_estimate_stability.py,
-board_twist_residual.py, -board_patch_size_calibration.py, -arm_press_boards.py, -hexe_dead_weight.py) and the
history digest's facts, which are written into the prose below with their sources. Writes
docs/experiments/20261119-fingertip_sensing/20261119-magnetic_fingertip_contact_sensing_on_the_sr2_hand.html. No
formulas (corvus has no TeX).

    DOCS_REFRESH=0 /usr/bin/python3 scripts/fingertip/fingertip_sensing_state_page.py     (on corvus)
"""
from __future__ import annotations

import json
import re
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

STUDY = ROOT / "docs/experiments/20261117-sr2_system_identification"
S3 = STUDY / "data/s3"
D = ROOT / "docs/experiments/20261119-fingertip_sensing"
PAGE = D / "20261119-magnetic_fingertip_contact_sensing_on_the_sr2_hand.html"
TEMPLATE = ROOT / "scripts/ur5/ur5_tip_calibration_page.template.html"
F_RIG = S3 / "20261119-board_contact_place_rig.json"
F_ARM = S3 / "20261119-arm_press_contact_place.json"
F_BOARDS = S3 / "20261119-arm_press_boards.json"
F_STAB = S3 / "20261119-board_estimate_stability.json"
F_TWIST = S3 / "20261119-board_twist_residual.json"
F_PATCH = S3 / "20261119-board_patch_size_calibration.json"
F_DW = S3 / "20261119-hexe_dead_weight.json"
F_DIRGP = S3 / "20261119-direction_gp_hand.json"
F_TRIALS = S3 / "20261119-trial_contacts.json"
F_SWEEPS = S3 / "20261119-pose_sweep_own_pose.json"
GROUPS = ["index, PIP 0", "middle, PIP 0", "thumb, PIP 0", "index, PIP 10", "index, PIP 20"]
C_GP, C_PAT, C_GEO, C_FORCE = "#1f4e9c", "#d9822b", "#111111", "#8a8a8a"
F_EDGES = [0.1, 0.3, 0.5, 1.0, 2.0, 4.0, 8.0]
L_EDGES = [0.0, 0.1, 0.2, 0.3, 0.45, 0.7, 1.2]
Z_MEAN = 10.45                      # the deployed GP's mean contact place along z (its training holds' mean)


def rel(p):
    import os
    return os.path.relpath(p, D)


def link(p, text):
    return f'<a href="{P.esc(rel(p))}">{text}</a>'


def code(s):
    return f"<code>{P.esc(s)}</code>"


def rng(v, f="{:.2f}"):
    v = [x for x in v if x is not None]
    a, b = f.format(min(v)), f.format(max(v))
    return a if a == b else f"{a}-{b}"


def held_out_rows(rig):
    """The held-out holds of the session-left-out runs, one dict per hold (the file keeps them as columns)."""
    out = []
    for run in rig["runs"]:
        H = run.get("holds")
        if run["protocol"] == "session left out" and H:
            keys = list(H)
            out += [{k: v for k, v in zip(keys, vals) if v is not None} for vals in zip(*(H[k] for k in keys))]
    return out


def fig_shear(rows, out):
    plt = P.plt_setup()
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6), dpi=110)
    edges = [0.0, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0]
    b = binned([r for r in rows if r.get("sh_ang") is not None], "ft", edges, lambda r: r["sh_ang"])
    xs = [(lo + hi) / 2 for _, lo, hi, *_ in b]
    ax[0].plot(xs, [m for *_, m, _p, _n in b], "o-", color=C_GP, lw=2, label="median")
    ax[0].plot(xs, [p for *_, _m, p, _n in b], "o--", color=C_GP, lw=1.2, mfc="white", label="90th percentile")
    ax[0].set_xlabel("shear, |f_t| (N)")
    ax[0].set_ylabel("shear direction error (deg)")
    ax[0].set_title("Direction within the surface")
    b = binned(rows, "ft", edges, lambda r: r["sh_e"])
    xs = [(lo + hi) / 2 for _, lo, hi, *_ in b]
    ax[1].plot(xs, [m for *_, m, _p, _n in b], "o-", color=C_GP, lw=2, label="median")
    ax[1].plot(xs, [p for *_, _m, p, _n in b], "o--", color=C_GP, lw=1.2, mfc="white", label="90th percentile")
    ax[1].set_xlabel("shear, |f_t| (N)")
    ax[1].set_ylabel("shear vector error (N)")
    ax[1].set_title("Error of the shear vector")
    for a in ax:
        a.set_ylim(0, None)
        a.legend(frameon=False)
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def binned(rows, key, edges, val, q=(50, 90)):
    x = np.array([r[key] for r in rows])
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        v = np.array([val(r) for r in rows if lo <= r[key] < hi and val(r) is not None], float)
        if len(v) >= 10:
            out.append(((lo + hi) / 2 if hi < 7 else lo * 1.4, lo, hi, *np.percentile(v, q), len(v)))
    del x
    return out


def fig_rig(rows, out):
    plt = P.plt_setup()
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.9), dpi=110)
    for lab, col, k in (("GP (the trials' estimator)", C_GP, "gp_e"), ("normal-only pattern match", C_PAT, "pat_e")):
        b = binned([r for r in rows if k in r], "f", F_EDGES, lambda r: r[k])
        xs = np.sqrt([lo * hi for _, lo, hi, *_ in b])
        ax[0].plot(xs, [m for *_, m, _p, _n in b], "o-", color=col, lw=2, label=f"{lab}, median")
        ax[0].plot(xs, [p for *_, _m, p, _n in b], "o--", color=col, lw=1.2, mfc="white", label="90th percentile")
        b = binned([r for r in rows if k in r], "lean", L_EDGES, lambda r: r[k])
        xs = [(lo + min(hi, 1.0)) / 2 for _, lo, hi, *_ in b]
        ax[1].plot(xs, [m for *_, m, _p, _n in b], "o-", color=col, lw=2)
        ax[1].plot(xs, [p for *_, _m, p, _n in b], "o--", color=col, lw=1.2, mfc="white")
    ax[0].set_xscale("log")
    ax[0].set_xticks([0.2, 0.5, 1, 2, 5], ["0.2", "0.5", "1", "2", "5"])
    ax[0].set_xlabel("contact force (N)")
    ax[0].set_ylabel("place error (mm)")
    ax[0].set_title("By force")
    ax[0].legend(frameon=False, loc="upper right")
    ax[1].set_xlabel("lean |f_t| / f_n (friction share)")
    ax[1].set_title("By the force's lean")
    far = [r for r in rows if abs(r["z"] - Z_MEAN) > 3]
    b = binned(far, "f", F_EDGES, lambda r: r["gp_dz"] * np.sign(Z_MEAN - r["z"]), q=(25, 50, 75))
    xs = np.sqrt([lo * hi for _, lo, hi, *_ in b])
    ax[2].axhline(0, color="0.6", lw=1)
    ax[2].fill_between(xs, [q1 for *_, q1, _m, _q3, _n in b], [q3 for *_, _q1, _m, q3, _n in b], color=C_GP, alpha=0.18,
                       lw=0, label="quartiles")
    ax[2].plot(xs, [m for *_, _q1, m, _q3, _n in b], "o-", color=C_GP, lw=2, label="median")
    ax[2].set_xscale("log")
    ax[2].set_xticks([0.2, 0.5, 1, 2, 5], ["0.2", "0.5", "1", "2", "5"])
    ax[2].set_xlabel("contact force (N)")
    ax[2].set_ylabel("GP error toward its mean place (mm)")
    ax[2].set_title("Low force: pulled toward the apex")
    ax[2].legend(frameon=False, loc="upper right")
    for a in ax[:2]:
        a.set_ylim(0, None)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_hand_places(arm, stab, out):
    plt = P.plt_setup()
    fig, ax = plt.subplots(figsize=(11.5, 5.0), dpi=110)
    for k, g in enumerate(GROUPS):
        R = [r for r in arm["rows"] if r["group"] == g and r.get("z_gp") is not None]
        H = [r for r in stab["rows"] if r["group"] == g and r["hold_force_hexe_N"] > 1]
        y = len(GROUPS) - 1 - k
        ax.plot([R[0]["z_geometry"]], [y], "o", ms=13, mfc="white", mec=C_GEO, mew=2,
                label="plate geometry at the joint readings" if k == 0 else None)
        if R[0]["z_geometry_tags"] != R[0]["z_geometry"]:
            ax.plot([R[0]["z_geometry_tags"]], [y], "o", ms=13, mfc="white", mec="#2e8b57", mew=2.5)
            ax.annotate("", xy=(R[0]["z_geometry_tags"] - 0.3, y), xytext=(R[0]["z_geometry"] + 0.3, y),
                        arrowprops=dict(arrowstyle="->", color="#2e8b57", lw=1.5))
        ax.plot([r["hold_place_z_mm"] for r in H], [y + 0.12] * len(H), "D", ms=8, color=C_GP,
                label="GP over the 2 s hold" if k == 0 else None)
        ax.plot([r["z_pattern"] for r in R], [y - 0.12] * len(R), "s", ms=8, color=C_PAT, mfc=C_PAT if
                min(r["best_cos"] for r in R) > 0.9 else "white", label="normal-only pattern" if k == 0 else None)
        ax.plot([r["z_force"] for r in R], [y - 0.26] * len(R), "^", ms=8, color=C_FORCE,
                label="frictionless reading of the HEX-E's force" if k == 0 else None)
    ax.plot([], [], "o", ms=13, mfc="white", mec="#2e8b57", mew=2.5,
            label="the same with the DIP link as flexed as the 11-18 tags saw")
    ax.axvline(10.86, color="0.7", lw=1, ls=":")
    ax.text(10.95, len(GROUPS) - 0.45, "apex", color="0.4", fontsize=12)
    ax.set_yticks(range(len(GROUPS)), GROUPS[::-1])
    ax.set_xlim(0, 21.41)
    ax.set_ylim(-1.75, len(GROUPS) - 0.3)
    ax.set_xlabel("place along the pad, z (mm): 0 proximal mount face, 21.4 distal end")
    ax.legend(frameon=False, loc="lower left", ncol=2, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_loading(stab, arm, out):
    plt = P.plt_setup()
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), dpi=110)
    geo = {r["group"]: r["z_geometry"] for r in arm["rows"]}
    cols = {"index, PIP 0": "#1f4e9c", "middle, PIP 0": "#b8481a", "thumb, PIP 0": "#2e8b57", "index, PIP 10": "#6a3d9a",
            "index, PIP 20": "#c51b7d"}
    for a, gs, title in ((ax[0], GROUPS[:3], "Straight fingers: flat proximal half"),
                         (ax[1], GROUPS[3:], "Bent PIP: distal of the apex")):
        for g in gs:
            for k, r in enumerate([r for r in stab["rows"] if r["group"] == g and r["hold_force_hexe_N"] > 1]):
                P_ = np.array(r["loading_path"])
                if not len(P_):
                    continue
                o = np.argsort(P_[:, 0])
                a.plot(P_[o, 0], P_[o, 1], "-", color=cols[g], lw=1.4, alpha=0.85, label=g if k == 0 else None)
            a.axhline(geo[g], color=cols[g], lw=1.2, ls="--")
        a.axhline(Z_MEAN, color="0.6", lw=1, ls=":")
        a.set_xlim(0.15, 2.7)
        a.set_xlabel("HEX-E force along the press (N)")
        a.set_title(title)
        a.legend(frameon=False, loc="best", fontsize=11.5)
        a.spines[["top", "right"]].set_visible(False)
    ax[0].set_ylabel("GP place along the pad, z (mm)")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_twist(tw, out):
    plt = P.plt_setup()
    fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=110)
    loads = sorted({round(r["load_N"]) for r in tw["rows"]})
    cmap = {1: "#9ecae1", 2: "#4292c6", 4: "#2171b5", 5: "#08306b", 6: "#08306b"}
    for L in loads:
        R = [r for r in tw["rows"] if round(r["load_N"]) == L]
        jit = (L - 3) * 0.25
        ax.plot([r["turn_deg"] + jit for r in R], [r["size_uT"] for r in R], "o", ms=6, color=cmap.get(L, "#08306b"),
                label=f"load about {L} N")
    t = np.linspace(0, 21, 50)
    ax.plot(t, t * 1.0, "-", color="0.4", lw=1.2, label="1 uT per degree")
    nz = np.median([r["expected_noise_uT"] for r in tw["rows"]])
    ax.axhspan(0, nz, color="0.85", lw=0, label=f"repeat noise ({nz:.1f} uT)")
    ax.set_xlabel("turn about the contact normal (deg)")
    ax.set_ylabel("field change beyond the force (uT)")
    ax.set_xlim(0, 22)
    ax.set_ylim(0, None)
    ax.legend(frameon=False, loc="upper left", fontsize=11.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def build():
    rig = json.loads(F_RIG.read_text())
    arm = json.loads(F_ARM.read_text())
    stab = json.loads(F_STAB.read_text())
    tw = json.loads(F_TWIST.read_text())
    patch = json.loads(F_PATCH.read_text())
    dw = json.loads(F_DW.read_text()) if F_DW.exists() else None
    rows = held_out_rows(rig)
    tmp = Path(tempfile.mkdtemp())
    fig_rig(rows, tmp / "f1.png")
    fig_hand_places(arm, stab, tmp / "f2.png")
    fig_loading(stab, arm, tmp / "f3.png")
    fig_twist(tw, tmp / "f4.png")
    fig_shear(rows, tmp / "f5.png")
    # numbers for the prose
    runs = {(r["protocol"], (r.get("session") or r["held_out"][0]).split("/")[-1]): r for r in rig["runs"]}
    sess = [r for r in rig["runs"] if r["protocol"] == "session left out"]
    tipo = [r for r in rig["runs"] if r["protocol"] == "tip left out"]
    dep_in = [r for r in rig["runs"] if r["protocol"] == "deployed, in sample"]
    dep_out = [r for r in rig["runs"] if r["protocol"] == "deployed, out of sample"]
    med = lambda rr, k="place_mm": [r["summary"]["all"][k]["median"] for r in rr]   # noqa: E731
    p90 = lambda rr, k="place_mm": [r["summary"]["all"][k]["p90"] for r in rr]      # noqa: E731
    pat = [r["pattern"]["all"]["place_mm"]["median"] for r in sess if r.get("pattern")]
    shear = [r["summary"]["shear"] for r in sess]
    sh_med = lambda k: [s[k]["median"] for s in shear if s.get(k)]          # noqa: E731
    sh_by = lambda band: [s["by_shear_N"][band]["direction_deg"]["median"] for s in shear   # noqa: E731
                          if s["by_shear_N"][band]["direction_deg"]]
    hold = [r for r in stab["rows"] if r["hold_force_hexe_N"] > 1]
    bent = [r for r in hold if "PIP 1" in r["group"] or "PIP 2" in r["group"]]
    straight = [r for r in hold if "PIP 0" in r["group"]]
    jz = [r["jitter_y_z_mm_50ms"][1] for r in hold]
    jz_bent = [r["jitter_y_z_mm_50ms"][1] for r in hold if "PIP 0" not in r["group"]]
    jz_straight = [r["jitter_y_z_mm_50ms"][1] for r in hold if "PIP 0" in r["group"]]
    jy = [r["jitter_y_z_mm_50ms"][0] for r in hold]
    far = [r for r in rows if abs(r["z"] - Z_MEAN) > 3]
    low = [r for r in far if 0.1 <= r["f"] < 0.5]
    tow_low = np.mean([r["gp_dz"] * np.sign(Z_MEAN - r["z"]) > 0 for r in low])
    arm_b = [r for r in arm["rows"] if r["group"] in GROUPS[3:] and r.get("z_gp") is not None]
    geo = {r["group"]: (r["z_geometry"], r["z_geometry_tags"]) for r in arm_b}
    hold_b = [r for r in stab["rows"] if r["group"] in GROUPS[3:] and r["hold_force_hexe_N"] > 1]
    gp_off = [r["hold_place_z_mm"] - geo[r["group"]][0] for r in hold_b]
    tag_z = [v[1] for v in geo.values()]
    tag_off = [r["hold_place_z_mm"] - geo[r["group"]][1] for r in hold_b]
    dirgp = json.loads(F_DIRGP.read_text())
    sweeps = json.loads(F_SWEEPS.read_text())["sweeps"]
    sw_rows, pool = [], {"deployed": {f: [] for f in ("thumb", "index", "middle")},
                         "x70y110": {f: [] for f in ("thumb", "index", "middle")}}
    for sw in sweeps:
        Fs = sw["fingers"]
        lay = sw["design"].replace("hom_film_", "").split("_twist")[0].replace("x", "(").replace("_y", ", ") + ")"
        sw_rows.append([f"{sw['sweep'][9:11]}:{sw['sweep'][11:13]}", lay, str(Fs["thumb"]["n_moved"]),
                        " / ".join(f"{Fs[f]['resid_deployed_median_uT']:.1f}" for f in ("thumb", "index", "middle")),
                        " / ".join(f"{Fs[f]['resid_x70y110_median_uT']:.1f}" for f in ("thumb", "index", "middle"))])
        for k in pool:
            for f in pool[k]:
                pool[k][f] += [r[f"resid_{k}_uT"] for r in Fs[f]["poses"] if max(abs(x) for x in r["q"]) > 3]
    sw_pool = {k: [float(np.median(v[f])) for f in ("thumb", "index", "middle")] for k, v in pool.items()}
    same = [sw for sw in sweeps if "x70_y110" in sw["design"]]
    sw_same = [[sw["fingers"][f]["resid_deployed_median_uT"] for f in ("thumb", "index", "middle")]
               for sw in (same[0], same[-1])]
    dg_low_b = [r["board_gp"]["0.1-0.5"]["place_mm"] for r in dirgp["rig"]]
    dg_low_d = [r["direction_gp"]["0.1-0.5"]["place_mm"] for r in dirgp["rig"]]
    dg_tow_b = [r["board_gp"]["0.1-0.5"]["toward_share"] for r in dirgp["rig"] if r["board_gp"]["0.1-0.5"]["toward_share"]]
    dg_tow_d = [r["direction_gp"]["0.1-0.5"]["toward_share"] for r in dirgp["rig"]
                if r["direction_gp"]["0.1-0.5"]["toward_share"]]
    dg_all_b = [r["board_gp"]["all"]["place_mm"] for r in dirgp["rig"]]
    dg_all_d = [r["direction_gp"]["all"]["place_mm"] for r in dirgp["rig"]]
    hb = [r for r in dirgp["hand"] if r["board_gp"]["force_per_hexe_N"]]
    dg_st = [r["direction_gp"]["hold_z"] for r in hb if r["group"] in ("index, PIP 0", "middle, PIP 0")]
    dg_st_b = [r["board_gp"]["hold_z"] for r in hb if r["group"] in ("index, PIP 0", "middle, PIP 0")]
    dg_low_hand = [r[k]["low_force_z"] - r[k]["hold_z"] for r in hb if r["group"] in GROUPS[3:] for k in ("board_gp", "direction_gp")]
    tilt = [r["tilt_deg"] for r in arm["rows"] if r.get("z_gp") is not None and r["group"] != "middle, PIP 0"]
    by_turn = {}
    for r in tw["rows"]:
        by_turn.setdefault(r["turn_deg"], []).append(r["size_uT"])
    tw15 = np.median(by_turn[15.0])
    rods = [v["rod R12.5"]["cos_other"][1] for v in patch.values() if "rod R12.5" in v]
    r2 = [v["sphere R2"]["cos_other"][1] for v in patch.values() if "sphere R2" in v]
    selfc = [v[k]["cos_sphere_self"][1] for v in patch.values() for k in v if k != "n_normal"]
    lede = (f"The magnetic fingertips sense where a contact is and which way it pushes. They do not sense twist, or the "
            f"size of a small contact. Held out of the fit, the trials' GP places calibration contacts within "
            f"{rng(med(sess))} mm (median), and their force's direction within {rng(med(sess, 'force_dir_deg'), '{:.1f}')} "
            f"deg, from 0.5 N up. Below 0.5 N it pulls the place toward the pad's apex. On the SR2 hand the boards run at "
            f"150 Hz, and at 2.4 N a 50 ms place jitters by {rng(jz)} mm along the pad. On bent fingers the boards and "
            f"the DIP tags agree on the contact within a millimetre.")
    L = {k: ROOT / v for k, v in {
        "cal": "docs/experiments/20261026-fingertip_calibration/20261026-fingertip_calibration_on_the_ur5.html",
        "phys": "docs/experiments/20261026-fingertip_physical_model/20261026-fingertip_physical_model.html",
        "xh": "docs/experiments/20261104-sr2_hand_crosstalk/20261105-sr2_hand_fingertip_crosstalk.html",
        "force": "docs/experiments/20261117-sr2_system_identification/20261119-fingertip_board_force_by_contact_place.html",
        "homs": "docs/experiments/20261029-hom_control_review/20261029-hom_control_and_fingertip_sensing.html",
        "v3cm": "docs/experiments/20261030-v3_tip_contact_model/20261030-v3_fingertip_contact_models.html",
        "drift": "docs/experiments/20261022-ur5_tip_calibration/20261023-ur5_tip_board_drift.html",
        "ov": "docs/overviews/printed_fingertips.html",
        "bench": "docs/experiments/20261115-hom_bench_trials/README.md",
        "queue": "docs/experiments/20261117-sr2_system_identification/QUEUE.md",
        "notes": "docs/experiments/20261117-sr2_system_identification/NOTES.md",
        "tc": "src/sr2hand/driver/reskin5x/host/tip_contacts.py",
        "rx": "src/sr2hand/driver/reskin5x/README.md"}.items()}
    S = []
    S.append(("summary", "What the fingertips can and cannot tell", f"""
<p>The owner asked on 2026-11-19 for one place with the current understanding of the fingertip sensing, stale results
dropped. The priorities are contact pose first (location, direction and, ideally, orientation), then its smoothness,
steadiness and rate, then a pose filter with closed-loop control, and force last. Table 1 is the answer in brief; each
row names the section with its evidence. Section §superseded§ lists what this page replaces.</p>
{P.table(["quantity", "how well", "where it holds", "section"], [
        ["location on the pad", f"{rng(med(sess))} mm median, {rng(p90(sess))} mm 90th percentile (session held out); "
         f"{rng(med(tipo))} mm with the whole tip held out", "0.5 N and up, where a calibrated indenter reached", "§rig§"],
        ["location, low force", f"1.1-1.7 mm median at 0.1-0.5 N; {tow_low * 100:.0f} % of the errors point toward the apex",
         "below 0.5 N", "§rig§"],
        ["location on the hand", "bent fingers: within 1 mm of the plate's geometry once the DIP tags' finger pose is used",
         "the arm presses of 11-19", "§hand§"],
        ["surface normal", f"{rng(med(sess, 'normal_deg'), '{:.1f}')} deg median", "follows from the location", "§rig§"],
        ["force direction", f"{rng(med(sess, 'force_dir_deg'), '{:.1f}')} deg median; 2-21 deg from the HEX-E on the hand",
         "contacts with shear up to slip", "§shear§, §hand§"],
        ["Shear", f"{rng(sh_med('error_N'))} N median vector error; direction within {rng(sh_by('1.0-2.0'), '{:.0f}')} deg "
         "at 1-2 N of shear", "shear over about 0.3 N", "§shear§"],
        ["twist about the normal", "not detectable", "the physics of a 7.5 mm deep sensor", "§torsion§"],
        ["contact size", "not for patches of a few mm; a line contact changes the pattern", "extended contacts only", "§area§"],
        ["rate and latency", "150 Hz per board; 50 ms means; estimate 17-19 ms; data about 41 ms old at the trial",
         "the trials' estimator since 11-17", "§estimator§, §stability§"],
        ["jitter at a steady 2.4 N", f"{rng(jz)} mm along, {rng(jy)} mm across (50 ms means)", "the arm presses' holds", "§stability§"],
        ["force size", "within 7 % on the rig; 0.26-0.57 N per HEX-E newton on the hand", "lower priority", "§force§"],
    ], caption="What the boards give, how well, and where the evidence is. Rig figures are the UR5 calibrations of the "
       "hand's three tips; hand figures are the arm presses of 2026-11-19.", n=1)}"""))
    S.append(("sensor", "The sensor and its signal", f"""
<p>Each v3 fingertip (IDs 1-6) is a half-ellipsoid pad of Siraya 85A TPU. On the hand the thumb has ID 1, the middle
ID 2 and the index ID 3. The pad is 7.4 mm across and 8.5 mm deep, and runs 22 mm proximal and 11 mm distal of its
apex. Four N52 magnets of 3 x 2 mm sit in a 5 mm square, their tops 7.5 mm under the apex. Their poles alternate,
leaving 7-16 % of one magnet's moment uncancelled. Under the pad sits a ReSkin 5X board with five MLX90393 chips in a
plus, 7 mm apart: 15 field components.</p>
<p>A contact deforms the pad and moves the magnets a few micrometres, about 6 um per newton under a 5 mm sphere in the FE
model. The chips see that motion. So the field follows the force and where it acts, summed over the patch. Seen from
7.5 mm away, a patch 1-3 mm across is a point. A twist of the patch, a torque with no net force, barely reaches the
magnets. That one geometric fact sets most of what follows.</p>
<ul>
<li>Signal: 26-32 uT per newton on the face, 12.6-14.0 at its centre, about 41 on the edges and 58-60 on the flanks.
The centre reads least.</li>
<li>Noise: 1.88 uT per component per conversion on x and y and 2.9 on z; 0.65-1.0 uT over a 50 ms mean. That is about
63 mN of force per record at 1-2 N.</li>
<li>Rate: each chip converts every 7.0-7.1 ms; the three-board mux delivers 150-163 records a second, each with all
three boards.</li>
<li>Range: the chips run a 38.4 mT range chosen for the five-magnet pad, while the four-magnet pad puts at most 5.0 mT on
a chip. A finer gain has never been tried.</li>
<li>Temperature: each chip's gain falls 0.65-0.70 % per deg C. One degree on the centre chip is 53 uT, twice a 1 N
press. The correction leaves about 9 uT over a warm-up and 4-21 uT overnight.</li>
<li>Outside fields: the Earth's field, about 44 uT, turns with the board (Section §torsion§). Steel near a pad reads as 9-17 N;
a test against a uniform field plus a gradient catches most of it but is off by default.</li>
<li>Mechanics: the pad must not move on the board; 1 um reads as 0.16 N. A pad that rocked in its seat (10-20) left
standing contacts of 11-12 N, hence the cap's pin.</li>
</ul>
<p>Sources: {link(L['rx'], 'the board README')}, {link(L['phys'], 'the physical model page')}, {link(L['ov'],
'the printed fingertips overview')}.</p>"""))
    S.append(("calibration", "Calibration and its ground truth", f"""
<p>Eleven tips were calibrated in 43,696 holds between 10-22 and 11-04 ({link(L['cal'], 'the calibration page')}). On
the UR5 the fingertip rode on the OptoForce HEX-E at the flange and met fixed indenters: spheres of R 2 and R 5 mm, a rod
of R 12.5 mm and a tall R 5 sphere. Each session pressed, slid, rolled and twisted. Touch-offs registered the contact to
0.045-0.057 mm RMS. On the inverted rig a probe on the HEX-E pressed tips held pad up (IDs 4-6, and ID 3 once), with the
load along the sensor's axis.</p>
<p>The HEX-E is the reference, and it has limits of its own. In October two gravity fits put its z gain at 0.78-0.87
of its other axes'. On 11-19 a 200 g dead weight read 0.56-0.60 of its weight, along z. The UR5 sessions and the hand's
plate presses loaded mostly its lateral axes, so that factor may not carry over to them. Its moments are too noisy to
measure twist: about 4 N mm between holds, against the 1-3 N mm a twist makes. They cannot locate a contact either: on
11-19 the line of action missed known contacts by 40-63 mm. Its z drifts about 0.4 N a minute after an arm move.</p>
<p>Coverage limits what the boards can learn. The error grows with the distance to the nearest training press: 0.32 mm
within 1 mm, 0.98 mm at 1.5-2 mm and 6.1 mm beyond 4 mm. The robot reached the face and edges; the rig added the mount
end and parts of the flanks. No press came within 1.5 mm of the tip's end. Nothing calibrated covers a broad flat
contact on the proximal half (Section §hand§).</p>"""))
    S.append(("estimator", "The estimator the trials run", f"""
<p>{link(L['tc'], 'tip_contacts.py')} has run the boards in the trials since 11-17, in a process of its own. Per
board:</p>
<ol>
<li>Each chip's reading is divided by its gain at its temperature.</li>
<li>The change is the 50 ms mean less the rest: the mean of 1 s at the open pose before the close. There is no re-zero
during a trial, since it would absorb a held press.</li>
<li>The own-pose term: a GP on the finger's three joint angles predicts the board's change with no contact, and the
difference between the rest's pose and the present one is taken off.</li>
<li>The physical model decides whether a contact exists. It uses an FE basis of pad 05R4 with the magnets as fitted
at rest. A contact needs an explained chi-square over 25 and 0.25 N to start, and 0.15 N to stay.</li>
<li>The pooled board GP gives the point and the 3D force, blended with the physical model by w, the GP's confidence.</li>
<li>The point goes into the palm frame through the hand model at the measured joints.</li>
</ol>
<p>An estimate for three boards takes 17-19 ms, and the data reach the trial about 41 ms old. The board GP is
{code('20261105-board_gp_v3_id1_id3_rig_id3_id6.npz')}: 14,185 holds of six tips, presses and slides, length 1.94 and
noise 0.3. The own-pose GP the trials load is the 11-04/05 fit over many layouts. In free-air sweeps at 11 layouts on
the night of 11-19 it left a median of {rng(sw_pool['deployed'], '{:.1f}')} uT, about 0.2-0.3 N (Section §hand2§,
Table 4). A refit at (70, 110) ({code('20261119-own_pose_gp_sweep_x70y110_7v5.npz')}) left 4.4-5.7 uT in its own
session but no less than the deployed GP in later sweeps there, and more at the wide layouts. So the deployed GP
stays.</p>"""))
    rig_body = (f"""
<p>The UR5 pressed, slid, rolled and twisted spheres of R 2 and R 5 mm and a rod of R 12.5 mm on every tip. The HEX-E
read the force, and the arm's registration gave the contact place and normal. To measure the trials' GP
honestly, its recipe was fitted again with one session left out, and that session's presses, slides, rolls and twists
were predicted. The deployed GP's length and noise were kept.</p>
<p>The result is in Table 2. With a session left out, the median place error is {rng(med(sess))} mm. With the whole
tip left out, it is {rng(med(tipo))} mm: a tip's own calibration helps, but another tip's nearly does. In sample, the
deployed GP gives {rng(med(dep_in))} mm; that figure flatters it. The deployed GP never saw two tall-sphere sessions.
It places their contacts within {rng(med(dep_out))} mm, but both sessions carry a constant 0.9-1.1 mm offset across the
pad. That points at the tall tool's
registration, not the boards.</p>
{P.table(["held out", "tip", "holds", "GP place median / 90th pct (mm)", "normal (deg)", "force direction (deg)",
          "normal-only pattern (mm)"],
         [[{"session left out": "session", "tip left out": "whole tip"}.get(run["protocol"], "deployed, unseen"),
           str(run["tip"]),
           str(run["n"]), f"{run['summary']['all']['place_mm']['median']:.2f} / {run['summary']['all']['place_mm']['p90']:.2f}",
           f"{run['summary']['all']['normal_deg']['median']:.1f}", f"{run['summary']['all']['force_dir_deg']['median']:.1f}",
           f"{run['pattern']['all']['place_mm']['median']:.2f}" if run.get("pattern") else "&#8211;"]
          for run in sess + tipo + dep_out],
         num_cols=(1, 2, 3, 4, 5, 6), caption="The trials' GP on held-out calibration sessions (session: refitted without "
         "that session; tip: without all of that tip's sessions; deployed: the GP as used, on the tall-sphere sessions it "
         "never saw). Pattern: the best cosine with the same tip's near-normal presses in the training sessions.", n=2)}
<p>Figure 1 splits the held-out errors. The GP's error falls from about 1.5 mm at 0.1-0.5 N to about 0.6 mm above
1 N. It barely grows with the force's lean. A pattern match against normal presses only does worse as the lean grows,
because a leaning force looks like a normal press elsewhere on the pad. The low-force error is not noise. Below
0.5 N, {tow_low * 100:.0f} % of the GP's place errors point toward its mean place, z {Z_MEAN} mm, near the apex. A GP
falls back on its prior when the signal is small, and the field of a 0.3 N press is only 3-5 uT against 1.7 uT of
noise per component.</p>
{P.fig(1, tmp / 'f1.png', 'Held-out place errors of the trials&#8217; GP (blue) and of a normal-only pattern match (orange), pooled over the seven sessions of the hand&#8217;s tips: by contact force (left) and by the force&#8217;s lean (middle). Right: the GP&#8217;s error along the pad toward its mean place, for contacts at least 3 mm from it.')}""")
    S.append(("rig", "Location and direction where the truth is known", rig_body + f"""
<p>Two older results complete the picture. A tip's own presses matter: another tip's GP misses by 1.0-1.4 mm, and
pooling the other five tips brings that to {rng(med(tipo))} mm (Table 2). And a GP whose input is only the field's
direction, the change divided by its size, was tested in October on v2 ID 1 and v3 IDs 1-2. It placed contacts within
0.26-0.28 mm overall and 0.59-0.75 mm under 1 N. Trained on presses up to 2 N, it placed 2-8 N presses within
0.25-0.31 mm and read their force within 5-6 %. There the board GP missed by 1-4 mm and read 45-87 % too little, falling
back on its training mean. The direction GP was never deployed. Its input does not shrink with the force, which is what
pulls the board GP toward the apex below 0.5 N.</p>
<p>Tried tonight on the same held-out sessions, it helps but does not cure
({link(STUDY / 'sim/20261119-direction_gp_hand.py', 'sim/20261119-direction_gp_hand.py')}). Overall it places contacts within {rng(dg_all_d)} mm against
{rng(dg_all_b)} for the board GP. At 0.1-0.5 N it gives {rng(dg_low_d)} mm against {rng(dg_low_b)}. There
{rng([x * 100 for x in dg_tow_d], '{:.0f}')} % of its errors point toward the apex, against
{rng([x * 100 for x in dg_tow_b], '{:.0f}')} %. At such forces the field is 3-15 uT against 0.7 uT of noise per 50 ms,
and no input transform can restore what the noise hides.</p>"""))
    S.append(("shear", "Shear: the force along the surface", (f"""
<p>Shear is the part of the force along the pad's surface: friction, and what a twisting or sliding object exerts. With
a session held out, the GP's shear vector is off by {rng(sh_med('error_N'))} N (median). Its size reads
{rng(sh_med('size_ratio_over_0.2N'))} of the HEX-E's, and the normal force {rng(sh_med('normal_force_ratio'))}. Its
direction within the surface depends on how much shear there is (Figure 5). From 1 to 2 N of shear it is within
{rng(sh_by('1.0-2.0'), '{:.0f}')} deg (median); from 0.2 to 0.5 N only within {rng(sh_by('0.2-0.5'), '{:.0f}')} deg.
The error has a floor of about 0.15-0.2 N, so a shear under about 0.3 N gives little more than its presence.</p>
{P.fig(5, tmp / 'f5.png', 'Held-out shear of the trials&#8217; GP against the HEX-E, by the true shear: the angle between estimated and true shear within the surface (left, shear over 0.2 N) and the error of the shear vector (right).')}
<p>Can twist come from a slower filter over shear? At one contact, no: a pure twist of the patch puts no net shear on
that contact, so its shear is blind to it. At the grasp, yes, in two ways. The moment the fingers apply to the object
is the sum of each contact's lever arm crossed with its shear. With 0.15-0.2 N of error per finger at lever arms of
about 20 mm, that moment is known to about 5-10 N mm. And when the object turns in the grasp, the three contacts move
on their pads in one rotational pattern. A slow filter over the places (0.3-0.5 mm of jitter per 50 ms at 2.4 N) can
follow that turn to a few degrees. Averaging over time removes the jitter but not the GP's fixed errors, which set
those limits.</p>""") + """
<p>Slides must be in the training. In October the low sphere's shear erred 0.33 N from presses alone, and 0.13 N with
its own slides. Sliding contacts were placed within 0.47-0.52 mm with slides in training, and 0.98-1.07 mm without.</p>"""))
    S.append(("torsion", "Twist about the normal is not sensed", (f"""
<p>Each tip's sphere session ended with twist groups. At one place and load, the arm turned the tip about the
contact normal by 5-20 deg and back. The HEX-E's moment about the normal wanders by about 4 N mm from hold to hold,
so the turn angle is the truth here, not the measured moment.</p>
<p>A local linear model of the field from the 3D force was fitted to each session's presses and slides near the group's
place. Beyond that model, the twist holds' fields grow linearly with the turn: about 1 uT per degree, {tw15:.1f} uT at 15
deg. They are the same at 1 N and at 5 N (Figure 4). Torsion of the pad cannot behave like that. A stuck patch's torque
grows with the cube of its radius, about 5 times from 1 to 5 N. A slipping patch's grows with the load to the 4/3, about
8 times. A field fixed in the room does grow with the turn alone, as a turning board sees it. Here it is about 25 uT
across the axis at each chip, the size of the Earth's field.</p>
<p>The physics agrees. The magnets sit 7.5 mm under a patch 1-3 mm across. A twist is a torque with no net force, and
its effect falls off with depth faster than a force's: 7.5 mm down it barely moves the magnets. So orientation is not a quantity
these boards can give, and no amount of calibration will change that.</p>
{P.fig(4, tmp / 'f4.png', 'Field change beyond the force model when the tip turns about the contact normal, by turn and load: every twist group of the hand&#8217;s three tips. The grey band is the repeat noise of the comparison.')}""") + """
<p>October's GPs agree. Their torsion correlated with the HEX-E's at r = 0.21-0.42, and on two of three tips they did no
better than predicting no torsion. The fixture planned to settle it, a twist with the face normal along the HEX-E's
axis, never ran; this analysis makes it unnecessary for the boards.</p>"""))
    S.append(("area", "Contact size", (f"""
<p>The boards measure force, not contact area (the page of 11-19 on force by contact place). An R 2 and an R 5 sphere
at the same place give the same field per newton within 10 %. Their patterns match at cosine {rng(r2, '{:.3f}')},
against {rng(selfc, '{:.3f}')} for repeats of one sphere. The R 12.5 rod's line contact does change the pattern
(cosine {rng(rods, '{:.3f}')}). So the boards can tell a long or flat contact from a small one, but not the size of a
small patch. With a known object, the patch follows from Hertz with the measured force and place.</p>""") + """
<p>Shape still matters for what the GP learns. Left out of training, an indenter costs 0.57-1.80 mm of location error,
and the rod's forces read 14 % low without its own presses. So every shape the hand will meet should be in the
training. The trials' sphere and cylinder are flatter than the R 5 sphere; the rod is the nearest calibrated shape.</p>"""))
    S.append(("hand", "On the hand: the place, the lean and the finger's pose", (f"""
<p>On 2026-11-19 the arm pushed each pad into a flat plate at 0.2 mm/s while the servos held, to about 2.5 N
(lab/arm_press.py). A flat plate touches a convex pad where the pad's normal points along the press. So the hand model at
the measured joints gives the contact place, with no contact model needed. Figure 2 compares four places per press.</p>
<p>The HEX-E's force leaned {rng(tilt, '{:.0f}')} deg from the press direction in every press but the middle's: the
plate's friction as the finger yielded. The boards agree that it leaned. Fed each press's field, the GP's force
direction lies 2-21 deg from the HEX-E's in the pad frame.</p>
<p>Over each 2 s hold, the GP places the bent index's contact {rng(gp_off, '{:.1f}')} mm distal of the geometry at the
joint readings. The DIP tags measured the reason on 2026-11-18: the index's DIP link sits 9.2 deg more flexed than its
joint readings say, the middle's 12.2 deg. With that flexion the geometry moves to z {rng(tag_z, '{:.1f}')} mm, and the
GP lands between {abs(min(tag_off)):.1f} mm short of it and {max(tag_off):.1f} mm beyond. The tags were seen at MCP
47-58 and PIP 65-80. An offset that also fits at PIP 10-20 points to a zero offset of the index rather than a pot reading
low near the top of its range. Two independent instruments, the cameras and the boards, thus agree on
the finger's pose, and the boards place the contact within a millimetre of where it must be. The normal-only pattern
match puts it 2-4 mm further out still, near where a frictionless contact would have to sit to push along the leaning
force. The earlier page's &#8220;the boards and the hand model disagree by 4-6 mm&#8221; was that pattern match against
the uncorrected joints, not the boards.</p>
{P.fig(2, tmp / 'f2.png', 'Contact place along the pad in the arm presses of 2026-11-19, three per group. Circles: the plate&#8217;s geometry at the joint readings (black) and with the DIP link as flexed as the 11-18 tags measured (green). Diamonds: the trials&#8217; GP over the 2 s hold. Squares: the normal-only pattern match, open where its best cosine is under 0.9. Triangles: where a frictionless contact would sit for the HEX-E&#8217;s leaning force.')}
<p>The straight fingers press a flat face onto the pad's flat proximal half, where the 22 mm semi-axis keeps the
surface within 12 deg of the apex normal. There the contact is a broad patch, no calibrated press matched its field
above cosine 0.45, and the GP's place drifts with the force (Figure 3, left). On the flat half the place is also the
least defined physically: a 1 deg change of the finger's lean moves a flat contact about 1 mm.
The direction GP puts the straight index's and middle's contacts at z {rng(dg_st, '{:.1f}')} mm, near the geometry at
the joint readings (7.5-7.8), where the board GP gives {rng(dg_st_b, '{:.1f}')}. If that is right, the extra flexion the
tags saw is there with the PIP bent but not at PIP 0, as backlash off a stop would be. Both GPs extrapolate there, so
this is a lead to test, not a result.</p>""")))
    trials = [t for t in json.loads(F_TRIALS.read_text())["trials"] if not t.get("stop")
              and min((t["fingers"].get("thumb") or {}).get("surface_mm", [-99])[1], 0) > -6]
    trow = []
    for t in trials:
        for f in ("thumb", "index", "middle"):
            g = t["fingers"].get(f) or {}
            if not g.get("estimates"):
                continue
            trow.append([t["record"][:15] + ", " + f, f"{g['contact_share'] * 100:.0f} %", f"{g['place_z_mm'][1]:.1f}",
                         f"{g['jitter_z_mm'][0]:.2f} / {g['jitter_z_mm'][1]:.2f}", f"{g['force_N'][0]:.2f}",
                         f"{g['surface_mm'][1]:+.1f} ({g['surface_mm'][0]:+.1f} to {g['surface_mm'][2]:+.1f})"])
    S.append(("twists", "In HOM twists on the hand", f"""
<p>The trials' estimator records each finger's contact every 20 ms. In the anchor grasp's full twists of 19:14 and 19:17
(g0 at (70, 110), torque budget 0.56, 50 deg of twist each), Table 3 gives what it said. The thumb pressed 2.5-2.7 N
and was in contact almost throughout. The index and middle pressed 0.7 and 1.2 N, in contact a fifth to a half of the
time. The place jitters 0.24-0.25 mm from one estimate to the next at the thumb's and middle's forces, and 0.7 mm at
the index's 0.7 N, as the jitter's growth at low force predicts (Section §stability§).</p>
<p>The tracked sphere gives the truth for one coordinate: a contact must lie on its surface. The estimated points lie
1.6-3.6 mm inside it (medians). The pad's own indentation, 0.5-1.5 mm at these forces, accounts for part, since the
estimate is a point of the undeformed pad. The rest fits fingers more flexed than their readings (Section §hand§). The night's
series will add every ranked layout to this table
({link(STUDY / 'sim/20261119-trial_contacts.py', 'sim/20261119-trial_contacts.py')}).</p>
{P.table(["trial, finger", "contact", "place z (mm)", "jitter median / p90 (mm)", "force (N)", "to the surface (mm)"],
         trow, num_cols=(1, 2, 3, 4, 5), caption="The trials' contacts in the two full twists: share of periods with a "
         "contact, median place along the pad, change between estimates, median normal force, and the point's distance "
         "from the tracked sphere's surface (negative inside; median and quartiles).", n=3)}"""))
    S.append(("hand2", "On the hand: own pose, drift, neighbours and blind spots", f"""
<ul>
<li>Own pose. With nothing touching, a finger's own motion changes its board by up to 166 uT, 5.8 N to the
estimator; the median is 54-71 uT. Most of it is the board turning in fields fixed to the hand and the room. A GP on
the finger's joints removes it to 4.4-7.3 uT when fitted at that layout or over many layouts. Under load it also
removes real signal when the joints give: on 11-18 the GP's slopes fell from 0.64-0.86 to 0.44-0.56 with the
term.</li>
<li>Layouts. On the night of 11-19 the hand ran the same free-air poses at 11 layouts, (60, 135) to (135, 65)
(Table 4). The many-layout GP left a median of {sw_pool['deployed'][0]:.1f}, {sw_pool['deployed'][1]:.1f} and
{sw_pool['deployed'][2]:.1f} uT on the thumb, index and middle, with no trend across layouts. The same layout,
(70, 110), left {rng(sw_same[0], '{:.1f}')} uT at 19:05 and {rng(sw_same[1], '{:.1f}')} uT at 21:22: the arm's pose
and the hour move the term as much as the layout does. A refit at one layout is no better at the others, so a sweep
at each new layout is not needed.</li>
<li>Fingers touching each other. At (70, 90) the pose with every MCP and PIP at 40 deg stopped the thumb's and
index's MCPs 6-7.5 deg short. On the CAD hand's collision spheres their DIP links meet in that pose (0.6 mm overlap);
the bench's check, three balls per finger, had them 27 mm apart. Both boards read 12-13 uT above their own-pose
prediction there, more than at any other layout in that pose; the middle's read nothing extra. Whether through the
pads or through the joints giving under the load, the boards saw the collision.</li>
<li>Turning the hand. The own-pose GP knows the palm only at the attitude of its sweep. Turned another way, each board
sees the Earth's field from another angle: about 1 uT per degree (Section §torsion§), 0.07 N per degree. Re-zero after a wrist
turn.</li>
<li>Drift. At rest the boards drift 18-22 uT over one to two hours, and each press leaves a 7 uT step, 0.29 of it
lasting. Re-zero at the open pose before each grasp, as the trials do.</li>
<li>Neighbours. The other fingers change a board by at most 5.4 uT (0.19 N) at the home layout, and 6-15 uT with pads
closer than 20 mm.</li>
<li>Blind spots. Contacts on the DIP link beside the pad read nothing: 14 side presses on 11-18. The middle board read
nothing in 18 of 20 twists on 11-17. So &#8220;no contact&#8221; means no pad contact, not none at all.</li>
</ul>
{P.table(["sweep", "layout (mm)", "poses", "deployed GP: thumb / index / middle (uT)", "refit at (70, 110) (uT)"],
          sw_rows, caption="The own-pose term at each layout, 2026-11-19 night: the median residual over the sweep's "
          "poses with a joint moved more than 3 deg, from the hold's mean field against the GP's predicted change from the "
          "zero pose. The sweeps after 21:10 ran in the dark; (70, 90) stopped at its fifth pose; (85, 60) kept two poses "
          "at the bench's 30 mm margin.", n=4)}
<p>Sources: {link(L['xh'], 'the crosstalk page')}, {link(L['drift'], 'the drift page')}, {link(L['bench'], 'the bench README')},
{link(STUDY / 'sim/20261119-pose_sweep_own_pose.py', 'sim/20261119-pose_sweep_own_pose.py')}.</p>"""))
    S.append(("stability", "Smoothness, steadiness and rate on the hand", (f"""
<p>The splitter logs 150 records a second, each with all three boards; the longest gaps in the presses were 10-24 ms.
In free air the field's noise is {rng([r['noise_free_uT_record'] for r in hold])} uT per component per record and
{rng([r['noise_free_uT_50ms'] for r in hold])} uT over the estimator's 50 ms window. Table 5 gives the place's jitter
while the pad was held at a steady force for 2 s.</p>
{P.table(["press", "force (N)", "place z (mm)", "jitter along / across, 50 ms (mm)", "per record (mm)", "drift (mm/s)",
          "force jitter (N)"],
         [[r["group"] + f", rep {r['rep']}", f"{r['hold_force_hexe_N']:.2f}", f"{r['hold_place_z_mm']:.1f}",
           f"{r['jitter_y_z_mm_50ms'][1]:.2f} / {r['jitter_y_z_mm_50ms'][0]:.2f}",
           f"{r['jitter_y_z_mm_record'][1]:.2f} / {r['jitter_y_z_mm_record'][0]:.2f}",
           f"{r['drift_y_z_mm_per_s'][1]:+.2f}", f"{r['force_sd_N_50ms']:.3f}"] for r in hold],
         num_cols=(1, 2, 3, 4, 5, 6), caption="The GP's place and force in the arm presses' 2 s holds: standard deviations "
         "about a straight line in time, and that line's slope along the pad.", n=5)}
<p>So at 2.4 N a 50 ms place is steady to {rng(jz_bent)} mm (sd) on the bent index, where the pad curves, and to
{rng(jz_straight)} mm on the straight fingers' flat half. Jitter scales roughly with the noise over the field, so at 0.5 N expect
about five times more. Figure 3 shows the larger effect: while the force rises, the bent fingers' place starts at the
apex and reaches its held value only from about 1 N.</p>
{P.fig(3, tmp / 'f3.png', 'The GP&#8217;s place along the pad while the HEX-E&#8217;s force rose, every press; dashed: the plate&#8217;s geometric place; dotted: the GP&#8217;s mean place.')}""")))
    S.append(("force", "Force, briefly", f"""
<p>On the rig the force's size is within about 7 % (median) on held-out holds. On the hand it is not: in the plate
presses of 11-18 and 11-19 the trials' estimate rose 0.26-0.57 N per HEX-E newton. Three causes are known
({link(L['force'], 'the page on force by contact place')}):</p>
<ul>
<li>Where the contact lands. A straight finger's flat contact gives 6-7 uT/N in a pattern no calibration contains; a
bent finger's gives 0.88-0.92 of its calibration.</li>
<li>The own-pose term removes signal when the joints give under load (Section §hand2§).</li>
<li>The board GP loses about a quarter of a well-matched field: a linear estimate at the matched calibration place
reads 0.83-0.87 where the GP reads 0.60-0.66.</li>
</ul>
<p>The HEX-E's own scale is open (Section §calibration§). The direction GP, which read force within 5-6 % beyond its training range,
is the first thing to try.</p>"""))
    S.append(("loop", "Closed loop so far", f"""
<ul>
<li>Bench. The boards have only supplied HOM's contact frames (the server's --tips on). In the A/B of 11-17, with 2-6
trials an arm, boards off scored 0.426-0.445. Boards on scored 0.463 with sub-steps and hysteresis, and 0.310 with exact
integration. In one step they scored 0.405, and 2 of 5 trials drifted out. Not conclusive.</li>
<li>Simulation. HOM tracked as well without the fingertips' contact points as with them. A force loop cut the
pin-push's force error from 1.05 to 0.31 N, but held the sphere twist in 72 of 120 runs against 119 without it.
A contact filter on UR5 rolls cut the record-to-record scatter from 0.76 to 0.02 mm, the error staying at the
estimate's own 0.46 mm ({link(L['homs'], 'the HOM sensing page')}).</li>
</ul>"""))
    S.append(("filter", "What a pose filter should do", f"""
<p>The measurements above set its design:</p>
<ol>
<li>State per finger: the contact's place on the pad as two surface coordinates, plus the 3D force. The normal follows
from the place, so it needs no state of its own.</li>
<li>Measurement: the GP at 50 ms means (20 Hz), or single records at 150 Hz with about 2.5 times the noise. Measurement
noise: {rng(jz_bent)} mm along the pad at 2.4 N where it curves, {rng(jz_straight)} mm on its flat half, growing
roughly as one over the force. Inflate it below
0.5 N, where the place is biased toward the apex, and drop contacts under 0.25 N.</li>
<li>Trust flags: the GP's w; the best cosine of the field with the calibration, where under about 0.7 means outside
it; and the physical model's residual.</li>
<li>Process model: between measurements, predict the contact from the finger's joints and the object's tracked pose
through the hand model. That needs the joint offsets fixed first (the index's DIP link sits 9 deg more flexed than read).</li>
<li>Time constant: at 2.4 N, five 50 ms means bring the jitter on the curved half to about 0.15-0.2 mm. In a press the place settles only
from about 1 N (Figure 3), so a filter should not hold on to a low-force place.</li>
<li>Grasp-level twist: the object's turn from the three contacts' places over time, and the twisting moment from each
contact's lever arm and shear (Section §shear§), at 2-5 Hz.</li>
<li>Rest: re-zero at the open pose before each grasp, with the many-layout own-pose GP (Section §hand2§).</li>
</ol>"""))
    S.append(("superseded", "What this page replaces", P.table(["earlier statement", "where", "now"], [
        ["the boards and the hand model disagree by 4-6 mm on the contact place", "force by place page, 11-19",
         "the normal-only pattern match read the force's lean; GP and tag-corrected geometry agree within 1 mm (Section §hand§)"],
        ["the boards read 0.26-0.38 of the HEX-E", "S3 presses, 11-18", "depends on where the contact lands (Section §force§)"],
        ["the fingertip model takes force within about 5 %", "HOM sensing page, 10-29", "true on the rig, not on the hand"],
        ["the HEX-E is the force reference", "calibration protocol, 10-22",
         "its z read a dead weight at 0.56-0.60 (11-19); its lateral axes are untested"],
        ["the own-pose GP leaves 5.9-7.3 uT with layouts left out", "crosstalk page, 11-05",
         "5.9-8.0 uT over 11 layouts on 11-19, no trend with layout (Table 4)"],
        ["a sweep and a GP refit at each new layout", "this page, 11-19 evening",
         "not needed: a one-layout refit is no better elsewhere (Section §hand2§)"],
        ["the estimator inside the trial's process, 57-73 ms", "bench README, 11-17 morning", "its own process, 17-19 ms"],
        ["torsion may become a calibrated output on a better fixture", "calibration page, 11-04",
         "not detectable: the twist response is the board turning in the room's field (Section §torsion§)"],
        ["the dead weight read 0.56-0.58 or 0.58-0.60", "notes and index, 11-19", "0.56-0.60, from both steps"],
        ["board GP versions of 10-26 and 11-02", "calibration page", "the 11-05 pooled GP is the one in use"],
    ], caption="Earlier results this page corrects or replaces.", n=6)))
    S.append(("open", "Open questions, in order", f"""
<ol>
<li>Low force is the weak point, and the direction GP only eases it (Section §rig§): on the hand both GPs still sit
{rng([-x for x in dg_low_hand], '{:.1f}')} mm toward the apex at 0.3-0.6 N. Treat places below about 1 N as weak in the filter, and
check whether a finer chip gain (item 5) lowers the noise floor that causes it.</li>
<li>Fix the index's and middle's joint offsets: 9 and 12 deg at the DIP link with the PIP bent, perhaps none
straight (Section §hand§). Plate presses at several PIP angles, and DIP tags seen at less curl, can tell a zero offset from
backlash off the PIP's stop and from pots that read low near the top of their range.</li>
<li>The own-pose term's last 6-8 uT moves with the arm's pose and the hour as much as with the layout (Table 4):
try the palm's attitude as an input, or re-zero after any wrist turn. The thumb's DIP tag, now tracked in the sweep,
sat 2.6 mm and 2.1 deg from the model at zero; its other poses need the lights on.</li>
<li>The HEX-E's scale along its lateral axes: a dead weight whose line loads x or y.</li>
<li>A finer chip gain: the four-magnet pads use under 5 mT of a 38.4 mT range. Does it lower the noise?</li>
<li>Broad flat contacts on the straight fingers: calibrate a flat indenter on the proximal half, or keep the fingers
bent so their contacts land where the calibration is.</li>
<li>The pose filter of Section §filter§ with the HOM loop, tested on recorded trials before the bench.</li>
</ol>"""))
    files = [STUDY / f"sim/20261119-{n}.py" for n in ("board_contact_place_rig", "arm_press_contact_place", "direction_gp_hand",
                                                     "board_estimate_stability", "board_twist_residual",
                                                     "board_patch_size_calibration", "arm_press_boards")]
    S.append(("data", "Data and scripts", "<ul>" + "".join(f"<li>{link(f, P.esc(str(f.relative_to(ROOT))))}</li>"
                                                          for f in files) +
              "<li>Their outputs: " + code("docs/experiments/20261117-sr2_system_identification/data/s3/20261119-*.json")
              + "</li><li>Calibration holds: the UR5 sessions under corvus's and the workstation's " + code("logs/")
              + " and " + code("docs/experiments/20261030-inverted_rig_seat_block/data/holds/") + "</li>"
              "<li>Lab notes: " + link(L["notes"], "NOTES.md") + ", 2026-11-19 18:10-18:35; the history behind this page "
              "is in " + link(L["ov"], "the printed fingertips overview") + " and the pages it lists.</li></ul>"))
    secno = {sid: str(k) for k, (sid, _, _) in enumerate(S, 1)}
    S = [(sid, hd, re.sub(r"§(\w+)§", lambda m: secno[m.group(1)], body)) for sid, hd, body in S]
    toc = "\n".join(f'<li><a href="#{sid}">{k}. {P.esc(hd)}</a></li>' for k, (sid, hd, _) in enumerate(S, 1))
    secs = "\n".join(f'<section id="{sid}"><h2>{k}. {P.esc(hd)}</h2>{body}</section>' for k, (sid, hd, body)
                     in enumerate(S, 1))
    byline = ("UR5 calibrations of the hand's tips (10-22 to 11-04), the inverted rig (11-02 to 11-04) and arm presses on "
              "the SR2 hand at corvus (2026-11-19), gathered for the owner's request of 18:4x. Page by "
              + code("scripts/fingertip/fingertip_sensing_state_page.py") + ".")
    page = TEMPLATE.read_text()
    for k, v in {"TITLE": "Fingertip contact sensing",
                 "H1": "Contact location, direction, twist and force from the magnetic fingertips of the SR2 hand",
                 "EYEBROW": "Fingertip sensing &#183; 2026-11-19", "LEDE": lede, "BYLINE": byline, "FILM": "",
                 "TOC": toc, "SECTIONS": secs, "FOOTER": str(PAGE.relative_to(ROOT)), "TEXCSS": ""}.items():
        page = page.replace("{{" + k + "}}", v)
    page = retro_style.apply(page)
    page = docs_index.tag(page, topic="fingertip", summary=lede, date="2026-11-19")
    D.mkdir(parents=True, exist_ok=True)
    PAGE.write_text(page)
    print(f"{PAGE.relative_to(ROOT)}: {PAGE.stat().st_size / 1e6:.2f} MB", flush=True)
    del dw, runs, bent, straight, pat, dep_in, dep_out


if __name__ == "__main__":
    build()
