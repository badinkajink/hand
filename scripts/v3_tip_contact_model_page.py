#!/usr/bin/env python3
"""Page on the contact models of the four-magnet v3 fingertips fitted to the UR5 presses (owner, 2026-10-30: "check if our
calibration results on the v3 four n52 magnet fingertips enable a more consistent FE / sphere-packed / hydroelastic
contact model in simulation"; one modulus, errors accepted at the extrema; forces before indentation; the area of
contact and the contact pose; "seems like we should change the modulus to match the real setting"). Reads
data/20261030-contact_model_fit.json (scripts/v3_tip_contact_model.py), data/20261030-numbers.json with the media of
scripts/v3_tip_contact_model_figures.py, and the replays and scripted grasps of scripts/ur5_shear_impratio.py at 10 MPa and
at the scenes' 2.5 MPa (docs/experiments/20261026-shear_friction_impratio/data/20261026-impratio_{replays,mjw}.jsonl); writes
docs/experiments/20261030-v3_tip_contact_model/20261030-v3_fingertip_contact_models.html.

    .venv/bin/python scripts/v3_tip_contact_model_page.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import docs_index  # noqa: E402
import retro_style  # noqa: E402
import texsvg  # noqa: E402
import ur5_tip_calibration_page as P  # noqa: E402   (esc, code, fig, film, table, TEX_RE)

D = ROOT / "docs/experiments/20261030-v3_tip_contact_model"
MEDIA, DATA = D / "media", D / "data"
PAGE = D / "20261030-v3_fingertip_contact_models.html"
TEMPLATE = ROOT / "scripts/ur5_tip_calibration_page.template.html"
TEX_CACHE = D / "texsvg_cache.json"
EXP = ROOT / "docs/experiments"
L = dict(
    contact=ROOT / "docs/overviews/contact_model.html",
    overview=ROOT / "docs/overviews/printed_fingertips.html",
    pm=EXP / "20261026-fingertip_physical_model/20261026-fingertip_physical_model.html",
    cal=EXP / "20261026-fingertip_calibration/20261026-fingertip_calibration_on_the_ur5.html",
    shear=EXP / "20261026-shear_friction_impratio/20261026-shear_friction_and_impratio_of_the_printed_fingertip.html",
    gt=EXP / "20261022-ur5_tip_calibration/20261022-ur5_tip_ground_truth_and_predictors.html",
    fec=EXP / "20261018-fe_contact_calibration/20261018-fe_contact_calibration.html",
    sens=EXP / "20261013-contact_feedback/20261013-contact_parameter_sensitivity.html",
    rig=EXP / "20261030-inverted_rig_seat_block",
)
SERVO_N_PER_M = 250.0          # each fingertip through the servos at 0.72 N m/rad (contact sensitivity page, 2026-10-13)
JOINT = ("v3id1_low", "v3id2_low", "v3id3_low", "v3id1_rod", "v3id2_rod", "v3id3_rod")
LOW_J, ROD_J = JOINT[:3], JOINT[3:]
TIPS = ("v3id1", "v3id2", "v3id3")


def link(p, text):
    return f'<a href="{P.esc(os.path.relpath(p, D))}">{text}</a>'


def f2(x):
    return f"{x:.2f}"


def rng(v, nd=2):
    a, b = f"{min(v):.{nd}f}", f"{max(v):.{nd}f}"
    return a if a == b else f"{a}&#8211;{b}"


def pooled(d, keys, model, E=None):
    """Measured over model force of every hold of `keys` at the joint modulus, with its level and region."""
    r, lv, rg = [], [], []
    J = d["fits"]["joint"]
    np.seterr(divide="ignore", invalid="ignore")
    for k in keys:
        h = d["sessions_detail"][k]["holds"]
        f = np.array(h["f_n"])
        if model == "pads":
            m = f / ((E or J["E_pads_MPa"]) * np.array(h["Gp"]))
        elif model == "field":
            m = f / ((E or J["E_field_MPa"]) * np.array(h["Gf"]))
        else:
            fe = np.array(h["fe_fit"])
            m = np.where(fe > 0, f / np.where(fe > 0, fe, 1.0), np.nan)
        r.append(m)
        lv.append(np.array(h["level"]))
        rg.append(np.array(h["region"]))
    r = np.concatenate(r)
    r[~np.isfinite(r) | (r <= 0)] = np.nan
    return r, np.concatenate(lv), np.concatenate(rg)


def med(v):
    v = v[np.isfinite(v)]
    return float(np.median(v))


def within(v, q=1.25):
    v = v[np.isfinite(v)]
    return float(np.mean(np.abs(np.log(v)) < np.log(q)))


SHEAR_DATA = EXP / "20261026-shear_friction_impratio/data"
E_SCENE = 2.5                  # MPa: reorient_backends.E_TPU since 2026-10-30, the joint modulus rounded
TR_SCENE = 0.04                # s: reorient_backends.pad_tr(E_TPU)


def _rows(name):
    return [json.loads(x) for x in open(SHEAR_DATA / name) if x.strip()]


def replays(E, noslip=0, ir=10.0):
    """The D1-D8 replays under the trainer's numerics at modulus E (MPa): runs, completed, held to the end, the held tool's
    creep (median, quartiles, um/s) and the pads' d0 range (scripts/ur5_shear_impratio.py replays)."""
    v = [r for r in _rows("20261026-impratio_replays.jsonl") if r["set"] == "d1d8" and r["ir"] == ir
         and abs(r["E_MPa"] - E) < 1e-9 and (r.get("noslip") or 0) == noslip]
    cr = np.array([r["creep_um_s"] for r in v if r.get("held_end") and r.get("creep_um_s") is not None])
    d0 = [x for r in v if isinstance(r.get("pad_d0"), dict) for x in r["pad_d0"].values()]
    ej = [r for r in v if r["status"] != "complete"]
    return dict(n=len(v), complete=len(v) - len(ej), held=sum(bool(r.get("held_end")) for r in v),
                creep=float(np.median(cr)) if len(cr) else None,
                creep_q=[float(np.percentile(cr, 25)), float(np.percentile(cr, 75))] if len(cr) else None,
                d0=[min(x[0] for x in d0), max(x[1] for x in d0)] if d0 else None, tr=v[0]["tr_s"] if v else None,
                ejected=[(r["plan"], r["seed"], r["t_end_s"]) for r in ej])


def grasps(E, sim, ir=10.0):
    """The scripted grasp's runs at modulus E (MPa) with a time constant that keeps d0 above 0.5 (the 10-26 runs at
    2.0 MPa and tc 10 ms diverged): held, worlds, slip and cost (scripts/ur5_shear_impratio.py mjw)."""
    v = [r for r in _rows("20261026-impratio_mjw.jsonl") if r.get("sim") == sim and r.get("ir") == ir
         and abs(r.get("E_MPa", -1) - E) < 1e-9 and (r.get("noslip") or 0) == 0 and r.get("status") != "unsupported"
         and r.get("d0") and min(r["d0"]) > 0.5]
    return dict(held=[r["held"] for r in v], n=v[0]["n"] if v else None, slip=[r["slip_um_s"]["median"] for r in v],
                us=[r.get("us_per_world_step") or r.get("us_per_step_median") for r in v],
                d0=[min(min(r["d0"]) for r in v), max(max(r["d0"]) for r in v)] if v else None)


def sections(d, N):
    J = d["fits"]["joint"]
    Fi = d["fits"]
    S = d["sessions_detail"]
    Ep, Ef, sf = J["E_pads_MPa"], J["E_field_MPa"], J["s_fe"]
    Efe = 12.0 / sf
    lo, ro = N["low"], N["rod"]
    rp, lv, rg = pooled(d, JOINT, "pads")
    rf, _, _ = pooled(d, JOINT, "field")
    re_, _, _ = pooled(d, JOINT, "fe")
    lvl = lambda r, L: med(r[lv == L])  # noqa: E731
    mid = (lv >= 1.0) & (lv <= 2.0)
    w_mid_p, w_mid_f = within(rp[mid]), within(rf[mid])
    lo_p, hi_p = lvl(rp, 0.25), lvl(rp, 8.0)
    q1n = np.nanpercentile(rp[lv == 1.0], [25, 75])
    fe_rng = [lvl(re_, L) for L in sorted(set(lv.tolist())) if (lv == L).sum() >= 8]
    p_rng = [lvl(rp, L) for L in sorted(set(lv.tolist())) if (lv == L).sum() >= 8]
    reg = {g: med(rp[rg == g]) for g in ("face", "edge", "flank")}
    old_s, old_r = Fi["old_sphere2"], Fi["old_rod1"]
    tips = [Fi[f"{t}_low_rod"]["E_pads_MPa"] for t in TIPS]
    low_t = [Fi[f"{t}_low"]["E_pads_MPa"] for t in TIPS]
    rod_t = [Fi[f"{t}_rod"]["E_pads_MPa"] for t in TIPS]
    tall = [Fi[f"{t}_tall"]["E_pads_MPa"] for t in TIPS]
    miss_tall = sum(Fi[f"{t}_tall"]["miss_pads"] for t in TIPS)
    n_tall = sum(Fi[f"{t}_tall"]["n"] for t in TIPS)
    sp = d.get("spring") or {}
    tw = N["twists"]
    dirs = {ind: [S[k]["direction_deg"] for k in keys] for ind, keys in (("low", LOW_J), ("rod", ROD_J))}
    ang_geo = {ind: [x["geometric"]["median"] for x in v] for ind, v in dirs.items()}
    ang_pad = {ind: [x["pads"]["median"] for x in v] for ind, v in dirs.items()}
    tang = {ind: [x["tangential_over_normal"]["median"] for x in v] for ind, v in dirs.items()}
    cen = {ind: [S[k]["centre_offset_um"]["pads"] for k in keys] for ind, keys in (("low", LOW_J), ("rod", ROD_J))}
    cen_f = {ind: [S[k]["centre_offset_um"]["field"] for k in keys] for ind, keys in (("low", LOW_J), ("rod", ROD_J))}
    rb = {ind: N[ind]["rbar_mm"] for ind in ("low", "rod")}
    # the grip force through the servo, with each tip compliance at 1 N in series
    srv = 1e3 / SERVO_N_PER_M                                            # mm/N
    k_ser = lambda c: 1e3 / (srv + c)  # noqa: E731
    c_meas = [lo["measured_compliance_1N_mm_N"], ro["measured_compliance_1N_mm_N"]]
    c_10 = [lo["compliance_1N_mm_N"]["pads_10MPa"], ro["compliance_1N_mm_N"]["pads_10MPa"]]
    c_fit = [lo["compliance_1N_mm_N"]["pads_fit"], ro["compliance_1N_mm_N"]["pads_fit"]]
    grip_dev = [100 * (1 - k_ser(cm) / k_ser(c1)) for cm, c1 in zip(c_meas, c_10)]
    grip_dev_fit = [100 * abs(1 - k_ser(cm) / k_ser(cf)) for cm, cf in zip(c_meas, c_fit)]
    arm10 = [rb[i]["2.5"]["pads_10MPa"] / rb[i]["2.5"]["pads_fit"] for i in ("low", "rod")]
    patch5 = 100 * (max(max(rb[i]['5'][m] for m in ('pads_fit', 'field_fit', 'fe_fit')) / min(rb[i]['5'][m] for m in ('pads_fit', 'field_fit', 'fe_fit')) for i in ('low', 'rod')) - 1)
    cal, shear = link(L["cal"], "calibration page"), link(L["shear"], "shear page")
    pm = link(L["pm"], "physical-model page")
    contact = link(L["contact"], "contact model overview")
    R10, R25 = replays(10.0), replays(E_SCENE)
    N10, N2 = replays(10.0, noslip=10), replays(2.0, noslip=10)
    G10c, G25c = grasps(10.0, "cpu"), grasps(E_SCENE, "cpu")
    G10g, G25g = grasps(10.0, "mjw"), grasps(E_SCENE, "mjw")
    ej_t = [t for _, _, t in R25["ejected"]]
    ej_hands = sorted({h for h, _, _ in R25["ejected"]})

    med_mid = med(rp[mid])
    dres = {k: S[k]["depth_resid_mm"]["pads"]["median"] * 1e3 for k in S}
    lede = (f"One modulus of {Ep:.2f}&#8202;MPa fits the 1&#8202;mm sphere pads of the scenes to three four-magnet v3 "
            "fingertips under the 5&#8202;mm sphere and the 12.5&#8202;mm rod. The scenes now run it, rounded to "
            f"{E_SCENE:g}&#8202;MPa, in place of 10&#8202;MPa. Fitted to each tip alone, the modulus is "
            f"{rng(tips)}&#8202;MPa. At 1&#8211;2&#8202;N the median measured force is {med_mid:.2f} times the pads&#8217;. At "
            f"0.25&#8202;N it is {lo_p:.2f} times, and at 8&#8202;N {hi_p:.2f} times. The finite-element (FE) pad with one "
            f"compliance scale stays within {min(fe_rng):.2f}&#8211;{max(fe_rng):.2f} at every level. With the relaxation time "
            "scaled as $1/\\sqrt{E}$, the held tool of the hand&#8217;s replays creeps "
            f"{R25['creep']:.0f}&#8202;&#181;m/s at {E_SCENE:g}&#8202;MPa against {R10['creep']:.0f}&#8202;&#181;m/s at 10&#8202;MPa.")
    summary = (f"One pad modulus, {Ep:.2f} MPa, fits the presses of three four-magnet v3 fingertips under the 5 mm sphere and "
               f"the 12.5 mm rod ({rng(tips).replace('&#8211;', '-')} MPa per tip); the scenes now run {E_SCENE:g} MPa with a "
               f"{TR_SCENE:g} s relaxation time. At 1-2 N the median force ratio is {med_mid:.2f} and at 8 N {hi_p:.2f}; the FE "
               f"with one scale follows the load trend. The held tool creeps {R25['creep']:.0f} um/s at {E_SCENE:g} MPa against "
               f"{R10['creep']:.0f} at 10 MPa (impratio 10).")

    S_ = []
    terms = [
        ("Press, hold", "A press moves the fingertip&#8217;s pad onto a fixed indenter along the target&#8217;s surface normal "
         "and stops at force levels from 0.25 to 8&#8202;N. Each level is held for about 1.5&#8202;s, which is called a hold. An "
         "OptoForce HEX-E six-axis force-torque sensor between the UR5&#8217;s flange and the finger measures the force."),
        ("Low sphere, tall sphere, rod", "The indenters, all printed in PLA. The low sphere has a radius of 5&#8202;mm on a short "
         "stand. The tall sphere has the same radius on a 160&#8202;mm stand, so that the robot reaches the flanks and the "
         "tip&#8217;s end. The rod is a cylinder of radius 12.5&#8202;mm."),
        ("Registered indentation", "The depth of the indenter below the pad&#8217;s nominal surface at a hold, in mm. It is "
         "computed from the robot&#8217;s pose, the registration of the tool and the indenter, and the HEX-E&#8217;s own "
         "deflection. Any other compliance between the pad and the robot is included in it."),
        ("Pads", "The builder-v2 sphere pads of the simulations, 1&#8202;mm apart on the FE model&#8217;s surface points (812 "
         "on this pad). Pad $i$ pushes with $K_i o_i$, where $o_i$ is its overlap with the indenter and "
         "$K_i=E\\,a_i/h_i$. Here $a_i$ is the pad&#8217;s share of the surface, and $h_i$ is the depth of surface point $i$ "
         "below the volume centroid of the whole tip, as the scenes build them."),
        ("Drake&#8217;s field", "Drake&#8217;s compliant-convex pressure field, $p=E\\,(1-g)$, where $g$ is the gauge "
         "of the pad&#8217;s convex surface about the centroid of the pad solid. It is integrated over the part of the "
         "indenter&#8217;s surface inside the pad, at points about 0.1&#8202;mm apart."),
        ("FE pad", "The finite-element model of the printed four-magnet pad (FE case b_05R4). Its walls and skins are solid "
         "TPU at 12&#8202;MPa around a homogenized gyroid core, and a frictionless rigid indenter presses it. Its compliance "
         "is scaled by $s$, and its modulus is quoted as $12/s$&#8202;MPa."),
        ("Joint modulus", f"One value per model, fitted over the {J['n']:,} holds of the low sphere and the rod of the three v3 "
         "tips. It is the exponential of the median of $\\ln(F_\\text{meas}/F_\\text{model})$, with each model at the "
         "indenter&#8217;s registered position. The FE&#8217;s scale $s$ is fitted the same way."),
        ("Force ratio", "The measured force over the model&#8217;s force at the registered position, at the joint modulus. "
         "One is a perfect fit."),
        ("Depth residual", "The distance along the contact normal, in &#181;m, by which the indenter would have to move from "
         "its registered position for the model&#8217;s force to equal the measured one. It is positive when the print "
         "sinks deeper than the model."),
        ("Mean patch radius $\\bar r$", "The pressure-weighted mean distance of the contact patch from its pressure "
         "centre, in the plane normal to the contact, in mm. At full slip the friction torque about the normal is "
         "$\\mu F \\bar r$."),
        ("Twist torsion", "The torque about the contact normal during a twist of the finger about the normal, in N&#8202;mm. "
         "It is half the difference between the outbound and the return hold at the same angle, so the sensor&#8217;s "
         "crosstalk from the push cancels."),
        ("Friction coefficient $\\mu$", f"Fitted to the UR5 slides of the same 85A pad material on a bare PLA sphere ({shear}): "
         f"{d['mu']['1']:.2f} at 1&#8202;N and {d['mu']['2.5']:.2f} from 2&#8202;N. In the scenes it is 1.0 to 2.4 on the tool, for "
         "the pad on the grip tape of the bench tools."),
        ("Force direction error", "The angle between the measured force and the model&#8217;s force, in degrees."),
        ("Time constant $t_c$, relaxation time $t_r$", "MuJoCo reaches each pad&#8217;s stiffness through its contact rows: "
         "the time constant $t_c$ of their reference acceleration (solref) and the impedance $d_0$ (solimp). The relaxation "
         "time is $t_r=2t_c$, in s."),
        ("Impratio", "MuJoCo&#8217;s ratio of the friction rows&#8217; impedance to the normal rows&#8217;. The policy trainer "
         "runs 10."),
        ("Creep", "The speed at which a held tool slides over the fingertips while every contact stays below the friction "
         "limit, in &#181;m/s. In the D1&#8211;D8 replays it is the tool&#8217;s drift over the last 4&#8202;s, after the hold, "
         "divided by that time."),
    ]
    S_.append(("terms", "Terms and metrics", '<dl class="terms">' + "".join(f"<dt>{t}</dt><dd>{x}</dd>" for t, x in terms)
               + "</dl>"))

    # ---- presses and models
    rows = []
    for k in [f"{t}_{i}" for i in ("low", "tall", "rod") for t in TIPS] + ["old_sphere2", "old_rod1"]:
        m = d["sessions"][k]
        lvs = sorted(set(S[k]["holds"]["level"]))
        rows.append([m["tip"], {"low": "Low sphere", "tall": "Tall sphere", "rod": "Rod"}[m["indenter"]],
                     P.code(Path(m["holds"]).name), f"{m['n']}", f"{min(lvs):g}&#8211;{max(lvs):g}"])
    t1 = P.table(["Tip", "Indenter", "Holds file", "Press holds", "Levels (N)"], rows, num_cols=(3,), n=1,
                 caption="UR5 sessions in the fit. The v3 sessions ran from 27 to 30 October 2026; tip 0&#8217;s sphere2 and rod1 "
                 "ran on 22 and 23 October and are shown for comparison. Press holds over 0.2&#8202;N.")
    S_.append(("data", "Presses and models", (
        "<p>Each v3 tip has three UR5 sessions: the low sphere, the tall sphere and the rod (Table&#160;1). Their holds and "
        f"registration are described on the {cal}, Sections 10&#8211;12. v3 ID 3&#8217;s sessions of 29 and 30 October were "
        "analysed for this page from the lab computer&#8217;s session folders, with the options of ID 1 and ID 2. Every press "
        "hold over 0.2&#8202;N is included in the fit: "
        f"{sum(d['sessions'][k]['n'] for k in d['sessions'] if k.startswith('v3')):,} holds on the three v3 tips. The model of "
        "each hold puts the indenter at its registered position in the pad&#8217;s frame. The three models share the pad&#8217;s "
        "surface, so they differ only in how a contact pushes back.</p>"
        "<p>The pads and Drake&#8217;s field are both local laws: the pressure at a point is set by the overlap at that point. "
        "The FE pad is elastic: a load at one surface point moves its neighbours, so a press spreads over the walls and "
        "skins. Each model&#8217;s force scales with its modulus, so one computation per hold is enough for every modulus. "
        "The modulus is fitted to the force at the registered position, as the owner asked (forces before indentation). The "
        "same comparison in millimetres is the depth residual.</p>" + t1)))

    # ---- one modulus
    def frow(name, f, n_=None):
        return [name, f"{f['n'] if n_ is None else n_:,}", f"{f['E_pads_MPa']:.2f}", f"{f['E_field_MPa']:.2f}",
                f"{f['s_fe']:.2f}", f"{f['E_fe_MPa']:.1f}"]
    rows = ([frow(f"Low sphere, v3 ID {i}", Fi[f"v3id{i}_low"]) for i in (1, 2, 3)] + [frow("Low sphere, all three", Fi["v3_low"])]
            + [frow(f"Rod, v3 ID {i}", Fi[f"v3id{i}_rod"]) for i in (1, 2, 3)] + [frow("Rod, all three", Fi["v3_rod"]),
               frow("<b>Low sphere and rod, all three (joint)</b>", J)]
            + [frow(f"Tall sphere, v3 ID {i}", Fi[f"v3id{i}_tall"]) for i in (1, 2, 3)]
            + [frow("Tip 0, sphere2 (22 October)", old_s), frow("Tip 0, rod1 (23 October)", old_r)])
    t2 = P.table(["Sessions fitted", "Press holds", "Pads E (MPa)", "Field E (MPa)", "FE scale s", "FE modulus 12/s (MPa)"], rows,
                 num_cols=(1, 2, 3, 4, 5), n=2,
                 caption="One modulus per model fitted to the measured force at the registered positions. Bold: the joint "
                 "fit that Figures 1&#8211;3 use.").replace("<th class=n>", '<th class=n style="white-space:normal">')
    lv_rows = []
    for Lv in (0.25, 0.5, 1.0, 2.0, 3.5, 5.0, 8.0):
        m_ = lv == Lv
        if m_.sum() >= 8:
            lv_rows.append([f"{Lv:g}", f"{m_.sum()}", f2(lvl(rp, Lv)), f2(lvl(rf, Lv)), f2(lvl(re_, Lv)),
                            f"{100 * within(rp[m_]):.0f}"])
    t3 = P.table(["Level (N)", "Holds", "Pads", "Field", "FE, one scale", "Pads within 25&#8202;% (%)"], lv_rows,
                 num_cols=(0, 1, 2, 3, 4, 5), n=3,
                 caption="Force ratio (measured over model) at the joint moduli by load level: medians over the low sphere and "
                 "the rod of the three v3 tips.").replace("<th class=n>", '<th class=n style="white-space:normal">')
    stiff = TIPS[int(np.argmax(low_t))].replace("v3id", "v3 ID ")
    S_.append(("modulus", "One modulus against the presses", (
        f"<p>Fitted to each indenter alone, the pads&#8217; modulus is {Fi['v3_low']['E_pads_MPa']:.2f}&#8202;MPa under the low "
        f"sphere and {Fi['v3_rod']['E_pads_MPa']:.2f}&#8202;MPa under the rod (Table&#160;2). Fitted to both, it is {Ep:.2f}&#8202;MPa. "
        f"Fitted to each tip alone, it is {tips[0]:.2f}, {tips[1]:.2f} and {tips[2]:.2f}&#8202;MPa for v3 ID 1, ID 2 and ID 3. Under "
        f"the rod the tips&#8217; moduli are {rng(rod_t)}&#8202;MPa. Under the sphere they are {rng(low_t)}&#8202;MPa, and "
        f"{stiff}&#8217;s is the highest. On tip 0 the fitted moduli were {old_s['E_pads_MPa']:.2f}&#8202;MPa under the sphere and "
        f"{old_r['E_pads_MPa']:.2f}&#8202;MPa under the rod, a ratio of {old_s['E_pads_MPa'] / old_r['E_pads_MPa']:.2f}. On the v3 "
        f"tips the ratio is {Fi['v3_low']['E_pads_MPa'] / Fi['v3_rod']['E_pads_MPa']:.2f}. At the joint modulus the median depth "
        f"residual of the v3 sessions is {rng([dres[k] for k in JOINT], 0)}&#8202;&#181;m. On tip 0&#8217;s sphere2 it is "
        f"{dres['old_sphere2']:+.0f}&#8202;&#181;m, so that print sank about 0.2&#8202;mm deeper at every load. For Drake&#8217;s field "
        f"the joint modulus is {Ef:.2f}&#8202;MPa, and the FE&#8217;s joint scale is {sf:.2f} ({Efe:.1f}&#8202;MPa).</p>"
        "<p>The pads measure each depth $h_i$ from the centroid of the whole tip, as the scenes build them. The fit of 30 "
        "October at 15:19 measured it from the centroid of the pad solid, which lies nearer the face. On those pads the joint "
        f"modulus is {J['E_pads_padcentroid_MPa']:.2f}&#8202;MPa over the three tips, and it was 1.67&#8202;MPa over the first "
        "two.</p>"
        f"<p>Under the tall sphere the fitted modulus is {rng(tall)}&#8202;MPa. In {miss_tall} of the {n_tall} tall holds of the "
        "three tips, the sphere at its registered position does not reach the pad&#8217;s surface, although the HEX-E measured "
        "a force. On v3 ID 1 and ID 2 the magnetic readout put the tall&#8217;s contacts 0.75&#8211;0.98&#8202;mm from its "
        f"registration, with the cause open ({cal}, Section 12). The tall sessions are left out of the joint fit.</p>" + t2 +
        P.fig(1, F["force"], "Normal force against registered indentation at the face. Dots: the face holds of v3 ID 1 "
              "(light grey), ID 2 (mid grey) and ID 3 (dark grey). Lines: medians over the face targets of each model&#8217;s force, with the "
              f"indenter moved along the target&#8217;s normal. The pads are at {Ep:.2f}&#8202;MPa and Drake&#8217;s field at "
              f"{Ef:.2f}&#8202;MPa, their joint moduli. The FE is drawn at its joint scale ({Efe:.1f}&#8202;MPa) and as computed "
              "(12&#8202;MPa). The dotted orange line is the pads at 10&#8202;MPa, the scenes&#8217; modulus before 30 October. "
              "(a) The low 5&#8202;mm sphere. (b) The 12.5&#8202;mm rod. Script: "
              + P.code("scripts/v3_tip_contact_model_figures.py") + ".") +
        f"<p>One modulus fits the middle of the range and misses the ends (Figure&#160;2, Table&#160;3). At 1&#8211;2&#8202;N the "
        f"median force ratio is {med_mid:.2f}. There {100 * w_mid_p:.0f}&#8202;% of the pads&#8217; forces and "
        f"{100 * w_mid_f:.0f}&#8202;% of the field&#8217;s lie within 25&#8202;% of the measured force. At 0.25&#8202;N the measured "
        f"force is {lo_p:.2f} times the pads&#8217;. At 8&#8202;N it is {hi_p:.2f} times, so the pads push {1 / hi_p:.1f} times "
        "as hard. The light end depends on the registration. At 0.25&#8202;N the indenter is 0.2&#8211;0.3&#8202;mm deep, so a "
        "registration error of 0.05&#8202;mm changes the pads&#8217; force by 30&#8211;50&#8202;%. The heavy end does not. The "
        f"scatter at one level is wide: at 1&#8202;N the quartiles of the ratio span {q1n[0]:.2f}&#8211;{q1n[1]:.2f}. By region "
        f"the median ratio is {reg['face']:.2f} on the face, {reg['edge']:.2f} on the edges and {reg['flank']:.2f} on the "
        "flanks.</p>"
        "<p>The FE pad with one scale follows the load trend. From 0.25 to 8&#8202;N its force ratio stays within "
        f"{min(fe_rng):.2f}&#8211;{max(fe_rng):.2f}, and the pads&#8217; spans {min(p_rng):.2f}&#8211;{max(p_rng):.2f}. In the FE a "
        "press is spread by the walls and skins, so its force grows more slowly with depth than a local law&#8217;s. The FE "
        f"calibration of 18 October found this difference between the FE and the local laws ({link(L['fec'], 'FE calibration')}). "
        f"The print now shows it as well. The FE&#8217;s scale at the contact, {sf:.2f}, differs from its scale in the bulk. "
        "The magnets of the v3 tips move 0.71&#8211;0.74 times as far as the FE predicts, a modulus of "
        f"16.3&#8211;16.9&#8202;MPa ({pm}, Section 8). The print is therefore about {sf:.0f} times softer than its FE at the "
        "contact and 1.4 times stiffer in the bulk.</p>"
        + P.fig(2, F["ratio"], "Measured force over model force at the joint moduli, by load level. Medians with quartiles over "
                "the holds of the three v3 tips. Filled circles: the low 5&#8202;mm sphere. Open squares: the 12.5&#8202;mm rod. "
                "Grey band: within 25&#8202;% (0.8&#8211;1.25).") + t3)))

    # ---- series spring
    if sp:
        jt = sp["joint"]
        at5 = {ind: min(sp[ind]["grid"], key=lambda r: abs(r["E_MPa"] - 5.0)) for ind in ("low", "rod")}
        S_.append(("spring", "Modulus with a series spring", (
            "<p>A spring in series with the pads adds an indentation $a+cF$ to the pads&#8217; own. Fitted together with "
            f"the modulus to both indenters, it gives {jt['E_MPa']:.2f}&#8202;MPa behind {jt['c_mm_N']:.3f}&#8202;mm/N, with an "
            f"offset of {jt['a_mm']:.2f}&#8202;mm. The depth residual falls from {jt['alone']['rms_mm'] * 1e3:.0f}&#8202;&#181;m RMS, "
            f"at the best modulus alone ({jt['alone']['E_MPa']:.2f}&#8202;MPa), to {jt['rms_mm'] * 1e3:.0f}&#8202;&#181;m. Fitted to "
            "each indenter alone, the residual keeps falling as the modulus grows. Above 5&#8202;MPa it changes by less than "
            f"10&#8202;&#181;m. At {at5['low']['E_MPa']:.1f}&#8202;MPa the spring is {at5['low']['c_mm_N']:.3f}&#8202;mm/N under the "
            f"sphere and {at5['rod']['c_mm_N']:.3f}&#8202;mm/N under the rod. A spring in the mount, the arm or the DIP link "
            "would add the same compliance under both indenters. The fitted compliance depends on the size of the contact, as "
            "on tip 0 (0.20 and 0.10&#8202;mm/N), so most of the extra give arises at the contact. The scenes keep one modulus "
            "without a spring.</p>")))

    # ---- patch and torsion
    t5row = []
    for key, name in (("low_1", "Sphere, 1&#8202;N level"), ("low_2.5", "Sphere, 2.5&#8202;N level"), ("low_5", "Sphere, 5&#8202;N level"),
                      ("rod_2.5", "Rod, 2.5&#8202;N level"), ("rod_5", "Rod, 5&#8202;N level")):
        if key in tw:
            x = tw[key]
            t5row.append([name, f"{x['n']}", f"{x['f_n']:.2f}", f"{x['M_median']:.2f}", f"{x['M_p25']:.2f}&#8211;{x['M_p75']:.2f}",
                          f"{x['pads']:.2f}", f"{x['field']:.2f}"])
    t4 = P.table(["Twist", "Pairs", "Normal force (N)", "Torsion, median (N&#8202;mm)", "Torsion, quartiles (N&#8202;mm)",
                  "Pads, $\\mu F\\bar r$ (N&#8202;mm)", "Field, $\\mu F\\bar r$ (N&#8202;mm)"], t5row,
                 num_cols=(1, 2, 3, 4, 5, 6), n=4,
                 caption="Twist torsion of the three v3 tips against the friction torque at full slip of the pads and the "
                 "field. Each model is at its joint modulus and at the twist&#8217;s measured normal force.").replace(
                     "<th class=n>", '<th class=n style="white-space:normal">')
    t5 = tw["low_5"]
    S_.append(("patch", "Contact patch and friction torque", (
        "<p>In the local laws the patch is the overlap of the undeformed shapes at the indentation. At a given force each "
        "model has its own indentation, and so its own patch. At the joint moduli the pads&#8217; mean patch radius under the "
        f"sphere is {rb['low']['1']['pads_fit']:.2f}, {rb['low']['2']['pads_fit']:.2f} and {rb['low']['5']['pads_fit']:.2f}&#8202;mm "
        f"at 1, 2 and 5&#8202;N. The field&#8217;s is {rb['low']['1']['field_fit']:.2f}, {rb['low']['2']['field_fit']:.2f} and "
        f"{rb['low']['5']['field_fit']:.2f}&#8202;mm (Figure&#160;3a). The FE, an elastic body, makes a smaller patch at the same "
        "depth but sinks deeper at high force. At 5&#8202;N its patch radius is "
        f"{rb['low']['5']['fe_fit']:.2f}&#8202;mm under the sphere and {rb['rod']['5']['fe_fit']:.2f}&#8202;mm under the rod. The "
        f"pads&#8217; are {rb['low']['5']['pads_fit']:.2f} and {rb['rod']['5']['pads_fit']:.2f}&#8202;mm, and the three models "
        f"differ by at most {patch5:.0f}&#8202;%. Below 2&#8202;N the FE&#8217;s surface nodes, 1&#8202;mm apart, are too coarse to "
        "resolve its patch.</p>"
        "<p>The friction torque at full slip measures $\\bar r$. The UR5 twists of the three v3 tips do not resolve it "
        f"(Table&#160;4, Figure&#160;3b). In the twists of the 5&#8202;N level, at a median normal force of {t5['f_n']:.1f}&#8202;N, "
        f"the torsion under the sphere is {t5['M_median']:.2f}&#8202;N&#8202;mm, near the pads&#8217; {t5['pads']:.2f}&#8202;N&#8202;mm. "
        f"The quartiles of the pairs span {t5['M_p25']:.2f}&#8211;{t5['M_p75']:.2f}&#8202;N&#8202;mm, and single pairs reach "
        f"{t5['M_min']:.1f} and {t5['M_max']:.1f}&#8202;N&#8202;mm. In these twists the face&#8217;s normal lies "
        "across the HEX-E&#8217;s axis. Between holds 5&#176; apart the torque reading changes by up to 10&#8202;N&#8202;mm. The "
        "torque that would separate the models is 1&#8211;3&#8202;N&#8202;mm. In the inverted rig the probe sits on the "
        "HEX-E and the tip in a seat, so the contact normal lies along the sensor&#8217;s axis. There the push has no moment "
        "about that axis (Future work).</p>"
        + P.fig(3, F["patch"], "(a) Mean patch radius against normal force at the face, each model at its joint modulus or "
                "scale. Solid lines: the 5&#8202;mm sphere. Dashed: the 12.5&#8202;mm rod. The FE is drawn from 2&#8202;N. (b) Twist "
                "torsion of the three v3 tips (dots) against normal force. The lines are the pads&#8217; and the field&#8217;s "
                "friction torque at full slip under the sphere, $\\mu F\\bar r$.") + t4)))

    # ---- location and direction
    rows = [[{"low": "5&#8202;mm sphere", "rod": "12.5&#8202;mm rod"}[ind], rng(ang_geo[ind], 1), rng(ang_pad[ind], 1),
             rng(tang[ind], 2), rng(cen[ind], 0), rng(cen_f[ind], 0)] for ind in ("low", "rod")]
    t5t = P.table(["Indenter", "Measured force to the surface normal (&#176;)", "Measured force to the pads&#8217; force (&#176;)",
                   "Shear over normal force", "Pads&#8217; centre to contact point (&#181;m)", "Field&#8217;s centre to contact point (&#181;m)"],
                  rows, num_cols=(1, 2, 3, 4, 5), n=5,
                  caption="Contact pose at the registered positions: medians over the press holds of v3 ID 1 to 3, as "
                  "ranges over the three tips. Distances in the plane normal to the contact.").replace(
                      "<th class=n>", '<th class=n style="white-space:normal">')
    S_.append(("pose", "Contact location and force direction", (
        "<p>The registered contact point is the pad&#8217;s surface point under the indenter&#8217;s centre. The pads place "
        f"their pressure centre {rng(cen['low'] + cen['rod'], 0)}&#8202;&#181;m from it, and Drake&#8217;s field "
        f"{rng(cen_f['low'] + cen_f['rod'], 0)}&#8202;&#181;m (Table&#160;5). Both come from the same surface, so this compares the "
        "models with the geometry. On the print the contact location is known only through the registration, to "
        "0.04&#8211;0.07&#8202;mm RMS.</p>"
        f"<p>The measured force lies {rng(ang_geo['low'] + ang_geo['rod'], 1)}&#176; from the surface normal. The pads&#8217; force "
        "is as far from it. The angle is the shear that the robot held during the press, "
        f"{rng(tang['low'] + tang['rod'], 2)} of the normal force, and the frictionless comparison leaves it out. The direction "
        "of a simulated contact force therefore follows from the pad&#8217;s surface and the friction.</p>" + t5t)))

    # ---- the scenes
    def cell_r(R):
        return [f"{R['complete']} of {R['n']}", f"{R['held']} of {R['n']}",
                f"{R['creep']:.0f} ({R['creep_q'][0]:.0f}&#8211;{R['creep_q'][1]:.0f})", f"{R['d0'][0]:.2f}&#8211;{R['d0'][1]:.2f}"]
    c10, c25 = cell_r(R10), cell_r(R25)
    g = lambda G, nn: (f"{rng(G['held'], 0)} of {nn}", rng(G["slip"], 0))  # noqa: E731
    h10c, s10c = g(G10c, G10c["n"])
    h25c, s25c = g(G25c, G25c["n"])
    h10g, s10g = g(G10g, G10g["n"])
    h25g, s25g = g(G25g, G25g["n"])
    t6 = P.table(["Quantity", f"10&#8202;MPa, $t_r$ 0.02&#8202;s", f"{E_SCENE:g}&#8202;MPa, $t_r$ {TR_SCENE:g}&#8202;s"], [
        ["D1&#8211;D8 replays completed", c10[0], c25[0]],
        ["D1&#8211;D8 replays holding the tool to the end", c10[1], c25[1]],
        ["Creep of the held tool, median (quartiles), &#181;m/s", c10[2], c25[2]],
        ["Pad $d_0$ in the D1&#8211;D8 scenes", c10[3], c25[3]],
        ["Scripted grasp, CPU: grasps held", h10c, h25c],
        ["Scripted grasp, CPU: slip, &#181;m/s", s10c, s25c],
        ["Scripted grasp, MuJoCo-Warp: worlds held", h10g, h25g],
        ["Scripted grasp, MuJoCo-Warp: slip, &#181;m/s", s10g, s25g],
        ["MuJoCo-Warp cost, &#181;s per world-step", rng(G10g["us"], 1), rng(G25g["us"], 1)],
    ], num_cols=(), n=6, caption="The two pad settings under the policy trainer&#8217;s numerics at impratio 10, on pads of this "
        "tip&#8217;s shape (rows in " + P.code("docs/experiments/20261026-shear_friction_impratio/data/") + ", from "
        + P.code("scripts/ur5_shear_impratio.py") + "). Ranges at 10&#8202;MPa are over the repeated runs of 26 October.")
    S_.append(("hand", "The fitted modulus in the scenes", (
        f"<p>Since 30 October the scenes build their pads at {E_SCENE:g}&#8202;MPa, the joint modulus rounded "
        "(" + P.code("E_TPU") + " in " + P.code("scripts/reorient_backends.py") + "; owner: &#8220;seems like we should change "
        "the modulus to match the real setting&#8221;). MuJoCo reaches a pad&#8217;s stiffness $K_i$ through the impedance "
        "$d_0=1-1/(t_c^2K_iA)$, where $A$ is the inverse weight of the fingertip and the tool. A softer pad therefore needs a "
        "longer time constant. The relaxation time scales as $t_r=0.02\\,\\mathrm{s}\\,\\sqrt{10\\,\\mathrm{MPa}/E}$, which gives "
        f"{TR_SCENE:g}&#8202;s at {E_SCENE:g}&#8202;MPa. Every pad then keeps the $d_0$ it had at 10&#8202;MPa: "
        f"{R25['d0'][0]:.2f}&#8211;{R25['d0'][1]:.2f} in the D1&#8211;D8 scenes and {G25g['d0'][0]:.2f} in the scripted grasp. Runs "
        "before 30 October used 10&#8202;MPa and 0.02&#8202;s, and their frozen scenes keep both.</p>"
        f"<p>Through the servos each fingertip is a spring of {SERVO_N_PER_M:.0f}&#8202;N/m along its palmar normal "
        f"({link(L['sens'], 'contact sensitivity page')}). At 1&#8202;N the print&#8217;s face adds "
        f"{c_meas[0]:.3f}&#8202;mm/N under the sphere and {c_meas[1]:.3f}&#8202;mm/N under the rod. The 10&#8202;MPa pads added "
        f"{c_10[0]:.3f} and {c_10[1]:.3f}&#8202;mm/N, which changed a commanded squeeze&#8217;s force by {rng(grip_dev, 0)}&#8202;%. "
        f"At the joint modulus the change is {rng(grip_dev_fit, 0)}&#8202;%. The patch changes more. At 2.5&#8202;N the "
        f"10&#8202;MPa pads had {rng(arm10, 2)} of the patch radius at the joint modulus, and the friction torque at full slip "
        "scales with that radius.</p>"
        "<p>Table&#160;6 compares the two settings under the policy trainer&#8217;s numerics: a 2&#8202;ms step, the "
        "implicitfast integrator, the elliptic cone, Newton&#8217;s method with 10 iterations and impratio 10 "
        f"({shear}, Section 3). In the 24 D1&#8211;D8 replays at {E_SCENE:g}&#8202;MPa, {R25['held']} held the tool to the end, "
        f"against {R10['held']} at 10&#8202;MPa. Hands {' and '.join(ej_hands)} drop the tool at both moduli. In "
        f"{len(R25['ejected'])} of their six runs at {E_SCENE:g}&#8202;MPa the tool left the fingertips at speed, and its spin "
        f"diverged in free flight at {min(ej_t):.1f}&#8211;{max(ej_t):.1f}&#8202;s. At 10&#8202;MPa none diverged. The scripted "
        f"grasp in MuJoCo-Warp held in {h25g} worlds at {E_SCENE:g}&#8202;MPa and {h10g} at 10&#8202;MPa, at the same cost per "
        "world-step.</p>"
        + t6 +
        "<p>Below slip the friction rows of MuJoCo&#8217;s soft contact act as dampers. A steady tangential force $F_t$ then "
        "slides the contact at about $v=F_t/(2\\,\\mathrm{impratio}\\;t_c\\sum_iK_i)$, with the sum over the pads in contact. "
        "For a curved tip on a face $\\sum_iK_i$ grows as $\\sqrt{E}$ at a given force, so the scaled time constant leaves $v$ "
        f"unchanged. In the replays the held tool creeps a median {R25['creep']:.0f}&#8202;&#181;m/s at {E_SCENE:g}&#8202;MPa and "
        f"{R10['creep']:.0f}&#8202;&#181;m/s at 10&#8202;MPa. Every setting that lowers $v$ also lowers the friction rows&#8217; "
        "regulariser, $R\\propto 1/(\\mathrm{impratio}\\;t_c^2K_i)$. From impratio 20 on, the CPU replays then diverge after a "
        f"drop ({shear}, Section 3). MuJoCo&#8217;s noslip pass removes the creep without a smaller regulariser. With 10 noslip "
        f"iterations the replays creep {N10['creep']:.0f}&#8202;&#181;m/s at 10&#8202;MPa and {N2['creep']:.0f}&#8202;&#181;m/s at "
        f"2.0&#8202;MPa, and none of the 24 runs at each modulus diverges. MuJoCo-Warp 3.6 refuses noslip "
        "(" + P.code("put_model") + " raises NotImplementedError), so the GPU trainer keeps the creep.</p>"
        "<p>The scenes&#8217; friction coefficient, 1.0 to 2.4 on the tool, stands for the pad on the rubber grip tape of the "
        "bench tools. The UR5 slides measured the pad on bare PLA (owner, 2026-10-30).</p>")))

    # ---- limitations and future work
    S_.append(("limits", "Limitations", (
        "<p>The registered indentation includes every compliance between the pad and the robot&#8217;s pose beyond the "
        "HEX-E&#8217;s own, and the fit includes it in the modulus. The light holds depend on the registration to within "
        "0.05&#8202;mm. The tall sphere&#8217;s registration is unresolved. With the face&#8217;s normal across the HEX-E&#8217;s "
        "axis the twists cannot measure the friction torque. The FE pad&#8217;s surface nodes are 1&#8202;mm apart. v3 ID 3&#8217;s "
        "holds were analysed for this page with the options of ID 1 and ID 2, before the UR5 coordinator&#8217;s own analysis. "
        f"The replays at {E_SCENE:g}&#8202;MPa ran on the D1&#8211;D8 plans of the real_v1 hands with this tip&#8217;s shape, and the "
        "CAD hand&#8217;s scenes were not replayed. The friction coefficient on the grip tape is unmeasured.</p>")))
    fut = [
        ("Friction torque on the HEX-E&#8217;s axis.", "In the inverted rig&#8217;s probe mode, press the 5&#8202;mm sphere probe on "
         "the face centre of a seated v3 tip at 1, 2.5 and 5&#8202;N. At each force, twist it about the sensor&#8217;s axis "
         f"5&#8211;20&#176; out and back, five repeats ({link(L['rig'], 'inverted rig folder')}). Read the torsion as half the "
         f"out-and-back difference. The pads predict {tw['low_1']['pads']:.2f}, {tw['low_2.5']['pads']:.2f} and "
         f"{tw['low_5']['pads']:.2f}&#8202;N&#8202;mm at the twists&#8217; forces. A median outside half to twice these falsifies "
         "the pads&#8217; patch."),
        ("Contact area under a flat.", "Press the face of a v3 tip on a glass plate at 0.5, 1, 2 and 5&#8202;N, and photograph "
         "the patch through the glass. An inked 5&#8202;mm sphere pressed at the same forces gives a print of the patch. At "
         f"1&#8202;N under the sphere, a local law at the joint modulus predicts a mean patch radius of "
         f"{rb['low']['1']['pads_fit']:.2f}&#8202;mm. An elastic half-space at the same indentation predicts a contact radius "
         "$1/\\sqrt 2$ times the overlap&#8217;s."),
        ("Friction on the grip tape.", "Run the UR5 slides of session cb2t at 1 and 2&#8202;N on a 5&#8202;mm sphere wrapped in "
         "the bench tools&#8217; grip tape. A fitted coefficient below 0.8 at either force makes the scenes&#8217; 1.0 too high."),
        ("Creep against the bench.", "Compare the tool&#8217;s drift in the bench&#8217;s held phases, tracked by AprilTags "
         f"({shear}, Future work), with the replays&#8217; {R25['creep']:.0f}&#8202;&#181;m/s at {E_SCENE:g}&#8202;MPa. A bench "
         "drift under 20&#8202;&#181;m/s calls for noslip in CPU evaluation."),
        ("An elastic contact model at the task level.", "The FE&#8217;s surface compliance with one scale follows the measured "
         "load trend. Its contact solve took about 1&#8202;ms per press on one CPU core here. Run the D1&#8211;D8 replays with "
         "that solve in place of the pads at one fingertip. Keep it only if the tool&#8217;s trajectory moves by more than "
         "the spread between placements."),
    ]
    S_.append(("future", "Future work", "<ul>" + "".join(f"<li><b>{a}</b> {b}</li>" for a, b in fut) + "</ul>"))
    return S_, lede, summary


F = dict(force=MEDIA / "20261030-force_indentation.png", ratio=MEDIA / "20261030-force_ratio_by_level.png",
         patch=MEDIA / "20261030-patch_and_twists.png", film=MEDIA / "20261030-patch_film.mp4",
         poster=MEDIA / "20261030-patch_film_poster.png")


def build():
    d = json.loads((DATA / "20261030-contact_model_fit.json").read_text())
    N = json.loads((DATA / "20261030-numbers.json").read_text())
    S, lede, summary = sections(d, N)
    import re
    minus = lambda t: re.sub(r"(?<![\w\-#&;])-(?=\d)", "&#8722;", t)  # noqa: E731   (a minus sign before a number)
    S = [(sid, hd, minus(body)) for sid, hd, body in S]
    lede = minus(lede)
    J = d["fits"]["joint"]
    film_html = P.film(0, F["film"], F["poster"], (
        "Contact patches at the face centre of v3 ID 1 under the 5&#8202;mm sphere, at equal normal force from 0.25 to "
        f"8&#8202;N and back. Left: the 1&#8202;mm pads at {J['E_pads_MPa']:.2f}&#8202;MPa (discs, area by force). Middle: Drake&#8217;s "
        f"field at {J['E_field_MPa']:.2f}&#8202;MPa on the sphere&#8217;s surface. Right: the FE pad&#8217;s loaded nodes at its "
        "joint scale. Dashed circles have the mean patch radius about the pressure centre.")).replace(
            'id="fig0"', 'id="film"').replace("Figure 0. ", "Film. ")
    toc = "\n".join(f'<li><a href="#{sid}">{k}. {P.esc(hd)}</a></li>' for k, (sid, hd, _) in enumerate(S, 1))
    secs = "\n".join(f'<section id="{sid}"><h2>{k}. {P.esc(hd)}</h2>{body}</section>' for k, (sid, hd, body) in enumerate(S, 1))
    byline = ("UR5 sessions of v3 ID 1, ID 2 and ID 3 (27&#8211;30 October 2026) and tip 0&#8217;s sphere2 and rod1 (22&#8211;23 "
              "October). Fit by " + P.code("scripts/v3_tip_contact_model.py") + ", figures and film by "
              + P.code("scripts/v3_tip_contact_model_figures.py") + ", replays at the scenes&#8217; modulus by "
              + P.code("scripts/ur5_shear_impratio.py") + ", page by " + P.code("scripts/v3_tip_contact_model_page.py")
              + ". The topic overview is " + link(L["contact"], "the contact model overview") + "; the fingertip&#8217;s sensing "
              "is on " + link(L["overview"], "printed fingertips and force sensing") + ".")
    page = TEMPLATE.read_text()
    for k, v in {"TITLE": "V3 fingertip contact models",
                 "H1": "Contact models of the four-magnet fingertips fitted to the UR5 presses",
                 "EYEBROW": "Fingertip contact model &#183; 2026-10-30", "LEDE": lede, "BYLINE": byline,
                 "FILM": film_html, "TOC": toc, "SECTIONS": secs, "FOOTER": str(PAGE.relative_to(ROOT)),
                 "TEXCSS": texsvg.CSS}.items():
        page = page.replace("{{" + k + "}}", v)
    items = [((m.group(1) if m.group(1) is not None else m.group(2)).strip(), m.group(1) is not None)
             for m in P.TEX_RE.finditer(page)]
    svgs = iter(texsvg.render(items, cache_path=TEX_CACHE))
    page = P.TEX_RE.sub(lambda m: next(svgs), page)
    page = retro_style.apply(page)
    page = docs_index.tag(page, topic="contact", summary=summary, date="2026-10-30")
    PAGE.write_text(page)
    print(f"{PAGE.relative_to(ROOT)}: {PAGE.stat().st_size / 1e6:.2f} MB, {len(items)} formulas", flush=True)


if __name__ == "__main__":
    build()
