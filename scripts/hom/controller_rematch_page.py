#!/usr/bin/env python3
"""Result page of 2026-11-20: the inner-loop controllers on the calibrated SR2 plant, each with its contact-force knob
tuned (owner, 2026-11-19 23:5x and 11-20 08:0x): the calibrated plant on the GPU and its check against the CPU plant,
HOM's commanded force, the tag tracker's rate and the pose filter, the sampling planners and UHAS, and a clearance term
in HOM's least squares.

    .venv/bin/python scripts/hom/controller_rematch_page.py

Data (docs/experiments/20261120-controller_rematch/data/): 20261120-gpu_vs_cpu.json (controller_rematch.py check),
20261120-rematch_summary.json (controller_rematch.py summary over the rows of every arm), 20261120-sensing.json (written
here from rows/hom_pads_sensing.jsonl when it exists), 20261120-clearance_hom.json (sim/20261120-clearance_hom.py).
A section appears once its data exists.
"""
from __future__ import annotations

import base64
import html
import io
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _paths  # noqa: E402,F401

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "docs/experiments/20261120-controller_rematch"
DATA = D / "data"
PAGE = D / "20261120-controllers_tuned_forces_calibrated_plant.html"
TEMPLATE = ROOT / "scripts/ur5/ur5_tip_calibration_page.template.html"
SENSE_ROWS = D / "rows/hom_pads_sensing.jsonl"
TEX_RE = re.compile(r"\\\[(.+?)\\\]|\\\((.+?)\\\)", re.S)
TEX_CACHE = D / "media/texsvg_cache.json"
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#3a9e5f", "#8a5cc2"
F_TRIPOD = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)
W_SPIN = (0.1, 0.25, 0.5, 1.0, 2.0)
REF_RMS_DEG = 59.4


def j(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def esc(s):
    return html.escape(str(s), quote=True)


def png_uri(figure, dpi=110):
    import matplotlib.pyplot as plt
    b = io.BytesIO()
    figure.savefig(b, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def fig(n, uri, caption):
    return (f'<figure id="fig{n}"><img src="{uri}" alt="{esc(caption.split(".")[0])}">'
            f"<figcaption>Figure {n}. {caption}</figcaption></figure>")


def table(head, body, num=(), caption=None, n=None):
    cap = f"<caption>Table {n}. {caption}</caption>" if caption else ""
    th = "".join(f'<th{" class=n" if k in num else ""}>{h}</th>' for k, h in enumerate(head))
    tr = "".join("<tr>" + "".join(f'<td{" class=n" if k in num else ""}>{c}</td>' for k, c in enumerate(r)) + "</tr>"
                 for r in body)
    return f'<div class="tw"><table>{cap}<thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


def f(x, nd=3):
    return "&ndash;" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def xy_s(xy):
    return f"({xy[0]:g}, {xy[1]:g})"


def sensing_data():
    """data/20261120-sensing.json from rows/hom_pads_sensing.jsonl (the GPU runs of 08:2x with the stem scenes): per
    sensing variant and force setting, the mean score and the drops over the 45 layouts at seeds 0-2."""
    out = DATA / "20261120-sensing.json"
    if SENSE_ROWS.exists():
        acc = {}
        for x in SENSE_ROWS.read_text().splitlines():
            if not x.strip():
                continue
            r = json.loads(x)
            key = f"{r['variant']}|{'f1 w0.5' if r.get('prm') else 'default'}"
            a = acc.setdefault(key, {"n": 0, "score": 0.0, "dropped": 0})
            a["n"] += 1
            a["score"] += r["track"]
            a["dropped"] += int(bool(r.get("dropped")))
        rows = {k: {"n": v["n"], "mean": round(v["score"] / v["n"], 4), "dropped": v["dropped"]} for k, v in acc.items()}
        out.write_text(json.dumps({"what": "HOM's sensing variants on the GPU cal75 plant in the stem scenes, 45 layouts, "
                                           "seeds 0-2 (controller_rematch.py hom --variant ... --tag _sensing)",
                                   "rows": rows}, indent=1) + "\n")
    return j(out)


def arm(S, name):
    return (S or {}).get("arms", {}).get(name)


def fig_force_grid(S):
    """HOM's mean score at fresh seeds over the force settings, pads and point contact."""
    import matplotlib.pyplot as plt
    fgs, axs = plt.subplots(1, 2, figsize=(10.5, 3.9), sharey=True)
    for ax, model, lab in zip(axs, ("pads", "pt"), ("Sphere pads", "Point contact")):
        Z = np.full((len(W_SPIN), len(F_TRIPOD)), np.nan)
        for i, w in enumerate(W_SPIN):
            for k, ft in enumerate(F_TRIPOD):
                a = arm(S, f"hom {model} notip0 f{ft:g} w{w:g}")
                if a and a.get("s101112") is not None:
                    Z[i, k] = a["s101112"]
        im = ax.imshow(Z, origin="lower", cmap="viridis", vmin=0.3, vmax=0.7, aspect="auto")
        for i in range(len(W_SPIN)):
            for k in range(len(F_TRIPOD)):
                if np.isfinite(Z[i, k]):
                    ax.text(k, i, f"{Z[i, k]:.2f}", ha="center", va="center", fontsize=10,
                            color="white" if Z[i, k] < 0.5 else "black")
        ax.set_xticks(range(len(F_TRIPOD)), [f"{v:g}" for v in F_TRIPOD], fontsize=11)
        ax.set_yticks(range(len(W_SPIN)), [f"{v:g}" for v in W_SPIN], fontsize=11)
        ax.set_xlabel("Commanded normal force per finger (N)", fontsize=12)
        ax.set_title(lab, fontsize=13)
    axs[0].set_ylabel("Weight of the spin rows", fontsize=12)
    cb = fgs.colorbar(im, ax=axs, shrink=0.9)
    cb.set_label("Mean tracking score, seeds 10-12", fontsize=12)
    return png_uri(fgs)


def fig_gpu_cpu(G):
    import matplotlib.pyplot as plt
    fgs, axs = plt.subplots(1, 2, figsize=(9.5, 4.2), sharey=True)
    for ax, (model, lab) in zip(axs, (("pads", "Sphere pads"), ("pt", "Point contact"))):
        m = G["models"].get(model)
        if not m:
            continue
        x = np.array([p["cpu"] for p in m["pairs"]])
        y = np.array([p["gpu"] for p in m["pairs"]])
        ax.plot([0, 0.7], [0, 0.7], color="#999", lw=1)
        ax.scatter(x, y, s=22, color=C1)
        ax.set_xlim(0, 0.7)
        ax.set_ylim(0, 0.7)
        ax.set_xlabel("CPU plant, mean of seeds 0-2", fontsize=12)
        ax.set_title(f"{lab}: rho {m['rho']:.2f}", fontsize=13)
        ax.tick_params(labelsize=11)
    axs[0].set_ylabel("GPU plant, mean of seeds 0-2", fontsize=12)
    return png_uri(fgs)


def fig_arms(rows):
    """Bars of the arms' fresh-seed scores, with the drops written on them."""
    import matplotlib.pyplot as plt
    fgs, ax = plt.subplots(figsize=(10, 0.42 * len(rows) + 1.2))
    y = np.arange(len(rows))[::-1]
    for yy, (lab, v, nd, n, col) in zip(y, rows):
        ax.barh(yy, v, color=col, height=0.7)
        ax.text(v + 0.005, yy, f"{v:.3f}" + (f"  ({nd}/{n} dropped)" if nd else ""), va="center", fontsize=11)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=11)
    ax.set_xlim(0, 0.85)
    ax.set_xlabel("Mean tracking score over the 45 layouts, seeds 10-12", fontsize=12)
    ax.tick_params(axis="x", labelsize=11)
    return png_uri(fgs)


# ------------------------------------------------------------------ page
def page():
    sys.path.insert(0, str(ROOT / "scripts/docs"))
    import docs_index
    import retro_style
    import texsvg
    G = j(DATA / "20261120-gpu_vs_cpu.json")
    S = j(DATA / "20261120-rematch_summary.json")
    SE = sensing_data()
    CL = j(DATA / "20261120-clearance_hom.json")
    if G is None or S is None:
        sys.exit("no data yet: run controller_rematch.py check and summary")
    gp, gt = G["models"]["pads"], G["models"]["pt"]
    d0 = arm(S, "hom pads notip0 f2.5 w1")
    best_name = max((k for k in S["arms"] if k.startswith("hom pads notip0 f")),
                    key=lambda k: S["arms"][k].get("s012", 0.0))
    b0 = S["arms"][best_name]
    bf = re.match(r"hom pads notip0 f([\d.]+) w([\d.]+)", best_name).groups()
    n_fig, n_tab = [0], [0]

    def nf():
        n_fig[0] += 1
        return n_fig[0]

    def nt():
        n_tab[0] += 1
        return n_tab[0]

    sec = []
    # -------------------------------------------------- terms
    sec.append(("terms", "Terms and metrics", """<dl>
<dt>Tracking score</dt><dd>The RMS angle between the sphere's orientation and the commanded one is taken over the task
phase. It is divided by the RMS of the commanded rotation, 59.4&deg; for this twist, and subtracted from one. The score is
at least 0, and 0 when the sphere is not held at the end. A score of 1 means the commanded twist was followed exactly; 0
means no better than holding the sphere still. The twist ranking and the bench use the same score.</dd>
<dt>Layouts</dt><dd>The 45 layouts of the morning series of 2026-11-20 in whose simulated twist the fingers stayed at
least 2 mm apart on the CAD collision spheres. Each is run at its plan's grasp (data/20261120-pairs.json).</dd>
<dt>Selection and evaluation seeds</dt><dd>A seed moves the sphere's start by up to 1 mm and 2&deg;. A setting is chosen
on seeds 0-2 and scored on seeds 10-12. Its score at seeds 10-12 is therefore free of the selection's optimism.</dd>
<dt>Calibrated plant</dt><dd>The plant of the blind predictions of 2026-11-17, which predicted 0.437 against the hand's
0.447. Its servos run at 7.5 V (0.825 N&nbsp;m/rad, 0.27 N&nbsp;m), the index's and the middle's yaw at 0.525
N&nbsp;m/rad. Its gears cannot be backdriven by the load, and its MCPs stop where the hand's stopped. The sphere sits on the
bench's stem; impratio is 1 and friction 1.0. Forces are converted to servo targets at the server's 0.55 N&nbsp;m/rad. Every
controller's targets pass the server's torque budget of 0.56 N&nbsp;m.</dd>
<dt>Commanded force and spin weight</dt><dd>HOM's normal force per finger (2.5 N by default) and the weight of the spin
rows in its least squares, Eq. (2) (1.0 by default).</dd>
<dt>Tracker models</dt><dd>Exact: the plant's object pose is passed to HOM every 20 ms period. 30 fps and 60 fps: the two
camera modes measured by the lab on 2026-11-13. A frame is 42 or 22 ms old, with 0.013-0.018 mm and 0.05-0.07&deg; of
jitter. A new frame arrives every second period at 30 fps and every period at 60 fps. The variants with calibration error
add an offset of 0.5 mm per axis to the object's frame, drawn once per run.</dd>
<dt>Pose filter</dt><dd>A complementary filter on the measured pose. Each period the estimate is turned by the commanded
angular velocity. Each frame is compared with the estimate kept for the frame's own time, and the estimate moves by
\\(\\Delta t/\\tau\\) of their difference.</dd>
<dt>Target reversals</dt><dd>Sign changes of a servo target's rate above 1&deg;/s, per joint per second.</dd>
<dt>Least CAD gap</dt><dd>The smallest surface gap between two fingers' CAD collision spheres along the simulated twist,
in mm. A twist counts as feasible when this gap is 10 mm or more under both contact models and the sphere is held.</dd>
</dl>"""))
    # -------------------------------------------------- plant
    t = (f"""<p>The calibrated plant existed on the CPU only. Its gears hold the load through a Python step: before every
physics step, each joint's friction loss is set from the contact torques. On the GPU the same rule runs as a kernel, with
one friction-loss row per world (<code>hom_gpu.py</code>, <code>HOM_GPU_PLANT=cal75</code>), in the bench's stem scenes
(<code>HOM_GPU_SCENE=stem</code>). Over the 45 layouts at seeds 0-2, HOM scored {f(gp['gpu_mean'])} on the GPU and
{f(gp['cpu_mean'])} on the CPU with the pads. Spearman's rho between the two was {f(gp['rho'], 2)}, and the mean absolute
difference per layout {f(gp['mean_abs_diff'])}. With point contact the scores were {f(gt['gpu_mean'])} and
{f(gt['cpu_mean'])}, with rho {f(gt['rho'], 2)}. Stage E's 24 force settings were run on both plants for seven layouts; the
settings' mean scores agree at rho 0.99, within 0.011-0.014 on average. In the cell's post scene, in place of the stem, the
sphere was dropped before the lift in 38 of 135 GPU runs that the CPU plant held.</p>""")
    t += fig(nf(), fig_gpu_cpu(G), "HOM's mean score per layout on the GPU plant against the CPU plant, seeds 0-2, 45 "
             "layouts. The line is equality.")
    sec.append(("plant", "The calibrated plant on the GPU", t))
    # -------------------------------------------------- HOM's force
    t = (f"""<p>At the default 2.5 N and spin weight 1.0, HOM scored {f(d0['s101112'])} at seeds 10-12. The setting with the
best mean at seeds 0-2 was {bf[0]} N with spin weight {bf[1]}. At seeds 10-12 it scored {f(b0['s101112'])}, with
{b0.get('s101112_dropped', 0)} drops in {b0.get('s101112_n', 0)} runs.</p>""")
    t += fig(nf(), fig_force_grid(S), "HOM's mean tracking score at seeds 10-12 over the commanded normal force and the "
             "weight of the spin rows, 45 layouts, exact object pose. Blank cells were not run.")
    sec.append(("force", "HOM's commanded force", t))
    # -------------------------------------------------- sensing
    sens = []
    for lab, sfx in (("Exact pose", "_notip0"), ("30 fps", "_cam30_holde2"),
                     ("30 fps, pose filter 0.05 s", "_cam30_pf0.05e2"), ("30 fps, pose filter 0.1 s", "_cam30_pf0.1e2"),
                     ("60 fps", "_cam60"), ("60 fps, pose filter 0.05 s", "_cam60_pf0.05"),
                     ("30 fps with calibration error", "_cam30b_holde2"), ("60 fps with calibration error", "_cam60b")):
        a = arm(S, "hom " + sfx)                       # inner_loop_mpc.py's rows: the prm tag goes before the variant's
        b = arm(S, "hom _prmftripod1wspintripod0.5" + sfx)
        if a or b:
            sens.append([lab, f(a and a.get("s101112")), str(a and a.get("s101112_dropped")) if a else "&ndash;",
                         f(a and a.get("reversals_ps"), 2), f(b and b.get("s101112")),
                         str(b.get("s101112_dropped")) if b else "&ndash;", f(b and b.get("reversals_ps"), 2)])
    if sens:
        t = """<p>The bench's tracker runs at 30 fps, and each frame is 42 ms old when it is passed to HOM. At 60 fps a frame
would be 22 ms old. These are the lab's measurements of 2026-11-13; the switch to 60 fps was left for a test with the block
moving. Table 1 gives HOM's score under each tracker model, at the default force and at 1.0 N with spin weight 0.5.</p>"""
        t += table(["Object pose", "Score, 2.5 N", "Dropped", "Reversals (1/s)", "Score, 1.0 N", "Dropped",
                    "Reversals (1/s)"], sens, num=(1, 2, 3, 4, 5, 6),
                   caption="HOM with each tracker model, 45 layouts, seeds 10-12 (135 runs per cell).", n=nt())
        sec.append(("tracker", "Tracker rate and the pose filter", t))
    if SE:
        R = SE["rows"]

        def g(v, p):
            x = R.get(f"{v}|{p}")
            return (f(x["mean"]), str(x["dropped"])) if x else ("&ndash;", "&ndash;")
        body = []
        for lab, v in (("No fingertips, exact pose", "notip0"), ("Exact fingertip contacts, exact pose", "rate50"),
                       ("No fingertips, 40 ms late, 0.5 mm and 1&deg; noise per frame", "notip"),
                       ("Fingertip model, 40 ms late, the same noise", "tip"), ("Fingertip model, errors x3", "tip3"),
                       ("Fingertip model and HOM's force loop", "tipfl")):
            body.append([lab, *g(v, "default"), *g(v, "f1 w0.5")])
        t = """<p>With the object pose exact, each contact is computed from the geometry, and exact fingertip contacts changed
no score. HOM's force loop on the fingertip forces lowered the score. These runs used an older tracker model: 40 ms late,
with 0.5 mm and 1&deg; of fresh noise per frame. That noise is about 20 times the jitter measured by the lab.</p>"""
        t += table(["HOM's sensing", "Score, 2.5 N", "Dropped", "Score, 1.0 N", "Dropped"], body, num=(1, 2, 3, 4),
                   caption="HOM with and without fingertip sensing, 45 layouts, seeds 0-2 (135 runs per cell).", n=nt())
        sec.append(("tips", "Fingertip sensing", t))
    # -------------------------------------------------- planners and UHAS
    names = {"ps": "Predictive sampling", "anneal": "Annealed sampling", "res_anneal": "HOM plus annealed residual",
             "uhas": "UHAS"}
    plan_rows = []
    for k, v in sorted(S["arms"].items()):
        meth = k.split(" ")[0]
        if meth in names and v.get("s101112") is not None:
            plan_rows.append([esc(k), f(v.get("s012")), f(v.get("s101112")), str(v.get("s101112_dropped", 0)),
                              f(v.get("reversals_ps"), 2), f(v.get("sat_frac"), 3), f(v.get("replan_s_mean"), 3)])
    if plan_rows:
        t = """<p>Each planner ran with HOM's grip at 2.5 N and at 1.0 N before the task, and with a cost on each pad's normal
force against 1.0 N. The arms' names give the method and its settings as the rows' files name them.</p>"""
        t += table(["Arm", "Score, seeds 0-2", "Score, seeds 10-12", "Dropped", "Reversals (1/s)", "Torque-limit share",
                    "Planning time per replanning (s)"], plan_rows, num=(1, 2, 3, 4, 5, 6),
                   caption="The planners and UHAS on the calibrated plant, 45 layouts.", n=nt())
        sec.append(("planners", "Sampling planners and UHAS with their force knobs", t))
    # -------------------------------------------------- clearance
    if CL and CL.get("layouts"):
        L = CL["layouts"]
        near = [x for x in L if x["group"] == "near" and "base" in x and "d20_w10" in x]
        clear = [x for x in L if x["group"] == "clear" and "base" in x and "d20_w10" in x]
        body = []
        for x in sorted(near, key=lambda x: -x["base"]["track"]):
            gf_ = x.get("grasps_force") or {}
            gc_ = x.get("grasps_clear_force") or {}
            body.append([xy_s(x["xy"]) + f" g{x['k']}", f(x["base"]["least_mm"], 1), f(x["base"]["track"]),
                         f(x["d20_w10"]["least_mm"], 1), f(x["d20_w10"]["track"]),
                         f"{gf_.get('feasible', 0)}/{gf_.get('grasps', 0)}, {f(gf_.get('best_track'))}" if gf_ else "&ndash;",
                         f"{gc_.get('feasible', 0)}/{gc_.get('grasps', 0)}, {f(gc_.get('best_track'))}" if gc_ else "&ndash;"])
        nfz = sum(x["d20_w10"]["feasible"] for x in near)
        t = (f"""<p>A clearance term adds rows to each finger's Eq. (2) for its closest link spheres to the other fingers:
\\[ \\sqrt{{w}}\\,\\bigl(\\mathbf n^{{\\top}} J\\,\\dot{{\\boldsymbol\\theta}} - k_c (d_s - g)\\bigr)/\\ell,\\qquad
w = w_c \\Bigl(\\frac{{d_s - g}}{{d_s}}\\Bigr)^2 \\text{{ for }} g < d_s. \\]
Here \\(g\\) is the spheres' surface gap in HOM's kinematic model and \\(\\mathbf n\\) the unit vector between their
centres. \\(J\\) is the Jacobian of the sphere's centre over the finger's joints, \\(k_c\\) = 5/s, and \\(\\ell\\) =
0.02 m is HOM's length scale. A gap above \\(d_s\\) leaves HOM unchanged. The term was tested on the CPU plant at seed 0
under both contact models, with the CAD check of the feasible search. At \\(d_s\\) = 20 mm and \\(w_c\\) = 10, {nfz}
of the {len(near)} near-touching layouts of the morning list became feasible at their plan's grasp. The {len(clear)} clear
layouts kept their gaps. The simulation's best layouts lost most of their score with the term (Table {n_tab[0] + 1}).</p>""")
        nf_ = sum(1 for x in near if (x.get("grasps_force") or {}).get("feasible", 0) > 0)
        t += (f"""<p>The search over every grasp of these layouts shows that feasibility depends on HOM's force. At 1.0 N and
spin weight 0.25, without the term, {nf_} of them have a feasible grasp, and the best of those score as well as the clear
layouts at that force. The simulation's top layouts have none at 1.0 N. The feasible landscape of the twist ranking was
computed at 2.5 N only.</p>""")
        t += table(["Layout and grasp", "Least gap (mm)", "Score", "Least gap with the term (mm)", "Score with the term",
                    "Feasible grasps at 1.0 N, best score", "Feasible grasps at 1.0 N with the term, best score"], body,
                   num=(1, 2, 3, 4, 5, 6),
                   caption="HOM with and without the clearance term on the CPU plant, seed 0, the mean of both contact "
                           "models (the least gap: the smaller of the two). The last two columns: of each layout's "
                           "grasps, those with a feasible twist, and the best one's score.", n=nt())
        sec.append(("clearance", "A clearance term in HOM's least squares", t))
    # -------------------------------------------------- future work
    sec.append(("future", "Future work", """<ul>
<li>Run the tracker at 60 fps on the bench after the lab's test with the block moving
(docs/experiments/20261113-tag_tracking_throughput), and compare the trials' drops at 1.0 N with Table 1.</li>
<li>Score the stage E trials of 2026-11-20 against two predictions, with the exact pose and with the 30 fps tracker. The
arms are named hom _notip0 and hom _cam30_holde2 in controller_rematch.py summary.</li>
<li>Fit the clearance term's gap to the CAD spheres: the least CAD gap against the kinematic model's along the same
twists, so that d_s is set in the CAD's millimetres.</li>
</ul>"""))

    lede = (f"""<p>Of the settings tested on the calibrated plant, HOM's commanded force changed its score most. Lowering it
from 2.5 N to {bf[0]} N, with the spin weight at {bf[1]}, raised HOM's score from {f(d0['s101112'])} to
{f(b0['s101112'])} over 45 layouts at fresh seeds. The plant was ported to the GPU and agrees with the CPU plant,
{f(gp['gpu_mean'])} against {f(gp['cpu_mean'])}. The sampling planners, UHAS, the tracker's rate and a clearance term were
run on the same plant.</p>""")
    summary = (f"On the calibrated SR2 plant, HOM's commanded force changed its tracking most: {bf[0]} N in place of 2.5 N raised it "
               f"from {f(d0['s101112'])} to {f(b0['s101112'])} over 45 layouts at fresh seeds; the GPU port of the plant "
               f"matches the CPU plant ({f(gp['gpu_mean'])} against {f(gp['cpu_mean'])}).")
    toc = "".join(f'<li><a href="#{k}">{t_}</a></li>' for k, t_, _ in sec)
    body = "".join(f'<section id="{k}"><h2>{t_}</h2>{s}</section>' for k, t_, s in sec)
    items = [((m.group(1) if m.group(1) is not None else m.group(2)).strip(), m.group(1) is not None)
             for m in TEX_RE.finditer(body)]
    TEX_CACHE.parent.mkdir(parents=True, exist_ok=True)
    svgs = iter(texsvg.render(items, cache_path=str(TEX_CACHE), scale=1.1))
    body = TEX_RE.sub(lambda m: next(svgs), body)
    title = "Inner-loop controllers with tuned contact forces on the calibrated SR2 plant"
    out = (TEMPLATE.read_text()
           .replace("{{TITLE}}", title)
           .replace("{{TEXCSS}}", texsvg.CSS)
           .replace("{{EYEBROW}}", "HOM on the SR2 hand · 2026-11-20")
           .replace("{{H1}}", title)
           .replace("{{LEDE}}", lede)
           .replace("{{BYLINE}}", "Workstation session hand-4a, 2026-11-20")
           .replace("{{FILM}}", "")
           .replace("{{TOC}}", toc)
           .replace("{{SECTIONS}}", body)
           .replace("{{FOOTER}}", "Built by scripts/hom/controller_rematch_page.py from "
                                  "docs/experiments/20261120-controller_rematch/data/."))
    out = retro_style.apply(out)
    out = docs_index.tag(out, topic="design", summary=summary)
    PAGE.write_text(out)
    print(PAGE, f"{PAGE.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    page()
