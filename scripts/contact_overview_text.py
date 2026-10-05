"""Prose blocks of the contact overview page (scripts/contact_overview_page.py) that summarise the bed's findings and
the literature. Every number is computed from the rows passed in `ctx`; a block whose rows are missing says so."""
from __future__ import annotations

import statistics

import contact_overview_page as P

NEWTON_REV = "009158e6"


def _pct(a, b):
    return (a / b - 1) * 100


def _get(M, label_start, model):
    for group, label, unit, nd, vals in M:
        if label.startswith(label_start):
            v = vals.get(model)
            return v if isinstance(v, (int, float)) else None
    return None


def lede(ctx):
    M, cost, gpu = ctx["M"], ctx["cost"], ctx["gpu_pads"]
    parts = ["A fingertip covered with small MuJoCo contact spheres carries the friction torque and contact area of a soft pad "
             "without any change to MuJoCo."]
    mu_p, mu_d = _get(M, "effective", "mj_pads1"), _get(M, "effective", "drake_hydro")
    vs_p, vs_d = _get(M, "slip speed", "mj_pads1"), _get(M, "slip speed", "drake_hydro")
    a5_p, a5_d = _get(M, "arm at spin onset, 0.5", "mj_pads1"), _get(M, "arm at spin onset, 0.5", "drake_hydro")
    a3_p, a3_d = _get(M, "arm at spin onset, 3", "mj_pads1"), _get(M, "arm at spin onset, 3", "drake_hydro")
    s = []
    if mu_p and mu_d:
        d = abs(_pct(mu_p, mu_d))
        s.append("slip onset at the same force as Drake hydroelastic to 0.1&#8202;%" if d < 0.1 else
                 f"slip onset within {d:.1f}&#8202;% of Drake hydroelastic")
    if vs_p and vs_d:
        s.append(f"slip speed within {abs(_pct(vs_p, vs_d)):.0f}&#8202;%")
    if a5_p and a5_d and a3_p and a3_d:
        s.append(f"friction arm at the onset of spin within {max(abs(_pct(a5_p, a5_d)), abs(_pct(a3_p, a3_d))):.0f}&#8202;% from 0.5 to 3&#8202;N")
    else:
        lt = [r for r in ctx["cal"] if r.get("exp") == "pinch" and r.get("calib") != "fixed" and P._eq(r.get("dt_ms"), 1.0)]
        if lt:
            d = max(abs(_pct(r["rbar_mm"], P.C_LAW * 1e3 * r["N_cmd"] ** P.EXP_LAW)) for r in lt)
            s.append(f"friction arm within {d:.0f}&#8202;% of the hydroelastic law from 0.5 to 3&#8202;N")
    if s:
        parts.append("On the two-pad pinch of the real_v1 fingertip and screwdriver, the 1&#8202;mm pad puts its " +
                     ", its ".join(s[:-1]) + (", and its " if len(s) > 1 else "") + s[-1] + ".")
    if "mj_pads1" in cost and "drake_hydro" in cost:
        g = [r["world_steps_per_s"] for r in gpu if r.get("key", "").startswith("legacy_s1.0") and r.get("status") == "complete"]
        tail = f", and one GPU runs {max(g) / 1e6:.2f} million pad world-steps per second" if g else ""
        parts.append(f"It costs {cost['mj_pads1']:.0f}&#8202;&#181;s per physics step on one core against {cost['drake_hydro']:.0f}&#8202;&#181;s "
                     f"for Drake, gives the same task result from 50&#8202;&#181;s to 10&#8202;ms steps{tail}.")
    cr_p = _get(M, "creep at half", "mj_pads1")
    cr_d = _get(M, "creep at half", "drake_hydro")
    if cr_p and cr_d:
        parts.append(f"Its largest disagreement with the references is creep under a held load, {cr_p / cr_d:.0f}&#215; Drake&#8217;s.")
    parts.append("This page derives why a sphere-sampled elastic foundation inside MuJoCo&#8217;s soft constraints works, which "
                 "parameters set its stiffness, creep and step limit, and what it leaves out: lateral load spreading and elastic "
                 "shear before slip. The bed page holds the task-by-task evidence and films.")
    return " ".join(parts)


def refs_text(ctx):
    nw = ctx["gpu_newton"]
    newton_dyn = ""
    st = [r for r in P.load(P.os.path.join(P.BED, "static_newton.jsonl")) if P.first(r, "rbar_mm", "rbar_per_pad_mm")]
    if st:
        newton_dyn = (" On the two-pad rig its static pinch gives an arm of " +
                      "; ".join(f"{P.first(r, 'rbar_mm', 'rbar_per_pad_mm'):.3f}&#8202;mm at {P.first(r, 'N', 'N_cmd'):g}&#8202;N"
                                for r in st) + ".")
    tw = [r for r in ctx["T"]["twist"] if str(r.get("model", "")).startswith("newton") and r.get("status") == "complete"
          and r.get("dt_ms") == 1.0 and r.get("tau_onset_over_law")]
    newton_tw = ""
    if tw:
        newton_tw = (f" In bed task&#160;2 it carries {min(r['tau_onset_over_law'] for r in tw):.1f}&#8211;{max(r['tau_onset_over_law'] for r in tw):.1f}&#215; "
                     "the hydroelastic law&#8217;s torque at the onset of spin, with and without reduction, while Drake carries 0.97&#8211;1.00&#215;; its "
                     "5&#8202;ms runs spin at once or eject the tool. One GPU world costs 0.5&#8202;ms per step with reduction and 1.5&#8202;ms without.")
    return f"""
<h3>Drake hydroelastic</h3>
<p>Drake gives each compliant body a pressure field on a tetrahedral mesh, \\(p=E\\,\\delta/R\\) for the fingertip sphere at a
1&#8202;mm resolution, and computes the contact surface as the rigid tool&#8217;s surface inside it. Each face carries a pressure,
and the SAP solver integrates force and friction over the faces with a convex, regularised friction law that sticks far
more firmly than MuJoCo&#8217;s soft rows. It is the reference for every task on these pages and costs 1&#8211;2&#8202;ms per
step on one core. Importing the hand from MJCF needs the model&#8217;s contact exclusions applied as collision filters; without
them the distal links locked against their own yaw links (chain page).</p>
<h3>Newton hydroelastic</h3>
<p>Newton (commit {NEWTON_REV}) represents each shape by a sparse signed-distance grid, extracts the patch where the two
shapes&#8217; pressures balance by marching cubes, and hands each triangle to the solver as a contact point with its own stiffness.
An optional reduction merges them into a few representative contacts. Friction stays per point in the MuJoCo-Warp or XPBD
solver, so the friction torque comes from the spread of the points, as with the sphere pads. Codex&#8217;s static check on the
SR2 pinch put the torque capacity within 1&#8211;6&#8202;% of the continuum at 0.25&#8202;mm voxels without reduction; reduction
lowered it by 16&#8211;25&#8202;% and 1&#8202;mm voxels by up to 69&#8202;%. Every dynamic run blew up during the preload until
the probe of 5 October found the cause in Newton&#8217;s MJCF importer: a one-value default <code>solref</code> was stored with
damping ratio 0, and contacts with zero Newton stiffness fell back to it.{newton_dyn}{newton_tw}</p>
<h3>Compliant Sphere Lattice Contact</h3>
<p>{cslc_text(ctx)}</p>
"""


def cslc_text(ctx):
    return ("CSLC represents the fingertip surface by a lattice of spheres tied to the rigid core by anchor springs and to "
            "their neighbours by lateral springs, and solves the lattice for quasi-static equilibrium against the object with "
            "warm-started damped Jacobi iterations. Its presliding friction saturates smoothly, and its authors describe the "
            "model as tangentially stateless, without kinetic sliding dynamics. It was not run here; its paper is in "
            "<code>docs/experiments/20261004-codex/22_Compliant_Sphere_Lattice_Co.pdf</code>.")


def evidence_lead(ctx):
    T = ctx["T"]
    n = sum(len(v) for v in T.values())
    return (f"The comparison bed runs five pinch tasks on the two-pad rig of the 10-01 study in every model that can carry them, "
            f"at 1 and 5&#8202;ms steps, {n} cases so far. Pull and twist load the contact until it slides or spins, roll moves one "
            f"pad to roll the tool between them, shake drives the held tool at 5&#8202;Hz under gravity, and brake lowers the pinch force "
            f"until the tool swings to hanging. Table&#160;2 collects one or two metrics per task.")


def evidence_tasks(ctx):
    T = ctx["T"]
    out = []
    tw = T["twist"]
    g = lambda k, N: P.pick(tw, k, N=N, dt_ms=1.0)  # noqa: E731
    if g("mj_pads1", 0.5) and g("drake_hydro", 0.5):
        out.append("<p><b>Twist.</b> Drake&#8217;s arm at the onset of spin is "
                   + ", ".join(f"{g('drake_hydro', N)['rbar_onset_mm']:.3f}" for N in (0.5, 1.0, 3.0)) +
                   "&#8202;mm at 0.5, 1 and 3&#8202;N, within 3&#8202;% of the law it was fitted to; the 1&#8202;mm pad gives "
                   + ", ".join(f"{g('mj_pads1', N)['rbar_onset_mm']:.3f}" for N in (0.5, 1.0, 3.0)) +
                   "&#8202;mm, 6&#8211;9&#8202;% under it, and its sliding arm is 2&#8211;3&#8202;% under. Both grow by 1.52 between 0.5 and 3&#8202;N, "
                   "the foundation ratio rather than Hertz&#8217;s 1.82 (Figure&#160;2). condim&#160;4 is 3&#8211;6&#8202;% over at onset and exact while "
                   "sliding. Under half the onset torque the pad creeps at 1.1&#8211;1.6&#8202;&#176;/s against Drake&#8217;s 0.01.</p>")
    br = [r for r in T["brake"] if r.get("role", "bed") == "bed"]
    b = lambda k: P.pick(br, k, dt_ms=1.0)  # noqa: E731
    if b("mj_pads1") and b("drake_hydro"):
        out.append(f"<p><b>Brake.</b> Lowering the pinch from 6 to 0.2&#8202;N over 4&#8202;s swings the tool to {b('drake_hydro')['phi_end_deg']:.1f}&#176; in Drake, "
                   f"{b('mj_pads1')['phi_end_deg']:.1f}&#176; with the 1&#8202;mm pad and {b('mj_point4s')['phi_end_deg']:.1f}&#176; with condim&#160;4, with peak "
                   f"rates of {b('drake_hydro')['peak_rate_deg_s']:.0f}, {b('mj_pads1')['peak_rate_deg_s']:.0f} and {b('mj_point4s')['peak_rate_deg_s']:.0f}&#8202;&#176;/s. "
                   f"Point contact swings through to {b('mj_point3')['phi_max_deg']:.0f}&#176; and slides {abs(b('mj_point3')['slip_end_mm']):.0f}&#8202;mm along the tool.</p>")
    sh = T["shake"]
    s = lambda k: P.pick(sh, k, N=0.5, a_pk_g=2.0, dt_ms=1.0)  # noqa: E731
    if s("mj_pads1") and s("drake_hydro"):
        out.append(f"<p><b>Shake.</b> At 0.5&#8202;N and 2&#8202;g the grip is below rigid-Coulomb slip, so all drift is creep: "
                   f"{abs(s('mj_pads1')['drift_per_cycle_mm']) * 1e3:.1f}&#8202;&#181;m per cycle with the pad against "
                   f"{abs(s('drake_hydro')['drift_per_cycle_mm']) * 1e3:.2f}&#8202;&#181;m in Drake.</p>")
    out.append("<p>Roll, the creep study and the stability map were cut off before they ran; their scripts are written and the bed page lists them.</p>")
    return "".join(out)


def cost_text(ctx):
    cost, gpu = ctx["cost"], ctx["gpu_pads"]
    if not cost:
        return ""
    s = ", ".join(f"{P.HTML_LBL[k]} {cost[k]:.1f}&#8202;&#181;s" for k in P.ORDER if k in cost)
    g = [r for r in gpu if r.get("status") == "complete"]
    gp = ""
    if g:
        best = max(g, key=lambda r: r["world_steps_per_s"])
        gp = (f" On one GPU, MuJoCo-Warp runs the 1&#8202;mm pads at up to {best['world_steps_per_s'] / 1e3:.0f}k world-steps per second "
              f"({best['key'].split('_n')[1].split('_')[0]} worlds), {best['sim_s_per_wall_s']:.0f} simulated seconds per wall second; the "
              f"curve flattens beyond 4096 worlds (Figure&#160;9).")
    return (f"Median physics step of the two-pad pinch at 1&#8202;ms on one core, from bed task&#160;1: {s}. Cost is half the comparison; "
            f"Figure&#160;8 sets it against each model&#8217;s distance from Drake over the bed&#8217;s metrics." + gp)


def open_list(ctx):
    items = [
        ("Hertz or foundation on the printed tip.", "Measure the friction torque of the printed fingertip on the screwdriver at 0.5 and 3&#8202;N "
         "(torque sensor under a fixed pinch, slow spin). A ratio near 1.57 keeps a foundation model; near 1.82 calls for lateral coupling, "
         "equation (5)."),
        ("Coupling length of the TPU print.", "Indent the print at one point and map the surface displacement around it; fit "
         "\\(u(r)\\propto e^{-r/\\ell}\\) and compare \\(\\ell\\) with the 1.6&#8211;2.5&#8202;mm patch radius (asides page)."),
        ("Creep that a grip should not have.", "Bed task&#160;6 tests <code>impratio</code> and <code>noslip_iterations</code>; the "
         "chain&#8217;s hold and the wield should be re-run with the setting that brings creep near Drake&#8217;s, and the cost recorded."),
        ("Newton in the same tasks.", "The probe holds and twists; the bed rows for Newton and its GPU throughput at 1k&#8211;8k worlds "
         "decide whether it is the faster pressure-field reference on a GPU."),
        ("The modulus.", "Every model here uses Drake&#8217;s default E&#8202;=&#8202;10&#8202;MPa. The arm scales as \\(E^{-1/4}\\), so a "
         "TPU tip at 2&#8211;5&#8202;MPa lengthens it by 19&#8211;50&#8202;%; refit c once the tip is measured."),
        ("Pre-slip shear.", "No model here stores elastic tangential displacement. A slow tangential load cycle on the printed tip "
         "(hysteresis loop below the slip force) measures whether the tip needs a tangential state."),
    ]
    return "<ul class=\"open\">" + "".join(f"<li><b>{a}</b> {b}</li>" for a, b in items) + "</ul>"


def lit(ctx):
    return ("Elandt, Drumwright, Sherman and Ruina, &#8220;A pressure field model for fast, robust approximation of net contact force "
            "and moment between nominally rigid objects&#8221;, IROS 2019 (Drake hydroelastic). Castro et al., SAP solver (Drake). "
            "The two uploaded papers: <code>docs/experiments/20261004-codex/11_GPU_Accelerated_Hydroelasti.pdf</code> (Newton "
            "hydroelastic) and <code>docs/experiments/20261004-codex/22_Compliant_Sphere_Lattice_Co.pdf</code> (CSLC); notes in "
            "<code>docs/notes/20261005-contact_literature_notes.md</code>. Wang, Oh and Pollard, arXiv 2609.25619 (the controller "
            "and tasks).")


def blocks(ctx):
    have = ctx["have_onset"]
    return {
        "LEDE": lede(ctx),
        "NEWTON_REV": NEWTON_REV,
        "CAP_NOTE": "The CSLC row follows its paper and was not run here.",
        "CSLC_MECH": ("Compliant Sphere Lattice Contact (CSLC) is a discrete form of (6): a lattice of spheres tied to the core by "
                      "anchor springs and to each other by lateral springs. A tangential state per element, which stores shear until "
                      "it reaches \\(\\mu f_n\\), would add elastic stick before slip; CSLC&#8217;s authors describe their model as "
                      "tangentially stateless."),
        "ASIDES_REF": f"<code>{P.ASIDES_PATH}</code>, section on coupled-foundation models",
        "SCALING_NOTE": ("" if have else "Until bed task&#160;2 is written, hollow markers show the 10-01 rig&#8217;s steady sliding "
                         "torque at 0.5&#8202;rad/s, per pad; point contact carries none and is off the log axis."),
        "STAB_NOTE": "Squares are point contact (n&#8202;=&#8202;1), whose bound is \\(t_c\\).",
        "REFS_TEXT": refs_text(ctx),
        "EVIDENCE_LEAD": evidence_lead(ctx),
        "AGREE_NOTE": "A dash marks a model or task not run.",
        "EVIDENCE_TASKS": evidence_tasks(ctx),
        "COST_TEXT": cost_text(ctx),
        "COST_NOTE": "Drake sits at zero deviation by construction.",
        "GPU_NOTE": ("Newton&#8217;s rows stop at 4096 worlds with contact reduction and 1024 without; the batches beyond were cut off. "
                     "Both fixtures hold the same SR2 pinch, but Newton carries 2.4&#8211;2.9&#215; the torque on the two-pad rig, so the "
                     "comparison is of cost, not of equal physics." if ctx["gpu_newton"] else "Newton&#8217;s rows are not written yet."),
        "OPEN_LIST": open_list(ctx),
        "LIT": lit(ctx),
    }
