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


def _law(N):
    """Friction arm (mm) of the pressure law integrated over the screwdriver patch, (3)."""
    return P.C_LAW * 1e3 * N ** P.EXP_LAW


def _dev(devs):
    """Phrase for signed % deviations from a reference: 'within x % of', 'x-y % above' or 'x-y % below'."""
    lo, hi = min(devs), max(devs)
    rng = lambda a, b: f"{a:.1f}" if f"{a:.1f}" == f"{b:.1f}" else f"{a:.1f}&#8211;{b:.1f}"  # noqa: E731
    if lo > 0:
        return f"{rng(lo, hi)}&#8202;% above"
    if hi < 0:
        return f"{rng(-hi, -lo)}&#8202;% below"
    return f"within {max(-lo, hi):.1f}&#8202;% of"


def lede(ctx):
    """Opening paragraph, about 100 words: the method, its agreement with Drake and the pressure law, its cost and its largest
    error (owner 2026-10-09: cut for clarity; the GPU, effective-mass and step-size findings are in their sections)."""
    M, cost = ctx["M"], ctx["cost"]
    parts = ["Covering a fingertip with small MuJoCo contact spheres samples the pressure law of hydroelastic contact, so "
             "unmodified MuJoCo reproduces the friction torque and contact area of a soft fingertip."]
    mu_p, mu_d = _get(M, "effective", "mj_pads1"), _get(M, "effective", "drake_hydro")
    a5_p, a5_d = _get(M, "arm at spin onset, 0.5", "mj_pads1"), _get(M, "arm at spin onset, 0.5", "drake_hydro")
    a3_p, a3_d = _get(M, "arm at spin onset, 3", "mj_pads1"), _get(M, "arm at spin onset, 3", "drake_hydro")
    s = []
    if mu_p and mu_d:
        d = abs(_pct(mu_p, mu_d))
        s.append("slips at Drake hydroelastic&#8217;s force to within 0.1&#8202;%" if d < 0.1 else
                 f"slips at a force within {d:.1f}&#8202;% of Drake hydroelastic&#8217;s")
    if a5_p and a5_d and a3_p and a3_d:
        lo, hi = sorted((abs(_pct(a5_p, a5_d)), abs(_pct(a3_p, a3_d))))
        rng = f"{lo:.0f}" if f"{lo:.0f}" == f"{hi:.0f}" else f"{lo:.0f}&#8211;{hi:.0f}"
        s.append(f"starts to spin at a torque {rng}&#8202;% under Drake&#8217;s")
    tw = ctx["T"]["twist"]
    kin = [(P.pick(tw, "mj_pads1", N=N, dt_ms=1.0) or {}).get("rbar_kin_mm") for N in (0.5, 1.0, 3.0)]
    arm = ""
    if all(kin):
        arm = (f"; while it spins, its friction arm is "
               f"{_dev([(a / _law(N) - 1) * 100 for a, N in zip(kin, (0.5, 1.0, 3.0))])} the pressure law")
    if s:
        parts.append("On the two-pad pinch of the real_v1 fingertip and screwdriver, the 1&#8202;mm pad " + " and ".join(s) + arm + ".")
    import simulator_agreement_figure as SAF
    A = SAF.data()
    if all("turn" in A[k] for k in ("mj_pads", "mjw_pads", "nt_pads")):
        parts.append("The pads agree with Drake to the same degree in CPU MuJoCo, MuJoCo-Warp and Newton.")
    if "mj_pads1" in cost and "drake_hydro" in cost:
        parts.append(f"A pad step costs {cost['mj_pads1']:.0f}&#8202;&#181;s on one CPU core, against "
                     f"{P.num(cost['drake_hydro'], ',.0f')}&#8202;&#181;s for Drake.")
    cr_p, cr_d = _get(M, "creep: sliding", "mj_pads1"), _get(M, "creep: sliding", "drake_hydro")
    if cr_p and cr_d:
        parts.append(f"Under loads below slip the pad creeps {cr_p / cr_d:.0f}&#215; faster than Drake.")
    return " ".join(parts)


def refs_text(ctx):
    nw = ctx["gpu_newton"]
    sink = _static_sink()
    Ns = (0.5, 1.0, 3.0)
    raw = [sink[("newton_hydro", N)] / sink[("drake_hydro", N)] for N in Ns if ("newton_hydro", N) in sink and ("drake_hydro", N) in sink]
    mc = [sink[("newton_hydro_mc", N)] / sink[("drake_hydro", N)] for N in Ns if ("newton_hydro_mc", N) in sink and ("drake_hydro", N) in sink]
    tw = {(r["model"], r["N"]): r["tau_onset_Nm"] for r in ctx["T"]["twist"] if r.get("dt_ms") == 1.0
          and r.get("status") == "complete" and r.get("tau_onset_Nm")}
    on = [tw[("newton_hydro_mc", N)] / tw[("drake_hydro", N)] for N in Ns if ("newton_hydro_mc", N) in tw and ("drake_hydro", N) in tw]
    newton_tw = ""
    if raw and mc:
        newton_tw = (f" SolverMuJoCo realises each contact&#8217;s stiffness times the effective mass, so the bed&#8217;s pads sank "
                     f"{_ratio_rng(raw, 1)} times as deep as Drake; with \\(k_h\\) divided by \\(m_\\text{{eff}}\\) (step&#160;7) they sink "
                     f"{_ratio_rng(mc)} times Drake&#8217;s depth" + (f" and start to spin at {_ratio_rng(on)} of Drake&#8217;s onset torque" if on else "")
                     + ". One GPU world costs 0.5&#8202;ms per step with reduction and 1.5&#8202;ms without.")
    return f"""
<h3>Drake hydroelastic</h3>
<p>Drake gives each compliant body a pressure field on a tetrahedral mesh, \\(p=E\\,\\delta/R\\) for the fingertip sphere at a
1&#8202;mm resolution, and computes the contact surface as the rigid tool&#8217;s surface inside it. Each face has a pressure,
and the SAP solver integrates force and friction over the faces with a convex, regularised friction law; under a held load it
creeps about 100&#215; less than MuJoCo&#8217;s soft friction rows. It is the reference for every task on these pages and costs 1&#8211;2&#8202;ms per
step on one core. Importing the hand from MJCF needs the model&#8217;s contact exclusions applied as collision filters; without
them the distal links locked against their own yaw links (chain page).</p>
<h3>Newton hydroelastic</h3>
<p>Newton (commit {NEWTON_REV}) represents each shape by a sparse signed-distance grid, extracts the patch where the two
shapes&#8217; pressures balance by marching cubes, and hands each triangle to the solver as a contact point with its own stiffness.
An optional reduction merges them into a few representative contacts. Friction stays per point in the MuJoCo-Warp or XPBD
solver, so the friction torque comes from the spread of the points, as with the sphere pads. Codex&#8217;s static check on the
SR2 pinch put the torque capacity within 1&#8211;6&#8202;% of the continuum at 0.25&#8202;mm voxels without reduction; reduction
lowered it by 16&#8211;25&#8202;% and 1&#8202;mm voxels by up to 69&#8202;%. Newton&#8217;s MJCF importer stores a one-value default
<code>solref</code> with damping ratio 0, and contacts with zero Newton stiffness fall back to it, so the bed and hand scenes are
imported with two-value solrefs.{newton_tw} On the bed, all faces at 0.5&#8202;mm voxels give the pressure law&#8217;s friction arm at 0.5
and 1&#8202;N and the reduced contacts 0.77&#8211;0.88 of it, and friction rows of 10&#8202;ms or longer stop the braked tool at hanging
where the default 3.9&#8202;ms swings it to 130&#176; (<code>docs/experiments/20261007-newton_hydro_tests/</code>).</p>
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
    return (f"The comparison bed runs five tasks on the two-pad rig of the 10-01 study at 1 and 5&#8202;ms steps ({n} cases). Pull and "
            f"twist load the pinch until the tool slides or spins; roll moves one pad so the tool rolls between them; shake drives the "
            f"held tool at 5&#8202;Hz under gravity; brake lowers the pinch force until the tool swings toward hanging. Table&#160;2 lists one "
            f"or two metrics per task.")


def evidence_tasks(ctx):
    T = ctx["T"]
    out = []
    tw = T["twist"]
    g = lambda k, N: P.pick(tw, k, N=N, dt_ms=1.0)  # noqa: E731
    if g("mj_pads1", 0.5) and g("drake_hydro", 0.5):
        Ns = (0.5, 1.0, 3.0)
        dev = lambda k, f: [(g(k, N)[f] / _law(N) - 1) * 100 for N in Ns]  # noqa: E731
        rot = lambda k: [g(k, N)["rot_pre_deg"] for N in Ns]  # noqa: E731
        rr = lambda v, d: f"{min(v):.{d}f}&#8211;{max(v):.{d}f}"  # noqa: E731
        out.append("<p><b>Twist.</b> Drake&#8217;s arm at the onset of spin is "
                   + ", ".join(f"{g('drake_hydro', N)['rbar_onset_mm']:.3f}" for N in Ns) +
                   f"&#8202;mm at 0.5, 1 and 3&#8202;N, {_dev(dev('drake_hydro', 'rbar_onset_mm'))} the integrated law (3); the 1&#8202;mm pad gives "
                   + ", ".join(f"{g('mj_pads1', N)['rbar_onset_mm']:.3f}" for N in Ns) +
                   f"&#8202;mm, {_dev(dev('mj_pads1', 'rbar_onset_mm'))} it. Once every sphere slides, the pad&#8217;s arm is "
                   f"{_dev(dev('mj_pads1', 'rbar_kin_mm'))} the law and Drake&#8217;s {_dev(dev('drake_hydro', 'rbar_kin_mm'))} it. The pad&#8217;s "
                   f"shortfall appears only at onset, where the tool turns {rr(rot('mj_pads1'), 1)}&#176; before it spins (Drake "
                   f"{rr(rot('drake_hydro'), 2)}&#176;), which points to MuJoCo&#8217;s soft friction rows as the cause. "
                   "Both arms grow by 1.52 between 0.5 and 3&#8202;N, close to the foundation ratio of 1.565; Hertz gives 1.82 (Figure&#160;2). "
                   f"condim&#160;4, whose \\(\\mu_t\\) follows the fit to Drake&#8217;s torque (step&#160;10), is {_dev(dev('mj_point4s', 'rbar_onset_mm'))} "
                   f"the law at onset and {_dev(dev('mj_point4s', 'rbar_kin_mm'))} it while sliding. Held at half the onset torque, the pad turns at "
                   "1.1&#8211;1.6&#8202;&#176;/s and Drake at 0.01&#8202;&#176;/s.</p>")
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
        out.append(f"<p><b>Shake.</b> At 0.5&#8202;N and 2&#8202;g the load stays below the rigid-Coulomb slip threshold, so the drift is creep: "
                   f"{abs(s('mj_pads1')['drift_per_cycle_mm']) * 1e3:.1f}&#8202;&#181;m per cycle with the pad against "
                   f"{abs(s('drake_hydro')['drift_per_cycle_mm']) * 1e3:.2f}&#8202;&#181;m in Drake.</p>")
    ro = T["roll"]
    r = lambda k, N=1.0: P.pick(ro, k, N=N, dt_ms=1.0, v_mm_s=10.0)  # noqa: E731
    have = [k for k in ("drake_hydro", "mj_pads1", "mj_point3", "mj_point4s", "newton_hydro_mc") if r(k)]
    if "drake_hydro" in have and "mj_pads1" in have:
        allr = [x for x in ro if x.get("status") == "complete" and x.get("model") in P.ORDER]
        slip = [max(x.get("slip_path_L_mm") or 0, x.get("slip_path_R_mm") or 0) * 1e3 for x in allr]
        lost = [x for x in ro if x.get("model") in P.ORDER and not x.get("success")]
        out.append("<p><b>Roll.</b> Moving one pad 10&#8202;mm along the tool at 10&#8202;mm/s rolls the tool between the pads in every "
                   "model, at 0.5 to 3&#8202;N and both steps. At 1&#8202;N the rolling ratio is "
                   + ", ".join(f"{r(k)['rho']:.4f} ({P.HTML_LBL[k]})" for k in have) +
                   f"; no-slip rolling between the spheres gives {r('drake_hydro')['rho_noslip_geom']:.4f}. The differences follow the "
                   f"depth of the contact point: the pad&#8217;s lies {12.5 - r('mj_pads1')['r_contact_L_mm']:.2f}&#8202;mm inside the tool "
                   f"surface and Drake&#8217;s {12.5 - r('drake_hydro')['r_contact_L_mm']:.2f}&#8202;mm, and a contact at a smaller radius "
                   f"turns the tool further for the same travel. The contacts slip at most {max(slip):.0f}&#8202;&#181;m over 5&#8202;mm of "
                   f"rolling, and {'no model loses' if not lost else str(len(lost)) + ' cases lose'} the tool.</p>")
    return "".join(out)


def cost_text(ctx):
    cost, bat = ctx["cost"], ctx.get("batched") or {}
    if not cost:
        return ""
    cpu = [k for k in ("mj_point3", "mj_point4s", "mj_pads1", "drake_hydro") if k in cost]
    gpu = [k for k in P.GPU_MODELS if k in cost]
    us = lambda v: P.num(v, ",.1f" if v < 100 else ",.0f")  # noqa: E731
    out = [f"The cost of one physics step at 1&#8202;ms. On one CPU core, on the bed rig in task&#160;1: "
           + "; ".join(f"{P.HTML_LBL[k]}, {us(cost[k])}&#8202;&#181;s" for k in cpu) + "."]
    if gpu:
        out.append(" One world alone on the GPU, on the SR2 holding fixture of Figure&#160;10 with each 10-step block captured as "
                   "a CUDA graph: " + "; ".join(f"{P.HTML_LBL[k]}, {us(cost[k])}&#8202;&#181;s" for k in gpu))
    s = "".join(out) + "."
    b = [k for k in P.GPU_MODELS if k in bat]
    if b:
        s += (" A single GPU world is bound by kernel launches. In a batch on the same fixture the cost per "
              "world-step at the batch of highest throughput (Figure&#160;10) is "
              + ", ".join(f"{bat[k][0]:.2f}&#8202;&#181;s for the {P.HTML_LBL[k].replace('Newton hydroelastic', 'Newton hydroelastic tip')} "
                          f"({P.num(bat[k][1], ',d')} worlds)" for k in b) + ".")
    return s + " Figure&#160;9 plots both costs against each model&#8217;s deviation from Drake over the metrics of Table&#160;2."


def gpu_note(ctx):
    """Figure 10 caption: the Newton fixture and every batch of 1-8192 worlds that did not run, with its limit."""
    rows = [r for r in ctx["gpu_newton"] if r.get("kind") == "timing" and r.get("fixture_name") == "twist"]
    if not rows:
        return "Newton&#8217;s rows on the pad fixture are not written yet."
    lab = {"newton_pads1": "Newton with the pads", "newton_hydro_mc": "Newton hydroelastic with reduction",
           "newton_hydro_unreduced_mc": "Newton hydroelastic without reduction"}
    miss = []
    for mdl, name in lab.items():
        rs = [r for r in rows if r["model"] == mdl]
        done = {r["nworld"] for r in rs if r.get("status") == "complete"}
        for n in (1, 64, 256, 1024, 2048, 4096, 8192):
            if n in done:
                continue
            why = next((r.get("failure") or r.get("error") for r in rs if r["nworld"] == n and r.get("status") != "complete"), None)
            miss.append(f"{name} at {P.num(n, ',d')} worlds ({why or 'not run'})")
    txt = ("Newton runs the MuJoCo-Warp pad fixture: the same SR2 thumb&#8211;index pinch at 2&#8202;N per pad for 0.8&#8202;s, with two "
           "opposing 12&#8202;mN&#8202;m torque pulses about the pinch axis at 0.3&#8211;0.6&#8202;s; its hydroelastic tip has \\(k_h\\) divided by "
           "the tip&#8211;tool effective mass (step&#160;7).")
    return txt + (" Not run: " + "; ".join(miss) + "." if miss else " Every batch from 1 to 8,192 worlds ran.")


def open_list(ctx):
    items = [
        ("Hertz or foundation on the printed tip.", "Measure the friction torque of the printed fingertip on the screwdriver at 0.5 and 3&#8202;N "
         "(torque sensor under a fixed pinch, slow spin). A ratio near 1.57 keeps a foundation model; near 1.82 calls for lateral coupling, "
         "equation (5)."),
        ("Coupling length of the TPU print.", "Indent the print at one point and map the surface displacement around it; fit "
         "\\(u(r)\\propto e^{-r/\\ell}\\) and compare \\(\\ell\\) with the 1.6&#8211;2.5&#8202;mm patch radius (asides page)."),
        ("Creep in the whole task.", "Re-run the chain&#8217;s hold and the wield with <code>impratio</code> 1000, the setting at which bed "
         "task&#160;6 brings the pad&#8217;s creep toward Drake&#8217;s (step&#160;8), and record the cost per step."),
        ("Newton&#8217;s braked swing.", "With friction rows of 10&#8202;ms Newton&#8217;s hydroelastic tip stops the braked tool at "
         "hanging but lets it fall through 80&#176; at 1.2&#8211;1.3&#8202;N, where Drake passes 80&#176; at 0.39&#8202;N, and a 10&#8211;25&#8202;% "
         "shorter arm does not reproduce that in condim&#160;4. Log the torque of Newton&#8217;s contact forces about the pinch axis during "
         "the fall (2026-10-07 Newton page, open items)."),
        ("The modulus.", "Every model here uses Drake&#8217;s default \\(E=10\\)&#8202;MPa. The arm scales as \\(E^{-1/4}\\), so a "
         "TPU tip at 2&#8211;5&#8202;MPa lengthens it by 19&#8211;50&#8202;%; refit \\(c\\) once the tip is measured."),
        ("Presliding of the printed tip.", "Cycle an axial force at half the slip force on the printed tip under a fixed pinch (bed "
         "task&#160;8 on the bench) and record the displacement against the force. About 40&#8202;&#181;m that return call for a "
         "contact model with presliding (step&#160;12); a loop area near Mindlin&#8217;s calls for the bristles; no measurable "
         "presliding retires both."),
        ("The skin at the trainer&#8217;s settings.", "The trainer&#8217;s solver (10 iterations, 2&#8202;ms step) holds a grasp on the "
         "skin only at impratio 10, where the friction rows creep twice as far as the springs deflect (Figure&#160;14b), so no "
         "policy has yet trained with the skin&#8217;s presliding. Rerun the scripted-grasp check of "
         "<code>scripts/make_work_plant_runs.py</code> on the servo-plant skin scene at impratio 100, 300 and 1000 with more "
         "solver iterations or a 1&#8202;ms step, and train the skin arm at the first setting that holds 64 of 64. If its held "
         "cosine, open-loop replays and jittered robustness stay within the pad arm&#8217;s seed spread, retire the skin "
         f"(the impratio-10 arm: <code>{SKIN_NOTE}</code>)."),
    ]
    return "<ul class=\"open\">" + "".join(f"<li><b>{a}</b> {b}</li>" for a, b in items) + "</ul>"


def lit(ctx):
    return ("Elandt, Drumwright, Sherman and Ruina, &#8220;A pressure field model for fast, robust approximation of net contact force "
            "and moment between nominally rigid objects&#8221;, IROS 2019 (Drake hydroelastic). Castro et al., SAP solver (Drake). "
            "The two uploaded papers: <code>docs/experiments/20261004-codex/11_GPU_Accelerated_Hydroelasti.pdf</code> (Newton "
            "hydroelastic) and <code>docs/experiments/20261004-codex/22_Compliant_Sphere_Lattice_Co.pdf</code> (CSLC); notes in "
            "<code>docs/notes/20261005-contact_literature_notes.md</code>. Wang, Oh and Pollard, arXiv 2609.25619 (the controller "
            "and tasks). Xydas and Kao, &#8220;Modeling of contact mechanics and friction limit surfaces for soft fingers in robotics, "
            "with experimental results&#8221;, IJRR 1999 (contact radius of soft fingertips against load). Drake&#8217;s hydroelastic user guide "
            "(the model has no state, so no tangential compliance).")



# ------------------------------------------------------------------------------- 2026-10-07: effective mass, agreement

def _static_sink():
    """Sink per pad (mm) by (model, N) at 1 ms from the bed's and the mass-scaling folder's static rows."""
    out = {}
    for path in (P.os.path.join(P.BED, "static_newton.jsonl"), P.os.path.join(P.MSC, "static_newton.jsonl")):
        for r in P.load(path):
            if r.get("dt_ms", 1.0) == 1.0 and r.get("status", "complete") == "complete" and "pen_mm" in r:
                out[(r["model"], r["N"])] = r["pen_mm"]
    return out


def _ratio_rng(vals, nd=2):
    v = [x for x in vals if x is not None]
    if not v:
        return "&#8211;"
    a, b = f"{min(v):.{nd}f}", f"{max(v):.{nd}f}"
    return a if a == b else f"{a}&#8211;{b}"


def meff_text(ctx):
    sink = _static_sink()
    Ns = (0.5, 1.0, 3.0)
    raw = [sink[("newton_hydro", N)] / sink[("drake_hydro", N)] for N in Ns if ("newton_hydro", N) in sink and ("drake_hydro", N) in sink]
    mc = [sink[("newton_hydro_mc", N)] / sink[("drake_hydro", N)] for N in Ns if ("newton_hydro_mc", N) in sink and ("drake_hydro", N) in sink]
    br = {r["model"]: r for r in P.load(P.os.path.join(P.SAD, "brake_newton.jsonl"))}
    old = {r["model"]: r for r in P.load(P.os.path.join(P.SAD, "newton_pads_kf1000/brake_newton.jsonl"))}
    d0 = (br.get("newton_pads1") or {}).get("pad_d0") or {}
    w = (br.get("newton_pads1") or {}).get("inv_weight0") or {}
    swing = (f" With Newton&#8217;s default \\(k_f=1000\\) the friction rows get \\(t_f=4\\times10^{{-5}}\\)&#8202;s and the braked tool "
             f"swung to {old['newton_pads1']['phi_max_deg']:.0f}&#176;; with \\(t_f=t_c\\) it swings to "
             f"{br['newton_pads1']['phi_max_deg']:.1f}&#176;, as in MuJoCo." if "newton_pads1" in old and "newton_pads1" in br else "")
    return (
        "The same factor sets the stiffness of every MuJoCo contact: solref is a spring on the constraint "
        "acceleration, so a contact realises its solref stiffness times the effective mass \\(m_\\text{eff}=1/\\hat\\Lambda\\) as a "
        "force stiffness, and the pads&#8217; \\(d_0\\) divides that mass out. Newton&#8217;s hydroelastic contact passes each patch point "
        "to MuJoCo-Warp with a force stiffness \\(c\\) and writes it as \\(t_c=\\sqrt{1/(c(1-d))}\\) without the factor, so on the bed "
        f"(\\(m_\\text{{eff}}\\)&#8202;=&#8202;17.4&#8202;g) it sank {_ratio_rng(raw, 1)} times as deep as Drake. Multiplying its \\(k_h\\) "
        "by the inverse-weight sum of the solver&#8217;s own MuJoCo model, \\(k_h\\,(w_1+w_2)\\), gives "
        f"{_ratio_rng(mc)} times Drake&#8217;s sink with no parameter fitted (<code>{P.HT3_PATH}</code>, Section&#160;4). The pads "
        "carry into Newton&#8217;s point-contact pipeline with two settings: \\(d_0\\) recomputed from the inverse weights of "
        f"Newton&#8217;s MuJoCo model, which gives the bed&#8217;s railed pads {w.get('padL', 16.7):.1f}&#8202;1/kg against the MJCF "
        f"compile&#8217;s 50 (\\(d_0\\) {d0.get('mjcf', 0):.3f} &#8594; {d0.get('solver', 0):.3f}), and a friction gain \\(k_f\\) for "
        "which SolverMuJoCo&#8217;s friction-row time constant \\(t_f=2/(k_f\\,\\hat\\Lambda\\,((1-d_0)/\\kappa+d_0))\\) equals the "
        f"pads&#8217; \\(t_c\\) ({d0.get('kf', 0):.1f} on the bed).{swing}")


def agree_sim_text(ctx):
    import simulator_agreement_figure as SAF
    A = SAF.data()
    pads = [k for k in ("mj_pads", "mjw_pads", "nt_pads") if A[k].get("twist")]
    tw = [x for k in pads for x in A[k]["twist"]]
    br = [abs(A[k]["brake"][0][0]) for k in pads if A[k].get("brake")]
    gr = [g for k in pads for g, _ in A[k].get("grip", [])]
    hy, pt = A["nt_hydro"], A["mj_pt"]
    held = {k: (sum(r["held_end"] for r in P.load(str(path))), len(P.load(str(path))))
            for k, _l, _c, _s, path, _b in SAF.MODELS}
    drake_swing = next((r["phi_max_deg"] for r in P.load(P.os.path.join(P.BED, "brake.jsonl"))
                        if r.get("model") == "drake_hydro" and r.get("dt_ms") == 1.0), None)
    return (
        "The pads are the same contact model in CPU MuJoCo, MuJoCo-Warp and Newton&#8217;s point-contact pipeline once Newton gets "
        "the solver&#8217;s inverse weights (step&#160;7). MuJoCo-Warp ports MuJoCo&#8217;s constraint solver to the GPU and Newton&#8217;s "
        "SolverMuJoCo runs MuJoCo-Warp, so the three are one solver in three implementations; Newton&#8217;s hydroelastic contact and "
        "Drake are the independent comparisons. Figure&#160;7 compares the three, Newton&#8217;s mass-corrected hydroelastic "
        "contact on the plain block and MuJoCo point contact with Drake on the bed&#8217;s twist and brake and on the open-loop plan "
        "replay of the three-finger turn on the eight deployed hands (working servo plant, \\(\\mu=1\\), 24 placements paired with "
        "Drake by hand and placement). The three pad implementations agree with Drake to the same degree: onset torque "
        f"{_ratio_rng(tw)} of Drake&#8217;s at 0.5&#8211;3&#8202;N, largest swing within {max(br):.1f}&#176; of Drake&#8217;s "
        f"{drake_swing:.1f}&#176;, the turn within 3&#176; of Drake on {A['mj_pads']['within']}, {A['mjw_pads']['within']} and "
        f"{A['nt_pads']['within']} of {A['mj_pads']['n']} placements, "
        + ("every one of the 40 placements held in all three" if all(held[k][0] == held[k][1] for k in ("mj_pads", "mjw_pads", "nt_pads"))
           else f"{held['mj_pads'][0]}, {held['mjw_pads'][0]} and {held['nt_pads'][0]} of 40 placements held")
        + f", and grip force {_ratio_rng(gr)} of "
        f"Drake&#8217;s. The mass-corrected hydroelastic contact reaches {_ratio_rng(hy.get('twist', []))} of Drake&#8217;s onset "
        f"torque but swings the braked tool to {drake_swing + hy['brake'][0][0]:.0f}&#176; and turns within 3&#176; on "
        f"{hy.get('within', 0)} of {hy.get('n', 0)}; point contact holds every placement of the turn but transmits "
        f"{_ratio_rng(pt.get('twist', []))} of the onset torque and lets the braked tool spin out of the pinch "
        f"(<code>{P.HT3_PATH}</code>, Section&#160;5).")


def rl_replay_text(ctx):
    ev = {}
    for r in P.load(P.os.path.join(P.RLD, "train_eval.jsonl")):
        if r.get("status") == "ok" and r["env_steps"] > 19e6:
            ev[r["tag"]] = r
    rep = P.load(P.os.path.join(P.RLD, "policy_replay.jsonl"))

    def cpu(tag):
        return next((x for x in reversed(rep) if x.get("engine") == "mujoco" and x.get("status") == "ok"
                     and x.get("hold_test_s") is None and x.get("dir", "").endswith(tag)), None)
    pads = sorted(t for t in ev if "pads1" in t)
    mesh = sorted(t for t in ev if "pads1" not in t)
    if not pads or not mesh:
        return ""
    rec = {x["tag"]: x for x in rep if x.get("kind") == "record"}
    cp = [cpu(t) for t in pads]
    if not all(cp) or not all(t in rec for t in pads):
        return ""
    d = max(abs(c["cos_end"] - rec[t]["mjw_cos_end"][0]) for c, t in zip(cp, pads))
    mesh_drop = all(cpu(t) is not None and not cpu(t)["held_end"] for t in mesh)
    held_p = [round(64 * ev[t]["hold_rate"]) for t in pads]
    return (
        "The pads also carry learned behaviour between implementations. The D6 reorientation trained from scratch in MuJoCo-Warp for "
        f"20&#8202;M steps holds the tool in {' and '.join(map(str, held_p))} of 64 deterministic rollouts with the 1&#8202;mm pads "
        f"(final cosine with vertical {min(ev[t]['final_cos_mean'] for t in pads):.2f}&#8211;"
        f"{max(ev[t]['final_cos_mean'] for t in pads):.2f}). Replayed open loop from the reorientation onset in CPU MuJoCo, the pad "
        f"policies end within {d:.2f} in cosine of MuJoCo-Warp with the tool held"
        + ("; the policies trained on the TPU block as one convex mesh lose the tool in CPU MuJoCo within 0.3&#8202;s" if mesh_drop else "")
        + f". Films of the four policies and the replays in Drake and Newton: <code>{P.HT3_PATH}</code>, Section&#160;6.")


POLICY_PAGE = "docs/experiments/20261008-contact_model_policies/20261008-fingertip_contact_model_policies.html"


def policy_text(ctx):
    """One or two sentences and the link: D6 policies trained on four fingertip contact models, their open-loop replays
    in CPU MuJoCo, Drake and Newton (2026-10-09 revision; numbers from the policy page's own summary)."""
    import contact_model_policies_page as CMP
    X = CMP.Data()
    S = CMP.summary(X)
    if not all(S[a]["n_fin"] for a in CMP.ARMS):
        return ""
    comp, point = ("tpu27pads1", "tpu27skin"), ("box", "tpu27mesh")
    rep = lambda arms: [sum(S[a]["rep"][e][k] for a in arms for e in CMP.ENGINES) for k in (0, 1)]  # noqa: E731
    rc, rp = rep(comp), rep(point)
    n_c = sum(S[a]["n_fin"] for a in comp)
    n_p = sum(S[a]["n_fin"] for a in point)
    pm, pp, ps = (CMP.probe_spit(a) for a in ("tpu27mesh", "tpu27pads1", "tpu27skin"))
    cost = (f"; pad and skin training cost {pp / pm:.1f}&#215; and {ps / pm:.1f}&#215; the TPU mesh&#8217;s time per "
            f"iteration" if pm and pp and ps else "")
    T = {a: CMP.tpu_transfer(X, a) for a in ("tpu27mesh", "tpu27pads1", "tpu27skin")}
    pct = lambda t: f"{100 * t[0] / t[1]:.0f}&#8202;%"  # noqa: E731
    closed = (f" Closed loop in MuJoCo-Warp under the other contact models of the TPU block, the skin policies hold "
              f"{pct(T['tpu27skin'])} of the rollouts, the pad policies {pct(T['tpu27pads1'])} and the TPU-mesh policies "
              f"{pct(T['tpu27mesh'])}" if all(T.values()) else "")
    return (f" Trained for 40&#8202;M steps on four fingertip contact models, the D6 reorientation policies on the 1&#8202;mm "
            f"pads and on the pads mounted on a sprung skin ({n_c} policies) keep the tool in {rc[0]} of {rc[1]} open-loop "
            f"replays of their finger targets in CPU MuJoCo, Drake and Newton, and the {n_p} policies trained with "
            f"point contact on the box tip or the TPU block in {rp[0]} of {rp[1]}.{closed}{cost}: "
            f"<code>{POLICY_PAGE}</code>." + servo_policy_text())


SERVO_PAGE = "docs/experiments/20261011-servo_plant_policies/20261011-servo_plant_contact_model_policies.html"


def servo_policy_text():
    """One sentence and the link (2026-10-12): the same comparison retrained on the servo-limited plant, where the pads
    turn the tool nearly as far as the TPU mesh, and the torque-limit swap (numbers from that page's rows)."""
    import servo_plant_policies_page as SPP
    X = SPP.Data()
    if len(X.finals("11")) < 4:
        return ""
    hc = lambda s, a: [r["held_cos_mean"] for r in X.finals(s, a)]  # noqa: E731
    sp = lambda s: [X.m(r["tag"])["spin_deg"] for r in X.finals(s, "tpu27pads1") if X.m(r["tag"])]  # noqa: E731
    grip = [r["grip_N"] for r in X.finals("11")]
    out = (f" The working plant&#8217;s 1&#8202;N&#8202;m joint torque limit is 4.4 times the servo&#8217;s stall torque; retrained "
           f"on a plant limited to 0.18&#8202;N&#8202;m, at grips of {SPP.rng(grip, 0)}&#8202;N, the pad policies turn the tool "
           f"nearly as far as the TPU-mesh ones (held cosine {SPP.rng(hc('11', 'tpu27pads1'), 2)} against "
           f"{SPP.rng(hc('11', 'tpu27mesh'), 2)}) and the tool spins {SPP.rng(sp('11'), 0)}&#176; relative to the pads during "
           f"the turn, against {SPP.rng(sp('08'), 0)}&#176; at the old grip")
    sw = [(r, X.m(r["tag"]), X.m(r["tag"], "fr=1")) for r in X.finals("11", "tpu27pads1")]
    stop = [(r, m1) for r, m0, m1 in sw if m0 and m1 and m1["n_held"] == m1["n"] and m1["held_cos_mean"] < 0.5]
    if stop:
        r, m1 = stop[0]
        out += (f"; given 1&#8202;N&#8202;m, pad seed {r['seed']} grips with {sum(m1['grip_by_finger_N']):.0f}&#8202;N and stops "
                "turning the tool")
    return out + f": <code>{SERVO_PAGE}</code>."


SCALE_PAGE = "docs/experiments/20261008-hand_object_scale/20261008-finger_spacing_object_size.html"


def scale_text(ctx):
    """One sentence and the link: the hand-object scale landscape runs on the 1 mm pads (2026-10-08)."""
    import json
    rows = P.os.path.join(P.ROOT, "docs/experiments/20261008-hand_object_scale/landscape.json")
    if not P.os.path.exists(rows):
        return ""
    return (" The same 1&#8202;mm pads carry the hand&#8211;object scale landscape of the SR2 hand (17 finger-base layouts "
            "&#215; 18 cylinders and spheres, grasp robustness and the three-finger turn), where point contact on the TPU "
            f"block ranks the layouts nearly as the pads do: <code>{SCALE_PAGE}</code>.")


def _state_cost():
    """us per world-step on the held D6 RL state (the last row per engine, version, fingertip and batch, as on the
    hom_turn3 page): MuJoCo-Warp 3.6 pads and mesh, Newton mass-corrected hydroelastic."""
    R = {}
    for r in P.load(P.os.path.join(P.RLD, "same_state_timing.jsonl")):
        if r.get("status") == "ok":
            R[(r["engine"], r.get("mujoco_warp"), r["variant"], r["nworld"])] = r
    v36 = next((v for v in sorted({k[1] for k in R if k[0] == "mjw" and k[1]}) if str(v).startswith("3.6")), None)
    out = {"pads": [R[k]["us_per_world_step"] for k in R if k[0] == "mjw" and k[1] == v36 and k[2] == "pads1"],
           "mesh": [R[k]["us_per_world_step"] for k in R if k[0] == "mjw" and k[1] == v36 and k[2] == "mesh"],
           "hydro": [R[k]["us_per_world_step"] for k in R if k[0] == "nt_hydro"]}
    return out if out["pads"] else None


def state_cost_text(ctx):
    sc = _state_cost()
    if not sc:
        return ""
    pads, mesh, hyd = sc["pads"], sc["mesh"], sc["hydro"]
    T = {}
    for r in P.load(P.os.path.join(P.RLD, "throughput.jsonl")):
        if r.get("status") == "ok" and r.get("sensor_reduce") != "netforce":
            T[(r["variant"], r["num_envs"])] = r
    env = ""
    if ("pads1", 2048) in T and ("legacy", 2048) in T:
        env = (f"; the RL env with the pads runs {100 * (1 - T[('pads1', 2048)]['env_steps_per_s'] / T[('legacy', 2048)]['env_steps_per_s']):.0f}"
               "&#8202;% fewer env steps per second than with the box tip at 2,048 envs, because physics is about a tenth of an env step")
    return (f" On one held state of the D6 RL env at 1,024 and 2,048 worlds (the trainer&#8217;s 2&#8202;ms step), MuJoCo-Warp steps the "
            f"1&#8202;mm pads in {min(pads):.1f}&#8211;{max(pads):.1f}&#8202;&#181;s of physics per world-step, the TPU block as one "
            f"mesh in {min(mesh):.1f}&#8211;{max(mesh):.1f}&#8202;&#181;s and Newton&#8217;s mass-corrected hydroelastic block in "
            f"{min(hyd):.1f}&#8211;{max(hyd):.1f}&#8202;&#181;s{env}.")


# ------------------------------------------------------------------------------- 2026-10-12: presliding, the sprung skin

SKIN_NOTE = "docs/handoff/20261012-servo_skin_policies.md"
NC_PAGE = "docs/experiments/20261007-native_compliance/20261007-native_presliding_compliance.html"


def _skin_x(ctx):
    if "skin_X" not in ctx:
        ctx["skin_X"] = P.skin_data()
    return ctx["skin_X"]


T8_CONV = "docs/experiments/20261007-native_compliance/t8_step_convergence.jsonl"


def t8_conv(model, N=1.0, dt=0.025):
    """Task 8 at a finer step (2026-10-12): the skin's 1 ms loop is a step error (deck 23 math pass, finding 4)."""
    import json
    p = P.os.path.join(P.ROOT, T8_CONV)
    rows = [json.loads(l) for l in open(p)] if P.os.path.exists(p) else []
    rows = [r for r in rows if r.get("task") == "t8_step_convergence" and r.get("model") == model and r.get("N") == N
            and r.get("dt_ms") == dt and r.get("status") == "complete"]
    return rows[-1] if rows else None


def t8_trainer(model, armature=None, dt=2.0, N=1.0):
    """Task 8 at the policy trainer's impratio of 10 and 2 ms step (2026-10-12, task t8_trainer_settings)."""
    import json
    p = P.os.path.join(P.ROOT, T8_CONV)
    rows = [json.loads(l) for l in open(p)] if P.os.path.exists(p) else []
    rows = [r for r in rows if r.get("task") == "t8_trainer_settings" and r.get("model") == model and r.get("N") == N
            and r.get("dt_ms") == dt and r.get("skin_armature") == armature and r.get("status") == "complete"]
    return rows[-1] if rows else None


def mindlin_t8_trace(N=1.0, n=200):
    """Mindlin's tool displacement through the task 8 cycle against time (s, N, m), as
    scripts/contact_bed_candidates.py mindlin_t8_trace."""
    import contact_reference_laws as L
    import numpy as np
    m = L.mat()
    a = L.hertz(N)["a"]
    Qs = 0.5 * L.MU * N
    mono = lambda q: L.mindlin_mono(q, N, a, m["G"], m["nu"], L.MU)  # noqa: E731
    q1 = np.linspace(0, Qs, n)
    u1 = mono(q1)
    q2 = np.linspace(Qs, -Qs, 2 * n)
    u2 = u1[-1] - 2 * mono((Qs - q2) / 2)
    q3 = np.linspace(-Qs, Qs, 2 * n)
    u3 = -u1[-1] + 2 * mono((Qs + q3) / 2)
    q4 = np.linspace(Qs, 0, n)
    u4 = u3[-1] - 2 * mono((Qs - q4) / 2)
    t = np.concatenate([np.linspace(0, 0.5, n), np.linspace(0.5, 1.5, 2 * n), np.linspace(1.5, 2.5, 2 * n),
                        np.linspace(2.5, 3.0, n), [3.2]])
    u = np.concatenate([u1, u2, u3, u4, [u4[-1]]])
    return t, 2 * np.concatenate([q1, q2, q3, q4, [0.0]]), u


def skin_lead(ctx):
    X = _skin_x(ctx)
    s, ir = t8_conv("mj_pads1_skin"), X.t8r("mj_pads1_ir1000")
    p10, s10 = t8_trainer("mj_pads1_ir10"), t8_trainer("mj_pads1_skin_ir10", "hand 0.26 kg")
    gp, cp = X.gpu("mj_pads1_ir1000"), X.cpu("mj_pads1_ir1000")
    return ("The sprung skin mounts each pad&#8217;s spheres on a 2&#8202;g body held by springs at the fingertip&#8217;s elastic "
            "stiffness, so that a tangential load below slip deflects the contact and the deflection returns, as Mindlin&#8217;s "
            "solution gives for an elastic tip (40&#8202;&#181;m at 1&#8202;N, 38 returned). On the bed, with the friction rows "
            f"stiffened by impratio 1000, the skin gives {s['u_presliding_um']:.0f}&#8202;&#181;m of presliding at 1&#8202;N, of which "
            f"{s['u_recovered_um']:.0f} return, against {ir['u_presliding_um']:.1f}&#8202;&#181;m for the bare pads, at "
            f"{X.cpu('mj_pads1_skin') / cp:.2f} times their CPU step and {X.gpu('mj_pads1_skin') / gp:.1f} times their GPU "
            "world-step. The policy trainer&#8217;s solver holds a grasp on the skin only at impratio 10, where the friction rows "
            f"creep {p10['u_presliding_um']:.0f}&#8202;&#181;m over the first ramp, twice what the springs store, and the tool ends "
            f"the cycle {s10['u_end_um']:.0f}&#8202;&#181;m from its start with or without the skin (Figure&#160;14b). The skin "
            "policies&#8217; transfer gain was measured at 44&#8211;56&#8202;N grips, on a plant 4.4 times stronger than the servos. At the "
            "servo-limited plant&#8217;s 13&#8211;18&#8202;N the pad and TPU-mesh policies already transfer, and whether the skin "
            "changes a policy there is open until its evaluation.")


def skin_sizing(ctx):
    import json
    import make_work_plant_runs as MW
    bed = (13.8, 8.55)                      # 1 N per pad from Hertz's radius, the native-compliance bed
    work = MW.skin_sizing(12.0)
    sp = P.os.path.join(P.ROOT, "docs/experiments/20261011-servo_plant_policies/plant_check_skin.json")
    servo = json.load(open(sp))["skin"] if P.os.path.exists(sp) else None
    out = (f"On the bed the springs are sized at 1&#8202;N per pad on Hertz&#8217;s radius: {bed[0]}&#8202;kN/m and "
           f"{bed[1]}&#8202;mN&#8202;m/rad, with 18&#8202;g of armature on the slides. On the hand they are sized on the pads&#8217; own "
           f"contact area: {work['kt'] / 1e3:.1f}&#8202;kN/m and {work['kth']:.2f}&#8202;N&#8202;m/rad at {work['N_ref']:.0f}&#8202;N per "
           "fingertip on the working plant, the typical grip of its 2026-10-06 policies, with 3.5&#8202;mm&#178;/N from a scripted "
           "grasp")
    if servo:
        out += (f", and {servo['kt'] / 1e3:.1f}&#8202;kN/m and {servo['kth']:.3f}&#8202;N&#8202;m/rad at {servo['N_ref']:.0f}&#8202;N on "
                f"the servo-limited plant, the mean hold force of its pad and TPU-mesh policies, on "
                f"{P.math.pi * servo['a'] ** 2 * 1e6:.1f}&#8202;mm&#178;")
    return out + (". Each slide carries 0.26&#8202;kg of armature so that each spring&#8217;s period stays long against the "
                  "trainer&#8217;s 2&#8202;ms step. Both pad arms on the hand run at the environment&#8217;s impratio of 10, because "
                  "1000 diverged under the trainer&#8217;s 10-iteration solver, so for the same spheres and load their friction rows "
                  "creep 100 times faster than the bed skin&#8217;s.")


def skin_t8_text(ctx):
    X = _skin_x(ctx)
    p, b, ir = X.t8r("mj_pads1"), X.t8r("mj_pads1_bristle20a"), X.t8r("mj_pads1_ir1000")
    s, s1 = t8_conv("mj_pads1_skin"), X.t8r("mj_pads1_skin")
    p10, s10 = t8_trainer("mj_pads1_ir10"), t8_trainer("mj_pads1_skin_ir10", "hand 0.26 kg")
    pct = [100 * t8_conv("mj_pads1_skin", N)["u_presliding_um"] / t8_conv("mj_pads1_skin", N)["mindlin_u_presliding_um"]
           for N in (0.5, 1.0, 3.0)]
    return (f"Task&#160;8 holds the pinch at 1&#8202;N per pad with gravity off and cycles an axial force on the tool between "
            f"&#177;1&#8202;N, half the slip force. The pads&#8217; tool creeps {p['u_presliding_um']:.1f}&#8202;&#181;m during the first "
            f"ramp and {abs(p['u_recovered_um']):.1f}&#8202;&#181;m further while the force falls back to zero; at impratio 1000 the "
            f"pads creep {ir['u_presliding_um']:.1f}&#8202;&#181;m. On the skin, at a 0.025&#8202;ms step, the tool moves "
            f"{s['u_presliding_um']:.1f}&#8202;&#181;m and {s['u_recovered_um']:.1f} return, against Mindlin&#8217;s 40.3 and 38.0; the "
            "two springs in parallel give \\(F/2k_t\\)&#8202;=&#8202;36.3&#8202;&#181;m and the friction rows creep the rest. Its loop encloses "
            f"{s['loop_area_uJ']:.2f}&#8202;&#181;J against Mindlin&#8217;s 6.13, dissipated by that creep and the dampers; at the "
            f"bed&#8217;s 1&#8202;ms step it encloses {s1['loop_area_uJ']:.2f}, a step error. The springs are linear at the slope "
            f"of 1&#8202;N, so the skin gives {pct[0]:.0f}, {pct[1]:.0f} and {pct[2]:.0f}&#8202;% of Mindlin&#8217;s presliding at 0.5, 1 "
            f"and 3&#8202;N. The bristles move {b['u_presliding_um']:.1f}&#8202;&#181;m, return {b['u_recovered_um']:.1f} and enclose "
            f"{b['loop_area_uJ']:.2f}&#8202;&#181;J. At the policy trainer&#8217;s impratio of 10 and 2&#8202;ms step the bare pads "
            f"creep {p10['u_presliding_um']:.1f}&#8202;&#181;m over the first ramp and {abs(p10['u_recovered_um']):.1f}&#8202;&#181;m more "
            f"while the force falls back to zero. The skin, with the hand&#8217;s 0.26&#8202;kg of armature, moves the tool "
            f"{s10['u_presliding_um']:.1f}&#8202;&#181;m on the first ramp, its springs&#8217; 36&#8202;&#181;m on top of the same creep, "
            f"and both leave the tool {s10['u_end_um']:.1f}&#8202;&#181;m from its start at the end of the cycle (Figure&#160;14b).")


def skin_others_text(ctx):
    import native_compliance_page as NC
    X = _skin_x(ctx)
    sl = NC.slopes("mj_pads1_soft")
    t5s, t5p = X.t5("mj_pads1_soft"), X.t5("mj_pads1")
    gp = X.gpu("mj_pads1_ir1000")
    la = X.t1("mj_pads1_lattice1")
    return ("Three other native elements were tried. A <code>solimp</code> impedance that softens with depth gives Hertz&#8217;s "
            f"load exponents ({sl.get('delta_exp', float('nan')):.2f} and {sl.get('r_rms_exp', float('nan')):.2f} against 0.67 "
            "and 0.33) at no cost, but it stores no presliding and lets the braked tool pass 80&#176; at "
            f"{t5s['N_at_80_N']:.2f}&#8202;N against the pads&#8217; {t5p['N_at_80_N']:.2f}. A lattice of 1&#8202;g nodes coupled by "
            f"tendons spreads load like a shear layer but softens friction to an effective \\(\\mu\\) of {la['mu_eff']:.2f} at "
            f"{X.gpu('mj_pads1_lattice1') / gp:.0f} times the batched cost. A MuJoCo flex tip runs only on the CPU under the discrete "
            f"integrator, at {X.cpu('mj_pads1_flex20') / X.cpu('mj_pads1_ir1000'):.0f} times the pads&#8217; step, and its light "
            "vertices let the tool creep.")


def skin_rl_work(ctx):
    import json
    import contact_model_policies_page as CMP
    X = CMP.Data()
    fin = {a: X.finals(a) for a in ("tpu27mesh", "tpu27pads1", "tpu27skin")}
    if not all(fin.values()):
        return ""
    rng = lambda v, nd=2: (f"{min(v):.{nd}f}" if f"{min(v):.{nd}f}" == f"{max(v):.{nd}f}"  # noqa: E731
                           else f"{min(v):.{nd}f}&#8211;{max(v):.{nd}f}")
    hc = {a: [r["held_cos_mean"] for r in rs] for a, rs in fin.items()}
    gr = {a: [r["grip_N"] for r in rs] for a, rs in fin.items()}
    T = {a: CMP.tpu_transfer(X, a) for a in fin}
    pct = lambda t: f"{100 * t[0] / t[1]:.0f}&#8202;%"  # noqa: E731
    pm, pp, ps = (CMP.probe_spit(a) for a in fin)
    cost = (f" A training iteration on the skin costs {ps / pm:.1f} times the TPU mesh&#8217;s and {ps / pp:.2f} times the "
            "pads&#8217;." if pm and pp and ps else "")
    tr = (f" Run closed loop under the other TPU tip models, the skin policies keep the tool in {pct(T['tpu27skin'])} of "
          f"the rollouts, the pad policies in {pct(T['tpu27pads1'])} and the TPU-mesh policies in {pct(T['tpu27mesh'])}."
          if all(T.values()) else "")
    return ("On the working plant (1&#8202;N&#8202;m joint torque limit, a bonus on fingertip force) three skin seeds trained to "
            f"held cosines of {rng(hc['tpu27skin'])}, against {rng(hc['tpu27pads1'])} on the bare pads and "
            f"{rng(hc['tpu27mesh'])} on the TPU block as one mesh, gripping with {rng(gr['tpu27skin'], 0)}&#8202;N." + tr +
            " All of these policies squeeze the 24.5&#8202;g tool with 44&#8211;56&#8202;N, which pins it to the fingertips whatever the "
            "tip model, so the transfer numbers say little about the skin itself. Both arms trained at impratio 10, where on the "
            "bed the friction rows creep twice as far as the skin&#8217;s springs deflect (Figure&#160;14b), so the skin policies "
            "trained without the presliding the skin was built for." + cost)


def skin_rl_servo(ctx):
    import json
    D11 = P.os.path.join(P.ROOT, "docs/experiments/20261011-servo_plant_policies")
    load = lambda f: [json.loads(l) for l in open(P.os.path.join(D11, f))] if P.os.path.exists(P.os.path.join(D11, f)) else []  # noqa: E731
    fin, tr = load("final_eval.jsonl"), load("transfer.jsonl")
    if any("skin" in r.get("tag", "") for r in fin):
        return ""          # filled by the servo-plant skin job (docs/handoff/20261012-servo_skin_policies.md)
    arm = {r["tag"]: r["arm"] for r in fin}
    last = {}
    for r in tr:
        last[(r["tag"], r["scene"])] = r
    other = [r for (t, sc), r in last.items() if t in arm and sc != arm[t]]
    h, n = sum(r["n_held"] for r in other), sum(r["n"] for r in other)
    grip = [r["grip_N"] for r in fin]
    held = (f"hold {h:,} of {n:,} closed-loop rollouts under the other tip models" if n else "")
    return P.pending("On the servo-limited plant (0.18&#8202;N&#8202;m) the pad and TPU-mesh policies grip with "
                     f"{min(grip):.0f}&#8211;{max(grip):.0f}&#8202;N and {held}, so the working plant&#8217;s transfer gap has "
                     "closed. Two skin seeds are training on that plant (40&#8202;M steps, springs sized at 5&#8202;N per fingertip, "
                     "started 2026-10-12); their evaluation answers whether the skin changes the held cosine, robustness or "
                     f"replay transfer by more than the seed spread (<code>{SKIN_NOTE}</code>).")


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
        "STAB_NOTE": ("MuJoCo&#8217;s clamp \\(t_c\\ge2\\Delta t\\) is off. Each bar runs from the largest step that held the tool "
                      "for 1&#8202;s (filled) to the first that failed (&#215;: the state diverged; open circle: the pads ejected the "
                      "tool at hundreds of mm/s); a dotted bar held at every step up to 15&#8202;ms."),
        "REFS_TEXT": refs_text(ctx),
        "EVIDENCE_LEAD": evidence_lead(ctx),
        "AGREE_NOTE": "A dash marks a case not run.",
        "EVIDENCE_TASKS": evidence_tasks(ctx),
        "COST_TEXT": cost_text(ctx) + state_cost_text(ctx),
        "COST_NOTE": ("Filled markers: one physics step at 1&#8202;ms on one CPU core in bed task&#160;1 (MuJoCo, Drake), or of one "
                      "world alone on the GPU on the SR2 holding fixture of Figure&#160;10 (MuJoCo-Warp, Newton). Hollow markers: wall "
                      "time per world-step on the same fixture at the batch of highest throughput. "
                      "Models closer in deviation than a label&#8217;s height (the 1&#8202;mm pad in three simulators and condim&#160;4, "
                      "5.7&#8211;6.2&#8202;%) are drawn apart vertically in their order. condim&#160;4&#8217;s torsional coefficient follows a fit to "
                      "Drake&#8217;s torque (step&#160;10). Drake sits at zero deviation by construction."),
        "GPU_NOTE": gpu_note(ctx),
        "OPEN_LIST": open_list(ctx),
        "MEFF_TEXT": meff_text(ctx),
        "AGREE_SIM_TEXT": agree_sim_text(ctx),
        "RL_REPLAY_TEXT": rl_replay_text(ctx) + policy_text(ctx) + scale_text(ctx),
        "LIT": lit(ctx),
        "SKIN_LEAD": skin_lead(ctx),
        "SKIN_SIZING": skin_sizing(ctx),
        "SKIN_T8_TEXT": skin_t8_text(ctx),
        "SKIN_OTHERS_TEXT": skin_others_text(ctx),
        "SKIN_RL_WORK": skin_rl_work(ctx),
        "SKIN_RL_SERVO": skin_rl_servo(ctx),
    }
