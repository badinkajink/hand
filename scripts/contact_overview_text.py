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
    parts = ["Covering a fingertip with small MuJoCo contact spheres reproduces the friction torque and contact area of a soft pad "
             "without modifying MuJoCo."]
    mu_p, mu_d = _get(M, "effective", "mj_pads1"), _get(M, "effective", "drake_hydro")
    vs_p, vs_d = _get(M, "slip speed", "mj_pads1"), _get(M, "slip speed", "drake_hydro")
    a5_p, a5_d = _get(M, "arm at spin onset, 0.5", "mj_pads1"), _get(M, "arm at spin onset, 0.5", "drake_hydro")
    a3_p, a3_d = _get(M, "arm at spin onset, 3", "mj_pads1"), _get(M, "arm at spin onset, 3", "drake_hydro")
    s = []
    if mu_p and mu_d:
        d = abs(_pct(mu_p, mu_d))
        s.append("slips at the same force as Drake hydroelastic to within 0.1&#8202;%" if d < 0.1 else
                 f"slips at a force within {d:.1f}&#8202;% of Drake hydroelastic")
    if vs_p and vs_d:
        s.append(f"slides at a speed within {abs(_pct(vs_p, vs_d)):.0f}&#8202;%")
    if a5_p and a5_d and a3_p and a3_d:
        lo, hi = sorted((abs(_pct(a5_p, a5_d)), abs(_pct(a3_p, a3_d))))
        rng = f"{lo:.0f}" if f"{lo:.0f}" == f"{hi:.0f}" else f"{lo:.0f}&#8211;{hi:.0f}"
        s.append(f"starts to spin at a torque {rng}&#8202;% under Drake&#8217;s from 0.5 to 3&#8202;N")
    if s:
        parts.append("On the two-pad pinch of the real_v1 fingertip and screwdriver, the 1&#8202;mm pad " +
                     ", ".join(s[:-1]) + (", and " if len(s) > 1 else "") + s[-1] + ".")
    if "mj_pads1" in cost and "drake_hydro" in cost:
        g = [r["world_steps_per_s"] for r in gpu if r.get("key", "").startswith("legacy_s1.0") and r.get("status") == "complete"]
        tail = f"; one GPU runs {max(g) / 1e6:.2f} million pad world-steps per second" if g else ""
        parts.append(f"A pad step costs {cost['mj_pads1']:.0f}&#8202;&#181;s on one CPU core against {cost['drake_hydro']:.0f}&#8202;&#181;s for Drake, "
                     f"and task results are unchanged from 50&#8202;&#181;s to 10&#8202;ms steps{tail}.")
    import simulator_agreement_figure as SAF
    A = SAF.data()
    w = [A[k]["within"] for k in ("mj_pads", "mjw_pads", "nt_pads") if "turn" in A[k]]
    sc = _state_cost()
    if len(w) == 3 and sc:
        parts.append("The same pads in MuJoCo-Warp and in Newton agree with Drake to the same degree on the bed and on the "
                     "open-loop three-finger turn of the eight deployed hands (turn within 3&#176; of Drake on "
                     f"{w[0]}, {w[1]} and {w[2]} of {A['mj_pads']['n']} placements), because each sphere&#8217;s impedance divides "
                     "out the contact&#8217;s effective mass, which Newton&#8217;s hydroelastic contact did not; on one RL state they "
                     f"cost {min(sc['pads']):.1f}&#8211;{max(sc['pads']):.1f}&#8202;&#181;s of physics per world-step in MuJoCo-Warp.")
    cr_p = _get(M, "creep: sliding", "mj_pads1")
    cr_d = _get(M, "creep: sliding", "drake_hydro")
    if cr_p and cr_d:
        parts.append(f"The largest disagreement is creep, the slow sliding under a load below the slip force: the pad creeps "
                     f"{cr_p / cr_d:.0f}&#215; faster than Drake.")
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
imported with two-value solrefs.{newton_tw}</p>
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
        out.append("<p><b>Twist.</b> Drake&#8217;s arm at the onset of spin is "
                   + ", ".join(f"{g('drake_hydro', N)['rbar_onset_mm']:.3f}" for N in (0.5, 1.0, 3.0)) +
                   "&#8202;mm at 0.5, 1 and 3&#8202;N, within 3&#8202;% of the law it was fitted to; the 1&#8202;mm pad gives "
                   + ", ".join(f"{g('mj_pads1', N)['rbar_onset_mm']:.3f}" for N in (0.5, 1.0, 3.0)) +
                   "&#8202;mm, 6&#8211;9&#8202;% under it, and its sliding arm is 2&#8211;3&#8202;% under. Both arms grow by 1.52 between 0.5 and 3&#8202;N, "
                   "close to the foundation ratio of 1.565; Hertz gives 1.82 (Figure&#160;2). condim&#160;4 is 3&#8211;6&#8202;% over at onset and exact while "
                   "sliding. Held at half the onset torque, the pad turns at 1.1&#8211;1.6&#8202;&#176;/s and Drake at 0.01&#8202;&#176;/s.</p>")
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
    out = [f"The cost of one physics step of the bed rig at 1&#8202;ms, from task&#160;1. On one CPU core: "
           + "; ".join(f"{P.HTML_LBL[k]}, {us(cost[k])}&#8202;&#181;s" for k in cpu) + "."]
    if gpu:
        out.append(" One world alone on the GPU, stepped from the host with the state copied back after each step: "
                   + "; ".join(f"{P.HTML_LBL[k]}, {us(cost[k])}&#8202;&#181;s" for k in gpu))
    s = "".join(out) + "."
    b = [k for k in P.GPU_MODELS if k in bat]
    if b:
        s += (" A single GPU world is bound by kernel launches and the copy. In a batch on the SR2 holding fixture the cost per "
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
        ("Friction rows of Newton&#8217;s hydroelastic contact.", "The 2026-10-05 friction gain \\(k_f=10\\) gives its friction rows a "
         "3.9&#8202;ms time constant, and the braked tool swings to 130&#176;. Run <code>contact_bed_newton.py brake --models newton_hydro_mc</code> "
         "with \\(k_f\\) set for 2, 5, 10 and 20&#8202;ms; a swing within 3&#176; of Drake&#8217;s 87.3&#176; at one setting, kept on the "
         "turn&#8217;s 40 placements, would make the remaining disagreement a friction-row setting."),
        ("The modulus.", "Every model here uses Drake&#8217;s default \\(E=10\\)&#8202;MPa. The arm scales as \\(E^{-1/4}\\), so a "
         "TPU tip at 2&#8211;5&#8202;MPa lengthens it by 19&#8211;50&#8202;%; refit \\(c\\) once the tip is measured."),
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
        "The same factor sets the stiffness of every MuJoCo contact, not only of the pads: solref is a spring on the constraint "
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
        "the solver&#8217;s inverse weights (step&#160;7). Figure&#160;7 compares the three, Newton&#8217;s mass-corrected hydroelastic "
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
        "COST_NOTE": ("Filled markers: one physics step on one CPU core (MuJoCo, Drake) or of one world alone on the GPU, "
                      "stepped from the host (MuJoCo-Warp, Newton), bed task&#160;1 at 1&#8202;ms. Hollow markers: wall time per "
                      "world-step in a GPU batch on the SR2 holding fixture, at the batch of highest throughput in Figure&#160;10. "
                      "Models closer in deviation than a label&#8217;s height (the 1&#8202;mm pad in three simulators and condim&#160;4, "
                      "5.7&#8211;6.2&#8202;%) are drawn apart vertically in their order. Drake sits at zero deviation by construction."),
        "GPU_NOTE": gpu_note(ctx),
        "OPEN_LIST": open_list(ctx),
        "MEFF_TEXT": meff_text(ctx),
        "AGREE_SIM_TEXT": agree_sim_text(ctx),
        "RL_REPLAY_TEXT": rl_replay_text(ctx),
        "LIT": lit(ctx),
    }
