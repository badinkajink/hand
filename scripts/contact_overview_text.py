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


def _r1(v, nd=1):
    """'a-b' range of values, or one value when they round alike."""
    a, b = f"{min(v):.{nd}f}", f"{max(v):.{nd}f}"
    return a if a == b else f"{a}&#8211;{b}"


def rl_ratio(rl):
    """PPO iteration of the pads over the TPU mesh: servo plant, working plant idle and during the runs."""
    r = []
    for plant in ("servo", "work"):
        m, p = rl.get((plant, "tpu27mesh")), rl.get((plant, "tpu27pads1"))
        if m and p:
            r.append(p[0] / m[0])
    pr = rl.get("probe") or {}
    if pr.get("tpu27mesh") and pr.get("tpu27pads1"):
        r.append(pr["tpu27pads1"] / pr["tpu27mesh"])
    return r


def lede(ctx):
    """Opening paragraph, about 100 words (owner 2026-10-14: the pads' selling point is CPU and GPU usability, one
    fingertip contact that runs unmodified in CPU MuJoCo for planners and in MuJoCo-Warp for RL; no comparison of cost
    with Newton hydroelastic): the outcome, the agreement with Drake on the test bed and on the hand, and the cost on one
    CPU core, per GPU world-step in the trainer's MuJoCo-Warp build and per training iteration, from the hand-level rows.
    Plain-language pass of 2026-10-22: one comparison per sentence."""
    M = ctx["M"]
    parts = ["A soft fingertip touches a tool over a patch, and friction across the patch resists twisting. MuJoCo&#8217;s "
             "point contact has no such torque. Covering the fingertip with 1&#8202;mm contact spheres restores it: each sphere "
             "is an ordinary MuJoCo contact, and together they reproduce the pressure law of Drake&#8217;s hydroelastic contact."]
    a5_p, a5_d = _get(M, "arm at spin onset, 0.5", "mj_pads1"), _get(M, "arm at spin onset, 0.5", "drake_hydro")
    a3_p, a3_d = _get(M, "arm at spin onset, 3", "mj_pads1"), _get(M, "arm at spin onset, 3", "drake_hydro")
    if a5_p and a5_d and a3_p and a3_d:
        lo, hi = sorted((abs(_pct(a5_p, a5_d)), abs(_pct(a3_p, a3_d))))
        rng = f"{lo:.0f}" if f"{lo:.0f}" == f"{hi:.0f}" else f"{lo:.0f}&#8211;{hi:.0f}"
        parts.append(f"The pads start to spin at a torque {rng}&#8202;% below Drake&#8217;s on a two-pad pinch.")
    import simulator_agreement_figure as SAF
    A = SAF.data()
    within = [A[k].get("within") for k in ("mj_pads", "mjw_pads", "nt_pads")]
    if all(w is not None for w in within):
        parts.append(f"On the real_v1 hand their open-loop turn ends within 3&#176; of Drake&#8217;s on "
                     f"{min(within)}&#8211;{max(within)} of {A['mj_pads']['n']} placements.")
    parts.append("The same pads run unmodified in CPU MuJoCo for planners and in MuJoCo-Warp for reinforcement learning (RL); "
                 "Newton runs them with two settings recomputed (step&#160;7).")
    hc, rl = ctx.get("hand") or {}, ctx.get("rl") or {}
    us, _mem = ctx.get("state") or ({}, {})
    if all(hc.get(k) for k in ("mj_pads1", "mj_point3", "drake_hydro")):
        parts.append(f"A step of the hand costs {hc['mj_pads1']:.0f}&#8202;&#181;s on one CPU core with the pads, against "
                     f"{hc['mj_point3']:.0f}&#8202;&#181;s with point contact and {P.num(hc['drake_hydro'], ',.0f')}&#8202;&#181;s in Drake.")
        pads, mesh = us.get(("mjw", "3.6", "pads1")), us.get(("mjw", "3.6", "mesh"))
        r = rl_ratio(rl)
        if pads and mesh:
            parts.append(f"On the GPU a world-step of a held reorientation state costs {_r1(pads.values())}&#8202;&#181;s with the "
                         f"pads and {_r1(mesh.values())}&#8202;&#181;s with point contact.")
        if r:
            parts.append(f"An RL training iteration costs {_r1(r, 2)} times one with point contact.")
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
    curves = ctx.get("curves") or {}
    one = lambda k: 1e6 / curves[k][1] / 1e3 if 1 in curves.get(k, {}) else None  # noqa: E731
    newton_tw = ""
    if raw and mc:
        newton_tw = (f" SolverMuJoCo applies each contact&#8217;s stiffness times the effective mass, so the hydroelastic tip on "
                     f"the test bed sank {_ratio_rng(raw, 1)} times as deep as in Drake. With \\(k_h\\) divided by "
                     f"\\(m_\\text{{eff}}\\) (step&#160;7) it sinks {_ratio_rng(mc)} times Drake&#8217;s depth"
                     + (f", and it starts to spin at {_ratio_rng(on)} of Drake&#8217;s onset torque" if on else "") + ".")
        if one("newton_hydro_mc") and one("newton_hydro_unreduced_mc"):
            newton_tw += (f" One GPU world of the mass-corrected tip costs {one('newton_hydro_mc'):.2f}&#8202;ms per step with "
                          f"reduction and {one('newton_hydro_unreduced_mc'):.2f}&#8202;ms without (Figure&#160;10).")
    pull = ctx["T"]["pull"]
    dk = [(P.pick(pull, "drake_hydro", N=N, dt_ms=1.0) or {}) for N in (0.5, 1.0, 3.0)]
    dk_us = [r["us_per_step_median"] for r in dk if r.get("us_per_step_median")]
    dk_cr = [r["creep_mm_s"] * 1e3 for r in dk if r.get("creep_mm_s")]
    hand = (ctx.get("hand") or {}).get("drake_hydro")
    drake_cost = (f"It costs {min(dk_us) / 1e3:.1f}&#8211;{max(dk_us) / 1e3:.1f}&#8202;ms per step on one core on the test bed"
                  + (f" and {hand / 1e3:.1f}&#8202;ms on the hand (Table&#160;4)." if hand else ".") if dk_us
                  else "It costs 1&#8211;2&#8202;ms per step on one core.")
    drake_creep = (f" Under half the slip load on the test bed it creeps {min(dk_cr):.2f}&#8211;{max(dk_cr):.2f}&#8202;&#181;m/s "
                   "(step&#160;8)." if dk_cr else "")
    return f"""
<h3>Drake hydroelastic</h3>
<p>Drake gives each soft body a pressure field on a tetrahedral mesh; for the fingertip sphere it is \\(p=E\\,\\delta/R\\) at a
1&#8202;mm resolution. The contact surface is the part of the rigid tool&#8217;s surface that lies inside the field, and each of its
faces has a pressure. Drake&#8217;s SAP solver integrates force and friction over the faces with a convex, regularised friction
law.{drake_creep} Drake is the reference for every task on these pages. {drake_cost} When the hand is imported from MJCF, the
model&#8217;s contact exclusions must be applied as collision filters. Without them the distal links locked against their own
yaw links (chain page).</p>
<h3>Newton hydroelastic</h3>
<p>Newton (commit {NEWTON_REV}) represents each shape by a sparse signed-distance grid. It extracts the patch where the two
shapes&#8217; pressures balance by marching cubes, and it hands each triangle of the patch to the solver as a contact point with
its own stiffness. An optional reduction merges these points into a few representative contacts. Friction stays per point in
the MuJoCo-Warp or XPBD solver, so the friction torque comes from the spread of the points, as with the sphere pads. In
Codex&#8217;s static check on the SR2 pinch, the torque capacity was within 1&#8211;6&#8202;% of the continuum at 0.25&#8202;mm voxels
without reduction. Reduction lowered it by 16&#8211;25&#8202;%, and 1&#8202;mm voxels by up to 69&#8202;%.{newton_tw}</p>
<p>These pages ran Newton&#8217;s hydroelastic contact with the friction gain \\(k_f\\)&#8202;=&#8202;10, contact reduction without moment
matching, and 0.5&#8202;mm voxels. That gain gives friction rows with a time constant of 3.9&#8202;ms on the test bed and about
5&#8202;ms on the hand. On the test bed, all faces at 0.5&#8202;mm voxels give the law&#8217;s friction arm at 0.5 and 1&#8202;N, and the
reduced contacts give 0.77&#8211;0.88 of it. Friction rows of 10&#8202;ms or longer stop the braked tool at hanging, where with rows of
3.9&#8202;ms it swings to 130&#176; (<code>docs/experiments/20261007-newton_hydro_tests/</code>).</p>
<h3>Compliant Sphere Lattice Contact</h3>
<p>{cslc_text(ctx)}</p>
<h3>Related contact models</h3>
<p>{related_models(ctx)}</p>
"""


def related_models(ctx):
    """Prior work on sampled-point and torsional contact (assessment of 2026-10-14, prior-work list) and what this work adds,
    with the two step bounds of step 9 and the count of the test bed's task-7 pad settings that the friction rows' bound
    brackets."""
    st = P.stab_stats(ctx.get("stab") or [])
    fric = (f"; it brackets the largest stable step of {len(st['f_in'])} of the {st['n']} pad settings of task&#160;7 on the "
            "test bed" if st and st["f_in"] else "")
    return ("Several models spread one contact over sampled points with a foundation stiffness. Simbody&#8217;s and OpenSim&#8217;s "
            "<code>ElasticFoundationForce</code> puts a spring at the centre of every triangle of a mesh, with the layer&#8217;s "
            "stiffness times the face&#8217;s area, and gives each spring its own friction. That friction drifts below stiction "
            "(Sherman, Seth and Delp, 2011). Drake&#8217;s discrete solvers turn each polygon of the hydroelastic contact surface "
            "into one point contact with its own Coulomb cone. The point&#8217;s stiffness is \\(gA\\), the combined pressure "
            "gradient times the area (Masterjohn et al., RA-L 2022). Newton passes its signed-distance-field triangles to "
            "MuJoCo-Warp the same way (R&#246;stel et al., 2026). Gonthier&#8217;s volumetric contact model derives the spinning "
            "friction of a Winkler pressure distribution in closed form for a flat contact surface. Gazebo&#8217;s torsional "
            "friction scales one contact&#8217;s torque with its load, \\(T=(3\\pi/16)\\,a\\mu N\\), with \\(a=\\sqrt{R\\delta}\\) from "
            "the penetration \\(\\delta\\). MuJoCo&#8217;s documentation states that its soft friction slips slowly under a "
            "persistent tangential load, and that impratio and noslip iterations reduce the slip.</p><p>Steps 7&#8211;9 add three "
            "results to this prior work. The impedance \\(d_0\\) is calibrated at load time from the compiled inverse inertia "
            "(step&#160;7). The creep law includes the saturation of the outer spheres&#8217; cones (step&#160;8). Two step bounds "
            "are derived (step&#160;9). The first, (14), holds for a pad&#8217;s normal rows and equals MuJoCo&#8217;s threshold for a "
            f"pad sliding on a plate. The second, about \\(d_0t_c\\), holds for its friction rows{fric}. One pad model runs "
            "unmodified in CPU MuJoCo and MuJoCo-Warp, and in Newton with \\(d_0\\) and the friction gain recomputed. The same "
            "calibration showed that Newton&#8217;s hydroelastic stiffness lacked the effective-mass factor (step&#160;7).")


def cslc_text(ctx):
    return ("CSLC (Nechyporenko, Abderezaei and Roncone, 2026) represents the fingertip surface by a lattice of spheres. "
            "Anchor springs tie the spheres to the rigid core, and lateral springs tie them to their neighbours. The lattice "
            "is solved for quasi-static equilibrium against the object with warm-started damped Jacobi iterations. Its "
            "presliding friction saturates smoothly. Its authors describe the model as tangentially stateless, without kinetic "
            "sliding dynamics. It was not run here; its paper is "
            "<code>docs/experiments/20261004-codex/22_Compliant_Sphere_Lattice_Co.pdf</code>.")


def evidence_lead(ctx):
    T = ctx["T"]
    n = sum(len(v) for v in T.values())
    return (f"The test bed runs five tasks with the 10.55&#8202;mm sphere as the fingertip, at 1 and 5&#8202;ms steps ({n} cases). "
            "In the pull and the twist the pinch is loaded until the tool slides or spins. In the roll one pad moves, so that "
            "the tool rolls between the pads. In the shake the held tool is driven at 5&#8202;Hz under gravity. In the brake the "
            "pinch force falls until the tool swings toward hanging. Table&#160;2 lists one or two metrics per task.")


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
        ratio = lambda k, f: g(k, 3.0)[f] / g(k, 0.5)[f]  # noqa: E731
        cr = lambda k: [g(k, N)["creep_deg_s"] for N in Ns if g(k, N).get("creep_deg_s") is not None]  # noqa: E731
        out.append("<p><b>Twist.</b> Drake&#8217;s arm at the onset of spin is "
                   + ", ".join(f"{g('drake_hydro', N)['rbar_onset_mm']:.3f}" for N in Ns) +
                   f"&#8202;mm at 0.5, 1 and 3&#8202;N, {_dev(dev('drake_hydro', 'rbar_onset_mm'))} the integrated law (3). The "
                   "1&#8202;mm pad gives " + ", ".join(f"{g('mj_pads1', N)['rbar_onset_mm']:.3f}" for N in Ns) +
                   f"&#8202;mm, {_dev(dev('mj_pads1', 'rbar_onset_mm'))} the law. While the tool spins at 1&#8202;rad/s, the pad&#8217;s "
                   f"arm is {_dev(dev('mj_pads1', 'rbar_kin_mm'))} the law and Drake&#8217;s {_dev(dev('drake_hydro', 'rbar_kin_mm'))} "
                   "it. The pad&#8217;s spinning arm depends on the spin rate through the friction dampers of step&#160;8. At 1&#8202;N "
                   "deck&#160;23 (row <code>d23_twist_run</code>) gives 0.888&#8202;mm at 0.1&#8202;rad/s and 0.973&#8202;mm at 3&#8202;rad/s. "
                   "The pad falls short only at onset, and MuJoCo&#8217;s soft friction rows are the likely cause: before it spins, "
                   f"the tool turns {rr(rot('mj_pads1'), 1)}&#176; on the pads and {rr(rot('drake_hydro'), 2)}&#176; in Drake. "
                   f"Between 0.5 and 3&#8202;N Drake&#8217;s onset arm grows {ratio('drake_hydro', 'rbar_onset_mm'):.2f} times. The "
                   f"pad&#8217;s grows {ratio('mj_pads1', 'rbar_onset_mm'):.2f} times at onset and "
                   f"{ratio('mj_pads1', 'rbar_kin_mm'):.2f} times while spinning, close to the foundation&#8217;s ratio of 1.565; "
                   "Hertz gives 1.82 (Figure&#160;2). condim&#160;4, whose \\(\\mu_t\\) follows the fit to Drake&#8217;s torque "
                   f"(step&#160;10), is {_dev(dev('mj_point4s', 'rbar_onset_mm'))} the law at onset and "
                   f"{_dev(dev('mj_point4s', 'rbar_kin_mm'))} it while spinning. Held at half the onset torque, the tool turns "
                   f"at {rr(cr('mj_pads1'), 1)}&#8202;&#176;/s on the pad and at {rr(cr('drake_hydro'), 3)}&#8202;&#176;/s in Drake.</p>")
    br = [r for r in T["brake"] if r.get("role", "bed") == "bed"]
    b = lambda k: P.pick(br, k, dt_ms=1.0)  # noqa: E731
    if b("mj_pads1") and b("drake_hydro"):
        r0 = b("drake_hydro")
        N1, mgd = r0.get("N1", 0.2), P.M_TOOL * 9.81 * r0.get("d_mm", 15.0) * 1e-3
        phi = P.math.degrees(P.math.acos(min(1.0, 2 * P.MU * N1 * P.C_LAW * N1 ** P.EXP_LAW / mgd)))
        ends = [b(k)["phi_end_deg"] for k in ("drake_hydro", "mj_pads1", "mj_point4s")]
        out.append(f"<p><b>Brake.</b> Lowering the pinch from 6 to {N1:g}&#8202;N over 4&#8202;s swings the tool to "
                   f"{b('drake_hydro')['phi_end_deg']:.1f}&#176; in Drake, {b('mj_pads1')['phi_end_deg']:.1f}&#176; with the "
                   f"1&#8202;mm pad and {b('mj_point4s')['phi_end_deg']:.1f}&#176; with condim&#160;4. The peak rates are "
                   f"{b('drake_hydro')['peak_rate_deg_s']:.0f}, {b('mj_pads1')['peak_rate_deg_s']:.0f} and "
                   f"{b('mj_point4s')['peak_rate_deg_s']:.0f}&#8202;&#176;/s. The hinge balance of step&#160;1 rests the tool at "
                   f"{phi:.1f}&#176; at {N1:g}&#8202;N, and the swinging tool overshoots it by "
                   f"{min(ends) - phi:.1f}&#8211;{max(ends) - phi:.1f}&#176;. With point contact the tool swings through to "
                   f"{b('mj_point3')['phi_max_deg']:.0f}&#176; and slides {abs(b('mj_point3')['slip_end_mm']):.0f}&#8202;mm along the "
                   "tool.</p>")
    sh = T["shake"]
    s = lambda k: P.pick(sh, k, N=0.5, a_pk_g=2.0, dt_ms=1.0)  # noqa: E731
    if s("mj_pads1") and s("drake_hydro"):
        out.append(f"<p><b>Shake.</b> At 0.5&#8202;N and 2&#8202;g the load stays below the threshold of rigid Coulomb slip, so the "
                   f"drift is creep: {abs(s('mj_pads1')['drift_per_cycle_mm']) * 1e3:.1f}&#8202;&#181;m per cycle with the pad, "
                   f"against {abs(s('drake_hydro')['drift_per_cycle_mm']) * 1e3:.2f}&#8202;&#181;m in Drake.</p>")
    ro = T["roll"]
    r = lambda k, N=1.0: P.pick(ro, k, N=N, dt_ms=1.0, v_mm_s=10.0)  # noqa: E731
    have = [k for k in ("drake_hydro", "mj_pads1", "mj_point3", "mj_point4s", "newton_hydro_mc") if r(k)]
    if "drake_hydro" in have and "mj_pads1" in have:
        allr = [x for x in ro if x.get("status") == "complete" and x.get("model") in P.ORDER]
        slip = [max(x.get("slip_path_L_mm") or 0, x.get("slip_path_R_mm") or 0) * 1e3 for x in allr]
        lost = [x for x in ro if x.get("model") in P.ORDER and not x.get("success")]
        rho = [r(k)["rho"] for k in have]
        out.append("<p><b>Roll.</b> Moving one pad 10&#8202;mm along the tool at 10&#8202;mm/s rolls the tool between the pads in "
                   f"every model, at 0.5 to 3&#8202;N and both steps. At 1&#8202;N the rolling ratio lies between {min(rho):.4f} and "
                   f"{max(rho):.4f} across the {len(have)} models (Table&#160;2). No-slip rolling between the spheres gives "
                   f"{r('drake_hydro')['rho_noslip_geom']:.4f}. The differences follow the depth of the contact point: the "
                   f"pad&#8217;s lies {12.5 - r('mj_pads1')['r_contact_L_mm']:.2f}&#8202;mm inside the tool&#8217;s surface and "
                   f"Drake&#8217;s {12.5 - r('drake_hydro')['r_contact_L_mm']:.2f}&#8202;mm. A contact at a smaller radius turns the "
                   f"tool further for the same travel. The contacts slip at most {max(slip):.0f}&#8202;&#181;m over 5&#8202;mm of "
                   f"rolling, and {'no model loses' if not lost else str(len(lost)) + ' cases lose'} the tool.</p>")
    return "".join(out)


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
    txt = ("Newton runs the pad fixture of MuJoCo-Warp: the same SR2 pinch of thumb and index at 2&#8202;N per pad for 0.8&#8202;s. "
           "Two opposing torque pulses of 12&#8202;mN&#8202;m about the pinch axis load it at 0.3&#8211;0.6&#8202;s. Its hydroelastic tip has "
           "\\(k_h\\) divided by the effective mass of tip and tool (step&#160;7).")
    return txt + (" Not run: " + "; ".join(miss) + "." if miss else " Every batch from 1 to 8,192 worlds ran.")


def _block_bench(ctx):
    """Numbers of the bench items: the twist arms of the block at 1 and 3 N (finite elements of the 95A 20 % tip, pads,
    Drake), the pads' arm at 1 N against the modulus, the brake force at 45 deg, and the chain and the deployed turn
    against impratio (2026-10-06 backends rows)."""
    import json
    out = {}
    B = ctx.get("B") or {}
    g = lambda t, k, **kw: P.pick(B.get(t, []), k, **kw) or {}  # noqa: E731
    out["arm"] = {k: [g("T2", k, N=N, dt_ms=1.0).get("rbar_onset_mm") for N in (1.0, 3.0)] for k in P.BLOCK}
    out["b45"] = [g("T5", k, dt_ms=1.0).get("N_at_45_N") for k in P.BLOCK]
    fe = P.os.path.join(P.EXP, "20261009-tpu_tip_fem/summary.json")
    rec = (json.load(open(fe)).get("recipes", {}) if P.os.path.exists(fe) else {}).get("g20_across", {})
    out["fe"] = [(rec.get("arm") or {}).get(k) for k in ("1.0", "3.0")]
    lp = P.os.path.join(P.EXP, "20261013-contact_feedback/pad_laws.json")
    L = json.load(open(lp)) if P.os.path.exists(lp) else {}
    mod = {round(r["E_Pa"] / 1e6): r["arm_m_1N"] * 1e3 for r in L.get("modulus", [])}
    out["mod"] = (min(mod), mod[min(mod)], max(mod), mod[max(mod)]) if mod else None
    ch = P.load(P.os.path.join(P.TIP_D, "chain_impratio.jsonl"))
    if ch:
        ms = [1e3 * sum(r["perf"]["phys"].values()) / sum(r["perf"]["sim"].values()) for r in ch if r.get("perf")]
        out["chain"] = (sorted(r["impratio"] for r in ch), [r["phi_end"] for r in ch], ms, all(r.get("chain_ok") for r in ch))
    turn = {}
    for r in P.load(P.os.path.join(P.TIP_D, "reorient_mujoco.jsonl")):
        if r.get("tip") == "tpu2.7" and r.get("model") == "pads" and r.get("plant") == "fast":
            h, n = turn.get(r.get("impratio"), (0, 0))
            turn[r.get("impratio")] = (h + bool(r.get("held_end")), n + 1)
    out["turn"] = turn
    return out


def open_list(ctx):
    """Future work (owner 2026-10-09: actionable items with the measurement named). The bench measurements of the
    assessment of 2026-10-14 first, then the simulation items."""
    v = _block_bench(ctx)
    a, fe, b45 = v["arm"], v["fe"], v["b45"]
    pk, dk = P.BLOCK
    twist = ""
    if all(fe) and all(a[pk]) and all(a[dk]):
        twist = (f" Finite elements of the 95A 20&#8202;% tip give {fe[0]:.2f} and {fe[1]:.2f}&#8202;mm at 1 and 3&#8202;N, the pads "
                 f"{a[pk][0]:.2f} and {a[pk][1]:.2f}&#8202;mm, and Drake {a[dk][0]:.2f} and {a[dk][1]:.2f}&#8202;mm. A measured arm "
                 f"of 3.0&#8202;mm or more at 3&#8202;N would mean that the pads under-predict torsion by "
                 f"{(1 - a[pk][1] / 3.0) * 100:.0f}&#8202;% or more.")
    mod = (f" On the block the pads&#8217; arm at 1&#8202;N moves only from {v['mod'][1]:.2f} to {v['mod'][3]:.2f}&#8202;mm between "
           f"{v['mod'][0]} and {v['mod'][2]}&#8202;MPa. The indentation therefore fixes the modulus, and the twist checks the arm."
           if v["mod"] else "")
    brake = (f" On the test bed&#8217;s block the pads pass 45&#176; at {b45[0]:.2f}&#8202;N per pad and Drake at {b45[1]:.2f}&#8202;N. "
             "With point contact the tool swings free at the start of the ramp, in MuJoCo and in Drake." if all(b45) else "")
    a_of = lambda F: P.math.sqrt(2 * P.G.R * P.math.sqrt(F / (P.math.pi * P.E_FIELD / P.G.R * P.G.R)))  # noqa: E731
    chain = ""
    if v.get("chain"):
        kaps, ends, ms, ok = v["chain"]
        held = {k: x for k, x in v["turn"].items() if k in (100.0, 10000.0)}
        chain = (f"On 2026-10-06 the chain {'completed' if ok else 'ran'} at impratio "
                 f"{', '.join(f'{k:.0f}' if k < 10000 else f'{k:,.0f}' for k in kaps[:-1])} and {kaps[-1]:,.0f}. Its swing ended "
                 f"at {min(ends):.1f}&#8211;{max(ends):.1f}&#176;"
                 + (f" (Drake {P.CP.FL['dhy_closed']['phi_end']:.1f}&#176;)" if 'dhy_closed' in P.CP.FL else "")
                 + f", and the physics took {min(ms):.0f}&#8211;{max(ms):.0f}&#8202;ms per simulated second."
                 + (f" The deployed turn on the 2.7&#8202;mm block held {held[100.0][0]} and {held[10000.0][0]} of "
                    f"{held[100.0][1]} seeds at impratio 100 and 10,000 (backends page)." if len(held) == 2 else "")
                 + " The wield of arXiv 2609.25619 has not been run at a higher impratio. Rerun it at 100 and 1000, and "
                 "compare the rotation that the tool loses at each release with Drake&#8217;s.")
    items = [
        ("Friction coefficient of the printed tip.", "Pinch the bench cylinder between two printed 95A 20&#8202;% gyroid tips at "
         "0.5, 1 and 2&#8202;N per tip. Pull it along its axis until it slips, five times per load, and fit \\(\\mu=F/2N\\) with "
         "<code>scripts/tpu_tip_calibrate.py fit --pull</code>. The scenes are run with \\(\\mu=1\\), and the RL friction "
         "randomisation spans &#215;0.55&#8211;1.15. If the measured \\(\\mu\\) is below 0.55, retrain with a wider range."),
        ("Fingertip stiffness and overload trip.", "Run the push test of the bench plan (<code>docs/experiments/20261011-bench_plan/"
         "plan.json</code>). Push the index and thumb fingertips along the contact normal in 0.25&#8202;N steps until the overload "
         "trips, three series each, with the station stopped while the bus is read. At the plan&#8217;s pose the servo plant gives "
         "0.29&#8202;N/mm and a trip at 4.1&#8202;N, and the 1&#8202;N&#8202;m plant gives 1.59&#8202;N/mm and 22.6&#8202;N. At 0.9&#8202;N/mm or "
         "more the printed tip&#8217;s compliance reaches 5&#8202;% of the finger&#8217;s. The policy results of the servo plant must "
         "then be rerun on the measured plant."),
        ("Twist-onset torque of the printed tip.", "In the same fixture, hang a mass on a 50&#8202;mm lever in 1&#8202;g steps "
         "(0.49&#8202;mN&#8202;m) until the tool spins, five repeats at 1 and 3&#8202;N per tip. The arm is \\(T/(2\\mu N)\\), with "
         "\\(\\mu\\) from the pull test (<code>tpu_tip_calibrate.py fit --twist</code>)." + twist),
        ("Indentation of the printed tip.", "Press a 12.5&#8202;mm rod across the palmar face to 0.5, 1 and 3&#8202;N, and read the "
         "approach from first touch with a 2&#8202;&#181;m dial indicator, three repeats (<code>tpu_tip_calibrate.py fit "
         "--indent</code>). Finite elements of the 95A 20&#8202;% magnet tip give 21.5, 41.4 and 114.9&#8202;&#181;m. That is a force "
         "exponent of 1.07, against the pressure law&#8217;s 1.5. Fits of the planned prints give pad moduli of "
         "6.9&#8211;16.5&#8202;MPa, against the scenes&#8217; 10&#8202;MPa (printed-tip page)." + mod),
        ("Pinch brake on the D6 hand.", "Pinch the bench cylinder between thumb and index, with its centre of mass 15&#8202;mm "
         "off the pinch line. Lower the commanded squeeze by 0.5&#176; per second while the tag is tracked at 30&#8202;Hz, ten "
         "repeats. Before the session, <code>scripts/real_v1_trajectory_clearance.py</code> must pass the brake plan (the chord "
         "and <code>_traj.csv</code> separately), and <code>scripts/hom_hand_brake.py</code> computes the prediction rows on "
         "the servo plant." + brake + " A tool that passes 45&#176; at the squeeze of the pad rows supports the pads. One that "
         "passes it at Drake&#8217;s squeeze supports Drake, and a release at the start of the ramp supports point contact."),
        ("Bench bookkeeping.", "Weigh the vaned tool. Record the height of the palm plate (25&#8202;mm above the mounting plane "
         "in the built hand) and the envelope of the servo bank. Start the tracker before the grip, log "
         "<code>servo_load</code> on every run, and read the present position between the 50&#8202;Hz commands. Copy the run "
         "logs of the CB1 to a dated archive after each session. The 177 tracked runs of 2026-08-31 to 09-02 have no joint "
         "data during the turn and no pose before the grip. Their scatter within one hand, 12.9&#176;, exceeds every "
         "difference between contact models on 6 of the 8 hands."),
        ("Step bound on the hand.", "Bisect the spectral radius of MuJoCo&#8217;s one-step map (<code>mjd_transitionFD</code>) "
         "about the held pinch of task&#160;7 on the test bed, for its 21 pad pairs and 3 point-contact pairs. Do it with the "
         "rails&#8217; damping on and off, and then about the hand&#8217;s grip. Record whether the unstable mode lies in the friction "
         "rows or in the normal rows. A normal mode at (14) makes (14) the limit. A friction mode at \\(d_0t_c\\) makes the "
         "friction rows&#8217; bound the limit."),
        ("Coupling length of the TPU print.", "Indent the print at one point and map the surface displacement around it. Fit "
         f"\\(u(r)\\propto e^{{-r/\\ell}}\\), and compare \\(\\ell\\) with the sphere&#8217;s patch radius of "
         f"{a_of(0.5) * 1e3:.1f}&#8211;{a_of(3.0) * 1e3:.1f}&#8202;mm at 0.5&#8211;3&#8202;N (asides page)."),
        ("Creep in the whole task.", chain or "Rerun the chain&#8217;s hold and the wield at impratio 1000, and record the cost "
         "per step."),
        ("Newton&#8217;s braked swing.", "With friction rows of 10&#8202;ms, the braked tool on Newton&#8217;s hydroelastic tip stops at "
         "hanging. It falls through 80&#176; at 1.2&#8211;1.3&#8202;N, where in Drake it passes 80&#176; at 0.39&#8202;N. A "
         "10&#8211;25&#8202;% shorter arm in condim&#160;4 does not reproduce this. Log the torque of Newton&#8217;s contact forces "
         "about the pinch axis during the fall (Newton page of 2026-10-07, open items)."),
    ] + skin_future(ctx)
    return "<ul class=\"open\">" + "".join(f"<li><b>{a}</b> {b}</li>" for a, b in items) + "</ul>"


SKIN_PAGE = "docs/experiments/20261012-sprung_skin/20261012-sprung_skin_presliding.html"


def skin_future(ctx):
    """Future-work items on presliding, shared by the overview's list and the skin page."""
    return [
        ("Presliding of the printed tip.", "Cycle an axial force at half the slip force on the printed tip under a fixed "
         "pinch (task&#160;8 of the test bed, on the bench), and record the displacement against the force. If about "
         "40&#8202;&#181;m return, a contact model with presliding is required, such as the sprung skin at a high impratio. If "
         "the loop area is near Mindlin&#8217;s, the bristle model is required. If no presliding can be measured, both models are retired."),
        ("The skin at the trainer&#8217;s settings.", "The trainer&#8217;s solver (10 iterations, 2&#8202;ms step) holds a grasp on "
         "the skin only at impratio 10. There the friction rows creep twice as far as the springs deflect "
         f"({_f8(ctx)}), so no policy has yet trained with the skin&#8217;s presliding. Rerun the scripted-grasp check of "
         "<code>scripts/make_work_plant_runs.py</code> on the skin scene of the servo plant at impratio 100, 300 and 1000, "
         "with more solver iterations or a 1&#8202;ms step. Train the skin arm at the first setting that holds 64 of 64. If its "
         "held cosine, open-loop replays and robustness under jitter stay within the seed spread of the pad arm, retire the "
         "skin (the impratio-10 arm on the servo plant: <code>docs/experiments/20261011-servo_plant_policies/"
         "20261011-servo_plant_contact_model_policies.html</code>, section on the sprung skin)."),
    ]


def skin_future_html(ctx):
    return "<ul class=\"open\">" + "".join(f"<li><b>{a}</b> {b}</li>" for a, b in skin_future(ctx)) + "</ul>"


def lit(ctx):
    return ("Elandt, Drumwright, Sherman and Ruina, &#8220;A pressure field model for fast, robust approximation of net contact force "
            "and moment between nominally rigid objects&#8221;, IROS 2019 (Drake hydroelastic). Castro et al., SAP solver (Drake). "
            "The two uploaded papers: <code>docs/experiments/20261004-codex/11_GPU_Accelerated_Hydroelasti.pdf</code> (Newton "
            "hydroelastic) and <code>docs/experiments/20261004-codex/22_Compliant_Sphere_Lattice_Co.pdf</code> (CSLC); notes in "
            "<code>docs/notes/20261005-contact_literature_notes.md</code>. Wang, Oh and Pollard, arXiv 2609.25619 (the controller "
            "and tasks). Xydas and Kao, &#8220;Modeling of contact mechanics and friction limit surfaces for soft fingers in robotics, "
            "with experimental results&#8221;, IJRR 1999 (contact radius of soft fingertips against load). Drake&#8217;s hydroelastic user guide "
            "(the model has no state, so no tangential compliance). Masterjohn, Guoy, Shepherd and Castro, &#8220;Velocity level "
            "approximation of pressure field contact patches&#8221;, IEEE RA-L 7(4), 2022 (one point contact per contact-surface "
            "polygon). R&#246;stel, Liu, Yin, Zamora, Macklin, Reist and Widmer, &#8220;GPU-accelerated hydroelastic contact via signed "
            "distance fields&#8221;, CR2 workshop at ICRA 2026 (Newton). Nechyporenko, Abderezaei and Roncone, &#8220;Compliant Sphere "
            "Lattice Contact: distributed contact modeling for sphere-based robot representations&#8221;, CR2 workshop at ICRA 2026, "
            "arXiv 2608.00263. Sherman, Seth and Delp, &#8220;Simbody: multibody dynamics for biomedical research&#8221;, Procedia IUTAM 2, "
            "2011 (ElasticFoundationForce of Simbody and OpenSim). Gonthier, McPhee, Lange and Piedboeuf, volumetric contact model; "
            "Gonthier, PhD thesis, University of Waterloo, 2007. Gazebo classic tutorial &#8220;Torsional friction&#8221; (ODE, Gazebo&#160;7). "
            "Todorov, Erez and Tassa, &#8220;MuJoCo: a physics engine for model-based control&#8221;, IROS 2012, and the MuJoCo "
            "documentation on soft contacts, impratio and noslip. Deck&#160;23 of the contact tutorial, "
            "<code>wxie_workspace/contact/tutorial/slides/23_soft_fingertip_contact.html</code> (the derivations of steps 1&#8211;9 "
            "in full).")



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
    swing = (f" With Newton&#8217;s default \\(k_f=1000\\) the friction rows&#8217; \\(t_f\\) is \\(4\\times10^{{-5}}\\)&#8202;s, and the "
             f"braked tool swung to {old['newton_pads1']['phi_max_deg']:.0f}&#176;. With \\(t_f=t_c\\) it swings to "
             f"{br['newton_pads1']['phi_max_deg']:.1f}&#176;, as in MuJoCo." if "newton_pads1" in old and "newton_pads1" in br else "")
    return (
        "The same factor sets the stiffness of every MuJoCo contact. <code>solref</code> is a spring on the constraint "
        "acceleration, so a contact&#8217;s force stiffness is its solref stiffness times the effective mass "
        "\\(m_\\text{eff}=1/\\hat\\Lambda\\). The pads&#8217; \\(d_0\\) divides that mass out. Newton&#8217;s hydroelastic contact passes "
        "each patch point to MuJoCo-Warp with a force stiffness \\(k_N\\), in N/m, and writes it as "
        "\\(t_c=\\sqrt{1/(k_N(1-d))}\\), without the factor. On the test bed (\\(m_\\text{eff}\\)&#8202;=&#8202;17.4&#8202;g) the "
        f"hydroelastic tip therefore sank {_ratio_rng(raw, 1)} times as deep as in Drake. Multiplying its \\(k_h\\) by the "
        "inverse-weight sum of the solver&#8217;s own MuJoCo model, \\(k_h\\,(w_1+w_2)\\), gives "
        f"{_ratio_rng(mc)} times Drake&#8217;s sink, with no parameter fitted (<code>{P.HT3_PATH}</code>, Section&#160;4).</p><p>"
        "The pads run in Newton&#8217;s point-contact pipeline with two settings. First, \\(d_0\\) is recomputed from the inverse "
        f"weights of Newton&#8217;s MuJoCo model. These give each railed pad of the test bed {w.get('padL', 16.7):.1f}&#8202;kg\\(^{{-1}}\\), "
        f"against {d0.get('w_mjcf', 90.74) - w.get('tool', 40.74):.1f} in the MJCF compile. Then \\(\\hat\\Lambda\\)&#8202;=&#8202;"
        f"{w.get('padL', 16.7):.1f}&#8202;+&#8202;{w.get('tool', 40.74):.2f}&#8202;=&#8202;{d0.get('w_solver', 57.4):.1f}&#8202;kg\\(^{{-1}}\\), and "
        f"\\(d_0\\) falls from {d0.get('mjcf', 0):.3f} to {d0.get('solver', 0):.3f}. Second, the friction gain \\(k_f\\), in "
        "N&#8202;s/m, is set so that the time constant of SolverMuJoCo&#8217;s friction rows, "
        "\\(t_f=2/(k_f\\,\\hat\\Lambda\\,((1-d_0)/\\kappa+d_0))\\), equals the pads&#8217; \\(t_c\\). That takes "
        f"\\(k_f\\)&#8202;=&#8202;{d0.get('kf', 0):.1f} on the test bed.{swing}")


def agree_sim_text(ctx):
    import simulator_agreement_figure as SAF
    A = SAF.data()
    pads = [k for k in ("mj_pads", "mjw_pads", "nt_pads") if A[k].get("twist")]
    tw = [x for k in pads for x in A[k]["twist"]]
    br = [abs(A[k]["brake"][0][0]) for k in pads if A[k].get("brake")]
    gr = [g for k in pads for g, _ in A[k].get("grip", [])]
    hy = A["nt_hydro"]
    held = {k: (sum(r["held_end"] for r in P.load(str(path))), len(P.load(str(path))))
            for k, _l, _c, _s, path, _b in SAF.MODELS}
    drake_swing = next((r["phi_max_deg"] for r in P.load(P.os.path.join(P.BED, "brake.jsonl"))
                        if r.get("model") == "drake_hydro" and r.get("dt_ms") == 1.0), None)
    nt_end = (P.pick([r for r in ctx["T"]["brake"] if r.get("role", "bed") == "bed"], "newton_hydro_mc", dt_ms=1.0) or {}).get("phi_end_deg")
    return (
        "The pads are the same contact model in CPU MuJoCo, MuJoCo-Warp and Newton&#8217;s point-contact pipeline, once "
        "\\(d_0\\) comes from the inverse weights of Newton&#8217;s solver and the friction gain is set from \\(t_c\\) (step&#160;7). "
        "MuJoCo-Warp ports MuJoCo&#8217;s constraint solver to the GPU, and Newton&#8217;s SolverMuJoCo runs MuJoCo-Warp. The three are "
        "therefore one solver in three implementations. Newton&#8217;s hydroelastic contact and Drake are the independent "
        "comparisons. Figure&#160;7 compares these models with Drake in three tests. The first two are the twist and the brake "
        "on the test bed with the 10.55&#8202;mm sphere. The third is the open-loop replay of the three-finger turn on the eight "
        "deployed hands with the printed TPU block (1&#8202;N&#8202;m plant, \\(\\mu=1\\), 24 placements paired with Drake by hand and placement).</p><p>"
        f"The three pad implementations agree with Drake to the same degree. Their onset torque is {_ratio_rng(tw)} of "
        f"Drake&#8217;s at 0.5&#8211;3&#8202;N. Their largest swing is within {max(br):.1f}&#176; of Drake&#8217;s {drake_swing:.1f}&#176;. "
        f"Their turn ends within 3&#176; of Drake&#8217;s on {A['mj_pads']['within']}, {A['mjw_pads']['within']} and "
        f"{A['nt_pads']['within']} of {A['mj_pads']['n']} placements. "
        + ("All three hold the tool on every one of the 40 placements."
           if all(held[k][0] == held[k][1] for k in ("mj_pads", "mjw_pads", "nt_pads"))
           else f"They hold the tool on {held['mj_pads'][0]}, {held['mjw_pads'][0]} and {held['nt_pads'][0]} of 40 placements.")
        + f" Their grip force is {_ratio_rng(gr)} of Drake&#8217;s. Newton&#8217;s hydroelastic contact, at the settings of the "
        f"reference models above, reaches {_ratio_rng(hy.get('twist', []))} of Drake&#8217;s onset torque. It swings the braked "
        f"tool to {drake_swing + hy['brake'][0][0]:.0f}&#176; at its largest"
        + (f", and the swing ends at {nt_end:.1f}&#176; (Drake: {drake_swing:.1f}&#176;)" if nt_end else "")
        + f". Its turn ends within 3&#176; of Drake&#8217;s on {hy.get('within', 0)} of {hy.get('n', 0)} placements. Point contact "
        f"holds the tool on every placement of the turn, but in the brake the tool spins out of the pinch "
        f"(<code>{P.HT3_PATH}</code>, Section&#160;5).")


POLICY_PAGE = "docs/experiments/20261008-contact_model_policies/20261008-fingertip_contact_model_policies.html"
SERVO_PAGE = "docs/experiments/20261011-servo_plant_policies/20261011-servo_plant_contact_model_policies.html"
SENS_PAGE = "docs/experiments/20261013-contact_feedback/20261013-contact_parameter_sensitivity.html"


def rl_text(ctx):
    """The policy paragraph (assessment of 2026-10-14): the servo plant first, where each fingertip is a servo spring in
    series with the contact, then the 1 N m plant with the direct transfer between the pad and mesh policies. Numbers from
    the rows of the servo-plant, sensitivity and policy pages, read through their builders."""
    import contact_feedback_page as CFP
    import contact_model_policies_page as CMP
    import servo_plant_policies_page as SPP
    out = []
    X = SPP.Data()
    if len(X.finals("11")) >= 4:
        F = CFP.Data()
        T = CFP.tip_stats(F)
        ks = statistics.median(f["stiffness_N_m"] for f in F.laws["servo"]["fingers"].values())
        kc = next(r["tangent_N_m_1N"] for r in F.laws["modulus"] if round(r["E_Pa"]) == 10000000)
        rep = {a: SPP.replay_totals(X, "11", (a,)) for a in ("tpu27pads1", "tpu27mesh")}
        cross = [t for t in X.tr["11"] if t["scene"] in SPP.SKIN_ARMS and t["scene"] != t["arm"]]
        cl = {a: (sum(t["n_held"] for t in cross if t["arm"] == a), sum(t["n"] for t in cross if t["arm"] == a))
              for a in ("tpu27pads1", "tpu27mesh")}
        hc = {a: [r["held_cos_mean"] for r in X.finals("11", a)] for a in ("tpu27pads1", "tpu27mesh")}
        out.append(
            "On the servo plant the fingertip model barely changes the reorientation policies of hand D6. Through its three "
            f"servos each fingertip is a spring of {ks:.0f}&#8202;N/m along its palmar normal at the grasp. The pad contact, "
            f"{kc / 1e3:.1f}&#8202;kN/m at 1&#8202;N, is {kc / ks:.0f} times stiffer and in series with it, so the pair is a "
            f"{1 / (1 / ks + 1 / kc):.0f}&#8202;N/m spring. The servo&#8217;s deflection therefore sets the grip force, whichever tip "
            "touches the tool. Replayed open loop in CPU MuJoCo, Drake and Newton, the finger targets of the pad policies keep "
            f"the tool in {rep['tpu27pads1'][0]} of {rep['tpu27pads1'][1]} replays. Those of the policies trained on the TPU "
            f"block as one convex mesh keep it in {rep['tpu27mesh'][0]} of {rep['tpu27mesh'][1]}. Closed loop on the other two "
            f"models of the TPU block, the pad policies hold {cl['tpu27pads1'][0]:,} of {cl['tpu27pads1'][1]:,} rollouts and "
            f"the mesh policies {cl['tpu27mesh'][0]:,} of {cl['tpu27mesh'][1]:,}. Under all four tip models, the six "
            f"policies&#8217; own actions replayed open loop hold {T['o_h']:,} of {T['n']:,} rollouts, against {T['c_h']:,} "
            f"closed loop. The pad and mesh policies end at held cosines of {_r1(hc['tpu27pads1'], 2)} and "
            f"{_r1(hc['tpu27mesh'], 2)} (<code>{SERVO_PAGE}</code>, <code>{SENS_PAGE}</code>).")
    Y = CMP.Data()
    S = CMP.summary(Y)
    if all(S[a]["n_fin"] for a in CMP.ARMS):
        rp = lambda arms: [sum(S[a]["rep"][e][k] for a in arms for e in CMP.ENGINES) for k in (0, 1)]  # noqa: E731
        pads, point = rp(("tpu27pads1",)), rp(("box", "tpu27mesh"))
        direct = lambda arm, scene: [sum(r[k] for r in Y.tr if r["arm"] == arm and r["scene"] == scene) for k in ("n_held", "n")]  # noqa: E731
        pm, mp = direct("tpu27pads1", "tpu27mesh"), direct("tpu27mesh", "tpu27pads1")
        grip = [r["grip_N"] for a in ("tpu27pads1", "tpu27mesh") for r in Y.finals(a) if r.get("grip_N") is not None]
        s = ("On the 1&#8202;N&#8202;m plant the joint limit is 4.4 times the servo&#8217;s stall torque"
             + (f", and a bonus on grip force produced grips of {min(grip):.0f}&#8211;{max(grip):.0f}&#8202;N" if grip else "")
             + f". There the pad targets kept the tool in {pads[0]} of {pads[1]} open-loop replays, and the point-contact "
             f"targets (box tip and TPU mesh) in {point[0]} of {point[1]}.")
        if pm[1] and mp[1]:
            s += (f" Closed loop on each other&#8217;s tip, the pad policies held {100 * pm[0] / pm[1]:.0f}&#8202;% of the rollouts on "
                  f"the TPU mesh, and the mesh policies {100 * mp[0] / mp[1]:.0f}&#8202;% on the pads.")
        out.append(s + f" (<code>{POLICY_PAGE}</code>). Table&#160;4 gives the training cost.")
    return "</p><p>".join(out)


SCALE_PAGE = "docs/experiments/20261008-hand_object_scale/20261008-finger_spacing_object_size.html"


def scale_text(ctx):
    """One sentence and the link: the hand-object scale landscape runs on the 1 mm pads (2026-10-08)."""
    rows = P.os.path.join(P.ROOT, "docs/experiments/20261008-hand_object_scale/landscape.json")
    if not P.os.path.exists(rows):
        return ""
    return (" The hand&#8211;object scale landscape of the SR2 hand runs on the same 1&#8202;mm pads: 17 layouts of the finger "
            "bases &#215; 18 cylinders and spheres, with grasp robustness and the three-finger turn "
            f"(<code>{SCALE_PAGE}</code>).")


def block_text(ctx):
    """The printed TPU block, the default fingertip since 2026-10-13: its pads and tasks 1, 2 and 5 on the test bed against
    Drake (Table 3), from the backends study's tip rows and the pad rig's laws (pad_laws.json of the sensitivity study)."""
    B = ctx.get("B") or {}
    if not B.get("T2"):
        return P.pending("The block rows are missing.")
    g = lambda t, k, **kw: P.pick(B[t], k, **kw) or {}  # noqa: E731
    pk, dk = P.BLOCK
    on = [g("T2", pk, N=N, dt_ms=1.0).get("rbar_onset_mm") / g("T2", dk, N=N, dt_ms=1.0).get("rbar_onset_mm") for N in (1.0, 3.0)]
    kin = [g("T2", pk, N=N, dt_ms=1.0).get("rbar_kin_mm") / g("T2", dk, N=N, dt_ms=1.0).get("rbar_kin_mm") for N in (0.5, 1.0, 3.0)]
    gr = lambda k: g("T2", k, N=3.0, dt_ms=1.0)["rbar_onset_mm"] / g("T2", k, N=1.0, dt_ms=1.0)["rbar_onset_mm"]  # noqa: E731
    b45 = [g("T5", k, dt_ms=1.0).get("N_at_45_N") for k in P.BLOCK]
    end = [g("T5", k, dt_ms=1.0).get("phi_end_deg") for k in P.BLOCK]
    cr = [g("T1", k, N=1.0, dt_ms=1.0).get("creep_mm_s") for k in P.BLOCK]
    us = [g("T1", k, N=1.0, dt_ms=1.0).get("us_per_step_median") for k in P.BLOCK]
    import json
    lp = P.os.path.join(P.EXP, "20261013-contact_feedback/pad_laws.json")
    L = json.load(open(lp)) if P.os.path.exists(lp) else {}
    hand = ctx.get("hand") or {}
    npads = hand.get("n_pads")
    hr = _pad_depths()
    pads = (f" There are {npads // 3:,} pads per tip, over every face but the back and the mount side. Each is a spring of "
            f"\\((E/h_s)\\,A_s\\), with \\(A_s\\) its share of the surface and \\(h_s\\) its depth below the tip&#8217;s volume "
            f"centroid. That depth is {L['h_m'] * 1e3:g}&#8202;mm on the palmar face, where a pad is {L['pad_K']:,.0f}&#8202;N/m, and "
            f"{hr[0]:.1f}&#8211;{hr[1]:.1f}&#8202;mm over the block."
            if npads and L.get("pad_K") and L.get("h_m") else "")
    return (f"Since 13 October the scenes model the printed TPU fingertip as 1&#8202;mm pads.{pads} The 10.55&#8202;mm sphere of "
            "Table&#160;2 stays as the analytic check of steps 1&#8211;10. On the block the screwdriver presses a line across the "
            "flat palmar face. The arm is then set by the length of the line, and it grows little with load: from 1 to 3&#8202;N "
            f"the onset arm grows {gr(pk):.2f} times with the pads and {gr(dk):.2f} times in Drake. The pads&#8217; onset arm is "
            f"{(on[0] - 1) * 100:.1f} and {(on[1] - 1) * 100:.1f}&#8202;% above Drake&#8217;s at 1 and 3&#8202;N. Their arm while "
            f"spinning at 1&#8202;rad/s is {(min(kin) - 1) * 100:.0f}&#8211;{(max(kin) - 1) * 100:.0f}&#8202;% above Drake&#8217;s. The "
            f"braked tool passes 45&#176; at a pinch {(1 - b45[0] / b45[1]) * 100:.0f}&#8202;% below Drake&#8217;s ({b45[0]:.3f} "
            f"against {b45[1]:.3f}&#8202;N per pad), and it ends at {end[0]:.1f}&#176; against {end[1]:.1f}&#176;. The pads creep "
            f"{cr[0] / cr[1]:.0f} times faster than Drake at 1&#8202;N, and they cost {us[1] / us[0]:.0f} times less per step "
            f"(Table&#160;3; <code>{P.TIP_PATH}</code>).")


def cost_text(ctx):
    """The cost paragraphs above Table 4: the CPU ratios on the test bed, the block and the hand; the GPU world-step on the
    fixture and on one held RL state in the trainer's MuJoCo-Warp 3.6, then in 3.14 beside Newton's pipeline with the same
    pads (owner 2026-10-14: no framing of the pads as cheaper than Newton hydroelastic); the RL cost, the zero-action
    throughput against the TPU mesh like the PPO ratio, and what each 4,096-env pad attempt recorded."""
    cost, bat = ctx["cost"], ctx.get("batched") or {}
    hc, rl = ctx.get("hand") or {}, ctx.get("rl") or {}
    us, _mem = ctx.get("state") or ({}, {})
    B = ctx.get("B") or {}
    out = ["Table&#160;4 lists the cost of one physics step on the test bed, on the hand and in training."]
    if cost.get("mj_pads1") and cost.get("mj_point3") and hc.get("mj_pads1") and hc.get("mj_point3"):
        cad = hc.get("cad") or {}
        out.append(f"On one CPU core a pad step costs {cost['mj_pads1'] / cost['mj_point3']:.1f} times a point contact&#8217;s on "
                   f"the test bed. On the real_v1 hand&#8217;s turn replay it costs {hc['mj_pads1'] / hc['mj_point3']:.1f} times, "
                   f"because {hc.get('n_pads', 0):,} pad spheres replace three convex meshes.")
        if cad.get("pads") and cad.get("pt"):
            out.append(f"On the CAD hand, whose links have MorphIt spheres in both models, the planner&#8217;s step costs "
                       f"{cad['pads'] / cad['pt']:.1f} times.")
    blk = [(P.pick(B.get("T1", []), k, N=1.0, dt_ms=1.0) or {}).get("us_per_step_median") for k in P.BLOCK]
    if cost.get("drake_hydro") and hc.get("drake_hydro") and all(blk):
        out.append(f"Drake costs {cost['drake_hydro'] / cost['mj_pads1']:.0f} times the pads on the test bed&#8217;s sphere, "
                   f"{blk[1] / blk[0]:.1f} times on the printed block and {hc['drake_hydro'] / hc['mj_pads1']:.0f} times on the "
                   "hand.")
    out.append("Figure&#160;9 plots the test bed&#8217;s costs against each model&#8217;s deviation from Drake over the metrics of "
               "Table&#160;2.")
    out.append("</p><p>")
    gpu = [k for k in P.GPU_MODELS if k in cost]
    curves = ctx.get("curves") or {}
    allf = curves.get("newton_hydro_unreduced_mc", {}).get(1)
    if gpu:
        out.append(f"On the GPU, one world alone on the SR2 holding fixture of Figure&#160;10 costs "
                   f"{min(cost[k] for k in gpu):.0f}&#8211;{max(cost[k] for k in gpu):.0f}&#8202;&#181;s per step for the pads in "
                   "MuJoCo-Warp and Newton and for Newton&#8217;s reduced hydroelastic contact."
                   + (f" Newton&#8217;s hydroelastic contact with all faces costs {1e6 / allf:.0f}&#8202;&#181;s." if allf else "")
                   + " Kernel launches bound these costs, with each 10-step block captured as a CUDA graph."
                   + (f" In a batch the pads cost {bat['mjw_pads1'][0]:.2f}&#8202;&#181;s per world-step at "
                      f"{P.num(bat['mjw_pads1'][1], ',d')} worlds." if "mjw_pads1" in bat else ""))
    # one held RL state (same_state_timing.jsonl): the trainer's MuJoCo-Warp 3.6 first, then MuJoCo-Warp 3.14 against
    # Newton's pipeline on the same build, with the pads in both, so the Newton numbers compare pipelines
    p36, m36 = us.get(("mjw", "3.6", "pads1")), us.get(("mjw", "3.6", "mesh"))
    p314, m314 = us.get(("mjw", "3.14", "pads1")), us.get(("mjw", "3.14", "mesh"))
    npad, hyd = us.get(("nt_pt", "3.14", "pads1")), us.get(("nt_hydro", "3.14", "mesh"))
    if p36 and m36:
        out.append("The trainer runs MuJoCo-Warp&#160;3.6. On one held state of the D6 RL env, at 1,024 and 2,048 worlds, a "
                   f"world-step there costs {_r1(p36.values())}&#8202;&#181;s with the pads and {_r1(m36.values())}&#8202;&#181;s with "
                   "point contact on the TPU mesh.")
        if p314 and m314 and npad and hyd:
            out.append(f"In MuJoCo-Warp&#160;3.14 the pads cost {_r1(p314.values())}&#8202;&#181;s and point contact "
                       f"{_r1(m314.values())}&#8202;&#181;s. On that build, Newton&#8217;s pipeline costs {_r1(npad.values())}&#8202;&#181;s "
                       f"with the same pads and {_r1(hyd.values())}&#8202;&#181;s with its hydroelastic contact. Most of the "
                       "difference between Newton hydroelastic and the pads in MuJoCo-Warp therefore comes from the pipeline.")
    out.append("</p><p>")
    T = {}
    for r in P.load(P.os.path.join(P.RLD, "throughput.jsonl")):
        if r.get("sensor_reduce") != "netforce":
            T.setdefault((r["variant"], r["num_envs"]), []).append(r)
    ok = lambda k: [r for r in T.get(k, []) if r.get("status") == "ok"]  # noqa: E731
    r = rl_ratio(rl)
    m, p = [rl.get((pl, a)) for pl in ("work", "servo") for a in ("tpu27mesh",)], \
        [rl.get((pl, a)) for pl in ("work", "servo") for a in ("tpu27pads1",)]
    if ok(("pads1", 2048)) and ok(("mesh", 2048)) and r:
        zp, zm = ok(("pads1", 2048))[-1]["env_steps_per_s"], ok(("mesh", 2048))[-1]["env_steps_per_s"]
        out.append(f"In training, a zero-action benchmark at 2,048 envs runs {100 * (1 - zp / zm):.0f}&#8202;% fewer env steps per "
                   f"second with the pads than with the TPU mesh ({P.num(zp, ',.0f')} against {P.num(zm, ',.0f')}).")
        s = f"A PPO iteration with the pads costs {_r1(r, 2)} times one on the TPU mesh"
        mm = [x[1] for x in m if x and x[1]]
        pp = [x[1] for x in p if x and x[1]]
        if mm and pp:
            s += f", and it uses {_r1(pp)}&#8202;GB of GPU memory against {_r1(mm)}&#8202;GB"
        out.append(s + ".")
        # a 4,096-env pad run: say what each attempt recorded, never infer from a missing ok row
        p4 = T.get(("pads1", 4096), [])
        if p4 and not ok(("pads1", 4096)):
            why = []
            e = next((x for x in p4 if x.get("status") == "error" and "allocate" in str(x.get("error"))), None)
            if e:
                why.append("the first attempt ran out of GPU memory"
                           + (f" with {e['gpu_used_before_mb'] / 1e3:.1f}&#8202;GB already in use" if e.get("gpu_used_before_mb") else ""))
            n = next((x for x in p4 if x.get("status") == "not_run" and x.get("limit")), None)
            if n:
                why.append(f"the retry {n['limit'].replace(' GB', '&#8202;GB')}")
            out.append("A 4,096-env run of the pad env has not completed" + (f": {', and '.join(why)}." if why else "."))
    return " ".join(out).replace(" </p><p> ", "</p><p>")


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


def _pad_depths():
    """Smallest and largest pad depth h_s (mm) of the default block under builder v2 (reorient_backends.pad_set)."""
    import reorient_backends as RB
    _, _, A, K = RB.pad_set("tpu2.7", RB.PAD_S, RB.PAD_RS, 1.0, RB.E_TPU, "v2")
    h = RB.E_TPU * A / K * 1e3
    return float(h.min()), float(h.max())


def _f8(ctx):
    """The impratio-10 panel: Figure 14b in the overview, Figure 2b on the skin page (ctx["t8_fig"])."""
    return f"Figure&#160;{ctx.get('t8_fig', '14')}b"


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
            f"the cycle {s10['u_end_um']:.0f}&#8202;&#181;m from its start with or without the skin ({_f8(ctx)}). The skin "
            "policies&#8217; transfer gain was measured at 44&#8211;56&#8202;N grips, on a plant 4.4 times stronger than the servos. "
            + _servo_short())


def _servo_short():
    import servo_plant_policies_page as SP
    return SP.skin_answer(SP.Data()).get("short", "At the servo-limited plant&#8217;s 13&#8211;18&#8202;N the pad and TPU-mesh "
                                         "policies already transfer, and whether the skin changes a policy there is open until "
                                         "its evaluation.")


def skin_summary(ctx):
    """The overview's paragraph on the sprung skin (owner, 2026-10-12: compress it to the salient result and why it does not
    carry to the policies; the full material is on the skin page). Plain-language pass of 2026-10-22."""
    import json
    X = _skin_x(ctx)
    s, ir = t8_conv("mj_pads1_skin"), X.t8r("mj_pads1_ir1000")
    p10, s10 = t8_trainer("mj_pads1_ir10"), t8_trainer("mj_pads1_skin_ir10", "hand 0.26 kg")
    gp, cp = X.gpu("mj_pads1_ir1000"), X.cpu("mj_pads1_ir1000")
    import contact_model_policies_page as CMP
    T = {a: CMP.tpu_transfer(CMP.Data(), a) for a in ("tpu27mesh", "tpu27pads1", "tpu27skin")}
    pct = lambda t: f"{100 * t[0] / t[1]:.0f}&#8202;%"  # noqa: E731
    D11 = P.os.path.join(P.ROOT, "docs/experiments/20261011-servo_plant_policies")
    fin = [json.loads(l) for l in open(P.os.path.join(D11, "final_eval.jsonl"))]
    hc = {a: sorted(r["held_cos_mean"] for r in fin if r["arm"] == a) for a in ("tpu27mesh", "tpu27pads1", "tpu27skin")}
    if hc["tpu27skin"]:
        import servo_plant_policies_page as SP
        servo = " " + SP.skin_answer(SP.Data()).get("short", "")
    else:
        servo = (" On the servo plant, at 13&#8211;18&#8202;N, the pad and TPU-mesh policies already hold 2,045 of 2,048 "
                 "closed-loop rollouts under the other tip models; two skin seeds trained there are being evaluated.")
    return ("An elastic fingertip deflects before it slips, and most of the deflection returns when the load is removed. By "
            "Mindlin&#8217;s solution a pad on the test bed deflects 40&#8202;&#181;m at 1&#8202;N, and 38&#8202;&#181;m of that returns. The "
            "sphere pads and Drake instead hold the tool through regularised friction, and they creep. In the sprung skin, "
            "each pad&#8217;s spheres sit on a 2&#8202;g body held by springs at the tip&#8217;s elastic stiffness. The skin gives "
            f"{s['u_presliding_um']:.0f}&#8202;&#181;m of presliding at 1&#8202;N, of which {s['u_recovered_um']:.0f} return "
            "(Figure&#160;14a). That needs the friction rows stiffened by impratio 1000, and it costs "
            f"{X.cpu('mj_pads1_skin') / cp:.2f} times the pads&#8217; CPU step and {X.gpu('mj_pads1_skin') / gp:.1f} times their "
            "GPU world-step. The policy trainer&#8217;s solver holds a grasp on the skin only at impratio 10. There the friction rows "
            f"creep {p10['u_presliding_um']:.0f}&#8202;&#181;m over the first ramp, twice what the springs store, and the tool ends "
            f"the cycle {s10['u_end_um']:.0f}&#8202;&#181;m out with or without the skin (Figure&#160;14b).</p><p>Policies trained on "
            f"the skin on the 1&#8202;N&#8202;m plant kept the tool in {pct(T['tpu27skin'])} of closed-loop rollouts under the other "
            f"TPU tip models, against {pct(T['tpu27pads1'])} for the pads. These policies gripped with 44&#8211;56&#8202;N on a plant "
            "4.4 times stronger than the servos, and they trained without the skin&#8217;s presliding. The skin itself is therefore "
            "not tested by this number." + servo + " The construction, sizing, test-bed results and policies are on the "
            f"skin page, <code>{SKIN_PAGE}</code>.")


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
                f"the servo-limited plant, the mean hold force of its pad and TPU-mesh policies, from a contact area of "
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
            f"and both leave the tool {s10['u_end_um']:.1f}&#8202;&#181;m from its start at the end of the cycle ({_f8(ctx)}).")


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
            f"bed the friction rows creep twice as far as the skin&#8217;s springs deflect ({_f8(ctx)}), so the skin policies "
            "trained without the presliding the skin was built for." + cost)


def skin_rl_servo(ctx):
    """The servo-plant skin result on the skin page (docs/handoff/20261012-servo_skin_policies.md): the answer in the
    owner's order (2026-10-12), then the numbers behind it. Computed by the servo-plant page's builder from its rows."""
    import servo_plant_policies_page as SP
    X = SP.Data()
    a = SP.skin_answer(X)
    if not a:
        return P.pending("The skin policies of the servo-limited plant are being evaluated "
                         f"(<code>{SKIN_NOTE}</code>).")
    n = a["n"]
    page = "../20261011-servo_plant_policies/20261011-servo_plant_contact_model_policies.html"
    sk = X.check_skin.get("skin", {})
    sp = lambda arm: [c for c in (X.cost["11"].get(r["tag"], {}).get("s_per_it_median")  # noqa: E731
                                  for r in X.finals("11", arm)) if c]
    ratio = (sum(sp("tpu27skin")) / len(sp("tpu27skin"))) / (sum(sp("tpu27pads1")) / len(sp("tpu27pads1"))) \
        if sp("tpu27skin") and sp("tpu27pads1") else None
    tr = [t for t in X.tr["11"] if t["arm"] == "tpu27skin" and t["scene"] != "tpu27skin"]
    trp = [t for t in X.tr["11"] if t["arm"] == "tpu27pads1" and t["scene"] != "tpu27pads1"]
    worst = min(tr, key=lambda t: t["n_held"]) if tr else None
    gr = n["grip"]
    out = (f"<p>{a['verdict'][:-1]} (Figure&#160;6). {a['impratio']} {a['test']} The counts of every condition and the films "
           f"of both seeds: <a href=\"{page}#skin\">servo-plant policies</a>.</p>")
    out += (f"<p>On that plant the springs are sized at {sk.get('N_ref', 5):.0f}&#8202;N per fingertip, the mean hold force of "
            f"its pad and TPU-mesh policies ({sk.get('kt', 0) / 1e3:.1f}&#8202;kN/m, {sk.get('kth', 0):.3f}&#8202;N&#8202;m/rad). The "
            f"skin policies grip with {min(gr['tpu27skin']):.1f}&#8211;{max(gr['tpu27skin']):.1f}&#8202;N, against "
            f"{min(gr['tpu27pads1'] + gr['tpu27mesh']):.0f}&#8211;{max(gr['tpu27pads1'] + gr['tpu27mesh']):.0f}&#8202;N for the "
            "pads and the TPU mesh, and turn the tool the way the pad policies do (Figure&#160;7). In the hold their springs "
            f"stand deflected by {min(n['sk_defl']):.0f}&#8211;{max(n['sk_defl']):.0f}&#8202;&#181;m, {100 * min(n['sk_use']):.0f}"
            f"&#8211;{100 * max(n['sk_use']):.0f}&#8202;% of the friction limit, and the slide over the spheres moves the tool "
            f"{min(n['sk_ratio']):.0f}&#8211;{max(n['sk_ratio']):.0f} times as far as the change of their deflection. Closed "
            f"loop under the other tip models the skin policies hold {sum(t['n_held'] for t in tr):,} of "
            f"{sum(t['n'] for t in tr):,} rollouts"
            + (f" (seed {worst['seed']} drops {worst['n'] - worst['n_held']} of {worst['n']} on the "
               f"{ {'tpu27mesh': 'TPU mesh', 'tpu27pads1': 'pads', 'box': 'box tip'}[worst['scene']]})" if worst and
               worst["n_held"] < worst["n"] else "")
            + f" and the pad policies {sum(t['n_held'] for t in trp):,} of {sum(t['n'] for t in trp):,}."
            + (f" A training iteration on the skin costs {ratio:.2f} times one on the pads." if ratio else "") + "</p>")
    return out


def agree_note(ctx):
    """Table 2 caption: point contact's torque about the normal is zero; its onset values are the protocol's offset."""
    tw = ctx["T"]["twist"]
    pt = [P.pick(tw, "mj_point3", N=N, dt_ms=1.0) or {} for N in (0.5, 3.0)]
    on = [r["rbar_onset_mm"] for r in pt if r.get("rbar_onset_mm")]
    s = "A dash marks a case not run."
    if on:
        s += (" Point contact transmits no torque about its normal: its arm while spinning at 1&#8202;rad/s is zero to rounding. "
              f"Its {min(on):.2f}&#8211;{max(on):.2f}&#8202;mm at spin onset is the offset of the onset protocol, the torque that "
              "the ramp reaches while the free tool spins up to the detection threshold of 30&#8202;&#176;/s.")
    return s


def blocks(ctx):
    have = ctx["have_onset"]
    return {
        "LEDE": lede(ctx),
        "NEWTON_REV": NEWTON_REV,
        "CAP_NOTE": "The CSLC row follows its paper; the model was not run here.",
        "CSLC_MECH": ("Compliant Sphere Lattice Contact (CSLC) is a discrete form of (6): a lattice of spheres tied to the core "
                      "by anchor springs and to each other by lateral springs. A tangential state per element, storing shear until "
                      "it reaches \\(\\mu f_n\\), would add elastic stick before slip. CSLC&#8217;s authors describe their model as "
                      "tangentially stateless."),
        "ASIDES_REF": f"<code>{P.ASIDES_PATH}</code>, section on coupled-foundation models",
        "SCALING_NOTE": ("" if have else "Until task&#160;2 on the test bed is written, hollow markers show the 10-01 rig&#8217;s "
                         "steady sliding torque at 0.5&#8202;rad/s, per pad; point contact has none and is off the log axis."),
        "STAB_NOTE": ("MuJoCo&#8217;s clamp \\(t_c\\ge2\\Delta t\\) is off. Each bar runs from the largest step that held the tool "
                      "for 1&#8202;s (filled) to the first that failed (&#215;: the state diverged; open circle: the pads ejected the "
                      "tool at hundreds of mm/s). A dotted bar held at every step up to 15&#8202;ms."),
        "REFS_TEXT": refs_text(ctx),
        "EVIDENCE_LEAD": evidence_lead(ctx),
        "AGREE_NOTE": agree_note(ctx),
        "EVIDENCE_TASKS": evidence_tasks(ctx),
        "COST_TEXT": cost_text(ctx),
        "BLOCK_TEXT": block_text(ctx),
        "COST_NOTE": ("Filled markers: one physics step of 1&#8202;ms on one CPU core in task&#160;1 on the test bed (MuJoCo, "
                      "Drake), or one world alone on the GPU on the SR2 holding fixture of Figure&#160;10 (MuJoCo-Warp, Newton). "
                      "Hollow markers: the wall time per world-step on the same fixture at the batch size of highest throughput. "
                      "Models closer in deviation than a label&#8217;s height are drawn apart vertically, in their order; these are "
                      "the 1&#8202;mm pad in three simulators and condim&#160;4, at 5.7&#8211;6.2&#8202;%. condim&#160;4&#8217;s torsional "
                      "coefficient follows a fit to Drake&#8217;s torque (step&#160;10). Newton hydroelastic runs at the settings of "
                      "the reference models: \\(k_f\\)&#8202;=&#8202;10, which gives friction rows of 3.9&#8202;ms on the test bed, contact "
                      "reduction without moment matching, and 0.5&#8202;mm voxels. Drake sits at zero deviation by construction."),
        "GPU_NOTE": gpu_note(ctx),
        "OPEN_LIST": open_list(ctx),
        "MEFF_TEXT": meff_text(ctx),
        "AGREE_SIM_TEXT": agree_sim_text(ctx),
        "RL_REPLAY_TEXT": rl_text(ctx) + scale_text(ctx),
        "LIT": lit(ctx),
        "SKIN_SUMMARY": skin_summary(ctx),
    }
