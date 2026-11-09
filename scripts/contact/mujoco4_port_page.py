#!/usr/bin/env python3
"""Builds docs/experiments/20261108-mujoco4_port/20261108-mujoco4_contact_and_stack.html: MuJoCo 3.6.0 against 4.0.0
for the sphere-pad contact model and the repository's simulation components (owner, 2026-11-08: "i am more interested in
the performance of underlying contact simulation, whether sphere packing got any faster/better ... or if there any
techniques we can take from the new release"; "a table showing function, age/last use, and compatibility with 4.0").

Data: the rows of scripts/contact/mujoco4_port.py under the folder's data/ (roll, rig_*, grasp_*, hand_gpu_*, compat_*,
press_*) and the HOM rows under logs/20261108-mujoco4_port/hom_<version>/ (scripts/hom/hom_gpu.py run).

    .venv/bin/python scripts/contact/mujoco4_port_page.py figs   # the figures
    .venv/bin/python scripts/contact/mujoco4_port_page.py        # the page
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path
import docs_index  # noqa: E402
import fp_review_page as P  # noqa: E402
import retro_style  # noqa: E402
import texsvg  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
table, img, film, esc = P.table, P.img, P.film, P.esc
D = ROOT / "docs/experiments/20261108-mujoco4_port"
DATA = D / "data"
MEDIA = D / "media"
LOGS = ROOT / "logs/20261108-mujoco4_port"
OUT = D / "20261108-mujoco4_contact_and_stack.html"
TEMPLATE = ROOT / "scripts/contact/mujoco4_port_page.template.html"
TEX_CACHE = D / "texsvg_cache.json"
P.TEX_CACHE = TEX_CACHE
VERS = ("3.6.0", "4.0.0")
TASKS = ("twist", "roll", "press", "pinpush")


# ------------------------------------------------------------------------------------------------ data

def rows(name):
    out = []
    for v in VERS:
        p = DATA / name.format(v=v)
        if p.exists():
            out += [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
    return out


def pick(rs, **kw):
    return [r for r in rs if all(r.get(k) == v for k, v in kw.items())]


def one(rs, **kw):
    x = pick(rs, **kw)
    return x[-1] if x else None


def hom_rows(v, task):
    tag = {"3.6.0": "360", "4.0.0": "400"}.get(v, v)
    p = LOGS / f"hom_{tag}" / f"{task}.jsonl"
    t = LOGS / f"hom_{tag}" / f"{task}_timing.jsonl"
    rs = [json.loads(s) for s in p.read_text().splitlines() if s.strip()] if p.exists() else []
    ts = [json.loads(s) for s in t.read_text().splitlines() if s.strip()] if t.exists() else []
    return rs, ts


def load():
    R = {"roll": [json.loads(s) for s in (DATA / "roll.jsonl").read_text().splitlines() if s.strip()],
         "pull": rows("rig_pull_{v}.jsonl"), "twist": rows("rig_twist_{v}.jsonl"), "rgpu": rows("rig_gpu_{v}.jsonl"),
         "grasp": rows("grasp_{v}.jsonl"), "hgpu": rows("hand_gpu_{v}.jsonl"), "compat": rows("compat_{v}.jsonl"),
         "press": {v: json.loads((DATA / f"press_rodv3id3_{v}.json").read_text()) for v in VERS
                   if (DATA / f"press_rodv3id3_{v}.json").exists()},
         "hom": {v: {t: hom_rows(v, t) for t in TASKS} for v in VERS + ("360b",)}}
    return R


def git_first(path):
    out = subprocess.run(["git", "log", "--follow", "--format=%ad", "--date=short", "--", path], cwd=ROOT,
                         capture_output=True, text=True).stdout.split()
    return out[-1] if out else "-"


def f(v, k=1, dash="-"):
    return dash if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{k}f}"


def pct(a, b):
    return 100.0 * (a / b - 1.0)


# ------------------------------------------------------------------------------------------------ numbers

def numbers(R):
    N = {}
    roll = {(r["mujoco"], r["case"]): r for r in R["roll"]}
    N["roll"] = roll
    o40 = roll[("4.0.0", "offset")]["contacts_50ms"]
    N["e40_mm"] = float(np.mean([abs(c["offset_mm"]) for c in o40]))
    g, r_ = 9.81, 0.0125
    N["alpha"] = g * N["e40_mm"] * 1e-3 / (1.5 * r_ ** 2)
    N["roll_pred_deg"] = np.degrees(0.5 * N["alpha"])
    pull, tw = R["pull"], R["twist"]
    N["rig"] = {}
    for v in VERS:
        for var in ("implicitfast", "noslip10", "discrete", "diagexact"):
            for dt in (1.0, 5.0):
                for n_ in (0.5, 1.0, 3.0):
                    p = one(pull, mujoco=v, variant=var, dt_ms=dt, N=n_)
                    t = one(tw, mujoco=v, variant=var, dt_ms=dt, N=n_)
                    if p:
                        N["rig"][(v, var, dt, n_)] = {"mu": p.get("mu_eff"), "creep": 1e3 * p["creep_mm_s"],
                                                      "us": p.get("us_per_step_median"),
                                                      "arm": t.get("rbar_onset_mm") if t else None,
                                                      "arm_kin": t.get("rbar_kin_mm") if t else None}
    N["rgpu"] = {(r["mujoco"], r["variant"], r["dt_ms"], r.get("nworld")): r for r in R["rgpu"]}
    N["hgpu"] = {(r["mujoco"], r["variant"], r.get("nworld")): r for r in R["hgpu"]}
    N["grasp"] = {(r["mujoco"], r["integrator"], r["sim"], r["ir"], r["noslip"]): r for r in R["grasp"]}
    return N


def gws(N, v, var, nw, kind="hgpu"):
    r = N[kind].get((v, var, nw)) if kind == "hgpu" else N[kind].get((v, var, 1.0, nw))
    return r["world_steps_per_s"] if r and r.get("status") == "ok" else None


def noslip_cost(N):
    """Cost of a noslip world-step over a plain one on the hand, at the smallest and the largest batch."""
    out = []
    for nw in (256, 2048):
        a, b = N["hgpu"].get(("4.0.0", "implicitfast", nw)), N["hgpu"].get(("4.0.0", "noslip10", nw))
        out.append(b["us_per_world_step"] / a["us_per_world_step"])
    return out


def gslip(N, v, integ, sim, ir, ns):
    r = N["grasp"].get((v, integ, sim, ir, ns))
    if not r or r.get("status") == "unsupported":
        return None
    return r["slip_um_s"]["median"]


# ------------------------------------------------------------------------------------------------ figures

def plt_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 14, "axes.titlesize": 14, "axes.labelsize": 14,
                         "legend.fontsize": 12, "xtick.labelsize": 13, "ytick.labelsize": 13, "mathtext.fontset": "cm"})
    return plt


def save(fig, name):
    import matplotlib.pyplot as plt
    MEDIA.mkdir(parents=True, exist_ok=True)
    fig.savefig(MEDIA / name, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", (MEDIA / name).relative_to(ROOT), flush=True)


STYLE = {("3.6.0", "implicitfast", 0): ("3.6.0", "#555555", "o"),
         ("4.0.0", "implicitfast", 0): ("4.0.0", "#1f5fa8", "s"),
         ("4.0.0", "discrete", 0): ("4.0.0, discrete", "#2e8b3a", "D"),
         ("4.0.0", "implicitfast", 10): ("4.0.0, noslip", "#c4561b", "^"),
         ("3.6.0", "implicitfast", 10): ("3.6.0, noslip", "#999999", "v")}


def fig_tradeoff(R, N):
    """Slip of the tool in the hand's scripted grasp against the cost of a step, CPU and GPU."""
    plt = plt_setup()
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.6))
    var_of = {"implicitfast": "implicitfast", "discrete": "discrete"}
    for ax, sim in zip(axs, ("cpu", "mjw")):
        for (v, integ, ns), (lab, col, mk) in STYLE.items():
            for ir, fill in ((10.0, True), (100.0, False)):
                s = gslip(N, v, integ, sim, ir, ns)
                if s is None:
                    continue
                if sim == "cpu":
                    c = N["grasp"][(v, integ, sim, ir, ns)]["us_per_step_median"]
                else:
                    c = N["hgpu"].get((v, "noslip10" if ns else var_of[integ], 256))
                    c = c["us_per_world_step"] if c and c.get("status") == "ok" else None
                if c is None:
                    continue
                ax.scatter([c], [s], s=80, marker=mk, color=col if fill else "white", edgecolors=col, linewidths=1.8,
                           zorder=3, label=f"{lab}, impratio {ir:g}")
        if sim == "mjw":
            from matplotlib.ticker import FixedLocator, NullFormatter
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(FixedLocator([5, 10, 20, 50, 100, 200]))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_xticklabels(["5", "10", "20", "50", "100", "200"])
            ax.set_xlim(6, 250)
        else:
            ax.set_xlim(0, None)
        ax.set_xlabel("CPU step, µs" if sim == "cpu" else "GPU world-step at 256 worlds, µs")
        ax.set_ylabel("Slip of the held tool, µm/s")
        ax.set_title("CPU MuJoCo, 16 placements" if sim == "cpu" else "MuJoCo Warp, 256 worlds")
        ax.set_ylim(0, None)
        ax.grid(alpha=0.3, which="both")
    h, lab = axs[0].get_legend_handles_labels()
    h2, lab2 = axs[1].get_legend_handles_labels()
    seen = dict(zip(lab, h))
    seen.update(dict(zip(lab2, h2)))
    fig.legend(list(seen.values()), list(seen.keys()), loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout()
    save(fig, "20261108-grasp_slip_against_cost.png")


def fig_throughput(R, N):
    """World-steps per second against the number of worlds, pinch rig (1 ms) and hand (2 ms), CUDA graphs."""
    plt = plt_setup()
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.4))
    series = [("3.6.0", "implicitfast"), ("4.0.0", "implicitfast"), ("4.0.0", "discrete"), ("4.0.0", "noslip10")]
    lab = {"implicitfast": "", "discrete": ", discrete", "noslip10": ", noslip"}
    for ax, kind, nws, title in ((axs[0], "rgpu", (1024, 4096), "Two-pad pinch rig, 1 ms step"),
                                 (axs[1], "hgpu", (256, 1024, 2048), "Hand pad scene, 2 ms step")):
        for v, var in series:
            st = STYLE[(v, "discrete" if var == "discrete" else "implicitfast", 10 if var == "noslip10" else 0)]
            xs, ys = [], []
            for nw in nws:
                r = N[kind].get((v, var, 1.0, nw)) if kind == "rgpu" else N[kind].get((v, var, nw))
                if r and r.get("status") == "ok" and (kind == "rgpu" or r.get("nefc_max", 0) <= 2048):
                    xs.append(nw)
                    ys.append(r["world_steps_per_s"] / 1e3)
            if xs:
                ax.plot(xs, ys, marker=st[2], color=st[1], lw=1.8, ms=8, label=f"{v}{lab[var]}")
        ax.set_xscale("log", base=2)
        ax.set_xticks(nws)
        ax.set_xticklabels([str(x) for x in nws])
        ax.set_xlabel("Worlds")
        ax.set_ylabel("Thousand world-steps per second")
        ax.set_title(title)
        ax.set_ylim(0, None)
        ax.grid(alpha=0.3)
        ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "20261108-gpu_throughput.png")


def cmd_figs():
    R = load()
    N = numbers(R)
    fig_tradeoff(R, N)
    fig_throughput(R, N)
    return 0


# ------------------------------------------------------------------------------------------------ sections

def sec_terms():
    terms = [
        ("Sphere pads", "The fingertip surface as spheres of radius 0.75 mm on a 1 mm grid. Each sphere is a soft "
         "contact with stiffness $K_i = E\\,a_i / h_i$. Here $E$ is the pad modulus, 2.5 MPa, fitted to the UR5 presses "
         "of 2026-10-30. $a_i$ is the surface area per sphere and $h_i$ its depth below the tip's volume centroid "
         "(<code>scripts/contact/reorient_backends.py</code>, builder v2)."),
        ("Two-pad pinch rig", "Two fingertip spheres (radius 10.55 mm, 20 g) on force-controlled rails squeeze the "
         "screwdriver with a force $N$ per pad. The screwdriver is a cylinder of radius 12.5 mm, 100 mm long and 24.5 g; "
         "friction is 1 and gravity is off (<code>scripts/hom/hom_contact_rig.py</code>, protocol in "
         "<code>docs/experiments/20261005-contact_bed/PROTOCOL.md</code>). Its pads here are 1 mm spheres at 2.5 MPa "
         "with a relaxation time of 0.04 s, at impratio 100."),
        ("Slip friction", "The axial force at which the pinched tool starts to slide, ramped at 2 N/s, divided by "
         "$2N$. For Coulomb friction of 1 it is 1."),
        ("Creep", "The steady sliding speed of the tool under an axial load held below the slip force, in µm/s. It is "
         "the slope of the displacement over the last 0.8 s of a 1 s hold at half the onset force. With rigid Coulomb "
         "friction it is 0. In MuJoCo's soft contact the creep is about $v = F_t / (2\\,\\kappa\\,t_c \\sum_i K_i)$, with $F_t$ "
         "the load, $\\kappa$ the impratio and $t_c$ the contact time constant."),
        ("Friction-torque arm", "The torque about the pinch axis at which the tool starts to spin, divided by "
         "$2\\mu N$ (mm): the pressure-weighted radius of the contact patch."),
        ("Scripted grasp", "The servo plant's pad scene of the RL runs: hand D6 with 3,177 pad spheres, rescaled to "
         "2.5 MPa. The fingers close on the screwdriver over 0.16 s, and the palm rises 0.1 m from 0.48 s; a run lasts "
         "2 s (<code>scripts/ur5/ur5_shear_impratio.py mjw</code>). Numerics of the policy trainer: 2 ms step, implicitfast, "
         "elliptic cone, Newton solver with 10 iterations. 16 placements on the CPU, 256 worlds over 64 placements on "
         "the GPU."),
        ("Held; slip", "Held: at 2 s the tool has risen with the palm and two fingers touch it. Slip: the tool's speed "
         "in the fingertips' frames from 1 to 2 s, median over the held placements, in µm/s."),
        ("Impratio", "MuJoCo's ratio of the friction rows' impedance to the normal rows'. With a larger impratio the "
         "friction is stiffer and the creep lower."),
        ("Noslip", "A post-pass after the constraint solver that removes slip in the friction directions "
         "(<code>noslip_iterations</code> = 10 here). CPU MuJoCo has it in both versions; MuJoCo Warp has it from 4.0.0."),
        ("Implicitfast, discrete", "MuJoCo integrators. Implicitfast is the repository's. Discrete (release 3.13) "
         "evaluates the contact spring-dampers at the end of the step, which keeps them stable at any time constant."),
        ("diagexact", "An option (release 3.9) that puts the exact diagonal of the constraint-space inertia into the "
         "solver's regulariser in place of the approximation made at compile time."),
        ("ccd_tolerance; condim", "ccd_tolerance: convergence tolerance of the convex collision detector, default "
         "1e-6. condim 3: sliding friction only; condim 6 adds torsional and rolling friction."),
        ("World-step; throughput", "One step of one world in MuJoCo Warp. Throughput: world-steps per second, the "
         "median over 20 to 100 launches of a 10-step CUDA graph."),
        ("CPU step", "The median wall time of one <code>mj_step</code> on one core of this workstation, in µs."),
    ]
    return "<dl class='terms'>" + "".join(f"<dt>{esc(a)}</dt><dd>{b}</dd>" for a, b in terms) + "</dl>"


def sec_setup(R):
    env = [["3.6.0", "<code>.venv</code>", "3.6.0", "3.6.0 (<code>external/mujoco_warp</code>)",
            "warp-lang 1.12.1, torch 2.11.0 (CUDA 12.8), mjlab 1.2.0, mink 1.1.0"],
           ["4.0.0", "<code>logs/20261108-mujoco4_port/venv</code>", "4.0.0", "4.0.0 (PyPI)",
            "warp-lang 1.18.0, torch 2.11.0 (CUDA 13.0), mink 1.4.0"]]
    tests = [["Rolling cylinder", "A cylinder at rest on a box, the cad_v3 screwdriver and mat, 1 s", "roll"],
             ["Pull and twist", "Slip friction, creep and friction-torque arm on the two-pad pinch rig, CPU", "rig"],
             ["Rig on the GPU", "Throughput and creep of the rig in MuJoCo Warp", "gpu"],
             ["UR5 presses", "Pad force at 24 rod presses of the v3 tip ID 3 (2026-10-29)",
              "<code>ur5_tip_calibration_pads.py</code>"],
             ["Scripted grasp", "Held and slip in the hand, CPU and GPU", "grasp"],
             ["Hand throughput", "World-steps per second of the hand's pad scene", "hand-gpu"],
             ["HOM", "Scores of four HOM tasks on 32 layouts each", "<code>hom_gpu.py run</code>"],
             ["Components", "Builds, imports and short runs of the simulation code", "compat"]]
    return (f"<p>Every test ran once in each environment of Table 1, on this workstation: one CPU core per job and "
            f"the 16 GB GPU. Table 2 lists the tests. The commands are subcommands of "
            f"<code>scripts/contact/mujoco4_port.py</code> unless the table names another script.</p>"
            + table(["Version", "Environment", "MuJoCo", "MuJoCo Warp", "Other packages"], env,
                    "The two environments.", num_from=9)
            + table(["Test", "What is measured", "Command"], tests, "The tests.", num_from=9))


def sec_roll(R, N):
    ro = N["roll"]
    r36, r40 = ro[("3.6.0", "offset")], ro[("4.0.0", "offset")]
    rows_ = []
    spec = [("Mat centred under the cylinder", "scene", "scene"),
            ("Mat centre 30 and 70 mm away (the scene)", "offset", "offset"),
            ("The same, elliptic cone", "offset_elliptic", "offset_elliptic"),
            ("The same, ccd_tolerance 1e-10", "offset_ccdtol1e-10", "offset_ccdtol1e-10"),
            ("The same, condim 6 on the screwdriver", "offset_condim6", "offset_condim6"),
            ("The same, multiccd switched (on in 3.6.0, off in 4.0.0)", "offset_multiccd_on", "offset_multiccd_off"),
            ("Mat centred, cylinder turned 15° about its axis", "spin15", "spin15")]
    for lab, k36, k40 in spec:
        a, b = ro.get(("3.6.0", k36)), ro.get(("4.0.0", k40))
        rows_.append([lab, f(abs(a["roll_deg_1s"]), 1), f(a["travel_mm_1s"], 1), str(a["ncon_1s"]),
                      f(abs(b["roll_deg_1s"]), 1), f(b["travel_mm_1s"], 1), str(b["ncon_1s"])])
    tab = table(["Cylinder on the box", "3.6.0 roll, deg", "3.6.0 travel, mm", "3.6.0 contacts", "4.0.0 roll, deg",
                 "4.0.0 travel, mm", "4.0.0 contacts"], rows_,
                "Roll and travel of the cylinder 1 s after release, and its contacts at 1 s. The cylinder and the box "
                "have the cad_v3 scene's sizes and contact parameters.")
    fm = film(MEDIA / "20261108-cad_v3_screwdriver_roll.mp4",
              "The cad_v3 scene from its first keyframe for 3 s, looking along the screwdriver's axis. A massless dark "
              "stripe along the screwdriver shows its roll. Left to right: MuJoCo 3.6.0, 4.0.0, and 4.0.0 with "
              "ccd_tolerance 1e-10.").replace("<figure>", "<figure class='wide'>")
    return (f"<p>In the cad_v3 scene the screwdriver rolls along its mat with nothing touching it, on both versions "
            f"(Film 1). In the first second it rolls {abs(r36['roll_deg_1s']):.0f}° and moves "
            f"{r36['travel_mm_1s']:.1f} mm on 3.6.0. On 4.0.0 it rolls {abs(r40['roll_deg_1s']):.0f}° and moves "
            f"{r40['travel_mm_1s']:.1f} mm. The mat is level and the screwdriver starts at rest, with its centre of mass "
            f"on its axis.</p>"
            + fm
            + f"<p>A cylinder and a box alone reproduce these numbers (Table 3). The roll appears only when the box's "
              f"centre lies away from the cylinder. With the scene's offset of 30 and 70 mm the cylinder rolls; centred over the box "
              f"it stays still. Its contact points are found by the convex collision detector, which stops at "
              f"ccd_tolerance = 1e-6. On 4.0.0 the cylinder has two contact points, from the multicontact for "
              f"cylinders of release 3.13. Both lie {N['e40_mm']:.3f} mm to one side of the vertical plane through the "
              f"axis. The weight then turns the cylinder about its contact line with the arm $e$:</p>"
              f"$$\\alpha = \\frac{{g\\,e}}{{\\tfrac{{3}}{{2}}\\,r^{{2}}}}$$"
              f"<p>With $e$ = {N['e40_mm']:.3f} mm and $r$ = 12.5 mm the roll accelerates at {N['alpha']:.1f} rad/s², "
              f"which gives {N['roll_pred_deg']:.0f}° after 1 s; {abs(r40['roll_deg_1s']):.0f}° is measured. Condim 3 "
              f"has no rolling friction to resist it. On 3.6.0 the single contact lies closer to that plane, and the "
              f"roll grows more slowly.</p>"
              f"<p>A tolerance of 1e-10 stops the roll on both versions, as does rolling friction (condim 6, with the "
              f"scene's own coefficient 0.02). With multiccd off, 4.0.0 gives one contact and rolls "
              f"{abs(ro[('4.0.0', 'offset_multiccd_off')]['roll_deg_1s']):.0f}°, as 3.6.0 does. The error also depends on "
              f"how the cylinder is turned about its axis. Turned 15° over the centred box, the 3.6.0 cylinder is thrown off "
              f"the box: it has no contact at 50 ms and travels {ro[('3.6.0', 'spin15')]['travel_mm_1s']:.0f} mm in "
              f"1 s.</p>" + tab)


def sec_rig(R, N):
    G = N["rig"]
    b36, b40 = G[("3.6.0", "implicitfast", 1.0, 1.0)], G[("4.0.0", "implicitfast", 1.0, 1.0)]
    ns, ds = G[("4.0.0", "noslip10", 1.0, 1.0)], G[("4.0.0", "discrete", 1.0, 1.0)]
    d5, i5 = G[("4.0.0", "discrete", 5.0, 1.0)], G[("4.0.0", "implicitfast", 5.0, 1.0)]
    rows_ = []
    for v, var, lab in (("3.6.0", "implicitfast", "3.6.0"), ("4.0.0", "implicitfast", "4.0.0"),
                        ("3.6.0", "noslip10", "3.6.0, noslip"), ("4.0.0", "noslip10", "4.0.0, noslip"),
                        ("4.0.0", "discrete", "4.0.0, discrete"), ("4.0.0", "diagexact", "4.0.0, diagexact")):
        a, b = G.get((v, var, 1.0, 1.0)), G.get((v, var, 5.0, 1.0))
        rows_.append([lab, f(a["mu"], 3), f(a["creep"], 1), f(b["creep"], 1), f(a["arm"], 3), f(b["arm"], 3),
                      f(a["us"], 1)])
    tab = table(["Setting", "Slip friction", "Creep 1 ms, µm/s", "Creep 5 ms, µm/s", "Arm 1 ms, mm", "Arm 5 ms, mm",
                 "CPU step, µs"], rows_,
                "Sphere pads on the two-pad pinch rig at 1 N per pad. Slip friction and CPU step at the 1 ms step.")
    lo = {v: G[(v, "implicitfast", 1.0, 0.5)]["creep"] for v in VERS}
    hi = {v: G[(v, "implicitfast", 1.0, 3.0)]["creep"] for v in VERS}
    return (f"<p>At the same settings, 3.6.0 and 4.0.0 give the same sphere-pad contact on the pinch rig (Table 4). "
            f"At 1 N per pad and a 1 ms step the slip friction is {b40['mu']:.3f} on both. The creep is "
            f"{b40['creep']:.1f} µm/s and the friction-torque arm {b40['arm']:.3f} mm on both. At 0.5 and 3 N the creep "
            f"is {lo['4.0.0']:.1f} and {hi['4.0.0']:.1f} µm/s on both. The CPU step is {b40['us']:.1f} µs on 4.0.0 and "
            f"{b36['us']:.1f} µs on 3.6.0.</p>"
            f"<p>Noslip removes the creep: the tool drifts back by {abs(ns['creep']):.1f} µm/s. A CPU step costs "
            f"{ns['us'] / b40['us']:.1f} times as much. The arm at the onset of spin falls by "
            f"{-pct(ns['arm'], b40['arm']):.0f} %, to {ns['arm']:.3f} mm, and the sliding arm stays at "
            f"{ns['arm_kin']:.3f} mm. The discrete integrator lowers the creep by {-pct(ds['creep'], b40['creep']):.0f} % "
            f"at 1 ms and by {-pct(d5['creep'], i5['creep']):.0f} % at 5 ms. It lowers the arm by "
            f"{-pct(ds['arm'], b40['arm']):.0f} % and {-pct(d5['arm'], i5['arm']):.0f} %. With implicitfast the 1 and "
            f"5 ms results differ by under 1 %. diagexact changes nothing on this rig, where the pads move on rails and "
            f"the tool's inertia is symmetric.</p>" + tab)


def sec_press(R):
    a, b = R["press"]["3.6.0"]["delta_E_meas"]["mujoco"], R["press"]["4.0.0"]["delta_E_meas"]["mujoco"]
    return (f"<p>The pads' static force does not change with the version. At {a['n']} of the rod presses on the v3 "
            f"tip ID 3, the MuJoCo pad rig was stepped to rest at the measured indenter position. The pad law, summed "
            f"over the spheres, and MuJoCo differ by a median {a['law_minus_mujoco_um']['median'] * 1e3:.0f} nm of "
            f"indentation on 3.6.0 and {b['law_minus_mujoco_um']['median'] * 1e3:.0f} nm on 4.0.0. The force over the "
            f"HEX-E's force is {a['force_over_measured']['median']:.4f} on both, at the modulus fitted to these presses "
            f"(<code>scripts/ur5/ur5_tip_calibration_pads.py --mujoco-every 20</code>).</p>")


def sec_grasp(R, N):
    rows_ = []
    for v, integ in (("3.6.0", "implicitfast"), ("4.0.0", "implicitfast"), ("4.0.0", "discrete")):
        for ir in (10.0, 100.0):
            for ns in (0, 10):
                c = N["grasp"].get((v, integ, "cpu", ir, ns))
                g = N["grasp"].get((v, integ, "mjw", ir, ns))
                if not c:
                    continue
                gok = g and g.get("status") != "unsupported"
                rows_.append([v, integ, f"{ir:g}", "on" if ns else "off",
                              f"{c['held']} of {c['n']}" + (f" ({c['nonfinite']} diverged)" if c["nonfinite"] else ""),
                              f(c["slip_um_s"]["median"], 0), f(c["us_per_step_median"], 0),
                              f"{g['held']} of {g['n']}" if gok else "not supported",
                              f(g["slip_um_s"]["median"], 0) if gok else "-"])
    tab = table(["MuJoCo", "Integrator", "Impratio", "Noslip", "CPU held", "CPU slip, µm/s", "CPU step, µs",
                 "GPU held", "GPU slip, µm/s"], rows_,
                "The hand's scripted grasp with 2.5 MPa pads. CPU: 16 placements; GPU: 256 worlds over 64 placements.")
    s36, s40 = gslip(N, "3.6.0", "implicitfast", "mjw", 10.0, 0), gslip(N, "4.0.0", "implicitfast", "mjw", 10.0, 0)
    sns = gslip(N, "4.0.0", "implicitfast", "mjw", 10.0, 10)
    c36, c40 = N["grasp"][("3.6.0", "implicitfast", "cpu", 100.0, 0)], N["grasp"][("4.0.0", "implicitfast", "cpu", 100.0, 0)]
    cd = N["grasp"][("4.0.0", "discrete", "cpu", 100.0, 0)]
    cns = gslip(N, "4.0.0", "implicitfast", "cpu", 10.0, 10)
    r256, r2048 = noslip_cost(N)
    fig = img(MEDIA / "20261108-grasp_slip_against_cost.png",
              "Slip of the held tool in the hand's scripted grasp against the cost of a step. Filled markers: impratio "
              "10; open markers: impratio 100. GPU costs are the hand throughput test's at 256 worlds; impratio does "
              "not change them.")
    return (f"<p>In the hand, every grasp held on 3.6.0 at impratio 10 is held on 4.0.0 as well (Table 5). On the GPU the slip "
            f"is {s40:.0f} µm/s on 4.0.0 and {s36:.0f} µm/s on 3.6.0.</p>"
            f"<p>Noslip on the GPU is new in 4.0.0. At impratio 10 it lowers the slip from {s40:.0f} to {sns:.0f} µm/s, "
            f"with all 256 worlds held. A world-step with noslip costs {r256:.0f} times as much at 256 worlds and "
            f"{r2048:.0f} times at 2048 (Figure 1). On the CPU noslip "
            f"gives {cns:.0f} µm/s on both versions.</p>"
            f"<p>At impratio 100 the slip without noslip is 42 to 85 µm/s, but not every CPU grasp survives. On 3.6.0, "
            f"{c36['held']} of 16 hold and {c36['nonfinite']} diverge. On 4.0.0 with implicitfast, {c40['held']} hold "
            f"and none diverge: the gyroscopic term of 3.11 stops the spin runaway of 2026-10-27, but the flung tools "
            f"are still dropped. With the discrete integrator {cd['held']} of 16 hold. On the GPU every setting holds "
            f"all 256 worlds.</p>" + fig + tab)


def sec_gpu(R, N):
    r36, r40 = gws(N, "3.6.0", "implicitfast", 4096, "rgpu"), gws(N, "4.0.0", "implicitfast", 4096, "rgpu")
    q36, q40 = gws(N, "3.6.0", "implicitfast", 1024, "rgpu"), gws(N, "4.0.0", "implicitfast", 1024, "rgpu")
    h36, h40 = gws(N, "3.6.0", "implicitfast", 1024), gws(N, "4.0.0", "implicitfast", 1024)
    k36, k40 = gws(N, "3.6.0", "implicitfast", 256), gws(N, "4.0.0", "implicitfast", 256)
    rd = gws(N, "4.0.0", "discrete", 4096, "rgpu")
    rn = gws(N, "4.0.0", "noslip10", 4096, "rgpu")
    hn = gws(N, "4.0.0", "noslip10", 1024)
    g36 = N["grasp"].get(("3.6.0", "implicitfast", "mjw", 10.0, 0))
    g40 = N["grasp"].get(("4.0.0", "implicitfast", "mjw", 10.0, 0))
    fig = img(MEDIA / "20261108-gpu_throughput.png",
              "Throughput of MuJoCo Warp in 10-step CUDA graphs. Left: the pinch rig at 1 N, 1 ms step. Right: the "
              "hand's pad scene from its first keyframe, 2 ms step. The 3.6.0 hand at 2048 worlds is left out: one "
              "world exceeded the 2048 constraint rows allocated.")
    hn_txt = (f" On the hand at 1024 worlds noslip reaches {hn / 1e3:.0f} thousand, {h40 / hn:.0f} times fewer."
              if hn and h40 else "")
    return (f"<p>The sphere pads are stepped faster in MuJoCo Warp 4.0.0 than in 3.6.0 (Figure 2). On the rig the "
            f"throughput is {pct(r40, r36):.0f} % higher at 4096 worlds and {pct(q40, q36):.0f} % higher at 1024. At 4096 "
            f"worlds that is {r40 / 1e6:.2f} million world-steps per second against {r36 / 1e6:.2f} million. On the "
            f"hand it is {pct(h40, h36):.0f} % higher at 1024 worlds and {pct(k40, k36):.0f} % higher at 256. The "
            f"discrete integrator reaches {rd / 1e6:.2f} million on the rig, as implicitfast does. With noslip the rig "
            f"reaches {rn / 1e6:.2f} million.{hn_txt}</p>"
            f"<p>Stepped one call at a time from Python, as in the scripted grasp, 4.0.0 is the slower version. At 256 "
            f"worlds a world-step lasts {g40['us_per_world_step']:.0f} µs on 4.0.0 and {g36['us_per_world_step']:.0f} µs "
            f"on 3.6.0.</p>" + fig)


def sec_options(N):
    G = N["rig"]
    rows_ = [
        ["Noslip in MuJoCo Warp", "4.0.0", "Removes slip in the friction rows after the solve",
         f"Hand slip {gslip(N, '4.0.0', 'implicitfast', 'mjw', 10.0, 0):.0f} to "
         f"{gslip(N, '4.0.0', 'implicitfast', 'mjw', 10.0, 10):.0f} µm/s; rig creep "
         f"{G[('4.0.0', 'implicitfast', 1.0, 1.0)]['creep']:.0f} to 0 µm/s",
         "Yes, except with the discrete integrator"],
        ["Discrete integrator", "3.13", "Contact springs at the end of the step, stable at any time constant",
         "Rig creep 7 to 22 % lower, arm 3 to 8 % lower; CPU grasps at impratio 100 all held",
         "Yes, without noslip"],
        ["Gyroscopic free-body term in implicitfast", "3.11; MuJoCo Warp 4.0.0", "Removes the energy gain of a spinning "
         "free body", "No CPU grasp diverges at impratio 100 (3.6.0: 2 of 16)", "Yes"],
        ["diagexact", "3.9", "Exact constraint-space inertia diagonal in the regulariser",
         "No change on the rig (under 0.2 %)", "No: MuJoCo Warp raises NotImplementedError"],
        ["Multicontact for cylinders and capsules; multiccd on", "3.13, 3.15, 3.8",
         "Two or more contacts per convex pair", "The cad_v3 screwdriver rolls 90° in 1 s (Section SEC[roll])", "Yes"],
        ["mujoco.sysid", "4.0.0", "Least-squares fit of model parameters to logged trajectories over batched rollouts",
         "Not tested", "CPU"],
        ["Contact sensors with slip velocity", "3.16", "Linear and angular velocity at a contact", "Not tested",
         "Yes, not with flexes"],
        ["Flex elasticity with the discrete integrator on the GPU", "MuJoCo Warp 4.0.0",
         "Stable volumetric soft fingertips on the GPU, absent in MuJoCo Warp on 2026-10-08", "Not tested", "Yes"],
    ]
    return ("<p>Two of the new options change the pads' creep or stability. Noslip on the GPU gives the lowest slip "
            "measured in the hand, at the highest cost per step. The discrete integrator costs nothing per step and "
            "keeps the high-impratio grasps stable on the CPU. It cannot be combined with noslip in MuJoCo Warp, and at "
            "5 ms it changes the friction-torque arm by 8 %. Table 6 lists the options with what was measured.</p>"
            + table(["Option", "Release", "What it changes", "Measured on the pads", "In MuJoCo Warp"], rows_,
                    "Options of MuJoCo 3.7 to 4.0 for the sphere pads.", num_from=9))


COMPONENTS = [
    # name, function, key path for the first commit, last run (dated output folder), environment now, on 4.0 (key or text)
    ("Hand and object scenes", "MJCF of the real_v1, cad_v2 and cad_v3 hands with tools; generators",
     "scripts/design/generate_morphology_xml.py", "2026-11-05", "<code>.venv</code>, 3.6.0", "scenes"),
    ("Sphere-pad builder", "Fingertips as 1 mm sphere pads, stiffness $E\\,a_i/h_i$",
     "scripts/contact/reorient_backends.py", "2026-11-03", "<code>.venv</code>, 3.6.0", "pads"),
    ("Two-pad pinch rig", "Pull, twist, creep and cost of contact models",
     "scripts/hom/hom_contact_rig.py", "2026-10-12", "<code>logs/20261001-hom_contact/venv</code>, 3.6.0",
     "Same slip friction, creep and arm (Section SEC[rig])"),
    ("UR5 press simulation", "Pad forces at the UR5 presses; the modulus fit",
     "scripts/ur5/ur5_tip_calibration_pads.py", "2026-11-03", "<code>.venv</code>, 3.6.0",
     "Same forces to 1e-9 (Section SEC[rig])"),
    ("Scripted grasp and impratio tests", "Hand grasps on CPU and GPU at the trainer's numerics",
     "scripts/ur5/ur5_shear_impratio.py", "2026-10-30", "<code>.venv</code>, 3.6.0",
     "Runs; noslip on the GPU; impratio 100 drops 7 of 16 CPU grasps (Section SEC[grasp])"),
    ("HOM on MuJoCo Warp", "HOM task scores over hand layouts in GPU worlds", "scripts/hom/hom_gpu.py", "2026-11-06",
     "<code>.venv</code>, 3.6.0", "hom"),
    ("HOM on the CPU", "HOM tasks and controller in CPU MuJoCo", "scripts/hom/hom_tasks.py", "2026-11-04",
     "<code>.venv</code>, 3.6.0", "Not tested"),
    ("CEM grasp synthesis", "Grasp and morphology search, MuJoCo, MuJoCo Warp or ComFree backends",
     "scripts/design/phase1_optimize_grasp.py", "2026-10-11", "<code>.venv</code>, 3.6.0", "Not tested"),
    ("Trajectory clearance gate", "Finger clearance along deploy plans before a bench session",
     "scripts/hardware/real_v1_trajectory_clearance.py", "2026-09-06", "<code>.venv</code>, 3.6.0", "clearance"),
    ("Films and filmstrips", "Rendered films of rollouts (EGL)", "scripts/rl/policy_filmstrip.py", "2026-11-08",
     "<code>.venv</code>, 3.6.0", "render"),
    ("RL trainer", "PPO lift and reorient policies on mjlab 1.2.0 and MuJoCo Warp", "scripts/rl/rl_train_cube.py",
     "2026-10-16", "<code>.venv</code>, 3.6.0",
     "Blocked: mjlab 1.2.0 needs warp-lang below 1.13, MuJoCo Warp 4.0.0 needs 1.15; mjlab 1.6.0 pins MuJoCo 3.11, "
     "its main branch 3.16"),
    ("MuJoCo Warp, direct calls", "GPU stepping in the rig, grasp and HOM scripts", "external/mujoco_warp",
     "2026-11-06", "<code>.venv</code>, 3.6.0", "mjw"),
    ("DiffMJX", "Gradients through MJX for grasp optimisation", "src/sr2hand/optimization/phase1_strategy_diffmjx.py",
     "2026-10-11", "<code>logs/20261007-diffmjx/venv</code>, 3.3.1",
     "Not tested; unused since the audit of 2026-10-11. On 3.6.0 MJX lacks the hand's cylinder-box collisions"),
    ("ComFree Warp", "Complementarity-free contact on Warp, a CEM backend", "src/sr2hand/backends/comfree_backend.py",
     "No reported run", "<code>.venv</code>, 3.6.0 (pins it)", "comfree"),
    ("Newton hydroelastic comparisons", "Pinch and creep against Newton's hydroelastic contact",
     "scripts/contact/newton_pinch_probe.py", "2026-10-07",
     "<code>logs/20261004-contact-transfer/venv</code>, 3.14.0, Newton 1.7.0.dev0",
     "Blocked: Newton 1.7.0 pins MuJoCo Warp 3.15, its main branch 3.17"),
    ("Flex fingertip probes", "Volumetric flexcomp fingertips as a native soft contact",
     "scripts/contact/flexcomp_skin_probe.py", "2026-10-08", "<code>logs/20261004-contact-current/venv</code>, 3.14.0",
     "Not tested; MuJoCo Warp 4.0.0 has the discrete integrator the GPU flex lacked"),
    ("Drake baselines", "Hydroelastic reference models and the SR2 port in Drake", "scripts/contact/contact_bed_common.py",
     "2026-10-18", "<code>logs/20261001-drake-port/venv</code>, 3.6.0, Drake 1.57", "Not affected: no MuJoCo dependency"),
    ("mink arm IK", "Differential IK of the UR5 for the arm-mounted hand", "src/sr2hand/tools/arm_ik.py",
     "2026-09-19", "<code>.venv</code>, 3.6.0, mink 1.1.0", "mink"),
    ("GraspGenX", "Learned grasp generation on the hand morphologies", "scripts/design/graspgenx_eval_phase1.py",
     "Summer 2026", "Its own environment, MuJoCo 3.5.0 and MuJoCo Warp 3.5.0.2", "Not tested; pinned to 3.5.0"),
]


def compat_text(R, key, hom_txt):
    c = {(r["mujoco"], r["check"]): r for r in R["compat"]}
    if key == "hom":
        return hom_txt
    if (("4.0.0", key) not in c):
        return key
    a, b = c.get(("3.6.0", key)), c[("4.0.0", key)]
    if b["status"] == "runs":
        extra = {"scenes": "; contacts change (Section SEC[roll])", "pads": "; forces the same (Section SEC[rig])",
                 "clearance": ", same verdict on the test plan", "render": "", "mjw": "", "comfree": "",
                 "mink": " (mink 1.4.0)"}.get(key, "")
        return "Runs" + extra
    return f"{b['status'].capitalize()}: {esc(b['detail'][:110])}"


def hom_summary(R):
    out, rows_ = {}, []
    for t in TASKS:
        a, ta = R["hom"]["3.6.0"][t]
        b, tb = R["hom"]["4.0.0"][t]
        if not a or not b:
            continue
        import hom_landscape as HL
        key = lambda r: json.dumps(r.get("d"), sort_keys=True) + str(r.get("seed"))  # noqa: E731
        sa = {key(r): HL.rollout_score(t, r) for r in a if r.get("status") == "ok"}
        sb = {key(r): HL.rollout_score(t, r) for r in b if r.get("status") == "ok"}
        ks = [k for k in sa if k in sb and sa[k] is not None and sb[k] is not None]
        if not ks:
            continue
        x, y = np.array([sa[k] for k in ks]), np.array([sb[k] for k in ks])
        from scipy.stats import spearmanr
        rho = spearmanr(x, y).correlation if np.ptp(x) > 0 and np.ptp(y) > 0 else float("nan")
        c, _ = R["hom"]["360b"][t]
        sc_ = {key(r): HL.rollout_score(t, r) for r in c if r.get("status") == "ok"}
        kc = [k for k in ks if k in sc_]
        rho_b = (spearmanr(x[[ks.index(k) for k in kc]], [sc_[k] for k in kc]).correlation if kc else None)
        held = [sum(bool(r.get("held")) for r in rr if r.get("status") == "ok") for rr in (a, b)]
        wa = sum(r.get("run_s", 0) for r in ta)
        wb = sum(r.get("run_s", 0) for r in tb)
        rows_.append([t, str(len(ks)), f"{held[0]}, {held[1]}", f(x.mean(), 3), f(y.mean(), 3), f(np.abs(x - y).max(), 3),
                      f(rho, 2), f(rho_b, 2) if rho_b is not None else "-", f(wa, 0) if wa else "-", f(wb, 0) if wb else "-"])
        out[t] = (rho, rho_b, held)
    return rows_, out


def sec_components(R):
    hrows, rho = hom_summary(R)
    hom_txt = ("Runs after the fix of this page; press and pin-push scores change (Table 8)" if hrows else
               "Not finished")
    rows_ = []
    for name, fn, path, last, env, key in COMPONENTS:
        on4 = compat_text(R, key, hom_txt)
        rows_.append([esc(name), fn, git_first(path), last, env, on4])
    n_run = sum(r[5].startswith("Runs") or r[5].startswith("Same") for r in rows_)
    n_blk = sum(r[5].startswith("Blocked") for r in rows_)
    txt = (f"<p>Of the {len(rows_)} simulation components in Table 7, {n_run} run on MuJoCo 4.0.0. ComFree Warp fails "
           f"there: it sets the enable flag of multiccd, which 4.0 removed. {n_blk} components are blocked by the version "
           f"pins of the packages they are built on: the RL trainer by mjlab, and the Newton comparisons by Newton. The "
           f"others were not tested or do not depend on MuJoCo. Today the repository's "
           f"simulation runs on MuJoCo 3.3.1, 3.5.0, 3.6.0 and 3.14.0, in six environments. Added is the date of the "
           f"key file's first commit; last run is the date of the newest output folder or page of the component.</p>")
    tab = table(["Component", "Function", "Added", "Last run", "Environment, MuJoCo", "On MuJoCo 4.0.0"], rows_,
                "Simulation components of the repository and their state on MuJoCo 4.0.0.", num_from=9)
    hom = ""
    if hrows:
        tw, ro_, pr, pp = (rho[t] for t in TASKS)
        hom = (f"<p>HOM's GPU runner had failed at its first measurement since the shared-math commit 83aeae6c of "
               f"2026-11-03. That commit removed the <code>@staticmethod</code> decorators of <code>Batch._quat_mat</code> "
               f"and <code>Batch._quat_mul</code> in <code>scripts/hom/hom_gpu.py</code>; they are restored. With the fix "
               f"it runs on MuJoCo Warp 4.0.0 without changes (Table 8). Twist and roll keep their scores, with rank "
               f"correlations of {tw[0]:.2f} and {ro_[0]:.2f} between versions. Press and pin-push change: "
               f"{pr[0]:.2f} and {pp[0]:.2f} between versions, against {pr[1]:.2f} and {pp[1]:.2f} between two runs on "
               f"3.6.0. In the press task {pr[2][1]} of 32 rollouts hold on 4.0.0 and {pr[2][0]} on 3.6.0.</p>"
               + table(["Task", "Rollouts", "Held, 3.6.0 and 4.0.0", "Mean score 3.6.0", "Mean score 4.0.0",
                        "Largest difference", "Rank correlation, versions", "Rank correlation, two 3.6.0 runs",
                        "Rollout time 3.6.0, s", "Rollout time 4.0.0, s"], hrows,
                       "HOM task scores of 32 Sobol layouts per task (seed 0) on the cad_v2 hand with the v4b tip, at the "
                       "toy's object sizes and the default plant. A rollout's score lies in [0, 1] and is 0 when the "
                       "object drops (scripts/hom/hom_landscape.py)."))
    return txt + tab + hom


def sec_limits():
    return ("<p>Each setting ran on one scene. The GPU timings are medians of 20 to 100 graph launches on the "
            "workstation's GPU, which the desktop shares. The grasp slips are medians over 16 CPU placements or 64 GPU "
            "placements. HOM ran 32 layouts per task with one seed. The 4.0.0 environment has warp-lang 1.18.0 and "
            "torch's CUDA 13.0 build. The 3.6.0 environment has warp-lang 1.12.1 and CUDA 12.8.</p>")


def sec_future():
    items = [
        "Stop the roll in the cad_v3 scenes: set <code>ccd_tolerance</code> to 1e-10 in "
        "<code>assets/mjcf/experimental/20261104-cad_v3/</code> and rerun <code>mujoco4_port.py roll</code> and the "
        "film. The fix fails if the screwdriver rolls more than 1° in 3 s on either version.",
        "Noslip in HOM: run <code>hom_gpu.py run</code> on 4.0.0 with <code>noslip_iterations</code> 10 on the 30 best "
        "SCS0009 layouts. Keep it if the rank correlation with the implicitfast scores is above 0.9 and a batch takes "
        "less than three times as long.",
        "The discrete integrator for CPU planning at impratio 100: rerun "
        "<code>ur5_shear_impratio.py replays --ir 100</code> on 4.0.0 with the discrete integrator. Keep it if all 24 "
        "D1-D8 replays are held and the held turn differs by less than 3° from the implicitfast replays at impratio 10.",
        "Fit the pad modulus and relaxation time with <code>mujoco.sysid</code> to the v3 tips' presses and slides, and "
        "compare with the 2.47 MPa fit of <code>scripts/fingertip/v3_tip_contact_model.py</code>.",
        "Flex fingertip on the GPU: run the flex candidate of <code>contact_bed_candidates.py</code> in MuJoCo Warp "
        "4.0.0 with the discrete integrator. It replaces the pads only if its creep and friction-torque arm on the "
        "pinch rig are within 10 % of the pads' at under three times their cost per world-step.",
        "Move the contact work (pads, pinch rig, UR5 presses, HOM) to a MuJoCo 4.0.0 environment. Keep "
        "<code>.venv</code> on 3.6.0 for the RL trainer until an mjlab release supports MuJoCo 4.0.",
    ]
    return "<ol>" + "".join(f"<li>{x}</li>" for x in items) + "</ol>"


def sec_files():
    items = [
        "<code>scripts/contact/mujoco4_port.py</code>: every test (subcommands roll, rig, gpu, grasp, hand-gpu, film, "
        "film-tile, compat).",
        "<code>scripts/contact/mujoco4_port_page.py</code> and its template: this page.",
        "<code>docs/experiments/20261108-mujoco4_port/data/</code>: the rows, one file per test and version (on this "
        "workstation).",
        "<code>logs/20261108-mujoco4_port/</code>: the 4.0.0 environment (<code>venv</code>), the HOM rows and the logs.",
        "The 4.0.0 environment: <code>uv venv logs/20261108-mujoco4_port/venv --python 3.12</code>, then <code>uv pip "
        "install mujoco==4.0.0 mujoco-warp==4.0.0 warp-lang==1.18.0 torch==2.11.0 mink==1.4.0</code> and numpy, scipy, "
        "trimesh, matplotlib, imageio, imageio-ffmpeg, pillow, pyyaml, tyro, shapely.",
    ]
    return "<ul>" + "".join(f"<li>{x}</li>" for x in items) + "</ul>"


def lede_and_summary(R, N):
    G = N["rig"]
    s40 = gslip(N, "4.0.0", "implicitfast", "mjw", 10.0, 0)
    sns = gslip(N, "4.0.0", "implicitfast", "mjw", 10.0, 10)
    r256, r2048 = noslip_cost(N)
    gains = [pct(gws(N, "4.0.0", "implicitfast", nw, kd), gws(N, "3.6.0", "implicitfast", nw, kd))
             for kd, nw in (("rgpu", 1024), ("rgpu", 4096), ("hgpu", 256), ("hgpu", 1024))]
    ro = N["roll"]
    r36, r40 = abs(ro[("3.6.0", "offset")]["roll_deg_1s"]), abs(ro[("4.0.0", "offset")]["roll_deg_1s"])
    cpu = [pct(G[("4.0.0", "implicitfast", 1.0, n_)]["us"], G[("3.6.0", "implicitfast", 1.0, n_)]["us"])
           for n_ in (0.5, 1.0, 3.0)]
    cpu += [pct(N["grasp"][("4.0.0", "implicitfast", "cpu", ir, ns)]["us_per_step_median"],
                N["grasp"][("3.6.0", "implicitfast", "cpu", ir, ns)]["us_per_step_median"])
            for ir in (10.0, 100.0) for ns in (0, 10)]
    hrows, rho = hom_summary(R)
    hom_s = (f"Of HOM's four tasks, twist and roll keep their scores; press and pin-push change, with rank correlations "
             f"of {rho['press'][0]:.2f} and {rho['pinpush'][0]:.2f} between versions. " if rho else "")
    lede = (f"MuJoCo 4.0.0 gives the same sphere-pad contact as 3.6.0 at the same settings. On the two-pad pinch rig the "
            f"slip friction, the creep and the friction torque are equal to four digits. The pads are stepped {min(gains):.0f} to {max(gains):.0f} % "
            f"faster in MuJoCo Warp 4.0.0, and a CPU step costs {max(0, min(cpu)):.0f} to {max(cpu):.0f} % more. "
            f"MuJoCo Warp now has noslip: in the hand's lifted grasp it lowers the slip from {s40:.0f} to {sns:.0f} µm/s, "
            f"at {r2048:.0f} to {r256:.0f} times the cost per world-step. {hom_s}In the cad_v3 scene the screwdriver "
            f"rolls by itself on both versions, {r36:.0f}° in 1 s on 3.6.0 and {r40:.0f}° on 4.0.0, until the collision "
            f"tolerance is 1e-10.")
    summary = (f"MuJoCo 4.0.0 gives the same sphere-pad contact as 3.6.0; MuJoCo Warp 4.0.0 is {min(gains):.0f} to "
               f"{max(gains):.0f} % faster and has noslip, which lowers the hand's grasp slip from {s40:.0f} to "
               f"{sns:.0f} µm/s at {r2048:.0f} to {r256:.0f} times the step cost. The cad_v3 screwdriver rolls by itself ({r36:.0f}° and "
               f"{r40:.0f}° in 1 s) until ccd_tolerance is 1e-10.")
    return lede, summary


def render(page):
    items = [((m.group(1) if m.group(1) is not None else m.group(2)).strip(), m.group(1) is not None)
             for m in P.TEX_RE.finditer(page)]
    svgs = texsvg.render(items, cache_path=TEX_CACHE)
    it = iter(svgs)
    return P.TEX_RE.sub(lambda m: next(it), page), len(items)


def cmd_page():
    R = load()
    N = numbers(R)
    sections = [
        ("terms", "Terms and metrics", sec_terms()),
        ("setup", "Environments and tests", sec_setup(R)),
        ("roll", "Rolling of the cad_v3 screwdriver on its mat", sec_roll(R, N)),
        ("rig", "Sphere pads on the two-pad pinch rig", sec_rig(R, N) + sec_press(R)),
        ("grasp", "Sphere pads in the hand's scripted grasp", sec_grasp(R, N)),
        ("gpu", "Throughput of the sphere pads in MuJoCo Warp", sec_gpu(R, N)),
        ("options", "MuJoCo 4.0 options for the sphere pads", sec_options(N)),
        ("components", "Simulation components on MuJoCo 4.0", sec_components(R)),
        ("limits", "Limitations", sec_limits()),
        ("future", "Future work", sec_future()),
        ("files", "Files", sec_files()),
    ]
    num_ = {sid: k for k, (sid, _h, _b) in enumerate(sections, 1)}
    sub = (lambda s_: re.sub(r"SEC\[(\w+)\]", lambda m_: str(num_[m_.group(1)]), s_))
    unit = (lambda s_: re.sub(r"(\d) (%|mm|deg|kg|µm/s|µs|nm|px|V\b|N\b|ms\b|s\b|g\b|m\b|min\b|MPa\b)", "\\1&nbsp;\\2", s_))
    body = [(sid, hd, unit(sub(b))) for sid, hd, b in sections]
    toc = "\n".join(f'<li><a href="#{sid}">{k}. {esc(hd)}</a></li>' for k, (sid, hd, _) in enumerate(body, 1))
    secs = "\n".join(f'<section id="{sid}"><h2>{k}. {esc(hd)}</h2>{b}</section>' for k, (sid, hd, b) in enumerate(body, 1))
    lede, summary = lede_and_summary(R, N)
    byline = ("Measured on the workstation on 2026-11-08 in two environments, MuJoCo 3.6.0 and 4.0.0. Builder: "
              "<code>scripts/contact/mujoco4_port_page.py</code>.")
    page = TEMPLATE.read_text()
    for k_, v in {"TITLE": "MuJoCo 4.0 port",
                  "H1": "Sphere-pad contact and the simulation stack on MuJoCo 4.0",
                  "EYEBROW": "Contact model, 2026-11-08", "LEDE": unit(lede), "BYLINE": byline, "TOC": toc,
                  "SECTIONS": secs, "FOOTER": str(OUT.relative_to(ROOT)), "TEXCSS": texsvg.CSS}.items():
        page = page.replace("{{" + k_ + "}}", v)
    page, n = render(page)
    page = docs_index.tag(page, topic="contact", summary=summary, date="2026-11-08")
    page = retro_style.apply(page)
    OUT.write_text(page)
    print(f"{OUT}: {OUT.stat().st_size / 1e6:.2f} MB, {n} formulas", flush=True)
    return 0


def main():
    a = sys.argv[1:]
    if a and a[0] == "figs":
        return cmd_figs()
    return cmd_page()


if __name__ == "__main__":
    sys.exit(main())
