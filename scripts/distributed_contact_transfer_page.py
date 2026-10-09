#!/usr/bin/env python3
"""Build the articulated contact report and portable backend benchmark data."""

import csv
import gzip
import json
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from contact_surface.records import ROOT, write_json
from distributed_contact import DOC
from distributed_contact_page import frame, save_figure, section, table
import retro_style

RESULTS = ROOT / "results/20261004-distributed-contact"
PHASES = [
    "transfer",
    "transfer_optimized",
    "transfer_compiled_fine",
    "holding",
    "holding_refine",
    "friction_transfer",
    "friction_refine",
    "payload",
    "mjlab_transfer",
    "drake_transfer",
    "newton_transfer",
    "newton_pressure",
    "transfer_pinned",
]


def load():
    datasets = {}
    export = DOC / "data/articulated"
    export.mkdir(parents=True, exist_ok=True)
    for phase in PHASES:
        rows = []
        for path in sorted((RESULTS / phase).glob("*/summary.json")):
            row = dict(json.loads(path.read_text()), run_id=path.parent.name, phase=phase)
            rows.append(row)
            trace = path.parent / "trace.json"
            if trace.exists():
                with gzip.open(export / f"{phase}-{path.parent.name}-trace.json.gz", "wt") as f:
                    f.write(trace.read_text())
        datasets[phase] = rows
        write_json(export / f"{phase}.json", rows)
        for name in ("manifest.json", "versions.json"):
            source = RESULTS / phase / name
            if source.exists():
                shutil.copy2(source, export / f"{phase}-{name}")
    for name in ("mapping_profile", "mapping_profile-scalar-pilot"):
        shutil.copy2(RESULTS / name / "summary.json", export / f"{name}.json")
    return datasets


def find(data, phase, rid):
    return next(r for r in data[phase] if r["run_id"] == rid)


def trace(row):
    return json.loads((RESULTS / row["phase"] / row["run_id"] / "trace.json").read_text())


def cost(row):
    return row.get("physics_s", 0) / max(row.get("simulated_s", 0), 1e-15)


def err(row):
    value = row.get("normal_L1_relative", row.get("normal_L1_sampled"))
    return None if value is None else value * 100


def fmt(value, places=3):
    return "—" if value is None else f"{value:.{places}g}"


def main():
    data = load()
    plt.rcParams.update({"font.size": 12.5, "axes.spines.top": False, "axes.spines.right": False})  # 13 px as shown
    parts = []
    parts.append(
        section(
            "What the comparison establishes",
            """
    <p><strong>Refining the sphere quadrature does not automatically increase friction.</strong>
    Total sample area and stiffness are held fixed. At about 3.89 N total normal load,
    the pressure-based torsional friction envelope is 4.34–4.36 mN·m across the tested resolutions.
    The normal force law survives the articulated medium-torque transfer with about 0.17% local L1 error.
    Holding behavior also depends on tangential creep, timestep, and the friction model.</p>
    <p>The physical-stiffness translation and <code>diagexact</code> solve different problems.
    Exact diagonals alone do not establish a material stiffness. Cancelling the diagonal selected by the
    solver recovers the prescribed coefficients; it also works with frozen approximate diagonals.
    That frozen mapping can be computed once, including on the GPU.</p>
    <p>Vectorizing the CPU adapter preserves its trajectories and forces while removing most Python mapping
    overhead. The optimized 1 mm case takes about 1.42 wall seconds per simulated second; the precomputed
    variant takes about 0.90. Fine 128-world mjlab batches achieve about 9.09 (1 mm) and 4.54 (0.5 mm)
    aggregate simulated seconds per wall second. These are normal-law and throughput results; they do not
    establish matching dry stiction or a complete brake controller.</p>
    <p><a href="20261004-task_contact_benchmark.html">SR2 task and sliding-geometry benchmark</a> ·
    <a href="20261004-distributed_contact.html">Earlier static and controlled-slip audit</a> ·
    <a href="20261004-distributed_contact_spec.html">Original specification</a> ·
    <a href="20261004-contact_research_asides.html">DeliGrasp / Delassus research notes</a></p>
    """,
            "outcomes",
        )
    )
    from distributed_contact_video_page import summary_section

    parts.append(summary_section())
    parts.append(
        section(
            "Shared physical problem and normalization",
            """
    <p>The fixed-palm SR2 has nine articulated finger joints and a free six-DOF cylindrical tool.
    Thumb and index use 10.55 mm spherical tips; the middle finger is parked.
    The original calibrated joint servos, limits, rotor inertia, body masses and inertias are retained.
    The tool radius is 12.5 mm, length 100 mm and nominal mass 24.5437 g; mass controls scale its inertia too.
    Imported Drake and Newton fingertip/tool poses are checked against the MJCF source to within 1 µm.</p>
    <p>Each cap covers 45 degrees, with 0.75 mm sample spheres and E = 10 MPa.
    Spacings 2 / 1 / 0.5 / 0.25 mm have 51 / 205 / 819 / 3275 samples per fingertip.
    Every resolution has the same total cap area, 204.83 mm², and total sample stiffness,
    167.529 kN/m per cap. Each kᵢ decreases with sample count. The surface is sampled more finely without
    assigning each extra bump another full-strength spring. Only active samples carry force.</p>
    <p>Sliding friction is µ = 1, condim = 3, with no point torsional or rolling friction.
    Nominal servo targets request 2 N per pad and settle at about 1.9465 N each. The targets are shared
    across backends. Geometry, pressure distribution and measured normal load are compared alongside torque.</p>
    <p>The torsion runs have zero gravity and a shared weak translational guide (20 N/m, 0.05 N·s/m)
    and rotational damper (0.00002 N·m·s/rad). Their mass sweep measures inertial effects, rather than
    a maximum lifted payload. A separate object-weight test below removes the translational spring.</p>
    <p>Open-loop torque pulses begin after 0.3 s settling: +τ sin² over 150 ms, then the matching negative pulse
    over 150 ms, followed by decay to 0.8 s. Peaks are 3, 12 and 48 mN·m.
    Holding tests instead ramp torque, and record angular-speed thresholds sustained for 20 ms.
    An object leaving under a strong pulse is a grasp outcome; release alone is not evidence of a broken solver.</p>
    """,
            "protocol",
        )
    )

    stable = []
    for mass in (0.25, 1.0, 4.0):
        for spacing in (1.0, 0.5):
            row = (
                find(data, "holding_refine", f"quarter_s{spacing}_dt1.25e-05")
                if mass == 0.25
                else find(data, "holding", f"hold_s{spacing}_m{mass}")
            )
            st = row["steady_pad_load"]
            stable.append((mass, spacing, row, st))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for mass in (0.25, 1.0, 4.0):
        cells = [r for r in stable if r[0] == mass]
        ax[0].plot(
            [r[1] for r in cells],
            [r[3]["pressure_torsion_capacity_Nm"] * 1000 for r in cells],
            "o-",
            label=f"{mass:g}× mass",
        )
    nominal = [
        find(data, "holding", rid)
        for rid in ("hold_s2.0_m1.0", "hold_s1.0_m1.0", "hold_s0.5_m1.0", "hold_s025_m1")
    ]
    ax[0].plot(
        [2, 1, 0.5, 0.25],
        [r["steady_pad_load"]["pressure_torsion_capacity_Nm"] * 1000 for r in nominal],
        "k--",
        alpha=0.5,
        label="nominal, all spacings",
    )
    ax[0].set(xlabel="Sample spacing (mm)", ylabel="Friction envelope (mN·m)", ylim=(4.25, 4.45))
    ax[0].legend(fontsize=12)
    for phase, rid, label in [
        ("holding", "hold_s1.0_m1.0", "1 mm, Cₜ/Cᵢ=10"),
        ("holding", "hold_s0.5_m1.0", "0.5 mm, Cₜ/Cᵢ=10"),
        ("friction_refine", "hold_ct100_dt25", "0.5 mm, Cₜ/Cᵢ=100"),
        ("drake_transfer", "drake_hold_m1.0", "Drake SAP"),
    ]:
        rows = [
            r
            for r in trace(find(data, phase, rid))
            if 0.3 <= r["time"] and abs(r["angular_velocity"]) < 1.5
        ]
        ax[1].plot(
            [r["torque"] * 1000 for r in rows],
            [abs(r["angular_velocity"]) for r in rows],
            label=label,
        )
    ax[1].axvline(4.34, color="gray", ls=":", label="nominal envelope")
    ax[1].set(
        xlabel="Applied ramp torque (mN·m)",
        ylabel="Angular speed (rad/s)",
        ylim=(0, 1.5),
        xlim=(0, 6),
    )
    ax[1].legend(fontsize=12)
    body = save_figure(
        fig,
        "articulated_holding",
        "Pressure-based capacity converges across resolution and mass. Operational holding curves also expose tangential creep. MuJoCo Cₜ/Cᵢ=10 ramps at 12 mN·m/s; the Cₜ/Cᵢ=100 and Drake curves ramp at 8 mN·m/s, so threshold differences include inertial lag.",
    )
    body += table(
        ["Mass scale", "Spacing mm", "dt µs", "Normal N", "Envelope mN·m", "Envelope / normal mm"],
        [
            [
                m,
                s,
                r["config"]["timestep"] * 1e6,
                fmt(st["normal_thumb"] + st["normal_index"], 5),
                fmt(st["pressure_torsion_capacity_Nm"] * 1000, 6),
                fmt(
                    st["pressure_torsion_capacity_Nm"]
                    / (st["normal_thumb"] + st["normal_index"])
                    * 1000,
                    6,
                ),
            ]
            for m, s, r, st in stable
        ],
    )
    body += """<p>The envelope is Σ µFₙᵢ ‖(I − nᵢnᵢᵀ)[u × (xᵢ − P)]‖.
    It predicts the maximum tangential contact moment for pure rotation about the pinch axis under the
    measured normal pressure field. It is not a measured static breakaway torque and ignores simultaneous
    translation demands. Raising commanded pad load from 2 to 4 N raises measured total normal load to
    7.884 N and the 0.5 mm envelope to 10.538 mN·m, because the contact patch grows as well as carrying more force.</p>
    <p>At nominal mass, 1 and 0.5 mm reach 0.1 rad/s at 1.188 and 1.176 mN·m on the same ramp,
    and reach 1 rad/s at 4.452 and 4.440 mN·m. Four-times mass delays these thresholds through inertia
    without materially changing the settled pressure envelope. Increasing sample count is not producing
    an unrestricted increase in friction.</p>
    <p>The quarter-mass 1 and 0.5 mm cases at 25 µs alternate load between fingers and are excluded from
    the converged capacity comparison. At 12.5 µs both fingers settle at about 1.9465 N; halving again to
    6.25 µs changes the envelope by less than 0.0001%. The two 25 µs records remain in the exported data.</p>
    <p>Drake’s pressure shape, rescaled per patch to its actual SAP normal resultant, predicts about
    4.426 mN·m at 3.8885 N, versus 4.337 mN·m at 3.8933 N for the 0.5 mm samples. The Drake geometric
    estimate uses the rotation lever without projecting onto each face tangent. Raw pressure integrals
    are exported separately and are not equated to the discrete SAP reaction force.</p>"""
    parts.append(
        section("Holding torque: resolution, mass and actual normal load", body, "holding")
    )

    rows = data["payload"]
    body = """<p>To separate payload support from the torsion inertia sweep, eight additional cases ramp an
    object-only downward acceleration from 0 to 9.81 m/s² during 0.3–0.4 s and run to 0.7 s.
    No translational spring supports the object; the common 0.05 N·s/m damper remains.
    Joint targets and µ remain fixed. This is a short finite-duration support test, not a certified maximum mass.</p>"""
    body += table(
        ["Spacing mm", "Mass g", "Weight N", "Outcome", "Peak displacement mm", "Normal L1 %"],
        [
            [
                r["config"]["spacing_mm"],
                fmt(24.5437 * r["config"]["mass_scale"], 5),
                fmt(0.0245437 * r["config"]["mass_scale"] * 9.81),
                "held to 0.7 s" if r["status"] == "complete" else "released / guard reached",
                fmt(r.get("peak_displacement_mm")),
                fmt(err(r)),
            ]
            for r in rows
        ],
    )
    body += "<p>Both resolutions remain within the grasp bound for the 24.5 g and 196.3 g cases over the declared horizon; both release in the 392.7 g and 785.4 g cases. The 196.3 g objects move about 4.76 mm, so this is finite-horizon retention with creep, not static payload support. It brackets this protocol’s retention similarly across resolutions without establishing an exact mass threshold or indefinite holding.</p>"
    parts.append(section("Payload support across resolutions", body, "payload"))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for phase, rid, label in [
        ("transfer_optimized", "optimized_s1", "MuJoCo 1 mm"),
        ("transfer_optimized", "optimized_s05", "MuJoCo 0.5 mm"),
        ("drake_transfer", "drake_tau0.012", "Drake 1 mm mesh"),
        ("mjlab_transfer", "fine_s0.5_batch128", "mjlab 0.5 mm, 128"),
    ]:
        rows = trace(find(data, phase, rid))
        axes[0].plot([r["time"] for r in rows], [r["angular_velocity"] for r in rows], label=label)
    axes[0].set(xlabel="Time (s)", ylabel="Angular velocity (rad/s)")
    axes[0].legend(fontsize=12)
    for phase, rid, label in [
        ("transfer", "fine_s0.5_tau0.003", "Cₜ/Cᵢ=10, 50 µs"),
        ("friction_transfer", "creep_ct100.0", "100, 50 µs"),
        ("friction_transfer", "creep_ct1000.0", "1000, 50 µs"),
        ("friction_refine", "creep_ct1000_dt6.25", "1000, 6.25 µs"),
        ("drake_transfer", "drake_tau0.003", "Drake, 500 µs"),
    ]:
        rows = trace(find(data, phase, rid))
        axes[1].plot([r["time"] for r in rows], [r["angular_velocity"] for r in rows], label=label)
    axes[1].set(xlabel="Time (s)", ylabel="Angular velocity (rad/s)")
    axes[1].legend(fontsize=12)
    body = save_figure(
        fig,
        "articulated_torsion",
        "Common 12 mN·m pulses give similar gross motion across fine CPU, GPU and Drake. The 3 mN·m pulses distinguish creep and timestep sensitivity; the oscillatory high-damping case is retained.",
    )
    body += """<p>At 12 mN·m, 1 / 0.5 / 0.25 mm yield peak angular speeds 19.251 / 19.275 / 19.245 rad/s.
    Halving the 0.5 mm timestep to 25 µs gives 19.274 rad/s. Their local normal L1 errors are about
    0.168–0.174%. Drake gives 18.742 rad/s at 500 µs, 18.919 at 100 µs, and 18.696 with a finer mesh.
    Similar peak motion alone would miss the 20–52% normal-law errors in the coarse original-impedance controls.</p>
    <p>At 3 mN·m, Cₜ/Cᵢ = 10 gives about 0.402 rad/s versus Drake’s 0.0832.
    Raising the ratio to 100 gives 0.126 rad/s with excellent normal fidelity. Ratio 1000 at 50 µs
    gives spurious oscillation and 14.45% normal error; 25 µs still gives 5.57%.
    At 6.25 µs, ratio 1000 gives 0.0929 rad/s and 0.00523% normal error, but costs about 71.65 wall
    seconds per simulated second in the original scalar adapter. Stronger friction damping can improve
    holding behavior, while exposing another timestep and cost tradeoff.</p>
    <p>In friction rows the position residual is zero. Finite soft tangential parameters permit velocity-dependent
    creep inside the Coulomb bound; matching kᵢ for the normal spring does not establish a zero-creep dry
    stiction law. The Cₜ/Cᵢ = 100 ramp reaches 0.1 rad/s at 4.032 mN·m, versus Drake’s 4.656 on the same
    8 mN·m/s ramp. This improves the comparison but does not make the friction laws equivalent.</p>
    <p>The 48 mN·m pulses release the distributed grasp at all refined resolutions, with peak speeds around
    130–131 rad/s before reaching the 25 mm escape bound. Their pre-escape normal errors are about 1.1–1.3%.
    Release is reported as a physical outcome separately from the 500 µs fine-impedance cases that violate
    robustness bounds during preload, before any torque. The successful timestep window is bounded.</p>"""
    parts.append(section("Dynamic transfer and the friction discriminator", body, "transfer"))

    profile = json.loads((RESULTS / "mapping_profile/summary.json").read_text())
    body = """<p>Let Λₛ be the normal diagonal actually selected for impedance scaling, and
    A(q) = JM⁻¹Jᵀ the true coupled inverse inertia. The implemented constant-impedance direct mapping is</p>
    <pre>sᵢ = (1 − d₀) Λₛ,ᵢ
solrefᵢ = (−kᵢ sᵢ, −cᵢ sᵢ),   cᵢ = 0.03 kᵢ
solreffrictionᵢ = (0, solrefᵢ[1] × (Cₜ/Cᵢ) / impratio)</pre>
    <p>With constant solimp this recovers physical kᵢ and cᵢ in the normal force-reference coefficient.
    It keeps d₀ fixed. It does not implement the pasted ChatGPT proposal to rewrite d every step.
    <code>diagexact</code> changes Λₛ to current Aᵢᵢ; it does not cancel Λₛ by itself.
    The correction is therefore complementary to the flag, rather than a replacement API feature.</p>
    <p>The fine variant uses d₀ = 0.0001, impratio = 10000 and Cₜ/Cᵢ = 10 unless declared otherwise.
    The signed interior normal identity remains fᵢ = kᵢδᵢ − cᵢvₙᵢ − (Jq̈)ᵢ/Rᵢ + cone correction.
    Coefficient recovery is exact; finite-acceleration constitutive fidelity is measured separately.</p>
    <p>For current exact diagonals, CPU code reads the native diagonal and vectorizes the scalar translation
    over active contacts. It rebuilds constraint stages after the contact update. No dense A matrix is
    constructed in the timed physics loop. Matrix inversions belong only to the diagnostic records.</p>"""
    body += table(
        [
            "Spacing mm",
            "Policy",
            "Contacts",
            "Total µs",
            "Mapping µs",
            "Refresh µs",
            "step1 µs",
            "step2 µs",
        ],
        [
            [
                r["config"]["spacing_mm"],
                r["config"]["policy"],
                r["ncon"],
                fmt(r["mean_us"]["total"], 4),
                fmt(r["mean_us"].get("scalar_translation")),
                fmt(r["mean_us"].get("constraint_refresh")),
                fmt(r["mean_us"].get("step1")),
                fmt(r["mean_us"].get("step2")),
            ]
            for r in profile
        ],
    )
    body += """<p>These are 1000-step means after 0.3 s settling, on this CPU and hand. Vectorized translation costs
    roughly 12–16 µs for 32–129 active contacts. The earlier scalar loop cost roughly 101–273 µs.
    The duplicate refresh costs about 5–20 µs. This is linear scalar work plus the native solve, not a
    guarantee about thousands of active contacts or arbitrary robots. Timing is a single warmed measurement,
    not a confidence interval or controlled multicore benchmark.</p>
    <p>If Λₛ is the frozen approximation, per-pair physical coefficients can be precomputed from the selected
    body weights. CPU <code>compiled_fine</code> and GPU fine cases do this and need no runtime callback.
    CPU precomputed 1 / 0.5 mm retain the same normal errors and almost identical peak motion as exact mapping.
    Exact inertia is accessible and useful, but these results do not show it is required for this envelope.</p>
    <p><a href="data/chatgpt_inertia_mapping_guidance.txt">Archived ChatGPT inertia guidance</a> is useful
    conceptually, but its description of our d-rewrite is incorrect. The positive-format expression
    d = 1 − 1/(t꜀²kΛ) is an alternative at damping ratio 1 and can leave the allowed impedance range.
    <a href="https://mujoco.readthedocs.io/en/3.14.0/modeling.html#solver-parameters">MuJoCo’s solver parameter
    documentation</a> describes the native conventions.</p>"""
    parts.append(section("Physical stiffness mapping: exactness and cost", body, "mapping"))

    selected = [
        find(data, "transfer", rid)
        for rid in (
            "compiled_s1.0",
            "compiled_s0.5",
            "standard_runtime",
            "refine_s0.5",
            "refine_s0.25",
        )
    ]
    selected += data["transfer_optimized"] + data["transfer_compiled_fine"]
    selected += [find(data, "friction_refine", "creep_ct1000_dt6.25")]
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = {
        "compiled_s1.0": "Original impedance, 1 mm / 500 µs",
        "compiled_s0.5": "Original impedance, 0.5 mm / 500 µs",
        "standard_runtime": "Exact diagonal, original impedance / 50 µs",
        "refine_s0.5": "Fine law, 0.5 mm / 25 µs, scalar adapter",
        "refine_s0.25": "Fine law, 0.25 mm / 25 µs, scalar adapter",
        "optimized_s1": "Fine law, 1 mm / 50 µs, vectorized",
        "optimized_s05": "Fine law, 0.5 mm / 50 µs, vectorized",
        "compiled_fine_s1": "Fine law, 1 mm / 50 µs, precomputed",
        "compiled_fine_s05": "Fine law, 0.5 mm / 50 µs, precomputed",
        "creep_ct1000_dt6.25": "Cₜ/Cᵢ=1000, 0.5 mm / 6.25 µs, scalar",
    }
    for index, row in enumerate(selected, 1):
        ax.scatter(cost(row), err(row), s=45, label=f"{index}. {labels[row['run_id']]}")
        ax.annotate(
            str(index),
            (cost(row), err(row)),
            xytext=(4, 5),
            textcoords="offset points",
            fontsize=11.5,
        )
    drake_cost = cost(find(data, "drake_transfer", "drake_tau0.012"))
    ax.axvline(drake_cost, ls="--", color="gray", label="Drake 500 µs cost (different normal law)")
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="Wall seconds / simulated second",
        ylabel="Local normal-law L1 error (%)",
        ylim=(0.002, 100),
    )
    ax.legend(fontsize=12, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2)
    body = save_figure(
        fig,
        "articulated_cost_fidelity",
        "CPU cost versus local normal-law fidelity. Cₜ=1000 uses the weak pulse; the other points use the medium pulse. The x-axis omits diagnostic logging. The Drake marker supplies a cost reference, without assigning it an artificial zero error against a different law.",
    )
    comparison = (
        data["transfer_optimized"]
        + data["transfer_compiled_fine"]
        + [
            find(data, "transfer", "compiled_s1.0"),
            find(data, "transfer", "compiled_s0.5"),
            find(data, "drake_transfer", "drake_tau0.012"),
            find(data, "drake_transfer", "drake_dt1ms"),
            find(data, "drake_transfer", "drake_mesh05"),
        ]
    )
    body += table(
        ["Case", "dt µs", "Wall s / sim s", "Normal L1 %", "Peak rad/s"],
        [
            [
                r["run_id"],
                r["config"]["timestep"] * 1e6,
                fmt(cost(r), 4),
                fmt(err(r), 4),
                fmt(r.get("peak_angular_speed"), 5),
            ]
            for r in comparison
        ],
    )
    body += """<p>The old roughly 30× MuJoCo/Drake speed claim described another fixture and coarse physics.
    It cannot be carried forward to the fine articulated contact law. Before vectorization, 1 and 0.5 mm
    cost 3.28 and 9.14 s/sim-s; afterward 1.42 and 3.86. Precomputing the frozen mapping gives 0.90 and 2.94.
    Drake here costs 3.33 at 500 µs, 1.71 at 1 ms, and 8.05 with its finer mesh.
    Hardware, software wrappers, integration schemes and mechanical criteria differ; these are observed
    operating points, not a universal engine ranking.</p>
    <p><a href="data/chatgpt_speed_fidelity_guidance.txt">Archived ChatGPT speed/fidelity guidance</a> correctly
    asks for this Pareto comparison. The stronger-R control already exists: reducing d suppresses the normal
    acceleration residual, but forces a smaller timestep. The articulated and batched results supersede
    its claim that these tests are still entirely pending. Hardware brake control and broad operating-envelope
    validation remain open.</p>"""
    parts.append(section("CPU speed and fidelity tradeoffs", body, "cost"))

    fig, ax = plt.subplots(figsize=(8, 4))
    for spacing in (1.0, 0.5):
        for policy, style in (("compiled", "--"), ("runtime_approx", "-")):
            rows = sorted(
                [
                    r
                    for r in data["mjlab_transfer"]
                    if r["config"]["spacing_mm"] == spacing and r["config"]["policy"] == policy
                ],
                key=lambda r: r["config"]["nworld"],
            )
            ax.plot(
                [r["config"]["nworld"] for r in rows],
                [r["aggregate_real_time_factor"] for r in rows],
                "o" + style,
                label=f"{spacing:g} mm, {'500 µs original impedance' if policy == 'compiled' else '50 µs fine normal law'}",
            )
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="Parallel worlds",
        ylabel="Aggregate sim seconds / wall second",
    )
    ax.legend(fontsize=12)
    body = save_figure(
        fig,
        "articulated_gpu_throughput",
        "Actual mjlab Simulation / MuJoCo Warp CUDA graph runs on the RTX 4070 Ti SUPER. Faster coarse physics has 21–53% normal-law error. The fine batch retains about 0.17% sampled normal error.",
    )
    body += table(
        ["Case", "Worlds", "Aggregate sim s / wall s", "Aggregate steps/s", "Normal L1 %"],
        [
            [
                r["run_id"],
                r["config"]["nworld"],
                fmt(r["aggregate_real_time_factor"], 5),
                fmt(r["aggregate_steps_per_s"], 6),
                fmt(err(r), 4),
            ]
            for r in data["mjlab_transfer"]
        ],
    )
    body += """<p>The pinned stack is mjlab 1.2.0, MuJoCo Warp 3.6.0, Warp 1.12.1 and torch 2.11.0+cu128.
    Fine 128-world runs sustain about 181756 steps/s at 1 mm and 90738 at 0.5 mm,
    or 9.09 and 4.54 aggregate sim-s/wall-s. All worlds receive the same open-loop experiment;
    this is simulator throughput, not a measured RL training rate.</p>
    <p>Each synchronized graph includes the applied wrench and every native physics step.
    Allocation, compilation, host transfers and force logging are outside physics timing; startup cost is
    exported separately. State guards cover every batch world. Normal error is audited at 100 Hz in world 0,
    whereas the CPU audit includes every physics step. GPU contact/constraint buffers are 2048/8192 per world.
    Earlier undersized-buffer trials are archived as pilots and excluded from the benchmark.</p>
    <p>Warp 3.6 contact scaling uses welded-body inverse weights where CPU 3.6 uses the geometry-body weights.
    The GPU per-pair translation uses its own selected weights so kᵢ and cᵢ match physically.
    These GPU runs do not implement changing exact Aᵢᵢ. A separate current MuJoCo Warp 3.14 probe rejects
    <code>diagexact</code> as unsupported; that feature gap is preserved in the environment record.</p>"""
    parts.append(section("mjlab: fine contact law in a real GPU batch", body, "gpu"))

    pressure = data["newton_pressure"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for reduced, matching, label in [
        (False, False, "unreduced"),
        (True, False, "reduced default"),
        (True, True, "reduced + moment matching"),
    ]:
        rows = sorted(
            [
                r
                for r in pressure
                if r["config"]["static_squeeze_m"] == 0.0002
                and r["config"]["reduce_contacts"] == reduced
                and r["config"]["moment_matching"] == matching
            ],
            key=lambda r: r["config"]["voxel_mm"],
        )
        x = [r["config"]["voxel_mm"] for r in rows]
        axes[0].plot(
            x,
            [r["normal_force_projection"] / r["continuum_normal_2pads"] for r in rows],
            "o-",
            label=label,
        )
        axes[1].plot(
            x,
            [r["torsion_capacity_friction_scaled"] / r["continuum_torque_2pads"] for r in rows],
            "o-",
            label=label,
        )
    for ax in axes:
        ax.axhline(1.0, color="gray", ls=":")
        ax.set(xlabel="SDF voxel spacing (mm)", ylabel="Ratio to Winkler continuum")
        ax.legend(fontsize=12)
    axes[0].set_title("Normal load, 0.2 mm indentation")
    axes[1].set_title("Torsional friction envelope")
    body = save_figure(
        fig,
        "articulated_newton_pressure",
        "Newton pressure generation separated from integration. Unreduced pressure converges with voxel refinement. Force-preserving reduction does not automatically preserve the torsional envelope; the optional friction moment correction improves this case.",
    )
    body += """<p>The Newton checkout is 1.7.0.dev0 at commit
    <code>009158e62b862b3b9d829397d6db583515ae1271</code>, with Warp 1.17.0 and MuJoCo Warp 3.14.0,
    installed in an isolated environment. It uses SDF hydroelastic pressure generation and
    <code>SolverMuJoCo(use_mujoco_contacts=False)</code> for dynamics. Hydroelastic collision generation
    and the constraint solver are distinct components; this test does not compare a wholly new solver.</p>
    <p>Both tip spheres and cylinder use SDFs. Pad pressure stiffness is E/R = 9.479×10⁸ N/m³;
    the tool is 10000× stiffer, with a 100× control. µ=1 and point torsion/rolling are zero.
    The static pressure-only study has 27 cases: three voxel resolutions, three indentations, and unreduced,
    default reduced, or reduced with <code>moment_matching</code>.</p>
    <p>At 0.2 mm indentation, 0.25 mm voxels produce 1.792 N versus the 1.834 N two-pad continuum target,
    a 2.25% deficit. Reduction retains 1.791 N with 24 contacts instead of 920. The unreduced friction
    envelope is 1.702 mN·m; default reduction gives 1.402, and enabled moment matching gives 1.601.
    Per-contact friction scaling is included in the corrected envelope. At 0.4 mm indentation, moment
    matching is less successful: 8.289 versus 9.657 mN·m unreduced. Force/wrench reduction needs an explicit
    torsional-capacity audit for the intended axis and loading.</p>
    <p><a href="https://github.com/newton-physics/newton/blob/009158e62b862b3b9d829397d6db583515ae1271/docs/concepts/collisions.rst">Newton collision documentation</a>
    and the pinned reduction implementation distinguish pressure generation, contact reduction and optional
    moment matching. <a href="https://drake.mit.edu/doxygen_cxx/group__hydroelastic__user__guide.html">Drake’s
    hydroelastic guide</a> provides the separate Drake model context.</p>"""
    body += table(
        [
            "Dynamic port case",
            "Status",
            "Valid sim s",
            "Normal L1 %",
            "Peak rad/s",
            "Failure / limitation",
        ],
        [
            [
                r["run_id"],
                r["status"],
                fmt(r.get("simulated_s")),
                fmt(err(r)),
                fmt(r.get("peak_angular_speed")),
                r.get("error")
                or r.get("failure")
                or "settled normal load ≈2.04 N, not the shared ≈3.89 N",
            ]
            for r in data["newton_transfer"]
        ],
    )
    body += """<p>The stock reduced dynamic port and reduced translated variants fail during preload, before the
    torque pulse. A 25 µs reduced trial encounters a CUDA graph capture error in the reduction diagnostic.
    In the historical record, the unreduced translated trial completed with 0.304% sampled normal error,
    settled at only 2.04 N total normal load and reached 35.75 rad/s. That was not a matched-preload
    torsion reference, and its completion is not reproduced in the video rerun below.</p>
    <p><strong>Video replay update:</strong> that previously completed unreduced trial now fails during preload
    when rerun for state capture. A separate control without recording also fails. Its historical summary is
    preserved, but successful dynamic transfer is not reproduced. See the
    <a href="20261004-contact_videos.html#newton-replay">actual video rerun and limitation</a>.
    Static pressure-generation results are evaluated independently.</p>
    <p>The translated trial is an explicit experimental adapter, rather than stock Newton: it converts
    generated face stiffness into native direct parameters before constraint construction, with the same
    fixed-d normal mapping. No solved force is overwritten. The failures and unmatched preload may reflect
    import, reduction, parameter or integration issues in this port; they do not establish a general Newton
    limitation or justify ranking its cost against the successful matched cases.</p>"""
    parts.append(
        section("Newton hydroelastic: pressure, reduction and port limits", body, "newton")
    )

    counts = []
    for phase in PHASES:
        rows = data[phase]
        counts.append(
            [
                phase,
                len(rows),
                sum(r.get("status") == "complete" for r in rows)
                if phase != "newton_pressure"
                else "pressure-only",
                sum(r.get("status") == "failed" for r in rows),
            ]
        )
    body = table(["Phase", "Recorded cases", "Completed", "Stopped / error"], counts)
    body += """<p>Every declared case in these manifests has a record. A status of failed can mean a deliberate
    physical escape guard, a preload instability, or a backend exception; the per-case record distinguishes
    them. A completed case can still have poor constitutive fidelity. No failed or inaccurate result is
    silently promoted into the fidelity comparison. Pressure-only cases are geometry evaluations, not trajectories.</p>
    <p>CPU normal L1 is Σ|Fₙᵢ − max(0,kᵢδᵢ − cᵢvₙᵢ)| / Σmax(0,kᵢδᵢ − cᵢvₙᵢ), over every
    physics step from 0.3 s. GPU audits sample 100 Hz in one world; Newton audits sample 200 Hz.
    The denominator includes all unilateral targets and the numerator includes contacts with zero target.
    Full CPU contacts and coupling diagnostics remain local under
    <code>results/20261004-distributed-contact</code>; source MJCF, manifests, logs and pilots are preserved.</p>
    <p>Portable <a href="data/articulated/cases.csv">case table</a>, phase JSON summaries, compressed traces,
    source/package hashes and stage profiles live in <code>data/articulated/</code>.
    Supplementary final environment snapshots identify the
    <a href="data/articulated/environment-current-cpu.json">current CPU</a>,
    <a href="data/articulated/environment-mjlab.json">mjlab</a>,
    <a href="data/articulated/environment-drake.json">Drake</a> and
    <a href="data/articulated/environment-newton.json">Newton</a> stacks; per-phase snapshots preserve
    the code hashes at the start of each original run. CPU measurements use the Intel Core Ultra 7 265KF.
    <a href="README.md">The runbook</a> gives interpreter paths and reproduction commands.
    These are spherical-tip fixed-palm transfer tests, not the full moving-palm brake task or hardware
    validation. The hardware rectangular tip’s 6 mm fillet remains a separate static study; its dynamic
    geometry co-design behavior is still untested.</p>"""
    parts.append(section("Records, checks and remaining scope", body, "records"))
    flat = []
    for phase, rows in data.items():
        for r in rows:
            flat.append(
                dict(
                    phase=phase,
                    run_id=r["run_id"],
                    status=r.get("status", "pressure-only"),
                    spacing_mm=r["config"].get("spacing_mm"),
                    timestep=r["config"].get("timestep"),
                    mass_scale=r["config"].get("mass_scale", 1),
                    normal_L1_percent=err(r),
                    physics_s=r.get("physics_s"),
                    simulated_s=r.get("simulated_s"),
                    peak_angular_speed=r.get("peak_angular_speed"),
                    failure=r.get("error") or r.get("failure"),
                )
            )
    with (DOC / "data/articulated/cases.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=flat[0])
        writer.writeheader()
        writer.writerows(flat)
    toc = (
        '<nav class="toc">'
        + "".join(
            f'<a href="#{anchor}">{label}</a>'
            for anchor, label in [
                ("videos", "Videos"),
                ("holding", "Holding"),
                ("payload", "Payload"),
                ("transfer", "Dynamics"),
                ("mapping", "Mapping"),
                ("cost", "CPU cost"),
                ("gpu", "GPU"),
                ("newton", "Newton"),
                ("records", "Records"),
            ]
        )
        + "</nav>"
    )
    page = frame(
        "Articulated contact: holding, torsion and computational cost",
        "Fixed-area quadrature, matched hand actuation, real CPU/GPU measurements, and an explicit audit of friction and hydroelastic reduction.",
        toc + "".join(parts),
        "Measured follow-up · spherical fingertips · all failed cases retained",
    )
    page = page.replace(
        "Generated by <code>scripts/distributed_contact_page.py</code> from the source DOCX and run manifests.",
        "Generated by <code>scripts/distributed_contact_transfer_page.py</code> from native experiment outputs.",
    )
    (DOC / "20261004-articulated_contact_transfer.html").write_text(retro_style.apply(page))  # plain page style (owner, 2026-10-09)
    print("Wrote articulated report:", len(flat), "case records")


if __name__ == "__main__":
    main()
