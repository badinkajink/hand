# Distributed fingertip contact audit — 4 October 2026

- [Measured report](20261004-distributed_contact.html)
- [Articulated holding, torsion, payload, CPU cost, mjlab and Newton follow-up](20261004-articulated_contact_transfer.html)
- [SR2 task benchmark, diagexact ablation and sliding geometry](20261004-task_contact_benchmark.html)
- [Live 50 Hz rolling task / physics timestep sweep](20261004-controller_timestep.html)
- [Video gallery: 59 individual clips and nine synchronized comparisons](20261004-contact_videos.html)
- [Original specification, converted to HTML](20261004-distributed_contact_spec.html)
- [Original DOCX](SR2_distributed_fingertip_contact_experiment_spec.docx)
- [Research asides: DeliGrasp, Delassus, lateral coupling, tangential friction and reduced skin](20261004-contact_research_asides.html)

This implements the specification's first-session stopping point: freeze and reproduce the
nominal MuJoCo/Drake chain, measure native contact forces on isolated plane/cylinder rigs,
and characterize mass/inertia/timestep/solver sensitivity. A separate static extension tests
the user's **6 mm fillet**. Spherical fingertips remain the primary baseline.

The original 492-run record comprises 2 regressions, 216 sphere-pad static runs, 108 scaling/solver
diagnostics, 126 fillet cases and a separate 40-run compensation extension. The follow-up adds
80 newer-engine mass/flag controls, 104 dynamic trials and 17 fine-step/transfer trials (693
declared cases). The coarse dynamic trials retain failures and oscillatory results; completing
a run does not imply that it passes a mechanical-fidelity check. Two original coarse cylinder load
runs do not settle within the declared 0.6 s; the report identifies them, without discarding
their data. Unsupported free-acceleration snapshots intentionally are not equilibria.

## Reproduction

Use the existing pinned environment (MuJoCo 3.6.0, Drake 1.57.0):

```bash
PY=logs/20261001-hom_contact/venv/bin/python
$PY scripts/distributed_contact.py regression --dry-run
$PY scripts/distributed_contact.py regression
$PY scripts/distributed_contact.py static --dry-run
$PY scripts/distributed_contact.py static
$PY scripts/distributed_contact.py sensitivity --dry-run
$PY scripts/distributed_contact.py sensitivity
$PY scripts/distributed_contact.py fillet --dry-run
$PY scripts/distributed_contact.py fillet
$PY scripts/distributed_contact.py compensation --dry-run
$PY scripts/distributed_contact.py compensation
$PY scripts/distributed_contact_page.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/test_distributed_contact.py -q
```

All experiment phases support `--config`, `--out`, `--dry-run` / `--list-config`, `--limit`,
and `--force`. `--limit` limits new launches, retaining the complete manifest and marking
missing runs. Completed run IDs are skipped; failed IDs retry. Use `--force` after changing
implementation code; configuration hashes alone do not detect code changes. A changed
manifest requires a separate output directory to preserve previous results.

The report builder validates that all 377 source paragraphs/table cells survive DOCX conversion.
Use `--spec-only` to convert the document without any simulation results. HTML uses the existing
MorphoHand result-page style; plots are both embedded and exported as SVG/PNG.

## Data

Local raw outputs: `results/20261004-distributed-contact/<phase>/<run_id>/` (Git ignored).
Each run contains configuration, source hashes/package versions, summary and stdout log.
Static runs include exact MJCF, generated sample parameters, physics-step aggregate CSV,
physics-step contacts in gzip CSV and final equilibrium contacts in plain CSV. Displacement
fixtures have one static sample, rather than a fabricated time trajectory. The full MuJoCo
chain logs sphere/tool contacts once per ten physics steps. Drake regression preserves the
existing aggregate force trace; hydroelastic face logging remains deferred.

`data/` holds portable machine-readable aggregates and version metadata. `figures/` holds
standalone plots. The per-phase manifest is authoritative for missing/failed runs, and the
report rebuilds its aggregates from each manifest and run summary. Existing reports and
controller files are preserved; the reports are regenerated with links to the follow-up.

## Interpretation boundaries

- The original report's static sphere-pressure drawings use intended `k * overlap`; the new
  measurements use `mj_contactForce`. They must not be treated as the same evidence.
- The frozen hand mapping is intentionally not recalibrated for the isolated fixed-pad fixture.
  Its nominal inverse-inertia mismatch predicts a 15.88% stiffness offset. At fixed indentation,
  force then scales directly with object mass. The successful nominal task alone does not
  validate a physical mass-independent spring law. The separate compensation extension now
  restores the same static normal stiffness in the isolated plane/cylinder fixture.
- Duplicate contacts remain separate constraint rows. Their summed force saturates in a free
  acceleration snapshot, but grows linearly in supported equilibrium. Rank redundancy does
  not imply that the solver deletes rows.
- Static displacement reactions use inverse dynamics followed by a forward equilibrium check.
  Separate load-controlled runs integrate forward from near first touch. The continuum
  reference is independent Winkler quadrature, not newly generated Drake static data.
- The reported pressure-weighted arm is a geometric proxy. Actual resting contact torque is
  also logged, but is not interpreted as measured sliding torsional capacity.
- The obsolete sharp-box tip is not tested. The fillet study uses analytic front patches with
  the measured 6 mm radius, hardware-facing width 14.8 mm from `scene_mutate.py`, and an assumed
  14.8 mm length. Front-only rounding topology and the omitted back remain CAD assumptions.
  No singularity was observed in this static sample; dynamics and geometry co-design remain
  unvalidated. Inspect resolution error, especially near the boundary at 85 degrees.
- Static Drake, dynamic 6 mm fillet and hardware testing remain downstream. Broader curvature
  and the complete brake task are revisited in the task/geometry report above.

## Analytic overlay and compensation extension

The report now plots measured `f_n / penetration` directly against
`1 / [tc² (1-d0) Lambda_approx]` over the original mass sweep. The compiler's
`efc_diagApprox` is the appropriate Lambda; `J M^-1 J^T` is logged separately.
The static overlay agrees at floating-point precision, and independently integrated
load equilibria corroborate the relation with finite residual velocities.

`configs/compensation.json` runs the frozen and compensated policies at 0.25, 0.5,
1, 2 and 4 times nominal mass, against the plane and cylinder, using both static
displacement and independent forward load control. The extension uses native direct
`solref = (-k_i*s, -c_i*s)` with `s=(1-d0)*Lambda_approx` and `c_i=k_i*relaxation`.
Geometry, physical normal stiffness, damping and impedance remain fixed. Only the
analytic simulator translation uses the compiled inverse inertia; there is no fitting
to measured force or another backend.

The 1 mm per-contact target is 817.212528 N/m. Compensation restores it across all
tested masses in the static fixture and produces nearly common equilibrium indentation
at a prescribed load. These static results do not establish compensated torsional capacity,
transient slip, or articulated-fingertip fidelity. The original adapter uses the shared
translational scaling of a fixed pad against one free body, not a general articulated
per-contact adapter.

New portable outputs: `data/stiffness_validation.csv`, `data/compensation.csv`,
`data/compensation_versions.json`, and the `stiffness_prediction` / `lambda_compensation`
plots in `figures/`. Earlier frozen-run logs and metadata are preserved; the extension
has its own manifest and implementation hashes.

The test command disables unrelated ROS pytest plugins inherited from the shell environment.

## Runtime inertia and controlled-slip follow-up

The follow-up uses an **isolated MuJoCo 3.14.0** environment; project dependencies and
the 3.6.0 regression environment are preserved. Setup and reproduction:

```bash
uv venv --python logs/20261001-hom_contact/venv/bin/python logs/20261004-contact-current/venv
uv pip install --python logs/20261004-contact-current/venv/bin/python mujoco==3.14.0 numpy==2.5.3 matplotlib==3.11.2 pytest==9.1.1
NEW=logs/20261004-contact-current/venv/bin/python
$NEW scripts/distributed_contact_dynamics.py diagexact --dry-run
$NEW scripts/distributed_contact_dynamics.py diagexact
$NEW scripts/distributed_contact_dynamics.py slip --dry-run
$NEW scripts/distributed_contact_dynamics.py slip
$NEW scripts/distributed_contact_dynamics.py slip --config docs/experiments/20261004-codex/configs/slip_fine.json --out results/20261004-distributed-contact/slip_fine
$NEW scripts/distributed_contact_page.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $NEW -m pytest tests/test_distributed_contact.py tests/test_distributed_contact_dynamics.py -q
```

The experimental coarse high-regularization cases intentionally retain failed results, so
the `slip` command returns a nonzero exit code after writing its aggregate. This is not
a missing run or a reason to discard those cases. Interrupted manifests resume and completed
cases are skipped; use a separate output directory when changing a configuration manifest.

MuJoCo 3.9 added `diagexact`; 3.10 renamed `efc_diagApprox` to `efc_diagA` and changed
the `mj_fullM` signature. The adapter supports the pinned and current APIs. The selected
**normal** diagonal is logged as `Lambda_solver`; `Lambda_exact` is independently computed
from J and M. `Lambda_approx` remains a legacy CSV alias for the selected value, and must not
be interpreted as approximate in newer exact-diagonal runs. Tangential diagonals receive
additional friction regularization, so the exact-normal statement does not apply to them.

Flag-only controls retain the original fixed 1 mm sphere parameters while sweeping 0.25–4×
object mass in static displacement and independent load simulations. Exact diagonals alone
do not remove the mass dependence. `runtime_selected_direct` maps the physical K and C into
per-contact native direct parameters at the current configuration, then reruns
`mj_makeConstraint`, `mj_island`, `mj_projectConstraint`, and `mj_fwdVelocity` before solving.
The articulated follow-up reuses this adapter on a fixed-palm, nine-joint hand. It is an
experimental adapter rather than a production controller.

The dynamic controls use 1 N normal load and either a 20 ms 0.5 N load pulse or a tangential
force ramp up to 1.2 N followed by braking. A shared 5 N·s/m external tangential damper bounds
travel; y and orientation have declared physical servos. The object is otherwise integrated
freely. Cylinder sliding is axial. Common physical relaxation is 30 ms for both 1 and 0.5 mm
quadratures. Each case saves contacts, aggregates, configuration and full normal coupling
matrices at five times. Fine-step contacts are logged every ten steps; all their aggregate
and error measurements still use every physics step.

The physical normal oracle is `max(0, K_i*delta_i - C_i*v_normal_i)`. Local-force L1 error is
the sum of absolute per-point deviations over time >=0.35 s, divided by the summed physical
target over that same interval. It includes points with zero predicted force in the numerator.
Acceleration and cone contributions are recorded separately; the signed interior identity is
`f_i = K_i*delta_i - C_i*v_normal_i - J_i*qacc/R_i + cone_correction_i`.
This checks the local law on visited states, not against an independently integrated friction solver.

The fine-step variant uses d0=0.0001 (large normal R), impratio=10000, and independent
`solreffriction` to retain C_t=10*C_i while changing the regularizer ratio. Smaller d reduces
the acceleration-force correction but exposes contact-damping timestep limits under
`implicitfast`. It is a combined variant; do not attribute all improvement to `diagexact`.
Fine-step transfer covers 0.25/1/4× masses, plane/cylinder contact, resolution, timestep and a
friction regularizer control. The report retains coarse-step failures and lists fine-step
errors separately. The articulated follow-up now records matched Drake, mjlab GPU and Newton
pressure/reduction trials. A later task study runs the full brake chain; hardware validation
remains open.

## Articulated holding, torsion and backend benchmarks

The separate follow-up contains 115 case records, plus six stage-cost profiles. It fixes total
sample area and stiffness across resolutions; logs actual normal force and operational angular-speed
thresholds; and distinguishes torsional friction envelopes from finite soft-contact creep. It includes
short payload-retention tests without the translational spring. Physical release, preload instability,
backend exceptions and completed-but-inaccurate trajectories remain in the data.

```bash
NEW=logs/20261004-contact-current/venv/bin/python
$NEW scripts/distributed_contact_transfer.py --config docs/experiments/20261004-codex/configs/transfer.json
$NEW scripts/distributed_contact_transfer.py --config docs/experiments/20261004-codex/configs/holding.json --out results/20261004-distributed-contact/holding
$NEW scripts/distributed_contact_transfer.py --config docs/experiments/20261004-codex/configs/holding_refine.json --out results/20261004-distributed-contact/holding_refine
$NEW scripts/distributed_contact_transfer.py --config docs/experiments/20261004-codex/configs/friction_transfer.json --out results/20261004-distributed-contact/friction_transfer
$NEW scripts/distributed_contact_transfer.py --config docs/experiments/20261004-codex/configs/friction_refine.json --out results/20261004-distributed-contact/friction_refine
$NEW scripts/distributed_contact_transfer.py --config docs/experiments/20261004-codex/configs/transfer_optimized.json --out results/20261004-distributed-contact/transfer_optimized
$NEW scripts/distributed_contact_transfer.py --config docs/experiments/20261004-codex/configs/transfer_compiled_fine.json --out results/20261004-distributed-contact/transfer_compiled_fine
$NEW scripts/distributed_contact_transfer.py --config docs/experiments/20261004-codex/configs/payload.json --out results/20261004-distributed-contact/payload
$NEW scripts/profile_contact_mapping.py
.venv/bin/python scripts/distributed_contact_gpu.py
logs/20261001-hom_contact/venv/bin/python scripts/distributed_contact_drake.py
logs/20261004-contact-transfer/venv/bin/python scripts/distributed_contact_newton.py
logs/20261004-contact-transfer/venv/bin/python scripts/distributed_contact_pressure.py
$NEW scripts/distributed_contact_transfer_page.py
$NEW scripts/distributed_contact_page.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $NEW -m pytest tests/test_distributed_contact.py tests/test_distributed_contact_dynamics.py tests/test_distributed_contact_transfer.py -q
```

The articulated runners retain every existing summary, including failures. A changed manifest
requires a new output directory. Reproduce code changes into fresh directories; do not overwrite
the historical scalar-adapter results. The optimized adapter vectorizes the contact parameter
translation without changing its physical law. `compiled_fine` uses the same low-impedance parameters
with the frozen selected diagonal precomputed per pair and no runtime refresh.

Newton uses isolated checkout commit `009158e62b862b3b9d829397d6db583515ae1271`; its environment
and checkout are under `logs/20261004-contact-transfer/`. The pinned project environment supplies
mjlab / MuJoCo Warp 3.6.0; current CPU MuJoCo 3.14.0 and Drake 1.57.0 remain in their original
separate environments. Current Warp 3.14 rejects `diagexact`; fine GPU cases instead precompute
their own backend-specific frozen-inertia mapping. Newton's dynamic port does not yet match the
shared preload, so its pressure-only reduction results are reported independently.

Portable summaries, compressed traces, source hashes, stage profiles and the case CSV are under
`data/articulated/`. Archived ChatGPT discussions are linked from the report with implementation
corrections and the measured speed/fidelity comparison. Raw contacts, scene XML and pilot failures
stay in the ignored local result directories. Figures are embedded in HTML and exported as SVG/PNG.

Final follow-up accounting: all 80 flag/mass controls and all 17 fine-step trials completed.
The coarse dynamic sweep completed 87 cases and records 17 out-of-fixture failures.
No declared case is missing. The 12 primary fine-step cases preserve the common physical
coefficients and show approximate independent normal-law behavior during acceleration and
sliding; the separate lower friction-regularizer control has larger force error. Read the
generated report for force-error bounds and trajectory convergence. Earlier unbounded pilots
and the outputs preceding the strengthened normal-excursion guard remain locally archived;
they are not counted as separate final-manifest cases.

The DeliGrasp/VLM prior, matrix-free Delassus connections, and reduced fingertip with lateral
coupling and tangential shear are catalogued as speculative backlog items with primary sources.
The page archives the supplied commentary and specifies a 20 ms controller sweep
over 50 µs, 0.5, 1, 2, 5 and 10 ms physics steps, with task metrics, cost and native-state videos.
It distinguishes the already-working legacy 1 ms MuJoCo baseline from physical-mapping variants,
and notes that published CSLC is quasistatic but does not model kinetic sliding dynamics.
No semantic estimator, probing controller or reduced skin was implemented for these asides.
The subsequent user-authorized nominal timestep batch is now implemented separately below.

## 50 Hz rolling timestep batch

`scripts/distributed_contact_timestep.py` runs 42 nominal cases: six physics steps × legacy,
legacy exact diagonal, fixed physical mapping with approximate/exact diagonal, runtime mapping
with approximate/exact diagonal, and Drake hydroelastic. MuJoCo workers use 3.14.0; Drake workers
use the existing 1.57.0 environment. Controller gains and commands remain fixed at 50 Hz.
The batch records native videos, controller-boundary tool poses/wrenches, a geometric slip proxy,
tracking metrics and physics timing with setup/rendering excluded. Wrenches are latest-solve
samples, not interval impulse averages; slip is not resolved material-point displacement.
A 10 ns schedule tolerance prevents timestep-dependent floating-point roundoff from adding a
controller tick to fixed-duration phases. Existing task callers retain their historical schedule.

```bash
NEW=logs/20261004-contact-current/venv/bin/python
$NEW scripts/distributed_contact_timestep.py --dry-run
$NEW scripts/distributed_contact_timestep.py
$NEW scripts/distributed_contact_timestep.py --report-only
```

Runs are sequential, each in a separate process, with a 30 minute per-case timeout. Restarting
skips every recorded outcome, including failures. The manifest freezes source hash and nominal
screening tolerances; source changes require a separate output directory rather than reusing
old results. The live HTML and portable aggregate refresh after every case, retaining pending
and failed entries. Raw data and `progress.json` are in
`results/20261004-distributed-contact/task_timestep/`; the detached batch log is `batch.log` there.
Mass/preload/speed perturbations, denser resolution and Newton are not in this nominal screen.

The earlier geometry follow-up executed both declared 17-case sliding manifests (10 force-ramp
completions, seven fixture exits; 17 guided completions). This is complete execution of that
follow-up, not completion of the full original geometry/load/resolution/phase or hardware plan.
Matched Drake static geometry and a reproducible Newton dynamic comparison remain open.

## Native-state videos

The gallery covers sphere resolution, torque pulses, holding ramps, mass and payload sweeps,
friction and timestep controls, CPU/mjlab/Drake comparisons, isolated normal pulses and sliding,
Newton preload failures and prescribed static pressure snapshots, and the 6 mm fillet.
It also links the earlier full closed-loop hand demonstration as a historical baseline.
Successful, drifting, released and numerically failed runs all remain visible.

Articulated CPU and Drake reruns save actual poses and native contact records at 200 Hz.
GPU clips replay the original batch's archived poses for world 0 at 100 Hz; their colors are
neutral because per-point forces were not archived. Rendering uses recorded poses without
interpolation or a new contact solve. Pressure colors, translucent geometry and orientation
markers are display aids. Every clip labels simulation time and playback speed; terminated
clips freeze at their last captured state with a visible guard badge. Static pressure and
fillet sequences are labelled as prescribed snapshots, not time-integrated motion.

The 45 native dynamic reruns reproduce 44 historical summaries within the exported scalar
tolerances. Newton's previously completed translated unreduced trial fails during preload
both with capture and in a separate control without recording. Its historical record remains
intact, but its dynamic completion is now unconfirmed. See
[capture validation](data/video_capture_validation.json) for the compared fields and control.
Videos help inspect motion and contact geometry; their sampling cannot expose every fine-step
force oscillation, so the original native traces remain necessary.

```bash
NEW=logs/20261004-contact-current/venv/bin/python
$NEW scripts/distributed_contact_videos.py capture --backend cpu
logs/20261001-hom_contact/venv/bin/python scripts/distributed_contact_videos.py capture --backend drake
logs/20261004-contact-transfer/venv/bin/python scripts/distributed_contact_videos.py capture --backend newton
$NEW scripts/distributed_contact_videos.py capture --backend isolated
logs/20261004-contact-transfer/venv/bin/python scripts/distributed_contact_video_snapshots.py pressure
$NEW scripts/distributed_contact_video_snapshots.py fillet
$NEW scripts/distributed_contact_videos.py render
$NEW scripts/distributed_contact_video_page.py
$NEW scripts/distributed_contact_transfer_page.py
$NEW scripts/distributed_contact_page.py
```

Capture writes separate raw runs under `results/20261004-distributed-contact/video_capture/`;
it does not overwrite the quantitative experiments. Capture overhead is not a solver timing
benchmark. Portable clip manifests and validation are in `data/video_*.json`; MP4s and JPEG
posters are in `media/`. Re-render changed displays with `render --force`; remove the affected
`media/compare-*.mp4` and poster before rebuilding a changed comparison.

## SR2 tasks and sliding geometry

The [task report](20261004-task_contact_benchmark.html) runs the same SR2 screwdriver
pickup, closed-loop brake, hold and insertion controller across legacy MuJoCo spheres,
physically mapped 1/0.5 mm spheres, point contacts and Drake hydroelastic. Its second
task adapts the cylinder-as-effector rolling-pinch steps in Wang, Oh and Pollard's
2026 preprint. The rolling headline runs use 500 Hz control, matching the paper's stated
rate, but retain this repository's hand, scene and controller implementation. The report
has native video of both tasks, synchronized grids and machine-readable per-stage scores.

MuJoCo 3.14 repeats the 1 mm hand tasks at 50 µs with `diagexact` off/on for legacy
contact, fixed physically mapped coefficients and per-step runtime retranslation.
It separates the exact-diagonal flag from preserving the prescribed local stiffness.
The 1 mm physical variants fail the full chain even when the runtime exact diagonal is
used; legacy contact succeeds with either flag. In rolling, fixed physical coefficients
and the exact flag can increase tool-axis rotation while also allowing more relative
sliding than legacy contact. The complete gain/slip comparison is in the report; a high
twist gain alone is not evidence of faithful rolling. A 0.5 mm mapped pad does better on
the rolling task but costs many more physics wall seconds per simulated second.
Holding the runtime exact normal mapping fixed while increasing tangential damping
from 10× to 100×/1000× increases relative sliding and angular tracking error; 100×
also worsens the full chain. This rules out a simple tangential-stiffening rescue.

The original geometry sweep applies a force ramp and brake to planes, cylinders of radii
6.25–50 mm along and across the cylinder axis, and spheres of radii 12.5/25 mm. It
includes 0.5 mm samples and 30°/60° phase controls at the representative radius. Small
curved objects can leave the 1 N/1.2 N force-controlled fixture during across-curvature
sliding; their saved partial traces and exit times remain in the results. A separate
3 mm guided-path protocol uses the same 1 N preload plus a declared normal spring that
follows the analytic surface height. It tests local sliding before loss of contact
dominates. Both use MuJoCo 3.14's selected runtime diagonal and the mapped normal law.
Five guided trajectories and two force-ramp fixture exits are filmed from native state;
the latter stop at the original exit guard and reproduce its time and pose.

```bash
OLD=logs/20261001-hom_contact/venv/bin/python
NEW=logs/20261004-contact-current/venv/bin/python
uv pip install --python "$NEW" scipy==1.16.2 imageio==2.37.0 imageio-ffmpeg==0.6.0 pillow==11.3.0
MUJOCO_GL=egl $OLD scripts/distributed_contact_tasks.py
MUJOCO_GL=egl $NEW scripts/distributed_contact_diagexact_tasks.py
$NEW scripts/distributed_contact_dynamics.py slip --config docs/experiments/20261004-codex/configs/sliding_geometry.json --out results/20261004-distributed-contact/sliding_geometry
$NEW scripts/distributed_contact_dynamics.py slip --config docs/experiments/20261004-codex/configs/sliding_geometry_guided.json --out results/20261004-distributed-contact/sliding_geometry_guided
MUJOCO_GL=egl $NEW scripts/distributed_contact_geometry_videos.py
MUJOCO_GL=egl $OLD scripts/distributed_contact_task_page.py
```

The force-ramp command returns a nonzero exit after recording controlled-contact exits.
They are measured fixture outcomes, not missing runs. Newton's earlier translated
hydroelastic transfer briefly completed one historical run but failed the independent
preload replay. A matched Newton task score would be misleading until that gate passes;
the native failure video and raw trace remain in the gallery. The 6 mm hardware fillet
has a static audit only; spherical fingertips remain the dynamic task baseline.
