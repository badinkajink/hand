# scripts/

Every script of the repository, in one folder per topic. A script imports any other script by its bare name:
`_paths.py` puts every folder here and `../src` on `sys.path`. A script in a folder starts with

```python
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/, for _paths
import _paths
```

and finds another script's file with `_paths.script("name.py")`. Scripts run from the repository root, through
`uv run --extra rl --extra gpu python scripts/<folder>/<script>.py`; result-page builders run with `.venv/bin/python`.
Until 2026-11-03 every script was directly in `scripts/`. `MOVED.tsv` maps each old path to its new one, for the
notes and pages written before. `archive/` has the scripts no longer used, with a README that says when and why each
was moved; move a script back before running it.

Shared numerical code is in the package `sr2hand` (`src/sr2hand/`), not in scripts: rotations, quaternions and rigid
transforms are in `sr2hand.geometry.rotations`, over scipy.

This file is written by `scripts/docs/scripts_map.py` from the scripts' docstrings.


## cad/ (40)

Printable parts: the DIP link, the UR5 mounts and the print plates.

| script | what it does |
|---|---|
| `cad_v1_collision_bench.py` | Step time of the screwdriver scene under four collision models of the finger (2026-10-13). |
| `cad_v1_finger_media.py` | Figures and films of the cad_v1 finger for its result page (2026-10-13). |
| `cad_v1_finger_page.py` | Page: MuJoCo model of the SR2 finger assembled from its CAD parts, with sphere collision geometry (2026-10-13). |
| `cad_v1_interference.py` | Interpenetration of the cad_v1 finger's CAD parts, posed by the compiled model. |
| `cad_v1_print_mass.py` | Printed masses of the cad_v1 finger's PLA parts, from PrusaSlicer (2026-10-13). |
| `cad_v1_sphere_contacts.py` | Self-contacts of the cad_v1 finger's collision spheres over the range of motion (2026-10-13). |
| `cad_v1_spheres.py` | Collision spheres for the cad_v1 finger: MorphIt on every CAD part, combined per body. |
| `cad_v2_assets.py` | assets/cad_v2: the DIP link with its fingertip parts, and the bench fixtures for the UR5e and the ATI Mini40 and Mini45, gathered from their experiment folders, with exact STEP files where those folders have only meshes  |
| `cad_v3_assets.py` | assets/cad_v3: the whole SR2 hand as built for the UR5, read from the owner's Onshape export of 2026-11-04 (assets/cad_v2/full_cad: the assembly "hand v3 (anthro config)", the gantries gantry_v3 and gantry_v3_thumb, and  |
| `cad_v3_film.py` | Stills and the film of the cad_v3 models, and the clocking of the hand read from the owner's photo of 2026-11-04, for the result page (scripts/cad/cad_v3_page.py, docs/experiments/20261104-cad_v3_hand/). |
| `cad_v3_page.py` | Builds docs/experiments/20261104-cad_v3_hand/20261104-cad_v3_hand_on_ur5.html: the SR2 hand as built on the UR5 CB3, its MuJoCo models and the transform from the flange to the palm (owner, 2026-11-04: "1) generate a full |
| `dip_b_four_compact_tip.py` | The 11/22 tip with four 3 x 2 mm magnets in a compact square centred on the board's centre chip U1, the tip that pad B05R4 is cut from. |
| `dip_b_magnet_layouts_figure.py` | Four and five magnets in concept b's 11/22 pad (2026-10-20 evening): the layouts over the board's five chips (palmar view) and the readout model's force resolution and magnet travel against the magnets' depth under the a |
| `dip_b_snap_fit_tips.py` | The 11/22 tips of pads 05R and 05R4 with snap-fit magnet slots, the tips that pads 05S and 05S4 are cut from. |
| `dip_b_tip_cap.py` | Caps over the tip end of DIP link concept b v3 (2026-10-20 evening): a PLA end plate over the finger's T-shaped tip end (the tongue and the cover, 21 x 5.1 mm, and the pad's flat end, 4.4 x 7.5 mm, all at z_F -98.46), tw |
| `dip_b_tip_cap_sketch.py` | Sketch of a T-shaped cap over the tip end of concept b v3 (2026-10-20 evening), drawn from dip_link_concept_b's numbers: the palmar view and the view from the tip end, with the pad, the cover with its walls and rails, th |
| `dip_b_v3_figures.py` | Figures comparing DIP link concept b v2 (printed 2026-10-19) with v3 (2026-10-20): the neck under the pad from the palmar side, the cover with its side walls, and the cable's leg toward the back from above; and a turntab |
| `dip_b_v3_ur5_v2_page.py` | Result page of 2026-10-20: DIP link concept b v3 and the UR5e calibration mount v2, the owner's notes after printing v2 b and mount v1 turned into geometry, with the checks that each change keeps the joint's clearance, t |
| `dip_b_v4.py` | DIP link concept b v4 and the wall-height sweep of 2026-10-21 16:20. |
| `dip_b_v4_page.py` | Result page of 2026-10-21: DIP link concept b v4 (the bridge's palmar edge recessed to a smaller column under the pad) and the sweep of retaining-wall heights (covers I15 with 4, 3 and 2 mm walls, caps with 4, 3 and 2 mm |
| `dip_fit_print_plates.py` | Print plates of the DIP-link fit tests: every test piece of a print run on one plate per printer, each piece with a printed ID (owner, 2026-10-19: "is there a more organized manner you can present the prints (and documen |
| `dip_fit_print_plates_page.py` | Print sheet of a DIP-link fit-test run, read at the printers on a phone: the two plates, every piece with its printed ID, the checks in order, the parameter each check sets, and the owner's records. |
| `dip_link_board_mount.py` | Final DIP link for the ReSkin 5X board (2026-10-18): concept a's window plate and dorsal clamp (judged against concepts b and c in docs/experiments/20261018-dip_link_board_mount/decision.json), built on the tips the owne |
| `dip_link_board_mount_page.py` | Page: the magnetometer board mount in the DIP link for the printed 85A fingertips (study 20261018-dip_link_board_mount; hand-off docs/handoff/20261018-calibration_and_dip_finish.md, step 5). |
| `dip_link_concept_a.py` | DIP link concept (a), window plate and dorsal clamp: a parametric generator and its checks (2026-10-18). |
| `dip_link_concept_b.py` | Concept b of the DIP-link redesign (2026-10-18): the ReSkin 5X board in a rigid cassette that takes the place of the printed tip's dorsal part, so the five chips sit inside the tip's former volume, closer to the magnets. |
| `dip_link_concept_c.py` | Concept c of the DIP-link redesign (2026-10-18): a tip cartridge printed in one job on the Prusa XL, PLA carrier and 85A tip, with the ReSkin 5X board placed at an M601 pause and captured by the PLA printed over it. |
| `dip_link_geometry.py` | Shared geometry of the DIP-link redesign for the five-chip magnetometer board (2026-10-18). |
| `dip_link_onshape_export.py` | STEP files for designing the DIP link in Onshape (owner, 2026-10-18 16:36: "i think concept A and B are both worthwhile, but need the qwiic connector correction. |
| `dip_link_params.py` | Parameter files of the DIP-link designs a and b (2026-10-18): every size, clearance and spacing the owner may change, applied to the generators at run time, with a reproduction check, the full clearance check of a parame |
| `dip_link_print_project.py` | Print project of a DIP link design for the Prusa XL 5T (hand-off docs/handoff/20261018-calibration_and_dip_finish.md, step 18): the DIP link, its clamp plate with the Qwiic pocket and the pin-fit coupon in PLA on tool 3, |
| `dip_link_spec.py` | Specification of the DIP link that clamps the ReSkin 5X board against the printed fingertip (2026-10-18). |
| `dip_link_v2.py` | Printable v2 of the DIP-link designs a and b (2026-10-18). |
| `dip_link_viewer.py` | Local viewer of the DIP-link designs a and b and their parameter files (2026-10-18). |
| `gcode_audit.py` | What a PrusaSlicer G-code file extrudes, layer by layer (2026-10-13). |
| `metric_magnet_prints.py` | Fingertip prints for the owner's 3 x 2 mm magnets (docs/handoff/20261016-metric_magnet_prints.md). |
| `tpu_print_audit.py` | Travel, retraction and per-part mass audit of PrusaSlicer G-code for the 85A prints (2026-10-19 night). |
| `v2_tip_fit_sizes_xl.py` | Concept a's 11/22 tip at four key-channel interferences and concept b's 11/22 pad at four rail-groove fits, 85A, on one Prusa XL plate (tool 2), without magnets, for the fit on the printed v2 links and cover. |
| `v2_tips_xl_plate.py` | Concept a's 11/22 tip and concept b's 11/22 pad (85A, five magnets) on one Prusa XL plate, tool 2. |
| `v2_ur5_pla_plate.py` | One Bambu Lab P1S PLA plate with the PLA parts of DIP-link v2 a and b and of the UR5e calibration mount. |

## cluster/ (6)

Jobs on the DeltaAI cluster.

| script | what it does |
|---|---|
| `deltaai_env_setup.sh` | One-time environment build on the DeltaAI LOGIN node (runbook step 3). |
| `deltaai_pull.sh` | Pull finished runs back down (runbook step 7). |
| `deltaai_push.sh` | Push the repo (code + scenes + the morphology runs a job needs) to DeltaAI. |
| `deltaai_queue.py` | Create, inspect and resume packed DeltaAI work queues. |
| `make_inline_breadth_queue.py` | Emit packed-queue manifests for the inline-hand questions that breadth can answer. |
| `make_seed_manifest.py` | Write the (design, seed) manifest that deltaai_seed_array.slurm consumes. |

## contact/ (82)

Fingertip contact models and simulators: the sphere pads, the contact bed, the reorientation on each backend, Drake, Newton, MuJoCo-Warp and DiffMJX.

| script | what it does |
|---|---|
| `contact_bed_brake.py` | Contact-model comparison bed, task 5 (T5 brake): the pinch as a friction hinge that lowers the tool. |
| `contact_bed_candidates.py` | Native-MuJoCo candidates for presliding compliance on the contact bed (2026-10-07), run through the bed's own task scripts by patching hom_contact_rig.make_rig for their specs. |
| `contact_bed_common.py` | Shared pieces of the contact-model comparison bed (docs/experiments/20261005-contact_bed/PROTOCOL.md). |
| `contact_bed_compliance.py` | Contact-model comparison bed, tasks 8 and 9 (2026-10-07): presliding compliance and load spread, against the elastic references of scripts/contact/contact_reference_laws.py (Mindlin, Hertz) and the pressure law. |
| `contact_bed_creep.py` | Contact-model comparison bed, task 6: creep of MuJoCo's soft friction under noslip and impratio. |
| `contact_bed_edge.py` | Contact-model comparison bed, edge test (2026-10-07): the pinch on a sharp feature. |
| `contact_bed_newton.py` | Contact-model comparison bed: Newton hydroelastic rows for the static pinch, T1 pull, T2 twist, T3 roll (when scripts/contact/contact_bed_roll.py exists), T4 shake and T5 brake. |
| `contact_bed_page.py` | Build docs/experiments/20261005-contact_bed/20261005-contact_model_bed.html from the bed's JSONL rows. |
| `contact_bed_pull.py` | Contact-model comparison bed, task 1: tangential pull-to-slip and pre-slip creep of a pinched tool. |
| `contact_bed_pull_consistency.py` | Contact-model comparison bed, task 1 (T1 pull) on the consistency set of the protocol. |
| `contact_bed_roll.py` | Contact-model comparison bed, task 3: the pinched tool rolled between a fixed pad and a pad sliding along z. |
| `contact_bed_shake.py` | Contact-model comparison bed, task 4: a pinched tool shaken along its own axis under gravity. |
| `contact_bed_stability.py` | Contact-model comparison bed, task 7: largest stable step of the sphere-pad pinch against d0 and tc. |
| `contact_bed_twist.py` | Contact-model comparison bed, task 2 (T2 twist): torque about the pinch axis until the pinched tool spins. |
| `contact_feedback_page.py` | Result page: sensitivity of the servo-plant reorientation policies to fingertip contact parameters, with analytical checks of the sphere pads (2026-10-13). |
| `contact_feedback_replay.py` | Closed loop against open loop under a swapped fingertip contact model, on the servo-limited plant (2026-10-13). |
| `contact_model_policies_page.py` | Build docs/experiments/20261008-contact_model_policies/20261008-fingertip_contact_model_policies.html. |
| `contact_overview_chain_figures.py` | Figures and numbers of the contact overview's first part, the chain from the printed fingertip to the simulated contact (docs/overviews/contact_model.html, section "chain"; owner, 2026-10-26: the contact overview as the  |
| `contact_overview_text.py` | Prose blocks of the contact overview page (scripts/docs/contact_overview_page.py) that summarise the bed's findings and the literature. |
| `contact_reference_laws.py` | Contact references that need no simulator, for the bed's compliance tasks T8 (tangential load cycle) and T9 (normal load sweep): the elastic solutions that the pressure law, the sphere pads and Drake hydroelastic leave o |
| `diffmjx_audit_page.py` | Build docs/experiments/20261011-diffmjx_audit/20261011-diffmjx_gradient_probe_audit.html. |
| `diffmjx_audit_probes.py` | CPU checks behind the 2026-10-11 audit of the 2026-10-07 DiffMJX and contact-gait studies. |
| `diffmjx_turn_onset.py` | Forward parity and gradient cost of the DiffMJX fork at the turn onset of the D1 continuation (2 mm packed pads). |
| `distributed_contact.py` | SR2 contact audit: frozen regression, static contact fields, and constraint scaling. |
| `distributed_contact_diagexact_tasks.py` | MuJoCo 3.14 hand-task ablation: exact diagonal versus impedance mapping. |
| `distributed_contact_drake.py` | Matched fixed-palm SR2 torsion in Drake SAP hydroelastic contact. |
| `distributed_contact_dynamics.py` | Isolated newer-MuJoCo flag, mass, acceleration and controlled-slip follow-up. |
| `distributed_contact_geometry_videos.py` | Film actual short guided sliding trajectories along/across curvature. |
| `distributed_contact_gpu.py` | mjlab Simulation (pinned 1.2 / MuJoCo Warp 3.6) matched hand benchmark. |
| `distributed_contact_newton.py` | Newton hydroelastic SR2 transfer; native mapping and explicit force-space adapter. |
| `distributed_contact_page.py` | Convert the source DOCX and generate the SR2 audit report from manifests/raw outputs. |
| `distributed_contact_pressure.py` | Separate Newton hydroelastic pressure quadrature from time integration. |
| `distributed_contact_task_page.py` | Build the task and foundational sliding report from native run summaries. |
| `distributed_contact_tasks.py` | Run the real screwdriver pickup/brake/insertion chain at matched controller settings. |
| `distributed_contact_timestep.py` | Restartable 50 Hz rolling-task sweep; isolated workers, native videos and live report. |
| `distributed_contact_transfer.py` | Articulated SR2 torsional transfer and backend throughput/fidelity benchmark. |
| `distributed_contact_transfer_page.py` | Build the articulated contact report and portable backend benchmark data. |
| `distributed_contact_video_page.py` | Synchronized video comparisons and a gallery built from real state captures. |
| `distributed_contact_video_snapshots.py` | Record explicitly static Newton pressure and 6 mm fillet snapshots for movies. |
| `distributed_contact_videos.py` | Capture real native trajectories and render contact movies without re-solving them. |
| `drake_sr2_hand.py` | Exploratory Drake model of the fixed-palm real_v1 SR² hand. |
| `drake_sr2_page.py` | Build the local Drake study page from probe.json; standard library only. |
| `drake_sr2_probe.py` | Reproduce the SR² → Drake feasibility study without changing the project env. |
| `flexcomp_skin_probe.py` | Compile probe: a MuJoCo flexcomp skin whose vertices play the role of the 1 mm pad spheres. |
| `gpu_scaling_mjwarp.py` | MuJoCo-Warp GPU scaling of the sphere-pad holding fixture (2026-10-05). |
| `hydroelastic_arm_integral.py` | Friction arm of the hydroelastic pressure law on the contact bed's geometry, integrated directly with no simulator: compliant fingertip sphere R = 10.55 mm with Drake's field p = E (1 - rho/R) against a rigid tool, every |
| `make_pad_morphology_run.py` | A morphology run whose frozen scene has the printed TPU fingertip instead of the legacy box tip. |
| `native_compliance_page.py` | Build docs/experiments/20261007-native_compliance/20261007-native_presliding_compliance.html. |
| `newton_hand_throughput.py` | GPU physics throughput of the hand holding the tool: MuJoCo-Warp against Newton, per fingertip contact model. |
| `newton_hydro_tests_page.py` | Build docs/experiments/20261007-newton_hydro_tests/20261007-newton_hydroelastic_friction_reduction_edge.html. |
| `newton_pinch_probe.py` | Newton hydroelastic stability probe for the SR2 fingertip pinch (2026-10-05). |
| `newton_scaling.py` | Newton hydroelastic GPU scaling on the SR2 holding fixture (2026-10-05). |
| `newton_turn.py` | The deployed open-loop three-finger turn in Newton with hydroelastic TPU fingertips (2026-10-06). |
| `pad_builder_check.py` | Pad builder v1 against v2 and Drake hydroelastic at trajectory level on the real_v1 hand (2026-10-15, before builder v2 became the default of reorient_backends.replace_tips). |
| `pad_contact_rig.py` | Rig for the contact laws of the production fingertip pads (reorient_backends.replace_tips, the default 1 mm TPU-tip pads), shared by tests/test_pad_contact_laws.py and the measurements of the 2026-10-13 contact-feedback  |
| `pad_mass_calibration.py` | Tool mass against the 1 mm sphere-pad calibration: fixed Lambda versus a load-time per-object Lambda. |
| `pad_usability_agree.py` | Scores of experiment U2 of the 2026-10-14 pad-usability study: agreement of six MuJoCo fingertip models with Drake hydroelastic on the eight deployed plan replays (servo-limited and working plant), the same at the RL tra |
| `pad_usability_backends.py` | Plan replays and HOM closed-loop turns of the eight deployed hands D1-D8 in six MuJoCo fingertip models and in Drake hydroelastic (experiment U2 of the 2026-10-14 pad-usability study; plan in docs/notes/20261014-contact_ |
| `pad_usability_bed.py` | Two-pad block bed at the RL trainer's numerics (experiment U2 (g) of the 2026-10-14 pad-usability study). |
| `pad_usability_cost_report.py` | Tables, key numbers and figures of U1 (per-step cost of the fingertip contact models) from its rows. |
| `pad_usability_film.py` | Film of the three-finger grip on the 16 mm cylinder (scale-study cell cylinder:16:0.625, servo-limited plant) with three fingertip models side by side: 1 mm sphere pads, 2 mm sphere pads and the TPU block mesh with condi |
| `pad_usability_gpu_cost.py` | GPU cost per world-step of the fingertip contact models at the servo plant's held RL state (U1, 2026-10-14). |
| `pad_usability_page.py` | Page: cost and control outcomes of the sphere-pad fingertip on the real_v1 hand (experiments U1-U5, 2026-10-14). |
| `pad_usability_planner_model.py` | Pinned iCEM on the 1 mm pad plant with a cheaper rollout model of the fingertip (U4 of the 2026-10-14 pad-usability study; plan in docs/notes/20261014-contact_model_assessment_plan.md). |
| `pad_usability_planner_report.py` | Scores, table and figure of experiment U4 of the 2026-10-14 pad-usability study: pinned iCEM planning on the 1 mm pad plant with its own pad model (icem_pin), with point contact (icem_pin_pt) and with condim 4 (icem_pin_ |
| `pad_usability_rl.py` | Servo-plant RL scenes of the cheaper fingertip models and short PPO probes of their training cost (U1, 2026-10-14). |
| `pad_usability_rl_report.py` | Scores, tables and figure of experiment U5 of the 2026-10-14 pad-usability study: the four servo-plant finals (TPU mesh and 1 mm pads, seeds 0 and 1) on four plant scenes under four conditions, closed loop and as open-lo |
| `pad_usability_step.py` | Largest stable and accurate physics step of the 1 mm sphere pads and of point contact on the real_v1 hand (experiment U3 of the 2026-10-14 pad-usability study; plan in docs/notes/20261014-contact_model_assessment_plan.md |
| `pad_usability_step_report.py` | Scores, tables and figures of experiment U3 of the 2026-10-14 pad-usability study: the largest stable and the largest accurate physics step of the 1 mm sphere pads, 2 mm pads, point contact and condim 4 on the real_v1 ha |
| `pad_usability_timing.py` | CPU cost per physics step of the fingertip contact models on the real_v1 hand at the scale study's three-finger grip. |
| `pads_cpu_gpu_consistency.py` | CPU MuJoCo against MuJoCo-Warp on the 1 mm sphere-pad holding fixture (2026-10-05). |
| `pads_hold_scaling.py` | Same-fixture control for scripts/contact/newton_scaling.py: 1 mm MuJoCo-Warp sphere pads on the hold. |
| `profile_contact_mapping.py` | Measure scalar mapping, duplicate constraint refresh and native solver stages. |
| `reorient_backends.py` | The deployed three-finger reorientation of the eight bench hands across contact models and simulators. |
| `reorient_backends_films.py` | Films of the bench reorientation in each contact model: one hand, one seed, every condition side by side. |
| `reorient_backends_films.sh` | Films of the 2026-10-06 reorientation comparison: per hand, the five MuJoCo panels in one process, each Drake panel in a process of its own (a Drake diagram is not freed inside one process), then the 3x3 tile. |
| `reorient_backends_gpu.py` | The bench reorientation of reorient_backends.py on MuJoCo-Warp: one batch of seeds per hand and condition. |
| `reorient_backends_grid.sh` | The 2026-10-06 grid of scripts/contact/reorient_backends.py: eight bench hands, seeds 0-8 (2 mm / 2 deg jitter), plant `fast` unless named, friction 2.4 unless named. |
| `reorient_backends_summary.py` | Tables of the reorientation grid (reorient_backends*.py): per hand and condition, the fraction of seeds holding the tool at the end of the bench hold and after 4 s more, the net turn and its spread over seeds, and the be |
| `same_state_timing.py` | Physics time per world-step of MuJoCo-Warp and Newton from one held state of the RL training env (2026-10-06). |
| `simulator_agreement_figure.py` | Agreement of fingertip contact models with Drake hydroelastic on three tasks (2026-10-06). |
| `sprung_skin_page.py` | Result page: presliding compliance from a sprung skin under the 1 mm sphere pads (2026-10-12). |

## contact_surface/ (9)

A package for the distributed-contact study of 2026-10-04 (`import contact_surface`).

| script | what it does |
|---|---|
| `fillet.py` | Front face of a rectangular tip with a measured 6 mm edge roundover. |
| `geometry.py` | The existing hom_chain cap formula, with its two area conventions made explicit. |
| `metrics.py` | Forces on the object, world frame; moments about (0, 0, pad radius). |
| `mujoco_model.py` | Fixed pad, free rigid indenter. |
| `records.py` | Restartable run directories; CSV/gzip fallback requires no optional dependencies. |
| `runtime.py` | Dense audit adapter for current and pinned MuJoCo APIs. |
| `scaling.py` | Physical normal coefficients from MuJoCo's constant-impedance parameterization. |
| `slip.py` | Force-controlled sliding and normal acceleration in the isolated contact fixture. |
| `video.py` | Recorded simulation state and native contact data for independent visual replay. |

## design/ (56)

Finger layouts and their search: the morphology generators, IK retargeting of the grasp keyframe, CEM grasp synthesis, the real_v1 Sobol pipeline and its screens, KaRMA and GraspGenX.

| script | what it does |
|---|---|
| `build_hand_urdf.py` | Generate a URDF of the hand for Lightning Grasp. |
| `fit_real_v1_pose.py` | Author the grasp keyframe for one `real_v1` design — palm pose AND finger angles together. |
| `generate_morphology_xml.py` | - |
| `generate_sphere_packed_scene.py` | Rewrite a frozen scene's collision primitives as a dense sphere packing. |
| `graspgenx_eval_phase1.py` | Score GraspGenX grasps on the real hand via Phase1GraspEvaluator. |
| `graspgenx_make_hand.py` | Non-interactive GraspGenX gripper exporter for the hand (MuJoCo-based). |
| `graspgenx_view_grasps.py` | Headless visual validation for GraspGenX hand grasps. |
| `ik_recem_landscape.py` | IK-retargeted re-CEM of the whole morphology landscape (2026-07-01). |
| `morph_landscape_sweep.py` | Morphology LANDSCAPE characterization (2026-06-23) — the original project goal. |
| `morph_pipeline_plots.py` | Full A->B pipeline sweep — analysis plots + markdown summary (2026-07-03). |
| `morph_pipeline_sweep.py` | Full per-morphology A->B pipeline sweep — the HONEST, health-gated design search (2026-07-03). |
| `morph_selfcollision_gate.py` | Is this perp morphology PHYSICALLY REAL? |
| `paper_morphology_diversity.py` | Design-space diversity figures: how far apart are the eight deployed real_v1 hands? |
| `phase1_optimize_grasp.py` | - |
| `pose_open_keyframe.py` | Author an `open` keyframe by IK-ing each fingertip to an explicit WORLD target. |
| `probe_hold_convergence.py` | Does the open-loop screen's "held" verdict survive a longer hold? |
| `probe_perp_mechanism.py` | Scripted open-loop probe of the perpendicular-hand grasp/reorient mechanism. |
| `probe_perp_thumb_engage.py` | Scripted viability probe: can the opposed-pair hand be made to REORIENT ON THREE FINGERS? |
| `probe_thumb_reach.py` | Can the thumb REACH the shaft after the perp hand has reoriented it to vertical? |
| `real_v1_budget_confirm.sh` | Stages 2-4 of the clip re-screen: choose one operating point per hand, then earn it. |
| `real_v1_budget_rescreen.sh` | Re-screen the Sobol-4096 population across the RESIDUAL CLIP, not at one value of it. |
| `real_v1_cem_filmstrip.py` | Filmstrips of CEM grasp samples on the deployed `real_v1` hands, open keyframe to hold. |
| `real_v1_contact_gait_page.py` | Build the dated contact-gait result page from raw experiment JSON. |
| `real_v1_contact_gait_search.py` | Search short release-and-recontact plans on the deployed D1 hand. |
| `real_v1_contact_gait_transfer.py` | Replay contact-gait plans across tips, contact models, and grip jitters. |
| `real_v1_contact_gait_warp.py` | Replay held-state contact gaits in MuJoCo-Warp parallel worlds. |
| `real_v1_control_diagnosis_page.py` | Build the control-diagnosis page from the ablation and sweep JSON. |
| `real_v1_design_search.py` | Sample `real_v1` hands, score each one, and measure the reorient it actually produces. |
| `real_v1_diffmjx_gate.py` | Check MJX contact dynamics at a replayed real_v1 grasp before optimizer studies. |
| `real_v1_diffmjx_local_search.py` | Compare local yaw-control search on the real_v1 1 mm sphere-pad D1 grasp. |
| `real_v1_diffmjx_official_gate.py` | Probe the released DiffMJX fork on a real_v1 held screwdriver state. |
| `real_v1_diffmjx_pad_cpu.py` | Compare full and tool-only 1 mm sphere-pad contacts in the D1 held state. |
| `real_v1_diffmjx_pad_jvp.py` | DiffMJX directional contact derivatives on the packed real_v1 D1 grip. |
| `real_v1_diffmjx_page.py` | Build the dated real_v1 contact-gradient result page from its JSON measurements. |
| `real_v1_diffmjx_recontact_jvp.py` | Measure a local DiffMJX control derivative immediately before index recontact. |
| `real_v1_fetch_scores.py` | Cache the operator's manual scores off the CB1 so the transfer analysis is reproducible. |
| `real_v1_filmstrip.py` | Filmstrips from the bench tape, for showing what a hold and a drop actually look like. |
| `real_v1_filter_reachable.py` | Keep only the sampled hands whose three mounts fit the MEASURED gantry travel. |
| `real_v1_grasp_closure.py` | Where the pads sit on the shaft, measured at closure, for every deployed plan. |
| `real_v1_grasp_refit.py` | Why a plan that reorients 4/4 in the deploy screen picks up nothing in the chain. |
| `real_v1_obs_sources.py` | Every column of the policy's observation, and what on the bench could supply it. |
| `real_v1_pad_elevation_page.py` | Build the pad-elevation result page from the studies' own JSON. |
| `real_v1_pipeline.py` | Per-design pipeline for the `real_v1` (CAD-matched hardware) hand: design -> scene -> pose -> CEM grasp -> Policy A -> Policy B -> handoff eval. |
| `real_v1_plan_angle_sweep.py` | The bench maneuver's commanded angle on the calibrated plant. |
| `real_v1_plan_band.py` | What each SHIPPED plan does as its residual clip moves -- the band, on its own cell. |
| `real_v1_plan_catalog.py` | Write the station's catalog.json: what simulation expects from each plan on the bench. |
| `real_v1_prepare_retention.py` | Merge population grasp shards and preserve each grasp's exact fitted geometry. |
| `real_v1_promote_designs.py` | Take chosen designs through every gate that stands between a screen row and a bench run. |
| `real_v1_pusher_reach.py` | Can the middle finger act as a pusher about the thumb-index pinch axis? |
| `real_v1_screen_chain_page.py` | Build the screen-vs-chain result page from the studies' own JSON. |
| `real_v1_select_budget.py` | Pick one (grasp cell, residual clip) operating point per morphology, and say what the clip recovered. |
| `real_v1_sobol8192.sh` | Extend the Sobol population from 4,096 hands to 8,192, and screen the new half. |
| `real_v1_tip_gait.py` | Two-tip gait on the bench maneuver: tip, rest, regrip, tip again. |
| `render_real_v1_retention.py` | Render confirmed torque-capped proof-lift/hold rollouts. |
| `retarget_keyframe_ik.py` | Retarget the 'about-to-contact' open keyframe across morphologies via fingertip IK. |
| `sweep_perp_thumb.py` | Can a perp morphology recruit the THUMB onto the reoriented shaft — without losing the swing? |

## docs/ (27)

Builders of the home page, the topic overviews and the page indices, and the parts every page builder uses: the page style, LaTeX to SVG, the prose check and the figure check.

| script | what it does |
|---|---|
| `check_scripts.py` | Static checks of scripts/ after the move into topic folders (2026-11-03). |
| `contact_overview_page.py` | Build docs/overviews/contact_model.html, the topic overview of the fingertip contact model. |
| `docs_changelog.py` | Revisions of the documentation pages (owner, 2026-11-02: "we should also have a changelog for these pages ... |
| `docs_home_page.py` | Build docs/index.html, the home page of the documentation: the hand in two sentences, the topics with their state and links, how to fetch the result pages, and where the code is; and docs/pages.html, the index of every r |
| `docs_index.py` | Indices of the documentation, rebuilt from the result pages themselves (owner, 2026-10-20: "we dont have a good way to automate index updates ... |
| `docs_inventory.py` | Inventory of the result pages under docs/experiments: date, file, title, topic, builder, artifact. |
| `interactive_figure.py` | Interactive data figures for the result pages: the default view is an inline SVG drawn at build time, and a small page script adds a hover readout, series toggles, x-zoom by drag and a CSV download of the figure's data. |
| `mj_snap.py` | Render a MuJoCo scene to PNG(s) so an agent can *look* at it. |
| `mjwasm_embed.py` | MuJoCo viewer for static result pages: MuJoCo 3.14.0 (WebAssembly) and three.js r147, inlined. |
| `overview_chain_text.py` | Text of the topic overview docs/overviews/grasp_to_gait_chain.html (builder: scripts/docs/topic_overview_page.py). |
| `overview_code_text.py` | Text of the overview docs/overviews/code_and_methods.html (builder: scripts/docs/topic_overview_page.py code). |
| `overview_design_text.py` | Text of the topic overview docs/overviews/morphology_design.html (builder: scripts/docs/topic_overview_page.py). |
| `overview_figures.py` | Interactive data figures of the topic overviews (scripts/docs/topic_overview_page.py), drawn from the rows the detailed pages were built from; no simulation runs here. |
| `overview_fingertip_figures.py` | Figures of the printed-fingertips overview that no result page draws as an image (scripts/docs/overview_fingertip_text.py). |
| `overview_fingertip_text.py` | Text of the topic overview docs/overviews/printed_fingertips.html (builder: scripts/docs/topic_overview_page.py). |
| `overview_hardware_text.py` | Text of the topic overview docs/overviews/hand_plant_bench.html (builder: scripts/docs/topic_overview_page.py). |
| `overview_mechanisms_text.py` | Text of the topic overview docs/overviews/reorientation.html (builder: scripts/docs/topic_overview_page.py). |
| `overview_policies_text.py` | Text of the topic overview docs/overviews/rl_policies.html (builder: scripts/docs/topic_overview_page.py). |
| `page_figure_check.py` | Figure check for the result pages: one screenshot per figure with the page's own styles, and the chart text as displayed, in headless Chrome. |
| `prose_gate.py` | Mechanical gate of a wording pass over a built page (owner, 2026-10-22: a cheaper model edits prose only "if we can automate the checks well"). |
| `prose_lint.py` | Flag the phrasings the owner's Writing rules (~/.claude/CLAUDE.md) ban, and count words, in a page or a text file; measure a page's sentences and flag the ones to rewrite under the elements of style. |
| `reorganize_videos.py` | Reorganize docs/rl/videos into TIMESTAMPED experiment folders, temporally ordered. |
| `retro_style.py` | Plain 1990s-homepage style for the result pages: Times on white, default link colours, 1px black rules. |
| `scripts_map.py` | Write scripts/README.md, the map of scripts/: one table per folder, each script with the first sentence of its docstring (or of its leading comment, for a shell script). |
| `texsvg.py` | Pre-render LaTeX math to inline SVG for static HTML result pages: no JavaScript, CDN or font files. |
| `topic_overview_page.py` | Build the topic overviews, docs/overviews/<file> (names in docs_inventory.OVERVIEW): what a collaborator new to the project needs from one topic of the result pages, with every finding linked to the page that measured it |
| `workshop_briefing_page.py` | Build the workshop briefing page: the record behind the SR2 Hand paper. |

## fingertip/ (148)

The printed fingertip: its geometry and prints, finite elements, magnets and the MLX90393 readout, the wrench model, and the calibration and crosstalk pages.

| script | what it does |
|---|---|
| `crosstalk_feed_watch.py` | Watcher of the UR5 coordinator's crosstalk results for the doc-writing job (owner, 2026-10-27: "a periodic poller ... |
| `curved_tips_analysis.py` | Summaries of the curved-tip study's rows (docs/handoff/20261015-curved_tips.md); each subcommand reads rows under docs/experiments/20261015-curved_tips/ and writes one JSON summary next to them. |
| `curved_tips_bed.py` | Contact-bed cases of the curved-tip study (docs/handoff/20261015-curved_tips.md, step 2): the two-pad pinch rig of the contact bed (two force-controlled rails squeeze the 25 mm, 24.5 g screwdriver, mu 1, 1 ms) with a cur |
| `curved_tips_cell_drake.py` | Ledger step 17 of the curved-tip study (docs/handoff/20261015-curved_tips.md): the CAD hand's 10 mm cylinder cell (cylinder:10:0.75 of hand_object_scale_sim's diagonal family) with the as-built block, turned by the HOM c |
| `curved_tips_common.py` | Members, labels and paths of the curved-fingertip study (docs/handoff/20261015-curved_tips.md): capsule and half-ellipsoid tips (tip_codesign_geometry.CurvedTip) against the printed block of the 2026-10-14 co-design stud |
| `curved_tips_cost.py` | Cost of the curved fingertips on the hand (docs/handoff/20261015-curved_tips.md, step 3), by the protocol of U1 on the pad-usability page (docs/experiments/20261014-pad_usability/: scripts/contact/pad_usability_timing.py |
| `curved_tips_fe.py` | Sensing of the quarter-ellipsoid tips (docs/handoff/20261015-curved_tips.md, ledger step 13): the magnet's travel per newton from finite elements of the whole printed tip, and the five-chip board's force noise with the b |
| `curved_tips_figures.py` | Figures of the curved-tip page (docs/experiments/20261015-curved_tips/), drawn from the study's summaries and shapes, in the style of scripts/fingertip/tip_codesign_figures.py (9.5 pt at 150 dpi on 7.2 in wide figures, a |
| `curved_tips_film.py` | Films of the curved-tip study (docs/handoff/20261015-curved_tips.md, step 5), rendered on the CPU through Mesa's software EGL with the 2026-10-14 study's film pieces (scripts/fingertip/tip_codesign_film.py: 640 x 480 pan |
| `curved_tips_law.py` | Quasi-static contact laws of the curved fingertips against the bed's tool cylinder (R 12.5 mm): the references of the curved-tip study (docs/handoff/20261015-curved_tips.md, step 1), computed with tip_codesign_law's fiel |
| `curved_tips_nail.py` | Fingernail on the quarter-ellipsoid tips (docs/handoff/20261015-curved_tips.md, ledger step 13): a rigid plate on the tip's flat back reaching past its distal end, and whether the bench tasks touch it. |
| `curved_tips_page.py` | Page: capsule, half-ellipsoid and quarter-ellipsoid fingertips against the printed block on the SR2 hand (study 20261015-curved_tips; hand-off docs/handoff/20261015-curved_tips.md; protocol of docs/notes/20261014-tip_cod |
| `curved_tips_print.py` | Printable files of the curved fingertips (docs/handoff/20261015-curved_tips.md, steps 1 and 6). |
| `curved_tips_queue.py` | Job queues of the curved-tip task screen (docs/handoff/20261015-curved_tips.md, step 5), the 2026-10-14 stages (docs/notes/20261014-tip_codesign_plan.md, steps 7-9) for the curved candidates, the as-built block and the b |
| `curved_tips_rig.py` | tip_codesign_rig (hand-level runs: plan replays, HOM turns on D1-D8, the scale study's CAD-hand cells, films and RL scenes with a chosen tip) for the curved-tip study, with its scene caches under logs/20261015-curved_tip |
| `curved_tips_stiffness.py` | Stiffness maps of four fingertips and how well each contact model reproduces them (ledger step 16 of the curved-tip study, docs/handoff/20261015-curved_tips.md; owner, 2026-10-15 16:20). |
| `curved_tips_traj.py` | Contact dynamics compared on trajectories (docs/handoff/20261015-curved_tips.md, step 4; owner, 2026-10-15: "held" is too coarse to show that two contact models give the same dynamics). |
| `curved_tips_traj_analysis.py` | Trajectory divergence between contact models (docs/handoff/20261015-curved_tips.md, step 4), from the traces of scripts/fingertip/curved_tips_traj.py (traj/traces.jsonl). |
| `eflesh_build.py` | An eFlesh lattice part from a closed shape in one command, ready for PrusaSlicer (2026-10-13). |
| `eflesh_printability.py` | Printability of eFlesh lattices and of slicer gyroid on the owner's PrusaSlicer FLEX profile (2026-10-13). |
| `eflesh_printability_page.py` | Page: printability, stiffness and a one-command build of eFlesh lattice fingertips (2026-10-13). |
| `eflesh_tip_fem.py` | Stiffness and magnet travel of a printed fingertip from its G-code, on the GPU (2026-10-13). |
| `fe_contact_calibration_fe.py` | Finite elements of the four 85A printed fingertips under the calibration probes (phase A of docs/experiments/20261018-fe_contact_calibration/plan.json). |
| `fe_contact_calibration_fit.py` | Model curves of Drake's field and the pads under the finite-element probes, and the three calibrations (phase B of docs/experiments/20261018-fe_contact_calibration/plan.json). |
| `fe_contact_calibration_page.py` | Page: calibration of Drake's pressure field and the 1 mm pads to finite elements of the 85A printed fingertips (docs/experiments/20261018-fe_contact_calibration/, plan.json). |
| `fe_contact_calibration_prior.py` | Numbers behind the finite-element calibration plan (docs/experiments/20261018-fe_contact_calibration/plan.json), computed from rows of earlier studies; no new simulation. |
| `fe_contact_calibration_sim.py` | Simulator runs of the finite-element calibration (phases C, D1 and D2 of docs/experiments/20261018-fe_contact_calibration/plan.json): the calibrated moduli of fit/fits.json in Drake 1.57 and in the 1 mm pads, on the cont |
| `fe_contact_calibration_summary.py` | Summary of the finite-element calibration (docs/experiments/20261018-fe_contact_calibration/): finite elements, fits, bed, the iCEM command replays (D1) and the D1-D8 replays (D2), with the plan's criteria applied per ti |
| `fingertip_backends_page.py` | Build docs/experiments/20261006-fingertip_backends/20261006-fingertip_contact_backends.html. |
| `fingertip_calibration_cell_render.py` | Figure 1 of the fingertip calibration page: the UR5 cell rendered in MuJoCo, with the pad at two touch-offs on the low sphere (owner, 2026-10-27: the coordinator's in-MuJoCo render of the calibration scene, "you can mayb |
| `fingertip_calibration_figures.py` | Figures and numbers of the page on the fingertip's calibration on the UR5 (docs/experiments/20261026-fingertip_calibration/): the six motions of a session, the press targets on the pad, the learning set-up, location erro |
| `fingertip_calibration_page.py` | Page on the calibration of the magnetic fingertip on the UR5 against the HEX-E (owner, 2026-10-26: "the calibration process, what motions we initiated, geometry of the calibration and the desired sensing result ... |
| `fingertip_calibration_text.py` | Prose of the page on the fingertip's calibration on the UR5 (scripts/fingertip/fingertip_calibration_page.py), kept apart from the plotting. |
| `fingertip_contact_estimation_page.py` | Result page of the model-based contact estimation study (2026-10-20): the measurement model of the five-chip board under the magnetic pad, its calibration at rest, the FE contact basis over the pad's surface, the estimat |
| `fingertip_contact_estimation_study.py` | Numbers of the model-based contact estimation study (2026-10-20), for its result page. |
| `fingertip_contact_film.py` | Film of the browser contact viewer (src/sr2hand/driver/reskin5x/host/wrench_viewer.py): headless Chrome opens the viewer's page and the DevTools protocol captures it at a fixed rate; ffmpeg encodes the frames. |
| `fingertip_crosstalk.py` | Crosstalk between two magnetic fingertips: the field of one tip's magnets at the other tip's chips, the contact that the viewer's estimator shows for it, and what is left once that field is predicted from the two tips' r |
| `fingertip_crosstalk_holder.py` | Crosstalk between v3 tips on the fingertip holder, measured on the UR5 and compared with the dipole model (owner, 2026-10-27: "a super complete calibration suite + crosstalk analysis"; the doc-writing session takes the r |
| `fingertip_crosstalk_page.py` | Page on magnetic crosstalk between neighbouring printed fingertips (hand-off 20261027-docs_update_impratio, step 4; owner 2026-10-27: "crosstalk is going to be problem, so itd be helpful to present some compelling result |
| `fingertip_drift_film.py` | Film of the contact viewer through the board's warm-up and two touches, with the settings of 2026-10-20 above and those of 2026-10-21 below: the recording of 2026-10-21 09:59 (data/20261021-095907_touches_records.npz, wr |
| `fingertip_foreign_field.py` | Rejection of fields from outside the pad (steel near the tip, another magnet) by the contact estimator: the test of fingertip_wrench_model.ForeignField on the owner's touches of a steel mass and a steel ruler on the seco |
| `fingertip_foreign_field_film.py` | Film of the contact viewer replaying the owner's six touches of steel on the second fingertip's face (2026-10-23 15:43:17-15:43:42, from data/20261023-153800_touch_records.npz of 20261023-foreign_field_rejection): first  |
| `fingertip_foreign_field_page.py` | Page of the test for fields from outside the pad in the fingertip's contact estimator (2026-10-23): the owner's touches of steel on the second fingertip, why the estimator reads them as presses, the test (fingertip_wrenc |
| `fingertip_foreign_field_text.py` | The prose of the page on fields from outside the pad (scripts/fingertip/fingertip_foreign_field_page.py), from the study's data. |
| `fingertip_geometry.py` | The SR2 fingertip as built: a printed TPU block with rounded front edges, its sphere-packed pad, and writers for MuJoCo (mesh or sphere pads), Drake (compliant convex) and Newton (mesh SDF). |
| `fingertip_holder.py` | Fingertip holder for the UR5 cell (owner, 2026-10-27): two fingertips (pad, cassette 3T, top cover TW15H3 trimmed flush with the walls, and the ReSkin 5X board) held still while v2 ID 1 moves around them without contact, |
| `fingertip_hybrid_cv.py` | The physical model with a learned correction whose inputs are the physical estimate itself (contact point and normal force), on the real presses of the UR5 run sphere2: four folds, each holding out every fourth grid pres |
| `fingertip_magnet_transfer_sim.py` | Transfer of the fingertip's contact predictors across magnet strength and pole pattern, in simulation (owner, 2026-10-23: "how transferable do you think this result will be to the higher strength n52 magnets + the checke |
| `fingertip_physical_model_fe.py` | Sections of the magnetic fingertip's finite-element model under a press, for the page on the fingertip's physical model (owner, 2026-10-26: "clearly understand the role and implementation of FEA, its relation to the prin |
| `fingertip_physical_model_figures.py` | Figures and numbers of the page on the magnetic fingertip's physical model (docs/experiments/20261026-fingertip_physical_model/): the pad in section from the slicer's material classes, the finite elements under a press a |
| `fingertip_physical_model_page.py` | Page on the physical model of the magnetic fingertip (owner, 2026-10-26: "clearly understand the role and implementation of FEA, its relation to the printed tip, and it + the magnetic field sensing composing the physical |
| `fingertip_physical_model_text.py` | Prose of the page on the magnetic fingertip's physical model (scripts/fingertip/fingertip_physical_model_page.py), kept apart from the plotting. |
| `fingertip_physical_model_v3_check.py` | The physical model of the printed fingertip on the four-magnet tips v3 ID 1 to ID 3 beside the five-magnet v2 ID 1 (2026-10-29, after v3 ID 2's night suite of 10-28/29; v3 ID 3 added on 10-30, after its suite of 10-29/30 |
| `fingertip_print_procedure_figures.py` | Figures of the 85A fingertip print procedure (docs/handoff/20261018-print_procedure.md, step 2): each print geometry on the Prusa XL bed and, for each pause, the part as printed up to that pause with its magnets seated. |
| `fingertip_print_procedure_page.py` | Print procedure for the Siraya Tech 85A magnet fingertips on the Prusa XL (docs/handoff/20261018-print_procedure.md, step 2), read at the printer on a phone. |
| `fingertip_second_tip_film.py` | Film of the contact viewer replaying the second fingertip's six centre presses of 2026-10-23 12:27 (the recording data/20261023-122717_round2_records.npz, written by scripts/fingertip/fingertip_second_tip_transfer.py --s |
| `fingertip_second_tip_transfer.py` | The UR5 run sphere2's calibration, made on the robot fingertip (board 9c:d0, all five poles one way), applied to a second fingertip on the workstation (board d9:94, poles (-1, -1, 1, 1, 1)), from the live viewer's stream |
| `fingertip_second_tip_transfer_page.py` | Page of the second fingertip against the calibration of the UR5 fingertip (2026-10-23): the two pads' pole patterns and rest fits, the transfer in simulation, the owner's presses on the desk and each predictor's answers. |
| `fingertip_second_tip_transfer_text.py` | The prose of the second-fingertip page (scripts/fingertip/fingertip_second_tip_transfer_page.py), from the two rounds' results. |
| `fingertip_shape_rerun_analysis.py` | Summaries of the fingertip-shape rerun's rows (docs/handoff/20261103-fingertip_shape_rerun.md); each subcommand reads rows under docs/experiments/20261104-fingertip_shape_rerun/ and writes one JSON summary beside them. |
| `fingertip_shape_rerun_band.py` | Worst-case score over a band of scene frictions for the stage-3 tips of the fingertip-shape rerun (ledger step 12): the band option of scripts/hom/hom_grasp_refine.py (hand-20261104-hom_worst_case_score) with each tip in |
| `fingertip_shape_rerun_bed.py` | The contact bed of the fingertip-shape rerun (docs/handoff/20261103-fingertip_shape_rerun.md, ledger step 3): the cases of 2026-10-15 (scripts/fingertip/curved_tips_bed.py, curved_tips_law.py) at 2.5 MPa, the scenes' mod |
| `fingertip_shape_rerun_common.py` | Members, labels, paths and settings of the fingertip-shape rerun (docs/handoff/20261103-fingertip_shape_rerun.md): the screens of 2026-10-14 (blocks) and 2026-10-15 (capsules, half- and quarter-ellipsoids) and the FE che |
| `fingertip_shape_rerun_cost.py` | Simulation cost of the fingertip-shape rerun's tips (docs/handoff/20261103-fingertip_shape_rerun.md, ledger step 4), by the protocol of U1 on the pad-usability page (docs/experiments/20261014-pad_usability/) as scripts/f |
| `fingertip_shape_rerun_fe.py` | Finite elements of the fingertip-shape rerun's tips (docs/handoff/20261103-fingertip_shape_rerun.md, ledger step 6), through the machinery of the 10-19/20 checks of the 11/22 (scripts/fingertip/v2_tip_fe_checks.py) with  |
| `fingertip_shape_rerun_figures.py` | Figures of the fingertip-shape rerun page (docs/experiments/20261104-fingertip_shape_rerun/), from the run's summaries, in the style of scripts/fingertip/curved_tips_figures.py (9.5 pt at 150 dpi on 7.2 in wide figures,  |
| `fingertip_shape_rerun_film.py` | Films of the fingertip-shape rerun (docs/handoff/20261103-fingertip_shape_rerun.md, ledger step 7): the HOM turn on a cad_v2 cell per tip, rendered on the CPU through Mesa's software EGL with the film pieces of scripts/f |
| `fingertip_shape_rerun_mount.py` | Does each tip of the fingertip-shape rerun mount on the v4b finger as the 11/22 pad does (ledger step 2)? |
| `fingertip_shape_rerun_page.py` | Page: block, capsule, ellipsoid and hemisphere fingertips on the v4b finger of the SR2 hand, the rerun of the fingertip-shape screens of 2026-10-14 and 10-15 and the FE checks of 10-18/19 at 2.5 MPa on the current hand ( |
| `fingertip_shape_rerun_queue.py` | Job queue of the fingertip-shape rerun (docs/handoff/20261103-fingertip_shape_rerun.md), read by the workers logs/<run>/queue/worker.sh (one job line at a time under flock; each job through resguard and taskset on the wo |
| `fingertip_shape_rerun_rig.py` | Hand-level runs of the fingertip-shape rerun (docs/handoff/20261103-fingertip_shape_rerun.md, ledger step 5): the task screen of 2026-10-14 and 10-15 on the v4b hand, at 2.5 MPa and builder v2, with each tip in place of  |
| `fingertip_shape_rerun_traj.py` | The D1-D8 bench-plan replays of the fingertip-shape rerun (ledger step 5, the Drake agreement gate): scripts/ curved_tips_traj.py's arms on the SCS0009 plant at friction 1.0, with the pads (builder v2) and Drake's compli |
| `fingertip_wrench_fe.py` | FE basis of the contact estimator: magnet motions and palmar-face displacements per newton of patch load on a dense grid over a pad, for scripts/fingertip/fingertip_wrench_model.py and the live viewer. |
| `fingertip_wrench_model.py` | Measurement model of the magnetic fingertip on the ReSkin 5X board, its calibration from the reading at rest, and the contact estimator built on it. |
| `mlx90393_model.py` | MLX90393 readout model from the Melexis datasheet (revision 012, 18 Nov 2025), and the force each printed fingertip resolves with it (2026-10-13). |
| `mlx90393_readout_page.py` | Page: MLX90393 readout of printed fingertips on the five-chip board, with one or four embedded magnets (2026-10-13). |
| `printed_fingertip_status_page.py` | Status page of the printed-fingertip work: docs/experiments/20261018-printed_fingertip_status/ 20261018-printed_fingertip_and_dip_link_status.html. |
| `reskin5x_pad_fit.py` | Polarities and position of a pad's magnets, fitted to the static field the ReSkin 5X board reads. |
| `sim2real_fingertip_analysis.py` | Summaries of the fingertip deployment-readiness study (docs/handoff/20261016-sim2real_fingertip_readiness.md) from its rows, for the result page (scripts/fingertip/sim2real_fingertip_readiness_page.py). |
| `sim2real_fingertip_bed.py` | Contact on the bed for the printable 85A fingertips (docs/handoff/20261016-sim2real_fingertip_readiness.md, step 4). |
| `sim2real_fingertip_bench.py` | Bench procedure and first-bench package of the fingertip deployment-readiness study (docs/handoff/ 20261016-sim2real_fingertip_readiness.md, steps 8 and 9). |
| `sim2real_fingertip_cell10.py` | Pad builders v1 and v2 against Drake on the CAD hand's 10 mm cylinder cell, per printable variant (docs/handoff/20261016-sim2real_fingertip_readiness.md, step 4). |
| `sim2real_fingertip_cells.py` | CAD-hand control tasks for the printable 85A fingertips (docs/handoff/20261016-sim2real_fingertip_readiness.md, steps 3, 4 and 6). |
| `sim2real_fingertip_fe.py` | Finite elements of the 85A magnet blocks for the deployment-readiness study (docs/handoff/ 20261016-sim2real_fingertip_readiness.md, steps 3-4). |
| `sim2real_fingertip_manifest.py` | Inventory of the deployable SR2 hand for the fingertip deployment-readiness study (docs/handoff/ 20261016-sim2real_fingertip_readiness.md, step 1). |
| `sim2real_fingertip_policies.py` | Zero-shot transfer of the D6 servo-plant policies to the printable 85A fingertips (docs/handoff/ 20261016-sim2real_fingertip_readiness.md, step 7). |
| `sim2real_fingertip_prints.py` | Siraya Tech Flex TPU 85A print projects for the deployment-readiness study (docs/handoff/ 20261016-sim2real_fingertip_readiness.md, step 2), a companion of scripts/cad/metric_magnet_prints.py (MP). |
| `sim2real_fingertip_readiness_page.py` | Page: deployment readiness of the CAD hand with printed 85A magnet fingertips (study 20261016-sim2real_fingertip_readiness; hand-off docs/handoff/20261016-sim2real_fingertip_readiness.md). |
| `sim2real_fingertip_replays.py` | D1-D8 plan replays on the printable 85A fingertips (docs/handoff/20261016-sim2real_fingertip_readiness.md, step 5). |
| `sim2real_fingertip_scenes.py` | CAD-hand scenes of the printable 85A fingertips (docs/handoff/20261016-sim2real_fingertip_readiness.md, step 3). |
| `sim2real_fingertip_sensing.py` | Five-chip magnet readout of the 85A magnet blocks on the CAD finger (docs/handoff/20261016-sim2real_fingertip_readiness.md, step 8). |
| `spherical_magnet_check.py` | Spherical N52 magnets in place of the 3 x 2 mm cylinders of pad B00 (2026-10-21): the field at the ReSkin 5X chips, the error of the point-dipole model, the force resolution with spheres of 2-5 mm, and the torque between |
| `spherical_magnets_page.py` | Result page of the 2026-10-21 comparison of spherical and disc N52 magnets in pad B00 on the ReSkin 5X board. |
| `sr2_crosstalk_night.py` | Crosstalk between the fingertips on the SR2 hand, from the lab's sweep of 2026-11-04 (18:57 to 20:34; runbook and data in docs/experiments/20261104-sr2_hand_crosstalk, data/20261104-night). |
| `sr2_plan_ramp_replay.py` | SR2 hand: a crosstalk plan checked as lab/run_plan.py --ramp-deg 3 walks it (docs/experiments/20261104-sr2_hand_crosstalk). |
| `three_finger_soak_page.py` | Page on the three v1 fingertips left at rest overnight on the ESP32-C6 mux (hand-off 20261026-three_v1_fingers_overnight; owner, 2026-10-26 19:30: "ive plugged in all v1 finger ... |
| `tip_codesign_analysis.py` | Summaries of the fingertip co-design rows (docs/notes/20261014-tip_codesign_plan.md); each subcommand reads rows and writes one JSON summary next to them. |
| `tip_codesign_bed.py` | Contact-bed cases of the fingertip co-design study (docs/notes/20261014-tip_codesign_plan.md, steps 3 and 5). |
| `tip_codesign_fe.py` | Finite elements of the finalist fingertips (fingertip co-design study, plan step 10): the whole printed tip with its magnet (scripts/fingertip/tpu_tip_magnet.py, 358k DOF, CHOLMOD) with tpu_tip_design's geometry hooks (i |
| `tip_codesign_figures.py` | Figures of the fingertip co-design page (docs/experiments/20261014-tip_codesign/), drawn from the study's summaries. |
| `tip_codesign_film.py` | Films of the fingertip co-design study (docs/notes/20261014-tip_codesign_plan.md, step 12; ledger step 23), rendered on the CPU through Mesa's software EGL so they never wait for the GPU. |
| `tip_codesign_geometry.py` | Parametric convex fingertips of the co-design study (docs/notes/20261014-tip_codesign_plan.md, step 4): shape, hull meshes, surface samples for the contact laws, sphere pads, and the as-printed surface. |
| `tip_codesign_hullres.py` | Hull resolution of the Drake reference on the as-built design surface (fingertip co-design study, the open check of the hand-off note): Drake's 1 rad/s spin arm on tcd_rs3_re3 is 1.15 times the Drake-field law's, against |
| `tip_codesign_law.py` | Quasi-static contact laws of a convex fingertip against the bed's tool cylinder (R 12.5 mm), the references of the fingertip co-design study (docs/notes/20261014-tip_codesign_plan.md, step 3). |
| `tip_codesign_page.py` | Page: accuracy of the 1 mm sphere pads across fingertip geometry, and the co-design of the printed fingertip (geometry, print recipe, contact model) on the SR2 hand's tasks (study 20261014-tip_codesign; plan docs/notes/2 |
| `tip_codesign_print.py` | Printable files of the fingertip co-design finalists (docs/notes/20261014-tip_codesign_plan.md, step 11). |
| `tip_codesign_rig.py` | Hand-level runs of the fingertip co-design study (docs/notes/20261014-tip_codesign_plan.md): plan replays of the deployed hands D1-D8 and HOM closed-loop turns with a chosen pad modulus, friction and pad-model variant. |
| `tip_codesign_stage2_queue.py` | Write the stage-2 job queue (logs/20261014-tip_codesign/stage2/queue.txt) from the stage-1 ranking (docs/experiments/20261014-tip_codesign/screen/summary_stage1.json): the 8 highest J1, the as-built tip as printed (tcd_r |
| `tip_codesign_stage3_queue.py` | Write the stage-3 job queue (logs/20261014-tip_codesign/stage3/queue.txt) from the stage-2 finalists (docs/experiments/20261014-tip_codesign/screen/summary_stage2.json): fresh placements (scale cells seeds 3-5, HOM on D1 |
| `tip_codesign_sweep.py` | Deterioration sweep of the sphere pads over fingertip geometry (fingertip co-design study, plan steps 3 and 5). |
| `tip_codesign_tables.py` | Console tables of the fingertip co-design rows (bed variants; used while the study runs). |
| `tip_codesign_verify.py` | Verification of the fingertip co-design study (ledger step 24): recomputes the page's headline numbers from the rows with code independent of tip_codesign_analysis.py, and checks the cached scenes the rows ran on. |
| `tip_codesign_verify_numbers.py` | Headline numbers of the fingertip co-design page recomputed from the rows (ledger step 24), with code separate from tip_codesign_analysis.py, and compared with the summaries the page reads. |
| `tip_gait_data.py` | Collect the 2026-09-16 bench partial-reorientation study into one JSON for the page. |
| `tip_gait_page.py` | Build the bench partial-reorientation page from tip_gait_data.json. |
| `tpu_tip_calibrate.py` | Calibrate a printed SR2 fingertip back to the sphere pads (2026-10-12). |
| `tpu_tip_design.py` | The printed SR2 fingertip as a whole block with an embedded magnet (2026-10-12), for printing and for the finite-element model of scripts/fingertip/tpu_tip_magnet.py. |
| `tpu_tip_fem_analysis.py` | Tables for the TPU tip finite-element page (steps 4 and 11): the pressure-law modulus that represents each printed tip, how well one modulus does over loads and indenters, the spreads with print orientation and contact l |
| `tpu_tip_fem_bed.py` | Bed tasks T1 (pull) and T2 (twist) for the 1 mm sphere pads on the printed tip (tpu2.7) at the moduli fitted to the finite-element tip (step 4 of the TPU tip job). |
| `tpu_tip_fem_cells.py` | Periodic infill cells of printed TPU and their homogenized elasticity (step 2 of the TPU tip job). |
| `tpu_tip_fem_core.py` | Voxel finite elements for printed TPU: periodic homogenization of an infill cell and the stiffness of a voxelized fingertip, small strain, linear elastic, isotropic solid. |
| `tpu_tip_fem_film.py` | Film of the resolved printed tip under the screwdriver cylinder: the section through the contact (y = 0) with the voxels displaced (magnified) and coloured by displacement, beside the pressure on the palmar face, at appr |
| `tpu_tip_fem_gpu.py` | Matrix-free GPU solver for the voxelized printed fingertip (resolved gyroid), with frictionless contact of a rigid indenter under displacement control (active set). |
| `tpu_tip_fem_gpu_nl.py` | Large-deformation check of the voxel tip (step 18 of the TPU tip job): compressible Neo-Hookean, the energy of PolyFEM's NeoHookean, Psi = mu/2 (tr C - 3) - mu ln J + lambda/2 (ln J)^2, P = mu (F - F^-T) + lambda ln J F^ |
| `tpu_tip_fem_media.py` | Still images for the TPU tip page: the printed insert's wall, skin and core classes for both print orientations, and slices of the resolved 20 % gyroid tip (femtip env; reads the caches of tpu_tip_fem_tip.py). |
| `tpu_tip_fem_page.py` | Build docs/experiments/20261009-tpu_tip_fem/20261009-printed_tpu_fingertip_finite_elements.html. |
| `tpu_tip_fem_plate_check.py` | Plate stiffness of the solid printed insert with quadratic tetrahedra: an independent check on the two hexahedral tip models, the graded grid of tpu_tip_fem_tip.py and the 0.1 mm voxels of tpu_tip_fem_gpu.py, which give  |
| `tpu_tip_fem_polyfem.py` | PolyFEM (IPC contact) checks for the TPU tip job, step 18: the force PolyFEM exports against first-principles references, before PolyFEM is used for large deformation of the printed tip. |
| `tpu_tip_fem_smoke.py` | One-minute smoke tests of the finite-element tools in the femtip conda env (step 1 of the TPU tip job). |
| `tpu_tip_fem_tip.py` | The printed SR2 TPU fingertip insert as a finite-element model, and its contact with the bed's indenters (step 3 of the TPU tip job). |
| `tpu_tip_fem_voxel.py` | Low-memory voxelization of a closed triangle mesh by vertical scanlines (no ray-tracing library). |
| `tpu_tip_magnet.py` | Finite elements of the whole printed fingertip with an embedded magnet (2026-10-12): contact stiffness and the pressure-law modulus with and without the magnet, and the magnet's motion under normal and shear load at the  |
| `tpu_tip_magnet_layouts.py` | One magnet against four in the printed fingertip (2026-10-13): finite elements of the whole tip with several embedded magnets, under the screwdriver at the contact point and under small patch loads over the face. |
| `tpu_tip_magnet_sensing.py` | What one or four magnets let the five-chip board measure (2026-10-13): Cramer-Rao bounds on contact force and position from the finite-element rows of scripts/fingertip/tpu_tip_magnet_layouts.py. |
| `tpu_tip_print_plan.py` | Prints to send, their sphere-pad calibration, the expected gap between simulation and the printed tip, and the embedded magnet's signal (2026-10-12): data and HTML sections for scripts/fingertip/tpu_tip_fem_page.py. |
| `tpu_tip_rig.py` | The two-pad pinch rig of the contact bed with the SR2 TPU fingertip block in place of the 10.55 mm sphere. |
| `tpu_tip_widget.py` | Calculator for the printed fingertip in the page (2026-10-13): design of the magnet tip and bench calibration, live in the browser. |
| `v2_finger_bench.py` | First-bench package of the v2 finger (2026-10-20): scripts/fingertip/sim2real_fingertip_bench.py's `export` and `abort` on the 9.75 mm longer finger of scripts/fingertip/v2_finger_scenes.py, from its pinned-iCEM rows (sc |
| `v2_finger_cells.py` | The readiness study's control tasks (scripts/fingertip/sim2real_fingertip_cells.py: HOM on every candidate grasp of the four cells, pinned iCEM on each cell's best HOM grasp) on the v2 finger, 9.75 mm longer (scripts/fin |
| `v2_finger_scenes.py` | CAD-hand scenes with the DIP-link v2 finger (2026-10-19 night): the readiness study's variants (scripts/ sim2real_fingertip_scenes.py, assets/mjcf/experimental/20261016-sim2real_fingertip_readiness/<variant>/) with conce |
| `v2_tip_fe_checks.py` | Finite-element checks of the 11/22 half-ellipsoid fingertip that the DIP-link v2 carries (2026-10-19 night). |
| `v2_tip_fe_checks_page.py` | Result page of the 2026-10-19 night's checks of the v2 fingertips: finite elements of concept a's tip and concept b's pad against the calibrated 11/22 print, the grid check, under-extrusion, the G-code audit of the 85A p |
| `v2_tip_resolved_check.py` | Resolved-bead check of the homogenized 11/22 fingertip model (2026-10-21; owner, 2026-10-20 23:40: "yes please run these next jobs", the item "FE of the 11/22 pad with its beads resolved" of the printed-fingertips overvi |
| `v3_tip_contact_model.py` | One modulus for the contact models of the four-magnet v3 fingertips, fitted to the forces of the UR5 presses (owner, 2026-10-30: "check if our calibration results on the v3 four n52 magnet fingertips enable a more consis |
| `v3_tip_contact_model_figures.py` | Figures, film and numbers of the page on one contact-model modulus for the v3 fingertips (scripts/fingertip/v3_tip_contact_model.py writes the data; scripts/fingertip/v3_tip_contact_model_page.py builds the page). |
| `v3_tip_contact_model_page.py` | Page on the contact models of the four-magnet v3 fingertips fitted to the UR5 presses (owner, 2026-10-30: "check if our calibration results on the v3 four n52 magnet fingertips enable a more consistent FE / sphere-packed |
| `v3b_pad_b00_magnets_xl.py` | Concept b's 11/22 pad at groove fit B00 (0.00 mm per flank) with its 3 x 2 mm magnets and their pauses, 85A, on the Prusa XL's tool 2: the five magnets of the v2 tip (--magnets five) or the four of the 10-18 print (--mag |
| `v3b_ur5v2_pla_plate.py` | Bambu Lab P1S PLA plates of 2026-10-20: the UR5e calibration mount v2 with its two indenters and three covers of DIP link b v3 at three wall clearances (plate "calibration"), and DIP link b v3 (plate "link"). |

## hardware/ (39)

The bench hand: the servo plant and its fits, the tool tracker, the CB1 client, plan export and the trajectory clearance gate.

| script | what it does |
|---|---|
| `apply_measured_plant.py` | Rewrite a real_v1 scene so its actuators and masses are the ones the bench measured. |
| `bench_replay_calibration.py` | The finger servo plant fitted to the bench's own runs: every tracked bench run replayed in MuJoCo. |
| `build_cad_v1_finger.py` | Assemble the cad_v1 finger from its part STLs and write a MuJoCo model of it. |
| `build_cad_v2_finger.py` | The cad_v2 finger: the CAD hand of scripts/hardware/build_cad_v1_finger.py with the fingertip on the UR5 calibration rig since 2026-10-23 (assets/cad_v2): the DIP link b v4, the cassette 3T, the top cover TW15H3 trimmed  |
| `build_cad_v3_hand.py` | cad_v3: the SR2 hand as built, on the lab's UR5 (CB3), in MuJoCo. |
| `build_real_v1_scenes.py` | Build the `real_v1` base topology from the CAD-measured finger geometry. |
| `calibrate_plant_kp.py` | Find the actuator gain and torque ceiling that reproduce the bench's measured shortfall. |
| `fit_servo_compliance.py` | Fit the SCS0009's finite stiffness from the bench's own run logs. |
| `mh.py` | Tiny client for the CB1 control service (workstation side). |
| `plant_drop_gate.py` | Does the corrected plant make the simulator FAIL the way the bench fails? |
| `plant_ranking_test.py` | Does the corrected plant make the simulator RANK the hands the bench ranked? |
| `probe_real_v1_pivot.py` | Is the real_v1 grasp a rotational LOCK about the pinch axis? |
| `probe_real_v1_slip.py` | Why does the shaft creep out of the real_v1 grip, and which fix stops it? |
| `probe_real_v1_vertical_hold.py` | Can this hand hold the shaft VERTICAL, and can the policy command that pose? |
| `real_v1_bench_arrival.py` | How much of the commanded turn the hand actually performed, from the CB1's own run logs. |
| `real_v1_bench_grip.py` | Seat the grip: ramp open -> grip on the loaded plan, then report the pose and loads. |
| `real_v1_bench_page.py` | Build the doc page for the bench study: the flat table and the 100 mm wrist stack. |
| `real_v1_bench_plants.py` | The bench maneuver on every deployed hand, per plant, against the hardware's own turn. |
| `real_v1_bench_regrip.py` | Relieve a position-control over-clamp down to a target load band. |
| `real_v1_bench_report.py` | Read a pile of bench sessions and print the one table that matters. |
| `real_v1_bench_session.py` | Operator-driven bench session: one design, one sitting, one directory to hand back. |
| `real_v1_bench_stepped_run.py` | Instrumented, arrival-gated replay of a plan's turn segment. |
| `real_v1_bench_trials.py` | Per-trial table for the eight-hand open-loop transfer runs, from the CB1 log archive. |
| `real_v1_bench_video_array.py` | A 2x4 video array of the eight transfer-study hands holding a reorientation on the bench. |
| `real_v1_deploy_envelope.py` | Does the shippable schedule survive a hand that is not the simulated hand? |
| `real_v1_export_plan.py` | Turn a planned carry into something a bench can run: mount positions and a servo trajectory. |
| `real_v1_hand_commands.py` | Check the sim's hand design space against what the real gantries reach, and emit the firmware commands that put the machine into a given design. |
| `real_v1_render_deploy_plan.py` | Render the trajectory the hardware will actually replay. |
| `real_v1_tag_tracker.py` | The bench's object-pose sensor: two AprilTags, a RealSense, and the tape-measure geometry. |
| `real_v1_tracker_replay.py` | Watch a bench run back: the taped IR frames, captioned with the run's own trace. |
| `real_v1_tracker_service.py` | Persistent workstation companion for automatic web-UI AprilTag capture. |
| `real_v1_trajectory_clearance.py` | Finger-to-finger clearance along a real_v1 deploy trajectory. |
| `real_v1_transfer_figures.py` | First-pass figures for the real_v1 open-loop transfer study (8 morphologies x ~10 bench trials). |
| `real_v1_transfer_ranking.py` | Ranking analysis for the real_v1 open-loop transfer study. |
| `real_v1_transfer_study.py` | The pre-registered analysis of the morphology-ranking transfer study. |
| `real_v1_vane_angle.py` | Read the cylinder's turn angle out of a bench video, per frame. |
| `replay_real_v1_hardware_log.py` | Replay a real_v1 hardware command log in MuJoCo and emit an analysis-friendly NPZ. |
| `run_tracker_service.sh` | Workstation companion launcher: the counterpart to ~/run_control_station.sh on the CB1. |
| `servo_identification_protocol_page.py` | Page: identification protocol for the SR2 hand's serial-bus servos (draft, 2026-10-13). |

## hom/ (70)

Hand-object (HOM) control of the SR2 hand: the controller's rigs and tasks, the hand-object scale study, the layout landscapes and the FP toy.

| script | what it does |
|---|---|
| `bench_plan_page.py` | Build docs/experiments/20261011-bench_plan/20261011-bench_plan_finger_spacing.html and plan.json beside it. |
| `fp_hom_check.py` | Does a layout's kinematic foundational-pose score predict its HOM-controlled outcome? |
| `fp_hom_film.py` | Films for the foundational-pose review page: the HOM twist of a 40 mm sphere on two symmetric tripods of the 2-D case, side by side. |
| `fp_review_figures.py` | Figures of the 2026-10-29 foundational-pose review page (scripts/hom/fp_review_page.py), from the rows of scripts/hom/fp_sr2_toy.py in docs/experiments/20261029-foundational_pose_review/rows. |
| `fp_review_page.py` | Review page of the foundational-pose paper (arXiv 2409.14158), of how this project samples hand layouts, and of a plan to reproduce the paper's Pareto front and t-SNE analysis on the SR2 hand with HOM control, 2026-10-29 |
| `fp_roll_sensing_check.py` | The roll on Sobol layouts that track poorly with exact sensing, on the CPU runner (2026-10-29, the foundational-pose review page, Section "Sensing errors across the layouts"). |
| `fp_sr2_toy.py` | Foundational-pose sampling of SR2 gantry layouts: the method of the foundational-pose paper (Wang, Oh and Pollard, arXiv 2409.14158) run on the closed-form kinematics of the CAD-model SR2 hand, for the review page of 202 |
| `hand_object_scale_controllers.py` | Per-cell turn scores of the hand-object scale study under each controller and protocol, for the comparison page. |
| `hand_object_scale_films.py` | Films of the HOM turn on representative cells of the hand-object scale landscape: one object on three layouts, each at the grasp that turned it furthest (landscape.json), tiled three across. |
| `hand_object_scale_ilqr.py` | Gradient-based MPC for the scale-study turn: iterative LQR (Gauss-Newton) over 20 ms control intervals, with the dynamics and contact-sensor derivatives taken by finite differences of short MuJoCo rollouts (2026-10-13; o |
| `hand_object_scale_kin.py` | Finger-base spacing against object size on the real_v1 hand: the layout family, the reachable (layout, object) cells, and the fixed-contact precision-manipulation workspace. |
| `hand_object_scale_landscape.py` | The hand-object scale landscape: per-cell task scores J_T(s, d), the ridges s*_T(d), the scaling-law fits and the explanatory variables, from the study's rows. |
| `hand_object_scale_mpc.py` | Sampling model-predictive control of the in-hand turn on the cells of the hand-object scale study, a second controller beside the HOM turn of hand_object_scale_sim.py. |
| `hand_object_scale_mpc_gpu.py` | MuJoCo-Warp rollouts for the scale-study MPC: the planner of hand_object_scale_mpc.py with its candidate plans rolled out as parallel GPU worlds instead of CPU threads, so that a replanning can afford hundreds of samples |
| `hand_object_scale_mpc_timing.py` | Cost of the sphere-pad fingertip in the MPC's rollouts, against point contact on the TPU block, on one scale-study cell with all three fingers on the cylinder. |
| `hand_object_scale_page.py` | Build docs/experiments/20261008-hand_object_scale/20261008-finger_spacing_object_size.html. |
| `hand_object_scale_planners.py` | Planner comparison on the in-hand turn of the hand-object scale study: HOM against sampling MPC variants that keep three pads on the cylinder, on the same cells, grasps, placements and plant (2026-10-13; owner: "HOM look |
| `hand_object_scale_post_probe.py` | Traces of the HOM turn on chosen cells of the hand-object scale study, with and without the support post: per 10 ms tick the turn angle, the post's normal force on the object, the object's rise from its post-grip height  |
| `hand_object_scale_protocol.py` | The bench protocol of the hand-object scale study, generated from the simulated landscape: which cells, at which grasp, how many trials, in what order, and how many bench hours. |
| `hand_object_scale_sim.py` | Grasp robustness and in-hand reorientation over the (layout, object) cells of the hand-object scale study, in CPU MuJoCo on the bench-calibrated servo plant with the printed TPU fingertip as 1 mm sphere pads. |
| `hom_chain.py` | The screwdriver load-and-insert chain under relative contact-velocity control on the real_v1 (SR2) hand, in MuJoCo and Drake. |
| `hom_chain_figures.py` | Theory figures for docs/experiments/20261002-hom_chain: Drake's hydroelastic pressure field on the real_v1 fingertip against MuJoCo's sphere-packed pad, both pressed onto the screwdriver at the same pinch force, from the |
| `hom_chain_page.py` | Build docs/experiments/20261002-hom_chain/20261002-hom_screwdriver_chain.html from the study's jsonl files. |
| `hom_chain_study.py` | Driver for docs/experiments/20261002-hom_chain: Exp 1 of arXiv 2609.25619 on D8, the nominal chain filmed in six contact models, a perturbed MuJoCo batch (seeds 1-10), and a close-up of the sphere-packed pad. |
| `hom_code_page.py` | Walkthrough page of the HOM controller's code (src/sr2hand/control/hom.py, surfaces.py; scripts/hom/hom_tasks.py, scripts/hom/hom_gpu.py), a subpage of the HOM review page, 2026-10-30. |
| `hom_code_trace.py` | Instrumented rollouts of the HOM controller for the code-walkthrough page (scripts/hom/hom_code_page.py), 2026-10-30. |
| `hom_compute.py` | Compute ledger and CPU-against-GPU throughput of the HOM studies (owner, 2026-11-03: "let's make sure we're timing this all for compute-tracking purposes and throughput measuring, including cpu vs gpu speedups"). |
| `hom_contact_patch_page.py` | Build docs/experiments/20261001-hom_contact_patch/20261001-hom_pinch_contact_models.html from the study's JSON. |
| `hom_contact_rig.py` | Two-pad pinch rig in Drake and MuJoCo: which MuJoCo contact reproduces the hydroelastic pinch. |
| `hom_contact_study.py` | Run the 2026-10-01 contact-patch study on the two-pad pinch rig (scripts/hom/hom_contact_rig.py). |
| `hom_control.py` | Relative contact-velocity control of arXiv 2609.25619 (Wang, Oh, Pollard) on the real_v1 hand. |
| `hom_ea.py` | The evolutionary search of Yang et al. |
| `hom_ea_grasp.py` | The evolutionary search of scripts/hom/hom_ea.py with the grasp in the genome (owner, 2026-11-03: "can't we score and refine a grasp based off how well the grasp (+morphology) executes the HOM control? |
| `hom_films.py` | Films of the HOM tasks on the SR2 hand with the v4b fingertip, for the two 10-29 pages and the code walkthrough (2026-10-30; owner: "some of the videos on these pages use the incorrect older fingertips ... |
| `hom_framework_page.py` | Pages of the 2026-10-21 HOM controller study: the method page (the HOM paper's controller and its MuJoCo implementation in src/sr2hand/control/) and the results page (four tasks across object sizes, scripts/hom/hom_tasks |
| `hom_fresh_seeds.py` | Fresh-seed check of the one-design picks of hom_plant_transfer.py (2026-11-03; owner: "let's do those three remaining tasks", the first being this check). |
| `hom_gpu.py` | HOM control of the four tasks of scripts/hom/hom_tasks.py (pin-push, press, roll, twist) on MuJoCo-Warp: many hand layouts, seeds and sensing variants as parallel GPU worlds, with the controller of src/sr2hand/control/ho |
| `hom_grasp_overlap.py` | Finger-to-finger clearance of every grasp of the grid's grasp sets at its touch pose, with the v4b fingertip's pads and rigid sphere packs (2026-10-30). |
| `hom_grasp_refine.py` | Refinement of each layout's grasp, and of HOM's commanded forces and contact modes, on the HOM task score (owner, 2026-11-03: "can't we score and refine a grasp based off how well the grasp (+morphology) executes the HOM |
| `hom_grasp_search.py` | Grasp search by turn range for the fixed-contact three-finger turn (prototype, CPU; 2026-10-06). |
| `hom_ground.py` | Ground clearance of the grids' grasps: how far the plant's finger spheres reach below the plane through the object's lowest point, the table an object would rest on (owner, 2026-10-30: "only allow grasps that would not p |
| `hom_hand_brake.py` | The pinch brake of arXiv 2609.25619 on the real_v1 (SR2) hand, in MuJoCo and Drake. |
| `hom_hand_brake_page.py` | Build docs/experiments/20261001-hom_hand_brake/20261001-hom_hand_brake.html from hand_brake.jsonl. |
| `hom_hand_drake.py` | Drake side of scripts/hom/hom_hand_brake.py: the same D8 hand and tool, Drake 1.57 SAP. |
| `hom_landscape.py` | Landscapes of HOM task scores over the SR2 hand's layouts and object sizes: the size study of the design page (docs/experiments/20261029-foundational_pose_review), from the rows of scripts/hom/hom_gpu.py run on --layouts |
| `hom_modulus_scenes.py` | HOM reference scenes at another pad modulus (2026-11-02; owner: "seems like we need to queue up modulus studies across a range as well. |
| `hom_mpc_gpu.py` | Sampling MPC over the HOM controller's commands on MuJoCo-Warp, 2026-10-30. |
| `hom_paper_tasks.py` | The simulated tasks of arXiv 2609.25619 (Wang, Oh, Pollard) beyond Exp 1, on the real_v1 (SR2) hand D8 with sphere-packed MuJoCo fingertips, under the same controller as the screwdriver chain (scripts/hom/hom_chain.py):  |
| `hom_plant_compare.py` | The 10-30 searches' best layouts on three plants, from the CPU checks of hom_top_cpu_check.py (2026-11-02). |
| `hom_plant_page.py` | Builds docs/experiments/20261029-foundational_pose_review/20261102-hom_plant_modulus.html: how the HOM design landscapes of the SR2 hand change with the finger servo, the pad modulus and the scene friction (2026-11-02 an |
| `hom_plant_transfer.py` | How a HOM design landscape changes with the finger servo and the pad modulus (2026-11-02; owner: "i wonder if we could also characterize the change in landscape between the idealized 4nmrad plant, the current scs0009 cal |
| `hom_proxy_check.py` | Finger-to-finger contact on the GPU through a few proxy spheres per fingertip (2026-10-30). |
| `hom_reanchor_check.py` | The pin-push's middle finger with and without the re-anchor of hom.HomParams.reanchor, on the CPU (2026-10-30). |
| `hom_review_figures.py` | Figures of the 2026-10-29 HOM review page (scripts/hom/hom_review_page.py): small worked examples that explain the HOM paper's equations (arXiv 2609.25619v2), and the chart of the fingertip-sensing variants (scripts/hom/ |
| `hom_review_page.py` | Review page of the HOM paper (arXiv 2609.25619v2), of its MuJoCo implementation on the SR2 hand (src/sr2hand/control/), and of the magnetic fingertips as the controller's contact sensor, 2026-10-29. |
| `hom_rows.py` | Rows of the HOM task runs (scripts/hom/hom_tasks.py, scripts/hom/hom_gpu.py) as the page builders read them, without the simulation stack. |
| `hom_score_refine_page.py` | Builds docs/experiments/20261103-hom_score_refine/20261103-hom_score_refine.html: each SR2 layout scored three ways under HOM control (S1 the best of its sampled grasps at HOM's defaults; S2 the grasp refined on the HOM  |
| `hom_table_check.py` | The CPU runner with a table under the object until the lift, the full version of the ground check (owner, 2026-10-30: "doing these tasks with an initial grasp, raise, and then manipulate"; scripts/hom/hom_ground.py is th |
| `hom_task_conflict.py` | Tasks whose best finger layouts differ: one layout for all of them (a generalist), or a change of layout between tasks (owner, 2026-11-03 20:3x: "how can we accommodate tasks with conflicting morphology objectives? |
| `hom_task_conflict_page.py` | Builds docs/experiments/20261103-hom_task_conflict/20261104-hom_task_conflict.html: one SR2 finger layout for the four HOM tasks against one layout per task, on performance alone (owner, 2026-11-03 20:3x: "how can we acc |
| `hom_tasks.py` | Four manipulation tasks under the HOM controller (src/sr2hand/control/hom.py) on the CAD-model SR2 hand, across object sizes, in CPU MuJoCo on the bench servo plant (kp 4 N m/rad, 1 N m, damping 0.08) with the printed TP |
| `hom_tasks_mpc.py` | Sampling MPC on the four HOM tasks of hom_tasks.py: the baseline beside the HOM controller on the same CAD-model hand, servo plant, pads, objects, grasps and object placements. |
| `hom_top_cpu_check.py` | The CPU runner on the grid's top layouts of a task (by the GPU's best-of-16 score, seeds 0-2), each from its best grasp, seed 0, with all contacts and with the GPU evaluator's collision filter (pads and rigid spheres und |
| `hom_toy_reach_check.py` | Why the pose toy reaches more layouts since the finger clearance moved into its program (2026-10-30, 21d3d2877). |
| `hom_trends_page.py` | Builds docs/experiments/20261029-foundational_pose_review/20261030-hom_layout_trends.html: the best layouts of the SR2 hand under HOM control across object sizes, with the fingers kept 2 mm apart and with a table under t |
| `hom_turn3.py` | The three-finger reorientation of the bench, driven by the relative contact-velocity controller of arXiv 2609.25619 (Wang, Oh, Pollard) instead of the deployed open-loop joint trajectories. |
| `hom_turn3_films.py` | Films of the three-finger turn on the fitted plant: the deployed open-loop plan, the relative contact-velocity (HOM) controller with its governor, and without it. |
| `hom_turn_rl_contact_page.py` | Build docs/experiments/20261006-hom_turn3/20261006-servo_refit_hom_turn_pad_cost.html. |
| `scale_turn_cad_v1_page.py` | Page: the scale-turn study of 13-14 October repeated on the CAD-model hand (2026-10-14). |
| `scale_turn_controllers_page.py` | Page: in-hand turn of cylinders on the scale-study layouts under HOM control and sampling MPC (2026-10-13). |

## mechanisms/ (37)

Reorientation mechanisms on the real_v1 hands: the open-loop carry, pinch and swing, gaits and the grasp-to-gait chain on the arm.

| script | what it does |
|---|---|
| `build_real_v1_arm_scene.py` | Put the real_v1 hand on a UR5e, so the palm's 6 DOF have to be paid for by a real arm. |
| `build_screw_scene.py` | Turn the shaft into a screw and the floor into its seat. |
| `calibrated_plant_chain_data.py` | Collect the 2026-09-16 calibrated-plant chain sweeps into one JSON for the page. |
| `calibrated_plant_chain_page.py` | Build the calibrated-plant chain page from calibrated_plant.json. |
| `chain_handover_gait_page.py` | Build docs/experiments/20260923-chain_handover_gait/20260923-chain_handover_gait.html from the study's JSON. |
| `palm_driver.py` | One controller, two wrists: the floating gantry and a UR5e, behind the same interface. |
| `probe_axial_load.py` | How much AXIAL (world-Z) force can the hand's grip on the screwdriver resist? |
| `probe_real_v1_carry.py` | Can this hand CARRY the shaft to vertical without slipping in the grip? |
| `probe_real_v1_chain.py` | The whole task in one rollout: grasp -> lift -> reorient -> stand it down -> gait. |
| `probe_real_v1_gait.py` | Can three fingers SPIN the shaft about its own axis once it is standing on the ground? |
| `real_v1_arm_study.py` | Gantry versus UR5e, and how stiff the wrist has to be. |
| `real_v1_basketball_grasp.py` | Fit and probe a basketball grasp on a `real_v1` mount layout. |
| `real_v1_basketball_media.py` | Films, pose snapshots and the contact-model sensitivity for the basketball page. |
| `real_v1_basketball_page.py` | Build the basketball-grasp page from the sweep's own JSON. |
| `real_v1_chain_films.py` | Render the chain for every hand, one video per hand plus a labelled grid. |
| `real_v1_chain_hands.py` | The chain on both deployed hand sets, with the grip loop closed through the whole run. |
| `real_v1_chain_page.py` | Build the doc page for the chained task: grasp -> gait, on an arm, into a screw. |
| `real_v1_chain_policy.py` | Run a reorient checkpoint as the finger controller inside the UR5e chain (CPU MuJoCo). |
| `real_v1_chain_study.py` | grasp -> lift -> reorient -> stand -> gait, swept: what does the seam actually cost? |
| `real_v1_d1_continuation.py` | The real_v1 D1 continuation after the supported grasp: scene, held state, plan, score, metrics and MJX glue. |
| `real_v1_gait_page.py` | Build the ground-supported gaiting doc page from gait_study.json plus the rendered media. |
| `real_v1_gait_ring_screen.py` | Where each hand can hold a standing tool as a three-pad gait ring. |
| `real_v1_gait_study.py` | The ground-supported gaiting study: six sweeps over `probe_real_v1_gait.gait`, one JSON. |
| `real_v1_handover_page.py` | Build the doc page for the handover study: changing grasp without letting go. |
| `real_v1_held_page.py` | Build the table-stand result page from the study's own JSON. |
| `real_v1_reorient_landscape.py` | A seconds-per-design REORIENTABILITY metric for `real_v1`, and the landscape it draws. |
| `real_v1_screw_study.py` | Make the tool a screw and the floor its seat, then ask what changes. |
| `real_v1_turn_mechanism.py` | What turns the tool in the bench maneuver: the fingers, the post, or gravity? |
| `real_v1_turn_probe.py` | Per-step mechanics of the chain's finger turn: what stops the tool on a compliant plant. |
| `reorient_primitive.py` | The reorientation primitive: what a reorient policy does in the OBJECT's frame. |
| `reorientation_journey_data.py` | Collect every chain sweep run on the real_v1 deployed hands into one JSON. |
| `reorientation_journey_page.py` | Build the reorientation journey page from journey.json. |
| `rv05_carry_band.sh` | The held floor-free turn, on the hand that PHYSICALLY EXISTS (rv05_manual_stored). |
| `swing_reorient_data.py` | Collect the 2026-09-16 pinch-and-swing study into one JSON for the page builder. |
| `swing_reorient_page.py` | Build the pinch-and-swing page from swing_reorient.json. |
| `turn_mechanism_data.py` | Collect the 2026-09-16 turn-mechanism study into one JSON for the page builder. |
| `turn_mechanism_page.py` | Build the turn-mechanism page from turn_mechanism.json. |

## rl/ (62)

Reinforcement learning: the trainer and its launchers, the morphology runs, policy evaluation and the policy pages.

| script | what it does |
|---|---|
| `assert_config_parity.py` | Fail a run EARLY if its config diverges from a reference run in a way nobody intended. |
| `bench_env_throughput.py` | Where does the training wall clock actually go, and what moves it? |
| `compliance_dr_pipeline.py` | Compliance-DR pipeline — train a stiffness-ROBUST m05 policy and prove it. |
| `compliance_rate_sweep.py` | Compliance-robustness sweep, RATE edition — N batched rollouts per point. |
| `compliance_robustness_sweep.py` | Compliance-robustness sweep — how sensitive are TRAINED policies to contact stiffness? |
| `compute_budget_data.py` | Collect training-throughput data for the compute-budget page. |
| `compute_budget_page.py` | Build the compute-budget page from compute_budget.json. |
| `d6_cal_60m_page.py` | Build the D6-on-cal 60 M-step policy page from the eval JSONs and training curves. |
| `eval_blind_actor_2x2.sh` | Evaluate the blind-actor 2x2 (scripts/rl/train_blind_actor_2x2.sh) on ONE protocol. |
| `hands_tranche_page.py` | Build the cross-hand reorientation-policy page from the tranche queue's own outputs. |
| `hands_tranche_queue.py` | Serial, gate-aware, resumable queue of reorient finetunes across the deployed hands. |
| `karma_analysis.py` | Score KaRMA against this program's own labels on the real_v1 design family. |
| `karma_collect.py` | Collect a KaRMA sweep's per-run outputs into one analysis table. |
| `karma_page.py` | Build the KaRMA evaluation page from the sweep's own JSON. |
| `make_cal_morphology_run.py` | A morphology-run directory for the RL trainer from a chain flat scene on the `cal` plant. |
| `make_sweep_video_grids.py` | Build comparison-grid videos from a morph_pipeline_sweep tag (2026-07-13). |
| `make_tipmesh_morphology_run.py` | A morphology run whose frozen scene carries the screw-tip mesh on the tool's -z end. |
| `make_work_plant_runs.py` | Morphology runs for the RL contact comparison on the working plant (2026-10-06). |
| `policy_cpu_rollout.py` | Roll a reorient checkpoint out in plain CPU MuJoCo, rebuilding its observation from raw state. |
| `policy_eval_suite.py` | N-rollout evaluation for a reorient policy: success + stability metrics as a DISTRIBUTION. |
| `policy_filmstrip.py` | Turn a trained policy's rollout video into ONE labelled contact-sheet PNG. |
| `policy_healthcheck.py` | Single-policy trajectory-health check — flag a degenerate grasp/reorient policy. |
| `policy_reeval_traces.py` | Re-evaluate the policies behind the evaluation figures of four result pages and keep their per-step traces. |
| `policy_transfer_probe.py` | Zero-shot evaluation of a reorient checkpoint on perturbed copies of its own scene. |
| `probe_action_budget.py` | Can the policy COMMAND the behaviour you are about to pay it for? |
| `probe_grip_balance.py` | Probe: PER-FINGER contact force + contact-found fraction in a policy's own standalone reorient env. |
| `probe_grip_force.py` | Probe: in a policy's OWN standalone reorient env (clean held reset), measure the fingertip AND palm contact force + held-cos. |
| `probe_obs_ablation.py` | Closed-loop observation ablation of a trained A->B pair. |
| `probe_queue.sh` | probe_queue.sh — POLICY-BOTTLENECK probe queue (2026-07-10): P1 rescue -> P2 avar. |
| `rename_results_bids.sh` | Prepend A/B-registry IDs to the medium-flat reorient pipeline run dirs in results/rl, keeping the original descriptive name so paths stay self-documenting. |
| `reorient_variance_study.py` | Reorient-quality VARIANCE study — can we cut the seed noise enough to rank morphologies? |
| `rl_contact_eval.py` | Evaluation of the RL contact-model comparisons. |
| `rl_contact_post_queue.sh` | After scripts/rl/rl_contact_train_queue.sh: evaluate every checkpoint (scripts/rl/rl_contact_eval.py, one GPU job) and make the phase-aligned filmstrip of each run's newest eval video (scripts/rl/policy_filmstrip.py). |
| `rl_contact_sensor_check.py` | Regression check of the RL env's contact sensors after the switch to reduce="netforce" (2026-10-06). |
| `rl_contact_throughput.py` | Throughput of the RL training env with each fingertip contact model (2026-10-06). |
| `rl_contact_train_queue.sh` | RL contact comparison (owner-approved 2026-10-06 18:40): D6 reorientation from scratch on the working plant, the TPU block as one convex mesh (point contact) against 1 mm sphere pads, two seeds each, 20 M steps at 2,048  |
| `rl_contact_train_queue_20261008.sh` | RL contact-model comparison, 2026-10-08 (hand-off docs/handoff/20261007-contact_model_policy_training.md): D6 reorientation from scratch on the working plant, five fingertip contact models, three seeds each, one GPU job  |
| `rl_contact_train_queue_20261011.sh` | RL contact-model comparison on the servo-limited plant, 2026-10-11 (hand-off docs/handoff/20261011-servo_plant_policies.md): D6 reorientation from scratch with the TPU block as one convex mesh (point contact) and as 1 mm |
| `rl_demo_handoff_continuous.py` | State-CONTINUOUS Policy A -> Policy B handoff in a single environment. |
| `rl_diagnose_policy.py` | Diagnose whether a trained policy actually adapts its actions to cube pose. |
| `rl_eval_object.py` | Deterministic eval of any object policy at full DR + pose-grid videos. |
| `rl_eval_reorient_metrics.py` | Deterministic behavioral metrics for in-hand reorient policies — the HONEST evaluation (use this, NOT training reward sums, which conflate how-long x how-aligned and silently hid the v2 verticality regression). |
| `rl_play_policy.py` | Play a trained policy through N back-to-back episodes with random cube spawns, recording one continuous video. |
| `rl_plot_reorient.py` | Render side-by-side comparison plots for the in-hand reorientation runs. |
| `rl_plot_training.py` | Read PPO tensorboard logs for one or more runs and render diagnostic plots. |
| `rl_policy_replay.py` | Open-loop replay of a trained reorientation policy's finger targets in other simulators (2026-10-06). |
| `rl_record_handoff_states.py` | Record Policy A's ACTUAL terminal states (post-lift) into a 'handoff state bank' for train-the-handoff. |
| `rl_record_initiation_bank.py` | Record Policy B10's INITIATION SET — the object-state distribution B10 sits in at reorient-onset, where it reorients to held-cos ~0.977. |
| `rl_record_reorient_schedule.py` | Record a reorienter's held-cos-vs-time SCHEDULE from its clean spawn. |
| `rl_render_reorient.py` | Render a verification video for a skip-lift in-hand reorient policy. |
| `rl_render_rollout.py` | Render one deterministic rollout of a reorient checkpoint at a chosen resolution. |
| `rl_state_export.py` | Export a held state of the RL training env for timing other engines on it (2026-10-06). |
| `rl_train_cube.py` | PPO training entry point for the cube grasp MVP. |
| `robust_tranche_page.py` | Build the page for the 2026-09-20 continuation ("robust") tranche from the queue's own outputs. |
| `servo_plant_policies_page.py` | Build docs/experiments/20261011-servo_plant_policies/20261011-servo_plant_contact_model_policies.html. |
| `sim2real_robustness_sweep.py` | How far can the shipped inline hand's a10 -> b33 handoff be pushed before it stops working? |
| `train_A_on_morph.sh` | Retrain Policy A (lift + deliver) on a NEW morphology — for the full A->B handoff. |
| `train_blind_actor_2x2.sh` | BLIND-ACTOR 2x2 — what does the reorienter lose when the actor cannot see the object? |
| `train_corefine_BtoA.sh` | B->A CO-REFINEMENT — nudge Policy A's lift so it delivers a grip Policy B reorients better. |
| `train_handoff_liveA_reset.sh` | LIVE-A RESET — train Policy B from Policy A's ORGANIC delivery (zero teleport). |
| `train_real_v1_blind_pair.sh` | Real-hardware-family observation transfer: sighted vs hidden-object actor. |
| `train_reorient_on_morph.sh` | Skip-lift REORIENTER on a NEW morphology — does the balanced grasp give a balanced grip? |

## ur5/ (107)

The UR5 calibration cell: session plans, the cell model, registration and analysis of the presses, the ATI sensor and the inverted rig.

| script | what it does |
|---|---|
| `ati_calibration_analysis.py` | The ATI Mini45's calibration from a session of scripts/ur5/ati_calibration_session.py (or its --sim): each rest's mean gage values, gravity in the ATI's frame from the robot's joints, the drift from the reference station |
| `ati_calibration_film.py` | Film of an ATI calibration session (scripts/ur5/ati_calibration_session.py, real or --sim): the session's joint record (robot.bin) replayed in MuJoCo on the UR5's link meshes, with the tool (HEX-E, mount v3's adapter, th |
| `ati_calibration_fit.py` | Calibration of an ATI transducer's raw gage values against reference wrenches (the handoff of 2026-10-23, docs/experiments/20261023-ati_ft_calibration/): w = C (g - g0), C 6 x 6 from gage counts to N and N m at the ATI's |
| `ati_calibration_plan.py` | Robot plan of the ATI Mini45's (or, with --parts, the Mini40's) calibration with a weighed mass on the lab's UR5 (CB3): the stations, their order and their checks, for scripts/ur5/ati_calibration_session.py (owner, 2026- |
| `ati_calibration_session.py` | Calibration session of the lab's ATI Mini45 on the UR5 (CB3) with a weighed mass, for the lab computer: the robot visits the stations of a plan written by scripts/ur5/ati_calibration_plan.py while the Net F/T box streams |
| `ati_calibration_sim.py` | Can the UR5 calibration plan calibrate an ATI Mini45 stacked on the HEX-E? |
| `ati_calibration_sim_session.py` | Simulated robot and Net F/T box for scripts/ur5/ati_calibration_session.py --sim: the session runs unchanged in virtual time and writes the same files as on the lab's UR5, plus sim_truth.json, so that scripts/ur5/ati_cal |
| `ati_ft_calibration_page.py` | Page of 2026-10-23: the calibration of the lab's ATI Mini45 on the UR5 with a weighed mass: the coaxial stack on the HEX-E from ATI's drawing, the hook cross, the robot plan, the session program and its simulated session |
| `ati_gravity_pose_check.py` | Which gravity stations of a UR5 calibration plan are safe with the ATI stack and the hook cross on the tool (the ATI calibration plan of 2026-10-23, docs/experiments/20261023-ati_ft_calibration/): the tool's points (moun |
| `ati_netft_discover.py` | Finds ATI Net F/T boxes on the local network by ATI's RTA discovery, whatever their IP address and subnet: one UDP broadcast to port 51000, and the boxes answer on the multicast group 224.0.5.128, port 28250. |
| `ati_netft_logger.py` | Logger for an ATI Net F/T box (NETBA, serial FT35796 in the lab) during a UR5 calibration session: streams the box's six channels over RDT (UDP 49152) and writes them with host time stamps beside the session's own stream |
| `ati_reference_compare.py` | Which reference calibrates the ATI: the weighed mass, or the HEX-E with the ATI coaxial or turned 90 deg (owner, 2026-10-23: "the ati will be stuck onto the F/T sensor with the Z-axis aligned. |
| `ati_sim_sessions.py` | Simulated ATI calibration sessions for the plan page: the session program on the plan (scripts/ur5/ati_calibration_session.py --sim) for several seeds of the simulated sensor, each analysed (scripts/ur5/ati_calibration_a |
| `ati_stack_mount.py` | Printed parts that put the lab's ATI Mini45 between the UR5's HEX-E and calibration mount v3, and a hook cross for its calibration with a weighed mass (owner, 2026-10-23: "we can probably calibrate the ATI with just a 3d |
| `calibration_motions_page.py` | Page on the motions of the UR5 calibration for the four-magnet fingertips (owner, 2026-10-29: the indenter plate v2d waits "until we have 100% finalized all the motions we need for effective calibration"). |
| `corvus_pull.sh` | Copy a UR5 session folder from the lab computer corvus, read-only, for analysis before the lab pushes it. |
| `corvus_watch.py` | Wait until the UR5 lab computer corvus changes, then print what changed and exit (owner, 2026-10-27: renders and plans that follow the night's registrations without anyone at the workstation). |
| `indenter_array.py` | Indenter array for the UR5 cell (owner, 2026-10-27: "put all four of these indenters on one print, and that one piece to have multiple indexers so that we can do registration on multiple indexers on the same part"; the b |
| `indenter_plate_layout.py` | Layout of the indenter plate for the UR5 cell (owner, 2026-10-28: "the big ass indenter plate i printed wont actually work though, since the big sphere is in the way of the side-to-side tests ... |
| `indenter_plate_layout_page.py` | Page on the layout of the indenter plate for the UR5 calibration cell (owner, 2026-10-28: "the big ass indenter plate i printed wont actually work though, since the big sphere is in the way of the side-to-side tests ... |
| `inverted_rig_page.py` | Page on the inverted calibration rig for the UR5 cell: a probe on the HEX-E presses fingertips held on indenter plate v2e (owner, 2026-10-29 13:0x: "designing the adapter, modifying the v2d plate one more time, running t |
| `inverted_rig_print.py` | Bambu Lab P1S projects for the inverted calibration rig (owner, 2026-10-29: "yea, we can build the inverted rod ... |
| `tall_indenter.py` | Tall indenter for the UR5 cell (owner, 2026-10-26; specification docs/experiments/20261026-tall_indenter/ 20261026-tall_indenter_spec.md): a 5 mm PLA sphere on a neck of radius 3 mm that widens at 15 deg into a column of |
| `ur5_calibration_mount.py` | UR5e calibration mount for the DIP link with the ReSkin 5X board (hand-off docs/handoff/20261018-ur5_calibration_mount.md). |
| `ur5_calibration_mount_page.py` | Result page of the UR5e calibration mount (hand-off docs/handoff/20261018-ur5_calibration_mount.md, step 4). |
| `ur5_cb3_cell.py` | Geometry of the lab's UR5 (CB3) cell for the fingertip's calibration plan and film: the robot's links (the UR5 URDF of ur_description in rh56_controller/magpie_force_control, its collision meshes), the tool stack on the  |
| `ur5_cell_render.py` | Rendered stills of the lab's UR5 (CB3) calibration cell as it stood on 2026-10-27 (owner, 2026-10-27: "render the scene with the four indenters mounted relative to the UR5, preferably in MuJoCo, with some colour realism" |
| `ur5_checkerboard_tip_page.py` | Page of the first checkerboard fingertip pressed on the UR5 (session cb1t, 2026-10-23 evening, QUEUE.md R22): the pole pattern and the magnets' strength from the field at rest, the registration with the offset touch-offs |
| `ur5_fixture_from_seating.py` | A fixture's frame I in the UR5's base (T_BI) from a hand seating on its key: T_BI = T_BE_seat T_EM T_IM_home^-1, with T_IM_home from a plan of that fixture and T_EM from a registration of the tool. |
| `ur5_hexe_link_losses.py` | The HEX-E's link losses on the UR5 against the arm's joints (2026-10-30). |
| `ur5_inverted_rig.py` | Inverted calibration rig for the UR5 cell: a probe on the HEX-E presses fingertips held still on the indenter plate (owner, 2026-10-29 13:0x: "should we consider adding a rigid indenter mount for the ur5 and adding a thi |
| `ur5_mount_v3_page.py` | Result page of 2026-10-21: UR5e calibration mount v3 for the owner's M6 x 20 screws, tightened from above (owner, 17:54). |
| `ur5_mount_v4_page.py` | Result page of 2026-10-23: UR5 calibration mount v4, mount v3 with a stop plate behind the DIP link's tongue (the owner, through the lab session: the DIP link moves a little in v3's fork when the pad is pressed hard; QUE |
| `ur5_mount_v5_page.py` | Result page of 2026-10-23: UR5 calibration mount v5, mount v4's fork as printed with a PLA wedge pushed into the gap between its stop plate and the DIP link's tongue, in place of v4's three brass set screws (owner, after |
| `ur5_probe_calibration_analysis.py` | Ground truth of a probe-mode calibration session (the inverted rig, 2026-11-02 on: the tip held on its seat, the sphere or the half rod on the HEX-E), in the holds format of scripts/ur5/ur5_tip_calibration_analysis.py, s |
| `ur5_probe_joint_registration.py` | One sphere centre for several probe sessions with the same tool: the joint fit of the probe's c_M and each session's seat pose T_AP over their pass 1 touches (the lab, 15:12 on 11-03, after s2sph2's c_M stood about 0.5 m |
| `ur5_probe_plan.py` | Program plans for the probe mode of ur5_tip_calibration.py (the inverted rig: a probe on the HEX-E presses a fingertip held still on its seat) from the workstation's probe plans (scripts/ur5/ur5_inverted_rig.py replay, - |
| `ur5_rod_fitted_margins.py` | The rod's sweeps against their tip's body with the rod as a registration fitted it (lab, 17:47 on 11-03: seat 2's fit turned the rod -5.40 deg about z_M and 0.77 deg across, its c_M 1.17 mm off the plan's). |
| `ur5_rod_joint_registration.py` | One rod for several probe sessions: the joint fit of the rod's axis (its point c_M across the axis, its turn about z_M and across) and each session's pad pose T_AP over the rod sessions' pass 1 touches (workstation, 23:4 |
| `ur5_shear_figures.py` | Figures and the film of the page on shear, friction and impratio of the printed fingertip (docs/experiments/20261026-shear_friction_impratio/), from the rows of scripts/ur5/ur5_shear_impratio.py (impratio under the train |
| `ur5_shear_impratio.py` | Impratio of the 1 mm sphere pads under the policy trainer's solver: whether impratio 100 or 300 keeps the hand's replays and the scripted grasp stable at 10 solver iterations and a 2 ms step, how far it lowers the creep, |
| `ur5_shear_page.py` | Page on the shear, the friction coefficient and the friction torque of the printed fingertip measured on the UR5, and on impratio under the policy trainer's solver (hand-off 20261026-shear_friction_impratio, owner 2026-1 |
| `ur5_shear_skin.py` | The sprung skin against the UR5 slides of the printed fingertip: the 812 pads of the UR5 rig (ur5_tip_calibration_pads) on a skin body sized from the slides themselves, replaying the measured slides, against plain pads. |
| `ur5_shear_slides.py` | Shear against surface displacement in the UR5 slides of the printed fingertip: the give before slip, the shear at which the pad slips and the friction coefficient, and the shear left when the turn returns to 0 deg. |
| `ur5_shear_torsion.py` | Friction torque of the 1 mm pads about the contact normal under the 5 mm sphere of the UR5 cell, at the friction coefficient measured in the slides (scripts/ur5/ur5_shear_slides.py), against the torsion of the UR5 twists |
| `ur5_timelapse_short.py` | A short cut of the lab's overhead time-lapse of 11-02 to 11-03 (lab/timelapse_d435.py's frames, one every 2 s, made into lab/20261102-tipcal-timelapse_60fps.mp4: 35,401 frames, 9.8 min). |
| `ur5_tip_calibration_analysis.py` | Ground truth of a calibration session of the fingertip on the UR5 (src/sr2hand/driver/reskin5x/host/ ur5_tip_calibration.py), and the contact estimator scored against it. |
| `ur5_tip_calibration_board_gp.py` | The Gaussian process from the board's change to the contact point and the 3D force, pooled over UR5 sessions, fitted once and saved for the contact viewer (owner, 2026-10-26: ID 1 and ID 2 on the mux, "let's see if this  |
| `ur5_tip_calibration_cb1r_checks.py` | The new tests of cb1r (R24 on v1 ID 1, 2026-10-25) against cb1t, the same tip on 10-23: left-right twins, slides, twists at 5 N, and the rate set (owner, 10-25: "did we get good results from those new calibration tests?" |
| `ur5_tip_calibration_cb2t_checks.py` | The tests of cb2t (R24 on v1 ID 2, mount v7, 2026-10-25, passes 1 to 6) with its own grid as the training set, in the terms of scripts/ur5/ur5_tip_calibration_cb1r_checks.py, and two more: density location error (mm) at  |
| `ur5_tip_calibration_centre_coverage.py` | The calibrated GP's location error at the face centre, four magnets (v3 ID 1) against five (v2 ID 1), on the sphere presses both tips ran (owner, 2026-10-28: "do you think we can improve this error with greater coverage  |
| `ur5_tip_calibration_check.py` | Check of a calibration session on the UR5 (CB3) against Table 2 of the runbook (docs/experiments/20261022-ur5_tip_calibration/README.md), stage by stage, from whatever the session folder has. |
| `ur5_tip_calibration_coverage.py` | Press coverage of the pad by region (owner, 2026-10-28: "it seems the density is not that good even on the easily reachable space"): for each set of sessions, the share of the FE surface points (20261020-b_v2_surface_bas |
| `ur5_tip_calibration_day_repeat.py` | Last night's predictors on presses made a day later (QUEUE.md R21; owner, 2026-10-23: "we can run the 40 minute test then"): every predictor of scripts/ur5/ur5_tip_calibration_predict.py trained on all presses of one ses |
| `ur5_tip_calibration_density_curve.py` | Location error against the density of press targets on one tip (owner, 2026-10-26, through the lab session: "seems like increased grid points really helped. |
| `ur5_tip_calibration_direction.py` | Contact direction and location by region on the four checkerboard tips (owner, 2026-10-27: "are we any better at contact direction? |
| `ur5_tip_calibration_drift.py` | The board's zero over a calibration session of the fingertip on the UR5: how it drifts, what a press leaves behind, and what subtracting it buys the viewer's estimator (owner, 2026-10-23: "track drift over time and see i |
| `ur5_tip_calibration_drift_page.py` | Page of the board's drift over the lab's calibration session sphere2 (2026-10-22): the zero over time, what a press leaves, whether it can be learned, and the viewer's estimator against each zero. |
| `ur5_tip_calibration_drift_text.py` | The prose of the board-drift page (scripts/ur5/ur5_tip_calibration_drift_page.py), from the study's numbers R. |
| `ur5_tip_calibration_estimator_checks.py` | Checks of the fingertip's physical estimator on a UR5 session (owner, 2026-10-25: "it would be nice to not have to rely on the GP/MLP"; "since we have a good orientation estimator, doesnt that mean we can estimate wrench |
| `ur5_tip_calibration_fe.py` | The pad's FE model against the calibration session's ground truth: for every press hold, the normal force that the FE model (scripts/fingertip/fingertip_wrench_fe.py --surface, docs/experiments/20261020-fingertip_contact |
| `ur5_tip_calibration_fe_calibration.py` | The FE pad calibrated to the real tips on the UR5 (owner, 2026-10-28: "rebuild the FE surface basis ... |
| `ur5_tip_calibration_film.py` | Film of a fingertip calibration session on the UR5 (CB3): the session's joint record (robot.bin) replayed in MuJoCo on the URDF's link meshes with the tool stack (OptoForce HEX-E, UR5 mount v3, DIP link b v4 with cover,  |
| `ur5_tip_calibration_four_vs_five.py` | Four magnets (v3 ID 1, pad 05R4) against five (v2 ID 1, the b_v2 checkerboard) on the UR5's low sphere: the same press actions of the sphere stage, scored on each tip's own holds (owner, 2026-10-27: v3 ID 1's suite decid |
| `ur5_tip_calibration_gain_by_region.py` | The board's signal per newton on each region of the pad, for one or more analysed sessions (2026-10-29: whether the tip's end, first pressed on v3 ID 2 in the low sphere's top-edge pass of 10-28 19:21, gives as much sign |
| `ur5_tip_calibration_hard_extrapolation.py` | Extrapolation of the fingertip's contact estimators from light to hard presses, on v2 ID 1's sphere session on the UR5 (owner, 2026-10-27: "can we parameter tune the physical model to better match the resolution of the G |
| `ur5_tip_calibration_hard_pass.py` | Hard and fast contacts on v1 ID 3 (cb3t, mount v7), request R26 (owner, 2026-10-26 13:5x: "maybe we can add harder (much harder?) presses on id 3 ... |
| `ur5_tip_calibration_holder_apex_fit.py` | A held apex from the holder plan's apex search, and the holder's frame through the measured apexes (owner, 2026-10-27: the first touch-offs found no contact 1.2 mm past the expected apex; the plywood is level). |
| `ur5_tip_calibration_holder_paths.py` | The moves of a holder plan as the lab program makes them, checked for the UR5's singularities, its joint limits and the cell's clearances (owner, 2026-10-27, through the lab: "a lot of the rotated motions where the wrist |
| `ur5_tip_calibration_holder_plan.py` | UR5 plan on the fingertip holder (owner, 2026-10-27): the crosstalk field map, with the robot tip's apex held without contact at known poses off two held tips that carry their own boards, and the touch-offs that fix its  |
| `ur5_tip_calibration_k_check.py` | The magnets' strength k against their depth in the rest fit of a UR5 session's tip (owner, 2026-10-25, on the first checkerboard tip: "the magnets are the exact same as the past tip. |
| `ur5_tip_calibration_lab_page.py` | Page of the first calibration runs of the fingertip on the lab's UR5 (2026-10-22): the runs, the HEX-E's link, the wrist's stop at 19:59 and the checks it led to, the twist that reached the homing key, the pad's touch-of |
| `ur5_tip_calibration_lab_setup.sh` | Setup check of the UR5's computer for the fingertip calibration session (runbook: docs/experiments/20261022-ur5_tip_calibration/README.md). |
| `ur5_tip_calibration_motion_value.py` | What each motion of the UR5 calibration suite adds on the four-magnet v3 tips (owner, 2026-10-29: the indenter plate waits "until we have 100% finalized all the motions we need for effective calibration"), from v3 ID 1's |
| `ur5_tip_calibration_next_tip_suite.py` | The night suite for the next four-magnet tip on the UR5 cell (owner, 2026-10-28 13:4x: runs done before 19:00 "so i can swap fingertips for the other 4 magnet fingertips"): v3 ID 1's plan files as they ran or were checke |
| `ur5_tip_calibration_offset_touches.py` | Offset touch-offs added to a plan's pad stage, so that the registration sees the sphere's place along the finger. |
| `ur5_tip_calibration_pads.py` | The calibrated pad as the simulations' 1 mm sphere pads, against a calibration session's press holds: the pads' normal force on the indenter at the registered positions, in MuJoCo and in the pads' spring law, compared wi |
| `ur5_tip_calibration_page.py` | Page of the fingertip's calibration protocol on the lab's UR5 (CB3) with the OptoForce HEX-E (2026-10-22): the tool stack and its TCP, the cell, the session's stages, the ground truth they give and a whole session run ag |
| `ur5_tip_calibration_page_text.py` | Text of the calibration protocol page (scripts/ur5/ur5_tip_calibration_page.py builds it): sections(res, plan, h) returns the sections as (id, heading, html), the lede and the summary for the docs index; numbers come fro |
| `ur5_tip_calibration_pass6.py` | Pass 6 of R24 on cb2t (v1 ID 2 on mount v7), after pass 5 (owner, 2026-10-25 16:3x: "improvisationally devise more tests ... |
| `ur5_tip_calibration_plan.py` | Plan of the fingertip's calibration session on the lab's UR5 (CB3) with the OptoForce HEX-E, for src/sr2hand/driver/reskin5x/host/ur5_tip_calibration.py: the nominal frames of the tool stack, the pad's surface, the targe |
| `ur5_tip_calibration_pooled_checks.py` | Whether ID 2's data stand in for a recalibration of v1 ID 1 (owner, 2026-10-26: "you think it's worth putting the v1 id 1 fingertip back on after all this added calibration?"). |
| `ur5_tip_calibration_predict.py` | Predictors of the contact point (mm, pad frame) and the normal force (N) from the board's change at each hold of a calibration session on the UR5, trained on the session's own presses in time order (owner, 2026-10-22: "a |
| `ur5_tip_calibration_r24_pass3.py` | Pass 3 of R24 on the UR5 (owner, 2026-10-25): left-right twins of the grid, slides, twists at 5 N, and a final reference. |
| `ur5_tip_calibration_results_page.py` | Page of the fingertip's ground truth and contact predictors from a UR5 calibration session (2026-10-22): the pad's compliance against its FE model, the board's response against the HEX-E, the pivot check, the viewer's es |
| `ur5_tip_calibration_rig_coverage.py` | Can the inverted rig's presses widen the calibration of v3 IDs 1 to 3 (owner, 2026-11-04 23:1x: "can we see if the inverted rig presses, while they did not help the ids 4-6, can be used to improve the coverage of ids 1-3 |
| `ur5_tip_calibration_rod_pass2.py` | Rod pass 2 of R27 on v1 ID 3 (rod3t, mount v7), with the owner present (owner, 2026-10-26 15:05: "i think it can't hurt to get higher resolution + some data parity on the rod ... |
| `ur5_tip_calibration_rod_registration.py` | Registration of a rod session with the rod as a cylinder (QUEUE.md R19, 2026-10-23). |
| `ur5_tip_calibration_rod_tips.py` | How well v2 ID 1 (five N52 magnets) senses on the rod indenter, against ID 3's rod (owner, 2026-10-27 12:2x, through the lab session: "pass that analysis off to the main work pc session"). |
| `ur5_tip_calibration_scaled_transfer.py` | Whether the v1 tips' presses serve v2 ID 1 (five N52 magnets) once each tip's board change is brought to v2 ID 1's scale by the ratio of the magnets' strengths k fitted at rest (calib_rest.json). |
| `ur5_tip_calibration_seed_registration.py` | A seed registration folder for a new tip's pass on an indenter that is already registered (the lab, 2026-10-28 14:24: the program resumes a folder that holds only a seed registration.json, with --stages preflight,pad,piv |
| `ur5_tip_calibration_sphere_check.py` | Reference presses on the sphere fixture after R29 (owner, 2026-10-27 18:2x, through the lab: "we can build it and try to run it after r29 finishes"). |
| `ur5_tip_calibration_sphere_top_edge.py` | A short top-edge pass on the low sphere (owner, 2026-10-28 18:0x, through the lab: "i think you should do the top edge calbiration on the lower sphere, we are close to joint limits on the tall indenter"; "ok try again on |
| `ur5_tip_calibration_suite_value.py` | What each part of v2 ID 1's UR5 calibration suite cost in robot time and what it added to the GP's estimates, as the basis of a shorter suite for the four-magnet v3 tips (owner, 2026-10-27: "can you intelligently figure  |
| `ur5_tip_calibration_tall_pass.py` | R29 on the UR5: presses and slides where v2 ID 1's pad has no sphere press nearby and the tall indenter reaches, and twists to 20 deg at face and edge points (owner, 2026-10-26: "seems like contact density is a big one. |
| `ur5_tip_calibration_tall_r29.py` | v2 ID 1's presses on the tall indenter (R29, 2026-10-27) against its sphere presses (192158-v2id1). |
| `ur5_tip_calibration_tip_transfer.py` | One fingertip's predictors on another fingertip pressed on the UR5 (QUEUE.md R22, the first checkerboard tip; owner, 2026-10-23: "how transferable do you think this result will be to the higher strength n52 magnets + the |
| `ur5_tip_calibration_tip_vs_tip.py` | Two tips of one design on the same press actions (2026-10-29: v3 ID 2 against v3 ID 1, the two four-magnet tips on the UR5's low sphere). |
| `ur5_tip_calibration_top_edge_gp.py` | The calibrated GP on the tip's end (2026-10-29): how well the direction GP trained on the sphere's grid locates the presses of v3 ID 2's top-edge pass (10-28, 19:21 to 19:30: nine presses on the tip's end at 0.25 to 2 N  |
| `ur5_tip_calibration_top_edge_pass.py` | The top-edge pass of v3 ID 1 on the tall indenter, run with the owner present (owner, 2026-10-27: "we can calibrate the top edge finally. |
| `ur5_tip_calibration_touch_information.py` | How well a session's touch-offs fix the pad's place along the finger: the posterior standard deviation of the presses' contact points from the registration fit's covariance (src/.../ur5_tip_calibration.py fit_registratio |
| `ur5_tip_calibration_trimmed_cap_reach.py` | Reach of the tall indenter's presses with the trimmed tip cap (owner, 2026-10-27: "i trimmed the top cap so that there is no "fingertip" coming out to the surface of the fingertip now ... |
| `ur5_tip_calibration_twist_board.py` | Whether v2 ID 1's board shows the friction torque of the UR5 twists (owner, 2026-10-27: "you still think there's no good way to detect twists/wrenches?"). |
| `ur5_tip_calibration_v1_tips.py` | How well the three v1 tips sense, on their UR5 presses (owner, 2026-10-26 19:2x: "run a thorough set of analysis on the v1 id 3 data that we did get ... |
| `ur5_tip_calibration_v3_data_curve.py` | Held-out error of the board GP on the v3 tips against the number of training holds (lab, 11-02 14:59, for the owner's go on more runs: "if it's possible to get twists/wrenches, e.g. |
| `ur5_tip_calibration_v3_suite.py` | v3 ID 1's calibration suite on the UR5 cell (owner, 2026-10-27 21:5x, to the lab: "sure, we can run the v3 id 1 registration and suite ... |
| `ur5_tip_calibration_wrist_order.py` | Reorders a UR5 plan's actions so that no travel swings wrist 1 from negative to positive while wrist 2 rises (lab, 2026-10-30 00:13: three of the HEX-E's link losses came at the end of such travels in tall stage 2's grid |

## archive/ (140)

See `archive/README.md`.
