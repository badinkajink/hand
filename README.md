# SR² Hand

SR² Hand is a self-reconfigurable three-finger hand for sim-to-real verification of optimized
morphologies. We search morphology and control together in simulation, build the selected hands on
the bench, and measure whether the simulated ordering survives.

The platform has 15 DoF. Six are morphology: the planar finger-base coordinates
m = [x_T, y_T, x_I, y_I, x_M, y_M], driven by stepper gantries. The thumb base spans 110 × 60 mm,
the index and middle bases 60 × 60 mm. Nine are manipulation: MCP yaw, MCP pitch and PIP pitch per
finger, on SCS0009 servos.

The simulator's older generator also treats phalange length as a morphology parameter. The platform
does not. Hardware morphology is the six gantry coordinates.

The paper can be found at [`sr2-hand.github.io`](https://sr2-hand.github.io).

---

## Controllers

Hardware runs a CEM grasp followed by an open-loop geometric reorientation plan. For each
morphology we place the three fingertips in a near-grasp around the horizontal cylinder and
synthesize a force-closure pinch with CEM. The three contact targets are then rotated as a rigid
body about an axis raised above the contacts, and per-finger IK maps the moving targets to joint
commands. Joint-space interpolation from the grasp pose to the terminal IK solution reproduces the
full Cartesian trajectory closely enough that a plan is two keyframes.

Every hardware number in the paper comes from that controller. We sample 8,198 morphologies with a
scrambled Sobol sequence, filter to 227 that pass the simulated feasibility, retention and
perturbation screens, and build 8. Across 80 trials, 51 hold the cylinder through the turn.
Simulated and measured alignment rank at ρ_s = 0.50 (p = 0.21).

The residual RL pipeline has never driven the hardware. Its policies reorient in simulation, on the
older `m05` finger geometry, and four things keep them there:

- The vertical hold pose on `real_v1` lies outside the ±0.5 rad residual action budget the policies
  were trained with, so PPO cannot express the target.
- `b33` ignores its observations. Replaying a fixed action sequence over the whole 66-dim input
  costs nothing, so the learned reorienter is already open-loop.
- Contact hardening collapses it. `b33`'s reorient rate falls from 0.91 to 0.09 at solimp dmax
  0.997 ([`docs/experiments/HARDEN_B33.txt`](docs/experiments/HARDEN_B33.txt)). The rolling gait is
  an artifact of the simulator's compliance.
- Domain randomization over compliance relocates the policy inside its band rather than widening
  it.

The AprilTag tracker now supplies every observation column from hardware, so a sighted policy is
trainable. Nothing has been trained yet. Read `results/rl/REGISTRY.md` and `docs/rl/` as the
simulation lineage.

---

## Where to start

| You want to… | Go to |
|---|---|
| Read the project as a document | [`webpaper/`](webpaper/), a Typst site built by `webpaper/build.sh`. Its Control Station page is the sim2real bring-up record |
| Read the submission | [`paper/`](paper/). The `transfer_*.tex` files are the hardware-transfer sections |
| Know what any script does | [`scripts/README.md`](scripts/README.md), the map of the active scripts by role |
| Drive the real hand | [`docs/hardware_control_station.md`](docs/hardware_control_station.md) |
| Reproduce the transfer study | [`docs/experiments/20260831-real_v1-transfer-protocol/`](docs/experiments/20260831-real_v1-transfer-protocol/) for the protocol, [`20260901-real_v1-transfer-firstpass/`](docs/experiments/20260901-real_v1-transfer-firstpass/) for the first-pass analysis |
| Learn the conventions before editing | [`CLAUDE.md`](CLAUDE.md), which records the failure modes that were expensive to find |
| Read the RL history | [`docs/rl/reorientation.md`](docs/rl/reorientation.md). [`docs/RESEARCH_STATE.md`](docs/RESEARCH_STATE.md) is a handoff frozen at 2026-06-22 and describes no current work |

---

## Naming

Three ID schemes, none interchangeable.

`D1` through `D8` are the eight hands evaluated on hardware, numbered by their simulated rank so a
reader sees the predicted order on a plot axis. Their internal design tags (`sv1_w6689`, `g12`,
`rv05_manual` and so on) are mapped in
[`scripts/real_v1_transfer_ranking.py`](scripts/real_v1_transfer_ranking.py); their gantry
coordinates are in the paper appendix.

`aNN` and `bNN` are the simulation-only policy registry, lift/deliver and reorient.
[`scripts/rename_results_bids.sh`](scripts/rename_results_bids.sh) is the single source of truth
and regenerates `results/rl/REGISTRY.md`.

Design tags carry their origin: `sv1_*` is a Sobol draw, `rv*` and `g*` are earlier hand-picked and
gated designs, `m05` and `perp` are the retired simulation-only lineages.

Every generated folder is date-prefixed, as in `20260901-…`. Run logs go to `logs/`, experiment
summaries to `docs/experiments/`, free-form notes to `docs/notes/`.

---

## Environment

```bash
uv venv --python 3.10          # the project targets >=3.10,<3.14
uv sync --extra dev
```

Simulation and RL need the GPU extras, and every run goes through `uv run`:

```bash
uv run --extra rl --extra gpu python scripts/<script>.py
```

Two rules are not optional, for reasons [`CLAUDE.md`](CLAUDE.md) gives. Set `MUJOCO_GL=egl` for
headless rendering. Give every Warp process its own kernel cache with
`WARP_CACHE_PATH=$(mktemp -d)`, because a shared cache races and produces NaNs. The `.sh` launchers
set both.

There is one GPU, 16 GB. Train sequentially, and after killing a Warp run wait for GPU memory to
fall to about 1 GB before relaunching. The AprilTag camera stack (`pupil_apriltags`, RealSense)
lives in a separate conda environment.

---

## Design search

A design goes through the same chain every time. `scripts/README.md` documents each step.

```bash
# 1. bake a design into fixed geometry (a FROZEN morphology scene)
uv run --extra gpu python scripts/generate_morphology_xml.py ...

# 2. is it physically real? the mount rails run through the palm and nothing else checks
uv run --extra gpu python scripts/morph_selfcollision_gate.py --retarget ...

# 3. IK-retarget the grasp keyframe to this design (world-frame fingertips, NOT joint angles)
uv run --extra gpu python scripts/retarget_keyframe_ik.py ...

# 4. CEM grasp synthesis, writing the morphology run dir every downstream script consumes
uv run --extra rl --extra gpu python scripts/phase1_optimize_grasp.py \
    --scene-xml <frozen_scene.xml> --keyframe open_ik --optimizer cem
```

Steps 1 through 3 are not optional preprocessing. An unfrozen scene lets the morphology joints
drift during the rollout. A keyframe transferred in joint space instead of IK-retargeted lands the
fingertips in the wrong place. Both produce wrong verdicts. See
[`docs/frozen_scene_protocol.md`](docs/frozen_scene_protocol.md).

The `real_v1` hardware hand has its own chain in `scripts/real_v1_pipeline.py` and the `real_v1_*`
scripts, running from Sobol sampling (`real_v1_sobol8192.sh`) through the open-loop screen, the
finalist selection and plan export. Gate a design on time to drop rather than a mid-fall snapshot,
and score graspability on the held lift rather than the peak.

Two gates run before a plan reaches the bench, and both have caught real failures.

```bash
uv run --extra gpu python scripts/real_v1_trajectory_clearance.py <plan.json>   # finger-finger
uv run --extra gpu python scripts/real_v1_export_plan.py ...                    # servo travel limits
```

## Reinforcement learning (simulation only)

`scripts/rl_train_cube.py` is the trainer. `configs/recipes/*.yaml` pin the known-good
configurations and the launchers select them.

```bash
scripts/train_A_on_morph.sh <morphology-run-dir>          # A, always from scratch per design
scripts/train_handoff_liveA_reset.sh <...>                # B, via the live-A reset
```

Judge a policy on the deterministic held-cos and the trajectory-health scorecard, never on reward
sums, and look at it before explaining its numbers.

```bash
uv run --extra rl --extra gpu python scripts/rl_demo_handoff_continuous.py --run <run>   # the deploy eval
uv run --extra rl --extra gpu python scripts/policy_filmstrip.py --run <run>             # frames to read
uv run --extra rl --extra gpu python scripts/policy_eval_suite.py --run <run>            # a distribution, not one rollout
```

Aggregate reward hides late and idle fingers, two-finger pinches, jitter and de-centering. One
policy that appeared not to rotate was rotating hard to the wrong pole. The `policy-eyes`,
`policy-metrics`, `mujoco-eyes` and `morphology-scenes` skills under `.claude/skills/` wrap these
workflows.

---

## Hardware

```text
workstation                                CB1 (10.99.99.2)              hand
browser UI ───── HTTP/JSON ─────┐
plan runner    ─────────────────┴──> HandRuntime ── USB-CDC ──> 6 gantry steppers (Manta M8P)
                                       └────────── U2D2 TTL ──> 9 SCS0009 servos
RealSense D435 ──> AprilTag tracker ──> cylinder pose, offline
```

The whole control station runs with no hardware attached. This drives the real driver stack against
a simulated M8P on a pty and a simulated servo bus:

```bash
PYTHONPATH=src/morphohand/driver/manta/host \
  uv run --no-project --with pyserial python -m manta_hand.web \
  --fake --host 127.0.0.1 --port 8765
```

Then open `http://127.0.0.1:8765`. A `--mock` mode also exists, but it replaces the whole backend
and exercises none of the driver code, so use `--fake` for anything involving homing, motion or the
serial link.

Real launch, CB1 install, the homing flow, the API, logging and recovery are all in the runbook,
[`docs/hardware_control_station.md`](docs/hardware_control_station.md). Electrical, StallGuard and
servo-calibration details are in
[`src/morphohand/driver/manta/docs/`](src/morphohand/driver/manta/docs/).

Five things about the bench:

- Homing takes about two minutes and is meant to. Each axis gets a timeout that guarantees it
  covered its full measured travel. Four of six trip StallGuard2 in about 15 s, and J3 and J5 grind
  against their hardstop for about 33 s each. This is the current SGT tuning.
- The hand performs roughly half its commanded yaw. Over 20 bench runs the yaw joints arrived at
  0.44 to 0.90 of the commanded angle under load, while PIP arrived at 1.00. The cause is torque,
  so running slower does not help. Every simulated ranking is therefore in commanded units.
- Servo feedback is nine joint positions. There is no contact sensing and no current, since the
  SCS0009 has no current register. The bus reads the nine positions at 111 Hz and sustains 75 Hz
  for a full write-plus-read loop, but only once the FTDI latency timer is lowered; the default
  costs a factor of 16. Install the udev rule in the runbook.
- Object pose comes from two AprilTags. A static reference tag supplies world up and an absolute
  datum, and a 40 mm tag on a vane along the cylinder axis supplies the shaft pose, at 0.017° and
  0.03 mm rms. `SIM_TO_BENCH_Z_MM` in `src/morphohand/bench/tags.py` ties the bench floor to the
  simulator's floor through the fingertip datum.
- Query the board before theorising. A read-only `STATALL` has settled several hardware hypotheses
  here. A load pinned at exactly 200 is servo overload protection, which is a configuration value.

---

## Repository layout

```text
assets/mjcf/            hand + scene MJCFs. baseline/, experimental/, real_v1 scenes
configs/recipes/        pinned trainer configurations (a_lift, b_liveA, b_liveA_imit)
docs/                   MkDocs reference docs; docs/rl/ is the RL engineering log,
                        docs/experiments/ holds dated experiment artifacts
paper/                  the IROS submission, its figures and figure generators
scripts/                active scripts (see scripts/README.md); archive/ holds superseded
                        ones. Move them back to resurrect, don't run in place
src/morphohand/
  sampling/             morphology sampling, feasibility gating, scene freezing
  optimization/         CEM grasp synthesis and objective terms
  rl/                   env, trajectory-health scorecard, live-A runner, deploy builders
  bench/                AprilTag frames, bench/sim datum, trial replay
  studies/              shared sweep/run plumbing
  driver/manta/         hardware: STM32 firmware (firmware/) + CB1 host package (host/).
                        manta_hand/plan.py converts a sim design to hardware and is the
                        only place the servo-travel gate is enforced
tests/                  pytest suite
webpaper/               Typst -> HTML project site
results/                run outputs (gitignored except REGISTRY.md and summaries)
logs/                   run logs, sentinels, pids (gitignored)
```

## Tests

```bash
uv run --extra dev --with pyserial python -m pytest tests/ -q
```

The hardware tests need no hardware. `tests/test_manta_hardware_faults.py` runs the real driver
stack against `manta_hand.fake_hardware`. They do need `pyserial`, which the `dev` extra omits;
without it all 23 collect as errors. Clear `PYTHONPATH` first if ROS is on it.
