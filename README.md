# SR² Hand (`morphohand`)

A self-reconfigurable three-finger hand for sim-to-real verification of optimized morphologies:
search the finger **morphology** and the **grasp/manipulation** together in simulation, then put
the selected hands on the bench and measure whether the simulated order survives.

15 DoF, split the way the design problem splits:

- **morphology** — six gantry coordinates **m** = [x_T, y_T, x_I, y_I, x_M, y_M], the planar
  finger-base positions. Thumb 110 × 60 mm of travel, index and middle 60 × 60 mm, stepper-driven.
- **control** — nine joints, three per finger: MCP yaw, MCP pitch, PIP pitch, on SCS0009 servos.

Phalange length is a morphology parameter in the simulator's older generator but is **not** a
platform DoF; hardware morphology is the six gantry coordinates and nothing else.

Paper: `paper/hand_iros26-4.pdf`. Site: [sr2-hand.github.io](https://sr2-hand.github.io).

---

## What transfers, and what does not

The deployable method is **CEM grasp synthesis plus an open-loop geometric reorientation plan** —
two keyframes per morphology (grasp pose, terminal hold pose), the fingertip contacts rotated as a
rigid body about a raised pivot and mapped to joints by per-finger IK. Every hardware number in the
paper comes from that controller: 8,198 Sobol morphologies → 227 that pass the simulated feasibility
and retention screens → 8 built on the bench, 51/80 trials holding the cylinder through the turn,
sim-to-bench alignment ranking ρ_s = 0.50.

**The residual RL pipeline does not transfer.** The A/B policies (`aNN` lift/deliver, `bNN`
reorient) reorient only in simulation, only on the older `m05` finger geometry, and have never
driven the hardware. Four separate reasons, none of which is a tuning problem:

- On the `real_v1` topology the vertical hold pose lies outside the ±0.5 rad residual action
  budget the policies were trained with, so PPO cannot express the target at all.
- `b33` ignores its observations — replaying a fixed action sequence over the whole 66-dim input
  costs nothing. The learned reorienter is already open-loop, without the guarantees of one.
- Hardening contact toward hardware stiffness collapses it: `b33`'s reorient rate goes 0.91 →
  0.09 at solimp dmax 0.997 (`docs/experiments/HARDEN_B33.txt`). The rolling gait it learned
  is an artifact of the simulator's compliance.
- Domain randomization over compliance relocates the policy inside its band rather than widening
  it, so it buys nothing.

The AprilTag bench tracker now gives every observation column a hardware source, so a *sighted*
policy is trainable; that is a future direction, not a result. Treat `results/rl/REGISTRY.md` and
everything under `docs/rl/` as the simulation lineage.

---

## Where to start

| You want to… | Go to |
|---|---|
| Read the project as a document | **[`webpaper/`](webpaper/)** — Typst → HTML, the canonical readable write-up. `webpaper/build.sh`, then serve `webpaper/build/`. The **Control Station** page is the sim2real bring-up record |
| Read the submission | [`paper/`](paper/) — `hand_iros26-4.pdf` and its sources; `transfer_*.tex` are the hardware-transfer sections |
| Know what any script does | **[`scripts/README.md`](scripts/README.md)** — the map of the active scripts, grouped by role |
| Drive the real hand | **[`docs/hardware_control_station.md`](docs/hardware_control_station.md)** |
| Reproduce the transfer study | [`docs/experiments/20260831-real_v1-transfer-protocol/`](docs/experiments/20260831-real_v1-transfer-protocol/) (protocol) and [`20260901-real_v1-transfer-firstpass/`](docs/experiments/20260901-real_v1-transfer-firstpass/) (first-pass analysis) |
| Understand conventions and failure modes before editing | **[`CLAUDE.md`](CLAUDE.md)** — the load-bearing lessons, written down because relearning them is expensive |
| Read the RL history | [`docs/rl/reorientation.md`](docs/rl/reorientation.md) (chronological log). [`RESEARCH_STATE.md`](RESEARCH_STATE.md) is a handoff frozen at 2026-06-22 — historical, not current state |

---

## Naming

Three ID schemes, none interchangeable:

- **`D1`–`D8`** — the eight hands evaluated on hardware, numbered by their *simulated* rank so the
  predicted order is readable off a plot axis. The map to internal design tags (`sv1_w6689`, `g12`,
  `rv05_manual`, …) is in [`scripts/real_v1_transfer_ranking.py`](scripts/real_v1_transfer_ranking.py);
  the map to gantry coordinates is the paper appendix.
- **`aNN` / `bNN`** — the simulation-only policy registry (lift/deliver, reorient).
  [`scripts/rename_results_bids.sh`](scripts/rename_results_bids.sh) is the single source of truth;
  it regenerates `results/rl/REGISTRY.md`.
- **Design tags** — `sv1_*` (Sobol draw), `rv*`/`g*` (earlier hand-picked and gated designs),
  `m05`/`perp` (the retired simulation-only lineages).

Every generated folder is date-prefixed (`20260901-…`). Run logs go to `logs/`, experiment
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

Two rules that are not optional (see [`CLAUDE.md`](CLAUDE.md) for why):

- `MUJOCO_GL=egl` for headless rendering.
- **Every Warp process needs its own kernel cache**: `WARP_CACHE_PATH=$(mktemp -d)`. A shared
  cache races and produces NaNs. The `.sh` launchers set both.

There is one GPU (16 GB). Train sequentially; after killing a Warp run, wait for GPU memory to
fall to ~1 GB before relaunching. The AprilTag camera stack (`pupil_apriltags`, RealSense) lives in
a separate conda environment, not the uv one.

---

## Simulation: the design search

A design goes through the same chain every time. `scripts/README.md` documents each step:

```bash
# 1. bake a design into fixed geometry (a FROZEN morphology scene)
uv run --extra gpu python scripts/generate_morphology_xml.py ...

# 2. is it physically real? the mount rails run through the palm and nothing else checks
uv run --extra gpu python scripts/morph_selfcollision_gate.py --retarget ...

# 3. IK-retarget the grasp keyframe to this design (world-frame fingertips, NOT joint angles)
uv run --extra gpu python scripts/retarget_keyframe_ik.py ...

# 4. CEM grasp synthesis -> the "morphology run dir" every downstream script consumes
uv run --extra rl --extra gpu python scripts/phase1_optimize_grasp.py \
    --scene-xml <frozen_scene.xml> --keyframe open_ik --optimizer cem
```

Steps 1–3 are not optional preprocessing. Evaluating a design on an unfrozen scene lets the
morphology joints drift during the rollout, and transferring the grasp keyframe in joint space
instead of IK-retargeting it lands the fingertips in the wrong place — both produce confident,
wrong verdicts. See [`docs/frozen_scene_protocol.md`](docs/frozen_scene_protocol.md).

The `real_v1` hardware hand has its own chain — `scripts/real_v1_pipeline.py` and the `real_v1_*`
scripts — from Sobol sampling (`real_v1_sobol8192.sh`) through the open-loop screen, the finalist
selection, and plan export. Gate a design on **time to drop**, not on a mid-fall snapshot, and
score graspability on the **held** lift, never the peak.

Two gates run before any plan reaches the bench, and both have caught real failures:

```bash
uv run --extra gpu python scripts/real_v1_trajectory_clearance.py <plan.json>   # finger-finger
uv run --extra gpu python scripts/real_v1_export_plan.py ...                    # servo travel limits
```

## Simulation: the RL loop (simulation only)

`scripts/rl_train_cube.py` is the trainer; `configs/recipes/*.yaml` pin the known-good
configurations and the launchers select them:

```bash
scripts/train_A_on_morph.sh <morphology-run-dir>          # A, always from scratch per design
scripts/train_handoff_liveA_reset.sh <...>                # B, via the live-A reset
```

Judge a policy on the deterministic held-cos and the trajectory-health scorecard, never on reward
sums, and **look at it before explaining its numbers**:

```bash
uv run --extra rl --extra gpu python scripts/rl_demo_handoff_continuous.py --run <run>   # the deploy eval
uv run --extra rl --extra gpu python scripts/policy_filmstrip.py --run <run>             # frames to actually read
uv run --extra rl --extra gpu python scripts/policy_eval_suite.py --run <run>            # as a distribution, not one rollout
```

Aggregate reward hides late/idle fingers, two-finger pinches, jitter and de-centering; a "policy
that won't rotate" turned out to be rotating hard to the wrong pole. The `policy-eyes`,
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

Try the whole control station with no hardware — this runs the **real** driver stack against a
simulated M8P on a pty and a simulated servo bus:

```bash
PYTHONPATH=src/morphohand/driver/manta/host \
  uv run --no-project --with pyserial python -m manta_hand.web \
  --fake --host 127.0.0.1 --port 8765
```

Then open `http://127.0.0.1:8765`. (`--mock` also exists but replaces the whole backend, so it
exercises none of the driver code — use `--fake` for anything involving homing, motion or the
serial link.)

Real launch, CB1 install, the homing flow, the API, logging and recovery are in
[`docs/hardware_control_station.md`](docs/hardware_control_station.md) — that is the runbook.
Electrical, StallGuard and servo-calibration details are in
[`src/morphohand/driver/manta/docs/`](src/morphohand/driver/manta/docs/).

Facts worth knowing before you touch it:

- **Homing takes about two minutes** and is meant to. Each axis gets a timeout that guarantees it
  covered its full measured travel; four of six trip StallGuard2 in ~15 s and J3/J5 grind against
  their hardstop for ~33 s each. That is the current SGT tuning, not a fault.
- **The hand performs roughly half its commanded yaw.** Over 20 bench runs the yaw joints arrive at
  0.44–0.90 of the commanded angle under load while PIP arrives at 1.00. It is torque, not speed;
  running slower does not fix it. Every simulated ranking is therefore in *commanded* units.
- **Servo feedback is nine joint positions and nothing else** — no contact, no current (the SCS0009
  has no current register). 111 Hz for the nine positions, 75 Hz for a full write-plus-read loop,
  but only after lowering the FTDI latency timer; the default costs 16× and says nothing about it.
  Install the udev rule in the runbook.
- **Object pose comes from two AprilTags**, not the servos: a static reference tag and a 40 mm tag
  on a vane along the cylinder axis, 0.017° / 0.03 mm rms. `SIM_TO_BENCH_Z_MM` in
  `src/morphohand/bench/tags.py` is the only place the bench floor and the simulator's floor are
  tied together, through the fingertip datum.
- **Query the board before theorising.** One read-only `STATALL` has settled more hardware
  hypotheses here than any amount of reasoning about mechanics; a load pinned at exactly 200 is
  servo overload protection, which is configuration, not a mechanical ceiling.

---

## Repository layout

```text
assets/mjcf/            hand + scene MJCFs. baseline/, experimental/, real_v1 scenes
configs/recipes/        pinned trainer configurations (a_lift, b_liveA, b_liveA_imit)
docs/                   MkDocs reference docs; docs/rl/ is the RL engineering log,
                        docs/experiments/ holds dated experiment artifacts
paper/                  the IROS submission, its figures and figure generators
scripts/                active scripts (see scripts/README.md); archive/ holds superseded
                        ones — move them back to resurrect, don't run in place
src/morphohand/
  sampling/             morphology sampling, feasibility gating, scene freezing
  optimization/         CEM grasp synthesis and objective terms
  rl/                   env, trajectory-health scorecard, live-A runner, deploy builders
  bench/                AprilTag frames, bench↔sim datum, trial replay
  studies/              shared sweep/run plumbing
  driver/manta/         hardware: STM32 firmware (firmware/) + CB1 host package (host/);
                        manta_hand/plan.py is the sim design -> hardware conversion,
                        and the only place the servo-travel gate is enforced
tests/                  pytest suite
webpaper/               Typst -> HTML project site
results/                run outputs (gitignored except REGISTRY.md and summaries)
logs/                   run logs, sentinels, pids (gitignored)
```

## Tests

```bash
uv run --extra dev --with pyserial python -m pytest tests/ -q
```

The hardware tests need no hardware — `tests/test_manta_hardware_faults.py` runs the real driver
stack against `manta_hand.fake_hardware` — but they do need `pyserial`, which is not in the `dev`
extra; without it all 23 collect as errors. Clear `PYTHONPATH` first if ROS is on it.
