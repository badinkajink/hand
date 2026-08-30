# MorphoHand

Co-design of a three-finger reconfigurable hand: optimise the finger **morphology** and the
**grasp/manipulation** together, in simulation, and put the result on real hardware.

Two coupled parameter sets, and the whole repo is organised around the split:

- **morphology** — per finger `(x, y, length)`: where the finger mounts on the palm and how long it is
- **control** — per finger `(yaw, mcp, pip)`: the three joints it actuates

There are two simulation tracks (CEM grasp synthesis over a parametric hand, and RL for
lift → in-hand reorientation of a screwdriver) and one hardware track (`real_v1`: six morphology
gantry axes on a Manta M8P, nine Feetech servos, and a browser control station).

---

## Where to start

| You want to… | Go to |
|---|---|
| Read the project as a document | **[`webpaper/`](webpaper/)** — Typst → HTML, the canonical readable write-up. `webpaper/build.sh`, then serve `webpaper/build/`. The **Control Station** page is the sim2real bring-up record and catalogues the study artifacts |
| Know what any script does | **[`scripts/README.md`](scripts/README.md)** — the map of the ~80 active scripts, grouped by role |
| Pick up the RL work | **[`RESEARCH_STATE.md`](RESEARCH_STATE.md)** (self-contained handoff) then [`docs/rl/reorientation.md`](docs/rl/reorientation.md) (the chronological log, the RL source of truth) |
| Drive the real hand | **[`docs/hardware_control_station.md`](docs/hardware_control_station.md)** |
| Understand conventions and failure modes before editing | **[`CLAUDE.md`](CLAUDE.md)** — the load-bearing lessons, written down because relearning them is expensive |
| Find a trained policy | [`results/rl/REGISTRY.md`](results/rl/REGISTRY.md) — the `aNN`/`bNN` policy registry |

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
fall to ~1 GB before relaunching.

---

## Simulation: the grasp/morphology loop

A design goes through the same chain every time. `scripts/README.md` documents each step; the
short version:

```bash
# 1. bake a 9-param design into fixed geometry (a FROZEN morphology scene)
uv run --extra gpu python scripts/generate_morphology_xml.py ...

# 2. is it physically real? the mount rails run through the palm and nothing else checks
uv run --extra gpu python scripts/morph_selfcollision_gate.py --retarget ...

# 3. IK-retarget the grasp keyframe to this design (world-frame fingertips, NOT joint angles)
uv run --extra gpu python scripts/retarget_keyframe_ik.py ...

# 4. CEM grasp synthesis -> the "morphology run dir" every RL script consumes
uv run --extra rl --extra gpu python scripts/phase1_optimize_grasp.py \
    --scene-xml <frozen_scene.xml> --keyframe open_ik --optimizer cem
```

Steps 1–3 are not optional preprocessing. Evaluating a design on an unfrozen scene lets the
morphology joints drift during the rollout, and transferring the grasp keyframe in joint space
instead of IK-retargeting it lands the fingertips in the wrong place — both produce confident,
wrong verdicts. See [`docs/frozen_scene_protocol.md`](docs/frozen_scene_protocol.md).

The `real_v1` hardware hand has its own chain (different topology, nothing transfers from the
m05/baseline lineage) — `scripts/real_v1_pipeline.py` and the `real_v1_*` scripts.

## Simulation: the RL loop

Policy **A** lifts and delivers; policy **B** reorients. `scripts/rl_train_cube.py` is the
trainer; the launchers pin the known-good configurations as recipes:

```bash
scripts/train_A_on_morph.sh <morphology-run-dir>          # A, always from scratch per design
scripts/train_handoff_liveA_reset.sh <...>                # B, via the live-A reset
```

Judge a policy on the deterministic held-cos and the trajectory-health scorecard, never on
reward sums, and **look at it before explaining its numbers**:

```bash
uv run --extra rl --extra gpu python scripts/rl_demo_handoff_continuous.py --run <run>   # the deploy eval
uv run --extra rl --extra gpu python scripts/policy_filmstrip.py --run <run>             # frames to actually read
uv run --extra rl --extra gpu python scripts/policy_eval_suite.py --run <run>            # as a distribution, not one rollout
```

Aggregate reward hides late/idle fingers, two-finger pinches, jitter and de-centering; a
"policy that won't rotate" turned out to be rotating hard to the wrong pole. The
`policy-eyes`, `policy-metrics`, `mujoco-eyes` and `morphology-scenes` skills under
`.claude/skills/` wrap these workflows.

---

## Hardware: the `real_v1` hand

```text
workstation                                CB1 (10.99.99.2)              hand
browser UI ───── HTTP/JSON ─────┐
policy client  ─────────────────┴──> HandRuntime ── USB-CDC ──> 6 gantry steppers (Manta M8P)
                                       └────────── U2D2 TTL ──> 9 SCS0009 servos
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
The reasoning behind it, the 2026-08-29 bench failure, the measured bus limits and a catalogue
of the `real_v1` study artifacts are on the **Control Station** page of the webpaper
([`webpaper/src/control.typ`](webpaper/src/control.typ)). Electrical, StallGuard and
servo-calibration details are in
[`src/morphohand/driver/manta/docs/`](src/morphohand/driver/manta/docs/).

Two hardware facts worth knowing before you touch it:

- **Homing takes about two minutes** and is meant to. Each axis gets a timeout that guarantees
  it covered its full measured travel; four of six trip StallGuard2 in ~15 s and J3/J5 grind
  against their hardstop for ~33 s each. That is the current SGT tuning, not a fault.
- **Feedback is nine joint positions and nothing else** — no object pose, no contact, no
  current (the SCS0009 has no current register at all). Measured ceiling is 111 Hz for the nine
  positions, and 75 Hz for a full write-plus-read closed loop — but only after lowering the FTDI
  latency timer, which the default costs 16× and says nothing about. Install the udev rule in the
  runbook. The learned A/B policies need ~65 observations, so they remain simulation-only; the
  deployable method is the CEM grasp plus a buffered open-loop reorientation.

---

## Repository layout

```text
assets/mjcf/            hand + scene MJCFs. baseline/, experimental/, real_v1 scenes
configs/recipes/        pinned trainer configurations (a_lift, b_liveA, b_liveA_imit)
docs/                   MkDocs reference docs; docs/rl/ is the RL engineering log,
                        docs/experiments/ holds dated experiment artifacts
scripts/                ~80 active scripts (see scripts/README.md); archive/ holds
                        superseded ones — move them back to resurrect, don't run in place
src/morphohand/
  sampling/             morphology sampling, feasibility gating, scene freezing
  optimization/         CEM grasp synthesis and objective terms
  rl/                   env, trajectory-health scorecard, live-A runner, deploy builders
  studies/              shared sweep/run plumbing
  driver/manta/         hardware: STM32 firmware (firmware/) + CB1 host package (host/)
tests/                  pytest suite
webpaper/               Typst -> HTML project site
results/                run outputs (gitignored except REGISTRY.md and summaries)
logs/                   run logs, sentinels, pids (gitignored)
```

Naming rules that the tooling depends on: **date-prefix every generated folder**
(`20260829-...`), keep run logs in `logs/` and experiment summaries in `docs/experiments/`, and
use the `aNN`/`bNN` policy IDs everywhere — `scripts/rename_results_bids.sh` is the single
source of truth for those.

## Tests

```bash
uv run --extra dev python -m pytest tests/ -q
```

The hardware tests need no hardware: `tests/test_manta_hardware_faults.py` runs the real driver
stack against `manta_hand.fake_hardware`.

## Documentation surfaces

- **[`webpaper/`](webpaper/)** — Typst → static HTML, the canonical readable write-up
  (foundation → experimentation → results → analysis, with collapsible detail sections).
  `webpaper/build.sh`, then `python3 -m http.server -d webpaper/build 8080`.
- **[`docs/`](docs/)** — MkDocs reference docs and the chronological engineering logs.
  `uv run mkdocs serve`.
- **[`paper/`](paper/) and [`hand_paper/`](hand_paper/)** — LaTeX papers (simulation/morphology
  stack; hardware).
- **[`RESEARCH_STATE.md`](RESEARCH_STATE.md)** — the living RL handoff.
