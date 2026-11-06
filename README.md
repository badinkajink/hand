# SR² Hand

SR² Hand is a self-reconfigurable three-finger hand for sim-to-real verification of optimized morphologies. We search morphology and control together in simulation, configure the selected hands on hardware, and measure simulation fidelity.

The platform has 15 DoF. Six are "morphology" DoFs: the planar finger-base coordinates
m = [x_T, y_T, x_I, y_I, x_M, y_M] which are driven by stepper gantries. The thumb base spans 110 × 60 mm, and the
index and middle bases span 60 × 60 mm. Nine servo-driven DoFs are "control": MCP yaw, MCP pitch and PIP pitch per
finger. Each fingertip is sensorized to provide contact pose and three-axis force via Hall effect sensing of deformation of a magnetic field created by magnets embedded in a compliant, printed fingertip.

This repository contains modeling, simulation, and controls for the platform and its many configurations..

The paper and CAD can be found at [`sr2-hand.github.io`](https://sr2-hand.github.io).


The documentation starts at [`docs/index.html`](docs/index.html): open it in a browser from a clone, with no server or build step.

## Documentation and result pages

The home page links eight topic overviews in [`docs/overviews/`](docs/overviews/), and they link the dated result
pages in `docs/experiments/`. [`docs/pages.html`](docs/pages.html) searches every result page by title and summary.
`docs/experiments/INDEX.md` lists them by date and topic, and [`docs/changelog.html`](docs/changelog.html) lists each
page's commits. A result page is written by a builder script from its study's own data, with its media inlined, so
it opens as one file. Change a page through its builder, never by editing the HTML; the page index names the builder.

Notes between work sessions are in `docs/handoff/`, free-form notes in `docs/notes/`, and the reinforcement-learning
log in `docs/rl/`. [`CLAUDE.md`](CLAUDE.md) records the conventions and the failure modes that were expensive to find.
[`docs/RESEARCH_STATE.md`](docs/RESEARCH_STATE.md) is a hand-off frozen at 2026-06-22 and describes no current work.

## Cloning, large files and submodules

Code, models, paper sources and the overview pages are ordinary git files. Result pages, figures, videos and
data rows under `docs/` are stored with Git LFS, and a plain clone (about 30 MB) leaves them as small pointer
files:

```bash
git clone git@github.com:badinkajink/hand.git
```

Fetch them with Git LFS, one result folder or all of them (about 1.3 GB):

```bash
git lfs install
git -c lfs.fetchexclude= lfs pull --include="docs/experiments/20261008-hand_object_scale/**"
git -c lfs.fetchexclude= lfs pull
```

`git config lfs.fetchexclude ""` makes every later pull fetch them. History before 2026-10-09, including the
tracked `results/` runs and the retired experiment branches, is in the private repository
`badinkajink/hand-archive`.

MuJoCo-Warp, ComFree-Warp and the eFlesh fork are git submodules in `external/`. The `gpu` extra installs the first
two from there as editable packages:

```bash
git submodule update --init external/mujoco_warp external/comfree_warp external/eFlesh
```

## Python environments

Three Python environments run the code.

| environment | what runs in it |
|---|---|
| `.venv`, made by uv (Python 3.12) | simulation, RL, page builds and tests |
| `~/miniconda3/bin/python` | the RealSense camera, the AprilTag tracker and the fingertip viewer |
| `~/miniconda3/envs/femtip` | finite elements of the fingertip (scikit-fem, dolfinx, gmsh) and the UR5 rig scripts; it has no MuJoCo |

```bash
uv venv --python 3.12          # the project targets >=3.10,<3.14
uv sync --extra dev --extra gpu --extra rl
uv run --extra rl --extra gpu python scripts/<folder>/<script>.py
```

Result-page builders run with `.venv/bin/python`. Set `MUJOCO_GL=egl` for headless rendering. Give every Warp
process its own kernel cache with `WARP_CACHE_PATH=$(mktemp -d)`, because a shared cache races and produces NaNs. The
`.sh` launchers set both. There is one 16 GB GPU, so train one run at a time. After killing a Warp run, wait until GPU
memory falls to about 1 GB before starting the next.

## Repository layout

```text
assets/mjcf/            hand and scene MJCFs: real_v1 (the hardware hand), experimental/ (dated scene sets),
                        baseline and perp (retired simulation lineages)
assets/cad_v1, cad_v2/  parts of the CAD hand, as STEP and STL
configs/recipes/        pinned trainer configurations
docs/                   the documentation, described above
external/               submodules: mujoco_warp, comfree_warp, eFlesh
paper/                  the IROS submission (hand_iros26-4.pdf), its sources and figures
scripts/                one folder per topic: docs, design, hom, mechanisms, hardware, contact, fingertip,
                        ur5, cad, rl; scripts/README.md maps them, and archive/ has the retired ones
src/sr2hand/            the package: backends, bench, control (HOM), geometry, numerics, optimization, rl, sampling,
                        studies, tools; driver/manta (gantry and servo firmware, the CB1 host) and
                        driver/reskin5x (fingertip board firmware, the viewer, the UR5 session program)
tests/                  pytest suite
results/, logs/         run outputs and run logs (gitignored)
```

## Simulation defaults

- The fingertip on the CAD hand is the v4b tip that is being calibrated on the UR5 (`SCALE_HAND=cad_v2`, built by
  `scripts/hardware/build_cad_v2_finger.py`). It is covered with 1 mm sphere pads.
- Each pad's stiffness follows pad builder v2 from one 2.5 MPa modulus (`reorient_backends.E_TPU`). Runs before
  2026-10-30 used 10 MPa, and their frozen scenes keep it.
- A scene is frozen before any grasp or policy is evaluated, and the grasp keyframe is retargeted to each layout by
  inverse kinematics on world-frame fingertip targets. An unfrozen scene lets the morphology joints drift during the
  rollout.
- The HOM scripts take their servo plant from `HOS_PLANT`. The best layouts change with the plant and with the scene
  friction, so a landscape is quoted with both.

[`CLAUDE.md`](CLAUDE.md) gives the flags and the history of each default.

## Design search on the real_v1 hand

`scripts/design/real_v1_pipeline.py` takes one design through the pose fit, the CEM grasp, the lift and turn policies and
the hand-over evaluation. It is resumable and stage-selectable. The population comes from
`scripts/design/real_v1_sobol8192.sh`. Two gates run before a plan reaches the bench, and both have caught real failures:

```bash
uv run --extra gpu python scripts/hardware/real_v1_trajectory_clearance.py <plan.json>   # finger-to-finger clearance along the path
uv run --extra gpu python scripts/hardware/real_v1_export_plan.py ...                    # servo travel limits
```

A plan's `chord` and its `_traj.csv` are different paths, and the clearance gate checks each one.

## Hand-object control and layout landscapes

`scripts/hom/hand_object_scale_sim.py` scores grasps and in-hand turns over finger layouts and object sizes on the CAD
hand. `scripts/hom/hom_landscape.py` and `scripts/hom/hom_gpu.py` run the layout landscapes on the CPU and on MuJoCo-Warp.
`HOS_PLANT` selects the servo plant (`servo` is the SCS0009). `HOS_SCENES` points at a fresh scene cache, and
`HOM_ROWS_TAG`, required with either, is appended to the row directories. `scripts/hom/hom_plant_transfer.py` compares
landscapes across plants, pad moduli and scene friction.

## Fingertip calibration on the UR5

The UR5, its HEX-E force-torque sensor and the ATI Net box are driven by the lab computer corvus, which has its own
clone. Its runbook is
[`docs/experiments/20261022-ur5_tip_calibration/lab/README.md`](docs/experiments/20261022-ur5_tip_calibration/lab/README.md),
and `lab/STATE.md` beside it gives the present state of the cell. Plans are made and path-checked on the workstation
(`scripts/ur5/ur5_tip_calibration_holder_paths.py`) and reach corvus by git. The session program and its simulator are
`src/sr2hand/driver/reskin5x/host/ur5_tip_calibration.py` and `ur5_tip_calibration_sim.py`.

## The bench hand

```text
workstation                                CB1 (10.99.99.2)              hand
browser UI ───── HTTP/JSON ─────┐
plan runner    ─────────────────┴──> HandRuntime ── USB-CDC ──> 6 gantry steppers (Manta M8P)
                                       └────────── U2D2 TTL ──> 9 SCS0009 servos
RealSense D435 ──> AprilTag tracker ──> cylinder pose, offline
```

The whole control station runs with no hardware attached. This command drives the real driver stack against a
simulated M8P on a pty and a simulated servo bus:

```bash
PYTHONPATH=src/sr2hand/driver/manta/host \
  uv run --no-project --with pyserial python -m manta_hand.web \
  --fake --host 127.0.0.1 --port 8765
```

Then open `http://127.0.0.1:8765`. A `--mock` mode also exists, but it replaces the whole backend and exercises none
of the driver code, so use `--fake` for anything that involves homing, motion or the serial link.

The real launch, the CB1 install, the homing flow, the API, logging and recovery are in the runbook,
[`docs/hardware_control_station.md`](docs/hardware_control_station.md). Electrical, StallGuard and
servo-calibration details are in [`src/sr2hand/driver/manta/docs/`](src/sr2hand/driver/manta/docs/).

Five things about the bench:

- Homing takes about two minutes and is meant to. Each axis gets a timeout long enough to cover its full measured
  travel. Four of six trip StallGuard2 in about 15 s, and J3 and J5 grind against their hardstop for about 33 s
  each. This is the current SGT tuning.
- The hand reaches roughly half its commanded yaw. Over 20 bench runs the yaw joints arrived at 0.44 to 0.90 of the
  commanded angle under load, while PIP arrived at 1.00. The cause is torque, so running slower does not help.
  Every simulated ranking is therefore in commanded units.
- Servo feedback is nine joint positions. There is no contact sensing and no current, since the SCS0009 has no
  current register. The bus reads the nine positions at 111 Hz. It sustains 75 Hz for a full write-and-read loop
  once the FTDI latency timer is lowered; the default costs a factor of 16. Install the udev rule in the runbook.
- Object pose comes from two AprilTags. A static reference tag gives world up and an absolute datum, and a 40 mm tag
  on a vane along the cylinder axis gives the shaft pose, at 0.017° and 0.03 mm rms. `SIM_TO_BENCH_Z_MM` in
  `src/sr2hand/bench/tags.py` ties the bench floor to the simulator's floor through the fingertip datum.
- Query the board before theorising. A read-only `STATALL` has answered several hardware questions here. A load
  pinned at exactly 200 is servo overload protection, which is a configuration value.

## Fingertip viewer on the workstation

The ReSkin 5X fingertips plug into an I2C mux, and the mux into a board running
[`firmware/reskin5x_mux_stream`](src/sr2hand/driver/reskin5x/firmware/reskin5x_mux_stream/). One viewer reads that
board and shows every fingertip on the mux at `http://localhost:8790`. On 2026-10-27 two boards on the workstation had
this firmware. The mux on the SparkFun Thing Plus ESP32-C6 answered at 0x71, and the one on the QT Py ESP32-S3
b4:3a:45:b0:d9:94 at 0x70. `ls -l /dev/serial/by-id/` lists the boards plugged in.

The firmware scans the mux channels for fingertips once, when a program first opens the port after power-up. After a
fingertip is plugged in or unplugged, press the board's RESET button or replug its USB cable before starting the viewer.

To restart the viewer, stop the running one first, with Ctrl-C in its terminal or `kill -INT` on its PID. Its web server
then waits up to 60 s for open connections to close. Start the new viewer once `pgrep -af wrench_viewer.py` lists none:
two viewers on one board split its stream between them, and the second cannot open port 8790. A viewer whose board is
unplugged reopens on a QT Py's port if one is plugged in, else on the first `/dev/ttyACM*`. Stop it before plugging in
another board. Then, from the repository root and with the fingertips untouched:

```bash
pgrep -af wrench_viewer.py      # a running viewer: kill -INT the PID of its python line
cd ~/Programs/hand
BOARD=/dev/serial/by-id/usb-Espressif_USB_JTAG_serial_debug_unit_40:4C:CA:4C:49:C4-if00            # C6, mux 0x71
# BOARD=/dev/serial/by-id/usb-Adafruit_QT_Py_ESP32-S3__4MB_Flash_2MB_PSRAM__b4:3a:45:b0:d9:94-if00 # QT Py, mux 0x70
C=logs/20261020-contact_estimation/calib
ARGS=(--mux "$BOARD" --http 8790 --floor 1.0 --case four_05R4 --names "ID 1" "ID 2" "ID 3"
      --calib $C/20261102-v3_id_1-calib.json $C/20261102-v3_id_2-calib.json $C/20261102-v3_id_3-calib.json
      --correction logs/20261028-spherev3id1_twists logs/20261029-spherev3id2 logs/20261030-spherev3id3
      --board-gp docs/experiments/20261022-ur5_tip_calibration/data/20261102-board_gp_v3_id1_id3.npz)
~/miniconda3/bin/python src/sr2hand/driver/reskin5x/host/wrench_viewer.py "${ARGS[@]}"
```

A Claude session starts the viewer with the same `ARGS`, detached and under the memory guard, so that it outlives the
session:

```bash
nohup setsid ~/.claude/bin/resguard.sh run --mem 3G --cpu 200 -- ~/miniconda3/bin/python -u \
    src/sr2hand/driver/reskin5x/host/wrench_viewer.py "${ARGS[@]}" > logs/$(date +%Y%m%d-%H%M%S)-viewer.log 2>&1 &
```

Names, calibrations and UR5 corrections go one per board, in the order of the mux channels, with `-` for none. The
arguments above are the three v3 fingertips of 2026-11-02, ID 1, ID 2 and ID 3 on channels 0, 1 and 3 of the C6's mux.
`--case four_05R4` selects their pad, four magnets in a checkerboard. Each tip's calibration is its UR5 rest fit with
the board's offset limited to 0.43 to 1.03 mm, its correction is learned on its sphere session's presses, and the board
GP is pooled over the sphere and rod sessions of all three tips. At start the viewer prints the channels with a board
and the name given to each, as `boards on mux channels [0, 1, 3]: ID 1, ID 2, ID 3`. A board without a name is
labelled by its channel. Each fingertip's rest is calibrated in its first 2 s, so the pads stay untouched then. On
2026-11-02 the three UR5 corrections were learned in 21 to 27 s. `--floor 1.0` sets the noise floor to 1 µT; the
default is 2 µT. The calibrations and the UR5 sessions under `logs/` exist only on this workstation, because `logs/` is
not tracked by git. Stream logs and new calibrations are written to `logs/20261020-contact_estimation/stream/` and
`calib/`. The flags, the one-board firmware and the estimates are described in
[`src/sr2hand/driver/reskin5x/README.md`](src/sr2hand/driver/reskin5x/README.md).

The three five-magnet v1 fingertips of 2026-10-26, on channels 0, 1 and 2, took these arguments instead:

```bash
ARGS=(--mux "$BOARD" --http 8790 --floor 1.0 --names "ID 1" "ID 2" "ID 3"
      --calib $C/20261026-134217-id_1-calib.json
              docs/experiments/20261022-ur5_tip_calibration/data/20261025-152920-cb2t/calib_rest.json
              $C/20261025-112147-calib.json
      --correction docs/experiments/20261022-ur5_tip_calibration/data/20261023-203938-cb1t
                   docs/experiments/20261022-ur5_tip_calibration/data/20261025-152920-cb2t -
      --board-gp docs/experiments/20261022-ur5_tip_calibration/data/20261026-board_gp_id1_id2.npz)
```

## Reinforcement learning (simulation only)

The PPO policies add a learned correction to a scripted lift and turn, and none has run on the hand. They are
trained by `scripts/rl/rl_train_cube.py`. `configs/recipes/*.yaml` pin the known-good configurations, and the launchers
select them:

```bash
scripts/rl/train_A_on_morph.sh <morphology-run-dir>          # A (lift), always from scratch per design
scripts/rl/train_handoff_liveA_reset.sh <...>                # B (turn), through the live-A reset
```

Judge a policy on its deterministic held cosine and the trajectory-health scorecard, never on reward sums, and look
at it before explaining its numbers:

```bash
uv run --extra rl --extra gpu python scripts/rl/rl_demo_handoff_continuous.py --run <run>   # the deploy evaluation
uv run --extra rl --extra gpu python scripts/rl/policy_filmstrip.py --run <run>             # frames to read
uv run --extra rl --extra gpu python scripts/rl/policy_eval_suite.py --run <run>            # a distribution, not one rollout
```

The skills `policy-eyes`, `policy-metrics`, `mujoco-eyes` and `morphology-scenes` under `.claude/skills/` wrap these
steps. `results/rl/REGISTRY.md` and `docs/rl/reorientation.md` give the policy lineage.

## Naming

Three ID schemes, none interchangeable.

`D1` through `D8` are the eight hands evaluated on hardware, numbered by their simulated rank so a
reader sees the predicted order on a plot axis. Their internal design tags (`sv1_w6689`, `g12`,
`rv05_manual` and so on) are mapped in
[`scripts/hardware/real_v1_transfer_ranking.py`](scripts/hardware/real_v1_transfer_ranking.py); their gantry
coordinates are in the paper appendix.

`aNN` and `bNN` are the simulation-only policy registry, lift/deliver and reorient.
[`scripts/rl/rename_results_bids.sh`](scripts/rl/rename_results_bids.sh) is the single source of truth
and regenerates `results/rl/REGISTRY.md`.

Design tags show their origin: `sv1_*` is a Sobol draw, `rv*` and `g*` are earlier hand-picked and
gated designs, `m05` and `perp` are the retired simulation-only lineages.

Fingertips are named by version and print: `v3 ID 1` is the first printed four-magnet v3 tip, and `v4b` is the tip
of the CAD hand in simulation.

Every generated folder is date-prefixed, as in `20260901-…`. Run logs go to `logs/`, experiment
summaries to `docs/experiments/`, free-form notes to `docs/notes/`.

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --extra dev python -m pytest tests/ -q
```

Plugin autoloading is off because ROS Humble's pytest plugin, which is on this workstation's Python path, fails to
import. The tests cover the code that moves hardware and that has no other check off the bench: the Manta driver
against a fake board (`manta_hand.fake_hardware`), the frame map, the tag tracker and its service. They also cover the
pinned dependencies, and the rotation module against MuJoCo.
