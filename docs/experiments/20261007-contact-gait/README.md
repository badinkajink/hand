# Contact-gait experiment runbook

The dated result page is `20261007-contact_release_real_v1.html`. Its builder is
`scripts/real_v1_contact_gait_page.py`; its raw JSON inputs are in this
directory. The study is simulation-only. No command connects to the
hand controller.

The user-supplied discussion in
`docs/experiments/20261007-diffmjx/geminichat.pdf` motivated the paired
sampling-then-refinement and contact-release probes. The released DiffMJX
paper supplies the external evidence discussed on the page.

## Scene and score

`scripts/real_v1_diffmjx_gate.py` generates the D1 scene from
`assets/mjcf/real_v1/real_hand.xml` and the deployed D1 grasp plan at
`docs/experiments/20260902-residual-bench/deploy/sv1_w6689_b060_plan.json`.
The plan supplies the first 0.8 s of a supported grasp. The continuation
disables the support post and non-colliding palm display plate. `padsT` keeps
the 1 mm packed pad spheres for tool contact and disables pad-to-pad mesh
contact. The fingertip generator yields 796 spheres per finger at 1 mm and
227 per finger at 2 mm. The comparison geometries are one convex TPU6 mesh
and one sphere per finger. This D1 hardware-derived scene is a local simulation
probe; no collision-cleared hardware trajectory was exported.

The 400 ms plan has 13 values: three yaw targets in each of three phases, two
opening amplitudes until 180 ms, and two closing biases thereafter. The
`--forced-release` variant fixes the index opening amplitude at its maximum.
The search score adds final-minus-initial vertical cosine and subtracts 70 per
metre of tool descent beyond 3 mm and 0.7 per newton of missing final finger
force below 0.5 N. Held status separately requires tool descent of at most
8 mm and all three final fingertip normal forces of at least 0.5 N.

## CPU searches and transfer

The project's `uv run --no-sync python` environment has MuJoCo 3.6.0 and
SciPy 1.17.1. Each method gets at most 256 forward trajectory evaluations:
direct numerical two-point L-BFGS-B from the hold plan, CEM, or 128 CEM
evaluations followed by 128 local evaluations. The direct method has a hard
evaluation cap because SciPy may otherwise exceed `maxfun` while estimating
gradients. The force trace is sampled every 10 ms.

Run each CPU command under the workstation resource guard. The listed outputs
were produced with `RESGUARD_MAX_SWAP_GB=16` while swap used 15.6 GB but memory
pressure was zero; each CPU process stayed below the 1 GB cap. The default
guard threshold subsequently reopened as swap fell to 12 GB.

```bash
RESGUARD_MAX_SWAP_GB=16 ~/.claude/bin/resguard.sh run --mem 1G --cpu 300 -- \
  uv run --no-sync python scripts/real_v1_contact_gait_search.py \
  --tip tpu6 --contact-model padsT --seed 1 --budget 256 \
  --out docs/experiments/20261007-contact-gait/20261007-padsT-seed1-256.json

# Repeat for --seed 2 and 3; repeat seed 1 with --contact-model pt, and with
# --tip sphere --contact-model pt. Repeat those three seed-1 cases with
# --forced-release and the corresponding *-release-seed1-256.json output.

uv run --no-sync python scripts/real_v1_contact_gait_transfer.py \
  --plans docs/experiments/20261007-contact-gait/20261007-padsT-seed1-256.json \
  docs/experiments/20261007-contact-gait/20261007-mesh-seed1-256.json \
  docs/experiments/20261007-contact-gait/20261007-sphere-seed1-256.json \
  docs/experiments/20261007-contact-gait/20261007-padsT-release-seed1-256.json \
  docs/experiments/20261007-contact-gait/20261007-mesh-release-seed1-256.json \
  docs/experiments/20261007-contact-gait/20261007-sphere-release-seed1-256.json \
  --out docs/experiments/20261007-contact-gait/20261007-transfer-mj360.json
```

The transfer command was wrapped in the same 1 GB resource guard. Running it
with `logs/20261007-diffmjx/venv/bin/python` instead of `uv run` generated
`20261007-transfer-mj331.json` under MuJoCo 3.3.1. Every replay uses its own
geometry-specific 0.8 s grasp and the fixed seed-1 control vectors.

The coarse-planning control repeats the packed search with
`--pad-spacing-mm 2`, both with and without `--forced-release`, writing
`20261007-padsT2-seed1-256.json` and
`20261007-padsT2-release-seed1-256.json`. Run the transfer script with those
two files and `--include-2mm` to write
`20261007-transfer-padsT2-to-1mm-mj360.json`. This evaluates fixed 2 mm plans
on 2 mm pads, the exact 1 mm pads, the convex mesh and the single sphere.

## MuJoCo-Warp replay

The Warp script initializes each GPU world from the CPU MuJoCo 3.6 held state,
then runs the 400 ms continuation. It uses the same exact 1 mm packed surface,
controls and 10 ms force samples. The two 32-world runs contain eight grip
jitters and four plans. Use a distinct `WARP_CACHE_PATH` for each process.

```bash
WARP_CACHE_PATH=$(mktemp -d) ~/.claude/bin/resguard.sh run --mem 6G --cpu 500 -- \
  uv run --no-sync python scripts/real_v1_contact_gait_warp.py \
  --plans docs/experiments/20261007-contact-gait/20261007-padsT-seed1-256.json \
  --tip tpu6 --contact-model padsT --seeds 1 2 3 4 5 6 7 8 \
  --methods hold direct_gradient cem hybrid \
  --out docs/experiments/20261007-contact-gait/20261007-warp-padsT-free-8.json

# Repeat with the padsT-release plan JSON and the
# 20261007-warp-padsT-release-8.json output.
```

The paired before/after refinement replays used `--methods hybrid_seed hybrid`
with each packed source file and wrote `20261007-warp-padsT-free-seed-8.json`
and `20261007-warp-padsT-release-seed-8.json`.

## Released DiffMJX fork on GPU

The isolated Python 3.13 environment and source revisions are documented in
`docs/experiments/20261007-diffmjx/README.md`. This continuation installed
`jax[cuda12]==0.11.2` into that ignored environment. All GPU JSON here uses
64-bit JAX and the released fork of MuJoCo 3.3.1. The 20 ms timing script now
records CPU time, first-call compilation and five warm GPU repetitions.

```bash
JAX_PLATFORM_NAME=gpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
  ~/.claude/bin/resguard.sh run --mem 8G --cpu 500 -- \
  logs/20261007-diffmjx/venv/bin/python \
  scripts/real_v1_diffmjx_official_gate.py --steps 20 --tip tpu6 \
  --contact-model padsT --pad-spacing-mm 2 --skip-gradient --warm-repeats 5 \
  --out docs/experiments/20261007-contact-gait/20261007-mjx-gpu-padsT2-20-warm.json

# Repeat with --tip sphere --contact-model pt (4 GB cap), and with
# --tip tpu6 --contact-model pt --tool-capsule (4 GB cap).
```

The exact 1 mm packed attempt requested a 20 ms rollout without a derivative.
Its initial `mjx.forward` compilation completed in 129.106 s. The rollout
compilation had not finished at 475 s, when the process was stopped after
reaching 11.5 GiB host resident memory and 4.5 GiB available RAM. The
attempt is recorded in `20261007-mjx-exact1mm-attempt.json`.

For the native comparison, `logs/20261007-diffmjx/venv-native-gpu` is an
isolated Python 3.13 environment with MuJoCo and `mujoco-mjx` 3.6.0, JAX
0.11.2 and the CUDA 12 JAX plugin. The native script records five warm repeats
for the single-sphere case and three for 2 mm pads. It ran with
`JAX_PLATFORM_NAME=gpu`, `XLA_PYTHON_CLIENT_PREALLOCATE=false` and the same
resource guard. Use `JAX_ENABLE_X64=1` for the 64-bit cases and
`JAX_ENABLE_X64=0` for the 32-bit packed case:

```bash
logs/20261007-diffmjx/venv-native-gpu/bin/python \
  scripts/real_v1_diffmjx_gate.py --steps 20 --no-post --tip tpu6 \
  --contact-model padsT --pad-spacing-mm 2 --warm-repeats 3 \
  --out docs/experiments/20261007-contact-gait/20261007-native-mjx-gpu-padsT2-20-warm.json
```

The single-sphere native test used `--tip sphere --contact-model pt` and five
warm repeats. The 32-bit packed test used the same 2 mm command and wrote
`20261007-native-mjx-gpu-padsT2-f32-20-warm.json`.

## Derivative at recontact

`scripts/real_v1_diffmjx_recontact_jvp.py` replays the seed-1 packed-pad CEM
release plan in CPU MuJoCo 3.3.1 with 2 mm pads. At 280 ms, index force is
zero; at 290 ms it is 0.571 N. The script transfers the 280 ms state to the
released DiffMJX fork, differentiates the next 10 ms with respect to the index
MCP target, and compares AD with fork and CPU finite differences. The plain
run used an 8 GB memory cap and 64-bit JAX on GPU:

```bash
JAX_PLATFORM_NAME=gpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
  ~/.claude/bin/resguard.sh run --mem 8G --cpu 500 -- \
  logs/20261007-diffmjx/venv/bin/python \
  scripts/real_v1_diffmjx_recontact_jvp.py \
  --plans docs/experiments/20261007-contact-gait/20261007-padsT-release-seed1-256.json \
  --method cem --start-ms 280 --steps 10 \
  --out docs/experiments/20261007-contact-gait/20261007-fork-padsT2-recontact-jvp.json
```

The `--cfd --scan-loop` variant writes
`20261007-fork-padsT2-recontact-cfd-jvp.json`.
The plain AD orientation and angular-speed derivatives differed from CPU
finite differences by 2.5% and 2.1%. CFD changed those derivatives while its
forward vertical cosine differed from the plain fork by less than 1e-9.
The plain derivative compiled in 70.3 s and took 9.84 s warm; CFD compiled in
122.8 s and took 17.31 s warm. Both runs stayed within the 8 GB memory cap.

Rebuild the page with `python scripts/real_v1_contact_gait_page.py`.
