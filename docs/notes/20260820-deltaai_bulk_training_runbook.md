# Bulk RL training on NSF ACCESS / NCSA DeltaAI

Written 2026-08-20, adapting the ACT/LeRobot runbook (`docs/nsf_access_runbook.pdf`) to the
MuJoCo-Warp + PPO stack. That runbook trains one imitation policy in 1.5 GPU-hours; the question
here is different — not "can one run fit", but whether the *seed-dominated evaluator* that has
blocked the morphology program for two months becomes affordable.

Short answer: yes, and by a wide margin. The binding constraint has never been the cost of a run.
It is that one GPU forces every run to be sequential, which is why every design comparison so far
has been made at n=1 against a per-draw sd of 0.3–0.5.

## What the account gives you

| | |
|---|---|
| Exchange rate | 1,000 ACCESS credits → **7 DeltaAI GPU-hours** (~143 credits/GPU-h) |
| Charge unit | 1 GPU-hour = one GH200 for one hour; a node is 4× GH200 |
| Explore tier cap | 400,000 credits → **~2,800 GPU-hours** |
| Billing | on **reserved** time, so `--time` must be tight, not generous |

**What is actually spendable today (checked 2026-08-20 via `ssh deltaai accounts`):**

```
Account          Balance(Hours)   Deposited(Hours)  Project
bgcd-dtai-gh                 92                100  skin sensing for wh...
```

**92 GPU-hours, not 1,300.** Credits and GPU-hours are different things: the ~186k ACCESS credits
on the project dashboard are unexchanged, and only 100 hours have ever been converted into a
DeltaAI balance. Everything below is sized in GPU-hours, so read it against 92 until more credits
are exchanged at `allocations.access-ci.org → Credits + Resources → Exchange`, at 143 credits per
GPU-hour. The n=48 program needs ~80k credits exchanged; check what is still on the project first,
since the dashboard figure comes from a deck written before this account was shared.

## Feasibility, in our units

Measured locally on the RTX 4070 Ti SUPER, from the runs of 2026-08-19: **25M timesteps in ~40
minutes ≈ 10,400 steps/s**, and throughput is flat in `num_envs` because the GPU is already
compute-saturated. That gives a per-run price:

| Run type | Timesteps | Wall clock (local) | GPU-hours |
|---|---|---|---|
| Policy A, lift/deliver (`train_A_on_morph.sh`) | 30M | ~48 min | 0.80 |
| Policy B, reorient (`train_handoff_liveA_reset.sh`) | 20M | ~32 min | 0.53 |
| perp single-phase (`perp_single`) | 25M | ~40 min | 0.67 |

So the **92 hours on the account today are ~130 runs** — three days of uninterrupted local training,
delivered in an afternoon. A fully exchanged ~1,300-hour allocation would be ~2,000 runs (54 days
local), and the Explore cap ~4,000.

That split matters for what to do first. 92 hours is enough to prove the pipeline and run one real
seed study, not enough for the design program:

| | GPU-hours | fits in 92? |
|---|---|---|
| smoke test (2M steps + cold nvcc) | ~1 | yes |
| one design, n=8 seeds | 5.4 | yes |
| 5-design queue, n=8 | 27 | yes |
| 5-design queue, n=4 with A best-of-3 | 23 | yes |
| 20 designs at n=48 (Δ=0.2 resolution) | ~560 | **no — needs ~80k credits exchanged** |

Three concrete things the full allocation buys:

**1. The morph sweep at honest n.** A 5-design queue at n=4 seeds is 5 × (3 A-attempts × 0.80 +
4 B × 0.53) ≈ **23 GPU-hours** — about a day of solid local training, or ~1.5 hours of wall clock
here at 20-way concurrency. This is the cheap case and it is already the thing we have been
rationing.

**2. Statistics that can actually separate two designs.** With σ ≈ 0.35 on held-cos, separating two
designs whose true means differ by Δ needs about `n = 15.7 σ² / Δ²` seeds each:

| Δ (difference in mean held-cos) | seeds per design | GPU-hours per design |
|---|---|---|
| 0.5 | 8 | 6.6 |
| 0.3 | 21 | 13.5 |
| 0.2 | 48 | 28 |

n=4 resolves only Δ ≳ 0.7. That is why H06_04 (mean 0.748 against a ~0.3 field) replicated and
G02_00 did not — we could only ever see the largest effects. Twenty designs at n=48 is **~560
GPU-hours**, or 40% of one allocation. Locally it is 23 days and would not be attempted.

**3. Seeds as a first-class axis instead of a budget concession.** `--seed` is already a trainer
flag; the array job below turns it into the array index.

### What does not get faster

- **The A→B pipeline stays sequential per design.** B warmstarts from A and the live-A reset needs
  A's checkpoint. Parallelism is *across* designs and seeds, not within a design. Two array jobs
  chained with `--dependency=afterok:<jobid>` is the right shape; `morph_pipeline_sweep.py` is a
  serial driver and should not simply be lifted onto a compute node.
- **Anything you have to look at.** Filmstrips, eval videos and the health scorecard still gate
  every conclusion (CLAUDE.md lesson 2), and that is bench work here, not cluster work. Bulk
  training raises, not lowers, the risk of believing a reward table — 48 seeds produce 48 reward
  curves and zero pictures.
- **Per-run speed is unmeasured.** A GH200 is H100-class: ~67 TFLOP fp32 against the 4070 Ti
  SUPER's ~44, with 4 TB/s of HBM3 against 672 GB/s of GDDR6X. MuJoCo-Warp's contact solve is
  neither pure ALU nor pure bandwidth, so the honest estimate is **1–2.5×** per run and the real
  number comes out of `deltaai_smoke.slurm`. Every GPU-hour figure above uses the local rate, so
  they are upper bounds on cost.

## Porting risk: all clear on arch, three real traps

The ACT runbook's central pain was aarch64 — pip silently installing CPU-only wheels on ARM. Every
dependency we pin has a real aarch64 build (checked against PyPI, 2026-08-20):

| Package | aarch64 wheel |
|---|---|
| `mujoco==3.6.0` | `manylinux_2_27_aarch64` ✓ |
| `warp-lang` 1.12.0 / 1.12.1 | `manylinux_2_34_aarch64` ✓ |
| `torch` +cu128 (2.7–2.9.1) | `manylinux_2_28_aarch64` ✓ (cu126 also, if the driver is older) |
| `jaxlib` | `manylinux2014_aarch64` ✓ |
| `mjlab==1.2.0`, `mujoco-mjx==3.6.0` | pure python ✓ |

The traps that are ours rather than theirs:

1. **Warp JIT-compiles CUDA kernels on first import**, costing minutes, and every process needs its
   *own* cache — a shared one races and NaNs (CLAUDE.md). Thirty-two array tasks compiling onto
   Lustre simultaneously is the worst possible version of both facts. Fix: compile once in the
   smoke job, tar the cache to `$HOME`, and have each task explode it into node-local NVMe
   (`$SLURM_TMPDIR`). Implemented in the scripts.
2. **Compute nodes have no outbound network.** `wandb` defaults to on in `rl_train_cube.py` and
   will hang or fail there — the exact analogue of the ACT runbook's `push_to_hub` trap. Always
   `--no-wandb`; pull the tfevents down instead.
3. **The login node has no GPU and the compute nodes have no internet**, so `uv sync` must run on
   the login node and anything touching CUDA must run under `srun`/`sbatch`. On compute nodes use
   `source .venv/bin/activate`, not `uv run`, or uv will try to re-resolve the lockfile offline.

Also: `--no-record-videos` on the cluster. EGL on a GH200 compute node is untested and eval video
is GPU time better spent on steps; render locally from the pulled checkpoint.

## The steps

Step 0 (registration, proposal, PI letter) is the ACCESS web flow in the source runbook and is
unchanged — the difference is that the Correll support letter in that deck is already written in
your name, so if the project it backs is yours, you inherit its credits rather than proposing
again. Check at `allocations.access-ci.org → My Projects`.

Your SSH config entry is already installed (`~/.ssh/config`, host `deltaai`, user `wxie`,
`ControlPersist 12h`). Login is NCSA Kerberos password + Duo, which is interactive — I could not do
that part for you. Confirmed reachable: `dtai-login.delta.ncsa.illinois.edu:22` answers and demands
Kerberos+Duo.

```bash
ssh deltaai exit                                  # one Duo push, then 12 h of free reuse
ssh deltaai accounts                              # prints your Slurm account: <proj>-dtai-gh
```

Put that account into the two `#SBATCH --account=CHANGEME-dtai-gh` lines, then:

```bash
# 1. push code (~134 MB; excludes docs/uhas, GraspGenX, lightning-grasp, venvs, videos)
scripts/cluster/deltaai_push.sh results/phase1/perp_thumb_engage/sp25_manual

# 2. build the env — LOGIN node, has internet, no GPU
ssh deltaai 'cd hand && bash scripts/cluster/deltaai_env_setup.sh'

# 3. smoke test — COMPUTE node. Primes the Warp cache and prints real steps/s.
ssh deltaai 'cd hand && sbatch scripts/cluster/deltaai_smoke.slurm'
ssh deltaai 'tail -f hand/logs/deltaai_smoke_*.log'

# 4. bulk: one array task per (design, seed)
ssh deltaai 'cd hand && scripts/cluster/make_seed_manifest.py \
    --designs results/phase1/perp_thumb_engage/sp25_manual --seeds 8 --extra "--recipe perp_single" > manifest.tsv'
ssh deltaai 'cd hand && sbatch --array=1-8%32 scripts/cluster/deltaai_seed_array.slurm manifest.tsv'

# 5. pull checkpoints back, then LOOK at them
scripts/cluster/deltaai_pull.sh 'sp25_s*'
uv run python scripts/policy_filmstrip.py --run results/rl/<run> --width 960 --height 720
```

`%32` caps concurrency at 32 GPUs. Since billing is on reserved time, keep `--time` near the real
run length (2 h for a 40-minute job), not at the 10 h the ACT runbook used.

Scripts live in `scripts/cluster/`:

| File | Runs on | Does |
|---|---|---|
| `deltaai_push.sh` | laptop | rsync code + named morphology/checkpoint dirs up |
| `deltaai_env_setup.sh` | login node | uv sync, arch check |
| `deltaai_smoke.slurm` | compute node | GPU/torch/warp check, prime + bank Warp cache, time a 2M-step run |
| `make_seed_manifest.py` | either | write the (design, seed) manifest |
| `deltaai_seed_array.slurm` | compute node | one training run per array index, resumable via `.DONE` |
| `deltaai_pull.sh` | laptop | rsync checkpoints/configs/logs down (no videos) |

## Open items before this is load-bearing

- Neither the login nor the smoke path has been executed — everything above is verified against
  wheel indexes, the local throughput measurements and the source runbook, not against a live
  session on DeltaAI. The Duo login has to happen first.
- `--account` is a placeholder in both `.slurm` files until `accounts` is run.
- Whether the `ghx4` partition allows per-GPU (non node-exclusive) jobs decides whether a 1-GPU
  array task bills 1 GPU-hour or 4. The scripts request 1 GPU per task; confirm against
  `docs.ncsa.illinois.edu/systems/deltaai` before queueing anything large.
- `deltaai_smoke.slurm` times the `sp25_manual` perp design because that is the last thing trained
  here; set `MORPH_RUN=` to whatever you actually want the throughput number for.
- **Flag parity is the failure mode most likely to waste an allocation.** Recipes carry reward
  shaping and nothing else — `perp_single` does not set the timestep budget, `num_envs`, or the
  lift height, and a perp run at the default lift instead of 0.14 trains clean, logs nothing
  unusual and aligns 0.0%. That is what voided the r5 queue. Both `.slurm` files now pass those
  explicitly, and run-specific flags go in the manifest's `EXTRA` column, but the discipline still
  applies: diff one task's dumped `config.yaml` against the last known-good run *before* releasing
  an array of 48. At n=1 a silent flag omission costs 40 minutes; at n=48 it costs 28 GPU-hours and
  produces a confident, wrong mean.
