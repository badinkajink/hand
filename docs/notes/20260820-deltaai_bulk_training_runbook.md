# Bulk RL training on NSF ACCESS / NCSA DeltaAI

Written 2026-08-20, adapting the ACT/LeRobot runbook (`docs/nsf_access_runbook.pdf`) to the
MuJoCo-Warp + PPO stack. That runbook trains one imitation policy in 1.5 GPU-hours; the question
here is different — not "can one run fit", but whether the *seed-dominated evaluator* that has
blocked the morphology program for two months becomes affordable.

Short answer: yes, but for the opposite reason to the one expected. A GH200 is measurably *slower*
per run than the workstation (0.66×). What the cluster buys is that one GPU no longer forces every
run to be sequential — which is the actual constraint, and why every design comparison so far has
been made at n=1 against a per-draw sd of 0.3–0.5.

## What the account gives you

| | |
|---|---|
| Exchange rate | 1,000 ACCESS credits → **7 DeltaAI GPU-hours** (~143 credits/GPU-h) |
| Charge unit | 1 GPU-hour = one GH200 for one hour; a node is 4× GH200 |
| Explore tier cap | 400,000 credits → **~2,800 GPU-hours** |
| Billing | granularity **unresolved** — see below; keep `--time` tight either way |

**What is actually spendable today (checked 2026-08-20 via `ssh deltaai accounts`):**

```
Account          Balance(Hours)   Deposited(Hours)  Project
bgcd-dtai-gh                 92                100  skin sensing for wh...
```

**92 GPU-hours, not 1,300.** Credits and GPU-hours are different things: the ~186k ACCESS credits
on the project dashboard are unexchanged, and only 100 hours have ever been converted into a
DeltaAI balance. Everything below is sized in GPU-hours, so read it against 92 until more credits
are exchanged at `allocations.access-ci.org → Credits + Resources → Exchange`, at 143 credits per
GPU-hour. The n=48 program needs ~120k credits exchanged; check what is still on the project first,
since the dashboard figure comes from a deck written before this account was shared.

**Billing granularity is unresolved, and it decides how to shape an array.** Two jobs were run on
2026-08-20: `2989123` (45 min reserved, 6:37 elapsed) and `2989164` (50 min reserved, 4:19
elapsed). The balance went 92 → 91 after the first and stayed at 91 after the second. That is
consistent with per-job hour granularity *and* with elapsed-second billing plus a delayed,
floor-rounded display — 11 minutes of elapsed GPU time cannot produce a 1-hour drop, but neither
should a 45-minute reservation leave the second job free. Re-read `accounts` a day later, or after
one long job, before committing to a shape. It matters: if billing is hour-granular, 48 × 40-minute
array tasks cost 48 hours regardless of elapsed, and the right structure is several seeds packed
sequentially into each task rather than one seed per task.

## Throughput: the GH200 is SLOWER than the workstation

Measured 2026-08-20 on the real hardware (jobs 2989123, 2989164), against the local runs of
2026-08-19. Both at `num_steps_per_env=24`, so one PPO iteration is `num_envs × 24` steps:

| GPU | num_envs | s/iteration | steps/s |
|---|---|---|---|
| RTX 4070 Ti SUPER | 3072 | 7.00 | **10,530** |
| GH200 120GB | 3072 | 10.3–11.4 | **~6,950** |
| GH200 120GB | 8192 | 27.8 | 7,047 |
| GH200 120GB | 16384 | 55.7 | 7,054 |

**A GH200 delivers 0.66× the throughput of the 4070 Ti SUPER on this workload**, and the estimate
from FLOPs and bandwidth (which said 1–2.5× *faster*) was wrong in direction. Do not size anything
from spec sheets — mjwarp does not behave like a dense-matmul workload.

The `num_envs` sweep was run to test whether CPU-side kernel-launch overhead on the slower Grace
ARM cores explained the deficit, since 95 GiB of HBM makes large batches free. It does not:
iteration time is **exactly linear in `num_envs` from 3072 to 16384** (a 5.3× batch costs 5.3× the
time, ±0.1%), so the GPU is fully saturated at 3072 and the deficit is real compute, not overhead.
That also means the local finding — throughput flat in `num_envs` — holds here, and there is no
free lunch in bigger batches. 3072 remains the right setting.

**The cluster's entire value is concurrency, not speed.** Each run costs ~1.5× more GPU-time than
it does here, and against that you can hold 32+ of them at once. Net effective throughput at 32-way
concurrency is ~21× the local serial rate.

## Feasibility, in our units

At the measured 6,950 steps/s:

| Run type | Timesteps | GPU-hours (GH200) | (local, for reference) |
|---|---|---|---|
| Policy A, lift/deliver (`train_A_on_morph.sh`) | 30M | **1.20** | 0.80 |
| Policy B, reorient (`train_handoff_liveA_reset.sh`) | 20M | **0.80** | 0.53 |
| perp single-phase (`perp_single`) | 25M | **1.00** | 0.67 |

So the **92 hours on the account today are ~92 perp runs**. A fully exchanged ~1,300-hour
allocation would be ~1,300, and the Explore cap ~2,800.

What that does and does not cover:

| | GPU-hours | fits in 92? |
|---|---|---|
| smoke test (2M steps + cold nvcc) | ~0.2 elapsed | yes (done) |
| one design, n=8 seeds | 6.4 | yes |
| 5-design queue, n=4 with A best-of-3 | 34 | yes |
| 5-design queue, n=8 | 50 | yes, but over half the balance |
| 20 designs at n=48 (Δ=0.2 resolution) | **~830** | no — needs ~120k credits exchanged |

Three concrete things the full allocation buys:

**1. The morph sweep at honest n.** A 5-design queue at n=4 seeds is 5 × (3 A-attempts × 1.20 +
4 B × 0.80) ≈ **34 GPU-hours** — a day and a half of solid local training, delivered in ~2 hours of
wall clock here at 20-way concurrency. This is the cheap case and it is already the thing we have
been rationing.

**2. Statistics that can actually separate two designs.** With σ ≈ 0.35 on held-cos, separating two
designs whose true means differ by Δ needs about `n = 15.7 σ² / Δ²` seeds each:

| Δ (difference in mean held-cos) | seeds per design | GPU-hours per design |
|---|---|---|
| 0.5 | 8 | 10 |
| 0.3 | 21 | 20 |
| 0.2 | 48 | 42 |

n=4 resolves only Δ ≳ 0.7. That is why H06_04 (mean 0.748 against a ~0.3 field) replicated and
G02_00 did not — we could only ever see the largest effects. Twenty designs at n=48 is **~830
GPU-hours**, most of a fully exchanged allocation. Locally it is 35 days and would not be attempted.

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
- **Per-run speed gets 1.5× WORSE**, as measured above. Nothing about moving to the cluster makes
  an individual experiment finish sooner; a single run you are waiting on is better off here, on
  the workstation. Send breadth to the cluster, keep latency-sensitive one-offs local.

## Porting risk: all clear on arch, three real traps

The ACT runbook's central pain was aarch64 — pip silently installing CPU-only wheels on ARM. That
risk is now closed by observation, not by reading wheel tags. `uv sync --extra gpu --extra rl`
resolved clean on the login node and every import came up on the GH200 (job 2989123):

| | resolved on DeltaAI | note |
|---|---|---|
| python | 3.12.11 aarch64 | |
| `torch` | 2.11.0+cu128 | driver is 595.71.05, well past the ≥570 cu128 needs |
| `warp-lang` | 1.12.1 | initialized on `sm_90`, mempool enabled, 95 GiB visible |
| `mujoco` | 3.6.0 | |
| `mujoco_warp` | 3.6.0 | editable path dep, imports fine |
| `mjlab` | 1.2.0 | pure python (exposes no `__version__`) |

The cu126 fallback documented in `deltaai_env_setup.sh` therefore stays unused, but leave it there
— it is the fix if a future node images an older driver.

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

- Login, push, env build, smoke and the num_envs sweep have all RUN (2026-08-20); the numbers above
  are measurements, not projections. What has not run is a real multi-task array.
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
  an array of 48. At n=1 a silent flag omission costs 40 minutes; at n=48 it costs 42 GPU-hours —
  nearly half the current balance — and produces a confident, wrong mean.
