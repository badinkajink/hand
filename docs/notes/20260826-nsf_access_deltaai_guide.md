# Training on NSF ACCESS / NCSA DeltaAI — a guide written from a real allocation

Standalone. Nothing here depends on the repo it ships in; copy it wherever it is useful.

Everything marked **measured** was observed on DeltaAI's `ghx4` partition between 2026-08-20 and
2026-08-21, running a MuJoCo-Warp + PPO reinforcement-learning stack (Python 3.12, torch 2.11+cu128,
`warp-lang` 1.12, aarch64). Everything marked **inferred** is a reading of the documentation
corroborated by a balance drop, and is worth re-checking before you size a large job. The
performance findings generalise to any GPU workload; the exact numbers are ours.

The short version, if you read nothing else:

1. **Credits are not GPU-hours.** A dashboard showing 186,000 ACCESS credits can correspond to a
   spendable balance of 92 hours. Run `accounts` on the login node before planning anything.
2. **A GH200 was 0.66× the throughput of a desktop RTX 4070 Ti SUPER on our workload** — measured,
   against a FLOPs/bandwidth estimate that predicted 1–2.5× *faster*. Do not size cluster work from
   spec sheets. Measure with a smoke job before you queue anything in bulk.
3. **The cluster buys concurrency, not speed.** If per-run time gets worse and you can hold 32 runs
   at once, send *breadth* there (seeds, sweeps, ablations) and keep any single run you are actually
   waiting on wherever it runs fastest.
4. **Billing is on what a job reserves, rounded up.** One short run per job wastes the tail of every
   reservation. Reserve long blocks and keep them full.
5. **Login nodes have internet and no GPU; compute nodes have a GPU and no internet.** Every
   defaulted-on network call in your training script (experiment trackers, model-hub pushes, dataset
   downloads) will hang or fail on the compute node.

---

## 1. What the account actually gives you

| | |
|---|---|
| Hardware | GH200 (Grace-Hopper), **aarch64**, 96–120 GB HBM; a node is 4× GH200 |
| Partition | `ghx4` |
| Charge unit | 1 SU = one GH200 for one hour |
| Exchange rate | 1,000 ACCESS credits → **7 DeltaAI GPU-hours** (~143 credits/GPU-h) |
| Explore tier cap | 400,000 credits → ~2,800 GPU-hours |
| Login | NCSA Kerberos password + Duo (interactive; cannot be automated) |

**The credits-versus-hours trap is the most common way to plan a program that cannot be paid for.**
ACCESS credits sit on the *project*; DeltaAI GPU-hours are what you get after exchanging some of
them at `allocations.access-ci.org → Credits + Resources → Exchange`. Our project's deck implied
~1,300 hours. The real spendable balance was:

```
$ ssh <cluster> accounts
Account          Balance(Hours)   Deposited(Hours)  Project
<proj>-dtai-gh               92                100  ...
```

**92 hours** — because only 100 had ever been exchanged. Check `accounts` first, size the program in
GPU-hours, and exchange credits *before* you need them, not when a queue is ready to go.

---

## 2. Billing, and why it decides your job shape

**An SU is one GH200 for one hour, billed on what the job RESERVES rather than what it uses, and
rounded up** (inferred from NCSA's docs, corroborated: 13.9 hours of packed work drew 15 SU off the
balance). Three consequences, in descending order of how much money they cost:

**A short job wastes the rest of its reservation.** A 40-minute run submitted as its own job burns a
whole hour. Forty-eight of them throw away 16 hours — 17% of a 92-hour balance — for nothing. This
is the argument for packed queues (§6) over one-task-per-job arrays.

**Memory and CPU can silently multiply the GPU charge.** Charging is by the number of GH200s needed
to satisfy memory *or* cores, whichever is larger. On a 4-GPU node with ~110 GB and 72 cores per
GPU, asking for `--mem=220G` bills **2 SU per hour for the same one GPU**. Keep `--mem` and
`--cpus-per-task` inside one GPU's share unless you genuinely want more GPUs.

**Balance readings lag.** Two early jobs (6:37 and 4:19 elapsed, 45 and 50 minutes reserved) moved
the balance 92 → 91 → 91. That is consistent with several billing models and with a delayed display.
Do not infer the rule from one short job; run a long one, wait a day, then read `accounts` again.

---

## 3. Day-one setup

**SSH with a persistent authenticated socket.** Duo cannot be scripted, so authenticate once and let
every subsequent `rsync`/`sbatch`/`squeue` ride the same connection:

```sshconfig
# ~/.ssh/config
Host mycluster
    HostName <login-host>
    User <username>
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h-%p
    ControlPersist 12h
    ServerAliveInterval 60
```

```bash
ssh mycluster exit          # one Duo push, then 12 h of free reuse
ssh mycluster accounts      # your Slurm account, and your real balance
```

**Push code selectively.** Repositories are usually far larger than what training needs. Ours was
28 GB on disk; 134 MB and ~2,100 files actually had to go up. Exclude submodules you do not import,
rendered media, virtualenvs, and result directories — then name the specific result/checkpoint paths
a job reads as explicit extra arguments.

```bash
rsync -avz --exclude '.git/' --exclude '.venv*/' --exclude 'wandb/' --exclude 'logs/' \
      --exclude 'results/' --exclude '__pycache__/' --exclude '*.mp4' --exclude '*.pdf' \
      ./ mycluster:proj/
rsync -avz --relative ./results/<the-one-checkpoint-a-job-needs> mycluster:proj/
```

**Build the environment on the login node** (it has internet; it has no GPU, so nothing that touches
CUDA can be validated here). We let `uv` own the whole environment rather than layering a venv on a
site `torch` module — site modules will not match the versions a simulation stack pins.

```bash
ssh mycluster 'cd proj && module purge && module load cuda && uv sync --extra gpu --extra rl'
```

**Then a smoke job on a compute node** before anything else — §5.

---

## 4. The topology traps

These caused every early failure and they are structural, not incidental:

**Login node: internet, no GPU. Compute node: GPU, no internet.** So every download happens on the
login node and everything that touches CUDA happens under `srun`/`sbatch`. Corollaries:

- On compute nodes use `source .venv/bin/activate`, **not `uv run`** — `uv` will try to re-resolve
  the lockfile against an unreachable index and fail.
- **Any defaulted-on network call in your trainer will hang or die.** For us that was Weights &
  Biases (`--no-wandb` on every cluster invocation); the equivalents are `push_to_hub`, dataset
  downloads, `timm`/`transformers` weight fetches, and telemetry. Grep your entry point for anything
  that phones home and put an explicit off-switch on the command line.
- **Rendering is a compute-node risk, not a convenience.** EGL/OpenGL on an unfamiliar node image is
  untested surface, and video encoding is GPU time you are paying for. Train headless, pull the
  checkpoint, render on a machine with a display stack.

---

## 5. Throughput: measure it, do not project it

**This is the finding most worth carrying to another cluster.** Our estimate from FLOPs and memory
bandwidth said a GH200 would be 1–2.5× faster than the local desktop card. Measured:

| GPU | batch (envs) | s/iteration | steps/s |
|---|---|---|---|
| RTX 4070 Ti SUPER (desktop) | 3072 | 7.00 | **10,530** |
| GH200 120 GB | 3072 | 10.3–11.4 | **~6,950** |
| GH200 120 GB | 8192 | 27.8 | 7,047 |
| GH200 120 GB | 16384 | 55.7 | 7,054 |

**0.66× — wrong in direction, not just magnitude.** Simulation workloads that launch many small
kernels do not behave like dense matmul, and datacenter parts optimised for the latter do not
automatically win.

We then tested the obvious rescue — that slower Grace ARM cores were bottlenecking kernel launches,
fixable by amortising over larger batches, which 95 GiB of HBM makes free. **It is not the
mechanism:** iteration time is *exactly* linear in batch size from 3072 → 16384 (5.3× the batch for
5.3× the time, ±0.1%). The GPU is saturated at the small batch and bigger batches buy nothing. Run
this sweep yourself; it is one job and it either finds you free throughput or closes the question.

**So write your own smoke job before spending anything.** It should: assert the GPU is visible from
inside your env, print device/driver, run a short but *real* training job on a *real* input, and
print steps/s (or samples/s) next to your reference machine's number. Every hour estimate in your
plan depends on that one number.

**What this means strategically.** Per-run latency got 1.5× *worse*. What improved is that runs no
longer queue behind one another: at 32-way concurrency the effective throughput is ~21× the local
serial rate. Send breadth to the cluster — seeds, sweeps, ablations, anything where you want *n*
independent draws. Keep the single run you are impatiently watching on whatever hardware is fastest.
And note what does not parallelise: sequential pipeline stages (stage B warmstarting from stage A's
checkpoint) stay sequential per item; chain them with `--dependency=afterok:<jobid>` rather than
lifting a serial driver script onto a compute node.

---

## 6. Job shape: a packed work queue beats a job array

The natural shape is a Slurm array, one task per run:

```bash
sbatch --array=1-48%32 train_array.slurm manifest.tsv
```

It is fine, and it is what we wrote first. Under reservation billing it wastes the tail of 48
reservations. The better shape reserves a **long block and keeps it full**: a worker drains a shared
manifest, running tasks back to back, and stops starting new ones only when the remaining wall time
cannot fit the next one. Nothing is wasted but a single tail.

Three properties make this worth the fifty lines:

- **Self-balancing concurrency.** Tasks are claimed atomically with `mkdir` (the one POSIX primitive
  that is atomic on a shared filesystem). Submit *K* workers against the *same* manifest and get
  *K*-way concurrency with no index arithmetic, and no worker stranded behind a slow task.
- **Free resumption.** State lives in the queue directory, not in Slurm. If a block expires
  mid-queue, the next submission picks up exactly where it stopped — and "where did we leave off" is
  answerable without the scheduler.
- **Heterogeneous work.** Manifest lines are `EST_HOURS <TAB> LABEL <TAB> COMMAND`, run verbatim
  through `bash`, so a training run and an eval sweep pack into the same block without the runner
  knowing anything about their flags.

The runner core, portable as-is:

```bash
#!/usr/bin/env bash
#SBATCH --partition=ghx4
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8          # keep inside ONE GPU's share (see §2)
#SBATCH --mem=64G                  # ditto — 220G silently bills 2 SU/h
#SBATCH --time=08:00:00
#SBATCH --output=logs/pack_%j.log
set -uo pipefail                   # NOT -e: one failed task must not kill the block

QNAME="${1:?usage: sbatch pack.slurm <queue-name>}"
cd "$SLURM_SUBMIT_DIR"; module purge; module load cuda; source .venv/bin/activate
QDIR="queue/$QNAME"; mkdir -p "$QDIR"/{claimed,done,failed,logs}

# When does this reservation end? SLURM_JOB_END_TIME on recent Slurm; scontrol
# otherwise; start+TimeLimit if even that fails.
job_end_epoch() {
  [ -n "${SLURM_JOB_END_TIME:-}" ] && { echo "$SLURM_JOB_END_TIME"; return; }
  local et
  et=$(scontrol show job -o "$SLURM_JOB_ID" 2>/dev/null | tr ' ' '\n' \
       | grep '^EndTime=' | cut -d= -f2)
  if [ -n "$et" ]; then date -d "$et" +%s; else echo $(( $(date +%s) + 8*3600 )); fi
}
END_EPOCH=$(job_end_epoch)
MARGIN_S=${MARGIN_S:-300}          # so the tail task is not SIGKILLed mid-checkpoint
TOTAL=$(wc -l < "$QDIR/manifest.tsv")

while :; do
  IDX=""
  for i in $(seq 1 "$TOTAL"); do
    [ -e "$QDIR/done/$i" ] && continue
    [ -e "$QDIR/failed/$i" ] && continue
    mkdir "$QDIR/claimed/$i" 2>/dev/null && { IDX="$i"; break; }   # atomic claim
  done
  [ -n "$IDX" ] || { echo "queue drained"; break; }

  IFS=$'\t' read -r EST LABEL CMD <<<"$(sed -n "${IDX}p" "$QDIR/manifest.tsv")"
  EST_S=$(awk -v e="${EST:-1.0}" 'BEGIN{printf "%d", e*3600}')
  REMAIN=$(( END_EPOCH - $(date +%s) - MARGIN_S ))
  if [ "$REMAIN" -lt "$EST_S" ]; then
    rmdir "$QDIR/claimed/$IDX"     # release for the next block; queue stays intact
    echo "${REMAIN}s left, next task needs ${EST_S}s — stopping cleanly"; break
  fi

  echo "======== [$IDX/$TOTAL] $LABEL (est ${EST}h)"
  bash -c "$CMD" > "$QDIR/logs/${IDX}_${LABEL}.log" 2>&1 && date -Is > "$QDIR/done/$IDX" \
    || date -Is > "$QDIR/failed/$IDX"
  rmdir "$QDIR/claimed/$IDX" 2>/dev/null
done
```

A small companion CLI (`new` / `status` / `plan` / `retry`) pays for itself. The only subtle part is
`plan`: **round workers DOWN, in whole tasks.** A block fits `floor(usable_hours / longest_task)`
tasks, and reserving a worker that can only be half-filled is pure waste under reservation billing.

---

## 7. Prime your JIT / kernel cache once, then hand it out

Any stack that compiles at import — Warp, `torch.compile`, Triton, JAX/XLA, TensorRT — pays minutes
of `nvcc` on first run. Two facts collide badly at scale:

- **Every process needs its own cache.** A shared one races; in our case it produced NaNs, not an
  error.
- **A parallel filesystem is the wrong place for it.** Thirty-two tasks compiling onto Lustre
  simultaneously is a self-inflicted outage.

So: compile once in the smoke job, tar the cache to `$HOME`, and have every later task explode it
into **node-local NVMe**:

```bash
export WARP_CACHE_PATH="${SLURM_TMPDIR:-/tmp}/cache_$SLURM_JOB_ID"   # or TORCHINDUCTOR_CACHE_DIR,
mkdir -p "$WARP_CACHE_PATH"                                          # TRITON_CACHE_DIR, XLA cache…
[ -f "$HOME/kernel_cache.tar" ] && tar -C "$WARP_CACHE_PATH" -xf "$HOME/kernel_cache.tar"
trap 'rm -rf "$WARP_CACHE_PATH"' EXIT
# … at the end of the smoke job only:
tar -C "$WARP_CACHE_PATH" -cf "$HOME/kernel_cache.tar" .
```

---

## 8. aarch64: a real risk, closed by observation

Grace-Hopper nodes are ARM. The classic failure is `pip` silently resolving a CPU-only or
x86-flavoured wheel and the job "working" at a fraction of the speed, or importing and never seeing
the GPU. Resolve the environment, then **print versions from inside it on a compute node** rather
than reading wheel tags. Ours came up clean:

| package | resolved on DeltaAI |
|---|---|
| python | 3.12.11 aarch64 |
| torch | 2.11.0+cu128 (driver 595.71.05, well past the ≥570 cu128 needs) |
| warp-lang | 1.12.1, initialised on `sm_90`, 95 GiB visible |
| mujoco / mujoco_warp | 3.6.0 |

Keep a documented fallback anyway: cu128 wheels want a ≥570 driver, and the same torch versions ship
aarch64 **cu126** wheels. If a future node images an older driver, flipping the index is the fix.
Pin the CUDA wheel index explicitly in `pyproject.toml` — left to its own devices `uv` will resolve
a `+cu130` build that is too new for the driver:

```toml
[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu128" }
```

---

## 9. The discipline that protects an allocation

Scale multiplies the cost of quiet mistakes. Locally a bad launch costs 40 minutes and you notice;
at n=48 it costs 42 GPU-hours and produces a confident, wrong mean. Four habits, each of which we
learned the expensive way:

**Diff a new launcher's flags against the last known-good run's dumped config — before releasing the
array, not after.** Config systems and recipes typically carry *some* parameters and not others.
Ours carry reward shaping but not the timestep budget, batch size, or a critical task parameter; a
run with the wrong value trains clean, logs nothing unusual, and scores zero on the metric you care
about. Two five-run queues died 0/5 in one day this way. Run one task, dump its resolved config,
diff it, *then* release the rest.

**Know what your parity checker cannot see.** Ours compares dumped configs — and the warmstart
checkpoint path is not written into the dumped config. An in-flight parity check passed cleanly on
16 runs that were all loading the wrong policy. **A launcher's default value is not evidence of what
a published run used;** check provenance against your registry by hand. Better: make the trainer
record every input it loads.

**Make every task resumable and idempotent.** A `.DONE` sentinel per task, checked on entry, turns a
partially-drained queue into a no-op re-submission. Combined with on-disk queue state (§6),
expiring a block costs nothing.

**Bulk training raises, not lowers, the risk of believing a summary table.** Forty-eight seeds
produce forty-eight reward curves and zero pictures. Budget bench time to actually *look* at
outliers — we rendered the one tail draw in our band and it was a dropped object at a specific step,
not the "slightly worse policy" the aggregate number implied. That distinction changed the
conclusion.

---

## 10. A worked example, with the failure included

**The question.** Every downstream conclusion in a sim2real hardware spec rested on a single trained
policy. Its rollout-to-rollout spread was known (±0.1); its *training-draw* spread was not, and the
program-wide prior for that quantity was 0.3–0.5. If the reference policy sat at the top of a wide
band, the hardware spec had been written off a lucky draw. Locally this is 16 sequential runs — a
week of the only GPU — so it had never been asked.

**The cost.** 16 draws × 20M timesteps at the measured 6,950 steps/s ≈ 0.87 h each ≈ 13.9 GPU-hours
of work; **15 SU** off the balance, a few hours of wall clock.

**The result.** Between-draw sd **0.032**, against within-draw (rollout) sd **0.080** — the training
draws are *less* variable than the evaluation noise. The reference policy landed at 0.891 with 11 of
16 draws at or below it: an ordinary draw, which is what the study was run to check. And the
program-wide 0.3–0.5 prior turned out to belong to a *different regime* (from-scratch training of a
hard-exploration objective); warmstarting removes the lottery. That re-prices everything: resolving
a difference of 0.1 in the mean needs ~2 seeds, not 192.

**The failure, which is the more useful half.** The first version of that queue returned 1
completion and 15 NaN divergences. Cause: the warmstart checkpoint was taken from the launcher's
*default*, not from the reference run's provenance — so every task loaded a policy trained on
different geometry, and the physics solve diverged. Two hypotheses were tested and died first (the
aarch64 platform — disproven, the same seeds fail locally; and an initialisation-noise parameter —
disproven by a three-value probe). Substituting the correct checkpoint on the three fastest failures
turned NaN-at-iteration-21/36/22 into clean 40/40/40.

Three transferable lessons: **cross-context warmstarts are a leading cause of silent divergence at
scale**; **suspect your own launch before you suspect the cluster** (the platform hypothesis cost
real time and was wrong); and **record the failure**, because the two dead hypotheses are worth as
much as the finding to the next person.

---

## 11. Pre-flight checklist

Before the first job:

- [ ] `accounts` on the login node — the **balance in hours**, not the credits on the dashboard
- [ ] Exchange credits if the balance will not cover the plan (~143 credits/GPU-h)
- [ ] SSH `ControlPersist` configured; `ssh <host> exit` for one Duo push per day
- [ ] Env built on the **login** node; versions and GPU visibility printed from a **compute** node
- [ ] Smoke job run: real workload, real input, prints steps/s next to your reference machine
- [ ] Kernel/JIT cache primed and banked to `$HOME`
- [ ] Every network call in the trainer explicitly disabled

Before any array or queue larger than a few tasks:

- [ ] One task run end to end; its **dumped config diffed** against the last known-good run
- [ ] Warmstart/checkpoint paths verified against a registry **by hand** — the parity checker cannot
      see them
- [ ] `--mem` and `--cpus-per-task` inside one GPU's share (or you are paying for GPUs you asked for
      by accident)
- [ ] `--time` matched to the real block length; task hour-estimates honest enough that packing works
- [ ] Per-task `.DONE` sentinels so a re-submission resumes rather than repeats
- [ ] Bench time budgeted to *look* at the outliers the aggregate will hide

---

*Provenance: measured on NCSA DeltaAI (`ghx4`, GH200, aarch64), 2026-08-20/21, MuJoCo-Warp + PPO.
The reference implementation of the push/setup/smoke/array/packed-queue scripts described here lives
in `scripts/cluster/` of the MorphoHand repository, with the full study write-up in
`docs/notes/20260820-deltaai_bulk_training_runbook.md`.*
