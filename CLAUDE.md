# CLAUDE.md — MorphoHand standing guidance

Project context + hard-won lessons for AI assistants working in this repo. Read this first; it
encodes conventions and failure modes that are expensive to relearn. (Point-in-time memories live
in `~/.claude/.../memory/MEMORY.md`; this file is the durable, repo-level guidance.)

## What this project is

Co-design of a **3-finger reconfigurable hand** ("MorphoHand"): optimize the 9-param finger
morphology *and* the grasp/manipulation together, in simulation (MuJoCo-Warp), validated toward
hardware. Two tracks: **grasp/morphology optimization** (CEM grasp synthesis over a parametric hand)
and **RL manipulation** (lift → in-hand reorient of a flat screwdriver to vertical, finger-only).

## Environment & running

- Python via **`uv`**: `uv run --extra rl --extra gpu python <script>`. `MUJOCO_GL=egl` for
  headless render.
- **GPU is single (16 GB).** Train sequentially. **Every Warp process needs its own kernel cache:**
  `WARP_CACHE_PATH=$(mktemp -d)` — a shared cache races and NaNs. After killing a Warp run, wait for
  GPU memory to drop to ~1 GB before relaunching.
- `CUDA unknown error` / `torch.cuda.is_available()==False` after a hard-killed run = a wedged
  `nvidia_uvm`; reload the module or reboot (nvidia-smi still working is a red herring).
- Long jobs: launch **detached** (`nohup setsid …`) so they survive the session, make them
  **resumable** (per-item checkpoint), and watch for completion with a `run_in_background` waiter
  that matches the worker via `pgrep -f "[p]ython3 …"` (bracket-trick avoids self-match; don't rely
  on a captured PID — the launcher forks transient PIDs).

## Documentation — where things go (keep all three in sync for real work)

- **`webpaper/` (Typst → HTML) = the canonical readable doc**, tutorial-style: *foundation →
  experimentation → results → analysis*. Critical narrative in the main flow; comprehensive details
  in **collapsible dropdowns** via `#det(summary, body, kind: "…", open: false)` (there's a
  "full-paper" toggle that opens all). Pillars: `morphology.typ`, `rl.typ`, `hardware.typ`. Build:
  `webpaper/build.sh`. Authoring gotchas: compile needs `--features html`; inline math must be
  **Unicode prose** (bare `$…$` is dropped; only *display* `$ … $` survives); never put raw
  `_ * ^ ` in prose (Typst markup breaks compile); sources must live under `$HOME` (Typst is
  snap-confined). Helpers: `#media`, `#fig`, `#callout`.
- **LaTeX papers** (`paper/main.tex` = simulation/morphology stack; `hand_paper/main.tex` =
  hardware). Split content **main body vs appendix**; since results are still preliminary/negative,
  detailed sweeps + variance characterization go in the **appendix**, methodology + the durable
  findings in the body.
- **`docs/rl/*.md`** = the working engineering log (chronological, append-only; `reorientation.md`
  is the RL source of truth). `docs/rl/morph_sweep_STATUS.md` = live runbook for the morphology
  sweep.
- **Commit experiment code promptly** (it has been lost to `git` before). `results/` is gitignored.

## Critical lessons (the load-bearing ones — violating these has cost days)

0. **HARDWARE-PROVENANCE GATE: only current `real_v1` hands count for new manipulation
   experiments.** Any morphology used to train, evaluate, rank, render, or make a deployment claim
   must be generated from the real hardware model (`assets/mjcf/real_v1/real_hand.xml` and its
   `real_v1` scenes/generators), must be representable inside `REAL_V1_WORKSPACE` /
   `REAL_V1_MOUNTS`, and must be exportable through the real build/plan mechanism. Record the base
   scene, generator, compact design or mount coordinates, and deployment status before launching
   expensive work; if no collision-safe plan has been produced yet, label that explicitly rather
   than implying one exists. Abort if the hardware provenance or workspace fit cannot be
   established. `results/phase1/real_v1/*` and scenes generated from the current `real_v1` base are
   eligible for simulation studies; a hardware-transfer claim additionally requires the export and
   trajectory-clearance gates. Legacy simulation-only families—including
   `results/phase1/landscape/*`, `m05_ik_cem`, `perp`, and their old a10/b33 policies—are historical
   controls only: do not use them for a new policy study or generalize their result to hardware
   unless the user explicitly requests a non-deployable control. Observation compatibility does not
   make the hand geometry deployable.
1. **Monitor IN-TRAINING metrics, not just final reward.** Aggregate reward/tip-lost/held-object
   *hide* degeneracy (late/idle finger, 2-finger pinch, jitter, de-centering, over-clamp). Use the
   **trajectory-health scorecard** (`src/morphohand/rl/trajectory_health.py`, baked into handoff
   eval + `scripts/policy_healthcheck.py`) and per-run **watchdogs** (object-height collapse).
   Judge policies on the **deterministic held-cos**, never reward sums.
2. **LOOK at a finished policy before you explain its numbers** — skill **`policy-eyes`**.
   `scripts/policy_filmstrip.py --run <run>` tiles phase-aligned frames from the run's own eval
   video into one PNG; **read the PNG**, then get the signed per-step trace. The reward table is
   sign-blind and timing-blind: a "policy that won't rotate" was rotating hard to the **wrong
   pole** (held-cos −0.449), and a "too-weak reorient reward" was gated at step 65 for a rotation
   that finished by step 50. Always also run the **open-loop probe on the same frozen scene** —
   good open-loop + bad learned = policy/grip problem, *not* morphology (this test is what stops
   a pointless morphology sweep). Re-render at `--width/--height` for anything you'll act on;
   the in-training eval videos are 320x240.
3. **Train reorientation WITHOUT skipping the lift.** A skip-lift (teleport-spawn) reorienter is
   out-of-distribution at the A→B handoff seam and drops the object. Use the **live-A reset** (a
   frozen Policy A drives the real lift 0..onset; B's pre-onset steps are PPO-masked) so B sees the
   *organic* delivery. (`scripts/train_handoff_liveA_reset.sh`, `live_a_runner.py`.)
4. **Generate morphologies with FROZEN morphology parameters.** The base scene's morphology joints
   drift during rollout → linkage drift contaminates eval. **Freeze the scene**
   (`freeze_scene_for_eval`) before grasp/policy evaluation; `generate_morphology_xml.py` bakes the
   9-param design into fixed geometry.
5. **IK-retarget the grasp keyframe across morphologies** (world-frame fingertips, NOT joint-space).
   A repositioned/lengthened finger with the baseline joint angles lands its tip in the wrong world
   spot → spurious "2-finger / ungraspable" verdicts. `scripts/retarget_keyframe_ik.py` writes an
   `open_ik` keyframe; CEM seeds from it and the RL env must use it via **`--open-finger-from-keyframe`**
   (both the reset pose AND the LerpFinger start), or the grip is wrong and the object drops.
6. **Retrain Policy A from scratch per morphology** — do NOT warmstart it. A warmstarted A loads a
   grip-specific residual that **ejects** the re-CEM'd object; from scratch the residual ≈ 0 so the
   open-loop CEM grip + scripted lift does the lifting. (Warmstarting the *reorienter* B from a
   proven reorient prior is a different, useful lever — see below.)
7. **Per-design RL reorient quality is SEED-DOMINATED.** From-scratch reorient training converges to
   qualitatively different policies (peak reorient-cos 0.0–0.9) depending on the seed's early
   exploration — the reorient reward is a hard-exploration target. A single run per design cannot
   resolve morphology differences; you need **many seeds averaged** or a **shared reorient
   warm-start** to reduce evaluator variance before any design search is meaningful.
8. **Never strip drop / tip-loss terminations** when adding a finger-perturbing reward, and always
   **warmstart the critic** (actor-only warmstart wrecks finetunes — garbage early advantages knock
   the converged actor off its optimum).
9. **Never `pkill -f` a pattern that appears in your own command** (kills your shell). Use the
   `[p]` bracket trick.

## Naming & results

- **DATE-PREFIX every generated folder.** Any directory that holds videos, renders, filmstrips,
  reports, or documentation of an attempt gets a `YYYYMMDD-` (or `YYYYMMDD-HHMM-`) prefix:
  `20260803-uhas_zeroshot_renders/`, not `uhas_zeroshot_renders/`. Attempts are only interpretable
  against the timeline they were tried on — an undated folder is indistinguishable from a rerun
  three weeks later with different code, and this program reruns the same names constantly. Run
  dirs already do this; renders, `docs/experiments/` artifact folders and ad-hoc output dirs must
  too. Files inside inherit the folder's date; don't re-date each file.
- **A/B policy registry:** `aNN` = lift/deliver, `bNN` = reorient (b01–b13 published; b14+
  chronological), `bx_` = uncanonized exploration. The **single source of truth is
  `scripts/rename_results_bids.sh`** (edit its `MAP`, run `--apply`; it renames run dirs, rewrites
  path refs, and regenerates `results/rl/REGISTRY.md`). Current reference policies: **a10** (native
  lift on the co-designed m05 hand) → **b33** (its reorient; `handoff_m05_FIXED.mp4`). Use these IDs
  everywhere, never the old P1/P2/v2 schemes.
- Morphology co-design pipeline: `scripts/morph_pipeline_sweep.py` (health-gated per-design A→B,
  resumable) + `scripts/morph_pipeline_plots.py`.
- **Trainer recipes (2026-07-09):** `rl_train_cube.py --recipe <name>` loads
  `configs/recipes/<name>.yaml` (blessed: `a_lift`, `b_liveA`, `b_liveA_imit`) as defaults — CLI
  flags still win, unknown keys are fatal. The train/deploy parity knobs (gotcha #13:
  `finger_residual_scale`, easing, contact gate) live THERE; never re-type them as flag soup.
  The object-height collapse watchdog is trainer-side (`--watchdog-collapse-z`, writes
  `<log>.COLLAPSED`); shared study plumbing = `morphohand.studies.runlib`; deploy-time env/actor
  builders = `morphohand.rl.deploy`.
- **Workspace layout (2026-07-08):** run logs / sentinels / pids → `logs/` (gitignored; never the
  repo root), experiment summaries (.txt/.json/tables) → `docs/experiments/` (tracked), free-form
  notes → `docs/notes/`. `scripts/` holds only the ~36 active scripts (map: `scripts/README.md`);
  superseded ones live in `scripts/archive/` — resurrect by moving back, don't run in place.
