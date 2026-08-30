# 2026-08-30 — session handoff

Written mid-session for continuity; two training arms were still running. Branch
**`partial-obs-transfer`**, commits `83c02f4 → 19c1ef4`. Nothing merged to `main`; the
pre-existing dirty state (`hand_paper/` deletions, `.gitignore`, submodule, the one mjcf) is
untouched.

Two independent threads ran today. The second is the deliverable; the first is why the
program's stated next step was the wrong one.

---

## 1. The reference reorienter does not use its observations

`docs/rl/partial_observation_transfer.md` (Codex's memo) proposes a teacher/student + RMA
program premised on the deployable actor needing to infer hidden object state. Its own §4 says
audit observability before training anything; §10 then orders that audit third, behind two
implementation steps, so the premise was never checked. It costs 30 minutes.

**New probe `scripts/probe_obs_ablation.py`** intervenes on one observation block at a time
*inside* the closed loop of the continuous A→B handoff, 32 envs per condition. On a10 → b33
(m05), nominal task:

| condition | hold | cos&#124;held |
|---|---:|---:|
| baseline | 0.97 | +0.895 |
| replay all 11 hidden object dims | 0.94 | +0.895 |
| shuffle same | 0.94 | +0.912 |
| freeze same | 0.97 | +0.890 |
| **replay the ENTIRE 66-dim input from another env** | 0.97 | **+0.900** |
| zero the 11 hidden dims | 0.28 | +0.191 |

**b33 steers on none of its 66 inputs** — not the object blocks, not the proprioception. Its
reorientation is a learned open-loop residual trajectory. Consistent with the turn being 46–69%
floor work (`REORIENT_PRIMITIVE.txt`) and with the anchor running open-loop at cos 0.996.

**Therefore the memo's premise fails on the policy we have.** There is no closed-loop content in
b33 to distill; it would distill into a step-index lookup and inherit its brittleness. The
binding constraint is not observability — we have no reorient policy that uses feedback at all.

Three methodological points, each of which flips the conclusion on its own:

* **`zero` ablations lie.** `zero:hidden` collapses the policy while carrying the *same*
  information as `replay:hidden`, which costs nothing. The collapse is the off-manifold value.
  An ablation that only zeroed — the obvious implementation — would have "confirmed" the whole
  distillation program.
* **Report the across-env variance.** shuffle/replay can only destroy variance that exists; on a
  deterministic spawn, permuting envs is the identity map. Here across-env sd was 1.9–5.3× the
  across-time sd, so the interventions bit. `ref_finger_qpos` and `ref_object_pose` have exactly
  **zero** across-env spread — functions of the step index — so `ref_object_pose` is freely
  available on hardware, contra the memo's §1.1 table.
* **Score held rollouts only.** `peak_cos` reads 0.54–0.97 in every *failed* condition.

**b33 has no feedback to fall back on either.** 5 mm / 5° spawn jitter alone drops hold
0.97 → 0.41. Under jitter, `replay:joint_vel` makes it *better* (0.41 → 0.75): its observations
are net harmful there.

Full tables: `docs/experiments/20260830-obs_ablation/OBS_ABLATION.md`.

### Built for this

Genuine **asymmetric actor-critic**, which the env did not previously have (both groups got the
same terms; only noise differed — the memo's §2.3 caught this):

* `MorphoHandEnvCfg.actor_blind_terms` — actor's terms forced to zero via mjlab `scale=0.0`,
  critic keeps the truth. Blinding is scale, **not deletion**, so the vector stays 66-dim and
  b33's actor+critic still warmstart (warmstarted sd 0.032 vs from-scratch 0.3–0.5).
* `MorphoHandEnvCfg.actor_obs_history` — mjlab-native frame stacking. **No RNN needed:**
  `ObservationTermCfg` already has `history_length`, `delay_min_lag/max_lag`, `scale`, noise. The
  memo's §3.1/§8 RNNModel plumbing is largely unnecessary, and stacking keeps the ONNX export the
  CB1 deploy wants (HORA's adaptation module is a conv over proprioceptive history, not an RNN).

### Still running / how to resume

`scripts/train_blind_actor_2x2.sh` — sighted/blind × nominal/jittered, warmstarted from b33,
`assert_config_parity.py`-gated at **exact** parity with b33's config.
`logs/run_decisive_arms.sh` runs only the decisive pair at 5M timesteps, 2 seeds.

State at time of writing: `S1_s42` ✅, `B1_s42` ✅, `S1_s43` at 61/67, `B1_s43` pending.
Sentinels in `logs/blind2x2/*.DONE`; **re-running the script resumes** (skips completed arms).

**Score with `scripts/eval_blind_actor_2x2.sh`.** The blind arms MUST be evaluated with
`--actor-blind-terms` applied — a blind-trained actor read out sighted is gotcha #13 in an
observation coordinate.

**S1 − B1 on the jittered test is the whole point:** the memo's oracle-vs-AAC gate, measured.

---

## 2. What the bench should run: g12, and only g12

**All four exported plans reorient the tool** — physics replay of each `<design>_traj.csv` in its
own deploy scene (new `scripts/real_v1_render_deploy_plan.py --physics`):

| design | chord | csv | physics cos | z | contacts | verdict |
|---|---:|---:|---:|---:|---:|---|
| **g12** | **+8.7 mm** | **+8.8 mm** | +0.780 | 111 mm | 3 | **RUN** |
| g23 | +0.8 | +0.8 | +0.737 | 116 | 3 | no safe path |
| g24 | −5.3 | −4.2 | +0.779 | 116 | 2 | fingers interpenetrate |
| rv04_mid | −2.7 | −2.0 | +0.781 | 116 | 3 | fingers interpenetrate |

**Task performance does not separate these designs (0.737–0.781). Self-collision completely
does.** g12 is the only one clearing on both paths, and independently the best catalog entry
(careful-bench win 0.65, kept 0.84).

g12 trace: horizontal on the support at t=0 (cos +0.052), clear of the post by t≈1.3 s, turn
through 1.76–2.64 s, then flat at **cos 0.780 / z 111 mm on three contacts for the final 1.3 s.**
cos 0.78 ≈ 39° off vertical — the plans command −70°, not −90°. A *partial* reorientation,
stably held. Render + mp4 in `docs/experiments/20260830-deploy_renders/`.

**Why the existing renders could not have caught this:** everything visual we had was of the
*dense carry*, which is clean on all four (28.9–38.8 mm). The collisions exist only on the
exported paths. `scripts/real_v1_trajectory_clearance.py` (pre-existing — I duplicated part of it
before finding it; check `scripts/README.md` first) is the authoritative gate. **Run it before
any bench session.**

### Mass is not a constraint

Swept through g12's own exported CSV: **0.779 / 0.786 / 0.788 / 0.790 at 24 / 50 / 65 / 85 g**,
held at 107–112 mm, no cliff anywhere.

This *contradicts* the dense-carry envelope (`20260830-carry_mass_envelope/`), which had
rv05_manual dropping between 65 and 70 g — and the disagreement is the finding: **mass tolerance
belongs to the deployed trajectory, not to the hand.** g12 clamps 10 mm of pad squeeze where
that sweep used the CEM grasp, which is the grip-vs-reorientation trade the bench already found.
Sweep the plan you intend to run.

Measured objects: bench cylinder **24 g**, intended screwdriver **65 g**. Both inside g12's range.

**Carry to the bench:** at 65 g the shaft **sinks 2.2 mm over the final 0.8 s** while alignment
stays flat to 0.001. The plan's hold is 1.6 s; anything longer meets slip before it meets any
loss of orientation.

---

## 3. Paper corrections (`paper/appendix.tex` — gitignored, disk only, compiles clean)

* **"Policy B adds learned residual feedback"** → it does not. Rewritten to keep the design
  intent and state the measurement, plus the two cautions that make the test trustworthy.
* **Kept criterion** was "hold-phase minimum object height exceeds 0.05 m". At 70 g a *dropped*
  shaft stands upright on the table and reads **alignment 0.999 at z = 0.0503 m with zero
  contacts** — that rule scores a total failure as a near-perfect reorientation, by 0.3 mm.
  `probe_real_v1_carry.py` already required contact; only the paper's description was wrong.
* **"Doubling the mass causes the open-loop solution to fail"** → 2.7×, not 2×, plus the deployed
  trajectory's flat 24–85 g range and the slip note.

---

## 4. GPU throughput: there is no lever (I tested my own hypothesis and it failed)

`scripts/bench_env_throughput.py`. Live trainer: **Collection 7.908 s vs Learning 0.090 s** — the
PPO update is 1.1% of wall clock, so no learning-side knob matters. GPU utilisation reads 43–46%,
which looked like headroom. It is not:

| num_envs | steps/s |
|---:|---:|
| 1,024 | 9,473 |
| 3,072 | 9,973 |

3× the envs for 5% throughput, and a hard crash at 6144. CUDA graph capture is already on
(mjlab captures step/forward/reset/sense). The solver is already at `iterations=10,
ls_iterations=20` — **the memory claiming untuned 100/50 is stale.** Remaining knobs
(ls→10, elliptic→pyramidal cone) change contact dynamics, which is the one thing not to touch.

**The real lever is budget.** b33's own 20M run converged at **iteration 26 of 271** (target-axis
55.9 at it26 vs 55.1 at it134 vs 60.5 at it270; reward 405 → 407). 90% of every run in this
lineage is wasted. Queue default is now 5M and arms are selectable via `ARMS=`.

---

## 5. Open

1. **Finish + score the 2×2** (`bash logs/run_decisive_arms.sh` resumes; then
   `eval_blind_actor_2x2.sh`). S1 − B1 jittered is the number.
2. **Bench g12.** Set gantry blocks to thumb (−42.5, 0), index (42.5, 40), middle (42.5, −40) mm
   per `deploy/g12_build.txt`. Known mechanical failures from 2026-08-29 remain unaddressed: yaw
   joints 4–6° short under load, and position control over-clamping at 10 mm squeeze.
3. **Untested, not negative:** whether more grip extends the mass envelope on the carry path.
   The `--squeeze` sweep returned byte-identical numbers because it is unused on `--morph-run`.
   Same "no teeth" failure as the deterministic-spawn shuffle — check that a parameter moved
   before believing it did not matter.
