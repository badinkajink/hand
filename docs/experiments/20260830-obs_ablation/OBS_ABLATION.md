# The reference reorienter does not use its observations

Closed-loop observation ablation of **a10 → b33** on m05, 2026-08-30.
Probe: `scripts/probe_obs_ablation.py`. Raw: `OBS_ABLATION_b33_m05_{nominal,jitter}.json`.

This is §4 of `docs/rl/partial_observation_transfer.md` — the audit that memo puts *before* any
distillation training, on the grounds that no architecture can repair a non-observable task. It
cost no training time and it changes what the rest of that program should be.

## Protocol

The continuous A→B handoff (`rl_demo_handoff_continuous.py`'s env and physics, no reset at the
seam), batched over 32 envs so each condition is a distribution rather than a rollout. From the
handoff step onward, one block of b33's 66-dim actor observation is intervened on *inside* the
loop. b33's own deterministic eval is the `none` row.

Four interventions, because "the hardware cannot measure this" has four honest meanings:

| | what it does | what it tests |
|---|---|---|
| `zero` | block → 0 | information removed, but the value is off-manifold |
| `freeze` | block latched at its handoff value | one measurement, never updated |
| `shuffle` | block permuted across the env batch | marginal distribution kept, correlation destroyed |
| `replay` | block fed a baseline rollout's values at the same step | in-distribution, causally decoupled, **deployable by construction** |

`hidden` = `object_pos` + `object_pose_actual` + `target_axis_misalign`: exactly what the real_v1
bench cannot measure, since it has no object tracker. `deployable` = `joint_pos` + `joint_vel` +
`ref_finger_qpos` + `actions` — everything it *can*.

**The variance report is the control.** `shuffle` and `replay` can only destroy information that
across-env variance actually carries; if every env saw the same vector, permuting them is the
identity map. Post-handoff, across-env spread runs 1.9–5.3× the across-time spread on every
measured block, so the interventions had teeth. The two reference blocks (`ref_finger_qpos`,
`ref_object_pose`) have exactly zero across-env spread — they are functions of the step index
alone — so their rows are uninformative by construction and are flagged as such.

## Result 1 — in the nominal task, b33 is feed-forward

| condition | hold | min-z | final cos | cos&#124;held |
|---|---:|---:|---:|---:|
| `none` (baseline) | 0.97 | 0.120 | +0.867 | +0.895 |
| `replay:hidden` | 0.94 | 0.117 | +0.839 | +0.895 |
| `shuffle:hidden` | 0.94 | 0.116 | +0.849 | +0.912 |
| `freeze:hidden` | 0.97 | 0.121 | +0.862 | +0.890 |
| `replay:objstate` | 0.97 | 0.121 | +0.877 | +0.906 |
| `replay:target_axis_misalign` | 1.00 | 0.122 | +0.829 | +0.829 |
| **`replay:deployable`** | 0.97 | 0.121 | +0.845 | +0.900 |
| `zero:hidden` | 0.28 | 0.031 | −0.036 | +0.191 |

Destroying every hidden object observation costs nothing: 0.895 → 0.895 held-cos. Destroying the
*entire* observation vector — `replay:deployable` swaps the proprioception too — also costs
nothing: 0.900. b33 steers on none of its 66 inputs. Its reorientation is a learned open-loop
residual trajectory, which is consistent with the two independent results either side of it: the
turn is 46–69% floor-and-gravity work (`REORIENT_PRIMITIVE.txt`), and the same maneuver runs
open-loop from a bounded joint-space anchor at cos 0.996 with no policy at all
(`reorientation.md`, 2026-08-28).

## Result 2 — `zero` is a trap, and it is the obvious implementation

`zero:hidden` collapses the policy (hold 0.97 → 0.28, held-cos 0.895 → 0.191) while carrying
**the same information content** as `replay:hidden` and `shuffle:hidden`, which cost nothing. The
collapse is the off-manifold input value, not the missing signal.

An ablation that only zeroed blocks — the natural thing to write — would have concluded that b33
critically depends on object state, and would have justified the entire teacher/student program.
The memo warns that first-layer weight magnitude is not a reliance test; this is the same failure
one level down, and it is why the probe implements four interventions instead of one.

## Result 3 — there is no feedback to fall back on

With 5 mm xy / 5° yaw of spawn jitter and **no ablation at all**, the baseline drops to hold 0.41,
final cos +0.228. b33 does not degrade gracefully under perturbation; it stops working.

The interventions on top of that are the tell:

| condition | hold | cos&#124;held |
|---|---:|---:|
| `none` (jittered baseline) | 0.41 | +0.493 |
| `replay:joint_vel` | **0.75** | **+0.745** |
| `freeze:hidden` | 0.56 | +0.665 |
| `replay:ref_object_pose` | 0.50 | +0.780 |
| `shuffle:hidden` | 0.00 | — |

Replacing the true joint velocities with a nominal rollout's values makes the policy **better**
(0.41 → 0.75). Feeding a memorised trajectory the observations it expects keeps it on its track;
feeding it the truth about a perturbed world pushes it off. Under perturbation b33's observations
are not merely unused, they are net harmful.

## What this means for the transfer program

`partial_observation_transfer.md` is built on the premise that the deployable actor must infer
hidden object state in order to reproduce the privileged policy's behaviour, and that the gap
between them is an observability problem. On the policy we actually have, that premise does not
hold. There is no closed-loop content in b33 to distill. Its behaviour is a time-indexed
trajectory: it would distill into an 18-dim student trivially — a lookup on the step index would
fit it — and the student would inherit its brittleness exactly.

So the binding constraint is not observability. It is that **we do not yet have a reorientation
policy that uses feedback at all**, and a transfer program cannot transfer one that does not exist.

The question is only meaningful on a task an open-loop trajectory provably cannot solve. Result 3
supplies one for free: under spawn jitter the trajectory fails. That gives the 2×2 in
`scripts/train_blind_actor_2x2.sh` —

|  | nominal | jittered |
|---|---|---|
| **sighted** | S0: reproduces b33 (control) | S1: the oracle — solvable *with* object state? |
| **blind** | B0: was the object ever needed? | B1: solvable *without* it? = deployability |

where "blind" is genuine asymmetric actor-critic (actor's object terms forced to zero, critic keeps
them — `MorphoHandEnvCfg.actor_blind_terms`), and S1 − B1 is the memo's own oracle-vs-AAC decision
gate, measured rather than assumed.

## Gotchas earned here

* **A `zero` ablation and a `replay` ablation of the same block can give opposite verdicts.**
  Believe `replay`/`shuffle`; `zero` conflates missing information with an off-manifold value.
* **An ablation is only as strong as the across-env variance it destroys.** Report it. On a
  deterministic spawn, permuting across envs can be the identity map and the whole table means
  nothing. The probe prints this before the results and flags the rows it invalidates.
* **A blind-trained actor must be evaluated blind.** Handing it live values where it only ever saw
  zeros is gotcha #13 in an observation coordinate; `probe_obs_ablation.py --actor-blind-terms`
  exists for that.
* **`peak_cos` stays high in every failed condition** (0.54–0.97 while held-cos is ~0). It is
  measuring the shaft rotating on its way to the floor. Score `final_cos` on held rollouts only.
