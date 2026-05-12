# Grasp specification methods

Three orthogonal approaches to specifying / scoring grasps, each behind a
config toggle so they compose with the existing Phase 1 evaluator. All
default to off — enabling them is opt-in via `Phase1EvalConfig` weights
(except synergy, which is a different optimizer entry point entirely).

| Method | What it changes | Toggle | Doc |
|---|---|---|---|
| **Eigengrasp / synergy CEM** | Search in K-dim PCA subspace of historical grasps instead of raw 9D joints | use `optimize_finger_controls_synergy` instead of `optimize_finger_controls` | [eigengrasp.md](eigengrasp.md) |
| **Contact-target patches** | Reward fingertip landings on user-authored patches in object body-local frame | `objective_weight_contact_target_reward`, `_distance_penalty` (default 0.0) | [contact_targets.md](contact_targets.md) |
| **Force-closure energy** | Continuous Ferrari-Canny / DFC grasp-quality term replacing raw contact count | `objective_weight_force_closure` (default 0.0) | [force_closure.md](force_closure.md) |

## When to use which

- **Authoring a new object / scene and the baseline CEM fails or is
  unreliable**: try contact-target patches first. Cheapest authoring,
  largest documented win (prism +3.26 in the eval suite).
- **Sample efficiency matters and you have ≥10 historical CEM runs to
  fit a basis on**: try synergy CEM. K=4 captures 90% of joint variance.
- **You suspect the baseline contact-count term is rewarding the wrong
  thing (pile of contacts on one face)**: add force-closure energy. It
  identifies 3-finger wraps that the count term misses.

## Empirical comparison

See [method_comparison.md](method_comparison.md) for a side-by-side at
matched CEM budget on the active drill scene. Across the full eval
suite (8 objects, 3 seeds each, scenes frozen), the headline numbers are
in [eval_suite.md](eval_suite.md).

## Why each exists (research background)

These three approaches map to three distinct families in the dexterous
grasping literature — they are *complementary*, not redundant:

- Eigengrasps are the joint-space dimensionality-reduction tradition
  (Ciocarlie & Allen, RSS 2007).
- Contact-target patches are a hand-authored degenerate case of the
  contact-map representation used by ContactGrasp / ContactPose /
  GenDexGrasp.
- Force-closure energy is the analytic grasp-quality tradition
  (Ferrari-Canny) made continuous and used by the DFC line of work
  (Liu et al. 2021, DexGraspNet, GraspQP CoRL 2025).

Survey of recent learning-based work that informed these choices is
linked from each method's individual doc.

## All three together

The "combined" method (synergy + contact-target + FC) is currently
**broken** — at the budgets we've tested, the synergy subspace plus a
strong distance penalty plus FC-clamping-to-zero on no-contact configs
creates a perverse incentive where CEM converges to no contacts.
Detailed failure analysis and three concrete fix paths are in
[method_comparison.md](method_comparison.md#combined-synergy--contact-target--fc-broken-interaction).

## Required prep: frozen scenes

Every grasp eval **must** run against a frozen scene (morph DOFs baked
into body transforms). The base scenes have morph joints that drift
during the rollout, invalidating any cross-method comparison. The eval
harnesses and `phase1_optimize_grasp.py` already enforce this; new
scripts should call `morphohand.sampling.scene.freeze_scene_for_eval`.
See [frozen_scene_protocol.md](frozen_scene_protocol.md).
