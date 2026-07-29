# Contact target patches as a grasp specification

## What this is

A way to tell the optimizer "thumb should contact *here*, index *there*"
without dictating any joint angles. Instead of relying solely on a generic
"how close is each fingertip to the object" distance term, this branch lets
you author a per-object YAML file specifying named target patches on the
object's surface — and adds two new score terms that reward landing tips on
those patches.

The representation is the lightest-weight version of the "contact map"
parameterization used in ContactGrasp, ContactPose, GenDexGrasp and
Contact2Grasp: a sparse set of (position, radius, optional normal, optional
finger assignment) tuples in body-local frame. The body-local framing means
patches travel with the object as it tilts during the lift and pivot phases,
which is what we actually want.

## Files

- `src/morphohand/optimization/contact_targets.py` — `ContactTarget`,
  `ContactTargetSet`, scoring (`score_contact_targets`), YAML loader,
  Hungarian-style finger-to-patch assignment.
- `src/morphohand/optimization/phase1_common.py` — evaluator now accepts a
  `contact_target_set=` kwarg and exposes two new weights:
  - `objective_weight_contact_target_reward` (smooth in-patch bonus)
  - `objective_weight_contact_target_distance_penalty` (dense gradient pull)
- `assets/contact_targets/power_drill_short_proximal.yaml` — example
  three-patch spec for the active drill scene.
- `tests/test_contact_targets.py` — unit tests.

## YAML schema

```yaml
object_body: power_drill   # informational; must match the actual body name
patches:
  - name: thumb_proximal_left
    finger: thumb          # optional; one of thumb/index/middle, else auto-assign
    local_pos: [x, y, z]   # body-local meters
    local_normal: [nx, ny, nz]  # optional; informational at the moment
    radius: 0.012          # smooth-reward saturation radius
```

## Scoring

For each patch, the assigned fingertip's distance to the patch center yields:
- a smooth in-patch reward (1.0 inside `radius`, decaying to 0 by `3*radius`),
- and a raw mean-distance penalty.

Both signals are added to the existing weighted-sum objective with the new
weights, so this composes cleanly with the lift / contact-count / drift terms
already in the evaluator. Setting both weights to 0 (the default) restores
the original objective exactly.

## How to use

```python
from morphohand.optimization.contact_targets import ContactTargetSet
from morphohand.optimization.phase1_common import Phase1GraspEvaluator, Phase1EvalConfig

ts = ContactTargetSet.from_yaml("assets/contact_targets/power_drill_short_proximal.yaml")
cfg = Phase1EvalConfig(
    objective_weight_contact_target_reward=10.0,
    objective_weight_contact_target_distance_penalty=20.0,
)
ev = Phase1GraspEvaluator(
    scene_xml="assets/mjcf/baseline/scenes/scene_power_drill_short_proximal.xml",
    keyframe="open_flat",
    cfg=cfg,
    contact_target_set=ts,
)
```

The CEM strategy then optimizes finger control as before — but now toward
the patches rather than toward any contact at all.

## What to check experimentally

- With these weights set, CEM should reliably steer tips to the authored
  patches even before any drill geometry actually contacts the fingertips.
- A useful ablation is to flip just one finger's assignment (say, swap
  thumb and middle in the YAML) and confirm the assignment is still
  recovered correctly — patches with a `finger:` tag are pinned, while
  patches without one auto-assign.
- The radius is the main knob: tighter radius = sharper localization but
  weaker gradient far from the patch; the distance penalty fixes that.
