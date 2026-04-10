# Morphology vs. Control DOF Split: Design Notes

## The Question

The paper assigns yaw (ψ_i) to the **outer morphology loop** alongside gantry position and phalange length, rather than the **inner grasp loop** alongside MCP and PIP flexion. Is this right?

## Current Split

| Loop | DOFs | Per finger |
|---|---|---|
| Outer (MAP-Elites) | x_i, y_i, ψ_i, l_i | 4 |
| Inner (gradient / grasp synthesis) | q^p_i, q^d_i | 2 |

## Why ψ Ended Up in the Outer Loop

1. **Compute cost.** The inner loop runs thousands of times per MAP-Elites iteration. Every DOF added multiplies cost. A 6-DOF inner loop is much cheaper than 9-DOF.

2. **Scientific legibility.** Keeping ψ in the outer loop means the MAP-Elites archive explicitly shows "inward-pointing vs. outward-pointing fingers" as distinct morphology families — which is exactly the kind of structural variation the study is meant to illuminate.

3. **Hardware framing.** Yaw physically reorients the entire finger mounting direction, which feels more like repositioning hand geometry than articulating during a grasp.

## The Counterargument

In human hands, abduction/adduction is unambiguously a **control** degree of freedom used actively mid-grasp. The distinction between "shape DOF" and "actuated DOF" is a design choice, not a physical law.

Assigning ψ to the outer loop has a real cost: the inner optimizer cannot compensate for a badly oriented finger. MCP/PIP flexion cannot rescue a finger pointing in the wrong direction. This means MAP-Elites may find good (x, y, ψ, l) combinations but leave quality on the table because the inner loop never adjusts approach angle.

This cost is object-dependent:
- **Compact symmetric objects** (spheres, cubes): yaw sensitivity is low, many orientations work.
- **Elongated objects** (cylinders): optimal yaw is tightly constrained, wrong assignment is costly.
- **Flat objects** (tablets): approach angle dominates grasp quality entirely.

Given MorphoHand's object set includes both cylinders and tablets, this is not a negligible concern.

## Cleaner Alternative Split

| Loop | DOFs | Per finger | Total |
|---|---|---|---|
| Outer (structural geometry) | x_i, y_i, l_i | 3 | 9 |
| Inner (all joint-space variables) | ψ_i, q^p_i, q^d_i | 3 | 9 |

The outer archive is indexed by base position and phalange length only — purely geometric, straightforward to interpret. The inner optimizer has full freedom to find the best approach angle and flexion for a given geometry. The archive is smaller (9D outer vs. 12D), which also makes MAP-Elites more tractable.

**Tradeoff:** 50% more expensive inner loop (9 vs. 6 joint variables per gradient step), and the behavioral descriptors (fingertip spread, reach) are now influenced by ψ as well as q, which slightly complicates descriptor interpretation.

## Recommendation

Move ψ to the inner loop if:
- Object set includes elongated or flat objects (it does)
- Compute budget tolerates a 50% heavier inner loop
- You want the science to cleanly isolate structural geometry from actuation strategy

Keep ψ in the outer loop if:
- Compute is the binding constraint
- You want the archive to explicitly map finger orientation families
- You plan a follow-up ablation that tests ψ sensitivity separately

Either way, this decision should be stated explicitly and justified in the paper — the current framing implies the split is natural when it is actually a deliberate tradeoff.