# Simulator Backends for MorphoHand Morphology Optimization
*Reference notes — April 2026*

---

## The Three Backends at a Glance

| | MJX | MJWarp | ComFree Warp |
|---|---|---|---|
| **Differentiable** | ✅ Yes (JAX autodiff) | ❌ No | ❌ No |
| **GPU parallel** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Throughput vs MJX** | baseline | higher at scale | 2x+ vs MJWarp |
| **Contact model** | Penalty (soft) | Penalty (soft) | Analytical, complementarity-free |
| **API** | MuJoCo-compatible | MuJoCo-compatible | Drop-in MJWarp replacement |
| **Stability** | Mature | Mature | Newer, more prone to instability |
| **DOF scaling** | Good | Degrades >60 DOF | Similar to MJWarp |
| **Repo** | `google-deepmind/mujoco` (mjx/) | `google-deepmind/mujoco_warp` | `asu-iris/comfree_warp` |

---

## Role in the MorphoHand Pipeline

The three backends are **not competing for the same job**. They serve different roles in the bi-level optimization pipeline.

```
Outer loop (MAP-Elites morphology search)
│   ← no gradients needed, maximize throughput
│   → ComFree Warp (2x+ MJWarp) or MJWarp as fallback
│
└── Inner loop (grasp synthesis for fixed θ)
        ← gradient-based, optimize q to maximize ε(θ, q)
        → MJX (JAX autodiff through FK + GWS metric)
        → DiffMJX CFD trick for pre-contact gradients
        → Friction cone projection post-step (R-CTR principle)
```

### Specific role assignments

| Pipeline role | Best tool | Reasoning |
|---|---|---|
| Gradient ascent on ε w.r.t. **q** (joint angles) | **MJX** | Only differentiable option |
| Gradient flow from ε → q → **θ** (morphology) via FK chain | **MJX** | JAX autodiff through MJCF FK |
| MAP-Elites batch forward eval (non-differentiable) | **ComFree Warp** | 2x+ throughput; quasi-static tolerance for instability |
| Rollout-based quality eval / shake test sim | **ComFree or MJWarp** | No gradients needed; MJWarp more stable |
| Ground truth / contact debugging | **MuJoCo CPU** | Slowest, most accurate |

---

## Contact Gradient Issues in MJX and Fixes

MJX uses a **penalty-based soft contact model** (spring-damper). This causes two problems for gradient-based optimization:

1. **Zero gradient before contact.** When fingers are not touching the object, there is no contact force and therefore no gradient signal pushing them toward it.

2. **Inaccurate gradients at hard contacts.** Physically realistic contacts require stiff springs. Stiff springs create stiff ODEs → exploding or vanishing gradients under autodiff.

### Fix: DiffMJX (Contacts From Distance, CFD)

[DiffMJX](https://arxiv.org/abs/2506.14186) introduces two corrections applied only in the **backward pass** (straight-through trick), leaving the forward simulation unchanged:

- **Adaptive integration** at contact events to maintain gradient accuracy.
- **Contacts From Distance (CFD)**: artificial contact forces proportional to distance, providing gradient signal *before* fingers touch the object.

Both corrections preserve forward physical accuracy while making the gradient landscape useful for optimization.

---

## The Contact Trust Region (CTR) Paper — What Transfers

**Paper:** Suh & Pang et al., *"Dexterous Contact-Rich Manipulation via the Contact Trust Region"* (2025). Built on the **CQDC (Convex Quasidynamic Differentiable Contact)** model, implemented in **Drake**.

### What CQDC provides that MuJoCo does not

CQDC formulates each timestep as a **convex SOCP**. Sensitivity analysis on a convex program yields both:
- **Primal sensitivity**: $\partial q_+ / \partial q$, $\partial q_+ / \partial u$ (next-state Jacobians) — also available from MJX via autodiff
- **Dual sensitivity**: $\partial \lambda_i / \partial q$, $\partial \lambda_i / \partial u$ (contact force Jacobians) — **NOT available from MJX**

CTR requires the dual sensitivity to build a linearized model of how contact forces change with action, then impose friction cone constraints $\hat{\lambda}_{+,i} \in K^*_i$ on the trust region. This filters out physically impossible actions (e.g., "pulling" on an object).

### CTR is NOT directly usable with MJX/MJWarp/ComFree

MJX only provides primal autodiff gradients through the penalty softening. There is no structured primal/dual decomposition — you cannot extract the contact force Jacobians $C_{\kappa,i}$, $D_{\kappa,i}$ that CTR needs. You could finite-difference an approximation, but the signal would be noisy and the resulting friction cone geometry would be unreliable at soft contacts.

### What does transfer: the R-CTR principle

The core physical insight is worth keeping:

> Standard gradient descent + an ellipsoidal trust region allows the optimizer to propose joint angle updates that would require **tensile normal forces** — i.e., the fingers "pulling" on the object, which is physically impossible. This causes catastrophic failures when the current contact configuration does not form a force-closure grasp.

The **Relaxed CTR (R-CTR)** fixes this by only imposing **dual feasibility** (friction cone constraints) rather than both primal and dual feasibility.

**Practical approximation for MJX inner loop:**
After each gradient step on **q**, project the resulting contact forces back onto the friction cone:
- Check whether $\lambda_i \in K^*_i$ (Coulomb cone: $\mu \lambda_{n,i} \geq \|\lambda_{t,i}\|$)
- If not, project **q** back toward the previously feasible configuration

This is less principled than full CTR (no linear dual model) but prevents the same failure mode: the optimizer moving fingers to configurations that require pulling.

---

## ComFree and the CTR Long-Term Angle

Jin's ComFree formulation and the CQDC model used in CTR are **spiritually the same thing**: both replace iterative complementarity-based contact resolution with a complementarity-free analytical formulation that computes contact forces in closed form. Key citations:

- Jin 2024, *"Complementarity-free multi-contact modeling and optimization"* — directly related to ComFree
- Pang et al. 2023, CQDC — the model underlying CTR

The ComFree docs list **"differentiable contact dynamics learning (real-to-sim)"** as a future application, signaling that Warp autodiff through ComFree is on the roadmap.

**If/when ComFree exposes differentiability:**
- You'd have both primal and dual sensitivity available (ComFree resolves contact forces analytically, so $\partial \lambda_i / \partial u$ is accessible)
- CTR could be implemented directly on top of ComFree Warp
- This would give you a CTR-style inner loop with 2x+ throughput vs MJWarp — a meaningful contribution in itself

---

## Practical Comparison: ComFree vs MJWarp for the Outer Loop

These two backends are genuinely interchangeable in the outer MAP-Elites loop. The comparison worth running empirically:

**Question:** Does ComFree's harder analytical contacts produce morphology quality rankings that correlate better with physical hardware outcomes than MJWarp's softened penalty contacts?

ComFree's stiffer contact model may produce a GWS epsilon that is a more faithful proxy for real-world grasp stability, because it doesn't have the force-from-a-distance artifact that penalty softening introduces. If true, this would mean:
- ComFree is both faster and more predictive
- The sim-to-real rank correlation (Experiment 1 of MorphoHand) would be higher for ComFree-evaluated morphologies

This is a legitimate empirical claim to test by running MAP-Elites with both backends and comparing Kendall τ against physical hardware outcomes.

**ComFree instability caveats:**
- Requires tuning `comfree_stiffness` and `comfree_damping` per scene
- More prone to divergence on edge-case morphologies (extreme yaw, minimum phalange length)
- Keep MJWarp as a fallback for morphologies where ComFree misbehaves
- The docs themselves say: *"if simulation speed is not a primary concern, we recommend continuing to use MuJoCo (Warp)"*

---

## Migration Snippet: MJWarp → ComFree Warp

```python
# MJWarp
import mujoco_warp as mjwarp

m = mjwarp.put_model(mjm)
d = mjwarp.put_data(mjm, mjd, nworld=N, nconmax=1000, njmax=5000)
mjwarp.step(m, d)
mjwarp.get_data_into(mjd, mjm, d)

# ComFree Warp (near drop-in)
import comfree_warp as cfwarp

m = cfwarp.put_model(
    mjm,
    comfree_stiffness=0.2,   # tune per scene
    comfree_damping=0.001,
)
d = cfwarp.put_data(mjm, mjd, nworld=N, nconmax=1000, njmax=5000)
cfwarp.step(m, d)
cfwarp.get_data_into(mjd, mjm, d)
```

The contact resolution is entirely different internally (`forward_comfree` replaces MJWarp's forward path), but the workflow is identical.

---

## Summary Decision Tree

```
Do you need gradients?
├── YES → MJX
│         (+ DiffMJX CFD for pre-contact signal)
│         (+ friction cone projection for R-CTR principle)
└── NO  → Do you need maximum throughput?
          ├── YES → ComFree Warp
          │         (2x+ MJWarp, tune stiffness/damping)
          │         (fallback to MJWarp on unstable morphologies)
          └── STABILITY PRIORITY → MJWarp
                                   (more mature, less tuning)
```

---

## References

- Suh & Pang et al. (2025). *Dexterous Contact-Rich Manipulation via the Contact Trust Region.* arXiv:2505.02291
- Pang et al. (2023). *Global Planning for Contact-Rich Manipulation via Local Smoothing of Quasidynamic Contact Models.* IEEE T-RO.
- Jin (2024). *Complementarity-free multi-contact modeling and optimization for dexterous manipulation.* arXiv:2408.07855
- Borse, Xie, Huang, Jin (2026). *ComFree-Sim: A GPU-Parallelized Analytical Contact Physics Engine.* arXiv:2603.12185
- Sferrazza et al. (2025). *Hard Contacts with Soft Gradients: Refining Differentiable Simulators for Learning and Control (DiffMJX).* arXiv:2506.14186
- MJX docs: https://mujoco.readthedocs.io/en/latest/mjx.html
- MJWarp docs: https://mujoco.readthedocs.io/en/latest/mjwarp/
- ComFree docs: https://irislab.tech/comfree-doc/intro.html
