# Implementing DiffMJX from Scratch
### A Practical Guide to the Changes Needed in MuJoCo XLA

*Based on: Paulus, Geist et al. — "Differentiable Simulation of Hard Contacts with Soft Gradients for Learning and Control" (arXiv:2506.14186v2, March 2026)*

*Note: The `martius-lab/diffmjx` repository exists but has no code published as of this writing. This document extracts everything the paper describes with enough specificity to reimplement independently.*

---

## Overview: What DiffMJX Actually Is

DiffMJX is **three independent modifications to MJX**, each addressing a different failure mode:

| Modification | Problem solved | Difficulty | MorphoHand priority |
|---|---|---|---|
| **Smooth collision detection** | Hard case distinctions in collision geometry → discontinuous gradients | ⭐ Low | High (apply first) |
| **Contacts From Distance (CFD)** | Zero gradients when fingers not yet in contact | ⭐⭐ Medium | Critical |
| **Adaptive integration (Diffrax)** | Fixed-step discretization errors at stiff contacts | ⭐⭐⭐ High | Lower (quasi-static regime) |

They can be implemented independently. For MorphoHand's quasi-static grasp synthesis, **smooth collision detection + CFD gives you most of the benefit without the hardest engineering work.**

---

## Root Cause Analysis: Why MJX Gradients Break

Before touching code, it helps to understand *why* the gradients go wrong. The paper identifies two separate failure modes:

### Failure Mode 1: Discretization errors at stiff contacts

MJX uses semi-implicit Euler with a fixed step size `h` (default 2ms). At contact, the penalty force creates a very stiff ODE — spring stiffness `k = d(r) / (d_w^2 * t_c^2 * phi_d^2)` can be enormous when `solref` is set for hard contacts.

Stiff ODEs require small step sizes for accurate integration. At `h = 2ms`, the integrator takes large steps over the stiff contact region, accumulating discretization errors. The key insight: these discretization errors affect the *gradient* (which must be accurate to the same order as the ODE integration) even more than they affect the state.

The fix is to use an adaptive integrator that automatically takes small steps during contact and large steps elsewhere.

### Failure Mode 2: Vanishing gradients before contact

When there is no contact (`r > 0`), the impedance `d(r) = 0`, so contact forces are exactly zero. The gradient of any loss with respect to finger pose is also exactly zero — there is no signal pushing fingers toward the object. This is fatal for gradient-based grasp synthesis starting from a non-contact configuration.

The fix is CFD: apply artificial contact forces for `r > 0` *only in the backward pass*, leaving the forward simulation unchanged.

---

## Modification 1: Smooth Collision Detection

### What needs to change

MJX's collision detection functions contain discrete `if/else` branches (implemented as `jax.lax.cond` or direct conditionals) that pick different geometric formulas based on the relative configuration of bodies. These branches are non-differentiable at the boundaries.

The affected collision pairs (from the paper's Appendix C.1):
- `plane-cylinder`
- `sphere-capsule`
- `capsule-capsule`
- `plane-capsule`
- `plane-box`

Mesh-mesh collisions are not modified (not relevant for most robot hand tasks; fingertips are typically spheres, capsules, or boxes).

### The fix: sigmoid interpolation

Replace hard `jax.lax.cond(condition, branch_a, branch_b, ...)` with soft blending:

```python
# Hard (original MJX pattern)
result = jax.lax.cond(geom_parallel, fn_parallel, fn_general, ...)

# Soft (DiffMJX pattern)
w = sigmoid(alignment_score, low=threshold_low, high=threshold_high)
result = w * fn_parallel(...) + (1 - w) * fn_general(...)
```

Where the sigmoid helper is:
```python
def sigmoid(x, low, high):
    """Smooth step from 0 at x=low to 1 at x=high."""
    t = jnp.clip((x - low) / (high - low + 1e-8), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)  # smoothstep

def soft_where(condition, x, y):
    return condition * x + (1.0 - condition) * y
```

### Plane-box example (from the paper, directly)

The paper gives this concrete example for `_plane_box`:

```python
def _plane_box_smooth(plane: GeomInfo, convex: ConvexInfo) -> Collision:
    vert = convex.vert

    # Points in the convex frame
    plane_pos = convex.mat.T @ (plane.pos - convex.pos)
    n = convex.mat.T @ plane.mat[:, 2]
    support = (plane_pos - vert) @ n

    # Soft manifold selection instead of hard threshold
    threshold = jnp.maximum(0, support.max() - 1e-3)
    soft_poly_mask = sigmoid(support, low=threshold - 1e-3, high=threshold)
    support_masked = soft_where(soft_poly_mask, support, 0)

    dist_neg, idx = jax.lax.top_k(support_masked, 4)
    dist = -dist_neg
    pos = vert[idx]

    # Convert to world frame
    pos = convex.pos + pos @ convex.mat.T
    n = plane.mat[:, 2]
    frame = jnp.stack([make_frame(n)] * 4, axis=0)
    pos = pos - 0.5 * dist[:, None] * n
    return dist, pos, frame
```

The key change is `threshold` → soft mask vs. `support.max() - tol` → hard boolean mask.

### Where to find these functions in MJX source

MJX's collision functions live in `mujoco/mjx/_src/collision_driver.py` and imported geometry-specific files. Look for functions named:
- `_plane_convex` or `plane_box`, `plane_capsule`, `plane_cylinder`
- `_capsule_capsule`, `_sphere_capsule`

Each will have a `jax.lax.cond` or a Python `if` on a geometric predicate (e.g., `is_parallel = jnp.abs(jnp.dot(axis, normal)) > 1 - eps`). These are the targets.

### Muscle actuator discontinuities

If using muscle-tendon models (not relevant for MorphoHand's rigid fingers, but noted for completeness): MJX's tendon wrapping functions use `jnp.arcsin` which is undefined at ±1. Fix with the "double-where trick":

```python
# Double-where trick: prevent gradient at singularity
safe_arg = jnp.where(jnp.abs(arg) < 1.0 - eps, arg, 0.0)
result = jnp.where(jnp.abs(arg) < 1.0 - eps, jnp.arcsin(safe_arg), fallback)
```

### Tractability: ★★★★★ (Easy)

This is pure JAX code surgery — no algorithmic changes, just replacing hard conditionals with soft ones. Estimated effort: **3–7 days**. Each collision pair is a self-contained function; you can fix them one at a time and unit test each against central-difference gradients.

---

## Modification 2: Contacts From Distance (CFD)

### Core concept

MuJoCo's contact solver applies forces based on the impedance `d(r)` and reference acceleration `h(r)`. Normally:
- `d(r) = 0` for `r >= 0` (bodies not touching → no force)
- `h(r) = 0` for `r >= 0`

CFD extends both functions to produce small nonzero values for `r > 0` (bodies approaching but not yet touching), parameterized by a distance `wc` and shape parameters.

**The critical trick:** the modified functions are used **only in the backward pass** via the straight-through estimator. The forward pass uses the original unmodified simulation. This means:
- Forward physics: identical to standard MJX (no hovering artifacts, no sim-to-real gap)
- Backward gradients: include artificial near-contact forces that provide pre-contact signal

### Step 1: Implement the CFD impedance extension

Standard MuJoCo `solimp` parameterizes `d(r)` as a polynomial spline for `r < 0`. CFD adds a second spline for `r > 0`:

```python
def impedance_cfd(r: float, solimp: array, solimp_cfd: array) -> float:
    """
    solimp_cfd = [dc, d0, wc, mc, pc]
      dc:  value d tapers to at large r (default 0.0)
      d0:  value at r=0 (should match solimp's value at r=0 for smoothness)
      wc:  cutoff distance for artificial contacts (e.g. 0.05 m)
      mc:  midpoint of the spline
      pc:  power of the spline
    """
    dc, d0, wc, mc, pc = solimp_cfd

    # Standard impedance for r < 0 (contact active)
    d_contact = standard_impedance(r, solimp)

    # CFD extension for r > 0 (not yet in contact)
    # Polynomial spline from d0 at r=0 to dc at r=wc
    t = jnp.clip(r / (wc + 1e-8), 0.0, 1.0)
    mid = mc  # normalized midpoint
    if r <= wc * mid:
        d_cfd = d0 + (0.5 * dc - d0) * (t / mid) ** pc
    else:
        d_cfd = 0.5 * dc + 0.5 * dc * ((t - mid) / (1.0 - mid + 1e-8)) ** pc
    d_cfd = jnp.where(r > wc, dc, d_cfd)  # taper to dc beyond wc
    d_cfd = jnp.where(r > 0, d_cfd, 0.0)  # only active for r > 0

    return d_contact + d_cfd  # add; for r < 0, d_cfd = 0
```

Soften the reference acceleration (replace hard ReLU with softplus):
```python
# Original MJX: h(r) = max(0, -r) * k  (ReLU, non-differentiable at 0)
# DiffMJX:
def h_cfd(r, k):
    return jax.nn.softplus(-r * k_scale) / k_scale * k  # smooth at r=0
```

### Step 2: Implement the straight-through trick

This is the most important and simplest part. You need two forward passes:

```python
import jax
from jax.lax import stop_gradient

def step_cfd(model: mjx.Model, data: mjx.Data, cfd_params: dict) -> mjx.Data:
    """
    Run one simulation step with CFD gradients but standard forward physics.
    """
    # Forward pass 1: standard MJX (no CFD) — this is what you "physically" observe
    data_mjx = _step_standard(model, data)

    # Forward pass 2: CFD-augmented (produces artificial near-contact forces)
    data_cfd = _step_with_cfd(model, data, cfd_params)

    # Straight-through: forward = standard physics, gradient = CFD physics
    # Evaluates to data_mjx in forward pass
    # Gradient is d(data_cfd)/d(params) evaluated at data_mjx's trajectory
    reroute = lambda x_mjx, x_cfd: stop_gradient(x_mjx) + x_cfd - stop_gradient(x_cfd)
    return jax.tree_util.tree_map(reroute, data_mjx, data_cfd)
```

This is exactly the code shown in the paper (Section 4), just cleaned up. The `stop_gradient` calls ensure:
- `forward pass`: `stop_gradient(x_mjx) + x_cfd - stop_gradient(x_cfd)` = `x_mjx + 0` = `x_mjx` ✓
- `backward pass`: gradient of `x_cfd` evaluated at `x_mjx`'s trajectory ✓

### Step 3: Implement `_step_standard` and `_step_with_cfd`

Both functions are standard `mjx.step` calls, but `_step_with_cfd` modifies the contact model parameters before solving:

```python
def _step_with_cfd(model, data, cfd_params):
    # Temporarily patch the model's impedance function to include CFD extension
    model_cfd = model.replace(
        # Replace solimp with CFD-extended version
        # Exact field depends on MJX version; inspect mjx.Model dataclass
        geom_solimp=extend_solimp_with_cfd(model.geom_solimp, cfd_params),
    )
    # Also patch the reference acceleration to use softplus instead of ReLU
    # This may require modifying the contact solver call directly
    return mjx.step(model_cfd, data)
```

The challenge here is that MJX's `Model` is an immutable JAX pytree, and the impedance computation is buried inside `mjx/step`. You may need to:
1. Fork the relevant portion of `mjx/_src/constraint.py` that computes `aref` and `d(r)`
2. Add a CFD-mode flag and modified impedance path
3. Call your modified version instead of `mjx.step` directly

### Step 4: Important corner case — CFD + adaptive integration

From the paper (Appendix C.4): if you combine CFD with Diffrax's adaptive integration using *discretize-then-optimize* (unrolling), gradient oscillations can appear. This happens when the forward ODE is non-stiff (no real contacts happening) but the adjoint ODE is stiff (due to CFD forces). The fix: use *optimize-then-discretize* (Diffrax's `BacksolveAdjoint`) in this case.

For MorphoHand, where gravity keeps fingers in frequent contact with objects during grasp evaluation, this corner case is unlikely to matter. But if you're doing pre-contact approach trajectories with no other contacts, use `BacksolveAdjoint`.

### Tuning CFD parameters

From the paper's experiments:
```
solimp-CFD = [dc, d0, wc, mc, pc]

For system identification (cube toss):
  dc = 0.01, d0 = 0.0, wc = 1.0 m  (very wide, gentle)

For MPC manipulation tasks:
  dc = 0.1, d0 = 0.95, wc = 0.001 m  (narrow, steep)

Practical starting point for grasp synthesis:
  dc = 0.0, d0 = 0.01-0.05, wc = 0.05-0.10 m
  (match d0 to your solimp's value at r=0 for smoothness)
```

### Important limitation: CFD is bad for grasping cold start

The paper explicitly warns (Appendix D): "At a distance, the cumulative force arising from the hand's CFDs conglomerate into a force that **pushes objects away**. However, the gradient does not encode the possibility of grasping, as this signal only emerges when the object is inside the hand."

This is a real issue for pre-grasp approach. The fix they suggest: **combine CFD with sampling** (i.e., MAP-Elites outer loop) to handle the non-convex problem of getting fingers *around* the object, then use CFD+gradient for the final closing. This aligns naturally with MorphoHand's pipeline:
- MAP-Elites samples diverse morphologies (handles non-convexity)
- Inner-loop gradient with CFD fine-tunes joint angles for contact quality once fingers are near the object

### Tractability: ★★★★ (Moderate)

The straight-through trick itself is ~15 lines of JAX. The main effort is understanding how to patch MJX's internal impedance computation. Estimated effort: **1–3 weeks**. The CFD parameter tuning adds additional experimental iteration.

---

## Modification 3: Adaptive Integration via Diffrax

This is the hardest modification. The paper notes they "devoted substantial effort to seamlessly integrating quaternions and stateful actuators."

### What Diffrax needs

Diffrax expects an ODE in the form:
```
x_dot(t) = F(t, x, args)
```
where `x` is the continuous state and `F` is the vector field. You feed this to:
```python
from diffrax import diffeqsolve, ODETerm, Tsit5, PIDController, RecursiveCheckpointAdjoint

solution = diffeqsolve(
    ODETerm(F),
    solver=Tsit5(),
    t0=0.0, t1=h,  # integrate one MuJoCo step
    dt0=h/10,      # initial substep size
    y0=x0,
    stepsize_controller=PIDController(rtol=1e-4, atol=1e-6, pcoeff=0.2, icoeff=0.4, dcoeff=0.0),
    adjoint=RecursiveCheckpointAdjoint(checkpoints=32),
    max_steps=1000,
)
x_next = solution.ys[-1]
```

### Step 1: Express MJX's dynamics as a continuous ODE

MuJoCo's physics model is continuous: `mj_forward` computes the acceleration `v_dot` from `(q, v, u, params)`. The state is `x = (q, v)` (generalized position and velocity, plus actuator states `w`).

```python
def mjx_ode(t: float, x: tuple, args: tuple) -> tuple:
    """
    Continuous-time ODE for MuJoCo dynamics.
    Returns (q_dot, v_dot) = (v, M^{-1}(tau - c + J^T f))
    """
    q, v, act = x           # generalized pos, vel, actuator states
    model, u = args         # model parameters and control input

    # Create a temporary MjData with this state
    data = mjx.make_data(model)
    data = data.replace(qpos=q, qvel=v, act=act, ctrl=u, time=t)

    # Run forward (kinematics + contact + dynamics, but NOT integration)
    data = mjx.forward(model, data)

    # Extract accelerations
    q_dot = data.qvel      # q_dot = v (for standard DOFs)
    v_dot = data.qacc      # v_dot = M^{-1}(tau - c + J^T f)
    act_dot = data.act_dot  # actuator state derivatives

    return (q_dot, v_dot, act_dot)
```

**Critical**: use `mjx.forward` (computes forces and accelerations) not `mjx.step` (which also integrates). DiffMJX wraps `mjx.forward` as the ODE vector field.

### Step 2: Quaternion integration — the hard part

MuJoCo represents free body orientations as unit quaternions. The standard Runge-Kutta integrators operate on Euclidean vectors and do not automatically preserve the unit-norm constraint.

The angular velocity vector in `qvel` for a free joint is the body-frame angular velocity `omega`. The correct quaternion derivative is:
```
q_dot = 0.5 * q ⊗ [0, omega]  (quaternion product, omega as pure quaternion)
```

MJX's semi-implicit Euler uses `mj_integratePos` which integrates orientation in the Lie algebra (axis-angle), then maps back to the quaternion via the exponential map. For Diffrax you need to:

1. Express `q_dot` for free-joint quaternion slots correctly
2. Renormalize the quaternion after each RK stage

```python
def integrate_qpos(qpos: jnp.ndarray, qvel: jnp.ndarray, jnt_types: jnp.ndarray, h: float):
    """
    Integrate generalized position accounting for quaternion DOFs.
    For standard joints: q_next = q + h * v
    For free joints (quaternion): use Lie algebra integration
    """
    # ... (requires per-joint-type dispatch based on jnt_types)
    # Free joints (jnt_type == mjJNT_FREE):
    #   pos_next = pos + h * v[:3]  (translation)
    #   quat = qpos[3:7]
    #   omega = qvel[3:6]
    #   axis_angle = h * omega
    #   dquat = axis_angle_to_quat(axis_angle)
    #   quat_next = quat_multiply(quat, dquat)
    #   quat_next = quat_next / jnp.linalg.norm(quat_next)  # renormalize
```

The paper says they adapted Diffrax's solvers specifically for this. The Tsit5 solver computes multiple intermediate RK stages; each stage that steps through orientation must use the Lie algebra path, not naive Euclidean addition. This is non-trivial because Diffrax's generic RK framework assumes Euclidean state.

**Practical approach**: implement a custom Diffrax `AbstractTerm` that overrides `contr` (the "control term" that contracts the vector field with the differential) to use the quaternion-aware position integrator for the relevant state slots.

### Step 3: Stateful actuators

MuJoCo has actuators with internal state (position servos, muscles). Their dynamics are:
```
act_dot = (u - act) / tau   (for a first-order position actuator)
```
This is already handled by `mjx.forward` returning `act_dot`, so the ODE includes `act` as part of the state. No special treatment needed beyond including `act` in `x`.

### Step 4: Connect to Diffrax

```python
import diffrax

class MJXDynamics(diffrax.AbstractTerm):
    model: mjx.Model
    ctrl: jnp.ndarray

    def vf(self, t, y, args):
        q, v, act = y
        return mjx_ode(t, (q, v, act), (self.model, self.ctrl))

    def contr(self, t0, t1):
        # Override for quaternion-aware integration
        return t1 - t0  # dt (Euclidean; quaternion correction happens in vf)

def diffmjx_step(model: mjx.Model, data: mjx.Data, rtol=1e-4, atol=1e-6) -> mjx.Data:
    y0 = (data.qpos, data.qvel, data.act)
    term = MJXDynamics(model=model, ctrl=data.ctrl)

    solution = diffrax.diffeqsolve(
        term,
        solver=diffrax.Tsit5(),
        t0=data.time,
        t1=data.time + model.opt.timestep,
        dt0=model.opt.timestep / 10.0,
        y0=y0,
        stepsize_controller=diffrax.PIDController(
            rtol=rtol, atol=atol,
            pcoeff=0.2, icoeff=0.4, dcoeff=0.0
        ),
        adjoint=diffrax.RecursiveCheckpointAdjoint(checkpoints=32),
        max_steps=4096,
        throw=False,
    )

    q_next, v_next, act_next = solution.ys
    return data.replace(
        time=data.time + model.opt.timestep,
        qpos=q_next[-1],
        qvel=v_next[-1],
        act=act_next[-1],
    )
```

### Tuning adaptive integration (from the paper's Appendix B)

```
Error tolerance:     Most important hyperparameter.
                     rtol=1e-4, atol=1e-6 is a good starting point.
                     Tighten if gradients are still noisy.
                     Loosen to speed up (accept some gradient error).

Checkpoints:         Set as high as GPU memory allows.
                     Memory ≈ n_checkpoints × state_size × dtype_bytes
                     32–128 checkpoints is typically practical.

max_steps:           Set conservatively large (4096).
                     Compilation time scales with max_steps (JIT compiles
                     for max_steps iterations), so don't go to 100k.

Recommended solver:  Tsit5 (5th order, Tsitouras 2011 coefficients).
                     More accurate than Dopri5 for the same cost on
                     non-stiff physics ODEs.

PID controller:      pcoeff=0.2, icoeff=0.4, dcoeff=0.0
                     (Recommended by Diffrax docs for stiff ODEs)
```

### Parallelism note

When vmapping across environments, Diffrax's adaptive stepsize is **independent per environment** (the paper verifies this in Figure 20). A stiff contact in environment 7 does not slow down environment 3. This is important for MAP-Elites batch evaluation.

### Tractability: ★★ (Hard)

The quaternion integration is the genuinely hard part. Estimated effort: **4–8 weeks** for a robust implementation. Can be deferred — start with smooth collision detection + CFD, which solves the two most critical problems for MorphoHand's quasi-static regime.

---

## Priority Order for MorphoHand

### Why quasi-static matters

MorphoHand evaluates grasps under quasi-static conditions (slow finger closure, wait for stability). The error sources rank differently vs. dynamic tasks:

| Error source | Dynamic task (e.g., ball toss) | Quasi-static grasp | 
|---|---|---|
| Discretization at impact | Critical | **Less critical** (low contact velocity) |
| Zero gradient before contact | Critical | **Critical** (fingers start far from object) |
| Collision detection discontinuities | Moderate | **Moderate** (still causes gradient kinks) |

The practical ordering:

```
Phase 1 (Week 1–2): Smooth collision detection
  → Fixes kinks in all gradient-based optimization
  → Self-contained, testable, no algorithmic changes
  → Apply to: plane-box, plane-capsule, sphere-capsule, capsule-capsule, plane-cylinder

Phase 2 (Week 2–4): CFD with straight-through trick
  → Fixes zero-gradient before contact (critical for pre-contact approach)
  → Use with: MAP-Elites inner-loop gradient ascent on q
  → Tune wc=0.05m, d0=0.02 as starting point
  → Combine with sampling for global approach (CFD alone won't handle non-convexity)

Phase 3 (Optional, Week 5–10): Adaptive integration
  → Improves gradient accuracy at contact
  → Lower priority for quasi-static grasps with soft contact settings
  → Consider if you observe oscillating gradient descent in inner loop
  → Can be partially substituted by using softer solimp settings
```

---

## Testing and Validation

For each modification, validate against central differences:

```python
def check_gradients(loss_fn, params, eps=1e-5):
    """Compare autodiff gradients to central differences."""
    grad_auto = jax.grad(loss_fn)(params)
    
    grad_fd = jnp.zeros_like(params)
    for i in range(params.size):
        p_plus = params.at[i].set(params[i] + eps)
        p_minus = params.at[i].set(params[i] - eps)
        grad_fd = grad_fd.at[i].set((loss_fn(p_plus) - loss_fn(p_minus)) / (2 * eps))
    
    return jnp.max(jnp.abs(grad_auto - grad_fd))
```

**Suggested test sequence:**

1. **Bounce test** (from paper, Figure 3): Toss a box/capsule/cylinder at a plane, measure gradient of final height w.r.t. initial velocity. Should match central differences after smooth collision detection fix.

2. **Billiard test** (from paper, Figure 8): Ball A pushed toward ball B. Measure gradient of B's final position w.r.t. A's initial velocity. Should be nonzero even when A doesn't quite reach B, after CFD.

3. **Finger-approach test** (MorphoHand-specific): Place one finger 50mm from an object, measure gradient of GWS epsilon w.r.t. finger base position. Should be nonzero and correctly directed before contact.

4. **Full grasp synthesis test**: Start from a random morphology with fingers 30mm from a target sphere, run gradient ascent on joint angles `q`. Should converge to a force-closure grasp. Without smooth collision detection + CFD, this will stall.

---

## Key Files in MJX Source to Modify

All paths relative to the MuJoCo repo (`google-deepmind/mujoco`):

```
mjx/mujoco/mjx/_src/
├── constraint.py          ← Contact force computation (aref, impedance d(r))
│                            → ADD: CFD impedance extension
│                            → ADD: softplus for h(r)
├── collision_driver.py    ← Dispatches to geometry-specific collision functions
│                            → MODIFY: add smooth variants, swap in smoothed versions
├── collision_convex.py    ← plane-box, plane-capsule, plane-convex
│                            → MODIFY: _plane_box, _plane_capsule
├── collision_primitive.py ← sphere-capsule, capsule-capsule, plane-cylinder  
│                            → MODIFY: affected functions
├── smooth.py              ← Forward dynamics (fwdPosition, fwdVelocity, etc.)
│                            → READ: understand mj_forward structure for ODE wrapping
├── step.py                ← The step() function and integrators (euler, rk4, implicitfast)
│                            → ADD: diffmjx_step() using Diffrax
│                            → ADD: step_cfd() with straight-through trick
└── math.py                ← Utility functions
                             → ADD: sigmoid(), soft_where() helpers
```

---

## Dependency Setup

```bash
pip install mujoco-mjx          # MJX (JAX implementation)
pip install diffrax             # Adaptive ODE integrators (Phase 3)
pip install equinox             # Required by Diffrax
# All already JAX-compatible; no additional GPU setup needed beyond JAX
```

---

## Summary

| Component | Lines of code (est.) | Main challenge |
|---|---|---|
| Smooth collision detection | ~100 (5 geometry pairs × ~20 lines) | Reading MJX source to find all discontinuities |
| CFD impedance extension | ~50 | Matching d0 to solimp at r=0 for smooth join |
| Straight-through trick | ~15 | Conceptually simple; patching MJX model harder |
| Diffrax ODE wrapping | ~200 | Quaternion-aware integration |
| Quaternion handling | ~150 | Non-Euclidean state manifold in RK |
| Total | ~500 | — |

The code volume is modest; the difficulty is in navigating MJX's internal structure and correctly handling quaternion kinematics. For MorphoHand, Phases 1 and 2 (smooth collisions + CFD) are the high-value, tractable targets. Phase 3 (adaptive integration) can be deferred or substituted by using softer `solref`/`solimp` settings that reduce contact stiffness at the cost of some sim-to-real fidelity.

---

## References

- Paulus, Geist et al. (2026). *Differentiable Simulation of Hard Contacts with Soft Gradients for Learning and Control.* arXiv:2506.14186. https://github.com/martius-lab/diffmjx *(code not yet public)*
- Kidger (2021). *Diffrax: Numerical differential equation solvers in JAX.* https://github.com/patrick-kidger/diffrax
- MJX source: https://github.com/google-deepmind/mujoco/tree/main/mjx
