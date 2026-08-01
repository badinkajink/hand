# UHAS ↔ MorphoHand integration log

**UHAS** = *Cross-Embodiment Robot Manipulation via a Unified Hand Action Space* (Casas, Teal,
Shah, Tadepalli, Jin, Xiang; arXiv 2607.03570). One RL policy drives any dexterous hand by
acting in a canonical **sphere deformation** space; a **Cascade Inverse Kinematics (CIK)**
solver maps sphere deformations to that hand's joints.

Branch `experiment/uhas`. Paper source and the cloned repo live under `docs/uhas/`
(gitignored); everything tracked is in `src/morphohand/uhas/` and `scripts/uhas_*.py`.

---

## Why it fits us (established, not assumed)

The concern was that UHAS is built around 4- and 5-finger hands and we have three. It is not:

* `multi_env_cfg.py` sets `action_space = (1 + len(vector_phis)) * max_fingers = 3 x 5 = 15`
  and `observation_space = 100`, both **fixed**. Finger count does not appear in the
  policy's dimensions.
* Hands with fewer than 5 fingers ride on **`chain_indices`**, which maps physical fingers
  onto the 5 canonical driving planes. Slots that share a finger get their actions
  **averaged** (`multi_manipulation_env.py:669-689`). LEAP's 4 fingers come out
  `[[3],[2],[0,1],[4]]`.
* CIK requires every finger to own **one lateral joint plus a chain of encompassing
  joints** — `process_urdf.py:298` raises outright otherwise. Our `yaw` + `mcp` + `pip` is
  exactly that, and UHAS confirms it empirically: all three `*_yaw` classify as type **A**
  (lateral), all `*_mcp`/`*_pip` as type **B** (encompassing).

**No UHAS code change is needed for a 3-finger hand.**

## What runs where

`process_urdf.py` — the sphere/CIK builder, and the thing that answers "can this hand
reach and navigate the sphere" — is **pure URDF + numpy/trimesh and needs no Isaac at
all**. Only the RL env and policy rollout need Isaac Sim.

* Offline half: `.venv-uhas` (python 3.10, `uv venv`). Deps: trimesh, urdf-parser-py,
  pyvista, h5py, rtree, matplotlib, transformations, mujoco.
* Isaac half: conda env **`env_isaaclab`** — Isaac Sim 4.5.0, isaaclab 0.40.5 (= IsaacLab
  repo v2.1.0), rsl-rl-lib 2.3.3. Its torch was broken (missing `libnvJitLink.so.12`);
  fixed with `pip install nvidia-nvjitlink-cu12` → torch 2.5.1+cu121, CUDA available.
  **Not** a uv env — our `.venv` has *mjlab*, which is the Isaac Lab *API* on MuJoCo Warp,
  a different thing.

### The `tf.transformations` shim (load-bearing)

UHAS imports ROS's `tf.transformations`. Outside ROS the equivalent is Gohlke's
`transformations`, which differs in exactly one way: **ROS orders quaternions xyzw, Gohlke
wxyz**. The shim in `.venv-uhas/.../site-packages/tf/` re-orders the quaternion functions
to ROS convention.

This is not cosmetic. The Isaac env reads the sphere frame back as `quat[[3, 0, 1, 2]]`
(`multi_manipulation_env.py:384`), an explicit xyzw→wxyz conversion. A wxyz file yields a
plausible-looking, wrong rotation — silent, not loud.

### Pipeline validated against LEAP

Regenerating LEAP's `sphere_cik.json` from its URDF reproduces the shipped artifact:

| field | ours | shipped |
|---|---|---|
| `chain_order` / `chain_indices` | `[3,2,0,4]` / `[[3],[2],[0,1],[4]]` | identical |
| sphere radius | 0.091192 | 0.091192 |
| sphere xyz / quat | max diff 8e-17 / 1.3e-13 | — |
| palm_normal xyz / quat | max diff 7e-18 / 3e-16 | — |

The released `process_urdf.py` omits keys the shipped LEAP/Allegro/Shadow/MANO files carry
(`phi_roots`, `skeleton_*`, `viz_dict`, `q_max_phi`) — those came from a newer internal
version. **The env reads none of them**, nor even `plane_anchors`; `wuji`, the official
bring-your-own-hand example, has the same reduced key set our output does. Not a problem.

---

## Exporting a MorphoHand

`src/morphohand/uhas/mjcf_to_urdf.py`. Three URDF conventions are load-bearing and every
one of them fails silently:

1. a fixed joint literally named **`palm_normal`** whose +Z is the direction the fingers
   close (ours = π about X, since MCP axis +Y curls the fingers toward palm −Z);
2. one fixed joint per finger named **`<finger>_ft`**, +Z along the fingertip normal;
3. **mesh geometry only** — `load_link_meshes` handles `urdf_parser_py.urdf.Mesh` and
   ignores primitive tags, so our capsules/spheres are baked to STL.

Two choices worth restating: the URDF is written at **hinge = 0** (a URDF joint origin is
by definition the transform at joint value zero) with the open pose carried in
`config.json`, and the **morphology slides are baked** — an articulated morphology would
drift during rollout, the same frozen-scene rule as everywhere else in this repo.

Verified against MuJoCo FK at a non-trivial pose: worst tip error **4e-17 m**, and at the
open pose all three fingertip +Z have `dot(palm_normal) = +1.000000`, satisfying UHAS's
"fingertip normals align with the palm normal" exactly rather than approximately.

```bash
.venv-uhas/bin/python scripts/uhas_build_hand.py \
    --mjcf results/uhas/mjcf/hand_H0604_....xml --out results/uhas/hands/H0604 --figures
```

`scripts/uhas_process_urdf.py` runs the builder headless and **captures its verbose
figures**, which upstream throws at a screen that does not exist. The sphere construction
is only checkable by looking at it.

---

## Results so far

### Open-pose geometry decides the topology

| topology | pose | l (palm→tip) | r = 2l/π | verdict |
|---|---|---|---|---|
| baseline | **q = 0** | 0.1231 | 0.0783 | textbook UHAS open pose |
| baseline | keyframe | 0.1163 | 0.0740 | fingers already curled |
| perp | q = 0 | 0.0707 | 0.0450 | **degenerate** — tips converge inward |
| perp | keyframe | 0.1148 | 0.0731 | usable, but not "extended" |

The **baseline** topology at q=0 is what UHAS assumes. The **perp** topology points its
fingers at each other, so full extension collapses the tip spread and the sphere radius
estimate with it — it needs a bespoke open pose before it can be exported.

### Three morphologies through the pipeline

| hand | `chain_indices` | dead slots | dead action dims | sphere r | r / LEAP |
|---|---|---|---|---|---|
| baseline | `[[4],[3],[2]]` | 0, 1 | **6 / 15** | 0.0654 | 0.72 |
| m05 | `[[0,1],[4],[2]]` | 3 | **2 / 15** | 0.0687 | 0.75 |
| H06_04 | `[[4],[0,1],[2]]` | 3 | **2 / 15** | 0.0710 | 0.78 |

**The morphology co-design independently improved UHAS compatibility.** The co-designed
hands land on the canonical 3-finger slot mapping and drive 13 of 15 policy action
dimensions; the baseline's fingers cluster in azimuth (thumb and middle sit 2 mm apart in
y — the known 2-finger degeneracy) and it drives only 9.

Figures confirm it: for the baseline the constructed sphere sits *beside* the fingertip
markers, because the alignment step centres it on the median finger and the index is 6 cm
off alone. For H06_04 the sphere **contains** the fingertips, the flexed hand shows real
3-finger opposition, and the fingerprint fan resolves into three separated regions.

### ⚠ The real problem: our yaw is a poor lateral joint

Per-finger lateral θ range from the CIK lookup table (LEAP, for scale, is ~±0.55 on **all
five** planes — uniform and physical):

| hand | thumb | index | middle |
|---|---|---|---|
| baseline | **0.19** | 2.88 | **±2.80** |
| m05 | **0.17** | 3.60 | **±2.74** |
| H06_04 | **0.17** | 2.62 | **±2.81** |

Our `yaw` axis runs **along the finger's own longitudinal axis**, so it is a roll, not an
abduction. At full extension it moves the fingertip not at all; when the finger is curled
it sweeps the tip around a cone whose azimuthal effect is wildly non-uniform — near zero
for the thumb (0.17 rad) and near-2π for the middle finger, whose tip crosses close to the
sphere's pole where θ wraps.

Since each slot's action ∈ [−1,1] is rescaled by that slot's own min/max offsets, a shared
policy's lateral command means something completely different on each of our fingers. The
representation *builds*; **lateral (Δθ) control is predicted to be erratic**. This is the
main technical risk to zero-shot transfer, and it points at a concrete design change: give
the fingers a true abduction DOF (axis perpendicular to the finger, in the palm plane)
rather than re-aiming a roll.

---

## State / what is not done

Done: offline pipeline, exporter, LEAP validation, three morphologies built + rendered.

Not done:
1. **Pretrained checkpoint.** `multi_hand_policy.pt` is **not in the repo** — `**/models/*`
   is gitignored and it is Box-only
   (`utdallas.box.com/s/qq14yjwqzouv4a3dj95c3a5nght6g47j`). Zero-shot transfer is blocked
   on a manual download.
2. **Isaac version gap.** UHAS asks for Isaac Lab **2.2.1**; installed is **2.1.0**
   (isaaclab 0.40.5). Untested whether UHAS's env imports against 2.1.0.
3. URDF→USD conversion and wiring the hand into `multi_env_cfg.py`.
4. A quantitative **sphere-reachability score** — CIK-solve a battery of deformations and
   measure fingertip-to-target error. The lateral-span table above is the qualitative
   version of this.
5. An `isaac-eyes` skill (the offline analogue, figure capture, exists).
