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

### ⚠ CORRECTION (2026-08-01): the yaw joint is fine. The CALIBRATION POSE is not.

An earlier revision of this file claimed "our yaw is a poor lateral joint" and proposed
adding a true abduction DOF. **That was wrong, and no hardware change is warranted.**
It was inferred from the CIK θ-range table without ever measuring what the joint does.

Measured directly (`scripts/uhas_lateral_authority.py`), sweeping `yaw` and decomposing
fingertip motion in the palm frame — `lat` = in-palm-plane lateral (abduction), `oop` =
out-of-plane (parasitic), `eff` = fraction of motion that is lateral:

| pose | lat mm/rad | oop mm | eff |
|---|---|---|---|
| q = 0 (fully extended) | **0.0** | 0.0 | **0.00** |
| open keyframe, local ±0.2 rad | **108–123** | 2.2–2.5 | **0.99** |

**eff = 0.99 on every finger of every hand** (baseline, m05, H06_04, perp). At the pose the
hand actually operates at, `yaw` is a near-ideal abduction joint. It degenerates to a pure
roll at exactly one pose — q = 0, full extension — because the yaw axis and the finger are
colinear there. Nothing operates at q = 0. Rendered side by side in
`figs/lateral_sweep_H0604.png` (`scripts/uhas_render_lateral.py`): at q=0 the three frames
are identical and the tips travel 0 mm; at the open keyframe the same sweep moves them
215/214/221 mm across the palm.

**Why the θ table looked pathological.** Two compounding facts, both in UHAS:

1. `process_type_ABC_joints` classifies a joint **A** (lateral) if the fingertip normal is
   parallel to the joint axis, **C** (roll) if the fingertip *lies on* the axis. At q = 0
   our tip sits on the yaw axis, so all three yaws classify **C**, not A. (The earlier note
   here that "all three `*_yaw` classify type A" was a misreading of the log — the log says
   `'type': 'C'`, `'ft': [0, 0, 0.131]`, i.e. the tip is exactly on the axis.)
2. For a type-C joint, `compute_main_joints` measures θ range at the **IK-solved maximally
   flexed pose**, which is bounded by the URDF joint limit. Our thumb_mcp upper is 3.14 rad
   — folded fully back on itself, tip near the yaw axis again → θ range 0.17. Index/middle
   at 2.5 rad swing the tip across the sphere's pole, where azimuth wraps → ±2.8.

So the offsets are measured at two useless poses and the spread between fingers is an
artifact of *where the probe pose lands*, not of the joint. Sweeping the mcp upper limit
confirms the mechanism — the offsets are a smooth function of the probe pose:

| mcp upper (rad) | 1.0 | 1.3 | 1.6 | 1.9 | 2.2 | 2.5 |
|---|---|---|---|---|---|---|
| span ratio max/min | **1.5** | 1.7 | 1.9 | 2.3 | 3.7 | **6.6** |

At mcp_upper ≈ 1.0 our slot-to-slot consistency (1.5×) is *better than LEAP's* (1.7×).

**This is not a free fix.** The URDF limit also bounds what CIK may command at runtime, and
our thumb genuinely operates at mcp = 2.0 (`open` keyframe). Clamping to 1.0–1.3 would buy
uniform offsets by forbidding the hand's real working range. The clean fix decouples the
two — calibrate the lateral range at a sensible probe pose while leaving the actuation limit
alone — which needs a small change to `compute_main_joints`, not to the hand. **Open.**

---

### The perp topology exports cleanly now

Previously recorded as "degenerate — needs a bespoke open pose before it can be exported"
(at q=0 the fingers point at each other, tips converge, r collapses to 0.0450). The `open`
keyframe now in `perp_hand_morphology_actuated.xml` is that pose, and
`--open-from-keyframe open` uses it:

| | chain_indices | dead dims | sphere r |
|---|---|---|---|
| perp @ q=0 | *degenerate* | — | 0.0450 |
| perp @ `open` keyframe | `[[4],[2],[0,1]]` | **2 / 15** | **0.0685** (0.75× LEAP) |

That is the canonical 3-finger mapping, on par with m05 (0.0687) and H06_04 (0.0710).
Note the root body is `palm_pose`, not `palm` (`--palm-body palm_pose`).

**thumb_x toward the pair** (`--morph thumb_x=…`, sweeping its full −0.0075…0.0525 range):
the UHAS representation is *indifferent* — 3-finger spread and 2 dead dims hold throughout;
only the sphere shrinks ~7% (0.0693 → 0.0643) as the thumb closes in. So it neither helps
nor breaks UHAS. It does, however, fail the grasp gate: `morph_selfcollision_gate.py
--retarget` returns OK at thumb_x=0 (pinch 34.9 mm, thumb→pair 71.4 mm) but **UNREACHABLE**
at +0.030 and +0.0525 (IK residual 6.1 / 4.4 mm). ⚠ Read that with care — the gate scores
every design against the *reference grasp's* fingertip world targets, which were authored
for a thumb 65 mm out. A thumb moved 30 mm inward should be grasping a differently-placed
object, so this says "cannot do the old grasp", not "cannot grasp". Not self-collision this
time; the palm is innocent again.

---

## State / what is not done

Done: offline pipeline, exporter, LEAP validation, three morphologies + perp built and
rendered, the lateral-authority correction above, **and the pretrained policy running in
Isaac**.

### The pretrained checkpoint is `All_grippers/model_14999.pt`

There is no file named `multi_hand_policy.pt` in the Box download. The multi-hand policy is
`models/UGAS_Models/**All_grippers**/model_14999.pt` — its `params/env.yaml` lists
`robots: [shadow, allegro, modified_mano, leap]`, `num_robot_types: 4`, action 15, and the
actor is 98→…→15. The 25 sibling dirs are ablations and baselines; `*_OOD` are leave-one-out
(e.g. `MANO_OOD` trains on the other three). For a hand unseen by *all* of them, `All_grippers`
is the right one — most training diversity. `All_grippers_RW` is the real-world variant
(asymmetric, actor 61 / critic 142) and is **not** the sim policy.

### Isaac Lab 2.2.1 is NOT needed — 2.1.0 / Isaac Sim 4.5.0 runs it

UHAS's README asks for Isaac Lab 2.2.1, but 2.2.1 targets **Isaac Sim 5.0.0**, not the 4.5.0
the README pairs it with — so "just bump the version" is really a multi-GB Isaac Sim upgrade
that would disturb the working `env_isaaclab`. It is not necessary. The pretrained policy
loads and steps on the installed 2.1.0 with five documented patches, all marked
`MORPHOHAND PATCH` in the vendored tree:

1. `grippers/allegro/.../allegro_right.py` and `multi_env_cfg.py` — drop `dynamic_friction`
   (a 2.2 field; 2.2 split joint friction into static/dynamic). Both values are 0.02, so
   2.1's single `friction` reproduces the intent **exactly**.
2. `tasks/utils/dr_funcs.py` — `isaaclab.utils.version.compare_versions` is 2.2-only and is
   a *dead import* in UHAS; fall back to the verbatim upstream body.
3. `tasks/utils/dr_funcs.py` — `_validate_scale_range` is a 2.2 private helper; it is pure
   input validation (raises on a malformed range, no physics), backported verbatim.
4. `multi_env_cfg.py` — `vector_phis` 60.0 → **50.0** to match the checkpoint's env.yaml.
   These angles define what an action *means* as a sphere deformation; a mismatch silently
   misinterprets every action of a pretrained policy.
5. `multi_env_cfg.py` / `multi_manipulation_env.py` — `tip_rot`/`angvel_obs` off, plus new
   `ff_flag_input` and `symmetric_critic` flags.

That last one is the interesting one: **the released code is a different vintage than the
released checkpoints.** Released defaults give a 142-dim actor obs; the checkpoints are 98.
Turning off `tip_rot` and `angvel_obs` (per the checkpoint's env.yaml) gets to 97 — one
short. The missing dim is `ff_flag_input`, a 1-wide "is a finger disabled" flag whose whole
feature (`finger_failures`, `fail_frequency`) the released env dropped. It is constant 0
whenever no finger is failed, which is its value at eval (`force_failure: 0`), so a zero
column restores it faithfully. `symmetric_critic` handles the checkpoints having
critic.0 == actor.0 == 98 (no privileged asymmetry); it affects loading and value estimates
only, never the actions taken.

Environment fixes, in `env_isaaclab`: `pip install scipy matplotlib scikit-learn einops
transformations` + the `tf` shim copied from `.venv-uhas` (ROS xyzw ordering — see above,
getting it wrong is silent), and `pip install -e sphere_ctrl_isaaclab/source/...`.

```bash
cd docs/uhas/UHAS_sim/sphere_ctrl_isaaclab/scripts/rsl_rl
/home/humanoid/miniconda3/envs/env_isaaclab/bin/python play.py \
    --task UHAS-Inhand-Repose --headless --num_envs 16 \
    --checkpoint ../../../models/UGAS_Models/All_grippers/model_14999.pt
```

Verified: env builds all four hands, checkpoint loads with no size mismatch, actor
`98→512→512→256→128→15`, and it steps without crashing. **Not yet measured: whether it
actually reposes the cube** (success rate / consecutive reorientations). That is the next
thing to run, and it is now a matter of reading numbers off a run, not of infrastructure.

Not done:
1. **Zero-shot on OUR hand** — needs URDF→USD (Isaac Sim GUI import per `docs/add_hand.md`)
   and a `<hand>_right.py` `ArticulationCfg` + registration in `multi_env_cfg.py`. All five
   shipped hands already have USDs, which is why LEAP et al. ran without any conversion.
2. **A success-rate number for the pretrained policy**, on LEAP first (reproducing the
   paper) and then on ours.
3. Decoupling the CIK lateral-range probe pose from the actuation limit (see the correction
   above).
4. A quantitative **sphere-reachability score** — CIK-solve a battery of deformations and
   measure fingertip-to-target error.
5. An `isaac-eyes` skill (the offline analogue, figure capture, exists).
