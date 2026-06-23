# GraspGenX on the morphohand

Feasibility + evaluation of [NVlabs GraspGenX](https://github.com/NVlabs/GraspGenX)
(zero-shot 6-DOF grasp generation for arbitrary grippers) on our parametric
3-finger morphohand. Branch `graspgenx-morphology-eval`.

GraspGenX runs zero-shot from a URDF + `config.json`; it has a native
`revolute_3f` (3-finger) class. We author the config head-less (no Viser
wizard) and run inference + a MuJoCo lift test.

## Scripts (`scripts/`)
- `graspgenx_make_morphohand.py` — build the morphohand URDF + GraspGenX
  `config.json` (pocket-centered sweep boxes from the fingertip arc). Takes
  `--thumb/--index/--middle X Y Z` + `--baked-len` to sweep morphologies.
- `graspgenx_view_grasps.py` — place the gripper mesh at the top-K grasps,
  export `.glb` (+ PNG).
- `graspgenx_lift_eval.py` — execute each grasp on the real MuJoCo hand
  (mocap-free kinematic palm, extend→close→lift) and report hold rate.

## Renders (`renders/`)
- `banana_v1_fingertip_boxes` vs `banana_v2_pocket_boxes` — sweep-volume
  refinement: objects move from the fingertips into the grasp pocket.
- `task_{cube,prism,screwdriver}` — grasps on our real task meshes.
- `morph_{mwide,mopp,mlong}_cube` — three distinct morphologies on the cube
  (wide splay / z-raised opposed thumb / long fingers).
- `lift_cube_g0_DROP` vs `lift_cube_g9_HOLD` — MuJoCo lift strips: most grasps
  flick the cube away (non-opposed fingers can't cage it); a few "scoop" holds
  ride the object up.

## Key results
- Generation works: 50 grasps/object in ~0.5 s, conf 0.65–0.99.
- **Lift hold rate on the baseline (non-opposed) hand is low**: cube 2/10,
  prism 1/10, screwdriver 1/10 (top-10 grasps). Most fail because the three
  fingers curl the same way and bat the object out instead of opposing it.
- This is the structural "degenerate pinch" limitation, and the natural place
  an *opposed* morphology should help — but the current MJCF morph joints
  (x/y/len, no Z, ±0.03 range) can't express the z-raised / wide / long
  designs, so on-hand lift eval of those needs a hand-model extension.
