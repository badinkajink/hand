# What the bench should run: g12, and only g12

2026-08-30. `scripts/real_v1_render_deploy_plan.py`, `scripts/real_v1_trajectory_clearance.py`.
Renders and per-step traces in this folder; `RENDERS.json` is the index.

Everything visual we had of the real_v1 reorient was of the **dense carry** that
`probe_real_v1_carry.py` simulates. That is not what the bench runs. The control station
replays the **exported plan** — three set-points it interpolates, plus a CSV whose per-joint
timing differs from that chord. Three different paths, and until now only the first had ever
been rendered or physically simulated.

## The exported plans do reorient the tool

Stepping each `<design>_traj.csv` in its own deploy scene, from the plan's own
`replay_initial_qpos` / `replay_base_ctrl` so the CSV drives nothing but the fingers:

| design | final cos | object z | object contacts |
|---|---:|---:|---:|
| g12 | +0.780 | 111 mm | 3 |
| g23 | +0.737 | 116 mm | 3 |
| g24 | +0.779 | 116 mm | **2** |
| rv04_mid | +0.781 | 116 mm | 3 |

All four carry the shaft off its support, turn it, and hold. On g12 the trace reads horizontal
at t=0 (cos +0.052, resting on the post), clear of the post by t≈1.3 s, turning through
t=1.76–2.64 s, then **flat at cos 0.775 → 0.780 with z 113 → 111 mm on three contacts for the
final 1.3 s**. It is held, not falling, and nowhere near the floor.

cos 0.78 is about 39° off vertical, not vertical. The plans command −70°, not −90°. This is a
substantial *partial* reorientation that is stably held, and it should be described that way.

## The reorientation is not what separates the designs — the self-collision is

| design | chord | csv | physics cos | verdict |
|---|---:|---:|---:|---|
| **g12** | **+8.7 mm** | **+8.8 mm** | +0.780 | **RUN** |
| g23 | +0.8 mm | +0.8 mm | +0.737 | no safe path |
| g24 | −5.3 mm | −4.2 mm | +0.779 | fingers interpenetrate |
| rv04_mid | −2.7 mm | −2.0 mm | +0.781 | fingers interpenetrate |

Task performance is indistinguishable across all four (0.737–0.781). **Clearance decides
everything.** g12 is the only design whose commanded path keeps its fingers apart on both the
chord and the CSV, and it is independently the best entry in the deploy catalog
(careful-bench win 0.65, kept 0.84 — the highest of the four on both).

g23's +0.8 mm is not a pass. It is below the 5 mm bar on sim geometry whose link capsules and
tip boxes are *thinner than the printed parts*, so the real margin is smaller than the number.

## Why the dense carry could not have told us this

The clearance trace on the dense carry is clean for every design (28.9–38.8 mm, measured with
the new `probe_real_v1_carry.py --selfcollision`). The collisions exist only on the exported
paths. A gate run on the source trajectory would have passed all four and sent two
finger-crashing plans to the bench.

## Before the bench session

1. **Run `real_v1_trajectory_clearance.py` on any plan before executing it.** It is the
   authoritative gate and it takes seconds. A positive number there is necessary, not
   sufficient — sim geometry is thinner than the printed parts.
2. **Use g12.** Set the gantry blocks to thumb (−42.5, 0.0), index (42.5, 40.0), middle
   (42.5, −40.0) mm in the palm frame, per `deploy/g12_build.txt`.
3. **Mass is not a constraint for either object.** Swept through g12's own exported CSV:

   | mass | final cos | object z | contacts |
   |---:|---:|---:|---:|
   | 24 g (the bench cylinder) | +0.779 | 112 mm | 3 |
   | 50 g | +0.786 | 109 mm | 3 |
   | 65 g (the intended screwdriver) | +0.788 | 108 mm | 2-3 |
   | 85 g | +0.790 | 107 mm | 2 |

   Flat across the range, no cliff. **This contradicts the dense carry's envelope**
   (`20260830-carry_mass_envelope/`), which found rv05_manual dropping between 65 and 70 g,
   and the disagreement is the point: that envelope was measured on a different design with
   the CEM grip, while g12 clamps 10 mm of pad squeeze. Mass tolerance is a property of the
   deployed plan, not of the hand, so it has to be swept on the plan being run.

   The contact count flickering 2-3 is a single-instant `d.ncon` read, not a lost contact
   (mean 2.77 over the final 0.8 s at 65 g, alignment flat at +0.7873 -> +0.7884).

   The real number to carry to the bench: at 65 g the shaft **sinks 2.2 mm over the final
   0.8 s** (110.4 -> 108.2 mm). Alignment holds, but that is a slow slip, and the plan's hold
   is 1.6 s. Anything asking for a longer hold meets this first.
4. The known bench failures from 2026-08-29 are mechanical, not control: yaw joints came up
   4–6° short under load, and position control over-clamps because the plan asks for 10 mm of
   pad squeeze. Neither is addressed by anything here.

## Reproduce

    python3 scripts/real_v1_trajectory_clearance.py --all --substeps 8
    MUJOCO_GL=egl uv run --extra rl python scripts/real_v1_render_deploy_plan.py \
        --all --path csv --physics --frames 10

`--physics` steps the CSV; without it the replay is kinematic (`mj_forward` only) and shows the
commanded pose and its clearance but **not** whether the tool is carried — the object is not
held in that view, and reading it as a task result would be a mistake.
