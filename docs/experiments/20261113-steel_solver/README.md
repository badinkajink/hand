# Steel and Newton hydroelastic contact against the sphere pads: runbook

The owner asked on 2026-11-13 for a comparison with Skild AI's Steel solver: "if steel has hydroelastic contact built
in, and it is that much faster than warp, then i think we do have to compare to them". The question is whether the
twist benchmark's plant runs faster with Steel and Newton's hydroelastic fingertip than with MuJoCo-Warp and the 1 mm
sphere pads, and whether HOM reaches the same task score on it. The result page will be an HTML page in this folder;
this file says how to rerun the study.

## Prior numbers

Skild reports throughput only, on one RTX PRO 6000 Blackwell (www.skild.ai/blogs/steel, 2026-11-09). Over 20 Isaac
Lab tasks the geometric mean is 3.8 times PhysX for Steel and 2.9 times for MuJoCo-Warp. On the three Allegro tasks Steel
is 1.8 to 2.4 times MuJoCo-Warp. Skild's own preset notes are less favourable for hydroelastic contact
(`logs/20261113-steel/src/steel/examples/shelf/03_mesh_sdf_contacts/*/presets.toml`). On the Panda pick-and-place, Steel
with compliant hydroelastic rows is "still about 5x MuJoCo's frame time". On the nut and bolt, the compliant rows let 5 to
11 of 20 nuts slide off. Our own rows of 2026-10-06 on the D6 hand at 1024 worlds
(`docs/experiments/20261006-rl_contact/newton_throughput.jsonl`) gave 3.19 us per world-step for MuJoCo-Warp with
pads and about 12 us for Newton's SolverMuJoCo with hydroelastic tips.

## Environment

`logs/20261113-steel/venv` holds Skild's Newton fork at the commit the Steel examples pin
(`skild-ai/newton@f6fc82a8b06d036e1e6d02a61d77a2d300476e8d`, Newton 1.9.0.dev0), Warp 1.19.0, MuJoCo and MuJoCo-Warp
3.17.0, trimesh and scipy. Sources are cloned under `logs/20261113-steel/src` (steel, newton-skild, newton-main).
Rebuild it with:

    cd logs/20261113-steel && uv venv --python 3.12 venv
    VIRTUAL_ENV=$PWD/venv uv pip install -e "src/newton-skild[examples]" "mujoco~=3.17.0" "mujoco-warp~=3.17.0" trimesh scipy

The MuJoCo rows run in the repository's `.venv` (MuJoCo 3.6, MuJoCo-Warp 3.6), as the benchmark does.

## Plants

All plants carry the benchmark's hand, sphere, servo and scene friction (`scripts/hom/inner_loop_plant.py`). The script
is `scripts/contact/steel_hand_compare.py`; its docstring lists the plants:

| plant | solver | fingertip contact |
|---|---|---|
| `mjc_pads` | MuJoCo 3.6 CPU | 1 mm sphere pads, 2.5 MPa (the benchmark's model) |
| `mjc_pt` | MuJoCo 3.6 CPU | point contact on the v4b tip's convex mesh |
| `nt_mjc_hydro` | Newton SolverMuJoCo (MuJoCo-Warp 3.17) | hydroelastic tip meshes and sphere |
| `steel_hydro` | SolverSteel, `contact_compliance` | the same hydroelastic contacts as implicit spring-damper rows |
| `steel_hydro_hard` | SolverSteel | hard rows at the hydroelastic contact points |
| `steel_pt`, `steel_pt_pf` | SolverSteel | point contact on the tip mesh; friction patches, point friction |

The hydroelastic pressure is p = kh x depth with kh = 2.5 MPa / 8.5 mm on the tips and 100 kh on the sphere. The servos
are joint drives with `drive_mode="rows"`, which bounds the whole drive reaction at the 0.18 N m force range as MuJoCo
clamps the actuator force. Steel needs `articulation_constraint_max` raised from its default of 32 rows per world: the
grasp asks for about 180.

## Stages

1. HOM on each plant (`hom`). The CPU controller of `hom_gpu.cpu_rollout` runs through the twist on the 20 pairs of
   `docs/experiments/20261112-inner_loop_benchmark/rows/mpc_pairs_twist.json`, seeds 0 and 1. Rows: `rows/hom_<plant>.jsonl`
   with J (`hom_landscape.track_score`), held and the metrics. Trajectories (object position and axis, pad forces per
   period) and the state at 2.4 s go to `logs/20261113-steel/traj/<plant>/`. Each plant grasps from the open pose
   itself: started from the pads' squeezed grasp, with the tip meshes 0.36 to 0.62 mm inside the sphere, Steel's rigid
   rows threw the sphere out.
2. Throughput (`speed`) from each plant's own state at 2.4 s on the first pair, at 16, 256, 1024 and 2048 worlds,
   targets held: us per world-step, contacts per world, GPU memory, and the share of worlds still holding the sphere
   after the timed window. MuJoCo-Warp rows (`mjw_pads`, `mjw_pt`) use the benchmark's solver settings.
3. Tip stiffness and pinch torque, only for a Steel plant that holds the grasp and is faster than `mjw_pads` at 1024
   worlds: kh fitted to the v3 tips' UR5 presses (`docs/experiments/20261030-v3_tip_contact_model/`), and the pinch
   torque arm at 1 and 2.5 N against the pads' 1.4 and 1.8 mm.

A Steel plant would replace the pads in the planners only if it is at least 1.3 times as fast as `mjw_pads` at the
planner's batch (1024 to 2048 worlds), holds the sphere in every world, and gives HOM a mean J within the pads' seed
spread on the 20 pairs.

## GPU slots

GPU work runs as lines of the benchmark's lane (`logs/20261112-mpc/lane.txt`, run by `lane.sh`), never beside it, and
the script is not edited while a line that runs it is queued. Lines of 2026-11-13:

- `steel_smoke` (line 88, 17:05): one pair, seed 0, single world. Steel hydroelastic held the sphere at J 0.611 in 32 s
  of wall time. Newton-MuJoCo hydroelastic held it at J 0.017, with the importer's friction gain and 10 line-search
  steps. Steel point contact went to NaN with CUDA graphs. A graph-replay bug of the script skipped 10 steps at each
  capture; it is fixed.
- `steel_smoke2` (line 90, 17:10): failed at once, because the script was edited under the queued line.
- `steel_stage1` (line 110, before the lane's mu04 block): `logs/20261113-steel/stage1.sh`, HOM on the five Newton
  plants (40 worlds per batch) and throughput at 16, 256 and 1024 worlds, every step capped at 5 or 10 min.

## CPU references (2026-11-13)

HOM on `mjc_pads`: mean J 0.382, median 0.394, 40 of 40 held, 3.7 s per rollout. On `mjc_pt`: mean J 0.330 over 37
rollouts, all 37 held. The other 3 `mjc_pt` rollouts stopped when HOM's least-squares solve did not converge
("SVD did not converge"). On the CPU, Steel with point contact gave J 0.618 on the first pair at seed 0 (pads 0.773,
MuJoCo point contact 0.412), at 124 s per rollout.
