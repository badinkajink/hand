# real_v1 observation-transfer correction

This is the hardware-valid continuation of the observation ablation originally run on
legacy `m05`. Only morphologies generated from `assets/mjcf/real_v1/real_hand.xml` and
inside `REAL_V1_WORKSPACE` are admitted.

The two trained designs are platform-reconfigurable geometries with existing real-v1
frozen scenes, grasps, and per-design A/B policies. Neither is being claimed here to have
an already-cleared hardware trajectory: only g12 currently has the collision-safe exported
deployment plan, and g12 has no compatible learned A/B policy pair. The present experiment
tests observation transfer on the correct hardware geometry family; deployment remains a
separate export-and-clearance gate.

## Pre-training screen

The frozen policies were evaluated through the same continuous A-to-B handoff with 32
parallel environments. Replay means that a policy receives another environment's real
observation at the same timestep; unlike zeroing, it stays on the observation manifold.
The jitter distribution is +/-5 mm in object XY and +/-5 degrees in yaw and is OOD for
these old policies.

| design | test | condition | kept | final cos, kept only |
|---|---|---|---:|---:|
| rv03_narrowy | nominal | none | 0.59 | +0.859 |
| rv03_narrowy | nominal | replay hidden object state | 0.50 | +0.857 |
| rv03_narrowy | jitter | none | 0.53 | +0.398 |
| rv03_narrowy | jitter | replay hidden object state | 0.44 | +0.420 |
| rv05_manual | nominal | none | 0.56 | +0.970 |
| rv05_manual | nominal | replay hidden object state | 0.44 | +0.953 |
| rv05_manual | jitter | none | 0.34 | +0.851 |
| rv05_manual | jitter | replay hidden object state | 0.19 | +0.853 |

The eligible policies still look largely trajectory-driven once they retain the shaft:
replaying hidden object state barely changes held-only alignment. The main failure under
perturbation is retention. This makes the jittered sighted-vs-blind training pair useful:
it asks whether privileged object state can improve grip recovery, not merely whether it
can reproduce a nominal turn.

`rv00_wide` was screened and excluded from training: it retained the object but reached
only final cosine about +0.23, consistent with its historical failed-reorienter result.

## Launched comparison

`scripts/train_real_v1_blind_pair.sh` trains two 5M-timestep warmstarted arms on each of
`rv05_manual` and `rv03_narrowy`:

- `S1_sighted_jitter`: full simulated observation, the observability oracle.
- `B1_blind_jitter`: object position, pose, and axis error blinded for the actor while the
  critic remains privileged.

Each run is config-parity checked against that design's measured best anchor policy before
training and writes a rendered evaluation video at iteration 50.

Important limitation: `B1` is an object-state-blind 66-wide network with dead columns; it
is not yet the final 18-D `(9 servo positions, 9 unitless loads)` actor. The current RL env
has no servo-load observation, and B1 still receives joint velocity, reference trajectory,
and previous action. These runs isolate the value of object tracking on real-platform
morphologies; they do not close the hardware observation-interface work.
