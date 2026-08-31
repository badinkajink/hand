# SCS0009 napkin torque/load model for the Sobol-4096 screen

**Purpose:** provide a conservative actuator envelope for tonight's morphology funnel without
claiming that the SCS0009 load register is a calibrated torque sensor.

## Published numbers

Feetech specifies the SCS0009 at 6 V as:

- peak stall torque: 2.3 kg·cm = **0.2256 N·m**;
- rated torque: 0.7 kg·cm = **0.06865 N·m**;
- stall current: 1.0 A;
- case size: 23.2 x 12.1 x 25.25 mm.

Sources: [Feetech SCS0009 product page](https://www.feetechrc.com/6v-23kg-serial-bus-steering-gear_65522.html)
and [official product specification](https://www.feetechrc.com/Data/feetechrc/upload/file/20220915/6379883463905538176347522.pdf).

The existing MJCF uses +/-10 N·m per finger joint. That is a simulation placeholder, roughly
44 times the published stall torque, and cannot support hardware claims.

## Why treating load as percentage is acceptable for screening

The register is not force, current, or calibrated torque. However, the real bench data in
`docs/experiments/20260829-real_v1_bench_gripwindow/README.md` establishes a useful coarse map:

- factory `overload_torque = 80` trips when reported load passes approximately 800;
- factory `protective_torque = 20` produces an exact 200 plateau after protection;
- changing protective torque to 40 moves that plateau to 400.

Therefore the screen uses:

```text
load_proxy = 1000 * abs(simulated joint torque) / 0.2256 N m
```

This is a duty-cycle/percentage proxy. Gear friction, inertia, voltage, deadband, temperature,
and protection hysteresis remain unmodeled.

## Screening envelope

| phase | cap | rationale |
|---|---:|---|
| close and turn | 0.1804 N·m | 80% of stall; avoid entering the overload region |
| capture and long hold | 0.06865 N·m | published rated torque |
| load target | 250 units ~= 0.0564 N·m | below rated torque with a +/-30% deadband |

The capture controller sees only finger joint configuration and per-servo load proxy. It moves
each fingertip along an inward/outward FK direction; it does not read object pose or simulated
contact force.

## Retention pass

A candidate must:

1. execute the open-loop turn under the transient torque cap;
2. establish the load band;
3. raise the palm 60 mm and require the object to follow at least 48 mm, so a cylinder balanced
   on the bench cannot pass;
4. retain the free object for five seconds with no more than 10 mm vertical slip;
5. finish with alignment >= 0.7 and positive hand contact;
6. retain at least 5 mm simulated finger clearance.

The real servo/mount envelopes are still absent from collision geometry. A pass promotes a hand
to CAD-envelope and repeated simulation checks; it does not authorize hardware motion.
