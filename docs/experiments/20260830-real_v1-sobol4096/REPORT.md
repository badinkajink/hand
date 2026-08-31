# Real-v1 Sobol-4096 retention-aware morphology screen

**Date:** 2026-08-30
**Status:** population screen and simulation confirmation complete; no hardware-ready winner yet

## Bottom line

The larger search was worth doing. It found many hardware-configurable hands that can both
reorient and genuinely retain the cylinder in nominal simulation under a crude SCS0009 envelope.
The old result was not trustworthy because a transiently upright, nearly dropped object could
pass. This screen instead requires the object to follow a 60 mm proof lift and remain held for
five seconds with at most 10 mm vertical slip.

The strongest current result is `sv1_u2699`: 5/5 nominal confirmations, 0.867 mean retained
alignment, 12.2 mm simulated finger clearance, 2/4 half-error wins, and 5/20 aligned-and-retained
full-error wins. Four other designs reach 4/20 full-error wins. This demonstrates a plausible
18-D proprioceptive route in simulation; it does **not** yet demonstrate adequate robustness for
hardware.

The maneuver is no longer strictly open loop. The turn is open loop, then a capture/hold residual
uses only the nine joint positions and a servo-load proxy derived from nine simulated actuator
torques. It does not observe object pose, contact force, or privileged simulator state. In the six
rendered nominal successes the load correction used only 0.03--3.21 degrees of trim.

## Actuator model

Feetech specifies the SCS0009 at 6 V at 2.3 kg·cm stall torque and 0.7 kg·cm rated torque:

- stall: **0.2256 N·m**;
- rated: **0.06865 N·m**;
- transient screen cap: **0.1804 N·m**, 80% of stall;
- long-hold cap: **0.06865 N·m**, the published rated torque;
- controller target: **250 load units**, approximately 0.0564 N·m under the proxy.

Sources: [Feetech SCS0009 product page](https://www.feetechrc.com/6v-23kg-serial-bus-steering-gear_65522.html)
and [official specification PDF](https://www.feetechrc.com/Data/feetechrc/upload/file/20220915/6379883463905538176347522.pdf).

The proxy is:

```text
load = 1000 * abs(simulated joint torque) / 0.2256 N m
```

This is justified only as napkin screening. Bench protection behavior shows that reported load
tracks the configured percentage registers well enough to be useful: overload 80 trips near 800,
and protective-torque plateaus move from 200 to 400 when the register moves from 20 to 40. It is
not calibrated torque; voltage, temperature, gearbox friction, inertia, deadband, latency, and
protection hysteresis remain absent. See [`TORQUE_MODEL.md`](TORQUE_MODEL.md).

## Replacement success gate

Each strict nominal cell must:

1. fit a three-finger grasp at one of four straddle/thumb placements;
2. execute the -80 degree, `k=0.15` open-loop turn at the transient torque cap;
3. establish the 250-unit per-finger load band using position/load feedback only;
4. follow at least 48 mm of a commanded 60 mm palm lift;
5. retain the freely suspended object for 2,500 steps = five simulated seconds;
6. slip no more than 10 mm during that free hold;
7. finish with alignment at least 0.7 and positive hand contact;
8. keep at least 5 mm dynamic clearance between modeled finger links/tips.

The proof lift is the essential correction: an object balanced on the bench or post cannot follow
the palm. The rendered successes rise 57.8--58.5 mm and slip only 0.8--5.6 mm.

## Funnel

| gate | count |
|---|---:|
| generated | 4,096 Sobol + 6 known anchors |
| inside measured gantry travel | 3,318 (3,313 new + 5 anchors) |
| at least one fitted grasp | 1,656 |
| viable grasp cells across 32/40 mm x 10/20 mm | 5,696 |
| strict single-trial cells | 315 |
| unique strict single-trial morphologies | 248 |
| repeated nominal confirmation, aligned/clear | 119/248 |
| perfect nominal retention | 108/248 at 5/5 |
| at least one half-error aligned hold | 29/248 |
| half-error win rate at least 2/4 | 5/248 |
| independent full-error finalists | 32 (24 performance + 8 morphology diversity) |
| at least one full-error aligned hold | 21/32 |
| best full-error result | 5/20 (`sv1_u2699`) |

The old anchors do not pass this exact gate. For example, g12 retains at one 32/10 operating
point, but that trajectory has -0.2 mm modeled finger overlap and is correctly rejected.

## Best independent full-error results

`kept` includes retained objects whose final alignment is below 0.7; `full win` requires both
retention and alignment. Coordinates are palm-frame millimetres and are directly expressible by
the measured real-v1 rails.

| design | source | grasp S/T | nominal confirmation | half-error win | full-error win / kept | clearance | thumb / index / middle XY |
|---|---|---:|---:|---:|---:|---:|---|
| **sv1_u2699** | uniform | 32/10 | 0.867, 5/5 | 2/4 | **5/20 / 6/20** | 12.2 mm | (-38.5,-22.0) / (52.0,46.8) / (55.2,-38.1) |
| **sv1_w3408** | wide | 40/20 | 0.727, 5/5 | 0/4 | **4/20 / 13/20** | 19.3 mm | (-33.1,52.1) / (42.5,54.3) / (45.6,-48.4) |
| **sv1_w2592** | wide | 40/20 | 0.860, 5/5 | 1/4 | **4/20 / 12/20** | 14.0 mm | (-28.0,54.0) / (50.5,45.5) / (57.8,-40.2) |
| **sv1_u2153** | uniform | 32/10 | 0.842, 5/5 | 1/4 | **4/20 / 9/20** | 15.9 mm | (-62.0,-29.3) / (31.3,60.8) / (49.3,-49.3) |
| **sv1_u2825** | uniform | 32/10 | 0.819, 5/5 | 1/4 | **4/20 / 8/20** | 17.8 mm | (-64.7,-40.2) / (24.6,52.3) / (40.5,-53.4) |
| sv1_u0593 | uniform | 32/10 | 0.715, 5/5 | 1/4 | 3/20 / 7/20 | 18.9 mm | (-51.5,-26.5) / (40.7,57.7) / (55.8,-37.5) |
| sv1_w0459 | wide | 32/10 | 0.812, 5/5 | 1/4 | 3/20 / 6/20 | 18.2 mm | (-37.5,-27.2) / (62.8,60.1) / (74.8,-47.1) |
| sv1_u1364 | uniform | 40/20 | **0.913, 5/5** | **3/4** | 2/20 / 8/20 | 19.1 mm | (-36.7,31.6) / (40.5,27.0) / (48.3,-69.5) |

Twenty trials still give wide binomial uncertainty, but the ranking is sufficient to choose the
next controller/CAD experiments. `sv1_u1364` is the strongest half-error candidate; `sv1_u2699`
is the strongest independent full-error candidate; `sv1_w3408` retains most often under full
error and has unusually large simulated clearance, but often retains at the wrong angle.

## What the outward-X bias did

The 25% outward-X stratum helped the packaging filters slightly:

- measured-rail reachability: 87.0% wide versus 78.8% uniform;
- at least one fitted grasp: 51.3% wide versus 49.3% uniform;
- viable grasp cells: 44.6% wide versus 42.2% uniform.

It did not enrich strict reorientation. At the densest 32/10 cell, pass rates are 8.4% wide and
8.9% uniform. Across unique strict morphologies, 59/891 reachable wide hands pass versus
189/2,422 uniform hands: 6.6% versus 7.8%.

Successful hands are also more compact in X than the reachable population. Mean thumb-to-pair X
separation is about 92 mm among strict hands, versus 104 mm for reachable uniform and 107 mm for
reachable wide samples. This is a real performance/packaging tension, not evidence that the wide
bias was useless. Servo cases and brackets are still absent from collision geometry, so the
compact simulated optimum may be physically invalid. Keep an unbiased majority, retain a wide
stratum, and add measured housing envelopes before selecting a build.

## Interpretation

This screen changes the answer from “the object can look reoriented” to “some morphologies can
reorient and remain captured with a deployable observation set.” That is substantial progress.
It also shows that morphology sampling alone does not solve robustness: the best design wins only
25% of the deliberately broad full-error distribution.

The sensible controller architecture is now concrete:

```text
open-loop grasp and turn
    + 9 servo positions
    + 9 unitless loads
    -> small post-turn per-finger load trim
    -> proof lift / hold
```

Before returning to end-to-end RL, tune this small residual on the finalists and identify which
perturbations dominate failure. A history-based or privileged-teacher residual remains plausible
if the simple clamp plateaus, but it should learn corrections around these retention-capable
morphologies rather than rediscover the maneuver from scratch.

## Recommended next actions

1. Add conservative measured servo, horn, bracket, and cable envelopes to the exact top-eight
   trajectories. Do not build from finger-link clearance alone.
2. On the static bench, measure SCS0009 load versus commanded deflection at several voltage and
   protection settings. Estimate scale, bias, deadband, update latency, and thermal drift.
3. Sweep load target/gain, turn angle/k, speed, and re-seat timing on the top eight, selecting on
   an independent error set. The morphology sweep intentionally held these controller knobs fixed.
4. Add bus-rate latency, load quantization, protection hysteresis, and asymmetric servo strength
   to the simulator, then repeat at least 100 independent draws for the top three.
5. Only then run a staged hardware test: static capture, 10--20 mm proof lift over a catch tray,
   long hold, and finally the reorientation trajectory.

## Artifacts

- [`grasp_screen_manifest.json`](grasp_screen_manifest.json): all 4,102 raw candidates and sampling provenance.
- [`hardware_manifest.json`](hardware_manifest.json): 3,318 measured-rail-reachable candidates.
- [`retention/grasp_summary.json`](retention/grasp_summary.json): complete grasp-funnel counts.
- [`retention/strict_s32_t10.json`](retention/strict_s32_t10.json),
  [`retention/strict_s32_t20.json`](retention/strict_s32_t20.json),
  [`retention/strict_s40_t10.json`](retention/strict_s40_t10.json), and
  [`retention/strict_s40_t20.json`](retention/strict_s40_t20.json): 5,696 strict cells.
- [`retention/confirmation_level05.json`](retention/confirmation_level05.json): five nominal plus
  four half-error trials for 248 hands.
- [`retention/finalists_full_error.json`](retention/finalists_full_error.json): 20 independent
  full-error trials for 32 finalists.
- [`videos/`](videos/): six 15-second proof-lift/hold renders and `renders.json` metrics.

The earlier [`Sobol-128 report`](../20260830-real_v1-sobol128/REPORT.md) is explicitly marked as
superseded for retention claims; it remains an audit trail for how the faulty metric was found.
