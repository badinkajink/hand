# MorphoHand finger — as-simulated spec (2026-08-19)

Dimensions of the finger the shipped policies actually fly, pulled from the MJCF rather than from
any design document. **The authority is `results/phase1/landscape/m05_ik_cem/frozen_scene.xml`** —
that is the scene `a10` and `b33` were trained in and every robustness number was measured on.
Regenerate this with `scripts/build_morphohand_urdf.py` conventions or by reading the scene; do not
copy numbers out of older notes, several of which describe base scenes nothing was trained on.

Frame convention: each finger runs along its own local **+x** (outward from the palm), +z is the palm
normal. All lengths mm.

---

## 1. Kinematic chain (per finger, palm → tip)

| segment | body | length | capsule radius | notes |
|---|---|---|---|---|
| mount → yaw axis | `<f>_mount` → `<f>_yaw_frame` | 0 | — | mount XY is a design parameter (§4) |
| **proximal** | `<f>_mcp_frame` | **see §2** | **10.0** | collision capsule is a fixed 50 mm — see §3 |
| **middle** | `<f>_len_frame` | **40.0** | **8.5** | fixed across all designs |
| **distal** | `<f>_pip_frame` | **30.0** | **7.5** | fixed across all designs |
| **tip pad** | `<f>_tip` | 12.0 long | **5.0** | capsule, `fromto -6..+6` along x |

Tip pad reach: the pad's forward-most surface sits **11.0 mm** from the tip body origin (half-length
6 + radius 5). The object contacts it on the **hemispherical end cap**, i.e. the pad behaves as a
5 mm sphere — see `docs/rl/reorientation.md` §2026-08-18.

Link masses (density-derived): proximal 19.9 g, middle 11.65 g, distal 7.07 g, tip 1.47 g.
Palm plate: **120 × 90 × 2 mm** box, hand origin 134 mm above the table.

## 2. Proximal length — the only segment that varies

The proximal length is a **design parameter**, not a constant. It is the x-offset of `<f>_len_frame`
inside `<f>_mcp_frame`.

| build | thumb | index | middle | trained on it? |
|---|---|---|---|---|
| base scene "short proximal" | 25.0 | 25.0 | 25.0 | no policy directly |
| base scene "long proximal" | 50.0 | 50.0 | 50.0 | **no policy at all** |
| `a01` lineage (b01–b29) | 33.5 | 34.5 | 34.5 | yes — the whole baseline lineage |
| **`m05` SHIPPED (a10 → b33)** | **35.8** | **37.3** | **40.9** | **yes — current reference** |

m05 = the 25 mm short-proximal base **plus** a per-finger length parameter of 10.8 / 12.3 / 15.9 mm.

**Straight-finger reach**, yaw axis → pad surface: thumb **116.8**, index **118.3**, middle **121.9** mm.

## 3. Three things that will bite a hardware build

**(a) The proximal collision capsule does not shorten with the joint.** It is a 50 mm capsule in every
build, while the length parameter only moves where the middle phalanx attaches. On m05 that leaves
**9.1–14.2 mm of overlap** (25.0 mm on the short base) where the middle phalanx sits *inside* the
proximal capsule. As drawn it is not manufacturable, and it is not cosmetic: at operating grip force a
measurable share of the load goes through the phalanx links rather than the pads. Decide deliberately
whether the physical proximal is the kinematic length (36/37/41 mm) or the drawn 50 mm — they are
different robots, and the policy trained against the 50 mm collision geometry.

**(b) The URDF exporter cannot emit the shipped hand.** `scripts/build_morphohand_urdf.py` raises
`NotImplementedError: Per-finger MCP_LEN differs` — it assumes all three fingers share one proximal
length, and m05's do not (35.8/37.3/40.9). It also derives the proximal capsule from the *kinematic*
length, so it already disagrees with the MJCF per (a). Anything downstream of that URDF is describing
a different hand.

**(c) The actuator limits are not servos.** The MJCF uses position servos at `kp=30` with
`forcerange ±10 N·m` per joint, which is effectively unlimited and was never a hardware claim.
Measured requirement below.

## 4. Joint ranges, actuation, and the design space

| joint | range (rad) | range (deg) | damping | armature |
|---|---|---|---|---|
| `<f>_yaw` (all) | −1.100 … +1.100 | −63 … +63 | 0.5 | 0.001 |
| `index/middle_mcp` | 0.000 … +2.500 | 0 … +143 | 0.5 | 0.001 |
| `index/middle_pip` | −1.200 … +1.570 | −69 … +90 | 0.5 | 0.001 |
| `thumb_mcp` | 0.000 … +3.140 | 0 … +180 | 0.5 | 0.001 |
| `thumb_pip` | −1.200 … +0.500 | −69 … +29 | 0.5 | 0.001 |

9 actuated DOF (3 per finger: yaw, mcp, pip). No abduction beyond yaw, no distal joint.

**Morphology design space** (the co-design free variables, `scene_morphology_actuated.xml`): per finger
mount **x ±30 mm**, **y ±30 mm**, and proximal **length +0…35 mm** off the base scene. m05's values:
thumb (+14.7, +5.0, +10.8), index (+4.0, +2.2, +12.3), middle (+24.6, +24.2, +15.9) mm.

Resulting m05 mount positions in palm frame: thumb (−15.3, −23.0), index (+4.0, +32.2),
middle (+24.6, −5.8) mm.

**Measured torque at the grip** (scripted hold, 4.84 N total pad load): MCP is the loaded joint —
thumb −0.437, index +0.212, middle +0.233 N·m; PIP ≤0.12; yaw ≤0.012 N·m. The deployed policy runs
~20 N of pad load, ~4× this, so **size the MCP joint for ≈1 N·m** (~10 kg·cm) and the PIP for ~0.25
N·m. That extrapolation is linear-in-load and has not been measured directly at 20 N.

## 5. Object it is designed around

Screwdriver stand-in: cylinder **25 mm diameter × 100 mm long, 24.5 g**, friction (slide, torsion,
roll) = **2.4 / 0.2 / 0.02**. The 2.4 sliding coefficient is load-bearing and optimistic — see the
robustness note below.

## 6. Pad requirements carried over from the sim2real work

From `docs/experiments/SIM2REAL_ROBUSTNESS.txt` and the fingertip study:

- **μ ≳ 1.7** at the pad. At μ×0.5 (1.2) the hand drops the tool in 94% of rollouts, and DR did not fix it.
- **Pad deflection ~1–3 mm at operating load** (measured 0.82 mm at 4.84 N). A rigid tip does not
  reproduce the trained behaviour; stiffening contact collapses the reorient.
- **Compact convex tip, r 5–6 mm.** Flat pads, grooved pads and line contacts all degrade the reorient
  badly. r4 and r8 are both worse than r5–6.
- **Placement tolerance ±2 mm / ±0.1 rad** for the tool at pickup.

---

### Was long-vs-short proximal ever tested?

**No — not as a controlled comparison, and the 50 mm "long" build has no trained policy at all.**
The two named base scenes (25 / 50 mm) are starting points for the morphology generator, and every
policy that has ever trained ran on a search-chosen intermediate: ~34 mm for the `a01` lineage,
35.8/37.3/40.9 mm for m05. Length *was* one of the 9 co-design parameters, so it was explored — but
the morphology sweep was never able to resolve design differences at the sample sizes run (per-design
policy draw sd 0.3–0.5, `docs/rl/morph_sweep_STATUS.md`), and the later 6-dim sweep froze length at
m05's values rather than testing it. So the current lengths are "what a noisy search landed on", not
"what was shown to be best". Treat them as free to change on hardware grounds.
