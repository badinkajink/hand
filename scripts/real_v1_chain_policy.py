#!/usr/bin/env python3
"""Run a reorient checkpoint as the finger controller inside the UR5e chain (CPU MuJoCo).

    uv run --extra rl python scripts/real_v1_chain_policy.py --hand D6 \
        --policy results/rl/<run>/tensorboard/model_270.pt \
        --morphology-run results/phase1/real_v1/20260916-sv1_u0308_b050_cal \
        --squeeze 10 --plant cal --turn-steps 1500 --out docs/experiments/<folder>/d6_chain.json --video ...

The chain (probe_real_v1_chain.chain, invoked as real_v1_chain_hands does for the deployed
hands) grasps the tool on the post, lifts it 100 mm with the arm through the wrist stack, and
then, instead of sweeping the finger anchor, asks the policy for the finger set-points every 10
sim steps for `turn_steps`; the fingers then freeze and the chain continues into its re-pose,
set-down and gait. The observation is rebuilt from the chain's own state (CpuPolicy in
policy_cpu_rollout.py, validated against the GPU eval on the benchmark scene); the benchmark's
step counter is resumed at the run's residual-activation step so the reference-trajectory
terms line up, and the palm height term is the lift. Reports the chain's seams (cos, tilt, z,
pad contacts, pad force, roll and slide in the palm frame) and the policy's own trace.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

HANDS = {"D1": "sv1_w6689_b060", "D2": "sv1_w2360_b075", "D3": "sv1_u1364_b080", "D4": "g12_b095",
         "D5": "sv1_u0060_b75", "D6": "sv1_u0308_b050", "D7": "rv05_manual_b85", "D8": "sv1_w0099_b100"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hand", required=True)
    ap.add_argument("--policy", type=Path, required=True)
    ap.add_argument("--morphology-run", type=Path, required=True)
    ap.add_argument("--squeeze", type=float, default=10.0)
    ap.add_argument("--plant", default="cal")
    ap.add_argument("--plate-mm", type=float, default=25.0,
                    help="palm plate height above the mounting plane; every chain scene was generated with 0, the built hand is 25")
    ap.add_argument("--turn-steps", type=int, default=1500, help="sim steps the policy is in the loop (10 per policy step)")
    ap.add_argument("--hold-steps", type=int, default=300)
    ap.add_argument("--stand", default="air", choices=("air", "table"))
    ap.add_argument("--stages", default="policy", choices=("policy", "chain"),
                    help="grasp and lift profile: 'policy' = the RL environment's (eased closure 240 sim steps, lift "
                         "80, settle 260 -> the policy starts at sim step 580 as in training); 'chain' = the chain's "
                         "own (snap to anchor, 250 settle, lift 200, settle 200)")
    ap.add_argument("--arm-kp-scale", type=float, default=10.0,
                    help="UR5e servo position gain multiplier (kv unchanged); 1 is the menagerie servo, which is "
                         "still 4.9 mm below its command when the policy takes over; 10 holds the palm to 0.1 mm "
                         "as the arm on the bench does once its payload is declared")
    ap.add_argument("--finger-gravcomp", type=float, default=0.0,
                    help="gravity compensation on the finger bodies; the arm scene ships 1 (the payload declaration "
                         "compensates the whole hand subtree), the training scene and the bench hand carry the finger "
                         "links on the finger servos (0)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jitter", type=float, default=0.0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--video", type=Path, default=None)
    ap.add_argument("--film", type=Path, default=None)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(2)
    import real_v1_chain_hands as RH
    import probe_real_v1_chain as C
    from policy_cpu_rollout import CpuPolicy, FINGER_JOINTS

    tag = HANDS.get(a.hand, a.hand)
    h = {x["tag"]: x for x in RH.hands("AB")}[tag]
    fit = RH.prepare(h, squeeze_mm=a.squeeze)
    if fit is None:
        raise SystemExit(f"no fit for {tag} at squeeze {a.squeeze}")
    scene = RH.plant_scene(Path(fit["arm"]), a.plant)
    if a.plate_mm:
        from real_v1_plan_angle_sweep import plate_variant
        scene = plate_variant(scene, a.plate_mm)
    cell = dict(RH.TABLE if a.stand == "table" else RH.BASE)
    seat = dict(place_xy=None, seat_z=None, tip_len=0.0) if a.stand == "table" else dict(
        place_xy=fit["place_xy"], seat_z=fit["seat_z"], tip_len=fit["tip_len"])
    if a.stand != "table":
        cell["angle_deg"] = h["angle_deg"]
    cell["turn_steps"], cell["hold_steps"] = a.turn_steps, a.hold_steps
    if a.stages == "policy":
        cell.update(close_ease_steps=240, lift_ramp=80, post_lift_settle=260)
    cell["arm_kp_scale"] = a.arm_kp_scale
    cell["finger_gravcomp"] = a.finger_gravcomp

    state = {"pol": None, "last": np.zeros(len(FINGER_JOINTS), dtype=np.float32), "k0": None, "trace": [],
             "arm": [], "in_turn": False, "shadow_steps": 0}

    def arm_record(k, m, d):
        # THE ARM'S TRACKING ERROR, per joint and at the palm. A real UR5e is told its payload and
        # holds ~0.1 mm under it; the scene's arm is a P/D pair (kp 2000/500) with gravcomp on its
        # links and on the hand, so anything left here is what the tool's weight and the pads'
        # reactions bend the servo by. `droop_mm` is the palm site's z at the commanded joint
        # angles minus its z as simulated (positive = the palm hangs below its command).
        names = ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")
        defl = {}
        d2 = copy.copy(d)
        for n in names:
            j = m.joint(f"{n}_joint")
            act = m.actuator(n)
            defl[n] = round(float(d.qpos[j.qposadr[0]] - d.ctrl[act.id]) * 1000.0, 3)
            d2.qpos[j.qposadr[0]] = d.ctrl[act.id]
        mujoco.mj_kinematics(m, d2)
        s = m.site("palm_site").id
        z_cmd, z_act = float(d2.site_xpos[s][2]), float(d.site_xpos[s][2])
        xy_mm = float(np.linalg.norm(d2.site_xpos[s][:2] - d.site_xpos[s][:2])) * 1000.0
        state["arm"].append({"k": int(k), "defl_mrad": defl, "palm_z_cmd": round(z_cmd, 5),
                             "palm_z": round(z_act, 5), "droop_mm": round((z_cmd - z_act) * 1000.0, 3),
                             "xy_err_mm": round(xy_mm, 3)})

    ctx = {}

    def shadow(s):
        # THE POLICY RUNS FROM SIM STEP 0, AS IN TRAINING. The environment evaluates the actor at
        # every control step of the closure and the lift and masks its output until step 58, so
        # the `actions` slot of the first active observation holds the actor's own previous output
        # (saturated at the clip on most fingers) and not zeros. With `--stages policy` the chain's
        # timeline is the environment's (240 / 80 / 260 sim steps), so the shadow clock is the
        # sim step over the decimation and the palm height in the observation follows the ramp.
        if state["in_turn"] or a.stages != "policy":
            return
        if state["pol"] is None:
            state["pol"] = CpuPolicy(a.policy, a.morphology_run, ctx["m"], ctx["d"])
            state["k0"] = state["pol"].active_from // state["pol"].decim
        pol = state["pol"]
        if s % pol.decim == 0 and s // pol.decim < state["k0"]:
            state["last"] = pol.act(pol.obs(s // pol.decim, state["last"]))
            state["shadow_steps"] += 1

    def turn_ctrl(k, m, d, acts, anchor, sq0):
        state["in_turn"] = True
        if state["pol"] is None:
            state["pol"] = CpuPolicy(a.policy, a.morphology_run, m, d)
            state["k0"] = state["pol"].active_from // state["pol"].decim
        pol = state["pol"]
        if k % (25 * pol.decim) == 0:
            arm_record(k // pol.decim, m, d)
        kk = state["k0"] + k // pol.decim
        o = pol.obs(kk, state["last"], palm_dz=cell["lift"])
        if k == 0:
            state["obs0"] = [round(float(v), 5) for v in o]      # what the policy sees at its first active step
        act = pol.act(o)
        state["last"] = act
        anchor_vec = np.array([anchor[j] + sq0[j] for j in FINGER_JOINTS])
        tgt = pol.finger_target(act, anchor_vec)
        pf = pol.pad_forces()
        state["trace"].append([int(k // pol.decim), round(pol.tool_cos(), 4), round(float(d.xpos[pol.tool][2]), 4),
                               round(float(np.abs(act).max()), 3), round(pf["thumb"], 2), round(pf["index"], 2), round(pf["middle"], 2)])
        return {j: float(tgt[i]) for i, j in enumerate(FINGER_JOINTS)}

    r = C.chain(Path(tag), arm_ik=Path(fit["ik"]), scene_path=scene, anchor_ctrl=fit["anchor"],
                grip_depth=fit["depth_mm"] / 1000, axis_k=h["axis_k"], budget=h["budget"],
                **seat, **cell, jitter=a.jitter, seed=a.seed, turn_ctrl=turn_ctrl,
                step_hook=shadow, ctx=ctx, video=a.video, film=a.film)
    seams = {s["phase"]: s for s in r["seams"]}
    keep = ("t", "cos", "tilt_deg", "z", "pad_contacts", "pad_force_N", "roll_deg", "slide_mm", "hand_contacts")
    out = {"hand": a.hand, "tag": tag, "policy": str(a.policy), "scene": str(scene), "squeeze_mm": a.squeeze, "plant": a.plant,
           "turn_steps": a.turn_steps, "stand": a.stand, "stages": a.stages, "arm_kp_scale": a.arm_kp_scale,
           "finger_gravcomp": a.finger_gravcomp,
           "seams": {ph: {k: s.get(k) for k in keep if k in s} for ph, s in seams.items()},
           "held_turn": bool((seams.get("turned", {}).get("pad_contacts") or 0) >= 2 and (seams.get("turned", {}).get("z") or 0) > 0.08),
           "ok": r.get("ok"), "cycles": r.get("n_cycles", r.get("cycles_ok")), "reorient_deg": r.get("reorient_deg"),
           "policy_trace": state["trace"], "arm_tracking": state["arm"], "obs_at_active": state.get("obs0"),
           "shadow_steps": state["shadow_steps"], "turn_step0": int(ctx.get("turn", {}).get("step0", -1)),
           "chain_keys": sorted(k for k in r.keys() if isinstance(r[k], (int, float, bool, str)))[:40],
           "chain_scalars": {k: r[k] for k in r if isinstance(r[k], (int, float, bool)) and k not in ("seams",)}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    t, ro = seams.get("turned", {}), seams.get("reoriented", {})
    print(f"{a.hand} chain: lifted pads {seams.get('lifted', {}).get('pad_contacts')} | turned cos {t.get('cos')} z {t.get('z')} "
          f"pads {t.get('pad_contacts')} N {t.get('pad_force_N')} | reoriented cos {ro.get('cos')} pads {ro.get('pad_contacts')} "
          f"| held_turn {out['held_turn']} | chain ok {out['ok']} | seams {list(seams)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
