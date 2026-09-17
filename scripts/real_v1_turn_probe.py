#!/usr/bin/env python3
"""Per-step mechanics of the chain's finger turn: what stops the tool on a compliant plant.

    python3 scripts/real_v1_turn_probe.py --hand sv1_u0308_b050 --plant fast --out DIR

Runs one chain cell (grasp, lift, finger turn, hold; nothing after) with a per-step hook and
records, every `--every` steps, for each finger: commanded and achieved joint angles, the tip
the turn asked for (its lift-time tip rotated about the pivot), the tip the rigid IK model
puts at the COMMAND, and the achieved tip; for each pad: normal force, tangential force, and
slip speed at the contact; and for the tool: signed cosine, pose in the palm frame, and which
non-pad geoms touch it (proximal links, the palm plate). The analysis reports where the
commanded motion goes: into the tool, into spring deflection, or into slip.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
FINGERS = {"thumb": ("thumb_yaw", "thumb_mcp", "thumb_pip"),
           "index": ("index_yaw", "index_mcp", "index_pip"),
           "middle": ("middle_yaw", "middle_mcp", "middle_pip")}
TIPS = {"thumb": "thumb_tip", "index": "index_tip", "middle": "middle_tip"}


def contact_variant(scene: Path, cone: str | None, impratio: float | None, mu: float | None,
                    torsional: float | None = None) -> Path:
    """A copy of `scene` with the contact solver / pad friction changed; cached beside it.
    `torsional` (m) switches the pad and tool geoms to condim 4 with that torsional coefficient
    (MuJoCo: torque = coefficient x normal force), so a two-point pinch resists a swing."""
    if cone is None and impratio is None and mu is None and torsional is None:
        return scene
    import xml.etree.ElementTree as ET
    tag = "".join(x for x in (f"_{cone}" if cone else "", f"_ir{impratio:g}" if impratio else "",
                              f"_mu{mu:g}" if mu else "", f"_tor{torsional:g}" if torsional else ""))
    out = scene.with_name(scene.stem + "__c" + tag + ".xml")
    if out.exists():
        return out
    root = ET.parse(scene).getroot()
    opt = root.find("option")
    if opt is None:
        opt = ET.SubElement(root, "option")
    if cone:
        opt.set("cone", cone)
    if impratio:
        opt.set("impratio", f"{impratio:g}")
    if mu or torsional:
        for g in root.iter("geom"):
            fr = g.get("friction")
            if fr and fr.split()[0] in ("2.4", f"{mu:g}" if mu else "2.4"):
                parts = fr.split()
                if mu:
                    parts[0] = f"{mu:g}"
                if torsional:
                    parts[1] = f"{torsional:g}"
                    g.set("condim", "4")
                g.set("friction", " ".join(parts))
    ET.ElementTree(root).write(out, encoding="unicode")
    return out


def run(hand_tag: str, plant: str, squeeze: float, axis_k: float, budget: float, angle: float,
        turn_steps: int, every: int, seed: int, extra: dict,
        cone: str | None = None, impratio: float | None = None, mu: float | None = None,
        depth_mm: float | None = None, straddle_mm: float | None = None,
        torsional: float | None = None, yaw_deg: float = 0.0) -> dict:
    import mujoco
    import real_v1_chain_hands as H
    import probe_real_v1_chain as C
    h = dict(next(x for x in H.hands("AB") if x["tag"] == hand_tag))
    if depth_mm is not None:
        h["depth"] = depth_mm / 1000.0
        h["tag"] += f"_d{depth_mm:g}"
    if straddle_mm is not None:
        h["straddle"] = straddle_mm / 1000.0
        h["tag"] += f"_s{straddle_mm:g}"
    fit = H.prepare(h, yaw_deg=yaw_deg, squeeze_mm=squeeze)
    if fit is None:
        raise SystemExit(f"prepare failed for {h['tag']} (no grasp fits at depth {h['depth']*1e3:g} mm)")
    scene = contact_variant(H.plant_scene(Path(fit["arm"]), plant), cone, impratio, mu, torsional)
    m = mujoco.MjModel.from_xml_path(str(scene))
    obj = "screwdriver_medium"
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, obj)
    obj_geoms = {i for i in range(m.ngeom) if m.geom_bodyid[i] == bid}
    tip_bid = {f: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, TIPS[f]) for f in FINGERS}
    pad_geoms = {f: {i for i in range(m.ngeom) if m.geom_bodyid[i] == tip_bid[f]} for f in FINGERS}
    gname = {i: (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or
                 f"{mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[i])}/g{i}") for i in range(m.ngeom)}
    jadr = {j: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for js in FINGERS.values() for j in js}
    aid = {j: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"a_{j}") for js in FINGERS.values() for j in js}
    # a rigid copy of the model for "where does the COMMAND put the tip"
    mk = mujoco.MjModel.from_xml_path(str(scene))
    dk = mujoco.MjData(mk)
    rec = {"steps": [], "contacts_nonpad": {}}
    ctx: dict = {}
    dref = {"d": None}
    from morphohand.tools.keyframe_ik import ik_finger

    def feasibility(d):
        """Per-finger IK residual of the contact-fixed rotation, from the LIFTED state, over
        angle x pivot height x pivot placement. Reach limits come from the joint ranges the IK
        clips to; nothing about contact or compliance enters, so this is the kinematic ceiling."""
        tip0 = {f: d.body(TIPS[f]).xpos.copy() for f in FINGERS}
        cen = np.mean([tip0[f] for f in FINGERS], axis=0)
        span = abs(tip0["index"][1] - tip0["middle"][1]) / 2.0
        q_lift = d.qpos.copy()
        out = []
        pivots = {"centroid": cen.copy(), "thumb": tip0["thumb"].copy(),
                  "pair": 0.5 * (tip0["index"] + tip0["middle"])}
        for pname, p0 in pivots.items():
            for k in (0.0, 0.15, 0.3, 0.5, 0.75, 1.0):
                c = p0.copy(); c[2] += k * span
                for deg in (-10, -20, -30, -40, -50, -60, -75, -90, 10, 20, 30):
                    a = np.radians(deg)
                    Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
                    res = {}
                    for f in FINGERS:
                        dk.qpos[:] = q_lift
                        res[f] = round(ik_finger(mk, dk, f, c + Rx @ (tip0[f] - c), iters=300) * 1e3, 2)
                    out.append({"pivot": pname, "axis_k": k, "angle": deg, "res_mm": res,
                                "max_mm": max(res.values())})
        return {"span_mm": round(span * 1e3, 1), "cells": out}

    def hook(step):
        d = dref["d"]
        if d is None or step % every:
            return
        R = d.body(obj).xmat.reshape(3, 3)
        Rp = d.body("palm_pose").xmat.reshape(3, 3)
        p_palm = Rp.T @ (d.body(obj).xpos - d.body("palm_pose").xpos)
        # commanded tip: FK of the rigid copy at q = ctrl (fingers), rest of the state copied
        dk.qpos[:] = d.qpos
        for j, a in aid.items():
            dk.qpos[jadr[j]] = d.ctrl[a]
        mujoco.mj_kinematics(mk, dk)
        tip_cmd = {f: dk.body(TIPS[f]).xpos.copy() for f in FINGERS}
        tip = {f: d.body(TIPS[f]).xpos.copy() for f in FINGERS}
        # target tip on the arc, if the turn has started
        tip_tgt = None
        if "turn" in ctx and "feasibility" not in rec and step >= ctx["turn"]["step0"]:
            rec["feasibility"] = feasibility(d)
        if "turn" in ctx and step >= ctx["turn"]["step0"]:
            T = ctx["turn"]
            u = min(1.0, (step - T["step0"] + 1) / T["turn_steps"])
            a = T["angle"] * u
            c = np.array(T["centroid"])
            Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
            tip_tgt = {f: (c + Rx @ (np.array(T["tip0"][f]) - c)).tolist() for f in FINGERS}
        pads = {f: {"n": 0.0, "t": 0.0, "slip": 0.0, "k": 0} for f in FINGERS}
        nonpad = []
        plate = 0.0
        c6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            if not (g1 in obj_geoms or g2 in obj_geoms):
                continue
            other = g2 if g1 in obj_geoms else g1
            mujoco.mj_contactForce(m, d, i, c6)
            fn = abs(float(c6[0]))
            ft = float(np.hypot(c6[1], c6[2]))
            # relative velocity of the two bodies at the contact point, tangential part
            b1, b2 = m.geom_bodyid[g1], m.geom_bodyid[g2]
            v1 = np.zeros(6); v2 = np.zeros(6)
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b1, v1, 0)
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b2, v2, 0)
            p = c.pos
            vp1 = v1[3:] + np.cross(v1[:3], p - d.xpos[b1])
            vp2 = v2[3:] + np.cross(v2[:3], p - d.xpos[b2])
            dv = vp1 - vp2
            n = c.frame[:3]
            slip = float(np.linalg.norm(dv - np.dot(dv, n) * n))
            hit = False
            for f, gs in pad_geoms.items():
                if other in gs:
                    pads[f]["n"] += fn; pads[f]["t"] += ft; pads[f]["k"] += 1
                    pads[f]["slip"] = max(pads[f]["slip"], slip)
                    hit = True
            if not hit:
                nm = gname[other]
                nonpad.append((nm, round(fn, 3)))
                if "palm" in nm or m.geom_bodyid[other] == mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "palm_pose"):
                    plate += fn
                rec["contacts_nonpad"][nm] = rec["contacts_nonpad"].get(nm, 0) + 1
        rec["steps"].append({
            "step": int(step), "t": round(float(d.time), 4), "phase": len(dref["seams"]),
            "cos": round(float(R[2, 2]), 4), "z": round(float(d.body(obj).xpos[2]), 5),
            "p_palm": [round(float(v), 5) for v in p_palm],
            "q_cmd": {j: round(float(d.ctrl[a]), 5) for j, a in aid.items()},
            "q": {j: round(float(d.qpos[jadr[j]]), 5) for j in aid},
            "act_f": {j: round(float(d.actuator_force[a]), 4) for j, a in aid.items()},
            "tip": {f: [round(float(v), 5) for v in tip[f]] for f in FINGERS},
            "tip_cmd": {f: [round(float(v), 5) for v in tip_cmd[f]] for f in FINGERS},
            "tip_tgt": tip_tgt,
            "pads": {f: {k: round(v, 4) if isinstance(v, float) else v for k, v in pads[f].items()} for f in FINGERS},
            "nonpad": nonpad, "plate_N": round(plate, 3),
        })

    seams_ref: list = []
    dref["seams"] = seams_ref

    # chain() builds its own MjData; grab it through the hook's first call via a wrapper that
    # looks it up from the actuator array identity. Simpler: patch mujoco.MjData creation.
    real_MjData = mujoco.MjData

    def capture(model):
        d = real_MjData(model)
        if model.nu == m.nu and model.nbody == m.nbody and dref["d"] is None:
            dref["d"] = d
        return d
    C.mujoco.MjData = capture
    try:
        kw = dict(H.BASE)
        kw.update(dict(angle_deg=angle, axis_k=axis_k, budget=budget, turn_steps=turn_steps,
                       load_target=0.0, seed=seed, jitter=0.0005, cycles=1,
                       place_xy=fit["place_xy"], seat_z=fit["seat_z"], tip_len=fit["tip_len"]))
        kw.update(extra)
        for k in ("squeeze",):
            kw.pop(k, None)
        kw["squeeze"] = squeeze / 1000.0
        r = C.chain(Path(h["tag"]), arm_ik=Path(fit["ik"]), scene_path=scene,
                    anchor_ctrl=fit["anchor"], grip_depth=fit["depth_mm"] / 1000,
                    step_hook=hook, ctx=ctx, **kw)
    finally:
        C.mujoco.MjData = real_MjData
    # phases from the seams the chain recorded (their step is not stored; use t)
    seam_t = [(s["phase"], s["t"]) for s in r["seams"]]
    for st in rec["steps"]:
        st["phase"] = next((n for n, t in reversed(seam_t) if st["t"] >= t - 1e-9), "grasp")
    rec["turn"] = ctx.get("turn")
    rec["seams"] = [{k: s.get(k) for k in ("phase", "t", "cos", "z", "pad_contacts", "pad_force_N", "slide_mm", "roll_deg")} for s in r["seams"]]
    rec["result"] = {k: r.get(k) for k in ("drop_stage", "ok", "carry_ok", "reorient_deg")}
    rec["hand"], rec["plant"], rec["scene"] = hand_tag, plant, str(scene)
    return rec


def analyse(rec: dict) -> dict:
    T = rec["turn"]
    steps = [s for s in rec["steps"] if T and s["step"] >= T["step0"] and s["step"] < T["step0"] + T["turn_steps"] + 1]
    if not steps:
        return {}
    s0, s1 = steps[0], steps[-1]
    out = {"n": len(steps), "cos0": s0["cos"], "cos1": s1["cos"],
           "turn_deg": round(float(np.degrees(np.arcsin(np.clip(s1["cos"], -1, 1)) - np.arcsin(np.clip(s0["cos"], -1, 1)))), 1)}
    per = {}
    for f, js in FINGERS.items():
        tgt = np.array(s1["tip_tgt"][f]); tip1 = np.array(s1["tip"][f]); tip0 = np.array(s0["tip"][f])
        cmd1 = np.array(s1["tip_cmd"][f])
        asked = tgt - tip0
        got = tip1 - tip0
        per[f] = {
            "tip_asked_mm": round(float(np.linalg.norm(asked) * 1e3), 1),
            "tip_moved_mm": round(float(np.linalg.norm(got) * 1e3), 1),
            "tip_cmd_from_tgt_mm": round(float(np.linalg.norm(cmd1 - tgt) * 1e3), 1),
            "tip_from_cmd_mm": round(float(np.linalg.norm(tip1 - cmd1) * 1e3), 1),
            "dq_cmd_deg": {j: round(float(np.degrees(s1["q_cmd"][j] - s0["q_cmd"][j])), 1) for j in js},
            "dq_deg": {j: round(float(np.degrees(s1["q"][j] - s0["q"][j])), 1) for j in js},
            "err0_deg": {j: round(float(np.degrees(s0["q_cmd"][j] - s0["q"][j])), 1) for j in js},
            "err1_deg": {j: round(float(np.degrees(s1["q_cmd"][j] - s1["q"][j])), 1) for j in js},
            "pad_n0": s0["pads"][f]["n"], "pad_n1": s1["pads"][f]["n"],
            "pad_n_max": round(max(s["pads"][f]["n"] for s in steps), 3),
            "pad_t_max": round(max(s["pads"][f]["t"] for s in steps), 3),
            "slip_max_mm_s": round(max(s["pads"][f]["slip"] for s in steps) * 1e3, 1),
            "slip_mean_mm_s": round(float(np.mean([s["pads"][f]["slip"] for s in steps])) * 1e3, 1),
            "act_f_max": round(max(abs(s["act_f"][j]) for s in steps for j in js), 4),
        }
    out["fingers"] = per
    out["nonpad_steps"] = sum(1 for s in steps if s["nonpad"])
    out["nonpad_geoms"] = sorted({n for s in steps for n, _ in s["nonpad"]})
    out["plate_N_max"] = max(s["plate_N"] for s in steps)
    out["z0"], out["z1"] = s0["z"], s1["z"]
    out["p_palm0"], out["p_palm1"] = s0["p_palm"], s1["p_palm"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hand", default="sv1_u0308_b050")
    ap.add_argument("--plant", default="fast")
    ap.add_argument("--squeeze-mm", type=float, default=10.0)
    ap.add_argument("--axis-k", type=float, default=0.15)
    ap.add_argument("--budget", type=float, default=0.5)
    ap.add_argument("--angle", type=float, default=-90.0)
    ap.add_argument("--turn-steps", type=int, default=550)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extra", default="{}", help="json of extra chain() kwargs")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--cone", default=None, choices=(None, "pyramidal", "elliptic"))
    ap.add_argument("--impratio", type=float, default=None)
    ap.add_argument("--mu", type=float, default=None, help="pad/tool sliding friction (scene ships 2.4)")
    ap.add_argument("--depth-mm", type=float, default=None, help="palm height above the shaft centre (fit default 62.5)")
    ap.add_argument("--straddle-mm", type=float, default=None)
    ap.add_argument("--torsional", type=float, default=None, help="condim 4 torsional friction, m")
    ap.add_argument("--yaw", type=float, default=0.0, help="tool heading on the table, deg")
    a = ap.parse_args()
    rec = run(a.hand, a.plant, a.squeeze_mm, a.axis_k, a.budget, a.angle, a.turn_steps, a.every,
              a.seed, json.loads(a.extra), a.cone, a.impratio, a.mu, a.depth_mm, a.straddle_mm,
              a.torsional, a.yaw)
    an = analyse(rec)
    rec["analysis"] = an
    a.out.mkdir(parents=True, exist_ok=True)
    tag = a.tag or (f"{a.hand}__{a.plant}__sq{a.squeeze_mm:g}_k{a.axis_k:g}_b{a.budget:g}_a{a.angle:g}"
                    + (f"_{a.cone}" if a.cone else "") + (f"_ir{a.impratio:g}" if a.impratio else "")
                    + (f"_mu{a.mu:g}" if a.mu else "") + (f"_d{a.depth_mm:g}" if a.depth_mm else "")
                    + (f"_s{a.straddle_mm:g}" if a.straddle_mm else "") + (f"_tor{a.torsional:g}" if a.torsional else "")
                    + (f"_y{a.yaw:g}" if a.yaw else ""))
    (a.out / f"{tag}.json").write_text(json.dumps(rec, indent=None))
    print(json.dumps(an, indent=1))
    print(f"wrote {a.out / (tag + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
