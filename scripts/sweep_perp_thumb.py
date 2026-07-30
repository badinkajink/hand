"""Can a perp morphology recruit the THUMB onto the reoriented shaft — without losing the swing?

Background (`docs/rl/perp_topology.md`): the perp hand reorients the screwdriver by gravity on
the opposed index/middle pinch, open-loop cos +0.96. The thumb contributes ~0 N to that rotation
and cannot help afterwards either: it mounts at x = -0.065 while the vertical shaft hangs at
x = +0.035, i.e. **SHORT of its reach shell by ~31 mm**. Meanwhile the pinch force decays from
8.1 N to 0.4 N as the shaft goes vertical, so axial load capacity (= mu*N) collapses to ~1.2 N.
The thumb is the only candidate for restoring force closure on the reoriented object.

This sweeps thumb morphology (x / len, the two params that buy reach) and, per design, measures
the actual trade:

  1. **self-collision gate** — moving the mount forward can bury the thumb's mcp frame in the
     palm. That reads as a 686 N contact and every downstream number from it is garbage, so it
     is rejected first, before anything is believed.
  2. **stow pose, authored in WORLD space** — the thumb is IK'd to the same world position it
     occupies on the working base design, so it is out of the swing corridor by construction.
     Copying the base design's *joint angles* instead would put a moved thumb somewhere else
     entirely (the repo's most expensive recurring mistake); a thumb left at the default open
     pose physically blocks the shaft and reads as "this design can't reorient".
  3. **swing** — settle -> close the pair -> lift -> hold, thumb stowed. Reports cos.
  4. **press** — IK the thumb onto the settled shaft, then ramp an axial (+-Z) load and report
     the escape force, which is the number the thumb is supposed to improve.

A design only counts if it keeps the swing AND raises axial capacity. Reach alone is not the
result: the thumb can be inside its shell and still contribute nothing.

Run:
  MUJOCO_GL=egl uv run python scripts/sweep_perp_thumb.py --out docs/experiments/perp_thumb_sweep.md
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (PROJECT_ROOT / "scripts", PROJECT_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from probe_perp_mechanism import OBJ, obj_cos, tip_forces  # noqa: E402
from pose_open_keyframe import reach_shell  # noqa: E402
from probe_thumb_reach import build_hold, shaft_axis, dist_to_segment  # noqa: E402
from morphohand.tools.keyframe_ik import FINGERS, TIPS, ik_finger  # noqa: E402

BASE_SCENE = PROJECT_ROOT / "assets" / "mjcf" / "perp" / "scenes" / "scene_screwdriver_medium_perp.xml"
THUMB_ACTS = ("a_thumb_yaw", "a_thumb_mcp", "a_thumb_pip")


@dataclass
class Design:
    tx: float
    tlen: float
    imx: float = 0.0
    scene: Path | None = None
    status: str = ""
    cos_swing: float = float("nan")
    z_swing: float = float("nan")
    cos: float = float("nan")
    obj_z: float = float("nan")
    d_mount: float = float("nan")
    shell: tuple[float, float] = (float("nan"), float("nan"))
    reach: str = ""
    press_res: float = float("nan")
    stow_clear: float = float("nan")
    cos_regrip: float = float("nan")
    forces_regrip: dict = field(default_factory=dict)
    forces: dict = field(default_factory=dict)
    f_up: float = float("nan")
    f_dn: float = float("nan")


def generate(tx: float, tlen: float, imx: float, outdir: Path) -> Path:
    tag = f"t{tx:+.3f}_len{tlen:.3f}_im{imx:+.3f}".replace(".", "d")
    r = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "generate_morphology_xml.py"),
         "--base-hand-xml", str(BASE_SCENE), "--base-scene-xml", str(BASE_SCENE),
         "--hand-prefix", f"hand_{tag}", "--scene-prefix", f"scene_{tag}",
         "--output-dir", str(outdir),
         "--thumb", str(tx), "0", str(tlen),
         "--index", str(imx), "0", "0", "--middle", str(imx), "0", "0"],
        capture_output=True, text=True, env={**os.environ, "MUJOCO_GL": "egl"},
    )
    if r.returncode != 0:
        raise RuntimeError(f"generate failed for {tag}: {r.stderr[-400:]}")
    for line in r.stdout.splitlines():
        if "rigid scene XML" in line:
            return Path(line.split(":", 1)[1].strip())
    raise RuntimeError(f"no scene path in generator output for {tag}")


def robot_self_collision(model, data) -> tuple[bool, str]:
    """Any hand-vs-hand contact. The object and the floor are not the robot."""
    for c in range(data.ncon):
        b1 = model.body(model.geom_bodyid[data.contact[c].geom1]).name
        b2 = model.body(model.geom_bodyid[data.contact[c].geom2]).name
        if OBJ in (b1, b2) or "world" in (b1, b2):
            continue
        f = np.zeros(6)
        mujoco.mj_contactForce(model, data, c, f)
        if abs(f[0]) > 1e-6:
            return True, f"{b1}<->{b2} {abs(f[0]):.0f}N"
    return False, ""


def set_thumb_to_target(model, data, target_world: np.ndarray) -> float:
    """IK the thumb tip to a WORLD target; returns residual (m). World-space, never joint-space:
    a moved/lengthened thumb holding the base design's angles lands somewhere else entirely."""
    return ik_finger(model, data, "thumb", target_world)


def stow_thumb(model, data, corridor_palm) -> tuple[dict[str, float], float]:
    """Fold the thumb as far out of the swing corridor as its own joints allow.

    An earlier version aimed the tip at the base design's exact world position. That is
    over-constrained: it happens to sit at the base thumb's D_min, so any LONGER thumb cannot
    reach it and was rejected as "stow unreachable" — a property of the criterion, not of the
    design. What actually matters is only that the thumb is clear of the volume the shaft sweeps
    through, so scan the thumb's own (mcp, pip) and take the pose that maximises clearance.
    """
    jm, jp = model.joint("thumb_mcp"), model.joint("thumb_pip")
    am, ap = model.jnt_qposadr[jm.id], model.jnt_qposadr[jp.id]
    ay = model.jnt_qposadr[model.joint("thumb_yaw").id]
    saved = data.qpos.copy()
    data.qpos[ay] = 0.0
    palm_bid = model.body("palm_pose").id
    best, best_ctrl = -1.0, None
    for mcp in np.linspace(*model.jnt_range[jm.id], 21):
        for pip in np.linspace(*model.jnt_range[jp.id], 21):
            data.qpos[am], data.qpos[ap] = mcp, pip
            mujoco.mj_forward(model, data)
            if robot_self_collision(model, data)[0]:
                continue
            palm = data.body(palm_bid).xpos
            tip = data.body(TIPS["thumb"]).xpos - palm
            clear = min(dist_to_segment(tip, a, b)[0] for a, b in corridor_palm)
            if clear > best:
                best, best_ctrl = clear, {a: float(v) for a, v in
                                          zip(THUMB_ACTS, (0.0, mcp, pip))}
    if best_ctrl is None:
        data.qpos[:] = saved
        mujoco.mj_forward(model, data)
        return {}, -1.0
    data.qpos[ay], data.qpos[am], data.qpos[ap] = (
        best_ctrl["a_thumb_yaw"], best_ctrl["a_thumb_mcp"], best_ctrl["a_thumb_pip"])
    mujoco.mj_forward(model, data)
    return best_ctrl, best


def thumb_ctrl_from_qpos(model, data) -> dict[str, float]:
    out = {}
    for act, jname in zip(THUMB_ACTS, FINGERS["thumb"]):
        out[act] = float(data.qpos[model.jnt_qposadr[model.joint(jname).id]])
    return out


def apply_thumb_ctrl(model, data, ctrl: dict[str, float], blend: float = 1.0) -> None:
    for act, val in ctrl.items():
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act)
        data.ctrl[aid] = data.ctrl[aid] * (1 - blend) + val * blend


def axial_ramp(model, data, sign: int, max_force: float, steps: int, slip_tol: float) -> float:
    bid = model.body(OBJ).id
    palm_bid = model.body("palm_pose").id
    ref = (data.body(bid).xpos - data.body(palm_bid).xpos).copy()
    for step in range(steps):
        force = max_force * (step / max(1, steps - 1))
        data.xfrc_applied[bid, 2] = sign * force
        mujoco.mj_step(model, data)
        if float(np.linalg.norm((data.body(bid).xpos - data.body(palm_bid).xpos) - ref)) > slip_tol:
            data.xfrc_applied[bid, :] = 0.0
            return force
    data.xfrc_applied[bid, :] = 0.0
    return float("inf")


def evaluate(d: Design, args, stow_palm_frame: np.ndarray) -> Design:
    model = mujoco.MjModel.from_xml_path(str(d.scene))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("open").id)
    mujoco.mj_forward(model, data)

    bad, why = robot_self_collision(model, data)
    if bad:
        d.status = f"INVALID self-collision {why}"
        return d

    # --- stow the thumb clear of the swing corridor ----------------------------------------
    stow_ctrl, clearance = stow_thumb(model, data, stow_palm_frame)
    if not stow_ctrl:
        d.status = "INVALID no self-collision-free thumb pose"
        return d
    d.stow_clear = clearance

    # --- swing: settle -> close the pair -> lift -> hold, thumb stowed ---------------------
    apply_thumb_ctrl(model, data, stow_ctrl)
    build_hold(model, data, "closed", args.lift,
               args.settle_steps, args.close_steps, args.lift_steps, args.hold_steps)
    # build_hold drives only the pair + palm, so the thumb stays stowed; no re-assert needed.
    # Do NOT add settle steps here: once vertical the pinch carries only ~0.4 N and a couple of
    # hundred extra steps is enough for the shaft to slip out, which reads as a design failure.

    d.cos_swing = obj_cos(model, data)
    centre, axis, half = shaft_axis(model, data)
    d.z_swing = float(centre[2])
    if d.z_swing < 0.05:
        d.status = "DROPPED in swing"
        return d

    a, b = centre - axis * half, centre + axis * half
    mount = data.body("thumb_mount").xpos.copy()
    d.shell = reach_shell(model, data, "thumb")
    d.d_mount, nearest = dist_to_segment(mount, a, b)
    d.reach = ("SHORT" if d.d_mount > d.shell[1]
               else "TOO-CLOSE" if d.d_mount < d.shell[0] else "INSIDE")

    if d.reach != "INSIDE":
        d.status = f"no press: {d.reach} by " + (
            f"{(d.d_mount - d.shell[1])*1000:.1f} mm" if d.reach == "SHORT"
            else f"{(d.shell[0] - d.d_mount)*1000:.1f} mm")
        return d

    # --- press: IK the thumb onto the settled shaft, radially outward from its axis ---------
    radial = mount - nearest
    radial[2] = 0.0
    n = np.linalg.norm(radial)
    radial = radial / n if n > 1e-9 else np.array([-1.0, 0.0, 0.0])
    press_target = nearest + radial * args.shaft_radius
    # Drive to the IK solution even if it does not fully converge: the physical question is
    # whether the thumb makes contact and carries load, not whether the solver hit the target.
    # A few mm of residual still lands the pad on the shaft; the thumb force column is the judge.
    d.press_res = set_thumb_to_target(model, data, press_target)
    press_ctrl = thumb_ctrl_from_qpos(model, data)

    # re-run the swing, then ease the thumb in (the press must not disturb the held state)
    data2 = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data2, model.key("open").id)
    mujoco.mj_forward(model, data2)
    apply_thumb_ctrl(model, data2, stow_ctrl)
    build_hold(model, data2, "closed", args.lift,
               args.settle_steps, args.close_steps, args.lift_steps, args.hold_steps)
    apply_thumb_ctrl(model, data2, stow_ctrl)

    # --- optional REGRIP before the press --------------------------------------------------
    # Once the shaft is vertical the pinch carries only ~0.4 N, and pressing the thumb onto a
    # near-free-hanging shaft ejects it ("DROPPED at press" on every good-swing design). The
    # clamp only had to stay light to LET the rotation happen; the rotation is over by now, so
    # re-tightening here does not cost alignment. Less pip flexion = more force (measured).
    if args.regrip_pip:
        ids = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
               for n in ("a_index_pip", "a_middle_pip")}
        start = {n: float(data2.ctrl[i]) for n, i in ids.items()}
        for step in range(args.regrip_steps):
            f = min(1.0, step / max(1, args.regrip_steps - 100))
            for n, i in ids.items():
                data2.ctrl[i] = start[n] + args.regrip_pip * f
            mujoco.mj_step(model, data2)
        d.cos_regrip = obj_cos(model, data2)
        d.forces_regrip = tip_forces(model, data2)
        if float(data2.body(OBJ).xpos[2]) < 0.05:
            d.status = "DROPPED at regrip"
            return d

    for step in range(args.press_steps):
        t = min(1.0, step / max(1, args.press_steps - 200))
        for act, val in press_ctrl.items():
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act)
            data2.ctrl[aid] = stow_ctrl[act] * (1 - t) + val * t
        mujoco.mj_step(model, data2)

    d.cos = obj_cos(model, data2)
    d.obj_z = float(data2.body(OBJ).xpos[2])
    d.forces = tip_forces(model, data2)
    if d.obj_z < 0.05:
        d.status = "DROPPED at press"
        return d

    snapshot = (data2.qpos.copy(), data2.qvel.copy(), data2.ctrl.copy())
    d.f_up = axial_ramp(model, data2, +1, args.max_force, args.ramp_steps, args.slip_tol)
    data2.qpos[:], data2.qvel[:], data2.ctrl[:] = snapshot
    mujoco.mj_forward(model, data2)
    d.f_dn = axial_ramp(model, data2, -1, args.max_force, args.ramp_steps, args.slip_tol)
    d.status = "ok"
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thumb-x", default="0,0.01,0.02,0.03",
                    help="thumb_x morph values (mount is at -0.065; joint range +-0.03)")
    ap.add_argument("--thumb-len", default="0,0.01,0.02,0.035",
                    help="thumb_len morph values (joint range 0..0.035)")
    ap.add_argument("--im-x", default="0",
                    help="index+middle x, applied to BOTH so the pair stays symmetric. Moves the "
                         "pinch: forward = bigger off-COM lever (stronger swing) but the shaft "
                         "hangs further from the thumb; back = the reverse. Mounts at +0.035.")
    ap.add_argument("--lift", type=float, default=0.14)
    ap.add_argument("--settle-steps", type=int, default=200)
    ap.add_argument("--close-steps", type=int, default=400)
    ap.add_argument("--lift-steps", type=int, default=900)
    ap.add_argument("--hold-steps", type=int, default=1400)
    ap.add_argument("--press-steps", type=int, default=1600)
    ap.add_argument("--ramp-steps", type=int, default=1500)
    ap.add_argument("--max-force", type=float, default=15.0)
    ap.add_argument("--slip-tol", type=float, default=0.01)
    ap.add_argument("--shaft-radius", type=float, default=0.0075,
                    help="press target offset from the shaft axis (m)")
    ap.add_argument("--ik-tol", type=float, default=0.005)
    ap.add_argument("--regrip-pip", type=float, default=0.0,
                    help="index/middle pip ctrl delta applied AFTER the swing, before the thumb "
                         "press. Negative = tighter. The swing needs a light clamp; the press "
                         "needs a firm one, and those are different phases.")
    ap.add_argument("--regrip-steps", type=int, default=800)
    ap.add_argument("--outdir", type=Path,
                    default=PROJECT_ROOT / "results" / "perp_thumb_sweep")
    ap.add_argument("--out", type=Path, default=None, help="markdown table output")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    # The swing CORRIDOR in palm frame: the shaft's spawn segment (horizontal, on the floor)
    # and its settled segment (vertical, hanging off the pinch). The thumb must stay clear of
    # both or it blocks the rotation and the design reads as un-reorientable.
    base_model = mujoco.MjModel.from_xml_path(str(BASE_SCENE))
    base_data = mujoco.MjData(base_model)
    mujoco.mj_resetDataKeyframe(base_model, base_data, base_model.key("open").id)
    mujoco.mj_forward(base_model, base_data)
    palm0 = base_data.body("palm_pose").xpos.copy()
    c0, ax0, half0 = shaft_axis(base_model, base_data)
    spawn = (c0 - ax0 * half0 - palm0, c0 + ax0 * half0 - palm0)
    build_hold(base_model, base_data, "closed", args.lift,
               args.settle_steps, args.close_steps, args.lift_steps, args.hold_steps)
    palm1 = base_data.body("palm_pose").xpos.copy()
    c1, ax1, half1 = shaft_axis(base_model, base_data)
    settled = (c1 - ax1 * half1 - palm1, c1 + ax1 * half1 - palm1)
    stow = [spawn, settled]
    print(f"[sweep] swing corridor (palm frame): spawn {np.round(spawn[0],3)}..{np.round(spawn[1],3)}"
          f"  settled {np.round(settled[0],3)}..{np.round(settled[1],3)}")

    xs = [float(v) for v in args.thumb_x.split(",")]
    lens = [float(v) for v in args.thumb_len.split(",")]
    imxs = [float(v) for v in args.im_x.split(",")]
    designs = []
    for imx in imxs:
      for tx in xs:
        for tl in lens:
            d = Design(tx=tx, tlen=tl, imx=imx)
            try:
                d.scene = generate(tx, tl, imx, args.outdir)
                d = evaluate(d, args, stow)
            except Exception as e:  # keep the sweep going; a bad design is data
                d.status = f"ERROR {type(e).__name__}: {e}"[:90]
            designs.append(d)
            fu = "held" if d.f_up == float("inf") else f"{d.f_up:.2f}"
            fd = "held" if d.f_dn == float("inf") else f"{d.f_dn:.2f}"
            print(f"  im {imx:+.3f} tx {tx:+.3f} len {tl:.3f} | swing cos {d.cos_swing:+.3f} z {d.z_swing:.3f} "
                  f"| {d.reach:9s} d {d.d_mount:.4f} shell [{d.shell[0]:.4f},{d.shell[1]:.4f}] "
                  f"| N t/i/m {d.forces.get('thumb', float('nan')):.1f}/"
                  f"{d.forces.get('index', float('nan')):.1f}/"
                  f"{d.forces.get('middle', float('nan')):.1f} "
                  f"| regrip cos {d.cos_regrip:+.3f} | axial +Z {fu} -Z {fd} | {d.status}",
                  flush=True)

    if args.out:
        lines = ["# Perp thumb-morphology sweep", "",
                 f"lift {args.lift} m, stow target (palm frame) {np.round(stow, 4)}, "
                 f"slip tol {args.slip_tol*1000:.0f} mm, ramp to {args.max_force} N", "",
                 "| im_x | thumb_x | thumb_len | swing cos | obj z | reach | d_mount | shell | "
                 "N thumb/index/middle | axial +Z | axial -Z | status |",
                 "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for d in designs:
            fu = "held" if d.f_up == float("inf") else f"{d.f_up:.2f}"
            fd = "held" if d.f_dn == float("inf") else f"{d.f_dn:.2f}"
            lines.append(
                f"| {d.imx:+.3f} | {d.tx:+.3f} | {d.tlen:.3f} | {d.cos_swing:+.3f} | {d.z_swing:.3f} | {d.reach} | "
                f"{d.d_mount:.4f} | [{d.shell[0]:.4f},{d.shell[1]:.4f}] | "
                f"{d.forces.get('thumb', float('nan')):.1f}/"
                f"{d.forces.get('index', float('nan')):.1f}/"
                f"{d.forces.get('middle', float('nan')):.1f} | {fu} | {fd} | {d.status} |")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(lines) + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
