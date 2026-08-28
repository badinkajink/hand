"""Why does the shaft creep out of the real_v1 grip, and which fix stops it?

Reported 2026-08-27 while Policy A was training on rv00_wide: the shaft slowly slips out.
The trainer agrees — `Episode_Termination/tip_lost` runs 1.4-2.3 per batch even though
`object_height` sits at 0.106 against a 0.10 target, so the lift succeeds and then the grip
degrades. That pattern (lift fine, contact lost later) is a CONTACT problem, not a lift problem,
and it should reproduce with no policy at all.

So this probe removes the policy: it loads a morphology run's frozen scene, applies that run's
own CEM grip, ramps the palm up and then just holds, for far longer than the 1.4 s RL episode.
Anything that creeps out here is the physics, not PPO. CPU-only, so it runs while the GPU trains.

WHAT IT MEASURES (per variant, at the end of the hold)

    held        metres of lift still present            -- the headline; ~0 means it fell out
    axial       shaft displacement along its OWN axis   -- the failure the perp study found no
                                                           reward term measures
    radial      shaft displacement across its axis
    tilt        degrees the shaft's axis has rotated
    contacts    min / mean fingertip contacts over the hold
    force       mean total fingertip normal force

WHAT IT SWEEPS

The four candidate fixes, so they can be compared on one axis instead of argued about:

    pads      contact COMPLIANCE on the fingertips. The real pads are TPU; the scene gives them
              the same near-rigid solref/solimp as the links. A compliant pad spreads the
              contact and is the physically honest model.
    friction  torsional and rolling friction on the pads. MuJoCo's point contact has none worth
              speaking of by default (0.2 / 0.02), but a soft pad pressed into a 25 mm shaft has
              a real patch, and a patch is what resists the shaft ROLLING out from between the
              fingertips.
    mu        sliding friction, the blunt instrument.
    palm      palm height, +/- a few mm. Lower palm = fingers less extended = more wrap, but
              also less grip depth, which is what clears the shaft's upper half when it stands
              up (LINK_LENGTH_GATE). The probe prints both so the trade is visible.

    MUJOCO_GL=egl uv run python scripts/probe_real_v1_slip.py \
        --morph-run results/phase1/real_v1/rv00_wide_sp40 --seconds 4
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FINGERS = ("thumb", "index", "middle")
TIPS = tuple(f"{f}_tip" for f in FINGERS)
# The two geoms per finger that can actually touch the shaft: the pad sphere and the distal
# capsule it caps. The proximal links are behind the object and never contact it in a tripod.
PAD_BODIES = TIPS + tuple(f"{f}_pip_frame" for f in FINGERS)


def _pad_geoms(root: ET.Element):
    for body in root.iter("body"):
        if body.get("name") in PAD_BODIES:
            for g in body.findall("geom"):
                yield g


def mutate(src: Path, out: Path, *, solref=None, solimp=None, friction=None,
           palm_dz: float = 0.0) -> Path:
    """Write a variant of `src`. `friction` is (slide, torsion, roll); None keeps the value."""
    root = ET.parse(src).getroot()
    for g in _pad_geoms(root):
        if solref:
            g.set("solref", solref)
        if solimp:
            g.set("solimp", solimp)
        if friction:
            cur = [float(v) for v in (g.get("friction") or "1 0.005 0.0001").split()]
            cur += [0.0] * (3 - len(cur))
            g.set("friction", " ".join(str(n if n is not None else c)
                                       for n, c in zip(friction, cur)))
    if palm_dz:
        for body in root.iter("body"):
            if body.get("name") == "palm_pose":
                pos = [float(v) for v in (body.get("pos") or "0 0 0").split()]
                pos[2] += palm_dz
                body.set("pos", " ".join(f"{v:.6f}" for v in pos))
    out.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(out, encoding="utf-8", xml_declaration=False)
    return out


def _obj_frame(d, bid) -> tuple[np.ndarray, np.ndarray]:
    """(position, unit long-axis) of the cylinder — its local +z."""
    R = d.body(bid).xmat.reshape(3, 3)
    return d.body(bid).xpos.copy(), R[:, 2].copy()


def run(scene: Path, finger_ctrl: np.ndarray, keyframe: str, lift: float, seconds: float,
        object_body: str, settle: float = 0.6, ramp: float = 0.5) -> dict:
    m = mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key(keyframe).id)
    mujoco.mj_forward(m, d)

    ctrl = np.zeros(m.nu)
    for a in range(m.nu):
        jid = m.actuator_trnid[a, 0]
        ctrl[a] = float(d.qpos[m.jnt_qposadr[jid]]) if jid >= 0 else 0.0
    fing = [a for a in range(m.nu) if m.actuator(a).name.startswith("a_")
            and m.actuator(a).name.split("_", 1)[1].split("_")[0] in FINGERS]
    ctrl[fing] = finger_ctrl
    pz_a = next(a for a in range(m.nu) if m.actuator(a).name == "a_palm_pz")
    pz0 = float(ctrl[pz_a])
    d.ctrl[:] = ctrl

    bid = m.body(object_body).id
    tip_bids = {m.body(t).id for t in TIPS}
    dt = m.opt.timestep
    n_settle, n_ramp, n_hold = int(settle / dt), int(ramp / dt), int(seconds / dt)

    for _ in range(n_settle):
        mujoco.mj_step(m, d)
    for k in range(n_ramp):
        d.ctrl[pz_a] = pz0 + lift * (k + 1) / n_ramp
        mujoco.mj_step(m, d)

    p0, a0 = _obj_frame(d, bid)
    palm0 = float(d.body("palm_pose").xpos[2])
    ncons, forces, wrench = [], [], np.zeros(6)
    for _ in range(n_hold):
        mujoco.mj_step(m, d)
        n, f = 0, 0.0
        for c in range(d.ncon):
            con = d.contact[c]
            b1, b2 = m.geom_bodyid[con.geom1], m.geom_bodyid[con.geom2]
            if bid in (b1, b2) and (b1 in tip_bids or b2 in tip_bids):
                mujoco.mj_contactForce(m, d, c, wrench)
                n += 1
                f += float(np.linalg.norm(wrench[:3]))
        ncons.append(n)
        forces.append(f)

    p1, a1 = _obj_frame(d, bid)
    # Split the drift into "along the shaft" and "across it", in the shaft's ORIGINAL frame:
    # the axial component is the creep the perp study found no reward term measures.
    delta = p1 - p0
    axial = float(np.dot(delta, a0))
    radial = float(np.linalg.norm(delta - axial * a0))
    return {
        "held_mm": (float(d.body(bid).xpos[2]) - float(m.key(keyframe).qpos[2])) * 1000.0,
        "grip_depth_mm": (palm0 - p0[2]) * 1000.0,
        "axial_mm": axial * 1000.0,
        "radial_mm": radial * 1000.0,
        "tilt_deg": float(np.degrees(np.arccos(np.clip(abs(np.dot(a0, a1)), -1, 1)))),
        "contacts_min": int(min(ncons)) if ncons else 0,
        "contacts_mean": float(np.mean(ncons)) if ncons else 0.0,
        "force_mean_N": float(np.mean(forces)) if forces else 0.0,
    }


# The variants, in the order the report prints them. Each is one hypothesis about the slip.
VARIANTS: dict[str, dict] = {
    "base": {},
    # TPU pads. solref time constant 6 ms -> 20 ms and a lower dmax: the pad yields, the contact
    # patch grows, and the shaft settles INTO the pad instead of riding on a point.
    "pads_soft": dict(solref="0.02 1", solimp="0.90 0.96 0.002"),
    "pads_softer": dict(solref="0.03 1", solimp="0.85 0.94 0.003"),
    # A patch resists rotation. Point contact does not.
    "fric_torsion": dict(friction=(None, 1.0, 0.10)),
    "fric_mu4": dict(friction=(4.0, None, None)),
    # Palm height: less extended fingers wrap more, but grip depth pays for it.
    "palm_down5": dict(palm_dz=-0.005),
    "palm_up5": dict(palm_dz=+0.005),
    # The combination the physics argues for: compliant pad AND the patch friction it implies.
    "pads_soft+torsion": dict(solref="0.02 1", solimp="0.90 0.96 0.002",
                              friction=(None, 1.0, 0.10)),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--morph-run", type=Path, required=True)
    ap.add_argument("--keyframe", default="open_ik")
    ap.add_argument("--object-body", default="screwdriver_medium")
    ap.add_argument("--lift", type=float, default=0.10)
    ap.add_argument("--seconds", type=float, default=4.0,
                    help="hold time AFTER the lift; the RL episode is only 1.4 s")
    ap.add_argument("--only", default=None, help="comma list of variant names")
    ap.add_argument("--out", type=Path, default=None, help="write the table as JSON")
    ap.add_argument("--work", type=Path,
                    default=ROOT / "assets/mjcf/experimental/20260827-real_v1/slip")
    args = ap.parse_args()

    frozen = args.morph_run / "frozen_scene.xml"
    ctrl = np.load(args.morph_run / "best_rollout.npz")["best_finger_ctrl"]
    names = ([v.strip() for v in args.only.split(",")] if args.only else list(VARIANTS))

    print(f"{args.morph_run.name}   hold {args.seconds:.1f}s after a {args.lift*100:.0f} cm lift")
    print(f"{'variant':20} {'held':>7} {'depth':>7} {'axial':>7} {'radial':>7} {'tilt':>6} "
          f"{'cts min/mean':>13} {'force':>7}")
    rows = {}
    for name in names:
        spec = copy.deepcopy(VARIANTS[name])
        scene = (frozen if not spec
                 else mutate(frozen, args.work / f"{args.morph_run.name}__{name}.xml", **spec))
        try:
            r = run(scene, ctrl, args.keyframe, args.lift, args.seconds, args.object_body)
        except Exception as exc:
            print(f"{name:20} ERROR {type(exc).__name__}: {exc}")
            continue
        rows[name] = r
        print(f"{name:20} {r['held_mm']:6.1f}mm {r['grip_depth_mm']:6.1f}mm "
              f"{r['axial_mm']:+6.1f}mm {r['radial_mm']:6.1f}mm {r['tilt_deg']:5.1f}d "
              f"{r['contacts_min']:5d}/{r['contacts_mean']:6.2f} {r['force_mean_N']:6.2f}N")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
