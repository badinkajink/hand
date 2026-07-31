"""Is this perp morphology PHYSICALLY REAL? Bake it, settle it, and ask the contact solver.

The 9-param morph joint ranges are mount RAILS, and on the perp hand the rails pass through
the palm plate. Nothing in the sample -> generate -> evaluate path gates on that, so a design
from inside `PERP_T_WORKSPACE` can be geometrically impossible and still produce a scene that
loads, simulates, and reports plausible-looking numbers built on a 690 N interpenetration.

This is the missing gate. It bakes each design with the standard generator, settles it at the
`open` and `closed` keyframes, and reports every robot-vs-robot contact. A forward-kinematics
reach study cannot replace it: FK does not model the palm, so it happily reports a thumb tip
1 mm from the shaft when the thumb got there THROUGH solid geometry.

Run:
  # the compact-hand interpolation family (the "can the fingers be closer?" question)
  MUJOCO_GL=egl uv run python scripts/morph_selfcollision_gate.py --sweep

  # one design, either as interpolation knobs or as raw 9 params
  MUJOCO_GL=egl uv run python scripts/morph_selfcollision_gate.py --compact 0 0.5 1.0
  MUJOCO_GL=egl uv run python scripts/morph_selfcollision_gate.py \
      --thumb 0 0 0 --index -0.0225 -0.0355 0 --middle -0.0225 0.0355 0

  # stow the thumb first if it is the screwdriver it collides with, not the palm
  MUJOCO_GL=egl uv run python scripts/morph_selfcollision_gate.py --sweep --thumb-yaw 1.1
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (PROJECT_ROOT / "scripts", PROJECT_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from morphohand.sampling.morphology import (  # noqa: E402
    PERP_T_WORKSPACE,
    MorphologyValues,
    perp_compact_design,
    perp_mount_positions,
)
from morphohand.tools.morphology_xml import create_rigid_morphology_xml  # noqa: E402
from morphohand.tools.keyframe_ik import ik_finger  # noqa: E402

PERP_DIR = PROJECT_ROOT / "assets" / "mjcf" / "perp"
BASE_HAND = PERP_DIR / "perp_hand.xml"
BASE_SCENE = PERP_DIR / "scenes" / "scene_screwdriver_medium_perp.xml"
OBJ = "screwdriver_medium"
TIPS = ("thumb_tip", "index_tip", "middle_tip")


def reference_tip_targets(scene: Path) -> dict[str, dict[str, np.ndarray]]:
    """Fingertip WORLD positions of the reference grasp, per keyframe.

    These are the gate's invariant. The screwdriver does not move when a mount does, so a
    compact design has to reach the SAME three points; asking it to hold the same joint ANGLES
    instead is the repo's most expensive recurring mistake (CLAUDE.md gotcha #5) and reports
    spurious self-collisions for designs that are actually fine.
    """
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    out: dict[str, dict[str, np.ndarray]] = {}
    for key_name in ("open", "closed"):
        mujoco.mj_resetDataKeyframe(model, data, model.key(key_name).id)
        mujoco.mj_forward(model, data)
        out[key_name] = {f: data.body(f"{f}_tip").xpos.copy() for f in ("thumb", "index", "middle")}
    return out


@dataclass
class Verdict:
    label: str
    morph: MorphologyValues
    worst_force: float = 0.0
    worst_pair: str = ""
    contacts: tuple[str, ...] = ()
    pinch_gap: float = float("nan")
    thumb_gap: float = float("nan")
    ik_residual: float = 0.0

    @property
    def ok(self) -> bool:
        return self.worst_force <= 1e-6 and self.ik_residual < 2e-3


def robot_self_contacts(model, data) -> list[tuple[str, str, float]]:
    """Every hand-vs-hand contact with non-zero normal force. The object and floor are not the robot."""
    out: list[tuple[str, str, float]] = []
    for c in range(data.ncon):
        b1 = model.body(model.geom_bodyid[data.contact[c].geom1]).name
        b2 = model.body(model.geom_bodyid[data.contact[c].geom2]).name
        if OBJ in (b1, b2) or "world" in (b1, b2):
            continue
        f = np.zeros(6)
        mujoco.mj_contactForce(model, data, c, f)
        if abs(f[0]) > 1e-6:
            out.append((b1, b2, abs(float(f[0]))))
    return out


def settle(model, data, key: int, steps: int, thumb_yaw: float | None,
           targets: dict[str, np.ndarray] | None = None) -> float:
    """Reset to `key`, optionally re-IK the tips onto `targets`, settle. Returns worst residual."""
    mujoco.mj_resetDataKeyframe(model, data, key)
    residual = 0.0
    if targets is not None:
        # Retarget in WORLD space: same grasp, new mount geometry. A stowed thumb is exempt --
        # it is deliberately NOT on its grasp target, so IK-ing it would fight the stow and its
        # residual would be meaningless.
        for finger, target in targets.items():
            if thumb_yaw is not None and finger == "thumb":
                continue
            residual = max(residual, ik_finger(model, data, finger, target))
        for a in range(model.nu):
            jid = model.actuator_trnid[a, 0]
            if jid >= 0:
                data.ctrl[a] = float(data.qpos[model.jnt_qposadr[jid]])
    if thumb_yaw is not None:
        # Stow the thumb out of the swing corridor. thumb_yaw rolls the finger about its own
        # proximal axis, which only swings the TIP laterally once the mcp is flexed -- which it
        # is at every authored perp keyframe. Drive both qpos and ctrl so it starts settled.
        jid = model.joint("thumb_yaw").id
        data.qpos[model.jnt_qposadr[jid]] = thumb_yaw
        data.ctrl[model.actuator("a_thumb_yaw").id] = thumb_yaw
    for _ in range(steps):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    return residual


def evaluate(label: str, morph: MorphologyValues, outdir: Path, steps: int,
             thumb_yaw: float | None, base_scene: Path = BASE_SCENE,
             targets: dict[str, dict[str, np.ndarray]] | None = None) -> Verdict:
    scene = create_rigid_morphology_xml(
        base_xml_path=base_scene,
        morphology=morph,
        output_xml_path=outdir / f"scene_{label.replace(' ', '_').replace('/', '_')}.xml",
    )
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)

    v = Verdict(label=label, morph=morph)
    seen: dict[str, float] = {}
    for key_name in ("open", "closed"):
        key = model.key(key_name).id
        v.ik_residual = max(
            v.ik_residual,
            settle(model, data, key, steps, thumb_yaw,
                   None if targets is None else targets[key_name]),
        )
        for b1, b2, f in robot_self_contacts(model, data):
            pair = f"{b1}<->{b2}"
            seen[pair] = max(seen.get(pair, 0.0), f)
        if key_name == "closed":
            tips = {t: data.body(t).xpos.copy() for t in TIPS}
            v.pinch_gap = float(np.linalg.norm(tips["index_tip"] - tips["middle_tip"]))
            v.thumb_gap = float(np.linalg.norm(
                tips["thumb_tip"] - 0.5 * (tips["index_tip"] + tips["middle_tip"])))

    if seen:
        v.worst_pair, v.worst_force = max(seen.items(), key=lambda kv: kv[1])
        v.contacts = tuple(f"{k} {f:.0f}N" for k, f in sorted(seen.items(), key=lambda kv: -kv[1]))
    return v


def fmt_row(v: Verdict) -> str:
    mounts = perp_mount_positions(v.morph)
    geom = (f"thumb x{mounts['thumb'][0] * 1e3:+6.1f} | pair x{mounts['index'][0] * 1e3:+5.1f} "
            f"|y|{abs(mounts['index'][1]) * 1e3:4.1f}")
    if v.worst_force > 1e-6:
        return f"  {v.label:22s} {geom}  SELF-COLLIDE {v.worst_pair} {v.worst_force:.0f}N"
    if v.ik_residual >= 2e-3:
        return f"  {v.label:22s} {geom}  UNREACHABLE  grasp IK residual {v.ik_residual * 1e3:.1f} mm"
    return (f"  {v.label:22s} {geom}  OK           "
            f"pinch {v.pinch_gap * 1e3:5.1f} mm  thumb->pair {v.thumb_gap * 1e3:5.1f} mm")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep", action="store_true",
                   help="Sweep the compact-design interpolation family instead of one design.")
    p.add_argument("--compact", nargs=3, type=float, metavar=("THUMB_T", "PAIR_X_T", "PAIR_Y_T"),
                   help="One design from perp_compact_design, each knob in [0,1].")
    p.add_argument("--thumb", nargs=3, type=float, metavar=("X", "Y", "LEN"))
    p.add_argument("--index", nargs=3, type=float, metavar=("X", "Y", "LEN"))
    p.add_argument("--middle", nargs=3, type=float, metavar=("X", "Y", "LEN"))
    p.add_argument("--thumb-yaw", type=float, default=None,
                   help="Stow the thumb at this yaw (rad, |.|<=1.1) before settling.")
    p.add_argument("--steps", type=int, default=400, help="Settle steps per keyframe.")
    p.add_argument("--outdir", type=Path, default=None, help="Keep the baked scenes here.")
    p.add_argument("--base-scene", type=Path, default=BASE_SCENE,
                   help="Scene to bake designs from (for testing palm/geometry variants).")
    p.add_argument("--retarget", action="store_true",
                   help="IK each design onto the REFERENCE grasp's fingertip world targets "
                        "before settling, instead of reusing the authored joint angles. "
                        "Without this a moved mount is judged in a pose authored for the old "
                        "one, and reports self-collisions it would not actually have.")
    return p


def main() -> None:
    args = build_parser().parse_args()

    designs: list[tuple[str, MorphologyValues]] = []
    if args.sweep:
        ts = (0.0, 0.25, 0.5, 0.75, 1.0)
        designs.append(("shipped (0,0,0)", perp_compact_design()))
        designs += [(f"thumb_t {t:.2f}", perp_compact_design(thumb_t=t)) for t in ts[1:]]
        designs += [(f"pair_x_t {t:.2f}", perp_compact_design(pair_x_t=t)) for t in ts[1:]]
        designs += [(f"pair_y_t {t:.2f}", perp_compact_design(pair_y_t=t)) for t in ts[1:]]
        designs += [(f"pair both {t:.2f}", perp_compact_design(pair_x_t=t, pair_y_t=t)) for t in ts[1:]]
        designs += [(f"all {t:.2f}", perp_compact_design(t, t, t)) for t in ts[1:]]
    elif args.compact:
        t = tuple(args.compact)
        designs.append((f"compact {t}", perp_compact_design(*t)))
    elif args.thumb and args.index and args.middle:
        designs.append(("explicit", MorphologyValues(
            *args.thumb[:2], args.thumb[2], *args.index[:2], args.index[2],
            *args.middle[:2], args.middle[2])))
    else:
        raise SystemExit("Provide --sweep, --compact, or all of --thumb/--index/--middle.")

    tmp = None
    outdir = args.outdir
    if outdir is None:
        tmp = tempfile.TemporaryDirectory()
        outdir = Path(tmp.name)
    outdir.mkdir(parents=True, exist_ok=True)

    stow = "" if args.thumb_yaw is None else f", thumb stowed at yaw {args.thumb_yaw:+.2f} rad"
    print(f"[gate] base {args.base_scene.name}, settle {args.steps} steps/keyframe{stow}")
    mode = "RETARGETED onto the reference grasp" if args.retarget else "authored joint angles (NOT retargeted)"
    print(f"[gate] pose: {mode}")
    print(f"[gate] mount positions are PALM-FRAME mm; pinch = index_tip<->middle_tip at `closed`\n")

    targets = reference_tip_targets(BASE_SCENE) if args.retarget else None
    verdicts = [evaluate(label, m, outdir, args.steps, args.thumb_yaw, args.base_scene, targets)
                for label, m in designs]
    for v in verdicts:
        print(fmt_row(v))

    n_ok = sum(v.ok for v in verdicts)
    print(f"\n[gate] {n_ok}/{len(verdicts)} designs are physically real")
    worst = [v for v in verdicts if not v.ok]
    if worst:
        pairs = sorted({v.worst_pair for v in worst})
        print(f"[gate] blocking contacts: {', '.join(pairs)}")
    if tmp is not None:
        tmp.cleanup()


if __name__ == "__main__":
    main()
