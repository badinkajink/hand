"""Rank the perp compact-design workspace by the MECHANISM, before spending any GPU on it.

Per-design RL is seed-dominated (CLAUDE.md gotcha #7): a single training run cannot resolve
morphology differences, so training every design is the expensive way to learn nothing. This
sweeps the gated-valid designs with the scripted open-loop swing instead — minutes each, no GPU,
no policy — and ranks them on the gravity reorient the topology actually runs on.

Per design: bake -> self-collision gate -> retarget the grasp to the reference fingertip world
targets -> settle / close / lift / HOLD LONG -> report.

The hold is deliberately long. The perp release is a short-horizon artifact: a probe that stops
just after the rotation reports "cos +0.98, airborne, HELD" for a shaft that is about to end up
standing on the floor at cos +1.000. `held` here is asked of the physics — total fingertip normal
force, object height, and object<->floor contact — never inferred from cos.

Run:
  MUJOCO_GL=egl uv run python scripts/sweep_perp_compact.py --out docs/experiments/perp_compact_sweep.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (PROJECT_ROOT / "scripts", PROJECT_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from morph_selfcollision_gate import (  # noqa: E402
    BASE_SCENE, OBJ, TIPS, reference_tip_targets, robot_self_contacts,
)
from morphohand.sampling.morphology import (  # noqa: E402
    MorphologyValues, perp_compact_design, perp_mount_positions,
)
from morphohand.tools.morphology_xml import create_rigid_morphology_xml  # noqa: E402
from morphohand.tools.keyframe_ik import ik_finger  # noqa: E402


@dataclass
class Result:
    label: str
    thumb_t: float
    pair_x_t: float
    pair_y_t: float
    morph: dict
    status: str = "ok"
    peak_cos: float = float("nan")
    final_cos: float = float("nan")
    grip_n: float = float("nan")
    obj_z: float = float("nan")
    on_floor: bool = False
    held: bool = False
    score: float = float("-inf")


def obj_cos(model, data) -> float:
    """Signed alignment of the shaft's long axis with world +Z."""
    quat = data.body(OBJ).xquat
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, quat)
    return float(rot.reshape(3, 3)[:, 2][2])


def tip_force_total(model, data) -> float:
    total = 0.0
    tip_bodies = {model.body(t).id for t in TIPS}
    for c in range(data.ncon):
        b1 = model.geom_bodyid[data.contact[c].geom1]
        b2 = model.geom_bodyid[data.contact[c].geom2]
        names = {model.body(b1).name, model.body(b2).name}
        if OBJ in names and (b1 in tip_bodies or b2 in tip_bodies):
            f = np.zeros(6)
            mujoco.mj_contactForce(model, data, c, f)
            total += abs(float(f[0]))
    return total


def object_touches_floor(model, data) -> bool:
    for c in range(data.ncon):
        names = {model.body(model.geom_bodyid[data.contact[c].geom1]).name,
                 model.body(model.geom_bodyid[data.contact[c].geom2]).name}
        if OBJ in names and "world" in names:
            f = np.zeros(6)
            mujoco.mj_contactForce(model, data, c, f)
            if abs(f[0]) > 1e-6:
                return True
    return False


def _ctrl_from_qpos(model, data) -> np.ndarray:
    ctrl = np.zeros(model.nu)
    for a in range(model.nu):
        jid = model.actuator_trnid[a, 0]
        if jid >= 0:
            ctrl[a] = data.qpos[model.jnt_qposadr[jid]]
    return ctrl


def rollout(scene: Path, targets, args) -> tuple[float, float, float, float, bool, bool]:
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)

    # Retargeted CLOSE pose: same grasp points, this design's geometry.
    mujoco.mj_resetDataKeyframe(model, data, model.key("closed").id)
    for finger, target in targets["closed"].items():
        ik_finger(model, data, finger, target)
    closed_ctrl = _ctrl_from_qpos(model, data)

    mujoco.mj_resetDataKeyframe(model, data, model.key("open").id)
    for finger, target in targets["open"].items():
        ik_finger(model, data, finger, target)
    open_ctrl = _ctrl_from_qpos(model, data)

    mujoco.mj_resetDataKeyframe(model, data, model.key("open").id)
    data.ctrl[:] = open_ctrl
    for _ in range(args.settle_steps):
        mujoco.mj_step(model, data)

    pz = model.actuator("a_palm_pz").id
    peak = -2.0
    for phase, steps in (("close", args.close_steps), ("lift", args.lift_steps),
                         ("hold", args.hold_steps)):
        for i in range(steps):
            if phase == "close":
                a = (i + 1) / steps
                data.ctrl[:] = (1 - a) * open_ctrl + a * closed_ctrl
            elif phase == "lift":
                data.ctrl[pz] = args.lift * (i + 1) / steps
            mujoco.mj_step(model, data)
            if phase in ("lift", "hold"):
                peak = max(peak, obj_cos(model, data))

    mujoco.mj_forward(model, data)
    return (peak, obj_cos(model, data), tip_force_total(model, data),
            float(data.body(OBJ).xpos[2]), object_touches_floor(model, data), False)


def evaluate(label, knobs, morph, outdir, targets, args) -> Result:
    r = Result(label=label, thumb_t=knobs[0], pair_x_t=knobs[1], pair_y_t=knobs[2],
               morph=asdict(morph))
    scene = create_rigid_morphology_xml(
        base_xml_path=BASE_SCENE, morphology=morph,
        output_xml_path=outdir / f"scene_{label.replace('.', 'd')}.xml")

    # Gate first: an impossible design still simulates and still reports numbers.
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    for key in ("open", "closed"):
        mujoco.mj_resetDataKeyframe(model, data, model.key(key).id)
        for _ in range(300):
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        hits = robot_self_contacts(model, data)
        if hits:
            b1, b2, f = max(hits, key=lambda h: h[2])
            r.status = f"self-collide {b1}<->{b2} {f:.0f}N"
            return r

    peak, final, grip, z, floor, _ = rollout(scene, targets, args)
    r.peak_cos, r.final_cos, r.grip_n, r.obj_z, r.on_floor = peak, final, grip, z, floor
    # HELD is a physics question: still gripped, off the floor, above the spawn height.
    r.held = bool(grip > 0.5 and not floor and z > 0.06)
    r.score = (peak if r.held else peak - 1.0)
    return r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "docs" / "experiments" / "perp_compact_sweep.md")
    ap.add_argument("--scenes-out", type=Path,
                    default=PROJECT_ROOT / "assets" / "mjcf" / "experimental" / "perp_compact")
    ap.add_argument("--grid", type=int, default=5, help="knob levels per axis (0..1)")
    ap.add_argument("--lift", type=float, default=0.10)
    ap.add_argument("--settle-steps", type=int, default=200)
    ap.add_argument("--close-steps", type=int, default=400)
    ap.add_argument("--lift-steps", type=int, default=700)
    ap.add_argument("--hold-steps", type=int, default=1600,
                    help="long on purpose: a short hold reports a release as a success")
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    args.scenes_out.mkdir(parents=True, exist_ok=True)
    targets = reference_tip_targets(BASE_SCENE)

    ts = np.linspace(0.0, 1.0, args.grid)
    designs: list[tuple[str, tuple[float, float, float], MorphologyValues]] = []
    seen = set()
    for tt in ts:
        for px in ts:
            for py in ts:
                # pair_y is gate-blocked past ~0 (fingers scissor); keep 0 and one probe level.
                if py not in (ts[0], ts[1]):
                    continue
                knobs = (round(float(tt), 3), round(float(px), 3), round(float(py), 3))
                if knobs in seen:
                    continue
                seen.add(knobs)
                designs.append((f"t{knobs[0]:.2f}_x{knobs[1]:.2f}_y{knobs[2]:.2f}", knobs,
                                perp_compact_design(*knobs)))

    print(f"[sweep] {len(designs)} designs, hold {args.hold_steps} steps "
          f"({args.hold_steps * 0.002:.1f} s) after the lift\n")
    results = []
    for i, (label, knobs, morph) in enumerate(designs, 1):
        r = evaluate(label, knobs, morph, args.scenes_out, targets, args)
        results.append(r)
        m = perp_mount_positions(morph)
        tag = (f"peak {r.peak_cos:+.3f} final {r.final_cos:+.3f} grip {r.grip_n:5.1f}N "
               f"z {r.obj_z:.3f} {'HELD' if r.held else ('on-floor' if r.on_floor else 'released')}"
               if r.status == "ok" else r.status)
        print(f"  [{i:2d}/{len(designs)}] {label}  thumb x{m['thumb'][0]*1e3:+6.1f} "
              f"pair x{m['index'][0]*1e3:+5.1f}  {tag}")

    ok = sorted([r for r in results if r.status == "ok"], key=lambda r: -r.score)
    print(f"\n[sweep] {len(ok)}/{len(results)} gated valid; "
          f"{sum(r.held for r in ok)} HELD at the end of a {args.hold_steps * 0.002:.1f} s hold")

    lines = ["# Perp compact-design sweep (scripted open-loop swing)", "",
             f"{len(designs)} designs from `perp_compact_design`, gated on self-collision, grasp",
             "retargeted to the reference fingertip world targets. `held` is asked of the physics",
             f"(tip force, height, floor contact) after a {args.hold_steps * 0.002:.1f} s hold, never inferred from cos.", "",
             "| rank | design | thumb x | pair x | pair \\|y\\| | peak cos | final cos | grip N | obj z | verdict |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for rank, r in enumerate(ok, 1):
        m = perp_mount_positions(MorphologyValues(**r.morph))
        verdict = "**HELD**" if r.held else ("on floor" if r.on_floor else "released")
        lines.append(f"| {rank} | `{r.label}` | {m['thumb'][0]*1e3:+.1f} | {m['index'][0]*1e3:+.1f} | "
                     f"{abs(m['index'][1])*1e3:.1f} | {r.peak_cos:+.3f} | {r.final_cos:+.3f} | "
                     f"{r.grip_n:.1f} | {r.obj_z:.3f} | {verdict} |")
    rejected = [r for r in results if r.status != "ok"]
    if rejected:
        lines += ["", "## Gate-rejected", "", "| design | reason |", "|---|---|"]
        lines += [f"| `{r.label}` | {r.status} |" for r in rejected]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")

    top = ok[: args.top]
    payload = {"top": [asdict(r) for r in top], "all": [asdict(r) for r in results]}
    args.out.with_suffix(".json").write_text(json.dumps(payload, indent=2))
    print(f"[sweep] wrote {args.out} and {args.out.with_suffix('.json')}")
    print(f"[sweep] TOP {len(top)}:")
    for r in top:
        print(f"    {r.label}  peak {r.peak_cos:+.3f}  held={r.held}  score {r.score:+.3f}")


if __name__ == "__main__":
    main()
