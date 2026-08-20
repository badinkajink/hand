#!/usr/bin/env python3
"""Write a frozen scene with one physical axis moved — as a FILE, so it can be trained on.

`morphohand.studies.scene_mutate.Scene` already perturbs a scene along a single physical
axis, but every caller so far (sim2real_robustness_sweep, fingertip_policy_sweep,
probe_fingertip_mechanics) builds the mutated scene in-process and EVALUATES a policy on
it. Training on one needs the scene to exist on disk so `--frozen-scene-xml` can point at
it, and there was no way to do that from a shell.

That gap is why the friction cliff is still open. The μ×0.5 cliff was only ever attacked
by finetuning b33 under domain randomisation, which starts inside the high-friction basin
— the wrong initialisation for what looks like a hard-exploration problem. Training from
scratch AT μ×0.5 asks a different question, and needs this.

  scripts/mutate_scene.py --scene <frozen.xml> --out /tmp/mu05.xml --friction 0.5
  scripts/mutate_scene.py --scene <frozen.xml> --out /tmp/hard.xml --solimp-dmax 0.997
  scripts/mutate_scene.py --scene <frozen.xml> --out /tmp/thin.xml --object-radius 0.85

Mass is pinned across geometry edits (scene_mutate's own guarantee), and --check reports
the before/after mass delta so a geometry change can never become a silent mass change.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from morphohand.studies.scene_mutate import Scene, mass_check  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True, type=Path, help="frozen scene to mutate")
    ap.add_argument("--out", required=True, type=Path, help="where to write the mutated scene")
    ap.add_argument("--friction", type=float, help="scale the pad/object SLIDING coefficient")
    ap.add_argument("--solimp-dmax", type=float, help="contact stiffness; higher = harder")
    ap.add_argument("--solimp-dmin", type=float, default=0.9)
    ap.add_argument("--object-radius", type=float, help="scale the shaft radius")
    ap.add_argument("--object-density", type=float, help="scale the tool mass")
    ap.add_argument("--tip-shape", help="fingertip shape name (e.g. cap_cross, sphere)")
    ap.add_argument("--tip-r", type=float, default=0.005)
    ap.add_argument("--tip-h", type=float, default=0.006)
    ap.add_argument("--check", action="store_true", help="report the before/after mass delta")
    args = ap.parse_args()

    if not args.scene.is_file():
        print(f"!! no such scene: {args.scene}", file=sys.stderr)
        return 1

    s = Scene(args.scene)
    applied = []
    if args.friction is not None:
        s.scale_friction(args.friction); applied.append(f"friction x{args.friction}")
    if args.solimp_dmax is not None:
        s.set_solimp(args.solimp_dmin, args.solimp_dmax)
        applied.append(f"solimp {args.solimp_dmin}/{args.solimp_dmax}")
    if args.object_radius is not None:
        s.scale_object_radius(args.object_radius); applied.append(f"radius x{args.object_radius}")
    if args.object_density is not None:
        s.scale_object_density(args.object_density); applied.append(f"density x{args.object_density}")
    if args.tip_shape:
        s.set_tip_shape(args.tip_shape, r=args.tip_r, h=args.tip_h)
        applied.append(f"tip {args.tip_shape} r{args.tip_r} h{args.tip_h}")

    if not applied:
        print("!! no mutation requested — refusing to write an identical copy", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    s.write(args.out)
    print(f"{args.scene} -> {args.out}")
    for a in applied:
        print(f"  {a}")

    if args.check:
        d = mass_check(args.scene, args.out)
        print(f"  mass check: {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
