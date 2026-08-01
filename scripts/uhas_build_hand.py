#!/usr/bin/env python3
"""MJCF -> URDF -> UHAS sphere/CIK for one morphology, with a readable verdict.

This is the whole offline half of the UHAS integration in one command: it exports the
hand, runs UHAS's sphere construction, and then reports the numbers that say whether the
canonical sphere landed somewhere the hand can actually work.

    .venv-uhas/bin/python scripts/uhas_build_hand.py \
        --mjcf results/uhas/mjcf/hand_m05_....xml --out results/uhas/hands/m05 --figures

Run it with the UHAS venv (.venv-uhas): it needs trimesh + the `tf` shim as well as
mujoco. Skip --figures for sweeps; it roughly triples the runtime.

The verdict fields are the morphology signal, not decoration:

* ``chain_indices`` -- how the hand's fingers land on UHAS's 5 canonical driving planes.
  Well-spread fingers occupy separated slots (a 3-finger hand should look like
  [[0,1],[2],[4]]); fingers crowded into adjacent slots mean the azimuths cluster, and
  the policy's remaining action dimensions drive nothing.
* ``sphere offset`` -- how far the sphere centre sits from the fingertip centroid, in
  radii. The sphere is supposed to sit *in* the grasp workspace; a large offset means the
  hand is reaching across itself to touch it.
* ``lateral span`` -- per-plane theta range from the CIK lookup table. A plane with a
  near-zero span is a finger that cannot move laterally on the sphere at all; an
  implausibly large one (> ~2 rad) usually means the sweep found a degenerate solution.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from morphohand.uhas import export_hand_to_urdf  # noqa: E402

UHAS_PROCESS_DIR = REPO / "docs" / "uhas" / "UHAS_sim" / "process_urdf"
LEAP_REFERENCE_RADIUS = 0.09119  # what the UHAS env calls base_radius


def summarise(cik_path: Path) -> dict:
    """Pull the morphology-relevant facts out of a sphere_cik.json."""
    d = json.loads(cik_path.read_text())
    sphere = d["joint_info"]["sphere_frame"]
    radius = float(sphere[6])
    centre = np.asarray(sphere[0], dtype=float)

    chains = [c for c in d["kin_chains"]
              if "palm_normal" not in c and "sphere_frame" not in c]
    fingers = [c[-1].replace("_ft", "") for c in chains]

    # fingertip positions in base frame at the open pose, via the fingerprint samples
    tips = {}
    for finger, chain in zip(fingers, chains):
        ftj = chain[-1]
        info = d["joint_type_info"].get(ftj, {})
        if "ft" in info:
            tips[finger] = np.asarray(info["ft"], dtype=float)

    used = sorted({i for idx in d["chain_indices"] for i in idx})
    spans = [float(hi) - float(lo)
             for lo, hi in zip(d["min_offsets"], d["max_offsets"])]

    return {
        "fingers": fingers,
        "n_fingers": len(fingers),
        "chain_order": d["chain_order"],
        "chain_indices": d["chain_indices"],
        "slots_used": used,
        "slots_dead": [i for i in range(5) if i not in used],
        "sphere_radius": radius,
        "sphere_radius_vs_leap": radius / LEAP_REFERENCE_RADIUS,
        "sphere_centre": centre.tolist(),
        "lateral_spans": spans,
        "joint_types": {j: i["type"] for j, i in d["joint_type_info"].items()
                        if isinstance(i, dict) and "type" in i},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mjcf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--palm-body", default="palm")
    ap.add_argument("--open-mcp", type=float, default=0.0)
    ap.add_argument("--thumb-anchor", type=float, default=1.571)
    ap.add_argument("--figures", action="store_true",
                    help="also capture the verbose construction figures (slow)")
    ap.add_argument("--skip-export", action="store_true",
                    help="reuse an existing URDF in --out")
    args = ap.parse_args()

    out = Path(args.out)
    name = args.name or out.name
    out.mkdir(parents=True, exist_ok=True)

    if args.skip_export:
        urdf = out / f"{name}.urdf"
        print(f"[export] reusing {urdf}")
    else:
        exp = export_hand_to_urdf(args.mjcf, out, robot_name=name,
                                  palm_body=args.palm_body, open_mcp=args.open_mcp)
        urdf = exp.urdf_path
        print(f"[export] {urdf}  fingers={exp.finger_names}  "
              f"r_est={exp.sphere_radius_estimate:.4f}")

    cmd = [sys.executable]
    if args.figures:
        # process_urdf runs with cwd=UHAS_PROCESS_DIR, so figdir must be absolute or the
        # captures land under the UHAS checkout instead of this hand's folder.
        cmd += [str(REPO / "scripts" / "uhas_process_urdf.py"),
                "--figdir", str((out / "figures").resolve())]
    else:
        cmd += [str(UHAS_PROCESS_DIR / "process_urdf.py")]
    cmd += ["--robot_path", str(urdf.resolve()),
            "--base_link", args.palm_body,
            "--thumb_anchor", str(args.thumb_anchor),
            "--correct_axes"]
    if args.figures:
        cmd += ["--verbose"]

    env = dict(os.environ, MPLBACKEND="Agg", PYVISTA_OFF_SCREEN="true")
    log = out / "process_urdf.log"
    print(f"[uhas]   running process_urdf -> {log}")
    with log.open("w") as fh:
        r = subprocess.run(cmd, cwd=UHAS_PROCESS_DIR, env=env, stdout=fh,
                           stderr=subprocess.STDOUT)
    if r.returncode != 0:
        print(f"[uhas]   FAILED (exit {r.returncode}); tail of {log}:")
        print("\n".join(log.read_text().splitlines()[-25:]))
        return r.returncode

    cik = urdf.parent / "sphere_cik.json"
    if not cik.exists():
        print(f"[uhas]   no sphere_cik.json at {cik}")
        return 1

    s = summarise(cik)
    (out / "uhas_summary.json").write_text(json.dumps(s, indent=2))

    print()
    print(f"=== UHAS representation: {name} ===")
    print(f"  fingers        : {s['n_fingers']}  {s['fingers']}")
    lat = [j for j, t in s["joint_types"].items() if t == "A"]
    print(f"  lateral joints : {lat}")
    print(f"  chain_indices  : {s['chain_indices']}   (driving-plane slots per finger)")
    print(f"  slots dead     : {s['slots_dead']}  "
          f"-> {2 * len(s['slots_dead'])} of 15 policy action dims drive nothing")
    print(f"  sphere radius  : {s['sphere_radius']:.4f} m "
          f"({s['sphere_radius_vs_leap']:.2f}x LEAP)")
    print(f"  lateral spans  : {[round(x, 3) for x in s['lateral_spans']]}")
    print(f"  summary        : {out / 'uhas_summary.json'}")
    if args.figures:
        n = len(list((out / "figures").glob("*.png")))
        print(f"  figures        : {out / 'figures'} ({n} png)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
