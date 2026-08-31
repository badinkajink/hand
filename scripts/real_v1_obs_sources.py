#!/usr/bin/env python3
"""Every column of the policy's observation, and what on the bench could supply it.

    python3 scripts/real_v1_obs_sources.py \
        --scene assets/mjcf/real_v1/hand_frozen_morphology.xml \
        --checkpoint results/rl/20260828-0052-policyB_rv03_narrowy_reorient/tensorboard/model_200.pt

WHY THIS EXISTS.  `scripts/probe_obs_ablation.py` grouped three observation terms under the
name "hidden", with the comment "there is no object tracker, so anything derived from the
object's pose is hidden".  That was true when it was written and stopped being true on
2026-08-31, when two AprilTags and a set of tape-measure readings became
`morphohand.bench.tags`.  Eleven of the sixty-six columns changed status in one afternoon,
and the question "could we close the loop on hardware now?" deserves a column-by-column
answer rather than a recollection.

The widths are derived, not transcribed: `joint_pos`/`joint_vel` are one entry per actuated
joint in the frozen scene, and the rest are fixed by their term functions.  Pass a
checkpoint and the total is checked against the actor's own input width, which is the only
way to be sure this table describes the policy you are about to deploy.

READ THE VERDICT, NOT JUST THE TABLE.  "Supplyable" is a statement about the OBSERVATION,
and an observation is the cheapest of the things a closed loop needs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# name -> (width or "per actuated joint", where it comes from on the bench, status)
#
#   measured   an instrument on the bench reports it
#   replayable it is a function of the step index and the frozen plan, so it can be
#              played back from a file -- known, but carrying no feedback
#   internal   the controller already has it; it is its own previous output
#   absent     nothing on the bench produces it
TERMS = [
    ("joint_pos", "nu",
     "servo present-position, sync-read at ~50 Hz.  The 6 palm-pose columns are the arm, "
     "which the bench holds fixed, so they are constants -- and the 9 finger columns arrive "
     "at 0.44-0.90 of what was commanded on the yaw joints, which is the plant difference, "
     "not a sensing gap.", "measured"),
    ("joint_vel", "nu",
     "differenced servo positions.  Available but NOISY: the SCS0009 has no velocity "
     "register, the position quantum is 0.29 deg, and at 50 Hz that is 14.6 deg/s of "
     "quantisation on a signal the policy trained on as exact.", "measured"),
    ("object_pos", 3,
     "AprilTag: palm-frame vector to the cylinder centre.  NEW 2026-08-31.", "measured"),
    ("object_pose_actual", 7,
     "AprilTag: cylinder pose in the palm frame, position + quaternion.  NEW 2026-08-31. "
     "The tag gives full 6-DoF, so the quaternion is real and not a reconstruction -- but "
     "the bench x/y half of the position needs the heading calibration.", "measured"),
    ("ref_finger_qpos", 9,
     "the frozen CEM reference trajectory, indexed by step.  A file.", "replayable"),
    ("ref_object_pose", 7,
     "the same frozen reference.  A file.", "replayable"),
    ("actions", 9, "the controller's own previous output.", "internal"),
    ("target_axis_misalign", 1,
     "AprilTag: arccos of the shaft's alignment with world up, which is exactly what the "
     "tracker's cos column already is.  NEW 2026-08-31.", "measured"),
]

WAS_HIDDEN = {"object_pos", "object_pose_actual", "target_axis_misalign"}


def actuated(scene: Path) -> int:
    import mujoco
    return int(mujoco.MjModel.from_xml_path(str(scene)).nu)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", type=Path,
                    default=ROOT / "assets/mjcf/real_v1/hand_frozen_morphology.xml")
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="check the derived total against this actor's input width")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    nu = actuated(a.scene)
    rows, off = [], 0
    for name, width, source, status in TERMS:
        w = nu if width == "nu" else int(width)
        rows.append({"term": name, "columns": [off, off + w], "width": w,
                     "status": status, "source": source,
                     "was_hidden_before_20260831": name in WAS_HIDDEN})
        off += w
    total = off

    print(f"scene {a.scene.relative_to(ROOT) if a.scene.is_relative_to(ROOT) else a.scene}"
          f"  ({nu} actuated joints)\n")
    print(f"{'term':<22} {'cols':>9} {'w':>3}  {'status':<11} ")
    for r in rows:
        print(f"{r['term']:<22} {r['columns'][0]:3d}-{r['columns'][1] - 1:<5d} {r['width']:3d}  "
              f"{r['status']:<11} {'** was hidden **' if r['was_hidden_before_20260831'] else ''}")
    print(f"{'TOTAL':<22} {total:>9}")

    ok = None
    if a.checkpoint:
        from morphohand.rl.deploy import ckpt_obs_dim
        got = ckpt_obs_dim(a.checkpoint)
        ok = got == total
        print(f"\nactor input width from {a.checkpoint.name}: {got}  "
              f"-> {'MATCHES' if ok else 'DOES NOT MATCH'} the derived {total}")
        if not ok:
            print("  the term list here is out of step with _build_observations; fix it "
                  "before quoting any of the above.")

    by = {}
    for r in rows:
        by.setdefault(r["status"], 0)
        by[r["status"]] += r["width"]
    newly = sum(r["width"] for r in rows if r["was_hidden_before_20260831"])
    print(f"\n{'measured on the bench':<28} {by.get('measured', 0):3d} columns")
    print(f"{'replayable from a file':<28} {by.get('replayable', 0):3d}")
    print(f"{'the controller already has':<28} {by.get('internal', 0):3d}")
    print(f"{'nothing supplies':<28} {by.get('absent', 0):3d}")
    print(f"\n{newly} of {total} columns changed from absent to measured on 2026-08-31.")
    print("Every column of this observation now has a bench source. That is the OBSERVATION\n"
          "problem solved, and it is the smallest of the four:\n"
          "  1. rate      the tags run at 30 Hz and the policy at 50; the servo bus sustains\n"
          "               ~111 Hz of sync writes but sync READS do not work on the SCS0009 at\n"
          "               all, so joint_pos costs individual reads.\n"
          "  2. plant     the yaw joints arrive at 0.44-0.90 of what they are told (20 bench\n"
          "               runs, 2026-08-30). A policy trained where commanded == achieved is\n"
          "               driving a different hand, feedback or no feedback.\n"
          "  3. the policy b33 was shown to IGNORE its observations -- replaying the whole\n"
          "               66-dim input from another rollout cost nothing. Handing a policy\n"
          "               that ignores its inputs a real sensor changes nothing about what it\n"
          "               does. The tags make a SIGHTED policy trainable; they do not make an\n"
          "               open-loop one closed.\n"
          "  4. safety    nothing gates a streamed policy against finger-finger collision,\n"
          "               and three of four deployed designs interpenetrate along their own\n"
          "               planned path.")

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"scene": str(a.scene), "actuated_joints": nu,
                                     "total": total, "checkpoint_matches": ok,
                                     "terms": rows}, indent=2) + "\n")
        print(f"\nwrote {a.out}")
    return 0 if ok is not False else 1


if __name__ == "__main__":
    raise SystemExit(main())
