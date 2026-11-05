#!/usr/bin/env python3
"""Job queue of the fingertip-shape rerun (docs/handoff/20261103-fingertip_shape_rerun.md), read by the workers
logs/<run>/queue/worker.sh (one job line at a time under flock; each job through resguard and taskset on the worker's core).

  screen   stages 1 and 2 of 10-14 and 10-15 for the twelve tips: per condition the four cad_v2 cells (HOM seeds 0-2; the
           hold test at friction 0.7 and 0.5) and HOM on D1-D8 (seeds 0-4); the D1-D8 replays of the Drake agreement gate
           (the pads at the planner's and the RL trainer's numerics, and Drake, seeds 0-2); ordered so that J1 (friction
           0.7 on the SCS0009) and the gate finish first
  stage3   fresh placements for the stage-3 tips (screen/summary.json): cells seeds 3-5 and D1-D8 seeds 5-9 at the four
           conditions other than friction 1.0
  bed      the contact bed per tip (scripts/fingertip/fingertip_shape_rerun_bed.py): the law, then the pads, then Drake

Job lines:
  scale <tip> <mu> <plant> <what> <seeds, comma-separated> <out dir>
  hom <tip> <mu> <plant> <seeds, comma-separated> <out file>
  traj <tip> <arms, comma-separated> <hands, comma-separated> <seeds, comma-separated> <out file>
  bed <tip> <models, comma-separated> <cases, comma-separated>
  law <tip>

    python3 scripts/fingertip/fingertip_shape_rerun_queue.py screen [--append]
"""
import argparse
import fcntl
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path
import fingertip_shape_rerun_common as C  # noqa: E402

Q = C.LOGS / "queue" / "queue.txt"
HANDS = [f"D{i}" for i in range(1, 9)]


def screen_jobs():
    S = C.D / "screen"
    jobs = []
    for i, (mu, pl) in enumerate(C.CONDS):
        for k in C.TIPS:
            what = "hold,turn" if (mu, pl) in (("0.7", C.SERVO), ("0.5", C.SERVO)) else "turn"
            jobs.append(f"scale {k} {mu} {pl} {what} 0,1,2 {S}/scale/{k}_mu{mu}_{pl}")
            jobs.append(f"hom {k} {mu} {pl} 0,1,2,3,4 {S}/hom/{k}_mu{mu}_{pl}.jsonl")
        if i == 0:                                   # the gate's replays after J1's condition
            for k in C.TIPS:
                jobs.append(f"traj {k} plan,rl {','.join(HANDS)} 0,1,2 {C.D}/traj/{k}/plan_rl.jsonl")
                for h in HANDS:
                    jobs.append(f"traj {k} drake {h} 0,1,2 {C.D}/traj/{k}/drake_{h}.jsonl")
    return jobs


def stage3_jobs():
    S, F = C.D / "screen", C.D / "final"
    tips = json.loads((S / "summary.json").read_text())["stage3_tips"]
    jobs = []
    for mu, pl in C.CONDS:
        if (mu, pl) == ("1.0", C.SERVO):
            continue
        for k in tips:
            jobs.append(f"scale {k} {mu} {pl} turn 3,4,5 {F}/scale/{k}_mu{mu}_{pl}")
            jobs.append(f"hom {k} {mu} {pl} 5,6,7,8,9 {F}/hom/{k}_mu{mu}_{pl}.jsonl")
    return jobs


def bed_jobs():
    jobs = [f"law {k}" for k in C.BED]
    jobs += [f"bed {k} pads static,T2,spin,T5" for k in C.BED]
    jobs += [f"bed {k} drake static,T2,spin,T5" for k in C.BED]
    return jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["screen", "stage3", "bed"])
    ap.add_argument("--queue", type=Path, default=Q)
    ap.add_argument("--append", action="store_true")
    a = ap.parse_args()
    jobs = {"screen": screen_jobs, "stage3": stage3_jobs, "bed": bed_jobs}[a.what]()
    a.queue.parent.mkdir(parents=True, exist_ok=True)
    with open(a.queue.parent / "queue.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        with open(a.queue, "a" if a.append else "w") as fh:
            fh.write("\n".join(jobs) + "\n")
    print(len(jobs), "jobs", "appended to" if a.append else "->", a.queue)


if __name__ == "__main__":
    main()
