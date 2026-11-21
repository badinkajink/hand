#!/usr/bin/env python3
"""Trial losses, putdowns and pick-ups of the bench series of 2026-11-20 (hand-4a, 16:2x; owner, 16:1x: "we should get
some metrics on the pickup, drop rate incidence, pick up success, push->pickup mitigation").

Sources, copied from corvus's logs/20261116-hom_trials/ into runs_corvus/ (rsync): hom_series's rows
(20261120-series_morning_seed20261120.jsonl), the series log (20261120-day_series.log) and the automated pick-up's logs
(20261120-*_pickup.log).
    trial ending   a row's "stop": "controller" when the trial ran to the schedule's end, else corvus's drift guard
                   (the sphere 30 mm or more from its start)
    putdown        a row's "putdown": "LET GO" (also after a stopped levelling) is a clean release; "tilted ..." a
                   release of a tilted sphere; "dropped ..." a sphere lost in the trial; anything else a putdown that
                   stopped and was finished by the lab session
    pick-up        a pinch grasp of a sphere lying on the table: each automated attempt's "PICKUP RESULT" line, with a
                   push first when push_sphere.py ran before it in the same log; and the attempts the lab session ran
                   before the automation (14:04), listed in MANUAL with the series log's words, and those the
                   lab session ran itself later (MANUAL_LATE), the blind pinches of 17:54 and 19:34 among them
    planner refusal  a pick-up that found no plan (the pinch's clearances or the wrist's 45 deg limit)
Rates are counts over counts; with this few events the interval of each is wide, and the counts are given beside them.

    python3 docs/experiments/20261115-hom_bench_trials/analysis/20261120-recovery_metrics.py
Writes 20261120-recovery_metrics.json next to this script.
"""
import json
import re
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "runs_corvus"
ROWS = RUNS / "20261120-series_morning_seed20261120.jsonl"
OUT = HERE / "20261120-recovery_metrics.json"
# the pick-up attempts before the automation (14:04), from the series log's restart lines and the lab's messages
MANUAL = [
    {"t": "11:5x", "push": False, "ok": None, "refused": True, "on_cables": None,
     "words": "A1_r3_x115y160g8_t3 toppled and the pick-up failed: no pick-up plan keeps 8 mm off the table"},
    {"t": "12:4x", "push": False, "ok": False, "refused": False, "on_cables": True,
     "words": "at the (105, 70) pickup layout it plans (thumb-index, 9 mm above the table), but twice a fingertip came "
              "down on the ball (first of two)"},
    {"t": "12:4x", "push": False, "ok": False, "refused": False, "on_cables": True,
     "words": "twice a fingertip came down on the ball (second of two)"},
    {"t": "13:31", "push": True, "ok": True, "refused": False, "on_cables": False,
     "words": "the dropped sphere pushed off the cables (push_sphere.py, middle fingertip, 50 mm along the stem, 13:28) "
              "and picked up by the pinch (13:31-13:34, settled 0.4 deg)"},
    {"t": "13:3x", "push": False, "ok": None, "refused": True, "on_cables": None,
     "words": "A1_p25_x85y170g2_t2 dropped the object ... no pick-up plan (the stem 47.5 deg off, the wrist's limit 45)"},
    {"t": "13:56", "push": True, "ok": True, "refused": False, "on_cables": False,
     "words": "recovered at (105, 70) by a push (19 mm across, then on bare table) and the pinch pick-up (13:56-13:59)"},
]
# after the automation: attempts the lab session finished by its own commands, from the restart lines
MANUAL_LATE = [
    {"t": "14:58", "push": True, "ok": True, "refused": False, "on_cables": False,
     "words": "the automated pick-up's push and grasp met the cables three times; pushed off them at 5 N and picked up "
              "by hand 14:55-15:01 (settled 0.6 deg)"},
    {"t": "17:54", "push": False, "ok": True, "refused": False, "on_cables": None, "blind": True,
     "words": "owner 17:5x: 'lets just fly blind here and go for the pinch'; pinch_pickup.py grasp --blind: the grasp went "
              "124 mm down with at most 1.6 N; the putdown left the sphere 0.7 deg from vertical (lab, 17:5x)"},
    {"t": "19:34", "push": False, "ok": True, "refused": False, "on_cables": None, "blind": True,
     "words": "the watched pick-up lost the sphere 72.5 deg into an 84.9 deg wrist turn; the blind pinch then stood it up "
              "(0.7 deg) (lab, 19:49)"},
]


def putdown_class(s):
    s = str(s or "")
    if s.startswith("LET GO") or s.endswith("LET GO"):
        return "clean"
    if s.startswith("tilted"):
        return "tilted release"
    if s.startswith("dropped"):
        return "lost in the trial"
    return "stopped, finished by the session"


def automated():
    """[{t, push, ok, log}] of the automated pick-up logs, one per PICKUP RESULT line."""
    out = []
    for p in sorted(RUNS.glob("20261120-*_pickup.log")):
        push = False
        for line in p.read_text(errors="replace").splitlines():
            if "push_sphere.py" in line:
                push = True
            m = re.match(r"PICKUP RESULT: (.*)", line.strip())
            if m:
                out.append({"t": p.name[9:13], "push": push, "ok": m.group(1).startswith("OK"),
                            "result": m.group(1)[:100], "log": p.name})
                push = False
    return out


def rate(k, n):
    return {"count": k, "of": n, "share": round(k / n, 3) if n else None}


def main():
    rows = [json.loads(x) for x in ROWS.read_text().splitlines() if x.strip()]
    n = len(rows)
    drift = [r for r in rows if r.get("stop") != "controller"]
    pc = {}
    for r in rows:
        pc.setdefault(putdown_class(r.get("putdown")), []).append(r["label"])
    auto = automated()
    grasps = [e for e in MANUAL + MANUAL_LATE if not e["refused"]] + auto
    refused = [e for e in MANUAL if e["refused"]]
    with_push = [e for e in grasps if e["push"]]
    no_push = [e for e in grasps if not e["push"]]
    blind = [e for e in grasps if e.get("blind")]
    out = {"what": __doc__.split("\n\n")[0].replace("\n", " "), "generated": time.strftime("%F %H:%M"),
           "trials": n,
           "sphere_lost_in_trial": rate(len(drift), n) | {"labels": [r["label"] for r in drift]},
           "putdowns": {k: rate(len(v), n) | {"labels": v} for k, v in pc.items()},
           "pickup_grasps": rate(sum(1 for e in grasps if e["ok"]), len(grasps)),
           "pickup_with_push_first": rate(sum(1 for e in with_push if e["ok"]), len(with_push)),
           "pickup_without_push": rate(sum(1 for e in no_push if e["ok"]), len(no_push)),
           "pickup_blind": rate(sum(1 for e in blind if e["ok"]), len(blind)),
           "pickup_planner_refusals": len(refused),
           "attempts": {"manual": MANUAL + MANUAL_LATE, "automated": auto}}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"{n} trials; sphere lost in the trial (drift guard) {len(drift)} ({100 * len(drift) / n:.0f} %)")
    for k, v in pc.items():
        print(f"  putdown {k}: {len(v)} ({100 * len(v) / n:.0f} %)")
    for k in ("pickup_grasps", "pickup_with_push_first", "pickup_without_push", "pickup_blind"):
        print(f"  {k}: {out[k]['count']} of {out[k]['of']}")
    print(f"  planner refusals: {len(refused)}; automated attempts {len(auto)}: "
          + ", ".join(f"{a['t']} {'push+' if a['push'] else ''}{'OK' if a['ok'] else 'failed'}" for a in auto))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
