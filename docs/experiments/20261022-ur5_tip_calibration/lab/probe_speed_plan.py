"""Probe-mode plans that press existing targets of a seat's program plan again at other contact speeds (owner, 11-02 15:0x:
"varying speed of contact might matter more"). Only the plan's speed, force levels and hold change: the rests, the push
lines and max_depth_mm are the targets' own, as the workstation's replay checked them, so no new path is made.

  speed  12 normal presses (the reference, 4 face, 4 flank (2 a side), 3 edge points), each at 0.05, 0.15, 0.6 and 2.0 mm/s
         of contact (plan v 0.000167, 0.0005, 0.002, 0.00667 at the 0.30 slider), the order of the speeds turned by one
         per point so that speed and time do not go together; one ramp to 3 N (2 N on flanks and tips) held 1 s
  fast   every grid action again (twists excepted) at plan v 0.002 (0.6 mm/s, 4 times the grid's), levels as planned

Each runs on the seat's own session (--resume its folder, --stages preflight,grid): its registration, no pass 1. The first
press at a point travels the joint way (back safe_mm, a joint move, in); the repeats at the same point the straight way.

    /usr/bin/python3 docs/experiments/20261022-ur5_tip_calibration/lab/probe_speed_plan.py PLAN {speed,fast} OUT
"""
import json
import sys

SLIDER = 0.30
SPEEDS = ((0.05, "v005"), (0.15, "v015"), (0.6, "v060"), (2.0, "v200"))    # mm/s of contact at the slider


def pick(acts, region, k, side=None):
    rows = [a for a in acts if a["region"] == region and a["ws_kind"] == "press"
            and (side is None or (a["target"]["n"][1] > 0) == (side > 0))]
    rows.sort(key=lambda a: a["i"])
    if len(rows) <= k:
        return rows
    return [rows[round(j * (len(rows) - 1) / (k - 1))] for j in range(k)] if k > 1 else [rows[len(rows) // 2]]


def speed_set(plan):
    grid = [a for a in plan["actions"] if a["stage"] == "grid"]
    ref = next(a for a in grid if a["ws_kind"] == "reference")
    pts = [ref] + pick(grid, "face", 4) + pick(grid, "flank", 2, +1) + pick(grid, "flank", 2, -1) + pick(grid, "edge", 3)
    out = []
    for j, a in enumerate(pts):
        top = 2.0 if a["region"] in ("flank", "tip") else 3.0
        order = SPEEDS[j % 4:] + SPEEDS[:j % 4]
        base = a["id"].rsplit("-", 1)[0] if a["ws_kind"] == "reference" else a["id"]
        for k, (mm_s, tag) in enumerate(order):
            out.append(dict(a, id=f"{base}-{tag}", ws_kind="speed", v=round(mm_s / 1e3 / SLIDER, 7), levels=[top],
                            stop_N=top, hold_s=1.0, travel_ws="joint" if k == 0 else "straight"))
    return out


def fast_set(plan):
    """The grid's presses in their order; a press that followed a twist travels the joint way, since the replay checked
    its straight way from the twist's end, not from the press before it."""
    out, prev = [], None
    for a in plan["actions"]:
        if a["stage"] != "grid":
            continue
        if a["kind"] == "press":
            out.append(dict(a, id=f"{a['id']}-v060", v=round(0.6 / 1e3 / SLIDER, 7),
                            travel_ws="joint" if prev is not None and prev["kind"] != "press" else a["travel_ws"]))
        prev = a
    return out


def main():
    src, kind, dst = sys.argv[1], sys.argv[2], sys.argv[3]
    plan = json.load(open(src))
    acts = speed_set(plan) if kind == "speed" else fast_set(plan)
    keep = {k: v for k, v in plan.items() if k != "actions"}
    keep["notes"] = [f"2026-11-02, the lab: probe_speed_plan.py {kind} from {src}: {len(acts)} presses at the plan's own targets "
                     "(rests, push lines and max_depth_mm unchanged), run on the seat's session with --resume and --stages "
                     "preflight,grid."] + list(plan.get("notes", []))
    for a in acts:
        a["stage"] = "grid"
    keep["actions"] = acts
    open(dst, "w").write(json.dumps(keep, indent=1))
    print(f"{dst}: {len(acts)} presses ({kind})")


if __name__ == "__main__":
    main()
