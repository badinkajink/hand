"""The night's rod plans in halves (the workstation's pick of 19:0x on 11-03, under the owner's go of 16:3x): seat 2's
sweeps on the robot take 258 s each, so the _fit plans would end seat 2 at about 23:40 and leave seat 1 its pass 1 and
about 10 sweeps before the 05:50 stop. Each seat keeps every sweep location of its _fit plan, lighter:
  - offsets at azimuths 0, 90, 180 and 270 only, every ring and level kept (32 of 64);
  - slides at az 0 and 90 at 2 and 4 N (4 of 8; each comes back along the opposite direction);
  - twists at 5 N only (the -2.5N repeats out), after the second half's sweeps;
  - the 4 N rolls of the seat's extras (fca3d1cb) at the very end, after the last reference press.
Half a holds every other sweep in the plan's order (sorted by the stem's direction, so each half spans the pad and both
rod turns): probe-ref-00, sweeps 0, 2, 4, ..., probe-ref-a1. Half b: probe-ref-b0, sweeps 1, 3, 5, ..., the twists,
probe-ref-final, the rolls. The reference presses are probe-ref-00's press under new ids, so that a resumed session
runs each. The run order is 2a (seat 2's session, from where it stands), 3a and 1a (each a new session with its
pass 1), then 1b, 3b and 2b, each resuming its seat's session.

Then each seat's extras (the workstation's addition, relayed at 19:1x: the owner is in at 09:00, the hard stop moves
to 08:30): the extras of fca3d1cb (seat 1's flank-582 obliques out) as _extras_night, in the order 2, 3, 1. Each holds
probe-ref-x0, the rolls at 4 and 1.5 N (a 4 N roll that half b ran is skipped as done), the 20 and 35 deg obliques at
azimuths 0 and 180, those at 90 and 270, and probe-ref-x1.

Writes the nine plans here, and for the path check copies under CHECK_DIR without the pad stage where it does not
run, without the ids DONE in seat 2's session, and without the 4 N rolls in the extras.

    python3 20261103-rod_halves.py CHECK_DIR [DONE,...]
"""
import copy
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEATS = ((2, 5), (3, 4), (1, 6))
KEEP_OFF = {0.0, 90.0, 180.0, 270.0}
KEEP_SLIDE = {0.0, 90.0}
NOTE = time.strftime("%Y-%m-%d %H:%M") + ", lab: " + __doc__.split("\n\n")[0].replace("\n", " ")


def lite(a):
    a = copy.deepcopy(a)
    a["offsets"] = [[o for o in lv if float(o[1]) % 360 in KEEP_OFF] for lv in a["offsets"]]
    a["slides"] = [s for s in a["slides"] if float(s["az_deg"]) % 360 in KEEP_SLIDE]
    return a


def ref(a, i):
    a = copy.deepcopy(a)
    a["id"] = i
    return a


def main():
    chk = Path(sys.argv[1])
    chk.mkdir(parents=True, exist_ok=True)
    done = set(sys.argv[2].split(",")) if len(sys.argv) > 2 and sys.argv[2] else set()
    for seat, vid in SEATS:
        stem = f"20261103-plan_probe_table_rod_seat{seat}_v3id{vid}"
        fit = json.loads((HERE / f"{stem}_sweep8_fit.json").read_text())
        ext = json.loads((HERE / f"{stem}_extras.json").read_text())
        grid = [a for a in fit["actions"] if a["stage"] == "grid"]
        r0 = [a for a in grid if a["id"] == "probe-ref-00"][0]
        sweeps = [lite(a) for a in grid if a["kind"] == "sweep"]
        twists = [a for a in grid if a["kind"] == "twist" and not a["id"].endswith("-2.5N")]
        rolls = [a for a in ext["actions"] if a["stage"] == "grid" and a["kind"] == "roll" and float(a["press_N"]) == 4.0]
        halves = dict(a=[r0] + sweeps[0::2] + [ref(r0, "probe-ref-a1")],
                      b=[ref(r0, "probe-ref-b0")] + sweeps[1::2] + twists + [ref(r0, "probe-ref-final")] + rolls)
        xg = [a for a in ext["actions"] if a["stage"] == "grid"]
        obl = [a for a in xg if a["kind"] == "press" and not a["id"].startswith("probe-ref")]
        first = [a for a in obl if float(a["azimuth"]) % 180 == 0]
        halves["x"] = ([ref([a for a in xg if a["id"] == "probe-ref-00"][0], "probe-ref-x0")]
                       + [a for a in xg if a["kind"] == "roll"] + first + [a for a in obl if a not in first]
                       + [ref([a for a in xg if a["id"] == "probe-ref-final"][0], "probe-ref-x1")])
        for h, g in halves.items():
            p = copy.deepcopy(ext if h == "x" else fit)
            p["actions"] = [a for a in p["actions"] if a["stage"] != "grid"] + g
            n_sw = sum(a["kind"] == "sweep" for a in g)
            p["notes"].append(NOTE + (f" The extras of seat {seat}" if h == "x" else f" Half {h} of seat {seat}")
                              + f": {len(g)} grid actions ({n_sw} sweeps, {sum(a['kind'] == 'twist' for a in g)} twists, "
                              f"{sum(a['kind'] == 'roll' for a in g)} rolls, {sum(a['kind'] == 'press' for a in g)} presses) "
                              + (f"from {stem}_extras.json." if h == "x" else
                                 f"from {stem}_sweep8_fit.json and the rolls of {stem}_extras.json."))
            dst = HERE / (f"{stem}_extras_night.json" if h == "x" else f"{stem}_half_{h}.json")
            dst.write_text(json.dumps(p, indent=1))
            c = copy.deepcopy(p)                    # the path check's copy: the stages that run, less the done ids
            runs_pad = h == "a" and seat != 2
            ran = {a["id"] for a in rolls} if h == "x" else set()       # half b's 4 N rolls, done when the extras run
            c["actions"] = [a for a in c["actions"] if (a["stage"] == "grid" and a["id"] not in ran
                                                        and not (seat == 2 and a["id"] in done))
                            or (runs_pad and a["stage"] == "pad")]
            (chk / dst.name).write_text(json.dumps(c, indent=1))
            print(f"{dst.name}: {len(g)} grid actions ({n_sw} sweeps), offsets "
                  f"{sum(len(x) for a in g if a['kind'] == 'sweep' for x in a['offsets'])}, slides "
                  f"{sum(len(a['slides']) for a in g if a['kind'] == 'sweep')}; the check's copy {len(c['actions'])} actions"
                  + (" with pass 1" if runs_pad else ""))


if __name__ == "__main__":
    main()
