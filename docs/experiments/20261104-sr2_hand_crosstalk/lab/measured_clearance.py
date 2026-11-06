"""CAD clearance (handgeo's cad_v2 spheres) at the measured poses of run_plan.py poses chunks, at the layout's mounts (the
layout from the run dir's name, k_<layout>_poses). For each run: the least clearance over the dwell means (the servos
between t_settled and t_end, in the sim frame) and over every telemetry sample, moves included; the dwells under WARN mm.
With --follow (one run dir), it keeps reading a live run and prints each dwell under WARN as it settles, and a line every
30th step; it exits at the run's end event. --dwells-only skips the samples (each costs about 12 ms); --out writes the
dwells as JSON.

    nice uv run --no-project --python 3.10 --with mujoco --with numpy --with scipy python measured_clearance.py \\
        RUN_DIR [RUN_DIR ...] [--warn 5] [--follow] [--dwells-only] [--out DWELLS.json]
"""
import argparse
import json
import sys
import time
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
from make_layouts import LAYOUTS, h, local  # noqa: E402

FID = {"thumb": "0", "index": "1", "middle": "2"}
YAW_SIGN = {"thumb": 1.0, "index": -1.0, "middle": -1.0}      # run_plan.py's
F = ("thumb", "index", "middle")
ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--warn", type=float, default=5.0)
ap.add_argument("--follow", action="store_true")
ap.add_argument("--dwells-only", action="store_true")
ap.add_argument("--out")
a = ap.parse_args()


def sim(sv):
    return {f: {"yaw": YAW_SIGN[f] * sv[FID[f]]["aa"], "mcp": sv[FID[f]]["fe1"], "pip": sv[FID[f]]["fe2"]} for f in F}


def least(palm, pose):
    h.set(local(palm), pose)
    c = h.clearances()
    k = min(c, key=c.get)
    return c[k], k


class Lines:
    """A jsonl file read in whole lines from where the last call stopped."""
    def __init__(self, path):
        self.path, self.pos, self.buf = path, 0, ""

    def new(self):
        if not self.path.exists():
            return []
        with open(self.path) as f:
            f.seek(self.pos)
            data = f.read()
            self.pos = f.tell()
        self.buf += data
        *done, self.buf = self.buf.split("\n")
        return [json.loads(x) for x in done if x.strip()]


def run(rd):
    layout = rd.name.split("_", 1)[1].removesuffix("_poses")
    palm = LAYOUTS[layout]
    ev, tel = Lines(rd / "events.jsonl"), Lines(rd / "telemetry.jsonl")
    samples, dwells, low, worst_s, pending, latest = [], [], [], (1e9, None, None), [], 0.0
    while True:
        for s in tel.new():
            if s.get("servos"):
                try:
                    pose = sim(s["servos"])
                except (KeyError, TypeError):
                    continue
                samples.append((s["t_sample"], pose))
                latest = max(latest, s["t_sample"])
                if not a.dwells_only:
                    c, k = least(palm, pose)
                    if c < worst_s[0]:
                        worst_s = (c, k, s["t_sample"])
        end = False
        for e in ev.new():
            end |= e["kind"] == "end"
            if e["kind"] == "step" and e["op"] == "joints":
                pending.append(e)
        # a dwell waits until the telemetry has caught up with its end
        while pending and (pending[0]["t_end"] <= latest or end or not a.follow):
            e = pending.pop(0)
            inside = [p for t, p in samples if e["t_settled"] <= t <= e["t_end"]]
            samples = [x for x in samples if x[0] >= e["t_end"] - 5]
            if not inside:
                continue
            mean = {f: {j: sum(p[f][j] for p in inside) / len(inside) for j in ("yaw", "mcp", "pip")} for f in F}
            c, k = least(palm, mean)
            dwells.append((c, k, e["step"], e.get("clearance_mm"), e["family"], mean))
            if c < a.warn:
                low.append(dwells[-1])
                if a.follow:
                    print(time.strftime("%H:%M:%S"), f"LOW {rd.name} step {e['step']}: {c:.1f} mm {k} measured"
                          f" (plan {e.get('clearance_mm')} mm)", flush=True)
            if a.follow and len(dwells) % 30 == 0:
                m = min(dwells, key=lambda d: d[0])
                print(time.strftime("%H:%M:%S"), f"{rd.name}: {len(dwells)} dwells, least {m[0]:.1f} mm {m[1]} at step"
                      f" {m[2]}", flush=True)
        if (end and not pending) or not a.follow:
            break
        time.sleep(1.0)
    if dwells:
        m = min(dwells, key=lambda d: d[0])
        moves = "" if a.dwells_only else (f"; every sample, moves included: least {worst_s[0]:.1f} mm {worst_s[1]} at "
                                          f"{time.strftime('%H:%M:%S', time.localtime(worst_s[2]))}")
        print(f"{rd.name}: {len(dwells)} dwells, least {m[0]:.1f} mm {m[1]} at step {m[2]} (plan {m[3]} mm); "
              f"{sum(d[0] < 8 for d in dwells)} under 8 mm, {len(low)} under {a.warn:g}{moves}", flush=True)
    for d in sorted(low, key=lambda d: d[0])[:10]:
        print(f"  step {d[2]}: {d[0]:.1f} mm {d[1]} (plan {d[3]} mm)")
    return [{"run": rd.name, "layout": layout, "step": d[2], "family": d[4], "clearance_mm": round(d[0], 2), "pair": d[1],
             "plan_clearance_mm": d[3], "measured": {f: {j: round(v, 2) for j, v in d[5][f].items()} for f in F}}
            for d in dwells]

rows = [x for r in a.runs for x in run(Path(r))]
if a.out:
    json.dump({"palm_mm": {r["layout"]: LAYOUTS[r["layout"]] for r in rows}, "warn_mm": a.warn, "dwells": rows},
              open(a.out, "w"), indent=1)
    print(f"{len(rows)} dwells -> {a.out}")
