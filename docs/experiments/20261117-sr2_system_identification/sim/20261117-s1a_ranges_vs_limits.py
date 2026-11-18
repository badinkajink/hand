"""Each joint's reachable range on the hand at 7.5 V against the model's and the station's limits (W2 of the
identification queue, S1(a); the lab's sweeps of 2026-11-17, 20:53-21:03 and the redo of five ends at 21:28-21:31, in
data/s1/20261117-s1a_rows.jsonl, one row per 0.5 deg step: target, measured, load, all in simulator degrees).

Per joint and end:
  model     hand_object_scale_kin.limits(): the scene ROM meet the station's calibrated range, the joint ranges of
            HOM's scenes and of the planners
  rom       hand_object_scale_kin.SIM_ROM, the scene ROM alone
  station   manta_hand.servos.FINGER_JOINTS' calibrated range (through plan.JOINT_SIGN)
  safe      servo_limits.safe_range_sim: the calibrated range kept 5 deg from the potentiometer's ends; the server
            clips HOM's targets to it
  sweep_end the sweep's last target: the safe limit or the night's cap (yaw +-45, MCP <= 58, PIP <= 80 deg), the nearer
  reach     the farthest reading of the end's last sweep (the redo where there is one), the target it came at, the
            target-minus-reading lag at the sweep's end and the load there (0.1 % of PWM duty)
  stop      'stalled' when the lab's stall test stopped the sweep (the reading advancing under 0.6 deg over six steps
            while 2 deg behind), otherwise 'followed' to the sweep's end
  inside_deg  how far the reach lies inside the model's limit (deg; positive inside)
A joint is flagged when it stalled more than 1 deg inside the model's limit. An end that followed is not tested past
its sweep's end; its gap to the model is listed as untested.

    .venv/bin/python sim/20261117-s1a_ranges_vs_limits.py [--out data/s1/20261117-s1a_ranges_vs_limits.json]
"""
import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "scripts/hom"), str(ROOT / "src"),
                str(ROOT / "src/sr2hand/driver/manta/host")]
import hand_object_scale_kin as K  # noqa: E402
from manta_hand import plan as P, servos as SV  # noqa: E402
from sr2hand.control import servo_limits as SL  # noqa: E402

DATA = HERE.parent / "data/s1"
JOINTS = ("yaw", "mcp", "pip")
FLAG_DEG = 1.0
AFTERNOON = {("index", "mcp", "low"): -11.4, ("middle", "mcp", "low"): -10.0}   # 11-17 g13 plan, 21-22 % duty


def sweeps(rows):
    """The rows cut into sweeps: a new one at a change of (finger, joint, end) or when the target returns to the first
    step."""
    out, cur = [], None
    for r in rows:
        k = (r["finger"], r["joint"], r["end"])
        if cur is None or k != cur["key"] or (abs(r["target"]) <= 0.5 + 1e-9
                                              and abs(r["target"]) < abs(cur["rows"][-1]["target"]) - 1e-9):
            cur = {"key": k, "rows": []}
            out.append(cur)
        cur["rows"].append(r)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rows", default=str(DATA / "20261117-s1a_rows.jsonl"))
    ap.add_argument("--out", default=str(DATA / "20261117-s1a_ranges_vs_limits.json"))
    a = ap.parse_args()
    rows = [json.loads(line) for line in Path(a.rows).read_text().splitlines() if line.strip()]
    summ = {(e["finger"], e["joint"], e["end"]): e for e in json.loads((DATA / "20261117-s1a_summary.json").read_text())}
    redo = {(e["finger"], e["joint"], e["end"]): e
            for e in json.loads((DATA / "20261117-s1a_summary_redo.json").read_text())}
    sw = sweeps(rows)
    last, first = {}, {}
    for s in sw:                                            # the redo comes later in the file and replaces the first run
        first.setdefault(s["key"], s)
        last[s["key"]] = s
    model, rom = K.limits(False), K.limits(True)
    ends = []
    for f in K.FINGERS:
        for i, j in enumerate(JOINTS):
            sv = P.SIM_JOINT_TO_SERVO[j]
            _sid, _zero, (lo, hi) = SV.FINGER_JOINTS[P.FINGER_ID[f]][sv]
            sgn = P.JOINT_SIGN[(f, j)]
            st = sorted((lo / sgn, hi / sgn))
            sf = SL.safe_range_sim(f, j)
            for e, side in (("low", 0), ("high", 1)):
                key = (f, j, e)
                s = last[key]
                R = s["rows"]
                lab = redo.get(key) or summ[key]
                d = 1.0 if lab["target_deg"] >= 0 else -1.0   # the sweep's direction (the index PIP's low end is +1.78)
                inw = 1.0 if e == "low" else -1.0               # into the range from this end
                far = max(R, key=lambda r: d * r["measured"])
                m_lim = math.degrees(model[f][i, side])
                stop = "stalled" if lab["stopped"].startswith("stalled") else (
                    "followed" if lab["stopped"] == "reached the end" else lab["stopped"])
                still = 0                                   # trailing steps over which the reading did not advance
                for r0, r1 in zip(R[-2::-1], R[:0:-1]):
                    if d * (r1["measured"] - r0["measured"]) > 1e-6:
                        break
                    still += 1
                inside = inw * (far["measured"] - m_lim)
                rec = {"finger": f, "joint": j, "end": e,
                       "model_deg": round(m_lim, 2), "rom_deg": round(math.degrees(rom[f][i, side]), 2),
                       "station_deg": round(st[side], 2), "safe_deg": round(sf[side], 2),
                       "sweep_end_deg": lab["target_deg"],
                       "reach_deg": round(far["measured"], 3), "reach_at_target_deg": far["target"],
                       "lag_at_end_deg": round(d * (R[-1]["target"] - R[-1]["measured"]), 2),
                       "load_at_end": R[-1]["load"], "load_max": max(abs(r["load"]) for r in R),
                       "steps_without_advance_at_end": still,
                       "stop": stop, "lab_stopped": lab["stopped"], "redo": key in redo,
                       "inside_deg": round(inside, 2),
                       "flag": stop == "stalled" and inside > FLAG_DEG,
                       "untested_deg": round(inw * (lab["target_deg"] - m_lim), 2) if stop == "followed" else None}
                if key in redo:
                    F = first[key]["rows"]
                    ff = max(F, key=lambda r: d * r["measured"])
                    rec["first_run"] = {"reach_deg": round(ff["measured"], 3), "at_target_deg": ff["target"],
                                        "load_max": max(abs(r["load"]) for r in F), "lab_stopped": summ[key]["stopped"]}
                if key in AFTERNOON:
                    rec["afternoon_deg"] = AFTERNOON[key]
                ends.append(rec)
    hdr = f"{'joint':12s} {'end':4s} {'model':>7s} {'safe':>7s} {'sweep':>7s} {'reach':>8s} {'lag':>5s} {'load':>5s} " \
          f"{'stop':8s} {'inside':>6s}"
    print(hdr)
    for r in ends:
        print(f"{r['finger'] + '.' + r['joint']:12s} {r['end']:4s} {r['model_deg']:+7.2f} {r['safe_deg']:+7.2f} "
              f"{r['sweep_end_deg']:+7.2f} {r['reach_deg']:+8.2f} {r['lag_at_end_deg']:5.2f} {r['load_at_end']:5.0f} "
              f"{r['stop']:8s} {r['inside_deg']:+6.2f}{'  FLAG' if r['flag'] else ''}"
              f"{'  (first run ' + format(r['first_run']['reach_deg'], '+.2f') + ')' if 'first_run' in r else ''}")
    out = {"source": "lab S1(a), 2026-11-17 20:53-21:03 and the redo of five ends 21:28-21:31 (stall test)",
           "rows": str(Path(a.rows).relative_to(ROOT)), "script": str(Path(__file__).relative_to(ROOT)),
           "units": "simulator degrees; load in 0.1 % of PWM duty (SCS0009 present-load register)",
           "flag_rule": f"stalled more than {FLAG_DEG} deg inside the model's limit",
           "model": "hand_object_scale_kin.limits(): SIM_ROM meet manta_hand.servos.FINGER_JOINTS (through JOINT_SIGN)",
           "caps": {"yaw": [-45.0, 45.0], "mcp": [None, 58.0], "pip": [None, 80.0]},
           "ends": ends,
           "flagged": [f"{r['finger']}.{r['joint']}.{r['end']}" for r in ends if r["flag"]],
           "safe_inside_model": [{"joint": f"{r['finger']}.{r['joint']}.{r['end']}", "safe_deg": r["safe_deg"],
                                  "model_deg": r["model_deg"]} for r in ends if abs(r["safe_deg"] - r["model_deg"]) > 1e-6]}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("flagged:", out["flagged"])
    print("safe range inside the model:", out["safe_inside_model"])
    print("->", a.out)


if __name__ == "__main__":
    main()
