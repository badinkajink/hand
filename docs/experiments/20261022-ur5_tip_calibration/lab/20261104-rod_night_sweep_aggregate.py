"""The night of 11-03 to 11-04 (the rod on the inverted rig in probe mode), per seat, from each rod session's
sweep_summary.json (lab/sweep_test_summary.py run on the session with the night's plans as PLAN): the slides' friction
by the rod's turn (r0, r90), the slide's azimuth and its level; the offsets' push across over the flange's step (N/mm)
by level; each reference press's travel along d. Writes lab/20261104-rod_night_sweep_summary.json.

    python3 20261104-rod_night_sweep_aggregate.py
"""
import json
import statistics as stt
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEATS = ((1, 6, "214753-rod13_seed", "20261103-214753-rod13"), (2, 5, "172647-rod21_seed", "20261103-172647-rod21"),
         (3, 4, "202017-rod32_seed", "20261103-202017-rod32"))


def stats(v):
    v = sorted(x for x in v if x is not None and x == x)
    return dict(n=len(v), median=round(stt.median(v), 3), q1=round(v[len(v) // 4], 3),
                q3=round(v[3 * len(v) // 4], 3)) if v else dict(n=0)


def main():
    res = dict(what=("The night of 11-03 to 11-04 (the rod, R 12.5 mm, on the inverted rig in probe mode), summarized per "
                     "seat from lab/sweep_test_summary.py on each session with the night's plans: the slides' friction (the "
                     "push across over the push along, moving and held) by the rod's turn (r0, r90) and the slide's "
                     "azimuth, the offsets' push across over the flange's step (N/mm) by level, and each reference press's "
                     "travel along d from its rest to each level."),
               written=time.strftime("%Y-%m-%d %H:%M"), seats={})
    for seat, vid, s, folder in SEATS:
        d = json.loads((HERE / folder / "sweep_summary.json").read_text())
        r = dict(v3_id=vid, session=s, lab_folder=folder, sweeps=len(d["sweeps"]), slides={}, offsets_k_N_mm={}, refs={})
        for rod in ("r0", "r90"):
            for az in (0, 90):
                for N in (2.0, 4.0):
                    sel = [x for k, sw in d["sweeps"].items() if k.endswith("-" + rod) for x in sw["slides"]
                           if float(x["az_deg"]) % 180 == az and x["at_N"] == N]
                    r["slides"][f"{rod} az {az}/{az + 180} at {N:g} N"] = dict(
                        mu_move=stats([x.get("mu_move") for x in sel]), mu_hold=stats([x.get("mu_hold") for x in sel]))
        for N in (0.5, 1.0, 2.0, 4.0, 8.0):
            r["offsets_k_N_mm"][f"{N:g} N"] = stats([o.get("k_N_mm") for sw in d["sweeps"].values()
                                                     for o in sw["offsets"] if o["N"] == N])
        r["refs"] = {k: [dict(N=lv["N"], travel_mm=lv["travel_mm"], along=lv["along"]) for lv in v["levels"]]
                     for k, v in d["refs"].items() if k.startswith("probe-ref")}
        res["seats"][str(seat)] = r
    out = HERE / "20261104-rod_night_sweep_summary.json"
    out.write_text(json.dumps(res, indent=1))
    print("->", out)


if __name__ == "__main__":
    main()
