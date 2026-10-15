#!/usr/bin/env python3
"""Console tables of the fingertip co-design rows (bed variants; used while the study runs)."""
import json
import sys
from collections import defaultdict


def bed_table(path, models=None):
    V = defaultdict(dict)
    for r in map(json.loads, open(path)):
        m = r["model"]
        if models and m not in models:
            continue
        if r.get("status") != "complete":
            V[m]["fail"] = r.get("status")
            continue
        c = r["case"]
        if c == "static":
            V[m][("app", r["N"])] = r["approach_um"]
        elif c == "T2":
            V[m][("on", r["N"])] = r.get("rbar_onset_mm")
            V[m][("creep", r["N"])] = r.get("creep_deg_s")
        elif c == "spin":
            V[m][("spin", r["N"], r["omega"])] = r["rbar_spin_mm"]
        elif c == "T5":
            V[m]["N45"], V[m]["phi"] = r.get("N_at_45_N"), r.get("phi_end_deg")
    f = lambda x, d=2: ("%.*f" % (d, x)) if isinstance(x, (int, float)) else "-"
    hdr = ["app1", "on1", "on3", "crp1", "crp3"] + [f"w{w:g}" for w in (0.3, 1, 3, 5)] + ["N45", "phi"]
    print("%-36s" % "model" + "".join("%7s" % h for h in hdr))
    for m, d in V.items():
        vals = [f(d.get(("app", 1.0)), 1), f(d.get(("on", 1.0))), f(d.get(("on", 3.0))), f(d.get(("creep", 1.0))),
                f(d.get(("creep", 3.0)))] + [f(d.get(("spin", 1.0, w))) for w in (0.3, 1.0, 3.0, 5.0)] + \
               [f(d.get("N45"), 3), f(d.get("phi"), 1)]
        print("%-36s" % m + "".join("%7s" % v for v in vals) + ("  " + d["fail"] if "fail" in d else ""))


if __name__ == "__main__":
    bed_table(sys.argv[1], sys.argv[2:] or None)
