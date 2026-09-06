#!/usr/bin/env python3
"""Per-trial table for the eight-hand open-loop transfer runs, from the CB1 log archive.

One row per completed reorientation run that carries BOTH an AprilTag turn angle
(`object_track.deg_turned`) and an operator retention verdict (`manual_score.success`).

This is a LOOSER filter than `real_v1_transfer_study.py`, which additionally requires
visibility >= 0.9 and a non-null `cos_hold`. Counts differ for that reason and not because
the two read different archives: 139 trials here, 73 there.
"""
import argparse, glob, json, os, statistics as st

DID = {"sv1_w6689": "D1", "sv1_w2360": "D2", "sv1_u1364": "D3", "g12": "D4",
       "sv1_u0060": "D5", "sv1_u0308": "D6", "rv05_manual": "D7", "sv1_w0099": "D8"}
ARCHIVE = "docs/experiments/20260902-cb1-log-archive/logs"


def trials(archive=ARCHIVE):
    out = []
    for p in sorted(glob.glob(os.path.join(archive, "*_SUMMARY.json"))):
        s = json.load(open(p))
        if s.get("operation") != "reorientation" or s.get("status") != "complete":
            continue
        d = s.get("design")
        if d not in DID:
            continue
        t = s.get("object_track") or {}
        m = s.get("manual_score") or {}
        if t.get("deg_turned") is None or m.get("success") is None:
            continue
        out.append({"did": DID[d], "design": d, "run": s["run_id"],
                    "deg": t.get("deg_turned"), "slip": t.get("slip_mm"),
                    "cos": t.get("cos_hold"), "dropped": t.get("dropped"),
                    "vis": t.get("visibility"), "op_success": m.get("success"),
                    "op_deg": m.get("turn_deg")})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=ARCHIVE)
    ap.add_argument("--out", default="docs/experiments/20260906-rv05_band/bench_trials_D1_D8.json")
    a = ap.parse_args()
    rows = trials(a.archive)
    med = lambda v: st.median(v) if v else float("nan")
    print(f"{'D':<4} {'design':<12} | {'n':>3} {'turn':>6} {'slip':>6} {'cos':>6}"
          f" | {'n':>3} {'turn':>6} {'slip':>6} {'cos':>6}   (retained | not retained, medians)")
    for did in sorted({r["did"] for r in rows}):
        R = [r for r in rows if r["did"] == did and r["op_success"]]
        N = [r for r in rows if r["did"] == did and not r["op_success"]]
        f = lambda S, k: med([x[k] for x in S if x[k] is not None])
        name = (R or N)[0]["design"]
        print(f"{did:<4} {name:<12} | {len(R):>3} {f(R,'deg'):>6.1f} {f(R,'slip'):>6.1f} "
              f"{f(R,'cos'):>6.3f} | {len(N):>3} {f(N,'deg'):>6.1f} {f(N,'slip'):>6.1f} "
              f"{f(N,'cos'):>6.3f}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(rows, open(a.out, "w"), indent=1)
    print(f"\n{len(rows)} trials -> {a.out}")


if __name__ == "__main__":
    raise SystemExit(main())
