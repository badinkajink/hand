"""S3's head-on yaw presses of 2026-11-18 (16:33-17:17) in one table: per run the finger, direction, plate and rate;
first contact (the first step whose force change in x_P, y_P reached 0.15 N: its target offset and reading); the slope
in N per deg of target offset from contact to the first stall (the first step at 0.95 of the run's first plateau); the
largest force and its offset; the stop; and on the step back the offset at which the force fell below half its
largest. Writes data/s3/20261118-yaw_head_on_summary.json.

    python3 s3_yaw_summary.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
D = HERE.parent / "data/s3"


def main():
    out = []
    for p in sorted(D.glob("20261118-*_yaw_head_on.jsonl")):
        rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
        h, end = rows[0], rows[-1]
        st = [r for r in rows if r.get("phase") == "press"]
        bk = [r for r in rows if r.get("phase") == "back"]
        F = [r["dF_abs_N"] or 0.0 for r in st]
        c = next((i for i, f in enumerate(F) if f >= 0.15), None)
        imax = max(range(len(F)), key=lambda i: F[i]) if F else None
        slope = None
        if c is not None:
            j = next((i for i in range(c + 1, len(F)) if F[i] >= 0.95 * max(F[c:i + 3] or [0]) and i + 1 < len(F)
                      and F[i + 1] <= F[i] * 1.05), None)
            if j is not None and st[j]["offset"] - st[c]["offset"] > 0.5:
                slope = round((F[j] - F[c]) / (st[j]["offset"] - st[c]["offset"]), 3)
        half = next((r["offset"] for r in bk if (r["dF_abs_N"] or 0) < 0.5 * F[imax]), None) if imax is not None else None
        rec = {"run": p.name, "finger": h["finger"], "sign": h["sign"], "step_deg": h["step_deg"], "dwell_s": h["dwell_s"],
               "plate": "tag" if "tag" in (h.get("note") or "") else "taped",
               "contact": None if c is None else {"step": st[c]["k"], "offset": st[c]["offset"], "reading": st[c]["reading"],
                                                   "N": F[c]},
               "slope_N_per_deg": slope, "max_N": None if imax is None else F[imax],
               "max_at_offset": None if imax is None else st[imax]["offset"], "stop": end.get("reason"),
               "back_release_offset": half}
        out.append(rec)
        print(f"{p.name[:15]} {h['finger']:6s} {h['sign']:+d} {rec['plate']:5s} {h['step_deg']:.1f}/{h['dwell_s']:.0f}s "
              f"contact {rec['contact']['offset'] if c is not None else '-'} slope {slope} max {rec['max_N']:.2f} N at "
              f"{rec['max_at_offset']:.1f} deg; release {half}; {str(rec['stop'])[:28]}")
    (D / "20261118-yaw_head_on_summary.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
