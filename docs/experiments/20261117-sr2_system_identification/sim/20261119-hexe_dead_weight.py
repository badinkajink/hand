"""The HEX-E against a dead weight (2026-11-19 17:4x; the owner hung a 200 g steel mass, with 4-10 g of cable, from the
index fingertip while the arm stayed parked over the session's start). Reads the recorder's chunks
(lab/day/20261119_record_hexe.py, 100 Hz, in logs/20261119-sr2_dead_weight/, 3-5 MB each, rsynced), finds the quiet
plateaus (1 s means whose x and y spread stays under 0.12 N for 3 s) and the steps between them, and writes their sizes
against m g with an HTML paragraph for the page (scripts/fingertip/fingertip_board_force_place_page.py).

    /usr/bin/python3 sim/20261119-hexe_dead_weight.py [--mass-g 200 --cable-g 4 10]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
LOGS = ROOT / "logs/20261119-sr2_dead_weight"
G = 9.80665


def load():
    T, F = [], []
    for f in sorted(LOGS.glob("20261119-*-hexe.jsonl")):
        for line in open(f):
            r = json.loads(line)
            if r["t"]:
                T += r["t"]
                F += r["F"]
    T, F = np.asarray(T), np.asarray(F)
    o = np.argsort(T)
    T, F = T[o], F[o]
    keep = np.r_[True, np.diff(T) > 0]
    return T[keep], F[keep]


def plateaus(T, F, sd_max=0.12, w=3, min_s=20):
    secs = np.arange(np.floor(T[0]), np.ceil(T[-1]))
    S, M, SD = [], [], []
    for s in secs:
        m = (T >= s) & (T < s + 1)
        if m.sum() >= 50:
            S.append(s)
            M.append(F[m].mean(0))
            SD.append(F[m].std(0))
    S, M, SD = np.array(S), np.array(M), np.array(SD)
    st = np.array([i >= w and np.all(SD[i - w:i + 1, :2] < sd_max) and
                   np.all(np.abs(M[i - w:i + 1, :2] - M[i, :2]) < 0.08) for i in range(len(S))])
    out, cur = [], []
    for i in range(len(S)):
        if st[i]:
            cur.append(i)
        elif cur:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return [(S[p[0]], S[p[-1]] + 1, M[p].mean(0), len(p)) for p in out if len(p) >= min_s]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mass-g", type=float, default=200.0)
    ap.add_argument("--cable-g", type=float, nargs=2, default=(4.0, 10.0))
    ap.add_argument("--out", default=str(HERE.parent / "data/s3/20261119-hexe_dead_weight.json"))
    a = ap.parse_args()
    T, F = load()
    P = plateaus(T, F)
    hm = lambda t: time.strftime("%H:%M:%S", time.localtime(t))   # noqa: E731
    lo, hi = (a.mass_g + a.cable_g[0]) * 1e-3 * G, (a.mass_g + a.cable_g[1]) * 1e-3 * G
    steps = []
    for (s0, e0, m0, n0), (s1, e1, m1, n1) in zip(P[:-1], P[1:]):
        d = m1 - m0
        steps.append(dict(from_=f"{hm(s0)}-{hm(e0)}", to=f"{hm(s1)}-{hm(e1)}", dF_N=np.round(d[:3], 3).tolist(),
                          dT_Nm=np.round(d[3:6], 4).tolist(), size_N=round(float(np.linalg.norm(d[:3])), 3)))
    big = [s for s in steps if s["size_N"] > 0.5]
    ratios = [s["size_N"] / g for s in big for g in (hi, lo)]
    rec = dict(what=__doc__.split("\n\n")[0].replace("\n", " "), mass_g=a.mass_g, cable_g=list(a.cable_g),
               weight_N=[round(lo, 3), round(hi, 3)], samples=int(len(T)), span=[hm(T[0]), hm(T[-1])],
               plateaus=[dict(span=f"{hm(s)}-{hm(e)}", seconds=n, F_N=np.round(m[:3], 3).tolist(),
                              T_Nm=np.round(m[3:6], 4).tolist()) for s, e, m, n in P],
               steps=steps)
    if big:
        rec["ratio_range"] = [round(min(ratios), 3), round(max(ratios), 3)]
        rows = "".join(f"<li>{s['from_']} to {s['to']}: a step of {s['size_N']:.2f} N "
                       f"(dF {s['dF_N']} N)</li>" for s in big)
        rec["html"] = (f"<p>A dead weight checks the HEX-E itself. The owner hung a {a.mass_g:.0f} g steel mass, with "
                       f"{a.cable_g[0]:.0f}-{a.cable_g[1]:.0f} g of cable, from the index fingertip. The arm stayed parked. "
                       f"The weight is {lo:.2f}-{hi:.2f} N. The HEX-E, at 100 Hz, stepped between quiet plateaus as "
                       f"follows:</p><ul>{rows}</ul><p>That is {min(ratios):.2f}-{max(ratios):.2f} of the weight. The board "
                       f"comparisons above do not depend on it, because the calibrations used the same sensor and reader. "
                       f"But absolute HEX-E forces would be about {1 / max(ratios):.1f}-{1 / min(ratios):.1f} times larger "
                       f"in true newtons. That touches the pads&#8217; modulus fit, the stall torques of the S3 presses, and "
                       f"the 3 N rule as applied. With that factor, the index MCP&#8217;s stall (2 N at 67 mm) becomes about "
                       f"0.22 N m. That is the SCS0009&#8217;s rated stall torque.</p>"
                       f"<p>Whether the sensor read this low in October is open, and the comparisons depend on it. If it "
                       f"did, the calibrations share the arm presses&#8217; units, and the ratios above stand. If its scale "
                       f"changed after the calibrations, every ratio to the calibration shrinks by the same factor. The PIP "
                       f"20 presses would then read about half their calibration per true newton. The HEX-E was recabled to "
                       f"its own network on 11-16. Two weak hints point opposite ways. Its untared z barely moved: 651.25 N "
                       f"with the calibration tool on 10-22, 672.1 N with the hand now. A change of the box&#8217;s scale "
                       f"would have pulled it toward 390 N. But the October gravity sweep weighed that tool at 160 g, as the "
                       f"payload estimates of sensor and tool expect. At today&#8217;s scale the tool would have weighed "
                       f"275 g.</p><p>A test without the HEX-E would decide it. A curled finger&#8217;s pad faces up, so a "
                       f"weighed mass can rest on it through a small ball at the apex. The boards&#8217; field for that "
                       f"known force can then be compared with the calibration&#8217;s (tip 3: about 13.5 uT per "
                       f"calibration newton).</p>")
    Path(a.out).write_text(json.dumps(rec, indent=1))
    for p in rec["plateaus"]:
        print("plateau", p)
    for s in steps:
        print("step", s)
    print("ratio", rec.get("ratio_range"), "->", a.out)


if __name__ == "__main__":
    main()
