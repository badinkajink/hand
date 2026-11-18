"""The night of 2026-11-17 on the SR2 hand, summarised for the session page
(scripts/hardware/sr2_bench_20261117_page.py reads data/20261117-night_summary.json's "html" into its identification
section). Run on corvus after the night's runs, with their records committed:

    /usr/bin/python3 docs/experiments/20261117-sr2_system_identification/lab/night_summary.py

Reads: data/s2/ (S2's tables and diag lines), data/s1/rows.jsonl (S1(b)), the bench study's
data/20261117-night_repeats_tracking.json (the repeats scored by scripts/hardware/hom_trial_tracking.py; the arm of each
from corvus's record label), and logs/20261117-sr2_boards_sweep/s3/events.jsonl (the boards' sweep).
"""
import html
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[2]
B = ROOT / "docs/experiments/20261115-hom_bench_trials"
OUT = HERE / "data/20261117-night_summary.json"
ARM = {"base": "as before", "fstep": "paper-rate filter", "fstep_hyst": "paper-rate filter, contact hysteresis"}


def esc(s):
    return html.escape(str(s))


def table(head, rows, caption):
    th = "".join(f"<th>{esc(h)}</th>" for h in head)
    tb = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><caption>{caption}</caption><thead><tr>{th}</tr></thead><tbody>{tb}</tbody></table>"


def s2():
    d = HERE / "data/s2"
    if not (d / "dwells.tsv").exists():
        return "<p>S2 did not run.</p>", {}
    rows = [x.split("\t") for x in (d / "dwells.tsv").read_text().splitlines()]
    head, body = rows[0], rows[1:]
    ib, ie, ij = head.index("block"), head.index("err_deg"), head.index("joint")
    blocks = sorted({r[ib] for r in body}, key=["a1", "b1", "a2", "b2", "c", "c40"].index)
    out = []
    for blk in blocks:
        e = [abs(float(r[ie])) for r in body if r[ib] == blk and r[ie] not in ("", "nan")]
        out.append([blk, len([r for r in body if r[ib] == blk]), f"{np.median(e):.2f}", f"{np.percentile(e, 95):.2f}"])
    diag = (d / "supply_diag.txt").read_text()
    temps = [float(x) for x in re.findall(r"'\d+': ([0-9.]+)", " ".join(l for l in diag.splitlines() if "temperature" in l))]
    volts = [float(x) for x in re.findall(r"'\d+': ([0-9.]+)", " ".join(l for l in diag.splitlines() if "volts" in l))]
    txt = (f"<p>S2 ran {len(blocks)} of its six blocks at the (85, 170) layout. Between blocks the servos read "
           f"{min(volts):.1f}-{max(volts):.1f} V and {min(temps):.0f}-{max(temps):.0f} °C. The dwell error is the "
           "settled angle minus the command, per dwell (data/s2/dwells.tsv); the workstation fits the dead band, the "
           "drive per degree of error and the friction from these tables against 2026-11-06's.</p>")
    return txt + table(["Block", "Dwells", "Median |error| (°)", "95th percentile (°)"], out,
                       "Table 5. S2's dwells at 7.5 V, per block."), {"blocks": blocks}


def s1():
    f = HERE / "data/s1/rows.jsonl"
    if not f.exists():
        return "<p>S1(b) did not run.</p>", {}
    rows = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    out = []
    for fg in ("thumb", "index", "middle"):
        r = [x for x in rows if x["finger"] == fg]
        seen = [x for x in r if "d_mm" in x]
        if not r:
            continue
        d = [float(np.linalg.norm(x["d_mm"])) for x in seen]
        a = [x["normal_deg"] for x in seen]
        out.append([fg, f"{len(seen)} of {len(r)}", f"{np.median(d):.2f}" if d else "–", f"{max(d):.2f}" if d else "–",
                    f"{np.median(a):.1f}" if a else "–"])
    txt = ("<p>S1(b) measured the DIP tags of the thumb and the middle finger at nine poses each with both cameras, "
           "against the hand's model at the measured joints. The middle's tag was not detected at any of its poses, "
           "and the index's tag faced away from both cameras at every candidate pose of the night's arm pose.</p>")
    fit = HERE / "data/s1/20261117-thumb_zero_fit.json"
    if fit.exists():
        z = json.loads(fit.read_text())
        mo, ze, bo = z["mount"], z["zeros"], z["both"]
        txt += (f"<p>The workstation fitted the thumb's rows. As built, the tag's centre misses the model by "
                f"{z['rms_centre_mm_before']:.2f} mm RMS. Moving the thumb's mount by "
                f"({mo['params']['d_mount_x_mm']:+.1f}, {mo['params']['d_mount_y_mm']:+.1f}) mm in the palm frame "
                f"leaves {mo['rms_centre_mm']:.2f} mm. Fitting the joint zeros alone leaves {ze['rms_centre_mm']:.2f} mm. "
                f"Fitting both leaves {bo['rms_centre_mm']:.2f} mm ({bo['loo_rms_centre_mm']:.2f} mm leaving one pose "
                f"out), with a yaw zero of {bo['params']['d_yaw_deg']:+.1f}°. The nine poses do not separate the MCP's "
                "zero from the PIP's. So the thumb's error is mostly its mount's position; a ruler at the thumb's "
                "gantry would confirm a 5-7 mm error along x.</p>")
    return txt + table(["Finger", "Poses seen", "Median |tag − model| (mm)", "Largest (mm)", "Median normal error (°)"],
                       out, "Table 6. The fingertip tags against the hand's model (data/s1/rows.jsonl)."), {}


def repeats():
    f = B / "data/20261117-night_repeats_tracking.json"
    if not f.exists():
        return "<p>The twist repeats did not run.</p>", {}
    tt = json.loads(f.read_text())
    lab = {}
    for c in sorted((B / "runs_corvus").glob("20261117-*_twist_corvus.jsonl")):
        h = json.loads(c.open().readline())
        m = re.search(r"tips(off|on)_(base|fstep_hyst|fstep)", h.get("label") or "")
        if m:
            lab[c.name[:15]] = (m.group(1), m.group(2))
    per = {}
    rows = []
    for r in tt["trials"]:
        st = r["stamp"]
        key = min(lab, key=lambda k: abs(int(k[9:]) - int(st[9:]))) if lab else None
        if key is None or abs(int(key[9:]) - int(st[9:])) > 30:
            continue
        tips, arm = lab[key]
        tr = r["track"]
        trk = max(0.0, 1.0 - tr["err_rms_deg"] / tr["ref_rms_deg"]) if tr.get("ref_rms_deg") else 0.0
        ended = r["ended"].get("class", "")
        per.setdefault((tips, arm), []).append((trk, r["motion"].get("lat_max_mm"), ended))
        rows.append([f"{st[9:11]}:{st[11:13]}", tips, esc(ARM[arm]), f"{trk:.3f}",
                     f"{(tr.get('twist_task_end_deg') or 0):.1f}", f"{r['motion'].get('lat_max_mm', 0):.1f}", esc(ended)])
    summ = []
    for (tips, arm), v in sorted(per.items()):
        t = [x[0] for x in v]
        summ.append([tips, esc(ARM[arm]), len(v), f"{np.mean(t):.3f}", f"{min(t):.3f}-{max(t):.3f}",
                     sum(1 for x in v if x[2] == "end of schedule")])
    txt = ("<p>The twist g0 at (70, 110) ran in four arms in turn: the fingertip contacts off and on, each with HOM as "
           "before and with the complementary filter at the paper's rate (2 ms sub-steps); with the contacts on the "
           "filtered arm also had the 0.10 N hysteresis on the contact switch. The workstation's blind predictions for "
           "the arms as before are in data/20261117-g0_x70y110_twist_predictions.json.</p>")
    return txt + table(["Tips", "HOM", "Trials", "Mean tracking", "Range", "Full runs"], summ,
                       "Table 7. The night's twist repeats, per arm.") + table(
        ["Time", "Tips", "HOM", "Tracking", "Twist (°)", "Drift (mm)", "Ended"], rows,
        "Table 8. Every repeat."), {}


def sweep():
    f = ROOT / "logs/20261117-sr2_boards_sweep/s3/events.jsonl"
    if not f.exists():
        return "<p>The boards' random-pose sweep did not run.</p>", {}
    ev = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    done = sum(1 for e in ev if e.get("kind") == "step" or e.get("op") == "joints")
    end = ev[-1] if ev else {}
    return (f"<p>The boards' random-pose sweep ran {done} steps of 327 and ended with \"{esc(end.get('reason', ''))}\". "
            "The boards' records are in logs/20261117-three_finger_mux/20261117-180517-mux.bin on corvus; the "
            "workstation fits the boards' response to the hand's own pose from them.</p>"), {}


if __name__ == "__main__":
    parts = [s2(), s1(), repeats(), sweep()]
    OUT.write_text(json.dumps({"html": "<h3>The night's runs</h3>" + "".join(p[0] for p in parts)}))
    print(f"-> {OUT}")
