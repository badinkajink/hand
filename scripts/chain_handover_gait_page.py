#!/usr/bin/env python3
"""Build docs/experiments/20260923-chain_handover_gait/20260923-chain_handover_gait.html from the study's JSON.

    python3 scripts/chain_handover_gait_page.py

Reads runs/final/<policy>_s<seed>.json (real_v1_chain_policy.py, the chain setting stated on the page),
runs/damping/<policy>_d<damping>.json (the same policies' learned turn at four finger dampings),
runs/calfast/<hand>_{openloop,armonly}.json (the chain without a policy at damping 0),
20260923-gait_ring_screen.json and 20260923-gait_sweep_cal.json; films web/<policy>_chain.mp4 are
transcoded from videos/ when missing.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from hands_tranche_page import table  # noqa: E402

D = os.path.join(ROOT, "docs/experiments/20260923-chain_handover_gait")
OUT = os.path.join(D, "20260923-chain_handover_gait.html")
TPL = os.path.join(ROOT, "scripts/chain_handover_gait_page.template.html")
POL = {"r_d6_clip": ("D6", "clip"), "r_d6_clipsep": ("D6", "clip + separation"), "r_d5_clip": ("D5", "clip"),
       "r_d5_clipsep": ("D5", "clip + separation"), "r_d1_clipsep": ("D1", "clip + separation"), "r_d2_clip": ("D2", "clip"),
       "r_d1_clip": ("D1", "clip"), "r_d3_clip": ("D3", "clip"), "r_d3_clipsep": ("D3", "clip + separation"),
       "r_d7_clip": ("D7", "clip")}
SEAMS = ["turned", "regripped", "staged", "set_down", "seated", "pressed", "re_referenced", "released", "cleared",
         "over", "reindexed", "gait_grip", "gaited"]
STAGE = {"turned": "learned turn", "regripped": "regrip", "staged": "arm staging", "set_down": "set-down",
         "seated": "seat", "pressed": "press", "re_referenced": "re-reference", "released": "release",
         "cleared": "lift", "over": "re-pose", "reindexed": "descent", "gait_grip": "ring close", "gaited": "gait"}


def lost_at(c):
    """First seam from the turn on at which the tool is down: tilted past 45 deg on fewer than two pads,
    or its centre under 30 mm. A gait that stops early counts as lost in the gait."""
    sm = c["seams"]
    for ph in SEAMS:
        s = sm.get(ph)
        if s is None:
            continue
        if s["z"] < 0.03 or (s["tilt_deg"] > 45 and s["pad_contacts"] < 2):
            return ph
    pc = c.get("per_cycle") or []
    return "gaited" if len(pc) < 8 or not c.get("ok") else None


def fmt_turns(v):
    return f"{v:+.2f}"


def outcome_table(final):
    rows, summary = [], {}
    for pid in [p for p in POL if any(k[0] == p for k in final)]:
        cells, done, tsum = [], 0, []
        for sd in range(5):
            c = final.get((pid, sd))
            if c is None:
                cells.append("&#8211;"); continue
            l = lost_at(c)
            t = c["chain_scalars"].get("turns", 0.0)
            if l is None:
                done += 1; tsum.append(t)
                cells.append((f"{fmt_turns(t)} rev", "cell c4" if t > 0.05 else "cell c2"))
            else:
                cells.append((STAGE[l], "cell c0"))
        summary[pid] = (done, tsum)
        rows.append([POL[pid][0], POL[pid][1]] + cells + [f"{done} / 5"])
    head = ["hand", "policy", "nominal", "seed 1", "seed 2", "seed 3", "seed 4", "#complete"]
    return table(head, rows), summary


def damping_table():
    rows = []
    by = {}
    for p in glob.glob(os.path.join(D, "runs/damping/*.json")):
        b = os.path.basename(p)[:-5]
        pid, dmp = b.rsplit("_d", 1)
        by[(pid, dmp)] = json.load(open(p))
    counts = {}
    for pid in POL:
        cells = []
        for dmp in ("0.5", "0.25", "0.1", "0"):
            c = by.get((pid, dmp))
            if c is None:
                cells.append("&#8211;"); continue
            t = c["seams"].get("turned", {})
            held = t.get("pad_contacts", 0) >= 2 and t.get("z", 0) > 0.08
            counts[dmp] = counts.get(dmp, 0) + int(held)
            cells.append((f"{t.get('tilt_deg', 90):.0f}&#176; on {t.get('pad_contacts', 0)}" if held else "dropped",
                          "cell c4" if held and t.get("tilt_deg", 90) < 40 else ("cell c2" if held else "cell c0")))
        rows.append([POL[pid][0], POL[pid][1]] + cells)
    head = ["hand", "policy", "0.5 (trained; &#964; 1.04 s)", "0.25 (&#964; 0.54 s)", "0.1 (&#964; 0.24 s)", "0 (&#964; 0.04 s)"]
    return table(head, rows), counts


def calfast_table():
    rows = []
    for h in ("D1", "D2", "D3", "D5", "D6", "D7"):
        cells = []
        for lab in ("openloop", "armonly"):
            p = os.path.join(D, f"runs/calfast/{h}_{lab}.json")
            if not os.path.exists(p):
                cells.append("&#8211;"); continue
            c = json.load(open(p))
            l = lost_at(c)
            lf = c["seams"].get("lifted", {})
            name = "finger turn" if l == "turned" else STAGE.get(l, l)
            cells.append((f"lost at {name}" if l else "completed", "cell c0" if l else "cell c4"))
        rows.append([h, f"{lf.get('pad_contacts')} pads, {lf.get('pad_force_N'):.1f} N"] + cells)
    return table(["hand", "grasp at the lift", "open-loop finger turn, then the arm", "no finger turn, the arm alone"], rows)


def ring_table():
    r = json.load(open(os.path.join(D, "20260923-gait_ring_screen.json")))
    rows = []
    for h, v in r["hands"].items():
        b, c = v["best"], v["chain_default"]
        az = b["az_deg"]
        rows.append([h, f"{b['dx_mm']:+.1f} / {b['dy_mm']:+.1f}", f"{b['h_mm']:.0f}",
                     f"{az['thumb']:.0f} / {az['index']:.0f} / {az['middle']:.0f}", f"{b['gait_mm']:.1f}", f"{b['approach_mm']:.1f}",
                     f"{c['worst_mm']:.1f}"])
    return table(["hand", "tool in palm frame dx / dy (mm)", "#ring depth (mm)", "pad azimuth th / ix / md (&#176;)",
                  "#grip + open rings (mm)", "#approach ring (mm)", "#chain&#8217;s fixed ring (mm)"], rows)


def sweep_table():
    s = json.load(open(os.path.join(D, "20260923-gait_sweep_cal.json")))
    cfg = {}
    for r in s["rows"]:
        k = (r["twist_s"], r["move_s"], r["angle_gain"], r["squeeze_mm"], r["relay_gait"])
        cfg.setdefault(k, {})[r["policy"]] = r
    rows = []
    for k in sorted(cfg, key=lambda k: (-sum(1 for r in cfg[k].values() if r["ok"] and r["turns"] > 0.2), k)):
        cells = []
        for pid in ("r_d6_clip", "r_d6_clipsep", "r_d5_clip"):
            r = cfg[k].get(pid)
            if r is None:
                cells.append("&#8211;"); continue
            cells.append((f"{r['turns']:+.2f}" if r["ok"] else f"lost ({r['cycles']})",
                          "cell c4" if r["ok"] and r["turns"] > 0.2 else ("cell c2" if r["ok"] else "cell c0")))
        rows.append([f"{k[0]:.1f} / {k[1]:.1f}", str(k[2]), f"{k[3]:.0f}", "one finger" if k[4] else "all three"] + cells)
    return table(["twist / other phases (s)", "#boost gain", "#squeeze (mm)", "release", "D6 clip", "D6 clip + sep", "D5 clip"], rows)


def films(final):
    out = []
    os.makedirs(os.path.join(D, "web"), exist_ok=True)
    for pid in POL:
        src = os.path.join(D, "videos", f"{pid}_chain.mp4")
        web = os.path.join(D, "web", f"{pid}_chain.mp4")
        if os.path.exists(src) and not os.path.exists(web):
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-vf", "scale=640:480", "-c:v", "libx264", "-crf", "28",
                            "-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", web],
                           stdin=subprocess.DEVNULL, check=True)
        if not os.path.exists(web):
            continue
        c = final.get((pid, 0), {}); cs = c.get("chain_scalars", {})
        l = lost_at(c) if c else "?"
        what = (f"completes: {cs.get('cycles_run')} gait cycles, {cs.get('turns', 0):+.2f} rev, final tilt {cs.get('final_tilt_deg', 0):.1f}&#176;"
                if l is None else f"loses the tool at the {STAGE.get(l, l)}")
        out.append(f'<figure><video controls muted loop playsinline preload="metadata" width="640" height="480" '
                   f'src="web/{pid}_chain.mp4"></video><figcaption>{POL[pid][0]}, {POL[pid][1]}, nominal spawn, played at '
                   f'twice real time: grasp, lift, learned turn, arm staging, set-down, seat, press, release, re-pose, '
                   f'ring close, gait. The run {what}.</figcaption></figure>')
    return "\n".join(out)


def main():
    final = {}
    for p in glob.glob(os.path.join(D, "runs/final/*.json")):
        b = os.path.basename(p)[:-5]
        pid, sd = b.rsplit("_s", 1)
        final[(pid, int(sd))] = json.load(open(p))
    t_out, summ = outcome_table(final)
    t_damp, counts = damping_table()
    n_pol = len(summ)
    comp = sum(v[0] for v in summ.values())
    ts = [t for v in summ.values() for t in v[1]]
    fwd = sum(1 for t in ts if t > 0.05)
    four = [p for p in ("r_d6_clip", "r_d6_clipsep", "r_d5_clip", "r_d5_clipsep") if p in summ]
    comp4 = sum(summ[p][0] for p in four)
    sub = {"BUILT": time.strftime("%Y-%m-%d %H:%M"), "TABLE_OUTCOME": t_out, "TABLE_DAMPING": t_damp,
           "TABLE_CALFAST": calfast_table(), "TABLE_RING": ring_table(), "TABLE_SWEEP": sweep_table(),
           "FILMS": films(final), "N_POL": str(n_pol), "N_RUNS": str(len(final)), "N_COMPLETE": str(comp),
           "N_FWD": str(fwd), "N_COMP4": str(comp4), "N_RUNS4": str(5 * len(four)),
           "TURNS_RANGE": (f"{min(ts):.2f}&#8211;{max(ts):.2f}" if ts else "&#8211;"),
           "HELD_05": str(counts.get("0.5", 0)), "HELD_025": str(counts.get("0.25", 0)),
           "HELD_01": str(counts.get("0.1", 0)), "HELD_0": str(counts.get("0", 0))}
    html = open(TPL).read()
    for k, v in sub.items():
        html = html.replace("{{" + k + "}}", v)
    left = [w for w in html.split("{{")[1:]]
    if left:
        raise SystemExit(f"unfilled placeholders: {[w.split('}}')[0] for w in left]}")
    open(OUT, "w").write(html)
    print(f"wrote {OUT} ({os.path.getsize(OUT) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
