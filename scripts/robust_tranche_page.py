#!/usr/bin/env python3
"""Build the page for the 2026-09-20 continuation ("robust") tranche from the queue's own outputs.

    python3 scripts/robust_tranche_page.py [--root docs/experiments/20260920-robust_tranche]
                                           [--parent docs/experiments/20260919-hands_tranche]

Reads queue.json, tranche_results.json, <id>_eval.json, <id>_eval_j3dr.json, <id>_strip.png,
<id>_chain_pl{25,0}.json and each run's tensorboard scalars; the parent tranche's results and its
robust/<id>_eval_j3dr.json for the comparison; writes 20260920-robust_reorient_policies.html beside
the queue (the rollout videos are web/<id>.mp4 beside the page, transcoded from videos/).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hands_tranche_page import ARM_LABEL, cellc, curves, f1, plot_curves, table, uri, uri_jpeg  # noqa: E402

PLANT_ROWS = [
    ("tool spawn", "one nominal pose", "&#177;3 mm in x and y, &#177;10&#176; of yaw", "a policy trained at one pose has no reason to hold elsewhere; the CPU probe and the chain never start at the trained pose to the millimetre"),
    ("tool", "25 mm cylinder, 24.5 g", "+ screw-tip mesh, 25.6 g", "the chain&#8217;s tool and the real screwdriver carry it"),
    ("floor", "mjlab plane: &#956; 1 / 0.005 / 0.0001, solref 0.02, solimp 0.9&#8211;0.95", "the scene&#8217;s floor: &#956; 1.8 / 0.15 / 0.01, solref 0.006, solimp 0.97&#8211;0.995", "the tool lies on it through the closure and the start of the lift (<code>--scene-floor</code>)"),
    ("tool contact class", "MuJoCo defaults (solref 0.02, solimp 0.9&#8211;0.95); the pair with a pad averaged to 0.013 / 0.935", "the scene&#8217;s (0.006, 0.97&#8211;0.995), the same as the pads", "<code>make_object_spec_from_frozen</code> copied only type, size, mass and friction since the first mjlab run"),
]


def chain_cell(path):
    if not os.path.exists(path):
        return ("&#8211;", "cell")
    c = json.load(open(path)); tr = c.get("policy_trace", [])
    if not tr:
        return ("&#8211;", "cell")
    end = tr[-1]; peak = max(r[1] for r in tr)
    held = end[2] > 0.06 and sum(end[4:7]) >= 0.5
    return (f"{'held' if held else 'lost'} &#183; {end[1]:+.2f} (peak {peak:.2f})", "cell c4" if held else "cell c0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="docs/experiments/20260920-robust_tranche")
    ap.add_argument("--parent", default="docs/experiments/20260919-hands_tranche")
    a = ap.parse_args()
    R, P = a.root, a.parent
    q = json.load(open(os.path.join(R, "queue.json")))
    res = {r["id"]: r for r in (json.load(open(os.path.join(R, "tranche_results.json"))) if os.path.exists(os.path.join(R, "tranche_results.json")) else [])}
    pres = {r["id"]: r for r in json.load(open(os.path.join(P, "tranche_results.json")))}
    jobs = q["jobs"]
    done = [j for j in jobs if res.get(j["id"], {}).get("status") == "done"]

    def jit(root, jid, sub=""):
        for p in (os.path.join(root, sub, f"{jid}_eval_j3dr.json"), os.path.join(root, f"{jid}_eval_j3dr.json")):
            if os.path.exists(p):
                return json.load(open(p))
        return None

    def hc(e, key="hold_rate", cos="final_cos_mean"):
        if not e or e.get(key) is None:
            return ("&#8211;", "cell")
        n = int(e.get("n", 64)); h = int(round(e[key] * n))
        return cellc(h / n, f"{h}/{n} &#183; {e[cos]:+.2f}")

    # --- main table: continuation against its parent
    rows, reading_bits = [], []
    for j in done:
        r = res[j["id"]]; pr = pres.get(j.get("warm_start", ""), {})
        pj = jit(P, j.get("warm_start", ""), "robust"); cj = jit(R, j["id"])
        rows.append([j["hand"], ARM_LABEL.get(j["arm"], j["arm"]),
                     hc(pr), hc(pj), f1(pr.get("clearance_min_mm_mean")),
                     hc(r), hc(cj), f1(r.get("clearance_min_mm_mean")),
                     chain_cell(os.path.join(P, f"{j.get('warm_start', '')}_chain_pl25.json")),
                     chain_cell(os.path.join(R, f"{j['id']}_chain_pl25.json"))])
        if pj and cj:
            reading_bits.append((j["hand"], j["arm"], int(round(pj["hold_rate"] * 64)), int(round(cj["hold_rate"] * 64)),
                                 pj["final_cos_mean"], cj["final_cos_mean"]))
    table_main = table(["hand", "arm", "parent nominal: held &#183; cos", "parent jittered", "parent clearance (mm)",
                        "continued nominal", "continued jittered", "continued clearance (mm)",
                        "chain, parent", "chain, continued"], rows) if rows else "<p>No job has finished yet.</p>"
    up = [b for b in reading_bits if b[3] >= b[2] + 8]; down = [b for b in reading_bits if b[3] <= b[2] - 8]
    reading = ""
    if reading_bits:
        reading = ("<p>Jittered hold, continued against parent: " +
                   "; ".join(f"{h} {ARM_LABEL.get(a_, a_)} {p}&#8594;{c} of 64 (cos {pc:+.2f}&#8594;{cc:+.2f})" for h, a_, p, c, pc, cc in reading_bits) +
                   ". " + (f"Gains of 8 or more: {', '.join(b[0] for b in up)}. " if up else "") +
                   (f"Losses of 8 or more: {', '.join(b[0] for b in down)}. " if down else "") + "</p>")

    # --- plausibility
    prow = []
    for j in done:
        r = res[j["id"]]
        flags = []
        if max(r.get("pad_peak_thumb", 0), r.get("pad_peak_index", 0), r.get("pad_peak_middle", 0)) > 9: flags.append("pad peak &gt; 9 N")
        if (r.get("ctrl_gap_max_deg") or 0) > 60: flags.append("gap &gt; 60&#176;")
        if (r.get("clearance_min_mm_mean") or 99) < 10: flags.append("clearance &lt; 10 mm")
        if r.get("hold_rate", 1) < 1: flags.append("drops")
        prow.append([j["hand"], ARM_LABEL.get(j["arm"], j["arm"]),
                     f"{f1(r.get('pad_peak_thumb'))} / {f1(r.get('pad_peak_index'))} / {f1(r.get('pad_peak_middle'))}",
                     f"{f1(r.get('force_active_thumb'))} / {f1(r.get('force_active_index'))} / {f1(r.get('force_active_middle'))}",
                     f1(r.get("joint_speed_p99_deg_s"), "{:.0f}"), f1(r.get("residual_gt1_frac"), "{:.2f}"),
                     f1(r.get("ctrl_gap_max_deg"), "{:.0f}"), f1(r.get("raw_cmd_beyond_range_deg"), "{:.0f}"),
                     f"{f1(r.get('clearance_min_mm_mean'))} / {f1(r.get('clearance_end_mm_mean'))}",
                     ("; ".join(flags) if flags else "none", "cell c0" if flags else "cell c4")])
    table_plaus = table(["hand", "arm", "pad peak th / ix / md (N)", "pad force active th / ix / md (N)", "#joint speed p99 (&#176;/s)",
                         "#|a| &gt; 1 share", "#gap (&#176;)", "#raw cmd beyond range (&#176;)", "clearance min / end (mm)", "flags"], prow) if prow else "<p>No job has finished yet.</p>"

    # --- chain
    crow = []
    for j in done:
        crow.append([j["hand"], ARM_LABEL.get(j["arm"], j["arm"]),
                     chain_cell(os.path.join(R, f"{j['id']}_chain_pl25.json")), chain_cell(os.path.join(R, f"{j['id']}_chain_pl0.json"))])
    table_chain = table(["hand", "arm", "plate 25 mm: end cos", "plate 0: end cos"], crow) if crow else "<p>No chain run yet.</p>"

    # --- strips + videos (web/<id>.mp4 transcoded from videos/<id>.mp4 if missing)
    strips = []
    os.makedirs(os.path.join(R, "web"), exist_ok=True)
    for j in done:
        sp = os.path.join(R, f"{j['id']}_strip.png")
        if not os.path.exists(sp):
            continue
        src_v, web_v = os.path.join(R, "videos", f"{j['id']}.mp4"), os.path.join(R, "web", f"{j['id']}.mp4")
        if os.path.exists(src_v) and not os.path.exists(web_v):
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src_v, "-vf", "scale=640:480", "-c:v", "libx264", "-crf", "28",
                            "-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", web_v], stdin=subprocess.DEVNULL)
        r = res[j["id"]]; cj = jit(R, j["id"])
        video = (f'<video controls muted loop playsinline preload="metadata" width="640" height="480" src="web/{j["id"]}.mp4"></video>'
                 if os.path.exists(web_v) else "")
        strips.append(f'<figure><img src="{uri_jpeg(sp)}" alt="{j["id"]} filmstrip">{video}<figcaption>{j["hand"]}, '
                      f'{ARM_LABEL.get(j["arm"], j["arm"])}: nominal held {int(round(r["hold_rate"] * 64))}/64 at cos {r["final_cos_mean"]:+.3f}'
                      + (f', jittered {int(round(cj["hold_rate"] * 64))}/64 at {cj["final_cos_mean"]:+.3f}' if cj else "")
                      + f', clearance min {f1(r.get("clearance_min_mm_mean"))} mm. Steps 40, 58 (lifted, residual on), 80, 110, 140, 170, 200, 249.</figcaption></figure>')
    strips_html = "\n".join(strips) if strips else "<p>No filmstrip yet.</p>"

    # --- curves
    runs = {j["id"]: res[j["id"]]["run"] for j in done if res[j["id"]].get("run")}
    keys = ["Episode_Reward/target_axis_alignment", "Episode_Termination/tip_lost", "Metrics/lift_height/object_height", "Episode_Reward/grip_force_excess"]
    cv = curves(runs, keys) if runs else {}
    curves_html = "<p>No run yet.</p>"
    if cv:
        png = os.path.join(R, "20260920-robust_curves.png")
        plot_curves(cv, png)
        curves_html = f'<figure><img src="{uri(png)}" alt="training curves"><figcaption>Per-iteration alignment reward, tip-lost termination share, object height and grip excess for every finished job (10-iteration moving mean).</figcaption></figure>'

    n_done = len(done)
    lede = (f"{n_done} of {len(jobs)} continuation jobs have finished. " +
            (("Jittered hold (64 rollouts at &#177;3 mm / &#177;10&#176; with friction DR), continued against parent: " +
              "; ".join(f"{h} {c} against {p}" for h, a_, p, c, pc, cc in reading_bits) + ". ") if reading_bits else "") +
            ("Chain at plate 25 mm, held by: " + (", ".join(j["hand"] + ("" if j["arm"] == "clip" else " (separation)") for j in done
                                                          if chain_cell(os.path.join(R, f"{j['id']}_chain_pl25.json"))[1].endswith("c4")) or "none") + "."))
    sub = {"LEDE": lede, "TABLE_PLANT": table(["", "2026-09-19 tranche", "here", "why"], [list(r) for r in PLANT_ROWS]),
           "TABLE_MAIN": table_main, "READING": reading, "TABLE_PLAUS": table_plaus, "TABLE_CHAIN": table_chain,
           "STRIPS": strips_html, "CURVES": curves_html, "N_JOBS": str(len(jobs)), "N_DONE": str(n_done),
           "BUILT": time.strftime("%Y-%m-%d %H:%M")}
    tpl = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "robust_tranche_page.template.html")).read()
    for k, v in sub.items():
        tpl = tpl.replace("{{" + k + "}}", v)
    left = [ln for ln in tpl.splitlines() if "{{" in ln]
    assert not left, left[:3]
    out = os.path.join(R, "20260920-robust_reorient_policies.html")
    open(out, "w").write(tpl)
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
