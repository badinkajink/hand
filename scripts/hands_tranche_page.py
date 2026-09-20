#!/usr/bin/env python3
"""Build the cross-hand reorientation-policy page from the tranche queue's own outputs.

    python3 scripts/hands_tranche_page.py [--root docs/experiments/20260919-hands_tranche]

Reads queue.json, tranche_results.json, <id>_eval.json/.png, <id>_strip.png, <id>_trace.csv and
each run's tensorboard scalars; writes 20260919-reorient_policies_across_hands.html beside them.
"""
from __future__ import annotations

import argparse
import base64
import glob
import json
import mimetypes
import os
import re
import time

import numpy as np

HANDS = ["D6", "D3", "D4", "D5", "D7", "D1", "D2", "D8"]
ARM_LABEL = {"clip": "bounded residual (&#177;1 rad)", "clipsep": "bounded residual + index&#8211;middle clearance &#8805; 30 mm",
             "clip_scratch": "bounded residual, from scratch (60 M)", "asis": "unbounded residual"}


def uri_jpeg(p, width=1600, quality=82):
    """A PNG inlined as a JPEG at `width` px: sixteen 2560 px filmstrips at 1 MB each put the page over
    the 16 MB artifact limit; the full-resolution PNGs stay beside the page in the repo."""
    import io
    from PIL import Image
    im = Image.open(p).convert("RGB")
    im = im.resize((width, int(width * im.size[1] / im.size[0])), Image.LANCZOS)
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def uri(p):
    return f"data:{mimetypes.guess_type(p)[0]};base64," + base64.b64encode(open(p, "rb").read()).decode()


def table(head, rows):
    out = ['<div class="tw"><table><thead><tr>']
    for h in head:
        out.append(f'<th{" class=num" if h.startswith("#") else ""}>{h.lstrip("#")}</th>')
    out.append("</tr></thead><tbody>")
    for r in rows:
        out.append("<tr>")
        for h, c in zip(head, r):
            if isinstance(c, tuple):
                out.append(f'<td class="{c[1]}">{c[0]}</td>')
            else:
                out.append(f'<td{" class=num" if h.startswith("#") else ""}>{c}</td>')
        out.append("</tr>")
    out.append("</tbody></table></div>")
    return "\n".join(out)


def cellc(frac, text):
    cls = "c4" if frac >= 0.99 else "c3" if frac >= 0.6 else "c2" if frac >= 0.4 else "c1" if frac > 0 else "c0"
    return (text, f"cell {cls}")


def deg(c):
    return f"{np.degrees(np.arccos(np.clip(c, -1, 1))):.0f}"


def f1(v, fmt="{:.1f}", none="&#8211;"):
    return none if v is None or (isinstance(v, float) and np.isnan(v)) else fmt.format(v)


def gate_periods(logpath):
    """(start, end) of stretches the driver spent waiting for the launch gate, from its log."""
    if not os.path.exists(logpath):
        return []
    waits, cur = [], None
    for ln in open(logpath):
        m = re.match(r"(\S+ \S+) \[(\S+)\] (.*)", ln)
        if not m:
            continue
        t, tag, msg = m.groups()
        if "gate closed, waiting" in msg or "resguard refused" in msg:
            if cur is None:
                cur = [t, t, tag, msg.split(":")[-1].strip()[:80]]
            else:
                cur[1] = t
        elif ("TRAIN" in msg or "trained in" in msg) and cur is not None:
            cur[1] = t
            waits.append(cur); cur = None
    if cur is not None:
        waits.append(cur + ["(still waiting)"])
    return waits


def curves(runs, keys):
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception:
        return {}
    out = {}
    for rid, rdir in runs.items():
        tb = os.path.join(rdir, "tensorboard")
        if not os.path.isdir(tb):
            continue
        ea = EventAccumulator(tb); ea.Reload()
        out[rid] = {}
        for k in keys:
            if k in ea.Tags()["scalars"]:
                out[rid][k] = [(e.step, e.value) for e in ea.Scalars(k)]
    return out


def plot_curves(cv, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = [("Episode_Reward/target_axis_alignment", "alignment reward per episode"),
            ("Episode_Reward/grip_force_excess", "grip-excess term per episode"),
            ("Episode_Reward/finger_separation", "separation term per episode"),
            ("Episode_Termination/tip_lost", "tip-lost terminations per reset batch"),
            ("Metrics/lift_height/object_height", "mean object height (m)")]
    fig, axes = plt.subplots(1, len(keys), figsize=(3.4 * len(keys), 3.4), dpi=150)
    cmap = plt.get_cmap("tab20")
    for i, (rid, series) in enumerate(sorted(cv.items())):
        col = cmap(i % 20)
        for ax, (k, lab) in zip(axes, keys):
            if k not in series or len(series[k]) < 2:
                continue
            s, v = zip(*series[k])
            v = np.asarray(v, float); s = np.asarray(s, float)
            kk = 9
            if len(v) >= kk:
                v2 = np.convolve(v, np.ones(kk) / kk, mode="valid"); s2 = s[kk // 2: len(v) - (kk - 1 - kk // 2)]
            else:
                v2, s2 = v, s
            ax.plot(s2, v2, color=col, lw=1.3, label=rid)
            ax.set_title(lab, fontsize=9); ax.set_xlabel("PPO iteration", fontsize=8); ax.grid(alpha=0.25)
            ax.tick_params(labelsize=7)
    axes[3].set_ylim(0, 8)
    axes[0].legend(fontsize=6, frameon=False, ncol=2)
    fig.tight_layout(); fig.savefig(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="docs/experiments/20260919-hands_tranche")
    a = ap.parse_args()
    R = a.root
    q = json.load(open(os.path.join(R, "queue.json")))
    res_path = os.path.join(R, "tranche_results.json")
    res = {r["id"]: r for r in (json.load(open(res_path)) if os.path.exists(res_path) else [])}
    evals = {}
    for j in q["jobs"]:
        p = os.path.join(R, f"{j['id']}_eval.json")
        if os.path.exists(p):
            evals[j["id"]] = json.load(open(p))
    jobs = q["jobs"]
    done = [j for j in jobs if j.get("status") == "done" and j["id"] in evals]
    n_done, n_fail = len(done), sum(1 for j in jobs if j.get("status") == "failed")
    n_run = sum(1 for j in jobs if j.get("status") == "running")
    n_pend = sum(1 for j in jobs if j.get("status", "pending") == "pending")

    # --- cross-hand table
    head = ["hand", "arm", "held at end", "cos &#8805; 0.9 reached", "final cos", "#from vertical (deg)", "#steps to 0.9",
            "pad N active th / ix / md", "#index&#8211;middle clearance min / end (mm)", "#train (min)", "checkpoint"]
    rows = []
    for j in jobs:
        e = evals.get(j["id"]); r = res.get(j["id"], {})
        st = j.get("status", "pending")
        if e is None:
            rows.append([j["hand"], ARM_LABEL.get(j["arm"], j["arm"]) + (f" (seed {j['id'][-1]})" if not j["id"].endswith("s0") else ""),
                         (st, "cell c0" if st == "failed" else "cell"), "", "", "", "", "", "", f1(r.get("train_min")), f"<code>{r.get('run', '')}</code>" if r.get("run") else ""])
            continue
        n = e["n"]; held = int(round(e["hold_rate"] * n)); al = int(round(e["align_rate"] * n))
        rows.append([j["hand"], ARM_LABEL.get(j["arm"], j["arm"]) + (f" (seed {j['id'][-1]})" if not j["id"].endswith("s0") else ""),
                     cellc(held / n, f"{held}/{n}"), cellc(al / n, f"{al}/{n}"),
                     f"{e['final_cos_mean']:+.3f} &#177; {e['final_cos_sd']:.3f}", deg(e["final_cos_mean"]),
                     "&#8211;" if e.get("t_align_mean") is None else f"{e['t_align_mean']:.0f}",
                     f"{e.get('force_active_thumb', float('nan')):.1f} / {e.get('force_active_index', float('nan')):.1f} / {e.get('force_active_middle', float('nan')):.1f}",
                     f"{f1(e.get('clearance_min_mm_mean'))} / {f1(e.get('clearance_end_mm_mean'))}",
                     f1(r.get("train_min")), f"<code>{os.path.relpath(r.get('model', ''))}</code>"])
    table_main = table(head, rows)

    # --- plausibility table
    head2 = ["hand", "arm", "#pad peak N th / ix / md", "#residual |a|&gt;1 fraction", "#max |a|",
             "#command&#8211;achieved gap, clamped (deg)", "#raw command beyond joint range (deg)", "#joint speed p99 (deg/s)",
             "#clearance worst rollout (mm)", "verdict"]
    rows2 = []
    for j in done:
        e = evals[j["id"]]
        flags = []
        if e.get("pad_peak_thumb", 0) > 9 or e.get("pad_peak_index", 0) > 9 or e.get("pad_peak_middle", 0) > 9:
            flags.append("pad peak &gt; 9 N")
        if (e.get("ctrl_gap_max_deg") or 0) > 60:
            flags.append("gap &gt; 60&#176;")
        if (e.get("raw_cmd_beyond_range_deg") or 0) > 1:
            flags.append("commands beyond joint range")
        if (e.get("clearance_min_mm_worst") or 99) < 10:
            flags.append("clearance &lt; 10 mm")
        if e["hold_rate"] < 1.0:
            flags.append(f"{int(round((1 - e['hold_rate']) * e['n']))} dropped")
        verdict = ("; ".join(flags), "cell c1") if flags else ("no flags", "cell c4")
        rows2.append([j["hand"], ARM_LABEL.get(j["arm"], j["arm"]),
                      f"{e.get('pad_peak_thumb', 0):.1f} / {e.get('pad_peak_index', 0):.1f} / {e.get('pad_peak_middle', 0):.1f}",
                      f1(e.get("residual_gt1_frac"), "{:.2f}"), f1(e.get("residual_absmax"), "{:.1f}"),
                      f1(e.get("ctrl_gap_max_deg"), "{:.0f}"), f1(e.get("raw_cmd_beyond_range_deg"), "{:.0f}"),
                      f1(e.get("joint_speed_p99_deg_s"), "{:.0f}"), f1(e.get("clearance_min_mm_worst")), verdict])
    table_plaus = table(head2, rows2) if rows2 else "<p>No job has finished yet.</p>"

    # --- zero-shot transfer probes
    VAR_LABEL = {"base": "as trained", "tipmesh": "screw-tip mesh on the tool", "plate0": "plate at 0", "mu0.6": "pad &#956; 0.6",
                 "mu1.5": "pad &#956; 1.5", "mass1.3": "tool mass &#215; 1.3", "kp0.25": "servo kp 0.25"}
    trows, tvars = [], []
    for j in done:
        # the GPU pass (64 rollouts) runs after the last training job; the per-job CPU probe
        # (6 rollouts, 2 mm of spawn jitter) stands in for a variant the GPU pass lacks (the
        # tip-mesh scene does not compile under mjlab) and its base column is shown beside the
        # GPU's, since the two solvers disagree on the fragile draws
        tg = os.path.join(R, f"{j['id']}_transfer.json")
        tc = os.path.join(R, f"{j['id']}_transfer_cpu.json")
        t = json.load(open(tg)) if os.path.exists(tg) else {}
        cpu = json.load(open(tc)) if os.path.exists(tc) else {}
        if not t and not cpu:
            continue
        if not tvars:
            tvars = [v for v in VAR_LABEL if v in t or v in cpu]
        row = [j["hand"], ARM_LABEL.get(j["arm"], j["arm"])]
        def cell(e, tag=""):
            n = int(e.get("n", 64)); held = int(round(e["hold_rate"] * n))
            return cellc(min(held / n, e["align_rate"] if e["hold_rate"] >= 0.99 else held / n),
                         f"{held}/{n} &#183; {e['final_cos_mean']:+.2f}{tag}")
        for v in tvars:
            e = t.get(v, {})
            if e.get("hold_rate") is not None:
                row.append(cell(e))
            elif cpu.get(v, {}).get("hold_rate") is not None:
                row.append(cell(cpu[v], " (CPU)"))
            else:
                row.append(("&#8211;", "cell"))
        eb = cpu.get("base", {})
        row.append(cell(eb) if eb.get("hold_rate") is not None else ("&#8211;", "cell"))
        trows.append(row)
    table_transfer = table(["hand", "arm"] + [f"{VAR_LABEL[v]}: held &#183; cos" for v in tvars] + ["CPU replay of the trained scene"], trows) if trows else "<p>No transfer probe has run yet.</p>"

    # --- the policy inside the chain
    crows = []
    for j in done:
        row = [j["hand"], ARM_LABEL.get(j["arm"], j["arm"])]
        for pl in (25, 0):
            cp = os.path.join(R, f"{j['id']}_chain_pl{pl}.json")
            if not os.path.exists(cp):
                row += ["&#8211;", "&#8211;", "&#8211;"]
                continue
            c = json.load(open(cp))
            tr = c.get("policy_trace", [])
            def at(step):
                cand = [r for r in tr if r[0] == step]
                return cand[0] if cand else None
            peak = max((r[1] for r in tr), default=float("nan"))
            t_peak = next((r[0] for r in tr if r[1] == peak), None)
            t09 = next((r[0] for r in tr if r[1] >= 0.9), None)
            end = tr[-1] if tr else None
            held_end = bool(end and end[2] > 0.06 and sum(end[4:7]) >= 0.5) if end and len(end) >= 7 else None
            row.append(f"{peak:+.2f} at step {t_peak}" + (f", 0.9 at {t09}" if t09 is not None else ""))
            row.append(("held" if held_end else "lost", "cell c4" if held_end else "cell c0") if held_end is not None else "?")
            row.append(f"{end[1]:+.2f} &#183; {end[2]*1000:.0f} mm &#183; {end[4]:.1f}/{end[5]:.1f}/{end[6]:.1f} N" if end and len(end) >= 7 else "")
        crows.append(row)
    table_chain = table(["hand", "arm", "plate 25: peak cos", "at 5 s", "end: cos &#183; z &#183; pads N",
                         "plate 0: peak cos", "at 5 s", "end: cos &#183; z &#183; pads N"], crows) if crows else "<p>No chain run yet.</p>"
    ch_held, ch_lost = [], []
    for j in done:
        cp = os.path.join(R, f"{j['id']}_chain_pl25.json")
        if not os.path.exists(cp):
            continue
        tr = json.load(open(cp)).get("policy_trace", [])
        end = tr[-1] if tr else None
        seed = j["id"].rsplit("_s", 1)[-1]
        name = j["hand"] + (" (clip" if j["arm"] == "clip" else " (clip + separation") + (f", seed {seed})" if seed != "0" else ")")
        if end and end[2] > 0.06 and sum(end[4:7]) >= 0.5:
            ch_held.append(f"{name} at cos {end[1]:+.2f}")
        else:
            ch_lost.append(name)
    chain_lede = (f"Inside the UR5e chain with the training environment&#8217;s grasp and lift and the plate at 25 mm, one "
                  f"nominal rollout each: the tool is still held after 5 s of the policy by {', '.join(ch_held) or 'none'}; "
                  f"dropped by {', '.join(ch_lost) or 'none'}.") if (ch_held or ch_lost) else ""

    # --- the two D6 reference checkpoints in the final chain configuration (chain/*_final.json)
    def d6ref(name, label):
        outs = []
        for sd in range(4):
            cp = os.path.join(R, "chain", f"{name}_policy_pl25_s{sd}_final.json")
            if os.path.exists(cp):
                c = json.load(open(cp)); tr = c.get("policy_trace", [])
                end = tr[-1] if tr else None
                held = bool(end and end[2] > 0.06 and sum(end[4:7]) >= 0.5)
                outs.append((held, end[1] if end else float("nan"), sum(1 for v in end[4:7] if v >= 0.5) if end else 0,
                             sum(end[4:7]) if end else 0.0, next((r[0] for r in tr if r[1] >= 0.9), None)))
        if not outs:
            return f"{label}: not run yet"
        held = [o for o in outs if o[0]]
        cs = sorted(o[1] for o in held)
        t9 = [o[4] for o in outs if o[4] is not None]
        def rng(lo, hi, fmt):
            return fmt.format(lo) if fmt.format(lo) == fmt.format(hi) else f"{fmt.format(lo)}&#8211;{fmt.format(hi)}"
        return (f"{label} holds {len(held)} of {len(outs)}" +
                (f" at cos {rng(cs[0], cs[-1], '{:+.2f}')} on {rng(min(o[2] for o in held), max(o[2] for o in held), '{}')} pads, "
                 f"{rng(min(o[3] for o in held), max(o[3] for o in held), '{:.0f}')} N" if held else "") +
                (f", 0.9 first crossed at step {rng(min(t9), max(t9), '{}')} in {len(t9)} of {len(outs)}" if t9 else ""))
    d6ref_html = ("the 60 M checkpoint (17 N grip, the warm start of every job) " + d6ref("d6_60M_gp025", "") .strip() +
                  "; the 10.8 N grip finetune " + d6ref("d6_ft_gpm5", "").strip() + ".")

    # --- filmstrips
    strips = []
    for j in done:
        sp = os.path.join(R, f"{j['id']}_strip.png")
        if not os.path.exists(sp):
            continue
        e = evals[j["id"]]
        # the rollout video sits beside the page as web/<id>.mp4 (640x480, published with the page as a
        # supporting file); the 960x720 original is in videos/ (gitignored)
        vp = os.path.join(R, "web", f"{j['id']}.mp4")
        video = (f'<video controls muted loop playsinline preload="metadata" width="640" height="480" src="web/{j["id"]}.mp4">'
                 f'web/{j["id"]}.mp4</video>') if os.path.exists(vp) else ""
        strips.append(f'<figure><img src="{uri_jpeg(sp)}" alt="{j["id"]} filmstrip">{video}<figcaption>{j["hand"]}, {ARM_LABEL.get(j["arm"], j["arm"])}: '
                      f'held {int(round(e["hold_rate"] * e["n"]))}/{e["n"]}, final cos {e["final_cos_mean"]:+.3f}, '
                      f'clearance min {f1(e.get("clearance_min_mm_mean"))} mm. Steps 40, 58 (lifted, residual on), 80, 110, 140, 170, 200, 249; '
                      f'the video is the same rollout at 25 fps (10 s).</figcaption></figure>')
    strips_html = "\n".join(strips) if strips else "<p>No filmstrip yet.</p>"

    evpngs = []
    for j in done:
        p = os.path.join(R, f"{j['id']}_eval.png")
        if os.path.exists(p):
            evpngs.append(f'<figure><img src="{uri(p)}" alt="{j["id"]} eval"><figcaption>{j["hand"]}, {ARM_LABEL.get(j["arm"], j["arm"])}: 64 rollouts.</figcaption></figure>')
    ev_html = '<div class="duo">' + "\n".join(evpngs) + "</div>" if evpngs else ""

    # --- curves
    runs = {j["id"]: res[j["id"]]["run"] for j in jobs if j["id"] in res and res[j["id"]].get("run")}
    for j in jobs:
        if j.get("status") == "running" and j["id"] not in runs:
            g = sorted(glob.glob(f"results/rl/*-{j['id']}"))
            if g:
                runs[j["id"]] = g[-1]
    cv = curves(runs, ["Episode_Reward/target_axis_alignment", "Episode_Reward/grip_force_excess",
                       "Episode_Reward/finger_separation", "Episode_Termination/tip_lost", "Metrics/lift_height/object_height"])
    curves_png = os.path.join(R, "20260919-tranche_curves.png")
    if cv:
        plot_curves(cv, curves_png)
    curves_html = f'<figure><img src="{uri(curves_png)}" alt="training curves"><figcaption>Training curves of every job that has produced a run directory, 9-iteration running mean; all are 20 M-step finetunes (271 iterations) from the same checkpoint.</figcaption></figure>' if cv else ""

    # --- gate periods
    gp = gate_periods("logs/20260919-hands_tranche.log")
    gate_rows = [[g[0], g[1], g[2], g[3]] for g in gp]
    gate_html = table(["from", "to", "job waiting", "reason"], gate_rows) if gate_rows else "<p>The launch gate was open whenever the driver asked.</p>"

    # --- lede numbers
    held_hands = sorted({j["hand"] for j in done if evals[j["id"]]["hold_rate"] >= 0.99})
    turned = sorted({j["hand"] for j in done if evals[j["id"]]["align_rate"] >= 0.99 and evals[j["id"]]["hold_rate"] >= 0.99})
    clipsep_done = [j for j in done if j["arm"] == "clipsep"]
    sep_ok = sorted({j["hand"] for j in clipsep_done if (evals[j["id"]].get("clearance_min_mm_mean") or 0) >= 30})
    lede = (f"{n_done} of {len(jobs)} queued finetunes have finished ({n_fail} failed, {n_run} running, {n_pend} pending). "
            f"Hands that hold the tool in 64 of 64 rollouts under a bounded residual: {', '.join(held_hands) or 'none yet'}; "
            f"hands that also reach cos 0.9 in every rollout: {', '.join(turned) or 'none yet'}. "
            f"Separation arm keeping the index&#8211;middle chains 30 mm apart or more: {', '.join(sep_ok) or 'none yet'} "
            f"of {len(clipsep_done)} finished. " + chain_lede)
    sub = {"LEDE": lede, "TABLE_MAIN": table_main, "TABLE_PLAUS": table_plaus, "TABLE_TRANSFER": table_transfer, "TABLE_CHAIN": table_chain, "STRIPS": strips_html, "EVALS": ev_html,
           "D6REF": d6ref_html,
           "CURVES": curves_html, "GATE": gate_html, "N_JOBS": str(len(jobs)), "N_DONE": str(n_done),
           "BUILT": time.strftime("%Y-%m-%d %H:%M"), "INIT": q.get("init_checkpoint", "")}
    tpl = open("scripts/hands_tranche_page.template.html").read()
    for k, v in sub.items():
        tpl = tpl.replace("{{" + k + "}}", v)
    left = [ln for ln in tpl.splitlines() if "{{" in ln]
    assert not left, left[:3]
    out = os.path.join(R, "20260919-reorient_policies_across_hands.html")
    open(out, "w").write(tpl)
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
