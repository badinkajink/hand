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
import time

import numpy as np
import interactive_figure as IF
import retro_style
from policy_eval_suite import load_traces

HANDS = ["D6", "D3", "D4", "D5", "D7", "D1", "D2", "D8"]
ARM_LABEL = {"clip": "clipped residual (&#177;1 rad)", "clipsep": "clipped residual + index&#8211;middle clearance &#8805; 30 mm",
             "clip_scratch": "clipped residual, from scratch (60 M)", "asis": "unclipped residual"}


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


def reeval(R, run_dir, key):
    """Summary of the 2026-10-11 re-evaluation (scripts/policy_reeval_traces.py) of one run under one reset, or None."""
    p = os.path.join(R, "traces", f"{os.path.basename(run_dir)}_{key}.json")
    return json.load(open(p)) if run_dir and os.path.exists(p) else None


HAND_ORDER = ["D1", "D2", "D3", "D4", "D5", "D6", "D7"]
RESETS = [("nominal", "Nominal"), ("j3", "Spawn &#177;3 mm, &#177;10&#176;"), ("j3dr", "Spawn &#177;3 mm, &#177;10&#176;, friction DR")]
DASH = {("clip", "0"): "", ("clipsep", "0"): "7 4", ("clip", "1"): "2 3", ("clip", "2"): "10 3 2 3"}


def eval_figure(R, done, res, resets=RESETS, fid="eval-hands", caption=""):
    """One interactive figure for every finished job under each reset (traces/<run>.npz): colour by hand, dash by arm
    and seed, the reset as a radio choice."""
    col = {h: IF.PALETTE[i] for i, h in enumerate(HAND_ORDER)}
    items, ids_by = [], {}
    for j in done:
        run = os.path.basename(res[j["id"]]["run"])
        T = load_traces(os.path.join(R, "traces", f"{run}.npz"))
        seed = j["id"].rsplit("_s", 1)[-1] if "_s" in j["id"] else "0"
        lab = f"{j['hand']}, {'clipped residual' if j['arm'] == 'clip' else 'with separation term'}" + (f", seed {seed}" if seed != "0" else "")
        for key, _ in resets:
            if key not in T:
                continue
            sid = f"{j['id']}-{key}"
            items.append(dict(id=sid, label=lab, trace=T[key], color=col[j["hand"]], dash=DASH.get((j["arm"], seed), ""), width=1.6,
                              facets={"reset": key, "hand": j["hand"], "arm": j["arm"], "seed": seed}))
            for g in (j["hand"], j["arm"], f"s{seed}"):
                ids_by.setdefault(g, []).append(sid)
    hands = [h for h in HAND_ORDER if h in ids_by]
    seeds = sorted({it["facets"]["seed"] for it in items})
    legend = ([(h, col[h], "", ids_by[h]) for h in hands] +
              [("Clipped residual", "#000", "", [i for i in ids_by.get("clip", []) if i.split("-")[0].endswith("_s0")]),
               ("With separation term", "#000", "7 4", ids_by.get("clipsep", []))] +
              [(f"Seed {sd}", "#000", DASH[("clip", sd)], ids_by.get(f"s{sd}", [])) for sd in seeds if sd != "0"])
    facets = [{"key": "reset", "label": "Reset", "kind": "radio", "values": list(resets), "on": [resets[0][0]]},
              {"key": "hand", "label": "Hand", "values": [(h, h) for h in hands]},
              {"key": "arm", "label": "Arm", "values": [("clip", "Clipped residual"), ("clipsep", "With separation term")]},
              {"key": "seed", "label": "Seed", "values": [(sd, sd) for sd in seeds], "on": ["0"]}]
    return IF.eval_figure(fid, items, facets=facets, legend=legend, n_roll=0, show_band=False, caption=caption)


def jitter_text(R, done, res):
    """The perturbed-reset paragraph, from the 2026-10-11 re-evaluation on the policies' own plant (traces/), the published
    2026-09-20 evaluation (robust/) and its repeat with the tool on the scene's contact class."""
    held = lambda e: int(round(e["hold_rate"] * e["n"])) if e else None
    rows = []
    for j in done:
        run = res.get(j["id"], {}).get("run")
        seed = j["id"].rsplit("_s", 1)[-1]
        name = j["hand"] + (" with the separation term" if j["arm"] == "clipsep" else "") + (f" (clip, seed {seed})" if seed != "0" else "")
        own = [held(reeval(R, run, k)) for k in ("j3", "j3dr")]
        sc = [held(reeval(R, run, k)) for k in ("j3_scene", "j3dr_scene")]
        org = [held(json.load(open(q))) if os.path.exists(q) else None
               for q in (os.path.join(R, "robust", f"{j['id']}_eval_{m}.json") for m in ("j3", "j3dr"))]
        rows.append((name, own, sc, org, held(reeval(R, run, "nominal_scene"))))
    have = [r for r in rows if r[1][0] is not None]
    if not have:
        return ""

    def rng(v):
        v = sorted(v)
        return f"{v[0]}" if v[0] == v[-1] else f"{v[0]}&#8211;{v[-1]}"

    def by_hand(xs, k, i):                              # "the D2 and D3 policies 44-46"
        hands = sorted({r[0].split()[0] for r in xs})
        return (f"the {' and '.join(hands) if len(hands) < 3 else ', '.join(hands[:-1]) + ' and ' + hands[-1]} "
                f"polic{'y' if len(xs) == 1 else 'ies'} {rng([r[k][i] for r in xs])}")
    def keepers(keep, allr):                            # "the D2 and D3 policies (63-64) and D1 with the separation term (52)"
        hands = sorted({r[0].split()[0] for r in keep})
        whole = [h for h in hands if all(r in keep for r in allr if r[0].split()[0] == h)]
        part = [r for r in keep if r[0].split()[0] not in whole]
        bits = ([f"the {' and '.join(whole)} polic{'y' if len(whole) == 1 else 'ies'} "
                 f"({rng([r[4] for r in keep if r[0].split()[0] in whole])})"] if whole else [])
        bits += [f"{r[0]} ({r[4]})" for r in part]
        return " and ".join(bits) if len(bits) < 3 else ", ".join(bits[:-1]) + " and " + bits[-1]
    hi = [r for r in have if r[1][0] >= 52]
    lo = [r for r in have if r[1][0] < 52]
    dr_lo = [r for r in have if r[1][1] is not None and r[1][1] < 45]
    big = [r for r in have if r[3][0] is not None and r[1][0] - r[3][0] >= 16]
    t = (f"<p>On the plant they trained on, {len(hi)} of {len(have)} policies hold 52 or more of 64 rollouts with the tool "
         f"spawned &#177;3&#8202;mm and &#177;10&#176; off the trained pose" + (f", and {by_hand(lo, 1, 0)}" if lo else "") + ". "
         + (f"Friction DR on top leaves {by_hand(dr_lo, 1, 1)} of 64. " if dr_lo else ""))
    if big:
        d7 = [r[3][0] for r in big if r[0].startswith("D7")]
        t += (f"The evaluation published here on 2026-09-20 ran on two plants. While it ran, <code>env_build</code> began to copy the "
              "scene&#8217;s contact class onto the tool (solref 0.006, solimp 0.97&#8211;0.995, against MuJoCo&#8217;s defaults of "
              "0.02 and 0.9&#8211;0.95 in training). Its first four policies (D6 with seed 0 under both arms, D3 and D4 "
              "with the clipped residual) were therefore evaluated as trained. The other twelve were evaluated on the scene&#8217;s class, where "
              f"{len(big)} of them read {rng([r[3][0] for r in big])} of 64" + (f" (D7 {' and '.join(str(v) for v in sorted(d7))})" if d7 else "") + ".")
        nom = [r for r in have if r[4] is not None]
        if len(nom) == len(have):
            lo_n = [r for r in nom if r[4] < 32]
            keep = [r for r in nom if r[4] >= 32]
            t += (f" Repeated on the scene&#8217;s class on 2026-10-11 at the trained pose, with no jitter, {len(lo_n)} of the "
                  f"{len(have)} hold the tool in at most {max(r[4] for r in lo_n)} of 64 rollouts"
                  + (f" and only {keepers(keep, nom)} keep it" if keep else "")
                  + ". The drop in held rollouts comes from the contact class.")
        t += (" The CPU replays and the chain below also run the tool on the scene&#8217;s contact class, and the "
              "<a href=\"../20260920-robust_tranche/20260920-robust_reorient_policies.html\">2026-09-20 continuation</a> trains every "
              "checkpoint further on it, with the jitter.")
    return t + "</p>"


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
    """Training curves, 2 x 3 panels and the legend in the last cell. Drawn 1656 px wide for the 828 px column, so a
    font size of f pt shows at 1.39 f px: 9.5 pt and up keeps every label at 13 px or more (owner 2026-10-09)."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = [("Episode_Reward/target_axis_alignment", "Alignment reward per episode"),
            ("Episode_Reward/grip_force_excess", "Grip-excess term per episode"),
            ("Episode_Reward/finger_separation", "Separation term per episode"),
            ("Episode_Termination/tip_lost", "Tip-lost terminations per reset batch"),
            ("Metrics/lift_height/object_height", "Mean object height (m)")]
    fig, grid = plt.subplots(2, 3, figsize=(8.28, 6.4), dpi=200)
    axes = list(grid.flat)[:len(keys)]
    cmap = plt.get_cmap("tab20")
    for i, (rid, series) in enumerate(sorted(cv.items())):
        col = cmap(i % 20)
        for ax, (k, lab) in zip(axes, keys):
            ax.set_title(lab, fontsize=10.5); ax.set_xlabel("PPO iteration", fontsize=10); ax.grid(alpha=0.25)
            ax.tick_params(labelsize=9.5)
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
    axes[3].set_ylim(0, 8)
    leg = grid.flat[-1]
    leg.axis("off")
    hs, ls = axes[0].get_legend_handles_labels()
    leg.legend(hs, ls, fontsize=9.5, frameon=False, ncol=2, loc="center")
    fig.tight_layout(); fig.savefig(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="docs/experiments/20260919-hands_tranche")
    a = ap.parse_args()
    R = a.root
    q = json.load(open(os.path.join(R, "queue.json")))
    res_path = os.path.join(R, "tranche_results.json")
    res = {r["id"]: r for r in (json.load(open(res_path)) if os.path.exists(res_path) else [])}
    # evaluation numbers: the re-evaluation of 2026-10-11 where it exists (traces/), else the queue's own
    evals, evals_orig = {}, {}
    for j in q["jobs"]:
        p = os.path.join(R, f"{j['id']}_eval.json")
        if os.path.exists(p):
            evals_orig[j["id"]] = json.load(open(p))
        e = reeval(R, res.get(j["id"], {}).get("run"), "nominal")
        if e or j["id"] in evals_orig:
            evals[j["id"]] = e or evals_orig[j["id"]]
    when = sorted({e["evaluated"][:10] for e in evals.values() if e.get("evaluated")})
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

    # --- the same checkpoints under a perturbed reset: the 2026-10-11 re-evaluation on the policies' own plant (traces/), the
    # published 2026-09-20 evaluation (robust/<id>_eval_j3*.json) and its repeat with the tool on the scene's contact class
    def held(e):
        return f"{int(round(e['hold_rate'] * e['n']))}" if e else "&#8211;"
    jrows = []
    for j in done:
        e0, run = evals[j["id"]], res.get(j["id"], {}).get("run")
        row = [j["hand"], ARM_LABEL.get(j["arm"], j["arm"]),
               cellc(e0["hold_rate"], f"{int(round(e0['hold_rate'] * e0['n']))}/{e0['n']} &#183; {e0['final_cos_mean']:+.2f}")]
        for mode in ("j3", "j3dr"):
            e = reeval(R, run, mode)
            row.append(cellc(e["hold_rate"], f"{int(round(e['hold_rate'] * e['n']))}/{e['n']} &#183; {e['final_cos_mean']:+.2f}") if e
                       else ("&#8211;", "cell"))
        e = reeval(R, run, "j3dr")
        row.append(f1(e.get("clearance_min_mm_mean")) if e else "&#8211;")
        orig = [json.load(open(p)) if os.path.exists(p) else None for p in (os.path.join(R, "robust", f"{j['id']}_eval_{m}.json") for m in ("j3", "j3dr"))]
        row.append(f"{held(orig[0])} / {held(orig[1])}")
        row.append(f"{held(reeval(R, run, 'nominal_scene'))} / {held(reeval(R, run, 'j3_scene'))} / {held(reeval(R, run, 'j3dr_scene'))}")
        jrows.append(row)
    table_jitter = table(["hand", "arm", "nominal: held &#183; cos", "spawn &#177;3 mm, &#177;10&#176;: held &#183; cos",
                          "+ friction DR: held &#183; cos", "#clearance, + DR (mm)", "#2026-09-20 evaluation: held, &#177;3 mm / + DR",
                          "#tool on the scene&#8217;s contact class: held, nominal / &#177;3 mm / + DR"], jrows) if jrows else "<p>Not run yet.</p>"

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
    chain_lede = (f"; in the UR5e chain {len(ch_held)} of "
                  f"{len(ch_held) + len(ch_lost)} policies hold the tool after their 5&#8202;s.") if (ch_held or ch_lost) else "."

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
        # the rollout video web/<id>.mp4 (640x480; the 960x720 original is in videos/, gitignored) is inlined, so the
        # page opens from the file system (owner, 2026-10-09)
        vp = os.path.join(R, "web", f"{j['id']}.mp4")
        video = (f'<video controls muted loop playsinline preload="metadata" width="640" height="480" src="{uri(vp)}">'
                 f'</video>') if os.path.exists(vp) else ""
        strips.append(f'<figure><img src="{uri_jpeg(sp)}" alt="{j["id"]} filmstrip">{video}<figcaption>{j["hand"]}, {ARM_LABEL.get(j["arm"], j["arm"])}: '
                      f'held {int(round(e["hold_rate"] * e["n"]))}/{e["n"]}, final cos {e["final_cos_mean"]:+.3f}.</figcaption></figure>')
    strips_html = "\n".join(strips) if strips else "<p>No filmstrip yet.</p>"

    # --- evaluation traces of every policy (interactive; 2026-10-11 re-evaluation)
    have = [j for j in done if res.get(j["id"], {}).get("run") and os.path.exists(os.path.join(R, "traces", os.path.basename(res[j["id"]]["run"]) + ".npz"))]
    ev_html = eval_figure(R, have, res, caption=(
        "Medians over 64 rollouts of every policy under the nominal reset (the default view) and the two perturbed resets of the "
        "next section: the signed cosine, the tool height, the pad force sum and the share of rollouts held. Colour: hand; "
        "dashes: the separation arm and the D6 seeds 1 and 2, which the seed box adds.")).html() if have else ""

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
    # --- lede numbers
    held_hands = sorted({j["hand"] for j in done if evals[j["id"]]["hold_rate"] >= 0.99})
    turned = sorted({j["hand"] for j in done if evals[j["id"]]["align_rate"] >= 0.99 and evals[j["id"]]["hold_rate"] >= 0.99})
    clipsep_done = [j for j in done if j["arm"] == "clipsep"]
    sep_ok = sorted({j["hand"] for j in clipsep_done if (evals[j["id"]].get("clearance_min_mm_mean") or 0) >= 30})
    def andjoin(xs):
        return "none" if not xs else xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]
    every = "Every hand has a policy" if len(held_hands) == len({j["hand"] for j in done}) else andjoin(held_hands) + " have policies"
    j3 = [int(round(e["hold_rate"] * e["n"])) for e in (reeval(R, res.get(j["id"], {}).get("run"), "j3") for j in done) if e]
    lede = (f"{ {16: 'Sixteen'}.get(n_done, str(n_done)) } finetunes of one D6 policy train reorientation policies for the seven "
            f"deployed hands, residual clipped at &#177;1&#8202;rad. {every} holding the tool in 64 of 64 rollouts at the trained pose; "
            f"those of {andjoin(turned)} also turn it past cos 0.9 in every rollout. The clearance penalty keeps index and "
            f"middle 30&#8202;mm apart on {len(sep_ok) or 'none'} of {len(clipsep_done)} hands. " + (
                f"Spawned &#177;3&#8202;mm and &#177;10&#176; off that pose, they hold {min(j3)}&#8211;{max(j3)} of 64 on their own plant"
                if j3 else "") + chain_lede)
    sub = {"LEDE": lede, "JITTER_TEXT": jitter_text(R, done, res), "REEVAL": (
               f"evaluation numbers and figure from the re-evaluation of {', '.join(when)} on each policy&#8217;s own plant "
               f"(<code>traces/</code>)" if when else ""), "TABLE_MAIN": table_main, "TABLE_PLAUS": table_plaus, "TABLE_TRANSFER": table_transfer, "TABLE_CHAIN": table_chain, "STRIPS": strips_html, "EVALS": ev_html,
           "D6REF": d6ref_html, "TABLE_JITTER": table_jitter,
           "CURVES": curves_html, "N_JOBS": str(len(jobs)), "N_DONE": str(n_done),
           "BUILT": "2026-09-20 11:39", "INIT": q.get("init_checkpoint", "")}
    tpl = open("scripts/hands_tranche_page.template.html").read()
    for k, v in sub.items():
        tpl = tpl.replace("{{" + k + "}}", v)
    left = [ln for ln in tpl.splitlines() if "{{" in ln]
    assert not left, left[:3]
    out = os.path.join(R, "20260919-reorient_policies_across_hands.html")
    page = IF.add_assets(retro_style.apply(tpl))  # plain page style (owner, 2026-10-09); interactive figures (2026-10-11)
    open(out, "w").write(page)
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB, figure data {IF.data_bytes(page) / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
