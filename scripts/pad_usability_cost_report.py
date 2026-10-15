#!/usr/bin/env python3
"""Tables, key numbers and figures of U1 (per-step cost of the fingertip contact models) from its rows.

Reads docs/experiments/20261014-pad_usability/cost/{cpu_timing,rl_scene_check,gpu_export,gpu_timing,gpu_testspeed,
ppo_probes}.jsonl and the 40 M-step servo-plant training logs of 2026-10-11 (iteration times), and writes
  cost/report_U1.json            the tables and key numbers, each with its row file and how it was computed
  cost/media/cpu_step_cost.png   CPU time per 1 ms step at the held grip by model and stage, both cells
  cost/media/gpu_ppo_cost.png    MuJoCo-Warp time per world-step against batch size, PPO iteration time and peak
                                 GPU memory of the 12-iteration probes

    .venv/bin/python scripts/pad_usability_cost_report.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COST = ROOT / "docs/experiments/20261014-pad_usability/cost"
MEDIA = COST / "media"
REL = "docs/experiments/20261014-pad_usability/cost/"
LOG40 = ROOT / "logs/20261011-servo_plant_policies"
CELLS = {"cylinder:16:0.625": "16 mm cylinder", "cylinder:25:0.125": "25 mm cylinder"}
CPU_MODELS = [("pads", "1 mm pads"), ("pads_f", "1 mm pads, cylinder only"), ("pads_fx", "1 mm pads, filter compiled"),
              ("pads_fs", "1 mm pads, GPU solver (10 line-search it.)"),
              ("pads_fs50", "1 mm pads, tol 1e-6, 20 it., 50 line-search it."), ("pads2", "2 mm pads"),
              ("pads2_f", "2 mm pads, cylinder only"), ("pt", "Block mesh, point contact"),
              ("pt4", "Block mesh, condim 4"), ("ptmc", "Block mesh, multi-CCD")]
FIG1 = [("pads", "1 mm pads"), ("pads_fs50", "1 mm pads, tol 1e-6"), ("pads2", "2 mm pads"),
        ("ptmc", "Mesh, multi-CCD"), ("pt", "Mesh, point contact"), ("pt4", "Mesh, condim 4")]
STAGE_GROUPS = [("Kinematics, broadphase", ("POS_KINEMATICS", "POS_INERTIA", "COL_BROAD")),
                ("Narrowphase", ("COL_NARROW",)),
                ("Constraint assembly, solve", ("POS_MAKE", "POS_PROJECT", "CONSTRAINT")),
                ("Other", None)]
STAGE_GRAYS = ["#2f2e2c", "#5f5e5a", "#8c8b85", "#b3b2ab"]
GPU_VARIANTS = [("servo_pads1", "1 mm pads", "#2a78d6"), ("servo_pads2", "2 mm pads", "#eb6834"),
                ("servo_mesh", "Block mesh", "#1baf7a"), ("servo_meshc4", "Block mesh, condim 4", "#eda100")]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"


def rows(name):
    p = COST / name
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def cpu_get(R, cell, model, state, kind, n=0):
    for r in R:
        if (r["cell"], r["model"], r["state"], r["kind"], r.get("nthread", 0)) == (cell, model, state, kind, n):
            return r
    return None


def grip_stages(R, cell, model):
    """us per step of the stage groups at the own-plant grip: timer-pass shares scaled to the step-pass median."""
    st, tm = cpu_get(R, cell, model, "grip", "step"), cpu_get(R, cell, model, "grip", "timer")
    total = st["step_us"]["median"]
    sh = {k: v["median"] for k, v in tm["stage_share"].items()}
    out, used = {}, 0.0
    for name, keys in STAGE_GROUPS:
        if keys is None:
            out[name] = total * max(0.0, 1.0 - used)
        else:
            s = sum(sh[k] for k in keys)
            used += s
            out[name] = total * s
    return total, out


def log_iter_times(path):
    txt = path.read_text(errors="replace")
    return np.array([float(x) for x in re.findall(r"Iteration time:\s*([0-9.]+)s", txt)])


def asked(r) -> dict:
    """Largest per-world buffer sizes the trainer asked for in a probe log's overflow messages."""
    log = ROOT / "logs/20261014-pad_usability/rl_probes" / f"train_{r['variant']}_buf{r['buffers'].replace(',', '-')}_r{r['rep']}.log"
    if not log.exists():
        return {}
    out = {}
    for ln in log.read_text(errors="replace").splitlines():
        m = re.match(r"^nefc overflow - please increase njmax to (\d+)$", ln)
        if m:
            out["njmax"] = max(out.get("njmax", 0), int(m.group(1)))
        m = re.match(r"^contact match overflow: please increase Option.contact_sensor_maxmatch to (\d+)$", ln)
        if m:
            out["maxmatch"] = max(out.get("maxmatch", 0), int(m.group(1)))
    return out


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rep = {"tables": [], "key_numbers": []}
    K = rep["key_numbers"]

    # ------------------------------------------------------------------ CPU
    R = rows("cpu_timing.jsonl")
    if R:
        tab = []
        for cell, cname in CELLS.items():
            p3 = cpu_get(R, cell, "pads", "grip_ref", "rollout", 3)
            for m, label in CPU_MODELS:
                sg, sr, hg = (cpu_get(R, cell, m, s, k) for s, k in (("grip", "step"), ("grip_ref", "step"),
                                                                     ("grip", "held")))
                r1, r3 = cpu_get(R, cell, m, "grip_ref", "rollout", 1), cpu_get(R, cell, m, "grip_ref", "rollout", 3)
                if sg is None:
                    continue
                own = f"{sg['step_us']['median']:.1f}" if hg["held"] else f"{sg['step_us']['median']:.1f} (grip lost)"
                tab.append([cname, label, str(sg["ngeom"]), own, f"{sr['step_us']['median']:.1f}",
                            f"{sg['ncon_mean']:.0f}", f"{sg['nefc_mean']:.0f}", f"{r1['wall_s']['median']:.2f}",
                            f"{r3['wall_s']['median']:.3f}", f"{r3['wall_s']['median'] / p3['wall_s']['median']:.2f}",
                            f"{r3['plans_reset']}/64"])
        rep["tables"].append({
            "title": "CPU cost of the fingertip models at the three-finger grip (MuJoCo 3.6.0, 1 ms step, E-cores 17-19)",
            "columns": ["Cell", "Model", "Geoms", "Grip step, own plant (us)", "Step from the pads' grip (us)",
                        "Contacts", "Constraint rows", "64-plan replan, 1 thread (s)", "64-plan replan, 3 threads (s)",
                        "Replan / 1 mm pads", "Plans reset"], "rows": tab})
        for cell, cname in CELLS.items():
            tag = cname.split()[0]
            for m in ("pads", "pads_f", "pads_fs50", "pads2", "pt", "pt4", "ptmc"):
                r = cpu_get(R, cell, m, "grip", "step")
                K.append({"name": f"CPU step at the grip, {m}, {cname}", "value": f"{r['step_us']['median']:.1f}",
                          "unit": "us per 1 ms step", "n": "3 repeats x 500 steps",
                          "source": REL + "cpu_timing.jsonl",
                          "how": f"row cell={cell} model={m} state=grip kind=step: step_us.median "
                                 f"(range {r['step_us']['min']:.1f}-{r['step_us']['max']:.1f})"})
            for m in ("pads", "pads2", "pt", "pt4", "ptmc", "pads_fs50"):
                r = cpu_get(R, cell, m, "grip_ref", "rollout", 3)
                K.append({"name": f"64-plan replanning at 3 threads, {m}, {cname}", "value": f"{r['wall_s']['median']:.3f}",
                          "unit": "s", "n": "3 repeats of 64 plans x 500 steps", "source": REL + "cpu_timing.jsonl",
                          "how": f"row cell={cell} model={m} state=grip_ref kind=rollout nthread=3: wall_s.median"})
            r = cpu_get(R, cell, "pads_fs", "grip_ref", "rollout", 3)
            K.append({"name": f"Planner rollouts reset by a bad acceleration, pads_fs, {cname}",
                      "value": f"{r['plans_reset']}/64", "unit": "plans", "n": "64 plans x 500 steps",
                      "source": REL + "cpu_timing.jsonl",
                      "how": f"row cell={cell} model=pads_fs state=grip_ref kind=rollout: plans_reset (time goes back)"})
            del tag
        # figure 1
        plt.rcParams.update({"font.size": 9.5, "font.family": "DejaVu Sans", "axes.edgecolor": "#c3c2b7",
                             "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK})
        fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.5), dpi=150, sharey=True)
        ymax = 0.0
        for ax, (cell, cname) in zip(axs, CELLS.items()):
            for i, (m, label) in enumerate(FIG1):
                total, parts = grip_stages(R, cell, m)
                left = 0.0
                for k, (name, _) in enumerate(STAGE_GROUPS):
                    w = parts[name]
                    ax.barh(i, w, left=left, height=0.62, color=STAGE_GRAYS[k], edgecolor="white", linewidth=1.0,
                            label=name if (i == 0 and cell == next(iter(CELLS))) else None)
                    left += w
                nc = cpu_get(R, cell, m, "grip", "step")["ncon_mean"]
                ax.text(total + 3, i, f"{total:.0f} µs, {nc:.0f} contacts", va="center", ha="left", color=INK,
                        fontsize=8.6)
                ymax = max(ymax, total)
            ax.set_title(cname, fontsize=10, color=INK)
            ax.set_xlabel("CPU time per 1 ms step (µs)")
            ax.grid(axis="x", color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        for ax in axs:
            ax.set_xlim(0, ymax * 1.5)
        axs[0].set_yticks(range(len(FIG1)))
        axs[0].set_yticklabels([lab for _, lab in FIG1])
        axs[0].invert_yaxis()
        fig.legend(loc="lower center", ncol=4, frameon=False, fontsize=8.6, bbox_to_anchor=(0.5, -0.01),
                   handlelength=1.4, columnspacing=1.2)
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        MEDIA.mkdir(parents=True, exist_ok=True)
        fig.savefig(MEDIA / "cpu_step_cost.png")
        plt.close(fig)

    # ------------------------------------------------------------------ RL scenes and exports
    SC, EX = rows("rl_scene_check.jsonl"), rows("gpu_export.jsonl")
    if SC:
        rep["tables"].append({
            "title": "Servo-plant RL scenes, zero-action scripted grasp and lift at 64 envs",
            "columns": ["Variant", "Geoms", "Held worlds, step 60", "Held worlds, step 150", "Tool height, step 150 (mm)",
                        "Tip contacts, step 150 (thumb, index, middle)"],
            "rows": [[r["variant"], str(r["ngeom"]), f"{r['step60']['held_worlds']}/64", f"{r['step150']['held_worlds']}/64",
                      f"{r['step150']['tool_z_mm_median']:.1f}", ", ".join(f"{x:g}" for x in r["step150"]["tip_contacts_mean"])]
                     for r in SC]})
    # ------------------------------------------------------------------ GPU timing
    G, TS = rows("gpu_timing.jsonl"), rows("gpu_testspeed.jsonl")
    GT = [r for r in G if r.get("buffers") == "training"]
    G = [r for r in G if r.get("buffers", "exact") == "exact"]
    gtab = []
    gmean, gtrain = {}, {}
    for r in GT:
        if r.get("status") == "ok":
            gtrain[r["variant"]] = r
    for v, label, _ in GPU_VARIANTS:
        ex = next((r for r in EX if r["variant"] == v), None)
        for n in (1024, 2048, 4096, 8192):
            pts = [r for r in G if r["variant"] == v and r["nworld"] == n]
            ok = [r for r in pts if r.get("status") == "ok"]
            if ok:
                us = [r["us_per_world_step"] for r in ok]
                gmean[(v, n)] = float(np.mean(us))
                r0 = ok[0]
                gtab.append([label, str(n), " / ".join(f"{x:.3f}" for x in us), f"{r0['contacts_per_world']:.1f}",
                             f"{r0['candidates_per_world']:.0f}", f"{r0['nefc_world_mean']:.0f}", f"{r0['held_frac']:.3f}",
                             f"{max(r['gpu_proc_peak_mb'] for r in ok)}", str(r0["nconmax"]), str(r0["njmax"]),
                             "yes" if any(r["overflow"] for r in ok) else "no"])
            elif pts:
                gtab.append([label, str(n), pts[-1].get("status", "?") + ": " + pts[-1].get("reason", pts[-1].get("error", ""))[:60],
                             "", "", "", "", "", "", "", ""])
        del ex
        if v in gtrain:
            r0 = gtrain[v]
            gtab.append([label + " (trainer's buffers)", "2048", f"{r0['us_per_world_step']:.3f}",
                         f"{r0['contacts_per_world']:.1f}", f"{r0['candidates_per_world']:.0f}", f"{r0['nefc_world_mean']:.0f}",
                         f"{r0['held_frac']:.3f}", f"{r0['gpu_proc_peak_mb']}", str(r0["nconmax"]), str(r0["njmax"]),
                         "yes" if r0["overflow"] else "no"])
    if gtab:
        rep["tables"].append({
            "title": "MuJoCo-Warp physics at the held RL state (2 ms step, trainer solver options, 50-step CUDA graphs)",
            "columns": ["Variant", "Worlds", "us per world-step (repeats)", "Contacts per world", "Broadphase candidates per world",
                        "Constraint rows per world", "Tool held after 2.2 s", "Process GPU memory (MB)", "nconmax", "njmax",
                        "Overflow"], "rows": gtab})
        for v, label, _ in GPU_VARIANTS:
            if (v, 2048) in gmean:
                ok = [r for r in G if r["variant"] == v and r["nworld"] == 2048 and r.get("status") == "ok"]
                K.append({"name": f"GPU physics per world-step at 2,048 worlds, {label}", "value": f"{gmean[(v, 2048)]:.3f}",
                          "unit": "us", "n": f"{len(ok)} fresh processes x 20 replays of 50 steps",
                          "source": REL + "gpu_timing.jsonl",
                          "how": f"mean of us_per_world_step over rows variant={v} nworld=2048 status=ok"})
        for v, label, _ in GPU_VARIANTS:
            if v in gtrain:
                r0 = gtrain[v]
                K.append({"name": f"GPU physics per world-step at 2,048 worlds with the trainer's buffers, {label}",
                          "value": f"{r0['us_per_world_step']:.3f}", "unit": "us", "n": "1 fresh process x 20 replays of 50 steps",
                          "source": REL + "gpu_timing.jsonl",
                          "how": f"row variant={v} nworld=2048 buffers=training: us_per_world_step (nconmax {r0['nconmax']}, "
                                 f"njmax {r0['njmax']}); process GPU memory {r0['gpu_proc_peak_mb']} MB"})
        if ("servo_mesh", 2048) in gmean:
            for v, label, _ in GPU_VARIANTS[:2] + GPU_VARIANTS[3:]:
                if (v, 2048) in gmean:
                    K.append({"name": f"GPU physics ratio to the block mesh at 2,048 worlds, {label}",
                              "value": f"{gmean[(v, 2048)] / gmean[('servo_mesh', 2048)]:.2f}", "unit": "ratio",
                              "n": "2 fresh processes each", "source": REL + "gpu_timing.jsonl",
                              "how": f"mean us_per_world_step of {v} / that of servo_mesh, nworld=2048"})
    if TS:
        ev = []
        for v, label, _ in GPU_VARIANTS:
            r = next((x for x in TS[::-1] if x["variant"] == v and x.get("status") == "ok"), None)
            if r is None:
                continue
            mt = r["metrics"]
            top = {k: mt[k] for k in mt if k.startswith("step.") and k.count(".") == 1}
            tot = 1e6 / mt["steps_per_second"]
            parts = sorted(top.items(), key=lambda kv: -kv[1])[:5]
            ev.append([label, str(r["njmax"]), f"{tot:.3f}", "; ".join(f"{k[5:]} {val / 1000:.3f}" for k, val in parts),
                       f"{mt.get('ncon_mean', 0):.1f}", f"{mt.get('nefc_mean', 0):.0f}", f"{mt.get('solver_niter_mean', 0):.2f}",
                       f"{mt.get('total_memory', 0) / 2**20:.0f}", f"{mt.get('data_memory', 0) / 2**20:.0f}"])
        rep["tables"].append({
            "title": "mjwarp-testspeed at 2,048 worlds, 200 steps from world 0's held state with testspeed's command "
                     "noise (event trace on; runs without buffer overflow)",
            "columns": ["Variant", "njmax", "us per world-step", "Largest events (us per world-step)", "Contacts per world",
                        "Constraint rows", "Solver iterations", "GPU memory (MiB)", "Data memory (MiB)"], "rows": ev})

    # ------------------------------------------------------------------ PPO probes
    P_all = rows("ppo_probes.jsonl")
    P = [r for r in P_all if r.get("buffers", "env_build default") == "env_build default"]   # timing: default buffers
    PB = [r for r in P_all if r.get("buffers", "env_build default") != "env_build default"]  # exact-buffer probes
    ref = {}
    for arm, v in (("tpu27mesh", "servo_mesh"), ("tpu27pads1", "servo_pads1")):
        its = [log_iter_times(f) for f in sorted(LOG40.glob(f"train_20261011-d6_servo_{arm}_40M_s[01].log"))]
        if its:
            ref[v] = {"median_all": [float(np.median(x)) for x in its], "median_it3_12": [float(np.median(x[2:12])) for x in its]}
    ptab = []
    pmed = {}
    for v, label, _ in GPU_VARIANTS[::-1]:
        for r in [x for x in P if x["variant"] == v]:
            it = r.get("iteration_s") or {}
            ptab.append([label, str(r["rep"]), r["status"], f"{it.get('median', float('nan')):.2f}",
                         f"{it.get('q25', float('nan')):.2f}-{it.get('q75', float('nan')):.2f}",
                         f"{(r.get('collection_s') or {}).get('median', float('nan')):.2f}",
                         f"{(r.get('learning_s') or {}).get('median', float('nan')):.2f}",
                         f"{r.get('gpu_peak_mb')}", f"{r.get('gpu_proc_peak_mb')}", f"{r.get('host_rss_peak_gb')}",
                         str(r.get("overflow_lines"))])
            if r["status"] == "ok":
                pmed.setdefault(v, []).append(it["median"])
    if ptab:
        rep["tables"].append({
            "title": "12-iteration PPO probes at 2,048 envs (2026-10-11 servo-plant trainer command)",
            "columns": ["Variant", "Repeat", "Status", "Iteration time, it. 3-12, median (s)", "Interquartile range (s)",
                        "Collection time (s)", "Learning time (s)", "Peak GPU memory, whole GPU (MB)",
                        "Peak GPU memory, probe process (MB)", "Host peak RSS (GB)", "Overflow lines"], "rows": ptab})
        if PB:
            rep["tables"].append({
                "title": "12-iteration PPO probes with exact per-world contact buffers (nconmax, njmax, sensor maxmatch)",
                "columns": ["Variant", "Buffers", "Status", "Iterations logged", "Iteration time, it. 3-12, median (s)",
                            "Peak GPU memory, probe process (MB)", "Overflow lines", "Largest njmax asked",
                            "Largest maxmatch asked"],
                "rows": [[dict((g[0], g[1]) for g in GPU_VARIANTS)[r["variant"]], r["buffers"], r["status"],
                          str(r.get("iterations_logged")),
                          f"{(r.get('iteration_s') or {}).get('median', float('nan')):.2f}", f"{r.get('gpu_proc_peak_mb')}",
                          str(r.get("overflow_lines")), str(asked(r).get("njmax", "-")), str(asked(r).get("maxmatch", "-"))]
                         for r in PB]})
        if "servo_mesh" in pmed:
            mesh = float(np.mean(pmed["servo_mesh"]))
            for v, label, _ in GPU_VARIANTS:
                if v in pmed:
                    K.append({"name": f"PPO iteration time, {label}", "value": f"{float(np.mean(pmed[v])):.2f}", "unit": "s",
                              "n": f"{len(pmed[v])} probe(s) x iterations 3-12",
                              "source": REL + "ppo_probes.jsonl",
                              "how": f"mean over probes of iteration_s.median (iterations 3-12), variant={v}"})
                    K.append({"name": f"PPO iteration ratio to the block mesh, {label}",
                              "value": f"{float(np.mean(pmed[v])) / mesh:.2f}", "unit": "ratio",
                              "n": f"{len(pmed[v])} and {len(pmed['servo_mesh'])} probes",
                              "source": REL + "ppo_probes.jsonl", "how": f"mean iteration median of {v} / that of servo_mesh"})
            if ref:
                rep["iteration_40M"] = ref
                if "servo_pads1" in ref and "servo_mesh" in ref:
                    K.append({"name": "40 M-step iteration ratio, 1 mm pads to mesh, iterations 3-12",
                              "value": f"{np.mean(ref['servo_pads1']['median_it3_12']) / np.mean(ref['servo_mesh']['median_it3_12']):.2f}",
                              "unit": "ratio", "n": "2 seeds each, 10 iterations",
                              "source": "logs/20261011-servo_plant_policies/train_20261011-d6_servo_tpu27{pads1,mesh}_40M_s{0,1}.log",
                              "how": "median 'Iteration time' of iterations 3-12 per log, mean over seeds, pads1 / mesh"})
                    K.append({"name": "40 M-step iteration ratio, 1 mm pads to mesh, all 813 iterations",
                              "value": f"{np.mean(ref['servo_pads1']['median_all']) / np.mean(ref['servo_mesh']['median_all']):.2f}",
                              "unit": "ratio", "n": "2 seeds each, 813 iterations",
                              "source": "logs/20261011-servo_plant_policies/train_20261011-d6_servo_tpu27{pads1,mesh}_40M_s{0,1}.log",
                              "how": "median 'Iteration time' over the whole log, mean over seeds, pads1 / mesh"})
    # figure 2
    if gmean or pmed:
        fig, axs = plt.subplots(1, 3, figsize=(7.4, 3.0), dpi=150)
        ax = axs[0]
        for v, label, col in GPU_VARIANTS:
            ns = [n for n in (1024, 2048, 4096, 8192) if (v, n) in gmean]
            if ns:
                ax.plot(ns, [gmean[(v, n)] for n in ns], "-o", color=col, lw=2, ms=5, label=label,
                        markeredgecolor="white", markeredgewidth=1.0)
            if v in gtrain:
                ax.plot([2048], [gtrain[v]["us_per_world_step"]], "D", color="white", ms=5.5, markeredgecolor=col,
                        markeredgewidth=1.6)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1024, 2048, 4096, 8192])
        ax.set_xticklabels(["1,024", "2,048", "4,096", "8,192"])
        ax.set_xlabel("Worlds")
        ax.set_ylabel("Physics time per world-step (µs)")
        ax.set_ylim(bottom=0)
        ax.set_title("MuJoCo-Warp, held state", fontsize=9.5)
        ax = axs[1]
        for i, (v, label, col) in enumerate(GPU_VARIANTS):
            for r in [x for x in P if x["variant"] == v and x["status"] == "ok"]:
                it = r["iteration_s"]
                x = i + (0.12 * (r["rep"] - 0.5) if len([y for y in P if y["variant"] == v and y["status"] == "ok"]) > 1 else 0)
                ax.errorbar(x, it["median"], yerr=[[it["median"] - it["q25"]], [it["q75"] - it["median"]]], fmt="o",
                            color=col, ms=5, lw=1.5, capsize=0, markeredgecolor="white", markeredgewidth=1.0)
            if v in ref:
                ax.hlines(np.mean(ref[v]["median_it3_12"]), i - 0.3, i + 0.3, color=INK2, lw=1.2)
        ax.set_xticks(range(len(GPU_VARIANTS)))
        ax.set_xticklabels(["1 mm\npads", "2 mm\npads", "Mesh", "Mesh,\ncondim 4"], fontsize=8.6)
        ax.set_ylabel("PPO iteration time (s)")
        ax.set_ylim(bottom=0)
        ax.set_title("12-iteration probes", fontsize=9.5)
        ax = axs[2]
        for i, (v, label, col) in enumerate(GPU_VARIANTS):
            pk = [x["gpu_proc_peak_mb"] / 1000 for x in P if x["variant"] == v and x["status"] == "ok"
                  and x.get("gpu_proc_peak_mb")]
            ex = [x["gpu_proc_peak_mb"] / 1000 for x in PB if x["variant"] == v and x["status"] == "ok"
                  and x.get("gpu_proc_peak_mb")]
            w = 0.38 if ex else 0.55
            if pk:
                x0 = i - 0.2 if ex else i
                ax.bar(x0, max(pk), width=w, color=col)
                ax.text(x0, max(pk) + 0.2, f"{max(pk):.1f}", ha="center", va="bottom", fontsize=8.6, color=INK)
            if ex:
                ax.bar(i + 0.2, max(ex), width=w, color="white", edgecolor=col, hatch="////", linewidth=1.2)
                ax.text(i + 0.2, max(ex) + 0.2, f"{max(ex):.1f}", ha="center", va="bottom", fontsize=8.6, color=INK)
        ax.set_xticks(range(len(GPU_VARIANTS)))
        ax.set_xticklabels(["1 mm\npads", "2 mm\npads", "Mesh", "Mesh,\ncondim 4"], fontsize=8.6)
        ax.set_ylabel("Peak GPU memory of the probe (GB)")
        ax.set_ylim(0, 13.5)
        ax.set_title("Probe process, 16 GB card", fontsize=9.5)
        for ax in axs:
            ax.grid(axis="y", color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        h, lab = axs[0].get_legend_handles_labels()
        fig.legend(h, lab, loc="lower center", ncol=4, frameon=False, fontsize=8.6)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        fig.savefig(MEDIA / "gpu_ppo_cost.png")
        plt.close(fig)
    (COST / "report_U1.json").write_text(json.dumps(rep, indent=1))
    print(f"{len(rep['tables'])} tables, {len(K)} key numbers -> {(COST / 'report_U1.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
