#!/usr/bin/env python3
"""Re-evaluate the policies behind the evaluation figures of four result pages and keep their per-step traces.

    nohup setsid python3 scripts/policy_reeval_traces.py run > logs/20261011-reeval/queue.log 2>&1 &
    python3 scripts/policy_reeval_traces.py list                 # the 75 evaluations and their state
    python3 scripts/policy_reeval_traces.py compare              # re-evaluated against original, per page folder

Pages: the D6 60 M policy (20260917-d6_cal_60M), the bench partial reorientation (20260916-tip_gait), the policies
across the hands (20260919-hands_tranche, nominal and both jittered resets) and their continuation
(20260920-robust_tranche, nominal and jittered with friction DR). Each evaluation is scripts/policy_eval_suite.py with
the flags of the original (64 rollouts, 250 steps, the run's own timing and tool contact class), one GPU process at a
time through `resguard.sh run` (2.35 GB measured, so --mem 3100M), its own WARP_CACHE_PATH (deleted afterwards), and
writes <page folder>/traces/<run>_<key>.json and the key <key> of <page folder>/traces/<run>.npz. A finished
evaluation is skipped on a rerun.

`compare` writes <page folder>/traces/reeval_compare.json: every summary number of the original JSON next to the
re-evaluated one and the rollout spread, which is the standard deviation over the 64 rollouts of the per-rollout
quantity (from the new traces, or the original's own sd where it is larger); for a rate it is the binomial sd of the
count, at least one rollout. A difference larger than the spread is flagged.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from policy_eval_suite import load_traces  # noqa: E402

RESGUARD = Path.home() / ".claude/bin/resguard.sh"
UV = ["uv", "run", "--extra", "rl", "--extra", "gpu", "python", "scripts/policy_eval_suite.py"]
COMMON = ["--closed-ctrl-from-keyframe", "open_ik", "--open-finger-from-keyframe", "--lift-delta", "0.1",
          "--steps", "250", "--n", "64", "--held-min-n", "0.5", "--floor-z", "0.06"]
COND = {"nominal": [], "j3": ["--spawn-jitter-mm", "3", "--spawn-yaw-deg", "10"],
        "j3dr": ["--spawn-jitter-mm", "3", "--spawn-yaw-deg", "10", "--friction-dr"],
        "nominal_r0": ["--residual-from-step", "0"],
        "nominal_r0_stochastic": ["--residual-from-step", "0", "--stochastic"],
        # the 2026-09-20 jittered evaluations of the 09-19 policies ran while env_build's tool contact-class change
        # (35f04bff, committed 09:24) was in the working tree; these keys repeat them on the scene's contact class
        "j3_scene": ["--spawn-jitter-mm", "3", "--spawn-yaw-deg", "10", "--object-contact", "scene"],
        "j3dr_scene": ["--spawn-jitter-mm", "3", "--spawn-yaw-deg", "10", "--friction-dr", "--object-contact", "scene"]}
E = ROOT / "docs" / "experiments"
D6 = "results/phase1/real_v1/20260916-sv1_u0308_b050_cal"
LOGDIR = ROOT / "logs" / "20261011-reeval"


def jobs() -> list[dict]:
    """One dict per evaluation: page folder, run, checkpoint, morphology run, condition key, original JSON."""
    out = []

    def add(folder, ckpt, morph, key, orig, sub=None, label=None):
        run = Path(ckpt).parent.parent.name
        out.append(dict(folder=folder, run=run, policy=ckpt, morph=morph, key=key, orig=str(orig), sub=sub,
                        label=label or f"{run}_{key}"))

    p = E / "20260917-d6_cal_60M"
    add(p, "results/rl/20260916-2142-d6_cal_reorient_s0/tensorboard/model_270.pt", D6, "nominal",
        p / "20260917-d6_cal_20M_m270_eval_fixed.json")
    m812 = "results/rl/20260917-1141-d6_cal_reorient_gp025_60M_s0/tensorboard/model_812.pt"
    add(p, m812, D6, "nominal", p / "20260917-d6_cal_60M_eval.json")
    add(p, m812, D6, "nominal_r0", p / "20260917-d6_cal_60M_eval_residual0.json", sub="deterministic")
    add(p, m812, D6, "nominal_r0_stochastic", p / "20260917-d6_cal_60M_eval_stochastic.json")
    add(p, "results/rl/20260917-1653-d6_cal_reorient_gpm025_60M_s0/tensorboard/model_812.pt", D6, "nominal",
        p / "20260917-d6_cal_60M_gpm025_eval.json")
    add(p, "results/rl/20260917-1844-d6_cal_reorient_ft_gpm5_20M_s0/tensorboard/model_270.pt", D6, "nominal",
        p / "20260917-d6_cal_ft_gpm5_eval.json")
    p = E / "20260916-tip_gait"
    add(p, "results/rl/20260916-2142-d6_cal_reorient_s0/tensorboard/model_270.pt", D6, "nominal_r0",
        p / "d6_cal_rl_eval.json")
    for folder, conds in (("20260919-hands_tranche", ("nominal", "j3", "j3dr", "j3_scene", "j3dr_scene")),
                          ("20260920-robust_tranche", ("nominal", "j3dr"))):
        p = E / folder
        q = {j["id"]: j for j in json.load(open(p / "queue.json"))["jobs"]}
        for r in json.load(open(p / "tranche_results.json")):
            if r.get("status") != "done":
                continue
            ck = os.path.relpath(r["model"], ROOT)
            for c in conds:
                orig = {"nominal": p / f"{r['id']}_eval.json",
                        "j3": p / "robust" / f"{r['id']}_eval_j3.json",
                        "j3dr": (p / "robust" / f"{r['id']}_eval_j3dr.json") if folder.endswith("hands_tranche")
                        else p / f"{r['id']}_eval_j3dr.json"}[c.replace("_scene", "")]
                add(p, ck, q[r["id"]]["morph_run"], c, orig, label=f"{r['id']}_{c}")
    return out


def new_json(j) -> Path:
    return Path(j["folder"]) / "traces" / f"{j['run']}_{j['key']}.json"


def is_done(j) -> bool:
    nj = new_json(j)
    if not nj.exists():
        return False
    npz = Path(j["folder"]) / "traces" / f"{j['run']}.npz"
    try:
        return j["key"] in load_traces(npz)
    except (OSError, ValueError):
        return False


def gate_open() -> bool:
    return subprocess.run([str(RESGUARD), "status"], capture_output=True, text=True).returncode == 0


def gpu_ok(min_free_mb=4500) -> tuple[bool, str]:
    """At least 2 GB free beside this job's ~2.5 GB, and no other Python process on the card."""
    try:
        free = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True).stdout.split()[0])
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip().splitlines()
    except (OSError, IndexError, ValueError) as e:
        return False, f"nvidia-smi failed: {e}"
    others = [a for a in apps if "python" in a or "/uv" in a]
    return free >= min_free_mb and not others, f"free {free} MB, other jobs {others}"


def log(msg):
    print(time.strftime("%F %T"), msg, flush=True)


def append_row(row):
    LOGDIR.mkdir(parents=True, exist_ok=True)
    with open(LOGDIR / "rows.jsonl", "a") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def run_queue(a):
    todo = [j for j in jobs() if not is_done(j)]
    log(f"{len(todo)} evaluations to run")
    for i, j in enumerate(todo):
        while True:
            ok, why = gpu_ok()
            if gate_open() and ok:
                break
            log(f"waiting: gate {'open' if gate_open() else 'closed'}, GPU {why}")
            time.sleep(60)
        nj = new_json(j)
        cmd = [str(RESGUARD), "run", "--mem", "3100M", "--cpu", "200", "--"] + UV + [
            "--policy", j["policy"], "--morphology-run", j["morph"]] + COMMON + COND[j["key"]] + [
            "--json-out", str(nj), "--traces", str(nj.parent / f"{j['run']}.npz"), "--trace-key", j["key"],
            "--label", j["label"]]
        cache = tempfile.mkdtemp(prefix="warpcache-reeval-")
        env = dict(os.environ, WARP_CACHE_PATH=cache, MUJOCO_GL="egl")
        env.pop("PYTHONPATH", None)
        t0 = time.time()
        logf = LOGDIR / f"{j['label']}.log"
        LOGDIR.mkdir(parents=True, exist_ok=True)
        with open(logf, "w") as fh:
            fh.write(" ".join(cmd) + "\n")
            fh.flush()
            rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=ROOT).returncode
        shutil.rmtree(cache, ignore_errors=True)
        if rc == 3:
            log(f"[{j['label']}] resguard refused the launch; retry in 120 s")
            time.sleep(120)
            return run_queue(a)
        chk = subprocess.run([sys.executable, "scripts/policy_eval_suite.py", "--check",
                              f"{nj.parent / (j['run'] + '.npz')}:{j['key']}", str(nj)],
                             capture_output=True, text=True, cwd=ROOT) if nj.exists() else None
        row = dict(label=j["label"], folder=os.path.relpath(j["folder"], ROOT), key=j["key"], rc=rc,
                   seconds=round(time.time() - t0, 1), check=(chk.stdout.strip() if chk else "no JSON"),
                   finished=time.strftime("%F %T"))
        append_row(row)
        log(f"[{i + 1}/{len(todo)}] {j['label']} rc={rc} {row['seconds']} s; {row['check'].splitlines()[0] if chk else 'no JSON'}")
    log("queue done")


# ----------------------------------------------------------------------------------------------
# comparison with the original evaluations
# ----------------------------------------------------------------------------------------------

MAX_KIND = ("residual_absmax", "ctrl_gap_max_deg", "raw_cmd_beyond_range_deg")
MIN_KIND = ("clearance_min_mm_worst",)


def per_rollout(tr) -> dict:
    """Per-rollout values of every summary over rollouts: for a mean the spread is their sd; for an extreme over the
    64 rollouts (MAX_KIND, MIN_KIND) it is the distance from the extreme to the 90th (10th) percentile."""
    m = tr["meta"]
    cos, z, grip, pads = tr["cos"], tr["z"], tr["grip"], tr["pads"]
    held = (grip > m["held_min_n"]) & (z > m["floor_z"])
    aligned = cos >= m["align_thresh"]
    good = aligned & held
    T, N = cos.shape
    af = max(0, int(m["residual_from_step"]))
    t_align = np.where(aligned.any(0), aligned.argmax(0), np.nan).astype(float)
    hold_steps = np.zeros(N)
    drop = np.full(N, np.nan)
    for e in range(N):
        best = cur = 0
        for v in good[:, e]:
            cur = cur + 1 if v else 0
            best = max(best, cur)
        hold_steps[e] = best
        if held[:, e].any() and not held[-1, e]:
            drop[e] = np.nonzero(held[:, e])[0].max() + 1
    with np.errstate(invalid="ignore"):
        fmean = [np.where(good.any(0), (pads[..., k] * good).sum(0) / np.maximum(good.sum(0), 1), np.nan) for k in range(3)]
        three = np.where(good.any(0), ((pads.min(-1) > m["held_min_n"]) & good).sum(0) / np.maximum(good.sum(0), 1), np.nan)
    clear_min = tr["clearance"][af:].min(0) * 1000.0
    out = dict(final_cos_mean=cos[-1], final_z_mean=z[-1], peak_cos_mean=cos.max(0), t_align_mean=t_align,
               hold_steps_mean=hold_steps, drop_step_mean=drop, mean_force_thumb=fmean[0], mean_force_index=fmean[1],
               mean_force_middle=fmean[2], three_finger_share=three, clearance_min_mm_mean=clear_min,
               clearance_min_mm_worst=clear_min, clearance_end_mm_mean=tr["clearance"][-1] * 1000.0,
               residual_gt1_frac=tr["act_gt1"][af:].mean(0), residual_absmax=tr["act_absmax"][af:].max(0),
               ctrl_gap_max_deg=np.degrees(tr["gap_max"][af:].max(0)),
               raw_cmd_beyond_range_deg=np.degrees(tr["raw_exc"][af:].max(0)),
               joint_speed_p99_deg_s=np.degrees(np.percentile(tr["jvel_max"][af:], 99, axis=0)))
    for k, f in enumerate(("thumb", "index", "middle")):
        out[f"pad_peak_{f}"] = pads[af:, :, k].max(0)
        out[f"force_active_{f}"] = pads[af:, :, k].mean(0)
    return out


SD_OF = {"final_cos_mean": "final_cos_sd", "hold_steps_mean": "hold_steps_sd", "clearance_min_mm_mean": "clearance_min_mm_sd",
         "drop_step_mean": "drop_step_sd", "t_align_mean": "t_align_sd", "final_z_mean": "final_z_sd", "peak_cos_mean": "peak_cos_sd"}
RATES = ("hold_rate", "align_rate", "success_rate")


def compare_one(j) -> dict | None:
    nj = new_json(j)
    if not nj.exists() or not Path(j["orig"]).exists():
        return None
    old = json.load(open(j["orig"]))
    if j["sub"]:
        old = old[j["sub"]]
    new = json.load(open(nj))
    tr = load_traces(Path(j["folder"]) / "traces" / f"{j['run']}.npz")[j["key"]]
    pr = per_rollout(tr)
    n = new["n"]
    fields = []
    for k, ov in old.items():
        if k not in new or isinstance(ov, (str, bool)) or k in ("n", "steps", "align_thresh", "residual_from_step",
                                                                "spawn_jitter_mm", "spawn_yaw_deg") or k.endswith("_sd"):
            continue
        nv = new[k]
        if ov is None or nv is None:
            fields.append(dict(field=k, old=ov, new=nv, spread=None, flagged=(ov is None) != (nv is None)))
            continue
        if k in RATES or k == "n_lost":                     # counts of rollouts: binomial sd, at least one rollout
            per = n if k in RATES else 1                        # rollouts per unit of the field
            p = (ov + nv) / 2 * per / n
            sd = max(1.0, float(np.sqrt(n * p * (1 - p))))
            fields.append(dict(field=k, old=ov, new=nv, spread=sd / per, flagged=bool(abs(nv - ov) * per > sd + 1e-9)))
            continue
        vals = pr.get(k)
        sd = float(np.nanstd(vals)) if vals is not None and np.isfinite(vals).any() else 0.0
        if vals is not None and k in MAX_KIND + MIN_KIND and np.isfinite(vals).any():
            v = vals[np.isfinite(vals)]
            sd = max(sd, float(v.max() - np.percentile(v, 90)) if k in MAX_KIND else float(np.percentile(v, 10) - v.min()))
        for s in (old.get(SD_OF.get(k, "")), new.get(SD_OF.get(k, ""))):
            if isinstance(s, (int, float)):
                sd = max(sd, float(s))
        fields.append(dict(field=k, old=ov, new=nv, spread=sd, flagged=bool(abs(nv - ov) > sd + 1e-12)))
    return dict(label=j["label"], run=j["run"], key=j["key"], original=os.path.relpath(j["orig"], ROOT) + (f"[{j['sub']}]" if j["sub"] else ""),
                reevaluated=os.path.relpath(nj, ROOT), evaluated=new.get("evaluated"), fields=fields,
                n_flagged=sum(f["flagged"] for f in fields))


def compare(a):
    by_folder: dict = {}
    for j in jobs():
        c = compare_one(j)
        if c:
            by_folder.setdefault(str(j["folder"]), []).append(c)
    for folder, rows in by_folder.items():
        out = Path(folder) / "traces" / "reeval_compare.json"
        out.write_text(json.dumps(rows, indent=1))
        print(f"== {os.path.relpath(folder, ROOT)}: {len(rows)} evaluations, {sum(r['n_flagged'] > 0 for r in rows)} with a "
              f"difference beyond the rollout spread -> {os.path.relpath(out, ROOT)}")
        for r in rows:
            for f in r["fields"]:
                if f["flagged"]:
                    sp = "" if f["spread"] is None else f" (spread {f['spread']:.4g})"
                    print(f"   {r['label']}: {f['field']} {f['old']} -> {f['new']}{sp}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("run", "list", "compare"))
    a = ap.parse_args()
    if a.cmd == "list":
        js = jobs()
        for j in js:
            print(f"{'done' if is_done(j) else 'todo'}  {os.path.relpath(j['folder'], ROOT):38s} {j['label']}")
        print(f"{len(js)} evaluations, {sum(is_done(j) for j in js)} done")
    elif a.cmd == "run":
        run_queue(a)
    else:
        compare(a)


if __name__ == "__main__":
    main()
