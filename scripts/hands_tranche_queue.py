#!/usr/bin/env python3
"""Serial, gate-aware, resumable queue of reorient finetunes across the deployed hands.

    nohup setsid python3 scripts/hands_tranche_queue.py --queue docs/experiments/20260919-hands_tranche/queue.json \
        > logs/20260919-hands_tranche.log 2>&1 &

Each job: wait for `resguard.sh status` to open the launch gate -> train through `resguard.sh run`
(rc 3 = gate closed at launch: wait and retry) -> evaluate 64 rollouts with the run's own timing
(policy_eval_suite.py) -> render one 960x720 rollout + trace (rl_render_rollout.py) -> filmstrip
(policy_filmstrip.py) -> append a row to <queue dir>/tranche_results.json (fsynced). The queue file
is re-read before every job, so a job's flags or status can be edited while the queue runs; jobs
with status "done", "failed" or "skip" are not touched. One heavy process at a time, always.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESGUARD = Path.home() / ".claude/bin/resguard.sh"
UV = ["uv", "run", "--extra", "rl", "--extra", "gpu", "python"]


def log(msg):
    print(time.strftime("%F %T"), msg, flush=True)


def gate_open() -> bool:
    return subprocess.run([str(RESGUARD), "status"], capture_output=True, text=True).returncode == 0


def wait_gate(tag):
    waited = 0
    while not gate_open():
        if waited % 600 == 0:
            st = subprocess.run([str(RESGUARD), "status"], capture_output=True, text=True).stdout.splitlines()
            log(f"[{tag}] gate closed, waiting: {st[0] if st else ''} / {' '.join(l.strip() for l in st[-2:])}")
        time.sleep(60)
        waited += 60


def run_guarded(cmd, mem, cpu, logfile, tag):
    """resguard run with retry while the gate is closed at launch (rc 3)."""
    while True:
        wait_gate(tag)
        full = [str(RESGUARD), "run", "--mem", mem, "--cpu", str(cpu), "--"] + cmd
        env = dict(os.environ, WARP_CACHE_PATH=subprocess.check_output(["mktemp", "-d"], text=True).strip(),
                   MUJOCO_GL="egl")
        env.pop("PYTHONPATH", None)
        with open(logfile, "a") as fh:
            fh.write(f"\n# {time.strftime('%F %T')} {' '.join(full)}\n"); fh.flush()
            rc = subprocess.run(full, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=ROOT).returncode
        if rc == 3:
            log(f"[{tag}] resguard refused (gate closed at launch); retry in 120 s")
            time.sleep(120)
            continue
        return rc


def latest_model(run_dir: Path) -> Path | None:
    ms = sorted(run_dir.glob("tensorboard/model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    return ms[-1] if ms else None


def res_by_id(results, rid):
    for r in results:
        if r.get("id") == rid:
            return r
    return None


def save_json(path: Path, obj):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1)
        fh.flush(); os.fsync(fh.fileno())
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", type=Path, required=True)
    a = ap.parse_args()
    qdir = a.queue.parent
    results_path = qdir / "tranche_results.json"
    results = json.load(open(results_path)) if results_path.exists() else []
    # a job left "running" by a driver that died (no result row) goes back to pending
    q0 = json.load(open(a.queue))
    for jj in q0["jobs"]:
        if jj.get("status") == "running" and res_by_id(results, jj["id"]) is None:
            jj["status"] = "pending"
            log(f"[{jj['id']}] was left running by a dead driver; back to pending")
    save_json(a.queue, q0)
    while True:
        q = json.load(open(a.queue))
        pending = [j for j in q["jobs"] if j.get("status", "pending") == "pending"]
        if not pending:
            # Final pass: zero-shot transfer probes (perturbed plants) for every finished job that
            # lacks one, one at a time through the same gate, then exit. A wake-up that appends jobs
            # relaunches the driver; it re-reads the queue and skips finished probes.
            variants = q.get("transfer_variants", "base,tipmesh,plate0,mu0.6,mu1.5,mass1.3,kp0.25")
            for j in q["jobs"]:
                r = res_by_id(results, j["id"])
                if j.get("status") != "done" or not r or not r.get("model"):
                    continue
                tj = qdir / f"{j['id']}_transfer.json"
                have = json.load(open(tj)) if tj.exists() else {}
                if all(v in have and have[v].get("hold_rate") is not None for v in variants.split(",")):
                    continue
                log(f"[{j['id']}] TRANSFER PROBE {variants}")
                cmd = UV + ["scripts/policy_transfer_probe.py", "--policy", r["model"], "--morphology-run", j["morph_run"],
                            "--variants", variants, "--n", "64", "--out", str(tj)]
                rc = run_guarded(cmd, "8G", 400, ROOT / "logs" / f"{qdir.name}-{j['id']}_transfer.log", j["id"])
                log(f"[{j['id']}] transfer probe rc={rc}")
            log("queue empty and probes done; exiting")
            return 0
        j = pending[0]
        tag = j["id"]

        def set_status(s, **kw):
            q2 = json.load(open(a.queue))
            for jj in q2["jobs"]:
                if jj["id"] == tag:
                    jj["status"] = s; jj.update(kw)
            save_json(a.queue, q2)

        set_status("running", started=time.strftime("%F %T"))
        morph = j["morph_run"]
        common = q["common_flags"] + j.get("flags", [])
        train = UV + ["scripts/rl_train_cube.py", "--morphology-run", morph, "--tag", tag] + common
        log(f"[{tag}] TRAIN {' '.join(train[-len(common) - 4:])}")
        t0 = time.time()
        rc = run_guarded(train, q.get("train_mem", "10G"), q.get("train_cpu", 800),
                         ROOT / "logs" / f"{qdir.name}-{tag}.log", tag)
        runs = sorted(glob.glob(str(ROOT / "results/rl" / f"*-{tag}")))
        run_dir = Path(runs[-1]) if runs else None
        model = latest_model(run_dir) if run_dir else None
        if rc != 0 or model is None:
            log(f"[{tag}] TRAIN FAILED rc={rc} run={run_dir} model={model}")
            set_status("failed", rc=rc, run=str(run_dir))
            results.append({"id": tag, "hand": j["hand"], "arm": j["arm"], "status": "failed", "rc": rc,
                            "run": str(run_dir), "train_min": round((time.time() - t0) / 60, 1)})
            save_json(results_path, results)
            continue
        train_min = round((time.time() - t0) / 60, 1)
        log(f"[{tag}] trained in {train_min} min -> {model}")

        ev_json = qdir / f"{tag}_eval.json"
        ev = UV + ["scripts/policy_eval_suite.py", "--policy", str(model), "--morphology-run", morph,
                   "--closed-ctrl-from-keyframe", "open_ik", "--open-finger-from-keyframe", "--lift-delta", "0.1",
                   "--steps", "250", "--n", "64", "--held-min-n", "0.5", "--floor-z", "0.06",
                   "--json-out", str(ev_json), "--plot", str(qdir / f"{tag}_eval.png"), "--label", tag]
        rc_e = run_guarded(ev, "8G", 400, ROOT / "logs" / f"{qdir.name}-{tag}_eval.log", tag)
        video = qdir / "videos" / f"{tag}.mp4"
        rd = UV + ["scripts/rl_render_rollout.py", "--policy", str(model), "--morphology-run", morph,
                   "--closed-ctrl-from-keyframe", "open_ik", "--open-finger-from-keyframe", "--lift-delta", "0.1",
                   "--steps", "250", "--out", str(video)]
        rc_r = run_guarded(rd, "4G", 200, ROOT / "logs" / f"{qdir.name}-{tag}_render.log", tag)
        strip = qdir / f"{tag}_strip.png"
        if video.exists():
            subprocess.run(["uv", "run", "--extra", "rl", "python", "scripts/policy_filmstrip.py", "--video", str(video),
                            "--frames", "8", "--cols", "4", "--at", "40,58,80,110,140,170,200,249", "--fps", "25",
                            "--out", str(strip)], cwd=ROOT, env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"} | {"MUJOCO_GL": "egl"},
                           capture_output=True)
            try:
                from PIL import Image
                im = Image.open(strip); w, h = im.size
                im.resize((2560, int(h * 2560 / w)), Image.LANCZOS).save(strip, optimize=True)
            except Exception as e:
                log(f"[{tag}] strip resize failed: {e}")
            trace = video.with_suffix(".csv")
            if trace.exists():
                (qdir / f"{tag}_trace.csv").write_text(trace.read_text())
        # CPU transfer probe: 7 scene variants x 6 jittered rollouts at ~3 s each on one core. It does
        # not go through resguard run (it is not heavy) but it does wait for 8 GB of MemAvailable.
        try:
            while float(subprocess.run(["awk", "/MemAvailable/{print $2/1048576}", "/proc/meminfo"],
                                       capture_output=True, text=True).stdout or 0) < 8.0:
                time.sleep(60)
            tj = qdir / f"{tag}_transfer_cpu.json"
            variants = q.get("transfer_variants", "base,tipmesh,plate0,mu0.6,mu1.5,mass1.3,kp0.25")
            subprocess.run(["uv", "run", "--extra", "rl", "python", "scripts/policy_transfer_probe.py", "--policy", str(model),
                            "--morphology-run", morph, "--variants", variants, "--cpu", "--n", "6", "--out", str(tj)],
                           cwd=ROOT, env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"} | {"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"},
                           capture_output=True, timeout=3600)
            log(f"[{tag}] cpu transfer probe -> {tj.name}")
            # the policy inside the UR5e chain (CPU, ~5 s each): plate at the built 25 mm and at the chain
            # scenes' 0, policy in the loop for 5 s with no frozen hold
            sq = json.load(open(Path(morph) / "summary.json")).get("squeeze_mm", 10.0)
            for pl in (25, 0):
                cj = qdir / f"{tag}_chain_pl{pl}.json"
                subprocess.run(["uv", "run", "--extra", "rl", "python", "scripts/real_v1_chain_policy.py", "--hand", j["hand"],
                                "--policy", str(model), "--morphology-run", morph, "--squeeze", str(sq), "--plant", "cal",
                                "--plate-mm", str(pl), "--turn-steps", "2500", "--hold-steps", "0", "--out", str(cj)],
                               cwd=ROOT, env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"} | {"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"},
                               capture_output=True, timeout=1800)
            log(f"[{tag}] chain probes done")
        except Exception as e:
            log(f"[{tag}] cpu transfer/chain probe failed: {e}")
        row = {"id": tag, "hand": j["hand"], "arm": j["arm"], "status": "done", "run": str(run_dir),
               "model": str(model), "train_min": train_min, "eval_rc": rc_e, "render_rc": rc_r,
               "finished": time.strftime("%F %T")}
        if ev_json.exists():
            e = json.load(open(ev_json))
            row.update({k: e.get(k) for k in ("hold_rate", "align_rate", "final_cos_mean", "final_cos_sd", "t_align_mean",
                                              "peak_cos_mean", "n_lost", "three_finger_share", "clearance_min_mm_mean",
                                              "clearance_min_mm_worst", "clearance_end_mm_mean", "pad_peak_thumb",
                                              "pad_peak_index", "pad_peak_middle", "force_active_thumb",
                                              "force_active_index", "force_active_middle", "joint_speed_p99_deg_s",
                                              "residual_gt1_frac", "residual_absmax", "ctrl_gap_max_deg",
                                              "raw_cmd_beyond_range_deg")})
        results = [r for r in results if r["id"] != tag] + [row]
        save_json(results_path, results)
        set_status("done", run=str(run_dir), finished=row["finished"])
        log(f"[{tag}] DONE hold {row.get('hold_rate')} cos {row.get('final_cos_mean')} clearance_min {row.get('clearance_min_mm_mean')} mm")


if __name__ == "__main__":
    sys.exit(main())
