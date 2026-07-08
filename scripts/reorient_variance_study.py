"""Reorient-quality VARIANCE study — can we cut the seed noise enough to rank morphologies?

The large16/confirm sweeps showed per-design reorient held-cos has sd ~0.4 (spans negative to 0.8),
dominated by from-scratch training-convergence luck, so no design separates from m05. This tests two
variance-reduction levers (user's #1 + #2), CONTROLLED:

  - fix Policy A per design (reuse a known-good A) -> removes A's contribution to variance;
  - B warm-start mode:  self  = warm-start B from the design's own (holder) A  [current pipeline]
                        shared = warm-start B from the ONE proven reorienter b33 [the lever] which
                                 already knows the rolling motion, so B adapts rather than re-discovers;
  - N re-runs per (design, mode) -> the band (no --seed exists; runs differ by warp/GPU nondeterminism).

Two designs (m05 anchor, L01_13 apparent-lead). For each (design x mode) train B N times off the fixed
A via the live-A reset (open-finger-from-keyframe), eval the continuous handoff, record held-cos /
force / jerk. Question: does `shared` collapse the band, and do the two designs then separate?

Resumable (per-run JSON checkpoint), DONE sentinel. ~36 min per B-run.
Run: MUJOCO_GL=egl uv run --extra rl --extra gpu python scripts/reorient_variance_study.py [--n 3]
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time, statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
B33 = ROOT / "results/rl/b33_20260702-1353-policyB_m05_reorient_ik10/tensorboard/model_270.pt"
# design-NEUTRAL imitation reference: the object-relative fingertip trajectory of the blessed
# a10->b33 reorient (recorded once), imitated on ANY design -> a fair shared reorient prior.
IMIT_REF = ROOT / "results/reorient_ref/m05_a10b33_fingertip_obj.npz"
JSON = ROOT / "docs/experiments/REORIENT_VARIANCE.json"
TXT = ROOT / "docs/experiments/REORIENT_VARIANCE.txt"
SENTINEL = ROOT / "logs/REORIENT_VARIANCE.DONE"

# design -> (fixed A ckpt, morphology-run [rel to ROOT])
DESIGNS = {
    "m05": (ROOT / "results/rl/a10_20260702-1256-policyA_m05_ik10/tensorboard/model_609.pt",
            "results/phase1/landscape/m05_ik_cem"),
    "L01_13": (ROOT / "results/rl/20260704-1843-policyA_L01_13/tensorboard/model_609.pt",
               "results/phase1/morph_sweep/L01_13_cem"),
}


def _tmp():
    return subprocess.run(["mktemp", "-d"], capture_output=True, text=True).stdout.strip()


def train_B(design, a_ck: Path, cem_rel: str, mode: str, k: int, env):
    tag = f"vstudy_{design}_{mode}_k{k}"
    log = ROOT / f"logs/vstudy_{design}_{mode}_k{k}.trainer.log"
    # shared = warmstart b33; self/imit = warmstart the design's own (holder) A.
    b_ck = str(B33 if mode == "shared" else a_ck)
    extra = "--open-finger-from-keyframe"
    if mode == "imit":   # design-neutral reorient prior via object-relative fingertip imitation
        extra += (f" --imitation-ref-npz {IMIT_REF} --imitation-weight 60 --imitation-alpha 300"
                  f" --imitation-curriculum-iters 150 --imitation-weight-final 0")
    e = dict(env)
    e.update(MORPH=cem_rel, A_CKPT=str(a_ck), B_CKPT=b_ck, LIFT_DELTA="0.10",
             EXTRA_ARGS=extra,
             ONSET_STEP="40", BLEND="8", LIFT_TERM_START="58", REORIENT_START="58",
             TAG=tag, LOG=str(log), NUM_ENVS="3072", TOTAL_TS="20000000")
    subprocess.run(["bash", str(ROOT / "scripts/train_handoff_liveA_reset.sh")],
                   check=False, capture_output=True, text=True, env=e, timeout=7200)
    runs = sorted((ROOT / "results/rl").glob(f"*{tag}"), key=lambda p: p.stat().st_mtime)
    if not runs:
        return None, Path(str(log) + ".COLLAPSED").exists()
    cks = sorted((runs[-1] / "tensorboard").glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    return (cks[-1] if cks else None), Path(str(log) + ".COLLAPSED").exists()


def eval_handoff(design, a_ck: Path, b_ck: Path, cem_rel: str, tag: str, env):
    out = ROOT / f"docs/rl/videos/reorient/sweep/{tag}.mp4"
    e = dict(env); e["WARP_CACHE_PATH"] = _tmp()
    subprocess.run([sys.executable, str(ROOT / "scripts/rl_demo_handoff_continuous.py"),
                    "--policy-a", str(a_ck), "--policy-b", str(b_ck),
                    "--morphology-run", str(ROOT / cem_rel), "--lift-delta", "0.10",
                    "--open-finger-from-keyframe", "--output", str(out)],
                   check=False, capture_output=True, text=True, env=e, timeout=1800)
    hj = out.with_suffix(".health.json")
    return json.loads(hj.read_text()) if hj.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3, help="re-runs per (design, mode)")
    ap.add_argument("--modes", default="self,shared")
    args = ap.parse_args()
    env = dict(os.environ); env.setdefault("MUJOCO_GL", "egl")
    if SENTINEL.exists():
        SENTINEL.unlink()
    done = {d["id"]: d for d in json.loads(JSON.read_text())} if JSON.exists() else {}
    if not TXT.exists():
        TXT.write_text(f"# reorient variance study  {time.strftime('%Y-%m-%d %H:%M')}  n={args.n}/cell\n"
                       f"# fixed A per design; B warm-start: self=own A, shared=b33 (proven reorienter)\n")
    jobs = [(d, m, k) for d in DESIGNS for m in args.modes.split(",") for k in range(args.n)]
    for design, mode, k in jobs:
        rid = f"{design}_{mode}_k{k}"
        if rid in done and "cos" in done[rid]:
            print(f"[skip] {rid}"); continue
        a_ck, cem_rel = DESIGNS[design]
        t0 = time.time(); rec = {"id": rid, "design": design, "mode": mode, "k": k}
        try:
            print(f"[{time.strftime('%H:%M:%S')}] {rid}: train B ({mode} warmstart) ...", flush=True)
            b_ck, aborted = train_B(design, a_ck, cem_rel, mode, k, env)
            rec["b_aborted"] = aborted
            if b_ck is None:
                rec["note"] = "no B checkpoint"
            else:
                print(f"[{time.strftime('%H:%M:%S')}] {rid}: eval handoff ...", flush=True)
                h = eval_handoff(design, a_ck, b_ck, cem_rel, f"vstudy_{rid}", env) or {}
                m = h.get("metrics") or {}
                rec.update(cos=m.get("held_cos_tail"), peak=m.get("peak_cos"),
                           force=m.get("tip_force"), jerk=m.get("ang_jerk"),
                           minz=m.get("min_z_hold"), verdict=h.get("verdict"))
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {str(e)[:150]}"
        rec["secs"] = round(time.time() - t0)
        done[rid] = rec
        JSON.write_text(json.dumps(list(done.values()), indent=1))
        line = (f"{rid:22} cos {str(rec.get('cos','—')):>6} peak {str(rec.get('peak','—')):>6} "
                f"force {str(rec.get('force','—')):>5} jerk {str(rec.get('jerk','—')):>5} "
                f"{rec.get('verdict', rec.get('note', rec.get('error','')))}  ({rec['secs']}s)")
        print(line, flush=True)
        with TXT.open("a") as f:
            f.write(line + "\n")
    # summary: per (design, mode) band
    print("\n=== BANDS (held-cos mean ± sd) ===", flush=True)
    with TXT.open("a") as f:
        f.write("\n# --- bands (held-cos mean ± sd, range, n) ---\n")
        for design in DESIGNS:
            for mode in args.modes.split(","):
                xs = [r["cos"] for r in done.values()
                      if r.get("design") == design and r.get("mode") == mode and r.get("cos") is not None]
                if xs:
                    s = (f"{design:8} {mode:7}: cos {st.mean(xs):+.2f} ± "
                         f"{(st.pstdev(xs) if len(xs)>1 else 0):.2f}  [{min(xs):+.2f},{max(xs):+.2f}]  n={len(xs)}")
                    print(s, flush=True); f.write(s + "\n")
    SENTINEL.write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
    print(f"[variance-study] COMPLETE -> {TXT}")


if __name__ == "__main__":
    main()
