"""Full per-morphology A->B pipeline sweep — the HONEST, health-gated design search (2026-07-03).

Supersedes the 2026-06-25 landscape sweep (morph_landscape_sweep.py), which scored designs on a
skip-lift TELEPORT reorienter warmstarting B4 (no native Policy A) and a JOINT-space keyframe that
under-scored graspability. This sweep runs, per morphology, the exact CLEAN m05-fixed pipeline
(the policy in docs/rl/videos/20260702_reorient/1431_handoff_m05_FIXED.mp4, a10 -> b33):

  1. generate scene XML from the 9-param design vector      (generate_morphology_xml.py)
  2. IK-retarget the fingertip open keyframe -> `open_ik`    (retarget_keyframe_ik helpers)
  3. CEM grasp from open_ik -> graspability + persistence    (phase1_optimize_grasp.py)
       GRASPABILITY GATE: cube_lift < --lift-thresh => ungraspable, skip A/B (cheap screen).
  4. native Policy A (lift+deliver, from scratch, open-finger-from-keyframe, deliver@0.10)
       (scripts/train_A_on_morph.sh) + its baked-in trajectory-health acceptance gate.
  5. Policy B reorient via LIVE-A RESET, warmstart the hold-first A (single phase)
       (scripts/train_handoff_liveA_reset.sh).
  6. continuous A->B handoff eval -> video + trajectory-health scorecard .health.json
       (scripts/rl_demo_handoff_continuous.py): post-handoff min-z, held-cos, jitter, per-finger
       force, de-centering, PASS/WARN/FAIL verdict — the "in depth analysis" readout.

Robust for an unattended machine (the design goal): per-morphology checkpoint to JSON (RESUMABLE
— re-running skips finished designs), streamed human-readable rows to TXT, try/except per design
(one failure never sinks the sweep), generous subprocess timeouts, a DONE sentinel on completion.
Sequential (one GPU) — A ~55 min + B ~36 min + CEM/eval ~5 min ~= 100 min/graspable design.

POLICY-BOTTLENECK PROBES (2026-07-10, docs/rl/morph_sweep_STATUS.md §probes): the confirm/variance
studies showed the evaluator noise is dominated by Policy A's from-scratch training draw (joint
sd 0.41 vs B-only 0.09 self / 0.02 imit), and every large16 failure had an A-side event. Hence:
  --a-attempts N   A best-of-N: retry on collapse / health-FAIL / never-lift; EVERY attempt is
                   recorded in the JSON (raw draw data for the A-variance analysis).
  --b-recipe       plain = b_liveA (self warmstart, legacy)  |  imit = b_liveA_imit (the
                   design-neutral object-frame fingertip prior, the variance-solved evaluator)
                   |  both = imit AND plain on the SAME kept A (paired prior-fairness test).
  --morph-set rescue  the 5 large16 failures (flip rate = is the bottleneck the policy?)
  --morph-set avar    raw A-draw distribution cells (run with --a-attempts 1)
  --morph-set global  Latin-hypercube over the FULL 9-param box (the >16-design landscape)
  --only id1,id2      run a subset of the set (smoke / surgical re-runs)

Run (detached):
  nohup setsid env MUJOCO_GL=egl uv run --extra rl --extra gpu \
    python scripts/morph_pipeline_sweep.py --morph-set initial8 > sweep_initial8.run.log 2>&1 &
Smoke (validate the glue, ~10 min, tiny ts):  add --smoke
"""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from pathlib import Path
import numpy as np

from morphohand.studies import runlib
from morphohand.studies.runlib import ROOT, best_a_ckpt, final_ckpt, iter_objheight, latest_run
from morphohand.tools.keyframe_ik import retarget_scene

BASE_HAND = ROOT / "assets/mjcf/baseline/hand.xml"
BASE_SCENE = ROOT / "assets/mjcf/baseline/scenes/scene_screwdriver_medium_flat_short_proximal.xml"
GEN = ROOT / "assets/mjcf/experimental/morph_sweep"          # generated scenes (gitignored)
CEM_OUT = ROOT / "results/phase1/morph_sweep"                 # CEM grasp outputs
# handoff videos + .health.json, in the timestamped tree (morphohand.tools.video_paths)
from morphohand.tools.video_paths import experiment_dir  # noqa: E402
VID_OUT = experiment_dir("sweep")

# 9-param design vector = t(x,y,len) i(x,y,len) m(x,y,len). Bounds from morph-joint ranges.
BND = [(-0.025, 0.025), (-0.025, 0.025), (0.0, 0.030)] * 3
M05 = (0.0147, 0.005, 0.0108, 0.004, 0.0022, 0.0123, 0.0246, 0.0242, 0.0159)  # landscape winner
M00 = (0.0, 0.020, 0.0, 0.010, -0.0123, 0.0, 0.010, 0.0153, 0.0)              # baseline
# The proven m05 policies, for reference only. NOTE: warmstarting A from a10 was TRIED (initial8
# "fix") and REJECTED — a10's grip-specific residual ejects the re-CEM'd object (canary never
# lifted). From-scratch A + B-from-A + open-finger-from-keyframe is the correct, robust recipe.
A10 = ROOT / "results/rl/a10_20260702-1256-policyA_m05_ik10/tensorboard/model_609.pt"
B33 = ROOT / "results/rl/b33_20260702-1353-policyB_m05_reorient_ik10/tensorboard/model_270.pt"


def _clip(v):
    return tuple(round(float(np.clip(v[j], *BND[j])), 4) for j in range(9))


def _perturb(center, deltas):
    v = list(center)
    for idx, d in deltas.items():
        v[idx] += d
    return _clip(v)


# The five large16 designs that "never learned" / FAILed, verbatim vectors from
# MORPH_PIPELINE_large16.json. Every one of them had an A-side event (collapse or
# health-FAIL delivery) — the rescue probe asks whether a better OPTIMIZER draw
# (A best-of-2 + imitation-B) flips them, i.e. whether the verdicts were policy noise.
RESCUE = {
    "rs_L01_02": (0.0162, 0.0051, 0.013, 0.0003, 0.0014, 0.0104, 0.025, 0.0244, 0.0147),    # idle-finger FAIL (1.2/2.5/2.2 N)
    "rs_L01_03": (0.0108, 0.0037, 0.0108, 0.0026, 0.0087, 0.0163, 0.011, 0.0148, 0.0152),   # A collapsed at iter 0 — "never lifted"
    "rs_L01_05": (0.0121, -0.0032, 0.0115, 0.0045, -0.0039, 0.0096, 0.0242, 0.0195, 0.0155),  # A late-collapse salvage -> cos -0.45 FAIL
    "rs_L01_07": (0.0191, -0.0004, 0.0145, 0.0039, -0.004, 0.011, 0.0249, 0.025, 0.012),    # A health-FAIL -> cos 0.35
    "rs_L01_09": (0.0106, 0.0088, 0.0118, 0.0085, 0.0005, 0.0064, 0.024, 0.022, 0.019),     # A health-FAIL -> wrong-way cos -0.40
}


def morph_set(name: str, n: int, seed: int, center, replicas: int = 1, freeze_len: bool = False):
    """Return [(id, 9-vector), ...]. `initial8` = interpretable coordinate moves around m05.
    `replicas` (global set only) emits each design as independent `_rj` full-pipeline draws.
    `freeze_len` (global set only): explore ONLY the 6 XY placement dims per finger, holding the
    three proximal-phalange lengths FROZEN at m05 (ids prefixed `H` not `G`)."""
    if name == "initial8":
        # s00 m05 anchor (reproduce the known winner under the honest pipeline);
        # s01 baseline m00 (a full A->B on the base design — a calibration reference);
        # s02..s07 six single/double-axis hypotheses mapped to the plan's Stage 1(a/b/c):
        #   opposition (recruit the thumb) + seating (object rides toward the palm).
        return [
            ("s00_m05anchor",   M05),
            ("s01_baseline",    M00),
            ("s02_thumbreach",  _perturb(M05, {0: +0.006})),               # thumb_x  -> reach across
            ("s03_thumblong",   _perturb(M05, {2: +0.006})),               # thumb_len -> opposition/seat
            ("s04_seat_allen",  _perturb(M05, {2: +0.004, 5: +0.004, 8: +0.004})),  # all len+ -> seat higher
            ("s05_shortgrasp",  _perturb(M05, {5: -0.004, 8: -0.004})),    # index+mid len- -> ride higher
            ("s06_middlein",    _perturb(M05, {6: -0.006, 7: -0.006})),    # middle x/y in -> tighter tripod
            ("s07_thumb_opp",   _perturb(M05, {0: +0.005, 1: +0.008})),    # thumb x/y -> true opposition
        ]
    if name == "confirm":
        # multi-seed CONFIRMATION of the large16 top lead vs the anchor: re-run each vector 3× fresh
        # (run-to-run PPO/warp non-determinism samples the seed band) to separate design from luck.
        L01_13 = (0.0235, 0.0044, 0.008, 0.0047, 0.0012, 0.0157, 0.0248, 0.0243, 0.013)  # best lead
        out = []
        for k in range(3):
            out.append((f"cf_m05_s{k}", M05))
            out.append((f"cf_l13_s{k}", L01_13))
        return out
    if name == "rescue":
        return list(RESCUE.items())
    if name == "avar":
        # RAW A-draw distribution (run with --a-attempts 1 — retries would censor the draws):
        # k independent full-pipeline (CEM+A) draws per design, imit-B rides each viable A.
        # m05 = known-good control; L01_05 = a large16 FAIL (its rescue attempts pool in too).
        return ([(f"av_m05_k{k}", M05) for k in range(3)]
                + [(f"av_L01_05_k{k}", RESCUE["rs_L01_05"]) for k in range(2)])
    if name == "global":
        # Latin hypercube over the FULL 9-param box — the honest-pipeline GLOBAL landscape
        # (the 2026-06-25 global map used the superseded teleport proxy; this replaces it).
        # `replicas` re-emits every design as `_r0/_r1/...` = INDEPENDENT full-pipeline draws
        # (avar verdict: per-draw cos sd ≈ 0.3-0.5 and gate-invisible ⇒ n=1 draws are
        # uninterpretable; score designs on mean/max over replicas + collapse count).
        # Replica-major order: a full r0 pass over all designs first (complete n=1 map early),
        # then the r1 refinement pass.
        rng = np.random.default_rng(seed)
        # freeze_len => LHS over the 6 XY dims only; proximal lengths (idx 2,5,8) held at m05.
        xy_idx = [0, 1, 3, 4, 6, 7]
        dims = xy_idx if freeze_len else list(range(9))
        prefix = "H" if freeze_len else "G"          # H = XY-only (frozen proximal len)
        D = len(dims)
        strata = (rng.permuted(np.tile(np.arange(n), (D, 1)), axis=1).T + rng.random((n, D))) / n
        lo = np.array([BND[j][0] for j in dims]); hi = np.array([BND[j][1] for j in dims])
        designs = []
        for k in range(n):
            v = list(M05) if freeze_len else [0.0] * 9   # frozen dims keep m05's len
            for col, j in enumerate(dims):
                v[j] = lo[col] + strata[k, col] * (hi[col] - lo[col])
            designs.append((f"{prefix}{seed:02d}_{k:02d}", _clip(tuple(v))))
        if replicas <= 1:
            return designs
        return [(f"{mid}_r{j}", v) for j in range(replicas) for mid, v in designs]
    # local quasi-random (Gaussian) search around `center`, seeded + reproducible.
    rng = np.random.default_rng(seed)
    sig = np.array([0.005, 0.005, 0.004] * 3)                              # per-axis step
    c = np.array(center)
    items = [(f"L{seed:02d}_00_center", _clip(tuple(c)))]
    for k in range(1, n):
        items.append((f"L{seed:02d}_{k:02d}", _clip(tuple(c + rng.normal(0, sig)))))
    return items


def gen_scene(m, env) -> Path:
    GEN.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/generate_morphology_xml.py"),
                    "--base-hand-xml", str(BASE_HAND), "--base-scene-xml", str(BASE_SCENE),
                    "--output-dir", str(GEN),
                    "--thumb", *map(str, m[0:3]), "--index", *map(str, m[3:6]),
                    "--middle", *map(str, m[6:9])],
                   check=True, capture_output=True, text=True, env=env, timeout=120)
    scenes = sorted(GEN.glob("scene_*.xml"), key=lambda p: p.stat().st_mtime)
    return scenes[-1]


def run_cem(scene: Path, tag: str, env, iters: int) -> dict:
    e = runlib.warp_cache_env(env)
    subprocess.run([sys.executable, str(ROOT / "scripts/phase1_optimize_grasp.py"),
                    "--scene-xml", str(scene), "--keyframe", "open_ik",
                    "--iterations", str(iters), "--population", "80", "--skip-gif",
                    "--objective-weight-min-finger-persistence", "4.0",
                    "--objective-weight-contact-persistence", "1.5",
                    "--output-dir", str(CEM_OUT), "--tag", tag],
                   check=True, capture_output=True, text=True, env=e, timeout=1800)
    return json.loads((CEM_OUT / tag / "summary.json").read_text())["best_metrics"]


VERDICT_RANK = {"PASS": 3, "WARN": 2, "FAIL": 1, None: 0}


def _train_A_once(cem_dir: Path, mid: str, t: int, env, smoke: bool) -> dict:
    # ALWAYS from scratch: warmstarting A (from a01 OR a10) loads a grip-specific residual that
    # EJECTS the re-CEM'd object (confirmed: a10-warmstart canary never lifted, obj 0.0 from iter
    # 0). From scratch the residual~0, so the open-loop CEM grip + scripted lift does the lifting.
    tag = f"policyA_{mid}_t{t}"
    log = ROOT / f"logs/sweep_A_{mid}_t{t}.trainer.log"
    e = dict(env)
    e.update(MORPH_RUN=str(cem_dir), WARMSTART="none", LIFT_DELTA_A="0.10",
             EXTRA_ARGS="--open-finger-from-keyframe --lift-phase-start-step 60",
             TAG=tag, LOG=str(log), NUM_ENVS="2048", SMOKE="1" if smoke else "0")
    if not smoke:
        e["TOTAL_TS"] = "30000000"
    Path(str(log) + ".COLLAPSED").unlink(missing_ok=True)   # stale sentinel = false abort
    subprocess.run(["bash", str(ROOT / "scripts/train_A_on_morph.sh")],
                   check=False, capture_output=True, text=True, env=e, timeout=9000)
    run = latest_run(f"*{tag}")
    aborted = Path(str(log) + ".COLLAPSED").exists() or run is None
    # Checkpoint choice: on CLEAN completion use the FINAL ckpt (fully trained -> best-BALANCED
    # grip; early ckpts lift marginally HIGHER but have an under-refined grip -> idle-finger, which
    # sank valfix2). Only on a watchdog ABORT (mid-training collapse) salvage the best pre-collapse
    # ckpt by object-height. `best_oh` is that ckpt's logged lift, for the a-lift-floor gate.
    oh_map = iter_objheight(log) if run else {}
    if run and aborted:
        ck, best_oh = best_a_ckpt(run, log)
    elif run:
        ck = final_ckpt(run)
        best_oh = oh_map.get(int(ck.stem.split("_")[1])) if ck and oh_map else None
    else:
        ck, best_oh = None, None
    health = None
    if ck:
        hj = (final_ckpt(run) or ck).with_suffix(".health.json")
        if hj.exists():
            try:
                health = json.loads(hj.read_text())
            except Exception:
                pass
    return {"run": run, "ck": ck, "aborted": aborted, "health": health, "oh": best_oh}


def train_A(cem_dir: Path, mid: str, env, smoke: bool, attempts: int = 1,
            lift_floor: float = 0.06):
    """From-scratch A, best-of-`attempts` draws. Retry on collapse / health-FAIL / never-lift;
    EVERY attempt is returned (raw A-draw data for the variance probes). Kept attempt = best of
    (non-aborted > health PASS>WARN>FAIL>unknown > objheight)."""
    tried = []
    for t in range(1 if smoke else max(1, attempts)):
        a = _train_A_once(cem_dir, mid, t, env, smoke)
        tried.append(a)
        verdict = (a["health"] or {}).get("verdict")
        lifted = smoke or a["oh"] is None or a["oh"] >= lift_floor
        if (not a["aborted"]) and a["ck"] is not None and lifted and verdict != "FAIL":
            break                                # good draw — no retry needed

    def rank(a):
        return (0 if a["aborted"] else 1,
                VERDICT_RANK.get((a["health"] or {}).get("verdict"), 0),
                a["oh"] if a["oh"] is not None else -1.0)

    return max(tried, key=rank), tried


def train_B(cem_rel: str, a_ck: Path, mid: str, env, smoke: bool,
            recipe: str = "b_liveA", tag_suffix: str = ""):
    # live-A DRIVER = this design's own A (lift matches the design) AND B WARMSTART = the same A
    # (the hold-first prior — the m05/b33 recipe). CRITICAL: --open-finger-from-keyframe (the
    # live-A script omits it → B resets to the baseline flung-out-thumb open pose → wrong grip →
    # the driver drops the object → B never learns; this sank s03/s04/s06/s07 in the initial8).
    # recipe: b_liveA = plain (self warmstart only) | b_liveA_imit = + the design-neutral
    # object-frame fingertip imitation prior (the variance-solved evaluator, sd ±0.02).
    tag = f"policyB_{mid}_reorient{tag_suffix}"
    log = ROOT / f"logs/sweep_B{tag_suffix}_{mid}.trainer.log"
    e = dict(env)
    e.update(MORPH=cem_rel, A_CKPT=str(a_ck), B_CKPT=str(a_ck), LIFT_DELTA="0.10",
             RECIPE=recipe, EXTRA_ARGS="--open-finger-from-keyframe",
             ONSET_STEP="40", BLEND="8", LIFT_TERM_START="58", REORIENT_START="58",
             TAG=tag, LOG=str(log), NUM_ENVS="3072", SMOKE="1" if smoke else "0")
    if not smoke:
        e["TOTAL_TS"] = "20000000"
    Path(str(log) + ".COLLAPSED").unlink(missing_ok=True)   # stale sentinel = false abort
    subprocess.run(["bash", str(ROOT / "scripts/train_handoff_liveA_reset.sh")],
                   check=False, capture_output=True, text=True, env=e, timeout=7200)
    run = latest_run(f"*{tag}")
    aborted = Path(str(log) + ".COLLAPSED").exists() or run is None
    return run, final_ckpt(run), aborted


B_RECIPES = {                       # --b-recipe -> [(recipe yaml, tag/video suffix, record key)]
    "plain": [("b_liveA", "", "handoff")],
    "imit": [("b_liveA_imit", "", "handoff")],
    "both": [("b_liveA_imit", "", "handoff"), ("b_liveA", "SELF", "handoff_self")],
}


def eval_handoff(a_ck: Path, b_ck: Path, cem_dir: Path, mid: str, env, suffix: str = ""):
    VID_OUT.mkdir(parents=True, exist_ok=True)
    out = VID_OUT / f"{mid}_handoff{suffix}.mp4"
    e = runlib.warp_cache_env(env)
    r = subprocess.run([sys.executable, str(ROOT / "scripts/rl_demo_handoff_continuous.py"),
                        "--policy-a", str(a_ck), "--policy-b", str(b_ck),
                        "--morphology-run", str(cem_dir), "--lift-delta", "0.10",
                        "--open-finger-from-keyframe", "--output", str(out)],
                       check=False, capture_output=True, text=True, env=e, timeout=1800)
    min_z_post = None
    for ln in (r.stdout or "").splitlines():
        if "honest hold metric" in ln:          # the specific post-handoff min-z summary line
            for t in ln.split(":")[-1].split():  # " 0.1246 m (HELD >0.05)" -> 0.1246
                try:
                    min_z_post = float(t); break
                except ValueError:
                    continue
            break
    health = None
    hj = out.with_suffix(".health.json")
    if hj.exists():
        try:
            health = json.loads(hj.read_text())
        except Exception:
            pass
    if min_z_post is None and health:           # fallback = scorecard's own hold-phase min-z
        min_z_post = (health.get("metrics") or {}).get("min_z_hold")
    return (str(out) if out.exists() else None), min_z_post, health


def _row(rec: dict) -> str:
    g = rec
    r = rec.get("handoff") or {}
    hm = (r.get("health") or {}).get("metrics", {}) if r else {}
    verdict = (r.get("health") or {}).get("verdict", "—") if r else rec.get("note", "—")
    row = (f"{rec['id']:16} | lift {g.get('lift', float('nan')):.3f} "
           f"tip {g.get('tips', '-')} pers {g.get('pt', 0):.2f}/{g.get('pi', 0):.2f}/{g.get('pm', 0):.2f}"
           f" | minZ {str(r.get('min_z_post', '—')):>6} cos {str(hm.get('held_cos_tail', '—')):>6}"
           f" jerk {str(hm.get('ang_jerk', '—')):>5} force {str(hm.get('tip_force', '—')):>5}"
           f" drift {str(hm.get('net_drift_cm', '—')):>4}cm | {verdict}")
    a = rec.get("A") or {}
    if len(a.get("attempts") or []) > 1:
        row += f" | A×{len(a['attempts'])}"
    s = rec.get("handoff_self") or {}
    if s:
        shm = (s.get("health") or {}).get("metrics", {})
        row += (f" | selfB cos {str(shm.get('held_cos_tail', '—')):>6} "
                f"({(s.get('health') or {}).get('verdict', '—')})")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--morph-set", default="initial8",
                    help="initial8 | local | confirm | rescue | avar | global")
    ap.add_argument("--n", type=int, default=16, help="designs for --morph-set local/global")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--center", default="m05", help="local-search center: m05 | best | 9 comma-floats")
    ap.add_argument("--lift-thresh", type=float, default=0.03, help="cube_lift below this => ungraspable")
    ap.add_argument("--cem-iters", type=int, default=200)
    ap.add_argument("--smoke", action="store_true", help="tiny ts, validate the glue end-to-end")
    ap.add_argument("--tag", default=None, help="output tag (default = morph-set)")
    ap.add_argument("--a-lift-floor", type=float, default=0.06,
                    help="A's best-ckpt object_height below this => A never lifted, skip B")
    ap.add_argument("--a-attempts", type=int, default=1,
                    help="A best-of-N: retry on collapse/health-FAIL/never-lift (all draws recorded)")
    ap.add_argument("--b-recipe", default="plain", choices=sorted(B_RECIPES),
                    help="plain=b_liveA | imit=b_liveA_imit | both=imit AND plain on the same A")
    ap.add_argument("--only", default=None,
                    help="comma-separated design ids: run only this subset of the morph-set")
    ap.add_argument("--replicas", type=int, default=1,
                    help="global set: independent full-pipeline draws per design (_r0/_r1/...)")
    ap.add_argument("--freeze-len", action="store_true",
                    help="global set: explore only the 6 XY placement dims, freezing the three "
                         "proximal-phalange lengths at m05 (design ids prefixed H not G)")
    args = ap.parse_args()
    env = runlib.base_env()

    tag = args.tag or args.morph_set
    sentinel = runlib.Sentinel(ROOT / f"logs/MORPH_PIPELINE_{tag}.DONE")
    sentinel.clear()

    center = M05
    if args.center == "best" and (ROOT / "docs/experiments/MORPH_PIPELINE_best_center.json").exists():
        center = tuple(json.loads((ROOT / "docs/experiments/MORPH_PIPELINE_best_center.json").read_text()))
    elif "," in args.center:
        center = tuple(float(x) for x in args.center.split(","))
    items = morph_set(args.morph_set, args.n, args.seed, center, replicas=args.replicas,
                      freeze_len=args.freeze_len)
    if args.only:
        keep = {s.strip() for s in args.only.split(",")}
        missing = keep - {i[0] for i in items}
        if missing:
            sys.exit(f"--only ids not in morph-set '{args.morph_set}': {sorted(missing)}")
        items = [i for i in items if i[0] in keep]

    store = runlib.RecordStore(ROOT / f"docs/experiments/MORPH_PIPELINE_{tag}.json", key_field="id")
    report = runlib.TxtReport(
        ROOT / f"docs/experiments/MORPH_PIPELINE_{tag}.txt",
        f"# full A->B pipeline sweep '{tag}'  {time.strftime('%Y-%m-%d %H:%M')}  "
        f"{len(items)} designs{'  [SMOKE]' if args.smoke else ''}\n"
        f"# per design: gen -> IK open_ik -> CEM (grasp gate) -> native A -> live-A reset B "
        f"-> continuous handoff + trajectory-health scorecard.\n")
    cem_iters = 12 if args.smoke else args.cem_iters
    b_jobs = B_RECIPES[args.b_recipe]
    for mid, m in items:
        prev = store.get(mid)
        # done = all requested handoffs present, OR a terminal domain verdict ("note":
        # ungraspable / never lifted / no ckpt). "error" (exception) is transient -> retried.
        if prev is not None and (all(k in prev for _, _, k in b_jobs) or "note" in prev):
            print(f"[skip] {mid} (done)"); continue
        t0 = time.time(); rec = {"id": mid, "morph": list(m),
                                 "delta_m05": [round(a - b, 4) for a, b in zip(m, M05)]}

        def log(msg):
            print(f"[{time.strftime('%H:%M:%S')}] {mid}: {msg}", flush=True)

        try:
            log("gen scene + IK retarget ...")
            scene = gen_scene(m, env)
            rec["ik_resid"] = retarget_scene(scene, BASE_SCENE)
            log(f"CEM grasp ({cem_iters} iters) ...")
            g = run_cem(scene, f"{mid}_cem", env, cem_iters)
            log(f"CEM done: lift={g['cube_lift']:.3f} persist "
                f"{g['thumb_contact_persistence']:.2f}/{g['index_contact_persistence']:.2f}/"
                f"{g['middle_contact_persistence']:.2f}")
            rec.update(lift=g["cube_lift"], tips=g.get("cube_tip_contacts"),
                       pt=g["thumb_contact_persistence"], pi=g["index_contact_persistence"],
                       pm=g["middle_contact_persistence"], g_imbal=g["finger_persistence_imbalance"])
            cem_dir = CEM_OUT / f"{mid}_cem"
            cem_rel = f"results/phase1/morph_sweep/{mid}_cem"
            if g["cube_lift"] < args.lift_thresh:
                rec["note"] = f"ungraspable (lift {g['cube_lift']:.3f} < {args.lift_thresh})"
            else:
                log(f"train Policy A (from scratch, best-of-{args.a_attempts}, "
                    f"open-finger-from-keyframe) ...")
                a, a_tried = train_A(cem_dir, mid, env, args.smoke,
                                     attempts=args.a_attempts, lift_floor=args.a_lift_floor)
                a_ck, a_abort, a_oh = a["ck"], a["aborted"], a["oh"]
                rec["A"] = {"run": a["run"].name if a["run"] else None, "aborted": a_abort,
                            "best_ckpt": a_ck.name if a_ck else None, "best_objheight": a_oh,
                            "health_verdict": (a["health"] or {}).get("verdict"),
                            "attempts": [{"run": x["run"].name if x["run"] else None,
                                          "aborted": x["aborted"], "objheight": x["oh"],
                                          "health_verdict": (x["health"] or {}).get("verdict")}
                                         for x in a_tried]}
                if a_ck is None:
                    rec["note"] = f"A produced no checkpoint ({len(a_tried)} attempts)"
                    log("A produced no checkpoint — skipping B")
                elif (a_oh is not None) and (not args.smoke) and (a_oh < args.a_lift_floor):
                    rec["note"] = (f"A never lifted (best objheight {a_oh} < {args.a_lift_floor}, "
                                   f"{len(a_tried)} attempts)")
                    log(f"A never lifted (best objheight {a_oh}) — skipping B")
                else:
                    log(f"A ok (best {a_ck.name}, objheight {a_oh}, abort {a_abort}, "
                        f"{len(a_tried)} attempt(s))")
                    for b_recipe, sfx, key in b_jobs:
                        log(f"train Policy B [{b_recipe}] (live-A reset, from-A warmstart) ...")
                        b_run, b_ck, b_abort = train_B(cem_rel, a_ck, mid, env, args.smoke,
                                                       recipe=b_recipe, tag_suffix=sfx)
                        rec["B" + sfx] = {"run": b_run.name if b_run else None,
                                          "aborted": b_abort, "recipe": b_recipe}
                        if b_ck is None:
                            rec["note"] = f"B[{b_recipe}] produced no checkpoint"
                            log(f"B[{b_recipe}] produced no checkpoint")
                            continue
                        # SALVAGE: eval the last saved ckpt even if the watchdog aborted mid-run
                        # (a late collapse leaves a healthy earlier ckpt) — let the scorecard judge.
                        if b_abort:
                            log(f"B[{b_recipe}] watchdog-aborted; SALVAGE-eval last ckpt {b_ck.name} ...")
                        else:
                            log(f"B[{b_recipe}] done ({b_ck.name}); continuous handoff eval ...")
                        vid, minz, health = eval_handoff(a_ck, b_ck, cem_dir, mid, env,
                                                         suffix="_self" if sfx else "")
                        rec[key] = {"video": vid, "min_z_post": minz, "health": health,
                                    "b_aborted": bool(b_abort), "a_aborted": bool(a_abort)}
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {str(e)[:160]}"
        rec["secs"] = round(time.time() - t0)
        store.put(rec)
        report.line(_row(rec) + f"   ({rec['secs']}s)")
    sentinel.write(f"{len(store)} designs")
    print(f"[pipeline-sweep] COMPLETE — {len(store)} designs -> {report.path}  "
          f"(sentinel {sentinel.path.name})")


if __name__ == "__main__":
    main()
