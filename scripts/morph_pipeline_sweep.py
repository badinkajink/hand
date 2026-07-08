"""Full per-morphology A->B pipeline sweep — the HONEST, health-gated design search (2026-07-03).

Supersedes the 2026-06-25 landscape sweep (morph_landscape_sweep.py), which scored designs on a
skip-lift TELEPORT reorienter warmstarting B4 (no native Policy A) and a JOINT-space keyframe that
under-scored graspability. This sweep runs, per morphology, the exact CLEAN m05-fixed pipeline
(the policy in docs/rl/videos/reorient/handoff_m05_FIXED.mp4, a10 -> b33):

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

Run (detached):
  nohup setsid env MUJOCO_GL=egl uv run --extra rl --extra gpu \
    python scripts/morph_pipeline_sweep.py --morph-set initial8 > sweep_initial8.run.log 2>&1 &
Smoke (validate the glue, ~10 min, tiny ts):  add --smoke
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from retarget_keyframe_ik import tip_targets, ik_finger, _has_joint, _inject_keyframe, FINGERS  # noqa: E402
import mujoco  # noqa: E402

BASE_HAND = ROOT / "assets/mjcf/hand.xml"
BASE_SCENE = ROOT / "assets/mjcf/scene_screwdriver_medium_flat_short_proximal.xml"
GEN = ROOT / "assets/mjcf/experimental/morph_sweep"          # generated scenes (gitignored)
CEM_OUT = ROOT / "results/phase1/morph_sweep"                 # CEM grasp outputs
VID_OUT = ROOT / "docs/rl/videos/reorient/sweep"             # handoff videos + .health.json

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


def morph_set(name: str, n: int, seed: int, center):
    """Return [(id, 9-vector), ...]. `initial8` = interpretable coordinate moves around m05."""
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


def ik_retarget(scene: Path):
    """Inject an `open_ik` keyframe: fingertip world positions from the baseline keyframe, IK'd
    onto this morphology's (repositioned/lengthened) fingers. Returns per-finger mm residuals."""
    tips, palm, obj_qpos = tip_targets(str(BASE_SCENE), "open_short_manual")
    m = mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m)
    try:
        mujoco.mj_resetDataKeyframe(m, d, m.key("open_short_manual").id)
    except Exception:
        pass
    for j, v in palm.items():
        if _has_joint(m, j):
            d.qpos[m.jnt_qposadr[m.joint(j).id]] = v
    d.qpos[:7] = obj_qpos
    mujoco.mj_forward(m, d)
    errs = {f: ik_finger(m, d, f, tips[f]) for f in FINGERS}
    ctrl = []
    for a in range(m.nu):
        jid = m.actuator_trnid[a, 0]
        ctrl.append(float(d.qpos[m.jnt_qposadr[jid]]) if jid >= 0 else 0.0)
    _inject_keyframe(scene, "open_ik", " ".join(f"{v:.6g}" for v in d.qpos),
                     " ".join(f"{v:.6g}" for v in ctrl))
    return {f: round(errs[f] * 1000, 2) for f in FINGERS}


def run_cem(scene: Path, tag: str, env, iters: int) -> dict:
    e = dict(env); e["WARP_CACHE_PATH"] = _tmp()
    subprocess.run([sys.executable, str(ROOT / "scripts/phase1_optimize_grasp.py"),
                    "--scene-xml", str(scene), "--keyframe", "open_ik",
                    "--iterations", str(iters), "--population", "80", "--skip-gif",
                    "--objective-weight-min-finger-persistence", "4.0",
                    "--objective-weight-contact-persistence", "1.5",
                    "--output-dir", str(CEM_OUT), "--tag", tag],
                   check=True, capture_output=True, text=True, env=e, timeout=1800)
    return json.loads((CEM_OUT / tag / "summary.json").read_text())["best_metrics"]


def _tmp() -> str:
    return subprocess.run(["mktemp", "-d"], capture_output=True, text=True).stdout.strip()


def _latest_run(tag: str) -> Path | None:
    runs = sorted((ROOT / "results/rl").glob(f"*{tag}"), key=lambda p: p.stat().st_mtime)
    return runs[-1] if runs else None


def _final_ckpt(run: Path) -> Path | None:
    cks = sorted((run / "tensorboard").glob("model_*.pt"),
                 key=lambda p: int(p.stem.split("_")[1]))
    return cks[-1] if cks else None


def _iter_objheight(log: Path) -> dict[int, float]:
    """Parse the trainer log -> {iter: mean object_height}. For picking A's best lift ckpt."""
    import re
    if not log.exists():
        return {}
    text = log.read_text(errors="ignore")
    iters = [int(m.group(1)) for m in re.finditer(r"Learning iteration\s+(\d+)/", text)]
    chunks = re.split(r"Learning iteration\s+\d+/", text)
    oh = {}
    for chunk, it in zip(chunks[1:], iters):
        mm = re.search(r"lift_height/object_height:\s+([-\d.]+)", chunk)
        if mm:
            oh[it] = float(mm.group(1))
    return oh


def _best_a_ckpt(run: Path, log: Path):
    """A's BEST-lifting saved checkpoint (robust to a late mid-training collapse — the final ckpt
    may be post-collapse). Returns (ckpt, best_objheight). Falls back to final if no log data."""
    cks = {int(p.stem.split("_")[1]): p for p in (run / "tensorboard").glob("model_*.pt")}
    if not cks:
        return None, None
    oh = _iter_objheight(log)
    if not oh:
        return _final_ckpt(run), None
    best, best_oh = None, -1.0
    for n in sorted(cks):
        v = oh.get(n)
        if v is None:                       # ckpt iter has no logged oh -> nearest earlier iter
            earlier = [i for i in oh if i <= n]
            v = oh[max(earlier)] if earlier else -1.0
        if v > best_oh:
            best_oh, best = v, cks[n]
    return best, round(best_oh, 3)


def train_A(cem_dir: Path, mid: str, env, smoke: bool):
    # ALWAYS from scratch: warmstarting A (from a01 OR a10) loads a grip-specific residual that
    # EJECTS the re-CEM'd object (confirmed: a10-warmstart canary never lifted, obj 0.0 from iter
    # 0). From scratch the residual~0, so the open-loop CEM grip + scripted lift does the lifting.
    tag = f"policyA_{mid}"
    log = ROOT / f"logs/sweep_A_{mid}.trainer.log"
    e = dict(env)
    e.update(MORPH_RUN=str(cem_dir), WARMSTART="none", LIFT_DELTA_A="0.10",
             EXTRA_ARGS="--open-finger-from-keyframe --lift-phase-start-step 60",
             TAG=tag, LOG=str(log), NUM_ENVS="2048", SMOKE="1" if smoke else "0")
    if not smoke:
        e["TOTAL_TS"] = "30000000"
    subprocess.run(["bash", str(ROOT / "scripts/train_A_on_morph.sh")],
                   check=False, capture_output=True, text=True, env=e, timeout=9000)
    run = _latest_run(tag)
    aborted = Path(str(log) + ".COLLAPSED").exists() or run is None
    # Checkpoint choice: on CLEAN completion use the FINAL ckpt (fully trained -> best-BALANCED
    # grip; early ckpts lift marginally HIGHER but have an under-refined grip -> idle-finger, which
    # sank valfix2). Only on a watchdog ABORT (mid-training collapse) salvage the best pre-collapse
    # ckpt by object-height. `best_oh` is that ckpt's logged lift, for the a-lift-floor gate.
    oh_map = _iter_objheight(log) if run else {}
    if run and aborted:
        ck, best_oh = _best_a_ckpt(run, log)
    elif run:
        ck = _final_ckpt(run)
        best_oh = oh_map.get(int(ck.stem.split("_")[1])) if ck and oh_map else None
    else:
        ck, best_oh = None, None
    health = None
    if ck:
        hj = (_final_ckpt(run) or ck).with_suffix(".health.json")
        if hj.exists():
            try:
                health = json.loads(hj.read_text())
            except Exception:
                pass
    return run, ck, aborted, health, best_oh


def train_B(cem_rel: str, a_ck: Path, mid: str, env, smoke: bool):
    # live-A DRIVER = this design's own A (lift matches the design) AND B WARMSTART = the same A
    # (the hold-first prior — the m05/b33 recipe). CRITICAL: --open-finger-from-keyframe (the
    # live-A script omits it → B resets to the baseline flung-out-thumb open pose → wrong grip →
    # the driver drops the object → B never learns; this sank s03/s04/s06/s07 in the initial8).
    tag = f"policyB_{mid}_reorient"
    log = ROOT / f"logs/sweep_B_{mid}.trainer.log"
    e = dict(env)
    e.update(MORPH=cem_rel, A_CKPT=str(a_ck), B_CKPT=str(a_ck), LIFT_DELTA="0.10",
             EXTRA_ARGS="--open-finger-from-keyframe",
             ONSET_STEP="40", BLEND="8", LIFT_TERM_START="58", REORIENT_START="58",
             TAG=tag, LOG=str(log), NUM_ENVS="3072", SMOKE="1" if smoke else "0")
    if not smoke:
        e["TOTAL_TS"] = "20000000"
    subprocess.run(["bash", str(ROOT / "scripts/train_handoff_liveA_reset.sh")],
                   check=False, capture_output=True, text=True, env=e, timeout=7200)
    run = _latest_run(tag)
    aborted = Path(str(log) + ".COLLAPSED").exists() or run is None
    return run, (_final_ckpt(run) if run else None), aborted


def eval_handoff(a_ck: Path, b_ck: Path, cem_dir: Path, mid: str, env):
    VID_OUT.mkdir(parents=True, exist_ok=True)
    out = VID_OUT / f"{mid}_handoff.mp4"
    e = dict(env); e["WARP_CACHE_PATH"] = _tmp()
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
    return (f"{rec['id']:16} | lift {g.get('lift', float('nan')):.3f} "
            f"tip {g.get('tips', '-')} pers {g.get('pt', 0):.2f}/{g.get('pi', 0):.2f}/{g.get('pm', 0):.2f}"
            f" | minZ {str(r.get('min_z_post', '—')):>6} cos {str(hm.get('held_cos_tail', '—')):>6}"
            f" jerk {str(hm.get('ang_jerk', '—')):>5} force {str(hm.get('tip_force', '—')):>5}"
            f" drift {str(hm.get('net_drift_cm', '—')):>4}cm | {verdict}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--morph-set", default="initial8", help="initial8 | local")
    ap.add_argument("--n", type=int, default=16, help="designs for --morph-set local")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--center", default="m05", help="local-search center: m05 | best | 9 comma-floats")
    ap.add_argument("--lift-thresh", type=float, default=0.03, help="cube_lift below this => ungraspable")
    ap.add_argument("--cem-iters", type=int, default=200)
    ap.add_argument("--smoke", action="store_true", help="tiny ts, validate the glue end-to-end")
    ap.add_argument("--tag", default=None, help="output tag (default = morph-set)")
    ap.add_argument("--a-lift-floor", type=float, default=0.06,
                    help="A's best-ckpt object_height below this => A never lifted, skip B")
    args = ap.parse_args()
    env = dict(os.environ); env.setdefault("MUJOCO_GL", "egl")

    tag = args.tag or args.morph_set
    JSON = ROOT / f"docs/experiments/MORPH_PIPELINE_{tag}.json"
    TXT = ROOT / f"docs/experiments/MORPH_PIPELINE_{tag}.txt"
    SENTINEL = ROOT / f"logs/MORPH_PIPELINE_{tag}.DONE"
    if SENTINEL.exists():
        SENTINEL.unlink()

    center = M05
    if args.center == "best" and (ROOT / "docs/experiments/MORPH_PIPELINE_best_center.json").exists():
        center = tuple(json.loads((ROOT / "docs/experiments/MORPH_PIPELINE_best_center.json").read_text()))
    elif "," in args.center:
        center = tuple(float(x) for x in args.center.split(","))
    items = morph_set(args.morph_set, args.n, args.seed, center)

    done = {d["id"]: d for d in json.loads(JSON.read_text())} if JSON.exists() else {}
    if not TXT.exists():
        TXT.write_text(
            f"# full A->B pipeline sweep '{tag}'  {time.strftime('%Y-%m-%d %H:%M')}  "
            f"{len(items)} designs{'  [SMOKE]' if args.smoke else ''}\n"
            f"# per design: gen -> IK open_ik -> CEM (grasp gate) -> native A -> live-A reset B "
            f"-> continuous handoff + trajectory-health scorecard.\n")
    results = list(done.values())
    cem_iters = 12 if args.smoke else args.cem_iters
    for mid, m in items:
        if mid in done and "handoff" in done[mid]:
            print(f"[skip] {mid} (done)"); continue
        t0 = time.time(); rec = {"id": mid, "morph": list(m),
                                 "delta_m05": [round(a - b, 4) for a, b in zip(m, M05)]}

        def log(msg):
            print(f"[{time.strftime('%H:%M:%S')}] {mid}: {msg}", flush=True)

        try:
            log("gen scene + IK retarget ...")
            scene = gen_scene(m, env)
            rec["ik_resid"] = ik_retarget(scene)
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
                log("train Policy A (from scratch, open-finger-from-keyframe) ...")
                a_run, a_ck, a_abort, a_health, a_oh = train_A(cem_dir, mid, env, args.smoke)
                rec["A"] = {"run": a_run.name if a_run else None, "aborted": a_abort,
                            "best_ckpt": a_ck.name if a_ck else None, "best_objheight": a_oh,
                            "health_verdict": (a_health or {}).get("verdict")}
                if a_ck is None:
                    rec["note"] = "A produced no checkpoint"
                    log("A produced no checkpoint — skipping B")
                elif (a_oh is not None) and (not args.smoke) and (a_oh < args.a_lift_floor):
                    rec["note"] = f"A never lifted (best objheight {a_oh} < {args.a_lift_floor})"
                    log(f"A never lifted (best objheight {a_oh}) — skipping B")
                else:
                    log(f"A ok (best {a_ck.name}, objheight {a_oh}, abort {a_abort}); "
                        f"train Policy B (live-A reset, from-A warmstart) ...")
                    b_run, b_ck, b_abort = train_B(cem_rel, a_ck, mid, env, args.smoke)
                    rec["B"] = {"run": b_run.name if b_run else None, "aborted": b_abort}
                    if b_ck is None:
                        rec["note"] = "B produced no checkpoint"
                        log("B produced no checkpoint")
                    else:
                        # SALVAGE: eval the last saved ckpt even if the watchdog aborted mid-run
                        # (a late collapse leaves a healthy earlier ckpt) — let the scorecard judge.
                        if b_abort:
                            rec["b_aborted"] = True
                            log(f"B watchdog-aborted; SALVAGE-eval last ckpt {b_ck.name} ...")
                        else:
                            log(f"B done ({b_ck.name}); continuous handoff eval ...")
                        vid, minz, health = eval_handoff(a_ck, b_ck, cem_dir, mid, env)
                        rec["handoff"] = {"video": vid, "min_z_post": minz, "health": health,
                                          "b_aborted": bool(b_abort), "a_aborted": bool(a_abort)}
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {str(e)[:160]}"
        rec["secs"] = round(time.time() - t0)
        done[mid] = rec
        results = list(done.values())
        JSON.write_text(json.dumps(results, indent=1))
        line = _row(rec) + f"   ({rec['secs']}s)"
        print(line, flush=True)
        with TXT.open("a") as f:
            f.write(line + "\n")
    SENTINEL.write_text(time.strftime("%Y-%m-%d %H:%M:%S") + f"  {len(results)} designs\n")
    print(f"[pipeline-sweep] COMPLETE — {len(results)} designs -> {TXT}  (sentinel {SENTINEL.name})")


if __name__ == "__main__":
    main()
