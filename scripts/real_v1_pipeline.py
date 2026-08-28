"""Per-design pipeline for the `real_v1` (CAD-matched hardware) hand: design -> scene -> pose
-> CEM grasp -> Policy A -> Policy B -> handoff eval.

Same shape as `morph_pipeline_sweep.py`, which runs the m05-lineage baseline topology, and it
drives the same two training launchers so the recipes stay shared. Three things differ, all
forced by the hardware model:

  * BASE PAIR is `assets/mjcf/real_v1/` (built by scripts/build_real_v1_scenes.py), not the
    baseline hand. Different link lengths, different ROM, overlapping links, a real workspace.
  * DESIGN SPACE is 6-dimensional. The hardware has XY gantries and CAD-fixed link lengths, so
    `REAL_V1_WORKSPACE` pins len to 0 and the three `_len` slides are zero-travel shims.
  * POSE comes from `scripts/fit_real_v1_pose.py`, not `retarget_keyframe_ik.py`. There is no
    known-good grasp on this topology to transfer, and palm height has to be re-solved per
    design or short-fingered layouts get spurious "ungraspable" verdicts (LINK_LENGTH_GATE
    trap #1). The fitter writes `open_ik`, which CEM seeds from and which the RL env reads via
    `--open-finger-from-keyframe`.

NOTHING TRANSFERS ONTO THIS HAND. a10/b33 were trained on a 117 mm finger with coincident
yaw/MCP axes; b33 did not survive even a proximal-length change on the SAME topology. Policy A
trains from scratch (`WARMSTART=none`) and Policy B warmstarts only from THIS design's own A.

Stages are independently selectable and the JSON checkpoint is resumable, because the cheap
stages (generate/pose/CEM, ~5 min per design on CPU+1 GPU) want iterating on while the
expensive one (A+B, ~90 min per design) wants leaving alone overnight.

    # cheap: every design through the grasp, look at the videos before spending GPU-hours
    MUJOCO_GL=egl uv run --extra rl --extra gpu python scripts/real_v1_pipeline.py --stages grasp

    # expensive, detached
    nohup setsid env MUJOCO_GL=egl uv run --extra rl --extra gpu \
      python scripts/real_v1_pipeline.py --stages all --only rv00_nominal,rv01_compact \
      > logs/real_v1_pipeline.run.log 2>&1 & disown
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from morphohand.sampling.morphology import (  # noqa: E402
    morph_to_array,
    real_v1_compact_design,
    real_v1_mount_positions,
)
from morphohand.studies import runlib  # noqa: E402
from morphohand.studies.runlib import ROOT, best_a_ckpt, final_ckpt, iter_objheight, latest_run  # noqa: E402
from morphohand.tools.video_paths import experiment_dir  # noqa: E402

BASE_HAND = ROOT / "assets/mjcf/real_v1/real_hand.xml"
BASE_SCENE = ROOT / "assets/mjcf/real_v1/scenes/scene_screwdriver_medium.xml"
DATE = "20260827"                                        # the day this study was set up
GEN = ROOT / f"assets/mjcf/experimental/{DATE}-real_v1"   # generated scenes
CEM_OUT = ROOT / "results/phase1/real_v1"                 # results/ is gitignored
DOCS = ROOT / f"docs/experiments/{DATE}-real_v1"          # tracked summaries
VID_OUT = experiment_dir("real_v1")

STAGES = ("generate", "pose", "cem", "A", "B", "handoff")
CHEAP = ("generate", "pose", "cem")

# THE DESIGN SET is a 2x2 factorial plus a centre point, in the two dimensions that actually
# exist once the palm is fitted.
#
# The workspace has three knobs (thumb forward / pair back / pair inward), but `fit_real_v1_pose`
# re-centres palm X so the thumb-pair midpoint sits over the shaft. That makes "thumb forward"
# and "pair back" the SAME HAND: (1,0,0) and (0,1,0) both put the thumb 70 mm from the pair and
# both fit at 68.0 mm of grip depth with identical joint angles. Only two things vary:
#
#     thumb <-> pair separation along X   100 mm (0,0,*) .. 40 mm (1,1,*)
#     index <-> middle separation along Y 110 mm (*,*,0) .. 50 mm (*,*,1)
#
# So the set spans those two and drops the redundant single-knob designs. X separation sets how
# hard the pinch has to squeeze and how deep the palm can sit; Y separation sets how much of the
# shaft's length the pair straddles, which is the lever arm for tipping it upright.
DESIGNS: dict[str, tuple[float, float, float]] = {
    "rv00_wide":     (0.0, 0.0, 0.0),      # X 100  Y 110 -- CAD-nominal gantry centres
    "rv01_compact":  (1.0, 1.0, 1.0),      # X  40  Y  50 -- every gantry inboard
    "rv02_narrowx":  (1.0, 1.0, 0.0),      # X  40  Y 110 -- tight pinch, wide straddle
    "rv03_narrowy":  (0.0, 0.0, 1.0),      # X 100  Y  50 -- wide pinch, narrow straddle
    "rv04_mid":      (0.5, 0.5, 0.5),      # X  70  Y  80
}


def design_vector(mid: str) -> list[float]:
    return [round(float(v), 4) for v in morph_to_array(real_v1_compact_design(*DESIGNS[mid]))]


def gen_scene(vec: list[float], env) -> Path:
    GEN.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/generate_morphology_xml.py"),
                    "--base-hand-xml", str(BASE_HAND), "--base-scene-xml", str(BASE_SCENE),
                    "--output-dir", str(GEN),
                    "--thumb", *map(str, vec[0:3]), "--index", *map(str, vec[3:6]),
                    "--middle", *map(str, vec[6:9])],
                   check=True, capture_output=True, text=True, env=env, timeout=120)
    return sorted(GEN.glob("scene_*.xml"), key=lambda p: p.stat().st_mtime)[-1]


# Straddle candidates, in millimetres. CEM picks between them — see `fit_and_cem`.
SPREADS_MM = (30, 40)


def fit_pose(scene: Path, tag: str, env, spread_mm: int) -> dict:
    """Solve palm pose + finger angles at a GIVEN straddle and write `open_ik` / `open`.

    This replaces the retarget step. It must run on EVERY generated scene: the generator copies
    the base design's keyframes forward verbatim, so an unfitted scene carries a grasp authored
    for different mounts at a palm height its fingers cannot reach from.
    """
    out = CEM_OUT / tag / "pose_fit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([sys.executable, str(ROOT / "scripts/fit_real_v1_pose.py"),
                        "--scene", str(scene), "--write", "--also-open", "--quiet",
                        "--spread", str(spread_mm / 1000.0), "--json", str(out)],
                       check=False, capture_output=True, text=True, env=env, timeout=900)
    if r.returncode != 0 or not out.exists():
        raise RuntimeError(f"pose fit failed for {tag}: {r.stdout}\n{r.stderr}")
    return json.loads(out.read_text())


def fit_and_cem(scene: Path, mid: str, env, iters: int, render: bool) -> dict:
    """Fit + CEM at each candidate straddle; keep whichever actually HOLDS the shaft.

    The straddle — how far apart index and middle sit along the shaft — is a choice about where
    to grasp, not a hardware parameter, and it decides verdicts. Three ways of picking it from
    geometry alone were tried on 2026-08-27 and all three produced false verdicts:

      nominal 30 mm first          rv01_compact's grasp lifted then DROPPED (held -1.4 mm);
                                   the same hand at 40 mm held +47.0 mm.
      widest reachable             fixed rv01, broke rv00_wide the same way.
      scripted hold probe          says rv01 has no holding pose at all, when CEM finds one.

    Reachability cannot see the pitch-out failure, and a geometric grip is not a CEM grip. So
    the arbiter is CEM itself, on the metric that matters (held lift, not peak). It costs one
    extra CEM per design and it removes the guessing.
    """
    best = None
    errors = []
    for sp in SPREADS_MM:
        tag = f"{mid}_sp{sp}"
        # One unreachable straddle is not a failed design. rv02_narrowx cannot reach 30 mm and
        # grasps fine at 40 mm; aborting the design on the first miss threw that away.
        try:
            pose = fit_pose(scene, tag, env, sp)
            cem = run_cem(scene, tag, env, iters, render)
        except Exception as exc:
            errors.append(f"{sp}mm: {type(exc).__name__}: {exc}")
            runlib.log(f"  {mid} spread {sp}mm -> {type(exc).__name__}")
            continue
        runlib.log(f"  {mid} spread {sp}mm -> held {held_lift(cem)*1000:+.1f}mm "
                   f"pers {cem.get('contact_persistence', float('nan')):.2f}")
        if best is None or held_lift(cem) > held_lift(best["cem"]):
            best = {"spread_mm": sp, "tag": tag, "pose": pose, "cem": cem}
    if best is None:
        raise RuntimeError(f"no straddle worked for {mid}: " + " | ".join(errors))
    return best


def run_cem(scene: Path, mid: str, env, iters: int, render: bool) -> dict:
    e = runlib.warp_cache_env(env)
    cmd = [sys.executable, str(ROOT / "scripts/phase1_optimize_grasp.py"),
           "--scene-xml", str(scene), "--keyframe", "open_ik",
           "--iterations", str(iters), "--population", "80",
           "--objective-weight-min-finger-persistence", "4.0",
           "--objective-weight-contact-persistence", "1.5",
           "--output-dir", str(CEM_OUT), "--tag", mid]
    # Rendering a prelim grasp pass is not optional by default — metric-only debugging has
    # hidden finger drift and ground collisions in this repo before. --no-render is for reruns.
    if not render:
        cmd.append("--skip-gif")
    subprocess.run(cmd, check=True, capture_output=True, text=True, env=e, timeout=3600)
    return json.loads((CEM_OUT / mid / "summary.json").read_text())["best_metrics"]


def held_lift(cem: dict) -> float:
    """How high the shaft is STILL held at the end of the hold phase, above where it started.

    NOT `cube_lift`, which CEM defines as PEAK height minus start and therefore scores a grasp
    that lifts the shaft and then drops it identically to one that keeps it. On the 2026-08-27
    real_v1 set that difference decided two designs out of five: `rv01_compact` reported
    cube_lift 0.047 with a held lift of -0.001, i.e. the shaft was back on the table. Gating on
    the peak would have sent both into a 90-minute Policy A run on a grasp that does not hold.
    """
    if not cem:
        return float("nan")
    return float(cem.get("cube_z_after_hold", float("nan"))
                 - cem.get("cube_z_before_lift", float("nan")))


VERDICT_RANK = {"PASS": 3, "WARN": 2, "FAIL": 1, None: 0}


def _train_A_once(cem_dir: Path, mid: str, t: int, env, smoke: bool, lift_delta: float) -> dict:
    tag = f"policyA_{mid}_t{t}"
    log = ROOT / f"logs/real_v1_A_{mid}_t{t}.trainer.log"
    e = dict(env)
    e.update(MORPH_RUN=str(cem_dir), WARMSTART="none", LIFT_DELTA_A=str(lift_delta),
             EXTRA_ARGS="--open-finger-from-keyframe --lift-phase-start-step 60",
             TAG=tag, LOG=str(log), NUM_ENVS="2048", SMOKE="1" if smoke else "0")
    if not smoke:
        e["TOTAL_TS"] = "30000000"
    Path(str(log) + ".COLLAPSED").unlink(missing_ok=True)
    subprocess.run(["bash", str(ROOT / "scripts/train_A_on_morph.sh")],
                   check=False, capture_output=True, text=True, env=e, timeout=9000)
    run = latest_run(f"*{tag}")
    aborted = Path(str(log) + ".COLLAPSED").exists() or run is None
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
    return {"run": str(run) if run else None, "ck": str(ck) if ck else None,
            "aborted": aborted, "health": health, "oh": best_oh}


def train_A(cem_dir: Path, mid: str, env, smoke: bool, attempts: int, lift_floor: float,
            lift_delta: float):
    """From-scratch A, best-of-`attempts`. A's draw is the dominant evaluator noise source in
    this program (joint sd 0.41 vs 0.09 for B-only), so retry a collapsed or FAIL draw and keep
    every attempt in the record."""
    tried = []
    for t in range(1 if smoke else max(1, attempts)):
        a = _train_A_once(cem_dir, mid, t, env, smoke, lift_delta)
        tried.append(a)
        verdict = (a["health"] or {}).get("verdict")
        lifted = smoke or a["oh"] is None or a["oh"] >= lift_floor
        if (not a["aborted"]) and a["ck"] and lifted and verdict != "FAIL":
            break

    def rank(a):
        return (0 if a["aborted"] else 1,
                VERDICT_RANK.get((a["health"] or {}).get("verdict"), 0),
                a["oh"] if a["oh"] is not None else -1.0)

    return max(tried, key=rank), tried


def train_B(cem_rel: str, a_ck: str, mid: str, env, smoke: bool, lift_delta: float):
    """Live-A reset B, warmstarted from THIS design's own A.

    B_CKPT = the design's own A, never b33: b33's residual encodes m05's finger placement and
    ejects the shaft on a hand whose fingers sit somewhere else (gotcha #5, and it failed even
    across a proximal-length change on the same topology).
    """
    tag = f"policyB_{mid}_reorient"
    log = ROOT / f"logs/real_v1_B_{mid}.trainer.log"
    e = dict(env)
    e.update(MORPH=cem_rel, A_CKPT=a_ck, B_CKPT=a_ck, LIFT_DELTA=str(lift_delta),
             RECIPE="b_liveA", EXTRA_ARGS="--open-finger-from-keyframe",
             ONSET_STEP="40", BLEND="8", LIFT_TERM_START="58", REORIENT_START="58",
             TAG=tag, LOG=str(log), NUM_ENVS="3072", SMOKE="1" if smoke else "0")
    if not smoke:
        e["TOTAL_TS"] = "20000000"
    Path(str(log) + ".COLLAPSED").unlink(missing_ok=True)
    subprocess.run(["bash", str(ROOT / "scripts/train_handoff_liveA_reset.sh")],
                   check=False, capture_output=True, text=True, env=e, timeout=7200)
    run = latest_run(f"*{tag}")
    aborted = Path(str(log) + ".COLLAPSED").exists() or run is None
    ck = final_ckpt(run)
    return {"run": str(run) if run else None, "ck": str(ck) if ck else None, "aborted": aborted}


def eval_handoff(a_ck: str, b_ck: str, cem_dir: Path, mid: str, env, lift_delta: float):
    VID_OUT.mkdir(parents=True, exist_ok=True)
    out = VID_OUT / f"{mid}_handoff.mp4"
    e = runlib.warp_cache_env(env)
    r = subprocess.run([sys.executable, str(ROOT / "scripts/rl_demo_handoff_continuous.py"),
                        "--policy-a", a_ck, "--policy-b", b_ck,
                        "--morphology-run", str(cem_dir), "--lift-delta", str(lift_delta),
                        "--open-finger-from-keyframe", "--output", str(out)],
                       check=False, capture_output=True, text=True, env=e, timeout=1800)
    min_z_post = None
    for ln in (r.stdout or "").splitlines():
        if "honest hold metric" in ln:
            for t in ln.split(":")[-1].split():
                try:
                    min_z_post = float(t)
                    break
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
    if min_z_post is None and health:
        min_z_post = (health.get("metrics") or {}).get("min_z_hold")
    return {"video": str(out) if out.exists() else None, "min_z_post": min_z_post,
            "health": health}


def _row(rec: dict) -> str:
    p = rec.get("pose") or {}
    g = rec.get("cem") or {}
    h = rec.get("handoff") or {}
    hm = ((h.get("health") or {}).get("metrics") or {})
    verdict = (h.get("health") or {}).get("verdict") or rec.get("note", "—")
    return (f"{rec['id']:14} | depth {p.get('grip_depth_mm', float('nan')):5.1f}mm "
            f"ceil {p.get('clearance_ceiling', float('nan')):.2f} "
            f"| sp {rec.get('spread_mm', 0):2d}mm held {held_lift(g):+.3f} "
            f"peak {g.get('cube_lift', float('nan')):.3f} "
            f"tips {g.get('cube_tip_contacts', float('nan')):.1f} "
            f"pers {g.get('contact_persistence', float('nan')):.2f} "
            f"| minZ {str(h.get('min_z_post', '—')):>7} "
            f"cos {str(hm.get('held_cos_tail', '—')):>7} | {verdict}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", default="grasp",
                    help="'grasp' (generate+pose+cem), 'all', or a comma list from "
                         + ",".join(STAGES))
    ap.add_argument("--only", default=None, help="comma list of design ids")
    ap.add_argument("--tag", default="real_v1")
    ap.add_argument("--cem-iters", type=int, default=200)
    ap.add_argument("--no-render", action="store_true", help="skip the CEM grasp GIF")
    ap.add_argument("--lift-thresh", type=float, default=0.03,
                    help="HELD lift below this => ungraspable, skip A/B (see held_lift)")
    ap.add_argument("--lift-delta", type=float, default=0.10,
                    help="Policy A's delivery height above the object's start")
    ap.add_argument("--a-attempts", type=int, default=2)
    ap.add_argument("--a-lift-floor", type=float, default=0.06)
    ap.add_argument("--smoke", action="store_true", help="1M-timestep A and B, validates glue")
    args = ap.parse_args()

    stages = (CHEAP if args.stages == "grasp"
              else STAGES if args.stages == "all"
              else tuple(s.strip() for s in args.stages.split(",")))
    unknown = set(stages) - set(STAGES)
    if unknown:
        ap.error(f"unknown stage(s): {sorted(unknown)}")

    ids = [i.strip() for i in args.only.split(",")] if args.only else list(DESIGNS)
    missing = set(ids) - set(DESIGNS)
    if missing:
        ap.error(f"unknown design(s): {sorted(missing)}")

    env = runlib.base_env()
    store = runlib.RecordStore(DOCS / f"{args.tag}.json")
    report = runlib.TxtReport(
        DOCS / f"{args.tag}.txt",
        f"# real_v1 pipeline — {args.tag}\n"
        f"# design | grip depth + clearance ceiling (fit_real_v1_pose) "
        f"| CEM grasp: HELD lift (peak lift) tip contacts, contact persistence "
        f"| A->B handoff\n")
    done = runlib.Sentinel(DOCS / f"{args.tag}.DONE")
    done.clear()

    for mid in ids:
        rec = store.get(mid) or {"id": mid, "design": DESIGNS[mid], "vector": design_vector(mid)}
        rec["mounts_mm"] = {f: [round(v * 1000, 1) for v in xy] for f, xy
                            in real_v1_mount_positions(
                                real_v1_compact_design(*DESIGNS[mid])).items()}
        t0 = time.time()

        def need(stage: str, key: str) -> bool:
            """Run `stage` if it was asked for, or if a LATER requested stage needs its output.

            Not simply "or the output is missing": that quietly promotes a 5-minute GPU CEM into
            a `--stages generate,pose` run, which is how this study first spent half an hour
            doing something nobody asked for.
            """
            if stage in stages:
                return True
            if rec.get(key):
                return False
            later = STAGES[STAGES.index(stage) + 1:]
            return any(s in stages for s in later)

        try:
            if need("generate", "scene"):
                rec["scene"] = str(gen_scene(rec["vector"], env))
            scene = Path(rec["scene"])

            # pose and cem are one stage here: the straddle is chosen BY the CEM result, so
            # fitting without grasping would leave the scene at an arbitrary one of them.
            if need("pose", "cem") or need("cem", "cem"):
                won = fit_and_cem(scene, mid, env, args.cem_iters, not args.no_render)
                rec.update(pose=won["pose"], cem=won["cem"], cem_tag=won["tag"],
                           spread_mm=won["spread_mm"])
                if rec["pose"]["self_collisions"]:
                    rec["note"] = "SELF-COLLIDING POSE"
                    runlib.log(f"{mid}: {rec['note']} {rec['pose']['self_collisions']}")
            cem_dir = CEM_OUT / rec["cem_tag"]

            graspable = held_lift(rec.get("cem") or {}) >= args.lift_thresh
            if not graspable:
                rec["note"] = "UNGRASPABLE"
            elif "A" in stages:
                kept, tried = train_A(cem_dir, mid, env, args.smoke, args.a_attempts,
                                      args.a_lift_floor, args.lift_delta)
                rec["A"] = {"kept": kept, "attempts": tried}
                if kept["ck"] and "B" in stages:
                    rec["B"] = train_B(str(cem_dir.relative_to(ROOT)), kept["ck"], mid, env,
                                       args.smoke, args.lift_delta)
                    if rec["B"]["ck"] and "handoff" in stages:
                        rec["handoff"] = eval_handoff(kept["ck"], rec["B"]["ck"], cem_dir, mid,
                                                      env, args.lift_delta)
        except Exception as exc:                              # one design never sinks the run
            rec["error"] = f"{type(exc).__name__}: {exc}"
            runlib.log(f"{mid}: ERROR {rec['error']}")

        rec["seconds"] = round(time.time() - t0, 1)
        store.put(rec)
        report.line(_row(rec))

    done.write(f"{len(ids)} designs, stages={','.join(stages)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
