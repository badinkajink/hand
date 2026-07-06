"""IK-retargeted re-CEM of the whole morphology landscape (2026-07-01).

Re-scores graspability for every landscape morphology with the CORRECTED keyframe: the
2026-06-25 landscape seeded CEM from the baseline JOINT-space keyframe, which mis-placed the
fingertips on repositioned/lengthened fingers (see retarget_keyframe_ik.py). This sweep, per
morphology: generate scene -> IK-retarget the fingertip world positions -> inject `open_ik`
-> CEM from open_ik -> record per-finger contact persistence. Answers: were the landscape's
"2-finger / ungraspable" verdicts real geometry, or keyframe-transfer artifacts?

CPU-bound (MuJoCo CEM backend) -> safe to run alongside GPU training. Resumable.

Run: MUJOCO_GL=egl uv run --extra rl --extra gpu python scripts/ik_recem_landscape.py
"""
from __future__ import annotations
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from morph_landscape_sweep import morphologies, gen_scene  # reuse the exact design set
from retarget_keyframe_ik import tip_targets, ik_finger, FINGERS, _has_joint, _inject_keyframe
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE_SCENE = ROOT / "assets/mjcf/scene_screwdriver_medium_flat_short_proximal.xml"
CEM_OUT = ROOT / "results/phase1/landscape_ik"
JSON = ROOT / "IK_RECEM_LANDSCAPE.json"
TXT = ROOT / "IK_RECEM_LANDSCAPE.txt"
PALM_JOINTS = ["palm_px", "palm_py", "palm_pz", "palm_rx", "palm_ry", "palm_rz"]


def retarget(scene: Path):
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
    ctrl_vec = []
    for a in range(m.nu):
        jid = m.actuator_trnid[a, 0]
        ctrl_vec.append(float(d.qpos[m.jnt_qposadr[jid]]) if jid >= 0 else 0.0)
    _inject_keyframe(scene, "open_ik",
                     " ".join(f"{v:.6g}" for v in d.qpos),
                     " ".join(f"{v:.6g}" for v in ctrl_vec))
    return {f: round(errs[f] * 1000, 2) for f in FINGERS}  # mm residuals


def run_cem(scene: Path, tag: str, env):
    subprocess.run([sys.executable, str(ROOT / "scripts/phase1_optimize_grasp.py"),
                    "--scene-xml", str(scene), "--keyframe", "open_ik",
                    "--iterations", "200", "--population", "80", "--skip-gif",
                    "--objective-weight-min-finger-persistence", "4.0",
                    "--objective-weight-contact-persistence", "1.5",
                    "--output-dir", str(CEM_OUT), "--tag", tag],
                   check=True, capture_output=True, text=True, env=env, timeout=1200)
    return json.loads((CEM_OUT / tag / "summary.json").read_text())["best_metrics"]


def main():
    import os
    env = dict(os.environ); env.setdefault("MUJOCO_GL", "egl")
    done = {d["id"]: d for d in json.loads(JSON.read_text())} if JSON.exists() else {}
    items = morphologies(10)
    if not TXT.exists():
        TXT.write_text(
            f"# IK-retargeted re-CEM landscape  {time.strftime('%Y-%m-%d %H:%M')}\n"
            f"# CEM seeded from IK open_ik keyframe (fingertip-world-position retarget).\n"
            f"# vs 2026-06-25 (joint-space keyframe): does graspability change?\n"
            f"{'id':16} | {'lift':>6} {'tips':>4} {'persist t/i/m':>16} {'all3':>5} "
            f"{'ik_resid mm t/i/m':>18}\n")
    results = list(done.values())
    for mid, m in items:
        if mid in done:
            print(f"[skip] {mid}"); continue
        t0 = time.time(); rec = {"id": mid, "morph": list(m)}
        try:
            scene = gen_scene(m, env)
            resid = retarget(scene)
            g = run_cem(scene, f"{mid}_ik", env)
            rec.update(lift=g["cube_lift"], tips=g.get("cube_tip_contacts"),
                       pt=g["thumb_contact_persistence"], pi=g["index_contact_persistence"],
                       pm=g["middle_contact_persistence"],
                       all3=g["all_finger_contact_persistence"],
                       ik_resid=resid, scene=str(scene))
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {str(e)[:140]}"
        rec["secs"] = round(time.time() - t0)
        results.append(rec)
        JSON.write_text(json.dumps(results, indent=1))
        line = (f"{mid:16} | {rec.get('lift', float('nan')):6.3f} "
                f"{str(rec.get('tips', '-')):>4} "
                f"{rec.get('pt', 0):.2f}/{rec.get('pi', 0):.2f}/{rec.get('pm', 0):.2f}".ljust(16)
                + f" {rec.get('all3', 0):5.2f} "
                + (f"{resid['thumb']:.1f}/{resid['index']:.1f}/{resid['middle']:.1f}"
                   if 'error' not in rec else rec['error']))
        print(line + f"  ({rec['secs']}s)")
        with TXT.open("a") as f:
            f.write(line + "\n")
    print(f"[ik-recem] done -> {TXT}")


if __name__ == "__main__":
    main()
