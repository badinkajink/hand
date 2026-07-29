"""Thumb-opposition GRASP-BALANCE sweep (2026-06-22, autonomous overnight).

Tests the structural hypothesis behind the "excessive grip force": the live-A grip is a
degenerate pinch (one finger idle, the load on the other two) BECAUSE of the thumb's placement.
At the GRASP level we already see it — the baseline morphology's CEM-optimized grasp uses only
2 fingers (middle contact-persistence 0.0, finger_persistence_imbalance 0.21).

For a grid of thumb positions (x, y, len) — index/middle held at baseline — we run the existing
CEM grasp optimizer (~20-30 s each) and read its per-finger balance metrics. The question:
**is there a thumb placement where the optimized grasp becomes a balanced 3-finger tripod**
(all-finger contact, low imbalance) while still lifting the object? If yes, morphology is the
fix and that thumb position is the candidate for a full A+B retrain. CEM is cheap, each candidate
is independent, failures are isolated — robust to run unattended.

Results stream to MORPH_GRASP_SWEEP_RESULTS.txt (incremental, survives interruption).

Run:  WARP_CACHE_PATH=$(mktemp -d) MUJOCO_GL=egl uv run --extra rl --extra gpu \
        python scripts/sweep_thumb_grasp.py [--iterations 60]
"""
from __future__ import annotations
import argparse, itertools, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GEN_DIR = ROOT / "assets/mjcf/experimental/morph_sweep"
OUT_DIR = ROOT / "results/phase1/morph_sweep"
RESULTS = ROOT / "MORPH_GRASP_SWEEP_RESULTS.txt"
BASE_HAND = ROOT / "assets/mjcf/baseline/hand.xml"
BASE_SCENE = ROOT / "assets/mjcf/baseline/scenes/scene_screwdriver_medium_flat_short_proximal.xml"
# index/middle held at the run18 baseline; only the thumb is swept.
BASE_INDEX = (0.010, -0.0123, 0.0)
BASE_MIDDLE = (0.010, 0.0153, 0.0)
# baseline thumb = (0.0, 0.020, 0.0). Sweep around it within the morph joint ranges
# (x,y in [-0.03,0.03], len in [0,0.035]).
TX = [-0.02, -0.01, 0.0, 0.01, 0.02]
TY = [0.0, 0.01, 0.02, 0.03]
TLEN = [0.0, 0.020]


def tok(v: float) -> str:
    return f"{'n' if v < 0 else 'p'}{abs(v):.4f}".replace(".", "d")


def gen_scene(thumb, env) -> Path:
    """Generate the baked scene for a morphology; return the scene xml path."""
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/generate_morphology_xml.py"),
         "--base-hand-xml", str(BASE_HAND), "--base-scene-xml", str(BASE_SCENE),
         "--output-dir", str(GEN_DIR),
         "--thumb", *map(str, thumb),
         "--index", *map(str, BASE_INDEX), "--middle", *map(str, BASE_MIDDLE)],
        check=True, capture_output=True, text=True, env=env, timeout=120)
    enc = (f"t{tok(thumb[0])}_{tok(thumb[1])}_{tok(thumb[2])}_"
           f"i{tok(BASE_INDEX[0])}_{tok(BASE_INDEX[1])}_{tok(BASE_INDEX[2])}_"
           f"m{tok(BASE_MIDDLE[0])}_{tok(BASE_MIDDLE[1])}_{tok(BASE_MIDDLE[2])}")
    matches = list(GEN_DIR.glob(f"scene_{enc}*.xml"))
    if not matches:  # fall back to newest scene
        matches = sorted(GEN_DIR.glob("scene_*.xml"), key=lambda p: p.stat().st_mtime)
    return matches[-1]


def run_cem(scene: Path, tag: str, iters: int, env) -> dict:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/phase1_optimize_grasp.py"),
         "--scene-xml", str(scene), "--keyframe", "open_short_manual",
         "--iterations", str(iters), "--population", "40", "--skip-gif",
         "--output-dir", str(OUT_DIR), "--tag", tag],
        check=True, capture_output=True, text=True, env=env, timeout=600)
    return json.loads((OUT_DIR / tag / "summary.json").read_text())["best_metrics"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=60)
    args = ap.parse_args()
    env = dict(os.environ)
    env.setdefault("MUJOCO_GL", "egl")

    grid = list(itertools.product(TX, TY, TLEN))
    hdr = (f"# thumb-opposition grasp sweep  iters={args.iterations}  {len(grid)} candidates  "
           f"{time.strftime('%Y-%m-%d %H:%M')}\n"
           f"# baseline thumb (0.0,0.020,0.0); index/middle fixed. Want: cube_lift>0 (grasps), "
           f"contacts->3, imbalance LOW, fc_min HIGH (weak finger contributes).\n"
           f"{'tx':>7} {'ty':>7} {'tlen':>6} | {'score':>7} {'lift':>7} {'contacts':>8} "
           f"{'imbal':>7} {'persist_t/i/m':>18} {'fc_min':>7} {'fc_mean':>7}\n")
    RESULTS.write_text(hdr)
    print(hdr, end="")
    rows = []
    for i, (tx, ty, tlen) in enumerate(grid):
        tag = f"t{tok(tx)}_{tok(ty)}_{tok(tlen)}"
        t0 = time.time()
        try:
            scene = gen_scene((tx, ty, tlen), env)
            b = run_cem(scene, tag, args.iterations, env)
            row = dict(tx=tx, ty=ty, tlen=tlen, score=b["score"], lift=b["cube_lift"],
                       contacts=b.get("cube_tip_contacts", float("nan")),
                       imbal=b["finger_persistence_imbalance"],
                       pt=b["thumb_contact_persistence"], pi=b["index_contact_persistence"],
                       pm=b["middle_contact_persistence"],
                       fc_min=b["trajectory_fc_min_fingers"], fc_mean=b["trajectory_fc_mean_fingers"])
            rows.append(row)
            line = (f"{tx:+7.3f} {ty:+7.3f} {tlen:6.3f} | {row['score']:7.3f} {row['lift']:7.4f} "
                    f"{row['contacts']:8.1f} {row['imbal']:7.3f} "
                    f"{row['pt']:5.2f}/{row['pi']:.2f}/{row['pm']:.2f}   "
                    f"{row['fc_min']:7.3f} {row['fc_mean']:7.3f}")
        except Exception as e:
            line = f"{tx:+7.3f} {ty:+7.3f} {tlen:6.3f} | FAIL {type(e).__name__}: {str(e)[:80]}"
        line += f"   ({time.time()-t0:.0f}s, {i+1}/{len(grid)})"
        print(line)
        with RESULTS.open("a") as f:
            f.write(line + "\n")

    # ranking: graspable (lift>0.01) + balanced (low imbal) + weak finger contributes (high fc_min)
    good = [r for r in rows if r["lift"] > 0.01]
    good.sort(key=lambda r: (r["imbal"] - 2.0 * r["fc_min"]))
    foot = "\n# === TOP BALANCED GRASPS (graspable, low imbalance, weak finger engaged) ===\n"
    for r in good[:6]:
        foot += (f"#  thumb=({r['tx']:+.3f},{r['ty']:+.3f},{r['tlen']:.3f})  imbal={r['imbal']:.3f}  "
                 f"fc_min={r['fc_min']:.3f}  persist={r['pt']:.2f}/{r['pi']:.2f}/{r['pm']:.2f}  "
                 f"lift={r['lift']:.3f}\n")
    if not good:
        foot += "#  (none graspable — the sweep range may need widening)\n"
    print(foot, end="")
    with RESULTS.open("a") as f:
        f.write(foot)


if __name__ == "__main__":
    main()
