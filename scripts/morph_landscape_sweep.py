"""Morphology LANDSCAPE characterization (2026-06-23) — the original project goal.

For a diverse set of hand morphologies (9-param: per-finger x/y/len), measure how well the WHOLE
task is performed, decomposed to avoid the (separate) handoff-control confound:
  - GRASP/LIFT half:  CEM grasp optimizer (phase1_optimize_grasp) -> graspability + per-finger
    contact balance (the A-half: can this hand grasp+lift the flat object?).
  - REORIENT half:    a skip-lift reorienter trained ON that grasp (warmstart B4) -> held-cos,
    per-finger force + balance (the B-half: can this hand reorient in-hand?).
The question: does task quality VARY across the morphology landscape, or can they all do it equally?
If they're equivalent -> the lever is control synthesis; if not -> morphology matters and we have a
map of the good region.

Robust for an unattended (possibly sleeping) machine:
  - per-morphology checkpoint to MORPH_LANDSCAPE.json (RESUMABLE — re-running skips done ones);
  - human-readable rows streamed to MORPH_LANDSCAPE.txt;
  - try/except per morphology (one failure never sinks the sweep);
  - CEM ~28 s; skip-lift B ~25 min (15 M ts, warmstart B4, own watchdog). Sequential = simplest +
    sleep-safe. (Git commits are done by the session when re-invoked, NOT here — the checkout is
    shared with another session, so concurrent commits are unsafe.)

Run: WARP_CACHE_PATH=$(mktemp -d) MUJOCO_GL=egl uv run --extra rl --extra gpu \
       python scripts/morph_landscape_sweep.py [--b-timesteps 15000000] [--n-lhs 10]
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "assets/mjcf/experimental/landscape"
CEM_OUT = ROOT / "results/phase1/landscape"
TXT = ROOT / "MORPH_LANDSCAPE.txt"
JSON = ROOT / "MORPH_LANDSCAPE.json"
BASE_HAND = ROOT / "assets/mjcf/hand.xml"
BASE_SCENE = ROOT / "assets/mjcf/scene_screwdriver_medium_flat_short_proximal.xml"
B4 = ROOT / "results/rl/b04_20260603-1746-policyB_p2_lateral_only/tensorboard/model_541.pt"
# bounds: x,y in [-0.025,0.025], len in [0,0.030]  (morph-joint ranges are +/-0.03 / [0,0.035])
BND = [(-0.025, 0.025), (-0.025, 0.025), (0.0, 0.030)] * 3  # t(x,y,len) i(...) m(...)


def tok(v):  # generate_morphology_xml filename token, no separators within a finger
    return f"{'n' if v < 0 else 'p'}{abs(v):.4f}".replace(".", "d")


def lhs(n, bounds, seed=0):
    rng = np.random.default_rng(seed)
    out = np.zeros((n, len(bounds)))
    for j, (lo, hi) in enumerate(bounds):
        u = (rng.permutation(n) + rng.random(n)) / n
        out[:, j] = lo + u * (hi - lo)
    return out


def morphologies(n_lhs):
    base = (0.0, 0.020, 0.0, 0.010, -0.0123, 0.0, 0.010, 0.0153, 0.0)
    winner = (0.020, 0.020, 0.020, 0.010, -0.0123, 0.0, 0.010, 0.0153, 0.0)
    items = [("m00_baseline", base), ("m01_thumbWinner", winner)]
    for k, row in enumerate(lhs(n_lhs, BND, seed=0)):
        items.append((f"m{k+2:02d}_lhs", tuple(round(float(x), 4) for x in row)))
    return items


def gen_scene(m, env):
    GEN.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/generate_morphology_xml.py"),
                    "--base-hand-xml", str(BASE_HAND), "--base-scene-xml", str(BASE_SCENE),
                    "--output-dir", str(GEN),
                    "--thumb", *map(str, m[0:3]), "--index", *map(str, m[3:6]),
                    "--middle", *map(str, m[6:9])],
                   check=True, capture_output=True, text=True, env=env, timeout=120)
    enc = "scene_t" + "".join(tok(v) for v in m[0:3]) + "_i" + "".join(tok(v) for v in m[3:6]) \
          + "_m" + "".join(tok(v) for v in m[6:9])
    g = list(GEN.glob(enc + "*.xml")) or sorted(GEN.glob("scene_*.xml"), key=lambda p: p.stat().st_mtime)
    return g[-1]


def run_cem(scene, tag, env, iters=60):
    subprocess.run([sys.executable, str(ROOT / "scripts/phase1_optimize_grasp.py"),
                    "--scene-xml", str(scene), "--keyframe", "open_short_manual",
                    "--iterations", str(iters), "--population", "40", "--skip-gif",
                    "--output-dir", str(CEM_OUT), "--tag", tag],
                   check=True, capture_output=True, text=True, env=env, timeout=600)
    return CEM_OUT / tag, json.loads((CEM_OUT / tag / "summary.json").read_text())["best_metrics"]


def train_B(cem_dir, tag, ts, env):
    e = dict(env)
    e.update(MORPH_RUN=str(cem_dir), TAG=tag, TOTAL_TS=str(ts),
             LOG=str(ROOT / f"landscapeB_{tag}.trainer.log"))
    subprocess.run(["bash", str(ROOT / "scripts/train_reorient_on_morph.sh")],
                   check=False, capture_output=True, text=True, env=e, timeout=7200)
    rl = sorted((ROOT / "results/rl").glob(f"*{tag}*"), key=lambda p: p.stat().st_mtime)
    runlog = ROOT / f"landscapeB_{tag}.trainer.log"
    aborted = (Path(str(runlog) + ".COLLAPSED")).exists()
    return (rl[-1] if rl else None), aborted


def probe_B(b_dir, env):
    ck = sorted((b_dir / "tensorboard").glob("model_*.pt"), key=lambda p: p.stat().st_mtime)
    if not ck:
        return None
    out = subprocess.run([sys.executable, str(ROOT / "scripts/probe_grip_balance.py"),
                          f"x={b_dir}:{ck[-1].name}"],
                         capture_output=True, text=True, env=env, timeout=600)
    for ln in out.stdout.splitlines():
        if ln.strip().startswith("x "):
            p = ln.split()
            # x  held_cos | thumbF indexF midF | thumb% | contact-found: t i m
            try:
                return dict(held_cos=float(p[1]), thumbF=float(p[3]), indexF=float(p[4]),
                            midF=float(p[5]), found_t=float(p[-3]), found_i=float(p[-2]),
                            found_m=float(p[-1]))
            except Exception:
                return {"raw": ln.strip()}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b-timesteps", type=int, default=15_000_000)
    ap.add_argument("--n-lhs", type=int, default=10)
    ap.add_argument("--lift-thresh", type=float, default=0.02)
    args = ap.parse_args()
    env = dict(os.environ); env.setdefault("MUJOCO_GL", "egl")

    done = {}
    if JSON.exists():
        done = {d["id"]: d for d in json.loads(JSON.read_text())}
    items = morphologies(args.n_lhs)
    if not TXT.exists():
        TXT.write_text(f"# morphology landscape  {time.strftime('%Y-%m-%d %H:%M')}  "
                       f"B={args.b_timesteps} ts  {len(items)} morphologies\n"
                       f"# GRASP: lift>{args.lift_thresh}=graspable, imbal LOW, persist t/i/m.  "
                       f"REORIENT(skip-lift B, warmstart B4): cos, per-finger force, contact t/i/m.\n"
                       f"{'id':18} | {'lift':>6} {'gImbal':>6} {'g_persist':>14} | "
                       f"{'r_cos':>6} {'r_force t/i/m':>20} {'r_contact':>14}\n")
    results = list(done.values())
    for mid, m in items:
        if mid in done:
            print(f"[skip] {mid} (done)"); continue
        t0 = time.time(); rec = {"id": mid, "morph": list(m)}
        try:
            scene = gen_scene(m, env)
            cdir, g = run_cem(scene, f"{mid}_cem", env)
            rec.update(lift=g["cube_lift"], g_imbal=g["finger_persistence_imbalance"],
                       g_pt=g["thumb_contact_persistence"], g_pi=g["index_contact_persistence"],
                       g_pm=g["middle_contact_persistence"], g_contacts=g.get("cube_tip_contacts"))
            if g["cube_lift"] < args.lift_thresh:
                rec["note"] = "ungraspable (skipped B)"
            else:
                bdir, aborted = train_B(cdir, f"landscape_B_{mid}", args.b_timesteps, env)
                if aborted or bdir is None:
                    rec["note"] = "B aborted (dropped)"
                else:
                    r = probe_B(bdir, env) or {}
                    rec["reorient"] = r
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {str(e)[:120]}"
        rec["secs"] = round(time.time() - t0)
        results.append(rec)
        JSON.write_text(json.dumps(results, indent=1))
        r = rec.get("reorient") or {}
        line = (f"{mid:18} | {rec.get('lift', float('nan')):6.3f} "
                f"{rec.get('g_imbal', float('nan')):6.3f} "
                f"{rec.get('g_pt', 0):.2f}/{rec.get('g_pi', 0):.2f}/{rec.get('g_pm', 0):.2f}".ljust(14)
                + " | " + (f"{r.get('held_cos', float('nan')):6.3f} "
                          f"{r.get('thumbF', 0):.1f}/{r.get('indexF', 0):.1f}/{r.get('midF', 0):.1f}".ljust(20)
                          + f" {r.get('found_t', 0):.2f}/{r.get('found_i', 0):.2f}/{r.get('found_m', 0):.2f}"
                          if r and "held_cos" in r else f"  {rec.get('note', rec.get('error', '—'))}"))
        line += f"   ({rec['secs']}s)"
        print(line)
        with TXT.open("a") as f:
            f.write(line + "\n")
    print(f"[landscape] complete — {len(results)} morphologies -> {TXT}")


if __name__ == "__main__":
    main()
