#!/usr/bin/env python3
"""Does the open-loop screen's "held" verdict survive a longer hold?

The screen scores a rollout 1.6 s after the turn command finishes (`--hold-steps 800`
at dt=0.002) and calls it held if one fingertip is still touching, the object has not
fallen 20 mm, and it is off the support post.  Watching the renders, the shaft is
usually hanging off the tips at that moment rather than being gripped -- so the
question is whether 1.6 s is a settled state or a snapshot part-way through a fall.

This re-runs each finalist's OWN saved plan at several hold lengths and reports what
changes.  A settled grasp is invariant to the hold length.  A shaft on its way out is
not.

  python3 scripts/probe_hold_convergence.py \
      --plans docs/experiments/20260830-real_v1-sobol128/plans \
      --holds 800,2400,4800 --reps 3
"""
import argparse, json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
ROOT = Path(__file__).resolve().parents[1]


def _job(a):
    import real_v1_deploy_envelope as de
    plan_path, hold, rep, ft, fp = a
    plan = json.loads(Path(plan_path).read_text())
    r = de.execute(Path(plan["scene"]), plan, hold_steps=hold, seed=rep,
                   force_target=ft, force_phase=fp)
    return {"design": Path(plan_path).stem, "hold": hold, "rep": rep,
            "force_target": ft, "force_phase": fp,
            "final_cos": r["final_cos"], "peak_cos": r["peak_cos"],
            "ok": r["ok"], "contacts": r["contacts_hand"],
            "force_N": r["force_hand_N"], "final_z": r["final_z"],
            "lifted_z": r["lifted_z"], "min_z_hold": r["min_z_hold"],
            "on_post": r["on_post"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", type=Path, required=True, help="dir of <design>.json plans")
    ap.add_argument("--holds", default="800,2400,4800")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--force-target", type=float, default=0.0,
                    help="per-finger normal force the hold regulator aims for, N. The screen ran "
                         "at 0.0, i.e. open loop through the hold as well as the turn.")
    ap.add_argument("--force-phase", default="hold",
                    help="all | hold -- whether the regulator acts through the turn too")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    holds = [int(x) for x in a.holds.split(",")]
    jobs = [(str(p), h, r, a.force_target, a.force_phase)
            for p in sorted(a.plans.glob("*.json"))
            for h in holds for r in range(a.reps)]
    print(f"{len(jobs)} rollouts: {len(list(a.plans.glob('*.json')))} plans "
          f"x {len(holds)} holds x {a.reps} reps", flush=True)

    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(_job, jobs))

    designs = sorted({r["design"] for r in rows})
    w = max(len(d) for d in designs) + 1
    print(f"\n{'design':{w}s}" + "".join(f"{'hold ' + str(h) + ' (' + f'{h*0.002:.1f}' + 's)':>26s}"
                                          for h in holds))
    print(" " * w + "".join(f"{'cos / kept / N / dz mm':>26s}" for _ in holds))
    for dz in designs:
        line = f"{dz:{w}s}"
        for h in holds:
            g = [r for r in rows if r["design"] == dz and r["hold"] == h]
            cos = sum(r["final_cos"] for r in g) / len(g)
            kept = sum(1 for r in g if r["ok"])
            fN = sum(r["force_N"] for r in g) / len(g)
            dzm = 1000 * sum(r["lifted_z"] - r["final_z"] for r in g) / len(g)
            line += f"{cos:+7.3f} {kept}/{len(g)} {fN:5.2f} {dzm:6.1f}".rjust(26)
        print(line)

    if a.out:
        a.out.write_text(json.dumps(rows, indent=1))
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
