"""One video per ranked hand, at its own operating point, in the screened maneuver."""
from __future__ import annotations
import json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import real_v1_ranked_slip_study as S

OUT = ROOT / "docs/experiments/20260904-real_v1_ranked/videos"


def _one(job):
    import real_v1_deploy_envelope as de
    h, hold = job
    OUT.mkdir(parents=True, exist_ok=True)
    plan = json.loads((S.PLANS / f"{h['tag']}_q0.0_plan.json").read_text())
    v = OUT / f"20260904-{h['tag']}_hold{hold}.mp4"
    r = de.execute(Path(h["scene"]), plan, hold_steps=hold, seed=0, jitter=0.0005,
                   selfcollision=True, video=v, **S.RETENTION)
    return (f"{h['tag']:20} hold {hold:4} turn {r['turn_tilt_deg']:6.2f} -> "
            f"final {r['final_tilt_deg']:6.2f}  settle {r['settle_deg']:+6.2f}  "
            f"ok={r['ok']}  {v.name}")


def main() -> int:
    H = S.hands(S.POP / "deploy/promotion.json",
                S.POP / "selected/confirmed/selected_table.json")
    for h in H:
        S.plan_for(h, 0.0, S.PLANS)
    jobs = [(h, 2500) for h in H]
    with ProcessPoolExecutor(max_workers=4) as ex:
        for line in ex.map(_one, jobs):
            print(line, flush=True)
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
