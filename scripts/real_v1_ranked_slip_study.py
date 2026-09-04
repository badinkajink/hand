"""Does the reorientation's residual tilt behave the same way on the hands the paper ranks?

The 2026-09-04 bench study measured the turn's end tilt on ONE hand, `rv05_manual`, and reported
8.03 deg with a 0.42 deg seed spread, a settle that makes it worse, and a squeeze that destroys
it. A first replication went out over `results/phase1/real_v1/rv0*`, which is the wrong
population: that is the five-design screening family, not the Sobol sample the design study ranks.

THE POPULATION IS `20260831-real_v1-sobol8192` -- 8,192 sampled, 535 selected, 227 confirmed,
8 promoted with exported plans -- and every hand runs at ITS OWN operating point: its straddle,
thumb-axial offset, fitted grip depth and residual clip, read from the confirmed design table and
`deploy/promotion.json`. Nothing is retuned and nothing is borrowed from rv05_manual.

THE MANEUVER IS THE ONE THEY WERE RANKED ON, through the screen's own code:
`real_v1_deploy_envelope.make_plan` + `.execute` on the bench scene each plan names -- tool
standing on a 100 mm platform, palm fixed, close -> settle -> turn -> hold, with the retention
config of `real_v1_sobol8192.sh` stage D (load regulator at 250 units, the two servo torque
ceilings, the 60 mm proof lift). Rebuilding the maneuver out of `probe_real_v1_carry` on a table
with a palm lift reproduced none of it -- every hand read 78-89 deg of residual tilt -- which
measures the substitution, not the hand. `validate` exists so that stays visible.

  validate  the screened cell, hold 2500              does it reproduce the confirm's nom_cos?
  settle    hold 0..2500 steps                        is the turn's tilt a property of the HAND?
  squeeze   hold-squeeze 0..2 mm                      does re-gripping at the top control it?
  open      the load regulator off                    is the settle the REGULATOR's doing?

    uv run --extra rl python scripts/real_v1_ranked_slip_study.py \
        --out docs/experiments/20260904-real_v1_ranked
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics as st
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

POP = ROOT / "docs/experiments/20260831-real_v1-sobol8192"
PLANS = ROOT / "assets/mjcf/experimental/20260904-ranked_slip"
# `real_v1_sobol8192.sh` stage D, verbatim. The hold-phase load regulator is part of the
# deployed maneuver, not an extra: the screen's own comment is that closing the loop through
# the turn "holds the shaft better and turns it worse".
RETENTION = dict(load_target_units=250.0, load_gain=0.0024, capture_steps=400,
                 proof_lift=0.060, proof_lift_steps=700, proof_max_slip=0.010,
                 turn_torque_limit_nm=0.18044236, hold_torque_limit_nm=0.06864655)
CELL = dict(axis_k=0.15, angle_deg=-80.0, turn_steps=550)
HOLDS = (0, 150, 300, 500, 900, 2500)
SQZ = (0.0005, 0.0010, 0.0020)
SCREENED_HOLD = 2500


def hands(promotion: Path, table: Path) -> list[dict]:
    """One record per promoted tag: its bench scene and the cell it was screened at."""
    rows = {r["design"]: r for r in json.loads(table.read_text())}
    out = []
    for r in json.loads(promotion.read_text()):
        t, scene = rows.get(r["design"]), Path(r["scene"])
        if t is None or not scene.exists():
            continue
        # `--bench-height > 0` makes the screen substitute the fitted depth for the requested
        # one; without it `fit` searches a palm window that does not contain a reachable pose
        # and every design reports "no pose".
        depth = t.get("depth_req_mm") or t.get("depth_fit_mm")
        out.append({"tag": r["tag"], "design": r["design"], "scene": str(scene),
                    "straddle": t["straddle_mm"] / 1000, "thumb_axial": t["thumb_axial_mm"] / 1000,
                    "squeeze": t.get("squeeze_mm", 10.0) / 1000, "depth": depth / 1000,
                    "budget": r["budget_rad"], "verdict": r["verdict"]})
    return out


def screen_cos(design: str, budget: float) -> tuple[float, float] | tuple[None, None]:
    """What the confirmation itself scored this design AT THIS CLIP, for the validation arm.

    Keyed on the budget as well as the design: two of the promoted tags are the same hand at a
    different clip, and matching on the design alone compares this replay against a cell it was
    not run at (which read as a 8 deg disagreement that was not one).
    """
    for f in sorted(glob.glob(str(POP / "confirm_b*.json"))):
        for r in json.loads(Path(f).read_text()):
            if r["design"] == design and abs(r["budget_rad"] - budget) < 1e-9:
                return r["nom_cos"], r["nom_sd"]
    return None, None


def plan_for(h: dict, hold_squeeze: float, cache: Path) -> Path | None:
    """The deployed plan, from the screen's own planner. One per (hand, hold-squeeze)."""
    import real_v1_deploy_envelope as de
    cache.mkdir(parents=True, exist_ok=True)
    f = cache / f"{h['tag']}_q{hold_squeeze * 1000:.1f}_plan.json"
    if f.exists():
        return f
    pl = de.make_plan(Path(h["scene"]), straddle=h["straddle"], depth=h["depth"],
                      thumb_axial=h["thumb_axial"], squeeze=h["squeeze"],
                      lift=0.10, budget=h["budget"], hold_squeeze=hold_squeeze,
                      bench=True, **CELL)
    if pl is None:
        return None
    f.write_text(json.dumps(pl))
    return f


def _cell(kw):
    import real_v1_deploy_envelope as de
    tag, h, q = kw.pop("_tag"), kw.pop("_hand"), kw.pop("_q")
    plan = json.loads(Path(kw.pop("_plan")).read_text())
    cfg = dict(RETENTION, **kw.pop("_cfg", {}))
    try:
        r = de.execute(Path(h["scene"]), plan, selfcollision=True, **cfg, **kw)
    except Exception as exc:
        return {"arm": tag, "tag": h["tag"], "design": h["design"], "hold_squeeze_mm": q * 1000,
                "error": repr(exc), "ok": False,
                **{k: v for k, v in kw.items() if isinstance(v, (int, float))}}
    r.pop("load_units", None)
    r["arm"], r["tag"], r["design"], r["verdict"] = tag, h["tag"], h["design"], h["verdict"]
    r["hold_squeeze_mm"] = q * 1000
    r["regulated"] = cfg["load_target_units"] > 0.0
    # A shaft the hand is no longer touching reads whatever tilt gravity left it at, and one
    # sitting back down on its own post reads vertical, so every tilt is gated on the hand still
    # carrying it clear at the last commanded turn step.
    r["held_turn"] = bool(r.get("turn_contacts_hand", 0) >= 1
                          and r.get("turn_z", 0.0) > r.get("lifted_z", 0.0) - 0.02)
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--promotion", type=Path, default=POP / "deploy/promotion.json")
    ap.add_argument("--table", type=Path,
                    default=POP / "selected/confirmed/selected_table.json")
    ap.add_argument("--plans", type=Path, default=PLANS)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--arms", default="validate,settle,squeeze,open")
    args = ap.parse_args()
    arms = set(args.arms.split(","))

    H = hands(args.promotion, args.table)
    print(f"{len(H)} ranked hands: " + ", ".join(h["tag"] for h in H), flush=True)
    jobs, skipped = [], []
    for h in H:
        base = plan_for(h, 0.0, args.plans)
        if base is None:
            skipped.append({"tag": h["tag"], "why": "no pose"})
            print(f"  {h['tag']}: NO POSE", flush=True)
            continue
        for rep in range(args.reps):
            if "validate" in arms:
                jobs.append({"_tag": "validate", "_hand": h, "_plan": str(base), "_q": 0.0,
                             "hold_steps": SCREENED_HOLD, "jitter": 0.0005, "seed": rep})
            if "settle" in arms:
                for hs in HOLDS:
                    jobs.append({"_tag": "settle", "_hand": h, "_plan": str(base), "_q": 0.0,
                                 "hold_steps": hs, "jitter": 0.0005, "seed": rep})
            if "open" in arms:
                for hs in HOLDS:
                    jobs.append({"_tag": "open", "_hand": h, "_plan": str(base), "_q": 0.0,
                                 "_cfg": {"load_target_units": 0.0}, "hold_steps": hs,
                                 "jitter": 0.0005, "seed": rep})
        if "squeeze" in arms:
            for q in SQZ:
                f = plan_for(h, q, args.plans)
                if f is None:
                    continue
                for rep in range(args.reps):
                    jobs.append({"_tag": "squeeze", "_hand": h, "_plan": str(f), "_q": q,
                                 "hold_steps": 300, "jitter": 0.0005, "seed": rep})

    print(f"{len(jobs)} cells on {args.workers} workers", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(_cell, jobs, chunksize=1)):
            rows.append(r)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(jobs)}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    dst = args.out / "ranked_slip.json"
    dst.write_text(json.dumps({"rows": rows, "skipped": skipped, "hands": H,
                               "retention": RETENTION, "cell": CELL},
                              separators=(",", ":")))
    print(f"-> {dst}")

    m = lambda x: st.mean(x) if x else float("nan")
    sd = lambda x: st.pstdev(x) if len(x) > 1 else float("nan")

    if "validate" in arms:
        print("\n== validate: this replay vs the confirmation that ranked the hand")
        print(f"   {'tag':20} {'screen cos':>11} {'sd':>6} {'replay cos':>11} {'sd':>6} "
              f"{'screen tilt':>12} {'replay tilt':>12}")
        import math
        for h in H:
            g = [r for r in rows if r["arm"] == "validate" and r["tag"] == h["tag"]]
            if not g:
                continue
            sc, ssd = screen_cos(h["design"], h["budget"])
            # nom_cos is the HELD mean: a failed trial counts as 0, not as its tilt.
            held = [abs(r["final_cos"]) if r.get("ok") else 0.0 for r in g]
            print(f"   {h['tag']:20} {(sc if sc is not None else float('nan')):11.3f} "
                  f"{(ssd if ssd is not None else float('nan')):6.3f} {m(held):11.3f} "
                  f"{sd(held):6.3f} "
                  f"{(math.degrees(math.acos(min(1, sc))) if sc else float('nan')):12.1f} "
                  f"{(math.degrees(math.acos(min(1, m(held)))) if m(held) > 0 else float('nan')):12.1f}")

    for arm in ("settle", "open", "squeeze"):
        sub = [r for r in rows if r.get("arm") == arm]
        if not sub:
            continue
        var = "hold_squeeze_mm" if arm == "squeeze" else "hold_steps"
        print(f"\n== {arm}  ({len(sub)} cells)   held = carrying it clear at the last turn step")
        print(f"   {'tag':20} {var:>16} {'held':>6} {'turn tilt':>10} {'sd':>6} "
              f"{'final':>7} {'sd':>6} {'ok':>6} {'F_N':>6}")
        for tg in sorted({r["tag"] for r in sub}):
            for v in sorted({r.get(var, 0.0) or 0.0 for r in sub if r["tag"] == tg}):
                g = [r for r in sub if r["tag"] == tg and (r.get(var, 0.0) or 0.0) == v]
                hh = [r for r in g if r.get("held_turn")]
                f = lambda k: [float(r.get(k) or 0.0) for r in hh]
                print(f"   {tg:20} {v:16g} {len(hh):3}/{len(g):<2} "
                      f"{m(f('turn_tilt_deg')):10.2f} {sd(f('turn_tilt_deg')):6.2f} "
                      f"{m(f('final_tilt_deg')):7.2f} {sd(f('final_tilt_deg')):6.2f} "
                      f"{sum(1 for r in g if r.get('ok')):3}/{len(g):<2} "
                      f"{m(f('turn_force_hand_N')):6.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
