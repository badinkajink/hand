#!/usr/bin/env python3
"""The operating point of record, per hand and PER MANEUVER, built from the source studies.

WHY THIS FILE EXISTS. `axis_k`, the residual clip and the turn angle are not properties of a
hand. They are properties of a hand RUNNING A PARTICULAR MANEUVER, and this program has three:

    bench   fixed palm, tool standing on a 100 mm post, 9.6 s hold. What the deployed plans
            were fitted on and what the CB1 replays.
    carry   floor-free: the tool is lifted clear and turned in the air, --linear-anchor.
    chain   grasp -> lift -> reorient -> set down -> gait, on the UR5e.

They disagree, and rv05_manual is the proof: its bench band is 0.75-2.00 and a clip of 0.50
holds 0/4, while in the floor-free carry a clip of 0.50 gives cos 0.971 6/6 and 0.85 drops the
tool at every axis height. A single number carried between them is wrong in one of the two.

Everything that reads a clip or a pivot height should read it from here, with the maneuver
named, rather than from a flag typed at the command line. A value this file marks `measured:
false` has never been measured for that maneuver and must not be defaulted to another one's.

    uv run --extra rl python scripts/real_v1_operating_points.py --write
"""
from __future__ import annotations

import argparse, json, statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/experiments/OPERATING_POINTS.json"
BENCH = ROOT / "docs/experiments/20260830-real_v1-budget-rescreen/deploy_plan_bands.json"
DEPLOY = ROOT / "docs/experiments/20260829-real_v1_deploy/deploy"
A_TAGS = ["sv1_w6689_b060", "sv1_w2360_b075", "sv1_u1364_b080", "g12_b095",
          "sv1_u0060_b75", "sv1_u0308_b050", "rv05_manual_b85", "sv1_w0099_b100"]
DID = dict(zip(A_TAGS, "D1 D2 D3 D4 D5 D6 D7 D8".split()))
WEIGHT_N = 0.240   # the screwdriver; any "held" claim is against this


def bench_bands() -> dict:
    """Contiguous clip band per plan, load-tested: >= 2 contacts at >= the tool's weight."""
    rows = json.loads(BENCH.read_text())["rows"]
    by: dict = {}
    for r in rows:
        by.setdefault(r["plan"], {}).setdefault(r["budget"], []).append(r)
    out = {}
    for plan, per in by.items():
        # A MAJORITY, not unanimity. Requiring every rep splits a real band on one unlucky
        # rollout: rv05_manual holds 4/4 from 0.75 to 2.00 except single 3/4 dropouts at 0.80
        # and 1.35, and unanimity reports its band as 1.40-2.00 -- excluding the clip its own
        # plan ships and contradicting the band table this study already published.
        held = {b: sum(1 for x in g if (x.get("contacts") or 0) >= 2
                       and (x.get("force_N") or 0) >= WEIGHT_N) >= 0.75 * len(g)
                for b, g in per.items()}
        full = sorted(b for b, ok in held.items() if ok)
        # The CONTIGUOUS run containing the plan's own clip, not the min/max of all passing
        # values -- g12 passes at 0.40 and again from 0.70, and quoting "0.40-2.00" would put
        # its whole dead zone inside the band.
        band = None
        if full:
            runs, cur = [], [full[0]]
            for a, b in zip(full, full[1:]):
                (cur.append(b) if round(b - a, 6) <= 0.051 else (runs.append(cur),
                                                                 cur := [b]))
            runs.append(cur)
            band = max(runs, key=len)
        out[plan] = {
            "band_rad": [round(band[0], 2), round(band[-1], 2)] if band else None,
            "cos_at_band": {f"{b:.2f}": round(st.mean([x.get("final_cos") or 0.0
                                                       for x in per[b]]), 3)
                            for b in (band or [])},
            "n_per_clip": len(next(iter(per.values()))),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    bb = bench_bands()
    hands = {}
    for tag in A_TAGS:
        meta = json.loads((DEPLOY / f"{tag}_plan.json").read_text())["meta"]
        b = bb.get(tag, {})
        hands[tag] = {
            "id": DID[tag],
            "plan": {k: meta[k] for k in ("straddle_mm", "thumb_axial_mm", "grip_depth_mm",
                                          "squeeze_mm", "axis_k", "angle_deg", "budget_rad")},
            "bench": {
                "measured": b.get("band_rad") is not None,
                "axis_k": meta["axis_k"], "angle_deg": meta["angle_deg"],
                "clip_band_rad": b.get("band_rad"),
                "clip_at_plan_holds": (b.get("band_rad") is not None
                                       and b["band_rad"][0] - 1e-9 <= meta["budget_rad"]
                                       <= b["band_rad"][1] + 1e-9),
                "cos_in_band": b.get("cos_at_band"),
                "source": "docs/experiments/20260830-real_v1-budget-rescreen/"
                          "deploy_plan_bands.json",
            },
            "carry": {"measured": False, "axis_k": None, "clip_rad": None,
                      "note": "floor-free lifted turn; measured only on rv05_manual_stored "
                              "and rv03_narrowy_sp40, neither of which is one of these plans "
                              "at these settings",
                      "source": "docs/experiments/20260906-rv05_band"},
            "chain": {"measured": False, "axis_k": None, "clip_rad": None,
                      "note": "running 2026-09-06, docs/experiments/20260906-chain_band",
                      "source": None},
        }

    doc = {
        "note": ("Per hand AND per maneuver. A clip or pivot height is not a property of a "
                 "hand. Never carry a value between maneuvers; where `measured` is false, "
                 "measure it rather than defaulting to another maneuver's number."),
        "object_weight_N": WEIGHT_N,
        "held_definition": ">= 2 pad contacts at >= object_weight_N, floor-free",
        "sign_convention": ("cos is the tool's own +z against world +z. TIP DOWN is +1, HANDLE "
                            "DOWN is -1. `tilt_deg` is folded to arccos|cos| whenever tip_len "
                            "is 0 and CANNOT tell them apart."),
        "reference_configuration": {
            "run": "results/phase1/real_v1/rv05_manual_stored",
            "what": "the only configuration that has ever run the chain end to end",
            "result": "169/169 on 2026-09-03, reproduced 2026-09-06: cos +0.9965 at "
                      "`reoriented` on 3 pads at 10.47 N, tip down, gaits 8/8",
            "pads": "SPHERE r 10.55 mm (frozen_scene.xml), not the chain's flat pads",
            "grasp": "stored CEM best_finger_ctrl from best_rollout.npz, not _grip_from_fit",
            "axis_k": 0.25, "angle_deg": -90.0, "budget_rad": 0.5, "turn_steps": 550,
            "regression": "tests/test_chain_reference.py",
        },
        "hands": hands,
    }
    print(f"{'id':>3} {'plan':22} {'plan clip':>9} {'bench band':>12} {'holds?':>7}")
    for tag, h in hands.items():
        b = h["bench"]
        print(f"{h['id']:>3} {tag:22} {h['plan']['budget_rad']:9.2f} "
              f"{str(b['clip_band_rad']):>12} {str(b['clip_at_plan_holds']):>7}")
    if args.write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(doc, indent=1))
        print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
