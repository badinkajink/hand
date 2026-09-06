#!/usr/bin/env python3
"""Why a plan that reorients 4/4 in the deploy screen picks up nothing in the chain.

Every deployed plan records `grip_depth_mm` and `squeeze_mm` as a PAIR the fitter produced
together: squeeze sets the radial closure (pad centres at r_obj + r_pad + gap - squeeze from
the shaft axis) and depth sets the palm height above the object centre.

`real_v1_plan_band.py` passes both, and reproduces the bench. `real_v1_chain_hands.py` passes
the plan's depth but overrides squeeze 10.0 -> 2.0, because 10 mm ejects a shaft that is lying
on a table with nothing behind it. `_grip_from_fit` clamps its palm-height search to
[depth - 8 mm, depth], so depth is NOT re-fitted for the new squeeze and the pads land 8 mm
further out with nothing compensating.

Three grasps per hand, scored at closure with the palm stationary:

    plan    depth = plan's, squeeze = plan's (10 mm)   -- what the screen runs
    chain   depth = plan's, squeeze = 2 mm             -- what the chain runs
    refit   depth free,     squeeze = 2 mm             -- the proposed fix

    uv run --extra rl --extra arm python scripts/real_v1_grasp_refit.py --out <json>
"""
from __future__ import annotations

import argparse, json, sys, xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import real_v1_chain_hands as ch  # noqa: E402

OBJ = ch.BASE["obj"]
OUTS = ROOT / "assets/mjcf/experimental/20260906-grasp_refit"


def flat_scene(h: dict) -> Path:
    from morphohand.studies.scene_mutate import Scene
    OUTS.mkdir(parents=True, exist_ok=True)
    p = OUTS / f"{h['tag']}__flat.xml"
    if not p.exists():
        sc = Scene(Path(h["base"]))
        sc.set_finger_flat_pads(**ch.PAD)
        sc.write(p)
    return p


def one(h: dict, scene: Path, depth, squeeze_mm: float) -> dict:
    import probe_real_v1_carry as pc
    built = pc._grip_from_fit(scene, h["straddle"], 0.0, squeeze_mm / 1000.0, OBJ,
                              depth, h["thumb_axial"])
    if built is None:
        return {"fit": False}
    m, open_qpos, grip, depth_mm = built
    dz, npad, nonpad, fpad = ch._close_probe(m, open_qpos, grip, OBJ)
    return {"fit": True, "depth_mm": round(float(depth_mm), 1),
            "close_dz_mm": round(float(dz), 1), "pads": int(npad),
            "nonpad": int(nonpad), "pad_N": round(float(fpad), 2)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=ROOT / "docs/experiments/20260906-rv05_band/grasp_refit.json")
    a = ap.parse_args()
    rows = []
    hdr = f"{'tag':<18} {'arm':<6} {'depth':>6} {'dz':>6} {'pads':>5} {'nonpad':>7} {'pad_N':>8}"
    print(hdr); print("-" * len(hdr), flush=True)
    for h in ch.hands("A"):
        sc = flat_scene(h)
        for name, depth, sq in (("plan", h["depth"], h["plan_squeeze_mm"]),
                                ("chain", h["depth"], ch.SQUEEZE_MM),
                                ("refit", None, ch.SQUEEZE_MM)):
            r = one(h, sc, depth, sq)
            r.update(tag=h["tag"], arm=name, squeeze_mm=sq,
                     plan_depth_mm=round(1000 * h["depth"], 1))
            rows.append(r)
            if r["fit"]:
                print(f"{h['tag']:<18} {name:<6} {r['depth_mm']:>6.1f} {r['close_dz_mm']:>6.1f} "
                      f"{r['pads']:>5d} {r['nonpad']:>7d} {r['pad_N']:>8.2f}", flush=True)
            else:
                print(f"{h['tag']:<18} {name:<6} {'NO REACHABLE POSE':>36}", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rows, indent=1) + "\n")
    print(f"\n{len(rows)} fits -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
