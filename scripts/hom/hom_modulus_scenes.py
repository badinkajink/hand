#!/usr/bin/env python3
"""HOM reference scenes at another pad modulus (2026-11-02; owner: "seems like we need to queue up modulus studies across
a range as well. this is not something we can measure super well ... and also appears to be pretty fingertip-specific").

hom_gpu.py compiles the reference layout's scenes for each object (Scene, REF) and reads them from the cell-scene cache
(HOS_SCENES), so a cache built at modulus E gives pads at E to every grid or search that reads it. The pads' stiffness
E a_i / h_i is set through the relaxation time pad_tr(E) at a fixed solimp d0 (reorient_backends.pad_tr); no other code
on the GPU evaluator's path depends on E. replace_tips binds its default modulus when it is imported, so this script
wraps it with E for the scenes it builds.

The script refuses a cache directory whose scenes record another modulus, builds the reference scenes of the default
objects (the 40 mm sphere and the 25 mm cylinder) at E, and checks every scene's meta in the directory. With --check it
only checks, so a queue can confirm after a run that no scene at another modulus entered the cache.

    HOS_PLANT=servo HOS_SCENES=logs/20261102-hom_servo_plant/scenes_E5 HOM_ROWS_TAG=_svE5 \
        .venv/bin/python scripts/hom/hom_modulus_scenes.py --E-MPa 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("SCALE_HAND", "cad_v2")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path


def metas(scenes: Path):
    """(path, meta) of every cell scene in the cache directory."""
    for p in sorted(scenes.glob("*/*.json")):
        m = json.loads(p.read_text())
        if "pad_E_Pa" in m:
            yield p, m


def check(scenes: Path, E: float) -> int:
    bad = [(p, m["pad_E_Pa"]) for p, m in metas(scenes) if abs(m["pad_E_Pa"] - E) > 1.0]
    for p, e in bad:
        print(f"{p.relative_to(ROOT)}: pads at {e / 1e6:g} MPa, not {E / 1e6:g}")
    n = sum(1 for _ in metas(scenes))
    print(f"{scenes.relative_to(ROOT)}: {n - len(bad)} of {n} scenes at {E / 1e6:g} MPa")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description="the HOM reference scenes at another pad modulus")
    ap.add_argument("--E-MPa", type=float, required=True, help="the pads' modulus")
    ap.add_argument("--objects", nargs="*", default=["sphere:40", "cylinder:25"], help="shape:diameter_mm")
    ap.add_argument("--mu", type=float, default=None, help="the scenes' friction (hand_object_scale_sim.MU, 1.0; 11-02)")
    ap.add_argument("--check", action="store_true", help="only check the cache's scenes")
    a = ap.parse_args()
    if not os.environ.get("HOS_SCENES"):
        raise SystemExit("set HOS_SCENES to the cache directory of this modulus")
    scenes = (ROOT / os.environ["HOS_SCENES"]).resolve()
    E = a.E_MPa * 1e6
    if a.mu is not None:
        bad = [p for p, m in metas(scenes) if abs(m.get("mu", -1) - a.mu) > 1e-9] if scenes.exists() else []
        for p in bad:
            print(f"{p.relative_to(ROOT)}: friction {json.loads(p.read_text()).get('mu')}, not {a.mu:g}")
        if bad:
            return 1
    if a.check or (scenes.exists() and check(scenes, E)):
        return check(scenes, E) if a.check else 1

    if a.mu is not None:
        import hand_object_scale_sim as S          # cell_scenes reads MU when it builds; F_GRIP does not depend on it
        S.MU = a.mu
    import reorient_backends as RB
    RB.E_TPU = E                                     # each scene's meta records pad_E_Pa and pad_tr_s from it
    replace = RB.replace_tips
    RB.replace_tips = lambda *p, **k: replace(*p, **{"E": E, **k})   # cell_scenes passes no modulus
    import hom_gpu as HG                             # noqa: E402  (imports hand_object_scale_sim with HOS_* set)
    for o in a.objects:
        shape, d = o.split(":")
        sc = HG.Scene(shape, float(d))
        root = ET.parse(sc.pads_path).getroot()
        tcs = sorted({float(g.get("solref").split()[0]) for g in root.iter("geom")
                      if g.get("name", "").startswith(tuple(f"{f}_pad" for f in HG.FINGERS)) and g.get("solref")})
        m = json.loads(sc.pads_path.with_suffix(".json").read_text())
        print(f"{shape} {d} mm: {sc.pads_path.relative_to(ROOT)}  plant {m['plant']}  pads at {m['pad_E_Pa'] / 1e6:g} MPa, "
              f"pad_tr {m['pad_tr_s']:.4f} s (pad solref time constants {tcs[:1] + tcs[-1:] if tcs else 'none found'})")
    return check(scenes, E)


if __name__ == "__main__":
    sys.exit(main())
