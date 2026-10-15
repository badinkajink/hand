#!/usr/bin/env python3
"""Verification of the fingertip co-design study (ledger step 24): recomputes the page's headline numbers from the rows
with code independent of tip_codesign_analysis.py, and checks the cached scenes the rows ran on.

  scenes     every cached MuJoCo pad scene of a tcd tip (logs/20261014-tip_codesign/{scenes,scale}/**.xml): each
             <finger>_pad<i> sphere lies where TipShape.pads puts it today (max deviation, m); a scene built with
             displaced pads (the degenerate-facet defect of 2026-10-14) fails
  numbers    the headline numbers, each recomputed from rows and compared with the summary the page reads

Writes docs/experiments/20261014-tip_codesign/verify.json and exits 1 when a check fails.

    .venv/bin/python scripts/tip_codesign_verify.py scenes numbers
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
D = ROOT / "docs/experiments/20261014-tip_codesign"
LG = ROOT / "logs/20261014-tip_codesign"
SIGN = {"thumb": 1.0, "index": -1.0, "middle": -1.0}
KEY = re.compile(r"(tcd_rs[0-9.]+_re[0-9.]+(?:_cr[0-9.]+)?(?:_p[0-9.]+)?)")


def rows(p):
    p = Path(p)
    return [json.loads(l) for l in open(p) if l.strip()] if p.exists() else []


def check_scenes(limit_per_key=None):
    """Pad positions of every cached pad scene against today's builder."""
    import fingertip_geometry as G
    import tip_codesign_geometry as TG
    sign = {f: float(G.FACE_SIGN[f]) for f in ("thumb", "index", "middle")}
    files = [p for p in list((LG / "scenes").rglob("*.xml")) + list((LG / "scale").rglob("*.xml"))
             if "_pads_" in p.name and "superseded" not in str(p)]
    by_key = defaultdict(list)
    for p in files:
        m = KEY.search(str(p.relative_to(LG)))
        if m:
            by_key[m.group(1)].append(p)
    out, worst = {}, 0.0
    for k, ps in sorted(by_key.items()):
        sh = TG.shape(k)
        ref = {f: sh.pads(sign=sign[f], full=True)[0] for f in sign}
        dev, n = 0.0, 0
        for p in ps[:limit_per_key] if limit_per_key else ps:
            try:
                root = ET.parse(p).getroot()
            except ET.ParseError:
                continue
            for g in root.iter("geom"):
                nm = g.get("name") or ""
                mm = re.fullmatch(r"(thumb|index|middle)_pad(\d+)", nm)
                if not mm:
                    continue
                C = ref[mm.group(1)]
                i = int(mm.group(2))
                pos = np.array([float(x) for x in g.get("pos").split()])
                dev = max(dev, float(np.abs(pos - C[i]).max()) if i < len(C) else 1.0)
            n += 1
        out[k] = {"scenes": n, "max_dev_m": dev, "ok": dev < 2e-7}
        worst = max(worst, dev)
        print(f"{k:24s} {n:4d} scenes, max pad deviation {dev:.2e} m {'ok' if dev < 2e-7 else 'FAIL'}", flush=True)
    return {"keys": out, "worst_dev_m": worst, "ok": all(v["ok"] for v in out.values())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checks", nargs="+", choices=["scenes", "numbers"])
    ap.add_argument("--limit", type=int, default=None, help="scenes per key (default all)")
    a = ap.parse_args()
    res = json.loads((D / "verify.json").read_text()) if (D / "verify.json").exists() else {}
    if "scenes" in a.checks:
        res["scenes"] = check_scenes(a.limit)
    if "numbers" in a.checks:
        import tip_codesign_verify_numbers as VN
        res["numbers"] = VN.run()
    res["when"] = time.strftime("%Y-%m-%d %H:%M")
    (D / "verify.json").write_text(json.dumps(res, indent=1, default=float))
    bad = [c for c in a.checks if not res[c]["ok"]]
    print("verify:", "FAIL " + ", ".join(bad) if bad else "all checks pass")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
