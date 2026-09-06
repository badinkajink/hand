"""Pin the ONE configuration that has ever run the chain end to end.

`rv05_manual_stored` chained 169/169 on 2026-09-03. By 2026-09-06 the configuration would not
execute at all -- `GantryPalm` had no `worst_pos`, which `probe_real_v1_chain` reads
unconditionally when `reindex` is "full" -- and nothing noticed, because every run in between
went through the arm path. Days of work were then spent on maneuvers that had drifted away from
it with no baseline to diff against.

This asserts the SIGNED cosine. `tilt_deg` folds to arccos|cos| whenever `tip_len` is 0, so a
tool standing on its handle reads as a perfect 0.00 deg stand; 111 of 123 "stands" in the
2026-09-06 table-stand sweeps were handle down and scored as successes.

Slow (~40 s). Run it before and after touching probe_real_v1_chain, palm_driver, the scene
generators or the fitters:

    uv run --extra rl --extra arm python -m pytest tests/test_chain_reference.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

RUN = ROOT / "results/phase1/real_v1/rv05_manual_stored"
# The 2026-09-03 cell, verbatim.
CELL = dict(obj="screwdriver_medium", lift=0.10, angle_deg=-90.0, axis_k=0.25, turn_steps=550,
            budget=0.5, hold_steps=500, gap=0.002, press_mm=2.0, grip_depth=0.050,
            squeeze=0.002, reindex="full", stand_order="ground", airgrip="cradle", cycles=8)


@pytest.mark.skipif(not (RUN / "best_rollout.npz").exists(),
                    reason="rv05_manual_stored is not on this machine")
def test_reference_chain_reorients_tip_down_and_gaits():
    import probe_real_v1_chain as C
    r = C.chain(RUN, seed=0, jitter=0.0005, **CELL)
    seam = {s["phase"]: s for s in r["seams"]}
    ro = seam["reoriented"]
    # Tip DOWN, in the air, on a load-bearing grip. cos is signed; tilt_deg is not.
    assert ro["cos"] > 0.90, f"reoriented cos {ro['cos']} -- +1 is tip down, -1 is handle down"
    assert ro["pad_contacts"] >= 2, f"reoriented on {ro['pad_contacts']} pads"
    assert ro["pad_force_N"] >= 0.240, f"{ro['pad_force_N']} N below the tool's 0.240 N weight"
    assert ro["z"] > 0.08, f"reoriented at z {ro['z']} -- not floor-free"
    assert seam["upright"]["cos"] > 0.90, f"upright cos {seam['upright']['cos']}"
    assert r["ok"], "the reference chain did not complete"
    assert r["cycles_run"] == 8, f"gaited {r['cycles_run']}/8 cycles"
