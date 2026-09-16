#!/usr/bin/env python3
"""Films, pose snapshots and the contact-model sensitivity for the basketball page.

Reads the sweep's rows.jsonl to pick the cases (the lowest friction that holds on the wide
layout / size 7 under the measured servo, and one step lower), so the figures always show
the rows the tables report. Run after the sweep:

    MUJOCO_GL=egl uv run python scripts/real_v1_basketball_media.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import real_v1_basketball_grasp as G  # noqa: E402

ROOT = G.ROOT
SWEEP = ROOT / "docs/experiments/20260916-basketball/sweep"
MEDIA = ROOT / "docs/experiments/20260916-basketball/media"
MUS = [0.6, 0.8, 1.0, 1.5, 2.4]


def main() -> int:
    MEDIA.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(SWEEP / "rows.jsonl")]
    fits = json.load(open(SWEEP / "fits.json"))

    def scene_for(layout, size):
        sz = G.BALL_SIZES[size]
        return G.make_scene(layout, G.DESIGNS[layout], sz["radius"], sz["mass"], SWEEP / "scenes")

    def fit_for(layout, size, th):
        f = fits[f"{layout}|{size}|{th:g}|0.01"]
        return f

    # --- held / dropped pair on wide, size 7, measured servo saturated ----------------------
    meas = [r for r in rows if r["design"] == "wide" and r["size"] == "7" and r["plant"] == "measured"
            and r["overdrive"] == 8.0]
    held = [r for r in meas if r["held"]]
    scene7 = scene_for("wide", "7")
    if held:
        best = min(held, key=lambda r: (r["mu"], -r["theta_deg"]))
        th, mu = best["theta_deg"], best["mu"]
        lower = max([m for m in MUS if m < mu], default=None)
        f = fit_for("wide", "7", th)
        r1 = G.film(scene7, f, G.PLANTS["measured"], mu, MEDIA / "film_held.png",
                    f"wide size7 th{th:.0f} measured saturated mu{mu:g}", overdrive=8.0)
        print("film_held", r1["held"], r1["rise_mm"])
        if lower is not None:
            r2 = G.film(scene7, f, G.PLANTS["measured"], lower, MEDIA / "film_dropped.png",
                        f"wide size7 th{th:.0f} measured saturated mu{lower:g}", overdrive=8.0)
            print("film_dropped", r2["held"], r2["rise_mm"])
    else:
        th = max(t for t in (40, 45, 50, 55, 60, 65, 70, 75) if fits.get(f"wide|7|{t}|0.01"))
        f = fit_for("wide", "7", th)
        r2 = G.film(scene7, f, G.PLANTS["measured"], 2.4, MEDIA / "film_dropped.png",
                    f"wide size7 th{th:.0f} measured saturated mu2.4", overdrive=8.0)
        print("film_dropped", r2["held"], r2["rise_mm"])

    # --- pose snapshots -------------------------------------------------------------------
    th7 = max(t for t in (40, 45, 50, 55, 60, 65, 70, 75) if fits.get(f"wide|7|{t}|0.01"))
    f7 = fit_for("wide", "7", th7)
    G.inject_keyframe(scene7, "open_ik", " ".join(f"{v:.6g}" for v in f7["q_touch"]),
                      " ".join(f"{v:.6g}" for v in f7["grip_ctrl"]))
    G.inject_keyframe(scene7, "open", " ".join(f"{v:.6g}" for v in f7["q_open"]),
                      " ".join(f"{v:.6g}" for v in f7["open_ctrl"]))
    subprocess.run([sys.executable, str(HERE / "mj_snap.py"), "--scene", str(scene7), "--keyframe", "open_ik",
                    "--views", "front,side,top,iso", "--distance", "0.6",
                    "--out", str(MEDIA / "pose_wide_s7.png")], check=True, capture_output=True)
    print("pose_wide_s7 at theta", th7)

    ths3 = [t for t in (40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100) if fits.get(f"wide|3|{t}|0.01")]
    if ths3:
        scene3 = scene_for("wide", "3")
        f3 = fit_for("wide", "3", max(ths3))
        G.inject_keyframe(scene3, "open_ik", " ".join(f"{v:.6g}" for v in f3["q_touch"]),
                          " ".join(f"{v:.6g}" for v in f3["grip_ctrl"]))
        subprocess.run([sys.executable, str(HERE / "mj_snap.py"), "--scene", str(scene3), "--keyframe", "open_ik",
                        "--views", "front,side,top,iso", "--distance", "0.5",
                        "--out", str(MEDIA / "pose_wide_s3.png")], check=True, capture_output=True)
        h3 = [r for r in rows if r["design"] == "wide" and r["size"] == "3" and r["plant"] == "measured"
              and r["overdrive"] == 8.0 and r["held"]]
        if h3:
            b3 = min(h3, key=lambda r: (r["mu"], -r["theta_deg"]))
            r3 = G.film(scene3, fit_for("wide", "3", b3["theta_deg"]), G.PLANTS["measured"], b3["mu"],
                        MEDIA / "film_size3.png", f"wide size3 th{b3['theta_deg']:.0f} measured saturated mu{b3['mu']:g}",
                        overdrive=8.0)
            print("film_size3", r3["held"], r3["rise_mm"])

    # --- contact-model sensitivity ----------------------------------------------------------
    cases = [("shipped mu 2.4", "shipped", 2.4, 1.0), ("shipped mu 1.0", "shipped", 1.0, 1.0),
             ("measured saturated mu 2.4", "measured", 2.4, 8.0)]
    if held:
        cases.append((f"measured saturated mu {best['mu']:g}", "measured", best["mu"], 8.0))
    out = {"theta": th7, "labels": [c[0] for c in cases], "rows": []}
    for label, plant, mu, od in cases:
        for cname in ("default", "grasp"):
            p = G.probe(scene7, f7, G.PLANTS[plant], mu, contact=G.CONTACT[cname], overdrive=od)
            out["rows"].append({"label": label, "contact": cname, "held": p["held"], "rise_mm": p["rise_mm"],
                                "pads_settle": p["pads_settle"], "pads_end": p["pads_end"]})
            print(f"{label:28} {cname:8} {'HELD' if p['held'] else 'DROP'} rise {p['rise_mm']:.1f}")
    (SWEEP / "contact_sensitivity.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
