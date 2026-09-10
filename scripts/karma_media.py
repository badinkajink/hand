"""Render the page's media from the finished sweep: two reachable sets and one replay.

Picks the highest- and lowest-scoring designs out of the collected table so the figures are
the sweep's own extremes rather than hands chosen by hand, then renders each hand's KaRMA
reachable set and replays the top hand's rolling path through MuJoCo contact physics.

    python scripts/karma_media.py --table .../karma_table.json --work <workdir> \\
        --karma-root <checkout> --out-dir .../media
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("$", " ".join(str(c) for c in cmd[:6]), "...")
    p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True,
                       env={**__import__("os").environ, "MUJOCO_GL": "egl"})
    print((p.stdout or p.stderr).strip()[-500:])
    if p.returncode != 0:
        raise SystemExit(f"failed: {' '.join(str(c) for c in cmd)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--karma-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--python", default=str(ROOT / ".venv/bin/python"))
    args = ap.parse_args()

    rows = [r for r in json.loads(args.table.read_text())
            if r["variant"] == "scaled" and r["pair"] == "thumb-index"]
    rows.sort(key=lambda r: r["n_voxels"])
    lo, hi = rows[0], rows[-1]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for tag, r in (("best", hi), ("worst", lo)):
        rd = args.work / f"{r['design']}_thumb_index_scaled"
        run([args.python, ROOT / "scripts/karma_render_reachable.py", "--run", rd,
             f"--mounts={','.join(str(x) for x in r['mounts_mm'])}",
             "--out", args.out_dir / f"20260910-reach_{tag}.png",
             "--cam-dist", "0.24"])
        print(f"  {tag}: {r['design']} {r['n_voxels']} voxels, "
              f"retained={r['retained']}, KaRMA-T {r['karma_t']:.5f}")

    rd = args.work / f"{hi['design']}_thumb_index_scaled"
    mcfg = yaml.safe_load((rd / "metric.yaml").read_text())
    scale = (hi["l_ref_mm"] * 1e-3) / float(mcfg.get("l_ref_nominal_m", 0.2))
    sphere_mm = float(mcfg["sphere_radius_m"]) * scale * 1e3
    link_mm = float(mcfg["link_radius_m"]) * scale * 1e3
    run([args.python, ROOT / "scripts/karma_replay_mujoco.py",
         "--pkl", rd / "current.pkl", "--karma-root", args.karma_root,
         f"--mounts={','.join(str(x) for x in hi['mounts_mm'])}",
         "--sphere-radius-mm", f"{sphere_mm:.4f}", "--pad-radius-mm", f"{link_mm:.4f}",
         "--squeeze-deg", "5", "--cam-dist", "0.24", "--cam-az", "55", "--cam-elev", "8",
         "--video", args.out_dir / "20260910-karma_replay.mp4",
         "--json", args.out_dir.parent / "replay.json"])
    print(f"media -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
