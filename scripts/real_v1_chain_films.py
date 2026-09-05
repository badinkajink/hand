#!/usr/bin/env python3
"""Render the chain for every hand, one video per hand plus a labelled grid.

A sweep answers "how often", never "what happened". This takes a finished sweep
(`real_v1_chain_hands.py --out DIR`), picks per hand the cell that got furthest,
re-runs exactly that cell with the renderer on, and writes

    <out>/20260905-<tag>_sq<mm>_s<seed>.mp4     one hand, the whole chain
    <out>/20260905-<tag>_sq<mm>_s<seed>_seams.png   its eleven seam frames tiled
    <out>/20260905-chain_grid.mp4               all hands at once, labelled

Cell choice is by outcome, not by hand: a completed run if the hand has one,
otherwise the run that reached the latest gate, ties broken on `reorient_deg`.
Hands that never grasp still get a video -- the failure is the thing to watch.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

# How far a run got, so "the best cell" is well defined for a hand that never completes.
RANK = ("held_lift", "carry_ok", "stood_ok", "grip_ok", "ok")


def _reached(r: dict) -> int:
    n = 0
    for k in RANK:
        if not r.get(k):
            break
        n += 1
    return n


def _pick(runs: list[dict]) -> dict:
    return max(runs, key=lambda r: (_reached(r), r.get("reorient_deg") or -1.0,
                                    -(r.get("seed") or 0)))


def _sq_of(r: dict) -> float | None:
    a = r.get("arm") or ""
    return float(a.split("_sq")[-1].split("_")[0]) if "_sq" in a else None


def _cell(job: dict):
    import real_v1_chain_hands as H
    tag, sq, seed, out, size = (job["tag"], job["sq"], job["seed"],
                                Path(job["out"]), tuple(job["size"]))
    hand = next(h for h in H.hands("AB") if h["tag"] == tag)
    fit = H.prepare(hand, squeeze_mm=sq)
    if fit is None:
        return {"tag": tag, "error": "prepare failed"}
    stem = f"20260905-{tag}_sq{sq:g}_s{seed}"
    kw = {"_hand": hand, "_fit": fit, "_tag": f"film_sq{sq:g}", "_table": True,
          "load_target": 0.0, "seed": seed, "jitter": 0.0005, "cycles": job["cycles"],
          "video": out / f"{stem}.mp4", "video_size": size,
          "film": out / f"{stem}_seams.png",
          "cam": (-60.0, -20.0, 0.42), "cam_look": (0.02, -0.005, 0.045)}
    r = H._cell(kw)
    r["video"] = str(out / f"{stem}.mp4")
    r["film"] = str(out / f"{stem}_seams.png")
    r["sq"] = sq
    return r


def grid(vids: list[tuple[str, Path]], out: Path, cols: int, size: tuple[int, int]) -> bool:
    """Tile the per-hand videos into one labelled grid, padding short runs on their last frame."""
    if not vids:
        return False
    w, h = size
    rows = -(-len(vids) // cols)
    ins: list[str] = []
    for _, v in vids:
        ins += ["-i", str(v)]
    # tpad holds each clip's final frame so a run that ended early stays on screen
    fc = []
    for i, (lab, _) in enumerate(vids):
        txt = lab.replace(":", r"\:")
        fc.append(f"[{i}:v]scale={w}:{h},tpad=stop=-1:stop_mode=clone,"
                  f"drawtext=text='{txt}':x=6:y=6:fontsize=18:fontcolor=white:"
                  f"box=1:boxcolor=black@0.55:boxborderw=4[v{i}]")
    if len(vids) == 1:
        fc.append("[v0]null[out]")
    else:
        lay = "|".join(f"{(i % cols) * w}_{(i // cols) * h}" for i in range(len(vids)))
        fc.append("".join(f"[v{i}]" for i in range(len(vids))) +
                  f"xstack=inputs={len(vids)}:layout={lay}:shortest=0[out]")
    cmd = (["ffmpeg", "-y"] + ins +
           ["-filter_complex", ";".join(fc), "-map", "[out]",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "26",
            "-r", "40", str(out)])
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stderr[-2500:], flush=True)
    return p.returncode == 0 and out.exists()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", type=Path, required=True,
                    help="a chain_hands.json to pick the per-hand best cell from")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--cycles", type=int, default=6)
    ap.add_argument("--size", default="480,360")
    ap.add_argument("--only", default=None, help="comma list of tags")
    args = ap.parse_args()

    size = tuple(int(v) for v in args.size.split(","))
    rows = json.loads(args.sweep.read_text())["rows"]
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["tag"], []).append(r)
    if args.only:
        keep = set(args.only.split(","))
        by = {k: v for k, v in by.items() if k in keep}

    args.out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for tag in sorted(by):
        p = _pick(by[tag])
        jobs.append({"tag": tag, "sq": _sq_of(p) or 2.0, "seed": p.get("seed", 0),
                     "out": str(args.out), "size": list(size), "cycles": args.cycles})
        print(f"  {tag:22} sq{jobs[-1]['sq']:g} seed{jobs[-1]['seed']}  "
              f"reached {RANK[_reached(p) - 1] if _reached(p) else 'grasp'}", flush=True)

    done = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(_cell, jobs, chunksize=1):
            done.append(r)
            print(f"  -> {r.get('tag')}: ok={r.get('ok')} reorient={r.get('reorient_deg')} "
                  f"drop={r.get('drop_stage')}", flush=True)

    (args.out / "films.json").write_text(json.dumps(done, separators=(",", ":")))
    vids = [(f"{r['tag']}  sq{r['sq']:g}" + ("  CHAIN" if r.get("ok") else
            f"  {'stood' if r.get('stood_ok') else 'drop@' + str(r.get('drop_stage'))}"),
             Path(r["video"])) for r in done if r.get("video") and Path(r["video"]).exists()]
    g = args.out / "20260905-chain_grid.mp4"
    print(f"grid: {g} " + ("written" if grid(vids, g, args.cols, size) else "FAILED"))
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
