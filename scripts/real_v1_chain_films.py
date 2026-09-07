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
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

# How far a run got, so "the best cell" is well defined for a hand that never completes.
RANK = ("held_lift", "carry_ok", "stood_ok", "grip_ok", "ok")
WEIGHT_N = 0.240   # the screwdriver's own weight -- below this the pads are touching, not carrying

# The sweep builds its arm tag from its own knobs; parsing it back is how a film reproduces the
# exact cell rather than the base config. Keep in step with `real_v1_chain_hands.main`.
ARM_KEYS = {"c": ("clear", lambda v: v / 1000.0), "r": ("repose_steps", int),
            "g": ("gait_scan", int), "o": ("release_mm", float), "t": ("turn_steps", int),
            "b": ("budget", float), "k": ("axis_k", float), "a": ("angle_deg", float)}


def _knobs(arm: str) -> dict:
    """`load0_sq2_t550_b0.5_k0.35_a-90` -> the kwargs that produced it."""
    out: dict = {}
    for part in (arm or "").split("_"):
        m = re.fullmatch(r"([a-z])(-?[\d.]+)", part)
        if m and m.group(1) in ARM_KEYS:
            key, cast = ARM_KEYS[m.group(1)]
            out[key] = cast(float(m.group(2)))
    return out


def _seam(r: dict, phase: str) -> dict:
    for e in r.get("seams") or []:
        if e.get("phase") == phase:
            return e
    return {}


def _held_turn(r: dict) -> tuple:
    """Rank on the SIGNED held reorientation: +1 is tip down, -1 is the tool on its handle.

    `ok` ranks the chain, which in table mode completes with the tool standing handle-down --
    952 of 967 such stands were, and filming by `ok` filmed those. This ranks what the turn
    actually did, gated on the hand still carrying the tool clear of the floor.
    """
    s = _seam(r, "reoriented")
    carried = (s.get("pad_contacts") or 0) >= 2 and (s.get("pad_force_N") or 0) >= WEIGHT_N \
        and (s.get("z") or 0.0) > 0.08
    return (1 if carried else 0, s.get("cos") or -1.0, s.get("pad_force_N") or 0.0)


def _reached(r: dict) -> int:
    n = 0
    for k in RANK:
        if not r.get(k):
            break
        n += 1
    return n


def _pick(runs: list[dict], rank: str = "chain") -> dict:
    if rank == "held":
        return max(runs, key=_held_turn)
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
    stem = job.get("stem") or f"20260905-{tag}" + ("" if sq is None else f"_sq{sq:g}") + f"_s{seed}"
    kw = {"_hand": hand, "_fit": fit, "_tag": "film" if sq is None else f"film_sq{sq:g}",
          "_table": job.get("table", True),
          "load_target": 0.0, "seed": seed, "jitter": 0.0005, "cycles": job["cycles"],
          **job.get("knobs", {}),
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
                  f"xstack=inputs={len(vids)}:layout={lay}:fill=black:shortest=0[out]")
    # tpad clones each clip's last frame forever, so the output MUST be cut to the longest
    # input or ffmpeg encodes an endless stream (a 23 MB file with no moov atom).
    dur = 0.0
    for _, v in vids:
        q = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                            "-of", "csv=p=0", str(v)], capture_output=True, text=True)
        try:
            dur = max(dur, float(q.stdout.strip()))
        except ValueError:
            pass
    cmd = (["ffmpeg", "-y"] + ins +
           ["-filter_complex", ";".join(fc), "-map", "[out]",
            "-t", f"{dur:.2f}" if dur > 0 else "30",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "28",
            "-preset", "slow", "-movflags", "+faststart", "-r", "40", str(out)])
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
    ap.add_argument("--rank", choices=("chain", "held"), default="chain",
                    help="chain = furthest gate reached; held = best SIGNED held reorientation")
    ap.add_argument("--mode", choices=("air", "table", "sweep"), default="sweep",
                    help="sweep = whatever the sweep itself ran (its meta.stand)")
    ap.add_argument("--date", default="20260905", help="YYYYMMDD prefix for the output files")
    args = ap.parse_args()

    size = tuple(int(v) for v in args.size.split(","))
    sweep = json.loads(args.sweep.read_text())
    rows = sweep["rows"]
    stand = sweep.get("stand") or (sweep.get("meta") or {}).get("stand") or "table"
    table = stand == "table" if args.mode == "sweep" else args.mode == "table"
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["tag"], []).append(r)
    if args.only:
        keep = set(args.only.split(","))
        by = {k: v for k, v in by.items() if k in keep}

    args.out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for tag in sorted(by):
        p = _pick(by[tag], args.rank)
        kn = _knobs(p.get("arm", ""))
        sq = _sq_of(p)   # None = the plan's own squeeze, which is what a no-`_sq` arm ran
        stem = f"{args.date}-{tag}_" + "_".join(
            f"{k}{v:g}" for k, v in sorted(kn.items()) if isinstance(v, (int, float))
        ) + f"_s{p.get('seed', 0)}"
        jobs.append({"tag": tag, "sq": sq, "seed": p.get("seed", 0), "knobs": kn,
                     "table": table, "stem": stem,
                     "out": str(args.out), "size": list(size), "cycles": args.cycles})
        s = _seam(p, "reoriented")
        print(f"  {tag:22} {stand:5s} " + " ".join(f"{k}={v:g}" for k, v in sorted(kn.items()))
              + f" s{p.get('seed', 0)}  reoriented cos {s.get('cos', 0):+.3f} "
              f"{s.get('pad_contacts', 0)}p {s.get('pad_force_N', 0):.2f}N  "
              f"reached {RANK[_reached(p) - 1] if _reached(p) else 'grasp'}", flush=True)

    done = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(_cell, jobs, chunksize=1):
            done.append(r)
            print(f"  -> {r.get('tag')}: ok={r.get('ok')} reorient={r.get('reorient_deg')} "
                  f"drop={r.get('drop_stage')}", flush=True)

    (args.out / "films.json").write_text(json.dumps(done, separators=(",", ":")))
    vids = []
    for r in done:
        if not (r.get("video") and Path(r["video"]).exists()):
            continue
        s = _seam(r, "reoriented")
        lab = (f"{r['tag']}  cos {s.get('cos', 0):+.2f} {s.get('pad_contacts', 0)}p "
               f"{s.get('pad_force_N', 0):.1f}N  " +
               ("CHAIN" if r.get("ok") else "drop@" + str(r.get("drop_stage"))))
        vids.append((lab, Path(r["video"])))
    g = args.out / f"{args.date}-chain_grid.mp4"
    print(f"grid: {g} " + ("written" if grid(vids, g, args.cols, size) else "FAILED"))
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
