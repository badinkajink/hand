"""Build comparison-grid videos from a morph_pipeline_sweep tag (2026-07-13).

Reads docs/experiments/MORPH_PIPELINE_<tag>.json, picks the BEST replica per design
(max held-cos with an existing video), and renders labeled ffmpeg grids:

  <prefix>_lift_grid.mp4     — the lift/handoff phase (0..~1.5 s, slowed 2x): does
                               Policy A deliver? (program answer: yes, everywhere)
  <prefix>_reorient_grid.mp4 — the post-handoff reorient phase (0.8 s..end, real time):
                               the axis where designs/draws actually differ.

Optionally a highlights row (--highlights id=video ...) hstacked at 2x scale.

Videos are 320x240@50 (1 frame per control step; handoff onset = step 40 = 0.8 s).
Run: uv run python scripts/make_sweep_video_grids.py --tag global12x2 \
       --out-prefix docs/rl/videos/<YYYYMMDD>_sweep/global12
"""
from __future__ import annotations
import argparse, json, re, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
W, H = 320, 240


def best_per_design(tag: str):
    recs = json.loads((ROOT / f"docs/experiments/MORPH_PIPELINE_{tag}.json").read_text())
    by: dict[str, list] = {}
    for r in recs:
        base = re.sub(r"_r\d+$", "", r["id"])
        h = r.get("handoff") or {}
        cos = ((h.get("health") or {}).get("metrics") or {}).get("held_cos_tail")
        vid = h.get("video")
        if cos is None or not vid or not Path(vid).exists():
            continue
        by.setdefault(base, []).append((cos, r["id"], vid))
    out = []
    for base in sorted(by):
        cos, rid, vid = max(by[base])
        rep = rid[len(base):].lstrip("_")
        out.append((base, rep, cos, vid))
    return out


def drawtext(label: str) -> str:
    safe = label.replace(":", r"\:").replace("'", "")
    return (f"drawtext=fontfile={FONT}:text='{safe}':x=4:y=4:fontsize=15:"
            f"fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=3")


def grid(items, out: Path, trim: str, pts: str, cols: int):
    n = len(items)
    cmd = ["ffmpeg", "-y"]
    for _, _, _, vid in items:
        cmd += ["-i", vid]
    parts, layout = [], []
    for k, (base, rep, cos, _) in enumerate(items):
        label = f"{base} {rep}  cos {cos:+.2f}"
        parts.append(f"[{k}:v]trim={trim},setpts={pts},{drawtext(label)}[v{k}]")
        layout.append(f"{(k % cols) * W}_{(k // cols) * H}")
    parts.append("".join(f"[v{k}]" for k in range(n))
                 + f"xstack=inputs={n}:layout={'|'.join(layout)}[out]")
    cmd += ["-filter_complex", ";".join(parts), "-map", "[out]",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(out)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    print(f"wrote {out}")


def highlights(pairs, out: Path):
    cmd = ["ffmpeg", "-y"]
    parts = []
    for k, (label, vid) in enumerate(pairs):
        cmd += ["-i", vid]
        parts.append(f"[{k}:v]scale={W*2}:{H*2}:flags=neighbor,{drawtext(label)}[v{k}]")
    parts.append("".join(f"[v{k}]" for k in range(len(pairs)))
                 + f"hstack=inputs={len(pairs)}[out]")
    cmd += ["-filter_complex", ";".join(parts), "-map", "[out]",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(out)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="global12x2")
    ap.add_argument("--out-prefix", default=None,
                    help="default: docs/rl/videos/<today>_sweep/<HHMM>_grid (see morphohand.tools.video_paths)")
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--lift-end", default="1.5", help="lift-phase trim end (s); onset=0.8")
    ap.add_argument("--highlights", nargs="*", default=None,
                    help="label=video pairs for an hstacked highlights row")
    args = ap.parse_args()
    items = best_per_design(args.tag)
    print(f"{len(items)} designs: " + ", ".join(f"{b} {r} {c:+.2f}" for b, r, c, _ in items))
    if args.out_prefix:
        pre = ROOT / args.out_prefix
    else:
        from morphohand.tools.video_paths import video_out
        pre = video_out("grid", "sweep", ext="")
    # lift phase at half speed (it is only ~1.2 s of footage)
    grid(items, pre.with_name(pre.name + "_lift_grid.mp4"),
         trim=f"0:{args.lift_end}", pts="2*(PTS-STARTPTS)", cols=args.cols)
    # reorient phase (post-onset) in real time
    grid(items, pre.with_name(pre.name + "_reorient_grid.mp4"),
         trim="0.8:4.78", pts="PTS-STARTPTS", cols=args.cols)
    if args.highlights:
        pairs = [tuple(s.split("=", 1)) for s in args.highlights]
        highlights(pairs, pre.with_name(pre.name + "_highlights.mp4"))


if __name__ == "__main__":
    main()
