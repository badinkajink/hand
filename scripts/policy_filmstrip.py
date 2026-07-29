"""Turn a trained policy's rollout video into ONE labelled contact-sheet PNG.

Reward tables say what a policy was paid for; they do not say what it did. A
policy that reports "held the object, peak cos 0.03" looks identical in the
metrics whether it never tried to rotate, tried and was blocked by a finger, or
rotated the wrong way and back. Six frames answer that in seconds.

This samples frames at the run's own PHASE BOUNDARIES (settle / lift onset /
reorient onset / late) rather than uniformly, because the interesting failures
in this repo happen exactly at a phase switch. Frame index == env step: the
eval recorder writes one frame per control step at 50 fps.

Usage:
  # newest eval video of a finished run, phase-aligned frames
  uv run python scripts/policy_filmstrip.py --run results/rl/<run-dir>

  # a specific video / specific steps / more frames
  uv run python scripts/policy_filmstrip.py \
      --video path/to/rollout.mp4 --at 0,60,65,90,140,200 --out /tmp/strip.png

Then READ the PNG. Pair it with the numbers (probe_axial_load.py,
policy_healthcheck.py) — the picture says which number to distrust.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]

# config.yaml step fields that mark a real change in what the env is doing,
# in the order they fire. Label -> (config key, default if absent).
PHASE_KEYS: list[tuple[str, str]] = [
    ("lift", "lift_phase_start_step"),
    ("reorient", "reorient_start_step"),
    ("floor-gate", "floor_proximity_phase_start_step"),
    ("ang-acc damp", "object_ang_acc_phase_start_step"),
    ("handoff", "handoff_onset_step"),
]


def newest_eval_video(run_dir: Path) -> Path:
    """Highest-step eval video in a run dir (the most-trained policy)."""
    vids = list((run_dir / "eval_videos").glob("*.mp4"))
    if not vids:
        raise SystemExit(f"no eval videos under {run_dir / 'eval_videos'}")

    def step_of(p: Path) -> int:
        m = re.search(r"step-(\d+)", p.name)
        return int(m.group(1)) if m else -1

    return max(vids, key=step_of)


def frame_count(video: Path) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-count_frames", "-show_entries", "stream=nb_read_frames",
         "-of", "default=nw=1:nk=1", str(video)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return int(out)


def phase_marks(run_dir: Path | None) -> dict[int, str]:
    """step -> phase label, from the run's own serialized config."""
    if run_dir is None or not (run_dir / "config.yaml").exists():
        return {}
    env = yaml.safe_load((run_dir / "config.yaml").open()).get("env", {})
    marks: dict[int, str] = {}
    for label, key in PHASE_KEYS:
        step = env.get(key)
        if step is None:
            continue
        step = int(step)
        marks[step] = f"{marks[step]}+{label}" if step in marks else label
    return marks


def choose_steps(n_frames: int, n_video: int, marks: dict[int, str]) -> list[int]:
    """Phase boundaries first (they are where things break), then spread the
    remaining budget over the gaps so the whole episode is still covered."""
    picks = {0}
    for step in sorted(marks):
        if len(picks) < n_frames and 0 <= step < n_video:
            picks.add(step)
    picks.add(n_video - 1)
    remaining = n_frames - len(picks)
    if remaining > 0:
        for i in range(1, remaining + 1):
            picks.add(min(n_video - 1, round(i * (n_video - 1) / (remaining + 1))))
    return sorted(picks)[:n_frames]


def extract(video: Path, steps: list[int], workdir: Path) -> list[Path]:
    expr = "+".join(f"eq(n\\,{s})" for s in steps)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video),
         "-vf", f"select='{expr}'", "-vsync", "0",
         str(workdir / "f_%04d.png")],
        check=True,
    )
    got = sorted(workdir.glob("f_*.png"))
    if len(got) != len(steps):
        raise SystemExit(
            f"asked for {len(steps)} frames, ffmpeg returned {len(got)} "
            f"(video has {frame_count(video)} frames)"
        )
    return got


def _font(size: int) -> ImageFont.ImageFont:
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def tile(frames: list[Path], steps: list[int], marks: dict[int, str],
         cols: int, scale: float, fps: float, out: Path, title: str) -> None:
    imgs = [Image.open(p).convert("RGB") for p in frames]
    w, h = imgs[0].size
    w, h = int(w * scale), int(h * scale)
    imgs = [im.resize((w, h), Image.LANCZOS) for im in imgs]

    band = max(18, int(h * 0.10))          # per-tile caption strip
    head = max(20, int(h * 0.11))          # title strip
    rows = (len(imgs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, head + rows * (h + band)), (16, 16, 18))
    draw = ImageDraw.Draw(sheet)
    f_title, f_cap = _font(int(head * 0.55)), _font(int(band * 0.55))

    draw.text((6, head * 0.22), title, font=f_title, fill=(235, 235, 235))
    for i, (im, step) in enumerate(zip(imgs, steps)):
        r, c = divmod(i, cols)
        x, y = c * w, head + r * (h + band)
        sheet.paste(im, (x, y))
        cap = f"step {step}  t={step / fps:.2f}s"
        colour = (235, 235, 235)
        if step in marks:
            cap += f"  <- {marks[step]}"
            colour = (255, 196, 84)         # phase boundary: the frame to stare at
        draw.text((x + 5, y + h + band * 0.2), cap, font=f_cap, fill=colour)
        draw.rectangle([x, y, x + w - 1, y + h - 1], outline=(60, 60, 64))

    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # Not mutually exclusive: a re-render lives outside the run dir but still
    # wants the run's phase marks, so --video + --run is the common case.
    p.add_argument("--run", type=Path, help="run dir: source of phase marks, and "
                                            "of the newest eval video if --video is omitted")
    p.add_argument("--video", type=Path, help="explicit .mp4 (e.g. a re-render)")
    p.add_argument("--frames", type=int, default=8, help="tiles in the sheet")
    p.add_argument("--at", type=str, default=None,
                   help="explicit comma-separated frame/step indices; overrides --frames")
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--scale", type=float, default=2.0,
                   help="upscale tiles; eval videos are only 320x240")
    p.add_argument("--fps", type=float, default=50.0, help="control rate (frames == steps)")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found on PATH")

    if args.run is None and args.video is None:
        raise SystemExit("need --run and/or --video")

    run_dir = args.run
    video = args.video if args.video else newest_eval_video(run_dir)
    if run_dir is None:
        # <run>/eval_videos/x.mp4 -> <run>, so --video alone still gets phase marks
        cand = video.parent.parent
        run_dir = cand if (cand / "config.yaml").exists() else None

    n_video = frame_count(video)
    marks = phase_marks(run_dir)
    if args.at:
        steps = sorted({int(s) for s in args.at.split(",") if s.strip() != ""})
        bad = [s for s in steps if not 0 <= s < n_video]
        if bad:
            raise SystemExit(f"steps {bad} outside video (0..{n_video - 1})")
    else:
        steps = choose_steps(args.frames, n_video, marks)

    out = args.out or (video.parent.parent / "filmstrip" / f"{video.stem}.png")

    with tempfile.TemporaryDirectory() as td:
        frames = extract(video, steps, Path(td))
        tile(frames, steps, marks, args.cols, args.scale, args.fps, out,
             title=video.stem)

    print(f"video   {video}  ({n_video} frames @ {args.fps:g} Hz)")
    if marks:
        print("phases  " + ", ".join(f"{k}={v}" for k, v in sorted(marks.items())))
    print(f"frames  {steps}")
    print(f"\nwrote {out}\n\nNow READ that PNG — do not stop at this summary.")


if __name__ == "__main__":
    main()
