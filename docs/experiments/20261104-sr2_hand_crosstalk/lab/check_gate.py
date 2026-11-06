"""At a gate of hom1: the mounts chunk's zero dwell (before the move) and baseline dwell (after it), lab/layout_travel.py
on the depth frames there, and a before/after crop for a look. Prints layout_travel's lines; exit code as layout_travel.

    /usr/bin/python3 check_gate.py CHUNK FROM_LAYOUT TO_LAYOUT      (CHUNK e.g. 3_hom_generalist_70_110)
"""
import json
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image

H = Path("/home/will/workspace/hand")
LOGS = H / "logs/20261105-sr2_crosstalk"
RGBD = LOGS / "d435_rgbd"
chunk, src, dst = sys.argv[1:4]
ev = [json.loads(x) for x in open(LOGS / "hom1" / f"{chunk}_mounts" / "events.jsonl")]
st = {e["step"]: e for e in ev if e["kind"] == "step"}
hms = lambda t: time.strftime("%H:%M:%S", time.localtime(t))
before, after = hms(st[0]["t_end"] - 1.0), hms(st[2]["t_settled"] + 15.0)
print(f"before {before} (zero at {src}), after {after} (baseline at {dst})", flush=True)
r = subprocess.run(["/usr/bin/python3", str(Path(__file__).resolve().parent / "layout_travel.py"),
                    str(RGBD), before, after, src, dst], capture_output=True, text=True)
print(r.stdout + r.stderr, end="", flush=True)


def jpg(h):
    t = int(h[:2]) * 3600 + int(h[3:5]) * 60 + int(h[6:8])
    fs = sorted(RGBD.glob("*.jpg"))
    return min(fs, key=lambda f: abs(int(f.name[:2]) * 3600 + int(f.name[2:4]) * 60 + int(f.name[4:6]) - t))


a, b = Image.open(jpg(before)).crop((380, 60, 1180, 520)), Image.open(jpg(after)).crop((380, 60, 1180, 520))
s = Image.new("RGB", (a.size[0], 2 * a.size[1] + 10), "white")
s.paste(a, (0, 0))
s.paste(b, (0, a.size[1] + 10))
out = LOGS / "d435_rgbd" / f"gate_{chunk}.jpg"
s.save(out, quality=88)
print(f"before/after: {out}")
raise SystemExit(r.returncode)
