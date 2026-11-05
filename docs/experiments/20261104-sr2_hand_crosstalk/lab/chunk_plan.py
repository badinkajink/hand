"""Cut a layouts plan (make_layouts.py) into one mounts chunk and one poses chunk per layout, for gated_chain.sh. A mounts
chunk is the zero step before a layout move, the move and the baseline after it; the poses chunk is everything up to the
zero step before the next move. The chunks hold the plan's steps from FROM_STEP on, each once and in order, so the
plan's clearance checks still hold; only the pauses between chunks are new.

    python3 chunk_plan.py PLAN OUT_DIR [FROM_STEP]     (FROM_STEP: the zero step before a layout move; default 0)
"""
import json
import sys
from pathlib import Path

plan, out = Path(sys.argv[1]), Path(sys.argv[2])
start = int(sys.argv[3]) if len(sys.argv) > 3 else 0
p = json.load(open(plan))
st = p["steps"]
m = [i for i, s in enumerate(st) if s["op"] == "mounts" and i > start]
assert m and m[0] == start + 1, f"step {start} must be the zero step before a layout move"
out.mkdir(parents=True, exist_ok=True)
meta = {k: p[k] for k in ("layouts", "home_palm_mm", "min_pose_mm", "min_move_mm", "scene") if k in p}
cover = []
for k, mi in enumerate(m, start=1):
    name = st[mi]["family"].removeprefix("layout_")
    assert st[mi - 1]["family"] == "zero" and st[mi + 1]["family"].startswith("baseline"), mi
    end = m[k] - 1 if k < len(m) else len(st)
    for kind, a, b in (("mounts", mi - 1, mi + 2), ("poses", mi + 2, end)):
        if b <= a:
            continue
        json.dump(dict(meta, chunk_of=plan.name, first_step=a, steps=st[a:b]), open(out / f"{k}_{name}_{kind}.json", "w"),
                  indent=1)
        cover += list(range(a, b))
        print(f"{k}_{name}_{kind}.json: steps {a} to {b - 1}")
assert cover == list(range(start, len(st))), "the chunks must hold every step once, in order"
print(f"{len(st) - start} steps in {len(list(out.glob('*.json')))} chunks -> {out}")
