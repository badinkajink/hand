"""SR2 hand: a chunk folder of the crosstalk plans checked with every joint anywhere within the runner's error limits of
where it is sent (docs/experiments/20261104-sr2_hand_crosstalk). From 15:5x on 2026-11-05 lab/run_plan.py accepts a
yaw joint up to --max-err-yaw 10 deg from its goal and mcp and pip up to 6 deg (the middle's yaw stuck 8.1 deg short at
rv04_mid step 44 with nothing touching). Every move and every safety return to zero is walked in 3 deg waypoints as the
runner walks it. At each waypoint the pairs of fingers under 40 mm apart (cad_v2 collision spheres) are checked at the
64 corners of a box on the pair's six joints, the third finger at the waypoint.
  --box sym     each joint anywhere within the runner's limits of the waypoint, +-10 deg on a yaw, +-6 on mcp and pip
  --box short   (default) each joint short of the waypoint along its move by up to SHORT deg or past it by up to PAST;
                a joint that does not move keeps the interval of its last move. The night of 11-04 measured, at 711
                dwells, shortfalls along the move of at most 3.2 to 5.6 deg on the yaws and 1.3 to 5.8 on mcp and pip
                (p95 2.5 to 3.5 and 0.8 to 1.9), and overshoots of at most 2.5 to 3.0 deg on the yaws and 0.8 to 1.5 on
                mcp and pip.
Usage:
  .venv/bin/python scripts/fingertip/sr2_plan_tolerance_check.py CHUNK_DIR [FIRST_LAYOUT [FIRST_STEP]] [--box sym|short]
      [--short-yaw 10] [--past-yaw 3] [--short-curl 3] [--past-curl 1.5]
FIRST_STEP is the index of the first step of FIRST_LAYOUT's poses chunk to check (default 0)."""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs/experiments/20261104-sr2_hand_crosstalk/lab"))
from handgeo import F, Hand, ORIGIN  # noqa: E402

J = ("yaw", "mcp", "pip")
CAP = 40.0
h = Hand()
RMAX = {f: float(h.r[f].max()) for f in F}
PAIRS = list(itertools.combinations(F, 2))
CORNERS = np.array(list(itertools.product((-1, 1), repeat=6)), float)


def pose(v):
    return {f: {j: float(v[3 * i + k]) for k, j in enumerate(J)} for i, f in enumerate(F)}


def pair_clear(a, b):
    ca, cb = h.centres(a), h.centres(b)
    M = cKDTree(ca).sparse_distance_matrix(cKDTree(cb), CAP + RMAX[a] + RMAX[b], output_type="coo_matrix")
    return float((M.data - h.r[a][M.row] - h.r[b][M.col]).min()) if M.nnz else CAP


def at(local, v, lo, hi):
    """(nominal, with tolerance, pair, the joints at the worst corner) at one waypoint; each joint between v + lo and
    v + hi."""
    h.set(local, pose(v))
    nom = {f"{a}-{b}": pair_clear(a, b) for a, b in PAIRS}
    best = (min(nom.values()), min(nom.values()), min(nom, key=nom.get), None)
    for a, b in PAIRS:
        if nom[f"{a}-{b}"] >= CAP:
            continue
        idx = [3 * F.index(a) + k for k in range(3)] + [3 * F.index(b) + k for k in range(3)]
        for c in CORNERS:
            w = v.copy()
            w[idx] += np.where(c > 0, hi[idx], lo[idx])
            h.set(local, pose(w))
            d = pair_clear(a, b)
            if d < best[1]:
                best = (best[0], d, f"{a}-{b}", w)
    return best


def interval(a, last, start, target):
    """Each joint's interval about the waypoint on the move start -> target, and the move's direction per joint."""
    d = np.sign(target - start)
    d = np.where(d == 0, last, d)
    if a.box == "sym":
        t = np.tile([10.0, 6.0, 6.0], 3)
        return -t, t, d
    short = np.tile([a.short_yaw, a.short_curl, a.short_curl], 3)
    past = np.tile([a.past_yaw, a.past_curl, a.past_curl], 3)
    e0, e1 = -short * d, past * d
    sym = d == 0                                   # never moved: either side
    e0, e1 = np.where(sym, -short, np.minimum(e0, e1)), np.where(sym, short, np.maximum(e0, e1))
    return e0, e1, d


def walk(a, local, start, target, last):
    lo, hi, d = interval(a, last, start, target)
    n = int(np.max(np.abs(target - start)) // 3.0) + 1
    return min((at(local, start + (target - start) * i / n, lo, hi) for i in range(n + 1)), key=lambda r: r[1]), d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chunks")
    ap.add_argument("first", nargs="?", type=int, default=1)
    ap.add_argument("first_step", nargs="?", type=int, default=0)
    ap.add_argument("--box", choices=("sym", "short"), default="short")
    ap.add_argument("--short-yaw", type=float, default=10.0)
    ap.add_argument("--past-yaw", type=float, default=3.0)
    ap.add_argument("--short-curl", type=float, default=3.0)
    ap.add_argument("--past-curl", type=float, default=1.5)
    a = ap.parse_args()
    D, first, first_step = Path(a.chunks), a.first, a.first_step
    print(json.dumps(vars(a)), flush=True)
    for mf in sorted(D.glob("*_mounts.json"), key=lambda p: int(p.name.split("_")[0])):
        lay = mf.name[: -len("_mounts.json")]
        pf = D / f"{lay}_poses.json"
        if int(lay.split("_")[0]) < first or not pf.exists():
            continue
        st0 = next(s for s in json.load(open(mf))["steps"] if s["op"] == "mounts")
        local = {f: (st0["mounts"][f]["x"] - ORIGIN[f][0], st0["mounts"][f]["y"] - ORIGIN[f][1]) for f in F}
        steps = json.load(open(pf))["steps"]
        k0 = first_step if int(lay.split("_")[0]) == first else 0
        cur = np.zeros(9) if k0 == 0 else np.array([steps[k0 - 1]["joints"][f][j] for f in F for j in J], float)
        last = np.zeros(9)
        rows = []
        for k in range(k0, len(steps)):
            tgt = np.array([steps[k]["joints"][f][j] for f in F for j in J], float)
            mv, last = walk(a, local, cur, tgt, last)
            ret = walk(a, local, tgt, np.zeros(9), last)[0] if np.any(tgt) else mv
            rows.append((k, mv, ret))
            cur = tgt
        mv = min(rows, key=lambda r: r[1][1])
        ret = min(rows, key=lambda r: r[2][1])
        neg = [r[0] for r in rows if min(r[1][1], r[2][1]) < 0]
        print(f"{lay} steps {k0}-{len(steps) - 1}: moves nominal {min(r[1][0] for r in rows):.1f} mm, with tolerance "
              f"{mv[1][1]:.1f} mm (step {mv[0]}, {mv[1][2]}); safety returns nominal {min(r[2][0] for r in rows):.1f} mm,"
              f" with tolerance {ret[2][1]:.1f} mm (step {ret[0]}, {ret[2][2]}); steps under 0 with tolerance: "
              f"{len(neg)} {neg[:20]}", flush=True)
        for k, m, r in sorted(rows, key=lambda r: min(r[1][1], r[2][1]))[:5]:
            w = m if m[1] <= r[1] else r
            print(f"   step {k}: {'move' if w is m else 'return'} nominal {w[0]:.1f}, tolerance {w[1]:.1f} mm {w[2]}"
                  + (f" at {json.dumps(pose(np.round(w[3], 1)))}" if w[3] is not None else ""), flush=True)


if __name__ == "__main__":
    main()
