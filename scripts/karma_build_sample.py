"""Assemble the labelled real_v1 design sample that the KaRMA evaluation is scored on.

The population is the Sobol-8192 screen (`docs/experiments/20260831-real_v1-sobol8192`),
whose funnel is 8,198 sampled -> 6,629 reachable -> 535 retained -> 227 confirmed. Labels:

  retained   the design survives the retention screen at some residual clip (the binary the
             published incumbents are scored on: six raw mounts give logistic AUC 0.839,
             depth_fit gives 0.803)
  nom_cos    signed cosine of the tool's own +z against world +z at the end of the screened
             turn, best over the confirm cells for that design. +1 is tip down; never use
             tilt_deg, which folds 180 deg onto 0.
  nom_kept   how many of the n_nom repeats ended still held

Sampling is case-control: equal numbers of retained and not-retained designs drawn from the
reachable pool. AUC is invariant to class balance under case-control sampling, so this
estimates the population AUC while keeping the KaRMA compute affordable. The eight deployed
hands are added unconditionally and flagged, because within-eight correlations are selection
artifacts and must never be pooled with the population (see the design-diversity note: the
same correlation reads -0.90 over the eight and +0.076 over the 535).

Writes a JSON list of {design, mounts_mm, stage, retained, nom_cos, nom_kept, deployed}.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOBOL = ROOT / "docs/experiments/20260831-real_v1-sobol8192"
OPS = ROOT / "docs/experiments/OPERATING_POINTS.json"

BASE_MOUNTS = {"thumb": (-0.050, 0.0), "index": (0.050, 0.055), "middle": (0.050, -0.055)}


def mounts_mm(vec: list[float]) -> list[float]:
    """Absolute palm-frame mount millimetres from the nine-parameter offset vector."""
    off = {"thumb": (vec[0], vec[1]), "index": (vec[3], vec[4]), "middle": (vec[6], vec[7])}
    return [(BASE_MOUNTS[f][k] + off[f][k]) * 1e3
            for f in ("thumb", "index", "middle") for k in (0, 1)]


def load_labels() -> dict[str, dict]:
    man = json.load(open(SOBOL / "grasp_screen_manifest.json"))
    reach = set(open(SOBOL / "reachable.txt").read().split())
    sel: set[str] = set()
    conf: set[str] = set()
    for f in glob.glob(str(SOBOL / "selected/designs_b*.txt")):
        sel |= set(open(f).read().strip().split(","))
    for f in glob.glob(str(SOBOL / "selected/confirmed/designs_b*.txt")):
        conf |= set(open(f).read().strip().split(","))

    # Best screened turn per design, over every confirm cell it appears in.
    turn: dict[str, dict] = {}
    for f in sorted(glob.glob(str(SOBOL / "confirm_b*.json"))):
        for r in json.load(open(f)):
            c = r.get("nom_cos")
            if c is None:
                continue
            cur = turn.get(r["design"])
            if cur is None or c > cur["nom_cos"]:
                turn[r["design"]] = {
                    "nom_cos": c, "nom_kept": r.get("nom_kept"), "n_nom": r.get("n_nom"),
                    "budget_rad": r.get("budget_rad"),
                    "min_finger_clearance_mm": r.get("min_finger_clearance_mm"),
                }

    out: dict[str, dict] = {}
    for r in man["designs"]:
        t = r["design"]
        stage = 0
        if t in reach:
            stage = 1
        if t in sel:
            stage = 2
        if t in conf:
            stage = 3
        rec = {"design": t, "mounts_mm": mounts_mm(r["vector_m"]), "stage": stage,
               "retained": stage >= 2, "confirmed": stage >= 3, "source": r.get("source")}
        rec.update(turn.get(t, {}))
        out[t] = rec
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-per-class", type=int, default=250,
                    help="retained and not-retained designs drawn from the reachable pool")
    ap.add_argument("--seed", type=int, default=20260910)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    lab = load_labels()
    deployed = {v["id"]: k for k, v in json.load(open(OPS))["hands"].items()}

    reachable = [r for r in lab.values() if r["stage"] >= 1]
    pos = [r for r in reachable if r["retained"]]
    neg = [r for r in reachable if not r["retained"]]
    rng = random.Random(args.seed)
    pick = rng.sample(pos, min(args.n_per_class, len(pos))) + \
        rng.sample(neg, min(args.n_per_class, len(neg)))

    chosen = {r["design"]: dict(r, deployed=None) for r in pick}
    for did, tag in sorted(deployed.items()):
        # Plan tags carry a _b<clip> suffix; the design is the part before it.
        base = tag.rsplit("_b", 1)[0]
        rec = lab.get(base)
        if rec is None:
            print(f"  ! {did} ({tag}) not in the population manifest, skipped")
            continue
        chosen.setdefault(rec["design"], dict(rec, deployed=None))
        chosen[rec["design"]]["deployed"] = did

    rows = sorted(chosen.values(), key=lambda r: r["design"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=1))

    n_ret = sum(r["retained"] for r in rows)
    n_dep = sum(r["deployed"] is not None for r in rows)
    n_cos = sum(r.get("nom_cos") is not None for r in rows)
    print(f"population: {len(lab)} sampled, {len(reachable)} reachable, "
          f"{len(pos)} retained, {sum(r['confirmed'] for r in lab.values())} confirmed")
    print(f"sample: {len(rows)} designs, {n_ret} retained, {n_cos} with a screened turn, "
          f"{n_dep} deployed -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
