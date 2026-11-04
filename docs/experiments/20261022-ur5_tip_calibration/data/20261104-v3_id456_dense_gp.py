"""What a seat's dense normal presses (11-04, from 11:17) do to the board GP on its v3 tip (seat 1 ID 6, 2 ID 5, 3 ID 4).

A: the night's data alone (the seat's sphere sessions and its rod session before 11-04). B: the same plus the dense presses
(the rod session's actions whose ids are in the seat's dense plan). Usage: gp_dense.py ROD_HOLDS OUT.json [SEAT]. Grouping by point over A and B together (scripts/ur5_tip_calibration_density_curve.groups).
  fixed     a fifth of the night's point groups held out (seeds 0-3), trained on the other night groups (A) or on them
            and every dense group (B): location, direction, magnitude, torsion on the same test holds; location also for
            loads within 0.1 of the normal (|f_t| / |f_n|) and for tilted ones.
  dense     a fifth of the dense groups held out, trained on everything else (B): new points among dense neighbours.
  curve     the data curve of B (fractions of the training groups) for the slope at full data.
"""
import json
import sys

import numpy as np

sys.path.insert(0, "/home/humanoid/Programs/hand/scripts")
import ur5_tip_calibration_v3_data_curve as V  # noqa: E402

H_ = "docs/experiments/20261030-inverted_rig_seat_block/data/holds/"
SEAT = int(sys.argv[3]) if len(sys.argv) > 3 else 1
NIGHT = {1: [H_ + "20261102-145010-s1v6b", H_ + "20261103-122020-s1sph2b"], 2: [H_ + "20261102-182710-s2v5",
         H_ + "20261103-144808-s2sph2"], 3: [H_ + "20261102-220337-s3v4"]}[SEAT]
ROD = sys.argv[1]                                   # the seat's rod holds with the dense step
PLAN = ("docs/experiments/20261030-inverted_rig_seat_block/data/20261104-plan_probe_table_rod_seat%d_v3id%d_dense_fe_draft.json"
        % (SEAT, 7 - SEAT))
V.KINDS[:] = ["press", "sweep", "offset", "slide", "roll", "twist"]
floor = V.PR.floors(0)[:15]
dense_ids = {a["id"] for pl in (PLAN, PLAN.replace("_dense_fe_", "_dense_ft_")) if (V.ROOT / pl).exists()
             for a in json.load(open(V.ROOT / pl))["actions"] if a["stage"] == "grid"}   # the face and edge plan, and the
#                                                                                   flank and tip plan once it has run
H = V.load(NIGHT + [ROD])
raw = np.array([s.split(":", 1)[1] for s in H["id"].astype(str)])
src = np.array([int(s.split(":", 1)[0]) for s in H["id"].astype(str)])
dense = (src == len(NIGHT)) & np.isin(raw, list(dense_ids))
Y = np.c_[H["p"], H["f"], H["tn"]]
fm = np.linalg.norm(H["f"], axis=1)
fn = np.abs(np.sum(H["f"] * H["n"], 1))
tilt = np.sqrt(np.maximum(fm ** 2 - fn ** 2, 0)) / (fn + 1e-9)
G = np.unique(H["group"])
gd = np.unique(H["group"][dense])
mixed = np.intersect1d(gd, np.unique(H["group"][~dense]))
print(f"{len(Y)} holds, {dense.sum()} dense ({len(dense_ids)} dense actions in the plan, "
      f"{len(np.unique(raw[dense]))} with holds); groups {len(G)}, dense {len(gd)} ({len(mixed)} shared with the night)",
      flush=True)
gn = np.setdiff1d(G, gd)                           # night-only groups


def err(train, test):
    m = V.BG.fit(H["d_board"][train], Y[train], floor)
    P = V.BG.predict(m, H["d_board"][test])
    e = V.errors(H, P, test)
    dp = P[:, :3] - H["p"][test]
    n = H["n"][test]
    loc = np.linalg.norm(dp - np.sum(dp * n, 1)[:, None] * n, axis=1)
    hi = fm[test] >= 1.0
    tl = tilt[test]
    e["loc_normal_mm"] = float(np.median(loc[hi & (tl < 0.1)])) if (hi & (tl < 0.1)).any() else float("nan")
    e["loc_tilted_mm"] = float(np.median(loc[hi & (tl >= 0.1)])) if (hi & (tl >= 0.1)).any() else float("nan")
    e["n_train"] = int(train.sum())
    return e


out = dict(fixed=[], dense=[], curve=[])
K = ("location_mm", "loc_normal_mm", "loc_tilted_mm", "direction_deg", "magnitude", "torsion_Nmm", "torsion_zero_Nmm")
for seed in range(4):
    rng = np.random.default_rng(seed)
    g = rng.permutation(gn)
    tg = g[: len(g) // 5]
    test = np.isin(H["group"], tg)
    trainA = ~test & ~dense
    trainB = ~test
    a, b = err(trainA, test), err(trainB, test)
    out["fixed"].append(dict(seed=seed, A=a, B=b))
    print(f"fixed seed {seed}: A " + ", ".join(f"{k} {a[k]:.3f}" for k in K) + f" (train {a['n_train']})", flush=True)
    print(f"              B " + ", ".join(f"{k} {b[k]:.3f}" for k in K) + f" (train {b['n_train']})", flush=True)
    d = rng.permutation(gd)
    td = np.isin(H["group"], d[: max(1, len(d) // 5)])
    e = err(~td, td)
    out["dense"].append(dict(seed=seed, **e))
    print(f"dense seed {seed}: " + ", ".join(f"{k} {e[k]:.3f}" for k in K[:5]), flush=True)
for seed in range(2):                              # the curve of B, test a fifth of all groups
    rng = np.random.default_rng(10 + seed)
    g = rng.permutation(G)
    test = np.isin(H["group"], g[: len(g) // 5])
    pool = g[len(g) // 5:]
    for fr in (0.5, 0.7, 0.85, 1.0):
        train = np.isin(H["group"], pool[: int(round(fr * len(pool)))])
        e = err(train, test)
        out["curve"].append(dict(seed=seed, fraction=fr, groups=int(round(fr * len(pool))), **e))
        print(f"curve seed {seed} fraction {fr}: {int(round(fr * len(pool)))} groups, location {e['location_mm']:.3f} "
              f"(normal {e['loc_normal_mm']:.3f}), direction {e['direction_deg']:.2f}", flush=True)
med = lambda rows, k: float(np.median([r[k] for r in rows]))  # noqa: E731
print("SUMMARY fixed test, night only (A) vs with the dense presses (B):")
for k in K:
    print(f"  {k:18s} A {med([r['A'] for r in out['fixed']], k):.3f}  B {med([r['B'] for r in out['fixed']], k):.3f}")
print("SUMMARY dense points held out: " + ", ".join(f"{k} {med(out['dense'], k):.3f}" for k in K[:5]))
for fr in (0.5, 0.7, 0.85, 1.0):
    r = [x for x in out["curve"] if x["fraction"] == fr]
    print(f"SUMMARY curve {fr}: {med(r, 'groups'):.0f} groups, location {med(r, 'location_mm'):.3f}, normal "
          f"{med(r, 'loc_normal_mm'):.3f}")
json.dump(out, open(sys.argv[2], "w"), indent=1)
