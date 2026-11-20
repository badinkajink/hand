#!/usr/bin/env python3
"""Codesign of a two-rod differential for the SR2 finger's yaw and MCP joints (cad_v4c), with STS3032 servos.

The mechanism is the AmazingHand's (scripts/cad/amazinghand_differential.py): two servos turn horns of radius r, and a
rod with a ball joint at each end joins each horn's ball to a lug on the MCP link. The lugs sit h from the MCP axis and
w to either side of the yaw axis, at the joint centre's height when the arm is at rest. The servos stand side by side,
shafts along the yaw axis 12.1 mm apart (their case width), horns pointing outward, a rod length L below the lugs.

figure   redraw the figure from the sweep's rows, with the chosen design (CHOSEN, v4c)
sweep    every (r, h, w, L) of the grid: the reachable region (both rods close, horns within +-90 deg, each joint at least
         FLOOR of one STS3032's torque), the largest share of the SR2 driver's range (yaw +-85, MCP -15 to 92) that a
         placement of the region covers, the torques at rest and over that placement; writes
         docs/experiments/20261118-sr2_actuator_layout/data/20261119-v4c_codesign.json and the figure
FLOOR is 1 / 1.96: the current hand's one SCS0009 per joint, in STS3032 limits (4.5 against 2.3 kg cm at 6 V, the sheets).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _paths  # noqa: E402,F401

import itertools  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402

import numpy as np  # noqa: E402

import amazinghand_differential as AD  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "docs/experiments/20261118-sr2_actuator_layout"
OUT = PAGE / "data/20261119-v4c_codesign.json"
FIG = PAGE / "media/20261119-v4c_codesign.png"
STS_OVER_SCS = 4.5 / 2.3  # stall torque at 6 V, Feetech sheets
FLOORS = {"scs0009": 1.0 / STS_OVER_SCS, "sts3032": 1.0}  # joint torque per STS3032 limit
SHAFT_HALF = 6.05  # mm, half the STS3032's case width: two cases side by side
DRIVER = {"yaw": (-85.0, 85.0), "mcp": (-15.0, 92.0)}
GRID = {"r": (5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0), "h": (5.0, 6.0, 8.0, 10.0, 12.0, 14.0),
        "w": (6.0, 8.0, 10.0, 12.0, 14.0, 16.0), "L": (25.0, 33.0, 45.0)}
AMAZING = {"r": 5.0, "h": 6.2, "w": 8.0, "L": 33.0}
CHOSEN = {"r": 10.0, "h": 8.0, "w": 8.0, "L": 45.0}  # v4c: the front's design with both joints at 1.6 limits at rest


def make_geom(r, h, w, L, shaft_half=SHAFT_HALF):
    """The geometry in the form of amazinghand_differential.load_geom, in a frame with the joint centre at 0, the yaw
    axis along x, the MCP axis along y and the MCP link along +z at rest (mm)."""
    G = {"O": np.zeros(3), "e_a": np.array([1.0, 0.0, 0.0]), "e_f": np.array([0.0, 1.0, 0.0]), "f0": 0.0, "rods": []}
    for s in (1.0, -1.0):
        ax, u = np.array([-1.0, 0.0, 0.0]), np.array([0.0, s, 0.0])
        C, A = np.array([h, s * shaft_half, -L]), np.array([h, s * w, 0.0])
        rod = {"axis": ax, "u": u, "v": np.cross(ax, u), "C": C, "A": A, "r": r,
               "L": float(np.linalg.norm(A - (C + r * u))), "th0": 0.0}
        wv = A - C
        p_, q_ = wv @ rod["u"], wv @ rod["v"]
        k = (wv @ wv + r * r - rod["L"] ** 2) / (2 * r)
        acos = math.acos(float(np.clip(k / math.hypot(p_, q_), -1.0, 1.0)))
        rod["branch"] = min((1.0, -1.0), key=lambda b: abs(float(AD.wrap(math.atan2(q_, p_) + b * acos))))
        G["rods"].append(rod)
    return G


def window_cells():
    return (int(round((DRIVER["yaw"][1] - DRIVER["yaw"][0]) / AD.STEP)) + 1,
            int(round((DRIVER["mcp"][1] - DRIVER["mcp"][0]) / AD.STEP)) + 1)


def best_window(inside):
    """Largest share of the driver's rectangle inside the region over all placements (a summed-area table), and the
    placement's first cell."""
    nA, nF = window_cells()
    S = np.zeros((inside.shape[0] + 1, inside.shape[1] + 1))
    S[1:, 1:] = inside.cumsum(0).cumsum(1)
    W = S[nA:, nF:] - S[:-nA, nF:] - S[nA:, :-nF] + S[:-nA, :-nF]
    i, j = np.unravel_index(int(np.argmax(W)), W.shape)
    return float(W[i, j]) / (nA * nF), int(i), int(j)


def evaluate(r, h, w, L, floor):
    G = make_geom(r, h, w, L)
    a, f, th, inside, H, tn, cap_f, cap_a = AD.region(G, floor=floor)
    share, i, j = best_window(inside)
    nA, nF = window_cells()
    win = np.zeros_like(inside)
    win[i:i + nA, j:j + nF] = True
    sel = win & inside
    ik = (int(np.argmin(np.abs(AD.A_GRID))), int(np.argmin(np.abs(AD.F_GRID))))
    return {"r": r, "h": h, "w": w, "L": L, "rod_mm": round(G["rods"][0]["L"], 2), "floor": round(floor, 4),
            "area_deg2": float(inside.sum() * AD.STEP ** 2), "driver_share": round(share, 4),
            "window_yaw_deg": [float(AD.A_GRID[i]), float(AD.A_GRID[i + nA - 1])],
            "window_rest_flexion_offset_deg": [float(AD.F_GRID[j]), float(AD.F_GRID[j + nF - 1])],
            "rest_torque_sts": {"mcp": round(float(cap_f[ik]), 3), "yaw": round(float(cap_a[ik]), 3)},
            "window_torque_sts_min_median": {
                "mcp": [round(float(cap_f[sel].min()), 3), round(float(np.median(cap_f[sel])), 3)] if sel.any() else None,
                "yaw": [round(float(cap_a[sel].min()), 3), round(float(np.median(cap_a[sel])), 3)] if sel.any() else None},
            "yaw_span_at_rest_flexion_deg": AD.extent(inside)[1]}, (inside, cap_f, cap_a)


# --------------------------------------------------------------------------- the rods with bought joints (19 Nov night)
# A ball joint turns freely about its stud and tilts only a little. At the lugs the rod turns through both joint motions
# relative to the MCP link, so each lug is a universal joint: pin 1 along the MCP axis on the link, pin 2 on the rod,
# square to pin 1 and to the rod. The horn end keeps a ball joint (a rod end on a shoulder bolt along the horn axis),
# which then only tilts with the rod's lean.
HW = {"shaft_half": 9.3,  # mm; the STS3032's 18 mm aluminium disc horns side by side, 0.6 mm apart
      "rod_end": "Aurora MW-M3", "rod_end_tilt_deg": 12.5,  # catalogue a3, shoulder-bolt mounted: 25 deg in total
      "rod_end_head_r": 6.25,  # mm, half the housing's 12.5 mm diameter
      "pin2_deg": {"support": 70.0, "free": 100.0},  # the rod's turn about pin 2 toward pin 1's support and away from it
      "beam_clear_mm": 20.9,  # the x-stage beam's underside, 21.9 mm above v3's yaw axis, less 1 mm
      "tab_beyond_spline_mm": 10.29}  # the STS3032's short-end tab beyond its spline (Feetech drawing)
HW_GRID = {"r": (6.0, 8.0, 10.0, 12.0), "h": (6.0, 8.0, 10.0, 12.0), "w": (6.0, 8.0, 10.0, 12.0, 14.0),
           "L": (20.0, 25.0, 30.0, 35.0, 40.0, 45.0), "away": (45.0, 60.0, 90.0)}
TRIAL_BANDS = {"index": ((-41.9, 9.1), (-14.8, 36.3)), "middle": ((1.1, 35.3), (-15.0, 44.5))}  # 99 % bands, 65 trials
HW_OUT = PAGE / "data/20261119-v4c_codesign_hardware.json"


def joint_angles(G, a, f, th):
    """Per rod (arrays over the grid): the horn rod end's misalignment (deg; its socket set along the horn axis at
    rest), the rod's turn about the lug's pin 2 (deg, positive toward +pin 1) and the horn ball's height beyond the
    horns' plane, away from the joint (mm)."""
    R = AD.pose(G, a, f)
    p1 = R @ G["e_f"]
    ik = (int(np.argmin(np.abs(AD.A_GRID))), int(np.argmin(np.abs(AD.F_GRID))))
    mis, ang2, up = [], [], []
    for i, rod in enumerate(G["rods"]):
        B = rod["C"] + rod["r"] * (np.cos(th[i])[..., None] * rod["u"] + np.sin(th[i])[..., None] * rod["v"])
        A = G["O"] + R @ (rod["A"] - G["O"])
        d = A - B
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        p2 = np.cross(p1, d)
        p2 /= np.linalg.norm(p2, axis=-1, keepdims=True)
        t = np.cross(p2, p1)
        q = np.cross(d, p2)
        n0 = np.array([rod["axis"] @ p2[ik], rod["axis"] @ q[ik], rod["axis"] @ d[ik]])
        n = n0[0] * p2 + n0[1] * q + n0[2] * d
        mis.append(np.degrees(np.arccos(np.clip(np.abs(n @ rod["axis"]), 0.0, 1.0))))
        ang2.append(np.degrees(np.arctan2((d * p1).sum(-1), (d * t).sum(-1))))
        up.append(-(B[..., 2] - rod["C"][2]))
    return np.array(mis), np.array(ang2), np.array(up)


def band_share(inside, band):
    """Largest share of a trial band's rectangle (yaw by MCP, deg) inside the region over placements."""
    nA = int(round((band[0][1] - band[0][0]) / AD.STEP)) + 1
    nF = int(round((band[1][1] - band[1][0]) / AD.STEP)) + 1
    S = np.zeros((inside.shape[0] + 1, inside.shape[1] + 1))
    S[1:, 1:] = inside.cumsum(0).cumsum(1)
    W = S[nA:, nF:] - S[:-nA, nF:] - S[nA:, :-nF] + S[:-nA, :-nF]
    return float(W.max()) / (nA * nF)


def evaluate_hw(r, h, w, L, floor=FLOORS["scs0009"]):
    """The region with the bought joints, for each cap on the horns' swing away from the joint; rows and masks."""
    from scipy import ndimage
    G = make_geom(r, h, w, L, HW["shaft_half"])
    a, f, th, inside0, H, tn, cap_f, cap_a = AD.region(G, floor=floor)
    mis, ang2, up = joint_angles(G, a, f, th)
    ok_mis = (mis <= HW["rod_end_tilt_deg"]).all(0)
    lim_s, lim_f = HW["pin2_deg"]["support"], HW["pin2_deg"]["free"]
    sides = {}
    for i in range(2):  # pin 1's support on the + or the - side of each lug, whichever keeps more of the region
        cand = {+1: (ang2[i] <= lim_s) & (ang2[i] >= -lim_f), -1: (ang2[i] >= -lim_s) & (ang2[i] <= lim_f)}
        sides[i] = max(cand, key=lambda k: int((cand[k] & inside0).sum()))
        ok_mis &= cand[sides[i]]
    ik = (int(np.argmin(np.abs(AD.A_GRID))), int(np.argmin(np.abs(AD.F_GRID))))
    rows, masks = [], {}
    for away in HW_GRID["away"]:
        s_th = np.stack([np.degrees(th[i]) * G["rods"][i]["u"][1] for i in range(2)])  # + is away from the joint
        m = inside0 & ok_mis & (s_th <= away).all(0)
        lab, _n = ndimage.label(m)
        m = lab == lab[ik] if lab[ik] else np.zeros_like(m)
        share, i0, j0 = best_window(m) if m.any() else (0.0, 0, 0)
        top = max(L + HW["tab_beyond_spline_mm"], L + (float(up[:, m].max()) if m.any() else 0.0) + HW["rod_end_head_r"])
        nA, nF = window_cells()
        win = np.zeros_like(m)
        win[i0:i0 + nA, j0:j0 + nF] = True
        sel = win & m
        rows.append({"r": r, "h": h, "w": w, "L": L, "away_deg": away, "rod_mm": round(G["rods"][0]["L"], 2),
                     "driver_share": round(share, 4), "area_deg2": float(m.sum() * AD.STEP ** 2),
                     "trial_band_share": {k: round(band_share(m, b), 4) for k, b in TRIAL_BANDS.items()},
                     "rest_torque_sts": {"mcp": round(float(cap_f[ik]), 3), "yaw": round(float(cap_a[ik]), 3)},
                     "window_torque_sts_min": {"mcp": round(float(cap_f[sel].min()), 3) if sel.any() else None,
                                               "yaw": round(float(cap_a[sel].min()), 3) if sel.any() else None},
                     "window_yaw_deg": [float(AD.A_GRID[i0]), float(AD.A_GRID[i0 + nA - 1])],
                     "window_rest_flexion_offset_deg": [float(AD.F_GRID[j0]), float(AD.F_GRID[j0 + nF - 1])],
                     "top_above_joint_mm": round(top, 2), "joint_z_from_v3_yaw_axis_mm": round(HW["beam_clear_mm"] - top, 2),
                     "pin1_support_side": [sides[0], sides[1]],
                     "ideal_share": round(best_window(inside0)[0], 4) if inside0.any() else 0.0})
        masks[away] = m
    return rows, masks, (G, th, mis, ang2, up, cap_f, cap_a)


def cmd_hardware():
    rows = []
    for r, h, w, L in itertools.product(HW_GRID["r"], HW_GRID["h"], HW_GRID["w"], HW_GRID["L"]):
        rows += evaluate_hw(r, h, w, L)[0]
    out = {"script": "scripts/cad/v4c_codesign.py", "command": "hardware", "hw": HW, "grid": HW_GRID,
           "trial_bands": TRIAL_BANDS, "floor_sts": round(FLOORS["scs0009"], 4), "driver": DRIVER, "rows": rows}
    HW_OUT.write_text(json.dumps(out, indent=0))
    return rows


def cmd_sweep():
    rows = []
    for r, h, w, L in itertools.product(GRID["r"], GRID["h"], GRID["w"], GRID["L"]):
        for name, fl in FLOORS.items():
            row, _ = evaluate(r, h, w, L, fl)
            row["floor_name"] = name
            rows.append(row)
    ref = {name: evaluate(**AMAZING, floor=fl)[0] for name, fl in FLOORS.items()}
    best = {name: max((x for x in rows if x["floor_name"] == name),
                      key=lambda x: (x["driver_share"], x["rest_torque_sts"]["mcp"])) for name in FLOORS}
    # Pareto front over the scs0009 floor: share against the rest torque of the weaker joint
    sc = [x for x in rows if x["floor_name"] == "scs0009"]
    front = []
    for x in sorted(sc, key=lambda x: -x["driver_share"]):
        t = min(x["rest_torque_sts"].values())
        if not front or t > max(min(y["rest_torque_sts"].values()) for y in front):
            front.append(x)
    chosen = {name: evaluate(**CHOSEN, floor=fl)[0] for name, fl in FLOORS.items()}
    out = {"script": "scripts/cad/v4c_codesign.py", "command": "sweep", "sts_over_scs0009": round(STS_OVER_SCS, 3),
           "chosen": chosen,
           "floors_sts": {k: round(v, 4) for k, v in FLOORS.items()}, "shaft_half_mm": SHAFT_HALF, "grid": GRID,
           "driver": DRIVER, "amazinghand_geometry_with_sts3032": ref, "best": best, "pareto_scs0009_floor": front,
           "rows": rows}
    figure(out)
    out["figure"] = str(FIG.relative_to(ROOT))
    OUT.write_text(json.dumps(out, indent=1))
    return {k: out[k] for k in ("amazinghand_geometry_with_sts3032", "best")} | {"pareto": front}


def figure(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 19})
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.6))
    ax = axes[0]
    sc = [x for x in out["rows"] if x["floor_name"] == "scs0009"]
    t = np.array([min(x["rest_torque_sts"].values()) * STS_OVER_SCS for x in sc])
    s = np.array([100 * x["driver_share"] for x in sc])
    ax.scatter(t, s, s=10, color="0.6", alpha=0.6, label="grid designs")
    fr = out["pareto_scs0009_floor"]
    ax.plot([min(x["rest_torque_sts"].values()) * STS_OVER_SCS for x in fr], [100 * x["driver_share"] for x in fr], "k-o",
            ms=5, lw=1.6, label="best for a torque")
    ref = out["amazinghand_geometry_with_sts3032"]["scs0009"]
    ax.scatter([min(ref["rest_torque_sts"].values()) * STS_OVER_SCS], [100 * ref["driver_share"]], s=150, marker="*",
               color="#c0392b", zorder=5, label="AmazingHand rods")
    ch = out["chosen"]["scs0009"]
    ax.scatter([min(ch["rest_torque_sts"].values()) * STS_OVER_SCS], [100 * ch["driver_share"]], s=110, marker="D",
               color="#1f5fbf", zorder=5, label="v4c")
    ax.set_xlabel("Rest torque of the weaker joint (SCS0009 limits)")
    ax.set_ylabel("Driver's range covered (%)")
    ax.grid(alpha=0.25)
    ax.set_title("(a) Range against torque", fontsize=19)
    ax.legend(fontsize=15, frameon=False, loc="upper right")
    ax = axes[1]
    b = out["chosen"]["scs0009"]
    inside, cap_f, cap_a = evaluate(b["r"], b["h"], b["w"], b["L"], FLOORS["scs0009"])[1]
    cap = np.where(inside, np.minimum(cap_f, cap_a) * STS_OVER_SCS, np.nan)
    im = ax.pcolormesh(AD.A_GRID, AD.F_GRID, cap.T, cmap="viridis", vmin=1.0, vmax=4.0, shading="auto", rasterized=True)
    y0, y1 = b["window_yaw_deg"]
    f0, f1 = b["window_rest_flexion_offset_deg"]
    ax.plot([y0, y1, y1, y0, y0], [f0, f0, f1, f1, f0], "w--", lw=1.8)
    ax.set_xlabel("Yaw from rest (deg)")
    ax.set_ylabel("MCP from rest (deg)")
    ax.set_xlim(-90, 90)
    ax.set_ylim(-90, 110)
    ax.set_aspect("equal")
    ax.set_title(f"(b) v4c: r {b['r']:g}, h {b['h']:g}, w {b['w']:g}, L {b['L']:g} mm", fontsize=19)
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("Weaker joint's torque (SCS0009 limits)")
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=110)
    plt.close(fig)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "sweep"
    if cmd == "sweep":
        print(json.dumps(cmd_sweep(), indent=1))
    elif cmd == "hardware":
        rows = cmd_hardware()
        print(len(rows), "rows ->", HW_OUT)
    elif cmd == "figure":  # redraw from the sweep's rows, with the chosen design
        out = json.loads(OUT.read_text())
        out["chosen"] = {name: evaluate(**CHOSEN, floor=fl)[0] for name, fl in FLOORS.items()}
        ch = out["chosen"]["scs0009"]
        inside = evaluate(**CHOSEN, floor=FLOORS["scs0009"])[1][0]
        rest = DRIVER["mcp"][0] - ch["window_rest_flexion_offset_deg"][0]
        reach = {}
        for mcp in (DRIVER["mcp"][0], 0.0, rest, 70.0, DRIVER["mcp"][1]):
            col = inside[:, int(np.argmin(np.abs(AD.F_GRID - (mcp - rest))))]
            reach[f"{mcp:g}"] = [float(AD.A_GRID[col].min()), float(AD.A_GRID[col].max())] if col.any() else None
        ch["mcp_rest_deg"] = rest
        ch["yaw_reach_at_mcp_deg"] = reach
        figure(out)
        OUT.write_text(json.dumps(out, indent=1))
        print(json.dumps(out["chosen"], indent=1))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
