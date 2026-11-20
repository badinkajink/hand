#!/usr/bin/env python3
"""v4c with bought joints (19 November night): the rod differential of scripts/cad/v4c_codesign.py rebuilt from parts.

Each base rod is a laser-cut link with a spherical plain bearing (3 mm bore, 12 mm outside) at the horn, on a shoulder
bolt along the horn axis, and a miniature ball bearing at the lug, on pin 2 of an offset knuckle. The knuckle turns on
pin 1, which an ear on the MCP link holds along the MCP axis from the outboard side. Both joint motions of the finger are
then turns about pins; the spherical bearing only tilts with the rod's lean. The PIP rod is the mirror arrangement: the
knuckle at the PIP servo's horn (pin a along the horn axis) and the spherical bearing at the DIP link's lug, on a bolt
along the PIP axis. The dimensions are the block "v4c_hw" of assets/cad_v4/concepts/20261118-layout_params.json (mm,
the finger's mount frame with the gimbal centre G at the origin: x dorsal, y lateral, z up toward the palm).

kin     reach, torques and joint angles of the base pair and the PIP over a grid; writes data/20261119-v4c_hw_kin.json
balls   the reach with ideal joints, the bought joints and ball links at both ends (two lug-stud mountings, two tilt
        limits), on this geometry and on the AmazingHand's; writes data/20261119-v4c_hw_balls.{json,npz}
figure  the reach against the serial driver's range and the 65 trials (media/20261119-v4c_hw_workspace.png)
check   clearances between the parts over the reach (box-and-ball gaps of scripts/cad/cad_v4_layouts.py)
dxf     the flat parts laid out for 1/8-inch acrylic (cut/20261119-v4c_hw_acrylic.dxf) and a preview
bom     the bill of materials (data/20261119-v4c_hw_bom.csv)
all     everything
Outputs go to docs/experiments/20261119-v4c_hardware/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _paths  # noqa: E402,F401

import json  # noqa: E402
import math  # noqa: E402

import numpy as np  # noqa: E402

from sr2hand.geometry.rotations import rot_axis_angle, rot_x, rot_y  # noqa: E402

import amazinghand_differential as AD  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PARAMS = ROOT / "assets/cad_v4/concepts/20261118-layout_params.json"
OUT = ROOT / "docs/experiments/20261119-v4c_hardware"
DATA, MEDIA, CUT = OUT / "data", OUT / "media", OUT / "cut"
P = json.loads(PARAMS.read_text())
HW = P["v4c_hw"]
T_STS, T_SCS = P["servos"]["sts3032"]["stall_Nm_6V"], P["servos"]["scs0009"]["stall_Nm_6V"]
FLOOR = T_SCS / T_STS  # every joint at least one SCS0009, in STS3032 limits
YAW = np.arange(-90.0, 90.01, 1.0)
MCP = np.arange(-30.0, 125.01, 1.0)
PIP = np.arange(-45.0, 135.01, 1.0)
DRIVER = {"yaw": (-85.0, 85.0), "mcp": (-15.0, 92.0), "pip": (-18.0, 92.0)}
X, Y, Z = np.eye(3)


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def rot_about(n, ang):
    """Rotation matrices about unit axes n (..., 3) by angles ang (...) (rad)."""
    return rot_axis_angle(n, ang)


# ------------------------------------------------------------------------------------------------------ base pair
def base_rest(e=None, c=None):
    """Rest-pose geometry (mount frame) of the two base rods: horn centre C, rest direction u, v = x cross u, radius r;
    pin 1's foot P1 at the codesign's lug (h, +-w, 0) and pin 1 along y; the knuckle's direction k0 and pin 2 p20 at
    rest; the rod's lower bearing Q0 = P1 + e k0 + c p20 and the horn ball B0; the rod's length |B0 - Q0|. A knuckle that
    follows the rod makes pin 1's foot the effective lug, as in the codesign's universal joint."""
    B = HW["base"]
    rods = []
    e = B["knuckle_e"] if e is None else e
    c0 = B["rod_offset_c"] if c is None else c
    for s in (1.0, -1.0):
        c = c0 * (1.0 if B["rod_side"] == "dorsal" else -1.0)  # the rod's plane dorsal or palmar of the knuckle
        C = np.array([B["h"] + c, s * B["shaft_half"], B["L"]])  # the horn ball in the rod's plane
        u = np.array([0.0, s, 0.0])
        v = np.cross(X, u)
        B0 = C + B["r"] * u
        P1 = np.array([B["h"], s * B["w"], 0.0])
        Q0 = P1.copy()
        for _ in range(50):  # pin 2 square to pin 1 and to the rod; the knuckle square to both pins
            d0 = unit(B0 - Q0)
            p20 = unit(np.cross(Y, d0))
            if p20[0] < 0:
                p20 = -p20
            k0 = unit(np.cross(p20, Y))
            if k0 @ d0 < 0:
                k0 = -k0
            Q0 = P1 + e * k0 + c * p20  # e < 0 puts pin 2 on the far side of pin 1 from the horn
        L = float(np.linalg.norm(B0 - Q0))
        rods.append({"s": s, "C": C, "u": u, "v": v, "r": B["r"], "P1": P1, "k0": k0, "p20": p20, "c": c,
                     "e": e, "Q0": Q0, "B0": B0, "L": L, "branch": rest_branch(C, u, v, B["r"], Q0, L)})
    return rods


def pose_R(yaw, mcp):
    """Rotation of the MCP link from its rest pose (yaw about x, then MCP about y; deg arrays)."""
    B = HW["base"]
    a = np.radians(np.asarray(yaw, float) - B["yaw_rest"])
    f = np.radians(np.asarray(mcp, float) - B["mcp_rest"])
    return rot_x(a) @ rot_y(f)


def circle_sphere(C, u, v, r, A, L, branch):
    """Angle th of the point on the circle C + r(u cos th + v sin th) at distance L from A (arrays over poses), on the
    branch th = atan2(q, p) + branch acos(k / rho) that passes through the rest pose. Returns th, ok."""
    w = A - C
    p, q = (w * u).sum(-1), (w * v).sum(-1)
    k = ((w * w).sum(-1) + r * r - L * L) / (2 * r)
    rho = np.hypot(p, q)
    ok = np.abs(k) <= rho
    ac = np.arccos(np.clip(k / np.where(rho > 0, rho, 1.0), -1, 1))
    return AD.wrap(np.arctan2(q, p) + branch * ac), ok


def rest_branch(C, u, v, r, A, L):
    """The branch sign whose solution is th = 0 at rest."""
    return min((1.0, -1.0), key=lambda b: abs(float(circle_sphere(C, u, v, r, A, L, b)[0])))


def base_solve(yaw, mcp, iters=25, e=None, c=None):
    """Horn angles th (2, ...), knuckle turns chi (2, ...), closure ok (2, ...) and the rods' points over poses."""
    rods = base_rest(e, c)
    R = pose_R(yaw, mcp)
    out = {"th": [], "chi": [], "ok": [], "B": [], "Q": [], "P1": [], "k": [], "p2": [], "n1": []}
    for rod in rods:
        P1 = R @ rod["P1"]
        n1 = R @ Y
        k_r, p2_r = R @ rod["k0"], R @ rod["p20"]
        chi = np.zeros(P1.shape[:-1])
        th = np.zeros(P1.shape[:-1])
        for _ in range(iters):
            Rc = rot_about(n1, chi)
            k, p2 = np.einsum("...ij,...j->...i", Rc, k_r), np.einsum("...ij,...j->...i", Rc, p2_r)
            Q = P1 + rod["e"] * k + rod["c"] * p2
            th, ok = circle_sphere(rod["C"], rod["u"], rod["v"], rod["r"], Q, rod["L"], rod["branch"])
            B = rod["C"] + rod["r"] * (np.cos(th)[..., None] * rod["u"] + np.sin(th)[..., None] * rod["v"])
            b = B - P1
            al, be = (p2_r * b).sum(-1), (np.cross(n1, p2_r) * b).sum(-1)
            rho = np.hypot(al, be)
            ac = np.arccos(np.clip(rod["c"] / np.where(rho > 0, rho, 1.0), -1, 1))
            c1, c2 = AD.wrap(np.arctan2(be, al) + ac), AD.wrap(np.arctan2(be, al) - ac)
            kk1 = np.einsum("...ij,...j->...i", rot_about(n1, c1), k_r)
            kk2 = np.einsum("...ij,...j->...i", rot_about(n1, c2), k_r)
            f1, f2 = (kk1 * b).sum(-1), (kk2 * b).sum(-1)  # k points from pin 1 toward the horn ball
            chi = np.where(f1 >= f2, c1, c2)
        Rc = rot_about(n1, chi)
        k, p2 = np.einsum("...ij,...j->...i", Rc, k_r), np.einsum("...ij,...j->...i", Rc, p2_r)
        Q = P1 + rod["e"] * k + rod["c"] * p2
        th, ok = circle_sphere(rod["C"], rod["u"], rod["v"], rod["r"], Q, rod["L"], rod["branch"])
        B = rod["C"] + rod["r"] * (np.cos(th)[..., None] * rod["u"] + np.sin(th)[..., None] * rod["v"])
        resid = np.abs(np.linalg.norm(B - Q, axis=-1) - rod["L"]) + np.abs((p2 * (B - Q)).sum(-1))
        ok &= resid < 1e-6
        for key, val in (("th", th), ("chi", chi), ("ok", ok), ("B", B), ("Q", Q), ("P1", P1), ("k", k), ("p2", p2), ("n1", n1)):
            out[key].append(val)
    return {k: np.array(v) for k, v in out.items()}, rods


def connected(mask, k0):
    from scipy import ndimage
    lab, _ = ndimage.label(mask)
    return lab == lab[k0] if lab[k0] else np.zeros_like(mask)


def base_region(e=None, c=None):
    """Reach of the base pair over the yaw x MCP grid. Each limit is a mask: closure; horns within their limit; every
    joint at or above one SCS0009 (torques from finite differences of the solve); the spherical bearings within their
    tilt; the rod's turn about pin 2 within the knuckle's limits. "inside" is all of them, connected to rest."""
    B = HW["base"]
    Yg, Mg = np.meshgrid(YAW, MCP, indexing="ij")
    S, rods = base_solve(Yg, Mg, e=e, c=c)
    h = 0.05
    J = np.zeros(Yg.shape + (2, 2))  # d th_i / d q_j, q = (yaw, mcp) in rad
    for j, (dy, dm) in enumerate(((h, 0.0), (0.0, h))):
        Sp, _ = base_solve(Yg + dy, Mg + dm, e=e, c=c)
        Sm, _ = base_solve(Yg - dy, Mg - dm, e=e, c=c)
        J[..., :, j] = (AD.wrap(Sp["th"] - Sm["th"]) / math.radians(2 * h)).transpose(1, 2, 0)
    H = np.swapaxes(J, -1, -2)  # joint torque = H @ servo torque
    cap_a, cap_f = AD.axis_capacity(H, 0), AD.axis_capacity(H, 1)
    tilt = np.degrees(np.arccos(np.clip(np.abs(S["p2"] @ X), 0, 1)))  # spherical bearing: bolt along x, ring along p2
    d = unit(S["B"] - S["Q"])
    turn2 = np.degrees(np.arctan2((d * S["n1"]).sum(-1), (d * S["k"]).sum(-1)))  # the rod about pin 2, + toward +pin 1
    pin2 = np.ones_like(S["ok"])
    for i, rod in enumerate(rods):  # the ear holds pin 1 on the outboard side: +pin 1 for the + rod
        out = turn2[i] * rod["s"]
        pin2[i] = (out <= B["pin2_deg"]["toward_ear"]) & (out >= -B["pin2_deg"]["away"])
    k0 = (int(np.argmin(np.abs(YAW - B["yaw_rest"]))), int(np.argmin(np.abs(MCP - B["mcp_rest"]))))
    masks = {"closure": S["ok"].all(0), "horns": (np.abs(np.degrees(S["th"])) <= B["horn_limit_deg"]).all(0),
             "torque": (cap_a >= FLOOR) & (cap_f >= FLOOR),
             "bearing_tilt": (tilt <= HW["hardware"]["spherical_bearing"]["tilt_deg"]).all(0), "pin2": pin2.all(0)}
    inside0 = masks["closure"] & masks["horns"] & masks["torque"] & masks["pin2"]
    inside = connected(inside0 & masks["bearing_tilt"], k0)
    return {"S": S, "rods": rods, "inside": inside, "inside_no_tilt": connected(inside0, k0), "masks": masks,
            "cap_a": cap_a, "cap_f": cap_f, "tilt": tilt, "turn2": turn2, "H": H, "k0": k0, "Yg": Yg, "Mg": Mg}


def ball_need(B, A, sH, sL, k0, step=3.0):
    """Ball links at both ends of a rod from horn ball B to lug A (arrays over poses): the least over the rod's spin of
    the larger of the two tilts (deg). sH and sL are the studs' directions at each pose; the rod's two sockets are
    assembled square to their studs at the rest index k0."""
    d = unit(A - B)
    sH = np.broadcast_to(sH, d.shape)
    e1 = unit(np.cross(d, np.where(np.abs(d[..., :1]) < 0.9, X, Y)))
    e2 = np.cross(d, e1)
    aH = np.arctan2((sH * e2).sum(-1), (sH * e1).sum(-1))
    aL = np.arctan2((sL * e2).sum(-1), (sL * e1).sum(-1))
    tw = float(aL[k0] - aH[k0])
    best = np.full(d.shape[:-1], 180.0)
    for c in np.radians(np.arange(0.0, 360.0, step)):
        n = np.cos(c) * e1 + np.sin(c) * e2
        nL = np.cos(c + tw) * e1 + np.sin(c + tw) * e2
        m = np.maximum(np.degrees(np.arccos(np.clip(np.abs((n * sH).sum(-1)), 0, 1))),
                       np.degrees(np.arccos(np.clip(np.abs((nL * sL).sum(-1)), 0, 1))))
        best = np.minimum(best, m)
    return best


def yaw_from_straight(inside, mcps, yaw=YAW, mcp=MCP):
    """The yaw interval reachable from yaw 0 without leaving the region, at each MCP angle (deg); None where yaw 0 is
    outside. The regions have holes where a joint's torque falls below the floor, so the farthest yaw at an MCP angle can
    lie beyond a gap."""
    out = {}
    i0 = int(np.argmin(np.abs(yaw)))
    for m in mcps:
        col = inside[:, int(np.argmin(np.abs(mcp - m)))]
        if not col[i0]:
            out[str(m)] = None
            continue
        lo = hi = i0
        while lo > 0 and col[lo - 1]:
            lo -= 1
        while hi < len(col) - 1 and col[hi + 1]:
            hi += 1
        out[str(m)] = [float(yaw[lo]), float(yaw[hi])]
    return out


BALL_BETAS = (16, 30)
HALF_MCP = (-15, 0, 15, 30, 46, 60, 75, 92)


def cmd_balls():
    """The rod differential's reach on this geometry with ideal joints, with the bought joints, and with ball links at
    both ends (lug studs along the yaw axis or along the MCP axis, the AmazingHand's mounting) at two tilt limits; and the
    AmazingHand's own geometry with ball links."""
    import v4c_codesign as CD
    R0 = base_region(e=0.0, c=0.0)
    S0, k0, m = R0["S"], R0["k0"], R0["masks"]
    base = m["closure"] & m["horns"] & m["torque"]
    Yg, Mg = np.meshgrid(YAW, MCP, indexing="ij")
    Rp = pose_R(Yg, Mg)
    variants = {"ideal": connected(base, k0)}
    for name, s0 in (("studs_yaw", X), ("studs_mcp", Y)):
        need = np.array([ball_need(S0["B"][i], S0["Q"][i], X, Rp @ s0, k0) for i in range(2)])
        for beta in BALL_BETAS:
            variants[f"{name}_{beta}"] = connected(base & (need <= beta).all(0), k0)
    variants["bought"] = base_region()["inside"]
    res = {"script": "scripts/cad/v4c_hardware.py", "command": "balls", "geometry": {k: HW["base"][k] for k in ("r", "h", "w", "L", "shaft_half", "mcp_rest")},
           "variants": {}}
    for k, ins in variants.items():
        at, best, shift = driver_share(ins)
        res["variants"][k] = {"driver_share_at_position": round(at, 4), "driver_share_best_shift": round(best, 4),
                              "yaw_from_straight_at_mcp_deg": yaw_from_straight(ins, HALF_MCP)}
    # the 65 bench trials: share of finger-periods inside the reach at the design's mounting, and the best turn of the
    # whole module about the yaw and MCP axes (3 deg steps)
    import cad_v4_layouts as LY
    trows = LY.load_trials()
    tpts = {f: np.array([r["q"][f][:2] for r in trows]) for f in LY.FINGERS}

    def cover(ins, P_, da, dm):
        i = np.round(P_[:, 0] - da - YAW[0]).astype(int)
        j = np.round(P_[:, 1] - dm - MCP[0]).astype(int)
        okk = (i >= 0) & (i < ins.shape[0]) & (j >= 0) & (j < ins.shape[1])
        return float(ins[i[okk], j[okk]].sum()) / len(P_)
    for k in ("ideal", "bought"):
        res["variants"][k]["trials_inside_at_mounting"] = {f: round(cover(variants[k], tpts[f], 0, 0), 4) for f in LY.FINGERS}
        best_turn = {}
        for f in LY.FINGERS:
            cand = [(cover(variants[k], tpts[f], da, dm), da, dm) for da in range(-45, 46, 3) for dm in range(-30, 31, 3)]
            top = max(c[0] for c in cand)
            pick = min((c for c in cand if c[0] >= top - 1e-9), key=lambda c: abs(c[1]) + abs(c[2]))
            best_turn[f] = {"share": round(pick[0], 4), "yaw_turn_deg": pick[1], "mcp_turn_deg": pick[2]}
        res["variants"][k]["trials_best_module_turn"] = best_turn
    res["trial_periods"] = len(trows)
    # the published v4c (r 10, h 8, w 8, L 45, shafts 12.1 mm apart; ideal joints): its region in the codesign's frame
    pub = json.loads((ROOT / "docs/experiments/20261118-sr2_actuator_layout/data/20261119-v4c_codesign.json").read_text())["chosen"]["scs0009"]
    Gp = CD.make_geom(pub["r"], pub["h"], pub["w"], pub["L"])
    insp = AD.region(Gp, floor=CD.FLOORS["scs0009"])[3]
    res["published_v4c"] = {k: pub[k] for k in ("r", "h", "w", "L", "driver_share", "mcp_rest_deg")} | {
        "farthest_yaw_reported": pub["yaw_reach_at_mcp_deg"],
        "yaw_from_straight_at_mcp_deg": yaw_from_straight(insp, HALF_MCP, AD.A_GRID, AD.F_GRID + pub["mcp_rest_deg"])}
    profile = {}
    mprof = np.arange(-30.0, 120.01, 1.0)
    for k, ins in list(variants.items()) + [("published_v4c", None)]:
        if ins is None:
            prof = yaw_from_straight(insp, mprof, AD.A_GRID, AD.F_GRID + pub["mcp_rest_deg"])
        else:
            prof = yaw_from_straight(ins, mprof)
        profile[k] = np.array([np.nan if prof[str(m)] is None else prof[str(m)][1] for m in mprof])
    # the AmazingHand's own geometry (scripts/cad/amazinghand_differential.py) with ball links at its rods' ends
    G = AD.load_geom()
    a, f, th, ins, H, tn, cap_f, cap_a = AD.region(G)
    R = AD.pose(G, a, f)
    ik = (int(np.argmin(np.abs(AD.A_GRID))), int(np.argmin(np.abs(AD.F_GRID - G["f0"]))))
    ah = {"ideal": ins}
    for name, s0 in (("studs_abduction", G["e_a"]), ("studs_flexion", G["e_f"])):
        need = []
        for i, rod in enumerate(G["rods"]):
            Bh = rod["C"] + rod["r"] * (np.cos(th[i])[..., None] * rod["u"] + np.sin(th[i])[..., None] * rod["v"])
            Al = G["O"] + R @ (rod["A"] - G["O"])
            need.append(ball_need(Bh, Al, rod["axis"], R @ s0, ik))
        need = np.array(need)
        for beta in BALL_BETAS:
            from scipy import ndimage
            mm = ins & (need <= beta).all(0)
            lab, _ = ndimage.label(mm)
            ah[f"{name}_{beta}"] = lab == lab[ik] if lab[ik] else np.zeros_like(mm)
    res["amazinghand"] = {}
    for k, mm in ah.items():
        col = mm[:, ik[1]]
        res["amazinghand"][k] = {"driver_share_best_shift": round(CD.best_window(mm)[0], 4) if mm.any() else 0.0,
                                 "abduction_at_key_flexion_deg": [float(AD.A_GRID[col].min()), float(AD.A_GRID[col].max())] if col.any() else None}
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "20261119-v4c_hw_balls.json").write_text(json.dumps(res, indent=1))
    np.savez_compressed(DATA / "20261119-v4c_hw_balls.npz", yaw=YAW, mcp=MCP, profile_mcp=mprof,
                        **{k: v for k, v in variants.items()}, **{f"profile_{k}": v for k, v in profile.items()})
    return res


def cmd_figure():
    """Reach of the rod differential with real joints against the serial driver's range and the 65 trials."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    import cad_v4_layouts as L
    Z_ = np.load(DATA / "20261119-v4c_hw_balls.npz")
    rows = L.load_trials()
    pts = {f: np.array([r["q"][f][:2] for r in rows]) for f in L.FINGERS}
    plt.rcParams.update({"font.family": "serif", "font.size": 20})
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 7.6), gridspec_kw={"width_ratios": [1.15, 1.0]})
    colors = {"index": "#1f77b4", "middle": "#ff7f0e", "ideal": "#d4d4d4", "bought": "#3a6fb0", "yaw": "#2e8b57", "mcp": "#c0392b"}
    ax = axes[0]
    Yg, Mg = np.meshgrid(YAW, MCP, indexing="ij")
    ax.contourf(Yg, Mg, Z_["ideal"].astype(float), levels=[0.5, 1.5], colors=[colors["ideal"]])
    ax.contourf(Yg, Mg, Z_["bought"].astype(float), levels=[0.5, 1.5], colors=[colors["bought"]], alpha=0.85)
    ax.contour(Yg, Mg, Z_["studs_mcp_30"].astype(float), levels=[0.5], colors=[colors["mcp"]], linewidths=2.6)
    for f in L.FINGERS:
        ax.scatter(pts[f][::8, 0], pts[f][::8, 1], s=4, color=colors[f], alpha=0.45, rasterized=True)
    y0, y1 = DRIVER["yaw"]
    m0, m1 = DRIVER["mcp"]
    ax.plot([y0, y1, y1, y0, y0], [m0, m0, m1, m1, m0], "k--", lw=1.8)
    ax.set_xlim(-92, 92)
    ax.set_ylim(-32, 112)
    ax.set_xlabel("Yaw (deg)")
    ax.set_ylabel("MCP (deg)")
    ax.set_title("(a) Reach at the design's rest pose", fontsize=20)
    ax = axes[1]
    mm = Z_["profile_mcp"]
    ax.plot([m0, m0, m1, m1], [0, 85, 85, 0], "k--", lw=1.8)
    from scipy.ndimage import minimum_filter1d
    for key, col, lw, ls in (("published_v4c", "0.35", 2.2, ":"), ("ideal", "#9a9a9a", 2.6, "-"), ("bought", colors["bought"], 3.2, "-"),
                             ("studs_mcp_30", colors["mcp"], 2.4, "-")):
        prof = Z_[f"profile_{key}"]
        prof = np.where(np.isnan(prof), np.nan, minimum_filter1d(np.nan_to_num(prof, nan=-1.0), 3, mode="nearest"))  # one-column spikes
        ax.plot(mm, np.where(prof < 0, np.nan, prof), color=col, lw=lw, ls=ls)
    ax.set_xlim(-30, 120)
    ax.set_ylim(0, 95)
    ax.set_xlabel("MCP (deg)")
    ax.set_ylabel("Yaw reachable from straight (deg)")
    ax.set_title("(b) Yaw reach against MCP", fontsize=20)
    ax.grid(alpha=0.25)
    handles = [Line2D([], [], ls="--", color="k", lw=1.8, label="serial driver's range"),
               Patch(color=colors["ideal"], label="rod differential, ideal joints"),
               Patch(color=colors["bought"], label="rod differential, real joints"),
               Line2D([], [], color="0.35", lw=2.2, ls=":", label="v4c as published (did not fit)"),
               Line2D([], [], color=colors["mcp"], lw=2.4, label="ball links, AmazingHand mounting"),
               Line2D([], [], ls="", marker="o", ms=7, color=colors["index"], label="index trials"),
               Line2D([], [], ls="", marker="o", ms=7, color=colors["middle"], label="middle trials")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=17, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    MEDIA.mkdir(parents=True, exist_ok=True)
    out = MEDIA / "20261119-v4c_hw_workspace.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return str(out.relative_to(ROOT))


def driver_share(inside):
    """Share of the driver's yaw x MCP rectangle inside the region at its absolute position, and with the best whole
    shift of the module (summed-area table)."""
    ya, yb = (np.searchsorted(YAW, DRIVER["yaw"][0]), np.searchsorted(YAW, DRIVER["yaw"][1]))
    ma, mb = (np.searchsorted(MCP, DRIVER["mcp"][0]), np.searchsorted(MCP, DRIVER["mcp"][1]))
    at = float(inside[ya:yb + 1, ma:mb + 1].mean())
    nA, nF = yb - ya + 1, mb - ma + 1
    S = np.zeros((inside.shape[0] + 1, inside.shape[1] + 1))
    S[1:, 1:] = inside.cumsum(0).cumsum(1)
    W = S[nA:, nF:] - S[:-nA, nF:] - S[nA:, :-nF] + S[:-nA, :-nF]
    i, j = np.unravel_index(int(np.argmax(W)), W.shape)
    return at, float(W[i, j]) / (nA * nF), (float(YAW[i]) - DRIVER["yaw"][0], float(MCP[j]) - DRIVER["mcp"][0])


# ------------------------------------------------------------------------------------------------------------- PIP
def pip_rest():
    """PIP drive at rest, in the MCP link frame (G at the origin, the link along -z): the horn turns about x through
    (x_a, 0, z_s); pin a runs along x through the horn arm at radius r; the knuckle's offset e along k (perpendicular to
    x) and the rod's top bearing c along pin b = x cross k; the lug's spherical bearing at radius lug_r from the PIP axis,
    square to the rod at rest."""
    Q_ = HW["pip"]
    hw = HW["hardware"]
    s = 1.0
    x_a = Q_["spline_x"] + hw["disc_horn"]["face_beyond_spline"] + HW["plate_t"] + Q_["spacer"] + HW["plate_t"] / 2
    C = np.array([x_a, 0.0, Q_["spline_z"]])
    u = np.array([0.0, s, 0.0])
    v = np.cross(X, u)
    O = np.array([0.0, 0.0, -HW["proximal"]])
    Pa0 = C + Q_["r"] * u
    lug = O + np.array([Q_["lug_r"], Q_["lug_y"], 0.0])
    for _ in range(30):  # the lug arm square to the rod's projection on the flexion plane
        k0 = unit((lug - Pa0) * np.array([0.0, 1.0, 1.0]))
        pb0 = np.cross(X, k0)
        c = Q_["rod_offset_c"] * (1.0 if (pb0[1] * -s) > 0 else -1.0)  # rod toward the finger's middle plane
        Q0 = Pa0 + Q_["knuckle_e"] * k0 + c * pb0
        d = (Q0 - lug) * np.array([1.0, 0.0, 1.0])
        d /= np.linalg.norm(d)
        n = np.array([d[2], 0.0, -d[0]])
        n = n if n[0] > 0 else -n
        lug = O + Q_["lug_r"] * n + np.array([0.0, Q_["lug_y"], 0.0])
    L = float(np.linalg.norm(Q0 - lug))
    Cs0 = C + Q_["knuckle_e"] * k0 + c * pb0
    return {"C": C, "u": u, "v": v, "r": Q_["r"], "O": O, "A0": lug, "k0": k0, "pb0": pb0, "c": c, "e": Q_["knuckle_e"],
            "Pa0": Pa0, "Q0": Q0, "L": L, "x_a": x_a, "branch": rest_branch(Cs0, u, v, Q_["r"], lug, L)}


def pip_solve(pip, iters=30):
    """Horn angle th, knuckle angle psi (absolute, about x), rod points, over PIP angles (deg) in the link frame."""
    g = pip_rest()
    a = np.radians(np.asarray(pip, float) - HW["pip"]["pip_rest"])
    Ry = rot_y(a)
    A = g["O"] + np.einsum("...ij,j->...i", Ry, g["A0"] - g["O"])
    psi = np.full(A.shape[:-1], math.atan2(g["k0"][2], g["k0"][1]))
    th = np.zeros(A.shape[:-1])
    for _ in range(iters):
        k = np.stack([np.zeros_like(psi), np.cos(psi), np.sin(psi)], -1)
        pb = np.cross(X, k)
        Cs = g["C"] + g["e"] * k + g["c"] * pb  # the rod's top bearing runs on a circle about the horn axis
        th, ok = circle_sphere(Cs, g["u"], g["v"], g["r"], A, g["L"], g["branch"])
        Pa = g["C"] + g["r"] * (np.cos(th)[..., None] * g["u"] + np.sin(th)[..., None] * g["v"])
        w = A - Pa  # pin b . (A - Pa) = c, pin b = (0, -sin psi, cos psi)
        al, be = w[..., 2], -w[..., 1]
        rho = np.hypot(al, be)
        ac = np.arccos(np.clip(g["c"] / np.where(rho > 0, rho, 1.0), -1, 1))
        p1, p2 = AD.wrap(np.arctan2(be, al) + ac), AD.wrap(np.arctan2(be, al) - ac)
        f1 = np.cos(p1) * w[..., 1] + np.sin(p1) * w[..., 2]  # the knuckle points from pin a toward the lug
        f2 = np.cos(p2) * w[..., 1] + np.sin(p2) * w[..., 2]
        psi = np.where(f1 >= f2, p1, p2)
    k = np.stack([np.zeros_like(psi), np.cos(psi), np.sin(psi)], -1)
    pb = np.cross(X, k)
    Cs = g["C"] + g["e"] * k + g["c"] * pb
    th, ok = circle_sphere(Cs, g["u"], g["v"], g["r"], A, g["L"], g["branch"])
    Pa = g["C"] + g["r"] * (np.cos(th)[..., None] * g["u"] + np.sin(th)[..., None] * g["v"])
    Q = Pa + g["e"] * k + g["c"] * pb
    ok &= (np.abs(np.linalg.norm(A - Q, axis=-1) - g["L"]) + np.abs((pb * (A - Q)).sum(-1))) < 1e-6
    return {"th": th, "psi": psi, "ok": ok, "A": A, "Q": Q, "Pa": Pa, "k": k, "pb": pb}, g


def pip_range():
    """PIP angles where the rod closes, the horn stays within its limit, the PIP torque is at least one SCS0009 and the
    lug's spherical bearing tilts within its limit; the torque and tilt along the range."""
    S, g = pip_solve(PIP)
    h = 0.05
    Sp, _ = pip_solve(PIP + h)
    Sm, _ = pip_solve(PIP - h)
    dth = np.degrees(AD.wrap(Sp["th"] - Sm["th"])) / (2 * h)  # horn deg per PIP deg
    tq = np.where(np.abs(dth) > 1e-9, 1.0 / np.abs(dth), 0.0) * T_STS / T_SCS  # PIP torque in SCS0009 limits
    tilt = np.degrees(np.arccos(np.clip(np.abs(S["pb"][..., 1]), 0, 1)))  # bolt along y, the ring along pin b
    ok = S["ok"] & (np.abs(np.degrees(S["th"])) <= HW["pip"]["horn_limit_deg"]) & (tq >= 1.0) & \
        (tilt <= HW["hardware"]["spherical_bearing"]["tilt_deg"])
    i0 = int(np.argmin(np.abs(PIP - HW["pip"]["pip_rest"])))
    lo = hi = i0
    while lo > 0 and ok[lo - 1]:
        lo -= 1
    while hi < len(PIP) - 1 and ok[hi + 1]:
        hi += 1
    return {"S": S, "g": g, "ok": ok, "tq": tq, "tilt": tilt, "range": (float(PIP[lo]), float(PIP[hi])), "lo": lo, "hi": hi}


def cmd_kin():
    R = base_region()
    at, best, shift = driver_share(R["inside"])
    at0, best0, _ = driver_share(R["inside_no_tilt"])
    ins, k0 = R["inside"], R["k0"]
    reach = {}
    for m in (-15, 0, 30, 46, 60, 92):
        col = ins[:, int(np.argmin(np.abs(MCP - m)))]
        reach[str(m)] = [float(YAW[col].min()), float(YAW[col].max())] if col.any() else None
    ya, yb = np.searchsorted(YAW, DRIVER["yaw"][0]), np.searchsorted(YAW, DRIVER["yaw"][1])
    ma, mb = np.searchsorted(MCP, DRIVER["mcp"][0]), np.searchsorted(MCP, DRIVER["mcp"][1])
    win = np.zeros_like(ins)
    win[ya:yb + 1, ma:mb + 1] = True
    sel = ins & win
    chi = np.degrees(R["S"]["chi"])
    turn2 = R["turn2"]
    PR = pip_range()
    lo, hi = PR["lo"], PR["hi"]
    want = (PIP >= -25) & (PIP <= 92)
    res = {"script": "scripts/cad/v4c_hardware.py", "command": "kin", "params": "assets/cad_v4/concepts/20261118-layout_params.json v4c_hw",
           "floor_sts": round(FLOOR, 4),
           "base": {"rod_mm": [round(r["L"], 3) for r in R["rods"]],
                    "driver_share_at_position": round(at, 4), "driver_share_best_shift": round(best, 4), "best_shift_deg": shift,
                    "driver_share_ideal_bearings": round(at0, 4),
                    "yaw_reach_at_mcp_deg": reach,
                    "torque_scs0009_at_rest": {"yaw": round(float(R["cap_a"][k0]) / FLOOR, 2), "mcp": round(float(R["cap_f"][k0]) / FLOOR, 2)},
                    "torque_scs0009_min_in_window": {"yaw": round(float(R["cap_a"][sel].min()) / FLOOR, 2),
                                                     "mcp": round(float(R["cap_f"][sel].min()) / FLOOR, 2)} if sel.any() else None,
                    "torque_scs0009_median_in_window": {"yaw": round(float(np.median(R["cap_a"][sel])) / FLOOR, 2),
                                                        "mcp": round(float(np.median(R["cap_f"][sel])) / FLOOR, 2)} if sel.any() else None,
                    "bearing_tilt_deg_max": round(float(R["tilt"][:, ins].max()), 1),
                    "knuckle_pin1_turn_deg": [[round(float(chi[i][ins].min()), 1), round(float(chi[i][ins].max()), 1)] for i in range(2)],
                    "rod_turn_about_pin2_deg": [[round(float(turn2[i][ins].min()), 1), round(float(turn2[i][ins].max()), 1)] for i in range(2)],
                    "horn_deg": [[round(float(np.degrees(R["S"]["th"][i][ins]).min()), 1), round(float(np.degrees(R["S"]["th"][i][ins]).max()), 1)] for i in range(2)]},
           "pip": {"rod_mm": round(PR["g"]["L"], 3), "range_deg": PR["range"],
                   "torque_scs0009_min_m25_92": round(float(PR["tq"][want & PR["ok"]].min()), 2) if (want & PR["ok"]).any() else None,
                   "torque_scs0009_median_m25_92": round(float(np.median(PR["tq"][want & PR["ok"]])), 2) if (want & PR["ok"]).any() else None,
                   "bearing_tilt_deg_max": round(float(PR["tilt"][lo:hi + 1].max()), 1),
                   "horn_deg": [round(float(np.degrees(PR["S"]["th"][lo])), 1), round(float(np.degrees(PR["S"]["th"][hi])), 1)],
                   "lug_mm": [round(float(x), 2) for x in PR["g"]["A0"]], "x_a": round(PR["g"]["x_a"], 2)}}
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "20261119-v4c_hw_kin.json").write_text(json.dumps(res, indent=1))
    np.savez_compressed(DATA / "20261119-v4c_hw_region.npz", yaw=YAW, mcp=MCP, inside=R["inside"],
                        cap_a=R["cap_a"].astype(np.float32), cap_f=R["cap_f"].astype(np.float32))
    return res


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "kin"
    if cmd == "kin":
        print(json.dumps(cmd_kin(), indent=1))
    elif cmd == "balls":
        print(json.dumps(cmd_balls(), indent=1))
    elif cmd == "figure":
        print(cmd_figure())
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
