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
def base_rest():
    """Rest-pose geometry (mount frame) of the two base rods: horn centre C, rest direction u, v = x cross u, radius r;
    pin 1's foot P1 at the codesign's lug (h, +-w, 0) and pin 1 along y; the knuckle's direction k0 and pin 2 p20 at
    rest; the rod's lower bearing Q0 = P1 + e k0 + c p20 and the horn ball B0; the rod's length |B0 - Q0|. A knuckle that
    follows the rod makes pin 1's foot the effective lug, as in the codesign's universal joint."""
    B = HW["base"]
    rods = []
    for s in (1.0, -1.0):
        c = B["rod_offset_c"] * (1.0 if B["rod_side"] == "dorsal" else -1.0)  # the rod's plane dorsal or palmar of the knuckle
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
            Q0 = P1 + B["knuckle_e"] * k0 + c * p20  # knuckle_e < 0 puts pin 2 on the far side of pin 1 from the horn
        L = float(np.linalg.norm(B0 - Q0))
        rods.append({"s": s, "C": C, "u": u, "v": v, "r": B["r"], "P1": P1, "k0": k0, "p20": p20, "c": c,
                     "e": B["knuckle_e"], "Q0": Q0, "B0": B0, "L": L, "branch": rest_branch(C, u, v, B["r"], Q0, L)})
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


def base_solve(yaw, mcp, iters=25):
    """Horn angles th (2, ...), knuckle turns chi (2, ...), closure ok (2, ...) and the rods' points over poses."""
    rods = base_rest()
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


def base_region():
    """Reach of the base pair over the yaw x MCP grid: closure, horns within the limit, every joint at or above one
    SCS0009 (torques from finite differences of the solve), the spherical bearings within their tilt; connected to rest."""
    from scipy import ndimage
    B = HW["base"]
    Yg, Mg = np.meshgrid(YAW, MCP, indexing="ij")
    S, rods = base_solve(Yg, Mg)
    h = 0.05
    J = np.zeros(Yg.shape + (2, 2))  # d th_i / d q_j, q = (yaw, mcp) in rad
    for j, (dy, dm) in enumerate(((h, 0.0), (0.0, h))):
        Sp, _ = base_solve(Yg + dy, Mg + dm)
        Sm, _ = base_solve(Yg - dy, Mg - dm)
        J[..., :, j] = (AD.wrap(Sp["th"] - Sm["th"]) / math.radians(2 * h)).transpose(1, 2, 0)
    H = np.swapaxes(J, -1, -2)  # joint torque = H @ servo torque
    cap_a, cap_f = AD.axis_capacity(H, 0), AD.axis_capacity(H, 1)
    tilt = np.degrees(np.arccos(np.clip(np.abs(S["p2"] @ X), 0, 1)))  # spherical bearing: bolt along x, ring along p2
    d = unit(S["B"] - S["Q"])
    turn2 = np.degrees(np.arctan2((d * S["n1"]).sum(-1), (d * S["k"]).sum(-1)))  # the rod about pin 2, + toward +pin 1
    lim = B["horn_limit_deg"]
    pin2 = np.ones_like(S["ok"])
    for i, rod in enumerate(rods):  # the ear holds pin 1 on the outboard side: +pin 1 for the + rod
        out = turn2[i] * rod["s"]
        pin2[i] = (out <= HW["base"]["pin2_deg"]["toward_ear"]) & (out >= -HW["base"]["pin2_deg"]["away"])
    inside0 = S["ok"].all(0) & (np.abs(np.degrees(S["th"])) <= lim).all(0) & (cap_a >= FLOOR) & (cap_f >= FLOOR) & pin2.all(0)
    inside = inside0 & (tilt <= HW["hardware"]["spherical_bearing"]["tilt_deg"]).all(0)
    k0 = (int(np.argmin(np.abs(YAW - B["yaw_rest"]))), int(np.argmin(np.abs(MCP - B["mcp_rest"]))))
    lab, _ = ndimage.label(inside)
    inside = lab == lab[k0]
    return {"S": S, "rods": rods, "inside": inside, "inside_no_tilt": inside0, "cap_a": cap_a, "cap_f": cap_f, "tilt": tilt, "turn2": turn2,
            "H": H, "k0": k0, "Yg": Yg, "Mg": Mg}


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
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
