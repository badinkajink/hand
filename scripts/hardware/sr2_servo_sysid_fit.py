#!/usr/bin/env python3
"""Fit the SR2 hand's nine SCS0009 servos to the free-space identification of 2026-11-06 (QUEUE R36).

The lab ran lab/sysid_chain.sh on the hand as it hangs on the UR5, the arm powered off: staircases on every mcp and pip
(blocks a1, a2) and yaw (b1, b2), and yaw steps of 20 to 40 deg held 10 s, with the finger straight (c) and with its mcp at
40 deg (c40). The tables are docs/experiments/20261106-sr2_servo_sysid/data/dwells.tsv (one row a dwell, every servo's mean
angle and load) and holds_c.tsv (the moving yaw's samples during each hold). This script fits, per servo:

  controller  the load (the firmware's drive, per mille of full duty, sign of measured minus commanded) against the error
              in encoder steps (300/1024 deg): the dead band, the step at its edge and the slope
  torque      at rest the joint sticks wherever |c d + tau_g| <= f, with d = -load the drive, c the torque per unit of
              drive, tau_g the gravity torque on the joint from the cad_v3 model at the arm's pose of the SR2 night (key
              sr2_night) and f the static friction. The gravity torques reach 11 mN m on the mcp joints and 16 mN m on
              the yaws, below the yaws' friction, so they do not fix c: a least-squares c from them came out near zero.
              c is therefore the datasheet's, 2.3 kg cm of stall torque at 6 V scaled to the supply's 5.0-5.2 V, and
              each servo's friction band is the spread of c d + tau_g over its dwells (5th to 95th percentile).
  stalls      from holds_c.tsv: each yaw step's final shortfall, the time to come within one step of it, and the drift
              over the hold.

    .venv/bin/python scripts/hardware/sr2_servo_sysid_fit.py [--out JSON]

Run under resguard: the cad_v3 model loads meshes.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _paths  # noqa: E402,F401

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
S = ROOT / "docs/experiments/20261106-sr2_servo_sysid"
LAB = ROOT / "docs/experiments/20261104-sr2_hand_crosstalk/lab"
MODEL = ROOT / "assets/mjcf/experimental/20261104-cad_v3/ur5_full_k1.xml"
F = ("thumb", "index", "middle")
J = ("yaw", "mcp", "pip")
YAW_SIGN = {"thumb": 1.0, "index": -1.0, "middle": -1.0}
STEP_DEG = 300.0 / 1024.0
DUTY_PER_STEP = 15.0                    # register 21, P = 15 on all nine (registers.txt)
STALL_6V = 2.3 * 9.80665 / 100          # N m, SCS0009 datasheet stall torque at 6 V (2.3 kg cm)
PLANT = {"kp": 0.72, "forcerange": 0.18, "frictionloss": 0.0, "damping": 0.0144}   # HOS_PLANT=servo


def read_tsv(p):
    return list(csv.DictReader(open(p), delimiter="\t"))


def plans():
    """Every step's full commanded pose (sim degrees), from the lab's plan builder."""
    argv = sys.argv
    sys.argv = ["make_sysid_plan.py", "--tsv-only"]
    sys.path.insert(0, str(LAB))
    try:
        import make_sysid_plan as M
    finally:
        sys.argv = argv
    return M.BLOCKS


def servo_deg(f, j, sim):
    return YAW_SIGN[f] * sim if j == "yaw" else sim


class Gravity:
    """Gravity torque on each finger joint (N m, servo frame) of the cad_v3 hand on the UR5 at the arm's pose of the SR2
    night, for joint angles in the servo frame (deg)."""

    def __init__(self):
        import mujoco
        self.mj = mujoco
        self.m = mujoco.MjModel.from_xml_path(str(MODEL))
        self.d = mujoco.MjData(self.m)
        mujoco.mj_resetDataKeyframe(self.m, self.d, self.m.key("sr2_night").id)
        self.q0 = self.d.qpos.copy()
        self.adr = {(f, j): (self.m.jnt_qposadr[self.m.joint(f"{f}_{j}").id], self.m.jnt_dofadr[self.m.joint(f"{f}_{j}").id])
                    for f in F for j in J}

    def __call__(self, ang):
        d = self.d
        d.qpos[:] = self.q0
        d.qvel[:] = 0.0
        for (f, j), (qa, _) in self.adr.items():
            d.qpos[qa] = np.radians(servo_deg(f, j, ang[f, j]))      # the yaw's sign map is its own inverse
        self.mj.mj_forward(self.m, d)
        return {(f, j): servo_deg(f, j, -d.qfrc_bias[da]) for (f, j), (_, da) in self.adr.items()}


def dwells(blocks, grav):
    """One record a dwell and servo: command, mean angle and load, the gravity torque at the measured pose, and the
    direction the servo last moved (+1 up, -1 down, 0 never moved in this block)."""
    rows = read_tsv(S / "data/dwells.tsv")
    out, last_dir = [], {}
    for r in rows:
        b, k = r["block"], int(r["step"])
        if r["blocked"] == "1" or not r["n"] or int(r["n"]) < 2:
            continue
        st = blocks[b][k]
        prev = blocks[b][k - 1]["joints"] if k else {f: {j: 0.0 for j in J} for f in F}
        ang = {}
        for f in F:
            for j in J:
                v = r[f"{f}_{j}_deg_mean"]
                ang[f, j] = float(v) if v else np.nan
        if any(np.isnan(v) for v in ang.values()):
            continue
        tg = grav(ang)
        for f in F:
            for j in J:
                c0, c1 = servo_deg(f, j, prev[f][j]), servo_deg(f, j, st["joints"][f][j])
                key = (b, f, j)
                if c1 != c0:
                    last_dir[key] = 1 if c1 > c0 else -1
                load = r[f"{f}_{j}_load_mean"]
                out.append({"block": b, "step": k, "finger": f, "joint": j, "moving": f == st["finger"] and j == st["joint"],
                            "cmd": c1, "deg": ang[f, j], "deg_sd": float(r[f"{f}_{j}_deg_sd"] or 0),
                            "load": float(load) if load else np.nan, "load_sd": float(r[f"{f}_{j}_load_sd"] or 0),
                            "tau_g": tg[f, j], "dir": last_dir.get(key, 0), "dir_this": int(np.sign(c1 - c0))})
    return out


def controller(recs):
    """The drive against the error, from every servo at every dwell: median load per error in whole steps, the dead
    band (largest |error| whose median load is 0) and a least-squares line outside it."""
    e = np.array([(r["deg"] - r["cmd"]) / STEP_DEG for r in recs])
    lo = np.array([r["load"] for r in recs])
    ok = np.isfinite(lo)
    e, lo = e[ok], lo[ok]
    k = np.round(e).astype(int)
    table = {int(s): (int((k == s).sum()), float(np.median(lo[k == s]))) for s in np.unique(k) if (k == s).sum() >= 3}
    band = max((abs(s) for s, (n, med) in table.items() if abs(med) < 7.5), default=0)
    # the line on the medians from two steps outside the band (the band's edge mixes samples on both sides of it)
    ks = np.array([s for s in table if abs(s) >= band + 2], float)
    y = np.array([abs(table[int(s)][1]) for s in ks])
    slope, icpt = np.polyfit(np.abs(ks), y, 1)
    side = {sg: [float(v) for v in np.polyfit(np.abs(ks[np.sign(ks) == sg]), y[np.sign(ks) == sg], 1)]
            for sg in (-1, 1) if (np.sign(ks) == sg).sum() >= 3}
    return {"table": table, "dead_band_steps": int(band), "slope_per_step": float(slope), "intercept": float(icpt),
            "by_sign": {str(k): v for k, v in side.items()}, "n": int(sum(table[int(s)][0] for s in ks))}


def friction_bands(recs, c):
    """Per servo, r = c d + tau_g (N m) over every dwell: a joint at rest has |r - tau_b| <= f. The band's centre tau_b
    and half-width f from the 5th and 95th percentiles; also the medians on the dwells reached moving up and down."""
    out = {}
    for f in F:
        for j in J:
            rs = [r for r in recs if (r["finger"], r["joint"]) == (f, j) and np.isfinite(r["load"])]
            res = np.array([c * -r["load"] + r["tau_g"] for r in rs])
            up = np.array([c * -r["load"] + r["tau_g"] for r in rs if r["dir"] > 0])
            dn = np.array([c * -r["load"] + r["tau_g"] for r in rs if r["dir"] < 0])
            p5, p95 = np.percentile(res, [5, 95])
            out[f"{f}_{j}"] = {"n": len(rs), "centre_Nm": float((p5 + p95) / 2), "half_width_Nm": float((p95 - p5) / 2),
                               "max_abs_Nm": float(np.max(np.abs(res))),
                               "median_up_Nm": float(np.median(up)) if len(up) else None,
                               "median_down_Nm": float(np.median(dn)) if len(dn) else None,
                               "tau_g_max_Nm": float(np.max(np.abs([r["tau_g"] for r in rs])))}
    return out


def stalls():
    """Each yaw step of c and c40: the final shortfall (deg, measured minus commanded, servo frame), the time from the
    command to come within one step of it and stay, and the drift between 2 s and the end."""
    rows = read_tsv(S / "data/holds_c.tsv")
    groups = {}
    for r in rows:
        groups.setdefault((r["block"], int(r["step"])), []).append(r)
    out = []
    for (b, k), rs in sorted(groups.items()):
        t = np.array([float(r["t_from_cmd_s"]) for r in rs])
        q = np.array([float(r["deg"]) if r["deg"] else np.nan for r in rs])
        ld = np.array([float(r["load"]) if r["load"] else np.nan for r in rs])
        cmd = float(rs[0]["cmd_servo_deg"])
        ok = np.isfinite(q)
        t, q, ld = t[ok], q[ok], ld[ok]
        if len(t) < 5:
            continue
        final = float(np.median(q[t >= t.max() - 2.0]))
        far = np.where(np.abs(q - final) > STEP_DEG * 1.01)[0]
        t_in = float(t[far[-1] + 1]) if len(far) and far[-1] + 1 < len(t) else (0.0 if not len(far) else None)
        late = t >= 2.0
        out.append({"block": b, "step": k, "finger": rs[0]["finger"], "dir": rs[0]["dir"], "cmd_plan": float(rs[0]["cmd_plan_deg"]),
                    "cmd_servo": cmd, "short_deg": final - cmd, "t_within_1step_s": t_in,
                    "drift_deg": float(q[late][-1] - q[late][0]) if late.sum() > 1 else None,
                    "load_final": float(np.median(ld[t >= t.max() - 2.0])), "n": int(len(t))})
    return out


def loops(recs):
    """Staircase loop width per servo and command: the moving joint's mean angle on the way down minus on the way up,
    per block (deg)."""
    out = {}
    for blk in ("a1", "a2", "b1", "b2"):
        by = {}
        for r in recs:
            if r["block"] == blk and r["moving"] and r["dir_this"] != 0:
                by.setdefault((r["finger"], r["joint"], r["cmd"]), {})[r["dir_this"]] = r["deg"]
        w = {}
        for (f, j, cmd), d in by.items():
            if 1 in d and -1 in d:
                w.setdefault(f"{f}_{j}", []).append((cmd, d[-1] - d[1]))
        out[blk] = {k: {"median": float(np.median([x for _, x in v])), "max": float(max(x for _, x in v)),
                        "n": len(v), "by_cmd": v} for k, v in w.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(S / "data/20261106-sr2_servo_sysid_fit.json"))
    a = ap.parse_args()
    blocks = plans()
    grav = Gravity()
    recs = dwells(blocks, grav)
    ctl = controller(recs)
    volts = [float(x.split()[-3]) / 10 for x in open(S / "data/registers.txt") if "present voltage" in x]
    stall = STALL_6V * float(np.mean(volts)) / 6.0
    c = stall / 1000.0                                  # N m per unit of drive (per mille of full duty)
    k_rad = c * ctl["slope_per_step"] / np.radians(STEP_DEG)
    zero = grav({(f, j): 0.0 for f in F for j in J})
    res = {"made_by": "scripts/hardware/sr2_servo_sysid_fit.py", "model": str(MODEL.relative_to(ROOT)), "key": "sr2_night",
           "tau_g_at_zero_Nm": {f"{f}_{j}": float(v) for (f, j), v in zero.items()}, "volts_torque_off": volts,
           "controller": ctl, "c_Nm_per_unit": c, "stall_Nm": stall, "stiffness_Nm_per_rad": float(k_rad),
           "punch_Nm": float(c * (ctl["intercept"] + ctl["slope_per_step"] * (ctl["dead_band_steps"] + 1))),
           "bands": friction_bands(recs, c), "plant": PLANT, "loops": loops(recs), "stalls": stalls(),
           "records": [{k: (v if not isinstance(v, (np.floating,)) else float(v)) for k, v in r.items()} for r in recs]}
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    print(f"wrote {a.out}: {len(recs)} records")
    print(f"gravity at the zero pose (mN m): " + ", ".join(f"{k} {1000 * v:+.2f}" for k, v in res["tau_g_at_zero_Nm"].items()))
    print(f"controller: dead band {ctl['dead_band_steps']} steps, drive {ctl['intercept']:.1f} + {ctl['slope_per_step']:.2f}"
          f" per step (medians, n {ctl['n']})")
    print(f"scale: stall {stall:.3f} N m at {np.mean(volts):.2f} V; stiffness {k_rad:.3f} N m/rad (plant kp {PLANT['kp']}); "
          f"first drive outside the band {res['punch_Nm'] * 1000:.1f} mN m")
    for k, v in res["bands"].items():
        print(f"  {k:11s} n {v['n']:4d} band centre {1000 * v['centre_Nm']:+6.2f} half-width {1000 * v['half_width_Nm']:5.2f} "
              f"mN m; median up {1000 * (v['median_up_Nm'] or 0):+6.2f} down {1000 * (v['median_down_Nm'] or 0):+6.2f}; "
              f"|tau_g| max {1000 * v['tau_g_max_Nm']:5.2f}")


if __name__ == "__main__":
    main()
