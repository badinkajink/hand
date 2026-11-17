"""The pick-up by controlled slip in simulation, against the hand's first pick-up of 2026-11-16 (15:43-15:47; owner,
2026-11-16 21:4x: "can we replicate this for the sphere pickup from dropped? ... this is where it should demonstrate the
most difference"). The plan of the hand's run (plans/20261116-pinch_pickup_thumb_middle_plan.json: layout (105, 70), the
thumb against the middle along x_P, contacts 3 deg above the ball's equator, the index curled away) on the calibrated
servo (rev_clutch2: gears the load cannot backdrive, stiff joint friction, index and middle yaw gains 0.35, the bench's
kp 0.55) with the real object (ball, stem and block) lying on a table, its stem along -y_P:
  settle   1.0 s, the fingers at the plan's open pose, the object dropped from 0.5 mm above the table;
  close    1.0 s ramp to the plan's touch pose; squeeze 0.5 s ramp to its grip (PIPs +8 deg), then 1.0 s;
  lift     the table down 64 mm at 3 mm/s (the hand's lift, as relative motion: the hand stays, the table goes), 1.0 s;
  slip     the PIPs' squeeze lowered from 7.5 to 0 deg in 0.5 deg steps, 1.5 s each, as pinch_pickup.py slip does.
Per contact variant (the scene: 1 mm sphere pads or one tip mesh with point contact; impratio 1 or 100) the object's
tilt (its axis from vertical), the table's drop, the PIPs and the pinch forces every 20 ms.

    HOS_PLANT=kp0.55_kv0_fr0.18_fl0_dp0.011 OMP_NUM_THREADS=1 .venv/bin/python \\
        logs/20261116-sim_real_reversal/pickup_sim.py OUT.json [--variants pads_soft point_soft ...]
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path.home() / "Programs/hand"
D = ROOT / "logs/20261116-sim_real_reversal"
sys.path.insert(0, str(D)); sys.path.insert(0, str(ROOT / "scripts/hardware")); sys.path.insert(0, str(ROOT / "scripts"))
PLAN = ROOT / "docs/experiments/20261115-hom_bench_trials/plans/20261116-pinch_pickup_thumb_middle_plan.json"
SCENES = {"pads_soft": "sph040_stem_block_tape0_imp1.xml", "pads_stiff": "sph040_stem_block_tape0_imp100.xml",
          "point_soft": "sph040_stem_block_pt_imp1.xml", "point_stiff": "sph040_stem_block_pt_mirror.xml",
          "pads_imp3": "sph040_stem_block_tape0_imp3.xml", "pads_imp10": "sph040_stem_block_tape0_imp10.xml",
          "point_imp3": "sph040_stem_block_pt_imp3.xml", "point_imp10": "sph040_stem_block_pt_imp10.xml",
          # MuJoCo's noslip solver after the main one (30 iterations): the soft friction's creep removed
          "pads_soft_ns": "sph040_stem_block_tape0_imp1_ns30.xml", "pads_stiff_ns": "sph040_stem_block_tape0_imp100_ns30.xml",
          "point_soft_ns": "sph040_stem_block_pt_imp1_ns30.xml", "point_stiff_ns": "sph040_stem_block_pt_mirror_ns30.xml"}
FINGERS = ("thumb", "index", "middle")
MU = 1.0                                   # the plant's sliding friction (--mu)
FILM = None                                # --film DIR: the hang's 10 s filmed, 25 frames a second, one mp4 per variant
LABEL = {"pads_stiff_ns": "1 mm sphere pads, friction that does not creep", "pads_soft": "1 mm sphere pads, the twist's "
         "soft contact", "point_stiff_ns": "point contact", "point_soft": "point contact, soft"}
DT = 0.02
LIFT_MM, LIFT_V = 64.0, 0.003
SQUEEZE, STEP, STEP_S = 8.0, 0.5, 1.5


def table_scene(name):
    """The scene with its post as a mocap table, 30 cm across."""
    x = (D / "scenes" / name).read_text()
    m = re.search(r'<body name="post" pos="[^"]*"><geom name="post" type="box" size="[^"]*"', x)
    assert m, name
    x = x[:m.start()] + '<body name="post" mocap="true" pos="0 0 0"><geom name="post" type="box" size="0.15 0.15 0.004"' \
        + x[m.end():]
    out = D / "scenes" / (Path(name).stem + "_table.xml")
    out.write_text(x)
    return out


HAND_P_MM = (-1.66, -17.19, -84.0)   # the ball's centre in P at the hand's lift start (lift telemetry, 15:44:29)
HANG = {"tilt": 42.0, "hold_s": 1.5, "held_s": 10.0}  # the hand's tilt after the lift, held in the pinch about 10 s
                                                     # (the end of the lift through squeeze 7.5-4.5)


def run_hang(variant):
    """The object pinched in the air at the hand's measured ball centre and its post-lift tilt (42 deg), held in place
    for HANG hold_s while the squeeze builds from the touch pose to the grip; then free: held, and the slip steps."""
    return run(variant, hang=True)


def run(variant, hang=False):
    import rev_clutch2 as RC2
    RC2.STIFF, RC2.KP_YAW_IM, RC2.BREAKAWAY = True, 0.35, 0.0
    import rev_rescore as RR
    plan = json.loads(PLAN.read_text())
    h = plan["meta"]["hom"]
    poses = {p["name"]: {f: np.radians([p["joints"][f][j] for j in ("yaw", "mcp", "pip")]) for f in FINGERS}
             for p in plan["poses"]}
    d6 = h["d6_P_mm"]
    mxy = {"thumb": (d6[0] / 1e3, d6[1] / 1e3), "index": (d6[2] / 1e3, d6[3] / 1e3), "middle": (d6[4] / 1e3, d6[5] / 1e3)}
    setup = {"palm": np.asarray(h["setup"]["palm_m"], float), "quat": np.asarray(h["setup"]["quat"], float)}
    path = table_scene(SCENES[variant])
    GP = RC2.clutch_class()
    p = GP(path, setup, mxy, {**RR.SERVER, "mu": MU}, 0.0, c=0.6)
    m, d = p.m, p.d
    # the ball's centre below the pinch centre by r sin(lat); the object lying with its stem along -y_P, 85 deg tilted
    r, lat = 0.020, math.radians(h["pinch"]["lat_deg"])
    c = np.asarray(setup["palm"]) + np.asarray(h["pinch"]["c_P_mm"]) / 1e3 - np.array([0.0, 0.0, r * math.sin(lat)])
    if hang:
        c = np.asarray(setup["palm"]) + np.asarray(HAND_P_MM) / 1e3
    # the object at rest on the ball's bottom and the block's far rim (86.1 deg for the 20 mm ball; the hand's taped ball
    # lay at 85.5 deg), the table at the ball's bottom
    stem, bh, rb = 0.03689, 0.017, 0.015
    lo, hi = math.radians(80.0), math.radians(90.0)
    for _ in range(60):
        th = 0.5 * (lo + hi)
        if (r + stem + bh) * math.cos(th) + rb * math.sin(th) > r:
            lo = th
        else:
            hi = th
    if hang:
        th = math.radians(HANG["tilt"])
    quat = np.array([math.cos((math.pi / 2 - th) / 2), math.sin((math.pi / 2 - th) / 2), 0.0, 0.0])
    z_top0 = c[2] - r - (0.5 if hang else 0.0)                # in the hang the table is half a metre down
    mid = int(m.body_mocapid[m.body("post").id])
    d.mocap_pos[mid] = (0.0, c[1] - 0.04, z_top0 - 0.004)
    p.reset_to(poses["open"], c, quat)
    d.mocap_pos[mid] = (0.0, c[1] - 0.04, z_top0 - 0.004)

    out = {"variant": variant, "scene": path.name, "rows": []}
    t = 0.0

    def tilt():
        R = np.empty(9)
        import mujoco
        mujoco.mju_quat2Mat(R, d.qpos[p.qa + 3:p.qa + 7])
        return math.degrees(math.acos(float(np.clip(R.reshape(3, 3)[2, 1], -1.0, 1.0))))

    frames = []
    if FILM:
        import mujoco
        g = m.vis.global_
        g.offwidth, g.offheight = max(int(g.offwidth), 640), max(int(g.offheight), 480)
        rend = mujoco.Renderer(m, 480, 640)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = c - np.array([0.0, 0.0, 0.025])
        cam.distance, cam.azimuth, cam.elevation = 0.20, 180.0, -8.0      # along the pinch axis, x_P, from +x
        for gg in range(m.ngeom):                                            # the finger links see-through
            bn = m.body(m.geom_bodyid[gg]).name
            if any(bn.startswith(f + "_") for f in FINGERS) and "_pad" not in (m.geom(gg).name or ""):
                m.geom_rgba[gg, 3] = min(float(m.geom_rgba[gg, 3]), 0.35)

    def period(phase, q_cmd, drop_mm, sq):
        nonlocal t
        p.set_targets(q_cmd)
        d.mocap_pos[mid][2] = z_top0 - 0.004 - drop_mm / 1e3
        t += DT
        p.advance(t)
        if FILM and phase == "held" and len(out["rows"]) % 2 == 0:
            import hom_contact_rig as HR
            rend.update_scene(d, cam)
            img = HR.annotate(rend.render().copy(), LABEL.get(variant, variant), size=17)
            frames.append(HR.annotate_bottom(img, f"{t - HANG['hold_s']:4.1f} s after release, tilt {tilt():4.1f} deg",
                                             size=16))
        cop, Ftot = p.contacts()
        out["rows"].append({"t": round(t, 3), "phase": phase, "drop_mm": round(drop_mm, 2), "squeeze": sq,
                            "tilt": round(tilt(), 2), "z_mm": round(1e3 * float(d.qpos[p.qa + 2]), 2),
                            "pip": [round(math.degrees(float(d.qpos[p.qadr[f][2]])), 2) for f in ("thumb", "middle")],
                            "F": [round(float(cop[f][1]) if f in cop else 0.0, 3) for f in ("thumb", "middle")]})

    def ramp(a, b, s, phase, sq):
        n = max(1, int(round(s / DT)))
        for k in range(1, n + 1):
            u = k / n
            period(phase, {f: (1 - u) * a[f] + u * b[f] for f in FINGERS}, 0.0, sq)

    def squeezed(sq):
        q = {f: poses["touch"][f].copy() for f in FINGERS}
        for f in ("thumb", "middle"):
            q[f][2] += math.radians(sq)
        return q

    try:
        if hang:
            p.reset_to(poses["touch"], c, quat)
            p.hold_s = HANG["hold_s"]
            ramp(poses["touch"], squeezed(SQUEEZE), 1.0, "squeeze (held)", SQUEEZE)
            while t < HANG["hold_s"] - 1e-9:
                period("squeeze (held)", squeezed(SQUEEZE), 0.0, SQUEEZE)
            for _ in range(int(HANG["held_s"] / DT)):
                period("held", squeezed(SQUEEZE), 0.0, SQUEEZE)
            sq = SQUEEZE - STEP
            while sq > -1e-9:
                for _ in range(int(STEP_S / DT)):
                    period("slip", squeezed(sq), 0.0, round(sq, 2))
                sq -= STEP
            raise StopIteration
        for _ in range(int(1.0 / DT)):
            period("settle", poses["open"], 0.0, None)
        ramp(poses["open"], poses["touch"], 1.0, "close", 0.0)
        ramp(poses["touch"], squeezed(SQUEEZE), 0.5, "squeeze", SQUEEZE)
        for _ in range(int(1.0 / DT)):
            period("squeeze", squeezed(SQUEEZE), 0.0, SQUEEZE)
        n = int(round(LIFT_MM / 1e3 / LIFT_V / DT))
        for k in range(1, n + 1):
            period("lift", squeezed(SQUEEZE), LIFT_MM * k / n, SQUEEZE)
        for _ in range(int(1.0 / DT)):
            period("held", squeezed(SQUEEZE), LIFT_MM, SQUEEZE)
        sq = SQUEEZE - STEP
        while sq > -1e-9:
            for _ in range(int(STEP_S / DT)):
                period("slip", squeezed(sq), LIFT_MM, round(sq, 2))
            sq -= STEP
    except StopIteration:
        pass
    except Exception as e:                                       # noqa: BLE001  a diverged state
        out["error"] = repr(e)[:200]
    if FILM and frames:
        import hom_contact_rig as HR
        Path(FILM).mkdir(parents=True, exist_ok=True)
        HR.write_mp4(frames, Path(FILM) / f"pickup_hang_{variant}.mp4", fps=25)
        rend.close()
    rows = out["rows"]
    lift = [x for x in rows if x["phase"] == "lift"]
    out["summary"] = {"tilt_start": rows[0]["tilt"] if rows else None,
                      "tilt_after_squeeze": next((x["tilt"] for x in reversed(rows) if x["phase"] == "squeeze"), None),
                      "tilt_end_lift": lift[-1]["tilt"] if lift else None,
                      "tilt_held": next((x["tilt"] for x in reversed(rows) if x["phase"] == "held"), None),
                      "held_at": {f"{k:g}": next((x["tilt"] for x in rows if x["phase"] == "held"
                                                  and x["t"] >= HANG["hold_s"] + k - 1e-6), None) for k in (1, 3, 6, 10)},
                      "slip": {f"{s:.1f}": next((x["tilt"] for x in reversed(rows)
                                                 if x["phase"] == "slip" and abs(x["squeeze"] - s) < 1e-6), None)
                               for s in np.arange(SQUEEZE - STEP, -1e-9, -STEP)},
                      "dropped": bool(rows and rows[-1]["z_mm"] < rows[0]["z_mm"] - 40.0)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--variants", nargs="+", default=list(SCENES))
    ap.add_argument("--hang", action="store_true", help="the object pinched in the air at 42 deg, then the slip")
    ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--film", default=None, help="a folder for the hang's films (with --hang)")
    a = ap.parse_args()
    global MU, FILM
    MU, FILM = a.mu, a.film
    import multiprocessing as mp
    with mp.get_context("fork").Pool(len(a.variants)) as pool:
        res = pool.map(run_hang if a.hang else run, a.variants)
    Path(a.out).write_text(json.dumps(res) + "\n")
    for r in res:
        s = r["summary"]
        print(f"{r['variant']:12s} tilt start {s['tilt_start']} squeezed {s['tilt_after_squeeze']} end of lift "
              f"{s['tilt_end_lift']} held {s['tilt_held']} | slip " + " ".join(f"{k}:{v}" for k, v in s["slip"].items())
              + (f" | ERROR {r['error']}" if r.get("error") else "") + (" | DROPPED" if s["dropped"] else ""), flush=True)


if __name__ == "__main__":
    main()
