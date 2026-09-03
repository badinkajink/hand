"""The bench we actually have: one flat table, and 100 mm of hardware between wrist and hand.

Every arm result before this ran a UR5e standing on a 300 mm pedestal with the palm plate bolted
straight to the tool flange. Neither is true. The robot is bolted to the same flat table the work
sits on, and between the flange and the first yaw joint there is a coupling, a servo bank and its
wiring -- about 100 mm of it. Both are geometry, both are free to specify in a scene builder, and
until they are in the scene an arm result is a claim about a robot nobody owns.

  table     base z 0 .. 300 mm                       does the raised table buy anything?
  stack     0 .. 150 mm between flange and palm      what does the real wrist stack cost?
  mass      that stack at 0 .. 2x density            is the cost its REACH or its WEIGHT?
  preload   stack x carry squeeze, plane and seat    and is the cost RETUNABLE?
  place     where on the table the robot stands      is there room for one table?
  place2    that position x the preload, in the seat  can standing closer buy the stack back?
  torque    a thread's resisting torque, 0..20 mN m  what load can the primitive drive?

Two modes throughout, because they are the two answers to "how do you get from the reorientation
grasp to the gaiting grasp": RELEASE (let go, re-pose the palm, retake) and RELAY (walk one pad
at a time, never fewer than two on the tool). The 2026-09-03 handover study found them 6/6 and
6/6 on a floating palm. The wrist is where they come apart.

    uv run --extra rl --extra arm python scripts/real_v1_bench_study.py \
        --out docs/experiments/20260904-real_v1_bench
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

RUN = ROOT / "results/phase1/real_v1/rv05_manual_stored"
SCREW = ROOT / "docs/experiments/20260903-real_v1_screw/screw_a45_x40_y-11.json"

# The published chain cell, unchanged (docs/experiments/20260903-real_v1_chain). Nothing about
# the hand, the grasp or the gait is retuned here; the only new variables are where the robot
# stands and what is bolted between it and the hand.
BASE = dict(obj="screwdriver_medium", lift=0.10, angle_deg=-90.0, axis_k=0.25, turn_steps=550,
            budget=0.5, hold_steps=500, gap=0.002, carry_squeeze=0.0,
            descend_iters=1, descend_steps=400, airgrip="cradle", stand_order="ground",
            centre_x=0.004, grip_depth=0.050, squeeze=0.002, release_mm=6.0,
            twist_steps=120, move_steps=60)
# The screw study's own cell: a cone needs a deeper press and a longer carry than a plane does.
SEAT_BASE = dict(carry_squeeze=0.0003, press_mm=10.0, transport_steps=300)
MODES = {"relay": dict(reindex="relay", relay_gait=True),
         "release": dict(reindex="full", relay_gait=False)}


def _tag(base, stack, density, src, pgc=False) -> str:
    return (f"{'screw' if src else 'plane'}_x{base[0]:+.3f}_y{base[1]:+.3f}_z{base[2]:.3f}"
            f"_s{stack * 1000:.0f}_d{density:.0f}{'_pgc' if pgc else ''}").replace(".", "p")


def scene_for(cache: Path, base, stack: float, density: float = 700.0,
              src: Path | None = None, pgc: bool = False) -> tuple:
    """Build (or reuse) the arm scene + IK model for one bench geometry.

    Every build runs the eight-branch home solve, which is ~10 s, so the cache is not an
    optimisation -- without it a sweep spends longer choosing elbow configurations than it does
    running chains.
    """
    cache.mkdir(parents=True, exist_ok=True)
    t = _tag(base, stack, density, src, pgc)
    sc, ik = cache / f"{t}.xml", cache / f"{t}_ik.xml"
    if sc.exists() and ik.exists():
        return sc, ik
    cmd = [sys.executable, str(ROOT / "scripts/build_real_v1_arm_scene.py"),
           f"--base={base[0]},{base[1]},{base[2]}", "--wrist-stack", str(stack),
           "--stack-density", str(density), "--out", str(sc), "--ik-out", str(ik)]
    if pgc:
        cmd.append("--payload-gravcomp")
    if src is not None:
        cmd += ["--scene", str(src)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        for f in (sc, ik):
            f.unlink(missing_ok=True)
        return None, (p.stdout + p.stderr).strip().splitlines()[-1] if p.stderr else "build failed"
    return sc, ik


def droop(sc: Path, ik: Path) -> dict:
    """Palm error at the top of a 100 mm lift, holding the hand's own weight.

    Deterministic, unlike the chain, and it is the quantity the seam was already specced
    against: 0.97 mm of droop chains 6/6 and 1.59 mm chains 1/6. Every geometry in this study
    gets one, so a noisy pass rate can be read against a number that does not move.
    """
    import mujoco
    import numpy as np
    import palm_driver as pd
    m = mujoco.MjModel.from_xml_path(str(sc))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("open_ik").id)
    d.ctrl[:] = m.key_ctrl[m.key("open_ik").id]
    palm = pd.make(m, d, ik)
    mujoco.mj_forward(m, d)
    Rc, pc = palm.cmd_pose()
    u0 = palm.read()
    u1 = palm.solve(Rc, pc + np.array([0.0, 0.0, 0.10]))[0]
    for k in range(200):
        palm.write(u0 + (u1 - u0) * (k + 1) / 200)
        mujoco.mj_step(m, d)
    for _ in range(800):
        mujoco.mj_step(m, d)
    Rc, pc = palm.cmd_pose()
    Ra = d.body("palm_pose").xmat.reshape(3, 3)
    pa = d.body("palm_pose").xpos
    return {"droop_mm": round(float(np.linalg.norm(pa - pc)) * 1000, 3),
            "droop_deg": round(float(np.degrees(np.arccos(np.clip(
                (np.trace(Ra.T @ Rc) - 1) / 2, -1, 1)))), 3),
            "payload_g": round(float(m.body_subtreemass[m.body("palm_pose").id]) * 1000, 1)}


def _cell(kw):
    import probe_real_v1_chain as C
    tag, run = kw.pop("_tag"), Path(kw.pop("_run"))
    meta = kw.pop("_meta", {})
    kw.pop("_meta_geom", None)
    try:
        r = C.chain(run, **kw)
    except Exception as exc:
        return {"arm": tag, "run": run.name, "error": repr(exc), "ok": False, **meta}
    r["arm"] = tag
    r.update(meta)
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=ROOT / "results/phase1/real_v1/bench_scenes")
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--cycles", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--arms",
                    default="table,stack,mass,preload,place,place2,torque,payload,slip,wrist")
    ap.add_argument("--droop-all", action="store_true",
                    help="measure palm droop for every scene in the cache and merge those rows "
                         "in, without running any chains")
    ap.add_argument("--append", type=Path, default=None,
                    help="merge these rows into an existing bench_study.json instead of "
                         "replacing it, keyed by arm name")
    args = ap.parse_args()
    arms = set() if args.droop_all else set(args.arms.split(","))
    screw = json.loads(SCREW.read_text())
    jobs, unreachable = [], []

    def add(tag, sc, ik, reps, meta, **kw):
        for rep in range(reps):
            jobs.append({"_tag": tag, "_run": str(RUN), "_meta": dict(meta),
                         "_meta_geom": {k: meta[k] for k in
                                        ("base_x", "base_y", "base_z", "stack_mm",
                                         "stack_density", "seat", "pgc")},
                         # The carry is chaotic at the 1e-6 level, so a geometry judged at one
                         # seed is a claim about one carry, not about the geometry.
                         "seed": rep, "jitter": 0.0005, "cycles": args.cycles,
                         "arm_ik": ik, "scene_path": sc, **BASE, **kw})

    def geom(tag, base, stack, density=700.0, src=None, modes=MODES, reps=None, pgc=False,
             **kw):
        sc, ik = scene_for(args.cache, base, stack, density, src, pgc)
        meta0 = {"base_x": base[0], "base_y": base[1], "base_z": base[2],
                 "stack_mm": stack * 1000, "stack_density": density,
                 "seat": src is not None, "pgc": pgc}
        if sc is None:
            unreachable.append({"arm": tag, **meta0, "why": ik})
            print(f"  {tag} {meta0}: NO HOME POSE")
            return
        for mode, mk in modes.items():
            add(tag, sc, ik, reps or args.reps, {**meta0, "mode": mode}, **mk, **kw)

    seat_src = Path(screw["scene"])
    seat_kw = dict(SEAT_BASE, place_xy=screw["socket_xy"], seat_z=screw["seat_z"],
                   tip_len=screw.get("tip_len", 0.0))

    if "table" in arms:
        for z in (0.0, 0.10, 0.20, 0.30):
            geom("table", (-0.50, 0.0, z), 0.0)
    if "stack" in arms:
        for s in (0.0, 0.025, 0.050, 0.075, 0.100, 0.125, 0.150):
            geom("stack", (-0.50, 0.0, 0.0), s)
    if "mass" in arms:
        for rho in (0.0, 350.0, 700.0, 1400.0):
            geom("mass", (-0.50, 0.0, 0.0), 0.100, rho)
    if "preload" in arms:
        # The carry ends with the tool CRADLED at a fraction of a millimetre of interference,
        # not gripped, and `carry_squeeze` is how much of a squeeze is commanded on top. On a
        # floating palm the published value is 0 on a plane and 0.3 mm in a seat. An arm is a
        # spring, and 100 mm of stack lengthens the lever and adds 0.35 kg to the end of it, so
        # the SAME command buys less force. This grid asks whether that is all the stack costs.
        for s in (0.0, 0.050, 0.100, 0.150):
            for cs in (0.0, 0.0003, 0.0008, 0.0015, 0.0025):
                geom("preload_plane", (-0.50, 0.0, 0.0), s, carry_squeeze=cs,
                     modes={"relay": MODES["relay"]})
            for cs in (0.0003, 0.0008, 0.0015, 0.0020, 0.0030):
                geom("preload_seat", (-0.50, 0.0, 0.0), s, src=seat_src,
                     **{**seat_kw, "carry_squeeze": cs})
    if "place" in arms:
        for x in (-0.35, -0.425, -0.50, -0.575, -0.65):
            for y in (0.0, 0.20):
                # relay only: it is the mode the wrist is hard on, so it is the one that says
                # whether a standing position is usable rather than merely reachable.
                geom("place", (x, y, 0.0), 0.100, modes={"relay": MODES["relay"]})
    if "place2" in arms:
        for x in (-0.35, -0.425, -0.50):
            for cs in (0.0003, 0.0008, 0.0015, 0.0020):
                geom("place2", (x, 0.0, 0.0), 0.100, src=seat_src,
                     **{**seat_kw, "carry_squeeze": cs})
    if "payload" in arms:
        # A UR5e is told its payload and holds position under it. The whole droop story above
        # is measured on a model with no payload feedforward at all, so this arm asks whether
        # the wrist stack costs anything on the robot that actually exists.
        for s in (0.0, 0.050, 0.100, 0.150):
            for cs in (0.0003, 0.0008, 0.0015):
                geom("payload", (-0.50, 0.0, 0.0), s, src=seat_src, pgc=True,
                     **{**seat_kw, "carry_squeeze": cs})
    if "slip" in arms:
        # PREDICT / CONTROL / MEASURE the settle. `hold_steps` is how much of the slip you
        # allow; `turn_squeeze` is how much of it you suppress; the `turned` seam is where it
        # is measured from.
        for tsq in (0.0, 0.0005, 0.0010, 0.0015, 0.0020):
            for hs in (0, 150, 300, 500, 900):
                geom("slip", (-0.50, 0.0, 0.0), 0.100, src=seat_src, pgc=True,
                     modes={"relay": MODES["relay"]}, turn_squeeze=tsq, hold_steps=hs,
                     **{**seat_kw, "carry_squeeze": 0.0003})
        # Where the turn's residual tilt comes from: the same sweep with the payload
        # UNcompensated says how much of it is the wrist sagging rather than the fingers.
        for pg in (True, False):
            for hs in (0, 150, 300, 500, 900):
                geom("wrist", (-0.50, 0.0, 0.0), 0.100, src=seat_src, pgc=pg,
                     modes={"relay": MODES["relay"]}, hold_steps=hs,
                     **{**seat_kw, "carry_squeeze": 0.0003})
    if "torque_grip" in arms:
        # The drivable load turned out to depend on the grip, not on the handover schedule, so
        # the two have to be crossed rather than reported separately.
        for tq in (0.0, 0.008, 0.016):
            for cs in (0.0003, 0.0008):
                geom("torque_grip", (-0.50, 0.0, 0.0), 0.100, src=seat_src, screw_torque=tq,
                     pgc=True, modes={"relay": MODES["relay"]},
                     **{**seat_kw, "carry_squeeze": cs})
    if "torque" in arms:
        # On the corrected model -- payload compensated, full stack, published carry grip --
        # so the load is measured on the robot that exists rather than on one that sags.
        for tq in (0.0, 0.004, 0.008, 0.012, 0.016, 0.020):
            geom("torque", (-0.50, 0.0, 0.0), 0.100, src=seat_src, screw_torque=tq, pgc=True,
                 **{**seat_kw, "carry_squeeze": 0.0003})

    print(f"{len(jobs)} cells on {args.workers} workers, {len(unreachable)} geometries unreachable")
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(_cell, jobs, chunksize=1)):
            rows.append(r)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(jobs)}")
    # One droop reading per distinct geometry: deterministic, and the number the seam was
    # already specced against.
    dro, seen = [], set()
    if args.droop_all:
        for sc in sorted(args.cache.glob("*.xml")):
            if sc.stem.endswith("_ik"):
                continue
            parts = dict(x=0.0, y=0.0, z=0.0, s=0.0, d=0.0)
            for seg in sc.stem.split("_")[1:]:
                parts[seg[0]] = float(seg[1:].replace("p", "."))
            dro.append({"scene": sc.name, "base_x": parts["x"], "base_y": parts["y"],
                        "base_z": parts["z"], "stack_mm": parts["s"],
                        "stack_density": parts["d"], "seat": sc.stem.startswith("screw"),
                        **droop(sc, sc.with_name(sc.stem + "_ik.xml"))})
            print(f"   droop {sc.stem[:46]:48} {dro[-1]['droop_mm']:6.3f} mm  "
                  f"{dro[-1]['payload_g']:6.1f} g")
    for j in jobs:
        key = (str(j["scene_path"]), str(j["arm_ik"]))
        if key in seen:
            continue
        seen.add(key)
        g = {k: j["_meta_geom"][k] for k in ("base_x", "base_y", "base_z", "stack_mm",
                                             "stack_density", "seat", "pgc")}
        dro.append({"scene": Path(key[0]).name, **g, **droop(Path(key[0]), Path(key[1]))})
    for row in dro:
        print(f"   droop {row['scene'][:46]:48} {row['droop_mm']:6.3f} mm  "
              f"{row['payload_g']:6.1f} g")

    args.out.mkdir(parents=True, exist_ok=True)
    dst = args.out / "bench_study.json"
    if args.append and args.append.exists():
        old = json.loads(args.append.read_text())
        keep = [r for r in old["rows"] if r.get("arm") not in {r2["arm"] for r2 in rows}]
        rows = keep + rows
        dro = [r for r in old.get("droop", []) if r["scene"] not in {d["scene"] for d in dro}] + dro
        unreachable = old.get("unreachable", []) + unreachable
    # The per-finger station/radius dicts inside every seam are ~85% of the file and nothing
    # downstream reads them; the scalar seam fields are what the page and the log quote.
    seam_keep = ("phase", "t", "cos", "tilt_deg", "z", "pad_contacts", "pad_force_N",
                 "ground_contacts", "ground_force_N", "spin_deg")
    for r in rows:
        if "seams" in r:
            r["seams"] = [{k: s[k] for k in seam_keep if k in s} for s in r["seams"]]
    dst.write_text(json.dumps({"rows": rows, "unreachable": unreachable, "droop": dro},
                              separators=(",", ":")))
    print(f"-> {args.out / 'bench_study.json'}")

    import statistics as st
    for arm in sorted({r["arm"] for r in rows}):
        sub = [r for r in rows if r["arm"] == arm]
        print(f"\n== {arm}  ({len(sub)} cells)")
        kf = lambda r: (r.get("base_x"), r.get("base_y"), r.get("base_z"), r.get("stack_mm"),
                        r.get("stack_density"), r.get("carry_squeeze_mm", 0.0),
                        r.get("screw_torque", 0.0), r.get("mode"))
        keys = sorted({kf(r) for r in sub})
        print(f"   {'base xyz':26} {'stack':>7} {'rho':>6} {'sqz':>5} {'tq':>6} {'mode':8} "
              f"{'ok':>5} {'deg/cy':>7} {'free':>6} {'tilt':>6} {'ikfail':>6}")
        for k in keys:
            g = [r for r in sub if kf(r) == k]
            good = [r for r in g if r.get("ok")]
            f = lambda key, src=None: (st.mean([r.get(key, 0.0) or 0.0 for r in (src or g)])
                                       if (src or g) else 0.0)
            print(f"   {str(k[:3]):26} {k[3]:7.0f} {k[4]:6.0f} {k[5]:5.2f} {k[6]:6.3f} "
                  f"{str(k[7]):8} {sum(1 for r in g if r.get('ok')):2}/{len(g):<2} "
                  f"{f('gain_mean_deg', good):7.2f} {f('free_frac'):6.3f} "
                  f"{f('final_tilt_deg'):6.2f} {sum(r.get('arm_ik_fails', 0) for r in g):6}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
