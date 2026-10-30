"""P0 baseline: D1's shipped open-loop plan, replayed on the real_v3 hand over a cylinder grid.

THE PLAN is `docs/experiments/20260831-real_v1-sobol8192/deploy/sv1_w6689_b050_traj.csv` --
the joint trajectory the bench runs for D1 (sv1_w6689): open -> grip (0.5 s ramp, 0.8 s settle)
-> turn -80 deg about the pinch axis (1.1 s) -> hold to 4.0 s. It is replayed sample for sample
at 50 Hz, unchanged: this is what running the plan we had on the v3 hand does. Joint
conventions match between real_v1 and real_v3 (0 = straight, same axes and signs).

HAND is real_v3 at gantry D1 with the MorphIt collision model (sphere sets on the links, 1 mm
sphere grid on each fingertip) and SCS0009-like servos (scene.py).

STAGING follows the plan's meta (`sv1_w6689_b050_plan.json`): the tool lies along y with its
axis `grip_depth_mm` (50.5) below the finger mounts' yaw axes, offset from the mount centroid
as in the plan's own scene, 100 mm above the table on an 8 mm post at y = -35 mm from its
centre. Every cylinder size sits at that same axis position -- the plan is not adapted to the
object. Each trial adds hardware-like staging error (+-2 mm in x/y, +-2 deg about z). The tool
is held still until the grip ramp completes (the operator staging it), then released; the
post stays as an obstacle.

SCORES, over the reorientation phase (turn + hold, t >= 1.3 s):
    retention      fraction of steps with >= 2 fingers touching the cylinder  (the headline)
    all3           fraction with all three
    held           retention through the final step AND the tool never fell > 30 mm below
                   where the grip had it -- the drop test
    cos            final |cos| between tool axis and vertical (the bench's cos_hold)

    .venv-cpu/bin/python scripts/cpu_grasp_baseline_p0.py                 # the grid
    .venv-cpu/bin/python scripts/cpu_grasp_baseline_p0.py --record 25,24.5,3 --out run.mp4
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from morphohand.cpu_grasp import judge, scene  # noqa: E402

PLAN_DIR = ROOT / "docs/experiments/20260831-real_v1-sobol8192/deploy"
PLAN = "sv1_w6689_b050"
DT = 0.02
T_RELEASE = 0.5             # staging hold ends when the grip ramp does
T_REORIENT = 1.3            # 0.5 s close + 0.8 s settle
BENCH_Z = 0.100             # tool axis height above the table
POST_R = 0.008
DROP_TOL = 0.030
V1_MOUNTS_MM = {"thumb": (-54.32, -33.74), "index": (37.93, 42.08), "middle": (39.02, -43.03)}
V1_OBJ_IN_PALM_MM = (-7.9215, -11.566)   # object minus palm, from the plan's replay_initial_qpos
OUT_DIR = ROOT / "docs/experiments" / f"{datetime.now():%Y%m%d}-real_v3-p0-baseline"


def load_plan():
    rows = list(csv.DictReader(open(PLAN_DIR / f"{PLAN}_traj.csv")))
    t = np.array([float(r["t_s"]) for r in rows])
    q = np.radians(np.array([[float(r[f"{j}_deg"]) for j in scene.JOINTS] for r in rows]))
    meta = json.load(open(PLAN_DIR / f"{PLAN}_plan.json"))["meta"]
    return t, q, meta


class Bench:
    """The D1 v3 hand + post, staged where the plan expects the tool."""

    def __init__(self):
        self.t, self.q, self.meta = load_plan()
        self.proto = scene.build(post=True)
        m = self.proto
        v1_c = np.mean([V1_MOUNTS_MM[f] for f in scene.FINGERS], 0)
        v3_c = np.mean([m.body(f"{f}_mount").pos[:2] * 1000 for f in scene.FINGERS], 0)
        self.obj_xy = (v3_c + np.array(V1_OBJ_IN_PALM_MM) - v1_c) / 1000.0
        obj_z_palm = m.body("thumb_mount").pos[2] - self.meta["grip_depth_mm"] / 1000.0
        self.palm_joint = (BENCH_Z - obj_z_palm) - m.body("palm_pose").pos[2]
        self.post_y = self.meta["post_y_mm"] / 1000.0

    def make(self, radius, mass):
        m = copy.copy(self.proto)
        scene.set_object(m, radius, mass)
        g = m.geom("tool_post").id
        top = BENCH_Z - radius
        m.geom_size[g] = [POST_R, top / 2, 0]
        m.geom_pos[g] = [self.obj_xy[0], self.obj_xy[1] + self.post_y, top / 2]
        m.geom_rbound[g] = float(np.hypot(POST_R, top / 2))
        m.geom_aabb[g] = [0, 0, 0, POST_R, POST_R, top / 2]
        return m

    def rollout(self, radius, mass, seed, on_step=None) -> dict:
        m = self.make(radius, mass)
        d = mujoco.MjData(m)
        ix = scene.indices(m)
        rng = np.random.default_rng(seed)
        dxy = rng.uniform(-0.002, 0.002, 2)
        dyaw = np.radians(rng.uniform(-2.0, 2.0))
        qo = np.zeros(4)
        mujoco.mju_mulQuat(qo, [np.cos(dyaw / 2), 0, 0, np.sin(dyaw / 2)], [0.70710678, 0.70710678, 0, 0])
        oq, od = ix["obj_qpos"], ix["obj_dof"]
        obj0 = np.concatenate([[self.obj_xy[0] + dxy[0], self.obj_xy[1] + dxy[1], BENCH_Z], qo])
        d.qpos[oq:oq + 7] = obj0
        d.qpos[ix["joint_qpos"]] = self.q[0]
        d.qpos[ix["palm_qpos"]] = self.palm_joint
        mujoco.mj_forward(m, d)
        axis0 = judge.object_axis(d, ix)
        nsub = int(round(DT / m.opt.timestep))
        n = c2 = c3 = 0
        per = np.zeros(3)
        z_grip, z_min, fc = None, np.inf, np.zeros(3, bool)
        trace = []
        for k, t in enumerate(self.t):
            d.ctrl[ix["joint_act"]] = self.q[k]
            d.ctrl[ix["palm_act"]] = self.palm_joint
            for _ in range(nsub):
                if t < T_RELEASE:
                    d.qpos[oq:oq + 7] = obj0
                    d.qvel[od:od + 6] = 0
                mujoco.mj_step(m, d)
            fc = finger_contacts(m, d, ix)
            z = float(d.qpos[oq + 2])
            if t >= T_REORIENT - 1e-9:
                if z_grip is None:
                    z_grip = z
                z_min = min(z_min, z)
                n += 1
                per += fc
                c2 += fc.sum() >= 2
                c3 += fc.sum() == 3
            cos = float(abs(judge.object_axis(d, ix)[2]))
            trace.append((float(t), int(fc.sum()), cos, z))
            if on_step is not None:
                on_step(m, d, float(t), fc, cos)
        axis = judge.object_axis(d, ix)
        drop = (z_grip - z_min) if z_grip is not None else np.inf
        return dict(
            diameter_mm=radius * 2000, mass_g=mass * 1000, seed=seed,
            retention=c2 / n, all3=c3 / n,
            per_finger=(per / n).tolist(),
            held=bool(fc.sum() >= 2 and drop < DROP_TOL),
            cos=float(abs(axis[2])),
            turned_deg=float(np.degrees(np.arccos(np.clip(abs(axis @ axis0), -1, 1)))),
            drop_mm=float(drop * 1000),
            trace=trace,
        )


def finger_contacts(m, d, ix) -> np.ndarray:
    og = ix["obj_geom"]
    hit = np.zeros(3, bool)
    for c in d.contact[:d.ncon]:
        if og not in (c.geom1, c.geom2):
            continue
        b = m.geom_bodyid[c.geom2 if c.geom1 == og else c.geom1]
        for i, f in enumerate(scene.FINGERS):
            if b in ix["finger_bodies"][f]:
                hit[i] = True
    return hit


# ------------------------------------------------------------------------------------------
def record(bench: Bench, diameter, mass_g, seed, out: Path, width=1280, height=720):
    """Re-run one trial (deterministic) and write an mp4 with a status overlay."""
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont

    m = bench.make(diameter / 2000, mass_g / 1000)
    m.vis.global_.offwidth, m.vis.global_.offheight = width, height
    tile_w = width // 2
    r = mujoco.Renderer(m, height, tile_w)
    palm = m.body("palm_pose").id
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == palm:
            m.geom_rgba[g, 3] = 0.25
    m.site_rgba[:, 3] = 0
    cams = []
    frames = []
    try:
        font = ImageFont.truetype("/System/Library/Fonts/SFNSMono.ttf", 22)
    except OSError:
        font = ImageFont.load_default()

    # the run is replayed through bench.rollout's own loop; we render on each control step
    def on_step(mm, dd, t, fc, cos):
        if not cams:
            # end-on (the shaft starts as a circle and stands up into a bar) + 3/4 from below
            # (the pads on the shaft)
            for az, el, dist in ((270, -10, 0.28), (45, 15, 0.28)):
                c = mujoco.MjvCamera()
                c.lookat[:] = [bench.obj_xy[0], bench.obj_xy[1], BENCH_Z]
                c.distance, c.azimuth, c.elevation = dist, az, el
                cams.append(c)
        tiles = []
        for c in cams:
            r.update_scene(dd, c)
            tiles.append(r.render())
        img = Image.fromarray(np.hstack(tiles))
        draw = ImageDraw.Draw(img)
        phase = "close" if t < 0.5 else "settle" if t < 1.3 else "turn" if t < 2.4 else "hold"
        names = "".join(f.upper()[0] if fc[i] else "." for i, f in enumerate(scene.FINGERS))
        draw.rectangle([0, 0, width, 34], fill=(0, 0, 0))
        draw.text((12, 6), f"P0 open-loop  {diameter:.1f} mm / {mass_g:.0f} g   t={t:4.2f}s  {phase:6s}"
                           f"  contacts [{names}]  |cos| {cos:.2f}", fill=(255, 255, 255), font=font)
        frames.append(np.asarray(img))

    res = bench.rollout(diameter / 2000, mass_g / 1000, seed, on_step=on_step)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(out, frames, fps=int(round(1 / DT)), quality=8, macro_block_size=8)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diameters", default="20,22.5,25,27.5,30,32.5,35,37.5,40", help="mm")
    ap.add_argument("--masses", default="20,40,60,80,100,125,150", help="g")
    ap.add_argument("--trials", type=int, default=16)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--record", help="diameter_mm,mass_g,seed -- render that trial to --out instead of the grid")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    bench = Bench()
    if args.record:
        dmm, mg, sd = args.record.split(",")
        res = record(bench, float(dmm), float(mg), int(sd), args.out)
        print({k: v for k, v in res.items() if k != "trace"})
        return 0

    cells = [(float(dm), float(mg)) for dm in args.diameters.split(",") for mg in args.masses.split(",")]
    work = [(dm, mg, s) for dm, mg in cells for s in range(args.trials)]
    print(f"P0 = {PLAN} on real_v3 / D1 / MorphIt: {len(cells)} cells x {args.trials} trials = "
          f"{len(work)} rollouts, {args.threads} threads", flush=True)
    t0 = time.time()
    done = []

    def job(w):
        dm, mg, s = w
        return bench.rollout(dm / 2000, mg / 1000, s)

    with ThreadPoolExecutor(args.threads) as ex:
        for i, res in enumerate(ex.map(job, work)):
            done.append(res)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(work)}  ({time.time() - t0:.0f} s)", flush=True)
    print(f"done in {time.time() - t0:.0f} s")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "p0_trials.json").write_text(json.dumps(dict(plan=PLAN, trials=done), indent=0))
    print(f"\n{'diam':>5} {'mass':>5} | {'retain':>6} {'all3':>5} {'held':>5} {'cos|held':>8}")
    for dm, mg in cells:
        rs = [x for x in done if x["diameter_mm"] == dm and x["mass_g"] == mg]
        held = [x for x in rs if x["held"]]
        print(f"{dm:5.1f} {mg:5.0f} | {np.mean([x['retention'] for x in rs]):6.2f} "
              f"{np.mean([x['all3'] for x in rs]):5.2f} {len(held):2d}/{len(rs):<2d} "
              f"{(np.mean([x['cos'] for x in held]) if held else float('nan')):8.2f}")
    print(f"\nOVERALL retention {np.mean([x['retention'] for x in done]):.3f}  held "
          f"{np.mean([x['held'] for x in done]):.3f}  -> {OUT_DIR.relative_to(ROOT)}/p0_trials.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
