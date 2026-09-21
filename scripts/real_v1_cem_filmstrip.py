#!/usr/bin/env python3
"""Filmstrips of CEM grasp samples on the deployed `real_v1` hands, open keyframe to hold.

One strip per hand. Each row is one finger-control sample the cross-entropy loop actually
drew; the columns are the same six instants of that sample's rollout, so the rows differ only
in what the sampled control did to the object. The rollout is the evaluator's own
(`Phase1GraspEvaluator`: reset at the approach keyframe, close for 240 steps toward the sampled
control, lift the palm 50 mm over 220 steps, hold 140 steps). Nothing is staged for the picture.

The hand is fitted first, exactly as the deploy lane fits it: flat 15 x 21.1 mm pads on the
Sobol base scene, `fit_real_v1_pose.fit` at the plan's straddle, thumb-axial offset and grip
depth, writing `open_ik` (the pads 1 mm off the shaft: the CEM mean) and `open` (the approach
pose, backed off by `--open-gap`: the reset keyframe).

Rows are the fitted pose itself (sample 0 of the first population is the CEM mean), then
samples from the FIRST population -- the draws around the fitted pose at sigma 0.2 rad, before
the elite fit has narrowed anything -- spread evenly over that population's score range, then
the best sample of the last iteration. Every sample drawn is kept in the JSON.

The scene copy is cleaned for the picture only: flat white sky, matte floor, workspace sites
hidden, a finer shadow map. Collision geometry, materials' friction and the keyframes are the
deploy lane's.

    MUJOCO_GL=egl uv run --extra rl python scripts/real_v1_cem_filmstrip.py --hands D1,D3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import fit_real_v1_pose as fp  # noqa: E402
import real_v1_chain_hands as ch  # noqa: E402
from morphohand.optimization.phase1_common import Phase1EvalConfig, Phase1GraspEvaluator  # noqa: E402
from morphohand.optimization.phase1_strategy_cem import (  # noqa: E402
    Phase1OptimizationConfig, optimize_finger_controls,
)
from morphohand.studies.scene_mutate import Scene  # noqa: E402
from morphohand.tools.keyframe_ik import actuator_ctrl_from_qpos, inject_keyframe  # noqa: E402

OBJ = ch.BASE["obj"]
DESIGN_ID = {"sv1_w6689": "D1", "sv1_w2360": "D2", "sv1_u1364": "D3", "g12": "D4",
             "sv1_u0060": "D5", "sv1_u0308": "D6", "rv05_manual": "D7", "sv1_w0099": "D8"}
SCENES = ROOT / "assets/mjcf/experimental/20260920-cem_filmstrip"
OUT = ROOT / "docs/experiments/20260920-cem_filmstrip"

# The pipeline's CEM settings (`real_v1_pipeline.run_cem`): the two persistence weights are
# raised over the evaluator's defaults so a grasp that lifts and lets go scores below one that
# keeps all three pads on.
EVAL = Phase1EvalConfig(objective_weight_min_finger_persistence=4.0,
                        objective_weight_contact_persistence=1.5)

# (label, rollout step) for the columns. Step 0 is the reset pose before any dynamics.
FRAMES = [("open keyframe", 0), ("closing", 60), ("closed", 240),
          ("lifting", 320), ("lifted", 460), ("held", 599)]


def hand_record(did: str) -> dict:
    for h in ch.hands("A"):
        if DESIGN_ID.get(h["tag"].rsplit("_b", 1)[0]) == did:
            return h
    raise SystemExit(f"{did}: no deployed plan in set A")


def clean_visuals(p: Path) -> None:
    """Presentation only: nothing here touches collision geometry, contacts or keyframes."""
    tree = ET.parse(p, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)))
    root = tree.getroot()
    asset = root.find("asset")
    for t in asset.findall("texture"):
        if t.get("type") == "skybox":
            t.attrib.clear()
            t.attrib.update(type="skybox", builtin="flat", rgb1="1 1 1", rgb2="1 1 1",
                            width="32", height="32")
    for mat in asset.findall("material"):
        if mat.get("name") == "groundplane":
            mat.attrib.clear()
            mat.attrib.update(name="groundplane", rgba="0.90 0.90 0.89 1", reflectance="0")
        if mat.get("name") == "object_mat":
            mat.set("rgba", "0.80 0.36 0.18 1")   # grey shaft on a grey floor did not read
    for site in root.iter("site"):
        if (site.get("name") or "").startswith("workspace_"):
            site.set("rgba", "0 0 0 0")
    for g in root.iter("geom"):
        if g.get("name") == "floor":
            g.set("size", "0.6 0.6 0.1")       # a plane collides as infinite regardless
    root.find("visual/quality").set("shadowsize", "8192")
    root.find("visual/global").attrib.update(fovy="36", offwidth="1920", offheight="1440")
    tree.write(p)


def fitted_scene(h: dict, open_gap: float) -> tuple[Path, dict]:
    """Flat pads on the base scene, then the deploy fit, then `open_ik` and `open` keyframes."""
    SCENES.mkdir(parents=True, exist_ok=True)
    design = h["tag"].rsplit("_b", 1)[0]
    p = SCENES / f"{design}__flat_fit.xml"
    sc = Scene(Path(h["base"]))
    sc.set_finger_flat_pads(**ch.PAD)
    sc.write(p)
    clean_visuals(p)

    # Same palm-height window as `probe_real_v1_carry._grip_from_fit`: the plan's depth, and
    # up to 8 mm shallower, so the pose is the deployed one and not the fitter's deepest.
    m0 = mujoco.MjModel.from_xml_path(str(p))
    d0 = mujoco.MjData(m0)
    mujoco.mj_resetDataKeyframe(m0, d0, fp._seed_key(m0, "open"))
    mujoco.mj_forward(m0, d0)
    obj_z = float(fp._object_geometry(m0, d0, OBJ)[0][2])
    pz_hi = obj_z + h["depth"] - float(m0.body("palm_pose").pos[2])
    out = fp.fit(p, 0.001, h["straddle"], 0.0, OBJ, pz_hi - 0.008, pz_hi, 0.0025,
                 verbose=False, spreads=(h["straddle"],), squeeze=0.004, hold_min=-1.0,
                 thumb_axial=h["thumb_axial"])
    if out is None:
        raise SystemExit(f"{h['tag']}: fit failed")
    report, qpos, ctrl = out
    inject_keyframe(p, "open_ik", qpos, ctrl)

    # The approach pose: same palm, pads backed off radially by `open_gap`.
    m = mujoco.MjModel.from_xml_path(str(p))
    d = mujoco.MjData(m)
    seed = fp._seed_key(m, "open")
    mujoco.mj_resetDataKeyframe(m, d, seed)
    mujoco.mj_forward(m, d)
    centre, radius, _ = fp._object_geometry(m, d, OBJ)
    pal = report["palm"]
    fp.solve(m, d, fp.tip_targets(centre, radius, open_gap, h["straddle"], 0.0, 0.0,
                                  h["thumb_axial"]), pal["px"], pal["py"], pal["pz"], seed)
    inject_keyframe(p, "open", " ".join(f"{v:.6g}" for v in d.qpos),
                    " ".join(f"{v:.6g}" for v in actuator_ctrl_from_qpos(m, d)))
    return p, report


def run_cem(scene: Path, population: int, iterations: int, seed: int) -> dict:
    """The pipeline's CEM, with every sample it draws recorded in draw order."""
    ev = Phase1GraspEvaluator(scene, keyframe="open", cfg=EVAL)
    m = mujoco.MjModel.from_xml_path(str(scene))
    kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "open_ik")
    mean = np.asarray(m.key_ctrl[kid])[ev.finger_actuator_ids].astype(np.float64)

    log: list[dict] = []
    evaluate = ev.evaluate

    def recorded(x):
        s, met = evaluate(x)
        log.append({"iteration": len(log) // population, "index": len(log) % population,
                    "ctrl": [float(v) for v in x], "score": float(s),
                    **{k: float(met[k]) for k in ("cube_tip_contacts", "cube_z_before_lift",
                                                   "cube_z_after_hold", "cube_z_drop_from_peak",
                                                   "cube_xy_drift", "cube_axis_tilt",
                                                   "contact_persistence",
                                                   "min_finger_contact_persistence")}})
        return s, met

    ev.evaluate = recorded
    t0 = time.perf_counter()
    res = optimize_finger_controls(
        ev, Phase1OptimizationConfig(iterations=iterations, population=population,
                                     sigma_init=0.20, seed=seed, log_every=1),
        initial_finger_ctrl=mean)
    for r in log:
        r["held_mm"] = round(1000 * (r["cube_z_after_hold"] - r["cube_z_before_lift"]), 1)
    return {"evaluator": ev, "mean_ctrl": mean.tolist(), "samples": log,
            "best_ctrl": [float(v) for v in res["best_finger_ctrl"]],
            "best_score": float(res["best_score"]), "history": res["history"],
            "wall_s": round(time.perf_counter() - t0, 1)}


def choose_rows(samples: list[dict], n_first: int, population: int) -> list[dict]:
    """The fitted pose (sample 0 of iteration 0 is the CEM mean), `n_first` rows spread over
    the rest of the first population's score range, then the last iteration's best. The first
    population is where the samples are furthest apart."""
    mean_row = samples[0]
    first = sorted(samples[1:population], key=lambda r: r["score"])
    picks = [first[int(round(i * (len(first) - 1) / max(1, n_first - 1)))]
             for i in range(n_first)]
    last_it = samples[-1]["iteration"]
    best_last = max((r for r in samples if r["iteration"] == last_it), key=lambda r: r["score"])
    rows = []
    seen = set()
    for r in [mean_row] + picks + [best_last]:
        key = (r["iteration"], r["index"])
        if key not in seen:
            seen.add(key)
            rows.append(r)
    return rows


def render_rollout_frames(ev: Phase1GraspEvaluator, finger_ctrl: np.ndarray, steps: list[int],
                          width: int, height: int, cam: dict) -> list[np.ndarray]:
    """The evaluator's rollout, photographed at `steps` (0 = the reset pose)."""
    m = ev.model
    m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), width)
    m.vis.global_.offheight = max(int(m.vis.global_.offheight), height)
    renderer = mujoco.Renderer(m, height=height, width=width)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth, camera.elevation = cam["azimuth"], cam["elevation"]
    camera.distance = cam["distance"]
    camera.lookat[:] = cam["lookat"]

    finger_ctrl = np.clip(np.asarray(finger_ctrl, dtype=np.float64),
                          ev.finger_ctrl_min, ev.finger_ctrl_max)
    ev._reset_to_keyframe()
    settle_ctrl = ev._build_full_ctrl(finger_ctrl, lift=False)
    total = int(ev.cfg.settle_steps + ev.cfg.lift_steps + ev.cfg.pivot_steps + ev.cfg.hold_steps)
    want = set(steps)
    frames: dict[int, np.ndarray] = {}
    if 0 in want:
        renderer.update_scene(ev.data, camera=camera)
        frames[0] = renderer.render().copy()
    for t in range(total):
        if t < ev.cfg.settle_steps:
            ev.data.ctrl[:] = settle_ctrl
        else:
            ev.data.ctrl[:] = ev._ctrl_for_dynamic_step(t - ev.cfg.settle_steps,
                                                        lambda _t: finger_ctrl)
        ev._step_dynamics(force_sync=False)
        ev._sync_mujoco_from_backend()
        if (t + 1) in want:
            renderer.update_scene(ev.data, camera=camera)
            frames[t + 1] = renderer.render().copy()
    renderer.close()
    return [frames[s] for s in steps]


def _font(size: int) -> ImageFont.FreeTypeFont:
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"):
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def row_caption(r: dict, mean_ctrl: list[float], tag: str) -> list[str]:
    dev = float(np.max(np.abs(np.asarray(r["ctrl"]) - np.asarray(mean_ctrl))))
    pads = int(r["cube_tip_contacts"])
    held = r["held_mm"]
    if held >= 20.0:
        verdict = f"held +{held:.0f} mm"
    elif r["cube_z_drop_from_peak"] >= 0.02:
        verdict = "lifted, dropped"
    else:
        verdict = "not lifted"
    if pads == 0:
        verdict = "no contact"
    return [tag, f"score {r['score']:+.2f}", f"{pads} pad{'s' if pads != 1 else ''} at closure",
            verdict, f"max |ctrl \u2212 mean| {dev:.2f} rad"]


def build_strip(hand: dict, fit: dict, cem: dict, rows: list[dict], frames: list[list[np.ndarray]],
                out_png: Path, tile_w: int, tile_h: int) -> None:
    label_w, head_h, gap = 390, 84, 6
    n_r, n_c = len(rows), len(FRAMES)
    W = label_w + n_c * (tile_w + gap) + gap
    H = head_h + n_r * (tile_h + gap) + gap
    img = Image.new("RGB", (W, H), (250, 250, 249))
    dr = ImageDraw.Draw(img)
    f_head, f_cap, f_small = _font(22), _font(20), _font(17)
    did = DESIGN_ID[hand["tag"].rsplit("_b", 1)[0]]
    dr.text((gap + 4, 8), f"{did} {hand['tag'].rsplit('_b', 1)[0]}", fill=(20, 20, 20), font=f_head)
    dr.text((gap + 4, 34), f"straddle {hand['straddle']*1000:.0f} mm, thumb axial "
            f"{hand['thumb_axial']*1000:.0f} mm", fill=(90, 90, 90), font=f_small)
    dr.text((gap + 4, 56), f"grip depth {fit['grip_depth_mm']:.1f} mm, CEM sigma 0.20 rad",
            fill=(90, 90, 90), font=f_small)
    for j, (name, step) in enumerate(FRAMES):
        x = label_w + gap + j * (tile_w + gap)
        dr.text((x, 8), name, fill=(20, 20, 20), font=f_head)
        dr.text((x, 34), f"step {step}  ({step*0.002:.2f} s)", fill=(90, 90, 90), font=f_small)
    for i, (r, fr) in enumerate(zip(rows, frames)):
        y = head_h + gap + i * (tile_h + gap)
        if i == 0:
            tag = "fitted pose (CEM mean)"
        elif i == len(rows) - 1:
            tag = f"CEM best, iteration {r['iteration']}"
        else:
            tag = f"sample {r['index']}, iteration {r['iteration']}"
        lines = row_caption(r, cem["mean_ctrl"], tag)
        for k, line in enumerate(lines):
            dr.text((gap + 4, y + 6 + k * 26), line,
                    fill=(20, 20, 20) if k in (0, 3) else (90, 90, 90),
                    font=f_cap if k in (0, 3) else f_small)
        for j, a in enumerate(fr):
            x = label_w + gap + j * (tile_w + gap)
            img.paste(Image.fromarray(a), (x, y))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png)


def build_video(rows: list[dict], cem: dict, out_mp4: Path, tile_w: int, tile_h: int,
                cam: dict, title: str, stride: int = 4, fps: int = 25) -> None:
    """The same rollouts as the strip, every `stride` steps, tiled 3 x 2 with a caption."""
    from morphohand.optimization.phase1_common import _write_mp4
    ev = cem["evaluator"]
    total = int(ev.cfg.settle_steps + ev.cfg.lift_steps + ev.cfg.pivot_steps + ev.cfg.hold_steps)
    steps = list(range(0, total + 1, stride))
    clips = [render_rollout_frames(ev, np.asarray(r["ctrl"]), steps, tile_w, tile_h, cam)
             for r in rows]
    cols, cap_h, head_h, gap = 3, 54, 34, 4
    n_rows = (len(rows) + cols - 1) // cols
    W = cols * tile_w + (cols + 1) * gap
    H = head_h + n_rows * (tile_h + cap_h) + (n_rows + 1) * gap
    f_cap, f_small = _font(19), _font(16)
    captions = []
    for i, r in enumerate(rows):
        if i == 0:
            tag = "fitted pose (CEM mean)"
        elif i == len(rows) - 1:
            tag = f"CEM best, iteration {r['iteration']}"
        else:
            tag = f"sample {r['index']}, iteration {r['iteration']}"
        lines = row_caption(r, cem["mean_ctrl"], tag)
        captions.append((lines[0], f"{lines[2]}, {lines[3]}"))
    frames = []
    for k, step in enumerate(steps):
        img = Image.new("RGB", (W, H), (250, 250, 249))
        dr = ImageDraw.Draw(img)
        for i, clip in enumerate(clips):
            x = gap + (i % cols) * (tile_w + gap)
            y = head_h + gap + (i // cols) * (tile_h + cap_h + gap)
            img.paste(Image.fromarray(clip[k]), (x, y))
            dr.text((x + 4, y + tile_h + 4), captions[i][0], fill=(20, 20, 20), font=f_cap)
            dr.text((x + 4, y + tile_h + 28), captions[i][1], fill=(90, 90, 90), font=f_small)
        phase = ("close" if step < ev.cfg.settle_steps
                 else "lift" if step < ev.cfg.settle_steps + ev.cfg.lift_steps else "hold")
        dr.text((gap + 4, 8), title, fill=(20, 20, 20), font=f_cap)
        dr.text((W - 150, 10), f"{phase}  {step * 0.002:5.2f} s", fill=(90, 90, 90),
                font=f_small)
        frames.append(np.asarray(img))
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    _write_mp4(out_mp4, frames, fps)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", default="D1,D3")
    ap.add_argument("--population", type=int, default=80)
    ap.add_argument("--iterations", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rows", type=int, default=4, help="rows from the first population")
    ap.add_argument("--open-gap", type=float, default=0.008)
    ap.add_argument("--tile", default="480x320")
    ap.add_argument("--azimuth", type=float, default=145.0)
    ap.add_argument("--elevation", type=float, default=-10.0)
    ap.add_argument("--distance", type=float, default=0.285)
    ap.add_argument("--lookat", default="0 0 0.062")
    ap.add_argument("--video", action="store_true",
                    help="also write the six rollouts tiled as an mp4, every 4th step")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--reuse", action="store_true",
                    help="re-render from <out>/<hand>_cem.json instead of re-running CEM")
    a = ap.parse_args()
    tile_w, tile_h = (int(v) for v in a.tile.split("x"))
    cam = {"azimuth": a.azimuth, "elevation": a.elevation, "distance": a.distance,
           "lookat": [float(v) for v in a.lookat.split()]}
    a.out.mkdir(parents=True, exist_ok=True)

    for did in a.hands.split(","):
        h = hand_record(did)
        design = h["tag"].rsplit("_b", 1)[0]
        scene, fit = fitted_scene(h, a.open_gap)
        print(f"[{did}] {design}: fitted grip depth {fit['grip_depth_mm']:.1f} mm, residuals "
              + " ".join(f"{k} {v:.1f}" for k, v in fit["tip_residual_mm"].items())
              + f" mm, self-collisions {fit['self_collisions'] or 'none'}", flush=True)
        jpath = a.out / f"20260920-{did}_{design}_cem.json"
        if a.reuse and jpath.exists():
            rec = json.loads(jpath.read_text())
            cem = {**rec, "evaluator": Phase1GraspEvaluator(scene, keyframe="open", cfg=EVAL)}
        else:
            cem = run_cem(scene, a.population, a.iterations, a.seed)
            rec = {k: v for k, v in cem.items() if k != "evaluator"}
            rec.update(hand=h, fit=fit, scene=str(scene), population=a.population,
                       iterations=a.iterations, cem_seed=a.seed, open_gap=a.open_gap)
            jpath.write_text(json.dumps(rec, indent=1))
            print(f"[{did}] CEM {a.iterations}x{a.population} in {cem['wall_s']} s, "
                  f"best score {cem['best_score']:+.3f}", flush=True)
        rows = choose_rows(cem["samples"], a.rows, cem.get("population", a.population))
        for r in rows:
            print(f"   it {r['iteration']:2d} #{r['index']:2d}  score {r['score']:+7.3f}  "
                  f"pads {int(r['cube_tip_contacts'])}  held {r['held_mm']:+6.1f} mm  "
                  f"drop {1000*r['cube_z_drop_from_peak']:5.1f} mm  xy {1000*r['cube_xy_drift']:5.1f} mm")
        frames = [render_rollout_frames(cem["evaluator"], np.asarray(r["ctrl"]),
                                        [s for _, s in FRAMES], tile_w, tile_h, cam)
                  for r in rows]
        png = a.out / f"20260920-{did}_{design}_cem_filmstrip.png"
        build_strip(h, fit, cem, rows, frames, png, tile_w, tile_h)
        print(f"[{did}] wrote {png}", flush=True)
        if a.video:
            mp4 = a.out / f"20260920-{did}_{design}_cem_samples.mp4"
            build_video(rows, cem, mp4, tile_w, tile_h, cam,
                        f"{did} {design}: CEM grasp samples, straddle {h['straddle']*1000:.0f} mm")
            print(f"[{did}] wrote {mp4}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
