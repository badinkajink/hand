"""Film of one of the night's g0 twists (2026-11-17, tips off): a tracking camera's frames of the hand beside the
replay drawn from the same camera, with the sphere's twist about z_P (hand and replay) below.

The hand's films are the trial recordings of the two D435s (hom_trial.py --video-dir, on corvus in
logs/20261116-hom_trials/20261117-night_repeats*/, copied to logs/20261117-night_trial_films/), 960 x 540 at 30 fps,
each frame stamped with corvus's clock; the trial's t = 0 is the corvus record's t0_ms, which the server's record shares.
The replay is sim/20261118-s3_plant_replay.py's (plant, friction, give and contact model as given), run with the
plant's states recorded at every physics step. Each camera is the trials' tracker's: the lens of
20261113-lens_board_k3_arm.json scaled to 960 x 540, and its pose in the arm's base from
20261113-handeye_tcp_fitted_lens_moved836.json (docs/experiments/20261113-camera_arm_calibration/data/fits). Its pose in
P at each frame is (T_BE T_EP)^-1 T_BC, with the TCP pose T_BE from the corvus record (50 Hz, nearest sample) and T_EP
from sr2hand.robot.tool_chain.SR2Hand, as the tracker computed the object's pose; P is the simulator's world frame moved
to the palm. The replay's hand stays still and its table goes at the lift, so the drawn table plate moves down by the
arm's rise, and the floor plane is hidden. The lens distortion is not drawn. On each panel the object's x axis (red) and
y axis (blue), 35 mm from its centre, show its turn: on the hand's frames the tracker's pose nearest the frame's capture
time, on the replay's the simulated pose, both projected through the same pinhole.

    MUJOCO_GL=egl HOS_PLANT=kp0.55_kv0_fr0.18_fl0_dp0.011 SCALE_HAND=cad_v2 OMP_NUM_THREADS=1 \\
        .venv/bin/python sim/20261118-night_trial_film.py --trial 20261117-194243 [--contact pads_stiff] \\
        [--plant s3tc] [--mu 1.0] [--kc 2.0] [--cam 832412070344] [--slow 4.6 5.8 0.2] [--stills 0 3 5.07] [--out FILE]
"""
import argparse
import csv
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[2]
_spec = importlib.util.spec_from_file_location("s3_plant_replay", HERE / "20261118-s3_plant_replay.py")
S3R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S3R)
NR = S3R.NR

FILMS = ROOT / "logs/20261117-night_trial_films"
CAL = ROOT / "docs/experiments/20261113-camera_arm_calibration/data/fits"
LENS, HANDEYE = CAL / "20261113-lens_board_k3_arm.json", CAL / "20261113-handeye_tcp_fitted_lens_moved836.json"
W, H = 960, 540
CROP = {"832412070344": (270, 90, 610, 410), "836612071918": (400, 40, 740, 360)}   # x0, y0, x1, y1 of the frame
SCALE = 2                                                                          # crop enlargement in the film
LABEL = {"s3tc": "S3 servos", "s3": "S3 servos (fast free air)"}
CONTACT = {"pads": "soft pads", "pads_stiff": "stiff pads", "point": "point contact"}


def replay(job):
    """(the replay's output row, its plant with the recorded states, the toy setup)."""
    S3R.PLANT, S3R.MU = job["plant"], job["mu"]
    S3R.YG.K_C, S3R.YG.ETA, S3R.YG.BACKDRIVE = job["k_c"], 1.0, False
    NR.RPL.RG.gear_class = lambda: S3R.YG.give_class(S3R.s3_class())
    NR.RPL.RC2.STIFF, NR.RPL.RC2.BREAKAWAY = True, 0.0
    NR.RPL.RG.RF.RL.rows_file()
    NR.RPL.RG.init(str(NR.RPL.RG.RF.SCENE))
    box, make = {}, NR.make_plant

    def make_rec(contact, volts):
        p, setup = make(contact, volts)
        p.rec = []
        box["p"], box["setup"] = p, setup
        return p, setup

    NR.make_plant = make_rec
    try:
        out = NR.run1(job)
    finally:
        NR.make_plant = make
    return out, box["p"], box["setup"]


def cameras():
    from sr2hand.geometry.rotations import tf_inv
    L, He = json.loads(LENS.read_text()), json.loads(HANDEYE.read_text())
    s = W / 1920.0
    out = {}
    for sn, c in He["cams"].items():
        l = L["cams"][sn]["lens"]
        out[sn] = {"fx": l["fx"] * s, "fy": l["fy"] * s, "cx": (l["cx"] + 0.5) * s - 0.5, "cy": (l["cy"] + 0.5) * s - 0.5,
                   "T_BC": tf_inv(np.array(c["T_CB"], float))}
    return out


def render_model(p, contact, cam):
    """The replay's scene with a mocap body that carries the camera, its bodies placed as in the plant's model."""
    import mujoco
    spec = mujoco.MjSpec.from_file(str(NR.RPL.D / "scenes" / NR.RPL.SCENES[contact]))
    b = spec.worldbody.add_body(name="film_cam", mocap=True)
    c = b.add_camera(name="film", quat=[0.0, 1.0, 0.0, 0.0])     # OpenCV (x right, y down, z ahead) to MuJoCo's camera
    c.resolution = [W, H]
    c.focal_pixel = [cam["fx"], cam["fy"]]
    c.principal_pixel = [(W - 1) / 2 - cam["cx"], (H - 1) / 2 - cam["cy"]]
    c.sensor_size = [W * 1e-6, H * 1e-6]
    m = spec.compile()
    nb = p.m.nbody
    assert m.nbody == nb + 1 and m.nq == p.m.nq
    m.body_pos[:nb], m.body_quat[:nb] = p.m.body_pos, p.m.body_quat
    m.vis.global_.offwidth, m.vis.global_.offheight = max(W, 1280), max(H, 720)
    return m


def arm_track(corvus_file):
    """(trial times, T_BP per sample in mm) from the corvus record's TCP poses."""
    from sr2hand.geometry.rotations import tf_from_ur
    from sr2hand.robot.tool_chain import SR2Hand
    T_EP = SR2Hand.load().T_E_P()
    rows = [json.loads(x) for x in Path(corvus_file).read_text().splitlines() if x.strip()]
    rows = [r for r in rows if r.get("arm") is not None and r.get("t") is not None]
    t = np.array([float(r["t"]) for r in rows])
    T = np.stack([tf_from_ur(np.asarray(r["arm"], float), scale=1.0) @ T_EP for r in rows])
    return t, T


def film_frames(st, sn, t0_ms, t_from, t_to, tmp):
    """[(trial time, png path)] of the camera's film between t_from and t_to."""
    f = next(FILMS.glob(f"{st}_trial_*_{sn}.mp4"))
    rows = list(csv.DictReader(open(str(f).replace(".mp4", "_frames.csv"))))
    t = (np.array([float(r["t_host_ms"]) for r in rows]) - t0_ms) / 1e3
    keep = np.flatnonzero((t >= t_from) & (t <= t_to))
    d = Path(tmp) / f"real_{sn}"
    d.mkdir()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(f), "-vf",
                    f"select=between(n\\,{keep[0]}\\,{keep[-1]})", "-vsync", "0", "-start_number", str(keep[0]),
                    "-q:v", "2", str(d / "f%05d.jpg")], check=True)
    return [(float(t[i]), d / f"f{i:05d}.jpg") for i in keep]


def project(T_PC, cam, X_P):
    """Pixels (n, 2) of points in P (mm) through the camera's pinhole (960 x 540)."""
    X = (np.asarray(X_P, float) - T_PC[:3, 3]) @ T_PC[:3, :3]
    return np.c_[cam["fx"] * X[:, 0] / X[:, 2] + cam["cx"], cam["fy"] * X[:, 1] / X[:, 2] + cam["cy"]]


def draw_axes(dr, uv, crop, scale, at):
    """The object's x (red) and y (blue) axes from its centre, uv = pixels of (centre, x end, y end) in the frame,
    drawn on the panel whose top left corner is at `at` on the canvas."""
    x0, y0 = crop[:2]
    c, ex, ey = [((u - x0) * scale + at[0], (v - y0) * scale + at[1]) for u, v in uv]
    dr.line([c, ey], fill=(30, 90, 230), width=5)
    dr.line([c, ex], fill=(230, 30, 30), width=5)
    dr.ellipse([c[0] - 5, c[1] - 5, c[0] + 5, c[1] + 5], fill=(255, 200, 0))


def trace_panel(out, width, t_from, t_to):
    """The twist traces as an image, and the function from trial time to its x pixel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tr = out["trace"]
    dpi = 100
    fig, ax = plt.subplots(figsize=(width / dpi, 2.6), dpi=dpi)
    ax.plot(tr["t"], tr["twist_hand"], color="black", lw=2.0, label="hand (tracked)")
    ax.plot(tr["t"], tr["twist_replay"], color="tab:red", lw=2.0, label="replay")
    ax.set_xlim(t_from, t_to)
    ax.set_xlabel("time from the trial's start (s)", fontsize=13)
    ax.set_ylabel("twist about z_P (deg)", fontsize=13)
    ax.tick_params(labelsize=12)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=12, frameon=False)
    fig.tight_layout()
    fig.canvas.draw()
    bb = ax.get_window_extent()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    x0, x1 = bb.x0, bb.x1
    y0, y1 = img.shape[0] - bb.y1, img.shape[0] - bb.y0
    return img, (lambda t: x0 + (t - t_from) / (t_to - t_from) * (x1 - x0)), (y0, y1)


def main():
    import mujoco
    import hom_trial_tracking as TT
    from PIL import Image, ImageDraw, ImageFont
    from sr2hand.geometry.rotations import from_scipy, quat_to_mat, tf_inv
    from scipy.spatial.transform import Rotation
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", default="20261117-194243")
    ap.add_argument("--contact", default="pads_stiff", choices=list(NR.RPL.SCENES))
    ap.add_argument("--plant", default="s3tc", choices=["s3", "s3tc"])
    ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--kc", type=float, default=2.0)
    ap.add_argument("--cam", default="832412070344")
    ap.add_argument("--window", type=float, nargs=2, default=[-0.3, 8.3])
    ap.add_argument("--slow", type=float, nargs=3, default=[4.6, 5.8, 0.2], help="from, to, speed (1 = real time)")
    ap.add_argument("--speed", type=float, default=0.5, help="the speed outside --slow")
    ap.add_argument("--stills", type=float, nargs="*", default=[], help="also write these frames as PNG")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    trial = next(t for t in NR.trials() if t["stamp"] == a.trial)
    sc = json.loads((ROOT / "docs/experiments/20261115-hom_bench_trials/data/20261117-night_repeats_tracking.json")
                    .read_text())
    corvus_file = ROOT / next(t for t in sc["trials"] if t["stamp"] == a.trial)["corvus"]["file"]
    st = Path(corvus_file).name[:15]
    t0_ms = json.loads(Path(corvus_file).read_text().splitlines()[0])["t0_ms"]
    variant = f"{a.plant} mu{a.mu:g}" + (f" kc{a.kc:g}" if a.kc else "")
    job = {**trial, "contact": a.contact, "volts": "7v5", "plant": a.plant, "mu": a.mu, "k_c": a.kc or None,
           "variant": variant}
    out, p, setup = replay(job)
    if out.get("diverged") or "error" in out:
        sys.exit(f"replay failed: {out.get('error') or 'diverged'}")
    rt = np.array([r[0] for r in p.rec])
    rq9 = np.stack([r[1] for r in p.rec])
    rx = np.stack([r[2] for r in p.rec])
    cam = cameras()[a.cam]
    m = render_model(p, a.contact, cam)
    d = mujoco.MjData(m)
    d.qpos[:] = p.d.qpos
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor >= 0:
        m.geom_rgba[floor, 3] = 0.0
    post = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "post")
    post0 = m.body_pos[post].copy()
    palm = np.asarray(setup["palm"], float)
    ta, TBP = arm_track(corvus_file)
    _head, srows, _end = TT.load_server(ROOT / trial["file"])
    seen = [r for r in srows if r.get("obj_quat") is not None and r.get("obj_pos_mm") is not None]
    t_seen = np.array([(float(r["t_capture_ms"]) - t0_ms) / 1e3 if r.get("t_capture_ms") else float(r["t"]) for r in seen])
    AX = np.array([[0.0, 0.0, 0.0], [35.0, 0.0, 0.0], [0.0, 35.0, 0.0]])
    T_BC = cam["T_BC"]
    TBP0 = TBP[int(np.argmin(abs(ta)))]
    ren = mujoco.Renderer(m, H, W, max_geom=20000)
    tmp = Path(tempfile.mkdtemp(dir=str(ROOT / "logs/20261117-night_trial_films")))
    frames = film_frames(st, a.cam, t0_ms, a.window[0], a.window[1], tmp)
    x0, y0, x1, y1 = CROP[a.cam]
    cw, ch = (x1 - x0) * SCALE, (y1 - y0) * SCALE
    bar = 34
    panel, tx, (py0, py1) = trace_panel(out, 2 * cw, a.window[0], a.window[1])
    font = ImageFont.load_default(size=22)
    hand_tw = out["hand"]["motion"]["twist_end_deg"]
    rep_tw = out["replay"]["motion"]["twist_end_deg"]
    left = f"hand, camera {a.cam[-4:]}: twist {hand_tw:.0f} deg at the end"
    right = f"replay: {LABEL[a.plant]}" + (f", bend {a.kc:g} N m/rad" if a.kc else "") + \
        f", {CONTACT[a.contact]}, friction {a.mu:g}: {rep_tw:.0f} deg"
    name = a.out or str(STUDY / "media" / f"20261118-night_twist_{a.trial[-6:]}_{a.cam[-4:]}_hand_vs_replay.mp4")
    Path(name).parent.mkdir(parents=True, exist_ok=True)
    size = (2 * cw, bar + ch + panel.shape[0])
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
                            f"{size[0]}x{size[1]}", "-r", "30", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                            "-crf", "21", name], stdin=subprocess.PIPE)
    n_out = 0
    stills = []
    for t, png in frames:
        i = int(np.clip(np.searchsorted(rt, t), 0, len(rt) - 1))      # before the first step: the start's state
        d.qpos[p.qadr9] = rq9[i]
        d.qpos[p.qa:p.qa + 7] = rx[i]
        k = int(np.argmin(abs(ta - t)))
        T_PC = tf_inv(TBP[k]) @ T_BC
        d.mocap_pos[0] = palm + T_PC[:3, 3] / 1e3
        d.mocap_quat[0] = from_scipy(Rotation.from_matrix(T_PC[:3, :3]))
        rise = (TBP[k][:3, 3] - TBP0[:3, 3]) / 1e3                     # the arm's rise since t = 0, in B
        m.body_pos[post] = post0 - TBP0[:3, :3].T @ rise              # in P, the table moves down by it
        mujoco.mj_forward(m, d)
        ren.update_scene(d, camera="film")
        sim = Image.fromarray(ren.render())
        real = Image.open(png).convert("RGB")
        canvas = Image.new("RGB", size, "white")
        canvas.paste(real.crop((x0, y0, x1, y1)).resize((cw, ch), Image.LANCZOS), (0, bar))
        canvas.paste(sim.crop((x0, y0, x1, y1)).resize((cw, ch), Image.LANCZOS), (cw, bar))
        canvas.paste(Image.fromarray(panel), (0, bar + ch))
        dr = ImageDraw.Draw(canvas)
        j = int(np.argmin(abs(t_seen - t)))
        if abs(t_seen[j] - t) < 0.1:
            R, c = quat_to_mat(np.asarray(seen[j]["obj_quat"], float)), np.asarray(seen[j]["obj_pos_mm"], float)
            draw_axes(dr, project(T_PC, cam, c + AX @ R.T), (x0, y0), SCALE, (0, bar))
        R, c = quat_to_mat(rx[i][3:7]), (rx[i][:3] - palm) * 1e3
        draw_axes(dr, project(T_PC, cam, c + AX @ R.T), (x0, y0), SCALE, (cw, bar))
        dr.text((8, 5), left, fill="black", font=font)
        dr.text((cw + 8, 5), right, fill="black", font=font)
        dr.text((2 * cw - 130, bar + 8), f"t {t:5.2f} s", fill="white", font=font)
        x = tx(t)
        dr.line([(x, bar + ch + py0), (x, bar + ch + py1)], fill=(40, 90, 200), width=3)
        sp = a.slow[2] if a.slow[0] <= t <= a.slow[1] else a.speed
        reps = max(1, int(round(1.0 / sp)))
        raw = canvas.tobytes()
        for _ in range(reps):
            enc.stdin.write(raw)
            n_out += 1
        if any(abs(t - s) < 0.017 for s in a.stills):
            stills.append((t, canvas.copy()))
    ren.close()
    enc.stdin.close()
    enc.wait()
    for t, im in stills:
        im.save(name.replace(".mp4", f"_t{t:.2f}.png"))
    shutil.rmtree(tmp)
    print("wrote", name, f"({n_out} frames)", "stills", [round(t, 2) for t, _ in stills])
    print("hand twist", hand_tw, "replay twist", rep_tw, "track hand", out["hand"]["track"].get("score"), "replay",
          out["replay"]["track"].get("score"))


if __name__ == "__main__":
    main()
