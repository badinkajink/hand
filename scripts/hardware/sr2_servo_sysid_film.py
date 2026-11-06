#!/usr/bin/env python3
"""Film of the yaw steps of 2026-11-06 (block c of the SR2 servo identification, QUEUE R36): the cad_v3 hand on the UR5
at the arm's pose of the SR2 night, each yaw replayed from its measured telemetry (holds_c.tsv) for the first 2 s of
every step away from zero. A red line runs from the yaw axis through the fingertip at the commanded angle and a blue
line at the measured one, both drawn a little toward the camera so the finger cannot hide them. Each finger is seen along
its yaw axis, so the swing and the shortfall lie in the picture's plane.

    .venv/bin/python scripts/hardware/sr2_servo_sysid_film.py      # under resguard: it loads the cad_v3 model

Writes docs/experiments/20261106-sr2_servo_sysid/media/20261106-yaw_steps_replay.mp4 and its poster .jpg.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _paths  # noqa: E402,F401

import csv  # noqa: E402
import subprocess  # noqa: E402

import numpy as np  # noqa: E402

import cad_v3_film as CF  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "docs/experiments/20261106-sr2_servo_sysid"
MODEL = ROOT / "assets/mjcf/experimental/20261104-cad_v3/ur5_full_k1.xml"
OUT = D / "media/20261106-yaw_steps_replay.mp4"
F = ("thumb", "index", "middle")
YAW_SIGN = {"thumb": 1.0, "index": -1.0, "middle": -1.0}
HOME = {"thumb": (-0.030, -0.055), "index": (0.030, 0.030), "middle": (0.030, -0.030)}   # gantry joints at home (m)
W, H, FPS, SHOW, PAUSE, DIST = 960, 640, 25, 2.0, 0.6, 0.24


def steps():
    """Block c's steps away from zero, in run order: (finger, servo command, [(t, servo deg)])."""
    rows = list(csv.DictReader(open(D / "data/holds_c.tsv"), delimiter="\t"))
    out = {}
    for r in rows:
        if r["block"] == "c" and r["dir"] == "out" and r["deg"]:
            out.setdefault(int(r["step"]), (r["finger"], float(r["cmd_servo_deg"]), []))[2].append(
                (float(r["t_from_cmd_s"]), float(r["deg"])))
    return [out[k] for k in sorted(out)]


def main(stills=False):
    import mujoco
    from PIL import ImageFont
    m = CF.load(MODEL, W, H, floor=False)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("sr2_night").id)
    adr = {n: m.jnt_qposadr[m.joint(n).id] for n in [f"{f}_{j}" for f in F for j in ("x", "y", "bridge", "yaw", "mcp", "pip")]}
    for j in range(m.njnt):                                  # the scene's screwdriver out of the picture
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            d.qpos[m.jnt_qposadr[j]:m.jnt_qposadr[j] + 3] = (0, 0, -5)
    for f in F:
        d.qpos[adr[f"{f}_x"]], d.qpos[adr[f"{f}_y"]], d.qpos[adr[f"{f}_bridge"]] = HOME[f][0], HOME[f][1], HOME[f][1]
        for j in ("yaw", "mcp", "pip"):
            d.qpos[adr[f"{f}_{j}"]] = 0.0
    q0 = d.qpos.copy()

    def put(f, servo_deg):
        d.qpos[:] = q0
        d.qpos[adr[f"{f}_yaw"]] = np.radians(YAW_SIGN[f] * servo_deg)
        mujoco.mj_kinematics(m, d)
        mujoco.mj_camlight(m, d)

    cams, toward = {}, {}
    put("thumb", 0.0)
    centre = np.mean([d.xanchor[m.joint(f"{f}_yaw").id] for f in F], 0)
    for f in F:
        put(f, 0.0)
        ax = d.xaxis[m.joint(f"{f}_yaw").id].copy()
        ax /= np.linalg.norm(ax)
        look = 0.35 * d.xanchor[m.joint(f"{f}_yaw").id] + 0.65 * d.body(f"{f}_tip").xpos
        if np.dot(look - centre, ax) < 0:                    # look along the axis from outside the hand
            ax = -ax
        fwd = -ax                                            # the camera sits at look + DIST * ax, looking along -ax
        toward[f] = ax
        cams[f] = CF.free_cam(look, DIST, float(np.degrees(np.arctan2(fwd[1], fwd[0]))), float(np.degrees(np.arcsin(fwd[2]))))
    r = mujoco.Renderer(m, H, W, max_geom=12000)
    opt = CF.scene_option()
    font = ImageFont.truetype(CF.FONT, 24)
    enc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                            "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-preset",
                            "slow", "-movflags", "+faststart", str(OUT)], stdin=subprocess.PIPE)
    if stills:
        from PIL import Image
        for f in F:
            put(f, 40.0)
            r.update_scene(d, cams[f], scene_option=opt)
            Image.fromarray(r.render()).save(f"/tmp/claude-1001/-home-humanoid-Programs-hand/ec307d8f-8a52-45e1-992b-f83a6be75e2a/scratchpad/still_{f}.jpg")
        r.close()
        return
    poster, n = None, 0
    try:
        def ray(f):
            """From the yaw axis through the fingertip, 30 % past it, shifted 40 mm toward the camera."""
            a = d.xanchor[m.joint(f"{f}_yaw").id].copy()
            t_ = d.body(f"{f}_tip").xpos.copy()
            return a + 0.04 * toward[f], a + 1.3 * (t_ - a) + 0.04 * toward[f]

        def line(ends, rgba):
            if r.scene.ngeom < r.scene.maxgeom:
                g = r.scene.geoms[r.scene.ngeom]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                                    np.array(rgba, np.float32))
                mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.0012, ends[0], ends[1])
                r.scene.ngeom += 1

        for f, cmd, samples in steps():
            put(f, cmd)
            target = ray(f)
            ts = np.array([s[0] for s in samples])
            qs = np.array([s[1] for s in samples])
            final = float(np.median(qs[ts >= ts.max() - 2.0]))
            for k in range(int((SHOW + PAUSE) * FPS)):
                t = min(k / FPS, SHOW)
                i = np.searchsorted(ts, t, side="right") - 1
                q = qs[max(i, 0)] if i >= 0 else 0.0
                put(f, q)
                r.update_scene(d, cams[f], scene_option=opt)
                line(target, (0.85, 0.10, 0.10, 1.0))
                line(ray(f), (0.13, 0.40, 0.80, 1.0))
                text = (f"{f.capitalize()} yaw, step to {cmd:+.0f} deg (servo frame); t = {t:.1f} s; measured {q:+.1f} deg"
                        + (f"; {final - cmd:+.2f} deg at rest" if t >= SHOW else ""))
                frame = CF.label_bar(r.render(), text, font)
                enc.stdin.write(frame.tobytes())
                n += 1
                if poster is None and f == "middle" and abs(cmd) == 40 and t >= SHOW:
                    poster = frame
    finally:
        enc.stdin.close()
        enc.wait()
        r.close()
    if poster is not None:
        from PIL import Image
        Image.fromarray(poster).save(OUT.with_suffix(".jpg"), quality=88)
    print(f"wrote {OUT}: {n} frames, {n / FPS:.0f} s, {OUT.stat().st_size / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main(stills="stills" in sys.argv[1:])
