#!/usr/bin/env python3
"""The deployed three-finger reorientation of the eight bench hands across contact models and simulators.

Each hand D1-D8 replays its deployed plan, unchanged, as the bench ran it: palm fixed, the 24.5 g
screwdriver lying on the 100 mm post, grip held 0.8 s, then the plan's 200-row joint trajectory (1.3 s
more grip, a 1.1 s turn of all three fingers, 1.6 s hold), then 1.5 s of hold (the 2026-09-16 replay,
`real_v1_turn_mechanism.replay`) and 4 s more to watch retention. What varies is the contact:

  fingertip  `legacy`  the sharp 21.1 x 14.8 mm box the plans were exported against
             `sphere`  the 10.55 mm sphere of real_hand.xml (every contact study before 2026-10-06)
             `tpu<r>`  the printed TPU block (fingertip_geometry.py) with fillet r mm on its front edges
  model      MuJoCo point contact on the single geom (`pt`); MuJoCo 1 mm sphere pads on the block (`pads`; `padsT`: pads
             that touch only the tool, the block's mesh taking every other contact,
             stiffness E/h A_s with the inverse weight taken once at load, per finger); Drake hydroelastic
             with the tip as a compliant shape (E 10 MPa, relaxation 0.01 s, SAP)
  impratio   100 (the bed's) or 10000 (Drake's creep on the bed, 2026-10-06)
  plant      `rigid` kp 30 kv 0.5 forcerange 10 N m (shipped); `compliant` kp 0.5 kv 0.02 forcerange 0.35
             (the calibrated gain, as in hom_chain.PLANT); joint damping and frictionloss 0 in both, since
             Drake's SAP has no frictionloss
  seed       tool x, y jitter sd 2 mm and yaw sd 2 deg on the post (seed 0: no jitter)

Friction is 1.0 on every geom (MuJoCo takes the larger of two geoms' coefficients, Drake combines
2 m1 m2 / (m1 + m2)); every model uses a 1 ms step, the elliptic cone and MuJoCo's Newton solver.

    PY=logs/20261001-hom_contact/venv/bin/python
    $PY scripts/reorient_backends.py run --hands D7 --tips tpu6 --models pads --plants compliant --seeds 0
    $PY scripts/reorient_backends.py run --sim drake --hands D7 --tips tpu6 --seeds 0 1 2
Rows: docs/experiments/20261006-fingertip_backends/reorient.jsonl, one fsynced line per rollout.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fingertip_geometry as G  # noqa: E402

OUT = ROOT / "docs/experiments/20261006-fingertip_backends/reorient.jsonl"
SCENES = ROOT / "assets/mjcf/experimental/20261006-reorient_backends"
SCRIPT = "scripts/reorient_backends.py"
FINGERS = ("thumb", "index", "middle")
JOINTS = ("yaw", "mcp", "pip")
OBJ = "screwdriver_medium"
HANDS = {   # D-number: (plan tag, folder), as in real_v1_bench_plants.HANDS
    "D1": ("sv1_w6689_b060", "20260902-residual-bench"),
    "D2": ("sv1_w2360_b075", "20260902-residual-bench"),
    "D3": ("sv1_u1364_b080", "20260902-residual-bench"),
    "D4": ("g12_b095", "20260829-real_v1_deploy"),
    "D5": ("sv1_u0060_b75", "20260830-real_v1-sobol128"),
    "D6": ("sv1_u0308_b050", "20260829-real_v1_deploy"),
    "D7": ("rv05_manual_b85", "20260902-residual-bench"),
    "D8": ("sv1_w0099_b100", "20260829-real_v1_deploy"),
}
# apply_measured_plant.py arguments (masses always the measured ones; joint damping stays the template's 0.5)
PLANTS = {"fast_fl": dict(kp=0.5, kv=0.02, forcerange=0.35, frictionloss=0.0035),   # 2026-09-16 'fast'
          "fast": dict(kp=0.5, kv=0.02, forcerange=0.35, frictionloss=0.0),          # the same, Drake-compatible
          "rigid": dict(kp=30.0, kv=0.5, forcerange=10.0, frictionloss=0.0)}         # shipped gains
NUMERICS = {"scene": None,                                                            # 2 ms, pyramidal, impratio 1
            "bed": dict(timestep="0.001", integrator="implicitfast", cone="elliptic", solver="Newton",
                        iterations="100", ls_iterations="50", tolerance="1e-10")}
MU_SCENE = 2.4             # the deploy scenes' finger and tool friction; MuJoCo takes the larger of two geoms'
DT = 0.001
E_TPU = 1e7
PAD_S, PAD_RS, PAD_TR = 0.001, 0.00075, 0.02
JIT_XY, JIT_YAW = 0.002, math.radians(2.0)
T_GRIP, T_HOLD, T_EXTRA = 0.8, 1.5, 4.0
TRACE_DT = 0.02


def load_plan(hand):
    tag, folder = HANDS[hand]
    p = ROOT / "docs/experiments" / folder / "deploy" / f"{tag}_plan.json"
    plan = json.loads(p.read_text())
    traj = list(csv.DictReader(open(p.with_name(f"{tag}_traj.csv"))))
    return plan, traj


def tip_r(tip):
    return float(tip[3:]) * 1e-3 if tip.startswith("tpu") else None


# ---------------------------------------------------------------------------------- scene building

def _palm_and_start(base: Path, plan: dict):
    """Palm pose, tool pose and finger angles of the plan's replay start, from the scene's own kinematics."""
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(base))
    d = mujoco.MjData(m)
    d.qpos[:] = np.asarray(plan["meta"]["replay_initial_qpos"], float)
    mujoco.mj_forward(m, d)
    pb = m.body("palm_pose").id
    q = {f"{f}_{j}": float(d.qpos[m.jnt_qposadr[m.joint(f"{f}_{j}").id]]) for f in FINGERS for j in JOINTS}
    a = m.jnt_qposadr[m.body(OBJ).jntadr[0]]
    return d.xpos[pb].copy(), d.xquat[pb].copy(), d.qpos[a:a + 7].copy(), q


PLANT_KEYS = {"kp": "kp", "kv": "kv", "fr": "forcerange", "fl": "frictionloss", "dp": "damping"}


def plant_args(plant: str) -> dict:
    """apply_measured_plant.py arguments of a PLANTS key or of a spec 'kp2_kv0.02_fr0.35_fl0_dp0.06' (dp = finger
    joint damping; without it the template's 0.5 stays)."""
    if plant in PLANTS:
        return dict(PLANTS[plant])
    out = {}
    for tok in plant.split("_"):
        k = tok.rstrip("0123456789.e-")
        out[PLANT_KEYS[k]] = float(tok[len(k):])
    return out


def plant_scene(hand: str, plant: str, out_dir: Path = SCENES) -> Path:
    """The plan's bench scene rewritten by apply_measured_plant.py with the plant's gains and the measured masses."""
    import subprocess
    plan, _ = load_plan(hand)
    base = Path(plan["meta"]["scene"])
    out = out_dir / "plants" / f"{HANDS[hand][0]}__{plant}.xml"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        P = plant_args(plant)
        args = ["--kp", f"{P['kp']:g}", "--kv", f"{P['kv']:g}", "--forcerange", f"{P['forcerange']:g}",
                "--frictionloss", f"{P['frictionloss']:g}"]
        if "damping" in P:
            args += ["--damping", f"{P['damping']:g}"]
        tmp = out.with_name(f"{out.stem}.{os.getpid()}.xml")
        subprocess.run([sys.executable, str(ROOT / "scripts/apply_measured_plant.py"), "--scene", str(base),
                        "--out", str(tmp)] + args, check=True, capture_output=True)
        os.replace(tmp, out)
    return out


def build_scene(hand: str, tip: str, model: str, plant: str, numerics: str, ir: float, mu,
                out_dir: Path = SCENES):
    """MJCF of the bench scene and its metadata. The palm is welded at the plan's pose, the finger servos carry
    the plant (apply_measured_plant.py), `numerics` 'scene' keeps the deploy scene's solver options and 'bed'
    sets a 1 ms step, the elliptic cone and impratio `ir`; `mu` 'scene' keeps the scene's friction (2.4 on the
    tool and fingers) or sets every geom's sliding friction; the tip is replaced per `tip` / `model` ('pads' =
    sphere pads on the TPU block, 'pt' = the single geom)."""
    import mujoco
    plan, _ = load_plan(hand)
    base = plant_scene(hand, plant, out_dir)
    palm_p, palm_q, tool7, q0 = _palm_and_start(base, plan)
    root = ET.parse(base).getroot()
    for kf in root.findall("keyframe"):             # sized for the unwelded palm
        root.remove(kf)
    opt = root.find("option")
    if NUMERICS[numerics] is not None:
        for k, v in NUMERICS[numerics].items():
            opt.set(k, v)
        opt.set("impratio", f"{ir:g}")
    # weld the palm
    palm = root.find(".//body[@name='palm_pose']")
    for jn in list(palm.findall("joint")):
        palm.remove(jn)
    palm.attrib.pop("gravcomp", None)
    palm.set("pos", " ".join(f"{v:.7f}" for v in palm_p))
    palm.set("quat", " ".join(f"{v:.8f}" for v in palm_q))
    act = root.find("actuator")
    for a in list(act):
        if a.get("joint", "").startswith("palm_"):
            act.remove(a)
    if mu != "scene":
        for g in root.iter("geom"):
            g.set("friction", f"{float(mu):g} 0.005 0.0001")
        for dg in root.find("default").findall("geom"):
            dg.set("friction", f"{float(mu):g} 0.005 0.0001")
    MU = MU_SCENE if mu == "scene" else float(mu)
    tool_body = root.find(f".//body[@name='{OBJ}']")
    tool_body.set("pos", " ".join(f"{v:.7f}" for v in tool7[:3]))
    tool_body.set("quat", " ".join(f"{v:.8f}" for v in tool7[3:]))
    # scene_mutate.set_object_platform puts the tool's centre AND the post's top at the platform height, so the
    # 12.5 mm tool starts 12.5 mm inside the post and is ejected in the first 20 ms. Shorten the post so the
    # tool rests on it at the plan's pose.
    r_tool = float(tool_body.find("geom").get("size").split()[0])
    post_b = root.find(".//body[@name='tool_platform']")
    post_top = None
    if post_b is not None:
        hh = (tool7[2] - r_tool) / 2.0
        px, py, _ = (float(v) for v in post_b.get("pos").split())
        post_b.set("pos", f"{px:.6f} {py:.6f} {hh:.6f}")
        pg = post_b.find("geom")
        pr = float(pg.get("size").split()[0])
        pg.set("size", f"{pr:.6f} {hh:.6f}")
        post_top = 2 * hh
    meta = {"hand": hand, "tag": HANDS[hand][0], "tip": tip, "model": model, "plant": plant, "impratio": ir,
            "numerics": numerics, "mu": MU, "post_top_z": post_top, "base_scene": str(base), "q0": q0, "tool7": tool7.tolist(), "palm_pos": palm_p.tolist(),
            "palm_quat": palm_q.tolist()}
    meshes, pads_meta = replace_tips(root, tip, model, MU, out_dir, tool=OBJ, tool_only=(model == "padsT"))
    meta.update(pads_meta)
    meta["meshes"] = {str(k): str(v) for k, v in meshes.items()}
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{hand}_{tip}_{model}_{plant}_{numerics}_ir{ir:g}_mu{MU:g}.xml"
    path = out_dir / name
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    ET.ElementTree(root).write(tmp)
    os.replace(tmp, path)
    return path, meta


def replace_tips(root, tip: str, model: str, MU: float, out_dir: Path, tool: str = OBJ, s: float = PAD_S,
                 rs: float = PAD_RS, tr: float = PAD_TR, E: float = E_TPU, region=None, tool_only: bool = False):
    """Replace each fingertip of a parsed real_v1 scene by `tip` ('legacy' keeps the geoms, 'sphere', 'tpu<r>') as
    `model` ('pt' one geom, 'pads' sphere pads at spacing `s`, radius `rs`, relaxation `tr`). `region(C, N, sign)`
    optionally masks the pads (C centres, N normals in the tip frame, sign = the palmar face's x direction). Pad stiffness K_s = E/h A_s is reached at
    load through solimp d0 = 1 - 1/(tc^2 K diag), diag = inverse weights of the tip body and `tool`.
    `tool_only`: the pads collide with the tool alone (contype 2, conaffinity 0; tool geoms contype 2, conaffinity 3)
    and the block's convex mesh takes every other contact (floor, other fingers: contype 4, conaffinity 1).
    Returns (meshes, meta)."""
    import mujoco
    r = tip_r(tip)
    meshes = {}
    if r is not None:
        meshes = G.mesh_paths(out_dir / "meshes", r)
        asset = root.find("asset")
        for sgn, p in meshes.items():
            ET.SubElement(asset, "mesh", name=f"tpu_{'pos' if sgn > 0 else 'neg'}", file=str(p))
    pads = {}
    for f in FINGERS:
        sgn = G.FACE_SIGN[f]
        tipb = root.find(f".//body[@name='{f}_tip']")
        pipb = root.find(f".//body[@name='{f}_pip_frame']")
        old = tipb.findall("geom")
        if tip == "legacy":
            for g in old:
                g.set("name", f"{f}_tipgeom")
            continue
        for g in old:
            tipb.remove(g)
        if tip == "sphere":
            ET.SubElement(tipb, "geom", name=f"{f}_tipgeom", type="sphere", size=f"{G.CAPSULE_R}",
                          material="finger_mat", friction=f"{MU:g} 0.005 0.0001")
            continue
        # TPU block: shorten the distal capsule so its cap ends at the block's top face
        for g in pipb.findall("geom"):
            if g.get("type") == "capsule":
                z_end = -(G.PIP_TO_TIP - (G.CENTRE[2] + G.HALF[2])) + G.CAPSULE_R
                g.set("fromto", f"0 0 0 0 0 {min(z_end, -1e-4):.6f}")
                g.set("name", f"{f}_distal")
        mesh = f"tpu_{'pos' if sgn > 0 else 'neg'}"
        if model == "pt":
            ET.SubElement(tipb, "geom", name=f"{f}_tipgeom", type="mesh", mesh=mesh, material="finger_mat",
                          friction=f"{MU:g} 0.005 0.0001")
        else:
            vis = ET.SubElement(tipb, "geom", name=f"{f}_tipvis", type="mesh", mesh=mesh, contype="0",
                                conaffinity="0", group="1", rgba="0.85 0.55 0.35 0.35")
            if tool_only:
                vis.set("contype", "4")
                vis.set("conaffinity", "1")
                vis.set("friction", f"{MU:g} 0.005 0.0001")
            C, Nn, A, K = G.pad_spheres(r, s, rs, sgn, E)
            if region is not None:
                keep = region(C, Nn, sgn)
                C, Nn, A, K = C[keep], Nn[keep], A[keep], K[keep]
            pads[f] = (C, float(np.median(K)))
            for i, c in enumerate(C):
                g = ET.SubElement(tipb, "geom", name=f"{f}_pad{i}", type="sphere", size=f"{rs}",
                                  pos=f"{c[0]:.7f} {c[1]:.7f} {c[2]:.7f}", condim="3", priority="1",
                                  friction=f"{MU:g} 0 0", rgba="0.75 0.4 0.25 1", mass="0")
                if tool_only:
                    g.set("contype", "2")
                    g.set("conaffinity", "0")
    meta = {}
    if pads and tool_only:
        for g in root.find(f".//body[@name='{tool}']").iter("geom"):
            if g.get("contype", "1") != "0" or g.get("conaffinity", "1") != "0":
                g.set("contype", "2")
                g.set("conaffinity", "3")
    if pads:
        m0 = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        tc = tr / 2.0
        d0s = {}
        for f, (C, K) in pads.items():
            diag = m0.body_invweight0[m0.body(f"{f}_tip").id, 0] + m0.body_invweight0[m0.body(tool).id, 0]
            d0 = 1.0 - 1.0 / (tc ** 2 * K * diag)
            if d0 < 0.05:
                raise ValueError(f"{f}: pad d0 {d0:.3f} below 0.05 at relaxation {tr} s")
            d0s[f] = d0
            tipb = root.find(f".//body[@name='{f}_tip']")
            for g in tipb.findall("geom"):
                if g.get("name", "").startswith(f"{f}_pad"):
                    g.set("solref", f"{tc:.6g} 1")
                    g.set("solimp", f"{d0:.6g} {d0:.6g} 0.001 0.5 2")
        meta = dict(pad_d0=d0s, pad_K=float(np.median([k for _, k in pads.values()])),
                    n_pads={f: len(c) for f, (c, _) in pads.items()})
    return meshes, meta


# ---------------------------------------------------------------------------------- schedule

def schedule(plan, traj):
    """[(t_end, targets dict joint -> rad)] segments of the bench replay: grip, trajectory rows, hold, extra."""
    poses = {p["name"]: p["joints"] for p in plan["poses"]}
    grip = {f"{f}_{j}": math.radians(poses["grip"][f][j]) for f in FINGERS for j in JOINTS}
    segs = [(T_GRIP, grip, "grip")]
    span = float(traj[-1]["t_s"]) or 1.6
    t = T_GRIP
    last = grip
    for r in traj:
        t += span / len(traj)
        last = {f"{f}_{j}": math.radians(float(r[f"{f}_{j}_deg"])) for f in FINGERS for j in JOINTS}
        segs.append((t, last, "traj"))
    segs.append((t + T_HOLD, last, "hold"))
    segs.append((t + T_HOLD + T_EXTRA, last, "extra"))
    return segs


JITTER_MODE = "j2"


def jitter(seed, mode=None):
    """Tool placement error (dx, dy, dyaw). 'j2': seed 0 none, else sd 2 mm and 2 deg; 'j0916': the 2026-09-16
    replay's draws, sd 4 mm in x and y from default_rng(1000 + seed), no yaw."""
    mode = mode or JITTER_MODE
    if mode == "j0916":
        rng = np.random.default_rng(1000 + seed)
        return float(rng.normal(0, 0.004)), float(rng.normal(0, 0.004)), 0.0
    if seed == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(1000 + seed)
    return float(rng.normal(0, JIT_XY)), float(rng.normal(0, JIT_XY)), float(rng.normal(0, JIT_YAW))


def _yaw_quat(q, yaw):
    c, s = math.cos(yaw / 2), math.sin(yaw / 2)
    w1, x1, y1, z1 = c, 0.0, 0.0, s
    w2, x2, y2, z2 = q
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def score(trace, segs):
    """Bench metrics from a trace of (t, cos, x, y, z, pads{finger: N}, n_pads{finger: n})."""
    t = np.array([s["t"] for s in trace])
    def at(tt):
        return trace[int(np.clip(np.searchsorted(t, tt - 1e-9), 0, len(trace) - 1))]
    t_traj_end = [e for e, _, k in segs if k == "traj"][-1]
    g = at(T_GRIP)
    h = at(t_traj_end + T_HOLD)
    e = trace[-1]
    def turn(a, b):
        return math.degrees(math.acos(max(-1, min(1, a["cos"]))) - math.acos(max(-1, min(1, b["cos"]))))
    out = {"cos_grip": g["cos"], "cos_hold": h["cos"], "cos_end": e["cos"],
           "turn_hold_deg": turn(g, h), "turn_end_deg": turn(g, e),
           "z_grip": g["z"], "z_hold": h["z"], "z_end": e["z"],
           "dropped_hold": bool(h["z"] < g["z"] - 0.020), "dropped_end": bool(e["z"] < g["z"] - 0.020),
           "pads_hold": int(sum(1 for f in FINGERS if h["n"][f] > 0)), "pads_end": int(sum(1 for f in FINGERS if e["n"][f] > 0)),
           "F_pads_grip_N": round(sum(g["F"].values()), 4), "F_pads_hold_N": round(sum(h["F"].values()), 4),
           "F_pads_end_N": round(sum(e["F"].values()), 4),
           "drift_hold_to_end_mm": float(1e3 * math.dist((h["x"], h["y"], h["z"]), (e["x"], e["y"], e["z"]))),
           "dcos_hold_to_end": e["cos"] - h["cos"]}
    out["held_hold"] = (not out["dropped_hold"]) and out["pads_hold"] >= 2
    out["held_end"] = (not out["dropped_end"]) and out["pads_end"] >= 2
    return out


# ---------------------------------------------------------------------------------- MuJoCo

def run_mujoco(scene: Path, meta: dict, plan, traj, seed: int, film: Path | None = None):
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m)
    tb = m.body(OBJ).id
    qa = m.jnt_qposadr[m.body_jntadr[tb]]
    tool7 = np.asarray(meta["tool7"], float)
    dx, dy, dyaw = jitter(seed)
    d.qpos[qa:qa + 3] = tool7[:3] + [dx, dy, 0.0]
    d.qpos[qa + 3:qa + 7] = _yaw_quat(tool7[3:], dyaw)
    act = {}
    for f in FINGERS:
        for j in JOINTS:
            n = f"{f}_{j}"
            d.qpos[m.jnt_qposadr[m.joint(n).id]] = meta["q0"][n]
            act[n] = m.actuator(f"a_{n}").id
    mujoco.mj_forward(m, d)
    tool_geoms = {i for i in range(m.ngeom) if m.geom_bodyid[i] == tb}
    finger_of = {}
    for i in range(m.ngeom):
        bn = m.body(m.geom_bodyid[i]).name
        for f in FINGERS:
            if bn.startswith(f + "_"):
                finger_of[i] = f
    f6 = np.zeros(6)
    segs = schedule(plan, traj)
    trace, W = [], []
    renderer = frames = None
    if film is not None:
        renderer = mujoco.Renderer(m, 360, 480)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance, cam.azimuth, cam.elevation = 0.30, 150.0, -20.0
        frames = []

    def snap():
        F = {f: 0.0 for f in FINGERS}
        n = {f: 0 for f in FINGERS}
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom[0]), int(c.geom[1])
            if g1 not in tool_geoms and g2 not in tool_geoms:
                continue
            o = g2 if g1 in tool_geoms else g1
            f = finger_of.get(o)
            if f is None:
                continue
            mujoco.mj_contactForce(m, d, i, f6)
            if f6[0] > 1e-6:
                F[f] += float(f6[0])
                n[f] += 1
        R9 = np.zeros(9)
        mujoco.mju_quat2Mat(R9, d.qpos[qa + 3:qa + 7])
        p = d.qpos[qa:qa + 3]
        trace.append({"t": round(float(d.time), 4), "cos": float(R9[8]), "x": float(p[0]), "y": float(p[1]),
                      "z": float(p[2]), "F": {k: round(v, 4) for k, v in F.items()}, "n": n})

    dt = m.opt.timestep
    k_trace = max(1, int(round(TRACE_DT / dt)))
    k_film = max(1, int(round(1 / 25 / dt)))
    step = 0
    bad = False
    for t_end, tgt, _ in segs:
        for n_, a in act.items():
            d.ctrl[a] = tgt[n_]
        while d.time < t_end - 1e-9:
            w0 = time.perf_counter()
            mujoco.mj_step(m, d)
            W.append(time.perf_counter() - w0)
            step += 1
            if step % k_trace == 0:
                snap()
                if not np.all(np.isfinite(d.qpos)) or abs(d.qpos[qa + 2]) > 2.0:
                    bad = True
                    break
            if renderer is not None and step % k_film == 0:
                cam.lookat[:] = d.xpos[tb]
                renderer.update_scene(d, cam)
                frames.append(renderer.render().copy())
        if bad:
            break
    if renderer is not None and frames:
        import imageio
        film.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(str(film), frames, fps=25, codec="libx264", quality=6)
    res = score(trace, segs) if not bad else {}
    res.update(status="ejected" if bad else "complete", us_per_step_median=float(np.median(W) * 1e6),
               ncon_max=None, n_steps=step)
    return res, trace


# ---------------------------------------------------------------------------------- Drake

class DrakeBench:
    """The same scene in Drake 1.57: parsed from the MJCF with the tips removed, each tip registered as a
    compliant hydroelastic shape (TPU block convex, sphere or legacy box), the distal capsules compliant, tool,
    post and floor rigid; PD servos with the plant's gains and effort limit; the hand's exclude pairs as
    collision filters."""

    def __init__(self, scene: Path, meta: dict, rt: float = 0.01, E: float = E_TPU):
        from pydrake.all import (AddCompliantHydroelasticProperties, AddContactMaterial, AddMultibodyPlant,
                                 AddRigidHydroelasticProperties, Box, CollisionFilterDeclaration, Convex,
                                 CoulombFriction, DiagramBuilder, GeometrySet, MultibodyPlantConfig, Parser,
                                 PdControllerGains, ProximityProperties, RigidTransform, RoleAssign, Simulator,
                                 Sphere)
        self.meta = meta
        root = ET.parse(scene).getroot()
        tipgeom = {}
        for f in FINGERS:
            tipb = root.find(f".//body[@name='{f}_tip']")
            for g in list(tipb.findall("geom")):
                if g.get("contype") != "0" and g.get("type") in ("box", "sphere", "mesh"):
                    tipgeom[f] = dict(g.attrib)
                tipb.remove(g)
        for tag in ("visual", "keyframe", "actuator", "option", "statistic", "size", "contact"):
            for el in root.findall(tag):
                root.remove(el)
        asset = root.find("asset")
        if asset is not None:
            for el in list(asset):
                if el.tag in ("mesh", "texture"):
                    asset.remove(el)
        for parent in root.iter():
            for child in list(parent):
                if child.tag in {"site", "light", "texture", "position", "motor", "camera"}:
                    parent.remove(child)
            if parent.tag == "material":
                for attr in ("texture", "texuniform", "texrepeat", "reflectance"):
                    parent.attrib.pop(attr, None)
            if parent.tag in ("geom", "default", "joint", "body"):
                for attr in ("solref", "solimp", "condim", "frictionloss", "contype", "conaffinity", "group",
                             "priority", "gravcomp", "solmix", "margin", "gap"):
                    parent.attrib.pop(attr, None)
        for parent in root.iter():
            for child in list(parent):
                if child.tag == "geom" and child.get("rgba") in ("0.85 0.3 0.2 1", "0.12 0.13 0.16 1"):
                    parent.remove(child)
        b = DiagramBuilder()
        cfg = MultibodyPlantConfig(time_step=DT, discrete_contact_approximation="sap",
                                   contact_model="hydroelastic_with_fallback")
        plant, sg = AddMultibodyPlant(cfg, b)
        parser = Parser(plant, sg)
        self.instance, = parser.AddModelsFromString(ET.tostring(root, encoding="unicode"), "xml")
        P = plant_args(meta["plant"])
        self.joints = [f"{f}_{j}" for f in FINGERS for j in JOINTS]
        for n in self.joints:
            j = plant.GetJointByName(n)
            a = plant.AddJointActuator("a_" + n, j, P["forcerange"])
            a.set_controller_gains(PdControllerGains(p=P["kp"], d=P["kv"]))
            a.set_default_rotor_inertia(0.001)
            a.set_default_gear_ratio(1.0)
        sid = plant.get_source_id()
        self.finger_gid = {}

        def props(kind):
            pp = ProximityProperties()
            if kind == "compliant":
                AddCompliantHydroelasticProperties(0.001, E, pp)
            else:
                AddRigidHydroelasticProperties(0.0005, pp)
            AddContactMaterial(dissipation=10.0, point_stiffness=1e4,
                               friction=CoulombFriction(meta["mu"], meta["mu"]), properties=pp)
            pp.AddProperty("material", "relaxation_time", rt)
            return pp
        for f in FINGERS:
            body = plant.GetBodyByName(f"{f}_tip")
            ga = tipgeom[f]
            pos = np.array([float(v) for v in ga.get("pos", "0 0 0").split()])
            X = RigidTransform(pos)
            if ga["type"] == "sphere":
                shape = Sphere(float(ga["size"].split()[0]))
            elif ga["type"] == "box":
                hx, hy, hz = (float(v) for v in ga["size"].split())
                shape = Box(2 * hx, 2 * hy, 2 * hz)
            else:
                shape = Convex(meta["meshes"][str(G.FACE_SIGN[f])])
                X = RigidTransform()
            gid = plant.RegisterCollisionGeometry(body, X, shape, f"{f}_tipgeom", props("compliant"))
            self.finger_gid[gid] = f
        plant.Finalize()
        inspector = sg.model_inspector()
        for f in FINGERS:
            for bn in (f"{f}_pip_frame",):
                for gid in plant.GetCollisionGeometriesForBody(plant.GetBodyByName(bn)):
                    sg.AssignRole(sid, gid, props("compliant"), RoleAssign.kReplace)
                    self.finger_gid[gid] = f
            for bn in (f"{f}_yaw_frame", f"{f}_mcp_frame"):
                for gid in plant.GetCollisionGeometriesForBody(plant.GetBodyByName(bn)):
                    sg.AssignRole(sid, gid, props("rigid"), RoleAssign.kReplace)
                    self.finger_gid[gid] = f
        self.tool = plant.GetBodyByName(OBJ)
        self.tool_gids = set(plant.GetCollisionGeometriesForBody(self.tool))
        for gid in self.tool_gids:
            sg.AssignRole(sid, gid, props("rigid"), RoleAssign.kReplace)
        for gid in plant.GetCollisionGeometriesForBody(plant.world_body()):
            sg.AssignRole(sid, gid, props("rigid"), RoleAssign.kReplace)
        for bn in ("palm_pose",):
            try:
                for gid in plant.GetCollisionGeometriesForBody(plant.GetBodyByName(bn)):
                    sg.AssignRole(sid, gid, props("rigid"), RoleAssign.kReplace)
            except RuntimeError:
                pass
        cfm = sg.collision_filter_manager()
        for e in ET.parse(scene).getroot().iter("exclude"):
            g1 = plant.GetCollisionGeometriesForBody(plant.GetBodyByName(e.get("body1")))
            g2 = plant.GetCollisionGeometriesForBody(plant.GetBodyByName(e.get("body2")))
            if g1 and g2:
                cfm.Apply(CollisionFilterDeclaration().ExcludeBetween(GeometrySet(g1), GeometrySet(g2)))
        # a finger's own bodies never touch each other
        for f in FINGERS:
            gs = []
            for bn in (f"{f}_yaw_frame", f"{f}_mcp_frame", f"{f}_pip_frame", f"{f}_tip"):
                gs += plant.GetCollisionGeometriesForBody(plant.GetBodyByName(bn))
            cfm.Apply(CollisionFilterDeclaration().ExcludeWithin(GeometrySet(gs)))
        self.plant, self.sg = plant, sg
        self.diagram = b.Build()
        self.sim = Simulator(self.diagram)
        self.ctx = self.sim.get_mutable_context()
        self.pc = plant.GetMyMutableContextFromRoot(self.ctx)
        self.act_order = [plant.get_joint_actuator(i).joint().name()
                          for i in plant.GetJointActuatorIndices(self.instance)]

    def reset(self, q0, tool7, seed):
        from pydrake.all import Quaternion, RigidTransform
        p = self.plant
        p.SetVelocities(self.pc, np.zeros(p.num_velocities()))
        for n in self.joints:
            p.GetJointByName(n).set_angle(self.pc, float(q0[n]))
        dx, dy, dyaw = jitter(seed)
        q = _yaw_quat(np.asarray(tool7[3:], float), dyaw)
        X = RigidTransform(Quaternion(q / np.linalg.norm(q)), np.asarray(tool7[:3], float) + [dx, dy, 0.0])
        p.SetFreeBodyPose(self.pc, self.tool, X)
        self.ctx.SetTime(0.0)
        self.sim.Initialize()

    def set_targets(self, tgt):
        pos = np.array([tgt[n] for n in self.act_order])
        self.plant.get_desired_state_input_port(self.instance).FixValue(self.pc, np.r_[pos, np.zeros(len(pos))])
        self.plant.get_actuation_input_port().FixValue(self.pc, np.zeros(len(pos)))

    def snap(self):
        p = self.plant
        X = p.EvalBodyPoseInWorld(self.pc, self.tool)
        R = X.rotation().matrix()
        F = {f: 0.0 for f in FINGERS}
        n = {f: 0 for f in FINGERS}
        cr = p.get_contact_results_output_port().Eval(self.pc)
        for i in range(cr.num_hydroelastic_contacts()):
            info = cr.hydroelastic_contact_info(i)
            srf = info.contact_surface()
            ids = (srf.id_M(), srf.id_N())
            if not (set(ids) & self.tool_gids):
                continue
            f = self.finger_gid.get(ids[0]) or self.finger_gid.get(ids[1])
            if f is None:
                continue
            Fv = np.array(info.F_Ac_W().translational())
            nrm = srf.face_normal(0) if srf.num_faces() else np.zeros(3)
            fn = float(abs(Fv @ np.asarray(nrm))) if np.linalg.norm(nrm) > 0 else float(np.linalg.norm(Fv))
            if fn > 1e-6:
                F[f] += fn
                n[f] += 1
        for i in range(cr.num_point_pair_contacts()):
            info = cr.point_pair_contact_info(i)
            pp = info.point_pair()
            if not ({pp.id_A, pp.id_B} & self.tool_gids):
                continue
            f = self.finger_gid.get(pp.id_A) or self.finger_gid.get(pp.id_B)
            if f is None:
                continue
            fn = float(abs(np.array(info.contact_force()) @ np.array(pp.nhat_BA_W)))
            if fn > 1e-6:
                F[f] += fn
                n[f] += 1
        t = X.translation()
        return {"t": round(float(self.ctx.get_time()), 4), "cos": float(R[2, 2]), "x": float(t[0]),
                "y": float(t[1]), "z": float(t[2]), "F": {k: round(v, 4) for k, v in F.items()}, "n": n}


def run_drake(db: "DrakeBench", meta: dict, plan, traj, seed: int):
    """One rollout on a DrakeBench built once per scene (a new diagram per seed is never freed)."""
    db.reset(meta["q0"], meta["tool7"], seed)
    segs = schedule(plan, traj)
    trace, W = [], []
    bad = False
    t = 0.0
    for t_end, tgt, _ in segs:
        db.set_targets(tgt)
        while t < t_end - 1e-9:
            t_next = min(t_end, (math.floor(t / TRACE_DT + 1e-6) + 1) * TRACE_DT)
            w0 = time.perf_counter()
            db.sim.AdvanceTo(t_next)
            W.append((time.perf_counter() - w0) / max(1, round((t_next - t) / DT)))
            t = t_next
            if abs((t / TRACE_DT) - round(t / TRACE_DT)) < 1e-6:
                s = db.snap()
                trace.append(s)
                if not np.isfinite(s["z"]) or abs(s["z"]) > 2.0:
                    bad = True
                    break
        if bad:
            break
    res = score(trace, segs) if not bad else {}
    res.update(status="ejected" if bad else "complete", us_per_step_median=float(np.median(W) * 1e6),
               n_steps=int(round(t / DT)))
    return res, trace


# ---------------------------------------------------------------------------------- driver

def append_row(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(row, default=float) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def done(path):
    have = set()
    if Path(path).exists():
        for line in open(path):
            try:
                r = json.loads(line)
                have.add(r["case_id"])
            except Exception:
                pass
    return have


def git_rev():
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True).stdout.strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--sim", default="mujoco", choices=["mujoco", "drake"])
    r.add_argument("--hands", nargs="+", default=list(HANDS))
    r.add_argument("--tips", nargs="+", default=["tpu6"])
    r.add_argument("--models", nargs="+", default=["pads"], help="mujoco: pads | pt")
    r.add_argument("--plants", nargs="+", default=["fast"])
    r.add_argument("--numerics", nargs="+", default=["bed"], help="scene | bed")
    r.add_argument("--ir", nargs="+", type=float, default=[100.0])
    r.add_argument("--mu", nargs="+", default=["scene"], help="scene (2.4) or a value")
    r.add_argument("--seeds", nargs="+", type=int, default=[0])
    r.add_argument("--jitter", default="j2", choices=["j2", "j0916"])
    r.add_argument("--out", type=Path, default=OUT)
    r.add_argument("--film", action="store_true", help="MuJoCo: film seed 0 into media/")
    r.add_argument("--traces", action="store_true", help="keep the 50 Hz trace in the row")
    a = ap.parse_args()
    global JITTER_MODE
    JITTER_MODE = a.jitter
    have = done(a.out)
    models = a.models if a.sim == "mujoco" else ["hydro"]
    for hand in a.hands:
        plan, traj = load_plan(hand)
        for tip in a.tips:
            for model in models:
                if a.sim == "mujoco" and model == "pads" and not tip.startswith("tpu"):
                    continue
                for plant in a.plants:
                    for numerics in (a.numerics if a.sim == "mujoco" else ["bed"]):
                        for ir in (a.ir if (a.sim == "mujoco" and numerics == "bed") else [0.0 if a.sim == "drake" else 1.0]):
                          for mu in a.mu:
                            scene = meta = db = None
                            import gc
                            gc.collect()
                            for seed in a.seeds:
                                cid = f"{a.sim}|{hand}|{tip}|{model}|{plant}|{numerics}|{ir:g}|{mu}|{a.jitter}|{seed}"
                                if cid in have:
                                    continue
                                t0 = time.time()
                                row = {"case_id": cid, "sim": a.sim, "hand": hand, "tag": HANDS[hand][0], "tip": tip,
                                       "model": model, "plant": plant, "numerics": numerics,
                                       "impratio": ir if a.sim == "mujoco" else None, "mu": mu, "jitter_mode": a.jitter,
                                       "seed": seed, "jitter": jitter(seed)}
                                try:
                                    if scene is None:
                                        scene, meta = build_scene(hand, tip, "pads" if model == "pads" else "pt", plant,
                                                                  numerics, ir if ir > 0 else 100.0, mu)
                                    film = None
                                    if a.film and seed == a.seeds[0] and a.sim == "mujoco":
                                        film = OUT.parent / "media" / f"{hand}_{tip}_{model}_{plant}_{numerics}_ir{ir:g}_mu{mu}.mp4"
                                    if a.sim == "mujoco":
                                        res, trace = run_mujoco(scene, meta, plan, traj, seed, film)
                                    else:
                                        if db is None:
                                            db = DrakeBench(scene, meta)
                                        res, trace = run_drake(db, meta, plan, traj, seed)
                                    row.update(res)
                                    row["film"] = str(film.relative_to(ROOT)) if film else None
                                    row["scene"] = str(scene.relative_to(ROOT))
                                    row["n_pads"] = meta.get("n_pads")
                                    row["pad_d0"] = meta.get("pad_d0")
                                    if a.traces:
                                        row["trace"] = trace
                                    else:
                                        row["trace_coarse"] = [[s_["t"], round(s_["cos"], 4), round(s_["z"], 5),
                                                                round(sum(s_["F"].values()), 3),
                                                                sum(1 for f in FINGERS if s_["n"][f] > 0)]
                                                               for s_ in trace[::5]]
                                except Exception as e:
                                    row.update(status="failed", error=repr(e), traceback=traceback.format_exc()[-2000:])
                                row.update(script=SCRIPT, git_rev=git_rev(), wall_s=time.time() - t0,
                                           when=time.strftime("%Y-%m-%d %H:%M"))
                                append_row(a.out, row)
                                print(f"{cid:64s} {row.get('status'):9s} turn {row.get('turn_hold_deg', float('nan')):+6.1f}"
                                      f"/{row.get('turn_end_deg', float('nan')):+6.1f} deg cos {row.get('cos_end', float('nan')):+.3f}"
                                      f" held {row.get('held_hold')}/{row.get('held_end')} pads {row.get('pads_end')} "
                                      f"F {row.get('F_pads_end_N')} N  {row.get('us_per_step_median', 0):.0f} us/step "
                                      f"wall {row['wall_s']:.1f} s", flush=True)

if __name__ == "__main__":
    main()
