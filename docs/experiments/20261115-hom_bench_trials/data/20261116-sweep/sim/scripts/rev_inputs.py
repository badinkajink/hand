"""HOM's inputs as on the hand (hand-4a, 2026-11-16, for the corvus session: "(1) HOM's pose input delayed 60 ms and
refreshed at 30 Hz ... (2) HOM's mirror given the stem object (offset centre of mass, stem and block inertia)").

The closed-loop runs so far (hom_bench.simulate) gave HOM the plant's pose every period, each finger's centre of pressure
and normal force (contacts on), and a model of the bare ball. On the hand HOM gets the tag tracker's pose, 57-67 ms old at
use (median) and new in about half the periods, no fingertip contacts (no trial streamed them), and keeps the bare-ball
model. Variants, each on the stem scene with the soft tangential contact (friction 1.0, impratio 1) and the server's plant:
  exact          the runs so far
  no contacts    HOM gets no centre of pressure or force, so its contact frames come from the model's witness points
  late pose      the pose of a 30 Hz camera delivered 42 ms after its capture (the tracker's measured 29.9 Hz and 42 ms
                 frame age, 2026-11-13); at use it is 42-75 ms old
  bench inputs   no contacts and the late pose
  stem model     HOM's model (the mirror) carries the stem and block: its object's centre of mass and inertia are the
                 real object's
  all            bench inputs and the stem model
Per run over the task (2.0 s to the schedule's end), servo degrees: the middle yaw's and thumb MCP's target minus measured
(median), the thumb PIP's median; the twist at the end, rise and tilt (rev_sweep.feats_arrays); tracking 1 - err/ref.

    OMP_NUM_THREADS=1 .venv/bin/python rev_inputs.py OUT.json [--workers 8]
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path.home() / "Programs/hand"
D = ROOT / "logs/20261116-sim_real_reversal"
sys.path.insert(0, str(D))
import rev_lat as RL  # noqa: E402
import rev_rescore as RR  # noqa: E402

GRASPS = [("g10_lat8", "lat", 329, 3), ("g10_lat4", "lat", 329, 4), ("gr_g13_x105_y70", "grid2dgr", 329, 13),
          ("gr_g0_x70_y110", "grid2dgr", 162, 0)]
VARIANTS = {"exact": dict(contacts=True, late=False, stem=False),
            "no contacts": dict(contacts=False, late=False, stem=False),
            "late pose": dict(contacts=True, late=True, stem=False),
            "bench inputs": dict(contacts=False, late=True, stem=False),
            "stem model": dict(contacts=True, late=False, stem=True),
            "all": dict(contacts=False, late=True, stem=True)}
SCENE = D / "scenes" / "sph040_stem_block_tape0_imp1.xml"      # the plant's scene; never rebuilt here
STEM_PT = D / "scenes" / "sph040_stem_block_pt_mirror.xml"       # HOM's model with the stem and block
LATENCY_S, RATE_HZ = 0.042, 30.0
_G = {}


def stem_pt(src: Path) -> Path:
    """HOM's model scene (the pt scene of the reference cell) with rev_stem.build's object: ball 21.0 g, stem 36.9 mm
    of 0.6 g, block 6.5 g, and the post replaced by the table under the block."""
    x = src.read_text()
    old = '<geom type="sphere" size="0.020000" mass="0.025" material="object_mat" friction="1 0.005 0.0001" />'
    assert x.count(old) == 1, "object geom"
    r, stem, bh = 0.020, 0.03689, 0.017
    y_s0, y_s1 = -r, -(r + stem)
    new = (f'<geom type="sphere" size="0.020000" mass="0.02100" material="object_mat" friction="1 0.005 0.0001" />'
           f'<geom name="obj_stem" type="cylinder" fromto="0 {y_s0 + 0.001:.5f} 0 0 {y_s1:.5f} 0" size="0.0020" '
           f'mass="0.00060" material="object_mat" friction="1 0.005 0.0001" />'
           f'<geom name="obj_block" type="cylinder" fromto="0 {y_s1:.5f} 0 0 {y_s1 - bh:.5f} 0" size="0.0150" '
           f'mass="0.00650" material="object_mat" friction="1 0.005 0.0001" />')
    x = x.replace(old, new)
    m = re.search(r'<body name="post" pos="[^"]*"><geom name="post"[^/]*/></body>', x)
    assert m, "post"
    z_top = 0.12 - (r + stem + bh)
    x = x[:m.start()] + (f'<body name="post" pos="0 0 {z_top - 0.004:.5f}"><geom name="post" type="box" '
                         f'size="0.06 0.06 0.004" rgba="0.4 0.4 0.45 1" friction="1 0.005 0.0001" /></body>') + x[m.end():]
    STEM_PT.write_text(x)
    return STEM_PT


def init(path):
    import sr2_hom_replay as RP
    toy = RP.TOY_ROWS
    RL.init(path)
    _G["RP"], _G["lat_rows"], _G["toy_rows"] = RP, RP.TOY_ROWS, toy


def simulate(HB, bench, seed, contacts=True, late=False):
    """hom_bench.simulate with HOM's pose taken from a 30 Hz camera LATENCY_S late when `late`, and its contacts
    switched off when not `contacts`; the plant, the schedule and the metrics as there."""
    T = bench.T
    plant = T.Plant(bench.p_pads, bench.setup)
    plant.hold_s = bench.hold_s
    plant.reset(bench.q_open, seed)
    bench.reset()
    dt = bench.prm.dt
    t, lifted, s0, t_cap = 0.0, False, None, 0.0
    stream, rows, caps = [], [], []
    while t < bench.t_end - 1e-9:
        s = plant.state()
        if t >= T.T_LIFT and not lifted:
            plant.post_off()
            lifted = True
        if s0 is None and t >= T.T_LIFT + 0.4 - 1e-9:
            s0 = s
        smp = HB.bench_sample(bench, s)
        pos, quat = smp["obj_pos_mm"], smp["obj_quat"]
        if late:
            while t_cap <= t + 1e-9:                       # the captures since the last period see this state
                caps.append((t_cap, pos, quat))
                t_cap += 1.0 / RATE_HZ
            ready = [c for c in caps if c[0] + LATENCY_S <= t + 1e-9]
            _tc, pos, quat = ready[-1] if ready else caps[0]
        out, info = bench.step(t, smp["joints"], pos, quat, smp["contacts"] if contacts else None)
        plant.set_targets(HB.sim_from_servo(out))
        stream.append({"t": round(t, 4), **smp, "targets": out, "phase": info["phase"], "clipped": info["clipped"]})
        rows.append(HB._metric_row(bench.task, s, info["schedule"], t))
        t += dt
        plant.advance(t)
    return stream, T.metrics(bench.task, bench.setup, rows, s0, plant.state())


def run(job):
    try:
        return _run(*job)
    except Exception as e:                                       # noqa: BLE001  HOM's least squares on a dropped sphere
        return {"grasp": job[0], "seed": job[4], "variant": job[5], "error": repr(e)[:120]}


def _run(name, src, call, k, seed, variant):
    v = VARIANTS[variant]
    RP = _G["RP"]
    RP.TOY_ROWS = _G["lat_rows"] if src == "lat" else _G["toy_rows"]
    from sr2hand.control import hom_bench as HB
    import mujoco
    import hand_object_scale_sim as S
    import hom_tasks as T
    from rev_sweep import feats_arrays
    params = {**RR.SERVER, "mu": 1.0}
    setup, mxy, _layout = RP.toy_setup("twist", 40.0, call, k, "grid2dff" if src == "lat" else src)
    p_pads, p_pt = RP.ref_scene("sphere", 40.0)
    m_pt = mujoco.MjModel.from_xml_path(str(STEM_PT if v["stem"] else p_pt))
    RP.move_mounts(m_pt, mxy, RP.mount_base(m_pt))
    q_open, _ = T.open_pose(setup, m_pt)
    S.set_palm(m_pt, setup["palm"])
    bench = HB.HomBench("twist", 40.0, setup, m_pt, q_open, dt=HB.DT_BENCH, p_pads=p_pads)
    bench.T = RP._Tasks(lambda path, st: RP.SR2Plant(path, st, mxy, params, 0.0))
    w0 = time.perf_counter()
    stream, res = simulate(HB, bench, seed, v["contacts"], v["late"])
    t = np.array([float(s["t"]) for s in stream])
    P = np.array([s["obj_pos_mm"] for s in stream], float)
    Q = np.array([s["obj_quat"] for s in stream], float)
    f = feats_arrays(t, P, Q)
    task = [s for s in stream if 2.0 - 1e-9 <= float(s["t"]) <= bench.t_end + 1e-9]
    gap = (lambda fi, j: round(float(np.median([float(s["targets"][fi][j]) - float(s["joints"][fi][j]) for s in task])), 1))
    err, ref = res.get("track_err_rms_deg"), res.get("track_ref_rms_deg")
    return {"grasp": name, "seed": seed, "variant": variant, "held": bool(res.get("held")),
            "track": None if not err or not ref else round(max(0.0, 1.0 - err / ref), 3),
            "twist_end": round(float(f["twist_end"]), 1), "rise_max": round(float(f["rise_max"]), 2),
            "tilt_end": round(float(f["tilt_end"]), 1), "gap_middle_aa": gap("middle", "aa"),
            "gap_index_aa": gap("index", "aa"), "gap_thumb_fe1": gap("thumb", "fe1"),
            "thumb_pip_median": round(float(np.median([float(s["joints"]["thumb"]["fe2"]) for s in task])), 1),
            "wall_s": round(time.perf_counter() - w0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    assert SCENE.exists(), SCENE
    RL.rows_file()
    import sr2_hom_replay as RP
    stem_pt(Path(RP.ref_scene("sphere", 40.0)[1]))
    import multiprocessing as mp
    jobs = [(n, s, c, k, seed, var) for var in VARIANTS for (n, s, c, k) in GRASPS for seed in (0, 1, 2)]
    with mp.get_context("fork").Pool(a.workers, initializer=init, initargs=(str(SCENE),)) as pool:
        rows = pool.map(run, jobs)
    Path(a.out).write_text(json.dumps(rows, indent=1) + "\n")
    for var in VARIANTS:
        print(var, flush=True)
        for n, *_ in GRASPS:
            rs = [r for r in rows if r["variant"] == var and r["grasp"] == n and "error" not in r]
            bad = sum(1 for r in rows if r["variant"] == var and r["grasp"] == n and "error" in r)
            if not rs:
                print(f"   {n:16s} all failed", flush=True)
                continue
            m = lambda key: np.mean([r[key] for r in rs])                              # noqa: E731
            print(f"   {n:16s} held {sum(r['held'] for r in rs)}/{len(rs) + bad} track {np.mean([r['track'] or 0 for r in rs]):.2f}"
                  f" twist {m('twist_end'):5.1f} rise {m('rise_max'):5.1f} | gap middle.aa {m('gap_middle_aa'):+6.1f} "
                  f"index.aa {m('gap_index_aa'):+5.1f} thumb.fe1 {m('gap_thumb_fe1'):+5.1f} | thumb PIP med "
                  f"{m('thumb_pip_median'):+6.1f}" + (f" | {bad} failed" if bad else ""), flush=True)


if __name__ == "__main__":
    main()
