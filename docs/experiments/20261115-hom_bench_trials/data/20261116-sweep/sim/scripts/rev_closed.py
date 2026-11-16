"""Closed-loop HOM twist in the CPU plant of sr2_hom_replay on the four grasps of 2026-11-16's real trials (film:
grid2dff g10 at call 329 (105,70), g11 at call 162 (70,110); ground-clear: grid2dgr g13 at 329, g0 at 162), seeds 0-2,
on a chosen scene and plant: hom_landscape.track_score (J) and the trajectory features of rev_sweep (rise along z_P,
twist about z_P, tilt). --scene stem: the plant runs the real object (rev_stem: ball, stem, block, a table under the
block until the lift); the controller keeps its own model of the bare ball, as the real HOM server does.
    .venv/bin/python rev_closed.py OUT.json --scene default|stem [--set mu=0.6 ...] [--seeds 0 1 2]
"""
import sys, json, argparse, time
from pathlib import Path
import numpy as np
ROOT = Path.home() / "Programs/hand"; D = ROOT / "logs/20261116-sim_real_reversal"
sys.path.insert(0, str(D)); sys.path.insert(0, str(ROOT / "scripts/hardware")); sys.path.insert(0, str(ROOT / "scripts"))
STEM = D / "scenes" / "sph040_stem_block.xml"
GRASPS = [("film_g10_x105_y70", "grid2dff", 329, 10), ("film_g11_x70_y110", "grid2dff", 162, 11),
          ("gr_g13_x105_y70", "grid2dgr", 329, 13), ("gr_g0_x70_y110", "grid2dgr", 162, 0)]
_G = {}

def init(scene):
    import sr2_hom_replay as RP
    if scene == "stem":
        orig = RP.ref_scene
        def rs(shape, d):
            pads, pt = orig(shape, d)
            return (STEM, pt) if shape == "sphere" else (pads, pt)
        RP.ref_scene = rs
    _G["RP"] = RP

def run(job):
    try:
        return _run(job)
    except Exception as e:                                  # a numerical failure of one run (SVD at mu 0.45)
        return {"grasp": job[0], "seed": job[4], "params": job[5], "error": repr(e)[:120]}

def _run(job):
    name, src, call, grasp, seed, params = job
    RP = _G["RP"]
    import hom_landscape as L
    from sr2hand.control import hom_bench as HB
    import mujoco
    import hand_object_scale_sim as S
    import hom_tasks as T
    setup, mxy, layout = RP.toy_setup("twist", 40.0, call, grasp, src)
    p_pads, p_pt = RP.ref_scene("sphere", 40.0)
    m_pt = mujoco.MjModel.from_xml_path(str(p_pt))
    RP.move_mounts(m_pt, mxy, RP.mount_base(m_pt))
    q_open, _ = T.open_pose(setup, m_pt)
    S.set_palm(m_pt, setup["palm"])
    bench = HB.HomBench("twist", 40.0, setup, m_pt, q_open, dt=HB.DT_BENCH, p_pads=p_pads)
    bench.T = RP._Tasks(lambda path, st: RP.SR2Plant(path, st, mxy, params, 0.0))
    w0 = time.perf_counter()
    stream, res = HB.simulate(bench, seed)
    from rev_sweep import feats_arrays
    t = np.array([float(s["t"]) for s in stream])
    P = np.array([s["obj_pos_mm"] for s in stream], float); Q = np.array([s["obj_quat"] for s in stream], float)
    f = feats_arrays(t, P, Q)
    from scipy.spatial.transform import Rotation as Rot
    ok = np.isfinite(P).all(1)
    R = Rot.from_quat(Q[ok][:, [1, 2, 3, 0]]); rv = (R * R[0].inv()).as_rotvec()
    f["tilt_max"] = float(np.degrees(np.linalg.norm(rv[:, :2], axis=1)).max())
    task = [x for x in stream if float(x["t"]) >= 2.0]
    f["pad_frac"] = {g: round(float(np.mean([float(((x.get("contacts") or {}).get(g) or [None, 0.0])[1]) >= 0.05
                                             for x in task])), 2) for g in ("thumb", "index", "middle")}
    err, ref = res.get("track_err_rms_deg"), res.get("track_ref_rms_deg")
    track = None if not err or not ref else max(0.0, 1.0 - err / ref)
    return {"grasp": name, "seed": seed, "params": params, "J": round(float(L.track_score("twist", res)), 4),
            "track": None if track is None else round(track, 3),
            "held": bool(res.get("held")), "err": res.get("track_err_rms_deg"), "ref": res.get("track_ref_rms_deg"),
            "contact_all": res.get("contact_all_frac"), **{k: (round(v, 2) if isinstance(v, float) else v) for k, v in f.items()},
            "wall_s": round(time.perf_counter() - w0, 1)}

def pads(rs):
    return "/".join("%.2f" % np.mean([r["pad_frac"][g] for r in rs]) for g in ("thumb", "index", "middle"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--scene", default="default", choices=["default", "stem"])
    ap.add_argument("--set", nargs="*", default=[]); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    params = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in a.set}
    if a.scene == "stem":
        import rev_stem
        assert rev_stem.build() == STEM
    import multiprocessing as mp
    jobs = [(n, s, c, g, seed, params) for (n, s, c, g) in GRASPS for seed in a.seeds]
    with mp.get_context("fork").Pool(a.workers, initializer=init, initargs=(a.scene,)) as pool:
        rows = pool.map(run, jobs)
    Path(a.out).write_text(json.dumps(rows, indent=1))
    print(f"scene {a.scene}, plant {params or 'default'}")
    for n, *_ in GRASPS:
        bad = [r for r in rows if r["grasp"] == n and "error" in r]
        rs = [r for r in rows if r["grasp"] == n and "error" not in r]
        if bad:
            print(f"   {n:20s} {len(bad)} run(s) failed: {bad[0]['error']}")
        if not rs:
            continue
        J = [r["J"] for r in rs]
        print(f"   {n:20s} J {np.mean(J):.3f} ({', '.join(f'{x:.3f}' for x in J)}) track {np.mean([r['track'] or 0 for r in rs]):.2f} "
              f"held {sum(r['held'] for r in rs)}/{len(rs)} | rise {np.mean([r['rise_max'] for r in rs]):5.1f} "
              f"twist_end {np.mean([r['twist_end'] for r in rs]):6.1f} tilt end/max {np.mean([r['tilt_end'] for r in rs]):5.1f}/"
              f"{np.mean([r['tilt_max'] for r in rs]):5.1f} | contact_all {np.mean([r['contact_all'] or 0 for r in rs]):.2f} "
              f"pads>0.05N th/in/mi {pads(rs)} "
              f"| {np.mean([r['wall_s'] for r in rs]):.0f} s", flush=True)
