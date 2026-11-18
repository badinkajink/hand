"""The lat0 press grasp's closure at the onset, in the calibrated simulation (hand-4a, 2026-11-17 19:0x; owner via the lab,
18:5x: "the grasp, for one reason or another, on the press task, is not perfectly aligned / does not have force closure at
the very onset, produces some skew"; lab: "check the press grasp's closure at the onset in simulation (the grasp's margin,
and whether the pads' contacts give force closure about the cylinder's axis)").

At t = 1.30, 1.58 (end of the squeeze, before the lift at 1.6 s), 2.0 and 2.5 s, from every pad-cylinder contact
(mj_contactForce): per finger the normal force, the centre of pressure and the force-weighted outward normal; the pinch's
antipodality (angle between the thumb's and the index's normals, 180 deg antipodal) and the angle of the pinch line to each
normal (inside the friction cone when below atan(mu)); the torque about the cylinder's own axis that the contacts' friction
can resist (sum of mu N_j |P_t (a x r_j)|) against the torque the contact forces apply about it; and the Ferrari-Canny
margin epsilon of the contact set (unit normal forces, 8-edge cones, torques over R = 12.5 mm; 0 = no force closure),
with all three fingers and with the thumb-index pinch alone. Also the cylinder's turn about its axis since 0.3 s.

    PRESS_SCENE=cyl025_stem_block_pads_imp100_ns30.xml HOS_PLANT=kp0.55_kv0_fr0.18_fl0_dp0.011 SCALE_HAND=cad_v2 \\
        OMP_NUM_THREADS=1 .venv/bin/python press_closure.py OUT.json
"""
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import press_replan as PR  # noqa: E402

TIMES = (0.30, 1.30, 1.58, 2.00, 2.50)
R_CYL = 0.0125
PAD = re.compile(r"(thumb|index|middle)_(pad\d+|tipgeom)")


def epsilon(contacts, c, mu, n_edges=8):
    """Ferrari-Canny epsilon: radius of the largest origin-centred ball in the hull of the cone-edge wrenches (torque / R)."""
    from scipy.spatial import ConvexHull
    from sr2hand.geometry.rotations import frame_from_axis
    cols = []
    for p, n in contacts:
        F = frame_from_axis(n, np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0]))
        t1, t2 = F[:, 0], F[:, 1]
        for k in range(n_edges):
            th = 2 * math.pi * k / n_edges
            f = -n + mu * (math.cos(th) * t1 + math.sin(th) * t2)       # force on the object: into it, plus friction
            cols.append(np.concatenate([f, np.cross(p - c, f) / R_CYL]))
    W = np.array(cols)
    if len(W) < 8:
        return 0.0
    try:
        h = ConvexHull(W, qhull_options="QJ")
    except Exception:                                    # noqa: BLE001  degenerate (planar) wrench set
        return 0.0
    off = h.equations[:, -1]
    return float(max(0.0, -off.max())) if np.all(off < 0) else 0.0


def snapshot(plant, mu_list=(1.0, 0.5, 0.4)):
    import mujoco
    m, d = plant.m, plant.d
    tb = plant.tb
    cyl = [g for g in range(m.ngeom) if m.geom_bodyid[g] == tb and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER]
    g0 = cyl[0] if cyl else int(np.flatnonzero(m.geom_bodyid == tb)[0])
    c = d.geom_xpos[g0].copy()
    a = d.geom_xmat[g0].reshape(3, 3)[:, 2].copy()                   # the cylinder's axis
    per = {f: {"N": 0.0, "p": np.zeros(3), "n": np.zeros(3), "k": 0, "pts": []} for f in ("thumb", "index", "middle")}
    tau_axis, F_net, tau_max = 0.0, np.zeros(3), {mu: 0.0 for mu in mu_list}
    fr = np.zeros(6)
    for i in range(d.ncon):
        con = d.contact[i]
        n1, n2 = m.geom(con.geom1).name, m.geom(con.geom2).name
        b1, b2 = m.geom_bodyid[con.geom1], m.geom_bodyid[con.geom2]
        if b1 == tb and PAD.match(n2 or ""):
            fing, sgn = PAD.match(n2).group(1), -1.0           # frame[0] points geom1 -> geom2: object -> pad
        elif b2 == tb and PAD.match(n1 or ""):
            fing, sgn = PAD.match(n1).group(1), 1.0
        else:
            continue
        mujoco.mj_contactForce(m, d, i, fr)
        fr_w = con.frame.reshape(3, 3).T @ fr[:3]                     # force on geom2 from geom1, world frame
        f_obj = fr_w if b2 == tb else -fr_w                           # on the object
        n_out = con.frame.reshape(3, 3)[0] if b1 == tb else -con.frame.reshape(3, 3)[0]   # object -> pad
        N = max(0.0, float(fr[0]))
        p = con.pos.copy()
        r = p - c
        q = per[fing]
        q["N"] += N
        q["p"] += N * p
        q["n"] += N * n_out
        q["k"] += 1
        if N > 1e-4:
            q["pts"].append((p, n_out, N))
        tau_axis += float(np.cross(r, f_obj) @ a)
        F_net += f_obj
        v = np.cross(a, r)
        vt = v - (v @ n_out) * n_out
        for mu in mu_list:
            tau_max[mu] += mu * N * float(np.linalg.norm(vt))
    out = {"axis": a.round(4).tolist(), "centre_mm": (1000 * c).round(2).tolist(),
           "tau_axis_applied_mNm": round(1000 * tau_axis, 3),
           "tau_axis_resistible_mNm": {str(mu): round(1000 * v, 2) for mu, v in tau_max.items()},
           "F_net_contacts_N": F_net.round(3).tolist(), "fingers": {}}
    for f, q in per.items():
        if q["N"] > 1e-6:
            nn = q["n"] / np.linalg.norm(q["n"])
            out["fingers"][f] = {"N": round(q["N"], 3), "contacts": q["k"], "cop_mm": (1000 * q["p"] / q["N"]).round(2).tolist(),
                                 "normal": nn.round(4).tolist(), "_cop": q["p"] / q["N"], "_n": nn}
        else:
            out["fingers"][f] = {"N": 0.0, "contacts": q["k"]}
    F = out["fingers"]
    if all("normal" in F[f] for f in ("thumb", "index")):
        nt, ni = F["thumb"]["_n"], F["index"]["_n"]
        line = F["index"]["_cop"] - F["thumb"]["_cop"]
        line /= np.linalg.norm(line)
        out["pinch"] = {"normals_angle_deg": round(math.degrees(math.acos(np.clip(nt @ ni, -1, 1))), 2),
                        "line_to_thumb_normal_deg": round(math.degrees(math.acos(np.clip(-line @ nt, -1, 1))), 2),
                        "line_to_index_normal_deg": round(math.degrees(math.acos(np.clip(line @ ni, -1, 1))), 2),
                        "line_to_axis_deg": round(math.degrees(math.acos(abs(np.clip(line @ a, -1, 1)))), 2)}
    for f in F.values():
        f.pop("_cop", None)
        f.pop("_n", None)
    top = lambda f, k=40: [(p, n) for p, n, N in sorted(per[f]["pts"], key=lambda x: -x[2])[:k]]  # noqa: E731
    out["epsilon"] = {}
    for mu in mu_list:
        out["epsilon"][str(mu)] = {"three": round(epsilon(top("thumb") + top("index") + top("middle"), c, mu), 4),
                                   "pinch": round(epsilon(top("thumb") + top("index"), c, mu), 4)}
    return out


def run(job):
    import mujoco
    bench, plant = PR._bench_plant(job["q_deg"], job["volts"], job["stops"])
    snaps, axis0 = {}, {}

    def hook(t, pl):
        for T in TIMES:
            if T not in snaps and t >= T - 1e-6:
                s = snapshot(pl)
                if not axis0:
                    axis0["a"] = np.array(s["axis"])
                tb = pl.tb
                q = pl.d.xquat[tb].copy()
                s["obj_quat"] = q.round(5).tolist()
                snaps[T] = s
    rows, err = PR.PP.simulate(bench, plant, job["seed"], station=False, contacts=False, hook=hook)
    # the cylinder's turn about its own axis: angle of the body's x axis about a, from the first snapshot
    from sr2hand.geometry.rotations import quat_to_mat
    ts = sorted(snaps)
    R0 = quat_to_mat(np.array(snaps[ts[0]]["obj_quat"]))
    for T in ts:
        R = quat_to_mat(np.array(snaps[T]["obj_quat"]))
        dR = R @ R0.T
        a = np.array(snaps[T]["axis"])
        from scipy.spatial.transform import Rotation
        rv = Rotation.from_matrix(dR).as_rotvec()
        snaps[T]["turn_about_axis_deg"] = round(math.degrees(float(rv @ a)), 2)
        snaps[T]["turn_total_deg"] = round(math.degrees(float(np.linalg.norm(rv))), 2)
    return {"cand": job["cand"], "volts": job["volts"], "seed": job["seed"], "error": err,
            "snapshots": {str(k): v for k, v in snaps.items()}}


def main():
    import multiprocessing as mp
    out = Path(sys.argv[1])
    P = json.loads((PR.OUT / "press_poses.json").read_text())
    P["plan@sim"] = P["plan"]
    jobs = [{"cand": c, "q_deg": P[c]["deg"], "volts": v, "seed": 0, "stops": c != "plan@sim"}
            for c in ("plan@sim", "plan", "y+0_p4_t4") for v in (5.0, 7.5)]
    with mp.get_context("fork").Pool(6) as pool:
        res = pool.map(run, jobs)
    out.write_text(json.dumps(res, indent=1) + "\n")
    for r in res:
        for T, s in r["snapshots"].items():
            F = s["fingers"]
            print(f"{r['cand']:10s} {r['volts']:3.1f} t {T:>4s} N t/i/m {F['thumb']['N']:.2f}/{F['index']['N']:.2f}/"
                  f"{F['middle']['N']:.2f} pinch {s.get('pinch')} | spin resist mu1 {s['tau_axis_resistible_mNm']['1.0']} "
                  f"applied {s['tau_axis_applied_mNm']} mN m | eps mu1 {s['epsilon']['1.0']} mu0.4 {s['epsilon']['0.4']} | "
                  f"turn about axis {s['turn_about_axis_deg']} total {s['turn_total_deg']}", flush=True)


if __name__ == "__main__":
    main()
