#!/usr/bin/env python3
"""tool0 of the draft MJCF (ur5_cb3.xml) against the UR5 (CB3) DH model and the URDF, at 5 random joint configurations
(seed 0, each joint uniform in [-pi, pi], as scripts/ur5/ur5_cb3_cell.py check); the keyframes' flange poses in B; and the
controller's reported flange pose (zero TCP) against nominal DH at the 27 poses the lab recorded
(docs/experiments/20261022-ur5_tip_calibration/lab/poses.jsonl).

    .venv/bin/python logs/20261104-cad_v3/ur5/dh_check.py [--no-meshes]

--no-meshes compiles the model with its visual meshes stripped (the kinematic tree, inertials and collision primitives
only), so the check loads no mesh.
"""
import json
import math
import re
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path("/home/humanoid/Programs/hand")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/sr2hand/driver/reskin5x/host"))
sys.path.insert(0, str(ROOT / "scripts"))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path
from sr2hand.geometry import rotations as R  # noqa: E402
import ur5_tip_calibration as C  # noqa: E402
import ur5_tip_calibration_sim as SIM  # noqa: E402
import ur5_cb3_cell as CELL  # noqa: E402

ARM = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")


def load(no_meshes):
    xml = (HERE / "ur5_cb3.xml").read_text()
    if no_meshes:
        xml = re.sub(r"\s*<mesh [^>]*/>", "", xml)
        xml = re.sub(r'\s*<geom class="visual"[^>]*/>', "", xml)
        return mujoco.MjModel.from_xml_string(xml)
    return mujoco.MjModel.from_xml_path(str(HERE / "ur5_cb3.xml"))


def site_T(m, d, name):
    i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, name)
    return R.tf(d.site_xmat[i].reshape(3, 3).copy(), d.site_xpos[i] * 1e3)


def fk_mj(m, d, q):
    for j, v in zip(ARM, q):
        d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]] = v
    mujoco.mj_kinematics(m, d)
    return site_T(m, d, "tool0")


def main():
    no_meshes = "--no-meshes" in sys.argv
    m = load(no_meshes)
    d = mujoco.MjData(m)
    out = dict(model=str(HERE / "ur5_cb3.xml"), meshes_stripped=no_meshes, nq=m.nq, nu=m.nu, nbody=m.nbody,
               mass_kg=round(float(m.body_mass.sum()), 4))
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(5):
        q = rng.uniform(-math.pi, math.pi, 6)
        Tm = fk_mj(m, d, q)
        Td = SIM.fk(q)
        Tu = CELL.link_frames(q)["tool0"]
        e_md, e_ud = C.pose_error(Tm, Td), C.pose_error(Tu, Td)
        rows.append(dict(q_deg=np.round(np.degrees(q), 2).tolist(), mjcf_vs_dh_mm_deg=[float(f"{e_md[0]:.3g}"), float(f"{e_md[1]:.3g}")],
                         urdf_vs_dh_mm_deg=[float(f"{e_ud[0]:.3g}"), float(f"{e_ud[1]:.3g}")],
                         tool0_B_mm=np.round(Tm[:3, 3], 3).tolist()))
        print(rows[-1])
    out["dh_check"] = rows
    out["dh_check_max_mm_deg"] = [max(r["mjcf_vs_dh_mm_deg"][0] for r in rows), max(r["mjcf_vs_dh_mm_deg"][1] for r in rows)]
    # keyframes
    keys = {}
    for k in range(m.nkey):
        q = m.key_qpos[k][:6]
        T = fk_mj(m, d, q)
        keys[m.key(k).name] = dict(q_deg=np.round(np.degrees(q), 2).tolist(), flange_B_mm=np.round(T[:3, 3], 1).tolist(),
                                   tool_axis_B=np.round(T[:3, 2], 3).tolist(), x_axis_B=np.round(T[:3, 0], 3).tolist())
        print(m.key(k).name, keys[m.key(k).name])
    out["keyframes"] = keys
    # the controller's reported flange (zero TCP) against nominal DH
    lab = [json.loads(l) for l in open(ROOT / "docs/experiments/20261022-ur5_tip_calibration/lab/poses.jsonl") if l.strip()]
    errs = []
    for r in lab:
        p = np.array(r["tcp_pose_m_rad"])
        Tc = R.tf(R.exp_so3(p[3:]), p[:3] * 1e3)
        Tm = fk_mj(m, d, np.array(r["q_rad"]))
        e = C.pose_error(Tc, Tm)
        errs.append(dict(label=r["label"], time=r["time"], mm=round(e[0], 3), deg=round(e[1], 3),
                         dp_B_mm=np.round(Tc[:3, 3] - Tm[:3, 3], 2).tolist()))
    E = np.array([[e["mm"], e["deg"]] for e in errs])
    out["controller_vs_mjcf"] = dict(n=len(errs), mm_median=round(float(np.median(E[:, 0])), 2),
                                     mm_range=[round(float(E[:, 0].min()), 2), round(float(E[:, 0].max()), 2)],
                                     deg_median=round(float(np.median(E[:, 1])), 3),
                                     deg_range=[round(float(E[:, 1].min()), 3), round(float(E[:, 1].max()), 3)],
                                     mean_dp_B_mm=np.round(np.mean([e["dp_B_mm"] for e in errs], axis=0), 2).tolist(),
                                     rows=errs)
    print("controller vs MJCF:", {k: v for k, v in out["controller_vs_mjcf"].items() if k != "rows"})
    (HERE / "dh_check.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
