"""Compile the current hand scenes and record what other code relies on: joints and their qpos addresses, actuators,
keyframes, bodies, palm_pose's geoms, masses and contact excludes. Writes probe_scenes.json beside this file."""
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path("/home/humanoid/Programs/hand")
OUT = ROOT / "logs/20261104-cad_v3/current"
SCENES = {
    "real_v1_hand": ROOT / "assets/mjcf/real_v1/real_hand.xml",
    "real_v1_scene": ROOT / "assets/mjcf/real_v1/scenes/scene_screwdriver_medium.xml",
    "cad_v2_scene": ROOT / "assets/mjcf/experimental/20261030-cad_v2/scene_screwdriver_medium.xml",
}
JT = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
rep = {}
for key, path in SCENES.items():
    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m)
    r = {"path": str(path.relative_to(ROOT)), "nq": m.nq, "nv": m.nv, "nu": m.nu, "nbody": m.nbody, "ngeom": m.ngeom,
         "nsite": m.nsite, "nkey": m.nkey, "nexclude": m.nexclude}
    r["joints"] = [[m.joint(j).name, JT[int(m.jnt_type[j])], int(m.jnt_qposadr[j]), m.body(m.jnt_bodyid[j]).name,
                    np.round(m.jnt_axis[j], 3).tolist(), np.round(m.jnt_range[j], 6).tolist(), bool(m.jnt_limited[j])]
                   for j in range(m.njnt)]
    r["actuators"] = [[m.actuator(a).name, m.joint(m.actuator_trnid[a, 0]).name] for a in range(m.nu)]
    r["keys"] = [m.key(k).name for k in range(m.nkey)]
    r["bodies"] = [[m.body(b).name, m.body(m.body_parentid[b]).name, np.round(m.body_pos[b] * 1000, 3).tolist(),
                    round(float(m.body_mass[b]), 6)] for b in range(m.nbody)]
    r["sites"] = [m.site(s).name for s in range(m.nsite)]
    try:
        pb = m.body("palm_pose").id
        r["palm_geoms"] = [[m.geom(g).name, int(m.geom_type[g]), np.round(m.geom_pos[g] * 1000, 3).tolist(),
                            np.round(m.geom_size[g] * 1000, 3).tolist(), int(m.geom_contype[g]), int(m.geom_conaffinity[g])]
                           for g in range(m.ngeom) if m.geom_bodyid[g] == pb]
        r["palm_mass_kg"] = round(float(m.body_mass[pb]), 6)
        r["palm_subtree_mass_kg"] = round(float(m.body_subtreemass[pb]), 6)
        r["palm_gravcomp"] = float(m.body_gravcomp[pb])
    except KeyError:
        pass
    r["excludes"] = [[m.body(int(sig >> 16)).name, m.body(int(sig & 0xFFFF)).name] for sig in m.exclude_signature]
    # geom counts per body, collidable only
    cnt = {}
    for g in range(m.ngeom):
        if m.geom_contype[g] or m.geom_conaffinity[g]:
            b = m.body(m.geom_bodyid[g]).name
            cnt[b] = cnt.get(b, 0) + 1
    r["collidable_geoms_per_body"] = cnt
    # at the 'open' key: palm and mount positions in the world, tip positions relative to the mount
    if m.nkey:
        k = [m.key(i).name for i in range(m.nkey)].index("open") if "open" in r["keys"] else 0
        mujoco.mj_resetDataKeyframe(m, d, k)
        mujoco.mj_forward(m, d)
        try:
            r["open_key"] = {"palm_world_mm": np.round(d.body("palm_pose").xpos * 1000, 3).tolist(),
                             "qpos_len": int(m.key_qpos.shape[1]),
                             "mount_world_mm": {f: np.round(d.body(f"{f}_mount").xpos * 1000, 3).tolist()
                                                for f in ("thumb", "index", "middle")},
                             "tip_world_mm": {f: np.round(d.body(f"{f}_tip").xpos * 1000, 3).tolist()
                                              for f in ("thumb", "index", "middle")},
                             "ncon": int(d.ncon)}
        except KeyError:
            pass
    rep[key] = r
    print(key, {k: r[k] for k in ("nq", "nv", "nu", "nbody", "ngeom", "nkey", "nexclude")}, flush=True)
(OUT / "probe_scenes.json").write_text(json.dumps(rep, indent=1))
print("wrote", OUT / "probe_scenes.json")
