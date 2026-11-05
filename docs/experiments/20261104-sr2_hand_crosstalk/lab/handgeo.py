"""FK and finger-to-finger clearance of the SR2 hand in the cad_v2 scene (collision spheres of every part, the v4b tip's
DIP link and cassette), for mounts in each finger's local frame (mm) and joints in sim degrees (yaw, mcp, pip)."""
import itertools
from pathlib import Path

import mujoco
import numpy as np

SCENE = str(Path(__file__).resolve().parents[4] / "assets/mjcf/experimental/20261030-cad_v2/scene_screwdriver_medium.xml")
F = ("thumb", "index", "middle")
J = ("yaw", "mcp", "pip")
HOME_LOCAL = {"thumb": (-30.0, -55.0), "index": (30.0, 30.0), "middle": (30.0, -30.0)}
ORIGIN = {"thumb": (-50.0, 0.0), "index": (50.0, 55.0), "middle": (50.0, -55.0)}


class Hand:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(SCENE)
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.geoms = {f: [] for f in F}
        for g in range(m.ngeom):
            if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_SPHERE or not m.geom_contype[g]:
                continue
            b = m.geom_bodyid[g]
            while b > 0:
                n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
                hit = next((f for f in F if n.startswith(f)), None)
                if hit:
                    self.geoms[hit].append(g)
                    break
                b = m.body_parentid[b]
        self.geoms = {f: np.array(v) for f, v in self.geoms.items()}
        self.r = {f: m.geom_size[v, 0] * 1000 for f, v in self.geoms.items()}
        self.adr = {f"{f}_{a}": m.jnt_qposadr[m.joint(f"{f}_{a}").id] for f in F for a in ("x", "y") + J}
        # the screwdriver out of the way
        self.free = [m.jnt_qposadr[j] for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]

    def set(self, local, joints):
        d = self.d
        d.qpos[:] = self.m.qpos0
        for a in self.free:
            d.qpos[a:a + 3] = (0.0, 0.0, -5.0)
        for f in F:
            d.qpos[self.adr[f"{f}_x"]] = local[f][0] / 1000
            d.qpos[self.adr[f"{f}_y"]] = local[f][1] / 1000
            for j in J:
                d.qpos[self.adr[f"{f}_{j}"]] = np.radians(joints.get(f, {}).get(j, 0.0))
        mujoco.mj_kinematics(self.m, self.d)

    def centres(self, f):
        return self.d.geom_xpos[self.geoms[f]] * 1000

    def clearance(self, a, b):
        ca, cb = self.centres(a), self.centres(b)
        dist = np.linalg.norm(ca[:, None, :] - cb[None, :, :], axis=2) - self.r[a][:, None] - self.r[b][None, :]
        return float(dist.min())

    def clearances(self):
        return {f"{a}-{b}": self.clearance(a, b) for a, b in itertools.combinations(F, 2)}

    def tip(self, f):
        return self.d.body(f"{f}_tip").xpos * 1000

    def tip_frame(self, f):
        return self.d.body(f"{f}_tip").xmat.reshape(3, 3)
