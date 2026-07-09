"""Object-relative fingertip imitation (user idea #3, 2026-07-06).

The morphology sweep showed per-design reorient quality is dominated by training-seed luck: from
scratch, whether PPO discovers the "roll the cylinder to vertical" motion is a lottery (peak cos
0.0-0.9 across seeds). One variance-reducer: give every design the SAME good reorientation to
imitate, so the policy adapts a known skill instead of re-discovering it.

The transferable representation is the **object-relative fingertip trajectory**: the 3 fingertip
positions expressed in the SCREWDRIVER's frame, over time. Unlike joint angles (morphology-specific),
the object-relative fingertip motion of a good reorientation is (approximately) morphology-invariant
-- a different hand can reproduce the same object-frame fingertip path with different joint angles
(exactly the logic of the world-space keyframe retarget). So we record it once from the blessed
reorienter (a10->b33 on m05) and imitate it on any design, with a curriculum that fades the
imitation weight as the task reward takes over.

This module is self-contained (a reference loader + a torch reward term); wire the reward into
`env_cfg` behind a flag. Records via `scripts/rl_record_reorient_fingertip_traj.py`.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch

from morphohand.rl.math import quat_rotate_inv as _quat_rotate_inv

TIP_BODIES = ("thumb_tip", "index_tip", "middle_tip")


def fingertips_in_object_frame(env, object_name: str = "cube",
                               tip_bodies: tuple = TIP_BODIES) -> torch.Tensor:
    """Current 3 fingertip positions expressed in the object frame -> (B, 3, 3)."""
    robot = env.scene["robot"]
    obj = env.scene[object_name]
    obj_pose = obj.data.root_link_pose_w                 # (B, 7) pos+quat(wxyz)
    obj_pos = obj_pose[:, :3]
    obj_quat = obj_pose[:, 3:7]
    ids = [robot.body_names.index(b) for b in tip_bodies]
    tips_w = robot.data.body_link_pose_w[:, ids, :3]     # (B, 3, 3) world
    rel = tips_w - obj_pos.unsqueeze(1)                  # (B, 3, 3) world, object-centered
    B = rel.shape[0]
    q = obj_quat.unsqueeze(1).expand(B, 3, 4).reshape(B * 3, 4)
    out = _quat_rotate_inv(q, rel.reshape(B * 3, 3)).reshape(B, 3, 3)
    return out


class FingertipObjReference:
    """A recorded object-relative fingertip trajectory (T, 3, 3), sample-able at continuous time."""

    def __init__(self, traj: np.ndarray, dt: float, t0: float = 0.0):
        self.traj = torch.as_tensor(traj, dtype=torch.float32)   # (T, 3, 3)
        self.dt = float(dt)
        self.t0 = float(t0)                                       # ref time of trajectory[0]
        self._dev = False

    @classmethod
    def load(cls, path: str | Path) -> "FingertipObjReference":
        d = np.load(path)
        return cls(d["fingertip_obj"], float(d["dt"]), float(d.get("t0", 0.0)))

    def to(self, device):
        self.traj = self.traj.to(device); self._dev = True
        return self

    def sample(self, times: torch.Tensor) -> torch.Tensor:
        """Linear-interp the reference at per-env `times` (seconds, (B,)) -> (B, 3, 3). Clamps ends."""
        if not self._dev:
            self.traj = self.traj.to(times.device); self._dev = True
        T = self.traj.shape[0]
        x = ((times - self.t0) / self.dt).clamp(0.0, T - 1.0)
        lo = x.floor().long().clamp(0, T - 1)
        hi = (lo + 1).clamp(0, T - 1)
        a = (x - lo.float()).view(-1, 1, 1)
        return (1.0 - a) * self.traj[lo] + a * self.traj[hi]


_REF_CACHE: dict[str, FingertipObjReference] = {}


def track_fingertip_obj(env, ref_path: str, alpha: float = 300.0,
                        object_name: str = "cube", reorient_start_step: int = 0,
                        tip_bodies: tuple = TIP_BODIES) -> torch.Tensor:
    """Imitation reward: exp(-alpha * mean-sq object-frame fingertip error) vs the recorded
    reference at each env's current (post-reorient-onset) time. Weight it POSITIVE; anneal via a
    curriculum. Gated to fire only after `reorient_start_step` (before that the object is being
    lifted and the reference -- a reorient trajectory -- does not apply)."""
    ref = _REF_CACHE.get(ref_path)
    if ref is None:
        ref = _REF_CACHE[ref_path] = FingertipObjReference.load(ref_path).to(env.device)
    step = env.episode_length_buf                                # (B,) int
    dt = float(getattr(env, "step_dt", getattr(env, "physics_dt", 0.02)))
    t = (step - reorient_start_step).clamp(min=0).to(torch.float32) * dt
    cur = fingertips_in_object_frame(env, object_name, tip_bodies)   # (B, 3, 3)
    tgt = ref.sample(t)                                              # (B, 3, 3)
    err = (cur - tgt).pow(2).mean(dim=(1, 2))                        # (B,)
    r = torch.exp(-alpha * err)
    return torch.where(step >= reorient_start_step, r, torch.zeros_like(r))
