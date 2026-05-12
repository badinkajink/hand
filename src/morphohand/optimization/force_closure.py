"""Force-closure quality scoring as a continuous replacement for contact count.

The standard Phase-1 objective rewards each tip-object collision pair with a
constant weight (``objective_weight_contact``), which is a very weak proxy
for whether the grasp is actually stable. This module adds a continuous,
geometry-aware quality score derived from the resolved contact set:

  - **normal balance** — penalize ``|| sum_i n_i ||`` (opposed normals cancel);
  - **wrench spread** — reward ``sqrt(det(W W^T))`` (Gram volume of the wrench
    matrix); independent wrenches => non-degenerate grasp;
  - **Q1 distance** — distance from the origin to the convex hull of the
    (optionally friction-cone-discretized) contact wrenches. This is the
    classical Ferrari-Canny / DexGraspNet DFC quality measure (Liu et al.,
    2021, "Synthesizing Diverse and Physically Stable Grasps with Arbitrary
    Hand Structures using Differentiable Force Closure Estimator").

The implementation is pure NumPy and consumes ``mujoco.MjData`` resolved
contacts, so it composes with any backend that maintains MuJoCo's contact
buffer. It is NOT differentiable in the JAX sense — it serves as a more
faithful CEM objective term, not a gradient signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # pragma: no cover - mujoco is a runtime dep of the evaluator
    import mujoco  # pyright: ignore[reportMissingImports]
except ImportError:  # pragma: no cover
    mujoco = None  # type: ignore[assignment]


@dataclass
class ContactWrench:
    """A single contact, normalized to world frame.

    `normal` points OUT of the object (toward the fingertip).
    `point` is the world position of the contact.
    """

    point: np.ndarray
    normal: np.ndarray
    finger_idx: int

    def __post_init__(self) -> None:
        self.point = np.asarray(self.point, dtype=np.float64).reshape(3)
        n = np.asarray(self.normal, dtype=np.float64).reshape(3)
        nn = float(np.linalg.norm(n))
        if nn < 1e-12:
            raise ValueError("ContactWrench: zero-length normal")
        self.normal = n / nn


def extract_finger_contacts(
    data: "mujoco.MjData",
    model: "mujoco.MjModel",
    tip_body_ids: list[int],
    object_body_id: int,
) -> list[ContactWrench]:
    """Pull fingertip-object contacts out of the active MuJoCo contact set.

    The contact normal in MuJoCo's frame points along ``contact.frame[0:3]``.
    We orient it so it points from the object toward the fingertip.
    """
    if mujoco is None:
        raise ImportError("mujoco is required for extract_finger_contacts")

    tip_index = {bid: i for i, bid in enumerate(tip_body_ids)}
    ngeom = int(model.ngeom)
    out: list[ContactWrench] = []
    for i in range(int(data.ncon)):
        contact = data.contact[i]
        g1 = int(contact.geom1)
        g2 = int(contact.geom2)
        if g1 < 0 or g2 < 0 or g1 >= ngeom or g2 >= ngeom:
            continue
        b1 = int(model.geom_bodyid[g1])
        b2 = int(model.geom_bodyid[g2])

        if b1 == object_body_id and b2 in tip_index:
            tip_body = b2
            sign = +1.0
        elif b2 == object_body_id and b1 in tip_index:
            tip_body = b1
            sign = -1.0
        else:
            continue

        point = np.asarray(contact.pos, dtype=np.float64).reshape(3)
        normal = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)[0]
        normal = sign * normal
        try:
            out.append(ContactWrench(point=point, normal=normal, finger_idx=tip_index[tip_body]))
        except ValueError:
            continue
    return out


def _build_friction_cone_wrenches(
    contacts: list[ContactWrench],
    object_com: np.ndarray,
    mu: float = 0.5,
    n_edges: int = 4,
) -> np.ndarray:
    """Stack 6D wrenches for each friction-cone edge of every contact.

    Returns array of shape (6, n_contacts * n_edges).
    """
    if not contacts:
        return np.zeros((6, 0), dtype=np.float64)

    cols: list[np.ndarray] = []
    for c in contacts:
        n = c.normal
        # Build two tangent directions perpendicular to n
        helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        t1 = helper - np.dot(helper, n) * n
        t1 /= np.linalg.norm(t1) + 1e-12
        t2 = np.cross(n, t1)

        r = c.point - object_com
        for k in range(n_edges):
            theta = 2.0 * np.pi * k / n_edges
            f = n + mu * (np.cos(theta) * t1 + np.sin(theta) * t2)
            f /= np.linalg.norm(f) + 1e-12
            tau = np.cross(r, f)
            cols.append(np.concatenate([f, tau]))
    return np.stack(cols, axis=1)


def q1_distance_convex_hull(W: np.ndarray) -> float:
    """Min ||W c||_2 s.t. sum c = 1, c >= 0.

    Iterative active-set / Frank-Wolfe on the simplex; converges quickly for
    the small W we deal with (3 contacts * 4 cone edges = 12 columns).
    """
    if W.size == 0:
        return float("inf")

    n = W.shape[1]
    if n == 1:
        return float(np.linalg.norm(W[:, 0]))

    # Initialize c at uniform; do Frank-Wolfe with simplex constraint.
    c = np.ones(n) / n
    for _ in range(64):
        residual = W @ c  # 6-vector
        grad = W.T @ residual  # n-vector
        s = np.zeros(n)
        s[np.argmin(grad)] = 1.0  # extreme point of simplex along -grad
        d = s - c
        # Line search: minimize || W (c + alpha d) ||^2 = || r + alpha W d ||^2
        Wd = W @ d
        denom = float(Wd @ Wd)
        if denom < 1e-18:
            break
        alpha = float(-residual @ Wd / denom)
        alpha = max(0.0, min(1.0, alpha))
        c = c + alpha * d
        if alpha < 1e-9:
            break
    return float(np.linalg.norm(W @ c))


def wrench_spread(W: np.ndarray) -> float:
    """Geometric mean of singular values: sqrt det(W^T W) ^ (1/min(m,n))."""
    if W.size == 0:
        return 0.0
    s = np.linalg.svd(W, compute_uv=False)
    if s.size == 0 or s[0] < 1e-12:
        return 0.0
    return float(np.exp(np.mean(np.log(np.maximum(s, 1e-12)))))


def normal_balance(contacts: list[ContactWrench]) -> float:
    """Higher (less negative) is worse: penalty proxy.

    Returns ``|| sum normals ||`` directly so the caller can use it as a
    minimize-this signal.
    """
    if not contacts:
        return 0.0
    s = np.sum(np.stack([c.normal for c in contacts], axis=0), axis=0)
    return float(np.linalg.norm(s))


@dataclass
class ForceClosureMetrics:
    n_contacts: int
    fingers_engaged: int
    normal_balance: float  # smaller is better (||sum n||)
    wrench_spread: float  # larger is better (geom mean of sv)
    q1_distance: float    # smaller is better (Ferrari-Canny / DFC residual)
    score: float          # combined: spread - lam1*balance - lam2*q1

    def to_dict(self) -> dict[str, float]:
        return {
            "fc_n_contacts": float(self.n_contacts),
            "fc_fingers_engaged": float(self.fingers_engaged),
            "fc_normal_balance": self.normal_balance,
            "fc_wrench_spread": self.wrench_spread,
            "fc_q1_distance": self.q1_distance,
            "fc_score": self.score,
        }


def force_closure_metrics(
    contacts: list[ContactWrench],
    object_com: np.ndarray,
    *,
    mu: float = 0.5,
    n_edges: int = 4,
    weight_balance: float = 0.5,
    weight_q1: float = 1.0,
) -> ForceClosureMetrics:
    """Compute all force-closure quality proxies in one pass.

    The combined `score` is suitable for adding to a maximization objective:

        score = wrench_spread - weight_balance * normal_balance - weight_q1 * q1_distance

    All three pieces are reported separately so the caller can weight or
    log them independently.
    """
    if not contacts:
        return ForceClosureMetrics(0, 0, 0.0, 0.0, float("inf"), -float("inf"))

    fingers_engaged = len({c.finger_idx for c in contacts})
    W_cone = _build_friction_cone_wrenches(contacts, object_com, mu=mu, n_edges=n_edges)
    # Centroidal wrench matrix without friction discretization (for spread).
    W_centroidal = np.stack(
        [np.concatenate([c.normal, np.cross(c.point - object_com, c.normal)]) for c in contacts],
        axis=1,
    )

    balance = normal_balance(contacts)
    spread = wrench_spread(W_centroidal)
    q1 = q1_distance_convex_hull(W_cone)

    combined = spread - weight_balance * balance - weight_q1 * q1
    return ForceClosureMetrics(
        n_contacts=len(contacts),
        fingers_engaged=fingers_engaged,
        normal_balance=balance,
        wrench_spread=spread,
        q1_distance=q1,
        score=combined,
    )
