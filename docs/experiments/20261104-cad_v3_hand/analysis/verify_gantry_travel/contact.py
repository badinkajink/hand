"""First contact of a rigid group translated along a coordinate axis, from the STL meshes (independent of gantry/sweep.py).

Method. A polyhedron translated along d first touches another at a vertex-face or an edge-edge contact. Every vertex and
points every H mm along every edge of each solid are moved EPS mm into that solid (along the vertex or edge normal), so
that flush faces and sliding contacts do not register, and a line along d is cast from each point. A hit counts when it
enters the other solid: a moving point travelling +d meets a static face whose outward normal has n.d < 0, or a static
point meets a moving face with n.d > 0 (both directions are cast). The smallest distance per pair of solids is the first
contact for that pair; edge-edge contacts are found within about H/2 times the face slope.
"""
import numpy as np
import trimesh

H = 0.2     # edge sample spacing, mm
EPS = 0.01  # inward offset of the sample points, mm
CELL = 0.5  # bucket size of the transverse grid, mm


def load(path):
    m = trimesh.load(path, process=True)
    if m.volume < 0:
        m.invert()
    return m


def samples(m, h=H, eps=EPS):
    V = m.vertices
    pts = [V - eps * m.vertex_normals]
    fa, fae = m.face_adjacency, m.face_adjacency_edges
    fn = m.face_normals
    ne = fn[fa[:, 0]] + fn[fa[:, 1]]
    nn = np.linalg.norm(ne, axis=1)
    ok = nn > 1e-9
    ne[ok] /= nn[ok, None]
    a, b = V[fae[:, 0]], V[fae[:, 1]]
    L = np.linalg.norm(b - a, axis=1)
    k = np.maximum(1, np.ceil(L / h).astype(int))
    idx = np.repeat(np.arange(len(fae)), k)
    start = np.repeat(np.cumsum(k) - k, k)
    s = (np.arange(len(idx)) - start + 0.5) / k[idx]
    P = a[idx] + s[:, None] * (b - a)[idx] - eps * ne[idx]
    pts.append(P)
    return np.vstack(pts)


class Caster:
    """Lines along axis `ax` against one mesh's triangles, through a bucket grid on the two other axes."""

    def __init__(self, m, ax):
        self.ax = ax
        self.tr = (ax + 1) % 3, (ax + 2) % 3
        T = m.triangles
        n = m.face_normals
        keep = np.abs(n[:, ax]) > 1e-9          # faces parallel to the line are never entered
        self.T, self.n = T[keep], n[keep]
        i, j = self.tr
        U = self.T[:, :, i]
        W = self.T[:, :, j]
        self.lo = np.array([U.min(1), W.min(1)]).T
        self.hi = np.array([U.max(1), W.max(1)]).T
        self.bmin = self.lo.min(0) - 1.0
        c0 = np.floor((self.lo - self.bmin) / CELL).astype(int)
        c1 = np.floor((self.hi - self.bmin) / CELL).astype(int)
        self.ncol = int(c1[:, 1].max()) + 2
        cnt = (c1[:, 0] - c0[:, 0] + 1) * (c1[:, 1] - c0[:, 1] + 1)
        tri = np.repeat(np.arange(len(cnt)), cnt)
        off = np.arange(len(tri)) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        w = (c1[:, 1] - c0[:, 1] + 1)[tri]
        ci = c0[tri, 0] + off // w
        cj = c0[tri, 1] + off % w
        cell = ci * self.ncol + cj
        o = np.argsort(cell, kind='stable')
        self.cell, self.tri = cell[o], tri[o]

    def cast(self, P, chunk=20000):
        """For points P: (point index, hit coordinate along ax, sign of the face normal along ax) of every crossing."""
        i, j = self.tr
        out_p, out_k, out_s = [], [], []
        for c0 in range(0, len(P), chunk):
            Q = P[c0:c0 + chunk]
            uv = np.array([Q[:, i], Q[:, j]]).T
            cc = np.floor((uv - self.bmin) / CELL).astype(int)
            inside = (cc >= 0).all(1)
            cell = cc[:, 0] * self.ncol + cc[:, 1]
            l = np.searchsorted(self.cell, cell, 'left')
            r = np.searchsorted(self.cell, cell, 'right')
            r[~inside] = l[~inside]
            cnt = r - l
            if cnt.sum() == 0:
                continue
            pi = np.repeat(np.arange(len(Q)), cnt)
            ti = self.tri[np.repeat(l, cnt) + (np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt))]
            u, v = uv[pi, 0], uv[pi, 1]
            m = (u >= self.lo[ti, 0]) & (u <= self.hi[ti, 0]) & (v >= self.lo[ti, 1]) & (v <= self.hi[ti, 1])
            pi, ti, u, v = pi[m], ti[m], u[m], v[m]
            A, B, C = self.T[ti, 0], self.T[ti, 1], self.T[ti, 2]
            x0, y0 = A[:, i], A[:, j]
            x1, y1 = B[:, i] - x0, B[:, j] - y0
            x2, y2 = C[:, i] - x0, C[:, j] - y0
            px, py = u - x0, v - y0
            det = x1 * y2 - x2 * y1
            ok = np.abs(det) > 1e-14
            det = np.where(ok, det, 1.0)
            b1 = (px * y2 - x2 * py) / det
            b2 = (x1 * py - px * y1) / det
            tol = 1e-9
            m = ok & (b1 >= -tol) & (b2 >= -tol) & (b1 + b2 <= 1 + tol)
            k = A[:, self.ax] + b1 * (B[:, self.ax] - A[:, self.ax]) + b2 * (C[:, self.ax] - A[:, self.ax])
            out_p.append(pi[m] + c0)
            out_k.append(k[m])
            out_s.append(np.sign(self.n[ti[m], self.ax]))
        if not out_p:
            return np.zeros(0, int), np.zeros(0), np.zeros(0)
        return np.concatenate(out_p), np.concatenate(out_k), np.concatenate(out_s)


def pair_gap(Pm, Cm, Ps, Cs, ax):
    """Free travel (t_plus >= 0, t_minus <= 0) of moving solid m (samples Pm, caster Cm) against static s."""
    tp, tm = np.inf, -np.inf
    wp = wm = None
    i, j = Cs.tr
    Pm = Pm[(Pm[:, i] >= Cs.lo[:, 0].min()) & (Pm[:, i] <= Cs.hi[:, 0].max()) & (Pm[:, j] >= Cs.lo[:, 1].min()) & (Pm[:, j] <= Cs.hi[:, 1].max())]
    Ps = Ps[(Ps[:, i] >= Cm.lo[:, 0].min()) & (Ps[:, i] <= Cm.hi[:, 0].max()) & (Ps[:, j] >= Cm.lo[:, 1].min()) & (Ps[:, j] <= Cm.hi[:, 1].max())]
    # moving points against static faces
    pi, k, s = Cs.cast(Pm)
    if len(pi):
        d = k - Pm[pi, ax]
        sel = (s < 0) & (d >= 0)            # travelling +ax into a face that faces -ax
        if sel.any():
            q = np.argmin(np.where(sel, d, np.inf)); tp = d[q]; wp = Pm[pi[q]]
        sel = (s > 0) & (d <= 0)
        if sel.any():
            q = np.argmax(np.where(sel, d, -np.inf)); tm = d[q]; wm = Pm[pi[q]]
    # static points against moving faces: moving +t brings its face at k to the point at p
    pi, k, s = Cm.cast(Ps)
    if len(pi):
        d = Ps[pi, ax] - k
        sel = (s > 0) & (d >= 0)
        if sel.any():
            q = np.argmin(np.where(sel, d, np.inf))
            if d[q] < tp:
                tp = d[q]; wp = Ps[pi[q]]
        sel = (s < 0) & (d <= 0)
        if sel.any():
            q = np.argmax(np.where(sel, d, -np.inf))
            if d[q] > tm:
                tm = d[q]; wm = Ps[pi[q]]
    return tp, tm, wp, wm
