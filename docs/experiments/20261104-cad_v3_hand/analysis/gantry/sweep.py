#!/usr/bin/env python3
"""Translational sweeps of closed meshes along a coordinate axis.

A closed, outward-oriented mesh is scan-converted on a grid of lines parallel to axis k: every grid line
(spacing h, in the plane of the two other axes) crosses the surface at sorted depths that pair into solid
intervals [lo, hi] along k. A moving part overlaps a fixed part after a displacement s along +k exactly when, on a
shared grid line, one of its intervals [lo_m + s, hi_m + s] meets one of the fixed part's intervals [lo_f, hi_f],
i.e. s in (lo_f - hi_m, hi_f - lo_m). The free range around s = 0 is the gap between the blocked intervals; it is
exact along k and sampled at h across it. A shift of the moving group across the grid (along the other two axes) is
an integer shift of its grid indices, so travel maps over a second axis cost one join per shift.
"""
from __future__ import annotations

import numpy as np

OFF_CELL = 1 << 20          # grid index offset, so cell ids are non-negative
MUL = 1 << 21


def cell_id(a, b):
    return (np.asarray(a, np.int64) + OFF_CELL) * MUL + (np.asarray(b, np.int64) + OFF_CELL)


def rasterize(mesh, k: int, h: float, off=(0.01234, 0.04321), chunk=4_000_000):
    """Solid intervals of `mesh` along axis k. Returns dict(cell, lo, hi, bad) with cell ids of the grid lines
    u = off[0] + a h, v = off[1] + b h on the two other axes (in increasing axis order), and `bad`, the number of
    grid lines whose crossings did not alternate entry/exit (dropped)."""
    i, j = [m for m in range(3) if m != k]
    T = np.asarray(mesh.triangles, float)
    nk = np.asarray(mesh.face_normals, float)[:, k]
    keep = np.abs(nk) > 1e-9
    T, nk = T[keep], nk[keep]
    U, V, W = T[:, :, i], T[:, :, j], T[:, :, k]
    a0 = np.ceil((U.min(1) - off[0]) / h).astype(np.int64)
    a1 = np.floor((U.max(1) - off[0]) / h).astype(np.int64)
    b0 = np.ceil((V.min(1) - off[1]) / h).astype(np.int64)
    b1 = np.floor((V.max(1) - off[1]) / h).astype(np.int64)
    na, nb = a1 - a0 + 1, b1 - b0 + 1
    cnt = np.where((na > 0) & (nb > 0), na * nb, 0)
    cells, depth, sgn = [], [], []
    order = np.arange(len(T))
    cum = np.cumsum(cnt)
    starts = np.searchsorted(cum, np.arange(0, cum[-1] if len(cum) else 0, chunk), side="right") if len(cum) else []
    bounds = list(starts) + [len(T)]
    for s0, s1 in zip(bounds[:-1], bounds[1:]):
        idx = order[s0:s1]
        c = cnt[idx]
        if c.sum() == 0:
            continue
        t = np.repeat(idx, c)
        st = np.cumsum(c) - c
        loc = np.arange(c.sum()) - np.repeat(st, c)
        nar = np.repeat(na[idx], c)
        aa = np.repeat(a0[idx], c) + loc % nar
        bb = np.repeat(b0[idx], c) + loc // nar
        pu, pv = off[0] + aa * h, off[1] + bb * h
        u0, u1, u2 = U[t, 0], U[t, 1], U[t, 2]
        v0, v1, v2 = V[t, 0], V[t, 1], V[t, 2]
        d = (v1 - v2) * (u0 - u2) + (u2 - u1) * (v0 - v2)
        ok = np.abs(d) > 1e-14
        d = np.where(ok, d, 1.0)
        l0 = ((v1 - v2) * (pu - u2) + (u2 - u1) * (pv - v2)) / d
        l1 = ((v2 - v0) * (pu - u2) + (u0 - u2) * (pv - v2)) / d
        l2 = 1.0 - l0 - l1
        ins = ok & (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
        w = l0 * W[t, 0] + l1 * W[t, 1] + l2 * W[t, 2]
        cells.append(cell_id(aa[ins], bb[ins]))
        depth.append(w[ins])
        sgn.append(np.sign(nk[t[ins]]))
    if not cells:
        return dict(cell=np.zeros(0, np.int64), lo=np.zeros(0), hi=np.zeros(0), bad=0)
    cell = np.concatenate(cells)
    dep = np.concatenate(depth)
    sg = np.concatenate(sgn)
    o = np.lexsort((dep, cell))
    cell, dep, sg = cell[o], dep[o], sg[o]
    # group by cell; a valid line has an even count with signs -,+,-,+ (enter through an outward normal along -k)
    uc, first, count = np.unique(cell, return_index=True, return_counts=True)
    pos = np.arange(len(cell)) - np.repeat(first, count)
    expect = np.where(pos % 2 == 0, -1.0, 1.0)
    okc = np.repeat(count % 2 == 0, count) & True
    good_line = np.ones(len(uc), bool)
    badpos = (sg != expect) | ~okc
    if badpos.any():
        good_line[np.unique(np.searchsorted(uc, cell[badpos]))] = False
    keep = np.repeat(good_line, count)
    cell, dep, pos = cell[keep], dep[keep], pos[keep]
    lo = dep[pos % 2 == 0]
    hi = dep[pos % 2 == 1]
    return dict(cell=cell[pos % 2 == 0], lo=lo, hi=hi, bad=int((~good_line).sum()))


def join(M, F, shift_cells=0):
    """All pairs of intervals of M (shifted by shift_cells in cell id units) and F on shared grid lines.
    Returns (iM, iF) index arrays into M's and F's interval arrays."""
    cm = M["cell"] + shift_cells
    of = np.argsort(F["cell"], kind="stable")
    cf = F["cell"][of]
    left = np.searchsorted(cf, cm, side="left")
    right = np.searchsorted(cf, cm, side="right")
    n = right - left
    iM = np.repeat(np.arange(len(cm)), n)
    st = np.cumsum(n) - n
    iF = of[np.repeat(left, n) + (np.arange(n.sum()) - np.repeat(st, n))]
    return iM, iF


def blocked(M, F, shift_cells=0):
    """Displacement intervals (s_lo, s_hi) along +k at which M overlaps F, one per interval pair, with indices."""
    iM, iF = join(M, F, shift_cells)
    return F["lo"][iF] - M["hi"][iM], F["hi"][iF] - M["lo"][iM], iM, iF


def free_range(s_lo, s_hi, tol=0.02):
    """The free displacement range around 0: (lower, upper, index of the pair setting each, penetration depth).
    Pairs whose blocked interval contains 0 by more than tol on both sides are a penetration at s = 0."""
    if len(s_lo) == 0:
        return -np.inf, np.inf, -1, -1, 0.0
    pen = (s_lo < -tol) & (s_hi > tol)
    depth = float(np.max(np.minimum(-s_lo[pen], s_hi[pen]))) if pen.any() else 0.0
    ahead = ~pen & (s_lo >= -tol)
    behind = ~pen & (s_hi <= tol)
    up = np.where(ahead, s_lo, np.inf)
    dn = np.where(behind, s_hi, -np.inf)
    iu = int(np.argmin(up)) if ahead.any() else -1
    idn = int(np.argmax(dn)) if behind.any() else -1
    return (float(dn[idn]) if idn >= 0 else -np.inf, float(up[iu]) if iu >= 0 else np.inf, idn, iu, depth)


def cell_uv(cell, h, off=(0.01234, 0.04321)):
    a = cell // MUL - OFF_CELL
    b = cell % MUL - OFF_CELL
    return off[0] + a * h, off[1] + b * h


def concat(parts: dict, names):
    """Stack several parts' intervals, keeping the part name of every interval."""
    cells, lo, hi, who = [], [], [], []
    for nm in names:
        R = parts[nm]
        cells.append(R["cell"])
        lo.append(R["lo"])
        hi.append(R["hi"])
        who.append(np.full(len(R["cell"]), nm, dtype=object))
    return dict(cell=np.concatenate(cells), lo=np.concatenate(lo), hi=np.concatenate(hi), who=np.concatenate(who))
