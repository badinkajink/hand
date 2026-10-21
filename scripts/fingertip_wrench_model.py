"""Measurement model of the magnetic fingertip on the ReSkin 5X board, its calibration from the reading at rest, and the
contact estimator built on it.

Frames (mm, the tip frame of tpu_tip_design): x along the pad's normal toward the palmar face, y across the finger, z along
it toward the tip end. An FE row of docs/experiments/20261019-v2_tip_fe_checks/sense_fe.jsonl places the magnets;
mlx90393_model.board_points places the five chips on the sensing plane (U1 at the centre, U2 7 mm toward the tip end,
U3 7 mm toward +y, U4 7 mm toward the mount, U5 7 mm toward -y).

Chip c reports b_c = R_c (B(p_c) + g): B is the magnets' field, g a uniform field (Earth's), and R_c the rotation from
the tip frame to the chip's axes. The chip's z axis is s e_x and its x axis lies at angle theta0 + phi_c from e_y about
e_x, phi_c being the part's rotation in the board's pick-and-place file (U1 0, U2 180, U3 0, U4 90, U5 270 degrees); s and
theta0 describe the part's axes in its footprint and are common to the five chips. The stream's chip i sits at U[perm[i]].
Magnet j sits at its FE centre shifted by (dx, dy, dz) (the sensing plane dx farther away, the pad offset dy, dz on the
board), with moment k s_j Br V / mu0 along e_x (N52: Br 1.45 T; s_j its pole, k the strength relative to N52).
"""
from __future__ import annotations

import itertools
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import mlx90393_model as MLX  # noqa: E402
import tpu_tip_design as TD  # noqa: E402

FE_ROWS = ROOT / "docs/experiments/20261019-v2_tip_fe_checks/sense_fe.jsonl"
GAP_B = 3.32 - 6.45            # concept b's sensing plane behind the whole tip's back face, mm (v2_tip_fe_checks.SENSE_GAP)
BR = 1.45
PHI_PP = {"U1": 0.0, "U2": 180.0, "U3": 0.0, "U4": 90.0, "U5": 270.0}   # 5X_ReSkin.Zip pick-and-place rotations, degrees
U_ORDER = ("U1", "U2", "U3", "U4", "U5")
NOISE_UT = np.array([1.65, 1.65, 2.57])   # chip x, y, z noise per sample at GAIN_SEL 1, RES 1, DIG_FILT 5 (2026-10-20)


def fe_row(case="b_v2"):
    for line in open(FE_ROWS):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("case") == case:
            return r
    raise KeyError(case)


def magnets(row):
    """Centres (mm, (5, 3)) and nominal N52 moments (A m^2) of the row's magnets."""
    c, m = [], []
    for d, t, y, z in row["magnets"]:
        c.append([TD.X_FACE - row["depth_mm"] - t / 2, y, z])
        m.append(BR * math.pi * (d / 2e3) ** 2 * (t / 1e3) / (4e-7 * math.pi))
    return np.array(c), np.array(m)


def chip_points(gap=GAP_B):
    """Sensing points (mm, (5, 3)) of U1-U5."""
    return np.array(MLX.board_points(gap))


def cylinder_quadrature(d, t, n_r=3, n_a=8, n_z=3):
    """Points (mm, offsets from the centre, axis along x) and weights (sum 1) of a volume quadrature of a cylinder of
    diameter d and thickness t mm: Gauss-Legendre in r (with the r dr measure) and in x, uniform in angle. A uniformly
    magnetized cylinder's field outside it equals that of the weighted point dipoles."""
    xr, wr = np.polynomial.legendre.leggauss(n_r)
    xz, wz = np.polynomial.legendre.leggauss(n_z)
    r = (xr + 1) / 2 * d / 2
    wr = wr * r                                   # r dr
    pts, w = [], []
    for ri, wri in zip(r, wr):
        for a in np.arange(n_a) * 2 * math.pi / n_a + (math.pi / n_a if ri > 0 else 0):
            for zi, wzi in zip(xz * t / 2, wz):
                pts.append([zi, ri * math.cos(a), ri * math.sin(a)])
                w.append(wri * wzi)
    w = np.array(w)
    return np.array(pts), w / w.sum()


def dipole_field(p, c, m):
    """Field (T) at points p (n, 3) mm of dipoles at c (j, 3) mm with moment vectors m (j, 3) A m^2, summed over j."""
    r = (p[:, None, :] - c[None, :, :]) * 1e-3
    d = np.linalg.norm(r, axis=2)
    h = r / d[..., None]
    hm = np.einsum("njk,jk->nj", h, m)
    return 1e-7 * np.einsum("nj,njk->nk", 1 / d ** 3, 3 * h * hm[..., None] - m[None])


def chip_rotation(phi_deg, theta0_deg, s):
    """Rows: the chip's x, y, z axes in the tip frame (right-handed, z = s e_x)."""
    a = math.radians(theta0_deg + phi_deg)
    ex, ey, ez = np.eye(3)
    xc = math.cos(a) * ey + math.sin(a) * ez
    zc = s * ex
    return np.stack([xc, np.cross(zc, xc), zc])


@dataclass
class Calib:
    """Parameters of the measurement model; perm[i] is the U index (0-4) of stream chip i."""
    case: str = "b_v2"
    poles: tuple = (1, 1, 1, 1, 1)
    s: int = 1
    theta0: float = 0.0
    perm: tuple = (0, 1, 2, 3, 4)
    k: float = 1.0
    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0
    g_uT: tuple = (0.0, 0.0, 0.0)
    rms_mT: float = float("nan")
    b_rest_uT: list = field(default_factory=list)

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=1) + "\n")

    @staticmethod
    def load(path):
        d = json.loads(Path(path).read_text())
        for key in ("poles", "perm", "g_uT"):
            d[key] = tuple(d[key])
        return Calib(**d)


class Model:
    """The measurement model of one calibration: rest field, chip rotations and the field change for magnet motions."""

    def __init__(self, cal: Calib, row=None):
        self.cal = cal
        self.row = row or fe_row(cal.case)
        c, m = magnets(self.row)
        self.mc = c                                      # magnets at their FE centres, where the FE motions apply
        self.mu = (cal.k * np.array(cal.poles) * m)[:, None] * np.array([1.0, 0.0, 0.0])[None]
        d, t = self.row["magnets"][0][:2]
        self.q_off, self.q_w = cylinder_quadrature(d, t)
        P = chip_points() - np.array([cal.dx, cal.dy, cal.dz])[None]     # the chips carry the offsets
        self.P = P[list(cal.perm)]                       # sensing point of each stream chip
        self.R = np.stack([chip_rotation(PHI_PP[U_ORDER[u]], cal.theta0, cal.s) for u in cal.perm])

    def field(self, mc, mu, rot=None):
        """Field (T, (5, 3) tip frame) at the chips of the magnets at centres mc with moments mu, each a cylinder by
        quadrature; rot (j, 3) small rotation vectors (rad) turn each magnet's quadrature points."""
        off = self.q_off[None].repeat(len(mc), 0)
        if rot is not None:
            off = off + np.cross(rot[:, None, :], off)
        pts = (mc[:, None, :] + off).reshape(-1, 3)
        mom = (mu[:, None, :] * self.q_w[None, :, None]).reshape(-1, 3)
        return dipole_field(self.P, pts, mom)

    def rest_uT(self, g_uT=None):
        """Reading (uT, (5, 3) chip frames) at rest."""
        g = np.array(self.cal.g_uT if g_uT is None else g_uT) * 1e-6
        B = self.field(self.mc, self.mu) + g[None]
        return np.einsum("cij,cj->ci", self.R, B) * 1e6

    def motion_uT(self, motions):
        """Reading change (uT, (15,)) for rigid magnet motions: [(t_um (3), w_mrad (3)) per magnet], or an array
        (magnets, 6) of t_um then w_mrad."""
        if isinstance(motions, np.ndarray) and motions.ndim == 2 and motions.shape[1] == 6:
            t, w = motions[:, :3] * 1e-3, motions[:, 3:] * 1e-3
        else:
            t = np.array([mo[0] for mo in motions]) * 1e-3          # mm
            w = np.array([mo[1] for mo in motions]) * 1e-3          # rad
        dB = self.field(self.mc + t, self.mu + np.cross(w, self.mu), w) - self.field(self.mc, self.mu)
        return (np.einsum("cij,cj->ci", self.R, dB) * 1e6).ravel()


def _resid(b_rest, row, poles, s, perm, p, theta0=None):
    k, dx, dy, dz = p[:4]
    th = p[4] if theta0 is None else theta0
    g = p[5:8] if theta0 is None else p[4:7]
    cal = Calib(poles=poles, s=s, theta0=th, perm=perm, k=k, dx=dx, dy=dy, dz=dz, g_uT=tuple(g))
    return ((Model(cal, row).rest_uT() - b_rest) / 1e3).ravel()      # mT


def calibrate(b_rest_uT, case="b_v2", poles=None, top=24, verbose=True):
    """Fit the model to the reading at rest (uT, (5, 3), streaming order). poles=None searches the 16 patterns with the
    centre magnet's pole +1 (the global sign goes into s). Returns the best Calib and the ranked fits."""
    row = fe_row(case)
    b = np.asarray(b_rest_uT, float)
    pole_set = [poles] if poles is not None else [(c[0], c[1], 1, c[2], c[3]) for c in itertools.product((1, -1), repeat=4)]
    # stage 1: rotation-invariant features (normal component, in-plane magnitude) choose the candidates
    cand = []
    for pl in pole_set:
        for s in (1, -1):
            for perm in itertools.permutations(range(1, 5)):
                perm = (0,) + perm
                def f_inv(p):
                    cal = Calib(poles=pl, s=s, perm=perm, k=p[0], dx=p[1], dy=p[2], dz=p[3])
                    r = Model(cal, row).rest_uT()
                    return np.r_[r[:, 2] - b[:, 2], np.hypot(r[:, 0], r[:, 1]) - np.hypot(b[:, 0], b[:, 1])] / 1e3
                best = min((least_squares(f_inv, x0, bounds=([0.1, -3, -4, -4], [2.0, 8, 4, 4]))
                            for x0 in ([1, 0, 0, 0], [0.4, 0, 0, 0])), key=lambda r: r.cost)
                cand.append((math.sqrt(2 * best.cost / 10), pl, s, perm, best.x))
    cand.sort(key=lambda c: c[0])
    # stage 2: all 15 components with the pick-and-place rotations and one common offset theta0; for the best pole
    # pattern and sign, every assignment, so that the 180-degree turn of the board is compared directly
    pl0, s0 = cand[0][1], cand[0][2]
    stage2 = cand[:top] + [c for c in cand[top:] if c[1] == pl0 and c[2] == s0]
    fits = []
    for rms_inv, pl, s, perm, x in stage2:
        for th0 in (0.0, 90.0, 180.0, 270.0):
            r = least_squares(lambda p: _resid(b, row, pl, s, perm, p), list(x) + [th0, 0, 0, 0],
                              bounds=([0.1, -3, -4, -4, th0 - 60, -100, -100, -100], [2.0, 8, 4, 4, th0 + 60, 100, 100, 100]))
            fits.append(dict(rms_mT=float(np.sqrt(np.mean(r.fun ** 2))), rms_inv_mT=rms_inv, poles=pl, s=s, perm=perm,
                             k=r.x[0], dx=r.x[1], dy=r.x[2], dz=r.x[3], theta0=r.x[4] % 360, g_uT=r.x[5:8]))
    fits.sort(key=lambda f: f["rms_mT"])
    f = fits[0]
    cal = Calib(case=case, poles=tuple(int(v) for v in f["poles"]), s=int(f["s"]), theta0=float(f["theta0"]),
                perm=tuple(int(v) for v in f["perm"]), k=float(f["k"]), dx=float(f["dx"]), dy=float(f["dy"]),
                dz=float(f["dz"]), g_uT=tuple(float(v) for v in f["g_uT"]), rms_mT=f["rms_mT"], b_rest_uT=b.tolist())
    if verbose:
        print("rms (mT)  invariant  poles              s   perm (U of stream 0-4)   theta0   k     dx    dy    dz")
        for q in fits[:10]:
            print(f"  {q['rms_mT']:.3f}    {q['rms_inv_mT']:.3f}     {str(q['poles']):18s} {q['s']:+d}  {str(q['perm']):22s} "
                  f"{q['theta0']:6.1f}  {q['k']:.3f} {q['dx']:+.2f} {q['dy']:+.2f} {q['dz']:+.2f}")
    return cal, fits


def calibrate_frames_fixed(b_rest_uT, board: Calib, case=None, poles=None):
    """Fit poles (16 patterns unless given), strength and offsets with the chip frames and assignment of `board` (a
    previous full calibration of the same board) kept: seconds instead of a minute, for a new pad or new magnets."""
    row = fe_row(case or board.case)
    b = np.asarray(b_rest_uT, float)
    pole_set = [poles] if poles is not None else [(c[0], c[1], 1, c[2], c[3]) for c in itertools.product((1, -1), repeat=4)]
    best = None
    for pl in pole_set:
        for x0 in ([1.0, 0, 0, 0], [0.4, 0, 0, 0]):
            r = least_squares(lambda p: _resid(b, row, pl, board.s, board.perm, p, theta0=board.theta0), x0 + [0, 0, 0],
                              bounds=([0.1, -3, -4, -4, -100, -100, -100], [2.0, 8, 4, 4, 100, 100, 100]))
            if best is None or r.cost < best[0].cost:
                best = (r, pl)
    r, pl = best
    return Calib(case=case or board.case, poles=tuple(int(v) for v in pl), s=board.s, theta0=board.theta0, perm=board.perm,
                 k=float(r.x[0]), dx=float(r.x[1]), dy=float(r.x[2]), dz=float(r.x[3]), g_uT=tuple(float(v) for v in r.x[4:7]),
                 rms_mT=float(np.sqrt(np.mean(r.fun ** 2))), b_rest_uT=b.tolist())


def load_basis(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


class Estimator:
    """Contact point and force from one reading change d (uT, (15,), chip frames, streaming order), by variable
    projection. The model of d is S(u) f + U g + b0 eps + noise: S (15 x 3) the change per newton of a patch load at face
    point u = (y, z) pressing along -x and shearing along +y and +z (FE basis through the field model), U (15 x 3) the
    chips' rotations (a uniform field change g, from turning the finger in Earth's field), b0 the reading at rest (a
    relative change eps of the magnets' strength, from temperature). For every u of a fine grid the force and the nuisance
    terms are solved by weighted least squares, and u maximizes the explained energy e(u) = d^T W S (S^T W S)^-1 S^T W d
    after the nuisance terms are projected out."""

    def __init__(self, model: Model, basis: dict, step=0.25, noise_uT=NOISE_UT):
        from scipy.interpolate import CloughTocher2DInterpolator
        self.model, self.basis = model, basis
        pts = basis["pts"]
        S_fe = np.array([[model.motion_uT(basis["motions"][i, k]) for k in range(3)] for i in range(len(pts))])
        S_fe = S_fe.transpose(0, 2, 1)                      # (n_fe, 15, 3)
        ys = np.arange(pts[:, 0].min(), pts[:, 0].max() + 1e-9, step)
        zs = np.arange(pts[:, 1].min(), pts[:, 1].max() + 1e-9, step)
        Y, Z = np.meshgrid(ys, zs, indexing="ij")
        interp = CloughTocher2DInterpolator(pts, S_fe.reshape(len(pts), -1))
        S = interp(np.c_[Y.ravel(), Z.ravel()]).reshape(len(ys), len(zs), 15, 3)
        self.ys, self.zs = ys, zs
        self.valid = ~np.isnan(S).any(axis=(2, 3))
        self.S = np.nan_to_num(S)
        self.sig = np.tile(np.asarray(noise_uT, float), 5)
        self.U = np.concatenate(list(model.R), 0)            # (15, 3): chip c reads R_c g
        self.b0 = np.asarray(model.cal.b_rest_uT, float).ravel() if model.cal.b_rest_uT else model.rest_uT().ravel()
        self.set_noise_scale(1.0)

    def set_noise_scale(self, scale):
        """Noise of the averaged reading relative to one sample (1/sqrt(conversions averaged))."""
        sig = self.sig * scale
        N = np.c_[self.U, self.b0] / sig[:, None]
        self.Q, _ = np.linalg.qr(N)
        self.N_w = N
        Sw = self.S / sig[None, None, :, None]
        Sp = Sw - np.einsum("ij,abjk->abik", self.Q, np.einsum("ji,abjk->abik", self.Q, Sw))
        self.Sp = Sp
        G = np.einsum("abik,abil->abkl", Sp, Sp)
        G[~self.valid] = np.eye(3)
        self.Ginv = np.linalg.inv(G)
        self.sig_now = sig

    def _interp(self, y, z, arr):
        """Bilinear interpolation of arr (ny, nz, ...) at points (y, z) (k,), and whether all four corners are on the
        face."""
        h = self.ys[1] - self.ys[0]
        fy, fz = (np.asarray(y, float) - self.ys[0]) / h, (np.asarray(z, float) - self.zs[0]) / h
        i0 = np.clip(np.floor(fy).astype(int), 0, len(self.ys) - 2)
        j0 = np.clip(np.floor(fz).astype(int), 0, len(self.zs) - 2)
        sh = (-1,) + (1,) * (arr.ndim - 2)
        ty, tz = (fy - i0).reshape(sh), (fz - j0).reshape(sh)
        ok = self.valid[i0, j0] & self.valid[i0 + 1, j0] & self.valid[i0, j0 + 1] & self.valid[i0 + 1, j0 + 1]
        ok &= (fy >= 0) & (fy <= len(self.ys) - 1) & (fz >= 0) & (fz <= len(self.zs) - 1)
        v = (arr[i0, j0] * (1 - ty) * (1 - tz) + arr[i0 + 1, j0] * ty * (1 - tz) + arr[i0, j0 + 1] * (1 - ty) * tz
             + arr[i0 + 1, j0 + 1] * ty * tz)
        return v, ok

    def estimate(self, d):
        """Best contact for reading change d: dict with contact (bool), y, z (mm), f (press along -x, shear along +y,
        +z; N), cov (5 x 5 over f, y, z), e (explained energy, chi-square units), chi2_none (the reading's energy after
        the nuisance terms), g_uT and eps (the nuisance estimates)."""
        dw = np.asarray(d, float) / self.sig_now
        dp = dw - self.Q @ (self.Q.T @ dw)
        T = np.einsum("abik,i->abk", self.Sp, dp)
        f = np.einsum("abkl,abl->abk", self.Ginv, T)
        e = np.einsum("abk,abk->ab", T, f)
        e[~self.valid] = -np.inf
        e[f[..., 0] < 0] = -np.inf                          # a contact presses
        if not np.isfinite(e).any():
            nuis = np.linalg.lstsq(self.N_w, dw, rcond=None)[0]
            return dict(contact=False, chi2_none=float(dp @ dp), e=0.0, g_uT=nuis[:3].tolist(), eps=float(nuis[3]))
        i, j = np.unravel_index(np.argmax(e), e.shape)
        # continuous refinement: the projected sensitivities interpolated over +-1 grid step at 1/10 of a step
        h = self.ys[1] - self.ys[0]
        o = np.linspace(-h, h, 21)
        Y, Z = np.meshgrid(self.ys[i] + o, self.zs[j] + o, indexing="ij")
        Y, Z = Y.ravel(), Z.ravel()
        Sl, ok = self._interp(Y, Z, self.Sp)
        Tl = np.einsum("kil,i->kl", Sl, dp)
        Gl = np.einsum("kil,kim->klm", Sl, Sl)
        Gl[~ok] = np.eye(3)
        fl = np.linalg.solve(Gl, Tl[..., None])[..., 0]
        el = np.einsum("kl,kl->k", Tl, fl)
        el[~ok | (fl[:, 0] < 0)] = -np.inf
        if np.isfinite(el).any():
            k = int(np.argmax(el))
            y, z, fb, eb, Sb = float(Y[k]), float(Z[k]), fl[k], float(el[k]), Sl[k]
        else:
            y, z, fb, eb, Sb = float(self.ys[i]), float(self.zs[j]), f[i, j], float(e[i, j]), self.Sp[i, j]
        # Fisher information of (f, y, z) at the estimate, nuisance projected out
        dd = 0.05
        Sy = (self._interp([y + dd], [z], self.Sp)[0][0] - self._interp([y - dd], [z], self.Sp)[0][0]) / (2 * dd)
        Sz = (self._interp([y], [z + dd], self.Sp)[0][0] - self._interp([y], [z - dd], self.Sp)[0][0]) / (2 * dd)
        J = np.c_[Sb, Sy @ fb, Sz @ fb]
        try:
            cov = np.linalg.inv(J.T @ J)
        except np.linalg.LinAlgError:
            cov = np.full((5, 5), np.nan)
        Su = self._interp([y], [z], self.S)[0][0] / self.sig_now[:, None]
        nuis = np.linalg.lstsq(self.N_w, dw - Su @ fb, rcond=None)[0]
        return dict(contact=True, y=y, z=z, f=fb.tolist(), cov=cov.tolist(), e=eb, chi2_none=float(dp @ dp),
                    g_uT=nuis[:3].tolist(), eps=float(nuis[3]))
