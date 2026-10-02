"""Theory figures for docs/experiments/20261002-hom_chain: Drake's hydroelastic pressure field on the real_v1
fingertip against MuJoCo's sphere-packed pad, both pressed onto the screwdriver at the same pinch force, from the
real geometry (fingertip sphere R 10.55 mm, tool cylinder r 12.5 mm, E 10 MPa, pad spheres 0.75 mm on a 45 deg cap,
per-sphere spring (E/R) A_s ((R - r_s)/R)^2 as in scripts/hom_chain.chain_scene). Also the benchmark plot.
Pure numpy; imported by scripts/hom_chain_page.py.
"""
from __future__ import annotations

import math

import numpy as np

R, RC, E, RS, CAP = 0.01055, 0.0125, 1e7, 0.00075, math.radians(45.0)
AREA_CAP = 2 * math.pi * R ** 2 * (1 - math.cos(CAP))


def fib_cap(n, cap):
    k = np.arange(n) + 0.5
    z = 1.0 - (1.0 - math.cos(cap)) * k / n
    r = np.sqrt(1.0 - z ** 2)
    phi = k * math.pi * (3.0 - math.sqrt(5.0))
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


# Tool axis along x, contact at the top of the cylinder (z = RC); fingertip centre above it at RC + R - delta0.
def field(delta0, h=2.5e-5, L=0.0042):
    """Drake's compliant-sphere pressure p = E d / R on the rigid tool surface (d = depth inside the sphere)."""
    s = np.arange(-L, L + h / 2, h)
    X, S = np.meshgrid(s, s, indexing="ij")              # x along the axis, S arc length around it
    th = S / RC
    Py, Pz = RC * np.sin(th), RC * np.cos(th)
    d = R - np.sqrt(X ** 2 + Py ** 2 + (Pz - (RC + R - delta0)) ** 2)
    p = np.where(d > 0, E * d / R, 0.0)
    dA = h * h
    rho = np.sqrt(X ** 2 + Py ** 2)
    F = float(p.sum() * dA)
    T = float((p * rho).sum() * dA)                      # friction torque / mu about the contact normal
    return {"F": F, "arm": T / F if F > 0 else 0.0, "area": float((p > 0).sum() * dA), "pmax": float(p.max()),
            "x": s, "p": p, "delta0": delta0}


def pad(spacing, delta0):
    n = max(1, int(round(AREA_CAP / spacing ** 2)))
    A_s = AREA_CAP / n
    K = E / R * A_s * ((R - RS) / R) ** 2
    dirs = fib_cap(n, CAP) * np.array([1.0, 1.0, -1.0])  # cap facing the tool (-z)
    c = np.array([0.0, 0.0, RC + R - delta0]) + (R - RS) * dirs
    gap = np.sqrt(c[:, 1] ** 2 + c[:, 2] ** 2) - RC
    over = RS - gap
    f = np.where(over > 0, K * over, 0.0)
    rho = np.sqrt(c[:, 0] ** 2 + c[:, 1] ** 2)
    F = float(f.sum())
    return {"F": F, "arm": float((f * rho).sum() / F) if F > 0 else 0.0, "n": n, "n_in": int((f > 0).sum()),
            "A_s": A_s, "K": K, "centres": c, "f": f, "delta0": delta0, "spacing": spacing}


def solve(fn, F, lo=1e-6, hi=2e-3):
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if fn(mid)["F"] < F:
            lo = mid
        else:
            hi = mid
    return fn(0.5 * (lo + hi))


def compute(F=3.0):
    out = {"F": F, "field": solve(field, F)}
    for s in (0.002, 0.001, 0.0005):
        out[s] = solve(lambda d, s=s: pad(s, d), F)
    return out


if __name__ == "__main__":
    for F in (0.5, 3.0):
        r = compute(F)
        fd = r["field"]
        print(f"F {F} N  field: delta0 {fd['delta0'] * 1e3:.3f} mm  area {fd['area'] * 1e6:.1f} mm2  pmax {fd['pmax'] / 1e6:.3f} MPa  "
              f"arm {fd['arm'] * 1e3:.3f} mm  (rig law {0.996 * F ** 0.25:.3f})")
        for s in (0.002, 0.001, 0.0005):
            q = r[s]
            print(f"   pad {s * 1e3:g} mm: n {q['n']}  in contact {q['n_in']}  delta0 {q['delta0'] * 1e3:.3f} mm  arm {q['arm'] * 1e3:.3f} mm  "
                  f"K {q['K']:.0f} N/m")


# ------------------------------------------------------------------------------------------- SVG

def _f(v):
    return f"{v:.1f}"


def _poly(pts):
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)


DRAKE, SPHERE, C4 = "var(--c-drake)", "var(--c-sphere)", "var(--c-c4)"
TXT = 'font-family="var(--f-mono)" font-size="12" fill="currentColor"'
TXT2 = 'font-family="var(--f-mono)" font-size="11.5" fill="currentColor" opacity="0.72"'


def svg_section(res, spacing=0.001):
    """Figure 1: section along the tool axis through the patch centre, field (left) against a row of pad spheres
    (right), true scale, and the pressure each delivers to the tool surface below."""
    F = res["F"]
    fd, pd = res["field"], res[spacing]
    k = 66.0                                  # px per mm
    W, X0 = 1040, (30.0, 550.0)
    xmin, xmax = -3.5, 3.5
    zt, zb = 1.95, -0.62                      # mm above/below the tool surface shown
    yt = 58.0
    Y = lambda z: yt + (zt - z) * k           # noqa: E731
    y0 = Y(0.0)
    ybot = Y(zb)
    ps0, ps1 = ybot + 34, ybot + 34 + 118     # pressure strip top/bottom
    pmax = 0.36                               # MPa full scale
    P = lambda p: ps1 - p / pmax * (ps1 - ps0)  # noqa: E731
    H_ = int(ps1 + 46)
    out = [f'<svg viewBox="0 0 {W} {H_}" role="img" aria-label="Section through the thumb contact at {F:g} N: Drake '
           f'integrates a pressure field over the overlap of fingertip and tool; the sphere pad replaces it with one '
           f'spring per sphere overlap, and the spring pressures sample the same profile.">',
           f'<defs><clipPath id="f1l"><rect x="{X0[0]}" y="{yt - 2}" width="{(xmax - xmin) * k}" height="{ybot - yt + 2}"/></clipPath>'
           f'<clipPath id="f1r"><rect x="{X0[1]}" y="{yt - 2}" width="{(xmax - xmin) * k}" height="{ybot - yt + 2}"/></clipPath>'
           f'<marker id="f1a" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
           f'<path d="M0,0 L8,4 L0,8 z" fill="currentColor"/></marker></defs>']
    R_mm, RS_mm = R * 1e3, RS * 1e3
    for side, X in zip(("field", "pad"), X0):
        Xp = lambda x: X + (x - xmin) * k     # noqa: E731
        d0 = (fd if side == "field" else pd)["delta0"] * 1e3
        cz = R_mm - d0                         # fingertip centre height above the tool surface
        xs = np.linspace(xmin, xmax, 141)
        arc = [(Xp(x), Y(cz - math.sqrt(R_mm ** 2 - x ** 2))) for x in xs]
        clip = "f1l" if side == "field" else "f1r"
        out.append(f'<g clip-path="url(#{clip})">')
        # tool body and surface
        out.append(f'<rect x="{Xp(xmin):.1f}" y="{y0:.1f}" width="{(xmax - xmin) * k:.1f}" height="{ybot - y0:.1f}" '
                   f'fill="currentColor" opacity="0.08"/>')
        # fingertip body above its arc
        body = [(Xp(xmin), yt - 4)] + arc + [(Xp(xmax), yt - 4)]
        if side == "field":
            out.append(f'<polygon points="{_poly(body)}" fill="currentColor" opacity="0.05"/>')
            a = math.sqrt(R_mm ** 2 - cz ** 2)
            lens_x = np.linspace(-a, a, 81)
            lens = [(Xp(x), Y(cz - math.sqrt(R_mm ** 2 - x ** 2))) for x in lens_x]
            out.append(f'<polygon points="{_poly(lens)}" style="fill:{DRAKE}" opacity="0.55"/>')
            out.append(f'<polyline points="{_poly(arc)}" fill="none" stroke="currentColor" stroke-width="1.4"/>')
        else:
            out.append(f'<polyline points="{_poly(arc)}" fill="none" stroke="currentColor" stroke-width="1.1" '
                       f'stroke-dasharray="5 4" opacity="0.55"/>')
            ang = spacing * 1e3 / (R_mm - RS_mm)
            for j in range(-6, 7):
                th = j * ang
                cx, czs = (R_mm - RS_mm) * math.sin(th), cz - (R_mm - RS_mm) * math.cos(th)
                if not (xmin - 1 < cx < xmax + 1):
                    continue
                out.append(f'<circle cx="{Xp(cx):.1f}" cy="{Y(czs):.1f}" r="{RS_mm * k:.1f}" style="fill:{SPHERE};stroke:{SPHERE}" '
                           f'fill-opacity="0.07" stroke-width="1"/>')
                over = RS_mm - czs
                if over > 0:
                    hw = math.sqrt(max(RS_mm ** 2 - czs ** 2, 0.0))
                    cap = [(Xp(cx + x), Y(czs - math.sqrt(max(RS_mm ** 2 - x ** 2, 0.0)))) for x in np.linspace(-hw, hw, 25)]
                    out.append(f'<polygon points="{_poly(cap)}" style="fill:{SPHERE}" opacity="0.8"/>')
                out.append(f'<circle cx="{Xp(cx):.1f}" cy="{Y(czs):.1f}" r="1.6" fill="currentColor" opacity="0.6"/>')
        out.append(f'<line x1="{Xp(xmin):.1f}" y1="{y0:.1f}" x2="{Xp(xmax):.1f}" y2="{y0:.1f}" stroke="currentColor" stroke-width="1.6"/>')
        out.append('</g>')
        # labels on the section
        title = ("Drake hydroelastic: a pressure field over the overlap" if side == "field"
                 else f"MuJoCo sphere pad ({spacing * 1e3:g} mm): one spring per overlap")
        out.append(f'<text x="{X:.1f}" y="{yt - 30:.1f}" {TXT} font-size="13" font-weight="600">{title}</text>')
        if side == "field":
            out.append(f'<text x="{Xp(xmax) - 6:.1f}" y="{ybot - 7:.1f}" text-anchor="end" {TXT2}>rigid tool (r 12.5 mm)</text>')
        out.append(f'<text x="{Xp(xmin) + 6:.1f}" y="{yt + 12:.1f}" {TXT2}>fingertip (R 10.55 mm)</text>')
        if side == "field":
            out.append(f'<line x1="{Xp(0):.1f}" y1="{y0:.1f}" x2="{Xp(0):.1f}" y2="{Y(-d0):.1f}" stroke="currentColor" '
                       f'stroke-width="1" marker-start="url(#f1a)" marker-end="url(#f1a)"/>')
            out.append(f'<text x="{Xp(0) + 8:.1f}" y="{Y(-d0 * 0.5) + 4:.1f}" {TXT} font-weight="600">&#948;&#8320; = {d0:.2f} mm</text>')
            a = math.sqrt(R_mm ** 2 - cz ** 2)
            out.append(f'<line x1="{Xp(-a):.1f}" y1="{y0 - 10:.1f}" x2="{Xp(a):.1f}" y2="{y0 - 10:.1f}" stroke="currentColor" '
                       f'stroke-width="1" marker-start="url(#f1a)" marker-end="url(#f1a)" opacity="0.7"/>')
            out.append(f'<text x="{Xp(0):.1f}" y="{y0 - 15:.1f}" text-anchor="middle" {TXT2}>2a = {2 * a:.1f} mm</text>')
            out.append(f'<text x="{Xp(-a) + 4:.1f}" y="{ybot - 7:.1f}" {TXT}>overlap: p = E&#183;d / R</text>')
        else:
            out.append(f'<text x="{Xp(xmin) + 6:.1f}" y="{ybot - 7:.1f}" {TXT}>each overlap &#948; is a spring: f = k&#948;, k = {pd["K"]:.0f} N/m</text>')
        # pressure strip
        out.append(f'<line x1="{Xp(xmin):.1f}" y1="{ps1:.1f}" x2="{Xp(xmax):.1f}" y2="{ps1:.1f}" stroke="currentColor" opacity="0.5"/>')
        for t in (0.0, 0.1, 0.2, 0.3):
            out.append(f'<line x1="{Xp(xmin):.1f}" y1="{P(t):.1f}" x2="{Xp(xmax):.1f}" y2="{P(t):.1f}" stroke="currentColor" '
                       f'opacity="0.10"/>')
            out.append(f'<text x="{Xp(xmin) - 5:.1f}" y="{P(t) + 4:.1f}" text-anchor="end" {TXT2} font-size="10.5">{t:.1f}</text>')
        a_f = math.sqrt(R_mm ** 2 - (R_mm - fd["delta0"] * 1e3) ** 2)
        prof = [(Xp(x), P(E * (math.sqrt(R_mm ** 2 - x ** 2) - (R_mm - fd["delta0"] * 1e3)) / R_mm * 1e-6))
                for x in np.linspace(-a_f, a_f, 81)]
        if side == "field":
            out.append(f'<polygon points="{_poly([(Xp(-a_f), P(0))] + prof + [(Xp(a_f), P(0))])}" style="fill:{DRAKE}" opacity="0.35"/>')
            out.append(f'<polyline points="{_poly(prof)}" fill="none" style="stroke:{DRAKE}" stroke-width="2"/>')
            out.append(f'<text x="{Xp(xmin):.1f}" y="{ps1 + 30:.1f}" {TXT2}>pressure on the tool surface, MPa '
                       f'(peak {fd["pmax"] / 1e6:.2f}, friction arm {fd["arm"] * 1e3:.2f} mm)</text>')
        else:
            out.append(f'<polyline points="{_poly(prof)}" fill="none" style="stroke:{DRAKE}" stroke-width="1.5" '
                       f'stroke-dasharray="5 4"/>')
            ang = spacing * 1e3 / (R_mm - RS_mm)
            for j in range(-6, 7):
                th = j * ang
                cx, czs = (R_mm - RS_mm) * math.sin(th), cz - (R_mm - RS_mm) * math.cos(th)
                over = RS_mm - czs
                if over <= 0 or not (xmin < cx < xmax):
                    continue
                peq = pd["K"] * over * 1e-3 / pd["A_s"] * 1e-6
                out.append(f'<rect x="{Xp(cx) - 4:.1f}" y="{P(peq):.1f}" width="8" height="{P(0) - P(peq):.1f}" rx="2" '
                           f'style="fill:{SPHERE}"/>')
            out.append(f'<text x="{Xp(xmin):.1f}" y="{ps1 + 30:.1f}" {TXT2}>spring force / its share of pad area, MPa '
                       f'(dashed: the field)</text>')
    out.append('</svg>')
    return "\n".join(out)


def _contours(fd, levels=(0.0, 0.25, 0.5, 0.75)):
    """Semi-axes (along the tool axis, around it) of the field's pressure contours, from its centre lines."""
    p, x = fd["p"], fd["x"]
    i0 = len(x) // 2
    out = []
    for lv in levels:
        thr = lv * fd["pmax"]
        ax = x[np.where(p[:, i0] > thr)[0]]
        ay = x[np.where(p[i0, :] > thr)[0]]
        if len(ax) and len(ay):
            out.append((lv, (ax.max() - ax.min()) / 2 * 1e3, (ay.max() - ay.min()) / 2 * 1e3))
    return out


def svg_patch(res_hi, res_lo):
    """Figure 2: the contact area seen from the fingertip at two pinch forces: the field's pressure contours and,
    for each pad spacing, the sphere centres with each touching sphere's force as the area of its disc."""
    cols = ["field", 0.002, 0.001, 0.0005]
    cw, chh, k = 246.0, 196.0, 30.0          # cell size, px per mm
    x_l, y_t = 112.0, 44.0
    W = int(x_l + 4 * cw + 6)
    H_ = int(y_t + 2 * chh + 50)
    out = [f'<svg viewBox="0 0 {W} {H_}" role="img" aria-label="Contact area of the thumb pad on the tool at '
           f'{res_hi["F"]:g} N and {res_lo["F"]:g} N: Drake pressure contours against the sphere centres of the 2, 1 and '
           f'0.5 mm pads; at the low force the 2 mm pad touches with two spheres.">']
    heads = {"field": "Drake field", 0.002: "2 mm pad", 0.001: "1 mm pad", 0.0005: "0.5 mm pad"}
    for j, c in enumerate(cols):
        sub = "pressure contours" if c == "field" else f"{res_hi[c]['n']} spheres"
        out.append(f'<text x="{x_l + j * cw + cw / 2:.1f}" y="{y_t - 22:.1f}" text-anchor="middle" {TXT} font-size="13" '
                   f'font-weight="600">{heads[c]}</text>')
        out.append(f'<text x="{x_l + j * cw + cw / 2:.1f}" y="{y_t - 7:.1f}" text-anchor="middle" {TXT2}>{sub}</text>')
    for i, res in enumerate((res_hi, res_lo)):
        fd = res["field"]
        cont = _contours(fd)
        yc0 = y_t + i * chh
        label = f"{res['F']:g} N"
        out.append(f'<text x="{x_l - 14:.1f}" y="{yc0 + chh / 2 - 4:.1f}" text-anchor="end" {TXT} font-size="13" font-weight="600">{label}</text>')
        out.append(f'<text x="{x_l - 14:.1f}" y="{yc0 + chh / 2 + 12:.1f}" text-anchor="end" {TXT2}>'
                   f'{"hold" if i == 0 else "end of brake"}</text>')
        for j, c in enumerate(cols):
            cx, cy = x_l + j * cw + cw / 2, yc0 + chh / 2 - 8
            out.append(f'<rect x="{x_l + j * cw + 6:.1f}" y="{yc0 + 4:.1f}" width="{cw - 12:.1f}" height="{chh - 34:.1f}" rx="8" '
                       f'fill="currentColor" opacity="0.035"/>')
            if c == "field":
                for lv, ax, ay in cont:
                    out.append(f'<ellipse cx="{cx:.1f}" cy="{cy:.1f}" rx="{ax * k:.1f}" ry="{ay * k:.1f}" style="fill:{DRAKE}" '
                               f'opacity="{0.16 if lv > 0 else 0.12}"/>')
                lv, ax, ay = cont[0]
                out.append(f'<ellipse cx="{cx:.1f}" cy="{cy:.1f}" rx="{ax * k:.1f}" ry="{ay * k:.1f}" fill="none" '
                           f'style="stroke:{DRAKE}" stroke-width="1.5"/>')
                arm, foot = fd["arm"], f"{fd['area'] * 1e6:.1f} mm&#178; &#183; arm {fd['arm'] * 1e3:.2f} mm"
            else:
                pdq = res[c]
                lv, ax, ay = cont[0]
                out.append(f'<ellipse cx="{cx:.1f}" cy="{cy:.1f}" rx="{ax * k:.1f}" ry="{ay * k:.1f}" fill="none" '
                           f'style="stroke:{DRAKE}" stroke-width="1" stroke-dasharray="4 3" opacity="0.8"/>')
                cc, f = pdq["centres"], pdq["f"]
                xs = cc[:, 0] * 1e3
                ss = RC * np.arctan2(cc[:, 1], cc[:, 2]) * 1e3
                lim_x, lim_y = (cw / 2 - 12) / k, (chh / 2 - 22) / k
                for x_, s_, f_ in sorted(zip(xs, ss, f), key=lambda t: t[2]):
                    if abs(x_) > lim_x or abs(s_) > lim_y:
                        continue
                    if f_ > 0:
                        r_ = 11.0 * math.sqrt(f_)
                        out.append(f'<circle cx="{cx + x_ * k:.1f}" cy="{cy + s_ * k:.1f}" r="{r_:.1f}" style="fill:{SPHERE};'
                                   f'stroke:var(--card)" stroke-width="1.2"><title>{f_:.3f} N</title></circle>')
                    else:
                        out.append(f'<circle cx="{cx + x_ * k:.1f}" cy="{cy + s_ * k:.1f}" r="1.8" fill="none" '
                                   f'stroke="currentColor" opacity="0.3"/>')
                arm = pdq["arm"]
                dev = (pdq["arm"] / fd["arm"] - 1) * 100
                foot = f"{pdq['n_in']} touching &#183; arm {pdq['arm'] * 1e3:.2f} mm ({dev:+.0f} %)"
            # friction arm as a dashed circle
            out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{arm * 1e3 * k:.1f}" fill="none" stroke="currentColor" '
                       f'stroke-width="1" stroke-dasharray="2 3" opacity="0.75"/>')
            out.append(f'<text x="{cx:.1f}" y="{yc0 + chh - 14:.1f}" text-anchor="middle" {TXT2}>{foot}</text>')
    # scale bar and key, two lines
    yk = y_t + 2 * chh + 14
    out.append(f'<line x1="{x_l + 8:.1f}" y1="{yk:.1f}" x2="{x_l + 8 + k:.1f}" y2="{yk:.1f}" stroke="currentColor" stroke-width="2"/>')
    out.append(f'<text x="{x_l + 16 + k:.1f}" y="{yk + 4:.1f}" {TXT2}>1 mm</text>')
    out.append(f'<circle cx="{x_l + 140:.1f}" cy="{yk:.1f}" r="{11 * math.sqrt(0.25):.1f}" style="fill:{SPHERE}"/>')
    out.append(f'<text x="{x_l + 152:.1f}" y="{yk + 4:.1f}" {TXT2}>touching sphere: disc area &#8733; its force (0.25 N shown)</text>')
    out.append(f'<circle cx="{x_l + 600:.1f}" cy="{yk:.1f}" r="2" fill="none" stroke="currentColor" opacity="0.35"/>')
    out.append(f'<text x="{x_l + 610:.1f}" y="{yk + 4:.1f}" {TXT2}>sphere not touching</text>')
    yk += 22
    out.append(f'<circle cx="{x_l + 16:.1f}" cy="{yk:.1f}" r="8" fill="none" stroke="currentColor" stroke-width="1" '
               f'stroke-dasharray="2 3"/>')
    out.append(f'<text x="{x_l + 32:.1f}" y="{yk + 4:.1f}" {TXT2}>friction arm (torque about the normal = &#956;&#183;F&#183;arm)</text>')
    out.append(f'<ellipse cx="{x_l + 470:.1f}" cy="{yk:.1f}" rx="12" ry="7" fill="none" style="stroke:{DRAKE}" '
               f'stroke-dasharray="4 3"/>')
    out.append(f'<text x="{x_l + 488:.1f}" y="{yk + 4:.1f}" {TXT2}>edge of the field&#8217;s contact area</text>')
    out.append('</svg>')
    return "\n".join(out)


FAMILY = {"drake": DRAKE, "spheres": SPHERE, "point": C4}


def svg_bench(rows):
    """Figure 3: physics time per simulated second (dot = median over repeats, bar = range), log axis, one row per
    model, coloured by family; the controller's share is the open tick."""
    rows = sorted(rows, key=lambda r: r["phys_med"])
    W, xl, xr = 1040, 300.0, 960.0
    lo, hi = 10.0, 5000.0
    X = lambda v: xl + (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * (xr - xl)  # noqa: E731
    rh, yt = 30.0, 58.0
    H_ = int(yt + rh * len(rows) + 54)
    out = [f'<svg viewBox="0 0 {W} {H_}" role="img" aria-label="Physics time per simulated second for each contact '
           f'model on one core, log scale; the sphere pads sit between MuJoCo point contact and Drake hydroelastic.">']
    # legend
    lx = xl
    for fam, name in (("point", "MuJoCo, one contact point"), ("spheres", "MuJoCo, sphere pads"), ("drake", "Drake")):
        out.append(f'<circle cx="{lx + 6:.1f}" cy="18" r="6" style="fill:{FAMILY[fam]}"/>')
        out.append(f'<text x="{lx + 18:.1f}" y="22" {TXT}>{name}</text>')
        lx += 22 + 8.2 * len(name) + 30
    out.append(f'<line x1="{lx:.1f}" y1="18" x2="{lx + 14:.1f}" y2="18" stroke="currentColor" stroke-width="2"/>')
    out.append(f'<text x="{lx + 20:.1f}" y="22" {TXT}>controller</text>')
    yb = yt + rh * len(rows)
    for v in (10, 30, 100, 300, 1000, 3000):
        out.append(f'<line x1="{X(v):.1f}" y1="{yt - 8:.1f}" x2="{X(v):.1f}" y2="{yb:.1f}" stroke="currentColor" opacity="0.10"/>')
        out.append(f'<text x="{X(v):.1f}" y="{yb + 18:.1f}" text-anchor="middle" {TXT2}>{v:g}</text>')
    out.append(f'<line x1="{X(1000):.1f}" y1="{yt - 8:.1f}" x2="{X(1000):.1f}" y2="{yb:.1f}" stroke="currentColor" '
               f'stroke-dasharray="4 3" opacity="0.6"/>')
    out.append(f'<text x="{X(1000) + 5:.1f}" y="{yt - 12:.1f}" {TXT2}>real time</text>')
    out.append(f'<text x="{(xl + xr) / 2:.1f}" y="{yb + 40:.1f}" text-anchor="middle" {TXT2}>ms of one core per simulated second '
               f'(physics only; log scale)</text>')
    for i, r in enumerate(rows):
        y = yt + rh * i + rh / 2
        col = FAMILY[r["family"]]
        out.append(f'<text x="{xl - 14:.1f}" y="{y + 4:.1f}" text-anchor="end" {TXT}>{r["name"]}</text>')
        out.append(f'<g><title>{r["name"]}: physics {r["phys_med"]:.0f} ms per simulated s (range {r["phys_min"]:.0f}&#8211;'
                   f'{r["phys_max"]:.0f}, {r["reps"]} runs), controller {r["ctrl_med"]:.0f} ms/s</title>')
        out.append(f'<line x1="{X(r["phys_min"]):.1f}" y1="{y:.1f}" x2="{X(max(r["phys_max"], r["phys_min"] * 1.0001)):.1f}" '
                   f'y2="{y:.1f}" style="stroke:{col}" stroke-width="3" stroke-linecap="round" opacity="0.6"/>')
        out.append(f'<line x1="{X(r["ctrl_med"]):.1f}" y1="{y - 7:.1f}" x2="{X(r["ctrl_med"]):.1f}" y2="{y + 7:.1f}" '
                   f'stroke="currentColor" stroke-width="2" opacity="0.55"/>')
        out.append(f'<circle cx="{X(r["phys_med"]):.1f}" cy="{y:.1f}" r="6" style="fill:{col};stroke:var(--card)" stroke-width="2"/>')
        out.append(f'<text x="{X(r["phys_med"]) + 12:.1f}" y="{y + 4:.1f}" {TXT}>{r["phys_med"]:.0f}</text></g>')
    out.append('</svg>')
    return "\n".join(out)
