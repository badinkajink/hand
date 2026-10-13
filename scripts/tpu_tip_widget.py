#!/usr/bin/env python3
"""Calculator for the printed fingertip in the page (2026-10-13): design of the magnet tip and bench calibration, live in
the browser.

Owner, 2026-10-13: "the tools you generated are cool. is it possible to create a dynamic viewer / browser-receiving input
so that i can play with it in the docpage html?"

The script in the page repeats, on the finite-element rows of magnet_runs.jsonl:
  scripts/tpu_tip_design.py       pocket, pause layer, the whole tip's STL (built in the browser and downloaded)
  scripts/tpu_tip_print_plan.py   point-dipole field at the MLX90393, signal per newton and force noise
  scripts/tpu_tip_calibrate.py    fit of the solid modulus, pad modulus and friction to bench measurements
Depths between the runs at 1.5, 3 and 5 mm are interpolated linearly; the 3.18 mm magnet has one run, at 3 mm, and other
depths add the 4.76 mm magnet's change with depth to it. The page needs no network and no data file.

    ~/miniconda3/envs/femtip/bin/python scripts/tpu_tip_widget.py check    # browser against Python, headless Chrome
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import tpu_tip_design as TD  # noqa: E402
import tpu_tip_print_plan as PP  # noqa: E402

RECIPES = [("w95_20", "95A, gyroid 20 %"), ("w85_50", "85A, gyroid 50 %"), ("w85_30", "85A, gyroid 30 %"),
           ("w95_10", "95A, gyroid 10 %")]
BASE_SCENE = "results/phase1/real_v1/20261011-sv1_u0308_b050_servo_tip"


def data(X=None):
    """Rows, geometry, sensor and pads' arm table, as one JSON object for the page."""
    rows = {}
    for (rec, mag, dep), r in sorted(PP.rows().items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2] or 0)):
        pf = {k: dict(d=v["delta_um"], arm=v["arm_mm"], t=v.get("magnet_t_um"), w=v.get("magnet_w_mrad"))
              for k, v in r["per_force"].items()}
        rows.setdefault(rec, {}).setdefault(mag, []).append(dict(
            depth=dep, E_s=r["E_s"], E_best=r["E_best_MPa"], pf=pf,
            szt=r.get("shear_z_magnet_t_um_per_N"), szw=r.get("shear_z_magnet_w_mrad_per_N"),
            syt=r.get("shear_y_magnet_t_um_per_N"), syw=r.get("shear_y_magnet_w_mrad_per_N")))
    pads = []
    if X is not None:
        for E in np.exp(np.linspace(math.log(2.5), math.log(77.0), 61)):
            a = X.pads_arm(float(E), 3.0)
            if a:
                pads.append([round(float(E), 4), round(float(a), 5)])
    geom = dict(X_FACE=TD.X_FACE, X_BACK=TD.X_BACK, Y_HALF=TD.Y_HALF, Z_END=TD.Z_END, Z_C=TD.Z_C, R_SIDE=TD.R_SIDE,
                R_END=TD.R_END)
    return dict(geom=geom, magnets={k: list(v) for k, v in TD.MAGNETS.items()}, br=TD.BR_N52, noise=PP.NOISE_UT,
                rate=PP.RATE_HZ, range_mT=PP.B_RANGE_MT, layer=TD.LAYER, rows=rows, pads=pads,
                recipes=[list(r) for r in RECIPES], base_scene=BASE_SCENE,
                material={"w95_20": "TPU 95A", "w95_10": "TPU 95A", "w85_50": "TPU 85A", "w85_30": "TPU 85A"},
                infill={"w95_20": 20, "w95_10": 10, "w85_50": 50, "w85_30": 30})


CSS = """
.tipcalc{margin:10px 0 18px}
.tipcalc .tc-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px 22px;margin:6px 0 12px}
.tipcalc label{display:block;font-size:16px}
.tipcalc label span.tc-l{display:block;margin-bottom:2px}
.tipcalc select,.tipcalc input[type=number],.tipcalc textarea{font:16px "Times New Roman",Times,serif;max-width:100%}
.tipcalc input[type=range]{width:100%;max-width:330px;vertical-align:middle}
.tipcalc output{font-variant-numeric:tabular-nums;margin-left:6px}
.tipcalc textarea{width:100%;box-sizing:border-box;height:5.2em}
.tipcalc table.tc-out td:first-child{width:44%}
.tipcalc .tc-note{font-size:15px;color:#333;margin:4px 0 8px}
.tipcalc .tc-warn{color:#b00020}
.tipcalc button{font:16px "Times New Roman",Times,serif;margin:4px 10px 4px 0;padding:2px 10px}
.tipcalc pre.tc-cmd{white-space:pre-wrap;word-break:break-all;font-size:14px;border:1px solid #000;padding:6px;margin:6px 0}
.tipcalc figure{margin:12px 0}
.tipcalc .tc-svg svg{width:100%;height:auto;display:block}
@media (max-width:640px){.tipcalc .tc-grid{grid-template-columns:minmax(0,1fr)}}
"""

# Plain ES5. The page's builder renders TeX between backslash-parenthesis and backslash-bracket pairs and fails on
# double braces, so the script contains neither and parses numbers without regular expressions.
JS = r"""
(function () {
"use strict";
var D = JSON.parse(document.getElementById("tc-data").textContent);
var G = D.geom, HS = " ";

function el(id) { return document.getElementById(id); }
function mix(a, b, t) {
  if (a === null || a === undefined) return a;
  if (typeof a === "number") return a + (b - a) * t;
  if (Array.isArray(a)) return a.map(function (v, i) { return mix(v, b[i], t); });
  var o = {}; for (var k in a) o[k] = mix(a[k], b[k], t); return o;
}
function shift(a, b, c) {
  if (a === null || a === undefined) return a;
  if (typeof a === "number") return a + (b - c);
  if (Array.isArray(a)) return a.map(function (v, i) { return shift(v, b[i], c[i]); });
  var o = {}; for (var k in a) o[k] = shift(a[k], b[k], c[k]); return o;
}

// Finite-element row of a print, magnet and depth. The 4.76 mm magnet: linear in depth between the runs. The 3.18 mm
// magnet: its run at 3 mm plus the 4.76 mm magnet's change from 3 mm to the depth.
function rowAt(rec, mag, dep) {
  var R = D.rows[rec];
  if (mag === "none") return { r: R.none[0], kind: "run" };
  var L = R["d4.76x1.59"];
  function at(d) {
    d = Math.min(Math.max(d, L[0].depth), L[L.length - 1].depth);
    for (var i = 0; i < L.length - 1; i++) {
      if (d <= L[i + 1].depth + 1e-9) return mix(L[i], L[i + 1], (d - L[i].depth) / (L[i + 1].depth - L[i].depth));
    }
    return L[L.length - 1];
  }
  var onRun = function (list) { return list.some(function (x) { return Math.abs(x.depth - dep) < 1e-6; }); };
  if (mag === "d4.76x1.59") return { r: at(dep), kind: onRun(L) ? "run" : "interpolated" };
  var r3 = R[mag][0];
  return { r: shift(r3, at(dep), at(r3.depth)), kind: onRun(R[mag]) ? "run" : "shifted" };
}

function dipole(ps, pm, m) {
  var r = [ps[0] - pm[0], ps[1] - pm[1], ps[2] - pm[2]];
  var d = Math.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
  var h = [r[0] / d, r[1] / d, r[2] / d], hm = h[0] * m[0] + h[1] * m[1] + h[2] * m[2], c = 1e-7 / (d * d * d);
  return [c * (3 * h[0] * hm - m[0]), c * (3 * h[1] * hm - m[1]), c * (3 * h[2] * hm - m[2])];
}
function cross(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
function norm(v) { return Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); }
function moment(dm, t, br) { return br * Math.PI * Math.pow(dm / 2e3, 2) * (t / 1e3) / (4e-7 * Math.PI); }
function pocket(dm, t, depth, cr, ct) {
  var hi = G.X_FACE - depth, lo = hi - t;
  return { pk: { r: dm / 2 + cr, lo: lo - ct / 2, hi: hi + ct / 2 }, mg: { r: dm / 2, lo: lo, hi: hi } };
}
function pauseAt(pk, layer) {
  var top = pk.hi - G.X_BACK, bot = pk.lo - G.X_BACK, n = Math.ceil(top / layer + 0.5 - 1e-9);
  return { n: n, z: n * layer, top: top, bot: bot };
}

// Field at the chip (sensing point at x = X_BACK - gap under the contact point), its change per newton for the three
// loads, and the force whose signal equals one sample's noise. k: solid modulus over the run's; rate: output rate (Hz),
// the per-axis noise scaling as the square root of the rate (averaged 100 Hz samples).
function field(r, dm, t, depth, gap, br, k, rate) {
  var g = pocket(dm, t, depth, 0.1, 0.1).mg;
  var pm = [0.5 * (g.lo + g.hi) * 1e-3, 0, G.Z_C * 1e-3], ps = [(G.X_BACK - gap) * 1e-3, 0, G.Z_C * 1e-3];
  var m0 = [moment(dm, t, br), 0, 0], B0 = dipole(ps, pm, m0), f = Math.sqrt(rate / D.rate);
  var sig = [D.noise.z * f, D.noise.xy * f, D.noise.xy * f];
  var out = { B0: norm(B0) * 1e3 };
  var loads = { normal: [r.pf["1"].t, r.pf["1"].w], across: [r.syt, r.syw], along: [r.szt, r.szw] };
  for (var key in loads) {
    var tt = loads[key][0], ww = loads[key][1];
    var tm = [tt[0] * 1e-6 / k, tt[1] * 1e-6 / k, tt[2] * 1e-6 / k];
    var wm = cross([ww[0] * 1e-3 / k, ww[1] * 1e-3 / k, ww[2] * 1e-3 / k], m0);
    var B1 = dipole(ps, [pm[0] + tm[0], pm[1] + tm[1], pm[2] + tm[2]], [m0[0] + wm[0], m0[1] + wm[1], m0[2] + wm[2]]);
    var S = [(B1[0] - B0[0]) * 1e6, (B1[1] - B0[1]) * 1e6, (B1[2] - B0[2]) * 1e6];
    var q = Math.pow(S[0] / sig[0], 2) + Math.pow(S[1] / sig[1], 2) + Math.pow(S[2] / sig[2], 2);
    out[key] = { S: norm(S), sigF: q > 0 ? 1 / Math.sqrt(q) : Infinity, travel: norm(tm) * 1e6 };
  }
  return out;
}

function padsArm(E) {
  var P = D.pads;
  if (!P.length || E < P[0][0] || E > P[P.length - 1][0]) return null;
  for (var i = 0; i < P.length - 1; i++) {
    if (E <= P[i + 1][0]) {
      var u = (Math.log(E) - Math.log(P[i][0])) / (Math.log(P[i + 1][0]) - Math.log(P[i][0]));
      return P[i][1] + (P[i + 1][1] - P[i][1]) * u;
    }
  }
  return P[P.length - 1][1];
}

// Finite-element approach (um) at force F (N): log-log between the runs at 0.5, 1 and 3 N, extended along the end
// segments outside them.
function approachAt(r, F) {
  var xs = [0.5, 1, 3].map(Math.log), ys = [r.pf["0.5"].d, r.pf["1"].d, r.pf["3"].d].map(Math.log), x = Math.log(F);
  var i = x <= xs[1] ? 0 : 1;
  return Math.exp(ys[i] + (ys[i + 1] - ys[i]) * (x - xs[i]) / (xs[i + 1] - xs[i]));
}

function fitBench(r, hasMagnet, dm, t, depth, gap, br, indent, fieldPts, pull, twist) {
  var E0 = r.E_s, est = [];
  indent.forEach(function (p) { est.push({ kind: "Indentation at " + fmt(p[0], 2) + HS + "N", E: E0 * approachAt(r, p[0]) / p[1] }); });
  if (fieldPts.length && hasMagnet) {
    var S = field(r, dm, t, depth, gap, br, 1, D.rate).normal.S;
    fieldPts.forEach(function (p) { est.push({ kind: "Field at " + fmt(p[0], 2) + HS + "N", E: E0 * S * p[0] / p[1] }); });
  }
  var out = { est: est, E0: E0 };
  if (est.length) {
    var s = 0; est.forEach(function (e) { s += Math.log(e.E); });
    out.E_s = Math.exp(s / est.length);
    out.E_star = r.E_best * out.E_s / E0;
  }
  if (pull) out.mu = pull[1] / (2 * pull[0]);
  if (twist && out.mu) out.arm = twist[1] / (2 * out.mu * twist[0]);
  return out;
}

function gfmt(x, p) {            // Python's "%.<p>g", exponent with two digits
  var s = Number(x).toPrecision(p);
  if (s.indexOf("e") >= 0) {
    var a = s.split("e"), m = a[0], e = a[1], sign = e.charAt(0) === "-" ? "-" : "+", digits = e.replace("-", "").replace("+", "");
    if (m.indexOf(".") >= 0) { while (m.charAt(m.length - 1) === "0") m = m.slice(0, -1); if (m.charAt(m.length - 1) === ".") m = m.slice(0, -1); }
    return m + "e" + sign + (digits.length < 2 ? "0" + digits : digits);
  }
  if (s.indexOf(".") >= 0) { while (s.charAt(s.length - 1) === "0") s = s.slice(0, -1); if (s.charAt(s.length - 1) === ".") s = s.slice(0, -1); }
  return s;
}
function sceneCommand(Estar, mu) {
  var c = "python scripts/make_pad_morphology_run.py --src " + D.base_scene + " --out results/phase1/real_v1/<run>_tpu2.7pads1_E" +
          Estar.toFixed(1) + " --model pads --E " + gfmt(Estar * 1e6, 4);
  if (mu) c += " --mu " + gfmt(mu, 3);
  return c;
}

// ---------------------------------------------------------------- whole tip as a watertight STL, pocket as a void
function surfaceX(y, z) {
  var dy = Math.min(Math.max(Math.abs(y) - (G.Y_HALF - G.R_SIDE), 0), G.R_SIDE);
  var dz = Math.min(Math.max(z - (G.Z_END - G.R_END), 0), G.R_END);
  var xs = (G.X_FACE - G.R_SIDE) + Math.sqrt(Math.max(G.R_SIDE * G.R_SIDE - dy * dy, 0));
  var xe = (G.X_FACE - G.R_END) + Math.sqrt(Math.max(G.R_END * G.R_END - dz * dz, 0));
  return Math.min(xs, xe, G.X_FACE);
}
function nodes(a, b, r, h, nArc, both) {   // flat part [a, b - r] in steps of about h, then the fillet in nArc angular steps
  var out = [], flatLo = both ? a + r : a, flatHi = b - r, n = Math.max(1, Math.round((flatHi - flatLo) / h)), i;
  if (both) for (i = 0; i < nArc; i++) out.push(flatLo - r * Math.cos(Math.PI / 2 * i / nArc));
  for (i = 0; i <= n; i++) out.push(flatLo + (flatHi - flatLo) * i / n);
  for (i = 1; i <= nArc; i++) out.push(flatHi + r * Math.sin(Math.PI / 2 * i / nArc));
  return out;
}
function tipTriangles(pk) {
  var T = [], ys = nodes(-G.Y_HALF, G.Y_HALF, G.R_SIDE, 0.2, 24, true), zs = nodes(0, G.Z_END, G.R_END, 0.2, 24, false), i, j;
  function tri(a, b, c, dir) {
    var n = cross([b[0] - a[0], b[1] - a[1], b[2] - a[2]], [c[0] - a[0], c[1] - a[1], c[2] - a[2]]);
    if (n[0] * dir[0] + n[1] * dir[1] + n[2] * dir[2] < 0) { var s = b; b = c; c = s; n = [-n[0], -n[1], -n[2]]; }
    var l = norm(n) || 1; T.push([[n[0] / l, n[1] / l, n[2] / l], a, b, c]);
  }
  function F(i, j) { return [surfaceX(ys[i], zs[j]), ys[i], zs[j]]; }
  function B(i, j) { return [G.X_BACK, ys[i], zs[j]]; }
  var ny = ys.length - 1, nz = zs.length - 1;
  for (i = 0; i < ny; i++) for (j = 0; j < nz; j++) {
    tri(F(i, j), F(i + 1, j), F(i + 1, j + 1), [1, 0, 0]); tri(F(i, j), F(i + 1, j + 1), F(i, j + 1), [1, 0, 0]);
    tri(B(i, j), B(i + 1, j), B(i + 1, j + 1), [-1, 0, 0]); tri(B(i, j), B(i + 1, j + 1), B(i, j + 1), [-1, 0, 0]);
  }
  for (j = 0; j < nz; j++) {
    tri(B(0, j), F(0, j), F(0, j + 1), [0, -1, 0]); tri(B(0, j), F(0, j + 1), B(0, j + 1), [0, -1, 0]);
    tri(B(ny, j), F(ny, j), F(ny, j + 1), [0, 1, 0]); tri(B(ny, j), F(ny, j + 1), B(ny, j + 1), [0, 1, 0]);
  }
  for (i = 0; i < ny; i++) {
    tri(B(i, 0), F(i, 0), F(i + 1, 0), [0, 0, -1]); tri(B(i, 0), F(i + 1, 0), B(i + 1, 0), [0, 0, -1]);
    tri(B(i, nz), F(i, nz), F(i + 1, nz), [0, 0, 1]); tri(B(i, nz), F(i + 1, nz), B(i + 1, nz), [0, 0, 1]);
  }
  if (pk) {
    var N = 96, lo = [pk.lo, 0, G.Z_C], hi = [pk.hi, 0, G.Z_C];
    for (i = 0; i < N; i++) {
      var a0 = 2 * Math.PI * i / N, a1 = 2 * Math.PI * (i + 1) / N, am = 0.5 * (a0 + a1);
      var p0 = [0, pk.r * Math.cos(a0), G.Z_C + pk.r * Math.sin(a0)], p1 = [0, pk.r * Math.cos(a1), G.Z_C + pk.r * Math.sin(a1)];
      var l0 = [pk.lo, p0[1], p0[2]], l1 = [pk.lo, p1[1], p1[2]], h0 = [pk.hi, p0[1], p0[2]], h1 = [pk.hi, p1[1], p1[2]];
      var inward = [0, -Math.cos(am), -Math.sin(am)];   // the solid's outward normal points into the void
      tri(l0, l1, h1, inward); tri(l0, h1, h0, inward);
      tri(lo, l0, l1, [1, 0, 0]); tri(hi, h0, h1, [-1, 0, 0]);
    }
  }
  return T;
}
function stlBlob(T, title) {
  var buf = new ArrayBuffer(84 + 50 * T.length), dv = new DataView(buf), o = 84, i, k;
  for (i = 0; i < Math.min(title.length, 80); i++) dv.setUint8(i, title.charCodeAt(i) & 127);
  dv.setUint32(80, T.length, true);
  T.forEach(function (t) {
    for (k = 0; k < 4; k++) { dv.setFloat32(o, t[k][0], true); dv.setFloat32(o + 4, t[k][1], true); dv.setFloat32(o + 8, t[k][2], true); o += 12; }
    dv.setUint16(o, 0, true); o += 2;
  });
  return new Blob([buf], { type: "model/stl" });
}
function save(blob, name) {
  var a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = name;
  document.body.appendChild(a); a.click();
  setTimeout(function () { URL.revokeObjectURL(a.href); a.remove(); }, 2000);
}

// ---------------------------------------------------------------- formatting and drawing
function fmt(x, nd) { return (x === null || x === undefined || !isFinite(x)) ? "&#8211;" : Number(x).toFixed(nd); }
function f3(x) { if (!isFinite(x)) return "&#8211;"; var a = Math.abs(x); return a >= 10 ? x.toFixed(1) : a >= 1 ? x.toFixed(2) : x.toFixed(3); }
function pct(x) { return (x >= 0 ? "+" : "&#8722;") + Math.abs(x).toFixed(0) + HS + "%"; }
function esc(s) { return String(s).split("&").join("&amp;").split("<").join("&lt;").split(">").join("&gt;"); }
function svgOpen(w, h, label) { return '<svg viewBox="0 0 ' + w + " " + h + '" role="img" aria-label="' + label + '" style="font:15px Times New Roman,Times,serif">'; }
function txt(x, y, s, anchor, style) { return '<text x="' + x.toFixed(1) + '" y="' + y.toFixed(1) + '"' + (anchor ? ' text-anchor="' + anchor + '"' : "") + ' style="fill:var(--ink);' + (style || "") + '">' + s + "</text>"; }
function line(x1, y1, x2, y2, style) { return '<line x1="' + x1.toFixed(1) + '" y1="' + y1.toFixed(1) + '" x2="' + x2.toFixed(1) + '" y2="' + y2.toFixed(1) + '" style="' + style + '"/>'; }
function rect(x, y, w, h, style) { return '<rect x="' + x.toFixed(1) + '" y="' + y.toFixed(1) + '" width="' + w.toFixed(1) + '" height="' + h.toFixed(1) + '" style="' + style + '"/>'; }

function drawSection(s) {
  var W = 820, H = 450, k = 15, z0 = 200, xb = 330;            // 15 px per mm; z to the right, print height upward
  function X(x) { return xb - (x - G.X_BACK) * k; }
  function Z(z) { return z0 + z * k; }
  var o = [svgOpen(W, H, "Section of the tip through the contact point in the print's orientation")];
  // screwdriver shaft, radius 12.5 mm, touching the face at the contact point
  var R = 12.5 * k, cx = Z(G.Z_C), cy = X(G.X_FACE) - R, sa = Math.sin(0.6), ca = Math.cos(0.6);
  o.push('<path d="M' + (cx - R * sa).toFixed(1) + "," + (cy + R * ca).toFixed(1) + " A" + R + "," + R + " 0 0 0 " +
         (cx + R * sa).toFixed(1) + "," + (cy + R * ca).toFixed(1) + '" style="fill:none;stroke:var(--ink3);stroke-width:1.5"/>');
  o.push(txt(cx + R * sa + 8, cy + R * ca + 4, "Screwdriver shaft, radius 12.5" + HS + "mm", "start", "fill:var(--ink3)"));
  // tip outline (section y = 0), palmar end edge rounded
  var pts = [[G.X_BACK, 0], [G.X_FACE, 0], [G.X_FACE, G.Z_END - G.R_END]], i;
  for (i = 1; i <= 24; i++) { var a = Math.PI / 2 * i / 24; pts.push([G.X_FACE - G.R_END + G.R_END * Math.cos(a), G.Z_END - G.R_END + G.R_END * Math.sin(a)]); }
  pts.push([G.X_BACK, G.Z_END]);
  o.push('<path d="M' + pts.map(function (p) { return Z(p[1]).toFixed(1) + "," + X(p[0]).toFixed(1); }).join(" L") +
         ' Z" style="fill:rgba(42,120,214,0.10);stroke:var(--ink);stroke-width:1.6"/>');
  if (s.pk) {
    // pause: the print stops at the top of the layer below the first layer over the pocket
    var yp = X(G.X_BACK + (s.pause.n - 1) * s.layer);
    o.push(rect(Z(G.Z_C - s.pk.r), X(s.pk.hi), 2 * s.pk.r * k, (s.pk.hi - s.pk.lo) * k, "fill:#fff;stroke:var(--ink);stroke-dasharray:4 3"));
    o.push(rect(Z(G.Z_C - s.mg.r), X(s.mg.hi), 2 * s.mg.r * k, (s.mg.hi - s.mg.lo) * k, "fill:#555;stroke:#000"));
    o.push(line(Z(-0.6), yp, Z(G.Z_END + 0.6), yp, "stroke:#b00020;stroke-width:1.3;stroke-dasharray:6 4"));
    o.push(txt(Z(G.Z_END + 0.9), yp + 5, "Pause before layer " + s.pause.n + " (" + s.pause.z.toFixed(2) + HS + "mm)", "start", "fill:#b00020"));
    var ym = X(0.5 * (s.mg.lo + s.mg.hi));
    o.push(line(Z(-0.4), ym, Z(G.Z_C - s.mg.r), ym, "stroke:var(--ink);stroke-width:0.8"));
    o.push(txt(Z(-0.7), ym + 5, "Magnet " + s.dm + " &#215; " + s.t + HS + "mm", "end"));
    var xd = Z(G.Z_C - s.mg.r - 1.2);
    o.push(line(xd, X(G.X_FACE), xd, X(s.mg.hi), "stroke:var(--ink);stroke-width:1"));
    o.push(line(xd - 5, X(G.X_FACE), xd + 5, X(G.X_FACE), "stroke:var(--ink)") + line(xd - 5, X(s.mg.hi), xd + 5, X(s.mg.hi), "stroke:var(--ink)"));
    o.push(txt(xd - 8, 0.5 * (X(G.X_FACE) + X(s.mg.hi)) + 5, "Depth " + s.depth.toFixed(1) + HS + "mm", "end"));
  }
  o.push('<circle cx="' + Z(G.Z_C).toFixed(1) + '" cy="' + X(G.X_FACE).toFixed(1) + '" r="3.5" style="fill:#b00020"/>');
  // magnetometer: 3 x 3 x 1 mm package with its sensing point at the chip distance behind the back face
  var xs = G.X_BACK - s.gap;
  o.push(rect(Z(G.Z_C - 1.5), X(xs + 0.5), 3 * k, 1 * k, "fill:#1baf7a;stroke:#000"));
  o.push(txt(Z(G.Z_C + 2.2), X(xs) + 5, "MLX90393, " + s.gap.toFixed(1) + HS + "mm behind the back face", "start"));
  o.push(txt(Z(0) - 8, X(G.X_FACE) + 5, "Palmar face, up", "end"));
  o.push(txt(Z(0) - 8, X(G.X_BACK) + 5, "Back face on the bed", "end"));
  o.push(line(Z(0), X(G.X_BACK) + 44, Z(5), X(G.X_BACK) + 44, "stroke:var(--ink);stroke-width:1.5") + txt(Z(5) + 6, X(G.X_BACK) + 49, "5" + HS + "mm", "start"));
  o.push("</svg>");
  return o.join("");
}

function axisLog(o, x0, y0, w, h, xr, yr, xt, yt, xl, yl) {
  function fx(x) { return x0 + (x - xr[0]) / (xr[1] - xr[0]) * w; }
  function fy(y) { return y0 + h - (Math.log(y) - Math.log(yr[0])) / (Math.log(yr[1]) - Math.log(yr[0])) * h; }
  o.push(rect(x0, y0, w, h, "fill:none;stroke:var(--ink)"));
  xt.forEach(function (x) { o.push(line(fx(x), y0 + h, fx(x), y0 + h + 5, "stroke:var(--ink)") + txt(fx(x), y0 + h + 21, String(x), "middle")); });
  yt.forEach(function (y) { o.push(line(x0 - 5, fy(y), x0, fy(y), "stroke:var(--ink)") + line(x0, fy(y), x0 + w, fy(y), "stroke:#ddd") + txt(x0 - 8, fy(y) + 5, String(y), "end")); });
  o.push(txt(x0 + w / 2, y0 + h + 44, xl, "middle"));
  o.push('<text transform="translate(' + (x0 - 58) + "," + (y0 + h / 2) + ') rotate(-90)" text-anchor="middle" style="fill:var(--ink)">' + yl + "</text>");
  return [fx, fy];
}
function poly(o, pts, fx, fy, style) {
  if (pts.length < 2) return;
  o.push('<path d="M' + pts.map(function (p) { return fx(p[0]).toFixed(1) + "," + fy(p[1]).toFixed(1); }).join(" L") + '" style="fill:none;' + style + '"/>');
}

function drawNoise(s) {
  var W = 820, H = 380, o = [svgOpen(W, H, "Force noise against chip distance")];
  var yr = [0.01, 10], xr = [-4, 6];
  var ax = axisLog(o, 100, 20, 560, 290, xr, yr, [-4, -2, 0, 2, 4, 6], [0.01, 0.1, 1, 10], "Chip distance behind the back face (mm)", "Force noise per sample, 1" + HS + "&#963; (N)");
  var fx = ax[0], fy = ax[1], g, curves = { normal: [], across: [], along: [] }, sat = null;
  for (g = xr[0]; g <= xr[1] + 1e-9; g += 0.1) {
    var f = field(s.r, s.dm, s.t, s.depth, g, s.br, s.k, s.rate);
    if (f.B0 > D.range_mT) sat = g;
    ["normal", "across", "along"].forEach(function (key) { var v = f[key].sigF; if (v >= yr[0] && v <= yr[1]) curves[key].push([g, v]); });
  }
  if (sat !== null) o.push(rect(fx(xr[0]), 20, fx(sat) - fx(xr[0]), 290, "fill:rgba(0,0,0,0.08)") +
                           txt(fx(xr[0]) + 6, 40, "Field over " + D.range_mT + HS + "mT", "start", "fill:var(--ink3)"));
  var st = { normal: "stroke:#2a78d6;stroke-width:2.4", across: "stroke:#1baf7a;stroke-width:2.2;stroke-dasharray:7 4", along: "stroke:#eb6834;stroke-width:2.2;stroke-dasharray:2 4" };
  ["normal", "across", "along"].forEach(function (key) { poly(o, curves[key], fx, fy, st[key]); });
  o.push(line(fx(s.gap), 20, fx(s.gap), 310, "stroke:#b00020;stroke-width:1.2"));
  var lab = { normal: "Normal", across: "Shear across", along: "Shear along" }, ly = 60;
  ["normal", "across", "along"].forEach(function (key) {
    o.push(line(680, ly - 5, 720, ly - 5, st[key]) + txt(728, ly, lab[key], "start")); ly += 26;
  });
  o.push(line(680, ly - 5, 720, ly - 5, "stroke:#b00020;stroke-width:1.2") + txt(728, ly, "Chosen chip", "start"));
  o.push("</svg>");
  return o.join("");
}

function drawFit(r, fit, pts) {
  var W = 820, H = 360, o = [svgOpen(W, H, "Approach against force: finite-element model and bench points")];
  var x0 = 100, y0 = 20, w = 560, h = 280, Fmax = 4;
  pts.forEach(function (p) { Fmax = Math.max(Fmax, p[0] * 1.1); });
  var dmax = 0, F;
  for (F = 0.05; F <= Fmax; F += 0.05) dmax = Math.max(dmax, approachAt(r, F) * (fit.E_s ? Math.max(1, r.E_s / fit.E_s) : 1));
  pts.forEach(function (p) { dmax = Math.max(dmax, p[1]); });
  dmax = Math.ceil(dmax / 50) * 50;
  function fx(x) { return x0 + x / Fmax * w; }
  function fy(y) { return y0 + h - y / dmax * h; }
  o.push(rect(x0, y0, w, h, "fill:none;stroke:var(--ink)"));
  for (var t = 0; t <= Fmax + 1e-9; t += (Fmax > 6 ? 2 : 1)) o.push(line(fx(t), y0 + h, fx(t), y0 + h + 5, "stroke:var(--ink)") + txt(fx(t), y0 + h + 21, String(t), "middle"));
  for (var u = 0; u <= dmax + 1e-9; u += dmax / 5) o.push(line(x0 - 5, fy(u), x0, fy(u), "stroke:var(--ink)") + line(x0, fy(u), x0 + w, fy(u), "stroke:#ddd") + txt(x0 - 8, fy(u) + 5, u.toFixed(0), "end"));
  o.push(txt(x0 + w / 2, y0 + h + 44, "Force on the face (N)", "middle"));
  o.push('<text transform="translate(' + (x0 - 58) + "," + (y0 + h / 2) + ') rotate(-90)" text-anchor="middle" style="fill:var(--ink)">Approach (&#181;m)</text>');
  var nom = [], fitted = [];
  for (F = 0.05; F <= Fmax + 1e-9; F += 0.05) { var d = approachAt(r, F); nom.push([F, d]); if (fit.E_s) fitted.push([F, d * r.E_s / fit.E_s]); }
  poly(o, nom, fx, fy, "stroke:var(--ink3);stroke-width:1.8;stroke-dasharray:6 4");
  poly(o, fitted, fx, fy, "stroke:#2a78d6;stroke-width:2.4");
  pts.forEach(function (p) { o.push('<circle cx="' + fx(p[0]).toFixed(1) + '" cy="' + fy(p[1]).toFixed(1) + '" r="5" style="fill:#b00020"/>'); });
  var lx = x0 + 18, ly = y0 + 26;
  o.push(line(lx, ly - 5, lx + 40, ly - 5, "stroke:var(--ink3);stroke-width:1.8;stroke-dasharray:6 4") + txt(lx + 48, ly, "Model at " + r.E_s + HS + "MPa", "start"));
  if (fit.E_s) o.push(line(lx, ly + 21, lx + 40, ly + 21, "stroke:#2a78d6;stroke-width:2.4") + txt(lx + 48, ly + 26, "Model at the fitted " + fit.E_s.toFixed(1) + HS + "MPa", "start"));
  o.push('<circle cx="' + (lx + 20) + '" cy="' + (ly + 47) + '" r="5" style="fill:#b00020"/>' + txt(lx + 48, ly + 52, "Bench", "start"));
  o.push("</svg>");
  return o.join("");
}

function pairs(text) {
  var out = [];
  String(text || "").split("\n").forEach(function (ln) {
    var c = ln.split(",").join(" ").split(":").join(" ").split(";").join(" ").split("\t").join(" ").split(" ").filter(function (x) { return x.length; });
    if (c.length >= 2 && isFinite(+c[0]) && isFinite(+c[1]) && +c[0] > 0 && +c[1] > 0) out.push([+c[0], +c[1]]);
  });
  return out;
}

// ---------------------------------------------------------------- the page
function state() {
  var rec = el("tc-recipe").value, mag = el("tc-magnet").value, depth = +el("tc-depth").value;
  var x = rowAt(rec, mag, depth), nm = D.rows[rec].none[0];
  var dmt = mag === "none" ? null : D.magnets[mag];
  var s = { rec: rec, mag: mag, depth: depth, r: x.r, kind: x.kind, none: nm, gap: +el("tc-gap").value,
            k: +el("tc-es").value, br: +el("tc-br").value, rate: +el("tc-rate").value, layer: +el("tc-layer").value };
  if (dmt) {
    s.dm = dmt[0]; s.t = dmt[1];
    var p = pocket(s.dm, s.t, depth, 0.1, 0.1); s.pk = p.pk; s.mg = p.mg; s.pause = pauseAt(p.pk, s.layer);
    s.f = field(s.r, s.dm, s.t, depth, s.gap, s.br, s.k, s.rate);
  }
  return s;
}

function update() {
  var s = state(), E0 = s.r.E_s, Es = E0 * s.k;
  el("tc-depth").disabled = s.mag === "none";
  el("tc-depth-o").innerHTML = s.depth.toFixed(1) + HS + "mm";
  el("tc-gap-o").innerHTML = s.gap.toFixed(1) + HS + "mm";
  el("tc-es-o").innerHTML = Es.toFixed(1) + HS + "MPa (model: " + E0 + HS + "MPa" + (D.material[s.rec] === "TPU 95A" ? "; datasheets 26&#8211;67" + HS + "MPa" : "") + ")";
  var Estar = s.r.E_best * s.k, EstarNone = s.none.E_best * s.k, rows = [];
  rows.push(["Pad modulus", fmt(Estar, 1) + HS + "MPa" + (s.mag === "none" ? "" : ", " + pct((Estar / EstarNone - 1) * 100) + " over the print without a magnet")]);
  rows.push(["Approach at 1" + HS + "N", fmt(s.r.pf["1"].d / s.k, 1) + HS + "&#181;m"]);
  var at = s.r.pf["3"].arm, pa = padsArm(s.r.E_best);
  rows.push(["Twist arm at 3" + HS + "N, tip and pads (model&#8217;s modulus)", fmt(at, 2) + " and " + fmt(pa, 2) + HS + "mm" + (pa ? " (pads " + pct((pa / at - 1) * 100) + ")" : "")]);
  if (s.f) {
    var f = s.f, over = f.B0 > D.range_mT;
    rows.push(["Magnet travel per newton: normal, shear across, shear along", f3(f.normal.travel) + ", " + f3(f.across.travel) + ", " + f3(f.along.travel) + HS + "&#181;m/N"]);
    rows.push(["Field at the chip at rest", '<span class="' + (over ? "tc-warn" : "") + '">' + fmt(f.B0, 2) + HS + "mT" + (over ? ", over the " + D.range_mT + HS + "mT range of the highest gain" : " (range " + D.range_mT + HS + "mT)") + "</span>"]);
    rows.push(["Signal per newton: normal, across, along", f3(f.normal.S) + ", " + f3(f.across.S) + ", " + f3(f.along.S) + HS + "&#181;T/N"]);
    rows.push(["Force noise per sample at " + s.rate + HS + "Hz: normal, across, along", f3(f.normal.sigF) + ", " + f3(f.across.sigF) + ", " + f3(f.along.sigF) + HS + "N"]);
    rows.push(["Pocket", fmt(2 * s.pk.r, 2) + HS + "mm across, " + fmt(s.pk.hi - s.pk.lo, 2) + HS + "mm high, print heights " + fmt(s.pause.bot, 2) + "&#8211;" + fmt(s.pause.top, 2) + HS + "mm"]);
    rows.push(["Pause", "Before layer " + s.pause.n + ", which ends at " + fmt(s.pause.z, 2) + HS + "mm; the print stops at " + fmt((s.pause.n - 1) * s.layer, 2) + HS + "mm"]);
  }
  el("tc-out").innerHTML = rows.map(function (r) { return "<tr><td>" + r[0] + "</td><td>" + r[1] + "</td></tr>"; }).join("");
  var note = { run: "Finite-element run at this depth.", interpolated: "Interpolated between the finite-element runs at 1.5, 3 and 5" + HS + "mm.",
               shifted: "The 3.18" + HS + "mm magnet has one run, at 3" + HS + "mm; this depth adds the 4.76" + HS + "mm magnet&#8217;s change with depth to it." };
  el("tc-kind").innerHTML = (s.mag === "none" ? "Finite-element run without a magnet." : note[s.kind]) +
    (Math.abs(s.k - 1) > 1e-9 ? " Stiffness and travel scaled from the run&#8217;s " + E0 + HS + "MPa to " + Es.toFixed(1) + HS + "MPa; twist arms at the run&#8217;s modulus." : "");
  el("tc-sec").innerHTML = drawSection(s);
  el("tc-noise").innerHTML = s.f ? drawNoise(s) : '<p class="tc-note">No magnet: no field.</p>';
  el("tc-tipname").innerHTML = D.recipes.filter(function (x) { return x[0] === s.rec; })[0][1] + ", " +
    (s.mag === "none" ? "no magnet" : "magnet " + s.dm + " &#215; " + s.t + HS + "mm at " + s.depth.toFixed(1) + HS + "mm");
  updateFit(s);
}

function updateFit(s) {
  s = s || state();
  var ind = pairs(el("tc-indent").value), fp = pairs(el("tc-field").value), pl = pairs(el("tc-pull").value)[0] || null, tw = pairs(el("tc-twist").value)[0] || null;
  var fit = fitBench(s.r, s.mag !== "none", s.dm, s.t, s.depth, s.gap, s.br, ind, fp, pl, tw), rows = [];
  fit.est.forEach(function (e) { rows.push([e.kind, fmt(e.E, 1) + HS + "MPa"]); });
  if (fit.E_s) {
    rows.push(["Solid modulus, geometric mean", "<b>" + fmt(fit.E_s, 1) + HS + "MPa</b> (model " + s.r.E_s + HS + "MPa)"]);
    rows.push(["Pad modulus for the scenes", fmt(fit.E_star, 2) + HS + "MPa"]);
  }
  if (fit.mu) rows.push(["Friction coefficient", fmt(fit.mu, 3)]);
  if (fit.arm) rows.push(["Twist arm on the bench", fmt(fit.arm, 2) + HS + "mm (model at 3" + HS + "N: " + fmt(s.r.pf["3"].arm, 2) + HS + "mm)"]);
  el("tc-fitout").innerHTML = rows.length ? rows.map(function (r) { return "<tr><td>" + r[0] + "</td><td>" + r[1] + "</td></tr>"; }).join("")
                                          : "<tr><td>No measurements</td><td>Enter at least one indentation or field pair.</td></tr>";
  el("tc-cmd").textContent = fit.E_star ? sceneCommand(fit.E_star, fit.mu) : "";
  el("tc-fitfig").innerHTML = drawFit(s.r, fit, ind);
}

function init() {
  if (!el("tc-recipe")) return;
  el("tc-recipe").innerHTML = D.recipes.map(function (x) { return '<option value="' + x[0] + '">' + esc(x[1]) + "</option>"; }).join("");
  ["tc-recipe", "tc-magnet", "tc-depth", "tc-gap", "tc-es", "tc-br", "tc-rate", "tc-layer"].forEach(function (id) {
    el(id).addEventListener("input", update); el(id).addEventListener("change", update);
  });
  ["tc-indent", "tc-field", "tc-pull", "tc-twist"].forEach(function (id) { el(id).addEventListener("input", function () { updateFit(); }); });
  el("tc-stl").addEventListener("click", function () {
    var s = state(), T = tipTriangles(s.pk || null);
    var name = s.mag === "none" ? "sr2_tip_whole.stl" : "sr2_tip_" + s.mag + "_depth" + s.depth.toFixed(1).replace(".", "p") + ".stl";
    save(stlBlob(T, "SR2 fingertip, whole block" + (s.pk ? ", pocket for " + s.mag + " at " + s.depth.toFixed(1) + " mm" : "")), name);
  });
  el("tc-sheet").addEventListener("click", function () {
    var s = state(), rec = { material: D.material[s.rec], slicer: { walls: 2, line_width_mm: 0.45, layer_mm: s.layer, top_layers: 4, bottom_layers: 4,
      infill: "gyroid", infill_percent: D.infill[s.rec], orientation: "back face on the bed, palmar face up (layers across the load)", supports: "none" } };
    if (s.pk) {
      rec.magnet = s.mag; rec.magnet_diameter_mm = s.dm; rec.magnet_thickness_mm = s.t; rec.magnet_top_depth_mm = s.depth;
      rec.pocket_diameter_mm = +(2 * s.pk.r).toFixed(3); rec.pocket_height_mm = +(s.pk.hi - s.pk.lo).toFixed(3);
      rec.pocket_bottom_print_height_mm = +s.pause.bot.toFixed(3); rec.pocket_top_print_height_mm = +s.pause.top.toFixed(3);
      rec.pause_before_layer_at_mm = +s.pause.z.toFixed(3); rec.pause_layer_number = s.pause.n;
      rec.note = "Print face up, the part on its back face. Pause before the first layer printed above the pocket's top, press the magnet in with its north pole toward the palmar face, fix it with a drop of cyanoacrylate, and resume.";
    }
    rec.model = { E_s_MPa: s.r.E_s * s.k, pad_modulus_E_star_MPa: +(s.r.E_best * s.k).toFixed(2), source: "docs/experiments/20261009-tpu_tip_fem/magnet_runs.jsonl", depth_kind: s.kind };
    if (s.f) rec.model.field = { chip_distance_mm: s.gap, B0_mT: +s.f.B0.toFixed(3), force_noise_N: { normal: +s.f.normal.sigF.toFixed(3), across: +s.f.across.sigF.toFixed(3), along: +s.f.along.sigF.toFixed(3), rate_Hz: s.rate } };
    save(new Blob([JSON.stringify(rec, null, 1)], { type: "application/json" }), "sr2_tip_" + s.rec + "_" + (s.mag === "none" ? "nomagnet" : s.mag + "_depth" + s.depth.toFixed(1).replace(".", "p")) + ".json");
  });
  el("tc-copy").addEventListener("click", function () { if (navigator.clipboard) navigator.clipboard.writeText(el("tc-cmd").textContent); });
  update();
}

window.TipCalc = { rowAt: rowAt, field: field, pocket: pocket, pauseAt: pauseAt, padsArm: padsArm, approachAt: approachAt,
                   fitBench: fitBench, sceneCommand: sceneCommand, tipTriangles: tipTriangles, D: D };
if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
"""


def section(H, X, number_figs=True):
    """HTML of the calculator section: controls, outputs, three live figures, and the data and script."""
    d = json.dumps(data(X), separators=(",", ":"))
    controls = """
<div class="tc-grid">
<label><span class="tc-l">Print</span><select id="tc-recipe"></select></label>
<label><span class="tc-l">Magnet (N52 disc, axis normal to the face)</span><select id="tc-magnet">
<option value="d4.76x1.59">4.76 &#215; 1.59&#8202;mm, eFlesh&#8217;s fingertip magnet</option>
<option value="d3.18x1.59">3.18 &#215; 1.59&#8202;mm</option><option value="none">None</option></select></label>
<label><span class="tc-l">Magnet depth below the palmar face</span><input type="range" id="tc-depth" min="1.5" max="5" step="0.1" value="5"><output id="tc-depth-o"></output></label>
<label><span class="tc-l">Chip distance behind the back face</span><input type="range" id="tc-gap" min="-4" max="6" step="0.1" value="1"><output id="tc-gap-o"></output></label>
<label><span class="tc-l">Solid modulus of the TPU, times the model&#8217;s</span><input type="range" id="tc-es" min="0.5" max="2.5" step="0.05" value="1"><output id="tc-es-o"></output></label>
<label><span class="tc-l">Magnet grade</span><select id="tc-br"><option value="1.45">N52, remanence 1.45&#8202;T</option><option value="1.32">N42, remanence 1.32&#8202;T</option></select></label>
<label><span class="tc-l">Magnetometer output rate (averaged 100&#8202;Hz samples)</span><select id="tc-rate"><option value="100">100&#8202;Hz</option><option value="50">50&#8202;Hz</option><option value="25">25&#8202;Hz</option><option value="10">10&#8202;Hz</option></select></label>
<label><span class="tc-l">Layer height</span><select id="tc-layer"><option value="0.12">0.12&#8202;mm</option><option value="0.16">0.16&#8202;mm</option><option value="0.2" selected>0.20&#8202;mm</option><option value="0.24">0.24&#8202;mm</option><option value="0.28">0.28&#8202;mm</option></select></label>
</div>"""
    f_sec = H.figure('<div class="tc-svg" id="tc-sec"></div>',
                     "Section of the tip through the contact point as printed, palmar face up, with the magnet, its pocket (dashed), the "
                     "pause and the magnetometer; it redraws with the settings above.", key="calc_sec")
    f_noise = H.figure('<div class="tc-svg" id="tc-noise"></div>',
                       "Force noise per sample against the chip distance for the settings above: the load whose field change equals one "
                       "sample&#8217;s noise, for a normal load and for shear across and along the finger. Grey: the field at rest exceeds the "
                       "4.8&#8202;mT range of the MLX90393&#8217;s highest gain.", key="calc_noise")
    f_fit = H.figure('<div class="tc-svg" id="tc-fitfig"></div>',
                     "Approach against force for the tip chosen above: the finite-element model at its solid modulus (dashed), the model "
                     "scaled to the fitted modulus (solid) and the bench points.", key="calc_fit")
    calib = """
<p class="tc-note">Tip: <span id="tc-tipname"></span>. Enter one pair per line, separated by a space or a comma.</p>
<div class="tc-grid">
<label><span class="tc-l">Indentation: force (N) and approach (&#181;m) of the shaft or a 12.5&#8202;mm rod, axis across the finger, from first touch</span><textarea id="tc-indent">0.5 22
1 41
3 118</textarea></label>
<label><span class="tc-l">Field: force (N) and change of the field magnitude at the chip (&#181;T)</span><textarea id="tc-field"></textarea></label>
<label><span class="tc-l">Pull test: pinch force per pad (N) and the axial pull at slip (N)</span><textarea id="tc-pull">2 3.1</textarea></label>
<label><span class="tc-l">Twist test: pinch force per pad (N) and the torque at slip (N&#8202;mm)</span><textarea id="tc-twist"></textarea></label>
</div>
<table class="tc-out"><tbody id="tc-fitout"></tbody></table>
<p class="tc-note">Command that writes the scene with the fitted pad modulus and friction:
<button type="button" id="tc-copy">Copy</button></p>
<pre class="tc-cmd" id="tc-cmd"></pre>"""
    intro = ("<p>The calculator repeats the print plan, the magnet&#8217;s field and the calibration of this page in the browser, on the "
             "20 finite-element runs of Table&#160;[[tab:magnet]]. Depths between the runs at 1.5, 3 and 5&#8202;mm are interpolated "
             "linearly; stiffness and magnet travel scale with the solid modulus as in the calibration. The whole tip with the chosen "
             "pocket downloads as an STL, built in the page, and the print sheet as JSON; the default bench numbers are the example "
             "of <code>scripts/tpu_tip_calibrate.py</code>. The script is <code>scripts/tpu_tip_widget.py</code>; "
             "<code>scripts/tpu_tip_widget.py check</code> compares it with the Python tools.</p>")
    return (f"<style>{CSS}</style>{intro}<div class='tipcalc'>{controls}"
            f"<p class='tc-note' id='tc-kind'></p><table class='tc-out'><tbody id='tc-out'></tbody></table>"
            f"<p><button type='button' id='tc-stl'>Download STL of this tip</button><button type='button' id='tc-sheet'>Download print sheet</button></p>"
            f"{f_sec}{f_noise}<h3>Bench calibration</h3>{calib}{f_fit}</div>"
            f'<script type="application/json" id="tc-data">{d}</script><script>{JS}</script>')


# ------------------------------------------------------------------------------------------ check against Python

CHECK_JS = r"""
var C = window.TipCalc, out = [];
var cases = [["w95_20","d4.76x1.59",5],["w85_50","d4.76x1.59",5],["w85_30","d4.76x1.59",5],["w95_10","d4.76x1.59",5],
             ["w95_20","d4.76x1.59",3],["w95_20","d4.76x1.59",1.5],["w95_20","d3.18x1.59",3]];
cases.forEach(function (c) {
  var r = C.rowAt(c[0], c[1], c[2]).r, m = C.D.magnets[c[1]];
  var f1 = C.field(r, m[0], m[1], c[2], 1.0, C.D.br, 1, C.D.rate), f4 = C.field(r, m[0], m[1], c[2], 4.0, C.D.br, 1, C.D.rate);
  var p = C.pocket(m[0], m[1], c[2], 0.1, 0.1), pz = C.pauseAt(p.pk, 0.2);
  var fit = C.fitBench(r, true, m[0], m[1], c[2], 1.0, C.D.br, [[0.5,22],[1,41],[3,118]], [[1,9]], [2,3.1], [2,10]);
  out.push({ key: c, B0: f1.B0, sN1: f1.normal.sigF, sA1: f1.across.sigF, sL1: f1.along.sigF, SN1: f1.normal.S, sN4: f4.normal.sigF,
             pause_n: pz.n, pause_z: pz.z, E_s: fit.E_s, E_star: fit.E_star, mu: fit.mu, cmd: C.sceneCommand(fit.E_star, fit.mu),
             padsArm: C.padsArm(r.E_best) });
});
var T = C.tipTriangles(C.pocket(4.76, 1.59, 5, 0.1, 0.1).pk), vol = 0;
T.forEach(function (t) { var a = t[1], b = t[2], c = t[3];
  vol += (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0])) / 6; });
document.body.textContent = JSON.stringify({ cases: out, mesh: { triangles: T.length, volume: vol } });
"""


def check():
    import tpu_tip_calibrate as CAL
    import tpu_tip_fem_page as PG
    X = PG.Data()
    d = json.dumps(data(X), separators=(",", ":"))
    html = (f'<!doctype html><meta charset="utf-8"><body><script type="application/json" id="tc-data">{d}</script>'
            f"<script>{JS}</script><script>{CHECK_JS}</script></body>")
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, dir=str(ROOT / "logs")) as fh:
        fh.write(html)
        path = fh.name
    res = subprocess.run(["google-chrome", "--headless=new", "--disable-gpu", "--no-sandbox", "--dump-dom", f"file://{path}"],
                         capture_output=True, text=True, timeout=120)
    Path(path).unlink()
    body = res.stdout.split("<body>", 1)[1].rsplit("</body>", 1)[0]
    J = json.loads(body.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">"))
    R = PP.rows()
    worst = 0.0
    for c in J["cases"]:
        rec, mag, dep = c["key"]
        r = R[(rec, mag, float(dep))]
        f1, f4 = PP.field(r, 1.0), PP.field(r, 4.0)
        sh = TD.print_sheet(mag, float(dep))
        fit = CAL.fit(r, ["0.5:22", "1:41", "3:118"], ["1:9"], ["2:3.1"], ["2:10"], 1.0)
        pairs = [(c["B0"], f1["B0_mT"]), (c["sN1"], f1["sigF_normal_N"]), (c["sA1"], f1["sigF_shear_y_N"]),
                 (c["sL1"], f1["sigF_shear_z_N"]), (c["SN1"], f1["S_normal_norm"]), (c["sN4"], f4["sigF_normal_N"]),
                 (c["E_s"], fit["E_s_MPa"]), (c["E_star"], fit["E_star_MPa"]), (c["mu"], fit["mu"]),
                 (c["padsArm"], X.pads_arm(r["E_best_MPa"], 3.0))]
        err = max(abs(a / b - 1) for a, b in pairs if b)
        worst = max(worst, err)
        same_pause = (c["pause_n"], round(c["pause_z"], 3)) == (sh["pause_layer_number"], sh["pause_before_layer_at_mm"])
        same_cmd = c["cmd"].split("--model")[1] == fit["scene_command"].split("--model")[1]
        print(f"{rec} {mag} {dep}: max rel diff {err:.2e}; pause {c['pause_n']} {c['pause_z']:.2f} "
              f"{'=' if same_pause else '!='} {sh['pause_layer_number']} {sh['pause_before_layer_at_mm']}; command {'same' if same_cmd else 'DIFFERENT'}")
    m = TD.mesh("d4.76x1.59", 5.0)
    print(f"mesh: browser {J['mesh']['triangles']} triangles, volume {J['mesh']['volume']:.2f} mm^3; "
          f"Python convex hull {len(m.faces)} faces, volume {m.volume:.2f} mm^3")
    print(f"worst relative difference {worst:.2e}")
    return worst


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        sys.exit(0 if check() < 1e-3 else 1)   # the pads' arm is a 61-point table in the page
    print(__doc__)
