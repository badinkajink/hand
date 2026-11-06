#!/usr/bin/env python3
"""An explorable version of the adjoint figure of the HOM review page (scripts/hom/hom_eq1_figures.py adjoint): a planar
rigid body with frames {B} and {A}; the reader drags {A}, turns it, and sets the twist (w, v_B) of the body, and the page
shows the velocity of the body's points, the lever-arm term w x r at {A}, the twist's numbers in both frames and the
planar adjoint with its numbers. Plain SVG and JavaScript inside the page: no library, no fetch, so it works from a
file:// clone. The script has no dollar sign, because the page builder renders $...$ as LaTeX.

Conventions as the static figure and src/sr2hand/control/hom.py: every frame's y axis points out of the page, its z axis
is its x axis turned 90 deg clockwise, w is positive counter-clockwise, and a planar twist is (w_y; v_x, v_z).

    import hom_adjoint_widget as AW;  html = AW.html(caption)
"""
from __future__ import annotations

CSS = """
.adx { margin: 18px 0; }
.adx svg { display: block; margin: 0 auto; max-width: 100%; height: auto; border: 1px solid #000; touch-action: none; }
.adx .ctl { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 4px 18px; margin: 8px 0 4px; font-size: 16px; }
.adx .ctl label { display: flex; align-items: center; gap: 8px; }
.adx .ctl input[type=range] { flex: 1; min-width: 80px; }
.adx .ctl output { min-width: 4.5em; text-align: right; font-family: "Courier New", Courier, monospace; font-size: 15px; }
.adx .out { display: grid; grid-template-columns: 1fr; gap: 2px; font-size: 16px; margin: 4px 0; text-align: left; }
.adx .out .num { font-family: "Courier New", Courier, monospace; font-size: 15px; }
.adx table.adm { border-collapse: collapse; margin: 4px 0; border-left: 2px solid #000; border-right: 2px solid #000; }
.adx table.adm td { border: 0 !important; padding: 1px 6px !important; text-align: right; font-family: "Courier New", Courier, monospace; font-size: 15px; }
.adx table.adm td.lev { color: #7b3fa0 !important; }
.adx .reset { font-size: 15px; }
@media (max-width: 640px) { .adx .ctl { grid-template-columns: 1fr; } }
"""

SCRIPT = r"""
(function () {
  var NS = "http://www.w3.org/2000/svg";
  var svg = document.getElementById("adx-svg");
  if (!svg) return;
  var S = 7.0, X0 = -20.0, Y1 = 31.0;                       // px per mm; the window's left edge and top (mm)
  var SC = 0.42;                                              // mm of arrow per mm/s
  var C = {fin: "#00838f", lev: "#7b3fa0", rel: "#111111", x: "#d62728", z: "#1f63b4", y: "#2ca02c",
           grey: "#a9a9a9", body: "#efefef"};
  var PSI = 15 * Math.PI / 180;                               // the body and {B} are turned 15 deg
  var st = {w: -25, vx: 7.2, vz: -11.5, turn: -110, ax: 0, ay: 0};
  function rot(a, v) { return [Math.cos(a) * v[0] - Math.sin(a) * v[1], Math.sin(a) * v[0] + Math.cos(a) * v[1]]; }
  var A0 = rot(PSI, [47, 6]);
  st.ax = A0[0]; st.ay = A0[1];
  function P(p) { return [(p[0] - X0) * S, (Y1 - p[1]) * S]; }
  function el(tag, attrs, parent) {
    var e = document.createElementNS(NS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    (parent || svg).appendChild(e);
    return e;
  }
  var defs = el("defs", {});
  ["fin", "lev", "rel", "x", "z", "grey"].forEach(function (k) {
    var m = el("marker", {id: "adx-m-" + k, viewBox: "0 0 10 10", refX: "8.5", refY: "5", markerWidth: k === "grey" ? "6" : "7",
                          markerHeight: k === "grey" ? "6" : "7", orient: "auto-start-reverse", markerUnits: "userSpaceOnUse"}, defs);
    m.setAttribute("markerWidth", k === "grey" ? "9" : "13");
    m.setAttribute("markerHeight", k === "grey" ? "9" : "13");
    el("path", {d: "M0,0 L10,5 L0,10 z", fill: C[k]}, m);
  });
  function arrow(col, w, dash) {
    var a = {stroke: C[col], "stroke-width": w, fill: "none", "marker-end": "url(#adx-m-" + col + ")"};
    if (dash) a["stroke-dasharray"] = dash;
    return el("line", a);
  }
  function setLine(l, p, q) {
    var a = P(p), b = P(q);
    l.setAttribute("x1", a[0]); l.setAttribute("y1", a[1]); l.setAttribute("x2", b[0]); l.setAttribute("y2", b[1]);
    var len = Math.hypot(b[0] - a[0], b[1] - a[1]);
    l.style.display = len < 2 ? "none" : "";
  }
  function text(s, col, size, italic) {
    var t = el("text", {fill: col, "font-size": size || 17, "font-family": "Times New Roman, Times, serif"});
    if (italic) t.setAttribute("font-style", "italic");
    t.textContent = s;
    return t;
  }
  function place(t, p, dx, dy) { var a = P(p); t.setAttribute("x", a[0] + (dx || 0)); t.setAttribute("y", a[1] + (dy || 0)); }
  // the body: a slab 70 x 26 mm with rounded corners, turned 15 deg about {B}
  (function () {
    var pts = [], r = 6, hw = 35, hh = 13, cx = 23, i, a;
    var corners = [[cx + hw - r, hh - r, 0], [cx - hw + r, hh - r, 90], [cx - hw + r, -hh + r, 180], [cx + hw - r, -hh + r, 270]];
    corners.forEach(function (c) {
      for (i = 0; i <= 6; i++) { a = (c[2] + 15 * i) * Math.PI / 180; pts.push(rot(PSI, [c[0] + r * Math.cos(a), c[1] + r * Math.sin(a)])); }
    });
    el("polygon", {points: pts.map(function (p) { return P(p).join(","); }).join(" "), fill: C.body, stroke: "#555", "stroke-width": 1.4});
  })();
  var field = [];
  for (var gx = -4; gx <= 54; gx += 10) for (var gz = -7; gz <= 7; gz += 14) field.push({p: rot(PSI, [gx, gz]), l: arrow("grey", 1.2)});
  var icr = text("×", "#7a7a7a", 22); icr.setAttribute("text-anchor", "middle"); icr.setAttribute("dominant-baseline", "central");
  var rLine = el("line", {stroke: "#555", "stroke-width": 1.2, "stroke-dasharray": "5 4"});
  var frames = {};
  function makeFrame(name) {
    var f = {x: arrow("x", 2), z: arrow("z", 2), ring: el("circle", {r: 8, fill: "#fff", stroke: C.y, "stroke-width": 2}),
             dot: el("circle", {r: 3, fill: C.y}), lab: text("{" + name + "}", "#000", 18, true)};
    frames[name] = f;
    return f;
  }
  var vB = arrow("fin", 3), vBc = arrow("fin", 2.2, "7 4"), lev = arrow("lev", 2.8), vA = arrow("rel", 3.4);
  function subtext(main, sub, col) {
    var t = text(main, col, 19, true), s_ = document.createElementNS(NS, "tspan");
    s_.setAttribute("font-size", "14"); s_.setAttribute("dy", "6"); s_.textContent = sub; t.appendChild(s_);
    return t;
  }
  var lw = text("ω × r", C.lev, 18, true), lvB = subtext("v", "B", C.fin), lvA = subtext("v", "A", C.rel);
  var lr = text("r", "#444", 17, true);
  var fB = makeFrame("B"), fA = makeFrame("A");
  fA.ring.setAttribute("r", 10); fA.ring.style.cursor = "grab";
  var hint = text("drag {A}", "#666", 15);
  function setFrame(f, o, x) {
    var z = [x[1], -x[0]];
    setLine(f.x, o, [o[0] + 7 * x[0], o[1] + 7 * x[1]]);
    setLine(f.z, o, [o[0] + 7 * z[0], o[1] + 7 * z[1]]);
    var a = P(o);
    f.ring.setAttribute("cx", a[0]); f.ring.setAttribute("cy", a[1]);
    f.dot.setAttribute("cx", a[0]); f.dot.setAttribute("cy", a[1]);
  }
  function fmt(v, d) { var s = v.toFixed(d); return (v >= 0 ? "+" : "") + s; }
  function out(id, s) { var e = document.getElementById(id); if (e) e.textContent = s; }
  function draw() {
    var w = st.w * Math.PI / 180;
    var xB = rot(PSI, [1, 0]), zB = [xB[1], -xB[0]];
    var vb = [st.vx * xB[0] + st.vz * zB[0], st.vx * xB[1] + st.vz * zB[1]];   // v_B on the page
    function vel(p) { return [vb[0] - w * p[1], vb[1] + w * p[0]]; }          // v_B + w x (p - o_B), o_B = 0
    field.forEach(function (f) { var v = vel(f.p); setLine(f.l, f.p, [f.p[0] + SC * v[0], f.p[1] + SC * v[1]]); });
    if (Math.abs(w) > 1e-6) { place(icr, [-vb[1] / w, vb[0] / w]); icr.style.display = ""; } else icr.style.display = "none";
    var oA = [st.ax, st.ay];
    var xA = rot(st.turn * Math.PI / 180, [1, 0]), zA = [xA[1], -xA[0]];
    var r = oA, lv = [-w * r[1], w * r[0]], va = [vb[0] + lv[0], vb[1] + lv[1]];
    setLine(rLine, [0, 0], oA);
    setLine(vB, [0, 0], [SC * vb[0], SC * vb[1]]);
    setLine(vBc, oA, [oA[0] + SC * vb[0], oA[1] + SC * vb[1]]);
    setLine(lev, [oA[0] + SC * vb[0], oA[1] + SC * vb[1]], [oA[0] + SC * va[0], oA[1] + SC * va[1]]);
    setLine(vA, oA, [oA[0] + SC * va[0], oA[1] + SC * va[1]]);
    setFrame(fB, [0, 0], xB); setFrame(fA, oA, xA);
    place(fB.lab, [0, 0], -40, 30); place(fA.lab, oA, -46, 30); place(hint, oA, -40, 50);
    place(lvB, [SC * vb[0], SC * vb[1]], 9, 2);
    place(lvA, [oA[0] + SC * va[0], oA[1] + SC * va[1]], 10, 16);
    place(lw, [oA[0] + SC * (vb[0] + 0.5 * lv[0]), oA[1] + SC * (vb[1] + 0.5 * lv[1])], 12, 4);
    place(lr, [0.5 * oA[0], 0.5 * oA[1]], -4, -8);
    // numbers: V_B in {B}'s axes, V_A in {A}'s axes, p_AB and R_AB in {A}'s axes, the planar adjoint
    var dot = function (a, b) { return a[0] * b[0] + a[1] * b[1]; };
    var vaA = [dot(va, xA), dot(va, zA)];
    var pB = [-oA[0], -oA[1]], p = [dot(pB, xA), dot(pB, zA)];
    var R = [[dot(xB, xA), dot(zB, xA)], [dot(xB, zA), dot(zB, zA)]];
    var wd = st.w;
    out("adx-vb", "(" + fmt(wd, 0) + " deg/s; " + fmt(st.vx, 1) + ", " + fmt(st.vz, 1) + " mm/s)");
    out("adx-va", "(" + fmt(wd, 0) + " deg/s; " + fmt(vaA[0], 1) + ", " + fmt(vaA[1], 1) + " mm/s)");
    out("adx-p", "(" + fmt(p[0], 1) + ", " + fmt(p[1], 1) + ") mm, |r| = " + Math.hypot(p[0], p[1]).toFixed(1) + " mm");
    out("adx-lev", "(" + fmt(dot(lv, xA), 1) + ", " + fmt(dot(lv, zA), 1) + ") mm/s");
    var m = [["1", "0", "0"], [(-p[1]).toFixed(1), R[0][0].toFixed(3), R[0][1].toFixed(3)],
             [(p[0]).toFixed(1), R[1][0].toFixed(3), R[1][1].toFixed(3)]];
    for (var i = 0; i < 3; i++) for (var j = 0; j < 3; j++) out("adx-m" + i + j, m[i][j]);
    out("adx-w", fmt(wd, 0)); out("adx-vx", fmt(st.vx, 1)); out("adx-vz", fmt(st.vz, 1)); out("adx-turn", fmt(st.turn, 0));
  }
  // controls
  [["adx-in-w", "w"], ["adx-in-vx", "vx"], ["adx-in-vz", "vz"], ["adx-in-turn", "turn"]].forEach(function (c) {
    var e = document.getElementById(c[0]);
    if (!e) return;
    e.value = st[c[1]];
    e.addEventListener("input", function () { st[c[1]] = parseFloat(e.value); draw(); });
  });
  var reset = document.getElementById("adx-reset");
  if (reset) reset.addEventListener("click", function () {
    st.w = -25; st.vx = 7.2; st.vz = -11.5; st.turn = -110; st.ax = A0[0]; st.ay = A0[1];
    [["adx-in-w", "w"], ["adx-in-vx", "vx"], ["adx-in-vz", "vz"], ["adx-in-turn", "turn"]].forEach(function (c) {
      document.getElementById(c[0]).value = st[c[1]]; });
    draw();
  });
  // dragging {A}
  var drag = false;
  function world(ev) {
    var pt = svg.createSVGPoint(); pt.x = ev.clientX; pt.y = ev.clientY;
    var q = pt.matrixTransform(svg.getScreenCTM().inverse());
    return [q.x / S + X0, Y1 - q.y / S];
  }
  fA.ring.addEventListener("pointerdown", function (ev) { drag = true; fA.ring.style.cursor = "grabbing"; svg.setPointerCapture(ev.pointerId); ev.preventDefault(); });
  svg.addEventListener("pointermove", function (ev) {
    if (!drag) return;
    var q = world(ev);
    st.ax = Math.max(-16, Math.min(84, q[0])); st.ay = Math.max(-24, Math.min(26, q[1]));
    draw();
  });
  svg.addEventListener("pointerup", function () { drag = false; fA.ring.style.cursor = "grab"; });
  svg.addEventListener("pointercancel", function () { drag = false; });
  draw();
})();
"""


def html(caption: str) -> str:
    """The explorer: an SVG, four sliders, the twist's numbers in both frames and the planar adjoint, under a caption."""
    def slider(id_, label, lo, hi, step, unit):
        return (f'<label><span>{label}</span><input type="range" id="adx-in-{id_}" min="{lo}" max="{hi}" step="{step}">'
                f'<output id="adx-{id_}"></output><span>{unit}</span></label>')
    ctl = "".join([slider("w", "&omega;", -60, 60, 1, "deg/s"), slider("turn", "turn of {A}", -180, 180, 1, "deg"),
                   slider("vx", "v<sub>B</sub> along x<sub>B</sub>", -20, 20, 0.1, "mm/s"),
                   slider("vz", "v<sub>B</sub> along z<sub>B</sub>", -20, 20, 0.1, "mm/s")])
    lev = ' class="lev"'
    mat = ('<table class="adm">' + "".join(
        "<tr>" + "".join(f'<td id="adx-m{i}{j}"{lev if j == 0 and i > 0 else ""}></td>' for j in range(3))
        + "</tr>" for i in range(3)) + "</table>")
    outs = (f'<div>Twist at {{B}}, in {{B}}&#39;s axes: <span class="num" id="adx-vb"></span></div>'
            f'<div>Twist at {{A}}, in {{A}}&#39;s axes: <span class="num" id="adx-va"></span></div>'
            f'<div>Position of {{B}} in {{A}}: <span class="num" id="adx-p"></span></div>'
            f'<div>Lever-arm term at {{A}}: <span class="num" id="adx-lev"></span></div>'
            f'<div>Planar adjoint, rows (&omega;, v<sub>x</sub>, v<sub>z</sub>), &omega; in rad/s and lengths in mm: {mat}</div>'
            f'<div><button type="button" class="reset" id="adx-reset">Reset</button></div>')
    return (f'<style>{CSS}</style><figure class="adx"><svg id="adx-svg" viewBox="0 0 756 350" width="756" height="350" '
            f'role="img" aria-label="Explorable adjoint"></svg><div class="ctl">{ctl}</div><div class="out">{outs}</div>'
            f'<figcaption>{caption}</figcaption></figure><script>{SCRIPT}</script>')
