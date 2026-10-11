#!/usr/bin/env python3
"""Interactive data figures for the result pages: the default view is an inline SVG drawn at build time, and a small
page script adds a hover readout, series toggles, x-zoom by drag and a CSV download of the figure's data.

    from interactive_figure import Figure, Panel, Series, add_assets
    fig = Figure("eval-60m", x, "Control step", [Panel("cos", "Signed cosine", (-1, 1))], [Series("a", "60 M", "#2a78d6",
                 data={"cos": {"med": m, "lo": lo, "hi": hi}})], caption="...")
    page = add_assets(page.replace("{{FIG}}", fig.html()))          # style and script once per page, before </body>

    python3 scripts/interactive_figure.py --exercise PAGE.html [--out DIR]   # drive every control in headless Chrome

The static SVG stays when JavaScript is off and in print (owner, 2026-10-11: "a default static view"); its text is
14 px at the 828 px column, sentence case, in the page's own font. Panels stack vertically. A panel shares the figure's
x axis (eval traces: one crosshair across the panels) unless it has an x_domain or x_label of its own; x and y axes can
be logarithmic. A series is a line, markers (a scatter) or both, with an optional 10-90 % band and single rollouts, and
its own x values per panel where it has them. The data sit inline as JSON in the figure (<script
type="application/json">), each array as integers at the panel's decimals (log10 of the value on a log axis),
delta-coded, null for a gap, so a page holds a few hundred traces of 200 points within the 300 kB data budget. The
script (about 9 kB, once per page) uses no library, no CDN and no fetch, so a page works from file:// and inside an
artifact. Interaction redraws only the data layer and the x axes; the y axes keep the build-time domain.

Eval traces of scripts/policy_eval_suite.py go through eval_figure() (median and 10-90 % band over the rollouts, the
held share, a few single rollouts), downsampled to 200 points.
"""
from __future__ import annotations

import argparse
import html as htmlmod
import json
import math
import os
import time
from dataclasses import dataclass, field

import numpy as np

W, LM, RM = 828, 58, 14                      # viewBox width = column width in px; left and right margins
TITLE_H, GAP, XAX_H = 24, 12, 44             # panel title row, gap below a panel, x-axis band under a panel
FS = 14                                      # chart text, px at the column width
GRID, INK, MUTED = "#d9d9d9", "#000", "#444"
# Categorical slots in fixed order (dataviz reference palette, light mode; adjacent pairs pass the CVD gate)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
DATA_BUDGET = 300_000                        # bytes of figure data per page (owner, 2026-10-11)
LOG_DEC = 3                                  # decimals of log10(value) on a log axis: 0.23 % steps


@dataclass
class Panel:
    key: str
    title: str                               # sentence case, unit in brackets
    domain: tuple | None = None              # y domain; None = nice bounds of the data
    refs: tuple = ()                         # horizontal reference lines: ((y, "label"), ...)
    band: bool = True                        # series may carry lo / hi (band) and roll (single rollouts)
    dec: int | None = None                   # decimals of the stored y data and the readout; None = 3 significant digits
    height: int = 120
    pad: float = 0.04                        # drawn domain extends this share of the span beyond the ticked one, so a
                                             # line at the domain's edge stays off the frame
    ylog: bool = False
    x_domain: tuple | None = None            # an x axis of the panel's own (tick labels and label under the panel)
    x_label: str | None = None
    xlog: bool = False
    x_dec: int | None = None
    vrefs: tuple = ()                        # vertical reference lines: ((x, "label"), ...), keyed in the legend


@dataclass
class Series:
    id: str
    label: str
    color: str
    data: dict                               # panel key -> {"x": [...] (own x, optional), "med": [...], "lo", "hi", "roll"}
    dash: str = ""
    facets: dict = field(default_factory=dict)
    width: float = 2.0
    mode: str = "line"                       # "line", "markers" or "both"
    open: bool = False                       # hollow markers
    point_labels: dict = field(default_factory=dict)   # panel key -> one label per point, shown by the hover readout


def nice_step(span: float, n: int) -> float:
    raw = max(span, 1e-12) / max(n, 1)
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if m * mag >= raw * (1 - 1e-9):
            return m * mag
    return 10 * mag


def ticks(lo: float, hi: float, n: int = 5) -> tuple[list[float], int]:
    st = nice_step(hi - lo, n)
    t = math.ceil(lo / st - 1e-9) * st
    out = []
    while t <= hi + 1e-9 * st:
        out.append(round(t, 10))
        t += st
    dec = max(0, -math.floor(math.log10(st) + 1e-9)) + (1 if round(st / 10 ** math.floor(math.log10(st)), 6) == 2.5 else 0)
    return out, dec


def log_ticks(lo: float, hi: float) -> list[float]:
    """Decades inside [lo, hi]; with fewer than three, the 2x and 5x multiples as well."""
    a, b = math.floor(math.log10(lo) - 1e-9), math.ceil(math.log10(hi) + 1e-9)
    dec = [10.0 ** e for e in range(a, b + 1) if lo * (1 - 1e-9) <= 10.0 ** e <= hi * (1 + 1e-9)]
    if len(dec) >= 3:
        return dec
    return [m * 10.0 ** e for e in range(a, b + 1) for m in (1, 2, 5) if lo * (1 - 1e-9) <= m * 10.0 ** e <= hi * (1 + 1e-9)]


def fmt(v: float, dec: int) -> str:
    s = f"{abs(v):,.{dec}f}"
    return ("&#8722;" + s) if v < 0 and float(s.replace(",", "")) != 0 else s


def fmt_log(v: float) -> str:
    return fmt(v, max(0, -math.floor(math.log10(v) + 1e-9)))


def auto_dec(values: np.ndarray) -> int:
    a = np.abs(values[np.isfinite(values)])
    a = a[a > 0]
    if not a.size:
        return 2
    return int(max(0, min(4, 2 - math.floor(math.log10(float(np.median(a)))))))


def delta_ints(a, dec: int, log: bool = False) -> list:
    """Integers at `dec` decimals (of log10 of the value when `log`), each as the difference to the previous finite one;
    None for a gap."""
    out, prev = [], 0
    for v in np.asarray(a, dtype=float).ravel():
        if not np.isfinite(v) or (log and v <= 0):
            out.append(None)
            continue
        q = int(round((math.log10(v) if log else v) * 10 ** dec))
        out.append(q - prev)
        prev = q
    return out


def _nice_domain(lo: float, hi: float, log: bool) -> tuple[float, float]:
    if log:
        return 10.0 ** math.floor(math.log10(lo) + 1e-9), 10.0 ** math.ceil(math.log10(hi) - 1e-9)
    if hi - lo < 1e-9:
        hi = lo + 1
    st = nice_step(hi - lo, 4)
    return math.floor(lo / st + 1e-9) * st, math.ceil(hi / st - 1e-9) * st


class Figure:
    def __init__(self, fid: str, x, x_label: str, panels: list[Panel], series: list[Series], *, facets=(), legend=None,
                 caption: str = "", vlines=(), show_band: bool = True, show_rollouts: bool = False,
                 x_seconds: float | None = None, x_dec: int | None = None, show_median: bool = True, note: str = "",
                 x_domain=None, xlog: bool = False):
        self.fid, self.x_label = fid, x_label
        self.x = None if x is None else np.asarray(x, dtype=float)
        self.panels, self.series, self.facets = panels, series, list(facets)
        # legend entries (label, colour, dash, ids of the series the entry stands for); hidden series drop out
        self.legend = legend if legend is not None else [(s.label, s.color, s.dash, [s.id]) for s in series]
        self.legend = [e if len(e) == 4 else (*e, []) for e in self.legend]
        self.caption, self.vlines, self.note = caption, list(vlines), note
        self.show_band, self.show_rollouts, self.show_median = show_band, show_rollouts, show_median
        self.x_seconds, self.xlog = x_seconds, xlog
        shared_x = [self.x] if self.x is not None else []
        for p in panels:
            if not self.own_x(p):
                shared_x += [np.asarray(s.data[p.key]["x"], float) for s in series if p.key in s.data and "x" in s.data[p.key]]
        allx = np.concatenate([np.ravel(a) for a in shared_x]) if shared_x else np.zeros(1)
        self.x_dec = x_dec if x_dec is not None else (LOG_DEC if xlog else auto_dec(allx))
        self.x_domain = tuple(x_domain) if x_domain else (float(np.nanmin(allx)), float(np.nanmax(allx)))
        for p in panels:
            vals = np.concatenate([np.ravel(np.asarray(v, dtype=float)) for s in series for k, v in s.data.get(p.key, {}).items()
                                   if k != "x"] or [np.zeros(1)])
            vals = vals[np.isfinite(vals) & ((vals > 0) if p.ylog else True)]
            if p.dec is None:
                p.dec = LOG_DEC if p.ylog else auto_dec(vals)
            if p.domain is None:
                lo, hi = (float(vals.min()), float(vals.max())) if vals.size else (0.0, 1.0)
                for y, _ in p.refs:
                    lo, hi = min(lo, y), max(hi, y)
                p.domain = _nice_domain(lo, hi, p.ylog)
            if self.own_x(p):
                xs = np.concatenate([np.ravel(np.asarray(s.data[p.key].get("x", self.x if self.x is not None else []), float))
                                     for s in series if p.key in s.data] or [np.zeros(1)])
                xs = xs[np.isfinite(xs) & ((xs > 0) if p.xlog else True)]
                if p.x_domain is None:
                    p.x_domain = _nice_domain(float(xs.min()), float(xs.max()), p.xlog) if xs.size else (0.0, 1.0)
                if p.x_dec is None:
                    p.x_dec = LOG_DEC if p.xlog else auto_dec(xs)
        # layout: panel boxes in viewBox px; an own x axis takes a band under its panel, the shared one under the last
        self.boxes, y = [], 0
        last_shared = max([k for k, p in enumerate(panels) if not self.own_x(p)], default=-1)
        self.labels_x = []
        for k, p in enumerate(panels):
            top = y + TITLE_H
            self.boxes.append((top, top + p.height))
            band = self.own_x(p) or k == last_shared
            self.labels_x.append(band)
            y = top + p.height + (XAX_H if band and k != len(panels) - 1 else GAP)
        self.H = y - GAP + XAX_H                  # the last panel always carries x tick labels

    # --- geometry (the page script repeats these maps) ---
    @staticmethod
    def own_x(p: Panel) -> bool:
        return p.x_domain is not None or p.x_label is not None

    def xdom(self, k):
        p = self.panels[k]
        return (p.x_domain, p.xlog) if self.own_x(p) else (self.x_domain, self.xlog)

    def sx(self, v, k):
        (a, b), lg = self.xdom(k)
        t = (lambda u: math.log10(u)) if lg else (lambda u: u)
        return LM + (W - LM - RM) * (t(v) - t(a)) / (t(b) - t(a))

    def ydom(self, k):
        p = self.panels[k]
        lo, hi = p.domain
        if p.ylog:
            a, b = math.log10(lo), math.log10(hi)
            return (10 ** (a - p.pad * (b - a)), 10 ** (b + p.pad * (b - a)))
        return (lo - p.pad * (hi - lo), hi + p.pad * (hi - lo))

    def sy(self, v, k):
        p = self.panels[k]
        lo, hi = self.ydom(k)
        t = (lambda u: math.log10(u)) if p.ylog else (lambda u: u)
        y0, y1 = self.boxes[k]
        return y1 - (y1 - y0) * (t(v) - t(lo)) / (t(hi) - t(lo))

    def visible(self, s: Series) -> bool:
        for f in self.facets:
            if f["key"] in s.facets and s.facets[f["key"]] not in f.get("on", [v for v, _ in f["values"]]):
                return False
        return True

    def _xs(self, s, k):
        p = self.panels[k]
        return np.asarray(s.data[p.key].get("x", self.x), dtype=float)

    def _ok(self, a, b, k):
        p = self.panels[k]
        _, xl = self.xdom(k)
        return np.isfinite(a) and np.isfinite(b) and (a > 0 or not xl) and (b > 0 or not p.ylog)

    def _path(self, xs, ys, k):
        out, pen = [], "M"
        for a, b in zip(xs, ys):
            if not self._ok(a, b, k):
                pen = "M"
                continue
            out.append(f"{pen}{self.sx(a, k):.1f},{self.sy(b, k):.1f}")
            pen = "L"
        return "".join(out)

    def _xaxis(self, k, labels: bool) -> str:
        y0, y1 = self.boxes[k]
        (a, b), lg = self.xdom(k)
        if lg:
            tv, lab = log_ticks(a, b), fmt_log
        else:
            tv, dec = ticks(a, b, 6)
            lab = lambda t: fmt(t, dec)
        out = []
        for t in tv:
            x = self.sx(t, k)
            out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{y0}" y2="{y1}" stroke="{GRID}" stroke-width="1"/>')
            if labels:
                out.append(f'<text x="{x:.1f}" y="{y1 + 18}" text-anchor="middle" font-size="{FS}">{lab(t)}</text>')
        return "".join(out)

    def _data(self, k) -> str:
        p, out = self.panels[k], []
        vis = [s for s in self.series if self.visible(s) and p.key in s.data]
        for s in vis:
            d, xs = s.data[p.key], self._xs(s, k)
            if p.band and self.show_band and "lo" in d:
                up = [(self.sx(a, k), self.sy(b, k)) for a, b in zip(xs, d["hi"]) if self._ok(a, b, k)]
                dn = [(self.sx(a, k), self.sy(b, k)) for a, b in zip(xs, d["lo"]) if self._ok(a, b, k)][::-1]
                if up and dn:
                    pts = "L".join(f"{a:.1f},{b:.1f}" for a, b in up + dn)
                    out.append(f'<path class="b" d="M{pts}Z" fill="{s.color}" fill-opacity="0.18" stroke="none"/>')
        for s in vis:
            d, xs = s.data[p.key], self._xs(s, k)
            if p.band and self.show_rollouts:
                for r in d.get("roll", []):
                    out.append(f'<path d="{self._path(xs, r, k)}" fill="none" stroke="{s.color}" stroke-opacity="0.45" stroke-width="1"/>')
        for s in vis:
            d, xs = s.data[p.key], self._xs(s, k)
            if s.mode in ("line", "both") and (self.show_median or not p.band):
                da = f' stroke-dasharray="{s.dash}"' if s.dash else ""
                out.append(f'<path d="{self._path(xs, d["med"], k)}" fill="none" stroke="{s.color}" '
                           f'stroke-width="{s.width:g}"{da} stroke-linejoin="round"/>')
            if s.mode in ("markers", "both"):
                fill = "#fff" if s.open else s.color
                out.append("".join(f'<circle cx="{self.sx(a, k):.1f}" cy="{self.sy(b, k):.1f}" r="4.5" fill="{fill}" '
                                   f'stroke="{s.color if s.open else "#fff"}" stroke-width="{1.8 if s.open else 1.2}"/>'
                                   for a, b in zip(xs, d["med"]) if self._ok(a, b, k)))
        return "".join(out)

    def svg(self) -> str:
        f, out = self.fid, []
        titles = ", ".join(htmlmod.escape(htmlmod.unescape(p.title)) for p in self.panels)
        out.append(f'<svg class="ifig-svg" viewBox="0 0 {W} {self.H}" width="{W}" height="{self.H}" role="img" '
                   f'aria-label="{titles}" style="font-family:inherit;max-width:100%;height:auto" tabindex="0">')
        out.append("<defs>" + "".join(f'<clipPath id="{f}-c{k}"><rect x="{LM}" y="{y0}" width="{W - LM - RM}" height="{y1 - y0}"/></clipPath>'
                                      for k, (y0, y1) in enumerate(self.boxes)) + "</defs>")
        for k, p in enumerate(self.panels):
            y0, y1 = self.boxes[k]
            out.append(f'<text x="{LM}" y="{y0 - 7}" font-size="{FS}" fill="{INK}">{p.title}</text>')
            if p.ylog:
                yt = [(t, fmt_log(t)) for t in log_ticks(*p.domain)]
            else:
                tv, dec = ticks(p.domain[0], p.domain[1], 4)
                yt = [(t, fmt(t, dec)) for t in tv]
            for t, lab in yt:
                y = self.sy(t, k)
                out.append(f'<line x1="{LM}" x2="{W - RM}" y1="{y:.1f}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
                out.append(f'<text x="{LM - 6}" y="{y + 5:.1f}" text-anchor="end" font-size="{FS}">{lab}</text>')
            out.append(f'<g class="xa">{self._xaxis(k, self.labels_x[k])}</g>')
            for yv, lab in p.refs:
                y = self.sy(yv, k)
                out.append(f'<line x1="{LM}" x2="{W - RM}" y1="{y:.1f}" y2="{y:.1f}" stroke="{MUTED}" stroke-width="1" stroke-dasharray="5 4"/>')
                if lab:
                    ty = y - 4 if y - 4 - FS > y0 else y + FS + 2
                    out.append(f'<text x="{W - RM - 4}" y="{ty:.1f}" text-anchor="end" font-size="{FS}" fill="{MUTED}">{lab}</text>')
            out.append(f'<g class="vl" clip-path="url(#{f}-c{k})">')
            vls = list(p.vrefs) + ([] if self.own_x(p) else self.vlines)
            for xv, lab in vls:
                x = self.sx(xv, k)
                out.append(f'<line data-x="{xv:g}" x1="{x:.1f}" x2="{x:.1f}" y1="{y0}" y2="{y1}" stroke="{MUTED}" stroke-width="1" stroke-dasharray="5 4"/>')
            out.append("</g>")
            out.append(f'<g class="d" clip-path="url(#{f}-c{k})">{self._data(k)}</g>')
            out.append(f'<rect x="{LM}" y="{y0}" width="{W - LM - RM}" height="{y1 - y0}" fill="none" stroke="{INK}" stroke-width="1"/>')
            xl = p.x_label if self.own_x(p) else (self.x_label if self.labels_x[k] else None)
            if xl:
                out.append(f'<text x="{(LM + W - RM) / 2:.0f}" y="{y1 + 38}" text-anchor="middle" font-size="{FS}">{xl}</text>')
        out.append('<g class="ov"></g></svg>')
        return "".join(out)

    def data(self) -> dict:
        ser = []
        for s in self.series:
            v = {}
            for k, p in enumerate(self.panels):
                if p.key not in s.data:
                    continue
                d = s.data[p.key]
                e = {"m": delta_ints(d["med"], p.dec, p.ylog)}
                if "x" in d:
                    _, xl = self.xdom(k)
                    e["x"] = delta_ints(d["x"], p.x_dec if self.own_x(p) else self.x_dec, xl)
                if p.band and "lo" in d:
                    e["lo"], e["hi"] = delta_ints(d["lo"], p.dec, p.ylog), delta_ints(d["hi"], p.dec, p.ylog)
                if p.band and len(d.get("roll", [])):
                    e["r"] = [delta_ints(r, p.dec, p.ylog) for r in d["roll"]]
                if s.point_labels.get(p.key):
                    e["pl"] = [htmlmod.unescape(str(t)) for t in s.point_labels[p.key]]
                v[p.key] = e
            ser.append({"id": s.id, "l": htmlmod.unescape(s.label), "c": s.color, "w": s.width, "da": s.dash, "f": s.facets,
                        "mo": s.mode, "op": s.open, "v": v})
        panels = []
        for k, p in enumerate(self.panels):
            (xa, xb), xl = self.xdom(k)
            panels.append({"k": p.key, "t": htmlmod.unescape(p.title), "d": list(self.ydom(k)), "dec": p.dec, "b": p.band,
                           "yl": p.ylog, "xl": xl, "xd": [xa, xb], "own": self.own_x(p),
                           "xdec": p.x_dec if self.own_x(p) else self.x_dec,
                           "xt": htmlmod.unescape(p.x_label if self.own_x(p) else self.x_label), "lab": self.labels_x[k]})
        return {"x": delta_ints(self.x, self.x_dec, self.xlog) if self.x is not None else None, "xd": self.x_dec,
                "xlg": self.xlog, "xl": htmlmod.unescape(self.x_label), "xs": self.x_seconds,
                "p": panels, "s": ser, "fa": [{"k": f["key"], "t": htmlmod.unescape(f["label"]), "kind": f.get("kind", "check"),
                                               "v": [[v, htmlmod.unescape(lab)] for v, lab in f["values"]],
                                               "on": f.get("on", [v for v, _ in f["values"]])} for f in self.facets],
                "band": self.show_band, "roll": self.show_rollouts, "med": self.show_median, "lg": LOG_DEC,
                "lay": {"W": W, "L": LM, "R": RM, "fs": FS, "box": self.boxes, "last": len(self.panels) - 1}}

    def data_json(self) -> str:
        return json.dumps(self.data(), separators=(",", ":")).replace("</", "<\\/")

    def html(self) -> str:
        shown = {s.id for s in self.series if self.visible(s)}

        def key_svg(c, d, mode="line", op=False):
            if mode == "markers":
                return (f'<svg width="16" height="12" aria-hidden="true"><circle cx="8" cy="6" r="4.5" fill="{"#fff" if op else c}" '
                        f'stroke="{c}" stroke-width="1.5"/></svg>')
            return (f'<svg width="30" height="12" aria-hidden="true"><line x1="1" x2="29" y1="6" y2="6" stroke="{c}" stroke-width="2.5"'
                    f'{f" stroke-dasharray={chr(34)}{d}{chr(34)}" if d else ""}/></svg>')
        mode = {s.id: (s.mode, s.open) for s in self.series}
        leg = ""
        for lab, c, d, ids in self.legend:
            m, op = mode.get(ids[0], ("line", False)) if ids else ("line", False)
            hide = "" if (not ids or shown & set(ids)) else ' style="display:none"'
            leg += f'<span class="ifig-key" data-s="{",".join(ids)}"{hide}>{key_svg(c, d, m, op)}{lab}</span>'
        # vertical markers are keyed in the legend (a label inside the plot collides with the reference lines)
        vl = [lab for _, lab in self.vlines if lab] + [lab for p in self.panels for _, lab in p.vrefs if lab]
        leg += "".join(f'<span class="ifig-key"><svg width="14" height="16" aria-hidden="true"><line x1="7" x2="7" y1="1" y2="15" '
                       f'stroke="{MUTED}" stroke-width="1.5" stroke-dasharray="4 3"/></svg>{lab}</span>' for lab in dict.fromkeys(vl))
        note = f'<p class="ifig-note">{self.note}</p>' if self.note else ""
        return (f'<figure class="ifig" id="{self.fid}"><div class="ifig-legend">{leg}</div>{note}'
                f'<div class="chart">{self.svg()}</div>'
                f'<script type="application/json" class="ifig-data">{self.data_json()}</script>'
                f'<figcaption>{self.caption}</figcaption></figure>')


# ----------------------------------------------------------------------------------------------
# eval traces (scripts/policy_eval_suite.py) -> panels and series data
# ----------------------------------------------------------------------------------------------

def sample_index(T: int, npts: int = 200) -> np.ndarray:
    return np.unique(np.round(np.linspace(0, T - 1, min(npts, T))).astype(int))


def eval_panels(meta: dict, z_domain=None, grip_domain=None) -> list[Panel]:
    return [Panel("cos", "Signed cosine of the tool axis with vertical", (-1.0, 1.0), ((meta["align_thresh"], f"{meta['align_thresh']:g}"),), dec=3),
            Panel("z", "Tool height (mm)", z_domain, ((meta["floor_z"] * 1000, "floor limit"),), dec=0),
            Panel("grip", "Pad force sum (N)", grip_domain, ((meta["held_min_n"], ""),), dec=1),
            Panel("held", "Share of rollouts held", (0.0, 1.0), band=False, dec=2, height=90)]


def eval_data(tr: dict, idx: np.ndarray, n_roll: int = 0, which=("cos", "z", "grip", "held")) -> dict:
    """Median and 10-90 % band over the rollouts, the held share, and the first n_roll rollouts, at the steps idx."""
    m = tr["meta"]
    held = (tr["grip"] > m["held_min_n"]) & (tr["z"] > m["floor_z"])
    src = {"cos": tr["cos"], "z": tr["z"] * 1000.0, "grip": tr["grip"]}
    out = {}
    for k in which:
        if k == "held":
            out[k] = {"med": held.mean(axis=1)[idx]}
            continue
        a = src[k][idx].astype(float)
        out[k] = {"med": np.median(a, axis=1), "lo": np.percentile(a, 10, axis=1), "hi": np.percentile(a, 90, axis=1)}
        if n_roll:
            out[k]["roll"] = [a[:, i] for i in range(min(n_roll, a.shape[1]))]
    return out


def eval_figure(fid: str, items: list[dict], *, caption: str, facets=(), legend=None, n_roll: int = 8,
                show_band: bool = True, show_rollouts: bool = False, note: str = "", z_domain=None, grip_domain=None,
                which=("cos", "z", "grip", "held")) -> Figure:
    """One interactive figure of eval traces. items: dicts with id, label, trace (a key of load_traces()), color and
    optionally dash, facets, width. The x axis is the control step; a dashed vertical line marks the residual onset the
    first item's run trained with."""
    tr0 = items[0]["trace"]
    idx = sample_index(tr0["cos"].shape[0])
    series = [Series(it["id"], it["label"], it["color"], eval_data(it["trace"], idx, n_roll, which), dash=it.get("dash", ""),
                     facets=it.get("facets", {}), width=it.get("width", 2.0)) for it in items]
    m = tr0["meta"]
    onset = int(m.get("run_residual_from_step", m["residual_from_step"]))
    panels = [p for p in eval_panels(m, z_domain, grip_domain) if p.key in which]
    return Figure(fid, idx, "Control step (0.02 s)", panels, series, facets=facets, legend=legend, caption=caption,
                  vlines=[(onset, f"Residual on (step {onset}, as trained)")] if onset else [], show_band=show_band,
                  show_rollouts=show_rollouts, x_seconds=float(m.get("control_dt_s", 0.02)), note=note, x_dec=0)


# ----------------------------------------------------------------------------------------------
# page assets: one style block and one script per page
# ----------------------------------------------------------------------------------------------

CSS = """
figure.ifig{text-align:left}
.ifig-legend{display:flex;flex-wrap:wrap;gap:4px 16px;margin:0 0 6px;font-size:16px}
.ifig-key{display:inline-flex;align-items:center;gap:6px;white-space:nowrap}
.ifig-key svg{border:0!important;display:inline!important;margin:0!important}
.ifig-note{font-size:16px!important;margin:0 0 6px!important}
.ifig-ctl{display:flex;flex-wrap:wrap;gap:6px 18px;align-items:center;margin:0 0 8px;font-size:16px}
.ifig-ctl fieldset{border:0;margin:0;padding:0;display:flex;flex-wrap:wrap;gap:2px 10px;align-items:center}
.ifig-ctl legend{float:left;padding:0 6px 0 0;font-weight:bold}
.ifig-ctl label{white-space:nowrap;cursor:pointer}
.ifig-ctl button{font:inherit;font-size:15px;padding:1px 8px}
.ifig .chart{position:relative}
.ifig svg.ifig-svg{touch-action:none;cursor:crosshair}
.ifig svg.ifig-svg:focus{outline:none}
.ifig svg.ifig-svg:focus-visible{outline:1px dotted #000}
.ifig-tip{position:absolute;pointer-events:none;background:#fff;border:1px solid #000;padding:4px 8px;font-size:15px;
  line-height:1.3;white-space:nowrap;z-index:5;text-align:left}
.ifig-tip b{font-weight:bold}
.ifig-tip .h{color:#444;margin-bottom:2px}
@media print{.ifig-ctl,.ifig-tip{display:none!important}}
"""

JS = r"""
(function(){
"use strict";
var NS="http://www.w3.org/2000/svg";
function el(t,a,p){var e=document.createElementNS(NS,t);for(var k in a)e.setAttribute(k,a[k]);if(p)p.appendChild(e);return e;}
function h(t,txt,p){var e=document.createElement(t);if(txt!=null)e.textContent=txt;if(p)p.appendChild(e);return e;}
function clear(g){while(g.firstChild)g.removeChild(g.firstChild);}
function undelta(a,d,lg){if(!a)return null;var o=new Array(a.length),v=0,f=Math.pow(10,d);
  for(var i=0;i<a.length;i++){if(a[i]===null){o[i]=null;continue;}v+=a[i];o[i]=lg?Math.pow(10,v/f):v/f;}return o;}
function niceStep(span,n){var raw=Math.max(span,1e-12)/Math.max(n,1),mag=Math.pow(10,Math.floor(Math.log10(raw)));
  var ms=[1,2,2.5,5,10];for(var i=0;i<ms.length;i++)if(ms[i]*mag>=raw*(1-1e-9))return ms[i]*mag;return 10*mag;}
function ticks(lo,hi,n){var st=niceStep(hi-lo,n),t=Math.ceil(lo/st-1e-9)*st,out=[];
  while(t<=hi+1e-9*st){out.push(Math.round(t*1e10)/1e10);t+=st;}
  var mg=Math.pow(10,Math.floor(Math.log10(st))),dec=Math.max(0,-Math.floor(Math.log10(st)+1e-9))+(Math.abs(st/mg-2.5)<1e-6?1:0);
  return [out,dec];}
function logTicks(lo,hi){var a=Math.floor(Math.log10(lo)-1e-9),b=Math.ceil(Math.log10(hi)+1e-9),o=[],e,m;
  for(e=a;e<=b;e++){m=Math.pow(10,e);if(m>=lo*(1-1e-9)&&m<=hi*(1+1e-9))o.push(m);}if(o.length>=3)return o;o=[];
  for(e=a;e<=b;e++)[1,2,5].forEach(function(c){m=c*Math.pow(10,e);if(m>=lo*(1-1e-9)&&m<=hi*(1+1e-9))o.push(m);});return o;}
function fmt(v,d){var s=Math.abs(v).toLocaleString("en-US",{minimumFractionDigits:d,maximumFractionDigits:d});
  return (v<0&&parseFloat(s.replace(/,/g,""))!==0?"−":"")+s;}
function fmtLog(v){return fmt(v,Math.max(0,-Math.floor(Math.log10(v)+1e-9)));}
function sig(v){if(v==null||!isFinite(v))return "–";if(v===0)return "0";var d=Math.max(0,2-Math.floor(Math.log10(Math.abs(v))));return fmt(v,Math.min(d,6));}
function T(v,lg){return lg?Math.log10(v):v;}
function init(fig){
  var D=JSON.parse(fig.querySelector("script.ifig-data").textContent),svg=fig.querySelector("svg.ifig-svg"),L=D.lay,P=D.p,S=D.s;
  var X=D.x?undelta(D.x,D.xd,D.xlg):null;
  P.forEach(function(p){p.dom=p.xd.slice();p.full=p.xd.slice();});
  S.forEach(function(s){P.forEach(function(p){var v=s.v[p.k];if(!v)return;
    v.M=undelta(v.m,p.dec,p.yl);v.X=v.x?undelta(v.x,p.xdec,p.xl):X;
    if(v.lo){v.LO=undelta(v.lo,p.dec,p.yl);v.HI=undelta(v.hi,p.dec,p.yl);}
    if(v.r)v.R=v.r.map(function(r){return undelta(r,p.dec,p.yl);});});});
  var on={};D.fa.forEach(function(f){on[f.k]={};f.on.forEach(function(v){on[f.k][v]=1;});});
  var st={band:D.band,roll:D.roll,med:D.med};
  function vis(s){for(var k in on)if(s.f.hasOwnProperty(k)&&!on[k][s.f[k]])return false;return true;}
  function sx(v,k){var p=P[k],a=T(p.dom[0],p.xl),b=T(p.dom[1],p.xl);return L.L+(L.W-L.L-L.R)*(T(v,p.xl)-a)/(b-a);}
  function ix(px,k){var p=P[k],a=T(p.dom[0],p.xl),b=T(p.dom[1],p.xl),t=a+(px-L.L)/(L.W-L.L-L.R)*(b-a);return p.xl?Math.pow(10,t):t;}
  function sy(v,k){var p=P[k],b=L.box[k],lo=T(p.d[0],p.yl),hi=T(p.d[1],p.yl);return b[1]-(b[1]-b[0])*(T(v,p.yl)-lo)/(hi-lo);}
  function ok(x,y,k){var p=P[k];return x!=null&&y!=null&&isFinite(x)&&isFinite(y)&&(!p.xl||x>0)&&(!p.yl||y>0);}
  function path(xs,ys,k){var s="",pen="M";for(var i=0;i<ys.length;i++){if(!ok(xs[i],ys[i],k)){pen="M";continue;}
    s+=pen+sx(xs[i],k).toFixed(1)+","+sy(ys[i],k).toFixed(1);pen="L";}return s;}
  var gd=svg.querySelectorAll("g.d"),gx=svg.querySelectorAll("g.xa"),gv=svg.querySelectorAll("g.vl"),ov=svg.querySelector("g.ov");
  function xaxis(k){var p=P[k],g=gx[k],b=L.box[k];clear(g);
    var tv=p.xl?[logTicks(p.dom[0],p.dom[1]),null]:ticks(p.dom[0],p.dom[1],6);
    tv[0].forEach(function(t){var x=sx(t,k);el("line",{x1:x.toFixed(1),x2:x.toFixed(1),y1:b[0],y2:b[1],stroke:"#d9d9d9","stroke-width":"1"},g);
      if(p.lab){var e=el("text",{x:x.toFixed(1),y:b[1]+18,"text-anchor":"middle","font-size":L.fs},g);e.textContent=p.xl?fmtLog(t):fmt(t,tv[1]);}});}
  function redraw(){
    P.forEach(function(p,k){var g=gd[k];clear(g);var vs=S.filter(function(s){return vis(s)&&s.v[p.k];});
      if(p.b&&st.band)vs.forEach(function(s){var v=s.v[p.k];if(!v.LO)return;var up=[],dn=[];
        for(var i=0;i<v.X.length;i++){if(ok(v.X[i],v.HI[i],k))up.push(sx(v.X[i],k).toFixed(1)+","+sy(v.HI[i],k).toFixed(1));
          if(ok(v.X[i],v.LO[i],k))dn.push(sx(v.X[i],k).toFixed(1)+","+sy(v.LO[i],k).toFixed(1));}
        if(up.length&&dn.length)el("path",{d:"M"+up.concat(dn.reverse()).join("L")+"Z",fill:s.c,"fill-opacity":"0.18",stroke:"none"},g);});
      if(p.b&&st.roll)vs.forEach(function(s){var v=s.v[p.k];(v.R||[]).forEach(function(r){
        el("path",{d:path(v.X,r,k),fill:"none",stroke:s.c,"stroke-opacity":"0.45","stroke-width":"1"},g);});});
      vs.forEach(function(s){var v=s.v[p.k];
        if((s.mo==="line"||s.mo==="both")&&(st.med||!p.b)){var a={d:path(v.X,v.M,k),fill:"none",stroke:s.c,"stroke-width":s.w,"stroke-linejoin":"round"};
          if(s.da)a["stroke-dasharray"]=s.da;el("path",a,g);}
        if(s.mo==="markers"||s.mo==="both")for(var i=0;i<v.M.length;i++)if(ok(v.X[i],v.M[i],k))
          el("circle",{cx:sx(v.X[i],k).toFixed(1),cy:sy(v.M[i],k).toFixed(1),r:"4.5",fill:s.op?"#fff":s.c,stroke:s.op?s.c:"#fff","stroke-width":s.op?"1.8":"1.2"},g);});
      xaxis(k);
      gv[k].querySelectorAll("[data-x]").forEach(function(e){var x=sx(parseFloat(e.getAttribute("data-x")),k).toFixed(1);e.setAttribute("x1",x);e.setAttribute("x2",x);});});
    legend();}
  var keys=fig.querySelectorAll(".ifig-key[data-s]"),byId={};S.forEach(function(s){byId[s.id]=s;});
  function legend(){keys.forEach(function(e){var ids=e.getAttribute("data-s").split(",").filter(Boolean);if(!ids.length)return;
    e.style.display=ids.some(function(i){return byId[i]&&vis(byId[i]);})?"":"none";});}
  // controls
  var ctl=h("div",null);ctl.className="ifig-ctl";fig.insertBefore(ctl,fig.querySelector(".ifig-legend"));
  D.fa.forEach(function(f){var fs=h("fieldset",null,ctl);h("legend",f.t,fs);
    f.v.forEach(function(v){var lab=h("label",null,fs),i=h("input",null,lab);i.type=f.kind==="radio"?"radio":"checkbox";
      i.name=fig.id+"-"+f.k;i.checked=!!on[f.k][v[0]];i.value=v[0];lab.appendChild(document.createTextNode(" "+v[1]));
      i.addEventListener("change",function(){if(f.kind==="radio")on[f.k]={};on[f.k][v[0]]=i.checked?1:0;redraw();});});});
  var hasBand=S.some(function(s){return P.some(function(p){return p.b&&s.v[p.k]&&s.v[p.k].lo;});});
  var hasRoll=S.some(function(s){return P.some(function(p){return s.v[p.k]&&s.v[p.k].r;});});
  if(hasBand){var fs=h("fieldset",null,ctl);h("legend","Show",fs);
    [["med","Median"],["band","10–90 % band"]].concat(hasRoll?[["roll","Single rollouts"]]:[]).forEach(function(o){
      var lab=h("label",null,fs),i=h("input",null,lab);i.type="checkbox";i.checked=!!st[o[0]];i.value=o[0];
      lab.appendChild(document.createTextNode(" "+o[1]));i.addEventListener("change",function(){st[o[0]]=i.checked;redraw();});});}
  var rb=h("button","Reset zoom",ctl);rb.type="button";rb.className="ifig-reset";
  rb.addEventListener("click",function(){P.forEach(function(p){p.dom=p.full.slice();});redraw();});
  var cb=h("button","Download CSV",ctl);cb.type="button";cb.className="ifig-csv";cb.addEventListener("click",function(){
    var q=function(t){return '"'+String(t).replace(/"/g,"'")+'"';},rows=["series,label,panel,stat,x,value,point"];
    S.forEach(function(s){P.forEach(function(p){var v=s.v[p.k];if(!v)return;
      var parts=[[p.b&&v.LO?"median":"value",v.M]];if(v.LO){parts.push(["p10",v.LO]);parts.push(["p90",v.HI]);}
      (v.R||[]).forEach(function(r,j){parts.push(["rollout "+(j+1),r]);});
      parts.forEach(function(c){for(var i=0;i<c[1].length;i++)rows.push([s.id,q(s.l),p.k,c[0],v.X[i],c[1][i],v.pl?q(v.pl[i]):""].join(","));});});});
    var a=document.createElement("a");a.href=URL.createObjectURL(new Blob([rows.join("\n")+"\n"],{type:"text/csv"}));
    a.download=fig.id+".csv";document.body.appendChild(a);a.click();setTimeout(function(){URL.revokeObjectURL(a.href);a.remove();},1000);});
  // hover readout, keyboard, drag-zoom
  var tip=h("div",null,fig.querySelector(".chart"));tip.className="ifig-tip";tip.style.display="none";
  var cross=null,sel=null,drag=null,cur=null;
  function pt(e){var m=svg.getScreenCTM().inverse(),q=svg.createSVGPoint();q.x=e.clientX;q.y=e.clientY;return q.matrixTransform(m);}
  function panelAt(y){for(var k=0;k<L.box.length;k++)if(y>=L.box[k][0]-L.fs&&y<=L.box[k][1]+4)return k;return -1;}
  function near(xs,xv,k){var b=-1,bd=1e18,p=P[k];for(var i=0;i<xs.length;i++){if(xs[i]==null||xs[i]<p.dom[0]||xs[i]>p.dom[1])continue;
    var d=Math.abs(T(xs[i],p.xl)-T(xv,p.xl));if(d<bd){bd=d;b=i;}}return b;}
  function xfmt(v,p){return p.xl?sig(v):fmt(v,p.xdec);}
  function key(s,parent){var k=document.createElementNS(NS,"svg");k.setAttribute("width","22");k.setAttribute("height","10");
    if(s.mo==="markers")el("circle",{cx:11,cy:5,r:4,fill:s.op?"#fff":s.c,stroke:s.c,"stroke-width":"1.5"},k);
    else el("line",{x1:1,x2:21,y1:5,y2:5,stroke:s.c,"stroke-width":"2.5","stroke-dasharray":s.da||""},k);parent.appendChild(k);}
  function place(x,y){tip.style.display="block";var cb=svg.getBoundingClientRect(),hb=fig.querySelector(".chart").getBoundingClientRect(),sc=cb.width/L.W;
    var left=cb.left-hb.left+x*sc+14,top=cb.top-hb.top+y*sc+14;
    if(left+tip.offsetWidth>hb.width)left=Math.max(0,cb.left-hb.left+x*sc-14-tip.offsetWidth);
    if(top+tip.offsetHeight>hb.height)top=Math.max(0,cb.top-hb.top+y*sc-14-tip.offsetHeight);
    tip.style.left=left+"px";tip.style.top=top+"px";}
  function hide(){cur=null;tip.style.display="none";if(cross)cross.remove();cross=null;}
  function show(k,xv,yv,key2){var p=P[k],vs=S.filter(function(s){return vis(s)&&s.v[p.k];});if(!vs.length){hide();return;}
    if(!cross)cross=el("g",{},ov);clear(cross);clear(tip);
    var mk=vs.some(function(s){return s.mo!=="line";});
    if(mk){var best=null,bd=1e18,py=yv==null?(L.box[k][0]+L.box[k][1])/2:yv;
      vs.forEach(function(s){var v=s.v[p.k];for(var i=0;i<v.M.length;i++){if(!ok(v.X[i],v.M[i],k)||v.X[i]<p.dom[0]||v.X[i]>p.dom[1])continue;
        var dx=sx(v.X[i],k)-sx(xv,k),dy=sy(v.M[i],k)-py,d=key2?Math.abs(dx):dx*dx+dy*dy;if(d<bd){bd=d;best=[s,i];}}});
      if(!best){hide();return;}var s=best[0],v=s.v[p.k],i=best[1];cur={k:k,x:v.X[i]};
      el("circle",{cx:sx(v.X[i],k),cy:sy(v.M[i],k),r:"8",fill:"none",stroke:"#000","stroke-width":"1.5"},cross);
      var hd=h("div",null,tip);hd.className="h";key(s,hd);hd.appendChild(document.createTextNode(" "+s.l+(v.pl?" · "+v.pl[i]:"")));
      var r=h("div",null,tip);h("b",sig(v.M[i]),r);r.appendChild(document.createTextNode(" "+p.t+" at "+xfmt(v.X[i],p)+" ("+p.xt+")"));
      place(sx(v.X[i],k),sy(v.M[i],k));return;}
    cur={k:k,x:xv};var x=sx(xv,k);
    P.forEach(function(q,j){if(j===k||(!p.own&&!q.own))el("line",{x1:x,x2:x,y1:L.box[j][0],y2:L.box[j][1],stroke:"#000","stroke-width":"1"},cross);});
    var hd=h("div",(p.xt||"").replace(/ \(.*$/,"")+" "+xfmt(xv,p)+(D.xs&&!p.own?" · "+(xv*D.xs).toFixed(2)+" s":"")+" · "+p.t,tip);hd.className="h";
    vs.forEach(function(s){var v=s.v[p.k],i=near(v.X,xv,k);if(i<0)return;var r=h("div",null,tip);key(s,r);
      h("b"," "+(p.yl?sig(v.M[i]):fmt(v.M[i],p.dec)),r);
      if(v.LO&&st.band)r.appendChild(document.createTextNode(" ("+(p.yl?sig(v.LO[i]):fmt(v.LO[i],p.dec))+" to "+(p.yl?sig(v.HI[i]):fmt(v.HI[i],p.dec))+")"));
      r.appendChild(document.createTextNode("  "+s.l));});
    place(x,yv==null?L.box[k][0]:yv);}
  function xs0(k){var p=P[k],o=[];S.forEach(function(s){if(vis(s)&&s.v[p.k])s.v[p.k].X.forEach(function(x){if(x!=null&&x>=p.dom[0]&&x<=p.dom[1])o.push(x);});});
    return o.sort(function(a,b){return a-b;});}
  svg.addEventListener("pointermove",function(e){var q=pt(e);
    if(drag){var x=Math.max(L.L,Math.min(L.W-L.R,q.x)),b0=L.box[drag.k][0],b1=L.box[drag.k][1];
      if(!P[drag.k].own){b0=L.box[0][0];b1=L.box[L.last][1];}
      if(!sel)sel=el("rect",{y:b0,height:b1-b0,fill:"#000","fill-opacity":"0.08",stroke:"#000","stroke-width":"1"},ov);
      sel.setAttribute("x",Math.min(drag.x,x));sel.setAttribute("width",Math.abs(x-drag.x));}
    var k=panelAt(q.y);if(k<0||q.x<L.L||q.x>L.W-L.R){if(!drag)hide();return;}show(k,ix(q.x,k),q.y);});
  svg.addEventListener("pointerleave",function(){if(!drag)hide();});
  svg.addEventListener("pointerdown",function(e){var q=pt(e),k=panelAt(q.y);if(q.x<L.L||q.x>L.W-L.R||k<0)return;
    drag={x:q.x,k:k};svg.setPointerCapture(e.pointerId);});
  svg.addEventListener("pointerup",function(e){if(drag==null)return;var q=pt(e),x=Math.max(L.L,Math.min(L.W-L.R,q.x)),k=drag.k;
    if(Math.abs(x-drag.x)>6){var a=ix(Math.min(drag.x,x),k),b=ix(Math.max(drag.x,x),k);
      P.forEach(function(p,j){if(j===k||(!P[k].own&&!p.own))p.dom=[a,b];});redraw();}
    drag=null;if(sel){sel.remove();sel=null;}});
  svg.addEventListener("dblclick",function(){P.forEach(function(p){p.dom=p.full.slice();});redraw();});
  svg.addEventListener("keydown",function(e){var k=cur?cur.k:0;
    if(e.key==="ArrowRight"||e.key==="ArrowLeft"){var xs=xs0(k);if(!xs.length)return;var x;
      if(!cur)x=xs[0];else if(e.key==="ArrowRight"){x=xs.filter(function(v){return v>cur.x+1e-12;})[0];if(x==null)x=xs[xs.length-1];}
      else{var lo=xs.filter(function(v){return v<cur.x-1e-12;});x=lo.length?lo[lo.length-1]:xs[0];}
      show(k,x,null,true);e.preventDefault();}
    else if((e.key==="ArrowDown"||e.key==="ArrowUp")&&cur){var j=Math.max(0,Math.min(P.length-1,k+(e.key==="ArrowDown"?1:-1)));
      show(j,Math.max(P[j].dom[0],Math.min(P[j].dom[1],cur.x)),null,true);e.preventDefault();}
    else if(e.key==="Escape")hide();});
  fig.ifig={redraw:redraw,show:show,state:st,on:on,panels:P};
}
function start(){document.querySelectorAll("figure.ifig").forEach(function(f){try{init(f);}catch(err){if(window.console)console.error(err);}});}
if(document.readyState==="loading")document.addEventListener("DOMContentLoaded",start);else start();
})();
"""


def page_assets() -> str:
    """Style and script for every interactive figure on a page; insert once, before </body>."""
    return f'<style id="ifig-style">{CSS.strip()}</style>\n<script id="ifig-script">{JS.strip()}</script>\n'


def add_assets(page: str) -> str:
    if 'id="ifig-script"' in page or "figure class=\"ifig\"" not in page:
        return page
    i = page.lower().rfind("</body>")
    return page[:i] + page_assets() + page[i:] if i >= 0 else page + page_assets()


def data_bytes(page: str) -> int:
    import re
    return sum(len(m) for m in re.findall(r'<script type="application/json" class="ifig-data">(.*?)</script>', page, re.S))


# ----------------------------------------------------------------------------------------------
# --exercise: drive each control of every interactive figure in headless Chrome (CDP over websockets)
# ----------------------------------------------------------------------------------------------

def exercise(page: str, out: str, width: int = 900) -> list[dict]:
    import asyncio
    import shutil
    import socket
    import subprocess
    import tempfile
    import urllib.request

    import websockets

    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    prof = tempfile.mkdtemp(prefix="ifig-cdp-")
    dl = os.path.join(out, "downloads")
    shutil.rmtree(dl, ignore_errors=True)            # a file left by an earlier run would be overwritten, not added
    os.makedirs(dl, exist_ok=True)
    proc = subprocess.Popen([chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--remote-debugging-port={port}",
                             f"--user-data-dir={prof}", f"--window-size={width},1400", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    report = []

    async def run():
        for _ in range(100):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list"))
                ws_url = next(t["webSocketDebuggerUrl"] for t in tabs if t["type"] == "page")
                break
            except Exception:
                await asyncio.sleep(0.1)
        async with websockets.connect(ws_url, max_size=2 ** 28) as ws:
            n = [0]

            async def cmd(method, **params):
                n[0] += 1
                mid = n[0]
                await ws.send(json.dumps({"id": mid, "method": method, "params": params}))
                while True:
                    msg = json.loads(await ws.recv())
                    if msg.get("id") == mid:
                        if "error" in msg:
                            raise RuntimeError(f"{method}: {msg['error']}")
                        return msg.get("result", {})

            async def js(expr):
                r = await cmd("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)
                return r.get("result", {}).get("value")

            async def shot(name, fid):
                rect = await js(f"(function(){{var r=document.getElementById('{fid}').getBoundingClientRect();"
                                f"return [r.left,r.top+window.scrollY,r.width,r.height];}})()")
                # within the viewport: a capture beyond it re-lays out the page, which fires pointerleave and hides the readout
                r = await cmd("Page.captureScreenshot", format="png", captureBeyondViewport=False,
                              clip={"x": rect[0], "y": rect[1], "width": rect[2], "height": rect[3] + 4, "scale": 1})
                import base64
                p = os.path.join(out, f"{fid}_{name}.png")
                open(p, "wb").write(base64.b64decode(r["data"]))
                return p

            async def mouse(kind, x, y, buttons=0):
                await cmd("Input.dispatchMouseEvent", type=kind, x=x, y=y, button="left" if kind != "mouseMoved" or buttons else "none",
                          buttons=buttons, clickCount=1 if kind in ("mousePressed", "mouseReleased") else 0)

            async def click_sel(sel):
                r = await js(f"(function(){{var e=document.querySelector({json.dumps(sel)});if(!e)return null;e.scrollIntoView({{block:'center'}});"
                             f"var r=e.getBoundingClientRect();return [r.left+r.width/2,r.top+r.height/2];}})()")
                if r is None:
                    return False
                await mouse("mousePressed", r[0], r[1], 1)
                await mouse("mouseReleased", r[0], r[1], 0)
                await asyncio.sleep(0.15)
                return True

            await cmd("Page.enable")
            await cmd("Runtime.enable")
            await cmd("Browser.setDownloadBehavior", behavior="allow", downloadPath=os.path.abspath(dl))
            await cmd("Page.navigate", url="file://" + os.path.abspath(page))
            while True:
                msg = json.loads(await ws.recv())
                if msg.get("method") == "Page.loadEventFired":
                    break
            await asyncio.sleep(0.5)
            figs = await js("Array.from(document.querySelectorAll('figure.ifig')).map(function(f){return f.id;})")
            errs = await js("(function(){return Array.from(document.querySelectorAll('figure.ifig')).filter(function(f){return !f.ifig;}).map(function(f){return f.id;});})()")
            for fid in figs:
                R = {"figure": fid, "initialised": fid not in (errs or []), "shots": []}
                q = f"document.getElementById('{fid}')"
                await js(f"{q}.scrollIntoView({{block:'start'}})")
                await asyncio.sleep(0.1)
                R["shots"].append(await shot("1_default", fid))
                paths0 = await js(f"{q}.querySelectorAll('g.d path').length")
                # hover: the middle of the first panel
                box = await js(f"(function(){{var s={q}.querySelector('svg.ifig-svg'),r=s.getBoundingClientRect(),d=JSON.parse({q}.querySelector('script.ifig-data').textContent).lay,k=r.width/d.W;"
                               f"return [r.left,r.top,k,d.L,d.W-d.R,d.box[0][0],d.box[0][1],d.box[d.last][1]];}})()")
                left, top, k, x0, x1, y0, y1, ylast = box
                hx, hy = left + k * (x0 + 0.62 * (x1 - x0)), top + k * (y0 + 0.5 * (y1 - y0))
                await mouse("mouseMoved", hx, hy)
                await asyncio.sleep(0.2)
                R["tooltip"] = await js(f"(function(){{var t={q}.querySelector('.ifig-tip');return t&&t.style.display!=='none'?t.textContent:null;}})()")
                R["shots"].append(await shot("2_hover", fid))
                await mouse("mouseMoved", left - 20, top - 20)
                # series toggle: untick the first checkbox of the first facet
                ok = await click_sel(f"#{fid} .ifig-ctl fieldset input")
                paths1 = await js(f"{q}.querySelectorAll('g.d path').length")
                R["toggle"] = {"clicked": ok, "paths_before": paths0, "paths_after": paths1}
                R["shots"].append(await shot("3_toggle", fid))
                await click_sel(f"#{fid} .ifig-ctl fieldset input")              # back on
                # band and rollouts
                for v in ("band", "roll"):
                    sel = f"#{fid} .ifig-ctl input[value='{v}']"
                    if await js(f"!!document.querySelector({json.dumps(sel)})"):
                        await click_sel(sel)
                        R[f"{v}_paths"] = await js(f"{q}.querySelectorAll('g.d path').length")
                        R["shots"].append(await shot(f"4_{v}", fid))
                        await click_sel(sel)
                # drag-zoom over the middle third of the bottom panel's x range
                await js(f"{q}.scrollIntoView({{block:'start'}})")
                await asyncio.sleep(0.1)
                box = await js(f"(function(){{var s={q}.querySelector('svg.ifig-svg'),r=s.getBoundingClientRect();return [r.left,r.top];}})()")
                left, top = box
                ya = top + k * (y0 + 0.5 * (y1 - y0))
                xa, xb = left + k * (x0 + 0.33 * (x1 - x0)), left + k * (x0 + 0.66 * (x1 - x0))
                ticks0 = await js(f"Array.from({q}.querySelectorAll('g.xa text')).map(function(t){{return t.textContent;}}).join(' ')")
                await mouse("mousePressed", xa, ya, 1)
                for t in np.linspace(0, 1, 6)[1:]:
                    await mouse("mouseMoved", xa + t * (xb - xa), ya, 1)
                await mouse("mouseReleased", xb, ya, 0)
                await asyncio.sleep(0.2)
                ticks1 = await js(f"Array.from({q}.querySelectorAll('g.xa text')).map(function(t){{return t.textContent;}}).join(' ')")
                R["zoom"] = {"ticks_before": ticks0, "ticks_after": ticks1}
                R["shots"].append(await shot("5_zoom", fid))
                await click_sel(f"#{fid} .ifig-reset")
                R["reset_ticks"] = await js(f"Array.from({q}.querySelectorAll('g.xa text')).map(function(t){{return t.textContent;}}).join(' ')")
                R["shots"].append(await shot("6_reset", fid))
                before = set(os.listdir(dl))
                await click_sel(f"#{fid} .ifig-csv")
                for _ in range(30):
                    new = [f for f in set(os.listdir(dl)) - before if not f.endswith(".crdownload")]
                    if new:
                        break
                    await asyncio.sleep(0.1)
                R["csv"] = None
                if new:
                    p = os.path.join(dl, new[0])
                    lines = open(p).read().splitlines()
                    R["csv"] = {"file": p, "rows": len(lines) - 1, "header": lines[0]}
                report.append(R)

    try:
        asyncio.run(run())
    finally:
        proc.terminate()
        proc.wait(timeout=10)
        shutil.rmtree(prof, ignore_errors=True)
    json.dump(report, open(os.path.join(out, "exercise.json"), "w"), indent=1)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--exercise", nargs="+", metavar="PAGE")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs",
                                                  time.strftime("%Y%m%d") + "-ifig_exercise"))
    a = ap.parse_args()
    for page in a.exercise or []:
        out = os.path.join(a.out, os.path.splitext(os.path.basename(page))[0])
        os.makedirs(out, exist_ok=True)
        html = open(page, encoding="utf-8").read()
        print(f"{page}: {len(html) / 1e6:.2f} MB, figure data {data_bytes(html) / 1e3:.0f} kB")
        for r in exercise(page, out):
            t = r["toggle"]
            print(f"  {r['figure']}: init {r['initialised']}; hover {'ok' if r.get('tooltip') else 'NO TOOLTIP'}; toggle paths "
                  f"{t['paths_before']} -> {t['paths_after']}; band paths {r.get('band_paths')}; rollout paths {r.get('roll_paths')}; "
                  f"zoom ticks '{r['zoom']['ticks_before']}' -> '{r['zoom']['ticks_after']}', reset '{r['reset_ticks']}'; "
                  f"csv {r['csv']['rows'] if r['csv'] else 'NONE'} rows")
        print(f"  screenshots in {out}")


if __name__ == "__main__":
    main()
