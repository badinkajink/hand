#!/usr/bin/env python3
"""LaTeX and TikZ figures of the co-design formulation for the hand-design talk (owner, 2026-11-10: "beautiful, didactic
diagrams"; "an evolving latex formula/equation that captures the co-design problem"; the SR2 part "hierarchical, rather
than cyclical"; HOM "even simpler/clearer"). Each figure is a standalone LaTeX file compiled with pdflatex and rasterized
by pdftoppm at 400 dpi into docs/experiments/20261110-codesign_formulation/media/:

  20261110-eq_cd{01..09}.png       the co-design problem, one term more per slide; one canvas and one baseline for all
  20261110-eq_final_{outer,inner,dyn}.png   the final form with one level in colour
  20261110-eq_{metric,outer,inner,sim}{k}.png   the metric's and the levels' formulas, one series each
  20261110-nested_{all,outer,inner,dyn}.png the nested problem as nested boxes, whole and with one level highlighted
  20261110-nested_loop_{finite,infinite,metric}.png   the boxes with the tasks, object sizes and metrics above them as the
                                   outermost loop
  20261110-contact_modes.png        what HOM commands at each fingertip: stick, roll, slide, press
  20261110-hom_period.png           one HOM control period
  20261110-hand_{mounts,joints,layout}.png the hand with its 6 mount coordinates, its 9 joints or its 2 layout
                                   distances coloured (steps 3 and 4, the morphology section)
  20261110-bd_rule.png             Borras and Dollar's rule as a schematic (slides 24-25), with eq_thesis{1,2}

    .venv/bin/python scripts/docs/codesign_tikz.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/experiments/20261110-codesign_formulation/media"
DPI = 400

PRE = r"""\documentclass[border=%s]{standalone}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb,xcolor}
\usepackage[scaled=0.93]{helvet}
\renewcommand{\familydefault}{\sfdefault}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,calc,decorations.markings,fit,backgrounds,shapes.misc}
\definecolor{ink}{HTML}{1F2937}\definecolor{grey}{HTML}{9CA3AF}\definecolor{pale}{HTML}{F3F4F6}
\definecolor{new}{HTML}{7C3AED}\definecolor{outer}{HTML}{2563EB}\definecolor{inner}{HTML}{059669}
\definecolor{sim}{HTML}{D97706}\definecolor{outerl}{HTML}{DBEAFE}\definecolor{innerl}{HTML}{D1FAE5}
\definecolor{siml}{HTML}{FEF3C7}
\newcommand{\lab}[1]{\text{\sffamily #1}}
\newcommand{\clab}[1]{\text{\makebox[0pt]{\sffamily #1}}}
\begin{document}
"""


def compile_tex(name, body, border="14pt"):
    """body (LaTeX inside document) -> OUT/name.png at DPI."""
    tmp = Path(tempfile.mkdtemp(prefix="cdtikz_"))
    (tmp / "f.tex").write_text(PRE % border + body + "\n\\end{document}\n")
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "f.tex"], cwd=tmp,
                       capture_output=True, text=True)
    if r.returncode != 0:
        err = [l for l in r.stdout.splitlines() if l.startswith("!") or "Error" in l][:6]
        raise RuntimeError(f"{name}: " + " | ".join(err))
    subprocess.run(["pdftoppm", "-r", str(DPI), "-png", "-singlefile", "f.pdf", "f"], cwd=tmp, check=True)
    shutil.copy(tmp / "f.png", OUT / f"{name}.png")
    shutil.rmtree(tmp)
    return OUT / f"{name}.png"


def same_canvas(paths, size=None):
    """Pad the PNGs to one canvas, centred, so a slide sequence keeps its size and position. With size, the canvas is
    that many pixels, so a series keeps the canvas of images already pasted into slides; a wider image loses white
    margin only."""
    ims = [Image.open(p).convert("RGB") for p in paths]
    w, h = size or (max(i.width for i in ims), max(i.height for i in ims))
    for p, im in zip(paths, ims):
        if im.width > w or im.height > h:
            l, t = max(0, (im.width - w) // 2), max(0, (im.height - h) // 2)
            box = (l, t, l + min(w, im.width), t + min(h, im.height))
            ink = ImageOps.invert(im.convert("L")).getbbox()
            if ink[0] < box[0] or ink[1] < box[1] or ink[2] > box[2] or ink[3] > box[3]:
                raise RuntimeError(f"{p}: the formula is wider or taller than the canvas {w} x {h}")
            im = im.crop(box)
        c = Image.new("RGB", (w, h), "white")
        c.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
        c.save(p)


# ------------------------------------------------------------------------------------------------ the formula build
# Owner, 2026-11-10: "make sure that we are incrementally, atomically building up our mathematical/conceptual
# formalisms. less is more". Every image adds one idea and carries no text but the formula and its brace. A series
# shares one canvas and one baseline (a strut of fixed height and depth), so its formula grows in place.
STRUT = r"\rule[-1.45cm]{0pt}{2.3cm}"
N = r"\color{new}"
MANIFEST = []


def eq_img(name, tex, series, what, slide="", strut=True, size=21):
    body = (r"\fontsize{%d}{%d}\selectfont$\displaystyle " % (size, size + 5) + (STRUT if strut else "") + tex + "$")
    MANIFEST.append(dict(name=name, series=series, slide=slide, what=what))
    return compile_tex(name, body, border="{90pt 16pt}")


def C(body, label):
    """A new term in the highlight colour with its brace label."""
    return r"{" + N + r"\underbrace{" + body + r"}_{\clab{" + label + r"}}}"


CD_CANVAS = (3556, 564)  # px, the canvas of the build's images since 11-10 noon: steps already pasted keep their scale
CD = [  # (slide, what, LaTeX); owner, 2026-11-10 afternoon: after the bi-level step the 6 mount coordinates, then the 9
    # joint commands, and only then the object sizes and the tasks. Evening: the score is subject to the dynamics, real or
    # simulated (the simulator's c, p and s move to the dynamics section), and the grasp g and HOM's parameters phi
    # first appear with HOM (series "inner"). Later that evening: the metric after the dynamics, since its space, large or
    # infinite, is introduced there
    ("21", "The best hand for a task maximizes the task's score.",
     r"m^\star_T=\arg\max_{m}\ \underbrace{J_T(m)}_{\clab{score of hand $m$ on task $T$}}"),
    ("23", "A hand is scored through its best controller: a bi-level problem.",
     r"m^\star_T=\arg\max_{m}\ " + C(r"\max_{\pi}", "best controller") + r"\ J_T(m,{" + N + r"\pi})"),
    ("28", "The hand: six mount coordinates.",
     r"m^\star_T=" + C(r"\arg\max_{m\in\mathbb{R}^{6}}", "6 mount coordinates") + r"\ \max_{\pi}\ J_T(m,\pi)"),
    ("28", "The controller: nine joint commands every 10 ms.",
     r"m^\star_T=\arg\max_{m\in\mathbb{R}^{6}}\ " + C(r"\max_{\pi:\,x\mapsto u\in\mathbb{R}^{9}}", "9 joint commands")
     + r"\ J_T(m,\pi)"),
    ("24-27", "One hand for a range of object sizes.",
     r"m^\star_T=\arg\max_{m\in\mathbb{R}^{6}}\ " + C(r"\mathbb{E}_{d}", "object sizes")
     + r"\Big[\max_{\pi:\,x\mapsto u\in\mathbb{R}^{9}}\ J_T(m,\pi;{" + N + r"d})\Big]"),
    ("29", "Tasks vary too: the three toy tasks sample a distribution.",
     r"m^\star=\arg\max_{m\in\mathbb{R}^{6}}\ " + C(r"\mathbb{E}_{T,d}", "tasks and sizes") + r"\Big[\max_{\pi}\ J_{" + N + r"T}(m,\pi;d)\Big]"),
    ("30", "The score is computed on a rollout, and the rollout obeys the dynamics of hand, object and world.",
     r"m^\star=\arg\max_{m\in\mathbb{R}^{6}}\ \mathbb{E}_{T,d}\Big[\max_{\pi}\ J_T({" + N + r"x_{0:H}})\ \ "
     + C(r"\mathrm{s.t.}\ \ x_{t+1}=f(x_t,u_t;\,m,d)", "dynamics, real or simulated") + r"\Big]"),
    ("31", "The score itself is a choice.",
     r"m^\star=\arg\max_{m\in\mathbb{R}^{6}}\ \mathbb{E}_{T,d}\Big[\max_{\pi}\ " + C(r"J_T(x_{0:H})", "a choice of metric")
     + r"\ \ \mathrm{s.t.}\ \ x_{t+1}=f(x_t,u_t;\,m,d)\Big]"),
]
LEVEL_COLOUR = {"outer": "outer", "inner": "inner", "dyn": "sim"}


def final(active=None):
    """The final form; active in (None, 'outer', 'inner', 'dyn'): None colours all three braces."""
    col = {k: (c if active in (None, k) else "grey") for k, c in LEVEL_COLOUR.items()}
    return (r"m^\star={\color{" + col["outer"] + r"}\underbrace{\arg\max_{m\in\mathbb{R}^{6}}}_{\clab{morphology "
            r"optimization}}}\qquad\mathbb{E}_{T,d}\Big[\;{\color{" + col["inner"]
            + r"}\underbrace{\max_{\pi}}_{\clab{multi-hand control}}}\ \ J_T(x_{0:H})\ \ {\color{" + col["dyn"]
            + r"}\underbrace{\mathrm{s.t.}\ \ x_{t+1}=f(x_t,u_t;\,m,d)}_{\clab{dynamics}}}\Big]")


THESIS = [  # (what, LaTeX); the rule under test on slides 24-25, and the question
    ("Borras and Dollar (2015): the workspace is largest when the palm is one finger length wider than the object.",
     r"D_{\mathrm{palm}}\ \approx\ d\ +\ " + C(r"L", "one finger length")),
    ("Does the rule hold when the manipulation is executed?",
     r"\frac{\mathrm{d}D^\star_{\mathrm{palm}}}{\mathrm{d}d}\ \overset{?}{=}\ 1\quad\text{for}\quad"
     + C(r"\max_{\pi}\ J_T(x_{0:H})\ \ \mathrm{s.t.}\ \ x_{t+1}=f(x_t,u_t;\,m,d)", "executed in-hand manipulation")),
]

PADS = [  # (what, LaTeX); owner, 2026-11-10 night: the sphere-packed contact model as the segue of the dynamics section
    ("Hydroelastic contact: a pressure field from the overlap depth, summed over the patch.",
     r"F=\int_{A}p\;\mathrm{d}A,\qquad p=E\,\frac{\delta}{R}"),
    ("Sphere pads: the same law as a sum of springs, one per 1 mm sphere.",
     r"F\approx" + C(r"\sum_i k_i\,\delta_i", "one spring per 1 mm sphere") + r",\qquad k_i=\frac{E\,a_i}{h_i}"),
]

MORPH = [  # (what, LaTeX); owner, 2026-11-10 evening: the morphology section with the co-design build's incrementalism.
    # The search's hands are symmetric tripods: two distances on the 5 mm lattice of the stored grids (25 x 25 = 625)
    ("A hand of the search: two distances on a 5 mm lattice, 625 hands.",
     r"m=" + C(r"(s_1,\,s_2)", "thumb to pair, index to middle") + r"\ \in\ \{40,45,\dots,160\}\times\{50,55,\dots,170\}\ \mathrm{mm}"),
    ("Every hand on the lattice is simulated: a landscape of scores.",
     r"m=(s_1,\,s_2)\ \mapsto\ " + C(r"J_T(s_1,s_2;\,d)", "every hand, simulated")),
    ("Each task and size has its best hand.",
     r"m^\star_T(d)=" + C(r"\arg\max_{s_1,\,s_2}", "best hand per task and size") + r"\ J_T(s_1,s_2;\,d)"),
    ("Does the best hand follow Borras and Dollar's rule?",
     r"\mathrm{d}s_1^\star/\mathrm{d}d\ \ \overset{?}{=}\ \ " + C(r"1", "Borràs and Dollar (2015)")),
    ("The fewest hands that reach a share alpha of every task's best.",
     C(r"K^\star(\alpha)", "fewest hands") + r"=\min_{\mathcal{M}}\ |\mathcal{M}|\quad\mathrm{s.t.}\quad\max_{m\in\mathcal{M}}J_T(m)"
     r"\ \ge\ \alpha\,\max_{m}J_T(m)\quad\forall\,T"),
]

OUTER = [  # (column label, line); the series shows lines 1..k, line k in colour, the rest held as phantoms
    ("fitness", r"F(m)=\frac{1}{3}\sum_{s=1}^{3}\ \max_{\pi}\ J(m,\pi;\,s)", "A hand's fitness: its best controller's score, averaged over three seeds."),
    ("population", r"P_t=\{m_1,\dots,m_N\}", "A generation is a population of N hands."),
    ("keep", r"E_t=\operatorname{top}_{N/2}(P_t)", "Keep the best half."),
    ("parents", r"A_t=\operatorname{top}_{N/8}(P_t)", "The best eighth are the parents."),
    ("mutate", r"m'=\operatorname{round}_{5\,\mathrm{mm}}\big(a+\delta\,\epsilon\big),\ \ a\in A_t,\ \ \epsilon\sim\mathcal{U}\big[-\tfrac12,\tfrac12\big]^2,\ \ \delta=30\,\mathrm{mm}",
     "A child is a parent moved by up to 15 mm per coordinate, on the 5 mm lattice."),
    ("next", r"P_{t+1}=E_t\cup\{m'_1,\dots,m'_{N/2}\}", "The kept half and the children form the next generation."),
]

INNER = [  # (what, LaTeX); owner, 2026-11-10 evening: the grasp g and HOM's parameters phi first appear here, with HOM
    ("For one hand, the inner loop searches over controllers.",
     r"J^\star(m)=\max_{\pi:\,x\mapsto u\in\mathbb{R}^{9}}\ J(m,\pi)"),
    ("The controller is HOM, held fixed: only a grasp and a few parameters remain.",
     r"J^\star(m)={\color{inner}\underbrace{\max_{g,\,\phi}}_{\clab{grasp, HOM parameters}}}\ \ J(m,g,\phi)"),
    ("Two choices: a grasp, then HOM's parameters.",
     r"J^\star(m)=" + r"{\color{inner}\underbrace{\max_{g}}_{\clab{grasp synthesis}}}\qquad\qquad{\color{inner}\underbrace{\max_{\phi}}_{\clab{HOM tuning}}}\ \ J(m,g,\phi)"),
    ("The foundational pose: the grasp farthest from every constraint.",
     r"g_0=\arg\max_{v,\,\sigma}\ \sigma\quad\mathrm{s.t.}\quad c_j(v;\,m)\ge\sigma\ \ \forall j"),
    ("Sixteen candidate grasps per hand.",
     r"\mathcal{G}(m)=\{g_0,\,g_1,\,\dots,\,g_{15}\}"),
    ("HOM: the joint speeds that best produce each fingertip's contact motion.",
     r"u=\arg\min_{u}\ \sum_k w_k\Big(v_{\mathrm{ref},k}-\big[{\color{inner}\underbrace{A(m,g)}_{\clab{the hand}}}\,u+b\big]_k\Big)^2+w_{\mathrm{d}}\lVert u\rVert^2"),
    ("Press with force F through the servo's spring.",
     r"q_{\mathrm{cmd}}=q_{\mathrm{ref}}+\frac{1}{k_p}\,J^{\top}F\,n"),
    ("What tuning changes: the forces and the contact-mode weights.",
     r"\phi=(F,\,w)"),
    ("The cross-entropy method: sample, keep the elite, refit.",
     r"\phi^{(k)}\sim\mathcal{N}(\mu,\Sigma),\qquad(\mu,\Sigma)\leftarrow\operatorname{fit}\big(\text{elite}\big)"),
]

SIM = [  # owner, 2026-11-10 evening: model the dynamics as closely as possible, at the highest throughput, so that a hand's
    # quality is predicted before it is built
    ("A simulator stands in for the real dynamics, so that the simulated score predicts the real one.",
     r"f^{\,\mathrm{sim}}\approx f^{\,\mathrm{real}}\quad\Longrightarrow\quad J^{\,\mathrm{sim}}\approx J^{\,\mathrm{real}}"),
    ("The simulator's contact c, servo p and sensing s are calibrated on hardware.",
     r"f^{\,\mathrm{sim}}(x_t,u_t;\,m\mid{\color{sim}c,\,p,\,s})"),
    ("Rollouts per EA generation: one GPU batch.",
     r"\underbrace{32}_{\lab{new hands}}\times\underbrace{16}_{\lab{grasps}}\times\underbrace{3}_{\lab{seeds}}=1536"),
    ("Sensing depends on the hand.", r"f^{\,\mathrm{sim}}(x_t,u_t;\,m\mid c,\,p,\,{\color{sim}s(m)})"),
]

METRIC = [
    ("Several metrics, weighted into one.", r"J=\sum_k\lambda_k\,J^{(k)}"),
    ("Ours: the share of the commanded rotation followed, on three pads; 0 if dropped.",
     r"J=\underbrace{\mathbf{1}[\mathrm{held}]}_{\lab{0 if dropped}}\ \underbrace{\max\!\Big(0,\ 1-\frac{e_{\mathrm{rms}}}{\theta_{\mathrm{rms}}}\Big)}"
     r"_{\lab{rotation followed}}\ \underbrace{c_3}_{\lab{three pads}}"),
]


def outer_line(k):
    rows = []
    for i, (lab, tex, _w) in enumerate(OUTER, 1):
        line = r"\text{\sffamily " + lab + r"} &\quad " + tex
        if i > k:
            rows.append(r"\phantom{\text{\sffamily " + lab + r"}} &\quad \phantom{" + tex + "}")
        elif i == k:
            rows.append(r"\textcolor{outer}{\text{\sffamily " + lab + r"}} &\quad {\color{outer}" + tex + "}")
        else:
            rows.append(line)
    return r"\begin{aligned}" + r"\\[3pt]".join(rows) + r"\end{aligned}"


def formulas():
    MANIFEST.clear()
    for old in OUT.glob("20261110-eq_*.png"):
        old.unlink()
    cd = [eq_img(f"20261110-eq_cd{k:02d}", tex, "codesign", what, sl) for k, (sl, what, tex) in enumerate(CD, 1)]
    cd.append(eq_img(f"20261110-eq_cd{len(CD) + 1:02d}", final(), "codesign", "Each brace is one section of the talk.", "32"))
    for a, what in (("outer", "Zoom 1: morphology optimization."), ("inner", "Zoom 2: multi-hand control."),
                    ("dyn", "Zoom 3: dynamics.")):
        cd.append(eq_img(f"20261110-eq_final_{a}", final(a), "zoom", what))
    same_canvas(cd, CD_CANVAS)
    same_canvas([eq_img(f"20261110-eq_metric{k}", tex, "metric", what, "31", strut=False)
                 for k, (what, tex) in enumerate(METRIC, 1)])
    for k, (what, tex) in enumerate(THESIS, 1):  # on two different slides: each its own canvas
        eq_img(f"20261110-eq_thesis{k}", tex, "thesis", what, "24-25")
    same_canvas([eq_img(f"20261110-eq_morph{k}", tex, "morph", what, "32") for k, (what, tex) in enumerate(MORPH, 1)])
    same_canvas([eq_img(f"20261110-eq_pads{k}", tex, "pads", what, "36-38") for k, (what, tex) in enumerate(PADS, 1)])
    same_canvas([eq_img(f"20261110-eq_outer{k}", outer_line(k), "outer", OUTER[k - 1][2], "32-33", strut=False, size=19)
                 for k in range(1, len(OUTER) + 1)])
    same_canvas([eq_img(f"20261110-eq_inner{k}", tex, "inner", what, "34-35", strut=False)
                 for k, (what, tex) in enumerate(INNER, 1)])
    same_canvas([eq_img(f"20261110-eq_sim{k}", tex, "sim", what, "36-38", strut=False)
                 for k, (what, tex) in enumerate(SIM, 1)])
    (OUT / "20261110-formulas.json").write_text(__import__("json").dumps(MANIFEST, indent=1))


# ------------------------------------------------------------------------------------------------ the nested problem
# Owner, 2026-11-10 evening: each level a title and one line, no more; the innermost level is the dynamics ("our bi-level
# optimization problem is subject to the dynamics and constraints of the world/robot/environment, in reality and in
# simulation"); the controller as u in R^9 until HOM is explained; the tasks T and object sizes d out of the boxes and
# shown after them, in black, as "the true outer loop": a small finite set, then T and d drawn from D, of which there
# are infinitely many; then the metric J drawn from a space of metrics, script J (owner: not a lowercase j).
LEVELS = [  # (key, title, line, formula), outermost first
    ("outer", "Morphology optimization", "Evolutionary algorithms for hand selection", r"\arg\max_{m\in\mathbb{R}^{6}}"),
    ("inner", "Multi-hand control", "Grasp synthesis and in-hand manipulation", r"\max_{\pi:\,x\mapsto u\in\mathbb{R}^{9}}"),
    ("dyn", "Dynamics", "Physical modeling and calibration", r"x_{t+1}=f(x_t,u_t;\,m)"),
]
TASK_LOOP = {
    "finite": r"\textbf{for}\ \ $T\in\{\text{twist},\,\text{roll},\,\text{press}\},\ \ d\in\{d_1,\dots,d_4\}$:",
    "infinite": r"\textbf{for}\ \ $T,\,d\sim\mathcal{D}$:",
    "metric": r"\textbf{for}\ \ $T,\,d\sim\mathcal{D},\ \ J\sim\mathcal{J}$:",   # metrics drawn from the space J
}


def nested(active, loop=None):
    """The three levels as nested boxes, whole ('all') or with one level highlighted; loop ('finite' or 'infinite') writes
    the tasks and object sizes above the boxes, in black, as the outermost loop. 'all' and the loop versions share one
    canvas, so on successive slides only the loop's line appears."""
    W, inset, top, head = 18.2, 0.9, 8.1, 2.2        # cm: outer width, inset per level, outer top, a level's header
    out = [r"\begin{tikzpicture}[font=\sffamily, every node/.style={inner sep=0pt}]"]
    if active == "all":
        out.append(r"\useasboundingbox (-0.75,-0.05) rectangle (%.2f,%.2f);" % (W + 0.05, top + 1.45))
    for i, (key, title, line, tex) in enumerate(LEVELS):
        on = active in ("all", key)
        c = LEVEL_COLOUR[key]
        x0, x1, y1 = i * inset, W - i * inset, top - i * head
        y0 = 0.7 * i if i < len(LEVELS) - 1 else y1 - 2.25
        out.append(r"\draw[rounded corners=%dpt, %s] (%.2f,%.2f) rectangle (%.2f,%.2f);"
                   % (10 - i, f"fill={c}l, draw={c}, line width=1.6pt" if on else "fill=pale, draw=grey!70, line width=1pt",
                      x0, y0, x1, y1))
        tc = c if on else "grey"
        out.append(r"\node[anchor=north west, text=%s] at (%.2f,%.2f) {\fontsize{17}{20}\selectfont\bfseries %s};"
                   % (tc, x0 + 0.45, y1 - 0.45, title))
        out.append(r"\node[anchor=north west, text=%s] at (%.2f,%.2f) {\fontsize{14}{17}\selectfont %s};"
                   % (tc, x0 + 0.45, y1 - 1.2, line))
        out.append(r"\node[anchor=east, text=%s] at (%.2f,%.2f) {\fontsize{20}{24}\selectfont$\displaystyle %s$};"
                   % (tc, x1 - 0.45, y1 - 1.05, tex))
    if loop:
        out.append(r"\node[anchor=south west, text=ink] at (-0.7,%.2f) {\fontsize{19}{23}\selectfont %s};"
                   % (top + 0.45, TASK_LOOP[loop]))
    out.append(r"\end{tikzpicture}")
    return compile_tex(f"20261110-nested_{'loop_' + loop if loop else active}", "\n".join(out))


# ------------------------------------------------------------------------------------------------ HOM pictures
def contact_modes():
    """What HOM sets at each fingertip, relative to the object: stick, roll, slide, and the press along the normal."""
    panel = r"""
\begin{scope}[shift={(%(x)s,0)}]
\fill[pale] (0,0) circle (1.7);
\draw[ink!70, line width=1pt] (0,0) circle (1.7);
\node[text=ink!60] at (0,-0.55) {object};
%(tip)s
\node[anchor=north, text=ink] at (0,-2.05) {\fontsize{16}{19}\selectfont\bfseries %(title)s};
\node[anchor=north, text=ink, align=center, text width=4.3cm] at (0,-2.75) {%(desc)s};
\end{scope}"""
    tip = r"""\begin{scope}[shift={(0,1.7)}, rotate=%(rot)s]
\draw[inner, line width=1.3pt, fill=innerl] (-0.55,1.25) -- (-0.55,0.45) arc[start angle=180, end angle=360, x radius=0.55, y radius=0.45] -- (0.55,1.25);
\fill[inner] (0,0) circle (0.07);
\end{scope}"""
    ar = "-{Stealth[length=7pt,width=6pt]}"
    p = {
        "stick": (r"\draw[%s, ink, line width=1.6pt] (0.95,2.55) -- (1.95,2.55); \draw[%s, ink, line width=1.6pt]"
                  r" (0.95,0.0) -- (1.95,0.0); \node[text=ink] at (1.45,0.32) {$V$}; \node[text=ink] at (1.45,2.87) {$V$};"
                  % (ar, ar), "Stick", r"the fingertip moves with the object: no relative motion"),
        "roll": (r"\draw[%s, inner, line width=1.6pt] (0.9,2.4) arc[start angle=10, end angle=120, radius=0.85];"
                 r" \node[text=inner] at (1.25,3.05) {$\omega_{\mathrm{roll}}$};"
                 r" \draw[%s, ink!60, line width=1pt] (0.15,1.55) arc[start angle=80, end angle=50, radius=1.7];" % (ar, ar),
                 "Roll", r"the fingertip turns on the surface; the contact point moves, no slip"),
        "slide": (r"\draw[%s, inner, line width=1.8pt] (0.75,2.05) -- (2.0,2.05);"
                  r" \node[text=inner] at (1.4,2.4) {$v_{\mathrm{slide}}$};" % ar,
                  "Slide", r"the fingertip slides along the surface"),
        "press": (r"\draw[%s, inner, line width=1.8pt] (0,3.6) -- (0,2.35); \node[anchor=west, text=inner] at (0.12,3.1) "
                  r"{$F\,n$};" % ar, "Press", r"push along the normal $n$ with force $F$"),
    }
    body = [r"\begin{tikzpicture}[font=\sffamily]", r"\fontsize{13}{16}\selectfont"]
    for k, (name, x) in enumerate(zip(("stick", "roll", "slide", "press"), (0, 5.2, 10.4, 15.6))):
        extra, title, desc = p[name]
        body.append(panel % {"x": x, "tip": tip % {"rot": 0} + extra, "title": title, "desc": desc})
    body.append(r"\node[anchor=south, text=ink, align=center] at (7.8,4.25) {\fontsize{16}{19}\selectfont HOM commands, "
                r"for each fingertip, how it moves \emph{relative to the object} at the contact};")
    body.append(r"\end{tikzpicture}")
    return compile_tex("20261110-contact_modes", "\n".join(body))


def hom_period():
    """One control period: object command, contact references, least squares, servo targets; the plant feeds back."""
    body = r"""\begin{tikzpicture}[font=\sffamily, box/.style={draw=inner, line width=1.4pt, rounded corners=7pt, fill=white,
  minimum width=5.3cm, minimum height=3.4cm, align=center, text width=5.0cm, font=\sffamily\fontsize{15}{18}\selectfont}, ar/.style={-{Stealth[length=9pt,width=8pt]},
  line width=1.5pt}]
\fontsize{13}{16}\selectfont
\node[box] (a) at (0,0) {{\fontsize{17}{20}\selectfont\bfseries Object command}\\[6pt] how the object should move\\[4pt] $V$};
\node[box] (b) at (6.0,0) {{\fontsize{17}{20}\selectfont\bfseries Fingertip targets}\\[6pt] per fingertip: stick, roll, slide\\[4pt] $v_{\mathrm{ref}},\ w$};
\node[box, fill=inner, text=white] (c) at (12.0,0) {{\fontsize{17}{20}\selectfont\bfseries Least squares}\\[6pt] joint speeds that best make it happen\\[5pt] $\min_u\lVert v_{\mathrm{ref}}-(Au+b)\rVert_w^2$};
\node[box] (d) at (18.0,0) {{\fontsize{17}{20}\selectfont\bfseries Servo targets}\\[6pt] integrate $u$, press with $F$\\[5pt] $q_{\mathrm{ref}}+J^{\top}Fn/k_p$};
\draw[ar, inner] (a) -- (b); \draw[ar, inner] (b) -- (c); \draw[ar, inner] (c) -- (d);
\node[draw=grey, line width=1.1pt, rounded corners=7pt, fill=pale, minimum width=11.4cm, minimum height=1.25cm] (p) at (15.0,-3.7)
  {hand and object: joint angles, object pose, pad forces};
\draw[ar, ink] (d.south) -- (d.south |- p.north);
\draw[ar, ink] (c.south |- p.north) -- (c.south);
\node[anchor=east, text=ink] at ($(c.south |- p.north)!0.5!(c.south)+(-0.2,0)$) {every 10 ms};
\node[anchor=south, text=inner, align=center] (h) at (12.0,1.75) {\bfseries the hand enters only here:\\ $A=A(m,g)$};
\node[anchor=south, text=ink!70, align=center] at (6.0,1.75) {tuned per hand, if at all:\\ $\phi=(F,\,w)$};
\end{tikzpicture}"""
    return compile_tex("20261110-hom_period", body)


def evaluator():
    """Sim-to-real as an evaluator built for co-design: fast, calibrated, observable, each with its measured numbers."""
    body = r"""\begin{tikzpicture}[font=\sffamily, card/.style={draw=sim, line width=1.4pt, rounded corners=9pt, fill=white,
  minimum width=7.4cm, minimum height=8.6cm}]
\fontsize{13}{16}\selectfont
\foreach \x/\t in {0/Fast, 8.0/Calibrated, 16.0/Observable} {
  \node[card] at (\x,0) {};
  \fill[sim, rounded corners=9pt] (\x-3.7,3.6) rectangle (\x+3.7,4.3);
  \fill[sim] (\x-3.7,3.6) rectangle (\x+3.7,3.9);
  \node[text=white, font=\sffamily\bfseries\fontsize{18}{21}\selectfont] at (\x,3.95) {\t};
}
% fast: cost per rollout on a log axis
\node[align=center, text=ink] at (0,2.75) {thousands of rollouts per search,\\ so one rollout must be cheap};
\draw[grey, line width=0.8pt] (-3.1,-0.9) -- (3.1,-0.9);
\foreach \v/\l in {0/1\,ms, 1/10\,ms, 2/100\,ms, 3/1\,s, 4/10\,s} {\draw[grey] ({-3.1+1.45*\v},-0.9) -- ++(0,-0.12) node[below, text=ink!70, font=\sffamily\fontsize{11}{13}\selectfont] {\l};}
\fill[grey!60] (-3.1,0.65) rectangle ({-3.1+1.45*3.70},1.25);
\node[anchor=west, text=ink] at (-3.1,1.55) {CPU, one rollout: 5.0\,s};
\fill[sim] (-3.1,-0.55) rectangle ({-3.1+1.45*1.56},0.05);
\node[anchor=west, text=ink] at (-3.1,0.35) {GPU, in a batch of 1536: 36.5\,ms};
\node[align=center, text=ink] at (0,-2.45) {one EA generation, 32 new hands\\ $\times$ 16 grasps $\times$ 3 seeds,\\ is one GPU batch: about 1 minute};
\node[align=center, text=sim, font=\sffamily\bfseries] at (0,-3.75) {1 mm sphere pads: GPU-friendly};
% calibrated: fitted on the UR5 and on the hand
\node[align=center, text=ink] at (8,2.75) {model parameters\\ measured on hardware};
\node[text=sim, font=\sffamily\bfseries\fontsize{26}{30}\selectfont] at (8,1.3) {2.47 MPa};
\node[align=center, text=ink] at (8,0.35) {pad modulus, fitted to\\ UR5 presses of three printed tips};
\node[text=sim, font=\sffamily\bfseries\fontsize{26}{30}\selectfont] at (8,-1.35) {0.55 N\,m/rad};
\node[align=center, text=ink] at (8,-2.3) {servo stiffness identified on the hand\\ (the plant uses 0.72); 0.59$^\circ$ dead band};
\node[align=center, text=sim, font=\sffamily\bfseries] at (8,-3.75) {fit once, then simulate many};
% observable: the fingertip measures what the model and HOM use
\node[align=center, text=ink] at (16,2.75) {the fingertip measures\\ what HOM and the model use};
\begin{scope}[shift={(16,1.05)}, scale=0.92]
  \draw[ink!70, line width=1.2pt, fill=pale] (-1.25,1.0) -- (-1.25,-0.2) arc[start angle=180, end angle=360, x radius=1.25, y radius=1.15] -- (1.25,1.0);
  \foreach \mx in {-0.55,0.55} {\fill[ink!80] (\mx-0.14,0.35) rectangle (\mx+0.14,0.63);}
  \node[text=ink!70, font=\sffamily\fontsize{11}{13}\selectfont] at (0,0.85) {magnets};
  \fill[sim] (0.55,-1.12) circle (0.08);
  \draw[sim, line width=1pt, dashed] (0.55,-1.12) circle (0.3);
  \draw[-{Stealth[length=7pt]}, sim, line width=1.4pt] (1.25,-1.75) -- (0.62,-1.2);
  \node[anchor=west, text=sim] at (1.3,-1.7) {$F$};
\end{scope}
\node[align=center, text=ink] at (16,-2.15) {contact location within 0.46--0.51\,mm\\ (tips 1--3) and 0.78--0.82\,mm (tips 4--6),\\ and the normal force};
\node[align=center, text=sim, font=\sffamily\bfseries] at (16,-3.75) {sensing depends on the hand: $s(m)$};
\end{tikzpicture}"""
    return compile_tex("20261110-evaluator", body)


# ------------------------------------------------------------------------------------------------ the rule under test
# Owner, 2026-11-10 evening: slides 24-25 state the thesis, a test of Borras and Dollar's rule (IJRR 2015) under executed
# in-hand manipulation. A schematic of the rule as stated in their abstract (drawn here, not copied): a symmetric tripod
# around a round object, each finger base half a finger length outside the object's surface, so the palm's diameter is
# the object's plus one finger length.
def bd_rule():
    r, L = 1.0, 1.7                                   # object radius and finger length, arbitrary units
    R = r + L / 2
    import math
    out = [r"\begin{tikzpicture}[x=1.55cm, y=1.55cm, line cap=round, line join=round]",
           r"\fill[pale] (0,0) circle (%.3f); \draw[ink!60, line width=1.1pt] (0,0) circle (%.3f);" % (r, r),
           r"\draw[grey, line width=1.1pt, dash pattern=on 5pt off 4pt] (0,0) circle (%.3f);" % R,
           r"\node[text=ink!70] at (0,0) {\fontsize{17}{20}\selectfont object};"]
    for k, ang in enumerate((90, 210, 330)):
        a = math.radians(ang)
        bx, by, cx, cy = R * math.cos(a), R * math.sin(a), r * math.cos(a), r * math.sin(a)
        out.append(r"\draw[grey!80, line width=7pt] (%.3f,%.3f) -- (%.3f,%.3f);" % (bx, by, cx, cy))
        out.append(r"\fill[grey!80] (%.3f,%.3f) circle (0.075);" % (cx, cy))
        out.append(r"\fill[new] (%.3f,%.3f) circle (0.085);" % (bx, by))
    a = math.radians(330)
    n = (math.cos(a + math.pi / 2), math.sin(a + math.pi / 2))
    o = 0.32
    p0 = (r * math.cos(a) + o * n[0], r * math.sin(a) + o * n[1])
    p1 = (R * math.cos(a) + o * n[0], R * math.sin(a) + o * n[1])
    out.append(r"\draw[new, line width=1.8pt, {Stealth[length=7pt]}-{Stealth[length=7pt]}] (%.3f,%.3f) -- (%.3f,%.3f);"
               % (p0 + p1))
    out.append(r"\node[text=new, anchor=south west] at (%.3f,%.3f) {\fontsize{19}{22}\selectfont $L/2$};"
               % ((p0[0] + p1[0]) / 2 + 0.05, (p0[1] + p1[1]) / 2 + 0.05))
    out.append(r"\draw[ink!60, line width=1pt, {Stealth[length=6pt]}-{Stealth[length=6pt]}] (%.3f,%.3f) -- (%.3f,%.3f);"
               % (-r, -R - 0.45, r, -R - 0.45))
    out.append(r"\node[text=ink, anchor=north] at (0,%.3f) {\fontsize{17}{20}\selectfont object diameter $d$};" % (-R - 0.55))
    out.append(r"\draw[grey, line width=1pt, {Stealth[length=6pt]}-{Stealth[length=6pt]}] (%.3f,%.3f) -- (%.3f,%.3f);"
               % (-R, R + 0.42, R, R + 0.42))
    out.append(r"\node[text=ink, anchor=south] at (0,%.3f) {\fontsize{17}{20}\selectfont palm diameter $d+L$};" % (R + 0.5))
    out.append(r"\node[text=new, anchor=west] at (%.3f,%.3f) {\fontsize{15}{18}\selectfont finger base};"
               % (R * math.cos(math.radians(90)) + 0.15, R + 0.0))
    out.append(r"\end{tikzpicture}")
    return compile_tex("20261110-bd_rule", "\n".join(out))


# ------------------------------------------------------------------------------------------------ morphology and control
# Owner, 2026-11-10 afternoon: "really stripped down, bare figures that communicate the essence of the split between
# morphology and control", beside steps 3 and 4. One line drawing of the hand of the films that follow, layout (70, 110)
# mm, with the joints of the real_v1 model coloured by class: the six slides of class morph (x and y of each finger's
# carriage, inside its gantry's travel) or the nine hinges of class ctrl (yaw, MCP and PIP of each finger). No text.
HAND_VIEW = (30.0, 30.0)        # degrees: azimuth about the palm normal, elevation of the view above the palm plane
HAND_COLOURS = {                # boxes' fill and edge, mounts, slide arrows, hinges
    "mounts": ("new!10", "new", "new", "new", None),
    "joints": ("pale", "grey", "grey", None, "new"),
    "layout": ("pale", "grey", "new", None, None),   # with the two distances of the symmetric family in colour
}


def hand_geometry(layout=(70.0, 110.0), q_deg=None):
    """Travel boxes and joint chains (mm, palm frame: mount, MCP, PIP, pad centre) of the symmetric tripod `layout` with
    the fingers at the joint angles q_deg (yaw, MCP, PIP in degrees; default a gentle curl toward the palm's centre line).
    Kinematics, gantry travel and the cad_v2 finger (20.75, 30.80 and 36.36 mm) from fp_sr2_toy."""
    import numpy as np
    sys.path.insert(0, str(ROOT / "scripts"))
    import _paths  # noqa: F401
    import fp_sr2_toy as FT
    q_deg = q_deg or {"thumb": (0.0, 12.0, 32.0), "index": (-20.0, 12.0, 32.0), "middle": (20.0, 12.0, 32.0)}
    M = FT.mounts6(FT.layout2(*layout))
    chains, boxes = {}, {}
    for i, f in enumerate(FT.K.FINGERS):
        fr = FT.finger_frames(f, np.radians(q_deg[f]), M[f])
        chains[f] = 1000.0 * np.array([fr[k][1] for k in ("yaw_frame", "mcp_frame", "pip_frame", "tip")])
        (x0, y0), (x1, y1) = FT.LO6[2 * i:2 * i + 2], FT.HI6[2 * i:2 * i + 2]
        boxes[f] = np.array([[x0, y0, 0.0], [x1, y0, 0.0], [x1, y1, 0.0], [x0, y1, 0.0]])
    return boxes, chains


def hand(variant):
    """The hand drawing with one class of joints in the highlight colour: 'mounts' (step 3) or 'joints' (step 4).
    Both share one canvas, so on successive slides only the colour moves."""
    import numpy as np
    from sr2hand.geometry.rotations import rot_x, rot_z
    boxes, chains = hand_geometry()
    az, el = np.radians(HAND_VIEW)
    V = rot_x(el) @ rot_z(az)

    def uv(P):
        Q = np.asarray(P, float) @ V.T
        return Q[..., 0], Q[..., 2], Q[..., 1]            # right, up, depth (larger is farther)

    def pt(P):
        u, v, _ = uv(P)
        return f"({u:.2f},{v:.2f})"

    fill, edge, mount, slide, hinge = HAND_COLOURS[variant]
    arrow = 14.0                                          # mm each way along x and y of the palm frame
    tips = r"{Stealth[length=8pt,width=8pt]}"
    ends = list(boxes.values()) + list(chains.values())
    ends.append(np.array([c[0] + s * arrow * e for c in chains.values() for s in (-1, 1) for e in np.eye(3)[:2]]))
    u, v, _ = uv(np.concatenate(ends))
    pad = 9.0
    out = [r"\begin{tikzpicture}[x=0.5mm, y=0.5mm, line join=round, line cap=round]",
           r"\useasboundingbox (%.1f,%.1f) rectangle (%.1f,%.1f);" % (u.min() - pad, v.min() - pad, u.max() + pad,
                                                                      v.max() + pad)]
    for b in boxes.values():                      # drawn as a transparent plane: the fingers stay visible to their mounts
        out.append(r"\filldraw[fill=%s, draw=%s, line width=1.2pt] %s -- cycle;" % (fill, edge, " -- ".join(map(pt, b))))
    for f in sorted(chains, key=lambda f: -uv(chains[f])[2].mean()):         # farthest finger first
        ch = chains[f]
        out.append(r"\draw[grey!80, line width=6.5pt] " + " -- ".join(map(pt, ch)) + ";")
        out.append(r"\fill[grey!80] %s circle[radius=6pt];" % pt(ch[-1]))
    for ch in chains.values():
        out.append(r"\fill[%s] %s circle[radius=4.5pt];" % (mount, pt(ch[0])))
        if slide:
            for e in np.eye(3)[:2]:
                out.append(r"\draw[%s, line width=2.4pt, %s-%s] %s -- %s;"
                           % (slide, tips, tips, pt(ch[0] - arrow * e), pt(ch[0] + arrow * e)))
    if hinge:
        for ch in chains.values():
            for p in ch[:3]:
                out.append(r"\filldraw[fill=white, draw=%s, line width=2.6pt] %s circle[radius=5.5pt];" % (hinge, pt(p)))
    if variant == "layout":                      # thumb to the pair's midpoint (s1, along x), index to middle (s2, along y)
        th, ix, md = (chains[f][0] for f in ("thumb", "index", "middle"))
        for a, b, lab, t, anchor in ((th, (ix + md) / 2, "s_1", 0.3, "south east"), (ix, md, "s_2", 0.5, "west")):
            out.append(r"\draw[new, line width=2.4pt, %s-%s, shorten <=6pt, shorten >=6pt] %s -- %s;"
                       % (tips, tips, pt(a), pt(b)))
            out.append(r"\node[text=new, anchor=%s, inner sep=7pt] at %s {\fontsize{24}{28}\selectfont$%s$};"
                       % (anchor, pt(a + t * (b - a)), lab))
    out.append(r"\end{tikzpicture}")
    return compile_tex(f"20261110-hand_{variant}", "\n".join(out), border="6pt")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    formulas()
    for old in OUT.glob("20261110-nested_*.png"):
        old.unlink()
    for a in ("outer", "inner", "dyn"):
        nested(a)
    same_canvas([nested("all")] + [nested("all", loop) for loop in TASK_LOOP])
    bd_rule()
    contact_modes()
    hom_period()
    evaluator()
    for v in HAND_COLOURS:
        hand(v)
    print("ok", len(list(OUT.glob("20261110-eq_*.png"))), "formula images", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
