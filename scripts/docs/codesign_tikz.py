#!/usr/bin/env python3
"""LaTeX and TikZ figures of the co-design formulation for the hand-design talk (owner, 2026-11-10: "beautiful, didactic
diagrams"; "an evolving latex formula/equation that captures the co-design problem"; the SR2 part "hierarchical, rather
than cyclical"; HOM "even simpler/clearer"). Each figure is a standalone LaTeX file compiled with pdflatex and rasterized
by pdftoppm at 400 dpi into docs/experiments/20261110-codesign_formulation/media/:

  20261110-eq_step{1..8}.png       the co-design problem, one term more per slide; one canvas and one baseline for all
  20261110-eq_final_{outer,inner,sim}.png   the final form with one level in colour
  20261110-eq_outer.png, eq_inner_{split,grasp,hom,servo,tune}.png, eq_sim_{budget,sensing}.png   the levels' formulas
  20261110-nested_{all,outer,inner,sim}.png the nested problem as nested boxes, whole and with one level highlighted
  20261110-contact_modes.png        what HOM commands at each fingertip: stick, roll, slide, press
  20261110-hom_period.png           one HOM control period

    .venv/bin/python scripts/docs/codesign_tikz.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

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


def same_canvas(paths):
    """Pad the PNGs to one canvas, centred, so a slide sequence keeps its size and position."""
    ims = [Image.open(p).convert("RGB") for p in paths]
    w, h = max(i.width for i in ims), max(i.height for i in ims)
    for p, im in zip(paths, ims):
        c = Image.new("RGB", (w, h), "white")
        c.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
        c.save(p)


# ------------------------------------------------------------------------------------------------ the formula build
# A strut of fixed height and depth makes every step's baseline sit at the same height of the same canvas.
STRUT = r"\rule[-1.55cm]{0pt}{2.45cm}"


def eq_img(name, tex, note=""):
    body = (r"\fontsize{21}{26}\selectfont\begin{tabular}{c}$\displaystyle " + STRUT + tex + r"$\\[2pt]"
            + (r"{\fontsize{15}{19}\selectfont " + note + "}" if note else r"\rule{0pt}{19pt}") + r"\end{tabular}")
    return compile_tex(name, body, border="{90pt 16pt}")


N = r"\color{new}"
STEPS = [
    (r"m^\star_T=\arg\max_{m}\ \underbrace{J_T(m)}_{\clab{how well hand $m$ does task $T$}}", ""),
    (r"m^\star_T=\arg\max_{m}\ {" + N + r"\underbrace{\max_{\pi}}_{\clab{its best controller}}}\ J_T(m,{" + N + r"\pi})",
     r"the hand can only be scored through a controller: a \textcolor{new}{bi-level} problem"),
    (r"m^\star_T=\arg\max_{m}\ {" + N + r"\underbrace{\mathbb{E}_{d\sim\mathcal{D}}}_{\clab{over object sizes}}}"
     r"\Big[\max_{\pi}\ J_T(m,\pi;{" + N + r"d})\Big]", r"one hand for a range of sizes, or one hand per size"),
    (r"m^\star_T=\arg\max_{{" + N + r"m\in\mathbb{R}^{6}}}\ \mathbb{E}_{d\sim\mathcal{D}}\Big[\max_{{" + N
     + r"\pi:\,x_t\mapsto u_t\in\mathbb{R}^{9}}}\ J_T(m,\pi;d)\Big]",
     r"\textcolor{new}{6} mount coordinates for the hand, \textcolor{new}{9} joint commands every 10\,ms"),
    (r"m^\star=\arg\max_{m\in\mathbb{R}^{6}}\ {" + N + r"\underbrace{\mathbb{E}_{(T,d)\sim\mathcal{D}}}_{\clab{over tasks "
     r"and sizes}}}\Big[\max_{\pi}\ J_{{" + N + r"T}}(m,\pi;d)\Big]",
     r"three toy tasks are three samples of a distribution over tasks"),
    (r"m^\star=\arg\max_{m\in\mathbb{R}^{6}}\ \mathbb{E}_{(T,d)\sim\mathcal{D}}\Big[\max_{\pi}\ {" + N
     + r"\underbrace{J_T(m,\pi;d)}_{\clab{a choice of metric}}}\Big]",
     r"$J=\sum_k\lambda_k J^{(k)}$ or a Pareto front; ours: $J=\mathbf{1}[\mathrm{held}]\,\max\!\big(0,\,1-"
     r"\frac{e_{\mathrm{rms}}}{\theta_{\mathrm{rms}}}\big)\,c_3$"),
    (r"m^\star=\arg\max_{m\in\mathbb{R}^{6}}\ \mathbb{E}_{(T,d)\sim\mathcal{D}}\Big[\max_{\pi}\ {" + N
     + r"\underbrace{J^{\,\mathrm{sim}}_T(m,\pi;d\mid c,p,s)}_{\clab{computed in simulation}}}\Big]",
     r"contact model $c$, servo model $p$, sensing model $s$: each calibrated on hardware"),
]


def final(active=None):
    """The final form; active in (None, 'outer', 'inner', 'sim'): None colours all three braces."""
    col = {k: (k if active in (None, k) else "grey") for k in ("outer", "inner", "sim")}
    return (r"m^\star={\color{" + col["outer"] + r"}\underbrace{\arg\max_{m\in\mathbb{R}^{6}}}_{\clab{morphology "
            r"optimization}}}\qquad \mathbb{E}_{(T,d)\sim\mathcal{D}}\Big[\;{\color{" + col["inner"]
            + r"}\underbrace{\max_{g,\ \phi}}_{\clab{multi-hand control}}}\ {\color{" + col["sim"]
            + r"}\underbrace{J^{\,\mathrm{sim}}_T(m,g,\phi;d\mid c,p,s)}_{\clab{sim-to-real}}}\Big]")


def formulas():
    paths = [eq_img(f"20261110-eq_step{k}", t, n) for k, (t, n) in enumerate(STEPS, 1)]
    paths.append(eq_img("20261110-eq_step8", final(),
                        r"the controller is HOM with a grasp $g$ and parameters $\phi$; each brace is one section"))
    for a in ("outer", "inner", "sim"):
        paths.append(eq_img(f"20261110-eq_final_{a}", final(a), {
            "outer": r"zoom 1: choose the hand", "inner": r"zoom 2: run the hand",
            "sim": r"zoom 3: score the hand"}[a]))
    same_canvas(paths)


def level_eqs():
    big = r"\fontsize{19}{24}\selectfont"
    eqs = {
        "20261110-eq_outer": (r"$\displaystyle\begin{aligned}"
                              r"&\textcolor{outer}{\lab{fitness}} && F(m)=\frac{1}{3}\sum_{s=1}^{3}\ \max_{g}\ J^{\,\mathrm{sim}}(m,g,\phi_0;\,s)\\[4pt]"
                              r"&\textcolor{outer}{\lab{population}} && P_t=\{m_1,\dots,m_N\}\\[2pt]"
                              r"&\textcolor{outer}{\lab{keep}} && E_t=\operatorname{top}_{N/2}(P_t;\,F)\\[2pt]"
                              r"&\textcolor{outer}{\lab{parents}} && A_t=\operatorname{top}_{N/8}(P_t;\,F)\\[2pt]"
                              r"&\textcolor{outer}{\lab{mutate}} && m'=\Pi_{\Lambda}\big(a+\delta\,\epsilon\big),\quad a\in A_t,\ \ "
                              r"\epsilon\sim\mathcal{U}\big[-\tfrac12,\tfrac12\big]^2,\ \ \delta=30\,\mathrm{mm}\\[2pt]"
                              r"&\textcolor{outer}{\lab{next}} && P_{t+1}=E_t\cup\{m'_1,\dots,m'_{N/2}\}"
                              r"\end{aligned}$"),
        "20261110-eq_inner_split": (r"$\displaystyle J^\star(m)=\textcolor{inner}{\underbrace{\max_{g\in\mathcal{G}(m)}}"
                                    r"_{\lab{grasp synthesis}}}\ \ \textcolor{inner}{\underbrace{\max_{\phi}}_{\lab{HOM tuning}}}"
                                    r"\ \ J^{\,\mathrm{sim}}\big(m,\ g,\ \phi\big)$"),
        "20261110-eq_inner_grasp": (r"$\displaystyle g_0=\arg\max_{v,\,\sigma}\ \sigma\quad\text{s.t.}\quad "
                                    r"\underbrace{c_j(v;\,m)\ge\sigma}_{\lab{every constraint has slack $\ge\sigma$}}\ \ \forall j"
                                    r"\qquad\qquad \mathcal{G}(m)=\{g_0,\dots,g_{15}\}$"),
        "20261110-eq_inner_hom": (r"$\displaystyle u_i=\arg\min_{u}\ \underbrace{\sum_k w_k\big(v_{\mathrm{ref},k}-"
                                  r"[\,\textcolor{inner}{A_i(m,g)}\,u+b_i]_k\big)^2}_{\lab{make the contact motion happen}}"
                                  r"\ +\ \underbrace{w_{\mathrm{d}}\lVert u\rVert^2}_{\lab{small joint speeds}}$"),
        "20261110-eq_inner_servo": (r"$\displaystyle q_{\mathrm{cmd}}=\underbrace{q_{\mathrm{ref}}}_{\lab{integrated $u$}}"
                                    r"+\underbrace{\frac{1}{k_p}\,J_i^{\top}F_i\,n_i}_{\lab{press with force $F_i$}}$"),
        "20261110-eq_inner_tune": (r"$\displaystyle \phi=(\textcolor{inner}{F},\textcolor{inner}{w}),\qquad "
                                   r"\phi^{(k)}\sim\mathcal{N}(\mu,\Sigma),\qquad (\mu,\Sigma)\leftarrow"
                                   r"\operatorname{fit}\big(\operatorname{elite}\{\phi^{(k)}\}\big)$"),
        "20261110-eq_sim_budget": (r"$\displaystyle \underbrace{32}_{\lab{new hands}}\times\underbrace{16}_{\lab{grasps}}"
                                   r"\times\underbrace{3}_{\lab{seeds}}=1536\ \ \lab{rollouts per generation}$"),
        "20261110-eq_sim_sensing": (r"$\displaystyle J^{\,\mathrm{sim}}\big(m,g,\phi\mid c,\ p,\ \textcolor{sim}{s(m)}\big)$"),
    }
    for name, tex in eqs.items():
        compile_tex(name, big + tex, border="16pt")


# ------------------------------------------------------------------------------------------------ the nested problem
def nested(active):
    """Three nested boxes: the hand search around the controller choice around the simulator."""
    on = {k: active in ("all", k) for k in ("outer", "inner", "sim")}
    def st(k):
        return (f"fill={k}l, draw={k}, line width=1.6pt" if on[k] else "fill=pale, draw=grey!70, line width=1pt")
    def tc(k):
        return k if on[k] else "grey"
    body = r"""\begin{tikzpicture}[font=\sffamily, every node/.style={inner sep=0pt}]
\fontsize{13}{16}\selectfont
\draw[rounded corners=10pt, %(so)s] (0,0) rectangle (17,9.2);
\node[anchor=north west, text=%(to)s] at (0.45,8.75) {\fontsize{17}{20}\selectfont\bfseries Morphology optimization};
\node[anchor=north west, text=%(to)s] at (0.45,8.05) {outer loop: choose the hand};
\node[anchor=north east, text=%(to)s] at (16.55,8.85) {\fontsize{20}{24}\selectfont$\displaystyle\arg\max_{m\in\mathbb{R}^6}$};
\node[anchor=north east, text=%(to)s, align=right] at (16.55,7.65) {Yang et al.'s evolutionary algorithm};
\draw[rounded corners=9pt, %(si)s] (0.9,0.75) rectangle (16.1,7.05);
\node[anchor=north west, text=%(ti)s] at (1.35,6.6) {\fontsize{17}{20}\selectfont\bfseries Multi-hand control};
\node[anchor=north west, text=%(ti)s] at (1.35,5.9) {inner loop: run this hand};
\node[anchor=north east, text=%(ti)s] at (15.65,6.75) {\fontsize{20}{24}\selectfont$\displaystyle\max_{g,\ \phi}$};
\node[anchor=north east, text=%(ti)s, align=right] at (15.65,5.65) {grasp $g$ and HOM parameters $\phi$};
\draw[rounded corners=8pt, %(ss)s] (1.8,1.45) rectangle (15.2,4.75);
\node[anchor=north west, text=%(ts)s] at (2.25,4.35) {\fontsize{17}{20}\selectfont\bfseries Simulation};
\node[anchor=north west, text=%(ts)s] at (2.25,3.65) {score one rollout};
\node[anchor=north east, text=%(ts)s] at (14.75,4.45) {\fontsize{20}{24}\selectfont$\displaystyle J^{\,\mathrm{sim}}_T(m,g,\phi;\,d\mid c,p,s)$};
\node[anchor=north west, text=%(ts)s] at (2.25,2.95) {contact $c$, servo $p$, sensing $s$: calibrated on hardware};
\node[anchor=north west, text=%(ts)s] at (2.25,2.3) {tasks $T$ and object sizes $d$ drawn from $\mathcal{D}$};
\end{tikzpicture}"""
    return compile_tex(f"20261110-nested_{active}", body % {"so": st("outer"), "si": st("inner"), "ss": st("sim"),
                                                            "to": tc("outer"), "ti": tc("inner"), "ts": tc("sim")})


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


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    formulas()
    level_eqs()
    for a in ("all", "outer", "inner", "sim"):
        nested(a)
    contact_modes()
    hom_period()
    evaluator()
    print("ok", len(list(OUT.glob("20261110-eq_*.png"))), "formula images", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
