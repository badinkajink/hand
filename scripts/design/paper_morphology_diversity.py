"""Design-space diversity figures: how far apart are the eight deployed real_v1 hands?

The paper reports what the eight morphologies DID and never says how different they are.
This script quantifies the separation three ways and writes the figures + a numbers report.

Representation. A configuration is six numbers: the palm-frame (x, y) mount of each of the
three fingers, in millimetres. The linkages are fixed, so the mounts are the whole design
space and Euclidean distance in that space is a physical distance -- the 6-vector norm of how
far the three gantries would have to travel to turn one hand into another.

Two facts drive the figure choices:

* PCA fit on the POPULATION is close to useless here. A scrambled Sobol draw over an axis-
  aligned box has a diagonal covariance, so the principal axes are just the box axes ordered
  by travel: PC1 is thumb_y (the only +-55 mm slide; the rest are +-30) and PC2..PC6 are tied
  near 0.12 each. The PC2 direction is an arbitrary rotation inside a near-isotropic 5-D
  subspace and would move under a reseed. It is plotted because it is the conventional view,
  with the degeneracy stated, not because it separates anything.
* PCA fit on the SURVIVORS is informative, because the funnel is what breaks the symmetry.
  The 535 configurations that pass the retention screen carry [0.47, 0.21, 0.17, 0.08, 0.05,
  0.02] -- the last three directions collapse. Selection concentrates the population onto an
  effectively 3-D sheet, and that is a statement about the task, not about the sampler.

Sources (all read-only, none regenerated here):
  docs/experiments/20260831-real_v1-sobol8192/grasp_screen_manifest.json   8,198 designs
  .../reachable.txt, selected/designs_b*.txt, selected/confirmed/designs_b*.txt   the funnel
  docs/experiments/20260829-real_v1_deploy/deploy/<plan>_plan.json|_traj.csv   the eight
"""
import csv, glob, json, os, re
import numpy as np

SOBOL = "docs/experiments/20260831-real_v1-sobol8192"
DEPLOY = "docs/experiments/20260829-real_v1_deploy/deploy"
OUT = "paper/figures_diversity"

#: paper label -> (internal tag, exported plan).  Table \ref{tab:transfer-map}.
DESIGNS = {
    "D1": ("sv1_w6689", "sv1_w6689_b060"), "D2": ("sv1_w2360", "sv1_w2360_b075"),
    "D3": ("sv1_u1364", "sv1_u1364_b080"), "D4": ("g12", "g12_b095"),
    "D5": ("sv1_u0060", "sv1_u0060_b75"), "D6": ("sv1_u0308", "sv1_u0308_b050"),
    "D7": ("rv05_manual", "rv05_manual_b85"), "D8": ("sv1_w0099", "sv1_w0099_b100"),
}
#: D4 is a hand-placed anchor and D7 a manually specified hand; neither is a Sobol draw.
NOT_SAMPLED = {"D4", "D7"}
COORDS = ["thumb x", "thumb y", "index x", "index y", "middle x", "middle y"]

# dataviz reference palette, light mode. Slots 1-3 are the all-pairs validated subset.
C_THUMB, C_INDEX, C_MIDDLE = "#2a78d6", "#eb6834", "#1baf7a"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8a8a", "#e4e3df", "#ffffff"


def load_population():
    """Return (tags, X[n,6] mounts in mm, source, stage) for all 8,198 configurations."""
    man = json.load(open(f"{SOBOL}/grasp_screen_manifest.json"))
    tags, X, src = [], [], []
    for r in man["designs"]:
        m = r["geometry"]["mounts_mm"]
        tags.append(r["design"]); src.append(r["source"])
        X.append(m["thumb"] + m["index"] + m["middle"])
    tags, X = np.array(tags), np.array(X, float)

    reach = set(open(f"{SOBOL}/reachable.txt").read().split())
    sel, conf = set(), set()
    for f in glob.glob(f"{SOBOL}/selected/designs_b*.txt"):
        sel |= set(open(f).read().strip().split(","))
    for f in glob.glob(f"{SOBOL}/selected/confirmed/designs_b*.txt"):
        conf |= set(open(f).read().strip().split(","))
    # nested: 0 unreachable, 1 reachable, 2 passes retention at some clip, 3 confirmed
    stage = np.zeros(len(tags), int)
    stage[[t in reach for t in tags]] = 1
    stage[[t in sel for t in tags]] = 2
    stage[[t in conf for t in tags]] = 3
    return tags, X, np.array(src), stage


def load_eight(tags, X):
    """Mount vectors, grasp-cell parameters and joint trajectories for D1..D8."""
    idx = {t: i for i, t in enumerate(tags)}
    labs = list(DESIGNS)
    V = np.array([X[idx[DESIGNS[d][0]]] for d in labs])
    cells, trajs = {}, {}
    for d in labs:
        plan = DESIGNS[d][1]
        cells[d] = json.load(open(f"{DEPLOY}/{plan}_plan.json"))["meta"]
        rows = list(csv.DictReader(open(f"{DEPLOY}/{plan}_traj.csv")))
        # palm_z_mm is identically zero on every deployed plan (fixed palm); drop it so it
        # cannot dilute the distance.  index_pip is kept -- it is near-zero but not constant.
        cols = [k for k in rows[0] if k not in ("t_s", "palm_z_mm")]
        trajs[d] = np.array([[float(r[k]) for k in cols] for r in rows])
    return labs, V, cells, trajs, cols


#: the fixed grasp cell the population panels are read at.  Every design in the screen carries
#: this cell, so the design effect is isolated from the choice of cell.
FIXED_CELL = (40.0, 20.0)
FING = ["arm_mm", "reach_mm", "extend_mm", "fn_N", "sweep_mm", "sweep_ratio"]
AGG = ["ceiling_deg", "tau_cap_Nmm", "tau_pair_Nmm", "tau_thumb_Nmm", "sweep_min_mm",
       "sweep_sum_mm", "grip_N", "grip_depth_mm", "grasp_z", "depth_fit_mm"]
GRASP_NAMES = [f"{f}.{k}" for f in ("thumb", "index", "middle") for k in FING] + AGG


def _cell(entry, cell):
    g = [x for x in entry["grasps"]
         if (x["straddle_mm"], x["thumb_axial_mm"]) == tuple(cell)]
    return g[0] if g and g[0].get("pose") else None


def load_grasp_space(cell=FIXED_CELL):
    """Per-design grasp mechanics at one cell: (tags, G[n,28], retained[n]).

    The features are what the closed-form grasp fit reports -- per-finger moment arm, reach,
    extension, normal force, sweep and sweep ratio, plus the aggregates.  They are a derived,
    correlated description of the same six mount numbers, which is why PCA has something to
    find here and nothing to find on the mounts themselves.
    """
    screen = json.load(open(f"{SOBOL}/retention/grasp_screen.json"))
    sel = set()
    for f in glob.glob(f"{SOBOL}/selected/designs_b*.txt"):
        sel |= set(open(f).read().strip().split(","))
    tags, rows, keep = [], [], []
    for r in screen:
        g = _cell(r, cell)
        if g is None:
            continue
        sc = g["scores"]
        v = [sc["fingers"][f][k] for f in ("thumb", "index", "middle") for k in FING] + \
            [sc[k] for k in AGG]
        if any(x is None for x in v):
            continue
        tags.append(r["design"]); rows.append(v); keep.append(r["design"] in sel)
    return np.array(tags), np.array(rows, float), np.array(keep)


def load_confirmed_depth():
    """(depth_fit_mm, nom_cos) for the 535 configurations that pass at some clip."""
    screen = {r["design"]: r for r in json.load(open(f"{SOBOL}/retention/grasp_screen.json"))}
    out = []
    for f in sorted(glob.glob(f"{SOBOL}/confirm_b*.json")):
        for r in json.load(open(f)):
            g = _cell(screen[r["design"]], (r["straddle_mm"], r["thumb_axial_mm"]))
            d = g["scores"].get("depth_fit_mm") if g else None
            if d is not None and r.get("nom_cos") is not None:
                out.append((d, r["nom_cos"]))
    return np.array(out, float)


#: the published, corrected bench table.  data.json in the transfer-firstpass directory is a day
#: older than the 2026-09-01 session-selection fix and disagrees with it, so the table is the
#: source of truth for anything that reaches the paper.
BENCH_ROW = re.compile(r"\\texttt\{(D\d)\}(?:\$\^\{\*\}\$)?\s*&\s*\$([\d.]+)\^\{\d\}\$"
                       r"\s*&\s*(\d+)/(\d+)\s*&\s*(\d+)\s*&\s*\$([\d.]+)\\pm([\d.]+)\^\{\d\}\$")


def load_bench(path="paper/transfer_table.tex"):
    out = {}
    for m in BENCH_ROW.finditer(open(path).read()):
        d, sim, held, n, meas, bc, bsd = m.groups()
        out[d] = dict(sim_cos=float(sim), n=int(n), n_held=int(held),
                      hold_rate=int(held) / int(n), n_meas=int(meas),
                      bench_cos=float(bc), bench_sd=float(bsd))
    if len(out) != 8:
        raise SystemExit(f"parsed {len(out)} bench rows from {path}, expected 8")
    return out


def auc(score, label):
    """Rank AUC of `score` for the boolean `label` -- no sklearn dependency."""
    r = np.argsort(np.argsort(score)); n1 = int(label.sum()); n0 = len(label) - n1
    return (r[label].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)


def pca(A):
    """Return (mean, components[6,6], explained_variance_ratio) of A[n,6]."""
    mu = A.mean(0)
    U, S, Vt = np.linalg.svd(A - mu, full_matrices=False)
    ev = S ** 2 / (len(A) - 1)
    return mu, Vt, ev / ev.sum()


def subset_null(P, k=8, n=20000, seed=7):
    """Median and minimum pairwise distance of random k-subsets of P -- the spread null."""
    rng = np.random.default_rng(seed)
    iu = np.triu_indices(k, 1)
    med, mn = np.empty(n), np.empty(n)
    for i in range(n):
        s = P[rng.choice(len(P), k, replace=False)]
        d = np.linalg.norm(s[:, None] - s[None], axis=-1)[iu]
        med[i], mn[i] = np.median(d), d.min()
    return med, mn


def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 7.2, "axes.labelsize": 7.2, "axes.titlesize": 7.6,
        "xtick.labelsize": 6.6, "ytick.labelsize": 6.6, "legend.fontsize": 6.6,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
        "grid.color": GRID, "grid.linewidth": 0.5,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    })
    return plt


def label_points(ax, xy, labs, size=6.4, seed=0, gap=3.0, avoid_legend=True):
    """Direct labels placed by candidate search -- identity is carried by the text, not colour.

    For each label a ring of candidate offsets around its own marker is scored against the
    labels already placed, every marker, and the axes border; the cheapest is taken.  This is
    the usual cartographic placement and it terminates, where the force iteration it replaced
    oscillated: a two-character label is twice as wide as it is tall, so pushing on a circular
    pad left horizontally adjacent labels touching and pushing on the cheaper axis cycled.
    Markers never move; a label more than about two line-heights from home grows a leader.
    Any legend already on the axes is treated as an obstacle, so create the legend first.
    """
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    pts = np.asarray(xy, float)
    home = np.array([ax.transData.transform(p) for p in pts])
    h = size * fig.dpi / 72.0

    arts = []
    for p0, d in zip(pts, labs):
        arts.append(ax.annotate(d, tuple(p0), fontsize=size, color=INK, weight="bold",
                                ha="center", va="center", zorder=6))
    fig.canvas.draw()
    dims = np.array([[t.get_window_extent(r).width, t.get_window_extent(r).height]
                     for t in arts])
    ab = ax.get_window_extent(r)
    blocked = []
    if avoid_legend and ax.get_legend() is not None:
        lb = ax.get_legend().get_window_extent(r)
        blocked.append((lb.x0 - gap, lb.y0 - gap, lb.x1 + gap, lb.y1 + gap))

    def box(c, wh):
        return (c[0] - wh[0] / 2 - gap, c[1] - wh[1] / 2 - gap,
                c[0] + wh[0] / 2 + gap, c[1] + wh[1] / 2 + gap)

    def overlap(a, b):
        return (max(0.0, min(a[2], b[2]) - max(a[0], b[0])) *
                max(0.0, min(a[3], b[3]) - max(a[1], b[1])))

    angles = np.deg2rad([270, 90, 0, 180, 315, 225, 45, 135,
                         293, 247, 23, 157, 337, 203, 67, 113])
    radii = [1.5, 2.2, 3.0, 4.0, 5.2, 6.6, 8.2, 10.0]
    def search(i, others):
        """Cheapest ring position for label i against the boxes in `others`."""
        wh, best, bestcost = dims[i], None, None
        for rad in radii:
            for a in angles:
                c = home[i] + np.array([np.cos(a), np.sin(a)]) * rad * h
                bb = box(c, wh)
                cost = 40.0 * rad                                    # prefer close to home
                cost += 500.0 * sum(overlap(bb, q) for q in others)   # near-hard
                cost += 500.0 * sum(overlap(bb, q) for q in blocked)  # legend box
                for k in range(len(home)):                            # every marker
                    cost += 200.0 * overlap(bb, box(home[k], np.array([5.0, 5.0])))
                dx = max(0.0, ab.x0 + 2 - bb[0]) + max(0.0, bb[2] - ab.x1 + 2)
                dy = max(0.0, ab.y0 + 2 - bb[1]) + max(0.0, bb[3] - ab.y1 + 2)
                cost += 300.0 * (dx + dy)                             # stay inside the axes
                if bestcost is None or cost < bestcost:
                    best, bestcost = c, cost
            if bestcost is not None and bestcost <= 40.0 * rad + 1e-6:
                break                                                 # clean slot at this radius
        return best, bestcost

    order = np.argsort(-home[:, 1])          # deterministic: top-down
    out = np.zeros_like(home)
    placed = []
    for i in order:
        best, _ = search(i, placed)
        placed.append(box(best, dims[i])); out[i] = best
    # Repair: greedy placement can box a late label in, because it only ever saw the labels
    # placed before it.  Re-search each one against every OTHER final box until nothing moves.
    for _ in range(24):
        boxes = [box(out[k], dims[k]) for k in range(len(home))]
        worst, wi = 0.0, None
        for i in range(len(home)):
            ov = sum(overlap(boxes[i], boxes[j]) for j in range(len(home)) if j != i)
            if ov > worst:
                worst, wi = ov, i
        if wi is None:
            break
        best, _ = search(wi, [boxes[j] for j in range(len(home)) if j != wi])
        if np.allclose(best, out[wi]):
            break
        out[wi] = best

    for t, c, p0 in zip(arts, out, pts):
        t.set_position(ax.transData.inverted().transform(c))
        if np.hypot(*(c - ax.transData.transform(p0))) > 2.1 * h:
            ax.annotate("", xy=tuple(p0), xytext=ax.transData.inverted().transform(c),
                        arrowprops=dict(arrowstyle="-", lw=0.4, color=MUTED,
                                        shrinkA=1.5, shrinkB=3.0), zorder=5)


# ------------------------------------------------------------------ 1. PCA of the population
def fig_pca(plt, X, stage, V, labs, rep):
    mu_p, W_p, ev_p = pca(X)
    surv = X[stage >= 2]
    mu_s, W_s, ev_s = pca(surv)

    fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.32))

    for a, (mu, W, ev, ttl) in enumerate([
            (mu_p, W_p, ev_p, "fit on all 8,198 sampled"),
            (mu_s, W_s, ev_s, "fit on the 535 that pass retention")]):
        A = ax[a]
        P = (X - mu) @ W[:2].T
        # nested strata, recessive -> emphatic
        A.scatter(*P[stage == 0].T, s=1.2, c=GRID, lw=0, rasterized=True, zorder=1)
        A.scatter(*P[stage == 1].T, s=1.2, c=SEQ[1], lw=0, rasterized=True, zorder=2)
        A.scatter(*P[stage >= 2].T, s=3.0, c=SEQ[4], lw=0, rasterized=True, zorder=3)
        Q = (V - mu) @ W[:2].T
        A.scatter(*Q.T, s=26, facecolor="none", edgecolor=INK, lw=1.0, zorder=5)
        label_points(A, Q, labs)
        A.set_xlabel(f"PC1  ({ev[0]*100:.0f} %)")
        A.set_ylabel(f"PC2  ({ev[1]*100:.0f} %)" if a == 0 else "")
        A.set_title(ttl, color=INK2, pad=4)
        A.grid(True, lw=0.4, zorder=0)
        A.set_axisbelow(True)
        if a == 0:
            for c, t in [(GRID, "outside the rails"), (SEQ[1], "reachable"),
                         (SEQ[4], "passes retention")]:
                A.scatter([], [], s=7, c=c, lw=0, label=t)
            A.scatter([], [], s=22, facecolor="none", edgecolor=INK, lw=1.0,
                      label="deployed (D1--D8)")
            lg = A.legend(loc="upper left", handlelength=1.0, borderpad=0.25,
                          labelspacing=0.25, fontsize=5.9, framealpha=0.92,
                          facecolor=SURF, edgecolor="none")
            lg.set_zorder(7)

    A = ax[2]
    w, xs = 0.38, np.arange(1, 7)
    A.bar(xs - w / 2, ev_p * 100, w, color=SEQ[1], lw=0, label="all 8,198")
    A.bar(xs + w / 2, ev_s * 100, w, color=SEQ[4], lw=0, label="535 retained")
    A.axhline(100 / 6, color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=0)
    A.annotate("isotropic, 16.7 %", (6.35, 100 / 6), fontsize=6.0, color=MUTED,
               ha="right", va="bottom")
    A.set_xlabel("principal component"); A.set_ylabel("variance explained (%)")
    A.set_xticks(xs); A.set_title("selection collapses three directions", color=INK2, pad=4)
    A.legend(frameon=False, loc="upper right", handlelength=1.1)
    A.grid(True, axis="y", lw=0.4, zorder=0); A.set_axisbelow(True)

    fig.tight_layout(pad=0.35, w_pad=1.1)
    fig.savefig(f"{OUT}/fig_design_pca.pdf"); fig.savefig(f"{OUT}/fig_design_pca.png", dpi=200)
    plt.close(fig)

    rep.append("PCA (mount space, mm)")
    rep.append(f"  population n=8198  ratio {np.round(ev_p, 3).tolist()}")
    rep.append(f"  survivors  n={len(surv)}   ratio {np.round(ev_s, 3).tolist()}")
    for k in range(3):
        rep.append(f"  survivor PC{k+1} loadings " +
                   ", ".join(f"{c} {v:+.2f}" for c, v in zip(COORDS, W_s[k])))
    return ev_p, ev_s


# ----------------------------------------------------------- 2. the mounts, in the palm plane
def fig_mount_plane(plt, X, stage, V, labs):
    """The design space as it physically is: three gantry travels seen from above the palm.

    Faceted by finger.  The three travels are disjoint regions of the palm, so drawing them on
    one axes and joining each hand's three mounts into a tripod produces 24 crossing chords and
    hides the thing the figure is for.  Each panel spans the same 74 x 124 mm window, so box
    sizes are comparable across panels: the thumb rail is the long one.
    """
    surv = X[stage >= 2]
    fingers = [("thumb", 0, C_THUMB), ("index", 2, C_INDEX), ("middle", 4, C_MIDDLE)]
    fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.78))
    for A, (name, j, col) in zip(ax, fingers):
        lo, hi = X[:, j:j + 2].min(0), X[:, j:j + 2].max(0)
        cx, cy = (lo + hi) / 2
        A.add_patch(plt.Rectangle(lo, *(hi - lo), fill=False, ec=MUTED, lw=0.7,
                                  ls=(0, (3, 2)), zorder=2))
        A.scatter(*X[stage == 1][:, j:j + 2].T, s=1.4, c=GRID, lw=0, rasterized=True, zorder=1)
        A.scatter(*surv[:, j:j + 2].T, s=3.0, c=col, alpha=0.30, lw=0,
                  rasterized=True, zorder=3)
        A.scatter(*V[:, j:j + 2].T, s=17, c=col, lw=0.5, ec=SURF, zorder=5)
        label_points(A, V[:, j:j + 2], labs, size=5.7)
        A.set_xlim(cx - 37, cx + 37); A.set_ylim(cy - 62, cy + 62)
        A.set_aspect("equal"); A.grid(True, lw=0.4, zorder=0); A.set_axisbelow(True)
        A.set_title(f"{name} mount", color=col, pad=4, weight="bold")
        A.set_xlabel("palm-frame x (mm)")
    ax[0].set_ylabel("palm-frame y (mm)")
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], ls="", marker="o", ms=2.6, mfc=GRID, mec="none", label="reachable"),
        Line2D([], [], ls="", marker="o", ms=3.4, mfc=C_THUMB, mec="none", alpha=0.6,
               label="passes retention (in this panel's colour)"),
        Line2D([], [], ls=(0, (3, 2)), lw=0.7, color=MUTED, label="gantry travel"),
        Line2D([], [], ls="", marker="o", ms=4.2, mfc=C_THUMB, mec=SURF, mew=0.5,
               label="deployed (D1--D8)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=6.0,
               handlelength=1.3, columnspacing=1.5, borderpad=0.0, bbox_to_anchor=(0.5, -0.012))
    fig.tight_layout(pad=0.35, w_pad=1.0, rect=(0, 0.062, 1, 1))
    fig.savefig(f"{OUT}/fig_mount_plane.pdf"); fig.savefig(f"{OUT}/fig_mount_plane.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------- 3. Euclidean separation, and a spread null
def fig_distance(plt, X, stage, V, labs, rep):
    iu = np.triu_indices(8, 1)
    Dm = np.linalg.norm(V[:, None] - V[None], axis=-1)
    obs = Dm[iu]

    fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.55),
                           gridspec_kw={"width_ratios": [1.0, 1.28]})
    A = ax[0]
    M = np.ma.masked_where(np.eye(8, dtype=bool), Dm)
    im = A.imshow(M, cmap="Blues", vmin=0, vmax=100, zorder=2)
    for i in range(8):
        for j in range(8):
            if i == j:
                continue
            A.annotate(f"{Dm[i,j]:.0f}", (j, i), ha="center", va="center", fontsize=5.9,
                       color=SURF if Dm[i, j] > 58 else INK, zorder=3)
    A.set_xticks(range(8), labs); A.set_yticks(range(8), labs)
    A.set_title("pairwise mount separation (mm)", color=INK2, pad=4)
    A.tick_params(length=0)
    for s in A.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=A, fraction=0.045, pad=0.03)
    cb.outline.set_visible(False); cb.ax.tick_params(length=2, labelsize=6.0)

    A = ax[1]
    pools = [("all 8,198 sampled", X, SEQ[1]), ("6,629 reachable", X[stage >= 1], SEQ[3]),
             ("535 retained", X[stage >= 2], SEQ[5])]
    for name, P, col in pools:
        med, _ = subset_null(P)
        A.hist(med, bins=60, histtype="step", lw=1.0, color=col, label=name, zorder=2)
        rep.append(f"  null median spread, {name:20s} {np.median(med):5.1f} mm "
                   f"[{np.percentile(med,5):.0f}, {np.percentile(med,95):.0f}]  "
                   f"observed percentile {100*(med < np.median(obs)).mean():.1f}")
    A.axvline(np.median(obs), color=INK, lw=1.2, zorder=4)
    A.annotate(f"the eight:\n{np.median(obs):.1f} mm", (np.median(obs) - 1.6, A.get_ylim()[1] * 0.92),
               fontsize=6.4, color=INK, weight="bold", ha="right", va="top", zorder=5)
    A.set_xlabel("median pairwise separation of a random eight (mm)")
    A.set_ylabel("draws")
    A.set_title("the eight are closer together than chance", color=INK2, pad=4)
    A.legend(frameon=False, loc="upper right", handlelength=1.2)
    A.grid(True, axis="y", lw=0.4, zorder=0); A.set_axisbelow(True)

    fig.tight_layout(pad=0.35, w_pad=1.4)
    fig.savefig(f"{OUT}/fig_design_distance.pdf")
    fig.savefig(f"{OUT}/fig_design_distance.png", dpi=200)
    plt.close(fig)

    # the same statistic over the six Sobol members alone -- the anchor and the manual hand
    # are not what makes the set narrow, and the Scope paragraph says so on these numbers.
    six = [i for i, d in enumerate(labs) if d not in NOT_SAMPLED]
    iu6 = np.triu_indices(len(six), 1)
    o6 = np.linalg.norm(V[six][:, None] - V[six][None], axis=-1)[iu6]
    rep.append("Mount separation among the six Sobol members only (mm)")
    rep.append(f"  min {o6.min():.1f}  median {np.median(o6):.1f}  max {o6.max():.1f}")
    rng6 = np.random.default_rng(11)
    for name, P, _c in pools:
        m6 = np.array([np.median(np.linalg.norm(
            (q := P[rng6.choice(len(P), len(six), replace=False)])[:, None] - q[None],
            axis=-1)[iu6]) for _ in range(20000)])
        rep.append(f"  null median spread, {name:20s} {np.median(m6):5.1f} mm  "
                   f"observed percentile {100*(m6 < np.median(o6)).mean():.1f}")
    rep.append("Mount separation among the eight (mm)")
    rep.append(f"  min {obs.min():.1f} ({labs[iu[0][obs.argmin()]]}-{labs[iu[1][obs.argmin()]]})"
               f"  median {np.median(obs):.1f}"
               f"  max {obs.max():.1f} ({labs[iu[0][obs.argmax()]]}-{labs[iu[1][obs.argmax()]]})")
    return Dm


# ------------------------------------------------ 4. the deployed behaviour, not the geometry
CELL_KEYS = [("straddle_mm", "straddle (mm)"), ("thumb_axial_mm", "thumb axial (mm)"),
             ("grip_depth_mm", "grip depth (mm)"), ("axis_k", "turn axis $k$"),
             ("angle_deg", "turn angle (deg)"), ("budget_rad", "residual clip (rad)")]


def fig_behaviour(plt, labs, cells, trajs, cols, Dm, rep):
    """Does a different hand get a different grasp and a different motion? Yes, and predictably.

    No PCA here.  Eight trajectories in 1,791 dimensions admit seven principal components, so a
    2-D score plot is an arbitrary rotation of eight points -- in the first draft three designs
    tied on PC2 to within 0.1, which is a coincidence of the projection and not a fact about
    the hands.  The rms joint difference is the same information in degrees.
    """
    T = np.stack([trajs[d].ravel() for d in labs])
    iu = np.triu_indices(8, 1)
    # rms per-joint, per-sample difference in degrees -- physical units, no projection
    Dt = np.linalg.norm(T[:, None] - T[None], axis=-1) / np.sqrt(T.shape[1])

    fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.55),
                           gridspec_kw={"width_ratios": [1.0, 1.18, 0.92]})

    A = ax[0]
    M = np.ma.masked_where(np.eye(8, dtype=bool), Dt)
    im = A.imshow(M, cmap="Blues", vmin=0, vmax=32, zorder=2)
    for i in range(8):
        for j in range(8):
            if i != j:
                A.annotate(f"{Dt[i,j]:.0f}", (j, i), ha="center", va="center", fontsize=5.9,
                           color=SURF if Dt[i, j] > 19 else INK, zorder=3)
    A.set_xticks(range(8), labs); A.set_yticks(range(8), labs); A.tick_params(length=0)
    A.set_title("trajectory difference (deg rms)", color=INK2, pad=4)
    for sp in A.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=A, fraction=0.045, pad=0.03)
    cb.outline.set_visible(False); cb.ax.tick_params(length=2, labelsize=6.0)

    # what the per-design optimiser actually chose, one row per knob, colour = within-row rank
    A = ax[1]
    Vals = np.array([[float(cells[d][k]) for d in labs] for k, _ in CELL_KEYS])
    N = np.zeros_like(Vals)
    for i, row in enumerate(Vals):
        lo, hi = row.min(), row.max()
        N[i] = 0.5 if hi == lo else (row - lo) / (hi - lo)
    A.imshow(N, cmap="Blues", vmin=-0.15, vmax=1.12, aspect="auto", zorder=2)
    for i in range(len(CELL_KEYS)):
        for j in range(8):
            v = Vals[i, j]
            A.annotate(f"{v:g}", (j, i), ha="center", va="center", fontsize=5.7,
                       color=SURF if N[i, j] > 0.62 else INK, zorder=3)
    A.set_xticks(range(8), labs)
    A.set_yticks(range(len(CELL_KEYS)), [t for _, t in CELL_KEYS])
    A.tick_params(length=0)
    A.set_title("the grasp and turn each hand was given", color=INK2, pad=4)
    for sp in A.spines.values():
        sp.set_visible(False)

    A = ax[2]
    from scipy import stats as st
    rho, p = st.spearmanr(Dm[iu], Dt[iu])
    b = np.polyfit(Dm[iu], Dt[iu], 1)
    xs = np.array([Dm[iu].min(), Dm[iu].max()])
    A.plot(xs, np.polyval(b, xs), lw=0.9, color=MUTED, ls=(0, (3, 2)), zorder=2)
    A.scatter(Dm[iu], Dt[iu], s=14, c=SEQ[4], lw=0, zorder=3)
    ptxt = "$p$ < 0.0001" if p < 1e-4 else f"$p$ = {p:.4f}"
    A.annotate(f"$\\rho_s$ = {rho:.2f},  {ptxt}\n28 pairs", (0.96, 0.05),
               xycoords="axes fraction", fontsize=6.4, color=INK, ha="right", va="bottom")
    A.set_xlabel("mount separation (mm)")
    A.set_ylabel("trajectory difference (deg rms)")
    A.set_title("geometry predicts motion", color=INK2, pad=4)
    A.grid(True, lw=0.4, zorder=0); A.set_axisbelow(True)

    fig.tight_layout(pad=0.35, w_pad=1.3)
    fig.savefig(f"{OUT}/fig_behaviour.pdf"); fig.savefig(f"{OUT}/fig_behaviour.png", dpi=200)
    plt.close(fig)

    rep.append("Deployed behaviour")
    rep.append(f"  trajectory difference (deg rms): min {Dt[iu].min():.1f} "
               f"({labs[iu[0][Dt[iu].argmin()]]}-{labs[iu[1][Dt[iu].argmin()]]})"
               f"  median {np.median(Dt[iu]):.1f}  max {Dt[iu].max():.1f} "
               f"({labs[iu[0][Dt[iu].argmax()]]}-{labs[iu[1][Dt[iu].argmax()]]})")
    rep.append(f"  spearman(mount separation, trajectory difference) = {rho:.3f}, p = {p:.2e}")
    rep.append("  grasp cell / turn parameters actually deployed:")
    rep.append("    " + "design".ljust(8) + "".join(k.rjust(15) for k, _ in CELL_KEYS))
    for j, d in enumerate(labs):
        rep.append("    " + d.ljust(8) + "".join(f"{Vals[i, j]:15g}"
                                                 for i in range(len(CELL_KEYS))))
    for i, (k, _) in enumerate(CELL_KEYS):
        rep.append(f"    {k:16s} distinct {len(set(Vals[i])):d}  "
                   f"range {Vals[i].min():g} .. {Vals[i].max():g}")


# --------------------------------------- 5. grasp mechanics: what the screen selects, and the bench
def fig_grasp_space(plt, labs, cells, rep):
    """The grasp descriptor is where PCA earns its place, and depth of fit is the whole story.

    Unlike the mount coordinates -- an independent uniform draw, so their covariance is diagonal
    by construction -- the grasp features are derived and correlated, and PCA finds real
    structure: 28 features, 6 components for 95 %, which is exactly the dimension of the design
    space they were computed from.
    """
    gtags, G, kept = load_grasp_space()
    Z = (G - G.mean(0)) / G.std(0)
    mu, W, ev = pca(Z)
    P = (Z - mu) @ W[:2].T
    idx = {t: i for i, t in enumerate(gtags)}
    Q = np.array([P[idx[DESIGNS[d][0]]] for d in labs])
    depth = G[:, GRASP_NAMES.index("depth_fit_mm")]
    a_depth = auc(depth, kept)

    conf = load_confirmed_depth()
    bench = load_bench()
    screen = {r["design"]: r for r in json.load(open(f"{SOBOL}/retention/grasp_screen.json"))}
    dep8 = np.array([_cell(screen[DESIGNS[d][0]],
                           (cells[d]["straddle_mm"], cells[d]["thumb_axial_mm"]))
                     ["scores"]["depth_fit_mm"] for d in labs])

    fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.52),
                           gridspec_kw={"width_ratios": [1.0, 0.92, 1.06]})

    A = ax[0]
    A.scatter(*P[~kept].T, s=1.6, c=GRID, lw=0, rasterized=True, zorder=1, label="not retained")
    A.scatter(*P[kept].T, s=3.0, c=SEQ[4], lw=0, rasterized=True, zorder=2, label="retained")
    A.scatter(*Q.T, s=24, facecolor="none", edgecolor=INK, lw=1.0, zorder=4,
              label="deployed (D1--D8)")
    A.set_xlabel(f"grasp PC1  ({ev[0]*100:.0f} %)")
    A.set_ylabel(f"grasp PC2  ({ev[1]*100:.0f} %)")
    A.set_title(f"28 grasp features, {int(np.searchsorted(np.cumsum(ev), 0.95))+1} PCs for 95 %",
                color=INK2, pad=4)
    lg = A.legend(loc="upper left", fontsize=5.9, handlelength=1.0, borderpad=0.25,
                  labelspacing=0.25, framealpha=0.92, facecolor=SURF, edgecolor="none")
    lg.set_zorder(7)
    A.grid(True, lw=0.4, zorder=0); A.set_axisbelow(True)
    label_points(A, Q, labs)

    A = ax[1]
    bins = np.linspace(depth.min(), depth.max(), 42)
    A.hist(depth[~kept], bins=bins, density=True, histtype="stepfilled", lw=0,
           color=GRID, zorder=1, label=f"not retained ($n$={(~kept).sum()})")
    A.hist(depth[kept], bins=bins, density=True, histtype="step", lw=1.2,
           color=SEQ[4], zorder=3, label=f"retained ($n$={kept.sum()})")
    A.axvline(np.median(depth[kept]), color=SEQ[5], lw=0.8, ls=(0, (3, 2)), zorder=4)
    A.axvline(np.median(depth[~kept]), color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=4)
    A.annotate(f"AUC {a_depth:.3f}\nmedian {np.median(depth[kept]):.1f} vs "
               f"{np.median(depth[~kept]):.1f} mm", (0.97, 0.62), xycoords="axes fraction",
               fontsize=6.2, color=INK, ha="right", va="top", zorder=5)
    A.set_xlabel("depth of grasp fit (mm)"); A.set_ylabel("density")
    A.set_title("one feature carries the screen", color=INK2, pad=4)
    A.legend(loc="upper left", frameon=False, fontsize=5.9, handlelength=1.0,
             labelspacing=0.25, borderpad=0.1)
    A.grid(True, axis="y", lw=0.4, zorder=0); A.set_axisbelow(True)

    A = ax[2]
    from scipy import stats as st
    rs_sim = st.spearmanr(conf[:, 0], conf[:, 1])
    bc = np.array([bench[d]["bench_cos"] for d in labs])
    bsd = np.array([bench[d]["bench_sd"] for d in labs])
    rs_ben = st.spearmanr(dep8, bc)
    A.scatter(conf[:, 0], conf[:, 1], s=2.4, c=SEQ[2], lw=0, rasterized=True, zorder=1,
              label=f"sim, confirmed ($n$={len(conf)})")
    A.errorbar(dep8, bc, yerr=bsd, fmt="o", ms=4.0, mfc=C_INDEX, mec=SURF, mew=0.5,
               ecolor=C_INDEX, elinewidth=0.8, capsize=1.6, zorder=4,
               label="bench, deployed ($n$=8)")
    A.annotate(f"sim  $\\rho_s$ = {rs_sim.statistic:+.2f} ($p$ = {rs_sim.pvalue:.2f})\n"
               f"bench $\\rho_s$ = {rs_ben.statistic:+.2f} ($p$ = {rs_ben.pvalue:.2f})",
               (0.03, 0.03), xycoords="axes fraction", fontsize=6.0, color=INK,
               ha="left", va="bottom", zorder=6)
    A.set_xlabel("depth of grasp fit (mm)"); A.set_ylabel("alignment $\\cos$")
    A.set_title("and none of the alignment", color=INK2, pad=4)
    A.legend(loc="upper right", fontsize=5.9, handlelength=1.0, labelspacing=0.25,
             borderpad=0.25, framealpha=0.92, facecolor=SURF, edgecolor="none").set_zorder(7)
    A.grid(True, lw=0.4, zorder=0); A.set_axisbelow(True)
    label_points(A, np.column_stack([dep8, bc]), labs, size=5.8)

    fig.tight_layout(pad=0.35, w_pad=1.2)
    fig.savefig(f"{OUT}/fig_grasp_space.pdf")
    fig.savefig(f"{OUT}/fig_grasp_space.png", dpi=200)
    plt.close(fig)

    rep.append("Grasp mechanics")
    rep.append(f"  feature matrix {G.shape} at cell {FIXED_CELL}, retained {int(kept.sum())}")
    rep.append(f"  explained {np.round(ev, 3)[:8].tolist()}")
    rep.append(f"  components for 95 % {int(np.searchsorted(np.cumsum(ev), 0.95))+1} "
               f"(design space is 6-dimensional)")
    for k in range(3):
        o = np.argsort(-np.abs(W[k]))[:5]
        rep.append(f"  grasp PC{k+1}: " +
                   ", ".join(f"{GRASP_NAMES[i]} {W[k][i]:+.2f}" for i in o))
    rep.append(f"  AUC(depth_fit -> retained) {a_depth:.3f}; median retained "
               f"{np.median(depth[kept]):.1f} mm vs {np.median(depth[~kept]):.1f} mm")
    rep.append(f"  depth_fit vs sim nom_cos over {len(conf)} confirmed: "
               f"rho {rs_sim.statistic:+.3f} p {rs_sim.pvalue:.3f}  -- flat")
    rep.append(f"  depth_fit vs BENCH cos over the 8: rho {rs_ben.statistic:+.2f} "
               f"p {rs_ben.pvalue:.3f}")
    hr = np.array([bench[d]["hold_rate"] for d in labs])
    r2 = st.spearmanr(dep8, hr)
    rep.append(f"  depth_fit vs BENCH hold rate over the 8: rho {r2.statistic:+.2f} "
               f"p {r2.pvalue:.3f}")
    sb = st.spearmanr([bench[d]["sim_cos"] for d in labs], bc)
    rep.append(f"  sim cos vs bench cos over the 8: rho {sb.statistic:+.2f} p {sb.pvalue:.3f}")
    rep.append("  deployed depth_fit at each hand's OWN cell: " +
               ", ".join(f"{d} {v:.1f}" for d, v in zip(labs, dep8)))


# ------------------------------------- 6. the appendix plate: eight hands, drawn side by side
def finger_motion(traj, cols):
    """Total commanded travel of each finger over the plan, in degrees summed across its joints."""
    out = {}
    for f in ("thumb", "index", "middle"):
        j = [i for i, c in enumerate(cols) if c.startswith(f)]
        out[f] = float(np.abs(traj[-1, j] - traj[0, j]).sum())
    return out


def fig_hand_cards(plt, X, stage, labs, V, cells, trajs, cols, rep):
    """One card per deployed hand: the tripod it is, against the tripod the population usually
    draws, with each finger's share of the turn on the mount that does it.

    Three channels, one picture.  Geometry is the triangle (and the grey ghost behind it is the
    median sampled hand, so a card reads as a deviation rather than as an absolute).  The grasp
    is the depth bar.  The trajectory is the disc area at each mount: total commanded travel of
    that finger over the plan.  The middle finger carries 45-76 % of it on every hand.
    """
    med = np.median(X[stage >= 1], axis=0)
    fingers = [("thumb", 0, C_THUMB), ("index", 2, C_INDEX), ("middle", 4, C_MIDDLE)]
    mot = {d: finger_motion(trajs[d], cols) for d in labs}
    mmax = max(v for m in mot.values() for v in m.values())
    rad = lambda v: 9.5 * np.sqrt(v / mmax)          # area proportional to commanded travel
    depths = np.array([cells[d]["grip_depth_mm"] for d in labs], float)
    d0, d1 = 45.0, 70.0

    fig, ax = plt.subplots(2, 4, figsize=(7.16, 3.62))
    for k, d in enumerate(labs):
        A = ax[k // 4][k % 4]
        for name, j, col in fingers:                       # the rails, as pale context
            lo, hi = X[:, j:j + 2].min(0), X[:, j:j + 2].max(0)
            A.add_patch(plt.Rectangle(lo, *(hi - lo), fc=GRID, ec="none", alpha=0.42, zorder=1))
        A.add_patch(plt.Polygon(med.reshape(3, 2), closed=True, fill=False, ec=MUTED,
                                lw=0.7, ls=(0, (2.5, 2)), zorder=2))
        A.add_patch(plt.Polygon(V[k].reshape(3, 2), closed=True, fill=False, ec=INK,
                                lw=0.9, zorder=4))
        for name, j, col in fingers:
            A.add_patch(plt.Circle(V[k, j:j + 2], rad(mot[d][name]), fc=col, ec=SURF,
                                   lw=0.5, alpha=0.85, zorder=5))
        A.set_xlim(-95, 95); A.set_ylim(-101, 106)
        A.set_aspect("equal")
        A.set_xticks([]); A.set_yticks([])
        for sp in A.spines.values():
            sp.set_color(GRID)
        A.annotate(d, (0.035, 0.965), xycoords="axes fraction", fontsize=8.2, weight="bold",
                   color=INK, ha="left", va="top", zorder=7)
        A.annotate(f"clip {cells[d]['budget_rad']:.2f} rad", (0.965, 0.965),
                   xycoords="axes fraction", fontsize=5.9, color=INK2, ha="right", va="top")
        if d in NOT_SAMPLED:                        # provenance, where it is not a Sobol draw
            A.annotate("anchor" if d == "D4" else "manual", (0.038, 0.885),
                       xycoords="axes fraction", fontsize=5.6, color=MUTED,
                       style="italic", ha="left", va="top")
        # grip depth, as a track across the foot of the card
        t = float(np.clip((depths[k] - d0) / (d1 - d0), 0, 1))
        A.plot([0.07, 0.79], [0.055, 0.055], transform=A.transAxes, lw=2.4, color=GRID,
               solid_capstyle="butt", zorder=6)
        A.plot([0.07, 0.07 + 0.72 * t], [0.055, 0.055], transform=A.transAxes, lw=2.4,
               color=SEQ[4], solid_capstyle="butt", zorder=7)
        A.annotate(f"{depths[k]:.1f}", (0.82, 0.055), xycoords="axes fraction", fontsize=5.9,
                   color=INK2, ha="left", va="center", zorder=7)

    from matplotlib.lines import Line2D
    handles = [Line2D([], [], ls=(0, (2.5, 2)), lw=0.7, color=MUTED,
                      label="median sampled hand"),
               Line2D([], [], lw=0.9, color=INK, label="this hand")]
    handles += [Line2D([], [], ls="", marker="o", ms=4.5, mfc=c, mec=SURF, mew=0.5, label=n)
                for n, _, c in fingers]
    handles += [Line2D([], [], lw=2.4, color=SEQ[4], label=f"grip depth, {d0:.0f}-{d1:.0f} mm")]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=6.0,
               handlelength=1.4, columnspacing=1.3, borderpad=0.0,
               bbox_to_anchor=(0.5, 0.034))
    fig.text(0.5, 0.012, "disc area is proportional to that finger's total commanded travel "
             "over the plan; the largest is D8's middle finger at 140 deg",
             ha="center", va="bottom", fontsize=5.8, color=INK2)
    fig.tight_layout(pad=0.25, w_pad=0.35, h_pad=1.0, rect=(0, 0.102, 1, 1))
    fig.savefig(f"{OUT}/fig_hand_cards.pdf"); fig.savefig(f"{OUT}/fig_hand_cards.png", dpi=200)
    plt.close(fig)

    rep.append("Hand cards (appendix plate)")
    rep.append(f"  median sampled tripod (mm): " +
               ", ".join(f"{c} {v:.1f}" for c, v in zip(COORDS, med)))
    rep.append("  commanded travel per finger (deg) and the middle finger's share:")
    for d in labs:
        m = mot[d]; tot = sum(m.values())
        rep.append(f"    {d}  thumb {m['thumb']:6.1f}  index {m['index']:6.1f}  "
                   f"middle {m['middle']:6.1f}  total {tot:6.1f}  middle share "
                   f"{m['middle']/tot:.2f}  thumb share {m['thumb']/tot:.2f}")
    sh = [mot[d]["middle"] / sum(mot[d].values()) for d in labs]
    th = [mot[d]["thumb"] / sum(mot[d].values()) for d in labs]
    tt = [sum(mot[d].values()) for d in labs]
    rep.append(f"  middle share {min(sh):.2f}-{max(sh):.2f}; thumb share {min(th):.2f}-"
               f"{max(th):.2f}; total travel {min(tt):.0f}-{max(tt):.0f} deg")


# ---------------------------------- 7. the compact plate: one panel, distances true to the page
def classical_mds(D):
    """Classical MDS (PCoA) of a distance matrix -> (Y[n,2], variance fraction in 2-D).

    Fitted to the EIGHT, not to the population.  The question the figure answers is how far
    apart these eight are, so the projection should be chosen to preserve their distances rather
    than to describe the population's axes: on this set MDS holds plotted-against-true distance
    at r = 0.98 (rms 6.6 mm) where the population-PCA plane manages r = 0.90 (rms 14.7 mm).
    """
    n = len(D)
    J = np.eye(n) - np.ones((n, n)) / n
    w, V = np.linalg.eigh(-0.5 * J @ (D ** 2) @ J)
    o = np.argsort(-w); w, V = w[o], V[:, o]
    pos = np.clip(w, 0, None)
    return V[:, :2] * np.sqrt(pos[:2]), float(pos[:2].sum() / pos.sum())


def fig_hand_map(plt, X, stage, labs, V, cells, trajs, cols, rep):
    """The minimal version: the eight placed so page distance is millimetres of mount travel.

    Markers stay small on purpose.  The closest pair is 14.4 mm apart, so anything large enough
    to carry a three-way split -- a miniature tripod, a wedge marker -- is wider than the gaps
    the embedding exists to show, and glyphs from different hands merge.  A distance-true map
    and a per-finger glyph cannot both fit; `fig_hand_cards` gives up the distances to keep the
    glyph, and this one gives up the glyph to keep the distances.
    """
    iu = np.triu_indices(8, 1)
    D = np.linalg.norm(V[:, None] - V[None], axis=-1)
    Y, frac = classical_mds(D)
    d2 = np.linalg.norm(Y[:, None] - Y[None], axis=-1)
    r = float(np.corrcoef(d2[iu], D[iu])[0, 1])
    rms = float(np.sqrt(((d2[iu] - D[iu]) ** 2).mean()))

    mot = {d: finger_motion(trajs[d], cols) for d in labs}
    tot = np.array([sum(mot[d].values()) for d in labs])
    dep = np.array([cells[d]["grip_depth_mm"] for d in labs])

    P = X[stage >= 1][:1500]
    pop_med = float(np.median(np.linalg.norm(P[:, None] - P[None], axis=-1)
                              [np.triu_indices(len(P), 1)]))

    fig, ax = plt.subplots(figsize=(3.46, 2.95))
    ax.set_xlim(Y[:, 0].min() - 20, Y[:, 0].max() + 20)
    ax.set_ylim(Y[:, 1].min() - 22, Y[:, 1].max() + 20)
    ax.set_aspect("equal")
    # a bar of the population's typical separation, drawn to the same scale as the axis
    y0 = ax.get_ylim()[0] + 0.09 * np.ptp(ax.get_ylim())
    x0 = ax.get_xlim()[0] + 0.06 * np.ptp(ax.get_xlim())
    ax.plot([x0, x0 + pop_med], [y0, y0], lw=1.6, color=GRID, solid_capstyle="butt", zorder=1)
    ax.annotate(f"{pop_med:.0f} mm: two random hands", (x0, y0 + 3.5), fontsize=5.6,
                color=MUTED, ha="left", va="bottom", zorder=2)

    sc = ax.scatter(Y[:, 0], Y[:, 1], s=14 + 116 * (tot - tot.min()) / np.ptp(tot),
                    c=dep, cmap="Blues", vmin=44, vmax=70, lw=0.6, ec=INK, zorder=4)
    label_points(ax, Y, labs, size=6.4)
    cb = fig.colorbar(sc, ax=ax, fraction=0.043, pad=0.03)
    cb.set_label("grip depth (mm)", fontsize=6.0); cb.ax.tick_params(labelsize=5.8, length=2)
    cb.outline.set_visible(False)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], ls="", marker="o", mfc=GRID, mec=INK, mew=0.6,
                ms=np.sqrt(14 + 116 * f) / 1.6, label=f"{v:.0f}$\\degree$")
         for f, v in [(0.0, tot.min()), (1.0, tot.max())]]
    ax.legend(handles=h, loc="upper left", frameon=False, fontsize=5.8, handlelength=1.0,
              labelspacing=0.55, borderpad=0.1, title="commanded travel",
              title_fontsize=5.8)
    ax.set_xlabel("millimetres of mount travel")
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.grid(True, axis="x", lw=0.4, zorder=0); ax.set_axisbelow(True)
    ax.set_title(f"page distance = mount distance ($r$ = {r:.2f})", color=INK2, pad=4,
                 fontsize=7.2)
    fig.tight_layout(pad=0.35)
    fig.savefig(f"{OUT}/fig_hand_map.pdf"); fig.savefig(f"{OUT}/fig_hand_map.png", dpi=220)
    plt.close(fig)

    rep.append("Compact map (classical MDS on the eight)")
    rep.append(f"  2-D captures {frac*100:.1f} % of their distance structure; "
               f"plotted vs true distance r {r:.4f}, rms {rms:.2f} mm")
    mu, W, ev = pca(X)
    Pp = (V - mu) @ W[:2].T
    dp = np.linalg.norm(Pp[:, None] - Pp[None], axis=-1)
    rep.append(f"  for comparison, the population-PCA plane: r "
               f"{float(np.corrcoef(dp[iu], D[iu])[0, 1]):.4f}, rms "
               f"{float(np.sqrt(((dp[iu]-D[iu])**2).mean())):.2f} mm")
    rep.append(f"  population median pairwise separation {pop_med:.1f} mm")


# ------------------------------ 8. the eight inside the population, in interpretable coordinates
def load_geometry(tags):
    """Interpretable reparameterisation of the six mounts, straight out of the manifest.

    x_sep is the thumb-to-pair span and y_sep the pair opening; both are linear in the mount
    coordinates, so nothing is projected or fitted and the axes are millimetres of real hardware.
    """
    geo = {r["design"]: r["geometry"]
           for r in json.load(open(f"{SOBOL}/grasp_screen_manifest.json"))["designs"]}
    return (np.array([geo[t]["x_sep_mm"] for t in tags], float),
            np.array([geo[t]["y_sep_mm"] for t in tags], float))


def fig_population_plane(plt, tags, X, stage, labs, rep):
    """Where the eight sit in the sampled population, without a projection.

    No PCA: a 2-D view of a six-dimensional near-uniform box is lossy whichever plane is chosen,
    and the two planes worth choosing here are physical rather than fitted.  The funnel acts on
    exactly these two coordinates, so the marginals carry the story that a projection blurs.
    """
    xs, ys = load_geometry(tags)
    idx = {t: i for i, t in enumerate(tags)}
    e = np.array([[xs[idx[DESIGNS[d][0]]], ys[idx[DESIGNS[d][0]]]] for d in labs])
    reach, kept = stage >= 1, stage >= 2
    mx, my = float(np.median(xs[reach])), float(np.median(ys[reach]))
    frac = {"reachable": ((xs[reach] < mx) & (ys[reach] < my)).mean(),
            "retained": ((xs[kept] < mx) & (ys[kept] < my)).mean(),
            "confirmed": ((xs[stage >= 3] < mx) & (ys[stage >= 3] < my)).mean(),
            "the eight": ((e[:, 0] < mx) & (e[:, 1] < my)).mean()}

    fig = plt.figure(figsize=(3.46, 3.42))
    gs = fig.add_gridspec(2, 2, width_ratios=[4.4, 1], height_ratios=[1, 4.4],
                          wspace=0.06, hspace=0.06)
    A = fig.add_subplot(gs[1, 0]); Tx = fig.add_subplot(gs[0, 0], sharex=A)
    Ry = fig.add_subplot(gs[1, 1], sharey=A)

    A.axvline(mx, color=MUTED, lw=0.6, ls=(0, (3, 2)), zorder=2)
    A.axhline(my, color=MUTED, lw=0.6, ls=(0, (3, 2)), zorder=2)
    A.scatter(xs[reach], ys[reach], s=1.3, c=GRID, lw=0, rasterized=True, zorder=1,
              label=f"sampled, in the rails ({reach.sum():,})")
    A.scatter(xs[kept], ys[kept], s=2.6, c=SEQ[3], lw=0, rasterized=True, zorder=3,
              label=f"passes retention ({kept.sum()})")
    A.scatter(*e.T, s=22, facecolor="none", edgecolor=INK, lw=1.0, zorder=5,
              label="deployed (D1--D8)")
    A.set_xlabel("thumb-to-pair span (mm)"); A.set_ylabel("pair opening (mm)")
    A.grid(True, lw=0.4, zorder=0); A.set_axisbelow(True)
    lg = A.legend(loc="upper right", fontsize=5.6, handlelength=1.0, borderpad=0.25,
                  labelspacing=0.25, framealpha=0.92, facecolor=SURF, edgecolor="none")
    lg.set_zorder(8)
    A.annotate(f"in the compact quadrant:\n{100*frac['reachable']:.0f} % sampled  |  "
               f"{100*frac['retained']:.0f} % retained  |  "
               f"{int(round(8*frac['the eight']))} of 8 deployed",
               (0.03, 0.03), xycoords="axes fraction", fontsize=5.7, color=INK2,
               ha="left", va="bottom", zorder=8,
               bbox=dict(fc=SURF, ec="none", alpha=0.85, pad=1.2))
    label_points(A, e, labs, size=6.0)

    for AX, v8, vr, vk, horiz in ((Tx, e[:, 0], xs[reach], xs[kept], True),
                                  (Ry, e[:, 1], ys[reach], ys[kept], False)):
        lo, hi = (A.get_xlim() if horiz else A.get_ylim())
        b = np.linspace(lo, hi, 46)
        o = "vertical" if horiz else "horizontal"
        AX.hist(vr, bins=b, density=True, orientation=o, color=GRID, lw=0, zorder=1)
        AX.hist(vk, bins=b, density=True, orientation=o, histtype="step", lw=0.9,
                color=SEQ[3], zorder=2)
        for q in v8:                                  # the eight as rug ticks
            if horiz:
                AX.plot([q, q], [0, AX.get_ylim()[1] * 0.42], lw=0.7, color=INK, zorder=3)
            else:
                AX.plot([0, AX.get_xlim()[1] * 0.42], [q, q], lw=0.7, color=INK, zorder=3)
        # NOT set_xticks([]) -- these axes share with A, so that would strip A's numbers too
        AX.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False,
                       labelsize=0, length=0)
        for sp in AX.spines.values():
            sp.set_visible(False)
    fig.tight_layout(pad=0.35)
    fig.savefig(f"{OUT}/fig_population_plane.pdf")
    fig.savefig(f"{OUT}/fig_population_plane.png", dpi=220)
    plt.close(fig)

    rep.append("The eight inside the population (interpretable plane)")
    rep.append(f"  reachable medians: x_sep {mx:.1f} mm, y_sep {my:.1f} mm")
    for nm, m in (("reachable", reach), ("retained", kept), ("confirmed", stage >= 3)):
        rep.append(f"  {nm:11s} x_sep {xs[m].mean():6.1f} +- {xs[m].std():5.1f}   "
                   f"y_sep {ys[m].mean():6.1f} +- {ys[m].std():5.1f}   n={int(m.sum())}")
    rep.append(f"  the eight   x_sep {e[:,0].mean():6.1f} +- {e[:,0].std():5.1f}   "
               f"y_sep {e[:,1].mean():6.1f} +- {e[:,1].std():5.1f}   n=8")
    rep.append("  share in the compact quadrant (both below the reachable median): " +
               ", ".join(f"{k} {100*v:.1f} %" for k, v in frac.items()))
    for nm, v, v8 in (("x_sep", xs[reach], e[:, 0]), ("y_sep", ys[reach], e[:, 1])):
        pc = sorted(int(round(100 * (v < q).mean())) for q in v8)
        rep.append(f"  {nm} percentile of each of the eight: {pc}")
    # which coordinates the funnel actually constrains: sd of the retained set over sd of the
    # reachable one.  A ratio near 1 means the screen is indifferent to that coordinate.
    geo = {r["design"]: r["geometry"]
           for r in json.load(open(f"{SOBOL}/grasp_screen_manifest.json"))["designs"]}
    GK = ["x_sep_mm", "y_sep_mm", "thumb_y_mm", "pair_x_offset_mm", "pair_y_mid_mm"]
    G = np.array([[geo[t][k] for k in GK] for t in tags], float)
    rep.append("  sd(retained)/sd(reachable) per interpretable coordinate:")
    for j, k in enumerate(GK):
        rep.append(f"    {k:18s} {G[reach, j].std():5.1f} -> {G[kept, j].std():5.1f} mm   "
                   f"ratio {G[kept, j].std()/G[reach, j].std():.2f}")


def main():
    os.makedirs(OUT, exist_ok=True)
    tags, X, src, stage = load_population()
    labs, V, cells, trajs, cols = load_eight(tags, X)
    plt = style()
    rep = ["real_v1 design-space diversity -- numbers behind paper/figures_diversity",
           f"population 8,198 sampled / {(stage>=1).sum()} reachable / "
           f"{(stage>=2).sum()} retained / {(stage>=3).sum()} confirmed",
           f"of the eight, {8-len(NOT_SAMPLED)} are Sobol members; "
           f"{', '.join(sorted(NOT_SAMPLED))} are not (anchor / manual)", ""]
    fig_pca(plt, X, stage, V, labs, rep); rep.append("")
    fig_mount_plane(plt, X, stage, V, labs)
    Dm = fig_distance(plt, X, stage, V, labs, rep); rep.append("")
    fig_behaviour(plt, labs, cells, trajs, cols, Dm, rep); rep.append("")
    fig_grasp_space(plt, labs, cells, rep); rep.append("")
    fig_hand_cards(plt, X, stage, labs, V, cells, trajs, cols, rep); rep.append("")
    fig_hand_map(plt, X, stage, labs, V, cells, trajs, cols, rep); rep.append("")
    fig_population_plane(plt, tags, X, stage, labs, rep)
    open(f"{OUT}/numbers.txt", "w").write("\n".join(rep) + "\n")
    print("\n".join(rep))


if __name__ == "__main__":
    main()
