"""Ranking analysis for the real_v1 open-loop transfer study.

Two questions, kept separate:
  METRIC CONSISTENCY   -- do different ways of scoring a hand agree on the ORDER of the hands?
  SIM2REAL CONSISTENCY -- does any simulated quantity reproduce a bench order?

Trial scoring is imported from real_v1_transfer_figures.py, and the operator's verdict decides
whether a trial held. Nothing else does.

WHY THERE IS NO FLOOR-CONTACT EXCLUSION. An earlier version of this file dropped any hold whose
tag centre came within 8 mm of the bench floor, on the reasoning that a shaft resting on the
table is not a shaft the hand is holding. That test does not measure what it claims to. The vane
carries the tag 77 mm along the shaft on the END THAT SWINGS DOWN, so the tag's minimum height is
a rigid-body consequence of the tilt: over the 47 measured holds it falls 0.79 mm per degree of
turn, r = -0.89, and the per-design residuals are +-10 mm around that line with `sv1_u0060` no
further off it than three other hands. The rule therefore excluded hands in proportion to how far
they turned -- which is the quantity being ranked -- and it is gone. `tag_height_confound`
recomputes the relation so the claim stays checkable, and `min_tag_z` is still reported.
"""
import argparse, csv, glob, json, os, re, sys
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from real_v1_transfer_figures import bench, sim, PLAN          # noqa: E402

OUT = "paper/figures"
# no FLOOR_MM: see the module docstring. Tag height is turn angle wearing a disguise.
EJECT_SLIP_MM = 20.0    # a drop that travels this far never had the shaft in the first place
SHORT = {"sv1_w6689": "w6689", "sv1_w2360": "w2360", "sv1_u1364": "u1364", "g12": "g12",
         "sv1_u0060": "u0060", "sv1_u0308": "u0308", "rv05_manual": "rv05", "sv1_w0099": "w0099"}
#: Publication names. Every hand is D<k> where k is its rank in simulation, so a reader can see
#: the predicted order on the axis and the measured order in the data. The mapping from D<k> to
#: the nine morphology parameters belongs in the appendix, not on an axis tick.
DESIGN_ID = {"sv1_w6689": "D1", "sv1_w2360": "D2", "sv1_u1364": "D3", "g12": "D4",
             "sv1_u0060": "D5", "sv1_u0308": "D6", "rv05_manual": "D7", "sv1_w0099": "D8"}


def min_tag_z(per_design=10):
    """Lowest tag height reached in each trace. Reported, never used as a gate."""
    out = {}
    for p in sorted(glob.glob("logs/tracker/*_SUMMARY.json"), key=os.path.getmtime):
        d = json.load(open(p))
        if d.get("axial_mm") != 77.0:
            continue
        m = re.match(r"\d{8}-\d{6}-(.+)-([0-9a-f]{6})$",
                     os.path.basename(p).replace("_track_SUMMARY.json", ""))
        rows = [r for r in csv.DictReader(open(p.replace("_SUMMARY.json", ".csv"))) if r.get("cos")]
        if rows:
            out[m.group(2)] = min(float(r["tag_z_bench_mm"]) for r in rows)
    return out


def tag_height_confound(per_design=10):
    """How much of a hold's minimum tag height is just its turn angle.

    The vane puts the tag 77 mm along the shaft on the end that swings down, so the tag sinks
    as the shaft stands up. This is the measurement that retired the floor-contact exclusion:
    if tag height is a function of the turn, thresholding it penalises the hands that turn."""
    B, F = bench(per_design), min_tag_z()
    x, y, who = [], [], []
    for dsg, g in B.items():
        for t in g:
            if t["outcome"] != "HELD" or t["deg"] is None or F.get(t["tag"]) is None:
                continue
            x.append(t["deg"]); y.append(F[t["tag"]]); who.append(DESIGN_ID[dsg])
    x, y = np.array(x), np.array(y)
    r = stats.pearsonr(x, y)
    sl, ic = np.polyfit(x, y, 1)
    res = y - (ic + sl * x)
    return dict(n=len(x), r=float(r.statistic), p=float(r.pvalue), slope=float(sl),
                intercept=float(ic), resid_sd=float(res.std(ddof=1)),
                per_design={d: float(np.mean([res[i] for i in range(len(who)) if who[i] == d]))
                            for d in sorted(set(who))})


def build(per_design=10):
    B, S, F = bench(per_design), sim(), min_tag_z()
    rows = []

    def mode(drops, holds):
        """Three failure modes, and they are not the same defect.

        EJECTION:  the shaft leaves the grip before any turn happens -- net turn near zero and
                   20-55 mm of travel. A grasp that never closed.
        OVERSHOOT: the turn completes and keeps going, then the grip lets go. The drops turn
                   FURTHER than the same hand's holds (w0099: 65 deg against 44) and travel
                   little. This is the one that argues for closing the loop.
        STALL:     the drops travel like a hold but stop SHORT of one (u1364: 25 deg against
                   42). Neither of the above, and it was invisible while slip was the only
                   test -- a stalled turn slips no more than a good one."""
        if not drops:
            return "--"
        sl = np.mean([t["slip"] for t in drops if t["slip"] is not None])
        if sl >= EJECT_SLIP_MM:
            return "eject"
        dd = [t["deg"] for t in drops if t["deg"] is not None]
        hd = [t["deg"] for t in holds if t["deg"] is not None]
        if dd and hd and np.mean(dd) < np.mean(hd):
            return "stall"
        return "overshoot"

    for dsg, g in B.items():
        # `extra` trials are individually admitted holds from another session (see
        # EXTRA_RUNS). They add ALIGNMENT samples and nothing else. They must never enter a
        # hold RATE: they were picked because the operator scored them and they succeeded,
        # so counting them into a denominator selects on the outcome being measured. Every
        # rate, turn, slip and failure mode below is computed on the session alone.
        core = [t for t in g if not t.get("extra")]
        held = heldall = [t for t in core if t["outcome"] == "HELD"]
        drops = [t for t in core if t["outcome"] == "DROPPED"]
        # ...if the tag survived to the hold window.
        pool = held + [t for t in g if t.get("extra") and t["outcome"] == "HELD"]
        h = np.array([t["cos_hold"] for t in pool if t["cos_hold"] is not None])
        # EVERY hand carries an alignment number, so that every hand carries a rank and the
        # sim/bench comparison is over the same eight. One needs a substitute, marked with the
        # symbol in `qual` wherever it is printed:
        #   *  the tag dies before the hold window on every hold, so the run's PEAK alignment
        #      stands in. g12 holds 9 of 10 and the cylinder crowds its vane out of decode
        #      range as the shaft comes up, which is an instrument failure, not a grip one.
        qual = ""
        if not len(h):
            h = np.array([t["cos_peak"] for t in pool if t["cos_peak"] is not None])
            qual = "*"
        mins = [F.get(t["tag"]) for t in heldall if F.get(t["tag"]) is not None]
        rows.append(dict(design=dsg, short=SHORT[dsg], n=len(core), n_held=len(held),
                         n_extra=len(g) - len(core),
                         n_drop=sum(t["outcome"] == "DROPPED" for t in core),
                         n_clean=len(held), n_cos=int(len(h)), qual=qual,
                         did=DESIGN_ID[dsg],
                         n_unres=sum(t["outcome"] == "UNRESOLVED" for t in core),
                         n_heldall=len(heldall), mode=mode(drops, heldall),
                         hold_rate_op=len(heldall) / len(core),
                         held_slip=float(np.mean([t["slip"] for t in heldall
                                                  if t["slip"] is not None])) if heldall else None,
                         drop_slip=float(np.mean([t["slip"] for t in drops
                                                  if t["slip"] is not None])) if drops else None,
                         held_deg=float(np.mean([t["deg"] for t in heldall
                                                 if t["deg"] is not None])) if heldall else None,
                         drop_deg=float(np.mean([t["deg"] for t in drops
                                                 if t["deg"] is not None])) if drops else None,
                         min_tag_z=float(np.median(mins)) if mins else None,
                         bench_cos=float(h.mean()) if len(h) else None,
                         bench_med=float(np.median(h)) if len(h) else None,
                         bench_sd=float(h.std(ddof=1)) if len(h) > 1 else None,
                         bench_peak=float(np.mean([t["cos_peak"] for t in core
                                                   if t["cos_peak"] is not None])),
                         bench_deg=float(np.mean([t["deg"] for t in core
                                                  if t["deg"] is not None])),
                         bench_slip=float(np.mean([t["slip"] for t in core
                                                   if t["slip"] is not None])),
                         **{f"sim_{k}": S.get(dsg, {}).get(k) for k in
                            ("final_cos", "peak_cos", "ok_rate", "contacts", "force_N")}))
    return sorted(rows, key=lambda r: -(r["sim_final_cos"] or 0))


def rank_table(rows, keys):
    """spearman between every pair of metrics, over designs where both are defined."""
    n = len(keys)
    M, N = np.full((n, n), np.nan), np.zeros((n, n), int)
    for i, a in enumerate(keys):
        for j, b in enumerate(keys):
            pairs = [(r[a], r[b]) for r in rows if r.get(a) is not None and r.get(b) is not None]
            if len(pairs) >= 3:
                M[i, j] = stats.spearmanr([p[0] for p in pairs], [p[1] for p in pairs]).statistic
                N[i, j] = len(pairs)
    return M, N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-design", type=int, default=10)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    allrows = build(a.per_design)
    rows = allrows      # every hand carries a number, so every hand carries a rank

    print(f"{'design':<8}{'n':>3}{'held':>5}{'drop':>5}{'unres':>6}{'meas':>6}{'medTagZ':>9}"
          f"{'bench':>8}{'sd':>7}{'peak':>7}{'deg':>6}{'simcos':>8}{'simN':>7}")
    for r in allrows:
        bc = f"{r['bench_cos']:+.3f}{r['qual']}" if r["bench_cos"] is not None else "     -"
        sd = f"{r['bench_sd']:.3f}" if r["bench_sd"] is not None else "    -"
        flag = "  * peak stands in" if "*" in r["qual"] else ""
        print(f"{r['short']:<8}{r['n']:>3}{r['n_held']:>5}{r['n_drop']:>5}{r['n_unres']:>6}"
              f"{r['n_cos']:>3}/{r['n_heldall']:<2}"
              f"{(r['min_tag_z'] if r['min_tag_z'] is not None else float('nan')):>9.1f}{bc:>8}{sd:>7}{r['bench_peak']:>+7.3f}"
              f"{r['bench_deg']:>6.1f}{r['sim_final_cos']:>8.3f}{r['sim_force_N']:>7.2f}{flag}")
    C = tag_height_confound(a.per_design)
    print(f"\ntag height vs turn over {C['n']} holds: r = {C['r']:+.3f} (p = {C['p']:.1e}), "
          f"{C['slope']:+.2f} mm/deg, residual sd {C['resid_sd']:.1f} mm -- a floor threshold on "
          f"tag height is a threshold on turn angle, which is why there is not one")

    bench_keys = ["bench_cos", "bench_med", "bench_peak", "bench_deg", "hold_rate_op",
                  "held_slip"]
    sim_keys = ["sim_final_cos", "sim_peak_cos", "sim_ok_rate", "sim_contacts", "sim_force_N"]
    Mb, Nb = rank_table(rows, bench_keys)
    Ms, Ns = rank_table(rows, sim_keys + bench_keys)
    json.dump(dict(rows=allrows, used=[r["short"] for r in rows],
                   bench_keys=bench_keys, sim_keys=sim_keys,
                   tag_height_confound=C,
                   metric_consistency=np.nan_to_num(Mb, nan=0).tolist(),
                   cross=np.nan_to_num(Ms, nan=0).tolist()),
              open(f"{OUT}/ranking.json", "w"), indent=1)
    return allrows, rows, bench_keys, sim_keys, Mb, Nb, Ms, Ns




# ------------------------------------------------------------------ paper figures
BL, RD, OR, MU, GR = "#1f4e79", "#c0392b", "#e08a00", "#8a8a8a", "#4a7c59"


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "axes.facecolor": "white",
                         "pdf.fonttype": 42, "axes.labelsize": 8.5, "legend.fontsize": 7})
    return plt


def tick(r):
    """D<k>, carrying its evidence caveat as a superscript the caption can decode."""
    return r["did"] + (r"$^{*}$" if r["qual"] else "")


def all_hold_cos(g):
    """Mean alignment over EVERY hold the operator called, table-supported ones included."""
    v = [t["cos_hold"] for t in g if t["outcome"] == "HELD" and t["cos_hold"] is not None]
    if v:
        return float(np.mean(v)), ""
    v = [t["cos_peak"] for t in g if t["outcome"] == "HELD" and t["cos_peak"] is not None]
    return (float(np.mean(v)), "*") if v else (None, "")


#: The paper wants the same figures with the statistics moved out of the titles and panels
#: named (a)/(b) for the caption to refer to. Rather than keep a second, drifting copy of the
#: plotting code, `figures(clean=True)` re-renders everything into `*_clean.pdf` from the same
#: data. Whatever the analysis says, both variants say.
def figures(clean=False):
    plt = _style()
    suf = "_clean" if clean else ""

    def title(ax, letter, plain, rich=""):
        """Panel title. The clean variant is captioned prose, so it carries a letter and no
        numbers; the working variant states the statistic where it can be seen."""
        ax.set_title(f"({letter}) {plain}" if clean else (plain + rich), loc="left",
                     fontsize=8.5 if clean else 8)

    allrows, rows, bk, sk, Mb, Nb, Ms, Ns = main()
    for r in allrows:
        r["hold_rate_op"] = r["n_heldall"] / r["n"]
    B, F = bench(10), min_tag_z()
    lbl = {"bench_cos": "Measured alignment", "bench_med": "Median alignment",
           "bench_peak": "Peak alignment", "bench_deg": "Net turn",
           "hold_rate_op": "Hold rate", "held_slip": "Slip",
           "sim_final_cos": "Simulated alignment", "sim_peak_cos": "Simulated peak",
           "sim_ok_rate": "Simulated retention", "sim_contacts": "Simulated contacts",
           "sim_force_N": "Simulated force"}

    # ---- fig 1: per-design alignment, every hand present
    f, ax = plt.subplots(figsize=(6.6, 3.5))
    rng = np.random.default_rng(4)
    for i, r in enumerate(allrows):
        for t in B[r["design"]]:
            # exactly the trials that enter this hand's mean, so the spread a reader sees is
            # the spread the number was taken over. A hold whose tag died before the window
            # contributes nothing unless the whole hand fell back to peak (`qual`), and the
            # table's `meas` column is where those holds are counted.
            if t["outcome"] != "HELD":
                continue
            pk = bool(r["qual"])
            y = t["cos_peak"] if pk else t["cos_hold"]
            if y is None:
                continue
            x = i + rng.uniform(-.16, .16)
            if pk:
                ax.scatter(x, y, s=22, marker="^", facecolors="none", edgecolors=BL, lw=.8,
                           zorder=3)
            else:
                ax.scatter(x, y, s=17, c=BL, lw=0, zorder=4)
        mean, = ax.plot([i - .32, i + .32], [r["bench_cos"]] * 2, c=BL, lw=2.2, zorder=5)
        pred = ax.scatter([i], [r["sim_final_cos"]], s=110, marker="_", c=OR, lw=2.2, zorder=6)
        # spelled out, not an arrow: a glyph next to a design whose bench mean sits ABOVE its
        # simulated prediction reads as "this hand fell short", which is the opposite of what
        # it counts. It counts trials in which the object hit the table.
        if r["n_drop"]:
            ax.text(i, 0.03, f"{r['n_drop']} dropped", ha="center", fontsize=6.5, color=RD)
    ax.set_xticks(range(len(allrows)))
    ax.set_xticklabels([tick(r) for r in allrows])
    ax.set_ylim(0, 1.04); ax.set_xlim(-.6, len(allrows) - .4)
    ax.set_xlabel("Design (ordered by simulated rank)")
    ax.set_ylabel(r"Shaft alignment, $\cos(\hat{s},\hat{z})$")
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], ls="none", marker="_", ms=9, mew=2.2, color=OR,
               label="Simulated prediction"),
        Line2D([], [], color=BL, lw=2.2, label="Bench mean"),
        Line2D([], [], ls="none", marker="o", ms=4, color=BL, label="Hold, measured"),
        Line2D([], [], ls="none", marker="^", ms=5, mfc="none", mec=BL,
               label="Hold, peak (tag lost before the hold window)")],
        frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(.5, -0.15))
    f.tight_layout(); f.savefig(f"{OUT}/fig_transfer_designs{suf}.pdf")
    f.savefig(f"{OUT}/fig_transfer_designs{suf}.png", dpi=200)

    # ---- fig 2: the simulator against each of the two bench axes, all eight hands
    f, ax = plt.subplots(1, 2, figsize=(6.6, 3.6))
    xs = np.array([r["sim_final_cos"] for r in allrows])
    from matplotlib.lines import Line2D
    for a, ys, ylab, ttl, diag in [
            (ax[0], [r["bench_cos"] for r in allrows], "Measured alignment",
             "Does the simulator rank the turn?", True),
            (ax[1], [r["hold_rate_op"] for r in allrows], "Hold rate",
             "Does the simulator rank the grip?", False)]:
        _ = ttl
        ys = np.array(ys)
        for x, y, r in zip(xs, ys, allrows):
            sub = diag and bool(r["qual"])
            a.scatter([x], [y], s=42, zorder=4, c="none" if sub else BL, edgecolors=BL, lw=1.1,
                      marker="^" if sub else "o")
            a.annotate(tick(r) if diag else r["did"], (x, y), textcoords="offset points",
                       xytext=(6, 3), fontsize=7)
        # The trend is Theil-Sen, the median of the pairwise slopes, NOT least squares. The
        # reported statistic is a rank correlation, and with eight hands one of them can put
        # the two in opposite directions: D3 ejects the shaft, so it sits far below the others
        # in value while holding its rank, and least squares reads that as a slope of -0.15
        # against a rho of +0.33. A median slope cannot be dragged by one point, so the line a
        # reader sees agrees in sign with the number in the title.
        gx = np.linspace(xs.min() - .02, xs.max() + .02, 2)
        sl, ic = stats.theilslopes(ys, xs)[:2]
        a.plot(gx, ic + sl * gx, c=RD, lw=1.4, zorder=2)
        if diag:
            a.plot([.45, .95], [.45, .95], ls=(0, (4, 3)), c=MU, lw=.9, zorder=1)
        rho = stats.spearmanr(xs, ys)
        extra = ""
        if diag:
            q = [r for r in allrows if not r["qual"]]
            rr = stats.spearmanr([r["sim_final_cos"] for r in q], [r["bench_cos"] for r in q])
            extra = f"\n{rr.statistic:+.2f} with the substituted hand removed (n = {len(q)})"
        title(a, "a" if diag else "b", "Measured alignment" if diag else "Hold rate",
              f"\n" + r"$\rho_s$ = " + f"{rho.statistic:+.2f} (n = 8, p = {rho.pvalue:.2f})"
              f"{extra}")
        a.set_xlabel("Simulated alignment"); a.set_ylabel(ylab)

    ax[0].legend(handles=[
        Line2D([], [], c=RD, lw=1.4, label="Robust trend"),
        Line2D([], [], ls=(0, (4, 3)), c=MU, lw=.9, label="Equality"),
        Line2D([], [], ls="none", marker="^", ms=6, mfc="none", mec=BL,
               label=r"$*$ peak stands in")],
        frameon=False, loc="lower left", fontsize=6.5, handlelength=1.6,
        borderpad=.1, labelspacing=.35)
    ax[1].legend(handles=[Line2D([], [], c=RD, lw=1.4, label="Robust trend")],
                 frameon=False, loc="lower right", fontsize=6.5, handlelength=1.6)
    ax[0].set_ylim(0.08, 1.02); ax[1].set_ylim(-0.05, 1.15)
    f.tight_layout(); f.savefig(f"{OUT}/fig_transfer_sim2real{suf}.pdf")
    f.savefig(f"{OUT}/fig_transfer_sim2real{suf}.png", dpi=200)

    # ---- fig 3: rank flow + what predicts what (the readable version)
    f, ax = plt.subplots(1, 2, figsize=(6.6, 3.3), gridspec_kw={"width_ratios": [1, 1.25]})
    order = sorted(allrows, key=lambda r: -r["bench_cos"])
    mrank = {r["did"]: k + 1 for k, r in enumerate(order)}
    a = ax[0]
    for i, r in enumerate(allrows):
        y0, y1 = i + 1, mrank[r["did"]]
        a.plot([0, 1], [y0, y1], c=RD if abs(y0 - y1) >= 3 else BL,
               lw=1.6 if abs(y0 - y1) >= 3 else 1.0, alpha=.9, zorder=2)
        a.text(-.06, y0, tick(r), ha="right", va="center", fontsize=7.5)
        a.text(1.06, y1, tick(r), ha="left", va="center", fontsize=7.5)
    a.set_xlim(-.35, 1.35); a.set_ylim(len(allrows) + .6, .4)
    a.set_xticks([0, 1]); a.set_xticklabels(["Simulated", "Measured"])
    a.set_yticks(range(1, len(allrows) + 1)); a.set_ylabel("Rank")
    a.spines["bottom"].set_visible(False); a.tick_params(axis="x", length=0)
    a.plot([], [], c=RD, lw=1.6, label="Moves 3 places or more")
    a.legend(frameon=False, loc="lower center", bbox_to_anchor=(.5, -.28))
    title(a, "a", "Simulated and measured ranks")

    a = ax[1]
    preds = ["sim_final_cos", "sim_peak_cos", "sim_force_N", "sim_contacts", "sim_ok_rate"]
    tgts = [("bench_cos", "Measured alignment", BL), ("hold_rate_op", "Hold rate", GR)]
    h = 0.36
    for j, (t, tl, c) in enumerate(tgts):
        v = [stats.spearmanr([r[p] for r in allrows], [r[t] for r in allrows]).statistic
             for p in preds]
        a.barh([i + (j - .5) * h for i in range(len(preds))], v, height=h, color=c, label=tl)
    a.axvline(0, c="#333", lw=.8)
    a.set_yticks(range(len(preds))); a.set_yticklabels([lbl[p] for p in preds])
    a.invert_yaxis(); a.set_xlim(-.75, .75)
    a.set_xlabel(r"Spearman $\rho$ against the bench  (n = 8)")
    a.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, -.02))
    title(a, "b", "Simulation-to-bench rank correlation")
    f.tight_layout(); f.savefig(f"{OUT}/fig_transfer_rankflow{suf}.pdf")
    f.savefig(f"{OUT}/fig_transfer_rankflow{suf}.png", dpi=200)

    # ---- fig 3b (appendix): the full matrices
    bk2 = ["bench_peak", "bench_deg", "hold_rate_op", "held_slip", "bench_cos"]
    Mb2, _ = rank_table(allrows, bk2)
    Mx, _ = rank_table(allrows, sk + bk2)
    f, ax = plt.subplots(1, 2, figsize=(7.0, 3.4),
                         gridspec_kw={"width_ratios": [len(bk2), len(sk)]})
    for a, M, rl, cl, ttl in [
            (ax[0], Mb2, bk2, bk2, "(a) Bench metrics against each other" if clean
             else "Bench metrics against each other"),
            (ax[1], Mx[len(sk):, :len(sk)], bk2, sk, "(b) Simulated against bench" if clean
             else "Simulated against bench")]:
        im = a.imshow(M, vmin=-1, vmax=1, cmap="RdBu_r")
        a.set_xticks(range(len(cl))); a.set_xticklabels([lbl[k] for k in cl], rotation=40,
                                                        ha="right")
        a.set_yticks(range(len(rl)))
        a.set_yticklabels([lbl[k] for k in rl] if a is ax[0] else [])
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if not np.isnan(M[i, j]):
                    a.text(j, i, f"{M[i,j]:+.2f}", ha="center", va="center", fontsize=6.5,
                           color="white" if abs(M[i, j]) > .6 else "#222")
        a.set_title(ttl, loc="left", fontsize=8)
    f.colorbar(im, ax=ax, shrink=.8, label=r"Spearman $\rho$")
    f.savefig(f"{OUT}/fig_transfer_ranking{suf}.pdf", bbox_inches="tight")
    f.savefig(f"{OUT}/fig_transfer_ranking{suf}.png", dpi=200, bbox_inches="tight")

    # ---- fig 4: the three failure modes, in one plane
    BAND = (20.0, 60.0)     # where 44 of the 48 holds sit, and 4 of the 28 drops
    f, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.axvspan(*BAND, color=BL, alpha=.05, zorder=0)
    for g in B.values():
        for t in g:
            if t["deg"] is None or t["slip"] is None:
                continue
            if t["outcome"] == "HELD":
                ax.scatter(t["deg"], t["slip"], s=20, c=BL, zorder=3, lw=0)
            elif t["outcome"] == "DROPPED":
                ax.scatter(t["deg"], t["slip"], s=26, marker="x", c=RD, zorder=4, lw=1.1)
    ax.axhline(EJECT_SLIP_MM, c=MU, lw=.7, ls="--")
    for v in BAND:
        ax.axvline(v, c=MU, lw=.7, ls=":")
    # The three regions a drop can be in, named where the reader will look for them.
    for x, y, txt, ha in [(-4, 45, "Ejected", "left"), (-4, 15, "Stalled", "left"),
                          (77, 15, "Overshot", "right")]:
        ax.text(x, y, txt, fontsize=7.5, color=MU, ha=ha, style="italic")
    ax.set_xlabel("Net turn (deg)"); ax.set_ylabel("Slip (mm)")
    ax.scatter([], [], s=20, c=BL, label="Held")
    ax.scatter([], [], s=26, marker="x", c=RD, lw=1.1, label="Dropped")
    ax.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(.55, 1.02))
    title(ax, "a", "Trial outcomes" if clean else
          "Holds live in a band of turn;\ndrops leave it three ways")
    f.tight_layout(); f.savefig(f"{OUT}/fig_transfer_drops{suf}.pdf")
    f.savefig(f"{OUT}/fig_transfer_drops{suf}.png", dpi=200)

    print(f"\nwrote 5 figures to {OUT}/ (suffix {suf!r})")
    return allrows


if __name__ == "__main__":
    figures()
    figures(clean=True)
