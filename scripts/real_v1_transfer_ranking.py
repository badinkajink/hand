"""Ranking analysis for the real_v1 open-loop transfer study.

Two questions, kept separate:
  METRIC CONSISTENCY   -- do different ways of scoring a hand agree on the ORDER of the hands?
  SIM2REAL CONSISTENCY -- does any simulated quantity reproduce a bench order?

Trial scoring is imported from real_v1_transfer_figures.py. One rule is added here:

FLOOR CONTACT. The vane can reach the table during the turn, and a shaft resting on the table is
not a shaft the hand is holding on its own. `sv1_u0060` does this on 10 of 10 trials -- tag centre
-2.8 to +3.7 mm above the bench floor, against 12-21 mm for `sv1_w6689` -- which is why it scores
best in the set. It is reported, and excluded from every ranking.
"""
import argparse, csv, glob, json, os, re, sys
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from real_v1_transfer_figures import bench, sim, PLAN          # noqa: E402

OUT = "paper/figures"
FLOOR_MM = 8.0          # tag centre this close to the bench floor => the vane is on the table
EJECT_SLIP_MM = 20.0    # a drop that travels this far never had the shaft in the first place
SHORT = {"sv1_w6689": "w6689", "sv1_w2360": "w2360", "sv1_u1364": "u1364", "g12": "g12",
         "sv1_u0060": "u0060", "sv1_u0308": "u0308", "rv05_manual": "rv05", "sv1_w0099": "w0099"}
#: Publication names. Every hand is D<k> where k is its rank in simulation, so a reader can see
#: the predicted order on the axis and the measured order in the data. The mapping from D<k> to
#: the nine morphology parameters belongs in the appendix, not on an axis tick.
DESIGN_ID = {"sv1_w6689": "D1", "sv1_w2360": "D2", "sv1_u1364": "D3", "g12": "D4",
             "sv1_u0060": "D5", "sv1_u0308": "D6", "rv05_manual": "D7", "sv1_w0099": "D8"}


def floor_contact(per_design=10):
    """min tag height per trial, so a floor-supported turn can be told from a held one."""
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


def build(per_design=10):
    B, S, F = bench(per_design), sim(), floor_contact()
    rows = []

    def mode(drops):
        """Two failure modes, and they are not the same defect.

        OVERSHOOT: the turn completes and keeps going, then the grip lets go. The drops turn
        FURTHER than the same hand's holds (w0099: 65 deg vs 44) and travel little.
        EJECTION:  the shaft leaves the grip before any turn happens -- net turn near zero and
        20-55 mm of travel. Only overshoot is an argument for closing the loop; ejection is a
        grasp that never closed."""
        if not drops:
            return "--"
        sl = np.mean([t["slip"] for t in drops if t["slip"] is not None])
        return "eject" if sl >= EJECT_SLIP_MM else "overshoot"

    for dsg, g in B.items():
        for t in g:
            t["floor"] = F.get(t["tag"], 1e9) < FLOOR_MM
        # a held trial only counts toward alignment if the hand held it WITHOUT the table
        heldall = [t for t in g if t["outcome"] == "HELD"]
        drops = [t for t in g if t["outcome"] == "DROPPED"]
        held = [t for t in heldall if not t["floor"]]
        # ...and only if the tag survived to the end.
        h = np.array([t["cos_hold"] for t in held if t["cos_hold"] is not None])
        # EVERY hand carries an alignment number, so that every hand carries a rank and the
        # sim/bench comparison is over the same eight. Two need a substitute, and each is
        # marked with the symbol in `qual` wherever it is printed:
        #   *  the tag dies before the hold window, so the run's PEAK alignment stands in.
        #      g12 holds 9 of 10; its vane is crowded out of decode range by the cylinder,
        #      which is an instrument failure and not a manipulation one.
        #   d  no hold was clear of the table, so the table-supported holds are used. u0060
        #      turns furthest in the set and every one of its three holds was resting on it,
        #      so this number is an upper bound on what the hand does unaided.
        qual, pool = "", held
        if not len(h):                          # no table-free hold carried a hold-window cos
            if not pool:                        # ...because no hold was clear of the table
                pool, qual = heldall, qual + "d"
            got = [t["cos_hold"] for t in pool if t["cos_hold"] is not None]
            if got:
                h = np.array(got)
            else:                               # ...and the tag never survived to the window
                h, qual = np.array([t["cos_peak"] for t in pool
                                    if t["cos_peak"] is not None]), qual + "*"
        mins = [F.get(t["tag"]) for t in heldall if F.get(t["tag"]) is not None]
        onfloor = sum(1 for v in mins if v < FLOOR_MM)
        rows.append(dict(design=dsg, short=SHORT[dsg], n=len(g), n_held=len(held),
                         n_drop=sum(t["outcome"] == "DROPPED" for t in g),
                         n_clean=len(held), n_cos=int(len(h)), qual=qual,
                         did=DESIGN_ID[dsg],
                         n_unres=sum(t["outcome"] == "UNRESOLVED" for t in g),
                         n_heldall=len(heldall), mode=mode(drops),
                         hold_rate_op=len(heldall) / len(g),
                         held_slip=float(np.mean([t["slip"] for t in heldall
                                                  if t["slip"] is not None])) if heldall else None,
                         drop_slip=float(np.mean([t["slip"] for t in drops
                                                  if t["slip"] is not None])) if drops else None,
                         held_deg=float(np.mean([t["deg"] for t in heldall
                                                 if t["deg"] is not None])) if heldall else None,
                         drop_deg=float(np.mean([t["deg"] for t in drops
                                                 if t["deg"] is not None])) if drops else None,
                         floor_trials=onfloor, min_tag_z=float(np.median(mins)) if mins else None,
                         held_rate=sum(t["outcome"] == "HELD" and not t["floor"]
                                       for t in g) / len(g),
                         bench_cos=float(h.mean()) if len(h) else None,
                         bench_med=float(np.median(h)) if len(h) else None,
                         bench_sd=float(h.std(ddof=1)) if len(h) > 1 else None,
                         bench_peak=float(np.mean([t["cos_peak"] for t in g
                                                   if t["cos_peak"] is not None])),
                         bench_deg=float(np.mean([t["deg"] for t in g
                                                  if t["deg"] is not None])),
                         bench_slip=float(np.mean([t["slip"] for t in g
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

    print(f"{'design':<8}{'n':>3}{'held':>5}{'drop':>5}{'unres':>6}{'floor':>6}{'medTagZ':>9}"
          f"{'bench':>8}{'sd':>7}{'peak':>7}{'deg':>6}{'simcos':>8}{'simN':>7}")
    for r in allrows:
        bc = f"{r['bench_cos']:+.3f}{r['qual']}" if r["bench_cos"] is not None else "     -"
        sd = f"{r['bench_sd']:.3f}" if r["bench_sd"] is not None else "    -"
        flag = ("  * peak stands in" if "*" in r["qual"] else "") + \
               ("  d table-supported" if "d" in r["qual"] else "")
        print(f"{r['short']:<8}{r['n']:>3}{r['n_held']:>5}{r['n_drop']:>5}{r['n_unres']:>6}"
              f"{r['floor_trials']:>3}/{r['n_heldall']:<2}"
              f"{(r['min_tag_z'] if r['min_tag_z'] is not None else float('nan')):>9.1f}{bc:>8}{sd:>7}{r['bench_peak']:>+7.3f}"
              f"{r['bench_deg']:>6.1f}{r['sim_final_cos']:>8.3f}{r['sim_force_N']:>7.2f}{flag}")

    bench_keys = ["bench_cos", "bench_med", "bench_peak", "bench_deg", "hold_rate_op",
                  "held_slip"]
    sim_keys = ["sim_final_cos", "sim_peak_cos", "sim_ok_rate", "sim_contacts", "sim_force_N"]
    Mb, Nb = rank_table(rows, bench_keys)
    Ms, Ns = rank_table(rows, sim_keys + bench_keys)
    json.dump(dict(rows=allrows, used=[r["short"] for r in rows],
                   bench_keys=bench_keys, sim_keys=sim_keys,
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
    return r["did"] + {"": "", "*": r"$^{*}$", "d": r"$^{\dagger}$",
                       "d*": r"$^{*\dagger}$"}.get(r["qual"], "")


def all_hold_cos(g):
    """Mean alignment over EVERY hold the operator called, table-supported ones included."""
    v = [t["cos_hold"] for t in g if t["outcome"] == "HELD" and t["cos_hold"] is not None]
    if v:
        return float(np.mean(v)), ""
    v = [t["cos_peak"] for t in g if t["outcome"] == "HELD" and t["cos_peak"] is not None]
    return (float(np.mean(v)), "*") if v else (None, "")


def figures():
    plt = _style()
    allrows, rows, bk, sk, Mb, Nb, Ms, Ns = main()
    for r in allrows:
        r["hold_rate_op"] = r["n_heldall"] / r["n"]
    B, F = bench(10), floor_contact()
    lbl = {"bench_cos": "Measured alignment", "bench_med": "Median alignment",
           "bench_peak": "Peak alignment", "bench_deg": "Net turn",
           "hold_rate_op": "Hold rate", "held_slip": "Slip",
           "sim_final_cos": "Simulated alignment", "sim_peak_cos": "Simulated peak",
           "sim_ok_rate": "Simulated retention", "sim_contacts": "Simulated contacts",
           "sim_force_N": "Simulated force"}

    # ---- fig 1: per-design alignment, every hand present
    f, ax = plt.subplots(figsize=(6.6, 3.5))
    for i, r in enumerate(allrows):
        clean = sup = peak = None
        for t in B[r["design"]]:
            if t["outcome"] != "HELD":
                continue
            fl = F.get(t["tag"], 1e9) < FLOOR_MM
            y, m = (t["cos_hold"], "cos") if t["cos_hold"] is not None else (t["cos_peak"], "pk")
            if y is None:
                continue
            x = i + np.random.uniform(-.16, .16)
            if m == "pk":
                peak = ax.scatter(x, y, s=22, marker="^", facecolors="none", edgecolors=BL, lw=.8,
                                  zorder=3)
            elif fl:
                sup = ax.scatter(x, y, s=20, facecolors="none", edgecolors=MU, lw=.8, zorder=3)
            else:
                clean = ax.scatter(x, y, s=17, c=BL, lw=0, zorder=4)
        mean, = ax.plot([i - .32, i + .32], [r["bench_cos"]] * 2, c=BL, lw=2.2, zorder=5)
        pred = ax.scatter([i], [r["sim_final_cos"]], s=110, marker="_", c=OR, lw=2.2, zorder=6)
        if r["n_drop"]:
            ax.text(i, 0.03, f"{r['n_drop']}$\\downarrow$", ha="center", fontsize=7, color=RD)
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
        Line2D([], [], ls="none", marker="o", ms=4, color=BL, label="Hold, clear of the table"),
        Line2D([], [], ls="none", marker="o", ms=5, mfc="none", mec=MU,
               label="Hold, table-supported"),
        Line2D([], [], ls="none", marker="^", ms=5, mfc="none", mec=BL,
               label="Peak (tag lost before the hold window)")],
        frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(.5, -0.16))
    f.tight_layout(); f.savefig(f"{OUT}/fig_transfer_designs.pdf")
    f.savefig(f"{OUT}/fig_transfer_designs.png", dpi=200)

    # ---- fig 2: the simulator against each of the two bench axes, all eight hands
    f, ax = plt.subplots(1, 2, figsize=(6.6, 3.2))
    xs = [r["sim_final_cos"] for r in allrows]
    for a, ys, ylab, ttl, diag in [
            (ax[0], [r["bench_cos"] for r in allrows], "Measured alignment",
             "Does the simulator rank the turn?", True),
            (ax[1], [r["hold_rate_op"] for r in allrows], "Hold rate",
             "Does it rank the grip?", False)]:
        if diag:
            a.plot([.45, .95], [.45, .95], ls="--", c=MU, lw=.8, zorder=1)
        for x, y, r in zip(xs, ys, allrows):
            q = r["qual"] if diag else ""
            a.scatter([x], [y], s=42, zorder=3, c=BL if not q else "none", edgecolors=BL,
                      lw=1.1, marker="^" if "*" in q else "o")
            a.annotate(tick(r) if diag else r["did"], (x, y), textcoords="offset points",
                       xytext=(6, 3), fontsize=7)
        rho = stats.spearmanr(xs, ys)
        a.set_title(f"{ttl}\n" + r"$\rho_s$ = " + f"{rho.statistic:+.2f} (n=8,"
                    f" p={rho.pvalue:.2f})", loc="left")
        a.set_xlabel("Simulated alignment"); a.set_ylabel(ylab)
    sub = [r for r in allrows if not r["qual"]]
    rr = stats.spearmanr([r["sim_final_cos"] for r in sub], [r["bench_cos"] for r in sub])
    ax[0].plot([], [], " ", label=r"substituted hands out: $\rho_s$ = "
               + f"{rr.statistic:+.2f} (n={len(sub)})")
    ax[0].scatter([], [], s=40, facecolors="none", edgecolors=BL, marker="^",
                  label=r"$*$ peak stands in")
    ax[0].scatter([], [], s=40, facecolors="none", edgecolors=BL,
                  label=r"$\dagger$ table-supported")
    ax[0].legend(frameon=False, loc="lower left", fontsize=6.5)
    ax[0].set_ylim(0.08, 1.02); ax[1].set_ylim(-0.05, 1.12)
    f.tight_layout(); f.savefig(f"{OUT}/fig_transfer_sim2real.pdf")
    f.savefig(f"{OUT}/fig_transfer_sim2real.png", dpi=200)

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
    a.set_title("Rank in simulation against rank on the bench", loc="left")

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
    a.set_title("What the simulator predicts", loc="left")
    f.tight_layout(); f.savefig(f"{OUT}/fig_transfer_rankflow.pdf")
    f.savefig(f"{OUT}/fig_transfer_rankflow.png", dpi=200)

    # ---- fig 3b (appendix): the full matrices
    bk2 = ["bench_peak", "bench_deg", "hold_rate_op", "held_slip", "bench_cos"]
    Mb2, _ = rank_table(allrows, bk2)
    Mx, _ = rank_table(allrows, sk + bk2)
    f, ax = plt.subplots(1, 2, figsize=(7.0, 3.4),
                         gridspec_kw={"width_ratios": [len(bk2), len(sk)]})
    for a, M, rl, cl, ttl in [
            (ax[0], Mb2, bk2, bk2, "Bench metrics against each other"),
            (ax[1], Mx[len(sk):, :len(sk)], bk2, sk, "Simulated against bench")]:
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
    f.savefig(f"{OUT}/fig_transfer_ranking.pdf", bbox_inches="tight")
    f.savefig(f"{OUT}/fig_transfer_ranking.png", dpi=200, bbox_inches="tight")

    # ---- fig 4: the two failure modes, in one plane
    f, ax = plt.subplots(figsize=(4.3, 3.2))
    for g in B.values():
        for t in g:
            if t["deg"] is None or t["slip"] is None:
                continue
            if t["outcome"] == "HELD":
                ax.scatter(t["deg"], t["slip"], s=20, c=BL, zorder=3, lw=0)
            elif t["outcome"] == "DROPPED":
                ax.scatter(t["deg"], t["slip"], s=26, marker="x", c=RD, zorder=4, lw=1.1)
    ax.axhline(EJECT_SLIP_MM, c=MU, lw=.7, ls="--")
    ax.text(74, EJECT_SLIP_MM + 1.5, "Ejected", fontsize=7, color=MU, ha="right")
    ax.set_xlabel("Net turn (deg)"); ax.set_ylabel("Slip (mm)")
    ax.scatter([], [], s=20, c=BL, label="Held")
    ax.scatter([], [], s=26, marker="x", c=RD, lw=1.1, label="Dropped")
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("A drop either overshoots the turn\nor never gets one", loc="left")
    f.tight_layout(); f.savefig(f"{OUT}/fig_transfer_drops.pdf")
    f.savefig(f"{OUT}/fig_transfer_drops.png", dpi=200)

    print(f"\nwrote 5 figures to {OUT}/")
    return allrows


if __name__ == "__main__":
    figures()
