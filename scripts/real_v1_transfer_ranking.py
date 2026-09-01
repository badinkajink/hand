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
        held = [t for t in g if t["outcome"] == "HELD" and not t["floor"]]
        # ...and only if the tag survived to the end. g12 holds 9 of 10 and measures 0 of them:
        # its tag dies optically (the cylinder crowds the quiet zone), not by falling, so its
        # hold rate is real and its alignment is simply absent. The two are separate columns.
        h = np.array([t["cos_hold"] for t in held if t["cos_hold"] is not None])
        heldall = [t for t in g if t["outcome"] == "HELD"]
        drops = [t for t in g if t["outcome"] == "DROPPED"]
        mins = [F.get(t["tag"]) for t in heldall if F.get(t["tag"]) is not None]
        onfloor = sum(1 for v in mins if v < FLOOR_MM)
        rows.append(dict(design=dsg, short=SHORT[dsg], n=len(g), n_held=len(held),
                         n_drop=sum(t["outcome"] == "DROPPED" for t in g),
                         n_clean=len(held), n_cos=int(len(h)),
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
    rows = [r for r in allrows if r["n_cos"] >= 3]   # need >=3 measured table-free holds

    print(f"{'design':<8}{'n':>3}{'held':>5}{'drop':>5}{'unres':>6}{'floor':>6}{'medTagZ':>9}"
          f"{'bench':>8}{'sd':>7}{'peak':>7}{'deg':>6}{'simcos':>8}{'simN':>7}")
    for r in allrows:
        bc = f"{r['bench_cos']:+.3f}" if r["bench_cos"] is not None else "     -"
        sd = f"{r['bench_sd']:.3f}" if r["bench_sd"] is not None else "    -"
        flag = " <-- too few measured holds" if r["n_cos"] < 3 else ""
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
def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    allrows, rows, bk, sk, Mb, Nb, Ms, Ns = main()
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "axes.facecolor": "white",
                         "pdf.fonttype": 42})
    BL, RD, OR, MU = "#1f4e79", "#c0392b", "#e08a00", "#8a8a8a"
    B = bench(10)
    F = floor_contact()

    # ---- fig 1: per-design outcome and alignment
    f, ax = plt.subplots(figsize=(6.6, 3.0))
    for i, r in enumerate(allrows):
        g = B[r["design"]]
        for t in g:
            fl = F.get(t["tag"], 1e9) < FLOOR_MM
            if t["outcome"] == "HELD" and t["cos_hold"] is not None:
                ax.scatter(i + np.random.uniform(-.15, .15), t["cos_hold"], s=15, lw=.7,
                           c="none" if fl else BL, edgecolors=MU if fl else "none", zorder=3)
        if r["bench_cos"] is not None:
            ax.plot([i - .3, i + .3], [r["bench_cos"]] * 2, c=BL, lw=2, zorder=4)
        ax.scatter([i], [r["sim_final_cos"]], s=48, marker="_", c=OR, lw=2, zorder=5)
        if r["n_drop"]:
            ax.text(i, -0.07, f"{r['n_drop']}$\\downarrow$", ha="center", fontsize=7, color=RD)
        if r["n_unres"]:
            ax.text(i, -0.15, f"{r['n_unres']}?", ha="center", fontsize=7, color=MU)
    ax.set_xticks(range(len(allrows)))
    ax.set_xticklabels([r["short"] for r in allrows], rotation=30, ha="right")
    ax.axhline(0, c=MU, lw=.5); ax.set_ylim(-0.2, 1.02)
    ax.set_ylabel(r"$\cos(\hat{s},\,\hat{z})$ held")
    ax.set_title("bench alignment per hand, ordered by the simulator's prediction", loc="left")
    f.tight_layout(); f.savefig(f"{OUT}/fig_transfer_designs.pdf"); f.savefig(f"{OUT}/fig_transfer_designs.png", dpi=200)

    # ---- fig 2: sim vs bench, with and without the table
    f, ax = plt.subplots(1, 2, figsize=(6.6, 3.1), sharey=True)
    for k, (a, keep, ttl) in enumerate([
            (ax[0], allrows, "all held trials"),
            (ax[1], rows, "table-free trials only")]):
        xs = [r["sim_final_cos"] for r in keep if r["bench_cos"] is not None]
        ys = [r["bench_cos"] for r in keep if r["bench_cos"] is not None]
        nm = [r["short"] for r in keep if r["bench_cos"] is not None]
        if k == 0:      # panel A must use the unfiltered per-design means
            xs, ys, nm = [], [], []
            for r in allrows:
                g = [t["cos_hold"] for t in B[r["design"]]
                     if t["outcome"] == "HELD" and t["cos_hold"] is not None]
                if g:
                    xs.append(r["sim_final_cos"]); ys.append(float(np.mean(g))); nm.append(r["short"])
        a.plot([.45, .9], [.45, .9], ls="--", c=MU, lw=.8)
        a.scatter(xs, ys, s=34, c=BL, zorder=3)
        for x, y, n in zip(xs, ys, nm):
            a.annotate(n, (x, y), textcoords="offset points", xytext=(5, 3), fontsize=7)
        rho = stats.spearmanr(xs, ys)
        a.set_title(f"{ttl}\n" + r"$\rho_s$ = " + f"{rho.statistic:+.2f} (n={len(xs)}, p={rho.pvalue:.2f})",
                    loc="left")
        a.set_xlabel("simulated cos")
    ax[0].set_ylabel("bench cos held")
    f.tight_layout(); f.savefig(f"{OUT}/fig_transfer_sim2real.pdf"); f.savefig(f"{OUT}/fig_transfer_sim2real.png", dpi=200)

    # ---- fig 3: rank agreement between metrics
    lbl = {"bench_cos": "cos held", "bench_med": "median cos", "bench_peak": "cos peak",
           "bench_deg": "turn (deg)", "held_rate": "held rate", "bench_slip": "slip",
           "hold_rate_op": "hold rate", "held_slip": "slip (held)",
           "sim_final_cos": "sim cos", "sim_peak_cos": "sim peak", "sim_ok_rate": "sim retained",
           "sim_contacts": "sim contacts", "sim_force_N": "sim force"}
    use = [r for r in allrows if r["bench_peak"] is not None]
    bk2 = ["bench_peak", "bench_deg", "hold_rate_op", "held_slip", "bench_cos"]
    Mb2, Nb2 = rank_table(use, bk2)
    Mx, Nx = rank_table(use, sk + bk2)
    f, ax = plt.subplots(1, 2, figsize=(7.0, 3.4),
                         gridspec_kw={"width_ratios": [len(bk2), len(sk)]})
    for a, M, N, rl, cl, ttl in [
            (ax[0], Mb2, Nb2, bk2, bk2, "bench metrics agree with each other"),
            (ax[1], Mx[len(sk):, :len(sk)], Nx[len(sk):, :len(sk)], bk2, sk,
             "each simulated metric vs each bench metric")]:
        im = a.imshow(M, vmin=-1, vmax=1, cmap="RdBu_r")
        a.set_xticks(range(len(cl))); a.set_xticklabels([lbl[k] for k in cl], rotation=40, ha="right")
        a.set_yticks(range(len(rl)))
        a.set_yticklabels([lbl[k] for k in rl] if a is ax[0] else [])
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if not np.isnan(M[i, j]):
                    a.text(j, i, f"{M[i,j]:+.2f}", ha="center", va="center", fontsize=6.5,
                           color="white" if abs(M[i, j]) > .6 else "#222")
        a.set_title(ttl, loc="left", fontsize=8)
    f.colorbar(im, ax=ax, shrink=.8, label="Spearman $\\rho$")
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
    ax.text(74, EJECT_SLIP_MM + 1.5, "ejected", fontsize=7, color=MU, ha="right")
    ax.set_xlabel("net turn (deg)"); ax.set_ylabel("slip (mm)")
    ax.scatter([], [], s=20, c=BL, label="held"); ax.scatter([], [], s=26, marker="x", c=RD,
                                                             lw=1.1, label="dropped")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.set_title("a drop either overshoots the turn\nor never gets one", loc="left")
    f.tight_layout(); f.savefig(f"{OUT}/fig_transfer_drops.pdf")
    f.savefig(f"{OUT}/fig_transfer_drops.png", dpi=200)

    print(f"\nwrote 4 figures to {OUT}/")
    return allrows, rows, Mb2, Mx, bk2, sk


if __name__ == "__main__":
    figures()
