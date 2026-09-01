"""First-pass figures for the real_v1 open-loop transfer study (8 morphologies x ~10 bench trials).

Scoring rules, stated once here so they can be argued with:

* Only traces on the 40 mm / 77 mm vane count. The vane rebuild changed the task, not just the
  instrument: on sv1_w6689 the mean turn fell 4 deg and cos_hold fell 0.044 (Welch p = 0.019).
  Pooling across it would put an instrument change inside a design comparison.
* Each design keeps its LAST `--per-design` trials; the operator's early trials on a morphology
  are staging/calibration, not data.
* Three outcomes, because the instrument admits three:
    HELD        complete trace, no fall  -> carries cos_hold
    DROPPED     the shaft lost >= DROP_MM against its own baseline while still observed
    UNRESOLVED  the vane left view mid-run, so the ending was never seen. NOT a success and NOT
                a failure. Scoring these as either one is the single easiest way to manufacture
                a result here: 12 of g12's 12 trials are unresolved.
* cos at the last detection is NOT a stand-in for cos_hold. Measured over 52 complete traces the
  two correlate at r = 0.82, but the residual sd is 0.103 -- three times the within-design
  spread -- because the turn is still moving at 1.5 s. It ranks (spearman 0.86); it does not
  substitute.
"""
import argparse, csv, glob, json, os, re, collections
import numpy as np

SIM = "docs/experiments/20260831-real_v1-sobol8192/deploy_plan_bands_all.json"
DROP_MM = 25.0
#: morphology -> the plan actually deployed, from the operator's own session notes.
PLAN = {"sv1_w6689": ("sv1_w6689_b060", 0.60), "sv1_w2360": ("sv1_w2360_b075", 0.75),
        "sv1_u1364": ("sv1_u1364_b080", 0.80), "g12": ("g12_b095", 0.95),
        "sv1_u0060": ("sv1_u0060_b75", 0.75), "sv1_u0308": ("sv1_u0308_b050", 0.50),
        "rv05_manual": ("rv05_manual_b85", 0.85), "sv1_w0099": ("sv1_w0099_b100", 1.00)}
EXCLUDE = {"b01cac"}          # operator-flagged mis-staging


def bench(per_design=10):
    rows = []
    for p in sorted(glob.glob("logs/tracker/*_SUMMARY.json"), key=os.path.getmtime):
        d = json.load(open(p))
        if d.get("axial_mm") != 77.0:
            continue
        s = d["summary"]
        m = re.match(r"\d{8}-(\d{6})-(.+)-([0-9a-f]{6})$",
                     os.path.basename(p).replace("_track_SUMMARY.json", ""))
        if m.group(3) in EXCLUDE:
            continue
        void = s.get("cos_hold") is None
        dropped = s["z_drop_mm"] >= DROP_MM
        rows.append(dict(design=m.group(2), tag=m.group(3), clock=m.group(1),
                         outcome="DROPPED" if dropped else ("UNRESOLVED" if void else "HELD"),
                         cos_hold=s.get("cos_hold"), cos_peak=s["cos_peak"],
                         deg=s["deg_turned"], z_drop=s["z_drop_mm"], slip=s["slip_mm"],
                         vis=s["visibility"]))
    by = collections.defaultdict(list)
    for r in rows:
        by[r["design"]].append(r)
    return {k: v[-per_design:] for k, v in by.items()}


def sim():
    rows = json.load(open(SIM))["rows"]
    out = {}
    for dsg, (plan, clip) in PLAN.items():
        g = [r for r in rows if r["plan"] == plan and abs(r["budget"] - clip) < 1e-9]
        if not g:
            continue
        out[dsg] = dict(plan=plan, clip=clip, n=len(g),
                        final_cos=float(np.mean([r["final_cos"] for r in g])),
                        peak_cos=float(np.mean([r["peak_cos"] for r in g])),
                        ok_rate=float(np.mean([bool(r["ok"]) for r in g])),
                        contacts=float(np.mean([r["contacts"] for r in g])),
                        force_N=float(np.mean([r["force_N"] for r in g])),
                        held_z=float(np.mean([r["final_z"] for r in g])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-design", type=int, default=10)
    ap.add_argument("--out", default="docs/experiments/20260901-real_v1-transfer-firstpass")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    B, S = bench(a.per_design), sim()

    table = []
    for dsg in sorted(B, key=lambda k: -S.get(k, {}).get("final_cos", -1)):
        g = B[dsg]
        held = [r for r in g if r["outcome"] == "HELD"]
        h = np.array([r["cos_hold"] for r in held])
        table.append(dict(design=dsg, **{k: S.get(dsg, {}).get(k) for k in
                                         ("plan", "clip", "final_cos", "ok_rate", "contacts", "force_N")},
                          n=len(g),
                          n_held=len(held),
                          n_dropped=sum(r["outcome"] == "DROPPED" for r in g),
                          n_unresolved=sum(r["outcome"] == "UNRESOLVED" for r in g),
                          bench_cos=float(h.mean()) if len(h) else None,
                          bench_sd=float(h.std(ddof=1)) if len(h) > 1 else None,
                          bench_peak=float(np.mean([r["cos_peak"] for r in g])),
                          bench_deg=float(np.mean([r["deg"] for r in g]))))
    json.dump(dict(rows=table, trials={k: v for k, v in B.items()}),
              open(f"{a.out}/data.json", "w"), indent=1)

    w = f"{'design':<13}{'plan clip':>6}{'sim cos':>9}{'sim ok':>8}{'sim ct':>7}{'sim N':>7}" \
        f"{'n':>4}{'held':>6}{'drop':>6}{'unres':>7}{'bench cos':>11}{'sd':>7}{'deg':>7}"
    print(w); print("-" * len(w))
    for r in table:
        bc = f"{r['bench_cos']:+.3f}" if r["bench_cos"] is not None else "     -"
        sd = f"{r['bench_sd']:.3f}" if r["bench_sd"] is not None else "    -"
        print(f"{r['design']:<13}{r['clip']:>6.2f}{r['final_cos']:>9.3f}{r['ok_rate']:>8.2f}"
              f"{r['contacts']:>7.2f}{r['force_N']:>7.2f}{r['n']:>4}{r['n_held']:>6}"
              f"{r['n_dropped']:>6}{r['n_unresolved']:>7}{bc:>11}{sd:>7}{r['bench_deg']:>7.1f}")
    return table, B, S




# --------------------------------------------------------------------------- stats + figures
def analyse():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats
    table, B, S = main()
    out = "docs/experiments/20260901-real_v1-transfer-firstpass"
    FG, AX, TX, MU = "#ffffff", "#2b2b2b", "#1a1a1a", "#8a8a8a"
    plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": AX, "axes.labelcolor": TX,
                         "xtick.color": TX, "ytick.color": TX, "text.color": TX,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": FG, "axes.facecolor": FG})
    short = {"sv1_w6689": "w6689", "sv1_w2360": "w2360", "sv1_u1364": "u1364", "g12": "g12",
             "sv1_u0060": "u0060", "sv1_u0308": "u0308", "rv05_manual": "rv05", "sv1_w0099": "w0099"}
    fig, ax = plt.subplots(2, 2, figsize=(11, 8.2))
    lines = []

    # A -- per-design bench alignment, ordered by what the simulator predicted
    a = ax[0][0]
    for i, r in enumerate(table):
        g = B[r["design"]]
        held = [t["cos_hold"] for t in g if t["outcome"] == "HELD"]
        drp = [t for t in g if t["outcome"] == "DROPPED"]
        unr = [t for t in g if t["outcome"] == "UNRESOLVED"]
        if held:
            a.scatter(np.full(len(held), i) + np.random.uniform(-.13, .13, len(held)), held,
                      s=17, c="#1f4e79", alpha=.75, zorder=3, lw=0)
            a.plot([i - .28, i + .28], [np.mean(held)] * 2, c="#1f4e79", lw=2, zorder=4)
        if drp:
            a.text(i, -0.06, f"{len(drp)}v", ha="center", va="center", fontsize=8,
                   color="#c0392b", weight="bold")
        if unr:
            a.text(i, -0.15, f"{len(unr)}?", ha="center", va="center", fontsize=8, color=MU)
        a.scatter([i], [r["final_cos"]], s=52, marker="_", c="#e08a00", lw=2.2, zorder=5)
    a.set_xticks(range(len(table)))
    a.set_xticklabels([short[r["design"]] for r in table], rotation=35, ha="right")
    a.axhline(0, color=MU, lw=.6)
    a.set_ylabel("cos(shaft, up) held")
    a.set_title("A   bench alignment per design, ordered by the simulator's prediction\n"
                "orange dash = simulated;  v = dropped;  open square = ending unobserved",
                loc="left", fontsize=8.5)
    a.set_ylim(-0.2, 1.05)

    # B -- does the simulator rank the hands?
    b = ax[0][1]
    xs = [r["final_cos"] for r in table if r["bench_cos"] is not None]
    ys = [r["bench_cos"] for r in table if r["bench_cos"] is not None]
    nm = [short[r["design"]] for r in table if r["bench_cos"] is not None]
    b.scatter(xs, ys, s=46, c="#1f4e79", zorder=3, lw=0)
    for x, y, n in zip(xs, ys, nm):
        b.annotate(n, (x, y), textcoords="offset points", xytext=(6, 3), fontsize=7.5, color=TX)
    lo, hi = 0.35, 1.0
    b.plot([lo, hi], [lo, hi], ls="--", c=MU, lw=.9)
    rho = stats.spearmanr(xs, ys)
    pr = stats.pearsonr(xs, ys)
    b.set_xlabel("simulated cos (mean of replicates, at the deployed clip)")
    b.set_ylabel("bench cos_hold (mean of held trials)")
    b.set_title(f"B   the simulator does not rank the hands\n"
                f"spearman {rho.statistic:+.2f} (p={rho.pvalue:.2f}), "
                f"pearson {pr[0]:+.2f};  dashed = perfect transfer", loc="left", fontsize=8.5)
    lines.append(f"sim->bench alignment: spearman {rho.statistic:+.3f} p={rho.pvalue:.3f}, "
                 f"pearson {pr[0]:+.3f} p={pr[1]:.3f}, n={len(xs)}")

    # C -- the stability hypothesis: does simulated grip predict staying held?
    c = ax[1][0]
    fx = [r["force_N"] for r in table]
    fy = [r["n_dropped"] / r["n"] for r in table]
    c.scatter(fx, fy, s=46, c="#c0392b", zorder=3, lw=0)
    seen = {}
    for x, y, r in zip(fx, fy, table):
        k = round(y, 3)
        seen[k] = seen.get(k, 0) + 1
        c.annotate(short[r["design"]], (x, y), textcoords="offset points",
                   xytext=(5, 4 + 9 * (seen[k] - 1)), fontsize=7.5, color=TX)
    rf = stats.spearmanr(fx, fy)
    c.set_xlabel("simulated contact force at the deployed clip (N)")
    c.set_ylabel("bench drop rate")
    c.set_title(f"C   simulated grip force vs dropping on the bench\n"
                f"spearman {rf.statistic:+.2f} (p={rf.pvalue:.2f});  "
                f"every drop happened at force <= 0.43 N", loc="left", fontsize=8.5)
    lines.append(f"sim force_N -> bench drop rate: spearman {rf.statistic:+.3f} p={rf.pvalue:.3f}")
    rc = stats.spearmanr([r["contacts"] for r in table], fy)
    lines.append(f"sim contacts -> bench drop rate: spearman {rc.statistic:+.3f} p={rc.pvalue:.3f}")

    # D -- the case for a residual controller: the turn happens, then it is lost
    d = ax[1][1]
    pk, fn, lb = [], [], []
    for dsg, g in B.items():
        for t in g:
            if t["outcome"] == "DROPPED":
                pk.append(t["cos_peak"]); fn.append(t["cos_hold"] if t["cos_hold"] is not None else 0.0)
                lb.append(short[dsg])
    for p_, f_ in zip(pk, fn):
        d.plot([0, 1], [p_, f_], c="#c0392b", lw=1.1, alpha=.65, zorder=3)
        d.scatter([0, 1], [p_, f_], s=22, c=["#e08a00", "#c0392b"], zorder=4, lw=0)
    heldpk = [t["cos_peak"] for g in B.values() for t in g if t["outcome"] == "HELD"]
    heldfn = [t["cos_hold"] for g in B.values() for t in g if t["outcome"] == "HELD"]
    for p_, f_ in zip(heldpk, heldfn):
        d.plot([0, 1], [p_, f_], c="#1f4e79", lw=.6, alpha=.25, zorder=2)
    d.set_xticks([0, 1]); d.set_xticklabels(["peak of the turn", "what was held"])
    d.set_xlim(-.18, 1.18); d.set_ylabel("cos(shaft, up)")
    d.set_title(f"D   the {len(pk)} drops reorient first, then lose the shaft\n"
                f"peak {np.mean(pk):+.2f} -> held {np.mean(fn):+.2f};  "
                f"blue = the {len(heldpk)} trials that held", loc="left", fontsize=8.5)
    turn = 90.0 - np.degrees(np.arccos(np.clip(np.mean(pk), -1, 1)))
    lines.append(f"dropped trials: peak {np.mean(pk):+.3f} -> final {np.mean(fn):+.3f}; the shaft "
                 f"had already turned {turn:.0f} deg of the 90 before it was lost, n={len(pk)}")

    fig.tight_layout()
    fig.savefig(f"{out}/transfer_firstpass.png", dpi=190)
    print("\n" + "\n".join(lines))
    open(f"{out}/stats.txt", "w").write("\n".join(lines) + "\n")
    print(f"\nwrote {out}/transfer_firstpass.png")


if __name__ == "__main__":
    analyse()
