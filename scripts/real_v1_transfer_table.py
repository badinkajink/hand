"""One table for the open-loop transfer study: what the simulator ranked, what the bench did.

Column choices, and why these and not the others:

  sim cos        the ranking the study was launched on (per-plan mean at the deployed clip).
  held           TABLE-FREE holds / trials. A trial whose vane passes within FLOOR_MM of the
                 bench floor is not a hold the hand produced, so it counts against neither
                 column -- it is reported separately under `table`.
  cos held       the bench answer to the sim column. sd is over that hand's table-free holds.
  z end          cylinder centre at the end of the trace. The bench holds the shaft 60-130 mm
                 lower than the plan does, and the hands that drop are the ones holding lowest,
                 so this is the stability channel the alignment columns cannot see.
  slip           mean lateral travel of the cylinder centre -- the movement/jitter channel.
  table          trials the floor participated in. This is an instrument confound, not a
                 property of the hand: the 77 mm vane reaches the table on turns where the
                 shaft alone would clear.

Ranks are printed only where the quantity is defined on >= 3 table-free holds.
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from real_v1_transfer_ranking import build, FLOOR_MM                    # noqa: E402
from real_v1_transfer_figures import bench, PLAN                        # noqa: E402
import csv, glob                                                        # noqa: E402

OUT = "paper/transfer_table.tex"
MIN_HELD = 3


def held_z():
    """cylinder centre (bench mm) at the end of each trace, averaged per design."""
    out = {}
    for d, g in bench(10).items():
        v = []
        for t in g:
            p = glob.glob(f"logs/tracker/*-{t['tag']}_track.csv")
            r = [x for x in csv.DictReader(open(p[0])) if x.get("cos")]
            v.append(float(r[-1]["z_bench_mm"]))
        out[d] = float(np.mean(v))
    return out


def ranks(vals):
    """1 = best; None passes through."""
    ok = sorted([i for i, v in enumerate(vals) if v is not None], key=lambda i: -vals[i])
    r = [None] * len(vals)
    for k, i in enumerate(ok):
        r[i] = k + 1
    return r


def main():
    rows, Z = build(10), held_z()
    for r in rows:
        r["z_end"] = Z[r["design"]]
        r["rankable"] = r["n_held"] >= MIN_HELD
    rows.sort(key=lambda r: -r["sim_final_cos"])
    sr = ranks([r["sim_final_cos"] for r in rows])
    br = ranks([r["bench_cos"] if r["rankable"] else None for r in rows])

    L = [r"\begin{table}[t]", r"\centering", r"\footnotesize",
         r"\setlength{\tabcolsep}{4pt}",
         r"\caption{Eight \texttt{real\_v1} hands, one exported open-loop plan each, ten bench "
         r"trials each. \emph{held} counts only trials in which the vane stayed clear of the "
         r"table; \emph{table} counts the trials it did not, which are excluded from every "
         r"alignment column because a shaft the table is supporting is not one the hand is "
         r"holding. Superscripts are ranks; the bench rank is defined only for hands with at "
         r"least three table-free holds.}",
         r"\label{tab:transfer}",
         r"\begin{tabular}{lccccccc}", r"\toprule",
         r"hand & sim $\cos$ & held & bench $\cos$ & $z$ end & slip & turn & table \\",
         r" & & & (mean\,$\pm$\,sd) & (mm) & (mm) & (deg) & \\",
         r"\midrule"]
    for r, s, b in zip(rows, sr, br):
        bc = (rf"${r['bench_cos']:.3f}\pm{r['bench_sd']:.3f}^{{{b}}}$"
              if r["rankable"] else
              (rf"$({r['bench_cos']:.3f})$" if r["bench_cos"] is not None else "--"))
        L.append(rf"\texttt{{{r['short']}}} & ${r['sim_final_cos']:.3f}^{{{s}}}$ & "
                 rf"{r['n_held']}/{r['n']} & {bc} & {r['z_end']:.0f} & {r['bench_slip']:.1f} & "
                 rf"{r['bench_deg']:.0f} & {r['floor_trials']}/{r['n']} \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(L))

    hdr = f"{'hand':<7}{'simcos':>8}{'r':>3}{'held':>7}{'benchcos':>10}{'r':>3}{'zend':>7}{'slip':>7}{'turn':>6}{'table':>7}"
    print(hdr); print("-" * len(hdr))
    for r, s, b in zip(rows, sr, br):
        bc = f"{r['bench_cos']:.3f}" if r["bench_cos"] is not None else "-"
        print(f"{r['short']:<7}{r['sim_final_cos']:>8.3f}{s:>3}{r['n_held']:>4}/{r['n']:<2}"
              f"{bc:>10}{b if b else '-':>3}{r['z_end']:>7.0f}{r['bench_slip']:>7.1f}"
              f"{r['bench_deg']:>6.0f}{r['floor_trials']:>4}/{r['n']:<2}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
