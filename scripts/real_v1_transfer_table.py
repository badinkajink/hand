"""One table for the open-loop transfer study: what the simulator ranked, what the bench did.

Column choices, and why these and not the others:

  sim cos      the ranking the study was launched on (per-plan mean at the deployed clip).
  hold         the OPERATOR'S hold rate over the design's last session. The tag cannot supply
               this: it dies on 34 of 74 trials and the operator was standing there.
  meas         holds that are both table-free and completely traced -- the only trials that
               can carry an alignment number. `hold` and `meas` differ for two unrelated
               reasons and the table keeps them apart: g12 holds 9 of 10 and measures none of
               them because its vane reaches the table (and its tag then dies optically);
               u0060 holds 3 and measures none because all three were resting on it.
  bench cos    the bench answer to the sim column, over `meas` trials only.
  turn         net degrees turned, holds vs drops. Where the drops turn FURTHER than the holds
               the hand is overshooting, not failing to turn.
  slip         lateral travel of the cylinder centre, holds vs drops. This is what separates
               the two failure modes; see `mode`.
  mode         overshoot (turn completes, then the grip lets go) or eject (the shaft leaves
               before any turn). Only the first is an argument for closed-loop control.

Ranks are printed only where the quantity rests on at least MIN_HELD measured holds.
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from real_v1_transfer_ranking import build                              # noqa: E402

OUT = "paper/transfer_table.tex"
MIN_HELD = 3
SUSPECT = {"u1364": "mount re-staged mid-session; 7 trials, none scored"}


def ranks(vals):
    """1 = best; None passes through."""
    ok = sorted([i for i, v in enumerate(vals) if v is not None], key=lambda i: -vals[i])
    r = [None] * len(vals)
    for k, i in enumerate(ok):
        r[i] = k + 1
    return r


def fmt(v, n=1, sign=False):
    return "--" if v is None else f"{v:{'+' if sign else ''}.{n}f}"


def main():
    rows = build(10)
    for r in rows:
        r["rankable"] = r["n_cos"] >= MIN_HELD and r["short"] not in SUSPECT
    rows.sort(key=lambda r: -r["sim_final_cos"])
    sr = ranks([r["sim_final_cos"] for r in rows])
    br = ranks([r["bench_cos"] if r["rankable"] else None for r in rows])

    L = [r"\begin{table}[t]", r"\centering", r"\footnotesize",
         r"\setlength{\tabcolsep}{4pt}",
         r"\caption{Eight \texttt{real\_v1} hands, one exported open-loop plan each, one bench "
         r"session each. \emph{hold} is the operator's verdict; \emph{meas} counts the holds "
         r"that are both clear of the table and completely traced, which are the only ones "
         r"carrying an alignment number. \emph{turn} and \emph{slip} are given as "
         r"holds\,/\,drops: where the drops turn further and travel little the hand is "
         r"overshooting, and where they barely turn at all the shaft was ejected. Superscripts "
         r"are ranks. \texttt{u1364} is bracketed -- its mount was re-staged mid-session.}",
         r"\label{tab:transfer}",
         r"\begin{tabular}{lccccccl}", r"\toprule",
         r"hand & sim $\cos$ & hold & meas & bench $\cos$ & turn (deg) & slip (mm) & mode \\",
         r"\midrule"]
    for r, s, b in zip(rows, sr, br):
        if r["bench_cos"] is None:
            bc = "--"
        elif r["rankable"]:
            bc = rf"${r['bench_cos']:.3f}\pm{r['bench_sd']:.3f}^{{{b}}}$"
        else:
            bc = rf"$({r['bench_cos']:.3f})$"
        nm = rf"\texttt{{{r['short']}}}"
        if r["short"] in SUSPECT:
            nm = rf"[{nm}]"
        L.append(rf"{nm} & ${r['sim_final_cos']:.3f}^{{{s}}}$ & "
                 rf"{r['n_heldall']}/{r['n']} & {r['n_cos']} & {bc} & "
                 rf"{fmt(r['held_deg'], 0)}\,/\,{fmt(r['drop_deg'], 0)} & "
                 rf"{fmt(r['held_slip'])}\,/\,{fmt(r['drop_slip'])} & {r['mode']} \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(L))

    hdr = (f"{'hand':<7}{'simcos':>8}{'r':>3}{'hold':>7}{'meas':>6}{'benchcos':>10}{'r':>3}"
           f"{'turn h/d':>14}{'slip h/d':>13}  mode")
    print(hdr); print("-" * len(hdr))
    for r, s, b in zip(rows, sr, br):
        bc = f"{r['bench_cos']:.3f}" if r["bench_cos"] is not None else "-"
        print(f"{r['short']:<7}{r['sim_final_cos']:>8.3f}{s:>3}"
              f"{r['n_heldall']:>4}/{r['n']:<2}{r['n_cos']:>6}{bc:>10}{b if b else '-':>3}"
              f"{fmt(r['held_deg'],0):>7}/{fmt(r['drop_deg'],0):<6}"
              f"{fmt(r['held_slip']):>6}/{fmt(r['drop_slip']):<6}  {r['mode']}")
    print(f"\nwrote {OUT}")
    return rows


if __name__ == "__main__":
    main()
