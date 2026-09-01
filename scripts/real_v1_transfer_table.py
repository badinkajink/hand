"""One table for the open-loop transfer study: what the simulator ranked, what the bench did.

Column choices, and why these and not the others:

  sim cos      the ranking the study was launched on (per-plan mean at the deployed clip).
  hold         the OPERATOR'S hold rate over the design's last session. The tag cannot supply
               this: it dies on 34 of 74 trials and the operator was standing there.
  meas         holds whose tag survived to the hold window -- the only trials that can carry
               an alignment number, and fewer than `hold` because the cylinder crowds the
               vane's quiet zone as the shaft comes up. Where a hand measures none of its
               holds (g12) the run's PEAK alignment stands in and the row is marked.
  bench cos    the bench answer to the sim column, over `meas` trials only.
  turn         net degrees turned, holds vs drops. Where the drops turn FURTHER than the holds
               that difference is the shaft's motion after it left the grasp.
  slip         lateral travel of the cylinder centre, holds vs drops. This is what separates
               the two failure modes; see `mode`.
  mode         eject (the shaft leaves the grasp sideways, 20-55 mm, before any real turn)
               or stall (the drops end up turned LESS than the same hand's holds while
               travelling no further). A hand whose drops merely record MORE turn gets no
               mode: that difference is the fall, and telling it from a real overshoot needs
               servo load, which these sessions did not record.

Every hand is ranked. There is no floor-contact exclusion; see the docstring of
real_v1_transfer_ranking.py for the measurement that retired it.
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from real_v1_transfer_ranking import build                              # noqa: E402
from real_v1_transfer_figures import PLAN                                # noqa: E402

OUT = "paper/transfer_table.tex"
MAP = "paper/transfer_appendix_designs.tex"


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
    rows.sort(key=lambda r: -r["sim_final_cos"])
    sr = ranks([r["sim_final_cos"] for r in rows])
    br = ranks([r["bench_cos"] for r in rows])       # every hand is ranked

    L = [r"\begin{table}[t]", r"\centering", r"\footnotesize",
         r"\setlength{\tabcolsep}{4pt}",
         r"\caption{Eight \texttt{real\_v1} hands, one exported open-loop plan each, one bench "
         r"session each, named \texttt{D1}--\texttt{D8} in the order simulation ranked them "
         r"(Appendix~\ref{tab:transfer-map}). \emph{Hold} is the operator's verdict; "
         r"\emph{meas} counts the holds carrying an alignment measurement. \emph{Turn} and "
         r"\emph{slip} are held\,/\,dropped. A drop's turn is measured while the shaft is "
         r"still moving and includes travel after it left the grasp, so \emph{mode} names "
         r"only what slip and a shortfall in turn can establish. "
         r"Superscripts are ranks. $*$: on this hand the tag stopped resolving before the "
         r"hold window on every held trial, so peak alignment stands in.}",
         r"\label{tab:transfer}",
         r"\begin{tabular}{lccccccl}", r"\toprule",
         r"design & sim $\cos$ & hold & meas & bench $\cos$ & turn (deg) & slip (mm) & mode \\",
         r"\midrule"]
    sym = {"": "", "*": r"$^{*}$"}
    for r, s, b in zip(rows, sr, br):
        sd = rf"\pm{r['bench_sd']:.3f}" if r["bench_sd"] is not None else ""
        bc = rf"${r['bench_cos']:.3f}{sd}^{{{b}}}$"
        L.append(rf"\texttt{{{r['did']}}}{sym[r['qual']]} & ${r['sim_final_cos']:.3f}^{{{s}}}$ & "
                 rf"{r['n_heldall']}/{r['n']} & {r['n_cos']} & {bc} & "
                 rf"{fmt(r['held_deg'], 0)}\,/\,{fmt(r['drop_deg'], 0)} & "
                 rf"{fmt(r['held_slip'])}\,/\,{fmt(r['drop_slip'])} & {r['mode']} \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(L))

    M = [r"\begin{table}[t]", r"\centering", r"\footnotesize",
         r"\caption{Design labels. \texttt{D}$k$ is the hand simulation ranked $k$th; the "
         r"internal tag and the exported plan are given so every number in "
         r"Table~\ref{tab:transfer} can be traced to a run directory.}",
         r"\label{tab:transfer-map}", r"\begin{tabular}{llll}", r"\toprule",
         r"design & internal tag & exported plan & clip \\", r"\midrule"]
    for r in rows:
        plan, clip = PLAN[r["design"]]
        M.append(rf"\texttt{{{r['did']}}} & \texttt{{{r['short']}}} & "
                 rf"\texttt{{{plan.replace('_', chr(92) + '_')}}} & {clip:.2f} \\")
    M += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    open(MAP, "w").write("\n".join(M))

    hdr = (f"{'hand':<10}{'simcos':>8}{'r':>3}{'hold':>7}{'meas':>6}{'benchcos':>10}{'r':>3}"
           f"{'turn h/d':>14}{'slip h/d':>13}  mode")
    print(hdr); print("-" * len(hdr))
    for r, s, b in zip(rows, sr, br):
        bc = f"{r['bench_cos']:.3f}{r['qual']}"
        print(f"{r['did']} {r['short']:<7}{r['sim_final_cos']:>8.3f}{s:>3}"
              f"{r['n_heldall']:>4}/{r['n']:<2}{r['n_cos']:>6}{bc:>10}{b if b else '-':>3}"
              f"{fmt(r['held_deg'],0):>7}/{fmt(r['drop_deg'],0):<6}"
              f"{fmt(r['held_slip']):>6}/{fmt(r['drop_slip']):<6}  {r['mode']}")
    print(f"\nwrote {OUT} and {MAP}")
    return rows


if __name__ == "__main__":
    main()
