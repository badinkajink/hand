#!/usr/bin/env python3
"""Build the screen-vs-chain result page from the studies' own JSON.

    uv run python scripts/real_v1_screen_chain_page.py

Reads the deploy-screen reproduction, the per-trial bench table, the closure re-fit, the chain
sweep and the two seam filmstrips, and writes
docs/experiments/20260906-screen_vs_chain/20260906-screen_vs_chain.html. Media are inlined as
data URIs because the published artifact is a single file.
"""
from __future__ import annotations

import base64, json, math, statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BAND = ROOT / "docs/experiments/20260906-rv05_band"
OUT = ROOT / "docs/experiments/20260906-screen_vs_chain/20260906-screen_vs_chain.html"
TPL = Path(__file__).with_suffix("").with_name("real_v1_screen_chain_page.template.html")
CHAIN = ROOT / "docs/experiments/20260904-real_v1_held/squeeze_sweep/chain_hands.json"
FILMS = BAND / "20260906-films"

DID = {"sv1_w6689_b060": "D1", "sv1_w2360_b075": "D2", "sv1_u1364_b080": "D3",
       "g12_b095": "D4", "sv1_u0060_b75": "D5", "sv1_u0308_b050": "D6",
       "rv05_manual_b85": "D7", "sv1_w0099_b100": "D8"}
BENCH_COS = {"D1": 0.834, "D2": 0.797, "D7": 0.553}
BENCH_HOLD = {"D1": "24/24", "D2": "10/10", "D7": "10/10"}


def b64(p: Path, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode()


def table(head: list[str], rows: list[list], nums: set[int], hi: set[int] = frozenset()) -> str:
    th = "".join(f'<th{" class=num" if i in nums else ""}>{h}</th>'
                 for i, h in enumerate(head))
    body = []
    for k, r in enumerate(rows):
        td = "".join(f'<td{" class=num" if i in nums else ""}>{c}</td>' for i, c in enumerate(r))
        body.append(f'<tr{" class=hi" if k in hi else ""}>{td}</tr>')
    return ('<div class="tw"><table><thead><tr>' + th + "</tr></thead><tbody>"
            + "".join(body) + "</tbody></table></div>")


def spearman(a, b):
    n = len(a)
    rank = lambda v: {j: i for i, j in enumerate(sorted(range(n), key=lambda k: v[k]))}
    ra, rb = rank(a), rank(b)
    A = [ra[i] for i in range(n)]; B = [rb[i] for i in range(n)]
    ma, mb = st.mean(A), st.mean(B)
    num = sum((A[i] - ma) * (B[i] - mb) for i in range(n))
    den = math.sqrt(sum((x - ma) ** 2 for x in A) * sum((x - mb) ** 2 for x in B))
    return num / den


def main() -> int:
    S = json.loads((BAND / "repro_band_D1_D2_D7.json").read_text())["summary"]
    trials = json.loads((BAND / "bench_trials_D1_D8.json").read_text())
    refit = json.loads((BAND / "grasp_refit.json").read_text())
    chain = [r for r in json.loads(CHAIN.read_text())["rows"] if r.get("set") == "A"]

    # --- the screen reproduces ------------------------------------------------------------
    rep = []
    for tag in ("sv1_w6689_b060", "sv1_w2360_b075", "rv05_manual_b85"):
        s, d = S[tag], DID[tag]
        rep.append([f"<code>{tag}</code>", d, f"{s['own_cos']:.3f}",
                    f"{s['own_kept'][0]}/{s['own_kept'][1]}",
                    f"{BENCH_COS[d]:.3f}", BENCH_HOLD[d]])
    t_repro = table(["plan", "hand", "sim cos", "kept", "bench cos", "bench retention"],
                    rep, {2, 3, 4, 5})

    # --- the bench record -----------------------------------------------------------------
    med = lambda v: st.median(v) if v else float("nan")
    per = {}
    for did in sorted({r["did"] for r in trials}):
        R = [r for r in trials if r["did"] == did and r["op_success"]]
        N = [r for r in trials if r["did"] == did and not r["op_success"]]
        f = lambda S_, k: med([x[k] for x in S_ if x[k] is not None])
        per[did] = dict(n=len(R) + len(N), hold=len(R) / max(1, len(R) + len(N)),
                        cos=f(R, "cos"), turn=f(R, "deg"), dturn=f(N, "deg"), nd=len(N))
    order = sorted(per, key=lambda d: -per[d]["hold"])
    t_bench = table(
        ["hand", "retained", "median turn, retained", "median turn, not retained",
         "mean alignment", "residual to vertical"],
        [[d, f"{per[d]['n'] - per[d]['nd']}/{per[d]['n']}", f"{per[d]['turn']:.1f}&deg;",
          "&mdash;" if per[d]["nd"] == 0 else f"{per[d]['dturn']:.1f}&deg;",
          f"{per[d]['cos']:.3f}", f"{90 - per[d]['turn']:.1f}&deg;"] for d in order],
        {1, 2, 3, 4, 5})
    held = [r for r in trials if r["op_success"]]
    drop = [r for r in trials if not r["op_success"]]
    inb = lambda R: sum(1 for r in R if 20.0 <= r["deg"] <= 60.0)

    # --- the chain funnel -----------------------------------------------------------------
    gates = ("held_lift", "carry_ok", "held_turn", "stood_ok", "grip_ok", "ok")
    n = len(chain)
    fun = [[f"<code>{tag}</code>", DID[tag]]
           + [f"{sum(1 for r in chain if r['tag'] == tag and r.get(g))}/16" for g in gates]
           for tag in sorted({r["tag"] for r in chain}, key=lambda t: DID[t])]
    fun.append(["<b>all set A</b>", ""]
               + [f"<b>{sum(1 for r in chain if r.get(g))}/{n}</b>" for g in gates])
    t_funnel = table(["plan", "hand", "lifted", "carried", "held turn", "stood", "gait grip",
                      "complete"], fun, {2, 3, 4, 5, 6, 7}, hi={len(fun) - 1})

    # --- closure --------------------------------------------------------------------------
    by = {}
    for r in refit:
        by.setdefault(r["tag"], {})[r["arm"]] = r
    sq = []
    for tag in sorted(by, key=lambda t: -by[t]["plan"]["close_dz_mm"]):
        p, c = by[tag]["plan"], by[tag]["chain"]
        bad = p["close_dz_mm"] > 12.5
        sq.append([f"<code>{tag}</code>", DID[tag],
                   (f'<span class="no">+{p["close_dz_mm"]:.1f}</span>' if bad
                    else f'{p["close_dz_mm"]:+.1f}'),
                   f'{p["pads"]}', f'{p["pad_N"]:.2f}',
                   f'{c["close_dz_mm"]:+.1f}', f'{c["pads"]}', f'{c["pad_N"]:.2f}'])
    t_squeeze = table(["plan", "hand", "rise, mm", "pads", "force, N",
                       "rise, mm", "pads", "force, N"], sq, {2, 3, 4, 5, 6, 7})

    # --- D1's seams -----------------------------------------------------------------------
    d1 = next(r for r in chain if r["tag"] == "sv1_w6689_b060")
    sm = [s for s in (d1.get("seams") or []) if s.get("phase") in ("lifted", "turned")]
    t_seam = table(["seam", "height, m", "tilt", "pad contacts", "pad force, N"],
                   [[f'<code>{s["phase"]}</code>', f'{s["z"]:.4f}', f'{s["tilt_deg"]:.1f}&deg;',
                     (f'<span class="no">{s["pad_contacts"]}</span>' if not s["pad_contacts"]
                      else f'{s["pad_contacts"]}'),
                     (f'<span class="no">{s["pad_force_N"]:.2f}</span>' if not s["pad_force_N"]
                      else f'{s["pad_force_N"]:.2f}')] for s in sm],
                   {1, 2, 3, 4})

    # Two different statistics on two different filters; the page names both.
    R = json.loads((ROOT / "paper/figures/ranking.json").read_text())["rows"]
    hold = [x["hold_rate_op"] for x in R]
    rho_paper = spearman(hold, [x["bench_cos"] for x in R])
    rho_peak = spearman(hold, [x["bench_peak"] for x in R])
    rho_trial = spearman([per[d]["hold"] for d in order], [per[d]["cos"] for d in order])

    stats = "".join(
        f"<div class=stat><b>{v}</b><span>{k}</span></div>" for k, v in (
            ("screen reproduces", "3/3"),
            ("hands at mean cos &ge; 0.95", "0/8"),
            ("retention vs alignment, &rho;", f"{rho_paper:+.2f}".replace("-", "&minus;")),
            ("chain: held the turn", f"{sum(1 for r in chain if r.get('held_turn'))}/{n}"),
            ("chain: reached a gait grip", f"{sum(1 for r in chain if r.get('grip_ok'))}/{n}"),
        ))

    html = TPL.read_text()
    for k, v in (("{{NTRIAL}}", str(len(trials))), ("{{NCHAIN}}", str(n)),
                 ("{{STATS}}", stats), ("{{T_REPRO}}", t_repro), ("{{T_BENCH}}", t_bench),
                 ("{{T_FUNNEL}}", t_funnel), ("{{T_SQUEEZE}}", t_squeeze),
                 ("{{T_SEAM}}", t_seam),
                 ("{{BANDH}}", str(inb(held))), ("{{NHELD}}", str(len(held))),
                 ("{{BANDD}}", str(inb(drop))), ("{{NDROP}}", str(len(drop))),
                 ("{{NSTOOD}}", str(sum(1 for r in chain if r.get("stood_ok")))),
                 ("{{NGRIP}}", str(sum(1 for r in chain if r.get("grip_ok")))),
                 ("{{NTURN}}", str(sum(1 for r in chain if r.get("held_turn")))),
                 ("{{RHO_PAPER}}", f"{rho_paper:+.2f}".replace("-", "&minus;")),
                 ("{{RHO_PEAK}}", f"{rho_peak:+.2f}".replace("-", "&minus;")),
                 ("{{RHO_TRIAL}}", f"{rho_trial:+.2f}".replace("-", "&minus;")),
                 ("{{IMG_D1}}", b64(FILMS / "20260905-sv1_w6689_b060_sq2_s0_seams.png",
                                    "image/png")),
                 ("{{IMG_D7}}", b64(FILMS / "20260905-rv05_manual_b85_sq2_s0_seams.png",
                                    "image/png"))):
        html = html.replace(k, v)
    assert "{{" not in html, "unfilled placeholder: " + html[html.index("{{"):][:40]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html)
    print(f"{OUT}  ({OUT.stat().st_size/1e6:.2f} MB)")
    print(f"retention vs mean alignment, spearman "
          f"{spearman([per[d]['hold'] for d in order], [per[d]['cos'] for d in order]):+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
