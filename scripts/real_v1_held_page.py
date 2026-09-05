"""Build the table-stand result page from the study's own JSON.

    uv run python scripts/real_v1_held_page.py

Reads docs/experiments/20260904-real_v1_held/chain_hands.json plus the hero video and its seam
filmstrip, and writes 20260904-real_v1_table_stand.html beside them. The media are inlined as
data URIs because the published artifact is a single file.
"""
from __future__ import annotations

import base64
import json
import statistics as stx
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "docs/experiments/20260904-real_v1_held"
OUT = DIR / "20260904-real_v1_table_stand.html"


def b64(p: Path, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode()


def rows_table(D):
    rows, H = D["rows"], D["hands"]
    out = []
    for h in H:
        g = [r for r in rows if r.get("tag") == h["tag"]]
        if not g:
            continue
        sd = [r for r in g if r.get("stood_ok")]
        m = lambda k, src: (stx.mean([float(x.get(k) or 0.0) for x in src]) if src else None)
        out.append(dict(tag=h["tag"], set=h["set"], n=len(g),
                        lift=sum(1 for r in g if r.get("held_lift")),
                        stood=len(sd), ok=sum(1 for r in g if r.get("ok")),
                        up=m("tilt_upright_deg", sd), end=m("final_tilt_deg", sd),
                        deg=m("gain_mean_deg", sd), roll=m("roll_max_deg", g),
                        free=m("free_frac", g)))
    out.sort(key=lambda r: (-r["ok"], -r["stood"], r["tag"]))
    return out


def fmt(v, n=2):
    return "&mdash;" if v is None else f"{v:.{n}f}"


def main() -> int:
    D = json.loads((DIR / "chain_hands.json").read_text())
    T = rows_table(D)
    rows, n = D["rows"], len(D["rows"])
    tot = dict(ok=sum(1 for r in rows if r.get("ok")),
               stood=sum(1 for r in rows if r.get("stood_ok")),
               lift=sum(1 for r in rows if r.get("held_lift")), n=n)
    body = []
    for r in T:
        cls = "ok" if r["ok"] else ("part" if r["stood"] else "bad")
        body.append(
            f'<tr class="{cls}"><td class="tag">{r["tag"]}</td><td class="s">{r["set"]}</td>'
            f'<td class="num">{r["lift"]}/{r["n"]}</td><td class="num">{r["stood"]}/{r["n"]}</td>'
            f'<td class="num">{fmt(r["up"])}</td><td class="num strong">{r["ok"]}/{r["n"]}</td>'
            f'<td class="num">{fmt(r["deg"], 1)}</td><td class="num">{fmt(r["end"])}</td>'
            f'<td class="num">{fmt(r["roll"], 1)}</td><td class="num">{fmt(r["free"])}</td></tr>')
    html = TEMPLATE.replace("{{ROWS}}", "\n".join(body)) \
        .replace("{{OK}}", str(tot["ok"])).replace("{{N}}", str(tot["n"])) \
        .replace("{{STOOD}}", str(tot["stood"])).replace("{{LIFT}}", str(tot["lift"])) \
        .replace("{{VIDEO}}", b64(DIR / "20260904-table_stand_u1364.mp4", "video/mp4")) \
        .replace("{{FILM}}", b64(DIR / "20260904-table_stand_u1364_seams.png", "image/png"))
    OUT.write_text(html)
    print(f"-> {OUT}  ({OUT.stat().st_size/1e6:.1f} MB)")
    return 0


TEMPLATE = (Path(__file__).resolve().parent / "real_v1_held_page.template.html").read_text()

if __name__ == "__main__":
    raise SystemExit(main())
