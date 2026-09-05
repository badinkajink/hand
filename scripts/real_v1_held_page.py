"""Build the table-stand result page from the study's own JSON.

    uv run python scripts/real_v1_held_page.py

Reads the squeeze sweep (`squeeze_sweep/chain_hands.json`), the per-hand films
(`20260905-films/films.json`) and the media beside them, and writes
20260904-real_v1_table_stand.html. Media are inlined as data URIs because the
published artifact is a single file; the per-hand clips are the `web/` re-encodes.
"""
from __future__ import annotations

import base64
import json
import statistics as stx
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "docs/experiments/20260904-real_v1_held"
FILMS = DIR / "20260905-films"
SWEEP = DIR / "squeeze_sweep/chain_hands.json"
OUT = DIR / "20260904-real_v1_table_stand.html"
SQ = (2.0, 3.0, 4.0, 6.0)


def b64(p: Path, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode()


def sq_of(r: dict) -> float:
    return float(r["arm"].split("_sq")[-1])


def seams(r: dict) -> dict:
    return {s["phase"]: s for s in r.get("seams", [])}


def reorient_deg(r: dict) -> float | None:
    u = seams(r).get("upright") or seams(r).get("staged")
    return None if u is None else 90.0 - float(u["tilt_deg"])


def upright(r: dict) -> dict:
    """The `upright` seam -- the moment the maneuver claims a vertical tool."""
    return seams(r).get("upright") or seams(r).get("staged") or {}


def held(r: dict) -> bool:
    """Is the hand still ON the tool when it reads vertical? `stood_ok` never asked."""
    u = upright(r)
    return int(u.get("pad_contacts") or 0) >= 2 and float(u.get("pad_force_N") or 0.0) > 1.0


def funnel(rows: list[dict]) -> list[dict]:
    """Where the 64 runs of each squeeze arm die, on the chain's own gate order."""
    out = []
    for s in SQ:
        g = [r for r in rows if sq_of(r) == s]
        d = dict(sq=s, n=len(g), carry=0, stand=0, hand=0, gait=0, ok=0)
        for r in g:
            if not r.get("carry_ok"):
                d["carry"] += 1
            elif not r.get("stood_ok"):
                d["stand"] += 1
            elif not r.get("grip_ok"):
                d["hand"] += 1
            elif not r.get("ok"):
                d["gait"] += 1
            else:
                d["ok"] += 1
        st = [reorient_deg(r) for r in g if r.get("stood_ok")]
        st = [v for v in st if v is not None]
        d["reo"] = stx.mean(st) if st else None
        d["reo_sd"] = stx.pstdev(st) if len(st) > 1 else 0.0
        out.append(d)
    return out


def close_probe() -> dict:
    """The close probe per hand per squeeze: what closing the grasp does to a tool on a table."""
    D = ROOT / "assets/mjcf/experimental/20260904-chain_bothsets"
    out: dict = {}
    for f in sorted(D.glob("*_fit.json")):
        r = json.loads(f.read_text())
        if "close_dz_mm" not in r:
            continue
        out.setdefault(r["tag"].split("_sq")[0], {})[r.get("squeeze_mm")] = r
    return out


def per_hand(rows: list[dict], cp: dict, films: list[dict]) -> list[dict]:
    fv = {f["tag"]: f for f in films}
    by: dict[str, dict] = {}
    for r in rows:
        by.setdefault(r["tag"], {}).setdefault(sq_of(r), []).append(r)
    out = []
    for tag, per in by.items():
        cells = {}
        for s in SQ:
            g = per.get(s, [])
            cells[s] = dict(n=len(g), carry=sum(1 for r in g if r.get("carry_ok")),
                            stood=sum(1 for r in g if r.get("stood_ok")),
                            ok=sum(1 for r in g if r.get("ok")))
        best = max(SQ, key=lambda s: (cells[s]["ok"], cells[s]["stood"], -s))
        g = per.get(best, [])
        reo = [reorient_deg(r) for r in g if r.get("stood_ok")]
        reo = [v for v in reo if v is not None]
        c = (cp.get(tag) or {}).get(best, {})
        f = fv.get(tag, {})
        allg = [r for s_ in SQ for r in per.get(s_, [])]
        pads = [int(upright(r).get("pad_contacts") or 0) for r in allg if r.get("stood_ok")]
        out.append(dict(tag=tag, set=g[0]["set"] if g else "?", best=best, cells=cells,
                        ok=cells[best]["ok"], n=cells[best]["n"],
                        carry=cells[best]["carry"], stood=cells[best]["stood"],
                        pads=(max(pads) if pads else None), nstood=len(pads),
                        heldn=sum(1 for r in allg if r.get("stood_ok") and held(r)),
                        reo=stx.mean(reo) if reo else None,
                        dz=c.get("close_dz_mm"), nonpad=c.get("close_nonpad"),
                        force=c.get("close_pad_N"),
                        drop=f.get("drop_stage"), video=f.get("video")))
    out.sort(key=lambda r: (-r["ok"], -r["stood"], r["tag"]))
    return out


def fmt(v, n=2, dash="&mdash;"):
    return dash if v is None else f"{v:.{n}f}"


def main() -> int:
    D = json.loads(SWEEP.read_text())
    rows = D["rows"]
    films = json.loads((FILMS / "films.json").read_text())
    cp = close_probe()
    F = funnel(rows)
    T = per_hand(rows, cp, films)

    fun = "\n".join(
        f'<tr><td class="num">{d["sq"]:.0f}</td><td class="num bad">{d["carry"]}</td>'
        f'<td class="num bad">{d["stand"]}</td><td class="num warn">{d["hand"]}</td>'
        f'<td class="num">{d["gait"]}</td><td class="num strong ok">{d["ok"]}</td>'
        f'<td class="num">{fmt(d["reo"], 1)} &plusmn; {fmt(d["reo_sd"], 1)}</td></tr>'
        for d in F)

    clip = {}
    for r in T:
        if not r["video"]:
            continue
        w = FILMS / "web" / Path(r["video"]).name
        if w.exists():
            clip[r["tag"]] = b64(w, "video/mp4")

    body = []
    for r in T:
        cls = "ok" if r["ok"] == r["n"] else ("part" if r["stood"] else "bad")
        cells = " ".join(
            f'<span class="chip{" hi" if s == r["best"] else ""}">{r["cells"][s]["ok"]}</span>'
            for s in SQ)
        v = clip.get(r["tag"])
        vid = (f'<video src="{v}" controls muted loop playsinline preload="none"></video>'
               if v else "")
        body.append(
            f'<tr class="{cls}"><td class="tag">{r["tag"]}<div class="clip">{vid}</div></td>'
            f'<td class="s">{r["set"]}</td><td class="num">{r["best"]:.0f}</td>'
            f'<td class="num">{fmt(r["dz"], 1)}</td><td class="num">{fmt(r["force"], 1)}</td>'
            f'<td class="num">{r["nonpad"] if r["nonpad"] is not None else "&mdash;"}</td>'
            f'<td class="num">{r["carry"]}/{r["n"]}</td><td class="num">{r["stood"]}/{r["n"]}</td>'
            f'<td class="num">{r["pads"] if r["pads"] is not None else "&mdash;"}</td>'
            f'<td class="num strong">{r["heldn"]}/16</td>'
            f'<td class="num">{r["ok"]}/{r["n"]}</td><td class="cells">{cells}</td></tr>')

    st_all = [r for r in rows if r.get("stood_ok")]
    padc = {k: sum(1 for r in st_all if int(upright(r).get("pad_contacts") or 0) == k)
            for k in (0, 1, 2)}
    n_held = sum(1 for r in st_all if held(r))
    n_below = sum(1 for r in st_all
                  if (seams(r).get("gait_grip", {}).get("z") or 1.0) < 0.02)
    tot = {s: sum(1 for r in rows if sq_of(r) == s and r.get("ok")) for s in SQ}
    best_tot = sum(r["ok"] for r in T)
    n_arm = sum(1 for r in rows if sq_of(r) == 2.0)

    html = TEMPLATE
    for k, v in {
        "{{FUNNEL}}": fun, "{{ROWS}}": "\n".join(body),
        "{{N}}": str(n_arm), "{{OK2}}": str(tot[2.0]), "{{OK4}}": str(tot[4.0]),
        "{{BEST}}": str(best_tot),
        "{{REO}}": fmt(F[0]["reo"], 1), "{{REOSD}}": fmt(F[0]["reo_sd"], 1),
        "{{NALL}}": str(len(rows)), "{{NSTOOD}}": str(len(st_all)),
        "{{P0}}": str(padc[0]), "{{P1}}": str(padc[1]), "{{P2}}": str(padc[2]),
        "{{HELD}}": str(n_held), "{{BELOW}}": str(n_below),
        "{{STOOD2}}": str(sum(1 for r in rows if sq_of(r) == 2.0 and r.get("stood_ok"))),
        "{{GRID}}": b64(FILMS / "20260905-chain_grid.mp4", "video/mp4"),
        "{{FILM_OK}}": b64(FILMS / "20260905-sv1_u1364_b080_sq2_s0_seams.png", "image/png"),
        "{{FILM_HAND}}": b64(FILMS / "20260905-sv1_w2360_b075_sq3_s0_seams.png", "image/png"),
        "{{FILM_DROP}}": b64(FILMS / "20260905-sv1_w6689_b050_sq2_s0_seams.png", "image/png"),
    }.items():
        html = html.replace(k, v)
    OUT.write_text(html)
    print(f"-> {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB)")
    return 0


TEMPLATE = (Path(__file__).resolve().parent / "real_v1_held_page.template.html").read_text()

if __name__ == "__main__":
    raise SystemExit(main())
