#!/usr/bin/env python3
"""Result page of 2026-11-13: Steel (Skild AI's PGS solver in Newton) and Newton's hydroelastic fingertip contact
against MuJoCo-Warp with the sphere pads on the twist of the inner-loop benchmark (owner: "if steel has hydroelastic
contact built in, and it is that much faster than warp, then i think we do have to compare to them").

    data   docs/experiments/20261113-steel_solver/data/20261113-steel_solver.json from the folder's rows/
    figs   the matplotlib figures into media/
    page   docs/experiments/20261113-steel_solver/20261113-steel_solver_vs_sphere_pads.html

    .venv/bin/python scripts/contact/steel_solver_page.py data figs page
The rows come from scripts/contact/steel_hand_compare.py, the film from scripts/contact/steel_hand_film.py.
"""
from __future__ import annotations

import base64
import html
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "docs/experiments/20261113-steel_solver"
ROWS, DATA, MEDIA = D / "rows", D / "data", D / "media"
PAGE = D / "20261113-steel_solver_vs_sphere_pads.html"
DJ = DATA / "20261113-steel_solver.json"
TEMPLATE = ROOT / "scripts/ur5/ur5_tip_calibration_page.template.html"
TEX_CACHE = D / "texsvg_cache.json"
PAIRS = ROOT / "docs/experiments/20261112-inner_loop_benchmark/rows/mpc_pairs_twist.json"

# plant key, label, solver, fingertip contact
PLANTS = [
    ("mjc_pads", "MuJoCo, sphere pads", "MuJoCo 3.6 (CPU)", "1,018 sphere pads per tip, 2.5 MPa (the benchmark's model)"),
    ("mjc_pt", "MuJoCo, point contact", "MuJoCo 3.6 (CPU)", "point contact on the convex hull of the v4b tip"),
    ("steel_hydro", "Steel, compliant hydroelastic", "SolverSteel, contact_compliance", "hydroelastic tips and sphere, "
     "contacts as implicit spring-damper rows"),
    ("steel_hydro_hard", "Steel, hard hydroelastic rows", "SolverSteel", "hard rows at the hydroelastic contact points"),
    ("steel_pt", "Steel, point contact", "SolverSteel", "point contact, Steel's friction patches"),
    ("steel_pt_pf", "Steel, point contact and point friction", "SolverSteel, friction_anchor_beta 0",
     "point contact, friction per contact point"),
    ("nt_mjc_hydro", "Newton-MuJoCo, hydroelastic", "Newton's SolverMuJoCo (MuJoCo-Warp 3.17)",
     "hydroelastic tips and sphere"),
]
LABEL = {k: lab for k, lab, _s, _c in PLANTS}
SPEED = [("mjw_pads", "MuJoCo-Warp, sphere pads"), ("mjw_pt", "MuJoCo-Warp, point contact"),
         ("steel_pt", LABEL["steel_pt"]), ("steel_pt_pf", LABEL["steel_pt_pf"]),
         ("steel_hydro", LABEL["steel_hydro"]), ("steel_hydro_hard", LABEL["steel_hydro_hard"]),
         ("nt_mjc_hydro", LABEL["nt_mjc_hydro"])]
NW = [16, 64, 256, 1024, 2048]
NEWTON = ["steel_hydro", "steel_hydro_hard", "steel_pt", "steel_pt_pf", "nt_mjc_hydro"]


def rows(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def last_by_key(p):
    out = {}
    for r in rows(p):
        out[(r["pair"], r["seed"])] = r
    return out


# ------------------------------------------------------------------ data
def data():
    from sr2hand.numerics.stats import bootstrap_ci
    DATA.mkdir(parents=True, exist_ok=True)
    pairs = [f"x{p['x']:g}_y{p['y']:g}_g{p['grasp']}" for p in json.loads(PAIRS.read_text())["pairs"]]
    keys = [(p, s) for p in pairs for s in (0, 1)]

    def hom(tag):
        f = ROWS / f"hom_{tag}.jsonl"
        if not f.exists():
            return None
        d = last_by_key(f)
        # a rollout stopped by HOM's least squares had dropped the sphere (all 11 checked, 2026-11-13): J 0, not held
        J = {k: (d[k]["J"] if d.get(k, {}).get("J") is not None else 0.0) for k in keys}
        held = {k: bool(d.get(k, {}).get("held")) if d.get(k, {}).get("J") is not None else False for k in keys}
        stopped = sum(1 for k in keys if d.get(k, {}).get("J") is None)
        wall = [d[k].get("batch_wall_s") or d[k].get("wall_s") for k in keys if d.get(k, {}).get("J") is not None]
        return {"J": J, "held": held, "stopped": stopped, "wall_s": float(np.median([w for w in wall if w])) if wall else None}

    pads = hom("mjc_pads")
    out = {"pairs": pairs, "hom": {}, "speed": {}, "speed_bench_filter": {}}
    for tag in ["mjc_pads", "mjc_pt"] + NEWTON + [f"{p}_sap" for p in NEWTON]:
        h = hom(tag)
        if h is None:
            continue
        Jv = np.array([h["J"][k] for k in keys])
        diff_pair = np.array([np.mean([h["J"][(p, s)] - pads["J"][(p, s)] for s in (0, 1)]) for p in pairs])
        out["hom"][tag] = {"mean_J": float(Jv.mean()), "median_J": float(np.median(Jv)),
                           "seed_means": [float(np.mean([h["J"][(p, s)] for p in pairs])) for s in (0, 1)],
                           "held": int(sum(h["held"].values())), "n": len(keys), "stopped": h["stopped"],
                           "diff": float(diff_pair.mean()),
                           "diff_ci": bootstrap_ci(diff_pair) if tag != "mjc_pads" else None,
                           "per_pair_diff": diff_pair.round(4).tolist(),
                           "J": {f"{p}|{s}": h["J"][(p, s)] for p, s in keys}, "wall_s": h["wall_s"]}
    sp = rows(ROWS / "speed.jsonl")
    for r in sp:
        if r.get("status") != "ok":
            continue
        if r.get("pairs") == "obj":
            k = "speed"
        elif r["plant"].startswith("mjw_"):
            k = "speed_bench_filter"                     # stage 1: the benchmark's own filter (hom_gpu.Batch)
        else:
            continue                                     # stage 1's Newton rows: sweep and prune over all shapes
        cur = out[k].setdefault(r["plant"], {})
        # the hydroelastic rows above 64 worlds count only with stage 3's broad-phase buffer (0.02, 60)
        if r.get("hydro_buf") == [0.02, 40] and "hydro" in r["plant"]:
            out.setdefault("speed_overflowed", {}).setdefault(r["plant"], {})[str(r["nworld"])] = r
            continue
        cur[str(r["nworld"])] = {x: r.get(x) for x in ("us_per_world_step", "contacts_per_world", "held_frac",
                                                        "drift_mm_median", "gpu_used_mb", "host_rss_mb", "build_s",
                                                        "pair", "hydro_buf", "when")}
    films = sorted(ROWS.glob("film_twist_p*_s*.json"))
    out["films"] = {f.stem: json.loads(f.read_text()) for f in films}
    DJ.write_text(json.dumps(out, indent=1))
    print(DJ, f"{DJ.stat().st_size / 1e3:.0f} kB", sorted(out["hom"]), sorted(out["speed"]))


# ------------------------------------------------------------------ figures
def figs():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 13, "font.family": "serif"})
    MEDIA.mkdir(parents=True, exist_ok=True)
    d = json.loads(DJ.read_text())
    cols = {"mjw_pads": "k", "mjw_pt": "0.5", "steel_pt": "tab:blue", "steel_pt_pf": "tab:cyan",
            "steel_hydro": "tab:red", "steel_hydro_hard": "tab:orange", "nt_mjc_hydro": "tab:green",
            "mjc_pt": "0.5", "mjc_pads": "k"}
    # throughput against worlds
    fig, ax = plt.subplots(figsize=(10, 5.6))
    for k, lab in SPEED:
        s = d["speed"].get(k, {})
        x = [n for n in NW if str(n) in s]
        if not x:
            continue
        y = [s[str(n)]["us_per_world_step"] for n in x]
        ls = "-" if k.startswith("mjw") else "--"
        ax.plot(x, y, ls, marker="o", color=cols[k], lw=2, label=lab)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(NW, [str(n) for n in NW])
    ax.set_xlabel("worlds in the batch")
    ax.set_ylabel("GPU time per world and step (µs)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=11, loc="upper right")
    fig.tight_layout()
    fig.savefig(MEDIA / "20261113-throughput_vs_worlds.png", dpi=110)
    plt.close(fig)
    # paired J differences per pair
    order = [k for k in ["mjc_pt", "steel_hydro", "steel_pt", "nt_mjc_hydro", "steel_hydro_hard", "steel_pt_pf"]
             if k in d["hom"]]
    fig, ax = plt.subplots(figsize=(10, 0.62 * len(order) + 1.6))
    rng = np.random.default_rng(0)
    for i, k in enumerate(order):
        h = d["hom"][k]
        y = np.array(h["per_pair_diff"])
        ax.scatter(y, i + rng.uniform(-0.18, 0.18, len(y)), s=22, color=cols[k], alpha=0.7)
        lo, hi = h["diff_ci"]
        ax.plot([lo, hi], [i, i], color="k", lw=3)
        ax.plot([h["diff"]], [i], "o", color="w", mec="k", ms=9, mew=2)
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(range(len(order)), [LABEL[k] for k in order])
    ax.invert_yaxis()
    ax.set_xlabel("ΔJ: J on the plant minus J on the sphere pads,\nper pair (mean of seeds 0 and 1)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(MEDIA / "20261113-paired_J_difference.png", dpi=110)
    plt.close(fig)
    print("figures written to", MEDIA)


# ------------------------------------------------------------------ page helpers
def esc(s):
    return html.escape(str(s), quote=True)


def img_uri(p, width=1100, quality=86):
    from PIL import Image
    im = Image.open(p).convert("RGB")
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    b = io.BytesIO()
    im.save(b, "JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def fig(n, p, caption, width=1100):
    return (f'<figure id="fig{n}"><img src="{img_uri(p, width)}" alt="{esc(caption.split(".")[0])}">'
            f"<figcaption>Figure {n}. {caption}</figcaption></figure>")


def film_html(n, mp4, poster, caption):
    v = "data:video/mp4;base64," + base64.b64encode(Path(mp4).read_bytes()).decode()
    return (f'<figure id="fig{n}"><video controls muted loop playsinline autoplay preload="metadata" '
            f'poster="{img_uri(poster, 1400)}" src="{v}" style="width:100%"></video>'
            f"<figcaption>Figure {n}. {caption}</figcaption></figure>")


def table(head, body, num=(), caption=None, n=None):
    cap = f"<caption>Table {n}. {caption}</caption>" if caption else ""
    th = "".join(f'<th{" class=n" if k in num else ""}>{h}</th>' for k, h in enumerate(head))
    tr = "".join("<tr>" + "".join(f'<td{" class=n" if k in num else ""}>{c}</td>' for k, c in enumerate(r)) + "</tr>"
                 for r in body)
    return f'<div class="tw"><table>{cap}<thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


if __name__ == "__main__":
    from steel_solver_page_text import page
    for cmd in sys.argv[1:] or ["data", "figs", "page"]:
        {"data": data, "figs": figs, "page": page}[cmd]()
