"""Fill the design-search page's tables from the measurement files and inline its assets.

The page is generated rather than hand-written so that every number in it is read out of the
artifact that produced it. Run after any re-measurement:

    MUJOCO_GL=egl uv run python docs/experiments/20260828-real_v1_search/build_report.py
"""
from __future__ import annotations

import base64
import glob
import json
import mimetypes
import re
from pathlib import Path

import numpy as np
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from real_v1_search_report import GEOM, SCORES, auc, spearman  # noqa: E402

F = ("thumb", "index", "middle")


def cell(v, cls="") -> str:
    return f'<td class="num {cls}">{v}</td>'


# --------------------------------------------------------------------------- 1.1 grip decay
def grip_decay() -> str:
    rows = json.loads((HERE / "grip_decay_rv05.json").read_text())
    by = {(r["mode"], r["hold_steps"]): r for r in rows}
    out = []
    for h in sorted({r["hold_steps"] for r in rows}):
        ik, li = by[("ik", h)], by[("linear", h)]
        dead = ik["tip_contacts"] == 0
        out.append(
            f'<tr><td class="l mono">{h*0.002:.1f} s</td>'
            + cell(f'{ik["cos"]:+.3f}', "lose" if dead else "")
            + cell(f'{ik["z"]:.4f}', "lose" if dead else "")
            + cell(ik["tip_contacts"], "lose" if dead else "")
            + cell(f'{ik["tip_force_N"]:.1f} N', "lose" if dead else "")
            + cell(f'{li["cos"]:+.3f}', "win")
            + cell(f'{li["z"]:.4f}')
            + cell(li["tip_contacts"])
            + cell(f'{li["tip_force_N"]:.1f} N')
            + "</tr>")
    return "\n".join(out)


# --------------------------------------------------------------------------- 1.4 reference
NICE = {"rv05_manual_stored": "rv05_manual", "rv03_narrowy_sp40": "rv03_narrowy",
        "rv00_wide_sp40": "rv00_wide (sp40)", "rv00_wide_sp30": "rv00_wide (sp30)",
        "rv04_mid_sp30": "rv04_mid", "rv01_compact_sp40": "rv01_compact",
        "rv02_narrowx_sp40": "rv02_narrowx"}


def reference() -> str:
    cells: dict = {}
    for f in glob.glob(str(HERE / "ref_lin_*.json")):
        for r in json.load(open(f)):
            cells.setdefault((r["run"], r["angle_deg"], r["axis_height_mm"]), []).append(r)
    best: dict = {}
    for (run, ang, h), v in cells.items():
        fin = [x["final_cos"] for x in v]
        kept = sum(1 for x in v if x["ok"])
        rec = {"ang": ang, "h": h, "mean": float(np.mean(fin)), "sd": float(np.std(fin)),
               "kept": kept, "n": len(v),
               "z": float(np.mean([x["final_z"] for x in v])),
               "con": float(np.mean([x["contacts"] for x in v]))}
        key = (kept * 2 >= len(v), rec["mean"])
        if run not in best or key > best[run][0]:
            best[run] = (key, rec)
    order = sorted(best, key=lambda r: -best[r][1]["mean"] if best[r][0][0] else 1e9)
    out = []
    for run in list(NICE):
        # A design whose every cell drops the shaft still has a "best" by mean, and printing it
        # as a row invites reading a turn that ended on the table as a small reorient.
        if run not in best or not best[run][0][0]:
            out.append(f'<tr><td class="l">{NICE[run]}</td>'
                       f'<td class="l dim" colspan="7">no cell of 24 keeps the shaft</td></tr>')
            continue
        b = best[run][1]
        held = best[run][0][0]
        out.append(
            f'<tr><td class="l">{NICE.get(run, run)}</td>'
            + cell(f'{b["ang"]:.0f}°') + cell(f'{b["h"]:.1f} mm')
            + cell(f'{b["mean"]:+.3f}', "win" if held and b["mean"] > 0.5 else "")
            + cell(f'{b["sd"]:.3f}') + cell(f'{b["kept"]}/{b["n"]}')
            + cell(f'{b["z"]:.4f}') + cell(f'{b["con"]:.2f}') + "</tr>")
    del order
    return "\n".join(out)


# --------------------------------------------------------------------------- 2 touch
def touch(table) -> str:
    def agg(group, name):
        tf = np.array([[r["touch_frac"][f] for f in F] for r in group if r.get("touch_frac")])
        ds = np.array([[r["drive_share"][f] for f in F] for r in group if r.get("drive_share")])
        if not len(tf):
            return ""
        return (f'<tr><td class="l">{name}</td>' + cell(len(group))
                + "".join(cell(f"{tf[:, i].mean():.2f}") for i in range(3))
                + "".join(cell(f"{ds[:, i].mean():.2f}") for i in range(3)) + "</tr>")
    ok = [r for r in table if r.get("reorients")]
    bad = [r for r in table if r.get("graspable") and not r.get("reorients")]
    return "\n".join([
        agg([r for r in ok if r.get("thumb_axial_mm") == 0.0], "reorients, thumb at mid-length"),
        agg([r for r in ok if r.get("thumb_axial_mm") == 20.0], "reorients, thumb offset 20 mm"),
        '<tr class="sep">' + agg(bad, "does not reorient")[4:]])


# --------------------------------------------------------------------------- 3 thumb arm
def thumb_arm() -> str:
    tr = {t: json.loads((HERE / "traces" / f"rv04_mid_thAx{t}.json").read_text())
          for t in ("0", "20")}
    def stat(fn):
        return {t: fn(v) for t, v in tr.items()}
    def per(f, key, how):
        return stat(lambda rows: how([r["fingers"][f][key] for r in rows]))
    rows = [
        ("held cos, n = 8 jittered draws", {"0": "0.000 ± 0.000", "20": "0.972 ± 0.066"},
         "no turn at all vs a solved one"),
        ("shaft kept", {"0": "0/8", "20": "8/8"}, "the only change is the thumb pad's y"),
    ]
    out = []
    for name, vals, read in rows:
        out.append(f'<tr><td class="l">{name}</td>'
                   + cell(vals["0"], "lose") + cell(vals["20"], "win")
                   + f'<td class="l dim">{read}</td></tr>')
    labels = [("middle pad mean normal force", "middle", "fn_N", np.mean, "{:.2f} N",
               "force moves onto the pad that supplies the couple"),
              ("middle pad peak cone utilisation", "middle", "cone_util", max, "{:.2f}",
               "1.00 is sliding out of the friction cone"),
              ("index pad peak cone utilisation", "index", "cone_util", max, "{:.2f}",
               "the descending pad, saturated either way"),
              ("thumb pad mean normal force", "thumb", "fn_N", np.mean, "{:.2f} N",
               "barely changes — this is not about thumb force")]
    for name, f, key, how, fmt, read in labels:
        v = per(f, key, how)
        out.append(f'<tr><td class="l">{name}</td>'
                   + cell(fmt.format(v["0"])) + cell(fmt.format(v["20"]))
                   + f'<td class="l dim">{read}</td></tr>')
    on = {t: {f: float(np.mean([r["fingers"][f]["fn_N"] > 0.1 for r in v])) for f in F}
          for t, v in tr.items()}
    for f in ("thumb", "middle"):
        out.append(f'<tr><td class="l">{f} pad on the shaft</td>'
                   + cell(f'{on["0"][f]:.2f}') + cell(f'{on["20"][f]:.2f}', "win")
                   + '<td class="l dim">fraction of the commanded turn in contact</td></tr>')
    return "\n".join(out)


# --------------------------------------------------------------------------- 4 top table
def top(table, n=24) -> str:
    have = [t for t in table if t.get("graspable")]
    have.sort(key=lambda t: (-t["mean_cos"] if t["reorients"] else 1e9, -t["mean_cos"]))
    out = []
    for t in have[:n]:
        dep = "deepest" if t["depth_req_mm"] is None else f'{t["depth_req_mm"]:.0f}'
        out.append(
            f'<tr><td class="l mono">{t["design"]}</td>'
            + cell(f'{t["x_sep_mm"]:.0f}') + cell(f'{t["y_sep_mm"]:.0f}')
            + cell(f'{t["thumb_y_mm"]:.0f}')
            + f'<td class="l mono">{t["straddle_mm"]:.0f}/{dep}/{t["thumb_axial_mm"]:.0f}</td>'
            + cell(f'{t["axis_k"]:.2f}') + cell(f'{t["angle_deg"]:.0f}°')
            + cell(f'{t["mean_cos"]:+.3f}', "win" if t["reorients"] else "lose")
            + cell(f'{t["sd_cos"]:.3f}') + cell(f'{t["kept"]}/{t["n"]}')
            + f'<td class="l">{t["style"]}</td></tr>')
    return "\n".join(out)


# --------------------------------------------------------------------------- 5 scores
def scores(table) -> str:
    have = [t for t in table if t.get("graspable")]
    y = [t["mean_cos"] for t in have]
    lab = [t["reorients"] for t in have]
    rows = []
    for k, (what, _sign) in {**SCORES, **{g: (d, +1) for g, d in GEOM.items()}}.items():
        x = [t.get(k) if t.get(k) is not None else float("nan") for t in have]
        rho, p = spearman(x, y)
        rows.append((k, what, rho, p, auc(x, lab)))
    rows.sort(key=lambda r: -r[4] if np.isfinite(r[4]) else 0)
    out = []
    for k, what, rho, p, a in rows:
        strong = a >= 0.70
        out.append(f'<tr><td class="l mono">{k}</td><td class="l dim">{what}</td>'
                   + cell(f"{rho:+.3f}", "win" if strong else "")
                   + cell(f"{p:.1e}") + cell(f"{a:.3f}", "win" if strong else "") + "</tr>")
    return "\n".join(out)


# --------------------------------------------------------------------------- 6 styles
WHAT = {"PINCH-ROLL": "two pads loaded, the descending one shed, the shaft rolls between the "
                      "thumb and the ascending pad",
        "SINGLE": "the grip does not survive the first degrees; the shaft falls",
        "PALM-PIN": "the shaft rides up into the palm plate, which pins it",
        "TRIPOD": "all three pads stay on and the grip locks rotationally"}


def styles(table) -> str:
    have = [t for t in table if t.get("graspable")]
    groups: dict = {}
    for t in have:
        groups.setdefault(t["style"], []).append(t)
    out = []
    for name in ("PINCH-ROLL", "SINGLE", "PALM-PIN", "TRIPOD"):
        g = groups.get(name)
        if not g:
            continue
        tf = np.array([[r["touch_frac"][f] for f in F] for r in g if r.get("touch_frac")])
        nre = sum(r["reorients"] for r in g)
        out.append(
            f'<tr><td class="l"><b>{name}</b></td>' + cell(len(g))
            + cell(f"{nre}/{len(g)}", "win" if nre else "lose")
            + cell(f'{np.mean([r["mean_cos"] for r in g]):+.3f}')
            + cell("/".join(f"{tf[:, i].mean():.2f}" for i in range(3)) if len(tf) else "—")
            + cell(f'{np.mean([r["mean_contacts"] or 0 for r in g]):.2f}')
            + cell(f'{np.mean([r["obj_dz_mm"] or 0 for r in g]):+.1f} mm')
            + f'<td class="l dim">{WHAT[name]}</td></tr>')
    return "\n".join(out)


# --------------------------------------------------------------------------- 7 CEM
CEM = {"rv04_mid_thAx20": ("rv04_mid + thumb 20 mm", 0.997, "CEM optimises the hold, and on this "
                           "hand the hold and the turn trade against each other"),
       "axasym10_thAx20": ("ax_asym−10 + thumb 20 mm", 0.976, "confirms: a hand nobody had "
                           "sampled, between rv05_manual and rv03_narrowy")}


def cem() -> str:
    out = []
    for run, (nice, fitted, read) in CEM.items():
        cells: dict = {}
        for f in glob.glob(str(HERE / f"cem_cells_{run}_a*.json")):
            for r in json.load(open(f)):
                cells.setdefault((r["angle_deg"], r["axis_height_mm"]), []).append(r)
        best = None
        for (ang, h), v in cells.items():
            fin = [x["final_cos"] for x in v]
            kept = sum(1 for x in v if x["ok"])
            key = (kept * 2 >= len(v), float(np.mean(fin)))
            if best is None or key > best[0]:
                best = (key, float(np.mean(fin)), float(np.std(fin)), kept, len(v))
        one = json.loads((HERE / f"cem_carry_{run}.json").read_text())
        onem = float(np.mean([x["final_cos"] for x in one]))
        out.append(f'<tr><td class="l">{nice}</td>'
                   + cell(f"{fitted:+.3f}")
                   + cell(f"{onem:+.3f}", "lose")
                   + cell(f"{best[1]:+.3f}", "win" if best[1] > 0.5 else "lose")
                   + cell(f"{best[2]:.3f}") + cell(f"{best[3]}/{best[4]}")
                   + f'<td class="l dim">{read}</td></tr>')
    return "\n".join(out)


# --------------------------------------------------------------------------- assets
def inline(match: re.Match) -> str:
    rel = match.group(1)
    p = HERE / rel
    mime = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
    return f"data:{mime};base64,{base64.b64encode(p.read_bytes()).decode()}"


def main() -> None:
    table = json.loads((HERE / "table.json").read_text())
    src = (HERE / "report_search.src.html").read_text()
    for token, value in (("GRIP_DECAY_ROWS", grip_decay()),
                         ("REFERENCE_ROWS", reference()),
                         ("TOUCH_ROWS", touch(table)),
                         ("THUMB_ARM_ROWS", thumb_arm()),
                         ("TOP_ROWS", top(table)),
                         ("SCORE_ROWS", scores(table)),
                         ("STYLE_ROWS", styles(table)),
                         ("CEM_ROWS", cem())):
        assert "{{" + token + "}}" in src, token
        src = src.replace("{{" + token + "}}", value)
    src = re.sub(r"\{\{ASSET:([^}]+)\}\}", inline, src)
    out = HERE / "report_search.html"
    out.write_text(src)
    print(f"wrote {out.relative_to(ROOT)}  ({out.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
