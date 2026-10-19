#!/usr/bin/env python3
"""Print procedure for the Siraya Tech 85A magnet fingertips on the Prusa XL (docs/handoff/20261018-print_procedure.md,
step 2), read at the printer on a phone.

The page is built from the files of the 2026-10-16 85A print set and regenerates when they change: the per-project sheets
and README of docs/experiments/20261016-sim2real_fingertip_readiness/print/ (geometry, pauses, slice checks, print time and
filament), template_changes.json (the Siraya profile in slot 2), the readiness study's bench/procedure.json (expected tip
masses), the figures of scripts/fingertip_print_procedure_figures.py (renders and the seat-wall check) and the owner's
records in records/seat_coupon.csv and records/prints.csv, which this builder creates with blank measurement columns and
never overwrites. When the seat coupon's rows name a seat, the page names it and uses the sheets built with it where they
exist.

    .venv/bin/python scripts/fingertip_print_procedure_page.py
"""
from __future__ import annotations

import base64
import csv
import html
import io
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import retro_style  # noqa: E402

READY = ROOT / "docs/experiments/20261016-sim2real_fingertip_readiness"
PRINT = READY / "print"
D = ROOT / "docs/experiments/20261018-fingertip_print_procedure"
PAGE = D / "20261018-siraya85a_fingertip_print_procedure.html"
MEDIA = D / "media"
RECORDS = D / "records"
TEMPLATE = ROOT / "scripts/fingertip_print_procedure_page.template.html"
CURVED = ["tcd_cap8.5", "tcd_ell7.4x8.5x16x32", "tcd_ell7.4x8.5x11x22"]
SEAT_CSV = RECORDS / "seat_coupon.csv"
PRINTS_CSV = RECORDS / "prints.csv"
SEAT_COLS = ["coupon", "project", "seat_kind", "seat_mm", "takes_to_floor", "wall_split", "keeps_when_tapped", "notes"]
PRINT_COLS = ["order", "print", "project", "printed_on", "seat_mm", "magnets_seated", "polarity_checked", "tip_mass_g",
              "expected_mass_g", "defects", "notes"]
OWNER = '<span class="fill">&#160;</span> (owner&#8217;s setting)'


def rel(p):
    return str(Path(p).relative_to(ROOT)) if str(p).startswith(str(ROOT)) else str(p)


def esc(s):
    return html.escape(str(s), quote=True)


def minutes(txt):
    m = 0.0
    for part in txt.split():
        m += float(part[:-1]) * {"d": 1440, "h": 60, "m": 1, "s": 1 / 60}[part[-1]]
    return m


def fmt_min(m):
    m = round(m)
    return f"{m // 60} h {m % 60} min" if m >= 60 else f"{m} min"


def num(v, nd=2):
    return f"{v:.{nd}f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v)


# ------------------------------------------------------------------------------------------------- data

def sheets():
    out = []
    for f in sorted(PRINT.glob("20261016-siraya85a_*.json")):
        if f.name != "template_changes.json":
            out.append(json.loads(f.read_text()))
    return out


def is_coupon(r):
    return "tip" not in r


def walls(r):
    return r.get("pocket_walls") or 2


def label(r):
    if is_coupon(r):
        return "Seat coupon, " + ("two walls" if walls(r) == 2 else "one wall") + " at each seat"
    lab = r["tip_label"][0].upper() + r["tip_label"][1:]
    lab += ", five magnets" if r["layout"].startswith("five") else ", four magnets"
    if walls(r) == 1:
        lab += ", one wall at the pockets"
    return f"{lab}, {r['infill_percent']} % gyroid"


def short_label(r):
    if is_coupon(r):
        return "Coupon, " + ("two walls" if walls(r) == 2 else "one wall")
    lab = {"block": "Block", "tcd_cap8.5": "Capsule r 8.5", "tcd_ell7.4x8.5x16x32": "Half-ellipsoid 16/32",
           "tcd_ell7.4x8.5x11x22": "Half-ellipsoid 11/22"}[r["tip"]]
    lab += ", five magnets" if r["layout"].startswith("five") else ""
    lab += ", one wall" if walls(r) == 1 else ""
    return f"{lab}, {r['infill_percent']} %"


def mass_key(r):
    t = {"block": "block", "tcd_cap8.5": "cap", "tcd_ell7.4x8.5x16x32": "ell16", "tcd_ell7.4x8.5x11x22": "ell11"}[r["tip"]]
    if t == "block":
        t += {"four": "4", "five": "5"}[r["layout"].split("_")[0]] + ("w1" if walls(r) == 1 else "")
    return f"{t}_g{r['infill_percent']}"


def expected_masses():
    proc = json.loads((READY / "bench/procedure.json").read_text())
    return next(s for s in proc["steps"] if s["id"] == "mass")["predicts"]


def read_csv(path):
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def yes(v):
    return str(v).strip().lower() in ("yes", "y", "1", "true")


def no(v):
    return str(v).strip().lower() in ("no", "n", "0", "false")


def chosen_seats(rows):
    """{(coupon, kind): narrowest seat that took the magnet to its floor, did not split and kept it; None if no row is
    filled; 'none' if rows are filled and no seat passed}."""
    out = {}
    for coupon in ("two walls", "one wall"):
        for kind in ("round", "slot"):
            rs = [r for r in rows if r["coupon"] == coupon and r["seat_kind"] == kind]
            filled = [r for r in rs if r["takes_to_floor"].strip() or r["keeps_when_tapped"].strip()]
            ok = [float(r["seat_mm"]) for r in rs if yes(r["takes_to_floor"]) and no(r["wall_split"])
                  and yes(r["keeps_when_tapped"])]
            out[(coupon, kind)] = min(ok) if ok else ("none" if filled else None)
    return out


def order(S, seats):
    """The prints in the owner's order, one sheet each: coupons (two walls, one wall), the bench block, the three curved
    tips at 30 %, then the 50 %, five-magnet and one-wall variants. Where sheets of several seats exist, the one of the
    coupon's chosen seat (round pockets for the blocks, slots for the curved tips), else 2.7 mm."""
    def pick(cands, kind):
        if not cands:
            raise SystemExit("a print of the order has no sheet")
        want = seats.get(("two walls", kind))
        for r in cands:
            if isinstance(want, float) and abs(r.get("seat_mm", 0) - want) < 1e-6:
                return r
        return min(cands, key=lambda r: abs(r.get("seat_mm", 2.7) - 2.7))

    def find(tip, lay, w, infill):
        c = [r for r in S if not is_coupon(r) and r["tip"] == tip and walls(r) == w and r["infill_percent"] == infill
             and (r["layout"] == lay or r["layout"].startswith(lay + "_z"))]     # five_z7p4: the 2.9 mm five-magnet block
        return pick(c, "round" if tip == "block" else "slot")
    coupons = sorted([r for r in S if is_coupon(r)], key=lambda r: -walls(r))
    seq = [("Seat test", c) for c in coupons]
    seq += [("Bench tip", find("block", "four", 2, 30))]
    seq += [("Curved tips", find(t, "four", 2, 30)) for t in CURVED]
    seq += [("Variants", find("block", "four", 2, 50)), ("Variants", find("block", "five", 2, 30)),
            ("Variants", find("block", "five", 2, 50))]
    seq += [("Variants", find(t, "four", 2, 50)) for t in CURVED]
    seq += [("Variants", find("block", "four", 1, i)) for i in (30, 50)]
    return seq


def ensure_records(seq, mass):
    """Create the record CSVs with blank measurement columns; add rows for projects they do not list yet."""
    RECORDS.mkdir(parents=True, exist_ok=True)
    rows = read_csv(SEAT_CSV)
    have = {(r["coupon"], r["seat_kind"], r["seat_mm"]) for r in rows}
    for _, r in seq:
        if not is_coupon(r):
            continue
        cp = "two walls" if walls(r) == 2 else "one wall"
        for kind in ("round", "slot"):
            for p in r["pockets"]:
                if p["kind"] != ("pocket" if kind == "round" else "slot"):
                    continue
                key = (cp, kind, f"{p['seat_diameter_mm']:g}")
                if key not in have:
                    rows.append(dict(coupon=cp, project=Path(r["project"]).name, seat_kind=kind, seat_mm=key[2],
                                     takes_to_floor="", wall_split="", keeps_when_tapped="", notes=""))
                    have.add(key)
    with open(SEAT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, SEAT_COLS)
        w.writeheader()
        w.writerows(rows)
    prows = read_csv(PRINTS_CSV)
    have = {r["project"] for r in prows}
    for k, (_, r) in enumerate(seq, start=1):
        proj = Path(r["project"]).name
        if proj in have:
            continue
        prows.append(dict(order=k, print=label(r), project=proj, printed_on="",
                          seat_mm="" if is_coupon(r) else f"{r['seat_mm']:g}", magnets_seated="", polarity_checked="",
                          tip_mass_g="", expected_mass_g="" if is_coupon(r) else f"{mass[mass_key(r)]:.2f}", defects="",
                          notes=""))
    with open(PRINTS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, PRINT_COLS)
        w.writeheader()
        w.writerows(prows)
    return read_csv(SEAT_CSV), read_csv(PRINTS_CSV)


# ------------------------------------------------------------------------------------------------- figures

_URI = {}


def img_uri(png, width=1000, jpeg=True):
    from PIL import Image
    key = (str(png), width, jpeg)
    if key not in _URI:
        im = Image.open(png).convert("RGB")
        if im.width > width:
            im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
        buf = io.BytesIO()
        if jpeg:
            im.save(buf, "JPEG", quality=80, optimize=True, progressive=True)
            mime = "image/jpeg"
        else:
            im.quantize(colors=32).save(buf, "PNG", optimize=True)
            mime = "image/png"
        _URI[key] = (f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode(), im.width, im.height)
    return _URI[key]


FIG = [0]


def figure(png, caption, alt, width=1000, jpeg=True):
    if not png.exists():
        return f"<p><i>Render missing: {esc(rel(png))} (run scripts/fingertip_print_procedure_figures.py)</i></p>"
    uri, w, h = img_uri(png, width, jpeg)
    FIG[0] += 1
    return (f'<figure><img src="{uri}" width="{w}" height="{h}" alt="{esc(alt)}">'
            f"<figcaption>Figure {FIG[0]}. {caption}</figcaption></figure>")


def stem(r):
    return Path(r["stl"]).stem


# ------------------------------------------------------------------------------------------------- sections

def pause_minutes(r):
    at = r["slice"]["check"].get("pause_at_min") or [None] * len(r.get("pauses", []))
    return at


def and_join(items):
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def block_positions(r, idx):
    xs = sorted({round(r["pockets"][i]["centre_print_mm"][0], 2) for i in idx})
    ys = sorted({round(r["pockets"][i]["centre_print_mm"][1], 2) for i in idx})
    pts = [r["pockets"][i]["centre_print_mm"] for i in idx]
    yo = [y for y in ys if y > 0]
    across = (f"{num(yo[0])} mm either side of the centre line" if len(yo) == 1 and -yo[0] in ys
              else f"{', '.join(num(y) for y in ys)} mm across")
    txt = (f"centres {and_join([num(x) for x in xs])} mm from the distal end (the end toward &#8722;X "
           f"on the bed), {across}")
    if len(idx) == 5:
        c = [p for p in pts if abs(p[1]) < 1e-6]
        txt += f": the corners of a 2 &#215; 2 grid and one at the contact point ({num(c[0][0])}, 0)"
    else:
        txt += ": a 2 &#215; 2 grid"
    return txt


def facts_table(r, mass):
    rows = [("Project", f"<code>{esc(Path(r['project']).name)}</code> in <code>{esc(rel(PRINT))}/</code>"),
            ("Tool and filament", "Tool 2 (extruder 2; T1 in the G-code), Siraya Tech TPU 85A @XL 0.4mm Nozzle"),
            ("On the bed", esc(r["orientation"]) + f"; {num(r['recipe']['layer_height_mm'])} mm layers after a "
             f"{num(r['recipe']['first_layer_mm'])} mm first layer"),
            ("Infill", f"Gyroid {r['recipe']['fill_density'].rstrip('%')} %; two perimeters at the outer surface and "
             f"{'two' if walls(r) == 2 else 'one'} around each {'seat' if is_coupon(r) else 'pocket'}; four top and four "
             "bottom layers"),
            ("Print time", r["slice"]["check"]["print_time"]),
            ("Filament", f"{r['slice']['check']['filament_g']:.2f} g, skirt included")]
    if not is_coupon(r):
        at = pause_minutes(r)
        ps = "; ".join(f"{num(p['print_z'])} mm (layer {p['layer']}, about {a:.0f} min in), {len(p['magnets'])} magnets"
                       for p, a in zip(r["pauses"], at))
        rows += [("Magnets", f"{r['magnet']['count']} magnets of 3 &#215; 2 mm, north pole toward the palmar face, top face "
                  f"{num(r['magnet']['top_depth_mm'])} mm under the apex; seat {num(r['seat_mm'])} mm"),
                 ("Pauses", ps),
                 ("Least TPU around a pocket", f"{r['wall_min_mm']:.2f} mm"),
                 ("Expected tip mass", f"{mass[mass_key(r)]:.2f} g with its magnets")]
    else:
        rows += [("Pauses", "No pause; the magnets go into the finished coupon")]
    body = "".join(f"<tr><th>{a}</th><td>{b}</td></tr>" for a, b in rows)
    return f'<div class="tw"><table class="facts"><tbody>{body}</tbody></table></div>'


def pause_box(r, k, p, minute, prev_ref):
    idx = p["magnets"]
    pk = [r["pockets"][i] for i in idx]
    n = len(idx)
    words = {2: "two", 4: "four", 5: "five"}
    one = len(r["pauses"]) == 1
    head = (f"Pause{'' if one else ' ' + str(k)} at {num(p['print_z'])} mm (layer {p['layer']}, about {minute:.0f} min in): "
            f"{words.get(n, n)} magnets")
    steps = [f"The printer stops and shows: <span class=\"msg\">{esc(p['message'])}</span>"]
    if r["tip"] == "block":
        q = pk[0]
        depth = q["top_mm"] - q["floor_mm"]
        gap = min(g["gap_mm"] for g in r["pocket_gaps"])
        steps += [f"Open now: {words.get(n, n)} round pockets {num(q['seat_diameter_mm'])} mm across and {num(depth)} mm deep, "
                  f"{block_positions(r, idx)}.",
                  "Take one magnet at a time from the top of the reference stack, north face up (Section 4).",
                  f"Set it into a pocket north face up and press it to the floor with the flat end of a non-magnetic tool; "
                  f"its top face then lies {num(q['air_above_magnet_mm'])} mm below the rim.",
                  f"Seat each magnet before taking the next: the pockets are {num(gap)} mm apart at their closest and the "
                  "magnets pull toward each other.",
                  f"Check all {words.get(n, n)}: flat on the floor, none above the rim, none moved to a neighbour. Resume."]
    else:
        q = pk[0]
        th = q["slot_x_mm"][1] - q["slot_x_mm"][0]
        where = "proximal (lower)" if "proximal" in p["message"] else "distal (upper)"
        hts = sorted({round(x["centre_print_mm"][1], 2) for x in pk})
        across = sorted({round(abs(x["centre_print_mm"][0]), 2) for x in pk})
        steps += [f"Open now: the {where} pair of slots, {num(th)} mm across and {num(q['seat_diameter_mm'])} mm wide, "
                  f"{num(across[0])} mm either side of the centre line, seat centres {', '.join(num(h) for h in hts)} mm "
                  "above the bed.",
                  "Take one magnet from the top of the reference stack (north face up) and turn it onto its edge with its "
                  "north face toward the curved palmar side, which faces +X on the bed (the blue arrow).",
                  f"Lower it into a slot and press its top edge down with a non-magnetic tool until it rests on the seat; "
                  f"its top edge then lies {num(q['air_above_magnet_mm'])} mm below the slot&#8217;s rim.",
                  "Seat the first magnet before taking the second.",
                  "Check both: down on the seat, north face toward the curved side, none above the rim. Resume."]
    ol = "".join(f"<li>{s}</li>" for s in steps)
    cap = (f"The part at the {'' if one else ['first ', 'second '][k - 1]}pause, cut where the printer stops, with the magnets of this pause seated: red is the north "
           "half, grey the south half; the blue arrow points to the palmar side.")
    png = MEDIA / f"{stem(r)}_pause{k}.png"
    return (f'<div class="pause"><h3>{head}</h3><ol>{ol}</ol>'
            + figure(png, cap, f"Part at pause {k} with magnets") + "</div>")


def good_bad(r):
    ck = r["slice"]["check"]
    br = [q["bridge"]["covered"] for q in ck["pockets"] if q.get("bridge")]
    if is_coupon(r):
        good = ["The three-loop skirt and the first layer lie flat and continuous.",
                "Every seat is open, with clean walls and nothing across it: the G-code lays no bead inside a seat.",
                f"The coupon is {num(r['size_mm'][2])} mm thick and {num(r['size_mm'][0])} &#215; "
                f"{num(r['size_mm'][1])} mm, with the chamfered corner at the 2.7 mm end."]
        bad = ["Strings or blobs across a seat, or a seat squeezed oval.",
               "The first layer or a corner lifting from the sheet.",
               "A seat wall that splits when a magnet is pressed in: record it, it is a test result."]
    else:
        good = ["The three-loop skirt and the first layer lie flat and continuous.",
                "At each pause the open pockets are clean, with nothing inside: the G-code lays no bead in a pocket.",
                f"After each resume the next layer bridges the pockets; the G-code covers "
                + (f"{min(br) * 100:.1f}&#8211;{max(br) * 100:.1f} %" if round(min(br), 3) < round(max(br), 3)
                   else f"{max(br) * 100:.0f} %") + " of each pocket&#8217;s footprint with it.",
                "The finished palmar face is closed and smooth over every magnet, and the tip weighs close to its "
                "expected mass."]
        air = min(p["air_above_magnet_mm"] for p in r["pockets"])
        bad = [f"A magnet above the rim or tilted at a pause: the next layer passes {num(air)} mm over a seated magnet, "
               "so a high magnet meets the nozzle. Press it down again before resuming.",
               "A flipped magnet: only the reference stack shows it before the bridge closes the pocket; afterwards it "
               "cannot be turned.",
               "A magnet that jumped to its neighbour or onto the tool.",
               "Strands sagging into a pocket after the resume, or a dent, hole or bulge in the palmar face over a magnet.",
               "A step or seam in the walls at a pause height (the head returning to a different place).",
               "The first layer or a corner lifting from the sheet."]
    return ("<h3>Signs of a good print</h3><ul>" + "".join(f"<li>{x}</li>" for x in good) + "</ul>"
            "<h3>Signs of a failed print</h3><ul>" + "".join(f"<li>{x}</li>" for x in bad) + "</ul>")


def print_section(k, group, r, mass, seq):
    sid = f"p{k}"
    out = [f'<section id="{sid}"><h2>Print {k}: {label(r)}</h2>', facts_table(r, mass)]
    if is_coupon(r):
        out.append(figure(MEDIA / f"{stem(r)}_bed.png", "The coupon on the bed as the project places it, flat, seats up.",
                          "Seat coupon on the bed"))
        out.append(figure(MEDIA / f"{stem(r)}_top.png",
                          "The coupon from above: round pockets of 2.7, 2.8 and 2.9 mm (the block&#8217;s seat, magnet "
                          "flat) and slots of the same widths (the curved tips&#8217; seat, magnet on its edge), the "
                          "2.7 mm seats at the chamfered end.", "Seat coupon top view", width=900, jpeg=False))
        out.append("<p>The seat test of Section 5 uses this coupon once it is off the sheet.</p>")
    else:
        same = [j for j, (_, q) in enumerate(seq, start=1) if j < k and not is_coupon(q) and stem(q) == stem(r)]
        if same:
            same_w = [j for j in same if walls(seq[j - 1][1]) == walls(r)]
            same_i = [j for j in same if seq[j - 1][1]["infill_percent"] == r["infill_percent"]]
            ref = (same_w or same_i or same)[-1]
            out.append(f"<p>Same part, magnets and pauses as print {ref}; "
                       + ("only the infill differs." if walls(r) == walls(seq[ref - 1][1])
                          else "the project adds a modifier with one perimeter around each pocket.") + "</p>")
        if walls(r) == 1:
            out.append("<p>Print it only if the one-wall coupon kept its magnets in the seat test.</p>")
        ori = ("palmar face up: the magnets lie flat, north pole up" if r["tip"] == "block" else
               "standing on its mount face, the curved palmar side toward +X: the magnets stand on their edge, north "
               "pole toward +X")
        out.append(figure(MEDIA / f"{stem(r)}_bed.png",
                          f"The tip on the bed as the project places it (bed centre, no rotation), {ori}. Blue arrow: "
                          "the palmar side; squares 10 mm.", "Tip on the bed"))
        for j, (p, a) in enumerate(zip(r["pauses"], pause_minutes(r)), start=1):
            out.append(pause_box(r, j, p, a, same))
    out.append(good_bad(r))
    out.append("</section>")
    return "\n".join(out)


def order_section(seq, mass):
    rows, groups = [], {}
    for k, (g, r) in enumerate(seq, start=1):
        t = minutes(r["slice"]["check"]["print_time"])
        groups.setdefault(g, []).append(t)
        pz = ", ".join(f"{num(p['print_z'])} mm" for p in r.get("pauses", [])) or "none"
        mg = r["magnet"]["count"] if not is_coupon(r) else len(r["pockets"])
        rows.append(f"<tr><td>{k}</td><td><a href='#p{k}'>{short_label(r)}</a></td><td>{r['slice']['check']['print_time']}</td>"
                    f"<td>{pz}</td><td>{mg}</td></tr>")
    tot = "; ".join(f"{g.lower()} {fmt_min(sum(v))}" for g, v in groups.items())
    return (f'<section id="order"><h2>1. Order of prints</h2>'
            "<p>The coupons come first because their result sets the seat of every tip. The four-magnet block at 30 % is "
            "the tip of the first-bench package; the capsule and the two half-ellipsoids follow; the variants come last, "
            "the one-wall blocks only if the one-wall coupon keeps its magnets. No quarter-ellipsoid has a print project: "
            "on the five-chip board its flat back saturates the chips (capsule and ellipsoid fingertips page, "
            "Section 9).</p>"
            f'<div class="tw"><table><thead><tr><th>#</th><th>Print</th><th>Time</th><th>Pauses</th><th>Magnets</th></tr>'
            f"</thead><tbody>{''.join(rows)}</tbody></table></div>"
            f"<p>Print time by group: {tot}. Magnets for a coupon: one per seat, pressed in after the print.</p></section>")


def tpu_section(tc, S):
    ch = {c["key"]: c["siraya"] for c in tc["changed"]}
    flow = max(r["slice"]["material"]["volumetric_flow_mm3_s"]["max"] for r in S)
    kept = {c["key"]: c for c in tc["kept"]}
    rows = [("Printer profile", esc(kept["printer_settings_id"]["template"])),
            ("Filament in slot 2", esc(ch["filament_settings_id"].strip('"'))),
            ("Nozzle", f"{ch['temperature']} &#176;C, first layer {ch['first_layer_temperature']} &#176;C"),
            ("Bed", f"{ch['bed_temperature']} &#176;C, first layer {ch['first_layer_bed_temperature']} &#176;C"),
            ("First layer", f"0.2 mm at {ch['first_layer_speed']} mm/s"),
            ("Fan", f"Off for the first {ch['disable_fan_first_layers']} layers, {ch['max_fan_speed']} % from layer "
             f"{ch['full_fan_speed_layer']}"),
            ("Skirt", f"{ch['skirts']} loops {ch['skirt_distance']} mm from the part, {ch['skirt_height']} layers high"),
            ("Flow", f"Extrusion multiplier {ch['extrusion_multiplier']}; at most "
             f"{ch['filament_max_volumetric_speed']} mm&#179;/s, and the largest flow of any move on a part is "
             f"{flow:.2f} mm&#179;/s"),
            ("Retraction", f"{ch['filament_retract_length']} mm, {ch['filament_retract_length_toolchange']} mm at a "
             "tool change"),
            ("Speeds", f"Perimeters {ch['perimeter_speed']} mm/s, external perimeters {ch['external_perimeter_speed']} "
             f"mm/s, infill {ch['infill_speed']} mm/s, bridges {ch['bridge_speed']} mm/s")]
    body = "".join(f"<tr><th>{a}</th><td>{b}</td></tr>" for a, b in rows)
    owner = [("Drying the spool", "dryer, temperature and time: " + OWNER),
             ("Feed path and loading on tool 2", "the XL&#8217;s TPU loading procedure: " + OWNER),
             ("Print sheet and adhesion aid", OWNER),
             ("First-layer offset for TPU on tool 2", OWNER)]
    ow = "".join(f"<li>{a}: {b}</li>" for a, b in owner)
    return ('<section id="tpu"><h2>3. TPU 85A on tool 2 of the XL</h2>'
            "<p>Every project assigns the part to extruder 2, so the G-code selects only T1, the XL&#8217;s second "
            "tool, and heats only that nozzle for the part. Load the Siraya Tech TPU 85A spool on tool 2 and check in "
            "PrusaSlicer that the plater shows the part on extruder 2 before exporting. The projects contain Siraya&#8217;s "
            "profile in filament slot 2 (print/template_changes.json lists every key):</p>"
            f'<div class="tw"><table class="facts"><tbody>{body}</tbody></table></div>'
            "<p>The project files contain none of the following; fill them in from the XL and the dryer:</p>"
            f"<ul>{ow}</ul>"
            "<p>On the first layer, watch the skirt: it primes the nozzle and shows whether the TPU sticks before the "
            "part starts. A skirt that drags, strings or lifts means the first layer of the part will do the same; stop "
            "and fix the first layer before the part begins.</p></section>")


def magnet_section(r0, seq):
    need = sum((r["magnet"]["count"] if not is_coupon(r) else len(r["pockets"])) for _, r in seq)
    first6 = sum((r["magnet"]["count"] if not is_coupon(r) else len(r["pockets"])) for _, r in seq[:6])
    return ('<section id="magnets"><h2>4. Magnets and their polarity</h2>'
            f"<p>Magnets: {esc(r0['magnet']['source'])}. The first six prints take {first6} magnets and all "
            f"{len(seq)} prints {need}. Every magnet goes in with its north pole toward the palmar face; a flipped magnet "
            "reverses its field at the magnetometer chips.</p>"
            "<ol><li>Mark one magnet&#8217;s north face with a dot: a compass needle&#8217;s north end swings toward a "
            "magnet&#8217;s south face, so the face it turns away from is north. This is the reference magnet.</li>"
            "<li>Lay the reference on the bench, marked face up, and stack the other magnets on it. Each magnet sticks "
            "with its south face down, so the top face of the stack is always a north face.</li>"
            "<li>At a pause, slide the top magnet off the stack sideways, keeping its top face as north.</li>"
            "<li>Handle and press magnets with non-magnetic tools only (plastic tweezers, a wooden or brass rod): a "
            "magnet follows steel tweezers back out of its pocket.</li>"
            "<li>Keep the stack and any loose magnet away from an open pocket: a seated magnet jumps to a magnet held "
            "close above it.</li></ol>"
            "<p>After the print, lay the reference magnet loose on the palmar face over each magnet: it turns its south "
            "face down onto a north pole, so its marked face shows upward at every magnet of a correct tip.</p></section>")


def seat_lede(seats, S):
    """The lede's seat sentence: the two-wall coupon's chosen seat and what that seat changes in the tips, read from the
    sheets built at it; the provisional 2.7 mm before the coupon has a result."""
    s = seats.get(("two walls", "round"))
    if not isinstance(s, float):
        return "Every tip has a 2.7 mm seat until the coupons show that a wider one is needed."
    at = [r for r in S if not is_coupon(r) and abs(r.get("seat_mm", 0) - s) < 1e-6]
    ell = next((r for r in at if r["tip"] == "tcd_ell7.4x8.5x11x22"), None)
    five = next((r for r in at if r["layout"].startswith("five")), None)
    extra = []
    if ell and abs(ell["magnet"]["top_depth_mm"] - 7.0) > 1e-6:
        extra.append(f"the 11/22 half-ellipsoid's magnets sit {num(ell['magnet']['top_depth_mm'])} mm under the apex "
                     f"({ell['wall_min_mm']:.2f} mm of TPU to the surface)")
    if five and five["layout"] != "five":
        g = min(x["gap_mm"] for x in five["pocket_gaps"])
        extra.append(f"the five-magnet block's corner pockets sit {five['layout'][6:].replace('p', '.')} mm apart along "
                     f"the finger ({g:.2f} mm between pockets)")
    txt = f"The two-wall seat coupon chose {s:g} mm (owner, 2026-10-18), and every tip here has {s:g} mm seats"
    if extra:
        txt += "; to keep 1.8 mm of TPU around every pocket at that seat, " + " and ".join(extra)
    return txt + "."


def seat_section(seat_rows, seats, walls_json, cmd_prefix):
    def cell(v):
        v = (v or "").strip()
        return f"<td>{esc(v)}</td>" if v else '<td class="blank"></td>'
    rows = "".join(f"<tr><td>{esc(r['coupon']).capitalize()}</td><td>{esc(r['seat_kind']).capitalize()}</td>"
                   f"<td>{esc(r['seat_mm'])}</td>"
                   + cell(r["takes_to_floor"]) + cell(r["wall_split"]) + cell(r["keeps_when_tapped"]) + "</tr>"
                   for r in seat_rows)

    def seat_txt(c, k):
        v = seats.get((c, k))
        return "pending" if v is None else ("none of the three" if v == "none" else f"{v:g} mm")
    res = (f"Seat from the two-wall coupon: round pockets {seat_txt('two walls', 'round')}, slots "
           f"{seat_txt('two walls', 'slot')}; one-wall coupon: round pockets {seat_txt('one wall', 'round')}.")
    wt = ""
    if walls_json:
        W = json.loads(walls_json.read_text())
        names = {"block": "Block", "tcd_cap8.5": "Capsule r 8.5 mm", "tcd_ell7.4x8.5x16x32": "Half-ellipsoid 16/32 mm",
                 "tcd_ell7.4x8.5x11x22": "Half-ellipsoid 11/22 mm"}
        by = {}
        for x in W["rows"]:
            by.setdefault((x["tip"], x["layout"]), {})[x["seat_mm"]] = x
        def least(x):
            return min(x["wall_min_mm"], x["gap_min_mm"]) if x["gap_min_mm"] is not None else x["wall_min_mm"]
        trs = []
        for (tip, lay), d in by.items():
            lab = names[tip] + (", five magnets" if lay == "five" else "")
            tds = "".join(f"<td>{least(d[s]):.{3 if abs(least(d[s]) - W['rule_mm']) < 0.01 else 2}f}"
                          f"{'' if d[s]['walls_ok'] else ' (fails)'}</td>"
                          for s in (2.7, 2.8, 2.9) if s in d)
            trs.append(f"<tr><td>{lab}</td><td>{num(next(iter(d.values()))['depth_mm'])}</td>{tds}</tr>")
        fails = []
        for x in W["rows"]:
            if x["walls_ok"]:
                continue
            lab = names[x["tip"]] + (", five magnets" if x["layout"] == "five" else "")
            why = ("between pockets" if x["gap_min_mm"] is not None and x["gap_min_mm"] < W["rule_mm"]
                   else "to the surface")
            fails.append(f"the {lab[0].lower() + lab[1:]} at {x['seat_mm']:g} mm ({least(x):.2f} mm {why})")
        ok_all = sorted({x["seat_mm"] for x in W["rows"]} - {x["seat_mm"] for x in W["rows"] if not x["walls_ok"]})
        wt = (f"<p>A wider seat thins the TPU around each pocket. The tip builder stops when a pocket keeps less "
              f"than {num(W['rule_mm'])} mm to the surface or to another pocket (four 0.45 mm perimeters). At the 85A "
              f"magnet depths the least TPU around a pocket is (scripts/fingertip_print_procedure_figures.py walls, "
              f"<code>{esc(rel(walls_json))}</code>):</p>"
              f'<div class="tw"><table><thead><tr><th>Tip</th><th>Magnet top under the apex (mm)</th>'
              f"<th>Seat 2.7 mm</th><th>Seat 2.8 mm</th><th>Seat 2.9 mm</th></tr></thead><tbody>{''.join(trs)}"
              "</tbody></table></div>"
              + (f"<p>Every tip passes at {and_join([f'{v:g}' for v in ok_all])} mm. The rule fails for "
                 f"{and_join(fails)}. "
                 "The builder works through its list in order and stops at the first tip that fails, so at that seat "
                 "these tips need a new layout or magnet depth before the rebuild.</p>" if fails else
                 "<p>Every tip passes the rule at all three seats.</p>"))
    b = (f"{cmd_prefix} scripts/sim2real_fingertip_prints.py build --tips block --seat 2.8\n"
         f"{cmd_prefix} scripts/sim2real_fingertip_prints.py build --tips {' '.join(CURVED)} --seat 2.8\n"
         f"{cmd_prefix} scripts/sim2real_fingertip_prints.py readme\n"
         "~/.claude/bin/resguard.sh run --mem 1G -- ~/miniconda3/envs/femtip/bin/python "
         "scripts/fingertip_print_procedure_figures.py cut\n"
         "MUJOCO_GL=egl ~/.claude/bin/resguard.sh run --mem 1500M -- .venv/bin/python "
         "scripts/fingertip_print_procedure_figures.py render\n"
         ".venv/bin/python scripts/fingertip_print_procedure_page.py")
    return ('<section id="seat"><h2>5. Seat test on the two coupons</h2>'
            "<p>Every tip has a 2.7 mm seat, 90 % of the magnet&#8217;s 3 mm diameter, until the coupons show which of "
            "2.7, 2.8 and 2.9 mm takes the magnet to its floor and keeps it in 85A. Round pockets test the block&#8217;s "
            "seat (magnet flat); slots test the curved tips&#8217; seat (magnet on its edge). The two-wall coupon sets "
            "the seat of the two-wall tips, the one-wall coupon tells whether the one-wall blocks are worth printing.</p>"
            "<ol><li>Take the coupon off the sheet once it has cooled.</li>"
            "<li>Press one magnet into each round pocket, flat, with a flat non-magnetic tool, and one into each slot on "
            "its edge.</li>"
            "<li>For each seat note whether the magnet reached the floor (its top face 0.15&#8211;0.25 mm below the "
            "surface), whether the wall split, and whether the magnet stays when the coupon is turned over and tapped "
            "on the table.</li>"
            "<li>The seat to use is the narrowest that takes the magnet to its floor without splitting the wall and "
            "keeps it.</li>"
            f"<li>Write the result into <code>{esc(rel(SEAT_CSV))}</code> (yes or no in each column).</li></ol>"
            f'<div class="tw"><table><thead><tr><th>Coupon</th><th>Seat</th><th>mm</th><th>Takes magnet to floor</th>'
            f"<th>Wall split</th><th>Keeps it when tapped</th></tr></thead><tbody>{rows}</tbody></table></div>"
            f"<p>{res}</p>"
            "<p>If 2.7 mm takes and keeps the magnet, print the tips as they are. If only 2.8 or 2.9 mm does, rebuild the "
            "tips of that family with the chosen seat before printing them (2.8 shown; the seat is in the new file "
            "names, and the 2.7 mm files stay), then the README, the figures and this page:</p>"
            f'<pre class="cmd">{esc(b)}</pre>{wt}</section>')


def removal_section():
    return ('<section id="removal"><h2>6. Removal and inspection</h2>'
            "<ol><li>When the print ends, take the steel sheet off and flex it to release the part.</li>"
            "<li>Peel the skirt away. Lift the tip by its sides, not by the palmar face.</li>"
            "<li>Look at the palmar face over every magnet: closed and smooth, no dent, hole or bulge.</li>"
            "<li>Check the polarity with the loose reference magnet (Section 4): marked face up over every magnet.</li>"
            "<li>Weigh the tip on a scale that reads 0.01 g and compare with its expected mass.</li>"
            f"<li>Write the date, seat, magnets seated, polarity, mass and any defect into "
            f"<code>{esc(rel(PRINTS_CSV))}</code>.</li></ol></section>")


def record_section(prows):
    def cell(v):
        v = (v or "").strip()
        return f"<td>{esc(v)}</td>" if v else '<td class="blank"></td>'
    rows = "".join(f"<tr><td>{esc(r['order'])}</td><td>{esc(r['print'])}</td>" + cell(r["printed_on"]) + cell(r["seat_mm"])
                   + cell(r["magnets_seated"]) + cell(r["polarity_checked"]) + cell(r["tip_mass_g"])
                   + cell(r["expected_mass_g"]) + cell(r["defects"]) + cell(r.get("notes")) + "</tr>" for r in prows)
    return ('<section id="record"><h2>7. Record of the prints</h2>'
            "<p>Each result goes to a file in the repository; the page shows what the files contain when it is rebuilt.</p>"
            '<div class="tw"><table><thead><tr><th>Result</th><th>File</th></tr></thead><tbody>'
            f"<tr><td>Seat that takes and keeps the magnet, per coupon, seat kind and size</td><td><code>{esc(rel(SEAT_CSV))}"
            "</code></td></tr>"
            f"<tr><td>Date, seat, magnets seated, polarity checked, tip mass, defects, per print</td><td><code>"
            f"{esc(rel(PRINTS_CSV))}</code></td></tr>"
            "<tr><td>Tips rebuilt with a new seat</td><td><code>docs/experiments/20261016-sim2real_fingertip_readiness/"
            "print/</code> and its README.md</td></tr></tbody></table></div>"
            '<div class="tw"><table><thead><tr><th>#</th><th>Print</th><th>Printed on</th><th>Seat (mm)</th>'
            "<th>Magnets seated</th><th>Polarity checked</th><th>Tip mass (g)</th><th>Expected (g)</th><th>Defects</th>"
            f"<th>Notes</th></tr></thead><tbody>{rows}</tbody></table></div></section>")


COUPON_DIR = ROOT / "docs/experiments/20261018-dip_link_board_mount/final/print"
FIT_CSV = RECORDS / "fit_coupon.csv"
FIT_COLS = ["block", "interference_mm_per_flank", "project", "printed_on", "slides_on", "clicks_over_detent", "stays_on_pull",
            "stays_on_twist", "comes_off_by_hand", "notes"]
DIP_PAGE = "../20261018-dip_link_board_mount/20261018-dip_link_magnetometer_board_mount.html"


def fit_coupon_section():
    """Section 8: the attachment fit coupon of the new DIP link's dovetail keys (test T1 of the DIP page), from the
    coupon's reports in the DIP study (scripts/dip_link_print_project.py --coupon-only); the record CSV gets one row per
    channel block."""
    kp = json.loads((COUPON_DIR / "20261018-dip_link_final_fit_coupon_keys_pla_t3.json").read_text())
    cp = json.loads((COUPON_DIR / "20261018-dip_link_final_fit_coupon_channels_85a_t2.json").read_text())
    widths = cp["slice"]["channel_widths"]
    rows = read_csv(FIT_CSV)
    have = {r["interference_mm_per_flank"] for r in rows}
    for k, fl in enumerate(sorted(widths), 1):
        if fl not in have:
            rows.append(dict(block=f"{k} notch" + ("es" if k > 1 else ""), interference_mm_per_flank=fl,
                             project=Path(cp["project"]).name, printed_on="", slides_on="", clicks_over_detent="",
                             stays_on_pull="", stays_on_twist="", comes_off_by_hand="", notes=""))
    with open(FIT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, FIT_COLS)
        w.writeheader()
        w.writerows(rows)
    dev = max(abs(r["sliced_mm"] - r["design_mm"]) for v in widths.values() for r in v[1:] if r["sliced_mm"] is not None)
    first = [v[0]["sliced_mm"] - v[0]["design_mm"] for v in widths.values()]

    def cell(v):
        v = (v or "").strip()
        return f"<td>{esc(v)}</td>" if v else '<td class="blank"></td>'
    rec = "".join(f"<tr><td>{esc(r['block'])}</td><td>{esc(r['interference_mm_per_flank'])}</td>" + cell(r["printed_on"])
                  + cell(r["slides_on"]) + cell(r["clicks_over_detent"]) + cell(r["stays_on_pull"])
                  + cell(r["stays_on_twist"]) + cell(r["comes_off_by_hand"]) + cell(r["notes"]) + "</tr>" for r in rows)
    proj = "".join(f"<tr><td><a href='../20261018-dip_link_board_mount/final/print/{Path(r['project']).name}'><code>"
                   f"{esc(Path(r['project']).name)}</code></a></td><td>{esc(t)}</td><td>{esc(r['slice']['print_time'])}</td>"
                   f"<td>{r['slice']['filament_g']:.1f}</td></tr>"
                   for r, t in ((kp, "Three PLA keys with detents; tool 3, PLA, bed 60 \u00b0C, flange on the bed"),
                                (cp, "Three 85A channel blocks; tool 2, Siraya 85A, bed 35 \u00b0C, back face on the bed, "
                                     "no pause")))
    return ('<section id="fit"><h2>8. Attachment fit coupon for the new DIP link</h2>'
            f"<p>The new DIP link (<a href='{DIP_PAGE}'>DIP page</a>, concept a) holds the tip on two PLA dovetail keys, "
            "each with a 0.25 mm detent, and the tip&#8217;s channels are narrower than the keys by an interference per "
            "flank. The coupon tries three interferences before any tip with channels gets magnets: three PLA keys on one "
            "part, and three 85A channel blocks at 0.00, 0.05 and 0.10 mm per flank, marked by one, two and three notches at "
            "their closed end. The design uses 0.05 mm. The tips printed with 2.9 mm seats have no channels.</p>"
            '<div class="tw"><table><thead><tr><th>Project</th><th>Parts, tool and bed</th><th>Time</th><th>Filament (g)</th>'
            f"</tr></thead><tbody>{proj}</tbody></table></div>"
            "<p>Each block slides onto its key from the key&#8217;s free end, open end first, until that end meets the "
            "flange; the detent sits 4.3 mm from the flange. On each block, by hand:</p><ol>"
            "<li>Slides on: the block goes on with a firm push until it meets the flange.</li>"
            "<li>Clicks: the detent drops into the block&#8217;s groove with a felt click.</li>"
            "<li>Stays on when pulled along the key.</li>"
            "<li>Stays on when twisted about the axis normal to the flange&#8217;s face.</li>"
            "<li>Comes off by hand with a firm pull, without tearing the 85A.</li></ol>"
            "<p>The tightest block that passes all five sets <code>fit_flank</code> in <code>scripts/dip_link_concept_a.py</code> "
            "(P) for the reprinted tips; then <code>scripts/dip_link_board_mount.py build</code> and the hanging-load test T2 "
            "on the DIP page. As sliced, each channel&#8217;s width equals the design within "
            f"{dev:.3f} mm on every layer from the second; the first layer opens "
            + (f"{min(first):.2f}" if round(min(first), 2) == round(max(first), 2) else f"{min(first):.2f}&#8211;{max(first):.2f}")
            + " mm "
            "wider, which is PrusaSlicer&#8217;s elephant-foot compensation of 0.2 mm per side.</p>"
            f"<p>Record each block in <code>{esc(rel(FIT_CSV))}</code>:</p>"
            '<div class="tw"><table><thead><tr><th>Block</th><th>Interference (mm per flank)</th><th>Printed on</th>'
            "<th>Slides on</th><th>Clicks</th><th>Stays on, pulled</th><th>Stays on, twisted</th><th>Comes off by hand</th>"
            f"<th>Notes</th></tr></thead><tbody>{rec}</tbody></table></div></section>")


def terms_section():
    T = [("Palmar face", "The tip&#8217;s contact face, toward the object. The block prints with it up; the capsule and "
          "half-ellipsoids print standing, with it toward +X on the bed."),
         ("Proximal, distal", "Toward the finger&#8217;s base and toward its end. On the standing tips the proximal "
          "magnets are the lower pair, since the mount face lies on the bed."),
         ("Pocket, slot", "A pocket is a closed round hole in the block for a magnet lying flat; a slot is a hole in a "
          "standing tip for a magnet on its edge, open upward until its pause."),
         ("Seat", "The press-fit diameter of a pocket or the width of a slot, in mm; 2.7 mm is 90 % of the magnet&#8217;s "
          "3 mm diameter."),
         ("Pause", "A stop (M601) before the first layer over a pocket; the printer shows the project&#8217;s message."),
         ("Walls", "Perimeter loops: two at the outer surface of every print, and two or one around each pocket."),
         ("Gyroid infill", "The infill density of the core, 30 or 50 %: the tip&#8217;s stiffness choice."),
         ("Tool 2", "The XL&#8217;s second tool head: extruder 2, selected as T1 in the G-code."),
         ("Filament", "PrusaSlicer&#8217;s estimate of the filament the print uses, skirt included, in g."),
         ("Expected tip mass", "The TPU the G-code extrudes on the part plus its magnets, from the readiness study "
          "(bench/procedure.json), in g.")]
    dl = "".join(f"<dt>{a}</dt><dd>{b}</dd>" for a, b in T)
    return f'<section id="terms"><h2>2. Terms</h2><dl class="glossary">{dl}</dl></section>'


def future_section(seats):
    proc = {x["id"]: x for x in json.loads((READY / "bench/procedure.json").read_text())["steps"]}
    ret = proc["retention"]
    ind = proc["indentation"]["predicts"]["block, four magnets, 30 % gyroid"]["1 N"]["E_s 12 MPa"]
    return ('<section id="future"><h2>Future work</h2><ul>'
            f"<li>Seat: fill <code>{esc(rel(SEAT_CSV))}</code> from both coupons; a seat wider than 2.7 mm means "
            "rebuilding the tips (Section 5) before any tip is printed.</li>"
            f"<li>Masses: weigh every tip into <code>{esc(rel(PRINTS_CSV))}</code>; a tip far below its expected mass "
            "points to under-extrusion of the 85A on tool 2.</li>"
            f"<li>Bench, on the printed 30 % four-magnet block: {esc(ret['what'])} (expected: {esc(ret['predicts'])}); "
            f"then the indentation under the 25 mm rod at 0.5, 1 and 3 N, {ind:g} &#181;m at 1 N if Siraya&#8217;s solid "
            "modulus is 12 MPa (docs/experiments/20261016-sim2real_fingertip_readiness/bench/procedure.json).</li>"
            "</ul></section>")


# ------------------------------------------------------------------------------------------------- build

def build():
    S = sheets()
    tc = json.loads((PRINT / "template_changes.json").read_text())
    mass = expected_masses()
    seat_rows = read_csv(SEAT_CSV)
    seats = chosen_seats(seat_rows) if seat_rows else {}
    seq = order(S, seats)
    seat_rows, prows = ensure_records(seq, mass)
    seats = chosen_seats(seat_rows)
    doc = (ROOT / "scripts/sim2real_fingertip_prints.py").read_text()
    m = re.search(r"(~/\.claude/bin/resguard\.sh run [^\n]*?python)\s*\\\\\s*\n\s*scripts/sim2real_fingertip_prints\.py", doc)
    cmd_prefix = m.group(1) if m else "~/miniconda3/envs/femtip/bin/python"
    walls_json = D / "seat_walls.json"

    first = seq[:6]
    t6 = sum(minutes(r["slice"]["check"]["print_time"]) for _, r in first)
    cp = [r for _, r in seq if is_coupon(r)]
    bt = seq[2][1]
    cv = [r for g, r in seq if g == "Curved tips"]
    cvt = [r["slice"]["check"]["print_time"] for r in cv]
    pz = " and ".join(f"{num(p['print_z'])}" for p in cv[0]["pauses"])
    lede = (f"The {len(seq)} PrusaSlicer projects of the 85A set print the magnet fingertips in Siraya Tech TPU 85A on tool 2 of the Prusa XL, "
            f"at 240 &#176;C on a 35 &#176;C bed. Print them in this order: the two seat coupons "
            f"({' and '.join(r['slice']['check']['print_time'] for r in cp)}), whose result sets the magnet seat; the "
            f"30 % four-magnet block of the bench package ({bt['slice']['check']['print_time']}, one pause at "
            f"{num(bt['pauses'][0]['print_z'])} mm); the capsule and the two half-ellipsoids at 30 % "
            f"({min(cvt, key=minutes)} to {max(cvt, key=minutes)}, pauses at {pz} mm); then the 50 %, five-magnet and "
            f"one-wall variants. The first six prints take {fmt_min(t6)}. " + seat_lede(seats, S))
    secs = [order_section(seq, mass), terms_section(), tpu_section(tc, S), magnet_section(bt, seq),
            seat_section(seat_rows, seats, walls_json if walls_json.exists() else None, cmd_prefix), removal_section(),
            record_section(prows), fit_coupon_section()]
    for k, (g, r) in enumerate(seq, start=1):
        secs.append(print_section(k, g, r, mass, seq))
    secs.append(future_section(seats))
    toc = ["<li><a href='#order'>1. Order of prints</a></li>", "<li><a href='#terms'>2. Terms</a></li>",
           "<li><a href='#tpu'>3. TPU 85A on tool 2 of the XL</a></li>",
           "<li><a href='#magnets'>4. Magnets and their polarity</a></li>",
           "<li><a href='#seat'>5. Seat test on the two coupons</a></li>",
           "<li><a href='#removal'>6. Removal and inspection</a></li>",
           "<li><a href='#record'>7. Record of the prints</a></li>",
           "<li><a href='#fit'>8. Attachment fit coupon for the new DIP link</a></li>"]
    toc += [f"<li><a href='#p{k}'>Print {k}: {short_label(r)}</a></li>" for k, (_, r) in enumerate(seq, start=1)]
    toc.append("<li><a href='#future'>Future work</a></li>")
    up = "../.."
    eyebrow = (f'<a href="{up}/index.html">SR&#178; Hand documentation</a> &#183; <a href="{up}/pages.html">index of '
               f'result pages</a> &#183; <a href="{up}/overviews/printed_fingertips.html">printed fingertips and force '
               f'sensing</a> &#183; 18 October 2026')
    byline = (f'Siraya Tech TPU 85A &#183; Prusa XL 5T, tool 2, 0.4 mm nozzle &#183; 3 &#215; 2 mm N52 magnets &#183; '
              f'print files and sheets: <code>{esc(rel(PRINT))}/</code> (README.md); geometry and checks: '
              f'<a href="../20261016-sim2real_fingertip_readiness/20261016-cad_hand_fingertip_deployment_readiness.html">'
              f'deployment readiness page</a> &#183; built {time.strftime("%Y-%m-%d %H:%M")}')
    footer = ("Rebuild: <code>.venv/bin/python scripts/fingertip_print_procedure_page.py</code> (template "
              "<code>scripts/fingertip_print_procedure_page.template.html</code>); figures and the seat-wall check: "
              "<code>scripts/fingertip_print_procedure_figures.py cut|walls|render</code>. Sources: the sheets, README.md "
              f"and template_changes.json in <code>{esc(rel(PRINT))}/</code>, "
              "<code>docs/experiments/20261016-sim2real_fingertip_readiness/bench/procedure.json</code>, and the records "
              f"in <code>{esc(rel(RECORDS))}/</code>. Hand-off: <code>docs/handoff/20261018-print_procedure.md</code>.")
    page = TEMPLATE.read_text()
    for k, v in dict(TITLE="Fingertip print procedure",
                     H1="Print procedure for the Siraya Tech 85A magnet fingertips on the Prusa XL",
                     EYEBROW=eyebrow, LEDE=lede, BYLINE=byline, TOC="\n".join(toc), SECTIONS="\n".join(secs),
                     FOOTER=footer).items():
        page = page.replace("{{" + k + "}}", v)
    if "{{" in page:
        raise SystemExit(f"unfilled placeholder: {re.findall(r'{{[A-Z_]+}}', page)}")
    page = retro_style.apply(page)
    D.mkdir(parents=True, exist_ok=True)
    PAGE.write_text(page)
    print(f"wrote {rel(PAGE)}: {len(page) / 1e6:.2f} MB, {FIG[0]} figures, {len(seq)} prints", flush=True)


if __name__ == "__main__":
    build()
