#!/usr/bin/env python3
"""Result page of 2026-10-23: UR5 calibration mount v4, mount v3 with a stop plate behind the DIP link's tongue (the owner,
through the lab session: the DIP link moves a little in v3's fork when the pad is pressed hard; QUEUE.md R16 and R18 of the
UR5 tip calibration).

Reads docs/experiments/20261023-ur5_calibration_mount_v4/ (parts, STEP read-back, link checks, payload, clearances along the
calibration plan, the P1S plate's record, the renders and the turntable film), written by scripts/ur5_calibration_mount.py
with UR5M_REV=v4, scripts/v3b_ur5v2_pla_plate.py mount_v4_fork and scripts/ur5_tip_calibration_check.py --clearance with
UR5_MOUNT=v3 and v4, and v3's check for the comparison; writes
docs/experiments/20261023-ur5_calibration_mount_v4/20261023-ur5_calibration_mount_v4.html.

    .venv/bin/python scripts/ur5_mount_v4_page.py
"""
from __future__ import annotations

import base64
import json
import math
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import docs_index  # noqa: E402
import retro_style  # noqa: E402
import texsvg  # noqa: E402
import v2_tip_fe_checks_page as VP  # noqa: E402  (table, img_uri, esc)

D = ROOT / "docs/experiments/20261023-ur5_calibration_mount_v4"
PFX = "20261023-ur5_calibration_mount_v4"
PAGE = D / f"{PFX}.html"
TEMPLATE = ROOT / "scripts/ur5_tip_calibration_page.template.html"
TEX_CACHE = D / "texsvg_cache.json"
TEX_RE = re.compile(r"\$\$(.+?)\$\$|\$(.+?)\$", re.S)
D3 = ROOT / "docs/experiments/20261021-ur5_calibration_mount_v3"
P3 = "20261021-ur5_calibration_mount_v3"
CAL = ROOT / "docs/experiments/20261022-ur5_tip_calibration"
PAGES = dict(v3=D3 / f"{P3}.html", results=CAL / "20261022-ur5_tip_ground_truth_and_predictors.html",
             lab=CAL / "20261022-ur5_tip_calibration_lab_runs.html", queue=CAL / "QUEUE.md",
             status=ROOT / "docs/experiments/20261018-printed_fingertip_status/20261018-printed_fingertip_and_dip_link_status.html")
table, img_uri, esc = VP.table, VP.img_uri, VP.esc


def rel(p):
    return os.path.relpath(p, D)


def link(p, text):
    return f'<a href="{rel(p)}">{text}</a>'


def code(p):
    return f"<code>{esc(os.path.relpath(p, ROOT) if isinstance(p, Path) else p)}</code>"


def fig(n, png, caption, width=1100):
    return (f'<figure id="fig{n}"><img src="{img_uri(png, width)}" alt="{esc(caption.split(".")[0])}">'
            f"<figcaption>Figure {n}. {caption}</figcaption></figure>")


def film(n, mp4, poster, caption):
    v = "data:video/mp4;base64," + base64.b64encode(Path(mp4).read_bytes()).decode()
    return (f'<figure id="fig{n}"><video controls muted loop playsinline autoplay preload="metadata" poster="{img_uri(poster, 960)}" '
            f'src="{v}" style="width:100%"></video><figcaption>Figure {n}. {caption}</figcaption></figure>')


def load():
    j = lambda p: json.loads(Path(p).read_text())                              # noqa: E731
    return dict(parts=j(D / f"{PFX}_parts.json"), step=j(D / f"{PFX}_step.json")["parts"], chk=j(D / f"{PFX}_check.json"),
                chk3=j(D3 / f"{P3}_check.json"), pay=j(D / f"{PFX}_payload.json"), pay3=j(D3 / f"{P3}_payload.json"),
                plate=j(D / f"print/{PFX}_p1s_pla.json"), clr=j(D / f"{PFX}_clearance.json"))


def ands(parts):
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def deg(v):
    return "over 5" if v is None else f"{v:g}"


def sections(R):
    P = R["parts"]["parameters"]
    c4, c3 = R["chk"]["b_links"]["v4b"], R["chk3"]["b_links"]["v4b"]
    t3, t4, t4s = c3["turn_to_contact_deg"], c4["turn_to_contact_deg"], c4["turn_to_contact_deg_screws_set"]
    arm = 87.91 - 51.55                                    # the contact point below the PIP axis (z_F), mm
    play = arm * math.sin(math.radians(t3["about_y_pip_-"]))
    clr = R["clr"]
    pl = R["plate"]
    fork = next(o for o in pl["objects"] if o["id"] == "U2")
    it4 = {i["item"]: i for i in R["pay"]["items"]}
    it3 = {i["item"]: i for i in R["pay3"]["items"]}
    dm = it4["standoff_fork"]["mass_g"] - it3["standoff_fork"]["mass_g"]
    hw = 3 * 0.9 + 3 * 0.25 + 4 * 0.4                      # set screws, brass nuts, inserts (g)
    lede = (f"On the UR5 run sphere2 of 22 October, the presses travelled 0.33 mm plus 0.20 mm per newton beyond the pad's "
            f"finite-element indentation. After the run the owner felt the DIP link move in mount v3's fork. In v3 the link is held "
            f"by its legs alone. With the clamp loose it turns {t3['about_y_pip_-']:g}° about the PIP axis before it meets the "
            f"fork, which moves the contact point {play:.2f} mm. Mount v4 adds a plate behind the tongue with three brass M3 set "
            f"screws, turned onto the tongue's back once the link is clamped. With the screws set, a press on the face can turn "
            f"the link {t4s['about_y_pip_-']:g}°, the smallest step of the check. The plate stays "
            f"{clr['v4_stop_plate']['registered_mm']:.0f} mm from the cell over the whole plan. Only the standoff-fork is "
            f"reprinted: {pl['time']}, {pl['filament_g']:.1f} g.")
    summary = (f"Mount v4 = v3 + a stop plate behind the DIP link's tongue with three brass M3 set screws: the link's turn about "
               f"the PIP axis under a face press falls from {t3['about_y_pip_-']:g} to {t4s['about_y_pip_-']:g} deg; the plate "
               f"stays {clr['v4_stop_plate']['registered_mm']:.0f} mm from the cell; fork reprint {pl['time']}, "
               f"{pl['filament_g']:.0f} g; not yet printed or measured.")
    S = []
    terms = [
        ("Frames F and M", "F is the DIP link's frame: x_F toward the dorsal side, y_F along the PIP axis, z_F along the "
         "finger, negative toward the tip. M is the mount's frame: its origin at the adapter's face on the HEX-E, z_M along "
         "the tool axis toward the tip, x_M toward the palmar side. Figures are drawn in M."),
        ("Turn to contact", "The least angle, in degrees, through which the link can be turned about an axis through the "
         "bridge's middle before its mesh overlaps the fork's. The clamp plate and the slotted arm's screw are left off, so "
         "the number measures the fork's shape. The check steps through 0.05, 0.1, 0.15 ... 5°."),
        ("Indentation beyond the FE pad", "The press travel that the pad's finite-element model does not account for, "
         "fitted as $\\Delta = a + c\\,f_n$ on the UR5 run sphere2 (" + link(PAGES["results"], "results page") + ")."),
        ("Clearance", "The least distance, in mm, from a tool part to the sphere indenter, the cell and the rod's fixture "
         "over the plan's press poses and every degree of its turns. In the plan, 3 mm is required for the mount and 0.8 mm "
         "for the finger."),
    ]
    S.append(("terms", "Terms and metrics",
              '<dl class="terms">' + "".join(f"<dt>{t}</dt><dd>{d}</dd>" for t, d in terms) + "</dl>"))

    body = (f"<p>The UR5 run sphere2 measured the travel of 357 press holds against the HEX-E's force. Beyond the FE pad's "
            f"indentation it found 0.33 mm plus 0.20 mm per newton. The slope was 0.21 mm/N on the face, 0.17 on the edges and "
            f"0.14 on the flanks. The force read from the board was within 22 % of the HEX-E's, so the pad itself is not "
            f"the compliant part. After the run the owner pressed the pad by hand and saw the DIP link move a little in the "
            f"fork (" + link(PAGES["queue"], "QUEUE.md") + ", R16 and R18).</p>"
            f"<p>In mount v3 the link is located by walls 0.15 mm off the outlines of its two legs. The clamp plate and one M3 "
            f"press the legs onto the inner block along the PIP axis. A press on the face pushes the pad toward the dorsal "
            f"side, {arm:.1f} mm below the PIP axis, and so turns the link about that axis. Only friction at the legs resists "
            f"this until the walls are met. With the clamp loose, the turn to contact in v3's check is "
            f"{t3['about_y_pip_-']:g}° about the PIP axis, {t3['about_x_+']:g}° about x and {t3['about_z_tool_+']:g}° about "
            f"the tool axis. A turn of {t3['about_y_pip_-']:g}° about the PIP axis moves the contact point {play:.2f} mm. The "
            f"order of these turns matches the order of the measured slopes: face, then edges, then flanks.</p>")
    S.append(("play", "Play of the DIP link in mount v3", body))

    xs, xo = P["stop_x"]
    rows = [["Plate's inner face (x_F)", f"{xs:.2f} mm", "0.5 mm off the neck's back (9.45), so the link slides on along the tool axis"],
            ["Plate's outer face (x_F)", f"{xo:.2f} mm", "inside the fork's footprint (17.0 mm)"],
            ["Width (y_F)", f"±{P['stop_y']:.1f} mm", "the tongue's back face spans ±8.95 mm below z_F -79"],
            ["Length (z_F)", f"{P['stop_top_z']:.1f} to {P['stop_bot_z']:.1f} mm", "from the dorsal ear's bottom to 2.1 mm below the tip cap's top"],
            ["Set screws (y_F, z_F)", "; ".join(f"({y:+.0f}, {z:.1f})" for y, z in P["stop_screws"]),
             "a three-point seat behind the board's chips U2 and U1, U3, U5"],
            ["Screw reach", f"{xs - 8.0:.2f} mm", "from the plate's inner face to the tongue's back (x_F 8.0)"],
            ["Insert bores", f"{P['stop_ins_hole']:.1f} mm, {P['stop_ins_len'] + 0.5:.1f} mm deep",
             "M3 x 5.7 brass heat-set inserts pressed from the outer face, 0.3 mm chamfer at the mouth"],
            ["Slotted arm's screw", "M3 x 12 into an insert", f"a {P['stop_ins_hole']:.1f} mm bore {P['sl_ins_depth']:.1f} mm deep in the inner block (v3: a 2.75 mm self-forming hole)"]]
    body = (f"<p>The plate hangs from the fork's dorsal ear behind the tongue, the flat part of the link on which the "
            f"board sits (Fig. 1, Fig. 2). Three brass M3 set screws run through it in heat-set inserts. Once the link is clamped as "
            f"in v3, each screw is turned until it touches the tongue's back and is locked with a jam nut. The tongue is then "
            f"held by the screws where the walls and the clamp placed it. A press on the face or an edge then pushes the tongue onto them, "
            f"5 to 8 mm behind the contact point. In v3 the same press turned the link about the PIP axis. The screws touch the "
            f"tongue only after it is seated, so they do not set the link's place, which was the owner's objection to a set "
            f"screw in v2.</p>"
            f"<p>The plate is part of the standoff-fork and is printed with it. The plate is a cantilever of printed PLA (E about 2 GPa): "
            f"6.05 mm thick, 19 mm wide and 19.5 mm from its root to the lower screws. Under 5 N it bends about 0.02 mm. The run of 22 October measured "
            f"1.3 mm of extra travel at 5 N. The slotted arm's M3 now goes into a brass insert, so it holds a higher clamping force than the "
            f"self-formed thread of v3. The adapter and the clamp plate are v3's.</p>"
            + table(["Feature", "Value", "Note"], rows, num_cols=(), caption="The stop plate and its screws (F frame).", n=1)
            + fig(1, D / f"media/{PFX}_stop.png", "The stop plate behind the DIP link's tongue, seen from the dorsal side, "
                  "with the three set screws and their jam nuts. DIP link b v4 with cover 3S, pad and tip cap TW15H3S.")
            + fig(2, D / f"media/{PFX}_stop_sections.png", "Sections through the set screws (M frame, the tool axis up, the "
                  "palmar side to the right). The screws reach from the plate to the tongue's back, behind the contact point "
                  "of a press on the face (red arrow)."))
    S.append(("stop", "Stop plate behind the tongue", body))

    axes = (("about_x", "About x (the face's normal)"), ("about_y_pip", "About y (the PIP axis)"), ("about_z_tool", "About z (the tool axis)"))
    trows = []
    for k, lab in axes:
        for sg, nm in (("-", "the sense a face press drives") if k == "about_y_pip" else ("-", "one sense"), ("+", "other sense")):
            trows.append([f"{lab}, {nm}", deg(t3[f'{k}_{sg}']), deg(t4[f'{k}_{sg}']), deg(t4s[f'{k}_{sg}'])])
    ins = all(R["chk"]["b_links"][k]["insertion_clear"] for k in ("v2b", "v3b", "v4b"))
    pos = R["chk"]["poses_table"]
    pose_ok = all(all(v == 0.0 for kk, v in pos[f"{kind}_{ps}"].items() if kk != "socket_floor_minus_pin_top_mm")
                  for kind in ("sphere", "rod") for ps in ("touch", "homing"))
    crow = [["Mount", "3.0", f"{clr['v3']['mount']['registered_mm']:.2f} ({clr['v3']['mount']['registered_at']})",
             f"{clr['v4']['mount']['registered_mm']:.2f} ({clr['v4']['mount']['registered_at']})",
             f"{clr['v4']['mount']['nominal_mm']:.2f}"],
            ["Stop plate alone", "3.0", "&#8211;", f"{clr['v4_stop_plate']['registered_mm']:.2f} ({clr['v4_stop_plate']['registered_at']})",
             f"{clr['v4_stop_plate']['nominal_mm']:.2f}"],
            ["Finger", "0.8", f"{clr['v3']['finger']['registered_mm']:.2f}", f"{clr['v4']['finger']['registered_mm']:.2f}",
             f"{clr['v4']['finger']['nominal_mm']:.2f}"],
            ["HEX-E", "6.0", f"{clr['v3']['hex_e']['registered_mm']:.2f}", f"{clr['v4']['hex_e']['registered_mm']:.2f}",
             f"{clr['v4']['hex_e']['nominal_mm']:.2f}"]]
    st = R["step"]["standoff_fork"]
    body = (f"<p>The links b v2, v3 and v4 all slide onto the fork along the tool axis without overlap: insertion "
            f"{'clear' if ins else 'blocked'} in steps of 0.25 mm over 30 mm. The tongue's back stands "
            f"{c4['gaps_by_region_mm']['tongue_back_to_back_plate']:.2f} mm from the plate and the neck's back "
            f"{c4['gaps_by_region_mm']['neck_back_to_back_plate']:.2f} mm. The legs' gaps to their walls are unchanged at "
            f"{c4['gaps_by_region_mm']['horn_leg_outline_to_walls']:.2f} and {c4['gaps_by_region_mm']['slotted_leg_outline_to_walls']:.2f} mm. "
            f"With the screws modelled 0.02 mm off the tongue, the turn that a face press drives falls from "
            f"{t3['about_y_pip_-']:g}° to {t4s['about_y_pip_-']:g}° (Table 2). The turn about the tool axis falls from "
            f"{t3['about_z_tool_+']:g}° to {t4s['about_z_tool_+']:g}° and {t4s['about_z_tool_-']:g}°. The turn about x, which "
            f"flank presses drive, stays at {t4s['about_x_+']:g}°.</p>"
            + table(["Turn of the link", "v3", "v4, screws backed off", "v4, screws set"], trows, num_cols=(1, 2, 3),
                    caption="Turn to contact in degrees, DIP link b v4.", n=2)
            + f"<p>Along the calibration plan of 22 October, with sphere2's registration, the stop plate stays "
              f"{clr['v4_stop_plate']['registered_mm']:.1f} mm from the indenter and the cell (Table 3). The mount's least "
              f"clearance is at grid-027 in both versions, on the part that v4 shares with v3. There it reads "
              f"{clr['v4']['mount']['registered_mm']:.2f} mm for v4 and {clr['v3']['mount']['registered_mm']:.2f} mm for v3; "
              f"the tool's surface samples, about 2 mm apart, set that difference. The run of 22 October went through at that "
              f"clearance. At homing and at the touching pose of both indenters, "
              f"{'no part overlaps the indenter' if pose_ok else 'a part overlaps the indenter'}. Read back, the STEP file has {st['readback_solids']} solid of "
              f"{st['readback_volume_mm3']:.0f} mm³.</p>"
            + table(["Tool part", "Needed (mm)", "v3, registered (mm)", "v4, registered (mm)", "v4, nominal (mm)"], crow,
                    num_cols=(1, 2, 3, 4), caption="Least clearance over the plan of 22 October (reg2144).", n=3))
    S.append(("checks", "Fit, play and clearance checks", body))

    prow = [["Standoff-fork v4", "1", "PLA on the P1S, the 3MF named above"],
            ["M3 x 5.7 brass heat-set insert (4.6 mm across the knurls)", "4", "three in the stop plate, one in the inner block"],
            ["M3 x 12 set screw, flat point, brass", "3", "A2 if brass is not at hand; it sits about 4.5 mm behind the board's chips"],
            ["M3 brass nut", "3", "jam nuts on the set screws"],
            ["Adapter, clamp plate and their screws", "&#8211;", "v3's, as printed"]]
    body = (f"<p>The fork prints alone on the P1S at Bambu's 0.20 mm Standard with 15 % grid and six walls: {pl['time']}, "
            f"{pl['filament_g']:.1f} g, fill ratio {fork['fill_ratio']:.3f} ({code(D / ('print/' + PFX + '_p1s_pla.3mf'))}). Six "
            f"walls, against v3's four, put the inserts in solid plastic and print the plate nearly solid. The stop plate stands "
            f"upright on top of the fork as printed and bridges 5.5 mm over the cable slot. The fork weighs "
            f"{it4['standoff_fork']['mass_g']:.1f} g against {it3['standoff_fork']['mass_g']:.1f} g for v3. With the screws, "
            f"nuts and inserts the tool gains about {dm + hw:.0f} g. The UR5 program's payload therefore goes from 0.384 kg to "
            f"about {0.384 + (dm + hw) / 1000:.3f} kg, and its centre of mass about 2 mm further along the tool axis.</p>"
            + table(["Part", "Quantity", "Note"], prow, num_cols=(1,), caption="Parts for v4.", n=4))
    S.append(("print", "Print, parts and payload", body))

    steps = ["Press the four inserts with a soldering iron: three into the stop plate's outer face, one into the inner block's "
             "face on the slotted side. Leave each 0.2 mm below its face; the chamfer at the mouth takes the plastic the insert "
             "pushes up.",
             "Seat the fork on the adapter as in v3 (M6 x 20 captive, four M3 x 12 into the adapter's inserts).",
             "Run the three set screws into their inserts, their tips about 1 mm short of the plate's inner face: the "
             "link's neck passes that face 0.5 mm off as the link goes on.",
             "Slide DIP link b onto the fork along the tool axis as in v3. Put on the clamp plate (two M3) and the slotted "
             "arm's M3 x 12 with its washer, now into the insert, and tighten.",
             "Turn each set screw with a hex key held between two fingers until it stops on the tongue. Turn the screw at "
             "z_F -80.5 first and then the two lower ones. Do not push the link: the screws only close the 1.95 mm gap.",
             "Hold each screw with the key and tighten its jam nut against the plate.",
             "Re-run the pad touch-offs and the registration: the link may sit up to 1° differently from v3."]
    body = "<ol>" + "".join(f"<li>{x}</li>" for x in steps) + "</ol>"
    S.append(("assembly", "Assembly", body))

    body = ("<p>Before the first run on v4, a dial indicator on the DIP link's dorsal side measures its travel. A rigid "
            "part presses the pad at 1 to 5 N, once on v3 and once on v4. Then the reference press of the calibration program "
            "gives the same comparison on the robot. Its depth to 2 N was 0.71 to 0.76 mm on v3. On v4 it should fall toward "
            "the FE pad's indentation, about 0.27 mm, plus the HEX-E's 0.08 mm. A depth still above 0.5 mm places the rest of "
            "the compliance outside the DIP link's seat: in the arm, the HEX-E or the indenter's block.</p>")
    S.append(("proof", "Proof on the bench", body))

    body = ("<p>The stop acts on presses that push the tongue toward the dorsal side: the face and, in part, the edges. A "
            "flank press loads the link sideways, which the set screws do not hold, and the turn about x keeps v3's "
            f"{t4s['about_x_+']:g}°. The screws are set by feel, so a heavy hand can push the link within its play before the "
            "nuts lock. The plate's stiffness is a beam estimate. Printed PLA creeps under a held load, and "
            "the screws may need setting again after a long session. The design has been checked in simulation and is "
            "not yet printed.</p>")
    S.append(("limits", "Limitations", body))
    fw = ["Print the fork, assemble it as above and measure the link's travel with a dial indicator at 1, 2 and 5 N on v3 "
          "and v4. A travel on v4 above a third of v3's at 5 N means the plate or its inserts give.",
          "Run sphere2's grid again on v4 and refit the indentation beyond the FE pad with "
          + code(ROOT / "scripts/ur5_tip_calibration_fe.py") + ". A face slope still above 0.1 mm/N places the compliance "
          "outside the link's seat.",
          "If the flank slope (0.14 mm/N on v3) stays, close the walls around the legs from 0.15 to 0.08 mm on a test print. "
          "Flank presses turn the link about x within those gaps."]
    S.append(("future", "Future work", "<ol>" + "".join(f"<li>{x}</li>" for x in fw) + "</ol>"))
    return S, lede, summary


def main():
    R = load()
    S, lede, summary = sections(R)
    toc = "\n".join(f'<li><a href="#{sid}">{k}. {esc(hd)}</a></li>' for k, (sid, hd, _) in enumerate(S, 1))
    secs = "\n".join(f'<section id="{sid}"><h2>{k}. {esc(hd)}</h2>{body}</section>' for k, (sid, hd, body) in enumerate(S, 1))
    head_film = film(0, D / f"media/{PFX}_turntable.mp4", D / f"media/{PFX}_turntable.png",
                     "Mount v4 on the adapter with DIP link b v4, turned about the tool axis. The stop plate with its three "
                     "brass set screws stands behind the tongue.").replace("Figure 0. ", "")
    byline = (f"Workstation, built from {code(D)} by {code(ROOT / 'scripts/ur5_mount_v4_page.py')}; geometry from "
              f"{code(ROOT / 'scripts/ur5_calibration_mount.py')} with UR5M_REV=v4. Mount v3: {link(PAGES['v3'], 'its page')}; "
              f"the measurement that led here: {link(PAGES['results'], 'UR5 ground truth and predictors')}; "
              f"status: {link(PAGES['status'], 'status page')}.")
    page = TEMPLATE.read_text()
    for k, v in {"TITLE": "UR5 mount v4", "H1": "Calibration mount v4 with a stop behind the DIP link's tongue",
                 "EYEBROW": "Printed fingertips and force sensing &#183; 2026-10-23", "LEDE": lede, "BYLINE": byline,
                 "FILM": head_film, "TOC": toc, "SECTIONS": secs, "FOOTER": str(PAGE.relative_to(ROOT)),
                 "TEXCSS": texsvg.CSS}.items():
        page = page.replace("{{" + k + "}}", v)
    items = [((m.group(1) if m.group(1) is not None else m.group(2)).strip(), m.group(1) is not None)
             for m in TEX_RE.finditer(page)]
    svgs = iter(texsvg.render(items, cache_path=TEX_CACHE))
    page = TEX_RE.sub(lambda m: next(svgs), page)
    page = retro_style.apply(page)
    page = docs_index.tag(page, topic="fingertip", summary=summary)
    PAGE.write_text(page)
    print(f"{PAGE.relative_to(ROOT)}: {PAGE.stat().st_size / 1e6:.2f} MB, {len(items)} formulas", flush=True)


if __name__ == "__main__":
    main()
