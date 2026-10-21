#!/usr/bin/env python3
"""Result page of 2026-10-21: DIP link concept b v4 (the bridge's palmar edge recessed to a smaller column under the pad) and
the sweep of retaining-wall heights (covers I15 with 4, 3 and 2 mm walls, caps with 4, 3 and 2 mm side blocks), after the
owner's notes of 16:20.

Reads docs/experiments/20261018-dip_link_board_mount/params/20261021-v4_concept_b.json, v4/b/20261021-dip_link_v4b_checks.json
and _heights.json (scripts/dip_b_v4.py), the P1S plates' records in v4/b/print/ (scripts/v3b_ur5v2_pla_plate.py heights
link_v4), the readout model's rows of the column cases (docs/experiments/20261019-v2_tip_fe_checks/sensing.json,
scripts/v2_tip_fe_checks.py --sense), applies scripts/retro_style.py and writes
docs/experiments/20261021-dip_link_b_v4/20261021-dip_link_b_v4_and_wall_heights.html.

    .venv/bin/python scripts/dip_b_v4_page.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import docs_index  # noqa: E402
import retro_style  # noqa: E402
import v2_tip_fe_checks_page as VP  # noqa: E402  (table, img_uri, esc)

D = ROOT / "docs/experiments/20261021-dip_link_b_v4"
PAGE = D / "20261021-dip_link_b_v4_and_wall_heights.html"
TEMPLATE = ROOT / "scripts/dip_b_v4_page.template.html"
DIP = ROOT / "docs/experiments/20261018-dip_link_board_mount"
V4 = DIP / "v4/b"
PFX = "20261021-dip_link_v4b"
FE = ROOT / "docs/experiments/20261019-v2_tip_fe_checks"
PAGES = dict(v3=ROOT / "docs/experiments/20261020-ur5_calibration_mount_v2/20261020-dip_link_b_v3_and_ur5_mount_v2.html",
             dip=DIP / "20261018-dip_link_magnetometer_board_mount.html",
             status=ROOT / "docs/experiments/20261018-printed_fingertip_status/20261018-printed_fingertip_and_dip_link_status.html")
table, img_uri, esc = VP.table, VP.img_uri, VP.esc
MM, DEG = "&#8202;mm", "&#176;"


def rel(p):
    return os.path.relpath(p, D)


def link(p, text):
    return f'<a href="{rel(p)}">{text}</a>'


def code(p):
    return f"<code>{esc(os.path.relpath(p, ROOT))}</code>"


def fig(n, png, caption, width=1100):
    return (f'<figure id="fig{n}"><img src="{img_uri(png, width)}" alt="{esc(caption.split(".")[0])}">'
            f"<figcaption>Figure {n}. {caption}</figcaption></figure>")


def f2(v, d=2):
    return "&#8211;" if v is None else f"{0.0 if abs(v) < 0.5 * 10 ** -d else v:.{d}f}"


def load():
    j = lambda p: json.loads(Path(p).read_text())                              # noqa: E731
    par = j(DIP / "params/20261021-v4_concept_b.json")
    pv = {q["key"]: q["value"] for q in par["parameters"]}
    chk = j(V4 / f"{PFX}_checks.json")
    hts = j(V4 / f"{PFX}_heights.json")
    plates = {k: j(V4 / f"print/{PFX}_p1s_pla_{k}.json") for k in ("heights", "link")}
    sens = {r["case"]: r for r in j(FE / "sensing.json")["rows"]}
    rev2 = j(DIP / "v3/b/20261021-dip_link_v3b_solid_covers_and_caps2.json")
    return pv, chk, hts, plates, sens, rev2


def point(sens, case, y, dz):
    for yy, zz, f in sens[case]["gain"]["points_y_dz_force_N"]:
        if abs(yy - y) < 1e-6 and abs(zz - dz) < 1e-6:
            return f
    return None


def ptime(rec):
    return rec["time"].replace(" h", "&#8202;h").replace(" min", "&#8202;min")


def main():
    pv, chk, hts, plates, sens, rev2 = load()
    col, des = chk["column"], chk["design"]
    st = chk["stops"]
    fo = chk["flat_object"]["summary"]
    neck = {k: v[0] for k, v in chk["neck_sections"].items()}
    first = lambda kk, end: min(st[kk][end]["first_contact"].values(), key=lambda v: abs(v["q_deg"] - st[kk][end]["range_end_q_deg"]))  # noqa: E731
    fl3, fl4 = first("v3", "flexion"), first("v4", "flexion")
    ex3, ex4 = first("v3", "extension"), first("v4", "extension")
    cov, caps = hts["covers"], hts["caps"]
    ph, pl = plates["heights"], plates["link"]
    cases = ("bm_five_d75_cap_face", "bm_five_d75_cap_colv3", "bm_five_d75_cap_colv4")
    have_fe = all(c in sens for c in cases)
    c0 = {c: sens[c]["gain"]["centre_force_N"] for c in cases} if have_fe else {}
    rim = {c: point(sens, c, 0.0, -10.0) for c in cases} if have_fe else {}

    lede = (f"DIP link b v4 cuts the bridge between the clevis arms back from x_F &#8722;10.55{MM}, flush with the pad's apex, to "
            f"{des['bridge_x_palmar_F']:.2f}{MM}, and shrinks the column under the pad to {des['col_margin_mm']:.1f}{MM} inside the pad's "
            f"mount face on the palmar side. The pad now stands {fo['v4']['lead_head_on_mm']:.1f}{MM} proud of the DIP link toward a flat "
            f"object (v3: {abs(fo['v3']['lead_head_on_mm']):.1f}{MM}); the MCP link's palmar face, unchanged, stays level with the pad at PIP 0{DEG}. "
            f"The joint still stops on the horn arm at {fl4['q_deg']:.0f}{DEG}. "
            + (f"Near the mount face the readout resolves {rim['bm_five_d75_cap_colv4']:.2f}&#8202;N, against {rim['bm_five_d75_cap_colv3']:.2f}"
               f"&#8202;N on v3's column. " if have_fe else "")
            + f"Covers I15 with 2&#8211;4{MM} walls and caps with 2&#8211;4{MM} blocks print on one P1S plate in {ptime(ph)}.")

    S = []
    S.append(("terms", "Terms and metrics", f"""
<dl class="terms">
<dt>F</dt><dd>The DIP study's finger frame, in mm: x toward the dorsal side, y along the PIP axis, z up, the straight finger hanging
along &#8722;z with the PIP axis through (0, y, &#8722;51.55). The pad's apex, its most palmar point, lies at x_F &#8722;10.55, z_F &#8722;87.91;
its mount face, against the column, at z_F &#8722;77.05; the bridge's underside at z_F &#8722;67.30.</dd>
<dt>Bridge</dt><dd>The DIP link's plate between the two clevis arms (y_F &#8722;17.05 to 13.40), from the arms' mount plane up to
0.8{MM} below the volume the MCP link and the PIP servo sweep over PIP &#8722;18 to 92{DEG}. Its palmar face is the long edge the owner
named.</dd>
<dt>Column, column recess</dt><dd>The neck between the pad's mount face and the bridge, palmar of the tongue's front face x_F 3.95.
Its section is the pad's mount-face outline (a half-ellipse 7.39 by 6.44{MM} on a straight back) moved inward by the column recess:
on the palmar side (<code>col_margin</code>) and at the flanks (<code>col_side_margin</code>), in mm. The pad's overhang is how far
its mount face reaches past the column.</dd>
<dt>Lead</dt><dd>How far the pad stands out of the rigid parts toward a flat object approaching from the palmar side:
max over the pad of u&#183;p minus max over the rigid parts of u&#183;p, u the unit normal of the object's face pointing away from the
finger, in mm. A positive lead is how far the pad can be pressed in before a rigid part touches; the pad's finite-element approach
under a 3&#8202;N contact is 0.154{MM}. Rigid parts: the DIP link and cover I15, and with the joint also the MCP link and the PIP
servo at PIP 0{DEG}.</dd>
<dt>Proximal tilt, side tilt</dt><dd>The angle by which u leans from the palmar normal (&#8722;x) toward the proximal side (+z),
or across the finger toward the slotted arm (+y) or the horn arm (&#8722;y), in degrees.</dd>
<dt>First contact past the range</dt><dd>The PIP angle beyond the joint's range (92{DEG} in flexion, &#8722;18{DEG} in extension) at
which a region of the DIP link first comes within 0.05{MM} of the MCP link or the PIP servo (their surfaces sampled at 0.25{MM}, the
link's at 0.3{MM}, PIP in 0.5{DEG} steps). The region touched first is the joint's hard stop.</dd>
<dt>Clearance over the range</dt><dd>Least distance from a part fixed to the DIP link to the MCP link and the PIP servo over PIP
&#8722;18 to 92{DEG} in 0.5{DEG} steps, in mm.</dd>
<dt>Interference</dt><dd>How far a wall's or side block's inner face lies inside the pad's flank as designed, on each side, in mm.
I15 has 0.15; cap TW2 had 0.05.</dd>
<dt>Wall height, block height</dt><dd>How far a cover's side walls (h) or a cap's side blocks (H) reach toward the palmar side from
the cover's front face x_F 2.90, in mm. The pad's flank is straight for {hts['pad_flank_straight_mm']:.2f}{MM} from its back face,
then curves in; "flank held" is the part of that straight flank a wall faces.</dd>
<dt>Seat</dt><dd>Height of the proximal board screws' heads above the cover's front face, on solid wall, in mm.</dd>
<dt>Thread in the tongue</dt><dd>Length of a board screw inside the DIP link's tongue, from the board's seat x_F 4.40 to its tip,
in mm; the self-forming M2 thread is that long.</dd>
<dt>Force resolution</dt><dd>One standard deviation of the normal force estimated from one sample of the five MLX90393 chips under a
Gaussian patch load (&#963; 0.6{MM}) at a point of the pad's palmar face, with the contact position and a uniform background field
unknown, at the gain the field needs (Cram&#233;r&#8211;Rao bound of the readout model on the finite elements of pad B00 with the walls and
cap held), in N; dz is the point's distance along the finger from the contact point, negative toward the mount face.</dd>
</dl>"""))

    t1 = [["&#8220;the 0.15mm inset is still perfectly fine ... lower the heights on both the top cover and the cassette side walls. "
           "let's print a few out and see how low we can go while maintaining this retaining&#8221;",
           f"Covers I15H4, I15H3, I15H2 (walls 4, 3, 2{MM}) and caps TW15H4, TW15H3, TW15H2 (side blocks 4, 3, 2{MM}), all at "
           f"{hts['interference_mm']:.2f}{MM}; the height engraved on each", f"One P1S plate, {ptime(ph)}"],
          ["&#8220;there's no requirement for the dip link to have a long continuous edge. it just serves as a hard stop for the servo. "
           "but it also prevents large objects from contacting the fingertip ... recess/sculpt this edge ... while making sure our servo "
           "and clevis connections are solid&#8221;",
           f"The bridge's palmar face from x_F &#8722;10.55 to {des['bridge_x_palmar_F']:.2f} with a {des['bridge_chamfer_mm']:.0f}{MM} "
           "chamfer along its top; the slotted arm's palmar spur cut to the same plane; the arms, the horn's bolt circle and the "
           "bridge's top unchanged", f"Stop at {fl4['q_deg']:.1f}{DEG} as on v3; head-on lead {fo['v4']['lead_head_on_mm']:.1f}{MM}"],
          ["&#8220;make the underlying column even smaller radius too. then the edge detections / bottom face can be usable too&#8221;",
           f"Column {des['col_margin_mm']:.1f}{MM} inside the pad's mount face on the palmar side and {des['col_side_margin_mm']:.2f}{MM} at "
           f"the flanks (v3: 0.6 and 0.6)",
           (f"Force resolution 0.9{MM} from the mount face {rim['bm_five_d75_cap_colv4']:.3f}&#8202;N (v3's column "
            f"{rim['bm_five_d75_cap_colv3']:.3f}&#8202;N)" if have_fe else "Finite elements pending")]]
    S.append(("requests", "Requests of 2026-10-21 16:20 and the changes", table(
        ["Owner", "Change", "Check"], t1, caption="The owner's notes after fitting cover I15 and the second caps, and what v4 and the "
                                                    "sweep change.", n=1)))

    v3m, v4m = chk["mass_g"]["v3"], chk["mass_g"]["v4"]
    t2 = [["Bridge's palmar face, x_F", "&#8722;10.55", f"{des['bridge_x_palmar_F']:.2f}"],
          ["Column's palmar apex, x_F", f"{col['v3']['palmar_x_F']:.2f}", f"{col['v4']['palmar_x_F']:.2f}"],
          ["Column's half-width", f"{col['v3']['half_width_mm']:.2f}", f"{col['v4']['half_width_mm']:.2f}"],
          ["Pad's overhang past the column, palmar", f"{col['v3']['palmar_x_F'] - col['pad_mount_face']['palmar_x_F']:.2f}",
           f"{col['v4']['pad_overhang_palmar_mm']:.2f}"],
          ["Pad's overhang past the column, flanks", f"{col['pad_mount_face']['half_width_mm'] - col['v3']['half_width_mm']:.2f}",
           f"{col['v4']['pad_overhang_side_mm']:.2f}"],
          ["PLA beside J1's slot", f"{col['v3']['half_width_mm'] - 3.95:.2f}", f"{col['v4']['pla_beside_j1_mm']:.2f}"],
          ["Least wall in the neck's section at J1 (mm)", "&#8211;", f"{chk['walls']['neck_z-72.30_around_J1']['min_wall_mm']:.2f}"],
          ["Neck's section at z_F &#8722;72.30, area (mm&#178;)", f"{neck['v3']['area_mm2']:.0f}", f"{neck['v4']['area_mm2']:.0f}"],
          ["Its second moment for a palmar load (mm&#8308;)", f"{neck['v3']['I_about_y_mm4']:.0f}", f"{neck['v4']['I_about_y_mm4']:.0f}"],
          ["Its second moment for a load across the finger (mm&#8308;)", f"{neck['v3']['I_about_x_mm4']:.0f}", f"{neck['v4']['I_about_x_mm4']:.0f}"],
          ["Link volume (mm&#179;)", f"{chk['volume_mm3']['v3']:.0f}", f"{chk['volume_mm3']['v4']:.0f}"],
          ["Link mass at the sliced density (g)", f"{v3m['sliced']:.2f}", f"{v4m['sliced']:.2f}"],
          ["Bridge's top, least radius about the PIP axis", f"{chk['bridge_top_least_radius_mm']['v3']:.2f}",
           f"{chk['bridge_top_least_radius_mm']['v4']:.2f}"]]
    S.append(("bridge", "Bridge edge and column", f"""
<p>The bridge spans the 30.5{MM} between the clevis arms. On v3 its palmar face stood at x_F &#8722;10.55, the x of the pad's apex, and
the slotted arm carried a 1.2{MM} spur out to the same plane, so a flat object pressed along the palmar normal met the bridge and the
pad together, 20.6{MM} apart along the finger. v4 moves the bridge's palmar face to the column's apex, x_F {des['bridge_x_palmar_F']:.2f}, chamfers
its top edge {des['bridge_chamfer_mm']:.0f}{MM} at 45{DEG} and cuts the spur back to the same plane (Figure 1). The horn arm, which takes the servo's horn,
reaches only to x_F &#8722;6.31 at the bridge, so its whole joint with the bridge stays; the slotted arm keeps its full outline above
z_F &#8722;62.3. The cut removes {chk['removed_from_v3_mm3']:.0f}&#8202;mm&#179; and leaves nothing outside v3 ({chk['v4_outside_v3_mm3']:.2f}&#8202;mm&#179;), so the
UR5e mount's walls, molded around v2's and v3's legs, take v4 as they are, and the bridge's top, on which the mount's inner block
rests, keeps its least radius about the PIP axis.</p>
<p>The column under the pad now lies {des['col_margin_mm']:.1f}{MM} inside the pad's mount face on the palmar side and
{des['col_side_margin_mm']:.2f}{MM} at the flanks; the flank recess keeps {col['v4']['pla_beside_j1_mm']:.2f}{MM} of PLA on each side of J1's slot, which
runs up through the neck behind the pad. The pad's mount face reaches {col['v4']['pad_overhang_palmar_mm']:.1f}{MM} past the column on the palmar side
(v3: {col['v3']['palmar_x_F'] - col['pad_mount_face']['palmar_x_F']:.1f}{MM}). The neck loses {100 * (1 - neck['v4']['I_about_y_mm4'] / neck['v3']['I_about_y_mm4']):.0f}&#8202;% of its second moment for a palmar load (Table 2);
the board mount behind the column, 5.5{MM} thick and 21{MM} wide, makes up most of it.</p>
{fig(1, V4 / 'media' / f'{PFX}_v3_v4.png', 'DIP link b v3 (left) and v4 (right) with cover I15 and pad B00R, finger up: from the palmar side (top) and from the slotted arm (bottom). On v4 the bridge ends at the column, which is narrower than v3&#8217;s; the slotted arm&#8217;s spur below the bridge is gone.')}
{table(["Quantity (mm unless noted)", "v3", "v4"], t2, num_cols=(1, 2), caption="The bridge and the column of v3 and v4.", n=2)}"""))

    regs = sorted(set(st["v3"]["flexion"]["first_contact"]) | set(st["v4"]["flexion"]["first_contact"])
                  | set(st["v3"]["extension"]["first_contact"]) | set(st["v4"]["extension"]["first_contact"]))
    gq = lambda kk, end, r: st[kk][end]["first_contact"].get(r, {}).get("q_deg")  # noqa: E731
    t3 = [[r] + [f2(gq(kk, end, r), 1) for end in ("flexion", "extension") for kk in ("v3", "v4")] for r in regs]
    clr = chk["clearance_pip_range"]
    S.append(("stops", "Joint stops and clearance", f"""
<p>Past 92{DEG} of flexion the MCP link first touches the horn arm's palmar edge, at {fl3['q_deg']:.1f}{DEG} on v3 and {fl4['q_deg']:.1f}{DEG}
on v4, at x_F {fl4['at_F'][0]:.1f}, y_F {fl4['at_F'][1]:.1f}, z_F {fl4['at_F'][2]:.1f}: the arm from the original link is the hard stop.
v3's long edge was a second stop {gq('v3', 'flexion', 'bridge, palmar of x_F -6') - fl3['q_deg']:.1f}{DEG} later; on v4 the
recessed bridge meets the MCP link {f2(gq('v4', 'flexion', 'bridge'), 1)}{DEG} into flexion. Past &#8722;18{DEG} of extension the bridge's
dorsal top meets the PIP servo at {ex4['q_deg']:.1f}{DEG} on both. The slotted arm's spur never touches within the range or past it.
Over the range the new body keeps {clr['v4']['dip_link_new_body']:.2f}{MM} from the MCP link (v3: {clr['v3']['dip_link_new_body']:.2f}) and the
Qwiic cable {clr['v4']['qwiic_cable']:.2f}{MM}; the arms' own bearing faces, from the original link, sit as before.</p>
{table(["Region of the DIP link", "Flexion, v3 (&#176;)", "Flexion, v4 (&#176;)", "Extension, v3 (&#176;)", "Extension, v4 (&#176;)"], t3,
       num_cols=(1, 2, 3, 4), caption="First contact of each region with the MCP link or the PIP servo past the joint's range (dash: "
                                      "none within 112 or &#8722;36&#176;).", n=3)}"""))

    def frow(kk):
        f = fo[kk]
        return [kk, f2(f["lead_head_on_mm"]), f2(f["lead_head_on_with_joint_mm"]), f2(f["proximal_tilt_lead_ge_1mm_deg"], 0),
                f2(f["proximal_tilt_lead_ge_0_deg"], 0), f2(f["proximal_tilt_with_joint_ge_0_deg"], 0),
                f2(f["side_tilt_lead_ge_0_deg"], 0), f2(f["side_tilt_other_side_lead_ge_0_deg"], 0)]
    S.append(("flat", "Flat objects against the pad", f"""
<p>Seen from a large flat object, v3's pad and bridge stood level: the lead was {abs(fo['v3']['lead_head_on_mm']):.2f}{MM} head-on and turned negative for
any tilt toward the proximal side. On v4 the pad leads by {fo['v4']['lead_head_on_mm']:.2f}{MM} head-on, keeps a lead of 1{MM} or more up to
{fo['v4']['proximal_tilt_lead_ge_1mm_deg']:.0f}{DEG} of proximal tilt and stays ahead up to {fo['v4']['proximal_tilt_lead_ge_0_deg']:.0f}{DEG} (Figure 2, Table 4). Across the finger the clevis arms, 46{MM} wide
against the pad's 14.8{MM}, take over at {fo['v4']['side_tilt_lead_ge_0_deg']:.0f}{DEG} toward the slotted arm and {fo['v4']['side_tilt_other_side_lead_ge_0_deg']:.0f}{DEG} toward the horn arm;
the clevis arms set those limits. The MCP link's palmar face, which we do not change, lies in the apex's plane at PIP 0{DEG} too
(dashed in Figure 2) and swings toward the palmar side as the PIP flexes, so a flat face long enough to reach it, 29{MM} or more
proximal of the apex, still meets it with the pad on both links. The recess gives the pad its lead for shorter faces, for faces tilted
toward the distal side, and whenever the PIP is extended past 0{DEG}.</p>
{fig(2, V4 / 'media' / f'{PFX}_flat_object.png', 'The pad&#8217;s lead over the rigid parts for a flat object approaching from the palmar side, against the tilt of the object&#8217;s face toward the proximal side (left) and across the finger (right). Solid: the DIP link and cover I15; dashed: with the MCP link and the PIP servo at PIP 0&#176;. Above zero the pad touches first.', 1100)}
{table(["Link", "Lead head-on (mm)", "With the joint (mm)", "Proximal tilt, lead &#8805; 1 mm (&#176;)", "Proximal tilt, lead &#8805; 0 (&#176;)",
        "With the joint, lead &#8805; 0 (&#176;)", "Toward the slotted arm, lead &#8805; 0 (&#176;)", "Toward the horn arm, lead &#8805; 0 (&#176;)"],
       [frow("v3"), frow("v4")], num_cols=tuple(range(1, 8)), caption="The pad&#8217;s lead over the rigid parts and the largest tilts "
       "at which it holds (dash: not even head-on).", n=4)}"""))

    if have_fe:
        pts = [(0.0, 0.0), (0.0, -4.0), (0.0, -6.0), (0.0, -8.0), (0.0, -10.0), (3.0, -10.0), (-3.0, -10.0)]
        t5 = [[f"y {y:+.0f}, dz {dz:+.0f}" if y else f"y 0, dz {dz:+.0f}"] + [f2(point(sens, c, y, dz), 3) for c in cases] for y, dz in pts]
        t5.append(["Magnet travel at 1&#8202;N, contact point (&#181;m)"] + [f"{sens[c]['magnet_travel_um_per_N']:.2f}" for c in cases])
        t5.append(["Approach at 1&#8202;N, contact point (&#181;m)"] + [f"{sens[c]['approach_1N_um']:.1f}" for c in cases])
        S.append(("readout", "Readout toward the pad's mount face", f"""
<p>The finite elements of pad B00 under cover I15 and a TW cap (walls and cap held, as in the second revision's bound) hold the pad's
mount face where a column meets it: all of it (as every earlier case), v3's column, or v4's. Toward the mount face the readout model
resolves the normal force more finely the more of the face is free: 0.9{MM} from the mount face on the centre line
{rim['bm_five_d75_cap_face']:.3f}&#8202;N with the whole face held, {rim['bm_five_d75_cap_colv3']:.3f}&#8202;N with v3's column and
{rim['bm_five_d75_cap_colv4']:.3f}&#8202;N with v4's. At the contact point the three give {c0['bm_five_d75_cap_face']:.4f},
{c0['bm_five_d75_cap_colv3']:.4f} and {c0['bm_five_d75_cap_colv4']:.4f}&#8202;N (Table 5).</p>
{table(["Load point", "Whole face held (N)", "v3 column (N)", "v4 column (N)"], t5, num_cols=(1, 2, 3),
       caption="Force resolution along the pad's palmar face with the mount face held three ways (readout model, GAIN_SEL as the "
               "field needs).", n=5)}"""))

    tc = [[cid, f"{r['wall_h_mm']:.0f}", f"{r['flank_held_mm']:.2f}", f"{r['seat_mm']:.0f}", r["proximal_screws"],
           f"{r['thread_in_tongue_mm']:.1f}", f"{r['interference_with_B00R_mm3']:.1f}", f"{r['mass_solid_g']:.2f}"]
          for cid, r in cov.items()]
    i15 = rev2["covers"]["I15"]
    tc.insert(0, ["I15 (printed)", "6", f"{hts['pad_flank_straight_mm']:.2f}", "3", "M2 x 8", "3.5",
                  f"{i15['interference_with_B00R_mm3']:.1f}", f"{i15['mass_solid_g']:.2f}"])
    tk = [[vid, f"{r['block_h_mm']:.0f}", f"{r['interference_mm']:.2f}", r["distal_screws"], f"{r['thread_in_tongue_mm']:.1f}",
           f"{r['interference_sides_B00R_mm3']:.1f}", f"{r['pip_sweep']['min_mm']:.1f}", f"{r['mass_solid_g']:.2f}"]
          for vid, r in caps.items()]
    tw2 = rev2["caps"]["TW2"]
    tk.insert(0, ["TW2 (printed)", "5", "0.05", "M2 x 10", "3.5", "&#8211;", f"{tw2['pip_sweep']['min_mm']:.1f}", f"{tw2['mass_solid_g']:.2f}"])
    S.append(("heights", "Wall-height sweep", f"""
<p>Each cover keeps I15's {hts['interference_mm']:.2f}{MM} interference and solid walls; only their height changes (Table 6). The pad's flank is
straight for {hts['pad_flank_straight_mm']:.2f}{MM} from its back face, so I15's 6{MM} walls and a 5{MM} wall would hold the same flank; 4, 3 and 2{MM} hold
less of it. The proximal heads stay on their 3{MM} seats where the wall is taller, and sit on the wall's top on I15H2, where an M2 x 6
leaves 2.5{MM} of thread in the tongue. The caps keep TW2's plate, pin and chamfers, take I15's interference in place of TW2's 0.05{MM} and
lower the side blocks (Table 7); the distal heads sit on the blocks, M2 x 8 on TW15H4 and TW15H3 and M2 x 6 on TW15H2. The wall or block
height is engraved, {hts['engraving']['depth']:.1f}{MM} deep, on each cover's wall tops and on each cap's end plate (Figure 3). No part
overlaps the link, the board, the magnets or the screws; every STL reads back watertight and every STEP file is one valid solid.</p>
{table(["Cover", "Wall height (mm)", "Flank held (mm)", "Seat (mm)", "Proximal screws", "Thread in the tongue (mm)",
        "Interference with B00R (mm&#179;)", "Mass, solid (g)"], tc, num_cols=(1, 2, 3, 5, 6, 7),
       caption="The covers of the sweep and I15.", n=6)}
{table(["Cap", "Block height (mm)", "Interference (mm)", "Distal screws", "Thread in the tongue (mm)", "Interference with B00R (mm&#179;)",
        "Clearance over the range (mm)", "Mass, solid (g)"], tk, num_cols=(1, 2, 4, 5, 6, 7), caption="The caps of the sweep and TW2.", n=7)}
{fig(3, V4 / 'media' / f'{PFX}_heights.png', 'Covers I15H4, I15H3 and I15H2 (top) and caps TW15H4, TW15H3 and TW15H2 on pad B00R (bottom); the height is engraved on each.', 1100)}"""))

    files_print = [V4 / "print" / f"{PFX}_p1s_pla_heights.3mf", V4 / "print" / f"{PFX}_p1s_pla_link.3mf"]
    S.append(("printing", "Printing and assembly", f"""
<p>Two P1S plates in PLA, as the earlier covers, caps and link (Bambu Studio, 0.1{MM} layers):</p>
<ul>
<li>{code(files_print[0])}: the three covers and three caps, solid, {ptime(ph)}, {ph['filament_g']:.1f}&#8202;g.</li>
<li>{code(files_print[1])}: DIP link b v4 with tree supports, {ptime(pl)}, {pl['filament_g']:.1f}&#8202;g.</li>
</ul>
<p>The plate check flags TW15H2 at a fill ratio of {ph['objects'][-1]['fill_ratio']:.3f} against the 0.85 asked of solid parts (six
walls, 100&#8202;% infill; its 2{MM} blocks leave narrow gaps); the other five parts pass. The link prints without a brim of its own, as v3's
did.</p>
<p>Screws: M2 x 8 for the proximal pair on I15H4 and I15H3, M2 x 6 on I15H2; M2 x 8 for the distal pair on TW15H4 and TW15H3, M2 x 6
on TW15H2. Pad B00R fits every cover and cap of the sweep; a B00 or B00P pad needs its proximal notches cut 3{MM} deeper, as for I15.
The v4 link takes the board, a cover, a pad and a cap as v3 does. On the bench, fit pad B00R in I15H4 with TW15H4 and push the pad
sideways at its apex as hard as popped it off W00, then step down to the 3 and 2{MM} parts; cover and cap can be lowered separately.</p>"""))

    S.append(("future", "Future work", f"""
<ul>
<li>Fit test of the sweep: with pad B00R, push each cover and cap pair sideways at the apex as before, ten pushes per pair, from
H4 down. The lowest pair that keeps the pad on all ten pushes sets <code>wall_h</code> in the parameter file and the cap's block height;
if H2 holds, a 1.5{MM} wall is the next step (<code>COVER_H</code> in {code(ROOT / 'scripts/dip_b_v4.py')}).</li>
<li>Read the five chips with pad B00R on v4 while pressing the pad's palmar face with the UR5e 1{MM} from its mount face at 0.5, 1 and
3&#8202;N, ten times each, and once at the contact point; Table 5 predicts a spread at the edge
{f"{rim['bm_five_d75_cap_colv4'] / c0['bm_five_d75_cap_colv4']:.1f} times" if have_fe else "several times"} that at the contact point.
A spread more than twice that means the column holds more of the mount face than modelled (the printed column's edge, the pad's
notch over J1's slot) and <code>col_margin</code> 3.5 frees more.</li>
<li>Press a flat plate onto the finger with the UR5e along the palmar normal and at 5 and 10{DEG} toward the proximal side, to 3&#8202;N:
the bridge must not mark the plate's paint or foil before the force reaches 3&#8202;N; a mark at 10{DEG} on v4 means the chamfer is too
small (<code>bridge_chamfer</code> 1.5).</li>
<li>Drive the PIP to 92{DEG} and command 3{DEG} further: the horn arm must stop it with no contact on the bridge (a witness mark of
marker ink on the bridge's top); if the bridge marks, the arm deflects more than the 3&#8202;mm-wide spur's loss allows and a stop pad on
the bridge's slotted-arm end is the fix.</li>
<li>When the fingers go on the hand, rebuild the finger scenes with v4's link and run the trajectory-clearance gate: the link only lost
material, so the closest approach of the index and middle links can only grow.</li>
</ul>"""))

    S.append(("limits", "Limitations", f"""
<p>The lead and the stops are geometry: the pad as designed and rigid, the link and the joint as modelled, no deflection of the arms
or of the servo's horn. The readout numbers bound the stiffening from above, since the walls, cap and column hold the pad's nodes in both
directions where the printed parts only push. The sweep's retention is untested; Table 6 gives what each wall faces, not what it
holds.</p>"""))

    files = [DIP / "params/20261021-v4_concept_b.json", V4 / f"{PFX}_checks.json", V4 / f"{PFX}_heights.json", V4 / "cad", V4 / "print",
             V4 / "media", ROOT / "scripts/dip_link_concept_b.py", ROOT / "scripts/dip_link_v2.py", ROOT / "scripts/dip_b_v4.py",
             ROOT / "scripts/dip_b_tip_cap.py", ROOT / "scripts/v3b_ur5v2_pla_plate.py", ROOT / "scripts/v2_tip_fe_checks.py",
             FE / "sensing.json", ROOT / "scripts/dip_b_v4_page.py"]
    S.append(("files", "Files", "<ul>" + "".join(f"<li>{code(f)}</li>" for f in files if Path(f).exists()) + "</ul>"
              + f"<p>Related pages: {link(PAGES['v3'], 'DIP link b v3 and UR5e mount v2')}, {link(PAGES['dip'], 'DIP-link redesign')}, "
                f"{link(PAGES['status'], 'printed-fingertip status')}.</p>"))

    toc, body = [], []
    for k, (sid, title, html_) in enumerate(S):
        num = "" if sid == "terms" else f"{k}. "
        toc.append(f'<li><a href="#{sid}">{num}{esc(title)}</a></li>')
        body.append(f'<section id="{sid}"><h2>{num}{esc(title)}</h2>{html_}</section>')
    out = (TEMPLATE.read_text().replace("{{TITLE}}", "DIP link b v4 and wall-height sweep")
           .replace("{{TEXCSS}}", "").replace("{{EYEBROW}}", "Printed fingertip &#183; 2026-10-21")
           .replace("{{H1}}", "DIP link b v4 and a sweep of retaining-wall heights").replace("{{LEDE}}", lede)
           .replace("{{BYLINE}}", f"Built {time.strftime('%Y-%m-%d %H:%M')} by scripts/dip_b_v4_page.py")
           .replace("{{TOC}}", "\n".join(toc)).replace("{{SECTIONS}}", "\n".join(body))
           .replace("{{FOOTER}}", "Local page; data in the folders named under Files."))
    summary = (f"DIP link b v4 recesses the bridge's palmar edge to x_F {des['bridge_x_palmar_F']:.2f} mm and shrinks the column under the "
               f"pad, so a flat object meets the pad {fo['v4']['lead_head_on_mm']:.1f} mm before any rigid part (v3: "
               f"{abs(fo['v3']['lead_head_on_mm']):.1f} mm); the joint still stops on the horn arm at {fl4['q_deg']:.1f} deg. Covers I15 with 4, 3 "
               f"and 2 mm walls and caps with 4, 3 and 2 mm blocks at 0.15 mm print on one plate for the owner's height test.")
    out = docs_index.tag(out, topic="fingertip", summary=summary, date="2026-10-21")
    out = retro_style.apply(out)
    D.mkdir(parents=True, exist_ok=True)
    PAGE.write_text(out)
    print(f"page -> {PAGE.relative_to(ROOT)} ({PAGE.stat().st_size / 1e6:.2f} MB); lede {len(lede.split())} words")


if __name__ == "__main__":
    main()
