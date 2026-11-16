"""Text of the gantry travel page (scripts/hardware/gantry_travel_page.py page): every number from its data file."""
from __future__ import annotations

import sys

from gantry_travel_page import DJ, MEDIA, ORDER, PAGE, ROOT, SNS, TEMPLATE, fig, film_html, label, moves, table


def page():
    sys.path.insert(0, str(ROOT / "scripts/docs"))
    import docs_index
    import retro_style
    M = moves()
    J = {s: m["solves"]["joint"] for s, m in M.items() if "joint" in m["solves"]}
    ix, ty1, ty2, tx = J["170337"], J["170626"], J["171108"], J["171309"]
    iy, mx = M["170427"]["solves"][SNS[0]], M["170735"]["solves"][SNS[1]]
    st = J["165655"]
    j4a, j4b = M["170813"]["solves"][SNS[1]], M["171230"]["solves"][SNS[0]]
    turns = [ix["turn_deg"], ty1["turn_deg"], ty2["turn_deg"], tx["turn_deg"]]
    edges = [e for v in J.values() for e in v["edge_mm"]]
    sd_one = [v[k] for m in M.values() for key, v in m["solves"].items() if key != "joint"
              for k in ("sd_before_mm", "sd_after_mm")]
    sd_two = [v[k] for v in J.values() for k in ("sd_before_mm", "sd_after_mm")]
    diffs = [m["solves"][sn]["along_mm"] - m["solves"]["joint"]["along_mm"] for s, m in M.items() if s in J
             and J[s]["along_mm"] > 20 for sn in SNS if sn in m["solves"]]

    lede = (f"With each AprilTag solved from both cameras at once, three stages of the SR2 hand travelled 0.99 to 1.00 "
            f"times their commanded distance. The index's x stage moved {ix['along_mm']:.2f} mm of 60, the thumb's y stage "
            f"{ty1['along_mm']:.2f} and {ty2['along_mm']:.2f} mm of 110, and its x stage {tx['along_mm']:.2f} mm of 60. The middle "
            f"finger's palm-y stage J4 moved {j4a['along_mm']:.2f} and {j4b['along_mm']:.2f} mm when commanded 60 mm each "
            f"way, so its motor looks unpowered. Every axis seen by both cameras turned {min(turns):.1f} to "
            f"{max(turns):.1f}° about the palm normal, which puts the hand about 2° in yaw off the tool chain's palm "
            f"frame. The tags rode on the backs of the DIP links while the station drove each stage to its far end and "
            f"back.")
    summary = (f"Gantry travel from DIP-link tags, both cameras: index x {ix['along_mm']:.2f} of 60 mm, thumb y "
               f"{ty1['along_mm']:.2f}/{ty2['along_mm']:.2f} of 110, thumb x {tx['along_mm']:.2f} of 60; the middle's "
               f"palm-y stage J4 moved nothing; the axes turned {min(turns):.1f} to {max(turns):.1f}° about the palm "
               f"normal (the hand about 2° in yaw off P).")

    S_ = []
    S_.append(("terms", "Terms and metrics", """
<dl class="terms">
<dt>Stage</dt><dd>One of the six linear axes that place a finger's base in the palm plane. J0 to J5 name their stepper
channels on the stepper board. J0 and J1 move the thumb along y and x; J2 and J3 move the index, and J4 and J5 the middle,
in the same order (the station's palm coordinates).</dd>
<dt>Palm frame P</dt><dd>The hand's frame from the tool chain, the CAD on the arm's flange
(<code>sr2hand.robot.tool_chain.SR2Hand</code>). Its x<sub>P</sub> points toward the index and middle, y<sub>P</sub> from
the middle toward the index, and z<sub>P</sub> along the palm normal. The station's palm coordinates are P's axes moved by
a translation.</dd>
<dt>Still windows</dt><dd>The 1.5 s before a command, and the 2 s that start 1 s after the move's end in the station's
state.</dd>
<dt>Travel</dt><dd>The tag's displacement between the two still windows (mean over each window), projected on the
commanded direction, in mm.</dd>
<dt>Scale</dt><dd>Travel divided by the commanded distance.</dd>
<dt>Turn</dt><dd>The angle about z<sub>P</sub> from the commanded direction to the measured displacement,
counterclockwise positive, in degrees.</dd>
<dt>Height change</dt><dd>The displacement along z<sub>P</sub>, in mm.</dd>
<dt>Both cameras</dt><dd>The tag's pose solved from both cameras' corners at once
(<code>sr2hand.tracking.pose.solve_multi</code>), over frame pairs less than half a frame (33 ms) apart. The two views
set its depth.</dd>
<dt>One camera</dt><dd>One camera's own solve of the tag (IPPE, then Levenberg-Marquardt), as logged. It takes the tag's
distance from the printed size assumed, 14.0 mm.</dd>
<dt>Tag edge</dt><dd>The mean edge of the tag's black square from triangulating its four corners in both views, in mm.</dd>
<dt>Window sd</dt><dd>The standard deviation of the tag's position over a still window (the norm over x, y and z).</dd>
</dl>"""))

    S_.append(("setup", "Measurement", f"""
<p>The owner stuck tag36h11 stickers with a 14.0 mm black square on the back of each DIP link: thumb 24, index 25 and
middle 26 (<code>assets/tracking/20261115-dip_link_tags/</code>). With servo torque off, the owner posed the fingers by
hand. The index and middle were flexed so that their DIP backs faced the cameras, and the thumb was kept clear of them.
The station adopted the stepper board's home, which sets each stage's counter to 0 at its home end. The arm stood still, and
its TCP pose was read once.</p>
<p>The corvus lab session ran <code>lab/gantry_follow.py move</code> for each stage. It commanded one finger's mount to
the far end of its travel through the station's manual mount move, then back. Both D435s ran at 1920 × 1080 and 15 fps
with the calibration's fixed exposure, and every frame's tag poses were logged per camera. All three tags were in view
of camera {SNS[0]}. The thumb hid the middle's tag from camera {SNS[1]} during part of the session, which lasted from 16:53
to 17:13.</p>"""))

    head = ["Move", "Stage", "Commanded (mm)", "Both cameras (mm)", "Scale", "Turn (°)", "Height change (mm)",
            f"Camera {SNS[0][-4:]} alone (mm)", f"Camera {SNS[1][-4:]} alone (mm)", "Note"]
    body = []
    for s in ["165655", "170256"] + [x for x in ORDER if x != "165655"]:
        m = M[s]
        jn = m["solves"].get("joint")
        one = [f"{m['solves'][sn]['along_mm']:.2f}" if sn in m["solves"] else "–" for sn in SNS]
        body.append([f"{s[:2]}:{s[2:4]}:{s[4:]}", label(m), f"{m['cmd_mm']:.0f}",
                     f"{jn['along_mm']:.2f}" if jn else "–", f"{jn['scale']:.4f}" if jn and jn["along_mm"] > 20 else "–",
                     f"{jn['turn_deg']:+.2f}" if jn and jn["along_mm"] > 20 else "–",
                     f"{jn['dz_mm']:+.2f}" if jn and jn["along_mm"] > 20 else "–", *one, m["note"] or ""])
    tab = table(head, body, num=(2, 3, 4, 5, 6, 7, 8), n=1,
                caption="Every move of the session. The stage's channel is in brackets. Scale, turn and height change "
                        "are given where both cameras saw the tag before and after the move. A dash: not seen in a window.")

    S_.append(("travel", "Travel of the stages", f"""
<p>Three stages were seen by both cameras before and after their moves. The index's x stage (J3) travelled
{ix['along_mm']:.2f} mm of 60, a scale of {ix['scale']:.4f}. The thumb's y stage (J0) travelled {ty1['along_mm']:.2f} mm
of 110 going out and {ty2['along_mm']:.2f} mm coming back. The thumb's x stage (J1) travelled {tx['along_mm']:.2f} mm of
60. Over each still window the jointly solved tag varied by at most {max(sd_two):.3f} mm (window sd), so these
travels are known to about 0.02 mm.</p>
<p>The index's y move and the middle's x move were in only one camera's view in both windows. The index's y stage
travelled {iy['along_mm']:.2f} mm of 60 by camera {SNS[0][-4:]}, and the middle's x stage {mx['along_mm']:.2f} mm by camera
{SNS[1][-4:]}. These two stages are known only to about 2 mm, the error of one-camera travels (Section 6). The first index x move stopped at {st['along_mm']:.2f} mm: the limp fingertip caught the stepper rail.
The owner pushed the carriage home, and the move repeated in full at 17:03:37.</p>
<p>Along z<sub>P</sub> the index's tag moved {ix['dz_mm']:+.2f} mm over its 60 mm. The thumb's tag moved
{ty1['dz_mm']:+.2f} mm over 110 mm going out and {ty2['dz_mm']:+.2f} mm coming back, and {tx['dz_mm']:+.2f} mm on its
x move.</p>
{tab}"""))

    S_.append(("j4", "The middle finger's palm-y stage J4", f"""
<p>J4 was commanded from −85 to −25 mm and back. The middle's tag moved {j4a['along_mm']:.2f} and {j4b['along_mm']:.2f}
mm, within one camera's noise. The station's step count reached 60 mm both ways. The motor made no sound. The owner turns
its lead screw freely by hand, so the motor looks unpowered (its plug, its cable or its driver). Its counter was reset to
0, and the middle sits at its home end, palm y −85 mm. J4 passed the station's homing. An axis that homes on its timeout
(result 3) is marked homed whether it moved or not.</p>
<p>Each plan of the first HOM series puts the middle's palm y between −25.0 and −81.6 mm
(<a href="../20261115-hom_bench_trials/README.md">HOM trials runbook</a>, blocker 0). No layout can be set until J4
moves.</p>"""))

    S_.append(("yaw", "Yaw of the hand against the palm frame", f"""
<p>Each jointly solved move turned clockwise about z<sub>P</sub> (Figure 3). The thumb's three moves turned
{ty1['turn_deg']:+.2f}, {ty2['turn_deg']:+.2f} and {tx['turn_deg']:+.2f}°, and the index's x move {ix['turn_deg']:+.2f}°.
A yaw of the whole hand against P turns every stage alike, and the thumb's y and x stages agree to
{abs(ty1['turn_deg'] - tx['turn_deg']):.2f}°. The tracker's P is then about 2° in yaw off the hand. Two sources fit this:
the hand's clocking on the HEX-E against <code>assets/cad_v3/frames.json</code>, or a yaw in the cameras' hand-eye poses.
Tag 6 on the hand's tag plate can separate them.</p>
<p>With the palm level, gravity lies along z<sub>P</sub>, and a yaw about z<sub>P</sub> leaves its direction in P
unchanged. A 2° yaw moves a point 50 to 90 mm from the z<sub>P</sub> axis, such as a finger's base, by 1.7 to 3.1 mm along
its tangent.</p>"""))

    S_.append(("cameras", "One camera against two", f"""
<p>One camera's solve of a small tag takes the tag's distance from its apparent size, and the distance scales every
lateral move. On the moves seen by both cameras, one-camera travels differed from the joint travel by {min(diffs):+.2f}
to {max(diffs):+.2f} mm. Their window sd was {min(sd_one):.2f} to {max(sd_one):.2f} mm, against {min(sd_two):.3f} to
{max(sd_two):.3f} mm for the joint solve. Triangulated from both views, the tags' edges measured {min(edges):.2f} to
{max(edges):.2f} mm, so the print's size accounts for at most {100 * max(abs(e / 14.0 - 1) for e in edges):.1f} % of a
travel. The rest is the depth error of one view of a 14 mm tag at 300 to 460 mm. The scale errors of 1 to 2 % in the
live one-camera numbers came from it. Scale and turn for a stage need both cameras.</p>"""))

    S_.append(("future", "Future work", """
<ol>
<li>J4: check the motor's plug and its cable at both ends. If the driver on the stepper board is dead, move the motor to
a spare slot (6 of the board's 8 are used) and change the pin map. Check with <code>lab/gantry_follow.py move --finger
middle --y -25</code>: a travel under 59 mm with both cameras means the stage still does not move.</li>
<li>An open-load check of each stepper driver after homing (the TMC drivers' DRV_STATUS flags) in the station, which
would have flagged J4 at startup.</li>
<li>Tag 6: its pose from both cameras against T<sub>BE</sub> T<sub>EP</sub> and its CAD pose in P. A 2° yaw of tag 6 puts
the error in the hand's clocking (correct <code>frames.json</code>); none puts it in the hand-eye poses.</li>
<li>Measure the index's y and the middle's x with both cameras. Pose the fingers, or move camera 836612071918, so that
tags 25 and 26 are in both views. Then run <code>lab/gantry_travel.py sweep</code>, which also gives the backlash every
5 mm.</li>
<li>The mechanical stops lie 2.6 to 4.8 mm beyond the station's 60 mm box at one end of each stage in the CAD
(<code>frames.json</code> stops_disp_mm). Commanding past the box needs a change of the driver's limits; the tags would
show where each stage stops.</li>
</ol>"""))

    S_.append(("files", "Data, rerun and files", f"""
<p>The fit, every move's still-window poses, travels and per-camera tracks, is <code>data/{DJ.name}</code> in this
folder. The follow logs (17 files, 7.3 MB) are on corvus and on the workstation in
<code>logs/20261115-gantry_travel/follow/</code>, outside git.</p>
<pre class="cmd">rsync -a corvus:workspace/hand/logs/20261115-gantry_travel/follow logs/20261115-gantry_travel/
~/.claude/bin/resguard.sh run --mem 2G -- ~/miniconda3/bin/python scripts/hardware/gantry_follow_fit.py \\
    --json docs/experiments/20261115-gantry_travel/data/{DJ.name}
~/.claude/bin/resguard.sh run --mem 2G -- .venv/bin/python scripts/hardware/gantry_travel_page.py figs
~/.claude/bin/resguard.sh run --mem 2G -- .venv/bin/python scripts/hardware/gantry_travel_page.py page</pre>
<p>Lab: <code>lab/gantry_follow.py</code> (the session's moves, written by the corvus lab session) and
<code>lab/gantry_travel.py</code> (look, pose and sweep). Fits: <code>scripts/hardware/gantry_follow_fit.py</code>
and <code>scripts/hardware/gantry_travel_fit.py</code>. Tags: <code>scripts/hardware/dip_tag_print.py</code>. This page:
<code>scripts/hardware/gantry_travel_page.py</code>.</p>"""))

    filmfig = film_html(1, MEDIA / "20261115-gantry_tags_film.mp4", MEDIA / "20261115-gantry_tags_film_poster.jpg",
                        "The session's moves seen from above the palm, replayed from the tags' tracked positions in P at "
                        "8 times real time. The cameras' images were not saved. Green: the moving finger's tag and its "
                        "path. Grey: the other tags. Dashed: the moving finger's travel box drawn through its tag. Red x: "
                        "the commanded end. The J4 moves show the middle's tag standing still.")
    figs_ = [fig(2, MEDIA / "20261115-travel_traces.png",
                 "Travel of the moving tag along the commanded direction against time, per camera, for each move. "
                 "Dashed: the commanded distance. Above each panel: the travel from both cameras where the tag was in "
                 "both views, else from one camera.",
                 width=1100),
             fig(3, MEDIA / "20261115-axis_turn.png",
                 "Turn of each measured axis from its commanded direction about the palm normal. Circles: both cameras. "
                 "Squares: one camera alone.", width=1100)]
    S_[2] = (S_[2][0], S_[2][1], S_[2][2] + figs_[0])
    S_[4] = (S_[4][0], S_[4][1], S_[4][2] + figs_[1])
    toc = "\n".join(f'<li><a href="#{k}">{i + 1}. {t}</a></li>' for i, (k, t, _) in enumerate(S_))
    secs = "\n".join(f'<section id="{k}"><h2>{i + 1}. {t}</h2>{b}</section>' for i, (k, t, b) in enumerate(S_))
    title = "Gantry travel of the SR2 hand from AprilTags on the DIP links"
    h = TEMPLATE.read_text()
    for k, v in dict(TITLE=title, TEXCSS="", EYEBROW="SR2 hand on the UR5 &middot; finger gantries, toward the HOM trials",
                     H1=title, LEDE=lede,
                     BYLINE="2026-11-15. The owner at the hand with the corvus lab session, 16:53-17:13; fit and page "
                            "on the workstation.",
                     FILM=filmfig, TOC=toc, SECTIONS=secs,
                     FOOTER="Built by <code>scripts/hardware/gantry_travel_page.py</code> from <code>data/</code>.").items():
        h = h.replace("{{" + k + "}}", v)
    h = docs_index.tag(h, topic="hardware", summary=summary, date="2026-11-15")
    h = retro_style.apply(h)
    PAGE.write_text(h)
    print(PAGE, f"{PAGE.stat().st_size / 1e6:.1f} MB")
