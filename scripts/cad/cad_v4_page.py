#!/usr/bin/env python3
"""Builds docs/experiments/20261118-sr2_actuator_layout/20261118-sr2_finger_actuator_layouts.html: two actuator layouts
for the SR2 finger (cad_v4) against the hand as built (owner, 2026-11-18: "if we wanted to rotate the servos to all be
skinny side facing, durable linkage/transmission design that didnt sacrifice torque ... we dont want to incur any new
collisions/loss of operational workspace"; "is there any benefit to trying to put the mcp flexion and the
abduction/adduction on a single spherical joint?"; "cad v4 can have two proposed designs (or three) if these two ideas
cannot be neatly merged"). Owner, 2026-11-19, on the AmazingHand's differential: "can you verify that the amazing hand
(please look it up) differential is not as good as what youve proposed here? or if it is and im just silly, a way to
re-incorporate their design, but with our degrees of freedom".

Data: the folder's data/20261118-{build,check,replay,forces,joints,render}.json from scripts/cad/cad_v4_layouts.py and
the earlier sphere replay 20261118-index_middle_clearance_replay.json (scripts/mechanisms/sr2_index_middle_clearance_replay.py),
and data/20261119-amazinghand{,_geometry}.json from scripts/cad/amazinghand_differential.py.

    .venv/bin/python scripts/cad/cad_v4_page.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import _paths  # noqa: E402,F401
import docs_index  # noqa: E402
import fp_review_page as H  # noqa: E402
import retro_style  # noqa: E402
import texsvg  # noqa: E402

D = ROOT / "docs/experiments/20261118-sr2_actuator_layout"
DATA, MEDIA = D / "data", D / "media"
OUT = D / "20261118-sr2_finger_actuator_layouts.html"
TEMPLATE = ROOT / "scripts/cad/cad_v3_page.template.html"
TEX_CACHE = D / "texsvg_cache.json"
PARAMS = ROOT / "assets/cad_v4/concepts/20261118-layout_params.json"
table, img, esc = H.table, H.img, H.esc


def load():
    J = {k: json.loads((DATA / f"20261118-{k}.json").read_text())["result"]
         for k in ("build", "check", "replay", "forces", "joints", "render")}
    J["params"] = json.loads(PARAMS.read_text())
    J["sphere_replay"] = json.loads((D / "20261118-index_middle_clearance_replay.json").read_text())
    J["amazing"] = json.loads((DATA / "20261119-amazinghand.json").read_text())
    J["amazing_geom"] = json.loads((DATA / "20261119-amazinghand_geometry.json").read_text())
    J["codesign"] = json.loads((DATA / "20261119-v4c_codesign.json").read_text())
    return J


def mn(x, nd=1):
    """A number with a typographic minus."""
    return f"{x:.{nd}f}".replace("-", "&minus;")


def pct(x, nd=1):
    return f"{100 * x:.{nd}f} %"


def sec_terms(J):
    return """<dl class="terms">
<dt>Yaw, MCP and PIP angles</dt><dd>The three joint angles of a finger in the simulator's convention, in degrees. The yaw
turns the finger about $x$, the axis that points from the finger's palmar side to its dorsal side; the MCP and PIP flex it
about $y$. The driver's nominal ranges are yaw &plusmn;85&deg;, MCP &minus;15&deg; to 92&deg; and PIP &minus;18&deg; to 92&deg;.</dd>
<dt>DIP angle $\\theta$</dt><dd>The angle of the distal link against the yaw link, $\\theta = q_{\\mathrm{MCP}} + q_{\\mathrm{PIP}}$.
In both new layouts the PIP servo sets $\\theta$.</dd>
<dt>Gap</dt><dd>The least distance in mm between the index finger's and the middle finger's collision geometry at one
control period. The geometry is cad_v3's MorphIt spheres, which are larger than the printed parts, plus boxes and capsules for
every new part. A negative gap is an overlap of the spheres. The replay takes every second period of the 65 logged bench
trials of 15 to 17 November, 10,350 periods.</dd>
<dt>Overlap share, share under 5 mm</dt><dd>The fraction of those periods with a gap below 0 and below 5 mm.</dd>
<dt>Self-clearance</dt><dd>The least distance in mm between a finger's new proximal parts (servo housings, cranks,
yoke) and its own moving lower finger. It is taken over a 3&deg; grid of the driver's MCP and PIP ranges. Parts fixed to the
carriage are also checked at five yaw angles.</dd>
<dt>Pinch force at the servo limit</dt><dd>The force along the pad's normal at which the first servo reaches its torque limit
$T$. A servo's lever is the moment arm of that force about its joint. It is $b + a\\cos q_{\\mathrm{PIP}}$ for v3's MCP
servo and $a\\cos q_{\\mathrm{PIP}}$ for an MCP servo whose PIP servo sets $\\theta$. Every PIP servo's lever is $b$. Here
$a$ = 30.8 mm is the distance from the MCP axis to the PIP axis and $b$ = 36.36 mm from the PIP axis to the pad centre.</dd>
<dt>Yaw lever</dt><dd>The distance from the yaw axis to the pad centre, perpendicular to the axis, in mm. A sideways force at
the pad loads the yaw servo through this lever.</dd>
<dt>Transmission angle $\\mu$</dt><dd>For a parallel four-bar or a parallelogram, 90&deg; less the crank's angle from the
position perpendicular to the coupler. The rod's force rises as $1/\\sin\\mu$; 0&deg; is a dead point.</dd>
<dt>Rod differential, horn radius $r$</dt><dd>Two servos in the palm turn horns of radius $r$. A rod with a ball joint at
each end joins each horn to the proximal link beside the joint's centre. Pushing both rods flexes the finger; pushing one and
pulling the other abducts it. A horn's dead points lie 90&deg; to either side of its zero, where the horn's ball moves across
its rod.</dd>
<dt>Reachable region</dt><dd>The abduction and flexion angles at which both rods close, both horns stay within
&plusmn;90&deg; of their zero, and each joint can take at least one servo's torque. The region is taken on a 0.5&deg; grid, and
only its connected part around the rest pose counts.</dd>
<dt>Share of the driver's range</dt><dd>The largest fraction of the driver's yaw and MCP rectangle, &plusmn;85&deg; by
&minus;15&deg; to 92&deg;, that one placement of a reachable region covers. A placement turns the whole finger module, servos
included, about the yaw and MCP axes.</dd>
<dt>Torque floor</dt><dd>The least joint torque a design must keep everywhere in its reachable region. For v4c it is the
current hand's, one SCS0009 per joint: 0.51 STS3032 limits, from the sheets' stall torques at 6 V (2.3 and 4.5 kg cm).</dd>
<dt>Joint torque per servo limit</dt><dd>The largest torque about one joint, with no torque about the other, when each servo
gives at most its limit $T$, divided by $T$. One servo per joint gives 1; a bevel differential at 1:1 gives 2.</dd>
<dt>Differential ratio $g$, motor range $\\Theta$</dt><dd>Two servos can drive the yaw
and the MCP together through a bevel differential. Then $g$ is a servo's turns per joint turn, and $\\Theta$ is the servo's
travel: 300&deg; for the SCS0009 and 360&deg; for the STS3032 in position mode.</dd>
</dl>"""


def sec_layouts(J):
    P, B, C = J["params"], J["build"], J["check"]
    a, b = B["v4a"]["index"], B["v4b"]["index"]
    K3, Ka, Kb = C["carriage"]["v3"]["index"], C["carriage"]["v4a"]["index"], C["carriage"]["v4b"]["index"]
    V4b = P["v4b"]
    rows = [
        ["PIP servo", "in the MCP link, across the finger", "STS3032 in a housing on the yaw link, behind the MCP servo",
         "STS3032 on the yoke, right behind the MCP servo"],
        ["MCP servo", "in the yaw link, across the finger", "as built", "on the yoke, upright, shaft through the gimbal centre"],
        ["Yaw servo", "in the carriage, shaft on the yaw axis", "as built", f"moved {V4b['yaw_spline_x'] + 2.55:.1f} mm back along the yaw axis"],
        ["Yaw axis to MCP axis (mm)", "20.75", "20.75", "0"],
        ["MCP axis to PIP axis (mm)", "30.8", "30.8", f"{V4b['proximal']:.2f}"],
        ["Width below the MCP joint (mm)", "41.5 (servo across the link)", "21 (cassette) plus the rod", "21 (cassette) plus the rod"],
        ["Four-bar coupler (mm)", "&ndash;", f"{a['coupler_mm']:.1f}", f"{b['coupler_mm']:.1f}"],
        ["Least self-clearance over the driver's range (mm)", "&ndash;", f"{C['v4a']['least_mm']:.2f}", f"{C['v4b']['least_mm']:.2f}"],
    ]
    t = table(["", "v3 as built", "v4a: PIP servo on the yaw link", "v4b: gimbal MCP"], rows,
              "The two layouts against the hand as built. Every new part is a box or capsule in the scenes; the parts that do not "
              "change keep cad_v3's meshes.", num_from=9)
    poses = ("Postures (yaw, MCP, PIP): open (0&deg;, 0&deg;, 10&deg;), curl (0&deg;, 70&deg;, 80&deg;), yaw 35&deg; "
             "(35&deg;, 30&deg;, 40&deg;).")
    fig = img(MEDIA / "20261118-finger_layouts_side.png",
              "The index finger as built (top row), v4a, v4b and v4c (bottom row), seen along the MCP axis with the palm to the "
              "left. Orange boxes are servo housings; grey bars are rods, couplers, cranks and horns; v4b's carriage is a "
              "translucent block. v4c's servos are dark boxes: two in the carriage and one lying flat on the MCP link. " + poses,
              jpeg=True)
    fig += img(MEDIA / "20261118-finger_layouts_front.png",
               "The same fingers seen from the dorsal side, the side away from the thumb. Below the MCP joint v4a and v4b are as wide "
               "as the sensor cassette; v4c is as wide as its two plates. " + poses, jpeg=True)
    return f"""<p>Both layouts take the PIP servo off the finger. The servo turns an idler on the MCP axis. A rod parallel to the MCP
link joins a crank on the idler to a crank on the distal link. This parallelogram gives the distal link the idler's angle
$\\theta$.
A parallel four-bar turns the idler from the PIP servo's crank. Below the MCP joint the finger then needs only the sensor
cassette's 21 mm. The rod runs inside that width except for 15 mm next to the idler, where it sits 3.7 mm outside it.</p>
<p>v4a keeps the yaw link, the yaw servo and the MCP servo as built and puts the PIP servo in a housing on the yaw link,
behind the MCP servo. v4b puts the yaw axis and the MCP axis through one point G on the yaw axis, a gimbal. The MCP servo
stands on the yoke with its shaft through G. The PIP servo sits on the yoke right behind it. The yaw servo moves back
along the yaw axis. The yaw link's 20.75 mm moves into the MCP link, which grows to {V4b['proximal']:.2f} mm, so the reach from the yaw
axis stays the same. Section SEC[clearance] places the gimbal 20.75 mm lower as well, at v3's MCP point.</p>
{t}{fig}
<p>Neither layout runs into itself over the driver's ranges (Table TAB[selfc]). The tightest pairs were set by
design. In v4a the coupler passes the rod in the next layer at {C['v4a']['least_mm']:.2f} mm. In v4b the MCP link's web passes the MCP
servo's tab at {C['v4b']['least_mm']:.2f} mm.</p>
<p>That check left out the carriage. In the trials' postures v3's MCP link kept at least {K3['least_mm']:.2f} mm from the carriage's
collision spheres. v4a's index MCP link overlapped them in {Ka['overlapping']} of {Ka['poses']:,} periods, by up to {mn(-Ka['least_mm'], 2)} mm.
The overlap is at the top of the link's plate on the servo-base side, at yaw {mn(Ka['overlap_yaw_deg'][0])}&deg; to
{mn(Ka['overlap_yaw_deg'][1])}&deg; and MCP {Ka['overlap_mcp_deg'][0]}&deg; to {Ka['overlap_mcp_deg'][1]}&deg;. In those postures the
carriage's mesh stays {Ka['mesh_least_mm_where_spheres_overlap']:.2f} mm from v4a's link, so only the spheres overlap. v4b's MCP link
clears the carriage's spheres by {Kb['least_mm']:.2f} mm over the driver's whole yaw and MCP ranges.</p>"""


def sec_selfclear(J):
    C = J["check"]
    rows = []
    for v in ("v4a", "v4b"):
        for k, (g, yaw, mcp, pip) in list(C[v]["least_by_pair"].items())[:5]:
            rows.append([v, esc(k), f"{g:.2f}", f"{yaw:g}", f"{mcp:g}", f"{pip:g}"])
    return table(["Layout", "Parts (proximal | moving)", "Least gap (mm)", "Yaw (deg)", "MCP (deg)", "PIP (deg)"], rows,
                 "The five closest pairs of each layout over the driver's MCP and PIP ranges (3&deg; grid; v4b also at yaw "
                 "&minus;85&deg;, &minus;45&deg;, 0&deg;, 45&deg; and 85&deg;) and the posture where each pair is closest.", num_from=2)


def sec_jointspace(J):
    Jn = J["joints"]
    pf = Jn["per_finger_trials"]
    ratio = Jn["max_ratio_containing_driver_rectangle"]
    fig = img(MEDIA / "20261118-yaw_mcp_joint_space.png",
              "(a) The yaw and MCP angles of the index and the middle finger in every second period of the 65 bench trials, the "
              "driver's range of the two serial joints, and the range that a bevel differential reaches with two servos, each "
              "limited to its travel: SCS0009 and STS3032 at ratio 1:1, and STS3032 at 2:1. The differential's range is a "
              "diamond centred on the driver's range. (b) The joint torques available when each servo is limited to $T$: a "
              "square for one servo per joint, a diamond of half-width $2gT$ for the differential.")
    return f"""<p>In a differential each servo turns by the sum or the difference of the two joint angles. The joint angles
that two servos reach therefore lie inside a diamond:
$$|q_{{\\mathrm{{yaw}}}} - c_{{\\mathrm{{yaw}}}}| + |q_{{\\mathrm{{MCP}}}} - c_{{\\mathrm{{MCP}}}}| \\le \\frac{{\\Theta}}{{2g}}.$$
The driver's whole range, &plusmn;85&deg; of yaw and &minus;15&deg; to 92&deg; of MCP, lies inside the diamond when $\\Theta/g$ is at
least {Jn['driver_rectangle_motor_range_needed_at_1to1_deg']:.0f}&deg;. That allows a ratio of up to {ratio['scs0009']:.2f} with the SCS0009 and {ratio['sts3032']:.2f} with the STS3032. The
trials used much less. The index's 99 % band spanned {mn(pf['index']['yaw_band'][0])}&deg; to {mn(pf['index']['yaw_band'][1])}&deg; of yaw and
{mn(pf['index']['mcp_band'][0])}&deg; to {mn(pf['index']['mcp_band'][1])}&deg; of MCP. It fits inside the diamond up to a ratio of
{pf['index']['max_ratio_sts3032']:.1f} with STS3032s. In return each joint can take the torque of both servos:
$|\\tau_{{\\mathrm{{yaw}}}}| + |\\tau_{{\\mathrm{{MCP}}}}| \\le 2gT$.</p>
{fig}
<p>At these ratios the diamond contains the hand's whole joint range. The differential's costs are elsewhere. Its bevel gears add
backlash after the servos' sensors. With the hand's joint order, yaw first, the two side gears sit on the yaw axis. One of
them is then on the palmar side, where the MCP link points at 90&deg; of flexion. Hands with a differential at the base put
the MCP first. In that order the MCP link also turns about a second axis that the idler does not follow, and the
parallelogram drive of the PIP no longer works. Neither cad_v4 layout uses a differential.</p>"""


def sec_amazing(J):
    A, G = J["amazing"], J["amazing_geom"]
    S = A["scales"]
    s1, s15, s2, s3 = S["1"], S["1.5"], S["2"], S["3"]
    off = A["geometry"]["attach_offsets_mm"]
    tr = s1["trial_share_inside_best_shift"]
    area = s1["reachable_area_deg2"] / A["bevel_diamond_scs0009_1to1_area_deg2"]
    fr, ar = s1["flexion_range_at_zero_abduction_deg"], s1["abduction_range_at_key_flexion_deg"]
    rat, rod, hrn = s1["servo_deg_per_joint_deg_at_key"], s1["joint_error_per_0.1mm_rod_play_at_key_deg"], \
        s1["joint_error_per_1deg_horn_play_at_key_deg"]
    fig = img(MEDIA / "20261119-amazinghand_joint_space.png",
              f"(a) The AmazingHand's reachable region on each finger's bench trials, the SR2 driver's range and the bevel "
              f"differential's diamond at 1:1 with the SCS0009. The index's region is turned by "
              f"{mn(tr['index']['shift_yaw_deg'], 0)}&deg; of yaw, the middle's by {tr['middle']['shift_yaw_deg']:.0f}&deg; of yaw and "
              f"{mn(tr['middle']['shift_mcp_deg'], 0)}&deg; of MCP. (b) The region in the AmazingHand's own angles with the angle of "
              f"each horn, and the regions with horns 1.5 and 2 times longer.")
    n = H.FIG[0]
    return f"""<h3>The AmazingHand's rod differential</h3>
<p>The AmazingHand's rod differential reached {pct(area)} of the area of the bevel differential's diamond (SCS0009, 1:1). It
reached {pct(s1['driver_rectangle_share_inside_best_shift'])} of the driver's range of yaw and MCP. Turned onto each finger's
trials, its region covered {pct(tr['index']['share'])} of the index's logged periods and {pct(tr['middle']['share'])} of the
middle's (Figure {n}a). At rest each joint took {s1['flexion_torque_per_servo_limit_at_key']:.2f} (flexion) and
{s1['abduction_torque_per_servo_limit_at_key']:.2f} (abduction) times one servo's torque.</p>
<p>Pollen Robotics' AmazingHand (2025) drives each finger's universal joint with two SCS0009 servos in the palm. Its
Enhanced version of April 2026 uses STS3032s on a new printed horn; the project's MuJoCo model is of the original. The geometry of
the right hand's first finger was read from the project's MuJoCo model (commit {G['commit'][:7]}). Its horns have a radius of
{A['geometry']['horn_radius_mm'][0]:.1f} mm, and its rods are {A['geometry']['rod_mm'][0]:.1f} mm between ball centres. The
rods meet the proximal link {off[0]['palmar']:.1f} mm palmar of the joint centre, {off[0]['lateral']:.1f} and
{-off[1]['lateral']:.1f} mm to either side. The rods were solved in closed form over abduction and flexion. At ten servo pairs
the closed form and MuJoCo's own closure differed by at most {G['closure_check']['max_error_deg']:.3f}&deg;.</p>
<p>A horn moves its rod at most {A['geometry']['horn_radius_mm'][0]:.0f} mm either way, at its dead points. The horns are kept within
&plusmn;90&deg; of zero in the model and in the project's demo code. Over that range the finger flexed from {mn(fr[0])}&deg; to
{fr[1]:.1f}&deg; at zero abduction. At its rest flexion of {A['geometry']['flexion_at_key_deg']:.1f}&deg; it abducted from
{mn(ar[0])}&deg; to {ar[1]:.1f}&deg;; the project's own code limits abduction to &plusmn;20&deg;. Near rest a servo turned
{rat['pure_flexion']:.2f}&deg; per degree of flexion and {rat['pure_abduction']:.2f}&deg; per degree of abduction. A bevel
differential at 1:1 turns its servos 1&deg; per joint degree over their whole 300&deg;.</p>
<p>Longer horns move the rods further for the same servo angle, at a lower torque. With {s15['horn_radius_mm'][0]:.1f} mm horns
the region covered {pct(s15['driver_rectangle_share_inside_best_shift'])} of the driver's range, and the flexion torque at rest fell
to {s15['flexion_torque_per_servo_limit_at_key']:.2f} times one servo's. With {s2['horn_radius_mm'][0]:.0f} mm horns the region
shrank to {pct(s2['driver_rectangle_share_inside_best_shift'])} of the driver's range under the floor of one servo's torque. With
{s3['horn_radius_mm'][0]:.0f} mm horns the rest pose itself fell below that floor, at
{s3['flexion_torque_per_servo_limit_at_key']:.2f} in flexion. The rods push along the finger, and their moment arm about a
joint shrinks as the joint turns away from rest (Figure {n}b). No horn length tried covered the driver's whole range; the
bevel differential covers it at ratios up to {J['joints']['max_ratio_containing_driver_rectangle']['scs0009']:.2f}.</p>
<p>At rest the finger moved up to {rod['flexion']:.2f}&deg; in flexion and {rod['abduction']:.2f}&deg; in abduction for
0.1 mm of play at one ball joint. For one degree of play at one horn it moved {hrn['flexion']:.2f}&deg; and
{hrn['abduction']:.2f}&deg;, so the servo's own play is divided by more than two at the joint. In the project's README the
angle errors of its prototypes are put down to the prints, the hand-adjusted rods, the reworked horns and the plastic's
flexibility.</p>
{fig}"""


def sec_v4c(J):
    C, P4 = J["codesign"], J["params"]["v4c"]
    ch, chs = C["chosen"]["scs0009"], C["chosen"]["sts3032"]
    best, bests, ah = C["best"]["scs0009"], C["best"]["sts3032"], C["amazinghand_geometry_with_sts3032"]["scs0009"]
    K, R = J["check"]["v4c"], J["replay"]["summary"]["v4c_same_tips"]
    F = J["forces"]["v4c"]
    B, Q = P4["base"], P4["pip"]
    rods = J["build"]["v4c"]["index"]["rod_mm"]
    k = C["sts_over_scs0009"]
    yr = ch["yaw_reach_at_mcp_deg"]
    n_fp = 2 * R["periods"]
    tq = lambda x: f"{x * k:.1f}"  # noqa: E731  STS3032 limits to SCS0009 limits
    fig = img(MEDIA / "20261119-v4c_codesign.png",
              "(a) The share of the driver's yaw and MCP range that each design of the sweep covers, against the rest torque "
              "of its weaker joint. Every joint keeps at least one SCS0009's torque. (b) v4c's reachable region from its "
              "rest pose, coloured by the weaker joint's torque; the dashed lines bound the driver's range at the best "
              "placement.")
    first = max(F["first_limit_v4c"], key=F["first_limit_v4c"].get)
    pair0, w0 = next(iter(K["least_by_pair"].items()))
    LM = J["check"]["v4c_limits"]
    PAIRS = {"pipservo_boss": "its servo's spline boss", "pipservo_case": "its servo's case"}
    ik = R["ik"]
    return f"""<p>v4c reaches {pct(ch['driver_share'])} of the driver's yaw and MCP range while each joint keeps at least the
current hand's torque, one SCS0009 per joint. Over that range its MCP and yaw take a median of {tq(ch['window_torque_sts_min_median']['mcp'][1])}
and {tq(ch['window_torque_sts_min_median']['yaw'][1])} times that torque. Over the 65 trials {pct(1 - R['finger_periods_outside_reach'] / n_fp)} of
finger-periods lie inside its reach, and index and middle overlapped in {pct(R['share_overlap'])} of periods. The pinch force at
the servo limits was {F['ratio_v4c_over_v3_p5_p50_p95'][1]:.2f} times v3's at the median.</p>
<p>v4c keeps v4b's gimbal: the yaw and MCP axes cross at G on v3's yaw axis, and the MCP link is {P4['proximal']:.2f} mm long. Two
STS3032s stand side by side in the carriage, shafts along the yaw axis {2 * B['shaft_y']:.1f} mm apart. Their {B['horn_r']:.0f} mm horns
push {rods[0]:.1f} mm rods with a ball joint at each end. The rods meet lugs on the MCP link's plates {B['lug_x']:.0f} mm dorsal of G
and {B['lug_y']:.0f} mm to either side, square to the rods at an MCP angle of {B['mcp_rest']:.0f}&deg;. Turning both horns one way
flexes the MCP; turning them opposite ways yaws the finger. The yoke turns on one pin on the dorsal side, so the palmar
side stays open for flexion to 92&deg;.</p>
<p>The PIP servo lies flat between the MCP link's plates, its {2 * J['params']['servos']['sts3032']['half_t']:.0f} mm thickness across the
finger and its shaft along the yaw axis. Its spline sits near the PIP, {-Q['spline_z']:.0f} mm below G. Its {Q['horn_r']:.0f} mm horn
pushes one {rods[2]:.1f} mm rod, below the servo and between the plates, to a {Q['lug_r']:.0f} mm lug on the DIP link's centre
plane. The lug is square to the rod at a PIP angle of {Q['pip_rest']:.0f}&deg;. Over the driver's PIP range the PIP takes at least
{tq(F['pip_drive_over_driver_range']['torque_sts_min_median'][0])} times one SCS0009's torque, with a median of
{tq(F['pip_drive_over_driver_range']['torque_sts_min_median'][1])}. Below the gimbal the finger is
{2 * P4['mcp_link']['plate_y'][1]:.0f} mm wide, the width of its two plates.</p>
<p>The rods' dimensions came from a sweep over the horn radius $r$, the lug offsets $h$ and $w$ and the rod length $L$:
{len(C['rows']) // 2} designs (<code>scripts/cad/v4c_codesign.py</code>). Each design was scored by the largest share of the driver's
range that one placement of its reachable region covers. The floor was the current hand's torque, {C['floors_sts']['scs0009']:.2f}
STS3032 limits per joint (2.3 against 4.5 kg cm at 6 V, the sheets). The AmazingHand's rods covered {pct(ah['driver_share'])} with
STS3032s. The best design covered {pct(best['driver_share'])} ($r$ {best['r']:g}, $h$ {best['h']:g}, $w$ {best['w']:g}, $L$ {best['L']:g} mm),
with {tq(best['rest_torque_sts']['mcp'])} times the current MCP torque at rest. v4c's design ($r$ {ch['r']:g}, $h$ {ch['h']:g}, $w$ {ch['w']:g},
$L$ {ch['L']:g} mm) covered {pct(ch['driver_share'])} with {tq(ch['rest_torque_sts']['mcp'])} times it in both joints. With a floor
of one STS3032 per joint the best design covered {pct(bests['driver_share'])}, and v4c's {pct(chs['driver_share'])}.</p>
<p>The range left out lies at large yaw with the finger curled. With the MCP at &minus;15&deg; the finger yaws to
&plusmn;{yr['-15'][1]:.1f}&deg;; at {B['mcp_rest']:.0f}&deg; to &plusmn;{yr[f"{B['mcp_rest']:g}"][1]:.0f}&deg;; at 92&deg; to
&plusmn;{yr['92'][1]:.0f}&deg;.</p>
{fig}
<p>In MuJoCo the three rods close on their lugs through connect constraints, and the horns stop at &plusmn;90&deg;. Over the
driver's ranges, at the {K['poses_in_reach']:,} poses of a 3&deg; and 6&deg; grid inside the reach, the least self-clearance was
{mn(K['least_mm'], 2)} mm. That is the PIP rod against {PAIRS.get(pair0.split(' | ')[1], esc(pair0))}, at yaw {mn(w0[1], 0)}&deg;, MCP
{mn(w0[2], 0)}&deg; and PIP {mn(w0[3], 0)}&deg;.
A base rod stays clear of the other side's lug and plate up to &plusmn;{LM['base_clear_to_yaw_deg']:g}&deg; of yaw and meets the lug at
&plusmn;85&deg;. Below a PIP angle of {mn(LM['pip_clear_deg'][0], 0)}&deg; the PIP rod passes over its servo's spline boss, so the PIP
extends only to {mn(LM['pip_clear_deg'][0], 0)}&deg; until the horn moves. Of the trials' finger-periods {R['finger_periods_outside_reach']:,}
lie outside v4c's reach; on the gimbal the logged tips need angles outside the driver's ranges in {ik['outside_driver_limits']:,}, as
for v4b. The pinch force at the servo limits ranged from {F['pinch_N_v4c_sts3032_p5_p50_p95'][0]:.1f} to
{F['pinch_N_v4c_sts3032_p5_p50_p95'][2]:.1f} N over the trials' postures (5th to 95th percentile). For v3 it ranged from
{F['pinch_N_v3_scs0009_p5_p50_p95'][0]:.1f} to {F['pinch_N_v3_scs0009_p5_p50_p95'][2]:.1f} N. In v4c the {first.upper()} reached its limit first in
{F['first_limit_v4c'][first]:,} of {F['finger_periods']:,} finger-periods.</p>
<p>Every v4c part but the servos, the ball links and the pins is flat. The MCP link's plates, the yoke and the carriage can be
cut from 1/8-inch Delrin. The horns are printed for the STS3032's 25-tooth spline, as on the AmazingHand Enhanced. Per
finger the parts to order are three servos, six M2 ball links on three M2 rods, and screws as pins.</p>"""


def sec_clearance(J):
    R, S = J["replay"]["summary"], J["sphere_replay"]["summary"]
    names = [("as_built", "v3 as built"), ("as_built_index_mirrored", "v3, index finger mirrored"),
             ("v4a", "v4a, PIP servo on the yaw link"), ("v4b_same_tips", "v4b, gimbal on the yaw axis"),
             ("v4b_low_same_tips", "v4b, gimbal at v3's MCP point")]
    rows = []
    for k, nm in names:
        s = R[k]
        top = s["closest_parts_under_5mm (index | middle)"][:2]
        rows.append([nm, f"{s['min_mm']:.2f}", f"{s['p1_mm']:.2f}", f"{s['p10_mm']:.2f}", f"{s['median_mm']:.2f}",
                     pct(s["share_overlap"]), pct(s["share_under_5mm"]), esc("; ".join(f"{p} ({n})" for p, n in top))])
    t = table(["Hand", "Least gap (mm)", "1st pct. (mm)", "10th pct. (mm)", "Median (mm)", "Overlap", "Under 5 mm",
               "Closest parts under 5 mm (index | middle, periods)"], rows,
              "Index-middle clearance over the 10,350 periods. v3 and v4a replay the logged joint angles; both v4b rows reach the "
              "same tip positions by inverse kinematics from the same mounts.", num_from=1)
    ik, ikl = R["v4b_same_tips"]["ik"], R["v4b_low_same_tips"]["ik"]
    n = 2 * R["v4b_same_tips"]["periods"]
    sph = S["as_built"]
    return f"""<p>{t}</p>
<p>The as-built row reproduces the earlier sphere replay ({pct(sph['share_overlap'])} overlap, least gap {sph['min_mm']:.2f} mm). Most of the
overlap is the two distal links, whose clevises straddle the PIP servos. With the PIP servo off the finger (v4a) the overlap
falls to {pct(R['v4a']['share_overlap'])}, and the closest parts become the fingertip caps, whose distance the grasp sets.
The as-built index finger can also be mirrored, so that its servo bases face the middle finger as the middle's face the
index. With no other change the overlap falls to {pct(R['as_built_index_mirrored']['share_overlap'])}. Each servo's horn side reaches 24.0 mm from
the finger's centre plane, its base side 17.4 mm.</p>
<p>With the gimbal on the yaw axis, v4b reaches every logged tip position from the same mount, at the same yaw angle as v3.
Its longer MCP link reaches them with other MCP and PIP angles. In {ik['outside_driver_limits']} of {n} finger-periods
({pct(ik['outside_driver_limits'] / n)}) the PIP then goes past the driver's &minus;18&deg;, down to
{mn(min(ik['pip_range_deg']['index'][0], ik['pip_range_deg']['middle'][0]), 0)}&deg;. With the gimbal at v3's MCP point the yaw lever is
20.75 mm shorter. Then {ikl['unreachable_finger_periods']} of {n} finger-periods ({pct(ikl['unreachable_finger_periods'] / n)}) are out of reach from the same
mounts, and the gantry would have to move the mounts sideways. That placement is left out of the table.</p>"""


def sec_forces(J):
    F = J["forces"]
    r = F["pinch_ratio_v4_over_v3_p5_p50_p95"]
    s3, st = F["pinch_N_at"]["scs0009_s3_trip"], F["pinch_N_at"]["sts3032_stall_6V"]
    yl = F["yaw_lever_mm_p5_p50_p95"]
    lf = F["lateral_force_ratio_v4b_over_v3_p5_p50_p95"]
    tr, trd = F["transmission_deg_min_p1"], F["driver_range_transmission_deg"]
    rows = [["v3 as built, SCS0009 at the S3 trip (0.17 N m)", *[f"{x:.2f}" for x in s3["v3_p5_p50_p95"]]],
            ["v4a or v4b, SCS0009 at the S3 trip (0.17 N m)", *[f"{x:.2f}" for x in s3["v4_p5_p50_p95"]]],
            ["v3 as built, STS3032 at stall, 6 V (0.44 N m)", *[f"{x:.2f}" for x in st["v3_p5_p50_p95"]]],
            ["v4a or v4b, STS3032 at stall, 6 V (0.44 N m)", *[f"{x:.2f}" for x in st["v4_p5_p50_p95"]]]]
    t = table(["Hand and servo limit", "5th pct. (N)", "Median (N)", "95th pct. (N)"], rows,
              f"Pinch force along the pad normal at the first servo's limit, over the {F['postures']} finger postures of the replay.")
    return f"""<p>Once the PIP servo sets $\\theta$, the distal link's moment no longer loads the MCP servo:
$$F_{{\\mathrm{{v3}}}} = \\frac{{T}}{{b + a\\cos q_{{\\mathrm{{PIP}}}}}}, \\qquad
F_{{\\mathrm{{v4}}}} = \\min\\!\\left(\\frac{{T}}{{a\\,|\\cos q_{{\\mathrm{{PIP}}}}|}}, \\frac{{T}}{{b}}\\right) = \\frac{{T}}{{b}}.$$
Over the trials' postures the pinch force at the servo limit rose by a factor of {r[1]:.2f} at the median. The factor was
{r[0]:.2f} at the 5th percentile and {r[2]:.2f} at the 95th. The PIP servo sets the limit everywhere. At the median the MCP
servo bears {F['mcp_servo_share_of_v3_load_p50']:.2f} of its v3 load.</p>
{t}
<p>The yaw lever is the same for v3, v4a and v4b with the gimbal on the yaw axis: {yl['v3'][1]:.1f} mm at the median.
With the gimbal at v3's MCP point it is {yl['v4b'][1]:.1f} mm, which raises the sideways force at the yaw servo's limit by a factor
of {lf[1]:.2f}.</p>
<p>The rod and the four-bar were phased for the middle of the trials' ranges, PIP {F['pip_mid_deg']:.1f}&deg; and
$\\theta$ = {F['dip_abs_mid_deg']:.1f}&deg;. Over the trials their least transmission angles were {tr['rod'][0]:.0f}&deg; for the rod and
{tr['four_bar'][0]:.0f}&deg; for the four-bar. Over the driver's full PIP range the rod's falls to {trd['rod']:.0f}&deg;. The driver's full range of
$\\theta$, &minus;33&deg; to 184&deg;, takes a single four-bar through its dead point. A full curl therefore needs a second rod at
90&deg; to the first at each stage, or a spur pair in place of the four-bar.</p>"""


def sec_limits():
    return """<p>The new parts are envelopes: boxes for servo cases, housings and plates, capsules for rods and cranks.
They are not printable parts, and the yoke of v4b and the carriage around it are blocks that show where material can go.
The SCS0009's case height (25.0 mm) is estimated; the STS3032's dimensions are from Feetech's drawing. The rod's force,
bearing loads and the stiffness of the printed links are not computed. The v4b rows of the replay keep the logged mounts
and tip positions; plans made for v4b would choose their own. The pinch force uses the pad centre as the contact point and
the pad normal as the force direction, and it ignores friction and the yaw servo. The AmazingHand's region leaves out
collisions of its rods and links with each other and with the neighbouring fingers, and the ball joints' angle limits.
v4c's rods are ideal: the ball links' angle limits and play are not modelled, and its self-clearance is taken at five yaw
angles. Its servo torques are the sheets' stall values at 6 V.</p>"""


def sec_future():
    return """<ul>
<li>Detail v4a in Onshape from the STEP envelopes (<code>assets/cad_v4/concepts/step/</code>, written by
<code>freecadcmd scripts/cad/cad_v4_step.py</code>): the yaw link's housing, the idler on MR52 bearings, a 3 mm steel rod with
pressed pins. The claim to test is the overlap share: rerun <code>scripts/cad/cad_v4_layouts.py replay</code> on the detailed
parts' spheres. An overlap share above 5 % refutes the slim finger's gain.</li>
<li>Measure the drive's lash on a printed v4a finger with the S3 head-on press: the PIP servo's reading against the HEX-E
force at 1 deg/s to 2 N. More than 1&deg; of give between the PIP servo and the pad, beyond the servo's own, refutes the claim
that pinned rods add no slop.</li>
<li>Mirror the index finger as built (servo bases toward the middle finger) and rerun the replay with its spheres
mirrored in <code>scripts/mechanisms/sr2_index_middle_clearance_replay.py</code>; it needs no new parts.</li>
<li>Re-plan the HOM bench tasks for v4b with the gimbal on the yaw axis (<code>scripts/hom/hom_gpu.py</code> on a cad_v4 scene).
Count the plans that bend the PIP back past &minus;18&deg;. If the share stays above 5 %, give the PIP stop more
hyperextension or split the 20.75 mm between the MCP link and a fixed metacarpal.</li>
<li>Give the AmazingHand's base the SR2 finger's PIP joint. A PIP servo on the gimbal, its shaft on the flexion axis, turns the
crank of a parallelogram rod to the distal link; the crank replaces the four-bar's fixed pivot on the gimbal. Model it as a third layout in
<code>scripts/cad/cad_v4_layouts.py</code> with 7.5 mm horns and replay the 65 trials. Fewer than 99 % of the trials' postures
inside its reachable region, or an overlap share above v4a's, refutes it.</li>
<li>Cut and print a v4c index finger (1/8-inch Delrin plates, printed horns, M2 ball links) and drive its horns to
&plusmn;90&deg; while the D435 tracks the DIP link. A reachable region more than 10 % smaller than the closed form's, or more
than 1&deg; of give at the joints under 2 N at the pad beyond the servos' own, refutes the model.</li>
<li>Move v4c's PIP horn so its rod clears the spline boss in extension (a longer horn, or the servo further palmar), and rerun
<code>scripts/cad/cad_v4_layouts.py check</code> (key <code>v4c_limits</code>). A PIP range that still stops above &minus;18&deg; refutes
the change.</li>
</ul>"""


def sec_files():
    rows = [["<code>scripts/cad/cad_v4_layouts.py</code>", "builds the scenes; check, replay, forces, joint space, renders"],
            ["<code>scripts/cad/cad_v4_page.py</code>", "this page"],
            ["<code>assets/cad_v4/concepts/20261118-layout_params.json</code>", "every dimension of the two layouts"],
            ["<code>assets/mjcf/experimental/20261118-cad_v4/v4a.xml</code>, <code>v4b.xml</code>",
             "MuJoCo scenes on the cad_v3 hand. Under physics, equality constraints keep the drive on the MCP and PIP joints. "
             "<code>scripts/cad/cad_v4_layouts.py play v4a</code> opens one in the viewer with the least index-middle gap "
             "shown; <code>--kinematic</code> poses the joints exactly"],
            ["<code>scripts/cad/amazinghand_differential.py</code>",
             "the AmazingHand's rod differential: geometry from its MuJoCo model, closed-form rods, reachable region, torques"],
            ["<code>scripts/cad/v4c_codesign.py</code>", "the sweep of rod dimensions for v4c: reach against torque"],
            ["<code>assets/mjcf/experimental/20261118-cad_v4/v4c.xml</code>",
             "v4c on the cad_v3 hand: connect constraints close its three rods; <code>scripts/cad/cad_v4_layouts.py play v4c</code>"],
            ["<code>docs/experiments/20261118-sr2_actuator_layout/data/</code>", "the JSON behind every table and figure"],
            ["<code>scripts/mechanisms/sr2_index_middle_clearance_replay.py</code>", "the earlier sphere replay (commit adc1a0d3b)"]]
    return table(["File", "Content"], rows, "Files.", num_from=9)


def lede_and_summary(J):
    R, F = J["replay"]["summary"], J["forces"]
    r = F["pinch_ratio_v4_over_v3_p5_p50_p95"]
    ratio = J["joints"]["max_ratio_containing_driver_rectangle"]
    ik = R["v4b_same_tips"]["ik"]
    n = 2 * R["v4b_same_tips"]["periods"]
    am = J["amazing"]["scales"]["1"]
    cd = J["codesign"]["chosen"]["scs0009"]
    lede = (f"Taking the PIP servo off the finger cut the index-middle overlap in the bench trials from "
            f"{pct(R['as_built']['share_overlap'])} to {pct(R['v4a']['share_overlap'])} of control periods. It raised the pinch force at "
            f"the servo limit by a factor of {r[1]:.2f} at the median. In v4a the PIP servo moves to the yaw link and drives the "
            f"distal link through a crank on the MCP axis and a parallelogram rod. v4b also puts the yaw and MCP axes through one "
            f"point. On the yaw axis this gimbal reaches every logged tip ({pct(R['v4b_same_tips']['share_overlap'])} overlap), but "
            f"the PIP goes past &minus;18&deg; in {pct(ik['outside_driver_limits'] / n)} of finger-periods. With a bevel "
            f"differential the driver's whole range lies inside the reachable diamond up to a ratio of {ratio['scs0009']:.2f} "
            f"(SCS0009) or {ratio['sts3032']:.2f} (STS3032). The AmazingHand's rod differential reached "
            f"{pct(am['driver_rectangle_share_inside_best_shift'])} of that range, and "
            f"{pct(min(v['share'] for v in am['trial_share_inside_best_shift'].values()))} or more of the logged postures. "
            f"A codesigned rod differential with STS3032s (v4c) reached {pct(cd['driver_share'])} of the range with every joint at "
            f"or above the current hand's torque.")
    summary = (f"Two cad_v4 finger layouts take the PIP servo off the finger. The overlap of index and middle over the 65 bench "
               f"trials falls from {pct(R['as_built']['share_overlap'])} to {pct(R['v4a']['share_overlap'])} (v4a), and the pinch force at the "
               f"servo limit rises {r[1]:.2f}x at the median; a gimbal MCP (v4b) reaches every logged tip with "
               f"{pct(R['v4b_same_tips']['share_overlap'])} overlap. The AmazingHand's rod differential reaches "
               f"{pct(am['driver_rectangle_share_inside_best_shift'])} of the driver's yaw and MCP range; a codesigned one with "
               f"STS3032s (v4c) reaches {pct(cd['driver_share'])} with every joint at or above the current hand's torque.")
    return lede, summary


def main():
    J = load()
    sections = [("terms", "Terms and metrics", sec_terms(J)),
                ("layouts", "Two layouts with the PIP servo off the finger", sec_layouts(J) + sec_selfclear(J)),
                ("jointspace", "Joint space of a two-servo differential at the MCP", sec_jointspace(J) + sec_amazing(J)),
                ("v4c", "v4c: a codesigned two-rod differential and the PIP servo on the MCP link", sec_v4c(J)),
                ("clearance", "Index-middle clearance over the bench trials", sec_clearance(J)),
                ("forces", "Pinch force at the servo limits, yaw lever and transmission angles", sec_forces(J)),
                ("limits", "Limitations", sec_limits()),
                ("future", "Future work", sec_future()),
                ("files", "Files", sec_files())]
    num_ = {sid: k for k, (sid, _h, _b) in enumerate(sections, 1)}
    sub = (lambda s: re.sub(r"SEC\[(\w+)\]", lambda m_: str(num_[m_.group(1)]), s))
    unit = (lambda s_: re.sub(r"(\d) (%|mm|deg|N\b|g\b)", "\\1&nbsp;\\2", s_))
    body = [(sid, hd, unit(sub(b))) for sid, hd, b in sections]
    html_all = "\n".join(b for _s, _h, b in body)
    caps = dict((cap[:30], n) for n, cap in re.findall(r"<caption>Table (\d+)\. ([^<]{0,40})", html_all))
    ref = {"selfc": next(n for c, n in caps.items() if c.startswith("The five closest pairs"))}
    body = [(sid, hd, re.sub(r"TAB\[(\w+)\]", lambda m_: ref[m_.group(1)], b)) for sid, hd, b in body]
    toc = "\n".join(f'<li><a href="#{sid}">{k}. {esc(hd)}</a></li>' for k, (sid, hd, _) in enumerate(body, 1))
    secs = "\n".join(f'<section id="{sid}"><h2>{k}. {esc(hd)}</h2>{b}</section>' for k, (sid, hd, b) in enumerate(body, 1))
    lede, summary = lede_and_summary(J)
    byline = ("Layout study in MuJoCo on the cad_v3 hand with the 65 logged bench trials of 15 to 17 November. Nothing was "
              "printed. Builders: <code>scripts/cad/cad_v4_layouts.py</code>, <code>scripts/cad/cad_v4_page.py</code>.")
    page = TEMPLATE.read_text().replace("The cad_v3 page (scripts/cad/cad_v3_page.py)", "The cad_v4 layouts page (scripts/cad/cad_v4_page.py)")
    for k_, v in {"TITLE": "SR2 finger layouts", "H1": "PIP drive from the yaw link and a gimbal MCP for the SR2 finger",
                  "EYEBROW": "Hand and bench, 2026-11-18", "LEDE": unit(lede), "BYLINE": byline, "TOC": toc, "SECTIONS": secs,
                  "FOOTER": str(OUT.relative_to(ROOT)), "TEXCSS": texsvg.CSS}.items():
        page = page.replace("{{" + k_ + "}}", v)
    items = [((m.group(1) if m.group(1) is not None else m.group(2)).strip(), m.group(1) is not None)
             for m in H.TEX_RE.finditer(page)]
    svgs = iter(texsvg.render(items, cache_path=TEX_CACHE))
    page = H.TEX_RE.sub(lambda m: next(svgs), page)
    page = docs_index.tag(page, topic="hardware", summary=summary, date="2026-11-18")
    page = retro_style.apply(page)
    OUT.write_text(page)
    print(f"{OUT}: {OUT.stat().st_size / 1e6:.2f} MB, {len(items)} formulas", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
