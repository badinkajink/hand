// BAM (Better Actuator Models) pendulum / double-pendulum test rig
// One print set fits both Feetech SCS0009 and STS3032 micro servos.
//
//   PART = "base"    clamps to a table edge; holds servo 1, shaft horizontal, over the edge
//   PART = "link1"   double pendulum link 1: on servo 1's horn, carries servo 2 at L1
//   PART = "arm"     identification arm: weight holes at 45 / 60 / 90 mm. Mounted ONCE per servo, so
//                    BAM's single zero offset (q_offset) stays valid across every recording.
//                    Its real mass properties are logged (stock BAM's uniform-rod model doesn't fit it).
//   PART = "slim_arm" (-D SLIM_L=60) slim single-hole arm: link 2 of the double pendulum
//   PART = "coupon"  10-minute fit test: servo window + slots, and a horn pocket. PRINT FIRST.
//   PART = "plate"   everything laid out on one bed
//   PART = "assembly" visual check only (not for printing)
//
// Export:  openscad -D 'PART="base"' -o base.stl bam_pendulum.scad
//
// Why one print fits both servos:
//  * Both bodies are 23.2 x 12.1 mm; only the height differs (SCS0009 25.25, STS3032 28.5).
//    The servo goes THROUGH a window and hangs on its tabs, so its height doesn't matter.
//  * The tab screw holes are slots (+-SLOT_PLAY) in case the STS3032 tab spacing differs.
//  * The output splines differ (SCS0009 20T/3.95 mm, STS3032 25T/4.95 mm), so arms couple
//    to each servo's STOCK horn: horn sits in a shallow pocket, is fixed with 2 small
//    self-tapping screws drilled through the horn's own holes, plus the center screw.
//
// Servo geometry below is measured from Pollen Robotics' AmazingHand scs0009.stl (CC BY 4.0).
// STS3032 tab position assumed similar; the clamp straps tolerate a different hole pattern.

PART = "plate";
$fn = 64;

// ---------------- servo (shaft-relative, along the body's long axis) ----------------
BODY_W      = 12.1;
BODY_L      = 23.2;
SHAFT_TO_NEAR_END = 5.95;     // shaft axis to the near end of the body
SHAFT_TO_FAR_END  = 17.35;    // = BODY_L - SHAFT_TO_NEAR_END (rounded per mesh)
HOLE_NEAR   = 8.6;            // shaft to near tab hole
HOLE_FAR    = 20.0;           // shaft to far tab hole (spacing 28.6)
TAB_NEAR    = 10.65;          // shaft to near tab end
TAB_FAR     = 21.85;          // shaft to far tab end
TAB_W       = 12.0;
CLEAR       = 0.25;           // window clearance per side (tune with the coupon)
SLOT_PLAY   = 1.0;            // slot half-length along the body axis
SCREW_D     = 2.3;            // M2 clearance

// ---------------- stock DISC horn pocket (measured 2026-10-28) ----------------
// STS3032 aluminum disc: 18.0 mm dia, 2.0 mm thick.  SCS0009 white disc: 17.84 mm, 1.55 mm.
// Same diameter to 0.16 mm, so one round pocket; depth is less than the thinner disc so the
// part clamps on either disc's face. The center screw centers the disc; 2 screws through the
// disc's own holes take out the remaining 0.1 mm of play.
DISC_D       = 18.0;
DISC_POCKET  = 1.4;           // < 1.55 (thinner disc)
HORN_CLEAR   = 0.2;           // per side; FDM holes print a little small
HORN_POCKET  = DISC_POCKET;   // depth reference used by the assembly view
HUB_D        = DISC_D + 6;

// ---------------- tab clamp straps ----------------
// Tabs are clamped, not screwed through, so any tab hole pattern works (the two servos differ:
// SCS0009 one hole per tab, STS3032 two holes and a notch). Each strap bridges one tab and
// is screwed into the printed part just outside the tab's width.
STRAP_X      = 8.6;           // clamp screw offset from the body's center line
STRAP_W      = 4;             // strap width along the body axis: must fit between the body end and the tab end (4.7 mm)
STRAP_T      = 3.0;
PILOT_D      = 1.7;           // M2 self-tapping pilot hole
TAB_MID_NEAR = 8.3;           // shaft to the middle of the near tab
TAB_MID_FAR  = 19.6;          // shaft to the middle of the far tab
CTR_SCREW_D = 2.4;            // horn center screw clearance
CTR_HEAD_D  = 4.6;            // counterbore for its head
FLOOR       = 1.5;            // material left under the center-screw head
// Disc screw pair, measured from IMG_7637 (5 mm grid, 2026-10-28): both discs have opposite holes
// 12.0 mm apart (aluminum: all 8 holes on a 12.0 circle; white: the two small side pairs).
// The disc's angle on the arm is free (it's screwed to the arm BEFORE going on the spline),
// so only the spacing matters. Screws go in from the part's front face into the disc.
PAIR_SPACING = 12.0;
PAIR_D       = 2.4;           // M2 clearance (+0.2 for photo-measurement error)
PAIR_CB_D    = 4.4;           // head counterbore, only where the part is thicker than PAIR_GRIP
PAIR_GRIP    = 3.6;           // material between the screw head and the disc (arm: full 3.6 mm)

// ---------------- rig dimensions ----------------
WALL_T   = 6;      // base wall thickness
AXIS_Z   = 20;     // servo 1 shaft height above the tabletop
OVERHANG = 10;     // wall front face this far in front of the table edge
LINK_T   = 5;      // link / arm plate thickness
L1       = 60;     // double pendulum: servo 1 axis -> servo 2 axis
BOSS_H   = 8;      // link1 rear hub boss: pushes link1 (and servo 2's body) away from the wall
ARM_HOLES = [45, 60, 90];   // weight-bolt positions from the axis (mm)
ARM_LEN  = 96;
M4_D     = 4.4;

// ======================================================================
module servo_window(h) {     // window + tab slots + clamp pilots; shaft at origin, body toward +X
    translate([-SHAFT_TO_NEAR_END - CLEAR, -BODY_W/2 - CLEAR, -1])
        cube([BODY_L + 2*CLEAR, BODY_W + 2*CLEAR, h + 2]);
    for (p = [-HOLE_NEAR, HOLE_FAR])
        hull() for (d = [-SLOT_PLAY, SLOT_PLAY])
            translate([p + d, 0, -1]) cylinder(d = SCREW_D, h = h + 2);
    for (p = [-TAB_MID_NEAR, TAB_MID_FAR], y = [-STRAP_X, STRAP_X])
        translate([p, y, -1]) cylinder(d = PILOT_D, h = h + 2);
}

module horn_pocket() {       // at origin, opening at z = 0, going +Z
    translate([0, 0, -0.01]) cylinder(d = DISC_D + 2*HORN_CLEAR, h = DISC_POCKET + 0.01);
}

module hub_holes(t) {        // center screw + counterbore from the top face
    translate([0, 0, -1]) cylinder(d = CTR_SCREW_D, h = t + 2);
    translate([0, 0, DISC_POCKET + FLOOR]) cylinder(d = CTR_HEAD_D, h = t);
    for (y = [-PAIR_SPACING/2, PAIR_SPACING/2]) translate([0, y, 0]) {   // across the arm axis
        translate([0, 0, -1]) cylinder(d = PAIR_D, h = t + 2);
        if (t > DISC_POCKET + PAIR_GRIP + 0.01)
            translate([0, 0, DISC_POCKET + PAIR_GRIP]) cylinder(d = PAIR_CB_D, h = t);
    }
}

// ---------------- identification arm (single pendulum) ----------------
// Printed flat. Bottom face (z=0) = rear face, sits on the horn. Arm points along +X.
module arm() {
    difference() {
        hull() {
            cylinder(d = HUB_D, h = LINK_T);
            translate([ARM_LEN - 6, 0, 0]) cylinder(d = 12, h = LINK_T);
        }
        horn_pocket();
        hub_holes(LINK_T);
        for (r = ARM_HOLES) translate([r, 0, -1]) cylinder(d = M4_D, h = LINK_T + 2);
        // engraved distance labels on the front face
        for (r = ARM_HOLES) translate([r - 7.5, 0, LINK_T - 0.4]) rotate([0, 0, 90])
            linear_extrude(1) text(str(r), size = 2.6, halign = "center", valign = "center");
    }
}

// ---------------- slim single-length arm (BAM-style: one arm per length) ----------------
// A uniform bar from the hub to a single weight hole at SLIM_L, ending just past the hole.
// It is close to BAM's "point mass at the end of a uniform rod" model, so stock BAM works with the
// real weighed masses.
SLIM_L = 60;
function slim_w(L) = L >= 90 ? 14 : 12;   // 90 mm arm wider: first bending mode ~42 Hz with 240 g
module slim_arm(L = SLIM_L) let(SLIM_W = slim_w(L)) difference() {
    union() {
        cylinder(d = HUB_D, h = LINK_T);
        translate([0, -SLIM_W/2, 0]) cube([L, SLIM_W, LINK_T]);
        translate([L, 0, 0]) cylinder(d = SLIM_W, h = LINK_T);
    }
    horn_pocket();
    hub_holes(LINK_T);
    translate([L, 0, -1]) cylinder(d = M4_D, h = LINK_T + 2);
    translate([L - 9, 0, LINK_T - 0.4]) rotate([0, 0, 90])
        linear_extrude(1) text(str(L), size = 3, halign = "center", valign = "center");
}

// ---------------- double pendulum link 1 ----------------
// Modelled boss-down; PRINTED front-face-down (link1_print) so nothing overhangs. Servo 2 sits at +X = L1, its body
// extending back toward servo 1 (keeps its mass close to axis 1).
module link1() {
    t = BOSS_H + LINK_T;
    difference() {
        union() {
            cylinder(d = HUB_D, h = t);                                  // hub + rear boss
            hull() {                                                     // bar
                translate([0, 0, BOSS_H]) cylinder(d = 16, h = LINK_T);
                translate([L1 - TAB_FAR - 2, -7, BOSS_H]) cube([1, 14, LINK_T]);
            }
            translate([L1 - TAB_FAR - 2, -(BODY_W/2 + CLEAR + 3.5), BOSS_H])  // servo 2 seat
                cube([TAB_FAR + TAB_NEAR + 4, BODY_W + 2*CLEAR + 7, LINK_T]);
            translate([L1 - TAB_FAR - 2, -12, BOSS_H])                     // wider for strap screws
                cube([TAB_FAR + TAB_NEAR + 4, 24, LINK_T]);
        }
        horn_pocket();
        hub_holes(t);
        // servo 2: shaft at L1, body toward the hub  (mirror so +X of window points to hub)
        translate([L1, 0, BOSS_H]) mirror([1, 0, 0]) servo_window(LINK_T);
        // zip-tie slots for servo 2's cable
        for (s = [-1, 1]) translate([24, s * 5.5 - 0.9, BOSS_H - 1]) cube([3.5, 1.8, LINK_T + 2]);
    }
}

// ---------------- base ----------------
// Printed as it stands. Table edge at y = 0, table toward +Y, servo 1 shaft points to -Y.
// The wall's front face is OVERHANG in front of the edge, so arms hang clear of the table.
module base() {
    wall_y0 = -OVERHANG;                 // wall front face
    top = AXIS_Z + TAB_FAR + 4;
    difference() {
        union() {
            translate([-40, wall_y0, 0]) cube([80, 70 + OVERHANG, 6]);          // foot
            translate([-22, wall_y0, 0]) cube([44, WALL_T, top]);               // wall
            for (s = [-1, 1]) translate([s * 16 - 2, wall_y0 + WALL_T - 0.01, 0])  // gussets
                rotate([90, 0, 90]) linear_extrude(4)
                    polygon([[0, 0], [34, 0], [0, top]]);
        }
        // servo 1 window: shaft at (0, *, AXIS_Z), body upward (+Z)
        translate([0, wall_y0 + WALL_T, AXIS_Z]) rotate([90, 0, 0]) rotate([0, 0, 90])
            servo_window(WALL_T);
        // table-edge alignment groove + label
        translate([-40, -0.5, 5.4]) cube([80, 1, 1]);
        translate([24, 3, 5.6]) linear_extrude(1) text("EDGE", size = 4, halign = "center");
        // optional wood-screw holes; otherwise C-clamp the foot behind the gussets
        for (x = [-30, 30]) translate([x, 50, -1]) cylinder(d = 4.5, h = 10);
    }
}

// ---------------- tab clamp strap (print 6: 2 per servo + 2 spare) ----------------
module strap() {
    difference() {
        union() {   // 4 mm bar between body end and tab end; 6 mm ends (still clear the body corners)
            hull() for (y = [-STRAP_X, STRAP_X]) translate([0, y, 0]) cylinder(d = STRAP_W, h = STRAP_T);
            for (y = [-STRAP_X, STRAP_X]) translate([0, y, 0]) cylinder(d = 6, h = STRAP_T);
        }
        for (y = [-STRAP_X, STRAP_X]) translate([0, y, -1]) cylinder(d = SCREW_D, h = STRAP_T + 2);
    }
}

// ---------------- fit coupon ----------------
module coupon() {
    difference() {
        translate([-TAB_NEAR - 4, -12, 0]) cube([TAB_NEAR + TAB_FAR + 8, 24, 4]);
        servo_window(4);
    }
    translate([0, -30, 0]) difference() {
        cylinder(d = HUB_D, h = 4.5);
        horn_pocket();
        hub_holes(4);
    }
}

// ---------------- layout ----------------
// Dummy servo for the assembly view: shaft along +Z at the origin, top face at z = 0,
// body long axis toward +X. BODY_H = 21.2 (SCS0009) or 24.45 (STS3032) below the top face.
STS = true;                    // assembly view uses the taller STS3032 body (worst case)
module servo_dummy() {
    h = STS ? 24.45 : 21.2;
    color("dimgray") {
        translate([-SHAFT_TO_NEAR_END, -BODY_W/2, -h]) cube([BODY_L, BODY_W, h]);
        translate([-TAB_NEAR, -TAB_W/2, -4.4]) cube([TAB_NEAR + TAB_FAR, TAB_W, 1.5]);
        cylinder(d = 12, h = 4);
        cylinder(d = 4, h = 7.2);
    }
}
// Measured 2026-10-28, disc fully seated: tab REAR face to disc FRONT face = 12.48 mm (STS3032),
// 12.5 mm (SCS0009). Tab rear face is 4.4 mm below the servo's top face.
TAB_TO_DISC = 12.5;
HORN_TOP = TAB_TO_DISC - 4.4;     // disc front face above the servo's top face (8.1)
Q1 = 30;  Q2 = -50; // joint angles for the assembly view (deg, 0 = hanging down)
DOUBLE = true;
WEIGHT_AT = 60;     // assembly view: weight stack on this arm hole (0 = none)
// Weight stack, SPLIT across the arm to keep its center of mass close to the servo's output bearing.
// Real stacks are long: 30 mm steel washers are ~8 g per 1.5 mm, so 240 g is ~45 mm of washers.
// Behind the arm there is room for REAR_MAX washers plus the bolt head before the wall (single
// pendulum) or link 1's front face (link 2). Everything else goes in front.
WASHER_OD = 30;  WASHER_T = 1.5;
REAR_MAX  = 4;                     // washers behind the arm (6 mm); checked by PART="clearance"
REAR_N = 4;  FRONT_N = 8;          // assembly view: ~95 g
module weight_stack(rear_n = REAR_N, front_n = FRONT_N) {   // z = 0 at the arm's rear face
    r = rear_n * WASHER_T;  f = front_n * WASHER_T;
    color("silver") {
        translate([0, 0, -r - 2.2]) cylinder(d = 7.6, h = 2.2);              // M4 button head
        translate([0, 0, -r]) cylinder(d = 4, h = r + LINK_T + f + 3.2);     // shank
        if (r > 0) translate([0, 0, -r]) cylinder(d = WASHER_OD, h = r);     // rear washers
        translate([0, 0, LINK_T]) cylinder(d = WASHER_OD, h = f);            // front washers
        translate([0, 0, LINK_T + f]) cylinder(d = 7.7, h = 3.2, $fn = 6);   // nut
    }
}
module arm_loaded(link2 = false) {
    color("orange") if (link2) slim_arm(60); else arm();
    if (WEIGHT_AT > 0) translate([WEIGHT_AT, 0, 0]) weight_stack();
}

// Swept volume of the rear half-stack (bolt head + REAR_MAX washers) at hole r, over a full turn,
// grown by `m` mm on every side. Frame: the arm's rear face at z = 0, axis = Z.
module rear_sweep(r, m = 1) {
    hr = REAR_MAX * WASHER_T + 2.2;
    translate([0, 0, -hr - m]) rotate_extrude($fn = 128)
        translate([max(0, r - WASHER_OD/2 - m), 0]) square([WASHER_OD + 2*m, hr + m]);
}

module assembly() {
    color("tan") base();
    // servo 1: shaft at (0, -, AXIS_Z) pointing -Y, body up; tab rear face on the wall front
    s1_top_y = -OVERHANG - 4.4;
    // frame F1: origin on servo 1's top face, +Z = shaft direction (world -Y), +X = world +Z
    multmatrix([[0, 0, 0, 0], [0, 0, -1, s1_top_y], [1, 0, 0, AXIS_Z], [0, 0, 0, 1]]
               + [[0, -1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]) {
        servo_dummy();
        // rotate about the shaft; joint angle 0 = link pointing world -Z (= frame -X)
        rotate([0, 0, 180 + Q1]) translate([0, 0, HORN_TOP - HORN_POCKET]) {
            if (DOUBLE) {
                color("steelblue") link1();
                translate([L1, 0, BOSS_H + LINK_T + 4.4]) {
                    rotate([0, 0, 180]) servo_dummy();
                    rotate([0, 0, Q2]) translate([0, 0, HORN_TOP - HORN_POCKET])
                        arm_loaded(link2 = true);
                }
            } else arm_loaded();
        }
    }
}

// Clearance check: should render EMPTY. (1) single pendulum: the rear half-stack swept a full turn at
// each hole vs the base and servo 1 (straps are inside r = 23.3 mm, the sweep starts at r >= 29).
// (2) link 2: the same sweep around servo 2's axis vs link 1 and servo 2.
module clearance() {
    s1_top_y = -OVERHANG - 4.4;
    F1 = [[0, -1, 0, 0], [0, 0, -1, s1_top_y], [1, 0, 0, AXIS_Z], [0, 0, 0, 1]];
    intersection() {
        base();
        multmatrix(F1) translate([0, 0, HORN_TOP - HORN_POCKET]) for (r = ARM_HOLES) rear_sweep(r);
    }
    multmatrix(F1) translate([0, 0, HORN_TOP - HORN_POCKET]) intersection() {
        union() { link1(); translate([L1, 0, BOSS_H + LINK_T + 4.4]) rotate([0, 0, 180]) servo_dummy(); }
        translate([L1, 0, BOSS_H + LINK_T + 4.4 + HORN_TOP - HORN_POCKET]) rear_sweep(60);
    }
}

module link1_print() translate([0, 0, BOSS_H + LINK_T]) rotate([180, 0, 0]) link1();

if (PART == "base") base();
else if (PART == "link1") link1_print();
else if (PART == "arm") arm();
else if (PART == "coupon") coupon();
else if (PART == "strap") strap();
else if (PART == "slim_arm") slim_arm(SLIM_L);
else if (PART == "assembly") assembly();
else if (PART == "clearance") clearance();
else {   // print plate: the functional set
    translate([0, 30, 0]) base();
    translate([-55, 0, 0]) arm();
    translate([-55, -30, 0]) slim_arm(60);
    translate([-50, -62, 0]) link1_print();
    for (k = [0 : 5]) translate([40 + k * 9, -70, 0]) strap();
}
