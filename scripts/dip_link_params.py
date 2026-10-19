#!/usr/bin/env python3
"""Parameter files of the DIP-link designs a and b (2026-10-18): every size, clearance and spacing the owner may change,
applied to the generators at run time, with a reproduction check, the full clearance check of a parameter set and its
export.

Owner, 2026-10-18 18:15: "inevitably i will need to tweak sizes, spacing. so either we make these things cad-interfacable
(for a human) or enable programmatic reconfiguration (with a viewer, ideally)".

A parameter file (docs/experiments/20261018-dip_link_board_mount/params/*.json) lists each parameter with its key, value,
unit, what it sets, allowed range and the checks it affects, and the design's discrete choices. Concept a is the final
design of scripts/dip_link_board_mount.py (window plate, dorsal clamp plate, Qwiic header on the clamp plate, tips of the
2.9 mm print set): apply_a() writes a file's values into dip_link_concept_a.P and replaces the final's Qwiic placement
(dip_link_board_mount.qwiic_header_final) by one that reads the file; neither script is edited. Concept b
(scripts/dip_link_concept_b.py) reads its parameter dict from the file itself; apply_b() passes a file's values to it.

    E=~/miniconda3/envs/femtip/bin/python
    RG="$HOME/.claude/bin/resguard.sh run --mem 3G --cpu 400 -- taskset -c 8-11 env NUMPY_MADVISE_HUGEPAGE=0"
    $E scripts/dip_link_params.py init                      # the default files from the committed designs
    $RG $E scripts/dip_link_params.py verify --concept a    # regenerate at a file's values, compare with final/cad
    $RG $E scripts/dip_link_params.py verify --concept b    # ... with concept_b/cad (runs freecadcmd for the B-rep parts)
    $RG $E scripts/dip_link_params.py check --params FILE   # full clearance check of a parameter set -> logs/
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import scipy.spatial  # noqa: F401  (loads the environment's libstdc++ before manifold3d does; matplotlib needs it)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("DIPA_REPO", str(REPO))
os.environ.setdefault("DLF_REPO", str(REPO))

STUDY = REPO / "docs/experiments/20261018-dip_link_board_mount"
PARAMS = STUDY / "params"
DEFAULT = {"a": PARAMS / "20261018-concept_a.json", "b": PARAMS / "20261018-concept_b_final.json",
           "b0": PARAMS / "20261018-concept_b.json"}
WORK = REPO / "logs/20261018-dip_link/params"              # regenerated parts of the checks (untracked)
E_FEMTIP = Path.home() / "miniconda3/envs/femtip/bin/python"
RESGUARD = Path.home() / ".claude/bin/resguard.sh"

# ----------------------------------------------------------------------------------------------- the checks
CHECKS = {
    "clearance": "least signed distance (mm) from a part fixed to the DIP link to the MCP link and the PIP servo over PIP "
                 "-18 to 92 deg; R7 asks 0.5 mm for rigid parts and 1.0 mm for the cable",
    "gap": "distance (mm) from the tip's back face (concept a: x_F 6.45; concept b: the pad's back face) to the chips' "
           "sensing plane, 0.42 mm under the package top; R1 asks at most 0.52 mm",
    "chip_to_magnet": "least distance (mm) from a chip's sensing point to a magnet's surface; smaller resolves finer forces",
    "force": "force resolution at the contact point (N, Cramer-Rao bound of the readout model); needs the sense command",
    "wall": "least wall thickness (mm) of a printed part in the checked sections; R11 asks 0.9 mm (0.8 mm for a PLA "
            "wall printed with two 0.4 mm lines)",
    "fit": "a fit between two parts (pin, screw, nut, key, rail, window); no check computes it, a print decides it",
    "r4": "least distance (mm) from the tip's channels (a) or the pad's grooves and notches (b) to a magnet pocket; R4 asks "
          "2.0 mm",
    "width": "assembly width over the tip's length (mm); R5 asks at most 21.0 mm",
    "mass": "mass of the DIP link, clamp plate or cover, board and fasteners without the tip (g); R12 asks at most 7.6 g",
}

# ----------------------------------------------------------------------------------------------- concept a
# (key, group, label, what it sets, unit, (low, high), checks, committed value). The flat keys shoulder_y0/1 and
# wire_channel_* map to the tuples of dip_link_concept_a.P; the qwiic_* keys set the final's header placement.
SCHEMA_A = [
    ("chip_h", "Board stack", "Chip height", "MLX90393 package height above the FR-4; the window plate prints chip_h - "
     "chip_proud thick", "mm", (0.80, 1.00), ["gap", "chip_to_magnet", "clearance", "wall"], 0.90),
    ("chip_proud", "Board stack", "Chip protrusion", "how far the chip tops stand proud of the window plate into the tip's "
     "back face", "mm", (0.00, 0.15), ["gap", "chip_to_magnet", "clearance", "wall"], 0.05),
    ("fr4", "Board stack", "Board thickness", "FR-4 thickness of the ReSkin 5X (0.6 mm nominal, to be measured)", "mm",
     (0.40, 1.00), ["clearance", "fit"], 0.60),
    ("clamp_t", "Board stack", "Clamp plate thickness", "thickness of the dorsal clamp plate over the board's bare face",
     "mm", (2.00, 4.50), ["clearance", "wall", "mass"], 3.0),
    ("ear_t", "Board stack", "Ear thickness", "thickness of the clamp plate's two ears over the shoulders (below "
     "clamp_t)", "mm", (1.00, 2.50), ["wall", "clearance"], 1.5),
    ("ear_gap", "Board stack", "Ear gap", "gap between each ear and its shoulder, so that the M2 preload presses the board "
     "and not the shoulders", "mm", (0.10, 0.60), ["fit"], 0.30),
    ("half_w", "Widths and windows", "Half-width", "half-width |y| of the window plate and the clamp plate (the board's "
     "10.0 mm and a rim)", "mm", (10.00, 11.00), ["width", "clearance", "wall"], 10.5),
    ("clear_chip", "Widths and windows", "Chip window clearance", "clearance of the window plate's windows around each "
     "MLX90393", "mm", (0.10, 0.40), ["wall", "fit"], 0.20),
    ("clear_passive", "Widths and windows", "Passive window clearance", "clearance of the windows and the tab pocket around "
     "C1-C5 and R1-R2 (solder fillets)", "mm", (0.10, 0.40), ["wall", "fit"], 0.25),
    ("tab_len", "Widths and windows", "Tab length", "length of the cut tab past the board's main square (J1 removed; its "
     "four signal pads end 4.38 mm past it)", "mm", (4.40, 6.00), ["clearance"], 4.9),
    ("pin_d", "Locating pins", "Pin diameter", "diameter of the clamp plate's two locating pins in the board's 2.40 mm "
     "proximal holes (the pin-fit coupon chooses it)", "mm", (2.20, 2.40), ["fit"], 2.35),
    ("pin_hole_d", "Locating pins", "Pin hole diameter", "diameter of the pins' holes in the window plate (reamed)", "mm",
     (2.30, 2.60), ["fit", "wall"], 2.40),
    ("pin_into_plate", "Locating pins", "Pin depth", "how far each pin enters the window plate (less than the window "
     "plate's thickness)", "mm", (0.20, 0.80), ["fit"], 0.55),
    ("diamond_half", "Locating pins", "Diamond pin half-width", "half-width along y of the +y (diamond) pin, which locates "
     "z only", "mm", (0.40, 1.10), ["fit"], 0.60),
    ("cs_d", "M1.6 countersunk screws and nuts", "Screw shank", "shank diameter of the two countersunk screws through the "
     "board's distal holes", "mm", (1.40, 2.00), ["fit", "wall"], 1.6),
    ("cs_k", "M1.6 countersunk screws and nuts", "Head height", "countersunk head height (90 deg seat in the window plate)",
     "mm", (0.80, 1.30), ["fit"], 0.96),
    ("cs_dk", "M1.6 countersunk screws and nuts", "Head diameter", "countersunk head diameter", "mm", (2.60, 3.80),
     ["fit", "wall"], 3.0),
    ("cs_len", "M1.6 countersunk screws and nuts", "Screw length", "countersunk screw length; its end stays inside the nut",
     "mm", (3.00, 5.00), ["fit"], 4.0),
    ("nut_s", "M1.6 countersunk screws and nuts", "Nut across flats", "hex nut size across flats (its pocket in the clamp "
     "plate's back)", "mm", (2.80, 4.00), ["fit", "wall"], 3.2),
    ("nut_m", "M1.6 countersunk screws and nuts", "Nut height", "hex nut height = pocket depth", "mm", (1.00, 1.60),
     ["fit", "wall"], 1.3),
    ("nut_clear", "M1.6 countersunk screws and nuts", "Nut pocket clearance", "extra width of the nut pocket across flats "
     "(0: the nut is pressed in)", "mm", (0.00, 0.20), ["fit"], 0.00),
    ("m2_y", "M2 screws into the shoulders", "Screw |y|", "|y| of the two M2 screws through the ears into the shoulders",
     "mm", (7.50, 9.00), ["wall", "clearance"], 8.1),
    ("m2_z", "M2 screws into the shoulders", "Screw z", "z of the two M2 screws", "mm", (-66.50, -64.50),
     ["wall", "clearance"], -65.35),
    ("m2_pilot", "M2 screws into the shoulders", "Pilot diameter", "pilot hole in the shoulders (the screws form their "
     "threads)", "mm", (1.50, 1.90), ["fit"], 1.7),
    ("m2_depth", "M2 screws into the shoulders", "Pilot depth", "depth of the pilot holes", "mm", (2.50, 4.50),
     ["fit", "wall"], 3.5),
    ("m2_head_d", "M2 screws into the shoulders", "Head diameter", "M2 cheese-head diameter", "mm", (3.40, 4.00),
     ["clearance"], 3.8),
    ("m2_head_k", "M2 screws into the shoulders", "Head height", "M2 cheese-head height", "mm", (1.00, 1.60),
     ["clearance"], 1.3),
    ("m2_len", "M2 screws into the shoulders", "Screw length", "M2 screw length", "mm", (4.00, 6.00), ["fit"], 5.0),
    ("m2_clear_d", "M2 screws into the shoulders", "Ear hole diameter", "clearance hole of the M2 screws in the ears", "mm",
     (2.20, 2.60), ["fit", "wall"], 2.4),
    ("bridge_r", "Bridge and shoulders", "Bridge radius", "radius about the PIP axis of the bridge's top arc (the MCP "
     "link's end lies at 12.70)", "mm", (12.90, 14.00), ["clearance", "wall"], 13.35),
    ("bridge_flat", "Bridge and shoulders", "Bridge flat", "z at which the bridge's arc is cut flat on the palmar side",
     "mm", (-64.00, -62.50), ["clearance"], -63.10),
    ("shoulder_r", "Bridge and shoulders", "Shoulder radius", "radius about the PIP axis of the shoulders' top", "mm",
     (13.00, 14.00), ["clearance", "wall"], 13.45),
    ("shoulder_cap", "Bridge and shoulders", "Shoulder top", "highest z of the shoulders", "mm", (-62.50, -61.00),
     ["clearance"], -61.70),
    ("shoulder_y0", "Bridge and shoulders", "Shoulder inner |y|", "inner |y| of the shoulders; the tab and the wires pass "
     "between them", "mm", (5.50, 7.00), ["wall", "clearance"], 6.0),
    ("shoulder_y1", "Bridge and shoulders", "Shoulder outer |y|", "outer |y| of the shoulders", "mm", (10.00, 11.00),
     ["width", "clearance"], 10.5),
    ("shoulder_z0", "Bridge and shoulders", "Shoulder bottom", "lowest z of the shoulders", "mm", (-68.50, -67.30),
     ["wall"], -67.86),
    ("ear_top", "Bridge and shoulders", "Ear top", "highest z of the clamp plate's ears", "mm", (-63.50, -62.00),
     ["clearance"], -62.60),
    ("tab_cover_top", "Bridge and shoulders", "Tab cover top", "highest z of the clamp plate over the tab", "mm",
     (-64.00, -62.50), ["clearance"], -63.00),
    ("key_yc", "Tip keys and channels", "Key |y|", "|y| of the two dovetail keys' centres", "mm", (3.00, 4.50),
     ["fit", "r4", "wall"], 3.6),
    ("key_root", "Tip keys and channels", "Key root width", "key width at the window plate (the tip's back face)", "mm",
     (1.80, 3.00), ["fit", "wall"], 2.4),
    ("key_head", "Tip keys and channels", "Key head width", "key width at its head, key_depth palmar of the back face",
     "mm", (2.80, 4.40), ["fit", "wall", "r4"], 3.6),
    ("key_depth", "Tip keys and channels", "Key depth", "how far the keys reach into the tip from its back face", "mm",
     (1.20, 2.40), ["fit", "r4"], 1.8),
    ("key_len", "Tip keys and channels", "Key length", "key length from the tip's mount face", "mm", (3.00, 7.00),
     ["fit", "r4"], 5.0),
    ("detent_h", "Tip keys and channels", "Detent height", "height of the detent on each key's head", "mm", (0.00, 0.40),
     ["fit"], 0.25),
    ("detent_z", "Tip keys and channels", "Detent z", "z of the detents (inside the keys' length)", "mm", (-72.00, -69.50),
     ["fit"], -71.60),
    ("fit_flank", "Tip keys and channels", "Channel interference", "how much narrower than the key the tip's channel is "
     "per flank", "mm", (0.00, 0.15), ["fit"], 0.05),
    ("fit_floor", "Tip keys and channels", "Channel floor clearance", "how much deeper than the key's head the channel "
     "is", "mm", (0.00, 0.20), ["fit", "r4"], 0.10),
    ("fit_len", "Tip keys and channels", "Channel extra length", "how much longer than the key the channel is", "mm",
     (0.00, 0.30), ["fit"], 0.10),
    ("groove_r", "Tip keys and channels", "Detent groove radius", "radius of the detent's groove in the channel floor",
     "mm", (0.20, 0.50), ["fit", "r4"], 0.35),
    ("wire_channel_z0", "Wire channel", "Channel distal z", "distal z of the wire channel beside the tab", "mm",
     (-66.50, -65.00), ["wall"], -65.70),
    ("wire_channel_z1", "Wire channel", "Channel proximal z", "proximal z of the wire channel", "mm", (-65.00, -64.00),
     ["wall"], -64.50),
    ("wire_channel_x0", "Wire channel", "Channel palmar x", "palmar x of the wire channel (the tip's back face is 6.45)",
     "mm", (6.45, 7.00), ["wall"], 6.55),
    ("qwiic_y", "Qwiic header on the clamp plate", "Header y", "y of the SM04B-SRSS-TB header's centre on the clamp "
     "plate's back", "mm", (-3.00, 3.00), ["clearance"], 0.0),
    ("qwiic_z_back", "Qwiic header on the clamp plate", "Header back z", "z of the header's back; its mating face is "
     "4.25 mm proximal (toward the PIP joint), where the plug enters", "mm", (-80.00, -74.00), ["clearance"], -77.0),
    ("qwiic_pocket_depth", "Qwiic header on the clamp plate", "Pocket depth", "depth of the header's pocket in the clamp "
     "plate's back (the clamp plate under it is clamp_t minus this)", "mm", (0.50, 1.50), ["clearance", "wall"], 1.0),
    ("qwiic_pocket_clear", "Qwiic header on the clamp plate", "Pocket clearance", "clearance of the pocket around the "
     "header's body and leads", "mm", (0.00, 0.15), ["fit"], 0.05),
    ("qwiic_groove_depth", "Qwiic header on the clamp plate", "Pigtail groove depth", "depth of the groove that takes the "
     "four wires from the clamp plate's notch to the header's leads", "mm", (0.40, 1.00), ["wall"], 0.6),
]
DESIGN_A = [
    ("qwiic", "Qwiic connection", "header_on_clamp_plate",
     {"header_on_clamp_plate": "J1 removed and the tab cut; a pigtail from J1's pads to an SM04B-SRSS-TB header in a pocket "
                               "of the clamp plate's back, mating face proximal (dip_link_board_mount.py)"}),
    ("tips", "Printed tips", "seat2p9",
     {"seat2p9": "the 2.9 mm-seat print set the owner prints (dip_link_board_mount.SHEETS), with two key channels cut"}),
]

# ----------------------------------------------------------------------------------------------- concept b
SCHEMA_B = [
    ("t_behind", "Cassette depth and board stack", "TPU behind the magnets", "TPU left behind the block's magnets; the "
     "pad's back face, the cover's front face and the chip tops lie at x_F -3.55 + t_behind. The coarse finite elements "
     "exist at 3, 4, 6 and 10 mm", "mm", (3.00, 6.00), ["chip_to_magnet", "force", "clearance", "r4", "mass"], 4.0),
    ("chip_h", "Cassette depth and board stack", "Chip height", "MLX90393 package height; the cover's web is as thick as "
     "the chips", "mm", (0.80, 1.00), ["clearance", "wall"], 0.90),
    ("fr4", "Cassette depth and board stack", "Board thickness", "FR-4 thickness of the ReSkin 5X (0.6 mm nominal, to be "
     "measured)", "mm", (0.40, 1.00), ["clearance"], 0.60),
    ("rim_h", "Cassette depth and board stack", "Seat pocket depth", "depth of the board's seat pocket in the tongue; the "
     "rim stops short of the cover's back face", "mm", (0.20, 0.60), ["fit", "wall"], 0.45),
    ("t_tongue", "Cassette depth and board stack", "Tongue thickness", "thickness of the DIP link's tongue behind the "
     "board's seat", "mm", (2.50, 5.00), ["clearance", "wall", "mass"], 3.6),
    ("y_rim", "Widths and windows", "Half-width", "half-width |y| of the cover and the tongue (the board's 10.0 mm and a "
     "rim)", "mm", (10.00, 11.00), ["width", "clearance"], 10.5),
    ("window_margin", "Widths and windows", "Window clearance", "clearance of the cover's windows around every part on "
     "the board", "mm", (0.10, 0.40), ["wall", "fit"], 0.20),
    ("cover_tab_y", "Widths and windows", "Cover notch half-width", "half-width of the cover's notch over the tab", "mm",
     (5.20, 6.50), ["wall"], 5.60),
    ("cover_tab_z", "Widths and windows", "Cover notch z", "z where the cover stops over the tab", "mm", (-69.00, -67.50),
     ["wall"], -68.30),
    ("tab_cut_v", "Widths and windows", "Tab cut", "board-frame v of the tab cut (R1 and R2 at v -11.40 stay)", "mm",
     (-12.50, -11.60), ["clearance"], -11.70),
    ("screw_d", "M1.6 screws and nuts", "Screw shank", "shank diameter of the four cheese-head screws through cover, board "
     "and tongue", "mm", (1.40, 2.00), ["fit"], 1.6),
    ("screw_clear", "M1.6 screws and nuts", "Screw hole", "clearance hole of the screws in the cover and the tongue", "mm",
     (1.60, 2.20), ["fit", "wall"], 1.8),
    ("head_d", "M1.6 screws and nuts", "Head diameter", "cheese-head diameter", "mm", (2.60, 3.80), ["fit", "r4"], 3.0),
    ("head_h", "M1.6 screws and nuts", "Head height", "cheese-head height; the heads stand palmar of the cover", "mm",
     (0.80, 1.40), ["fit", "r4"], 1.0),
    ("screw_l", "M1.6 screws and nuts", "Screw length", "screw length", "mm", (4.00, 6.00), ["fit"], 5.0),
    ("nut_af", "M1.6 screws and nuts", "Nut across flats", "hex nut size across flats", "mm", (2.80, 4.00), ["fit"], 3.2),
    ("nut_h", "M1.6 screws and nuts", "Nut height", "hex nut height", "mm", (1.00, 1.60), ["fit"], 1.3),
    ("nut_pocket_af", "M1.6 screws and nuts", "Nut pocket across flats", "nut pocket size across flats on the tongue's "
     "back", "mm", (3.00, 4.20), ["fit", "wall"], 3.35),
    ("nut_pocket_h", "M1.6 screws and nuts", "Nut pocket depth", "nut pocket depth from the tongue's back", "mm",
     (1.10, 2.00), ["fit", "wall"], 1.45),
    ("head_notch_d", "M1.6 screws and nuts", "Pad notch diameter", "diameter of the pad's four corner notches around the "
     "screw heads", "mm", (3.00, 4.20), ["fit", "r4"], 3.4),
    ("head_notch_h", "M1.6 screws and nuts", "Pad notch depth", "depth of the notches into the pad's back face", "mm",
     (0.90, 1.60), ["fit", "r4"], 1.2),
    ("rail_y", "Rails on the cover, grooves in the pad", "Rail |y|", "|y| of the two dovetail rails on the cover's front "
     "face", "mm", (3.80, 5.00), ["fit", "r4"], 4.40),
    ("rail_base", "Rails on the cover, grooves in the pad", "Rail base width", "rail width at the cover", "mm",
     (1.00, 1.80), ["fit", "wall"], 1.30),
    ("rail_top", "Rails on the cover, grooves in the pad", "Rail top width", "rail width at its top (palmar)", "mm",
     (1.40, 2.40), ["fit"], 1.90),
    ("rail_h", "Rails on the cover, grooves in the pad", "Rail height", "rail height into the pad", "mm", (0.60, 1.40),
     ["fit", "r4"], 1.00),
    ("rail_z0", "Rails on the cover, grooves in the pad", "Rail distal z", "distal end of the rails", "mm",
     (-86.00, -80.00), ["fit"], -83.80),
    ("rail_z1", "Rails on the cover, grooves in the pad", "Rail proximal z", "proximal end of the rails", "mm",
     (-70.00, -67.60), ["fit"], -68.60),
    ("groove_fit", "Rails on the cover, grooves in the pad", "Groove interference", "how much narrower than the rail the "
     "pad's groove is per flank (the pad's preload)", "mm", (0.00, 0.15), ["fit"], 0.05),
    ("groove_floor", "Rails on the cover, grooves in the pad", "Groove floor clearance", "clearance between the rail's top "
     "and the groove's floor", "mm", (0.00, 0.20), ["fit", "r4"], 0.05),
    ("wire_slot_y0", "Wire exit and cable tie", "Wire exit -y", "-y edge of the wire exit through the tongue under the "
     "tab's root", "mm", (-5.00, -3.00), ["wall"], -4.0),
    ("wire_slot_y1", "Wire exit and cable tie", "Wire exit +y", "+y edge of the wire exit", "mm", (3.00, 5.00), ["wall"],
     4.0),
    ("wire_slot_z0", "Wire exit and cable tie", "Wire exit distal z", "distal z of the wire exit", "mm", (-69.00, -67.80),
     ["wall"], -68.30),
    ("wire_slot_z1", "Wire exit and cable tie", "Wire exit proximal z", "proximal z of the wire exit", "mm",
     (-67.80, -67.00), ["wall"], -67.40),
    ("tie_y0", "Wire exit and cable tie", "Tie block -y", "-y edge of the cable-tie block on the tongue's back", "mm",
     (-4.00, -2.00), ["clearance"], -3.0),
    ("tie_y1", "Wire exit and cable tie", "Tie block +y", "+y edge of the tie block", "mm", (2.00, 4.00), ["clearance"],
     3.0),
    ("tie_z0", "Wire exit and cable tie", "Tie block distal z", "distal z of the tie block", "mm", (-77.00, -73.00),
     ["clearance"], -75.2),
    ("tie_z1", "Wire exit and cable tie", "Tie block proximal z", "proximal z of the tie block", "mm", (-72.00, -69.50),
     ["clearance"], -70.6),
    ("tie_x_top", "Wire exit and cable tie", "Tie block top x", "dorsal face of the tie block", "mm", (6.80, 8.50),
     ["clearance"], 7.75),
    ("tie_tunnel_x0", "Wire exit and cable tie", "Tunnel palmar x", "palmar x of the tie's tunnel (the tongue's back face "
     "at the committed values)", "mm", (5.00, 6.50), ["wall"], 5.55),
    ("tie_tunnel_x1", "Wire exit and cable tie", "Tunnel dorsal x", "dorsal x of the tie's tunnel", "mm", (6.20, 7.50),
     ["wall"], 6.75),
    ("tie_tunnel_z0", "Wire exit and cable tie", "Tunnel distal z", "distal z of the tie's tunnel", "mm", (-75.00, -73.00),
     ["wall"], -74.2),
    ("tie_tunnel_z1", "Wire exit and cable tie", "Tunnel proximal z", "proximal z of the tie's tunnel", "mm",
     (-72.50, -70.80), ["wall"], -71.6),
    ("keep_margin", "Bridge and spine", "Bridge margin", "the bridge's top stays this far from the volume the MCP link and "
     "the PIP servo sweep (0.5 mm of R7 and the keep-out's 0.3 mm)", "mm", (0.50, 1.50), ["clearance", "wall"], 0.80),
    ("x_spine_back", "Bridge and spine", "Spine back x", "dorsal face of the spine that joins both clevis arms behind the "
     "tongue (the current plate's back face)", "mm", (8.00, 11.00), ["clearance", "mass"], 9.45),
    ("z_web_low", "Bridge and spine", "Web bottom z", "lowest z of the webs beside the tongue (|y| 10.5 to the arms)",
     "mm", (-72.00, -68.00), ["width", "wall"], -70.05),
    ("qwiic_y", "Qwiic header on the tongue", "Header y", "y of the SM04B-SRSS-TB header's centre on the tongue's back "
     "(design qwiic = header_on_tongue)", "mm", (-2.00, 2.00), ["wall"], 0.0),
    ("qwiic_z_back", "Qwiic header on the tongue", "Header back z", "z of the header's back; its mating face is 4.25 mm "
     "proximal, the plug's end 6.25 mm, and the cable then runs over the tie block", "mm", (-87.50, -83.00),
     ["clearance", "wall"], -86.4),
    ("qwiic_pocket_depth", "Qwiic header on the tongue", "Pocket depth", "depth of the header's pocket in the tongue's back "
     "(the tongue under it is t_tongue minus this)", "mm", (0.50, 1.50), ["wall"], 1.0),
    ("qwiic_pocket_clear", "Qwiic header on the tongue", "Pocket clearance", "clearance of the pocket around the header's "
     "body and leads", "mm", (0.00, 0.15), ["fit"], 0.05),
    ("qwiic_groove_depth", "Qwiic header on the tongue", "Pigtail groove depth", "depth of the groove that takes the four "
     "wires from the wire exit to the header's leads", "mm", (0.40, 1.00), ["wall"], 0.6),
    ("qwiic_groove_y0", "Qwiic header on the tongue", "Groove inner y", "inner y edge of the pigtail groove (beside the tie "
     "block, which ends at tie_y1)", "mm", (3.10, 4.50), ["wall"], 3.4),
    ("qwiic_groove_y1", "Qwiic header on the tongue", "Groove outer y", "outer y edge of the pigtail groove (the nut pockets "
     "begin at y 6.07)", "mm", (4.50, 5.50), ["wall"], 5.0),
]
DESIGN_B = [
    ("qwiic", "Qwiic connection", "none",
     {"none": "the tab cut at tab_cut_v, J1 removed, four wires from the vias through the exit slot (the committed "
              "concept b)",
      "header_on_tongue": "J1 removed with the tab (no orientation of J1 fits, concept_b/final/*_qwiic.json); the four "
                          "wires run from the exit slot down a groove of the tongue's back to an SM04B-SRSS-TB header in "
                          "a pocket there, mating face proximal; the cable leaves over the tie block"}),
    ("pads", "Printed pads", "seat2p7",
     {"seat2p7": "pads cut from the 2.7 mm-seat print STLs (the committed concept b)",
      "seat2p9": "pads cut from the 2.9 mm-seat print STLs the owner prints"}),
]

TITLES = {"a": "DIP link concept a, final: window plate, dorsal clamp plate, Qwiic header on the clamp plate",
          "b": "DIP link concept b, final: the board in a cassette in place of the pad's dorsal 6 mm, Qwiic header on the "
               "tongue, pads of the 2.9 mm print set",
          "b0": "DIP link concept b as committed on 2026-10-18 12:14: the board in a cassette, tab cut and wires soldered, "
                "pads of the 2.7 mm print set"}
REPRODUCES = {"a": "docs/experiments/20261018-dip_link_board_mount/final/cad and final/print "
                   "(scripts/dip_link_board_mount.py build)",
              "b": "docs/experiments/20261018-dip_link_board_mount/concept_b/final/cad (scripts/dip_link_concept_b.py prep, cad "
                   "with this file in $DIPB_PARAMS)",
              "b0": "docs/experiments/20261018-dip_link_board_mount/concept_b/cad (scripts/dip_link_concept_b.py prep, cad)"}
GENERATOR = {"a": "scripts/dip_link_params.py apply_a: dip_link_concept_a.P and dip_link_board_mount.qwiic_header_final",
             "b": "scripts/dip_link_concept_b.py (its dict P), through scripts/dip_link_params.py apply_b",
             "b0": "scripts/dip_link_concept_b.py (its dict P), through scripts/dip_link_params.py apply_b"}
_S = "docs/experiments/20261018-dip_link_board_mount"
OUTPUTS = {  # where the committed design's files are; a saved parameter set has none and is built under logs/ or exports/
    "a": dict(cad=f"{_S}/final/cad", print=f"{_S}/final/print"),
    "b": dict(cad=f"{_S}/concept_b/final/cad", print=f"{_S}/concept_b/final/print", media=f"{_S}/concept_b/final/media",
              evaluation=f"{_S}/concept_b/final/evaluation.json", work="logs/20261018-dip_link/concept_b_final",
              prefix="20261018-concept_b_final"),
    "b0": dict(cad=f"{_S}/concept_b/cad", media=f"{_S}/concept_b/media", evaluation=f"{_S}/concept_b/evaluation.json",
               work="logs/20261018-dip_link/concept_b", prefix="20261018-concept_b"),
}
DESIGN_VALUES = {"a": {}, "b": {"qwiic": "header_on_tongue", "pads": "seat2p9"}, "b0": {"qwiic": "none", "pads": "seat2p7"}}
FRAME = ("F of spec.json: mm, x toward the dorsal side, y along the PIP axis, z up; the straight finger hangs along -z, the "
         "PIP axis passes through (0, y, -51.55), the tip's back face lies at x 6.45 and its mount face at z -67.30")


def schema(concept):
    return SCHEMA_A if concept == "a" else SCHEMA_B


def design_schema(concept):
    return DESIGN_A if concept == "a" else DESIGN_B


def default_doc(name):
    """The parameter file of a committed design: 'a', 'b' (b's final) or 'b0' (b as committed before the Qwiic study)."""
    concept = name[0]
    dv = DESIGN_VALUES[name]
    return dict(
        title=f"Parameters of {TITLES[name]}", concept=concept, date=time.strftime("%Y-%m-%d"),
        reproduces=REPRODUCES[name], generator=GENERATOR[name], frame=FRAME,
        note=("Edit a value inside its range, save the file under a new dated name in this folder, then build, check or "
              "export it (scripts/dip_link_params.py, or the viewer scripts/dip_link_viewer.py). 'default' is the "
              "committed design's value; 'checks' names the checks each parameter changes (defined under 'checks'). "
              "'outputs' names the committed design's folders; a saved set has none, so it never overwrites them."),
        outputs=OUTPUTS[name], checks=CHECKS,
        design=[dict(key=k, label=lab, value=dv.get(k, v), options=opts) for k, lab, v, opts in design_schema(concept)],
        parameters=[dict(key=k, group=g, label=lab, value=val, default=val, unit=u, sets=s, range=list(rng), checks=c)
                    for k, g, lab, s, u, rng, c, val in schema(concept)])


# ----------------------------------------------------------------------------------------------- reading a file
def load(path):
    """(concept, {key: value}, {design key: value}, document) of a parameter file."""
    doc = json.loads(Path(path).read_text())
    vals = {q["key"]: float(q["value"]) for q in doc["parameters"]}
    des = {d["key"]: d["value"] for d in doc.get("design", [])}
    return doc["concept"], vals, des, doc


def problems(concept, vals, des=None):
    """Values outside their allowed ranges, unknown keys and missing keys, as sentences (empty when the set is valid)."""
    out = []
    known = {k: rng for k, _, _, _, _, rng, _, _ in schema(concept)}
    for k, v in vals.items():
        if k not in known:
            out.append(f"{k}: not a parameter of concept {concept}")
        elif not (known[k][0] - 1e-9 <= v <= known[k][1] + 1e-9):
            out.append(f"{k} = {v:g} mm lies outside {known[k][0]:g}..{known[k][1]:g} mm")
    out += [f"{k}: missing" for k in known if k not in vals]
    for k, _, _, opts in design_schema(concept):
        if des and k in des and des[k] not in opts:
            out.append(f"{k} = {des[k]}: not one of {', '.join(opts)}")
    if concept == "a" and not out:
        if vals["ear_t"] >= vals["clamp_t"]:
            out.append("ear_t must be less than clamp_t")
        if vals["pin_into_plate"] >= vals["chip_h"] - vals["chip_proud"]:
            out.append("pin_into_plate must be less than the window plate (chip_h - chip_proud)")
        if vals["qwiic_pocket_depth"] >= vals["clamp_t"] - 0.6:
            out.append("qwiic_pocket_depth leaves less than 0.6 mm of clamp plate under the header")
        if not (-67.30 - vals["key_len"] < vals["detent_z"] < -67.30):
            out.append("detent_z must lie on the keys (-67.30 - key_len .. -67.30)")
        if vals["key_head"] <= vals["key_root"]:
            out.append("key_head must exceed key_root (a dovetail)")
    if concept == "b" and not out:
        if vals["rail_top"] <= vals["rail_base"]:
            out.append("rail_top must exceed rail_base (a dovetail)")
        if vals["tie_tunnel_x1"] >= vals["tie_x_top"]:
            out.append("the tie's tunnel must stay under the tie block's top")
    return out


# ----------------------------------------------------------------------------------------------- concept a
_A = {}


def modules_a():
    """dip_link_concept_a and dip_link_board_mount, imported once (the latter points the former at final/ and the 2.9 mm
    tips, and replaces its clamp plate by the final's with the Qwiic pocket)."""
    if not _A:
        import dip_link_board_mount as DLF
        import dip_link_concept_a as A
        _A.update(A=A, DLF=DLF, qwiic=dict(y=0.0, z_back=-77.0, pocket_depth=1.0, pocket_clear=0.05, groove_depth=0.6))
        DLF.qwiic_header_final = qwiic_header_param
    return _A["A"], _A["DLF"]


def qwiic_header_param():
    """dip_link_board_mount.qwiic_header_final with its placement read from the parameter file: the header's centre y and
    back z, the pocket's depth and clearance and the pigtail groove's depth. At the committed values it returns the
    original's boxes (to the last bit of each coordinate but 1e-15 mm)."""
    A, DLF = modules_a()
    Q, QW = _A["qwiic"], DLF.QW
    L = A.levels()
    xb = L["x_c1"]                                 # the clamp plate's back face
    x0 = xb - Q["pocket_depth"]                    # pocket floor = the header's base
    y0, z_back, c, g = Q["y"], Q["z_back"], Q["pocket_clear"], Q["groove_depth"]
    z_mate = z_back + QW["hdr_d"]
    parts = [("q_header", (x0, y0 - QW["hdr_w"] / 2, z_back), (x0 + QW["hdr_h"], y0 + QW["hdr_w"] / 2, z_mate)),
             ("q_leads", (x0, y0 - 1.8, z_back - QW["lead"]), (x0 + 0.3, y0 + 1.8, z_back)),
             ("q_plug", (x0, y0 - QW["plug_w"] / 2, z_mate), (x0 + QW["plug_h"], y0 + QW["plug_w"] / 2, z_mate + QW["plug_out"]))]
    for k in range(4):
        y = y0 + (k - 1.5) * 1.0
        zc = z_mate + QW["plug_out"]
        parts.append((f"q_wire{k}", (x0 + 1.4 - 0.4, y - 0.4, zc), (x0 + 1.4 + 0.4, y + 0.4, zc + QW["wire_run"])))
    parts.append(("q_pigtail", (xb - g, 4.6, z_back - QW["lead"] - 0.5), (xb, 5.8, -64.3)))
    parts.append(("q_pigtail_turn", (xb - g, y0 - 1.8, z_back - QW["lead"] - 0.5), (xb, 5.8, z_back - QW["lead"])))
    cuts = [("pocket_body", (x0, y0 - QW["hdr_w"] / 2 - c, z_back), (xb + 0.1, y0 + QW["hdr_w"] / 2 + c, z_mate + c)),
            ("pocket_leads", (x0, y0 - 1.8 - c, z_back - QW["lead"] - 0.1), (xb + 0.1, y0 + 1.8 + c, z_back + 0.01)),
            ("groove_run", (xb - g, 4.4, z_back - QW["lead"] - g), (xb + 0.1, 6.0, -63.9)),
            ("groove_turn", (xb - g, y0 - 1.8 - c, z_back - QW["lead"] - g), (xb + 0.1, 6.0, z_back - QW["lead"] + 0.01))]
    return parts, cuts


def apply_a(vals):
    """Write a parameter set into dip_link_concept_a.P (in place: the generator's functions hold P as their default
    argument) and into the Qwiic placement."""
    A, _ = modules_a()
    P = A.P
    for k, v in vals.items():
        if k in P and not isinstance(P[k], (tuple, dict)):
            P[k] = float(v)
    P["shoulder_y"] = (float(vals["shoulder_y0"]), float(vals["shoulder_y1"]))
    P["wire_channel"] = dict(z=(float(vals["wire_channel_z0"]), float(vals["wire_channel_z1"])),
                             x0=float(vals["wire_channel_x0"]))
    _A["qwiic"].update(y=float(vals["qwiic_y"]), z_back=float(vals["qwiic_z_back"]),
                       pocket_depth=float(vals["qwiic_pocket_depth"]), pocket_clear=float(vals["qwiic_pocket_clear"]),
                       groove_depth=float(vals["qwiic_groove_depth"]))


def values_from_a():
    """The flat parameter values that dip_link_concept_a.P and the final's Qwiic placement hold now."""
    A, _ = modules_a()
    P, Q = A.P, _A["qwiic"]
    out = {k: P[k] for k, *_ in SCHEMA_A if k in P and not isinstance(P[k], (tuple, dict))}
    out.update(shoulder_y0=P["shoulder_y"][0], shoulder_y1=P["shoulder_y"][1], wire_channel_z0=P["wire_channel"]["z"][0],
               wire_channel_z1=P["wire_channel"]["z"][1], wire_channel_x0=P["wire_channel"]["x0"],
               qwiic_y=Q["y"], qwiic_z_back=Q["z_back"], qwiic_pocket_depth=Q["pocket_depth"],
               qwiic_pocket_clear=Q["pocket_clear"], qwiic_groove_depth=Q["groove_depth"])
    return out


def redirect_a(root):
    """Point the final's generator at another folder (cad/, print/, media/, work/ under root), so a parameter set's parts
    never overwrite final/."""
    A, DLF = modules_a()
    root = Path(root)
    DLF.FINAL, DLF.CAD, DLF.PRINT, DLF.MEDIA, DLF.WORK = root, root / "cad", root / "print", root / "media", root / "work"
    A.OUT, A.MEDIA, A.CTX = root, root / "media", root / "work"
    for d in (root / "cad", root / "print", root / "media", root / "work"):
        d.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------------------------- concept b
def apply_b(vals, des=None):
    import dip_link_concept_b as B
    B.apply(vals, des)
    return B


# ----------------------------------------------------------------------------------------------- commands
def cmd_init(a):
    PARAMS.mkdir(parents=True, exist_ok=True)
    A, _ = modules_a()
    now = values_from_a()
    diff = {k: (now[k], v) for k, _, _, _, _, _, _, v in SCHEMA_A if abs(now[k] - v) > 1e-12}
    if diff:
        raise SystemExit(f"dip_link_concept_a.P differs from the schema's committed values: {diff}")
    for c in a.concept:
        dst = DEFAULT[c]
        if dst.exists() and not a.force:
            print(f"{dst.relative_to(REPO)} exists (pass --force to rewrite it)")
            continue
        dst.write_text(json.dumps(default_doc(c), indent=1) + "\n")
        print(f"{dst.relative_to(REPO)}: {len(schema(c[0]))} parameters")


def _volume(path):
    import trimesh
    return float(trimesh.load(path, force="mesh").volume)


def compare(pairs):
    """[(name, regenerated STL, committed STL)] -> rows with both volumes and their difference (mm^3)."""
    rows = []
    for name, new, old in pairs:
        if not Path(old).exists():
            rows.append(dict(part=name, committed=None, regenerated=str(Path(new).relative_to(REPO)), ok=False))
            print(f"{name:34s} no committed file", flush=True)
            continue
        vn, vo = _volume(new), _volume(old)
        rows.append(dict(part=name, committed=str(Path(old).relative_to(REPO)), volume_committed_mm3=round(vo, 4),
                         volume_regenerated_mm3=round(vn, 4), difference_mm3=round(vn - vo, 6), ok=bool(abs(vn - vo) <= 0.01)))
        print(f"{name:34s} committed {vo:12.4f}  regenerated {vn:12.4f}  difference {vn - vo:+.6f} mm3", flush=True)
    return rows


def verify_a(path, out):
    """Regenerate concept a's final at a parameter file's values into out/ (dip_link_board_mount.cmd_build, redirected) and
    compare every STL with final/cad and final/print."""
    A, DLF = modules_a()
    concept, vals, des, _ = load(path)
    assert concept == "a", path
    committed = {"cad": DLF.CAD, "print": DLF.PRINT}
    apply_a(vals)
    redirect_a(out)
    DLF.cmd_build(argparse.Namespace())
    pairs = []
    for d, src in committed.items():                   # every STL the build writes (fit coupons come from other commands)
        for new in sorted((Path(out) / d).glob("*.stl")):
            pairs.append((f"{d}/{new.stem.replace(DLF.PFX + '_', '')}", new, src / new.name))
    return compare(pairs)


def verify_b(path, out):
    """Regenerate concept b at a parameter file's values into out/: the arms, the bridge's profile and the pads
    (dip_link_concept_b prep) and the DIP link, its new body and the cover (fc_build, in freecadcmd); compare every part's
    STL with the folder the file's outputs name (concept_b/cad, or concept_b/final/cad for b's final)."""
    concept, vals, des, doc = load(path)
    assert concept == "b", path
    out = Path(out)
    env = dict(os.environ, DIPB_PARAMS=str(Path(path).resolve()), DIPB_OUT=str(out / "cad"), DIPB_WORK=str(out / "work"))
    (out / "cad").mkdir(parents=True, exist_ok=True)
    (out / "work").mkdir(parents=True, exist_ok=True)
    r = subprocess.run([str(E_FEMTIP), str(REPO / "scripts/dip_link_concept_b.py"), "prep"], env=env, cwd=str(REPO),
                       capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-3000:], flush=True)
    if r.returncode:
        raise SystemExit("concept b prep failed")
    r = subprocess.run(["freecadcmd", str(REPO / "scripts/dip_link_concept_b.py")], env=dict(env, DIPB_FREECAD="1"),
                       cwd=str(REPO), capture_output=True, text=True)
    print("\n".join(l[:300] for l in r.stdout.splitlines() if l.strip() and "%" not in l)[-4000:], flush=True)
    if r.returncode or "done" not in r.stdout:
        print(r.stderr[-3000:])
        raise SystemExit("freecadcmd fc_build failed")
    src = REPO / doc["outputs"]["cad"]
    pre = doc["outputs"].get("prefix", "20261018-concept_b") + "_"
    pairs = [(new.stem.replace(pre, ""), new, src / new.name) for new in sorted((out / "cad").glob("*.stl"))]
    return compare(pairs)


def cmd_verify(a):
    t0 = time.time()
    rec = {}
    for c in a.concept:
        path = Path(a.params) if a.params else DEFAULT[c]
        out = WORK / f"verify_{c}_{time.strftime('%Y%m%d-%H%M%S')}"
        rows = verify_a(path, out) if c == "a" else verify_b(path, out)       # b, b0: both through verify_b
        rec[c] = dict(params=str(path.relative_to(REPO)), regenerated_in=str(out.relative_to(REPO)), tolerance_mm3=0.01,
                      all_ok=all(r["ok"] for r in rows), parts=rows)
        print(f"concept {c}: {sum(r['ok'] for r in rows)}/{len(rows)} parts within 0.01 mm3 [{time.time() - t0:.0f} s]",
              flush=True)
        dst = PARAMS / f"{Path(path).stem}_verify.json"
        dst.write_text(json.dumps(dict(
            what="volumes of the parts regenerated at this parameter file's values against the committed STL files",
            when=time.strftime("%Y-%m-%d %H:%M"), **rec[c]), indent=1) + "\n")
        print("->", dst.relative_to(REPO), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("init", help="write the default parameter files from the committed designs")
    s.add_argument("--concept", nargs="+", default=["a", "b", "b0"], choices=["a", "b", "b0"])
    s.add_argument("--force", action="store_true")
    s = sub.add_parser("verify", help="regenerate at a file's values and compare volumes with the committed STL files")
    s.add_argument("--concept", nargs="+", default=["a", "b", "b0"], choices=["a", "b", "b0"])
    s.add_argument("--params", help="parameter file (default: the concept's committed file)")
    a = ap.parse_args()
    {"init": cmd_init, "verify": cmd_verify}[a.cmd](a)


if __name__ == "__main__":
    main()
