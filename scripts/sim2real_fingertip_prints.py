#!/usr/bin/env python3
"""Siraya Tech Flex TPU 85A print projects for the deployment-readiness study (docs/handoff/
20261016-sim2real_fingertip_readiness.md, step 2), a companion of scripts/metric_magnet_prints.py (MP).

The 2026-10-16 magnet prints (docs/experiments/20261016-metric_magnet_prints/print/) are 95A projects: Generic FLEX @XL on
tool 2 of the owner's v17 template, 230 C, extrusion multiplier 1.08, 2 mm3/s. The owner prints Siraya Tech Flex TPU 85A
on extruder 2 of the Prusa XL 5T. This builder derives the v17 template with Siraya's Prusa XL 0.4 mm profile in filament
slot 2 and Siraya's print speeds, then writes the same geometry (MP.build: 3 x 2 mm magnets, 2.7 mm seats, the depths of
the 95A prints) as 85A projects at 30 % and 50 % gyroid, slices each and checks the G-code.

  template   print/20261016-siraya85a_template_v17_tool2.3mf (config only) and template_changes.json: every key taken
             from Siraya's 3MF ("Siraya Tech TPU 85A XL 0.4mm 20260610.3mf"), with the keys kept from the template and why
  build      the tips: block four, five and one-wall (pocket walls 1), and the three curved four-magnet finalists
  coupon     the seat coupon, two-wall and one-wall, in 85A
  plate      several tips of the set on one bed with their pauses merged (default: the four candidate shapes at 30 %
             gyroid with 2.9 mm seats), sliced and checked per tip
  readme     print/README.md from the sheets

Kept from the template on purpose: the 5T printer settings and tool-2 assignment, two perimeters and four top and four
bottom layers (the recipe of every finite-element model and of the 1.8 mm pocket-wall rule), thin bridges (Siraya's thick
bridges lay a 0.4 mm round strand whose underside hangs 0.2 mm below a 0.2 mm layer, onto a magnet 0.1-0.2 mm under the
bridge), and the Arachne perimeter generator (the loop counts of the one-wall checks). Infill is the design variable.

    ~/.claude/bin/resguard.sh run --mem 3G --cpu 200 -- taskset -c 18,19 ~/miniconda3/envs/femtip/bin/python \\
        scripts/sim2real_fingertip_prints.py build
"""
from __future__ import annotations

import argparse
import io
import json
import math
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import eflesh_build as EB  # noqa: E402
import metric_magnet_prints as MP  # noqa: E402
import tpu_tip_design as TD  # noqa: E402

DATE = "20261016"
D = ROOT / "docs/experiments/20261016-sim2real_fingertip_readiness"
OUT = D / "print"
WORK = ROOT / "logs/20261016-sim2real_fingertip_readiness/print"
SIRAYA_ZIP = Path.home() / "Downloads/Siraya Tech TPU 85A XL.zip"
SIRAYA_FILAMENT_INI = (Path.home() / ".var/app/com.prusa3d.PrusaSlicer/config/PrusaSlicer/filament/SirayaTech 85A.ini")
TEMPLATE = OUT / f"{DATE}-siraya85a_template_v17_tool2.3mf"
SLOT = 1                                      # filament slot of extruder 2 (0-based)
INFILLS = (30, 50)
SEAT = 2.7                                    # provisional until the owner's coupon result
CHOSEN_SEAT = 2.9                             # owner, 2026-10-18: "the 2.9mm holes are what we should use"
CHOSEN_FIXES = {"tcd_ell7.4x8.5x11x22": "magnets 7.5 mm deep (1.76 mm to the surface at 7.0 mm)",
                "five_z7p4": "corners 7.4 mm apart along the finger (1.71 mm web at 7.0 mm)"}   # at 2.9 mm, the 1.8 mm rule
DEPTHS = {"block": 5.0, "tcd_cap8.5": 5.0, "tcd_ell7.4x8.5x16x32": 5.5, "tcd_ell7.4x8.5x11x22": 7.0}   # the 95A prints'
PRODUCTS = [("block", "four", 2), ("block", "five", 2), ("block", "four", 1),
            ("tcd_cap8.5", "four", 2), ("tcd_ell7.4x8.5x16x32", "four", 2), ("tcd_ell7.4x8.5x11x22", "four", 2)]
# Siraya's print settings taken over (speeds, accelerations, skirt, travel): 85A needs the slower moves
PRINT_KEYS = ("perimeter_speed", "external_perimeter_speed", "small_perimeter_speed", "infill_speed", "solid_infill_speed",
              "top_solid_infill_speed", "bridge_speed", "over_bridge_speed", "gap_fill_speed", "first_layer_speed",
              "first_layer_infill_speed", "overhang_speed_0", "overhang_speed_1", "overhang_speed_2", "overhang_speed_3",
              "enable_dynamic_overhang_speeds", "default_acceleration", "perimeter_acceleration",
              "external_perimeter_acceleration", "infill_acceleration", "solid_infill_acceleration",
              "top_solid_infill_acceleration", "bridge_acceleration", "first_layer_acceleration", "avoid_crossing_perimeters",
              "infill_overlap", "skirts", "skirt_height", "skirt_distance", "travel_speed", "travel_speed_z")
KEPT = {"perimeters": "two perimeters: the recipe of the finite elements and of the 1.8 mm pocket-wall rule (Siraya 3)",
        "top_solid_layers": "four 0.2 mm skins of the finite elements (Siraya 5)",
        "fill_density": "the design variable: 30 % or 50 % gyroid per project (Siraya 25 %)",
        "thick_bridges": "off: a thick bridge's 0.4 mm strand hangs 0.2 mm below a 0.2 mm layer, onto the magnet (Siraya on)",
        "perimeter_generator": "Arachne, as in the 95A prints whose loop counts the one-wall check reads (Siraya classic)",
        "printer_settings_id": "Original Prusa XL - 5T Input Shaper 0.4 nozzle: the owner's printer (Siraya's 3MF: single-tool XL)",
        "retract_length": "printer-side retraction of the 5T profile (the filament's 0.6 mm retraction overrides it on tool 2)"}


def kv_lines(text):
    out = {}
    for line in text.splitlines():
        if line.startswith("; ") and " = " in line:
            k, v = line[2:].split(" = ", 1)
            out[k] = v
    return out


def siraya_config():
    z = zipfile.ZipFile(SIRAYA_ZIP)
    name = next(n for n in z.namelist() if n.endswith(".3mf"))
    inner = zipfile.ZipFile(io.BytesIO(z.read(name)))
    return name, kv_lines(inner.read("Metadata/Slic3r_PE.config").decode())


def split_vector(val):
    """A PrusaSlicer vector option: string vectors are ';'-separated quoted strings (quotes and escapes respected), numeric
    vectors ','-separated."""
    if val.startswith('"'):
        parts, cur, q, esc = [], [], False, False
        for ch in val:
            if esc:
                cur.append(ch)
                esc = False
            elif ch == "\\":
                cur.append(ch)
                esc = True
            elif ch == '"':
                cur.append(ch)
                q = not q
            elif ch == ";" and not q:
                parts.append("".join(cur))
                cur = []
            else:
                cur.append(ch)
        parts.append("".join(cur))
        return parts, ";"
    if ";" in val:
        return val.split(";"), ";"
    return val.split(","), ","


def filament_keys():
    """The option names of a PrusaSlicer filament preset: the keys of the owner's saved Siraya filament preset."""
    keys = [l.split(" = ", 1)[0] for l in SIRAYA_FILAMENT_INI.read_text().splitlines() if " = " in l and not l.startswith("#")]
    return [k for k in keys if k not in ("inherits", "compatible_printers", "compatible_prints",
                                         "compatible_printers_condition", "compatible_prints_condition")]


def cmd_template(a=None):
    """The v17 template's config with Siraya's filament in slot 2 and Siraya's print speeds, as a config-only 3MF."""
    tz, tcfg, tkv = EB.template_config(EB.TEMPLATE)
    sname, skv = siraya_config()
    changes, lines = [], []
    fkeys = set(filament_keys())
    for line in tcfg.splitlines():
        if not (line.startswith("; ") and " = " in line):
            lines.append(line)
            continue
        k, val = line[2:].split(" = ", 1)
        new = val
        if k in fkeys and k in skv:
            parts, sep = split_vector(val)
            if len(parts) == 5:
                sv = skv[k]
                if parts[SLOT] != sv and {parts[SLOT], sv} != {"", '""'}:
                    changes.append(dict(key=k, kind="filament slot 2", template=parts[SLOT], siraya=sv))
                    parts[SLOT] = sv
                    new = sep.join(parts)
        elif k in PRINT_KEYS and k in skv and skv[k] != val:
            changes.append(dict(key=k, kind="print", template=val, siraya=skv[k]))
            new = skv[k]
        elif k == "compatible_printers_condition_cummulative":
            parts, sep = split_vector(val)
            s_parts, _ = split_vector(skv[k])
            if len(parts) >= 3 and len(s_parts) >= 2 and parts[1 + 1] != s_parts[1]:
                parts[1 + 1] = s_parts[1]                     # [print, filament 1, filament 2, ...]
                new = sep.join(parts)
        lines.append(f"; {k} = {new}")
    cfg = "\n".join(lines) + ("\n" if tcfg.endswith("\n") else "")
    OUT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(TEMPLATE, "w", zipfile.ZIP_DEFLATED) as zout:
        zout.writestr("[Content_Types].xml", tz.read("[Content_Types].xml"))
        zout.writestr("Metadata/Slic3r_PE.config", cfg)
    kept = []
    for k, why in KEPT.items():
        kept.append(dict(key=k, template=tkv.get(k), siraya=skv.get(k), why=why))
    rec = dict(template=MP._rel(TEMPLATE), source_template=str(EB.TEMPLATE).replace(str(Path.home()), "~"),
               siraya_profile=dict(zip=str(SIRAYA_ZIP).replace(str(Path.home()), "~"), project=sname,
                                   filament=skv.get("filament_settings_id"), print=skv.get("print_settings_id"),
                                   printer=skv.get("printer_settings_id")),
               filament_keys_from=str(SIRAYA_FILAMENT_INI).replace(str(Path.home()), "~"), slot=SLOT + 1,
               changed=changes, kept=kept, script="scripts/sim2real_fingertip_prints.py template",
               when=time.strftime("%Y-%m-%d %H:%M"))
    (OUT / "template_changes.json").write_text(json.dumps(rec, indent=1))
    print(f"{MP._rel(TEMPLATE)}: {len(changes)} keys from Siraya ({sum(c['kind'] == 'print' for c in changes)} print), "
          f"{len(kept)} kept", flush=True)
    return rec


def use_template():
    if not TEMPLATE.exists():
        cmd_template()
    EB.TEMPLATE = TEMPLATE


# ------------------------------------------------------------------------------------------------- G-code material check

def check_material(gc):
    """Tool selects, nozzle and bed setpoints, the slot-2 values PrusaSlicer wrote into the G-code's configuration block,
    the fan, and the volumetric flow of every extruding move (G1 and the template's G2/G3 arcs) on the part."""
    tools, noz, bed, fans = [], {}, set(), set()
    cur_t, F, x, y = None, None, 0.0, 0.0
    rates, typ = [], ""
    now, during, part_started, part_tool, first_setpoint, pending = {}, set(), False, None, None, []
    area = math.pi * (1.75 / 2) ** 2
    rx = re.compile(r"([XYZEFIJST])(-?\d*\.?\d+)")
    footer = {}
    for line in open(gc):
        if line.startswith(";"):
            if line.startswith(";TYPE:"):
                typ = line[6:].strip()
            elif line.startswith("; ") and " = " in line:
                k, val = line[2:].rstrip("\n").split(" = ", 1)
                footer[k] = val
            continue
        s = line.split(";", 1)[0].strip()
        if not s:
            continue
        mt = re.match(r"T(\d)(\s|$)", s)
        if mt:
            cur_t = int(mt.group(1))
            tools.append(cur_t)
            continue
        cmd = s.split()[0]
        f = dict((k, float(v)) for k, v in rx.findall(s[len(cmd):]))
        if cmd in ("M104", "M109"):
            t = int(f["T"]) if "T" in f else cur_t
            if "S" in f:
                noz.setdefault(t, set()).add(f["S"])
                now[t] = f["S"]
                if part_started and part_tool is not None and t == part_tool:
                    pending.append(f["S"])
        elif cmd in ("M140", "M190") and "S" in f:
            bed.add(f["S"])
        elif cmd == "M106" and "S" in f:
            fans.add(round(f["S"] / 2.55))
        elif cmd in ("G0", "G1", "G2", "G3"):
            if "F" in f:
                F = f["F"]
            nx, ny = f.get("X", x), f.get("Y", y)
            e = f.get("E", 0.0)
            if cmd in ("G2", "G3"):
                cx, cy = x + f.get("I", 0.0), y + f.get("J", 0.0)
                r = math.hypot(x - cx, y - cy)
                a0, a1 = math.atan2(y - cy, x - cx), math.atan2(ny - cy, nx - cx)
                da = (a0 - a1) % (2 * math.pi) if cmd == "G2" else (a1 - a0) % (2 * math.pi)
                L = r * (da if da > 1e-9 else 2 * math.pi)
            else:
                L = math.hypot(nx - x, ny - y)
            if e > 0 and L > 1e-3 and F and typ not in ("Custom", "Wipe tower", "Skirt/Brim", "Skirt"):
                rates.append(e * area / (L / (F / 60.0)))
                during.update(pending)                       # setpoints changed between two moves on the part
                pending.clear()
                if not part_started:
                    part_started, part_tool = True, cur_t
                    first_setpoint = now.get(cur_t)
                    during.add(first_setpoint)
            x, y = nx, ny

    def slot(k):
        if k not in footer:
            return None
        parts, _ = split_vector(footer[k])
        return parts[SLOT] if len(parts) > SLOT else parts[0]
    r = np.asarray(rates) if rates else np.zeros(1)
    return dict(tools_selected=sorted(set(tools)), tool_changes=len(tools), part_tool=part_tool,
                part_nozzle_setpoint_C=first_setpoint, part_nozzle_setpoints_C=sorted(x for x in during if x is not None),
                nozzle_setpoints_C={f"T{t}": sorted(v) for t, v in sorted((k, v) for k, v in noz.items() if k is not None)},
                bed_setpoints_C=sorted(bed), fan_percent=sorted(fans),
                slot2=dict(filament=slot("filament_settings_id"), temperature=slot("temperature"),
                           first_layer_temperature=slot("first_layer_temperature"),
                           extrusion_multiplier=slot("extrusion_multiplier"),
                           max_volumetric_speed=slot("filament_max_volumetric_speed"),
                           retract_length=slot("filament_retract_length"), bed=slot("bed_temperature")),
                perimeters=footer.get("perimeters"), fill_density=footer.get("fill_density"),
                thick_bridges=footer.get("thick_bridges"), perimeter_speed=footer.get("perimeter_speed"),
                volumetric_flow_mm3_s=dict(max=round(float(r.max()), 2), p99=round(float(np.percentile(r, 99)), 2),
                                           median=round(float(np.median(r)), 2), moves=len(rates)))


# ------------------------------------------------------------------------------------------------- build

def name_of(tip, lay, seat, depth, walls, infill):
    base = MP.name_of(tip, lay, seat, depth, walls if walls == 1 else None)          # 20261016-<tip>_<lay>_d3x2_...
    return base.replace(f"{DATE}-", f"{DATE}-siraya85a_", 1) + f"_g{infill}"


def recipe(infill, layer):
    return dict(material="Siraya Tech Flex TPU 85A", slicer_filament="Siraya Tech TPU 85A @XL 0.4mm Nozzle (tool 2)",
                perimeters=2, line_width_mm=0.45, fill_pattern="gyroid", fill_density=f"{infill}%", top_solid_layers=4,
                bottom_solid_layers=4, supports="none", layer_height_mm=layer, first_layer_mm=0.2, thick_bridges=0)


def write_and_check(part, pockets, gaps, mesh, name, tip, lay, depth, walls, infill, body, centres, a):
    upright = tip != "block"
    layer = 0.1 if upright else 0.2
    prow = MP.pauses(pockets, tip)
    settings = dict(perimeters=2, fill_density=f"{infill}%", fill_pattern="gyroid", top_solid_layers=4, bottom_solid_layers=4)
    proj = OUT / f"{name}.3mf"
    mods, mod_info = [], None
    if walls == 1:
        mod, mod_layers = MP.onewall_modifier(body, pockets, upright)
        mods = [(mod, dict(perimeters=1))]
        mod_stl = OUT / f"{name}_modifier.stl"
        mod.export(mod_stl)
        mod_info = dict(stl=MP._rel(mod_stl), perimeters=1, inset_mm=round(MP.inset(upright), 4), layers=len(mod_layers),
                        z_mm=[mod_layers[0][0], mod_layers[-1][1]], volume_mm3=round(float(mod.volume), 1))
    MP.write_project(proj, part, name, settings, layer, prow, mods)
    rep = dict(name=name, tip=tip, tip_label=MP.LABEL.get(tip, tip), layout=lay, infill_percent=infill,
               project=MP._rel(proj), recipe=recipe(infill, layer), template=MP._rel(TEMPLATE),
               orientation=("upright, mount face on the bed" if upright else "back face on the bed, palmar face up"),
               extruder=2, magnet=dict(source=MP.MAGNET_SRC, diameter_mm=MP.MAGNET[0], thickness_mm=MP.MAGNET[1],
                                       count=len(pockets), orientation="north pole toward the palmar face",
                                       top_depth_mm=depth),
               seat_mm=a.seat, seat_status="provisional: the owner's 85A coupon result sets the production seat",
               pockets=pockets, pocket_gaps=gaps, wall_min_mm=min(p["wall_min_mm"] for p in pockets),
               walls_ok=MP.walls_pass(pockets, gaps), pauses=prow, mesh=mesh, pocket_walls=walls,
               modifier=mod_info, chip_field=MP.chip_field(centres, depth),
               script="scripts/sim2real_fingertip_prints.py", when=time.strftime("%Y-%m-%d %H:%M"))
    if not a.no_slice:
        gc = MP.slice_project(proj, WORK / name)
        rep["slice"] = dict(MP.audit(proj, gc), check=MP.check_gcode(gc, pockets), material=check_material(gc))
        zl = sorted({round(0.5 * (p["floor_mm"] + p["top_mm"]) if p["kind"] == "pocket" else p["centre_print_mm"][1] + 0.5, 1)
                     for p in pockets})
        rep["slice"]["loops"] = MP.count_loops(gc, pockets, zl)
        rep["gcode"] = MP._rel(gc)
    (OUT / f"{name}.json").write_text(json.dumps(rep, indent=1, default=float))
    sl = rep.get("slice", {})
    ck, mt = sl.get("check", {}), sl.get("material", {})
    print(f"{name}: walls {rep['wall_min_mm']} ok {rep['walls_ok']}; tools {mt.get('tools_selected')} part on T"
          f"{mt.get('part_tool')} at {mt.get('part_nozzle_setpoints_C')} C, bed {mt.get('bed_setpoints_C')} EM {mt.get('slot2', {}).get('extrusion_multiplier')} "
          f"flow max {mt.get('volumetric_flow_mm3_s', {}).get('max')}; pauses {ck.get('pauses_found_z')} time "
          f"{ck.get('print_time')} islands {sl.get('islands_max')} supported {sl.get('supported_area')}; pockets "
          + "; ".join(f"void {q.get('void_layers')} in {q.get('beads_in_void_max')} floor {(q.get('floor') or {}).get('covered')} "
                      f"bridge {(q.get('bridge') or {}).get('covered')} pause {q.get('pause_before_bridge')}"
                      for q in ck.get("pockets", [])), flush=True)
    return rep


def cmd_build(a):
    use_template()
    OUT.mkdir(parents=True, exist_ok=True)
    over = dict(kv.split("=", 1) for kv in a.depth)          # --depth TIP=MM
    for tip, lay, walls in PRODUCTS:
        if a.tips and tip not in a.tips:
            continue
        if lay == "five" and a.five:                          # --five five_z7p4: corners further apart along the finger
            lay = a.five
        t0 = time.time()
        depth = float(over.get(tip, DEPTHS[tip]))
        centres = MP.layout(lay)
        body, part, pockets, gaps, mesh = MP.build(tip, centres, a.seat, depth)
        if not MP.walls_pass(pockets, gaps):
            msg = f"{tip} {lay}: pocket walls fail the {MP.WALL_MIN} mm rule at depth {depth}"
            if a.skip_failing:
                print(msg + "; skipped", flush=True)
                continue
            raise SystemExit(msg)
        stl = OUT / f"{MP.name_of(tip, lay, a.seat, depth).replace(f'{DATE}-', f'{DATE}-siraya85a_', 1)}.stl"
        part.export(stl)
        for infill in a.infill:
            name = name_of(tip, lay, a.seat, depth, walls, infill)
            if (OUT / f"{name}.json").exists() and not a.force:
                print(f"{name}: done", flush=True)
                continue
            pk = json.loads(json.dumps(pockets, default=float))
            rep = write_and_check(part, pk, gaps, mesh, name, tip, lay, depth, walls, infill, body, centres, a)
            rep["stl"] = MP._rel(stl)
            rep["seconds"] = round(time.time() - t0, 1)
            (OUT / f"{name}.json").write_text(json.dumps(rep, indent=1, default=float))


def cmd_coupon(a):
    use_template()
    OUT.mkdir(parents=True, exist_ok=True)
    part, pockets, slab = MP.coupon_part()
    base = f"{DATE}-siraya85a_seat_coupon_d3x2_seats2p7_2p8_2p9"
    stl = OUT / f"{base}.stl"
    part.export(stl)
    for walls in (2, 1):
        name = base + ("_pocketwalls1" if walls == 1 else "") + f"_g{a.coupon_infill}"
        settings = dict(perimeters=2, fill_density=f"{a.coupon_infill}%", fill_pattern="gyroid", top_solid_layers=4,
                        bottom_solid_layers=4)
        proj = OUT / f"{name}.3mf"
        mods, mod_info = [], None
        if walls == 1:
            mod, mod_layers = MP.onewall_modifier(slab, pockets, False)
            mods = [(mod, dict(perimeters=1))]
            mod_stl = OUT / f"{name}_modifier.stl"
            mod.export(mod_stl)
            mod_info = dict(stl=MP._rel(mod_stl), perimeters=1, inset_mm=round(MP.inset(False), 4), layers=len(mod_layers))
        MP.write_project(proj, part, name, settings, 0.2, [], mods)
        pk = json.loads(json.dumps(pockets, default=float))
        rep = dict(name=name, stl=MP._rel(stl), project=MP._rel(proj), recipe=recipe(a.coupon_infill, 0.2),
                   template=MP._rel(TEMPLATE), orientation="flat, 22 x 16 mm on the bed",
                   size_mm=[MP.COUPON["length"], MP.COUPON["width"], MP.COUPON["height"]], pockets=pk,
                   mesh=dict(watertight=bool(part.is_watertight), volume_mm3=round(float(part.volume), 1)),
                   pocket_walls=walls, modifier=mod_info,
                   use=("Press a magnet into each seat with a flat, non-magnetic tool. The seat to use is the narrowest that "
                        "takes the magnet to its floor without splitting the wall and keeps it when the coupon is turned "
                        "over and tapped. Round pockets: the block's seat (magnet flat). Slots: the curved tips' seat "
                        "(magnet on its edge)."), script="scripts/sim2real_fingertip_prints.py coupon",
                   when=time.strftime("%Y-%m-%d %H:%M"))
        if not a.no_slice:
            gc = MP.slice_project(proj, WORK / name)
            rep["slice"] = dict(MP.audit(proj, gc), check=MP.check_gcode(gc, pk), material=check_material(gc),
                                loops=MP.count_loops(gc, pk, [MP.COUPON["height"] - 1.0]))
            rep["gcode"] = MP._rel(gc)
        (OUT / f"{name}.json").write_text(json.dumps(rep, indent=1, default=float))
        mt = rep.get("slice", {}).get("material", {})
        print(f"{name}: time {rep.get('slice', {}).get('check', {}).get('print_time')}, nozzle {mt.get('nozzle_setpoints_C')}, "
              f"bed {mt.get('bed_setpoints_C')}, loops {rep.get('slice', {}).get('loops')}", flush=True)


# ------------------------------------------------------------------------------------------------- plate

PLATE_DATE = "20261018"
PLATE_TIPS = ("block_four_d3x2_depth5_seat2p9", "tcd_cap8p5_four_d3x2_depth5_seat2p9",
              "tcd_ell7p4x8p5x16x32_four_d3x2_depth5p5_seat2p9", "tcd_ell7p4x8p5x11x22_four_d3x2_depth7p5_seat2p9")
SHORT = {"block": "block", "tcd_cap8.5": "capsule", "tcd_ell7.4x8.5x16x32": "ellipsoid 16/32",
         "tcd_ell7.4x8.5x11x22": "ellipsoid 11/22"}
SHORT_ID = {"block": "block", "tcd_cap8.5": "cap8p5", "tcd_ell7.4x8.5x16x32": "ell16x32", "tcd_ell7.4x8.5x11x22": "ell11x22"}
PLATE_KEYS = ("perimeters", "fill_density", "fill_pattern", "top_solid_layers", "bottom_solid_layers")


def _pause_position(r, p):
    zs = [r["pockets"][i]["z_mm"] for i in p["magnets"]]
    if len(p["magnets"]) == len(r["pockets"]):
        return ""
    return "proximal" if max(zs) < TD.Z_C else ("distal" if min(zs) > TD.Z_C else "middle")


def plate_pauses(rows):
    """One pause per distinct print_z over all tips, its message naming the magnets each tip takes there."""
    at = {}
    for i, r in enumerate(rows):
        for p in r["pauses"]:
            at.setdefault(round(p["print_z"], 3), []).append((i, p))
    out = []
    n = len(at)
    for k, (z, items) in enumerate(sorted(at.items()), 1):
        took = []
        for i, p in items:
            pos = _pause_position(rows[i], p)
            took.append(f"{rows[i]['label']} {len(p['magnets'])}" + (f" {pos}" if pos else ""))
        kinds = {rows[i]["tip"] == "block" for i, _ in items}
        how = ([] if False not in kinds else ["slot magnets on edge to the slot floor, north toward the curved palmar side"]) + \
              ([] if True not in kinds else ["pocket magnets flat to the floor, north up"])
        busy = {i for i, _ in items}
        idle = [r["label"] for j, r in enumerate(rows) if j not in busy]
        total = sum(len(p["magnets"]) for _, p in items)
        msg = (f"Pause {k}/{n}, {total} magnets: " + ", ".join(took) + "; " + "; ".join(how)
               + (f". None in {', '.join(idle)}" if idle else "") + ". Resume")
        out.append(dict(print_z=z, message=msg, magnets_total=total,
                        tips=[dict(object=i, label=rows[i]["label"], magnets=p["magnets"], position=_pause_position(rows[i], p),
                                   source_message=p["message"]) for i, p in items]))
    return out


def plate_layout(parts, cols, gap):
    """Translations (x, y) of each part on the bed: a grid of `cols` columns, row 0 at the front (low Y), each part's
    bounding box centred in its cell, the whole set centred on the bed centre BED_XY."""
    b = [np.asarray(p.bounds)[:, :2] for p in parts]
    size = [bb[1] - bb[0] for bb in b]
    nrow = (len(parts) + cols - 1) // cols
    w = [max(size[i][0] for i in range(len(parts)) if i % cols == c) for c in range(cols)]
    h = [max(size[i][1] for i in range(len(parts)) if i // cols == r) for r in range(nrow)]
    t = []
    for i, bb in enumerate(b):
        c, r = i % cols, i // cols
        cx = sum(w[:c]) + c * gap + w[c] / 2
        cy = sum(h[:r]) + r * gap + h[r] / 2
        t.append(np.array([cx, cy]) - bb.mean(axis=0))
    lo = np.min([bb[0] + ti for bb, ti in zip(b, t)], axis=0)
    hi = np.max([bb[1] + ti for bb, ti in zip(b, t)], axis=0)
    shift = np.asarray(MP.BED_XY) - (lo + hi) / 2
    return [np.round(ti + shift, 4) for ti in t], np.round(lo + shift, 3).tolist(), np.round(hi + shift, 3).tolist()


def write_plate(path, parts, rows, trans, pause_rows, global_set):
    """PrusaSlicer project on the template: one object per tip at its translation, the tip's layer height and recipe as
    object settings, the shared recipe in the print config, and the merged pauses. The tip projects keep their recipe on
    the volume; a PrusaSlicer GUI save of the 11/22 project on 2026-10-18 dropped those volume keys and kept the object's
    layer height, so the plate writes them on the object and, where all tips agree, in the print config."""
    from xml.sax.saxutils import quoteattr
    z, cfg, _ = EB.template_config(TEMPLATE)
    lines = []
    for line in cfg.splitlines():
        if line.startswith("; ") and " = " in line and line[2:].split(" = ", 1)[0] in global_set:
            k = line[2:].split(" = ", 1)[0]
            line = f"; {k} = {global_set[k]}"
        lines.append(line)
    cfg = "\n".join(lines) + "\n"
    objs, items, mobjs = [], [], []
    for k, (part, r, t) in enumerate(zip(parts, rows, trans), 1):
        V, F = np.asarray(part.vertices), np.asarray(part.faces)
        verts = "\n".join(f'     <vertex x="{a:.5f}" y="{b:.5f}" z="{c:.5f}"/>' for a, b, c in V)
        tris = "\n".join(f'     <triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in F)
        objs.append(f'  <object id="{k}" type="model">\n   <mesh>\n    <vertices>\n{verts}\n    </vertices>\n'
                    f'    <triangles>\n{tris}\n    </triangles>\n   </mesh>\n  </object>\n')
        items.append(f'  <item objectid="{k}" transform="1 0 0 0 1 0 0 0 1 {t[0]:.4f} {t[1]:.4f} 0" printable="1"/>\n')
        st = dict(extruder=2, layer_height=f"{r['recipe']['layer_height_mm']:g}",
                  **{key: r["object_settings"][key] for key in PLATE_KEYS})
        meta = "".join(f'  <metadata type="object" key="{a}" value="{b}"/>\n' for a, b in st.items())
        mobjs.append(f' <object id="{k}" instances_count="1">\n  <metadata type="object" key="name" value="{r["object_name"]}"/>\n'
                     f'{meta}  <volume firstid="0" lastid="{len(F) - 1}">\n'
                     f'   <metadata type="volume" key="name" value="{r["object_name"]}"/>\n'
                     f'   <metadata type="volume" key="volume_type" value="ModelPart"/>\n'
                     f'   <metadata type="volume" key="matrix" value="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"/>\n'
                     f'   <mesh edges_fixed="0" degenerate_facets="0" facets_removed="0" facets_reversed="0" backwards_edges="0"/>\n'
                     f'  </volume>\n </object>\n')
    model = (f'<?xml version="1.0" encoding="UTF-8"?>\n<model unit="millimeter" xml:lang="en-US" '
             f'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" xmlns:slic3rpe="http://schemas.slic3r.org/3mf/2017/06">\n'
             f' <metadata name="slic3rpe:Version3mf">1</metadata>\n <metadata name="Title">{path.stem}</metadata>\n'
             f' <metadata name="Application">scripts/sim2real_fingertip_prints.py plate</metadata>\n <resources>\n'
             + "".join(objs) + ' </resources>\n <build>\n' + "".join(items) + ' </build>\n</model>\n')
    mcfg = '<?xml version="1.0" encoding="UTF-8"?>\n<config>\n' + "".join(mobjs) + '</config>\n'
    codes = "".join(f'<code print_z="{p["print_z"]:g}" type="1" extruder="2" color="" extra={quoteattr(p["message"])} '
                    f'gcode="M601"/>\n' for p in pause_rows)
    pz = (f'<?xml version="1.0" encoding="utf-8"?>\n<custom_gcodes_per_print_z bed_idx="0">\n{codes}'
          f'<mode value="MultiAsSingle"/>\n</custom_gcodes_per_print_z>\n')
    tmp = path.with_name(path.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as out:
        out.writestr("[Content_Types].xml", z.read("[Content_Types].xml"))
        out.writestr("_rels/.rels", '<?xml version="1.0" encoding="UTF-8"?>\n<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
                     ' <Relationship Target="/3D/3dmodel.model" Id="rel-1" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>\n</Relationships>\n')
        out.writestr("3D/3dmodel.model", model)
        out.writestr("Metadata/Slic3r_PE.config", cfg)
        out.writestr("Metadata/Slic3r_PE_model.config", mcfg)
        out.writestr("Metadata/Prusa_Slicer_custom_gcode_per_print_z.xml", pz)
    tmp.replace(path)
    return cfg


def gcode_labels(gc):
    """M486 object labels of the G-code header: name -> id."""
    out, cur = {}, None
    for line in open(gc):
        if line.startswith("M486 S"):
            cur = int(line.split()[1][1:])
        elif line.startswith("M486 A") and cur is not None:
            out[line[6:].strip()] = cur
        elif line.startswith(";LAYER_CHANGE"):
            break
    return out


def check_plate_object(gc, label, offset, pockets, bounds):
    """MP.check_gcode on one object of the plate: its own M486 label, its bed offset, and layer heights from its own
    layers (the merged G-code's ;HEIGHT: lines mix the 0.1 and 0.2 mm objects). Also the object's extent in the G-code
    against its placed bounds."""
    real, bed = MP.parse_arcs, MP.BED_XY

    def parse(path, obj=0, **kw):
        L, T = real(path, label, **kw)
        prev = 0.0
        for Lr in L:
            Lr["h"] = round(Lr["z"] - prev, 4)
            prev = Lr["z"]
        return L, T
    MP.parse_arcs, MP.BED_XY = parse, (float(offset[0]), float(offset[1]))
    try:
        ck = MP.check_gcode(gc, pockets)
        L, _ = parse(gc)
    finally:
        MP.parse_arcs, MP.BED_XY = real, bed
    seg = np.vstack([Lr["seg"] for Lr in L])
    ext = [[float(min(seg[:, 0].min(), seg[:, 2].min())), float(min(seg[:, 1].min(), seg[:, 3].min()))],
           [float(max(seg[:, 0].max(), seg[:, 2].max())), float(max(seg[:, 1].max(), seg[:, 3].max()))]]
    dev = float(np.max(np.abs(np.asarray(ext) - np.asarray(bounds)[:, :2])))
    ck["extent_mm"] = np.round(ext, 3).tolist()
    ck["extent_dev_mm"] = round(dev, 3)
    ck["top_z_mm"] = L[-1]["z"]
    return ck


def plate_figure(parts, rows, trans, pause_rows, lo, hi, dst):
    """Top view of the bed: each tip's footprint, its label, the pauses that take its magnets; front of the printer down."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.spatial import ConvexHull
    plt.rcParams.update({"font.family": "serif", "font.size": 13})
    fig, ax = plt.subplots(figsize=(7.2, 6.4), dpi=150)
    for i, (p, r, t) in enumerate(zip(parts, rows, trans)):
        xy = np.asarray(p.vertices)[:, :2] + t
        hull = xy[ConvexHull(xy).vertices]
        ax.fill(hull[:, 0], hull[:, 1], color="#d9d9d9", ec="black", lw=1)
        cx = 0.5 * (hull[:, 0].min() + hull[:, 0].max())
        ps = [f"{k + 1} ({len(tp['magnets'])})" for k, pr in enumerate(pause_rows) for tp in pr["tips"] if tp["object"] == i]
        ax.text(cx, hull[:, 1].max() + 0.8, r["label"].capitalize(), ha="center", va="bottom", fontsize=13)
        ax.text(cx, hull[:, 1].min() - 0.8, f"{r['recipe']['layer_height_mm']:g} mm layers\npause{'s' if len(ps) > 1 else ''} " + ", ".join(ps),
                ha="center", va="top", fontsize=11)
    pad = 12
    ax.set_xlim(lo[0] - pad, hi[0] + pad)
    ax.set_ylim(lo[1] - pad - 6, hi[1] + pad)
    ax.annotate("", xy=(0.5 * (lo[0] + hi[0]), lo[1] - pad - 4), xytext=(0.5 * (lo[0] + hi[0]), lo[1] - pad + 2),
                arrowprops=dict(arrowstyle="->", lw=1))
    ax.text(0.5 * (lo[0] + hi[0]) + 1.5, lo[1] - pad - 1, "Front of the printer", fontsize=11, va="center")
    ax.set_aspect("equal")
    ax.set_xlabel("Bed X (mm)")
    ax.set_ylabel("Bed Y (mm)")
    ax.set_title("Plate of " + ", ".join(r["label"] for r in rows), fontsize=13)
    fig.tight_layout()
    fig.savefig(dst)
    plt.close(fig)


def cmd_plate(a):
    import trimesh
    use_template()
    rows = []
    for t in a.tips:
        r = json.loads((OUT / f"{DATE}-siraya85a_{t}_g{a.infill}.json").read_text())
        r["label"] = SHORT[r["tip"]] + ("" if r["layout"] == "four" else ", five magnets") + \
            ("" if r["pocket_walls"] == 2 else ", one wall")
        r["object_name"] = f"{SHORT_ID[r['tip']]}" + ("" if r["layout"] == "four" else f"_{r['layout']}") + \
            ("" if r["pocket_walls"] == 2 else "_pocketwalls1") + f"_g{r['infill_percent']}"
        rc = r["recipe"]
        r["object_settings"] = dict(perimeters=rc["perimeters"], fill_density=rc["fill_density"], fill_pattern=rc["fill_pattern"],
                                    top_solid_layers=rc["top_solid_layers"], bottom_solid_layers=rc["bottom_solid_layers"])
        if r["modifier"]:
            raise SystemExit(f"{r['name']}: one-wall modifiers are not carried onto the plate")
        rows.append(r)
    parts = [trimesh.load(ROOT / r["stl"], force="mesh") for r in rows]
    seats = {r["seat_mm"] for r in rows}
    name = (f"{PLATE_DATE}-siraya85a_plate_" + "_".join(r["object_name"].rsplit("_g", 1)[0] for r in rows)
            + "_seat" + "_".join(f"{s:g}".replace(".", "p") for s in sorted(seats))
            + "_g" + "_".join(sorted({str(r["infill_percent"]) for r in rows})))
    trans, lo, hi = plate_layout(parts, a.cols, a.gap)
    pause_rows = plate_pauses(rows)
    shared = {k: {str(r["object_settings"][k]) for r in rows} for k in PLATE_KEYS}
    global_set = {k: v.pop() for k, v in shared.items() if len(v) == 1}
    # PrusaSlicer refuses a wipe tower over objects of unequal layer heights; one tool prints every tip, so it has no role
    global_set["wipe_tower"] = "0"
    proj = OUT / f"{name}.3mf"
    cfg = write_plate(proj, parts, rows, trans, pause_rows, global_set)
    objects = []
    for i, (r, p, t) in enumerate(zip(rows, parts, trans)):
        bb = np.asarray(p.bounds).copy()
        bb[:, :2] += t
        objects.append(dict(object=i, object_name=r["object_name"], label=r["label"], tip=r["tip"], layout=r["layout"],
                            infill_percent=r["infill_percent"], layer_height_mm=r["recipe"]["layer_height_mm"],
                            seat_mm=r["seat_mm"], magnets=r["magnet"]["count"], magnet_top_depth_mm=r["magnet"]["top_depth_mm"],
                            source_sheet=MP._rel(OUT / f"{r['name']}.json"), source_project=r["project"], stl=r["stl"],
                            translation_mm=t.tolist(), bed_bounds_mm=np.round(bb, 3).tolist(),
                            tip_print_time=r["slice"]["check"]["print_time"], tip_filament_g=r["slice"]["check"]["filament_g"]))
    rep = dict(name=name, project=MP._rel(proj), template=MP._rel(TEMPLATE), extruder=2,
               layout=dict(cols=a.cols, gap_mm=a.gap, front_row=[o["label"] for o in objects[:a.cols]],
                           set_bounds_mm=[lo, hi], bed_centre_mm=list(MP.BED_XY)),
               print_config_set=global_set, objects=objects, pauses=pause_rows,
               magnets_total=sum(o["magnets"] for o in objects),
               script="scripts/sim2real_fingertip_prints.py plate", when=time.strftime("%Y-%m-%d %H:%M"))
    fig = OUT / f"{name}_bed.png"
    plate_figure(parts, rows, trans, pause_rows, lo, hi, fig)
    rep["figure"] = MP._rel(fig)
    if not a.no_slice:
        work = WORK / name
        work.mkdir(parents=True, exist_ok=True)
        kv = kv_lines(cfg)
        ini = work / "plate.ini"
        ini.write_text("\n".join(f"{k} = {v}" for k, v in kv.items() if k != "binary_gcode") + "\nbinary_gcode = 0\n")
        gc = work / f"{name}.gcode"
        res = subprocess.run(EB.PRUSA + ["--load", str(ini), "--export-gcode", "-o", str(gc), str(proj)],
                                capture_output=True, text=True)
        if res.returncode != 0 or not gc.exists():
            raise SystemExit(f"PrusaSlicer failed: {(res.stdout + res.stderr)[-800:]}")
        labels = gcode_labels(gc)
        per = []
        for o, r in zip(objects, rows):
            ck = check_plate_object(gc, labels[o["object_name"]], o["translation_mm"], r["pockets"], o["bed_bounds_mm"])
            want = {round(p["print_z"], 3) for p in r["pauses"]}
            for q, pk in zip(ck["pockets"], r["pockets"]):
                q["pause_z_expected"] = pk["pause_print_z"]
            per.append(dict(object=o["object"], label=o["label"], gcode_label=labels[o["object_name"]], **ck,
                            pauses_expected_z=sorted(want)))
        L, T, pz, t_print, grams, t_pause = MP.gcode_layers(gc)
        mat = check_material(gc)
        ok = all(q.get("void_layers", 0) > 0 and q["beads_in_void_max"] < 0.01 and q["pause_before_bridge"]
                 and abs(q["bridge"]["z"] - q["pause_z_expected"]) < 1e-6 and q["floor"]["covered"] > 0.99
                 for c in per for q in c["pockets"]) and all(c["extent_dev_mm"] < 1.0 for c in per)
        rep["slice"] = dict(gcode=MP._rel(gc), print_time=t_print, filament_g=grams, pauses_found_z=pz, pause_at_min=t_pause,
                            m601=len(pz), material=mat, objects=per, all_checks_pass=bool(ok and len(pz) == len(pause_rows)))
        for p, tmin in zip(pause_rows, t_pause):
            p["at_min"] = tmin
    (OUT / f"{name}.json").write_text(json.dumps(rep, indent=1, default=float))
    sl = rep.get("slice", {})
    print(f"{name}: {len(rows)} tips, {rep['magnets_total']} magnets, pauses "
          + "; ".join(f"{p['print_z']:g} mm ({p['magnets_total']}) at {p.get('at_min')} min" for p in pause_rows)
          + f"; time {sl.get('print_time')}, filament {sl.get('filament_g')} g, tools {sl.get('material', {}).get('tools_selected')}, "
            f"checks {'pass' if sl.get('all_checks_pass') else 'FAIL'}", flush=True)
    for c in sl.get("objects", []):
        print(f"  {c['label']} (M486 S{c['gcode_label']}): extent dev {c['extent_dev_mm']} mm, layers {c['layer_heights_mm']}, "
              + "; ".join(f"void {q.get('void_layers')} in {q.get('beads_in_void_max')} floor {(q.get('floor') or {}).get('covered')} "
                          f"bridge {(q.get('bridge') or {}).get('covered')} at {(q.get('bridge') or {}).get('z')} "
                          f"(pause {q.get('pause_before_bridge')})" for q in c["pockets"]),
              flush=True)
    return rep


# ------------------------------------------------------------------------------------------------- README

def sheets():
    out = []
    for f in sorted(OUT.glob(f"{DATE}-siraya85a_*.json")):
        r = json.loads(f.read_text())
        out.append(r)
    return out


def _set(vals):
    return ", ".join(f"{v:g}" if isinstance(v, (int, float)) else str(v) for v in sorted(set(vals)))


def _min(txt):
    return MP._minutes(txt) if txt else float("nan")


def cmd_readme(a):
    S = sheets()
    tips = [r for r in S if "tip" in r]
    cps = [r for r in S if "tip" not in r]
    tc = json.loads((OUT / "template_changes.json").read_text())
    sl = {c["key"]: c for c in tc["changed"]}

    def g(k):
        return sl[k]["siraya"] if k in sl else "unchanged"
    order = {(t, l, w): i for i, (t, l, w) in enumerate(PRODUCTS)}
    tips.sort(key=lambda r: (abs(r.get("seat_mm", SEAT) - CHOSEN_SEAT) > 1e-6,
                             order.get((r["tip"], r["layout"].split("_")[0], r["pocket_walls"]), len(order)),
                             r["infill_percent"]))
    L = [f"# Siraya Tech TPU 85A fingertip prints for 3 x 2 mm magnets ({DATE[:4]}-{DATE[4:6]}-{DATE[6:]})", "",
         "Every file here is an 85A print: PrusaSlicer projects on the owner's v17 template (Original Prusa XL 5T Input Shaper, "
         "0.4 mm nozzles) with Siraya Tech's Prusa XL 0.4 mm TPU 85A filament profile in slot 2 and the part on extruder 2. "
         "The 95A projects of the same geometry in docs/experiments/20261016-metric_magnet_prints/print/ use Generic FLEX at "
         "230 C and are not 85A prints. `scripts/sim2real_fingertip_prints.py` (template, build, coupon, readme) writes "
         "them; each project has a sheet (JSON) with the geometry and the G-code checks.", "",
         "## Seat test first", "",
         f"The owner's seat test of 2026-10-18 chose {CHOSEN_SEAT:g} mm ('the 2.9mm holes are what we should use'): print "
         f"the seat{str(CHOSEN_SEAT).replace('.', 'p')} projects, listed first in the table of tips. Two tips change at that seat "
         "to keep the 1.8 mm pocket-wall rule: " + "; ".join(f"{k}, {v}" for k, v in CHOSEN_FIXES.items()) + ". The 2.7 mm "
         "projects below them are the provisional set printed before the test. `scripts/sim2real_fingertip_prints.py build "
         "--seat 2.9 --depth tcd_ell7.4x8.5x11x22=7.5 --five five_z7p4` rewrites the 2.9 mm set.", "",
         "| Coupon | Walls at each seat | Print time | Project |", "|---|---|---|---|"]
    for r in cps:
        L.append(f"| Seats 2.7, 2.8, 2.9 mm, round and slot | {r['pocket_walls']} | {r['slice']['check']['print_time']} | "
                 f"{Path(r['project']).name} |")
    L += ["", "## Profile", "",
          f"Starting point: {tc['siraya_profile']['filament']} and {tc['siraya_profile']['print']} from "
          f"{tc['siraya_profile']['project']} (Siraya Tech, for the single-tool Original Prusa XL). Taken over in slot 2: "
          f"nozzle {g('temperature')} C (first layer {g('first_layer_temperature')} C), bed {g('bed_temperature')} C, "
          f"extrusion multiplier {g('extrusion_multiplier')}, maximum volumetric speed {g('filament_max_volumetric_speed')} "
          f"mm3/s, retraction {g('filament_retract_length')} mm, fan {g('min_fan_speed')} %. Taken over from Siraya's print "
          f"profile: perimeters at {g('perimeter_speed')} mm/s, external perimeters {g('external_perimeter_speed')} mm/s, "
          f"infill {g('infill_speed')} mm/s, top layers {g('top_solid_infill_speed')} mm/s, bridges {g('bridge_speed')} mm/s, "
          f"a {g('skirts')}-loop skirt. Kept from the template, with the reason in template_changes.json: the 5T printer and "
          "tool 2, two 0.45 mm perimeters and four top and four bottom layers (the finite-element recipe and the pocket-wall "
          "rule), thin bridges (a thick bridge's 0.4 mm strand would hang onto the magnet under the 0.1-0.2 mm air gap), and "
          "the Arachne perimeter generator. Infill is gyroid at 30 % or 50 %, the stiffness choice; the 2026-10-12 models "
          "put 85A at 30 % near half the pad modulus of 50 % (6.9 against 16.5 MPa with a 4.76 mm magnet), with a solid "
          "modulus of 12 MPa taken from NinjaFlex 85A, so Siraya's printed stiffness is still to be measured.", "",
          "## Tips", "",
          "| Print | Infill | Orientation and layers | Magnets, top under the apex | Least TPU around a pocket (mm) | Pauses | "
          "Print time | Project |", "|---|---|---|---|---|---|---|---|"]
    for r in tips:
        ori = "back face on the bed, 0.2 mm" if r["tip"] == "block" else "upright on the mount face, 0.1 mm"
        lab = r["tip_label"].capitalize() + (", five magnets" if r["layout"] == "five" else "") + \
            (", one wall at the pockets" if r["pocket_walls"] == 1 else "")
        ck = r["slice"]["check"]
        pz = "; ".join(f"{p['print_z']:g} mm ({len(p['magnets'])})" for p in r["pauses"])
        L.append(f"| {lab} | {r['infill_percent']} % | {ori} | {r['magnet']['count']}, {r['magnet']['top_depth_mm']:g} mm | "
                 f"{r['wall_min_mm']:.2f} | {pz} | {ck['print_time']} | {Path(r['project']).name} |")
    for f in sorted(OUT.glob(f"{PLATE_DATE}-siraya85a_plate_*.json")):
        P = json.loads(f.read_text())
        sp = P.get("slice", {})
        ob = P["objects"]
        rows_ = [ob[i:i + P["layout"]["cols"]] for i in range(0, len(ob), P["layout"]["cols"])]
        where = "; ".join(f"{'front' if k == 0 else ('back' if k == len(rows_) - 1 else 'middle')} row, left to right: "
                          + ", ".join(o["label"] for o in row) for k, row in enumerate(rows_))
        lay = "; ".join(f"{h:g} mm: " + ", ".join(o["label"] for o in ob if o["layer_height_mm"] == h)
                        for h in sorted({o["layer_height_mm"] for o in ob}, reverse=True))
        L += ["", "## Plate of the candidate tips", "",
              f"{Path(P['project']).name} prints " + ", ".join(o["label"] for o in ob) + f" ({ob[0]['infill_percent']} % gyroid, "
              f"{ob[0]['seat_mm']:g} mm seats) on one bed, each from its tip's STL with its tip's recipe and layer height ("
              + lay + f"); {where} (figure {Path(P['figure']).name}). The recipe is written on each "
              "object and, where the tips agree, in the print config: PrusaSlicer's GUI save of the 11/22 project on 2026-10-18 "
              "dropped the per-volume recipe of the tip projects and kept the object's layer height. The plate stops "
              f"{len(P['pauses'])} times for its {P['magnets_total']} magnets, where the tips' own projects stop "
              f"{sum(len(p['tips']) for p in P['pauses'])} times; the wipe tower is off, since PrusaSlicer refuses one over "
              "objects of unequal layer heights and one tool prints every tip. At each pause only the tips named in the printer's message take magnets. "
              + (f"Print time {sp['print_time']}, filament {sp['filament_g']:.1f} g. " if sp else "")
              + "The G-code labels each tip as an object (M486), so a failed tip can be cancelled at the printer without "
              "stopping the others.", "",
              "| Pause | Height (mm) | Time from start (min) | Magnets | Tips |", "|---|---|---|---|---|"]
        for k, p in enumerate(P["pauses"], 1):
            L.append(f"| {k} | {p['print_z']:g} | {p.get('at_min', '')} | {p['magnets_total']} | "
                     + ", ".join(f"{t['label']} {len(t['magnets'])}" + (f" {t['position']}" if t["position"] else "")
                                 for t in p["tips"]) + " |")
        if sp:
            ck = [q for c in sp["objects"] for q in c["pockets"]]
            L += ["", f"Checks of the plate's G-code: only T{_set(sp['material']['tools_selected'])} selected; "
                  f"{sp['m601']} pauses (M601) at " + ", ".join(f"{z:g}" for z in sp["pauses_found_z"]) + " mm; every pocket "
                  f"of every tip empty on the layers it spans (largest share under a bead {max(q['beads_in_void_max'] for q in ck):.3f}), "
                  f"the layer under it covering {min(q['floor']['covered'] for q in ck) * 100:.0f} % or more of it, and bridged right after the pause "
                  f"assigned to it ({min(q['bridge']['covered'] for q in ck) * 100:.0f}-{max(q['bridge']['covered'] for q in ck) * 100:.0f} %); "
                  f"each tip's extent in the G-code within {max(c['extent_dev_mm'] for c in sp['objects']):.2f} mm of its "
                  "place on the bed."]
    mats = [r["slice"]["material"] for r in tips + cps]
    br = [q["bridge"]["covered"] for r in tips for q in r["slice"]["check"]["pockets"]]
    fl = [q["floor"]["covered"] for r in tips for q in r["slice"]["check"]["pockets"]]
    vd = [q["beads_in_void_max"] for r in tips for q in r["slice"]["check"]["pockets"]]
    isl = [r["slice"]["islands_max"] for r in tips]
    flow = [m["volumetric_flow_mm3_s"]["max"] for m in mats]
    L += ["", "## Checks of the sliced G-code", "",
          f"PrusaSlicer 2.9.2 sliced every project (G-code in logs/20261016-sim2real_fingertip_readiness/print/, untracked). "
          f"In every file the only tool selected is T{_set(t for m in mats for t in m['tools_selected'])} (extruder 2), "
          f"the part prints at {_set(s for m in mats for s in m['part_nozzle_setpoints_C'])} C on that tool and the bed at "
          f"{_set(s for m in mats for s in m['bed_setpoints_C'] if s > 0)} C, and the configuration block lists Siraya's filament "
          f"in slot 2 with extrusion multiplier {_set(m['slot2']['extrusion_multiplier'] for m in mats)}. The largest "
          f"volumetric flow of any extruding move on a part is {max(flow):.2f} mm3/s against the profile's 12. Each pocket is "
          f"empty on every layer it spans (largest share of a pocket's section under a bead: {max(vd):.3f}), the layer under "
          f"it covers {'its whole footprint' if min(fl) >= 0.999 else f'{min(fl) * 100:.0f} % or more of its footprint'}, and the first layer over it is printed right "
          f"after its pause and covers {min(br) * 100:.0f}-{max(br) * 100:.0f} % of the footprint. Islands per layer: at most "
          f"{max(isl)}. At mid-pocket height every pocket has "
          f"{_set(n for r in tips + cps if r['pocket_walls'] == 2 for l in r['slice']['loops'] for n in l['loops_around_pockets'])} "
          f"perimeter loops in the two-wall prints and "
          f"{_set(n for r in tips + cps if r['pocket_walls'] == 1 for l in r['slice']['loops'] for n in l['loops_around_pockets'])} "
          f"in the one-wall prints, and the outer surface "
          f"{_set(l['loops_at_outer_surface'] for r in tips + cps for l in r['slice']['loops'])}. Filament on the part: "
          f"{min(r['slice']['check']['filament_g'] for r in tips):.1f}-{max(r['slice']['check']['filament_g'] for r in tips):.1f} g "
          "per tip (skirt included).", "",
          "## Pauses", "",
          "The printer stops (M601) before the first layer over a pocket. Press each magnet to its floor with a flat, "
          "non-magnetic tool, north pole toward the palmar face, and resume. The block takes all its magnets at one pause; the "
          "curved tips take the two proximal magnets at the first pause and the two distal ones at the second. Check each pole "
          "with a marked magnet: a flipped magnet reverses its field at the chips.", "",
          "## Not yet measured", "",
          "Siraya 85A's printed modulus and friction, the seat fit and retention in 85A, and the tips' indentation at 0.5, 1 "
          "and 3 N. The study page (docs/experiments/20261016-sim2real_fingertip_readiness/) gives the model values and the "
          "bench procedure."]
    (OUT / "README.md").write_text("\n".join(L) + "\n")
    print((OUT / "README.md").read_text())


def cmd_recheck(a):
    """Recompute the G-code material block of every sheet from its kept G-code."""
    for f in sorted(OUT.glob(f"{DATE}-siraya85a_*.json")):
        r = json.loads(f.read_text())
        if "gcode" not in r:
            continue
        r["slice"]["material"] = check_material(ROOT / r["gcode"])
        f.write_text(json.dumps(r, indent=1, default=float))
        m = r["slice"]["material"]
        print(f"{r['name']}: T{m['part_tool']} {m['part_nozzle_setpoints_C']} C, bed {m['bed_setpoints_C']}, EM "
              f"{m['slot2']['extrusion_multiplier']}, flow max {m['volumetric_flow_mm3_s']['max']}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("template")
    sub.add_parser("recheck")
    sub.add_parser("readme")
    b = sub.add_parser("build")
    b.add_argument("--tips", nargs="*", default=[])
    b.add_argument("--infill", nargs="+", type=int, default=list(INFILLS))
    b.add_argument("--seat", type=float, default=SEAT)
    b.add_argument("--no-slice", action="store_true")
    b.add_argument("--force", action="store_true")
    b.add_argument("--depth", action="append", default=[], metavar="TIP=MM", help="magnet depth of one tip")
    b.add_argument("--five", default=None, metavar="LAYOUT", help="layout of the five-magnet block, e.g. five_z7p4")
    b.add_argument("--skip-failing", action="store_true", help="skip a tip whose pocket walls fail instead of stopping")
    c = sub.add_parser("coupon")
    c.add_argument("--coupon-infill", type=int, default=30)
    c.add_argument("--no-slice", action="store_true")
    p = sub.add_parser("plate")
    p.add_argument("--tips", nargs="+", default=list(PLATE_TIPS), help="sheet stems without the date, prefix and infill")
    p.add_argument("--infill", type=int, default=30)
    p.add_argument("--cols", type=int, default=2)
    p.add_argument("--gap", type=float, default=12.0, help="clearance between neighbouring tips' bounding boxes, mm")
    p.add_argument("--no-slice", action="store_true")
    a = ap.parse_args()
    {"template": cmd_template, "build": cmd_build, "coupon": cmd_coupon, "readme": cmd_readme,
     "recheck": cmd_recheck, "plate": cmd_plate}[a.cmd](a)


if __name__ == "__main__":
    main()
