#!/usr/bin/env python3
"""Concept a's 11/22 tip and concept b's 11/22 pad (85A, five magnets) on one Prusa XL plate, tool 2.

Owner, 2026-10-19 18:40: "let's prep the A and B TPU fingers to go on the prusa slicer. the only print ive had good luck with
is .../20261018-siraya85a_plate_block_cap8p5_ell16x32_ell11x22_seat2p9_g30.3mf". The two v2 projects
(v2/a/print/*_print_tip_ell11_5_channels_85a_t2_g30.3mf, v2/b/print/*_print_pad_ell11_5_85a_t2_g30.3mf) are merged: both
objects with their object settings, 30 mm apart, and their common magnet pauses (7.3, 12.2, 17.1 mm); the global settings
take the good project's values where they differ (30 % gyroid, four top layers, no wipe tower). Sliced with PrusaSlicer and
checked: three M601 pauses, both objects, tool 2 only.
Output: docs/experiments/20261018-dip_link_board_mount/v2/print/20261019-dip_link_v2_tip_a_pad_b_ell11_5_85a_t2_g30.3mf.

    ~/.claude/bin/resguard.sh run --mem 3G --cpu 200 -- taskset -c 14,15 ~/miniconda3/envs/femtip/bin/python -u \\
        scripts/v2_tips_xl_plate.py
"""
from __future__ import annotations

import json
import re
import sys
import time
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import dip_link_print_project as PP  # noqa: E402

V2 = REPO / "docs/experiments/20261018-dip_link_board_mount/v2"
SRC = [("a", V2 / "a/print/20261018-dip_link_v2a_print_tip_ell11_5_channels_85a_t2_g30.3mf", -15.0),
       ("b", V2 / "b/print/20261018-dip_link_v2b_print_pad_ell11_5_85a_t2_g30.3mf", 15.0)]
GOOD = REPO / "docs/experiments/20261016-sim2real_fingertip_readiness/print/20261018-siraya85a_plate_block_cap8p5_ell16x32_ell11x22_seat2p9_g30.3mf"
OUT = V2 / "print/20261019-dip_link_v2_tip_a_pad_b_ell11_5_85a_t2_g30.3mf"
WORK = REPO / "logs/20261019-v2_tips_xl"


def read(p):
    with zipfile.ZipFile(p) as z:
        return {n: z.read(n) for n in z.namelist()}


def cfg_keys(text):
    return {m.group(1): m.group(2) for m in re.finditer(r"^; (\S+) = (.*)$", text, re.M)}


def main():
    t0 = time.time()
    fa, fb = read(SRC[0][1]), read(SRC[1][1])
    good = read(GOOD)
    # the model: b's object as object 2, each item moved along x
    ma, mb = fa["3D/3dmodel.model"].decode(), fb["3D/3dmodel.model"].decode()
    ob = re.search(r'<object id="1" type="model">.*?</object>', mb, re.S).group(0).replace('<object id="1"', '<object id="2"', 1)
    ma = ma.replace("</resources>", ob + "\n</resources>", 1)
    item = re.search(r'<item objectid="1" transform="([^"]+)"([^/]*)/>', ma)
    T = item.group(1).split()
    x0 = float(T[9])
    Ta, Tb = T[:], T[:]
    Ta[9], Tb[9] = f"{x0 + SRC[0][2]:.4f}", f"{x0 + SRC[1][2]:.4f}"
    ma = ma.replace(item.group(0), f'<item objectid="1" transform="{" ".join(Ta)}"{item.group(2)}/>\n'
                    f' <item objectid="2" transform="{" ".join(Tb)}"{item.group(2)}/>', 1)
    # the object settings: b's object block as object 2
    ca, cb = fa["Metadata/Slic3r_PE_model.config"].decode(), fb["Metadata/Slic3r_PE_model.config"].decode()
    bo = re.search(r' <object id="1".*?</object>\n', cb, re.S).group(0).replace('<object id="1"', '<object id="2"', 1)
    ca = ca.replace("</config>", bo + "</config>", 1)
    # the pauses: a's and b's are the same heights and steps
    pa, pb = (f["Metadata/Prusa_Slicer_custom_gcode_per_print_z.xml"].decode() for f in (fa, fb))
    za = re.findall(r'print_z="([\d.]+)"', pa)
    if za != re.findall(r'print_z="([\d.]+)"', pb):
        raise SystemExit(f"pause heights differ: {za} and {re.findall(r'print_z=.([0-9.]+)', pb)}")
    pz = re.sub(r'extra="', 'extra="Both, a&apos;s tip and b&apos;s pad: ', pa)
    # the global settings: the good project's values where a's differ
    ga, gg = fa["Metadata/Slic3r_PE.config"].decode(), good["Metadata/Slic3r_PE.config"].decode()
    ka, kg = cfg_keys(ga), cfg_keys(gg)
    changed = {}
    for k in sorted(set(ka) | set(kg)):
        if ka.get(k) != kg.get(k) and k in kg and k in ka:
            ga = re.sub(rf"^; {re.escape(k)} = .*$", lambda m: f"; {k} = {kg[k]}", ga, count=1, flags=re.M)
            changed[k] = [ka[k], kg[k]]
    files = dict(fa)
    files.update({"3D/3dmodel.model": ma.encode(), "Metadata/Slic3r_PE_model.config": ca.encode(),
                  "Metadata/Prusa_Slicer_custom_gcode_per_print_z.xml": pz.encode(), "Metadata/Slic3r_PE.config": ga.encode()})
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for n, b in files.items():
            z.writestr(n, b)
    # object settings beside the good project's
    og = {m.group(1): m.group(2) for m in re.finditer(r'key="(\w+)" value="([^"]*)"', good["Metadata/Slic3r_PE_model.config"].decode())}
    oa = {m.group(1): m.group(2) for m in re.finditer(r'key="(\w+)" value="([^"]*)"', ca)}
    obj_diff = {k: [og.get(k), oa.get(k)] for k in sorted(set(og) | set(oa))
                if og.get(k) != oa.get(k) and k not in ("name", "matrix", "source_file", "source_object_id",
                                                        "source_volume_id", "source_offset_x", "source_offset_y",
                                                        "source_offset_z", "volume_type")}
    gc, log = PP.slice_project(OUT, ga, WORK)
    summ = PP.gcode_summary(gc)
    text = Path(gc).read_text()
    pz_g = sorted({round(float(m), 2) for m in re.findall(r"^;Z:([\d.]+)\s*\n(?:.*\n){0,40}?M601", text, re.M)})
    temps = sorted({int(float(v)) for v in re.findall(r"^M10[49] (?:T\d )?S(\d+)", text, re.M) if float(v) > 0})
    beds = sorted({int(float(v)) for v in re.findall(r"^M1[49]0 S(\d+)", text, re.M)})
    problems = []
    if summ["m601"] != len(za):
        problems.append(f"{summ['m601']} M601 pauses, {len(za)} asked")
    if len(summ["labels"]) != 2:
        problems.append(f"objects in the G-code: {list(summ['labels'])}")
    if summ.get("tool_changes"):
        problems.append(f"{summ['tool_changes']} tool changes")
    rec = dict(owner="2026-10-19 18:40: 'let's prep the A and B TPU fingers to go on the prusa slicer'", project=PP.rel(OUT),
               sources=[PP.rel(p) for _, p, _ in SRC], good_project=PP.rel(GOOD), global_changed_to_good=changed,
               object_settings_vs_good=obj_diff, pauses_mm=[float(z) for z in za], gcode=PP.rel(gc),
               summary={k: v for k, v in summ.items() if k != "labels"}, objects=list(summ["labels"]), nozzle_c=temps,
               bed_c=beds, problems=problems, when=time.strftime("%Y-%m-%d %H:%M"))
    OUT.with_suffix(".json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({k: rec[k] for k in ("global_changed_to_good", "object_settings_vs_good", "pauses_mm", "summary",
                                          "objects", "nozzle_c", "bed_c", "problems")}, indent=1), flush=True)
    print(f"-> {PP.rel(OUT)} [{time.time() - t0:.0f} s]", flush=True)
    if problems:
        raise SystemExit("; ".join(problems))


if __name__ == "__main__":
    main()
