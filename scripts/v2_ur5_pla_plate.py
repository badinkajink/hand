#!/usr/bin/env python3
"""One Bambu Lab P1S PLA plate with the PLA parts of DIP-link v2 a and b and of the UR5e calibration mount.

Owner, 2026-10-19 18:40: "let's just put all the concept A/B + UR5 PLA mounting stuff ... onto one PLA plate on the p1s and
send it so i can go home". The owner slices Bambu projects again in Bambu Studio, which kept the global process's infill, so
every part's settings are written into the project as object settings (Metadata/model_settings.config), which Bambu Studio
keeps when it slices again: the DIP links at 0.1 mm with tree supports from the plate and a 5 mm brim (v2/*/print/*_pla_bambu
.json; 40 % gyroid in place of 15 % with 100 % modifiers over the shoulders), the clamp plates and the cover solid at 0.1 mm,
the UR5 parts at 0.15 mm as their XL projects (40 % gyroid, the standoff-fork solid in place of its 100 % modifier over the
fork). Bambu Studio's command line builds the plate, the object settings go into the project, and it slices it again; the
G-code is checked part by part (layer heights, supports and brim only where asked, the P1S's own start G-code).
Output: docs/experiments/20261019-v2_ur5_pla_plate/.

    unset VIRTUAL_ENV
    ~/.claude/bin/resguard.sh run --mem 5G --cpu 400 -- taskset -c 14-18 env NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=1 \\
        ~/miniconda3/envs/femtip/bin/python -u scripts/v2_ur5_pla_plate.py
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import dip_fit_print_plates as D  # noqa: E402
import dip_link_print_project as PP  # noqa: E402

OUT = REPO / "docs/experiments/20261019-v2_ur5_pla_plate"
WORK = REPO / "logs/20261019-v2_ur5_pla_plate"
PFX = "20261019-v2_ur5_pla_p1s"
V2 = REPO / "docs/experiments/20261018-dip_link_board_mount/v2"
UR5 = REPO / "docs/experiments/20261018-ur5_calibration_mount/cad"

# object settings, Bambu Studio keys
LINK = dict(layer_height="0.1", wall_loops="3", top_shell_layers="6", bottom_shell_layers="5", sparse_infill_density="40%",
            sparse_infill_pattern="gyroid", enable_support="1", support_type="tree(auto)", support_on_build_plate_only="1")
SOLID = dict(layer_height="0.1", wall_loops="4", sparse_infill_density="100%", sparse_infill_pattern="zig-zag",
             enable_support="0", ironing_type="top")
MOUNT = dict(layer_height="0.15", wall_loops="5", top_shell_layers="6", bottom_shell_layers="6", sparse_infill_density="40%",
             sparse_infill_pattern="gyroid", enable_support="0")
MOUNT_SOLID = dict(MOUNT, wall_loops="4", sparse_infill_density="100%", sparse_infill_pattern="zig-zag")
INDENTER = dict(MOUNT, wall_loops="4", top_shell_layers="5", bottom_shell_layers="5")
# the UR5 parts' print frames (scripts/ur5_calibration_mount.py PRINT_SPEC: rows = print axes in the mount frame M)
R_UR5 = {"adapter": [[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]], "standoff_fork": np.eye(3),
         "clamp_plate": [[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]], "indenter_sphere": np.eye(3), "indenter_rod": np.eye(3)}
PARTS = [  # ID, label, source, print rotation (None: the file is in print orientation), object settings
    ("U2", "UR5 standoff-fork", UR5 / "20261018-ur5_calibration_mount_standoff_fork.stl", R_UR5["standoff_fork"], MOUNT_SOLID),
    ("U1", "UR5 adapter disc", UR5 / "20261018-ur5_calibration_mount_adapter.stl", R_UR5["adapter"], MOUNT),
    ("U4", "UR5 sphere indenter", UR5 / "20261018-ur5_calibration_mount_indenter_sphere.stl", R_UR5["indenter_sphere"], INDENTER),
    ("U5", "UR5 rod indenter", UR5 / "20261018-ur5_calibration_mount_indenter_rod.stl", R_UR5["indenter_rod"], INDENTER),
    ("A1", "v2 a DIP link", V2 / "a/print/20261018-dip_link_v2a_dip_link_pla_bambu.stl", None, LINK),
    ("B1", "v2 b DIP link", V2 / "b/print/20261018-dip_link_v2b_dip_link_pla_bambu.stl", None, LINK),
    ("A2", "v2 a clamp plate", V2 / "a/print/20261018-dip_link_v2a_clamp_plate_pla_bambu.stl", None, SOLID),
    ("B2", "v2 b cover", V2 / "b/print/20261018-dip_link_v2b_cover_pla_bambu.stl", None, SOLID),
    ("U3", "UR5 clamp plate", UR5 / "20261018-ur5_calibration_mount_clamp_plate.stl", R_UR5["clamp_plate"],
     dict(MOUNT_SOLID, wall_loops="4")),
]
GLOBAL = dict(D.P1S_PROCESS, layer_height="0.1", sparse_infill_pattern="gyroid", sparse_infill_density="40%",
              brim_type="outer_only", brim_width="5", enable_support="0", support_type="tree(auto)",
              support_on_build_plate_only="1")      # the command line ignores a brim set per object: a 5 mm brim on all
FILAMENT_AREA = np.pi * 1.75 ** 2 / 4


def inject(src, dst, settings):
    """A copy of the project src with each object's settings written into Metadata/model_settings.config after its name."""
    with zipfile.ZipFile(src) as z:
        files = {n: z.read(n) for n in z.namelist()}
    ms = files["Metadata/model_settings.config"].decode()
    for name, st in settings.items():
        pat = re.compile(rf'(<object id="\d+">\s*\n(\s*)<metadata key="name" value="{re.escape(name)}"/>\n)')
        m = pat.search(ms)
        if not m:
            raise SystemExit(f"object {name} not in model_settings.config")
        ind = m.group(2)
        add = "".join(f'{ind}<metadata key="{k}" value="{v}"/>\n' for k, v in st.items())
        ms = ms[:m.end()] + add + ms[m.end():]
    files["Metadata/model_settings.config"] = ms.encode()
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        for n, b in files.items():
            z.writestr(n, b)


def per_object(gcode, objs, pad=1.0):
    """Each object's layer z values, features and support/brim filament (mm), an extrusion going to the object whose bed
    bounds (grown by pad; 6 mm for a brim) contain its end point."""
    out = {o["id"]: dict(z=set(), feats={}) for o in objs}
    x = y = z = None
    feat = None
    for ln in gcode.splitlines():
        if ln.startswith("; FEATURE:"):
            feat = ln.split(":", 1)[1].strip()
            continue
        if ln.startswith("; Z_HEIGHT:"):
            z = float(ln.split(":")[1])
            continue
        if not ln.startswith(("G1", "G2", "G3")):
            continue
        mx, my, me = (re.search(rf"{c}(-?[\d.]+)", ln) for c in "XYE")
        x = float(mx.group(1)) if mx else x
        y = float(my.group(1)) if my else y
        if not me or not (mx or my) or feat is None or x is None or float(me.group(1)) <= 0:
            continue                                      # a move that extrudes (a bare G1 E is a retraction's recovery)
        for o in objs:
            (x0, y0, _), (x1, y1, _) = o["bounds_mm"]
            g = (15.0 if o["settings"].get("enable_support") == "1" else 6.0) if feat == "Brim" else pad   # a brim rings
            #                                                          the tree supports' base, which reaches past the part
            if x0 - g <= x <= x1 + g and y0 - g <= y <= y1 + g:
                r = out[o["id"]]
                r["feats"][feat] = r["feats"].get(feat, 0.0) + float(me.group(1))
                if feat not in ("Brim", "Skirt", "Custom"):
                    r["z"].add(round(z, 3))
                break
    return out


def main():
    import trimesh
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(WORK, ignore_errors=True)
    placed = WORK / "placed"
    placed.mkdir(parents=True)
    paths = {}
    for kind, d in (("machine", D.bbl_preset("machine", D.P1S["machine"])),
                    ("process", D.bbl_preset("process", D.P1S["process"], GLOBAL)),
                    ("filament", D.bbl_preset("filament", D.P1S["filament"]))):
        paths[kind] = WORK / f"{kind}.json"
        paths[kind].write_text(json.dumps(d, indent=1))
    meshes = []
    for pid, label, src, R, st in PARTS:
        tm = trimesh.load(src, force="mesh")
        tm, _ = PP.to_print(tm, R)
        meshes.append(tm)
    pos = D.layout(meshes, D.P1S_BED, gap=14.0, width=236.0)
    files, objs = [], []
    for (pid, label, src, R, st), tm, p_ in zip(PARTS, meshes, pos):
        tm.apply_translation((p_[0], p_[1], 0.0))
        f = placed / f"{pid}_{re.sub(r'[^a-z0-9]+', '_', label.lower()).strip('_')}.stl"
        tm.export(f)
        files.append(str(f))
        objs.append(dict(id=pid, label=label, name=f.name, source=D.rel(src), settings=st,
                         bounds_mm=np.round(tm.bounds, 2).tolist(), volume_mm3=round(float(tm.volume), 1)))
    base = [*D.BAMBU, "--debug", "1", "--load-settings", f"{paths['machine']};{paths['process']}",
            "--load-filaments", str(paths["filament"])]
    r = subprocess.run(base + ["--arrange", "0", "--orient", "0", "--export-3mf", "built.3mf", "--outputdir", str(WORK)]
                       + files, capture_output=True, text=True)
    (WORK / "cli_build.log").write_text(r.stdout + r.stderr)
    if not (WORK / "built.3mf").exists():
        raise SystemExit(f"Bambu Studio did not write the plate (log {D.rel(WORK / 'cli_build.log')})")
    inject(WORK / "built.3mf", WORK / "with_settings.3mf", {o["name"]: o["settings"] for o in objs})
    proj = f"{PFX}.3mf"
    r = subprocess.run(base + ["--slice", "0", "--export-3mf", proj, "--outputdir", str(WORK), str(WORK / "with_settings.3mf")],
                       capture_output=True, text=True)
    (WORK / "cli_slice.log").write_text(r.stdout + r.stderr)
    res = json.loads((WORK / "result.json").read_text())
    if res.get("return_code") != 0:
        raise SystemExit(f"Bambu Studio: {res.get('error_string')} (log {D.rel(WORK / 'cli_slice.log')})")
    shutil.copy(WORK / proj, OUT / proj)
    with zipfile.ZipFile(OUT / proj) as z:
        info = z.read("Metadata/slice_info.config").decode()
        gcode = z.read("Metadata/plate_1.gcode").decode(errors="replace")
        msets = z.read("Metadata/model_settings.config").decode()
    pred = float(re.search(r'key="prediction" value="([\d.]+)"', info).group(1))
    weight = float(re.search(r'key="weight" value="([\d.]+)"', info).group(1))
    layers = int(re.search(r"; total layer number: (\d+)", gcode).group(1))
    problems = D.check_bambu_gcode(gcode, "P1S", 220, 55)
    if msets.count('<metadata key="layer_height"') < len(objs):
        problems.append("object settings missing from the sliced project")
    po = per_object(gcode, objs)
    for o in objs:
        r_ = po[o["id"]]
        zs = sorted(r_["z"])
        steps = np.round(np.diff(zs), 3) if len(zs) > 1 else np.array([])
        lh = float(o["settings"]["layer_height"])
        o["layers"], o["top_z"] = len(zs), (zs[-1] if zs else None)
        o["layer_step_mm"] = float(np.median(steps)) if len(steps) else None
        o["support_mm"] = round(sum(v for k, v in r_["feats"].items() if k.startswith("Support")), 1)
        o["brim_mm"] = round(r_["feats"].get("Brim", 0.0), 1)
        o["sparse_infill_mm"] = round(r_["feats"].get("Sparse infill", 0.0), 1)
        o["solid_infill_mm"] = round(r_["feats"].get("Internal solid infill", 0.0), 1)
        part_e = sum(v for k, v in r_["feats"].items() if not k.startswith(("Support", "Brim", "Skirt", "Custom", "Ironing")))
        o["fill_ratio"] = round(part_e * FILAMENT_AREA / o["volume_mm3"], 3)      # extruded plastic over the part's volume
        if o["layer_step_mm"] is None or abs(o["layer_step_mm"] - lh) > 1e-3:
            problems.append(f"{o['id']} layer step {o['layer_step_mm']} mm, asked {lh}")
        if o["top_z"] is None or abs(o["top_z"] - o["bounds_mm"][1][2]) > lh + 0.05:
            problems.append(f"{o['id']} top {o['top_z']} mm, part {o['bounds_mm'][1][2]} mm")
        if (o["support_mm"] > 0) != (o["settings"]["enable_support"] == "1"):
            problems.append(f"{o['id']} support {o['support_mm']} mm, asked {o['settings']['enable_support']}")
        if o["settings"].get("enable_support") == "1" and o["brim_mm"] <= 0:
            problems.append(f"{o['id']} has no brim")
        if o["settings"]["sparse_infill_density"] == "100%" and o["fill_ratio"] < 0.85:
            problems.append(f"{o['id']} fill ratio {o['fill_ratio']} though asked solid")
    rec = dict(owner="2026-10-19 18:40: 'let's just put all the concept A/B + UR5 PLA mounting stuff ... onto one PLA plate on "
                     "the p1s and send it so i can go home'",
               project=D.rel(OUT / proj), printer="Bambu Lab P1S 0.4 nozzle", filament=D.P1S["filament"], slot="AMS slot 1",
               plate="Textured PEI Plate, 55 C", process_base=D.P1S["process"], global_settings=GLOBAL, time_s=pred,
               time=D.fmt_time(pred), filament_g=weight, layers=layers, objects=objs, problems=problems,
               slicer="Bambu Studio 2.5.0.66 command line: plate built, object settings written into "
                      "Metadata/model_settings.config, sliced again", when=time.strftime("%Y-%m-%d %H:%M"))
    fig = OUT / f"{PFX}_map.png"
    D.plate_map([(dict(id=o["id"], material="PLA", short=o["label"]), trimesh.load(placed / o["name"])) for o in objs],
                D.P1S_BED, f"Bambu Lab P1S, PLA (AMS slot 1): v2 a and b, UR5 mount, {D.fmt_time(pred)}, {weight:.0f} g", fig)
    rec["map"] = D.rel(fig)
    (OUT / f"{PFX}.json").write_text(json.dumps(rec, indent=1) + "\n")
    for o in objs:
        print(f"{o['id']:3s} {o['label']:22s} layers {o['layers']:4d} step {o['layer_step_mm']} top {o['top_z']} "
              f"(part {o['bounds_mm'][1][2]}) support {o['support_mm']} brim {o['brim_mm']} sparse {o['sparse_infill_mm']} "
              f"solid {o['solid_infill_mm']} fill {o['fill_ratio']}", flush=True)
    print(f"P1S PLA plate: {len(objs)} parts, {D.fmt_time(pred)}, {weight:.1f} g, {layers} layers [{time.time() - t0:.0f} s] -> "
          f"{D.rel(OUT / proj)}", flush=True)
    if problems:
        raise SystemExit("P1S PLA plate: " + "; ".join(problems))


if __name__ == "__main__":
    main()
