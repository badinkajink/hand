#!/usr/bin/env python3
"""Print project of a DIP link design for the Prusa XL 5T (hand-off docs/handoff/20261018-calibration_and_dip_finish.md,
step 18): the DIP link, its clamp plate with the Qwiic pocket and the pin-fit coupon in PLA on tool 3, and one 85A block tip
with the two key channels on tool 2, printed without magnets (no pause), so the dovetail fit is tried before magnets go into
tips (owner, 2026-10-18 16:36: "i'm hesitant to run further prints and waste magnets until we can print some distal
links"). Two PrusaSlicer projects on the owner's v17 template with Siraya 85A in slot 2
(docs/experiments/20261016-sim2real_fingertip_readiness/print/20261016-siraya85a_template_v17_tool2.3mf), each sliced with
PrusaSlicer 2.9 and checked: the PLA parts, and the tip alone. On one plate PrusaSlicer heated the bed to the first tool's
35 C (the tip's) for the first layer of the PLA parts and changed tools 169 times, each from the 70 C idle temperature.

Orientations and settings are those of final/20261018-dip_link_final_print_settings.json:
  DIP link     tip end up (print X = -x_F, Y = y_F, Z = -z_F): the bridge's mount face is a top surface, the window plate,
               keys and shoulders rise as vertical walls; 0.1 mm layers, 3 perimeters, 15 % gyroid, 100 % in the shoulders
               (a modifier volume); supports from three support enforcers only (support_material_auto = 0): under the
               horn-side arm, around the slotted arm's end on the bed, and under the bridge's face toward the PIP axis
  clamp plate  back face on the bed (X = z_F, Y = y_F, Z = 10.9 - x_F): the board face is a top surface with ironing, the
               Qwiic pocket and the nut pockets open on the bed side; 0.1 mm layers, 4 perimeters, 100 %
  coupon       as built (plate on the bed, pins up); 0.1 mm layers, 4 perimeters, 100 %
  tip          the block sheet's print frame (back face on the bed, the channels open to the bed); Siraya 85A, 0.2 mm layers,
               2 perimeters, 30 % gyroid, 4 top and bottom layers; no M601
Seams: rear, which is the +y_F side of the DIP link (the slotted arm's side face, away from the window plate's faces).

build_project() takes any parameter set's parts (the DIP link and the clamp plate as STL in frame F, the coupon and the tip
in their print frames), so the parametric viewer's exports slice the same way.

    RG="$HOME/.claude/bin/resguard.sh run --mem 2G --cpu 200 -- taskset -c 2-3 env NUMPY_MADVISE_HUGEPAGE=0"
    $RG ~/miniconda3/envs/femtip/bin/python scripts/dip_link_print_project.py                   # concept a's final
    $RG ~/miniconda3/envs/femtip/bin/python scripts/dip_link_print_project.py --dip-link L.stl --clamp-plate C.stl \\
        [--coupon K.stl] [--tip T.stl] --out DIR --name NAME [--no-slice]
Out per project: <out>/<name>.3mf, <name>.json (settings, placements, slice checks), <name>_bed.png; G-code under logs/
(untracked).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

STUDY = ROOT / "docs/experiments/20261018-dip_link_board_mount"
FINAL = STUDY / "final"
PFX = "20261018-dip_link_final"
TEMPLATE = ROOT / "docs/experiments/20261016-sim2real_fingertip_readiness/print/20261016-siraya85a_template_v17_tool2.3mf"
WORK = ROOT / "logs/20261018-dip_link/print"
PRUSA = ["flatpak", "run", "--command=prusa-slicer", "com.prusa3d.PrusaSlicer"]
BED_XY = (180.0, 180.0)                 # centre of the XL's 360 x 360 mm bed
GAP = 20.0                              # between parts on the bed, mm
PLA_TOOL, TPU_TOOL = 3, 2               # PrusaSlicer extruder numbers (1-based) in the v17 template
X_BACK = 6.45                           # the tip's back face, x_F
ARM_Y = (-17.05, 13.40)                 # the clevis arms lie at y_F <= -17.05 and y_F >= 13.40

# print frames: rows are the print axes in F coordinates
R_LINK = np.array([[-1.0, 0, 0], [0, 1.0, 0], [0, 0, -1.0]])        # X = -x_F, Y = y_F, Z = -z_F
R_CLAMP = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])        # X = z_F, Y = y_F, Z = -x_F

PLA_OBJECT = dict(layer_height="0.1", seam_position="rear")
SETTINGS = {
    "dip_link": dict(PLA_OBJECT, perimeters="3", fill_density="15%", fill_pattern="gyroid", top_solid_layers="6",
                     bottom_solid_layers="5"),
    "clamp_plate": dict(PLA_OBJECT, perimeters="4", fill_density="100%", fill_pattern="rectilinear", ironing="1",
                        ironing_type="top"),
    "pin_fit_coupon": dict(PLA_OBJECT, perimeters="4", fill_density="100%", fill_pattern="rectilinear"),
    "tip": dict(layer_height="0.2", perimeters="2", fill_density="30%", fill_pattern="gyroid", top_solid_layers="4",
                bottom_solid_layers="4"),
}
GLOBAL = dict(wipe_tower="0",               # PrusaSlicer refuses a wipe tower over objects of unequal layer heights
              support_material="1", support_material_auto="0",      # supports from the enforcers only
              first_layer_height="0.2", brim_width="0", binary_gcode="0")
LABEL = {"dip_link": "DIP link", "clamp_plate": "Clamp plate", "pin_fit_coupon": "Pin-fit coupon",
         "tip": "85A block tip with key channels, no magnets"}


def rel(p):
    p = Path(p)
    return str(p.relative_to(ROOT)) if p.is_absolute() and ROOT in p.parents else str(p)


# ------------------------------------------------------------------------------------------------ geometry
def to_print(tm, R=None):
    """The mesh turned into its print frame (R: rows = print axes in the mesh's coordinates) and moved so that its lowest
    point lies on the bed and its footprint is centred on the origin. Returns the mesh and (R, t) with p_print = R p + t."""
    import trimesh
    R = np.eye(3) if R is None else np.asarray(R, float)
    V = np.asarray(tm.vertices) @ R.T
    lo, hi = V.min(axis=0), V.max(axis=0)
    t = -np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
    out = trimesh.Trimesh(V + t, np.asarray(tm.faces), process=False)
    if out.volume < 0:
        out.invert()
    return out, (R, t)


def box_F_to_print(lo, hi, Rt):
    """An axis-aligned box in F as an axis-aligned box in the print frame (the print frames are signed permutations)."""
    R, t = Rt
    c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]) @ R.T + t
    return c.min(axis=0), c.max(axis=0)


def box_mesh(lo, hi):
    import trimesh
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return trimesh.creation.box(extents=hi - lo, transform=trimesh.transformations.translation_matrix((lo + hi) / 2))


def link_volumes(link_print, Rt):
    """Support enforcers and the shoulders' infill modifier of the DIP link, in its print frame (before placement).
    Enforcers: (1) under the horn-side arm, from the bed to 1.2 mm above its lowest point; (2) around the slotted arm's end
    on the bed, to 1.2 mm; (3) between the arms, the band of the bridge's face toward the PIP axis (z_F -61.4 to -66.4 mm),
    which faces the bed in this orientation. The arms' bores and horn holes, the chip windows and the keys lie outside
    them, so PrusaSlicer leaves them unsupported."""
    V = np.asarray(link_print.vertices)
    lo, hi = V.min(axis=0), V.max(axis=0)
    yb = lambda y: float(box_F_to_print((0, y, 0), (0, y, 0), Rt)[0][1])   # noqa: E731
    horn = V[V[:, 1] <= yb(ARM_Y[0]) + 1e-6]
    slot = V[V[:, 1] >= yb(ARM_Y[1]) - 1e-6]
    z_bridge = sorted(float(box_F_to_print((0, 0, z), (0, 0, z), Rt)[0][2]) for z in (-61.4, -66.4))
    enf = [("enforcer_horn_arm", (lo[0] - 1, lo[1] - 1, 0.0), (hi[0] + 1, yb(ARM_Y[0]) + 0.05, float(horn[:, 2].min()) + 1.2)),
           ("enforcer_slotted_arm", (lo[0] - 1, yb(ARM_Y[1]) - 0.05, 0.0), (hi[0] + 1, hi[1] + 1, 1.2)),
           ("enforcer_bridge_pip_face", (lo[0] - 1, yb(ARM_Y[0]) + 0.05, z_bridge[0]),
            (hi[0] + 1, yb(ARM_Y[1]) - 0.05, z_bridge[1]))]
    import dip_link_concept_a as A
    L, P = A.levels(), A.P
    mods = []
    for s in (-1, 1):
        y0, y1 = sorted((s * P["shoulder_y"][0], s * P["shoulder_y"][1]))
        blo, bhi = box_F_to_print((X_BACK - 1.0, y0, P["shoulder_z0"] - 0.1), (L["x_sh1"] + 0.2, y1, P["shoulder_cap"] + 0.1), Rt)
        mods.append((f"shoulder_{'plus' if s > 0 else 'minus'}_y_100pct", tuple(blo), tuple(bhi)))
    return enf, mods


def window_boxes(Rt):
    """The five chip windows of the window plate in the print frame (centres from the board model, 3.4 mm squares)."""
    import dip_link_concept_a as A
    L, P = A.levels(), A.P
    it = A.board_items()
    w = 3.0 + 2 * P["clear_chip"]
    out = {}
    for k, v in it["parts"].items():
        if v["kind"] != "MLX90393":
            continue
        out[k] = box_F_to_print((X_BACK, v["y"] - w / 2, v["z"] - w / 2), (L["x_cf"], v["y"] + w / 2, v["z"] + w / 2), Rt)
    return out


def layout(parts):
    """Translations (x, y) on the bed: the parts in one row along X, GAP apart, the row centred on BED_XY."""
    widths = [p.bounds[1][0] - p.bounds[0][0] for p in parts]
    total = sum(widths) + GAP * (len(parts) - 1)
    x = BED_XY[0] - total / 2
    out = []
    for p, w in zip(parts, widths):
        cx = x + w / 2 - (p.bounds[0][0] + p.bounds[1][0]) / 2
        out.append(np.array([cx, BED_XY[1] - (p.bounds[0][1] + p.bounds[1][1]) / 2]))
        x += w + GAP
    return out


# ------------------------------------------------------------------------------------------------ 3mf
def template_config():
    z = zipfile.ZipFile(TEMPLATE)
    cfg = z.read("Metadata/Slic3r_PE.config").decode()
    kv = {}
    for line in cfg.splitlines():
        if line.startswith("; ") and " = " in line:
            k, v = line[2:].split(" = ", 1)
            kv[k] = v
    return z, cfg, kv


def write_project(path, objs):
    """PrusaSlicer project on the template: one object per part at its bed translation with its extruder and settings as
    object metadata, and its extra volumes (support enforcers, parameter modifiers) appended to the object's mesh."""
    z, cfg, _ = template_config()
    lines = []
    for line in cfg.splitlines():
        if line.startswith("; ") and " = " in line and line[2:].split(" = ", 1)[0] in GLOBAL:
            k = line[2:].split(" = ", 1)[0]
            line = f"; {k} = {GLOBAL[k]}"
        lines.append(line)
    cfg = "\n".join(lines) + "\n"
    res, items, mobjs = [], [], []
    for k, o in enumerate(objs, 1):
        meshes = [o["mesh"]] + [v["mesh"] for v in o["volumes"]]
        V = np.vstack([np.asarray(m.vertices) for m in meshes])
        F, off, ranges = [], 0, []
        for m in meshes:
            F.append(np.asarray(m.faces) + off)
            ranges.append((sum(len(f) for f in F[:-1]), sum(len(f) for f in F) - 1))
            off += len(m.vertices)
        F = np.vstack(F)
        verts = "\n".join(f'     <vertex x="{a:.5f}" y="{b:.5f}" z="{c:.5f}"/>' for a, b, c in V)
        tris = "\n".join(f'     <triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in F)
        res.append(f'  <object id="{k}" type="model">\n   <mesh>\n    <vertices>\n{verts}\n    </vertices>\n'
                   f'    <triangles>\n{tris}\n    </triangles>\n   </mesh>\n  </object>\n')
        t = o["translation"]
        items.append(f'  <item objectid="{k}" transform="1 0 0 0 1 0 0 0 1 {t[0]:.4f} {t[1]:.4f} 0" printable="1"/>\n')
        meta = "".join(f'  <metadata type="object" key="{a}" value="{b}"/>\n'
                       for a, b in dict(extruder=o["extruder"], **o["settings"]).items())
        vols = [(o["name"], "ModelPart", {})] + [(v["name"], v["type"], v.get("settings", {})) for v in o["volumes"]]
        vx = ""
        for (nm, typ, st), (a, b) in zip(vols, ranges):
            keys = "".join(f'   <metadata type="volume" key="{kk}" value="{vv}"/>\n' for kk, vv in st.items())
            vx += (f'  <volume firstid="{a}" lastid="{b}">\n   <metadata type="volume" key="name" value="{nm}"/>\n'
                   f'   <metadata type="volume" key="volume_type" value="{typ}"/>\n'
                   f'   <metadata type="volume" key="matrix" value="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"/>\n{keys}'
                   f'   <mesh edges_fixed="0" degenerate_facets="0" facets_removed="0" facets_reversed="0" backwards_edges="0"/>\n'
                   f'  </volume>\n')
        mobjs.append(f' <object id="{k}" instances_count="1">\n  <metadata type="object" key="name" value="{o["name"]}"/>\n'
                     f'{meta}{vx} </object>\n')
    model = (f'<?xml version="1.0" encoding="UTF-8"?>\n<model unit="millimeter" xml:lang="en-US" '
             f'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" xmlns:slic3rpe="http://schemas.slic3r.org/3mf/2017/06">\n'
             f' <metadata name="slic3rpe:Version3mf">1</metadata>\n <metadata name="Title">{path.stem}</metadata>\n'
             f' <metadata name="Application">scripts/dip_link_print_project.py</metadata>\n <resources>\n'
             + "".join(res) + ' </resources>\n <build>\n' + "".join(items) + ' </build>\n</model>\n')
    mcfg = '<?xml version="1.0" encoding="UTF-8"?>\n<config>\n' + "".join(mobjs) + '</config>\n'
    pz = ('<?xml version="1.0" encoding="utf-8"?>\n<custom_gcodes_per_print_z bed_idx="0">\n'
          '<mode value="MultiAsSingle"/>\n</custom_gcodes_per_print_z>\n')
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


def slice_project(proj, cfg, work):
    work.mkdir(parents=True, exist_ok=True)
    kv = {}
    for line in cfg.splitlines():
        if line.startswith("; ") and " = " in line:
            k, v = line[2:].split(" = ", 1)
            kv[k] = v
    ini = work / "project.ini"
    ini.write_text("\n".join(f"{k} = {v}" for k, v in kv.items() if k != "binary_gcode") + "\nbinary_gcode = 0\n")
    gc = work / (proj.stem + ".gcode")
    r = subprocess.run(PRUSA + ["--load", str(ini), "--export-gcode", "-o", str(gc), str(proj)], capture_output=True, text=True)
    if r.returncode != 0 or not gc.exists():
        raise SystemExit(f"PrusaSlicer failed: {(r.stdout + r.stderr)[-1500:]}")
    return gc, (r.stdout + r.stderr)[-3000:]


# ------------------------------------------------------------------------------------------------ G-code checks
def gcode_summary(gc):
    """Estimated time, filament per tool (g), tool changes, M601 count, and the M486 object labels."""
    out = dict(tool_changes=0, m601=0, labels={})
    cur = None
    last_t = None
    for line in open(gc):
        if line.startswith("M486 S"):
            cur = int(line.split()[1][1:])
        elif line.startswith("M486 A") and cur is not None:
            out["labels"][line[6:].strip()] = cur
        elif line.startswith("M601"):
            out["m601"] += 1
        elif line[:1] == "T" and line[1:2].isdigit() and (len(line) < 3 or line[2] in " \n;"):
            t = int(line[1])
            if last_t is not None and t != last_t:
                out["tool_changes"] += 1
            last_t = t
        elif "estimated printing time (normal mode)" in line:
            out["print_time"] = line.split("=")[1].strip()
        elif line.startswith("; filament used [g]"):
            out["filament_g_per_tool"] = [float(v) for v in line.split("=")[1].split(",")]
        elif line.startswith("; total filament used [g]"):
            out["filament_g"] = float(line.split("=")[1])
        elif line.startswith("; total toolchanges"):
            out["toolchanges_reported"] = int(line.split("=")[1])
    return out


def object_segments(gc, label):
    """Extrusion segments of one object: (z, x0, y0, x1, y1, type) with the arcs split into chords (MP.parse_arcs)."""
    import metric_magnet_prints as MP
    L, T = MP.parse_arcs(gc, label, skip_types=("Custom", "Wipe tower", "Skirt/Brim", "Skirt"))
    rows = []
    for Lr in L:
        s = Lr["seg"]
        rows.append(np.column_stack([np.full(len(s), Lr["z"]), s[:, :4], s[:, 5]]))
    return (np.vstack(rows) if rows else np.zeros((0, 6))), T


def check_link(gc, label, enf, windows, plate_box, t):
    """Supports only inside the enforcers' footprints and under their tops; the chip windows open (no bead inside a window
    on the layers through it); the beads at the windows' tops and over the window plate by type (bridges, overhangs)."""
    S, T = object_segments(gc, label)
    tid = {v: k for k, v in T.items()}
    sup_t = [tid[k] for k in ("Support material", "Support material interface") if k in tid]
    sup = S[np.isin(S[:, 5], sup_t)]
    mid = np.column_stack([(sup[:, 1] + sup[:, 3]) / 2 - t[0], (sup[:, 2] + sup[:, 4]) / 2 - t[1], sup[:, 0]])
    inside = np.zeros(len(sup), bool)
    per = {}
    for nm, lo, hi in enf:
        m = np.where(mid[:, 2] <= 0.25, 3.0, 0.5)           # the support's first layer spreads up to 3 mm for adhesion
        k = ((mid[:, 0] >= lo[0] - m) & (mid[:, 0] <= hi[0] + m) & (mid[:, 1] >= lo[1] - m) & (mid[:, 1] <= hi[1] + m)
             & (mid[:, 2] <= hi[2] + 0.25))
        inside |= k
        L_ = np.hypot(sup[k, 3] - sup[k, 1], sup[k, 4] - sup[k, 2]).sum()
        per[nm] = dict(segments=int(k.sum()), length_mm=round(float(L_), 1),
                       z_max=round(float(mid[k, 2].max()), 2) if k.any() else None)
    stray = mid[~inside]
    res = dict(support_segments=int(len(sup)), support_inside_enforcers=int(inside.sum()),
               support_outside=int((~inside).sum()), support_outside_examples=np.round(stray[:5], 2).tolist(),
               support_top_z=round(float(mid[:, 2].max()), 2) if len(mid) else None, per_enforcer=per)
    # chip windows: beads of the object inside each window's section on the layers through it
    obj = np.column_stack([(S[:, 1] + S[:, 3]) / 2 - t[0], (S[:, 2] + S[:, 4]) / 2 - t[1], S[:, 0]])
    win = {}
    for k, (lo, hi) in windows.items():
        shrink = 0.2
        on = (obj[:, 2] > lo[2] + shrink) & (obj[:, 2] < hi[2] - shrink)
        inn = on & (obj[:, 0] > lo[0] - 0.1) & (obj[:, 0] < hi[0] + 0.1) & (obj[:, 1] > lo[1] + shrink) & (obj[:, 1] < hi[1] - shrink)
        top = (obj[:, 2] > hi[2] - 0.05) & (obj[:, 2] < hi[2] + 0.35) & (obj[:, 0] > lo[0] - 0.1) & (obj[:, 0] < hi[0] + 0.1) \
            & (obj[:, 1] > lo[1]) & (obj[:, 1] < hi[1])
        types = sorted({T[int(v)] for v in S[top, 5]})
        win[k] = dict(beads_inside=int(inn.sum()), layers_through=int(len(np.unique(obj[on, 2]))), top_types=types)
    lo, hi = plate_box
    over = (obj[:, 0] > lo[0] - 0.1) & (obj[:, 0] < hi[0] + 0.1) & (obj[:, 1] > lo[1]) & (obj[:, 1] < hi[1]) & \
        (obj[:, 2] > lo[2] + 0.15)
    bt = tid.get("Bridge infill")
    res["windows"] = win
    res["window_plate"] = dict(types=sorted({T[int(v)] for v in S[over, 5]}),
                               bridge_infill_segments=int(((S[:, 5] == bt) & over).sum()) if bt is not None else 0)
    res["types"] = sorted(T.values())
    return res


def check_tip(gc, label):
    S, T = object_segments(gc, label)
    tid = {v: k for k, v in T.items()}
    bt = tid.get("Bridge infill")
    zs = sorted({round(float(z), 2) for z in S[S[:, 5] == bt, 0]}) if bt is not None else []
    return dict(layers=int(len(np.unique(S[:, 0]))), top_z=round(float(S[:, 0].max()), 2), bridge_infill_z=zs,
                types=sorted(T.values()))


# ------------------------------------------------------------------------------------------------ figure
def bed_figure(objs, dst, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.spatial import ConvexHull
    plt.rcParams.update({"font.family": "serif", "font.size": 13})
    fig, ax = plt.subplots(figsize=(11.0, 5.2), dpi=150)
    col = {PLA_TOOL: "#cfd8e6", TPU_TOOL: "#ead7b0"}
    allp = []
    for o in objs:
        xy = np.asarray(o["mesh"].vertices)[:, :2] + o["translation"]
        allp.append(xy)
        h = xy[ConvexHull(xy).vertices]
        ax.fill(h[:, 0], h[:, 1], color=col[o["extruder"]], ec="black", lw=1)
        for v in o["volumes"]:
            if v["type"] == "SupportEnforcer":
                lo, hi = np.asarray(v["mesh"].bounds)
                ax.add_patch(plt.Rectangle((lo[0] + o["translation"][0], lo[1] + o["translation"][1]), hi[0] - lo[0],
                                           hi[1] - lo[1], fc="none", ec="#b03030", lw=1.2, ls="--"))
        cx = 0.5 * (h[:, 0].min() + h[:, 0].max())
        zt = o["mesh"].bounds[1][2]
        lab = o.get("short", o["label"])
        ax.text(cx, h[:, 1].max() + 1.5, f"{lab}\n{zt:.1f} mm tall", ha="center", va="bottom", fontsize=13)
    allp = np.vstack(allp)
    ax.set_xlim(allp[:, 0].min() - 10, allp[:, 0].max() + 10)
    ax.set_ylim(allp[:, 1].min() - 12, allp[:, 1].max() + 22)
    enf = any(v["type"] == "SupportEnforcer" for o in objs for v in o["volumes"])
    ax.text(allp[:, 0].min() - 8, allp[:, 1].min() - 9, "Front of the printer" + ("; dashed red: support enforcers" if enf else ""),
            fontsize=12)
    ax.set_aspect("equal")
    ax.set_xlabel("Bed X (mm)")
    ax.set_ylabel("Bed Y (mm)")
    ax.set_title(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(dst)
    plt.close(fig)


def preview_figure(gc, label, t, enf, dst):
    """Side views of the DIP link's sliced extrusions: the part grey, the supports red, the enforcers dashed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    S, T = object_segments(gc, label)
    tid = {v: k for k, v in T.items()}
    sup = np.isin(S[:, 5], [tid[k] for k in ("Support material", "Support material interface") if k in tid])
    x0, y0, x1, y1, z = S[:, 1] - t[0], S[:, 2] - t[1], S[:, 3] - t[0], S[:, 4] - t[1], S[:, 0]
    plt.rcParams.update({"font.family": "serif", "font.size": 14})
    fig, axs = plt.subplots(1, 2, figsize=(12.0, 6.4), dpi=130, gridspec_kw=dict(width_ratios=[21.3, 40.7]))
    for ax, (a0, a1, lab, k_) in zip(axs, ((x0, x1, "Print X (mm) = -x_F, palmar to the right", 0),
                                           (y0, y1, "Print Y (mm) = y_F, slotted arm to the right", 1))):
        for m, c, lw in ((~sup, "#9a9a9a", 0.35), (sup, "#c0392b", 0.35)):
            segs = np.stack([np.column_stack([a0[m], z[m]]), np.column_stack([a1[m], z[m]])], axis=1)
            ax.add_collection(LineCollection(segs, colors=c, linewidths=lw))
        for nm, lo, hi in enf:
            ax.add_patch(plt.Rectangle((lo[k_], lo[2]), hi[k_] - lo[k_], hi[2] - lo[2], fc="none", ec="#7a1f1f", lw=1.0, ls="--"))
        ax.set_xlabel(lab)
        ax.set_aspect("equal")
        ax.autoscale_view()
        ax.set_ylim(-1, z.max() + 2)
        ax.grid(True, lw=0.3, alpha=0.4)
    axs[0].set_ylabel("Print Z (mm)")
    axs[0].set_title("Seen along Y")
    axs[1].set_title("Seen along X")
    fig.suptitle("DIP link as sliced: part grey, supports red, support enforcers dashed", fontsize=15)
    fig.tight_layout()
    fig.savefig(dst)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ project
def build_project(parts, out_dir, name, do_slice=True, source=None):
    """parts: {"dip_link": STL in F, "clamp_plate": STL in F, "pin_fit_coupon": STL in its print frame (optional),
    "tip": STL in its print frame (optional)}. Writes <out_dir>/<name>.3mf, .json and _bed.png; slices and checks."""
    import trimesh
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    objs = []
    for key in list(parts):
        if not parts.get(key):
            continue
        tm = trimesh.load(parts[key], force="mesh")
        R = R_LINK if key in ("dip_link", "key_coupon") else (R_CLAMP if (key == "clamp_plate" or key.startswith("channel_block"))
                                                               else None)
        m, Rt = to_print(tm, R)
        o = dict(key=key, name=key, label=LABEL[key], mesh=m, Rt=Rt, source=rel(parts[key]),
                 extruder=TPU_TOOL if (key == "tip" or key.startswith("channel_block")) else PLA_TOOL,
                 settings=dict(SETTINGS[key]), volumes=[])
        if key.startswith("channel_block"):
            n = 1 + COUPON_FLANKS.index(float(key.split("_")[-1].replace("p", ".")))
            o["short"] = f"{key.split('_')[-1].replace('p', '.')} mm per flank\n{n} notch" + ("es" if n > 1 else "")
        if key == "dip_link":
            enf, mods = link_volumes(m, Rt)
            o["enforcers"] = enf
            o["volumes"] += [dict(name=n, type="SupportEnforcer", mesh=box_mesh(lo, hi)) for n, lo, hi in enf]
            o["volumes"] += [dict(name=n, type="ParameterModifier", mesh=box_mesh(lo, hi),
                                  settings=dict(fill_density="100%", fill_pattern="rectilinear")) for n, lo, hi in mods]
            o["modifiers"] = mods
            o["windows"] = window_boxes(Rt)
            o["window_plate"] = box_F_to_print((X_BACK, -10.5, -88.71), (X_BACK + 0.85, 10.5, -67.30), Rt)
        objs.append(o)
    for o, t in zip(objs, layout([o["mesh"] for o in objs])):
        o["translation"] = t
    proj = out_dir / f"{name}.3mf"
    cfg = write_project(proj, objs)
    rep = dict(name=name, project=rel(proj), template=rel(TEMPLATE), source=source,
               print_config_set=GLOBAL, settings_file=rel(FINAL / f"{PFX}_print_settings.json"),
               objects=[dict(key=o["key"], label=o["label"], source_stl=o["source"], extruder=o["extruder"],
                             settings=o["settings"], rotation_print_from_source=o["Rt"][0].tolist(),
                             bed_translation_mm=np.round(o["translation"], 3).tolist(),
                             size_mm=np.round(o["mesh"].extents, 2).tolist(),
                             volume_mm3=round(float(o["mesh"].volume), 1),
                             volumes=[dict(name=v["name"], type=v["type"], bounds_print_mm=np.round(v["mesh"].bounds, 2).tolist(),
                                           settings=v.get("settings", {})) for v in o["volumes"]])
                        for o in objs],
               when=time.strftime("%Y-%m-%d %H:%M"))
    fig = out_dir / f"{name}_bed.png"
    tools = sorted({(o["extruder"], o["settings"]["layer_height"]) for o in objs})
    bed_figure(objs, fig, "Bed of the project: " + "; ".join(f"tool {t_}, {lh} mm layers" for t_, lh in tools))
    rep["figure"] = rel(fig)
    if do_slice:
        gc, log = slice_project(proj, cfg, WORK / name)
        sm = gcode_summary(gc)
        rep["slice"] = dict(gcode=rel(gc), **{k: v for k, v in sm.items() if k != "labels"}, labels=sm["labels"])
        try:
            import sim2real_fingertip_prints as SP
            rep["slice"]["material"] = SP.check_material(gc)
        except Exception as e:                       # noqa: BLE001
            rep["slice"]["material"] = dict(error=str(e))
        for o in objs:
            lab = sm["labels"].get(o["name"])
            if lab is None:
                continue
            t = np.asarray(o["translation"])
            if o["key"] == "dip_link":
                rep["slice"]["dip_link"] = check_link(gc, lab, o["enforcers"], o["windows"], o["window_plate"], t)
                pv = out_dir / f"{name}_dip_link_sliced.png"
                preview_figure(gc, lab, t, o["enforcers"], pv)
                rep["slice"]["dip_link"]["preview"] = rel(pv)
            elif o["key"] == "tip":
                rep["slice"]["tip"] = check_tip(gc, lab)
            dl = rep["slice"].get("dip_link")
        rep["slice"]["checks_pass"] = bool(
            sm.get("m601", 0) == 0 and (dl is None or (
                dl["support_outside"] == 0 and all(w["beads_inside"] == 0 for w in dl["windows"].values())
                and dl["window_plate"]["bridge_infill_segments"] == 0)))
    (out_dir / f"{name}.json").write_text(json.dumps(rep, indent=1, default=float))
    sl = rep.get("slice", {})
    print(f"{name}: time {sl.get('print_time')}, filament {sl.get('filament_g')} g {sl.get('filament_g_per_tool')}, tool "
          f"changes {sl.get('tool_changes')}, M601 {sl.get('m601')}, checks {'pass' if sl.get('checks_pass') else 'FAIL'}",
          flush=True)
    if sl.get("dip_link"):
        d = sl["dip_link"]
        print(f"  supports: {d['support_segments']} segments, {d['support_outside']} outside the enforcers, top z "
              f"{d['support_top_z']}; per enforcer {d['per_enforcer']}", flush=True)
        print(f"  windows: {d['windows']}", flush=True)
        print(f"  window plate: {d['window_plate']}", flush=True)
    if sl.get("tip"):
        print(f"  tip: {sl['tip']}", flush=True)
    return rep


# ------------------------------------------------------------------------------------------------ attachment fit coupon
COUPON_FLANKS = (0.0, 0.05, 0.10)       # interference per flank of the 85A channel on the PLA key, mm


def fit_coupon_meshes(flanks=COUPON_FLANKS):
    """Test T1 of decision.json: three PLA key units (the DIP link's key with its detent, a 2 mm backing plate for the
    window plate, a stop flange for the bridge) on one part, and one 85A channel block per interference: the tip's back
    region 4 mm deep, 9 mm wide, 7 mm long, with the channel and detent groove of dip_link_concept_a.tip_cutter at that
    flank, its far end marked by 1, 2 or 3 notches. All in frame F; the keys at y_F = key_yc + 11 k."""
    import dip_link_concept_a as A
    K, P = A.MeshK(), dict(A.P)
    yc, Zm = P["key_yc"], A.Z_MOUNT
    key = [k for k in A.keys(K, P)][1]                     # the key at +key_yc, with its detent
    units = []
    for k in range(3):
        unit = K.union([key, K.box((X_BACK - 0.01, yc - 4.5, Zm - 6.0), (X_BACK + 2.0, yc + 4.5, Zm + 0.01)),
                        K.box((X_BACK - 3.0, yc - 4.5, Zm), (X_BACK + 2.0, yc + 4.5, Zm + 2.0))])
        units.append(A.to_trimesh(unit).apply_translation((0.0, 11.0 * k, 0.0)))
    import trimesh
    pla = trimesh.util.concatenate(units)
    blocks = []
    for k, fl in enumerate(flanks):
        Pk = dict(P, fit_flank=fl)
        blk = K.box((X_BACK - 4.0, yc - 4.5, Zm - 7.0), (X_BACK, yc + 4.5, Zm))
        cut = A.tip_cutter(K, Pk)
        notches = K.union([K.box((X_BACK - 4.1, yc - 3.6 + 1.4 * j, Zm - 7.1), (X_BACK - 3.4, yc - 3.0 + 1.4 * j, Zm - 6.4))
                           for j in range(k + 1)])
        blocks.append((fl, A.to_trimesh(K.sub(K.sub(blk, cut), notches))))
    return pla, blocks, P


def channel_widths(gc, label, t, P, flank, x_win=(0.3, 2.8)):
    """Width of the channel void on each layer of a block as sliced, from the bead centrelines crossing the middle of the
    channel's length (print X in x_win, the groove left out) less one bead width, against the design's width at the layer's
    mid-height. The template prints the first layer 0.5 mm wide with 0.2 mm of elephant-foot compensation per side."""
    S, T = object_segments(gc, label)
    out = []
    d = P["key_depth"]
    for z in sorted(np.unique(S[:, 0])):
        L = S[S[:, 0] == z]
        pts = []
        for x0, y0, x1, y1 in L[:, 1:5]:
            n = max(2, int(np.hypot(x1 - x0, y1 - y0) / 0.05))
            u = np.linspace(0, 1, n)
            pts.append(np.column_stack([x0 + (x1 - x0) * u - t[0], y0 + (y1 - y0) * u - t[1]]))
        Q = np.vstack(pts)
        Q = Q[(Q[:, 0] > x_win[0]) & (Q[:, 0] < x_win[1]) & (np.abs(Q[:, 1]) < 4.0)]
        lo, hi = Q[Q[:, 1] < 0, 1], Q[Q[:, 1] > 0, 1]
        zm = z - 0.1                                           # mid-height of a 0.2 mm layer
        design = P["key_root"] - 2 * flank + (P["key_head"] - P["key_root"]) * min(zm, d + P["fit_floor"]) / d \
            if zm < d + P["fit_floor"] else None
        if design is None:
            break
        bead = 0.5 if z <= 0.2 + 1e-6 else 0.45                # the template's first-layer and perimeter widths
        meas = float(hi.min() - lo.max() - bead) if len(lo) and len(hi) else None
        out.append(dict(z=round(float(z), 2), design_mm=round(design, 3), sliced_mm=None if meas is None else round(meas, 3)))
    return out


def build_fit_coupon(out_dir, do_slice=True):
    """The PLA key part on tool 3 and the three 85A channel blocks on tool 2, as two projects with their STL."""
    pla, blocks, P = fit_coupon_meshes()
    out_dir = Path(out_dir)
    stl = {"key_coupon": out_dir / f"{PFX}_fit_coupon_keys_pla.stl"}
    pla.export(stl["key_coupon"])
    for fl, m in blocks:
        k = f"channel_block_{fl:.2f}".replace(".", "p")
        stl[k] = out_dir / f"{PFX}_fit_coupon_{k}_85a.stl"
        m.export(stl[k])
        SETTINGS[k] = dict(SETTINGS["tip"])
        LABEL[k] = f"85A channel block, {fl:.2f} mm per flank"
    SETTINGS["key_coupon"] = dict(PLA_OBJECT, perimeters="3", fill_density="100%", fill_pattern="rectilinear")
    LABEL["key_coupon"] = "PLA keys with detents, three"
    r1 = build_project({"key_coupon": str(stl["key_coupon"])}, out_dir, f"{PFX}_fit_coupon_keys_pla_t3", do_slice=do_slice,
                       source="scripts/dip_link_print_project.py coupon")
    r2 = build_project({k: str(v) for k, v in stl.items() if k.startswith("channel_block")}, out_dir,
                       f"{PFX}_fit_coupon_channels_85a_t2", do_slice=do_slice, source="scripts/dip_link_print_project.py coupon")
    if do_slice:
        gc = ROOT / r2["slice"]["gcode"]
        widths = {}
        for o, (fl, _) in zip(r2["objects"], blocks):
            lab = r2["slice"]["labels"][o["key"]]
            widths[f"{fl:.2f}"] = channel_widths(gc, lab, o["bed_translation_mm"], P, fl)
        r2["slice"]["channel_widths"] = widths
        (out_dir / f"{PFX}_fit_coupon_channels_85a_t2.json").write_text(json.dumps(r2, indent=1, default=float))
        for fl, w in widths.items():
            print(f"  channel {fl} mm/flank: " + "; ".join(f"z {r['z']}: {r['design_mm']} vs {r['sliced_mm']}" for r in w), flush=True)
    return r1, r2


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dip-link", default=str(FINAL / "cad" / f"{PFX}_dip_link.stl"))
    ap.add_argument("--clamp-plate", default=str(FINAL / "cad" / f"{PFX}_clamp_plate.stl"))
    ap.add_argument("--coupon", default=str(FINAL / "print" / f"{PFX}_pin_fit_coupon.stl"))
    ap.add_argument("--tip", default=str(FINAL / "print" / f"{PFX}_tip_block4_print.stl"))
    ap.add_argument("--out", default=str(FINAL / "print"))
    ap.add_argument("--name", default=f"{PFX}_print_link_clamp_coupon_pla_t3")
    ap.add_argument("--tip-name", default=f"{PFX}_print_tip_block4_channels_85a_t2_no_magnets")
    ap.add_argument("--no-slice", action="store_true")
    ap.add_argument("--coupon-only", action="store_true", help="only the attachment fit coupon (test T1)")
    a = ap.parse_args()
    if a.coupon_only:
        build_fit_coupon(a.out, do_slice=not a.no_slice)
        return
    pla = dict(dip_link=a.dip_link, clamp_plate=a.clamp_plate, pin_fit_coupon=a.coupon or None)
    build_project(pla, a.out, a.name, do_slice=not a.no_slice, source="scripts/dip_link_print_project.py")
    if a.tip:
        build_project(dict(tip=a.tip), a.out, a.tip_name, do_slice=not a.no_slice, source="scripts/dip_link_print_project.py")


if __name__ == "__main__":
    main()
