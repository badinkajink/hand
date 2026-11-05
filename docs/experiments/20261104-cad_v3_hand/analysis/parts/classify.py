"""Named inventory of every solid of the cad_v3 export (logs/20261104-cad_v3/extract/parts.json).

Parts are named "Part n" in the STEP files, so each solid is classified by its geometry: the B-rep volume (which repeats
exactly across instances of one part), the bounding box, the face census and the cylinders. The kinds were confirmed by
looking at renders (r2_small_*.png, r1_*.png). The moving body of each gantry solid comes from match_gantry.py: a solid
of the gantry file whose centre of mass falls on its instance with zero offset is fixed to the frame, one offset along
the x axis by the x stage's travel rides the x stage, and one offset along x and y rides the y stage. The finger solids
are assigned to links by the joint axes (finger_axes.py): a joint's bearing sits in the parent, its keyed shaft and the
servo horn turn with the child, the servo body sits in the parent.

Writes inventory.json (one row per solid) and summary.json next to this script."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

R = Path("/home/humanoid/Programs/hand")
HERE = R / "logs/20261104-cad_v3/parts"
PJ = json.load(open(R / "logs/20261104-cad_v3/extract/parts.json"))
P = PJ["parts"]
MESH = json.load(open(HERE / "measure.json"))["mesh"]

PLA, TPU, STEEL, BRASS, AL = 1240.0, 1210.0, 7850.0, 8500.0, 2700.0

# kind: (B-rep volume mm^3, component name, material, density kg/m^3 or None, datasheet mass g or None, drop-able
# fastener-size detail for a visual model, description)
KINDS = {
    "robot_mount": (770210, "robot mount (UR5 side plate, adaptor pocket)", "PLA", PLA, None, False,
                    "251.0 x 249.9 x 25.0 mm plate, z_A 3..28, pocket for the adaptor"),
    "base_plate": (184847, "base plate", "PLA (material not recorded)", PLA, None, False,
                   "336.6 x 242.9 x 3.0 mm plate, z_A 0..3; the gantry frames hang under it"),
    "ur5_adaptor": (64185, "UR5 adaptor (ISO 9409-1-50-4-M6 side, no dowel)", "PLA", PLA, None, False,
                    "60.4 mm rounded square, 22 mm thick, four M6 counterbored holes on a 50 mm circle"),
    "gantry_frame": (53883, "gantry frame (index/middle)", "PLA", PLA, None, False, "110 x 120 x 22 mm frame"),
    "gantry_frame_thumb": (77381, "gantry frame (thumb)", "PLA", PLA, None, False, "160 x 120 x 22 mm frame"),
    "x_stage_beam": (50194, "x-stage beam (the y axis is mounted on it)", "PLA", PLA, None, False,
                     "22.5 x 156.9 x 49.9 mm beam along y_A; rides the x rail and the x nut"),
    "x_stage_beam_thumb": (51850, "x-stage beam, thumb (the y axis is mounted on it)", "PLA", PLA, None, False,
                           "24.5 x 156.9 x 49.9 mm beam along y_A"),
    "x_stepper_mount": (4927, "x-axis stepper mount", "PLA", PLA, None, False,
                        "56.8 x 20 x 24 mm bracket: plate on the frame, NEMA 8 face"),
    "x_stepper_mount_thumb": (4930, "x-axis stepper mount, thumb", "PLA", PLA, None, False,
                              "56.8 x 20 x 24 mm bracket (holes r 2.2 where the index/middle part has r 2.25)"),
    "nema8_stepper": (11194, "NEMA 8 stepper", "steel/copper", None, 60.0, False,
                      "20 x 23.5 x 28.6 mm body with connector; 4 mm shaft is a separate solid"),
    "stepper_shaft": (117, "NEMA 8 output shaft (4 mm D-shaft, 10 mm)", "steel", None, 0.0, True,
                      "separate solid; its mass is inside the stepper's 60 g"),
    "shaft_coupler": (6739, "shaft coupler 4 mm to 5 mm (D19 x 25)", "aluminium (assumed)", AL, None, False,
                      "plain cylinder D19 x 25 with 4 and 5 mm bores and a set-screw hole"),
    "lead_screw_100": (2271, "lead screw 5 mm x 100 mm", "steel", STEEL, None, False,
                       "threaded; B-rep volume 2271 is wrong, mesh volume 1603 mm^3 is used for the mass"),
    "lead_screw_150": (8194, "lead screw 5 mm x 150 mm", "steel", STEEL, None, False,
                       "threaded; B-rep volume 8194 is wrong, mesh volume 2405 mm^3 is used for the mass"),
    "lead_nut": (1077, "lead-screw nut, flanged (flange D20, barrel D8)", "brass (assumed)", BRASS, None, False,
                 "three M3 flange holes"),
    "mgn7_rail_100": (3254, "MGN7 rail 100 mm", "steel", STEEL, None, False, "7 x 4.8 mm section"),
    "mgn7_rail_150": (4919, "MGN7 rail 150 mm", "steel", STEEL, None, False, "7 x 4.8 mm section"),
    "mgn7_carriage": (1923, "MGN7C carriage block", "steel", None, 10.0, False,
                      "22.5 x 17 x 6.5 mm simplified block; mass from the HIWIN MGN7C datasheet (10 g)"),
    "mr105_bearing": (236, "MR105 bearing 5 x 10 x 4", "steel", STEEL, None, True, "drawn as a plain ring"),
    "ring_5x10x5": (295, "ring 5 x 10 x 5 mm (shaft collar or 5 mm wide bearing; drawn as a plain ring)", "steel",
                    STEEL, None, True, "sits next to an MR105 at one end of each lead screw"),
    "scs0009_servo": (6673, "Feetech SCS0009 servo", "plastic/steel", None, 13.1, False,
                      "mass 13.1 g as cad_v1 (Switch Science 8042; owner 13 g)"),
    "horn_disc": (429, "SCS0009 disc horn D18", "plastic", PLA, None, False,
                  "yaw and MCP joints; visible outside the link, so kept in a visual model"),
    "horn_arm": (191, "SCS0009 arm horn", "plastic", PLA, None, False,
                 "PIP joint; enters the DIP link's 2 mm slot; visible, kept"),
    "joint_shaft": (214, "keyed joint shaft D5 (through the bearing, key in the child's slotted bore)", "PLA",
                    PLA, None, True, "one per joint"),
    "finger_carriage": (23913, "finger carriage (cad_v1 carriage)", "PLA", PLA, None, False,
                        "identical volume to assets/cad_v1/carriage.stl (23911.8 mm^3); the yaw servo and the y nut sit in it"),
    "yaw_link": (10683, "yaw link (cad_v1 mcp_mount_YAW-link)", "PLA", PLA, None, False,
                 "identical volume to assets/cad_v1/mcp_mount_YAW-link.stl (10684.2 mm^3); the MCP servo sits in it"),
    "mcp_link": (21293, "MCP link (cad_v1 pip_mount_MCP_link)", "PLA", PLA, None, False,
                 "identical volume to assets/cad_v1/pip_mount_MCP_link.stl (21294.3 mm^3); the PIP servo sits in it"),
    "dip_link": (5189, "DIP link, original (cad_v1 fingertip_mount_DIP_link)", "PLA", PLA, None, False,
                 "identical volume to assets/cad_v1/fingertip_mount_DIP_link.stl (5188.9 mm^3); not the v4b link"),
    "fingertip_tpu_block": (4822, "fingertip, original TPU block (cad_v1 fingertip)", "TPU 85A", TPU, None, False,
                            "identical volume to assets/cad_v1/fingertip.stl (4822.0 mm^3); not the v4b tip"),
    "sensor_arm": (21325, "sensor mount arm (camera and AprilTag side; top level)", "PLA", PLA, None, False,
                   "40 x 184.5 x 10 mm bracket under the base plate's +x_A edge"),
    "sensor_bracket": (7224, "sensor bracket (camera and AprilTag side; top level)", "PLA", PLA, None, False,
                       "45 x 16.7 x 75 mm plate with a gusset at the +x_A edge"),
}
SLICED = json.load(open(R / "assets/mjcf/experimental/20261013-cad_v1/print_mass.json"))["parts"]
SLICED_KEY = {"finger_carriage": "carriage", "yaw_link": "yaw_link", "mcp_link": "mcp_link", "dip_link": "dip_link"}


def kind_of(p):
    v = p["volume"]
    best = min(KINDS, key=lambda k: abs(KINDS[k][0] - v))
    if abs(KINDS[best][0] - v) > 0.002 * v + 1.5:
        raise SystemExit(f"unclassified {p['file']}/{p['name']} vol {v}")
    return best


def T_of(p):
    T = np.eye(4)
    T[:3, :3] = Rot.from_quat(p["placement"]["quat_xyzw"]).as_matrix()
    T[:3, 3] = p["placement"]["pos"]
    return T


# ---- gantry matching: which body each gantry solid rides -------------------------------------------------------------
GANTRY_INSTANCES = {"gantry_v3 <1>": ("gantry", 53883), "gantry_v3 <2>": ("gantry", 53883),
                    "gantry_v3_thumb <1>": ("gantry_thumb", 77381)}
hand = [p for p in P if p["file"] == "hand"]
key = lambda p: f"{p['file']}/{p['name']}"
match = {}          # file part -> (instance, hand part or None, offset)
inst_T = {}
stage_offsets = {}
body_of_gantry_file_part = {}
for inst, (fname, fv) in GANTRY_INSTANCES.items():
    fparts = [p for p in P if p["file"] == fname]
    iparts = [p for p in hand if p["chain"][0] == inst]
    ff = [p for p in fparts if abs(p["volume"] - fv) < 5][0]
    fi = [p for p in iparts if abs(p["volume"] - fv) < 5][0]
    T = T_of(fi) @ np.linalg.inv(T_of(ff))
    inst_T[inst] = T
    taken = set()
    pairs = []
    for p in fparts:
        c = T[:3, :3] @ np.array(p["com"]) + T[:3, 3]
        If = T[:3, :3] @ np.array(p["inertia_mm5"]) @ T[:3, :3].T
        for q in iparts:
            if abs(q["volume"] - p["volume"]) > 0.002 * p["volume"] + 1:
                continue
            dI = np.linalg.norm(If - np.array(q["inertia_mm5"])) / np.linalg.norm(q["inertia_mm5"])
            d = np.array(q["com"]) - c
            if dI < 1e-3:                                   # same orientation
                pairs.append((np.linalg.norm(d), p["name"], q["name"], d))
    pairs.sort(key=lambda t: t[0])
    used_f, used_q = set(), set()
    res = {}
    for dist, fn, qn, d in pairs:
        if fn in used_f or qn in used_q:
            continue
        # offsets allowed: 0 (fixed), along x (x stage), x + y (y stage); anything else is not a match
        if abs(d[2]) > 0.05:
            continue
        used_f.add(fn); used_q.add(qn)
        res[fn] = (qn, d)
    for p in fparts:
        m = res.get(p["name"])
        match[(inst, key(p))] = m
    # classify offsets
    offs = [m[1] for m in res.values()]
    xs = sorted({round(float(o[0]), 2) for o in offs if abs(o[0]) > 0.05})
    stage_offsets[inst] = xs
missing = defaultdict(list)
body_hand = {}
for (inst, fk), m in match.items():
    fp = next(p for p in P if key(p) == fk)
    kind = kind_of(fp)
    if m is None:
        missing[inst].append(fk)
        continue
    qn, d = m
    if np.linalg.norm(d) < 0.05:
        b = "fixed"
    elif abs(d[1]) < 0.05:
        b = "x_stage"
    else:
        b = "y_stage"
    body_hand[f"hand/{qn}"] = (inst, b, fk, [round(float(x), 3) for x in d])

# ---- finger links ----------------------------------------------------------------------------------------------------
FINGER_GANTRY = {}
for inst in ["Finger v3 <1>", "Finger v3 <2>", "Finger v3 <3>"]:
    nut = next(p for p in hand if p["chain"][0] == inst and kind_of(p) == "lead_nut")
    # the y lead screw (100 mm, along y_A) that passes through the nut's bore
    best = None
    for p in hand:
        if kind_of(p) != "lead_screw_100" or p["chain"][0] == "hand v3 (anthro config)":
            continue
        cy = [c for c in p["cylinders"] if abs(abs(c["axis"][1]) - 1) < 1e-6]
        if not cy:
            continue
        pt = np.array(cy[0]["point"])
        d = np.hypot(nut["com"][0] - pt[0], nut["com"][2] - pt[2])
        if best is None or d < best[0]:
            best = (d, p["chain"][0], p["name"])
    FINGER_GANTRY[inst] = dict(gantry=best[1], y_screw=f"hand/{best[2]}", nut=f"hand/{nut['name']}",
                               nut_to_screw_axis_mm=round(float(best[0]), 2))


def finger_link(p, inst):
    """Assign a Finger v3 solid to its link from the joint it sits on (finger_axes.py output, recomputed here)."""
    k = kind_of(p)
    if k in ("finger_carriage", "lead_nut"):
        return "carriage"
    if k == "yaw_link":
        return "yaw_link"
    if k == "mcp_link":
        return "mcp_link"
    if k == "dip_link":
        return "dip_link"
    if k == "fingertip_tpu_block":
        return "tip"
    joint = JOINT_OF[(inst, p["name"])]
    parent, child = {"yaw": ("carriage", "yaw_link"), "mcp": ("yaw_link", "mcp_link"),
                     "pip": ("mcp_link", "dip_link")}[joint]
    if k in ("scs0009_servo", "mr105_bearing"):
        return parent
    if k in ("horn_disc", "horn_arm", "joint_shaft"):
        return child
    raise SystemExit(f"no link for {p['name']} {k}")


def main_axis(p, r):
    cs = [c for c in p["cylinders"] if abs(c["r"] - r) < 0.02]
    c = max(cs, key=lambda c: c["area"])
    a = np.array(c["axis"]) / np.linalg.norm(c["axis"])
    a = a * np.sign(a[np.argmax(np.abs(a))])
    pt = np.array(c["point"])
    com = np.array(p["com"])
    return a, pt + a * np.dot(com - pt, a)


JOINT_OF = {}
JOINT_AXES = {}
for inst in ["Finger v3 <1>", "Finger v3 <2>", "Finger v3 <3>"]:
    ps = [p for p in hand if p["chain"][0] == inst]
    shafts = [p for p in ps if kind_of(p) == "joint_shaft"]
    ax = {s["name"]: main_axis(s, 2.5) for s in shafts}
    # the yaw shaft is the one along y_A; of the two others the MCP is nearer the yaw axis
    yaw = max(ax, key=lambda n: abs(ax[n][0][1]))
    rest = sorted([n for n in ax if n != yaw], key=lambda n: -ax[n][1][2])   # higher (less negative z) first
    names = {yaw: "yaw", rest[0]: "mcp", rest[1]: "pip"}
    for n, j in names.items():
        JOINT_OF[(inst, n)] = j
        JOINT_AXES[(inst, j)] = ax[n]
    for p in ps:
        k = kind_of(p)
        r = {"scs0009_servo": 6.05, "mr105_bearing": 5.0, "horn_disc": 9.0, "horn_arm": 3.0}.get(k)
        if r is None:
            continue
        a, q = main_axis(p, r)
        best = None
        for n, (a2, q2) in ax.items():
            ang = np.degrees(np.arccos(min(1.0, abs(float(np.dot(a, a2))))))
            dd = q2 - q
            off = np.linalg.norm(dd - a * np.dot(dd, a))
            if ang < 1.0 and (best is None or off < best[0]):
                best = (off, n)
        assert best[0] < 0.5, (inst, p["name"], best)
        JOINT_OF[(inst, p["name"])] = names[best[1]]

# ---- axis of a drive part: the direction of its main cylinder (or its long side), x or y of its file's frame ---------
AXIS_R = {"lead_screw_100": None, "lead_screw_150": None, "nema8_stepper": 8.0, "stepper_shaft": 2.0,
          "shaft_coupler": 9.5, "lead_nut": 10.0, "mr105_bearing": 5.0, "ring_5x10x5": 5.0}


def drive_axis(p, k, body):
    if k in ("mgn7_rail_100", "mgn7_rail_150"):
        d = np.array(p["bbox"][3:]) - np.array(p["bbox"][:3])
        return "xyz"[int(np.argmax(d))]
    if k == "mgn7_carriage":
        return "x" if "x_stage" in body else ("y" if "y_stage" in body else None)
    if k not in AXIS_R:
        return None
    r = AXIS_R[k]
    cs = [c for c in p["cylinders"] if r is None or abs(c["r"] - r) < 0.05]
    if not cs:
        return None
    c = max(cs, key=lambda c: c["area"])
    return "xyz"[int(np.argmax(np.abs(c["axis"])))]


# index/middle/thumb: z_P = z_A (both up, fingers along -z) and the thumb at x_P = -50 mm on the high-y_A side give
# x_P = -y_A + c and y_P = +x_A + b, so the index (y_P = +55 mm) rides the gantry at the larger x_A
FINGER_NAME = {}
for inst, fg in FINGER_GANTRY.items():
    g = fg["gantry"]
    FINGER_NAME[inst] = {"gantry_v3 <1>": "index", "gantry_v3 <2>": "middle", "gantry_v3_thumb <1>": "thumb"}[g]

# ---- rows ------------------------------------------------------------------------------------------------------------
rows = []
for p in P:
    k = kind_of(p)
    vol_b, comp, mat, rho, ds_mass, drop, desc = KINDS[k]
    kk = key(p)
    mesh_vol = MESH[kk]["mesh_vol"]
    vol_for_mass = mesh_vol if k.startswith("lead_screw") else p["volume"]
    if ds_mass is not None:
        mass, basis = ds_mass, "datasheet"
    else:
        mass, basis = vol_for_mass * rho * 1e-6, f"{mat} {rho:.0f} kg/m^3 x {'mesh' if k.startswith('lead_screw') else 'B-rep'} volume, solid"
    sub = p["chain"][0] if len(p["chain"]) > 1 else None
    joint = None
    if p["file"] == "hand":
        if sub and sub.startswith("Finger"):
            g = FINGER_GANTRY[sub]["gantry"]
            link = finger_link(p, sub)
            body = f"{sub} {link}"
            if link == "carriage":
                body_detail = f"{sub} carriage (rides the y stage of {g})"
            else:
                body_detail = f"{sub} {link}"
            joint = JOINT_OF.get((sub, p["name"]))
            src = None
        elif sub and sub.startswith("gantry"):
            inst, b, fk, d = body_hand[kk]
            body = f"{sub} {b}" if b != "fixed" else "palm (fixed)"
            body_detail = (f"{sub} {b.replace('_', ' ')}" if b != "fixed" else f"palm (fixed; {sub} frame side)")
            src = dict(gantry_file_part=fk, offset_from_gantry_file_mm=d)
        else:
            body, body_detail, src = "palm (fixed)", "palm (fixed; top level of the hand file)", None
            if k in ("nema8_stepper", "shaft_coupler"):
                body_detail = "palm (fixed; top level, the x drive of gantry_v3 <2>)"
            if k.startswith("sensor"):
                body_detail = "palm (fixed; sensor mounts, top level)"
    else:
        # parts of the gantry files and the single-part files: name the body they would ride
        body = None
        body_detail = None
        src = None
        if p["file"] in ("gantry", "gantry_thumb"):
            insts = [i for i, (f, _) in GANTRY_INSTANCES.items() if f == p["file"]]
            per = {}
            for inst in insts:
                m = match[(inst, kk)]
                if m is None:
                    per[inst] = "absent from this instance"
                else:
                    qn, d = m
                    b = "fixed" if np.linalg.norm(d) < 0.05 else ("x_stage" if abs(d[1]) < 0.05 else "y_stage")
                    per[inst] = f"hand/{qn} ({b})"
            src = dict(instances=per)
            # body by the first matched instance
            bs = [v.split("(")[-1].rstrip(")") for v in per.values() if v.startswith("hand/")]
            if p["file"] == "gantry" and k == "lead_nut" and not bs:
                bs = ["y_stage (moved into Finger v3 in the hand file)"]
            if p["file"] == "gantry_thumb" and k == "lead_nut" and not bs:
                bs = ["y_stage (moved into Finger v3 in the hand file)"]
            if not bs and k in ("x_stepper_mount", "nema8_stepper", "shaft_coupler", "stepper_shaft"):
                bs = ["fixed"]
            body = (bs[0] if bs else "fixed").replace("fixed", "palm (fixed)")
            body_detail = body
        else:
            body = body_detail = "palm (fixed)"
    ax = drive_axis(p, k, body or "")
    if p["file"] == "hand" and sub and sub.startswith("gantry") or p["file"] in ("gantry", "gantry_thumb") or (
            p["file"] == "hand" and k in ("nema8_stepper", "shaft_coupler") and sub is None):
        if k.startswith("gantry_frame") or k.startswith("x_stepper_mount"):
            role = "frame (fixed)" if k.startswith("gantry_frame") else "x drive: stepper mount"
        elif k.startswith("x_stage_beam"):
            role = "x stage: beam"
        elif k == "mgn7_carriage":
            role = f"{ax} stage: MGN7 carriage"
        elif k == "lead_nut":
            role = f"{ax} stage: lead-screw nut"
        else:
            role = f"{ax} drive: {k.replace('_', ' ')}"
    elif sub and sub.startswith("Finger"):
        role = {"lead_nut": "y stage: lead-screw nut (in the finger sub-assembly)"}.get(k, f"finger: {k.replace('_', ' ')}")
    else:
        role = "palm structure" if k in ("robot_mount", "base_plate", "ur5_adaptor") else (
            "sensor mount" if k.startswith("sensor") else None)
    row = dict(file=p["file"], name=p["name"], label=p["label"], sub_assembly=sub, kind=k, component=comp,
               role=role, finger=(f"{sub} ({FINGER_NAME[sub]}, inferred)" if sub in FINGER_NAME else None),
               body=body, body_detail=body_detail, joint=joint, material=mat,
               density_kg_m3=None if ds_mass is not None else rho, mass_g=round(mass, 2), mass_basis=basis,
               visual_detail_droppable=drop, volume_brep_mm3=round(p["volume"], 1), volume_mesh_mm3=round(mesh_vol, 1),
               com_mm=[round(x, 3) for x in p["com"]],
               bbox_mm=[round(x, 3) for x in p["bbox"]],
               bbox_size_mm=[round(p["bbox"][3 + i] - p["bbox"][i], 2) for i in range(3)],
               faces=p["nfaces"], census=p["census"], stl=p["stl"], note=desc)
    # one name per rigid body of the hand file: the gantry's y-stage MGN7 carriage and the finger carriage (with its
    # nut and yaw servo) are bolted together, so they form one body
    if p["file"] == "hand":
        if body.startswith("palm"):
            rb = "palm"
        elif sub and sub.startswith("gantry") and body.endswith("x_stage"):
            rb = f"{sub} x stage"
        elif sub and sub.startswith("gantry") and body.endswith("y_stage"):
            fin = next(f for f, v in FINGER_GANTRY.items() if v["gantry"] == sub)
            rb = f"{sub} y stage + {fin} carriage"
        elif sub and sub.startswith("Finger") and body.endswith("carriage"):
            rb = f"{FINGER_GANTRY[sub]['gantry']} y stage + {sub} carriage"
        else:
            rb = body
        if k.startswith("sensor"):
            row["body"] = "sensor (mount, fixed to the palm)"
    else:
        rb = None
    row["rigid_body"] = rb
    if k in SLICED_KEY:
        row["mass_g_as_printed_15pct"] = SLICED[SLICED_KEY[k]]["grams"]
    if src:
        row["gantry_match"] = src
    rows.append(row)

# ---- summary ---------------------------------------------------------------------------------------------------------
counts = {}
for f in ["hand", "gantry", "gantry_thumb", "base_plate", "robot_mount", "adaptor"]:
    counts[f] = dict(sorted(Counter(r["kind"] for r in rows if r["file"] == f).items()))
missing_out = {}
for inst, lst in missing.items():
    missing_out[inst] = [dict(part=fk, kind=kind_of(next(p for p in P if key(p) == fk)),
                              com_in_A_if_present_mm=[round(float(x), 2) for x in (
                                  inst_T[inst][:3, :3] @ np.array(next(p for p in P if key(p) == fk)["com"])
                                  + inst_T[inst][:3, 3])]) for fk in sorted(lst)]
summary = dict(
    counts_by_component=counts,
    finger_gantry=FINGER_GANTRY,
    instance_transforms={i: dict(R=np.round(T[:3, :3], 6).tolist(), t_mm=np.round(T[:3, 3], 3).tolist())
                         for i, T in inst_T.items()},
    stage_offsets_mm=stage_offsets,
    missing=missing_out,
    body_mass_g={},
)
bm = defaultdict(float)
for r in rows:
    if r["file"] == "hand":
        bm[r["rigid_body"]] += r["mass_g"]
summary["body_mass_g"] = {k: round(v, 1) for k, v in sorted(bm.items())}
summary["joint_axes_A"] = {f"{i} {j}": dict(axis=np.round(a, 4).tolist(), point_mm=np.round(q, 2).tolist())
                           for (i, j), (a, q) in JOINT_AXES.items()}
json.dump(rows, open(HERE / "inventory.json", "w"), indent=1)
json.dump(summary, open(HERE / "summary.json", "w"), indent=1)
print(json.dumps({k: v for k, v in summary.items() if k != "joint_axes_A"}, indent=1)[:6000])
