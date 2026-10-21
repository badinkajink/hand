#!/usr/bin/env python3
"""Inventory of the deployable SR2 hand for the fingertip deployment-readiness study (docs/handoff/
20261016-sim2real_fingertip_readiness.md, step 1).

Writes docs/experiments/20261016-sim2real_fingertip_readiness/manifest.json: the CAD part revision, the real_v1 base, finger
dimensions and masses, the servo plant, the control and policy IDs, the fingertip outer shapes, the magnet pockets, the
TPU recipes, the pad builder and the solver numerics. Every value carries its provenance:

  measured    a bench measurement on this hand or its parts (the source names the session)
  datasheet   a manufacturer's value
  owner       stated by the owner, not measured in a logged procedure
  cad         computed from the CAD STLs (lengths, volumes, poses)
  model       computed by a model: simulation, finite elements, slicer estimate or a fit to bench readbacks
  assumed     a modelling choice or a proxy with no measurement behind it
  pending     a hardware measurement that has not been made

Every cited file is checked: it must exist, and its size, SHA-256 and last commit are recorded. The two hand models are
compiled to read joint ranges, axis spacings and masses (one MuJoCo process, about 0.3 GB).

    ~/.claude/bin/resguard.sh run --mem 2G --cpu 200 -- .venv/bin/python scripts/sim2real_fingertip_manifest.py
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "docs/experiments/20261016-sim2real_fingertip_readiness/manifest.json"
HOME = Path.home()
FINGERS, JOINTS = ("thumb", "index", "middle"), ("yaw", "mcp", "pip")
PROVENANCE = {
    "measured": "a bench measurement on this hand or its parts",
    "datasheet": "a manufacturer's value",
    "owner": "stated by the owner, not measured in a logged procedure",
    "cad": "computed from the CAD STLs",
    "model": "computed by a model: simulation, finite elements, slicer estimate or a fit to bench readbacks",
    "assumed": "a modelling choice or a proxy with no measurement behind it",
    "pending": "a hardware measurement that has not been made",
}
CITED: dict[str, dict] = {}


def v(value, unit, prov, source, note=None):
    assert prov in PROVENANCE, prov
    out = {"value": value, "unit": unit, "provenance": prov, "source": source}
    if note:
        out["note"] = note
    return out


def cite(path, role):
    """Record a file the manifest relies on: existence, size, SHA-256, last commit (tracked files)."""
    p = Path(path).expanduser()
    p = p if p.is_absolute() else ROOT / p
    key = str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p).replace(str(HOME), "~")
    rec = {"role": role, "exists": p.exists()}
    if p.exists() and p.is_file():
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        rec.update(bytes=p.stat().st_size, sha256=h.hexdigest(),
                   mtime=time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime)))
        if p.is_relative_to(ROOT):
            r = subprocess.run(["git", "log", "-1", "--format=%h %cs", "--", key], cwd=ROOT, capture_output=True, text=True)
            rec["last_commit"] = r.stdout.strip() or "untracked or ignored"
    elif p.is_dir():
        rec["files"] = sum(1 for _ in p.iterdir())
    CITED[key] = rec
    return key


def probe(xml):
    """Joint ranges (deg), axis spacings (mm) and per-finger masses (g) of a compiled hand model."""
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(xml))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    out = {}
    for f in FINGERS:
        jid = {j: m.joint(f"{f}_{j}").id for j in JOINTS}
        anc = {j: d.xanchor[jid[j]].copy() for j in JOINTS}
        ax = {j: d.xaxis[jid[j]].copy() for j in JOINTS}

        def axis_dist(a, b):
            n = ax[b] / max(1e-12, (ax[b] ** 2).sum() ** 0.5)       # MCP and PIP axes are parallel
            w = anc[b] - anc[a]
            return float(((w - (w @ n) * n) ** 2).sum() ** 0.5)
        root = m.body(f"{f}_mount").id if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{f}_mount") >= 0 else None
        sub = [b for b in range(m.nbody) if m.body(b).name.startswith(f + "_")]
        out[f] = {"range_deg": {j: [round(math.degrees(x), 2) for x in m.jnt_range[jid[j]]] for j in JOINTS},
                  "yaw_axis_to_mcp_axis_mm": round(1e3 * float(((anc["mcp"] - anc["yaw"]) ** 2).sum() ** 0.5), 3),
                  "mcp_axis_to_pip_axis_mm": round(1e3 * axis_dist("mcp", "pip"), 3),
                  "finger_mass_g": round(1e3 * float(sum(m.body_mass[b] for b in sub)), 2),
                  "bodies": {m.body(b).name: round(1e3 * float(m.body_mass[b]), 3) for b in sub},
                  "root_body": m.body(root).name if root is not None else None}
    acts = [m.actuator(a).name for a in range(m.nu)]
    return {"nbody": m.nbody, "ngeom": m.ngeom, "nu": m.nu, "fingers": out, "actuators": acts,
            "timestep_s": float(m.opt.timestep), "impratio": float(m.opt.impratio)}


def plan_files():
    import reorient_backends as RB
    out = {}
    for hand, (tag, folder) in RB.HANDS.items():
        base = ROOT / "docs/experiments" / folder / "deploy"
        out[hand] = {"tag": tag, "plan": cite(base / f"{tag}_plan.json", f"{hand} deployed plan"),
                     "traj": cite(base / f"{tag}_traj.csv", f"{hand} per-step trajectory")}
    return out


def main():
    t0 = time.time()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cad_dir = "assets/mjcf/experimental/20261013-cad_v1"
    A = json.loads((ROOT / cad_dir / "assembly.json").read_text())
    PM = json.loads((ROOT / cad_dir / "print_mass.json").read_text())
    for f in ("carriage.stl", "mcp_mount_YAW-link.stl", "pip_mount_MCP_link.stl", "fingertip_mount_DIP_link.stl",
              "fingertip.stl", "fingertip_and_DIP_link.stl", "feetech scs0009.stl", "diagram.png"):
        cite(f"assets/cad_v1/{f}", "CAD part from the owner (2026-10-13)")
    for f in ("finger.xml", "hand.xml", "scene_screwdriver_medium.xml", "assembly.json", "spheres.json", "print_mass.json",
              "interference.json", "collision_bench.json", "sphere_contacts.json"):
        cite(f"{cad_dir}/{f}", "CAD-hand model (scripts/build_cad_v1_finger.py)")
    cite("scripts/build_cad_v1_finger.py", "CAD-hand builder")
    cite("assets/mjcf/real_v1/real_hand.xml", "real_v1 base hand")
    cite("src/morphohand/sampling/morphology.py", "REAL_V1_MOUNTS and REAL_V1_WORKSPACE")
    for dd in ("010.0", "016.0", "025.0"):
        cite(f"assets/mjcf/experimental/20261014-hand_object_scale_cad_v1/base/scene_cyl{dd}.xml",
             "CAD-hand scale-study base scene")
    real = probe(ROOT / "assets/mjcf/real_v1/real_hand.xml")
    cad = probe(ROOT / cad_dir / "hand.xml")

    sections = {}
    sections["cad_revision"] = {
        "parts_folder": "assets/cad_v1/",
        "mounted_mcp_link": v("pip_mount_MCP_link.stl", None, "owner", "owner 2026-10-13 (CAD finger page)",
                              "the STL of the MCP link mounted on the hand"),
        "model_folder": cad_dir,
        "tip_registration_residual_mm": v(A["report"]["parts"]["tip_registration_residual_mm"], "mm", "cad",
                                          f"{cad_dir}/assembly.json"),
        "carriage_export_rotation_deg": v(round(A["report"]["parts"]["carriage_export_rotation_deg"], 3), "deg", "cad",
                                          f"{cad_dir}/assembly.json", "removed by the builder"),
    }
    K = A["report"]["kinematics_mm"]
    sections["finger_dimensions"] = {
        "yaw_axis_to_mcp_axis": v(K["yaw_to_mcp"], "mm", "cad", f"{cad_dir}/assembly.json"),
        "mcp_axis_to_pip_axis": v(K["mcp_to_pip"], "mm", "cad", f"{cad_dir}/assembly.json",
                                  "real_v1 20.75 mm: the CAD finger reaches 10.05 mm further"),
        "pip_axis_to_tip_end": v(K["pip_to_tip_end"], "mm", "cad", f"{cad_dir}/assembly.json"),
        "tip_block_extent_depth_width_length": v(A["report"]["parts"]["tip_block_extent_mm"][::-1], "mm", "cad",
                                                 f"{cad_dir}/assembly.json", "17.0 x 14.8 x 21.41 mm (palmar normal, "
                                                 "across, along the finger); the RL default tpu2.7 is 22.0 mm long"),
        "compiled_real_v1": {f: {k: real["fingers"][f][k] for k in ("yaw_axis_to_mcp_axis_mm", "mcp_axis_to_pip_axis_mm")}
                             for f in FINGERS},
        "compiled_cad_v1": {f: {k: cad["fingers"][f][k] for k in ("yaw_axis_to_mcp_axis_mm", "mcp_axis_to_pip_axis_mm")}
                            for f in FINGERS},
        "joint_ranges_deg": v(real["fingers"]["index"]["range_deg"], "deg", "measured",
                              "assets/mjcf/real_v1/real_hand.xml; the range of motion measured on the hand (CAD finger page)",
                              "the CAD model keeps real_v1's ranges"),
        "joint_ranges_cad_model_deg": v(cad["fingers"]["index"]["range_deg"], "deg", "cad", f"{cad_dir}/hand.xml"),
        "joint_stops_cad": v({"yaw_link_carriage": [-85, 83], "mcp_link_carriage_from_yaw": -79, "mcp_flexion": 91,
                              "sphere_model_yaw_stop": -68}, "deg", "cad", "docs/experiments/20261013-cad_v1_finger/"
                             "20261013-cad_v1_finger_model.html (Joint stops of the CAD parts)"),
        "yaw_signs": v({"thumb": 1, "index": -1, "middle": -1}, None, "measured", "docs/hardware_control_station.md "
                       "(measured 2026-08-29)"),
        "yaw_command_bound_deg": v(70, "deg", "assumed", "docs/hardware_control_station.md (control station bound)"),
    }
    masses = {k: v(PM["parts"][k]["grams"], "g", "model", f"{cad_dir}/print_mass.json",
                   "PrusaSlicer 2.9.2 estimate, PLA 15 % infill, not weighed") for k in PM["parts"]}
    masses["servo_scs0009"] = v(13.1, "g", "datasheet", "Switch Science 8042 (assembly.json masses_note); owner 13 g")
    masses["tip_tpu_solid_model"] = v(round(A["inertia"]["tip"]["mass_kg"] * 1e3, 2), "g", "model",
                                      f"{cad_dir}/assembly.json", "the tip modelled as solid TPU at 1210 kg/m^3; a 20 % "
                                      "gyroid print weighs less (step 3 sets the printed mass)")
    masses["finger_model_total"] = v(cad["fingers"]["index"]["finger_mass_g"], "g", "model", f"{cad_dir}/hand.xml",
                                     "printed parts, three servos and the tip; no screws, bearings, shafts, horns, cables")
    masses["finger_assembled_owner_estimate"] = v(120, "g", "owner", "owner 2026-10-13 (CAD finger page)")
    masses["finger_assembled_weighed"] = v(None, "g", "pending", "CAD finger page, Future work: weigh one finger")
    masses["real_v1_finger_model"] = v(real["fingers"]["index"]["finger_mass_g"], "g", "model",
                                       "assets/mjcf/real_v1/real_hand.xml", "capsules at 1000 kg/m^3")
    sections["masses"] = masses
    sections["workspace_and_mounts"] = {
        "mounts_m": v({"thumb": [-0.050, 0.000], "index": [0.050, 0.055], "middle": [0.050, -0.055]}, "m", "cad",
                      "src/morphohand/sampling/morphology.py REAL_V1_MOUNTS"),
        "gantry_box_m": v({"thumb": [[-0.030, 0.030], [-0.055, 0.055]], "index": [[-0.030, 0.030], [-0.030, 0.030]],
                           "middle": [[-0.030, 0.030], [-0.030, 0.030]]}, "m", "measured",
                          "src/morphohand/sampling/morphology.py REAL_V1_WORKSPACE (the machine's travel)"),
        "gantry_travel_mm": v({"J0": 110.0, "J1": 60.0, "J2": 60.0, "J3": 60.0, "J4": 60.0, "J5": 60.0}, "mm", "measured",
                              "docs/hardware_control_station.md (FULL_EXTENSION_MM, corrected 2026-09-01)"),
        "palm_raised_for_cad_mm": v(A["pads"]["palm_raised_mm"], "mm", "cad", f"{cad_dir}/assembly.json",
                                    "keys and palm of the CAD scene raised by the finger's extra length"),
    }
    sections["servo_plant"] = {
        "spec": "kp0.72_kv0_fr0.18_fl0_dp0.0144",
        "position_gain": v(0.72, "N m/rad", "model", "scripts/make_work_plant_runs.py; servo-plant page",
                           "0.23 N m datasheet stall over the 18 deg of error at full duty"),
        "torque_limit": v(0.18, "N m", "datasheet", "SCS0009 overload protection at 80 % of the 0.23 N m stall at 6 V",
                          "stall torque on this hand not measured"),
        "joint_damping": v(0.0144, "N m s/rad", "model", "servo-plant page", "keeps the fitted 20 ms time constant"),
        "time_constant": v(0.02, "s", "model", "docs/experiments/20261006-servo_recalibration/ (fit to bench readbacks)"),
        "stall_torque": v(0.23, "N m", "datasheet", "SCS0009 at 6 V"),
        "stall_torque_on_hand": v(None, "N m", "pending", "servo identification protocol (20261013-servo_identification)"),
        "position_readback_rate": v(111, "Hz", "measured", "docs/hardware_control_station.md (9 servos, 1 ms FTDI latency)"),
        "closed_loop_write_read_rate": v(75, "Hz", "measured", "docs/hardware_control_station.md"),
        "present_load_calibrated": v(False, None, "measured", "docs/hardware_control_station.md: uncalibrated duty number"),
        "current_sensing": v(False, None, "datasheet", "SCS0009 has no present_current register"),
        "position_resolution": v(0.29, "deg", "datasheet", "1024 steps (servo identification protocol)"),
        "frozen_scene": cite("results/phase1/real_v1/20261011-sv1_u0308_b050_servo_tip_tpu2.7pads1/frozen_scene.xml",
                             "servo-plant D6 scene with v1 pads (the CLAUDE.md default servo plant run)"),
    }
    plans = plan_files()
    clearance = {"D1": 8.5, "D2": 10.4, "D3": 5.0, "D4": 8.7, "D5": 9.9, "D6": 9.2, "D7": 10.3, "D8": 8.2}
    policies = {}
    for tag in ("20261011-d6_servo_tpu27mesh_40M_s0", "20261011-d6_servo_tpu27mesh_40M_s1",
                "20261011-d6_servo_tpu27pads1_40M_s0", "20261011-d6_servo_tpu27pads1_40M_s1",
                "20261012-d6_servo_tpu27skin_40M_s0", "20261012-d6_servo_tpu27skin_40M_s1"):
        policies[tag] = {"checkpoint": cite(f"results/rl/{tag}/tensorboard/model_812.pt", "final D6 servo-plant policy"),
                         "config": cite(f"results/rl/{tag}/config.yaml", "training configuration")}
    for f in ("final_eval.jsonl", "transfer.jsonl", "robust.jsonl"):
        cite(f"docs/experiments/20261011-servo_plant_policies/{f}", "servo-plant policy evaluation rows")
    import hand_object_scale_planners as HP
    import hom_turn3 as HT
    sections["control_and_policies"] = {
        "deployed_plans_D1_D8": plans,
        "plan_clearance_capsule_gate_mm": v(clearance, "mm", "model", "docs/hardware_control_station.md (2026-09-08, "
                                            "scripts/real_v1_trajectory_clearance.py on real_v1 capsules; PASS >= 5.0 mm)",
                                            "the gate uses real_v1's capsules; the CAD links reach 30.9 mm dorsally"),
        "plans_on_cb1": v(sorted(t for t, _ in [(p["tag"], 0) for p in plans.values()]), None, "measured",
                          "docs/hardware_control_station.md (CB1 served directory as of 2026-09-08)"),
        "hom": v(dict(HT.DEFAULTS), None, "assumed", "scripts/hom_turn3.py DEFAULTS",
                 "the studies run it with kp_model 0.72 on the servo plant"),
        "pinned_icem": v(HP.prm_of("icem_pin"), None, "assumed", "scripts/hand_object_scale_planners.py icem_pin"),
        "icem_landscape_rows": cite("docs/experiments/20261014-scale_turn_cad_v1/planners/landscape_icem_pin.jsonl",
                                    "pinned iCEM on the CAD hand, 39 cells, placement 0"),
        "d6_servo_policies": policies,
        "policy_object_contact": v("scene", None, "model", "scripts/policy_eval_suite.py (run config scene_floor)",
                                   "evaluate with the run's own tool contact class"),
        "observation_size_A_B": v([65, 66], None, "model", "docs/hardware_control_station.md: learned actors read object "
                                  "pose and velocity, which the bench lacks"),
    }
    import tip_codesign_geometry as TG
    import tpu_tip_design as TD
    shapes = {}
    for key, label in (("tcd_rs3_re3_p0.2", "as-built block, printed (simulation key)"),
                       ("tcd_rs3_re6_p0.2", "block (3, 6), 2026-10-14 finalist"),
                       ("tcd_cap8.5_u0.1", "capsule r 8.5 mm"),
                       ("tcd_ell7.4x8.5x16x32_u0.1", "half-ellipsoid 7.4 x 8.5 x 16/32 mm"),
                       ("tcd_ell7.4x8.5x11x22_u0.1", "half-ellipsoid 7.4 x 8.5 x 11/22 mm")):
        sh = TG.shape(key)
        shapes[key] = {"label": label, "envelope_mm": [17.0, 14.8, 21.41], "provenance": "cad",
                       "source": "scripts/tip_codesign_geometry.py", "class": type(sh).__name__}
    shapes["print_block_tpu_tip_design"] = {
        "label": "block of the magnet prints (scripts/tpu_tip_design.py)", "provenance": "cad",
        "extent_mm": [TD.X_FACE - TD.X_BACK, 2 * TD.Y_HALF, TD.Z_END], "side_fillet_mm": TD.R_SIDE,
        "end_fillet_mm": TD.R_END,
        "note": "palmar long edges r 2.7 mm and end edge r 3.0 mm, the CAD insert's radii; the simulation key "
                "tcd_rs3_re3_p0.2 rounds both at 3.0 mm and is printed in 0.2 mm terraces"}
    shapes["rl_default_tpu2.7"] = {"label": "RL default block (fingertip_geometry.py)", "extent_mm": [17.0, 14.8, 22.0],
                                   "fillet_mm": 2.7, "provenance": "assumed",
                                   "note": "the D6 servo-plant policies trained on it; 0.59 mm longer than the CAD insert"}
    sections["fingertip_shapes"] = shapes
    sections["magnets_and_pockets"] = {
        "magnet": v("AplysiaTech N52 disc 3 x 2 mm, axially magnetized, nickel coated, +-0.1 mm", None, "datasheet",
                    "amazon.com/dp/B0G52HW8SY; scripts/metric_magnet_prints.py"),
        "magnet_mass_each": v(round(math.pi * 1.5 ** 2 * 2.0 * 7.5e-3, 4), "g", "model",
                              "nominal volume x 7.5 g/cm^3 sintered NdFeB", "not weighed"),
        "layout_four": v("6 mm across x 7 mm along, centred on the contact point (z 10.86 mm)", None, "cad",
                         "scripts/metric_magnet_prints.py layout('four')"),
        "layout_five": v("the four plus one at the contact point", None, "cad", "scripts/metric_magnet_prints.py"),
        "seat_diameter": v(2.7, "mm", "assumed", "owner's 90 % press fit; provisional until the coupon result",
                           "the 2.7/2.8/2.9 mm coupon decides"),
        "seat_coupon_result": v(None, None, "pending", "owner (coupon print and press test)"),
        "magnet_top_depth_mm": v({"block": 5.0, "tcd_cap8.5": 5.0, "tcd_ell7.4x8.5x16x32": 5.5, "tcd_ell7.4x8.5x11x22": 7.0},
                                 "mm", "cad", "docs/experiments/20261016-metric_magnet_prints/print/README.md",
                                 "shallowest 0.5 mm step leaving 1.8 mm TPU around each pocket"),
        "five_magnet_feasible": v({"block": True, "curved_upright": False}, None, "cad",
                                  "metric magnet README (upright slots 0.99 mm apart)"),
        "board": v("ReSkin 5X, five MLX90393, U2-U5 7.00 mm from U1, 20 x 20 mm, FR-4 0.6 mm", None, "datasheet",
                   "memory project_mlx_readout_2026_10_13; reskin_sensor circuits"),
        "fe_rows": cite("docs/experiments/20261016-metric_magnet_prints/fem.jsonl", "four/five/one-wall FE, 95A 20 %"),
        "sensing_rows": cite("docs/experiments/20261016-metric_magnet_prints/sensing.json", "Cramer-Rao bounds"),
    }
    for p in sorted((ROOT / "docs/experiments/20261016-metric_magnet_prints/print").glob("*.3mf")):
        cite(p, "95A print project of 2026-10-16 (Generic FLEX @XL)")
    sections["tpu_recipes"] = {
        "printed_95A_recipe_20261016": v(dict(material="TPU 95A", slicer_filament="Generic FLEX @XL", nozzle_C=230,
                                              extrusion_multiplier=1.08, max_volumetric_mm3_s=2.0, retract_mm=2.5,
                                              perimeters=2, infill="gyroid 20 %", top_bottom_layers=[4, 4]), None, "model",
                                         "~/wxie_workspace/fingers/merged_finger_v17.3mf filament slot 2; metric magnet sheets"),
        "siraya_85A_profile": v(dict(filament="Siraya Tech TPU 85A @XL 0.4mm Nozzle", print="ST TPU 85A 0.20mm QUALITY @XL 0.4",
                                     printer="Original Prusa XL 0.4 nozzle", nozzle_C=240, bed_C=35, extrusion_multiplier=0.96,
                                     max_volumetric_mm3_s=12, retract_mm=0.6, perimeters=3, infill="gyroid 25 %",
                                     speeds_mm_s=dict(perimeter=60, external=40, infill=60, solid=60, top=40, bridge=25)),
                                None, "datasheet", "~/Downloads/Siraya Tech TPU 85A XL.zip (3MF dated 2026-06-10); the "
                                "owner's saved presets 2026-10-16 14:14"),
        "owner_printer": v("Original Prusa XL - 5T Input Shaper 0.4 nozzle, TPU on extruder 2", None, "owner",
                           "~/wxie_workspace/fingers/merged_finger_v17.3mf; owner 2026-10-16"),
        "solid_modulus_95A": v(30.0, "MPa", "assumed", "scripts/tpu_tip_magnet.py RECIPES (E_s)"),
        "solid_modulus_85A_proxy": v(12.0, "MPa", "assumed", "scripts/tpu_tip_fem_tip.py: NinjaFlex 85A tensile modulus "
                                     "(ASTM D638)", "Siraya's printed value unmeasured; step 4 varies it"),
        "poisson": v(0.45, None, "assumed", "scripts/tpu_tip_magnet.py"),
        "pad_modulus_E_star_MPa": v({"95A 20 % one 4.76 mm magnet 5 mm": 14.6, "85A 50 %": 16.5, "85A 30 %": 6.9,
                                     "95A 20 % four 3x2 mm": 16.7, "95A 20 % five 3x2 mm": 17.88,
                                     "95A 20 % four 3x2 mm one wall": 16.4}, "MPa", "model",
                                    "docs/experiments/20261009-tpu_tip_fem/magnet_runs.jsonl; 20261016 sensing.json"),
        "as_printed_95A_20_E_star": v(19.1, "MPa", "model", "scripts/eflesh_tip_fem.py on the G-code with its arcs "
                                      "(docs/experiments/20261013-eflesh_printability/fem_arcs.jsonl, 2026-10-21)",
                                      "stiffer than the idealised model's 14.6 MPa at the beads' nominal widths; the 7.3 MPa of "
                                      "2026-10-13 came from a G-code reader that skipped arc moves (11 % of the filament)"),
        "printed_tip_indentation": v(None, "um", "pending", "bench probe map (curved tips page, Future work)"),
        "printed_tip_friction": v(None, None, "pending", "pull test, scripts/tpu_tip_calibrate.py fit --pull"),
    }
    cite("~/wxie_workspace/fingers/merged_finger_v17.3mf", "owner's PrusaSlicer template (v17)")
    cite("~/Downloads/Siraya Tech TPU 85A XL.zip", "Siraya Tech Prusa XL 0.4 mm profile")
    import reorient_backends as RB
    sections["contact_and_numerics"] = {
        "pads": v(dict(radius_mm=RB.PAD_RS * 1e3, spacing_mm=RB.PAD_S * 1e3, E_MPa=RB.E_TPU / 1e6, relaxation_s=RB.PAD_TR,
                       builder_default=RB.PAD_BUILDER, per_tip_cad=A["pads"]["n_pads"]["index"]), None, "assumed",
                  "scripts/reorient_backends.py", "E 10 MPa is not fitted to a print"),
        "builder_v1": "one K per finger, h 8.5 mm (runs before 2026-10-15)",
        "builder_v2": "K_i = E a_i / h_i, h_i the pad's depth below the tip's volume centroid (default since 2026-10-15)",
        "builder_v2_open_issue": v("v2 drops the CAD hand's 10 mm cylinder cell that Drake and v1 hold (8/12 against 12/12)",
                                   None, "model", "docs/experiments/20261015-curved_tips/cell10/summary.json"),
        "planner_and_evaluation": v(dict(timestep_s=0.001, impratio=100, cone="elliptic", solver="Newton", iterations=100,
                                         tolerance=1e-10), None, "assumed", "reorient_backends.NUMERICS['bed']"),
        "rl_trainer": v(dict(timestep_s=0.002, impratio=10, iterations=10, tolerance=1e-8), None, "assumed",
                        "curved_tips_traj.py ARMS['rl']; servo-plant page (MuJoCo-Warp at 2 ms, impratio 10)"),
        "drake": v(dict(version="1.57", hydroelastic_E_MPa=10, hunt_crossley_s_m=10, relaxation_s=0.01, solver="SAP",
                        timestep_s=0.001), None, "assumed", "curved tips page (Terms)"),
        "friction_mu": v(1.0, None, "assumed", "every replay and cell at mu 1.0 (studies vary 0.5-1.0)",
                         "printed-tip friction unmeasured"),
        "compiled_real_v1": {k: real[k] for k in ("timestep_s", "impratio", "nbody", "ngeom")},
        "compiled_cad_v1": {k: cad[k] for k in ("timestep_s", "impratio", "nbody", "ngeom")},
    }
    cite("docs/experiments/20261015-curved_tips/cell10/summary.json", "builder v1/v2/Drake on the 10 mm cylinder")
    for s in ("reorient_backends.py", "tip_codesign_rig.py", "tip_codesign_geometry.py", "curved_tips_traj.py",
              "hom_turn3.py", "hand_object_scale_planners.py", "hand_object_scale_mpc.py", "hand_object_scale_sim.py",
              "make_work_plant_runs.py", "real_v1_trajectory_clearance.py", "policy_eval_suite.py",
              "metric_magnet_prints.py", "tpu_tip_design.py", "tpu_tip_magnet.py"):
        cite(f"scripts/{s}", "code the study runs")
    cite("docs/hardware_control_station.md", "bench runbook")
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.splitlines()
    missing = [k for k, r in CITED.items() if not r["exists"]]
    man = {"title": "Deployable SR2 hand and fingertip data, inventory for the deployment-readiness study",
           "date": time.strftime("%Y-%m-%d %H:%M"), "git_head": head, "dirty_paths_left_untouched": dirty,
           "provenance_labels": PROVENANCE, "sections": sections,
           "compiled_models": {"real_v1": real, "cad_v1": cad}, "cited_files": CITED, "missing_files": missing,
           "script": "scripts/sim2real_fingertip_manifest.py", "seconds": round(time.time() - t0, 1)}
    OUT.write_text(json.dumps(man, indent=1, default=float) + "\n")
    print(f"{OUT.relative_to(ROOT)}: {len(CITED)} cited files, {len(missing)} missing {missing}; real_v1 index MCP-PIP "
          f"{real['fingers']['index']['mcp_axis_to_pip_axis_mm']} mm, CAD {cad['fingers']['index']['mcp_axis_to_pip_axis_mm']} mm; "
          f"finger mass real_v1 {real['fingers']['index']['finger_mass_g']} g, CAD {cad['fingers']['index']['finger_mass_g']} g")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
