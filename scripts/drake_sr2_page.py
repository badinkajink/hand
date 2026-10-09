#!/usr/bin/env python3
"""Build the local Drake study page from probe.json; standard library only."""
from __future__ import annotations

import html
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import retro_style

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs/experiments/20261001-drake-port"


def table(head, rows):
    def cells(tag, values):
        return "".join(f"<{tag}>{html.escape(str(v))}</{tag}>" for v in values)
    return ('<div class="tw"><table><thead><tr>' + cells("th", head)
            + '</tr></thead><tbody>'
            + "".join("<tr>" + cells("td", row) + "</tr>" for row in rows)
            + "</tbody></table></div>")


def build():
    data = json.loads((DATA / "probe.json").read_text())
    k = data["kinematics"]
    ik = data["planning"]["ik"]
    raw = data["import_audit"]
    yaw = next(a for a in raw[0]["actuators"] if a["name"] == "a_thumb_yaw")
    gantry = next(a for a in raw[1]["actuators"] if a["name"] == "a_thumb_x")
    designs = {r["design"]: r["morphology_offsets_m"] for r in ik}
    source = ET.parse(ROOT / raw[0]["source"]).getroot()
    fingers = ("thumb", "index", "middle")
    origins = [[float(v) for v in source.find(f".//body[@name='{f}_mount']").get("pos").split()][:2]
               for f in fingers]
    bounds = [[float(v) for v in source.find(f".//joint[@name='{f}_{a}']").get("range").split()]
              for f in fingers for a in ("x", "y")]
    viewer = dict(designs=designs, origins=origins,
                  lower=[b[0] for b in bounds], upper=[b[1] for b in bounds])
    failure = data["live_unlock"]
    if failure["returncode"]:
        status = "Live lock → unlock → advance exits on an internal SAP assertion (Drake " + data["versions"]["drake"] + "). Re-initializing the simulator did not resolve it."
    else:
        status = "The recorded live lock → unlock → advance reproduction completed."
    values = {
        "DRAKE": data["versions"]["drake"], "MUJOCO": data["versions"]["mujoco"],
        "PYTHON": data["python"], "SAMPLES": k["samples"], "IK_COUNT": len(ik),
        "FK_ERROR": f'{k["max_tip_position_error_m"]:.2e}',
        "MIN_CLEARANCE": f'{min(r["min_enabled_collision_distance_m"] for r in ik) * 1000:.3f}',
        "SOLVERS": ", ".join(sorted({r["solver"] for r in ik})),
        "RAW_YAW": yaw["effort"], "RAW_X": gantry["effort"], "RAW_KD": yaw["kd"],
        "UNLOCK_STATUS": html.escape(status),
        "UNLOCK_LOG": html.escape(failure["stderr"] + failure["stdout"]),
        "GANTRY_MOVE": f'{data["locking"]["unlocked_max_displacement_m"] * 1000:.4f}',
        "VIEWER_DATA": json.dumps(viewer).replace("<", "\\u003c"),
        "PROVENANCE": html.escape(json.dumps({key: data[key] for key in (
            "date", "python", "platform", "versions", "source_sha256")}, indent=2)),
        "CHECK_TABLE": table(["Check", "Maximum discrepancy / result"], [
            ["Fingertip position", f'{k["max_tip_position_error_m"]:.3e} m'],
            ["Fingertip rotation matrix entry", f'{k["max_tip_rotation_matrix_error"]:.3e}'],
            ["Translational Jacobian entry", f'{k["max_tip_jacobian_error"]:.3e}'],
            ["Mass matrix entry (mixed SI coordinates)", f'{k["max_mass_matrix_entry_error"]:.3e}'],
            ["Frozen vs locked fingertip coordinates", f'{data["planning"]["max_frozen_locked_tip_difference_m"]:.3e} m'],
            ["Optimized design re-frozen; target error", f'{data["planning"]["optimized_morphology_frozen_tip_error_m"]:.3e} m'],
            ["Locked gantry drift in 0.1 s rollout", f'{data["locking"]["locked_max_drift_m"]:.3e} m'],
        ]),
        "IK_TABLE": table(["Design", "Mode", "Positions", "Tip error (μm)",
                           "Clearance (mm)", "Solve (ms)"], [
            [r["design"], r["mode"], r["nq"], f'{r["max_tip_error_m"] * 1e6:.3f}',
             f'{r["min_enabled_collision_distance_m"] * 1000:.3f}',
             f'{r["solve_s"] * 1000:.2f}'] for r in ik]),
        "MORPH_TABLE": table(["Design", "Thumb XY offset (mm)", "Index XY offset (mm)",
                              "Middle XY offset (mm)"], [
            [name] + [", ".join(f"{v * 1000:+.2f}" for v in m[i:i+2]) for i in (0, 2, 4)]
            for name, m in designs.items()]),
        "IMPORT_TABLE": table(["Original source", "q", "v", "Actuators", "Warnings"], [
            [Path(r["source"]).name, r["nq"], r["nv"], r["nu"], len(r["warnings"])] for r in raw]),
        "WARNINGS": "".join("<h3>" + html.escape(r["source"]) + "</h3><pre>"
                            + html.escape("\n".join(r["warnings"])) + "</pre>" for r in raw),
    }
    template = (ROOT / "scripts/drake_sr2_page.template.html").read_text()
    for key, value in values.items():
        template = template.replace("{{" + key + "}}", str(value))
    remaining = re.findall(r"\{\{[A-Z_]+\}\}", template)
    if remaining:
        raise ValueError(f"Unresolved placeholders: {remaining}")
    out = DATA / "20261001-drake_sr2_planning.html"
    out.write_text(retro_style.apply(template))  # plain page style (owner, 2026-10-09)
    print(out)


if __name__ == "__main__":
    build()
