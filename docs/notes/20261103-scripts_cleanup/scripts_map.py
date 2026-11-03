#!/usr/bin/env python3
"""Write scripts/README.md, the map of scripts/: one table per folder, each script with the first sentence of its
docstring (or of its leading comment, for a shell script).

    python3 scripts/docs/scripts_map.py            # rewrite scripts/README.md
    python3 scripts/docs/scripts_map.py --check    # exit 1 if scripts/README.md is out of date

Run it after adding, moving or archiving a script. The folder descriptions are in FOLDERS below.
"""
import argparse
import ast
import re
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
OUT = SCRIPTS / "README.md"
FOLDERS = {
    "docs": "Builders of the home page, the topic overviews and the page indices, and the parts every page builder "
            "uses: the page style, LaTeX to SVG, the prose check and the figure check.",
    "design": "Finger layouts and their search: the morphology generators, IK retargeting of the grasp keyframe, CEM "
              "grasp synthesis, the real_v1 Sobol pipeline and its screens, KaRMA and GraspGenX.",
    "hom": "Hand-object (HOM) control of the SR2 hand: the controller's rigs and tasks, the hand-object scale study, "
           "the layout landscapes and the FP toy.",
    "mechanisms": "Reorientation mechanisms on the real_v1 hands: the open-loop carry, pinch and swing, gaits and the "
                  "grasp-to-gait chain on the arm.",
    "hardware": "The bench hand: the servo plant and its fits, the tool tracker, the CB1 client, plan export and the "
                "trajectory clearance gate.",
    "contact": "Fingertip contact models and simulators: the sphere pads, the contact bed, the reorientation on each "
               "backend, Drake, Newton, MuJoCo-Warp and DiffMJX.",
    "fingertip": "The printed fingertip: its geometry and prints, finite elements, magnets and the MLX90393 readout, "
                 "the wrench model, and the calibration and crosstalk pages.",
    "ur5": "The UR5 calibration cell: session plans, the cell model, registration and analysis of the presses, the ATI "
           "sensor and the inverted rig.",
    "cad": "Printable parts: the DIP link, the UR5 mounts and the print plates.",
    "rl": "Reinforcement learning: the trainer and its launchers, the morphology runs, policy evaluation and the "
          "policy pages.",
    "cluster": "Jobs on the DeltaAI cluster.",
    "contact_surface": "A package for the distributed-contact study of 2026-10-04 (`import contact_surface`).",
}
HEAD = """# scripts/

Every script of the repository, in one folder per topic. A script imports any other script by its bare name:
`_paths.py` puts every folder here and `../src` on `sys.path`. A script in a folder starts with

```python
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/, for _paths
import _paths
```

and finds another script's file with `_paths.script("name.py")`. Scripts run from the repository root, through
`uv run --extra rl --extra gpu python scripts/<folder>/<script>.py`; result-page builders run with `.venv/bin/python`.
Until 2026-11-03 every script was directly in `scripts/`. `MOVED.tsv` maps each old path to its new one, for the
notes and pages written before. `archive/` has the scripts no longer used, with a README that says when and why each
was moved; move a script back before running it.

Shared numerical code is in `src/morphohand/geometry/` and `src/morphohand/numerics/`, not in scripts: rotations,
quaternions and rigid transforms in `morphohand.geometry.rotations`, over scipy.

This file is written by `scripts/docs/scripts_map.py` from the scripts' docstrings.
"""


def first_sentence(p):
    text = p.read_text(errors="replace")
    doc = ""
    if p.suffix == ".py":
        try:
            doc = ast.get_docstring(ast.parse(text)) or ""
        except SyntaxError:
            doc = ""
    else:
        lines = [l[1:].strip() for l in text.splitlines()[1:20] if l.startswith("#") and not l.startswith("#!")]
        doc = " ".join(l for l in lines if l and not set(l) <= set("=-#"))
    doc = " ".join(doc.split())
    m = re.match(r"(.+?[.!?])(?:\s|$)", doc)
    s = (m.group(1) if m else doc)[:220]
    return s.replace("|", "\\|") or "-"


def build():
    out = [HEAD]
    for folder in sorted(p for p in SCRIPTS.iterdir() if p.is_dir() and not p.name.startswith(("_", "."))
                         and p.name != "archive"):
        files = sorted(q for q in folder.iterdir() if q.suffix in (".py", ".sh") and q.name != "__init__.py")
        if not files:
            continue
        out.append(f"\n## {folder.name}/ ({len(files)})\n\n{FOLDERS.get(folder.name, '')}\n\n| script | what it does |\n|---|---|")
        out += [f"| `{q.name}` | {first_sentence(q)} |" for q in files]
    n = len(list((SCRIPTS / "archive").glob("*.py"))) + len(list((SCRIPTS / "archive").glob("*.sh")))
    out.append(f"\n## archive/ ({n})\n\nSee `archive/README.md`.\n")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    text = build()
    if a.check:
        if not OUT.exists() or OUT.read_text() != text:
            print(f"{OUT} is out of date; run python3 {Path(__file__).relative_to(SCRIPTS.parent)}")
            sys.exit(1)
        return
    OUT.write_text(text)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
