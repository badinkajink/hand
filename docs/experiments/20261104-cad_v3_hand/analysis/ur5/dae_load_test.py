#!/usr/bin/env python3
"""Does MuJoCo read the UR5's COLLADA visual mesh directly? Compiles a one-geom model on wrist3.dae and prints the error.

    ~/.claude/bin/resguard.sh run --mem 3G -- .venv/bin/python logs/20261104-cad_v3/ur5/dae_load_test.py
"""
from pathlib import Path

import mujoco

DAE = Path.home() / "Programs/rh56_controller/magpie_force_control/src/ur_description/meshes/ur5/visual/wrist3.dae"
xml = f'<mujoco><asset><mesh name="w3" file="{DAE}"/></asset><worldbody><geom type="mesh" mesh="w3"/></worldbody></mujoco>'
try:
    mujoco.MjModel.from_xml_string(xml)
    print("MuJoCo", mujoco.__version__, "loaded the DAE")
except Exception as e:                                            # noqa: BLE001
    print("MuJoCo", mujoco.__version__, "refused the DAE:", str(e).strip().splitlines()[0])
