"""Validate the sweep schedule and native contact-force sign with an actual gravity load."""

from pathlib import Path
import sys
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from distributed_contact_timestep import cases, tool_wrench


def test_timestep_grid_exactly_partitions_controller_interval():
    grid = list(cases())
    assert len(grid) == len({c["id"] for c in grid}) == 42
    assert {c["physics_dt"] for c in grid} == {0.00005, 0.0005, 0.001, 0.002, 0.005, 0.01}
    for c in grid:
        assert c["physics_steps_per_control"] * c["physics_dt"] == pytest.approx(1 / c["controller_hz"])


def test_native_wrench_balances_gravity_on_free_object():
    model = mujoco.MjModel.from_xml_string('''<mujoco><option timestep=".001"/>
      <worldbody><geom type="plane" size="1 1 .1"/>
      <body name="tool" pos="0 0 .05"><freejoint/>
      <geom type="sphere" size=".05" mass="1"/></body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    for _ in range(1000):
        mujoco.mj_step(model, data)
    pl = SimpleNamespace(sim="mujoco", mj=mujoco, m=model, d=data, tool=model.body("tool").id,
                         state=lambda: {"tool_pos": data.qpos[:3]})
    assert tool_wrench(pl) == pytest.approx(np.array([0, 0, 9.81, 0, 0, 0]), abs=1e-6)
