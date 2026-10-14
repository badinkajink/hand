"""Recorded simulation state and native contact data for independent visual replay."""

import gzip
import json

import numpy as np

from contact_surface.records import write_json


class StateCapture:
    def __init__(self, directory, kind, interval=0.005):
        self.directory = directory
        self.kind = kind
        self.interval = interval
        self.last = -1.0
        self.count = 0
        self.file = gzip.open(directory / "states.jsonl.gz", "wt")
        self.render_model = None
        self.render_data = None

    def __call__(self, *args):
        return getattr(self, self.kind)(*args)

    def due(self, t):
        return t >= self.last + self.interval - 1e-9

    def save(self, qpos, t, points, record, meta):
        if not self.due(t) and record.get("stage") != "stopped":
            return
        if not np.isfinite(qpos).all():
            write_json(
                self.directory / "terminal_event.json",
                dict(time=t, event="nonfinite state; last finite frame retained"),
            )
            return
        self.last = t
        self.file.write(
            json.dumps(
                dict(time=t, qpos=np.asarray(qpos).tolist(), contacts=points, metrics=record),
                allow_nan=False,
            )
            + "\n"
        )
        self.count += 1
        if self.count == 1:
            write_json(self.directory / "fixture.json", meta)

    def cpu(self, model, data, meta, rows, record):
        t = float(data.time)
        if not self.due(t) and record.get("stage") != "stopped":
            return
        points = [
            [
                *[r[f"position_{i}"] for i in range(3)],
                *[r[f"normal_{i}"] for i in range(3)],
                r["f_n"],
                r["pressure_proxy"],
                r["A_i"],
                r["geom_sample"],
            ]
            for r in rows
        ]
        self.save(data.qpos, t, points, record, meta)

    def drake(self, rig, meta, faces, record):
        if self.render_model is None:
            from distributed_contact_transfer import build

            self.render_model, self.render_data, _, _ = build(
                dict(spacing_mm=1.0, policy="compiled")
            )
        rig.copy_to_mujoco(self.render_model, self.render_data)
        self.save(self.render_data.qpos, rig.t, faces, record, meta)

    def newton(self, model, solver, state, meta, points, record):
        if self.render_model is None:
            from distributed_contact_transfer import build

            self.render_model, self.render_data, _, _ = build(
                dict(spacing_mm=1.0, policy="compiled")
            )
            self.joint_starts = {
                n.split("/")[-1]: int(s)
                for n, s in zip(model.joint_label, model.joint_q_start.numpy())
            }
        q = state.joint_q.numpy()
        m, d = self.render_model, self.render_data
        for f in ("thumb", "index", "middle"):
            for suffix in ("yaw", "mcp", "pip"):
                name = f + "_" + suffix
                d.qpos[m.jnt_qposadr[m.joint(name).id]] = q[self.joint_starts[name]]
        pose = state.body_q.numpy()[meta["newton_tool"]]
        qa = meta["tqa"]
        d.qpos[qa : qa + 3] = pose[:3]
        d.qpos[qa + 3 : qa + 7] = pose[[6, 3, 4, 5]]
        self.save(d.qpos, float(record["time"]), points, record, meta)

    def close(self):
        self.file.close()
        write_json(
            self.directory / "capture.json",
            dict(
                kind=self.kind,
                frames=self.count,
                nominal_state_hz=1 / self.interval,
                qpos_order="source MJCF; quaternion wxyz",
                native_contacts=True,
            ),
        )


def load_states(path):
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f]
