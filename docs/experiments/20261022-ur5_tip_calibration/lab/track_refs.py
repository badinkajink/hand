"""The grid's reference presses of a session, in order: where the push passes 0.1 N and 2 N past the rest pose, along
the program's approach (down the indenter's z), and the depth between. A growing onset or depth over the night means
that the pad, its clamp or the indenter moved (the workstation's R16, 2026-10-22: the chain from the board to the HEX-E
is softer than the model). Reads the session's events, registration, ft.bin and robot.bin; moves nothing.

    /usr/bin/python3 docs/experiments/20261022-ur5_tip_calibration/lab/track_refs.py SESSION
"""
import datetime
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src/morphohand/driver/reskin5x/host"))
import ur5_tip_calibration as C  # noqa: E402

sd = Path(sys.argv[1])
reg = json.loads((sd / "registration.json").read_text())
K = np.array(reg.get("K", np.eye(3).tolist()))
d = -np.array(reg["T_BI"])[:3, 2]
f = np.fromfile(sd / "ft.bin", "<f8").reshape(-1, 9)
r = np.fromfile(sd / "robot.bin", "<f8").reshape(-1, 15)
first = None
for s in open(sd / "events.jsonl"):
    e = json.loads(s)
    if e.get("event") != "end" or e.get("status") != "ok" or not str(e.get("id", "")).startswith("ref-"):
        continue
    rec = e["rec"]
    T_rest = np.array(rec["T_rest"])
    m = (f[:, 0] >= rec["t_move0"]) & (f[:, 0] <= rec["t_stop"])
    mr = (r[:, 0] >= rec["t_move0"] - 0.05) & (r[:, 0] <= rec["t_stop"] + 0.05)
    P = np.array([C.ur_to_T(v[8:14])[:3, 3] for v in r[mr]])
    sx = np.interp(f[m, 0], r[mr, 0], (P - T_rest[:3, 3]) @ d)
    tare = np.array(rec["tare"])[:3]
    push = np.array([-(T_rest[:3, :3] @ np.linalg.solve(K, v - tare)) @ d for v in f[m, 3:6]])
    ps = np.convolve(push, np.ones(15) / 15, mode="same")
    on, top = float(sx[np.argmax(ps > 0.1)]), float(sx[np.argmax(ps > 1.95)])
    first = first or (on, top - on)
    print(f"{datetime.datetime.fromtimestamp(e['t']).strftime('%H:%M:%S')} {e['id']}: 0.1 N at {on:.3f} mm, 2 N at {top:.3f} mm, "
          f"depth {top - on:.3f} mm; from the first: onset {on - first[0]:+.3f} mm, depth {top - on - first[1]:+.3f} mm")
