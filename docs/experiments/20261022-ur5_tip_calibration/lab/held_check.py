"""Reads the held tips' mux board for SECONDS with the program's reader: channels, records/s, each chip's mean field
and temperature and its 1-sigma noise; and, from a session's held.bin, the same channel's last LAST_S seconds before
the session ended, for a before/after comparison. Moves nothing; run only while no program holds the port.

    /usr/bin/python3 docs/experiments/20261022-ur5_tip_calibration/lab/held_check.py PORT SECONDS [SESSION_DIR CHANNEL LAST_S]
"""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/home/will/workspace/hand")
sys.path.insert(0, str(ROOT / "src/sr2hand/driver/reskin5x/host"))
import ur5_tip_calibration as U  # noqa: E402

port, secs = sys.argv[1], float(sys.argv[2])


def show(tag, v):
    v = v.reshape(-1, 5, 4)
    print(f"  {tag}: {len(v)} records")
    for k in range(5):
        m, s = v[:, k, 1:].mean(axis=0), v[:, k, 1:].std(axis=0)
        print(f"    chip {k}: T {v[:, k, 0].mean():6.1f}  B ({m[0]:9.1f} {m[1]:9.1f} {m[2]:9.1f}) uT  "
              f"sd ({s[0]:5.1f} {s[1]:5.1f} {s[2]:5.1f})")


h = U.HeldBoardStream(port)
print("channels found:", h.channels)
h.start()
t0 = time.time()
time.sleep(secs)
h.stop = True
time.sleep(0.5)
for c in h.channels:
    t, v = h.rings[c].since(t0 + 1.0)
    print(f"mux channel {c}: {len(t) / (secs - 1.0):.0f} records/s")
    if len(v):
        show("now", np.asarray(v))
if len(sys.argv) > 3:
    sess, ch, last = Path(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
    dt = U.HeldBoardStream.DTYPE
    mm = np.memmap(sess / "held.bin", dtype=dt, mode="r")
    tail = np.array(mm[-int(200 * last * 4):])
    tail = tail[tail["b"] == ch]
    tail = tail[tail["t"] > tail["t"][-1] - last]
    print(f"{sess.name} channel {ch}, its last {last:.0f} s (to {time.strftime('%H:%M:%S', time.localtime(tail['t'][-1]))}):")
    show("before", tail["r"])
