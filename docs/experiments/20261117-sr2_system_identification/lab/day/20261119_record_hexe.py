"""Record the HEX-E (100 Hz, the calibration program's stream, filter 3, no bias) for --minutes to a .npy-like text log, no
motion: for the owner's dead-weight check of 2026-11-19 17:3x (a weighed mass hung from a fingertip)."""
import sys, time, json
from pathlib import Path
sys.path[:0] = ["/home/will/workspace/hand/src/sr2hand/driver/reskin5x/host", "/home/will/workspace/hand/src",
                "/home/will/workspace/hand/docs/experiments/20261115-hom_bench_trials/lab"]
import numpy as np
import hom_trial as HT
minutes = float(sys.argv[1]) if len(sys.argv) > 1 else 15.0
out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/home/will/workspace/hand/logs/20261119-sr2_dead_weight") / time.strftime("%Y%m%d-%H%M%S-hexe.jsonl")
ft = HT.open_ft(True)
time.sleep(1.0)
t_end, t_last = time.time() + 60 * minutes, time.time() - 1.0
with open(out, "w") as fh:
    while time.time() < t_end:
        time.sleep(5.0)
        t1 = time.time()
        tt, vv = ft.ring.since(t_last, t1)
        fh.write(json.dumps({"t": np.asarray(tt).tolist(), "F": np.round(np.asarray(vv)[:, :6], 4).tolist()}) + "\n")
        fh.flush()
        if len(tt):
            m = np.asarray(vv)[:, :3].mean(0)
            print(f"{time.strftime('%H:%M:%S')} n {len(tt)} mean F {np.round(m, 3).tolist()}", flush=True)
        t_last = t1
ft.stop = True
print("done", out, flush=True)
