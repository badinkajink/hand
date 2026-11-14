"""Throughput of the live two-camera tag tracker (sr2hand.tracking.live) on corvus, with the hex block in view.

Runs the tracker for SECONDS at one camera mode, logs every output (one JSON line per sample: times, flags, tags, RMS,
skew, solve time, the pose, and per camera its frame time, detection mode and time), receives its own UDP stream on
localhost, and writes a summary:
  per camera   frames per second, detection time by mode (roi / full), the frame age when detection finished;
  outputs      rate, tracked share (both cameras, one camera, lost), skew between the paired frames, solve time,
               frame age at publish (publish time minus the mean frame time) and at UDP receipt on localhost;
  static block the spread of the pose (position sd per axis and its norm, mm; rotation sd, deg) over the run;
  CPU          the process's CPU time per second of wall time (100 % = one core), and corvus's total load.

    /usr/bin/python3 tracker_bench.py OUT_PREFIX [--seconds 60] [--width 1920 --height 1080 --fps 30]
        [--exposure 166 --gain 64] [--nthreads 2] [--no-roi] [--arm] [--udp HOST:PORT ...]
"""
import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
from sr2hand.geometry.rotations import log_so3, quat_to_mat  # noqa: E402
from sr2hand.tracking import live  # noqa: E402
from sr2hand.tracking.stream import LOST, SINGLE, UdpReceiver  # noqa: E402

CAL = ROOT / "docs/experiments/20261113-camera_arm_calibration/data/fits"
ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--seconds", type=float, default=60)
ap.add_argument("--width", type=int, default=1920)
ap.add_argument("--height", type=int, default=1080)
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--exposure", type=float, default=166)
ap.add_argument("--gain", type=float, default=64)
ap.add_argument("--nthreads", type=int, default=2)
ap.add_argument("--no-roi", action="store_true", help="search the whole frame every time (the worst case)")
ap.add_argument("--arm", action="store_true", help="read the UR5's TCP pose for T_PO")
ap.add_argument("--udp", action="append", default=[], help="extra listener HOST:PORT")
ap.add_argument("--lens", default=str(CAL / "20261113-lens_board_k3_arm.json"))
ap.add_argument("--handeye", default=str(CAL / "20261113-handeye_tcp_fitted_lens_moved836.json"))
ap.add_argument("--map", default=str(CAL / "20261113-hex_block_map.json"))
a = ap.parse_args()
out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)

LOCAL_PORT = 45811
rx = UdpReceiver(LOCAL_PORT, host="127.0.0.1")
rx_rows = []


def receive():
    while not stop.is_set():
        m, t = rx.recv(timeout=0.2)
        if m is not None:
            rx_rows.append((m.seq, t - m.t_capture_ms, t - m.t_publish_ms))


rows, lock = [], threading.Lock()


def on_sample(s):
    m = s.msg
    r = dict(seq=m.seq, t_capture=m.t_capture_ms, t_publish=m.t_publish_ms, skew=m.skew_ms, flags=m.flags,
             n_cams=m.n_cams, n_tags=m.n_tags, rms=m.rms_px, solve_ms=s.solve_ms,
             p_BO=None if s.T_BO is None else np.round(m.p_BO, 4).tolist(),
             q_BO=None if s.T_BO is None else np.round(m.q_BO, 7).tolist(),
             p_PO=None if np.isnan(m.p_PO).any() else np.round(m.p_PO, 4).tolist(),
             cams={sn: dict(frame=d.frame, t_frame=d.t_frame_ms, t_done=d.t_done_ms, mode=d.mode,
                            det_ms=round(d.det_ms, 3), tags=sorted(d.corners)) for sn, d in s.dets.items()})
    with lock:
        rows.append(r)


dets = {}


def count(d):
    dets.setdefault(d.sn, []).append((d.frame, d.t_frame_ms, d.t_done_ms, d.mode, d.det_ms, len(d.corners)))


cams = live.load_cameras(a.lens, a.handeye, a.width, a.height)
tm = live.load_map(a.map)
settings = dict(live.SESSION, exposure=a.exposure, gain=a.gain)
arm, T_EP = None, None
if a.arm:
    from sr2hand.robot.tool_chain import SR2Hand
    arm = live.ArmPose()
    arm.start()
    T_EP = SR2Hand.load().T_E_P()
udp = [("127.0.0.1", LOCAL_PORT)] + [(h, int(p)) for h, p in (x.rsplit(":", 1) for x in a.udp)]
tr = live.Tracker(cams, tm, fps=a.fps, udp_to=udp, on_sample=on_sample, arm=arm, T_EP=T_EP, settings=settings,
                  nthreads=a.nthreads, roi=not a.no_roi)
for w in tr.workers:                                   # count every detection, used in an output or not
    w.on_detection = (lambda f: (lambda d: (count(d), f(d))))(w.on_detection)
stop = threading.Event()
threading.Thread(target=receive, daemon=True).start()


def cpu_total():
    v = [float(x) for x in open("/proc/stat").readline().split()[1:]]
    return v[3] + v[4], sum(v)                          # idle + iowait, all


tr.start()
time.sleep(3.0)                                        # cameras start, exposure settles
with lock:
    rows.clear()
for k in list(dets):
    dets[k] = []
rx_rows.clear()
t0, c0, (idle0, all0) = time.time(), os.times(), cpu_total()
time.sleep(a.seconds)
t1, c1, (idle1, all1) = time.time(), os.times(), cpu_total()
stop.set()
tr.stop()
if arm is not None:
    arm.stop()
wall = t1 - t0
proc_cpu = (c1.user - c0.user + c1.system - c0.system) / wall * 100.0
sys_load = (1.0 - (idle1 - idle0) / (all1 - all0)) * 100.0 * os.cpu_count()

with lock:
    R = list(rows)
with open(str(out) + "_samples.jsonl", "w") as f:
    for r in R:
        f.write(json.dumps(r) + "\n")


def q(x, p=(50, 95)):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return {f"p{k}": round(float(np.percentile(x, k)), 3) for k in p} | {"n": int(x.size)} if x.size else {"n": 0}


summ = dict(mode=f"{a.width}x{a.height}@{a.fps}", exposure=a.exposure, gain=a.gain, nthreads=a.nthreads, roi=not a.no_roi,
            seconds=round(wall, 2), handeye=Path(a.handeye).name, map=Path(a.map).name,
            cameras={w.sn: w.info for w in tr.workers}, errors={w.sn: w.error for w in tr.workers if w.error})
cam = {}
for sn, D in dets.items():
    D = [d for d in D if t0 * 1000 <= d[2] <= t1 * 1000]
    fr = np.array([d[0] for d in D])
    modes = {m: q([d[4] for d in D if d[3] == m]) for m in ("roi", "full", "none")}
    cam[sn] = dict(fps=round(len(D) / wall, 2), dropped=int((np.diff(fr) - 1).clip(0).sum()) if len(fr) > 1 else 0,
                   det_ms=modes, seen_share=round(float(np.mean([d[5] > 0 for d in D])), 4) if D else 0.0,
                   tags_median=float(np.median([d[5] for d in D])) if D else 0.0,
                   age_at_detection_ms=q([d[2] - d[1] for d in D]))
summ["per_camera"] = cam
fl = np.array([r["flags"] for r in R])
ok = [r for r in R if not r["flags"] & LOST]
both = [r for r in ok if not r["flags"] & SINGLE]
summ["outputs"] = dict(rate_hz=round(len(R) / wall, 2), tracked_share=round(len(ok) / max(len(R), 1), 4),
                       both_share=round(len(both) / max(len(R), 1), 4),
                       single_share=round(float(np.mean((fl & SINGLE) > 0)) if len(fl) else 0.0, 4),
                       skew_ms=q([r["skew"] for r in both], (50, 95, 100)), solve_ms=q([r["solve_ms"] for r in R]),
                       age_at_publish_ms=q([r["t_publish"] - r["t_capture"] for r in ok], (5, 50, 95)),
                       udp_local=dict(age_at_receipt_ms=q([x[1] for x in rx_rows]),
                                      hop_ms=q([x[2] for x in rx_rows], (50, 95, 99)), received=len(rx_rows)),
                       rms_px=q([r["rms"] for r in ok]), tags=q([r["n_tags"] for r in ok]))
if len(both) > 10:
    P = np.array([r["p_BO"] for r in both])
    Q = np.array([r["q_BO"] for r in both])
    R0 = quat_to_mat(Q[len(Q) // 2])
    w = np.array([log_so3(R0.T @ quat_to_mat(qq)) for qq in Q])
    summ["static"] = dict(position_sd_mm=np.round(P.std(0), 4).tolist(),
                          position_sd_norm_mm=round(float(np.linalg.norm(P.std(0))), 4),
                          rotation_sd_deg=round(float(np.degrees(np.linalg.norm(w.std(0)))), 4),
                          position_mm=np.round(P.mean(0), 2).tolist())
    PO = [r["p_PO"] for r in both if r["p_PO"] is not None]
    if PO:
        summ["static"]["p_PO_mm"] = np.round(np.mean(PO, 0), 2).tolist()
summ["cpu"] = dict(process_percent_of_one_core=round(proc_cpu, 1), corvus_cores_busy_percent=round(sys_load, 1),
                   cores=os.cpu_count())
Path(str(out) + "_summary.json").write_text(json.dumps(summ, indent=1))
o, c = summ["outputs"], summ["per_camera"]
print(f"{summ['mode']} exp {a.exposure:.0f}: outputs {o['rate_hz']} Hz, tracked {o['tracked_share']:.3f} "
      f"(both {o['both_share']:.3f}); age at publish p50 {o['age_at_publish_ms'].get('p50')} p95 "
      f"{o['age_at_publish_ms'].get('p95')} ms; skew p50 {o['skew_ms'].get('p50')} ms; solve p50 "
      f"{o['solve_ms'].get('p50')} ms; CPU {proc_cpu:.0f} % of a core", flush=True)
for sn, v in c.items():
    print(f"   {sn}: {v['fps']} fps, dropped {v['dropped']}, seen {v['seen_share']}, roi {v['det_ms']['roi']}, "
          f"full {v['det_ms']['full']}, age at detection {v['age_at_detection_ms']}", flush=True)
if "static" in summ:
    print(f"   static: position sd {summ['static']['position_sd_mm']} mm, rotation sd "
          f"{summ['static']['rotation_sd_deg']} deg, at {summ['static']['position_mm']}", flush=True)
print(f"   ts domain {[w.info.get('ts_domain') for w in tr.workers]}, metadata {[w.info.get('metadata') for w in tr.workers]}",
      flush=True)
os._exit(0)
