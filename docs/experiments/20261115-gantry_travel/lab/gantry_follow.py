"""One gantry axis at a time, followed live with the DIP tags (owner, 2026-11-15: "move each stepper to its max extent,
track with apriltags, i manually adjust, you verify with apriltags"). The fingers stay where the owner put them (servo
torque off); only the station's manual gantry moves run. Each tag's pose is solved per camera from its own four corners
(one camera sees all three fingers well), taken into the arm's base frame B by that camera's calibrated pose and into
the palm frame P by the arm's TCP pose and the tool chain.

    measure   a still burst: each tag's centre in P (mm), its turn from a saved reference, per camera
    move      one gantry to a palm-frame target through /api/v1/manual/mounts, followed during the move: per tag the
              displacement in P against the commanded one (scale, off-axis turn, height change, straightness), and
              how far the tag turned (a limp finger that swung)
    watch     each tag's offset from a saved reference, printed every 0.5 s, while the owner adjusts

    /usr/bin/python3 gantry_follow.py measure [--save NAME] [--ref NAME] [--seconds 2]
    /usr/bin/python3 gantry_follow.py move --finger index --x 20        (or --y; the other coordinate stays)
    /usr/bin/python3 gantry_follow.py watch --ref NAME [--seconds 120]

Every frame's tag poses go to logs/20261115-gantry_travel/follow/<stamp>_<cmd>.jsonl; references to refs/<NAME>.json
there.
"""
import argparse
import json
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "src"))
from sr2hand.geometry.rotations import tf_inv  # noqa: E402
from sr2hand.robot.sr2_layouts import P_from_driver  # noqa: E402
from sr2hand.robot.tool_chain import SR2Hand  # noqa: E402
from sr2hand.tracking import live  # noqa: E402
from sr2hand.tracking import pose as tp  # noqa: E402
from sr2hand.tracking import tags as TG  # noqa: E402

CAL = ROOT / "docs/experiments/20261113-camera_arm_calibration/data/fits"
LENS = CAL / "20261113-lens_board_k3_arm.json"
HANDEYE = CAL / "20261113-handeye_tcp_fitted_lens_moved836.json"
TAGS = {"thumb": 24, "index": 25, "middle": 26}
NAME = {v: k for k, v in TAGS.items()}
TAG_MM = 14.0
FPS = 15
OUT = ROOT / "logs/20261115-gantry_travel/follow"
STATION = "http://10.99.99.2:8765"
TMAP = {i: TG.plate_map("tag36h11", TAG_MM, [i], [(0.0, 0.0, 0.0)]) for i in TAGS.values()}


def rot_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


class Station:
    def __init__(self):
        p = Path.home() / ".sr2_station_token"
        self.token = p.read_text().strip()

    def _req(self, path, body=None, timeout=10):
        import urllib.request
        h = {"X-Manta-Token": self.token, "Content-Type": "application/json"}
        req = urllib.request.Request(STATION + path, None if body is None else json.dumps(body).encode(), h)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())

    def state(self):
        return self._req("/api/v1/state")

    def snap(self):
        s = self.state()
        t = s.get("telemetry", {})
        return {"mount_positions": s.get("mount_positions"), "steppers": [
            {k: x[k] for k in ("id", "position_mm", "target_mm", "moving")} for x in t.get("steppers", [])],
            "servos": t.get("servos"), "servo_torque": s.get("servo_torque"), "operation": s.get("operation"),
            "last_error": s.get("last_error")}

    def move(self, finger, x, y):
        return self._req("/api/v1/manual/mounts", {"mounts": {finger: {"x": x, "y": y}}})


class Cams:
    """A thread per camera: colour at 1080p, the DIP tags detected and each one's pose solved alone, in B."""

    def __init__(self, log):
        self.cal = live.load_cameras(LENS, HANDEYE)
        self.log, self.lock, self.stop_evt = log, threading.Lock(), threading.Event()
        self.rows, self.pipes, self.threads = [], {}, []
        for dev in rs.context().devices:
            sn = dev.get_info(rs.camera_info.serial_number)
            if sn not in self.cal:
                continue
            p, c = rs.pipeline(), rs.config()
            c.enable_device(sn)
            c.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, FPS)
            prof = p.start(c)
            rgb = [s for s in prof.get_device().query_sensors() if s.get_info(rs.camera_info.name) == "RGB Camera"][0]
            for o, v in live.SESSION.items():
                for k in range(10):
                    try:
                        rgb.set_option(getattr(rs.option, o), v)
                        break
                    except RuntimeError:
                        time.sleep(0.2)
            self.pipes[sn] = p
        if not self.pipes:
            raise SystemExit("no calibrated D435 opened (is RealSense Viewer open?)")
        for sn in self.pipes:
            th = threading.Thread(target=self._run, args=(sn,), daemon=True)
            th.start()
            self.threads.append(th)

    def _run(self, sn):
        det = tp.detector(nthreads=4, quad_decimate=1.0)
        cam, T_BC = self.cal[sn]
        while not self.stop_evt.is_set():
            try:
                fs = self.pipes[sn].wait_for_frames(2000)
            except RuntimeError:
                continue
            t = time.time()
            g = cv2.cvtColor(np.asanyarray(fs.get_color_frame().get_data()), cv2.COLOR_BGR2GRAY)
            for i, cc in tp.detect(det, g, ids=set(TAGS.values())).items():
                T_CT, info = tp.solve_body({i: cc}, TMAP[i], cam, max_view_deg=None)
                if T_CT is None:
                    continue
                T_BT = T_BC @ T_CT
                view = float(tp.view_angles(TMAP[i], [i], T_CT)[0])
                row = {"t": t, "sn": sn, "tag": i, "T_BT": np.round(T_BT, 5).tolist(), "view_deg": round(view, 1),
                       "dist_mm": round(float(np.linalg.norm(T_CT[:3, 3])), 1), "rms_px": info.get("rms_px"),
                       "corners": np.round(cc, 2).tolist()}
                with self.lock:
                    self.rows.append(row)
                self.log.write(json.dumps(row) + "\n")

    def since(self, t0, t1=None):
        with self.lock:
            return [r for r in self.rows if r["t"] >= t0 and (t1 is None or r["t"] <= t1)]

    def close(self):
        self.stop_evt.set()
        for th in self.threads:
            th.join(timeout=3)
        for p in self.pipes.values():
            p.stop()


def palm_tf():
    """T_PB from the arm's TCP pose (read-only RTDE) and the tool chain; None without the arm."""
    arm = live.ArmPose()
    arm.start()
    for _ in range(40):
        if arm.T_BE is not None:
            break
        time.sleep(0.05)
    T_BE = arm.T_BE
    arm.stop()
    if T_BE is None:
        print(f"no arm pose ({arm.error}); positions stay in B", flush=True)
        return None, None
    return tf_inv(T_BE @ SR2Hand.load().T_E_P()), np.asarray(T_BE).tolist()


def stats(rows, T_PB):
    """{tag: {sn: {p (mm, P or B), R, n, view, sd}}} from the median of the rows."""
    out = {}
    for r in rows:
        out.setdefault(r["tag"], {}).setdefault(r["sn"], []).append(r)
    res = {}
    for i, per in out.items():
        res[i] = {}
        for sn, lst in per.items():
            Ts = np.array([np.asarray(r["T_BT"]) for r in lst])
            if T_PB is not None:
                Ts = np.einsum("ij,njk->nik", T_PB, Ts)
            p = np.median(Ts[:, :3, 3], 0)
            k = int(np.argmin(np.linalg.norm(Ts[:, :3, 3] - p, axis=1)))
            res[i][sn] = {"p": p, "R": Ts[k, :3, :3], "n": len(lst), "view": float(np.median([r["view_deg"] for r in lst])),
                          "sd": float(np.linalg.norm(Ts[:, :3, 3].std(0)))}
    return res


def show(st, ref=None, prefix=""):
    for i in sorted(st):
        for sn in sorted(st[i]):
            s = st[i][sn]
            line = (f"{prefix}{NAME[i]:6s} tag {i} cam {sn[-4:]}: ({s['p'][0]:+8.2f}, {s['p'][1]:+8.2f}, {s['p'][2]:+8.2f})"
                    f" mm, n {s['n']}, sd {s['sd']:.3f} mm, view {s['view']:.0f} deg")
            if ref is not None and str(i) in ref and sn in ref[str(i)]:
                q = ref[str(i)][sn]
                d = s["p"] - np.asarray(q["p"])
                line += (f" | from ref ({d[0]:+.2f}, {d[1]:+.2f}, {d[2]:+.2f}) |{np.linalg.norm(d):.2f}| mm,"
                         f" turn {rot_deg(np.asarray(q['R']).T @ s['R']):.2f} deg")
            print(line, flush=True)


def jsonable(st):
    return {str(i): {sn: {"p": s["p"].tolist(), "R": s["R"].tolist(), "n": s["n"], "view": s["view"], "sd": s["sd"]}
                     for sn, s in per.items()} for i, per in st.items()}


def load_ref(name):
    return json.loads((OUT / "refs" / f"{name}.json").read_text())["tags"] if name else None


def save_ref(name, st, extra):
    (OUT / "refs").mkdir(parents=True, exist_ok=True)
    (OUT / "refs" / f"{name}.json").write_text(json.dumps({"tags": jsonable(st), **extra}, indent=1))
    print(f"reference saved: {OUT / 'refs' / (name + '.json')}", flush=True)


def cmd_measure(a, cams, T_PB, log, st_):
    time.sleep(0.6)
    t0 = time.time()
    time.sleep(a.seconds)
    st = stats(cams.since(t0), T_PB)
    snap = st_.snap()
    show(st, load_ref(a.ref))
    log.write(json.dumps({"measure": jsonable(st), "station": snap}) + "\n")
    if a.save:
        save_ref(a.save, st, {"station": snap, "t": time.time()})


def cmd_watch(a, cams, T_PB, log, st_):
    ref = load_ref(a.ref)
    t_end = time.time() + a.seconds
    while time.time() < t_end:
        t0 = time.time()
        time.sleep(0.5)
        st = stats(cams.since(t0), T_PB)
        print(time.strftime("%H:%M:%S"), flush=True)
        show(st, ref, "  ")


def cmd_move(a, cams, T_PB, log, st_):
    s0 = st_.snap()
    pos = (s0["mount_positions"] or {}).get(a.finger)
    if pos is None:
        raise SystemExit("the station has no mount position: adopt or run its home first")
    x = pos["x"] if a.x is None else a.x
    y = pos["y"] if a.y is None else a.y
    d_cmd = np.array([x - pos["x"], y - pos["y"], 0.0])        # the driver's palm axes are P's (a translation apart)
    time.sleep(0.6)
    t0 = time.time()
    time.sleep(a.before)
    t1 = time.time()
    print(f"{time.strftime('%H:%M:%S')} {a.finger}: ({pos['x']:+.1f}, {pos['y']:+.1f}) -> ({x:+.1f}, {y:+.1f}) mm",
          flush=True)
    res = st_.move(a.finger, x, y)
    log.write(json.dumps({"move": {"finger": a.finger, "from": pos, "to": {"x": x, "y": y}, "resolved": res,
                                   "t": t1, "station_before": s0}}) + "\n")
    while True:
        s = st_.state()
        if s.get("operation") in (None, "idle") and not s.get("busy"):
            break
        if time.time() - t1 > 120:
            raise SystemExit("the station is not idle after 120 s")
        time.sleep(0.1)
    t2 = time.time()
    if s.get("last_error"):
        print(f"  station's last error: {s['last_error']}", flush=True)
    time.sleep(a.settle)
    t3 = time.time()
    time.sleep(a.after)
    b, e = stats(cams.since(t0, t1), T_PB), stats(cams.since(t3), T_PB)
    path = cams.since(t1, t3)
    s1 = st_.snap()
    print(f"  moved in {t2 - t1:.1f} s; station now {s1['mount_positions'][a.finger]}", flush=True)
    summary = {}
    for i in sorted(set(b) | set(e)):
        for sn in sorted(set(b.get(i, {})) & set(e.get(i, {}))):
            p0, p1 = b[i][sn]["p"], e[i][sn]["p"]
            d = p1 - p0
            turn = rot_deg(b[i][sn]["R"].T @ e[i][sn]["R"])
            row = {"tag": i, "sn": sn, "d": d.tolist(), "turn_deg": turn, "n0": b[i][sn]["n"], "n1": e[i][sn]["n"]}
            moving = NAME[i] == a.finger
            line = (f"  {NAME[i]:6s} tag {i} cam {sn[-4:]}: moved ({d[0]:+.2f}, {d[1]:+.2f}, {d[2]:+.2f}) "
                    f"|{np.linalg.norm(d):.2f}| mm, tag turned {turn:.2f} deg")
            if moving and np.linalg.norm(d_cmd) > 0:
                u = d_cmd / np.linalg.norm(d_cmd)
                along = float(d @ u)
                side = d - along * u
                ang = float(np.degrees(np.arctan2(np.linalg.norm(side[:2]), along)))
                pts = [np.asarray(r["T_BT"]) for r in path if r["tag"] == i and r["sn"] == sn]
                if T_PB is not None:
                    pts = [T_PB @ T for T in pts]
                c = np.array([T[:3, 3] for T in pts]) - p0
                dn = d / max(np.linalg.norm(d), 1e-9)
                dev = float(np.max(np.linalg.norm(c - np.outer(c @ dn, dn), axis=1))) if len(c) else float("nan")
                row.update({"cmd_mm": float(np.linalg.norm(d_cmd)), "along_mm": along, "scale": along / np.linalg.norm(d_cmd),
                            "off_axis_deg": ang, "dz_mm": float(d[2]), "path_dev_mm": dev, "path_n": len(c)})
                line += (f"\n      commanded {np.linalg.norm(d_cmd):.2f} mm: along {along:.2f} mm (scale {along / np.linalg.norm(d_cmd):.4f}),"
                         f" off-axis {ang:.2f} deg in the palm plane, height {d[2]:+.2f} mm, straightness "
                         f"{dev:.2f} mm over {len(c)} frames")
            print(line, flush=True)
            summary.setdefault(str(i), {})[sn] = row
    log.write(json.dumps({"move_result": summary, "station_after": s1, "t_done": t2}) + "\n")
    if a.save:
        save_ref(a.save, e, {"station": s1, "t": time.time()})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("measure", "move", "watch"):
        p = sub.add_parser(name)
        p.add_argument("--ref", help="compare with refs/NAME.json")
        if name in ("measure", "move"):
            p.add_argument("--save", help="save the (final) still positions as refs/NAME.json")
        if name == "measure":
            p.add_argument("--seconds", type=float, default=2.0)
        if name == "watch":
            p.add_argument("--seconds", type=float, default=120.0)
        if name == "move":
            p.add_argument("--finger", required=True, choices=sorted(TAGS))
            p.add_argument("--x", type=float)
            p.add_argument("--y", type=float)
            p.add_argument("--before", type=float, default=1.5)
            p.add_argument("--settle", type=float, default=1.0)
            p.add_argument("--after", type=float, default=2.0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    st_ = Station()
    T_PB, T_BE = palm_tf()
    log = open(OUT / f"{time.strftime('%Y%m%d-%H%M%S')}_{a.cmd}.jsonl", "w", buffering=1)
    log.write(json.dumps({"cmd": a.cmd, "args": vars(a), "T_BE": T_BE, "frame": "P" if T_PB is not None else "B",
                          "lens": str(LENS.relative_to(ROOT)), "handeye": str(HANDEYE.relative_to(ROOT)),
                          "station": st_.snap(), "t": time.time()}) + "\n")
    cams = Cams(log)
    try:
        {"measure": cmd_measure, "move": cmd_move, "watch": cmd_watch}[a.cmd](a, cams, T_PB, log, st_)
    finally:
        cams.close()
        log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
