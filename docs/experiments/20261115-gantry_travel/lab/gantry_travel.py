"""Gantry travel of the SR2 hand measured with AprilTags on the backs of the DIP links, run on corvus with the two D435s
(owner, 2026-11-15: "for the middle/index, if we actuate both the mcp and pip flex to ~90 degrees, and make sure the
thumb is out of the way, we theoretically should be able to track travel"). The tags are tag36h11 with a 14.0 mm black
square: thumb 24, index 25, middle 26 (assets/tracking/20261115-dip_link_tags, which gives each tag's corners in its
DIP link's frame).

    look    both cameras live, the DIP tags found in each with their view angle and distance (a window on DISPLAY
            :1, or printed every 2 s with --headless for --seconds), and the station's joints and mounts; nothing moves
    pose    the viewing pose through the station's manual joints (the owner present): --index MCP,PIP, --middle
            MCP,PIP, --thumb YAW,MCP,PIP in simulator degrees, each optional, checked against the station's limits
    sweep   one finger's gantry through its travel along x, then y (the station's palm frame, mm), from the low end to
            the high end and back in --step mm, the other coordinate where it started; each point is held still and
            seen in a burst of --burst framesets per camera; then the gantry goes back to where it started

A sweep writes OUT/<stamp>_<finger>_sweep.jsonl: a header (limits, start, lenses and camera poses used), one line per
point (the commanded palm-frame mount and its stepper mm, the station's mount positions, servo joints and their age,
the arm's TCP pose, and per camera the tags seen with their mean corners in OpenCV's pixel convention, sd and frame
count) and an end line. The arm does not move; scripts/hardware/gantry_travel_fit.py fits the travel from the file.

    /usr/bin/python3 gantry_travel.py look [--headless --seconds 10]
    /usr/bin/python3 gantry_travel.py pose --index 85,85 --middle 85,85 --thumb 0,0,0
    /usr/bin/python3 gantry_travel.py sweep --finger index --out OUT [--step 5 --burst 10 --axes x,y]
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "src"))
from sr2hand.tracking import live  # noqa: E402
from sr2hand.tracking import pose as tp  # noqa: E402
from sr2hand.tracking import tags as TG  # noqa: E402

CAL = ROOT / "docs/experiments/20261113-camera_arm_calibration/data/fits"
LENS = CAL / "20261113-lens_board_k3_arm.json"
HANDEYE = CAL / "20261113-handeye_tcp_fitted_lens_moved836.json"
TAGS = {"thumb": 24, "index": 25, "middle": 26}
TAG_MM = 14.0
SETTINGS = live.SESSION
FPS = 15


# ------------------------------------------------------------------------------------------------ station

class Station:
    """The CB1 station's HTTP API: state and limits (read-only), manual joints and mounts (with the control token)."""

    def __init__(self, url, token_file):
        self.url = url.rstrip("/")
        p = Path(token_file)
        self.token = p.read_text().strip() if p.exists() else None

    def get(self, path):
        with urllib.request.urlopen(self.url + path, timeout=5) as r:
            return json.loads(r.read())

    def post(self, path, body):
        if not self.token:
            raise SystemExit("no station token (--token-file); the station refuses commands without it")
        req = urllib.request.Request(self.url + path, json.dumps(body).encode(),
                                     {"Content-Type": "application/json", "X-Manta-Token": self.token})
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())

    def wait_idle(self, what, timeout=90.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            s = self.get("/api/v1/state")
            if s.get("operation") in (None, "idle") and not s.get("busy"):
                if s.get("last_error"):
                    print(f"  station's last error: {s['last_error']}", flush=True)
                return s
            time.sleep(0.1)
        raise SystemExit(f"{what}: the station is not idle after {timeout:.0f} s ({self.get('/api/v1/state').get('operation')})")

    def snapshot(self):
        s = self.get("/api/v1/state")
        t = s.get("telemetry", {})
        return {"mount_positions": s.get("mount_positions"), "manual_mounts": s.get("manual_mounts"),
                "servos": t.get("servos"), "servo_age_s": t.get("servo_age_s"), "servo_torque": s.get("servo_torque"),
                "homed": s.get("homed"), "operation": s.get("operation"), "last_error": s.get("last_error")}


# ------------------------------------------------------------------------------------------------ cameras

def retry(fn, *args):
    for i in range(10):
        try:
            return fn(*args)
        except RuntimeError:
            if i == 9:
                raise
            time.sleep(0.2)


def open_cameras():
    cams = live.load_cameras(LENS, HANDEYE)
    out = {}
    for dev in rs.context().devices:
        sn = dev.get_info(rs.camera_info.serial_number)
        if sn not in cams:
            print(f"camera {sn} has no calibration in {HANDEYE.name}; left out", flush=True)
            continue
        p, c = rs.pipeline(), rs.config()
        c.enable_device(sn)
        c.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, FPS)
        prof = p.start(c)
        rgb = [s for s in prof.get_device().query_sensors() if s.get_info(rs.camera_info.name) == "RGB Camera"][0]
        for o, v in SETTINGS.items():
            retry(rgb.set_option, getattr(rs.option, o), v)
        out[sn] = {"pipe": p, "cam": cams[sn][0], "T_BC": cams[sn][1]}
    if not out:
        raise SystemExit("no calibrated D435 found")
    return out


DET = None


def grab(c, ids):
    fs = c["pipe"].wait_for_frames()
    col = np.asanyarray(fs.get_color_frame().get_data()).copy()
    return col, tp.detect(DET, cv2.cvtColor(col, cv2.COLOR_BGR2GRAY), ids=ids)


def one_tag_map(tag_id):
    return TG.plate_map("tag36h11", TAG_MM, [tag_id], [(0.0, 0.0, 0.0)])


def view_angle(corners, tag_id, cam):
    T, info = tp.solve_body({tag_id: corners}, one_tag_map(tag_id), cam, max_view_deg=None)
    if T is None:
        return None, None
    ang = float(tp.view_angles(one_tag_map(tag_id), [tag_id], T)[0])
    return ang, float(np.linalg.norm(T[:3, 3]))


def burst(cams, ids, n):
    """Per camera {id: (mean corners, sd px, frames)} over n framesets."""
    acc = {sn: {} for sn in cams}
    for _ in range(n):
        for sn, c in cams.items():
            _col, d = grab(c, ids)
            for i, cc in d.items():
                acc[sn].setdefault(i, []).append(cc)
    out = {}
    for sn, per in acc.items():
        out[sn] = {}
        for i, lst in per.items():
            a = np.array(lst)
            out[sn][str(i)] = {"corners": np.round(a.mean(0), 3).tolist(), "sd_px": round(float(a.std(0).max()), 3),
                               "n": len(lst)}
    return out


# ------------------------------------------------------------------------------------------------ commands

def cmd_look(a):
    global DET
    DET = tp.detector(nthreads=4)
    st = Station(a.station, a.token_file)
    try:
        print(json.dumps(st.snapshot(), indent=1), flush=True)
    except Exception as e:                                          # noqa: BLE001
        print(f"station: {e!r}", flush=True)
    cams = open_cameras()
    ids = sorted(TAGS.values())
    name = {v: k for k, v in TAGS.items()}
    WIN, PW, PH = "DIP tags (q quits)", 960, 540
    if not a.headless:
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, PW * len(cams), PH)
    last, t0 = 0.0, time.time()
    try:
        while not (a.headless and time.time() - t0 > a.seconds):
            panels = []
            for sn, c in cams.items():
                col, d = grab(c, ids)
                img, k = cv2.resize(col, (PW, PH)), PW / 1920
                txt = []
                for i, cc in d.items():
                    ang, dist = view_angle(cc, i, c["cam"])
                    ok = ang is not None and ang <= tp.MAX_VIEW_DEG
                    cv2.polylines(img, [np.int32(cc * k)], True, (0, 255, 0) if ok else (0, 165, 255), 2)
                    cv2.putText(img, f"{i}", tuple(np.int32(cc.mean(0) * k - [8, -6])), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 255), 2)
                    txt.append(f"{name[i]} {i}: {ang:.0f} deg, {dist:.0f} mm" if ang is not None else f"{name[i]} {i}")
                cv2.putText(img, f"{sn}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                for j, t in enumerate(txt):
                    cv2.putText(img, t, (12, 58 + 26 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                panels.append(img)
                if time.time() - last > 2.0:
                    print(f"{time.strftime('%H:%M:%S')} {sn}: " + ("; ".join(txt) or "no DIP tag"), flush=True)
            if time.time() - last > 2.0:
                last = time.time()
            if a.headless:
                continue
            cv2.imshow(WIN, np.hstack(panels))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        for c in cams.values():
            c["pipe"].stop()
        if not a.headless:
            cv2.destroyAllWindows()
    return 0


def cmd_pose(a):
    st = Station(a.station, a.token_file)
    lim = st.get("/api/v1/manual/limits")["joint_deg"]
    joints = {}
    for f, val, names in (("index", a.index, ("mcp", "pip")), ("middle", a.middle, ("mcp", "pip")),
                          ("thumb", a.thumb, ("yaw", "mcp", "pip"))):
        if val is None:
            continue
        v = [float(x) for x in val.split(",")]
        if len(v) != len(names):
            raise SystemExit(f"--{f} takes {','.join(names).upper()}")
        joints[f] = dict(zip(names, v))
        for j, x in joints[f].items():
            lo, hi = lim[f][j]
            if not lo <= x <= hi:
                raise SystemExit(f"{f} {j} {x:+.1f} deg is outside the station's range [{lo:.1f}, {hi:.1f}]")
    if not joints:
        raise SystemExit("nothing to move: give --index, --middle or --thumb")
    print(f"commanding {joints} at servo speed {a.servo_speed}", flush=True)
    print(json.dumps(st.post("/api/v1/manual/joints", {"joints": joints, "servo_speed": a.servo_speed})), flush=True)
    time.sleep(1.5)
    print(json.dumps(st.snapshot()["servos"]), flush=True)
    return 0


def sweep_points(lo, hi, start, step):
    """[(mm, pass)]: lo, then lo + step ... hi ("up"), back down to lo ("down"), then start ("start")."""
    up = [round(float(x), 3) for x in list(np.arange(lo, hi, step)) + [hi]]
    return [(v, "up") for v in up] + [(v, "down") for v in up[-2::-1]] + [(round(float(start), 3), "start")]


def cmd_sweep(a):
    global DET
    DET = tp.detector(nthreads=4)
    st = Station(a.station, a.token_file)
    s0 = st.snapshot()
    if not s0["homed"]:
        raise SystemExit("home the gantries first (the station's home)")
    lim = st.get("/api/v1/manual/limits")["mounts"][a.finger]
    pos = (s0["mount_positions"] or {}).get(a.finger)
    if pos is None:
        raise SystemExit(f"the station has no mount position for {a.finger}: apply a plan's morphology or a manual "
                         "mount move first")
    start = (float(pos["x"]), float(pos["y"]))
    arm = live.ArmPose()
    arm.start()
    cams = open_cameras()
    ids = sorted(TAGS.values())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{time.strftime('%Y%m%d-%H%M%S')}_{a.finger}_sweep.jsonl"
    plan = []
    for ax in a.axes.split(","):
        k = "xy".index(ax)
        for v, pas in sweep_points(lim[ax][0] + a.margin, lim[ax][1] - a.margin, start[k], a.step):
            xy = list(start)
            xy[k] = v
            plan.append((ax, pas, tuple(xy)))
    time.sleep(0.5)
    hdr = {"what": "gantry travel sweep with the DIP tags", "finger": a.finger, "tag": TAGS[a.finger], "tag_mm": TAG_MM,
           "start_mm": start, "limits_mm": lim, "margin_mm": a.margin, "step_mm": a.step, "burst": a.burst,
           "settle_s": a.settle, "points": len(plan), "lens": str(LENS.relative_to(ROOT)),
           "handeye": str(HANDEYE.relative_to(ROOT)), "cams": sorted(cams), "station_start": s0,
           "arm_error": arm.error, "t": time.time()}
    print(f"{len(plan)} points; writing {path}", flush=True)
    with open(path, "w") as fh:
        fh.write(json.dumps(hdr) + "\n")
        fh.flush()
        try:
            for n, (ax, pas, xy) in enumerate(plan):
                res = st.post("/api/v1/manual/mounts", {"mounts": {a.finger: {"x": xy[0], "y": xy[1]}}})
                st.wait_idle(f"point {n}")
                time.sleep(a.settle)
                seen = burst(cams, ids, a.burst)
                snap = st.snapshot()
                row = {"n": n, "axis": ax, "pass": pas, "cmd_mm": list(xy), "steppers_mm": res["targets"][a.finger]["steppers"],
                       "station": snap, "T_BE": None if arm.T_BE is None else np.round(arm.T_BE, 5).tolist(),
                       "t": time.time(), "cams": seen}
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                tag = str(TAGS[a.finger])
                got = ", ".join(f"{sn[-4:]} {seen[sn][tag]['n']}/{a.burst} sd {seen[sn][tag]['sd_px']:.2f}px"
                                if tag in seen[sn] else f"{sn[-4:]} -" for sn in sorted(seen))
                print(f"{time.strftime('%H:%M:%S')} {n + 1}/{len(plan)} {ax} -> ({xy[0]:+.2f}, {xy[1]:+.2f}) mm: "
                      f"tag {tag} {got}", flush=True)
        finally:
            fh.write(json.dumps({"end": time.time(), "station_end": st.snapshot()}) + "\n")
            for c in cams.values():
                c["pipe"].stop()
            arm.stop()
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("look", "pose", "sweep"):
        p = sub.add_parser(name)
        p.add_argument("--station", default="http://10.99.99.2:8765")
        p.add_argument("--token-file", default=str(Path.home() / ".sr2_station_token"))
        if name == "look":
            p.add_argument("--headless", action="store_true", help="no window: print the tags every 2 s")
            p.add_argument("--seconds", type=float, default=10.0, help="with --headless")
        if name == "pose":
            p.add_argument("--index", help="MCP,PIP (sim deg)")
            p.add_argument("--middle", help="MCP,PIP (sim deg)")
            p.add_argument("--thumb", help="YAW,MCP,PIP (sim deg)")
            p.add_argument("--servo-speed", type=int, default=40)
        if name == "sweep":
            p.add_argument("--finger", required=True, choices=sorted(TAGS))
            p.add_argument("--out", required=True)
            p.add_argument("--axes", default="x,y")
            p.add_argument("--step", type=float, default=5.0)
            p.add_argument("--margin", type=float, default=0.0, help="stay this far (mm) inside the station's limits")
            p.add_argument("--burst", type=int, default=10)
            p.add_argument("--settle", type=float, default=0.5)
    a = ap.parse_args()
    return {"look": cmd_look, "pose": cmd_pose, "sweep": cmd_sweep}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
