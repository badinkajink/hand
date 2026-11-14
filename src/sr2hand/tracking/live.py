"""Live tracking of one tag body with the D435s on corvus (plan: docs/notes/20261105-object_seating_and_tracking/,
"Tracking software"; calibration: docs/experiments/20261113-camera_arm_calibration/).

CameraWorker  one thread per camera: colour YUYV at the chosen mode, the luma plane taken without a colour conversion,
              tags found at full resolution in a window around the body (its corners in the previous frame, or the
              fused pose projected into this camera), or over the whole frame at quad_decimate 2 when it was lost.
              Every detection carries librealsense's global frame time (host ms).
Tracker       takes each new frame of the lead camera (the first serial; the other when the lead stops) with the
              other camera's frame nearest in time, waiting up to a period for that one's next frame when the newest
              lies more than max_skew_ms (half a period + 2 ms) away; it then solves the body's pose in the base frame from all corners at once
              (pose.solve_multi), warm-started at the previous pose. When only one camera sees the body, that camera
              solves it alone (flag SINGLE). Each output goes to the UDP listeners as a stream.PoseMsg and to
              on_sample.
ArmPose       the UR5's TCP pose from read-only RTDE in a thread, for T_PO = (T_BE T_EP)^-1 T_BO with T_EP from
              sr2hand.robot.tool_chain.SR2Hand (CAD).

Lenses are the 1920 x 1080 fits; another mode of the same field of view scales them (fx, fy by w / 1920; the principal
point as (c + 0.5) s - 0.5; distortion unchanged).
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sr2hand.geometry.rotations import exp_so3, skew, tf, tf_from_ur, tf_inv, tf_to_pose

from . import pose as tp
from .stream import LOST, NO_ARM, SINGLE, PoseMsg, UdpSender
from .tags import TagMap

SESSION = {"enable_auto_exposure": 0, "enable_auto_white_balance": 0, "exposure": 166, "gain": 64,
           "white_balance": 4600}


def load_cameras(lens_json, handeye_json, width=1920, height=1080):
    """{serial: (Camera, T_BC)} from the calibration's lens and hand-eye JSONs, scaled to width x height."""
    L, H = json.loads(Path(lens_json).read_text()), json.loads(Path(handeye_json).read_text())
    s = width / 1920.0
    out = {}
    for sn, c in H["cams"].items():
        l = L["cams"][sn]["lens"]
        K = np.array([[l["fx"] * s, 0, (l["cx"] + 0.5) * s - 0.5], [0, l["fy"] * s, (l["cy"] + 0.5) * s - 0.5],
                      [0, 0, 1.0]])
        dist = np.array([l["k1"], l["k2"], l["p1"], l["p2"], l["k3"]], float)
        out[sn] = (tp.Camera(K, dist, width, height), tf_inv(np.array(c["T_CB"], float)))
    return out


def solve_joint(obs_by_cam, tag_map: TagMap, cams, T_BC, T0_BO, iters=8, max_view_deg=tp.MAX_VIEW_DEG,
                max_tag_rms=1.0, tol=1e-6):
    """(T_BO, info): the body's pose in B from every camera's corners at once, by Gauss-Newton (Levenberg damping)
    from T0_BO with the projection Jacobians of cv2.projectPoints; the same objective as pose.solve_multi in about a
    tenth of its time. Tags seen more than max_view_deg from their normal at T0 are left out, then the worst tag
    whose corners reproject worse than max_tag_rms px, once, while more than one is left. None when nothing is left."""
    import cv2
    rows = []
    for obs, cam, Tbc in zip(obs_by_cam, cams, T_BC):
        T_CB = tf_inv(Tbc)
        use = sorted(i for i in obs if i in tag_map.corners)
        if not use:
            continue
        if max_view_deg is not None:
            ang = tp.view_angles(tag_map, use, T_CB @ T0_BO)
            use = [i for i, v in zip(use, ang) if v <= max_view_deg]
        if use:
            rows.append([cam, T_CB, use, np.concatenate([obs[i] for i in use])])
    if not rows:
        return None, {"n_cams": 0}
    zero = np.zeros(3)

    def residuals(T, jac):
        rs, Js = [], []
        for cam, T_CB, use, uv in rows:
            T_CO = T_CB @ T
            X = tag_map.stacked(use)
            Xc = X @ T_CO[:3, :3].T + T_CO[:3, 3]
            p, J = cv2.projectPoints(Xc.reshape(-1, 1, 3), zero, zero, cam.K, cam.dist)
            rs.append((p.reshape(-1, 2) - uv).ravel())
            if jac:
                dpdX = J[:, 3:6].reshape(-1, 2, 3)                       # d pixel / d camera-frame point
                R = T_CO[:3, :3]
                dXdw = -np.einsum("ij,njk->nik", R, skew(X))              # right perturbation T exp([w, v])
                dXdv = np.broadcast_to(R, (len(X), 3, 3))
                Js.append(np.concatenate([dpdX @ dXdw, dpdX @ dXdv], axis=2).reshape(-1, 6))
        return np.concatenate(rs), (np.concatenate(Js) if jac else None)

    def gauss_newton(T):
        lam = 1e-6
        r, J = residuals(T, True)
        cost = r @ r
        for _ in range(iters):
            H, g = J.T @ J, J.T @ r
            d = -np.linalg.solve(H + lam * np.diag(np.diag(H)), g)
            Tn = T @ tf(exp_so3(d[:3]), d[3:])
            rn, Jn = residuals(Tn, True)
            cn = rn @ rn
            if cn < cost:
                T, r, J, cost, lam = Tn, rn, Jn, cn, max(lam / 10, 1e-9)
                if np.abs(d).max() < tol:
                    break
            else:
                lam *= 10
        return T, r

    T, r = gauss_newton(T0_BO)
    e = np.sqrt((r.reshape(-1, 4, 2) ** 2).sum(2).mean(1))               # per tag RMS, rows in order
    if e.max() > max_tag_rms and len(e) > 1:
        k = int(e.argmax())
        n0 = 0
        for row in rows:
            if k < n0 + len(row[2]):
                j = k - n0
                del row[2][j]
                row[3] = np.delete(row[3], np.s_[4 * j:4 * j + 4], axis=0)
                break
            n0 += len(row[2])
        rows[:] = [row for row in rows if row[2]]
        T, r = gauss_newton(T)
    r2 = r.reshape(-1, 2)
    return T, {"n_cams": len(rows), "n_corners": len(r2), "rms_px": float(np.sqrt((r2 ** 2).sum(1).mean())),
               "tags": {id(row[0]): list(row[2]) for row in rows}}


def load_map(map_json) -> TagMap:
    m = json.loads(Path(map_json).read_text())
    return TagMap.from_corners(m["family"], m["size_mm"], {int(i): c for i, c in m["corners_mm"].items()})


@dataclass
class Detection:
    sn: str
    frame: int
    t_frame_ms: float               # librealsense global time of the frame
    t_done_ms: float                # host time when detection finished
    corners: dict                   # {id: (4, 2) px}
    mode: str                       # "roi", "full" or "none"
    det_ms: float
    roi: tuple = None               # (x0, y0, x1, y1)


class CameraWorker(threading.Thread):
    def __init__(self, sn, camera: tp.Camera, tag_map: TagMap, on_detection, fps=30, settings=None, nthreads=2,
                 margin_px=40, lost_after=3, predict=None, roi=True):
        super().__init__(daemon=True, name=f"cam-{sn}")
        self.use_roi = roi
        self.on_frame = None                         # optional callable(sn, luma (h, w) uint8 view, Detection)
        self.sn, self.cam, self.map = sn, camera, tag_map
        self.on_detection, self.fps = on_detection, fps
        self.settings = dict(SESSION if settings is None else settings)
        self.margin, self.lost_after = margin_px, lost_after
        self.predict = predict                       # callable(sn) -> (4n, 2) projected corners or None
        self.ids = set(tag_map.ids())
        self.det_full = tp.detector(nthreads=nthreads, quad_decimate=2.0)
        self.det_roi = tp.detector(nthreads=1, quad_decimate=1.0)
        self.stop_evt = threading.Event()
        self.info = {}
        self.last_corners, self.misses = None, 0
        self.error = None

    def _start(self):
        import pyrealsense2 as rs
        self.pipe, cfg = rs.pipeline(), rs.config()
        cfg.enable_device(self.sn)
        cfg.enable_stream(rs.stream.color, self.cam.width, self.cam.height, rs.format.yuyv, self.fps)
        prof = self.pipe.start(cfg)
        rgb = [s for s in prof.get_device().query_sensors() if s.get_info(rs.camera_info.name) == "RGB Camera"][0]
        for o, v in self.settings.items():
            for i in range(10):
                try:
                    rgb.set_option(getattr(rs.option, o), v)
                    break
                except RuntimeError:
                    time.sleep(0.2)
        vp = prof.get_stream(rs.stream.color).as_video_stream_profile()
        it = vp.get_intrinsics()
        self.info = dict(mode=f"{self.cam.width}x{self.cam.height}@{self.fps}",
                         factory=dict(fx=it.fx, fy=it.fy, cx=it.ppx, cy=it.ppy),
                         settings={o: rgb.get_option(getattr(rs.option, o)) for o in self.settings})
        self.rs = rs

    def _window(self):
        if not self.use_roi:
            return None
        pts = None
        if self.last_corners is not None and self.misses == 0:
            pts = self.last_corners
        elif self.predict is not None:
            pts = self.predict(self.sn)
        if pts is None or not len(pts):
            return None
        lo, hi = pts.min(0), pts.max(0)
        m = max(self.margin, 0.5 * float((hi - lo).max()))
        x0, y0 = int(max(lo[0] - m, 0)), int(max(lo[1] - m, 0))
        x1, y1 = int(min(hi[0] + m, self.cam.width)), int(min(hi[1] + m, self.cam.height))
        if x1 - x0 < 16 or y1 - y0 < 16:
            return None
        return x0, y0, x1, y1

    def run(self):
        try:
            self._start()
        except Exception as e:                       # noqa: BLE001 - reported through self.error
            self.error = repr(e)
            return
        rs = self.rs
        try:
            while not self.stop_evt.is_set():
                try:
                    fs = self.pipe.wait_for_frames(2000)
                except RuntimeError:
                    continue
                fr = fs.get_color_frame()
                t_frame = fr.get_timestamp()
                if "ts_domain" not in self.info:
                    md = rs.frame_metadata_value
                    self.info["ts_domain"] = str(fr.get_frame_timestamp_domain()).split(".")[-1]
                    self.info["metadata"] = {k: bool(fr.supports_frame_metadata(getattr(md, k)))
                                             for k in ("frame_timestamp", "sensor_timestamp", "time_of_arrival",
                                                       "backend_timestamp", "actual_exposure")}
                y = np.frombuffer(fr.get_data(), np.uint8).reshape(self.cam.height, self.cam.width * 2)[:, 0::2]
                t0 = time.perf_counter()
                win = self._window()
                found, mode = {}, "none"
                if win is not None:
                    x0, y0, x1, y1 = win
                    found = tp.detect(self.det_roi, np.ascontiguousarray(y[y0:y1, x0:x1]), ids=self.ids)
                    found = {i: c + (x0, y0) for i, c in found.items()}
                    mode = "roi"
                if not found and (win is None or self.misses + 1 >= self.lost_after):
                    found = tp.detect(self.det_full, np.ascontiguousarray(y), ids=self.ids)
                    mode = "full"
                if found:
                    self.last_corners = np.concatenate(list(found.values()))
                    self.misses = 0
                else:
                    self.misses += 1
                det_ms = (time.perf_counter() - t0) * 1000.0
                det = Detection(self.sn, fr.get_frame_number(), t_frame, time.time() * 1000.0, found, mode, det_ms,
                                win)
                if self.on_frame is not None:
                    self.on_frame(self.sn, y, det)
                self.on_detection(det)
        finally:
            self.pipe.stop()

    def stop(self):
        self.stop_evt.set()


class ArmPose(threading.Thread):
    """The newest TCP pose of the UR5 (T_BE, mm) from read-only RTDE, polled at hz."""

    def __init__(self, host="192.168.0.4", hz=125.0):
        super().__init__(daemon=True, name="arm")
        self.host, self.hz = host, hz
        self.T_BE, self.t_ms, self.error = None, None, None
        self.stop_evt = threading.Event()

    def run(self):
        try:
            import rtde_receive
            r = rtde_receive.RTDEReceiveInterface(self.host)
        except Exception as e:                       # noqa: BLE001
            self.error = repr(e)
            return
        try:
            while not self.stop_evt.is_set():
                self.T_BE, self.t_ms = tf_from_ur(r.getActualTCPPose()), time.time() * 1000.0
                time.sleep(1.0 / self.hz)
        finally:
            r.disconnect()

    def stop(self):
        self.stop_evt.set()


@dataclass
class Sample:
    msg: PoseMsg
    T_BO: np.ndarray
    dets: dict = field(default_factory=dict)        # sn -> Detection used
    solve_ms: float = 0.0


class Tracker:
    """Both cameras, the pairing and the solve; start(), then samples arrive at on_sample and the UDP listeners."""

    def __init__(self, cams: dict, tag_map: TagMap, fps=30, udp_to=(), max_skew_ms=None, on_sample=None,
                 arm: ArmPose | None = None, T_EP=None, settings=None, nthreads=2, max_rms_px=2.0, roi=True):
        self.cams, self.map, self.fps = cams, tag_map, fps
        self.sns = sorted(cams)
        self.max_skew = 500.0 / fps + 2.0 if max_skew_ms is None else max_skew_ms
        self.on_sample, self.arm, self.T_EP = on_sample, arm, T_EP
        self.max_rms = max_rms_px
        self.sender = UdpSender(udp_to) if udp_to else None
        self.lock = threading.Condition()
        self.latest = {sn: None for sn in self.sns}
        self.used = {sn: -1 for sn in self.sns}     # frame number of each camera's detection in the last output
        self.T = None                               # last T_BO
        self.T_ms = None
        self.seq = 0
        self.workers = [CameraWorker(sn, cams[sn][0], tag_map, self._on_detection, fps=fps, settings=settings,
                                     nthreads=nthreads, predict=self._predict, roi=roi) for sn in self.sns]
        self.fuser = threading.Thread(target=self._fuse_loop, daemon=True, name="fuse")
        self.stop_evt = threading.Event()

    # projected corners of the body's tags facing camera sn, from the last fused pose (for that camera's window)
    def _predict(self, sn):
        T, t = self.T, self.T_ms
        if T is None or t is None or time.time() * 1000.0 - t > 200.0:
            return None
        cam, T_BC = self.cams[sn]
        T_CO = tf_inv(T_BC) @ T
        ids = [i for i, a in zip(self.map.ids(), tp.view_angles(self.map, self.map.ids(), T_CO)) if a < 80]
        if not ids:
            return None
        return tp.project(cam, T_CO, self.map.stacked(ids))

    def _on_detection(self, d: Detection):
        with self.lock:
            self.latest[d.sn] = d
            self.lock.notify()

    def _ready(self):
        """The detections to fuse now, or None. A new frame of the lead camera is taken with each other camera's newest
        frame within max_skew of it; when that one is older, its next frame is awaited for up to a period."""
        now, P = time.time() * 1000.0, 1000.0 / self.fps
        live = [sn for sn in self.sns if self.latest[sn] is not None and now - self.latest[sn].t_done_ms <= 3 * P]
        if not live:
            return None
        lead = live[0]
        dl = self.latest[lead]
        if dl.frame == self.used[lead]:
            return None
        out = [dl]
        for sn in live[1:]:
            d = self.latest[sn]
            skew = d.t_frame_ms - dl.t_frame_ms
            if abs(skew) <= self.max_skew:
                out.append(d)
            elif skew < 0 and now - dl.t_done_ms < P:
                return None                          # its next frame is due; wait for it
        return out

    def _fuse_loop(self):
        while not self.stop_evt.is_set():
            with self.lock:
                ds = self._ready()
                while ds is None and not self.stop_evt.is_set():
                    self.lock.wait(0.004)
                    ds = self._ready()
                if ds is None:
                    return
                for d in ds:
                    self.used[d.sn] = d.frame
            self._fuse(ds)

    def _fuse(self, ds):
        t0 = time.perf_counter()
        seen = [d for d in ds if d.corners]
        msg = PoseMsg(seq=self.seq, t_capture_ms=float(np.mean([d.t_frame_ms for d in ds])), t_publish_ms=0.0,
                      skew_ms=float(np.ptp([d.t_frame_ms for d in ds])) if len(ds) > 1 else 0.0)
        T, info = None, {}
        if seen:
            idx = [self.sns.index(d.sn) for d in seen]
            cams = [self.cams[self.sns[k]][0] for k in idx]
            T_BC = [self.cams[self.sns[k]][1] for k in idx]
            T0 = self.T
            if T0 is None or self.T_ms is None or time.time() * 1000.0 - self.T_ms > 200.0:
                best = max(seen, key=lambda d: len(d.corners))
                Tc, _ = tp.solve_body(best.corners, self.map, self.cams[best.sn][0])
                T0 = None if Tc is None else self.cams[best.sn][1] @ Tc
            if T0 is not None:
                T, info = solve_joint([d.corners for d in seen], self.map, cams, T_BC, T0)
        ok = T is not None and info.get("n_cams", 0) > 0 and info.get("rms_px", np.inf) <= self.max_rms
        if ok:
            self.T, self.T_ms = T, time.time() * 1000.0
            p, q = tf_to_pose(T)
            msg.p_BO, msg.q_BO = np.asarray(p, float), np.asarray(q, float)
            msg.n_cams, msg.n_tags, msg.rms_px = info["n_cams"], info["n_corners"] // 4, info["rms_px"]
            if info["n_cams"] == 1:
                msg.flags |= SINGLE
            if self.arm is not None and self.arm.T_BE is not None and self.T_EP is not None:
                T_PO = tf_inv(self.arm.T_BE @ self.T_EP) @ T
                p, q = tf_to_pose(T_PO)
                msg.p_PO, msg.q_PO = np.asarray(p, float), np.asarray(q, float)
            else:
                msg.flags |= NO_ARM
        else:
            msg.flags |= LOST
        solve_ms = (time.perf_counter() - t0) * 1000.0
        msg.t_publish_ms = time.time() * 1000.0
        if self.sender is not None:
            self.sender.send(msg.pack())
        self.seq += 1
        if self.on_sample is not None:
            self.on_sample(Sample(msg, T if ok else None, {d.sn: d for d in ds}, solve_ms))

    def start(self):
        for w in self.workers:
            w.start()
        self.fuser.start()
        return self

    def stop(self):
        self.stop_evt.set()
        for w in self.workers:
            w.stop()
        with self.lock:
            self.lock.notify_all()
        for w in self.workers:
            w.join(timeout=3)
        self.fuser.join(timeout=3)
