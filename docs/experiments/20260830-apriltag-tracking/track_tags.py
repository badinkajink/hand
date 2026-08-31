#!/usr/bin/env python3
"""Track the cylinder's reorientation on the bench with two AprilTags.

  id 0  (30 mm) on the cylinder  -- its plane must CONTAIN the shaft axis
  id 6  (40 mm) static, vertical -- supplies the world-up datum

Reports cos(shaft axis, up), directly comparable to the sim's held-cos.
Run --probe first to check mounting signs before trusting a run.
"""
import argparse, csv, sys, time
import numpy as np, cv2, pyrealsense2 as rs
from pupil_apriltags import Detector

CYL_ID, REF_ID = 0, 6
CYL_SIZE, REF_SIZE = 0.030, 0.040          # metres, = the printed tag_size

def axis_from(flag):
    return {"x": np.array([1.,0,0]), "-x": np.array([-1.,0,0]),
            "y": np.array([0,1.,0]), "-y": np.array([0,-1.,0]),
            "z": np.array([0,0,1.]), "-z": np.array([0,0,-1.])}[flag]

class Live:
    """In-place status line on a terminal, plain throttled lines when redirected.
    Events are printed ABOVE the live line so a dropout survives in scrollback."""
    def __init__(self, enabled, hz):
        self.on = enabled
        self.tty = sys.stdout.isatty()
        self.every = (1.0/hz if hz > 0 else 0.0) if self.tty else max(1.0/max(hz,1e-9), 0.5)
        self.last, self.width = 0.0, 0
    def event(self, msg):
        if self.on and self.tty and self.width:
            sys.stdout.write("\r" + " "*self.width + "\r"); self.width = 0
        print(msg, flush=True)
    def update(self, line, force=False):
        if not self.on: return
        now = time.time()
        if not force and now - self.last < self.every: return
        self.last = now
        if self.tty:
            sys.stdout.write("\r" + line + " "*max(0, self.width-len(line)))
            sys.stdout.flush(); self.width = len(line)
        else:
            print(line, flush=True)
    def close(self):
        if self.on and self.tty and self.width:
            sys.stdout.write("\n"); sys.stdout.flush()
        self.width = 0

def deg_from_up(c):
    """Angle from UP, unfolded to [0,180]: 0 = vertical up, 90 = horizontal,
    180 = vertical down.  Folding with abs() would make a wrong-pole turn
    indistinguishable from a correct one."""
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))

def meter(v, n=13):
    """Position of cos on [-1,+1]; ':' marks horizontal, right is up."""
    pos = int(round((np.clip(v,-1.0,1.0)+1.0)/2.0*(n-1)))
    cells = ["."]*n
    cells[(n-1)//2] = ":"
    cells[pos] = "|"
    return "[" + "".join(cells) + "]"

def open_ir(w, h, fps, warmup=10, exposure=None, gain=None):
    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_stream(rs.stream.infrared, 1, w, h, rs.format.y8, fps)
    prof = pipe.start(cfg)
    # The emitter write MUST come after frames are flowing.  Issuing it between
    # start() and the first frame wedges the stream and wait_for_frames times out.
    pipe.wait_for_frames(8000)
    for s in prof.get_device().query_sensors():
        if s.supports(rs.option.emitter_enabled):
            s.set_option(rs.option.emitter_enabled, 0)   # dots would wreck the decode
    if exposure is not None:
        # Auto-exposure hunts as the shaft swings and varies the motion smear
        # frame to frame.  Pin it for anything you intend to measure.
        for s in prof.get_device().query_sensors():
            if s.supports(rs.option.enable_auto_exposure):
                s.set_option(rs.option.enable_auto_exposure, 0)
                s.set_option(rs.option.exposure, exposure)
                if gain is not None: s.set_option(rs.option.gain, gain)
    for _ in range(warmup):                              # let AE settle
        pipe.wait_for_frames(8000)
    I = prof.get_stream(rs.stream.infrared, 1).as_video_stream_profile().get_intrinsics()
    return pipe, (I.fx, I.fy, I.ppx, I.ppy), I

def detect(det, img, cam):
    out = {}
    for tid, size in ((CYL_ID, CYL_SIZE), (REF_ID, REF_SIZE)):
        for t in det.detect(img, estimate_tag_pose=True, camera_params=cam, tag_size=size):
            if t.tag_id == tid:
                out[tid] = t
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--width", type=int, default=1280); p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30); p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--shaft-axis", default="x", help="shaft direction in the CYLINDER tag frame")
    p.add_argument("--up-axis",    default="x",
                   help="world up in the REFERENCE tag frame "
                        "(x = measured on this rig 2026-08-31; the tag is mounted rotated 90 deg)")
    # 4000/64 measured on this rig 2026-08-31: best margins on BOTH tags and the
    # least motion smear.  AE had picked 8500/16, where the reference tag vanishes.
    p.add_argument("--exposure", type=float, default=4000.0, help="us; pins AE off. 0 = leave auto")
    p.add_argument("--gain", type=float, default=64.0)
    p.add_argument("--out", default="turn_trace.csv")
    p.add_argument("--latch-frames", type=int, default=20,
                   help="frames averaged to latch the static reference before the run")
    p.add_argument("--drift-deg", type=float, default=2.0,
                   help="warn if the latched reference moves more than this (camera bumped)")
    p.add_argument("--probe", action="store_true", help="one frame, print geometry, exit")
    p.add_argument("--quiet", action="store_true", help="suppress the live readout")
    p.add_argument("--live-hz", type=float, default=12.0, help="live line refresh rate")
    a = p.parse_args()

    pipe, cam, I = open_ir(a.width, a.height, a.fps,
                           exposure=(a.exposure if a.exposure else None), gain=a.gain)
    print(f"IR1 {I.width}x{I.height} fx={I.fx:.1f} ppx={I.ppx:.1f} "
          f"HFOV={2*np.degrees(np.arctan(I.width/(2*I.fx))):.1f} deg  emitter OFF")
    det = Detector(families="tag36h11", nthreads=4, quad_decimate=1.0)
    shaft_t, up_r = axis_from(a.shaft_axis), axis_from(a.up_axis)

    def latch_up(n):
        """The reference tag never moves, so read it ONCE up front and reuse it.
        Depending on it every frame makes a shadowed reference a single point of
        failure for the whole run."""
        got = []
        for _ in range(n):
            img = np.asanyarray(pipe.wait_for_frames(8000).get_infrared_frame().get_data())
            tags = detect(det, img, cam)
            if REF_ID in tags: got.append(tags[REF_ID].pose_R @ up_r)
        if not got:
            raise SystemExit(f"reference id{REF_ID} never seen in {n} frames - "
                             "light it better or re-aim before running")
        v = np.mean(got, axis=0); v /= np.linalg.norm(v)
        print(f"latched reference from {len(got)}/{n} frames "
              f"({100*len(got)/n:.0f}% visibility)")
        if len(got) < n:
            print("  note: intermittent reference. Latching covers it, but light it if you re-aim.")
        return v

    try:
        if a.probe:
            img = np.asanyarray(pipe.wait_for_frames(5000).get_infrared_frame().get_data())
            cv2.imwrite("probe_ir.png", img)
            tags = detect(det, img, cam)
            print(f"exposure looks {'OK' if 20 < img.mean() < 235 else 'BAD'} (mean {img.mean():.0f})")
            for tid, t in sorted(tags.items()):
                e = np.mean([np.linalg.norm(t.corners[k]-t.corners[(k+1)%4]) for k in range(4)])
                print(f"  id{tid}: margin {t.decision_margin:5.1f}  {e:5.1f}px edge = {e/8:.1f} px/cell  "
                      f"range {np.linalg.norm(t.pose_t)*1000:6.1f} mm")
            for tid in (CYL_ID, REF_ID):
                if tid not in tags: print(f"  id{tid}: NOT SEEN")
            if len(tags) == 2:
                up = tags[REF_ID].pose_R @ up_r
                sh = tags[CYL_ID].pose_R @ shaft_t
                cc = float(np.dot(sh, up))
                print(f"  cos(shaft, up) = {cc:+.4f}  ({deg_from_up(cc):.1f} deg from up, "
                      f"{abs(90.0-deg_from_up(cc)):.1f} deg off horizontal)")
                print("  -> lay the shaft flat: expect cos ~0.  Stand it up: expect cos ~+1.")
            print("wrote probe_ir.png")
            return

        up_latched = latch_up(a.latch_frames)
        live = Live(not a.quiet, a.live_hz)
        if not a.quiet:
            live.event(f"recording {a.seconds:.0f}s -- cos +1 = up, 0 = horizontal, -1 = down (wrong pole)."
                       f"  Ctrl-C stops early and still writes {a.out}")
        rows, t0, drift_warned = [], time.time(), False
        peak, worst, lost_since, stamps = -1.0, 1.0, None, []
        try:
            while time.time() - t0 < a.seconds:
                img = np.asanyarray(pipe.wait_for_frames(5000).get_infrared_frame().get_data())
                tags = detect(det, img, cam)
                t = time.time() - t0
                stamps.append(time.time())
                if len(stamps) > 15: stamps.pop(0)
                fps = (len(stamps)-1)/(stamps[-1]-stamps[0]) if len(stamps) > 1 else 0.0
                if REF_ID in tags and not drift_warned:      # free camera-bump check
                    d = np.degrees(np.arccos(np.clip(np.dot(tags[REF_ID].pose_R @ up_r, up_latched),-1,1)))
                    if d > a.drift_deg:
                        live.event(f"  [{t:6.2f}s] WARNING: reference moved {d:.1f} deg - camera bumped, "
                                   "angles after this point are referenced to a stale datum")
                        drift_warned = True
                if CYL_ID in tags:
                    if lost_since is not None:
                        live.event(f"  [{t:6.2f}s] cylinder tag REACQUIRED after {t-lost_since:.2f}s")
                        lost_since = None
                    up = up_latched
                    sh = tags[CYL_ID].pose_R @ shaft_t
                    d  = tags[CYL_ID].pose_t.flatten() * 1000
                    c  = float(np.dot(sh, up))
                    deg = deg_from_up(c)
                    margin = tags[CYL_ID].decision_margin
                    peak = max(peak, c)      # SIGNED: a wrong-pole turn must not score
                    worst = min(worst, c)
                    rows.append(dict(t=round(t,4), cos=round(c,5), deg=round(deg,3),
                                     x=round(d[0],2), y=round(d[1],2), z=round(d[2],2),
                                     margin=round(margin,1)))
                    live.update(f"{t:6.2f}s  cos {c:+.3f} {meter(c)} {deg:5.1f}deg from up"
                                f"   peak {peak:+.3f}  {fps:4.1f}fps  margin {margin:3.0f}")
                else:
                    if lost_since is None:
                        lost_since = t
                        live.event(f"  [{t:6.2f}s] cylinder tag LOST")
                    rows.append(dict(t=round(t,4), cos="", deg="", x="", y="", z="", margin=""))
                    live.update(f"{t:6.2f}s  -- TAG LOST for {t-lost_since:4.2f}s --"
                                f"                        {fps:4.1f}fps")
        except KeyboardInterrupt:
            live.close()
            print(f"interrupted at {time.time()-t0:.2f}s - writing what was captured")
        live.close()
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["t","cos","deg","x","y","z","margin"]); w.writeheader(); w.writerows(rows)
        seen = [r for r in rows if r["cos"] != ""]
        print(f"{len(rows)} frames, cylinder tag in {len(seen)} "
              f"({100*len(seen)/max(len(rows),1):.0f}%) -> {a.out}")
        if seen:
            c = [r["cos"] for r in seen]
            print(f"  cos start {c[0]:+.3f}  peak {max(c):+.3f}  final {c[-1]:+.3f}   "
                  f"({deg_from_up(c[0]):.1f} -> {deg_from_up(c[-1]):.1f} deg from up)")
            if min(c) < -0.15:
                print(f"  NOTE: cos reached {min(c):+.3f} - the shaft swung past horizontal to the "
                      "WRONG pole. Signed peak already excludes it; don't read it as progress.")
            if len(seen) < 0.9*len(rows):
                print("  WARNING: dropouts. A gap during the turn is a hole in the trace, not a slip.")
    finally:
        pipe.stop()

if __name__ == "__main__":
    sys.exit(main())
