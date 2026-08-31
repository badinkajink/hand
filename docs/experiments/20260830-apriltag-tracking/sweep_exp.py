import numpy as np, pyrealsense2 as rs
from pupil_apriltags import Detector
pipe,cfg = rs.pipeline(), rs.config()
cfg.enable_stream(rs.stream.infrared,1,1280,720,rs.format.y8,30)
prof = pipe.start(cfg); pipe.wait_for_frames(8000)
sens = [s for s in prof.get_device().query_sensors() if s.supports(rs.option.emitter_enabled)][0]
sens.set_option(rs.option.emitter_enabled, 0)
sens.set_option(rs.option.enable_auto_exposure, 0)
I = prof.get_stream(rs.stream.infrared,1).as_video_stream_profile().get_intrinsics()
cam=(I.fx,I.fy,I.ppx,I.ppy); det = Detector(families="tag36h11", nthreads=4, quad_decimate=1.0)
print(f"{'exp us':>7} {'gain':>5} {'mean':>6} {'sat%':>6} {'id0 margin':>11} {'id6 margin':>11}  smear px")
for exp in (4000, 6000, 8500, 12000):
    for g in (16, 32, 64):
        sens.set_option(rs.option.exposure, exp); sens.set_option(rs.option.gain, g)
        for _ in range(8): fs = pipe.wait_for_frames(8000)
        img = np.asanyarray(fs.get_infrared_frame().get_data())
        m = {t.tag_id: t.decision_margin for t in det.detect(img)}
        sat = 100.0*(img>=254).mean()
        smear = 73.0*(exp*1e-6)*I.fx/292.0   # 73 mm/s shaft speed over the 2 s turn
        print(f"{exp:7d} {g:5d} {img.mean():6.1f} {sat:6.2f} "
              f"{m.get(0,float('nan')):11.1f} {m.get(6,float('nan')):11.1f}  {smear:7.2f}")
pipe.stop()
