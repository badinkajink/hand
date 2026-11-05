"""Each listed solid alone, three views, to identify the small parts."""
import sys
sys.path.insert(0, '/home/humanoid/Programs/hand/logs/20261104-cad_v3/parts')
import numpy as np, mujoco
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from render_lib import BY_KEY, build_model
keys = sys.argv[2].split(',')
out = sys.argv[1]
W = H = 500
VIEWS = [(30, -25), (120, -25), (210, 35)]
fig, axs = plt.subplots(len(keys), len(VIEWS), figsize=(len(VIEWS) * 3.2, len(keys) * 3.4), dpi=110)
axs = np.atleast_2d(axs)
for r, k in enumerate(keys):
    p = BY_KEY[k]
    m, d = build_model([dict(key=k, rgba=(0.55, 0.65, 0.85, 1))], W, H)
    b = np.array(p['bbox']); ctr = (b[:3] + b[3:]) / 2000; ext = np.linalg.norm(b[3:] - b[:3]) / 1000
    with mujoco.Renderer(m, H, W) as ren:
        for c, (az, el) in enumerate(VIEWS):
            cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = ctr; cam.distance = ext * 2.2; cam.azimuth = az; cam.elevation = el
            ren.update_scene(d, camera=cam)
            axs[r, c].imshow(ren.render()); axs[r, c].axis('off')
    dims = b[3:] - b[:3]
    axs[r, 0].set_title(f"{k}  vol {p['volume']:.0f} mm3  bbox {dims[0]:.1f} x {dims[1]:.1f} x {dims[2]:.1f}", fontsize=8, loc='left')
fig.tight_layout()
fig.savefig(out)
print('wrote', out)
