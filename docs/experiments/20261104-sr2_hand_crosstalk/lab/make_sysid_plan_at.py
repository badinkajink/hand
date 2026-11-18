"""make_sysid_plan.py's plans and clearance check at another gantry layout (2026-11-17, the identification at 7.5 V,
docs/experiments/20261117-sr2_system_identification/QUEUE.md S2): the blocks of 11-06 unchanged, every step checked in
the CAD hand at MOUNTS (palm mm) instead of the home layout and the middle's four corners. The runner's measured-pose
gate (run_plan.py --min-clearance) checks the station's own mounts at run time as before.
    uv run --no-project --python 3.10 --with mujoco --with numpy --with scipy python make_sysid_plan_at.py OUT_DIR \
        --mounts=-28.8188,2.4004,28.8188,51.6003,28.8187,-51.6002
"""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import make_sysid_plan as M

arg = [x for x in sys.argv if x.startswith("--mounts=")]
v = [float(x) for x in arg[0].split("=", 1)[1].split(",")]
MOUNTS = {"thumb": (v[0], v[1]), "index": (v[2], v[3]), "middle": (v[4], v[5])}
KEY = "home (80, -85)"                       # make_sysid_plan's names; the place is MOUNTS
M.HOME = dict(MOUNTS)
M.MIDDLE = {KEY: MOUNTS["middle"]}

if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for name, steps in M.BLOCKS.items():
        assert all(M.in_range(s["joints"]) for s in steps), name
    with Pool(1) as pool:
        per = dict(pool.map(M.check_middle, list(M.MIDDLE)))
    print(f"checked in {time.time() - t0:.0f} s at {MOUNTS}", flush=True)
    summary = {}
    for name, steps in M.BLOCKS.items():
        for i, s in enumerate(steps):
            s["clearance_mm"] = round(min(per[KEY][name][i]["at"].values()), 1)
            s["clearance_box_mm"] = round(min(c for c, _ in per[KEY][name][i]["box"].values()), 1)
        lim = M.LIMIT[name[0]]
        json.dump({"block": name, "made_by": "lab/make_sysid_plan_at.py", "layout": "mounts", "mounts_palm_mm": MOUNTS,
                   "scene": "assets/mjcf/experimental/20261030-cad_v2/scene_screwdriver_medium.xml",
                   "runner": {"ramp_deg": 0, "min_clearance": 5, "max_blocked": 1, "keep_home": True,
                              "max_err_yaw": lim["yaw"], "max_err": lim["mcp"]},
                   "steps": steps}, open(out / f"20261106-sysid_{name}.json", "w"), indent=1)
        at = min(s["clearance_mm"] for s in steps)
        box = min(s["clearance_box_mm"] for s in steps)
        worst = min(steps, key=lambda s: s["clearance_box_mm"])
        dur = sum(s["dwell_s"] + (1.3 if s["dwell_s"] >= 10 else 0.5) for s in steps)
        summary[name] = {"steps": len(steps), "minutes": round(dur / 60, 1), "least_at_pose_mm": at,
                         "least_in_box_mm": box, "worst_family": worst["family"], "worst_cmd": worst["cmd"]}
        print(f"{name}: {len(steps)} steps, {dur / 60:.1f} min; least clearance {at:.1f} mm at a pose, {box:.1f} mm in "
              f"the move boxes ({worst['family']} {worst['cmd']}); margin {M.MIN_MM:g}", flush=True)
    json.dump({"made": time.strftime("%Y-%m-%d %H:%M:%S"), "mounts_palm_mm": MOUNTS, "margin_mm": M.MIN_MM,
               "blocks": summary}, open(out / "20261117-sysid_clearance_at_mounts.json", "w"), indent=1)
