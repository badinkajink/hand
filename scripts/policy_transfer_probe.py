#!/usr/bin/env python3
"""Zero-shot evaluation of a reorient checkpoint on perturbed copies of its own scene.

    uv run --extra rl --extra gpu python scripts/policy_transfer_probe.py \
        --policy results/rl/<run>/tensorboard/model_270.pt \
        --morphology-run results/phase1/real_v1/20260916-sv1_u0308_b050_cal \
        --variants tipmesh,plate0,mu0.6,mu1.5,mass1.3,kp0.25 --n 64 --out <dir>/<id>_transfer.json

The chain environment differs from the benchmark scene in the tool (a screw-tip mesh on the
-z end), the plate height (0 in every chain scene), the plant (kp re-fits to 0.25 settled) and
whatever the real pads' friction and the tool's mass are. Each variant is one of those changes
applied to the frozen scene alone; the reference trajectory and keyframe stay those the policy
trained with (copied into a variant morphology-run directory), and `policy_eval_suite.py` runs
its usual 64 rollouts with the run's own timing and action clip. Resumable per variant.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

TIP_OBJ = ROOT / "assets/mjcf/experimental/20260904-chain_bothsets/sv1_u0308_b050_screw_a45_x40_y-11_tip.obj"


def _tool_body(root):
    for b in root.iter("body"):
        if b.get("name") == "screwdriver_medium":
            return b
    raise SystemExit("no screwdriver_medium body")


def variant_scene(frozen: Path, name: str, out_dir: Path) -> Path:
    """Write the variant frozen_scene.xml into out_dir and return it."""
    from real_v1_turn_probe import contact_variant
    from real_v1_plan_angle_sweep import plate_variant
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "frozen_scene.xml"
    src = frozen
    if name == "base":
        shutil.copy(frozen, dst); return dst
    m = re.fullmatch(r"mu([\d.]+)", name)
    if m:
        shutil.copy(contact_variant(frozen, None, None, float(m.group(1))), dst); return dst
    m = re.fullmatch(r"tors([\d.]+)", name)
    if m:
        shutil.copy(contact_variant(frozen, None, None, None, torsional=float(m.group(1))), dst); return dst
    if name == "plate0":
        # plate_variant only moves the plate UP from 0; the frozen scene has it at +25, so edit directly
        root = ET.parse(frozen).getroot()
        for body in root.iter("body"):
            if body.get("name") == "palm_pose":
                for g in body.findall("geom"):
                    if (g.get("size") or "").startswith("0.085"):
                        g.set("pos", "0 0 0")
        ET.ElementTree(root).write(dst); return dst
    m = re.fullmatch(r"mass([\d.]+)", name)
    if m:
        root = ET.parse(frozen).getroot()
        for g in _tool_body(root).findall("geom"):
            if g.get("density"):
                g.set("density", f"{float(g.get('density')) * float(m.group(1)):g}")
        ET.ElementTree(root).write(dst); return dst
    if name == "tipmesh":
        root = ET.parse(frozen).getroot()
        asset = root.find("asset")
        if asset is None:
            asset = ET.SubElement(root, "asset")
        ET.SubElement(asset, "mesh", name="screw_tip", file=str(TIP_OBJ))
        tb = _tool_body(root)
        cyl = [g for g in tb.findall("geom") if g.get("type") == "cylinder"][0]
        g = ET.SubElement(tb, "geom", type="mesh", mesh="screw_tip", pos="0 0 -0.05", quat="1 0 0 0",
                          density=cyl.get("density", "500"), friction=cyl.get("friction", "1 0.2 0.02"))
        if cyl.get("material"):
            g.set("material", cyl.get("material"))
        ET.ElementTree(root).write(dst); return dst
    m = re.fullmatch(r"kp([\d.]+)", name)
    if m:
        # position actuators: kp -> value, kv scaled with it (measured kv 0.02 at kp 0.5)
        root = ET.parse(frozen).getroot()
        kp = float(m.group(1))
        # kp/kv live in the `ctrl` class default; explicit attributes on the finger actuators override it
        for a in root.iter("position"):
            if (a.get("joint") or "").split("_")[0] in ("thumb", "index", "middle"):
                a.set("kp", f"{kp:g}"); a.set("kv", f"{0.02 * kp / 0.5:g}")
        ET.ElementTree(root).write(dst); return dst
    raise SystemExit(f"unknown variant {name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", type=Path, required=True)
    ap.add_argument("--morphology-run", type=Path, required=True)
    ap.add_argument("--variants", default="base,tipmesh,plate0,mu0.6,mu1.5,mass1.3,kp0.25")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--work", type=Path, default=None, help="variant morphology-run dirs (default beside --out)")
    ap.add_argument("--cpu", action="store_true", help="CPU MuJoCo rollouts (policy_cpu_rollout.py) with spawn jitter "
                                                        "instead of the GPU eval suite; --n is then the number of seeds")
    ap.add_argument("--jitter-xy", type=float, default=2.0, help="spawn jitter (mm) for --cpu rollouts")
    a = ap.parse_args()
    morph = a.morphology_run.resolve()
    work = a.work or a.out.parent / (a.out.stem + "_variants")
    res = json.load(open(a.out)) if a.out.exists() else {}
    for v in [s.strip() for s in a.variants.split(",") if s.strip()]:
        if v in res and res[v].get("hold_rate") is not None:
            continue
        vd = work / v
        vd.mkdir(parents=True, exist_ok=True)
        for f in ("summary.json", "best_rollout.npz"):
            if not (vd / f).exists():
                shutil.copy(morph / f, vd / f)
        variant_scene(morph / "frozen_scene.xml", v, vd)
        # summary.json's frozen_scene_xml must point at the variant scene
        s = json.load(open(vd / "summary.json")); s["frozen_scene_xml"] = str(vd / "frozen_scene.xml")
        json.dump(s, open(vd / "summary.json", "w"), indent=1)
        if a.cpu:
            outs = []
            for seed in range(1, a.n + 1):
                oj = vd / f"cpu_s{seed}.json"
                cmd = [sys.executable, str(ROOT / "scripts/policy_cpu_rollout.py"), "--policy", str(a.policy),
                       "--morphology-run", str(morph), "--scene", str(vd / "frozen_scene.xml"), "--steps", str(a.steps),
                       "--jitter-xy", str(a.jitter_xy), "--seed", str(seed), "--out", str(oj)]
                p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
                if oj.exists():
                    outs.append(json.load(open(oj)))
            if outs:
                fc = [o["final_cos"] for o in outs]
                res[v] = {"hold_rate": sum(o["held_end"] for o in outs) / len(outs),
                          "align_rate": sum(o["peak_cos"] >= 0.9 for o in outs) / len(outs),
                          "final_cos_mean": float(sum(fc) / len(fc)),
                          "final_cos_sd": float((sum((x - sum(fc) / len(fc)) ** 2 for x in fc) / len(fc)) ** 0.5),
                          "t_align_mean": None, "n_lost": sum(not o["held_end"] for o in outs), "n": len(outs),
                          "peak_cos_mean": float(sum(o["peak_cos"] for o in outs) / len(outs)),
                          "clearance_min_mm_mean": float(sum(o["clearance_min_mm"] for o in outs) / len(outs)),
                          "force_active_thumb": float(sum(o["force_active"][0] for o in outs) / len(outs)),
                          "force_active_index": float(sum(o["force_active"][1] for o in outs) / len(outs)),
                          "force_active_middle": float(sum(o["force_active"][2] for o in outs) / len(outs)),
                          "backend": "cpu", "jitter_xy_mm": a.jitter_xy}
                print(f"{v:10s} cpu hold {res[v]['hold_rate']:.2f} align {res[v]['align_rate']:.2f} cos {res[v]['final_cos_mean']:+.3f}", flush=True)
            else:
                res[v] = {"error": (p.stdout + p.stderr)[-600:]}
            tmp = a.out.with_suffix(".tmp")
            with open(tmp, "w") as fh:
                json.dump(res, fh, indent=1); fh.flush(); os.fsync(fh.fileno())
            os.replace(tmp, a.out)
            continue
        ev_json = vd / "eval.json"
        cmd = [sys.executable, str(ROOT / "scripts/policy_eval_suite.py"), "--policy", str(a.policy),
               "--morphology-run", str(vd), "--closed-ctrl-from-keyframe", "open_ik", "--open-finger-from-keyframe",
               "--lift-delta", "0.1", "--steps", str(a.steps), "--n", str(a.n), "--held-min-n", "0.5", "--floor-z", "0.06",
               "--json-out", str(ev_json), "--label", f"{a.policy.parent.parent.name}__{v}"]
        env = dict(os.environ, WARP_CACHE_PATH=subprocess.check_output(["mktemp", "-d"], text=True).strip(), MUJOCO_GL="egl")
        p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
        if ev_json.exists():
            e = json.load(open(ev_json))
            res[v] = {k: e.get(k) for k in ("hold_rate", "align_rate", "final_cos_mean", "final_cos_sd", "t_align_mean", "n_lost",
                                             "peak_cos_mean", "clearance_min_mm_mean", "pad_peak_thumb", "pad_peak_index",
                                             "pad_peak_middle", "ctrl_gap_max_deg", "force_active_thumb", "force_active_index",
                                             "force_active_middle")}
            print(f"{v:10s} hold {res[v]['hold_rate']:.2f} align {res[v]['align_rate']:.2f} cos {res[v]['final_cos_mean']:+.3f}", flush=True)
        else:
            res[v] = {"error": (p.stdout + p.stderr)[-600:]}
            print(f"{v:10s} FAILED: {res[v]['error'][-200:]}", flush=True)
        tmp = a.out.with_suffix(".tmp")
        with open(tmp, "w") as fh:
            json.dump(res, fh, indent=1); fh.flush(); os.fsync(fh.fileno())
        os.replace(tmp, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
