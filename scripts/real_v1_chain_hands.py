"""The chain on both deployed hand sets, with the grip loop closed through the whole run.

TWO SETS, AND BOTH MATTER. Set A is the eight hands the user bench-tested for reorientation
(`docs/experiments/20260901-real_v1-transfer-firstpass/data.json`, plans in
`20260829-real_v1_deploy/deploy/`); it is the only set with hardware measurements. Set B is the
eight promoted by the Sobol-8192 screen (`20260831-real_v1-sobol8192/deploy/promotion.json`).
Only `sv1_w6689` is in both. Every hand runs at ITS OWN operating point -- straddle, thumb-axial
offset, fitted grip depth, pivot height, turn angle and residual clip, from its own plan.

THE MANEUVER IS THE CHAIN, NOT THE BENCH CARRY. Both sets' plans were screened on the fixed-palm
bench carry: tool standing on a 100 mm platform, palm fixed. This runs the chain instead -- the
tool lying on a table, picked up by a UR5e, reoriented, set into a countersink, gaited -- so the
numbers here are NOT comparable to `nom_cos` or to the bench trials. What carries across is the
hand and its grasp, not the schedule.

WHY THE LOOP IS CLOSED. The reorientation's settle to vertical is worth +30 deg on one ranked
hand and -17 on another from an identical command, so it cannot be built on. `--load-target`
runs `real_v1_deploy_envelope._load_step` through every phase of the chain: it gives up the
settle and buys a grasp that does not fail, which is what a harder reorient->gait pose
trajectory needs.

    uv run --extra rl --extra arm python scripts/real_v1_chain_hands.py --out <dir>
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

BASES = [ROOT / "assets/mjcf/experimental/20260830-real_v1-sobol4096",
         ROOT / "assets/mjcf/experimental/20260830-real_v1-sobol128"]
SET_A = ROOT / "docs/experiments/20260829-real_v1_deploy/deploy"
SET_B = ROOT / "docs/experiments/20260831-real_v1-sobol8192/deploy"
A_TAGS = ["sv1_w6689_b060", "sv1_w2360_b075", "sv1_u1364_b080", "g12_b095",
          "sv1_u0060_b75", "sv1_u0308_b050", "rv05_manual_b85", "sv1_w0099_b100"]
# Own directory: `20260904-chain_hands` is written by another study and a shared
# cache path silently returned records with half the fields.
SCENES = ROOT / "assets/mjcf/experimental/20260904-chain_bothsets"
PAD = dict(pad_len=0.015, width=0.0211, links=False)
# THE PLAN'S SQUEEZE IS A BENCH NUMBER AND IT MUST NOT BE TRANSPLANTED. Every deployed plan
# carries `squeeze_mm` 10.0, fitted on a scene where the shaft floats at 100 mm on a post that
# takes the reaction. It drives the pad CENTRES to r_obj + r_pad + gap - squeeze = 14.05 mm from
# the shaft axis when contact needs 23.05 -- 9 mm inside the surface. On a shaft lying on a
# table there is nothing behind it, so the pinch ejects it: measured at closure with the palm
# stationary, the tool rises +14 to +20 mm on nine of sixteen hands (its own radius is 12.5),
# rides up onto the pip frames, and the pad force FALLS as the squeeze rises (g12 10.4 -> 0.5 N
# over 2 -> 10 mm) because the shaft has escaped over the top of the pads. Three hands go the
# other way and crush it into the floor at 28-67 N. It is the wedge-sign failure
# `fit_real_v1_pose.tip_targets` warns about for positive elevation, arriving through excess
# squeeze instead of elevation.
#
# The fitter's own hold probe does not catch it, and cannot: `held` is measured against the
# object's height BEFORE the close, so the pop counts as retained lift. held ~= lift_probe + dz
# is the identity, and every grasp that "passes" the 20 mm gate at 65-75 mm on a 50 mm probe is
# passing by the amount it threw the tool. The tell is `held > lift_probe`, and nothing was
# looking at it.
SQUEEZE_MM = 2.0
STEM = "screw_a45_x40_y-11"

# The published chain cell (docs/experiments/20260903-real_v1_chain + the seat's carry). Only the
# per-hand fields below are overridden; the schedule itself is the same for every hand.
BASE = dict(obj="screwdriver_medium", lift=0.10, gap=0.002, descend_iters=1, descend_steps=400,
            airgrip="cradle", stand_order="ground", centre_x=0.004, squeeze=0.002,
            release_mm=6.0, twist_steps=120, move_steps=60, carry_squeeze=0.0003,
            press_mm=10.0, transport_steps=300, reindex="relay", relay_gait=True,
            turn_steps=550, hold_steps=300)


def hands(sets: str) -> list[dict]:
    out = []
    if "A" in sets:
        for tag in A_TAGS:
            f = SET_A / f"{tag}_plan.json"
            if f.exists():
                out.append(_rec(tag, json.loads(f.read_text())["meta"], "A"))
    if "B" in sets:
        for r in json.loads((SET_B / "promotion.json").read_text()):
            f = SET_B / f"{r['tag']}_plan.json"
            if f.exists():
                out.append(_rec(r["tag"], json.loads(f.read_text())["meta"], "B"))
    return [h for h in out if h is not None]


def _rec(tag: str, m: dict, which: str) -> dict | None:
    stem = Path(m["scene"]).stem.split("__")[0]
    base = next((d / f"{stem}.xml" for d in BASES if (d / f"{stem}.xml").exists()), None)
    if base is None:
        return None
    return {"tag": tag, "set": which, "base": str(base),
            "straddle": m["straddle_mm"] / 1000, "thumb_axial": m["thumb_axial_mm"] / 1000,
            "squeeze": SQUEEZE_MM / 1000, "plan_squeeze_mm": m["squeeze_mm"],
            "depth": m["grip_depth_mm"] / 1000,
            "axis_k": m["axis_k"], "angle_deg": m["angle_deg"], "budget": m["budget_rad"]}


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)


def prepare(h: dict, yaw_deg: float = 0.0, squeeze_mm: float | None = None) -> dict | None:
    """Design scene -> fitted grasp baked into `open_ik` -> countersink -> UR5e arm scene.

    The chain resets from the scene's `open_ik` keyframe and takes its finger anchor separately,
    so the fit has to land in BOTH: the open pose as the keyframe, the closed grip as
    `anchor_ctrl`. Cached per hand -- the fit is ~90% of the cost and does not depend on the
    schedule.
    """
    import mujoco
    import probe_real_v1_carry as pc
    from morphohand.studies.scene_mutate import Scene
    from morphohand.tools.keyframe_ik import FINGERS
    import math
    SCENES.mkdir(parents=True, exist_ok=True)
    # WHICH WAY THE TOOL LIES ON THE BENCH, and it is a spec rather than a detail: standing it
    # up is a rotation the grasp can carry in one direction and not the other, so the heading
    # the tool is laid down in decides whether the tool ends up standing or on the floor.
    sq = SQUEEZE_MM if squeeze_mm is None else float(squeeze_mm)
    tag = h["tag"] if not yaw_deg else f"{h['tag']}_y{int(round(yaw_deg))}"
    if sq != SQUEEZE_MM:
        tag += f"_sq{sq:g}"
    meta = SCENES / f"{tag}_fit.json"
    arm, ik = SCENES / f"{tag}_arm.xml", SCENES / f"{tag}_arm_ik.xml"
    NEED = ("anchor", "depth_mm", "arm", "ik", "place_xy", "seat_z", "tip_len",
            "close_dz_mm", "squeeze_mm")
    if meta.exists() and arm.exists() and ik.exists():
        rec = json.loads(meta.read_text())
        if all(k in rec for k in NEED) and rec.get("squeeze_mm") == sq:
            return rec

    flat = SCENES / f"{tag}__flat.xml"
    if not flat.exists():
        sc = Scene(Path(h["base"]))
        sc.set_finger_flat_pads(pad_len=PAD["pad_len"], width=PAD["width"], links=PAD["links"])
        sc.write(flat)
        if yaw_deg:
            root = ET.parse(flat).getroot()
            b = next(x for x in root.iter("body") if x.get("name") == BASE["obj"])
            q = tuple(float(v) for v in (b.get("quat") or "1 0 0 0").split())
            a = math.radians(yaw_deg) / 2.0
            b.set("quat", " ".join(f"{v:.9g}" for v in
                                   _qmul((math.cos(a), 0.0, 0.0, math.sin(a)), q)))
            ET.ElementTree(root).write(flat, encoding="unicode")
    built = pc._grip_from_fit(flat, h["straddle"], 0.0, sq / 1000.0, BASE["obj"],
                              h["depth"], h["thumb_axial"])
    if built is None:
        return None
    m, open_qpos, grip, depth_mm = built
    acts = {j: next(k for k in range(m.nu) if m.actuator(k).name == f"a_{j}")
            for js in FINGERS.values() for j in js}
    root = ET.parse(flat).getroot()
    for k in root.findall("keyframe"):
        root.remove(k)
    kf = ET.SubElement(root, "keyframe")
    q = " ".join(f"{v:.9g}" for v in open_qpos)
    ET.SubElement(kf, "key", {"name": "open_ik", "qpos": q,
                              "ctrl": " ".join(f"{v:.9g}" for v in grip)})
    ET.SubElement(kf, "key", {"name": "open", "qpos": q})
    ET.ElementTree(root).write(flat, encoding="unicode")

    seat = SCENES / f"{tag}_{STEM}.xml"
    p = subprocess.run([sys.executable, str(ROOT / "scripts/build_screw_scene.py"),
                        "--scene", str(flat), "--half-angle", "45",
                        "--socket-xy", "0.04,-0.011", "--out", str(seat),
                        "--out-json", str(seat.with_suffix(".json"))],
                       capture_output=True, text=True)
    if p.returncode != 0:
        return None
    p = subprocess.run([sys.executable, str(ROOT / "scripts/build_real_v1_arm_scene.py"),
                        "--scene", str(seat), "--base=-0.50,0,0", "--wrist-stack", "0.10",
                        "--payload-gravcomp", "--out", str(arm), "--ik-out", str(ik)],
                       capture_output=True, text=True)
    if p.returncode != 0:
        return None
    # WHAT THE CLOSE DOES TO A TOOL THAT IS LYING ON THE TABLE, before the palm moves at all.
    # `close_dz_mm` near zero is a grasp; +15 is the tool being thrown out of the pinch, and
    # `close_nonpad` counts contacts on anything that is not a pad -- the pip frames the tool
    # rides up onto, or the floor it is being crushed into.
    dz, npad, nonpad, fpad = _close_probe(m, open_qpos, grip, BASE["obj"])
    sj = json.loads(seat.with_suffix(".json").read_text())
    out = {"tag": tag, "close_dz_mm": dz, "close_pads": npad, "close_nonpad": nonpad,
           "close_pad_N": fpad, "squeeze_mm": sq,
           "anchor": {j: float(grip[a]) for j, a in acts.items()},
           "depth_mm": depth_mm, "arm": str(arm), "ik": str(ik),
           "place_xy": sj["socket_xy"], "seat_z": sj["seat_z"], "tip_len": sj["tip_len"]}
    meta.write_text(json.dumps(out))
    return out


# THE TABLE-STAND MANEUVER. The fingers do not turn the tool at all (`angle_deg` 0) and the
# tool is never in free flight: it is picked up, put back down on the table lying, and stood up
# by pivoting about its own foot with the table carrying its weight. That ordering is forced by
# statics -- the grasp holds 0.4 N on a 24 g tool in mid-air, and a 90 deg mid-air reorientation
# drops it whether the fingers or the arm do the rotating, on the UR5e and on a floating palm
# alike. Standing on a plane means standing HANDLE DOWN, so this mode has no countersink: `tip`
# geometry is switched off and the seat is not used.
TABLE = dict(obj="screwdriver_medium", lift=0.10, gap=0.002, angle_deg=0.0,
             descend_iters=1, descend_steps=400, airgrip="cradle", stand_order="ground",
             repose_iters=8, repose_steps=800, centre_x=0.004, squeeze=0.002,
             release_mm=6.0, twist_steps=120, move_steps=60, carry_squeeze=0.0003,
             press_mm=10.0, transport_steps=300, reindex="full", relay_gait=False,
             ring_az="pads", turn_steps=550, hold_steps=300)


def _close_probe(m, open_qpos, grip, obj: str):
    """Hold the fitted grip for 0.8 s with the palm still, and see where the tool goes."""
    import mujoco
    import numpy as np
    from morphohand.tools.keyframe_ik import FINGERS
    d = mujoco.MjData(m)
    d.qpos[:] = open_qpos
    d.ctrl[:] = grip
    mujoco.mj_forward(m, d)
    z0 = float(d.body(obj).xpos[2])
    for _ in range(400):
        mujoco.mj_step(m, d)
    bid = m.body(obj).id
    tips = {m.body(f"{f}_tip").id for f in FINGERS}
    npad = nonpad = 0
    fpad = 0.0
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if bid not in (b1, b2):
            continue
        other = b2 if b1 == bid else b1
        f6 = np.zeros(6)
        mujoco.mj_contactForce(m, d, i, f6)
        if other in tips:
            npad += 1
            fpad += float(abs(f6[0]))
        else:
            nonpad += 1
    return (round((float(d.body(obj).xpos[2]) - z0) * 1000, 2), npad, nonpad, round(fpad, 2))


def _cell(kw):
    import probe_real_v1_chain as C
    h, fit, tag = kw.pop("_hand"), kw.pop("_fit"), kw.pop("_tag")
    table = bool(kw.pop("_table", False))
    cell = dict(TABLE if table else BASE)
    seat = dict(place_xy=None, seat_z=None, tip_len=0.0) if table else dict(
        place_xy=fit["place_xy"], seat_z=fit["seat_z"], tip_len=fit["tip_len"])
    if not table:
        cell["angle_deg"] = h["angle_deg"]
    try:
        r = C.chain(Path(h["tag"]), arm_ik=Path(fit["ik"]), scene_path=Path(fit["arm"]),
                    anchor_ctrl=fit["anchor"], axis_k=h["axis_k"],
                    budget=h["budget"], grip_depth=fit["depth_mm"] / 1000,
                    **seat, **cell, **kw)
    except Exception as exc:
        return {"arm": tag, "tag": h["tag"], "set": h["set"], "error": repr(exc), "ok": False,
                **{k: v for k, v in kw.items() if isinstance(v, (int, float))}}
    seam = {s["phase"]: s for s in r["seams"]}
    t, L = seam.get("turned", {}), seam.get("lifted", {})
    r["seams"] = [{k: s[k] for k in ("phase", "t", "cos", "tilt_deg", "z", "pad_contacts",
                                     "pad_force_N", "spin_deg", "roll_deg", "slide_mm")
                   if k in s} for s in r["seams"]]
    r.pop("cycles", None)
    r["arm"], r["tag"], r["set"] = tag, h["tag"], h["set"]
    r["pads_turned"], r["force_turned_N"] = t.get("pad_contacts"), t.get("pad_force_N")
    r["z_turned"], r["pads_lifted"] = t.get("z"), L.get("pad_contacts")
    # A dropped shaft standing in a countersink reads vertical, so every tilt is gated on the
    # hand still carrying it clear at the last commanded turn step.
    r["held_turn"] = bool((t.get("pad_contacts") or 0) >= 2 and (t.get("z") or 0.0) > 0.08)
    r["held_lift"] = bool((L.get("pad_contacts") or 0) >= 2 and (L.get("z") or 0.0) > 0.08)
    # Where the tool ends up standing, before the hand goes anywhere near the gait grasp. In
    # the table-stand mode this is the reorientation's own result and `tilt_turned_deg` is not
    # -- there is no finger turn, so that column reads the tool still lying down.
    u = seam.get("upright", seam.get("staged", {}))
    r["tilt_upright_deg"] = u.get("tilt_deg")
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sets", default="AB")
    ap.add_argument("--stand", default="air", choices=("air", "table"),
                    help="air = the published cell (fingers turn the tool, arm stages it over "
                         "the countersink); table = no finger turn, the tool is stood up by "
                         "pivoting on the table and never leaves a surface")
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--cycles", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--squeeze-mm", default=None,
                    help="comma list of grasp squeeze values in mm, overriding SQUEEZE_MM. The "
                         "deployed plans all say 10.0, which is a BENCH number: on a tool lying "
                         "on a table it ejects the tool 14-20 mm out of the pinch at closure. "
                         "2.0 gives 64/64 lifts against 41/64 and 23/64 chains against 8/64.")
    ap.add_argument("--clears", default=None,
                    help="comma list of re-index clearance heights in m to sweep (table mode). "
                         "The palm lifts this far above the gait pose before it translates "
                         "across; 20 of 36 stands are still lost during that move.")
    ap.add_argument("--reposes", default=None,
                    help="comma list of repose_steps to sweep. In table mode this also sets the "
                         "length of each of the re-index's three legs, and g12 flips 0/4 to 3/4 "
                         "between 800 and 900.")
    ap.add_argument("--loads", default="0,250",
                    help="servo-load set-points for the grip loop; 0 = open loop")
    ap.add_argument("--video-seed", type=int, default=0)
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    H = hands(args.sets)
    print(f"{len(H)} hands: " + ", ".join(f"{h['tag']}[{h['set']}]" for h in H), flush=True)
    sqs = [None] if not args.squeeze_mm else [float(v) for v in args.squeeze_mm.split(",")]
    fits, jobs, skipped = {}, [], []
    vid = args.out / "videos"
    for h in H:
      for sq in sqs:
        f = prepare(h, squeeze_mm=sq)
        if f is None:
            skipped.append(h["tag"])
            print(f"  {h['tag']}: NO POSE / BUILD FAILED", flush=True)
            continue
        fits[f["tag"]] = f
        grid = [(c, rp) for c in ([None] if not args.clears else
                                  [float(v) for v in args.clears.split(",")])
                for rp in ([None] if not args.reposes else
                           [int(v) for v in args.reposes.split(",")])]
        for lt in (float(v) for v in args.loads.split(",")):
          for cl, rp in grid:
            for rep in range(args.reps):
                tg = f"load{lt:.0f}" + ("" if sq is None else f"_sq{sq:g}")
                if cl is not None:
                    tg += f"_c{cl*1000:.0f}"
                if rp is not None:
                    tg += f"_r{rp}"
                kw = {"_hand": h, "_fit": f, "_tag": tg,
                      "_table": args.stand == "table",
                      "load_target": lt, "seed": rep, "jitter": 0.0005,
                      "cycles": args.cycles}
                if cl is not None:
                    kw["clear"] = cl
                if rp is not None:
                    kw["repose_steps"] = rp
                if rep == args.video_seed and not args.no_video and len(grid) == 1 \
                        and len(sqs) == 1:
                    kw["video"] = vid / f"20260905-{h['tag']}_{args.stand}_{tg}.mp4"
                    kw["video_size"] = (640, 480)
                    kw["cam"] = (-60.0, -20.0, 0.42)
                    kw["cam_look"] = (0.02, -0.005, 0.045)
                jobs.append(kw)
    print(f"{len(jobs)} cells on {args.workers} workers, {len(skipped)} skipped", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(_cell, jobs, chunksize=1)):
            rows.append(r)
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(jobs)}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "chain_hands.json").write_text(json.dumps(
        {"rows": rows, "skipped": skipped, "hands": H,
         "base": TABLE if args.stand == "table" else BASE, "stand": args.stand}, separators=(",", ":")))
    print(f"-> {args.out / 'chain_hands.json'}")

    import statistics as st
    m = lambda x: st.mean(x) if x else float("nan")
    print(f"\n   {'tag':20} {'set':>3} {'cell':>14} {'lift':>5} {'upright':>8} {'stood':>6} "
          f"{'chain':>6} {'cyc':>5} {'deg/cy':>7} {'endtilt':>8} {'F_N':>6} {'roll':>6} "
          f"{'free':>5}")
    for h in H:
        for lt in sorted({r.get("arm", "") for r in rows if r.get("tag") == h["tag"]}):
            g = [r for r in rows if r.get("tag") == h["tag"] and r.get("arm") == lt]
            if not g:
                continue
            hh = [r for r in g if r.get("held_turn")]
            # `src or g` would fall back to the whole group whenever the filtered list is
            # EMPTY, which is exactly the hands that never stood the tool -- and then prints
            # their all-run mean under a column headed "over the runs that stood it".
            def f(k, src=None):
                return [float(r.get(k) or 0.0) for r in (g if src is None else src)]
            sd = [r for r in g if r.get("stood_ok")]
            print(f"   {h['tag']:20} {h['set']:>3} {str(lt)[:14]:>14} "
                  f"{sum(1 for r in g if r.get('held_lift')):2}/{len(g):<2} "
                  f"{m(f('tilt_upright_deg', sd)):8.2f} "
                  f"{len(sd):2}/{len(g):<2} "
                  f"{sum(1 for r in g if r.get('ok')):2}/{len(g):<2} "
                  f"{m(f('cycles_run')):5.1f} {m(f('gain_mean_deg')):7.2f} "
                  f"{m(f('final_tilt_deg', sd)):8.2f} {m(f('hand_force_N')):6.1f} "
                  f"{m(f('roll_max_deg')):6.1f} {m(f('free_frac')):5.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
