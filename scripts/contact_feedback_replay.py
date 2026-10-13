#!/usr/bin/env python3
"""Closed loop against open loop under a swapped fingertip contact model, on the servo-limited plant (2026-10-13).

Each final policy of the 2026-10-11 servo-plant study (scripts/rl_contact_eval.py --study 20261011: TPU mesh, 1 mm pads
and sprung skin, two seeds each) is rolled closed loop in its own tip scene, 64 rollouts of 250 policy steps (5 s) with
the policy mean as the action, and its 64 action sequences are saved (acts/<tag>.npz). On every tip scene of the study
(box, TPU mesh, 1 mm pads, sprung skin) the policy is then rolled closed loop, and its own-scene action sequences are
replayed open loop through the same action pipeline (rollout k replays sequence k; the scripted grasp and lift and the
residual's mapping to servo targets are unchanged). The open-loop replay in the policy's own scene is the control: it
differs from the closed loop only by the GPU solver's run-to-run spread.

The held load test (two fingertips each pressing at least 0.24 N, the tool above 60 mm) and the signed cosine of the
tool's axis with vertical are those of rl_contact_eval.metrics, so the closed-loop rows reproduce transfer.jsonl at 64
rollouts. Rows (fsynced): docs/experiments/20261013-contact_feedback/replay.jsonl, keyed by (tag, scene, mode, perturb).

`--perturb kind=value ...` adds a pass on each policy's own scene under one physical change at a time, closed loop and
open-loop replay of the nominal actions: friction (every geom's sliding coefficient scaled), mass (the tool's mass and
inertia scaled), kp (the finger servos' gain in N m/rad), fr (their torque limit in N m), as rl_contact_eval.Evaluator.perturb.

`--census` rolls each policy closed loop in its own scene and splits, per policy step, the normal force the tool receives
between the fingertip contact geoms (pads, or the tip mesh or box) and every other geom of the hand (finger links, palm),
whose contacts keep MuJoCo's default solref and solimp. Rows: census.jsonl.

`--film` renders FILMS: each a policy closed loop beside the open-loop replay of its nominal actions on one scene under
one condition, the same world of each, at real speed, tiled one condition per row (media/20261013-closed_open.mp4).

    MUJOCO_GL=egl WARP_CACHE_PATH=$(mktemp -d) uv run --extra rl --extra gpu python scripts/contact_feedback_replay.py
    ... --tags 20261011-d6_servo_tpu27pads1_40M_s0 --scenes tpu27pads1 tpu27mesh     # a subset
    ... --scenes --perturb friction=0.5 friction=0.7 friction=1.3 mass=0.8 mass=1.2 mass=2 kp=0.36 fr=0.12
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import rl_contact_eval as E  # noqa: E402

OUT = ROOT / "docs/experiments/20261013-contact_feedback"
ROWS = OUT / "replay.jsonl"
CENSUS = OUT / "census.jsonl"
MEDIA = OUT / "media"
# (tag, scene, perturbation or None, row label): the tip swap with the largest closed/open difference in held cosine, and
# the perturbation with the largest difference in held count
FILMS = [("20261011-d6_servo_tpu27mesh_40M_s0", "box", None, "TPU-mesh policy, seed 0, on the box tip"),
         ("20261011-d6_servo_tpu27mesh_40M_s1", "tpu27mesh", ("friction", 0.5),
          "TPU-mesh policy, seed 1, on its own tip, friction \u00d70.5")]
ACTS = OUT / "acts"
SCENES = ("tpu27mesh", "tpu27pads1", "tpu27skin", "box")


class _Recorded:
    """Stands in for deploy.act_b during a replay: returns the recorded action of the next policy step."""

    def __init__(self, acts, device):
        import torch
        self.acts = torch.as_tensor(acts, device=device)
        self.k = 0

    def __call__(self, actor, obs_td, stochastic):
        a = self.acts[self.k]
        self.k += 1
        return a


def replay(ev: E.Evaluator, acts: np.ndarray) -> dict:
    """ev.rollout() with the actor's action replaced by `acts` (steps x rollouts x 9)."""
    import morphohand.rl.deploy as D
    own = D.act_b
    D.act_b = _Recorded(acts, ev.u.device)
    try:
        return ev.rollout()
    finally:
        D.act_b = own


def census(ev: E.Evaluator) -> dict:
    """Closed-loop rollout recording, per policy step and rollout, the tool's normal contact force from the fingertip
    geoms and from the rest of the hand, and the hand geoms that touched it (elliptic cones: row 0 is the normal)."""
    import collections
    from morphohand.rl.deploy import act_b
    torch, mjm, wd = ev.torch, ev.mjm, ev.u.sim.wp_data
    T, N = ev.steps, ev.n
    bname = [mjm.body(i).name.split("/")[-1] for i in range(mjm.nbody)]
    hand = np.array([not ev.tool_geom[g] and ev.geom_finger[g] < 0 and mjm.geom_bodyid[g] != 0
                     for g in range(mjm.ngeom)])
    f_tip, f_hand = np.zeros((T, N)), np.zeros((T, N))
    by_body = collections.Counter()
    obs_td, _ = ev.wrapped.reset()
    with torch.no_grad():
        for s in range(T):
            obs_td, *_ = ev.wrapped.step(act_b(ev.actor, obs_td, False))
            nc = int(wd.nacon.numpy()[0])
            if not nc:
                continue
            g = wd.contact.geom.numpy()[:nc]
            wid = wd.contact.worldid.numpy()[:nc]
            adr = wd.contact.efc_address.numpy()[:nc, 0]
            on_tool = ev.tool_geom[g[:, 0]] | ev.tool_geom[g[:, 1]]
            other = np.where(ev.tool_geom[g[:, 0]], g[:, 1], g[:, 0])
            ok = on_tool & (adr >= 0)
            fn = np.zeros(nc)
            fn[ok] = wd.efc.force.numpy()[wid[ok], adr[ok]]
            tip = ok & (ev.geom_finger[other] >= 0)
            hnd = ok & hand[other]
            np.add.at(f_tip[s], wid[tip], fn[tip])
            np.add.at(f_hand[s], wid[hnd], fn[hnd])
            if s >= ev.residual_from:
                for gg, f in zip(other[hnd], fn[hnd]):
                    by_body[bname[mjm.geom_bodyid[gg]]] += float(f)
    return dict(f_tip=f_tip, f_hand=f_hand, by_body=by_body)


def row(tag, arm, seed, ck, scene, mode, r, ev, t0, perturb: str = "nominal") -> dict:
    return dict(tag=tag, arm=arm, seed=seed, iteration=E._it(ck), scene=scene, mode=mode, perturb=perturb,
                **E.metrics(r, ev.residual_from), status="ok", wall_s=round(time.perf_counter() - t0, 1),
                when=time.strftime("%Y-%m-%d %H:%M"))


def done() -> set:
    """(tag, scene, mode, perturb) of the rows written; rows before the perturbation pass are nominal."""
    import json
    if not ROWS.exists():
        return set()
    return {(r["tag"], r["scene"], r["mode"], r.get("perturb", "nominal")) for r in map(json.loads, open(ROWS))
            if r.get("status") == "ok"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tags", nargs="*", default=[])
    ap.add_argument("--scenes", nargs="*", default=list(SCENES), choices=SCENES)
    ap.add_argument("--perturb", nargs="*", default=[], help="kind=value, on each policy's own scene")
    ap.add_argument("--census", action="store_true", help="tool contact force: fingertips against the rest of the hand")
    ap.add_argument("--film", action="store_true", help="render FILMS")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--steps", type=int, default=250)
    args = ap.parse_args()
    E.use_study("20261011")
    pol = [j for j in E.finished() if not args.tags or j[0] in args.tags]
    have = {k[:3] for k in done() if k[3] == "nominal"}
    ACTS.mkdir(parents=True, exist_ok=True)

    # pass 1: each policy closed loop in its own scene; its action sequences are what pass 2 replays
    for arm in SCENES:
        mine = [j for j in pol if j[1] == arm and not (ACTS / f"{j[0]}.npz").exists()]
        if not mine:
            continue
        ev = E.Evaluator(E.ARMS[arm], E.final_ckpt(mine[0][0], mine[0][3]), n=args.n, steps=args.steps)
        for tag, _, seed, steps in mine:
            ck = E.final_ckpt(tag, steps)
            ev.load(ck)
            t0 = time.perf_counter()
            r = ev.rollout()
            np.savez_compressed(ACTS / f"{tag}.npz", acts=r["acts"], scene=arm, checkpoint=str(ck.relative_to(ROOT)))
            if (tag, arm, "closed") not in have:
                E.append(ROWS, row(tag, arm, seed, ck, arm, "closed", r, ev, t0))
                have.add((tag, arm, "closed"))
            print(f"{tag} closed on {arm}: {r['acts'].shape[1]} action sequences saved", flush=True)
        ev.close()

    # pass 2: every policy on every scene, closed loop and open-loop replay of its own-scene actions
    for scene in args.scenes:
        todo = [j for j in pol if (j[0], scene, "closed") not in have or (j[0], scene, "open") not in have]
        if not todo:
            continue
        ev = E.Evaluator(E.ARMS[scene], E.final_ckpt(todo[0][0], todo[0][3]), n=args.n, steps=args.steps)
        for tag, arm, seed, steps in todo:
            ck = E.final_ckpt(tag, steps)
            ev.load(ck)
            if (tag, scene, "closed") not in have:
                t0 = time.perf_counter()
                rw = row(tag, arm, seed, ck, scene, "closed", ev.rollout(), ev, t0)
                E.append(ROWS, rw)
                print(f"{tag} closed on {scene}: held {rw['n_held']}/{rw['n']}, held cos {rw['held_cos_mean']}",
                      flush=True)
            if (tag, scene, "open") not in have:
                acts = np.load(ACTS / f"{tag}.npz")["acts"]
                t0 = time.perf_counter()
                rw = row(tag, arm, seed, ck, scene, "open", replay(ev, acts), ev, t0)
                E.append(ROWS, rw)
                print(f"{tag} open on {scene}: held {rw['n_held']}/{rw['n']}, held cos {rw['held_cos_mean']}",
                      flush=True)
        ev.close()

    # pass 3: each policy on its own scene under one physical change at a time
    perts = [E._parse_perturb(p) for p in args.perturb]
    have4 = done()
    for arm in SCENES:
        mine = [j for j in pol if j[1] == arm]
        need = [(j, k, v) for j in mine for k, v in perts
                if {(j[0], arm, mode, f"{k}={v:g}") for mode in ("closed", "open")} - have4]
        if not need:
            continue
        ev = E.Evaluator(E.ARMS[arm], E.final_ckpt(mine[0][0], mine[0][3]), n=args.n, steps=args.steps)
        for (tag, _, seed, steps), k, v in need:
            ck = E.final_ckpt(tag, steps)
            ev.load(ck)
            ev.perturb(k, v)
            name = f"{k}={v:g}"
            for mode in ("closed", "open"):
                if (tag, arm, mode, name) in have4:
                    continue
                t0 = time.perf_counter()
                r = ev.rollout() if mode == "closed" else replay(ev, np.load(ACTS / f"{tag}.npz")["acts"])
                rw = row(tag, arm, seed, ck, arm, mode, r, ev, t0, perturb=name)
                E.append(ROWS, rw)
                print(f"{tag} {mode} {name}: held {rw['n_held']}/{rw['n']}, held cos {rw['held_cos_mean']}", flush=True)
            ev.perturb(None)
        ev.close()

    # pass 4: where the tool's contact force comes from
    if args.census:
        import json
        seen = {json.loads(x)["tag"] for x in open(CENSUS)} if CENSUS.exists() else set()
        for arm in SCENES:
            mine = [j for j in pol if j[1] == arm and j[0] not in seen]
            if not mine:
                continue
            ev = E.Evaluator(E.ARMS[arm], E.final_ckpt(mine[0][0], mine[0][3]), n=args.n, steps=args.steps)
            a0 = ev.residual_from
            for tag, _, seed, steps in mine:
                ck = E.final_ckpt(tag, steps)
                ev.load(ck)
                c = census(ev)
                ft, fh = c["f_tip"][a0:], c["f_hand"][a0:]
                tot = ft.sum() + fh.sum()
                E.append(CENSUS, dict(tag=tag, arm=arm, seed=seed, iteration=E._it(ck), n=args.n, from_step=a0,
                                      hand_share=float(fh.sum() / tot) if tot > 0 else None,
                                      steps_with_hand_force=float((fh > 0.01).mean()),
                                      hand_force_p95_N=float(np.percentile(fh, 95)),
                                      hand_force_max_N=float(fh.max()), tip_force_mean_N=float(ft.mean()),
                                      hand_force_by_body_Ns={k: round(v * 0.02 / args.n, 4) for k, v in
                                                             c["by_body"].most_common(8)},
                                      status="ok", when=time.strftime("%Y-%m-%d %H:%M")))
                print(f"{tag}: hand share {fh.sum() / max(tot, 1e-12):.4f}, steps with hand force "
                      f"{(fh > 0.01).mean():.3f}", flush=True)
            ev.close()

    if args.film:
        film()


FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def film(width: int = 480, height: int = 360):
    """FILMS as one tiled film: per row a title bar and the closed loop (left) beside the open-loop replay (right) of the
    same world, chosen as the world whose open-loop final cosine is the median of the 64 (a dropped tool counts with its
    final cosine). Each panel has a label bar with the counts over all 64 rollouts and a strip with the world's time,
    cosine and height; text is 19-22 px on a 960 px frame."""
    import json
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    E.use_study("20261011")
    jobs = {j[0]: j for j in E.finished()}
    f_title, f_lab = ImageFont.truetype(FONT_BOLD, 22), ImageFont.truetype(FONT, 19)
    TITLE_H, LAB_H, STRIP_H = 38, 32, 30
    rows, meta = [], []
    for tag, scene, pert, label in FILMS:
        _, arm, seed, steps = jobs[tag]
        ck = E.final_ckpt(tag, steps)
        ev = E.Evaluator(E.ARMS[scene], ck, n=64, steps=250, watch=True)
        ev.load(ck)
        if pert:
            ev.perturb(*pert)
        rc = ev.rollout()
        ro = replay(ev, np.load(ACTS / f"{tag}.npz")["acts"])
        fc = ro["cos"][-1]
        w = int(np.argsort(fc)[len(fc) // 2])
        info = dict(tag=tag, scene=scene, perturb=f"{pert[0]}={pert[1]:g}" if pert else "nominal", label=label, world=w)
        ks = list(range(0, 250, 2))
        panels = []
        for mode, r in (("Closed loop", rc), ("Open-loop replay", ro)):
            m = E.metrics(r, ev.residual_from)
            held_t = (r["z"] > E.FLOOR_Z) & ((r["force"] >= E.HELD_N).sum(-1) >= 2)
            head = f"{mode}: held {m['n_held']}/64" + (f", held cos {m['held_cos_mean']:.2f}"
                                                        if m["held_cos_mean"] is not None else "")
            out = []
            for k, fr in zip(ks, ev.render_world(r, w, ks, width, height)):
                im = Image.new("RGB", (width, LAB_H + height), "white")
                im.paste(Image.fromarray(fr[:, :, :3]), (0, LAB_H))
                dr = ImageDraw.Draw(im, "RGBA")
                dr.text((10, 5), head, font=f_lab, fill=(0, 0, 0))
                dr.line([(0, LAB_H - 1), (width, LAB_H - 1)], fill=(0, 0, 0), width=1)
                dropped = k >= ev.residual_from and not held_t[k, w]
                txt = (f"t {0.02 * (k + 1):.2f} s   cos {r['cos'][k, w]:+.2f}   z {1e3 * r['z'][k, w]:.0f} mm"
                       + ("   dropped" if dropped else ""))
                dr.rectangle([0, LAB_H + height - STRIP_H, width, LAB_H + height], fill=(255, 255, 255, 215))
                dr.text((10, LAB_H + height - STRIP_H + 4), txt, font=f_lab, fill=(150, 0, 0) if dropped else (0, 0, 0))
                out.append(np.asarray(im))
            panels.append(out)
            key = "closed" if mode == "Closed loop" else "open"
            info.update({f"{key}_cos": float(r["cos"][-1, w]), f"{key}_held": bool(held_t[-1, w]),
                         f"{key}_n_held": m["n_held"], f"{key}_held_cos": m["held_cos_mean"]})
        meta.append(info)
        title = Image.new("RGB", (2 * width, TITLE_H), "white")
        ImageDraw.Draw(title).text((10, 7), f"World {w}: {label}", font=f_title, fill=(0, 0, 0))
        title = np.asarray(title)
        sep = np.zeros((LAB_H + height, 2, 3), np.uint8)
        rows.append([np.concatenate([title, np.concatenate([a[:, :-1], sep, b[:, 1:]], axis=1)], axis=0)
                     for a, b in zip(*panels)])
        ev.close()
    frames = [np.concatenate(fr, axis=0) for fr in zip(*rows)]
    pad = (-frames[0].shape[0]) % 8
    if pad:
        frames = [np.concatenate([f, np.full((pad, f.shape[1], 3), 255, np.uint8)], axis=0) for f in frames]
    MEDIA.mkdir(parents=True, exist_ok=True)
    out = MEDIA / "20261013-closed_open.mp4"
    wr = imageio.get_writer(str(out), fps=25, codec="libx264", quality=None, pixelformat="yuv420p", macro_block_size=8,
                            ffmpeg_params=["-crf", "25", "-preset", "medium"], ffmpeg_log_level="error")
    for f in frames:
        wr.append_data(np.asarray(f, np.uint8))
    wr.close()
    Image.fromarray(frames[int(0.75 * (len(frames) - 1))]).save(out.with_suffix(".jpg"), quality=88)
    out.with_suffix(".json").write_text(json.dumps(dict(rows=meta, when=time.strftime("%Y-%m-%d %H:%M")), indent=1))
    print(f"wrote {out} ({len(frames)} frames, {frames[0].shape[1]}x{frames[0].shape[0]})", flush=True)

if __name__ == "__main__":
    main()
