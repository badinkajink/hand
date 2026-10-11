"""N-rollout evaluation for a reorient policy: success + stability metrics as a DISTRIBUTION.

Why this exists. `policy_healthcheck.py` rolls ONE episode and grades it. On the perp
topology that turned out not to be reproducible: three rollouts of the same deterministic
checkpoint, with stochastic sampling off and spawn jitter at zero, ended three different
ways (held at step 400 / lost at 430 / lost before 400). The actions are deterministic, so
the spread comes from the simulator — parallel contact solves do not reduce in a fixed
order on GPU. A single rollout therefore cannot tell you whether a policy holds; it tells
you what happened once.

So every number here is over N envs stepped together, and the headline is a RATE plus a
spread, not a value. Cheap: N rollouts cost about the same wall-clock as one, because the
envs are batched.

Metrics, all of them about the task rather than the reward:
  align_rate     fraction of envs that ever reach cos >= --align-thresh
  hold_rate      fraction still HOLDING at the end (the number that matters -- a shaft
                 standing on the floor reads cos +1.000 and is not a hold)
  t_align        steps to first reach the threshold (speed of the reorient)
  hold_steps     consecutive steps aligned AND held, per env (stability of the hold)
  final_cos/z    where it ended up
  drop_step      when the object was lost, for the envs that lost it

"Held" is asked of the physics, not inferred from height: total fingertip force > --held-min-n
AND object above --floor-z. That distinction is load-bearing on this topology, where the
documented failure is the shaft reaching vertical by sliding OUT of the pinch and ending
upright on the floor -- perfect cos, zero grip.

The plant is the run's own (2026-10-11). The timing knobs come from the run's config.yaml
(run_env_overrides), and so does the tool's contact class: on 2026-09-20 env_build began to
copy the scene's solref/solimp onto the tool (before, the tool trained on MuJoCo's defaults,
solref 0.02 / solimp 0.9-0.95), and the same change added `scene_floor` to every run config.
A run whose config.yaml has no `scene_floor` key trained on the defaults, and is evaluated on
them (--object-contact run); `scene` or `legacy` forces one.

Traces (2026-10-11). Every evaluation writes its per-step traces, so a figure or a new summary
never needs the policy run again: one npz per policy, `traces/<run>.npz` beside the --json-out
file (without --json-out, logs/eval_traces/<run>.npz; --traces names another file, --no-traces
skips it), one evaluation per key, the key naming the condition (nominal, j3 = spawn +-3 mm and
+-10 deg, j3dr = the same with friction DR, _r<step> = residual onset forced, _stochastic =
sampled actions; --trace-key overrides). A key written by a different checkpoint of the same
run gets the suffix _m<N>. Arrays of key K, T control steps (0.02 s) x N rollouts, float32:

  K/cos         signed cosine between the tool's long axis and world +z (+1 upright, -1 inverted)
  K/z           tool height in the world frame (m)
  K/grip        sum over the pads of the contact-force magnitude (N); the held flag thresholds it
  K/held        uint8: grip > held_min_n and z > floor_z
  K/pads        T x N x 3: contact-force magnitude per pad, thumb / index / middle (N)
  K/clearance   index-middle link-chain clearance (m)
  K/act_absmax  largest |action| over the nine outputs
  K/act_gt1     share of the nine outputs with |action| > 1
  K/jvel_max    largest finger-joint speed (rad/s)
  K/gap_max     largest commanded-minus-achieved finger angle, command clamped to the range (rad)
  K/raw_exc     largest raw command beyond a finger joint's range (rad)
  K/meta        0-d string, JSON: checkpoint, morphology run, the run's timing (residual onset,
                reorient step, residual scale, action clip), tool contact class, thresholds,
                reset perturbations, actor mode, N, T, control period, date and commit

`load_traces(path)` returns {key: {array name: array, "meta": dict}} without importing torch;
`summarize(trace)` computes the evaluation JSON's numbers from it. The check

    python3 scripts/policy_eval_suite.py --check TRACES.npz:KEY EVAL.json

recomputes the summaries from the file and compares them with the JSON field by field.

Usage:
  uv run --extra rl --extra gpu python scripts/policy_eval_suite.py \
      --policy results/rl/<run>/tensorboard/model_338.pt \
      --morphology-run results/phase1/perp/perp_v1 \
      --closed-ctrl-from-keyframe closed --open-finger-from-keyframe \
      --lift-delta 0.14 --steps 500 --n 64 --plot
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRACE_ARRAYS = ("cos", "z", "grip", "held", "pads", "clearance", "act_absmax", "act_gt1", "jvel_max", "gap_max",
                "raw_exc")


# ----------------------------------------------------------------------------------------------
# traces: reader, writer, summaries (numpy only, so page builders can import them)
# ----------------------------------------------------------------------------------------------

def load_traces(path) -> dict:
    """{key: {"cos": (T, N), ..., "pads": (T, N, 3), "meta": dict}} from a traces npz."""
    out: dict = {}
    with np.load(path, allow_pickle=False) as z:
        for name in z.files:
            key, _, arr = name.rpartition("/")
            out.setdefault(key, {})[arr] = json.loads(str(z[name])) if arr == "meta" else z[name]
    return out


def save_trace(path: Path, key: str, arrays: dict, meta: dict) -> str:
    """Add (or replace) one evaluation in the policy's npz; returns the key used. Written to a temporary file,
    fsynced and renamed, so an interrupted write leaves the previous file intact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    old = {}
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            old = {n: z[n] for n in z.files}
        ck = {json.loads(str(v)).get("checkpoint") for n, v in old.items() if n.endswith("/meta")}
        if ck and meta["checkpoint"] not in ck:
            key = f"{key}_m{Path(meta['checkpoint']).stem.split('_')[-1]}"
    old = {n: v for n, v in old.items() if not n.startswith(key + "/")}
    new = {f"{key}/{n}": arrays[n] for n in TRACE_ARRAYS}
    new[f"{key}/meta"] = np.array(json.dumps(meta, sort_keys=True))
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **old, **new)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return key


def summarize(tr: dict, align_thresh: float | None = None, held_min_n: float | None = None,
              floor_z: float | None = None, residual_from: int | None = None) -> dict:
    """The evaluation's summary numbers from its traces (arrays as stored, thresholds from the meta unless given)."""
    m = tr.get("meta", {})
    align_thresh = m["align_thresh"] if align_thresh is None else align_thresh
    held_min_n = m["held_min_n"] if held_min_n is None else held_min_n
    floor_z = m["floor_z"] if floor_z is None else floor_z
    residual_from = m["residual_from_step"] if residual_from is None else residual_from
    cos_t, z_t, grip_t, perf_t = tr["cos"], tr["z"], tr["grip"], tr["pads"]
    T, N = cos_t.shape
    held_t = (grip_t > held_min_n) & (z_t > floor_z)                  # (T, N)
    aligned_t = cos_t >= align_thresh
    good_t = aligned_t & held_t
    ever_aligned = aligned_t.any(axis=0)
    held_final = held_t[-1]
    # first step each env crosses the threshold (nan where it never does)
    t_align = np.where(ever_aligned, aligned_t.argmax(axis=0), np.nan)
    # longest run of aligned-AND-held per env
    hold_steps = np.zeros(N, dtype=int)
    for e in range(N):
        best = cur = 0
        for v in good_t[:, e]:
            cur = cur + 1 if v else 0
            best = max(best, cur)
        hold_steps[e] = best
    lost = held_t.any(axis=0) & ~held_final
    drop_step = np.full(N, np.nan)
    for e in np.nonzero(lost)[0]:
        drop_step[e] = int(np.nonzero(held_t[:, e])[0].max()) + 1
    # Three-finger share is reported over the ALIGNED+HELD steps only. Over the whole rollout it
    # is dominated by the pre-lift and swing phases, where a loaded thumb is not wanted anyway.
    if good_t.any():
        gf = perf_t[good_t]                                            # (K, 3)
        three = float((gf.min(axis=1) > held_min_n).mean())
    else:
        three, gf = 0.0, np.zeros((1, 3))
    act_from = max(0, int(residual_from))
    clear_t = tr["clearance"]
    clear_min = clear_t[act_from:].min(axis=0) * 1000.0                 # mm per rollout
    clear_end = clear_t[-1] * 1000.0
    pad_peak = perf_t[act_from:].max(axis=0)                            # (N, 3)
    jvel_p99 = np.degrees(np.percentile(tr["jvel_max"][act_from:], 99))
    gap = tr["gap_max"][act_from:]
    gap_deg = float(np.degrees(np.nanmax(gap))) if np.isfinite(gap).any() else float("nan")

    def nanstat(a, f):
        a = a[~np.isnan(a)]
        return float(f(a)) if a.size else None

    return dict(
        n=int(N), steps=int(T), align_thresh=float(align_thresh),
        align_rate=float(ever_aligned.mean()), hold_rate=float(held_final.mean()),
        success_rate=float((ever_aligned & held_final).mean()),
        t_align_mean=float(np.nanmean(t_align)) if ever_aligned.any() else None,
        t_align_sd=nanstat(t_align, np.std),
        hold_steps_mean=float(hold_steps.mean()), hold_steps_sd=float(hold_steps.std()),
        final_cos_mean=float(cos_t[-1].mean()), final_cos_sd=float(cos_t[-1].std()),
        final_z_mean=float(z_t[-1].mean()), final_z_sd=float(z_t[-1].std()),
        peak_cos_mean=float(cos_t.max(axis=0).mean()), peak_cos_sd=float(cos_t.max(axis=0).std()),
        drop_step_mean=float(np.nanmean(drop_step)) if lost.any() else None,
        drop_step_sd=nanstat(drop_step, np.std),
        n_lost=int(lost.sum()),
        three_finger_share=three,
        mean_force_thumb=float(gf[:, 0].mean()),
        mean_force_index=float(gf[:, 1].mean()),
        mean_force_middle=float(gf[:, 2].mean()),
        clearance_min_mm_mean=float(clear_min.mean()), clearance_min_mm_sd=float(clear_min.std()),
        clearance_min_mm_worst=float(clear_min.min()), clearance_end_mm_mean=float(clear_end.mean()),
        pad_peak_thumb=float(pad_peak[:, 0].mean()), pad_peak_index=float(pad_peak[:, 1].mean()),
        pad_peak_middle=float(pad_peak[:, 2].mean()),
        joint_speed_p99_deg_s=float(jvel_p99), residual_gt1_frac=float(tr["act_gt1"][act_from:].mean()),
        residual_absmax=float(tr["act_absmax"][act_from:].max()),
        ctrl_gap_max_deg=gap_deg, raw_cmd_beyond_range_deg=float(np.degrees(tr["raw_exc"][act_from:].max())),
        residual_from_step=int(residual_from),
        force_active_thumb=float(perf_t[act_from:, :, 0].mean()),
        force_active_index=float(perf_t[act_from:, :, 1].mean()),
        force_active_middle=float(perf_t[act_from:, :, 2].mean()),
    )


def check(npz: str, json_path: str) -> int:
    """Recompute the summaries of NPZ[:KEY] and compare every numeric field with the JSON; 0 when all agree."""
    path, _, key = npz.partition(":")
    traces = load_traces(path)
    if not key:
        if len(traces) != 1:
            raise SystemExit(f"{path} holds {sorted(traces)}; name one as {path}:KEY")
        key = next(iter(traces))
    got = summarize(traces[key])
    ref = json.loads(Path(json_path).read_text())
    bad, n = [], 0
    for k, v in got.items():
        if k not in ref:
            continue
        r = ref[k]
        n += 1
        if v is None or r is None:
            ok = v is None and r is None
        else:
            ok = float(v) == float(r) or (np.isnan(float(v)) and np.isnan(float(r)))
        if not ok:
            bad.append((k, v, r))
    print(f"{path}:{key} against {json_path}: {n} fields compared, {len(bad)} differ")
    for k, v, r in bad:
        print(f"  {k}: traces {v!r}, json {r!r}")
    return 1 if bad else 0


def _git_state() -> dict:
    try:
        head = subprocess.run(["git", "rev-parse", "--short=8", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "scripts/policy_eval_suite.py"],
                               capture_output=True, text=True, cwd=ROOT).stdout.strip()
        return {"commit": head, "dirty": bool(dirty)}
    except OSError:
        return {}


def _rel(p: Path) -> str:
    p = Path(p).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def _run_dir(policy: Path) -> Path:
    """The run directory of a checkpoint: the nearest parent with a config.yaml (else the checkpoint's folder)."""
    d = Path(policy).resolve().parent
    for cand in (d, d.parent, d.parent.parent):
        if (cand / "config.yaml").exists():
            return cand
    return d


def _default_key(args, residual_from: int, trained_from: int, contact: str, contact_run: str) -> str:
    key = "nominal"
    if args.spawn_jitter_mm or args.spawn_yaw_deg or args.friction_dr:
        key = ""
        if args.spawn_jitter_mm or args.spawn_yaw_deg:
            key = f"j{args.spawn_jitter_mm:g}" + (f"y{args.spawn_yaw_deg:g}" if args.spawn_yaw_deg and args.spawn_yaw_deg != 10 else "")
        key += "dr" if args.friction_dr else ""
    if residual_from != trained_from:
        key += f"_r{residual_from}"
    if args.stochastic:
        key += "_stochastic"
    if contact != contact_run:
        key += f"_{contact}"
    return key


def _use_legacy_object_contact():
    """Build the tool entity as env_build did before 2026-09-20: type, size, mass and friction from the scene, every
    other contact attribute at MjSpec's default (solref 0.02 1, solimp 0.9 0.95 0.001 0.5 2, condim 3)."""
    import mujoco
    from morphohand.rl import env_build
    probe = mujoco.MjSpec().worldbody.add_body().add_geom()
    attrs = ("solref", "solimp", "solmix", "priority", "condim", "margin", "gap", "contype", "conaffinity", "group")
    default = {a: (np.array(getattr(probe, a)).tolist() if np.ndim(getattr(probe, a)) else getattr(probe, a)) for a in attrs}
    orig = env_build.make_object_spec_from_frozen

    def legacy(*a, **k):
        spec = orig(*a, **k)
        for body in spec.worldbody.bodies:
            for g in body.geoms:
                for attr, v in default.items():
                    setattr(g, attr, v)
        return spec
    env_build.make_object_spec_from_frozen = legacy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", type=Path)
    ap.add_argument("--morphology-run", type=Path)
    ap.add_argument("--n", type=int, default=64, help="parallel rollouts")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--align-thresh", type=float, default=0.9,
                    help="cos counted as 'reoriented'")
    ap.add_argument("--held-min-n", type=float, default=0.5,
                    help="total fingertip force (N) below which the object is NOT held")
    ap.add_argument("--floor-z", type=float, default=0.06,
                    help="object z below which it is resting on the floor, not held")
    ap.add_argument("--lift-delta", type=float, default=0.14)
    ap.add_argument("--finger-residual-scale", default="0.5",
                    help="scalar, or a comma-separated per-joint list in thumb/index/middle "
                         "yaw,mcp,pip order. Must MATCH the run's own value (gotcha #13): a "
                         "policy scored at a different residual scale is a different policy.")
    ap.add_argument("--open-finger-from-keyframe", action="store_true")
    ap.add_argument("--closed-ctrl-from-keyframe", default=None,
                    help="MUST match how the policy was trained (gotcha #13)")
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--plot", type=Path, default=None, help="write a summary PNG here")
    ap.add_argument("--label", default=None)
    ap.add_argument("--residual-from-step", type=int, default=None,
                    help="finger_residual_active_from_step; default = the run's config.yaml "
                         "(a policy trained with the residual masked through the lift must be "
                         "evaluated the same way)")
    ap.add_argument("--reorient-start-step", type=int, default=None,
                    help="reward gate step; default = the run's config.yaml")
    ap.add_argument("--spawn-jitter-mm", type=float, default=0.0,
                    help="uniform tool spawn jitter in x and y, mm (the training runs used 0)")
    ap.add_argument("--spawn-yaw-deg", type=float, default=0.0, help="uniform tool spawn yaw jitter, deg")
    ap.add_argument("--friction-dr", action="store_true",
                    help="scale every geom's sliding friction per env by the training band (0.55-1.15) at reset")
    ap.add_argument("--stochastic", action="store_true",
                    help="sample actions from the policy's distribution (the training-time "
                         "behaviour) instead of its mean")
    ap.add_argument("--object-contact", choices=("run", "scene", "legacy"), default="run",
                    help="the tool's contact class: the run's own (default; legacy for a config.yaml without "
                         "scene_floor), the scene's solref/solimp, or MuJoCo's defaults (before 2026-09-20)")
    ap.add_argument("--traces", type=Path, default=None,
                    help="npz for the per-step traces; default traces/<run>.npz beside --json-out")
    ap.add_argument("--trace-key", default=None, help="condition name inside the npz; default from the flags")
    ap.add_argument("--no-traces", action="store_true")
    ap.add_argument("--check", nargs=2, metavar=("NPZ[:KEY]", "JSON"),
                    help="recompute the summaries of a traces file and compare them with an evaluation JSON")
    args = ap.parse_args()
    if args.check:
        sys.exit(check(*args.check))
    if args.policy is None or args.morphology_run is None:
        ap.error("--policy and --morphology-run are required")
    _frs = [float(v) for v in str(args.finger_residual_scale).split(",")]
    args.finger_residual_scale = _frs[0] if len(_frs) == 1 else tuple(_frs)

    import torch

    from morphohand.rl.deploy import (
        run_env_overrides,
        act, act_b, build_actor, ckpt_obs_dim, finger_ctrl_from_keyframe, make_env_cfg,
    )
    from morphohand.tools.video_paths import tmp_dir

    run = args.morphology_run.resolve()
    summ = json.loads((run / "summary.json").read_text())
    frozen = run / "frozen_scene.xml"
    if args.closed_ctrl_from_keyframe:
        bfc = finger_ctrl_from_keyframe(frozen, args.closed_ctrl_from_keyframe)
    else:
        bfc = tuple(float(v)
                    for v in np.load(run / "best_rollout.npz")["best_finger_ctrl"].reshape(-1))

    obs_dim = ckpt_obs_dim(args.policy)
    is_b = obs_dim == 66
    trained = run_env_overrides(args.policy)
    trained_from = int(trained.get("finger_residual_active_from_step", 0))
    residual_from = args.residual_from_step if args.residual_from_step is not None else trained_from
    reorient_from = args.reorient_start_step if args.reorient_start_step is not None \
        else int(trained.get("reorient_start_step", 10))
    contact_run = "scene" if (not trained or "scene_floor" in trained) else "legacy"
    contact = contact_run if args.object_contact == "run" else args.object_contact
    if contact == "legacy":
        _use_legacy_object_contact()
    if trained.get("actor_blind_terms"):
        print(f"[eval] actor blinded to {list(trained['actor_blind_terms'])} (from the run's config)")
    print(f"[eval] env timing from the run's config: residual active from step {residual_from}, "
          f"reorient reward from step {reorient_from}, lift_phase_start_step "
          f"{trained.get('lift_phase_start_step', 'default')}  ({'config.yaml found' if trained else 'NO config.yaml beside the checkpoint'})")
    print(f"[eval] tool contact class: {contact} ({'the scene' if contact == 'scene' else 'MuJoCo defaults'}"
          f"{', the run trained on it' if contact == contact_run else f', the run trained on {contact_run}'})")
    cfg = make_env_cfg(frozen, summ["keyframe"], run, bfc, enable_target_axis=is_b,
                       num_steps=args.steps,
                       finger_residual_scale=args.finger_residual_scale,
                       lift_delta=args.lift_delta,
                       open_finger_from_keyframe=args.open_finger_from_keyframe,
                       num_envs=args.n,
                       finger_residual_active_from_step=residual_from,
                       reorient_start_step=reorient_from,
                       lift_phase_start_step=trained.get("lift_phase_start_step"),
                       actor_blind_terms=tuple(trained.get("actor_blind_terms", ()) or ()),
                       extra_cfg=dict(scene_floor=bool(trained.get("scene_floor", False)),
                                      cube_spawn_x_jitter=args.spawn_jitter_mm * 1e-3,
                                      cube_spawn_y_jitter=args.spawn_jitter_mm * 1e-3,
                                      cube_spawn_yaw_jitter=float(np.radians(args.spawn_yaw_deg)),
                                      friction_dr=bool(args.friction_dr)))
    if args.spawn_jitter_mm or args.spawn_yaw_deg or args.friction_dr:
        print(f"[eval] reset perturbations: spawn +-{args.spawn_jitter_mm:g} mm, yaw +-{args.spawn_yaw_deg:g} deg, "
              f"friction DR {'on' if args.friction_dr else 'off'}")
    env, wrapped, actor = build_actor(cfg, args.policy, tmp_dir("evalsuite"))
    obs_td, _ = wrapped.reset()

    N = args.n
    cos_t = np.zeros((args.steps, N), dtype=np.float32)
    z_t = np.zeros((args.steps, N), dtype=np.float32)
    grip_t = np.zeros((args.steps, N), dtype=np.float32)
    # PER-FINGER, not just the sum. On the opposed hand the whole open question is whether the
    # third finger ever bears load: a two-finger pinch and a three-finger chuck can report the
    # same total while one of them is the degenerate grasp we are trying to leave behind.
    perf_t = np.zeros((args.steps, N, 3), dtype=np.float32)
    # Physical plausibility (2026-09-19): the bench index and middle carry servo housings and
    # cabling the capsules omit, so the index-middle chain clearance is tracked per step; with
    # it the largest raw residual (the +-0.5 rad "budget" is a scale, not a clip), the peak
    # finger joint speed and the commanded-minus-achieved gap the servo model produces.
    clear_t = np.zeros((args.steps, N), dtype=np.float32)
    act_absmax_t = np.zeros((args.steps, N), dtype=np.float32)
    act_gt1_t = np.zeros((args.steps, N), dtype=np.float32)
    jvel_max_t = np.zeros((args.steps, N), dtype=np.float32)
    gap_max_t = np.zeros((args.steps, N), dtype=np.float32)
    from morphohand.rl.terms_common import finger_pair_clearance
    robot = env.unwrapped.scene["robot"]
    finger_j = [i for i, n in enumerate(robot.joint_names)
                if n.split("_")[0] in ("thumb", "index", "middle")]
    # actuator -> joint map through the MjModel, for the commanded-minus-achieved gap
    import mujoco as _mj
    mjm = env.unwrapped.sim.mj_model
    act_ids, qadr = [], []
    for a_id in range(mjm.nu):
        if mjm.actuator_trntype[a_id] != _mj.mjtTrn.mjTRN_JOINT:
            continue
        jn = _mj.mj_id2name(mjm, _mj.mjtObj.mjOBJ_JOINT, int(mjm.actuator_trnid[a_id, 0])) or ""
        if jn.split("/")[-1].split("_")[0] in ("thumb", "index", "middle"):
            act_ids.append(a_id); qadr.append(int(mjm.jnt_qposadr[mjm.actuator_trnid[a_id, 0]]))
    act_ids_t = torch.tensor(act_ids, device=env.unwrapped.device)
    qadr_t = torch.tensor(qadr, device=env.unwrapped.device)
    jr = np.array([mjm.jnt_range[int(mjm.actuator_trnid[a, 0])] for a in act_ids], dtype=np.float32)
    jr_lo = torch.tensor(jr[:, 0], device=env.unwrapped.device)
    jr_hi = torch.tensor(jr[:, 1], device=env.unwrapped.device)
    raw_exc_t = np.zeros((args.steps, N), dtype=np.float32)   # raw command beyond the joint range (rad)

    sensor = "fingertip_cube_contact"
    with torch.no_grad():
        for s in range(args.steps):
            obs = obs_td["actor"]
            actions = act_b(actor, obs_td, args.stochastic) if is_b else act(actor, obs[:, :obs_dim])
            obs_td, *_ = wrapped.step(actions)
            pose = env.unwrapped.scene["cube"].data.root_link_pose_w      # (N, 7)
            qx, qy = pose[:, 4], pose[:, 5]
            cos_t[s] = (1.0 - 2.0 * (qx * qx + qy * qy)).cpu().numpy()
            z_t[s] = pose[:, 2].cpu().numpy()
            f = env.unwrapped.scene.sensors[sensor].data.force
            if f is not None:
                mag = f.norm(dim=-1)                                  # (N, n_tips)
                grip_t[s] = mag.sum(dim=-1).cpu().numpy()
                perf_t[s, :, :mag.shape[1]] = mag[:, :3].cpu().numpy()
            clear_t[s] = finger_pair_clearance(env.unwrapped).cpu().numpy()
            a = actions.abs()
            act_absmax_t[s] = a.amax(dim=-1).cpu().numpy()
            act_gt1_t[s] = (a > 1.0).float().mean(dim=-1).cpu().numpy()
            jvel_max_t[s] = robot.data.joint_vel[:, finger_j].abs().amax(dim=-1).cpu().numpy()
            try:
                ctrl_t = env.unwrapped.sim.data.ctrl[:, act_ids_t]
                q_t = env.unwrapped.sim.data.qpos[:, qadr_t]
                clamped = torch.maximum(torch.minimum(ctrl_t, jr_hi), jr_lo)
                gap_max_t[s] = (clamped - q_t).abs().amax(dim=-1).cpu().numpy()
                raw_exc_t[s] = ((ctrl_t - clamped).abs()).amax(dim=-1).cpu().numpy()
            except Exception:
                gap_max_t[s] = np.nan
    step_dt = float(getattr(env.unwrapped, "step_dt", 0.02))
    env.close()

    held_t = (grip_t > args.held_min_n) & (z_t > args.floor_z)      # (T, N)
    arrays = dict(cos=cos_t, z=z_t, grip=grip_t, held=held_t.astype(np.uint8), pads=perf_t, clearance=clear_t,
                  act_absmax=act_absmax_t, act_gt1=act_gt1_t, jvel_max=jvel_max_t, gap_max=gap_max_t,
                  raw_exc=raw_exc_t)
    s = summarize(arrays, args.align_thresh, args.held_min_n, args.floor_z, residual_from)

    def stat(a):
        a = np.asarray(a, dtype=float)
        a = a[~np.isnan(a)]
        if a.size == 0:
            return "—"
        return f"{a.mean():.3f} ± {a.std():.3f}  [{a.min():.3f}, {a.max():.3f}]"

    label = args.label or _run_dir(args.policy).name
    ever_aligned = (cos_t >= args.align_thresh).any(axis=0)
    held_final = held_t[-1]
    t_align = np.where(ever_aligned, (cos_t >= args.align_thresh).argmax(axis=0), np.nan)
    print(f"\n┌── eval suite: {label}   N={N} rollouts x {args.steps} steps")
    print(f"│  align_rate  (ever cos>={args.align_thresh})   "
          f"{s['align_rate']:6.1%}  ({int(ever_aligned.sum())}/{N})")
    print(f"│  HOLD_RATE   (still held at end)      "
          f"{s['hold_rate']:6.1%}  ({int(held_final.sum())}/{N})")
    print(f"│  success     (aligned AND held)       {s['success_rate']:6.1%}")
    print(f"│  t_align     (steps to threshold)     {stat(t_align)}")
    print(f"│  hold_steps  (aligned+held run)       {s['hold_steps_mean']:.3f} ± {s['hold_steps_sd']:.3f}")
    print(f"│  final_cos                            {stat(cos_t[-1])}")
    print(f"│  final_z                              {stat(z_t[-1])}")
    print(f"│  peak_cos                             {stat(cos_t.max(axis=0))}")
    if s["n_lost"]:
        print(f"│  drop_step   ({s['n_lost']} envs lost it)      {s['drop_step_mean']:.3f} ± {s['drop_step_sd']:.3f}")
    if ((cos_t >= args.align_thresh) & held_t).any():
        print(f"│  three_finger (of aligned+held steps) {s['three_finger_share']:6.1%}")
        print(f"│  mean N thumb/index/middle            "
              f"{s['mean_force_thumb']:.1f}/{s['mean_force_index']:.1f}/{s['mean_force_middle']:.1f}")
    else:
        print("│  three_finger                         — (never aligned AND held)")
    print(f"│  plausibility (steps >= {s['residual_from_step']}):")
    print(f"│    index-middle clearance min      {s['clearance_min_mm_mean']:.3f} ± {s['clearance_min_mm_sd']:.3f} "
          f"(worst {s['clearance_min_mm_worst']:.3f}) mm   (at end {s['clearance_end_mm_mean']:.1f} mm)")
    print(f"│    pad peak N thumb/index/middle   {s['pad_peak_thumb']:.1f}/{s['pad_peak_index']:.1f}/{s['pad_peak_middle']:.1f}")
    print(f"│    finger joint speed p99          {s['joint_speed_p99_deg_s']:.0f} deg/s")
    print(f"│    residual |a|>1 fraction / max   {s['residual_gt1_frac']:.3f} / {s['residual_absmax']:.2f}")
    print(f"│    commanded-achieved gap max      {s['ctrl_gap_max_deg']:.1f} deg  (command clamped to the joint range; "
          f"raw command beyond the range up to {s['raw_cmd_beyond_range_deg']:.0f} deg)")
    print("└" + "─" * 60)

    meta = dict(
        label=label, checkpoint=_rel(args.policy), run=_rel(_run_dir(args.policy)), morphology_run=_rel(run),
        n=N, steps=args.steps, control_dt_s=step_dt, align_thresh=args.align_thresh, held_min_n=args.held_min_n,
        floor_z=args.floor_z, residual_from_step=int(residual_from), run_residual_from_step=trained_from,
        reorient_start_step=int(reorient_from), lift_phase_start_step=trained.get("lift_phase_start_step"),
        finger_residual_scale=args.finger_residual_scale, run_finger_residual_scale=trained.get("finger_residual_scale"),
        clip_actions=trained.get("clip_actions"), actor_blind_terms=list(trained.get("actor_blind_terms", ()) or ()),
        scene_floor=bool(trained.get("scene_floor", False)), object_contact=contact, run_object_contact=contact_run,
        lift_delta=args.lift_delta, closed_ctrl_from_keyframe=args.closed_ctrl_from_keyframe,
        open_finger_from_keyframe=bool(args.open_finger_from_keyframe),
        spawn_jitter_mm=float(args.spawn_jitter_mm), spawn_yaw_deg=float(args.spawn_yaw_deg),
        friction_dr=bool(args.friction_dr), stochastic=bool(args.stochastic),
        evaluated=time.strftime("%Y-%m-%d %H:%M"), json=_rel(args.json_out) if args.json_out else None, **_git_state())
    traces_path = None
    if not args.no_traces:
        traces_path = args.traces
        if traces_path is None:
            tdir = (args.json_out.parent if args.json_out.parent.name == "traces" else args.json_out.parent / "traces") \
                if args.json_out else ROOT / "logs" / "eval_traces"
            traces_path = tdir / f"{_run_dir(args.policy).name}.npz"
        key = args.trace_key or _default_key(args, residual_from, trained_from, contact, contact_run)
        key = save_trace(traces_path, key, arrays, meta)
        print(f"[eval] traces -> {traces_path}:{key}")

    out = dict(label=label, **s, spawn_jitter_mm=float(args.spawn_jitter_mm), spawn_yaw_deg=float(args.spawn_yaw_deg),
               friction_dr=bool(args.friction_dr), stochastic=bool(args.stochastic), object_contact=contact,
               policy=meta["checkpoint"], morphology_run=meta["morphology_run"], evaluated=meta["evaluated"],
               traces=f"{_rel(traces_path)}:{key}" if traces_path else None)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(out, indent=2))
        print(f"[eval] -> {args.json_out}")

    if args.plot:
        _plot(args.plot, label, cos_t, z_t, grip_t, held_t, args)
        print(f"[eval] -> {args.plot}")


def _plot(path: Path, label, cos_t, z_t, grip_t, held_t, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T, N = cos_t.shape
    t = np.arange(T)
    fig, ax = plt.subplots(4, 1, figsize=(9, 11), sharex=True)

    def band(a, axis, color, ylabel, hline=None):
        med = np.median(a, axis=1)
        lo, hi = np.percentile(a, 10, axis=1), np.percentile(a, 90, axis=1)
        axis.fill_between(t, lo, hi, color=color, alpha=0.25, label="10-90%")
        axis.plot(t, med, color=color, lw=2, label="median")
        if hline is not None:
            axis.axhline(hline, ls="--", lw=1, color="k", alpha=0.6)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.3)

    band(cos_t, ax[0], "tab:blue", "alignment cos", args.align_thresh)
    ax[0].set_title(f"{label}   N={N} rollouts")
    ax[0].legend(loc="lower right", fontsize=8)
    band(z_t, ax[1], "tab:green", "object z (m)", args.floor_z)
    band(grip_t, ax[2], "tab:orange", "total grip (N)", args.held_min_n)
    ax[3].plot(t, held_t.mean(axis=1), color="tab:red", lw=2)
    ax[3].set_ylabel("fraction HELD")
    ax[3].set_xlabel("policy step")
    ax[3].set_ylim(-0.05, 1.05)
    ax[3].grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
