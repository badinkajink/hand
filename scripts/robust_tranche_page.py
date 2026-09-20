#!/usr/bin/env python3
"""Build the page for the 2026-09-20 continuation ("robust") tranche from the queue's own outputs.

    python3 scripts/robust_tranche_page.py [--root docs/experiments/20260920-robust_tranche]
                                           [--parent docs/experiments/20260919-hands_tranche]

Reads queue.json, tranche_results.json, <id>_eval.json, <id>_eval_j3dr.json, <id>_strip.png,
<id>_chain_pl{25,0}.json and each run's tensorboard scalars; the parent tranche's results and its
robust/<id>_eval_j3dr.json for the comparison; writes 20260920-robust_reorient_policies.html beside
the queue (the rollout videos are web/<id>.mp4 beside the page, transcoded from videos/).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hands_tranche_page import ARM_LABEL, cellc, curves, f1, plot_curves, table, uri, uri_jpeg  # noqa: E402

PLANT_ROWS = [
    ("tool spawn", "one nominal pose", "&#177;3 mm in x and y, &#177;10&#176; of yaw", "a policy trained at one pose has no reason to hold elsewhere; the CPU probe and the chain never start at the trained pose to the millimetre"),
    ("tool", "25 mm cylinder, 24.5 g", "+ screw-tip mesh, 25.6 g", "the chain&#8217;s tool and the real screwdriver carry it"),
    ("floor", "mjlab plane: &#956; 1 / 0.005 / 0.0001, solref 0.02, solimp 0.9&#8211;0.95", "the scene&#8217;s floor: &#956; 1.8 / 0.15 / 0.01, solref 0.006, solimp 0.97&#8211;0.995", "the tool lies on it through the closure and the start of the lift (<code>--scene-floor</code>)"),
    ("tool contact class", "MuJoCo defaults (solref 0.02, solimp 0.9&#8211;0.95); the pair with a pad averaged to 0.013 / 0.935", "the scene&#8217;s (0.006, 0.97&#8211;0.995), the same as the pads", "<code>make_object_spec_from_frozen</code> copied only type, size, mass and friction since the first mjlab run"),
]


# --- the pipeline section --------------------------------------------------------------------
# Fixed reward weights and shapes that the trainer does not expose as flags (env_build._build_rewards);
# the flagged ones are read from the run's config.yaml so the table cannot drift from the run.
REWARD_FIXED = [
    ("track_finger_qpos", "4", "exp(&#8722;20&#183;|q &#8722; q_ref|&#178;) against the open-loop rollout of the fitted grip", "all episode"),
    ("track_object_pos", "6", "exp(&#8722;200&#183;|p &#8722; p_ref|&#178;)", "all episode"),
    ("track_object_quat", "2", "exp(&#8722;10&#183;quat error&#178;)", "all episode"),
    ("track_finger_ctrl_anchor", "1", "exp(&#8722;4&#183;|ctrl &#8722; anchor|&#178;), keeps the set-point near the grip", "all episode"),
    ("contact_mean", "10", "mean pad-contact indicator", "all episode"),
    ("lift_height", "80", "object height above spawn, clipped at the 100 mm target", "lift phase"),
    ("object_drop", "&#8722;12", "indicator: object 20 mm below spawn", "lift phase"),
    ("fingertip_to_object", "&#8722;3", "mean pad-to-shaft distance", "all episode"),
    ("joint_pos_limits", "&#8722;2", "joint excursion beyond its range", "all episode"),
]

DECISIONS = [
    ("Hand provenance", "eight deployed real_v1 hands (D1&#8211;D8) from the 8,198-hand Sobol population; sv1_*, g12, rv05", "2026-08-27/28",
     "only hands generated from <code>assets/mjcf/real_v1/real_hand.xml</code> and exportable through the plan pipeline count; all eight were run on hardware (251 runs)", "measured"),
    ("Grasp", "chain-fitter anchor, <code>open_ik</code> keyframe, pads at or below the shaft equator, 10 mm commanded squeeze", "2026-08-27, 2026-09-16",
     "pads above the equator wedge the shaft out in 0.6 s of standing still (SLIP.md); at kp 0.5 a 2 mm squeeze lifts at 0.16&#8211;1.43 N, 10 mm at 2.5&#8211;4.5 N on 3&#8211;4 pads, which is what the exported plans carry", "measured"),
    ("Finger plant", "kp 0.5, kv 0.02, forcerange 0.35 N&#183;m, frictionloss 0.0035, elliptic cone with impratio 10, pad &#956; 1.0, palm plate +25 mm", "2026-09-02, 2026-09-16",
     "the shipped kp 30 predicts a +0.03&#176; deficit where the bench measures +11.7&#176;; the corrected plant drops 6 % of jittered trials against the bench&#8217;s 8 %", "measured (servo sysid, drop gate)"),
    ("Tool and floor", "screw-tip mesh (25.6 g), the scene&#8217;s floor and contact class inside mjlab", "2026-09-20",
     "the object entity had trained on MuJoCo&#8217;s default solref/solimp on mjlab&#8217;s plane since the first mjlab run; the chain and the bench carry the tip", "measured (this page)"),
    ("Control", "9 finger residuals on a set-point that eases (quadratic ease-out) from the open keyframe to the grip anchor over 80 sim steps; residual = 0.5 rad &#215; action, action clipped to &#177;2", "scale 0.5: 2026-07; clip 2.0: 2026-09-19",
     "action 0 reproduces the open-loop grasp and lift (lesson 6); 0.5 rad was Policy B&#8217;s budget hardcoded into every screen and the bench manoeuvres wanted more; at clip 2.0 every hand held 64/64 where the &#177;0.5 rad parents had not turned", "clip measured; scale inherited from the July deploy-parity setting"),
    ("Residual onset", "step 58 (sim step 580): 0.52 s after the 80-step lift ramp ends; the reorient reward and the tip-lost termination start there too", "2026-08-30",
     "the tool settles on the pads after the lift (4.9 mm of servo settling at 0.52 s measured in the chain on 2026-09-20)", "inherited; the settling measurement is consistent with it"),
    ("Reorient reward", "alignment 100 &#215; exp(&#8722;4(1 &#8722; cos)&#178;) + progress 300 &#215; &#916;cos per step, contact_min 15, lateral drift &#8722;8", "2026-06 (B10 block, b33 lineage)",
     "the progress term gives a dense gradient toward vertical where the exponential alone is flat; the block is the one the m05 handoff converged under", "inherited; never re-swept on real_v1"),
    ("Grip penalty", "&#8722;2 &#215; mean over pads of (max(F &#8722; 6 N, 0) / 4)&#178;", "2026-09-19 20:30",
     "the 10.8 N finetune (&#8722;5 above 4 N) turned to 0.91 in the chain and dropped at 3.2 s; the 17 N parent held 3/4, so the tranche keeps the parent&#8217;s grip and penalises only above 6 N", "measured in the chain; the value was not swept"),
    ("Separation term", "&#8722;10 &#215; (max(30 mm &#8722; clearance, 0) / 30 mm)&#178; on the index&#8211;middle link chains (the &#8220;separation&#8221; arm)", "2026-09-19",
     "the bench index and middle carry servo housings and cabling the capsules omit; policies that run them together fail on hardware. It turns D1/D3/D5/D6 with more clearance and kills the turn on D4 (&#8722;3.6 mm) and D7 (1.1 mm)", "measured; the 30 mm threshold is not calipered"),
    ("Terminations", "drop 20 mm below spawn; fewer than 3 pads for 15 consecutive steps; the slip guards at 0.5 m / 10 rad / 100 rad never fire", "recipe 2026-07; 15 steps since 2026-08-30",
     "the recipe lets only drop and tip-lost end an episode (lesson 8: never strip them). The recipe&#8217;s 3 steps became 15 in <code>train_real_v1_blind_pair.sh</code>; no measurement behind the 15 is on record", "inherited"),
    ("Domain randomisation", "friction &#215;[0.55, 1.15] on every geom per reset; observation noise &#177;0.01 rad, &#177;0.5 rad/s, &#177;5 mm; spawn &#177;3 mm / &#177;10&#176; (this tranche)", "friction and noise 2026-07; spawn 2026-09-20",
     "the 2026-09-19 policies collapse at &#177;3 mm (D7 0/64, D6 s1 8/64). Compliance DR is off since 2026-07-10: it relocated the policy to the band&#8217;s middle instead of broadening it", "spawn measured; friction band and noise inherited"),
    ("Observation", "sighted 66-dim actor (object pose in the palm frame included); critic sees the same terms without noise", "2026-08-30 (asymmetric), 2026-09-19 (sighted)",
     "a blind warm start had not recovered in 22 iterations and the bench tag tracker supplies the object pose at 50 Hz; blinding keeps the vector 66 wide (scale 0) so the warm start still loads", "measured once"),
    ("Optimiser", "PPO (rsl_rl), 3072 envs &#215; 24 steps per iteration, 5 epochs &#215; 4 minibatches, lr 5&#183;10&#8315;&#8308; adaptive on KL 0.01, &#947; 0.99, &#955; 0.95, clip 0.2, entropy 0.005, initial std 0.3, MLP 512-256-128 ELU", "2026-07",
     "the GPU is compute-saturated at ~10 k steps/s and num_envs buys nothing; the rest are rsl_rl defaults", "inherited"),
    ("Budget", "20 M steps (271 iterations, ~36 min) per finetune; 60 M for the from-scratch D6 parent", "2026-09-17/19",
     "the 60 M parent was still gaining +8 % alignment per 160 iterations; the 20 M grip finetune moved cos 0.933 &#8594; 0.969", "measured once; continued policies here end 30&#8211;45&#176; short, so 20 M on the harder plant is the open item"),
    ("Warm start", "actor and critic from the parent checkpoint, shape-partial load", "2026-09-17",
     "from-scratch reorient draws have sd 0.3&#8211;0.5 in cos, warm-started ones 0.032 (b33 band); an actor-only warm start lets a fresh critic knock the actor off its optimum", "measured (b33 band, gotcha 8)"),
    ("Seeding", "none: <code>--seed</code> is parsed and never applied (its only mention is the argument itself), mjlab&#8217;s environment seed is None", "2026-07",
     "found while writing this register; the GPU contact solver is non-deterministic in any case, so two runs of one recipe have always been two draws (the two 60 M D6 runs differ by 0.29 in cos)", "defect; seed the trainer after the queue and keep evaluating as a distribution"),
    ("Timing", "5 s episodes, 2 ms physics, control at 50 Hz (decimation 10)", "2026-07",
     "the bench servo bus closes a loop at 50&#8211;111 Hz", "inherited"),
]

EVAL_ROWS = [
    ("nominal", "<code>policy_eval_suite.py</code>, 64 GPU rollouts, deterministic actor, the run&#8217;s own onset and clip", "held = total pad force &gt; 0.5 N and object above the floor at the last step; a shaft standing on the floor reads cos 1 and is not a hold"),
    ("jittered", "the same with <code>--spawn-jitter-mm 3 --spawn-yaw-deg 10 --friction-dr</code>", "the judging number: 64 deterministic rollouts at one pose are one rollout under solver noise"),
    ("CPU replay and variants", "<code>policy_transfer_probe.py</code> on the frozen scene: base, plate 0, &#956; 0.6 / 1.5, mass &#215;1.3, kp 0.25", "the CPU solver is the chain&#8217;s; a GPU/CPU disagreement is a fragility, not a bug"),
    ("chain", "<code>real_v1_chain_policy.py --stages policy</code>: UR5e lift on the menagerie arm at kp &#215;10, finger gravcomp off, the policy shadowed from step 0, plates 25 and 0 mm", "the first active observation matches the benchmark within 1&#176; / 1.3 mm / 0.015 quat; what remains is the maneuver"),
    ("filmstrip and video", "<code>policy_filmstrip.py</code> frames at steps 40, 58, 80, 110, 140, 170, 200, 249; <code>rl_render_rollout.py</code> 640&#215;480", "looked at before any number is explained"),
]


def num(v):
    """Number with a true minus sign."""
    return f"{float(v):g}".replace("-", "&#8722;")


def pipeline_blocks(run_dir, common_flags=()):
    """Timeline, observation/action, reward, termination and PPO tables for the pipeline section, read
    from one finished run's config.yaml and rsl_rl_cfg.json so every number is the run's own."""
    import yaml
    cfg = yaml.safe_load(open(os.path.join(run_dir, "config.yaml")))
    env, ppo = cfg["env"], cfg["ppo"]
    rr = json.load(open(os.path.join(run_dir, "rsl_rl_cfg.json")))
    dec = int(env["decimation"]); dt = float(env["sim_timestep"]) * dec
    ep_steps = int(round(float(env["episode_length_s"]) / dt))
    close, settle, ramp = int(env["finger_close_sim_steps"]), int(env["settle_steps"]), int(env["lift_ramp_steps"])
    onset = int(env["finger_residual_active_from_step"]); onset_sim = onset * dec
    total_sim = ep_steps * dec

    # --- episode timeline (SVG): the first `onset_sim` steps get 55 % of the width, the turn the rest
    W, L, Rm, H = 760, 100, 16, 150
    split = 0.55
    def x(sim):
        if sim <= onset_sim:
            return L + (W - L - Rm) * split * sim / onset_sim
        return L + (W - L - Rm) * (split + (1 - split) * (sim - onset_sim) / (total_sim - onset_sim))
    phases = [(0, close, "close", ".45"), (close, settle, "seat", ".18"), (settle, settle + ramp, "lift", ".45"),
              (settle + ramp, onset_sim, "settle", ".18"), (onset_sim, total_sim, "residual on, reorient reward on, tip-lost live", ".32")]
    svg = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="episode timeline" style="max-width:{W}px;font-family:inherit;color:inherit">']
    for a, b, lab, op in phases:
        svg.append(f'<rect x="{x(a):.1f}" y="40" width="{x(b) - x(a):.1f}" height="34" fill="currentColor" fill-opacity="{op}" stroke="currentColor" stroke-opacity=".5" stroke-width="0.6"/>')
        svg.append(f'<text x="{(x(a) + x(b)) / 2:.1f}" y="61" font-size="11" text-anchor="middle" fill="currentColor">{lab}</text>')
    svg.append(f'<text x="{x(onset_sim):.1f}" y="30" font-size="10" text-anchor="middle" fill="currentColor">scale changes here</text>')
    svg.append(f'<path d="M{x(onset_sim) - 4:.1f},74 l4,8 l4,-8" fill="none" stroke="currentColor" stroke-width="0.8"/>')
    for sim in (0, close, settle, settle + ramp, onset_sim, total_sim):
        svg.append(f'<line x1="{x(sim):.1f}" y1="74" x2="{x(sim):.1f}" y2="82" stroke="currentColor" stroke-opacity=".6" stroke-width="0.8"/>')
        svg.append(f'<text x="{x(sim):.1f}" y="96" font-size="10" text-anchor="middle" fill="currentColor" fill-opacity=".8">{sim}</text>')
        svg.append(f'<text x="{x(sim):.1f}" y="112" font-size="10" text-anchor="middle" fill="currentColor" fill-opacity=".8">{sim // dec}</text>')
        svg.append(f'<text x="{x(sim):.1f}" y="128" font-size="10" text-anchor="middle" fill="currentColor" fill-opacity=".8">{sim * float(env["sim_timestep"]):.2f} s</text>')
    for yy, lab in ((96, "sim step"), (112, "policy step"), (128, "time")):
        svg.append(f'<text x="{L - 18}" y="{yy}" font-size="10" text-anchor="end" fill="currentColor" fill-opacity=".8">{lab}</text>')
    svg.append("</svg>")
    timeline = "".join(svg)

    # --- observation and action
    n_joint = 15
    obs_rows = [
        ("joint_pos", str(n_joint), "9 finger + 6 palm joint positions relative to default", "&#177;0.01 rad"),
        ("joint_vel", str(n_joint), "their velocities", "&#177;0.5 rad/s"),
        ("object_pos", "3", "tool position relative to the palm site", "&#177;5 mm"),
        ("object_pose_actual", "7", "tool position and quaternion in the palm frame", "&#177;5 mm"),
        ("ref_finger_qpos", "9", "finger joints of the open-loop reference rollout at this step", "none"),
        ("ref_object_pose", "7", "tool pose of the reference rollout at this step", "none"),
        ("actions", "9", "the actor&#8217;s previous output", "none"),
        ("target_axis_misalign", "1", "1 &#8722; cos between the shaft axis and vertical", "none"),
    ]
    obs_table = table(["term", "dims", "meaning", "actor noise (uniform)"], [list(r) for r in obs_rows])
    act = (f"<p>The actor is an MLP {'-'.join(str(d) for d in rr['actor']['hidden_dims'])} ({rr['actor']['activation']}) from the 66 inputs to 9 Gaussian "
           f"means with a learned per-dimension standard deviation (initial {rr['actor']['distribution_cfg']['init_std']}); the critic has the same shape and reads the "
           f"same 66 terms without noise. Each action is clipped to &#177;{ppo['clip_actions']} by the rsl_rl wrapper and multiplied by "
           f"{env['finger_residual_scale']} rad before it is added to the finger set-point, so the residual is bounded at "
           f"&#177;{float(ppo['clip_actions']) * float(env['finger_residual_scale']):.1f} rad and is zero before policy step {onset}. "
           f"The palm has no learned degree of freedom: its six joints follow the scripted lift (<code>enable_palm_rotation_residual</code> {str(env['enable_palm_rotation_residual']).lower()}).</p>")

    # --- rewards: flagged weights from the run, fixed ones from env_build
    rows = [
        ("target_axis_alignment", num(env['target_axis_weight']), f"exp(&#8722;{env['target_axis_alpha']:g}(1 &#8722; cos)&#178;), cos between the shaft axis and world +z", f"from step {env['reorient_start_step']}"),
        ("target_axis_progress", num(env['target_axis_progress_weight']), "cos(t) &#8722; cos(t &#8722; 1), signed", f"from step {env['reorient_start_step']}"),
        ("contact_min", num(env['contact_min_weight']), f"indicator: all {env['min_tips_in_contact']} pads on the tool", "all episode"),
        ("object_lateral_drift", num(env['lateral_drift_weight']), f"|xy &#8722; xy_spawn| beyond a {float(env['lateral_drift_deadband']) * 1000:g} mm deadband, power {env['lateral_drift_power']:g}, contact-gated", "all episode"),
        ("object_xy_drift", num(env['object_xy_drift_weight']), "xy drift from the reference, contact-gated", "all episode"),
        ("object_orientation_drift", num(env['object_orientation_drift_weight']), "orientation drift from the reference, contact-gated", "all episode"),
        ("finger_drift_from_grip", num(env['finger_drift_weight']), "finger joints away from the grip pose, contact-gated", "all episode"),
        ("action_rate_l2", num(env['action_rate_weight']), "|a(t) &#8722; a(t &#8722; 1)|&#178;", "all episode"),
        ("grip_force_excess", num(env['grip_force_penalty_weight']), f"mean over pads of (max(F &#8722; {env['grip_force_penalty_thresh']:g} N, 0) / {env['grip_force_penalty_scale']:g})&#178;", "all episode"),
    ]
    rows.append(("finger_separation_penalty", "0 in the clip arm; &#8722;10 in the separation arm",
                 "(max(d_min &#8722; clearance, 0) / d_min)&#178; on the index&#8211;middle link chains, d_min 30 mm", "all episode"))
    rows += [list(r) for r in REWARD_FIXED]
    reward_table = table(["term", "weight", "shape", "active"], [list(r) for r in rows])
    reward_note = (f"<p>mjlab multiplies every term by the control period ({dt:g} s), so a weight of 100 is 2 per step. The tracking terms follow the "
                   f"open-loop rollout of the fitted grip on the same scene (<code>best_rollout.npz</code> in the morphology run); the contact gate zeroes a "
                   f"stability term while the mean pad-contact indicator is below {env['contact_gate_min']:g}.</p>")

    # --- terminations
    def flag(name, default):
        cf = list(common_flags)
        return cf[cf.index(name) + 1] if name in cf else default
    term_rows = [
        ("time_out", f"{env['episode_length_s']:g} s ({ep_steps} steps)", "the normal end"),
        ("object_drop", f"tool {float(env['term_object_drop']) * 1000:g} mm below its spawn height during the lift phase (from step {env['lift_phase_start_step']})", "ends the episode"),
        ("tip_lost", f"fewer than {env['min_tips_in_contact']} pads on the tool for {env['term_tip_lost_steps']} consecutive steps, from step {env['lift_phase_start_step']}", "ends the episode"),
        ("object_slip / orientation_slip / finger_slip", f"{env['term_object_slip_xy']:g} m / {env['term_object_slip_yaw']:g} rad / {env['term_finger_slip']:g} rad", "thresholds the recipe set out of reach: they never fire"),
        ("collapse watchdog", f"the trainer aborts when the episode-mean object height falls below {flag('--watchdog-collapse-z', '0.03')} m after iteration {flag('--watchdog-from-iter', '40')}", "a run that has dropped the tool stops costing GPU time"),
    ]
    term_table = table(["termination", "condition", "effect"], [list(r) for r in term_rows])

    # --- PPO
    al = rr["algorithm"]
    ppo_rows = [
        ("environments &#215; steps per iteration", f"{ppo['num_envs']} &#215; {ppo['num_steps_per_env']} = {ppo['num_envs'] * ppo['num_steps_per_env']:,} samples"),
        ("iterations", f"{rr['max_iterations']} for {ppo['total_timesteps'] / 1e6:g} M steps"),
        ("epochs &#215; minibatches", f"{al['num_learning_epochs']} &#215; {al['num_mini_batches']}"),
        ("learning rate", f"{al['learning_rate']:g}, {al['schedule']} on a KL target of {al['desired_kl']:g}"),
        ("&#947;, &#955;, clip, entropy", f"{al['gamma']:g}, {al['lam']:g}, {al['clip_param']:g}, {al['entropy_coef']:g}"),
        ("value loss, gradient clip", f"{al['value_loss_coef']:g} (clipped), {al['max_grad_norm']:g}"),
        ("checkpoints", f"every {rr['save_interval']} iterations; <code>model_{rr['max_iterations'] - 1}.pt</code> is the one evaluated"),
        ("seed", "none is applied: <code>rl_train_cube.py</code> parses <code>--seed</code> and never uses it, mjlab&#8217;s environment seed stays None and "
                 f"the rsl_rl config field ({rr['seed']}) is not read by the runner; every run, s0/s1/s2 included, is an unseeded draw"),
    ]
    ppo_table = table(["PPO", ""], [list(r) for r in ppo_rows])

    return {"TIMELINE": timeline, "TABLE_OBS": obs_table, "ACTION_NOTE": act, "TABLE_REWARD": reward_table, "REWARD_NOTE": reward_note,
            "TABLE_TERM": term_table, "TABLE_PPO": ppo_table,
            "TABLE_DECISIONS": table(["decision", "value", "since", "reason on record", "status"], [list(r) for r in DECISIONS]),
            "TABLE_EVAL": table(["number", "how it is produced", "what it means"], [list(r) for r in EVAL_ROWS])}


def chain_cell(path):
    if not os.path.exists(path):
        return ("&#8211;", "cell")
    c = json.load(open(path)); tr = c.get("policy_trace", [])
    if not tr:
        return ("&#8211;", "cell")
    end = tr[-1]; peak = max(r[1] for r in tr)
    held = end[2] > 0.06 and sum(end[4:7]) >= 0.5
    return (f"{'held' if held else 'lost'} &#183; {end[1]:+.2f} (peak {peak:.2f})", "cell c4" if held else "cell c0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="docs/experiments/20260920-robust_tranche")
    ap.add_argument("--parent", default="docs/experiments/20260919-hands_tranche")
    a = ap.parse_args()
    R, P = a.root, a.parent
    q = json.load(open(os.path.join(R, "queue.json")))
    res = {r["id"]: r for r in (json.load(open(os.path.join(R, "tranche_results.json"))) if os.path.exists(os.path.join(R, "tranche_results.json")) else [])}
    pres = {r["id"]: r for r in json.load(open(os.path.join(P, "tranche_results.json")))}
    jobs = q["jobs"]
    done = [j for j in jobs if res.get(j["id"], {}).get("status") == "done"]

    def jit(root, jid, sub=""):
        for p in (os.path.join(root, sub, f"{jid}_eval_j3dr.json"), os.path.join(root, f"{jid}_eval_j3dr.json")):
            if os.path.exists(p):
                return json.load(open(p))
        return None

    def hc(e, key="hold_rate", cos="final_cos_mean"):
        if not e or e.get(key) is None:
            return ("&#8211;", "cell")
        n = int(e.get("n", 64)); h = int(round(e[key] * n))
        return cellc(h / n, f"{h}/{n} &#183; {e[cos]:+.2f}")

    # --- main table: continuation against its parent
    rows, reading_bits = [], []
    for j in done:
        r = res[j["id"]]; pr = pres.get(j.get("warm_start", ""), {})
        pj = jit(P, j.get("warm_start", ""), "robust"); cj = jit(R, j["id"])
        rows.append([j["hand"], ARM_LABEL.get(j["arm"], j["arm"]),
                     hc(pr), hc(pj), f1(pr.get("clearance_min_mm_mean")),
                     hc(r), hc(cj), f1(r.get("clearance_min_mm_mean")),
                     chain_cell(os.path.join(P, f"{j.get('warm_start', '')}_chain_pl25.json")),
                     chain_cell(os.path.join(R, f"{j['id']}_chain_pl25.json"))])
        if pj and cj:
            reading_bits.append((j["hand"], j["arm"], int(round(pj["hold_rate"] * 64)), int(round(cj["hold_rate"] * 64)),
                                 pj["final_cos_mean"], cj["final_cos_mean"]))
    for j in jobs:
        if j.get("status") == "failed":
            rows.append([j["hand"], ARM_LABEL.get(j["arm"], j["arm"]), "", "", "", ("training aborted: " + j.get("fail_note", "see the log"), "cell c0"), "", "", "", ""])
    table_main = table(["hand", "arm", "parent nominal: held &#183; cos", "parent jittered", "parent clearance (mm)",
                        "continued nominal", "continued jittered", "continued clearance (mm)",
                        "chain, parent", "chain, continued"], rows) if rows else "<p>No job has finished yet.</p>"
    up = [b for b in reading_bits if b[3] >= b[2] + 8]; down = [b for b in reading_bits if b[3] <= b[2] - 8]
    reading = ""
    if reading_bits:
        reading = ("<p>Jittered hold, continued against parent: " +
                   "; ".join(f"{h} {ARM_LABEL.get(a_, a_)} {p}&#8594;{c} of 64 (cos {pc:+.2f}&#8594;{cc:+.2f})" for h, a_, p, c, pc, cc in reading_bits) +
                   ". " + (f"Gains of 8 or more: {', '.join(b[0] for b in up)}. " if up else "") +
                   (f"Losses of 8 or more: {', '.join(b[0] for b in down)}. " if down else "") + "</p>")

    # --- plausibility
    prow = []
    for j in done:
        r = res[j["id"]]
        flags = []
        if max(r.get("pad_peak_thumb", 0), r.get("pad_peak_index", 0), r.get("pad_peak_middle", 0)) > 9: flags.append("pad peak &gt; 9 N")
        if (r.get("ctrl_gap_max_deg") or 0) > 60: flags.append("gap &gt; 60&#176;")
        if (r.get("clearance_min_mm_mean") or 99) < 10: flags.append("clearance &lt; 10 mm")
        if r.get("hold_rate", 1) < 1: flags.append("drops")
        prow.append([j["hand"], ARM_LABEL.get(j["arm"], j["arm"]),
                     f"{f1(r.get('pad_peak_thumb'))} / {f1(r.get('pad_peak_index'))} / {f1(r.get('pad_peak_middle'))}",
                     f"{f1(r.get('force_active_thumb'))} / {f1(r.get('force_active_index'))} / {f1(r.get('force_active_middle'))}",
                     f1(r.get("joint_speed_p99_deg_s"), "{:.0f}"), f1(r.get("residual_gt1_frac"), "{:.2f}"),
                     f1(r.get("ctrl_gap_max_deg"), "{:.0f}"), f1(r.get("raw_cmd_beyond_range_deg"), "{:.0f}"),
                     f"{f1(r.get('clearance_min_mm_mean'))} / {f1(r.get('clearance_end_mm_mean'))}",
                     ("; ".join(flags) if flags else "none", "cell c0" if flags else "cell c4")])
    table_plaus = table(["hand", "arm", "pad peak th / ix / md (N)", "pad force active th / ix / md (N)", "#joint speed p99 (&#176;/s)",
                         "#|a| &gt; 1 share", "#gap (&#176;)", "#raw cmd beyond range (&#176;)", "clearance min / end (mm)", "flags"], prow) if prow else "<p>No job has finished yet.</p>"

    # --- chain
    crow = []
    for j in done:
        crow.append([j["hand"], ARM_LABEL.get(j["arm"], j["arm"]),
                     chain_cell(os.path.join(R, f"{j['id']}_chain_pl25.json")), chain_cell(os.path.join(R, f"{j['id']}_chain_pl0.json"))])
    table_chain = table(["hand", "arm", "plate 25 mm: end cos", "plate 0: end cos"], crow) if crow else "<p>No chain run yet.</p>"

    # --- strips + videos (web/<id>.mp4 transcoded from videos/<id>.mp4 if missing)
    strips = []
    os.makedirs(os.path.join(R, "web"), exist_ok=True)
    for j in done:
        sp = os.path.join(R, f"{j['id']}_strip.png")
        if not os.path.exists(sp):
            continue
        src_v, web_v = os.path.join(R, "videos", f"{j['id']}.mp4"), os.path.join(R, "web", f"{j['id']}.mp4")
        if os.path.exists(src_v) and not os.path.exists(web_v):
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src_v, "-vf", "scale=640:480", "-c:v", "libx264", "-crf", "28",
                            "-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", web_v], stdin=subprocess.DEVNULL)
        r = res[j["id"]]; cj = jit(R, j["id"])
        video = (f'<video controls muted loop playsinline preload="metadata" width="640" height="480" src="web/{j["id"]}.mp4"></video>'
                 if os.path.exists(web_v) else "")
        strips.append(f'<figure><img src="{uri_jpeg(sp)}" alt="{j["id"]} filmstrip">{video}<figcaption>{j["hand"]}, '
                      f'{ARM_LABEL.get(j["arm"], j["arm"])}: nominal held {int(round(r["hold_rate"] * 64))}/64 at cos {r["final_cos_mean"]:+.3f}'
                      + (f', jittered {int(round(cj["hold_rate"] * 64))}/64 at {cj["final_cos_mean"]:+.3f}' if cj else "")
                      + f', clearance min {f1(r.get("clearance_min_mm_mean"))} mm. Steps 40, 58 (lifted, residual on), 80, 110, 140, 170, 200, 249.</figcaption></figure>')
    strips_html = "\n".join(strips) if strips else "<p>No filmstrip yet.</p>"

    # --- curves
    runs = {j["id"]: res[j["id"]]["run"] for j in done if res[j["id"]].get("run")}
    keys = ["Episode_Reward/target_axis_alignment", "Episode_Termination/tip_lost", "Metrics/lift_height/object_height", "Episode_Reward/grip_force_excess"]
    cv = curves(runs, keys) if runs else {}
    curves_html = "<p>No run yet.</p>"
    if cv:
        png = os.path.join(R, "20260920-robust_curves.png")
        plot_curves(cv, png)
        curves_html = f'<figure><img src="{uri(png)}" alt="training curves"><figcaption>Per-iteration alignment reward, tip-lost termination share, object height and grip excess for every finished job (10-iteration moving mean).</figcaption></figure>'

    n_done = len(done)
    lede = (f"{n_done} of {len(jobs)} continuation jobs have finished. " +
            (("Jittered hold (64 rollouts at &#177;3 mm / &#177;10&#176; with friction DR), continued against parent: " +
              "; ".join(f"{h} {c} against {p}" for h, a_, p, c, pc, cc in reading_bits) + ". ") if reading_bits else "") +
            ("Chain at plate 25 mm, held by: " + (", ".join(j["hand"] + ("" if j["arm"] == "clip" else " (separation)") for j in done
                                                          if chain_cell(os.path.join(R, f"{j['id']}_chain_pl25.json"))[1].endswith("c4")) or "none") + "."))
    sub = {"LEDE": lede, "TABLE_PLANT": table(["", "2026-09-19 tranche", "here", "why"], [list(r) for r in PLANT_ROWS]),
           "TABLE_MAIN": table_main, "READING": reading, "TABLE_PLAUS": table_plaus, "TABLE_CHAIN": table_chain,
           "STRIPS": strips_html, "CURVES": curves_html, "N_JOBS": str(len(jobs)), "N_DONE": str(n_done),
           "BUILT": time.strftime("%Y-%m-%d %H:%M")}
    sub.update(pipeline_blocks(res[done[0]["id"]]["run"], q.get("common_flags", ())) if done and res[done[0]["id"]].get("run") else
               {k: "<p>No finished run yet.</p>" for k in ("TIMELINE", "TABLE_OBS", "ACTION_NOTE", "TABLE_REWARD", "REWARD_NOTE", "TABLE_TERM", "TABLE_PPO", "TABLE_DECISIONS", "TABLE_EVAL")})
    tpl = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "robust_tranche_page.template.html")).read()
    for k, v in sub.items():
        tpl = tpl.replace("{{" + k + "}}", v)
    left = [ln for ln in tpl.splitlines() if "{{" in ln]
    assert not left, left[:3]
    out = os.path.join(R, "20260920-robust_reorient_policies.html")
    open(out, "w").write(tpl)
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
