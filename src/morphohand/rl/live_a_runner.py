"""Live-A reset runner: train Policy B from Policy A's *organic* delivery.

The A->B handoff seam never closed under any **teleport** reset (skip-lift
bank / normal-lift onset injection, static or Markov-complete — all saturated
at continuous-handoff min-z ~0.002-0.012, far below the 0.05 "held" bar). The
remaining train/deploy difference is the teleport itself: every B training
reset jumps to a snapshot, while at deploy A runs LIVE into the seam (real
contact-solver warmstart, real contact-force ramp, real `last_action`).

This runner removes the teleport entirely. During each B training episode,
**frozen Policy A drives the env live for steps 0..onset** (real physics), then
B's PPO rollout proceeds from the organic seam. The two pieces:

  (1) ENV applies A's action while the per-env step counter < onset. We compute
      frozen A's deterministic action from obs[:, :A_OBS_DIM] and substitute it
      into the action tensor handed to env.step, ignoring B's sampled action for
      those envs. (A is 65-dim, B's env is 66-dim with target_axis_misalign
      appended LAST, so obs[:, :65] is exactly A's training observation — the
      same contract rl_demo_handoff_continuous.py relies on.)

  (2) MASK pre-onset steps from the PPO update. The stored transition for a
      pre-onset step holds B's *sampled* action + log-prob (from alg.act) but
      the reward came from A's action — inconsistent. We zero those steps'
      advantages (renormalising over the kept post-onset steps), so they
      contribute exactly zero policy gradient; their stale action/log-prob then
      don't matter. Returns are left intact so the critic still learns truthful
      values on the (real) lift states.

Why NOT the tempting "wrap the policy to emit A's action" approach: alg.act
stores the action's log-prob *under B*; swapping in A's action post-hoc leaves a
stale log-prob -> corrupt PPO ratio. Overriding the *stepped* action (not the
stored one) + masking avoids any PPO-internal log-prob surgery.
"""
from __future__ import annotations

import time

import torch
from rsl_rl.utils import check_nan

from mjlab.tasks.manipulation.rl.runner import ManipulationOnPolicyRunner


class LiveAOnPolicyRunner(ManipulationOnPolicyRunner):
  """ManipulationOnPolicyRunner whose rollout runs frozen Policy A live for the
  first `live_a_onset` steps of every episode, then trains B only on the
  post-onset (organic-seam) transitions.

  Call `setup_live_a(actor_a, onset, a_obs_dim)` after construction/warmstart.
  If never called, `learn` behaves exactly like the base runner.
  """

  def setup_live_a(self, actor_a, onset: int, a_obs_dim: int = 65,
                   blend_steps: int = 0, drive_post: bool = False) -> None:
    self.live_a_actor = actor_a
    self.live_a_onset = int(onset)
    self.live_a_obs_dim = int(a_obs_dim)
    self.live_a_blend = max(0, int(blend_steps))
    # drive_post=False (default): frozen actor drives PRE-onset (the classic live-A
    #   reset — frozen A lifts, learner B reorients from the organic seam).
    # drive_post=True: frozen actor drives POST-onset (B→A CO-REFINEMENT — the learner
    #   is Policy A driving the lift 0..onset; a FROZEN reorienter B drives >=onset; we
    #   mask the B-driven steps but keep the whole reward stream, so A's lift steps get
    #   discounted downstream credit for B's reorient reward via GAE → A learns to
    #   deliver a grip that reorients better. Slow it with a low --learning-rate.
    self.live_a_drive_post = bool(drive_post)
    self.live_a_actor.eval()
    if self.live_a_drive_post:
      print(f"[live-a] CO-REFINEMENT: learner (Policy A) drives lift 0..{onset}; frozen "
            f"reorienter drives >= {onset} (obs dim {a_obs_dim}); B-driven steps masked "
            f"from the update. A's lift steps get downstream reorient credit via returns.")
    elif self.live_a_blend > 0:
      print(f"[live-a] enabled: frozen A drives steps 0..{onset} of every episode "
            f"(A obs dim {a_obs_dim}); then a {self.live_a_blend}-step SEAM RAMP-IN "
            f"alpha*B+(1-alpha)*A (alpha 0->1) eases B into the grip. All A-driven "
            f"AND blend steps masked from PPO; B trains only on fully-B steps "
            f">= {onset + self.live_a_blend}.")
    else:
      print(f"[live-a] enabled: frozen A drives steps 0..{onset} of every episode "
            f"(A obs dim {a_obs_dim}); pre-onset steps masked from PPO update.")

  def setup_schedule_tracking(self, schedule, weight: float, reorient_start: int) -> None:
    """On-policy trajectory tracking: pull B to MEET-OR-EXCEED a teacher's
    held-cos-vs-time `schedule` (e.g. B3's reorientation curve from its clean
    spawn). One-sided — reward = -weight * relu(schedule[phase] - cos) — so B is
    pulled UP toward the teacher curve and never penalized for exceeding it.
    Added to the env reward on B-driven reorient steps only (phase indexes from
    reorient_start). Tracks the target TRAJECTORY, not the teacher's raw actions
    (which are OOD on A's delivered grip — measured: B3 craters 0.98->0.26)."""
    self._sched = torch.as_tensor(schedule, dtype=torch.float32, device=self.device)
    self._sched_w = float(weight)
    self._sched_rs = int(reorient_start)
    self._sched_K = int(self._sched.numel())
    e = getattr(self.env, "unwrapped", self.env)
    self._sched_obj = e.scene["cube"]
    print(f"[sched-track] enabled: weight={weight} reorient_start={reorient_start} "
          f"len={self._sched_K} target cos {float(self._sched[0]):.3f}->{float(self._sched[-1]):.3f}")

  def _schedule_reward(self, step_in_ep: torch.Tensor) -> torch.Tensor:
    """Per-env one-sided below-curve penalty on B-driven reorient steps; 0 else."""
    q = self._sched_obj.data.root_link_pose_w[:, 3:7]
    qx, qy = q[:, 1], q[:, 2]
    cos = (1.0 - 2.0 * (qx * qx + qy * qy)).clamp(-1.0, 1.0)
    phase = (step_in_ep - self._sched_rs).clamp(0, self._sched_K - 1)
    target = self._sched[phase]
    active = (step_in_ep >= self._sched_rs).to(cos.dtype)
    return self._sched_w * (-(target - cos).clamp(min=0.0)) * active

  def _a_action(self, obs_td) -> torch.Tensor:
    # Match rl_demo_handoff_continuous.py's deploy path EXACTLY so the seam B
    # trains on is the seam it sees at deploy: feed A the leading A_OBS_DIM dims
    # as a raw tensor through `.mlp` (the rsl_rl MLPModel.forward needs a
    # TensorDict; `.mlp` takes the already-concatenated latent and returns the
    # deterministic action mean, relying on an identity obs-normalizer — the
    # same call the validated continuous-handoff demo uses).
    a_obs = obs_td["actor"][:, : self.live_a_obs_dim]
    return self.live_a_actor.mlp(a_obs)

  def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
    if getattr(self, "live_a_actor", None) is None:
      return super().learn(num_learning_iterations, init_at_random_ep_len)

    if init_at_random_ep_len:
      self.env.episode_length_buf = torch.randint_like(
        self.env.episode_length_buf, high=int(self.env.max_episode_length)
      )

    obs = self.env.get_observations().to(self.device)
    self.alg.train_mode()
    if self.is_distributed:
      self.alg.broadcast_parameters()
    self.logger.init_logging_writer()

    onset = self.live_a_onset
    blend = getattr(self, "live_a_blend", 0)
    keep_from = onset + blend          # first step where B fully drives = trainable
    num_envs = self.env.num_envs
    # Per-env step-in-episode counter (mirrors the deploy clock in
    # rl_demo_handoff_continuous.py: A acts steps 0..onset-1, B from onset).
    step_in_ep = torch.zeros(num_envs, dtype=torch.long, device=self.device)

    start_it = self.current_learning_iteration
    total_it = start_it + num_learning_iterations
    for it in range(start_it, total_it):
      start = time.time()
      keep_masks = []  # (num_envs,) bool per rollout step; True = B drove = trainable
      n_pre = 0
      sched_rew_sum = 0.0
      with torch.inference_mode():
        for _ in range(self.cfg["num_steps_per_env"]):
          # learner's sampled action (also stores its action/log-prob/value/obs).
          actions = self.alg.act(obs)
          stepped = actions.clone()
          if getattr(self, "live_a_drive_post", False):
            # CO-REFINEMENT: learner (A) drives the lift 0..onset (trainable); the frozen
            # reorienter drives >= onset (masked). No blend in this mode.
            keep = step_in_ep < onset            # A drove the lift = trainable
            need_a = step_in_ep >= onset         # frozen reorienter drives (masked)
            if bool(need_a.any()):
              a_act = self._a_action(obs).to(stepped.dtype)
              stepped = torch.where(need_a.unsqueeze(1), a_act, stepped)
              n_pre += int(need_a.sum())
          else:
            pre = step_in_ep < onset              # A's live-lift window (full A)
            in_blend = (step_in_ep >= onset) & (step_in_ep < keep_from)  # ramp-in
            keep = step_in_ep >= keep_from        # B fully drove = trainable
            need_a = pre | in_blend
            if bool(need_a.any()):
              a_act = self._a_action(obs).to(stepped.dtype)
              stepped = torch.where(pre.unsqueeze(1), a_act, stepped)
              if blend > 0 and bool(in_blend.any()):
                # alpha ramps 0->1 across the blend window: at the first blend step
                # (step==onset) alpha=1/(blend+1) (mostly A), approaching 1 at the
                # last. Keeps the reorienter from shocking A's grip at the seam.
                alpha = ((step_in_ep - onset + 1).to(stepped.dtype)
                         / float(blend + 1)).clamp_(0.0, 1.0).unsqueeze(1)
                blended = alpha * actions + (1.0 - alpha) * a_act
                stepped = torch.where(in_blend.unsqueeze(1), blended, stepped)
              n_pre += int(need_a.sum())
          obs, rewards, dones, extras = self.env.step(stepped.to(self.env.device))
          if self.cfg.get("check_for_nan", True):
            check_nan(obs, rewards, dones)
          obs, rewards, dones = (
            obs.to(self.device), rewards.to(self.device), dones.to(self.device))
          if getattr(self, "_sched", None) is not None:
            sr = self._schedule_reward(step_in_ep).to(rewards.dtype).reshape(rewards.shape)
            rewards = rewards + sr
            sched_rew_sum += float(sr.sum())
          self.alg.process_env_step(obs, rewards, dones, extras)
          intrinsic_rewards = self.alg.intrinsic_rewards if self.cfg["algorithm"]["rnd_cfg"] else None
          self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)

          keep_masks.append(keep.clone())
          step_in_ep = step_in_ep + 1
          done_flat = dones.reshape(num_envs).bool()
          step_in_ep = torch.where(done_flat, torch.zeros_like(step_in_ep), step_in_ep)

        stop = time.time()
        collect_time = stop - start
        start = stop

        self.alg.compute_returns(obs)
        # ---- live-A masking: kill the policy gradient on A-driven steps ------
        keep = torch.stack(keep_masks, dim=0)            # (T, num_envs) bool
        self._mask_pre_onset_advantages(keep)

      loss_dict = self.alg.update()

      stop = time.time()
      learn_time = stop - start
      self.current_learning_iteration = it

      frac_pre = n_pre / float(self.cfg["num_steps_per_env"] * num_envs)
      self.logger.log(
        it=it, start_it=start_it, total_it=total_it,
        collect_time=collect_time, learn_time=learn_time, loss_dict=loss_dict,
        learning_rate=self.alg.learning_rate,
        action_std=self.alg.get_policy().output_std,
        rnd_weight=self.alg.rnd.weight if self.cfg["algorithm"]["rnd_cfg"] else None,
      )
      if it % 10 == 0:
        sched_msg = ""
        if getattr(self, "_sched", None) is not None:
          per = sched_rew_sum / float(self.cfg["num_steps_per_env"] * num_envs)
          sched_msg = f" sched_rew/step={per:.4f} (<0 = below B3 curve)"
        print(f"[live-a] it={it} A-driven(masked) frac={frac_pre:.3f}{sched_msg}")

      if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
        self.save(f"{self.logger.log_dir}/model_{it}.pt")

    if self.logger.writer is not None:
      self.save(f"{self.logger.log_dir}/model_{self.current_learning_iteration}.pt")
      self.logger.stop_logging_writer()

  def _mask_pre_onset_advantages(self, keep: torch.Tensor) -> None:
    """Zero advantages on A-driven (pre-onset) steps, renormalising over the
    kept post-onset steps. keep: (T, num_envs) bool, True = keep (B drove).

    Leaves storage.returns intact: the critic still fits truthful values on the
    real lift states A produced. Only the *policy* gradient is suppressed
    pre-onset (advantage 0 -> surrogate 0 -> zero grad through the ratio)."""
    st = self.alg.storage
    m = keep.to(st.returns.device).unsqueeze(-1)          # (T, N, 1) bool
    adv = st.returns - st.values                          # raw advantages (T, N, 1)
    kept = adv[m]
    # ROBUSTNESS: if almost every episode dies at the seam (kept-set tiny or
    # ~constant), normalising over it produces huge/NaN advantages -> the PPO update
    # explodes the actor std -> `normal expects std >= 0` crash. Guard by zeroing ALL
    # advantages that iter (skip the policy update) so a transient bad seam can't kill
    # the whole run; the critic still learns from returns. Needs a healthy trainable
    # fraction to make progress (fix the warmstart/grace, not just this guard).
    frac = float(m.to(torch.float32).mean())
    if kept.numel() < 64 or float(kept.std()) < 1e-5 or frac < 0.02:
      print(f"[live-a] WARN: trainable frac={frac:.3f} kept={kept.numel()} "
            f"std={float(kept.std()) if kept.numel() else 0:.2e} -> zeroing advantages "
            f"(skipping policy update this iter to avoid a std-NaN blowup).")
      st.advantages = torch.zeros_like(adv)
      return
    adv = (adv - kept.mean()) / (kept.std() + 1e-8)
    adv = adv * m.to(adv.dtype)                            # zero pre-onset
    st.advantages = adv
