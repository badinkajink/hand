# Deployable reorientation from joint position and servo load

Research and implementation memo, 2026-08-29.

## Executive conclusion

The proposed direction is viable, but not as a feed-forward 65-dimensional-to-18-dimensional
behavior-cloning exercise. The deployable controller is solving a partially observable control
problem: object pose, object motion, contacts, and physical parameters are hidden, and the same
instantaneous joint position/load vector can correspond to different object states that require
different actions. Distillation cannot recover information that is absent from both the current
observation and its history.

The recommended first target is deliberately narrow:

- one known screwdriver geometry and mass range;
- a repeatable, bounded initial grasp distribution;
- one prescribed reorientation maneuver/axis;
- a low-authority learned residual around the already successful `real_v1` linear anchor;
- a recurrent actor using joint/load **history plus controller-known state**;
- a privileged critic during training;
- student-on-policy teacher queries followed by PPO fine-tuning on the student's observation
  process;
- temporary external object tracking during development, if at all possible, but no external
  sensor required at deployment.

For that bounded task, this is a credible skunkworks program. For arbitrary SO(3) goals, arbitrary
objects, and arbitrary initial grasps, nine positions plus nine uncalibrated load values are unlikely
to be sufficient. The literature's strongest general reorientation results add tactile or visual
sensing, while proprioception-only successes are usually continuous rotation around a fixed axis.

The highest-value change to the original proposal is therefore:

> Train the final actor directly for its POMDP, using a privileged critic and history. Use the
> privileged teacher as a curriculum, representation target, and data labeler—not as the definition
> of the final policy.

## 1. What the current system actually does

### 1.1 Policy observation

`src/morphohand/rl/env_build.py::_build_observations` currently gives the actor:

| Block | Dimensions | Deployment status |
|---|---:|---|
| all joint positions | 15 | 9 finger values measured; 6 palm values are commanded/scripted and therefore known |
| all joint velocities | 15 | finger values can be estimated from history; palm values are known |
| palm-to-object position | 3 | unavailable |
| actual object pose in palm frame | 7 | unavailable |
| reference finger positions | 9 | known from the anchor/reference generator |
| reference object pose | 7 | simulation reference, not a trustworthy real measurement |
| previous action | 9 | known exactly |
| target-axis misalignment, Policy B only | 1 | unavailable without object-state estimation |

That is 65 inputs for Policy A and 66 for Policy B. The saved `real_v1` checkpoints confirm a
`512 x 65` or `512 x 66` first actor layer and a 9-dimensional action. The code currently copies the
same terms to actor and critic, merely turning observation corruption off for the critic. In other
words, the present setup is **not** asymmetric actor-critic; both networks receive object state.

The actor and critic are feed-forward ELU MLPs with hidden widths `(512, 256, 128)` in
`src/morphohand/rl/ppo_config.py`, constructed by `src/morphohand/rl/ppo_runner.py`.

### 1.2 Control structure

The learned output is a 9-dimensional finger residual over a scripted/reference trajectory. Palm
motion is scripted. The recent system uses Policy A for lift/delivery and Policy B for reorientation,
with a handoff between them. The current `real_v1` mechanism results show that the linear anchor can
already reorient selected designs in simulation; learned Policy B mainly supplies grip regulation,
pole selection, and recovery around that mechanism.

This is favorable. A residual controller around a known mechanism asks the blind policy to correct
contact evolution, not invent the whole maneuver from sparse sensing. It also bounds the damage a
bad inference can cause.

### 1.3 Real feedback and loop constraints

`docs/hardware_control_station.md` and the servo driver establish:

- nine servo positions can be read at approximately 111 Hz;
- position plus `present_load` takes approximately 18 ms, or about 55 Hz;
- the load field is a duty-cycle-like, uncalibrated proxy—not force or torque;
- servo reads are sequential because sync-read is unsupported, so channels in one nominal frame
  are sampled at different times;
- a write plus position read has been measured around 75 Hz, making a 50 Hz loop plausible;
- the current policy stream is write-only and suspends telemetry reads while it owns the bus.

The last point is important: closed-loop deployment requires a new minimal
read-position/load -> infer -> sync-write path. The current cached telemetry route and HTTP policy
stream are not that path.

### 1.4 The observation is not really limited to 18 network features

There are only 18 raw sensor values, but the controller knows more without adding hardware:

- the previous command and previous residual;
- the current anchor joint target;
- maneuver phase/progress according to the command generator;
- the prescribed target axis;
- scripted palm state;
- timestamps, sample age, validity, and fault state.

The deployable per-step feature vector should exploit these. A proposed frame is:

```text
measured finger q                         9
raw servo load                           9
previous command - measured q            9
filtered delta-q / velocity estimate     9
previous residual/action                 9
current anchor finger target             9
anchor phase                              1-2
task command/axis                         as needed
sample ages and validity                  compact encoding
```

Only the first 18 are sensed. Everything else is causally available on the robot. In particular,
command tracking error and its history can be more informative about contact than raw load.

## 2. What prior work says

### 2.1 The closest positive precedent: HORA

[HORA / In-Hand Object Rotation via Rapid Motor Adaptation](https://proceedings.mlr.press/v205/qi23a.html)
is the closest match. It trains a privileged policy and an adaptation module, then deploys using only
a short history of joint positions and previous actions. It transfers fixed-axis fingertip rotation
from simulated cylinders to many real objects. The key lessons for this project are:

- temporal proprioception and action history, not a single proprioceptive frame;
- online inference of a compact latent rather than explicit full system identification;
- teacher queries on states induced by the adapting policy, not only an offline teacher dataset;
- randomization over object mass, center of mass, scale, friction, initial pose, and hand state;
- adaptation outperformed a single robust domain-randomized policy in its setting.

The original [RMA work](https://roboticsproceedings.org/rss17/p011.html) provides the general
two-stage template: a base policy consumes privileged environment parameters compressed into a
latent; an adaptation module reconstructs that latent from recent proprioception and actions.

The boundary matters: HORA demonstrates continuous rotation about a fixed axis. It does not prove
that proprioception alone can reliably stop at an arbitrary hidden target orientation.

### 2.2 Goal reorientation requires state estimation or richer sensing

[Pitz et al.'s modular tactile reorientation system](https://arxiv.org/abs/2303.04705) explicitly
frames goal reorientation without external sensing as a state-estimation problem. It uses a 0.5 s
observation window and a differentiable particle filter to estimate object state, then fine-tunes the
policy with the estimator in the loop. It reports 92% simulation success and real transfer across 24
discrete cube orientations. Its sensors are joint position and torque on a high-quality
torque-controlled hand, stronger and better characterized than this platform's load proxy.

[Purely tactile continuous rotation on DLR Hand-II](https://arxiv.org/abs/2204.03698) is also strong
evidence that intrinsic sensing can work, but it depended on precise modeling/system identification,
true position and torque sensing, and a domain-adapted curriculum. It is evidence for the control
principle, not evidence that raw hobby-servo load transfers directly.

[Intrinsic-sensing finger-gaiting](https://arxiv.org/abs/2109.12720) emphasizes the importance of a
useful initial-state distribution for exploration and transfer. This supports broadening the grasp
and handoff distributions during training rather than perfecting a single deterministic simulated
handoff.

### 2.3 Asymmetric training is the default baseline

[Asymmetric actor-critic](https://www.roboticsproceedings.org/rss14/p08.html) exists precisely for
simulation training where the critic may use full state while the deployed actor receives partial or
noisy observations. Here, the actor should receive only the hardware-realizable observation process;
the critic may receive object pose/velocity, contact state, and randomized parameters. Privileged
rewards are also acceptable during simulation training.

This baseline is more important than teacher distillation: it directly optimizes the best policy
possible under the deployable information constraint. A fully privileged teacher can learn
microstate-dependent behavior that no student can imitate.

### 2.4 Richer sensing buys broader capability

[RotateIt](https://proceedings.mlr.press/v229/qi23a.html) and
[AnyRotate](https://proceedings.mlr.press/v270/yang25c.html) obtain multi-axis/generalized behavior by
adding tactile and/or visual information and temporal fusion. [DeXtreme](https://arxiv.org/abs/2210.13702)
shows robust domain-randomized reorientation with a real-time vision pose estimator. These are useful
warnings against interpreting successful domain randomization as a substitute for observability.

A May 2026 preprint,
[Proprioceptive Transformer](https://arxiv.org/abs/2605.21330), is unusually relevant: it distills a
privileged teacher into a policy using only joint position/velocity histories and demonstrates real
continuous cube rotation on a tendon-driven hand. It reports better rotation speed and implicit
object-position estimation than an MLP baseline. It is encouraging, but it is a new preprint and again
targets continuous rotation rather than verified finite-goal reorientation.

A March 2026 preprint,
[PTLD](https://arxiv.org/abs/2603.04531), suggests an especially useful skunkworks tactic: use
temporary privileged sensing in the real world to label tactile/proprioceptive histories, then deploy
without that privileged sensor. It avoids pretending that tactile simulation is accurate. Its results
should be treated as provisional because the work is recent, but the data strategy is directly useful.

## 3. Recommended architecture

### 3.1 Make the final actor recurrent and asymmetric

Define:

- `x_t`: deployable current features (raw sensors plus controller-known state);
- `H_t`: the last 0.5-1.0 s of `x`, commands, and actions;
- `p_t`: privileged simulator state and episode parameters;
- `V(s_t, theta)`: critic using full state and randomized physics;
- `pi_S(a_t | x_t, H_t)`: the only actor that matters at deployment.

Start with a one-layer GRU, hidden size 128 or 256, followed by a small MLP. At 50 Hz this is cheap,
easy to export, and less data-hungry than a transformer. Compare a causal TCN if recurrent training is
unstable. Only try a small transformer if the GRU/TCN fails the observability benchmark while a longer
history demonstrably contains the needed information.

The installed RSL-RL 5.0.1 already contains `RNNModel` with GRU/LSTM support and TorchScript/ONNX
export. The local `ppo_runner.py` wrapper currently constructs the narrower `RslRlModelCfg` and always
uses `MLPModel`; it needs a local recurrent config path, not a new RL framework.

### 3.2 Keep the anchor and reduce authority

Use one learned residual around the `real_v1` reorientation anchor:

```text
q_cmd(t) = q_anchor(t) + clipped_and_rate_limited(delta_q_student(t))
```

Initially keep grasp/lift/delivery scripted or CEM-controlled and start the recurrent state before the
reorientation handoff so it observes settling/contact history. Avoid separately deploying the current
fully privileged Policy A and Policy B. If the handoff itself remains the dominant failure, train one
history-conditioned residual across late delivery, settle, and reorientation to remove the seam.

Begin with a much smaller residual envelope than the current approximately 0.5 scale and expand it by
curriculum only after it improves held-out success. Exact limits should come from clearance, speed, and
load tests on hardware rather than an arbitrary radian value.

### 3.3 Use a structured teacher, but do not stop at behavior cloning

Train or fine-tune a teacher in the same anchor-residual task:

```text
z* = mu(privileged object state, contacts, randomized physical parameters)
a_T = pi_T(x_t, z*)
z_hat = phi(H_t)
a_S = pi_S(x_t, z_hat)
```

Use a small privileged latent (roughly 8-16 dimensions), latent noise/dropout, and the same residual
limits as the student. This discourages a teacher that reacts to simulator microstate the student
cannot infer.

Train the student with three kinds of signal:

1. action distribution imitation or Huber action loss;
2. privileged-latent reconstruction;
3. auxiliary estimates of control-relevant state: target-axis alignment, alignment rate, object
   height/drop risk, slip/contact mode, and selected dynamics bins.

The installed RSL-RL `DistillationRunner` is a useful baseline: the student acts in the environment
and the teacher labels the resulting student states, addressing basic covariate shift. However, its
stock algorithm minimizes only action MSE. Extend it for latent/auxiliary losses and scheduled teacher
mixing, or use it unchanged as the naive-distillation control arm.

Finally, fine-tune the student on-policy with PPO, actor observations restricted to `x_t,H_t`, a
privileged critic, and decaying imitation/latent losses. This estimator-in-the-loop phase is mandatory:
the modular DLR work found that replacing true state with estimated state changes the policy's state
distribution enough to require further policy optimization.

### 3.4 Direct recurrent PPO is the primary baseline

Train a recurrent actor from scratch with deployable observations and a privileged critic. It may
outperform the teacher/student stack because it never learns actions that depend on irrecoverable
information. If it gets close to the teacher, keep the simpler method.

The recommended ranking is therefore:

1. recurrent asymmetric PPO around the anchor;
2. recurrent asymmetric PPO warm-started/regularized by privileged latent distillation;
3. pure student-teacher action distillation as an ablation, not the default deployment candidate.

## 4. Observability audit before expensive training

This experiment determines whether the task is learnable from the proposed sensor stream.

Generate randomized teacher and anchor rollouts and retain full simulator state. Construct datasets
with exactly the timing and channel process expected on hardware. For history lengths 0, 0.1, 0.25,
0.5, 1.0, and 2.0 s, train small probes to predict:

- target-axis alignment and signed alignment rate;
- object vertical position and drop-within-200-ms;
- contact/slip mode;
- the teacher action or privileged latent.

Compare feature sets:

- `q` only;
- `q + previous actions/commands`;
- `q + command tracking error`;
- all above plus load;
- all above plus perfect simulated actuator torque, as an upper bound;
- privileged object state, as the ceiling.

Use trajectory-disjoint and physics-disjoint validation. Also measure the conditional variance of the
teacher action among near-identical deploy histories. High variance means the teacher is using hidden
information and is intrinsically hard to imitate.

Separately perform **closed-loop interventional ablations** of the existing 65/66-dimensional policy:
zero, noise, delay, or replace each observation block while rolling out. First-layer weight magnitude
is not a reliance test; correlated features and closed-loop compensation make it misleading.

Stop or narrow the goal if histories cannot predict task progress well enough to decide when to slow
or hold. No network architecture can repair a non-observable task.

## 5. Domain randomization and the load problem

### 5.1 Calibrate the observation channel before physics

Perfect system identification is unnecessary, but a small amount of channel identification is high
leverage:

- free-space step/chirp trajectories at several speeds and temperatures to measure delay, deadband,
  backlash, quantization, velocity limits, and load baseline;
- repeated scripted grasps, ideally against a load cell or known compliant fixture, to characterize
  load sign, offset, scale variation, hysteresis, saturation, and cross-servo differences;
- repeated anchor trajectories with temporary vision/markers to record real object angle, height,
  slip, and outcome.

The objective is not to call `present_load` force. It is to model the statistics and failure modes of
the raw covariate the student will actually see.

### 5.2 Simulate the sensor, not an imagined force sensor

Build a load proxy from actuator effort, tracking error, velocity, and contact state, then apply a
random per-servo observation model:

- affine scale and offset;
- sign convention uncertainty where applicable;
- saturation/dead zone and nonlinear compression;
- low-pass lag, hysteresis, and drift;
- noise, dropouts, stuck values, and invalid values;
- correlated changes with supply voltage/temperature if real logs show them.

Randomize the sequential sampling skew across the nine servos and include sample ages in the actor
input. If the full position+load cycle leaves insufficient timing margin, test load at 25 Hz with
sample-and-hold while positions remain at 50 Hz.

Always retain a q/action-history-only arm. If the load proxy does not improve held-out simulation and
real shadow-mode state estimation, remove it rather than letting the policy exploit simulation-only
effort artifacts.

### 5.3 Physics randomization should be measured and curricular

Randomize per episode:

- joint zero offsets and geometry tolerances;
- effective servo stiffness/damping, speed, torque saturation, stiction, backlash, and latency;
- sticky/delayed actions and controller period jitter;
- object mass, center of mass, dimensions, friction, and contact compliance;
- initial object pose, grasp depth, and finger contact arrangement.

Randomize per step:

- position quantization/noise;
- load observation mapping and drift;
- sample ages/channel skew, dropped reads, and held values.

Use measured nominal support first, then widen one family at a time. Keep held-out combinations and
edges that are never used for training. The repository already contains a warning against indiscriminate
randomization: the recent compliance-DR experiments in `docs/rl/reorientation.md` relocated the
policy's narrow contact regime rather than broadening it, while friction DR preserved nominal behavior
but did not solve the cliff. Broad randomization is not automatically robustness.

## 6. Training curriculum

### Phase A: define the deployment contract

Freeze the first task's object, target axis/angle, initial grasp bounds, timing, success condition,
residual limits, and safety fallback. Define success as held final alignment with minimum object height;
do not score a transient correct pose while the object is already falling.

### Phase B: make the task observable by construction

Start from the proven anchor and a narrow initial-state/dynamics distribution. Train the recurrent
asymmetric baseline and run the probe suite. Increase history only when probes show benefit.

### Phase C: privileged teacher and student

Train a structured, randomized teacher. Roll the student, query the teacher on student-visited states,
and gradually reduce teacher action mixing. Reconstruct a compact latent and control-relevant
auxiliary variables. Do not train only on clean teacher rollouts.

### Phase D: student-distribution PPO

Optimize the recurrent student on-policy with the privileged critic. Decay action imitation; retain
small latent/auxiliary losses if they improve validation. Curriculum-expand initial grasps, dynamics,
sensor faults, and residual authority independently.

### Phase E: instrumented real data

Run the anchor on hardware while the student operates in shadow mode. Record raw positions, raw load,
commands, timings, proposed residuals, and temporary camera/marker object pose. Use those real histories
to:

- validate whether simulated sensor statistics cover reality;
- supervise the history encoder/auxiliary estimates;
- identify failure regions for additional simulation randomization;
- perform DAgger-style relabeling with an instrumented real teacher if safe.

External tracking is a development instrument, not a deployment dependency. This is likely the
highest-return skunkworks hardware addition because it replaces guesswork about whether the student
can infer object progress.

### Phase F: bounded closed-loop rollout

Start with a very small residual envelope, a cage/soft catch, automatic timeout, slew limits, joint
limits, load-proxy saturation checks, and a watchdog fallback to a validated hold pose. Expand only
after repeated held success across restarts and days.

## 7. Experiment matrix and decision gates

Run at least three independent training seeds and multiple rollout repeats. This repository has
already shown substantial policy-draw variance and contact nondeterminism; one seed is not evidence.

| Arm | Actor input/model | Purpose |
|---|---|---|
| Anchor | no learned residual | real and sim floor |
| FF-18 | one frame `q + load`, MLP | naive proposal baseline |
| Hist-q | `q + command/action` history, GRU | proprioceptive baseline |
| Hist-ql | above plus load, GRU | incremental value of load |
| AAC | full deploy feature history, recurrent PPO; privileged critic | primary baseline |
| RMA | privileged latent teacher + history encoder | adaptation hypothesis |
| BC | stock action-MSE distillation | covariate/teacher-dependence control |
| Oracle | full-state actor | achievable ceiling |

Evaluate:

- held-out success and teacher-student gap;
- final held alignment, not peak alignment;
- minimum object height and drops;
- time-to-goal and hold duration;
- residual magnitude, slew, chatter, and saturation;
- contact/load balance and tracking-error envelope;
- robustness across mass, grasp, friction/compliance, latency, temperature/day, and servo restart;
- probe error for alignment, alignment rate, and drop risk.

Suggested gates:

1. If recurrent AAC is within roughly 10 percentage points of the oracle on held-out simulation,
   skip the more complex RMA path initially.
2. If adding load does not beat q/command history consistently in held-out sim and real shadow data,
   omit load from the first deployed policy.
3. If alignment/progress remains poorly inferable and the policy cannot reliably stop, restrict the
   task to a fixed timed anchor or add a deployment sensor.
4. Do not proceed from shadow mode until inference timing, channel ages, and fallback behavior are
   deterministic under induced read failures.
5. Do not broaden objects or axes until the fixed-task controller beats the anchor across seeds and
   real-day repeats, not just at its best trial.

The 10-point gate is a triage heuristic, not a scientific threshold; set the actual threshold from
the anchor's real variance and the cost of failure.

## 8. Concrete repository changes

### Environment and observations

In `src/morphohand/rl/env_cfg.py`:

- replace the single `obs_mode` concept with explicit actor/critic observation modes;
- add sensor-model and observation-history configuration;
- add curriculum/randomization controls for initial state, actuator, transport, and raw load mapping.

In `src/morphohand/rl/mjlab_terms.py` or a focused new observation module:

- measured-finger-position proxy;
- raw-load proxy and sensor corruption;
- previous-command tracking error;
- filtered delta-q;
- anchor target/phase;
- sample age/validity;
- privileged latent targets and auxiliary labels.

In `src/morphohand/rl/env_build.py`:

- actor group: deployable terms only, with corruption;
- critic group: full simulator state, contacts, velocities, and randomized parameters;
- teacher group: structured privileged input;
- student group: deployable input;
- remove actual object pose and target misalignment from every deployable actor configuration.

### Learning stack

In `src/morphohand/rl/ppo_config.py` and `ppo_runner.py`:

- expose `RNNModel`, GRU type/width/layers, and observation normalization;
- allow genuinely different actor and critic groups/models;
- add recurrent export and reset tests.

Add a distillation configuration/runner that builds on RSL-RL's existing student-driven
`DistillationRunner`, adding:

- latent and auxiliary heads/losses;
- Huber or distribution-level action loss;
- teacher/student action mixing schedule;
- recurrent sequence batching and burn-in;
- optional PPO fine-tuning with a privileged critic.

Add an evaluation command that can apply observation-block interventions and replay the exact
hardware timing/sensor model.

### Hardware runtime

In the hardware runtime/servo layer:

- implement one purpose-built minimal policy cycle that owns the bus, reads the nine positions and
  loads, timestamps them, runs local inference, and sync-writes nine commands;
- do not route the control loop through cached UI telemetry or per-step HTTP;
- run fault/diagnostic reads at a slower side cadence;
- log every raw observation, age, command, model output, clip/slew intervention, loop duration, and
  watchdog event;
- reset recurrent state on new grasp, timeout, fault, or operator abort;
- prefer local TorchScript/ONNX inference on the CB1 if measured compute permits; otherwise use a
  persistent binary connection with the safety envelope enforced on the CB1.

## 9. Main risks and mitigations

| Risk | Consequence | Mitigation |
|---|---|---|
| Hidden object states alias under q/load history | no policy can choose the correct recovery/stop action | observability probes; restrict task; temporary tracking; add sensor if needed |
| Teacher relies on privileged microstate | low imitation loss in dataset, poor student rollouts | compact/noisy latent; student-state queries; direct AAC; student PPO fine-tune |
| Simulated load is unlike real load | policy exploits artifacts | q-only arm; real shadow data; randomized observation mapping; omit load if unhelpful |
| Broad DR destroys the useful contact mechanism | conservative or failed policy | measured, one-family-at-a-time curriculum and held-out tests |
| Policy A -> B handoff is out of distribution | early drop/bifurcation | include pre-handoff history; randomize handoff; ultimately one residual across the seam |
| 18 ms sequential read causes stale mixed frames | recurrent encoder interprets skew as dynamics | timestamps/ages, skew randomization, slower load cadence if required |
| Network/telemetry path adds jitter | unsafe/unstable loop | dedicated bus-owner loop, local safety layer, watchdog and hold fallback |

## 10. Recommended immediate sequence

1. Implement actor/critic observation separation and a deployable recurrent AAC baseline.
2. Add the exact hardware observation/timing model and q-only versus q+load probe dataset.
3. Run the observability audit and closed-loop ablation of the current privileged policy.
4. Train the experiment matrix on the narrow fixed screwdriver/anchor task, three seeds each.
5. In parallel with training, build the deterministic 50 Hz hardware read-infer-write loop and logger.
6. Collect instrumented anchor rollouts with temporary object tracking; use them to validate the
   sensor model before any learned residual controls hardware.
7. Deploy shadow mode, then bounded residual mode. Expand task diversity only after the decision gates.

This sequence produces useful answers even if the full student-teacher hypothesis fails: it will tell
us whether position history is sufficient, whether load adds information, which privileged variables
the present policy actually uses, and whether the real bottleneck is sensing, simulation, handoff, or
control-loop timing.

