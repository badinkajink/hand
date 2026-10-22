"""Text of the overview docs/overviews/code_and_methods.html (builder: scripts/topic_overview_page.py code).

Reader: a collaborator new to the project who wants to run or change the code. Takeaways: (1) where the code is and the
path of one hand from a sample to a bench plan; (2) the protocols in use: frozen scene, the grasp fitter and the CEM lane
gated on the held lift, distribution evaluation with stored traces; (3) the simulator backends and what each runs.
Budget: 1,400 words including Terms and the index. Sources: the mkdocs pages of April-July 2026, checked against the code
on 2026-10-11 (logs/20261011-mkdocs_fold/digest.md); the pages themselves are in git history.
"""

KEY = "code"
FILE = "code_and_methods.html"
TOPIC_TITLE = "Code and methods"
SHORT_TITLE = "Code and methods"
TITLE = "Code map, grasp synthesis, evaluation protocol and simulator backends"

LEDE = (
    "The package <code>src/morphohand/</code> contains the simulation, grasp, learning and bench code. The studies and page "
    "builders that call it are in <code>scripts/</code>. A real_v1 hand goes through five stages. It starts as a Sobol "
    "sample of six gantry coordinates and becomes a frozen MuJoCo scene. A grasp comes from the geometric fitter. A "
    "residual PPO policy is trained in MuJoCo-Warp through mjlab. An evaluation over 64 rollouts stores its per-step "
    "traces. A bench plan then passes a 5&#8202;mm clearance gate. "
    "MuJoCo on the CPU runs grasps, chains and plans; Newton and Drake run the contact comparisons in separate environments.")

TERMS = [
    ("Morphology run", "A directory under <code>results/phase1/</code> with <code>frozen_scene.xml</code>, "
     "<code>summary.json</code> and <code>best_rollout.npz</code>; trainers and evaluators read a hand from it."),
    ("Frozen scene", "A scene with the hand&#8217;s layout baked into fixed geometry, without the nine morphology joints."),
    ("<code>open_ik</code>", "The keyframe of the open fingers before the grasp: the CEM seed and the RL reset pose."),
    ("Grasp fitter", "<code>fit_real_v1_pose.fit</code>: the palm pose and finger angles that put the pads on the tool."),
    ("CEM", "Cross-entropy method over the nine finger controls (<code>phase1_optimize_grasp.py</code>)."),
    ("Held lift", "Tool height after the hold minus its height before the lift, in m."),
    ("Residual policy", "A PPO actor. Its output is scaled by 0.5&#8202;rad and clipped, and it is added to the scripted finger "
     "set-points from control step 58."),
    ("Recipe", "<code>configs/recipes/*.yaml</code> holds the trainer defaults of one policy type. Command-line flags override them."),
    ("Held, signed cosine", "Held: more than 0.5&#8202;N of pad force with the tool centre above 0.06&#8202;m at the last step. "
     "Signed cosine: of the tool axis with world +z, +1 upright."),
    ("Traces", "Per-step arrays of one evaluation, stored in <code>traces/&lt;run&gt;.npz</code> beside the page that plots them."),
]

SECTIONS = [
    ("map", "Code map",
     """<p>The directories of <code>src/morphohand/</code> divide the code as follows.</p>
<ul>
<li><code>sampling/</code>: the morphology dataclasses, <code>REAL_V1_WORKSPACE</code> and <code>freeze_scene_for_eval</code>.</li>
<li><code>tools/</code>: the scene generator and keyframe IK.</li>
<li><code>optimization/</code>: the grasp evaluator and CEM.</li>
<li><code>rl/</code>: the mjlab environment terms, the recipes, the deploy helpers and the trajectory-health scorecard.</li>
<li><code>bench/</code>, <code>geometry/</code> and <code>driver/</code>: the bench replay, the geometry and the servo firmware with the <code>manta_hand</code> host.</li>
</ul>
<p> <a href="../../scripts/README.md">scripts/README.md</a> maps the active scripts, and
superseded ones are in <code>scripts/archive/</code>. <code>assets/mjcf/real_v1/real_hand.xml</code> is the hardware model;
<code>assets/mjcf/</code> has one folder per hand topology plus <code>generated/</code> and dated <code>experimental/</code>
folders. Runs go to <code>results/</code>, which git ignores; pages go to <code>docs/</code>.</p>"""),
    ("path", "Path of one hand from a sample to a bench plan",
     """<ol>
<li>Sample: <code>real_v1_design_search.sobol_designs</code> draws a prefix-stable Sobol population over the six gantry
slides (thumb x &#177;30&#8202;mm and y &#177;55&#8202;mm, index and middle &#177;30&#8202;mm, link lengths fixed).
<code>real_v1_sobol8192.sh</code> runs 8,192 layouts.</li>
<li>Scene: <code>generate_morphology_xml.py --base-hand-xml assets/mjcf/real_v1/real_hand.xml</code> writes a rigid scene
without the morphology joints.</li>
<li>Grasp: the grasp fitter, called through <code>probe_real_v1_carry._grip_from_fit</code>, plus a commanded squeeze.</li>
<li>Screen: <code>real_v1_design_search.py</code> scores the open-loop carry, and <code>real_v1_promote_designs.py</code>
checks export, servo range, gantry travel and clearance.</li>
<li>Policy: <code>make_cal_morphology_run.py --hand &lt;tag&gt; --squeeze &lt;mm&gt;</code> writes a morphology run on the
calibrated plant. <code>rl_train_cube.py --recipe b_liveA --morphology-run &lt;dir&gt;</code> trains the policy through
<code>resguard.sh run</code>. <code>hands_tranche_queue.py</code> runs a queue of such jobs.</li>
<li>Evaluation: <code>policy_eval_suite.py</code>, then <code>rl_render_rollout.py</code> and <code>policy_filmstrip.py</code>
for a film and its frames.</li>
<li>Bench plan: <code>real_v1_export_plan.py --design &lt;id&gt;</code> writes <code>&lt;id&gt;_plan.json</code>,
<code>_traj.csv</code> (50&#8202;Hz, degrees), <code>_build.txt</code> and <code>_poses.txt</code>;
<code>real_v1_trajectory_clearance.py --plan &lt;id&gt;</code> checks the chord and the CSV against 5&#8202;mm. Plans are
copied to the bench computer by hand.</li>
</ol>"""),
    ("grasp", "Grasp synthesis",
     """<p>The eight deployed hands hold the tool with the fitter&#8217;s pose plus a 10&#8202;mm squeeze
(<code>real_v1_export_plan.py --squeeze-mm</code>), without CEM. <code>fit_real_v1_pose.py</code> places the pad targets on the
shaft&#8217;s equator, at the shaft radius plus the 10.55&#8202;mm pad and a 1&#8202;mm gap. It then searches palm heights
from &#8722;30 to +60&#8202;mm in 2.5&#8202;mm steps and takes the deepest one at which every tip comes within 4&#8202;mm,
with the joints 0.05&#8202;rad inside their limits. It keeps the pose if a hold probe leaves the tool at least 20&#8202;mm
up. It writes <code>open_ik</code> and,
with <code>--also-open</code>, an open pose 8&#8202;mm back.</p>
<p>The CEM lane, <code>real_v1_pipeline.py --stages grasp</code>, runs <code>phase1_optimize_grasp.py --keyframe open_ik</code>.
It runs 24 iterations of 40 candidates over the yaw, MCP and PIP controls of the three fingers. Each candidate closes for
240 steps, lifts the palm 50&#8202;mm and holds for 140 steps. The score&#8217;s lift term rewards the peak height, which a
lift followed by a drop also reaches. The pipeline therefore passes a design on a held lift of 0.03&#8202;m or more. Contact-target
patches, a force-closure energy and synergy (eigengrasp) CEM remain in the code with zero weight, and no current pipeline
enables them. <code>retarget_keyframe_ik.py</code> retargets the keyframes of the earlier nine-parameter hands, which are
kept as historical controls.</p>"""),
    ("eval", "Evaluation protocol",
     """<p>Every grasp and policy evaluation runs on a frozen scene. <code>freeze_scene_for_eval</code> bakes the keyframe&#8217;s
layout into fixed geometry, because the morphology joints drift during a rollout. It takes the real_v1 base scene from 31
to 22 position coordinates. <code>policy_eval_suite.py</code> steps 64 rollouts in parallel under the run&#8217;s own timing
and tool contact class, both read from the run&#8217;s <code>config.yaml</code>. It reports the align rate (signed cosine
of at least 0.9) and the hold rate. A single deterministic rollout on the GPU does not repeat, since contact solves reduce in a
varying order, so every claim rests on the 64. Each evaluation stores its traces, <code>--check</code> recomputes the summary
from them, and <code>policy_reeval_traces.py</code> re-runs a set of evaluations and compares them with the originals.</p>
<p>The trajectory-health scorecard (<code>trajectory_health.characterize_trajectory</code>) grades late and idle fingers, a
drop, jitter, de-centring and over-clamp. <code>policy_filmstrip.py</code> tiles phase-aligned frames of a rollout. The
frames are looked at before the rollout&#8217;s numbers are explained.</p>"""),
    ("backends", "Simulator backends",
     """<p>MuJoCo 3.6.0 on the CPU, pinned in <code>pyproject.toml</code>, runs the CEM evaluator, the grasp fitter, the
open-loop carry, the UR5e chain, the plan export and the clearance gate. MuJoCo-Warp through mjlab 1.2.0
(<code>uv run --extra rl --extra gpu</code>) runs PPO training, policy evaluation and the batched contact benchmarks. Newton
(hydroelastic contact under its MuJoCo solver) and Drake (SAP hydroelastic contact on the CPU) run in their own
environments for the comparisons of the <a href="contact_model.html">fingertip contact overview</a>. Drake also runs the
SR&#178; planning prototype (<code>drake_sr2_hand.py</code>). MJX runs on the CPU in the project environment and is used by two
non-default CEM optimizers and the DiffMJX scripts of 2026-10-07, whose runbook is the README of
<code>docs/experiments/20261007-diffmjx/</code>.</p>"""),
]

FIGURES = {}

FUTURE = [
    "Give <code>rl_render_rollout.py</code> and <code>policy_transfer_probe.py</code> the run&#8217;s tool contact class, as "
    "<code>policy_eval_suite.py --object-contact run</code> has it. Today they render and probe a policy trained before "
    "2026-09-20 on the scene&#8217;s class, where 11 of 16 such policies drop the tool at the trained pose. A render of D7 "
    "that holds the tool through step 249 shows the change.",
    "Store traces from the UR5e chain probe (<code>real_v1_chain_policy.py</code>) and the CPU transfer probe in the format "
    "of <code>traces/&lt;run&gt;.npz</code>, so their figures can be redrawn without a rerun.",
    "Run each unused grasp objective (contact targets, force closure, synergy CEM) and <code>morphohand.backends</code> once "
    "in a test on the real_v1 base scene. An objective that fails the test moves to <code>scripts/archive/</code> with its code.",
]

PAGES = [
    ("Design and grasp", "20260828-real_v1_search/20260828-real_v1_design_search.html", "Design search and the carry screen"),
    ("Policies and evaluation", "20260917-d6_cal_60M/20260917-d6_reorient_policy_60M.html", "The run&#8217;s timing in the evaluator"),
    ("Policies and evaluation", "20260919-hands_tranche/20260919-reorient_policies_across_hands.html",
     "Evaluation on the training plant, traces"),
    ("Policies and evaluation", "20260920-robust_tranche/20260920-robust_reorient_policies.html", "Training pipeline tables"),
    ("Policies and evaluation", "20260908-compute_budget/20260908-compute_budget.html", "Training throughput"),
    ("Simulators", "20261005-contact_bed/20261005-contact_model_bed.html", "Contact comparison bed"),
    ("Simulators", "20261001-drake-port/20261001-drake_sr2_planning.html", "Drake planning prototype"),
    ("Simulators", "20261007-diffmjx/20261007-contact_gradients_real_v1.html", "DiffMJX contact gradients"),
]
