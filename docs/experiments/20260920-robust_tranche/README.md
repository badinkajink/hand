# Robust continuation tranche (2026-09-20)

Driver: `nohup setsid python3 scripts/hands_tranche_queue.py --queue docs/experiments/20260920-robust_tranche/queue.json >> logs/20260920-robust_tranche.log 2>&1 &`
(started 09:23 MDT; 11 jobs, ~38 min each). Same per-job outputs as `20260919-hands_tranche/` plus
`<id>_eval_j3dr.json` (64 rollouts at ±3 mm / ±10° spawn jitter with friction DR).

## What changed against the 2026-09-19 tranche

| item | 2026-09-19 | here | why |
|---|---|---|---|
| tool spawn | one nominal pose | ±3 mm x/y, ±10° yaw | the 09-19 policies collapse under that jitter (D7 0/64, D6 clip s1 8/64, D5 26/64, D1 20/64) |
| tool | 25 mm cylinder | + screw-tip mesh (+1 g) | the chain's and the real tool have it; `make_tipmesh_morphology_run.py` |
| floor | mjlab plane (μ 1, solref 0.02) | the scene's floor (μ 1.8, solref 0.006), `--scene-floor` | the tool lies on it through closure and lift |
| tool contact | MuJoCo defaults (solref 0.02, solimp 0.9–0.95) | the scene's (0.006, 0.97–0.995) | `env_build.make_object_spec_from_frozen` had dropped them since the first mjlab run |
| warm start | D6 60 M checkpoint | each hand/arm's own 09-19 checkpoint | 20 M steps of continuation on the matched plant |

## Decision log
- 09:23 launched; order: D6 clipsep, D3, D5, D1, D6 (clip s2), D4, D7, D2 clip, then D3/D5/D1 clipsep.
- 11:40 page builder `scripts/robust_tranche_page.py` (+ template) written; page published
  https://claude.ai/artifact/NZLr9wdDAWjZC1HM6ed6ed (same URL from now on; videos in web/ as supporting files).
  A watcher (`logs/20260920-robust_page_watch.log`) rebuilds the local page every 10 min while the driver lives
  and once after it exits; republishing the artifact needs a session (Artifact tool, same file path).
- After 3 of 11: D5 jittered 23->57 of 64 and holds the chain (parent dropped); D6 clipsep 60->51, D3 51->50;
  every continued policy ends 30-45 deg short of vertical (cos 0.82-0.83 vs parents 0.92-0.96) -- see the page's
  next-steps item 2 (60 M continuation of one job decides converged-vs-jitter-cost).

- 12:31 r_d4_clip aborted at iteration 9: NaN in the actor observation (tip_lost 29.6/episode before it) on the g12
  tip-mesh scene; `r_d4_clip_s1` appended at the queue's end as a second draw (`--seed` is a dead flag: rl_train_cube.py
  parses it and never applies it; every run is an unseeded draw).
- 12:45 page v2: the training-pipeline section (stages, episode timeline, obs/action, reward, terminations, PPO, lineage,
  decision register measured-vs-inherited, evaluation stack) built from the first finished run's config.yaml.
- 12:55 page v3/v4: the chain's post-turn seams per policy and plate; the D6 clip plate-0 chain film. Finger commands and
  angles recorded through that run (`r_d6_clip_chain_pl0_ctrl_qpos.json`): the gait's commands move 5-27 deg, the fingers
  0.3-9 deg (index pip 59 deg behind), -0.002 turns; `ok` tests stance/grip/cycles, not turns. Plate 0 = the chain
  scenes' original geometry; plate 25 (`plate_variant`, only the plate geom moves) = the built hand = the training scenes.
- 14:20 7 of 12 done. Plate-25 chain films for D6 clip, D5 clip, D6 clipsep (the three that hold the turn): each stages at
  14-27 deg off vertical, the tip (50 mm lever) lands 12-23 mm from the aimed centre against a 6 mm capture radius and
  misses the hole; D6 clip then tips off at the handover grip, D5/D6 sep lean on the post at 47-55 deg. At plate 0 D6 clip
  staged at 8 deg (7 mm) and entered. Next measurement: a staging gate (cos >= 0.99 before the descent). Page v5.
- D7 continued: nominal 37/64 (parent 64), jittered 1->35; its strip shows the tool sliding out during the lift and
  standing on the floor at step 58. D2: nominal 64 at cos 0.65, turns 40-50 deg and holds, as its parent.
