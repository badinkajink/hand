Status: active
Task: Owner, 2026-11-03 19:3x: "we can add this rerun to shore up the numbers and documentation to the queue". Rerun the
fingertip-shape screens of 2026-10-14 and 10-15 and the FE calibration of 10-18 on the current hand and models, across
the axes the owner named that evening: "FE sensing, sphere packing / sim compute time, task level performance
[grasping, hom control], worst-case-sampling domain randomization robustness ... across the ellipsoid, hemisphere,
capsule shapes". The owner chose the 11/22 half-ellipsoid on 10-18 ("i do like the half ellipsoids most, the 11/22 one
especially"); this job documents that choice with current numbers. The sensor is a cited instrument, claim nothing
new about it (owner 11-03). Priority (owner 11-03): after the HOM chain on the GPU, before the scripts cleanup.
Read first: docs/notes/20261103-fingertip_shape_aside.md (the old numbers, the overview aside);
docs/experiments/20261014-tip_codesign/20261014-sphere_pad_accuracy_and_fingertip_codesign.html and
docs/notes/20261014-tip_codesign_plan.md (stage score, J1-J3, the Drake gate);
docs/experiments/20261015-curved_tips/20261015-capsule_and_ellipsoid_fingertips.html and docs/handoff/20261015-curved_tips.md
(how it ran: scripts/curved_tips_*.py, tip_codesign_*.py, the screen workers and queue files, the cost protocol, FE);
docs/experiments/20261018-fe_contact_calibration/20261018-fe_contact_calibration.html; docs/experiments/20261030-v3_tip_contact_model/;
docs/experiments/20261029-foundational_pose_review/20261102-hom_plant_modulus.html; CLAUDE.md sections "Fingertip on the
CAD hand", "Pad modulus 2.5 MPa" and "Servo plant of the HOM scripts". If scripts/ was reorganized, find each script by
name in scripts/README.md.
State: 11-04 19:5x (autopilot dac2b73b, colorado): steps 1, 2, 3, 10, 11 done. Screen done 19:56 (screen/summary.json):
J2 16/32 half-ellipsoid 0.252, capsule 8.5 0.203, as-built block 0.199, capsule 7.4 0.184, v4b 0.134 (its worst: friction
0.5), hemisphere 0.074; stage-3 tips 16/32, hemisphere (rule 3, three-finger share 0.49), capsule 8.5, as-built, v4b.
Detached, nothing to relaunch unless a [resguard] line names it or a log shows a crash:
- stage 3 on 4 workers queue/worker.sh (cores 4-7; queue.txt, 40 jobs from 19:56; rows final/);
- chain.sh: stage-3 summary and films after stage 3; GPU cost running (core 19, gpu.log); its FE pass waits on the per-tip
  locks of fingertip_shape_rerun_fe.py behind fe_now.sh (core 11, tip 6 of 11 at 19:57), then the CPU cost (cores 13-15);
- drake_fill.sh (core 12): the v4b tip's 4 Drake replays that failed on its design hull (degenerate facet; D2 s0, D8 s0-2),
  on the 15 October printed-upright hull -> traj/tcd_ell7.4x8.5x11x22/drake_fill_u0.1.jsonl; relabel those rows' tip to
  tcd_ell7.4x8.5x11x22 (keep drake_surface) before the traj and screen summaries are rerun.
Step 12 (band): scripts/fingertip_shape_rerun_band.py; the v4b tip's caches built and identical to the 11-04 scenes; run its
validate as the check, then the stage-3 tips, after the chain's GPU cost (one GPU job at a time). Page: protocol and bed
prose written; the rest waits for its summaries.
<!-- ap:steps: rendered by ~/.claude/autopilot/ap from its ledger; change it with `ap step`, not by hand -->
Steps, in order (the ledger; `ap step done <job> <id> "result"` marks one, `ap step add <job> "text"` adds one):
- [x] 1. Before any run: wait until the HOM chain is done and the GPU holds no other job's process (nvidia-smi; the HO… Done 11-04 16:35: 11-04 16:30: HOM jobs of 11-02..11-04 all done, no HOM process; GPU 1.2/16 GB, desktop only; compute ledger logs/20261104-fingertip_shape_rerun/compute.jsonl (wall, core-h, GPU-h per step)
- [x] 2. Tips: the current v4b tip (11/22 half-ellipsoid on cad_v2) as the reference; the as-built block and the 10-14… Done 11-04 16:54: 12 tips (fingertip_shape_rerun_common.TIPS; the 11/22 candidate is the v4b reference, design surface; hemisphere tcd_ell7.4x7.4x7.4, design surface) + tcd_sph10.55 for the bed. Mount check (scripts/fingertip_shape_rerun_mount.py, mount/mount.json): only the 11/22 and the 7.4x5x11/22 mount on DIP link b v4 + cassette 3T + TW15H3 as the 11/22 does; blocks, capsules 7.4/8.5 and the 16/32 put 9.5-78 mm3 of TPU into the cassette walls or cap side blocks; capsule 6 and the hemisphere lose 21-100 % of a part's bearing surface; quarter-ellipsoids end in front of the cassette and cannot take 05R4's magnets. Those 10 are simulated on the envelope alone (on the v4b finger's rigid parts).
- [x] 3. Contact bed per tip at 2.5 MPa, pad builder v2, against the law and Drake: approach, twist-onset arm, spin ar… Done 11-04 19:29: Bed at 2.5 MPa, 13 tips (bed/<tip>/, bed/summary.json, media/fig_bed.png; 2.4 core-h): law twist-onset arm at 1 N 1.24-1.42 mm on the curved tips (10 MPa: 0.93-1.00) against 2.82 on the blocks; builder-v2 pads 0.86-0.94 of the law's onset arm on the curved tips, 0.98-0.99 on the blocks, Drake 0.82-1.00; spin arm at 1 rad/s pads/law 0.90-1.19 (v4b 0.99, quarter-ellipsoids 1.13-1.19, blocks 1.16-1.21), Drake 0.98-1.14; pads' spin arm 3 over 0.3 rad/s 0.97-1.04 curved, 1.64-1.67 blocks; brake force pads/Drake 0.94-1.07 curved, 0.71-0.84 blocks; 16-24 pads touching per side at 1 N (10 MPa: 10-12), blocks 44 (20).
- [ ] 4. Simulation cost per tip: CPU grip step, GPU world-step and GPU memory at 2048 worlds (the U1 protocol of
    docs/experiments/20261014-pad_usability/).
- [ ] 5. Task screen on cad_v2 at 2.5 MPa with the SCS0009 plant (HOS_PLANT=servo, a fresh HOS_SCENES, HOM_ROWS_TAG): J1, J2
    over the five friction and servo conditions, the Drake agreement gate on the D1-D8 replays, J3 on fresh placements
    for the finalists against the reference. For the finalists, also the worst-case score over a band of scene
    frictions from hand-20261104-hom_worst_case_score (its band option of scripts/hom_grasp_refine.py), if that job
    delivered it.
- [ ] 6. FE per tip: approach and stiffness map under the 12.5 mm rod and the 2 mm sphere; force resolution at the contact
    point with the four-magnet 05R4 layout (patch-load FE and scripts/tpu_tip_magnet_sensing.py, as the 10-19/20 checks
    did for the 11/22 and the block), wherever the layout fits the shape; say where it does not.
- [ ] 7. Result page docs/experiments/<run date>-fingertip_shape_rerun/ (builder from scripts/curved_tips_page.py): every axis
    per tip, the 10-14/10-15/10-18 numbers beside the new, films of the finalists' HOM turns; meta tags; the docs_index
    refresh; commit with explicit paths.
- [ ] 8. Overview aside: put the new numbers into the aside of docs/notes/20261103-fingertip_shape_aside.md in the overview's
    section print. While hand-20261027-crosstalk_feed_poller claims the overview, give it the text with
    `ap step add hand-20261027-crosstalk_feed_poller "..."` instead of editing.
- [ ] 9. Status page: a row in docs/experiments/20261018-printed_fingertip_status/status.json, then rebuild
    scripts/printed_fingertip_status_page.py.
- [x] 10. Resguard events 16:41 and 16:52 (11-04): the mount check's first version (trimesh filled voxelization of the … Done 11-04 16:54: Reported in the ledger and compute.jsonl; mount check rewritten (ray parity, 8 s, 0.40 GB); nothing else affected
- [x] 11. Swap gate (11-04 17:36): resguard's launch gate closed on swap in use 15.5 GB > 15 GB, held by the desktop (V… Done 11-04 18:12: Gate reopened 18:12 (swap 14.7 GB, then 13.5); the 8 screen workers, the bed worker and the chain resumed by themselves. They wait in gate() whenever it closes again; nothing to relaunch.
- [ ] 12. Worst-case friction band for the stage-3 tips (step 5's last sentence): hom_grasp_refine.py validate --band k21mu0p4,k21mu0p7,k21 --picks-from k21 (the v4b tip's band picks, STS3032 at friction 0.4/0.7/1.0) with each tip in place of the v4b pad. Needs a scene builder fingertip_shape_rerun_scenes.py on the pattern of scripts/hom_modulus_scenes.py with hand_object_scale_sim.TIP set to the tip: the REF scenes of sphere:40 and cylinder:25 per friction into logs/20261104-fingertip_shape_rerun/hom_scenes/<tip>_mu<mu>/ (HOS_SCENES), and validate run through a wrapper that sets S.TIP in the parent (open poses on the tip's pt mesh); HOM_ROWS_TAG _fsr_<tip>. GPU about 6 min per tip; one GPU job at a time, after the chain's GPU cost. Report the lowest of the three frictions' scores per task against the v4b tip's on the same picks.
<!-- /ap:steps -->
Claimed: docs/experiments/<run date>-fingertip_shape_rerun/, logs/<run date>-fingertip_shape_rerun/, new scripts
scripts/fingertip_shape_rerun_*.py, docs/notes/20261103-fingertip_shape_aside.md.
Waiting on owner: none.
Do not: change the 10-14, 10-15 or 10-18 pages or rows; touch the HOM jobs' files or queues, the lab files (lab/,
QUEUE.md), the C6 board or the contact viewer on :8790; run beside another GPU job; publish artifacts; push.
