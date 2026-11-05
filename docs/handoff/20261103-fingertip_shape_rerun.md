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
State: 11-04 19:4x (autopilot dac2b73b, colorado reset 19:20): steps 1, 2, 3, 10, 11 done (bed: bed/summary.json,
media/fig_bed.png). Detached, nothing to relaunch unless a [resguard] line names it or a log shows a crash:
- 8 screen workers queue/worker.sh (cores 4-10, 12; queue.txt ~50 jobs left at 19:40, mostly Drake replays);
- queue/fe_now.sh on core 11 (one thread): the FE of the 11 tips other than the v4b tip, design-surface keys (the v4b
  tip's rows are in fe/ from the 19:3x smoke run; a run on the block's printed key was stopped at 19:38, see
  fe_now_printed_keys_stopped.log); progress fe_now.log, fe_jobs.jsonl;
- queue/chain.sh: after the screen, the screen summary, stage 3 on 4 workers (cores 4-7), the stage-3 summary, the films;
  the GPU cost (core 19); the FE again (skips done tips) and the CPU cost (cores 13-15). Log chain_done.log.
Changes 19:3x-19:4x: finalist rule (2) uses the replay cost paired over placements both tips hold (the v4b tip's pads
drop the tool in about half the replays); the FE runs on design surfaces as 10-18 did; the film camera looks from below the
palm (az 240, el 15). Step 12 (band): scripts/fingertip_shape_rerun_band.py written; the v4b tip's caches built in
logs/.../hom_scenes/ and identical to the 11-04 band's scenes; next: `validate --tip tcd_ell7.4x8.5x11x22 --task <t>`
for the 4 tasks as an end-to-end check against rows/band_k21mu/val/, then the stage-3 tips (scenes, then validate), one GPU
job at a time and never beside the chain's GPU cost. Page: scripts/fingertip_shape_rerun_page.py has the protocol and bed
prose; screen, cost, FE, lede and future work wait for their summaries.
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
