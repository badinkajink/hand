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
State: 11-04 21:2x (autopilot dac2b73b, colorado): steps 1-6 and 10-12 done (band committed 41d7152d3). Results in
docs/experiments/20261104-fingertip_shape_rerun/ (mount, bed, screen, final, fe, cost, band, media). Stage 3: no tip beats
the v4b tip (J3 0.195; 16/32 +0.064 [-0.017, +0.160]); band: the v4b tip's S3 picks score 0.02-0.90 lower per task on
the other stage-3 tips. No jobs running. Step 7 in progress: full page build done once (indices refreshed, not yet
committed; prose 341 flags per 1000), builder prose rewritten (32 edits, terms Friction band and Cost added); rebuild,
lint, look at the figures, commit page + INDEX.md, docs/index.html, pages.html, pages.json, changelog.html and
docs/overviews/printed_fingertips.html. Then step 8 (aside to the crosstalk poller by ap step add; the note
docs/notes/20261103-fingertip_shape_aside.md) and step 9 (status.json row, rebuild the status page).
<!-- ap:steps: rendered by ~/.claude/autopilot/ap from its ledger; change it with `ap step`, not by hand -->
Steps, in order (the ledger; `ap step done <job> <id> "result"` marks one, `ap step add <job> "text"` adds one):
- [x] 1. Before any run: wait until the HOM chain is done and the GPU holds no other job's process (nvidia-smi; the HO… Done 11-04 16:35: 11-04 16:30: HOM jobs of 11-02..11-04 all done, no HOM process; GPU 1.2/16 GB, desktop only; compute ledger logs/20261104-fingertip_shape_rerun/compute.jsonl (wall, core-h, GPU-h per step)
- [x] 2. Tips: the current v4b tip (11/22 half-ellipsoid on cad_v2) as the reference; the as-built block and the 10-14… Done 11-04 16:54: 12 tips (fingertip_shape_rerun_common.TIPS; the 11/22 candidate is the v4b reference, design surface; hemisphere tcd_ell7.4x7.4x7.4, design surface) + tcd_sph10.55 for the bed. Mount check (scripts/fingertip_shape_rerun_mount.py, mount/mount.json): only the 11/22 and the 7.4x5x11/22 mount on DIP link b v4 + cassette 3T + TW15H3 as the 11/22 does; blocks, capsules 7.4/8.5 and the 16/32 put 9.5-78 mm3 of TPU into the cassette walls or cap side blocks; capsule 6 and the hemisphere lose 21-100 % of a part's bearing surface; quarter-ellipsoids end in front of the cassette and cannot take 05R4's magnets. Those 10 are simulated on the envelope alone (on the v4b finger's rigid parts).
- [x] 3. Contact bed per tip at 2.5 MPa, pad builder v2, against the law and Drake: approach, twist-onset arm, spin ar… Done 11-04 19:29: Bed at 2.5 MPa, 13 tips (bed/<tip>/, bed/summary.json, media/fig_bed.png; 2.4 core-h): law twist-onset arm at 1 N 1.24-1.42 mm on the curved tips (10 MPa: 0.93-1.00) against 2.82 on the blocks; builder-v2 pads 0.86-0.94 of the law's onset arm on the curved tips, 0.98-0.99 on the blocks, Drake 0.82-1.00; spin arm at 1 rad/s pads/law 0.90-1.19 (v4b 0.99, quarter-ellipsoids 1.13-1.19, blocks 1.16-1.21), Drake 0.98-1.14; pads' spin arm 3 over 0.3 rad/s 0.97-1.04 curved, 1.64-1.67 blocks; brake force pads/Drake 0.94-1.07 curved, 0.71-0.84 blocks; 16-24 pads touching per side at 1 N (10 MPa: 10-12), blocks 44 (20).
- [x] 4. Simulation cost per tip: CPU grip step, GPU world-step and GPU memory at 2048 worlds (the U1 protocol of docs… Done 11-04 21:01: Cost at 2.5 MPa (cost/summary.json; 0.57 GPU-h, about 3 core-h): GPU world-step at 2048 worlds 2.52 us for the v4b tip (65 contacts per world, 6.3 GB), blocks 3.98-4.24 us (106-115, 11.4-11.7 GB), capsules 2.78-3.13, other half-ellipsoids 3.10-3.23, quarter-ellipsoids 2.06-2.78, hemisphere 2.43 (10 MPa on 15 October: 11/22 1.72 us, 31 contacts; block 2.93). CPU step at the 16 mm cell's three-finger grip 179-415 us (v4b 221 with 96 contacts, as-built 241 with 162), 64-plan replanning on 3 threads 2.7-4.9 s.
- [x] 5. Task screen on cad_v2 at 2.5 MPa with the SCS0009 plant (HOS_PLANT=servo, a fresh HOS_SCENES, HOM_ROWS_TAG): … Done 11-04 20:26: Screen on the v4b hand at 2.5 MPa, SCS0009 (screen/summary.json, final/summary.json, media/fig_screen.png; 23.8 core-h, 3.5 h wall). J2: 16/32 half-ellipsoid 0.252, capsule 8.5 0.203, as-built block 0.199, capsule 7.4 0.184, v4b tip 0.134 (worst at friction 0.5), hemisphere 0.074; gate (v4b 16/24 -> 14) failed by the quarter-ellipsoids 8.5 11/22 and 5 11/22 and the hemisphere. Stage 3 (fresh placements, against the v4b tip's J3 0.195): 16/32 +0.064 [-0.017, +0.160], as-built -0.008 [-0.107, +0.105], capsule 8.5 -0.069 [-0.152, +0.029], hemisphere -0.132 [-0.209, -0.026]; no tip beats the v4b tip. The cells' HOM turns put 0.35-0.48 of the normal force on rigid parts (the cap's end plate). Friction band: step 12.
- [x] 6. FE per tip: approach and stiffness map under the 12.5 mm rod and the 2 mm sphere; force resolution at the con… Done 11-04 20:28: FE of 12 tips (fe/summary.json, fe/sensing.json, media/fig_fe.png; 1.75 core-h), the pad cut where the v4b pad is cut and held there (quarter-ellipsoids on their back planes), design surfaces, print orientation: 2 mm sphere at the apex 232-263 um at 1 N, the seven map points within 1.04-1.32 of each other; 12.5 mm rod across 155-187 um on the curved tips, 68 um on the blocks; rod along 42-45 um on the capsules, 116-182 on the ellipsoids and the hemisphere; twist arm under the rod 0.48-0.64 mm curved, 2.85 mm blocks; within 3 % of 18 October's whole prints. Pad 05R4 fits 9 tips: force resolution 0.032-0.037 N at the contact point (gain for the 26.8 mT field), magnet travel 5.3-6.5 um/N; the quarter-ellipsoids cannot take it (the magnets' tops 7.5 mm deep lie behind their back planes).
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
- [x] 12. Worst-case friction band for the stage-3 tips (step 5's last sentence): hom_grasp_refine.py validate --band k… Done 11-04 21:14: Band of 11-04 (STS3032 at scene friction 0.4/0.7/1.0, the v4b tip's refined designs on 30 layouts per task; band/summary.json; 0.69 GPU-h): the v4b repeat matches 11-04 (median paired difference 0.000, quartiles within 0.002). With another tip in place of the pad the S3 designs' worst-case medians fall: 16/32 by 0.06-0.26 per task, hemisphere 0.02-0.61, capsule 8.5 0.02-0.28, as-built block 0.28-0.90 (0 on every task: it holds at friction 1.0, S3 means 0.39-0.62, and loses grip at 0.4, 0.02-0.15). The designs were refined for the 11/22; the other tips' own designs were not refined.
<!-- /ap:steps -->
Claimed: docs/experiments/<run date>-fingertip_shape_rerun/, logs/<run date>-fingertip_shape_rerun/, new scripts
scripts/fingertip_shape_rerun_*.py, docs/notes/20261103-fingertip_shape_aside.md.
Waiting on owner: none.
Do not: change the 10-14, 10-15 or 10-18 pages or rows; touch the HOM jobs' files or queues, the lab files (lab/,
QUEUE.md), the C6 board or the contact viewer on :8790; run beside another GPU job; publish artifacts; push.
