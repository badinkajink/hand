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
State: 11-04 17:3x (autopilot dac2b73b, colorado): steps 1, 2 done (mount/mount.json: only the 11/22 and the 7.4x5x11/22
mount on the v4b parts; the other ten tips run on the envelope alone). Running detached (nohup setsid), nothing to relaunch
unless a [resguard] line names it: 7 screen workers logs/20261104-fingertip_shape_rerun/queue/worker.sh on cores 4-10
(queue.txt, 228 jobs; rows docs/experiments/20261104-fingertip_shape_rerun/{screen,traj}/; per-job times queue/jobs.jsonl;
failures failed.txt); 1 bed worker worker_bed.sh on core 11 (bed.txt; rows bed/<tip>/). J1 done for all 12 tips at 17:2x
(scripts/fingertip_shape_rerun_analysis.py screen). Memory tight (8.5 GB available, swap 13.9 GB): add no job beside them.
Drake replays at 2.5 MPa take 156 s each. Finding: in the cad_v2 cells' HOM turns the cylinder meets the cap TW15H3's end
plate (tip frame z -12 mm, past the pad's end), 24-46 % of the normal force (median per tip; rows' rigid_share,
rigid_by_part); 0 % on the D1-D8 real_v1 hands; report beside J. Written, not yet run: fingertip_shape_rerun_cost.py
(RL scenes built in logs/.../rl/; cpu/export/time after the screen frees memory), fingertip_shape_rerun_fe.py (one tip
at a time, femtip, --mem 5G, after the screen). Next: figures + page builder (fingertip_shape_rerun_page.py), stage 3
queue when the screen ends (python3 scripts/fingertip_shape_rerun_analysis.py screen, then
python3 scripts/fingertip_shape_rerun_queue.py stage3 --append).
<!-- ap:steps: rendered by ~/.claude/autopilot/ap from its ledger; change it with `ap step`, not by hand -->
Steps, in order (the ledger; `ap step done <job> <id> "result"` marks one, `ap step add <job> "text"` adds one):
- [x] 1. Before any run: wait until the HOM chain is done and the GPU holds no other job's process (nvidia-smi; the HO… Done 11-04 16:35: 11-04 16:30: HOM jobs of 11-02..11-04 all done, no HOM process; GPU 1.2/16 GB, desktop only; compute ledger logs/20261104-fingertip_shape_rerun/compute.jsonl (wall, core-h, GPU-h per step)
- [x] 2. Tips: the current v4b tip (11/22 half-ellipsoid on cad_v2) as the reference; the as-built block and the 10-14… Done 11-04 16:54: 12 tips (fingertip_shape_rerun_common.TIPS; the 11/22 candidate is the v4b reference, design surface; hemisphere tcd_ell7.4x7.4x7.4, design surface) + tcd_sph10.55 for the bed. Mount check (scripts/fingertip_shape_rerun_mount.py, mount/mount.json): only the 11/22 and the 7.4x5x11/22 mount on DIP link b v4 + cassette 3T + TW15H3 as the 11/22 does; blocks, capsules 7.4/8.5 and the 16/32 put 9.5-78 mm3 of TPU into the cassette walls or cap side blocks; capsule 6 and the hemisphere lose 21-100 % of a part's bearing surface; quarter-ellipsoids end in front of the cassette and cannot take 05R4's magnets. Those 10 are simulated on the envelope alone (on the v4b finger's rigid parts).
- [ ] 3. Contact bed per tip at 2.5 MPa, pad builder v2, against the law and Drake: approach, twist-onset arm, spin arm,
    brake force (10-15 step 2's protocol).
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
<!-- /ap:steps -->
Claimed: docs/experiments/<run date>-fingertip_shape_rerun/, logs/<run date>-fingertip_shape_rerun/, new scripts
scripts/fingertip_shape_rerun_*.py, docs/notes/20261103-fingertip_shape_aside.md.
Waiting on owner: none.
Do not: change the 10-14, 10-15 or 10-18 pages or rows; touch the HOM jobs' files or queues, the lab files (lab/,
QUEUE.md), the C6 board or the contact viewer on :8790; run beside another GPU job; publish artifacts; push.
