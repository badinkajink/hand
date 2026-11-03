Status: active
Task: Result page on how the HOM design landscapes depend on the finger servo and the pad modulus (owner 2026-11-02 11:1x:
"characterize the change in landscape between the idealized 4nmrad plant, the current scs0009 calibration, and the ideal
sts3032 calibration ... there seems to be no transfer between layouts, controls optimized for one plant to another. is that
really true?"; 11:4x: "seems like we need to queue up modulus studies across a range as well"). All runs finished 17:30;
the findings are in the rows, the queue README and chat, and have no page yet.
Read first: logs/20261102-hom_servo_plant/README.md (every run, tag and number); logs/20261102-hom_servo_plant/transfer.log
and modulus_transfer.log; scripts/hom_plant_transfer.py, scripts/hom_plant_compare.py, scripts/hom_modulus_scenes.py;
docs/experiments/20261029-foundational_pose_review/20261030-hom_layout_trends.html and its builder scripts/hom_trends_page.py
(the newest HOM page: copy its components, maps and films); memory project_hom_plant_dependence_2026_11_02.md; CLAUDE.md
"Result pages"; ~/.claude/writing_reference.md.
State: rows pushed as d106d60cf (SCS0009 at 2.5 MPa: grids, EA at every size, CPU checks), 3d62ecd39 (STS3032 kp 2.1 grids,
three-plant transfer), 671a4d65c (SCS0009 at 1.25/5/10 MPa, work plant at 2.5 MPa; rows/20261102-modulus_transfer.json,
rows/20261102-plant_modulus_factorial.json). Code 03f1952fd, 245a89c58, 1d46fc865, d1225cfdb. Size-trend figures
media/20261030-size_*_sv25.png. Findings: on one plant, 2.5 vs 10 MPa Spearman 0.77-0.97; between plants at one modulus
-0.36 to 0.60; one layout for three servos loses <= 0.11 with the grasp re-picked; over 1.25-5 MPa one layout and grasp
lose <= 0.05 except the pin-push grasp (11 at 1.25 MPa, 12 above). The picks are three-seed means, not yet rerun.
Next:
1. Fresh-seed check of the one-design picks (the "robust" entries of rows/20261102-*transfer*.json and the factorial
   json): run each picked layout and grasp at seeds 3-7 on each grid's plant and scene cache (HOS_PLANT, HOS_SCENES,
   HOM_ROWS_TAG as in gpu_queue_modulus.sh; hom_gpu.py on those layouts only, or a short script), one GPU job at a time
   through resguard (--mem 3.2G: the grid jobs peaked at 2.44 GB) with the queues' retry on exit 3. Rows to
   rows/20261102-robust_fresh_seeds.jsonl. Report each pick's fresh-seed mean against its three-seed mean.
2. Page docs/experiments/20261029-foundational_pose_review/20261102-hom_plant_modulus.html from a new builder
   scripts/hom_plant_page.py and template: Terms and metrics; the three plants (gain, torque limit, why) and the moduli
   (calibration 2.22-2.73 MPa per tip; force ratio 1.54-0.49); Spearman and best-tenth tables; one-design tables (own,
   one layout, one layout and grasp) with the fresh-seed check; corrected size slopes and CPU checks
   (rows/cpu_best/20261102-plant_compare.json; README 12:4x); landscape maps per plant and modulus (hom_landscape views);
   a film of the shared twist and press designs on the three plants; Future work (STS3032 modulus sweep, friction sweep).
   docs_index tags; prose_lint; look at every figure at its displayed size. Commit and push to hand main.
3. Trends page (scripts/hom_trends_page.py): its limitation (work plant at 10 MPa) links the new page and quotes the
   corrected x slopes 1.41, 0.57, 1.20, 1.09 (work plant 0.98, 0.59, 0.98, 1.11). Rebuild, commit, push.
4. Talk (private contact repo, talks/20261117-hand_design/build_talk.py): the selection and servo slides gain the result
   (one layout across three servos within 0.11; 1.25-5 MPa changes designs by <= 0.05 except the pin-push grasp). Rebuild;
   commit under flock ~/.cache/contact-slides/git.lock; do not push.
Claimed: scripts/hom_plant_page.py, scripts/hom_plant_page.template.html,
docs/experiments/20261029-foundational_pose_review/20261102-hom_plant_modulus*, rows/20261102-robust_fresh_seeds.jsonl,
scripts/hom_trends_page.py, docs/experiments/20261029-foundational_pose_review/20261030-hom_layout_trends.html,
contact repo talks/20261117-hand_design/
Waiting on owner: push of the contact repo's talk commit; whether to sweep the modulus on the STS3032 plant (about 1 h of
GPU per modulus) and the scene friction (grip tape 1.0-2.4 in the scenes, 0.37-0.44 for TPU on bare PLA).
Do not: publish the talk or anything from it (private; contact repo only); publish artifacts (local pages only); start
multi-hour GPU grids; push the contact repo; edit the crosstalk poller's Claimed files.
