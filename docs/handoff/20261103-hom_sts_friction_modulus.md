Status: done
Task: Analyse the STS3032 friction and modulus sweeps of the HOM landscapes (owner 2026-11-02 17:4x: "we can push the
contact repo and run the friction and moduli sweep for the sts3032 ... trigger an autopilot run on this account for
analysis after 3am and be token efficient"). Add the result to the plant and modulus page of
hand-20261102-hom_plant_modulus_page.
Read first: logs/20261102-hom_servo_plant/README.md; gpu_queue_sts_sweep.sh and its log; sts_sweep_transfer.log;
docs/handoff/20261102-hom_plant_modulus_page.md (the page, its builder scripts/hom_plant_page.py and what it says);
memory project_hom_plant_dependence_2026_11_02.md.
State (03:3x on 11-03): done. Queue finished 01:10 (24 grids ok); rows 5d19cea3f (merge 583ceab48). Friction 0.4 vs
1.0 Spearman 0.33/0.64/0.40/0.27 (twist/roll/pin-push/press); STS3032 modulus 1.25-10 vs 2.5 MPa 0.85-0.98. The page
job never ran (six "Login expired" launches on personal), so this job built the page:
docs/experiments/20261029-foundational_pose_review/20261102-hom_plant_modulus.html (scripts/hom_plant_page.py,
f84e2cd1c). Talk servo notes: contact 068a478. README and memory updated; direction 7 added to the page job's ledger.
<!-- ap:steps: rendered by ~/.claude/autopilot/ap from its ledger; change it with `ap step`, not by hand -->
Steps, in order (the ledger; `ap step done <job> <id> "result"` marks one, `ap step add <job> "text"` adds one):
- [x] 1. Check the queue: "sts sweep grids and transfer done" in its log, 24 Q*.done files, no "failed" or "refused" l… Done 11-03 03:06: queue done 01:10: 24 of 24 grids, no failed or refused, scenes at their friction (0.4, 0.7, 2.4 in metas and XML) and modulus (1.25, 5, 10 MPa), row counts equal to the friction-1.0 reference, all rows ok; both transfer json written
- [x] 2. Commit and push the rows (rows/grid2dff_v4b_k21mu0p4, _k21mu0p7, _k21mu2p4, _k21E1p25, _k21E5, _k21E10 and th… Done 11-03 03:08: rows and both transfer json committed as 5d19cea3f, merged and pushed as 583ceab48 (495 MB LFS); message states friction Spearman vs 1.0 (0.27-0.64 at 0.4) and modulus 0.85-0.98 vs 2.5 MPa, one-design losses
- [x] 3. Add the friction and STS3032 modulus results to the page (Spearman against friction 1.0 and 2.5 MPa; one desi… Done 11-03 03:25: page built (the page job never ran: expired personal login): 20261102-hom_plant_modulus.html from scripts/hom_plant_page.py, pushed f84e2cd1c; friction 0.4 vs 1.0 Spearman 0.27-0.64, STS modulus 0.85-0.98, one design over friction 0.4-1.0 loses <=0.13, over 1.25-5 MPa <=0.07; talk servo notes updated, contact 068a478 pushed
- [x] 4. Append the results to logs/20261102-hom_servo_plant/README.md and to the memory file. Done 11-03 03:25: README 03:2x paragraph appended (friction and modulus Spearman, above-0.5 shares, one-design losses, film, page, talk); memory project_hom_plant_dependence_2026_11_02 (bullets + description), MEMORY.md line, and the login memory's recurrence note updated
<!-- /ap:steps -->
Claimed: rows/grid2dff_v4b_k21mu*, rows/grid2dff_v4b_k21E*, rows/20261102-sts_*_transfer.json, the plant and modulus page
and its builder (after hand-20261102-hom_plant_modulus_page finishes)
Waiting on owner: `~/.claude/autopilot/ap login personal`, then re-arm hand-20261102-hom_plant_modulus_page for
its remaining work (fresh-seed check of the one-design picks, trends page limitation, the talk's selection slide).
Do not: publish artifacts; start new multi-hour GPU grids; edit the crosstalk poller's Claimed files.
