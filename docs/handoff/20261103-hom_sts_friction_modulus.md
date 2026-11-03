Status: active
Task: Analyse the STS3032 friction and modulus sweeps of the HOM landscapes (owner 2026-11-02 17:4x: "we can push the
contact repo and run the friction and moduli sweep for the sts3032 ... trigger an autopilot run on this account for
analysis after 3am and be token efficient"). Add the result to the plant and modulus page of
hand-20261102-hom_plant_modulus_page.
Read first: logs/20261102-hom_servo_plant/README.md; gpu_queue_sts_sweep.sh and its log; sts_sweep_transfer.log;
docs/handoff/20261102-hom_plant_modulus_page.md (the page, its builder scripts/hom_plant_page.py and what it says);
memory project_hom_plant_dependence_2026_11_02.md.
State: gpu_queue_sts_sweep.sh started 19:26 on 11-02, about 62 min a set, six sets (friction 0.4, 2.4, 0.7 at 2.5 MPa;
moduli 1.25, 5, 10 MPa at friction 1.0), expected done about 01:40; it then writes sts_sweep_transfer.log and
rows/20261102-sts_friction_transfer.json, rows/20261102-sts_modulus_transfer.json. Reference: the STS3032 at friction 1.0
and 2.5 MPa (rows/grid2dff_v4b_k21). Scripts pushed as 91674f5d1.
Next:
1. Check the queue: "sts sweep grids and transfer done" in its log, 24 Q*.done files, no "failed" or "refused" left
   unretried, each set's scene check "4 of 4" (or more) at its modulus and friction. Rerun a failed grid alone.
2. Commit and push the rows (rows/grid2dff_v4b_k21mu0p4, _k21mu0p7, _k21mu2p4, _k21E1p25, _k21E5, _k21E10 and the two
   transfer json files) with a commit message that states the result in numbers.
3. Add the friction and STS3032 modulus results to the page (Spearman against friction 1.0 and 2.5 MPa; one design over
   friction 0.4-1.0 and over 1.25-5 MPa), rebuild, commit, push. Then add one line to the talk's servo slide if the
   result changes what it says (contact repo; push allowed).
4. Append the results to logs/20261102-hom_servo_plant/README.md and to the memory file.
Claimed: rows/grid2dff_v4b_k21mu*, rows/grid2dff_v4b_k21E*, rows/20261102-sts_*_transfer.json, the plant and modulus page
and its builder (after hand-20261102-hom_plant_modulus_page finishes)
Waiting on owner: none
Do not: publish artifacts; start new multi-hour GPU grids; edit the crosstalk poller's Claimed files.
