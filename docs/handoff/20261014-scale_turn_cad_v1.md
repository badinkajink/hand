Status: done
Task (owner, 2026-10-14 01:08): "can you now try to rerun this work, but with the newest CAD-faithful hand: /home/humanoid/Programs/hand/assets/mjcf/experimental/20261013-cad_v1? don't delete anything, if anything just replicate the page but with the updated CAD results". Deliver a new dated page beside the real_v1 one (which stays as it is) and publish it through a personal publish job.
Read first: docs/experiments/20261014-scale_turn_cad_v1/20261014-scale_turn_cad_v1.html (the CAD page); scripts/scale_turn_cad_v1_page.py (its builder); docs/experiments/20261014-scale_turn_cad_v1/run/ (the run recipe: hom_grids.sh, queue_cpu.sh, queue_gpu.sh, reruns.sh, films_cad_camera.sh); docs/handoff/20261014-publish_scale_turn_cad_v1.md (the publish job).
State (2026-10-14 15:50): done. Films re-rendered from 10 deg below the grasp (the same results as the first films), page rebuilt (7.45 MB) and committed; docs/handoff/20261014-publish_scale_turn_cad_v1.md armed on the personal account to publish it as a new artifact and fill the INDEX row.
Next:
1. When films_cad_camera.log says "queue done": rebuild (`SCALE_HAND=cad_v1 .venv/bin/python scripts/scale_turn_cad_v1_page.py`), look at two frames of each new film (ffmpeg -ss 3 and 6 s), check the captions against the new films.jsonl rows; commit the films, rows and page; push.
2. Arm docs/handoff/20261014-publish_scale_turn_cad_v1.md: `ap arm <note> --account personal --pin --week-room 100 --parallel --model sonnet --effort medium --at now`.
Claimed: scripts/scale_turn_cad_v1_page.py, scripts/scale_turn_cad_v1_page.template.html, docs/experiments/20261014-scale_turn_cad_v1/, logs/20261014-scale_turn_cad_v1/.
Waiting on owner: none for these steps.
Do not: change the real_v1 page, its builder or its rows (docs/experiments/20261013-scale_turn_controllers/); run anything without SCALE_HAND=cad_v1 into the CAD folders; launch past a closed resguard gate.
