Status: active (window 8955e04e, not armed; the session works it overnight)
Task: owner 2026-11-11 ~01:35, going to sleep: "take some agency and re-run whatever else you find in the slides that could
use a refresh/unification on the latest settings so that our results aren't so disjoint"; a v2 deck, an update in place,
or a universal toggle; "the actual plot/figure contents should stay absolutely the same, there should be no meta-level
discernment within them". Before that: the tuning chart under the tracking score with HOM tuning alone; how the 16 grasps
are chosen and that grasp tuning beyond them is not clear in the slides.
Settings of v2: every hand ranked by the tracking score J (hom_landscape.track_score); plant SCS0009 0.72 N m/rad, pads
2.5 MPa, MuJoCo 3.6, scenes logs/20261102-hom_servo_plant/scenes (the talk's); grids grid2dff_v4b_sv25r (all four tasks).
Output: a v2 deck, talks/20261117-hand_design/20261117-hand_design_talk_v2.html (contact repo), same build_talk.py.
Read first: docs/experiments/20261029-foundational_pose_review/20261109-hom_tracking_score.html (Section 7: the four
landscapes under J), scripts/hom/hom_track_landscapes.py, scripts/docs/codesign_section_figures.py,
scripts/docs/codesign_formulation_figures.py, scripts/hom/hom_grasp_refine.py (levels hom and both, --score track).
State 01:40: GPU queue 1 logs/20261111-hom_tuning_track/queue.sh (press, twist, roll x grasp, hom, both; 30 best layouts
by J; 32 x 8 x 3 rollouts per layout) running since 01:30, ~2.5 h; GPU queue 2 logs/20261111-talk_v2/gpu_queue_ea.sh
(size study under J, 31 EA runs) waits for queue 1, ~3.5 h.
State 02:40: done under J (files *_track in the formulation media; figures.json keys *_track): press and four-task
landscapes, covers (one hand (70, 110) at 0.62; K*(0.85) 3, K*(0.9) 4), EA panels, three-hands film, grasp candidates of
A (70, 65) (best = the foundational pose, 0.62), FP views of (105, 70). build_talk.py --v2 builds the v2 deck with
media_v2/ (contact f62daab + later edits); 5 texts read [pending] until the size study's slopes exist. Formulation
page Section 'v2'. Left: tuning chart (codesign_formulation_figures.py --track tuning, after queue 1), size trend
(codesign_section_figures.py --track, after queue 2), final v2 build with no [pending], commits, the owner's report.
Items (v2): film_three_hands under J (A = (70, 65), B = (70, 110), C = (125, 55)); press landscape and four-task
landscapes under J; EA anatomy on the J press landscape; covers K*(alpha) from the J grids; size trend and thesis slopes
from queue 2; grasp candidates of the new hand A with J scores; tuning chart from queue 1; slide text: how the 16 grasps
are chosen, grasp tuning beyond them (S2), HOM alone, both.
Claimed: scripts/docs/codesign_section_figures.py, codesign_formulation_figures.py, codesign_formulation_page.py,
scripts/hom/hom_grasp_refine.py, contact talks/20261117-hand_design/build_talk.py.
Do not: change the original deck's figures; git lfs push --all; GPU jobs outside the two queues.
