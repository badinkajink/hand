Status: active (window 11-10 00:xx, then autopilot step 15, first by rule)
Task: owner 2026-11-10 with the deck PDF: "continue building beautiful, clear, diagrams and visual content that is communicable
for my audience"; "generate/augment docpages as needed, and update the main slide deck as well afterward"; HOM diagrams
"ideally even simpler/clearer"; "the structure of the talk as ive attached it can be made clear"; SR2 part "hierarchical,
rather than cyclical". Then: "this is pretty important ... finish up, double check outputs on this task".
Read first: the owner's deck ~/wxie_workspace/contact/talks/20261117-hand_design/20261110-owner_talk_11-17_Hand_Design.pdf
(44 slides: 1-11 human dexterity, 12-19 robot dexterity, 20-31 SR2 co-design, 32-33 morphology optimization, 34-35
multi-hand control, 36-38 sim-to-real, 39 questions, 40-44 backup); the HTML deck build_talk.py there (its slide
"question" has m*_T(d) = argmax_m J_T(m, pi(m), d)); pages 20261109-hom_search_methods.html (EA, PSO), 20261103-hom_score_refine
(S1/S3 levels, CEM), 20261109-hom_tracking_score.html (the score J), the HOM explainer pages (20261029-hom_control_review),
contact_model and fingertip overviews; scripts/hom/hom_ea.py (EA: pop, elitist half, top eighth parents, U(-delta/2, delta/2),
delta 30 mm, 5 mm lattice).
Deliverables (owner's slide numbers):
1. Slides 21-31: one co-design formula that grows a term per slide (KaTeX in the deck, texsvg on the page): a hand for a
   task -> body and controller (bi-level) -> object sizes d -> m in R^6, control (grasp g, HOM parameters phi in R^9) ->
   task distribution -> metric J (tracking score; vector of metrics, scalarized or Pareto) -> J from simulation with contact
   model c, plant p, sensing s. Final form, kept for the rest of the talk, with underbraces naming the four SR2 sections:
   argmax_m (morphology optimization) E_{T,d} [ max_{g,phi} (multi-hand control) J^sim_{c,p,s}(m,g,phi;T,d) (sim-to-real) ].
2. Slides 32-33 (outer loop): Yang's EA from first principles (fitness = seed mean of the inner value; elitist selection;
   parents; mutation on the lattice) and a figure of EA population snapshots on a stored 2-D landscape, with the
   best-so-far curve against random and Sobol (media/20261109-search_compare.png exists).
3. Slides 34-35 (inner loop): max_g max_phi decomposition: grasp synthesis (the toy's most-interior SQP pose + variants, 16
   grasps), HOM's Eq. 2 as a fixed law in which the hand enters only through A_i(m, g), b_i, and HOM tuning phi (forces,
   contact-mode weights) by CEM (S1 vs S3). A simpler HOM block diagram.
4. Slides 36-38 (sim-to-real): J_real ~ J_sim_{c,p,s}; the evaluator must be fast (rollout budget arithmetic -> GPU pads),
   calibrated (pad modulus 2.5 MPa from UR5 presses, servo 0.72/0.55 N m/rad from sysid) and observable (magnetic tips
   measure contact location and force); sensing depends on the layout (crosstalk study on 11 layouts) -> sensors as design
   variables is the open extension.
5. One bi-level diagram reused per section with the active part highlighted (4 PNGs for the owner's slides).
6. A page docs/experiments/20261110-codesign_formulation/ with 1-5 and its figures in media/; then the HTML deck restructured
   to the owner's outline with these slides. Build both, look at every figure and slide, check KaTeX renders.
State: 2026-11-10 ~00:50: spec written; owner's PDF copied (untracked). Nothing built yet.
Claimed: docs/experiments/20261110-codesign_formulation/, scripts/docs/codesign_formulation_*.py, the talk's build_talk.py.
Do not: push the contact repo without the owner's word; commit the owner's PDF.
