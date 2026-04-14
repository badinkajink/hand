# Milestones

## Phase 0 (Completed)

- uv-based project scaffold
- backend abstraction skeleton
- baseline hand and scene MJCF assets
- docs + smoke tests

## Phase 1 (Completed)

Inner-loop grasp synthesis and morphology evaluation pipeline.

- CEM-based foundational pose (FP) search with multi-seed support
- MJX-autodiff and DiffMJX-MVP alternative optimizers (tested, CEM preferred for now)
- Practical cube-grasp proxy metric: distance + contacts + lift + stability + per-finger persistence + drift penalties
- Optimization trace, plots, trajectory export, rollout GIF animation
- Pollard-style morphology sampling (500 candidates) across cube + 3 prism scenes
- Foundational pose adaptation strategies: interval refresh, sparse per-morph, local perturbation
- Top-k CEM refinement for final candidate ranking
- Pareto front analysis and feasibility gating

### Phase 1 Key Results

- Run 3: 420/500 cube, 293-412/500 prisms with strict 3-finger stability gates
- Run 5: FP adaptation recovers +22% feasibility (1442 -> 1800 total) for minimal extra cost
- Best FP strategy: `sparse-per-morph` (5 samples, +316 gain in +223s)
- CPU MuJoCo confirmed as the right backend for current evaluation loop

## Phase 1.5 (Current — Outer Loop Foundation)

Transition from single-morphology evaluation to systematic morphology optimization.

Planned:

- [ ] Multi-object evaluation: add more object shapes beyond cube + 3 prisms
- [ ] Outer loop optimizer: MAP-Elites or CMA-ES over 9D morphology space
- [ ] Behavioral descriptors for MAP-Elites archive (e.g. finger spread, contact pattern)
- [ ] Cross-object aggregated fitness: single morphology scored across full object distribution
- [ ] Expand morphology bounds beyond current ±0.012m perturbation range

## Phase 2 (Planned — Full Outer Loop)

- MAP-Elites archive with behavioral descriptors
- Integrate inner-loop best-response (FP adaptation) in outer evaluation
- Large-scale morphology search (1000s of candidates)
- Cross-object Pareto analysis for generalist hand designs

## Phase 3 (Deferred)

Originally planned for warp backend integration. **Decision: stay on CPU MuJoCo.**

- ~~Wire `mjwarp` and `comfree-warp`~~
- GPU backends only worthwhile with batched evaluator (100s-1000s resident on GPU)
- May revisit if outer loop requires throughput beyond CPU parallel workers

## Phase 4 (Deferred)

- DiffMJX smooth collision detection (fixes gradient kinks)
- Contacts From Distance (CFD) for zero-gradient-before-contact problem
- Only needed if gradient-based inner loop becomes the bottleneck

## Phase 5 (Optional)

- CTR-inspired projection filter hooks
- Force-closure proxy term in evaluator
