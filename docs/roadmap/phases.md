# Milestones

## Phase 0 (Completed)

- uv-based project scaffold
- backend abstraction skeleton
- baseline hand and scene MJCF assets
- docs + smoke tests

## Phase 1

- implement real inner-loop grasp synthesis runner on generated scenes
- define practical cube-grasp proxy metric (distance + contacts + lift + stability)
- add optimization trace, plots, trajectory export, and rollout animation
- keep MJX-native autodiff wiring as next Phase 1 extension

## Phase 2

- add MAP-Elites archive and descriptors
- integrate inner-loop best response in outer evaluation

## Phase 3

- wire `mjwarp` and `comfree-warp`
- backend fallback and robust logging

## Phase 4

- implement `diffmjx-lite` features
- compare inner-loop performance vs native MJX

## Phase 5 (Optional)

- add CTR-inspired projection filter hooks
- evaluate if projected steps improve reliability
