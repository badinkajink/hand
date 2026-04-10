# DiffMJX Plan

## Implement First

1. Smooth collision branch blending in collision routines.
2. Contacts-from-distance (CFD) backward-pass rerouting.

## Defer

- Adaptive integration until baseline grasp optimization is stable.

## Why

For quasi-static grasp synthesis, smooth collision + CFD addresses the main gradient pathologies:

- zero gradient before contact,
- discontinuous branch gradients at collision mode switches.

## Validation

- finite-difference checks for selected states,
- convergence comparison against `mjx-native`,
- ablation for CFD on/off.
