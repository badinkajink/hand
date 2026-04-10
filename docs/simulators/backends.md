# Backend Strategy

## Roles

- `mjx-native`: default autodiff backend for the inner loop.
- `diffmjx-lite`: planned MJX variant with smooth collision and CFD gradient path.
- `mjwarp`: high-throughput non-diff fallback backend.
- `comfree-warp`: high-throughput analytical-contact backend candidate.

## Recommended Usage

- Inner loop: `mjx-native` first, then compare with `diffmjx-lite`.
- Outer loop: `comfree-warp` if stable; otherwise `mjwarp` fallback.

## Failure Handling

When a morphology causes instability on `comfree-warp`:

1. mark evaluation as unstable,
2. rerun same morphology in `mjwarp`,
3. record backend used in experiment logs.
