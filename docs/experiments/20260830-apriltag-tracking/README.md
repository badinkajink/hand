# AprilTag tracking — the bring-up

`gen_tags.py` writes the printable sheet (`apriltags_letter.pdf`): **id 6 at 40 mm** for the
static reference, **id 0 at 30 mm** for the cylinder's vane. `sweep_exp.py` is the
exposure/gain sweep that picked 4000 us / gain 64 — auto-exposure had settled on 8500/16, where
the reference tag stops decoding, and the smear column is what ruled out the longer exposures.
`track_tags.py` is the original one-file probe, kept as it was written.

**The maintained tool is `scripts/real_v1_tag_tracker.py`**, and the geometry it uses lives in
`morphohand.bench.tags` where it can be unit-tested without a camera. Use this directory for
re-running the bring-up (`--probe`, the exposure sweep, reprinting the sheet); use the script in
`scripts/` for anything whose numbers you intend to keep. What the maintained version adds, and
why, is in
[`docs/experiments/20260831-real_v1-object-tracking/`](../20260831-real_v1-object-tracking/README.md).
