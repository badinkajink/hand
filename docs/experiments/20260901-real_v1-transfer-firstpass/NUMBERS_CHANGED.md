# Transfer study: numbers that changed on 2026-09-01

Two corrections, both in `scripts/`, both regenerated into `paper/figures/` and
`paper/transfer_table.tex`. The `*_clean.pdf` figures dropped into `paper/figures/`
at 12:25 predate BOTH and disagree with the data; they need rebuilding from the
current `fig_transfer_{drops,rankflow,sim2real}.pdf`.

## 1. sv1_u1364 (D3) was scored from the wrong session

`bench()` kept each design's LAST session. u1364 ran twice: 12:36-12:45 (11 trials,
10 with an operator verdict, 4 held -- the operator's own "4/10" note) and
14:51-14:54 (7 trials, ZERO verdicts). The rule took the later one, so D3's
hold/drop labels came from the tag heuristic rather than from anyone watching.
The selector now takes the last session THE OPERATOR SCORED. No other design is
affected: every other last session is scored.

|              | before | after |
|--------------|--------|-------|
| D3 hold rate | 3/7 (heuristic) | 4/10 (operator) |
| D3 bench cos | 0.179 | 0.689 |
| D3 bench rank | 8 | 6 |
| D3 failure mode | eject | stall |
| trials in study | 74 | 77 |
| holds / drops | 47 / 26 | 48 / 28 |

## 2. The floor-contact exclusion is gone (earlier the same day)

Tag height is a rigid-body function of turn angle: -0.97 mm/deg, r = -0.888 over 48 holds, residual sd 5.8 mm.
Thresholding it removed hands in proportion to how far they turned. The operator's
verdict decides a hold; there is no floor rule and no dagger column. D4 is the only
substituted hand (peak alignment, marked `*`).

## Current table

| hand | sim cos | rank | hold | meas | bench cos | rank | turn h/d | slip h/d | mode |
|------|---------|------|------|------|-----------|------|----------|----------|------|
| D1 | 0.827 | 1 | 6/7 | 6 | 0.826 | 2 | 52 / -- | 5.0 / -- | -- |
| D2 | 0.726 | 2 | 10/10 | 9 | 0.797 | 4 | 48 / -- | 5.0 / -- | -- |
| D3 | 0.711 | 3 | 4/10 | 4 | 0.689 | 6 | 42 / 25 | 6.4 / 10.4 | stall |
| D4* | 0.627 | 4 | 9/10 | 9 | 0.819 | 3 | 53 / 54 | 7.1 / 8.4 | overshoot |
| D5 | 0.597 | 5 | 3/10 | 3 | 0.938 | 1 | 70 / 73 | 6.9 / 11.7 | overshoot |
| D6 | 0.585 | 6 | 2/10 | 2 | 0.614 | 7 | 35 / -1 | 4.9 / 37.6 | eject |
| D7 | 0.568 | 7 | 10/10 | 10 | 0.553 | 8 | 33 / -- | 10.9 / -- | -- |
| D8 | 0.501 | 8 | 4/10 | 4 | 0.773 | 5 | 44 / 65 | 5.9 / 8.2 | overshoot |

## Statements to update in the .tex

| claim | old | new |
|-------|-----|-----|
| trials | 74 | **77** |
| tag survives the run | 40 of 74 | **44 of 77** |
| sim2real, alignment | +0.33 (p = 0.42) | **+0.50 (p = 0.21)** |
| sim2real, substitution removed | +0.36 (n = 7) | **+0.54 (n = 7)** |
| sim2real, hold rate | +0.32 | **+0.27** |
| pooled peak-alignment, holds vs drops | drops +0.078 higher, p = 2e-4 | **NULL: +0.012, p = 0.60 -- delete this claim** |
| holds inside 20-60 deg | 40 of 47 | **44 of 48**, against **4 of 28** drops (Fisher OR 66, p = 8e-12) |
| failure modes | 2 (overshoot, eject) | **3 (overshoot, stall, eject)** |
| drops short of 20 deg / past 60 deg | -- | **12 / 12**, only 4 inside the band |
| max slip on a hold | 17 mm | **12.8 mm** |
| ejecting hands | D3 and D6 | **D6 only**; D3 stalls |
| simulated force -> measured alignment | +0.50 | **+0.64** |
| simulated contacts -> measured alignment | -0.33 | **-0.52** |

### The pooled overshoot claim must go

It was true when D3's drops were ejections read off the tag. With D3's real drops
(stalls, turning 25 deg against its holds' 42) the pooled effect cancels: drops are
bimodal in turn, not uniformly high. Replace it with the band contingency above and
the per-hand tests, which all survive: D8 44 vs 65 deg (p = 0.003), D3 42 vs 25
(p = 0.040), D5 70 vs 73 (p = 0.19, not separable).


## 3. sv1_w6689 (D1) gains three alignment samples, and loses precision

The 2026-09-01 04:18 re-run produced 15 trials of which the operator scored 3, all holds.
`EXTRA_RUNS` admits those three into D1's ALIGNMENT pool and nothing else: they were picked
because they were scored *and* succeeded, so putting them in a denominator would select on the
outcome being measured. D1's hold rate stays 6/7 from its own session; `meas` goes 6 -> 9.

| | before | after |
|---|---|---|
| D1 bench cos | 0.826 +- 0.032 (n = 6) | 0.834 +- 0.092 (n = 9) |
| D1 rank | 2 | 2 |

The standard deviation tripled. The three added trials read 0.655, 0.909 and 0.990 -- turns of
41, 55 and 72 deg from one plan on one hand -- against a spread of 0.032 across the whole of the
12:05 session. The extra samples make D1's number *less* certain, not more, and that is the
honest reading: the 04:18 staging was not repeatable.

Two defects in that session, both worth fixing before the next one:

* it was launched **without `--shaft-axis -x`**, so the axial offset went to the wrong end of
  the shaft and the tracker put the cylinder centre 72 mm under the bench. Height, x/y, slip and
  the drop verdict are void for all 15 runs; only the ANGLE survives, because it comes from the
  tag's own orientation and never sees the offset. The tracker flags this itself in
  `summary.notes` -- it is worth reading after the first run of a session.
* it was launched **without `--video-hz 4`**, so there are no frames and those runs can never
  appear in a filmstrip.

## 4. Why the 30 mm-tag trials are still out -- and it is not the tag

The 19 morning runs of 2026-08-31 used the 30 mm print on the old vane (axial 71.0 mm). They
look good: 14 scored holds, alignment 0.826-0.896, turns 52-58 deg. They are excluded, and the
usual reason given -- "different tag" -- is not the real one:

* the ANGLE never touches the axial offset. It is the tag's own orientation against the
  reference tag's up vector, so 71 mm and 77 mm are directly comparable.
* the up datum is stable to 2 deg across every session in the study, 30 mm and 40 mm alike.
* w6689's own step across the tag change (10:26 -> 12:05) moves its mean by 6.0 deg. Ordinary
  session-to-session drift on the SAME tag is +-5 deg (u0060 +4.0, u0308 +4.8, w0099 -4.5), and
  within the 30 mm era itself w6689 moved 14.5 deg between two consecutive sessions.

So the tag change is not the dominant term; re-staging is. The reason to leave them out is that
the study is a comparison ACROSS hands, and w6689 is the only hand with a 30 mm era. Pooling
would give one hand of eight a three-session average against everyone else's single session, in
a study whose between-session drift is +-5 deg -- it would shrink D1's variance alone and leave
the other seven where they are. That biases the comparison, not the number.

If the 30 mm runs are wanted, the defensible version is to give **every** hand its whole history
and report a between-session term, not to add the history of one hand.
