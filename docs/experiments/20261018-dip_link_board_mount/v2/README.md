# DIP link v2: print order

Printable v2 of concepts a and b (owner, 2026-10-18 18:45: five 3 x 2 mm magnets, the 11/22 half-ellipsoid, a longer
finger if J1 and its cable need it, M2 into 1.8 mm self-forming holes). Each project below is a sliced PrusaSlicer 3mf for
the Prusa XL 5T (PLA on tool 3 at 0.1 mm layers; Siraya 85A on tool 2 at 0.2 mm layers, 30 % gyroid, M601 pauses at the
2.9 mm magnet seats); its JSON beside it records the settings and the slice checks. The numbers are in
`20261018-v2_summary.json` (`scripts/dip_link_v2.py summary`).

Print first, in this order:

1. `a/print/20261018-dip_link_v2a_fit_coupon_keys_pla_t3.3mf` (15 min) and
   `a/print/20261018-dip_link_v2a_fit_coupon_channels_85a_t2.3mf` (6 min): choose a's channel interference
   (0, 0.05 or 0.10 mm per flank).
2. `b/print/20261018-dip_link_v2b_fit_coupon_rails_pla_t3.3mf` (6 min) and
   `b/print/20261018-dip_link_v2b_fit_coupon_grooves_85a_t2.3mf` (8 min): choose b's groove interference.
3. `a/print/20261018-dip_link_v2a_print_link_clamp_coupon_pla_t3.3mf` (1 h 52 min, 11.45 g) and
   `a/print/20261018-dip_link_v2a_print_tip_ell11_5_channels_85a_t2_g30.3mf` (34 min; pauses 7.3, 12.2, 17.1 mm).
4. `b/print/20261018-dip_link_v2b_print_link_cover_pla_t3.3mf` (2 h 6 min, 11.36 g) and
   `b/print/20261018-dip_link_v2b_print_pad_ell11_5_85a_t2_g30.3mf` (33 min; pauses 7.3, 12.2, 17.1 mm).

The five-magnet block versions are `a/print/*_tip_block5_channels_85a_t2_g30.3mf` (pause 12.4 mm) and
`b/print/*_pad_block5_85a_t2_g30.3mf` (printed on its cut back face, pause 8.8 mm). At each pause press the magnets to the
pocket floor, north pole toward the palmar face, and resume.

Hardware: a takes two M2 x 4 countersunk and two M2 x 5 cheese-head screws, b four M2 x 5 cheese-head screws, all brass
(R10: no ferromagnetic fastener within 15 mm of a chip); the board stays as supplied (J1 kept, tab not cut) and a Qwiic
cable plugs into J1. The assembly sequences and parts lists are in `a/evaluation.json` and `b/evaluation.json`
(`assembly`, `parts_list`). Both fingers are longer: PIP axis to the tip's end 45.66 mm (a) and 46.91 mm (b), 37.16 mm
today.

Regenerate from the parameter files (`../params/20261018-v2_concept_a.json`, `../params/20261018-v2_concept_b.json`):

    RG="$HOME/.claude/bin/resguard.sh run --mem 3G --cpu 400 -- taskset -c 8-11 env NUMPY_MADVISE_HUGEPAGE=0"
    E=~/miniconda3/envs/femtip/bin/python
    $RG $E scripts/dip_link_v2.py a build step check sense render print docs
    $RG $E scripts/dip_link_v2.py b build step check sense render print docs
    $E scripts/dip_link_v2.py summary
