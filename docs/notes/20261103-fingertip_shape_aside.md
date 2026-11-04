# Fingertip shape: the aside for the fingertip overview (2026-11-03)

Owner, 2026-11-03: "why did i pick the ellipsoid again vs the capsules, spheres? ... did we ever do this fingertip
geometry co-design ... across the ellipsoid, hemisphere, capsule shapes? would be nice to have it as an aside if anyone
asks". The overview (`docs/overviews/printed_fingertips.html`, text in `scripts/overview_fingertip_text.py`) is claimed
by hand-20261027-crosstalk_feed_poller, so the paragraphs below go in through its ledger: section `print`, after the
paragraph on 85A at 30 % gyroid. The rerun (`docs/handoff/20261103-fingertip_shape_rerun.md`) replaces the numbers when
it lands. Sources: the 10-14, 10-15 and 10-18 pages linked in the text; the owner's choice is row 13 of
`docs/experiments/20261018-printed_fingertip_status/status.json`.

```html
<p>The pad&#8217;s shape is the 11/22 half-ellipsoid, the owner&#8217;s choice of 2026-10-18 after two screens on the
SR2 hand and a calibration of the contact models to the FE. The screens scored 24 rounded blocks and nine curved tips:
three capsules, three half-ellipsoids and three quarter-ellipsoids. A tip&#8217;s score is the held turn of the grasps
on four cylinder cells and of HOM on D1&#8211;D8, as a fraction of 90&#8202;&#176;. Its worst over five friction and
servo conditions chose the finalists, which were scored again on fresh placements (J3). No tip beat the printed block:
the 11/22 half-ellipsoid scored J3 = 0.223 against 0.170, and the 90&#8202;% interval of the difference runs from
&#8722;0.004 to +0.100 ([fingertip co-design](exp:20261014-tip_codesign/20261014-sphere_pad_accuracy_and_fingertip_codesign.html),
[curved tips](exp:20261015-curved_tips/20261015-capsule_and_ellipsoid_fingertips.html)). Its GPU world-step took
1.72&#8202;&#181;s against the block&#8217;s 2.93, and its GPU memory was the least of the capsules and half-ellipsoids,
5.5 against 11.1&#8202;GB. One contact model fitted to the FE reproduced the half-ellipsoids, at 5.2&#8202;% RMS in Drake
and 5.6&#8202;% with the pads on the 11/22. No single spring fitted the capsule, which met 4.1 times the stiffness from a
rod along the finger as from one across it
([FE calibration](exp:20261018-fe_contact_calibration/20261018-fe_contact_calibration.html)). The quarter-ellipsoids
failed the screens&#8217; agreement gate with Drake, and their nearest chip saturates at 214&#8202;mT.</p>
<p>A hemisphere was tested only on the contact bed, and the force resolution only on the block and the 11/22. The
screens ran on the first CAD hand, before the pads&#8217; modulus moved to 2.5&#8202;MPa and before the measured servo
plant. A rerun on the current hand, with a hemisphere and a resolution for every shape, is queued after the HOM
studies.</p>
```
