# Fingertip shape: the aside for the fingertip overview (2026-11-03, numbers of the rerun 2026-11-04)

Owner, 2026-11-03: "why did i pick the ellipsoid again vs the capsules, spheres? ... did we ever do this fingertip
geometry co-design ... across the ellipsoid, hemisphere, capsule shapes? would be nice to have it as an aside if anyone
asks". The overview (`docs/overviews/printed_fingertips.html`, text in `scripts/overview_fingertip_text.py`) is claimed
by hand-20261027-crosstalk_feed_poller, so the paragraphs below go in through its ledger: section `print`, after the
paragraph on 85A at 30 % gyroid. Its step 42 (11-04 11:09) added the version of 11-03, with the numbers of the screens on
the first CAD hand. The rerun (`docs/handoff/20261103-fingertip_shape_rerun.md`, page
`docs/experiments/20261104-fingertip_shape_rerun/20261104-fingertip_shapes_v4b_finger.html`) landed on 11-04, and the two
paragraphs below replace the two of 11-03 (the poller's step 51, added on 11-04 at 21:29). Sources: the rerun's
page and its `screen`, `final`, `band`, `cost`, `mount` and `fe` summaries; the 10-14, 10-15 and 10-18 pages linked in
the text; the owner's choice is row 13 of `docs/experiments/20261018-printed_fingertip_status/status.json`.

```html
<p>The pad&#8217;s shape is the 11/22 half-ellipsoid, the owner&#8217;s choice of 2026-10-18. A rerun on the v4b finger
at 2.5&#8202;MPa scored twelve tips: the v4b tip, two blocks, three capsules, two other half-ellipsoids, three
quarter-ellipsoids and a hemisphere
([fingertip shapes on the v4b finger](exp:20261104-fingertip_shape_rerun/20261104-fingertip_shapes_v4b_finger.html)). A
tip&#8217;s score is the held turn of the grasps on four cylinder cells and of HOM on D1&#8211;D8, as a fraction of
90&#176;. Four finalists were chosen by this score under five friction and servo conditions, by the agreement with Drake
and by the share of grasps with three loaded fingers. They were scored again on fresh placements (J3). The best finalist was the half-ellipsoid 7.4&#8202;&#215;&#8202;8.5&#8202;&#215;&#8202;16/32&#8202;mm.
Its J3 was 0.259, against 0.195 for the v4b tip. The 90&#8202;% interval of the paired difference runs from &#8722;0.017
to +0.160, and the other three finalists scored below the v4b tip. The HOM grasps, forces and contact modes refined for
the v4b tip over scene friction 0.4&#8211;1.0 scored lower on every finalist, by a median 0.02&#8211;0.90 per task.</p>
<p>Only the v4b tip and the half-ellipsoid 7.4&#8202;&#215;&#8202;5&#8202;&#215;&#8202;11/22&#8202;mm fit the DIP link,
cassette and cap as built. The other ten would need a cassette and cap of their own. A MuJoCo-Warp world-step at 2048
worlds lasts 2.52&#8202;&#181;s on the v4b tip and 3.98&#8211;4.24&#8202;&#181;s on the blocks. On the other curved tips
it lasts 2.06&#8211;3.23&#8202;&#181;s. In the FE, pad 05R4&#8217;s four magnets fit nine of the tips, and on each the
force at the contact point is resolved to 0.032&#8211;0.037&#8202;N. They do not fit the quarter-ellipsoids. Under a
12.5&#8202;mm rod along the finger the capsules sink 42&#8211;45&#8202;&#181;m at 1&#8202;N. Across the finger they sink
184&#8211;187&#8202;&#181;m. On 18 October no single spring fitted the capsule. One contact model reproduced the
half-ellipsoids, at 5.2&#8202;% RMS in Drake
([FE calibration](exp:20261018-fe_contact_calibration/20261018-fe_contact_calibration.html)). The screens of 14 and 15
October, on the first CAD hand at 10&#8202;MPa, compared the tips with the printed block. There the 11/22 scored
J3 = 0.223 against 0.170
([fingertip co-design](exp:20261014-tip_codesign/20261014-sphere_pad_accuracy_and_fingertip_codesign.html),
[curved tips](exp:20261015-curved_tips/20261015-capsule_and_ellipsoid_fingertips.html)). The 90&#8202;% interval of the
difference ran from &#8722;0.004 to +0.100.</p>
```
