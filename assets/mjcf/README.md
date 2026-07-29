# `assets/mjcf/` layout

One directory per **hand topology**. The actuated hand (morph joints intact) sits at the top of
its directory; the object scenes that embed that hand live in its `scenes/`.

```
assets/mjcf/
  baseline/                          the original 3-finger hand: all fingers point +X
    hand.xml                         actuated hand only (morph x/y/len slides present)
    hand_morphology_actuated.xml     + actuators on the morph joints (for driving them)
    scenes/
      scene.xml                      cube
      scene_screwdriver_*.xml        screwdriver variants (flat / vertical / small / medium)
      scene_power_drill*.xml         power drill (two of these reference ../../../../external meshes)
      scene_prism*.xml, scene_cube_short_proximal.xml, scene_human_calf.xml
      scene_morphology_actuated.xml  pairs with ../hand_morphology_actuated.xml

  perp/                              opposed-pair hand: index/middle rotated 90 deg to FACE
    scenes/                          each other along Y, thumb still +X. docs/rl/perp_topology.md
      scene_screwdriver_medium_perp.xml

  generated/                         DERIVED, do not hand-edit — rigid scenes with the 9-param
                                     morphology baked into body transforms and the morph joints
                                     removed (generate_morphology_xml.py). Kept flat on purpose:
                                     filenames already carry the hand prefix and the morphology
                                     hash, and every consumer resolves them by glob.

  experimental/                      one-off contact/sim2real/landscape variants
```

## Notes

- **Scenes are self-contained**, not `<include>`-based: each embeds a full copy of its hand. So a
  hand geometry change must be applied to the hand XML *and* its scenes (or, better, regenerated).
  This duplication is why the per-hand grouping matters — it is otherwise invisible which scenes
  carry which hand.
- **`short_proximal` is a KEYFRAME variant, not a hand variant.** All scenes share the same 0.05 m
  proximal link; the difference is the open pose (`open_short_manual`). That is why there is no
  `short_proximal/` hand directory.
- **Relative paths inside scenes are depth-sensitive.** The two power-drill scenes reference
  `../../../../external/035_power_drill/...`; a scene moved to a different depth must have those
  rewritten (`morphology_xml.rebase_asset_file_paths` does this for generated output).
- **Two legacy scenes do not load** — `scene_power_drill_vertical.xml` and
  `scene_screwdriver_small_vertical.xml` both raise `keyframe 'open': invalid qpos size, expected
  31, got 32`. This predates the reorganisation (verified against the pre-move revision) and is
  left as-is; neither is on an active path.
