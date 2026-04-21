"""Shared primitives for Phase 1 morphology sampling drivers.

Every sampling script in `scripts/` composes these pieces:

- morphology: dataclasses, bounds, clip/sample/encode/parse
- foundational: load prior FP search summaries
- feasibility: per-sample feasibility criteria and Pareto fronts
- scene: scene XML patching, settle/keyframe helpers (MuJoCo)
- adapt: foundational-pose adaptation strategies
- io: CSV writing, standard scatter plots, row helpers
"""

from .adapt import (
    AdaptationMode,
    adapt_foundational_ctrl,
    local_perturb_fp,
)
from .feasibility import (
    PARETO_MAX_KEYS,
    PARETO_MIN_KEYS,
    FeasibilityCriteria,
    is_feasible,
    pareto_front_indices,
)
from .foundational import (
    FoundationalPose,
    load_best_pose_with_prefix,
    load_foundational_poses,
    load_foundational_poses_with_scene,
)
from .io import (
    morph_row_fields,
    plot_feasible_scatter,
    write_csv,
)
from .morphology import (
    FINGER_ACTUATOR_NAMES,
    MorphologyBounds,
    clip_morphology,
    load_base_morphology,
    morph_distance,
    morph_suffix,
    morph_to_array,
    parse_morphology_from_generated_filename,
    parse_morphology_from_keyframe,
    read_keyframe_qpos,
    sample_morphologies,
)
from .scene import (
    add_foundational_keyframe,
    simulate_settle_qpos,
    write_rigid_scene_with_object_size,
    write_scene_with_morphology,
)

__all__ = [
    "AdaptationMode",
    "FINGER_ACTUATOR_NAMES",
    "FeasibilityCriteria",
    "FoundationalPose",
    "MorphologyBounds",
    "PARETO_MAX_KEYS",
    "PARETO_MIN_KEYS",
    "adapt_foundational_ctrl",
    "add_foundational_keyframe",
    "clip_morphology",
    "is_feasible",
    "load_base_morphology",
    "load_best_pose_with_prefix",
    "load_foundational_poses",
    "load_foundational_poses_with_scene",
    "local_perturb_fp",
    "morph_distance",
    "morph_row_fields",
    "morph_suffix",
    "morph_to_array",
    "pareto_front_indices",
    "parse_morphology_from_generated_filename",
    "parse_morphology_from_keyframe",
    "plot_feasible_scatter",
    "read_keyframe_qpos",
    "sample_morphologies",
    "simulate_settle_qpos",
    "write_csv",
    "write_rigid_scene_with_object_size",
    "write_scene_with_morphology",
]
