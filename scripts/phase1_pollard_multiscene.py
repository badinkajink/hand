from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from morphohand.optimization.phase1_grasp import (  # noqa: E402
    Phase1OptimizationConfig,
    Phase1EvalConfig,
    Phase1GraspEvaluator,
    optimize_finger_controls,
)
from morphohand.tools.morphology_xml import (  # noqa: E402
    MorphologyValues,
    build_morphology_suffix,
    create_rigid_morphology_xml,
    extract_morphology_from_qpos,
)


FINGER_ACTUATOR_NAMES = [
    "a_thumb_yaw",
    "a_thumb_mcp",
    "a_thumb_pip",
    "a_index_yaw",
    "a_index_mcp",
    "a_index_pip",
    "a_middle_yaw",
    "a_middle_mcp",
    "a_middle_pip",
]


@dataclass(frozen=True)
class FoundationalPose:
    label: str
    finger_ctrl: np.ndarray
    score: float


@dataclass(frozen=True)
class SceneSpec:
    scene_key: str
    display_name: str
    size_x: float
    size_y: float
    size_z: float
    foundational_poses: list[FoundationalPose]
    max_mean_tip_distance: float
    min_contacts: float
    min_finger_contact_persistence: float
    max_finger_yaw_drift: float


@dataclass(frozen=True)
class MorphologyBounds:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    len_min: float
    len_max: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Large-scale Pollard-style morphology sampling across cube + prism scenes with "
            "foundational-pose feasibility checks, keyframe export, and top-k GIF rendering."
        )
    )
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--backend",
        choices=["mujoco", "mjwarp", "comfree-warp"],
        default="mujoco",
        help="Physics backend used by evaluator rollouts.",
    )
    parser.add_argument("--comfree-stiffness", type=float, default=0.2)
    parser.add_argument("--comfree-damping", type=float, default=0.001)
    parser.add_argument("--backend-nworld", type=int, default=1)
    parser.add_argument("--backend-nconmax", type=int, default=200)
    parser.add_argument("--backend-njmax", type=int, default=2000)
    parser.add_argument("--x-perturb", type=float, default=0.012)
    parser.add_argument("--y-perturb", type=float, default=0.012)
    parser.add_argument("--len-perturb", type=float, default=0.012)
    parser.add_argument("--x-min", type=float, default=-0.03)
    parser.add_argument("--x-max", type=float, default=0.03)
    parser.add_argument("--y-min", type=float, default=-0.03)
    parser.add_argument("--y-max", type=float, default=0.03)
    parser.add_argument("--len-min", type=float, default=0.0)
    parser.add_argument("--len-max", type=float, default=0.035)
    parser.add_argument("--cube-max-mean-tip-distance", type=float, default=0.012)
    parser.add_argument("--cube-min-contacts", type=float, default=2.0)
    parser.add_argument("--cube-min-finger-contact-persistence", type=float, default=0.55)
    parser.add_argument("--cube-max-finger-yaw-drift", type=float, default=0.30)
    parser.add_argument("--prism-max-mean-tip-distance", type=float, default=0.03)
    parser.add_argument("--prism-min-contacts", type=float, default=1.0)
    parser.add_argument("--prism-min-finger-contact-persistence", type=float, default=0.45)
    parser.add_argument("--prism-max-finger-yaw-drift", type=float, default=0.40)
    parser.add_argument("--settle-steps", type=int, default=240)
    parser.add_argument("--lift-steps", type=int, default=220)
    parser.add_argument("--hold-steps", type=int, default=140)
    parser.add_argument("--lift-delta-z", type=float, default=0.05)
    parser.add_argument("--lift-ramp-steps", type=int, default=100)
    parser.add_argument("--objective-weight-min-finger-persistence", type=float, default=2.4)
    parser.add_argument(
        "--objective-weight-finger-persistence-imbalance-penalty",
        type=float,
        default=1.2,
    )
    parser.add_argument("--objective-weight-finger-yaw-drift-penalty", type=float, default=1.0)
    parser.add_argument("--objective-weight-finger-flex-drift-penalty", type=float, default=0.5)
    parser.add_argument("--top-k-gifs", type=int, default=5)
    parser.add_argument("--refine-top-k", action="store_true")
    parser.add_argument("--refine-pool-size", type=int, default=15)
    parser.add_argument("--refine-iterations", type=int, default=8)
    parser.add_argument("--refine-population", type=int, default=24)
    parser.add_argument("--refine-elite-fraction", type=float, default=0.25)
    parser.add_argument("--refine-sigma-init", type=float, default=0.08)
    parser.add_argument("--refine-seed", type=int, default=0)
    parser.add_argument(
        "--fp-adaptation",
        choices=["none", "interval-open", "interval-initial-fp", "sparse-per-morph"],
        default="none",
        help="Adapt foundational pose strategy while traversing morphology samples.",
    )
    parser.add_argument(
        "--fp-refresh-interval",
        type=int,
        default=40,
        help="For interval modes, refresh foundational pose every N morphologies.",
    )
    parser.add_argument("--fp-adapt-iterations", type=int, default=12)
    parser.add_argument("--fp-adapt-population", type=int, default=24)
    parser.add_argument("--fp-adapt-elite-fraction", type=float, default=0.25)
    parser.add_argument("--fp-adapt-sigma-init", type=float, default=0.08)
    parser.add_argument("--fp-adapt-seed", type=int, default=0)
    parser.add_argument(
        "--base-scene-xml",
        type=Path,
        default=PROJECT_ROOT / "assets" / "mjcf" / "scene.xml",
    )
    parser.add_argument(
        "--cube-foundational-run-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "phase1" / "run_20260410_163959",
    )
    parser.add_argument(
        "--prism-foundational-run-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "phase1" / "run_20260413_prism_y_sweep_mjx_autodiff",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "phase1",
    )
    parser.add_argument("--tag", default=None)
    return parser


def _decode_token(token: str) -> float:
    if token.startswith("p"):
        sign = 1.0
        body = token[1:]
    elif token.startswith("n"):
        sign = -1.0
        body = token[1:]
    else:
        raise ValueError(f"Invalid morphology token sign: {token}")
    return sign * float(body.replace("d", ".", 1))


def _parse_morphology_from_generated_filename(scene_xml: Path) -> MorphologyValues:
    parts = scene_xml.stem.split("_")
    t_part = next((p for p in parts if p.startswith("t")), None)
    i_part = next((p for p in parts if p.startswith("i")), None)
    m_part = next((p for p in parts if p.startswith("m")), None)
    if t_part is None or i_part is None or m_part is None:
        raise ValueError(f"Scene filename missing t/i/m morphology groups: {scene_xml.name}")

    def split_triplet(group: str) -> tuple[float, float, float]:
        payload = group[1:]
        chunks = [payload[k : k + 7] for k in range(0, 21, 7)]
        return (_decode_token(chunks[0]), _decode_token(chunks[1]), _decode_token(chunks[2]))

    tx, ty, tl = split_triplet(t_part)
    ix, iy, il = split_triplet(i_part)
    mx, my, ml = split_triplet(m_part)
    return MorphologyValues(tx, ty, tl, ix, iy, il, mx, my, ml)


def _parse_morphology_from_open_keyframe(scene_xml: Path) -> MorphologyValues:
    root = ET.parse(scene_xml).getroot()
    keyframe = root.find("keyframe")
    if keyframe is None:
        raise ValueError(f"No keyframe section in {scene_xml}")

    open_key = None
    for key in keyframe.findall("key"):
        if key.get("name") == "open":
            open_key = key
            break
    if open_key is None:
        raise ValueError(f"No open keyframe in {scene_xml}")

    qpos_raw = open_key.get("qpos") or ""
    qpos = [float(v) for v in qpos_raw.replace("\n", " ").split()]
    return extract_morphology_from_qpos(qpos=qpos, has_scene_prefix=True)


def _load_base_morphology(scene_xml: Path) -> MorphologyValues:
    try:
        return _parse_morphology_from_generated_filename(scene_xml)
    except Exception:
        return _parse_morphology_from_open_keyframe(scene_xml)


def _load_cube_foundational_poses(run_dir: Path) -> tuple[list[FoundationalPose], MorphologyValues]:
    poses: list[FoundationalPose] = []
    scene_path: Path | None = None
    for subdir in sorted(run_dir.iterdir()):
        if not subdir.is_dir():
            continue
        summary_path = subdir / "summary.json"
        if not summary_path.exists():
            continue
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        poses.append(
            FoundationalPose(
                label=subdir.name,
                finger_ctrl=np.asarray(payload["best_finger_ctrl"], dtype=np.float64),
                score=float(payload.get("best_score", float("-inf"))),
            )
        )
        scene_path = Path(payload["scene_xml"])

    if not poses or scene_path is None:
        raise ValueError(f"No cube foundational summaries found in {run_dir}")

    poses = sorted(poses, key=lambda p: p.score, reverse=True)
    base_morphology = _load_base_morphology(scene_path)
    return poses, base_morphology


def _load_prism_pose(summary_path: Path, label: str) -> FoundationalPose:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return FoundationalPose(
        label=label,
        finger_ctrl=np.asarray(payload["best_finger_ctrl"], dtype=np.float64),
        score=float(payload.get("best_score", float("-inf"))),
    )


def _load_best_prism_pose(prism_run_dir: Path, prefix: str, label: str) -> FoundationalPose:
    candidates: list[tuple[float, Path]] = []
    for subdir in sorted(prism_run_dir.iterdir()):
        if not subdir.is_dir() or not subdir.name.startswith(prefix):
            continue
        summary_path = subdir / "summary.json"
        if not summary_path.exists():
            continue
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        candidates.append((float(payload.get("best_score", float("-inf"))), summary_path))

    if not candidates:
        raise ValueError(f"No prism foundational runs matching prefix '{prefix}' in {prism_run_dir}")

    _, best_summary = max(candidates, key=lambda item: item[0])
    return _load_prism_pose(best_summary, label)


def _load_scene_specs(
    cube_run_dir: Path,
    prism_run_dir: Path,
    cube_max_mean_tip_distance: float,
    cube_min_contacts: float,
    cube_min_finger_contact_persistence: float,
    cube_max_finger_yaw_drift: float,
    prism_max_mean_tip_distance: float,
    prism_min_contacts: float,
    prism_min_finger_contact_persistence: float,
    prism_max_finger_yaw_drift: float,
) -> tuple[list[SceneSpec], MorphologyValues]:
    cube_poses, base_morphology = _load_cube_foundational_poses(cube_run_dir)
    prism25 = _load_best_prism_pose(prism_run_dir, prefix="y0d0250", label="prism_y0d0250")
    prism30 = _load_best_prism_pose(prism_run_dir, prefix="y0d0300", label="prism_y0d0300")
    prism35 = _load_best_prism_pose(prism_run_dir, prefix="y0d0350", label="prism_y0d0350")

    specs = [
        SceneSpec(
            scene_key="cube",
            display_name="cube",
            size_x=0.02,
            size_y=0.02,
            size_z=0.02,
            foundational_poses=cube_poses,
            max_mean_tip_distance=cube_max_mean_tip_distance,
            min_contacts=cube_min_contacts,
            min_finger_contact_persistence=cube_min_finger_contact_persistence,
            max_finger_yaw_drift=cube_max_finger_yaw_drift,
        ),
        SceneSpec(
            scene_key="prism1",
            display_name="prism_y0d0250",
            size_x=0.02,
            size_y=0.025,
            size_z=0.02,
            foundational_poses=[prism25],
            max_mean_tip_distance=prism_max_mean_tip_distance,
            min_contacts=prism_min_contacts,
            min_finger_contact_persistence=prism_min_finger_contact_persistence,
            max_finger_yaw_drift=prism_max_finger_yaw_drift,
        ),
        SceneSpec(
            scene_key="prism2",
            display_name="prism_y0d0300",
            size_x=0.02,
            size_y=0.03,
            size_z=0.02,
            foundational_poses=[prism30],
            max_mean_tip_distance=prism_max_mean_tip_distance,
            min_contacts=prism_min_contacts,
            min_finger_contact_persistence=prism_min_finger_contact_persistence,
            max_finger_yaw_drift=prism_max_finger_yaw_drift,
        ),
        SceneSpec(
            scene_key="prism3",
            display_name="prism_y0d0350",
            size_x=0.02,
            size_y=0.035,
            size_z=0.02,
            foundational_poses=[prism35],
            max_mean_tip_distance=prism_max_mean_tip_distance,
            min_contacts=prism_min_contacts,
            min_finger_contact_persistence=prism_min_finger_contact_persistence,
            max_finger_yaw_drift=prism_max_finger_yaw_drift,
        ),
    ]
    return specs, base_morphology


def _clip_morphology(values: MorphologyValues, bounds: MorphologyBounds) -> MorphologyValues:
    return MorphologyValues(
        thumb_x=float(np.clip(values.thumb_x, bounds.x_min, bounds.x_max)),
        thumb_y=float(np.clip(values.thumb_y, bounds.y_min, bounds.y_max)),
        thumb_len=float(np.clip(values.thumb_len, bounds.len_min, bounds.len_max)),
        index_x=float(np.clip(values.index_x, bounds.x_min, bounds.x_max)),
        index_y=float(np.clip(values.index_y, bounds.y_min, bounds.y_max)),
        index_len=float(np.clip(values.index_len, bounds.len_min, bounds.len_max)),
        middle_x=float(np.clip(values.middle_x, bounds.x_min, bounds.x_max)),
        middle_y=float(np.clip(values.middle_y, bounds.y_min, bounds.y_max)),
        middle_len=float(np.clip(values.middle_len, bounds.len_min, bounds.len_max)),
    )


def _sample_morphologies(
    base: MorphologyValues,
    sample_count: int,
    rng: np.random.Generator,
    bounds: MorphologyBounds,
    x_perturb: float,
    y_perturb: float,
    len_perturb: float,
) -> list[MorphologyValues]:
    candidates: list[MorphologyValues] = [base]
    seen: set[tuple[float, ...]] = set()

    def key_for(v: MorphologyValues) -> tuple[float, ...]:
        return (
            round(v.thumb_x, 6),
            round(v.thumb_y, 6),
            round(v.thumb_len, 6),
            round(v.index_x, 6),
            round(v.index_y, 6),
            round(v.index_len, 6),
            round(v.middle_x, 6),
            round(v.middle_y, 6),
            round(v.middle_len, 6),
        )

    seen.add(key_for(base))
    while len(candidates) < sample_count:
        proposal = MorphologyValues(
            thumb_x=base.thumb_x + float(rng.uniform(-x_perturb, x_perturb)),
            thumb_y=base.thumb_y + float(rng.uniform(-y_perturb, y_perturb)),
            thumb_len=base.thumb_len + float(rng.uniform(-len_perturb, len_perturb)),
            index_x=base.index_x + float(rng.uniform(-x_perturb, x_perturb)),
            index_y=base.index_y + float(rng.uniform(-y_perturb, y_perturb)),
            index_len=base.index_len + float(rng.uniform(-len_perturb, len_perturb)),
            middle_x=base.middle_x + float(rng.uniform(-x_perturb, x_perturb)),
            middle_y=base.middle_y + float(rng.uniform(-y_perturb, y_perturb)),
            middle_len=base.middle_len + float(rng.uniform(-len_perturb, len_perturb)),
        )
        proposal = _clip_morphology(proposal, bounds)
        k = key_for(proposal)
        if k in seen:
            continue
        seen.add(k)
        candidates.append(proposal)

    return candidates


def _make_scene_for_spec(
    base_scene_xml: Path,
    scene_output_path: Path,
    morphology: MorphologyValues,
    spec: SceneSpec,
) -> Path:
    create_rigid_morphology_xml(
        base_xml_path=base_scene_xml,
        morphology=morphology,
        output_xml_path=scene_output_path,
        model_name=scene_output_path.stem,
    )

    root = ET.parse(scene_output_path).getroot()
    cube_body = None
    for body in root.iter("body"):
        if body.get("name") == "cube":
            cube_body = body
            break
    if cube_body is None:
        raise ValueError(f"Could not find cube body in {scene_output_path}")

    cube_geom = None
    for geom in cube_body.findall("geom"):
        if geom.get("type") == "box":
            cube_geom = geom
            break
    if cube_geom is None:
        raise ValueError(f"Could not find cube box geom in {scene_output_path}")

    cube_geom.set("size", f"{spec.size_x:.6f} {spec.size_y:.6f} {spec.size_z:.6f}")
    cube_body.set("pos", f"0.000000 0.000000 {spec.size_z:.6f}")

    ET.indent(root, space="  ")
    ET.ElementTree(root).write(scene_output_path, encoding="utf-8", xml_declaration=False)
    return scene_output_path


def _is_feasible(
    metrics: dict[str, float],
    max_mean_tip_distance: float,
    min_contacts: float,
    min_finger_contact_persistence: float,
    max_finger_yaw_drift: float,
) -> bool:
    return (
        float(metrics.get("mean_tip_distance", np.inf)) <= max_mean_tip_distance
        and float(metrics.get("cube_tip_contacts", 0.0)) >= min_contacts
        and float(metrics.get("min_finger_contact_persistence", 0.0)) >= min_finger_contact_persistence
        and float(metrics.get("finger_yaw_drift", np.inf)) <= max_finger_yaw_drift
    )


def _add_foundational_keyframe(scene_xml: Path, key_name: str, qpos: np.ndarray, ctrl: np.ndarray) -> None:
    root = ET.parse(scene_xml).getroot()
    keyframe = root.find("keyframe")
    if keyframe is None:
        keyframe = ET.SubElement(root, "keyframe")

    existing = None
    for key in keyframe.findall("key"):
        if key.get("name") == key_name:
            existing = key
            break

    qpos_text = "\n        " + " ".join(f"{v:.10g}" for v in qpos.tolist()) + "\n      "
    ctrl_text = "\n        " + " ".join(f"{v:.10g}" for v in ctrl.tolist()) + "\n      "

    if existing is None:
        ET.SubElement(keyframe, "key", name=key_name, qpos=qpos_text, ctrl=ctrl_text)
    else:
        existing.set("qpos", qpos_text)
        existing.set("ctrl", ctrl_text)

    ET.indent(root, space="  ")
    ET.ElementTree(root).write(scene_xml, encoding="utf-8", xml_declaration=False)


def _simulate_settle_qpos(scene_xml: Path, keyframe_name: str, finger_ctrl: np.ndarray, settle_steps: int) -> tuple[np.ndarray, np.ndarray]:
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe_name)
    if key_id < 0:
        raise ValueError(f"Keyframe '{keyframe_name}' not found in {scene_xml}")

    actuator_ids = []
    for name in FINGER_ACTUATOR_NAMES:
        idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if idx < 0:
            raise ValueError(f"Actuator '{name}' not found in {scene_xml}")
        actuator_ids.append(int(idx))

    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    ctrl = data.ctrl.copy()
    ctrl[np.asarray(actuator_ids, dtype=np.int32)] = finger_ctrl

    for _ in range(settle_steps):
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)

    return data.qpos.copy(), ctrl.copy()


def _write_csv(rows: list[dict[str, float | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _pareto_front_indices(rows: list[dict[str, float | str]]) -> list[int]:
    if not rows:
        return []

    max_keys = ("score", "cube_lift", "cube_tip_contacts")
    min_keys = ("mean_tip_distance", "cube_vel_norm")
    dominated = [False] * len(rows)

    for i, a in enumerate(rows):
        if dominated[i]:
            continue
        for j, b in enumerate(rows):
            if i == j or dominated[i]:
                continue
            b_not_worse = all(float(b[k]) >= float(a[k]) for k in max_keys) and all(
                float(b[k]) <= float(a[k]) for k in min_keys
            )
            b_strict = any(float(b[k]) > float(a[k]) for k in max_keys) or any(
                float(b[k]) < float(a[k]) for k in min_keys
            )
            if b_not_worse and b_strict:
                dominated[i] = True

    return [i for i, is_dominated in enumerate(dominated) if not is_dominated]


def _plot_scene_results(feasible_rows: list[dict[str, float | str]], pareto_indices: list[int], out_dir: Path) -> None:
    if not feasible_rows:
        return

    lift = np.asarray([float(r["cube_lift"]) for r in feasible_rows], dtype=np.float64)
    dist = np.asarray([float(r["mean_tip_distance"]) for r in feasible_rows], dtype=np.float64)
    contacts = np.asarray([float(r["cube_tip_contacts"]) for r in feasible_rows], dtype=np.float64)
    score = np.asarray([float(r["score"]) for r in feasible_rows], dtype=np.float64)

    plt.figure(figsize=(8, 4.8))
    scatter = plt.scatter(dist, lift, c=score, s=24, alpha=0.85, cmap="viridis")
    if pareto_indices:
        plt.scatter(
            dist[pareto_indices],
            lift[pareto_indices],
            facecolors="none",
            edgecolors="red",
            s=75,
            linewidths=1.2,
            label="Pareto front",
        )
        plt.legend()
    plt.xlabel("Mean tip distance")
    plt.ylabel("Cube/prism lift")
    plt.title("Feasible Morphologies")
    plt.grid(True, alpha=0.25)
    cbar = plt.colorbar(scatter)
    cbar.set_label("Score")
    plt.tight_layout()
    plt.savefig(out_dir / "feasible_scatter_lift_vs_distance.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4.8))
    plt.scatter(contacts, lift, c=score, s=24, alpha=0.85, cmap="plasma")
    if pareto_indices:
        plt.scatter(
            contacts[pareto_indices],
            lift[pareto_indices],
            facecolors="none",
            edgecolors="black",
            s=75,
            linewidths=1.2,
        )
    plt.xlabel("Contacts")
    plt.ylabel("Cube/prism lift")
    plt.title("Lift vs Contacts")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "feasible_scatter_lift_vs_contacts.png", dpi=180)
    plt.close()


def _make_evaluator(args: argparse.Namespace, scene_xml: Path, eval_cfg: Phase1EvalConfig) -> Phase1GraspEvaluator:
    return Phase1GraspEvaluator(
        scene_xml=scene_xml,
        keyframe="open",
        cfg=eval_cfg,
        backend=args.backend,
        comfree_stiffness=args.comfree_stiffness,
        comfree_damping=args.comfree_damping,
        backend_nworld=args.backend_nworld,
        backend_nconmax=args.backend_nconmax,
        backend_njmax=args.backend_njmax,
    )


def _adapt_foundational_ctrl(
    args: argparse.Namespace,
    evaluator: Phase1GraspEvaluator,
    candidate_idx: int,
    baseline_ctrl: np.ndarray,
    interval_ctrl: np.ndarray,
    adapt_cfg: Phase1OptimizationConfig,
) -> tuple[np.ndarray, bool, float, str]:
    mode = args.fp_adaptation
    if mode == "none":
        return interval_ctrl, False, 0.0, "none"

    should_refresh = False
    if mode in {"interval-open", "interval-initial-fp"}:
        interval = max(1, int(args.fp_refresh_interval))
        should_refresh = candidate_idx % interval == 0
    elif mode == "sparse-per-morph":
        should_refresh = True

    if not should_refresh:
        return interval_ctrl, False, 0.0, "reuse"

    if mode == "interval-open":
        init_ctrl = None
        source = "open"
    elif mode == "interval-initial-fp":
        init_ctrl = interval_ctrl
        source = "interval_fp"
    else:
        init_ctrl = baseline_ctrl
        source = "baseline_fp"

    t0 = time.perf_counter()
    refined = optimize_finger_controls(evaluator=evaluator, cfg=adapt_cfg, initial_finger_ctrl=init_ctrl)
    elapsed = float(time.perf_counter() - t0)
    ctrl = np.asarray(refined["best_finger_ctrl"], dtype=np.float64)
    return ctrl, True, elapsed, source


def main() -> None:
    args = build_parser().parse_args()
    rng = np.random.default_rng(args.seed)

    tag = args.tag or datetime.now().strftime("run_%Y%m%d_%H%M%S_pollard_multiscene")
    out_dir = args.output_dir / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    specs, base_morphology = _load_scene_specs(
        args.cube_foundational_run_dir,
        args.prism_foundational_run_dir,
        cube_max_mean_tip_distance=args.cube_max_mean_tip_distance,
        cube_min_contacts=args.cube_min_contacts,
        cube_min_finger_contact_persistence=args.cube_min_finger_contact_persistence,
        cube_max_finger_yaw_drift=args.cube_max_finger_yaw_drift,
        prism_max_mean_tip_distance=args.prism_max_mean_tip_distance,
        prism_min_contacts=args.prism_min_contacts,
        prism_min_finger_contact_persistence=args.prism_min_finger_contact_persistence,
        prism_max_finger_yaw_drift=args.prism_max_finger_yaw_drift,
    )
    bounds = MorphologyBounds(
        x_min=args.x_min,
        x_max=args.x_max,
        y_min=args.y_min,
        y_max=args.y_max,
        len_min=args.len_min,
        len_max=args.len_max,
    )
    candidates = _sample_morphologies(
        base=base_morphology,
        sample_count=args.samples,
        rng=rng,
        bounds=bounds,
        x_perturb=args.x_perturb,
        y_perturb=args.y_perturb,
        len_perturb=args.len_perturb,
    )

    eval_cfg = Phase1EvalConfig(
        settle_steps=args.settle_steps,
        lift_steps=args.lift_steps,
        hold_steps=args.hold_steps,
        lift_delta_z=args.lift_delta_z,
        lift_ramp_steps=args.lift_ramp_steps,
        objective_weight_min_finger_persistence=args.objective_weight_min_finger_persistence,
        objective_weight_finger_persistence_imbalance_penalty=(
            args.objective_weight_finger_persistence_imbalance_penalty
        ),
        objective_weight_finger_yaw_drift_penalty=args.objective_weight_finger_yaw_drift_penalty,
        objective_weight_finger_flex_drift_penalty=args.objective_weight_finger_flex_drift_penalty,
    )
    fp_adapt_cfg = Phase1OptimizationConfig(
        iterations=args.fp_adapt_iterations,
        population=args.fp_adapt_population,
        elite_fraction=args.fp_adapt_elite_fraction,
        sigma_init=args.fp_adapt_sigma_init,
        seed=args.fp_adapt_seed,
    )

    global_rows: list[dict[str, float | str]] = []
    scene_summaries: dict[str, dict[str, int]] = {}

    for spec in specs:
        scene_dir = out_dir / spec.scene_key
        scene_dir.mkdir(parents=True, exist_ok=True)
        generated_dir = scene_dir / "generated_mjcf"
        generated_dir.mkdir(parents=True, exist_ok=True)

        all_rows: list[dict[str, float | str]] = []
        feasible_rows: list[dict[str, float | str]] = []
        scene_eval_seconds_total = 0.0
        scene_adapt_seconds_total = 0.0
        scene_adapt_count = 0

        default_pose = max(spec.foundational_poses, key=lambda p: p.score)
        baseline_fp_ctrl = np.asarray(default_pose.finger_ctrl, dtype=np.float64)
        interval_fp_ctrl = baseline_fp_ctrl.copy()

        for idx, morphology in enumerate(candidates):
            suffix = build_morphology_suffix(morphology)
            scene_xml = generated_dir / f"scene_{spec.scene_key}_{suffix}.xml"
            _make_scene_for_spec(
                base_scene_xml=args.base_scene_xml,
                scene_output_path=scene_xml,
                morphology=morphology,
                spec=spec,
            )

            evaluator = _make_evaluator(args=args, scene_xml=scene_xml, eval_cfg=eval_cfg)

            fp_adapt_triggered = False
            fp_adapt_seconds = 0.0
            fp_adapt_source = "none"
            chosen_ctrl: np.ndarray
            chosen_pose_label = ""
            feasible_pose_count = 0

            if args.fp_adaptation == "none":
                pose_evals: list[tuple[FoundationalPose, float, dict[str, float], bool]] = []
                eval_t0 = time.perf_counter()
                for pose in spec.foundational_poses:
                    score, metrics = evaluator.evaluate(pose.finger_ctrl)
                    feasible = _is_feasible(
                        metrics,
                        spec.max_mean_tip_distance,
                        spec.min_contacts,
                        spec.min_finger_contact_persistence,
                        spec.max_finger_yaw_drift,
                    )
                    pose_evals.append((pose, score, metrics, feasible))
                eval_seconds = float(time.perf_counter() - eval_t0)
                feasible_hits = [p for p in pose_evals if p[3]]
                feasible_pose_count = len(feasible_hits)
                is_feasible = len(feasible_hits) > 0
                chosen_pool = feasible_hits if feasible_hits else pose_evals
                chosen_pose, chosen_score, chosen_metrics, _ = max(chosen_pool, key=lambda t: t[1])
                chosen_pose_label = chosen_pose.label
                chosen_ctrl = np.asarray(chosen_pose.finger_ctrl, dtype=np.float64)
            else:
                ctrl, fp_adapt_triggered, fp_adapt_seconds, fp_adapt_source = _adapt_foundational_ctrl(
                    args=args,
                    evaluator=evaluator,
                    candidate_idx=idx,
                    baseline_ctrl=baseline_fp_ctrl,
                    interval_ctrl=interval_fp_ctrl,
                    adapt_cfg=fp_adapt_cfg,
                )
                if args.fp_adaptation in {"interval-open", "interval-initial-fp"} and fp_adapt_triggered:
                    interval_fp_ctrl = ctrl.copy()

                eval_t0 = time.perf_counter()
                chosen_score, chosen_metrics = evaluator.evaluate(ctrl)
                eval_seconds = float(time.perf_counter() - eval_t0)
                is_feasible = _is_feasible(
                    chosen_metrics,
                    spec.max_mean_tip_distance,
                    spec.min_contacts,
                    spec.min_finger_contact_persistence,
                    spec.max_finger_yaw_drift,
                )
                feasible_pose_count = 1 if is_feasible else 0
                chosen_pose_label = f"adapted_{args.fp_adaptation}"
                chosen_ctrl = ctrl

            scene_eval_seconds_total += eval_seconds
            scene_adapt_seconds_total += fp_adapt_seconds
            scene_adapt_count += int(fp_adapt_triggered)

            qpos_foundational, ctrl_foundational = _simulate_settle_qpos(
                scene_xml=scene_xml,
                keyframe_name="open",
                finger_ctrl=chosen_ctrl,
                settle_steps=args.settle_steps,
            )
            _add_foundational_keyframe(
                scene_xml=scene_xml,
                key_name="foundational",
                qpos=qpos_foundational,
                ctrl=ctrl_foundational,
            )

            row = {
                "scene_key": spec.scene_key,
                "candidate_id": str(idx),
                "scene_xml": str(scene_xml),
                "selected_foundational_pose": chosen_pose_label,
                "feasible": str(is_feasible),
                "backend": args.backend,
                "fp_adaptation": args.fp_adaptation,
                "fp_adapt_triggered": str(fp_adapt_triggered),
                "fp_adapt_source": fp_adapt_source,
                "fp_adapt_seconds": float(fp_adapt_seconds),
                "evaluation_seconds": float(eval_seconds),
                "feasibility_max_mean_tip_distance": float(spec.max_mean_tip_distance),
                "feasibility_min_contacts": float(spec.min_contacts),
                "feasibility_min_finger_contact_persistence": float(
                    spec.min_finger_contact_persistence
                ),
                "feasibility_max_finger_yaw_drift": float(spec.max_finger_yaw_drift),
                "feasible_pose_count": float(feasible_pose_count),
                "foundational_pose_count": float(len(spec.foundational_poses)),
                "score": float(chosen_score),
                "cube_lift": float(chosen_metrics.get("cube_lift", 0.0)),
                "cube_tip_contacts": float(chosen_metrics.get("cube_tip_contacts", 0.0)),
                "mean_tip_distance": float(chosen_metrics.get("mean_tip_distance", 0.0)),
                "cube_vel_norm": float(chosen_metrics.get("cube_vel_norm", 0.0)),
                "cube_xy_drift": float(chosen_metrics.get("cube_xy_drift", 0.0)),
                "cube_z_drop_from_peak": float(chosen_metrics.get("cube_z_drop_from_peak", 0.0)),
                "contact_persistence": float(chosen_metrics.get("contact_persistence", 0.0)),
                "thumb_contact_persistence": float(chosen_metrics.get("thumb_contact_persistence", 0.0)),
                "index_contact_persistence": float(chosen_metrics.get("index_contact_persistence", 0.0)),
                "middle_contact_persistence": float(chosen_metrics.get("middle_contact_persistence", 0.0)),
                "all_finger_contact_persistence": float(
                    chosen_metrics.get("all_finger_contact_persistence", 0.0)
                ),
                "min_finger_contact_persistence": float(
                    chosen_metrics.get("min_finger_contact_persistence", 0.0)
                ),
                "finger_persistence_imbalance": float(
                    chosen_metrics.get("finger_persistence_imbalance", 0.0)
                ),
                "finger_yaw_drift": float(chosen_metrics.get("finger_yaw_drift", 0.0)),
                "finger_flex_drift": float(chosen_metrics.get("finger_flex_drift", 0.0)),
                "thumb_x": float(morphology.thumb_x),
                "thumb_y": float(morphology.thumb_y),
                "thumb_len": float(morphology.thumb_len),
                "index_x": float(morphology.index_x),
                "index_y": float(morphology.index_y),
                "index_len": float(morphology.index_len),
                "middle_x": float(morphology.middle_x),
                "middle_y": float(morphology.middle_y),
                "middle_len": float(morphology.middle_len),
                "chosen_ctrl_json": json.dumps(chosen_ctrl.tolist()),
            }
            all_rows.append(row)
            global_rows.append(row)
            if is_feasible:
                feasible_rows.append(row)

        pareto_idx = _pareto_front_indices(feasible_rows)
        pareto_rows = [feasible_rows[i] for i in pareto_idx]

        _write_csv(all_rows, scene_dir / "all_candidates.csv")
        _write_csv(feasible_rows, scene_dir / "feasible_candidates.csv")
        _write_csv(pareto_rows, scene_dir / "pareto_front.csv")
        _plot_scene_results(feasible_rows, pareto_idx, scene_dir)

        ranking_pool = feasible_rows if feasible_rows else all_rows

        if args.refine_top_k and ranking_pool:
            refine_cfg = Phase1OptimizationConfig(
                iterations=args.refine_iterations,
                population=args.refine_population,
                elite_fraction=args.refine_elite_fraction,
                sigma_init=args.refine_sigma_init,
                seed=args.refine_seed,
            )
            candidate_pool = sorted(ranking_pool, key=lambda r: float(r["score"]), reverse=True)[
                : max(args.top_k_gifs, args.refine_pool_size)
            ]
            refined_rows: list[dict[str, float | str]] = []
            for row in candidate_pool:
                evaluator = _make_evaluator(
                    args=args,
                    scene_xml=Path(str(row["scene_xml"])),
                    eval_cfg=eval_cfg,
                )
                init_ctrl = np.asarray(json.loads(str(row["chosen_ctrl_json"])), dtype=np.float64)
                refined = optimize_finger_controls(evaluator=evaluator, cfg=refine_cfg, initial_finger_ctrl=init_ctrl)
                ctrl_refined = np.asarray(refined["best_finger_ctrl"], dtype=np.float64)
                score_refined, metrics_refined = evaluator.evaluate(ctrl_refined)

                updated = dict(row)
                updated["score"] = float(score_refined)
                updated["cube_lift"] = float(metrics_refined.get("cube_lift", 0.0))
                updated["cube_tip_contacts"] = float(metrics_refined.get("cube_tip_contacts", 0.0))
                updated["mean_tip_distance"] = float(metrics_refined.get("mean_tip_distance", 0.0))
                updated["cube_vel_norm"] = float(metrics_refined.get("cube_vel_norm", 0.0))
                updated["cube_xy_drift"] = float(metrics_refined.get("cube_xy_drift", 0.0))
                updated["cube_z_drop_from_peak"] = float(metrics_refined.get("cube_z_drop_from_peak", 0.0))
                updated["contact_persistence"] = float(metrics_refined.get("contact_persistence", 0.0))
                updated["thumb_contact_persistence"] = float(
                    metrics_refined.get("thumb_contact_persistence", 0.0)
                )
                updated["index_contact_persistence"] = float(
                    metrics_refined.get("index_contact_persistence", 0.0)
                )
                updated["middle_contact_persistence"] = float(
                    metrics_refined.get("middle_contact_persistence", 0.0)
                )
                updated["all_finger_contact_persistence"] = float(
                    metrics_refined.get("all_finger_contact_persistence", 0.0)
                )
                updated["min_finger_contact_persistence"] = float(
                    metrics_refined.get("min_finger_contact_persistence", 0.0)
                )
                updated["finger_persistence_imbalance"] = float(
                    metrics_refined.get("finger_persistence_imbalance", 0.0)
                )
                updated["finger_yaw_drift"] = float(metrics_refined.get("finger_yaw_drift", 0.0))
                updated["finger_flex_drift"] = float(metrics_refined.get("finger_flex_drift", 0.0))
                updated["chosen_ctrl_json"] = json.dumps(ctrl_refined.tolist())
                updated["refined_for_topk"] = "True"
                refined_rows.append(updated)

            top_rows = sorted(refined_rows, key=lambda r: float(r["score"]), reverse=True)[: args.top_k_gifs]
        else:
            top_rows = sorted(ranking_pool, key=lambda r: float(r["score"]), reverse=True)[: args.top_k_gifs]
        gif_dir = scene_dir / "top_gifs"
        gif_dir.mkdir(parents=True, exist_ok=True)

        top_rows_export: list[dict[str, float | str]] = []
        for rank, row in enumerate(top_rows, start=1):
            evaluator = _make_evaluator(
                args=args,
                scene_xml=Path(str(row["scene_xml"])),
                eval_cfg=eval_cfg,
            )
            ctrl = np.asarray(json.loads(str(row["chosen_ctrl_json"])), dtype=np.float64)
            gif_path = gif_dir / f"rank{rank:02d}_candidate{int(row['candidate_id']):04d}.gif"
            evaluator.render_rollout_gif(ctrl, gif_path)
            item = dict(row)
            item["rank"] = float(rank)
            item["gif_path"] = str(gif_path)
            if "refined_for_topk" not in item:
                item["refined_for_topk"] = "False"
            top_rows_export.append(item)

        _write_csv(top_rows_export, scene_dir / "top5_with_gifs.csv")

        scene_summaries[spec.scene_key] = {
            "total_candidates": len(all_rows),
            "feasible_candidates": len(feasible_rows),
            "pareto_size": len(pareto_rows),
            "gif_count": len(top_rows_export),
            "gif_ranking_source": "feasible" if feasible_rows else "all_candidates",
            "backend": args.backend,
            "fp_adaptation": args.fp_adaptation,
            "fp_adapt_count": scene_adapt_count,
            "fp_adapt_seconds_total": scene_adapt_seconds_total,
            "evaluation_seconds_total": scene_eval_seconds_total,
            "mean_evaluation_seconds": scene_eval_seconds_total / max(1, len(all_rows)),
        }

        print(
            f"[{spec.scene_key}] total={len(all_rows)} feasible={len(feasible_rows)} pareto={len(pareto_rows)} gifs={len(top_rows_export)}"
        )

    _write_csv(global_rows, out_dir / "all_scenes_candidates.csv")

    summary = {
        "tag": tag,
        "samples": args.samples,
        "seed": args.seed,
        "backend": args.backend,
        "fp_adaptation": args.fp_adaptation,
        "fp_refresh_interval": args.fp_refresh_interval,
        "fp_adapt_config": {
            "iterations": args.fp_adapt_iterations,
            "population": args.fp_adapt_population,
            "elite_fraction": args.fp_adapt_elite_fraction,
            "sigma_init": args.fp_adapt_sigma_init,
            "seed": args.fp_adapt_seed,
        },
        "backend_config": {
            "comfree_stiffness": args.comfree_stiffness,
            "comfree_damping": args.comfree_damping,
            "backend_nworld": args.backend_nworld,
            "backend_nconmax": args.backend_nconmax,
            "backend_njmax": args.backend_njmax,
        },
        "scene_summaries": scene_summaries,
        "paths": {
            "out_dir": str(out_dir),
            "cube_foundational_run_dir": str(args.cube_foundational_run_dir),
            "prism_foundational_run_dir": str(args.prism_foundational_run_dir),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Multi-scene Pollard run complete: {out_dir}")


if __name__ == "__main__":
    main()
