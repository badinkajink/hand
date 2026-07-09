"""Unit tests for morphohand.rl.recipes (the --recipe YAML layer)."""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

from morphohand.rl import recipes


@dataclasses.dataclass
class Toy:
    morphology_run: Path                      # required (no default)
    recipe: str | None = None
    tag: str = "default"
    scale: float = 0.2
    bounds: tuple[float, float] = (0.97, 0.985)
    ckpt: Path | None = None
    easing: str = "linear"


def _write(tmp_path, text) -> Path:
    p = tmp_path / "r.yaml"
    p.write_text(text)
    return p


def test_load_recipe_coerces_types(tmp_path):
    p = _write(tmp_path, "morphology_run: results/x\nbounds: [0.1, 0.2]\nckpt: a/b.pt\nscale: 0.5\n")
    d = recipes.load_recipe(str(p), Toy)
    assert d["morphology_run"] == Path("results/x")
    assert d["bounds"] == (0.1, 0.2)
    assert d["ckpt"] == Path("a/b.pt")
    assert d["scale"] == 0.5


def test_unknown_key_is_fatal(tmp_path):
    p = _write(tmp_path, "finger_residual_scal: 0.5\n")  # typo'd key
    with pytest.raises(SystemExit, match="unknown trainer args"):
        recipes.load_recipe(str(p), Toy)


def test_missing_recipe_is_fatal():
    with pytest.raises(SystemExit, match="not found"):
        recipes.load_recipe("no_such_recipe_xyz", Toy)


def test_cli_overrides_recipe(tmp_path):
    p = _write(tmp_path, "scale: 0.5\neasing: ease_out_quad\n")
    args = recipes.cli_with_recipe(
        Toy, ["--recipe", str(p), "--morphology-run", "m", "--scale", "0.7"])
    assert args.scale == 0.7            # explicit CLI wins
    assert args.easing == "ease_out_quad"  # recipe fills the rest
    assert args.tag == "default"        # dataclass default preserved
    assert args.recipe == str(p)        # provenance recorded


def test_recipe_equals_syntax_and_no_recipe(tmp_path):
    p = _write(tmp_path, "tag: pinned\n")
    args = recipes.cli_with_recipe(Toy, [f"--recipe={p}", "--morphology-run", "m"])
    assert args.tag == "pinned"
    args = recipes.cli_with_recipe(Toy, ["--morphology-run", "m"])
    assert args.tag == "default" and args.recipe is None


def test_required_field_stays_required(tmp_path):
    p = _write(tmp_path, "tag: pinned\n")
    with pytest.raises(SystemExit):
        recipes.cli_with_recipe(Toy, ["--recipe", str(p)])  # morphology_run missing


def test_recipe_can_satisfy_required_field(tmp_path):
    p = _write(tmp_path, "morphology_run: results/m05\n")
    args = recipes.cli_with_recipe(Toy, ["--recipe", str(p)])
    assert args.morphology_run == Path("results/m05")


def test_shipped_recipes_validate_against_trainer_args():
    """Every configs/recipes/*.yaml must name only real trainer args (the parity guard)."""
    sys.path.insert(0, str(recipes.ROOT / "scripts"))
    try:
        from rl_train_cube import Args
    finally:
        sys.path.pop(0)
    shipped = sorted(recipes.RECIPE_DIR.glob("*.yaml"))
    assert shipped, "no shipped recipes found"
    for p in shipped:
        d = recipes.load_recipe(str(p), Args)
        assert d, f"{p} is empty"
