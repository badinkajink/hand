"""Named, versioned trainer recipes (CODEBASE_AUDIT.md step 4).

The trainer's ~120-flag surface is the real fragility: recipes used to live as
flag soup in the .sh launchers and study scripts, and train/deploy parity bugs
(gotcha #13, `finger_residual_scale`) recurred. A recipe YAML under
`configs/recipes/<name>.yaml` pins a named block of trainer args in ONE tracked
place; launchers and studies pass `--recipe <name>` plus only the run-specific
knobs (morphology run, checkpoints, tag, timesteps).

Precedence: explicit CLI flag > recipe value > Args default. The recipe is
applied as tyro's *default instance*, so anything typed on the command line
still wins — a launcher env-var knob overrides its recipe pin exactly like it
overrode the old hardcoded flag.

Unknown recipe keys are a HARD error (a typo'd key silently reverting to the
trainer default is precisely the parity bug this layer exists to kill).
"""
from __future__ import annotations

import dataclasses
import sys
import types
import typing
from pathlib import Path

import tyro
import yaml

# src/morphohand/rl/recipes.py -> repo root (editable install from src/).
ROOT = Path(__file__).resolve().parents[3]
RECIPE_DIR = ROOT / "configs" / "recipes"


def recipe_path(name_or_path: str) -> Path:
    """Resolve a recipe name (`a_lift`) or explicit path to a YAML file."""
    p = Path(name_or_path)
    if p.suffix in (".yaml", ".yml") or p.exists():
        return p
    return RECIPE_DIR / f"{name_or_path}.yaml"


def _coerce(value, hint):
    """Best-effort YAML->annotation coercion: str->Path, list->tuple."""
    if hint is None or value is None:
        return value
    origin = typing.get_origin(hint)
    if origin in (typing.Union, types.UnionType):
        non_none = [a for a in typing.get_args(hint) if a is not type(None)]
        if not non_none:
            return value
        hint = non_none[0]
        origin = typing.get_origin(hint)
    if hint is Path:
        return Path(value)
    if origin is tuple and isinstance(value, list):
        return tuple(value)
    return value


def load_recipe(name_or_path: str, args_cls) -> dict:
    """Load + validate a recipe against `args_cls`'s fields. Returns {field: value}."""
    path = recipe_path(name_or_path)
    if not path.exists():
        known = sorted(p.stem for p in RECIPE_DIR.glob("*.yaml"))
        raise SystemExit(f"[recipe] '{name_or_path}' not found ({path}). Known: {known}")
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"[recipe] {path} must be a YAML mapping of trainer args")
    field_names = {f.name for f in dataclasses.fields(args_cls)} - {"recipe"}
    unknown = sorted(set(data) - field_names)
    if unknown:
        raise SystemExit(f"[recipe] {path}: unknown trainer args {unknown} "
                         f"(typo? field renamed?) — refusing to run with silent defaults")
    hints = typing.get_type_hints(args_cls)
    return {k: _coerce(v, hints.get(k)) for k, v in data.items()}


def cli_with_recipe(args_cls, argv: list[str] | None = None):
    """tyro.cli with `--recipe <name-or-path>` support.

    The recipe becomes the default instance (fields without a recipe value or a
    dataclass default stay required), so explicit CLI flags override recipe
    values. If `args_cls` has a `recipe` field, it is set to the recipe name for
    provenance (it lands in the run's dumped config)."""
    argv = list(sys.argv[1:]) if argv is None else list(argv)
    recipe = None
    rest: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--recipe":
            recipe = argv[i + 1]
            i += 2
            continue
        if a.startswith("--recipe="):
            recipe = a.split("=", 1)[1]
            i += 1
            continue
        rest.append(a)
        i += 1
    if recipe is None:
        return tyro.cli(args_cls, args=rest)
    overrides = load_recipe(recipe, args_cls)
    defaults = {}
    for f in dataclasses.fields(args_cls):
        if f.name in overrides:
            defaults[f.name] = overrides[f.name]
        elif f.default is not dataclasses.MISSING:
            defaults[f.name] = f.default
        elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            defaults[f.name] = f.default_factory()  # type: ignore[misc]
        else:
            defaults[f.name] = tyro.MISSING
    print(f"[recipe] {recipe} ({recipe_path(recipe)}): pins {sorted(overrides)}")
    args = tyro.cli(args_cls, default=args_cls(**defaults), args=rest)
    if hasattr(args, "recipe"):
        args.recipe = recipe
    return args
