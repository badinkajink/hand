"""Put every folder of scripts/ and the repository's src/ on sys.path, so that a script imports any other script by its
bare name. scripts/ was one flat folder until 2026-11-03; scripts/MOVED.tsv maps the old paths to the new ones.

A script in a folder of scripts/ starts with

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/, for _paths
    import _paths

and finds another script's file with _paths.script("name.py").
"""
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
SKIP = {"archive", "__pycache__"}


def folders():
    """The folders of scripts/ with scripts in them; packages such as contact_surface/ are imported by their name."""
    return sorted(p for p in SCRIPTS.iterdir() if p.is_dir() and p.name not in SKIP
                  and not p.name.startswith((".", "_")) and not (p / "__init__.py").exists())


def script(name):
    """Path of the script `name` (with its extension) in whichever folder of scripts/ it is in."""
    for d in (SCRIPTS, *folders()):
        if (d / name).is_file():
            return d / name
    raise FileNotFoundError(f"scripts/*/{name}")


for _p in [str(ROOT / "src"), str(SCRIPTS), *map(str, folders())][::-1]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
