#!/usr/bin/env python3
"""Move the flat scripts/ folder into topic folders (2026-11-03).

    python3 move_scripts.py PLAN.json [--root REPO] [--dry-run]

PLAN.json maps a file name in scripts/ (a .py or .sh) to a folder name; "archive" means scripts/archive/. A builder's
template (<stem>.template.html, <stem>.html) and data file (<stem>.json) move with it. In each moved script (not in the
archive, whose files stay as they were so that they run again once moved back):

  - the repository root, computed from __file__, gains one level (parents[1] -> parents[2], one more dirname, "/.."
    -> "/../.." in shell scripts);
  - every insertion of scripts/ into sys.path is followed by `import _paths`, which puts every folder of scripts/ and
    src/ on sys.path, so a script still imports any other script by its bare name;
  - a script that imports a script of another folder without inserting scripts/ gets the two bootstrap lines.

Then every "scripts/<name>" in the repository's live text (scripts/, src/, tests/, configs/, .claude/, the root's
markdown, the runbooks) becomes "scripts/<folder>/<name>". Result pages, hand-off notes and logs keep their old paths;
scripts/MOVED.tsv maps them. The tool prints what it could not rewrite: every remaining use of __file__ that is not the
root, for a reader to check.
"""
import argparse
import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

LIVE_TEXT = ["scripts", "src", "tests", "configs", ".claude", "CLAUDE.md", "AGENTS.md", "README.md",
             "docs/hardware_control_station.md", "docs/rl/morph_sweep_STATUS.md"]
# the lab's STATE.md is not rewritten: the lab writes it during runs and takes one appended bullet (lab, 11-03 12:04)
LIVE_GLOBS = ["docs/experiments/*/README.md", "docs/experiments/*/*/README.md"]
TEXT_EXT = {".py", ".sh", ".md", ".txt", ".yaml", ".yml", ".toml", ".json", ".html", ".js", ".css", ".slurm", ".cfg"}
SKIP_PARTS = {"archive", "__pycache__", "firmware", "node_modules"}

ROOT_FIXES = [  # (pattern, replacement) for a script one folder deeper
    (re.compile(r"(Path\(__file__\)(?:\.resolve\(\))?\.parents\[)1(\])"), r"\g<1>2\2"),
    (re.compile(r"(Path\(__file__\)(?:\.resolve\(\))?\.parent\.parent)(?!\.parent)"), r"\1.parent"),
    (re.compile(r"os\.path\.dirname\(os\.path\.dirname\(os\.path\.abspath\(__file__\)\)\)"),
     "os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))"),
]
SH_FIXES = [
    (re.compile(r'(\$\(dirname "\$0"\)/)\.\.(?!/\.\.)'), r"\1../.."),
    (re.compile(r'(\$\(dirname "\$\{BASH_SOURCE\[0\]\}"\)/)\.\.(?!/\.\.)'), r"\1../.."),
]
# sys.path.insert(0, str(ROOT / "scripts")), os.path.join(ROOT, "scripts"), str(SCRIPTS) ...
SCRIPTS_INSERT = re.compile(
    r"""^(?P<ind>[ \t]*)_?sys\.path\.(?:insert\(0,|append\()\s*(?:str\()?\s*(?:"""
    r"""(?:[A-Z_]+|Path\(__file__\)(?:\.resolve\(\))?\.parents\[\d\])\s*/\s*["']scripts["']"""
    r"""|os\.path\.join\([A-Z_]+,\s*["']scripts["']\)"""
    r"""|[A-Z_]*SCRIPTS?[A-Z_]*)[ \t]*\)?[ \t]*\)?[^\n]*$""", re.M)
# a script that put its own folder (scripts/ until now) on sys.path: point it at scripts/, one level up
OWN_DIR = [
    (re.compile(r"^(?P<ind>[ \t]*)(?P<pre>_?sys\.path\.(?:insert\(0,|append\()\s*str\((?:__import__\('pathlib'\)\.)?"
                r"Path\(__file__\)\.resolve\(\))\.parent(?P<post>\)[ \t]*\))[ \t]*$", re.M), r"\g<pre>.parents[1]\g<post>"),
    (re.compile(r"^(?P<ind>[ \t]*)(?P<pre>_?sys\.path\.(?:insert\(0,|append\()\s*str\((?:__import__\('pathlib'\)\.)?"
                r"Path\(__file__\))\.parent(?P<post>\)[ \t]*\))[ \t]*$", re.M), r"\g<pre>.parent.parent\g<post>"),
    (re.compile(r"^(?P<ind>[ \t]*)(?P<pre>_?sys\.path\.(?:insert\(0,|append\()\s*)os\.path\.dirname\(os\.path\.abspath"
                r"\(__file__\)\)(?P<post>[ \t]*\))[ \t]*$", re.M),
     r"\g<pre>os.path.dirname(os.path.dirname(os.path.abspath(__file__)))\g<post>"),
]
BOOT = ("sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/, for _paths\n"
        "import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path\n")
PATHS_IMPORT = "import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path"
PATHS_PY = '''"""Put every folder of scripts/ and the repository's src/ on sys.path, so that a script imports any other script by its
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
'''
PKG_NAMES = ("geometry|numerics|rl|sampling|optimization|bench|studies|tools|control|backends|uhas|config|driver|"
             "create_backend|MorphologyValues|apply_morphology_to_qpos|build_morphology_suffix|build_rigid_model_name|"
             "create_rigid_hand_and_scene_xmls|create_rigid_morphology_xml|extract_morphology_from_qpos|__file__|__path__")
# phrases where the old name means the package, or sits in a string a test compares; applied before the word rules
PKG_PHRASES = [("mujoco, no morphohand.", "mujoco, no sr2hand."), ("mujoco + morphohand.", "mujoco + sr2hand."),
               ("ActionTerms for morphohand.", "ActionTerms for the hand."), ('== "morphohand"', '== "sr2hand"'),
               ('b"MorphoHand control"', 'b"SR2 Hand control"'), ('<span class="mark">MH</span>', '<span class="mark">SR2</span>')]
# the runbook's commands on the CB1 keep the CB1's own src/morphohand/; its workstation commands change by hand
PKG_KEEP_TEXT = {"docs/hardware_control_station.md"}
SPECIAL_PKG = [
    ("docs/hardware_control_station.md",
     "PYTHONPATH=src/morphohand/driver/manta/host python -m manta_hand.web --fake --port 8765",
     "PYTHONPATH=src/sr2hand/driver/manta/host python -m manta_hand.web --fake --port 8765"),
    ("docs/hardware_control_station.md",
     "  --directory src/morphohand/driver/manta/host/manta_hand/static",
     "  --directory src/sr2hand/driver/manta/host/manta_hand/static"),
    ("docs/hardware_control_station.md",
     "## CB1 install\n",
     "## CB1 install\n\nThe CB1's checkout (`~/hand`, never pulled) keeps the package's old folder `src/morphohand/`, so the\n"
     "commands run on the CB1 use it; on the workstation the folder is `src/sr2hand/` since 2026-11-03. A file shipped from\n"
     "`src/sr2hand/driver/manta/host/` goes to the same place under the CB1's `src/morphohand/driver/manta/host/`.\n"),
    ("docs/hardware_control_station.md", "# MorphoHand hardware control station", "# SR² Hand hardware control station"),
    ("pyproject.toml", 'name = "morphohand"', 'name = "sr2hand"'),
    ("uv.lock", 'name = "morphohand"', 'name = "sr2hand"'),
]
# code that builds a script's path instead of naming it in a string: (file after the move, old, new)
SPECIAL = [
    ("scripts/docs/docs_inventory.py",
     'glob.glob(os.path.join(ROOT, "scripts", "*.py"))',
     'glob.glob(os.path.join(ROOT, "scripts", "*", "*.py"))'),
    ("scripts/docs/topic_overview_page.py",
     'os.path.join(ROOT, "scripts", "overview_*_text.py")',
     'os.path.join(ROOT, "scripts", "docs", "overview_*_text.py")'),
    ("scripts/docs/prose_lint.py",
     'for m in dict.fromkeys(re.findall(r"scripts/(\\w+_page)\\.py", page)):\n'
     '            for cand in (f"scripts/{m}.py", f"scripts/{m}.template.html", f"scripts/{m.replace(\'_page\', \'_text\')}.py"):',
     'for m in dict.fromkeys(re.findall(r"scripts/(?:\\w+/)?(\\w+_page)\\.py", page)):\n'
     '            for cand in [os.path.relpath(p, root) for f in (f"{m}.py", f"{m}.template.html",\n'
     '                                                            f"{m.replace(\'_page\', \'_text\')}.py")\n'
     '                         for p in glob.glob(os.path.join(root, "scripts", "*", f))]:'),
    ("scripts/design/sweep_perp_thumb.py",
     'PROJECT_ROOT / "scripts" / "generate_morphology_xml.py"',
     'PROJECT_ROOT / "scripts" / "design" / "generate_morphology_xml.py"'),
    ("scripts/docs/retro_style.py",
     'exp = os.path.join(os.path.dirname(here), "docs", "experiments")',
     'exp = os.path.join(os.path.dirname(os.path.dirname(here)), "docs", "experiments")'),
    ("scripts/docs/retro_style.py",
     'glob.glob(os.path.join(os.path.dirname(here), "docs", "overviews", "*.html"))',
     'glob.glob(os.path.join(os.path.dirname(os.path.dirname(here)), "docs", "overviews", "*.html"))'),
    ("scripts/mechanisms/real_v1_basketball_media.py",
     'str(HERE / "mj_snap.py"), "--scene", str(scene7)', 'str(HERE.parent / "docs" / "mj_snap.py"), "--scene", str(scene7)'),
    ("scripts/mechanisms/real_v1_basketball_media.py",
     'str(HERE / "mj_snap.py"), "--scene", str(scene3)', 'str(HERE.parent / "docs" / "mj_snap.py"), "--scene", str(scene3)'),
    ("README.md",
     "scripts/                the scripts; scripts/README.md maps them, and archive/ has the retired ones\n",
     "scripts/                one folder per topic: docs, design, hom, mechanisms, hardware, contact, fingertip,\n"
     "                        ur5, cad, rl; scripts/README.md maps them, and archive/ has the retired ones\n"),
    ("README.md",
     "uv run --extra rl --extra gpu python scripts/<script>.py",
     "uv run --extra rl --extra gpu python scripts/<folder>/<script>.py"),
    ("CLAUDE.md",
     "notes → `docs/notes/`. `scripts/` holds only the active scripts (map: `scripts/README.md`);\n"
     "  superseded ones live in `scripts/archive/` — resurrect by moving back, don't run in place.",
     "notes → `docs/notes/`. **Scripts (2026-11-03):** one folder per topic under `scripts/` (`docs`, `design`,\n"
     "  `hom`, `mechanisms`, `hardware`, `contact`, `fingertip`, `ur5`, `cad`, `rl`); the map `scripts/README.md` is\n"
     "  written by `scripts/docs/scripts_map.py`. A new script goes in the folder of its topic and starts with\n"
     "  `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))` and `import _paths`, which puts every folder and\n"
     "  `src/` on `sys.path`, so scripts import each other by bare name; `scripts/docs/check_scripts.py` checks the\n"
     "  imports and paths. `scripts/MOVED.tsv` maps the paths before 2026-11-03. Unused scripts go to\n"
     "  `scripts/archive/` — resurrect by moving back, don't run in place."),
    ("scripts/fingertip/sim2real_fingertip_manifest.py",
     'cite(f"scripts/{s}", "code the study runs")',
     'cite(str(_paths.script(s).relative_to(ROOT)), "code the study runs")'),
]


def git(root, *args):
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout


def companions(scripts, stem):
    out = []
    for suffix in (".template.html", ".html", ".json"):
        p = scripts / f"{stem}{suffix}"
        if p.exists():
            out.append(p.name)
    return out


def imported_names(src):
    """Bare top-level module names a script imports anywhere in its body."""
    names = set()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return names
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant) \
                and isinstance(node.args[0].value, str) and (
                    getattr(node.func, "attr", None) == "import_module" or getattr(node.func, "id", None) == "__import__"):
            names.add(node.args[0].value.split(".")[0])
    return names


def first_import_line(src, wanted):
    """1-based line of the first top-level statement that imports one of `wanted`, else None."""
    tree = ast.parse(src)
    for node in tree.body:
        mods = []
        if isinstance(node, ast.Import):
            mods = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            mods = [node.module.split(".")[0]]
        elif isinstance(node, (ast.Try, ast.If)):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Import):
                    mods += [a.name.split(".")[0] for a in sub.names]
                elif isinstance(sub, ast.ImportFrom) and sub.module and not sub.level:
                    mods.append(sub.module.split(".")[0])
        if set(mods) & wanted:
            return node.lineno
    return None


def add_bootstrap(src, wanted):
    """Insert the two bootstrap lines before the first top-level import of a script from another folder (or after the
    module's leading imports when every such import is inside a function)."""
    lines = src.splitlines(keepends=True)
    tree = ast.parse(src)
    line = first_import_line(src, wanted)
    if line is None:  # lazy imports only: after the last leading import or docstring
        line = 1
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)) or (
                    isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)):
                line = node.end_lineno + 1
            else:
                break
    head = "".join(lines[:line - 1])
    need = ""
    if not re.search(r"^import sys\b|^import .*\bsys\b|^from sys\b", head, re.M):
        need += "import sys\n"
    if not re.search(r"^from pathlib import .*\bPath\b", head, re.M):
        need += "from pathlib import Path\n"
    return "".join(lines[:line - 1]) + need + BOOT + "".join(lines[line - 1:])


def live_text_files(root):
    files = []
    for entry in LIVE_TEXT:
        p = root / entry
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            for q in p.rglob("*"):
                if q.is_file() and q.suffix in TEXT_EXT and not (SKIP_PARTS & set(q.relative_to(root).parts)) \
                        and q.stat().st_size < 4_000_000:
                    files.append(q)
    for g in LIVE_GLOBS:
        files += [q for q in root.glob(g) if q.is_file()]
    return sorted(set(files))


def rename_package(root):
    """src/morphohand/ -> src/sr2hand/ (owner, 2026-11-03): the imports, the dotted names and the paths in the live
    code and text, MorphoHandEnvCfg -> HandEnvCfg, the project's name in pyproject.toml and uv.lock, and the venv's
    editable install. The CB1's checkout keeps src/morphohand/ (it is never pulled), so the runbook's CB1 commands keep
    it. Lightning Grasp's asset names (morphohand.urdf, assets/hand/morphohand) are another project's and stay."""
    git(root, "mv", "src/morphohand", "src/sr2hand")
    code = [p for p in [*root.glob("scripts/**/*.py"), *root.glob("scripts/**/*.sh"), *root.glob("src/**/*.py"),
                        *root.glob("tests/**/*.py"), *root.glob("paper/**/*.py"), *root.glob("configs/**/*.yaml")]
            if not ({"archive", "firmware", "__pycache__"} & set(p.relative_to(root).parts))]
    imp = re.compile(r"^(\s*)(from|import)(\s+)morphohand\b", re.M)
    dotted = re.compile(rf"\bmorphohand\.(?=(?:{PKG_NAMES})\b)")
    path = re.compile(r"(?<!~/hand/)(?<!irlab/hand/)\b(src/)morphohand\b")
    word = re.compile(r"(?<![\w\"'/.\-])morphohand(?![\w\"'/.\-])")   # prose: "the morphohand" -> "the hand"
    n = 0
    for f in sorted(set(code) | set(live_text_files(root))):
        if str(f.relative_to(root)) in PKG_KEEP_TEXT:
            continue
        try:
            t = f.read_text()
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        u = path.sub(r"\1sr2hand", dotted.sub("sr2hand.", t))
        if f in code:
            u = imp.sub(r"\1\2\3sr2hand", u).replace("MorphoHandEnvCfg", "HandEnvCfg")
        u = re.sub(r"\b_morphohand_", "_sr2hand_", u).replace("MORPHOHAND_", "SR2HAND_")
        for old_s, new_s in PKG_PHRASES:
            u = u.replace(old_s, new_s)
        u = re.sub(r"\bMorphoHand\b", "SR² Hand" if f.suffix == ".md" else "SR2 Hand", u)
        u = "".join(line if "--name morphohand" in line else word.sub("hand", line)
                    for line in u.splitlines(keepends=True))
        if u != t:
            f.write_text(u)
            n += 1
    for rel, old, new in SPECIAL_PKG:
        p = root / rel
        t = p.read_text()
        if old in t:
            p.write_text(t.replace(old, new, 1))
        else:
            print(f"special edit not applied: {rel}: {old[:60]}")
    print(f"package renamed to sr2hand; rewrote {n} files")
    left = subprocess.run(["rg", "-n", "-i", "morphohand", "scripts", "src", "tests", "configs", ".claude", "CLAUDE.md",
                           "AGENTS.md", "README.md", "pyproject.toml", "-g", "!**/archive/**", "-g", "!*.json"],
                          cwd=root, capture_output=True, text=True).stdout.splitlines()
    print(f"{len(left)} lines still name morphohand (expected: Lightning Grasp's and GraspGenX's asset names, the W&B "
          "project, an SVG hash salt, check_scripts.py):")
    print("\n".join(l[:160] for l in left))
    venv = root / ".venv" / "bin" / "python"
    if venv.exists():  # its editable install puts src/ on sys.path; reinstall under the new name
        for cmd in (["uv", "pip", "uninstall", "--python", str(venv), "morphohand"],
                    ["uv", "pip", "install", "--python", str(venv), "--no-deps", "-e", str(root)]):
            r = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
            print(" ".join(cmd[:3]), "->", r.returncode, (r.stderr.strip().splitlines() or [""])[-1][:120])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("plan")
    ap.add_argument("--root", default=".")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    root = Path(a.root).resolve()
    scripts = root / "scripts"
    plan = json.load(open(a.plan))

    moves = {}    # file name -> folder
    renamed = {}  # file name -> its new name, for a plan value "folder/new_name.py"
    for name, dest in plan.items():
        if not (scripts / name).exists():
            print(f"missing, skipped: scripts/{name}")
            continue
        folder, _, new_name = dest.partition("/")
        moves[name] = folder
        if new_name:
            renamed[name] = new_name
        for c in companions(scripts, Path(name).stem):
            moves.setdefault(c, folder)
    stems = sorted((Path(n).stem for n in moves if n.endswith(".py")), key=len, reverse=True)
    for p in sorted(scripts.glob("*.json")) + sorted(scripts.glob("*.html")):
        if p.name not in moves:  # a data file named after its script: docs_changelog_notes.json
            owner = next((s for s in stems if p.stem.startswith(s + "_")), None)
            if owner:
                moves[p.name] = moves[owner + ".py"]
    unplanned = sorted(p.name for p in scripts.iterdir() if p.is_file() and p.suffix in (".py", ".sh", ".json", ".html")
                       and p.name not in moves and p.name not in ("_paths.py",))
    if unplanned:
        print("not in the plan, left in scripts/:", " ".join(unplanned))

    folder_of = {Path(n).stem: f for n, f in moves.items() if n.endswith((".py", ".sh"))}
    if a.dry_run:
        from collections import Counter
        print(Counter(moves.values()))
        return

    for name, folder in sorted(moves.items()):
        (scripts / folder).mkdir(exist_ok=True)
        git(root, "mv", f"scripts/{name}", f"scripts/{folder}/{renamed.get(name, name)}")

    # 1. root depth, _paths after scripts/ insertions, bootstrap where a cross-folder import has none
    review = []
    for name, folder in sorted(moves.items()):
        if folder == "archive":
            continue
        p = scripts / folder / renamed.get(name, name)
        src = p.read_text()
        new = src
        if name.endswith(".py"):
            for rx, rep in ROOT_FIXES:
                new = rx.sub(rep, new)
            for rx, rep in OWN_DIR:
                new = rx.sub(lambda m: m.expand(f"\\g<ind>{rep}") + f"\n{m.group('ind')}{PATHS_IMPORT}", new)
            new = SCRIPTS_INSERT.sub(lambda m: m.group(0) + f"\n{m.group('ind')}{PATHS_IMPORT}", new)
            new = re.sub(rf"^([ \t]*){re.escape(PATHS_IMPORT)}\n(\1{re.escape(PATHS_IMPORT)}\n)+", r"\1" + PATHS_IMPORT
                         + "\n", new, flags=re.M)
            other = {m for m in imported_names(new) if folder_of.get(m) not in (None, folder, "archive")}
            if other and "import _paths" not in new:
                new = add_bootstrap(new, other)
            for i, line in enumerate(new.splitlines(), 1):
                if "__file__" in line and not any(rx.search(line) for rx, _ in ROOT_FIXES) \
                        and "parents[2]" not in line and ".parent.parent.parent" not in line \
                        and "parents[1]))  # scripts/" not in line:
                    review.append(f"scripts/{folder}/{name}:{i}: {line.strip()[:120]}")
        elif name.endswith(".sh"):
            for rx, rep in SH_FIXES:
                new = rx.sub(rep, new)
        if new != src:
            p.write_text(new)

    # 2. scripts/<name> -> scripts/<folder>/<name> in the live text
    names = sorted(moves, key=len, reverse=True)
    ref = re.compile(r"(?<![\w./-])((?:~/Programs/hand/|/home/humanoid/Programs/hand/)?scripts/)("
                     + "|".join(re.escape(n) for n in names) + r")(?![\w])")
    changed = 0
    for f in live_text_files(root):
        try:
            t = f.read_text()
        except UnicodeDecodeError:
            continue
        u = ref.sub(lambda m: f"{m.group(1)}{moves[m.group(2)]}/{renamed.get(m.group(2), m.group(2))}", t)
        if u != t:
            f.write_text(u)
            changed += 1

    # 3. src/ and tests/ that put scripts/ on sys.path also need _paths
    for f in [*root.glob("src/**/*.py"), *root.glob("tests/**/*.py")]:
        if "firmware" in f.parts:
            continue
        t = f.read_text()
        u = SCRIPTS_INSERT.sub(lambda m: m.group(0) + f"\n{m.group('ind')}{PATHS_IMPORT}", t)
        if u != t and PATHS_IMPORT not in t:
            f.write_text(u)
    (scripts / "_paths.py").write_text(PATHS_PY)
    for rel, old, new in SPECIAL:
        p = root / rel
        t = p.read_text() if p.exists() else ""
        if old in t:
            p.write_text(t.replace(old, new))
        else:
            print(f"special edit not applied: {rel}: {old[:60]}")

    with open(scripts / "MOVED.tsv", "w") as fh:
        fh.write("# scripts/ moved into topic folders on 2026-11-03; old path, new path\n")
        fh.write("# the package src/morphohand/ is src/sr2hand/ (import sr2hand) since the same day\n")
        for name, folder in sorted(moves.items()):
            fh.write(f"scripts/{name}\tscripts/{folder}/{renamed.get(name, name)}\n")

    rename_package(root)

    # 4. the archive's README, the map, the checker, and this tool among the spent ones
    archived = sorted(Path(n).stem for n, f in moves.items() if f == "archive" and n.endswith((".py", ".sh")))
    readme = scripts / "archive" / "README.md"
    era = ("- **Unreferenced scripts moved in the scripts reorganization (2026-11-03):** scripts that no runbook, overview,\n"
           "  page builder, test, current hand-off note or live script named, and that had not changed since before\n"
           "  2026-10-01 (reference scan and move tool: `move_scripts_20261103.py` here). "
           + ", ".join(f"`{n}`" for n in archived) + ".\n")
    t = readme.read_text()
    if "scripts reorganization (2026-11-03)" not in t:
        readme.write_text(t.replace("Eras, newest first:\n\n", "Eras, newest first:\n\n" + era, 1))
    here = Path(__file__).resolve().parent
    for src, dst in (("scripts_map.py", "docs/scripts_map.py"), ("verify_scripts.py", "docs/check_scripts.py"),
                     ("move_scripts.py", "archive/move_scripts_20261103.py")):
        (scripts / dst).write_text((here / src).read_text())
    subprocess.run([sys.executable, str(scripts / "docs" / "scripts_map.py")], cwd=root, check=True)
    print(f"moved {len(moves)} files; rewrote references in {changed} files")
    print(f"{len(review)} uses of __file__ to check by eye:")
    print("\n".join(review))


if __name__ == "__main__":
    main()
