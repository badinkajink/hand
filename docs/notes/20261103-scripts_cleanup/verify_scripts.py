#!/usr/bin/env python3
"""Static checks of scripts/ after the move into topic folders (2026-11-03). Runs no script's code.

    python3 scripts/docs/check_scripts.py [--root REPO]

1. every .py under scripts/ (archive aside), src/ and tests/ parses;
2. every bare import of a script resolves, through the folders _paths puts on sys.path, by importlib's finder;
3. a script run as a program reaches `import _paths` before its first top-level import of a script in another folder;
4. no moved script computes the root one level short (parents[1] or two dirnames of __file__, other than scripts/);
5. every "scripts/..." path in the live text names a file that exists;
6. no live code imports the package under its old name, morphohand.
Prints one line per problem and the counts; exits 1 if there is any problem.
"""
import argparse
import ast
import importlib.machinery
import re
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    a = ap.parse_args()
    root = Path(a.root).resolve()
    scripts = root / "scripts"
    sys.path.insert(0, str(scripts))
    import _paths  # noqa: E402
    search = [str(root / "src"), str(scripts), *map(str, _paths.folders())]
    where = {}
    for d in [scripts, *_paths.folders()]:
        for p in d.glob("*.py"):
            where[p.stem] = p
    archived = {p.stem for p in (scripts / "archive").glob("*.py")}
    problems = []
    files = [p for p in scripts.rglob("*.py") if "archive" not in p.relative_to(scripts).parts
             and "__pycache__" not in p.parts]
    files += [p for p in (root / "src").rglob("*.py") if "firmware" not in p.parts] + list((root / "tests").rglob("*.py"))
    n_imports = 0
    for p in files:
        rel = p.relative_to(root)
        src = p.read_text(errors="replace")
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            problems.append(f"{rel}: does not parse: {e}")
            continue
        # 2. imports of scripts (by bare name) resolve; none names an archived script
        first_cross = None
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [x.name.split(".")[0] for x in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                names = [node.module.split(".")[0]]
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("scripts."):
                parts = node.module.split(".")
                if not (scripts.joinpath(*parts[1:-1]) / f"{parts[-1]}.py").is_file():
                    problems.append(f"{rel}:{node.lineno}: from {node.module} names no file under scripts/")
            for n in names:
                if n in archived and n not in where:
                    problems.append(f"{rel}:{node.lineno}: imports archived script {n}")
                if n in where:
                    n_imports += 1
                    spec = importlib.machinery.PathFinder.find_spec(n, search)
                    if spec is None or Path(spec.origin) != where[n]:
                        problems.append(f"{rel}:{node.lineno}: import {n} resolves to "
                                        f"{spec.origin if spec else None}, expected {where[n]}")
                    if p.parent != where[n].parent and p.parent.parent == scripts and node in tree.body \
                            and (first_cross is None or node.lineno < first_cross):
                        first_cross = node.lineno
        # 3. _paths before the first top-level cross-folder import, for a script in a folder of scripts/
        if first_cross is not None and not (p.parent / "__init__.py").exists():
            m = re.search(r"^\s*import _paths\b", src, re.M)
            line = src[:m.start()].count("\n") + 1 if m else None
            if line is None or line > first_cross:
                problems.append(f"{rel}:{first_cross}: imports a script of another folder before `import _paths`")
        # 4. root one level short
        if p.parent.parent == scripts:
            for i, line in enumerate(src.splitlines(), 1):
                if re.search(r"Path\(__file__\)(\.resolve\(\))?\.parents\[1\]", line) and "# scripts/" not in line \
                        and not re.match(r"\s*(_?sys\.path\.(insert|append)|SCRIPTS\s*=)", line) \
                        and '"scripts"' not in line and "/ 'scripts'" not in line:
                    problems.append(f"{rel}:{i}: root one level short: {line.strip()[:100]}")
                if re.search(r"(?<!dirname\()os\.path\.dirname\(os\.path\.dirname\(os\.path\.abspath\(__file__\)\)\)",
                             line) and "sys.path" not in line:
                    problems.append(f"{rel}:{i}: root one level short: {line.strip()[:100]}")
    # 6. the package is sr2hand since 2026-11-03: no live code imports or names morphohand's modules
    old_pkg = re.compile(r"^\s*(from|import)\s+morphohand\b|\bmorphohand\.(geometry|numerics|rl|sampling|optimization|"
                         r"bench|studies|tools|control|backends|uhas|config|driver)\b", re.M)
    for p in files:
        t = p.read_text(errors="replace")
        for m in old_pkg.finditer(t):
            problems.append(f"{p.relative_to(root)}:{t[:m.start()].count(chr(10)) + 1}: names the old package: {m.group(0).strip()}")
    if not (root / "src" / "sr2hand" / "__init__.py").exists():
        problems.append("src/sr2hand/__init__.py does not exist")
    # 5. scripts/... paths in the live text
    live = [*files, *(scripts.rglob("*.sh")), *(scripts.rglob("*.html")), *(scripts.rglob("*.md")),
            *(root / ".claude").rglob("*.md"), root / "CLAUDE.md", root / "AGENTS.md", root / "README.md",
            *root.glob("docs/experiments/*/README.md"), *root.glob("docs/experiments/*/*/README.md")]
    ref = re.compile(r"(?<![\w./~-])scripts/((?:[\w-]+/)?[\w.-]+\.(?:py|sh|html|json))\b")
    n_refs = 0
    for p in live:
        if not p.is_file() or "archive" in p.relative_to(root).parts:
            continue
        t = p.read_text(errors="replace")
        for m in ref.finditer(t):
            n_refs += 1
            target = scripts / m.group(1)
            if not target.exists():
                line = t[:m.start()].count("\n") + 1
                problems.append(f"{p.relative_to(root)}:{line}: names scripts/{m.group(1)}, which does not exist")
    for x in problems:
        print(x)
    print(f"{len(files)} files, {n_imports} imports of scripts, {n_refs} script paths in the live text; "
          f"{len(problems)} problems")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
