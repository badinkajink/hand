#!/usr/bin/env python3
"""Second pass of the switch (00:4x on 11-05), for the code that the live text of move_scripts.py leaves out: the lab's
programs, study scripts and queues under docs/experiments/, the poster's scripts under paper/, .gitignore and the VS Code
workspace. In each .py and .sh there, scripts/<name> becomes its path in scripts/MOVED.tsv, src/morphohand becomes
src/sr2hand, imports and dotted names of the package follow, and `import _paths` comes after each line that puts scripts/
on sys.path. The CB1's paths (~/hand/src/morphohand, irlab/hand/src/morphohand) stay. Run once from the repository root."""
import re
import sys
from pathlib import Path

PATHS_IMPORT = "import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path"
PKG = r"\bmorphohand\.(?=(?:geometry|numerics|rl|sampling|optimization|bench|studies|tools|control|backends|uhas|config|driver)\b)"
INSERT = re.compile(r"""^(?P<ind>[ \t]*)sys\.path(?:\.insert\(0,\s*|\[:0\]\s*=\s*\[.*?)"""
                    r"""(?:str\((?:ROOT|REPO)\s*/\s*"scripts"\)|"(?:/home/humanoid/Programs/hand/)?scripts")[^\n]*$""", re.M)
root = Path(".").resolve()
moved = dict(l.rstrip("\n").split("\t") for l in open("scripts/MOVED.tsv") if not l.startswith("#"))
names = sorted((k.split("/", 1)[1] for k in moved), key=len, reverse=True)
ref = re.compile(r"(?<![\w./-])((?:~/Programs/hand/|/home/humanoid/Programs/hand/)?)scripts/("
                 + "|".join(re.escape(n) for n in names) + r")(?![\w])")
path = re.compile(r"(?<!~/hand/)(?<!irlab/hand/)\bsrc/morphohand\b")
imp = re.compile(r"^(\s*)(from|import)(\s+)morphohand\b", re.M)
files = [p for d in ("docs/experiments", "paper") for e in ("*.py", "*.sh") for p in (root / d).rglob(e)]
n = 0
for f in sorted(files):
    try:
        t = f.read_text()
    except UnicodeDecodeError:
        print("not text:", f)
        continue
    u = ref.sub(lambda m: m.group(1) + moved["scripts/" + m.group(2)], t)
    u = path.sub("src/sr2hand", imp.sub(r"\1\2\3sr2hand", re.sub(PKG, "sr2hand.", u)))
    if PATHS_IMPORT not in u:
        u = INSERT.sub(lambda m: m.group(0) + f"\n{m.group('ind')}{PATHS_IMPORT}", u)
    if u != t:
        f.write_text(u)
        n += 1
        print(f.relative_to(root))
print(f"{n} of {len(files)} files rewritten", file=sys.stderr)
for rel in (".gitignore", "hand_driver.code-workspace"):
    p = root / rel
    t = p.read_text()
    u = path.sub("src/sr2hand", t)
    if u != t:
        p.write_text(u)
        print(rel)
