#!/usr/bin/env python3
"""Keep the repository's Git LFS storage on GitHub to the documentation's content (owner, 2026-11-06: "LFS should really
only be for the docpage videos/content", "i would like to keep the commits, links, hyperlinks intact").

The git history is pushed unchanged, so every commit hash, and every link to a commit or to a page's version, stays
valid. Of the LFS objects that its pointers name, only the upload set goes to GitHub: the pages and their media under
docs/ (html, png, jpg, gif, svg, mp4, webm, mov, pdf), the docs index docs/pages.json and each page's formula cache
texsvg_cache.json, and the simulator's meshes under assets/ (stl, obj, png). CAD
exchange and print files (step, 3mf, zipped exports) stay local (owner, 2026-11-06 evening: "remove the cad files,
particularly the 3mf files (besides the most recent fingertip w 4 magnets)"), except the four-magnet pad 05R4 of the v3
tips (KEEP). Raw robot recordings (*.bin) and the data files committed before 2026-11-09 (json, jsonl, csv, npz, npy) stay on
the workstation and move by rsync; on GitHub their pointers stay in the history without content. Since 2026-11-09
data files are ordinary git files, and the pre-commit hook (`pre-commit`) stops one over 2 MB.

GitHub keeps every LFS object ever pushed to a repository and frees the storage only when the repository is deleted, so
the switch is: fetch every object that exists only on GitHub (needs an LFS budget above zero), delete and recreate the
empty repository under the same name, then `push` from the bare mirror. Afterwards the main clone's pre-push hook
(`install-hook`) uploads only the upload set of each push, when `git config lfslean.upload true` (off from the
afternoon of 2026-11-06, when the account had no budget, until the owner set one that evening). Since then GitHub
refuses a push whose commits name LFS objects that it does not have (GH008), so the hook stops a push whose commits
contain LFS files outside the upload set and lists them.

    python3 scripts/docs/lfs_lean.py catalog [--repo DIR] [--refs REF ...]    # sizes by category, upload set, missing
    python3 scripts/docs/lfs_lean.py refresh --mirror DIR                      # the mirror's branches from the main clone
    python3 scripts/docs/lfs_lean.py stage --mirror DIR                        # hard-link the upload set into its LFS store
    python3 scripts/docs/lfs_lean.py push --mirror DIR [--dry-run] [--allow-missing]   # push branches, upload, verify
    python3 scripts/docs/lfs_lean.py lost --out FILE                           # the objects only GitHub had
    python3 scripts/docs/lfs_lean.py verify [--repo DIR]                       # every upload object present on GitHub?
    python3 scripts/docs/lfs_lean.py install-hook                              # the main clone's pre-push and pre-commit hooks
    (git runs)  python3 scripts/docs/lfs_lean.py pre-commit                    # data files over 2 MB stay out of git
    (git runs)  python3 scripts/docs/lfs_lean.py pre-push REMOTE URL < refs     # the hook itself
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REMOTE_URL = "git@github.com:badinkajink/hand.git"
SLUG = "badinkajink/hand.git"
DOC_EXT = {".html", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".mp4", ".webm", ".mov", ".pdf"}
ASSET_EXT = {".stl", ".obj", ".png"}
KEEP = {"assets/cad_v2/fingertip_prints/pads_05R4_four_magnets_x3_xl.3mf",   # the v3 tips' pads, printed from 10-26
        "docs/experiments/20261018-dip_link_board_mount/v4/b/print/"
        "20261026-dip_link_print_pad_ell11_4c_05R4_x3_85a_t2_g30.3mf"}       # the same file
POINTER = re.compile(r"^([0-9a-f]{40}) blob \d+\n(version https://git-lfs[^\n]*\noid sha256:([0-9a-f]{64})\nsize (\d+))",
                     re.M)


def upload(path: str) -> bool:
    """Whether an LFS file at this path belongs to the upload set."""
    if path in KEEP or path == "docs/pages.json" or path.endswith("/texsvg_cache.json"):   # the docs index; a page's formulas
        return True
    ext = os.path.splitext(path)[1].lower()
    if path.startswith("docs/"):
        return ext in DOC_EXT
    return path.startswith("assets/") and ext in ASSET_EXT


def category(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return {".bin": "raw recordings (.bin)", ".html": "pages", ".json": "data rows", ".jsonl": "data rows",
            ".csv": "data rows", ".npz": "arrays", ".npy": "arrays", ".mp4": "videos", ".webm": "videos",
            ".mov": "videos", ".gif": "images", ".png": "images", ".jpg": "images", ".jpeg": "images",
            ".svg": "images"}.get(ext, "CAD, pdf, zip" if ext in (".stl", ".step", ".3mf", ".obj", ".pdf", ".zip")
                                   else f"other {ext}")


def git(repo, *args, inp=None, env=None):
    return subprocess.run(["git", "-C", str(repo), *args], input=inp, capture_output=True, text=True, errors="replace",
                          env=env, check=True).stdout


def lfs_store(repo: Path) -> Path:
    """The LFS object directory of a work tree (.git/lfs/objects) or of a bare repository (lfs/objects)."""
    gd = Path(git(repo, "rev-parse", "--absolute-git-dir").strip())
    return gd / "lfs" / "objects"


def obj_path(store: Path, oid: str) -> Path:
    return store / oid[:2] / oid[2:4] / oid


def pointers(repo: Path, revs: list[str]) -> dict[str, dict]:
    """Every LFS pointer reachable from revs ("rev-list" arguments): oid -> {size, paths}."""
    out = git(repo, "rev-list", "--objects", *revs)
    paths = {}
    for line in out.splitlines():
        sha, _, p = line.partition(" ")
        if p:
            paths[sha] = p
    if not paths:
        return {}
    chk = git(repo, "cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)", inp="\n".join(paths) + "\n")
    small = [ln.split()[0] for ln in chk.splitlines() if ln.split()[1] == "blob" and 100 < int(ln.split()[2]) < 300]
    recs: dict[str, dict] = {}
    if small:
        body = git(repo, "cat-file", "--batch", inp="\n".join(small) + "\n")
        for m in POINTER.finditer(body):
            r = recs.setdefault(m.group(3), {"size": int(m.group(4)), "paths": set()})
            r["paths"].add(paths.get(m.group(1), ""))
    return recs


def selection(recs):
    """(upload oids, kept-local oids): an object is uploaded when any of its paths is in the upload set."""
    up = {o for o, r in recs.items() if any(upload(p) for p in r["paths"])}
    return up, set(recs) - up


def heads(repo: Path) -> list[str]:
    return [ln for ln in git(repo, "for-each-ref", "--format=%(refname)", "refs/heads").splitlines() if ln]


def report(recs, store: Path):
    up, keep = selection(recs)
    agg = collections.defaultdict(lambda: [0, 0, 0, 0, 0])
    for o, r in recs.items():
        c = category(sorted(r["paths"])[0])
        a = agg[c]
        a[0] += 1
        a[1] += r["size"]
        if o in up:
            a[2] += r["size"]
        if not obj_path(store, o).exists():
            a[3] += 1
            a[4] += r["size"]
    gb = 1e9
    print(f"{'category':24s} {'objects':>8s} {'GB':>7s} {'uploaded GB':>12s} {'missing here':>13s}")
    for c, a in sorted(agg.items(), key=lambda x: -x[1][1]):
        print(f"{c:24s} {a[0]:8d} {a[1] / gb:7.2f} {a[2] / gb:12.2f} {a[3]:7d} ({a[4] / gb:.2f} GB)")
    su = sum(recs[o]["size"] for o in up)
    miss_up = [o for o in up if not obj_path(store, o).exists()]
    print(f"upload set: {len(up)} objects, {su / gb:.2f} GB ({su / 2 ** 30:.2f} GiB); kept local: {len(keep)} objects, "
          f"{sum(recs[o]['size'] for o in keep) / gb:.2f} GB; upload objects missing here: {len(miss_up)} "
          f"({sum(recs[o]['size'] for o in miss_up) / gb:.2f} GB)")
    return up, keep, miss_up


# ------------------------------------------------------------------------------------------------ commands
def cmd_catalog(a):
    repo = Path(a.repo)
    bare = git(repo, "rev-parse", "--is-bare-repository").strip() == "true"
    revs = a.refs or (heads(repo) if bare else ["refs/heads/main", "--remotes=origin"])
    recs = pointers(repo, revs)
    report(recs, lfs_store(Path(a.store_from) if a.store_from else repo))
    return 0


def cmd_refresh(a):
    """The mirror's branches: GitHub's branches as the main clone last fetched them, and the main clone's main."""
    m, src = Path(a.mirror), Path(a.source)
    git(m, "fetch", "-q", str(src), "+refs/remotes/origin/*:refs/heads/*")
    git(m, "fetch", "-q", str(src), "+refs/heads/main:refs/heads/main")
    for r in heads(m):
        if r.endswith("/HEAD"):
            git(m, "update-ref", "-d", r)
    print("\n".join(git(m, "for-each-ref", "--format=%(refname:short) %(objectname:short)", "refs/heads").splitlines()))
    return 0


def cmd_stage(a):
    """Hard-link the mirror's upload set from the main clone's LFS store into the mirror's own."""
    m, src = Path(a.mirror), lfs_store(Path(a.source))
    dst = lfs_store(m)
    recs = pointers(m, heads(m))
    up, _ = selection(recs)
    linked = present = missing = 0
    for o in sorted(up):
        d = obj_path(dst, o)
        if d.exists():
            present += 1
            continue
        s = obj_path(src, o)
        if not s.exists():
            missing += 1
            continue
        d.parent.mkdir(parents=True, exist_ok=True)
        os.link(s, d)
        linked += 1
    extra = [p for p in dst.rglob("*") if p.is_file() and p.name not in up]
    print(f"upload set {len(up)}: linked {linked}, already there {present}, missing in the main clone {missing}; "
          f"objects outside the set in the mirror's store: {len(extra)}")
    return 1 if missing else 0


def lfs_batch(oids_sizes, operation="download"):
    """GitHub's LFS batch API for (oid, size) pairs: {oid: True if the server has it (download) or wants it (upload)}."""
    auth = json.loads(subprocess.run(["ssh", "-o", "BatchMode=yes", "git@github.com", "git-lfs-authenticate", SLUG,
                                      operation], capture_output=True, text=True, check=True).stdout)
    out = {}
    pairs = list(oids_sizes)
    for k in range(0, len(pairs), 100):
        body = {"operation": operation, "transfers": ["basic"],
                "objects": [{"oid": o, "size": s} for o, s in pairs[k:k + 100]]}
        req = urllib.request.Request(auth["href"].rstrip("/") + "/objects/batch", data=json.dumps(body).encode(),
                                     method="POST")
        for h, v in auth.get("header", {}).items():
            req.add_header(h, v)
        req.add_header("Accept", "application/vnd.git-lfs+json")
        req.add_header("Content-Type", "application/vnd.git-lfs+json")
        d = json.loads(urllib.request.urlopen(req, timeout=60).read())
        for ob in d.get("objects", []):
            out[ob["oid"]] = "actions" in ob and "error" not in ob
    return out


def cmd_lost(a):
    """The LFS objects that the refs name and that this clone does not have, with their paths: what GitHub alone had."""
    repo = Path(a.repo)
    recs = pointers(repo, ["refs/heads/main", "--remotes=origin"])
    store = lfs_store(repo)
    rows = [(category(sorted(r["paths"])[0]), r["size"], sorted(r["paths"])[0], o) for o, r in recs.items()
            if not obj_path(store, o).exists()]
    rows.sort()
    Path(a.out).write_text("category\tbytes\tpath\toid\n" + "".join(f"{c}\t{s}\t{p}\t{o}\n" for c, s, p, o in rows))
    print(f"{len(rows)} objects, {sum(r[1] for r in rows) / 1e9:.2f} GB -> {a.out}")
    return 0


def cmd_verify(a):
    repo = Path(a.repo)
    revs = heads(repo) if git(repo, "rev-parse", "--is-bare-repository").strip() == "true" else ["--remotes=origin"]
    recs = pointers(repo, revs)
    up, _ = selection(recs)
    try:
        have = lfs_batch([(o, recs[o]["size"]) for o in sorted(up)])
    except urllib.error.HTTPError as e:
        print("GitHub's LFS API:", e.code, e.read()[:200].decode(errors="replace"))
        return 2
    n = sum(1 for o in up if have.get(o))
    print(f"upload set on GitHub: {n} of {len(up)} objects")
    return 0 if n == len(up) else 1


def cmd_push(a):
    """Push every branch of the mirror with the LFS hook off, then upload the upload set by object id, then verify."""
    m = Path(a.mirror)
    recs = pointers(m, heads(m))
    up, _ = selection(recs)
    store = lfs_store(m)
    miss = [o for o in up if not obj_path(store, o).exists()]
    if miss and not a.allow_missing:
        print(f"{len(miss)} upload objects are missing in the mirror's store: run stage (and fetch them first), or "
              f"pass --allow-missing to push without them")
        return 1
    up = {o for o in up if o not in miss}
    print(f"branches: {len(heads(m))}; upload set present: {len(up)} objects, "
          f"{sum(recs[o]['size'] for o in up) / 1e9:.2f} GB; missing and not uploaded: {len(miss)} objects, "
          f"{sum(recs[o]['size'] for o in miss) / 1e9:.2f} GB")
    if a.dry_run:
        return 0
    env = dict(os.environ, GIT_LFS_SKIP_PUSH="1")
    subprocess.run(["git", "-C", str(m), "push", a.remote, "refs/heads/*:refs/heads/*"], env=env, check=True)
    oids = sorted(up)
    for k in range(0, len(oids), 200):
        subprocess.run(["git", "-C", str(m), "lfs", "push", "--object-id", a.remote, *oids[k:k + 200]], check=True)
        print(f"uploaded {min(k + 200, len(oids))} of {len(oids)}", flush=True)
    have = lfs_batch([(o, recs[o]["size"]) for o in oids])
    n = sum(1 for o in oids if have.get(o))
    print(f"uploaded objects on GitHub: {n} of {len(oids)}")
    return 0 if n == len(oids) else 1


HOOK = """#!/bin/sh
# Filtered LFS upload (scripts/docs/lfs_lean.py, owner 2026-11-06): only the doc pages, their media and the simulator's
# meshes go to GitHub's LFS; CAD and print files, raw recordings and data rows stay local. Replaces git-lfs's own hook.
command -v git-lfs >/dev/null 2>&1 || { echo >&2 "git-lfs is not installed"; exit 2; }
exec python3 "$(git rev-parse --show-toplevel)/scripts/docs/lfs_lean.py" pre-push "$@"
"""


PRE_COMMIT = """#!/bin/sh
# Data files (json, jsonl, csv, tsv, npz, npy, ...) are ordinary git files up to lfslean.maxplain bytes (default 2 MB);
# a larger one stays out of git and moves by rsync (scripts/docs/lfs_lean.py pre-commit, owner 2026-11-09).
exec python3 "$(git rev-parse --show-toplevel)/scripts/docs/lfs_lean.py" pre-commit
"""
DATA_EXT = {".json", ".jsonl", ".csv", ".tsv", ".npz", ".npy", ".pkl", ".h5", ".hdf5", ".parquet"}
MAX_PLAIN = 2_000_000


def cmd_install_hook(a):
    for name, text in (("pre-push", HOOK), ("pre-commit", PRE_COMMIT)):
        hook = Path(git(ROOT, "rev-parse", "--git-path", f"hooks/{name}").strip())
        hook = hook if hook.is_absolute() else ROOT / hook
        old = hook.read_text() if hook.exists() else ""
        saved = bool(old) and "lfs_lean.py" not in old
        if saved:
            hook.with_suffix(".lfs-default").write_text(old)
        hook.write_text(text)
        hook.chmod(0o755)
        print(f"installed {hook}" + (f" (the previous hook saved as {hook.with_suffix('.lfs-default')})" if saved else ""))
    return 0


def cmd_pre_commit(a):
    """The data rule (owner, 2026-11-09: "use LFS for only truly enduring, necessary large artifacts that cannot be normal
    git objects"; raw data "can live offline for now"): a data file staged as an ordinary git object may not exceed
    lfslean.maxplain bytes. Larger ones stay out of git, listed in .gitignore, and move by rsync."""
    cfg = subprocess.run(["git", "-C", str(ROOT), "config", "--get", "lfslean.maxplain"], capture_output=True, text=True)
    limit = int(cfg.stdout.strip() or MAX_PLAIN)
    paths = [p for p in git(ROOT, "diff", "--cached", "--name-only", "-z", "--diff-filter=AM").split("\0")
             if p and os.path.splitext(p)[1].lower() in DATA_EXT]
    if not paths:
        return 0
    attrs = git(ROOT, "check-attr", "filter", "--stdin", inp="\n".join(paths) + "\n")
    lfs = {ln.split(": filter: ")[0] for ln in attrs.splitlines() if ln.endswith(": filter: lfs")}
    plain = [p for p in paths if p not in lfs]
    if not plain:
        return 0
    sizes = git(ROOT, "cat-file", "--batch-check=%(objectsize)", inp="".join(f":{p}\n" for p in plain)).split()
    big = [(int(s), p) for s, p in zip(sizes, plain) if s.isdigit() and int(s) > limit]
    if not big:
        return 0
    print(f"lfs_lean: {len(big)} data files over {limit / 1e6:g} MB are staged as ordinary git files. Large data stays out "
          "of git (owner, 2026-11-09): unstage each (git rm --cached FILE), list it in .gitignore and copy it by rsync. "
          "Commit a smaller summary instead.", file=sys.stderr)
    for s, p in sorted(big, reverse=True)[:30]:
        print(f"  {s / 1e6:8.2f} MB  {p}", file=sys.stderr)
    return 1


def cmd_pre_push(a):
    """The hook: for each ref being pushed, the LFS objects that the new commits name, filtered to the upload set. Off
    unless `git config lfslean.upload true` (owner, 2026-11-06: "we'll just stick to local git + rsync"): the push then
    sends the history and the pointers only."""
    on = subprocess.run(["git", "-C", str(ROOT), "config", "--get", "lfslean.upload"], capture_output=True,
                        text=True).stdout.strip() == "true"
    if not on:
        return 0
    oids, local = set(), {}
    zero = "0" * 40
    for line in sys.stdin.read().splitlines():
        parts = line.split()
        if len(parts) != 4 or parts[1] == zero:
            continue
        recs = pointers(ROOT, [parts[1], "--not", f"--remotes={a.remote}"])
        up, keep = selection(recs)
        oids |= up
        local.update({o: recs[o] for o in keep})
    if local:
        # GitHub refuses a push whose commits name LFS objects it does not have (GH008), so a commit with files kept
        # local cannot be pushed until they leave the commit or join the upload set. Stop here with their names
        # instead; never answer GH008 with `git lfs push --all`, which uploads every recording and data row here.
        try:
            have = lfs_batch([(o, r["size"]) for o, r in local.items()])
        except Exception:                                # offline or refused: name them all
            have = {}
        miss = {o: r for o, r in local.items() if not have.get(o)}
        if miss:
            print(f"lfs_lean: {len(miss)} LFS files in these commits are kept local (CAD and print files, data rows, "
                  f"recordings; {sum(r['size'] for r in miss.values()) / 1e6:.1f} MB), and GitHub refuses pointers "
                  "without content (GH008). Take them out of the commits, or ask the owner to widen the upload set "
                  "(scripts/docs/lfs_lean.py upload). Never run `git lfs push --all`.", file=sys.stderr)
            for o, r in sorted(miss.items(), key=lambda x: sorted(x[1]["paths"])[0])[:30]:
                print(f"  {r['size'] / 1e6:8.2f} MB  {sorted(r['paths'])[0]}", file=sys.stderr)
            if len(miss) > 30:
                print(f"  ... and {len(miss) - 30} more", file=sys.stderr)
            return 1
    if not oids:
        return 0
    store = lfs_store(ROOT)
    miss = [o for o in oids if not obj_path(store, o).exists()]
    if miss:
        print(f"lfs_lean: {len(miss)} objects of the upload set are missing locally and are not uploaded",
              file=sys.stderr)
    ol = sorted(o for o in oids if o not in miss)
    if os.environ.get("LFS_LEAN_DRY"):                  # a test: name the objects instead of uploading them
        print(f"lfs_lean: would upload {len(ol)} objects", file=sys.stderr)
        return 0
    for k in range(0, len(ol), 200):
        r = subprocess.run(["git", "-C", str(ROOT), "lfs", "push", "--object-id", a.remote, *ol[k:k + 200]])
        if r.returncode:
            return r.returncode
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sp = p.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("catalog")
    c.add_argument("--repo", default=str(ROOT))
    c.add_argument("--refs", nargs="*")
    c.add_argument("--store-from", help="read object presence from this repository's LFS store")
    for name in ("refresh", "stage", "push"):
        s = sp.add_parser(name)
        s.add_argument("--mirror", required=True)
        s.add_argument("--source", default=str(ROOT))
        s.add_argument("--remote", default=REMOTE_URL)
        s.add_argument("--dry-run", action="store_true")
        s.add_argument("--allow-missing", action="store_true", help="push without the upload objects missing here")
    lo = sp.add_parser("lost")
    lo.add_argument("--repo", default=str(ROOT))
    lo.add_argument("--out", required=True)
    v = sp.add_parser("verify")
    v.add_argument("--repo", default=str(ROOT))
    sp.add_parser("install-hook")
    sp.add_parser("pre-commit")
    h = sp.add_parser("pre-push")
    h.add_argument("remote")
    h.add_argument("url", nargs="?")
    a = p.parse_args()
    return {"catalog": cmd_catalog, "refresh": cmd_refresh, "stage": cmd_stage, "push": cmd_push, "verify": cmd_verify,
            "lost": cmd_lost,
            "install-hook": cmd_install_hook, "pre-push": cmd_pre_push, "pre-commit": cmd_pre_commit}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
