"""Commit plain files to BRANCH and push it once, without touching the working tree, the index or main. GitHub refuses
LFS transfers from 21:37 on 11-05 (the repository's LFS budget), so this refuses any path that .gitattributes makes an
LFS file (under docs/: json, jsonl, csv, npz, bin, images, html pages). The commit's parent is the branch's tip on
origin, or BASE when the branch does not exist there yet. One push: a failed push is reported, not retried.

    /usr/bin/python3 push_text.py --branch NAME --base REV --message TEXT LOCAL=REPO_PATH ...
"""
import argparse
import os
import subprocess
import sys

REPO = "/home/will/workspace/hand"
ap = argparse.ArgumentParser()
ap.add_argument("--branch", required=True)
ap.add_argument("--base", required=True)
ap.add_argument("--message", required=True)
ap.add_argument("files", nargs="+")
a = ap.parse_args()


def git(*args, data=None, env=None, check=True):
    r = subprocess.run(["git", *args], cwd=REPO, input=data, capture_output=True, env=env)
    if check and r.returncode:
        raise SystemExit(f"git {' '.join(args[:2])} failed: {r.stderr.decode().strip()}")
    return r.stdout.decode().strip()


pairs = [f.split("=", 1) for f in a.files]
for local, path in pairs:
    if git("check-attr", "filter", "--", path).endswith(": lfs"):
        raise SystemExit(f"{path} is an LFS path; nothing committed")
tip = git("ls-remote", "origin", f"refs/heads/{a.branch}").split("\t")[0]
parent = tip or git("rev-parse", "--verify", a.base)
if tip and git("cat-file", "-t", tip, check=False) != "commit":
    git("fetch", "origin", a.branch)
env = dict(os.environ, GIT_INDEX_FILE=os.path.join(REPO, ".git", "push_text.idx"))
git("read-tree", parent, env=env)
for local, path in pairs:
    blob = git("hash-object", "-w", "--stdin", data=open(local, "rb").read())
    git("update-index", "--add", "--cacheinfo", f"100644,{blob},{path}", env=env)
tree = git("write-tree", env=env)
if tree == git("rev-parse", f"{parent}^{{tree}}"):
    print(f"nothing new against {parent[:9]}")
    raise SystemExit(0)
commit = git("commit-tree", tree, "-p", parent, "-m", a.message)
r = subprocess.run(["git", "push", "origin", f"{commit}:refs/heads/{a.branch}"], cwd=REPO, capture_output=True)
out = (r.stdout + r.stderr).decode().strip().replace("\n", " | ")
print(f"{'pushed' if r.returncode == 0 else 'PUSH FAILED'} {commit[:9]} on {parent[:9]} to {a.branch}: {out}")
sys.exit(r.returncode)
