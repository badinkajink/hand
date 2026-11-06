# Re-pushing the repository with LFS for the documentation only (2026-11-06)

Owner, 2026-11-06: "set up a clone of this repo that doesnt waste our LFS on the robot data--LFS should really only be
for the docpage videos/content, and let me know when its ready for me to delete the current repo and repush. i would
like to keep the commits, links, hyperlinks intact".

GitHub refused every push and download with LFS objects from the evening of 11-05: "This repository exceeded its LFS
budget" (the account's budget is $0 and its included storage 10 GiB). The repository's refs name 16,331 LFS objects,
32.5 GB: raw robot recordings (`*.bin`) 20.4 GB, data rows (json, jsonl, csv) 7.1 GB, pages 2.7 GB, videos 0.8 GB,
images 0.6 GB, CAD, pdf and zip 0.6 GB, arrays 0.3 GB. GitHub frees LFS storage only when the repository is deleted.
The archive `badinkajink/hand-archive` has no LFS objects.

The history is pushed unchanged, so every commit hash and every link to a commit or a page version stays valid. Only
the upload set of `scripts/docs/lfs_lean.py` goes to GitHub's LFS: 5,242 objects, 4.69 GB (4.36 GiB), the pages, their
media and the CAD files. The 27.8 GB of recordings and data rows stay on the workstation; on GitHub their pointers
remain in the history without content.

The clone to push is `/home/humanoid/Programs/hand-lean.git`, a bare copy with GitHub's eight branches, `main` with the
commits that could not be pushed, the remote `github` and an LFS store of hard links to the upload set.

## Steps

1. Owner: raise the account's Git LFS budget above $0 for now (GitHub Settings, Billing and licensing, Budgets).
   1,464 objects (1.64 GB) exist only on GitHub: 469 old page versions, 632 data rows, 119 videos, 189 images and
   others. Downloads are refused (HTTP 403) until the budget allows them, and deleting the repository before they are
   downloaded loses them.
2. Workstation: `git fetch origin` (the branches the lab pushed meanwhile) and `git lfs fetch --all origin` in
   `/home/humanoid/Programs/hand`, then `refresh` the bare copy; then
   `python3 scripts/docs/lfs_lean.py catalog` must report 0 objects missing here, and `stage` then `push --dry-run`
   (`--mirror /home/humanoid/Programs/hand-lean.git`) must report no missing upload objects.
3. Owner: no session pushes during the switch (the lab on corvus pushes branches).
4. Owner: delete `badinkajink/hand` (Settings, Danger zone) and create an empty public repository `hand` under the same
   account, with no README, licence or .gitignore. The old one has no issues, pull requests, releases, forks, stars or
   wiki; its deploy keys and collaborators, if any, must be added again.
5. Workstation: `python3 scripts/docs/lfs_lean.py refresh --mirror /home/humanoid/Programs/hand-lean.git` (the local
   commits made meanwhile; GitHub's branches as fetched in step 2), `stage`, then `push`: the branches with the LFS upload off, then the upload set by object id, then `verify` against GitHub's
   LFS API. Then `git fetch origin` and `python3 scripts/docs/lfs_lean.py install-hook` in the main clone, so that later
   pushes upload only the upload set.
6. Owner: the budget can return to $0, since 4.36 GiB is under the included 10 GiB.

New raw recordings are ignored by git (`docs/**/*.bin`) and move by rsync; the tracked ones stay tracked, so no clone
deletes them on a pull. The main clone and the bare copy contain the complete history and every object, so they are the
backup during the switch.
