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

1. Owner, 2026-11-06: no budget raise ("im ok with losing those 1.6gb ... it's worth not paying money + being able to
   download"). The 1,464 objects (1.64 GB) that exist only on GitHub are not downloaded, and the re-push goes without
   them: 469 versions of 84 pages, 632 data rows, 119 videos, 189 images, 53 CAD, pdf and zip files and 2 arrays,
   listed in `20261106-lfs_objects_only_on_github.tsv` beside this note. Their pointers stay in the history; GitHub
   then shows a pointer in place of their content.
2. Workstation: `git fetch origin` (the branches the lab pushed meanwhile), `refresh` the bare copy, `stage`, and
   `push --dry-run --allow-missing` (`--mirror /home/humanoid/Programs/hand-lean.git`): 8 branches, 4,412 upload
   objects present (3.39 GB), 830 missing and not uploaded (1.30 GB).
3. Owner: no session pushes during the switch (the lab on corvus pushes branches).
4. Owner: delete `badinkajink/hand` (Settings, Danger zone) and create an empty public repository `hand` under the same
   account, with no README, licence or .gitignore. The old one has no issues, pull requests, releases, forks, stars or
   wiki; its deploy keys and collaborators, if any, must be added again.
5. Workstation: `python3 scripts/docs/lfs_lean.py refresh --mirror /home/humanoid/Programs/hand-lean.git` (the local
   commits made meanwhile; GitHub's branches as fetched in step 2), `stage`, then `push --allow-missing`: the branches with the LFS upload off, then the upload set by object id, then `verify` against GitHub's
   LFS API. Then `git fetch origin` and `python3 scripts/docs/lfs_lean.py install-hook` in the main clone, so that later
   pushes upload only the upload set.
6. The budget stays at $0: 3.39 GB is under the included 10 GiB of storage, and downloads stay free up to 10 GiB a
   month.

New raw recordings are ignored by git (`docs/**/*.bin`) and move by rsync; the tracked ones stay tracked, so no clone
deletes them on a pull. The main clone and the bare copy contain the complete history and every LFS object except the
1,464 listed beside this note, so they are the backup during the switch.

## Outcome, 16:0x to 16:2x on 2026-11-06

The owner deleted the repository and created the empty public `badinkajink/hand` at 16:02. The history went up from the
bare copy: all 8 branches with the same commit hashes, `main` at 994c45258 with the ten commits that had waited. The LFS
upload was refused with the same message, "This repository exceeded its LFS budget": the budget is the account's, and
the old repository's storage during this billing cycle already exceeded it. Deleting the repository stops further
storage but leaves the cycle's usage. No LFS object is on GitHub yet; on GitHub the pages and media show their
pointers, and everything is present here.

Remaining step, once the budget allows uploads (a small budget now, or the next billing cycle):

    python3 scripts/docs/lfs_lean.py push --mirror /home/humanoid/Programs/hand-lean.git --allow-missing --remote github

The git push in it changes nothing; the upload sends the 4,412 objects (3.39 GB) and checks them. Until then a push of
commits that add pages, media or CAD files fails in the pre-push hook, which is installed in the main clone; pushes of
code and notes go through.

## Decision, 16:1x on 2026-11-06

Owner: "ok we'll just stick to local git + rsync". No LFS content goes to GitHub: the hook uploads nothing unless
`git config lfslean.upload true`, and git pushes send the history and the pointers. Pages, media, data rows and
recordings move by rsync.

## Upload with a budget, 20:1x on 2026-11-06

The owner set an LFS budget on the account in the evening and asked to leave the CAD files out: "remove the cad files,
particularly the 3mf files (besides the most recent fingertip w 4 magnets). i dont think we need costly 3mf plating for
anything". The upload set of commit 11d132fb9 is the pages, figures, films and PDFs under `docs/` and the meshes (stl,
obj) under `assets/`, which the MuJoCo scenes load. STEP, 3mf and zipped CAD exports (0.51 GB) stay local, except the
print file of the four-magnet pad 05R4 that the v3 tips were printed from
(`assets/cad_v2/fingertip_prints/pads_05R4_four_magnets_x3_xl.3mf`; the same file is in the DIP-link folder).
`push --allow-missing` uploaded 4,135 objects, 2.88 GB, and GitHub's LFS API reports all 4,135; the other 794 objects
of the set (1.27 GB) are among the lost ones. The hook is on again (`git config lfslean.upload true`).

A clone elsewhere fetches the pages from GitHub after `git config lfs.fetchexclude ''` and
`git config lfs.fetchinclude 'docs/**/*.html'`: `git lfs pull` fetches the 120 pages of the current tree, and every
later checkout fetches the pages it changes. On a clone with an empty LFS store the two HOM pages arrived in full
(5,136,148 and 5,060,065 bytes) and `git status` stayed clean.
