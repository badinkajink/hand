#!/usr/bin/env bash
# Cherry-pick the migration commits of the local branch scripts-migrate onto the current branch, after the real switch, one
# commit each with its own message. The geometry and numerics modules and the docs section went to main before the switch
# (11-04), so HEAD's copies of those files stay: a chain session may have added a function to them.
#   bash docs/notes/20261103-scripts_cleanup/land_migration.sh [FIRST_COMMIT] [LAST_COMMIT]
# It stops at the first conflict outside the modules with the conflict in the working tree. Resolve it, run
# `git add <files> && git commit -C <commit>`, and start again from the next commit. RUNBOOK, "Landing the migration".
set -u
ALL="83dd092a1 c5007f002 a8a58bec7 dfae7348c 83255b038 764599ffe 0a16cb191 94a59c549 e9dd6f5ea 066a34c47 7d784b9b4 71be5fc56
     72c82f2e3 0bb15c5d2 a44e44e15 bf65cce89 c0463cf93 5cb62e1ba 8a8238aa9 d5198b948 f719012cd 92d136d5e 517cd84f5"
KEEP=(src/sr2hand/geometry/__init__.py src/sr2hand/geometry/rotations.py src/sr2hand/geometry/vectors.py
      src/sr2hand/geometry/distances.py src/sr2hand/geometry/sdf.py src/sr2hand/geometry/mesh.py
      src/sr2hand/geometry/sampling.py src/sr2hand/numerics)
FIRST=${1:-83dd092a1}
LAST=${2:-517cd84f5}
go=0
for c in $ALL; do
  [ "$c" = "$FIRST" ] && go=1
  [ $go = 1 ] || continue
  git cherry-pick -n "$c" >/dev/null 2>&1
  git checkout HEAD -- "${KEEP[@]}"
  left=$(git diff --name-only --diff-filter=U)
  if [ -n "$left" ]; then
    echo "conflict at $c ($(git log -1 --format=%s "$c" | cut -c1-70)):"
    echo "$left"
    echo "resolve, then: git add <files> && git commit -C $c, and rerun from the commit after it"
    exit 1
  fi
  if git diff --cached --quiet; then
    echo "empty    $c"
  else
    git commit -q -C "$c" && echo "landed   $c as $(git log -1 --format=%h)"
  fi
  [ "$c" = "$LAST" ] && break
done
