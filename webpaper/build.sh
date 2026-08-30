#!/usr/bin/env bash
# Build the MorphoHand project site (Typst HTML export -> webpaper/build/).
#
# Typst notes baked in:
#   --features html : HTML export is gated behind this flag in 0.14.x
#   --root <repo>   : path resolution + snap confinement need files under $HOME
#   math is rendered to inline SVG by the show rule in template.typ
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$REPO/webpaper/src"
OUT="$REPO/webpaper/build"
PAGES=(index morphology rl hardware control)

mkdir -p "$OUT/assets"

# Curated media: copy whatever the pages reference from src/assets/.
if compgen -G "$SRC/assets/*" > /dev/null; then
  cp -u "$SRC"/assets/* "$OUT/assets/"
fi

# Experiment artifacts. The self-contained HTML reports and the render sets produced
# during the sim2real build-up live under docs/experiments/<date>-<name>/ and are the
# primary record of those studies; the Control Station page links to them by name.
# Copied (not symlinked) so the built site can be served or archived on its own, and
# mirrored under the same dated folder names so a link in the page and a path in the
# repo are the same string.
mkdir -p "$OUT/artifacts"
for rel in \
  "20260827-real_v1/report.html" \
  "20260827-real_v1/report_mechanism.html" \
  "20260827-real_v1/report_methods.html" \
  "20260827-real_v1/SETUP.md" \
  "20260827-real_v1/MECHANISM.md" \
  "20260827-real_v1/SLIP.md" \
  "20260828-real_v1_search/report_search.html" \
  "20260828-real_v1_search/REPORT.md" \
  "20260828-real_v1_search/figs/landscape.png" \
  "20260828-real_v1_search/figs/scores.png" \
  "20260829-real_v1_deploy/DEPLOY.md" \
  "20260827-reorient_primitive/primitive_compare.png" \
  "20260827-linklen_renders/76x41.png" ; do
  src="$REPO/docs/experiments/$rel"
  if [ -f "$src" ]; then
    mkdir -p "$OUT/artifacts/$(dirname "$rel")"
    cp -u "$src" "$OUT/artifacts/$rel"
  else
    echo "  (missing artifact: $rel)"
  fi
done
if [ -d "$REPO/docs/experiments/20260828-real_v1_search/20260828-videos" ]; then
  mkdir -p "$OUT/artifacts/20260828-real_v1_search/20260828-videos"
  cp -u "$REPO"/docs/experiments/20260828-real_v1_search/20260828-videos/*.mp4 \
        "$OUT/artifacts/20260828-real_v1_search/20260828-videos/" 2>/dev/null || true
fi

for p in "${PAGES[@]}"; do
  echo "  typst  $p.typ -> build/$p.html"
  typst compile --features html --format html --root "$REPO" \
    "$SRC/$p.typ" "$OUT/$p.html" 2>&1 | grep -vE "warning: html export is under active|its behaviour may change|do not rely on this|see https://github" || true
done

echo "Done. Serve with:  python3 -m http.server -d $OUT 8080"
