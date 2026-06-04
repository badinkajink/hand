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
PAGES=(index morphology rl hardware)

mkdir -p "$OUT/assets"

# Curated media: copy whatever the pages reference from src/assets/.
if compgen -G "$SRC/assets/*" > /dev/null; then
  cp -u "$SRC"/assets/* "$OUT/assets/"
fi

for p in "${PAGES[@]}"; do
  echo "  typst  $p.typ -> build/$p.html"
  typst compile --features html --format html --root "$REPO" \
    "$SRC/$p.typ" "$OUT/$p.html" 2>&1 | grep -vE "warning: html export is under active|its behaviour may change|do not rely on this|see https://github" || true
done

echo "Done. Serve with:  python3 -m http.server -d $OUT 8080"
