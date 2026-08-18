#!/usr/bin/env bash
# Rebuild the standalone review page. Assets are inlined as data URIs because the artifact host
# blocks every external request -- a linked video or font would silently fail to load.
# Filmstrips and eval plots are downscaled to JPEG first; the raw PNGs are ~5 MB each.
set -euo pipefail
cd "$(dirname "$0")/../../.."
V=docs/rl/videos/20260817_perp_review
OUT=docs/experiments/20260818-perp_review_page
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

uv run python - "$TMP" "$V" <<'PY'
from PIL import Image; import sys, os
tmp, V = sys.argv[1], sys.argv[2]
for f, w, q in [("r4_filmstrip.png",1600,82),("r4_excl_eval.png",900,85),
                ("perp_r6_slip-1000_eval.png",900,85),("perp_r6_slip-3000_eval.png",900,85)]:
    im = Image.open(os.path.join(V,f)).convert("RGB")
    im = im.resize((w,int(im.height*w/im.width)), Image.LANCZOS)
    im.save(os.path.join(tmp,f.replace(".png",".jpg")),"JPEG",quality=q,optimize=True)
PY

uv run python - "$TMP" "$V" "$OUT" <<'PY'
import base64, sys, os
tmp, V, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
assets = {"r4_video":(os.path.join(V,"r4_600steps.mp4"),"video/mp4"),
          "r4_strip":(os.path.join(tmp,"r4_filmstrip.jpg"),"image/jpeg"),
          "r4_eval":(os.path.join(tmp,"r4_excl_eval.jpg"),"image/jpeg"),
          "r6a_eval":(os.path.join(tmp,"perp_r6_slip-1000_eval.jpg"),"image/jpeg"),
          "r6b_eval":(os.path.join(tmp,"perp_r6_slip-3000_eval.jpg"),"image/jpeg")}
html = open(os.path.join(OUT,"page.src.html")).read()
for k,(p,mime) in assets.items():
    html = html.replace("{{ASSET:%s}}"%k,
                        "data:%s;base64,%s"%(mime, base64.b64encode(open(p,"rb").read()).decode()))
assert "{{ASSET" not in html, "unreplaced asset token"
dst = os.path.join(OUT,"perp_reorientation.html"); open(dst,"w").write(html)
print("wrote %s  %.2f MB" % (dst, os.path.getsize(dst)/1048576))
PY
