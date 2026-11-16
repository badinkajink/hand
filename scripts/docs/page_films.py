#!/usr/bin/env python3
"""Films of the result pages as files beside them (owner, 2026-11-15: "we can switch to linking films vs. embeddings").

    link    pages whose films are inlined as data URIs get links to files in their folder's media/ (retro_style.link_films;
            page builders do this by themselves at exit)
    files   a page's linked films as the JSON map an artifact publish takes as `files`: {"media/x.mp4": "docs/.../x.mp4"}

    python3 scripts/docs/page_films.py link PAGE.html [...]
    python3 scripts/docs/page_films.py files PAGE.html
"""
from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import retro_style  # noqa: E402


def files(page):
    html = open(page, encoding="utf-8").read()
    rels = sorted(set(re.findall(r'<(?:video|source)[^>]*\ssrc="(media/[^"?#]+)"', html)))
    folder = os.path.relpath(os.path.dirname(os.path.abspath(page)))
    return {r: os.path.join(folder, r) for r in rels}


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ("link", "files"):
        print(__doc__)
        return 2
    if sys.argv[1] == "link":
        for page in sys.argv[2:]:
            k, w = retro_style.link_films(page)
            print(f"{page}: {k} films linked, {w} written to media/")
    else:
        print(json.dumps(files(sys.argv[2]), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
