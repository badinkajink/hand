#!/usr/bin/env python3
"""Build docs/index.html, the home page of the documentation: the hand in two sentences, the topics with their
state and links, how to fetch the result pages, and where the code is; and docs/pages.html, the index of every result
page by topic and by date (owner, 2026-10-13: "is the home view index exhaustive ... organize so that a non-LFS and LFS
user can view the index"), with each page's local file, size, builder, copy on claude.ai and the command that fetches it.

    python3 scripts/docs_home_page.py

Both pages are ordinary git files that open from a clone in a browser with no server (owner, 2026-10-09): relative
links, the one photo inlined as a data URI. Each topic heading links its overview (docs/overviews/, file names in
docs_inventory.OVERVIEW) once that file exists; the key result pages are linked under it. The page index lists every
page of docs_inventory.records(), so a page with a topic is never missing from it.

The page index is a folder tree (owner, 2026-10-18: "a branching structure would be more helpful, like a typical file
explorer set up"): docs/experiments/ opens to the topics, each topic to its dated study folders, newest first, and each
study folder to its result pages and a collapsed folder of its other tracked files, listed to two levels with at most
CAP files per folder (print files always). Folders without a result page and the files at the top of docs/experiments/
follow the topics, and a second tree lists the pages by month. The tree is nested <details>, so it works without
JavaScript; a short script adds the expand-all and collapse-all buttons and opens the folder a #fragment names.

A page is rewritten only when its text apart from the "updated" date changes, so a rebuild that changes nothing leaves
both files as they are in git.
"""
from __future__ import annotations

import base64
import collections
import html as htmllib
import os
import re
import subprocess
import time
from urllib.parse import quote

import docs_inventory as D
import retro_style
import texsvg

ROOT = D.ROOT
OUT = os.path.join(ROOT, "docs", "index.html")
PAGES_OUT = os.path.join(ROOT, "docs", "pages.html")
PHOTO = os.path.join(ROOT, "docs", "overviews", "media", "sr2_hand_bench.jpg")   # paper Fig. 1, 720 px
TEX_CACHE = os.path.join(ROOT, "docs", "overviews", "media", "texsvg_cache.json")   # shared with the overviews
TEX_RE = re.compile(r"\\\((.+?)\\\)", re.S)

E = "docs/experiments/"
# topic key -> (paragraph, key pages as (label, path relative to the repository)); the order is the reading order
TOPICS = [
    ("design",
     "Of 8,198 sampled finger layouts, 227 passed the simulated screens and eight (D1&#8211;D8) were built. They "
     "held the tool in 48 of 77 bench trials; over the eight hands, the Spearman correlation of the simulated and "
     "measured final tool angle is \\(\\rho_s = 0.50\\). In simulation, the layout with the largest workspace "
     "for a cylinder or sphere puts the finger mounts 35&#8211;37&#8202;mm outside its surface.",
     [("design search", E + "20260828-real_v1_search/20260828-real_v1_design_search.html"),
      ("finger spacing and object size", E + "20261008-hand_object_scale/20261008-finger_spacing_object_size.html"),
      ]),
    ("hardware",
     "Each finger joint is an SCS0009 servo that lags its command by 0.0186&#176; per unit of load and cuts out at "
     "a protective torque; under a grasp the yaw joints reach 0.44&#8211;0.90 of the command. The servo model "
     "reproduces 177 tracked bench runs within 2.89&#176;, and two AprilTags give the tool's pose to 0.017&#176;.",
     [("servo identification", E + "20260902-servo-sysid/20260902-servo_sysid.html"),
      ("servo model refit", E + "20261006-hom_turn3/20261006-servo_refit_hom_turn_pad_cost.html"),
      ("identification protocol, draft", E + "20261013-servo_identification/20261013-servo_identification_protocol.html"),
      ("finger model from the CAD", E + "20261013-cad_v1_finger/20261013-cad_v1_finger_model.html"),
      ("tool tracker", E + "20260831-real_v1-object-tracking/20260831-real_v1_object_tracking.html")]),
    ("mechanisms",
     "Rotating the three contacts as one rigid body about an axis above them stands the screwdriver up open loop, "
     "to about 8&#176; from vertical in simulation; the paper's bench results use this plan. On the servo model "
     "fitted to the bench it turns the tool 27&#8211;57&#176;. Gaits, a gravity swing and hand-object control are "
     "the alternatives studied.",
     [("fixed-contact turn", E + "20260827-real_v1/20260827-real_v1_rotational_lock.html"),
      ("eight deployed hands", E + "20260908-reorientation_journey/20260908-reorientation_journey.html"),
      ("hand-object control", E + "20261006-hom_turn3/20261006-servo_refit_hom_turn_pad_cost.html")]),
    ("policies",
     "PPO policies trained in MuJoCo-Warp add a learned correction to a scripted lift and turn; none has run on the "
     "hand yet. D6 policies trained with sphere-pad fingertips keep the tool in 216 of 220 replays of their finger "
     "commands in three other simulators, and those trained with point contact in 3 of 132.",
     [("four contact models", E + "20261008-contact_model_policies/20261008-fingertip_contact_model_policies.html"),
      ("policies across the hands", E + "20260919-hands_tranche/20260919-reorient_policies_across_hands.html")]),
    ("chain",
     "In simulation the whole task (grasp, lift, learned turn, set-down in a countersink, re-grip and a gait that "
     "turns the screw) runs as one rollout with the hand on a UR5e arm; 12 of 20 rollouts on D5 and D6 completed, "
     "on a servo model since replaced. The hand-over from the turn to the gait loses the most rollouts. The bench "
     "has run only the grasp and the turn.",
     [("seam and countersink", E + "20260903-real_v1_chain/20260903-real_v1_chain_and_countersink.html"),
      ("chain with the learned turn", E + "20260923-chain_handover_gait/20260923-chain_handover_gait.html")]),
    ("contact",
     "Covering each fingertip with 1&#8202;mm MuJoCo contact spheres reproduces the friction torque of a soft "
     "fingertip: on a two-pad pinch the pads slip within 0.1&#8202;% of the force of Drake's hydroelastic model and "
     "spin 6&#8202;% under its torque, at 15&#8202;&#181;s per step against Drake's 1,298&#8202;&#181;s.",
     [("policies on four contact models", E + "20261008-contact_model_policies/20261008-fingertip_contact_model_policies.html"),
      ("comparison bed", E + "20261005-contact_bed/20261005-contact_model_bed.html")]),
    ("fingertip",
     "Finite elements on its own G-code put the approach of a 95A TPU tip with 20&#8202;% gyroid infill at "
     "63&#8202;&#181;m under 1&#8202;N, and that of an eFlesh lattice of 8&#8202;mm cells at 381&#8211;613&#8202;&#181;m. "
     "On the five-chip MLX90393 board, one 4.76&#8202;&#215;&#8202;1.59&#8202;mm magnet 5&#8202;mm under the face gives "
     "0.044&#8202;N of force noise per sample at a known contact point; with the contact point unknown, four magnets hold "
     "0.14&#8211;0.16&#8202;N over the whole face, where one magnet ranges from 0.08 to 1.48&#8202;N.",
     [("finite elements of the printed tip", E + "20261009-tpu_tip_fem/20261009-printed_tpu_fingertip_finite_elements.html"),
      ("eFlesh printability", E + "20261013-eflesh_printability/20261013-eflesh_lattice_printability.html"),
      ("magnetometer readout", E + "20261013-magnet_layouts/20261013-mlx90393_fingertip_readout.html")]),
]

CSS = """
body{margin:0}
.wrap{max-width:860px;margin:0 auto;padding:24px 16px 48px}
figure{margin:16px 0}
figure img{max-width:100%;height:auto}
.topic p{margin:4px 0 6px}
.pages{font-size:16px}
pre{overflow-x:auto}
td,th{text-align:left}
.tw{overflow-x:auto}
table.pix{width:100%;font-size:16px}
table.pix td.num{white-space:nowrap}
table.pix td.mb{text-align:right;white-space:nowrap}
details{margin:4px 0 8px}
details pre{margin:4px 0}
"""


def photo_uri():
    return "data:image/jpeg;base64," + base64.b64encode(open(PHOTO, "rb").read()).decode()


def rel(path):
    return os.path.relpath(os.path.join(ROOT, path), os.path.dirname(OUT))


def topic_html(key, text, pages, titles):
    ov = os.path.join(ROOT, "docs", "overviews", D.OVERVIEW[key])
    head = titles[key]
    if os.path.exists(ov):
        head = f'<a href="{rel(os.path.relpath(ov, ROOT))}">{head}</a>'
    else:                                  # no overview yet: the topic's rows in the page index
        head = f'<a href="pages.html#{key}">{head}</a>'
    links = " &#183; ".join(f'<a href="{rel(p)}">{lab}</a>' for lab, p in pages)
    return (f'<div class="topic"><h3>{head}</h3><p>{text}</p>'
            f'<p class="pages">Pages: {links}.</p></div>')


def lfs_cmd(patterns):
    return 'git -c lfs.fetchexclude= lfs pull --include="' + ",".join(patterns) + '"'


# ---- docs/pages.html: the page index as a folder tree -------------------------------------------------------------
CAP = 12                    # files listed per folder in a study's other files, print files not counted
SUBCAP = 24                 # subfolders listed per folder
SKIP_FILES = re.compile(r"texsvg_cache.*\.json|\.gitignore|artifact_url\.txt")   # build caches and the copy's URL
KINDS = [   # key, glyph, singular, plural, extensions; the order of the per-folder counts
    ("print", "pr", "print file", "print files", {".3mf", ".stl", ".step", ".stp", ".gcode", ".bgcode"}),
    ("data", "da", "data file", "data files", {".json", ".jsonl", ".csv", ".tsv", ".npz", ".npy", ".gz", ".b64",
                                               ".pkl", ".h5"}),
    ("film", "fi", "film", "films", {".mp4", ".webm", ".mov", ".gif"}),
    ("image", "im", "image", "images", {".png", ".jpg", ".jpeg", ".svg", ".webp"}),
    ("text", "tx", "text file", "text files", {".md", ".txt", ".pdf", ".docx", ".log", ".tex"}),
    ("web", "pg", "HTML file", "HTML files", {".html", ".htm"}),
    ("scene", "co", "scene file", "scene files", {".xml", ".urdf", ".obj", ".mjb"}),
    ("code", "co", "script", "scripts", {".py", ".sh", ".yaml", ".yml", ".toml", ".ipynb", ".js", ".css"}),
    ("other", "ot", "other file", "other files", set()),
]
KIND_OF = {e: k for k, _, _, _, exts in KINDS for e in exts}
GLYPH = {k: g for k, g, *_ in KINDS}
NOUN = {k: (s, p) for k, _, s, p, _ in KINDS}

_SHEET = "<path d='M3.5 1.5h6l3 3v10h-9z' fill='#fff' stroke='#000'/><path d='M9.5 1.5v3h3' fill='none' stroke='#000'/>"
ICONS = {   # 16 px glyphs in the manner of a 1990s file manager; the twisty boxes are 9 px
    "fc": "<path d='M1.5 3.5h4.5l1.5 1.5h7v8.5h-13z' fill='#f2cf5b' stroke='#000'/><path d='M2 6.5h12' stroke='#a8862a'/>",
    "fo": "<path d='M1.5 13.5v-10h4.5l1.5 1.5h6v2' fill='#d9b041' stroke='#000'/>"
          "<path d='M1.5 13.5l2.5-6.5h11l-2.5 6.5z' fill='#f8e08e' stroke='#000' stroke-linejoin='round'/>",
    "pg": _SHEET + "<path d='M5 7.5h6M5 9.5h6M5 11.5h4' stroke='#0000ee'/>",
    "tx": _SHEET + "<path d='M5 7.5h6M5 9.5h6M5 11.5h4' stroke='#666'/>",
    "da": _SHEET + "<path d='M5.5 6.5h5v6h-5z' fill='#e6e6e6' stroke='#000'/><path d='M5.5 8.5h5M5.5 10.5h5M8 6.5v6' stroke='#000'/>",
    "fi": "<rect x='1.5' y='2.5' width='13' height='11' fill='#222' stroke='#000'/><rect x='4' y='5' width='8' height='6' fill='#8fb3e8'/>"
          "<path d='M2.5 3.5h1M5.5 3.5h1M8.5 3.5h1M11.5 3.5h1M2.5 12.5h1M5.5 12.5h1M8.5 12.5h1M11.5 12.5h1' stroke='#fff'/>",
    "im": "<rect x='1.5' y='2.5' width='13' height='11' fill='#fff' stroke='#000'/><circle cx='5' cy='6' r='1.5' fill='#e8a33a'/>"
          "<path d='M2 13l4-5 3 3 2-2 3.5 4z' fill='#5e8f5a'/>",
    "pr": "<path d='M8 1.5l6 3-6 3-6-3z' fill='#f6d7a7'/><path d='M2 4.5l6 3v7l-6-3z' fill='#be7514'/>"
          "<path d='M14 4.5l-6 3v7l6-3z' fill='#e8a54b'/>"
          "<path d='M8 1.5l6 3v7l-6 3-6-3v-7zM2 4.5l6 3 6-3M8 7.5v7' fill='none' stroke='#000' stroke-linejoin='round'/>",
    "co": _SHEET + "<path d='M7 7.5l-2 2 2 2M9 7.5l2 2-2 2' fill='none' stroke='#000'/>",
    "ot": _SHEET,
    "te": "<rect x='1.5' y='2.5' width='13' height='11' fill='#000' stroke='#000'/><path d='M4 6l2.5 2-2.5 2M8 10.5h4' fill='none' stroke='#fff'/>",
    "plus": "<rect x='.5' y='.5' width='8' height='8' fill='#fff' stroke='#808080'/><path d='M2 4.5h5M4.5 2v5' stroke='#000'/>",
    "minus": "<rect x='.5' y='.5' width='8' height='8' fill='#fff' stroke='#808080'/><path d='M2 4.5h5' stroke='#000'/>",
}


def icon_uri(key):
    n = 9 if key in ("plus", "minus") else 16
    svg = (f"<svg xmlns='http://www.w3.org/2000/svg' width='{n}' height='{n}' viewBox='0 0 {n} {n}' "
           f"shape-rendering='crispEdges'>{ICONS[key]}</svg>")
    return "url(data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode() + ")"


GUIDE = "1px dotted #808080"
TREE_CSS = (CSS + """
.tools{margin:10px 0 4px}
.tools button{font:15px "Times New Roman",Times,serif;margin-right:6px}
.tree,.tree ul{list-style:none;margin:0;padding:0}
.tree{margin:6px 0 0 !important}
.tree li{position:relative;margin:0 !important;padding:0 0 0 19px !important;font-size:16px !important;
  line-height:22px !important;overflow-wrap:anywhere}
.tree li.l{padding-left:52px !important}
.tree li.p{padding-left:31px !important}
.tree li::before{content:"";position:absolute;left:4px;top:0;bottom:0;border-left:GUIDE}
.tree li:last-child::before{bottom:auto;height:11px}
.tree li::after{content:"";position:absolute;left:4px;top:11px;width:14px;border-top:GUIDE}
.tree li.l::after,.tree li.p::after{width:26px}
.tree > li{padding-left:0 !important}
.tree > li::before,.tree > li::after{display:none}
.tree details{margin:0}
.tree summary{display:block;position:relative;padding-left:33px;cursor:pointer;list-style:none}
.tree summary::-webkit-details-marker{display:none}
.tree summary::before{content:"";position:absolute;left:0;top:7px;width:9px;height:9px;background:PLUS no-repeat}
.tree details[open] > summary::before{background-image:MINUS}
.tree details[open] > summary::after{content:"";position:absolute;left:4px;top:16px;bottom:0;border-left:GUIDE}
.tree i{position:absolute;left:12px;top:3px;width:16px;height:16px;background:no-repeat center/16px 16px}
.tree li.l > i{left:31px}
.tree small{color:#444;font-size:14px}
.tree small a{white-space:nowrap}
.tree .sj{display:block}
.tree .f,.tree .fn{font-family:"Courier New",Courier,monospace !important;font-size:15px}
.tree small .fn{font-size:14px}
.tree pre{margin:2px 0 6px !important;padding:4px 8px !important;font-size:14px !important;white-space:pre-wrap;
  overflow-wrap:anywhere}
@media (max-width:480px){.tree li{padding-left:15px !important}.tree li.l{padding-left:48px !important}
  .tree li.p{padding-left:27px !important}.tree li::after{width:10px}.tree li.l::after,.tree li.p::after{width:22px}
  .tree li.l > i{left:27px}.tree .f{font-size:14px;word-break:break-all}}
""").replace("GUIDE", GUIDE).replace("PLUS", icon_uri("plus")).replace("MINUS", icon_uri("minus")) + "".join(
    f".tree i.{k}{{background-image:{icon_uri(k)}}}\n" for k in ICONS if k not in ("plus", "minus")) + (
    ".tree details[open] > summary > i.fc{background-image:" + icon_uri("fo") + "}\n")

TREE_JS = """
(function () {
  var tools = document.getElementById('tools');
  var all = function () { return document.querySelectorAll('.tree details'); };
  if (tools) {
    tools.hidden = false;
    tools.addEventListener('click', function (e) {
      var act = e.target && e.target.getAttribute && e.target.getAttribute('data-act');
      if (!act) return;
      var ds = all();
      for (var i = 0; i < ds.length; i++) ds[i].open = act === 'open' || ds[i].classList.contains('root');
    });
  }
  function reveal() {
    var id = decodeURIComponent(location.hash.slice(1));
    var el = id && document.getElementById(id);
    if (!el) return;
    if (el.tagName === 'DETAILS') el.open = true;
    for (var p = el.parentElement; p; p = p.parentElement) if (p.tagName === 'DETAILS') p.open = true;
    el.scrollIntoView();
  }
  window.addEventListener('hashchange', reveal);
  reveal();
})();
"""


def esc(s):
    return htmllib.escape(htmllib.unescape(str(s)), quote=False)


def tracked_sizes():
    """{path relative to the repository: bytes on disk} of the files git tracks under docs/experiments (a clone has
    exactly these); every file on disk when git is unavailable."""
    try:
        out = subprocess.run(["git", "-C", ROOT, "ls-files", "-z", "--", "docs/experiments"], capture_output=True,
                             check=True).stdout.decode("utf-8", "replace")
        paths = [p for p in out.split("\0") if p]
    except (OSError, subprocess.CalledProcessError):
        paths = [os.path.relpath(os.path.join(d, f), ROOT) for d, _, fs in os.walk(D.EXP) for f in fs]
    sizes = {}
    for p in paths:
        try:
            sizes[p] = os.path.getsize(os.path.join(ROOT, p))
        except OSError:                    # staged, then deleted from the working tree
            pass
    return sizes


def kind(name):
    return KIND_OF.get(os.path.splitext(name)[1].lower(), "other")


def size_text(b):
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if b >= scale or unit == "kB":
            x = b / scale
            return f"{x:.0f}&#8201;{unit}" if x >= 100 or unit == "kB" and x >= 10 else f"{x:.1f}&#8201;{unit}"


def count_text(n, noun):
    return f"{n:,} {noun[0] if n == 1 else noun[1]}"


def kinds_text(counter):
    return ", ".join(count_text(counter[k], NOUN[k]) for k, *_ in KINDS if counter.get(k))


class Dir:
    """One folder of the tracked-file tree, with the file count, bytes and file kinds of everything below it."""

    def __init__(self, path):
        self.path, self.dirs, self.files = path, {}, {}
        self.n, self.bytes, self.kinds, self.has_print = 0, 0, collections.Counter(), False

    def add(self, rel, size):
        node = self
        for part in rel.split("/")[:-1]:
            node = node.dirs.setdefault(part, Dir(node.path + "/" + part))
        node.files[rel.rsplit("/", 1)[-1]] = size

    def total(self):
        for d in self.dirs.values():
            d.total()
            self.n, self.bytes, self.has_print = self.n + d.n, self.bytes + d.bytes, self.has_print or d.has_print
            self.kinds.update(d.kinds)
        for name, size in self.files.items():
            self.n, self.bytes = self.n + 1, self.bytes + size
            self.kinds[kind(name)] += 1
            self.has_print = self.has_print or kind(name) == "print"
        return self


def tree_of(sizes, folder, exclude=()):
    """Dir of docs/experiments/<folder>/ without the files in exclude and the build caches."""
    root, pre = Dir("docs/experiments/" + folder), "docs/experiments/" + folder + "/"
    for p, s in sizes.items():
        if p.startswith(pre) and p not in exclude and not SKIP_FILES.fullmatch(os.path.basename(p)):
            root.add(p[len(pre):], s)
    return root.total()


def leaf(glyph, body):
    return f'<li class="l"><i class="{glyph}"></i>{body}</li>'


def folder(glyph, body, children, is_open=False, ident=None, cls=None):
    attrs = (" open" if is_open else "") + (f' id="{ident}"' if ident else "") + (f' class="{cls}"' if cls else "")
    return f'<li><details{attrs}><summary><i class="{glyph}"></i>{body}</summary><ul>{"".join(children)}</ul></details></li>'


def meta(*parts):
    return " <small>" + " &#183; ".join(p for p in parts if p) + "</small>"


def pages_html(recs, titles):
    """docs/pages.html: a folder tree of docs/experiments/ by topic, then every result page by month."""
    order = [k for k, *_ in TOPICS] + [k for k, *_ in D.TOPICS if k not in {t[0] for t in TOPICS}]
    here = os.path.dirname(PAGES_OUT)
    total = sum(r["mb"] for r in recs)
    sizes = tracked_sizes()
    for r in recs:                         # a page on disk that git does not track yet still counts in its folder
        sizes.setdefault(r["file"], os.path.getsize(os.path.join(ROOT, r["file"])))
    page_files = {r["file"] for r in recs}
    overview_url = {m.group(1): m.group(2) for m in re.finditer(
        r"\(\.\./overviews/([\w.-]+\.html)\)[^\n]*?\((https://claude\.ai/(?:code/)?artifact/[\w-]+)\)",
        open(os.path.join(D.EXP, "INDEX.md"), encoding="utf-8").read())}

    def href(path):
        return htmllib.escape(quote(os.path.relpath(os.path.join(ROOT, path), here) + ("/" if path.endswith("/") else "")))

    def title(r):
        return esc(r["title"]) + (f" ({r['status']})" if r["status"] != "current" else "")

    def copy(url):
        return f'<a href="{htmllib.escape(url)}">copy on claude.ai</a>' if url else ""

    def page_leaf(r, with_title):
        mb = os.path.getsize(os.path.join(ROOT, r["file"])) / 1e6
        builder = f'built by <span class="fn">{esc(os.path.basename(r["builder"]))}</span>' if r["builder"] else ""
        body = (f'<a class="f" href="{href(r["file"])}">{esc(os.path.basename(r["file"]))}</a>'
                + meta(f"{mb:.2f}&#8201;MB", builder, copy(r["artifact"])))
        if with_title:
            body += f'<span class="sj">{title(r)}</span>'
        return leaf("pg", body)

    def listing(node, level):
        """<li> items of one folder's subfolders and files. Subfolders are listed to two levels below the dated
        folder; a deeper one, or any subfolder of a folder with more than SUBCAP of them, is one line that links the
        folder, unless it includes print files, and at most SUBCAP subfolders are shown. Of the files, every print file
        and at most CAP others, notes and summaries first, are shown by name."""
        items, dnames = [], sorted(node.dirs)
        many = len(dnames) > SUBCAP
        for name in dnames[:SUBCAP]:
            d = node.dirs[name]
            m = meta(count_text(d.n, ("file", "files")) + ", " + size_text(d.bytes))
            if (level >= 2 or many) and not d.has_print:
                items.append(leaf("fc", f'<a class="f" href="{href(d.path + "/")}">{esc(name)}/</a>' + m))
            else:
                items.append(folder("fc", f'<span class="f">{esc(name)}/</span>' + m, listing(d, level + 1)))
        if many:
            items.append(leaf("fc", f'<a href="{href(node.path + "/")}">{len(dnames) - SUBCAP} more folders in this '
                                    f'folder</a>'))
        names = sorted(node.files)
        prio = lambda n: (kind(n) != "text" and "summary" not in n.lower() and "readme" not in n.lower(), n)  # noqa: E731
        shown = {n for n in names if kind(n) == "print"}
        shown |= set(sorted((n for n in names if n not in shown), key=prio)[:CAP])
        for n in names:
            if n in shown:
                items.append(leaf(GLYPH[kind(n)], f'<a class="f" href="{href(node.path + "/" + n)}">{esc(n)}</a>'
                                  + meta(size_text(node.files[n]))))
        rest = [n for n in names if n not in shown]
        if rest:
            items.append(leaf("fc", f'<a href="{href(node.path + "/")}">{len(rest):,} more files in this folder</a>'
                              + meta(kinds_text(collections.Counter(kind(n) for n in rest)))))
        return items

    # each study folder's other files go under the topic with most of its pages
    by_folder = collections.defaultdict(list)
    for r in recs:
        by_folder[r["folder"]].append(r)
    home = {f: max(order, key=lambda k: (sum(r["topic"] == k for r in rs), k == D.FOLDER_TOPIC.get(f)))
            for f, rs in by_folder.items()}
    topic_name = {**{k: t for k, t, _ in D.TOPICS}, None: "No topic yet"}
    keys = order + ([None] if any(r["topic"] not in order for r in recs) else [])
    for f, rs in by_folder.items():
        if not any(r["topic"] == home[f] for r in rs):
            home[f] = rs[0]["topic"] if rs[0]["topic"] in order else None

    def node_id(key, f):
        return f"{key or 'none'}-{f}"

    topic_items = []
    for key in keys:
        rs = [r for r in recs if (r["topic"] if r["topic"] in order else None) == key]
        if not rs:
            continue
        folders = sorted({r["folder"] for r in rs}, reverse=True)
        mb = sum(r["mb"] for r in rs)
        kids = []
        ov = D.OVERVIEW.get(key or "")
        if ov and os.path.exists(os.path.join(ROOT, "docs", "overviews", ov)):
            kids.append(leaf("pg", f'<a href="{href("docs/overviews/" + ov)}">Topic overview</a>'
                             + meta(f'<span class="fn">overviews/{ov}</span>', copy(overview_url.get(ov)))))
        kids.append(folder("te", f"Command that fetches the {len(rs)} pages of this topic"
                           + meta(size_text(mb * 1e6)),
                           [f'<li class="p"><pre>{lfs_cmd(f"docs/experiments/{f}/*.html" for f in sorted(folders))}</pre></li>']))
        for f in folders:
            frs = sorted((r for r in rs if r["folder"] == f), key=lambda r: r["file"])
            whole = tree_of(sizes, f)
            sub = [page_leaf(r, len(frs) > 1) for r in frs]
            if home[f] == key:
                other = tree_of(sizes, f, exclude=page_files)
                if other.n:
                    sub.append(folder("fc", "Other files" + meta(count_text(other.n, ("file", "files")) + ", "
                                                                 + size_text(other.bytes) + ": " + kinds_text(other.kinds)),
                                      listing(other, 1)))
            else:
                sub.append(leaf("fc", f'Other files: in the <a href="#{node_id(home[f], f)}">same folder under '
                                      f'{topic_name[home[f]]}</a>'))
            subject = f'<span class="sj">{title(frs[0])}</span>' if len(frs) == 1 else ""
            kids.append(folder("fc", f'<span class="f">{esc(f)}/</span>'
                               + meta(count_text(len(frs), ("page", "pages")),
                                      count_text(whole.n, ("file", "files")) + ", " + size_text(whole.bytes)) + subject,
                               sub, ident=node_id(key, f)))
        head = f'<b>{topic_name[key]}</b>'
        topic_items.append(folder("fc", head + meta(f"{count_text(len(rs), ('page', 'pages'))} ({size_text(mb * 1e6)}) in "
                                                     f"{count_text(len(folders), ('folder', 'folders'))}"),
                                  kids, ident=key or "none"))

    # dated folders without a result page, by month, and the files at the top of docs/experiments/
    tops = sorted({p.split("/")[2] for p in sizes if p.count("/") >= 3})
    bare = [f for f in tops if f not in by_folder]
    months = collections.defaultdict(list)
    for f in bare:
        months[f[:4] + "-" + f[4:6] if re.match(r"\d{8}", f) else "undated"].append(f)
    month_items, bare_n, bare_b = [], 0, 0
    for m in sorted(months, key=lambda m: (m != "undated", m), reverse=True):
        fitems = []
        for f in sorted(months[m], reverse=True):
            t = tree_of(sizes, f)
            bare_n, bare_b = bare_n + t.n, bare_b + t.bytes
            fitems.append(folder("fc", f'<span class="f">{esc(f)}/</span>'
                                 + meta(count_text(t.n, ("file", "files")) + ", " + size_text(t.bytes) + ": "
                                        + kinds_text(t.kinds)),
                                 listing(t, 2), ident="bare-" + f))
        month_items.append(folder("fc", f'<span class="f">{m}</span>'
                                  + meta(count_text(len(months[m]), ("folder", "folders"))), fitems))
    loose = Dir("docs/experiments")
    for p, s in sizes.items():
        if p.count("/") == 2 and os.path.basename(p) != "INDEX.md":
            loose.add(os.path.basename(p), s)
    loose.total()
    root_items = topic_items + [
        folder("fc", "Folders without a result page"
               + meta(count_text(len(bare), ("folder", "folders")) + ", "
                      + count_text(bare_n, ("file", "files")) + ", " + size_text(bare_b)), month_items, ident="bare"),
        folder("fc", f'Files at the top of <span class="f">docs/experiments/</span>'
               + meta(count_text(loose.n, ("file", "files")) + ", " + size_text(loose.bytes) + ": "
                      + kinds_text(loose.kinds)), listing(loose, 1), ident="loose"),
        leaf("tx", f'<a class="f" href="{href("docs/experiments/INDEX.md")}">INDEX.md</a>'
             + meta("every page by date with a sentence on each, and the topic overviews")),
    ]
    n_folders = len(by_folder)
    n_topics = sum(1 for k in keys if k and any(r["topic"] == k for r in recs))
    all_n, all_b = len(sizes), sum(sizes.values())
    tree = (f'<ul class="tree">' + folder(
        "fc", '<span class="f">docs/experiments/</span>'
        + meta(f"{len(recs)} result pages in {n_folders} folders and {n_topics} topics",
               f"{len(tops)} folders, {all_n:,} files, {size_text(all_b)} in all"),
        root_items, is_open=True, cls="root") + "</ul>")

    by_month = collections.defaultdict(list)
    for r in sorted(recs, key=lambda r: (r["date"], r["file"]), reverse=True):
        by_month[r["date"][:7]].append(r)
    date_items = []
    for m in sorted(by_month, reverse=True):
        rows = [leaf("pg", f'{r["date"]} <a href="{href(r["file"])}">{title(r)}</a>'
                     + meta(f'<a href="#{node_id(r["topic"] if r["topic"] in order else None, r["folder"])}">'
                            f'{topic_name[r["topic"] if r["topic"] in order else None]}</a>',
                            f'{os.path.getsize(os.path.join(ROOT, r["file"])) / 1e6:.2f}&#8201;MB', copy(r["artifact"])))
                for r in by_month[m]]
        date_items.append(folder("fc", f'<span class="f">{m}</span>' + meta(count_text(len(rows), ("page", "pages"))),
                                 rows, ident="m" + m))
    date_tree = ('<ul class="tree">' + folder("fc", f"All {len(recs)} result pages, newest first", date_items,
                                               is_open=True, cls="root") + "</ul>")

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Index of result pages</title>
<style>{TREE_CSS}</style>
</head>
<body>
<div class="wrap">
<header>
<div class="eyebrow"><a href="index.html">SR&#178; Hand documentation</a> &#183; updated {time.strftime("%Y-%m-%d")}</div>
<h1>Result pages of the SR&#178; Hand documentation by topic and date</h1>
<p class="lede">The {len(recs)} dated result pages ({size_text(total * 1e6)}) under <code>docs/experiments/</code>, as a
folder tree in {n_topics} topics. A topic opens to its study folders, newest first. A study folder opens to its result
pages, each with its size, the script in <code>scripts/</code> that rebuilds it and its copy on claude.ai where they
exist, and to a folder of the other files beside them: data, films, images, print files and bench plans. That folder lists two levels of
subfolders and at most {CAP} files per folder besides the print files, with a link to the folder for the rest. The result
pages, data and media are Git LFS files; the commands that fetch them are <a href="#fetch">at the end</a>.</p>
</header>
<h2 id="bytopic">Pages by topic</h2>
<p class="tools" id="tools" hidden><button type="button" data-act="open">Expand all</button><button type="button"
data-act="close">Collapse all</button></p>
{tree}
<h2 id="bydate">Pages by date</h2>
{date_tree}
<h2 id="fetch">Fetching the pages and their files</h2>
<p>The result pages and the data and media files are Git LFS files, which open as a short pointer text until fetched;
this index, the home page and the topic overviews are ordinary files and open from any clone. Every result page
({size_text(total * 1e6)}):</p>
<pre>git lfs install
{lfs_cmd(["docs/experiments/**/*.html"])}</pre>
<p>One folder with its data and media, FOLDER being its dated name:</p>
<pre>{lfs_cmd(["docs/experiments/FOLDER/**"])}</pre>
<p>Each topic in the tree opens to the command that fetches its pages.</p>
<footer>
<p>Built by <code>python3 scripts/docs_home_page.py</code> from <code>scripts/docs_inventory.py</code>, which reads the
result folders and the rows of <code>docs/experiments/INDEX.md</code>, and from the files git tracks under
<code>docs/experiments/</code>.</p>
</footer>
</div>
<script>{TREE_JS}</script>
</body>
</html>
"""


STAMP_RE = re.compile(r"updated \d{4}-\d{2}-\d{2}")


def write_page(path, html):
    """Write a built page unless the file already has the same text apart from its "updated" date."""
    try:
        old = open(path, encoding="utf-8").read()
    except OSError:
        old = None
    if old is not None and STAMP_RE.sub("", old) == STAMP_RE.sub("", html):
        return False
    open(path, "w", encoding="utf-8").write(html)
    return True


def main():
    recs = D.records()
    titles = {k: t for k, t, _ in D.TOPICS}
    topics = "\n".join(topic_html(k, t, p, titles) for k, t, p in TOPICS)
    n_pages = len(recs)
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SR2 Hand documentation</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
<header>
<div class="eyebrow">MorphoHand &#183; updated {time.strftime("%Y-%m-%d")}</div>
<h1>Documentation of the SR&#178; Hand</h1>
<p class="lede">SR&#178; Hand is a three-finger robot hand whose finger bases ride on XY gantries: six gantry
coordinates set the finger layout (the morphology) and nine servos move the fingers. We search layouts and controllers
together in simulation, build the chosen hands on the bench, and measure whether the simulated ranking of the hands holds, on
one task: turning a 24.5&#8202;g screwdriver from flat to vertical and then driving a screw with it.</p>
</header>
<figure>
<img src="{photo_uri()}" alt="The SR2 Hand on the bench pinching the screwdriver between three fingertips, with AprilTags on the palm and on the tool." width="720" height="508">
<figcaption>The hand on the bench holding the screwdriver, with AprilTags on palm and tool (paper,
<a href="https://sr2-hand.github.io">sr2-hand.github.io</a>, Fig.&#160;1).</figcaption>
</figure>

<h2>Topics</h2>
<p>Each topic links its overview and two or three of its result pages. The <a href="pages.html">index of result pages</a>
lists all {n_pages}, by topic and by date.</p>
{topics}

<h2>Reading the pages</h2>
<p>The overviews in <code>docs/overviews/</code> open from a clone in any browser. The {n_pages} dated result pages
in <code>docs/experiments/</code> are Git LFS files, which open as a short pointer text until fetched
(about 120&#8202;MB):</p>
<pre>git lfs install
git -c lfs.fetchexclude= lfs pull --include="docs/experiments/**/*.html"</pre>
<p>Figures and films are inlined in the pages, except the 138 films of the four Codex contact pages (29&#8202;MB), which
stay files beside them: <code>git -c lfs.fetchexclude= lfs pull --include="docs/experiments/20261004-codex/**"</code>.
The <a href="pages.html">index of result pages</a> gives each page&#8217;s size, the script that rebuilds it, its copy on
claude.ai where one exists, and the command that fetches one topic; <a href="{rel("docs/experiments/INDEX.md")}">INDEX.md</a>
lists the same pages by date with a sentence on each.</p>

<h2>Code</h2>
<p><code>src/morphohand/</code> holds the simulation, RL, sampling and bench-replay code, <code>scripts/</code> the
experiment and page scripts (mapped in <a href="{rel("scripts/README.md")}">scripts/README.md</a>), and
<code>assets/mjcf/real_v1/real_hand.xml</code> the hand model. <a href="{rel("docs/overviews/code_and_methods.html")}">Code
and methods</a> follows one hand from a sample to a bench plan and describes the grasp synthesis, the evaluation protocol and
the simulator backends. <a href="{rel("README.md")}">README.md</a> describes the
environment and the names of hands and policies.</p>

<footer>
<p>Built by <code>python3 scripts/docs_home_page.py</code>.</p>
</footer>
</div>
</body>
</html>
"""
    items = [(m.group(1).strip(), False) for m in TEX_RE.finditer(html)]
    if items:
        svgs = iter(texsvg.render(items, cache_path=TEX_CACHE, scale=1.1))
        html = TEX_RE.sub(lambda m: next(svgs), html)
    html = retro_style.apply(html)
    done = write_page(OUT, html)
    print(f"{'wrote' if done else 'unchanged'} {OUT} ({os.path.getsize(OUT) / 1e3:.0f} kB)")
    done = write_page(PAGES_OUT, retro_style.apply(pages_html(recs, titles)))
    print(f"{'wrote' if done else 'unchanged'} {PAGES_OUT} ({os.path.getsize(PAGES_OUT) / 1e3:.0f} kB, {n_pages} pages)")


if __name__ == "__main__":
    main()
