#!/usr/bin/env python3
"""Build docs/index.html, the home page of the documentation: the hand in two sentences, the six topics with their
state and links, how to fetch the result pages, and where the code is.

    python3 scripts/docs_home_page.py

The page is an ordinary git file that opens from a clone in a browser with no server (owner, 2026-10-09): relative
links, the one photo inlined as a data URI. Each topic heading links its overview (docs/overviews/, file names in
docs_inventory.OVERVIEW) once that file exists; the key result pages are linked under it.
"""
from __future__ import annotations

import base64
import os
import re
import time

import docs_inventory as D
import retro_style
import texsvg

ROOT = D.ROOT
OUT = os.path.join(ROOT, "docs", "index.html")
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
     [("sphere-packed pads", E + "20261009-contact_overview/20261009-sphere_pad_contact_model.html"),
      ("comparison bed", E + "20261005-contact_bed/20261005-contact_model_bed.html")]),
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
    links = " &#183; ".join(f'<a href="{rel(p)}">{lab}</a>' for lab, p in pages)
    return (f'<div class="topic"><h3>{head}</h3><p>{text}</p>'
            f'<p class="pages">Pages: {links}.</p></div>')


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
{topics}

<h2>Reading the pages</h2>
<p>The overviews in <code>docs/overviews/</code> open from a clone in any browser. The {n_pages} dated result pages
in <code>docs/experiments/</code> are Git LFS files, which open as a short pointer text until fetched
(about 120&#8202;MB):</p>
<pre>git lfs install
git -c lfs.fetchexclude= lfs pull --include="docs/experiments/**/*.html"</pre>
<p><a href="{rel("docs/experiments/INDEX.md")}">INDEX.md</a> lists them by date, and each page names the script
that rebuilds it.</p>

<h2>Code</h2>
<p><code>src/morphohand/</code> holds the simulation, RL, sampling and bench-replay code, <code>scripts/</code> the
experiment and page scripts (mapped in <a href="{rel("scripts/README.md")}">scripts/README.md</a>), and
<code>assets/mjcf/real_v1/real_hand.xml</code> the hand model. <a href="{rel("README.md")}">README.md</a> describes the
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
    open(OUT, "w", encoding="utf-8").write(html)
    print(f"wrote {OUT} ({os.path.getsize(OUT) / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
