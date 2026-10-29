#!/usr/bin/env python3
"""Page on the calibration of the magnetic fingertip on the UR5 against the HEX-E (owner, 2026-10-26: "the calibration
process, what motions we initiated, geometry of the calibration and the desired sensing result ... formalizes and
justifies, mathematically and experimentally, the protocol ... the learning architecture, data processing/pooling across
sensors"): what the calibration measures, the cell and frames, the motions and what each identifies, the registration,
the ground truth of a hold, the sessions and their coverage, the learned models, the results on tips 0 to 2, the five-magnet tip v2 ID 1 and four
magnets against five on v3 ID 1 (step 9 of docs/handoff/20261027-crosstalk_feed_poller.md, owner 2026-10-28).
Reads the figures and numbers of scripts/fingertip_calibration_figures.py; writes
docs/experiments/20261026-fingertip_calibration/20261026-fingertip_calibration_on_the_ur5.html.

    .venv/bin/python scripts/fingertip_calibration_page.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import docs_index  # noqa: E402
import retro_style  # noqa: E402
import texsvg  # noqa: E402
import ur5_tip_calibration_page as P  # noqa: E402   (esc, code, fig, film, table, TEX_RE)

D = ROOT / "docs/experiments/20261026-fingertip_calibration"
MEDIA, DATA = D / "media", D / "data"
PAGE = D / "20261026-fingertip_calibration_on_the_ur5.html"
TEMPLATE = ROOT / "scripts/ur5_tip_calibration_page.template.html"
TEX_CACHE = D / "texsvg_cache.json"
EXP = ROOT / "docs/experiments"
UR5 = EXP / "20261022-ur5_tip_calibration"
L = dict(
    protocol=UR5 / "20261022-ur5_tip_calibration_protocol.html",
    lab=UR5 / "20261022-ur5_tip_calibration_lab_runs.html",
    gt=UR5 / "20261022-ur5_tip_ground_truth_and_predictors.html",
    cb=UR5 / "20261024-checkerboard_fingertip_on_the_ur5.html",
    drift=UR5 / "20261023-ur5_tip_board_drift.html",
    ati=EXP / "20261023-ati_ft_calibration/20261023-ati_ft_calibration_plan.html",
    phys=EXP / "20261026-fingertip_physical_model/20261026-fingertip_physical_model.html",
    contact=ROOT / "docs/overviews/contact_model.html",
    overview=ROOT / "docs/overviews/printed_fingertips.html",
    tall=EXP / "20261026-tall_indenter/README.md",
    crosstalk=EXP / "20261027-fingertip_crosstalk/20261027-magnetic_crosstalk_between_neighbouring_fingertips.html",
    v3=EXP / "20261027-v3_tip_calibration_and_crosstalk/README.md",
)


def link(p, text):
    return f'<a href="{P.esc(os.path.relpath(p, D))}">{text}</a>'


def build():
    from fingertip_calibration_text import sections
    N = json.loads((DATA / "20261026-numbers.json").read_text())
    F = dict(cell=MEDIA / "20261027-cell.png", motions=MEDIA / "20261026-motions.png",
             coverage=MEDIA / "20261026-coverage.png", learning=MEDIA / "20261026-learning.png",
             by_level=MEDIA / "20261026-error_by_level.png", density=MEDIA / "20261026-density.png",
             shear=MEDIA / "20261026-shear.png", transfer=MEDIA / "20261026-transfer_pooling.png",
             hard=MEDIA / "20261028-hard_presses_v2id1.png", four_five=MEDIA / "20261028-four_vs_five.png",
             film=UR5 / "media/20261022-lab_sphere2_overhead.mp4", film_poster=UR5 / "media/20261022-lab_sphere2_overhead.png")
    S, lede, summary, film_html = sections(N, F, P, link, L)
    toc = "\n".join(f'<li><a href="#{sid}">{k}. {P.esc(hd)}</a></li>' for k, (sid, hd, _) in enumerate(S, 1))
    secs = "\n".join(f'<section id="{sid}"><h2>{k}. {P.esc(hd)}</h2>{body}</section>' for k, (sid, hd, body) in enumerate(S, 1))
    byline = ("Sessions sphere2 to cb3t of tips 0 to 3, v2id1, rodv2id1 and tallv2id1 of v2 ID 1, spherev3id1, rodv3id1 and "
              "tallv3id1 of v3 ID 1, and spherev3id2, tallv3id2 and rodv3id2 of v3 ID 2, on the lab's UR5 with the OptoForce "
              "HEX-E, 2026-10-22 to 10-29, analysed on the workstation ("
              + link(UR5, "docs/experiments/20261022-ur5_tip_calibration/") + ") and v3 ID 2's on the lab computer. "
              "v2 ID 1's and the v3 tips' analyses are the UR5 "
              "coordinator's (commits 0a487cf5 to a4839dc7), read as committed. Figures by "
              + P.code("scripts/fingertip_calibration_figures.py") + "; page by " + P.code("scripts/fingertip_calibration_page.py")
              + ". Read " + link(L["phys"], "the physical model of the fingertip") + " first; the topic overview is "
              + link(L["overview"], "printed fingertips and force sensing") + ".")
    page = TEMPLATE.read_text()
    for k, v in {"TITLE": "Fingertip calibration on the UR5",
                 "H1": "Calibration of the magnetic fingertip on the UR5 against a six-axis force sensor",
                 "EYEBROW": "Printed fingertips and force sensing &#183; 2026-10-26", "LEDE": lede, "BYLINE": byline,
                 "FILM": film_html, "TOC": toc, "SECTIONS": secs, "FOOTER": str(PAGE.relative_to(ROOT)),
                 "TEXCSS": texsvg.CSS}.items():
        page = page.replace("{{" + k + "}}", v)
    items = [((m.group(1) if m.group(1) is not None else m.group(2)).strip(), m.group(1) is not None)
             for m in P.TEX_RE.finditer(page)]
    svgs = iter(texsvg.render(items, cache_path=TEX_CACHE))
    page = P.TEX_RE.sub(lambda m: next(svgs), page)
    page = retro_style.apply(page)
    page = docs_index.tag(page, topic="fingertip", summary=summary, date="2026-10-26")
    PAGE.write_text(page)
    print(f"{PAGE.relative_to(ROOT)}: {PAGE.stat().st_size / 1e6:.2f} MB, {len(items)} formulas", flush=True)


if __name__ == "__main__":
    build()
