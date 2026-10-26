#!/usr/bin/env python3
"""Page on the physical model of the magnetic fingertip (owner, 2026-10-26: "clearly understand the role and implementation
of FEA, its relation to the printed tip, and it + the magnetic field sensing composing the physical model that we used for
initial estimates"): the printed pad and its sensor, the model as a chain of linear maps, the finite elements, the
magnets' field, the magnetometers, how a reading encodes the contact, the estimator, and the untrained model against the
UR5 sessions. Reads the figures and numbers of scripts/fingertip_physical_model_figures.py (which reads the two FE
outputs of scripts/fingertip_physical_model_fe.py); writes
docs/experiments/20261026-fingertip_physical_model/20261026-fingertip_physical_model.html.

    .venv/bin/python scripts/fingertip_physical_model_page.py
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
import ur5_tip_calibration_page as P  # noqa: E402   (esc, code, fig, film, TEX_RE)

D = ROOT / "docs/experiments/20261026-fingertip_physical_model"
MEDIA, DATA = D / "media", D / "data"
PAGE = D / "20261026-fingertip_physical_model.html"
TEMPLATE = ROOT / "scripts/ur5_tip_calibration_page.template.html"
TEX_CACHE = D / "texsvg_cache.json"
EXP = ROOT / "docs/experiments"
L = dict(
    est=EXP / "20261020-fingertip_contact_estimation/20261020-model_based_contact_estimation.html",
    fe=EXP / "20261009-tpu_tip_fem/20261009-printed_tpu_fingertip_finite_elements.html",
    v2=EXP / "20261019-v2_tip_fe_checks/20261019-v2_fingertip_stiffness_checks.html",
    steel=EXP / "20261023-foreign_field_rejection/20261023-foreign_field_rejection.html",
    gt=EXP / "20261022-ur5_tip_calibration/20261022-ur5_tip_ground_truth_and_predictors.html",
    cal=EXP / "20261026-fingertip_calibration/20261026-fingertip_calibration_on_the_ur5.html",
    overview=ROOT / "docs/overviews/printed_fingertips.html",
    contact=ROOT / "docs/overviews/contact_model.html",
)


def link(p, text):
    return f'<a href="{P.esc(os.path.relpath(p, D))}">{text}</a>'


def build():
    from fingertip_physical_model_text import sections
    N = json.loads((DATA / "20261026-numbers.json").read_text())
    F = dict(anatomy=MEDIA / "20261026-fingertip_anatomy.png", chain=MEDIA / "20261026-model_chain.png",
             moduli=ROOT / "docs/overviews/media/fingertip_moduli_plot.png", fe=MEDIA / "20261026-fe_sections.png",
             field=MEDIA / "20261026-magnet_field.png", patterns=MEDIA / "20261026-press_patterns.png",
             likelihood=MEDIA / "20261026-likelihood_maps.png", ur5=MEDIA / "20261026-ur5_untrained_model.png",
             film=EXP / "20261020-fingertip_contact_estimation/media/20261020-viewer_surface_presses.mp4",
             film_poster=EXP / "20261020-fingertip_contact_estimation/media/20261020-viewer_surface_presses.png")
    S, lede, summary, film_html = sections(N, F, P, link, L)
    toc = "\n".join(f'<li><a href="#{sid}">{k}. {P.esc(hd)}</a></li>' for k, (sid, hd, _) in enumerate(S, 1))
    secs = "\n".join(f'<section id="{sid}"><h2>{k}. {P.esc(hd)}</h2>{body}</section>' for k, (sid, hd, body) in enumerate(S, 1))
    byline = ("Models of the pad B05R on DIP link b v4 (FE case b_v2), the five-chip ReSkin 5X board and the UR5 sessions "
              "sphere2, cb1t, cb1r and cb2t. Figures by " + P.code("scripts/fingertip_physical_model_figures.py")
              + " from the FE of " + P.code("scripts/fingertip_physical_model_fe.py") + "; page by "
              + P.code("scripts/fingertip_physical_model_page.py") + ". The topic overview is "
              + link(L["overview"], "printed fingertips and force sensing") + "; the sphere pads that represent this "
              "pad in simulation are in " + link(L["contact"], "the contact model overview") + ".")
    page = TEMPLATE.read_text()
    for k, v in {"TITLE": "Fingertip physical model",
                 "H1": "Elasticity, magnet motion and magnetic readout of the printed fingertip",
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
