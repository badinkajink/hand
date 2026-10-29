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
    xt=EXP / "20261027-fingertip_crosstalk/20261027-magnetic_crosstalk_between_neighbouring_fingertips.html",
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
    # the four-magnet pad 05R4 leads (owner, 2026-10-28); the five-magnet figures above follow in collapsed blocks
    F.update({k + "4": MEDIA / f"20261028-{n}_four.png" for k, n in (
        ("anatomy", "fingertip_anatomy"), ("chain", "model_chain"), ("fe", "fe_sections"), ("field", "magnet_field"),
        ("patterns", "press_patterns"), ("likelihood", "likelihood_maps"), ("ur5", "ur5_untrained_model"))})
    F["gain"] = MEDIA / "20261029-signal_by_region.png"     # v3 ID 2's suite of 2026-10-28/29 beside v3 ID 1 and v2 ID 1
    S, lede, summary, film_html = sections(N, F, P, link, L)
    toc = "\n".join(f'<li><a href="#{sid}">{k}. {P.esc(hd)}</a></li>' for k, (sid, hd, _) in enumerate(S, 1))
    secs = "\n".join(f'<section id="{sid}"><h2>{k}. {P.esc(hd)}</h2>{body}</section>' for k, (sid, hd, body) in enumerate(S, 1))
    byline = ("Models of the pads 05R4 (four magnets, FE case b_05R4) and B05R (five, b_v2) on DIP link b v4, the "
              "five-chip ReSkin 5X board and the UR5 sessions sphere2, cb1t, cb1r and cb2t of tips 0 to 2, the grid "
              "stage of v3 ID 1 and v2 ID 1, and the low sphere's session of v3 ID 2 (2026-10-28/29). The four-magnet "
              "layout was chosen on 2026-10-28; each five-magnet figure "
              "follows its four-magnet one under \u201cView with 5 magnets\u201d. Figures by " + P.code("scripts/fingertip_physical_model_figures.py")
              + " from the FE of " + P.code("scripts/fingertip_physical_model_fe.py") + " and the UR5 check of "
              + P.code("scripts/fingertip_physical_model_v3_check.py") + "; page by "
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
