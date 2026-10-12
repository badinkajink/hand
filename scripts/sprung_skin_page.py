#!/usr/bin/env python3
"""Result page: presliding compliance from a sprung skin under the 1 mm sphere pads (2026-10-12).

The contact overview carried this material as its section on the sprung skin (steps 11-12, Figures 13-16, Table 3) until
2026-10-12, when the owner asked for the skin to be compressed in the topic overviews: its presliding works on the bed at
impratio 1000 only, and the policy trainer runs it at impratio 10, where the friction rows' creep swamps the springs. This
page keeps the full account; the overview keeps a paragraph and the two-panel task 8 figure. Text blocks and figures come
from scripts/contact_overview_text.py (skin_*) and scripts/contact_overview_page.py (svg_skin, svg_t8_loops, skin_table).

    .venv/bin/python scripts/sprung_skin_page.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import contact_overview_page as P  # noqa: E402
import contact_overview_text as TXT  # noqa: E402
import native_compliance_page as NCP  # noqa: E402
import retro_style  # noqa: E402
import texsvg  # noqa: E402

ROOT = P.ROOT
D = os.path.join(ROOT, "docs/experiments/20261012-sprung_skin")
OUT = os.path.join(D, "20261012-sprung_skin_presliding.html")
TPL = os.path.join(ROOT, "scripts/sprung_skin_page.template.html")
TEX_CACHE = os.path.join(D, "texsvg_cache.json")
W8 = "docs/experiments/20261008-contact_model_policies/media/"

_NC = dict(NCP.GLOSSARY)
GLOSSARY = [
    ("presliding displacement", _NC["presliding displacement"]),
    ("recovered displacement", _NC["recovered displacement"]),
    ("loop area", _NC["loop area"]),
    ("creep", _NC["creep"]),
    ("impratio", "MuJoCo&#8217;s ratio of a contact&#8217;s frictional to its normal impedance. The friction rows&#8217; damping grows in "
     "proportion to it, so the creep speed under a held load falls in inverse proportion: on the bed the 1&#8202;mm pads creep "
     "71&#8202;&#181;m over task&#160;8&#8217;s first ramp at 10, 7.1 at 100 and 0.7 at 1000."),
    ("armature", "Inertia added to a joint&#8217;s own coordinate (kg on a slide, kg&#8202;m&#178; on a hinge), which MuJoCo adds to the "
     "diagonal of the mass matrix; it acts on the skin&#8217;s motion relative to the fingertip."),
    ("\\(G\\), \\(\\nu\\), \\(a\\)", _NC["\\(G\\), \\(\\nu\\), \\(a\\)"]),
    ("held cosine", "Cosine between the tool&#8217;s axis and the vertical at the end of a deterministic rollout, averaged over the "
     "rollouts that still hold the tool; 1 is upright."),
    ("closed-loop transfer", "A final policy run in MuJoCo-Warp with a fingertip model it was not trained on, 256 rollouts per "
     "model; the share of rollouts that keep the tool."),
]


def glossary():
    return '<dl class="glossary">' + "".join(f"<dt>{a}</dt><dd>{b}</dd>" for a, b in GLOSSARY) + "</dl>"


def render_tex(t):
    items = [((m.group(1) if m.group(1) is not None else m.group(2)).strip(), m.group(1) is not None)
             for m in P.TEX_RE.finditer(t)]
    svgs = iter(texsvg.render(items, cache_path=TEX_CACHE, scale=P.TEX_SCALE))
    return P.TEX_RE.sub(lambda m: next(svgs), t), len(items)


def main():
    import topic_overview_page as TOP
    os.makedirs(D, exist_ok=True)
    SX = P.skin_data()
    ctx = dict(skin_X=SX, t8_fig="2")
    v = {"STYLE": P.style_block(), "BUILT": time.strftime("%Y-%m-%d %H:%M")}
    v["LEDE"] = TXT.skin_lead(ctx)
    v["GLOSSARY"] = glossary()
    v["SKIN_SIZING"] = TXT.skin_sizing(ctx)
    v["FIG_SKIN"] = P.svg_skin()
    v["SKIN_T8_TEXT"] = TXT.skin_t8_text(ctx)
    v["FIG_T8"] = P.svg_t8_loops(SX)
    v["T8_FILM"] = P.film_tag(os.path.join(P.NC_D, "media/t8_compare.mp4"), os.path.join(P.NC_D, "media/t8_compare.jpg"),
                              "Figure&#160;3. The cycle of Figure&#160;2a on the three models at half speed: each pad seen through "
                              "the translucent tool with its spheres coloured by contact force (dark to light), the applied force, "
                              "and the tool&#8217;s displacement against time, Mindlin&#8217;s closed form dashed. Rendered by "
                              "<code>scripts/contact_bed_candidates.py film-compare</code>.")
    v["SKIN_TABLE"] = P.skin_table(SX)
    v["SKIN_OTHERS_TEXT"] = TXT.skin_others_text(ctx)
    v["SKIN_RL_WORK"] = TXT.skin_rl_work(ctx)
    spec = dict(film=[[(W8 + "20261008-d6_work_tpu27pads1_40M_s0_best.mp4", "1 mm pads: held cosine 0.69, 42–44 N"),
                       (W8 + "20261008-d6_work_tpu27skin_40M_s0_best.mp4", "Sprung skin: held cosine 0.79, 46–52 N")]],
                width=P.COLUMN_PX)
    mp4, jpg = TOP.film_files("contact", "skin_policies", spec, False)
    v["FIG_TRANSFER"] = (
        f'<figure><img src="{P.R.data_uri(os.path.join(ROOT, "docs/overviews/media/policies_transfer.png"), "image/png")}" '
        'width="766" alt="Grid of held shares and cosines per contact model and seed under each simulator.">'
        "<figcaption>Figure&#160;4. Transfer of the final D6 policies on the 1&#8202;N&#8202;m working plant, one row per contact "
        "model and seed: held share and cosine in closed loop under each fingertip model in MuJoCo-Warp (left) and open-loop "
        "replays of the finger commands in CPU MuJoCo, Drake and Newton (right). The skin policies keep the tool under every "
        "TPU tip model, and seed 0 also on the box tip, where every pad and TPU-mesh policy drops it; all of them grip with "
        "44&#8211;56&#8202;N and trained at impratio 10, where the skin&#8217;s springs store half of what its friction rows creep "
        "(Figure&#160;2b). From the <a href=\"../20261008-contact_model_policies/20261008-fingertip_contact_model_policies.html\">"
        "contact-model policies</a>, Figure&#160;4.</figcaption></figure>")
    v["SKIN_FILM"] = P.film_tag(mp4, jpg, "Figure&#160;5. D6 policies on the 1&#8202;N&#8202;m working plant, best seed of each arm, "
                                          "real speed: 1&#8202;mm pads (left) and pads on the sprung skin (right), both trained at "
                                          "impratio 10. Both turn the tool within a quarter second and hold it with 42&#8211;52&#8202;N; "
                                          "the skin policy stops it closer to vertical. Films from the "
                                          "<a href=\"../20261008-contact_model_policies/20261008-fingertip_contact_model_policies.html\">"
                                          "contact-model policies</a>.")
    v["SKIN_RL_SERVO"] = TXT.skin_rl_servo(ctx)
    v["FUTURE"] = TXT.skin_future_html(ctx)
    v["FOOTER"] = ("<p>Rebuild: <code>.venv/bin/python scripts/sprung_skin_page.py</code>. Rows: "
                   "<code>docs/experiments/20261007-native_compliance/</code> (the bed; <code>t8_step_convergence.jsonl</code> "
                   "for the finer steps and the trainer&#8217;s settings), <code>docs/experiments/20261008-contact_model_policies/</code> "
                   "and <code>docs/experiments/20261011-servo_plant_policies/</code> (the policies). Rigs: "
                   "<code>scripts/contact_bed_candidates.py</code>; skin scenes: <code>scripts/make_work_plant_runs.py --skin</code>.</p>")
    t = open(TPL).read()
    for k, val in v.items():
        t = t.replace("{{" + k + "}}", val)
    left = sorted(set(x.split("}}")[0] for x in t.split("{{")[1:]))
    if left:
        raise SystemExit(f"unfilled placeholders: {left}")
    t, n = render_tex(t)
    open(OUT, "w").write(retro_style.apply(t))  # plain page style (owner, 2026-10-09)
    print(f"formulas {n}; wrote {OUT} ({os.path.getsize(OUT) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
