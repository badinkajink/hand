#!/usr/bin/env python3
"""Inventory of the result pages under docs/experiments: date, file, title, topic, builder, artifact.

    python3 scripts/docs_inventory.py            # table grouped by topic
    python3 scripts/docs_inventory.py --check    # pages with no topic, and topics with no page

The home page (scripts/docs_home_page.py) and the topic overviews import records() and TOPICS. Titles come from the
page's row in docs/experiments/INDEX.md (the descriptive subject), else from its <title>. The topic of each result
folder is set by hand in FOLDER_TOPIC (owner, 2026-10-09: six topics), with per-page exceptions in FILE_TOPIC.
"""
from __future__ import annotations

import argparse
import glob
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "docs", "experiments")

TOPICS = [  # key, title, scope
    ("contact", "Fingertip contact model",
     "How the soft fingertip touches the tool in simulation: sphere-packed pads, hydroelastic references, cost."),
    ("mechanisms", "Reorientation mechanisms and control",
     "How the hand turns a screwdriver from flat to vertical: fixed-contact turns, pinch and swing, support by the "
     "table or floor, gaiting, and hand-object control."),
    ("policies", "Reinforcement-learning policies",
     "Lift and reorientation policies trained in MuJoCo-Warp: recipes, seeds, plants and contact models."),
    ("chain", "Grasp-to-gait chain and sim-to-real",
     "The full task on the arm (grasp, lift, turn, set down, gait) and what transfers to the bench."),
    ("hardware", "Hand, plant and bench",
     "The servos and their model, the gantries, the UR5e mount and the tool tracker."),
    ("design", "Morphology and design search",
     "Which finger layouts turn the shaft, kinematic metrics, and finger spacing against object size."),
    ("program", "Program record", "Records that span every topic."),
]

# the topic overviews: docs/overviews/<file> (owner, 2026-10-09: stable undated names, ordinary git files)
OVERVIEW = {
    "contact": "contact_model.html",
    "mechanisms": "reorientation.html",
    "policies": "rl_policies.html",
    "chain": "grasp_to_gait_chain.html",
    "hardware": "hand_plant_bench.html",
    "design": "morphology_design.html",
}

FOLDER_TOPIC = {
    "20260818-perp_review_page": "design",
    "20260827-real_v1": "chain",
    "20260828-real_v1_search": "design",
    "20260830-real_v1-budget-rescreen": "hardware",
    "20260831-real_v1-object-tracking": "hardware",
    "20260902-real_v1_gait": "mechanisms",
    "20260902-servo-sysid": "hardware",
    "20260903-real_v1_chain": "chain",
    "20260903-real_v1_handover": "chain",
    "20260903-sim2real-gates": "chain",
    "20260904-real_v1_bench": "hardware",
    "20260904-real_v1_held": "mechanisms",
    "20260906-control_diagnosis": "mechanisms",
    "20260906-pad_elevation": "mechanisms",
    "20260906-screen_vs_chain": "chain",
    "20260908-compute_budget": "policies",
    "20260908-reorientation_journey": "mechanisms",
    "20260910-karma_metric": "design",
    "20260916-basketball": "design",
    "20260916-calibrated_plant_chain": "chain",
    "20260916-swing_reorient": "mechanisms",
    "20260916-tip_gait": "mechanisms",
    "20260916-turn_mechanism": "mechanisms",
    "20260917-d6_cal_60M": "policies",
    "20260919-hands_tranche": "policies",
    "20260920-robust_tranche": "policies",
    "20260921-workshop_briefing": "program",
    "20260923-chain_handover_gait": "chain",
    "20261001-drake-port": "design",
    "20261001-hom_contact_patch": "contact",
    "20261001-hom_hand_brake": "mechanisms",
    "20261002-hom_chain": "mechanisms",
    "20261004-codex": "contact",
    "20261005-contact_bed": "contact",
    "20261005-contact_overview": "contact",
    "20261006-fingertip_backends": "contact",
    "20261006-hom_turn3": "mechanisms",
    "20261007-contact-gait": "mechanisms",
    "20261007-contact_overview": "contact",
    "20261007-diffmjx": "contact",
    "20261007-native_compliance": "contact",
    "20261007-newton_hydro_tests": "contact",
    "20261008-contact_model_policies": "policies",
    "20261008-contact_overview": "contact",
    "20261008-hand_object_scale": "design",
    "20261009-contact_overview": "contact",
}

# pages whose topic differs from their folder's
FILE_TOPIC = {
    "20260827-real_v1_first_night.html": "hardware",
    "20260827-real_v1_rotational_lock.html": "mechanisms",
    "20260827-real_v1_routes_to_vertical.html": "mechanisms",
}

# dated revisions of a page that a newer file replaces
SUPERSEDED = {"20261005-contact_overview", "20261007-contact_overview", "20261008-contact_overview"}


def index_rows():
    """{file name: (subject, artifact URL or None, retracted)} from docs/experiments/INDEX.md."""
    out = {}
    path = os.path.join(EXP, "INDEX.md")
    for line in open(path, encoding="utf-8"):
        if not re.match(r"\| 20\d\d-\d\d-\d\d", line):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 3:
            continue
        subject = re.sub(r"\s*\(.*", "", cells[1]).strip(" *")
        f = re.search(r"\(([^)]+\.html)\)", cells[2]) or re.search(r"([\w./-]+\.html)", cells[2])
        url = re.search(r"https://claude\.ai/(?:code/)?artifact/[\w-]+", cells[3] if len(cells) > 3 else "")
        if f:
            out[os.path.basename(f.group(1))] = (subject, url.group(0) if url else None, "RETRACTED" in cells[1])
    return out


PAGE_RE = re.compile(r"\d{8}-[\w.-]+\.html")
# a line that writes or assigns the output: OUT = ..., out = args.out or (D / "..."), (DOC / "...").write_text(...),
# open(..., "w"), a "--out PATH" usage line
WRITE_RE = re.compile(r"""\b(OUT\w*|out|output|out_path|page_path|PAGE|HTML)\s*=|write_text\(|--out\b|open\([^)]*["']w["']""")


def _builders():
    """{page file name: builder script}: the script under scripts/ (or in the page's own folder) that writes the page,
    i.e. names its file on a line that assigns or writes the output; a page named by several scripts goes to a *_page.py builder first, then to the
    one whose file name shares the most words with the page's name."""
    cand = {}
    for b in sorted(glob.glob(os.path.join(ROOT, "scripts", "*.py")) + glob.glob(os.path.join(EXP, "*", "*.py"))):
        for line in open(b, encoding="utf-8", errors="replace"):
            if not WRITE_RE.search(line):
                continue
            for name in PAGE_RE.findall(line):
                cand.setdefault(name, []).append(os.path.relpath(b, ROOT))
    out = {}
    for name, bs in cand.items():
        words = set(re.split(r"[-_.]", name.lower()))
        out[name] = max(sorted(set(bs)), key=lambda b: (b.endswith("_page.py"),
                                                         len(words & set(re.split(r"[-_./]", b.lower())))))
    return out


def _title(path):
    head = open(path, encoding="utf-8", errors="replace").read(30000)
    m = re.search(r"<h1[^>]*>(.*?)</h1>", head, re.S) or re.search(r"<title>(.*?)</title>", head, re.S)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", m.group(1))).strip() if m else ""


def records():
    idx, builders = index_rows(), _builders()
    out = []
    for path in sorted(glob.glob(os.path.join(EXP, "*", "*.html"))):
        name, folder = os.path.basename(path), os.path.basename(os.path.dirname(path))
        if name.endswith(".src.html"):
            continue
        subject, url, retracted = idx.get(name, (None, None, False))
        art = os.path.join(os.path.dirname(path), "artifact_url.txt")
        if not url and os.path.exists(art):
            url = open(art).read().strip() or None
        out.append({
            "date": f"{folder[:4]}-{folder[4:6]}-{folder[6:8]}",
            "file": os.path.relpath(path, ROOT),
            "folder": folder,
            "title": subject or _title(path),
            "page_title": _title(path),
            "topic": FILE_TOPIC.get(name) or FOLDER_TOPIC.get(folder),
            "builder": builders.get(name),
            "artifact": url,
            "status": "superseded" if folder in SUPERSEDED else "retracted" if retracted else "current",
            "mb": round(os.path.getsize(path) / 1e6, 1),
        })
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--check", action="store_true")
    a = p.parse_args()
    R = records()
    if a.check:
        missing = [r["file"] for r in R if not r["topic"]]
        empty = [k for k, *_ in TOPICS if not any(r["topic"] == k for r in R)]
        print("pages with no topic:", missing or "none")
        print("topics with no page:", empty or "none")
        return
    for key, title, _ in TOPICS:
        rs = [r for r in R if r["topic"] == key]
        print(f"\n{title} ({len(rs)} pages)")
        for r in rs:
            flag = f" ({r['status']})" if r["status"] != "current" else ""
            print(f"  {r['date']}  {r['title'][:80]}{flag}\n              {r['file']}  builder: {r['builder'] or '-'}  "
                  f"{'artifact' if r['artifact'] else 'local'}  {r['mb']} MB")


if __name__ == "__main__":
    main()
