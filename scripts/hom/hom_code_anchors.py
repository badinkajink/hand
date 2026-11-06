#!/usr/bin/env python3
"""Where each difference between the HOM paper (arXiv 2609.25619v2) and our library is implemented: links to the lines on
GitHub for the difference tables of the method page (scripts/hom/hom_framework_page.py) and the review page
(scripts/hom/hom_review_page.py). The lines are found by name or text through scripts/docs/code_blocks.py, so the links
follow the code.

    import hom_code_anchors as CA;  CA.cell("Applied force")   # the HTML of a table cell, "" when no code applies
"""
from __future__ import annotations

import code_blocks as CB

HOM = "src/sr2hand/control/hom.py"
SURF = "src/sr2hand/control/surfaces.py"
TASKS = "scripts/hom/hom_tasks.py"


def _def(path, name, text=None):
    return CB.link(path, CB.find_def(path, name), f"<code>{text or name}</code>")


def _step(text, until, name):
    return CB.link(HOM, CB.find_span(HOM, text, until=until, within="HomController.step"), f"<code>{name}</code>")


def _targets():
    return _step("off_gap, off_force, off_grav, gap = ", "out[f] = np.clip(", "step: servo targets")


def _lead():
    return _step("tr = self.theta_ref[f] + dt * u", "self.theta_ref[f] = np.clip(tr", "step: lead limit")


def _thumb():
    return CB.link(TASKS, CB.find_span(TASKS, "the thumb alone opposes the index and the middle", after=4,
                                       within="Schedule.__call__"), "<code>Schedule.__call__</code>")


CELLS = {
    "Joint actuation": lambda: _targets(),
    "Applied force": lambda: _targets() + ", " + _def(HOM, "HomParams"),
    "Contact model": lambda: _def(TASKS, "build", "hom_tasks.build"),
    "Contact point": lambda: _def(TASKS, "Plant.contacts"),
    "Closest points": lambda: _def(HOM, "Mirror.gcf_witness") + ", " + _def(HOM, "Mirror.gcf_from_cop"),
    "Curvature": lambda: _def(SURF, "HalfEllipsoid.shape_operator"),
    "Curvature, geodesics": lambda: _def(SURF, "HalfEllipsoid.shape_operator") + ", "
                                    + _def(SURF, "Surface.geodesic_step"),
    "Geodesic step": lambda: _def(SURF, "Surface.geodesic_step"),
    "Weights and units": lambda: _def(HOM, "solve_eq2"),
    "Weights, units, gains": lambda: _def(HOM, "solve_eq2") + ", " + _def(HOM, "FilterParams"),
    "Filter gains": lambda: _def(HOM, "FilterParams"),
    "Rates": lambda: _def(HOM, "HomParams"),
    "Windup": lambda: _lead() + ", " + _def(HOM, "HomController.reanchor"),
    "Reference windup": lambda: _lead() + ", " + _def(HOM, "HomController.reanchor"),
    "Pusher": lambda: _def(TASKS, "mode_weights"),
    "Pusher contact": lambda: _def(TASKS, "mode_weights"),
    "Pinch forces": lambda: _thumb(),
    "Palm and object pose": lambda: _def(HOM, "Measurement"),
    "Object pose": lambda: _def(HOM, "Measurement"),
}


def cell(item: str) -> str:
    f = CELLS.get(item)
    return f() if f else ""
