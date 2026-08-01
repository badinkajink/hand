#!/usr/bin/env python3
"""Run UHAS's process_urdf.py headless and CAPTURE its verbose figures as PNGs.

``process_urdf.py --verbose`` is the only real check that a hand's UHAS representation is
sane: it draws the sphere construction, the hand/sphere surface correspondences, the joint
classification and five sample sphere deformations. Upstream it calls ``plt.show()`` and
``pyvista.Plotter.show()``, which on a headless box either block or silently no-op, so the
one artifact worth reading never reaches disk.

This wrapper redirects both to files, so the verification is a directory of PNGs you can
actually open. Nothing in UHAS_sim is modified -- the patches live here.

    .venv-uhas/bin/python scripts/uhas_process_urdf.py \
        --robot_path results/uhas/hands/mh_baseline/mh_baseline.urdf \
        --base_link palm --correct_axes --verbose \
        --figdir results/uhas/hands/mh_baseline/figures
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

UHAS_PROCESS_DIR = (Path(__file__).resolve().parent.parent
                    / "docs" / "uhas" / "UHAS_sim" / "process_urdf")


def _install_capture(figdir: Path) -> list[str]:
    """Point matplotlib and pyvista `show()` at files instead of a screen."""
    figdir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    counter = {"n": 0}

    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    def _save_mpl(*args, **kwargs):
        for num in plt.get_fignums():
            fig = plt.figure(num)
            if not fig.get_axes():
                continue
            counter["n"] += 1
            title = ""
            ax0 = fig.get_axes()[0]
            for cand in (fig._suptitle.get_text() if fig._suptitle else "",
                         ax0.get_title()):
                if cand:
                    title = "_" + "".join(c if c.isalnum() else "-" for c in cand)[:48]
                    break
            path = figdir / f"{counter['n']:03d}_mpl{title}.png"
            fig.savefig(path, dpi=110, bbox_inches="tight")
            saved.append(str(path))
            print(f"[capture] {path}", flush=True)
        plt.close("all")

    plt.show = _save_mpl

    try:
        import pyvista as pv
        pv.OFF_SCREEN = True
        _orig_pv_show = pv.Plotter.show

        def _save_pv(self, *args, **kwargs):
            counter["n"] += 1
            path = figdir / f"{counter['n']:03d}_pv.png"
            kwargs["screenshot"] = str(path)
            kwargs["auto_close"] = True
            kwargs.pop("interactive", None)
            try:
                out = _orig_pv_show(self, *args, **kwargs)
            except Exception as exc:  # a failed render must not kill the pipeline
                print(f"[capture] pyvista show failed: {exc}", flush=True)
                return None
            saved.append(str(path))
            print(f"[capture] {path}", flush=True)
            return out

        pv.Plotter.show = _save_pv
    except ImportError:
        pass

    return saved


def main() -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--figdir", default=None,
                    help="where to write captured figures "
                         "(default: <urdf dir>/figures)")
    ap.add_argument("-h", "--help", action="store_true")
    known, rest = ap.parse_known_args()

    if known.help:
        print(__doc__)
        print("\nAll other flags are passed through to UHAS process_urdf.py:\n")
        os.chdir(UHAS_PROCESS_DIR)
        sys.argv = ["process_urdf.py", "--help"]
        runpy.run_path("process_urdf.py", run_name="__main__")
        return 0

    figdir = known.figdir
    if figdir is None:
        for i, a in enumerate(rest):
            if a == "--robot_path" and i + 1 < len(rest):
                figdir = str(Path(rest[i + 1]).resolve().parent / "figures")
    if figdir is None:
        figdir = "uhas_figures"

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

    saved = _install_capture(Path(figdir))

    # process_urdf.py does `from utils_urdf import *`, so it must run from its own dir.
    sys.path.insert(0, str(UHAS_PROCESS_DIR))
    os.chdir(UHAS_PROCESS_DIR)
    sys.argv = ["process_urdf.py", *rest]
    try:
        runpy.run_path("process_urdf.py", run_name="__main__")
    finally:
        print(f"\n[capture] {len(saved)} figure(s) written to {figdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
