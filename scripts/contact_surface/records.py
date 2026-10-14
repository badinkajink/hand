"""Restartable run directories; CSV/gzip fallback requires no optional dependencies."""

import csv
import gzip
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def versions():
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    packages = {}
    for name in [
        "mujoco",
        "drake",
        "numpy",
        "matplotlib",
        "scipy",
        "imageio",
        "imageio-ffmpeg",
        "Pillow",
        "mujoco-warp",
        "mjlab",
        "warp-lang",
        "newton",
    ]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    sources = list((ROOT / "scripts/contact_surface").glob("*.py"))
    sources += [
        ROOT / "scripts" / name
        for name in [
            "distributed_contact.py",
            "distributed_contact_dynamics.py",
            "distributed_contact_transfer.py",
            "distributed_contact_gpu.py",
            "distributed_contact_newton.py",
            "distributed_contact_drake.py",
            "distributed_contact_pressure.py",
            "distributed_contact_videos.py",
            "distributed_contact_video_snapshots.py",
            "distributed_contact_video_page.py",
            "distributed_contact_transfer_page.py",
            "profile_contact_mapping.py",
            "hom_chain.py",
            "hom_paper_tasks.py",
            "hom_control.py",
            "hom_contact_rig.py",
            "hom_hand_brake.py",
            "distributed_contact_tasks.py",
            "distributed_contact_diagexact_tasks.py",
            "distributed_contact_geometry_videos.py",
            "distributed_contact_task_page.py",
        ]
    ]
    cpu = next(
        (
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        ),
        platform.processor(),
    )
    return dict(
        git_commit=git("rev-parse", "HEAD"),
        git_status=git("status", "--porcelain"),
        python=sys.version,
        executable=sys.executable,
        packages=packages,
        cpu=cpu,
        device="CPU",
        source_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
            if p.exists()
        },
    )


class CsvLog:
    def __init__(self, path):
        self.file = (
            gzip.open(path, "wt", newline="")
            if str(path).endswith(".gz")
            else open(path, "w", newline="")
        )
        self.writer = None

    def row(self, row):
        if self.writer is None:
            self.writer = csv.DictWriter(self.file, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
