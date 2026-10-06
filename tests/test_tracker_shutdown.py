"""A tracker launched with SIGINT ignored must still save its trace when stopped."""
from __future__ import annotations

import csv
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _record_with_fake_camera(out: Path) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import real_v1_tag_tracker as tracker

    class Pipe:
        frames = 0

        def wait_for_frames(self, timeout):
            self.frames += 1
            # Signal readiness from inside the recording loop, after the reference latch.
            if self.frames == 4:
                out.with_suffix(".ready").touch()
            time.sleep(0.02)
            return SimpleNamespace(get_infrared_frame=lambda: SimpleNamespace(
                get_data=lambda: tracker.np.zeros((8, 8), dtype="uint8")))

        def stop(self):
            out.with_suffix(".camera_closed").touch()

    pipe = Pipe()
    ref = SimpleNamespace(pose_R=tracker.np.eye(3),
                          pose_t=tracker.np.array([[0.0], [0.0], [0.3]]))
    tracker._camera = lambda: (None, None, lambda **kwargs: None)
    tracker.open_ir = lambda *args, **kwargs: (pipe, None, None, {})
    tracker.detect = lambda *args, **kwargs: {tracker.T.REF_TAG_ID: ref}
    sys.argv = ["real_v1_tag_tracker.py", "--out", str(out), "--seconds", "300",
                "--latch-frames", "2", "--video-hz", "0", "--heading-deg", "90",
                "--quiet"]
    raise SystemExit(tracker.main())


@pytest.mark.skipif(os.name != "posix", reason="detached POSIX launcher signal inheritance")
@pytest.mark.parametrize("ignore_sigint", [False, True], ids=["foreground", "detached"])
def test_sigint_saves_trace_and_closes_camera(tmp_path, ignore_sigint):
    out = tmp_path / "trace.csv"
    # exec preserves SIG_IGN, as in the station's `nohup setsid ... &` launcher.
    launcher = (
        "import os, signal, sys; "
        f"signal.signal(signal.SIGINT, signal.{'SIG_IGN' if ignore_sigint else 'SIG_DFL'}); "
        "os.execv(sys.executable, [sys.executable, *sys.argv[1:]])"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", launcher, str(Path(__file__).resolve()), str(out)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})
    try:
        deadline = time.monotonic() + 10.0
        while not out.with_suffix(".ready").exists():
            if proc.poll() is not None:
                pytest.fail(f"tracker exited before recording: {proc.communicate()[0]}")
            assert time.monotonic() < deadline, "tracker did not enter the recording loop"
            time.sleep(0.01)
        proc.send_signal(signal.SIGINT)
        try:
            stdout, _ = proc.communicate(timeout=3.0)
        except subprocess.TimeoutExpired:
            pytest.fail("tracker ignored SIGINT and kept recording")
        assert proc.returncode == 0, stdout
        assert "interrupted at" in stdout
        with out.open() as stream:
            assert list(csv.DictReader(stream))
        summary = json.loads(out.with_name("trace_SUMMARY.json").read_text())
        assert summary["trace"] == str(out)
        assert out.with_suffix(".camera_closed").exists()
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.communicate(timeout=5.0)


if __name__ == "__main__":
    _record_with_fake_camera(Path(sys.argv[1]))
