"""The night's twist repeats (lab/repeat_cycle.sh of the bench study, 2026-11-17) into the bench study's records, scored.

    /usr/bin/python3 docs/experiments/20261117-sr2_system_identification/lab/collect_repeats.py [SINCE]

Copies corvus's records (logs/20261116-hom_trials/20261117-night_repeats/*_twist_corvus.jsonl) to the bench study's
runs_corvus/, fetches the HOM server's records of the same stamps from the workstation into runs/, and scores every
twist since SINCE (default 20261117-193000) with scripts/hardware/hom_trial_tracking.py into the bench study's
data/20261117-night_repeats_tracking.json.
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
B = ROOT / "docs/experiments/20261115-hom_bench_trials"
V = ROOT / "logs/20261116-hom_trials/20261117-night_repeats"
SINCE = sys.argv[1] if len(sys.argv) > 1 else "20261117-193000"
n = 0
for f in sorted(V.glob("*_twist_corvus.jsonl")):
    if f.name[:15] >= SINCE:
        shutil.copy(f, B / "runs_corvus" / f.name)
        n += 1
print(f"corvus records: {n}")
day = SINCE[:8]
ls = subprocess.run(["ssh", "-i", str(Path.home() / ".ssh/id_wxie"), "-o", "BatchMode=yes", "humanoid@100.65.73.59",
                     "ls ~/Programs/hand/docs/experiments/20261115-hom_bench_trials/runs/"], capture_output=True,
                    text=True, timeout=60).stdout.split()
want = [x for x in ls if x.startswith(day) and x[:15] >= SINCE and "_twist_" in x]
for x in want:
    subprocess.run(["scp", "-q", "-i", str(Path.home() / ".ssh/id_wxie"), "-o", "BatchMode=yes",
                    f"humanoid@100.65.73.59:Programs/hand/docs/experiments/20261115-hom_bench_trials/runs/{x}",
                    str(B / "runs" / x)], check=True, timeout=120)
print(f"server records: {len(want)}")
sys.path[:0] = [str(ROOT / "scripts/hardware"), str(ROOT / "scripts")]
import hom_trial_tracking as T  # noqa: E402
T.OUT = B / "data/20261117-night_repeats_tracking.json"
sys.argv = [sys.argv[0], "--since", SINCE]
T.main()
