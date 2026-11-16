"""The latitude sweep's table (2026-11-16): for each line of a hom_series.py summary, the HOM server's record of the
same trial (fetched from the workstation by its start time), the two records merged and scored by
sr2hand.control.trial_score, and the plan's contact latitude; printed and written as JSON.

    /usr/bin/python3 sweep_table.py SUMMARY.jsonl --out TABLE.json
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "src"))
from sr2hand.control import trial_score as TS  # noqa: E402

WS = "humanoid@100.65.73.59"
KEY = str(Path.home() / ".ssh/id_wxie")
WS_RUNS = "~/Programs/hand/docs/experiments/20261115-hom_bench_trials/runs"
CACHE = ROOT / "logs/20261116-hom_trials/server_records"


def latitude(plan):
    p = json.loads(Path(plan if Path(plan).is_absolute() else HERE / plan).read_text())
    st = p["meta"]["hom"]["setup"]
    c = np.array([0.0, 0.0, 0.120]) - np.asarray(st["palm_m"], float)
    lat = [np.degrees(np.arcsin((np.asarray(t)[2] - c[2]) / np.linalg.norm(np.asarray(t) - c)))
           for t in st["targets_m"].values()]
    return round(float(np.mean(lat)), 1)


def server_record(corvus_record):
    """The workstation's record whose name's stamp lies within 10 s after the corvus record's."""
    stamp = Path(corvus_record).name[:15]                      # YYYYmmdd-HHMMSS
    CACHE.mkdir(parents=True, exist_ok=True)
    ls = subprocess.run(["ssh", "-i", KEY, "-o", "BatchMode=yes", WS, f"ls {WS_RUNS}"], capture_output=True,
                        text=True, timeout=30).stdout.split()
    t0 = int(stamp[9:11]) * 3600 + int(stamp[11:13]) * 60 + int(stamp[13:15])
    for name in ls:
        if not name.startswith(stamp[:9]):
            continue
        t = int(name[9:11]) * 3600 + int(name[11:13]) * 60 + int(name[13:15])
        if 0 <= t - t0 <= 10:
            local = CACHE / name
            if not local.exists():
                subprocess.run(["scp", "-q", "-i", KEY, f"{WS}:{WS_RUNS}/{name}", str(local)], check=True, timeout=60)
            return local
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summary")
    ap.add_argument("--mass-g", type=float, default=28.1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = [json.loads(x) for x in Path(a.summary).read_text().splitlines() if x.strip()]
    out = []
    for r in rows:
        srv = server_record(r["record"])
        row = {k: r.get(k) for k in ("label", "plan", "record", "twist_min_deg", "twist_max_deg", "rise_max_mm",
                                     "drift_end_mm", "stop", "periods")}
        row["latitude_deg"] = latitude(r["plan"])
        if srv is not None:
            merged = CACHE / (Path(r["record"]).stem + "_merged.jsonl")
            h, s = TS.merge_records(str(srv), r["record"], a.mass_g / 1000.0)
            TS.write_trial(str(merged), h, s)
            sc = TS.score_file(str(merged))
            err, ref = sc.get("track_err_rms_deg"), sc.get("track_ref_rms_deg")
            row.update(server=str(srv), track_err_rms_deg=err, track_ref_rms_deg=ref,
                       track_ratio=None if not err or not ref else round(max(0.0, 1.0 - err / ref), 3))
        out.append(row)
        print(f"{row['label']:12s} lat {row['latitude_deg']:5.1f}  twist {row['twist_min_deg']:6.1f} .. "
              f"{row['twist_max_deg']:6.1f} deg  rise {row['rise_max_mm']:5.1f} mm  drift {row['drift_end_mm']:5.1f} mm  "
              f"1-err/ref {row.get('track_ratio')}  ({row['periods']} periods{', ' + row['stop'] if row['stop'] else ''})")
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
