"""Waits until every servo of the given fingers reads under --below C (the queue's resume limit), polling the station's
diagnostics every 15 s; exits 1 if any of them shows a flag or lost torque.

    /usr/bin/python3 day/wait_cool.py index [middle ...] [--below 40]
"""
import argparse
import sys
import time
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("fingers", nargs="+")
    ap.add_argument("--below", type=float, default=40.0)
    a = ap.parse_args()
    sys.argv = sys.argv[:1]
    import s3_press as S3
    FT, HS, SL = S3.imports()
    P, SV = SL._driver()
    sids = [str(SV.FINGER_JOINTS[P.FINGER_ID[f]][P.SIM_JOINT_TO_SERVO[k]][0]) for f in a.fingers for k in S3.JOINTS]
    st = HS.Station("http://10.99.99.2:8765", str(Path.home() / ".sr2_station_token"))
    while True:
        d = S3.diag(st)
        bad = {s: (d.get("status") or {}).get(s) for s in sids if (d.get("status") or {}).get(s) not in (None, 0, 0.0)}
        off = {s: (d.get("torque_enable") or {}).get(s) for s in sids
               if (d.get("torque_enable") or {}).get(s) not in (None, 1, 1.0)}
        if bad or off:
            print(f"{time.strftime('%H:%M:%S')} flags {bad}, torque off {off}", flush=True)
            return 1
        temps = {s: float((d.get("temperature") or {}).get(s) or 0.0) for s in sids}
        hot = max(temps.values())
        if hot < a.below:
            print(f"{time.strftime('%H:%M:%S')} {'/'.join(a.fingers)} under {a.below:g} C (hottest {hot:g})", flush=True)
            return 0
        print(f"{time.strftime('%H:%M:%S')} hottest servo {hot:g} C: waiting under {a.below:g}", flush=True)
        time.sleep(15.0)


if __name__ == "__main__":
    sys.exit(main())
