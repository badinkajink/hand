#!/usr/bin/env python3
"""Read-only benchmark of the nine-servo feedback path.

Run this before enabling telemetry in the web service.  It does not enable torque or
write a goal.  Each requested field is measured alone, then all available fields are
measured as one bundle to expose the cost of asking for load/voltage/etc. in addition
to position.

The SCS0009 product sheet promises load, speed, and input-voltage feedback; it does not
promise measured current.  ``rustypot`` APIs vary by version, so unsupported methods are
reported rather than guessed.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from manta_hand.servos import ServoBus

FIELDS = {
    "position": "sync_read_present_position",
    "load": "sync_read_present_load",
    "speed": "sync_read_present_speed",
    "voltage": "sync_read_present_voltage",
    "temperature": "sync_read_present_temperature",
    "current": "sync_read_present_current",
}


def measure(call, ids: list[int], seconds: float) -> dict:
    latencies, errors = [], []
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        start = time.monotonic()
        try:
            values = call(ids)
            if len(values) != len(ids):
                raise RuntimeError(f"returned {len(values)} values for {len(ids)} ids")
            latencies.append(time.monotonic() - start)
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    elapsed = sum(latencies)
    return {
        "successful_reads": len(latencies),
        "errors": len(errors),
        "first_error": errors[0] if errors else None,
        "mean_hz": (len(latencies) / elapsed if elapsed else 0.0),
        "latency_ms": {
            "mean": statistics.fmean(latencies) * 1000 if latencies else None,
            "median": statistics.median(latencies) * 1000 if latencies else None,
            "max": max(latencies) * 1000 if latencies else None,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baudrate", type=int, default=1_000_000)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--fields", default="position,load,voltage,temperature,current")
    ap.add_argument("--json", type=Path, help="also write the complete result here")
    args = ap.parse_args()
    if args.seconds <= 0:
        ap.error("--seconds must be positive")

    requested = [x.strip() for x in args.fields.split(",") if x.strip()]
    unknown = sorted(set(requested) - set(FIELDS))
    if unknown:
        ap.error(f"unknown fields: {', '.join(unknown)}")

    ids = list(range(9))
    with ServoBus(args.port, baudrate=args.baudrate) as bus:
        controller = bus.controller
        available = {field: callable(getattr(controller, FIELDS[field], None))
                     for field in requested}
        result = {
            "schema_version": 1, "port": args.port, "baudrate": args.baudrate,
            "servo_ids": ids, "seconds_per_test": args.seconds,
            "available": available, "fields": {}, "bundle": None,
        }
        print("READ-ONLY: no torque or goal registers will be written")
        for field in requested:
            if not available[field]:
                print(f"  {field:12} unsupported by installed rustypot")
                continue
            row = measure(getattr(controller, FIELDS[field]), ids, args.seconds)
            result["fields"][field] = row
            print(f"  {field:12} {row['mean_hz']:7.1f} complete 9-servo reads/s, "
                  f"median {row['latency_ms']['median']:.2f} ms, errors {row['errors']}")

        active = [f for f in requested if available[f]]
        if active:
            calls = [getattr(controller, FIELDS[f]) for f in active]

            def bundle(read_ids):
                values = []
                for call in calls:
                    values.extend(call(read_ids))
                return values

            # measure() expects one value per ID; adapt its count check while retaining timing.
            def one_bundle(_ids):
                bundle(ids)
                return ids

            result["bundle"] = measure(one_bundle, ids, args.seconds)
            row = result["bundle"]
            print(f"  {'+'.join(active):12} {row['mean_hz']:7.1f} complete bundles/s, "
                  f"median {row['latency_ms']['median']:.2f} ms, errors {row['errors']}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {args.json}")
    print("\nUse the sustained bundle rate, not baud-rate arithmetic, as the telemetry ceiling.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
