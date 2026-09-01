#!/usr/bin/env python3
"""Cache the operator's manual scores off the CB1 so the transfer analysis is reproducible.

The web UI writes `manual_score` into each run's summary on the CB1, not on the workstation,
so the bench traces here and the operator's own verdicts live on different machines. This
pulls the verdicts down into a small tracked file and joins on `run_id`.

NOTE ON `notes`: until 2026-08-31 the scoring dialog never cleared its textarea between runs,
so a note typed once was re-saved against every run scored afterwards -- one string is on 63
runs and another on 42. The notes column is therefore NOT per-run for anything recorded before
that fix and must not be read as such. `success` and `reorientation_deg` are written on every
open and are unaffected.
"""
import json, os, sys, urllib.request

OUT = "docs/experiments/20260901-real_v1-transfer-firstpass/manual_scores.json"
URL = "http://10.99.99.2:8765/api/v1/logs?limit=1000"


def main():
    tok = os.environ.get("MANTA_TOKEN", "")
    if not tok:
        for line in open(".dotenv"):
            if line.startswith("MANTA_TOKEN="):
                tok = line.split("=", 1)[1].strip().strip('"\'')
    req = urllib.request.Request(URL, headers={"Authorization": f"Bearer {tok}"})
    logs = json.load(urllib.request.urlopen(req, timeout=30))["logs"]
    out = {}
    for r in logs:
        ms, tk = r.get("manual_score"), r.get("object_track") or {}
        out[r["run_id"]] = dict(
            design=r["design"], status=r.get("status"),
            success=None if not ms else bool(ms["success"]),
            deg=None if not ms else ms.get("reorientation_deg"),
            notes=None if not ms else ms.get("notes"),
            tag_seen=tk.get("seen"), tag_vis=tk.get("visibility"))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1, sort_keys=True)
    n = sum(1 for v in out.values() if v["success"] is not None)
    print(f"{len(out)} runs, {n} with an operator verdict -> {OUT}")


if __name__ == "__main__":
    main()
