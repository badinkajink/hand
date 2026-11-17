"""The fingertip contact estimates and HOM's use of them, checked offline on the recordings of 2026-11-16 (workstation;
README section "Trials with the fingertips (from 2026-11-17)").

    press   the press with the pinch at the axis (13:19:27, corvus record 20261116-131927 and the splitter's log
            20261116-131831-mux.bin) through tip_contacts.replay, with and without the own-pose term: per finger the
            periods with a contact, the force and the gap between the contact and the tracked cylinder's surface in P,
            and beside it the gap of the pad's apex (where the hand's model puts the contact without the boards)
    pickup  the first pick-up by controlled slip (15:43-15:46: pinch_pickup.py's grasp, lift and slip telemetry at 20 Hz
            and 20261116-154014-mux.bin), the rest of 15:43:58 (pinch_pads_rest.json): the same per finger, against the
            tracked sphere (radius 20.75 mm, taped)
    hom     HOM servers on this host (hom_server serve --once, the measured plant) answering the press's recorded states
            sent again by hom_trial.py replay: --tips off without contacts (as on 11-16), --tips off with them, and
            --tips on with them; the targets against the bench's server record and against each other, and the source
            of each finger's contact frame

Writes data/20261117-tip_contacts/20261117-<check>.json (the summaries; the per-period rows stay in
logs/20261117-tip_contacts/).

    resguard run --mem 3G -- .venv/bin/python docs/experiments/20261115-hom_bench_trials/lab/tips_offline_check.py press
"""
import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(ROOT / "src/sr2hand/driver/reskin5x/host"), str(ROOT / "src"), str(ROOT / "src/sr2hand/driver/manta/host")]
import tip_contacts as TC  # noqa: E402

DATA = BENCH / "data/20261117-tip_contacts"
LOGS = ROOT / "logs/20261117-tip_contacts"
MUX = ROOT / "logs/20261116-three_finger_mux"
PRESS = dict(corvus=BENCH / "runs_corvus/20261116-131927_press_corvus.jsonl", mux=MUX / "20261116-131831-mux.bin",
             server=BENCH / "runs/20261116-131932_press_d25_trial000.jsonl",
             plan=BENCH / "plans/20261116-press_x70_y110_lat0_plan.json")
PICKUP = dict(telemetry=[BENCH / "data/20261116-pinch_pickup" / f for f in (
    "20261116-154328_pinch_grasp_telemetry.jsonl", "20261116-154429_pinch_lift_telemetry.jsonl",
    "20261116-154534_pinch_slip_telemetry.jsonl")], rest=BENCH / "data/20261116-pinch_pickup/20261116-pinch_pads_rest.json",
              mux=MUX / "20261116-154014-mux.bin", plan=BENCH / "plans/20261116-pinch_pickup_thumb_middle_plan.json")
R_SPHERE = 20.75                     # mm: the 40 mm sphere under its tape (pinch_pickup.R_TAPED)


def read_jsonl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def apex_gaps(out, rows, d6, shape, R_mm):
    """Per finger the gap (mm) between the pad's apex (tool_chain.SR2Hand's default contact point) at the row's joints
    and the object's surface, for the periods in which the boards show that finger's contact."""
    hand = TC.SR2Hand.load()
    m = {"thumb": d6[0:2], "index": d6[2:4], "middle": d6[4:6]}
    g = {f: [] for f in TC.FINGERS}
    for o, r in zip(out, rows):
        if r.get("joints") is None or r.get("obj") is None:
            continue
        q = TC.sim_deg(np.asarray(r["joints"], float).reshape(3, 3))
        for f in TC.FINGERS:
            if o["tips"].get(f) is None:
                continue
            c, _ = hand.box(f)
            T = hand.T_P_contact(f, m[f][0] - c[0], m[f][1] - c[1], *(math.radians(v) for v in q[f]))
            g[f].append(TC._obj_gap(T[:3, 3], r["obj"], shape, R_mm))
    return {f: None if not v else np.round(np.percentile(v, [10, 50, 90]), 2).tolist() for f, v in g.items()}


def summary(out, info, rows, d6, shape, R_mm):
    s = TC.summarize(out, info)
    s["apex_gap_mm"] = apex_gaps(out, rows, d6, shape, R_mm)
    s["rest"] = {f: {k: v.get(k) for k in ("model", "saved_fit_mT", "new_fit_mT", "q_rest")} for f, v in info.items()}
    return s


def cmd_press(a):
    lines = read_jsonl(PRESS["corvus"])
    head, rows = lines[0], [x for x in lines[1:] if "seq" in x]
    t0 = head["t0_ms"] / 1000.0
    plan = json.loads(PRESS["plan"].read_text())
    d6, R_mm = plan["meta"]["hom"]["d6_P_mm"], 1e3 * plan["meta"]["hom"]["setup"]["r_m"]
    feed = TC.MuxLog(PRESS["mux"], t0 - 6.0, rows[-1]["t_ms"] / 1000.0 + 1.0)
    res = {"record": str(PRESS["corvus"].relative_to(ROOT)), "mux": str(PRESS["mux"].relative_to(ROOT)),
           "rest": "1.0 s ending 0.2 s before t = 0, the first row's joints"}
    LOGS.mkdir(parents=True, exist_ok=True)
    for name, own in (("own_pose", True), ("no_own_pose", False)):
        info, out = TC.replay(rows, feed, d6, t0 - 0.2, 1.0, "cylinder", R_mm, own_pose=own)
        res[name] = summary(out, info, rows, d6, "cylinder", R_mm)
        with open(LOGS / f"20261117-press_131927_{name}.jsonl", "w") as fh:
            fh.write(json.dumps({"kind": "header", "rest": info}) + "\n")
            for o in out:
                fh.write(json.dumps(o) + "\n")
        mid = [o["own_uT"]["middle"] for o in out if o["own_uT"]["middle"] is not None and 0.5 <= o["t"] <= 1.3]
        if own and mid:
            res[name]["middle_own_pose_uT_0.5_to_1.3_s"] = np.round(np.percentile(mid, [10, 50, 90]), 1).tolist()
        res[name]["middle_d_uT_0.5_to_1.3_s"] = np.round(np.percentile(
            [o["d_uT"]["middle"] for o in out if 0.5 <= o["t"] <= 1.3], [10, 50, 90]), 1).tolist()
        res[name]["middle_e_0.5_to_1.3_s"] = np.round(np.percentile(
            [o["e"]["middle"] for o in out if 0.5 <= o["t"] <= 1.3], [10, 50, 90]), 1).tolist()
    return res


def cmd_pickup(a):
    from manta_hand import plan as MP
    rows = []
    for p in PICKUP["telemetry"]:
        step = p.name.split("_pinch_")[1].split("_")[0]
        for r in read_jsonl(p):
            if not r.get("joints"):
                continue
            J = [r["joints"][f][j] * MP.JOINT_SIGN[(f, j)] for f in TC.FINGERS for j in ("yaw", "mcp", "pip")]
            rows.append(dict(t_ms=1e3 * r["t"], t=None, joints=J, step=step,
                             obj=None if not r.get("obj") else {"p_PO": r["obj"]["p_P"]}))
    t_rest = json.loads(PICKUP["rest"].read_text())["t"]
    for r in rows:
        r["t"] = round(r["t_ms"] / 1e3 - t_rest, 3)                # s from the rest, the close's start
    jr = min(rows, key=lambda r: abs(r["t"]))["joints"]
    sel = [r for r in rows if r["t"] >= 0.0]
    plan = json.loads(PICKUP["plan"].read_text())
    d6 = plan["meta"]["hom"]["d6_P_mm"]
    feed = TC.MuxLog(PICKUP["mux"], t_rest - 6.0, sel[-1]["t_ms"] / 1e3 + 1.0)
    info, out = TC.replay(sel, feed, d6, t_rest, 0.6, "sphere", R_SPHERE, rest_joints=jr)
    res = {"telemetry": [str(p.relative_to(ROOT)) for p in PICKUP["telemetry"]], "mux": str(PICKUP["mux"].relative_to(ROOT)),
           "rest": "0.6 s ending at pinch_pads_rest.json's time (the close's start), the nearest row's joints",
           "summary": summary(out, info, sel, d6, "sphere", R_SPHERE)}
    by_step = {}
    for o, r in zip(out, sel):
        for f in ("thumb", "middle"):
            if o["tips"].get(f):
                by_step.setdefault(r["step"], {}).setdefault(f, []).append(o["tips"][f]["f_n"])
    res["f_n_by_step_N"] = {s: {f: np.round(np.percentile(v, [10, 50, 90]), 2).tolist() for f, v in d.items()}
                            for s, d in by_step.items()}
    with open(LOGS / "20261117-pickup_154328.jsonl", "w") as fh:
        fh.write(json.dumps({"kind": "header", "rest": info}) + "\n")
        for o in out:
            fh.write(json.dumps(o) + "\n")
    return res


def server_rows(path):
    L = read_jsonl(path)
    return L[0], {r["seq"]: r for r in L if "seq" in r}


def compare(A, B):
    ta = {s: np.asarray(r["targets_sim_deg"], float) for s, r in A.items()}
    tb = {s: np.asarray(r["targets_sim_deg"], float) for s, r in B.items()}
    common = sorted(set(ta) & set(tb))
    D = np.array([np.abs(ta[s] - tb[s]) for s in common])
    return dict(periods=len(common), max_deg=round(float(D.max()), 4), median_deg=round(float(np.median(D)), 4),
                rms_deg_per_joint=np.round(np.sqrt((D ** 2).mean(0)), 3).tolist(),
                share_periods_over_1deg=round(float((D.max(1) > 1.0).mean()), 3))


def cmd_hom(a):
    out = LOGS / "hom_dry"
    runs = {}
    for name, tips, flags in (("off_nocop", "off", ["--no-tips"]), ("off_cop", "off", []), ("on_cop", "on", [])):
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        srv = subprocess.Popen([str(ROOT / ".venv/bin/python"), "-m", "sr2hand.control.hom_server", "serve", "--task",
                                "press", "--d", "25", "--plant", "measured", "--plan", str(PRESS["plan"]), "--listen",
                                str(a.port), "--reply", f"127.0.0.1:{a.port - 1}", "--log-dir", str(d), "--once", "--tips",
                                tips], cwd=str(ROOT), stdout=open(d / "server.log", "w"), stderr=subprocess.STDOUT)
        try:
            subprocess.run([sys.executable, str(HERE / "hom_trial.py"), "replay", "--corvus", str(PRESS["corvus"]),
                            "--controller", f"127.0.0.1:{a.port}", "--listen", str(a.port - 1), "--out-dir", str(d), "--mux",
                            str(PRESS["mux"]), "--plan", str(PRESS["plan"])] + flags, cwd=str(HERE), check=True,
                           stdout=open(d / "replay.log", "w"), stderr=subprocess.STDOUT, timeout=600)
            srv.wait(timeout=60)
        finally:
            if srv.poll() is None:
                srv.kill()
        runs[name] = server_rows(sorted(d.glob("*_press_d25_trial*.jsonl"), key=lambda p: p.stat().st_mtime)[-1])
    _h, bench = server_rows(PRESS["server"])
    res = {"bench_record": str(PRESS["server"].relative_to(ROOT)),
           "off_nocop_vs_bench": compare(runs["off_nocop"][1], bench),
           "off_cop_vs_off_nocop": compare(runs["off_cop"][1], runs["off_nocop"][1]),
           "on_cop_vs_off_cop": compare(runs["on_cop"][1], runs["off_cop"][1])}
    for name in ("off_cop", "on_cop"):
        src = {f: {} for f in TC.FINGERS}
        for r in runs[name][1].values():
            for f, (s_, _d) in (r.get("gcf") or {}).items():
                src[f][s_] = src[f].get(s_, 0) + 1
        res[f"{name}_gcf_source_periods"] = src
        res[f"{name}_compute_ms_median"] = round(float(np.median([r["compute_ms"] for r in runs[name][1].values()
                                                                   if r.get("compute_ms") is not None
                                                                   and math.isfinite(r["compute_ms"])])), 2)
    on, off = runs["on_cop"][1], runs["off_cop"][1]
    ph = {}
    for s, r in on.items():
        if s in off:
            ph.setdefault(r["phase"], []).append(np.abs(np.asarray(r["targets_sim_deg"]) - np.asarray(off[s]["targets_sim_deg"])))
    res["on_vs_off_max_deg_by_phase"] = {p: dict(periods=len(v), max=np.round(np.max(v, 0), 2).tolist()) for p, v in ph.items()}
    res["joints_order"] = "thumb yaw mcp pip, index yaw mcp pip, middle yaw mcp pip (sim degrees)"
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("check", choices=("press", "pickup", "hom"))
    ap.add_argument("--port", type=int, default=47412, help="hom: the servers' port (the replay listens on port - 1)")
    a = ap.parse_args()
    t0 = time.time()
    res = {"press": cmd_press, "pickup": cmd_pickup, "hom": cmd_hom}[a.check](a)
    res["seconds"] = round(time.time() - t0, 1)
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / f"20261117-{a.check}.json"
    path.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))
    print(f"-> {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
