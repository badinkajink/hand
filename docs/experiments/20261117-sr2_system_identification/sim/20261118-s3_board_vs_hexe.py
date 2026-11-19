"""The fingertip boards against the HEX-E in S3's presses (W2 of the identification queue, 2026-11-18): does a board's
force estimate (tip_contacts, as in the night's trials) read the force that the HEX-E measures?

Per step of the press files (lab/s3_press.py; logs/20261118-sr2_s3_press/, data/s3/): the HEX-E's force change in P from
before the first step, x_P and y_P only (z drifts about 0.4 N a minute; NOTES, 2026-11-18 afternoon), the mean over the
dwell's last second. The MCP press's rows predate dF_P, so their dF_N turn into P with the yaw press's R_PH (the same
press pose). Beside it the board's estimate from the splitter's log (logs/20261118-three_finger_mux/
20261118-155402-mux.bin), averaged over the same second at ten times, at the index's measured joints (the station's
samples in the raw file; thumb and middle at the approach plan's joints). The rest is the second that ends 2.2 s before
the first step's row, with the pad off the plate. The estimator's threshold and force gate are off, so a step has
an estimate whenever the board's model finds a contact. Three estimators: the trials' (the board GP with the own-pose
term), the GP without it, and the physical model alone.

The presses: the index's MCP press and its yaw press with the MCP holding (15:54-16:06, the pad's face on the taped
plate), and the fourteen head-on yaw presses of 16:33-17:17 (lab/s3_yaw_summary.py), in which the side of the index's,
middle's or thumb's tip met the taped plate (-1) or the tag plate (+1); on the -1 side the DIP link's flat side, not
the pad, met the plate (NOTES, 16:46-17:18). Then the middle's MCP press of its pad face (18:36) and the angle presses
of 18:40-18:45: the index's and middle's pad faces pushed by the MCP with the PIP pre-bent 15 and 30 deg. The night's
presses of 19:43-21:39: three PIP presses each for the index and middle (taped plate), three MCP and three PIP presses
for the thumb (tag plate). Each press takes the other fingers' joints from its approach plan (the first stamp in its
note; for the night's presses, whose notes name none, the latest approach before the press; else the press before it),
and the boards' records from the splitter's file that was running at the time.

Per press and estimator: the steps (columns in "cols"), and least-squares lines from the HEX-E's force to the board's:
its normal force and its |f_xy| against the HEX-E's |F_xy|, and per axis of P against the HEX-E's component.

    resguard run --mem 3G -- .venv/bin/python sim/20261118-s3_board_vs_hexe.py [--out data/s3/20261118-board_vs_hexe.json]

With --day 20261119: the six angle presses of 2026-11-19 14:27-14:37 (the MCP pressing the pad face on, the PIP pre-bent
15 and 30 deg; the thumb on the tag plate, the index and middle on the taped plate), at the roll layout's mounts, into
data/s3/20261119-board_vs_hexe.json. Their steps dwell 0.5 s (the first index press 1 s), so a step's board estimate
is the mean over its dwell less 0.05 s. A fourth estimator replaces the trials' own-pose GP with the one fitted on the
sweep of 2026-11-17 at the same layout (85, 170): "gp_own_pose_x85y170".
"""
import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[2]
sys.path[:0] = [str(ROOT / "src/sr2hand/driver/reskin5x/host"), str(ROOT / "src"), str(ROOT / "src/sr2hand/driver/manta/host")]
import tip_contacts as TC  # noqa: E402
from manta_hand import plan as MP  # noqa: E402

LOGS = ROOT / "logs/20261118-sr2_s3_press"
MUXES = sorted((ROOT / "logs/20261118-three_finger_mux").glob("20261118-*-mux.bin"))
R_PH_FROM = "20261118-160047_index_yaw"           # the MCP press's rows predate dF_P; its pose is the yaw press's
ESTIMATORS = {"gp_own_pose": (True, True), "gp": (True, False), "physical": (False, False)}
OWN_X85Y170 = TC.OWN_POSE.parent / "20261117-own_pose_gp_sweep_x85y170_7v5.npz"
DRAGS19 = [("20261119-143939_index_drag1N_w", "index", "20261119-143821"),
           ("20261119-144044_index_drag2N_w", "index", "20261119-143821"),
           ("20261119-144343_middle_drag1N_w", "middle", "20261119-144227"),
           ("20261119-144447_middle_drag2N_w", "middle", "20261119-144227"),
           ("20261119-144805_thumb_drag1N_w", "thumb", "20261119-144645"),
           ("20261119-144916_thumb_drag2N_w", "thumb", "20261119-144645")]
DRAG_DT = 0.2
DAY19 = [("20261119-142715_index_mcp", "index", "20261119-142629"), ("20261119-142936_index_mcp", "index", "20261119-142905"),
         ("20261119-143129_middle_mcp", "middle", "20261119-143045"),
         ("20261119-143251_middle_mcp", "middle", "20261119-143219"),
         ("20261119-143453_thumb_mcp", "thumb", "20261119-143406"), ("20261119-143659_thumb_mcp", "thumb", "20261119-143610")]
REST_BEFORE_S, REST_S = 2.2, 1.0
N_EVAL = 10


def servo33(sim):
    """{finger: {yaw, mcp, pip}} in simulator degrees -> servo degrees (3, 3) (tip_contacts.sim_deg inverted)."""
    col = {"aa": 0, "fe1": 1, "fe2": 2}
    J = np.zeros((3, 3))
    for k, f in enumerate(TC.FINGERS):
        for j in ("yaw", "mcp", "pip"):
            J[k, col[MP.SIM_JOINT_TO_SERVO[j]]] = sim[f][j] * MP.JOINT_SIGN[(f, j)]
    return J


def load(name):
    lines = [json.loads(x) for x in (LOGS / f"{name}.jsonl").read_text().splitlines() if x.strip()]
    head, rows = lines[0], [x for x in lines[1:] if "k" in x]
    st = []
    for x in (LOGS / f"{name}_raw.jsonl").read_text().splitlines():
        if x.strip():
            st += [(s[0], s[1]) for s in json.loads(x)["station"]]
    st.sort(key=lambda s: s[0])
    return head, rows, np.array([s[0] for s in st]), np.array([[s[1][j] for j in ("yaw", "mcp", "pip")] for s in st])


def fit(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return None
    A = np.c_[x[ok], np.ones(ok.sum())]
    (a, b), *_ = np.linalg.lstsq(A, y[ok], rcond=None)
    r = y[ok] - A @ [a, b]
    return {"slope": round(float(a), 3), "offset_N": round(float(b), 3), "rms_N": round(float(np.sqrt(np.mean(r ** 2))), 3),
            "n": int(ok.sum())}


def presses():
    """[(log stem, finger, approach plan)] in time order: the index's two presses, then the head-on yaw presses."""
    out = [("20261118-155403_index_mcp", "index", "20261118-155238"), ("20261118-160047_index_yaw", "index", "20261118-155238")]
    last = None
    later = sorted([*(STUDY / "data/s3").glob("20261118-*_yaw_head_on.jsonl"),
                    *(STUDY / "data/s3").glob("20261118-1[89]*_mcp_press.jsonl"),
                    *(STUDY / "data/s3").glob("20261118-2*_mcp_press.jsonl"),
                    *(STUDY / "data/s3").glob("20261118-*_pip_press.jsonl"),
                    *(STUDY / "data/s3").glob("20261118-18*_mcp_pip_prebend.jsonl")])
    approaches = sorted(p.name[:15] for p in (STUDY / "data/s3").glob("20261118-*_approach_*.json"))
    for p in later:
        h = json.loads(p.read_text().splitlines()[0])
        m = re.search(r"(\d{8}-\d{6})", h.get("note") or "")
        if m:
            last = m.group(1)
        elif p.name[:9] == "20261118-" and p.name[9:11] in ("19", "20", "21"):
            last = max(a for a in approaches if a < p.name[:15])      # the night's notes name no approach
        stem = re.sub(r"_(head_on|press|pip_prebend)\.jsonl$", "", p.name)
        out.append((stem, h["finger"], last))
    return out


def mux_for(t):
    """The splitter's file that was recording at t (epoch s): the latest one started before it."""
    starts = [dt.datetime.strptime(f.name[:15], "%Y%m%d-%H%M%S").timestamp() for f in MUXES]
    k = max(i for i, s0 in enumerate(starts) if s0 <= t)
    return MUXES[k]


def run(name, finger, R_PH, others, gp, own, own_path=None):
    head, rows, ts, qs = load(name)
    t_first = rows[0]["t"]
    mux = ROOT / head["mux"] if LOGS.name.startswith("20261119") else mux_for(t_first)
    feed = TC.MuxLog(mux, t_first - REST_BEFORE_S - REST_S - TC.KEEP_S - 1.0, rows[-1]["t"] + 1.0)
    win = min(1.0, float(head.get("dwell_s", 1.0))) - (0.05 if head.get("dwell_s", 1.0) < 1.0 else 0.0)

    def joints(t):
        q = {f: dict(others[f]) for f in TC.FINGERS}
        q[finger] = {j: float(np.interp(t, ts, qs[:, i])) for i, j in enumerate(("yaw", "mcp", "pip"))}
        return servo33(q)

    tc = TC.TipContacts(feed, np.zeros(6), gp=gp, own_pose=own)
    for tip in tc.tips.values():
        tip.threshold, tip.min_force = -1e9, -1e9
        if own and own_path is not None:
            tip.own = TC.OwnPose(own_path)
    rest_end = t_first - REST_BEFORE_S
    info = tc.rest(rest_end, REST_S, refit=True, joints_servo=joints(rest_end))
    out = []
    for r in rows:
        F = np.asarray(r["dF_P"] if "dF_P" in r else R_PH @ np.asarray(r["dF_N"], float), float)
        fs, fn = [], []
        for t in r["t"] - win + win * (np.arange(N_EVAL) + 0.5) / N_EVAL:
            J = joints(t)
            est = tc.estimate(t, J)
            _cop, rec = tc.to_P(est, J)
            e = rec.get(finger)
            if e is not None and e.get("f_P") is not None:
                fs.append(e["f_P"])
                fn.append(e["f_n"])
        b = np.mean(fs, 0) if fs else [None] * 3
        out.append([r["k"], r["phase"], round(r["offset"], 2), round(r["reading"], 2), *np.round(F, 3).tolist(),
                     *[None if v is None else round(float(v), 3) for v in b],
                     round(float(np.mean(fn)), 3) if fn else None, len(fn)])
    return head, info, out


def run_drag(name, finger, others, gp, own, own_path=None):
    """A shear drag (the pad held on the plate, the arm moving it across the finger): every DRAG_DT s through the legs,
    the HEX-E's force in P (its mean over the interval, from the run's F0) beside the board's estimate (the mean of five
    evaluations in the interval)."""
    lines = [json.loads(x) for x in (LOGS / f"{name}.jsonl").read_text().splitlines() if x.strip()]
    head, legs = lines[0], [x for x in lines if x.get("phase") == "leg"]
    raw = [json.loads(x) for x in (LOGS / f"{name}_raw.jsonl").read_text().splitlines() if x.strip()]
    th = np.concatenate([np.array(r["hexe_t"], float) for r in raw])
    FH = np.concatenate([np.array(r["hexe"], float).reshape(-1, 6)[:, :3] for r in raw])
    o = np.argsort(th)
    th, FP = th[o], (FH[o] - np.array(head["F0_N"])) @ np.array(head["R_PH"]).T
    st = sorted([(q[0], q[1]) for r in raw for q in r.get("station", [])], key=lambda q: q[0])
    ts = np.array([q[0] for q in st])
    qs = np.array([[q[1][j] for j in ("yaw", "mcp", "pip")] for q in st])
    t_first = legs[0]["t0"]
    t_rest = float(head["t"]) + 0.5
    feed = TC.MuxLog(ROOT / head["mux"] if head.get("mux") else mux_for(t_first),
                     t_rest - REST_S - TC.KEEP_S - 1.0, legs[-1]["t1"] + 1.0)

    def joints(t):
        q = {f: dict(others[f]) for f in TC.FINGERS}
        q[finger] = {j: float(np.interp(t, ts, qs[:, i])) for i, j in enumerate(("yaw", "mcp", "pip"))}
        return servo33(q)

    tc = TC.TipContacts(feed, np.zeros(6), gp=gp, own_pose=own)
    for tip in tc.tips.values():
        tip.threshold, tip.min_force = -1e9, -1e9
        if own and own_path is not None:
            tip.own = TC.OwnPose(own_path)
    info = tc.rest(t_rest + REST_S, REST_S, refit=True, joints_servo=joints(t_rest))
    out = []
    for lg in legs:
        for t0 in np.arange(lg["t0"], lg["t1"] - DRAG_DT, DRAG_DT):
            m = (th >= t0) & (th < t0 + DRAG_DT)
            if m.sum() < 3:
                continue
            F = FP[m].mean(0)
            fs, fn = [], []
            for t in t0 + DRAG_DT * (np.arange(5) + 0.5) / 5:
                J = joints(t)
                _cop, rec = tc.to_P(tc.estimate(t, J), J)
                e = rec.get(finger)
                if e is not None and e.get("f_P") is not None:
                    fs.append(e["f_P"])
                    fn.append(e["f_n"])
            b = np.mean(fs, 0) if fs else [None] * 3
            out.append([round(float(t0 - t_first), 2), lg["to_mm"], *np.round(F, 3).tolist(),
                        *[None if v is None else round(float(v), 3) for v in b],
                        round(float(np.mean(fn)), 3) if fn else None, len(fn)])
    return head, info, out


DRAG_COLS = ["t_s", "leg_to_mm", "hexe_x_P_N", "hexe_y_P_N", "hexe_z_P_N", "board_x_P_N", "board_y_P_N", "board_z_P_N",
             "board_n_N", "found_of_5"]


COLS = ["k", "phase", "error_deg", "reading_deg", "hexe_x_P_N", "hexe_y_P_N", "hexe_z_P_N", "board_x_P_N", "board_y_P_N",
        "board_z_P_N", "board_n_N", "found_of_10"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(STUDY / "data/s3/20261118-board_vs_hexe.json"))
    ap.add_argument("--only", nargs="*", default=None, help="run these press stems and replace them in --out")
    ap.add_argument("--day", default="20261118", choices=["20261118", "20261119"])
    a = ap.parse_args()
    global LOGS, MUXES
    estimators = dict(ESTIMATORS)
    if a.day == "20261119":
        LOGS = ROOT / "logs/20261119-sr2_s3_press"
        MUXES = sorted((ROOT / "logs/20261119-three_finger_mux").glob("20261119-*-mux.bin"))
        estimators["gp_own_pose_x85y170"] = (True, True)
        if a.out.endswith("20261118-board_vs_hexe.json"):
            a.out = str(STUDY / "data/s3/20261119-board_vs_hexe.json")
    R_PH = np.asarray(json.loads((ROOT / "logs/20261118-sr2_s3_press" / f"{R_PH_FROM}.jsonl").read_text()
                                 .splitlines()[0])["R_PH"], float)
    res = {"what": __doc__.split("\n\n")[0].replace("\n", " "),
           "script": "docs/experiments/20261117-sr2_system_identification/sim/20261118-s3_board_vs_hexe.py",
           "mux": [str(f.relative_to(ROOT)) for f in MUXES], "cols": COLS, "presses": []}
    if a.only and Path(a.out).exists():
        res["presses"] = [p for p in json.loads(Path(a.out).read_text())["presses"]
                          if Path(p["file"]).name[:-6] not in a.only]
    for name, finger, appr in (DAY19 if a.day == "20261119" else presses()):
        if a.only is not None and name not in a.only:
            continue
        adir = LOGS if a.day == "20261119" else STUDY / "data/s3"
        others = json.loads(next(adir.glob(f"{appr}_approach_*.json")).read_text())["joints"]
        rec = {"file": f"{LOGS.relative_to(ROOT)}/{name}.jsonl", "finger": finger, "approach": appr, "estimators": {}}
        for en, (gp, own) in estimators.items():
            head, info, out = run(name, finger, R_PH, others, gp, own,
                                  OWN_X85Y170 if en == "gp_own_pose_x85y170" else None)
            rec["sign"], rec["joint"] = head.get("sign"), head.get("joint")
            have = [o for o in out if o[11]]
            hx = np.array([o[4:7] for o in have], float).reshape(-1, 3)
            bx = np.array([o[7:10] for o in have], float).reshape(-1, 3)
            fits = {"n_vs_xy": fit(np.hypot(hx[:, 0], hx[:, 1]), [o[10] for o in have]),
                    "fxy_vs_xy": fit(np.hypot(hx[:, 0], hx[:, 1]), np.hypot(bx[:, 0], bx[:, 1])),
                    "x_P": fit(hx[:, 0], bx[:, 0]), "y_P": fit(hx[:, 1], bx[:, 1])}
            rec["estimators"][en] = {"rest": info, "fits": fits, "steps": out}
            top = max(out, key=lambda o: np.hypot(o[4], o[5]))
            print(f"{name:26s} {en:12s} steps {len(out)}, with an estimate {len(have)}; at the largest HEX-E force "
                  f"{np.round(top[4:7], 2).tolist()} the board {top[7:10]} n {top[10]}; fit n {fits['n_vs_xy']}", flush=True)
        res["presses"].append(rec)
    res["presses"].sort(key=lambda p: Path(p["file"]).name)
    if a.day == "20261119" and a.only is None:
        res["drag_cols"], res["drags"] = DRAG_COLS, []
        for name, finger, appr in DRAGS19:
            others = json.loads(next(LOGS.glob(f"{appr}_approach_*.json")).read_text())["joints"]
            rec = {"file": f"{LOGS.relative_to(ROOT)}/{name}.jsonl", "finger": finger, "approach": appr,
                   "estimators": {}}
            for en, (gp, own) in estimators.items():
                head, info, out = run_drag(name, finger, others, gp, own,
                                           OWN_X85Y170 if en == "gp_own_pose_x85y170" else None)
                have = [o for o in out if o[9]]
                hx = np.array([o[2:5] for o in have], float).reshape(-1, 3)
                bx = np.array([o[5:8] for o in have], float).reshape(-1, 3)
                fits = {"normal_x_P": fit(np.abs(hx[:, 0]), [o[8] for o in have]),
                        "shear_y_P": fit(hx[:, 1], bx[:, 1]), "shear_z_P": fit(hx[:, 2], bx[:, 2])}
                rec["estimators"][en] = {"rest": info, "fits": fits, "samples": out}
                print(f"{name:30s} {en:20s} samples {len(out)} with an estimate {len(have)}; normal {fits['normal_x_P']}; "
                      f"shear across (y_P) {fits['shear_y_P']}", flush=True)
            res["drags"].append(rec)
    Path(a.out).write_text(json.dumps(res, separators=(",", ":")) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
