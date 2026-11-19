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


def run(name, finger, R_PH, others, gp, own):
    head, rows, ts, qs = load(name)
    t_first = rows[0]["t"]
    feed = TC.MuxLog(mux_for(t_first), t_first - REST_BEFORE_S - REST_S - TC.KEEP_S - 1.0, rows[-1]["t"] + 1.0)

    def joints(t):
        q = {f: dict(others[f]) for f in TC.FINGERS}
        q[finger] = {j: float(np.interp(t, ts, qs[:, i])) for i, j in enumerate(("yaw", "mcp", "pip"))}
        return servo33(q)

    tc = TC.TipContacts(feed, np.zeros(6), gp=gp, own_pose=own)
    for tip in tc.tips.values():
        tip.threshold, tip.min_force = -1e9, -1e9
    rest_end = t_first - REST_BEFORE_S
    info = tc.rest(rest_end, REST_S, refit=True, joints_servo=joints(rest_end))
    out = []
    for r in rows:
        F = np.asarray(r["dF_P"] if "dF_P" in r else R_PH @ np.asarray(r["dF_N"], float), float)
        fs, fn = [], []
        for t in r["t"] - 1.0 + (np.arange(N_EVAL) + 0.5) / N_EVAL:
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


COLS = ["k", "phase", "error_deg", "reading_deg", "hexe_x_P_N", "hexe_y_P_N", "hexe_z_P_N", "board_x_P_N", "board_y_P_N",
        "board_z_P_N", "board_n_N", "found_of_10"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(STUDY / "data/s3/20261118-board_vs_hexe.json"))
    ap.add_argument("--only", nargs="*", default=None, help="run these press stems and replace them in --out")
    a = ap.parse_args()
    R_PH = np.asarray(json.loads((LOGS / f"{R_PH_FROM}.jsonl").read_text().splitlines()[0])["R_PH"], float)
    res = {"what": __doc__.split("\n\n")[0].replace("\n", " "),
           "script": "docs/experiments/20261117-sr2_system_identification/sim/20261118-s3_board_vs_hexe.py",
           "mux": [str(f.relative_to(ROOT)) for f in MUXES], "cols": COLS, "presses": []}
    if a.only and Path(a.out).exists():
        res["presses"] = [p for p in json.loads(Path(a.out).read_text())["presses"]
                          if Path(p["file"]).name[:-6] not in a.only]
    for name, finger, appr in presses():
        if a.only is not None and name not in a.only:
            continue
        others = json.loads(next((STUDY / "data/s3").glob(f"{appr}_approach_*.json")).read_text())["joints"]
        rec = {"file": f"logs/20261118-sr2_s3_press/{name}.jsonl", "finger": finger, "approach": appr, "estimators": {}}
        for en, (gp, own) in ESTIMATORS.items():
            head, info, out = run(name, finger, R_PH, others, gp, own)
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
    Path(a.out).write_text(json.dumps(res, separators=(",", ":")) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
