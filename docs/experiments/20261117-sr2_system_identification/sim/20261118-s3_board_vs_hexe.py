"""The index fingertip's board against the HEX-E in S3's two index presses (W2 of the identification queue, 2026-11-18):
does the board's force estimate (tip_contacts, as in the night's trials) read the force that the HEX-E measures?

Per step of the press files (lab/s3_press.py; logs/20261118-sr2_s3_press/, data/s3/): the HEX-E's force change in P from
before the first step, x_P and y_P only (z drifts about 0.4 N a minute; NOTES, 2026-11-18 afternoon), the mean over the
dwell's last second. The MCP press's rows predate dF_P, so their dF_N turn into P with the yaw press's R_PH (the same
press pose). Beside it the board's estimate from the splitter's log (logs/20261118-three_finger_mux/
20261118-155402-mux.bin), averaged over the same second at ten times, at the index's measured joints (the station's
samples in the raw file; thumb and middle at the approach plan's joints). The rest is the second that ends 2.2 s before
the first step's row, with the pad 4.6 mm off the plate. The estimator's threshold and force gate are off, so a step has
an estimate whenever the board's model finds a contact. Three estimators: the trials' (the board GP with the own-pose
term), the GP without it, and the physical model alone.

Per press and estimator: the steps' rows, and a least-squares line from the HEX-E's force to the board's: its normal
force against the HEX-E's |F_xy| (the MCP press, the pad pushed head on), and per axis of P against the HEX-E's
component (the yaw press: x_P the MCP's hold, y_P the yaw's tangential force).

    resguard run --mem 3G -- .venv/bin/python sim/20261118-s3_board_vs_hexe.py [--out data/s3/20261118-board_vs_hexe.json]
"""
import argparse
import json
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
MUX = ROOT / "logs/20261118-three_finger_mux/20261118-155402-mux.bin"
APPROACH = STUDY / "data/s3/20261118-155238_approach_index.json"
PRESSES = {"mcp": "20261118-155403_index_mcp", "yaw": "20261118-160047_index_yaw"}
FINGER = "index"
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


def run(name, R_PH, others, gp, own):
    head, rows, ts, qs = load(name)
    t_first = rows[0]["t"]
    feed = TC.MuxLog(MUX, t_first - REST_BEFORE_S - REST_S - TC.KEEP_S - 1.0, rows[-1]["t"] + 1.0)

    def joints(t):
        q = {f: dict(others[f]) for f in TC.FINGERS}
        q[FINGER] = {j: float(np.interp(t, ts, qs[:, i])) for i, j in enumerate(("yaw", "mcp", "pip"))}
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
            e = rec.get(FINGER)
            if e is not None and e.get("f_P") is not None:
                fs.append(e["f_P"])
                fn.append(e["f_n"])
        out.append({"k": r["k"], "phase": r["phase"], "joint": r["joint"], "t": r["t"], "offset_deg": r["offset"],
                    "reading_deg": r["reading"], "hexe_P_N": np.round(F, 3).tolist(),
                    "hexe_xy_N": round(float(np.hypot(F[0], F[1])), 3),
                    "board_P_N": np.round(np.mean(fs, 0), 3).tolist() if fs else None,
                    "board_n_N": round(float(np.mean(fn)), 3) if fn else None, "board_found": f"{len(fn)}/{N_EVAL}"})
    return head, info, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(STUDY / "data/s3/20261118-board_vs_hexe.json"))
    a = ap.parse_args()
    R_PH = np.asarray(json.loads((LOGS / f"{PRESSES['yaw']}.jsonl").read_text().splitlines()[0])["R_PH"], float)
    others = json.loads(APPROACH.read_text())["joints"]
    res = {"what": __doc__.split("\n\n")[0].replace("\n", " "),
           "script": "docs/experiments/20261117-sr2_system_identification/sim/20261118-s3_board_vs_hexe.py",
           "mux": str(MUX.relative_to(ROOT)), "presses": {}}
    for press, name in PRESSES.items():
        res["presses"][press] = {"file": f"logs/20261118-sr2_s3_press/{name}.jsonl", "estimators": {}}
        for en, (gp, own) in ESTIMATORS.items():
            head, info, out = run(name, R_PH, others, gp, own)
            have = [o for o in out if o["board_P_N"] is not None]
            hx = np.array([o["hexe_P_N"] for o in have]) if have else np.zeros((0, 3))
            bx = np.array([o["board_P_N"] for o in have]) if have else np.zeros((0, 3))
            fits = {"n_vs_xy": fit([o["hexe_xy_N"] for o in have], [o["board_n_N"] for o in have]),
                    "x_P": fit(hx[:, 0], bx[:, 0]) if len(have) else None,
                    "y_P": fit(hx[:, 1], bx[:, 1]) if len(have) else None}
            res["presses"][press]["estimators"][en] = {"rest": info, "fits": fits, "steps": out}
            print(f"{press:4s} {en:12s} steps {len(out)}, with an estimate {len(have)}; fits {json.dumps(fits)}",
                  flush=True)
            for o in out[::max(1, len(out) // 12)]:
                print(f"    k {o['k']:3d} {o['phase']:5s} {o['joint']:3s} offset {o['offset_deg']:+5.1f} HEX-E "
                      f"{o['hexe_P_N']} | board {o['board_P_N']} n {o['board_n_N']} ({o['board_found']})", flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
