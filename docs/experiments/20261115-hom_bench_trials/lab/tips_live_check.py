"""The fingertip path of hom_trial.py run without hardware (2026-11-17, before the first trials with the fingertips).

A recorded splitter log is played into pseudo-terminals of a folder of its own (tip_contacts.play), a stand-in station
answers GET /api/v1/state with a recorded trial's joints at the log's pace, a HOM server answers on this host, and
hom_trial.py trial runs on them with --no-tracker: the boards' streams are checked, the rests taken, the estimator's
thread runs, every state sends StateMsg.cop and the records are written as in a trial. Without the tracker the server
has no object pose and holds the open pose, so HOM itself is tested by hom_trial.py replay. Nothing here reads the mux,
the cameras, the station or the arm.

    .venv/bin/python lab/tips_live_check.py --corvus runs_corvus/20261116-131927_press_corvus.jsonl \\
        --mux ../../../logs/20261116-three_finger_mux/20261116-131831-mux.bin \\
        --plan plans/20261116-press_x70_y110_lat0_plan.json --task press --d 25 \\
        --out-dir ../../../logs/20261117-tip_contacts/live_check [--trial-python /usr/bin/python3] [--cpu 5]
"""
import argparse
import http.server
import json
import os
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "src/sr2hand/driver/reskin5x/host")]
import tip_contacts as TC  # noqa: E402


class Joints:
    """The recorded trial's joints (servo degrees) at the log's time t: the row at or before t, the first before it."""

    def __init__(self, rows):
        ok = [r for r in rows if r.get("joints")]
        self.t = np.array([r["t_ms"] / 1000.0 for r in ok])
        self.j = np.array([r["joints"] for r in ok], float)

    def at(self, t):
        k = max(0, int(np.searchsorted(self.t, t, side="right")) - 1)
        return self.j[k].reshape(3, 3)


def station(port, joints, clock):
    """A stand-in for the CB1's station: GET /api/v1/state with the recorded joints (servo_age_s 0)."""
    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            J = joints.at(clock())
            body = json.dumps({"telemetry": {"servos": {str(k): dict(zip(("aa", "fe1", "fe2"), map(float, J[k])))
                                                        for k in range(3)}, "servo_age_s": 0.0,
                                             "servo_polling_suspended": False}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass
    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address, daemon_threads = True, True     # the poll's closed connections leave the port in TIME_WAIT
    srv = Server(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def summary(corvus_path, server_path):
    lines = [json.loads(x) for x in Path(corvus_path).read_text().splitlines() if x.strip()]
    head, rows, end = lines[0], [x for x in lines[1:] if "seq" in x], lines[-1]
    t = np.array([r["t_ms"] for r in rows])
    per = np.diff(t)
    out = {"states": len(rows), "period_ms": np.round(np.percentile(per, [5, 50, 95, 100]), 2).tolist(),
           "rest": {f: {k: v.get(k) for k in ("model", "saved_fit_mT", "new_fit_mT", "q_rest")}
                    for f, v in (head.get("tips") or {}).get("rest", {}).items()},
           "tips_end": end.get("tips")}
    age = np.array([r["tips_age_ms"] for r in rows if r.get("tips_age_ms") is not None])
    out["estimate_age_ms"] = np.round(np.percentile(age, [5, 50, 95, 100]), 1).tolist() if len(age) else None
    for f in TC.FINGERS:
        on = [r["tips"][f] for r in rows if isinstance(r.get("tips"), dict) and r["tips"].get(f)]
        data_age = [r["t_ms"] - 1e3 * r["tips"][f]["t"] for r in rows if isinstance(r.get("tips"), dict) and r["tips"].get(f)]
        out[f] = {"periods_with_contact": len(on), "first_t": next((r["t"] for r in rows if isinstance(r.get("tips"), dict)
                                                                   and r["tips"].get(f)), None),
                  "f_n_N": np.round(np.percentile([x["f_n"] for x in on], [10, 50, 90]), 3).tolist() if on else None,
                  "data_age_ms": np.round(np.percentile(data_age, [50, 95]), 1).tolist() if data_age else None}
    if server_path:
        S = [json.loads(x) for x in Path(server_path).read_text().splitlines() if x.strip()]
        srows = [x for x in S if "seq" in x]
        out["server"] = {"periods": len(srows), "tips": S[0].get("tips"),
                         "contact_force_min_N": S[0].get("contact_force_min_N"),
                         "periods_with_contacts": {f: sum(1 for r in srows if f in (r.get("contacts") or {}))
                                                   for f in TC.FINGERS}}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--corvus", required=True, help="a recorded trial (hom_trial.py's corvus record)")
    ap.add_argument("--mux", required=True, help="the splitter's .bin over that trial")
    ap.add_argument("--plan", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--d", type=float, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--lead-s", type=float, default=4.0, help="s of the log played before the recorded t = 0")
    ap.add_argument("--seconds", type=float, default=7.0, help="the trial's length (hom_trial.py --seconds)")
    ap.add_argument("--trial-python", default=sys.executable, help="the Python of the trial (corvus: /usr/bin/python3)")
    ap.add_argument("--cpu", type=int, default=None, help="pin the trial to this core (taskset), as on a loaded corvus")
    ap.add_argument("--station-port", type=int, default=47390)
    ap.add_argument("--server-port", type=int, default=47392)
    ap.add_argument("--trial-port", type=int, default=47391)
    ap.add_argument("--server-tips", default="on", choices=("off", "on"))
    ap.add_argument("--build-s", type=float, default=20.0, help="s the server gets to build its model before the trial")
    a = ap.parse_args()
    for k in ("corvus", "mux", "plan", "out_dir"):           # the server and the trial run in other folders
        setattr(a, k, str(Path(getattr(a, k)).resolve()))
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lines = [json.loads(x) for x in Path(a.corvus).read_text().splitlines() if x.strip()]
    head, rows = lines[0], [x for x in lines[1:] if "seq" in x]
    t_from = head["t0_ms"] / 1000.0 - a.lead_s
    links = Path(tempfile.mkdtemp(prefix="tips_live_check_"))
    srv_log = out / "server"
    server = subprocess.Popen([str(ROOT / ".venv/bin/python"), "-m", "sr2hand.control.hom_server", "serve", "--task",
                               a.task, "--d", str(a.d), "--plant", "measured", "--plan", a.plan,
                               "--listen", str(a.server_port), "--reply", f"127.0.0.1:{a.trial_port}", "--log-dir",
                               str(srv_log), "--once", "--tips", a.server_tips], cwd=str(ROOT),
                              env=dict(os.environ, PYTHONPATH=str(ROOT / "src")))
    stop = threading.Event()
    wall0 = []
    player = threading.Thread(target=lambda: (wall0.append(time.time()), TC.play(a.mux, t_from, a.lead_s + a.seconds + 6.0,
                                                                                   links, stop, quiet=True)), daemon=True)
    jt = Joints(rows)
    st = None
    try:
        st = station(a.station_port, jt, lambda: t_from + (time.time() - wall0[0]) if wall0 else t_from)
        time.sleep(a.build_s)                             # the server builds its model
        player.start()
        time.sleep(0.5)
        cmd = ([] if a.cpu is None else ["taskset", "-c", str(a.cpu)]) + [
            a.trial_python, str(HERE / "hom_trial.py"), "trial", "--task", a.task, "--controller",
            f"127.0.0.1:{a.server_port}", "--listen", str(a.trial_port), "--object",
            "cylinder" if a.task in ("press", "pinpush") else "sphere", "--no-tracker", "--no-video", "--station",
            f"http://127.0.0.1:{a.station_port}", "--plan", a.plan, "--out-dir", str(out),
            "--tips-links", str(links), "--seconds", str(a.seconds), "--lift-mm", "0"]
        t0 = time.time()
        r = subprocess.run(cmd, cwd=str(HERE), capture_output=True, text=True, timeout=120,
                           env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"))
        print(r.stdout[-3000:], r.stderr[-3000:], f"(trial process {time.time() - t0:.1f} s, exit {r.returncode})",
              flush=True)
    finally:
        stop.set()
        if st is not None:
            st.shutdown()
            st.server_close()
        try:
            server.wait(timeout=20)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=5)
    corvus = sorted(out.glob("*_corvus.jsonl"), key=lambda p: p.stat().st_mtime)[-1]
    srv = sorted(srv_log.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    res = summary(corvus, srv[-1] if srv else None)
    res.update(corvus=str(corvus), server_record=str(srv[-1]) if srv else None, trial_python=a.trial_python, cpu=a.cpu)
    (out / f"{corvus.stem}_check.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
