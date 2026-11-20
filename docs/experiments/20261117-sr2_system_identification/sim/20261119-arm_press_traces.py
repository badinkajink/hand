"""Per-bin traces of chosen arm presses (2026-11-19) for the page's figure: the HEX-E's force along the press and the
board's change from its rest (|d| and its part along the press's own field per newton g), 0.1 s bins, from the raw
records in logs/ and the splitter's file (as sim/20261119-arm_press_boards.py forms them).

    /usr/bin/python3 sim/20261119-arm_press_traces.py [--out data/s3/20261119-arm_press_traces.json]
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("apb", HERE / "20261119-arm_press_boards.py")
APB = importlib.util.module_from_spec(spec)
spec.loader.exec_module(APB)
TC = APB.TC
PICK = [("20261119-170343_index_arm_press.jsonl", 1), ("20261119-171923_index_arm_press.jsonl", 1),
        ("20261119-172508_index_arm_press.jsonl", 1), ("20261119-170846_middle_arm_press.jsonl", 1),
        ("20261119-171311_thumb_arm_press.jsonl", 2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE.parent / "data/s3/20261119-arm_press_traces.json"))
    a = ap.parse_args()
    G = {(r["run"], p["rep"]): np.array(p["g"]) for r in json.loads((HERE.parent / "data/s3/20261119-arm_press_boards.json")
                                                                     .read_text())["runs"] for p in r["presses"] if "g" in p}
    out = []
    for name, rep_no in PICK:
        run = APB.LOGS / name
        lines = [json.loads(x) for x in open(run) if x.strip()]
        head = lines[0]
        rep = [x for x in lines[1:] if x.get("rep") == rep_no][0]
        raw = [json.loads(x) for x in open(str(run).replace(".jsonl", "_raw.jsonl")) if x.strip()]
        raw = [x for x in raw if x["rep"] == rep_no][0]
        f = head["finger"]
        d_B = np.asarray(head["press_dir_B"], float)
        R_BH = np.asarray(head["T_BP_end"], float)[:3, :3] @ np.asarray(head["R_PH"], float)
        th = np.asarray(raw["hexe_t"], float)
        F0 = np.asarray(rep["F0"], float)
        Fp = -((np.asarray(raw["hexe"], float)[:, :3] - F0) @ R_BH.T) @ d_B
        t0, t1 = rep["t_start"], rep["t_end"]
        if rep.get("F_after") is not None:
            dr = float(-((np.asarray(rep["F_after"], float) - F0) @ R_BH.T) @ d_B)
            Fp = Fp - dr * np.clip((th - t0) / (t1 - t0), 0, 1)
        feed = TC.MuxLog(APB.mux_for(t0), t0 - 2.0 - TC.KEEP_S, t1 + 1.0)
        tc = TC.TipContacts(feed, np.zeros(6), gp=False, own_pose=False)
        tip = tc.tips[f]
        tc.rest(t0 - 0.1, 0.9, refit=False)
        g = G[(name, rep_no)]
        u = g / np.linalg.norm(g)
        rows = []
        for b in np.arange(t0, t1, 0.1):
            m = (th >= b) & (th < b + 0.1)
            tt, rr = feed.window(f, b + 0.1, 0.1)
            ok = TC.valid(rr)
            if m.sum() < 3 or ok.sum() < 3:
                continue
            d = tip.fields(rr[ok], rr[ok]).mean(0) - tip.base
            rows.append([round(float(b - t0), 2), round(float(Fp[m].mean()), 4), round(float(np.linalg.norm(d)), 2),
                         round(float(d @ u), 2)])
        out.append(dict(run=name, rep=rep_no, finger=f, indenter=head["indenter"], cols=["t_s", "F_N", "d_uT", "d_along_g_uT"],
                        rows=rows))
        print(name, rep_no, len(rows), flush=True)
    Path(a.out).write_text(json.dumps({"what": __doc__.split("\n\n")[0].replace("\n", " "), "traces": out},
                                      separators=(",", ":")) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
