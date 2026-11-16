"""Sweep of the replay plant's parameters over the six real twist trials of 2026-11-16 (sim-real reversal study): each
parameter set replays every trial open loop (sr2_hom_replay.replay) and reports the replay's costs and the trajectory
features of rev_features (rise along z_P, twist and tilt), against the real record's. Workers: a pool, one Replayer each.
    .venv/bin/python logs/20261116-sim_real_reversal/rev_sweep.py OUT.json 'mu=0.3' 'mu=0.45' 'tau_max=0.3 kp=1.0' ...
"""
import json, sys, os
from pathlib import Path
from multiprocessing import Pool
import numpy as np
ROOT = Path.home() / "Programs/hand"
sys.path.insert(0, str(ROOT / "scripts/hardware")); sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "logs/20261116-sim_real_reversal"))
D = ROOT / "logs/20261116-sim_real_reversal"
TRIALS = sorted((D / "trials").glob("merged_*.jsonl"))
_R = {}

def _init():
    import sr2_hom_replay as RP
    from sr2hand.control import trial_score as TS
    _R["RP"], _R["rep"] = RP, RP.Replayer()
    _R["tr"] = {p.name: RP.prepare_trial(*TS.load_trial(str(p))) for p in TRIALS}

def feats_arrays(t, P, Q):
    from scipy.spatial.transform import Rotation as Rot
    ok = np.isfinite(P).all(1) & (np.linalg.norm(Q, axis=1) > 0.5)
    t, P, Q = t[ok], P[ok], Q[ok]
    R = Rot.from_quat(Q[:, [1, 2, 3, 0]]); rv = (R * R[0].inv()).as_rotvec()
    dz = P[:, 2] - P[0, 2]; lat = np.hypot(*(P[:, :2] - P[0, :2]).T)
    return {"rise_max": float(dz.max()), "rise_end": float(dz[-1]), "lat_end": float(lat[-1]),
            "twist_end": float(np.degrees(rv[-1, 2])), "tilt_end": float(np.degrees(np.linalg.norm(rv[-1, :2]))),
            "dropped": bool(dz[-1] < -30 or lat[-1] > 60)}

def job(args):
    name, params = args
    RP, rep, tr = _R["RP"], _R["rep"], _R["tr"][name]
    r = rep.replay(tr, params)
    out = {"trial": name, "params": params, "diverged": r["diverged"]}
    if r["diverged"]:
        return out
    out.update({k: round(r[k], 3) for k in ("joint_rms_deg", "pos_rms_mm", "rot_rms_deg")})
    out["sim"] = feats_arrays(tr["t"], r["pos_sim_mm"], r["quat_sim"])
    out["real"] = feats_arrays(tr["t"], tr["pos_mm"], tr["quat"])
    return out

def parse(s):
    return {kv.split("=")[0]: float(kv.split("=")[1]) for kv in s.split()} if s.strip() else {}

if __name__ == "__main__":
    out = Path(sys.argv[1]); sets = [parse(s) for s in sys.argv[2:]] or [{}]
    jobs = [(p.name, prm) for prm in sets for p in TRIALS]
    with Pool(min(4, len(jobs)), initializer=_init) as pool:
        rows = pool.map(job, jobs)
    out.write_text(json.dumps(rows, indent=1))
    for prm in sets:
        rs = [r for r in rows if r["params"] == prm]
        line = []
        for r in rs:
            if r["diverged"]:
                line.append(f"{r['trial'][7:-6]}: DIVERGED"); continue
            s = r["sim"]
            line.append(f"{r['trial'][7:-6]}: rise {s['rise_max']:5.1f} twist {s['twist_end']:6.1f} tilt {s['tilt_end']:5.1f}"
                        + (" DROP" if s["dropped"] else "") + f" | q {r['joint_rms_deg']:.1f} p {r['pos_rms_mm']:.1f} r {r['rot_rms_deg']:.1f}")
        print(json.dumps(prm)); print("   " + "\n   ".join(line), flush=True)
