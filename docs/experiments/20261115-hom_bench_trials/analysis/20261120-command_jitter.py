"""2026-11-20 11:4x, the owner at the bench: "the controls are way more jittery and cause a ton of drift ... there's no
way this is what simulation predicts or commanded, no?". First look: are the twist trials' commanded targets jittery, or their execution? Per corvus record (cmd rev% = the share of frames where a commanded joint reverses its direction of travel, steps
over 0.1 deg; HF = RMS about a 7-frame moving average, the mean over the nine joints): the commanded
targets (target_sim_deg) and the measured joints (servo degrees -> sim through the driver's signs), 20 ms frames."""
import json, sys
from pathlib import Path
import numpy as np
LAB = Path(__file__).resolve().parents[1] / "lab"
sys.path[:0] = [str(LAB), str(LAB.parents[3] / "src")]
ARGS = sys.argv[1:]
sys.argv = ["x"]
import hom_setup as HS
import hom_series as S
P = HS._driver_plan()
F, JN = ("thumb", "index", "middle"), ("yaw", "mcp", "pip")
def sim(j):
    v = np.asarray(j, float).reshape(3, 3)
    return np.array([[v[k][("aa", "fe1", "fe2").index(P.SIM_JOINT_TO_SERVO[jn])] / P.JOINT_SIGN[(f, jn)] for jn in JN]
                     for k, f in enumerate(F)]).ravel()
def hf(X, w=7):
    k = np.ones(w) / w
    sm = np.column_stack([np.convolve(X[:, i], k, mode="same") for i in range(X.shape[1])])
    return np.sqrt(np.mean((X - sm)[w:-w] ** 2, axis=0))
def rev(X, eps=0.1):
    d = np.diff(X, axis=0)
    s = np.sign(np.where(np.abs(d) > eps, d, 0))
    a, b = s[1:], s[:-1]
    m = (a != 0) & (b != 0)
    return float(np.mean(a[m] != b[m])) if m.any() else 0.0
D = LAB.parents[3] / "logs/20261116-hom_trials"
names = ARGS or ["20261116-124945", "20261116-125501", "20261116-125922", "20261116-130318",
         "20261120-091915", "20261120-092039", "20261120-104355", "20261120-110018", "20261120-110200",
         "20261120-111342", "20261120-111552", "20261120-111805", "20261120-112127", "20261120-113200"]
print(f"{'record':16} {'n':>4} {'cmd step':>8} {'cmd rev%':>8} {'cmd HF':>7} {'meas HF':>7} {'|cmd-meas|':>10} {'max':>5}  twist      rise  drift")
for n in names:
    f = next(D.rglob(f"{n}_twist_corvus.jsonl"))
    rows = [json.loads(x) for x in f.read_text().splitlines() if '"seq"' in x]
    rows = [r for r in rows if "target_sim_deg" in r and r.get("joints")]
    if len(rows) < 30:
        print(f"{n:16} {len(rows):4d} (too few rows)"); continue
    T = np.array([r["target_sim_deg"] for r in rows], float)
    J = np.array([sim(r["joints"]) for r in rows], float)
    s = S.summary(str(f))
    step = float(np.sqrt(np.mean(np.diff(T, axis=0) ** 2)))
    e = np.abs(T - J)
    print(f"{n:16} {len(T):4d} {step:8.2f} {100 * rev(T):8.0f} {np.mean(hf(T)):7.2f} {np.mean(hf(J)):7.2f} "
          f"{e.mean():10.1f} {e.max():5.0f}  {s['twist_min_deg']:5.1f}..{s['twist_max_deg']:5.1f} {s['rise_max_mm']:5.1f} {s['drift_end_mm']:5.1f}")

# /usr/bin/python3 20261120-command_jitter.py [STAMP ...] > 20261120-command_jitter.txt (corvus, from lab/)
