"""HOM's fingertip feedback made faithful to the paper, checked in closed-loop simulation before the hand (hand-4a,
2026-11-17 19:2x; owner via the lab, 19:0x: "this point jitter with updating directly from the contact sensing is the
whole point of the complementary filter in the original paper ... you gotta do it right").

hom_tasks.rollout (HOM as the landscapes run it, at the bench's 50 Hz) on the calibrated plant of the bench predictions
(rev_clutch2: gears the load cannot backdrive, kp 0.55, the index's and middle's yaw 0.35, soft contact on the stem
scene), the twist at g13 (105, 70) and g0 (70, 110), seeds 0-2, with hom_tasks.VARIANTS' sensing: exact state (rate50),
the tag tracker without tips (notip), the tip model (tip, tip_lo: boards report from 0.05 N), and the new options:
_f the filter at the paper's 500 Hz inside the period (FilterParams.step_s 0.002), _h contact hysteresis (0.25 N in,
0.10 N out), _hq Eq. (1) in the filtered frames in contact. Per run: hom_tasks' metrics, and from the trace the servo
targets' jitter over the task (rms about a 0.5 s running mean per joint, the finger's largest, and the dominant
frequency), the GCF source switches per second, and the contact share.

    HOS_PLANT=kp0.55_kv0_fr0.18_fl0_dp0.011 SCALE_HAND=cad_v2 OMP_NUM_THREADS=1 .venv/bin/python ff_verify.py OUT.jsonl
"""
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import replan as RPL  # noqa: E402  (paths, rev_* imports, the 7.5 V plant)

LAYOUTS = {"g13": ("grid2dgr", 329, 13), "g0": ("grid2dgr", 162, 0)}
VARS = ["rate50", "rate50_f", "notip", "notip_f", "tip", "tip_lo", "tip_h", "tip_f", "tip_fh", "tip_fhq"]


def jitter(trace, t0=2.0):
    t = np.array([r["t"] for r in trace])
    sel = t >= t0
    out = {}
    for f in ("thumb", "index", "middle"):
        Q = np.array([r["q_cmd_deg"][f] for r in trace], float)[sel]
        if len(Q) < 60:
            continue
        k = 25
        best = (0.0, None)
        for j in range(3):
            x = Q[:, j]
            d = (x - np.convolve(x, np.ones(k) / k, mode="same"))[k // 2:-k // 2]
            rms = float(np.sqrt(np.mean(d ** 2)))
            if rms > best[0]:
                fr = np.fft.rfftfreq(len(d), 0.02)
                A = np.abs(np.fft.rfft(d - d.mean()))
                best = (rms, float(fr[1:][np.argmax(A[1:])]))
        src = [r["src"][f] for r in trace]
        sw = sum(1 for a, b in zip(src, src[1:]) if a != b)
        out[f] = {"jit_rms_deg": round(best[0], 3), "jit_hz": round(best[1], 1) if best[1] else None,
                  "src_switches_per_s": round(sw / (t[-1] - t[0] + 1e-9), 2),
                  "cop_share": round(sum(1 for s in src if s == "cop") / len(src), 3)}
    return out


def run(job):
    try:
        import mujoco
        import hand_object_scale_sim as S
        import hom_tasks as T
        from sr2hand.control import hom as H
        RP = RPL.RG.RF._G["RP"]
        RP.TOY_ROWS = RPL.RG.RF._G["toy_rows"]
        RPL.RC2.KP_YAW_IM = 0.35
        src, call, k = LAYOUTS[job["layout"]]
        setup, mxy, _ = RP.toy_setup("twist", 40.0, call, k, src)
        p_pads, p_pt = RP.ref_scene("sphere", 40.0)
        m_pt = mujoco.MjModel.from_xml_path(str(p_pt))
        RP.move_mounts(m_pt, mxy, RP.mount_base(m_pt))
        q_open, _ = T.open_pose(setup, m_pt)
        S.set_palm(m_pt, setup["palm"])
        fs, os_ = T.sr2_specs(setup)
        mirror = H.Mirror(m_pt, fs, os_)
        GP = RPL.RG._G["GP"]
        plant = GP(str(RPL.D / "scenes" / RPL.SCENES["pads"]), setup, mxy, {**RPL.RR.SERVER, "mu": 1.0}, 0.0, c=0.6)
        over, noise = T.VARIANTS[job["var"]]
        hp = replace(H.HomParams(dt=T.DT_C), **over)
        res, trace, _ = T.rollout("twist", setup, plant, mirror, q_open, job["seed"], T.PRM, hp, None, noise, 0.0)
        keep = ("held", "theta_rmse_deg", "w_rmse_ratio", "drift_max_mm", "ramp_three_pads_max_deg", "gain_min",
                "track_err_rms_deg", "track_ref_rms_deg", "J", "track")
        return {**job, **{k_: res.get(k_) for k_ in keep if k_ in res}, "jitter": jitter(trace),
                "dt": hp.dt, "step_s": hp.filter.step_s, "contact_off": hp.contact_force_off, "eq1": hp.eq1_frames}
    except Exception as e:                                       # noqa: BLE001
        import traceback
        return {**job, "error": repr(e)[:200], "tb": traceback.format_exc()[-600:]}


def main():
    import multiprocessing as mp
    out = Path(sys.argv[1])
    RPL.RG.gear_class = RPL.RC2.clutch_class
    RPL.RC2.STIFF, RPL.RC2.BREAKAWAY = True, 0.0
    RPL.RG.RF.RL.rows_file()
    vars_ = sys.argv[2:] or VARS
    jobs = [{"layout": L, "var": v, "seed": s} for L in LAYOUTS for v in vars_ for s in (0, 1, 2)]
    print(len(jobs), "runs", flush=True)
    with open(out, "w") as fo, mp.get_context("fork").Pool(8, initializer=RPL.RG.init,
                                                           initargs=(str(RPL.RG.RF.SCENE),)) as pool:
        for r in pool.imap_unordered(run, jobs):
            fo.write(json.dumps(r) + "\n")
            fo.flush()
            print(json.dumps({k: r.get(k) for k in ("layout", "var", "seed", "held", "theta_rmse_deg", "error")}),
                  flush=True)


if __name__ == "__main__":
    main()
