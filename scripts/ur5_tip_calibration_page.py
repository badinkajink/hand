#!/usr/bin/env python3
"""Page of the fingertip's calibration protocol on the lab's UR5 (CB3) with the OptoForce HEX-E (2026-10-22): the tool
stack and its TCP, the cell, the session's stages, the ground truth they give and a whole session run against the
simulator with known errors.

    .venv/bin/python scripts/ur5_tip_calibration_page.py collect logs/20261022-tip_calibration/<session>
        copies the session's small files (events, registration, touches, pivots, the simulator's truth) to data/<session>/
        and writes results.json from them, the plan and the analysis (scripts/ur5_tip_calibration_analysis.py)
    .venv/bin/python scripts/ur5_tip_calibration_page.py            figures and the page from docs/ alone

Writes docs/experiments/20261022-ur5_tip_calibration/20261022-ur5_tip_calibration_protocol.html.
"""
from __future__ import annotations

import base64
import html
import io
import json
import math
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src/morphohand/driver/reskin5x/host"))
import docs_index  # noqa: E402
import retro_style  # noqa: E402
import texsvg  # noqa: E402

D = ROOT / "docs/experiments/20261022-ur5_tip_calibration"
MEDIA = D / "media"
DATA = D / "data"
PLAN = D / "plan/20261022-plan_sphere.json"
PAGE = D / "20261022-ur5_tip_calibration_protocol.html"
TEMPLATE = ROOT / "scripts/ur5_tip_calibration_page.template.html"
TEX_CACHE = D / "texsvg_cache.json"
TEX_RE = re.compile(r"\$\$(.+?)\$\$|\$(.+?)\$", re.S)
HOST = ROOT / "src/morphohand/driver/reskin5x/host"
RUNNER = HOST / "ur5_tip_calibration.py"
SIMF = HOST / "ur5_tip_calibration_sim.py"
MOUNT_V3 = ROOT / "docs/experiments/20261021-ur5_calibration_mount_v3/20261021-ur5_calibration_mount_v3.html"
EST_PAGE = ROOT / "docs/experiments/20261020-fingertip_contact_estimation/20261020-model_based_contact_estimation.html"
STATUS_PAGE = ROOT / "docs/experiments/20261018-printed_fingertip_status/20261018-printed_fingertip_and_dip_link_status.html"
DATASHEET = "https://www.spie-escad.at/wp-content/uploads/ESCAD_OPTOFORCE_Kraftsensor_Hex-E.pdf"
UR_GUIDE = "https://www.aicobot.de/wp-content/uploads/2017/05/OptoForce-User-Guide-for-UR-V1.3.pdf"


def esc(s):
    return html.escape(str(s), quote=True)


def rel(p):
    import os
    return os.path.relpath(p, D)


def link(p, text):
    return f'<a href="{esc(rel(p))}">{text}</a>'


def code(p):
    return f"<code>{esc(p)}</code>"


def img_uri(png, width=1100, quality=88):
    from PIL import Image
    im = Image.open(png).convert("RGB")
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    b = io.BytesIO()
    im.save(b, "JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def fig(n, png, caption, width=1100):
    return (f'<figure id="fig{n}"><img alt="Figure {n}" src="{img_uri(png, width)}">'
            f"<figcaption>Figure {n}. {caption}</figcaption></figure>")


def film(n, mp4, poster, caption):
    v = "data:video/mp4;base64," + base64.b64encode(Path(mp4).read_bytes()).decode()
    return (f'<figure id="fig{n}"><video controls muted loop playsinline preload="metadata" poster="{img_uri(poster, 1100)}" '
            f'src="{v}" style="width:100%"></video><figcaption>Figure {n}. {caption}</figcaption></figure>')


def table(head, body, num_cols=(), caption=None, n=None):
    cap = f"<caption>Table {n}. {caption}</caption>" if caption else ""
    th = "".join(f'<th{" class=n" if k in num_cols else ""}>{h}</th>' for k, h in enumerate(head))
    tr = "".join("<tr>" + "".join(f'<td{" class=n" if k in num_cols else ""}>{c}</td>' for k, c in enumerate(r)) + "</tr>"
                 for r in body)
    return f'<div class="tw"><table>{cap}<thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


def plt_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 13, "axes.titlesize": 13, "axes.labelsize": 13,
                         "legend.fontsize": 12, "xtick.labelsize": 12, "ytick.labelsize": 12})
    return plt


# ------------------------------------------------------------------------------------------------ study (from a session)
def registration_errors(geo, plan, tr, T_EM, T_MP, c_B):
    """Error (mm) of the contact point the registration predicts, against the simulator's truth, at every press pose of
    the plan: tangential to the pad and along its normal."""
    import ur5_tip_calibration as C
    import ur5_tip_calibration_sim as SIM
    T_EP_e = T_EM @ T_MP
    T_EP_t, c_t = np.array(tr["T_EP"]), np.array(tr["c_B"])
    T_BI_e = C.Tm(np.array(tr["T_BI"])[:3, :3], c_B - np.array(tr["T_BI"])[:3, :3] @ geo.c_I)
    out = []
    q0 = np.array(plan["sim"]["q_seed"])
    for a in plan["actions"]:
        if a["stage"] not in ("grid", "pivot", "shear", "roll", "twist") or a["target"]["frame"] != "P":
            continue
        T_IP = geo.T_IP(a["target"]["p"], a["target"]["n"], a.get("twist", 0.0), 0.0)
        T_BE = T_BI_e @ T_IP @ C.inv(T_EP_e)
        q = SIM.ik(T_BE, q0)
        T_true = SIM.fk(q, tr["dh_true"])
        q_e = C.apply(C.inv(T_BE @ T_EP_e), c_B)
        q_t = C.apply(C.inv(T_true @ T_EP_t), c_t)
        n = np.array(a["target"]["n"])
        dq = q_t - q_e
        out.append(dict(region=a["target"]["region"], twist=a.get("twist", 0.0), tan=float(np.linalg.norm(dq - (dq @ n) * n)),
                        nrm=float(dq @ n)))
    return out


def registration_rows(sd, plan, geo, reg, tr):
    """Errors of the registration after each of its steps against the simulator's truth (Table 5)."""
    import ur5_tip_calibration as C
    T_EM0, T_MP0 = geo.T_EM0, geo.T_MP0
    T_BE_home = np.array(reg["T_BE_home"])
    c_home = C.apply(T_BE_home @ T_EM0 @ C.inv(geo.T_IM_home), geo.c_I)
    n1 = reg["notes"].get("register", {})
    n2 = reg["notes"].get("register_pivot", {})
    stages = [("CAD and homing", T_EM0, T_MP0, c_home)]
    if n1:
        if "T_EM" in n1:
            s1 = (np.array(n1["T_EM"]), np.array(n1["T_MP"]), np.array(n1["c_B"]))
        else:
            x = np.array(n1["x"])
            s1 = (C.perturb(T_EM0, x[0:6]), C.perturb(T_MP0, x[6:12]), c_home + x[12:15])
        stages.append(("Touch-offs", *s1))
        if n2:
            if "T_EM" in n2:
                s2 = (np.array(n2["T_EM"]), np.array(n2["T_MP"]), np.array(n2["c_B"]))
            else:
                x = np.array(n2["x"])
                s2 = (C.perturb(s1[0], x[0:6]), C.perturb(s1[1], x[6:12]), s1[2] + x[12:15])
            stages.append(("Touch-offs with the pivot folded in", *s2))
    fc = np.array([8.5, 0.0, 10.86])
    T_EP_t = np.array(tr["T_EP"])
    rows = []
    for name, T_EM, T_MP, c in stages:
        T = T_EM @ T_MP
        er = registration_errors(geo, plan, tr, T_EM, T_MP, c)
        tan = np.array([e["tan"] for e in er])
        rows.append(dict(name=name, face_centre_mm=float(np.linalg.norm(C.apply(T, fc) - C.apply(T_EP_t, fc))),
                         axes_deg=math.degrees(float(np.linalg.norm(C.log_so3(T[:3, :3].T @ T_EP_t[:3, :3])))),
                         centre_mm=float(np.linalg.norm(c - np.array(tr["c_B"]))),
                         tan_median=float(np.median(tan)), tan_p95=float(np.percentile(tan, 95)), tan_max=float(tan.max()),
                         nrm_median=float(np.median(np.abs([e["nrm"] for e in er]))), n=len(er), tan_all=tan.tolist(),
                         session=Path(sd).name))
    return rows


def collect(session, compare=None):
    import ur5_tip_calibration as C
    import ur5_tip_calibration_analysis as AN
    sd = Path(session)
    dst = DATA / sd.name
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("events.jsonl", "registration.json", "touches.json", "pivots.json", "sim_truth.json", "session.json"):
        if (sd / f).exists():
            shutil.copy2(sd / f, dst / f)
    plan = json.loads((sd / "plan.json").read_text())
    geo = C.Geometry(plan)
    reg = json.loads((sd / "registration.json").read_text())
    tr = json.loads((sd / "sim_truth.json").read_text()) if (sd / "sim_truth.json").exists() else None
    res = dict(session=sd.name, when=time.strftime("%Y-%m-%d %H:%M"), plan_checks=plan["checks"], payload=plan["payload"])
    ev = [json.loads(l) for l in open(sd / "events.jsonl") if l.strip()]
    ends = [e for e in ev if e.get("event") == "end"]
    res["actions_ok"] = sum(1 for e in ends if e.get("status") == "ok")
    res["actions_abort"] = [dict(id=e["id"], why=e.get("why")) for e in ends if e.get("status") == "abort"]
    res["duration_min"] = (ev[-1]["t"] - ev[0]["t"]) / 60.0
    st = {}
    for e in ev:
        if e.get("event") == "start":
            st.setdefault(e.get("stage"), [e["t"], e["t"]])[0] = min(st.get(e.get("stage"), [e["t"]])[0], e["t"])
        if e.get("event") == "end":
            stg = next((x.get("stage") for x in ev if x.get("event") == "start" and x.get("id") == e["id"]), None)
            if stg in st:
                st[stg][1] = max(st[stg][1], e["t"])
    res["stage_minutes"] = {k: (v[1] - v[0]) / 60.0 for k, v in st.items()}
    res["registration_notes"] = reg.get("notes", {})
    touches = json.loads((sd / "touches.json").read_text()) if (sd / "touches.json").exists() else []
    res["touch_onsets"] = [dict(id=t["id"], kind=t["kind"], s0_minus_expected=t["s0"] - t["expected_mm"], p=t["p"],
                                rms_N=t["rms_N"]) for t in touches]
    if tr is not None:
        res["registration_vs_truth"] = registration_rows(sd, plan, geo, reg, tr)
        if compare:
            cd = Path(compare)
            dst2 = DATA / cd.name
            dst2.mkdir(parents=True, exist_ok=True)
            for f_ in ("events.jsonl", "registration.json", "touches.json", "pivots.json", "sim_truth.json", "session.json"):
                if (cd / f_).exists():
                    shutil.copy2(cd / f_, dst2 / f_)
            reg2 = json.loads((cd / "registration.json").read_text())
            tr2 = json.loads((cd / "sim_truth.json").read_text())
            rows2 = registration_rows(cd, json.loads((cd / "plan.json").read_text()), geo, reg2, tr2)
            res["registration_vs_truth"] += [r for r in rows2 if r["name"].startswith("Touch-offs with")]
        res["truth"] = tr["truth"]
        res["gravity_truth"] = dict(mass_kg=tr["truth"]["payload_kg"], com_mm=tr["truth"]["payload_com_mm"],
                                    turn_deg=tr["truth"]["hex_turn_deg"])
        pc = reg.get("notes", {}).get("pivot_check")
        if pc:
            res["pivot_check"] = dict(worst_mm=pc["worst_mm"], pivots=[dict(id=p_["id"], board=p_["dp_board"],
                                                                            registration=p_.get("dp_registration"))
                                                                       for p_ in pc["pivots"]])
    # the analysis
    ses = AN.Session(sd)
    holds = ses.holds()
    cal, est = AN.estimator(ses, holds)
    AN.evaluate(holds, est)
    _, estm = AN.estimator(ses, holds, mask=AN.covered)
    AN.evaluate(holds, estm, "est_masked")
    S = AN.summarize(ses, holds)
    S["estimator_calibration"] = dict(k=cal.k, poles=list(cal.poles), dx=cal.dx, dy=cal.dy, dz=cal.dz, rms_mT=cal.rms_mT)
    AN.save(holds, dst / "holds.npz")
    (dst / "summary.json").write_text(json.dumps(S, indent=1, default=C.jsonable))
    res["analysis"] = S
    # a press and a twist for the traces figure
    pick = {}
    for h in holds:
        if h["kind"] == "press" and h["stage"] == "grid" and h["region"] == "face" and "press" not in pick:
            pick["press"] = h["id"]
        if h["kind"] == "twist" and "twist" not in pick:
            pick["twist"] = h["id"]
    tr_rows = {}
    ft = ses.d["ft"]
    for k, aid in pick.items():
        e = ses.ends[aid]["rec"]
        t0 = e["t_rest0"][0] - 0.5
        t1 = (e.get("t_rest1") or [e["t_stop"] + 2.0])[-1] + 0.5
        m = (ft[:, 0] >= t0) & (ft[:, 0] <= t1)
        T_rest = np.array(e["T_rest"])
        d = -(ses.reg.T_BI[:3, :3] @ np.array([0.0, 0.0, 1.0]))
        f = np.array([float(-(T_rest[:3, :3] @ (ses.Kinv @ (w[:3] - np.array(e["tare"])[:3]))) @ d) for w in ft[m, 3:9]])
        tq = ft[m, 3:9][:, 3:] - np.array(e["tare"])[3:]
        mb = (ses.d["t_b"] >= t0) & (ses.d["t_b"] <= t1)
        rest = ses.win(ses.d["t_b"], ses.b, *e["t_rest0"]).mean(axis=0)
        bn = np.linalg.norm(ses.b[mb] - rest, axis=1)
        tr_rows[k] = dict(id=aid, t_ft=(ft[m, 0] - t0).tolist()[::5], push=f.tolist()[::5],
                          torque=np.linalg.norm(tq, axis=1).tolist()[::5], t_b=(ses.d["t_b"][mb] - t0).tolist(), board=bn.tolist(),
                          levels=[[l["t"][0] - t0, l["t"][1] - t0, l["N"]] for l in e["levels"]],
                          turns=[[tu["t"][0] - t0, tu["t"][1] - t0, tu["deg"]] for mv in e.get("moves", []) for tu in mv["turns"]])
    res["traces"] = tr_rows
    (D / "results.json").write_text(json.dumps(res, indent=1, default=C.jsonable))
    print(f"results.json from {sd.name}: {len(holds)} holds, {res['actions_ok']} actions ok, {len(res['actions_abort'])} aborted, "
          f"{res['duration_min']:.0f} min")


# ------------------------------------------------------------------------------------------------ figures
def fig_surface(plan, out):
    plt = plt_setup()
    g = plan["geometry"]["surface"]
    P, N = np.array(g["pts"]), np.array(g["normals"])
    import ur5_tip_calibration_plan as PL
    reg = PL.regions(N)
    reach = np.zeros(len(P), bool)
    reach[plan["checks"]["reachable_index"]] = True
    acts = plan["actions"]
    kinds = {"pad": ("^", "Touch-off", "#000000"), "grid": ("o", "Calibration press", "#1f5fbf"),
             "shear": ("s", "Slide, roll, twist", "#c0392b"), "pivot": ("*", "Pivot", "#2e8b57")}
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 5.0), dpi=110)
    views = ((lambda n: n[:, 0] > 0.3, 2, 1, "y across the finger (mm)", "Palmar view: points facing the palm"),
             (lambda n: n[:, 1] > 0.3, 2, 0, "x along the face's normal (mm)", "Side view: points facing +y"))
    for j, (sel, a0, a1, lab1, title) in enumerate(views):
        side = sel(N)
        ax[j].scatter(P[side & ~reach, a0], P[side & ~reach, a1], s=6, c="#cfcfcf", label="Out of the sphere's reach")
        for r, c in (("face", "#f2c28f"), ("edge", "#e8a35c"), ("flank", "#b9783a")):
            m = side & reach & (reg == r)
            ax[j].scatter(P[m, a0], P[m, a1], s=8, c=c, label=f"Reached: {r}")
        for st, (mk, lab, col) in kinds.items():
            tg = [a["target"] for a in acts if (a["stage"] == st or (st == "shear" and a["stage"] in ("shear", "roll", "twist")))
                  and a["target"]["frame"] == "P"]
            if not tg:
                continue
            pts = np.array([t["p"] for t in tg])
            nn = np.array([t["n"] for t in tg])
            keep = sel(nn)
            ax[j].scatter(pts[keep, a0], pts[keep, a1], marker=mk, s=42 if mk != "*" else 170,
                          facecolors="none" if mk == "o" else col, edgecolors=col, linewidths=1.2, label=lab if j == 0 else None)
        ax[j].set_xlabel("z along the finger toward the tip end (mm)")
        ax[j].set_ylabel(lab1)
        ax[j].set_aspect("equal")
        ax[j].grid(alpha=0.3)
        ax[j].set_title(title)
    h, l_ = ax[0].get_legend_handles_labels()
    fig.legend(h, l_, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(out)
    plt.close(fig)


def fig_registration(res, out):
    plt = plt_setup()
    rows = res["registration_vs_truth"]
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 5.2), dpi=110)
    names = [r["name"] for r in rows]
    x = np.arange(len(rows))
    ax[0].bar(x - 0.2, [r["face_centre_mm"] for r in rows], 0.4, color="#1f5fbf", label="Face centre on the flange (mm)")
    ax[0].bar(x + 0.2, [r["axes_deg"] for r in rows], 0.4, color="#e8a35c", label="Pad axes on the flange (deg)")
    ax[0].set_xticks(x, [n_.replace(" with the pivot folded in", "\nwith the pivot\nfolded in").replace(" and homing", "\nand homing")
                         for n_ in names])
    ax[0].set_ylabel("Error against the simulator's truth")
    ax[0].set_ylim(0, 1.4 * max(max(r["face_centre_mm"], r["axes_deg"]) for r in rows))
    ax[0].legend(frameon=False)
    ax[0].grid(alpha=0.3, axis="y")
    for r, c in zip(rows, ("#999999", "#1f5fbf", "#2e8b57")):
        v = np.sort(r["tan_all"])
        ax[1].step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=c, label=r["name"])
    ax[1].set_xlabel("Contact point on the pad, error along the surface (mm)")
    ax[1].set_ylabel("Fraction of the plan's press poses")
    ax[1].set_xscale("log")
    tk = [0.05, 0.1, 0.2, 0.5, 1.0]
    ax[1].set_xticks(tk, [f"{t:g}" for t in tk])
    ax[1].minorticks_off()
    ax[1].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_truth(npz, out):
    plt = plt_setup()
    z = np.load(npz)
    have = np.isfinite(z["true_f"][:, 0]) if "true_f" in z.files else np.zeros(len(z["t"]), bool)
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 4.4), dpi=110)
    n_t = z["true_n"][have]
    fn_t = -np.einsum("ij,ij->i", z["true_f"][have], n_t)
    ax[0].plot(fn_t, z["f_n"][have], ".", ms=4, color="#1f5fbf")
    lim = [0, max(5.5, float(fn_t.max()) * 1.05)]
    ax[0].plot(lim, lim, "k-", lw=0.8)
    ax[0].set_xlabel("Simulator's normal force (N)")
    ax[0].set_ylabel("Ground truth from HEX-E and robot (N)")
    ax[0].grid(alpha=0.3)
    dp = z["c_P"][have] - z["true_c"][have]
    n_g = z["n"][have]
    tan = np.linalg.norm(dp - np.einsum("ij,ij->i", dp, n_g)[:, None] * n_g, axis=1)
    ax[1].hist(tan, bins=40, color="#1f5fbf")
    ax[1].set_xlabel("Sphere on the pad, along the surface:\nground truth less the simulator's (mm)")
    ax[1].set_ylabel("Holds")
    ax[1].grid(alpha=0.3)
    ft_t = z["true_f"][have] + fn_t[:, None] * n_t
    sh = ~np.isin(z["kind"][have], ["press"])
    ax[2].plot(np.linalg.norm(ft_t[sh], axis=1), np.linalg.norm(z["f_t"][have][sh], axis=1), ".", ms=4, color="#c0392b",
               label="Slide, roll and twist holds")
    ax[2].plot(np.linalg.norm(ft_t[~sh], axis=1), np.linalg.norm(z["f_t"][have][~sh], axis=1), ".", ms=4, color="#999999",
               label="Press holds")
    lim2 = [0, max(0.5, float(np.linalg.norm(ft_t, axis=1).max()) * 1.05)]
    ax[2].plot(lim2, lim2, "k-", lw=0.8)
    ax[2].set_xlabel("Simulator's shear (N)")
    ax[2].set_ylabel("Ground truth shear (N)")
    ax[2].legend(frameon=False, loc="upper left")
    ax[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_estimator(npz, out):
    plt = plt_setup()
    z = np.load(npz)
    ok = z["est_contact"] & (z["kind"] == "press")
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=110)
    for r, c in (("face", "#1f5fbf"), ("edge", "#e8a35c"), ("flank", "#2e8b57")):
        m = ok & (z["region"] == r)
        if not m.any():
            continue
        ax[0].plot(z["f_n"][m], z["est_f_n"][m], ".", ms=5, color=c, label=r.capitalize())
        dp = z["est_p"][m] - z["p"][m]
        n = z["n"][m]
        tan = np.linalg.norm(dp - np.einsum("ij,ij->i", dp, n)[:, None] * n, axis=1)
        lv = z["level_N"][m]
        ax[1].plot(lv * (1 + 0.04 * {"face": -1, "edge": 0, "flank": 1}[r]), tan, ".", ms=5, color=c, label=r.capitalize())
    lim = [0, 6.0]
    ax[0].plot(lim, lim, "k-", lw=0.8)
    ax[0].set_xlabel("Ground truth normal force (N)")
    ax[0].set_ylabel("Estimator's normal force (N)")
    ax[0].legend(frameon=False)
    ax[0].grid(alpha=0.3)
    ax[1].set_xscale("log")
    tk = [0.25, 0.5, 1.0, 2.0, 3.5, 5.0]
    ax[1].set_xticks(tk, [f"{t:g}" for t in tk])
    ax[1].minorticks_off()
    ax[1].set_xlabel("Force level of the hold (N)")
    ax[1].set_ylabel("Estimator's contact point error (mm)")
    ax[1].grid(alpha=0.3)
    ax[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_traces(res, out):
    plt = plt_setup()
    T = res["traces"]
    fig, ax = plt.subplots(2, 2, figsize=(12.0, 6.6), dpi=110, sharex="col")
    for j, k in enumerate(("press", "twist")):
        if k not in T:
            continue
        t = T[k]
        for l0, l1, N in t["levels"]:
            for a_ in ax[:, j]:
                a_.axvspan(l0, l1, color="#f2c28f", alpha=0.5, lw=0)
        for l0, l1, deg in t["turns"]:
            for a_ in ax[:, j]:
                a_.axvspan(l0, l1, color="#bcd3f2", alpha=0.6, lw=0)
        ax[0, j].plot(t["t_ft"], t["push"], color="#1f5fbf", lw=0.8, label="Push along the approach (N)")
        if k == "twist":
            ax[0, j].plot(t["t_ft"], np.array(t["torque"]) * 10, color="#c0392b", lw=0.8, label="HEX-E torque change x 10 (N m)")
        ax[0, j].set_ylabel("HEX-E (N)")
        ax[0, j].legend(frameon=False, loc="upper left")
        ax[0, j].grid(alpha=0.3)
        ax[1, j].plot(t["t_b"], t["board"], color="#2e8b57", lw=0.8)
        ax[1, j].set_ylabel("Board change from rest (uT)")
        ax[1, j].set_xlabel("Time (s)")
        ax[1, j].grid(alpha=0.3)
        ax[0, j].set_title(("A calibration press: " if k == "press" else "A twist at 1 N: ") + t["id"])
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ the page
def build():
    res = json.loads((D / "results.json").read_text())
    plan = json.loads(PLAN.read_text())
    MEDIA.mkdir(parents=True, exist_ok=True)
    sess = DATA / res["session"]
    npz = sess / "holds.npz"
    fig_surface(plan, MEDIA / "20261022-targets_on_the_pad.png")
    if res.get("registration_vs_truth"):
        fig_registration(res, MEDIA / "20261022-registration_against_truth.png")
    fig_truth(npz, MEDIA / "20261022-ground_truth_against_simulator.png")
    fig_estimator(npz, MEDIA / "20261022-estimator_on_the_session.png")
    fig_traces(res, MEDIA / "20261022-press_and_twist_traces.png")
    import ur5_tip_calibration_page_text as TXT
    S, lede, summary = TXT.sections(res, plan, dict(fig=fig, film=film, table=table, link=link, code=code, esc=esc, MEDIA=MEDIA,
                                                    D=D, RUNNER=RUNNER, SIMF=SIMF, MOUNT_V3=MOUNT_V3, EST_PAGE=EST_PAGE,
                                                    STATUS_PAGE=STATUS_PAGE, DATASHEET=DATASHEET, UR_GUIDE=UR_GUIDE, ROOT=ROOT,
                                                    PLAN=PLAN, sess=sess))
    toc = "\n".join(f'<li><a href="#{sid}">{k}. {esc(hd)}</a></li>' for k, (sid, hd, _) in enumerate(S, 1))
    secs = "\n".join(f'<section id="{sid}"><h2>{k}. {esc(hd)}</h2>{body}</section>' for k, (sid, hd, body) in enumerate(S, 1))
    page = TEMPLATE.read_text()
    filmfile = MEDIA / "20261022-simulated_session.mp4"
    head_film = (film(1, filmfile, filmfile.with_suffix(".png"), TXT.film_caption(res)) if filmfile.exists() else "")
    for k, v in {"TITLE": "UR5 calibration protocol",
                 "H1": "Calibration protocol of the magnetic fingertip on the UR5 and the OptoForce HEX-E",
                 "EYEBROW": "Printed fingertips and force sensing &#183; 2026-10-22", "LEDE": lede,
                 "BYLINE": TXT.byline(res), "FILM": head_film, "TOC": toc, "SECTIONS": secs,
                 "FOOTER": str(PAGE.relative_to(ROOT)), "TEXCSS": texsvg.CSS}.items():
        page = page.replace("{{" + k + "}}", v)
    items = [((m.group(1) if m.group(1) is not None else m.group(2)).strip(), m.group(1) is not None)
             for m in TEX_RE.finditer(page)]
    svgs = iter(texsvg.render(items, cache_path=TEX_CACHE))
    page = TEX_RE.sub(lambda m: next(svgs), page)
    page = retro_style.apply(page)
    page = docs_index.tag(page, topic="fingertip", summary=summary)
    PAGE.write_text(page)
    print(f"{PAGE.relative_to(ROOT)}: {PAGE.stat().st_size / 1e6:.2f} MB, {len(items)} formulas", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "collect":
        collect(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    else:
        build()
