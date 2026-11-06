#!/usr/bin/env python3
"""Builds docs/experiments/20261106-sr2_servo_sysid/20261106-sr2_servo_identification.html: the free-space
identification of the SR2 hand's nine SCS0009 servos on 2026-11-06 (QUEUE R36; the owner: "i put the ur5 computer in
charge of doing more sysid, it has the go ahead").

Data: the lab's tables under the folder's data/ (dwells.tsv, holds_c.tsv, registers.txt, rearm_readonly.txt, from
lab/sysid_chain.sh and lab/scs0009_registers.py) and the fit 20261106-sr2_servo_sysid_fit.json
(scripts/hardware/sr2_servo_sysid_fit.py, under resguard: it loads the cad_v3 model). The 11-04 telemetry of the index's
stop is data/20261104-night/run1_home/telemetry.jsonl of the crosstalk experiment.

    .venv/bin/python scripts/hardware/sr2_servo_sysid_page.py figs   # the figures
    .venv/bin/python scripts/hardware/sr2_servo_sysid_page.py        # the page
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path
import docs_index  # noqa: E402
import fp_review_page as P  # noqa: E402
import retro_style  # noqa: E402
import texsvg  # noqa: E402

table, img, film, esc = P.table, P.img, P.film, P.esc
D = ROOT / "docs/experiments/20261106-sr2_servo_sysid"
DATA = D / "data"
MEDIA = D / "media"
OUT = D / "20261106-sr2_servo_identification.html"
TEMPLATE = ROOT / "scripts/hardware/sr2_servo_sysid_page.template.html"
TEX_CACHE = D / "texsvg_cache.json"
FIT = DATA / "20261106-sr2_servo_sysid_fit.json"
NIGHT = ROOT / "docs/experiments/20261104-sr2_hand_crosstalk/data/20261104-night/run1_home/telemetry.jsonl"
F = ("thumb", "index", "middle")
J = ("yaw", "mcp", "pip")
STEP = 300.0 / 1024.0
SERVO_ID = {(f, j): 3 * i + k for i, f in enumerate(F) for k, j in enumerate(J)}


def load():
    R = json.loads(FIT.read_text())
    R["regs"] = registers()
    return R


def registers():
    """Each servo's decoded fields from data/registers.txt: {servo id: {address: (raw hex, value text)}}."""
    out, sid = {}, None
    for line in (DATA / "registers.txt").read_text().splitlines():
        m = re.match(r"servo (\d+) ", line)
        if m:
            sid = int(m.group(1))
            out[sid] = {}
            continue
        m = re.match(r"\s+(\d+)\s{2}(.{40})\s(.{6})\s+(.+?)\s{3}(Feetech|rustypot)", line)
        if m and sid is not None:
            out[sid][int(m.group(1))] = (m.group(3).strip(), m.group(4).strip(), m.group(2).strip())
    return out


def f1(v, k=1):
    return "&ndash;" if v is None else f"{v:.{k}f}"


def mnm(v, k=1):
    return f1(None if v is None else 1000 * v, k)


def name(key):
    f, j = key.split("_")
    return f"{f.capitalize()} {j}"


# ------------------------------------------------------------------------------------------------- figures
def plt_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 14, "axes.titlesize": 14, "axes.labelsize": 14,
                         "legend.fontsize": 12, "xtick.labelsize": 13, "ytick.labelsize": 13,
                         "mathtext.fontset": "cm"})
    return plt


def save(fig, name_):
    import matplotlib.pyplot as plt
    fig.savefig(MEDIA / name_, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", (MEDIA / name_).relative_to(ROOT), flush=True)


def fig_controller(R):
    plt = plt_setup()
    C = R["controller"]
    T = {int(k): v for k, v in C["table"].items()}
    ks = np.array(sorted(T))
    n = np.array([T[k][0] for k in ks])
    med = np.array([T[k][1] for k in ks])
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    b = C["dead_band_steps"]
    ax.axvspan(-b - 0.5, b + 0.5, color="#dddddd", zorder=0, label=f"no drive (|error| up to {b} steps)")
    ax.scatter(ks, med, s=10 + 45 * np.log10(n), color="#3288bd", edgecolor="k", linewidth=0.5, zorder=3,
               label="median over dwells (area grows with count)")
    x = np.linspace(b + 1, max(abs(ks)), 50)
    for sg in (-1, 1):
        sl, ic = C["by_sign"][str(sg)]
        ax.plot(sg * x, sg * (ic + sl * x), color="#d53e4f", lw=1.6, zorder=2,
                label="line from 4 steps out" if sg == 1 else None)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("Error (encoder steps of 0.29 deg)")
    ax.set_ylabel("Load (per mille of full duty)")
    ax.legend(loc="upper left", frameon=False, markerscale=0.6, fontsize=11)
    ax.grid(alpha=0.3)
    save(fig, "20261106-drive_against_error.png")


def fig_staircases(R):
    plt = plt_setup()
    k = R["stiffness_Nm_per_rad"]
    rec = R["records"]
    panels = [("a2", "thumb", "mcp"), ("a2", "middle", "mcp"), ("a2", "index", "pip"), ("b2", "middle", "yaw")]
    fig, axs = plt.subplots(2, 2, figsize=(10.5, 7.4), sharey=False)
    for ax, (blk, f, j) in zip(axs.flat, panels):
        rs = [r for r in rec if r["block"] == blk and r["moving"] and r["finger"] == f and r["joint"] == j
              and r["dir_this"] != 0]
        for d, mk, lab in ((1, "^", "reached moving up"), (-1, "v", "reached moving down")):
            pts = sorted((r["cmd"], r["deg"] - r["cmd"]) for r in rs if r["dir_this"] == d)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], mk + "-", ms=6, lw=0.8,
                    color="#d53e4f" if d == 1 else "#3288bd", label=lab)
        g = sorted((r["cmd"], np.degrees(r["tau_g"] / k)) for r in rs)
        ax.plot([p[0] for p in g], [p[1] for p in g], "k--", lw=1.2, label="gravity droop at 0.55 N m/rad")
        ax.axhspan(-R["controller"]["dead_band_steps"] * STEP, R["controller"]["dead_band_steps"] * STEP,
                   color="#eeeeee", zorder=0)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(f"{f.capitalize()} {j}, block {blk}")
        ax.set_xlabel("Commanded angle, servo frame (deg)")
        ax.set_ylabel("Error (deg)")
        ax.grid(alpha=0.3)
    h, lab = axs[0, 0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(h, lab, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.05))
    save(fig, "20261106-staircases.png")


def fig_bands(R):
    plt = plt_setup()
    c = R["c_Nm_per_unit"]
    rec = R["records"]
    keys = [f"{f}_{j}" for f in F for j in J]
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    rng = np.random.default_rng(0)
    for i, key in enumerate(keys):
        f, j = key.split("_")
        rs = [r for r in rec if r["finger"] == f and r["joint"] == j and np.isfinite(r["load"])
              and r["block"] in ("a2", "b2", "c", "c40") and r["dir"] != 0]
        for d, col, lab in ((1, "#d53e4f", "last move up"), (-1, "#3288bd", "last move down")):
            y = [1000 * (c * -r["load"] + r["tau_g"]) for r in rs if r["dir"] == d]
            x = i + (0.18 * d) + rng.uniform(-0.07, 0.07, len(y))
            ax.scatter(x, y, s=5, color=col, alpha=0.5, label=lab if i == 0 else None)
        b = R["bands"][key]
        lo, hi = 1000 * (b["centre_Nm"] - b["half_width_Nm"]), 1000 * (b["centre_Nm"] + b["half_width_Nm"])
        ax.plot([i - 0.38, i + 0.38], [lo, lo], "k-", lw=1.2)
        ax.plot([i - 0.38, i + 0.38], [hi, hi], "k-", lw=1.2, label="median and band" if i == 0 else None)
        ax.plot([i - 0.25, i + 0.25], [1000 * b["centre_Nm"]] * 2, "k-", lw=0.8)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(len(keys)), [name(k) for k in keys], rotation=30, ha="right")
    ax.set_ylabel("Drive torque plus gravity (mN m)")
    ax.legend(loc="upper left", frameon=False, ncol=2, markerscale=3)
    ax.grid(alpha=0.3, axis="y")
    save(fig, "20261106-friction_bands.png")


def fig_holds():
    plt = plt_setup()
    rows = list(csv.DictReader(open(DATA / "holds_c.tsv"), delimiter="\t"))
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 4.4), sharey=True)
    col = {20.0: "#fdae61", 30.0: "#f46d43", 40.0: "#9e0142"}
    for ax, f in zip(axs, F):
        groups = {}
        for r in rows:
            if r["finger"] == f and r["dir"] == "out":
                groups.setdefault((r["block"], int(r["step"])), []).append(r)
        for (blk, _k), rs in groups.items():
            cmd = float(rs[0]["cmd_servo_deg"])
            t = np.array([float(r["t_from_cmd_s"]) for r in rs])
            e = np.array([float(r["deg"]) - cmd if r["deg"] else np.nan for r in rs])
            ax.plot(t, e, "-" if blk == "c" else "--", color=col[abs(cmd)], lw=1.3 if blk == "c" else 1.0)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(f"{f.capitalize()} yaw")
        ax.set_xlabel("Time from the command (s)")
        ax.set_ylim(-6, 4)
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("Error, servo frame (deg)")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=col[a], lw=1.6, label=f"step to {a:.0f} deg") for a in sorted(col)] + \
        [Line2D([], [], color="k", lw=1.3, label="mcp at 0 (block c)"),
         Line2D([], [], color="k", ls="--", lw=1.0, label="mcp at 40 (block c40)")]
    fig.legend(handles=h, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    save(fig, "20261106-yaw_holds.png")


def cmd_figs():
    R = load()
    MEDIA.mkdir(parents=True, exist_ok=True)
    fig_controller(R)
    fig_staircases(R)
    fig_bands(R)
    fig_holds()
    return 0


# ------------------------------------------------------------------------------------------------- numbers
def night_stop():
    """The index mcp's stop on 11-04 (run1_home): the samples at the highest angle, the angle before, the loads."""
    rows = [json.loads(x) for x in NIGHT.read_text().splitlines() if x.strip()]
    pts = [(r["t_sample"], (r["servos"] or {}).get("1", {}).get("fe1"), (r["load"] or {}).get("4")) for r in rows]
    pts = [p for p in pts if p[1] is not None]
    top = max(p[1] for p in pts)
    at = [p for p in pts if abs(p[1] - top) < 0.01]
    steady = [p for p in at if abs(p[2]) == abs(at[0][2])]           # until the back-out reverses the drive
    t0, t1 = at[0][0], steady[-1][0]
    before = [p for p in pts if t0 - 3.0 <= p[0] < t0 - 0.15]       # the last 0.15 s is the new command's onset
    ev = {e["step"]: e for e in (json.loads(x) for x in (NIGHT.parent / "events.jsonl").read_text().splitlines() if x.strip())
          if e.get("kind") == "step"}
    return {"top": top, "loads": sorted({abs(p[2]) for p in steady}), "n_steady": len(steady), "dur": t1 - t0,
            "back_out_load": at[-1][2] if len(at) > len(steady) else None, "before": before[-1][1],
            "before_loads": sorted({abs(p[2]) for p in before}), "cmd_before": ev[125]["joints"]["index"]["mcp"],
            "err_before": ev[125]["err_deg"], "cmd": 64.0}


def stall_table(R):
    S = R["stalls"]
    out = {}
    for s in S:
        if s["dir"] != "out":
            continue
        out.setdefault((s["finger"], np.sign(s["cmd_servo"])), {})[(s["block"], abs(s["cmd_servo"]))] = s
    return out


def numbers(R):
    C = R["controller"]
    b = R["bands"]
    V = R["volts_torque_off"]
    st = stall_table(R)
    short_pos = [s["short_deg"] for (f, sg), d in st.items() if sg > 0 for s in d.values()]
    short_neg = [s["short_deg"] for (f, sg), d in st.items() if sg < 0 for s in d.values()]
    t_in = [s["t_within_1step_s"] for d in st.values() for s in d.values() if s["t_within_1step_s"] is not None]
    same = [abs(d[("c", a)]["short_deg"] - d[("c40", a)]["short_deg"]) for d in st.values() for a in (20.0, 30.0, 40.0)
            if ("c", a) in d and ("c40", a) in d]
    full = (1000 - C["intercept"]) / C["slope_per_step"]
    yaw_band = [b[f"{f}_yaw"]["half_width_Nm"] for f in ("index", "middle")]
    mcp_band = [b[f"{f}_mcp"]["half_width_Nm"] for f in F]
    pip_band = [b[f"{f}_pip"]["half_width_Nm"] for f in F]
    flex_band = mcp_band + pip_band
    zero_hold = max(((k, v["zero_drive_tau_g_max_Nm"]) for k, v in b.items() if k.endswith("mcp")), key=lambda kv: kv[1])
    tg = {j: max(b[f"{f}_{j}"]["tau_g_max_Nm"] for f in F) for j in J}
    k = R["stiffness_Nm_per_rad"]
    dev = {}
    for f in F:
        for j in ("mcp", "pip"):
            x = [abs(r["deg"] - r["cmd"] - np.degrees(r["tau_g"] / k)) for r in R["records"]
                 if r["block"] == "a2" and r["moving"] and r["finger"] == f and r["joint"] == j and r["dir_this"] != 0]
            dev[f"{f}_{j}"] = (float(np.median(x)), float(max(x)))
    away, toward = [], []
    for r in R["records"]:
        if r["block"] == "b2" and r["moving"] and r["finger"] == "middle" and r["joint"] == "yaw" and r["dir_this"] != 0:
            along = -(r["deg"] - r["cmd"]) * r["dir_this"]                 # short along the move: positive
            (away if r["cmd"] * r["dir_this"] > 0 else toward).append(along)
    ng, ps = C["by_sign"]["-1"], C["by_sign"]["1"]
    asym = [(ps[1] + ps[0] * e) - (ng[1] + ng[0] * e) for e in (3, 6)]
    ups = [b[f"{f}_yaw"]["median_up_Nm"] - b[f"{f}_yaw"]["median_down_Nm"] for f in ("index", "middle")]
    return {"vmin": min(V), "vmax": max(V), "band": C["dead_band_steps"], "band_deg": C["dead_band_steps"] * STEP,
            "slope": C["slope_per_step"], "icpt": C["intercept"], "k": R["stiffness_Nm_per_rad"],
            "c": R["c_Nm_per_unit"], "stall": R["stall_Nm"], "punch": R["punch_Nm"], "full_steps": full,
            "full_deg": full * STEP, "yaw_band": yaw_band, "flex_band": flex_band, "thumb_yaw_band": b["thumb_yaw"]["half_width_Nm"],
            "tg": tg, "short_pos": (min(short_pos), max(short_pos)), "short_neg": (min(short_neg), max(short_neg)),
            "t_in_med": float(np.median(t_in)), "same_max": max(same), "neg": C["by_sign"]["-1"], "pos": C["by_sign"]["1"],
            "plant": R["plant"], "dev": dev, "dev_med": float(np.median([v[0] for v in dev.values()])),
            "dev_max": max(dev.items(), key=lambda kv: kv[1][1]), "away": max(away), "toward": max(toward),
            "asym": asym, "updown": ups, "mcp_band": mcp_band, "pip_band": pip_band, "zero_hold": zero_hold,
            "index_yaw_centre": b["index_yaw"]["centre_Nm"]}


# ------------------------------------------------------------------------------------------------- sections
def sec_terms():
    terms = [
        ("Servo frame", "Each servo's angle from its calibrated zero (<code>FINGER_JOINTS</code> in the CB1's driver), in "
         "its own sense of rotation. The index's and the middle's yaw are minus the plans' yaw; the other seven angles "
         "equal the plans'."),
        ("Encoder step", "The servo's position resolution, 300/1024 = 0.293 deg."),
        ("Error $e$", "Measured minus commanded angle, in the servo frame. At a dwell it is the mean over the dwell's "
         "telemetry samples (about 8.6 a second)."),
        ("Load", "The servo's present-load register: the firmware's motor duty in per mille of full duty, signed like "
         "$e$. The drive $d = -\\mathrm{load}$ pushes the joint toward its command."),
        ("Dead band", "The errors at which the drive is zero."),
        ("Torque per unit of drive $c$", "The motor torque at stall per per mille of duty. Here it is the datasheet's stall "
         "torque, 2.3 kg cm (0.23 N m) at 6 V, scaled in proportion to the measured supply: $c$ = 0.192 N m per 1000 at "
         "5.1 V."),
        ("Stiffness $k$", "The torque per radian of error outside the dead band: $c$ times the drive per encoder step, over "
         "the step in radians."),
        ("Gravity torque $\\tau_g$", "The torque gravity puts on a joint at the dwell's measured angles, from the cad_v3 "
         "model of the hand on the UR5 (<code>ur5_full_k1.xml</code>) at the arm's pose of the SR2 night (key "
         "<code>sr2_night</code>). Positive $\\tau_g$ pushes the joint toward positive angles."),
        ("Friction band", "A joint at rest stays where $|c\\,d + \\tau_g - \\tau_b| \\le f$, with $f$ the static friction "
         "and $\\tau_b$ a constant bias. Both come from the steady dwells (2 s and 10 s, in blocks a2, b2, c and c40) of a "
         "servo that has moved in its block. The band runs from the 5th to the 95th percentile of $c\\,d + \\tau_g$ there: "
         "$\\tau_b$ is its midpoint and $f$ its half-width."),
        ("Loop width", "In a staircase, the mean angle at a command reached moving down minus that reached moving up."),
        ("Shortfall", "In a yaw step of block c or c40, the final angle (the median of the hold's last 2 s) minus the "
         "command."),
        ("Blocks", "In a1 and a2, each mcp and pip goes from 5 to 61 deg and back in 4 deg steps (the index's mcp to "
         "49). Their dwells are 0.45 s and 2 s. In b1 and b2, each yaw goes from -36 to +36 deg and back, with the same "
         "dwells. In c, each yaw steps from 0 to +A, back, to -A and back for A = 20, 30 and 40 deg, each pose held 10 s. "
         "Block c40 repeats c with the finger's mcp at 40 deg."),
    ]
    return '<dl class="terms">' + "".join(f"<dt>{a}</dt><dd>{b}</dd>" for a, b in terms) + "</dl>"


def sec_run(R):
    rows = [("a1", "186", "0.45", "3.1"), ("b1", "120", "0.45", "2.0"), ("a2", "186", "2", "7.5"), ("b2", "120", "2", "4.9"),
            ("c", "36", "10", "6.8"), ("c40", "42", "10", "7.1")]
    return (
        "<p>The hand hung on the UR5's flange with the arm powered off, its gantries at the home layout, and nothing "
        "touched the fingers. The lab ran six blocks through <code>lab/run_plan.py</code> from 13:12:49 to 13:44:50 on "
        "2026-11-06. Each step moved one joint and held the other eight at zero, with every command sent as one step "
        "(<code>--ramp-deg 0</code>). Every block ended with its plan done: no pose was blocked and no safety stop came. "
        "The runner logged the station's telemetry at about 8.6 samples a second. The station was then stopped, and the "
        "registers were read with READ packets only (<code>lab/scs0009_registers.py</code>).</p>"
        "<p>Before the run, the plans' clearance was checked in the cad_v2 hand with the middle finger's gantry at four "
        "positions, since its last move home was not seen. The least clearance with every joint inside its error box "
        "was 20.2 mm; with the middle at home it was 68.0 mm (QUEUE R36).</p>"
        + table(["Block", "Steps", "Dwell (s)", "Duration (min)"], rows,
                "The six blocks of the identification run, in the order they ran.")
    )


def sec_registers(R, N):
    G = R["regs"]
    same = []
    for addr, label in ((21, "P coefficient"), (22, "D coefficient"), (23, "I coefficient"), (24, "Minimum drive (punch), per mille"),
                        (26, "Dead band, clockwise (steps)"), (27, "Dead band, counter-clockwise (steps)"),
                        (16, "Maximum torque, per mille"), (37, "Protective torque (%)"), (38, "Protection time"),
                        (39, "Overload torque (%)"), (41, "Acceleration"), (46, "Goal speed, as left by the station"),
                        (82, "Maximum velocity, in 50 steps/s")):
        vals = {G[s][addr][1] for s in G if addr in G[s]}
        same.append((label, f"{addr}", ", ".join(sorted(vals))))
    lim = []
    for s in sorted(G):
        lo, hi = G[s][9][1].split()[0], G[s][11][1].split()[0]
        lim.append((["Thumb", "Index", "Middle"][s // 3] + " " + J[s % 3], str(s), G[s][9][0], lo, G[s][11][0], hi))
    return (
        f"<p>The nine servos have the same gains, dead bands and protection settings. With torque off, their supply "
        f"read {N['vmin']:.1f} to {N['vmax']:.1f} V, against the 6 V of the datasheet's stall torque.</p>"
        + table(["Register", "Address", "Value on all nine"], same,
                "Registers equal on the nine servos (<code>data/registers.txt</code>; words big-endian).", num_from=1)
        + "<p>The station sends its speed setting to rustypot, which takes radians per second. The goal speed was "
        "11,734 steps/s, the 60 rad/s of the runner's last move to zero. The plans' speed 80 therefore asked for "
        "80 rad/s, or 4,580 deg/s. The value in register 82 is 40, which in rustypot's units of 50 steps/s is "
        "2,000 steps/s (586 deg/s). Every move of the plans, here and on 09-02, ran at the servo's own top speed.</p>"
        "<p>Three servos have angle limits inside their range of travel. The other six have 00 00 and ff 03, which "
        "big-endian are 0 and 65,283 steps: no upper limit. The bytes ff 03 are 1,023 written little-endian. The "
        "index's mcp stops at 1,001 steps, +58.6 deg from its zero, and the index's yaw at 639 steps, +47.8 deg in its "
        "servo frame (plan yaw -47.8). The plans keep the index's mcp at 50 deg or less and its yaw within 40 deg.</p>"
        + table(["Servo", "ID", "Minimum, bytes", "Minimum, steps", "Maximum, bytes", "Maximum, steps"], lim,
                "Angle limits of the nine servos (registers 9 and 11).", num_from=1)
    )


def sec_controller(R, N):
    return (
        f"<p>The drive is zero while the error stays within {N['band']} encoder steps ({N['band_deg']:.2f} deg) of the "
        f"command. Beyond that it grows by {N['slope']:.1f} per mille a step, which matches the P coefficient of 15. A "
        f"line through the medians from 4 steps out gives</p>"
        f"$$d = -\\operatorname{{sign}}(e)\\,\\bigl({N['icpt']:.1f} + {N['slope']:.1f}\\,|e|\\bigr)\\quad\\text{{for }}"
        f"|e| > {N['band']},$$"
        f"<p>with $e$ in steps and $d$ in per mille. The first drive outside the band is about 70 per mille, or "
        f"{1000 * N['punch']:.1f} mN m at the datasheet's scale. Full duty comes at {N['full_steps']:.0f} steps, "
        f"{N['full_deg']:.1f} deg of error. The servo drives harder toward negative angles. For a negative error (drive "
        f"positive) the line is {N['neg'][1]:.1f} + {N['neg'][0]:.1f}&thinsp;$|e|$, and for a positive error it is "
        f"{N['pos'][1]:.1f} + {N['pos'][0]:.1f}&thinsp;$|e|$. At errors of 3 and 6 steps the drive toward negative angles is "
        f"{N['asym'][0]:.0f} and {N['asym'][1]:.0f} per mille larger, about one step's worth.</p>"
        + img(MEDIA / "20261106-drive_against_error.png",
              "Load against error over every servo's dwells, all six blocks. The grey band has no drive. The red lines "
              "are fitted to the medians from 4 steps out, one for each sign of the error.")
        + f"<p>At $c$ = {1000 * N['c']:.3f} mN m per unit of drive the slope is a stiffness of</p>"
        f"$$k = \\frac{{{N['slope']:.1f}\\,c}}{{0.293^\\circ}} = {N['k']:.2f}\\ \\mathrm{{N\\,m\\,rad^{{-1}}}}.$$"
        f"<p>The servo plant of the HOM scripts (<code>HOS_PLANT=servo</code>) has {N['plant']['kp']}, taken from the "
        f"datasheet's 0.23 N m at 18 deg of error. The supply's 5.1 V and the {N['full_deg']:.1f} deg to full duty account for the "
        f"difference: 0.72 &times; 5.1/6 &times; 18/{N['full_deg']:.1f} = {0.72 * 5.1 / 6 * 18 / N['full_deg']:.2f}.</p>"
    )


def sec_friction(R, N):
    b, L = R["bands"], R["loops"]
    rows = []
    for f in F:
        for j in J:
            k = f"{f}_{j}"
            rows.append((name(k), str(b[k]["n"]), mnm(b[k]["p5_Nm"]), mnm(b[k]["p95_Nm"]), mnm(b[k]["centre_Nm"]),
                         mnm(b[k]["half_width_Nm"]),
                         mnm(b[k]["zero_drive_tau_g_max_Nm"]), mnm(b[k]["tau_g_max_Nm"])))
    lrows = []
    for f in F:
        for j in J:
            k = f"{f}_{j}"
            blks = ("b1", "b2") if j == "yaw" else ("a1", "a2")
            lrows.append((name(k), *[f"{L[x][k]['median']:.2f}" if k in L[x] else "&ndash;" for x in blks],
                          *[f"{L[x][k]['max']:.2f}" if k in L[x] else "&ndash;" for x in blks]))
    P_ = N["plant"]
    lo_ = lambda v: f"{1000 * min(v):.0f} to {1000 * max(v):.0f}"  # noqa: E731
    prows = [("Stiffness (N m/rad)", f"{N['k']:.2f}", f"{P_['kp']}"),
             ("Torque at full duty (N m)", f"{N['stall']:.3f}", f"{P_['forcerange']} (force range)"),
             ("Dead band (deg)", f"&plusmn;{N['band_deg']:.2f}", "0"),
             ("First drive outside the band (mN m)", f"{1000 * N['punch']:.1f}", "0"),
             ("Static friction, index and middle yaw (mN m)", lo_(N["yaw_band"]), "0"),
             ("Static friction, thumb yaw (mN m)", f"{1000 * N['thumb_yaw_band']:.0f}", "0"),
             ("Static friction, mcp joints (mN m)", lo_(N["mcp_band"]), "0"),
             ("Static friction, pip joints (mN m)", lo_(N["pip_band"]), "0"),
             ("Joint damping (N m s/rad)", "not measured", f"{P_['damping']}")]
    zk, zv = N["zero_hold"]
    return (
        f"<p>Gravity alone cannot set the torque scale on the hanging hand. It reaches {1000 * N['tg']['mcp']:.0f} mN m on "
        f"the mcp joints at 61 deg, {1000 * N['tg']['yaw']:.0f} mN m on the yaws at 40 deg and {1000 * N['tg']['pip']:.1f} "
        f"mN m on the pip joints. A least-squares fit of the drive to these torques gave a scale near zero, because "
        f"the friction is as large. The scale here is therefore the datasheet's, and each servo's friction is read "
        f"from the spread of $c\\,d + \\tau_g$ over its steady dwells (Figure 3).</p>"
        f"<p>On the 2 s staircases the flexion joints come to rest a median of {N['dev_med']:.2f} deg from the gravity "
        f"droop that the stiffness predicts, on both sides of it. The largest distance is {N['dev_max'][1][1]:.1f} deg, on "
        f"the {name(N['dev_max'][0]).lower()}. The middle yaw stops short of each command along its move: by up to "
        f"{N['away']:.1f} deg on moves away from zero, against gravity, and by up to {N['toward']:.1f} deg on moves "
        f"toward zero (Figure 2).</p>"
        + img(MEDIA / "20261106-staircases.png",
              "Error at each dwell of the 2 s staircases, for four joints. Red marks were reached moving up, blue moving "
              "down. The dashed line is the droop that gravity gives at 0.55 N m/rad with no friction; the grey band is "
              "the dead band.")
        + img(MEDIA / "20261106-friction_bands.png",
              "Drive torque plus gravity, $c\\,d + \\tau_g$, at the steady dwells of each servo after a move of its "
              "own. A joint at rest stays within its friction band. Red points were last moved up and blue down. The "
              "long black bars are the 5th and 95th percentiles, $\\tau_b \\pm f$, and the short bar is $\\tau_b$.")
        + f"<p>The index and middle yaws stick within &plusmn;{1000 * N['yaw_band'][0]:.0f} and "
        f"&plusmn;{1000 * N['yaw_band'][1]:.0f} mN m, more than the {1000 * N['tg']['yaw']:.0f} mN m of gravity at "
        f"40 deg. In angle, 25 mN m is {np.degrees(0.025 / N['k']):.1f} deg at {N['k']:.2f} N m/rad. The thumb yaw sticks within &plusmn;{1000 * N['thumb_yaw_band']:.0f} mN m, the mcp joints within "
        f"&plusmn;{lo_(N['mcp_band'])} mN m and the pip joints within &plusmn;{lo_(N['pip_band'])} mN m. The "
        f"{name(zk).lower()} held {1000 * zv:.1f} mN m of gravity at a dwell with no drive at all, more than its band. "
        f"That torque is a lower bound on its static friction; the band is the spread of the friction the joint showed "
        f"at most of its dwells.</p>"
        + table(["Servo", "Dwells", "5th percentile (mN m)", "95th percentile (mN m)", "Centre &tau;<sub>b</sub> (mN m)",
                 "Half-width <i>f</i> (mN m)", "Largest gravity held with no drive (mN m)", "Largest |&tau;<sub>g</sub>| (mN m)"],
                rows, "Friction bands of the nine servos at $c$ = 0.192 N m per 1000, over the steady dwells after a "
                "move of the servo's own.")
        + table(["Servo", "Median, pass 1", "Median, pass 2", "Largest, pass 1", "Largest, pass 2"], lrows,
                "Loop widths of the staircases (deg): blocks a1 and a2 for mcp and pip, b1 and b2 for the yaws. Pass 1 "
                "dwells 0.45 s and pass 2 dwells 2 s.")
        + "<p>The lab read 09-02's staircase on the real_v1 hand with the same servos (<code>scripts/archive/"
        "servo_hysteresis_sweep.py</code>), by its own definition of the loop. Its medians were 0.59 to 1.17 deg on the "
        "flexion joints, against 0.07 to 1.14 deg on 11-06 by the same definition.</p>"
        + table(["Quantity", "Measured, 11-06", "HOM servo plant"], prows,
                "The servo as measured against the plant of the HOM scripts "
                "(<code>kp0.72_kv0_fr0.18_fl0_dp0.0144</code>). The torques use the datasheet's scale at 5.1 V.",
                num_from=1)
    )


def sec_yaw(R, N):
    st = stall_table(R)
    rows = []
    for f in F:
        for sg, side in ((1, "positive"), (-1, "negative")):
            d = st.get((f, sg), {})
            rows.append((f"{f.capitalize()}, {side}", *[f1(d[(blk, a)]["short_deg"], 2) if (blk, a) in d else "&ndash;"
                                                        for blk in ("c", "c40") for a in (20.0, 30.0, 40.0)]))
    return (
        f"<p>Each yaw step stopped short of its command toward zero, within {N['t_in_med']:.2f} s (median) of the "
        f"command, and stayed there to within one or two encoder steps for the rest of the 10 s hold. On the servos' "
        f"positive side the shortfall was {-N['short_pos'][1]:.2f} to {-N['short_pos'][0]:.2f} deg, and on the negative "
        f"side {N['short_neg'][0]:.2f} to {N['short_neg'][1]:.2f} deg. With the finger's mcp at 40 deg, the "
        f"shortfalls changed by {N['same_max']:.2f} deg at most, although bending the finger shortens the yaw's gravity "
        f"arm. The middle yaw fell 3.01, 3.05 and 3.09 deg short at 20, 30 and 40 deg, while the gravity torque nearly "
        f"doubles between those angles.</p>"
        + film(MEDIA / "20261106-yaw_steps_replay.mp4",
               "Block c's yaw steps away from zero, replayed on the cad_v3 hand as it hangs from the measured telemetry. "
               "Each finger is seen along its yaw axis. The red line runs from the yaw axis through the fingertip at the "
               "commanded angle, and the blue line at the measured one. The first 2 s of each step play at real time; "
               "<code>scripts/hardware/sr2_servo_sysid_film.py</code>.")
        + img(MEDIA / "20261106-yaw_holds.png",
              "The moving yaw's error through each 10 s hold of blocks c (solid) and c40 (dashed), for the steps away from "
              "zero. The lower curves are the steps to positive angles in the servo frame.")
        + table(["Yaw, side", "c, 20", "c, 30", "c, 40", "c40, 20", "c40, 30", "c40, 40"], rows,
                "Shortfall of the yaw steps away from zero (deg), by block and step size in deg.")
        + "<p>A shortfall that stays constant and does not follow the gravity torque is set by friction. The servo "
        "stops where its drive, less gravity, falls inside the friction band. The weaker drive toward positive angles "
        "(Section SEC[controller]) accounts for about one encoder step (0.3 deg) of the difference between the two "
        "sides. On 11-05 the middle yaw stuck 6.2 deg short at the layout g12, and at another pose it stuck for 5 s "
        "before it moved on. At the home layout no hold drifted by more than two steps.</p>"
    )


def sec_index(R, N):
    S = night_stop()
    return (
        f"<p>On 11-04 at 19:31, step 126 of <code>run1_home</code> sent the index's mcp from {S['cmd_before']:.1f} to "
        f"{S['cmd']:.0f} deg. It moved from "
        f"{S['before']:.1f} to {S['top']:.2f} deg and stayed there for {S['dur']:.1f} s. All {S['n_steady']} samples "
        f"there read a load of {', '.join(str(int(x)) for x in S['loads'])}, until the runner backed the joint out "
        f"and the next read {S['back_out_load']:+.0f}. The crosstalk experiment's README "
        f"called this the servo's protective torque. The register read shows a maximum angle of 1,001 steps, +58.6 deg from the "
        f"index mcp's zero. The servo clamps its goal there, and 3.2 deg of droop leaves the joint at 55.4 deg.</p>"
        f"<p>The load fits the clamp. The drive line of Section SEC[controller] gives about 190 per mille for 3.2 deg "
        f"(11 steps). An unclamped goal at 64 deg would leave 29 steps of error and about "
        f"{N['icpt'] + N['slope'] * 29:.0f} per mille. Before the step, at {S['cmd_before']:.1f} deg, the joint drooped "
        f"{S['err_before']:.1f} deg with the same load of {S['before_loads'][0]:.0f} to {S['before_loads'][-1]:.0f}. The lab corrected the crosstalk README on 11-06. The plans keep the index's mcp at 50 deg "
        f"or less.</p>"
    )


def sec_limits():
    items = [
        "The torque scale is the datasheet's stall torque, scaled in proportion to the supply voltage read with torque "
        "off. Neither the stall torque nor the voltage under load was measured. Every torque on this page scales with it.",
        "The gravity torques come from the cad_v3 model's masses and the arm's pose stored as the key <code>sr2_night</code>. "
        "The masses of the printed parts, servos, boards and cables were not weighed.",
        "The telemetry runs at about 8.6 samples a second, so step responses faster than about 0.2 s are not resolved. "
        "The first sample of a yaw step, 0.075 s after the command, already reads a load of 990.",
        "All dwells were taken at the home layout. The fingertip cables lay on the table, so a yaw's swing may have "
        "dragged them, and their drag is part of the yaw's friction band.",
        "Damping and the servo's dynamics were not identified. Those steps of the protocol need the bus at about "
        "111 Hz with the station stopped.",
    ]
    return "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"


def sec_future():
    items = [
        "Measure the torque scale: hang known masses on a finger at a known arm, or press the tip on a load cell "
        "(protocol of 10-13, steps 3 and 7). A stall torque more than 15 % from 0.192 N m at the measured supply "
        "changes every torque here in proportion.",
        "Read the supply under load: register 62 during a held press. A drop below 5.0 V lowers the stall torque further.",
        "Put the measured servo in the HOM plant: kp 0.55 N m/rad and a 0.59 deg dead band. Set frictionloss to 25 mN m "
        "on the index and middle yaws, 10 mN m on the thumb yaw and the mcp joints and 4 mN m on the pip joints. Then rerun "
        "<code>scripts/hom/hom_plant_transfer.py --set plants</code>. A landscape whose best layouts change would make the "
        "friction a design term.",
        "Repeat block c at the layout g12, where the middle yaw stuck 6.2 deg short on 11-05. A shortfall over 4 deg there "
        "would put the extra load in the layout, through the cables or the gantry, outside the servo.",
        "Rewrite the six servos' maximum angle limits big-endian (1,023), or leave them as they are with no limit. Decide "
        "whether the index's limits of 58.6 deg on the mcp and 47.8 deg on the yaw should stay.",
    ]
    return "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"


def sec_files():
    rows = [("docs/experiments/20261106-sr2_servo_sysid/data/dwells.tsv", "one row a dwell: command, every servo's mean "
             "angle and load (lab/sysid_tables.py)"),
            ("docs/experiments/20261106-sr2_servo_sysid/data/holds_c.tsv", "the moving yaw's samples in each hold of c "
             "and c40"),
            ("docs/experiments/20261106-sr2_servo_sysid/data/registers.txt", "the nine servos' registers, raw and decoded"),
            ("docs/experiments/20261106-sr2_servo_sysid/data/20261106-sr2_servo_sysid_fit.json", "the fit and every dwell "
             "record with its gravity torque"),
            ("scripts/hardware/sr2_servo_sysid_fit.py", "the fit (under resguard; it loads the cad_v3 model)"),
            ("scripts/hardware/sr2_servo_sysid_page.py", "this page and its figures"),
            ("scripts/hardware/sr2_servo_sysid_film.py", "the film of the yaw steps (under resguard)"),
            ("docs/experiments/20261104-sr2_hand_crosstalk/lab/", "make_sysid_plan.py, sysid_chain.sh, sysid_tables.py, "
             "scs0009_registers.py, push_text.py"),
            ("logs/20261106-sr2_sysid/ on corvus", "the full events, telemetry and D435 frames (outside git while GitHub "
             "refuses LFS transfers)"),
            ("docs/experiments/20261013-servo_identification/20261013-servo_identification_protocol.html", "the protocol")]
    return "<ul>" + "".join(f"<li><code>{a}</code>: {b}</li>" for a, b in rows) + "</ul>"


def lede_and_summary(N):
    lede = (f"The SR2 hand's nine SCS0009 servos give no drive within {N['band_deg']:.2f} deg of their command. Beyond "
            f"that the drive grows by {N['slope']:.1f} per mille of full duty per 0.29 deg step. At the datasheet's stall "
            f"torque scaled to the measured {N['vmin']:.1f} to {N['vmax']:.1f} V supply, that is {N['k']:.2f} N m/rad. "
            f"The servo plant of the HOM scripts has 0.72 N m/rad, with no dead band and no friction. The index and middle yaws stick "
            f"within &plusmn;{1000 * min(N['yaw_band']):.0f} to &plusmn;{1000 * max(N['yaw_band']):.0f} mN m of static "
            f"friction, more than the {1000 * N['tg']['yaw']:.0f} mN m of gravity at 40 deg. The mcp joints held up to "
            f"{1000 * N['zero_hold'][1]:.0f} mN m of gravity with no drive at all. "
            f"Yaw steps of 20 to 40 deg stop {-N['short_pos'][1]:.1f} to {-N['short_pos'][0]:.1f} deg short on the "
            f"servo's positive side and stay there for 10 s.")
    summary = (f"The SR2 hand's SCS0009 servos have a {N['band_deg']:.2f} deg dead band and {N['k']:.2f} N m/rad at the "
               f"datasheet's torque scaled to the measured 5.1 V supply (HOM plant 0.72, no friction). The index and "
               f"middle yaws stick within about 25 mN m of friction and stop 1.5 to 3.1 deg short of 20 to 40 deg steps.")
    return lede, summary


def render(page):
    items = [((m.group(1) if m.group(1) is not None else m.group(2)).strip(), m.group(1) is not None)
             for m in P.TEX_RE.finditer(page)]
    svgs = texsvg.render(items, cache_path=TEX_CACHE)
    it = iter(svgs)
    return P.TEX_RE.sub(lambda m: next(it), page), len(items)


def cmd_page():
    R = load()
    N = numbers(R)
    sections = [
        ("terms", "Terms and metrics", sec_terms()),
        ("run", "Identification run on the hanging hand", sec_run(R)),
        ("registers", "Registers of the nine servos", sec_registers(R, N)),
        ("controller", "Drive against position error", sec_controller(R, N)),
        ("friction", "Droop and static friction against gravity", sec_friction(R, N)),
        ("yaw", "Yaw steps of 20 to 40 deg", sec_yaw(R, N)),
        ("index", "Stop of the index mcp at 55.4 deg on 11-04", sec_index(R, N)),
        ("limits", "Limitations", sec_limits()),
        ("future", "Future work", sec_future()),
        ("files", "Files", sec_files()),
    ]
    num_ = {sid: k for k, (sid, _h, _b) in enumerate(sections, 1)}
    sub = (lambda s_: re.sub(r"SEC\[(\w+)\]", lambda m_: str(num_[m_.group(1)]), s_))
    unit = (lambda s_: re.sub(r"(\d) (%|mm|deg|kg|px|V\b|N\b|ms\b|s\b|g\b|m\b|min\b)", "\\1&nbsp;\\2", s_))
    body = [(sid, hd, unit(sub(b))) for sid, hd, b in sections]
    toc = "\n".join(f'<li><a href="#{sid}">{k}. {esc(hd)}</a></li>' for k, (sid, hd, _) in enumerate(body, 1))
    secs = "\n".join(f'<section id="{sid}"><h2>{k}. {esc(hd)}</h2>{b}</section>' for k, (sid, hd, b) in enumerate(body, 1))
    lede, summary = lede_and_summary(N)
    byline = ("Measured by the UR5 lab session on 2026-11-06, 13:12 to 13:45, on the hand as it hangs on the UR5; fitted "
              "on the workstation. Builder: <code>scripts/hardware/sr2_servo_sysid_page.py</code>.")
    page = TEMPLATE.read_text()
    for k_, v in {"TITLE": "SR2 servo identification",
                  "H1": "Static identification of the SR2 hand's SCS0009 servos in free space on the UR5",
                  "EYEBROW": "Hand and bench, 2026-11-06", "LEDE": unit(lede), "BYLINE": byline, "TOC": toc,
                  "SECTIONS": secs, "FOOTER": str(OUT.relative_to(ROOT)), "TEXCSS": texsvg.CSS}.items():
        page = page.replace("{{" + k_ + "}}", v)
    page, n = render(page)
    page = docs_index.tag(page, topic="hardware", summary=summary, date="2026-11-06")
    page = retro_style.apply(page)
    OUT.write_text(page)
    print(f"{OUT}: {OUT.stat().st_size / 1e6:.2f} MB, {n} formulas", flush=True)
    return 0


def main():
    a = sys.argv[1:]
    if a and a[0] == "figs":
        return cmd_figs()
    return cmd_page()


if __name__ == "__main__":
    sys.exit(main())
