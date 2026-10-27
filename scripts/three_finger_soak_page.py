#!/usr/bin/env python3
"""Page on the three v1 fingertips left at rest overnight on the ESP32-C6 mux (hand-off 20261026-three_v1_fingers_overnight;
owner, 2026-10-26 19:30: "ive plugged in all v1 finger ... dont know how much you can do with them just sitting there
plugged in (they are not close to each other either), but maybe there's some more testing you can do with all three on").
Reads the rows that src/morphohand/driver/reskin5x/host/reskin5x_mux_check.py, _rest.py, _soak.py, _allan.py, _steps.py
and _recovery.py wrote to docs/experiments/20261027-three_v1_fingers_overnight/data/, draws the figures into media/ and
writes docs/experiments/20261027-three_v1_fingers_overnight/20261027-three_v1_fingertips_at_rest_overnight.html.

    .venv/bin/python scripts/three_finger_soak_page.py

The commands that wrote the rows are in the page's Section 2 and in the hand-off note.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import docs_index  # noqa: E402
import retro_style  # noqa: E402
import texsvg  # noqa: E402
import ur5_tip_calibration_page as P  # noqa: E402   (esc, fig, table, plt_setup, TEX_RE)

D = ROOT / "docs/experiments/20261027-three_v1_fingers_overnight"
DATA = Path(os.environ.get("SOAK_PAGE_DATA", D / "data"))         # a test build reads its rows and writes elsewhere
MEDIA = DATA.parent / "media"
PAGE = (D if "SOAK_PAGE_DATA" not in os.environ else DATA.parent) / "20261027-three_v1_fingertips_at_rest_overnight.html"
TEMPLATE = ROOT / "scripts/three_finger_soak_page.template.html"
TEX_CACHE = D / "texsvg_cache.json"
MUX_LOG = ROOT / "logs/20261026-three_finger_mux/mux.log"
EXP = ROOT / "docs/experiments"
L = dict(
    estimation=EXP / "20261020-fingertip_contact_estimation/20261020-model_based_contact_estimation.html",
    foreign=EXP / "20261023-foreign_field_rejection",
    overview=ROOT / "docs/overviews/printed_fingertips.html",
    readme=ROOT / "src/morphohand/driver/reskin5x/README.md",
)
NAMES = ("ID 1", "ID 2", "ID 3")
CH = ("ch0", "ch1", "ch2")
COL = ("#1f5fbf", "#c0392b", "#2e8b57")
CHIP_COL = ("#000000", "#1f5fbf", "#c0392b", "#2e8b57", "#e8a35c")
BAD_EXAMPLE = "19:42:21"
LOG = "logs/20261026-three_finger_mux/20261026-193314-mux.bin"
H = "src/morphohand/driver/reskin5x/host"
OUT = "docs/experiments/20261027-three_v1_fingers_overnight/data"
CAL = ("logs/20261020-contact_estimation/calib/20261026-134217-id_1-calib.json",
       "docs/experiments/20261022-ur5_tip_calibration/data/20261025-152920-cb2t/calib_rest.json",
       "logs/20261020-contact_estimation/calib/20261025-112147-calib.json")
COMMANDS = f"""py=~/miniconda3/bin/python; LOG={LOG}; OUT={OUT}
$py {H}/reskin5x_mux_check.py $LOG --since 19:35 --until 19:45 --out $OUT/check_1935.json     # also 23:00-23:10, 03:50-04:00
$py {H}/reskin5x_mux_check.py $LOG --since 19:35 --until 04:00 --scan --out $OUT/check_night.json
$py {H}/reskin5x_mux_rest.py $LOG --ref 19:35:00 19:40:00 --since 19:35 --until 04:00 --names "ID 1" "ID 2" "ID 3" \\
    --bin-s 60 --tempfit --out $OUT/rest.json --series $OUT/rest_60s.npz          # and --no-thermal: rest_raw.json, rest_60s_raw.npz
$py {H}/reskin5x_mux_rest.py $LOG --ref 19:35:00 19:40:00 --since 19:35 --until 04:00 --names "ID 1" "ID 2" "ID 3" \\
    --bin-s 1 --series $OUT/rest_1s.npz
$py {H}/reskin5x_mux_steps.py $OUT/rest_1s.npz --out $OUT/steps.json             # and --min-uT 1.5: steps_1p5.json
.venv/bin/python {H}/reskin5x_mux_allan.py $LOG --since 19:35 --until 04:00 --names "ID 1" "ID 2" "ID 3" --out $OUT/allan.json
.venv/bin/python {H}/reskin5x_mux_soak.py $LOG --since 19:35 --until 04:00 --channels 0 1 2 --floor 2 \\
    --calib ch0={CAL[0]} ch1={CAL[1]} ch2={CAL[2]} --out $OUT/soak_floor2.json   # and --floor 1: soak_floor1.json
$py {H}/reskin5x_mux_recovery.py --names "ID 1" "ID 2" --ref 13:42:30 13:43:30 --presses 13:49 14:00 \\
    --streams <20261026-134212-id_1-stream.bin>,<20261026-141207-id_1-stream.bin> <the same for id_2> \\
    --mux $LOG --mux-index 0 1 --mux-since 19:35 --mux-until 04:00 --out $OUT/recovery.json"""


def link(p, text):
    return f'<a href="{P.esc(os.path.relpath(p, D))}">{text}</a>'


def j(name):
    return json.loads((DATA / name).read_text())


def hours_since(t, t0):
    return (np.asarray(t) - t0) / 3600.0


def clock_ticks(ax, t0, h_max, step=None):
    """Clock times (HH:MM) on the whole hours of an axis of hours since t0 (every second hour past 10 h)."""
    step = step or (2.0 if h_max > 10 else 1.0)
    first = (np.ceil(t0 / 3600.0) * 3600.0 - t0) / 3600.0
    ticks = np.arange(first, h_max + 0.01, step)
    ax.set_xticks(ticks)
    ax.set_xticklabels([time.strftime("%H:%M", time.localtime(t0 + 3600 * h)) for h in ticks], rotation=0)


# ------------------------------------------------------------------------------------------------------- the splitter log
def mux_log_summary():
    """Rate lines and restarts of the splitter's log (logs/ is not tracked, so the numbers go to data/mux_log.json)."""
    out = DATA / "mux_log.json"
    if MUX_LOG.exists():
        rates, restarts, lines = [], [], MUX_LOG.read_text().splitlines()
        for ln in lines:
            m = re.match(r"(\d\d:\d\d:\d\d) ([0-9.]+) records/s", ln)
            if m:
                rates.append((m.group(1), float(m.group(2))))
            elif "waiting for the board" in ln or "boards on mux channels" in ln:
                restarts.append(ln[:160])
        R = np.array([r for _, r in rates])
        out.write_text(json.dumps(dict(lines=len(rates), first=rates[0][0], last=rates[-1][0], rate_min=float(R.min()),
                                       rate_max=float(R.max()), rate_median=float(np.median(R)),
                                       starts=restarts), indent=1) + "\n")
    return json.loads(out.read_text())


# ------------------------------------------------------------------------------------------------------------- figures
def fig_rest(out):
    """Die temperature, change of the rest uncorrected and corrected, and the corrected change split, per board."""
    plt = P.plt_setup()
    z, zr = np.load(DATA / "rest_60s.npz"), np.load(DATA / "rest_60s_raw.npz")
    U = z["U"]
    t0 = float(z["0_t"][0])
    fig, ax = plt.subplots(3, 3, figsize=(13.5, 9.6), dpi=110, sharex=True)
    for b in range(3):
        h = hours_since(z[f"{b}_t"], t0)
        T, Dc, Dr, G = z[f"{b}_T"], z[f"{b}_d"], zr[f"{b}_d"], z[f"{b}_g"]
        Lf = Dc - G @ U.T
        for c in range(5):
            ax[0, b].plot(h, T[:, c], color=CHIP_COL[c], lw=1.0, label=f"Chip {c}")
        ax[1, b].plot(hours_since(zr[f"{b}_t"], t0), np.linalg.norm(Dr, axis=1), color="#999999", lw=1.0, label="Uncorrected")
        ax[1, b].plot(h, np.linalg.norm(Dc, axis=1), color="#000000", lw=1.0, label="Corrected")
        ax[2, b].plot(h, np.linalg.norm(G, axis=1), color="#1f5fbf", lw=1.0, label="Uniform field")
        ax[2, b].plot(h, np.linalg.norm(Lf, axis=1), color="#e8a35c", lw=1.0, label="Part left")
        ax[0, b].set_title(NAMES[b])
        clock_ticks(ax[2, b], t0, h[-1], step=2.0)
        for k in range(3):
            ax[k, b].grid(alpha=0.3)
    ax[0, 0].set_ylabel("Die temperature (\u00b0C)")
    ax[1, 0].set_ylabel("Change from 19:35 (\u00b5T)")
    ax[2, 0].set_ylabel("Corrected change (\u00b5T)")
    ax[0, 2].legend(frameon=False, fontsize=11, ncol=2)
    ax[1, 2].legend(frameon=False)
    ax[2, 2].legend(frameon=False)
    for b in range(3):
        ax[2, b].set_xlabel("Clock time")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_allan(out, A, Ar):
    plt = P.plt_setup()
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.8), dpi=110, sharey=True)
    for b, name in enumerate(NAMES):
        r = A["boards"][name]
        tau, S = np.array(r["tau_s"]), np.array(r["adev_uT"]).reshape(5, 3, -1)
        xy, zz = S[:, :2].reshape(10, -1), S[:, 2]
        ax[b].fill_between(tau, xy.min(0), xy.max(0), color="#1f5fbf", alpha=0.15, lw=0)
        ax[b].fill_between(tau, zz.min(0), zz.max(0), color="#c0392b", alpha=0.15, lw=0)
        ax[b].plot(tau, np.median(xy, 0), color="#1f5fbf", lw=1.6, label="x and y, median")
        ax[b].plot(tau, np.median(zz, 0), color="#c0392b", lw=1.6, label="z, median")
        rr = Ar["boards"][name]
        xr = np.array(rr["adev_uT"]).reshape(5, 3, -1)[:, :2].reshape(10, -1)
        ax[b].plot(np.array(rr["tau_s"]), np.median(xr, 0), color="#777777", lw=1.4, ls="--", label="x and y, uncorrected")
        for v, lab in ((0.05, "50 ms"), (3.0, "3 s")):
            ax[b].axvline(v, color="#555555", lw=0.8, ls="--")
            ax[b].text(v * 1.1, 4.0, lab, fontsize=12)
        for v in (1.0, 2.0):
            ax[b].axhline(v, color="#555555", lw=0.8, ls=":")
        ax[b].set_xscale("log")
        ax[b].set_yscale("log")
        ax[b].set_title(name)
        ax[b].set_xlabel("Averaging time (s)")
        ax[b].grid(alpha=0.3, which="both")
    ax[0].set_ylabel("Allan deviation (\u00b5T)")
    ax[0].set_ylim(0.01, 6)
    ax[2].legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_steps(out, St):
    plt = P.plt_setup()
    z = np.load(DATA / "rest_1s.npz")
    t0 = float(z["0_t"][0])
    fig, ax = plt.subplots(3, 1, figsize=(13.5, 7.6), dpi=110, sharex=True, sharey=True)
    for b, name in enumerate(NAMES):
        t, G = z[f"{b}_t"], z[f"{b}_g"].astype(float)
        k = np.round((t - t[0])).astype(int)
        g = np.full((k[-1] + 1, 3), np.nan)
        g[k] = G
        c = np.full(len(g), np.nan)
        c[1:-1] = np.linalg.norm(g[2:] - g[:-2], axis=1)
        h = hours_since(t[0] + np.arange(len(g)), t0)
        ax[b].plot(h, c, color=COL[b], lw=0.4)
        ax[b].axhline(St["min_uT"], color="#000000", lw=0.8, ls="--")
        ax[b].set_ylabel("Change over 2 s (\u00b5T)")
        ax[b].text(0.005, 0.85, name, transform=ax[b].transAxes, fontsize=13)
        ax[b].grid(alpha=0.3)
        ax[b].set_ylim(0, max(4.0, St["min_uT"] * 1.3))
    clock_ticks(ax[2], t0, h[-1])
    ax[2].set_xlabel("Clock time")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_recovery(out, R):
    plt = P.plt_setup()
    fig, ax = plt.subplots(2, 1, figsize=(13.5, 7.4), dpi=110, sharex=True)
    t0 = None
    for b, name in enumerate(NAMES[:2]):
        rows = R["boards"][name]["rows"]
        t = np.array([r["t"] for r in rows])
        t0 = t[0] if t0 is None else t0
        still = np.array([r["still"] for r in rows])
        left = np.array([r["left_uT"] for r in rows], float)
        uni = np.array([r["uniform_uT"] for r in rows], float)
        h = hours_since(t, t0)
        for k, (y, lab, col) in enumerate(((left, "Part left", "#e8a35c"), (uni, "Uniform field", "#1f5fbf"))):
            yy = np.where(still, y, np.nan)
            ax[b].plot(h, yy, color=col, lw=1.0, label=lab)
        ax[b].set_ylabel("Change from 13:42 (\u00b5T)")
        ax[b].text(0.005, 0.88, name, transform=ax[b].transAxes, fontsize=13)
        ax[b].grid(alpha=0.3)
        ax[b].set_ylim(0, None)
    for b in range(2):
        p0, p1 = (hours_since(clock_on(R["presses"][k], t0), t0) for k in range(2))
        ax[b].axvspan(p0, p1, color="#cccccc", alpha=0.6, lw=0)
        mv = hours_since(clock_on("19:32", t0), t0)
        ax[b].axvline(mv, color="#000000", lw=0.8, ls="--")
    ax[0].legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.16))
    clock_ticks(ax[1], t0, h[-1])
    ax[1].set_xlabel("Clock time (grey: the presses; dashed: the boards moved and ID 3 plugged in)")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def clock_on(s, t_start):
    h, m = (int(v) for v in s.split(":")[:2])
    lt = time.localtime(t_start)
    c = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, h, m, 0, 0, 0, -1))
    return c + 86400 if c < t_start - 1 else c


# ------------------------------------------------------------------------------------------------------------ sections
def sections(N):
    C, M, R0, R1, S2, S1, A, Ar, St, St15, Rc, Nt = (N[k] for k in ("checks", "mux", "rest", "rest_raw", "soak2", "soak1",
                                                                   "allan", "allan_raw", "steps", "steps15", "recovery",
                                                                   "night"))
    S = []
    hours = St["hours"]
    u, pc, dg = "&#8202;µT", "&#8202;%/&#176;C", "&#8202;&#176;C"
    nm = {n: n.replace(" ", "&#8202;") for n in NAMES}

    def mn(v, f="{:.2f}"):
        """A number with a typographic minus."""
        return f.format(v).replace("-", "&#8722;")

    def lst(vals, f="{:.1f}"):
        """'a, b and c'."""
        v = [f.format(x) for x in vals]
        return v[0] if len(v) == 1 else ", ".join(v[:-1]) + " and " + v[-1]

    # 1. terms and metrics
    terms = [
        ("Board, chip", "The ReSkin 5X board under each fingertip has five MLX90393 magnetometers, chips 0 to 4 in "
         "streaming order (0x0C, 0x11, 0x12, 0x13, 0x10); chip 0 is U1 at the centre. Each chip measures its die temperature "
         "and the field along x, y and z, so a board has 15 field components."),
        ("Record", "One sweep of the 15 chips of the three boards, 242 bytes on the USB serial port. The splitter "
         "<code>reskin5x_mux.py</code> stamps each record with the host time and logs it (248 bytes)."),
        ("Noise", r"The standard deviation of the difference of consecutive new conversions of one axis, divided by "
         r"$\sqrt{2}$ (&#181;T); drift does not enter it. In the log&#8217;s first 32&#8202;s, from 19:33:30, it was "
         "1.87&#8202;&#181;T on x and y and 2.88&#8202;&#181;T on z."),
        ("Correlation between boards", r"The largest absolute correlation of the record-to-record differences between a "
         r"component of one board and a component of another. For $n$ records of independent noise it stays under the "
         r"bound $4/\sqrt{n}$."),
        ("Malformed record", "A record that the viewer drops: a chip whose field is zero, a value that is not finite or "
         "exceeds 10<sup>5</sup>, or a die temperature outside &#8722;40 to 125&#8202;&#176;C."),
        ("Thermal correction", r"Each chip&#8217;s reading $b$ is divided by its gain at its die temperature $T$ (&#176;C), "
         r"$b_{\mathrm{c}} = b\,/\,[1 + \kappa\,(T - 25)]$, with $T$ averaged over the last 450 records (2.8&#8202;s). "
         r"The coefficients $\kappa$ are &#8722;0.65 to &#8722;0.70&#8202;%/&#176;C; they were fitted on the desk board on "
         "2026-10-21."),
        ("Rest change", "The median of a 60&#8202;s bin minus the median of the reference window 19:35&#8211;19:40, as the "
         "norm over a board&#8217;s 15 components (&#181;T), uncorrected or corrected."),
        ("Uniform field, part left", "The uniform field is the least-squares fit of one field vector read by every chip "
         "through its rotation on the board; the part left is the rest of the change. A board turned in Earth&#8217;s field "
         "changes the first; a magnet that moves in the pad changes mostly the second."),
        ("Gain coefficient", r"The change of a chip&#8217;s reading per degree of its die temperature as a share of its "
         r"reference field: the slope $\alpha$ of $\Delta b = \alpha\, b_{\mathrm{ref}}\,(T - T_{\mathrm{ref}})$ "
         "(%/&#176;C). It is fitted to the night&#8217;s 60&#8202;s bins after each bin&#8217;s uniform field is removed. "
         "With the correction on, it is the coefficient that the correction leaves; the correction is taken as valid when "
         "it is under 0.1&#8202;%/&#176;C. On the swings, the same fit is made to the bins&#8217; deviations from their "
         "60-minute running mean."),
        ("Viewer soak", "The frames of the contact viewer replayed over the log at 10 frames a second. Each frame "
         "averages a 50&#8202;ms window, and a new baseline is taken after 3&#8202;s at rest. A contact is shown when its "
         "explained chi-square exceeds 25 and its force 0.25&#8202;N. The noise floor is added to the chips&#8217; noise in the estimator: 2&#8202;&#181;T by "
         "default, 1&#8202;&#181;T in the owner&#8217;s viewer since 2026-10-26 14:00. The force before the gates is the "
         "normal force fitted to every frame, shown or not."),
        ("Allan deviation", r"At averaging time $\tau$, $\sigma(\tau) = \sqrt{\langle(\bar b_{k+1} - \bar b_k)^2\rangle/2}$, "
         r"with $\bar b_k$ the means of adjacent stretches of length $\tau$ (&#181;T). White noise falls as "
         r"$\tau^{-1/2}$; drift makes it rise."),
        ("Shared step", "A change of a board&#8217;s uniform field over 2&#8202;s, in 1&#8202;s bins, above "
         f"{St['min_uT']:g}{u}, seen on all three boards within {St['within_s']:g}&#8202;s."),
        ("Bin at rest", "A 60&#8202;s bin whose spread, the norm over the 15 components of the records&#8217; standard "
         "deviations, is under 12&#8202;&#181;T. Handling a finger raises the spread above it."),
    ]
    S.append(("terms", "Terms and metrics", '<dl class="terms">' + "".join(f"<dt>{a}</dt><dd>{b}</dd>" for a, b in terms)
              + "</dl>"))
    # 2. setup
    setup = (
        "<p>ID&#8202;1, ID&#8202;2 and ID&#8202;3 are the three v1 fingertips. Each pad has five magnets over a ReSkin 5X "
        "board. In the viewer&#8217;s models their poles are + &#8722; &#8722; &#8722; + and their strengths are 0.57, 0.47 "
        "and 0.20 of an N52 magnet&#8217;s. The boards were on channels 0, 1 and 2 of an I&#178;C mux at 0x71 on a SparkFun "
        "Thing Plus ESP32-C6, which ran <code>firmware/reskin5x_mux_stream</code>. The owner plugged ID&#8202;3 in at about "
        "19:30 on 2026-10-26 and left the three fingers apart on the desk. From 19:33:14 the splitter logged every record "
        f"to <code>{LOG}</code>. The analysis covers 19:35 to {N['until']}, {Nt['span_s'] / 3600:.1f}&#8202;h in which "
        "nothing touched the fingers.</p>"
        "<p>For the replays, ID&#8202;1&#8217;s calibration of 10-26 13:42, ID&#8202;2&#8217;s rest calibration of the UR5 "
        "session cb2t and ID&#8202;3&#8217;s of 10-25 on the QT Py were used. ID&#8202;3&#8217;s was checked at 02:42 by starting the viewer "
        "on its channel. A saved model is kept when, with a field from outside the pad, it fits the present rest as well "
        "as a new calibration does. The 10-25 model fitted at 0.18&#8202;mT and a new calibration at 0.31&#8202;mT, so the "
        "10-25 model was kept.</p>"
        "<pre class=\"cmd\">" + P.esc(COMMANDS).replace("$", "&#36;") + "</pre>")     # no dollar for the TeX pass
    S.append(("setup", "Fingertips, mux and log", setup))
    # 3. readout
    rows = []
    for key, c in C.items():
        rows.append([f"{c['since']}&#8211;{c['until']}", f"{c['rate']:.1f}", str(c["gaps_over_0p1s"]), str(c["malformed"]),
                     str(sum(sum(b["zero_records"]) for b in c["boards"])), f"{c['noise_xy_uT']:.2f}", f"{c['noise_z_uT']:.2f}",
                     f"{c['corr_cross_max']:.4f}", f"{c['corr_bound']:.4f}", str(c["identical_readings"])])
    ok_noise = all(abs(c["noise_xy_uT"] / 1.87 - 1) <= 0.05 and abs(c["noise_z_uT"] / 2.88 - 1) <= 0.05 for c in C.values())
    ok_corr = all(c["corr_cross_max"] <= c["corr_bound"] for c in C.values())
    restarts = max(0, len(M["starts"]) - 1)
    ran = "without a restart" if restarts == 0 else f"with {restarts} restart{'s' if restarts > 1 else ''}"
    readout = (
        f"<p>The readout {'passed' if ok_noise and ok_corr else 'failed'} in the three 10-minute windows (Table&#8202;1). "
        f"The noise stayed {'within' if ok_noise else 'outside'} 5&#8202;% of the 1.87 and 2.88{u} of the log&#8217;s first "
        f"32&#8202;s. The largest correlation between boards stayed {'under' if ok_corr else 'over'} the bound for "
        "independent noise. No zero record appeared, and no chip reading repeated on another board.</p>"
        f"<p>Over the whole night {Nt['records']:,} records arrived at {Nt['rate']:.2f} records/s. The longest step "
        f"between records was {Nt['longest_step_s'] * 1e3:.0f}&#8202;ms. The splitter ran {ran} from {M['first']} to "
        f"{M['last']}. Its minute lines report {M['rate_min']:.1f} to {M['rate_max']:.1f} records/s.</p>"
        f"<p>The log has {Nt['malformed']} malformed records in {Nt['span_s'] / 3600:.1f}&#8202;h, each a single record. In "
        "each, the first 64 bytes are valid and the rest are not, though the record ends in CR&#8202;LF; the next record is "
        f"valid again. The viewer drops them, and every tool on this page leaves them out ({BAD_EXAMPLE} is the first).</p>"
        + P.table(["Window", "Records/s", "Gaps over 0.1&#8202;s", "Malformed", "Zero records", "Noise x, y (µT)",
                   "Noise z (µT)", "Largest correlation", "Bound", "Repeated readings"], rows,
                  num_cols=range(1, 10), n=1, caption="Readout of the three boards in 10-minute windows "
                  "(<code>reskin5x_mux_check.py</code>). Noise is the median over the chips."))
    S.append(("readout", "Readout through the night", readout))
    # 4. rest and temperature
    trows = []
    for name in NAMES:
        r0, r1 = R0["boards"][name], R1["boards"][name]
        for c in range(5):
            trows.append([name if c == 0 else "", str(c), f"{r0['die_C_min'][c]:.2f}&#8211;{r0['die_C_max'][c]:.2f}",
                          f"{r0['die_C_max'][c] - r0['die_C_min'][c]:.2f}", mn(r1['tempfit_pct_per_C'][c], "{:+.2f}"),
                          mn(r0['tempfit_pct_per_C'][c], "{:+.2f}"), mn(r1['tempfit_hp_pct_per_C'][c], "{:+.2f}"),
                          mn(r0['tempfit_hp_pct_per_C'][c], "{:+.2f}")])
    rngs = [R0["boards"][n]["die_C_max"][c] - R0["boards"][n]["die_C_min"][c] for n in NAMES for c in range(5)]
    held = [abs(R0["boards"][n]["tempfit_pct_per_C"][c]) < 0.1 for n in NAMES for c in range(5)]
    end0 = [R0["boards"][n]["rows"][-1] for n in NAMES]
    end1 = [R1["boards"][n]["rows"][-1] for n in NAMES]
    over = []
    for n in NAMES:
        cs = [c for c in range(5) if abs(R0["boards"][n]["tempfit_pct_per_C"][c]) >= 0.1]
        if cs:
            over.append(f"{nm[n]}&#8217;s chip{'s' if len(cs) > 1 else ''}&#8202;{lst(cs, '{}')}")
    worst = min(((n, c) for n in NAMES for c in range(5)), key=lambda nc: R0["boards"][nc[0]]["tempfit_pct_per_C"][nc[1]])
    hp_raw = [v for n in NAMES for v in R1["boards"][n]["tempfit_hp_pct_per_C"]]
    hp_cor = [v for n in NAMES for v in R0["boards"][n]["tempfit_hp_pct_per_C"]]
    sw = [v for n in NAMES for v in R0["boards"][n]["die_swing_C"]]
    id3 = end0[2]["chip_uT"]
    rest = (
        "<p>The thermal correction removed most of the night&#8217;s drift (Figure&#8202;1). From 19:35 to "
        f"{end0[0]['t'][:5]} the rest of ID&#8202;1, ID&#8202;2 and ID&#8202;3 changed by "
        f"{lst([r['change_uT'] for r in end1])}{u} uncorrected and by {lst([r['change_uT'] for r in end0])}{u} corrected. "
        "Without the uniform field, a component varied over the night&#8217;s bins by "
        f"{lst([R1['boards'][n]['tempfit_rms_uT'][0] for n in NAMES], '{:.2f}')}{u} rms uncorrected and by "
        f"{lst([R0['boards'][n]['tempfit_rms_uT'][0] for n in NAMES], '{:.2f}')}{u} corrected.</p>"
        f"<p>The dies cooled by {min(rngs):.1f} to {max(rngs):.1f}{dg}, which is enough to test the coefficient. The "
        f"coefficient that the correction leaves is under 0.1{pc} on {sum(held)} of the 15 chips (Table&#8202;2). It "
        f"exceeds that on {lst(over, '{}')}. The largest is {mn(R0['boards'][worst[0]]['tempfit_pct_per_C'][worst[1]])}{pc}, "
        f"on {nm[worst[0]]}&#8217;s chip&#8202;{worst[1]}. The readings of ID&#8202;3&#8217;s chips 0 and 4 changed by "
        f"{id3[0]:.0f} and {id3[4]:.0f}{u}, rising through the night as the dies cooled.</p>"
        f"<p>The swings about the 60-minute running mean have a standard deviation of {min(sw):.2f} to {max(sw):.2f}{dg}. "
        f"On them the uncorrected coefficient is {mn(min(hp_raw))} to {mn(max(hp_raw))}{pc}, with a median of "
        f"{mn(np.median(hp_raw))}; &#954; is &#8722;0.65 to &#8722;0.70{pc}. The correction therefore leaves "
        f"{mn(min(hp_cor), '{:+.2f}')} to {mn(max(hp_cor), '{:+.2f}')}{pc} on the swings.</p>"
        + P.fig(1, MEDIA / "rest.png", "The three boards at rest through the night, in 60&#8202;s bins from the reference "
                "window 19:35&#8211;19:40. Top: each chip&#8217;s die temperature. Middle: the change of the rest over the 15 "
                "components, uncorrected (grey) and with each chip&#8217;s thermal correction (black). Bottom: the corrected "
                "change split into a uniform field (blue) and the part it leaves (orange). Script: "
                "<code>reskin5x_mux_rest.py</code>.", width=1100)
        + P.table(["Board", "Chip", "Die temperature (&#176;C)", "Range (&#176;C)", "Uncorrected (%/&#176;C)",
                   "Left by the correction (%/&#176;C)", "Uncorrected, swings (%/&#176;C)",
                   "Left on the swings (%/&#176;C)"], trows, num_cols=range(1, 8), n=2,
                  caption="Each chip&#8217;s gain coefficient against its die temperature, fitted over the night after a "
                  "uniform field per 60&#8202;s bin is removed. The correction&#8217;s coefficients are &#8722;0.65 to "
                  "&#8722;0.70&#8202;%/&#176;C. The last two columns fit only the swings about the 60-minute running mean."))
    S.append(("rest", "Rest drift and die temperature", rest))
    # 5. viewer soak
    srows = []
    for fl, Sx in (("2", S2), ("1", S1)):
        for ch, name in zip(CH, NAMES):
            r = Sx["boards"][ch]
            srows.append([name, fl, f"{r['frames']:,}", str(r["shown"]), str(len(r["events"])), f"{r['d_median_uT']:.1f}",
                          f"{r['d_p99_uT']:.1f}", f"{r['d_max_uT']:.1f}", f"{r['f_n_p99_N']:.2f}", str(r["rezeros"]),
                          f"{r['rest_drift_uT']:.1f}"])
    shown2 = sum(S2["boards"][c]["shown"] for c in CH)
    shown1 = sum(S1["boards"][c]["shown"] for c in CH)
    ev = [(name, e) for ch, name in zip(CH, NAMES) for e in S1["boards"][ch]["events"]]
    fh = sum(S1["boards"][c]["hours"] for c in CH)
    every = [S2["boards"][c]["hours"] * 3600 / max(1, S2["boards"][c]["rezeros"]) for c in CH]
    f2 = [S2["boards"][c]["f_n_p99_N"] for c in CH]
    f1 = [S1["boards"][c]["f_n_p99_N"] for c in CH]
    if ev:
        n0, e0 = ev[0]
        ev_txt = (f"At 1{u} {shown1} frame{'s' if shown1 > 1 else ''} with a contact appeared, the first on {nm[n0]} at "
                  f"{e0['t']}: {e0['s']:.1f}&#8202;s with a change of {e0['d_max_uT']:.1f}{u} and a force of "
                  f"{e0['f_max_N']:.1f}&#8202;N. That is {len(ev)} false contact{'s' if len(ev) > 1 else ''} in "
                  f"{fh:.0f} finger-hours.")
    else:
        ev_txt = f"At 1{u} none was shown either."
    soak = (
        f"<p>At the 2{u} floor the viewer&#8217;s logic showed "
        f"{'no contact' if shown2 == 0 else f'{shown2} frames with a contact'} on the three fingertips in "
        f"{S2['boards']['ch0']['frames']:,} frames each (Table&#8202;3). The stack soak of 2026-10-25 showed none in 23,264 "
        f"frames a finger at the same floor. {ev_txt}</p>"
        f"<p>The force before the gates has a p99 of {min(f2):.1f} to {max(f2):.1f}&#8202;N at 2{u} and {min(f1):.1f} to "
        f"{max(f1):.1f}&#8202;N at 1{u}. {'Apart from the frame above, each' if ev else 'Each'} frame with a force over "
        "0.25&#8202;N had an explained chi-square under 25. The viewer set a new baseline every "
        f"{lst(every, '{:.0f}')}&#8202;s on ID&#8202;1, ID&#8202;2 and ID&#8202;3.</p>"
        + P.table(["Board", "Floor (µT)", "Frames", "Frames with a contact", "Events", "Change median (µT)",
                   "p99 (µT)", "Largest (µT)", "Force before the gates, p99 (N)", "New baselines",
                   "Rest drift (µT)"], srows,
                  num_cols=range(1, 11), n=3, caption="The viewer&#8217;s frames over the night "
                  "(<code>reskin5x_mux_soak.py</code>). The change is what the viewer reads against its baseline; the rest "
                  "drift is the change between the first and the last 1,500 records."))
    S.append(("soak", "Contacts shown by the viewer at rest", soak))
    # 6. allan

    def at_tau(tau, s, v):
        return float(np.exp(np.interp(np.log(v), np.log(tau), np.log(s))))
    arows, lows, lows_raw, ends, ends_raw, x50, z50 = [], [], [], [], [], [], []
    for name in NAMES:
        for lab, key in (("x, y", "median_xy_uT"), ("z", "median_z_uT")):
            tau, s = np.array(A["boards"][name]["tau_s"]), np.array(A["boards"][name][key])
            tr, sr = np.array(Ar["boards"][name]["tau_s"]), np.array(Ar["boards"][name][key])
            k, kr = int(np.argmin(s)), int(np.argmin(sr))
            arows.append([name if lab == "x, y" else "", lab, f"{s[0]:.2f}", f"{at_tau(tau, s, 0.05):.2f}",
                          f"{at_tau(tau, s, 3.0):.2f}", f"{s[k]:.2f}", f"{tau[k]:.1f}", f"{s[-1]:.2f}", f"{sr[kr]:.2f}",
                          f"{tr[kr]:.1f}", f"{sr[-1]:.2f}"])
            if lab == "x, y":
                lows.append((s[k], tau[k]))
                lows_raw.append((sr[kr], tr[kr]))
                ends.append(s[-1])
                ends_raw.append(sr[-1])
                x50.append(at_tau(tau, s, 0.05))
            else:
                z50.append(at_tau(tau, s, 0.05))
    c0z = [(A["boards"][n]["adev_min_uT"][2], A["boards"][n]["tau_min_s"][2]) for n in NAMES]
    allan = (
        f"<p>With the thermal correction, the x and y deviation falls as white noise to {min(x50):.2f}&#8211;"
        f"{max(x50):.2f}{u} at the viewer&#8217;s 50&#8202;ms window (Figure&#8202;2, Table&#8202;4). Its median stops falling at "
        + lst([f"{t:.1f}&#8202;s on {nm[n]} ({v:.2f}{u})" for n, (v, t) in zip(NAMES, lows)], "{}") + ". "
        "Without the correction it falls further, to "
        + lst([f"{v:.2f}{u} at {t:.0f}&#8202;s" for v, t in lows_raw], "{}") + ". "
        "The correction divides each reading by a gain computed from the die temperature, so the temperature&#8217;s "
        "fluctuation over seconds enters the corrected field, scaled by &#954; and by the field itself. Over hours the "
        f"correction removes the cooling: at 2&#8202;h the median is {min(ends):.2f} to {max(ends):.2f}{u} with it and "
        f"{min(ends_raw):.2f} to {max(ends_raw):.2f}{u} without it.</p>"
        f"<p>At 50&#8202;ms the z deviation is {min(z50):.2f} to {max(z50):.2f}{u}, at the 1{u} floor; x and y are under "
        "it. The z component of chip&#8202;0, the centre chip, stops falling at about "
        f"{np.median([t for _, t in c0z]):.0f}&#8202;s on every board, at {min(v for v, _ in c0z):.2f} to "
        f"{max(v for v, _ in c0z):.2f}{u}.</p>"
        + P.fig(2, MEDIA / "allan.png", "Allan deviation of the 15 thermally corrected components of each board at rest. Lines: "
                "median over the ten x and y components (blue) and the five z components (red); bands: their range. Grey "
                "dashed: the x and y median without the correction. Dashed verticals: the viewer&#8217;s 50&#8202;ms window and "
                "its 3&#8202;s re-zero hold. Dotted: the 1 and 2&#8202;µT noise floors. Script: "
                "<code>reskin5x_mux_allan.py</code>.", width=1100)
        + P.table(["Board", "Axes", "One record (µT)", "50&#8202;ms (µT)", "3&#8202;s (µT)", "Least (µT)",
                   "At (s)", "2&#8202;h (µT)", "Least, uncorrected (µT)", "At (s)", "2&#8202;h, uncorrected (µT)"],
                  arows, num_cols=range(2, 11), n=4,
                  caption="Allan deviation, median over the components, with the thermal correction and without it."))
    S.append(("allan", "Noise against averaging time", allan))
    # 7. shared steps
    sh, sh15 = St["shared"], St15["shared"]
    B, B15 = [St["boards"][n] for n in NAMES], [St15["boards"][n] for n in NAMES]
    big = [(n, b) for n, b in zip(NAMES, B) if b["steps"]]
    big_txt = ("No board had a step over " + f"{St['min_uT']:g}{u}." if not big else
               f"Steps over {St['min_uT']:g}{u} appeared only on " + lst([f"{nm[n]} ({b['steps']}, largest "
                                                                         f"{b['size_max_uT']:.1f}{u})" for n, b in big], "{}") + ".")
    steps = (
        f"<p>{'No change' if sh['count'] == 0 else str(sh['count']) + ' changes'} of the uniform field over "
        f"{St['min_uT']:g}{u} appeared on all three boards within {St['within_s']:g}&#8202;s in {hours:.1f}&#8202;h, and "
        f"{'none' if sh15['count'] == 0 else sh15['count']} over {St15['min_uT']:g}{u} (Figure&#8202;3). Over 2&#8202;s the "
        f"uniform field changed by a median of {lst([b['change_2s_median_uT'] for b in B], '{:.2f}')}{u} on ID&#8202;1, "
        f"ID&#8202;2 and ID&#8202;3. Its p99 was {lst([b['change_2s_p99_uT'] for b in B], '{:.2f}')}{u}. {big_txt} Over "
        f"{St15['min_uT']:g}{u} the boards had {lst([b['steps'] for b in B15], '{}')} steps"
        f"{', none of them on all three' if sh15['count'] == 0 else ''}. The part that the uniform field leaves changed by "
        f"a median of {min(b['left_change_2s_median_uT'] for b in B):.1f} to {max(b['left_change_2s_median_uT'] for b in B):.1f}"
        f"{u} over 2&#8202;s, with {sum(b['left_steps'] for b in B) or 'no'} step"
        f"{'' if sum(b['left_steps'] for b in B) == 1 else 's'} over {St['left_min_uT']:g}{u}.</p>"
        + P.fig(3, MEDIA / "steps.png", "Change of each board&#8217;s uniform field over 2&#8202;s, from 1&#8202;s bins. Dashed: the "
                f"{St['min_uT']:g}&#8202;µT threshold. Scripts: <code>reskin5x_mux_rest.py --bin-s 1</code>, "
                "<code>reskin5x_mux_steps.py</code>.", width=1100))
    S.append(("steps", "Field changes shared by the three boards", steps))
    # 8. recovery
    rrows = []
    for name in NAMES[:2]:
        for hr in Rc["boards"][name]["hourly"]:
            rrows.append([name, f"{hr['start']}", str(hr["bins"]), f"{hr['left_uT']:.1f}", f"{hr['uniform_uT']:.1f}",
                          f"{hr['die_C']:.2f}", "mux" if any(s.startswith("mux") for s in hr["sources"]) else "viewer"])
    pre, post = {}, {}
    for name in NAMES[:2]:
        hh = Rc["boards"][name]["hourly"]
        pre[name] = [h for h in hh if not any(s.startswith("mux") for s in h["sources"])]
        post[name] = [h for h in hh if all(s.startswith("mux") for s in h["sources"])]
    p_end = f"{int(pre['ID 1'][-1]['start'][:2]) + 1:02d}:00"
    rec = (
        "<p>The presses left a change in the pads&#8217; rest that had not fallen by 04:00, 14&#8202;h later (Figure&#8202;4, "
        "Table&#8202;5). Against the rest at 13:42, the part left was "
        + lst([f"{min(h['left_uT'] for h in pre[n]):.1f} to {max(h['left_uT'] for h in pre[n]):.1f}{u} on {nm[n]}"
               for n in NAMES[:2]], "{}")
        + f" from {pre['ID 1'][0]['start']} to {p_end}. In the same hours the uniform field fell from "
        + lst([f"{pre[n][0]['uniform_uT']:.1f} to {pre[n][-1]['uniform_uT']:.1f}{u} on {nm[n]}" for n in NAMES[:2]], "{}")
        + ". The uniform field also changes with the room&#8217;s field.</p>"
        "<p>ID&#8202;1&#8217;s part left rose by about 7&#8202;\u00b5T between 18:50 and 19:12, across a stretch of bins "
        "that are not at rest, before the boards were moved at about 19:30. ID&#8202;2&#8217;s uniform field fell in two "
        "steps, at about 16:25 and 18:30, each across such a stretch. Through the night the part left rose from "
        + lst([f"{post[n][0]['left_uT']:.1f} to {post[n][-1]['left_uT']:.1f}{u} on {nm[n]}" for n in NAMES[:2]], "{}")
        + ". Over the same hours the boards&#8217; mean die temperature fell by "
        + lst([post[n][0]["die_C"] - post[n][-1]["die_C"] for n in NAMES[:2]]) + f"{dg}.</p>"
        + P.fig(4, MEDIA / "recovery.png", "Rest of ID&#8202;1 and ID&#8202;2 after the presses, in 60&#8202;s bins at rest, "
                "thermally corrected. Orange: the part left; blue: the uniform field. Script: "
                "<code>reskin5x_mux_recovery.py</code>.", width=1100)
        + P.table(["Board", "Hour from", "Bins at rest", "Part left (µT)", "Uniform field (µT)",
                   "Die temperature (&#176;C)", "Log"], rrows,
                  num_cols=range(2, 6), n=5, caption="Hourly means over the bins at rest after the presses ended, against "
                  "the rest at 13:42:30&#8211;13:43:30. Until 19:32 the rows come from the viewer&#8217;s stream logs, then "
                  "from the night&#8217;s log."))
    S.append(("recovery", "Recovery of the pads after hard presses", rec))
    # 9. limitations
    S.append(("limits", "Limitations", (
        "<p>The night tests one desk, one room and a cooling of about 1&#8202;&#176;C. The thermal coefficients were fitted "
        "on the desk board on 2026-10-21. A slow creep of a pad that runs alongside the cooling enters the gain fit as a "
        "temperature coefficient. The drift of ID&#8202;3&#8217;s chips 0 and 4 cannot be split from such a creep. The swings "
        "about the running mean are a few hundredths of a degree. The viewer soak replays the viewer&#8217;s logic at 10 "
        "frames a second, without its recalibration on a large move of the rest. The pads&#8217; recovery is measured "
        "against a rest 7&#8202;min before the presses, and the fingers were handled between the presses and 19:30.</p>")))
    # 10. future work
    fw = [
        ("Each board&#8217;s own thermal coefficients.", "Warm each finger by 3&#8202;&#176;C, with a desk lamp or the "
         "warm-up after plugging in. Fit &#954; per chip with <code>reskin5x_mux_rest.py --tempfit --no-thermal</code>, on "
         "the first 10&#8202;min and on the last hour. A coefficient more than 0.1&#8202;%/&#176;C from the desk "
         "board&#8217;s falsifies a shared correction. A fast coefficient near half the slow one, as on this night&#8217;s "
         "swings, calls for a lag in the correction."),
        ("A longer average of the die temperature.", "Average T over 60&#8202;s in place of 450 records in "
         "<code>reskin5x_mux_rest.smoothed_T</code> and in the viewer&#8217;s <code>fields</code>, and rerun "
         "<code>reskin5x_mux_allan.py</code> on this log. The x and y minimum on ID&#8202;1 and ID&#8202;2 should fall to the "
         "uncorrected 0.10&#8211;0.11&#8202;µT. A minimum still above 0.15&#8202;µT falsifies the temperature&#8217;s "
         "fluctuation as the source of the added noise."),
        ("A floor set from the deviation.", "Run <code>reskin5x_mux_soak.py --floor</code> at 0.5, 1 and 2&#8202;µT on "
         "this log and on a log with presses. The viewer&#8217;s default becomes the lowest floor at which no contact is "
         "shown at rest and every press is shown."),
        ("Recovery after controlled presses.", "Press ID&#8202;1 and ID&#8202;2 on the UR5 at 10&#8211;15&#8202;N with 10&#8202;min at "
         "rest after each press, logged through <code>reskin5x_mux.py</code>, and rerun "
         "<code>reskin5x_mux_recovery.py</code>. A part left that returns under 2&#8202;µT within an hour falsifies a "
         "lasting change from presses."),
    ]
    S.append(("future", "Future work", "<ul class=\"open\">" + "".join(f"<li><b>{a}</b> {b}</li>" for a, b in fw) + "</ul>"))
    nx = np.median([c["noise_xy_uT"] for c in C.values()])
    nz = np.median([c["noise_z_uT"] for c in C.values()])
    lasting = [np.mean([h["left_uT"] for h in pre[n]]) for n in NAMES[:2]]
    if ev:
        one = (f" and {shown1} frame{'s' if shown1 > 1 else ''} at 1{u} ({ev[0][1]['s']:.1f}&#8202;s, "
               f"{ev[0][1]['f_max_N']:.1f}&#8202;N)")
    else:
        one = f" and none at 1{u}"
    shown_2 = "no contact" if shown2 == 0 else f"{shown2} frames with a contact"
    lede = (
        f"Three v1 fingertips lay untouched on the desk for {Nt['span_s'] / 3600:.1f}&#8202;h, from 19:35 to {N['until']}, "
        f"read through one I&#178;C mux at {M['rate_median']:.1f} records/s without a gap. The noise stayed at {nx:.2f}{u} "
        f"on x and y and {nz:.2f}{u} on z, uncorrelated between boards. Replayed through the contact viewer, the "
        f"fingertips showed {shown_2} at the 2{u} noise floor{one}. The thermal correction cut the night&#8217;s drift from "
        f"{min(r['change_uT'] for r in end1):.0f}&#8211;{max(r['change_uT'] for r in end1):.0f}{u} to "
        f"{min(r['change_uT'] for r in end0):.0f}&#8211;{max(r['change_uT'] for r in end0):.0f}{u}. The coefficient it "
        f"leaves was under 0.1{pc} on {sum(held)} of 15 chips. The presses of 2026-10-26 left {lasting[0]:.0f} and "
        f"{lasting[1]:.0f}{u} on ID&#8202;1 and ID&#8202;2, which had not fallen 14&#8202;h later.")
    summary = (f"Three v1 fingertips at rest for {Nt['span_s'] / 3600:.1f} h on the ESP32-C6 mux: {M['rate_median']:.1f} "
               f"records/s without a gap, noise {nx:.2f}/{nz:.2f} uT, {shown_2} at the 2 uT floor{one}. The thermal "
               f"correction cut the drift to {min(r['change_uT'] for r in end0):.0f}-{max(r['change_uT'] for r in end0):.0f} "
               f"uT; the presses of 10-26 left {lasting[0]:.0f} and {lasting[1]:.0f} uT on ID 1 and ID 2.")
    summary = re.sub(r"<[^>]+>", "", summary).replace("&#8202;", " ").replace("&#8217;", "'").replace("µ", "u")
    return S, lede, summary


def build():
    MEDIA.mkdir(parents=True, exist_ok=True)
    N = dict(checks={k: j(f"check_{k}.json") for k in ("1935", "2300", "0350")}, mux=mux_log_summary(), rest=j("rest.json"),
             rest_raw=j("rest_raw.json"), soak2=j("soak_floor2.json"), soak1=j("soak_floor1.json"), allan=j("allan.json"),
             allan_raw=j("allan_raw.json"),
             steps=j("steps.json"), steps15=j("steps_1p5.json"), recovery=j("recovery.json"), night=j("check_night.json"))
    N["until"] = time.strftime("%H:%M", time.gmtime(sum(int(v) * w for v, w in zip(N["night"]["until"].split(":"),
                                                                                   (3600, 60, 1))) + 1))
    fig_rest(MEDIA / "rest.png")
    fig_allan(MEDIA / "allan.png", N["allan"], N["allan_raw"])
    fig_steps(MEDIA / "steps.png", N["steps"])
    fig_recovery(MEDIA / "recovery.png", N["recovery"])
    S, lede, summary = sections(N)
    toc = "\n".join(f'<li><a href="#{sid}">{k}. {hd}</a></li>' for k, (sid, hd, _) in enumerate(S, 1))
    secs = "\n".join(f'<section id="{sid}"><h2>{k}. {hd}</h2>{body}</section>' for k, (sid, hd, body) in enumerate(S, 1))
    byline = ("ID&#8202;1, ID&#8202;2 and ID&#8202;3 on a SparkFun Thing Plus ESP32-C6 with an I&#178;C mux, at rest on the desk, "
              "night of 2026-10-26/27. Tools <code>src/morphohand/driver/reskin5x/host/reskin5x_mux_*.py</code>; page by "
              "<code>scripts/three_finger_soak_page.py</code>; rows in " + link(DATA, "data/") + ". The estimator is described "
              "on " + link(L["estimation"], "the contact estimation page") + "; the fingertips on "
              + link(L["overview"], "the printed fingertips overview") + ".")
    page = TEMPLATE.read_text()
    for k, v in {"TITLE": "Three v1 fingertips at rest overnight on the ESP32-C6 mux",
                 "H1": "Three v1 fingertips at rest overnight on the ESP32-C6 mux",
                 "EYEBROW": "Printed fingertips &#183; 2026-10-27", "LEDE": lede, "BYLINE": byline, "FILM": "",
                 "TOC": toc, "SECTIONS": secs, "FOOTER": str((D / PAGE.name).relative_to(ROOT)), "TEXCSS": texsvg.CSS}.items():
        page = page.replace("{{" + k + "}}", v)
    items = [((m.group(1) if m.group(1) is not None else m.group(2)).strip(), m.group(1) is not None)
             for m in P.TEX_RE.finditer(page)]
    svgs = iter(texsvg.render(items, cache_path=TEX_CACHE))
    page = P.TEX_RE.sub(lambda m: next(svgs), page)
    page = retro_style.apply(page)
    page = docs_index.tag(page, topic="fingertip", summary=summary, date="2026-10-27")
    PAGE.write_text(page)
    print(f"{PAGE}: {PAGE.stat().st_size / 1e6:.2f} MB, {len(items)} formulas", flush=True)


if __name__ == "__main__":
    build()
