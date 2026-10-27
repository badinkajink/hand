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


def clock_ticks(ax, t0, h_max):
    """Clock times (HH:MM) on an axis of hours since t0."""
    ticks = np.arange(0, h_max + 0.01, 1.0)
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
        clock_ticks(ax[2, b], t0, h[-1])
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


def fig_allan(out, A):
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
        ax[b].set_yscale("symlog", linthresh=10)
    for b in range(2):
        p0, p1 = (hours_since(clock_on(R["presses"][k], t0), t0) for k in range(2))
        ax[b].axvspan(p0, p1, color="#cccccc", alpha=0.6, lw=0)
        mv = hours_since(clock_on("19:32", t0), t0)
        ax[b].axvline(mv, color="#000000", lw=0.8, ls="--")
    ax[0].legend(frameon=False, loc="upper right")
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
    C, M, R0, R1, S2, S1, A, St, St15, Rc, Nt = (N[k] for k in ("checks", "mux", "rest", "rest_raw", "soak2", "soak1",
                                                               "allan", "steps", "steps15", "recovery", "night"))
    S = []
    hours = St["hours"]
    # 1. terms and metrics
    terms = [
        ("Board, chip", "The ReSkin 5X board under each fingertip has five MLX90393 magnetometers, chips 0 to 4 in "
         "streaming order (0x0C, 0x11, 0x12, 0x13, 0x10); chip 0 is U1 at the centre. Each chip gives its die temperature "
         "and the field along x, y and z, so a board gives 15 field components."),
        ("Record", "One sweep of the 15 chips of the three boards, 242 bytes on the USB serial port. The splitter "
         "<code>reskin5x_mux.py</code> stamps each record with the host time and logs it (248 bytes)."),
        ("Noise", "The standard deviation of the difference of consecutive new conversions of one axis, over "
         "\u221a2 (\u00b5T). Drift does not enter it."),
        ("Separation", "The absolute correlation of the record-to-record differences between components on different "
         "boards. Independent noise stays under 4/\u221an for n records."),
        ("Malformed record", "A record that the viewer drops: a chip whose field reads zero, a value that is not finite or "
         "exceeds 10<sup>5</sup>, or a die temperature outside &#8722;40 to 125&#8202;&#176;C."),
        ("Rest change", "The median of a 60&#8202;s bin minus the median of the reference window 19:35&#8211;19:40 (\u00b5T, "
         "the norm over 15 components). With the thermal correction, each reading is first divided by its chip&#8217;s gain "
         "1&#8202;+&#8202;\u03ba(T&#8202;&#8722;&#8202;25&#8202;&#176;C), \u03ba&#8202;=&#8202;&#8722;0.65 to &#8722;0.70&#8202;%/&#176;C."),
        ("Uniform field, part left", "The least-squares fit of one field vector read by every chip through its rotation; "
         "the part left is the rest of the change. A board turned in Earth&#8217;s field changes the first; a magnet that "
         "moves in the pad changes the second."),
        ("Gain coefficient", "The change of a chip&#8217;s reading per &#176;C of its die temperature, as a share of its "
         "reference field (%/&#176;C). It is fitted to all bins of the night after a uniform field per bin is removed. "
         "With the correction on, it is what the correction leaves."),
        ("Viewer soak", "The frames of the contact viewer replayed over the log: a 50&#8202;ms window, a new baseline after "
         "3&#8202;s at rest, and a contact shown above an explained chi-square of 25 and 0.25&#8202;N. The noise floor is "
         "added to the chips&#8217; noise in the estimator (2&#8202;\u00b5T by default, 1&#8202;\u00b5T in the owner&#8217;s viewer "
         "since 2026-10-26 14:00)."),
        ("Allan deviation", "At averaging time \u03c4, the root of half the mean square difference between the means of "
         "adjacent stretches of length \u03c4 (\u00b5T). White noise falls as \u03c4<sup>&#8722;1/2</sup>; drift makes it rise."),
        ("Shared step", "A change of a board&#8217;s uniform field over 2&#8202;s, in 1&#8202;s bins, above "
         f"{St['min_uT']:g}&#8202;\u00b5T, seen on all three boards within {St['within_s']:g}&#8202;s."),
    ]
    S.append(("terms", "Terms and metrics", '<dl class="terms">' + "".join(f"<dt>{a}</dt><dd>{b}</dd>" for a, b in terms)
              + "</dl>"))
    # 2. setup
    setup = (
        "<p>ID&#8202;1, ID&#8202;2 and ID&#8202;3 are the three v1 fingertips. Each has five N52 magnets in its pad over a "
        "ReSkin 5X board. Their boards were on channels 0, 1 and 2 of an I&#178;C mux at 0x71 on a SparkFun Thing Plus "
        "ESP32-C6, running <code>firmware/reskin5x_mux_stream</code>. The owner plugged ID&#8202;3 in at about 19:30 on 2026-10-26 "
        "and left the three fingers apart on the desk. From 19:33:14 the splitter logged every record to "
        "<code>logs/20261026-three_finger_mux/20261026-193314-mux.bin</code>. Nothing touched the fingers until the "
        f"analysis at {N['until']}.</p>"
        "<p>The viewer&#8217;s calibrations used for the replays are ID&#8202;1&#8217;s of 10-26 13:42, ID&#8202;2&#8217;s rest "
        "calibration of the UR5 session cb2t, and ID&#8202;3&#8217;s of 10-25 on the QT Py. At 02:42 the viewer was started on "
        "ID&#8202;3&#8217;s channel from that calibration. Its rule keeps a saved model when, with a field from outside the pad, "
        "the model fits the present rest as well as a new calibration. The 10-25 model fitted at 0.18&#8202;mT and a new "
        "calibration at 0.31&#8202;mT, so it was kept: poles + &#8722; &#8722; &#8722; +, strength 0.20 of N52.</p>"
        "<pre class=\"cmd\">" + P.esc(COMMANDS).replace("$", "&#36;") + "</pre>")     # no dollar for the TeX pass
    S.append(("setup", "Fingertips, mux and log", setup))
    # 3. readout
    rows = []
    for key, c in C.items():
        rows.append([f"{c['since']}&#8211;{c['until']}", f"{c['rate']:.1f}", str(c["gaps_over_0p1s"]), str(c["malformed"]),
                     str(sum(sum(b["zero_records"]) for b in c["boards"])), f"{c['noise_xy_uT']:.2f}", f"{c['noise_z_uT']:.2f}",
                     f"{c['corr_cross_max']:.3f}", f"{c['corr_bound']:.3f}", str(c["identical_readings"])])
    ok_noise = all(abs(c["noise_xy_uT"] / 1.87 - 1) <= 0.05 and abs(c["noise_z_uT"] / 2.88 - 1) <= 0.05 for c in C.values())
    ok_corr = all(c["corr_cross_max"] <= c["corr_bound"] for c in C.values())
    readout = (
        f"<p>The splitter ran without a restart from {M['first']} to {M['last']}. Its minute lines gave "
        f"{M['rate_min']:.1f} to {M['rate_max']:.1f} records/s. Table&#8202;1 gives three 10-minute windows. "
        f"The noise stayed {'within' if ok_noise else 'outside'} 5&#8202;% of the evening&#8217;s 1.87 and 2.88&#8202;\u00b5T, "
        f"and the largest correlation between boards stayed {'under' if ok_corr else 'over'} the bound for independent noise "
        "in every window. No record repeated another board&#8217;s chip.</p>"
        f"<p>The log has {Nt['malformed']} malformed records in {Nt['span_s'] / 3600:.1f}&#8202;h, each a single record. In each, "
        "the first 64 bytes are valid and the rest are not, though the record ends in CR&#8202;LF; the next record is "
        f"valid again. The viewer drops them, and every tool on this page leaves them out ({BAD_EXAMPLE} is the first).</p>"
        + P.table(["Window", "Records/s", "Gaps over 0.1&#8202;s", "Malformed", "Zero records", "Noise x, y (\u00b5T)",
                   "Noise z (\u00b5T)", "Largest correlation", "Bound", "Repeated readings"], rows,
                  num_cols=range(1, 10), n=1, caption="Readout of the three boards in 10-minute windows "
                  "(<code>reskin5x_mux_check.py</code>). Noise is the median over the chips."))
    S.append(("readout", "Readout through the night", readout))
    # 4. rest and temperature
    trows = []
    for name in NAMES:
        r0, r1 = R0["boards"][name], R1["boards"][name]
        for c in range(5):
            trows.append([name if c == 0 else "", str(c), f"{r0['die_C_min'][c]:.2f}&#8211;{r0['die_C_max'][c]:.2f}",
                          f"{r0['die_C_max'][c] - r0['die_C_min'][c]:.2f}", f"{r1['tempfit_pct_per_C'][c]:+.2f}",
                          f"{r0['tempfit_pct_per_C'][c]:+.2f}", f"{r0['tempfit_hp_pct_per_C'][c]:+.2f}"])
    rngs = [R0["boards"][n]["die_C_max"][c] - R0["boards"][n]["die_C_min"][c] for n in NAMES for c in range(5)]
    held = [abs(R0["boards"][n]["tempfit_pct_per_C"][c]) < 0.1 for n in NAMES for c in range(5)]
    rest = (
        f"<p>The dies cooled by {min(rngs):.1f} to {max(rngs):.1f}&#8202;&#176;C over the night (Figure&#8202;1, top). "
        "Uncorrected, each board&#8217;s rest followed the temperature. With the chips&#8217; correction, "
        f"{sum(held)} of 15 chips have a coefficient left under 0.1&#8202;%/&#176;C (Table&#8202;2).</p>"
        + P.fig(1, MEDIA / "rest.png", "The three boards at rest through the night, in 60&#8202;s bins from the reference "
                "window 19:35&#8211;19:40. Top: each chip&#8217;s die temperature. Middle: the change of the rest over the 15 "
                "components, uncorrected (grey) and with each chip&#8217;s thermal correction (black). Bottom: the corrected "
                "change split into a uniform field (blue) and the part it leaves (orange). Script: "
                "<code>reskin5x_mux_rest.py</code>.", width=1100)
        + P.table(["Board", "Chip", "Die temperature (&#176;C)", "Range (&#176;C)", "Uncorrected (%/&#176;C)",
                   "Left by the correction (%/&#176;C)", "Left on the swings (%/&#176;C)"], trows, num_cols=range(1, 7), n=2,
                  caption="Each chip&#8217;s gain coefficient against its die temperature, fitted over the night after a "
                  "uniform field per 60&#8202;s bin is removed. The correction&#8217;s coefficients are &#8722;0.65 to "
                  "&#8722;0.70&#8202;%/&#176;C. The last column fits only the swings about the 60-minute running mean."))
    S.append(("rest", "Rest drift and die temperature", rest))
    # 5. viewer soak
    srows = []
    for fl, Sx in (("2", S2), ("1", S1)):
        for ch, name in zip(CH, NAMES):
            r = Sx["boards"][ch]
            srows.append([name, fl, f"{r['frames']:,}", str(r["shown"]), str(len(r["events"])), f"{r['d_median_uT']:.1f}",
                          f"{r['d_p99_uT']:.1f}", f"{r['d_max_uT']:.1f}", str(r["rezeros"]), f"{r['rest_drift_uT']:.1f}"])
    shown2 = sum(S2["boards"][c]["shown"] for c in CH)
    shown1 = sum(S1["boards"][c]["shown"] for c in CH)
    soak = (
        f"<p>Replayed through the viewer&#8217;s logic at 10 frames a second from 19:35 to {S2['until'][:5]}, the three "
        f"fingertips showed {shown2} frames with a contact at the 2&#8202;\u00b5T floor and {shown1} at 1&#8202;\u00b5T "
        "(Table&#8202;3). The stack soak of 2026-10-25 showed none in 23,264 frames a finger at 2&#8202;\u00b5T.</p>"
        + P.table(["Board", "Floor (\u00b5T)", "Frames", "Frames with a contact", "Events", "Change median (\u00b5T)",
                   "p99 (\u00b5T)", "Largest (\u00b5T)", "New baselines", "Rest drift (\u00b5T)"], srows,
                  num_cols=range(1, 10), n=3, caption="The viewer&#8217;s frames over the night "
                  "(<code>reskin5x_mux_soak.py</code>). The change is what the viewer reads against its baseline; the rest "
                  "drift is the change between the first and the last 1,500 records."))
    S.append(("soak", "The viewer&#8217;s contacts at rest", soak))
    # 6. allan
    arows = []
    for name in NAMES:
        r = A["boards"][name]
        tau = np.array(r["tau_s"])
        for lab, key in (("x, y", "median_xy_uT"), ("z", "median_z_uT")):
            s = np.array(r[key])
            k = int(np.argmin(s))
            arows.append([name, lab, f"{s[0]:.2f}", f"{np.exp(np.interp(np.log(0.05), np.log(tau), np.log(s))):.2f}",
                          f"{np.exp(np.interp(np.log(3.0), np.log(tau), np.log(s))):.2f}", f"{s[k]:.2f}", f"{tau[k]:.1f}",
                          f"{s[-1]:.2f}"])
    allan = (
        "<p>Averaging lowers the noise as white noise would up to about a second (Figure&#8202;2). The viewer&#8217;s 50&#8202;ms "
        "window leaves 0.6&#8211;1.0&#8202;\u00b5T a component, under the 1 and 2&#8202;\u00b5T floors. Past the turning point the "
        "deviation stays near 0.1&#8211;0.3&#8202;\u00b5T for hours (Table&#8202;4).</p>"
        + P.fig(2, MEDIA / "allan.png", "Allan deviation of the 15 thermally corrected components of each board at rest. Lines: "
                "median over the ten x and y components (blue) and the five z components (red); bands: their range. Dashed: the "
                "viewer&#8217;s 50&#8202;ms window and its 3&#8202;s re-zero hold. Dotted: the 1 and 2&#8202;\u00b5T noise floors. Script: "
                "<code>reskin5x_mux_allan.py</code>.", width=1100)
        + P.table(["Board", "Axes", "One record (\u00b5T)", "50&#8202;ms (\u00b5T)", "3&#8202;s (\u00b5T)", "Least (\u00b5T)",
                   "At (s)", "2&#8202;h (\u00b5T)"], arows, num_cols=range(2, 8), n=4,
                  caption="Allan deviation, median over the components, at the viewer&#8217;s averaging times."))
    S.append(("allan", "Noise against averaging time", allan))
    # 7. shared steps
    sh, sh15 = St["shared"], St15["shared"]
    steps = (
        f"<p>Over {hours:.1f}&#8202;h, {sh['count']} changes of the uniform field over {St['min_uT']:g}&#8202;\u00b5T appeared on all "
        f"three boards within {St['within_s']:g}&#8202;s, and {sh15['count']} over {St15['min_uT']:g}&#8202;\u00b5T. Per board, the "
        "uniform field changed by a median of "
        + ", ".join(f"{St['boards'][n]['change_2s_median_uT']:.2f}" for n in NAMES)
        + "&#8202;\u00b5T over 2&#8202;s (p99 "
        + ", ".join(f"{St['boards'][n]['change_2s_p99_uT']:.2f}" for n in NAMES)
        + f"&#8202;\u00b5T), with {', '.join(str(St['boards'][n]['steps']) for n in NAMES)} steps over {St['min_uT']:g}&#8202;\u00b5T "
        "on ID&#8202;1, ID&#8202;2 and ID&#8202;3 (Figure&#8202;3).</p>"
        + P.fig(3, MEDIA / "steps.png", "Change of each board&#8217;s uniform field over 2&#8202;s, from 1&#8202;s bins. Dashed: the "
                f"{St['min_uT']:g}&#8202;\u00b5T threshold. Scripts: <code>reskin5x_mux_rest.py --bin-s 1</code>, "
                "<code>reskin5x_mux_steps.py</code>.", width=1100))
    S.append(("steps", "Field changes shared by the three boards", steps))
    # 8. recovery
    rrows = []
    for name in NAMES[:2]:
        for hr in Rc["boards"][name]["hourly"]:
            rrows.append([name, f"{hr['start']}", str(hr["bins"]), f"{hr['left_uT']:.1f}", f"{hr['uniform_uT']:.1f}",
                          "mux" if any(s.startswith("mux") for s in hr["sources"]) else "viewer"])
    rec = (
        "<p>The owner pressed ID&#8202;1 and ID&#8202;2 hard between 13:49 and 14:00 on 2026-10-26. Figure&#8202;4 and "
        "Table&#8202;5 give their rest against the rest at 13:42:30&#8211;13:43:30, through the viewer&#8217;s stream logs and "
        "the night&#8217;s log. At about 19:30 the boards were moved, so across that time only the part left compares.</p>"
        + P.fig(4, MEDIA / "recovery.png", "Rest of ID&#8202;1 and ID&#8202;2 after the presses, in 60&#8202;s bins at rest "
                "(spread under 12&#8202;\u00b5T), thermally corrected. Orange: the part left; blue: the uniform field. Script: "
                "<code>reskin5x_mux_recovery.py</code>.", width=1100)
        + P.table(["Board", "Hour from", "Bins at rest", "Part left (\u00b5T)", "Uniform field (\u00b5T)", "Log"], rrows,
                  num_cols=range(2, 5), n=5, caption="Hourly means over the bins at rest after the presses ended."))
    S.append(("recovery", "Recovery of the pads after hard presses", rec))
    # 9. limitations
    S.append(("limits", "Limitations", (
        "<p>The night tests one desk, one room and one cooling of about a degree. The thermal coefficients were fitted on "
        "the desk board on 2026-10-21. The gain fit cannot tell a slow creep of a pad that runs alongside the cooling from "
        "a temperature coefficient; the fit on the swings about the running mean separates them only for swings of "
        "a few hundredths of a degree. The viewer soak replays the viewer&#8217;s logic at 10 frames a second, without its "
        "recalibration on a large rest move. The pads&#8217; recovery is measured against a rest 7&#8202;min before the "
        "presses, and the fingers were handled between the presses and 19:30.</p>")))
    # 10. future work
    fw = [
        ("Each board&#8217;s own thermal coefficients.", "Warm each finger by 3&#8202;&#176;C (a desk lamp or the board&#8217;s "
         "plug-in warm-up) and fit \u03ba per chip with <code>reskin5x_mux_rest.py --tempfit --no-thermal</code>. A coefficient "
         "more than 0.1&#8202;%/&#176;C from the desk board&#8217;s falsifies a shared correction."),
        ("A floor set from the deviation.", "Run <code>reskin5x_mux_soak.py --floor</code> at 0.5, 1 and 2&#8202;\u00b5T on this "
         "log and on a log with presses; the lowest floor that shows no contact at rest and keeps every press sets the "
         "viewer&#8217;s default."),
    ]
    S.append(("future", "Future work", "<ul class=\"open\">" + "".join(f"<li><b>{a}</b> {b}</li>" for a, b in fw) + "</ul>"))
    lede = (f"Three v1 fingertips lay untouched on the desk from 19:35 to {N['until']} on the ESP32-C6 mux. The readout ran at "
            f"{M['rate_median']:.1f} records/s without a gap, with noise of "
            f"{np.median([c['noise_xy_uT'] for c in C.values()]):.2f} and {np.median([c['noise_z_uT'] for c in C.values()]):.2f}"
            f"&#8202;\u00b5T and no correlation between boards. The viewer&#8217;s logic showed {shown2} frames with a contact at "
            f"the 2&#8202;\u00b5T floor and {shown1} at 1&#8202;\u00b5T. No field change over {St['min_uT']:g}&#8202;\u00b5T appeared on all "
            "three boards." if sh["count"] == 0 else "")
    summary = re.sub(r"<[^>]+>", "", lede).replace("&#8202;", " ").replace("&#8217;", "'")
    return S, lede, summary


def build():
    MEDIA.mkdir(parents=True, exist_ok=True)
    N = dict(checks={k: j(f"check_{k}.json") for k in ("1935", "2300", "0350")}, mux=mux_log_summary(), rest=j("rest.json"),
             rest_raw=j("rest_raw.json"), soak2=j("soak_floor2.json"), soak1=j("soak_floor1.json"), allan=j("allan.json"),
             steps=j("steps.json"), steps15=j("steps_1p5.json"), recovery=j("recovery.json"), night=j("check_night.json"))
    N["until"] = N["night"]["until"][:5]
    fig_rest(MEDIA / "rest.png")
    fig_allan(MEDIA / "allan.png", N["allan"])
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
