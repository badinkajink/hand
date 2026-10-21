"""Live plot of the ReSkin 5X stream (firmware/reskin5x_stream): the change of each chip's field from a baseline.

The top panel shows |dB| of every chip; the five panels under it show dBx, dBy and dBz of one chip each (uT, chip frame,
streaming order), over the last --window seconds. The baseline is the mean of the first 0.5 s of records; key b takes it
again from the last 0.5 s, so press b with the pad at rest before touching it. The y axes follow the data, with at least
+-10 uT so that noise (about 2 uT) stays small. A chip whose reading is zeros (no conversion yet, or a failed transfer)
is left out of that record.

    ~/miniconda3/bin/python src/morphohand/driver/reskin5x/host/reskin5x_plot.py [--port P] [--window 10] [--out f.csv]
    ... --png f.png --seconds 5      render once after 5 s and exit, without a window
"""
import argparse
import glob
import os
import threading
import time
from collections import deque

import numpy as np
import serial

N_CHIPS = 5
FRAME = 4 * 4 * N_CHIPS + 2
ADDRS = ("0x0C", "0x11", "0x12", "0x13", "0x10")   # streaming order of the workstation's board
BASE_S = 0.5
PORT_GLOBS = ("/dev/serial/by-id/*QT_Py*", "/dev/ttyACM*")


def find_port():
    """The board's serial port: its /dev/serial/by-id name, which stays the same when the board is plugged in again,
    else the first /dev/ttyACM*; None when neither exists."""
    for g in PORT_GLOBS:
        hits = sorted(glob.glob(g))
        if hits:
            return hits[0]
    return None


class Reader(threading.Thread):
    """Reads the port into records (host time, chips x [t x y z]) and keeps the last `keep` seconds of them. When the
    board is unplugged, the reader closes the port, sets `connected` False and opens the board again once it is back
    (on `port` while that exists, else on find_port()); `reconnects` counts the reopenings."""

    def __init__(self, port, keep):
        super().__init__(daemon=True)
        self.port_arg = port
        self.port = port or find_port()
        if self.port is None:
            raise SystemExit("no serial port of the board: plug it in (looked for " + ", ".join(PORT_GLOBS) + ")")
        self.s = serial.Serial(self.port, 115200, timeout=0.05)
        self.connected, self.reconnects = True, 0
        self.keep, self.lock, self.all = keep, threading.Lock(), []
        self.t, self.r = deque(), deque()
        self.buf, self.synced, self.stop = bytearray(), False, False

    def _sync(self):
        b = self.buf
        for p in range(len(b) - 2 * FRAME):
            if b[p + FRAME - 2:p + FRAME] == b"\r\n" and b[p + 2 * FRAME - 2:p + 2 * FRAME] == b"\r\n" \
                    and not all(32 <= c < 127 for c in b[p:p + FRAME - 2]):
                del b[:p]
                return True
        del b[:max(0, len(b) - 2 * FRAME)]
        return False

    def _reopen(self):
        try:
            self.s.close()
        except (serial.SerialException, OSError):
            pass
        self.connected, self.synced = False, False
        self.buf.clear()
        while not self.stop:
            time.sleep(0.5)
            port = self.port_arg if self.port_arg and os.path.exists(self.port_arg) else find_port()
            if port is None:
                continue
            try:
                self.s = serial.Serial(port, 115200, timeout=0.05)
            except (serial.SerialException, OSError):
                continue                              # the device node exists before udev grants access to it
            self.port, self.connected = port, True
            self.reconnects += 1
            return

    def run(self):
        while not self.stop:
            try:
                chunk = self.s.read(8192)
            except (serial.SerialException, OSError):
                self._reopen()
                continue
            self.buf += chunk
            now, recs = time.time(), []
            while len(self.buf) >= 2 * FRAME:
                if not self.synced:
                    self.synced = self._sync()
                    if not self.synced:
                        break
                if self.buf[FRAME - 2:FRAME] != b"\r\n":
                    self.synced = False
                    del self.buf[:1]
                    continue
                recs.append(np.frombuffer(bytes(self.buf[:FRAME - 2]), "<f4").reshape(N_CHIPS, 4).astype(float))
                del self.buf[:FRAME]
            with self.lock:
                for k, r in enumerate(recs):
                    t = now - (len(recs) - 1 - k) / 375.0
                    self.t.append(t)
                    self.r.append(r)
                    self.all.append((t, r))
                while self.t and now - self.t[0] > self.keep:
                    self.t.popleft()
                    self.r.popleft()
        self.s.close()

    def snapshot(self):
        with self.lock:
            return np.array(self.t), (np.array(self.r) if self.r else np.zeros((0, N_CHIPS, 4)))


def field(r):
    """Fields (n, chips, 3) with chips whose reading is zeros set to NaN."""
    b = r[:, :, 1:].copy()
    b[np.all(r == 0, axis=2)] = np.nan
    return b


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", default=None)
    ap.add_argument("--window", type=float, default=10.0, help="seconds shown")
    ap.add_argument("--out", default=None, help="write every record on exit: host time, then t x y z per chip")
    ap.add_argument("--png", default=None, help="render once after --seconds to this file and exit")
    ap.add_argument("--seconds", type=float, default=5.0)
    a = ap.parse_args()
    import matplotlib
    if a.png:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rd = Reader(a.port, max(a.window, BASE_S) + 1.0)
    rd.start()
    state = {"base": None}

    def baseline(last):
        t, r = rd.snapshot()
        if len(t) < 2:
            return
        sel = (t >= t[-1] - BASE_S) if last else (t <= t[0] + BASE_S)
        state["base"] = np.nanmean(field(r[sel]), axis=0)

    fig, axs = plt.subplots(N_CHIPS + 1, 1, figsize=(10, 9.5), sharex=True)
    colors = ("tab:red", "tab:green", "tab:blue")
    mag = [axs[0].plot([], [], label=f"{ADDRS[c]}")[0] for c in range(N_CHIPS)]
    comp = [[axs[c + 1].plot([], [], color=colors[k], label=f"dB{'xyz'[k]}")[0] for k in range(3)] for c in range(N_CHIPS)]
    axs[0].set_ylabel("|dB| (uT)")
    axs[0].set_title("|dB| now (uT)", fontsize=10)
    axs[0].legend(loc="upper left", ncol=5, fontsize=9)
    for c in range(N_CHIPS):
        axs[c + 1].set_ylabel(f"{ADDRS[c]} (uT)")
        axs[c + 1].legend(loc="upper left", ncol=3, fontsize=9)
    axs[-1].set_xlabel("Time (s)")
    for ax in axs:
        ax.grid(alpha=0.3)
    fig.canvas.mpl_connect("key_press_event", lambda e: baseline(True) if e.key == "b" else None)
    fig.tight_layout()

    def draw():
        t, r = rd.snapshot()
        if len(t) < 2:
            return
        if state["base"] is None and t[-1] - t[0] >= BASE_S:
            baseline(False)
        if state["base"] is None:
            return
        db = field(r) - state["base"]
        x = t - t[-1]
        for c in range(N_CHIPS):
            mag[c].set_data(x, np.linalg.norm(db[:, c], axis=1))
            for k in range(3):
                comp[c][k].set_data(x, db[:, c, k])
        for i, ax in enumerate(axs):
            v = np.linalg.norm(db, axis=2) if i == 0 else db[:, i - 1]
            lo, hi = np.nanmin(v), np.nanmax(v)
            mid, half = (lo + hi) / 2, max((hi - lo) / 2 * 1.15, 10.0)
            ax.set_ylim((0, max(hi * 1.15, 20.0)) if i == 0 else (mid - half, mid + half))
        axs[-1].set_xlim(-a.window, 0)
        now = np.linalg.norm(np.nanmean(db[x > -0.2], axis=0), axis=1)
        axs[0].set_title("|dB| now (uT): " + "  ".join(f"{ADDRS[c]} {now[c]:.0f}" for c in range(N_CHIPS))
                         + "     key b: new baseline", fontsize=10)

    if a.png:
        time.sleep(a.seconds)
        draw()
        fig.savefig(a.png, dpi=90)
        print("wrote", a.png)
    else:
        timer = fig.canvas.new_timer(interval=50)
        timer.add_callback(lambda: (draw(), fig.canvas.draw_idle()))
        timer.start()
        plt.show()
    rd.stop = True
    rd.join(timeout=1.0)
    if a.out and rd.all:
        np.savetxt(a.out, np.array([np.r_[t, r.ravel()] for t, r in rd.all]), fmt="%.4f",
                   header="host_time " + " ".join(f"{q}{c}" for c in range(N_CHIPS) for q in "txyz"))
        print("wrote", a.out)


if __name__ == "__main__":
    main()
