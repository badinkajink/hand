"""Serve the board logger's live log as a mux board on a pseudo-terminal, so that host/wrench_viewer.py --mux can run while
lab/board_log.py holds the C6's port (and run_plan.py's board watch reads that logger's file). It follows the newest
<time>-mux.bin in BOARDS_DIR from its end and writes each new record as firmware/reskin5x_mux_stream sends it: the 80 n
float32 bytes, then "\\r\\n". A byte 'i' from the reader gets the firmware's answer between two records ("Boards on mux
channels: ...", "Record bytes: ..."), from the log's sidecar. LINK is a symlink to the pseudo-terminal. The records come
in the logger's bursts (its file buffer), about 0.2 s apart; the viewer spaces each burst's records evenly.

    /usr/bin/python3 board_log_pty.py BOARDS_DIR LINK
"""
import json
import os
import select
import sys
import time
import tty
from pathlib import Path

boards, link = Path(sys.argv[1]), Path(sys.argv[2])
master, slave = os.openpty()
tty.setraw(slave)
os.set_blocking(master, False)
link.unlink(missing_ok=True)
link.symlink_to(os.ttyname(slave))
print(f"{time.strftime('%H:%M:%S')} serving {boards} on {os.ttyname(slave)} ({link})", flush=True)
cur = f = None
pos = rec = 0
chans = []
want_info = False
while True:
    files = sorted(boards.glob("*-mux.bin"))
    if files and files[-1] != cur:
        cur = files[-1]
        chans = json.loads(cur.with_suffix(".json").read_text())["channels"]
        rec = 8 + 80 * len(chans)
        f = open(cur, "rb")
        pos = os.path.getsize(cur) // rec * rec
        print(f"{time.strftime('%H:%M:%S')} following {cur.name}, channels {chans}", flush=True)
    if select.select([master], [], [], 0.05)[0]:
        try:
            want_info |= b"i" in os.read(master, 1024)
        except OSError:
            pass
    out = bytearray()
    if want_info and chans:
        out += f"Mux at 0x71\r\nBoards on mux channels: {' '.join(map(str, chans))} \r\nRecord bytes: {80 * len(chans) + 2}\r\n".encode()
        want_info = False
    if f is not None:
        f.seek(pos)
        data = f.read()
        n = len(data) // rec
        for k in range(n):
            out += data[k * rec + 8:(k + 1) * rec] + b"\r\n"
        pos += n * rec
    if out:
        try:
            os.write(master, out)
        except (BlockingIOError, OSError):
            pass                           # nobody reading, or reading too slowly: drop, the reader resyncs on \r\n
    time.sleep(0.02)
