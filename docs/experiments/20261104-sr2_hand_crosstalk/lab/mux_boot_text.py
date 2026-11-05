"""Reset the mux board through its USB serial unit's RTS line and print its start-up text (the channel scan)."""
import sys, time
import serial

p = serial.Serial(sys.argv[1], 115200, timeout=0.05)
p.dtr = False
p.rts = True
time.sleep(0.2)
p.rts = False
t0 = time.time()
buf = b""
while time.time() - t0 < float(sys.argv[2]) if len(sys.argv) > 2 else 4.0:
    try:
        buf += p.read(4096)
    except serial.SerialException as e:
        print("serial:", e); time.sleep(0.2)
        try:
            p.close(); p = serial.Serial(sys.argv[1], 115200, timeout=0.05)
        except Exception as e2:
            print("reopen:", e2)
    if len(buf) > 200000:
        break
# the text before the binary records
text = buf.split(b"Record bytes")[0] + b"Record bytes" + buf.split(b"Record bytes")[1][:12] if b"Record bytes" in buf else buf[:3000]
print(text.decode("ascii", "replace"))
print(len(buf), "bytes in all")
