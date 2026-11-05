"""Tiny client for the CB1 control service (workstation side)."""
import json, time, urllib.request

BASE = "http://10.99.99.2:8765/api/v1"
TOKEN = "5dbd5b618d7d02af4cb571cf813cebdc"

def get(path, timeout=8):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return json.load(r)

def post(path, body=None, timeout=30):
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(BASE + path, data=data, method="POST",
                                 headers={"Content-Type": "application/json",
                                          "X-Manta-Token": TOKEN})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{path} -> {e.code} {e.read().decode()[:300]}") from None

def wait_idle(timeout=240, poll=0.5):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        s = get("/state")
        if not s["busy"]:
            return s
        time.sleep(poll)
    raise TimeoutError("operation did not finish")
