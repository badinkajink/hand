import json, sys
sys.path.insert(0, ".")
from labels import L
rows = [json.loads(l) for l in open("sample.jsonl")]
MAP = {"split": {"long", "nums", "math"}, "anim": {"anim"}, "compress": {"stack", "relcl"},
       "style": {"cleft", "notx", "dash", "banned"}, "term": set()}
def prf(pred, gold):
    tp = len(pred & gold); p = tp / len(pred) if pred else 0; r = tp / len(gold) if gold else 0
    return f"P {p:.2f} R {r:.2f} F1 {2*p*r/(p+r) if p+r else 0:.2f} (flagged {len(pred)}, true {len(gold)})"
ids = {r["id"] for r in rows}
gold = set(L)
def spacy_pred(r, keys=None):
    f = r["flags"]
    return bool([k for k in f if keys is None or k in keys])
print("spaCy flags, any flag      :", prf({r["id"] for r in rows if spacy_pred(r)}, gold))
nb = lambda r: {k for k in r["flags"] if not (k == "banned" and all(w.lower() in ("held", "hold", "holds", "holding") for w in r["flags"]["banned"]))}
print("spaCy flags, 'held' dropped:", prf({r["id"] for r in rows if nb(r)}, gold))
for reason, keys in MAP.items():
    g = {i for i, v in L.items() if reason in v.split()}
    print(f"  {reason:<9}", prf({r["id"] for r in rows if nb(r) & keys}, g))
