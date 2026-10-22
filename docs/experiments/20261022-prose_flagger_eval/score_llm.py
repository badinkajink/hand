import json, sys
sys.path.insert(0, ".")
from labels import L
from score import prf, nb, MAP
rows = {json.loads(l)["id"]: json.loads(l) for l in open("sample.jsonl")}
gold = set(L)
for f in sys.argv[1:]:
    out = [json.loads(l) for l in open(f)]
    bad = sum(o["parsed"] is None for o in out)
    pred = {o["id"] for o in out if o["parsed"] and o["parsed"].get("rewrite")}
    print(f"\n{f}: unparsed {bad}")
    print("  LLM rewrite        :", prf(pred, gold))
    sp = {i for i, r in rows.items() if nb(r)}
    print("  LLM or spaCy       :", prf(pred | sp, gold))
    print("  LLM and spaCy      :", prf(pred & sp, gold))
    for reason in ["split", "anim", "compress", "term", "style"]:
        g = {i for i, v in L.items() if reason in v.split()}
        p = {o["id"] for o in out if o["parsed"] and reason in (o["parsed"].get("reasons") or [])}
        s = {i for i, r in rows.items() if nb(r) & MAP[reason]}
        print(f"  {reason:<9} LLM {prf(p, g)}   | LLM or spaCy {prf(p | s, g)}")
