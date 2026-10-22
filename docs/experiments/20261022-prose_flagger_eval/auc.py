import json, sys
sys.path.insert(0, ".")
from labels import L
from score import nb
rows = {json.loads(l)["id"]: json.loads(l) for l in open("sample.jsonl")}
def auc(score, gold):
    pos = [score[i] for i in score if i in gold]; neg = [score[i] for i in score if i not in gold]
    if not pos or not neg: return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))
def spacy_score(r, reason):
    f = nb(r) and r["flags"]
    if reason == "rewrite": return r["score"] if r["flags"] else 0
    if reason == "split": return r["words"] + 5 * min(len([1 for k in ("nums", "math") if k in r["flags"]]), 2)
    if reason == "anim": return 1 if "anim" in r["flags"] else 0
    if reason == "compress": return len(r["flags"].get("stack", [])) + len(r["flags"].get("relcl", []))
    if reason == "style": return 1 if any(k in r["flags"] for k in ("cleft", "notx", "dash")) or ("banned" in r["flags"] and nb(r) & {"banned"}) else 0
    return 0
gold = {"rewrite": set(L)}
for reason in ["split", "anim", "compress", "term", "style"]:
    gold[reason] = {i for i, v in L.items() if reason in v.split()}
llm = {f: {json.loads(l)["id"]: json.loads(l) for l in open(f)} for f in sys.argv[1:]}
print(f"{'question':<10} {'n':>3}  {'spaCy':>6}  " + "  ".join(f"{f.split('_', 1)[1].replace('.jsonl', ''):>10}" for f in llm))
for k, g in gold.items():
    s = auc({i: spacy_score(r, k) for i, r in rows.items()}, g)
    print(f"{k:<10} {len(g):>3}  {s:>6.2f}  " + "  ".join(f"{auc({i: v[k] for i, v in d.items()}, g):>10.2f}" for d in llm.values()))
