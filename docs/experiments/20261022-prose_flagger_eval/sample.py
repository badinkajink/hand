import json, random, subprocess, sys
pages = {"contact_model": 40, "printed_fingertips": 20, "rl_policies": 15, "hand_plant_bench": 10, "reorientation": 10,
         "code_and_methods": 5}
out = sys.argv[1]
rng = random.Random(20261022)
rows = []
for p, n in pages.items():
    js = subprocess.run(["python3", "scripts/prose_lint.py", "--flags", f"docs/overviews/{p}.html"], capture_output=True,
                        text=True).stdout.splitlines()
    rs = [json.loads(l) for l in js if l.startswith("{")]
    for r in rng.sample(rs, n):
        r["page"] = p
        rows.append(r)
for i, r in enumerate(rows):
    r["id"] = i
with open(out, "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
for r in rows:
    print(f"{r['id']:>3} [{r['page'][:8]}] {r['text']}")
