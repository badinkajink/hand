"""P(yes) from a local model's next-token logits, per sentence and per question; one JSON line per sentence."""
import json, sys, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

name, src, out = sys.argv[1], sys.argv[2], sys.argv[3]
quant = len(sys.argv) > 4 and sys.argv[4] == "nf4"
READER = "The reader is an engineer who does not know the mathematics or the jargon well. X stands for a formula."
Q = {
    "rewrite": ("Would this sentence from engineering documentation be clearly easier to read if rewritten? Problems: "
                "more than one comparison or step, over 30 words, more than four numbers, a thing that sees, decides, "
                "lets or finds, stacked nouns, telegraphic fragments, undefined jargon or coined names, clefts, "
                "'X, not Y', inflated words.",
                [("The bed gives 30.76, 3.08 and 1.03 µm/s at impratio 100, 1000 and 3000, and 12.31 µm/s at X = 25 ms.", "yes"),
                 ("Each spring pushes back in proportion to its compression.", "no"),
                 ("Its arm 0.91 and 0.93 of the elements' at 1 and 3 N under the reader-patience budget.", "yes"),
                 ("Left: the pressure contours of the field at 3 N.", "no")]),
    "split": ("Does this sentence pack more than one comparison or step, run over about 30 words, or list more than "
              "four numbers, so that it should be split into several sentences?",
              [("The creep is 30.76 µm/s at impratio 100, falls to 3.08 and 1.03 µm/s at 1000 and 3000, and a longer X lowers it to 12.31 µm/s at 25 ms, while Drake creeps 0.14 µm/s.", "yes"),
               ("The creep falls to 3.08 µm/s at impratio 1000.", "no"),
               ("A sphere meets the tool under its centre, the centres lie on radius X, so the lattice shrinks by X and its points are X times as dense.", "yes"),
               ("Each touching sphere is a soft contact with its own friction cone.", "no")]),
    "anim": ("Does this sentence make a thing (a model, method, quantity, document, program or part) the subject of a "
             "verb of perception, intention, speech or decision, such as sees, knows, wants, decides, lets, finds, "
             "needs, selects, reports or calls for? Physical verbs like pushes, presses, turns or slides do not count.",
             [("The coefficient encodes the patch, and the solver decides how much torque the pad needs.", "yes"),
              ("The springs push back and the thumb presses the cylinder.", "no"),
              ("The point contacts let the tool swing to 175 degrees.", "yes"),
              ("The tool swings to 175 degrees between the point contacts.", "no")]),
    "compress": ("Is this sentence compressed: three or more nouns stacked together, a noun used as a verb, a relative "
                 "clause without 'that', a telegraphic fragment with missing verbs, or a chain of reasoning squeezed "
                 "into one clause?",
                 [("Its arm 0.91 and 0.93 of the elements' at 1 and 3 N.", "yes"),
                  ("Its arm is 0.91 and 0.93 times the elements' arm at 1 and 3 N.", "no"),
                  ("The friction-row time constant sets the reader-patience budget.", "yes"),
                  ("The time constant of the friction rows sets how fast they respond.", "no")]),
    "term": ("Does this sentence use a coined name or jargon without defining it (a name the field does not use for a "
             "setup or concept, such as 'the working plant', 'the wield', 'rolling bristles', 'concept a'), or an "
             "abbreviation that is not spelled out?",
             [("On the working plant the wield keeps the tool.", "yes"),
              ("The finger servos have a torque limit of 0.18 N m.", "no"),
              ("Concept a resolves 0.065 N on the 11/22 tip.", "yes"),
              ("The tool is a screwdriver of radius 12.5 mm.", "no")]),
    "style": ("Does this sentence use a cleft ('It is X that'), an inversion ('not X but Y'), 'X, not Y', 'rather than', "
              "or a word such as holds (as a container), carries, serves, lands, leverage, notably, importantly, "
              "crucially, fundamentally, robust, matched, diagnostic, bounded, explicitly, directly, specifically or evidence?",
              [("It is the friction torque that keeps the tool up, not the normal force.", "yes"),
               ("The friction torque keeps the tool up.", "no"),
               ("The page holds every result and serves as the reference.", "yes"),
               ("The page lists every result.", "no")]),
}
tok = AutoTokenizer.from_pretrained(name)
tok.padding_side = "left"
kw = {"dtype": torch.bfloat16, "device_map": "cuda"}
if quant:
    from transformers import BitsAndBytesConfig
    kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(name, **kw).eval()
yes = [tok.encode(w, add_special_tokens=False)[0] for w in ("yes", " yes", "Yes")]
no = [tok.encode(w, add_special_tokens=False)[0] for w in ("no", " no", "No")]
def prompt(q, shots, text):
    m = [{"role": "system", "content": READER + " " + q + " Answer yes or no."}]
    for s, a in shots:
        m += [{"role": "user", "content": "Sentence: " + s}, {"role": "assistant", "content": a}]
    m.append({"role": "user", "content": "Sentence: " + text})
    try:
        return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
rows = [json.loads(l) for l in open(src)]
res = {r["id"]: {"id": r["id"]} for r in rows}
t0 = time.time()
for k, (q, shots) in Q.items():
    for i in range(0, len(rows), 8):
        batch = rows[i:i + 8]
        enc = tok([prompt(q, shots, r["text"]) for r in batch], return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            lg = model(**enc).logits[:, -1, :].float()
        lp = torch.log_softmax(lg, -1)
        py = torch.logsumexp(lp[:, yes], -1)
        pn = torch.logsumexp(lp[:, no], -1)
        p = torch.sigmoid(py - pn).tolist()
        for r, v in zip(batch, p):
            res[r["id"]][k] = round(v, 4)
with open(out, "w") as f:
    for v in res.values():
        f.write(json.dumps(v) + "\n")
print(f"scored {len(rows)} x {len(Q)} in {time.time() - t0:.0f} s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
