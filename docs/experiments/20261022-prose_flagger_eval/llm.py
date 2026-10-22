"""Classify sentences for readability problems with a local Qwen model; one JSON line per sentence."""
import json, re, sys, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

name, src, out = sys.argv[1], sys.argv[2], sys.argv[3]
quant = len(sys.argv) > 4 and sys.argv[4] == "nf4"
SYS = """You check sentences from engineering documentation for readability problems. The reader is an engineer who does not know the mathematics or the jargon well. Mark a sentence for rewrite when it has one of these problems:
- split: more than one comparison or step, more than about 30 words, or a list of more than four numbers.
- anim: a thing (a model, method, quantity, document, program or part) is the subject of a verb of perception, intention, speech or decision (sees, knows, wants, decides, lets, finds, needs, calls for, selects, reports, keeps). Physical verbs are fine: the spring pushes, the finger presses, the tool turns.
- compress: three or more nouns stacked ("friction-row time constant"), a noun used as a verb, a relative clause without "that", a telegraphic fragment ("arm 0.91 and 0.93"), or a chain of reasoning squeezed into one clause.
- term: a coined name or jargon used without a definition ("working plant", "the wield", "rolling bristles", "concept a"), or an abbreviation that is not spelled out.
- style: a cleft ("It is X that"), "X, not Y", "rather than", or one of the words holds (as a container), carries, serves, settle, lands, lives in, buys, leverage, notably, importantly, crucially, key insight, fundamentally, robust, matched, diagnostic, bounded, explicitly, directly, specifically, evidence.
X stands for a formula. Figure captions, legends, glossary entries and short formula fragments are fine unless they have one of these problems.
Answer with JSON only: {"rewrite": true or false, "reasons": [...]}"""
SHOTS = [
    ("The bed gives 30.76, 3.08 and 1.03 µm/s at impratio 100, 1000 and 3000, and 12.31 µm/s at X = 25 ms.", '{"rewrite": true, "reasons": ["split"]}'),
    ("The coefficient encodes the patch, and the solver decides how much torque the pad needs.", '{"rewrite": true, "reasons": ["anim"]}'),
    ("Its arm 0.91 and 0.93 of the elements' at 1 and 3 N under the reader-patience budget.", '{"rewrite": true, "reasons": ["compress"]}'),
    ("Each spring pushes back in proportion to its compression.", '{"rewrite": false, "reasons": []}'),
    ("Left: the pressure contours of the field at 3 N.", '{"rewrite": false, "reasons": []}'),
]
tok = AutoTokenizer.from_pretrained(name)
tok.padding_side = "left"
kw = {"dtype": torch.bfloat16, "device_map": "cuda"}
if quant:
    from transformers import BitsAndBytesConfig
    kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(name, **kw).eval()
print(f"loaded {name} in {time.time() - t0:.0f} s, {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
rows = [json.loads(l) for l in open(src)]
def prompt(text):
    m = [{"role": "system", "content": SYS}]
    for s, a in SHOTS:
        m += [{"role": "user", "content": "Sentence: " + s}, {"role": "assistant", "content": a}]
    m.append({"role": "user", "content": "Sentence: " + text})
    try:
        return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
t0 = time.time()
with open(out, "w") as f:
    for i in range(0, len(rows), 16):
        batch = rows[i:i + 16]
        enc = tok([prompt(r["text"]) for r in batch], return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=48, do_sample=False)
        for r, seq in zip(batch, g[:, enc["input_ids"].shape[1]:]):
            ans = tok.decode(seq, skip_special_tokens=True)
            m = re.search(r"\{.*\}", ans, re.S)
            try:
                j = json.loads(m.group(0)) if m else None
            except Exception:
                j = None
            f.write(json.dumps({"id": r["id"], "raw": ans.strip(), "parsed": j}, ensure_ascii=False) + "\n")
        f.flush()
print(f"classified {len(rows)} in {time.time() - t0:.0f} s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
