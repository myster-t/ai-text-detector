"""Perplexity + burstiness (GPTZero-style) for a set of text files.
Usage: python perplexity.py MODEL file1 file2 ...   (MODEL e.g. gpt2 or Qwen/Qwen2.5-1.5B)
Reports: document perplexity, mean per-sentence perplexity, burstiness (std of sentence ppl)."""
import os, sys, re, math, json, pathlib
HERE = pathlib.Path(__file__).resolve().parent
_md = os.environ.get("AITD_MODELS_DIR") or ((HERE / ".models-dir").read_text().strip() if (HERE / ".models-dir").is_file() else "") or str(HERE / "models")
if pathlib.Path(_md).is_dir(): os.environ.setdefault("HF_HOME", _md)
os.environ.setdefault('HF_HUB_OFFLINE', '1'); os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
model_id, files = sys.argv[1], sys.argv[2:]
dev = "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(model_id); m = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).to(dev).eval()
def ppl(text):
    ids = tok(text, return_tensors="pt", truncation=True, max_length=1024).input_ids.to(dev)
    if ids.shape[1] < 2: return float("nan")
    with torch.no_grad(): loss = m(ids, labels=ids).loss.item()
    return math.exp(loss)
rows = []
for f in files:
    t = open(f).read().strip()
    sents = [s for s in re.split(r"(?<=[.!?])\s+", t) if len(s.split()) >= 4]
    sp = [ppl(s) for s in sents]
    mean = sum(sp)/len(sp); std = (sum((x-mean)**2 for x in sp)/len(sp))**0.5
    rows.append({"file": f.split("/")[-1], "words": len(t.split()), "doc_ppl": round(ppl(t),2), "sent_ppl_mean": round(mean,2), "burstiness_sd": round(std,2), "n_sent": len(sp)})
print(json.dumps({"model": model_id, "rows": rows}, indent=1))
