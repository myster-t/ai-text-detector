"""Binoculars (Hans et al. 2024) zero-shot detector, reference pairing: observer tiiuae/falcon-7b, performer tiiuae/falcon-7b-instruct.
Score = PPL_performer(text) / cross-PPL(observer -> performer). LOWER = more likely AI.
Reference thresholds: 0.9015 (accuracy-optimal), 0.8536 (low false-positive).  Usage: python binoculars.py file1 file2 ..."""
import os, sys, json, pathlib
HERE = pathlib.Path(__file__).resolve().parent.parent   # repo root (this file lives in aitd/)
_md = os.environ.get("AITD_MODELS_DIR") or ((HERE / ".models-dir").read_text().strip() if (HERE / ".models-dir").is_file() else "") or str(HERE / "models")
if pathlib.Path(_md).is_dir(): os.environ.setdefault("HF_HOME", _md)
os.environ.setdefault('HF_HUB_OFFLINE', '1'); os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
OBS, PERF = "tiiuae/falcon-7b", "tiiuae/falcon-7b-instruct"
T_ACC, T_FPR = 0.9015310749276843, 0.8536432310785527
dev = "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(OBS)
if tok.pad_token is None: tok.pad_token = tok.eos_token
obs = AutoModelForCausalLM.from_pretrained(OBS, dtype=torch.bfloat16).to(dev).eval()
perf = AutoModelForCausalLM.from_pretrained(PERF, dtype=torch.bfloat16).to(dev).eval()
def score(text):
    enc = tok(text, return_tensors="pt", truncation=True, max_length=512).to(dev)
    with torch.no_grad():
        lo = obs(**enc).logits[:, :-1].float(); lp = perf(**enc).logits[:, :-1].float()
    labels = enc.input_ids[:, 1:]
    ppl = F.cross_entropy(lp.transpose(1, 2), labels, reduction="mean").item()            # performer perplexity (log)
    p_obs = F.softmax(lo, dim=-1); logq = F.log_softmax(lp, dim=-1)
    xppl = (-(p_obs * logq).sum(-1)).mean().item()                                          # cross-perplexity (log)
    return ppl / xppl, ppl, xppl, labels.shape[1]
rows = []
for f in sys.argv[1:]:
    t = open(f).read().strip(); s, ppl, xppl, n = score(t)
    verdict = "AI (below low-FPR line)" if s < T_FPR else ("AI (below accuracy line)" if s < T_ACC else "human")
    rows.append({"file": f.split("/")[-1], "tokens": n, "binoculars": round(s, 4), "log_ppl": round(ppl, 3), "log_xppl": round(xppl, 3), "verdict": verdict})
print(json.dumps({"observer": OBS, "performer": PERF, "thresholds": {"accuracy": T_ACC, "low_fpr": T_FPR}, "rows": rows}, indent=1))
