"""Token-level language-model statistics from a single forward pass of one causal LM.
Gives: per-token log-probability, rank of the actual token, next-token entropy; and document aggregates used by
GLTR (Gehrmann et al. 2019: share of tokens in the model's top-10/100/1000), DetectLLM's log-likelihood/log-rank ratio
(Su et al. 2023), mean log-rank, mean entropy, and perplexity. Used by report.py; also runnable on its own:
python lmfeat.py MODEL file.txt [...] -> JSON"""
import os, sys, math, json, pathlib
HERE = pathlib.Path(__file__).resolve().parent
_md = os.environ.get("AITD_MODELS_DIR") or ((HERE / ".models-dir").read_text().strip() if (HERE / ".models-dir").is_file() else "") or str(HERE / "models")
if pathlib.Path(_md).is_dir(): os.environ.setdefault("HF_HOME", _md)
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch, torch.nn.functional as F

def token_stats(model, tok, text, dev, max_length=1024):
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_length, return_offsets_mapping=True)
    ids = enc.input_ids.to(dev); offs = enc["offset_mapping"][0].tolist()
    if ids.shape[1] < 2: return None
    with torch.no_grad(): lg = model(input_ids=ids).logits[0, :-1].float()
    logp = F.log_softmax(lg, -1); tgt = ids[0, 1:]
    lp = logp.gather(1, tgt[:, None])[:, 0]                                  # log p(actual token)
    rank = (logp > lp[:, None]).sum(1) + 1                                    # 1 = model's top choice
    ent = -(logp.exp() * logp).sum(1)                                         # next-token entropy (nats)
    lp, rank, ent = lp.tolist(), rank.tolist(), ent.tolist()
    toks = [{"s": offs[i + 1][0], "e": offs[i + 1][1], "lp": lp[i], "rank": rank[i], "ent": ent[i]} for i in range(len(lp))]
    n = len(lp); lr = [math.log(r) for r in rank]
    dyn = dynamics(lp)
    return {"n": n, "tokens": toks, "ppl": math.exp(-sum(lp) / n), "mean_logrank": sum(lr) / n, **dyn,
            "lrr": (-sum(lp) / n) / (sum(lr) / n) if sum(lr) else float("nan"),       # DetectLLM LRR: higher = more machine-like
            "mean_entropy": sum(ent) / n,
            "top1": sum(r == 1 for r in rank) / n, "top10": sum(r <= 10 for r in rank) / n,
            "top100": sum(r <= 100 for r in rank) / n, "top1000": sum(r <= 1000 for r in rank) / n}

def dynamics(lp):
    """Surprisal-dynamics features of the per-token log-probability sequence (the Lastde / DivEye / late-stage-stability
    family): the sequence's spread, its local volatility, lag-1 autocorrelation, and second-half vs first-half volatility."""
    n = len(lp)
    if n < 8: return {"lp_sd": float("nan"), "lp_volatility": float("nan"), "lp_autocorr1": float("nan"), "late_stability": float("nan")}
    mean = sum(lp) / n; sd = (sum((x - mean) ** 2 for x in lp) / n) ** 0.5
    d = [abs(b - a) for a, b in zip(lp, lp[1:])]; vol = sum(d) / len(d)
    ac = sum((a - mean) * (b - mean) for a, b in zip(lp, lp[1:])) / (sum((x - mean) ** 2 for x in lp) or 1e-9)
    h = n // 2; v1 = sum(d[:h - 1]) / max(h - 1, 1); v2 = sum(d[h:]) / max(len(d) - h, 1)
    return {"lp_sd": sd, "lp_volatility": vol, "lp_autocorr1": ac, "late_stability": (v2 / v1) if v1 else float("nan")}

def shuffle_ratio(model, tok, text, dev, seed=7):
    """log( perplexity of the text with its sentences shuffled / perplexity of the original ). After Luminol-AIDetect
    (La Cava & Tagarelli 2026): the two groups respond differently to losing sentence order."""
    import random, re
    sents = [x for x in re.split(r"(?<=[.!?])\s+", text.strip()) if x]
    if len(sents) < 3: return float("nan")
    rnd = random.Random(seed); sh = sents[:]; rnd.shuffle(sh)
    a = token_stats(model, tok, text, dev); b = token_stats(model, tok, " ".join(sh), dev)
    return math.log(b["ppl"] / a["ppl"]) if a and b else float("nan")

MEASURES = [  # key, title, higher_is_human, note
    ("top10", "GLTR: share of tokens in the model's top-10 predictions", False, "Gehrmann et al. 2019. Sampled text draws mostly from the head of the distribution; human text reaches further into the tail."),
    ("top1", "GLTR: share of tokens that were the model's first choice", False, ""),
    ("mean_logrank", "Mean log-rank of the actual token", True, "Averaged log of where the real next token sat in the model's ranking. Higher = less predictable."),
    ("lrr", "Log-likelihood / log-rank ratio (DetectLLM LRR)", False, "Su et al. 2023. Ratio of mean negative log-probability to mean log-rank; machine text scores higher."),
    ("mean_entropy", "Mean next-token entropy (nats)", True, "How spread the model's prediction was at each position; low-entropy contexts are where generators follow the model."),
    ("lp_sd", "Surprisal spread (sd of token log-probability)", True, "Lastde / DivEye family: human text swings more from token to token."),
    ("lp_volatility", "Surprisal volatility (mean |step change|)", True, "Mean absolute change between consecutive token log-probabilities."),
    ("late_stability", "Late-stage stability (2nd-half / 1st-half volatility)", True, "Sun et al. 2026: generated text settles into lower volatility in its second half."),
    ("lp_autocorr1", "Surprisal lag-1 autocorrelation", None, "Whether surprising tokens cluster."),
    ("shuffle_ratio", "Perplexity shift under sentence shuffling (log ratio)", False, "After Luminol-AIDetect 2026: machine text loses more predictability when its sentence order is destroyed."),
]

if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForCausalLM
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    mid, files = sys.argv[1], sys.argv[2:]
    tok = AutoTokenizer.from_pretrained(mid); m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32).to(dev).eval()
    out = {}
    for f in files:
        r = token_stats(m, tok, open(f, encoding="utf-8").read().strip(), dev); r.pop("tokens"); out[f] = r
    print(json.dumps(out, indent=1))
