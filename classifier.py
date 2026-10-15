"""Optional supervised vote: a fine-tuned encoder classifier (default tabularisai/ai-text-detection, ModernBERT-base,
149M parameters, MIT, trained in 2026 on ~11M RAID-era rows; label 1 = AI). Text is NFKC-normalised first, which closes
the zero-width / homoglyph attacks the model card reports. Supervised detectors score very high in-distribution and
collapse on generators and registers they were not trained on, so this is one vote among many, never a verdict.
Usage: python classifier.py file.txt [...] -> JSON"""
import os, sys, json, pathlib, unicodedata
HERE = pathlib.Path(__file__).resolve().parent
_md = os.environ.get("AITD_MODELS_DIR") or ((HERE / ".models-dir").read_text().strip() if (HERE / ".models-dir").is_file() else "") or str(HERE / "models")
if pathlib.Path(_md).is_dir(): os.environ.setdefault("HF_HOME", _md)
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch

def load(mid, dev):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(mid); m = AutoModelForSequenceClassification.from_pretrained(mid).to(dev).eval()
    return tok, m

def p_ai(model, tok, text, dev, ai_label=1):
    """Probability the classifier assigns to the AI label, on the first 512 tokens (its training length)."""
    enc = tok(unicodedata.normalize("NFKC", text), return_tensors="pt", truncation=True, max_length=512).to(dev)
    with torch.no_grad(): p = model(**enc).logits.softmax(-1)[0]
    return float(p[ai_label])

MEASURES = [("clf_p_ai", "Supervised classifier: P(AI) (ModernBERT, 2026)", False, "tabularisai/ai-text-detection; RAID AUROC 0.99 in-distribution, weak on unseen generators.")]

if __name__ == "__main__":
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok, m = load("tabularisai/ai-text-detection", dev)
    print(json.dumps({f: p_ai(m, tok, open(f, encoding="utf-8").read(), dev) for f in sys.argv[1:]}, indent=1))
