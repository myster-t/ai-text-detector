"""Data-derived "slop" vocabulary and phrase rates. Lists are vendored in third_party/ with their licences:
- antislop-sampler (Sam Paech, Apache 2.0): 2,000 words and 2,500 phrases over-represented in LLM output relative to
  human text (built from creative-writing generations, 2025-04-07), plus three construction regexes.
- slop-forensics (Sam Paech, MIT): essay-domain lists of over-represented words, bigrams and trigrams across ten
  2025 models (GPT-4.1-nano, GPT-4o-mini, gpt-oss-20b, Gemini 2.5 Flash-Lite, Gemma 3, Llama 4 Scout, Mistral Nemo,
  Qwen3-30B, Grok 4 Fast, GLM-4.6) versus human writing.
Rates are per 1,000 words; matches carry positions so the report can mark them. These describe vocabulary habits of
2025-era models; they are editorial evidence, not proof of authorship. Usage: python slop.py file.txt [...] -> JSON"""
import sys, json, re, pathlib
HERE = pathlib.Path(__file__).resolve().parent
_A = HERE / "third_party" / "antislop"; _F = HERE / "third_party" / "slop-forensics"
def _load():
    aw = {w for w, _ in json.loads((_A / "slop_words_2025-04-07.json").read_text())}
    ap = [p for p, _ in json.loads((_A / "slop_phrases_2025-04-07.json").read_text())]
    rx = []
    for l in (_A / "slop_regexes.txt").read_text().splitlines():
        l = l.strip().strip(',').strip('"')
        if l: rx.append(re.compile(l.replace("(?i)", "(?i)(?<![A-Za-z])", 1) if l.startswith("(?i)") else "(?<![A-Za-z])" + l))
    fw = {w[0] for w in json.loads((_F / "slop_list.json").read_text())}
    fb = {" ".join(b[0].split()) for b in json.loads((_F / "slop_list_bigrams.json").read_text())}
    ft = {" ".join(t[0].split()) for t in json.loads((_F / "slop_list_trigrams.json").read_text())}
    top = [w for w, _ in json.loads((_F / "slop_list_by_freq.json").read_text())[:300]]
    return aw, ap, rx, fw, fb, ft, set(top)
AW, AP, RX, FW, FB, FT, FTOP = _load()
_W = re.compile(r"[A-Za-z][A-Za-z'’-]*")

def analyze(text):
    toks = [(m.group().lower().replace("’", "'"), m.start(), m.end()) for m in _W.finditer(text)]
    n = max(len(toks), 1); low = text.lower()
    hits = []
    # single words (essay-domain list, strongest for prose; antislop list is fiction-leaning, reported separately)
    fw_hits = [(s, e, "slop word (essays list)") for w, s, e in toks if w in FW]
    ftop_hits = [(s, e, "slop word (top-300 by frequency)") for w, s, e in toks if w in FTOP]
    aw_hits = [(s, e, "slop word (antislop, fiction-leaning)") for w, s, e in toks if w in AW]
    # n-grams
    bg, tg = [], []
    for i in range(len(toks) - 1):
        b = toks[i][0] + " " + toks[i + 1][0]
        if b in FB: bg.append((toks[i][1], toks[i + 1][2], "slop bigram"))
        if i < len(toks) - 2:
            t = b + " " + toks[i + 2][0]
            if t in FT: tg.append((toks[i][1], toks[i + 2][2], "slop trigram"))
    ph = []
    for p in AP:
        for m in re.finditer(r"(?<![a-z])" + re.escape(p) + r"(?![a-z])", low): ph.append((m.start(), m.end(), "slop phrase (antislop)"))
    cx = []
    for r in RX:
        for m in r.finditer(text): cx.append((m.start(), m.end(), "construction: " + r.pattern[:30]))
    per_k = lambda k: k / n * 1000
    return {"words": len(toks),
            "slop_word_per_k": per_k(len(fw_hits)), "slop_top300_per_k": per_k(len(ftop_hits)), "antislop_word_per_k": per_k(len(aw_hits)),
            "slop_ngram_per_k": per_k(len(bg) + len(tg)), "slop_phrase_per_k": per_k(len(ph)), "slop_construction_per_k": per_k(len(cx)),
            "matches": sorted(set(fw_hits + bg + tg + ph + cx))[:400]}

MEASURES = [  # key, title, higher_is_human, note
    ("slop_word_per_k", "Slop vocabulary rate, essays list (per 1,000 words)", False, "slop-forensics 2025: words over-represented in ten models' essays vs human writing."),
    ("slop_top300_per_k", "Slop vocabulary rate, top-300 list (per 1,000 words)", False, "The 300 most frequent over-represented essay words (crucial, strategic, challenges...)."),
    ("slop_ngram_per_k", "Slop bigram and trigram rate (per 1,000 words)", False, "slop-forensics essay n-grams (plays crucial role, make informed decisions...)."),
    ("slop_phrase_per_k", "Slop phrase rate, antislop list (per 1,000 words)", False, "antislop 2025 phrases (took a deep breath, couldn't help but feel...); fiction-leaning."),
    ("slop_construction_per_k", "Template constructions (not X but, each/every a)", False, "antislop regexes."),
]
if __name__ == "__main__":
    out = {}
    for f in sys.argv[1:]:
        r = analyze(open(f, encoding="utf-8").read()); r["matches"] = r["matches"][:20]; out[f] = r
    print(json.dumps(out, indent=1))
