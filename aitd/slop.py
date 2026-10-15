"""Data-derived "slop" vocabulary and phrase rates. Lists are vendored in third_party/ with their licences:
- antislop-sampler (Sam Paech, Apache 2.0): 2,000 words and 2,500 phrases over-represented in LLM output relative to
  human text (built from creative-writing generations, 2025-04-07), plus three construction regexes.
- slop-forensics (Sam Paech, MIT): essay-domain lists of over-represented words, bigrams and trigrams across ten
  2025 models (GPT-4.1-nano, GPT-4o-mini, gpt-oss-20b, Gemini 2.5 Flash-Lite, Gemma 3, Llama 4 Scout, Mistral Nemo,
  Qwen3-30B, Grok 4 Fast, GLM-4.6) versus human writing.
Rates are per 1,000 words; matches carry positions so the report can mark them. These describe vocabulary habits of
2025-era models; they are editorial evidence, not proof of authorship. Usage: python slop.py file.txt [...] -> JSON"""
import sys, json, re, pathlib
HERE = pathlib.Path(__file__).resolve().parent.parent   # repo root (this file lives in aitd/)
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

# Kobak et al. 2025 (MIT): 2024 excess "style" words in PubMed abstracts (verbs/adjectives/adverbs a model over-uses).
import csv
KOBAK = {row["word"].lower() for row in csv.DictReader(open(HERE / "third_party" / "llm-excess-vocab" / "excess_words.csv", encoding="utf-8")) if row.get("type") == "style"}
# Juzek & Ward 2025 (data CC0): the 21 focal words that spike in both PubMed 2020-24 and GPT-3.5 rewrites.
JUZEK = {"delves", "delved", "delving", "delve", "showcasing", "showcases", "boasts", "underscores", "underscoring", "underscore", "comprehending",
         "intricacies", "intricate", "surpassing", "surpasses", "garnered", "emphasizing", "realm", "groundbreaking", "advancements", "aligns"}
# Wikipedia "Signs of AI writing" core words-to-watch box (CC BY-SA 4.0), 2025-26.
WIKI = {"additionally", "boasts", "bolstered", "crucial", "delve", "delves", "delving", "emphasizing", "enduring", "enhance", "enhancing", "fostering", "foster",
        "garner", "garnered", "interplay", "intricate", "intricacies", "landscape", "meticulous", "meticulously", "pivotal", "robust", "showcase", "showcasing",
        "tapestry", "testament", "underscore", "underscores", "underscoring", "valuable", "vibrant"}
WIKI_PHRASES = ["align with", "aligns with", "deep dive", "serves as a", "stands as a", "functions as a", "it's important to note", "it is important to note",
                "worth noting", "in conclusion", "in summary", "i hope this helps", "certainly!", "you're absolutely right", "as of my last knowledge update"]
# 2026-era construction frames: the tells that survive in current models (Graphite 2026 cross-generation study; Wikipedia
# "Signs of AI writing"): corrective framing, importance flagging, reframing transitions.
FRAMES = [(r"\bnot\s+(?:just|simply|merely|only)\s+[^.;!?]{1,60}?[,;]?\s+(?:but|it['’]s|it is|rather)\b", "corrective framing (not just X but Y)"),
          (r"\b(?:it['’]s|it is|this is|that is|was)\s+not\s+(?:a|an|the)?\s*[^.;!?]{1,40}?[,;]\s*(?:it['’]s|it is|but)\b", "corrective framing (it's not X, it's Y)"),
          (r"\brather than (?:simply|merely|just)\b", "rather than merely"),
          (r"\b(?:is|was|are)\s+more than\s+(?:a|an|just)\b[^.;!?]{1,40}?[,;]\s*(?:it|they)\b", "is more than a X, it"),
          (r"\bless like\s+(?:a|an)\b[^.;!?]{1,40}?\band more like\b", "less like X and more like Y"),
          (r"\bno\s+\w+,\s*no\s+\w+,\s*(?:just|only)\b", "no X, no Y, just Z"),
          (r"\b(?:this|that|it)\s+matters\b|\bwhy\s+\w+(?:\s+\w+)?\s+matters\b|\bmatters because\b", "importance flagging (this matters)"),
          (r"\badds another (?:layer|dimension)\b|\banother dimension\b|\blooking ahead\b|\bultimately,? this\b", "reframing transition (adds another layer / looking ahead)"),
          (r"\b(?:the honest answer|the short answer|the real question|here['’]s the thing)\b", "aphoristic signposting")]
FRAMES = [(re.compile(rx, re.I), name) for rx, name in FRAMES]
_PUNCT = ".,;:—()!?\"“”-"

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
    kob = [(s, e, "excess vocabulary (Kobak 2024 style word)") for w, s, e in toks if w in KOBAK]
    juz = [(s, e, "classic 2023-24 AI vocabulary (Juzek & Ward)") for w, s, e in toks if w in JUZEK]
    wik = [(s, e, "Wikipedia signs-of-AI-writing word") for w, s, e in toks if w in WIKI]
    for wp in WIKI_PHRASES:
        for m in re.finditer(r"(?<![a-z])" + re.escape(wp) + r"(?![a-z])", low): wik.append((m.start(), m.end(), "Wikipedia signs-of-AI-writing phrase"))
    fr = []
    for rx, name in FRAMES:
        for m in rx.finditer(text): fr.append((m.start(), m.end(), "2026 frame: " + name))
    # punctuation diversity: Shannon entropy over punctuation marks (post-2026 models tend to monotone punctuation)
    pc = {}
    for ch in text:
        if ch in _PUNCT: pc[ch] = pc.get(ch, 0) + 1
    tot = sum(pc.values()); import math as _m
    pent = -sum(c / tot * _m.log2(c / tot) for c in pc.values()) if tot else 0.0
    per_k = lambda k: k / n * 1000
    return {"words": len(toks), "kobak_per_k": per_k(len(kob)), "juzek_per_k": per_k(len(juz)), "wiki_per_k": per_k(len(wik)), "frames_2026_per_k": per_k(len(fr)),
            "punct_entropy": pent,
            "slop_word_per_k": per_k(len(fw_hits)), "slop_top300_per_k": per_k(len(ftop_hits)), "antislop_word_per_k": per_k(len(aw_hits)),
            "slop_ngram_per_k": per_k(len(bg) + len(tg)), "slop_phrase_per_k": per_k(len(ph)), "slop_construction_per_k": per_k(len(cx)),
            "matches": sorted(set(fw_hits + bg + tg + ph + cx + kob + juz + wik + fr))[:600]}

MEASURES = [  # key, title, higher_is_human, note
    ("frames_2026_per_k", "2026-era construction frames (not just X but Y, this matters...) per 1,000 words", False, "Graphite 2026 cross-generation study and Wikipedia Signs of AI writing: the tells that survive in GPT-5/6, Claude 5 and Gemini 3 output."),
    ("kobak_per_k", "Excess vocabulary, Kobak 2024 style words (per 1,000 words)", False, "Kobak et al. 2025: 379 style words over-used in 2024 PubMed abstracts (delve, underscore, showcasing, crucial...)."),
    ("juzek_per_k", "Classic 2023-24 AI vocabulary, Juzek & Ward (per 1,000 words)", False, "21 focal words (delve, intricate, showcase, realm...). Dated: 2026 frontier models use them 41-86% less."),
    ("wiki_per_k", "Wikipedia signs-of-AI-writing words and phrases (per 1,000 words)", False, "WikiProject AI Cleanup core list, 2025-26 (CC BY-SA)."),
    ("punct_entropy", "Punctuation diversity (entropy over marks)", True, "Low diversity is the post-2026 pattern once em dashes were trained out."),
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
