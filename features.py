"""Model-free stylometric measures: lexical diversity, sentence- and paragraph-length distribution, punctuation and
construction rates. Pure Python; every value is a scalar that the report plots against the control texts.
Usage: python features.py file.txt [...] -> JSON"""
import sys, json, re, math, statistics

_WORD = re.compile(r"[A-Za-z][A-Za-z'’-]*")
_SENT = re.compile(r"(?<=[.!?])\s+")
_CONTRACTION = re.compile(r"\b\w+(?:n['’]t|['’](?:re|ve|ll|d|s|m))\b", re.I)
_FIRST = re.compile(r"\b(?:I|me|my|mine|myself|we|us|our|ours)\b")
_NOT_BUT = re.compile(r"\bnot\b[^.;:!?]{1,60}?,?\s+but\b", re.I)      # "not X, but Y" (negation-contrast template)
_TRICOLON = re.compile(r"\b\w+(?:\s\w+){0,2},\s\w+(?:\s\w+){0,2},\s(?:and|or)\s\w+", re.I)   # "X, Y, and Z"
_RHET_Q = re.compile(r"[^.!?]*\?")
_LONG_WORD = 7

def words(t): return _WORD.findall(t)
def sentences(t): return [s.strip() for s in _SENT.split(t.strip()) if s.strip()]
def paragraphs(t): return [p.strip() for p in re.split(r"\n\s*\n", t.strip()) if p.strip()]

def mattr(ws, window=100):
    """Moving-average type-token ratio (Covington & McFall 2010): TTR without the text-length bias."""
    ws = [w.lower() for w in ws]
    if len(ws) <= window: return len(set(ws)) / max(len(ws), 1)
    return statistics.fmean(len(set(ws[i:i + window])) / window for i in range(len(ws) - window + 1))

def mtld(ws, thr=0.72):
    """Measure of Textual Lexical Diversity (McCarthy & Jarvis 2010): mean factor length at TTR threshold, both directions."""
    ws = [w.lower() for w in ws]
    if len(ws) < 10: return float("nan")
    def one(seq):
        factors, types, n = 0.0, set(), 0
        for w in seq:
            types.add(w); n += 1
            if len(types) / n <= thr: factors += 1; types, n = set(), 0
        if n: factors += (1 - len(types) / n) / (1 - thr)
        return len(seq) / factors if factors else float("nan")
    return (one(ws) + one(ws[::-1])) / 2

def cv(xs):
    xs = [x for x in xs if x == x]
    return (statistics.pstdev(xs) / statistics.fmean(xs)) if len(xs) > 1 and statistics.fmean(xs) else float("nan")

def per_k(count, n, k=1000): return count / n * k if n else float("nan")

def analyze(t):
    ws, ss, ps = words(t), sentences(t), paragraphs(t)
    nw = len(ws) or 1
    slen = [len(words(s)) for s in ss] or [0]
    plen = [len(words(p)) for p in ps] or [0]
    chars = len(t)
    low = [w.lower() for w in ws]
    hapax = sum(1 for w in set(low) if low.count(w) == 1) / nw
    return {
        "words": len(ws), "sentences": len(ss), "paragraphs": len(ps),
        "sent_len_mean": statistics.fmean(slen), "sent_len_sd": statistics.pstdev(slen) if len(slen) > 1 else 0.0, "sent_len_cv": cv(slen),
        "sent_len_max": max(slen), "sent_len_min": min(slen),
        "para_len_cv": cv(plen) if len(plen) > 1 else float("nan"),
        "mattr": mattr(ws), "mtld": mtld(ws), "hapax_ratio": hapax,
        "word_len_mean": statistics.fmean(len(w) for w in ws) if ws else float("nan"),
        "long_word_ratio": sum(1 for w in ws if len(w) >= _LONG_WORD) / nw,
        "em_dash_per_k": per_k(t.count("—") + t.count("--"), nw), "en_dash_per_k": per_k(t.count("–"), nw),
        "semicolon_per_k": per_k(t.count(";"), nw), "colon_per_k": per_k(len(re.findall(r":(?!\d)", t)), nw),
        "exclaim_per_k": per_k(t.count("!"), nw), "question_per_k": per_k(t.count("?"), nw),
        "paren_per_k": per_k(t.count("("), nw), "comma_per_sent": t.count(",") / max(len(ss), 1),
        "contraction_per_k": per_k(len(_CONTRACTION.findall(t)), nw), "first_person_per_k": per_k(len(_FIRST.findall(t)), nw),
        "not_but_per_k": per_k(len(_NOT_BUT.findall(t)), nw), "tricolon_per_k": per_k(len(_TRICOLON.findall(t)), nw),
        "digit_ratio": sum(ch.isdigit() for ch in t) / max(chars, 1),
        "uppercase_ratio": sum(ch.isupper() for ch in t) / max(chars, 1),
        "sent_start_repeat": _start_repeat(ss),
    }

def _start_repeat(ss):
    """Share of sentences that open with the same word as the previous one (anaphora or lazy repetition)."""
    starts = [(words(s) or [""])[0].lower() for s in ss]
    return sum(1 for a, b in zip(starts, starts[1:]) if a and a == b) / max(len(starts) - 1, 1)

# What each measure means and which direction is typically more machine-like, for the report. "higher_is_human" None = no claim.
MEASURES = [
    ("mattr", "Lexical diversity, MATTR (100-word window)", True, "Moving-average type-token ratio. Generated text often reuses a narrower vocabulary."),
    ("mtld", "Lexical diversity, MTLD", True, "Mean run length before type-token ratio falls to 0.72. Less length-sensitive than raw TTR."),
    ("hapax_ratio", "Hapax ratio (words used once / words)", True, "Share of vocabulary used exactly once."),
    ("sent_len_cv", "Sentence-length variation (coefficient of variation)", True, "Standard deviation of sentence length over its mean. Flat rhythm is a common generation tell."),
    ("para_len_cv", "Paragraph-length variation (coefficient of variation)", True, "Uniform paragraph sizes are typical of templated generation."),
    ("sent_len_mean", "Mean sentence length (words)", None, ""),
    ("word_len_mean", "Mean word length (letters)", False, "Longer average words go with the formal register generators default to."),
    ("long_word_ratio", "Long words (7+ letters) share", False, ""),
    ("em_dash_per_k", "Em dashes per 1,000 words", False, "Heavy em-dash use is a widely noted 2024-2026 chatbot signature."),
    ("semicolon_per_k", "Semicolons per 1,000 words", None, ""),
    ("colon_per_k", "Colons per 1,000 words", False, "Colon-led explanations and list introductions."),
    ("exclaim_per_k", "Exclamation marks per 1,000 words", None, ""),
    ("question_per_k", "Question marks per 1,000 words", None, "Rhetorical questions are a known template move."),
    ("contraction_per_k", "Contractions per 1,000 words", True, "Assistants default to a formal, uncontracted register unless asked otherwise."),
    ("first_person_per_k", "First-person pronouns per 1,000 words", None, "Register-dependent."),
    ("not_but_per_k", "\"Not X, but Y\" constructions per 1,000 words", False, "Negation-contrast template."),
    ("tricolon_per_k", "Three-item lists (X, Y, and Z) per 1,000 words", False, "Rule-of-three lists."),
    ("comma_per_sent", "Commas per sentence", None, ""),
    ("sent_start_repeat", "Consecutive sentences sharing an opening word", None, ""),
]

if __name__ == "__main__":
    print(json.dumps({f: analyze(open(f, encoding="utf-8").read()) for f in sys.argv[1:]}, indent=1))
