"""Hidden-character and mixed-script check. Pure Python (unicodedata), no tables to maintain.
Finds invisible/format characters, non-standard spaces, bidi controls, variation selectors, tag characters, and
Latin words containing Cyrillic/Greek lookalikes. These are evidence of copy-paste, watermark carriers or tampering,
not of AI authorship. Usage: python unicode_check.py file.txt [...] -> JSON"""
import sys, json, unicodedata, re
ZERO_WIDTH = {0x200B: "zero width space", 0x200C: "zero width non-joiner", 0x200D: "zero width joiner", 0x2060: "word joiner",
              0xFEFF: "byte order mark / ZWNBSP", 0x00AD: "soft hyphen", 0x180E: "Mongolian vowel separator", 0x061C: "Arabic letter mark"}
BIDI = {0x200E: "LRM", 0x200F: "RLM", 0x202A: "LRE", 0x202B: "RLE", 0x202C: "PDF", 0x202D: "LRO", 0x202E: "RLO",
        0x2066: "LRI", 0x2067: "RLI", 0x2068: "FSI", 0x2069: "PDI"}
ODD_SPACES = {0x00A0: "no-break space", 0x2000: "en quad", 0x2001: "em quad", 0x2002: "en space", 0x2003: "em space", 0x2004: "three-per-em space",
              0x2005: "four-per-em space", 0x2006: "six-per-em space", 0x2007: "figure space", 0x2008: "punctuation space", 0x2009: "thin space",
              0x200A: "hair space", 0x202F: "narrow no-break space", 0x205F: "medium mathematical space", 0x3000: "ideographic space", 0x1680: "Ogham space mark"}
def classify(cp):
    if cp in ZERO_WIDTH: return "zero-width/invisible", ZERO_WIDTH[cp]
    if cp in BIDI: return "bidirectional control", BIDI[cp]
    if cp in ODD_SPACES: return "non-standard space", ODD_SPACES[cp]
    if 0xFE00 <= cp <= 0xFE0F or 0xE0100 <= cp <= 0xE01EF: return "variation selector", f"U+{cp:04X}"
    if 0xE0000 <= cp <= 0xE007F: return "tag character (hidden text carrier)", f"U+{cp:04X}"
    cat = unicodedata.category(chr(cp))
    if cat == "Cf": return "format character", unicodedata.name(chr(cp), f"U+{cp:04X}")
    if cat == "Cc" and cp not in (9, 10, 13): return "control character", f"U+{cp:04X}"
    if cat == "Co": return "private-use character", f"U+{cp:04X}"
    return None
def script_of(ch):
    try: n = unicodedata.name(ch)
    except ValueError: return None
    for s in ("LATIN", "CYRILLIC", "GREEK", "ARMENIAN"):
        if n.startswith(s): return s
    return None
def check(text):
    hidden = []
    for i, ch in enumerate(text):
        c = classify(ord(ch))
        if c: hidden.append({"pos": i, "cp": f"U+{ord(ch):04X}", "kind": c[0], "name": c[1]})
    mixed = []
    for m in re.finditer(r"[^\W\d_]+", text):
        w = m.group(); scripts = {s for s in (script_of(ch) for ch in w) if s}
        if "LATIN" in scripts and len(scripts) > 1:
            mixed.append({"pos": m.start(), "word": w, "scripts": sorted(scripts),
                          "foreign": [f"{ch} U+{ord(ch):04X} {unicodedata.name(ch, '')}" for ch in w if script_of(ch) not in (None, "LATIN")]})
    by_kind = {}
    for h in hidden: by_kind[h["kind"]] = by_kind.get(h["kind"], 0) + 1
    return {"chars": len(text), "hidden_total": len(hidden), "hidden_by_kind": by_kind, "hidden": hidden[:200], "mixed_script_words": mixed[:200],
            "status": "clean" if not hidden and not mixed else "attention"}
if __name__ == "__main__":
    print(json.dumps({f: check(open(f, encoding="utf-8").read()) for f in sys.argv[1:]}, indent=1, ensure_ascii=False))
