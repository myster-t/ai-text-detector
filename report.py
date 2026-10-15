"""ai-text-detector: Binoculars, Fast-DetectGPT, token statistics (GLTR, log-rank), perplexity/burstiness, stylometry,
writing-pattern (AI-ism) and hidden-character checks; one HTML report per file plus an index.
Usage:  ./detect file1.txt [file2 ...] [--out DIR] [--older-models | --profile NAME] [--selftest]
Models are loaded for the run and released when it exits; nothing stays resident."""
import os, sys, re, math, json, html, argparse, datetime, pathlib
HERE = pathlib.Path(__file__).resolve().parent
_md = os.environ.get("AITD_MODELS_DIR") or ((HERE / ".models-dir").read_text().strip() if (HERE / ".models-dir").is_file() else "") or str(HERE / "models")
if pathlib.Path(_md).is_dir(): os.environ.setdefault("HF_HOME", _md)
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1"); os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, str(HERE)); import style as style_mod, unicode_check, features, lmfeat, charts
from charts import esc

PROFILES = json.loads((HERE / "profiles.json").read_text())
ap = argparse.ArgumentParser(description="Score text files for machine generation and write HTML reports.")
ap.add_argument("files", nargs="*", help="UTF-8 text files to score")
ap.add_argument("--out", default="reports", help="output folder (default: ./reports)")
ap.add_argument("--profile", default=None, choices=sorted(PROFILES), help="model set by name (default: latest)")
ap.add_argument("--older-models", action="store_true", help="use the original 2023 Falcon-7B / Qwen2.5 set with the Binoculars paper's thresholds (same as --profile original)")
ap.add_argument("--selftest", action="store_true", help="score the bundled controls and check they separate")
args = ap.parse_args()
if args.older_models and args.profile and args.profile != "original": ap.error("--older-models means --profile original; drop one of them")
args.profile = "original" if args.older_models else (args.profile or "latest")
P = PROFILES[args.profile]; CAL = bool(P.get("calibrated"))
T_ACC, T_FPR = P.get("threshold_accuracy"), P.get("threshold_low_fpr")
dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

HUMAN = sorted((HERE / "controls" / "human").glob("*.txt")); AI = sorted((HERE / "controls" / "ai").glob("*.txt")); CONTROLS = HUMAN + AI
ANCHOR_H, ANCHOR_A = HERE / "controls" / "human" / "stewart-letter-1913.txt", HERE / "controls" / "ai" / "claude-marketing-paragraph.txt"
if args.selftest: args.files = [str(ANCHOR_H), str(ANCHOR_A)]; args.out = str(HERE / "reports" / f"selftest-{args.profile}")
if not args.files: ap.error("give at least one text file, or --selftest")
out = pathlib.Path(args.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
targets = [pathlib.Path(f).expanduser().resolve() for f in args.files]
for p in targets:
    if not p.is_file(): sys.exit(f"not a file: {p}")
texts = {p: open(p, encoding="utf-8").read().strip() for p in targets + [c.resolve() for c in CONTROLS]}
is_human = {c.resolve(): True for c in HUMAN}; is_human.update({c.resolve(): False for c in AI})
HC, MC = [c.resolve() for c in HUMAN], [c.resolve() for c in AI]

def load(mid, dtype):
    tok = AutoTokenizer.from_pretrained(mid)
    try: m = AutoModelForCausalLM.from_pretrained(mid, dtype=dtype)
    except Exception:
        from transformers import AutoModelForImageTextToText
        m = AutoModelForImageTextToText.from_pretrained(mid, dtype=dtype)
    return tok, m.to(dev).eval()
def release(*objs):
    for o in objs: del o
    if dev == "mps": torch.mps.empty_cache()
    elif dev == "cuda": torch.cuda.empty_cache()
def sentences(t): return [s for s in re.split(r"(?<=[.!?])\s+", t) if len(s.split()) >= 4]
def log(msg): print(msg, file=sys.stderr, flush=True)
NAN = float("nan")

# ---- perplexity models: whole-document token statistics for every model, sentence-level for the first ----
lm = {}; sent_ppl = {}
for i, (mid, label) in enumerate(P["perplexity_models"]):
    log(f"[{label}] loading"); tok, m = load(mid, torch.float32 if "gpt2" in mid else torch.bfloat16)
    for p, t in texts.items():
        r = lmfeat.token_stats(m, tok, t, dev) or {"n": 0, "tokens": [], "ppl": NAN, "mean_logrank": NAN, "lrr": NAN, "mean_entropy": NAN, "top1": NAN, "top10": NAN, "top100": NAN, "top1000": NAN, "lp_sd": NAN, "lp_volatility": NAN, "lp_autocorr1": NAN, "late_stability": NAN}
        if i: r["tokens"] = []
        r["shuffle_ratio"] = lmfeat.shuffle_ratio(m, tok, t, dev) if i == 0 else float("nan")
        lm[(p, label)] = r
        if i == 0:
            sp = []
            for s in sentences(t):
                rs = lmfeat.token_stats(m, tok, s, dev); sp.append((s, rs["ppl"] if rs else NAN))
            vals_ = [v for _, v in sp if v == v] or [NAN]; mean = sum(vals_) / len(vals_)
            sent_ppl[p] = {"sent": sp, "mean": mean, "sd": (sum((v - mean) ** 2 for v in vals_) / len(vals_)) ** 0.5}
    release(tok, m)
PPL_LABELS = [l for _, l in P["perplexity_models"]]; L0 = PPL_LABELS[0]

# ---- binoculars + Fast-DetectGPT (analytic) from the same two forward passes ----
log("[binoculars] loading observer + performer"); otok, obs = load(P["observer"], torch.bfloat16); ptok, perf = load(P["performer"], torch.bfloat16)
def bino(t):
    enc = otok(t, return_tensors="pt", truncation=True, max_length=512).to(dev); ids = enc.input_ids
    with torch.no_grad(): lo = obs(input_ids=ids).logits[:, :-1].float(); lp = perf(input_ids=ids).logits[:, :-1].float()
    v = min(lo.shape[-1], lp.shape[-1]); lo, lp = lo[..., :v], lp[..., :v]; lab = ids[:, 1:]
    logq, logp = F.log_softmax(lo, -1), F.log_softmax(lp, -1); q = logq.exp()
    lppl = F.cross_entropy(lp.transpose(1, 2), lab).item(); xppl = (-(q * logp).sum(-1)).mean().item()
    # Fast-DetectGPT (Bao et al. 2024), analytic form: sampling distribution q = observer, scoring model p = performer.
    lp_tok = logp.gather(-1, lab[..., None])[..., 0]; mu = (q * logp).sum(-1); var = (q * logp ** 2).sum(-1) - mu ** 2
    fd = ((lp_tok.sum() - mu.sum()) / var.sum().clamp_min(1e-9).sqrt()).item()
    # Alignment imprint (after Wu et al. 2026, LAPD, unweighted form): how much more the aligned performer likes each token
    # than its base observer does. Positive = the text follows the aligned model's preferences = more machine-like.
    lq_tok = logq.gather(-1, lab[..., None])[..., 0]; imprint = (lp_tok - lq_tok).mean().item()
    return lppl / xppl, lab.shape[1], fd, imprint
bino_res = {p: bino(t) for p, t in texts.items()}
def _bt(p):
    """Fast-DetectGPT and imprint read from the tuple; keeps the older 3-tuple shape readable."""
    return bino_res[p]
release(otok, obs, ptok, perf)

# ---- deterministic checks ----
style_res = {p: style_mod.analyze(t) for p, t in texts.items()}
uni_res = {p: unicode_check.check(t) for p, t in texts.items()}
feat_res = {p: features.analyze(t) for p, t in texts.items()}

# ---- measure registry: (group, key, title, higher_is_human, note, getter, log) ----
MEAS = [("Model-based signals", "binoculars", "Binoculars score", True, "performer perplexity / observer-performer cross-perplexity", lambda p: bino_res[p][0], False),
        ("Model-based signals", "fastdetectgpt", "Fast-DetectGPT (analytic)", False, "probability curvature; sampling = observer, scoring = performer", lambda p: bino_res[p][2], False),
        ("Model-based signals", "imprint", "Alignment imprint (performer − observer log-likelihood)", False, "after LAPD, Wu et al. 2026; unweighted", lambda p: bino_res[p][3], False),
        ("Model-based signals", f"ppl_{L0}", f"Perplexity, {L0}", True, "log scale", lambda p: lm[(p, L0)]["ppl"], True),
        ("Model-based signals", "burst", f"Burstiness, {L0}", True, "sd of sentence perplexity, log scale", lambda p: sent_ppl[p]["sd"], True)]
for l in PPL_LABELS[1:]: MEAS.append(("Model-based signals", f"ppl_{l}", f"Perplexity, {l}", True, "log scale", (lambda l: lambda p: lm[(p, l)]["ppl"])(l), True))
for k, title, hih, note in lmfeat.MEASURES: MEAS.append(("Token statistics", k, title, hih, note, (lambda k: lambda p: lm[(p, L0)][k])(k), False))
MEAS.append(("Writing patterns", "aiisms", "AI-ism score (avoid-ai-writing, 0-100)", False, "deterministic vocabulary and template rules", lambda p: style_res[p].get("score"), False))
STYLE_KEYS = ("mattr", "mtld", "hapax_ratio", "sent_len_cv", "para_len_cv", "sent_len_mean", "word_len_mean", "long_word_ratio", "sent_start_repeat")
for k, title, hih, note in features.MEASURES:
    MEAS.append(("Style and rhythm" if k in STYLE_KEYS else "Punctuation and constructions", k, title, hih, note, (lambda k: lambda p: feat_res[p][k])(k), False))
vals = {(p, k): g(p) for p in texts for _, k, _, _, _, g, _ in MEAS}
def hv(k): return [vals[(p, k)] for p in HC]
def mv(k): return [vals[(p, k)] for p in MC]
LOGK = {k for _, k, _, _, _, _, lg in MEAS if lg}
def cutpoint(k):
    """The value that best separates the two control groups for this measure in its claimed direction: the midpoint
    between adjacent sorted control values that maximises balanced accuracy; ties go to the cut nearest the midpoint of
    the two group medians. Computed in log space for log-scale measures. None when no direction is claimed."""
    hih = next(h for _, kk, _, h, _, _, _ in MEAS if kk == k)
    f = (lambda x: math.log10(max(x, 1e-9))) if k in LOGK else (lambda x: x)
    h = sorted(f(x) for x in hv(k) if x == x); m = sorted(f(x) for x in mv(k) if x == x)
    if hih is None or not h or not m: return None
    allv = sorted(h + m); cands = [(a + b) / 2 for a, b in zip(allv, allv[1:]) if b > a]
    if not cands: return None
    med = (h[len(h) // 2] + m[len(m) // 2]) / 2
    def bal(c):
        th = sum(1 for x in h if (x >= c) == hih) / len(h); tm = sum(1 for x in m if (x >= c) != hih) / len(m)
        return (th + tm) / 2
    best = max(cands, key=lambda c: (round(bal(c), 6), -abs(c - med)))
    return 10 ** best if k in LOGK else best
def side_text(val, k, hih):
    c = cutpoint(k)
    if c is None or val != val: return ""
    return "human" if (val >= c) == hih else "machine"
def separation(k):
    """AUC of this measure alone on the controls: P(human control is on the human side of a machine control)."""
    hih = next(h for _, kk, _, h, _, _, _ in MEAS if kk == k)
    h = [x for x in hv(k) if x == x]; m = [x for x in mv(k) if x == x]
    if not h or not m: return NAN
    auc = sum((1.0 if a > b else 0.5 if a == b else 0.0) for a in h for b in m) / (len(h) * len(m))
    return auc if hih is None or hih else 1 - auc

# ---- html pieces ----
CSS = """:root{--bg:#fbfaf6;--panel:#ffffff;--ink:#1b1b1b;--mut:#5c5b56;--rule:#d8d5cb;--th:#efede6;--link:#0b57d0;--hum:#1f8a4c;--mach:#d2452f;--tgt:#111111;--ok:#1f8a4c;--bad:#c0392b;--warn:#b7791f;
--g0:#1f8a4c;--g1:#7cc47f;--g2:#f2c94c;--g3:#f08a4b;--g4:#a65cd1;--mark:#ffe08a;--mark-hi:#ffb3a7;--sent:#fff3c2;--hid:#e03b3b}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#121311;--panel:#1c1d1a;--ink:#ecebe4;--mut:#aaa89f;--rule:#3b3c36;--th:#262822;--link:#8ec5ff;--hum:#5fd38a;--mach:#ff8a73;--tgt:#ffffff;--ok:#5fd38a;--bad:#ff8a73;--warn:#f0c060;
--g0:#3fbf74;--g1:#8fd58f;--g2:#e8c84a;--g3:#f09a60;--g4:#c08ae0;--mark:#5a4a12;--mark-hi:#6b2a22;--sent:#3a3514;--hid:#ff6b6b}}
:root[data-theme=dark]{--bg:#121311;--panel:#1c1d1a;--ink:#ecebe4;--mut:#aaa89f;--rule:#3b3c36;--th:#262822;--link:#8ec5ff;--hum:#5fd38a;--mach:#ff8a73;--tgt:#ffffff;--ok:#5fd38a;--bad:#ff8a73;--warn:#f0c060;
--g0:#3fbf74;--g1:#8fd58f;--g2:#e8c84a;--g3:#f09a60;--g4:#c08ae0;--mark:#5a4a12;--mark-hi:#6b2a22;--sent:#3a3514;--hid:#ff6b6b}
html,body{background:var(--bg)}body{margin:0;padding:32px 16px 72px;color:var(--ink);font:15.5px/1.55 -apple-system,"Segoe UI",Helvetica,Arial,sans-serif}main{max-width:1040px;margin:0 auto}
a{color:var(--link);text-decoration:underline;text-underline-offset:2px}a:visited{color:var(--link)}
h1{font-size:1.75rem;margin:0 0 4px;text-wrap:balance}h2{font-size:1.25rem;margin:38px 0 10px;padding-top:14px;border-top:2px solid var(--rule)}h3{font-size:1rem;margin:22px 0 6px}p{max-width:78ch;margin:0 0 10px}
.by{color:var(--mut);font-size:.88rem;margin-bottom:20px}.mut{color:var(--mut)}.ok{color:var(--ok)}.bad{color:var(--bad)}.warn{color:var(--warn)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px;margin:14px 0 6px}
.tile{background:var(--panel);border:1px solid var(--rule);border-radius:10px;padding:12px 14px}.tile .k{font-size:.78rem;color:var(--mut);text-transform:uppercase;letter-spacing:.04em}
.tile .v{font-size:1.6rem;font-weight:700;line-height:1.2;margin:2px 0;font-variant-numeric:tabular-nums}.tile .s{font-size:.84rem;margin-top:2px}.tile .side{font-weight:600}.posbar{width:100%;height:22px;display:block;margin-top:6px}
.verdict{background:var(--panel);border:1px solid var(--rule);border-left:6px solid var(--link);border-radius:10px;padding:14px 18px;margin:0 0 14px}.verdict b{font-size:1.3rem}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:10px}.violin{width:100%;height:auto;display:block;background:var(--panel);border:2px solid var(--rule);border-radius:8px}
.violin.vh{border-color:var(--hum)}.violin.vm{border-color:var(--mach);box-shadow:0 0 0 3px color-mix(in srgb,var(--mach) 25%,transparent)}
.violin.weak{filter:saturate(.15);border-style:dashed;border-color:var(--mut)}
.pkey{display:flex;flex-wrap:wrap;gap:10px 18px;margin:8px 0 14px;font-size:.86rem}.pkey span{display:inline-flex;align-items:center;gap:7px}.kb{display:inline-block;width:26px;height:16px;border-radius:4px;border:2px solid var(--rule);background:var(--panel)}
.kb.h{border-color:var(--hum)}.kb.m{border-color:var(--mach);box-shadow:0 0 0 3px color-mix(in srgb,var(--mach) 25%,transparent)}.kb.w{border-style:dashed;border-color:var(--mut);filter:saturate(.15)}
table{border-collapse:collapse;width:100%;font-size:.86rem;line-height:1.4}th,td{border:1px solid var(--rule);padding:6px 8px;vertical-align:top;text-align:left}th{background:var(--th)}td.n,th.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.wrap{overflow-x:auto;margin:10px 0 16px}.bar{display:inline-block;height:9px;background:var(--link);vertical-align:middle;border-radius:2px}.tgt{font-weight:600}
.strip{width:100%;max-width:1000px;height:auto;display:block;margin:4px 0 14px}.legend{font-size:.84rem;color:var(--mut);max-width:none}.sw{display:inline-block;width:10px;height:10px;border-radius:50%;vertical-align:middle;margin-right:4px}
.txt{background:var(--panel);border:1px solid var(--rule);border-radius:10px;padding:16px 18px;font:15px/1.75 Georgia,"Iowan Old Style",serif;max-width:none;word-wrap:break-word}
.tk{border-radius:2px;padding:0 1px}.g0{background:var(--g0);color:#fff}.g1{background:var(--g1);color:#111}.g2{background:var(--g2);color:#111}.g3{background:var(--g3);color:#111}.g4{background:var(--g4);color:#fff}
.key{display:inline-block;padding:1px 7px;border-radius:4px;margin-right:6px;font-size:.82rem}
mark{background:var(--mark);color:inherit;border-radius:3px;padding:0 2px;border-bottom:2px solid var(--warn)}mark.hi{background:var(--mark-hi);border-bottom-color:var(--bad)}mark.lo{border-bottom-style:dotted}
.sent{background:var(--sent)}.hid{display:inline-block;background:var(--hid);color:#fff;font:700 .72rem/1.3 -apple-system,Helvetica,Arial,sans-serif;padding:0 4px;border-radius:3px;margin:0 1px;vertical-align:baseline;cursor:help}
.mix{outline:2px solid var(--bad);outline-offset:1px;border-radius:2px}
@media print{body{padding:0}h2{break-after:avoid}tr,.tile,.violin{break-inside:avoid}}"""
def fmt(v, d=1): return f"{v:,.{d}f}" if isinstance(v, (int, float)) and v == v and abs(v) != float("inf") else "n/a"
def fmtk(k, v):
    if not isinstance(v, (int, float)) or v != v: return "n/a"
    if k in ("binoculars", "lrr", "mattr", "hapax_ratio", "long_word_ratio", "digit_ratio", "sent_start_repeat", "top1", "top10", "top100", "top1000"): return f"{v:.3f}"
    if k in ("fastdetectgpt", "mean_logrank", "mean_entropy", "sent_len_cv", "para_len_cv", "comma_per_sent"): return f"{v:.2f}"
    if k.startswith("ppl") or k in ("burst", "mtld"): return f"{v:.1f}"
    return f"{v:.3g}"
legend = '<p class=legend><span class=sw style="background:var(--hum)"></span>human controls &nbsp; <span class=sw style="background:var(--mach)"></span>machine controls &nbsp; <span style="display:inline-block;width:9px;height:9px;background:var(--tgt);transform:rotate(45deg);margin-right:4px"></span>this file</p>'
now = datetime.date.today().isoformat()
model_line = f"Binoculars pair {P['observer']} / {P['performer']}; perplexity {', '.join(m for m, _ in P['perplexity_models'])}"
def verdict_of(b):
    if not CAL: return "no verdict (uncalibrated model set)"
    return "human" if b >= T_ACC else ("AI-like (accuracy line)" if b >= T_FPR else "AI-like (strict line)")
def cls_b(b): return "" if not CAL else ("ok" if b >= T_ACC else "bad")

def annotate(text, st, un):
    """The text with AI-ism matches, flagged sentence spans, hidden characters and mixed-script words marked inline."""
    n = len(text); marks = []   # (start, end, cls, title)
    for i in st.get("issues", []):
        word = str(i.get("text") or ""); ttl = f"{i.get('typeLabel', i.get('type'))} ({i.get('severity')})" + (f": {i.get('suggestion')}" if i.get("suggestion") else "")
        cls = "hi" if i.get("severity") == "high" else ("" if i.get("severity") == "medium" else "lo")
        if isinstance(i.get("index"), int) and 0 <= i["index"] < n and text[i["index"]:i["index"] + len(word)].lower() == word.lower():
            marks.append((i["index"], i["index"] + len(word), cls, ttl))
        elif word and len(word) <= 40 and word.strip() == word and not word.startswith("Sentence-rhythm"):
            for m in re.finditer(r"(?<![A-Za-z])" + re.escape(word) + r"(?![A-Za-z])", text, re.I): marks.append((m.start(), m.end(), cls, ttl))
    hid = {h["pos"]: h for h in un.get("hidden", [])}
    mixw = [(m["pos"], m["pos"] + len(m["word"]), f"mixed scripts: {', '.join(m['scripts'])}") for m in un.get("mixed_script_words", [])]
    sent_spans = [(s["start"], s["end"]) for s in st.get("highlight_sentence_for_ai", []) if isinstance(s.get("start"), int)]
    in_sent = [False] * (n + 1)
    for s, e in sent_spans:
        for j in range(max(0, s), min(n, e)): in_sent[j] = True
    mark_at = [None] * (n + 1)
    for s, e, cls, ttl in sorted(marks, key=lambda x: (x[0], -(x[1] - x[0]))):
        if any(mark_at[j] for j in range(s, min(e, n))): continue
        for j in range(s, min(e, n)): mark_at[j] = (s, e, cls, ttl)
    mix_at = [None] * (n + 1)
    for s, e, ttl in mixw:
        for j in range(s, min(e, n)): mix_at[j] = (s, e, ttl)
    out, j = [], 0
    while j < n:
        if j in hid:
            h = hid[j]; out.append(f'<span class=hid title="{esc(h["kind"])}: {esc(h["name"])} ({h["cp"]}) at position {j}">{esc(h["cp"])}</span>'); j += 1; continue
        if mark_at[j] and mark_at[j][0] == j:
            s, e, cls, ttl = mark_at[j]; out.append(f'<mark class="{cls}" title="{esc(ttl)}">{esc(text[s:e])}</mark>'); j = e; continue
        if mix_at[j] and mix_at[j][0] == j:
            s, e, ttl = mix_at[j]; out.append(f'<span class=mix title="{esc(ttl)}">{esc(text[s:e])}</span>'); j = e; continue
        k = j + 1
        while k < n and k not in hid and not (mark_at[k] and mark_at[k][0] == k) and not (mix_at[k] and mix_at[k][0] == k) and in_sent[k] == in_sent[j]: k += 1
        seg = esc(text[j:k]).replace("\n", "<br>")
        out.append(f'<span class=sent title="sentence span flagged by the pattern detector">{seg}</span>' if in_sent[j] else seg); j = k
    return "".join(out), len(set((s, e) for s, e, _, _ in marks))

summary_rows = []
for p in targets:
    t = texts[p]; b, ntok, fd, imp = bino_res[p]; st, un, ft = style_res[p], uni_res[p], feat_res[p]; q = sent_ppl[p]; L = lm[(p, L0)]
    verdict = verdict_of(b)
    summary_rows.append((p.name, len(t.split()), b, verdict, [lm[(p, l)]["ppl"] for l in PPL_LABELS], q["sd"], st.get("score"), st.get("label"), un["status"], un["hidden_total"], len(un["mixed_script_words"]), fd))
    if CAL:
        box = f"<p><b class={cls_b(b)}>Binoculars {b:.3f}: {esc(verdict)}</b></p><p>Scores below {T_ACC:.4f} read as machine-generated at the paper's accuracy-optimal setting; below {T_FPR:.4f} at the low-false-positive setting. Higher is more human-like. This file is {abs(b-T_ACC):.3f} {'above' if b>=T_ACC else 'below'} the accuracy line.</p>"
        tlines = [(T_FPR, "strict"), (T_ACC, "accuracy")]
    else:
        box = f"<p><b>Binoculars {b:.3f}</b> under an uncalibrated model set, so no verdict is given; read the position against the controls.</p>"; tlines = []
    sides = [(k, side_text(vals[(p, k)], k, hih)) for _, k, _, hih, _, _, _ in MEAS if hih is not None]
    nh, nm = sum(1 for _, s in sides if s == "human"), sum(1 for _, s in sides if s == "machine")
    strong = [(k, s) for k, s in sides if separation(k) == separation(k) and separation(k) >= 0.85]
    sh, sm = sum(1 for _, s in strong if s == "human"), sum(1 for _, s in strong if s == "machine")
    box += f"<p>Across the {len(sides)} directional measures below, this file lands on the human side of the controls' best cut-point for <b>{nh}</b> and on the machine side for <b>{nm}</b>. Counting only the {len(strong)} measures that separate the controls well under this profile (AUC ≥ 0.85): human side <b>{sh}</b>, machine side <b>{sm}</b>. Model-based signals carry most of the weight; stylometric measures describe register and rhythm and are weaker evidence on their own.</p>"
    def tile(k, label, v, hih, sub=""):
        s = side_text(v, k, hih); scls = "ok" if s == "human" else ("bad" if s == "machine" else "")
        return f'<div class=tile><div class=k>{esc(label)}</div><div class=v>{fmtk(k, v)}</div><div class=s>{sub}{(" · <span class=\"side " + scls + "\">" + s + " side</span>") if s else ""}</div>{charts.position_bar(hv(k), mv(k), v, hih)}</div>'
    tiles = tile("binoculars", "Binoculars", b, True, esc(verdict) if CAL else "no threshold") + tile("fastdetectgpt", "Fast-DetectGPT", fd, False, "curvature; lower = more human") + tile(f"ppl_{L0}", f"Perplexity ({L0})", L["ppl"], True, "higher = less predictable") + tile("top10", "GLTR top-10 share", L["top10"], False, "share of tokens the model expected") + tile("aiisms", "AI-isms", st.get("score"), False, esc(st.get("label", ""))) + tile("sent_len_cv", "Sentence-length variation", ft["sent_len_cv"], True, "coefficient of variation")
    tiles += f'<div class=tile><div class=k>Hidden characters</div><div class="v {"ok" if un["status"] == "clean" else "warn"}">{un["hidden_total"] + len(un["mixed_script_words"])}</div><div class=s>{esc(un["status"])}{(": " + esc(", ".join(f"{v} {k}" for k, v in un["hidden_by_kind"].items()))) if un["hidden_by_kind"] else ""}{(" · " + str(len(un["mixed_script_words"])) + " mixed-script words") if un["mixed_script_words"] else ""}</div></div>'
    groups = {}
    for grp, k, title, hih, note, g, lg in MEAS: groups.setdefault(grp, []).append((k, title, hih, note, lg))
    vio = ""
    for grp, items in groups.items():
        vio += f"<h3>{esc(grp)}</h3><div class=grid>"
        items = sorted(items, key=lambda it: -(separation(it[0]) if separation(it[0]) == separation(it[0]) else 0))
        for k, title, hih, note, lg in items:
            auc = separation(k); sep = f"AUC {auc:.2f}" if auc == auc else ""
            vio += charts.violin(title, hv(k), mv(k), vals[(p, k)], hih, note=(sep + ("; " if note and sep else "") + note), log=lg, auc=auc, side=side_text(vals[(p, k)], k, hih) or None, cut=cutpoint(k))
        vio += "</div>"
    strips = charts.strip("Binoculars score (higher = more human-like)", [(c.name, bino_res[c.resolve()][0], is_human[c.resolve()]) for c in CONTROLS], b, p.name, lines=tlines)
    strips += charts.strip("Fast-DetectGPT (lower = more human-like)", [(c.name, bino_res[c.resolve()][2], is_human[c.resolve()]) for c in CONTROLS], fd, p.name)
    gl = charts.gltr_html(t, L["tokens"]) if L["tokens"] else "<p class=mut>n/a</p>"
    gkey = f'<p class=legend><span class="key g0">top-1 {L["top1"]*100:.0f}%</span><span class="key g1">top-10 {(L["top10"]-L["top1"])*100:.0f}%</span><span class="key g2">top-100 {(L["top100"]-L["top10"])*100:.0f}%</span><span class="key g3">top-1000 {(L["top1000"]-L["top100"])*100:.0f}%</span><span class="key g4">beyond {(1-L["top1000"])*100:.0f}%</span> &nbsp; share in top-10, controls: human {min(hv("top10"))*100:.0f} to {max(hv("top10"))*100:.0f}%, machine {min(mv("top10"))*100:.0f} to {max(mv("top10"))*100:.0f}%</p>'
    ann, nmarks = annotate(t, st, un)
    mx = max((v for _, v in q["sent"] if v == v), default=1) or 1
    sent_rows = "".join(f"<tr><td class=n>{i+1}</td><td>{esc(s)}</td><td class=n>{fmt(v)}</td><td><span class=bar style='width:{int(160*v/mx) if v == v else 0}px'></span></td></tr>" for i, (s, v) in enumerate(q["sent"]))
    issues = st.get("issues", [])
    style_tbl = ('<div class=wrap><table><tr><th>Severity</th><th>Pattern</th><th>Text</th><th class=n>Position</th><th>Suggestion</th></tr>' + ''.join(f"<tr><td class={'bad' if i.get('severity')=='high' else ('warn' if i.get('severity')=='medium' else 'mut')}>{esc(i.get('severity',''))}</td><td>{esc(i.get('typeLabel',''))}</td><td>{esc(str(i.get('text',''))[:120])}</td><td class=n>{i.get('index') if isinstance(i.get('index'), int) else ''}</td><td class=mut>{esc(str(i.get('suggestion') or ''))[:160]}</td></tr>" for i in issues[:80]) + '</table></div>') if issues else '<p class=mut>No patterns flagged.</p>'
    hid_tbl = ('<div class=wrap><table><tr><th class=n>Position</th><th>Code point</th><th>Kind</th><th>Name</th><th>Context</th></tr>' + ''.join(f"<tr><td class=n>{h['pos']}</td><td>{h['cp']}</td><td>{esc(h['kind'])}</td><td>{esc(h['name'])}</td><td class=mut>…{esc(t[max(0,h['pos']-30):h['pos']])}<span class=hid>{h['cp']}</span>{esc(t[h['pos']+1:h['pos']+30])}…</td></tr>" for h in un['hidden'][:80]) + '</table></div>') if un['hidden'] else ''
    mix_tbl = ('<div class=wrap><table><tr><th class=n>Position</th><th>Word</th><th>Scripts</th><th>Non-Latin letters</th></tr>' + ''.join(f"<tr><td class=n>{m['pos']}</td><td>{esc(m['word'])}</td><td>{esc(', '.join(m['scripts']))}</td><td>{esc('; '.join(m['foreign']))}</td></tr>" for m in un['mixed_script_words'][:80]) + '</table></div>') if un['mixed_script_words'] else ''
    mkeys = [(k, title) for _, k, title, _, _, _, _ in MEAS]
    thead = "<tr><th>Text</th><th>Kind</th><th class=n>Words</th>" + "".join(f"<th class=n>{esc(title)}</th>" for _, title in mkeys) + "</tr>"
    rows = []
    for cp, label, hum in [(p, p.name + " (this file)", None)] + [(c.resolve(), c.name, is_human[c.resolve()]) for c in CONTROLS if c.resolve() != p]:
        rows.append(f"<tr{' class=tgt' if cp == p else ''}><td>{esc(label)}</td><td>{'' if hum is None else ('human' if hum else 'machine')}</td><td class=n>{len(texts[cp].split())}</td>" + "".join(f"<td class=n>{fmtk(k, vals[(cp, k)])}</td>" for k, _ in mkeys) + "</tr>")
    doc = f"""<!DOCTYPE html><html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>AI text report: {esc(p.stem)}</title><style>{CSS}</style></head><body><main>
<h1>AI text report</h1><p class=by>{esc(p.name)} · {len(t.split())} words · run {now} · profile <b>{esc(args.profile)}</b>{'' if CAL else ' (uncalibrated)'} · {esc(model_line)} · all processing local · <a href="index-report.html">all files</a></p>
<div class=verdict>{box}</div>
<div class=tiles>{tiles}</div>
<p class=legend>Each tile's bar shows the control ranges (<span class=sw style="background:var(--hum)"></span>human, <span class=sw style="background:var(--mach)"></span>machine) and this file as a diamond.</p>
<h2>The shape of this text against the controls</h2>
<p>Every measure is computed for {len(HUMAN)} human and {len(AI)} machine control texts in the same run with the same models. Each panel shows the two control distributions as violins (points are the individual controls, the bar is the median), the dotted line is the cut-point that best separates the two groups, and the dashed line is this file. <b>Up is always the human direction</b>: panels for measures where lower values are more human-like have their axis reversed. Panels are ordered by "AUC", how well that measure alone separates the two control groups (1.0 = perfectly, 0.5 = not at all).</p>
<div class=pkey><span><i class="kb h"></i> green border: this file is on the <b>human</b> side of the cut</span><span><i class="kb m"></i> red border and halo: this file is on the <b>machine</b> side; skim for these</span><span><i class="kb w"></i> grey, dashed, desaturated: <b>weak measure</b> (AUC under 0.7), it barely separates the controls, so its side is not evidence and is left out of the strong count above</span></div>
{vio}
<h2>Number lines with thresholds</h2>{legend}{strips}
<h2>Token predictability map ({esc(L0)})</h2>
<p>Each token is coloured by where the real next token sat in the model's ranking (GLTR, Gehrmann et al. 2019). Sampled text is mostly green; human writing reaches further into yellow, orange and purple. Hover a token for its rank, log-probability and the model's entropy at that point. First {L["n"]} tokens scored.</p>
{gkey}<div class=txt>{gl}</div>
<h2>Annotated text: AI-isms and hidden characters</h2>
<p><b>AI-ism score {st.get('score')} / 100: {esc(st.get('label'))}</b>{(' · ' + esc(st.get('document_classification')) + ', ' + esc(st.get('confidence_category')) + ' confidence') if st.get('document_classification') else ''}. {nmarks} pattern matches marked: <mark class=hi>red</mark> high severity, <mark>amber</mark> medium, <mark class=lo>dotted</mark> low; <span class=sent>shaded sentences</span> are spans the detector flags as AI-leaning; <span class=hid>U+00A0</span> marks a hidden or non-standard character; <span class=mix>outlined words</span> mix scripts. Hover any mark for the rule and suggestion.</p>
<div class=txt>{ann}</div>
<h3>Pattern list</h3>
<p>Deterministic rules from the <a href="https://github.com/conorbronsdon/avoid-ai-writing">avoid-ai-writing</a> detector (Conor Bronsdon, MIT): vocabulary tiers, transitions, template phrases, chatbot artifacts, generic conclusions, hedging, formatting and rhythm signals. Control range: human {min(style_res[c].get('score', 0) for c in HC)} to {max(style_res[c].get('score', 0) for c in HC)}, machine {min(style_res[c].get('score', 0) for c in MC)} to {max(style_res[c].get('score', 0) for c in MC)}. These describe how the text reads and are useful for editing; they are weak evidence of authorship on their own.</p>{style_tbl}
<h3>Hidden characters and mixed scripts</h3>
<p><b>{esc(un['status'])}</b>: {un['hidden_total']} hidden or non-standard characters, {len(un['mixed_script_words'])} words mixing Latin with another script, in {un['chars']} characters. Zero-width characters, soft hyphens, bidirectional controls, variation selectors, tag characters and lookalike letters can carry watermarks or disguise text, but also arrive through ordinary copy-and-paste. They are evidence about the file, not about who wrote the words.</p>{hid_tbl}{mix_tbl}
<h2>Sentence-level perplexity ({esc(L0)})</h2><p>Each sentence scored on its own. Wide variation between sentences (burstiness) is characteristic of human writing; generated text tends to be uniformly predictable.</p>
<div class=wrap><table><tr><th class=n>#</th><th>Sentence</th><th class=n>Perplexity</th><th></th></tr>{sent_rows}</table></div>
<h2>All measurements, this file and every control</h2><div class=wrap><table>{thead}{''.join(rows)}</table></div>
<p class=mut>Controls are described in controls/README.md: public-domain and Creative Commons human writing from several decades and registers, and machine text from ChatGPT (2023) and Claude (2026).</p>
<h2>Methods and limits</h2>
<p><b>Binoculars</b> (Hans et al., ICML 2024): the text's perplexity under a performer model divided by the cross-perplexity between an observer and the performer, which normalises for prompts and topics that make text predictable for reasons other than machine generation. {'This run uses the paper&#39;s reference models and thresholds.' if CAL else 'This run uses a different model pair than the paper, so the published thresholds do not apply and none are shown; read the position relative to the controls instead.'} Input truncated to 512 tokens ({ntok} scored here).</p>
<p><b>Fast-DetectGPT</b> (Bao et al., ICLR 2024), analytic form: how much more probable the actual tokens are than tokens sampled from the observer's distribution, normalised by the expected spread. Machine text sits at a local maximum of probability (positive curvature); human text does not. Computed from the same two forward passes as Binoculars. <b>Alignment imprint</b> (after Wu et al. 2026, "LAPD"): mean per-token log-likelihood under the aligned performer minus under its base observer; text that follows the aligned model's preferences scores positive. This report uses the unweighted form.</p>
<p><b>Surprisal dynamics</b> (Lastde, Xu et al. 2024; DivEye, Basani &amp; Chen 2026; late-stage stability, Sun et al. 2026): statistics of the per-token log-probability sequence as a time series: spread, local volatility, lag-1 autocorrelation, and second-half versus first-half volatility. <b>Shuffle shift</b> (after La Cava &amp; Tagarelli 2026): the log ratio of perplexity after shuffling the sentences to the original.</p>
<p><b>Token statistics</b> (GLTR, Gehrmann et al. 2019; DetectLLM log-likelihood/log-rank ratio, Su et al. 2023): where each real token sat in the {esc(L0)} model's ranking, and how spread the model's prediction was. <b>Perplexity and burstiness</b>: how predictable the text is and how much that varies sentence to sentence. Weak on short texts, formulaic genres and text dense with names or numbers.</p>
<p><b>Stylometry</b> (lexical diversity: MATTR, Covington &amp; McFall 2010; MTLD, McCarthy &amp; Jarvis 2010; sentence and paragraph rhythm; punctuation and construction rates): model-free descriptions of register. They separate the controls only partly and shift with genre, so read them as context. <b>Writing patterns and hidden characters</b>: deterministic checks that need no model.</p>
<p>No detector proves authorship. Edited or paraphrased machine text can pass, formal human writing can be flagged, and every model-based score depends on the specific models used. Read these numbers alongside drafts, sources and the writer's history.</p>
</main></body></html>"""
    (out / f"{p.stem}-report.html").write_text(doc)

# ---- index ----
cpb = [(c.name, bino_res[c.resolve()][0], is_human[c.resolve()]) for c in CONTROLS]; cpf = [(c.name, bino_res[c.resolve()][2], is_human[c.resolve()]) for c in CONTROLS]
idx_strip = charts.strip("Binoculars score, controls (higher = more human-like)", cpb, NAN, "", lines=([(T_FPR, "strict"), (T_ACC, "accuracy")] if CAL else []))
idx_strip += charts.strip("Fast-DetectGPT, controls (lower = more human-like)", cpf, NAN, "")
tmarks = "".join(f'<li><a href="{esc(n)[:-4]}-report.html"><b>{esc(n)}</b></a>: Binoculars {b:.3f}, Fast-DetectGPT {fd:.2f}, {esc(v)}</li>' for n, _, b, v, _, _, _, _, _, _, _, fd in summary_rows)
hdr = f"<tr><th>File</th><th class=n>Words</th><th class=n>Binoculars</th><th>Verdict</th><th class=n>Fast-DetectGPT</th>{''.join(f'<th class=n>ppl {esc(l)}</th>' for l in PPL_LABELS)}<th class=n>Burstiness</th><th class=n>AI-isms</th><th>Hidden chars</th></tr>"
idx_rows = "".join(f"<tr><td><a href='{esc(n)[:-4]}-report.html'>{esc(n)}</a></td><td class=n>{w}</td><td class='n {cls_b(b)}'>{b:.3f}</td><td>{esc(v)}</td><td class=n>{fd:.2f}</td>{''.join(f'<td class=n>{fmt(x)}</td>' for x in pp)}<td class=n>{fmt(sd)}</td><td class=n>{ss}</td><td>{esc(us)}{(' (' + str(uh + um) + ')') if us != 'clean' else ''}</td></tr>" for n, w, b, v, pp, sd, ss, sl, us, uh, um, fd in summary_rows)
ctrl_rows = "".join(f"<tr><td class=mut>control ({'human' if is_human[c.resolve()] else 'machine'}): {esc(c.name)}</td><td class=n>{len(texts[c.resolve()].split())}</td><td class='n {cls_b(bino_res[c.resolve()][0])}'>{bino_res[c.resolve()][0]:.3f}</td><td></td><td class=n>{bino_res[c.resolve()][2]:.2f}</td>{''.join(f'<td class=n>{fmt(lm[(c.resolve(), l)][chr(112)+chr(112)+chr(108)])}</td>' for l in PPL_LABELS)}<td class=n>{fmt(sent_ppl[c.resolve()]['sd'])}</td><td class=n>{style_res[c.resolve()].get('score')}</td><td>{esc(uni_res[c.resolve()]['status'])}</td></tr>" for c in CONTROLS)
sep_rows = "".join(f"<tr><td>{esc(grp)}</td><td>{esc(title)}</td><td class=n>{separation(k):.2f}</td><td>{'higher' if hih else ('lower' if hih is False else 'no claim')}</td></tr>" for grp, k, title, hih, _, _, _ in sorted(MEAS, key=lambda m: -(separation(m[1]) if separation(m[1]) == separation(m[1]) else 0)))
(out / "index-report.html").write_text(f"""<!DOCTYPE html><html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>AI text reports</title><style>{CSS}</style></head><body><main>
<h1>AI text reports</h1><p class=by>{len(targets)} files · run {now} · profile <b>{esc(args.profile)}</b>{'' if CAL else ' (uncalibrated: no thresholds, no verdicts)'} · {esc(model_line)} · all local</p>
<ul>{tmarks}</ul>{legend}{idx_strip}
<div class=wrap><table>{hdr}{idx_rows}{ctrl_rows}</table></div>
<p>{('Binoculars: below ' + f'{T_ACC:.4f}' + ' reads as machine-generated (accuracy-optimal), below ' + f'{T_FPR:.4f}' + ' at the low-false-positive setting.') if CAL else 'This profile has no published thresholds. Compare each file with the control ranges.'} Each file links to its own report with violins, token map, annotated text and sentence-level detail.</p>
<h2>Which measures separate the controls under this profile</h2>
<p>AUC = probability that a random human control sits on the human side of a random machine control for that measure alone. 1.0 is perfect separation, 0.5 is none, below 0.5 means the controls run opposite to the expected direction. The "human side" column says which direction counts as human.</p>
<div class=wrap><table><tr><th>Group</th><th>Measure</th><th class=n>AUC</th><th>Human side</th></tr>{sep_rows}</table></div>
</main></body></html>""")

if args.selftest:
    h = [bino_res[c][0] for c in HC]; a = [bino_res[c][0] for c in MC]
    ah, aa = bino_res[ANCHOR_H.resolve()][0], bino_res[ANCHOR_A.resolve()][0]
    if CAL: ok = ah >= T_ACC and aa < T_FPR; msg = f"anchor human control {ah:.3f} (expect >= {T_ACC:.4f}), anchor machine control {aa:.3f} (expect < {T_FPR:.4f}); all-control means human {sum(h)/len(h):.3f} vs machine {sum(a)/len(a):.3f}"
    else: ok = ah > aa and (sum(h) / len(h)) > (sum(a) / len(a)); msg = f"anchor human {ah:.3f} vs anchor machine {aa:.3f}; all-control means human {sum(h)/len(h):.3f} vs machine {sum(a)/len(a):.3f} (expect human higher)"
    print(f"self-test [{args.profile}]: {msg} -> {'PASS' if ok else 'FAIL'}"); sys.exit(0 if ok else 1)
for n, w, b, v, pp, sd, ss, sl, us, uh, um, fd in summary_rows:
    print(f"{n:40} Binoculars {b:.3f} {v:36} FastDetectGPT {fd:6.2f}  ppl {pp[0]:7.1f}  AI-isms {ss:>3}/100  hidden chars: {us}")
print(f"reports: {out}/index-report.html")
