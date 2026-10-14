"""ai-text-detector: Binoculars + perplexity/burstiness + writing-pattern + hidden-character checks, one HTML report per file.
Usage:  ./detect file1.txt [file2 ...] [--out DIR] [--profile original|latest] [--selftest]
Models are loaded for the run and released when it exits; nothing stays resident."""
import os, sys, re, math, json, html, argparse, datetime, pathlib
HERE = pathlib.Path(__file__).resolve().parent
_md = os.environ.get("AITD_MODELS_DIR") or ((HERE / ".models-dir").read_text().strip() if (HERE / ".models-dir").is_file() else "") or str(HERE / "models")
if pathlib.Path(_md).is_dir(): os.environ.setdefault("HF_HOME", _md)
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1"); os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, str(HERE)); import style as style_mod, unicode_check

PROFILES = json.loads((HERE / "profiles.json").read_text())
ap = argparse.ArgumentParser(description="Score text files for machine generation and write HTML reports.")
ap.add_argument("files", nargs="*", help="UTF-8 text files to score")
ap.add_argument("--out", default="reports", help="output folder (default: ./reports)")
ap.add_argument("--profile", default="original", choices=sorted(PROFILES), help="model set (default: original = calibrated Falcon-7B pair)")
ap.add_argument("--selftest", action="store_true", help="score the bundled controls and check they separate")
args = ap.parse_args()
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
ctrl_set = {c.resolve() for c in CONTROLS}
texts = {p: open(p, encoding="utf-8").read().strip() for p in targets + [c.resolve() for c in CONTROLS]}
is_human = {c.resolve(): True for c in HUMAN}; is_human.update({c.resolve(): False for c in AI})

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

# ---- perplexity models ----
ppl_res = {}
for mid, label in P["perplexity_models"]:
    tok, m = load(mid, torch.float32 if "gpt2" in mid else torch.bfloat16)
    def ppl(s):
        ids = tok(s, return_tensors="pt", truncation=True, max_length=1024).input_ids.to(dev)
        if ids.shape[1] < 2: return float("nan")
        with torch.no_grad(): lg = m(input_ids=ids).logits[:, :-1].float()
        return math.exp(F.cross_entropy(lg.transpose(1, 2), ids[:, 1:]).item())
    for p, t in texts.items():
        sp = [(s, ppl(s)) for s in sentences(t)]; vals = [v for _, v in sp] or [float("nan")]
        mean = sum(vals) / len(vals); sd = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        ppl_res[(p, label)] = {"doc": ppl(t), "sent": sp, "mean": mean, "sd": sd}
    release(tok, m)
PPL_LABELS = [l for _, l in P["perplexity_models"]]

# ---- binoculars ----
otok, obs = load(P["observer"], torch.bfloat16); ptok, perf = load(P["performer"], torch.bfloat16)
def bino(t):
    enc = otok(t, return_tensors="pt", truncation=True, max_length=512).to(dev); ids = enc.input_ids
    with torch.no_grad(): lo = obs(input_ids=ids).logits[:, :-1].float(); lp = perf(input_ids=ids).logits[:, :-1].float()
    v = min(lo.shape[-1], lp.shape[-1]); lo, lp = lo[..., :v], lp[..., :v]; lab = ids[:, 1:]
    lppl = F.cross_entropy(lp.transpose(1, 2), lab).item(); xppl = (-(F.softmax(lo, -1) * F.log_softmax(lp, -1)).sum(-1)).mean().item()
    return lppl / xppl, lab.shape[1]
bino_res = {p: bino(t) for p, t in texts.items()}
release(otok, obs, ptok, perf)

# ---- deterministic checks ----
style_res = {p: style_mod.analyze(t) for p, t in texts.items()}
uni_res = {p: unicode_check.check(t) for p, t in texts.items()}

# ---- html ----
CSS = """:root{--bg:#f7f6f2;--ink:#1d1d1b;--mut:#5f5e58;--rule:#d9d6cc;--acc:#2b4c7e;--ok:#2d6a3e;--bad:#9d2b2b;--th:#ebe8df;--panel:#fffdf8;--hum:#3a8f5a;--mach:#c0392b;--tgt:#1d1d1b}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#161614;--ink:#ebe8e0;--mut:#a8a59b;--rule:#3a3934;--acc:#8fb0e0;--ok:#7fc290;--bad:#e07b7b;--th:#262622;--panel:#1f1f1c;--hum:#6fcf8f;--mach:#ef7b6f;--tgt:#ebe8e0}}
:root[data-theme=dark]{--bg:#161614;--ink:#ebe8e0;--mut:#a8a59b;--rule:#3a3934;--acc:#8fb0e0;--ok:#7fc290;--bad:#e07b7b;--th:#262622;--panel:#1f1f1c;--hum:#6fcf8f;--mach:#ef7b6f;--tgt:#ebe8e0}
html,body{background:var(--bg)}body{margin:0;padding:36px 16px 72px;color:var(--ink);font:15px/1.5 Georgia,"Iowan Old Style",serif}main{max-width:900px;margin:0 auto}
h1{font-size:1.7rem;margin:0 0 4px;text-wrap:balance}h2{font-size:1.15rem;margin:34px 0 8px;padding-top:12px;border-top:2px solid var(--rule)}h3{font-size:.95rem;margin:18px 0 4px;font-family:-apple-system,Helvetica,Arial,sans-serif}p{max-width:72ch;margin:0 0 10px}
.by{color:var(--mut);font:.85rem -apple-system,Helvetica,Arial,sans-serif;margin-bottom:22px}.verdict{background:var(--panel);border-left:5px solid var(--acc);padding:14px 18px;margin:0 0 18px}
.verdict b{font-size:1.25rem}.ok{color:var(--ok)}.bad{color:var(--bad)}table{border-collapse:collapse;width:100%;font:.84rem/1.4 -apple-system,Helvetica,Arial,sans-serif}
th,td{border:1px solid var(--rule);padding:6px 8px;vertical-align:top;text-align:left}th{background:var(--th)}td.n,th.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.wrap{overflow-x:auto;margin:10px 0 16px}.bar{display:inline-block;height:9px;background:var(--acc);vertical-align:middle;border-radius:2px}.mut{color:var(--mut)}.tgt{font-weight:600}
.strip{width:100%;max-width:860px;height:auto;display:block;margin:4px 0 14px}.legend{font:.8rem -apple-system,Helvetica,Arial,sans-serif;color:var(--mut)}
.sw{display:inline-block;width:10px;height:10px;border-radius:50%;vertical-align:middle;margin-right:4px}
@media print{body{padding:0}h2{break-after:avoid}tr{break-inside:avoid}}"""
def esc(s): return html.escape(str(s))
def fmt(v): return f"{v:,.1f}" if isinstance(v, (int, float)) and v == v else "n/a"
def strip_svg(title, points, target_val, target_name, log=False, lines=()):
    """Number-line strip: control dots (human green, machine red), target as a black diamond, optional threshold lines."""
    vals = [v for _, v, _ in points if v == v] + ([target_val] if target_val == target_val else []) + [v for v, _ in lines]
    f = (lambda x: math.log10(max(x, 1e-9))) if log else (lambda x: x)
    lo, hi = min(map(f, vals)), max(map(f, vals)); pad = (hi - lo) * 0.08 or 0.5; lo, hi = lo - pad, hi + pad
    W, H, L, R = 860, 96, 70, 20
    X = lambda v: L + (f(v) - lo) / (hi - lo) * (W - L - R)
    s = [f'<svg class=strip viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{esc(title)}">',
         f'<text x="0" y="14" font-size="12" font-family="-apple-system,Helvetica,Arial,sans-serif" fill="currentColor">{esc(title)}</text>',
         f'<line x1="{L}" y1="58" x2="{W-R}" y2="58" stroke="currentColor" stroke-opacity=".35" stroke-width="1"/>']
    nt = 6
    for i in range(nt + 1):
        v = lo + (hi - lo) * i / nt; x = L + (W - L - R) * i / nt; lab = f"{10**v:.3g}" if log else f"{v:.3g}"
        s.append(f'<line x1="{x:.1f}" y1="54" x2="{x:.1f}" y2="62" stroke="currentColor" stroke-opacity=".35"/><text x="{x:.1f}" y="78" font-size="10" text-anchor="middle" fill="currentColor" fill-opacity=".7" font-family="-apple-system,Helvetica,Arial,sans-serif">{lab}</text>')
    for v, name in lines:
        s.append(f'<line x1="{X(v):.1f}" y1="26" x2="{X(v):.1f}" y2="66" stroke="currentColor" stroke-opacity=".6" stroke-dasharray="3 3"/><text x="{X(v):.1f}" y="92" font-size="10" text-anchor="middle" fill="currentColor" fill-opacity=".8" font-family="-apple-system,Helvetica,Arial,sans-serif">{esc(name)}</text>')
    for name, v, human in points:
        if v != v: continue
        y = 44 if human else 58 + 0; y = 40 if human else 58
        s.append(f'<circle cx="{X(v):.1f}" cy="{y}" r="5" fill="{"var(--hum)" if human else "var(--mach)"}" fill-opacity=".8"><title>{esc(name)}: {v:.3g}</title></circle>')
    if target_val == target_val:
        x = X(target_val); s.append(f'<polygon points="{x:.1f},40 {x+7:.1f},49 {x:.1f},58 {x-7:.1f},49" fill="var(--tgt)"><title>{esc(target_name)}: {target_val:.3g}</title></polygon>')
    s.append('</svg>'); return "".join(s)
legend = '<p class=legend><span class=sw style="background:var(--hum)"></span>human controls &nbsp; <span class=sw style="background:var(--mach)"></span>machine controls &nbsp; <span style="display:inline-block;width:9px;height:9px;background:var(--tgt);transform:rotate(45deg);margin-right:4px"></span>this file</p>'
now = datetime.date.today().isoformat()
model_line = f"Binoculars pair {P['observer']} / {P['performer']}; perplexity {', '.join(m for m, _ in P['perplexity_models'])}"
def verdict_of(b):
    if not CAL: return "no verdict (uncalibrated model set)"
    return "human" if b >= T_ACC else ("AI-like (accuracy line)" if b >= T_FPR else "AI-like (strict line)")
def cls_b(b): return "" if not CAL else ("ok" if b >= T_ACC else "bad")
cpoints_b = [(c.name, bino_res[c.resolve()][0], is_human[c.resolve()]) for c in CONTROLS]
def cpoints_ppl(label, key): return [(c.name, ppl_res[(c.resolve(), label)][key], is_human[c.resolve()]) for c in CONTROLS]
def rank_text(val, pts, higher_is_human):
    hv = sorted(v for _, v, h in pts if h and v == v); av = sorted(v for _, v, h in pts if not h and v == v)
    if not hv or not av or val != val: return ""
    side = "human" if (val >= (max(av) + min(hv)) / 2) == higher_is_human else "machine"
    pos = sum(1 for v in hv + av if (v < val) == higher_is_human)
    return f"{len(hv)} human controls span {min(hv):.3g} to {max(hv):.3g}; {len(av)} machine controls span {min(av):.3g} to {max(av):.3g}. This file sits on the {side} side of the gap between the two groups' nearest edges."

summary_rows = []
for p in targets:
    t = texts[p]; b, ntok = bino_res[p]; st, un = style_res[p], uni_res[p]; q = ppl_res[(p, PPL_LABELS[0])]
    verdict = verdict_of(b)
    summary_rows.append((p.name, len(t.split()), b, verdict, [ppl_res[(p, l)]["doc"] for l in PPL_LABELS], q["sd"], st.get("score"), st.get("label"), un["status"], un["hidden_total"], len(un["mixed_script_words"])))
    if CAL:
        box = f"<p><b class={cls_b(b)}>Binoculars {b:.3f}: {esc(verdict)}</b></p><p>Scores below {T_ACC:.4f} read as machine-generated at the paper's accuracy-optimal setting; below {T_FPR:.4f} at the low-false-positive setting. Higher is more human-like. This file is {abs(b-T_ACC):.3f} {'above' if b>=T_ACC else 'below'} the accuracy line.</p>"
        tlines = [(T_FPR, "strict"), (T_ACC, "accuracy")]
    else:
        box = f"<p><b>Binoculars {b:.3f}</b> under an uncalibrated model set, so no verdict is given.</p><p>{esc(rank_text(b, cpoints_b, True))}</p>"
        tlines = []
    box += f"<p>Perplexity under {PPL_LABELS[0]} is {fmt(q['doc'])} with sentence-level burstiness {fmt(q['sd'])}. {esc(rank_text(q['doc'], cpoints_ppl(PPL_LABELS[0], 'doc'), True))}</p>"
    strips = strip_svg("Binoculars score (higher = more human-like)", cpoints_b, b, p.name, lines=tlines)
    for l in PPL_LABELS:
        strips += strip_svg(f"Document perplexity, {l} (log scale; higher = less predictable)", cpoints_ppl(l, "doc"), ppl_res[(p, l)]["doc"], p.name, log=True)
    strips += strip_svg(f"Burstiness, {PPL_LABELS[0]} (sd of sentence perplexity; log scale)", cpoints_ppl(PPL_LABELS[0], "sd"), q["sd"], p.name, log=True)
    rows = []
    for cp, label, hum in [(p, p.name + " (this file)", None)] + [(c.resolve(), c.name, is_human[c.resolve()]) for c in CONTROLS if c.resolve() != p]:
        cb = bino_res[cp][0]; cells = "".join(f"<td class=n>{fmt(ppl_res[(cp, l)]['doc'])}</td>" for l in PPL_LABELS)
        rows.append(f"<tr{' class=tgt' if cp == p else ''}><td>{esc(label)}</td><td>{'' if hum is None else ('human' if hum else 'machine')}</td><td class=n>{len(texts[cp].split())}</td><td class='n {cls_b(cb)}'>{cb:.3f}</td>{cells}<td class=n>{fmt(ppl_res[(cp, PPL_LABELS[0])]['sd'])}</td><td class=n>{style_res[cp].get('score')}</td><td>{esc(uni_res[cp]['status'])}</td></tr>")
    mx = max((v for _, v in q["sent"] if v == v), default=1) or 1
    sent_rows = "".join(f"<tr><td class=n>{i+1}</td><td>{esc(s)}</td><td class=n>{fmt(v)}</td><td><span class=bar style='width:{int(140*v/mx) if v == v else 0}px'></span></td></tr>" for i, (s, v) in enumerate(q["sent"]))
    issues = st.get("issues", [])
    style_tbl = ('<div class=wrap><table><tr><th>Severity</th><th>Pattern</th><th>Text</th><th>Suggestion</th></tr>' + ''.join(f"<tr><td>{esc(i.get('severity',''))}</td><td>{esc(i.get('typeLabel',''))}</td><td>{esc(str(i.get('text',''))[:120])}</td><td class=mut>{esc(str(i.get('suggestion') or ''))[:160]}</td></tr>" for i in issues[:60]) + '</table></div>') if issues else '<p class=mut>No patterns flagged.</p>'
    hid_tbl = ('<div class=wrap><table><tr><th class=n>Position</th><th>Code point</th><th>Kind</th><th>Name</th></tr>' + ''.join(f"<tr><td class=n>{h['pos']}</td><td>{h['cp']}</td><td>{esc(h['kind'])}</td><td>{esc(h['name'])}</td></tr>" for h in un['hidden'][:60]) + '</table></div>') if un['hidden'] else ''
    mix_tbl = ('<div class=wrap><table><tr><th class=n>Position</th><th>Word</th><th>Scripts</th><th>Non-Latin letters</th></tr>' + ''.join(f"<tr><td class=n>{m['pos']}</td><td>{esc(m['word'])}</td><td>{esc(', '.join(m['scripts']))}</td><td>{esc('; '.join(m['foreign']))}</td></tr>" for m in un['mixed_script_words'][:60]) + '</table></div>') if un['mixed_script_words'] else ''
    hs, as_ = style_res[HUMAN[0].resolve()].get('score') if HUMAN else '', style_res[AI[0].resolve()].get('score') if AI else ''
    doc = f"""<!DOCTYPE html><html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>AI text report: {esc(p.stem)}</title><style>{CSS}</style></head><body><main>
<h1>AI text report</h1><p class=by>{esc(p.name)} · {len(t.split())} words · run {now} · profile <b>{esc(args.profile)}</b>{'' if CAL else ' (uncalibrated)'} · {esc(model_line)} · all processing local</p>
<div class=verdict>{box}</div>
<h2>Where this file sits among the controls</h2>
<p>{len(HUMAN)} human and {len(AI)} machine control texts are scored in the same run with the same models. Each strip is a number line; hover a dot for the control's name and value.</p>
{legend}{strips}
<h2>Measurements against controls</h2><div class=wrap><table><tr><th>Text</th><th>Kind</th><th class=n>Words</th><th class=n>Binoculars</th>{''.join(f'<th class=n>ppl {esc(l)}</th>' for l in PPL_LABELS)}<th class=n>Burstiness</th><th class=n>AI-isms</th><th>Hidden chars</th></tr>{''.join(rows)}</table></div>
<p class=mut>Controls are described in controls/README.md: public-domain and Creative Commons human writing from several decades and registers, and machine text from ChatGPT (2023) and Claude (2026).</p>
<h2>Sentence-level perplexity ({esc(PPL_LABELS[0])})</h2><p>Each sentence scored on its own. Wide variation between sentences (burstiness) is characteristic of human writing; generated text tends to be uniformly predictable.</p>
<div class=wrap><table><tr><th class=n>#</th><th>Sentence</th><th class=n>Perplexity</th><th></th></tr>{sent_rows}</table></div>
<h2>Writing-pattern check (AI-isms)</h2>
<p><b>Score {st.get('score')} / 100: {esc(st.get('label'))}</b>{(' · ' + esc(st.get('document_classification')) + ', ' + esc(st.get('confidence_category')) + ' confidence') if st.get('document_classification') else ''}. Control range: human {min(style_res[c.resolve()].get('score', 0) for c in HUMAN)} to {max(style_res[c.resolve()].get('score', 0) for c in HUMAN)}, machine {min(style_res[c.resolve()].get('score', 0) for c in AI)} to {max(style_res[c.resolve()].get('score', 0) for c in AI)}.</p>
<p>Deterministic rules from the <a href="https://github.com/conorbronsdon/avoid-ai-writing">avoid-ai-writing</a> detector (Conor Bronsdon, MIT): vocabulary tiers, transitions, template phrases, chatbot artifacts, generic conclusions, hedging, formatting and rhythm signals. These describe how the text reads and are useful for editing; they are weak evidence of authorship on their own.</p>{style_tbl}
<h2>Hidden characters and mixed scripts</h2>
<p><b>{esc(un['status'])}</b>: {un['hidden_total']} hidden or non-standard characters, {len(un['mixed_script_words'])} words mixing Latin with another script, in {un['chars']} characters.</p>
<p>Zero-width characters, soft hyphens, bidirectional controls, variation selectors, tag characters and lookalike letters from other alphabets can carry watermarks or disguise text, but also arrive through ordinary copy-and-paste. They are evidence about the file, not about who wrote the words.</p>{hid_tbl}{mix_tbl}
<h2>Methods and limits</h2>
<p><b>Binoculars</b> (Hans et al., ICML 2024): the text's perplexity under a performer model divided by the cross-perplexity between an observer and the performer, which normalises for prompts and topics that make text predictable for reasons other than machine generation. {'This run uses the paper&#39;s reference models and thresholds.' if CAL else 'This run uses a different model pair than the paper, so the published thresholds do not apply and none are shown; read the position relative to the controls instead.'} Input truncated to 512 tokens ({ntok} scored here).</p>
<p><b>Perplexity and burstiness</b>: how predictable the text is to a language model, and how much that varies sentence to sentence. Weak on short texts, formulaic genres and text dense with names or numbers.</p>
<p><b>Writing patterns and hidden characters</b>: deterministic checks that need no model.</p>
<p>No detector proves authorship. Edited machine text can pass, and terse or formulaic human writing can be flagged. Read these numbers alongside drafts, sources and the writer's history.</p>
</main></body></html>"""
    (out / f"{p.stem}-report.html").write_text(doc)

# ---- index ----
all_pts = cpoints_b + [(n, b, None) for n, _, b, *_ in summary_rows]
idx_strip = strip_svg("Binoculars score, all files and controls (higher = more human-like)", cpoints_b, float("nan"), "", lines=([(T_FPR, "strict"), (T_ACC, "accuracy")] if CAL else []))
tmarks = "".join(f'<li><b>{esc(n)}</b>: {b:.3f}</li>' for n, _, b, *_ in summary_rows)
hdr = f"<tr><th>File</th><th class=n>Words</th><th class=n>Binoculars</th><th>Verdict</th>{''.join(f'<th class=n>ppl {esc(l)}</th>' for l in PPL_LABELS)}<th class=n>Burstiness</th><th class=n>AI-isms</th><th>Hidden chars</th></tr>"
idx_rows = "".join(f"<tr><td><a href='{esc(n)[:-4]}-report.html'>{esc(n)}</a></td><td class=n>{w}</td><td class='n {cls_b(b)}'>{b:.3f}</td><td>{esc(v)}</td>{''.join(f'<td class=n>{fmt(x)}</td>' for x in pp)}<td class=n>{fmt(sd)}</td><td class=n>{ss}</td><td>{esc(us)}{(' (' + str(uh + um) + ')') if us != 'clean' else ''}</td></tr>" for n, w, b, v, pp, sd, ss, sl, us, uh, um in summary_rows)
ctrl_rows = "".join(f"<tr><td class=mut>control ({'human' if is_human[c.resolve()] else 'machine'}): {esc(c.name)}</td><td class=n>{len(texts[c.resolve()].split())}</td><td class='n {cls_b(bino_res[c.resolve()][0])}'>{bino_res[c.resolve()][0]:.3f}</td><td></td>{''.join(f'<td class=n>{fmt(ppl_res[(c.resolve(), l)][chr(100)+chr(111)+chr(99)])}</td>' for l in PPL_LABELS)}<td class=n>{fmt(ppl_res[(c.resolve(), PPL_LABELS[0])]['sd'])}</td><td class=n>{style_res[c.resolve()].get('score')}</td><td>{esc(uni_res[c.resolve()]['status'])}</td></tr>" for c in CONTROLS)
(out / "index-report.html").write_text(f"""<!DOCTYPE html><html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>AI text reports</title><style>{CSS}</style></head><body><main>
<h1>AI text reports</h1><p class=by>{len(targets)} files · run {now} · profile <b>{esc(args.profile)}</b>{'' if CAL else ' (uncalibrated: no thresholds, no verdicts)'} · {esc(model_line)} · all local</p>
{legend}{idx_strip}<ul class=legend>{tmarks}</ul>
<div class=wrap><table>{hdr}{idx_rows}{ctrl_rows}</table></div>
<p>{('Binoculars: below ' + f'{T_ACC:.4f}' + ' reads as machine-generated (accuracy-optimal), below ' + f'{T_FPR:.4f}' + ' at the low-false-positive setting.') if CAL else 'This profile has no published thresholds. Compare each file with the control ranges above.'} Each file links to its own report with strips, sentence-level detail, writing patterns and hidden characters.</p></main></body></html>""")

if args.selftest:
    hv = [bino_res[c.resolve()][0] for c in HUMAN]; av = [bino_res[c.resolve()][0] for c in AI]
    ah, aa = bino_res[ANCHOR_H.resolve()][0], bino_res[ANCHOR_A.resolve()][0]
    if CAL: ok = ah >= T_ACC and aa < T_FPR; msg = f"anchor human control {ah:.3f} (expect >= {T_ACC:.4f}), anchor machine control {aa:.3f} (expect < {T_FPR:.4f}); all-control means human {sum(hv)/len(hv):.3f} vs machine {sum(av)/len(av):.3f}"
    else: ok = ah > aa and (sum(hv) / len(hv)) > (sum(av) / len(av)); msg = f"anchor human {ah:.3f} vs anchor machine {aa:.3f}; all-control means human {sum(hv)/len(hv):.3f} vs machine {sum(av)/len(av):.3f} (expect human higher)"
    print(f"self-test [{args.profile}]: {msg} -> {'PASS' if ok else 'FAIL'}"); sys.exit(0 if ok else 1)
for n, w, b, v, pp, sd, ss, sl, us, uh, um in summary_rows:
    print(f"{n:40} Binoculars {b:.3f} {v:36} ppl {pp[0]:7.1f}  AI-isms {ss:>3}/100  hidden chars: {us}")
print(f"reports: {out}/index-report.html")
