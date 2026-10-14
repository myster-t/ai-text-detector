"""ai-text-detector report: Binoculars (Falcon-7B pair) + perplexity/burstiness (Qwen2.5-1.5B, GPT-2), one self-contained HTML per file.
Usage:  HF_HUB_OFFLINE=1 .venv/bin/python report.py --out DIR file1.txt [file2.txt ...]
Models load once per run and are released when the process exits; nothing stays resident."""
import sys, re, math, json, html, argparse, datetime, pathlib, torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
HERE = pathlib.Path(__file__).resolve().parent
CONTROLS = sorted((HERE / "controls").glob("*.txt"))
PPL_MODELS = [("Qwen/Qwen2.5-1.5B", "Qwen2.5-1.5B"), ("openai-community/gpt2", "GPT-2")]
OBS, PERF = "tiiuae/falcon-7b", "tiiuae/falcon-7b-instruct"
T_ACC, T_FPR = 0.9015310749276843, 0.8536432310785527
dev = "mps" if torch.backends.mps.is_available() else "cpu"
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("files", nargs="+"); args = ap.parse_args()
out = pathlib.Path(args.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
targets = [pathlib.Path(f).expanduser() for f in args.files]
texts = {p: open(p).read().strip() for p in targets + CONTROLS}
def sentences(t): return [s for s in re.split(r"(?<=[.!?])\s+", t) if len(s.split()) >= 4]
# ---- perplexity ----
ppl_res = {}  # (path, label) -> dict
for mid, label in PPL_MODELS:
    tok = AutoTokenizer.from_pretrained(mid); m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32).to(dev).eval()
    def ppl(s):
        ids = tok(s, return_tensors="pt", truncation=True, max_length=1024).input_ids.to(dev)
        if ids.shape[1] < 2: return float("nan")
        with torch.no_grad(): return math.exp(m(ids, labels=ids).loss.item())
    for p, t in texts.items():
        sp = [(s, ppl(s)) for s in sentences(t)]; vals = [v for _, v in sp]
        mean = sum(vals) / len(vals); sd = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        ppl_res[(p, label)] = {"doc": ppl(t), "sent": sp, "mean": mean, "sd": sd}
    del m, tok; torch.mps.empty_cache() if dev == "mps" else None
# ---- binoculars ----
tok = AutoTokenizer.from_pretrained(OBS); tok.pad_token = tok.pad_token or tok.eos_token
obs = AutoModelForCausalLM.from_pretrained(OBS, dtype=torch.bfloat16).to(dev).eval()
perf = AutoModelForCausalLM.from_pretrained(PERF, dtype=torch.bfloat16).to(dev).eval()
def bino(t):
    enc = tok(t, return_tensors="pt", truncation=True, max_length=512).to(dev)
    with torch.no_grad(): lo = obs(**enc).logits[:, :-1].float(); lp = perf(**enc).logits[:, :-1].float()
    lab = enc.input_ids[:, 1:]
    lppl = F.cross_entropy(lp.transpose(1, 2), lab).item(); xppl = (-(F.softmax(lo, -1) * F.log_softmax(lp, -1)).sum(-1)).mean().item()
    return lppl / xppl, lab.shape[1]
bino_res = {p: bino(t) for p, t in texts.items()}
del obs, perf
# ---- html ----
CSS = """:root{--bg:#f7f6f2;--ink:#1d1d1b;--mut:#5f5e58;--rule:#d9d6cc;--acc:#2b4c7e;--ok:#2d6a3e;--bad:#9d2b2b;--th:#ebe8df;--panel:#fffdf8}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#161614;--ink:#ebe8e0;--mut:#a8a59b;--rule:#3a3934;--acc:#8fb0e0;--ok:#7fc290;--bad:#e07b7b;--th:#262622;--panel:#1f1f1c}}
:root[data-theme=dark]{--bg:#161614;--ink:#ebe8e0;--mut:#a8a59b;--rule:#3a3934;--acc:#8fb0e0;--ok:#7fc290;--bad:#e07b7b;--th:#262622;--panel:#1f1f1c}
html,body{background:var(--bg)}body{margin:0;padding:36px 16px 72px;color:var(--ink);font:15px/1.5 Georgia,"Iowan Old Style",serif}main{max-width:880px;margin:0 auto}
h1{font-size:1.7rem;margin:0 0 4px;text-wrap:balance}h2{font-size:1.15rem;margin:34px 0 8px;padding-top:12px;border-top:2px solid var(--rule)}p{max-width:70ch;margin:0 0 10px}
.by{color:var(--mut);font:.85rem -apple-system,Helvetica,Arial,sans-serif;margin-bottom:22px}.verdict{background:var(--panel);border-left:5px solid var(--acc);padding:14px 18px;margin:0 0 18px}
.verdict b{font-size:1.25rem}.ok{color:var(--ok)}.bad{color:var(--bad)}table{border-collapse:collapse;width:100%;font:.84rem/1.4 -apple-system,Helvetica,Arial,sans-serif}
th,td{border:1px solid var(--rule);padding:6px 8px;vertical-align:top;text-align:left}th{background:var(--th)}td.n,th.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.wrap{overflow-x:auto;margin:10px 0 16px}.bar{display:inline-block;height:9px;background:var(--acc);vertical-align:middle;border-radius:2px}.mut{color:var(--mut)}.tgt{font-weight:600}
@media print{body{padding:0}h2{break-after:avoid}tr{break-inside:avoid}}"""
def esc(s): return html.escape(s)
def cls_b(v): return "ok" if v >= T_ACC else "bad"
def fmt(v): return f"{v:,.1f}" if v == v else "n/a"
now = datetime.date.today().isoformat()
summary_rows = []
for p in targets:
    t = texts[p]; b, ntok = bino_res[p]; q = ppl_res[(p, "Qwen2.5-1.5B")]; g = ppl_res[(p, "GPT-2")]
    verdict = "human" if b >= T_ACC else ("AI-like (accuracy line)" if b >= T_FPR else "AI-like (strict line)")
    summary_rows.append((p.name, len(t.split()), b, verdict, q["doc"], q["sd"], g["doc"]))
    # metrics table incl. controls
    rows = []
    for cp, label in [(p, p.name + " (this file)")] + [(c, "control: " + c.stem.replace("-", " ")) for c in CONTROLS if c.resolve() != p.resolve()]:
        cb, _ = bino_res[cp]; cq = ppl_res[(cp, "Qwen2.5-1.5B")]; cg = ppl_res[(cp, "GPT-2")]
        rows.append(f"<tr{' class=tgt' if cp == p else ''}><td>{esc(label)}</td><td class=n>{len(texts[cp].split())}</td>"
                    f"<td class='n {cls_b(cb)}'>{cb:.3f}</td><td class=n>{fmt(cq['doc'])}</td><td class=n>{fmt(cq['mean'])}</td><td class=n>{fmt(cq['sd'])}</td><td class=n>{fmt(cg['doc'])}</td><td class=n>{fmt(cg['sd'])}</td></tr>")
    mx = max(v for _, v in q["sent"]) or 1
    sent_rows = "".join(f"<tr><td class=n>{i+1}</td><td>{esc(s)}</td><td class=n>{fmt(v)}</td><td><span class=bar style='width:{int(140*v/mx)}px'></span></td></tr>" for i, (s, v) in enumerate(q["sent"]))
    doc = f"""<!DOCTYPE html><html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>AI writing report: {esc(p.stem)}</title><style>{CSS}</style></head><body><main>
<h1>AI writing report</h1><p class=by>{esc(p.name)} · {len(t.split())} words · run {now} · all processing local (Falcon-7B pair, Qwen2.5-1.5B, GPT-2)</p>
<div class=verdict><p><b class={cls_b(b)}>Binoculars {b:.3f}: {esc(verdict)}</b></p><p>Scores below {T_ACC:.4f} read as AI-generated at the paper's accuracy-optimal setting; below {T_FPR:.4f} at the low-false-positive setting. Higher is more human-like. This text is {abs(b-T_ACC):.3f} {'above' if b>=T_ACC else 'below'} the accuracy line.</p>
<p>Perplexity under Qwen2.5-1.5B is {fmt(q['doc'])} with sentence-level burstiness {fmt(q['sd'])}. Language-model text in our controls sits near 5 with burstiness near 10; human prose runs far higher on both.</p></div>
<h2>Measurements against controls</h2><div class=wrap><table><tr><th>Text</th><th class=n>Words</th><th class=n>Binoculars</th><th class=n>Qwen doc ppl</th><th class=n>Qwen sentence mean</th><th class=n>Qwen burstiness</th><th class=n>GPT-2 doc ppl</th><th class=n>GPT-2 burstiness</th></tr>{''.join(rows)}</table></div>
<p class=mut>Controls are re-scored in the same run so the numbers are directly comparable. The human control is the opening of a 1913 letter by Elinore Pruitt Stewart (<i>Letters of a Woman Homesteader</i>, public domain); the machine control is an unedited marketing paragraph generated by a language model.</p>
<h2>Sentence-level perplexity (Qwen2.5-1.5B)</h2><p>Each sentence scored on its own. Wide variation between sentences (burstiness) is characteristic of human writing; generated text tends to be uniformly predictable.</p>
<div class=wrap><table><tr><th class=n>#</th><th>Sentence</th><th class=n>Perplexity</th><th></th></tr>{sent_rows}</table></div>
<h2>Methods and limits</h2>
<p><b>Binoculars</b> (Hans et al., 2024, ICML): the text's perplexity under a performer model (Falcon-7B-Instruct) divided by the cross-perplexity between an observer (Falcon-7B) and the performer. This normalises for prompts and topics that make text predictable for reasons other than machine generation. Reference models and thresholds are the paper's. Input truncated to 512 tokens ({ntok} scored here).</p>
<p><b>Perplexity and burstiness</b>: how predictable the text is to a language model, and how much that varies sentence to sentence. Simple, well understood, and the basis of commercial tools; weak on short texts, formulaic genres and text dense with names or numbers.</p>
<p>No detector proves authorship. Edited machine text can pass, and terse or formulaic human writing can be flagged. Read these numbers alongside drafts, sources and the writer's history.</p>
</main></body></html>"""
    (out / f"{p.stem}-report.html").write_text(doc)
# index
idx_rows = "".join(f"<tr><td><a href='{esc(n)[:-4]}-report.html'>{esc(n)}</a></td><td class=n>{w}</td><td class='n {cls_b(b)}'>{b:.3f}</td><td>{esc(v)}</td><td class=n>{fmt(qd)}</td><td class=n>{fmt(qs)}</td><td class=n>{fmt(gd)}</td></tr>" for n, w, b, v, qd, qs, gd in summary_rows)
ctrl_rows = "".join(f"<tr><td class=mut>control: {esc(c.stem.replace('-',' '))}</td><td class=n>{len(texts[c].split())}</td><td class='n {cls_b(bino_res[c][0])}'>{bino_res[c][0]:.3f}</td><td></td><td class=n>{fmt(ppl_res[(c,'Qwen2.5-1.5B')]['doc'])}</td><td class=n>{fmt(ppl_res[(c,'Qwen2.5-1.5B')]['sd'])}</td><td class=n>{fmt(ppl_res[(c,'GPT-2')]['doc'])}</td></tr>" for c in CONTROLS)
(out / "index-report.html").write_text(f"""<!DOCTYPE html><html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>AI writing reports</title><style>{CSS}</style></head><body><main>
<h1>AI writing reports</h1><p class=by>{len(targets)} files · run {now} · Binoculars (Falcon-7B pair), Qwen2.5-1.5B and GPT-2 perplexity · all local</p>
<div class=wrap><table><tr><th>File</th><th class=n>Words</th><th class=n>Binoculars</th><th>Verdict</th><th class=n>Qwen ppl</th><th class=n>Qwen burstiness</th><th class=n>GPT-2 ppl</th></tr>{idx_rows}{ctrl_rows}</table></div>
<p>Binoculars: below {T_ACC:.4f} reads as AI (accuracy-optimal), below {T_FPR:.4f} at the low-false-positive setting. Each file links to its own report with sentence-level detail.</p></main></body></html>""")
print(json.dumps({"written": [str(out / f"{p.stem}-report.html") for p in targets] + [str(out / "index-report.html")]}, indent=1))
