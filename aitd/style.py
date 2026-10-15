"""AI-ism / writing-pattern check. Runs the avoid-ai-writing detector (Conor Bronsdon, MIT; vendored unmodified in
third_party/avoid-ai-writing/) in-process on the V8 engine bundled with the mini-racer package. No Node.js needed.
This is editorial evidence about how the text reads, not proof of authorship.
Usage: python style.py file.txt [file2 ...]   -> JSON  (score 0-100, label, issues[], stats)"""
import sys, json, pathlib
from py_mini_racer import MiniRacer
HERE = pathlib.Path(__file__).resolve().parent.parent   # repo root (this file lives in aitd/)
_SRC = (HERE / "third_party" / "avoid-ai-writing" / "patterns.js").read_text()
_ctx = None
def _engine():
    global _ctx
    if _ctx is None:
        _ctx = MiniRacer(); _ctx.eval("var module={exports:{}}; var exports=module.exports;"); _ctx.eval(_SRC)
    return _ctx
def analyze(text, context_mode="general"):
    c = _engine()
    r = json.loads(c.eval(f"JSON.stringify(AIDetector.analyzeText({json.dumps(text)}, {{contextMode: {json.dumps(context_mode)}}}))"))
    labels = json.loads(c.eval("JSON.stringify(AIDetector.TYPE_LABELS)"))
    for i in r.get("issues", []): i["typeLabel"] = labels.get(i.get("type"), i.get("type"))
    return r
if __name__ == "__main__":
    print(json.dumps({f: analyze(open(f, encoding="utf-8").read()) for f in sys.argv[1:]}, indent=1))
