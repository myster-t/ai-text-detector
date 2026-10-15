"""Fetch a Google Doc as plain text, one text per tab, without any Google API credentials.

Works for documents shared as "Anyone with the link" (or published): Google's export endpoint
  https://docs.google.com/document/d/<id>/export?format=txt            whole document
  https://docs.google.com/document/d/<id>/export?format=txt&tab=<tabId> one tab (the tab id is the `tab=` value in the
                                                                        browser URL when that tab is open, e.g. t.0)
A private document returns an HTML sign-in page instead of text; the tool then explains what to do.

Tab discovery: tabs come from (1) the ids you pass (`--tabs t.0,t.abc123`), (2) the document's preview page, which lists
every tab id for a link-shared document, or (3) a `tab=` in the URL; if none of those yields anything, the whole document
is scored as one text.
Usage: python gdoc.py URL [--tabs t.0,t.xyz] -> writes <title>.<tab>.txt files to the current folder and prints them."""
import sys, re, json, html, urllib.request, urllib.error, urllib.parse, pathlib

UA = {"User-Agent": "Mozilla/5.0 (ai-text-detector; +https://github.com/myster-t/ai-text-detector)"}

def doc_id(url):
    m = re.search(r"/document/d/([A-Za-z0-9_-]{20,})", url)
    if not m:
        kind = re.search(r"docs\.google\.com/(\w+)/", url)
        raise ValueError(f"only Google Docs documents are supported, not {kind.group(1)} links" if kind else "not a Google Docs URL (expected https://docs.google.com/document/d/<id>/...)")
    return m.group(1)

def url_tab(url):
    q = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query); frag = urllib.parse.urlsplit(url).fragment
    t = q.get("tab", [None])[0] or dict(urllib.parse.parse_qsl(frag)).get("tab")
    return t

PRIVATE_MSG = ("Google will not export this document without signing in, so it is not shared as 'Anyone with the link'. Open it, use "
               "Share > General access > Anyone with the link (Viewer), and run again; or download it (File > Download > Plain Text) and pass the file.")
def _get(url, timeout=60):
    req = urllib.request.Request(url, headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read(); ctype = r.headers.get("Content-Type", "")
    except urllib.error.HTTPError as e:
        if e.code in (401, 403): raise PermissionError(PRIVATE_MSG)
        if e.code == 404: raise FileNotFoundError("Google says this document does not exist (404); check the link.")
        raise
    return data, ctype

def export_text(did, tab=None):
    u = f"https://docs.google.com/document/d/{did}/export?format=txt" + (f"&tab={urllib.parse.quote(tab)}" if tab else "")
    data, ctype = _get(u)
    text = data.decode("utf-8-sig", errors="replace")
    if "text/html" in ctype or re.search(r"<html|accounts\.google\.com|ServiceLogin", text[:2000], re.I): raise PermissionError(PRIVATE_MSG)
    return text.replace("\r\n", "\n").strip()

def discover_tabs(did):
    """Tab ids in document order from the document's preview page, which Google serves without sign-in for link-shared
    documents and which embeds the tab list in its page model (the first tab is listed as "" and exported as t.0).
    Returns a list of tab ids, or [] when nothing recognisable is found."""
    try: data, _ = _get(f"https://docs.google.com/document/d/{did}/preview")
    except Exception: return []
    h = data.decode("utf-8", errors="replace")
    # the page model contains maps keyed by tab id: {"":false,"t.abc123":false,...}; take the first such map in page order
    m = re.search(r'\{"":(?:\[[^\]]*\]|false|true)(?:,"t\.[A-Za-z0-9_-]+":(?:\[[^\]]*\]|false|true))+\}', h)
    if not m: return []
    ids = re.findall(r'"(t\.[A-Za-z0-9_-]+)"', m.group(0))
    return ["t.0"] + [t for t in ids if t != "t.0"]

def title_of(did):
    for u in (f"https://docs.google.com/document/d/{did}/preview", f"https://docs.google.com/document/d/{did}/export?format=html"):
        try:
            data, _ = _get(u); m = re.search(r"<title>(.*?)</title>", data.decode("utf-8", errors="replace"), re.S)
            if m:
                t = html.unescape(m.group(1)).strip(); t = re.sub(r"\s*-\s*Google Docs$", "", t)
                if t: return t
        except Exception: pass
    return did[:8]

def fetch(url, tabs=None):
    """Return a list of (label, text): one entry per tab, labelled with the document title, the tab number and the tab's
    opening words; or a single entry for the whole document when no tabs can be determined."""
    did = doc_id(url); title = title_of(did)
    ids = [t.strip() for t in (tabs or "").split(",") if t.strip()] if tabs else []
    if not ids:
        found = discover_tabs(did)
        if len(found) > 1: ids = found
        elif url_tab(url): ids = [url_tab(url)]
    out = []
    for n, t in enumerate(ids, 1):
        try: txt = export_text(did, t)
        except PermissionError: raise
        except Exception as e: print(f"[gdoc] tab {t}: {e}", file=sys.stderr); continue
        if txt:
            first = " ".join(txt.split()[:6]); out.append((f"{title} - tab {n} - {first}" if len(ids) > 1 else f"{title} - {t}", txt))
    if not out: out.append((title, export_text(did)))
    return out

def safe_name(s): return re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-")[:80] or "doc"

if __name__ == "__main__":
    args = sys.argv[1:]; tabs = None
    if "--tabs" in args: i = args.index("--tabs"); tabs = args[i + 1]; del args[i:i + 2]
    for label, text in fetch(args[0], tabs):
        p = pathlib.Path(safe_name(label) + ".txt"); p.write_text(text + "\n"); print(p, len(text.split()), "words")
