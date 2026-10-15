"""Fetch a Google Doc as plain text, one text per tab, without any Google API credentials.

Works for documents shared as "Anyone with the link" (or published): Google's export endpoint
  https://docs.google.com/document/d/<id>/export?format=txt            whole document
  https://docs.google.com/document/d/<id>/export?format=txt&tab=<tabId> one tab (the tab id is the `tab=` value in the
                                                                        browser URL when that tab is open, e.g. t.0)
A private document returns an HTML sign-in page instead of text; the tool then explains what to do.

Tab discovery: the export has no tab list, so tabs come from (1) the ids you pass (`--tabs t.0,t.abc123`), (2) a `tab=` in
the URL, or (3) the document's HTML export, which carries one bookmark anchor per tab in current Google Docs; if none of those
yields anything, the whole document is scored as one text.
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
    """Tab ids and titles from the HTML export's outline, when present. Returns [] if nothing recognisable."""
    try: data, ctype = _get(f"https://docs.google.com/document/d/{did}/export?format=html")
    except Exception: return []
    h = data.decode("utf-8", errors="replace")
    tabs = []
    # current exports mark each tab with an anchor / heading carrying the tab id (t.0, t.<hash>); keep the order of appearance
    for m in re.finditer(r'(?:id|name)="(t\.[A-Za-z0-9_-]+)"[^>]*>(?:\s*<[^>]+>)*\s*([^<]{0,120})', h):
        tid, title = m.group(1), html.unescape(m.group(2)).strip()
        if tid not in [t for t, _ in tabs]: tabs.append((tid, title))
    return tabs

def title_of(did):
    try:
        data, _ = _get(f"https://docs.google.com/document/d/{did}/export?format=html")
        m = re.search(r"<title>(.*?)</title>", data.decode("utf-8", errors="replace"), re.S)
        return html.unescape(m.group(1)).strip() if m else did[:8]
    except Exception: return did[:8]

def fetch(url, tabs=None):
    """Return a list of (label, text). label = '<doc title> - <tab title or id>' or just the title for a single text."""
    did = doc_id(url); title = title_of(did)
    ids = [t.strip() for t in (tabs or "").split(",") if t.strip()] if tabs else []
    if not ids and url_tab(url): ids = [url_tab(url)]
    found = discover_tabs(did) if not ids else []
    if found and len(found) > 1: ids = [t for t, _ in found]
    names = dict(found) if found else {}
    out = []
    if ids:
        for t in ids:
            try: txt = export_text(did, t)
            except PermissionError: raise
            except Exception as e: print(f"[gdoc] tab {t}: {e}", file=sys.stderr); continue
            if txt: out.append((f"{title} - {names.get(t) or t}", txt))
    if not out: out.append((title, export_text(did)))
    return out

def safe_name(s): return re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-")[:80] or "doc"

if __name__ == "__main__":
    args = sys.argv[1:]; tabs = None
    if "--tabs" in args: i = args.index("--tabs"); tabs = args[i + 1]; del args[i:i + 2]
    for label, text in fetch(args[0], tabs):
        p = pathlib.Path(safe_name(label) + ".txt"); p.write_text(text + "\n"); print(p, len(text.split()), "words")
