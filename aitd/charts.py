"""SVG chart helpers for the HTML report: violin small-multiples (human vs machine controls with the file marked),
number-line strips, a position bar, and a GLTR token heat map. Pure Python, no dependencies."""
import math, html

FONT = "-apple-system,Helvetica,Arial,sans-serif"
def esc(s): return html.escape(str(s))
def _finite(xs): return [x for x in xs if isinstance(x, (int, float)) and x == x and abs(x) != float("inf")]

def _kde(xs, grid, bw=None):
    xs = _finite(xs)
    if len(xs) < 2: return [0.0] * len(grid)
    n = len(xs); mean = sum(xs) / n; sd = (sum((x - mean) ** 2 for x in xs) / (n - 1)) ** 0.5
    if bw is None: bw = max(1.06 * sd * n ** (-0.2), 1e-9) if sd > 0 else 1e-9
    return [sum(math.exp(-0.5 * ((g - x) / bw) ** 2) for x in xs) / (n * bw * (2 * math.pi) ** 0.5) for g in grid]

def _wrap(title, n=38):
    if len(title) <= n: return [title]
    cut = title.rfind(" ", 0, n)
    return [title[:cut], title[cut + 1:]] if cut > 10 else [title[:n], title[n:]]

def violin(title, human, machine, target, higher_is_human=None, note="", log=False, fmt="{:.3g}", W=300, H=250, auc=None, side=None, cut=None):
    """Two vertical violins (human green, machine red), clipped to each group's range, with the control points, group
    medians, and the file's value as a line across both. The axis is oriented so that UP is always the human direction
    (reversed for lower-is-human measures; tick labels keep the real values). side = "human"/"machine"/None colours the
    file line and the panel; auc < 0.7 dims the panel as a weak measure on these controls."""
    f = (lambda x: math.log10(max(x, 1e-9))) if log else (lambda x: x)
    hv, mv = _finite(human), _finite(machine); tv = target if (isinstance(target, (int, float)) and target == target) else None
    allv = [f(x) for x in hv + mv + ([tv] if tv is not None else [])]
    weak = auc is not None and auc == auc and auc < 0.7
    cls = "violin" + (" vh" if side == "human" else " vm" if side == "machine" else "") + (" weak" if weak else "")
    lines = _wrap(title); top0 = 18 + 13 * len(lines); badge_y = top0 + 16
    out = [f'<svg class="{cls}" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{esc(title)}">']
    for i, ln in enumerate(lines): out.append(f'<text x="8" y="{16 + 13 * i}" font-size="12.5" font-weight="600" font-family="{FONT}" fill="currentColor">{esc(ln)}</text>')
    if not allv: out.append(f'<text x="8" y="{top0 + 14}" font-size="11" fill="currentColor" fill-opacity=".6" font-family="{FONT}">n/a</text></svg>'); return "".join(out)
    lo, hi = min(allv), max(allv); pad = (hi - lo) * 0.10 or (abs(hi) * 0.1 or 0.5); lo, hi = lo - pad, hi + pad
    if not log and min(hv + mv + ([tv] if tv is not None else [])) >= 0: lo = max(lo, 0.0)
    flip = higher_is_human is False
    top, bot, L, R = badge_y + 14, H - 26, 56, 12
    Y = (lambda v: top + (f(v) - lo) / (hi - lo) * (bot - top)) if flip else (lambda v: bot - (f(v) - lo) / (hi - lo) * (bot - top))
    Yf = (lambda g: top + (g - lo) / (hi - lo) * (bot - top)) if flip else (lambda g: bot - (g - lo) / (hi - lo) * (bot - top))
    cx_h, cx_m = L + (W - L - R) * 0.3, L + (W - L - R) * 0.72; half = (W - L - R) * 0.19
    if note:
        out.append(f'<text x="8" y="{top0}" font-size="10" font-family="{FONT}" fill="currentColor" fill-opacity=".65">{esc(note[:50] + ("…" if len(note) > 50 else ""))}</text>')
    badge = {"human": ("human side", "var(--hum)"), "machine": ("machine side", "var(--mach)")}.get(side)
    if badge:
        out.append(f'<rect x="{W-R-84}" y="{badge_y-11}" width="84" height="15" rx="7.5" fill="{badge[1]}"/><text x="{W-R-42}" y="{badge_y}" font-size="9.5" font-weight="700" text-anchor="middle" font-family="{FONT}" fill="#fff">{badge[0]}</text>')
    elif higher_is_human is None:
        out.append(f'<text x="{W-R}" y="{badge_y}" font-size="9.5" text-anchor="end" font-family="{FONT}" fill="currentColor" fill-opacity=".6">no direction claimed</text>')
    if weak: out.append(f'<text x="8" y="{badge_y}" font-size="9.5" font-weight="700" font-family="{FONT}" fill="currentColor" fill-opacity=".8">weak measure here (AUC {auc:.2f})</text>')
    for i in range(5):
        v = lo + (hi - lo) * i / 4; y = Yf(v)
        lab = fmt.format(10 ** v) if log else fmt.format(v)
        out.append(f'<line x1="{L-4}" y1="{y:.1f}" x2="{W-R}" y2="{y:.1f}" stroke="currentColor" stroke-opacity=".12"/><text x="{L-7}" y="{y+3.5:.1f}" font-size="9.5" text-anchor="end" font-family="{FONT}" fill="currentColor" fill-opacity=".7">{lab}</text>')
    if higher_is_human is not None:
        out.append(f'<text x="8" y="{H-9}" font-size="9.5" font-family="{FONT}" fill="var(--hum)">&#9650; human</text>')
    for cx, vals, col, name in ((cx_h, hv, "var(--hum)", "human"), (cx_m, mv, "var(--mach)", "machine")):
        fv = [f(x) for x in vals]
        if len(fv) >= 2 and max(fv) > min(fv):
            g0, g1 = min(fv), max(fv); grid = [g0 + (g1 - g0) * i / 40 for i in range(41)]
            d = _kde(fv, grid); mx = max(d) or 1; ys = [Yf(g) for g in grid]
            left = [(cx - half * k / mx, y) for k, y in zip(d, ys)]; right = [(cx + half * k / mx, y) for k, y in zip(d, ys)][::-1]
            path = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in left + right) + " Z"
            out.append(f'<path d="{path}" fill="{col}" fill-opacity=".28" stroke="{col}" stroke-opacity=".7" stroke-width="1"/>')
        for j, x in enumerate(vals):
            jx = cx + ((j * 7919) % 11 - 5) * (half / 9)
            out.append(f'<circle cx="{jx:.1f}" cy="{Y(x):.1f}" r="2.6" fill="{col}" fill-opacity=".9"><title>{name} control: {fmt.format(x)}</title></circle>')
        if fv:
            s = sorted(fv); med = s[len(s) // 2] if len(s) % 2 else (s[len(s) // 2 - 1] + s[len(s) // 2]) / 2; ym = Yf(med)
            out.append(f'<line x1="{cx-half:.1f}" y1="{ym:.1f}" x2="{cx+half:.1f}" y2="{ym:.1f}" stroke="currentColor" stroke-width="2"><title>{name} median</title></line>')
        out.append(f'<text x="{cx:.1f}" y="{H-9}" font-size="10.5" text-anchor="middle" font-family="{FONT}" fill="currentColor" fill-opacity=".8">{name} ({len(vals)})</text>')
    if cut is not None and cut == cut:
        yc = Y(cut)
        out.append(f'<line x1="{L}" y1="{yc:.1f}" x2="{W-R}" y2="{yc:.1f}" stroke="currentColor" stroke-opacity=".55" stroke-width="1" stroke-dasharray="2 3"><title>control cut-point {fmt.format(cut)}</title></line><text x="{L+2}" y="{yc-3:.1f}" font-size="8.5" font-family="{FONT}" fill="currentColor" fill-opacity=".6">cut {fmt.format(cut)}</text>')
    if tv is not None:
        y = Y(tv); col = badge[1] if badge else "var(--tgt)"
        out.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W-R}" y2="{y:.1f}" stroke="{col}" stroke-width="2.5" stroke-dasharray="6 3"/>'
                   f'<polygon points="{W-R-1},{y:.1f} {W-R-9},{y-5:.1f} {W-R-9},{y+5:.1f}" fill="{col}"/>'
                   f'<text x="{W-R-12}" y="{y-5 if y-5 > top+10 else y+13:.1f}" font-size="10.5" font-weight="700" text-anchor="end" font-family="{FONT}" fill="{col}" stroke="var(--panel)" stroke-width="3" paint-order="stroke">this file {fmt.format(tv)}</text>')
    out.append("</svg>"); return "".join(out)

def strip(title, points, target_val, target_name, log=False, lines=(), W=860, H=96):
    """Horizontal number line: control dots (human above, machine below), file as a diamond, optional threshold lines."""
    vals = [v for _, v, _ in points if v == v] + ([target_val] if target_val == target_val else []) + [v for v, _ in lines]
    f = (lambda x: math.log10(max(x, 1e-9))) if log else (lambda x: x)
    lo, hi = min(map(f, vals)), max(map(f, vals)); pad = (hi - lo) * 0.08 or 0.5; lo, hi = lo - pad, hi + pad
    L, R = 70, 20; X = lambda v: L + (f(v) - lo) / (hi - lo) * (W - L - R)
    s = [f'<svg class=strip viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{esc(title)}">',
         f'<text x="0" y="14" font-size="12.5" font-weight="600" font-family="{FONT}" fill="currentColor">{esc(title)}</text>',
         f'<line x1="{L}" y1="58" x2="{W-R}" y2="58" stroke="currentColor" stroke-opacity=".35"/>']
    for i in range(7):
        v = lo + (hi - lo) * i / 6; x = L + (W - L - R) * i / 6; lab = f"{10**v:.3g}" if log else f"{v:.3g}"
        s.append(f'<line x1="{x:.1f}" y1="54" x2="{x:.1f}" y2="62" stroke="currentColor" stroke-opacity=".35"/><text x="{x:.1f}" y="78" font-size="10" text-anchor="middle" fill="currentColor" fill-opacity=".7" font-family="{FONT}">{lab}</text>')
    for v, name in lines:
        s.append(f'<line x1="{X(v):.1f}" y1="26" x2="{X(v):.1f}" y2="66" stroke="currentColor" stroke-opacity=".6" stroke-dasharray="3 3"/><text x="{X(v):.1f}" y="92" font-size="10" text-anchor="middle" fill="currentColor" fill-opacity=".8" font-family="{FONT}">{esc(name)}</text>')
    for name, v, human in points:
        if v != v: continue
        y = 40 if human else 58
        s.append(f'<circle cx="{X(v):.1f}" cy="{y}" r="5" fill="{"var(--hum)" if human else "var(--mach)"}" fill-opacity=".85"><title>{esc(name)}: {v:.3g}</title></circle>')
    if target_val == target_val:
        x = X(target_val); s.append(f'<polygon points="{x:.1f},38 {x+8:.1f},49 {x:.1f},60 {x-8:.1f},49" fill="var(--tgt)"><title>{esc(target_name)}: {target_val:.3g}</title></polygon>')
    s.append("</svg>"); return "".join(s)

def position_bar(human, machine, target, higher_is_human=True, W=260, H=22):
    """Compact bar for a summary tile: machine range (red) and human range (green) as bands, file as a marker."""
    hv, mv = _finite(human), _finite(machine)
    if not hv or not mv or not isinstance(target, (int, float)) or target != target: return ""
    lo, hi = min(hv + mv + [target]), max(hv + mv + [target]); pad = (hi - lo) * 0.06 or 0.1; lo, hi = lo - pad, hi + pad
    X = lambda v: (v - lo) / (hi - lo) * W
    s = [f'<svg class=posbar viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">',
         f'<rect x="0" y="9" width="{W}" height="4" fill="currentColor" fill-opacity=".15"/>',
         f'<rect x="{X(min(mv)):.1f}" y="6" width="{max(X(max(mv))-X(min(mv)),2):.1f}" height="10" rx="2" fill="var(--mach)" fill-opacity=".5"><title>machine controls {min(mv):.3g} to {max(mv):.3g}</title></rect>',
         f'<rect x="{X(min(hv)):.1f}" y="6" width="{max(X(max(hv))-X(min(hv)),2):.1f}" height="10" rx="2" fill="var(--hum)" fill-opacity=".5"><title>human controls {min(hv):.3g} to {max(hv):.3g}</title></rect>',
         f'<polygon points="{X(target):.1f},1 {X(target)+6:.1f},11 {X(target):.1f},21 {X(target)-6:.1f},11" fill="var(--tgt)"><title>this file {target:.3g}</title></polygon></svg>']
    return "".join(s)

GLTR_CLASSES = [(1, "g0"), (10, "g1"), (100, "g2"), (1000, "g3")]
def gltr_html(text, tokens):
    """Text with each scored token wrapped in a span coloured by the rank bucket of the model's prediction."""
    out, cur = [], 0
    for t in tokens:
        s, e = t["s"], t["e"]
        if s < cur: continue
        out.append(esc(text[cur:s]))
        cls = "g4"
        for lim, c in GLTR_CLASSES:
            if t["rank"] <= lim: cls = c; break
        out.append(f'<span class="tk {cls}" title="rank {t["rank"]}, log p {t["lp"]:.2f}, entropy {t["ent"]:.2f}">{esc(text[s:e])}</span>'); cur = e
    out.append(esc(text[cur:]))
    return "".join(out).replace("\n", "<br>")
