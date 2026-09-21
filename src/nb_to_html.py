"""Render a notebook to a single self-contained HTML file.

WHY THIS EXISTS. A `.ipynb` is a data format, not a document. GitHub, JupyterLab and VS Code all
render it, but opening the file straight from disk in a browser shows raw JSON, which makes the
analysis unreadable for anyone who just wants to look at it. Jupyter itself is not a dependency of
this project and `nbconvert` is not installed, so this does the one job that is actually needed:
notebook in, one HTML file out, images and all, nothing to install.

    python src/nb_to_html.py                       # notebooks/01_analysis.ipynb -> .html
    python src/nb_to_html.py path/to/other.ipynb

The markdown subset covered is the subset this project's notebooks use: headings, bold, italic,
inline and fenced code, links, bullet and numbered lists, tables, rules, block quotes and display
maths. Maths is handed to MathJax from a CDN; everything else renders offline.
"""
from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------------- markdown subset

def _inline(t: str) -> str:
    """Inline spans. Code is extracted first so nothing formats inside it."""
    holds: list[str] = []

    def hold(m):
        holds.append(m.group(1))
        return "\x00%d\x00" % (len(holds) - 1)

    t = re.sub(r"`([^`]+)`", hold, t)
    t = html.escape(t, quote=False)
    t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"<em>\1</em>", t)
    t = re.sub(r"\$([^$\n]+)\$", r"\\(\1\\)", t)          # inline maths for MathJax
    for i, c in enumerate(holds):
        t = t.replace("\x00%d\x00" % i, "<code>%s</code>" % html.escape(c, quote=False))
    return t


def _table(block: list[str]) -> str:
    rows = [[c.strip() for c in r.strip().strip("|").split("|")] for r in block]
    body = [r for r in rows[2:]] if len(rows) > 1 and set(rows[1][0]) <= set("-: ") else rows[1:]
    out = ["<table><thead><tr>"]
    out += ["<th>%s</th>" % _inline(c) for c in rows[0]]
    out.append("</tr></thead><tbody>")
    for r in body:
        out.append("<tr>" + "".join("<td>%s</td>" % _inline(c) for c in r) + "</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def markdown(src: str) -> str:
    lines = src.split("\n")
    out, i = [], 0
    while i < len(lines):
        ln = lines[i]

        if not ln.strip():
            i += 1
            continue

        if ln.strip().startswith("$$"):                    # display maths
            # A one-line block closes itself: "$$ ... $$". Scanning ahead for a closing delimiter
            # in that case runs off the end of the cell and swallows the prose that follows the
            # formula, which is how a paragraph ended up inside a maths div with its ** markers
            # showing through unrendered.
            if ln.strip().count("$$") >= 2:
                out.append('<div class="math">%s</div>'
                           % html.escape(ln.strip(), quote=False))
                i += 1
                continue
            buf = [ln]
            i += 1
            while i < len(lines) and "$$" not in lines[i]:
                buf.append(lines[i]); i += 1
            if i < len(lines):
                buf.append(lines[i]); i += 1
            out.append('<div class="math">%s</div>'
                       % html.escape("\n".join(buf), quote=False))
            continue

        if re.match(r"^\s*```", ln):                       # fenced code
            i += 1
            buf = []
            while i < len(lines) and not re.match(r"^\s*```", lines[i]):
                buf.append(lines[i]); i += 1
            i += 1
            out.append("<pre class='md'><code>%s</code></pre>"
                       % html.escape("\n".join(buf), quote=False))
            continue

        if re.match(r"^\s*(---+|\*\*\*+)\s*$", ln):
            out.append("<hr>"); i += 1; continue

        m = re.match(r"^(#{1,6})\s+(.*)$", ln)
        if m:
            lv = len(m.group(1))
            out.append("<h%d>%s</h%d>" % (lv, _inline(m.group(2)), lv))
            i += 1
            continue

        if ln.lstrip().startswith("|") and i + 1 < len(lines) and lines[i + 1].lstrip().startswith("|"):
            buf = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                buf.append(lines[i]); i += 1
            out.append(_table(buf))
            continue

        if re.match(r"^\s*>\s?", ln):
            buf = []
            while i < len(lines) and re.match(r"^\s*>\s?", lines[i]):
                buf.append(re.sub(r"^\s*>\s?", "", lines[i])); i += 1
            out.append("<blockquote>%s</blockquote>" % markdown("\n".join(buf)))
            continue

        m = re.match(r"^\s*(\d+)\.\s+(.*)$", ln)
        if m:
            buf = []
            while i < len(lines) and (re.match(r"^\s*\d+\.\s+", lines[i])
                                      or (lines[i].startswith("   ") and lines[i].strip())):
                if re.match(r"^\s*\d+\.\s+", lines[i]):
                    buf.append(re.sub(r"^\s*\d+\.\s+", "", lines[i]))
                else:
                    buf[-1] += " " + lines[i].strip()
                i += 1
            out.append("<ol>" + "".join("<li>%s</li>" % _inline(b) for b in buf) + "</ol>")
            continue

        if re.match(r"^\s*[-*]\s+", ln):
            buf = []
            while i < len(lines) and (re.match(r"^\s*[-*]\s+", lines[i])
                                      or (lines[i].startswith("  ") and lines[i].strip())):
                if re.match(r"^\s*[-*]\s+", lines[i]):
                    buf.append(re.sub(r"^\s*[-*]\s+", "", lines[i]))
                else:
                    buf[-1] += " " + lines[i].strip()
                i += 1
            out.append("<ul>" + "".join("<li>%s</li>" % _inline(b) for b in buf) + "</ul>")
            continue

        buf = []
        while i < len(lines) and lines[i].strip() and not re.match(
                r"^\s*(#{1,6}\s|```|\||>|[-*]\s|\d+\.\s|---+\s*$|\$\$)", lines[i]):
            buf.append(lines[i]); i += 1
        if buf:
            out.append("<p>%s</p>" % _inline(" ".join(buf)))
        else:
            i += 1
    return "\n".join(out)


# ------------------------------------------------------------------------------------- outputs

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def render_output(o: dict) -> str:
    t = o.get("output_type")
    if t == "stream":
        txt = "".join(o.get("text", []))
        return "<pre class='out'>%s</pre>" % html.escape(ANSI.sub("", txt), quote=False)
    if t in ("execute_result", "display_data"):
        d = o.get("data", {})
        if "image/png" in d:
            png = d["image/png"]
            png = png if isinstance(png, str) else "".join(png)
            return ('<div class="fig"><img alt="figure" src="data:image/png;base64,%s"></div>'
                    % png.replace("\n", ""))
        if "text/html" in d:
            return "<div class='tbl'>%s</div>" % "".join(d["text/html"])
        if "text/plain" in d:
            return "<pre class='out'>%s</pre>" % html.escape("".join(d["text/plain"]),
                                                             quote=False)
    if t == "error":
        return ("<pre class='err'>%s</pre>"
                % html.escape(ANSI.sub("", "\n".join(o.get("traceback", []))), quote=False))
    return ""


CSS = (ROOT / "web/static/theme.css").read_text(encoding="utf-8") + (
    ROOT / "web/static/notebook.css").read_text(encoding="utf-8")


def convert(nb_path: Path, out_path: Path) -> Path:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    title = "Notebook"
    for c in nb["cells"]:
        if c["cell_type"] == "markdown":
            m = re.search(r"^#\s+(.+)$", "".join(c["source"]), re.M)
            if m:
                title = m.group(1).strip()
                break

    parts = []
    for c in nb["cells"]:
        src = "".join(c["source"])
        if c["cell_type"] == "markdown":
            parts.append('<div class="cell">%s</div>' % markdown(src))
        elif c["cell_type"] == "code":
            if not src.strip():
                continue
            body = ['<div class="cell"><div class="src">%s</div>'
                    % html.escape(src.rstrip(), quote=False)]
            for o in c.get("outputs", []):
                body.append(render_output(o))
            body.append("</div>")
            parts.append("".join(body))

    doc = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>%s</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500&family=DM+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>%s</style>
<script>%s</script>
<link rel="icon" href="data:image/svg+xml,%s" type="image/svg+xml">
<script>window.MathJax={tex:{inlineMath:[["\\\\(","\\\\)"]],displayMath:[["$$","$$"]]}};</script>
<script async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-mml-chtml.min.js"></script>
</head><body><div class="wrap">
<header class="app-header"><a class="app-brand" href="/" id="app-home"><svg viewBox="0 0 36 36" aria-hidden="true"><path d="M4 27 14 10l6 10 4-6 8 13M9 27l5-8 5 8M4 32h28"/></svg><span>Rangamati<small>Analysis notebook</small></span></a><button class="theme-button" id="themer" type="button" aria-label="Switch color theme" style="margin-left:auto">Theme</button></header>
<div class="banner">Analysis notebook &middot; <b>%s</b><br>The data, evaluation, and decisions behind the susceptibility model.</div>
<script>if(location.protocol==='file:')document.getElementById('app-home').href='http://127.0.0.1:5000/';</script>
%s
</div></body></html>""" % (html.escape(title), CSS,
                           (ROOT / "web/static/theme.js").read_text(encoding="utf-8"),
                           quote((ROOT / "web/static/favicon.svg").read_text(encoding="utf-8")),
                           html.escape(nb_path.name),
                           "\n".join(parts))

    out_path.write_text(doc, encoding="utf-8")

    # Unrendered markdown is invisible unless you go looking for it, and it always means a
    # renderer bug rather than a content mistake. Say so instead of shipping a page with raw
    # ** markers in the prose. Code blocks legitimately contain them, so they are excluded.
    body = re.sub(r"<pre.*?</pre>", "", doc, flags=re.S)
    stray = len(re.findall(r"\*\*", body))
    if stray:
        print("WARNING: %d unrendered '**' markers remain; a markdown construct is unhandled"
              % stray)
    return out_path


if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "notebooks" / "01_analysis.ipynb"
    if not src.exists():
        sys.exit("not found: %s" % src)
    out = convert(src, src.with_suffix(".html"))
    print("wrote %s  (%.0f KB)" % (out, out.stat().st_size / 1024))
    print("open it directly in a browser")
