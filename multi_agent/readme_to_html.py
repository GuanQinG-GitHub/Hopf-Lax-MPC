"""
readme_to_html.py -- render README.md as a standalone README.html for viewing in a browser.

  * Markdown -> HTML with tables, fenced code and heading anchors (the `markdown` package);
  * LaTeX ($...$ and $$...$$) is protected from the Markdown parser and typeset by MathJax;
  * a floating contents tree on the right (wide screens), highlighting the current section;
  * light theme;
  * images are embedded (base64), so README.html is self-contained; links to scripts and figures stay
    relative and open from this folder.

USAGE (from multi_agent/)
    python readme_to_html.py            # README.md -> README.html
Needs: pip install markdown   (MathJax is loaded from a CDN when the page is opened)
"""
from __future__ import annotations

import base64
import mimetypes
import os
import re
import sys

import markdown

HERE = os.path.dirname(os.path.abspath(__file__))


def protect_math(text):
    """Swap every $$...$$ and $...$ for a placeholder so Markdown does not touch the LaTeX."""
    store = []

    def keep(m):
        store.append(m.group(0))
        return f"@@MATH{len(store) - 1}@@"

    parts = re.split(r"(```.*?```|`[^`\n]*`)", text, flags=re.S)        # leave code untouched
    for i in range(0, len(parts), 2):
        s = re.sub(r"\$\$.+?\$\$", keep, parts[i], flags=re.S)
        s = re.sub(r"(?<![\\$])\$(?!\$)[^$\n]+?(?<!\\)\$", keep, s)
        parts[i] = s
    return "".join(parts), store


def restore_math(html, store):
    def put(m):
        tex = store[int(m.group(1))]
        tex = tex.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return tex
    return re.sub(r"@@MATH(\d+)@@", put, html)


def embed_images(html):
    def emb(m):
        src = m.group(2)
        path = os.path.join(HERE, src)
        if src.startswith(("http:", "https:", "data:")) or not os.path.exists(path):
            return m.group(0)
        mime = mimetypes.guess_type(path)[0] or "image/png"
        data = base64.b64encode(open(path, "rb").read()).decode()
        return f'{m.group(1)}data:{mime};base64,{data}{m.group(3)}'
    return re.sub(r'(<img[^>]*src=")([^"]+)(")', emb, html)


PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Multi-agent scaling</title>
<script>
window.MathJax = {{ tex: {{ inlineMath: [['$', '$']], displayMath: [['$$', '$$']] }},
                   options: {{ skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'] }} }};
</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
<style>
:root {{ --bg: #ffffff; --fg: #1f2328; --muted: #59636e; --line: #d1d9e0; --code: #f6f8fa; --link: #0969da;
         --navw: 280px; --active: #ddf4ff; }}
html {{ background: var(--bg); color-scheme: light; scroll-padding-top: 16px; }}
body {{ background: var(--bg); color: var(--fg); margin: 0;
       font: 16px/1.6 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }}
main {{ max-width: 980px; margin: 0 auto; padding: 32px 16px 96px; }}
@media (min-width: 1320px) {{ main {{ margin-right: calc(var(--navw) + 40px); margin-left: auto; }} }}
nav.toc {{ position: fixed; top: 16px; right: 16px; width: var(--navw); max-height: calc(100vh - 32px); overflow-y: auto;
          background: #fbfcfd; border: 1px solid var(--line); border-radius: 8px; padding: 12px 10px;
          font-size: 13.5px; line-height: 1.45; box-shadow: 0 1px 3px rgba(31,35,40,.06); }}
nav.toc .ttl {{ font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: .06em; color: var(--muted);
               padding: 0 8px 6px; }}
nav.toc ul {{ list-style: none; margin: 0; padding-left: 12px; }}
nav.toc > div > ul {{ padding-left: 0; }}
nav.toc li {{ margin: 1px 0; }}
nav.toc a {{ display: block; padding: 3px 8px; border-radius: 6px; color: var(--fg); border-left: 2px solid transparent; }}
nav.toc a:hover {{ background: var(--code); text-decoration: none; }}
nav.toc a.active {{ background: var(--active); border-left-color: var(--link); color: var(--link); font-weight: 600; }}
@media (max-width: 1319px) {{ nav.toc {{ display: none; }} }}
h1, h2 {{ border-bottom: 1px solid var(--line); padding-bottom: .3em; }}
h1 {{ font-size: 2em; }} h2 {{ font-size: 1.5em; margin-top: 2em; }}
a {{ color: var(--link); text-decoration: none; }} a:hover {{ text-decoration: underline; }}
code {{ background: var(--code); padding: .15em .35em; border-radius: 6px; font-size: 85%;
        font-family: ui-monospace, Consolas, monospace; }}
pre {{ background: var(--code); padding: 14px 16px; border-radius: 6px; overflow-x: auto; }}
pre code {{ background: none; padding: 0; font-size: 85%; }}
table {{ border-collapse: collapse; display: block; overflow-x: auto; margin: 1em 0; }}
th, td {{ border: 1px solid var(--line); padding: 6px 12px; text-align: left; vertical-align: top; }}
th {{ background: var(--code); }}
img {{ max-width: 100%; background: #fff; border-radius: 4px; }}
mjx-container[display="true"] {{ overflow-x: auto; overflow-y: hidden; padding: 4px 0; }}
hr {{ border: 0; border-top: 1px solid var(--line); margin: 2em 0; }}
.note {{ color: var(--muted); font-size: 90%; }}
</style>
</head>
<body>
<nav class="toc" aria-label="Contents"><div class="ttl">Contents</div><div>{toc}</div></nav>
<main>
{body}
<hr><p class="note">Generated from README.md by readme_to_html.py.</p>
</main>
<script>
(function () {{
  var links = Array.prototype.slice.call(document.querySelectorAll('nav.toc a[href^="#"]'));
  var targets = links.map(function (a) {{ return document.getElementById(decodeURIComponent(a.getAttribute('href').slice(1))); }});
  function spy() {{
    var y = window.scrollY + 80, cur = 0;
    for (var i = 0; i < targets.length; i++) {{ if (targets[i] && targets[i].offsetTop <= y) cur = i; }}
    links.forEach(function (a, i) {{ a.classList.toggle('active', i === cur); }});
  }}
  window.addEventListener('scroll', spy, {{ passive: true }}); window.addEventListener('load', spy); spy();
}})();
</script>
</body>
</html>
"""


def main():
    src = os.path.join(HERE, "README.md")
    out = os.path.join(HERE, "README.html")
    text, store = protect_math(open(src, encoding="utf-8").read())
    md = markdown.Markdown(extensions=["tables", "fenced_code", "toc"],
                           extension_configs={"toc": {"toc_depth": "2-3", "slugify": lambda s, sep: re.sub(r"[^\w\- ]", "", s).strip().lower().replace(" ", "-")}})
    html = md.convert(text)
    html = embed_images(restore_math(html, store))
    html = re.sub(r'href="([^"#:]+)\.md"', r'href="\1.html"', html)
    toc = restore_math(md.toc, store)
    toc = re.sub(r'^<div class="toc">\s*|\s*</div>\s*$', '', toc.strip())
    open(out, "w", encoding="utf-8").write(PAGE.format(body=html, toc=toc))
    print(f"wrote {out} ({os.path.getsize(out) / 1e3:.0f} kB, {len(store)} math expressions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
