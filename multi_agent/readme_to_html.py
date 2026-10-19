"""
readme_to_html.py -- render README.md as a standalone README.html for viewing in a browser.

  * Markdown -> HTML with tables, fenced code and heading anchors (the `markdown` package);
  * LaTeX ($...$ and $$...$$) is protected from the Markdown parser and typeset by MathJax;
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
:root {{ --bg: #ffffff; --fg: #1f2328; --muted: #59636e; --line: #d1d9e0; --code: #f6f8fa; --link: #0969da; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg: #0d1117; --fg: #e6edf3; --muted: #9198a1; --line: #3d444d; --code: #151b23; --link: #4493f8; }}
}}
html {{ background: var(--bg); }}
body {{ background: var(--bg); color: var(--fg); margin: 0 auto; max-width: 980px; padding: 32px 16px 96px;
       font: 16px/1.6 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }}
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
{body}
<hr><p class="note">Generated from README.md by readme_to_html.py.</p>
</body>
</html>
"""


def main():
    src = os.path.join(HERE, "README.md")
    out = os.path.join(HERE, "README.html")
    text, store = protect_math(open(src, encoding="utf-8").read())
    md = markdown.Markdown(extensions=["tables", "fenced_code", "toc"],
                           extension_configs={"toc": {"slugify": lambda s, sep: re.sub(r"[^\w\- ]", "", s).strip().lower().replace(" ", "-")}})
    html = md.convert(text)
    html = embed_images(restore_math(html, store))
    html = re.sub(r'href="([^"#:]+)\.md"', r'href="\1.html"', html)
    open(out, "w", encoding="utf-8").write(PAGE.format(body=html))
    print(f"wrote {out} ({os.path.getsize(out) / 1e3:.0f} kB, {len(store)} math expressions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
