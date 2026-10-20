"""
readme_to_html.py -- render this folder's README.md as a standalone README.html for viewing in a browser.

Reuses ../multi_agent/readme_to_html.py unchanged (Markdown + MathJax, embedded figures, floating
contents tree, light theme), pointed at this folder and with this page's title.

USAGE (from multi_agent_interaction/)
    python readme_to_html.py            # README.md -> README.html
Needs: pip install markdown   (MathJax is loaded from a CDN when the page is opened)
"""
from __future__ import annotations

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_src = os.path.join(os.path.dirname(HERE), "multi_agent", "readme_to_html.py")
_spec = importlib.util.spec_from_file_location("readme_to_html_multi_agent", _src)   # by path: same module name here
R2H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R2H)

R2H.HERE = HERE                                                       # read README.md / embed figures from this folder
R2H.PAGE = R2H.PAGE.replace("<title>Multi-agent scaling</title>", "<title>Interacting agents</title>")

if __name__ == "__main__":
    sys.exit(R2H.main())
