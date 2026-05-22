#!/usr/bin/env python3
"""Build RESEARCH_PROPOSAL_RU.html with embedded figures for PDF print."""

import base64
import re
from pathlib import Path

VAULT_ROOT = Path(__file__).resolve().parent.parent.parent
ROOT = VAULT_ROOT
md_path = VAULT_ROOT / "RESEARCH_PROPOSAL_RU.md"
md = md_path.read_text(encoding="utf-8")


def embed_img(rel: str) -> str:
    p = ROOT / rel
    b64 = base64.b64encode(p.read_bytes()).decode()
    return f"data:image/png;base64,{b64}"


def md_images_to_html(text: str) -> str:
    def wiki_repl(m: re.Match[str]) -> str:
        path = f"Files/{m.group(1)}"
        src = embed_img(path)
        return f'<figure style="margin:16px 0;"><img src="{src}" style="max-width:100%;" alt="{m.group(1)}"/></figure>'

    def md_repl(m: re.Match[str]) -> str:
        cap, path = m.group(1), m.group(2)
        if not path.startswith("Files/"):
            path = f"Files/{path}"
        src = embed_img(path)
        return (
            f'<figure style="margin:16px 0;">'
            f'<figcaption style="font-weight:bold;margin-bottom:6px;">{cap}</figcaption>'
            f'<img src="{src}" style="max-width:100%;" alt="{cap}"/>'
            f"</figure>"
        )

    text = re.sub(r"!\[\[(?:Files/)?([^\]]+)\]\]", wiki_repl, text)
    return re.sub(r"!\[([^\]]*)\]\((Files/[^)]+)\)", md_repl, text)


lines = md_images_to_html(md).splitlines()
html_parts: list[str] = []
in_table = False
table_rows: list[str] = []

for line in lines:
    s = line.strip()
    if s.startswith("|"):
        if not in_table:
            in_table = True
            table_rows = []
        if re.match(r"^\|[-: |]+\|$", s):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        table_rows.append(cells)
        continue
    if in_table:
        html_parts.append('<table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse;margin:12px 0;font-size:10pt;">')
        for i, row in enumerate(table_rows):
            tag = "th" if i == 0 else "td"
            html_parts.append("<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in row) + "</tr>")
        html_parts.append("</table>")
        in_table = False
        table_rows = []

    if s == "---":
        html_parts.append("<hr/>")
    elif s.startswith("# "):
        html_parts.append(f"<h1>{s[2:]}</h1>")
    elif s.startswith("## "):
        html_parts.append(f"<h2>{s[3:]}</h2>")
    elif s.startswith("### "):
        html_parts.append(f"<h3>{s[4:]}</h3>")
    elif s.startswith("- "):
        html_parts.append(f"<li>{s[2:]}</li>")
    elif s.startswith("**Рис."):
        html_parts.append(f"<p><strong>{s.strip('*')}</strong></p>" if "**" in s else f"<p>{s}</p>")
    elif s.startswith("![") or s.startswith("<figure"):
        if s.startswith("<figure"):
            html_parts.append(s)
        continue  # wiki images already converted
    elif s:
        t = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
        t = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", t)
        t = re.sub(
            r"\[([^\]]+)\]\(([^)]+)\)",
            r'<a href="\2">\1</a>',
            t,
        )
        html_parts.append(f"<p>{t}</p>")

if in_table and table_rows:
    html_parts.append('<table border="1" cellpadding="6" style="border-collapse:collapse;">')
    for i, row in enumerate(table_rows):
        tag = "th" if i == 0 else "td"
        html_parts.append("<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in row) + "</tr>")
    html_parts.append("</table>")

body = "\n".join(html_parts)
html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8"/>
  <title>Research Proposal — Ермаков Лаврентий Андреевич</title>
  <style>
    body {{ font-family: "Times New Roman", Georgia, serif; max-width: 210mm; margin: 20mm auto; padding: 0 15mm; line-height: 1.4; font-size: 11pt; color: #111; }}
    h1 {{ font-size: 16pt; text-align: center; }}
    h2 {{ font-size: 13pt; margin-top: 1em; }}
    h3 {{ font-size: 11.5pt; }}
    table {{ width: 100%; }}
    figure {{ page-break-inside: avoid; }}
    @media print {{ body {{ margin: 15mm; }} }}
  </style>
</head>
<body>
{body}
</body>
</html>
"""
out = ROOT / "RESEARCH_PROPOSAL_RU.html"
out.write_text(html, encoding="utf-8")
print(f"Wrote {out} ({out.stat().st_size // 1024} KB)")
