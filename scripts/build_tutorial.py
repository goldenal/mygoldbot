#!/usr/bin/env python3
"""Build docs/tutorial.html: one self-contained page (images, styles and scripts inline, no internet needed) from docs/TUTORIAL.md and docs/img/.

  python scripts/build_tutorial.py        # needs the 'markdown' package:  pip install markdown
  open docs/tutorial.html                 # macOS; or just double-click the file

Edit docs/TUTORIAL.md (and add screenshots to docs/img/), then run this again."""
import base64
import re
import sys
from pathlib import Path

try:
    import markdown
except ImportError:
    sys.exit("this script needs the 'markdown' package: pip install markdown")

DOCS = Path(__file__).resolve().parent.parent / "docs"
SRC, OUT = DOCS / "TUTORIAL.md", DOCS / "tutorial.html"

md = markdown.Markdown(extensions=["tables", "fenced_code", "toc", "sane_lists"])
body = md.convert(SRC.read_text())


def figure(m: re.Match) -> str:
    alt, src = m.group(1), m.group(2)
    data = base64.b64encode((DOCS / src).read_bytes()).decode()
    return (f'<figure><button class="zoom" type="button" aria-label="Enlarge screenshot: {alt}">'
            f'<img src="data:image/png;base64,{data}" alt="{alt}"></button><figcaption>{alt}. <span class="hint">Click to enlarge.</span></figcaption></figure>')


body = re.sub(r'<p><img alt="([^"]*)" src="([^"]+)" ?/?></p>', figure, body)
body = body.replace("<table>", '<div class="table-wrap"><table>').replace("</table>", "</table></div>")
body = body.replace("<blockquote>", '<aside class="callout" role="note">').replace("</blockquote>", "</aside>")

m = re.match(r'\s*<h1 id="[^"]*">(.*?)</h1>(.*?)(<hr ?/?>|<h2 )', body, re.S)
if not m:
    sys.exit("TUTORIAL.md must start with a '# title', an introduction, then '---' or a '## section'")
title, intro, rest = m.group(1), m.group(2), body[m.end(2):]
toc = "".join(f'<li><a href="#{i}">{t}</a></li>' for i, t in re.findall(r'<h2 id="([^"]+)">(.*?)</h2>', rest))

CSS = r"""
:root {
  color-scheme: light dark;
  --bg: #f4f5f6; --surface: #ffffff; --fg: #17191c; --muted: #596068; --line: #dcdfe3; --line-soft: #e9ebee;
  --accent: #9a5b00; --warn-bg: #fff4e0; --warn-line: #e3b25a; --code-bg: #eceef0;
  --shadow: 0 1px 2px rgba(20,24,28,.06), 0 8px 24px rgba(20,24,28,.06);
  --font-display: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  --font-body: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  --font-mono: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root { --bg: #101214; --surface: #171a1d; --fg: #e8eaed; --muted: #9aa2ab; --line: #2a2f35; --line-soft: #22262b; --accent: #e3a93f;
    --warn-bg: #261f12; --warn-line: #7a5a1c; --code-bg: #1e2227; --shadow: 0 1px 2px rgba(0,0,0,.4), 0 8px 24px rgba(0,0,0,.35); }
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
@media (prefers-reduced-motion: reduce) { html { scroll-behavior: auto; } }
body { margin: 0; background: var(--bg); color: var(--fg); font-family: var(--font-body); font-size: 16px; line-height: 1.65; padding: 0 16px 64px; }
a { color: var(--accent); text-underline-offset: 3px; }
a:focus-visible, button:focus-visible, summary:focus-visible { outline: 2px solid var(--accent); outline-offset: 3px; border-radius: 4px; }
.wrap { max-width: 1120px; margin: 0 auto; }

.hero { padding: 44px 0 28px; border-bottom: 1px solid var(--line); margin-bottom: 32px; }
.eyebrow { font-family: var(--font-display); font-size: 12px; font-weight: 600; letter-spacing: .14em; text-transform: uppercase; color: var(--accent); margin: 0 0 10px; }
.hero h1 { font-family: var(--font-display); font-weight: 700; font-size: clamp(30px, 5vw, 46px); line-height: 1.1; letter-spacing: -.025em; margin: 0 0 16px; text-wrap: balance; }
.intro { max-width: 68ch; color: var(--muted); }
.intro p { margin: 0 0 12px; } .intro ol { margin: 0 0 12px; padding-left: 1.2em; } .intro li { margin-bottom: 4px; } .intro strong { color: var(--fg); }

.layout { display: grid; grid-template-columns: minmax(0, 1fr); gap: 32px; }
@media (min-width: 980px) { .layout { grid-template-columns: 220px minmax(0, 1fr); gap: 48px; align-items: start; } }
.toc { display: none; }
@media (min-width: 980px) { .toc { display: block; position: sticky; top: 20px; max-height: calc(100vh - 40px); overflow: auto; } .toc-mobile { display: none; } }
.toc-title { font-family: var(--font-display); font-size: 12px; font-weight: 600; letter-spacing: .14em; text-transform: uppercase; color: var(--muted); margin: 0 0 10px; }
.toc ol, .toc-mobile ol { list-style: none; margin: 0; padding: 0; display: grid; gap: 2px; }
.toc a, .toc-mobile a { display: block; padding: 5px 10px; border-left: 2px solid var(--line); color: var(--muted); text-decoration: none; font-size: 14px; line-height: 1.35; }
.toc a:hover, .toc-mobile a:hover { color: var(--fg); border-left-color: var(--accent); }
.toc-mobile { border: 1px solid var(--line); border-radius: 8px; background: var(--surface); padding: 4px 12px; margin-bottom: 8px; }
.toc-mobile summary { cursor: pointer; padding: 8px 0; font-family: var(--font-display); font-weight: 600; }
.toc-mobile ol { padding-bottom: 10px; }

main { min-width: 0; max-width: 780px; }
main h2 { font-family: var(--font-display); font-weight: 700; font-size: clamp(24px, 3.2vw, 30px); line-height: 1.2; letter-spacing: -.015em; margin: 56px 0 14px; padding-top: 22px; border-top: 1px solid var(--line); text-wrap: balance; scroll-margin-top: 20px; }
main h2:first-of-type { margin-top: 8px; }
main h3 { font-family: var(--font-display); font-weight: 650; font-size: 19px; line-height: 1.3; margin: 34px 0 10px; text-wrap: balance; scroll-margin-top: 20px; }
main p, main li { max-width: 68ch; }
main p { margin: 0 0 14px; }
main ul, main ol { padding-left: 1.3em; margin: 0 0 16px; }
main li { margin-bottom: 6px; }
main hr { display: none; }
code { font-family: var(--font-mono); font-size: .88em; background: var(--code-bg); padding: .08em .3em; border-radius: 4px; }
.code-box { position: relative; margin: 0 0 18px; }
pre { margin: 0; background: var(--code-bg); border: 1px solid var(--line-soft); border-radius: 8px; padding: 14px 16px; overflow-x: auto; font-size: 13.5px; line-height: 1.55; }
pre code { background: none; padding: 0; font-size: inherit; }
.copy { position: absolute; top: 8px; right: 8px; font: 12px var(--font-body); color: var(--muted); background: var(--surface); border: 1px solid var(--line); border-radius: 6px; padding: 3px 9px; cursor: pointer; }
.copy:hover { color: var(--fg); border-color: var(--muted); }

.table-wrap { overflow-x: auto; margin: 0 0 20px; border: 1px solid var(--line); border-radius: 8px; background: var(--surface); }
table { border-collapse: collapse; width: 100%; font-size: 14.5px; line-height: 1.5; }
th { text-align: left; font-family: var(--font-display); font-weight: 600; font-size: 12.5px; letter-spacing: .04em; text-transform: uppercase; color: var(--muted); padding: 10px 14px; border-bottom: 1px solid var(--line); background: var(--code-bg); }
td { padding: 10px 14px; border-bottom: 1px solid var(--line-soft); vertical-align: top; }
tr:last-child td { border-bottom: 0; }
td:first-child { white-space: nowrap; font-weight: 500; }
@media (max-width: 560px) { td:first-child { white-space: normal; } }

.callout { margin: 0 0 20px; padding: 14px 18px; background: var(--warn-bg); border: 1px solid var(--warn-line); border-radius: 8px; }
.callout p { margin: 0 0 8px; max-width: none; } .callout p:last-child { margin-bottom: 0; }
.hero .callout { margin-top: 20px; max-width: 68ch; color: var(--fg); }

figure { margin: 22px 0 28px; }
figure .zoom { display: block; width: 100%; padding: 0; margin: 0; border: 1px solid var(--line); border-radius: 10px; overflow: hidden; background: var(--surface); box-shadow: var(--shadow); cursor: zoom-in; }
figure img { display: block; width: 100%; height: auto; max-width: 100%; }
figcaption { margin-top: 8px; font-size: 13.5px; color: var(--muted); }
.hint { opacity: .8; }
.lightbox { position: fixed; inset: 0; z-index: 50; background: rgba(8,10,12,.86); display: grid; place-items: center; padding: 16px; cursor: zoom-out; overflow: auto; }
.lightbox img { max-width: min(1400px, 100%); max-height: 100%; width: auto; height: auto; border-radius: 8px; box-shadow: 0 20px 60px rgba(0,0,0,.5); }
.foot { max-width: 780px; margin-top: 56px; padding-top: 20px; border-top: 1px solid var(--line); color: var(--muted); font-size: 14px; }

@media print {
  body { background: #fff; color: #000; padding: 0; font-size: 11pt; }
  .toc, .toc-mobile, .copy, .hint { display: none !important; }
  .layout { display: block; } main { max-width: none; }
  figure, pre, .table-wrap, .callout { break-inside: avoid; }
  figure .zoom { box-shadow: none; }
  a { color: inherit; }
}
"""

JS = r"""
(function () {
  document.querySelectorAll("main pre").forEach(function (pre) {
    var box = document.createElement("div"); box.className = "code-box";
    pre.parentNode.insertBefore(box, pre); box.appendChild(pre);
    var b = document.createElement("button"); b.type = "button"; b.className = "copy"; b.textContent = "Copy"; box.appendChild(b);
    function say(t) { b.textContent = t; setTimeout(function () { b.textContent = "Copy"; }, 1500); }
    function select() { var r = document.createRange(); r.selectNodeContents(pre); var s = getSelection(); s.removeAllRanges(); s.addRange(r); say("Selected"); }
    b.addEventListener("click", function () {
      try { navigator.clipboard.writeText(pre.innerText).then(function () { say("Copied"); }, select); } catch (e) { select(); }
    });
  });
  var lb = null;
  function close() { if (lb) { lb.remove(); lb = null; } }
  document.querySelectorAll("figure .zoom").forEach(function (z) {
    z.addEventListener("click", function () {
      close();
      lb = document.createElement("div"); lb.className = "lightbox"; lb.setAttribute("role", "dialog"); lb.setAttribute("aria-label", "Screenshot");
      var im = document.createElement("img"); var src = z.querySelector("img"); im.src = src.src; im.alt = src.alt; lb.appendChild(im);
      lb.addEventListener("click", close); document.body.appendChild(lb);
    });
  });
  document.addEventListener("keydown", function (e) { if (e.key === "Escape") close(); });
})();
"""

page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>goldsys Tutorial</title>
<link rel="icon" href="data:,">
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
  <header class="hero">
    <p class="eyebrow">Field guide</p>
    <h1>{title}</h1>
    <div class="intro">{intro}</div>
  </header>
  <div class="layout">
    <nav class="toc" aria-label="Contents"><p class="toc-title">Contents</p><ol>{toc}</ol></nav>
    <div>
      <details class="toc-mobile"><summary>Contents</summary><ol>{toc}</ol></details>
      <main>{rest}</main>
      <p class="foot">Screenshots are from the real XAUUSD history imported from Deriv MT5. Times are UTC. Nothing here has been validated as profitable.
      This page is generated from docs/TUTORIAL.md by scripts/build_tutorial.py and works offline.</p>
    </div>
  </div>
</div>
<script>{JS}</script>
</body>
</html>
"""
OUT.write_text(page)
print(f"wrote {OUT} ({len(page) / 1e6:.2f} MB, {page.count('<figure>')} screenshots, {len(re.findall('<h2 ', page))} sections)")
