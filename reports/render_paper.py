"""Render a report markdown (with TITLE/AUTHOR comments) to paper-style HTML."""
import markdown, os, base64, re, html as htmlmod, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

CSS = """
<style>
@page { size: A4; margin: 2.0cm 2.1cm; }
body { font-family: Georgia,'Times New Roman',serif; font-size:10.5pt; line-height:1.42; color:#111; text-align:justify; }
.title { text-align:center; font-size:17.5pt; font-weight:bold; line-height:1.25; margin:0 0 .6em; }
table.authors { margin:0 auto 1.4em; border-collapse:collapse; }
table.authors td { border:none !important; padding:0 1.5em; text-align:center; vertical-align:top; }
.authors .nm { font-weight:bold; font-size:11pt; }
.authors .af { font-size:10pt; } .authors .em { font-family:'Courier New',monospace; font-size:9.5pt; }
h2 { font-size:12pt; font-weight:bold; margin:1.3em 0 .4em; page-break-after:avoid; }
h2.abs { text-align:center; font-size:11.5pt; margin-bottom:.3em; }
h3 { font-size:11pt; font-weight:bold; font-style:italic; margin:1.0em 0 .3em; }
p { margin:.5em 0; }
div.abstract { margin:0 1.6em; font-size:10pt; }
code { font-family:'Courier New',monospace; font-size:9pt; background:#f3f3f3; padding:0 2px; }
pre { font-family:'Courier New',monospace; font-size:8.5pt; background:#f6f6f6; border:1px solid #e3e3e3; padding:.6em; overflow-x:auto; }
table { border-collapse:collapse; margin:.4em auto 1.0em; font-size:9.5pt; page-break-inside:avoid; }
th,td { border:1px solid #999; padding:3px 8px; text-align:center; }
p.cap { text-align:center; font-size:9.5pt; font-style:italic; margin:.2em 0 .4em; }
figure { text-align:center; margin:1.0em 0; }
figure img { max-width:88%; border:1px solid #ccc; }
figcaption { font-size:9.5pt; font-style:italic; text-align:center; margin-top:.3em; padding:0 1.2em; }
</style>
"""

def build(name):
    raw = open(os.path.join(HERE, f"{name}.md")).read()
    title = re.search(r"<!--TITLE:(.*?)-->", raw).group(1).strip()
    au = re.search(r"<!--AUTHOR:(.*?)-->", raw, re.DOTALL).group(1).strip()
    raw = re.sub(r"<!--.*?-->", "", raw, flags=re.DOTALL)
    def emb(m):
        alt, path = m.group(1), m.group(2)
        full = path if os.path.isabs(path) else os.path.join(ROOT, path)
        if os.path.exists(full):
            return f'![{alt}](data:image/png;base64,{base64.b64encode(open(full,"rb").read()).decode()})'
        return m.group(0)
    raw = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', emb, raw)
    body = markdown.markdown(raw, extensions=["tables", "fenced_code", "sane_lists"])
    def figrepl(m):
        alt = re.search(r'alt="([^"]*)"', m.group(0)); src = re.search(r'src="([^"]*)"', m.group(0))
        cap = htmlmod.unescape(alt.group(1)) if alt else ""
        return f'<figure><img src="{src.group(1)}"><figcaption>{cap}</figcaption></figure>'
    body = re.sub(r'<p>\s*<img[^>]*>\s*</p>', figrepl, body)
    body = body.replace("<h2>Abstract</h2>", '<h2 class="abs">Abstract</h2>')
    body = re.sub(r'(<h2 class="abs">Abstract</h2>\s*)(<p>.*?</p>)',
                  r'\1<div class="abstract">\2</div>', body, flags=re.DOTALL)
    body = re.sub(r'<p>(Table \d+\..*?)</p>', r'<p class="cap">\1</p>', body, flags=re.DOTALL)
    cells = ""
    for a in au.split("##"):
        pr = [x.strip() for x in a.strip().split("|")]
        if len(pr) >= 3:
            cells += f'<td><span class="nm">{pr[0]}</span><br><span class="af">{pr[1]}</span><br><span class="em">{pr[2]}</span></td>'
    head = f'<div class="title">{title}</div><table class="authors"><tr>{cells}</tr></table>'
    open(os.path.join(HERE, f"{name}.html"), "w").write(
        f"<!doctype html><html><head><meta charset='utf-8'>{CSS}</head><body>{head}{body}</body></html>")
    print("built", name)

if __name__ == "__main__":
    names = sys.argv[1:] or ["midterm_report", "final_report"]
    for n in names:
        build(n)
