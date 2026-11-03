"""Build an editable PPTX from the Marp slide markdown (text + tables + images + notes)."""
import os, re
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

HERE = os.path.dirname(os.path.abspath(__file__))
MD = os.path.join(HERE, "final_presentation.md")
OUT = os.path.join(HERE, "final_presentation.pptx")

GREEN = RGBColor(0x11, 0x99, 0x55)
DARK = RGBColor(0x22, 0x22, 0x22)
GREY = RGBColor(0x55, 0x55, 0x55)

def clean(t):
    t = re.sub(r"\*\*(.+?)\*\*", r"\1", t)
    t = re.sub(r"`(.+?)`", r"\1", t)
    t = t.replace("$$", "").replace("$", "").replace("\\;", " ")
    t = t.replace("\\gamma", "γ").replace("\\alpha", "α").replace("\\arg\\max", "argmax")
    t = re.sub(r"\\frac\{(.+?)\}\{(.+?)\}", r"(\1)/(\2)", t)
    t = re.sub(r"_\\text\{(.+?)\}", r"_\1", t).replace("\\cdot", "·")
    return t.strip()

def parse_slides(md):
    parts = re.split(r"(?m)^---\s*$", md)
    # parts[0] empty, parts[1] frontmatter, rest are slides
    return [p for p in parts[2:] if p.strip()]

def slide_bits(raw):
    notes = "\n".join(clean(x) for x in re.findall(r"<!--(.*?)-->", raw, re.DOTALL))
    body = re.sub(r"<!--.*?-->", "", raw, flags=re.DOTALL)
    imgs = [os.path.normpath(os.path.join(HERE, m)) for m in
            re.findall(r"!\[[^\]]*\]\(([^)]+)\)", body)]
    imgs = [p for p in imgs if os.path.exists(p)]
    body = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", body)
    title = ""
    lines, table = [], []
    for ln in body.splitlines():
        s = ln.rstrip()
        if not s.strip():
            continue
        if s.lstrip().startswith("#"):
            if not title:
                title = clean(s.lstrip("#").strip())
            else:
                lines.append(("h", clean(s.lstrip("#").strip())))
            continue
        if s.lstrip().startswith("|"):
            table.append(s); continue
        if s.lstrip().startswith(">"):
            lines.append(("quote", clean(s.lstrip().lstrip(">").strip()))); continue
        m = re.match(r"^(\s*)[-*]\s+(.*)", s)
        if m:
            lvl = 1 if len(m.group(1)) >= 2 else 0
            lines.append(("bullet", (lvl, clean(m.group(2))))); continue
        lines.append(("text", clean(s)))
    return title, lines, table, imgs, notes

def parse_table(tbl):
    rows = []
    for r in tbl:
        if re.match(r"^\s*\|[\s:|-]+\|\s*$", r):
            continue
        cells = [clean(c.strip()) for c in r.strip().strip("|").split("|")]
        rows.append(cells)
    return rows

prs = Presentation()
prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
blank = prs.slide_layouts[6]
SW, SH = prs.slide_width, prs.slide_height

def add_text(slide, left, top, width, height, lines, title_done=True):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame; tf.word_wrap = True
    first = True
    for kind, val in lines:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        if kind == "bullet":
            lvl, txt = val
            p.level = lvl
            run = p.add_run(); run.text = ("• " if lvl == 0 else "– ") + txt
            run.font.size = Pt(18 if lvl == 0 else 15); run.font.color.rgb = DARK
        elif kind == "h":
            run = p.add_run(); run.text = val
            run.font.size = Pt(18); run.font.bold = True; run.font.color.rgb = GREEN
        elif kind == "quote":
            run = p.add_run(); run.text = "“" + val + "”"
            run.font.size = Pt(17); run.font.italic = True; run.font.color.rgb = GREEN
        else:
            run = p.add_run(); run.text = val
            run.font.size = Pt(16); run.font.color.rgb = DARK
        p.space_after = Pt(5)
    return tb

for i, raw in enumerate(parse_slides(open(MD).read())):
    title, lines, table, imgs, notes = slide_bits(raw)
    slide = prs.slides.add_slide(blank)

    if i == 0:  # title slide
        tb = slide.shapes.add_textbox(Inches(0.8), Inches(2.3), Inches(11.7), Inches(3))
        tf = tb.text_frame; tf.word_wrap = True
        p = tf.paragraphs[0]; r = p.add_run(); r.text = title
        r.font.size = Pt(40); r.font.bold = True; r.font.color.rgb = GREEN
        for kind, val in lines:
            p = tf.add_paragraph(); r = p.add_run()
            r.text = val if kind != "bullet" else val[1]
            r.font.size = Pt(20); r.font.color.rgb = GREY; p.space_before = Pt(10)
        if notes:
            slide.notes_slide.notes_text_frame.text = notes
        continue

    # title bar
    tb = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12.3), Inches(0.9))
    p = tb.text_frame.paragraphs[0]; r = p.add_run(); r.text = title
    r.font.size = Pt(28); r.font.bold = True; r.font.color.rgb = GREEN
    tb.text_frame.word_wrap = True

    has_img = bool(imgs); has_txt = bool(lines); has_tbl = bool(table)
    top0 = Inches(1.35)

    if has_img:
        # text in a compact top band (if any), images side-by-side below
        if has_txt:
            add_text(slide, Inches(0.5), Inches(1.25), Inches(12.3), Inches(1.9), lines)
            img_top = Inches(3.3)
        else:
            img_top = Inches(1.7)
        n = len(imgs)
        if n == 1:
            w = 8.6; x = Inches((13.333 - w)/2)
            slide.shapes.add_picture(imgs[0], x, img_top, width=Inches(w))
        else:
            w, gap = 6.2, 0.3
            total = n*w + (n-1)*gap; x = (13.333 - total)/2
            for im in imgs:
                slide.shapes.add_picture(im, Inches(x), img_top, width=Inches(w))
                x += w + gap
    else:
        # text and/or table, no image
        cur = top0
        if has_txt:
            add_text(slide, Inches(0.6), cur, Inches(12.1), Inches(4.0), lines)
        if has_tbl:
            rows = parse_table(table)
            if rows:
                nr, nc = len(rows), max(len(r) for r in rows)
                gtop = Inches(1.5 + (0.45*len(lines) if has_txt else 0))
                gh = Inches(min(0.5*nr, 5.0))
                gtbl = slide.shapes.add_table(nr, nc, Inches(0.6), gtop,
                                              Inches(12.1), gh).table
                for ri, row in enumerate(rows):
                    for ci in range(nc):
                        cell = gtbl.cell(ri, ci)
                        cell.text = row[ci] if ci < len(row) else ""
                        for para in cell.text_frame.paragraphs:
                            for rr in para.runs:
                                rr.font.size = Pt(13)
                                if ri == 0: rr.font.bold = True
    if notes:
        slide.notes_slide.notes_text_frame.text = notes

prs.save(OUT)
print("wrote", OUT, "with", len(prs.slides._sldIdLst), "slides")
