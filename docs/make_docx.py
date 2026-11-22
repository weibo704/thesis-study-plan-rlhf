"""Assemble the stage-1 report and its supporting evidence into one Word file.

The markdown reports are the source of truth; this renders them as a single
readable document with a title page, contents, the flow figure, and the
supporting reports as appendices.

    python docs/make_docx.py
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

DOCS = Path(__file__).parent
ROOT = DOCS.parent

BODY = "Cambria"
HEAD = "Calibri Light"
MONO = "Consolas"
INK = RGBColor(0x1A, 0x1F, 0x26)
MUTED = RGBColor(0x5A, 0x65, 0x72)
# Headings print black. The teal that was here read as green on
# screen and in print, which is not what a thesis wants.
ACCENT = RGBColor(0x00, 0x00, 0x00)
ACCENT2 = RGBColor(0x00, 0x00, 0x00)

SOURCES: List[Tuple[str, Path, str]] = [
    ("", DOCS / "stage1-report.md", "main"),
    ("Appendix A — Stage-1 evidence", ROOT / "reports/stage1_evidence.md", "appendix"),
    ("Appendix B — Stage-1 diagnosis", ROOT / "reports/stage1_diagnosis.md", "appendix"),
    ("Appendix C — Randomized-graph ablation",
     ROOT / "reports/ablation_randomized_graph.md", "appendix"),
]

FIGURE_AFTER = "how they fit together at runtime"


# --------------------------------------------------------------- inline text
TOKEN = re.compile(r"(\*\*.+?\*\*|`[^`]+`|\*[^*]+\*|\[[^\]]+\]\([^)]+\))")


def add_runs(p, text: str, size=10.5, color=INK, italic=False):
    """Render **bold**, `code`, *italic* and [links](x) as Word runs."""
    text = text.replace("\\u2192", "→")
    for part in TOKEN.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            r = p.add_run(part[2:-2]); r.bold = True
        elif part.startswith("`") and part.endswith("`"):
            r = p.add_run(part[1:-1]); r.font.name = MONO
            r.font.size = Pt(size - 1.2); r.font.color.rgb = ACCENT
            continue
        elif part.startswith("[") and "](" in part:
            label = part[1:part.index("](")]
            r = p.add_run(label); r.font.color.rgb = ACCENT2
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            r = p.add_run(part[1:-1]); r.italic = True
        else:
            r = p.add_run(part)
        r.font.name = BODY
        r.font.size = Pt(size)
        r.font.color.rgb = color
        if italic:
            r.italic = True


def shade(cell, hexcolor: str):
    el = OxmlElement("w:shd")
    el.set(qn("w:val"), "clear")
    el.set(qn("w:fill"), hexcolor)
    cell._tc.get_or_add_tcPr().append(el)


def style_heading(p, size, color, space_before=16, space_after=6):
    for r in p.runs:
        r.font.name = HEAD
        r.font.size = Pt(size)
        r.font.color.rgb = color
        r.bold = True
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.keep_with_next = True


# -------------------------------------------------------------------- blocks
def flush_table(doc, rows: List[str]):
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    cells = [c for c in cells if not all(set(x) <= set("-: ") for x in c)]
    if not cells:
        return
    ncol = max(len(r) for r in cells)
    t = doc.add_table(rows=len(cells), cols=ncol)
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(cells):
        for j in range(ncol):
            cell = t.cell(i, j)
            cell.text = ""
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            txt = row[j] if j < len(row) else ""
            add_runs(p, txt, size=9.5, color=INK if i else RGBColor(0xFF, 0xFF, 0xFF))
            if i == 0:
                shade(cell, "262626")   # was teal; headings print black
                for r in p.runs:
                    r.bold = True
            elif i % 2 == 0:
                shade(cell, "F4F7F8")
    doc.add_paragraph().paragraph_format.space_after = Pt(4)


def render(doc, md: str, appendix_title: str = "", offset: int = 0):
    lines = md.split("\n")
    i, table, code, buf = 0, [], False, []

    def flush_para():
        if not buf:
            return
        text = " ".join(x.strip() for x in buf).strip()
        buf.clear()
        if not text:
            return
        p = doc.add_paragraph()
        add_runs(p, text)
        p.paragraph_format.space_after = Pt(8)
        p.paragraph_format.line_spacing = 1.15

    while i < len(lines):
        ln = lines[i]
        s = ln.strip()

        if s.startswith("```"):
            flush_para()
            code = not code
            if code:
                block = []
                i += 1
                while i < len(lines) and not lines[i].strip().startswith("```"):
                    block.append(lines[i]); i += 1
                code = False
                p = doc.add_paragraph()
                r = p.add_run("\n".join(block))
                r.font.name = MONO; r.font.size = Pt(9); r.font.color.rgb = INK
                p.paragraph_format.left_indent = Inches(0.3)
                p.paragraph_format.space_before = Pt(6)
                p.paragraph_format.space_after = Pt(10)
                shade_p = OxmlElement("w:shd")
                shade_p.set(qn("w:val"), "clear"); shade_p.set(qn("w:fill"), "F4F7F8")
                p._p.get_or_add_pPr().append(shade_p)
            i += 1
            continue

        if s.startswith("|"):
            flush_para(); table.append(s); i += 1
            continue
        if table:
            flush_table(doc, table); table = []

        if not s:
            flush_para(); i += 1; continue

        if s.startswith("#"):
            flush_para()
            level = len(s) - len(s.lstrip("#"))
            title = s.lstrip("#").strip()
            if level == 1:
                if appendix_title:
                    i += 1; continue           # appendix heading already written
                i += 1; continue               # main title lives on the title page
            lvl = min(level - 1 + offset, 4)
            p = doc.add_heading("", level=lvl)
            add_runs(p, title)
            style_heading(p, 15 if lvl == 1 else 12.5,
                          ACCENT if lvl == 1 else ACCENT2,
                          space_before=20 if lvl == 1 else 14)
            i += 1
            continue

        if s in ("---", "***", "___"):
            flush_para(); i += 1; continue

        if s.startswith("> "):
            flush_para()
            p = doc.add_paragraph()
            add_runs(p, s[2:], size=10, color=MUTED, italic=True)
            p.paragraph_format.left_indent = Inches(0.3)
            p.paragraph_format.space_after = Pt(8)
            i += 1
            continue

        m = re.match(r"^(\d+)\.\s+(.*)", s)
        if s.startswith(("- ", "* ")) or m:
            flush_para()
            body = m.group(2) if m else s[2:]
            cont = []
            i += 1
            while i < len(lines) and lines[i].startswith(("   ", "\t")) \
                    and lines[i].strip() and not lines[i].strip().startswith(("-", "*", "|")):
                cont.append(lines[i].strip()); i += 1
            p = doc.add_paragraph(style="List Number" if m else "List Bullet")
            add_runs(p, " ".join([body] + cont))
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.line_spacing = 1.1
            continue

        buf.append(ln)
        if FIGURE_AFTER in ln:
            flush_para()
            fig = DOCS / "stage1-flow.png"
            if fig.exists():
                doc.add_picture(str(fig), width=Inches(6.6))
                doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
                cap = doc.add_paragraph()
                add_runs(cap, "Figure 1 — How a user query becomes a study plan, "
                              "and how feedback changes the next one. The editable "
                              "source is docs/stage1-flow.drawio.",
                         size=9, color=MUTED, italic=True)
                cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                cap.paragraph_format.space_after = Pt(14)
        i += 1

    if table:
        flush_table(doc, table)
    flush_para()


def build() -> Path:
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = BODY
    st.font.size = Pt(10.5)
    for s in doc.sections:
        s.top_margin = s.bottom_margin = Inches(0.9)
        s.left_margin = s.right_margin = Inches(0.95)

    # ---- title page
    for _ in range(5):
        doc.add_paragraph()
    t = doc.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t.add_run("Does reinforcement learning improve")
    r.font.name = HEAD; r.font.size = Pt(26); r.bold = True; r.font.color.rgb = INK
    r.add_break()
    r2 = t.add_run("LLM-generated study plans?")
    r2.font.name = HEAD; r2.font.size = Pt(26); r2.bold = True; r2.font.color.rgb = INK
    sub = doc.add_paragraph(); sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = sub.add_run("Stage 1 Report — what has been built, what it shows, "
                    "and what it does not")
    r.font.name = HEAD; r.font.size = Pt(13); r.font.color.rgb = MUTED
    doc.add_paragraph()
    meta = doc.add_paragraph(); meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = meta.add_run("Pre-registered bundle poc-v6-stratified  ·  165 tests passing")
    r.font.name = BODY; r.font.size = Pt(11); r.font.color.rgb = MUTED
    r.add_break()
    r2 = meta.add_run("5 October 2026")
    r2.font.name = BODY; r2.font.size = Pt(11); r2.font.color.rgb = MUTED
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    # ---- contents (a real Word field, so the reader can refresh it)
    h = doc.add_paragraph(); add_runs(h, "Contents")
    style_heading(h, 18, ACCENT, space_before=0, space_after=10)
    p = doc.add_paragraph()
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), r'TOC \o "1-2" \h \z \u')
    inner = OxmlElement("w:r")
    txt = OxmlElement("w:t")
    txt.text = "Right-click here and choose “Update Field” to build the contents."
    inner.append(txt); fld.append(inner)
    p._p.append(fld)
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    for title, path, kind in SOURCES:
        if not path.exists():
            continue
        if kind == "appendix":
            doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
            h = doc.add_heading("", level=1); add_runs(h, title)
            style_heading(h, 18, ACCENT, space_before=0, space_after=4)
            note = doc.add_paragraph()
            add_runs(note, f"Generated by the pipeline; source {path.relative_to(ROOT)}.",
                     size=9.5, color=MUTED, italic=True)
            note.paragraph_format.space_after = Pt(12)
        render(doc, path.read_text(encoding="utf-8"), appendix_title=title,
               offset=1 if kind == "appendix" else 0)

    out = DOCS / "Stage-1-Report.docx"
    doc.save(out)
    return out


if __name__ == "__main__":
    p = build()
    print(f"wrote {p} ({p.stat().st_size/1024:.0f} KB)")
