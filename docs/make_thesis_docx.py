"""Build the thesis draft as a Word document in Northeastern MS format.

Matches the sample thesis supplied by the author: the standard title page
block, roman-numbered front matter, arabic-numbered body, chapters opening
with "Chapter N" above the title, running headers of the form
"CHAPTER 4. RESULTS", and bold run-in paragraph headings.

Reuses the inline and table rendering from make_docx.py.

    python docs/make_thesis_docx.py
"""
from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

import make_docx as md

DOCS = Path(__file__).parent
SRC = DOCS / "thesis-draft.md"
OUT = DOCS / "Thesis-Draft.docx"

NOTE_BG = "FFF6E5"
NOTE_INK = RGBColor(0x7A, 0x4A, 0x00)

# Front-matter sections, in the order the sample presents them. Anything
# else at heading level 1 is a numbered chapter or an appendix.
FRONT = ["Dedication", "List of Acronyms", "Acknowledgments",
         "Abstract of the Thesis"]


def meta(text: str) -> dict:
    out = {}
    for m in re.finditer(r"<!--\s*([A-Z]+):\s*(.*?)\s*-->", text):
        out[m.group(1)] = m.group(2)
    return out


# ------------------------------------------------------------------ fields
def _field(p, instr: str, size=10.5):
    r = p.add_run()
    a = OxmlElement("w:fldChar"); a.set(qn("w:fldCharType"), "begin")
    b = OxmlElement("w:instrText"); b.set(qn("xml:space"), "preserve"); b.text = f" {instr} "
    c = OxmlElement("w:fldChar"); c.set(qn("w:fldCharType"), "end")
    r._r.append(a); r._r.append(b); r._r.append(c)
    r.font.name = md.BODY; r.font.size = Pt(size); r.font.color.rgb = md.MUTED


def _numbering(section, fmt: str, start: int | None):
    """Roman for front matter, arabic restarting at 1 for the body."""
    pr = section._sectPr
    el = pr.find(qn("w:pgNumType"))
    if el is None:
        el = OxmlElement("w:pgNumType"); pr.append(el)
    el.set(qn("w:fmt"), fmt)
    if start is not None:
        el.set(qn("w:start"), str(start))


def _footer_page(section):
    section.footer.is_linked_to_previous = False
    p = section.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for r in list(p.runs):
        r._r.getparent().remove(r._r)
    _field(p, "PAGE")


def _running_header(section):
    """A STYLEREF field, so Word prints whichever chapter the page is in.

    A static string would only ever show the last chapter written, because
    one section carries the whole body; the sample's per-page header is a
    field, not text.
    """
    section.header.is_linked_to_previous = False
    p = section.header.paragraphs[0]
    for r in list(p.runs):
        r._r.getparent().remove(r._r)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    r = p.add_run()
    a = OxmlElement("w:fldChar"); a.set(qn("w:fldCharType"), "begin")
    b = OxmlElement("w:instrText"); b.set(qn("xml:space"), "preserve")
    b.text = r' STYLEREF "Heading 1" \\* MERGEFORMAT '
    c = OxmlElement("w:fldChar"); c.set(qn("w:fldCharType"), "separate")
    t = OxmlElement("w:t"); t.text = "Chapter"
    e = OxmlElement("w:fldChar"); e.set(qn("w:fldCharType"), "end")
    r._r.append(a); r._r.append(b); r._r.append(c); r._r.append(t); r._r.append(e)
    r.font.name = md.BODY; r.font.size = Pt(9)
    r.font.color.rgb = md.MUTED; r.font.all_caps = True


def _author_note(doc, text: str):
    p = doc.add_paragraph()
    md.add_runs(p, text, size=10, color=NOTE_INK, italic=True)
    pf = p.paragraph_format
    pf.left_indent = pf.right_indent = Inches(0.22)
    pf.space_before = Pt(8); pf.space_after = Pt(12)
    sh = OxmlElement("w:shd"); sh.set(qn("w:val"), "clear"); sh.set(qn("w:fill"), NOTE_BG)
    p._p.get_or_add_pPr().append(sh)


# ------------------------------------------------------------------ render
def render(doc, text: str, m: dict):
    lines = text.split("\n")
    i, table, buf, quote = 0, [], [], []
    current, started_body = "", False
    # The acronym list is one term per line. The ordinary paragraph rule
    # joins consecutive lines, which ran all eighteen into a single block.
    in_acronyms = False

    def flush_para():
        if buf:
            s = " ".join(x.strip() for x in buf).strip(); buf.clear()
            if s:
                p = doc.add_paragraph()
                md.add_runs(p, s)
                p.paragraph_format.space_after = Pt(9)
                p.paragraph_format.line_spacing = 1.4

    def flush_quote():
        if quote:
            _author_note(doc, " ".join(x.strip() for x in quote).strip()); quote.clear()

    while i < len(lines):
        raw, s = lines[i], lines[i].strip()

        if s.startswith("<!--"):
            i += 1; continue

        if s.startswith("```"):
            flush_para(); flush_quote()
            blk, i = [], i + 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                blk.append(lines[i]); i += 1
            p = doc.add_paragraph()
            r = p.add_run("\n".join(blk))
            r.font.name = md.MONO; r.font.size = Pt(9.5); r.font.color.rgb = md.INK
            p.paragraph_format.left_indent = Inches(0.35)
            p.paragraph_format.space_before = Pt(6); p.paragraph_format.space_after = Pt(12)
            sh = OxmlElement("w:shd"); sh.set(qn("w:val"), "clear"); sh.set(qn("w:fill"), "F4F7F8")
            p._p.get_or_add_pPr().append(sh)
            i += 1; continue

        if s.startswith("|"):
            flush_para(); flush_quote(); table.append(s); i += 1; continue
        if table:
            md.flush_table(doc, table); table = []

        if s.startswith("> "):
            flush_para(); quote.append(s[2:]); i += 1; continue
        if quote:
            flush_quote()

        if not s:
            flush_para(); i += 1; continue

        img = re.match(r"^!\[(.*?)\]\((.+?)\)\s*$", s)
        if img:
            flush_para()
            fig = DOCS / img.group(2)
            if fig.exists():
                doc.add_picture(str(fig), width=Inches(6.1))
                doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
                cap = doc.add_paragraph()
                md.add_runs(cap, img.group(1), size=9, color=md.MUTED)
                cap.alignment = WD_ALIGN_PARAGRAPH.LEFT
                cap.paragraph_format.space_after = Pt(14)
            else:
                print(f"  ! figure missing: {fig}")
            i += 1; continue

        if s.startswith("#"):
            flush_para(); flush_quote()
            level = len(s) - len(s.lstrip("#"))
            title = s.lstrip("#").strip()

            if level == 1:
                num = re.match(r"^(\d+|[A-Z])\s+(.*)$", title)
                is_front = title in FRONT
                # The body starts at the first numbered chapter; page
                # numbering restarts there, as in the sample.
                if num and not started_body:
                    sec = doc.add_section(WD_SECTION.NEW_PAGE)
                    _numbering(sec, "decimal", 1)
                    _footer_page(sec); _running_header(sec)
                    started_body = True
                elif doc.paragraphs:
                    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

                if num:
                    label = ("Chapter " + num.group(1)) if num.group(1).isdigit() \
                            else ("Appendix " + num.group(1))
                    lab = doc.add_paragraph()
                    r = lab.add_run(label)
                    r.font.name = md.HEAD; r.font.size = Pt(15)
                    r.font.color.rgb = md.MUTED
                    lab.paragraph_format.space_after = Pt(2)
                    head_title = num.group(2)
                    current = f"{label}. {head_title}"
                else:
                    head_title = title
                    current = title

                h = doc.add_heading("", level=1)
                md.add_runs(h, head_title)
                md.style_heading(h, 22, md.ACCENT, space_before=0, space_after=18)

                in_acronyms = (head_title == "List of Acronyms")

                if head_title == "Abstract of the Thesis":
                    # The sample repeats a title block under this heading:
                    # title, author, degree, institution and advisor, each on
                    # its own centred line.
                    centred(doc, m.get("TITLE", ""), 13, bold=True,
                            font=md.HEAD, after=10)
                    centred(doc, "by", 11, after=3)
                    centred(doc, m.get("AUTHOR", ""), 12, bold=True, after=10)
                    for line in (f"{m.get('DEGREE','')} in {m.get('PROGRAM','')}",
                                 f"Northeastern University, {m.get('DATE','')}",
                                 f"{m.get('ADVISOR','Advisor')}, Advisor"):
                        centred(doc, line, 11, color=md.MUTED, after=3)
                    doc.paragraphs[-1].paragraph_format.space_after = Pt(16)
            else:
                h = doc.add_heading("", level=min(level, 4))
                md.add_runs(h, title)
                md.style_heading(h, 13.5 if level == 2 else 12,
                                 md.ACCENT if level == 2 else md.ACCENT2,
                                 space_before=18 if level == 2 else 13)
            i += 1; continue

        if s in ("---", "***", "___"):
            flush_para(); i += 1; continue

        mnum = re.match(r"^(\d+)\.\s+(.*)", s)
        if s.startswith(("- ", "* ")) or mnum:
            flush_para()
            body = mnum.group(2) if mnum else s[2:]
            cont, i = [], i + 1
            while (i < len(lines) and lines[i].startswith(("   ", "\t"))
                   and lines[i].strip()
                   and not lines[i].strip().startswith(("-", "*", "|", ">"))):
                cont.append(lines[i].strip()); i += 1
            p = doc.add_paragraph(style="List Number" if mnum else "List Bullet")
            md.add_runs(p, " ".join([body] + cont))
            p.paragraph_format.space_after = Pt(5)
            p.paragraph_format.line_spacing = 1.25
            continue

        if in_acronyms:
            # Hanging indent, so a definition that wraps lines up under
            # itself rather than under the acronym.
            p = doc.add_paragraph()
            md.add_runs(p, s)
            pf = p.paragraph_format
            pf.left_indent = Inches(1.35)
            pf.first_line_indent = Inches(-1.35)
            pf.space_after = Pt(3)
            pf.line_spacing = 1.0
            i += 1
            continue

        buf.append(raw); i += 1

    if table:
        md.flush_table(doc, table)
    flush_para(); flush_quote()


def centred(doc, text, size, bold=False, font=None, color=None, after=4):
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text)
    r.font.name = font or md.BODY; r.font.size = Pt(size); r.bold = bold
    r.font.color.rgb = color or md.INK
    p.paragraph_format.space_after = Pt(after)
    return p


def build() -> Path:
    text = SRC.read_text(encoding="utf-8")
    m = meta(text)

    doc = Document()
    st = doc.styles["Normal"]; st.font.name = md.BODY; st.font.size = Pt(11)
    sec = doc.sections[0]
    sec.top_margin = sec.bottom_margin = Inches(1.0)
    sec.left_margin = Inches(1.4); sec.right_margin = Inches(1.0)

    # ---- title page, matching the sample's block exactly
    for _ in range(3):
        doc.add_paragraph()
    centred(doc, m.get("TITLE", "Untitled"), 19, bold=True, font=md.HEAD, after=22)
    for line, size in (("A Thesis Presented", 12), ("by", 12)):
        centred(doc, line, size, color=md.INK, after=4)
    centred(doc, m.get("AUTHOR", "Author Name"), 13, bold=True, after=16)
    centred(doc, "to", 12, after=4)
    centred(doc, m.get("DEPARTMENT", "Department"), 12, after=16)
    for line in ("in partial fulfillment of the requirements",
                 "for the degree of"):
        centred(doc, line, 12, after=4)
    centred(doc, m.get("DEGREE", "Master of Science"), 13, bold=True, after=4)
    centred(doc, "in", 12, after=4)
    centred(doc, m.get("PROGRAM", "Program"), 13, bold=True, after=22)
    centred(doc, "Northeastern University", 12, after=4)
    centred(doc, "Boston, Massachusetts", 12, after=4)
    centred(doc, m.get("DATE", ""), 12, after=4)

    # ---- front matter: roman numerals, starting after the unnumbered title page
    front = doc.add_section(WD_SECTION.NEW_PAGE)
    front.top_margin = front.bottom_margin = Inches(1.0)
    front.left_margin = Inches(1.4); front.right_margin = Inches(1.0)
    _numbering(front, "lowerRoman", 1)
    _footer_page(front)

    # Contents sits before the other front matter, as in the sample.
    body_md = text[text.index("\n#"):]
    first = body_md.split("\n")
    # Abstract title block is inserted when that heading is reached.
    ABSTRACT_BLOCK = (m.get("TITLE", ""), m.get("AUTHOR", ""),
                      f"{m.get('DEGREE','')} in {m.get('PROGRAM','')}",
                      f"Northeastern University, {m.get('DATE','')}",
                      f"{m.get('ADVISOR','Advisor')}, Advisor")

    h = doc.add_paragraph(); md.add_runs(h, "Contents")
    md.style_heading(h, 22, md.ACCENT, space_before=0, space_after=14)
    p = doc.add_paragraph()
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), r'TOC \o "1-3" \h \z \u')
    ir = OxmlElement("w:r"); it = OxmlElement("w:t")
    it.text = "Right-click here and choose \u201cUpdate Field\u201d to build the contents."
    ir.append(it); fld.append(ir); p._p.append(fld)

    render(doc, body_md, m)
    doc.save(OUT)
    return OUT


if __name__ == "__main__":
    out = build()
    print(f"wrote {out} ({out.stat().st_size/1024:.0f} KB)")
