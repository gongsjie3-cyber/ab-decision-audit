"""
把 reports/report.md 转成 Word 交付件 reports/report.docx。

用法：  python src/make_report_docx.py

支持：一级/二级/三级标题、正文段落、无序列表、粗体 **...**、行内代码 `...`、
Markdown 表格、水平线，以及在报告中按文件名自动嵌入 figures/ 下的图。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "reports" / "report.md"
OUT = ROOT / "reports" / "report.docx"
FIGDIR = ROOT / "figures"

CJK_FONT = "PingFang SC"   # macOS 系统自带；Windows/Word 会自动回退到中文字体
LATIN_FONT = "Calibri"      # 拉丁字符用通用字体，避免跨平台缺字
MONO = "Menlo"


def set_cjk(run, name=CJK_FONT):
    """中文字体必须显式写入 w:eastAsia，否则 Word 会按主题字体渲染。"""
    run.font.name = LATIN_FONT
    rpr = run._element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = rpr.makeelement(qn("w:rFonts"), {})
        rpr.append(rf)
    rf.set(qn("w:eastAsia"), name)
    rf.set(qn("w:ascii"), LATIN_FONT)
    rf.set(qn("w:hAnsi"), LATIN_FONT)


def _run(par, text, base_size=10.5, bold=False, italic=False, sub=False, sup=False, mono=False):
    """写入一个 run，并统一处理字体（含 CJK 的 eastAsia 设置）。"""
    if text == "":
        return None
    r = par.add_run(text)
    r.bold = bold
    r.italic = italic
    if sub:
        r.font.subscript = True
    if sup:
        r.font.superscript = True
    if mono:
        r.font.name = MONO
        r.font.size = Pt(base_size - 0.5)
        r.font.color.rgb = RGBColor(0xB0, 0x30, 0x60)
    else:
        set_cjk(r, CJK_FONT)
        if r.font.size is None:
            r.font.size = Pt(base_size)
    return r


# 行内标记：**粗体** / *斜体* / `代码` / <sub>下标</sub> / <sup>上标</sup> / [文字](链接)
# 注意顺序：** 必须排在 * 之前，否则粗体会被当成两个斜体。
_INLINE = re.compile(r"(\*\*.+?\*\*|<sub>.*?</sub>|<sup>.*?</sup>|`[^`]+`|\*[^*\n]+?\*|\[[^\]]+\]\([^)]+\))")


def add_runs(par, text, base_size=10.5, bold=False, italic=False):
    """递归解析行内标记。

    递归是必要的：`**n<sub>ti</sub>**` 这种"下标嵌在粗体里"的写法，若只做一次
    切分，粗体 token 会把整段（连同 <sub> 标签）一起吞掉，标签就被原样打印了。
    """
    for tok in _INLINE.split(text):
        if not tok:
            continue
        if tok.startswith("**") and tok.endswith("**") and len(tok) > 4:
            add_runs(par, tok[2:-2], base_size, bold=True, italic=italic)
        elif tok.startswith("<sub>") and tok.endswith("</sub>"):
            _run(par, tok[5:-6], base_size, bold, italic, sub=True)
        elif tok.startswith("<sup>") and tok.endswith("</sup>"):
            _run(par, tok[5:-6], base_size, bold, italic, sup=True)
        elif tok.startswith("`") and tok.endswith("`") and len(tok) > 2:
            _run(par, tok[1:-1], base_size, bold, italic, mono=True)
        elif tok.startswith("*") and tok.endswith("*") and len(tok) > 2:
            add_runs(par, tok[1:-1], base_size, bold, italic=True)
        elif tok.startswith("[") and "](" in tok:
            _run(par, tok[1:tok.index("]")], base_size, bold, italic)   # 只保留链接文字
        else:
            _run(par, tok, base_size, bold, italic)


def strip_md(s):
    s = re.sub(r"</?su[bp]>", "", s)
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
    s = re.sub(r"\*([^*\n]+?)\*", r"\1", s)
    s = re.sub(r"`([^`]+)`", r"\1", s)
    s = re.sub(r"\[(.+?)\]\((.+?)\)", r"\1", s)
    return s.strip()


def set_code_font(run, size=9.0):
    """代码块字体：拉丁用等宽，中文回退到正文字体（等宽字体普遍无 CJK 字形）。"""
    run.font.name = MONO
    run.font.size = Pt(size)
    rpr = run._element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = rpr.makeelement(qn("w:rFonts"), {})
        rpr.append(rf)
    rf.set(qn("w:ascii"), MONO)
    rf.set(qn("w:hAnsi"), MONO)
    rf.set(qn("w:eastAsia"), CJK_FONT)


def set_shading(par, fill="F2F2F2"):
    """给段落加底色，用于渲染代码块。"""
    pPr = par._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    pPr.append(shd)


def set_left_border(par, color="9AA0A6", size=18, space=10):
    """给段落加左侧竖线，用于渲染 Markdown 引用块。"""
    pPr = par._p.get_or_add_pPr()
    pBdr = OxmlElement("w:pBdr")
    left = OxmlElement("w:left")
    left.set(qn("w:val"), "single")
    left.set(qn("w:sz"), str(size))
    left.set(qn("w:space"), str(space))
    left.set(qn("w:color"), color)
    pBdr.append(left)
    pPr.append(pBdr)


def add_table(doc, rows):
    header, body = rows[0], rows[1:]
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(header):
        cell = t.rows[0].cells[i]
        cell.text = ""
        p = cell.paragraphs[0]
        r = p.add_run(strip_md(h))
        r.bold = True
        r.font.size = Pt(9.5)
        set_cjk(r)
    for row in body:
        cells = t.add_row().cells
        for i, v in enumerate(row[: len(header)]):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            add_runs(p, v, base_size=9.5)
    doc.add_paragraph()


def main():
    doc = Document()
    # 页面与默认字体
    st = doc.styles["Normal"]
    st.font.size = Pt(10.5)
    st.font.name = LATIN_FONT
    st.element.rPr.rFonts.set(qn("w:eastAsia"), CJK_FONT)
    for sec in doc.sections:
        sec.left_margin = sec.right_margin = Cm(2.4)
        sec.top_margin = sec.bottom_margin = Cm(2.2)

    lines = SRC.read_text(encoding="utf-8").splitlines()
    i = 0
    fig_used = []
    while i < len(lines):
        line = lines[i].rstrip()
        if not line.strip():
            i += 1
            continue
        # 水平线
        if re.fullmatch(r"-{3,}", line.strip()):
            doc.add_paragraph()
            i += 1
            continue
        # 代码块 —— 必须先于标题判断，否则块内以 # 开头的注释会被当成 Markdown 标题
        if line.strip().startswith("```"):
            i += 1
            block = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.append(lines[i].rstrip())
                i += 1
            i += 1  # 跳过收尾的 ```
            while block and not block[-1].strip():
                block.pop()
            for b in block:
                cp = doc.add_paragraph()
                cp.paragraph_format.space_before = Pt(0)
                cp.paragraph_format.space_after = Pt(0)
                cp.paragraph_format.left_indent = Cm(0.45)
                cp.paragraph_format.line_spacing = 1.0
                set_shading(cp)
                set_code_font(cp.add_run(b if b.strip() else "\u00a0"))
            doc.add_paragraph().paragraph_format.space_after = Pt(2)
            continue
        # 标题
        m = re.match(r"^(#{1,4})\s+(.*)$", line)
        if m:
            lvl = len(m.group(1))
            text = strip_md(m.group(2))
            if lvl == 1:
                h = doc.add_heading("", level=0)
                r = h.add_run(text)
                r.font.size = Pt(20)
                set_cjk(r)
                h.alignment = WD_ALIGN_PARAGRAPH.CENTER
            else:
                h = doc.add_heading("", level=min(lvl - 1, 4))
                r = h.add_run(text)
                set_cjk(r)
                r.font.color.rgb = RGBColor(0x1F, 0x38, 0x64)
            i += 1
            continue
        # 表格
        if line.strip().startswith("|"):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i].strip())
                i += 1
            rows = []
            for b in block:
                if re.fullmatch(r"\|[\s\-:|]+\|", b):
                    continue
                cells = [c.strip() for c in b.strip("|").split("|")]
                rows.append(cells)
            if rows:
                add_table(doc, rows)
            continue
        # 列表
        if re.match(r"^\s*([-*]|\d+\.)\s+", line):
            text = re.sub(r"^\s*([-*]|\d+\.)\s+", "", line)
            p = doc.add_paragraph(style="List Bullet")
            add_runs(p, text)
            i += 1
            continue
        # 图（Markdown 图片语法）
        mi = re.match(r"^!\[(.*?)\]\((.+?)\)\s*$", line.strip())
        if mi:
            cap, path = mi.group(1), mi.group(2)
            cand = Path(path) if Path(path).is_absolute() else (SRC.parent / path).resolve()
            if cand.exists():
                doc.add_picture(str(cand), width=Cm(16.4))
                doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
                # 每张图的 PNG 里已经烙好了「图 N ｜ ...」标题，这里不再重复加题注，
                # 否则 PDF 里每张图都会出现两遍标题。
                fig_used.append(str(cand))
            i += 1
            continue
        # 引用块（Markdown 的 "> ..."）
        if line.lstrip().startswith(">"):
            quote = re.sub(r"^\s*>\s?", "", line)
            if not quote.strip():
                i += 1
                continue
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.7)
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(4)
            set_left_border(p)
            add_runs(p, quote.strip())
            i += 1
            continue
        # 正文
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.first_line_indent = Cm(0)
        add_runs(p, line.strip())
        i += 1

    # ---- 正文已内嵌图片；这里只列出未在正文出现的图，避免重复 ----
    figs = sorted(FIGDIR.rglob("fig*.png"))
    missing = [f for f in figs if str(f) not in fig_used]
    if missing:
        doc.add_page_break()
        h = doc.add_heading("", level=1)
        r = h.add_run("附录 C：补充图表")
        set_cjk(r)
        for f in missing:
            doc.add_picture(str(f), width=Cm(16.2))
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            cp = doc.add_paragraph()
            r = cp.add_run(f.stem.replace("_", " "))
            r.font.size = Pt(9)
            set_cjk(r)
            cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.save(OUT)
    print(f"[docx] {OUT.relative_to(ROOT)}  （正文 {len(lines)} 行，内嵌图 {len(fig_used)} 张，补充图 {len([f for f in figs if str(f) not in fig_used])} 张）")


if __name__ == "__main__":
    sys.exit(main())
