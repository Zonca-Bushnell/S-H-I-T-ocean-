from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
MD_PATH = ROOT / "docs" / "acc_boundary_tangent_literature_review.md"
REPORT_ROOT = Path(r"E:\Report\01_Eddy_correspond\03_Original_detection\ACC\2026_09_11_boundary_tangent_literature")
DOC_DIR = REPORT_ROOT / "documents"
PDF_PATH = DOC_DIR / "boundary_tangent_literature_review.pdf"
MD_COPY_PATH = DOC_DIR / "boundary_tangent_literature_review.md"


TITLE = "ACC 涡旋识别中边界单调、切向对齐与对称性检验的文献依据"


def register_font() -> str:
    candidates = [
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
    ]
    for path in candidates:
        if path.exists():
            pdfmetrics.registerFont(TTFont("CJK", str(path)))
            return "CJK"
    return "Helvetica"


def markdown_to_story(markdown: str, font_name: str) -> list:
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "TitleCJK",
        parent=styles["Title"],
        fontName=font_name,
        fontSize=16,
        leading=22,
        textColor=colors.HexColor("#163A5B"),
        spaceAfter=12,
    )
    h1 = ParagraphStyle(
        "H1CJK",
        parent=styles["Heading1"],
        fontName=font_name,
        fontSize=15,
        leading=21,
        textColor=colors.HexColor("#155C7A"),
        spaceBefore=12,
        spaceAfter=8,
    )
    h2 = ParagraphStyle(
        "H2CJK",
        parent=styles["Heading2"],
        fontName=font_name,
        fontSize=12.5,
        leading=18,
        textColor=colors.HexColor("#1E4F68"),
        spaceBefore=8,
        spaceAfter=6,
    )
    body = ParagraphStyle(
        "BodyCJK",
        parent=styles["BodyText"],
        fontName=font_name,
        fontSize=10,
        leading=16,
        spaceAfter=6,
    )
    mono = ParagraphStyle(
        "MonoCJK",
        parent=body,
        fontName="Courier",
        fontSize=8,
        leading=11,
        backColor=colors.HexColor("#F4F6F8"),
        leftIndent=8,
        rightIndent=8,
        spaceBefore=4,
        spaceAfter=8,
    )

    story: list = []
    in_code = False
    code_lines: list[str] = []
    table_lines: list[str] = []

    def flush_code() -> None:
        nonlocal code_lines
        if code_lines:
            text = "<br/>".join(line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;") for line in code_lines)
            story.append(Paragraph(text, mono))
            code_lines = []

    def flush_table() -> None:
        nonlocal table_lines
        if not table_lines:
            return
        rows = []
        for line in table_lines:
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if cells and all(set(cell) <= set("-: ") for cell in cells):
                continue
            rows.append(cells)
        if rows:
            table = Table(rows, repeatRows=1)
            table.setStyle(
                TableStyle(
                    [
                        ("FONTNAME", (0, 0), (-1, -1), font_name),
                        ("FONTSIZE", (0, 0), (-1, -1), 8),
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DDEBF3")),
                        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#AAB7C4")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 4),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ]
                )
            )
            story.append(table)
            story.append(Spacer(1, 5 * mm))
        table_lines = []

    def clean_inline(value: str) -> str:
        return value.replace("`", "")

    for raw in markdown.splitlines():
        line = raw.rstrip()
        if line.startswith("```"):
            if in_code:
                flush_code()
                in_code = False
            else:
                flush_table()
                in_code = True
            continue
        if in_code:
            code_lines.append(line)
            continue
        if line.startswith("|"):
            table_lines.append(line)
            continue
        flush_table()
        if not line:
            story.append(Spacer(1, 2 * mm))
        elif line.startswith("# "):
            story.append(Paragraph(clean_inline(line[2:]), title))
        elif line.startswith("## "):
            story.append(Paragraph(clean_inline(line[3:]), h1))
        elif line.startswith("### "):
            story.append(Paragraph(clean_inline(line[4:]), h2))
        elif line.startswith(("1. ", "2. ", "3. ", "4. ", "5. ")):
            story.append(Paragraph(line, body))
        else:
            story.append(Paragraph(clean_inline(line), body))
    flush_table()
    flush_code()
    return story


def footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#777777"))
    canvas.drawRightString(200 * mm, 10 * mm, f"{doc.page}")
    canvas.restoreState()


def main() -> None:
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    markdown = MD_PATH.read_text(encoding="utf-8")
    MD_COPY_PATH.write_text(markdown, encoding="utf-8")
    font_name = register_font()
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=TITLE,
    )
    story = markdown_to_story(markdown, font_name)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(PDF_PATH)
    print(MD_COPY_PATH)


if __name__ == "__main__":
    main()

