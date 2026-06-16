from pathlib import Path
import sys
from typing import List

try:
    from reportlab.lib.enums import TA_CENTER, TA_RIGHT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
except ImportError as exc:
    raise SystemExit(
        "Missing reportlab. Install it with `python -m pip install reportlab`."
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "cover_letter_source.md"
OUTPUT = ROOT / "cover_letter_npj_computational_materials.pdf"


def _paragraphs(text: str) -> List[str]:
    return [block.strip().replace("\n", " ") for block in text.split("\n\n") if block.strip()]


def build_pdf() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    blocks = _paragraphs(text)

    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=0.85 * inch,
        leftMargin=0.85 * inch,
        topMargin=0.72 * inch,
        bottomMargin=0.72 * inch,
        title="Cover Letter - npj Computational Materials",
        author="Tianyi Chen",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CoverTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        spaceAfter=14,
    )
    date_style = ParagraphStyle(
        "DateRight",
        parent=styles["Normal"],
        alignment=TA_RIGHT,
        fontName="Times-Roman",
        fontSize=10.5,
        leading=14,
        spaceAfter=14,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Times-Roman",
        fontSize=10.6,
        leading=15.4,
        firstLineIndent=0,
        spaceAfter=9,
    )
    sign_style = ParagraphStyle(
        "Signature",
        parent=body_style,
        fontName="Times-Roman",
        leading=14.5,
        spaceAfter=4,
    )

    story = [
        Paragraph("Cover Letter", title_style),
        Paragraph(blocks[0], date_style),
    ]

    for block in blocks[1:9]:
        story.append(Paragraph(block, body_style))

    story.append(Spacer(1, 6))
    for block in blocks[9:]:
        story.append(Paragraph(block, sign_style))

    doc.build(story)


if __name__ == "__main__":
    if not SOURCE.exists():
        sys.exit(f"Missing source file: {SOURCE}")
    build_pdf()
    print(f"Wrote {OUTPUT}")
