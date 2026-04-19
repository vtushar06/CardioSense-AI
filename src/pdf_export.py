import io
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, HRFlowable, Table, TableStyle
)


def _style(name, base="Normal", **kwargs):
    styles = getSampleStyleSheet()
    s = ParagraphStyle(name=name, parent=styles[base], **kwargs)
    return s


def generate_pdf_report(patient: dict, prediction: dict, sections: dict) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    story = []

    title_style = _style("Title", base="Title",
                          fontSize=20, textColor=colors.HexColor("#1A365D"),
                          spaceAfter=4)
    subtitle_style = _style("Subtitle", base="Normal",
                             fontSize=11, textColor=colors.HexColor("#4A5568"),
                             spaceAfter=16)
    section_style = _style("Section", base="Heading2",
                            fontSize=13, textColor=colors.HexColor("#2B6CB0"),
                            spaceBefore=14, spaceAfter=6)
    body_style = _style("Body", base="Normal",
                         fontSize=10, leading=15,
                         textColor=colors.HexColor("#2D3748"))
    disclaimer_style = _style("Disclaimer", base="Normal",
                               fontSize=9, leading=13,
                               textColor=colors.HexColor("#718096"),
                               borderColor=colors.HexColor("#E2E8F0"),
                               borderWidth=1, borderPadding=8,
                               backColor=colors.HexColor("#F7FAFC"))

    risk_pct = round(prediction.get("probability", 0) * 100, 1)
    risk_label = prediction.get("risk_label", "Unknown")

    label_color = {
        "High Risk":   "#E53E3E",
        "Low Risk":    "#38A169",
        "Medium Risk": "#D69E2E",
    }.get(risk_label, "#2B6CB0")

    story.append(Paragraph("🫀 CardioSense AI", title_style))
    story.append(Paragraph("Cardiovascular Risk Assessment Report", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#2B6CB0")))
    story.append(Spacer(1, 0.4 * cm))

    sex = "Male" if patient.get("sex") == 1 else "Female"
    table_data = [
        ["Field", "Value"],
        ["Age", f"{patient.get('age')} years"],
        ["Sex", sex],
        ["Blood Pressure", f"{patient.get('trestbps')} mmHg"],
        ["Cholesterol", f"{patient.get('chol')} mg/dl"],
        ["Max Heart Rate", f"{patient.get('thalach')} bpm"],
        ["ST Depression", str(patient.get("oldpeak"))],
        ["Risk Level", risk_label],
        ["Risk Probability", f"{risk_pct}%"],
    ]

    table = Table(table_data, colWidths=[6 * cm, 10 * cm])
    table.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
        ("TEXTCOLOR",    (0, 0), (-1, 0), colors.white),
        ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, 0), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#EBF4FF"), colors.white]),
        ("TEXTCOLOR",    (0, 1), (-1, -1), colors.HexColor("#2D3748")),
        ("FONTNAME",     (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",     (0, 1), (-1, -1), 10),
        ("GRID",         (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
    ]))

    story.append(Paragraph("Patient Data", section_style))
    story.append(table)
    story.append(Spacer(1, 0.4 * cm))

    section_map = [
        ("Patient Risk Summary",          sections.get("risk_summary", "")),
        ("Key Contributing Factors",      sections.get("contributing", "")),
        ("Preventive Recommendations",    sections.get("recommendations", "")),
        ("When to Seek Medical Attention", sections.get("seek_attention", "")),
        ("References",                    sections.get("references", "")),
    ]

    for heading, content in section_map:
        if not content:
            continue
        story.append(Paragraph(heading, section_style))
        for line in content.split("\n"):
            line = line.strip()
            if not line:
                story.append(Spacer(1, 0.15 * cm))
                continue
            line = line.lstrip("- ").lstrip("* ")
            line = line.replace("**", "<b>", 1).replace("**", "</b>", 1)
            story.append(Paragraph(f"• {line}", body_style))
        story.append(Spacer(1, 0.2 * cm))

    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#E2E8F0")))
    story.append(Spacer(1, 0.3 * cm))

    disclaimer_text = sections.get(
        "disclaimer",
        "This report is for educational purposes only and does not constitute medical advice. "
        "Always consult a qualified healthcare professional."
    )
    story.append(Paragraph("⚕️ Medical Disclaimer", section_style))
    story.append(Paragraph(disclaimer_text, disclaimer_style))

    doc.build(story)
    return buffer.getvalue()
