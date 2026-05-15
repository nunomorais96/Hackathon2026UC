from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak
)

from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.platypus.tables import Table, TableStyle
from reportlab.lib import colors

import pandas as pd


def generate_pdf_report(
    filename,
    df,
    financial_summary,
    sentiment_summary,
    risk_summary,
    final_report
):

    doc = SimpleDocTemplate(
        filename,
        pagesize=letter
    )

    styles = getSampleStyleSheet()
    elements = []

    title = Paragraph(
        "AlphaLens Investment Research Brief",
        styles["Title"]
    )

    elements.append(title)
    elements.append(Spacer(1, 20))

    # Table
    elements.append(
        Paragraph("Company Comparison", styles["Heading2"])
    )

    table_df = df[
        [
            "ticker",
            "risk_score",
            "risk_level",
            "current_price",
            "pe_ratio"
        ]
    ].copy()

    table_data = [table_df.columns.tolist()]

    for _, row in table_df.iterrows():
        table_data.append(row.tolist())

    table = Table(table_data)

    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
        ])
    )

    elements.append(table)
    elements.append(Spacer(1, 20))

    # Sections
    sections = [
        ("Financial Agent Summary", financial_summary),
        ("News & Sentiment Agent Summary", sentiment_summary),
        ("Risk Agent Summary", risk_summary),
        ("Final Investment Brief", final_report),
    ]

    for title_text, content in sections:

        elements.append(
            Paragraph(title_text, styles["Heading2"])
        )

        cleaned_content = content.replace("\n", "<br/>")

        elements.append(
            Paragraph(cleaned_content, styles["BodyText"])
        )

        elements.append(Spacer(1, 20))

    disclaimer = Paragraph(
        "Disclaimer: Educational prototype only. Not financial advice.",
        styles["Italic"]
    )

    elements.append(disclaimer)

    doc.build(elements)