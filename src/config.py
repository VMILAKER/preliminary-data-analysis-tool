import os

from dotenv import load_dotenv
from reportlab.lib import colors
from reportlab.lib.styles import (TA_CENTER, TA_JUSTIFY, TA_LEFT,
                                  ParagraphStyle, getSampleStyleSheet)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import TableStyle

load_dotenv()


def get_streamlit_port():
    streamlit_port = os.getenv('STREAMLIT_PORT') if os.getenv(
        'STREAMLIT_PORT') else None
    if streamlit_port is not None:
        return streamlit_port


class TextStyle:
    def __init__(self, font_name: str, font_path: str):
        self.styles = getSampleStyleSheet()
        self.styles['Normal'].fontName = font_name
        pdfmetrics.registerFont(
            TTFont(font_name, os.path.join('fonts', font_path), 'UTF-8'))

    def justified_style(self, font_size):
        return ParagraphStyle(
            name="JustifiedStyle", parent=self.styles["Normal"], alignment=TA_JUSTIFY, leading=24, spaceAfter=24, firstLineIndent=25, fontSize=font_size)

    def centered_style(self, font_size):
        return ParagraphStyle(name="CenteredStyle", parent=self.styles["Normal"], alignment=TA_CENTER, spaceAfter=24, fontSize=font_size)

    def left_style(self, font_size):
        return ParagraphStyle(name="LeftStyle", parent=self.styles["Normal"], alignment=TA_LEFT, spaceAfter=16, fontSize=font_size)


def get_table_style():
    return TableStyle([
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ])
