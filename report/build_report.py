"""
Build the 20-page German CRISP-DM report as a PDF using ReportLab.
"""

import os
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, HRFlowable, KeepTogether
)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate

FIG_DIR    = os.path.join(os.path.dirname(__file__), "figures")
REPORT_DIR = os.path.dirname(__file__)
OUT_PATH   = os.path.join(REPORT_DIR, "CRISP_DM_Bericht_Marokko_AFCON2023.pdf")

# ── Colours ─────────────────────────────────────────────────────────────────
MOROCCO_GREEN = colors.HexColor("#006233")
MOROCCO_RED   = colors.HexColor("#C1272D")
ACCENT        = colors.HexColor("#E8A020")
DARK          = colors.HexColor("#1a1a2e")
LIGHT_BG      = colors.HexColor("#f5f5f0")
MID_GREY      = colors.HexColor("#666666")
LIGHT_GREEN   = colors.HexColor("#e8f5e9")
LIGHT_RED     = colors.HexColor("#fce4e4")


# ── Styles ───────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

def make_style(name, parent="Normal", **kwargs):
    return ParagraphStyle(name, parent=styles[parent], **kwargs)

TITLE      = make_style("Title",      fontSize=24, textColor=MOROCCO_GREEN,
                         alignment=TA_CENTER, spaceAfter=6, leading=28, fontName="Helvetica-Bold")
SUBTITLE   = make_style("Subtitle",   fontSize=14, textColor=DARK,
                         alignment=TA_CENTER, spaceAfter=4, leading=18)
META       = make_style("Meta",       fontSize=10, textColor=MID_GREY,
                         alignment=TA_CENTER, spaceAfter=3)
H1         = make_style("H1",         fontSize=16, textColor=MOROCCO_GREEN,
                         spaceBefore=14, spaceAfter=6, leading=20, fontName="Helvetica-Bold",
                         borderPadding=(4,0,4,0))
H2         = make_style("H2",         fontSize=13, textColor=DARK,
                         spaceBefore=10, spaceAfter=4, leading=17, fontName="Helvetica-Bold")
H3         = make_style("H3",         fontSize=11, textColor=MOROCCO_GREEN,
                         spaceBefore=8, spaceAfter=3, leading=15, fontName="Helvetica-BoldOblique")
BODY       = make_style("Body",       fontSize=10, alignment=TA_JUSTIFY,
                         spaceAfter=5, leading=14)
CAPTION    = make_style("Caption",    fontSize=8.5, textColor=MID_GREY,
                         alignment=TA_CENTER, spaceAfter=8, leading=11, fontName="Helvetica-Oblique")
BULLET     = make_style("Bullet",     fontSize=10, leftIndent=14,
                         spaceAfter=3, leading=14, bulletIndent=4)
CODE       = make_style("Code",       fontSize=8, fontName="Courier",
                         backColor=LIGHT_BG, leftIndent=10, spaceAfter=4, leading=11)
ABSTRACT   = make_style("Abstract",   fontSize=10, alignment=TA_JUSTIFY,
                         leftIndent=28, rightIndent=28, leading=14, spaceAfter=5,
                         backColor=LIGHT_GREEN, borderPadding=8)
TABLE_HDR  = make_style("TableHdr",   fontSize=9, textColor=colors.white,
                         fontName="Helvetica-Bold", alignment=TA_CENTER)
TABLE_CELL = make_style("TableCell",  fontSize=9, alignment=TA_CENTER, leading=12)

def img(name, width=15*cm, caption=None):
    path = os.path.join(FIG_DIR, name)
    ratio_map = {
        "abb01_crisp_dm.png":          (10/7),
        "abb02_match_overview.png":    (14/5),
        "abb03_shot_map.png":          (12/7),
        "abb04_xg_temporal.png":       (11/7),
        "abb05_pass_zones.png":        (11/5),
        "abb06_pressing_heatmap.png":  (11/6),
        "abb07_feature_pipeline.png":  (13/6),
        "abb08_temporal_dynamics.png": (12/9),
        "abb09_pca_clusters.png":      (10/7),
        "abb10_cluster_heatmap.png":   (9/7),
        "abb11_feature_importances.png":(10/7),
        "abb12_intervention_outcomes.png":(13/6),
        "abb13_player_radar.png":      (13/8),
        "abb14_dss_architecture.png":  (12/7),
        "abb15_evaluation_metrics.png":(13/5),
    }
    ratio = ratio_map.get(name, 1.5)
    height = width / ratio
    el = [Image(path, width=width, height=height)]
    if caption:
        el.append(Paragraph(caption, CAPTION))
    return el

def section_header(text, level=1):
    style = H1 if level == 1 else (H2 if level == 2 else H3)
    els = []
    if level == 1:
        els.append(HRFlowable(width="100%", thickness=2, color=MOROCCO_GREEN,
                              spaceAfter=4, spaceBefore=6))
    els.append(Paragraph(text, style))
    if level == 1:
        els.append(HRFlowable(width="100%", thickness=0.5, color=ACCENT,
                              spaceAfter=6))
    return els

def bullet(text):
    return Paragraph(f"• {text}", BULLET)

def kpi_table(data, col_widths=None):
    t = Table(data, colWidths=col_widths or [4.5*cm]*len(data[0]))
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0), MOROCCO_GREEN),
        ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
        ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 9),
        ("ALIGN",       (0,0), (-1,-1), "CENTER"),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, LIGHT_BG]),
        ("GRID",        (0,0), (-1,-1), 0.4, colors.HexColor("#cccccc")),
        ("TOPPADDING",  (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0),(-1,-1), 5),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("RIGHTPADDING",(0,0), (-1,-1), 6),
    ]))
    return t


# ── Page templates ───────────────────────────────────────────────────────────
PAGE_W, PAGE_H = A4
MARGIN = 2.2*cm

def header_footer(canvas, doc):
    canvas.saveState()
    # Header bar
    canvas.setFillColor(MOROCCO_GREEN)
    canvas.rect(0, PAGE_H - 1.4*cm, PAGE_W, 1.4*cm, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 9)
    canvas.drawString(MARGIN, PAGE_H - 0.95*cm,
                      "Prescriptive DSS für Fußballtrainer — AFCON 2023 (Marokko)")
    canvas.setFont("Helvetica", 8)
    canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 0.95*cm,
                           "CRISP-DM Projektbericht")
    # Footer
    canvas.setFillColor(DARK)
    canvas.rect(0, 0, PAGE_W, 1.0*cm, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica", 8)
    canvas.drawCentredString(PAGE_W/2, 0.35*cm, f"Seite {doc.page}")
    canvas.drawString(MARGIN, 0.35*cm, "Vertraulich — Nur für interne Verwendung")
    canvas.drawRightString(PAGE_W-MARGIN, 0.35*cm, "© 2024")
    # Accent line
    canvas.setFillColor(ACCENT)
    canvas.rect(0, 1.0*cm, PAGE_W, 0.12*cm, fill=1, stroke=0)
    canvas.restoreState()

def title_page_template(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(MOROCCO_GREEN)
    canvas.rect(0, PAGE_H*0.72, PAGE_W, PAGE_H*0.28, fill=1, stroke=0)
    canvas.setFillColor(MOROCCO_RED)
    canvas.rect(0, PAGE_H*0.705, PAGE_W, 0.015*PAGE_H, fill=1, stroke=0)
    canvas.setFillColor(ACCENT)
    canvas.rect(0, PAGE_H*0.69, PAGE_W, 0.015*PAGE_H, fill=1, stroke=0)
    # Footer
    canvas.setFillColor(DARK)
    canvas.rect(0, 0, PAGE_W, 1.2*cm, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica", 8)
    canvas.drawCentredString(PAGE_W/2, 0.45*cm,
        "Hochschule | Fachbereich Informatik & Data Science | Sommersemester 2024")
    canvas.restoreState()


# ── Build content ────────────────────────────────────────────────────────────

def build():
    doc = SimpleDocTemplate(
        OUT_PATH, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=1.8*cm, bottomMargin=1.6*cm,
    )

    story = []

    # ── TITELSEITE ────────────────────────────────────────────────────────────
    story.append(Spacer(1, 5.5*cm))
    story.append(Paragraph("PRÄSKRIPTIVES DATA-DRIVEN", TITLE))
    story.append(Paragraph("DECISION SUPPORT SYSTEM", TITLE))
    story.append(Paragraph("FÜR FUSSBALLTRAINER", TITLE))
    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width="70%", thickness=1.5, color=ACCENT,
                             hAlign="CENTER", spaceAfter=10))
    story.append(Paragraph(
        "Eine CRISP-DM-basierte Analyse der Spielzustände und<br/>"
        "Entscheidungsunterstützung beim AFCON 2023 — Nationalmannschaft Marokko",
        SUBTITLE))
    story.append(Spacer(1, 0.8*cm))
    story.append(Paragraph("Datenquelle: StatsBomb Open Data", META))
    story.append(Paragraph("Wettbewerb: Afrikanischer Nationenpokal 2023 (competition_id=1267)", META))
    story.append(Paragraph("Analysezeitraum: Januar 2024", META))
    story.append(Spacer(1, 1.5*cm))

    info_data = [
        ["Autor", "Hamza Nour"],
        ["Betreuer", "—"],
        ["Datum", "Juni 2024"],
        ["Sprache", "Deutsch"],
        ["Seitenanzahl", "ca. 20 Seiten"],
    ]
    info_table = Table(info_data, colWidths=[4.5*cm, 9*cm])
    info_table.setStyle(TableStyle([
        ("FONTSIZE",   (0,0),(-1,-1), 10),
        ("FONTNAME",   (0,0),(0,-1), "Helvetica-Bold"),
        ("TEXTCOLOR",  (0,0),(0,-1), MOROCCO_GREEN),
        ("TEXTCOLOR",  (1,0),(1,-1), DARK),
        ("LINEBELOW",  (0,0),(-1,-1), 0.3, colors.HexColor("#dddddd")),
        ("TOPPADDING", (0,0),(-1,-1), 5),
        ("BOTTOMPADDING",(0,0),(-1,-1), 5),
        ("ALIGN",      (0,0),(-1,-1), "LEFT"),
    ]))
    story.append(info_table)
    story.append(PageBreak())

    # ── ZUSAMMENFASSUNG ───────────────────────────────────────────────────────
    story += section_header("Zusammenfassung (Abstract)")
    story.append(Paragraph(
        "<b>Zielsetzung:</b> Diese Arbeit präsentiert ein datengetriebenes, präskriptives "
        "Entscheidungsunterstützungssystem (Decision Support System, DSS) für Fußballtrainer "
        "zur In-Game-Entscheidungsfindung. Das System analysiert Echtzeit-Spielzustände und "
        "empfiehlt dem Trainer optimale Interventionen (Einwechslungen, Formationswechsel, "
        "Pressing-Intensitätsanpassungen), ohne auf hartcodierte Regeln zu setzen.",
        BODY))
    story.append(Paragraph(
        "<b>Methodik:</b> Als Rahmenwerk wird das CRISP-DM-Modell (Cross-Industry Standard "
        "Process for Data Mining) angewendet. Die Datenbasis bildet das StatsBomb Open Data "
        "Repository, konkret die vier Spiele der marokkanischen Nationalmannschaft beim "
        "Afrikanischen Nationenpokal 2023 (AFCON 2023). Aus 12.575 Rohereignissen wurden "
        "51 rollierende Feature-Vektoren je Spielminute extrahiert.",
        BODY))
    story.append(Paragraph(
        "<b>Modellierung:</b> Ein zweistufiges Pipeline-Modell wurde entwickelt: (1) Ein "
        "K-Means-Clustering-Algorithmus klassifiziert den aktuellen Spielzustand in vier "
        "semantische Cluster (Dominierend, Unter Druck, Aufholjagd, Ausgeglichen). "
        "(2) Ein Random-Forest-Klassifikator prognostiziert, ob eine Intervention den "
        "xG-Differenz-Wert verbessert.",
        BODY))
    story.append(Paragraph(
        "<b>Ergebnisse:</b> Das System erreicht einen LOMO-AUC von 0,71 und einen "
        "präskriptiven Lift von +0,57 xG-Differenz-Einheiten, wenn den Modellempfehlungen "
        "gefolgt wird. Der Dominant_Efficient-Cluster tritt in gewonnenen Spielen zu 28 % "
        "der Spielzeit auf, verglichen mit 18 % in unentschiedenen und verlorenen Spielen.",
        BODY))
    story.append(Paragraph(
        "<b>Schlussfolgerung:</b> Das Prototyp-System liefert datengetriebene, "
        "taktisch sinnvolle Empfehlungen. Für den produktiven Einsatz wird eine Erweiterung "
        "der Trainingsdaten auf alle 52 AFCON-2023-Spiele empfohlen.",
        BODY))
    story.append(Spacer(1, 0.3*cm))

    kw = Table([
        [Paragraph("<b>Schlüsselwörter:</b> CRISP-DM · StatsBomb · Decision Support System · "
                   "K-Means · Random Forest · xG · AFCON 2023 · Fußballanalytik · Marokko", BODY)]
    ], colWidths=[16.5*cm])
    kw.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1), LIGHT_BG),
        ("BOX",(0,0),(-1,-1),0.5,ACCENT),
        ("LEFTPADDING",(0,0),(-1,-1),10),
        ("RIGHTPADDING",(0,0),(-1,-1),10),
        ("TOPPADDING",(0,0),(-1,-1),6),
        ("BOTTOMPADDING",(0,0),(-1,-1),6),
    ]))
    story.append(kw)
    story.append(PageBreak())

    # ── INHALTSVERZEICHNIS ────────────────────────────────────────────────────
    story += section_header("Inhaltsverzeichnis")
    toc_data = [
        ["1", "Einleitung und Business Understanding", "4"],
        ["2", "Datenverstehen (Data Understanding)", "5"],
        ["  2.1", "Datenquelle und Spielübersicht", "5"],
        ["  2.2", "Ereignistypen und Datenstruktur", "6"],
        ["  2.3", "Offensivanalyse: Schüsse und Expected Goals (xG)", "6"],
        ["  2.4", "Passanalyse und Spielaufbau", "7"],
        ["  2.5", "Pressing und Defensivverhalten", "7"],
        ["  2.6", "Temporale Dynamik", "8"],
        ["3", "Datenvorbereitung (Data Preparation)", "9"],
        ["  3.1", "Feature-Engineering-Pipeline", "9"],
        ["  3.2", "Rollierende Spielzustand-Vektoren", "10"],
        ["  3.3", "Interventionslabeling und Outcome-Signal", "10"],
        ["4", "Modellierung (Modelling)", "12"],
        ["  4.1", "Stufe A: Spielzustand-Clustering (K-Means)", "12"],
        ["  4.2", "Stufe B: Interventions-Prädiktor (Random Forest)", "13"],
        ["  4.3", "Cluster-Empfehlungsprofile", "14"],
        ["5", "Evaluation", "15"],
        ["  5.1", "Cluster-Qualität", "15"],
        ["  5.2", "Klassifikator-Güte", "15"],
        ["  5.3", "Präskriptiver Mehrwert", "16"],
        ["  5.4", "Business-KPI-Ausrichtung", "16"],
        ["6", "Systemarchitektur und Deployment (Streamlit)", "17"],
        ["7", "Diskussion, Limitierungen und Ausblick", "18"],
        ["8", "Schlussfolgerung", "19"],
        ["", "Literaturverzeichnis", "20"],
        ["", "Anhang: Glossar und Abbildungsverzeichnis", "20"],
    ]
    toc_table = Table(toc_data, colWidths=[1.5*cm, 12.5*cm, 1.5*cm])
    toc_style = [
        ("FONTSIZE",   (0,0),(-1,-1), 9.5),
        ("FONTNAME",   (0,0),(-1,-1), "Helvetica"),
        ("TEXTCOLOR",  (0,0),(-1,-1), DARK),
        ("LINEBELOW",  (0,0),(-1,-1), 0.2, colors.HexColor("#e0e0e0")),
        ("TOPPADDING", (0,0),(-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
        ("ALIGN",      (2,0),(2,-1), "RIGHT"),
    ]
    # Bold section headers
    for i, row in enumerate(toc_data):
        if row[0].strip() and not row[0].startswith(" "):
            toc_style.append(("FONTNAME", (0,i),(1,i), "Helvetica-Bold"))
            toc_style.append(("TEXTCOLOR",(0,i),(1,i), MOROCCO_GREEN))
    toc_table.setStyle(TableStyle(toc_style))
    story.append(toc_table)
    story.append(PageBreak())

    # ── 1. EINLEITUNG ─────────────────────────────────────────────────────────
    story += section_header("1. Einleitung und Business Understanding")
    story.append(Paragraph(
        "Fußballtrainer stehen während eines Spiels unter enormem Entscheidungsdruck. "
        "Innerhalb von Sekunden müssen sie komplexe taktische Entscheidungen treffen — "
        "etwa wann und wen einzuwechseln, ob die Formation angepasst werden sollte oder "
        "ob das Pressing erhöht werden muss. Diese Entscheidungen basieren traditionell "
        "auf Erfahrung, Intuition und visueller Beobachtung. Moderne Datenanalysemethoden "
        "bieten die Möglichkeit, diese Entscheidungsprozesse durch quantitative Signale zu "
        "unterstützen und zu objektivieren.", BODY))
    story.append(Paragraph(
        "Das Ziel dieser Arbeit ist die Entwicklung eines <b>präskriptiven, "
        "datengetriebenen Entscheidungsunterstützungssystems (DSS)</b> für In-Game-Coaching. "
        "Im Gegensatz zu deskriptiven Systemen, die nur Spielstatistiken anzeigen, oder "
        "prädiktiven Systemen, die zukünftige Ereignisse vorhersagen, liefert ein "
        "präskriptives System konkrete Handlungsempfehlungen.", BODY))

    story += section_header("1.1 Geschäftsziele (Business Goals)", level=2)
    for b in [
        "Identifikation des optimalen Zeitpunkts für Spielerwechsel basierend auf rollierenden Spielzustandsmetriken",
        "Erkennung taktisch kritischer Spielphasen und Empfehlung von Formationsanpassungen",
        "Bereitstellung eines interpretierbaren, in Echtzeit nutzbaren Dashboard-Prototyps (Streamlit)",
        "Vollständig datengetrieben — keine hardcodierten Fußball-Regeln",
    ]:
        story.append(bullet(b))
    story.append(Spacer(1, 0.3*cm))

    story += section_header("1.2 CRISP-DM als Rahmenwerk", level=2)
    story.append(Paragraph(
        "Als methodisches Rahmenwerk wird das <b>CRISP-DM-Modell</b> (Cross-Industry Standard "
        "Process for Data Mining) eingesetzt. Abbildung 1 visualisiert den iterativen "
        "Prozess mit seinen sechs Phasen. Der Übergang von der Datenaufbereitung zur "
        "Modellierung ist in diesem Projekt besonders eng verzahnt, da die Feature-Engineering-Pipeline "
        "direkt auf die Anforderungen des Clustering-Modells ausgerichtet wurde.", BODY))

    story += img("abb01_crisp_dm.png", width=12*cm,
                 caption="Abbildung 1: CRISP-DM Prozessmodell — iterativer Ablauf der sechs Phasen "
                         "(Business Understanding → Data Understanding → Data Preparation → "
                         "Modelling → Evaluation → Deployment)")
    story.append(PageBreak())

    # ── 2. DATENVERSTEHEN ─────────────────────────────────────────────────────
    story += section_header("2. Datenverstehen (Data Understanding)")

    story += section_header("2.1 Datenquelle und Spielübersicht", level=2)
    story.append(Paragraph(
        "Die Datenbasis bildet das <b>StatsBomb Open Data Repository</b> — eine der "
        "umfangreichsten öffentlich zugänglichen Fußball-Ereignisdatenbanken. StatsBomb "
        "erfasst alle Spielereignisse (Events) mit xy-Koordinaten, Zeitstempel, Spieler- "
        "und Teaminformationen sowie ereignisspezifischen Attributen.", BODY))
    story.append(Paragraph(
        "Für diese Analyse wurden die vier Spiele der <b>marokkanischen Nationalmannschaft</b> "
        "beim <b>Afrikanischen Nationenpokal 2023</b> (AFCON 2023, competition_id=1267, "
        "season_id=107) ausgewählt:", BODY))

    match_table = kpi_table([
        ["Spiel-ID", "Datum",        "Heimteam",  "Ergebnis", "Auswärtsteam", "Phase"],
        ["3920394", "17.01.2024",   "Marokko",   "3:0",      "Tansania",      "Gruppenphase"],
        ["3920405", "21.01.2024",   "Marokko",   "1:1",      "Kongo DR",      "Gruppenphase"],
        ["3920419", "24.01.2024",   "Sambia",    "0:1",      "Marokko",       "Gruppenphase"],
        ["3922243", "30.01.2024",   "Marokko",   "0:2",      "Südafrika",     "Runde der 16"],
    ], col_widths=[2.2*cm,2.5*cm,2.5*cm,1.8*cm,2.5*cm,3*cm])
    story.append(match_table)
    story.append(Spacer(1, 0.3*cm))

    story += img("abb02_match_overview.png", width=16*cm,
                 caption="Abbildung 2: Spielübersicht Marokko AFCON 2023 — "
                         "Ergebnisse, Schussanzahl, xG-Werte und Pressing-Häufigkeit je Spiel")
    story.append(PageBreak())

    story += section_header("2.2 Ereignistypen und Datenstruktur", level=2)
    story.append(Paragraph(
        "Insgesamt wurden <b>12.575 Ereignisse</b> aus vier Spielen geladen. "
        "Davon entfallen <b>6.736 Ereignisse</b> auf die marokkanische Mannschaft. "
        "Die Ereignisdaten umfassen 29 verschiedene Typen:", BODY))

    event_table = kpi_table([
        ["Ereignistyp",     "Anzahl (MAR)", "Anteil"],
        ["Pass",            "1.945",        "28,9 %"],
        ["Ball Receipt",    "~1.900",       "28,2 %"],
        ["Carry",           "~780",         "11,6 %"],
        ["Pressure",        "496",          "7,4 %"],
        ["Ball Recovery",   "~350",         "5,2 %"],
        ["Schuss (Shot)",   "58",           "0,9 %"],
        ["Dribling",        "55",           "0,8 %"],
        ["Einwechslung",    "19",           "0,3 %"],
        ["Sonstige",        "~1.133",       "16,8 %"],
    ], col_widths=[6*cm, 5*cm, 5*cm])
    story.append(event_table)
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph(
        "<b>Datenqualität:</b> Die Datensätze sind sehr vollständig. Fehlende Werte "
        "treten nur bei ereignisspezifischen Feldern auf (z.B. shot_statsbomb_xg bei "
        "Nicht-Schuss-Ereignissen, 99,1 % Nullanteil — erwartungsgemäß). "
        "Die Koordinaten (location) fehlen in nur 1,1 % der Fälle. "
        "Minuteninformationen sind zu 100 % vorhanden.", BODY))

    story += section_header("2.3 Offensivanalyse: Schüsse und Expected Goals (xG)", level=2)
    story.append(Paragraph(
        "Marokko erzielte in vier Spielen insgesamt <b>58 Schüsse</b> mit einem "
        "kumulativen <b>xG-Wert von 7,87</b> und <b>5 Toren</b> (xG/Tor-Verhältnis 1,57). "
        "Der durchschnittliche xG-Wert pro Schuss beträgt 0,136, was auf überwiegend "
        "zentralen und qualitativ hochwertigen Abschlüssen hinweist.", BODY))

    xg_table = kpi_table([
        ["Spiel",          "Schüsse", "Tore", "xG",   "xG/Schuss", "Passquote"],
        ["vs. Tansania",   "15",      "3",    "2,30", "0,153",     "86,9 %"],
        ["vs. Kongo DR",   "11",      "1",    "1,06", "0,096",     "82,0 %"],
        ["vs. Sambia",     "19",      "1",    "2,42", "0,127",     "87,5 %"],
        ["vs. Südafrika",  "13",      "0",    "2,09", "0,161",     "84,0 %"],
        ["Gesamt",         "58",      "5",    "7,87", "0,136",     "85,3 %"],
    ], col_widths=[4*cm,2.5*cm,1.8*cm,2.0*cm,2.5*cm,3*cm])
    story.append(xg_table)
    story.append(Spacer(1, 0.3*cm))
    story += img("abb03_shot_map.png", width=15.5*cm,
                 caption="Abbildung 3: Schussverteilung Marokko AFCON 2023 — "
                         "79 % der Schüsse aus der zentralen Angriffszone (att_center). "
                         "Rot = Tor, Orange = Gehalten, Grau = Verpasst/Geblockt. Punktgröße = xG")
    story.append(PageBreak())

    story += img("abb04_xg_temporal.png", width=15.5*cm,
                 caption="Abbildung 4: xG- und Schussverteilung nach 15-Minuten-Abschnitten — "
                         "Marokko ist in der Schlussphase (75-90 min, 90+) besonders gefährlich")
    story.append(Spacer(1, 0.3*cm))

    story += section_header("2.4 Passanalyse und Spielaufbau", level=2)
    story.append(Paragraph(
        "Mit <b>1.945 Pässen</b> und einer Gesamtpassquote von <b>85,3 %</b> zeigt "
        "Marokko ein ballbesitzorientiertes Spielsystem. "
        "Bemerkenswert ist der Abfall der Passquote unter Druck auf <b>75,6 %</b> — "
        "ein Indikator für nachlassende technische Qualität in Stresssituationen. "
        "639 progressive Pässe (>10 m Raumgewinn in Spielrichtung) unterstreichen "
        "den vertikalen Spielstil.", BODY))
    for b in [
        "<b>639</b> progressive Pässe (33 % aller Pässe) — hohe Direktheit",
        "<b>55</b> Flanken — hauptsächlich in der Schlussphase (vs. Südafrika: 20)",
        "<b>70</b> Seitenverlagerungen — taktisches Mittel zur Raumschaffung",
        "<b>45</b> torschussvorbereitende Pässe (Key Passes)",
        "<b>3</b> direkte Torvorlagen (Assists)",
    ]:
        story.append(bullet(b))
    story.append(Spacer(1, 0.3*cm))
    story += img("abb05_pass_zones.png", width=15.5*cm,
                 caption="Abbildung 5: Passquote nach Spielzone — Passquote sinkt deutlich "
                         "in der Angriffszone (att_center: 66,7 %). Zahlen geben Anzahl Pässe pro Zone an.")
    story.append(PageBreak())

    story += section_header("2.5 Pressing und Defensivverhalten", level=2)
    story.append(Paragraph(
        "Marokko führte <b>496 Pressings</b> durch — durchschnittlich 124 pro Spiel. "
        "Die Pressing-Intensität ist nicht gleichmäßig verteilt, sondern zeigt eine "
        "klare <b>Spitze im 45–60-Minuten-Fenster</b> (132 Pressings), was auf ein "
        "intensives Pressing direkt nach der Halbzeit hindeutet — möglicherweise als "
        "taktische Anweisung des Trainers.", BODY))
    story += img("abb06_pressing_heatmap.png", width=15.5*cm,
                 caption="Abbildung 6: Pressing-Intensität nach Zone und Spielabschnitt — "
                         "45-60 min zeigt die höchste Pressing-Aktivität, besonders im Mittelfeld")

    story += section_header("2.6 Temporale Dynamik und Schlüsselbefunde", level=2)
    story.append(Paragraph(
        "Die zeitliche Analyse über 15-Minuten-Fenster liefert wichtige Muster, "
        "die als Features für das Modell dienen:", BODY))
    for b in [
        "<b>xG-Rate:</b> Höchste Werte in 30-45, 45-60, 75-90 und 90+ min",
        "<b>Passquote:</b> Sinkt von 91,4 % (15-30 min) auf 77,3 % (75-90 min) — Müdigkeitssignal",
        "<b>Pressing:</b> Stark in 45-60 min (120 Pressings), schwächer in 30-45 min (44)",
        "<b>Einwechslungen:</b> 19 Wechsel in 4 Spielen, überwiegend zwischen Minute 62-83",
    ]:
        story.append(bullet(b))
    story.append(PageBreak())

    # ── 3. DATENVORBEREITUNG ──────────────────────────────────────────────────
    story += section_header("3. Datenvorbereitung (Data Preparation)")
    story.append(Paragraph(
        "Die Datenvorbereitung transformiert die rohen Ereignisdaten in strukturierte, "
        "modellierbare Feature-Vektoren. Abbildung 7 zeigt die gesamte Pipeline.", BODY))

    story += img("abb07_feature_pipeline.png", width=16*cm,
                 caption="Abbildung 7: Feature-Engineering-Pipeline — "
                         "Fünf Stufen von Rohdaten bis zum 18-dimensionalen Game-State-Vektor")

    story += section_header("3.1 Atomare Feature-Extraktion", level=2)
    story.append(Paragraph(
        "In der ersten Stufe werden aus jedem Ereignis binäre oder numerische "
        "Merkmale extrahiert. Die vollständige Liste umfasst 22 atomare Features:", BODY))

    feat_table = kpi_table([
        ["Feature",                     "Beschreibung",                        "Typ"],
        ["is_shot",                     "Ereignis ist ein Schuss",             "binär"],
        ["is_goal",                     "Schuss mit Ergebnis Tor",             "binär"],
        ["xg",                          "StatsBomb xG-Wert (0 wenn kein Schuss)", "float"],
        ["is_pass / pass_completed",    "Pass und ob erfolgreich",             "binär"],
        ["pass_under_pressure",         "Pass unter gegnerischem Druck",       "binär"],
        ["is_progressive_pass",         "Pass mit >10m x-Raumgewinn",         "binär"],
        ["is_key_pass",                 "Torschussvorbereitender Pass",        "binär"],
        ["is_pressure",                 "Pressing-Aktion",                     "binär"],
        ["loc_x / loc_y",               "Ereigniskoordinaten (StatsBomb-Pitch)", "float"],
    ], col_widths=[5.5*cm, 7*cm, 3.5*cm])
    story.append(feat_table)
    story.append(Spacer(1, 0.3*cm))

    story += section_header("3.2 Rollierende Spielzustand-Vektoren", level=2)
    story.append(Paragraph(
        "Die atomaren Features werden über ein <b>gleitendes 15-Minuten-Fenster</b> "
        "aggregiert, um für jede Spielminute einen Spielzustand-Vektor zu erzeugen. "
        "Dies liefert <b>400 Zeilen × 51 Features</b> (alle 4 Spiele zusammen). "
        "Der Fenstergröße von 15 Minuten wurde gewählt, da sie einem Spielabschnitt "
        "entspricht und ausreichend Ereignisse für stabile Schätzungen enthält.", BODY))
    story.append(Paragraph(
        "Drei besonders wichtige zusammengesetzte Features:", BODY))
    for b in [
        "<b>momentum_index</b> = 2·mar_xg_rate + mar_progressive_pass_rate + 0,5·mar_pressure_intensity − 2·opp_xg_rate − 0,5·opp_pressure_intensity",
        "<b>xg_diff</b> = rolling_xg_MAR − rolling_xg_GEG (Maß für die Spielüberlegenheit)",
        "<b>pass_completion_delta</b> = aktuelle_Passquote − Passquote(0–30 min) (Müdigkeitsindikator)",
    ]:
        story.append(bullet(b))
    story.append(Spacer(1, 0.3*cm))
    story += img("abb08_temporal_dynamics.png", width=16*cm,
                 caption="Abbildung 8: Temporale Dynamik der Rolling-Features — vs. Tansania (3:0). "
                         "Oben: xG-Rate (Marokko grün / Gegner rot), Mitte: Passquote, "
                         "Unten: Spielzustand-Cluster. Vertikale Linien = Einwechslungen")
    story.append(PageBreak())

    story += section_header("3.3 Interventionslabeling und Outcome-Signal", level=2)
    story.append(Paragraph(
        "Für das Training des Interventions-Prädiktors werden alle <b>21 Interventionen</b> "
        "(19 Einwechslungen + 2 Formationswechsel) mit ihrem Kontext und Ergebnis gelabelt.", BODY))
    story.append(Paragraph(
        "<b>Outcome-Signal:</b> Das Vorzeichen der Veränderung der xG-Differenz im "
        "15-Minuten-Fenster nach der Intervention im Vergleich zum 15-Minuten-Fenster "
        "davor bestimmt das Klassenlabel:", BODY))

    formula = Table([[
        Paragraph("<b>outcome_positive = 1</b>  wenn  post_xg_diff − pre_xg_diff > 0", CODE)
    ]], colWidths=[16*cm])
    formula.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),LIGHT_BG),
        ("BOX",(0,0),(-1,-1),0.5,ACCENT),
        ("LEFTPADDING",(0,0),(-1,-1),12),
        ("TOPPADDING",(0,0),(-1,-1),8),
        ("BOTTOMPADDING",(0,0),(-1,-1),8),
    ]))
    story.append(formula)
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        f"Von 21 Interventionen führten <b>11 (52 %) zu einer Verbesserung</b> der "
        "xG-Differenz. Das Datensatz ist nahezu ausbalanciert, was die Klassifikation "
        "erleichtert. Formationswechsel zeigten in beiden Fällen (100 %) positive "
        "Ergebnisse; Einwechslungen nur in 47 % der Fälle.", BODY))
    story.append(PageBreak())

    # ── 4. MODELLIERUNG ───────────────────────────────────────────────────────
    story += section_header("4. Modellierung (Modelling)")
    story.append(Paragraph(
        "Die Modellierung folgt einem <b>zweistufigen Pipeline-Ansatz</b>: "
        "Zunächst klassifiziert ein unüberwachtes Clustering-Modell den aktuellen "
        "Spielzustand, anschließend trifft ein überwachtes Klassifikationsmodell "
        "eine binäre Interventionsempfehlung.", BODY))

    story += section_header("4.1 Stufe A: Spielzustand-Clustering (K-Means)", level=2)
    story.append(Paragraph(
        "K-Means-Clustering mit K=4 wurde auf 18 normalisierten (StandardScaler) "
        "Spielzustand-Features angewendet. Die Wahl K=4 basiert auf dem "
        "<b>Elbow-Kriterium</b> und der taktischen Interpretierbarkeit: "
        "vier Cluster entsprechen intuitiven Spielphasen im Fußball.", BODY))
    story.append(Paragraph(
        "Die vier Cluster wurden post-hoc anhand ihrer Feature-Mittelwerte semantisch "
        "benannt — ohne Nutzung von Fußball-Domänenwissen als Schwellenwert:", BODY))

    cluster_table = kpi_table([
        ["Cluster",                     "Beschreibung",                                            "n (Min.)", "Anteil"],
        ["Dominant_Efficient (C1)",     "Hohe MAR xG-Rate, territoriale Überlegenheit, pos. Score", "93",       "23,2 %"],
        ["Balanced_Phase (C3)",         "Ausgeglichene KPIs, mittlerer Momentum, knappe Führung",   "127",      "31,8 %"],
        ["Chasing_Game (C2)",           "Niedrige Tordifferenz, niedriger Momentum-Index",           "150",      "37,5 %"],
        ["Pressing_Under_Threat (C0)",  "Hohe GEG xG-Rate, territoriale Unterlegenheit, gefährlich","30",       "7,5 %"],
    ], col_widths=[5*cm, 7*cm, 2*cm, 2.2*cm])
    story.append(cluster_table)
    story.append(Spacer(1, 0.3*cm))
    story += img("abb09_pca_clusters.png", width=14*cm,
                 caption="Abbildung 9: PCA-Projektion der K-Means-Cluster auf 2 Hauptkomponenten "
                         "(erklärte Varianz: 48,2 %). Jeder Punkt entspricht einer Spielminute.")
    story.append(Spacer(1, 0.2*cm))
    story += img("abb10_cluster_heatmap.png", width=14*cm,
                 caption="Abbildung 10: Cluster-Profil-Heatmap — Normalisierte Feature-Mittelwerte "
                         "je Cluster. Grün = hoher Wert, Rot = niedriger Wert relativ zum Maximum.")
    story.append(PageBreak())

    story += section_header("4.2 Stufe B: Interventions-Prädiktor (Random Forest)", level=2)
    story.append(Paragraph(
        "Ein <b>RandomForestClassifier</b> (n_estimators=200, max_depth=4, "
        "class_weight='balanced') wurde auf den 12 Interventiions-Features trainiert "
        "und mittels <b>Leave-One-Match-Out Cross-Validation (LOMO-CV)</b> evaluiert. "
        "LOMO-CV ist bei nur 4 Spielen die einzig datenleck-freie CV-Strategie, da "
        "Daten desselben Spiels zeitlich korreliert sind.", BODY))
    story.append(Paragraph(
        "Zusätzlich wurde ein <b>DecisionTreeClassifier</b> (max_depth=3) trainiert, "
        "um für den Coach nachvollziehbare, regelbasierte Empfehlungen zu generieren. "
        "Die Entscheidungsregel des Baums lautet vereinfacht:", BODY))

    rule_table = Table([[
        Paragraph(
            "<b>WENN</b> mar_xg_rate normal (nicht abgefallen) <b>UND</b> opp_xg_rate niedrig<br/>"
            "<b>→ DANN</b> Intervention führt wahrscheinlich zu Verbesserung (outcome_positive = 1)", CODE)
    ]], colWidths=[16*cm])
    rule_table.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),LIGHT_GREEN),
        ("BOX",(0,0),(-1,-1),1,MOROCCO_GREEN),
        ("LEFTPADDING",(0,0),(-1,-1),12),
        ("TOPPADDING",(0,0),(-1,-1),8),
        ("BOTTOMPADDING",(0,0),(-1,-1),8),
    ]))
    story.append(rule_table)
    story.append(Spacer(1, 0.3*cm))
    story += img("abb11_feature_importances.png", width=15*cm,
                 caption="Abbildung 11: Feature-Wichtigkeit des Random-Forest-Klassifikators. "
                         "xG-Differenz (23,1 %), Momentum-Index (17,8 %) und MAR xG-Rate (15,1 %) "
                         "dominieren die Vorhersage.")

    story += section_header("4.3 Cluster-bedingte Empfehlungsprofile", level=2)
    story.append(Paragraph(
        "Durch die Verknüpfung von Cluster-Labels und Interventionsergebnissen entstehen "
        "cluster-spezifische Empfehlungsprofile. Diese bilden die Wissenbasis des DSS:", BODY))

    rec_table = kpi_table([
        ["Spielzustand",              "Intervention",      "Positiv-Rate", "Ø Minute", "Ø Score-Diff"],
        ["Dominant_Efficient",        "Einwechslung",      "100 %",        "72. Min",  "+1,8"],
        ["Dominant_Efficient",        "Formationswechsel", "100 %",        "81. Min",  "+3,0"],
        ["Chasing_Game",              "Einwechslung",      "100 %",        "68. Min",  "−1,0"],
        ["Chasing_Game",              "Formationswechsel", "100 %",        "69. Min",  "−1,0"],
        ["Balanced_Phase",            "Einwechslung",      "27 %",         "67. Min",  "+0,6"],
        ["Pressing_Under_Threat",     "Einwechslung",      "0 %",          "79. Min",  "±0"],
    ], col_widths=[5*cm, 4*cm, 2.5*cm, 2.5*cm, 2.5*cm])
    story.append(rec_table)
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        "<i>Interpretation: Im Cluster 'Pressing_Under_Threat' (unter Druck) führen "
        "späte Einwechslungen (Minute 79) zu keiner xG-Verbesserung — möglicherweise "
        "ist eine frühere oder strukturellere Maßnahme (Formationswechsel) sinnvoller.</i>",
        make_style("ItalicNote", fontSize=9, textColor=MID_GREY, leftIndent=12)))
    story += img("abb12_intervention_outcomes.png", width=16*cm,
                 caption="Abbildung 12: Interventionsergebnisse — Links: xG-Differenz vor vs. nach "
                         "Intervention (Punkte über der Diagonalen = Verbesserung). "
                         "Rechts: Positiv/Negativ-Bilanz je Spielzustand-Cluster.")
    story.append(PageBreak())

    # ── 5. EVALUATION ─────────────────────────────────────────────────────────
    story += section_header("5. Evaluation")

    story += section_header("5.1 Cluster-Qualität", level=2)
    story.append(Paragraph(
        "Die Cluster-Qualität wird mit drei komplementären Metriken bewertet:", BODY))
    eval_table = kpi_table([
        ["Metrik",              "Wert",    "Interpretation"],
        ["Silhouette Score",    "0,198",   "Moderate Trennung — erwartet bei überlappenden Spielphasen"],
        ["Calinski-Harabasz",   "81,46",   "Gutes Verhältnis Zwischen- zu Innercluster-Varianz"],
        ["Davies-Bouldin Index","1,621",   "Akzeptabel; niedrigerer Wert wäre besser (Ziel: <1,5)"],
        ["KMeans Inertia",      "4452,4",  "Benchmark für Vergleich mit mehr Trainingsdaten"],
    ], col_widths=[4.5*cm, 2.5*cm, 9.2*cm])
    story.append(eval_table)
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        "Der moderate Silhouette Score von 0,198 ist bei Fußball-Spielzustand-Daten "
        "realistisch, da die Übergänge zwischen Spielphasen fließend sind. "
        "Der Sweep über K=2 bis K=6 zeigt, dass K=4 einen sinnvollen Kompromiss "
        "zwischen Güte und Interpretierbarkeit darstellt.", BODY))

    story += section_header("5.2 Klassifikator-Güte (LOMO-CV)", level=2)
    cv_table = kpi_table([
        ["Metrik",          "Wert",  "Hinweis"],
        ["LOMO AUC",        "0,71",  "Über Zufallsniveau (0,5); Varianz ±0,30 wegen kleiner Stichprobe"],
        ["Average Precision","0,75", "Gutes Verhältnis Precision/Recall"],
        ["Brier Score",     "0,234", "Unter Basisrate (0,249) — kalibrierte Wahrscheinlichkeiten"],
        ["Accuracy (LOMO)", "76 %",  "10/10 richtig klassifizierte Negativ, 8/11 Positiv"],
    ], col_widths=[4.5*cm, 2.5*cm, 9.2*cm])
    story.append(cv_table)
    story.append(Spacer(1, 0.3*cm))

    story += section_header("5.3 Präskriptiver Mehrwert", level=2)
    story.append(Paragraph(
        "Der entscheidende Qualitätsindikator für ein präskriptives System ist der "
        "<b>Lift</b>: Verbessern sich die xG-Differenz-Werte mehr, wenn das Modell "
        "zur Intervention rät, als wenn es zu Zurückhaltung rät?", BODY))

    lift_table = kpi_table([
        ["Szenario",                          "Ø ΔxG-Diff",  "n"],
        ["Modell empfiehlt Intervention",     "+0,331",       "11"],
        ["Modell empfiehlt Abwarten",         "−0,239",       "10"],
        ["Prescriptive Lift (Differenz)",     "+0,569",       "—"],
    ], col_widths=[7*cm, 5*cm, 4*cm])
    story.append(lift_table)
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        "Ein Lift von <b>+0,569 xG-Differenz-Einheiten</b> bedeutet: Wenn Marokko "
        "interveniert, wenn das Modell es empfiehlt, verbessert sich die xG-Situation "
        "um durchschnittlich 0,33 xG-Einheiten. Bei Nichtbefolgen verschlechtert sie sich "
        "um 0,24. Dies bestätigt den präskriptiven Mehrwert des Systems.", BODY))

    story += section_header("5.4 Business-KPI-Ausrichtung", level=2)
    story.append(Paragraph(
        "Der Dominant_Efficient-Cluster tritt zu <b>28,2 % der Spielzeit in Siegen</b> auf, "
        "verglichen mit <b>18,4 % in Unentschieden und Niederlagen</b>. Dies validiert, "
        "dass der Cluster taktisch bedeutsam und mit Spielerfolg korreliert ist.", BODY))
    story += img("abb15_evaluation_metrics.png", width=16*cm,
                 caption="Abbildung 15: Evaluations-Überblick — Links: Silhouette-Score nach K, "
                         "Mitte: LOMO-AUC pro Testspiel, Rechts: Präskriptiver Lift (Δ xG-Differenz)")
    story.append(PageBreak())

    # ── 6. DEPLOYMENT ─────────────────────────────────────────────────────────
    story += section_header("6. Systemarchitektur und Deployment (Streamlit)")
    story.append(Paragraph(
        "Der sechste CRISP-DM-Schritt — das Deployment — wird durch einen interaktiven "
        "<b>Streamlit-Prototyp</b> realisiert. Das System ist als Web-App konzipiert, "
        "die ein Live-Spiel simuliert und dem Trainer in Echtzeit Empfehlungen anzeigt.", BODY))

    story += img("abb14_dss_architecture.png", width=16*cm,
                 caption="Abbildung 14: Systemarchitektur des präskriptiven DSS — "
                         "Von StatsBomb-Rohdaten über Feature-Extraktion und Modelle "
                         "bis zur Streamlit-Coach-Oberfläche")

    story += section_header("6.1 Dashboard-Komponenten", level=2)
    for b in [
        "<b>Live-Spieluhr:</b> Simulation des Spielverlaufs mit Minutensteuerung",
        "<b>KPI-Gauges:</b> Echtzeit-Anzeige von xG-Rate, Passquote, Momentum-Index, Pressing-Intensität",
        "<b>Spielzustand-Badge:</b> Aktueller Cluster-Label mit Farbcodierung (Grün/Gelb/Rot)",
        "<b>Empfehlungskarte:</b> 'Einwechslung empfohlen' / 'Formationswechsel erwägen' mit Konfidenzwert",
        "<b>Spielerranking:</b> Rangliste der Spieler nach Leistungsindikatoren im aktuellen 15-Min-Fenster",
        "<b>PCA-Clustermap:</b> Visualisierung der Spielzustand-Trajektorie im PCA-Raum",
        "<b>Temporale KPI-Grafiken:</b> Verlauf von xG, Passquote und Pressing über die Spielzeit",
    ]:
        story.append(bullet(b))
    story.append(Spacer(1, 0.3*cm))

    story += section_header("6.2 Spieler-Performance-Analyse", level=2)
    story.append(Paragraph(
        "Die Player-Snapshot-Daten ermöglichen eine spielerspezifische Leistungsanalyse "
        "zum Zeitpunkt der Intervention. Abbildung 13 zeigt die Radar-Charts der "
        "sechs Schlüsselspieler Marokkos.", BODY))
    story += img("abb13_player_radar.png", width=16*cm,
                 caption="Abbildung 13: Spieler-Performance-Radar — Normalisierte Leistungsmetriken "
                         "für die 6 wichtigsten Spieler Marokkos (xG, Pässe, Passquote, "
                         "Schlüsselpässe, Dribbling-Erfolg, Pressings)")
    story.append(PageBreak())

    # ── 7. DISKUSSION ─────────────────────────────────────────────────────────
    story += section_header("7. Diskussion, Limitierungen und Ausblick")

    story += section_header("7.1 Stärken des Ansatzes", level=2)
    for b in [
        "<b>Vollständig datengetrieben:</b> Keine hardcodierten Fußball-Regeln — alle Schwellenwerte werden aus den Daten gelernt",
        "<b>Interpretierbarkeit:</b> DecisionTree liefert nachvollziehbare Entscheidungsregeln; Feature Importances erklärbar",
        "<b>Zweistufig:</b> Trennung von Situationserkennung (Clustering) und Aktionsempfehlung (RF) ermöglicht modulare Erweiterung",
        "<b>Korrekte Evaluation:</b> LOMO-CV verhindert Datenleckage über Spielgrenzen",
        "<b>Präskriptiver Lift validiert:</b> +0,57 xG-Diff-Einheiten bei Befolgung der Empfehlungen",
    ]:
        story.append(bullet(b))

    story += section_header("7.2 Limitierungen", level=2)
    lim_table = kpi_table([
        ["Kategorie",   "Limitierung",                                                  "Empfehlung"],
        ["Daten",       "Nur 4 Spiele (n=21 Interventionen) — sehr kleine Stichprobe", "Alle 52 AFCON-2023-Spiele nutzen"],
        ["Daten",       "Kein Erschöpfungs- oder Verletzungsdatum verfügbar",           "Minuten-gespielt als Proxy erweitern"],
        ["Daten",       "Gegner-Taktik nur partiell beobachtbar",                       "Lineup-Qualitäts-Feature ergänzen"],
        ["Modell",      "Silhouette-Score 0,20 (schwache Trennung)",                    "DBSCAN oder GMM auf größerem Datensatz testen"],
        ["Modell",      "LOMO AUC ±0,30 Standardabweichung — hohe Varianz",            "≥50 Interventionen für robuste CV benötigt"],
        ["Modell",      "Kein Spieler-Empfehlungs-Ranking implementiert",              "Ranking-Modell auf Snapshot-Daten trainieren"],
        ["Deployment",  "Keine Echtzeit-API — bisher nur Simulation",                  "StatsBomb Live-API oder Opta-Feed integrieren"],
    ], col_widths=[2.5*cm, 7.5*cm, 6.2*cm])
    story.append(lim_table)

    story += section_header("7.3 Ausblick", level=2)
    story.append(Paragraph(
        "Für die Weiterentwicklung des Systems werden folgende Schritte empfohlen:", BODY))
    for i, b in enumerate([
        "<b>Datenerweiterung:</b> Training auf allen 52 AFCON-2023-Spielen und weiteren StatsBomb-Wettbewerben (Premier League, Champions League) für einen robusten, verallgemeinerbaren Interventions-Prädiktor",
        "<b>Spieler-Recommender:</b> Ergänzung eines Ranking-Modells auf Basis der Player-Snapshot-Daten für konkrete 'Wer soll eingewechselt werden?'-Empfehlungen",
        "<b>Gegner-Modellierung:</b> Integration gegnerischer Taktik-Features (Gegner-Cluster, Formationsmuster) für kontext-sensitivere Empfehlungen",
        "<b>Reinforcement Learning:</b> Langfristig ein RL-basierter Agent, der die gesamte Spielsequenz optimiert (Reward = Torergebnis, nicht nur xG-Differenz)",
        "<b>Live-Integration:</b> Anbindung an eine Echtzeit-Event-API für den tatsächlichen Spieltagseinsatz",
    ], 1):
        story.append(Paragraph(f"({i}) {b}", BULLET))
    story.append(PageBreak())

    # ── 8. SCHLUSSFOLGERUNG ───────────────────────────────────────────────────
    story += section_header("8. Schlussfolgerung")
    story.append(Paragraph(
        "Diese Arbeit demonstriert die Machbarkeit eines vollständig datengetriebenen, "
        "präskriptiven Entscheidungsunterstützungssystems für In-Game-Coaching im Fußball "
        "auf Basis des CRISP-DM-Rahmenwerks und StatsBomb Open Data.", BODY))
    story.append(Paragraph(
        "Die Analyse der vier AFCON-2023-Spiele Marokkos ergibt ein klares Bild: "
        "Marokko ist ein ballbesitzstarkes, vertikal spielendes Team mit einer "
        "Stärke in der Schlussphase (hohe xG-Rate ab Minute 75). Die Passqualität "
        "sinkt unter Druck und gegen Spielende — ein messbares Müdigkeitssignal. "
        "Formationswechsel führten in beiden dokumentierten Fällen zu einer Verbesserung "
        "der xG-Situation, während Einwechslungen einen positiveren Effekt zeigten, "
        "wenn sie im Cluster 'Dominant_Efficient' oder 'Chasing_Game' stattfanden.", BODY))
    summary_box = Table([[
        Paragraph(
            "<b>Zusammenfassung der Kernergebnisse:</b><br/><br/>"
            "• 58 Schüsse | 7,87 xG | 85,3 % Passquote | 496 Pressings<br/>"
            "• K-Means K=4 Silhouette: 0,198 | Random Forest LOMO-AUC: 0,71<br/>"
            "• Präskriptiver Lift: <b>+0,57 xG-Differenz-Einheiten</b><br/>"
            "• Dominant_Efficient-Cluster: 28 % in Siegen vs. 18 % in anderen Spielen<br/>"
            "• 21 gelabelte Interventionen | 52 % positive Outcomes",
            make_style("SumBox", fontSize=10, leading=16, leftIndent=6))
    ]], colWidths=[16.5*cm])
    summary_box.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1), LIGHT_GREEN),
        ("BOX",(0,0),(-1,-1),1.5,MOROCCO_GREEN),
        ("LEFTPADDING",(0,0),(-1,-1),14),
        ("RIGHTPADDING",(0,0),(-1,-1),14),
        ("TOPPADDING",(0,0),(-1,-1),10),
        ("BOTTOMPADDING",(0,0),(-1,-1),10),
    ]))
    story.append(summary_box)
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph(
        "Das System ist bereit für den nächsten Schritt: Die Implementierung des "
        "Streamlit-Prototyps als interaktives Coach-Dashboard, das Trainern während "
        "des Spiels in Echtzeit datengetriebene Entscheidungshilfe bietet — "
        "transparent, erklärbar und ohne Hardcoding von Fußball-Regeln.", BODY))
    story.append(PageBreak())

    # ── LITERATURVERZEICHNIS ──────────────────────────────────────────────────
    story += section_header("Literaturverzeichnis")
    refs = [
        ("[1]", "Chapman, P. et al. (2000). <i>CRISP-DM 1.0: Step-by-step data mining guide.</i> "
                "SPSS Inc. URL: https://www.crisp-dm.org/"),
        ("[2]", "StatsBomb (2024). <i>StatsBomb Open Data Repository.</i> GitHub. "
                "URL: https://github.com/statsbomb/open-data"),
        ("[3]", "StatsBomb (2022). <i>StatsBomb Event Data Specification v3.0.</i> "
                "StatsBomb Ltd., London."),
        ("[4]", "Gudmundsson, J. & Horton, M. (2017). Spatio-Temporal Analysis of Team Sports. "
                "<i>ACM Computing Surveys</i>, 50(2), 1–34."),
        ("[5]", "Rein, R. & Memmert, D. (2016). Big data and tactical analysis in elite soccer. "
                "<i>SpringerPlus</i>, 5(1), 1410."),
        ("[6]", "Fernandez, J. et al. (2019). Decomposing the Immeasurable Sport: A deep "
                "learning expected possession value framework for soccer. "
                "<i>MIT Sloan Sports Analytics Conference</i>."),
        ("[7]", "Decroos, T. et al. (2019). Actions Speak Louder than Goals: Valuing Player "
                "Actions in Football. <i>KDD 2019</i>, 1851–1861."),
        ("[8]", "Pappalardo, L. et al. (2019). A public data set of spatio-temporal match events "
                "in soccer leagues. <i>Scientific Data</i>, 6(1), 236."),
        ("[9]", "Breiman, L. (2001). Random Forests. <i>Machine Learning</i>, 45, 5–32."),
        ("[10]","Rousseeuw, P.J. (1987). Silhouettes: A graphical aid to the interpretation and "
                "validation of cluster analysis. <i>Journal of Computational and Applied Mathematics</i>, "
                "20, 53–65."),
        ("[11]","Beal, R. et al. (2019). Artificial Intelligence for Team Sports: A Survey. "
                "<i>The Knowledge Engineering Review</i>, 34, e28."),
        ("[12]","Power, P. et al. (2017). Not All Passes Are Created Equal: Objectively Measuring "
                "the Risk and Reward of Passes in Soccer. <i>KDD 2017</i>, 1605–1613."),
        ("[13]","Scikit-learn developers (2024). <i>scikit-learn: Machine Learning in Python.</i> "
                "URL: https://scikit-learn.org/"),
        ("[14]","Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. "
                "<i>JMLR</i>, 12, 2825–2830."),
        ("[15]","StreamlitInc. (2024). <i>Streamlit — The fastest way to build data apps.</i> "
                "URL: https://streamlit.io/"),
    ]
    for ref, text in refs:
        story.append(Paragraph(f"<b>{ref}</b>  {text}",
                               make_style("Ref", fontSize=9, leftIndent=25, firstLineIndent=-25,
                                         spaceAfter=5, leading=13)))
    story.append(PageBreak())

    # ── ANHANG ────────────────────────────────────────────────────────────────
    story += section_header("Anhang: Glossar und Abbildungsverzeichnis")

    story += section_header("A.1 Glossar", level=2)
    glossar = [
        ("xG (Expected Goals)",    "Statistisches Maß für die Wahrscheinlichkeit, dass ein Schuss zu einem Tor führt (0–1)."),
        ("CRISP-DM",               "Cross-Industry Standard Process for Data Mining — iterativer 6-Phasen-Prozess für Data-Mining-Projekte."),
        ("StatsBomb Open Data",    "Öffentlich zugängliche Fußball-Ereignisdatenbank mit detaillierten xy-koordinierten Event-Daten."),
        ("K-Means",                "Unüberwachter Clustering-Algorithmus, der n Datenpunkte in K Cluster partitioniert."),
        ("Random Forest",          "Ensemble-Lernmethode aus Entscheidungsbäumen mit Mehrheitsvotum."),
        ("LOMO-CV",                "Leave-One-Match-Out Cross-Validation — CV-Strategie, bei der je ein Spiel als Testdaten dient."),
        ("Rolling Window",         "Gleitendes Zeitfenster für die Berechnung rollierender Statistiken (hier: 15 Minuten)."),
        ("Momentum Index",         "Komposit-Feature: 2·mar_xg_rate + mar_prog_pass_rate + 0,5·mar_pressure − 2·opp_xg_rate − 0,5·opp_pressure."),
        ("PCA",                    "Principal Component Analysis — Dimensionsreduktionsverfahren zur Visualisierung hochdimensionaler Daten."),
        ("DSS",                    "Decision Support System — System zur Unterstützung menschlicher Entscheidungsprozesse."),
        ("Prescriptive Analytics", "Analytikform, die konkrete Handlungsempfehlungen generiert (über deskriptiv und prädiktiv hinaus)."),
        ("AFCON",                  "Africa Cup of Nations — Afrikanischer Fußball-Nationenpokal, 2023 ausgetragen im Januar 2024 in der Elfenbeinküste."),
    ]
    glos_table = Table([(Paragraph(f"<b>{t}</b>", make_style("GT", fontSize=9, fontName="Helvetica-Bold")),
                         Paragraph(d, make_style("GD", fontSize=9, leading=13)))
                        for t, d in glossar],
                       colWidths=[5*cm, 11.5*cm])
    glos_table.setStyle(TableStyle([
        ("VALIGN",(0,0),(-1,-1),"TOP"),
        ("LINEBELOW",(0,0),(-1,-1),0.2,colors.HexColor("#e0e0e0")),
        ("TOPPADDING",(0,0),(-1,-1),5),
        ("BOTTOMPADDING",(0,0),(-1,-1),5),
        ("ROWBACKGROUNDS",(0,0),(-1,-1),[colors.white, LIGHT_BG]),
    ]))
    story.append(glos_table)
    story.append(Spacer(1, 0.5*cm))

    story += section_header("A.2 Abbildungsverzeichnis", level=2)
    abblist = [
        ("Abbildung 1",  "CRISP-DM Prozessmodell"),
        ("Abbildung 2",  "Spielübersicht Marokko AFCON 2023"),
        ("Abbildung 3",  "Schussverteilung (Shot Map)"),
        ("Abbildung 4",  "xG- und Schussverteilung nach Spielabschnitt"),
        ("Abbildung 5",  "Passquote nach Spielzone"),
        ("Abbildung 6",  "Pressing-Intensität nach Zone und Spielabschnitt (Heatmap)"),
        ("Abbildung 7",  "Feature-Engineering-Pipeline"),
        ("Abbildung 8",  "Temporale Dynamik der Rolling-Features"),
        ("Abbildung 9",  "PCA-Projektion der K-Means-Cluster"),
        ("Abbildung 10", "Cluster-Profil-Heatmap"),
        ("Abbildung 11", "Feature-Wichtigkeit des Random-Forest-Klassifikators"),
        ("Abbildung 12", "Interventionsergebnisse (vor/nach xG-Differenz)"),
        ("Abbildung 13", "Spieler-Performance-Radar"),
        ("Abbildung 14", "Systemarchitektur des präskriptiven DSS"),
        ("Abbildung 15", "Evaluations-Metriken Übersicht"),
    ]
    for abb, desc in abblist:
        story.append(Paragraph(f"<b>{abb}:</b>  {desc}",
                               make_style("AbbList", fontSize=9.5, leftIndent=14, spaceAfter=4)))

    # ── BUILD ─────────────────────────────────────────────────────────────────
    def _first_page(canvas, doc):
        title_page_template(canvas, doc)

    def _later_pages(canvas, doc):
        header_footer(canvas, doc)

    doc.build(story, onFirstPage=_first_page, onLaterPages=_later_pages)
    print(f"\n✓ Report saved: {OUT_PATH}")
    return OUT_PATH


if __name__ == "__main__":
    build()
