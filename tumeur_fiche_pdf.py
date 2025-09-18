# tumeur_fiche_pdf.py — version i18n complète
import os, re, unicodedata, datetime as dt, tempfile, shutil
from typing import Any, Dict, Iterable, Optional

import duckdb, pandas as pd

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# essayer pypdf puis PyPDF2 pour concat
try:
    from pypdf import PdfMerger
except Exception:
    try:
        from PyPDF2 import PdfMerger
    except Exception:
        PdfMerger = None

# Largeur utile (A4 avec marges 36/36)
_PAGE_W = A4[0]
_LEFT = 36
_RIGHT = 36
_AVAIL_W = _PAGE_W - _LEFT - _RIGHT

# --------------------------------------------------------------------
# Police Unicode (optionnelle — ignore si absente sur le système)
try:
    pdfmetrics.registerFont(TTFont("DejaVuSans", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
    _FONT = "DejaVuSans"
except Exception:
    _FONT = None

# --- Lang helpers (alignés avec indicators) ---
SUPPORTED_LANGS = {"fr","en","es","it","de"}

def _norm_lang(x: str | None) -> str:
    x = (x or "").strip().lower()
    if not x: return "fr"
    if "-" in x: x = x.split("-", 1)[0]
    x = x[:2]
    return x if x in SUPPORTED_LANGS else "fr"

def _current_lang(locale: str | None = None) -> str:
    # 1) priorité à l'argument explicite
    if locale: 
        return _norm_lang(locale)
    # 2) si on est en contexte Flask: lire g.lang (comme indicators)
    try:
        from flask import g  # import léger; ok hors Flask aussi
        gl = getattr(g, "lang", None)
        if gl:
            return _norm_lang(gl)
    except Exception:
        pass
    # 3) fallback sur APP_LANG (pour exécutions en sous-process ou CLI)
    env = os.environ.get("APP_LANG")
    if env:
        return _norm_lang(env)
    # 4) défaut
    return "fr"

# ===================== i18n helpers =====================
def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"['\"{}()\\[\\]:=]", " ", s)
    s = re.sub(r"\\s+", "", s)
    s = s.replace("-", "").replace("_", "")
    return s

# Canonique -> alias multilingues pour extractions directes dans des dicts
ALIASES: Dict[str, Iterable[str]] = {
    "date_doc": [
        "date_doc","document date","date du document","fecha del documento",
        "data del documento","datum des dokuments","documento fecha"
    ],
    "localization": [
        "localisation","location","localización","localizzazione","lokalisation","site"
    ],
    "histology_observations": [
        "observations","findings","observaciones","osservazioni","befunde"
    ],
    "histology_conclusion": [
        "conclusion","conclusión","conclusione","schlussfolgerung",
        "final diagnosis","diagnostic final"
    ],
    "tnm": [
        "tnm","tnm staging","stadification tnm","stadificación tnm",
        "stadiazione tnm","tnm-klassifikation"
    ],
    "oms": ["oms","ecog","ps","performance status"],
    "protocol": ["protocole","protocol","protocolo","protocollo","protokoll"],
    "treatment": [
        "chimiothérapie prescrite","chemotherapy prescribed","quimioterapia prescrita",
        "chemioterapia prescritta","verschriebenen chemotherapie","treatments","traitements"
    ],
    "other_meds": [
        "autres médicaments","other medications","otros medicamentos",
        "altri farmaci","weitere medikamente"
    ],
}

ALIAS_TO_CANON = {}
for k, aliases in ALIASES.items():
    for a in aliases:
        ALIAS_TO_CANON[_norm(a)] = k
for k in ALIASES:
    ALIAS_TO_CANON[_norm(k)] = k

def _take_val(v: Any) -> str:
    if v is None: return ""
    if isinstance(v, (int, float)): return str(v)
    if isinstance(v, str): return v
    if isinstance(v, dict):
        for key in ("value","text","date","start","end","val","content"):
            if key in v and v[key]:
                return _take_val(v[key])
        if "name" in v and v.get("value"):
            return _take_val(v["value"])
    if isinstance(v, list) and v:
        return _take_val(v[0])
    return str(v)

def get_field(fields: Any, canonical_key: str) -> str:
    target = _norm(canonical_key)
    if isinstance(fields, dict):
        for k, v in fields.items():
            nk = _norm(str(k))
            if nk == target or _norm(nk.replace("name","")) == target:
                return _take_val(v)
            if isinstance(v, dict):
                vn = _norm(str(v.get("name") or v.get("field") or ""))
                if vn == target or ALIAS_TO_CANON.get(vn) == canonical_key:
                    return _take_val(v)
            if ALIAS_TO_CANON.get(nk) == canonical_key:
                return _take_val(v)
    if isinstance(fields, list):
        for it in fields:
            if not isinstance(it, dict): 
                continue
            vn = _norm(str(it.get("name") or it.get("field") or it.get("key") or ""))
            if vn == target or ALIAS_TO_CANON.get(vn) == canonical_key:
                return _take_val(it)
    return ""

try:
    import dateparser  # pip install dateparser
except Exception:
    dateparser = None
from dateutil import parser as du_parser

def parse_any_date(s: str, lang_hint: Optional[str] = None) -> Optional[str]:
    s = (s or "").strip()
    if not s: return None
    dtv = None
    if dateparser is not None:
        kw = {}
        if lang_hint: kw["languages"] = [lang_hint]
        dtv = dateparser.parse(s, settings={"DATE_ORDER":"DMY"}, **kw)
    if dtv is None:
        try:
            dtv = du_parser.parse(s, dayfirst=True, fuzzy=True)
        except Exception:
            return None
    try:
        return dtv.date().isoformat()
    except Exception:
        return None

# Libellés UI
TR = {
    "fr": {
        "title":"Fiche tumeur",
        "date_doc":"Date du document",
        "localization":"Localisation",
        "histology":"Histologie",
        "observations":"Observations",
        "conclusion":"Conclusion",
        "tnm":"Stadification TNM",
        "oms":"OMS",
        "treatments":"Traitements",
        "other_meds":"Autres médicaments",

        "organ_major":"Organe majoritaire",
        "time_range":"Plage temporelle",
        "col_var":"Variable",
        "col_val":"Valeur",
        "section_general":"Caracs. Générales",
        "section_tumor":"Caracs. Tumeur",
        "section_treatment":"Traitement",

        "annex_title":"ANNEXE – Fiche tumeur",
        "source":"Source",
        "retained_value":"Valeur retenue",
        "rule_choice":"Règle de choix",
        "sample":"Échantillon",
        "sample_cat":"cat~",
        "all_rows":"toutes lignes",
        "generated_on":"Annexe générée le",
    },
    "en": {
        "title":"Tumor summary",
        "date_doc":"Document date",
        "localization":"Location",
        "histology":"Histology",
        "observations":"Findings",
        "conclusion":"Conclusion",
        "tnm":"TNM staging",
        "oms":"ECOG / Performance status",
        "treatments":"Treatments",
        "other_meds":"Other medications",

        "organ_major":"Main organ",
        "time_range":"Time range",
        "col_var":"Field",
        "col_val":"Value",
        "section_general":"General",
        "section_tumor":"Tumor",
        "section_treatment":"Treatment",

        "annex_title":"ANNEX – Tumor summary",
        "source":"Source",
        "retained_value":"Selected value",
        "rule_choice":"Selection rule",
        "sample":"Sample",
        "sample_cat":"cat~",
        "all_rows":"all rows",
        "generated_on":"Annex generated on",
    },
    "es": {
        "title":"Ficha tumoral",
        "date_doc":"Fecha del documento",
        "localization":"Localización",
        "histology":"Histología",
        "observations":"Observaciones",
        "conclusion":"Conclusión",
        "tnm":"Clasificación TNM",
        "oms":"ECOG / Estado funcional",
        "treatments":"Tratamientos",
        "other_meds":"Otros medicamentos",

        "organ_major":"Órgano principal",
        "time_range":"Intervalo temporal",
        "col_var":"Campo",
        "col_val":"Valor",
        "section_general":"General",
        "section_tumor":"Tumor",
        "section_treatment":"Tratamiento",

        "annex_title":"ANEXO – Ficha tumoral",
        "source":"Fuente",
        "retained_value":"Valor seleccionado",
        "rule_choice":"Regla de selección",
        "sample":"Muestra",
        "sample_cat":"cat~",
        "all_rows":"todas las filas",
        "generated_on":"Anexo generado el",
    },
    "it": {
        "title":"Scheda tumore",
        "date_doc":"Data del documento",
        "localization":"Localizzazione",
        "histology":"Istologia",
        "observations":"Osservazioni",
        "conclusion":"Conclusione",
        "tnm":"Stadiazione TNM",
        "oms":"ECOG / Stato funzionale",
        "treatments":"Trattamenti",
        "other_meds":"Altri farmaci",

        "organ_major":"Organo principale",
        "time_range":"Intervallo temporale",
        "col_var":"Campo",
        "col_val":"Valore",
        "section_general":"Generali",
        "section_tumor":"Tumore",
        "section_treatment":"Trattamento",

        "annex_title":"ALLEGATO – Scheda tumore",
        "source":"Fonte",
        "retained_value":"Valore selezionato",
        "rule_choice":"Regola di selezione",
        "sample":"Esempio",
        "sample_cat":"cat~",
        "all_rows":"tutte le righe",
        "generated_on":"Allegato generato il",
    },
    "de": {
        "title":"Tumor-Steckbrief",
        "date_doc":"Dokumentdatum",
        "localization":"Lokalisation",
        "histology":"Histologie",
        "observations":"Befunde",
        "conclusion":"Schlussfolgerung",
        "tnm":"TNM-Klassifikation",
        "oms":"ECOG / Leistungsstatus",
        "treatments":"Therapien",
        "other_meds":"Weitere Medikamente",

        "organ_major":"Hauptorgan",
        "time_range":"Zeitraum",
        "col_var":"Feld",
        "col_val":"Wert",
        "section_general":"Allgemein",
        "section_tumor":"Tumor",
        "section_treatment":"Behandlung",

        "annex_title":"ANLAGE – Tumor-Steckbrief",
        "source":"Quelle",
        "retained_value":"Ausgewählter Wert",
        "rule_choice":"Auswahlregel",
        "sample":"Stichprobe",
        "sample_cat":"cat~",
        "all_rows":"alle Zeilen",
        "generated_on":"Anlage erstellt am",
    },
}
def t(key: str, lang: str) -> str:
    return TR.get(lang, TR["fr"]).get(key, key)

# ===== Libellés d'affichage pour les 21 champs (par langue) =====
LBL = {
    "fr": {
        "Date de 1ère venue au Centre [lésion en cours]": "Date de 1ère venue au Centre [lésion en cours]",
        "Circonstances de découverte (clinique, dépistage, imagerie fortuite, etc.)": "Circonstances de découverte (clinique, dépistage, imagerie fortuite, etc.)",
        "Date de biopsie diagnostique/initiale": "Date de biopsie diagnostique/initiale",
        "Date de diagnostic": "Date de diagnostic",
        "Date de RCP initiale": "Date de RCP initiale",
        "Siège du primitif": "Siège du primitif",
        "Type histologique du primitif": "Type histologique du primitif",
        "Grade histopronostique": "Grade histopronostique",
        "Stade cTNM": "Stade cTNM",
        "Stade pTNM": "Stade pTNM",
        "Intervention chirurgicale (Oui/Non)": "Intervention chirurgicale (Oui/Non)",
        "Nature de l'intervention": "Nature de l'intervention",
        "Date de l'intervention": "Date de l'intervention",
        "Chimiothérapie (Oui/Non)": "Chimiothérapie (Oui/Non)",
        "Date de début de chimiothérapie": "Date de début de chimiothérapie",
        "Protocole chimiothérapie adjuvante": "Protocole chimiothérapie adjuvante",
        "Hormonothérapie (Oui/Non)": "Hormonothérapie (Oui/Non)",
        "Immunothérapie (Oui/Non)": "Immunothérapie (Oui/Non)",
        "Immunothérapie : Nature": "Immunothérapie : Nature",
        "Radiothérapie (Oui/Non)": "Radiothérapie (Oui/Non)",
        "Date de début de radiothérapie": "Date de début de radiothérapie",
    },
    "en": {
        "Date de 1ère venue au Centre [lésion en cours]": "First visit date to center [current lesion]",
        "Circonstances de découverte (clinique, dépistage, imagerie fortuite, etc.)": "Circumstances of discovery (clinical, screening, incidental imaging, etc.)",
        "Date de biopsie diagnostique/initiale": "Date of diagnostic/initial biopsy",
        "Date de diagnostic": "Date of diagnosis",
        "Date de RCP initiale": "Initial tumor board date",
        "Siège du primitif": "Primary site",
        "Type histologique du primitif": "Primary histologic type",
        "Grade histopronostique": "Histoprognostic grade",
        "Stade cTNM": "cTNM stage",
        "Stade pTNM": "pTNM stage",
        "Intervention chirurgicale (Oui/Non)": "Surgery (Yes/No)",
        "Nature de l'intervention": "Nature of the procedure",
        "Date de l'intervention": "Date of surgery",
        "Chimiothérapie (Oui/Non)": "Chemotherapy (Yes/No)",
        "Date de début de chimiothérapie": "Chemotherapy start date",
        "Protocole chimiothérapie adjuvante": "Adjuvant chemotherapy protocol",
        "Hormonothérapie (Oui/Non)": "Hormonal therapy (Yes/No)",
        "Immunothérapie (Oui/Non)": "Immunotherapy (Yes/No)",
        "Immunothérapie : Nature": "Immunotherapy: agent/protocol",
        "Radiothérapie (Oui/Non)": "Radiotherapy (Yes/No)",
        "Date de début de radiothérapie": "Radiotherapy start date",
    },
    "es": {
        "Date de 1ère venue au Centre [lésion en cours]": "Fecha de la primera visita al centro [lesión actual]",
        "Circonstances de découverte (clinique, dépistage, imagerie fortuite, etc.)": "Circunstancias del descubrimiento (clínico, cribado, hallazgo incidental, etc.)",
        "Date de biopsie diagnostique/initiale": "Fecha de biopsia diagnóstica/inicial",
        "Date de diagnostic": "Fecha de diagnóstico",
        "Date de RCP initiale": "Fecha del comité de tumores inicial",
        "Siège du primitif": "Asiento primario",
        "Type histologique du primitif": "Tipo histológico primario",
        "Grade histopronostique": "Grado histopronóstico",
        "Stade cTNM": "Estadio cTNM",
        "Stade pTNM": "Estadio pTNM",
        "Intervention chirurgicale (Oui/Non)": "Cirugía (Sí/No)",
        "Nature de l'intervention": "Naturaleza de la intervención",
        "Date de l'intervention": "Fecha de la cirugía",
        "Chimiothérapie (Oui/Non)": "Quimioterapia (Sí/No)",
        "Date de début de chimiothérapie": "Fecha de inicio de quimioterapia",
        "Protocole chimiothérapie adjuvante": "Protocolo de quimioterapia adyuvante",
        "Hormonothérapie (Oui/Non)": "Hormonoterapia (Sí/No)",
        "Immunothérapie (Oui/Non)": "Inmunoterapia (Sí/No)",
        "Immunothérapie : Nature": "Inmunoterapia: fármaco/protocolo",
        "Radiothérapie (Oui/Non)": "Radioterapia (Sí/No)",
        "Date de début de radiothérapie": "Fecha de inicio de radioterapia",
    },
    "it": {
        "Date de 1ère venue au Centre [lésion en cours]": "Data della prima visita al centro [lesione in corso]",
        "Circonstances de découverte (clinique, dépistage, imagerie fortuite, etc.)": "Circostanze della scoperta (clinica, screening, reperto incidentale, ecc.)",
        "Date de biopsie diagnostique/initiale": "Data della biopsia diagnostica/inziale",
        "Date de diagnostic": "Data della diagnosi",
        "Date de RCP initiale": "Data del MDT iniziale",
        "Siège du primitif": "Sede primaria",
        "Type histologique du primitif": "Tipo istologico primario",
        "Grade histopronostique": "Grado istoprognostico",
        "Stade cTNM": "Stadio cTNM",
        "Stade pTNM": "Stadio pTNM",
        "Intervention chirurgicale (Oui/Non)": "Chirurgia (Sì/No)",
        "Nature de l'intervention": "Natura dell'intervento",
        "Date de l'intervention": "Data dell'intervento",
        "Chimiothérapie (Oui/Non)": "Chemioterapia (Sì/No)",
        "Date de début de chimiothérapie": "Data di inizio chemioterapia",
        "Protocole chimiothérapie adjuvante": "Protocollo chemioterapico adiuvante",
        "Hormonothérapie (Oui/Non)": "Ormonoterapia (Sì/No)",
        "Immunothérapie (Oui/Non)": "Immunoterapia (Sì/No)",
        "Immunothérapie : Nature": "Immunoterapia: farmaco/protocollo",
        "Radiothérapie (Oui/Non)": "Radioterapia (Sì/No)",
        "Date de début de radiothérapie": "Data di inizio radioterapia",
    },
    "de": {
        "Date de 1ère venue au Centre [lésion en cours]": "Erstes Besuchsdatum im Zentrum [aktuelle Läsion]",
        "Circonstances de découverte (clinique, dépistage, imagerie fortuite, etc.)": "Entdeckungsumstände (klinisch, Screening, Zufallsbefund, etc.)",
        "Date de biopsie diagnostique/initiale": "Datum der diagnostischen/initialen Biopsie",
        "Date de diagnostic": "Diagnosedatum",
        "Date de RCP initiale": "Datum der ersten Tumorkonferenz",
        "Siège du primitif": "Primärer Sitz",
        "Type histologique du primitif": "Primärer histologischer Typ",
        "Grade histopronostique": "Histopronostischer Grad",
        "Stade cTNM": "cTNM-Stadium",
        "Stade pTNM": "pTNM-Stadium",
        "Intervention chirurgicale (Oui/Non)": "Operation (Ja/Nein)",
        "Nature de l'intervention": "Art des Eingriffs",
        "Date de l'intervention": "Operationsdatum",
        "Chimiothérapie (Oui/Non)": "Chemotherapie (Ja/Nein)",
        "Date de début de chimiothérapie": "Beginn der Chemotherapie",
        "Protocole chimiothérapie adjuvante": "Adjuvantes Chemotherapieprotokoll",
        "Hormonothérapie (Oui/Non)": "Hormontherapie (Ja/Nein)",
        "Immunothérapie (Oui/Non)": "Immuntherapie (Ja/Nein)",
        "Immunothérapie : Nature": "Immuntherapie: Wirkstoff/Protokoll",
        "Radiothérapie (Oui/Non)": "Strahlentherapie (Ja/Nein)",
        "Date de début de radiothérapie": "Beginn der Strahlentherapie",
    },
}

YESNO_KEYS = {
    "Intervention chirurgicale (Oui/Non)",
    "Chimiothérapie (Oui/Non)",
    "Hormonothérapie (Oui/Non)",
    "Immunothérapie (Oui/Non)",
    "Radiothérapie (Oui/Non)",
}


# Traduction des en-têtes de DataFrames (annexe)
DF_COL_TR = {
    "fr": {
        "category":"Catégorie",
        "date_doc":"Date du document",
        "date":"Date",
        "date_of_discovery":"Date de découverte",
        "circumstances_of_discovery":"Circonstances de découverte",
        "conclusion":"Conclusion",
        "histologic_type":"Type histologique",
        "procedure_title":"Intitulé de l’intervention",
        "medications":"Médicaments",
        "chemotherapy_protocol":"Protocole de chimiothérapie",
        "other_medications":"Autres médicaments",
        "department":"Service",
    },
    "en": {
        "category":"Category",
        "date_doc":"Document date",
        "date":"Date",
        "date_of_discovery":"Date of discovery",
        "circumstances_of_discovery":"Circumstances of discovery",
        "conclusion":"Conclusion",
        "histologic_type":"Histologic type",
        "procedure_title":"Procedure title",
        "medications":"Medications",
        "chemotherapy_protocol":"Chemotherapy protocol",
        "other_medications":"Other medications",
        "department":"Department",
    },
    "es": {
        "category":"Categoría",
        "date_doc":"Fecha del documento",
        "date":"Fecha",
        "date_of_discovery":"Fecha de descubrimiento",
        "circumstances_of_discovery":"Circunstancias del descubrimiento",
        "conclusion":"Conclusión",
        "histologic_type":"Tipo histológico",
        "procedure_title":"Título del procedimiento",
        "medications":"Medicamentos",
        "chemotherapy_protocol":"Protocolo de quimioterapia",
        "other_medications":"Otros medicamentos",
        "department":"Servicio",
    },
    "it": {
        "category":"Categoria",
        "date_doc":"Data del documento",
        "date":"Data",
        "date_of_discovery":"Data della scoperta",
        "circumstances_of_discovery":"Circostanze della scoperta",
        "conclusion":"Conclusione",
        "histologic_type":"Tipo istologico",
        "procedure_title":"Titolo della procedura",
        "medications":"Farmaci",
        "chemotherapy_protocol":"Protocollo chemioterapico",
        "other_medications":"Altri farmaci",
        "department":"Reparto",
    },
    "de": {
        "category":"Kategorie",
        "date_doc":"Dokumentdatum",
        "date":"Datum",
        "date_of_discovery":"Entdeckungsdatum",
        "circumstances_of_discovery":"Umstände der Entdeckung",
        "conclusion":"Schlussfolgerung",
        "histologic_type":"Histologischer Typ",
        "procedure_title":"Eingriffsbezeichnung",
        "medications":"Medikamente",
        "chemotherapy_protocol":"Chemotherapieprotokoll",
        "other_medications":"Weitere Medikamente",
        "department":"Abteilung",
    },
}
def _norm_key_for_header(c):
    c = str(c).strip()
    c = c.lower().replace("’","'")
    c = re.sub(r"[^a-z0-9]+", "_", c).strip("_")
    return c

def _translate_headers_for_df(df, lang):
    if df is None or df.empty: return df
    tr = DF_COL_TR.get(lang, DF_COL_TR["fr"])
    new_cols = []
    for c in df.columns:
        key = _norm_key_for_header(c)
        new_cols.append(tr.get(key, str(c)))
    df2 = df.copy()
    df2.columns = new_cols
    return df2

# ===================== Table utils =====================
def _compute_col_widths(df, avail=_AVAIL_W, min_w=60, max_w=260, sample_rows=5):
    if df is None or df.empty:
        return []
    lens = []
    for c in df.columns:
        L = len(str(c))
        for v in df[c].head(sample_rows):
            s = "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)
            L = max(L, len(s))
        lens.append(max(1, L))
    total = sum(lens)
    raw = [max(min_w, avail * (L / total)) for L in lens]
    raw = [min(max_w, w) for w in raw]
    s = sum(raw)
    if s > avail:
        k = avail / s
        raw = [w * k for w in raw]
    return raw

def _wrap_table_data_from_df(df, style_cell, style_head, lang="fr"):
    from reportlab.platypus import Paragraph
    df = _translate_headers_for_df(df, lang)
    head = [Paragraph(str(c), style_head) for c in df.columns]
    rows = []
    for _, r in df.iterrows():
        row = []
        for c in df.columns:
            v = r.get(c, "")
            s = "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)
            row.append(Paragraph(s, style_cell))
        rows.append(row)
    return [head] + rows

def _preview_table(flow, title, df, max_rows=6, max_chars=180, lang="fr"):
    if df is None or df.empty:
        return
    from reportlab.platypus import Paragraph, Spacer, Table, TableStyle
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

    styles = getSampleStyleSheet()
    Cell = ParagraphStyle("AnnexCell",
                          parent=styles["BodyText"],
                          fontSize=9, leading=12, wordWrap="CJK")
    CellHead = ParagraphStyle("AnnexHead",
                              parent=styles["BodyText"],
                              fontSize=9, leading=12, wordWrap="CJK")
    if _FONT:
        Cell.fontName = _FONT
        CellHead.fontName = _FONT

    flow.append(Paragraph(f"<b>{title}</b>", styles["BodyText"]))

    df2 = df.copy()
    for c in df2.columns:
        df2[c] = df2[c].map(lambda x: (str(x)[:max_chars] + "…")
                            if isinstance(x, str) and len(x) > max_chars else x)

    df2 = _translate_headers_for_df(df2, lang)
    col_widths = _compute_col_widths(df2, avail=_AVAIL_W, min_w=60, max_w=260)
    data = _wrap_table_data_from_df(df2.head(max_rows), Cell, CellHead, lang=lang)

    tab = Table(data, colWidths=col_widths)
    tab.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.25, colors.HexColor("#BBBBBB")),
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#EEEEEE")),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("ALIGN", (0,0), (-1,-1), "LEFT"),
    ]))
    flow.append(tab)
    flow.append(Spacer(1,6))

def concat_pdfs(paths, out_path):
    if not PdfMerger:
        print("[ANNEXE] Concat PDF: bibliothèque pypdf/PyPDF2 introuvable — annexe conservée séparément.")
        return out_path
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.close()
    merger = PdfMerger()
    for p in paths:
        if p and os.path.exists(p):
            merger.append(p)
    merger.write(tmp.name)
    merger.close()
    shutil.copy2(tmp.name, out_path)
    os.unlink(tmp.name)
    return out_path

# ===================== Annexe : règles (texte) =====================
# ===== Règles i18n pour les 21 champs =====
RULES_I18N = {
    "fr": {
        "Date de 1ère venue au Centre [lésion en cours]":
            "Prend la plus ancienne date des lignes « Consultation » si présentes, sinon la première date globale.",
        "Circonstances de découverte (clinique, dépistage, imagerie fortuite, etc.)":
            "Cherche une colonne dédiée (ex. « circonstances_de_la_decouverte ») dans « RCP », sinon « Consultation » (ex. « motif »).",
        "Date de biopsie diagnostique/initiale":
            "Prend colonne « date_biopsie* » si disponible, sinon la première date en « Histologie ».",
        "Date de diagnostic":
            "Prend colonne « date_diagnostic* » si disponible, sinon le min des dates entre « RCP » et « Histologie ».",
        "Date de RCP initiale":
            "Première date en « RCP ».",
        "Siège du primitif":
            "Valeur majoritaire parmi (siege/site/localisation/organe*), sinon organe majoritaire.",
        "Type histologique du primitif":
            "Colonnes préférées « type_histologique » / « conclusion » dans « Histologie ».",
        "Grade histopronostique":
            "Colonnes « grade* » / « sbr » / « elston » si présentes, sinon extraction regex « grade ... » dans le texte d’histologie.",
        "Stade cTNM":
            "Colonne cTNM si présente, sinon parsing regex cT/cN/cM dans le texte (RCP/Histologie/Imagerie/Consultation).",
        "Stade pTNM":
            "Colonne pTNM si présente, sinon parsing regex pT/pN/pM dans le texte (RCP/Histologie/Imagerie/Consultation).",
        "Intervention chirurgicale (Oui/Non)":
            "Détection par catégorie (« opératoire » / « hospitalisation ») ou mots-clés chirurgicaux dans le texte.",
        "Nature de l'intervention":
            "Colonnes « nature » / « intervention » / « acte » si présentes ; sinon extrait du texte pertinent.",
        "Date de l'intervention":
            "Plus ancienne date trouvée parmi les lignes identifiées comme chirurgicales.",
        "Chimiothérapie (Oui/Non)":
            "Détection via « Ordonnance » + mots-clés chimiothérapies.",
        "Date de début de chimiothérapie":
            "Colonne « date_debut_chimio » si disponible, sinon première date sur les lignes identifiées.",
        "Protocole chimiothérapie adjuvante":
            "Colonnes « protocole* » dans « Ordonnance » si présentes ; sinon extrait du texte.",
        "Hormonothérapie (Oui/Non)":
            "Détection via « Ordonnance » + mots-clés hormonothérapies.",
        "Immunothérapie (Oui/Non)":
            "Détection via « Ordonnance » + mots-clés immunothérapies.",
        "Immunothérapie : Nature":
            "Colonnes médicament/molécule/protocole si présentes ; sinon extrait du texte.",
        "Radiothérapie (Oui/Non)":
            "Détection via catégorie « Séance de radiothérapie » ou mots-clés radiothérapie.",
        "Date de début de radiothérapie":
            "Première date en « Séance de radiothérapie » si disponible, sinon la plus ancienne date trouvée dans les indices radio.",
    },
    "en": {
        "Date de 1ère venue au Centre [lésion en cours]":
            "Take the earliest date among “Consultation” rows if present; otherwise the earliest overall date.",
        "Circonstances de découverte (clinique, dépistage, imagerie fortuite, etc.)":
            "Look for a dedicated column (e.g., “circonstances_de_la_decouverte”) in “RCP”; otherwise use “Consultation” (e.g., “motif”).",
        "Date de biopsie diagnostique/initiale":
            "Use column “date_biopsie*” if available; otherwise the first date found in “Histology”.",
        "Date de diagnostic":
            "Use column “date_diagnostic*” if available; otherwise the minimum of dates between “RCP” and “Histology”.",
        "Date de RCP initiale":
            "Earliest date in “RCP”.",
        "Siège du primitif":
            "Most frequent value among (siege/site/localisation/organe*); otherwise the majority organ.",
        "Type histologique du primitif":
            "Prefer columns “type_histologique” / “conclusion” within “Histology”.",
        "Grade histopronostique":
            "Use “grade*” / “sbr” / “elston” columns when present; otherwise regex extraction “grade ...” from histology text.",
        "Stade cTNM":
            "Use cTNM column if present; otherwise parse cT/cN/cM from text (RCP/Histology/Imaging/Consultation).",
        "Stade pTNM":
            "Use pTNM column if present; otherwise parse pT/pN/pM from text (RCP/Histology/Imaging/Consultation).",
        "Intervention chirurgicale (Oui/Non)":
            "Detect via category (“operative” / “hospitalization”) or surgical keywords in the text.",
        "Nature de l'intervention":
            "Columns “nature” / “intervention” / “acte” when present; otherwise extract from relevant text.",
        "Date de l'intervention":
            "Earliest date among rows identified as surgical.",
        "Chimiothérapie (Oui/Non)":
            "Detect via “Ordonnance” plus chemotherapy keywords.",
        "Date de début de chimiothérapie":
            "Column “date_debut_chimio” if available; otherwise first date on chemotherapy-flagged rows.",
        "Protocole chimiothérapie adjuvante":
            "Columns “protocole*” within “Ordonnance” when present; otherwise extract from text.",
        "Hormonothérapie (Oui/Non)":
            "Detect via “Ordonnance” plus hormone-therapy keywords.",
        "Immunothérapie (Oui/Non)":
            "Detect via “Ordonnance” plus immunotherapy keywords.",
        "Immunothérapie : Nature":
            "Medication/molecule/protocol columns when present; otherwise extract from text.",
        "Radiothérapie (Oui/Non)":
            "Detect via “Radiotherapy session” category or radiotherapy keywords.",
        "Date de début de radiothérapie":
            "First date in “Radiotherapy session” if available; otherwise the earliest date among radiotherapy indicators.",
    },
    "es": {
        "Date de 1ère venue au Centre [lésion en cours]":
            "Toma la fecha más antigua de las filas de « Consulta » si existen; en caso contrario, la primera fecha global.",
        "Circonstances de découverte (clinique, dépistage, imagerie fortuite, etc.)":
            "Busca una columna específica (p. ej., « circonstances_de_la_decouverte ») en « RCP »; si no, en « Consulta » (p. ej., « motif »).",
        "Date de biopsie diagnostique/initiale":
            "Usa la columna « date_biopsie* » si está disponible; en su defecto, la primera fecha en « Histología ».",
        "Date de diagnostic":
            "Usa la columna « date_diagnostic* » si está disponible; en su defecto, el mínimo entre las fechas de « RCP » y « Histología ».",
        "Date de RCP initiale":
            "Primera fecha en « RCP ».",
        "Siège du primitif":
            "Valor mayoritario entre (siege/site/localisation/organe*); en su defecto, órgano mayoritario.",
        "Type histologique du primitif":
            "Columnas preferidas « type_histologique » / « conclusion » en « Histología ».",
        "Grade histopronostique":
            "Columnas « grade* » / « sbr » / « elston » si existen; si no, extracción por regex « grade ... » del texto de histología.",
        "Stade cTNM":
            "Columna cTNM si existe; de lo contrario, parsing de cT/cN/cM desde el texto (RCP/Histología/Imagen/Consulta).",
        "Stade pTNM":
            "Columna pTNM si existe; de lo contrario, parsing de pT/pN/pM desde el texto (RCP/Histología/Imagen/Consulta).",
        "Intervention chirurgicale (Oui/Non)":
            "Detección por categoría (« operatorio » / « hospitalización ») o por palabras clave quirúrgicas en el texto.",
        "Nature de l'intervention":
            "Columnas « nature » / « intervention » / « acte » si existen; si no, extraído del texto pertinente.",
        "Date de l'intervention":
            "Fecha más antigua entre las filas identificadas como quirúrgicas.",
        "Chimiothérapie (Oui/Non)":
            "Detección mediante « Ordonnance » + palabras clave de quimioterapia.",
        "Date de début de chimiothérapie":
            "Columna « date_debut_chimio » si existe; en su defecto, primera fecha en las filas marcadas.",
        "Protocole chimiothérapie adjuvante":
            "Columnas « protocole* » en « Ordonnance » si existen; si no, extraído del texto.",
        "Hormonothérapie (Oui/Non)":
            "Detección mediante « Ordonnance » + palabras clave de hormonoterapia.",
        "Immunothérapie (Oui/Non)":
            "Detección mediante « Ordonnance » + palabras clave de inmunoterapia.",
        "Immunothérapie : Nature":
            "Columnas de fármaco/molécula/protocolo si existen; si no, extraído del texto.",
        "Radiothérapie (Oui/Non)":
            "Detección mediante la categoría « Sesión de radioterapia » o palabras clave de radioterapia.",
        "Date de début de radiothérapie":
            "Primera fecha en « Sesión de radioterapia » si existe; en su defecto, la fecha más antigua entre los indicios de radioterapia.",
    },
    "it": {
        "Date de 1ère venue au Centre [lésion en cours]":
            "Prendi la data più antica tra le righe di « Visita/Consultazione » se presenti; altrimenti la prima data globale.",
        "Circonstances de découverte (clinique, dépistage, imagerie fortuite, etc.)":
            "Cerca una colonna dedicata (es. « circonstances_de_la_decouverte ») in « RCP/Tumor board »; altrimenti in « Visita » (es. « motif »).",
        "Date de biopsie diagnostique/initiale":
            "Usa la colonna « date_biopsie* » se disponibile; altrimenti la prima data in « Istologia ».",
        "Date de diagnostic":
            "Usa la colonna « date_diagnostic* » se disponibile; altrimenti il minimo tra le date di « RCP » e « Istologia ».",
        "Date de RCP initiale":
            "Prima data in « RCP ».",
        "Siège du primitif":
            "Valore più frequente tra (siege/site/localisation/organe*); in alternativa, organo maggioritario.",
        "Type histologique du primitif":
            "Colonne preferite « type_histologique » / « conclusion » in « Istologia ».",
        "Grade histopronostique":
            "Colonne « grade* » / « sbr » / « elston » se presenti; altrimenti estrazione regex « grade ... » dal testo istologico.",
        "Stade cTNM":
            "Colonna cTNM se presente; altrimenti parsing cT/cN/cM dal testo (RCP/Istologia/Imaging/Visita).",
        "Stade pTNM":
            "Colonna pTNM se presente; altrimenti parsing pT/pN/pM dal testo (RCP/Istologia/Imaging/Visita).",
        "Intervention chirurgicale (Oui/Non)":
            "Rilevazione per categoria (« operatorio » / « ricovero ») o tramite parole chiave chirurgiche nel testo.",
        "Nature de l'intervention":
            "Colonne « nature » / « intervention » / « acte » se presenti; altrimenti estratte dal testo pertinente.",
        "Date de l'intervention":
            "Data più antica tra le righe identificate come chirurgiche.",
        "Chimiothérapie (Oui/Non)":
            "Rilevazione tramite « Ordonnance/Prescrizione » + parole chiave di chemioterapia.",
        "Date de début de chimiothérapie":
            "Colonna « date_debut_chimio » se presente; altrimenti prima data sulle righe identificate.",
        "Protocole chimiothérapie adjuvante":
            "Colonne « protocole* » in « Prescrizione » se presenti; altrimenti estratte dal testo.",
        "Hormonothérapie (Oui/Non)":
            "Rilevazione tramite « Prescrizione » + parole chiave di ormonoterapia.",
        "Immunothérapie (Oui/Non)":
            "Rilevazione tramite « Prescrizione » + parole chiave di immunoterapia.",
        "Immunothérapie : Nature":
            "Colonne farmaco/molecola/protocollo se presenti; altrimenti estratte dal testo.",
        "Radiothérapie (Oui/Non)":
            "Rilevazione tramite categoria « Seduta di radioterapia » o parole chiave di radioterapia.",
        "Date de début de radiothérapie":
            "Prima data in « Seduta di radioterapia » se presente; altrimenti la più antica tra gli indizi di radioterapia.",
    },
    "de": {
        "Date de 1ère venue au Centre [lésion en cours]":
            "Nimm das früheste Datum aus „Konsultation“-Zeilen, falls vorhanden; sonst das früheste Gesamtdatum.",
        "Circonstances de découverte (clinique, dépistage, imagerie fortuite, etc.)":
            "Suche eine eigene Spalte (z. B. „circonstances_de_la_decouverte“) in „RCP/Tumorboard“; sonst in „Konsultation“ (z. B. „motif“).",
        "Date de biopsie diagnostique/initiale":
            "Verwende die Spalte „date_biopsie*“, falls vorhanden; sonst das erste Datum in „Histologie“.",
        "Date de diagnostic":
            "Verwende die Spalte „date_diagnostic*“, falls vorhanden; sonst das Minimum der Daten aus „RCP“ und „Histologie“.",
        "Date de RCP initiale":
            "Frühestes Datum in „RCP“.",
        "Siège du primitif":
            "Häufigster Wert unter (siege/site/localisation/organe*); andernfalls Mehrheitsorgan.",
        "Type histologique du primitif":
            "Bevorzugte Spalten „type_histologique“ / „conclusion“ in „Histologie“.",
        "Grade histopronostique":
            "Spalten „grade*“ / „sbr“ / „elston“, falls vorhanden; sonst Regex-Extraktion „grade ...“ aus Histologietext.",
        "Stade cTNM":
            "cTNM-Spalte falls vorhanden; sonst cT/cN/cM aus Text parsen (RCP/Histologie/Bildgebung/Konsultation).",
        "Stade pTNM":
            "pTNM-Spalte falls vorhanden; sonst pT/pN/pM aus Text parsen (RCP/Histologie/Bildgebung/Konsultation).",
        "Intervention chirurgicale (Oui/Non)":
            "Erkennung über Kategorie („operativ“ / „Krankenhausaufenthalt“) oder chirurgische Stichwörter im Text.",
        "Nature de l'intervention":
            "Spalten „nature“ / „intervention“ / „acte“ falls vorhanden; sonst aus relevantem Text extrahieren.",
        "Date de l'intervention":
            "Frühestes Datum unter den als chirurgisch identifizierten Zeilen.",
        "Chimiothérapie (Oui/Non)":
            "Erkennung über „Ordonnance/Verordnung“ + Chemotherapie-Stichwörter.",
        "Date de début de chimiothérapie":
            "Spalte „date_debut_chimio“ falls vorhanden; sonst erstes Datum in den markierten Zeilen.",
        "Protocole chimiothérapie adjuvante":
            "Spalten „protocole*“ in „Verordnung“ falls vorhanden; sonst aus Text extrahieren.",
        "Hormonothérapie (Oui/Non)":
            "Erkennung über „Verordnung“ + Hormontherapie-Stichwörter.",
        "Immunothérapie (Oui/Non)":
            "Erkennung über „Verordnung“ + Immuntherapie-Stichwörter.",
        "Immunothérapie : Nature":
            "Spalten Medikament/Molekül/Protokoll falls vorhanden; sonst aus Text extrahieren.",
        "Radiothérapie (Oui/Non)":
            "Erkennung über Kategorie „Strahlentherapie-Sitzung“ oder Radiotherapie-Stichwörter.",
        "Date de début de radiothérapie":
            "Erstes Datum in „Strahlentherapie-Sitzung“, falls vorhanden; andernfalls das früheste Datum unter den Radiotherapie-Indizien.",
    },
}

def _rule_text(label: str, lang: str | None) -> str:
    lg = _current_lang(lang)
    return RULES_I18N.get(lg, {}).get(label) or RULES_I18N["fr"].get(label, "Règle non documentée.")

# ===================== Chargement Excel -> DuckDB =====================
def _slug(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii","ignore").decode("ascii")
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_")
    return s.lower() or "sheet"

def _norm_col(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii","ignore").decode("ascii")
    s = s.lower().strip()
    s = re.sub(r"\\s+", "_", s)
    s = s.replace("’","'")
    return s

def _as_date_str(x):
    if pd.isna(x) or x is None: return ""
    try:
        if isinstance(x, (dt.date, dt.datetime, pd.Timestamp)):
            return pd.to_datetime(x).strftime("%Y-%m-%d")
        return pd.to_datetime(str(x), errors="coerce").strftime("%Y-%m-%d")
    except Exception:
        return str(x)

def _load_excel_to_duckdb(xlsx_path: str):
    xlsx_path = str(xlsx_path)
    sheets = pd.read_excel(xlsx_path, sheet_name=None, dtype=object, engine="openpyxl")
    con = duckdb.connect(database=":memory:")
    table_names = []
    for sheet_name, df in sheets.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        df = df.copy()
        df.columns = [_norm_col(c) for c in df.columns]
        for dc in ["date_doc","date","date_debut","date_debut_chimio","date_chimio","date_radiotherapie","date_rt",
                   "date_intervention","date_biopsie","date_diagnostic","date_rcp"]:
            if dc in df.columns:
                df[dc] = df[dc].map(_as_date_str)
        if "categorie" in df.columns and "category" not in df.columns:
            df["category"] = df["categorie"]
        if "cluster_name" in df.columns and "category" not in df.columns:
            df["category"] = df["cluster_name"]
        tname = f"sheet_{_slug(sheet_name)}"
        con.register(tname, df)
        table_names.append(tname)
    if not table_names:
        raise RuntimeError("Excel vide ou illisible.")
    union_sql = " UNION ALL ".join([f"SELECT * FROM {t}" for t in table_names])
    con.execute(f"CREATE OR REPLACE VIEW t AS {union_sql}")
    return con, table_names

# ===================== Query helpers =====================
def _list_cols(con, table="t"):
    try:
        return [row[0] for row in con.execute(f"DESCRIBE {table}").fetchall()]
    except Exception:
        return [row[0] for row in con.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE lower(table_name)=lower(?) ORDER BY ordinal_position", [table]
        ).fetchall()]

def _df_where(con, where_sql: str, params=()):
    try:
        return con.execute(f"SELECT * FROM t WHERE {where_sql}", params).fetch_df()
    except Exception:
        return pd.DataFrame()

# Synonymes de catégories (multilingues)
CAT = {
  "consult":  ["consult","consultation","consulta","visita","sprechstunde"],
  "rcp":      ["rcp","tumor board","tumorboard","multidisciplinary","mdt","mtb","comité de tumores"],
  "histolog": ["histolog","histology","patholog","biopsy","biopsia","istolog"],
  "imager":   ["imager","imaging","radiology","imagen","radiologia"],
  "operat":   ["operat","chirurg","surgery","operación","operazione","operation","operative","operatório"],
  "hospital": ["hospital","hospitalisation","ingreso","ricovero","krankenhaus"],
  "ordon":    ["ordon","prescription","ordonnance","medication","medicacion","receta","verordnung"],
  "radiother":["radiother","radioth","radiation","rt","radioterapia","radiothérapie","strahlentherapie"],
}

def _df_cat_any(con, likes):
    where = " OR ".join(["LOWER(CAST(category AS VARCHAR)) LIKE ?"] * len(likes))
    params = [f"%{l}%" for l in likes]
    return _df_where(con, where, params)

def _df_cat_like(con, key_or_like):
    likes = CAT.get(key_or_like, [key_or_like])
    return _df_cat_any(con, likes)

def _min_date_in_df(df):
    if df is None or df.empty: return ""
    col = "date_doc" if "date_doc" in df.columns else None
    if not col:
        for c in df.columns:
            if "date" in c:
                col = c; break
    if not col: return ""
    ds = pd.to_datetime(df[col], errors="coerce")
    if ds.isna().all(): return ""
    return ds.min().strftime("%Y-%m-%d")

def _max_date_in_df(df):
    if df is None or df.empty: return ""
    col = "date_doc" if "date_doc" in df.columns else None
    if not col:
        for c in df.columns:
            if "date" in c:
                col = c; break
    if not col: return ""
    ds = pd.to_datetime(df[col], errors="coerce")
    if ds.isna().all(): return ""
    return ds.max().strftime("%Y-%m-%d")

def _pick_first_nonempty(series):
    for v in series:
        if isinstance(v, (list, tuple)) and len(v):
            v = v[0]
        if isinstance(v, dict):
            v = v.get("value") or v.get("text") or v.get("date") or v.get("val") or ""
        if v is None: 
            continue
        vs = str(v).strip()
        if vs: return vs
    return ""

def _first_text_from_df(df, prefer=()):
    if df is None or df.empty: return ""
    for p in prefer:
        if p in df.columns:
            v = _pick_first_nonempty(df[p].tolist())
            if v: return "" if v is None else str(v)
    for c in df.columns:
        if any(x in c for x in ["date","category"]): continue
        v = _pick_first_nonempty(df[c].tolist())
        if isinstance(v, str) and len(v.strip())>0:
            return v
    return ""

def _first_value_in_cols(con, like: str, prefer_cols=()):
    df = _df_cat_like(con, like)
    return _first_text_from_df(df, prefer_cols)

def _first_date_in_cat(con, like: str):
    df = _df_cat_like(con, like)
    return _min_date_in_df(df)

def _any_keyword_in_df(df, keywords):
    if df is None or df.empty: return (False, "", "")
    kws = [k.lower() for k in keywords]
    date_col = "date_doc" if "date_doc" in df.columns else None
    if not date_col:
        for c in df.columns:
            if "date" in c: date_col = c; break
    earliest = None
    snippet = ""
    for _, row in df.iterrows():
        for c, val in row.items():
            if isinstance(val, str):
                s = val.strip(); low = s.lower()
                if any(k in low for k in kws):
                    d = pd.to_datetime(row.get(date_col, None), errors="coerce") if date_col else pd.NaT
                    if earliest is None or (pd.notna(d) and d < earliest):
                        earliest = d
                    idx = next((low.find(k) for k in kws if k in low), -1)
                    if idx >= 0:
                        start = max(0, idx-30); end = min(len(s), idx+60)
                        snippet = s[start:end]
    return (snippet != "" or earliest is not None, (earliest.strftime("%Y-%m-%d") if pd.notna(earliest) else ""), snippet)

def _guess_organe(con):
    cols = _list_cols(con, "t")
    col_org = None
    for cand in ("organe","organe_majoritaire","localisation","localization","location","siege","site","org","primary_site"):
        if cand in cols: col_org = cand; break
    if col_org:
        try:
            row = con.execute(f"""
                SELECT {col_org} AS val, COUNT(*) AS n
                FROM t
                WHERE {col_org} IS NOT NULL AND TRIM(CAST({col_org} AS VARCHAR)) <> ''
                GROUP BY 1 ORDER BY n DESC LIMIT 1
            """).fetchone()
            if row and row[0]:
                return str(row[0]).strip()
        except Exception:
            pass
    return ""

def _min_max_dates(con):
    try:
        row = con.execute("""
            SELECT 
              MIN(NULLIF(TRIM(CAST(date_doc AS VARCHAR)), '')) AS first,
              MAX(NULLIF(TRIM(CAST(date_doc AS VARCHAR)), '')) AS last
            FROM t
        """).fetchone()
        if row: return (row[0] or "", row[1] or "")
    except Exception:
        pass
    return ("","")

# ===================== Extracteurs domaine (21 critères) =====================
def _extract_tnm(con, kind="c"):
    cols = _list_cols(con, "t")
    if kind == "c":
        c = None
        for cand in ("ctnm","c_tnm","c_tnm_global","tnm_c"):
            if cand in cols: c = cand; break
    else:
        c = None
        for cand in ("ptnm","p_tnm","p_tnm_global","tnm_p"):
            if cand in cols: c = cand; break
    if c:
        try:
            row = con.execute(
                f"SELECT {c} FROM t WHERE {c} IS NOT NULL AND TRIM(CAST({c} AS VARCHAR))<>'' "
                "ORDER BY date_doc DESC NULLS LAST LIMIT 1"
            ).fetchone()
            if row and row[0]:
                return str(row[0])
        except Exception:
            pass

    df = pd.concat([
        _df_cat_like(con, "rcp"),
        _df_cat_like(con, "histolog"),
        _df_cat_like(con, "imager"),
        _df_cat_like(con, "consult"),
    ], ignore_index=True)
    if df.empty:
        df = _df_where(con, "1=1")
    patT = re.compile(rf"\\b{kind}[tT]\\s*[:=]?\\s*([0-9][0-9a-zA-Z]*)", re.IGNORECASE)
    patN = re.compile(rf"\\b{kind}[nN]\\s*[:=]?\\s*([0-9][0-9a-zA-Z]*)", re.IGNORECASE)
    patM = re.compile(rf"\\b{kind}[mM]\\s*[:=]?\\s*([0-9][0-9a-zA-Z]*)", re.IGNORECASE)
    best = ""
    if not df.empty:
        if "date_doc" in df.columns:
            df = df.sort_values("date_doc", ascending=False)
        for _, row in df.iterrows():
            texts = []
            for c in df.columns:
                if "date" in c or "category" in c: continue
                v = row.get(c, None)
                if isinstance(v, str) and len(v.strip())>0:
                    texts.append(v)
            if not texts: continue
            txt = " | ".join(texts)
            t_m = patT.search(txt)
            n_m = patN.search(txt)
            m_m = patM.search(txt)
            if t_m or n_m or m_m:
                t_s = f"{kind.upper()}T{t_m.group(1)}" if t_m else ""
                n_s = f"{kind.upper()}N{n_m.group(1)}" if n_m else ""
                m_s = f"{kind.upper()}M{m_m.group(1)}" if m_m else ""
                best = "".join([t_s, n_s, m_s]).strip()
                if best: break
    return best

def _extract_yesno_and_date(con, like, keywords=(), prefer_cols=()):
    df = _df_cat_like(con, like)
    found_direct = not df.empty
    date_direct = _min_date_in_df(df) if found_direct else ""

    found_kw, date_kw, snippet = _any_keyword_in_df(df if found_direct else _df_where(con, "1=1"), keywords)

    found = found_direct or found_kw
    date_final = date_direct or date_kw or ""

    details = ""
    if found_direct:
        details = _first_text_from_df(df, prefer_cols)
    if not details:
        details = snippet
    if details:
        details = re.sub(r"^\\s*Category\\s*:\\s*", "", details, flags=re.I)  # nettoie un éventuel "Category: …"
    return ("Oui" if found else "Non"), date_final, details

def _extract_21(con):
    cols = _list_cols(con, "t")

    # 1) Générales
    date_premiere_venue = _first_date_in_cat(con, "consult") or _min_max_dates(con)[0]

    # 2) Tumeur
    circ_cols = (
        "circonstances_de_la_decouverte","circ_decouverte","circonstances",
        "motif","motif_de_consultation",
        "circumstances_of_discovery","discovery_circumstances",
        "circunstancias_del_descubrimiento",
        "umstände_der_entdeckung",
        "circostanze_della_scoperta"
    )
    circonstances = _first_value_in_cols(con, "rcp", circ_cols)
    if not circonstances:
        circonstances = _first_value_in_cols(con, "consult", ("motif","motif_de_consultation","chief_complaint","motivo"))

    date_biopsie = ""
    for c_biop in ("date_biopsie","date_biopsie_initiale"):
        if c_biop in cols:
            try:
                row = con.execute(
                    f"SELECT {c_biop} FROM t WHERE {c_biop} IS NOT NULL AND TRIM(CAST({c_biop} AS VARCHAR))<>'' "
                    f"ORDER BY {c_biop} ASC LIMIT 1"
                ).fetchone()
                if row and row[0]:
                    date_biopsie = _as_date_str(row[0]); break
            except Exception:
                pass
    if not date_biopsie:
        date_biopsie = _first_date_in_cat(con, "histolog")

    date_diag = ""
    for c_diag in ("date_diagnostic","date_diag"):
        if c_diag in cols:
            try:
                row = con.execute(
                    f"SELECT {c_diag} FROM t WHERE {c_diag} IS NOT NULL AND TRIM(CAST({c_diag} AS VARCHAR))<>'' "
                    f"ORDER BY {c_diag} ASC LIMIT 1"
                ).fetchone()
                if row and row[0]:
                    date_diag = _as_date_str(row[0]); break
            except Exception:
                pass
    if not date_diag:
        d_rcp = _first_date_in_cat(con, "rcp")
        d_his = _first_date_in_cat(con, "histolog")
        cand = [d for d in [d_rcp, d_his] if d]
        date_diag = min(cand) if cand else ""

    date_rcp_init = _first_date_in_cat(con, "rcp")

    siege = ""
    for cand in ("siege","siège","site_prim","site","localisation","localization","location",
                 "organe","organe_majoritaire","organ","primary_site"):
        if cand in cols:
            try:
                row = con.execute(
                    f"SELECT {cand} FROM t WHERE {cand} IS NOT NULL AND TRIM(CAST({cand} AS VARCHAR))<>'' "
                    f"GROUP BY {cand} ORDER BY COUNT(*) DESC LIMIT 1"
                ).fetchone()
                if row and row[0]: siege = str(row[0]); break
            except Exception:
                pass
    if not siege:
        siege = _guess_organe(con)

    type_histo = _first_value_in_cols(con, "histolog", ("type_histologique","type","conclusion","histologic_type"))

    grade = ""
    for col_grade in ("grade_histopronostique","grade","sbr","elston"):
        if col_grade in cols:
            try:
                row = con.execute(
                    f"SELECT {col_grade} FROM t WHERE {col_grade} IS NOT NULL AND TRIM(CAST({col_grade} AS VARCHAR))<>'' "
                    "ORDER BY date_doc DESC NULLS LAST LIMIT 1"
                ).fetchone()
                if row and row[0]: grade = str(row[0]); break
            except Exception:
                pass
    if not grade:
        df_h = _df_cat_like(con, "histolog")
        txt = _first_text_from_df(df_h, ("grade","conclusion"))
        m = re.search(r"\\bgrade\\s*[:=]?\\s*([A-Za-z0-9\\+\\-]+)", txt or "", flags=re.I)
        if m: grade = m.group(1)

    ctnm = _extract_tnm(con, "c")
    ptnm = _extract_tnm(con, "p")

    # 3) Traitements
    oui_non_chir, date_chir, nature_interv = _extract_yesno_and_date(
        con,
        like="operat",
        keywords=("intervention","chirurg","opérat","résection","mastect","colect","lobect","thyroïd","surgery","operat"),
        prefer_cols=("procedure_title","nature","intervention","acte","procedure","intitule","intitulé")
    )
    if oui_non_chir == "Non":
        alt_chir = _extract_yesno_and_date(
            con, like="hospital",
            keywords=("intervention","chirurg","opérat","surgery"),
            prefer_cols=("procedure_title","nature","intervention","acte")
        )
        if alt_chir[0] == "Oui":
            oui_non_chir, date_chir, nature_interv = alt_chir

    oui_non_chimio, date_deb_chimio, proto_chimio = _extract_yesno_and_date(
        con,
        like="ordon",
        keywords=("chimio","fluoro","cisplat","carboplat","docetax","paclitax","fec","ac","ec","folfox","folfiri"),
        prefer_cols=("chemotherapy_protocol","protocole_chimio","protocole_chimiotherapie_adjuvante","protocole","traitement")
    )
    for col_ddc in ("date_debut_chimio","date_chimio"):
        if col_ddc in cols and not date_deb_chimio:
            try:
                row = con.execute(
                    f"SELECT {col_ddc} FROM t WHERE {col_ddc} IS NOT NULL AND TRIM(CAST({col_ddc} AS VARCHAR))<>'' "
                    f"ORDER BY {col_ddc} ASC LIMIT 1"
                ).fetchone()
                if row and row[0]: date_deb_chimio = _as_date_str(row[0])
            except Exception:
                pass

    oui_non_hormono, _, _ = _extract_yesno_and_date(
        con, like="ordon",
        keywords=("hormono","tamoxi","letroz","anastro","exemest","goser","leupro","triptore","fulvestr")
    )

    oui_non_immuno, date_immuno, nature_immuno = _extract_yesno_and_date(
        con, like="ordon",
        keywords=("immuno","pembro","nivo","atezo","durval","avelum","ipi","cemipli","anti-pd","anti-ctla"),
        prefer_cols=("medicament","médicament","molecules","molécules","protocole","traitement")
    )

    oui_non_radio, date_deb_radio, _ = _extract_yesno_and_date(
        con, like="radiother",
        keywords=("radioth","rayon","irradiation","rt","radiation","fraction")
    )
    d_rt_cat = _first_date_in_cat(con, "radiother")
    if d_rt_cat and not date_deb_radio:
        date_deb_radio = d_rt_cat

    return {
        "Date de 1ère venue au Centre [lésion en cours]": date_premiere_venue or "",
        "Circonstances de découverte (clinique, dépistage, imagerie fortuite, etc.)": circonstances or "",
        "Date de biopsie diagnostique/initiale": date_biopsie or "",
        "Date de diagnostic": date_diag or "",
        "Date de RCP initiale": date_rcp_init or "",
        "Siège du primitif": siege or "",
        "Type histologique du primitif": type_histo or "",
        "Grade histopronostique": grade or "",
        "Stade cTNM": ctnm or "",
        "Stade pTNM": ptnm or "",
        "Intervention chirurgicale (Oui/Non)": oui_non_chir or "",
        "Nature de l'intervention": nature_interv or "",
        "Date de l'intervention": date_chir or "",
        "Chimiothérapie (Oui/Non)": oui_non_chimio or "",
        "Date de début de chimiothérapie": date_deb_chimio or "",
        "Protocole chimiothérapie adjuvante": (proto_chimio or ""),
        "Hormonothérapie (Oui/Non)": oui_non_hormono or "",
        "Immunothérapie (Oui/Non)": oui_non_immuno or "",
        "Immunothérapie : Nature": nature_immuno or "",
        "Radiothérapie (Oui/Non)": oui_non_radio or "",
        "Date de début de radiothérapie": date_deb_radio or "",
    }

def _yn(val, lang):
    d = {"fr":{"Oui":"Oui","Non":"Non"},
         "en":{"Oui":"Yes","Non":"No"},
         "es":{"Oui":"Sí","Non":"No"},
         "it":{"Oui":"Sì","Non":"No"},
         "de":{"Oui":"Ja","Non":"Nein"}}
    return d.get(lang, d["fr"]).get(val, val)

# ===================== API extraction directe (option) =====================
def extract_for_fiche(doc_fields: Dict[str, Any], lang: str = "fr") -> Dict[str, str]:
    raw_date = get_field(doc_fields, "date_doc")
    date_iso = parse_any_date(raw_date, lang_hint=lang) or raw_date
    return {
        "date_doc": date_iso,
        "localization": get_field(doc_fields, "localization"),
        "histology_observations": get_field(doc_fields, "histology_observations"),
        "histology_conclusion": get_field(doc_fields, "histology_conclusion"),
        "tnm": get_field(doc_fields, "tnm"),
        "oms": get_field(doc_fields, "oms"),
        "protocol": get_field(doc_fields, "protocol"),
        "treatment": get_field(doc_fields, "treatment"),
        "other_meds": get_field(doc_fields, "other_meds"),
    }

# ===================== ANNEXE PDF =====================
def build_annex_pdf(con, xlsx_path, out_pdf_annex, fields_dict, lang: str | None = None):
    lang = _current_lang(lang)
    os.makedirs(os.path.dirname(out_pdf_annex), exist_ok=True)


    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="H1A", parent=styles["Heading1"], fontSize=16, spaceAfter=8))
    styles.add(ParagraphStyle(name="H2A", parent=styles["Heading2"], fontSize=13, textColor=colors.HexColor("#333"), spaceBefore=8, spaceAfter=4))
    styles.add(ParagraphStyle(name="MonoA", parent=styles["BodyText"], fontName="Courier", fontSize=9, textColor=colors.HexColor("#444")))
    body = styles["BodyText"]
    if _FONT:
        body.fontName = _FONT
        styles["Heading1"].fontName = _FONT
        styles["Heading2"].fontName = _FONT
        styles["Title"].fontName = _FONT

    flow = []
    flow.append(Paragraph(t("annex_title", lang), styles["H1A"]))
    flow.append(Paragraph(f"{t('source', lang)} : {os.path.basename(xlsx_path)}", styles["MonoA"]))
    flow.append(Spacer(1,8))

    order = [
        "Date de 1ère venue au Centre [lésion en cours]",
        "Circonstances de découverte (clinique, dépistage, imagerie fortuite, etc.)",
        "Date de biopsie diagnostique/initiale",
        "Date de diagnostic",
        "Date de RCP initiale",
        "Siège du primitif",
        "Type histologique du primitif",
        "Grade histopronostique",
        "Stade cTNM",
        "Stade pTNM",
        "Intervention chirurgicale (Oui/Non)",
        "Nature de l'intervention",
        "Date de l'intervention",
        "Chimiothérapie (Oui/Non)",
        "Date de début de chimiothérapie",
        "Protocole chimiothérapie adjuvante",
        "Hormonothérapie (Oui/Non)",
        "Immunothérapie (Oui/Non)",
        "Immunothérapie : Nature",
        "Radiothérapie (Oui/Non)",
        "Date de début de radiothérapie",
    ]

    for i, label in enumerate(order, start=1):
        disp = LBL.get(lang, LBL["fr"]).get(label, label)
        flow.append(Paragraph(f"{i}. {disp}", styles["H2A"]))
        val = fields_dict.get(label, "") or "—"
        if label in YESNO_KEYS:
            val = _yn(val, lang)
        flow.append(Paragraph(f"<b>{t('retained_value', lang)} :</b> {val}", body))
        flow.append(Paragraph(f"<b>{t('rule_choice', lang)} :</b> {_rule_text(label, lang)}", body))
        flow.append(Spacer(1,4))

        like_sets = []
        if label in ("Date de 1ère venue au Centre [lésion en cours]",):
            like_sets = [("consult", ("date","motif","compte","texte"))]
        elif label in (
            "Circonstances de découverte (clinique, dépistage, imagerie fortuite, etc.)",
            "Date de RCP initiale","Stade cTNM","Stade pTNM",
        ):
            like_sets = [("rcp", ("date","circ","motif","tnm","compte","texte"))]
        elif label in ("Date de biopsie diagnostique/initiale","Type histologique du primitif","Grade histopronostique",):
            like_sets = [("histolog", ("date","type","conclu","grade","texte"))]
        elif label in ("Siège du primitif",):
            like_sets = [("", ("siege","siège","site","local","organe","organe_majoritaire","localization","location","organ"))]
        elif label in ("Intervention chirurgicale (Oui/Non)","Nature de l'intervention","Date de l'intervention",):
            like_sets = [("operat", ("date","nature","intervention","acte","procedure","intitul","procedure_title")),
                         ("hospital", ("date","nature","intervention","acte"))]
        elif label in ("Chimiothérapie (Oui/Non)","Date de début de chimiothérapie","Protocole chimiothérapie adjuvante",):
            like_sets = [("ordon", ("date","protoc","trait","medic","médic","chemotherapy_protocol"))]
        elif label in ("Hormonothérapie (Oui/Non)",):
            like_sets = [("ordon", ("date","medic","médic","hormono","trait"))]
        elif label in ("Immunothérapie (Oui/Non)","Immunothérapie : Nature",):
            like_sets = [("ordon", ("date","immuno","medic","médic","protoc","trait"))]
        elif label in ("Radiothérapie (Oui/Non)","Date de début de radiothérapie",):
            like_sets = [("radiother", ("date","rt","seance","séance","session","fraction","dose","plan","intent","texte"))]
        else:
            like_sets = [("", ("date","category","texte","conclusion","resultat","résultat"))]

        shown = 0
        for like, kw in like_sets:
            df = _df_cat_like(con, like) if like else _df_where(con, "1=1")
            df = _select_cols_for_preview(df, kw)
            if df is not None and not df.empty:
                title = f"{t('sample', lang)} – {(t('sample_cat', lang)+like) if like else t('all_rows', lang)}"
                _preview_table(flow, title, df, lang=lang)
                shown += 1
            if shown >= 2:
                break

        flow.append(Spacer(1,8))
        flow.append(Paragraph("<hr/>", body))

    gen_ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    flow.append(PageBreak())
    flow.append(Paragraph(f"{t('generated_on', lang)} {gen_ts}.", styles["MonoA"]))

    doc = SimpleDocTemplate(out_pdf_annex, pagesize=A4, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    doc.build(flow)
    return out_pdf_annex

def _select_cols_for_preview(df, extra_keywords=()):
    if df is None or df.empty:
        return df
    cols = list(df.columns)
    keep = []
    for c in cols:
        c_l = str(c).lower()
        if ("date" in c_l) or (c_l == "category") or any(k in c_l for k in extra_keywords):
            keep.append(c)
    if len(keep) < 3:
        for c in cols:
            if c not in keep and df[c].dtype == object:
                keep.append(c)
            if len(keep) >= 8:
                break
    return df[keep]

# ===================== PDF principal =====================
def build_pdf(xlsx_path: str, out_pdf: str, job_id: str = "", locale: str | None = None):
    lang = _current_lang(locale)
    con, _ = _load_excel_to_duckdb(xlsx_path)

    organe = _guess_organe(con)
    d_first, d_last = _min_max_dates(con)
    fields = _extract_21(con)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="H1", parent=styles["Heading1"], fontSize=18, spaceAfter=8))
    styles.add(ParagraphStyle(name="H2", parent=styles["Heading2"], fontSize=13, textColor=colors.HexColor("#333333"), spaceBefore=6, spaceAfter=4))
    styles.add(ParagraphStyle(name="Body", parent=styles["BodyText"], leading=14))
    styles.add(ParagraphStyle(name="Mono", parent=styles["BodyText"], fontName="Courier", fontSize=9, textColor=colors.HexColor("#444")))
    styles.add(ParagraphStyle(name="BodyWrap", parent=styles["BodyText"], leading=14, wordWrap="CJK"))
    styles.add(ParagraphStyle(name="Cell", parent=styles["BodyText"], fontSize=9, leading=12, wordWrap="CJK"))
    styles.add(ParagraphStyle(name="CellHead", parent=styles["BodyText"], fontSize=9, leading=12, wordWrap="CJK", textColor=colors.HexColor("#000")))
    if _FONT:
        for k in ("Body","BodyWrap","Cell","CellHead","Heading1","Heading2","Title"):
            if k in styles.byName:
                styles.byName[k].fontName = _FONT

    story = []
    title = t("title", lang)
    if job_id: title += f" (job: {job_id})"
    story += [Paragraph(title, styles["H1"]), Spacer(1, 4)]
    if organe:
        story += [Paragraph(f"<b>{t('organ_major', lang)}:</b> {organe}", styles["BodyWrap"])]
    story += [Paragraph(f"<b>{t('time_range', lang)}:</b> {d_first or '—'} → {d_last or '—'}", styles["BodyWrap"]), Spacer(1, 8)]

    def table_section(title_key, keys):
        story.append(Paragraph(t(title_key, lang), styles["H2"]))
        data = [[Paragraph(t("col_var", lang), styles["CellHead"]), Paragraph(t("col_val", lang), styles["CellHead"])]]
        for key in keys:
            disp = LBL.get(lang, LBL["fr"]).get(key, key)  # libellé dans la langue choisie
            val = fields.get(key, "") or "—"
            if key in YESNO_KEYS:
                val = _yn(val, lang)
            data.append([Paragraph(disp, styles["Cell"]), Paragraph(val, styles["Cell"])])
        label_w = min(260, _AVAIL_W * 0.40)
        val_w   = _AVAIL_W - label_w
        tab = Table(data, colWidths=[label_w, val_w])
        tab.setStyle(TableStyle([
            ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#AAAAAA")),
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#F0F0F0")),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("ALIGN", (0,0), (-1,-1), "LEFT"),
        ]))
        story.extend([tab, Spacer(1,8)])


    table_section("section_general", [
        "Date de 1ère venue au Centre [lésion en cours]",
    ])

    table_section("section_tumor", [
        "Circonstances de découverte (clinique, dépistage, imagerie fortuite, etc.)",
        "Date de biopsie diagnostique/initiale",
        "Date de diagnostic",
        "Date de RCP initiale",
        "Siège du primitif",
        "Type histologique du primitif",
        "Grade histopronostique",
        "Stade cTNM",
        "Stade pTNM",
    ])

    table_section("section_treatment", [
        "Intervention chirurgicale (Oui/Non)",
        "Nature de l'intervention",
        "Date de l'intervention",
        "Chimiothérapie (Oui/Non)",
        "Date de début de chimiothérapie",
        "Protocole chimiothérapie adjuvante",
        "Hormonothérapie (Oui/Non)",
        "Immunothérapie (Oui/Non)",
        "Immunothérapie : Nature",
        "Radiothérapie (Oui/Non)",
        "Date de début de radiothérapie",
    ])

    gen_ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    story += [Spacer(1,10), Paragraph(f"{t('generated_on', lang)} {gen_ts} — {t('source', lang)} : {os.path.basename(xlsx_path)}", styles["Mono"])]


    out_dir = os.path.dirname(out_pdf)
    os.makedirs(out_dir, exist_ok=True)
    doc = SimpleDocTemplate(out_pdf, pagesize=A4, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    doc.build(story)

    annex_pdf = out_pdf.replace(".pdf", "_annexe.pdf")
    try:
        build_annex_pdf(con, xlsx_path, annex_pdf, fields, lang=lang)
        if os.path.exists(annex_pdf):
            concat_pdfs([out_pdf, annex_pdf], out_pdf)
        else:
            print(f"[ANNEXE] non créée: {annex_pdf}")
    except Exception as e:
        print(f"[ANNEXE] génération/concat échouée: {e}")

    return out_pdf

def generate_for_job(job_id: str, xlsx_path: str):
    out_pdf = os.path.join("runs", job_id, "work", "fiche_tumeur.pdf")
    build_pdf(xlsx_path, out_pdf, job_id=job_id)
    return out_pdf
