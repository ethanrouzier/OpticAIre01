# indicators.py
from __future__ import annotations
import os, sqlite3, unicodedata
from dataclasses import dataclass

from pathlib import Path
import pandas as pd
from datetime import date, datetime, timedelta  # + timedelta

# --------- Utils ---------
import re, unicodedata
import pandas as pd

from datetime import date, datetime
import pandas as pd

import re
import json
from collections import Counter


# --- i18n sheet & column aliases ---
SHEET_ALIASES = {
    # clé canonique -> noms possibles selon la langue
    "lab":                {"fr": ["Biologie"], "en": ["Laboratory"], "es": ["Laboratorio"], "it": ["Laboratorio"], "de": ["Labor"]},
    "tumor_board":        {"fr": ["RCP"], "en": ["Tumor Board"], "es": ["Comité de tumores"], "it": ["Tumor board"], "de": ["Tumorboard", "Tumor-Board"]},
    "histology":          {"fr": ["Histologie"], "en": ["Histology"], "es": ["Histología"], "it": ["Istologia"], "de": ["Histologie"]},
    "prescription":       {"fr": ["Ordonnance"], "en": ["Prescription"], "es": ["Prescripción"], "it": ["Prescrizione"], "de": ["Verordnung"]},
    "hospitalization":    {"fr": ["Hospitalisation"], "en": ["Hospitalization"], "es": ["Hospitalización"], "it": ["Ricovero"], "de": ["Krankenhausaufenthalt"]},
    "imaging":            {"fr": ["Imagerie"], "en": ["Imaging"], "es": ["Imagen", "Imágenes"], "it": ["Imaging"], "de": ["Bildgebung"]},
    "radiotherapy":       {"fr": ["Séance de radiothérapie"], "en": ["Radiotherapy session"], "es": ["Sesión de radioterapia"], "it": ["Seduta di radioterapia"], "de": ["Strahlentherapie-Sitzung"]},
    "endoscopy":          {"fr": ["Endoscopie"], "en": ["Endoscopy"], "es": ["Endoscopia"], "it": ["Endoscopia"], "de": ["Endoskopie"]},
    "operative_report":   {"fr": ["Compte rendu opératoire"], "en": ["Operative report"], "es": ["Informe operatorio"], "it": ["Referto operatorio"], "de": ["OP-Bericht"]},
    "remote_monitoring":  {"fr": ["Télésurveillance"], "en": ["Remote monitoring"], "es": ["Telemonitorización"], "it": ["Telemonitoraggio"], "de": ["Telemonitoring"]},
    "consultation":       {"fr": ["Consultation"], "en": ["Consultation"], "es": ["Consulta"], "it": ["Visita", "Consulto", "Consultazione"], "de": ["Konsultation"]},
}

# Colonnes canoniques utiles au calcul OECI (exemples)
# On garde "date_doc" comme colonne canonique pour les dates de documents (tu l’as uniformisée 👍)
COL_ALIASES = {
    "fr": {
        "Date du document": "date_doc",
        "date_document": "date_doc",
        "Organe": "organ",
        "Localisation": "location",
        "Circconstances de découverte": "circumstances",  # si faute possible
        "Circonstances de découverte": "circumstances",
        "Date de découverte": "discovery_date",
        "Proposition thérapeutique": "therapeutic_proposal",
        "Type": "type",
        "Service": "department",
        "Numéro de cycle": "cycle_number",
        "Type de chirurgie": "surgery_type",
        "Protocole de chimiothérapie": "chemo_protocol",
        "Évolution": "evolution",
        "Résultat": "resultat",  # si tu utilises 'result' en canon, mappe vers "result"
        "Résultats": "result",
        "Type de consultation (chirurgie, oncologie médicale, radiothérapie, spécialiste d’organe, ...)": "consultation_type",
    },
    "en": {
        "Document date": "date_doc",
        "Tumor organ": "organ",
        "Location": "location",
        "Circumstances of discovery": "circumstances",
        "Date of discovery": "discovery_date",
        "Therapeutic proposal": "therapeutic_proposal",
        "Type": "type",
        "Department": "department",
        "Cycle number": "cycle_number",
        "Surgery type": "surgery_type",
        "Chemotherapy protocol": "chemo_protocol",
        "Evolution": "evolution",
        "Result": "result",
        "Type of consultation (surgery, medical oncology, radiotherapy, organ specialist, ...)": "consultation_type",
    },
    "es": {
        "Fecha del documento": "date_doc",
        "Órgano tumoral": "organ",
        "Localización": "location",
        "Circunstancias del descubrimiento": "circumstances",
        "Fecha del descubrimiento": "discovery_date",
        "Propuesta terapéutica": "therapeutic_proposal",
        "Tipo": "type",
        "Servicio": "department",
        "Número de ciclo": "cycle_number",
        "Tipo de cirugía": "surgery_type",
        "Protocolo de quimioterapia": "chemo_protocol",
        "Evolución": "evolution",
        "Resultado": "result",
        "Tipo de consulta (cirugía, oncología médica, radioterapia, especialista de órgano, ...)": "consultation_type",
    },
    "it": {
        "Data del documento": "date_doc",
        "Organo tumorale": "organ",
        "Localizzazione": "location",
        "Circostanze della scoperta": "circumstances",
        "Data della scoperta": "discovery_date",
        "Proposta terapeutica": "therapeutic_proposal",
        "Tipo": "type",
        "Reparto": "department",
        "Numero del ciclo": "cycle_number",
        "Tipo di chirurgia": "surgery_type",
        "Protocollo di chemioterapia": "chemo_protocol",
        "Evoluzione": "evolution",
        "Risultato": "result",
        "Tipo di visita (chirurgia, oncologia medica, radioterapia, specialista d’organo, ...)": "consultation_type",
    },
    "de": {
        "Dokumentdatum": "date_doc",
        "Tumororgan": "organ",
        "Lokalisation": "location",
        "Umstände der Entdeckung": "circumstances",
        "Entdeckungsdatum": "discovery_date",
        "Therapievorschlag": "therapeutic_proposal",
        "Typ": "type",
        "Abteilung": "department",
        "Zyklusnummer": "cycle_number",
        "Operationstyp": "surgery_type",
        "Chemotherapieprotokoll": "chemo_protocol",
        "Verlauf": "evolution",
        "Ergebnis": "result",
        "Art der Konsultation (Chirurgie, internistische Onkologie, Strahlentherapie, Organspezialist, ...)": "consultation_type",
    },
}

def _countdown_status(start: date | None, target_days: int):
    """
    Si l'événement d'arrivée est manquant, calcule:
      - remaining_days: (start + target_days) - today
      - countdown_status: 'yellow' | 'orange' | 'red'
      - due_date: ISO 'YYYY-MM-DD'
    """
    if not start:
        return None, "neutral", None
    due = start + timedelta(days=target_days)
    remaining = (due - date.today()).days
    if remaining < 0:
        st = "red"
    elif remaining <= 5:
        st = "orange"
    else:
        st = "yellow"
    return remaining, st, due.isoformat()


def _pick_sheet_name(xls, lang, canonical_key):
    wanted_raw = SHEET_ALIASES.get(canonical_key, {}).get(lang, [])
    if not wanted_raw:
        wanted_raw = SHEET_ALIASES.get(canonical_key, {}).get("fr", [])
    def norm(s):
        import unicodedata, re
        s = unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode("ascii").lower().strip()
        return re.sub(r'[^a-z0-9]+','', s)
    wanted = {norm(n) for n in wanted_raw}
    for name in xls.sheet_names:
        if norm(name) in wanted:
            return name
    return None

def _normalize_columns(df, lang):
    """Renomme les colonnes localisées vers des noms canoniques (si besoin)."""
    aliases = COL_ALIASES.get(lang, {})
    rename_map = {src: dst for src, dst in aliases.items() if src in df.columns}
    if rename_map:
        df = df.rename(columns=rename_map)
    # Sécuriser date_doc en datetime si présent
    if "date_doc" in df.columns:
        try:
            df["date_doc"] = pd.to_datetime(df["date_doc"], errors="coerce")
        except Exception:
            pass
    return df

def _norm_date(s):
    if not s:
        return None
    s = str(s).strip()
    # si ISO "YYYY-MM-DD..." on coupe à 10
    if len(s) >= 10 and s[4] == '-' and s[7] == '-':
        return s[:10]
    return s

def _to_date_any(x) -> date | None:
    """Convertit n'importe quoi (str, datetime, Timestamp…) en date, ou None."""
    if x is None:
        return None
    if isinstance(x, date) and not isinstance(x, datetime):
        return x
    if isinstance(x, (datetime, pd.Timestamp)):
        return x.date()
    # strings / nombres -> parse robuste
    d = pd.to_datetime(x, errors="coerce", dayfirst=True)
    if pd.isna(d):
        d = pd.to_datetime(str(x), errors="coerce")
    if pd.isna(d):
        return None
    return d.date()

def _norm(s: str | None) -> str:
    if s is None: return ""
    s = str(s).strip()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return s.lower()

def _norm_key(s: str) -> str:
    # pour les NOMS DE COLONNES
    # enlève tout ce qui n'est pas [a-z0-9], remplace par espace, compresse, puis underscores
    s = _norm(s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.replace(" ", "_")

def _norm_val(s: str) -> str:
    # pour les VALEURS (catégories)
    s = _norm(s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()  # espaces simples, pas d'underscore

def _find_col(cols, *needles):
    # retrouve une colonne même si le header est du style "{'name': 'date_doc'}"
    keymap = {c: _norm_key(c) for c in cols}
    for needle in needles:
        nk = _norm_key(needle)
        # match exact OU inclus OU suffixe
        for original, k in keymap.items():
            if k == nk or nk in k or k.endswith(nk):
                return original
    return None
def _to_date(x) -> date | None:
    if x is None or (isinstance(x, float) and pd.isna(x)) or str(x).strip() == "":
        return None
    d = pd.to_datetime(x, errors="coerce", dayfirst=True)
    if pd.isna(d):  # second try: ISO-ish
        d = pd.to_datetime(str(x), errors="coerce")
    if pd.isna(d): 
        return None
    return d.date()

def _days(a: date | None, b: date | None) -> int | None:
    if not a or not b: return None
    return (b - a).days

def _status(val: int | None, target: int) -> str:
    if val is None: return "neutral"
    if val <= target: return "green"
    if val <= target + 5: return "orange"
    return "red"


@dataclass
class Hit:
    date: date | None
    row_excel: int | None
    id_dossier: str | None

# --------- Excel -> SQLite + requêtes ---------
def _load_excel_as_df(xlsx_path: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path)

    # --- détecte les colonnes utiles, même si le header est "dict-like"
    col_cat   = _find_col(df.columns, "category", "categorie", "cat")
    _ALL_DATE_HEADERS = {"date_doc", "date du document", "date_document", "datedoc"}
    for _L, _map in COL_ALIASES.items():
        for _src, _dst in _map.items():
            if _dst == "date_doc":
                _ALL_DATE_HEADERS.add(_src)
    col_date = _find_col(df.columns, *_ALL_DATE_HEADERS)
    col_id    = _find_col(df.columns, "id_dossier", "id dossier", "id")
    col_type  = _find_col(df.columns, "type de chirurgie", "type_chirurgie", "type chir")

    # contenu libre pour fallback "type de chirurgie" si pas de colonne dédiée
    col_content = _find_col(df.columns, "content", "contenu", "texte", "resume")

    # --- colonnes de travail internes
    df["__row_excel__"] = df.index + 2  # 1=header, 2=ligne 1
    df["__cat__"]       = df[col_cat].apply(_norm_val) if col_cat else ""
    # date robuste
    def _to_date(x):
        if x is None or (isinstance(x, float) and pd.isna(x)) or str(x).strip() == "":
            return None
        d = pd.to_datetime(x, errors="coerce", dayfirst=True)
        if pd.isna(d):
            d = pd.to_datetime(str(x), errors="coerce")
        if pd.isna(d):
            return None
        return d.date()
    df["__date__"]      = df[col_date].apply(_to_date) if col_date else None
    df["__id__"]        = df[col_id] if col_id else None

    # indicateur "Type de chirurgie non vide"
    if col_type:
        df["__has_surg__"] = df[col_type].astype(str).str.strip().ne("")
    elif col_content:
        # fallback: détecte la mention dans le texte libre (ex: "Type de chirurgie:")
        df["__has_surg__"] = df[col_content].astype(str).str.contains(r"type\s*de\s*chirurg", case=False, regex=True)
    else:
        df["__has_surg__"] = False

    return df


def _write_to_sqlite(df: pd.DataFrame) -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    df.to_sql("data", con, index=False, if_exists="replace")
    return con

def _first_by_category(con, cats) -> Hit:
    q = """
    SELECT __date__ as d, __row_excel__ as r, __id__ as i
    FROM data
    WHERE __cat__ IN ({})
      AND __date__ IS NOT NULL
    ORDER BY d ASC
    LIMIT 1
    """.format(",".join("?" for _ in cats))
    row = con.execute(q, tuple(cats)).fetchone()
    if not row:
        return Hit(None, None, None)
    return Hit(_to_date_any(row[0]),
               int(row[1]) if row[1] is not None else None,
               None if row[2] is None else str(row[2]))

def _first_treatment(con, cats) -> Hit:
    """
    First treatment = plus ancien entre :
      • le 1er compte rendu opératoire (CR opératoire / Operative report / Informe operatorio / Operationsbericht / Referto operatorio)
      • la 1ère hospitalisation avec colonne 'Chirurgie' (ou localisée) à OUI (oui/yes/sí/si/ja/sì)
    Gère FR / EN / ES / DE / IT.
    """
    import unicodedata

    # --- Normalisation locale (accents, casse, espaces/underscores) ---
    def _norm(s):
        if s is None:
            return ""
        s = str(s)
        s = unicodedata.normalize('NFD', s)
        s = ''.join(c for c in s if unicodedata.category(c) != 'Mn')
        return s.lower().replace(' ', '').replace('_', '')

    # Valeurs "oui"
    YES_VALUES = {_norm(v) for v in {
        "oui", "yes", "si", "sí", "ja", "sì", "true", "vrai", "1", "y"
    }}

    # Catégories CR opératoire (5 langues)
    OP_REPORT_CATS_NORM = {_norm(x) for x in {
        # FR
        "Compte rendu opératoire", "CR opératoire", "Bloc opératoire",
        # EN
        "Operative report", "Surgical report",
        # ES
        "Informe operatorio", "Parte operatorio",
        # DE
        "Operationsbericht",
        # IT
        "Referto operatorio", "Resoconto operatorio",
    }}

    # Catégories hospitalisation (5 langues)
    HOSP_CATS_NORM = {_norm(x) for x in {
        # FR
        "Hospitalisation",
        # EN
        "Hospitalization",
        # ES
        "Hospitalización", "Hospitalizacion", "Ingreso hospitalario",
        # DE
        "Hospitalisierung", "Krankenhausaufenthalt",
        # IT
        "Ospedalizzazione", "Ricovero",
    }}

    # Noms possibles de la colonne "Chirurgie" (localisée)
    CAND_SURG_NAMES = {
        # FR
        "Chirurgie", "Acte chirurgical",
        # EN
        "Surgery", "Surgical",
        # ES
        "Cirugía", "Cirugia", "Quirúrgico", "Quirurgico", "Quirúrgica", "Quirurgica",
        # DE
        "Operation", "Operativ", "Chirurgie",
        # IT
        "Chirurgia", "Operazione",
    }

    def _hit_from_row(row):
        return Hit(
            _to_date_any(row[0]),
            int(row[1]) if row[1] is not None else None,
            None if row[2] is None else str(row[2]),
        )

    # Récupérer toutes les lignes avec date, triées
    rows = con.execute("""
        SELECT __date__ AS d, __row_excel__ AS r, __id__ AS i, __cat__ AS c
        FROM data
        WHERE __date__ IS NOT NULL
        ORDER BY d ASC
    """).fetchall()

    # 1) Plus ancien CR opératoire
    row_op = None
    for d, r, i, c in rows:
        if _norm(c) in OP_REPORT_CATS_NORM:
            row_op = (d, r, i)
            break

    # 2) Plus ancienne hospitalisation avec "Chirurgie" = oui (colonne localisée)
    #    On détecte le vrai nom de la colonne présente dans la table.
    try:
        cols_info = con.execute("PRAGMA table_info('data')").fetchall()
        colnames = [row[1] for row in cols_info]  # SQLite & DuckDB: index 1 = name
    except Exception:
        # Fallback générique
        try:
            colnames = [r[0] for r in con.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name='data'"
            ).fetchall()]
        except Exception:
            colnames = []

    norm_to_actual = {_norm(n): n for n in colnames}
    surg_col = None
    for cand in CAND_SURG_NAMES:
        key = _norm(cand)
        if key in norm_to_actual:
            surg_col = norm_to_actual[key]
            break

    row_hosp = None
    if surg_col:
        rows2 = con.execute(f"""
            SELECT __date__ AS d, __row_excel__ AS r, __id__ AS i, __cat__ AS c, {surg_col} AS surg
            FROM data
            WHERE __date__ IS NOT NULL
            ORDER BY d ASC
        """).fetchall()
        for d, r, i, c, surg in rows2:
            if _norm(c) in HOSP_CATS_NORM and _norm(surg) in YES_VALUES:
                row_hosp = (d, r, i)
                break

    # 3) Comparaison & retour
    if not row_op and not row_hosp:
        return Hit(None, None, None)

    if row_op and row_hosp:
        d1 = _to_date_any(row_op[0])
        d2 = _to_date_any(row_hosp[0])
        if d1 and d2:
            return _hit_from_row(row_op) if d1 <= d2 else _hit_from_row(row_hosp)
        if d1:  # d2 invalide
            return _hit_from_row(row_op)
        if d2:  # d1 invalide
            return _hit_from_row(row_hosp)
        return _hit_from_row(row_op)  # tie-breaker
    elif row_op:
        return _hit_from_row(row_op)
    else:
        return _hit_from_row(row_hosp)


def _strip_accents(s: str) -> str:
    import unicodedata as _u
    return ''.join(c for c in _u.normalize('NFKD', str(s or "")) if not _u.combining(c))

def _nk(s: str) -> str:
    # normalise pour matcher la colonne "Organe" même si accents/cas/espace
    s = _strip_accents(str(s or "")).lower().strip()
    return re.sub(r'[^a-z0-9]+','', s)

def _majority_organe_from_excel(xlsx_path: str, sheet_name=None):
    """
    Lit l'Excel et retourne (organe_majoritaire, message_log).
    Cherche une colonne 'Organe' (tolérant variations).
    """
    try:
        df = pd.read_excel(xlsx_path, sheet_name=sheet_name)  # 1er onglet si None
        # si plusieurs onglets (dict), prendre le premier
        if isinstance(df, dict) and df:
            df = next(iter(df.values()))
    except Exception as e:
        return "", f"[Organe] Excel illisible: {e}"

    # trouve la colonne 'Organe'
    col = None
    for c in df.columns:
        n = _nk(c)
        if n == "organe tumoral" or n.endswith("organe tumoral"):
            col = c
            break
    if col is None:
        return "", "[Organe] Colonne 'Organe' introuvable"

    # valeurs non vides
    vals = (df[col].astype(str)
                 .map(lambda s: s.strip())
                 .replace({"nan": ""}))
    vals = [v for v in vals if v]
    if not vals:
        return "", "[Organe] Aucune valeur dans la colonne"

    # normalisation légère visuelle
    def _clean(v: str) -> str:
        v2 = _strip_accents(v)
        v2 = re.sub(r'\b(droit|droite|gauche)\b', '', v2, flags=re.I)
        v2 = re.sub(r'\s+', ' ', v2).strip()
        return v2 or v

    vals = [_clean(v) for v in vals]
    maj = Counter(vals).most_common(1)[0][0]
    return maj, f"[Organe] Majoritaire: {maj}"


# --------- API principale ---------
import re
from datetime import date, datetime
import pandas as pd

def compute_indicators_from_excel(xlsx_path: str, first_contact_date: str | None, lang: str = "fr") -> dict:
    """
    Retourne:
      items: [{id, name, target_days, value_days, row, status, id_dossier}]
      points: {consultation:{date,id,row}, histologie:{...}, rcp_ucp:{...}, traitement:{...}, first_contact:...}
      log: str
    """
    # 1) Charger et indexer (helpers existants)
    df = _load_excel_as_df(xlsx_path)
    con = _write_to_sqlite(df)

    # 2) Groupes de catégories (normalisées)
    # helper pour normaliser comme __cat__
    def _C(*labels): 
        return { _norm_val(x) for x in labels }

    # Consultation
    C_CONSULT = _C(
        "Consultation", "Consulta", "Visita", "Consulto", "Consultazione", "Konsultation"
    )

    # Histologie / Anapath
    C_HISTO = _C(
        "Histologie", "Anapath", "Anatomopathologie",
        "Histology", "Anatomopathology",
        "Histología", "Anatomopatología",
        "Istologia", "Anatomopatologia",
        "Histologie", "Anatomiepathologie"
    )

    # RCP / Tumor Board / MDT
    C_RCPUCP = _C(
        "RCP", "UCP",
        "Tumor Board", "Multidisciplinary Meeting", "Multidisciplinary Team", "MDT",
        "Comité de tumores", "Comite de tumores",
        "Tumorboard", "Tumor-Board"
    )

    # Traitement (CR op + Hosp)
    C_TRAIT = _C(
        "Compte rendu opératoire", "Hospitalisation",
        "Operative report", "Hospitalization",
        "Informe operatorio", "Hospitalización",
        "Referto operatorio", "Ricovero",
        "OP-Bericht", "Krankenhausaufenthalt"
    )

    # 3) Premiers événements
    first_consult = _first_by_category(con, C_CONSULT)
    first_histo   = _first_by_category(con, C_HISTO)
    first_rcpucp  = _first_by_category(con, C_RCPUCP)
    first_trait   = _first_treatment(con, C_TRAIT)

    # 4) 1er contact (le front envoie "YYYY-MM-DD" ; fallback swap si besoin)
    first_contact = _to_date_any(
        re.sub(r'^(\d{4})[-/](\d{1,2})[-/](\d{1,2})$', r'\1-\3-\2', str(first_contact_date or '').strip())
    )

    # 5) Cibles OECI
    targets = {
        "delai1": 10,  # contact -> 1er RV (Consultation)
        "delai2": 21,  # 1er RV -> 1ère anapath (si pas d’anapath avant RV)
        "delai3": 7,   # (1er RV ou anapath si post-RV) -> 1ère UCP/RCP
        "delai4": 10,  # 1ère UCP/RCP -> 1er traitement (CR op/Hosp + type_chir)
    }

    # 6) Calcul des délais
    v1 = _days(first_contact, first_consult.date)
    s1 = _status(v1, targets["delai1"])

    if first_histo.date and first_consult.date:
        if first_histo.date >= first_consult.date:
            v2 = _days(first_consult.date, first_histo.date)
        else:
            v2 = 0  # anapath avant 1er RV => non applicable → 0
    else:
        v2 = None
    s2 = _status(v2, targets["delai2"])

    if first_rcpucp.date and (first_consult.date or first_histo.date):
        base = first_consult.date
        if first_histo.date and first_consult.date and first_histo.date > first_consult.date:
            base = first_histo.date
        v3 = _days(base, first_rcpucp.date) if base else None
    else:
        v3 = None
    s3 = _status(v3, targets["delai3"])

    v4 = _days(first_rcpucp.date, first_trait.date) if (first_rcpucp.date and first_trait.date) else None
    s4 = _status(v4, targets["delai4"])
    # 6-bis) Countdown si arrivée manquante
    r1 = st1 = d1 = None
    if v1 is None and first_contact and not first_consult.date:
        r1, st1, d1 = _countdown_status(first_contact, targets["delai1"])

    r2 = st2 = d2 = None
    # D2 non applicable si anapath avant 1er RV (v2 == 0). Countdown seulement si RV connu et anapath manquante/post-RV.
    if v2 is None and first_consult.date and not first_histo.date:
        r2, st2, d2 = _countdown_status(first_consult.date, targets["delai2"])

    r3 = st3 = d3 = None
    # Base = max(1er RV, 1ère anapath si post-RV), countdown si base connue et RCP/UCP manquante
    base3 = first_consult.date
    if first_histo.date and first_consult.date and first_histo.date > first_consult.date:
        if first_histo.date<first_rcpucp.date:
            base3 = first_histo.date
    if v3 is None and base3 and not first_rcpucp.date:
        r3, st3, d3 = _countdown_status(base3, targets["delai3"])

    r4 = st4 = d4 = None
    # Countdown si 1ère UCP/RCP connue et 1er traitement manquant
    if v4 is None and first_rcpucp.date and not first_trait.date:
        r4, st4, d4 = _countdown_status(first_rcpucp.date, targets["delai4"])

    # 7) Items (même forme que ton ancien code)
    items = [
    {
        "id": "delai1",
        "name": "Délai 1 — contact centre → 1er RV (10 j)",
        "target_days": targets["delai1"],
        "value_days": v1,
        "row": first_consult.row_excel,
        "status": (st1 or s1),
        "id_dossier": first_consult.id_dossier,
        "remaining_days": r1,
        "due_date": d1,
    },
    {
        "id": "delai2",
        "name": "Délai 2 — 1er RV → 1ère anapath (21 j si pas d’anapath avant le 1er RV)",
        "target_days": targets["delai2"],
        "value_days": v2,
        "row": first_histo.row_excel,
        "status": (st2 or s2),
        "id_dossier": first_histo.id_dossier,
        "remaining_days": r2,
        "due_date": d2,
    },
    {
        "id": "delai3",
        "name": "Délai 3 — (1er RV ou 1ère anapath si post-RV) → 1ère UCP/RCP (7 j)",
        "target_days": targets["delai3"],
        "value_days": v3,
        "row": first_rcpucp.row_excel,
        "status": (st3 or s3),
        "id_dossier": first_rcpucp.id_dossier,
        "remaining_days": r3,
        "due_date": d3,
    },
    {
        "id": "delai4",
        "name": "Délai 4 — 1ère UCP/RCP → 1er traitement (10 j)",
        "target_days": targets["delai4"],
        "value_days": v4,
        "row": first_trait.row_excel,
        "status": (st4 or s4),
        "id_dossier": first_trait.id_dossier,
        "remaining_days": r4,
        "due_date": d4,
    },
]


    # 8) Log & points (même shape que ton ancien code)
    log = []
    def _fmt(h: Hit, label: str):
        if not h.date:
            return f"{label}: non trouvé"
        return f"{label}: {h.date.isoformat()} (row {h.row_excel}, id_dossier={h.id_dossier})"

    org, org_info = _majority_organe_from_excel(xlsx_path)
    items.append({
        "id": "organe tumoral",
        "name": "Organe tumoral",
        "value": org,
        "row": None,
        "status": "green" if org else "red",
    })


    log.append(f"Excel: {os.path.basename(xlsx_path)}")


    points = {
        "first_contact": (first_contact.isoformat() if isinstance(first_contact, (date, datetime))
                          else (str(first_contact) if first_contact else None)),
        "consultation":  first_consult.__dict__,
        "histologie":    first_histo.__dict__,
        "rcp_ucp":       first_rcpucp.__dict__,
        "traitement":    first_trait.__dict__,
    }
    
    return {"items": items, "points": points, "log": "\n".join(log)}
