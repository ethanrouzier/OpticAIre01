import re
import warnings
from dateutil.parser._parser import UnknownTimezoneWarning
warnings.filterwarnings("ignore", category=UnknownTimezoneWarning)

import pandas as pd
import random
import calendar
from datetime import datetime, date, timedelta
from dateutil import parser
try:
    from dateutil.relativedelta import relativedelta
except Exception:  # pragma: no cover
    relativedelta = None

# ==========================
#  Constantes et Regex FR
# ==========================
# Espaces FR robustes (inclut NBSP \u00A0 et fine insécable \u202F)
WS = r"[ \t\r\n\u00A0\u202F]+"
# Séparateurs numériques (- / . espaces FR)
SEP = r"[\-\/\.\u00A0\u202F ]"
# Jours de la semaine (optionnels)
WEEKDAY = r"(?:lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche)"
# Mois FR + abréviations
MONTH = (
    r"(?:janv(?:\.|ier)?|f[ée]vr(?:\.|ier)?|mars|avr(?:\.|il)?|mai|juin|"
    r"juil(?:\.|let)?|ao[uû]t|sept(?:\.|embre)?|oct(?:\.|obre)?|"
    r"nov(?:\.|embre)?|d[ée]c(?:\.|embre)?)"
)

# Traduction des mois FR -> EN pour dateutil.parse
_FR_MONTHS = [
    (r"(?i)\bjanv(?:\.|ier)?\b", "January"),
    (r"(?i)\bf[ée]vr(?:\.|ier)?\b", "February"),
    (r"(?i)\bmars\b", "March"),
    (r"(?i)\bavr(?:\.|il)?\b", "April"),
    (r"(?i)\bmai\b", "May"),
    (r"(?i)\bjuin\b", "June"),
    (r"(?i)\bjuil(?:\.|let)?\b", "July"),
    (r"(?i)\bao[uû]t\b", "August"),
    (r"(?i)\bsept(?:\.|embre)?\b", "September"),
    (r"(?i)\boct(?:\.|obre)?\b", "October"),
    (r"(?i)\bnov(?:\.|embre)?\b", "November"),
    (r"(?i)\bd[ée]c(?:\.|embre)?\b", "December"),
]

def _fr_months_to_en(text: str) -> str:
    out = text
    for pat, en in _FR_MONTHS:
        out = re.sub(pat, en, out)
    return out

# Ordre important: YMD -> DMY -> (jour) J mois AAAA
_DATE_PATTERNS = [
    re.compile(r"\b\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}\b"),
    re.compile(r"\b\d{1,2}(?:er)?\s+(?:janvier|février|mars|avril|mai|juin|juillet|ao[uû]t|sept(?:embre)?|oct(?:obre)?|nov(?:embre)?|d[ée]c(?:embre)?)\s+\d{4}\b", re.I),
    re.compile(r"\b(?:19|20)\d{2}-(0?[1-9]|1[0-2])-(0?[1-9]|[12]\d|3[01])\b"),  # ISO YYYY-MM-DD
]

# ==========================
#   Classe Pseudonymizer
# ==========================
class Pseudonymizer:
    """
    Pseudonymiseur réversible (texte/Excel/JSON-like).

    - Pseudonyme noms, emails, adresses, villes, téléphones, IDs, codes postaux
    - Dates: applique un décalage **cohérent par document** (ancré sur la 1ère date rencontrée)
      vers une année de 1901–1949, en conservant les écarts relatifs.
    """

    _ADDRESS_KEYWORDS = ["rue","avenue","boulevard","chemin","place","impasse"]
    _PHONE_REGEX = re.compile(r"(?<!\d)(?:\+|0)\d[\d\s\-\.\(\)]{6,}\d(?!\d)")
    _EMAIL_REGEX = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

    _TITLE_PATTERN = re.compile(
        r"(?i)\b(Dr|Pr|Docteur)\.?\s+"
        r"([A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'’\-]+|[A-ZÀ-ÖØ-Þ]{2,})"
        r"(?:\s+([A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'’\-]+|[A-ZÀ-ÖØ-Þ]{2,}))?"
        r"(?:\s+([A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'’\-]+|[A-ZÀ-ÖØ-Þ]{2,}))?"
        r"(?:\s+([A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'’\-]+|[A-ZÀ-ÖØ-Þ]{2,}))?"
    )

    def __init__(self, name_pseudos, email_pseudos, address_pseudos,
                 city_pseudos, phone_pseudos,
                 initial_name_mapping=None, extra_names=[ "BACLESSE", "Caen"],
                 common_names=None, seed=None):
        self.extra_names = [n.lower() for n in (extra_names or [])]
        self.common_names = [n.lower() for n in (common_names or [])]
        self.common_names_wholeword = (
            re.compile(r'(?i)\b(?:' + '|'.join(map(re.escape, self.common_names)) + r')\b')
            if self.common_names else None
        )
        self.rng = random.Random(seed)
        self.pools = {
            'names': list(name_pseudos),
            'emails': list(email_pseudos),
            'addresses': list(address_pseudos),
            'cities': list(city_pseudos),
            'phones': list(phone_pseudos)
        }
        for pool in self.pools.values():
            self.rng.shuffle(pool)
        self.mappings = {cat: {} for cat in self.pools}
        self.mappings['ids'] = {}
        self.mappings['postals'] = {}
        self.id_mapping = {}
        self.encountered = {cat: [] for cat in list(self.pools) + ['ids','postals']}
        if initial_name_mapping:
            for orig, pseudo in initial_name_mapping.items():
                key = str(orig).lower()
                self.mappings['names'][key] = pseudo
                self.encountered['names'].append(orig)
                if pseudo in self.pools['names']:
                    self.pools['names'].remove(pseudo)
        self.date_shift = None  # relativedelta ou timedelta
        self.drug_exceptions = ['CARBOPLATINE', 'PACLITAXEL']
        self.name_particles = {"de","du","des","d'","d’","de la","de l'","de l’","le","la","les","van","von","di","da"}

    # ---------------------
    #  Utilitaires mapping
    # ---------------------
    def _get_pseudo(self, orig, cat):
        key = str(orig)
        if cat == 'names':
            norm = key.lower().strip()
            if norm in self.name_particles:
                return key
        if cat == 'ids':  # ID patient numérique long
            if key in self.id_mapping:
                return self.id_mapping[key]
            while True:
                val = f"{self.rng.randint(0, 999999999999):012d}"
                if val not in self.id_mapping.values():
                    break
            self.id_mapping[key] = val
            self.encountered['ids'].append(orig)
            return val
        if cat == 'postals':
            mapping = self.mappings['postals']
            if key in mapping:
                return mapping[key]
            while True:
                val = f"{self.rng.randint(0, 99999):05d}"
                if val not in mapping.values():
                    break
            mapping[key] = val
            self.encountered['postals'].append(orig)
            return val
        mapping = self.mappings[cat]
        pool = self.pools[cat]
        lookup = key.lower()
        if lookup in mapping:
            return mapping[lookup]
        if not pool:
            raise ValueError(f"Plus de pseudos pour {cat}")
        pseudo = pool.pop(0)
        mapping[lookup] = pseudo
        self.encountered[cat].append(orig)
        return pseudo

    # ---------------------
    #  Dates (remplacement)
    # ---------------------
    def _repl_date_factory(self):
        def repl(m):
            s = m.group(0)
            try:
                s2 = _fr_months_to_en(s)
                dt = parser.parse(s2, dayfirst=True, fuzzy=True)
                if getattr(self, "date_shift", None) is None:
                    rng = random.Random(getattr(self, "seed", 42))
                    target_year = rng.randrange(1901, 1950)
                    delta_years = target_year - dt.year
                    if relativedelta is not None:
                        self.date_shift = relativedelta(years=delta_years)
                    else:
                        self.date_shift = timedelta(days=delta_years * 365)
                new_dt = dt + self.date_shift
                return new_dt.strftime('%d/%m/%Y')
            except Exception:
                return s
        return repl

    # ---------------------
    #  Cellule -> texte
    # ---------------------
    def pseudonymize_cell(self, cell):
        if not isinstance(cell, str):
            if isinstance(cell, (datetime, date)) and self.date_shift:
                try:
                    return cell + self.date_shift
                except Exception:
                    return cell
            return cell
        text = cell
        # Médicaments à préserver
        for drug in self.drug_exceptions:
            text = re.sub(rf"(?i)\b{re.escape(drug)}\b", drug, text)

        # Titres (Dr/Pr/Docteur) + 1 à 4 tokens suivants
        def repl_title(m):
            title = m.group(1)
            parts = [m.group(i) for i in (2,3,4,5) if m.group(i)]
            pseudos = [self._get_pseudo(p, 'names') for p in parts]
            return f"{title} " + " ".join(pseudos)
        text = self._TITLE_PATTERN.sub(repl_title, text)

        # Prénoms communs (mot entier)
        if self.common_names_wholeword:
            text = self.common_names_wholeword.sub(lambda m: self._get_pseudo(m.group(0), 'names'), text)

        # Noms supplémentaires (substring, insensible casse)
        for extra in self.extra_names:
            pattern = re.compile(re.escape(extra), flags=re.IGNORECASE)
            text = pattern.sub(lambda m: self._get_pseudo(m.group(0), 'names'), text)

        # Codes postaux (5 chiffres)
        text = re.sub(r"\b\d{5}\b", lambda m: self._get_pseudo(m.group(0), 'postals'), text)
        # Emails
        text = self._EMAIL_REGEX.sub(lambda m: self._get_pseudo(m.group(0), 'emails'), text)
        # Téléphones
        text = self._PHONE_REGEX.sub(lambda m: self._get_pseudo(m.group(0), 'phones'), text)
        # Adresses (mots-clés rue/avenue/… jusqu'à fin de segment)
        addr_pat = re.compile(r"\b(?:" + "|".join(self._ADDRESS_KEYWORDS) + r")\b[^,;\n]*", flags=re.IGNORECASE)
        text = addr_pat.sub(lambda m: self._get_pseudo(m.group(0), 'addresses'), text)

        # IDs (séquences numériques >= 7)
        text = re.sub(r"\b\d{7,}\b", lambda m: self._get_pseudo(m.group(0), 'ids'), text)

        # Si un nom pseudo est suivi d'un mot capitalisé : pseudo aussi
        for pseudo in list(self.mappings['names'].values()):
            pattern_follow = re.compile(rf"(?<=\b{re.escape(pseudo)}\b)\s+((?:[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'’\-]*|[A-ZÀ-ÖØ-Þ]{{2,}}))")
            text = pattern_follow.sub(lambda m: ' ' + self._get_pseudo(m.group(1), 'names'), text)

        # Remplacer les occurrences exactes des noms déjà mappés
        for orig_lower, pseudo in list(self.mappings['names'].items()):
            if orig_lower in self.name_particles:
                continue
            text = re.sub(rf"(?i)\b{re.escape(orig_lower)}\b", pseudo, text)

        # Dates inline (après tous les remplacements de texte)
        repl_date = self._repl_date_factory()
        for pat in _DATE_PATTERNS:
            text = pat.sub(repl_date, text)

        # Médicaments restaurés (au cas où)
        for drug in self.drug_exceptions:
            text = re.sub(rf"(?i)\b{re.escape(drug)}\b", drug, text)
        return text

    # ---------------------
    #  DataFrame -> DataFrame
    # ---------------------
    def _compute_date_shift(self, df: pd.DataFrame):
        """Heuristique: ancre sur la **première date** parsée dans le DF.
        Si aucune date, pas de décalage. (La première date inline initialisera
        aussi le shift si besoin dans pseudonymize_cell.)"""
        dt0 = None
        for col in df.columns:
            for cell in df[col]:
                if isinstance(cell, (datetime, date)):
                    dt0 = datetime.combine(cell, datetime.min.time()) if isinstance(cell, date) and not isinstance(cell, datetime) else cell
                    break
                if isinstance(cell, str):
                    s = _fr_months_to_en(cell)
                    try:
                        dt0 = parser.parse(s, dayfirst=True, fuzzy=True)
                        break
                    except Exception:
                        continue
            if dt0:
                break
        if not dt0:
            self.date_shift = None
            return
        rng = random.Random(getattr(self, "seed", 42))
        target_year = rng.randrange(1901, 1950)
        delta_years = target_year - dt0.year
        if relativedelta is not None:
            self.date_shift = relativedelta(years=delta_years)
        else:
            self.date_shift = timedelta(days=delta_years * 365)

    def pseudonymize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        self._compute_date_shift(df)
        df_out = df.copy()
        # Texte
        for col in df_out.select_dtypes(include=['object']).columns:
            df_out[col] = df_out[col].map(self.pseudonymize_cell)
        # Datetime natifs
        if self.date_shift:
            for col in df_out.select_dtypes(include=['datetime']).columns:
                try:
                    df_out[col] = df_out[col] + self.date_shift
                except Exception:
                    pass
        return df_out

    # ---------------------
    #  Rétro (optionnel simple)
    # ---------------------
    def _deanonymize_cell(self, cell):  # best-effort
        if not isinstance(cell, str):
            if isinstance(cell, pd.Timestamp) and self.date_shift:
                try:
                    return cell - self.date_shift
                except Exception:
                    return cell
            return cell
        text = cell
        # Inverses (best effort)
        for cat in ['emails', 'phones', 'addresses', 'cities', 'names']:
            inv = {v: k for k, v in self.mappings[cat].items()}
            for pseudo, orig in inv.items():
                text = re.sub(rf"(?i)\b{re.escape(str(pseudo))}\b", str(orig), text)
        # IDs
        inv_id = {v: k for k, v in self.id_mapping.items()}
        for pseudo, orig in inv_id.items():
            text = re.sub(rf"\b{re.escape(pseudo)}\b", orig, text)
        # Dates (approx)
        for pat in _DATE_PATTERNS:
            def repl_inv(m):
                s = m.group(0)
                try:
                    dt = parser.parse(s, dayfirst=True, fuzzy=True)
                    if self.date_shift:
                        orig_dt = dt - self.date_shift
                        return orig_dt.strftime('%d/%m/%Y')
                except Exception:
                    pass
                return s
            text = pat.sub(repl_inv, text)
        return text

    def deanonymize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        for col in df_out.select_dtypes(include=['object']).columns:
            df_out[col] = df_out[col].map(self._deanonymize_cell)
        if self.date_shift:
            for col in df_out.select_dtypes(include=['datetime']).columns:
                try:
                    df_out[col] = df_out[col] - self.date_shift
                except Exception:
                    pass
        return df_out

    # ---------------------
    #  Export mappings
    # ---------------------
    def export_mappings(self, outdir):
        import os
        os.makedirs(outdir, exist_ok=True)
        for cat, mapping in list(self.mappings.items()) + [('ids', self.id_mapping)]:
            df_map = pd.DataFrame(list(mapping.items()), columns=[f'original_{cat}', f'pseudo_{cat}'])
            df_enc = pd.DataFrame(self.encountered[cat], columns=[f'encountered_{cat}'])
            df_map.to_excel(f"{outdir}/mapping_{cat}.xlsx", index=False)
            df_enc.to_excel(f"{outdir}/encountered_{cat}.xlsx", index=False)

# ==========================
#  Aides au décryptage
# ==========================
import os
import json

def load_mappings(mapping_dir):
    """Charge les fichiers mapping_{cat}.xlsx et retourne un dict pseudo→original."""
    cats = ['names','emails','ids','addresses','cities','phones']
    inv = {}
    for cat in cats:
        path = os.path.join(mapping_dir, f'mapping_{cat}.xlsx')
        if not os.path.exists(path):
            inv[cat] = {}
            continue
        df_map = pd.read_excel(path)
        orig_col = f'original_{cat}'
        pseudo_col = f'pseudo_{cat}'
        inv[cat] = dict(zip(df_map[pseudo_col].astype(str), df_map[orig_col].astype(str)))
    return inv

def load_date_shift(mapping_dir):
    meta_path = os.path.join(mapping_dir, 'metadata.json')
    if not os.path.exists(meta_path):
        return timedelta(0)
    meta = json.load(open(meta_path, 'r', encoding='utf-8'))
    days = meta.get('date_shift_days', 0)
    try:
        return timedelta(days=int(days))
    except Exception:
        return timedelta(0)

def decrypt_text(text, mapping_dir):
    """Remplace les pseudos par les originaux et restaure les dates (best-effort)."""
    inv = load_mappings(mapping_dir)
    out = text
    for cat in ['emails','phones','addresses','cities','names','ids']:
        mapping = inv.get(cat, {})
        for pseudo, orig in mapping.items():
            out = re.sub(rf"\b{re.escape(str(pseudo))}\b", str(orig), out)
    # Dates (si metadata.json fournit un shift en jours)
    shift = load_date_shift(mapping_dir)
    if shift:
        for pat in _DATE_PATTERNS:
            def repl_date(m):
                s = m.group(0)
                try:
                    dt = parser.parse(s, dayfirst=True, fuzzy=True)
                    return (dt - shift).strftime('%d/%m/%Y')
                except Exception:
                    return s
            out = pat.sub(repl_date, out)
    return out
