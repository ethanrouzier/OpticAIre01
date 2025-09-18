

"""
Extraction de champs par catégorie — version STRICT "remplit les boîtes"
------------------------------------------------------------------------
Objectif (selon la demande) :
- Ne crée pas de nouveaux champs ; n'en renomme pas ; n'en devine pas.
- Utilise exactement les champs ("boîtes") définis par l'étape Choose Fields
  et éventuellement modifiés par l'utilisateur dans le catalog JSON.
- Pour chaque document, lit la catégorie et récupère le schéma correspondant
  dans le catalogue, puis remplit uniquement ces clés.
- Si un champ est vide dans le catalogue (boîte supprimée), il n'est pas extrait.
- Si aucune catégorie correspondante n'est trouvée : on ignore le fichier (warning).

Compat I/O : ajoute toujours `field_schema` (copie du schéma) et `fields` (valeurs) dans le JSON de sortie.

Usage exemple :
  python extract_fields_by_category.py \
      --src runs/JOB/work/json_from_files \
      --out-dst runs/JOB/work/json_extracted \
      --model mistral-large-latest \
      --catalog runs/JOB/work/catalog/fields_catalog.json \
      --extraction-extra "Dates AAAA-MM-JJ si possible" \
      --seed 42
"""

from __future__ import annotations
import argparse
import json
import os
import re
import random
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from tqdm import tqdm

# ===== LLM (Mistral) =====
try:
    from mistralai import Mistral
except Exception:
    Mistral = None  # type: ignore

# ===== Fichiers à ignorer (catalogues) =====
CATALOG_FILENAMES = {"fields_catalog.json", "field_catalog.json", "catalog.json"}




# ===== Utilitaires fichiers =====

def ensure_date_doc_everywhere(catalog_map: Catalog) -> Catalog:
    """Ajoute 'date_doc' en TÊTE de chaque catégorie si absent (dédupe au passage)."""
    for cat, fields in list(catalog_map.items()):
        fields = fields or []
        seen = set()
        boxed = []
        for f in fields:
            # sécurité: s'assurer qu'on a bien un dict Field
            if not isinstance(f, dict):
                name = str(f).strip()
                f = {"name": name} if name else None
            if not f:
                continue
            name = str(f.get("name", "")).strip()
            if not name:
                continue
            key = name.lower()
            if key not in seen:
                seen.add(key)
                boxed.append(f)

        if "date_doc" not in seen:
            boxed.insert(0, {"name": "date_doc"})  # type implicite "string"

        catalog_map[cat] = boxed
    return catalog_map

def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def iter_json_files(root: Path) -> List[Path]:
    return [p for p in root.rglob("*.json") if p.is_file()]

def dst_for(src_root: Path, out_root: Optional[Path], file_path: Path) -> Path:
    return file_path if out_root is None else (out_root / file_path.relative_to(src_root))

# ===== Catégorie & contenu =====
def get_category_for_json(doc: dict, json_path: Path) -> str:
    for k in ("category", "category_name", "type", "Type", "categorie", "Catégorie"):
        v = doc.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    fields = doc.get("Fields") or doc.get("fields") or {}
    if isinstance(fields, dict):
        for k in ("category", "category_name", "type", "Type"):
            v = fields.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return ""

def extract_text_content(doc: Any) -> str:
    if isinstance(doc, dict):
        for k in ("content","Content","resume","Résumé","text","Texte","full_text","ocr","OCR"):
            v = doc.get(k)
            if isinstance(v, str) and v.strip():
                return v
    try:
        return json.dumps(doc, ensure_ascii=False)
    except Exception:
        return str(doc)

# ===== Détection d'un JSON de type catalogue =====
def is_catalog_shape(data: Any) -> bool:
    if not isinstance(data, dict):
        return False
    if "category" in data:  # une vraie fiche
        return False
    return ("categories" in data) or ("fields_catalog" in data)

# ===== Normalisation du catalogue =====
Field = Dict[str, Any]  # {name, description?, type?}
Catalog = Dict[str, List[Field]]  # {category -> [Field, ...]}

def _as_field_dict(x: Any) -> Optional[Field]:
    if x is None:
        return None
    if isinstance(x, dict):
        name = str(x.get("name", "")).strip()
        if not name:
            return None
        out: Field = {"name": name}
        if x.get("description"):
            out["description"] = str(x["description"])  # facultatif
        if x.get("type"):
            out["type"] = str(x["type"]).strip()
        # NEW: conserve les valeurs autorisées si présentes
        if "allowed_values" in x and x["allowed_values"] is not None:
            try:
                out["allowed_values"] = list(x["allowed_values"])
            except Exception:
                pass
        return out
    if isinstance(x, str):
        s = x.strip()
        return {"name": s} if s else None
    return None

def normalize_catalog(raw: Any) -> Catalog:
    """Convertit les diverses formes en mapping {cat: [Field, ...]} en conservant les objets Field si présents."""
    out: Catalog = {}
    if isinstance(raw, dict) and isinstance(raw.get("categories"), list):
        for it in raw["categories"]:
            if not isinstance(it, dict):
                continue
            name = (it.get("name") or it.get("category") or "").strip()
            fields = it.get("fields") or []
            boxed = [f for f in (_as_field_dict(x) for x in fields) if f]
            if name:
                out[name] = boxed
        return out
    if isinstance(raw, dict) and isinstance(raw.get("fields_catalog"), dict):
        inner = raw["fields_catalog"]
        for k, v in inner.items():
            if isinstance(v, dict) and "fields" in v:
                fields = v.get("fields") or []
            else:
                fields = v or []
            boxed = [f for f in (_as_field_dict(x) for x in fields) if f]
            out[str(k)] = boxed
        return out
    if isinstance(raw, dict):
        for k, v in raw.items():
            if isinstance(v, dict) and "fields" in v:
                fields = v.get("fields") or []
            else:
                fields = v or []
            boxed = [f for f in (_as_field_dict(x) for x in fields) if f]
            out[str(k)] = boxed
        return out
    if isinstance(raw, list):
        for it in raw:
            if not isinstance(it, dict):
                continue
            name = (it.get("name") or it.get("category") or "").strip()
            fields = it.get("fields") or []
            boxed = [f for f in (_as_field_dict(x) for x in fields) if f]
            if name:
                out[name] = boxed
        return out
    return {}

# ===== Client LLM & parsing =====
def mistral_client(api_key: Optional[str]) -> Any:
    if Mistral is None:
        raise RuntimeError("Le paquet 'mistralai' n'est pas installé. pip install mistralai")
    key = api_key or os.environ.get("MISTRAL_API_KEY")
    if not key:
        raise RuntimeError("Fournissez une clé API Mistral via $MISTRAL_API_KEY ou --api-key")
    return Mistral(api_key=key)

def _strip_md_outside_strings(s: str) -> str:
    out = []
    in_str = False
    esc = False
    for ch in s:
        if in_str:
            out.append(ch)
            if ch == '"' and not esc:
                in_str = False
            esc = (ch == "\\" and not esc)
        else:
            if ch == '"':
                in_str = True
                out.append(ch)
                esc = False
            elif ch in ('*', '`'):
                continue
            else:
                out.append(ch)
    return ''.join(out)

def parse_json_object(text: str) -> Any:
    text = (text or "").replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'")
    candidates = re.findall(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if not candidates:
        candidates = [text]

    def _balanced_slice(s: str, oc: str, cc: str) -> Optional[str]:
        start = s.find(oc)
        if start < 0:
            return None
        depth = 0
        for i in range(start, len(s)):
            if s[i] == oc:
                depth += 1
            elif s[i] == cc:
                depth -= 1
                if depth == 0:
                    return s[start:i+1]
        return None

    extra = []
    for s in candidates:
        for oc, cc in [("{", "}"), ("[", "]")]:
            b = _balanced_slice(s, oc, cc)
            if b and b not in candidates and b not in extra:
                extra.append(b)
    candidates += extra

    for cand in candidates:
        cand_clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", cand.strip(), flags=re.IGNORECASE | re.DOTALL)
        cand_clean = _strip_md_outside_strings(cand_clean)
        try:
            data = json.loads(cand_clean)
            # cas {"fields": {...}}
            if isinstance(data, dict) and isinstance(data.get("fields"), dict):
                return data["fields"]
            # cas dict unique {"name": ..., "value": ...}
            if isinstance(data, dict) and "name" in data and any(k in data for k in ("value","val","text")):
                return {str(data.get("name") or "").strip(): data.get("value", data.get("val", data.get("text", "")))}
            # liste de {name,value}
            if isinstance(data, list) and all(isinstance(x, dict) for x in data) and all(("name" in x) for x in data):
                out = {}
                for x in data:
                    k = str(x.get("name", "")).strip()
                    if not k:
                        continue
                    v = x.get("value", x.get("val", x.get("text", "")))
                    out[k] = v
                return out
            # cas standard : dict
            if isinstance(data, dict):
                return data
            # cases exotiques : liste de paires
            if isinstance(data, list) and all(isinstance(x, list) and len(x) == 2 for x in data):
                try:
                    return dict(data)
                except Exception:
                    pass
            return data
        except Exception:
            continue
    raise ValueError("Réponse LLM non JSON.")

# ===== Appel LLM pour l'extraction (STRICT) =====
def _truncate(s: str, n: int) -> str:
    s = (s or "").replace("\u0000", " ").strip()
    return s if len(s) <= n else s[:n] + "…"


LANG_MAP = {
    "fr": "français",
    "en": "anglais",
    "es": "espagnol",
    "it": "italien",
    "de": "allemand",
}

# TUNING PROMPT

def build_fields_instructions(schema_fields: List[Field]) -> str:
    """Construit les contraintes par champ à injecter dans le prompt.
    Ajoute la phrase STRICTE uniquement si allowed_values est défini."""
    lines = []
    for f in schema_fields:
        name = str(f.get("name", "")).strip()
        if not name:
            continue
        allowed = f.get("allowed_values") or []
        if allowed:
            opts = ", ".join([f"\"{v}\"" for v in allowed])
            lines.append(
                f"- {name}: réponds STRICTEMENT par UNE SEULE valeur exactement égale à l'une de [{opts}]. Aucun autre texte."
            )
        else:
            lines.append(f"- {name}: donne une courte valeur pertinente, sans commentaire.")
    return "\n".join(lines)

lang_code = (os.getenv("APP_LANG", "fr") or "fr").lower()
lang_name = LANG_MAP.get(lang_code, "français")




def extract_values_strict(content: str, schema_fields: List[Field], extra: Optional[str], *, client: Any, model: str, max_chars: int = 7000) -> Dict[str, Any]:
    keys = [str(f.get("name", "")) for f in schema_fields if f.get("name")]
    fields_for_prompt = [
        {"name": str(f.get("name", "")), "description": str(f.get("description", "")), "type": str(f.get("type", "string"))}
        for f in schema_fields if f.get("name")
    ]
    fields_block = build_fields_instructions(schema_fields)
    sys_msg = (
    f"Réponds dans la langue des documents."
    "Tu es un extracteur. "
    "Réponds UNIQUEMENT par un OBJET JSON PUR {\"<name>\": <valeur>, ...}. "
    "N’ajoute aucune clé, n’en retire aucune. Pas de Markdown."
)

    user = {
        "schema": [{
            "name": str(f.get("name","")),
            "type": str(f.get("type","string")),
            "allowed_values": list(f.get("allowed_values") or [])
        } for f in schema_fields if f.get("name")],
        "keys_required": [str(f.get("name","")) for f in schema_fields if f.get("name")],
        "return_exactly_like": {k: "" for k in [str(f.get("name","")) for f in schema_fields if f.get("name")]},
        "content": _truncate(content, max_chars),
        
        "guidelines": (
            "Si le type est absent, considère 'string'. "
            "Retourne UNIQUEMENT l’objet JSON des valeurs. "
            "Si une valeur est absente: '' (ou [] pour type=list, false pour type=bool). "
            "SI un champ possède 'allowed_values', tu dois répondre par UNE SEULE valeur "
            "strictement égale à l'une de ces valeurs autorisées."
            "Donne les dates au format YYYY-MM-DD"
        ),
        "field_instructions": (
            "Champs à extraire et contraintes par champ :\n" + fields_block
        ),
    }

    if extra:
        user["extra_instructions"] = str(extra)

    kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
        temperature=0.0,
        max_tokens=3072,
    )
    try:
        kwargs["response_format"] = {"type": "json_object"}  # si supporté
    except Exception:
        pass

    resp = client.chat.complete(**kwargs)
    txt = resp.choices[0].message.content if resp and resp.choices else ""
    data = parse_json_object(txt)
    if not isinstance(data, dict):
        data = {}

    types = {str(f.get("name","")): str(f.get("type","string")).lower() for f in schema_fields if f.get("name")}
    out: Dict[str, Any] = {}
    for k in keys:
        v = data.get(k, None)
        t = types.get(k, "string")
        if v is None:
            v = [] if t == "list" else (False if t == "bool" else "")
        elif t == "list" and not isinstance(v, list):
            v = [v] if v != "" else []
        elif t == "bool" and not isinstance(v, bool):
            if isinstance(v, str):
                v_low = v.strip().lower()
                v = v_low in ("true","vrai","oui","1","affirmatif")
            else:
                v = bool(v)
        out[k] = v
    return out

# ===== Chargement du catalogue =====
def find_catalog_path(src: Path, out_dst: Optional[Path], explicit: Optional[Path]) -> Path:
    if explicit and explicit.exists():
        return explicit
    candidates = []
    if out_dst:
        candidates.append(out_dst / "fields_catalog.json")
    candidates += [
        src / "fields_catalog.json",
        src.parent / "catalog" / "fields_catalog.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("Aucun fields_catalog.json trouvé (spécifie --catalog)")

# ===== RUN =====
def run(
    src: Path,
    out_dst: Optional[Path],
    model: str,
    api_key: Optional[str],
    choose_fields_extra: Optional[str],  # ignoré (compat)
    extraction_extra: Optional[str],
    seed: int,
    catalog: Optional[Path],
):
    random.seed(seed)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    client = mistral_client(api_key)

    cat_path = find_catalog_path(src, out_dst, catalog)
    raw_catalog = read_json(cat_path)
    catalog_map = normalize_catalog(raw_catalog)          # {cat: [Field,...]}
    catalog_map = ensure_date_doc_everywhere(catalog_map) # + injection date_doc


    files = iter_json_files(src)
    pbar = tqdm(files, desc="Extraction des champs")

    for json_path in pbar:
        if json_path.name in CATALOG_FILENAMES:
            logging.info("Skip catalog file: %s", json_path.name)
            continue

        try:
            doc = read_json(json_path)
        except Exception as e:
            pbar.write(f"[WARN] Lecture: {json_path}: {e}")
            continue

        if is_catalog_shape(doc):
            logging.info("Skip catalog-shaped JSON: %s", json_path.name)
            continue

        cat = get_category_for_json(doc, json_path)
        if not cat or cat not in catalog_map:
            pbar.write(f"[WARN] Pas de schéma pour catégorie '{cat or '∅'}'. Fichier ignoré: {json_path.name}")
            continue

        schema_fields = catalog_map[cat]
        if not schema_fields:
            pbar.write(f"[WARN] Schéma vide pour catégorie '{cat}'. Ignoré: {json_path.name}")
            continue

        content = extract_text_content(doc)
        values = extract_values_strict(content, schema_fields, extraction_extra, client=client, model=model)

        out_path = dst_for(src, out_dst, json_path)
        out_doc = dict(doc)
        out_doc["field_schema"] = schema_fields
        out_doc["fields"] = values
        write_json(out_path, out_doc)

        pbar.set_postfix_str(cat[:40])

    pbar.close()

# ===== CLI =====
def main():
    ap = argparse.ArgumentParser(description="Extraction STRICTE de champs par catégorie (remplit les boîtes)")
    ap.add_argument("--src", required=True, type=Path, help="Dossier de JSON")
    ap.add_argument("--out-dst", type=Path, default=None, help="Dossier de destination (sinon in-place)")
    ap.add_argument("--model", default="mistral-large-latest", help="Modèle LLM")
    ap.add_argument("--api-key", default=None, help="Clé API Mistral (sinon $MISTRAL_API_KEY)")
    ap.add_argument("--choose-fields-extra", default=None, help="(ignoré — compat)")
    ap.add_argument("--extraction-extra", default=None, help="Consigne supplémentaire pour l'extraction")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--catalog", type=Path, default=None, help="Chemin vers fields_catalog.json (prioritaire)")
    args = ap.parse_args()

    run(
        src=args.src,
        out_dst=args.out_dst,
        model=args.model,
        api_key=args.api_key,
        choose_fields_extra=args.choose_fields_extra,
        extraction_extra=args.extraction_extra,
        seed=args.seed,
        catalog=args.catalog,
    )

if __name__ == "__main__":
    main()
