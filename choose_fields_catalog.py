
"""
Choix de champs par catégorie (LLM) — construit / met à jour le catalogue.
Conserve les mêmes options que le script monolithique (les options d'extraction sont ignorées ici).
- Lit les JSON dans --src
- Pour chaque *nouvelle* catégorie rencontrée : demande au LLM une liste de champs (schéma)
- Écrit/merge le résultat dans fields_catalog.json dans --out-dst (ou à la racine de --src si non fourni)

Usage (ex.) :
  python choose_fields_catalog.py --src /chemin/json --model mistral-large-latest \
      --choose-fields-extra "Conserver des champs textes larges." --seed 42
"""

from __future__ import annotations
import argparse, json, os, re, random
from pathlib import Path
from typing import Any, Dict, List, Optional
from tqdm import tqdm

# LLM (Mistral)
try:
    from mistralai import Mistral
except Exception:
    Mistral = None  # type: ignore

# ---------- Utilitaires fichiers ----------
def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"JSON invalide: {path} ({e})")

def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def iter_json_files(root: Path) -> List[Path]:
    return [p for p in root.rglob("*.json") if p.is_file()]

def _strip_md_outside_strings(s: str) -> str:
    """
    Supprime les **, * et ` situés hors des chaînes JSON (entre guillemets),
    pour neutraliser le Markdown (bold/italique/fences) qui casse json.loads.
    """
    out = []
    in_str = False
    esc = False
    for ch in s:
        if in_str:
            out.append(ch)
            if ch == '"' and not esc:
                in_str = False
            esc = (ch == '\\' and not esc)
        else:
            if ch == '"':
                in_str = True
                out.append(ch)
                esc = False
            elif ch in ('*', '`'):
                # on ignore ces marqueurs Markdown en dehors des chaînes
                continue
            else:
                out.append(ch)
    return ''.join(out)

# ---------- Gestion catégories & contenu ----------
def get_category_for_json(doc: dict, json_path: Path, clusters_map: Optional[dict] = None) -> str:
    # 1) top-level explicite
    for k in ("category", "category_name", "type", "Type", "categorie", "Catégorie"):
        v = doc.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # 2) champ 'fields'/'Fields'
    fields = doc.get("Fields") or doc.get("fields") or {}
    if isinstance(fields, dict):
        for k in ("category", "category_name", "type", "Type"):
            v = fields.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    # 3) mapping externe optionnel (non exposé en CLI — laissé pour compat)
    if clusters_map:
        key = json_path.name
        v = clusters_map.get(key) or clusters_map.get(key.replace(".json", ""))
        if isinstance(v, str) and v.strip():
            return v.strip()
    return "Inconnue"

def extract_text_content(doc: Any) -> str:
    # Cherche des champs typiques, sinon stringify global
    for k in ("content","Content","resume","Résumé","text","Texte","full_text","ocr","OCR"):
        v = doc.get(k) if isinstance(doc, dict) else None
        if isinstance(v, str) and v.strip():
            return v
    # Dernier recours : conversion compacte
    try:
        return json.dumps(doc, ensure_ascii=False)
    except Exception:
        return str(doc)

def _truncate(s: str, n: int) -> str:
    s = (s or "").replace("\u0000", " ").strip()
    return s if len(s) <= n else s[:n] + "…"

# ---------- Catalogue ----------
def catalog_path(base_out: Path) -> Path:
    return base_out / "fields_catalog.json"

def load_catalog(path: Path) -> Dict[str, List[dict]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def save_catalog(path: Path, catalog: Dict[str, List[dict]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")

def known_field_names(catalog: Dict[str, List[dict]]) -> List[str]:
    names = set()
    for schema in (catalog or {}).values():
        for f in schema or []:
            if isinstance(f, dict) and f.get("name"):
                names.add(str(f["name"]))
    return sorted(names)

# ---------- Client LLM ----------
def mistral_client(api_key: Optional[str]) -> Any:
    if Mistral is None:
        raise RuntimeError("Le paquet 'mistralai' n'est pas installé. pip install mistralai")
    key = api_key or os.environ.get("MISTRAL_API_KEY")
    if not key:
        raise RuntimeError("Fournissez une clé API Mistral via $MISTRAL_API_KEY ou --api-key")
    return Mistral(api_key=key)

def parse_json_object(text: str) -> Any:
    """
    Extraction robuste d'un objet/array JSON depuis `text`.
    Gère fences ```json, guillemets typographiques, et supprime le Markdown
    (**, *, `) en dehors des chaînes JSON.
    """
    # 0) normalise guillemets typographiques
    text = (text or "").replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'")

    # 1) récupère candidats dans des blocs ```json ... ```
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

    # 2) ajoute versions équilibrées {...} ou [...]
    more = []
    for s in candidates:
        for oc, cc in [('{', '}'), ('[', ']')]:
            b = _balanced_slice(s, oc, cc)
            if b and b not in candidates and b not in more:
                more.append(b)
    candidates += more

    # 3) tente json.loads après nettoyage Markdown hors chaînes
    for cand in candidates:
        cand_clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", cand.strip(), flags=re.IGNORECASE | re.DOTALL)
        cand_clean = _strip_md_outside_strings(cand_clean)
        try:
            data = json.loads(cand_clean)
            # normalisations fréquentes
            if isinstance(data, dict):
                if "fields" in data and isinstance(data["fields"], list):
                    return data["fields"]
                # dict mapping name -> {type, description} ou string
                if all(isinstance(k, str) for k in data.keys()) and all(isinstance(v, (str, dict)) for v in data.values()):
                    lst = []
                    for k, v in data.items():
                        if isinstance(v, dict):
                            lst.append({
                                "name": k,
                                "description": str(v.get("description","")).strip(),
                                "type": str(v.get("type","string")).strip().lower() or "string",
                            })
                        else:
                            lst.append({"name": k, "description": "", "type": "string"})
                    return lst
                # un seul objet de champ
                keyset = {kk.lower() for kk in data.keys()}
                if "name" in keyset:
                    return [data]
            return data
        except Exception:
            continue

    raise ValueError("Réponse LLM non JSON.")


LANG_MAP = {
    "fr": "français",
    "en": "anglais",
    "es": "espagnol",
    "it": "italien",
    "de": "allemand",
}

lang_code = (os.getenv("APP_LANG", "fr") or "fr").lower()
lang_name = LANG_MAP.get(lang_code, "français")





# ---------- Prompts ----------  TUNING
def propose_fields(category: str, example_content: str, extra: Optional[str], *, client: Any, model: str, max_chars: int = 4000, reuse_names: List[str] = None) -> List[dict]:
    reuse_names = reuse_names or []
    sys_msg = (
    f"N’écris que en {lang_name}. Si tu utilises une autre langue, corrige-toi immédiatement."
    "Tu es expert en extraction (français)."
    " Objectif: proposer ~8 NOMS DE CHAMPS pertinents pour cette catégorie."
    "Prends toujours le champs date_doc"
    " Réponds en JSON PUR, SANS Markdown."
    " Format minimal accepté:"
    "  - Soit un tableau de chaînes: [\"champ1\",\"champ2\", ...]"
    "  - Soit un tableau d'objets {\"name\":\"...\"}"
    " N’inclus NI description NI type."
)

    user = {
        "category": category,
        "example_content": _truncate(example_content, max_chars),
        "reuse_field_names": reuse_names[:100],
        "guidelines": (
            "Retourne UNIQUEMENT un tableau JSON de noms de champs "
            "ou d’objets {\"name\":\"...\"}. Aucun autre texte."
        ),
    }

    if extra:
        user["extra_instructions"] = str(extra)

    resp = client.chat.complete(
        model=model,
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
        temperature=0.0,   # <-- avant: 0.2
        max_tokens=2048,
    )
    txt = resp.choices[0].message.content if resp and resp.choices else ""
    data = parse_json_object(txt)


    # Normalisation légère
    out: List[dict] = []
    if isinstance(data, list):
        for it in data:
            if isinstance(it, str):
                nm = it.strip()
                if nm:
                    out.append({"name": nm})
            elif isinstance(it, dict):
                nm = str(it.get("name","")).strip()
                if nm:
                    out.append({"name": nm})
    # fallback ultra-min
    if not out:
        out = [{"name": "titre"}, {"name": "resume"}]
    return out[:16]


# ---------- RUN ----------
def run(
    src: Path,
    out_dst: Optional[Path],
    model: str,
    api_key: Optional[str],
    choose_fields_extra: Optional[str],
    extraction_extra: Optional[str],  # ignoré ici, gardé pour compat CLI
    seed: int,
):
    random.seed(seed)
    client = mistral_client(api_key)

    base_for_catalog = out_dst if out_dst else src
    cat_path = catalog_path(base_for_catalog)
    catalog = load_catalog(cat_path)
    reuse_names = known_field_names(catalog)

    files = iter_json_files(src)
    seen_categories = set(catalog.keys())

    pbar = tqdm(files, desc="Choix de champs par catégorie")
    for json_path in pbar:
        try:
            doc = read_json(json_path)
        except Exception as e:
            pbar.write(f"[WARN] Lecture: {json_path}: {e}")
            continue

        cat = get_category_for_json(doc, json_path)
        pbar.set_postfix_str((cat or "").replace("\n", " ")[:40])

        if cat in seen_categories:
            continue  # déjà dans le catalogue

        # premier exemple de la catégorie : choisir les champs
        content = extract_text_content(doc)
        schema = propose_fields(cat, content, choose_fields_extra, client=client, model=model, reuse_names=reuse_names)
        catalog[cat] = schema
        seen_categories.add(cat)
        reuse_names = known_field_names(catalog)  # enrichit la base de réutilisation

        # Sauvegarde immédiate pour résilience
        save_catalog(cat_path, catalog)

    pbar.close()
    # Sauvegarde finale (au cas où)
    save_catalog(cat_path, catalog)

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Choix de champs par catégorie (catalogue)")
    ap.add_argument("--src", required=True, type=Path, help="Dossier de JSON")
    ap.add_argument("--out-dst", type=Path, default=None, help="Dossier de sortie (catalogue). Défaut: --src")
    ap.add_argument("--model", default="mistral-large-latest", help="Modèle LLM")
    ap.add_argument("--api-key", default=None, help="Clé API Mistral (sinon $MISTRAL_API_KEY)")
    ap.add_argument("--choose-fields-extra", default=None, help="Phrase supplémentaire pour le prompt de CHOIX des champs")
    ap.add_argument("--extraction-extra", default=None, help="(ignoré ici — compat) Phrase supplémentaire pour le prompt d'EXTRACTION")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    run(
        src=args.src,
        out_dst=args.out_dst,
        model=args.model,
        api_key=args.api_key,
        choose_fields_extra=args.choose_fields_extra,
        extraction_extra=args.extraction_extra,
        seed=args.seed,
    )

if __name__ == "__main__":
    main()
