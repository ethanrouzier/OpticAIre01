
import os
import uuid
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple
from flask import abort
import pathlib, sys
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, render_template, send_from_directory
import pandas as pd
import csv, json
from urllib.parse import quote
import re
from pathlib import Path
from datetime import datetime
import glob
from indicators import compute_indicators_from_excel

from flask import g, request
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
# --- Répertoires ---
BASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = BASE_DIR / "runs"
UPLOADS_DIR = RUNS_DIR / "uploads"
STATIC_DIR = BASE_DIR / "static"
RUNS_DIR.mkdir(exist_ok=True, parents=True)
UPLOADS_DIR.mkdir(exist_ok=True, parents=True)
STATIC_DIR.mkdir(exist_ok=True, parents=True)

from pathlib import Path
import os

# --- Dossiers & chemins canoniques ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Fichier catalogue unique, utilisé par les scripts "choose" et "extract"
CATALOG_PATH = DATA_DIR / "fields_catalog.json"

# Excel nettoyé que ta page "/" lit pour la frise
EXCEL_CLEANED_PATH = DATA_DIR / "cleaned.xlsx"

# Expose le chemin du catalogue aux scripts appelés ailleurs
os.environ.setdefault("CATALOG_PATH", str(CATALOG_PATH))
# --- Localisation des scripts (PJ) ---
SCRIPTS = {
    "extract_content": BASE_DIR / "ExtractionContent.py",
    "cluster":         BASE_DIR / "cluster_jsons_with_llm.py",
    "choose_fields":   BASE_DIR / "choose_fields_catalog.py",
    "extract_fields":  BASE_DIR / "extract_fields_by_category.py",
    "json_to_excel":   BASE_DIR / "json_to_excel.py",
    "svc_predict":     BASE_DIR / "SVCPREDICTTYPE.py",
}

# Vérification présence scripts
for key, path in SCRIPTS.items():
    if not path.exists():
        print(f"⚠️  Script manquant: {key} -> {path}")

# --- Flask ---
app = Flask(__name__, template_folder=str(BASE_DIR), static_folder=str(STATIC_DIR))

SUPPORTED_LANGS = {'fr', 'en', 'es', 'it', 'de'}

# === [ADD] Constantes/chemins ===
JOBS_ROOT = Path(app.config.get("JOBS_ROOT", "runs"))
ALLOWED_XLSX = {".xlsx"}


# app.py
from flask import Flask, request, jsonify
from pathlib import Path
import json

# app.py — AJOUTS (imports)
from flask import send_from_directory
from werkzeug.utils import secure_filename
import glob, json, time

# NEW: import du builder PDF
from tumeur_fiche_pdf import generate_for_job



def _job_work_dir(job_id): return os.path.join("runs", job_id, "work")

def _set_last_excel(job_id, xlsx_path):
    # petit mémo pour retrouver l'Excel plus tard
    work = _job_work_dir(job_id)
    os.makedirs(work, exist_ok=True)
    with open(os.path.join(work, "last_excel.txt"), "w") as f:
        f.write(xlsx_path.strip())

def _get_last_excel(job_id):
    p = os.path.join(_job_work_dir(job_id), "last_excel.txt")
    if os.path.exists(p):
        return open(p).read().strip()
    # fallback: prends le plus récent .xlsx du work
    cand = sorted(glob.glob(os.path.join(_job_work_dir(job_id), "*.xlsx")), key=os.path.getmtime, reverse=True)
    return cand[0] if cand else None

def _fiche_paths(job_id):
    pdf_path = os.path.join(_job_work_dir(job_id), "fiche_tumeur.pdf")
    pdf_url  = f"/download/{job_id}/work/fiche_tumeur.pdf"
    return pdf_path, pdf_url

# ---- ROUTES FICHE ----


def _find_latest_excel_for_job(job_id: str) -> str:
    """
    Convention: runs/<job_id>/work/export.xlsx
    Fallback: dernier .xlsx sous runs/<job_id>/work
    """
    base = Path("runs") / job_id / "work"
    fixed = base / "export.xlsx"
    if fixed.exists(): return str(fixed)
    xs = sorted(base.glob("*.xlsx"))
    if xs: return str(xs[-1])
    raise FileNotFoundError(f"Aucun Excel trouvé pour le job {job_id} ({base})")

def build_cmd_svc_predict(src_json_dir: Path, model_path: Path, opts: Dict[str, Any]) -> List[str]:
    # Classifie chaque JSON avec un pipeline joblib et écrit 'category'
    cmd = [
        "python3", str(SCRIPTS["svc_predict"]),
        str(model_path), str(src_json_dir),
        "--recursive",
        "--content-key", str(opts.get("content_key", "content")),
        "--category-key", str(opts.get("category_key", "category")),
    ]
    if opts.get("dry_run"): cmd += ["--dry-run"]
    if opts.get("backup"):  cmd += ["--backup"]
    return cmd


def _read_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def _norm_date(s):
    if not s:
        return None
    s = str(s).strip()
    # si ISO "YYYY-MM-DD..." on coupe à 10
    if len(s) >= 10 and s[4] == '-' and s[7] == '-':
        return s[:10]
    return s




def _new_job_id() -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{ts}-{uuid.uuid4().hex[:6]}"

def _job_root(job_id: str) -> Path:
    root = RUNS_DIR / job_id
    root.mkdir(parents=True, exist_ok=True)
    (root / "work").mkdir(exist_ok=True)
    return root

def _job_paths(job_id: str):
    root = _job_root(job_id)
    work = root / "work"
    content_json = work / "json_from_files"
    cluster_out  = work / "clusters"
    catalog_dir  = work / "catalog"
    extracted    = work / "json_extracted"
    excel_path   = work / "export.xlsx"
    return root, work, content_json, cluster_out, catalog_dir, extracted, excel_path

def _catalog_file_for(job_id: str) -> tuple[Path, dict | list]:
    """Retourne (path, data_raw) pour le catalog. Tolère plusieurs noms / structures."""
    root, work, content_json, cluster_out, catalog_dir, *_ = _job_paths(job_id)
    candidates = [
        catalog_dir / "fields_catalog.json",
        catalog_dir / "catalog.json",
        catalog_dir / "field_catalog.json",
    ]
    path = next((p for p in candidates if p.exists()), candidates[0])
    if not path.exists():
        # Catalogue encore non créé
        return path, {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        raw = {}
    return path, raw

def _normalize_catalog(raw: dict | list) -> dict[str, list[str]]:
    """
    Normalise en { category: [field, ...] } à partir de plusieurs formes possibles:
      - { "CatA": ["a","b"], "CatB": [...] }
      - { "CatA": {"fields": [...]}, ... }
      - { "categories": [ {"name": "...", "fields": [...]}, ... ] }
      - [ {"category": "...", "fields": [...]}, ... ]
    """
    norm: dict[str, list[str]] = {}
    if isinstance(raw, dict):
        if "categories" in raw and isinstance(raw["categories"], list):
            for it in raw["categories"]:
                if not isinstance(it, dict): continue
                name = it.get("name") or it.get("category") or ""
                fields = it.get("fields") or []
                if name: norm[name] = [str(x).strip() for x in fields if str(x).strip()]
            return norm
        # mapping direct
        for k, v in raw.items():
            if isinstance(v, dict) and "fields" in v:
                fields = v.get("fields") or []
            else:
                fields = v or []
            norm[str(k)] = [str(x).strip() for x in fields if str(x).strip()]
        return norm
    elif isinstance(raw, list):
        for it in raw:
            if not isinstance(it, dict): continue
            name = it.get("name") or it.get("category") or ""
            fields = it.get("fields") or []
            if name: norm[name] = [str(x).strip() for x in fields if str(x).strip()]
        return norm
    return {}

def _write_catalog_preserving_shape(path: Path, raw: dict | list, edited: dict[str, list[str]]):
    """
    Met à jour le catalog en conservant au mieux sa forme d'origine.
    - 'edited' est {cat: [fields]} déjà nettoyé/uniq.
    - On n’enlève pas les catégories existantes (même si vides), on remplace juste leurs 'fields'.
    - Si une catégorie nouvelle apparaît, on l’ajoute en respectant le shape original.
    """
    def dedup_keep_order(xs: list[str]) -> list[str]:
        seen = set(); out=[]
        for x in xs:
            x = (x or "").strip()
            if not x or x in seen: 
                continue
            seen.add(x); out.append(x)
        return out

    edited = {k: dedup_keep_order(v or []) for k, v in (edited or {}).items()}

    # ---- Forme 1: {"categories":[{"name": "...", "fields":[...] , ...}, ...]}
    if isinstance(raw, dict) and isinstance(raw.get("categories"), list):
        by_name = {}
        for it in raw["categories"]:
            if isinstance(it, dict):
                key = it.get("name") or it.get("category")
                if key: by_name[key] = it
        # update existants / ajouter nouveaux
        for cat, fields in edited.items():
            if cat in by_name:
                by_name[cat]["fields"] = fields
            else:
                raw["categories"].append({"name": cat, "fields": fields})
        out = raw

    # ---- Forme 2: {"fields_catalog": {...}}  (mapping interne)
    elif isinstance(raw, dict) and isinstance(raw.get("fields_catalog"), dict):
        fc = raw["fields_catalog"]
        # shape sous-jacent: dict de lists ou dicts ?
        dict_shape = any(isinstance(v, dict) for v in fc.values()) if fc else False
        for cat, fields in edited.items():
            if cat in fc:
                if isinstance(fc[cat], dict):
                    fc[cat]["fields"] = fields
                else:
                    fc[cat] = fields
            else:
                fc[cat] = {"fields": fields} if dict_shape else fields
        out = raw

    # ---- Forme 3: dict direct par catégorie (valeur = list OU dict {"fields":...})
    elif isinstance(raw, dict) and raw:
        dict_shape = any(isinstance(v, dict) for v in raw.values())
        for cat, fields in edited.items():
            if cat in raw:
                if isinstance(raw[cat], dict):
                    raw[cat]["fields"] = fields
                else:
                    raw[cat] = fields
            else:
                raw[cat] = {"fields": fields} if dict_shape else fields
        out = raw

    # ---- Forme 4: liste d’objets [{"category"|"name":..., "fields":[...]}]
    elif isinstance(raw, list):
        by_name = {}
        for it in raw:
            if isinstance(it, dict):
                key = it.get("name") or it.get("category")
                if key: by_name[key] = it
        for cat, fields in edited.items():
            if cat in by_name:
                by_name[cat]["fields"] = fields
            else:
                raw.append({"category": cat, "fields": fields})
        out = raw

    else:
        # inconnu / vide -> mapping simple {cat:[...]}
        out = {cat: fields for cat, fields in edited.items()}

    # sauvegarde + backup
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        backup = path.with_suffix(path.suffix + f".bak-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def _first_png_in(dirpath: Path) -> Path | None:
    if not dirpath.exists():
        return None
    for ext in ("*.png", "*.svg", "*.webp", "*.jpg", "*.jpeg"):
        pngs = list(dirpath.glob(ext))
        if pngs:
            return pngs[0]
    return None

def _load_assignments(cluster_out: Path, content_json: Path) -> tuple[list[dict], dict]:
    """
    Retourne (docs, clusters) où:
      docs = [{json_path, filename, cluster_id, category}]
      clusters = { cluster_id: {"name": str, "count": int} }
    Tolère plusieurs schémas de CSV.
    """
    csv_path = cluster_out / "cluster_assignments.csv"
    if not csv_path.exists():
        return [], {}

    def guess_field(row: dict, candidates: list[str], default=""):
        for c in candidates:
            if c in row and row[c] != "":
                return row[c]
        return default

    docs: list[dict] = []
    clusters: dict = {}

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # Normalisation
            filename   = guess_field(r, ["json_path","path","filepath","file","filename","document"])
            fn = Path(filename).name
            # cluster id (string pour stabilité)
            cluster_id = str(guess_field(r, ["cluster_id","cluster","cluster_idx","label_id","cluster_index","k"]))
            # nom proposé
            name       = guess_field(r, ["proposed_name","category","cluster_name","label","name"], default="")
            # JSON absolu
            json_path = (content_json / fn)
            if not json_path.exists() and filename and Path(filename).exists():
                json_path = Path(filename)
            docs.append({
                "json_path": str(json_path),
                "filename": fn,
                "cluster_id": cluster_id if cluster_id else "",
                "category": name
            })
            if cluster_id:
                clusters.setdefault(cluster_id, {"name": name or f"cluster_{cluster_id}", "count": 0})
                clusters[cluster_id]["count"] += 1

    return docs, clusters

def _write_json_category(json_path: Path, category_value: str):
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    data["category"] = category_value
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return True

def _capture(cmd: list[str], env: dict[str, str] | None = None, cwd: Path | None = None):

    base_env = os.environ.copy()
    # Injecter la langue courante si on est dans le contexte Flask
    try:
        if getattr(g, 'lang', None):
            base_env['APP_LANG'] = g.lang
    except Exception:
        pass
    if env:
        base_env.update(env)

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=base_env,
        cwd=str(cwd) if cwd else None,
    )

    return proc.returncode, proc.stdout

def _safe_join(base: Path, *parts: str) -> Path:
    p = (base.joinpath(*parts)).resolve()
    if not str(p).startswith(str(base.resolve())):
        raise ValueError("Chemin non autorisé")
    return p

# ---------------------- PAGES & STATICS ----------------------

@app.get("/")
def index():
    # Valeurs par défaut pour petit confort
    defaults = {
        "model": "mistral-medium-2508",
        "sheet": "export",
        "max_cell_chars": 32760,
        "min_k": 3,
        "max_k": 12,
        "seed": 42,
        "embedder": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "naming_sample_size": 5,
    }
    return render_template("index.html", defaults=defaults)


@app.get("/download/<job_id>/<path:filename>")
def download(job_id: str, filename: str):
    job_dir = _job_root(job_id)
    file_path = _safe_join(job_dir, filename)  # doit renvoyer un Path dans job_dir

    if not file_path.is_file():
        return abort(404)

    # PDF -> inline (aperçu)
    if file_path.suffix.lower() == ".pdf":
        resp = send_from_directory(
            directory=str(file_path.parent),
            path=file_path.name,
            as_attachment=False,          # ← clé : pas de téléchargement forcé
            conditional=True              # support Range/ETag
        )
        # on force "inline" au cas où un middleware remettrait "attachment"
        resp.headers["Content-Disposition"] = f'inline; filename="{file_path.name}"'
        return resp

    # Autres fichiers -> téléchargement
    return send_from_directory(
        directory=str(file_path.parent),
        path=file_path.name,
        as_attachment=True,
        conditional=True
    )

# ---------------------- SOURCES : PATH vs UPLOAD ----------------------
@app.before_request
def _bind_lang():
    # ordre de priorité: Header -> query -> JSON body
    lang = request.headers.get('X-App-Lang') or request.args.get('lang')
    if not lang and request.is_json:
        data = request.get_json(silent=True) or {}
        lang = data.get('lang')

    g.lang = lang if lang in SUPPORTED_LANGS else 'fr'

@app.post("/api/source/path")
def set_source_path():
    """
    Mode 1 : l'utilisateur fournit un chemin local lisible par le serveur.
    """
    data = request.get_json(force=True)
    src_path = Path(data.get("path", "")).expanduser().resolve()
    if not src_path.exists() or not src_path.is_dir():
        return jsonify(ok=False, error="Chemin invalide ou introuvable."), 400

    job_id = _new_job_id()
    root = _job_root(job_id)
    # On ne copie pas : on référence seulement
    (root / "source_mode.txt").write_text("local_path", encoding="utf-8")
    (root / "source_path.txt").write_text(str(src_path), encoding="utf-8")
    return jsonify(ok=True, job_id=job_id, src_path=str(src_path))

@app.post("/api/source/upload")
def upload_folder():
    """
    Mode 2 : l'utilisateur **téléverse** un dossier (via input webkitdirectory / drag&drop).
    On reconstruit l'arborescence dans runs/uploads/<job_id>/src
    """
    job_id = _new_job_id()
    job_root = _job_root(job_id)
    src_root = job_root / "src"
    src_root.mkdir(parents=True, exist_ok=True)

    files = request.files.getlist("files")
    if not files:
        return jsonify(ok=False, error="Aucun fichier reçu."), 400

    for f in files:
        # essaye de préserver webkitRelativePath si fourni
        rel = request.form.get(f"{f.name}.relativePath") or request.form.get("relativePath") or getattr(f, "filename", "")
        rel = rel or f.filename
        rel = rel.lstrip("/").replace("\\", "/")
        dst = _safe_join(src_root, rel)
        dst.parent.mkdir(parents=True, exist_ok=True)
        f.save(str(dst))

    (job_root / "source_mode.txt").write_text("uploaded", encoding="utf-8")
    (job_root / "source_path.txt").write_text(str(src_root), encoding="utf-8")
    return jsonify(ok=True, job_id=job_id, src_path=str(src_root))

# ---------------------- BUILD COMMANDS ----------------------

def build_cmd_extract_content(src: Path, out_dir: Path, opts: Dict[str, Any]) -> List[str]:
    return ["python3", str(SCRIPTS["extract_content"]), "--src", str(src), "--dst", str(out_dir)]

def build_cmd_cluster(src_json_dir: Path, out_dir: Path, opts: Dict[str, Any]) -> List[str]:
    cmd = [
        "python3", str(SCRIPTS["cluster"]),
        "--src", str(src_json_dir),
        "--out", str(out_dir),
        "--embedder", str(opts.get("embedder", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")),
        "--min-k", str(opts.get("min_k", 3)),
        "--max-k", str(opts.get("max_k", 12)),
        "--seed", str(opts.get("seed", 42)),
        "--model", str(opts.get("model", "mistral-large-latest")),
        "--naming-sample-size", str(opts.get("naming_sample_size", 5)),
    ]
    if opts.get("k"):
        cmd += ["--k", str(opts["k"])]
    if opts.get("api_key"):
        cmd += ["--api-key", str(opts["api_key"])]
    if opts.get("name_extra"):
        cmd += ["--name-extra", str(opts["name_extra"])]
    if not bool(opts.get("write_back", True)):
        cmd += ["--no-write-back"]
    return cmd

def build_cmd_choose_fields(src_json_dir: Path, out_dir: Path, opts: Dict[str, Any]) -> List[str]:
    cmd = [
        "python3", str(SCRIPTS["choose_fields"]),
        "--src", str(src_json_dir),
        "--model", str(opts.get("model", "mistral-large-latest")),
        "--seed", str(opts.get("seed", 42)),
    ]
    # où écrire le catalogue
    cmd += ["--out-dst", str(out_dir)]
    if opts.get("api_key"):
        cmd += ["--api-key", str(opts["api_key"])]
    if opts.get("choose_fields_extra"):
        cmd += ["--choose-fields-extra", str(opts["choose_fields_extra"])]
    # extraction-extra est ignoré par le script, mais option laissée pour compat
    if opts.get("extraction_extra"):
        cmd += ["--extraction-extra", str(opts["extraction_extra"])]
    return cmd


def build_cmd_json_to_excel(src_json_dir: Path, out_xlsx: Path, opts: Dict[str, Any]) -> List[str]:
    cmd = [
        "python3", str(SCRIPTS["json_to_excel"]),
        "--src", str(src_json_dir),
        "--out", str(out_xlsx),
        "--sheet", str(opts.get("sheet", "export")),
        "--max-cell-chars", str(opts.get("max_cell_chars", 32760)),
    ]
    return cmd
from typing import Optional

def build_cmd_extract_fields(src_json_dir: Path, out_dir: Path, opts: Dict[str, Any],
                             extra_context: Optional[str] = None) -> List[str]:  # ← default None

    cmd = [
        "python3", str(SCRIPTS["extract_fields"]),
        "--src", str(src_json_dir),
        "--model", str(opts.get("model", "mistral-large-latest")),
        "--seed", str(opts.get("seed", 42)),
        "--out-dst", str(out_dir),
    ]
    if opts.get("api_key"):
        cmd += ["--api-key", str(opts["api_key"])]
    if opts.get("extraction_extra"):
        cmd += ["--extraction-extra", str(opts["extraction_extra"])]

    # ⚠️ le contexte UI : on l’envoie INLINE (pas le chemin de fichier)
    if extra_context:
        cmd += ["--extraction-extra", extra_context]

    # (si tu tiens au fichier, alors ajoute un vrai --extra-file côté script)
    return cmd


def _env_for_mode(mode: str) -> dict:
    if mode == "dev":   return {"NCLUSTER": "3"}   # trace la PCA même pour très peu de docs
    if mode == "unsup": return {"NCLUSTER": "3"} 
    if mode == "unsup": return {"NCLUSTER": "50"} # un seuil un peu plus haut
    return {}
# ---------------------- RUN ONE STEP ----------------------




# ---- Catégories autorisées par langue ----
ALLOWED_BY_LANG = {
    "fr": [
        "Biologie", "RCP", "Prescription", "Hospitalisation", "Imagerie",
        "Histologie", "Séance de radiothérapie", "Endoscopie",
        "Télésurveillance", "Consultation", "Compte rendu opératoire","Questionnaire",
    ],
    "en": [
        "Laboratory", "Tumor Board", "Prescription", "Hospitalization", "Imaging",
        "Histology", "Radiotherapy session", "Endoscopy",
        "Remote monitoring", "Consultation", "Operative report",
    ],
    "es": [
        "Laboratorio", "Comité de tumores", "Prescripción", "Hospitalización", "Imagen",
        "Histología", "Sesión de radioterapia", "Endoscopia",
        "Telemonitorización", "Consulta", "Informe operatorio",
    ],
    "it": [
        "Laboratorio", "Tumor board", "Prescrizione", "Ricovero", "Imaging",
        "Istologia", "Seduta di radioterapia", "Endoscopia",
        "Telemonitoraggio", "Visita", "Referto operatorio",
    ],
    "de": [
        "Labor", "Tumorboard", "Verordnung", "Krankenhausaufenthalt", "Bildgebung",
        "Histologie", "Strahlentherapie-Sitzung", "Endoskopie",
        "Telemonitoring", "Konsultation", "OP-Bericht",
    ],
}

# ---- Template du message d’instruction par langue ----
NAME_EXTRA_TPL = {
    "fr": ("Tu DOIS nommer chaque cluster UNIQUEMENT avec l’un des noms de cette liste stricte : {list}. "
           "Réponds exactement par un seul nom pris dans la liste, sans variante ni ajout."),
    "en": ("You MUST name each cluster using ONLY one of these exact labels: {list}. "
           "Answer with exactly one label from the list, no variants or additions."),
    "es": ("Debes nombrar cada clúster ÚNICAMENTE con uno de estos nombres exactos: {list}. "
           "Responde con un solo nombre de la lista, sin variantes ni añadidos."),
    "it": ("Devi nominare ogni cluster SOLO con una di queste etichette esatte: {list}. "
           "Rispondi con esattamente un’etichetta della lista, senza varianti o aggiunte."),
    "de": ("Du MUSST jeden Cluster NUR mit einem dieser exakten Labels benennen: {list}. "
           "Antworte mit genau einem Label aus der Liste, ohne Varianten oder Zusätze."),
}

@app.post("/api/run_step")
def run_step():
    """
    Exécute un step unique en mode supervisé.
    Inputs JSON:
      - job_id
      - step: one of ["extract_content","cluster","choose_fields","extract_fields","json_to_excel"]
      - src_mode: "local_path"|"uploaded"
      - src_path: chemin d'entrée (selon mode)
      - opts: dict d'options pour le step
    """
    data = request.get_json(force=True)
    mode = (data.get("mode") or "").strip().lower()
    job_id: str = data["job_id"]
    step: str = data["step"]
    src_path = Path(data["src_path"])
    opts: Dict[str, Any] = data.get("opts", {}) or {}
    # Fallback: si la clé UI est vide/absente, prends l'env MISTRAL_API_KEY
    if not (opts.get("api_key") or "").strip():
        env_key = os.getenv("MISTRAL_API_KEY", "").strip()
        if env_key:
            opts["api_key"] = env_key
        else:
            opts.pop("api_key", None)

    root = _job_root(job_id)
    work = root / "work"
    logs_dir = root / "logs"
    logs_dir.mkdir(exist_ok=True)

    # Chaînage minimal des dossiers intermédiaires
    #  - content_json -> cluster_out -> (catalog) -> extracted_json -> excel
    content_json = work / "json_from_files"
    cluster_out  = work / "clusters"
    catalog_dir  = work / "catalog"
    extracted    = work / "json_extracted"
    excel_path   = work / "export.xlsx"

    cmd: List[str] = []
    artifact: str | None = None

    if step == "extract_content":
        cmd = build_cmd_extract_content(src_path, content_json, opts)
        artifact = str(content_json)
    elif step == "cluster":
        # Input = JSON (content). Si l'utilisateur a fourni directement des JSON, cluster marchera aussi.
        src_for_cluster = content_json if content_json.exists() else src_path
        cmd = build_cmd_cluster(src_for_cluster, cluster_out, opts)
        artifact = str(cluster_out / "cluster_assignments.csv")
    elif step == "choose_fields":
        # src doit pointer vers les JSON (avec category). On prend cluster_out écrit en place, sinon content_json
        src_for_choose = content_json if content_json.exists() else src_path
        cmd = build_cmd_choose_fields(src_for_choose, catalog_dir, opts)
        artifact = str(catalog_dir / "fields_catalog.json")
    elif step == "extract_fields":
        src_for_extract = content_json if content_json.exists() else src_path
        # passer le chemin du catalogue
        catalog_json = catalog_dir / "fields_catalog.json"
        if catalog_json.exists():
            opts["catalog"] = str(catalog_json)
        # utile si tu veux que build_cmd ait le job_id pour reconstruire le chemin:
        opts["job_id"] = job_id

        cmd = build_cmd_extract_fields(src_for_extract, extracted, opts)
        artifact = str(extracted)

    elif step == "json_to_excel":
        # Par défaut, on exporte ce qui se trouve dans "extracted" s'il existe, sinon src_path
        src_for_excel = extracted if extracted.exists() else src_path
        cmd = build_cmd_json_to_excel(src_for_excel, excel_path, opts)
        artifact = str(excel_path)
    

    else:
        return jsonify(ok=False, error=f"Step inconnu: {step}"), 400

    env = os.environ.copy()
    env.update(_env_for_mode(mode))
    code, out = _capture(cmd, env=env)
    (logs_dir / f"{step}.log").write_text(out, encoding="utf-8")

    return jsonify(
        ok=(code == 0),
        returncode=code,
        log=out,
        artifact=artifact,
        job_id=job_id
    )

# ---------------------- RUN FULL PIPELINE (UNSUPERVISED) ----------------------

@app.post("/api/run_all")

def run_all():
    """
    Exécute la chaîne complète:
      1) Extraction contenu -> JSON
      2) Clustering + category
      3) Choix de champs -> fields_catalog.json
      4) Extraction de champs -> JSON enrichis
      5) Export Excel
    Inputs JSON:
      - job_id
      - src_path
      - mode: "medical" | "unsup" | "dev" (optionnel, sinon auto)
      - opts: { shared: {model, api_key, seed...}, steps: { per-step overrides } }
    """
    data = request.get_json(force=True)
    job_id: str = data["job_id"]
    src_path = Path(data["src_path"])
    opts_all: Dict[str, Any] = data.get("opts", {}) or {}
    mode = (data.get("mode") or opts_all.get("mode") or "").strip().lower()
    extra   = opts_all.get("extra") or {}
    fields_context = (extra.get("fields_context") or "").strip() or None

    # [ADD] récupérer le base_excel_path (priorité à celui uploadé pour ce job)
    # [PATCH] Chemin de base.xlsx dans CE run
    base_excel_path = opts_all.get("base_excel_path")
    if not base_excel_path:
        p = _job_root(job_id) / "base.xlsx"  # ← runs/<job_id>/base.xlsx
        base_excel_path = str(p) if p.exists() else None

    root = _job_root(job_id)
    work = root / "work"
    logs_dir = root / "logs"
    logs_dir.mkdir(exist_ok=True)
    work.mkdir(exist_ok=True)

    shared = opts_all.get("shared", {}) or {}
    # Fallback: si la clé UI est vide/absente, prends l'env MISTRAL_API_KEY
    if not (shared.get("api_key") or "").strip():
        env_key = os.getenv("MISTRAL_API_KEY", "").strip()
        if env_key:
            shared["api_key"] = env_key
        else:
            shared.pop("api_key", None)  # évite d'envoyer --api-key ""

    steps_opts = opts_all.get("steps", {}) or {}

    # Dossiers de travail
    content_json = work / "json_from_files"
    cluster_out  = work / "clusters"
    catalog_dir  = work / "catalog"
    extracted    = work / "json_extracted"
    excel_path   = work / "export.xlsx"

    def run_and_log(stepname: str, cmd: list[str], env_overrides: dict | None = None) -> tuple[int, str]:
        env = os.environ.copy()
        # ⚠️ N'applique les overrides QUE pour le clustering (et seulement NCLUSTER)
        if stepname == "cluster" and env_overrides:
            if "NCLUSTER" in env_overrides:
                env["NCLUSTER"] = str(env_overrides["NCLUSTER"])
        code, out = _capture(cmd, env=env)
        (logs_dir / f"{stepname}.log").write_text(out, encoding="utf-8")
        return code, out

    # Détection "mode médical"
    medical_ready = (BASE_DIR / "fields_catalog.json").exists()
    if mode == "medical":
        medical_ready = True
    elif mode == "unsup":
        medical_ready = False
    # sinon, on garde l'auto-détection
    # 1) extract_content (toujours d'abord)
    code, out = run_and_log(
        "extract_content",
        build_cmd_extract_content(src_path, content_json, steps_opts.get("extract_content", {}) | shared),
        env_overrides=_env_for_mode(mode)
    )
    # 1) extract_content (commun)
    if code != 0:
        return jsonify(ok=False, where="extract_content", log=out, job_id=job_id), 500

    if medical_ready:
        # ============ MODE MÉDICAL ============

        # langue actuelle (définie par @app.before_request)
        lang = getattr(g, "lang", "fr")

        # 1) liste autorisée selon la langue (fallback fr)
        allowed = ALLOWED_BY_LANG.get(lang, ALLOWED_BY_LANG["fr"])

        # 2) phrase d’instruction localisée
        name_extra = NAME_EXTRA_TPL.get(lang, NAME_EXTRA_TPL["fr"]).format(list=", ".join(allowed))

        cluster_opts = (steps_opts.get("cluster", {}) | shared) | {
            "name_extra": name_extra,
            "write_back": True,
            # "k": len(allowed),  # optionnel si tu veux forcer k
        }

        code, out = run_and_log(
        "cluster",
        build_cmd_cluster(content_json, cluster_out, cluster_opts),
        env_overrides=_env_for_mode(mode)
        )
        if code != 0:
            return jsonify(ok=False, where="cluster", log=out, job_id=job_id), 500

        # 3) Catalogue: copie le fields_catalog.json racine dans le dossier de travail
        catalog_dir.mkdir(parents=True, exist_ok=True)
        # langue courante (définie par @app.before_request)
        lang = getattr(g, "lang", "fr")

        # On cherche d'abord un catalogue spécifique à la langue, puis le générique
        src_candidates = [
            BASE_DIR / f"fields_catalog.{lang}.json",  # ex: fields_catalog.en.json
            BASE_DIR / "fields_catalog.json",          # fallback historique
            BASE_DIR / "fields_catalog.fr.json",       # fallback FR explicite
        ]

        src_cat = next((p for p in src_candidates if p.exists()), None)
        if not src_cat:
            return jsonify(
                ok=False, where="choose_fields",
                log=f"Aucun catalogue trouvé (cherché: {', '.join(str(p) for p in src_candidates)})",
                job_id=job_id
            ), 500

        # On conserve le nom attendu par le reste du pipeline dans le job
        dst_cat = catalog_dir / "fields_catalog.json"
        try:
            shutil.copy2(src_cat, dst_cat)
            app.logger.info("Catalogue utilisé (%s) copié vers %s", src_cat.name, dst_cat)
        except Exception as e:
            return jsonify(ok=False, where="choose_fields", log=f"Copie catalog échouée: {e}", job_id=job_id), 500
        # 4) Extraction de champs (on passe --catalog explicitement)
        code, out = run_and_log(
            "extract_fields",
            build_cmd_extract_fields(
                content_json,
                extracted,
                (steps_opts.get("extract_fields", {}) | shared) | {"catalog": src_cat},
                extra_context=fields_context,     
            )
        )
        if code != 0:
            return jsonify(ok=False, where="extract_fields", log=out, job_id=job_id), 500

    else:
        # ============ MODE NON SUPERVISÉ ============
        cluster_opts = (steps_opts.get("cluster", {}) | shared)
        code, out = run_and_log("cluster",
            build_cmd_cluster(content_json, cluster_out, cluster_opts),
            env_overrides=_env_for_mode(mode)
        )
        if code != 0:
            return jsonify(ok=False, where="cluster", log=out, job_id=job_id), 500

        code, out = run_and_log("choose_fields",
            build_cmd_choose_fields(content_json, catalog_dir, steps_opts.get("choose_fields", {}) | shared)
        )
        if code != 0:
            return jsonify(ok=False, where="choose_fields", log=out, job_id=job_id), 500

        # copie du catalog dans extracted (trace)
        extracted.mkdir(parents=True, exist_ok=True)
        src_cat = catalog_dir / "fields_catalog.json"
        dst_cat = extracted / "fields_catalog.json"
        if src_cat.exists():
            shutil.copy2(src_cat, dst_cat)

        code, out = run_and_log(
            "extract_fields",
            build_cmd_extract_fields(
                content_json,
                extracted,
                steps_opts.get("extract_fields", {}) | shared,
                extra_context=fields_context  # ← chaîne, pas un dict
            )
        )
        if code != 0:
            return jsonify(ok=False, where="extract_fields", log=out, job_id=job_id), 500

    # 5) json_to_excel
    code, out = run_and_log(
        "json_to_excel",
        build_cmd_json_to_excel(extracted, excel_path, steps_opts.get("json_to_excel", {}) | shared)
    )
    if code != 0:
        return jsonify(ok=False, where="json_to_excel", log=out, job_id=job_id), 500

    # mémorise le dernier excel pour le bouton "Régénérer" de la fiche

    # [PATCH] Fusionner EN PLACE runs/<job>/work/export.xlsx avec base.xlsx
    try:
        _combine_export_inplace_with_base(excel_path, base_excel_path)
    except Exception as e:
        app.logger.exception("combine_inplace_with_base failed: %s", e)
    # mémoriser l'excel pour "Régénérer"
    _set_last_excel(job_id, str(excel_path))
    # --- Split Excel par patient: une feuille par ID ---
    excels_by_id = []
    try:
        import pandas as pd

        df = pd.read_excel(excel_path)
        # On essaie plusieurs noms de colonnes possibles
        id_col = None
        for cand in ["id", "ID", "Id", "id_patient", "id_dossier", "patient_id"]:
            if cand in df.columns:
                id_col = cand
                break

        if id_col:
            by_dir = work / "exports_by_id"
            by_dir.mkdir(exist_ok=True)

            # Sanitize nom de fichier
            import re
            def safe(s: str) -> str:
                return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))[:80] or "unknown"

            for pid, sub in df.groupby(id_col, dropna=False):
                sid = safe(pid)
                outp = by_dir / f"export_{sid}.xlsx"
                sub.to_excel(outp, index=False)
                rel = outp.relative_to(root).as_posix()
                excels_by_id.append({
                    "id": str(pid),
                    "url": f"/download/{job_id}/{rel}",
                    "rows": int(len(sub)),
                })
        else:
            app.logger.info("Split tableur: colonne d'ID introuvable; aucun split réalisé.")
    except Exception as e:
        app.logger.exception("Split tableur par ID: échec: %s", e)

    # générer la fiche (ici je le fais seulement en mode médical; enlève la condition si tu veux toujours générer)
    if mode == "medical":
        try:
            app.logger.info("Generating tumor sheet for job %s from %s", job_id, excel_path)
            generate_for_job(job_id, str(excel_path))
        except Exception as e:
            app.logger.exception("Fiche: génération échouée: %s", e)


    # --- Réponse & génération fiche (médical uniquement) ---
    rel = excel_path.relative_to(root).as_posix()
    download_url = f"/download/{job_id}/{rel}"

    payload = {                       # <<< NEW: on crée payload AVANT de l'utiliser
        "ok": True,
        "job_id": job_id,
        "excel": download_url,
    }
    
    payload["excels_by_id"] = excels_by_id
    if medical_ready:                 # ne générer la fiche que pour le mode médical
        try:
            generate_for_job(job_id, str(excel_path))   # runs/<job>/work/fiche_tumeur.pdf
            _, pdf_url = _fiche_paths(job_id)           # "/download/<job>/work/fiche_tumeur.pdf"
            payload["fiche_url"] = pdf_url
        except Exception as e:
            payload["fiche_error"] = str(e)

    return jsonify(**payload)



@app.get("/api/cluster/summary")
def cluster_summary():
    job_id = request.args.get("job_id")
    if not job_id:
        return jsonify(ok=False, error="job_id requis"), 400
    root, work, content_json, cluster_out, *_ = _job_paths(job_id)
    if not cluster_out.exists():
        return jsonify(ok=False, error="Aucun répertoire de clustering pour ce job"), 404

    # PNG (ou image) du clustering
    img = _first_png_in(cluster_out)
    img_url = None
    if img:
        rel = img.relative_to(root)
        img_url = f"/download/{job_id}/{rel.as_posix()}"

    # CSV -> docs/clusters
    docs, clusters = _load_assignments(cluster_out, content_json)

    # Options de catégories disponibles (noms uniques)
    existing_names = sorted({c["name"] for c in clusters.values() if c.get("name")})

    return jsonify(ok=True, img_url=img_url, clusters=[
        {"id": cid, "name": c["name"], "count": c["count"]}
        for cid, c in sorted(clusters.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0])
    ], docs=docs, category_names=existing_names)

@app.post("/api/cluster/apply_names")
def cluster_apply_names():
    """
    Body:
      { job_id: str, names: { "<cluster_id>": "New Name", ... } }
    Applique le renommage cluster->nom et réécrit les JSON (champ 'category').
    Écrit aussi cluster_assignments_validated.csv
    """
    data = request.get_json(force=True)
    job_id = data.get("job_id")
    names: dict = data.get("names") or {}
    if not job_id:
        return jsonify(ok=False, error="job_id requis"), 400

    root, work, content_json, cluster_out, *_ = _job_paths(job_id)
    docs, clusters = _load_assignments(cluster_out, content_json)
    if not docs:
        return jsonify(ok=False, error="Aucun assignments CSV"), 400

    # Appliquer final_name par cluster_id
    updated_rows: list[dict] = []
    ok_count = 0
    for d in docs:
        cid = d["cluster_id"]
        final_name = names.get(cid, d["category"])
        d["final_category"] = final_name
        # write JSON
        if d.get("json_path") and final_name:
            if _write_json_category(Path(d["json_path"]), final_name):
                ok_count += 1
        updated_rows.append(d)

    # Écrire CSV de validation
    out_csv = cluster_out / "cluster_assignments_validated.csv"
    headers = ["filename","json_path","cluster_id","proposed_category","final_category"]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for d in updated_rows:
            w.writerow({
                "filename": d["filename"],
                "json_path": d["json_path"],
                "cluster_id": d["cluster_id"],
                "proposed_category": d["category"],
                "final_category": d.get("final_category") or d["category"]
            })

    return jsonify(ok=True, updated=ok_count, csv=f"/download/{job_id}/work/clusters/{out_csv.name}")

@app.post("/api/cluster/update_docs")
def cluster_update_docs():
    """
    Body:
      { job_id: str, updates: [ { "json_path": "...", "category": "..." }, ... ] }
    Met à jour le champ 'category' document par document.
    Réécrit/complète cluster_assignments_validated.csv.
    """
    data = request.get_json(force=True)
    job_id = data.get("job_id")
    updates: list[dict] = data.get("updates") or []
    if not job_id:
        return jsonify(ok=False, error="job_id requis"), 400

    root, work, content_json, cluster_out, *_ = _job_paths(job_id)
    ok_count = 0
    rows: list[dict] = []

    # Charge ce qu'on a déjà validé si présent
    validated_csv = cluster_out / "cluster_assignments_validated.csv"
    cache: dict[str, dict] = {}  # json_path -> row
    if validated_csv.exists():
        with validated_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                cache[r.get("json_path","")] = r

    for u in updates:
        p = Path(u.get("json_path",""))
        cat = u.get("category","")
        if not p.exists() or not cat:
            continue
        if _write_json_category(p, cat):
            ok_count += 1
            key = str(p)
            row = cache.get(key, {
                "filename": p.name,
                "json_path": key,
                "cluster_id": "",
                "proposed_category": "",
                "final_category": ""
            })
            row["final_category"] = cat
            cache[key] = row

    # Réécrire le CSV consolidé
    if cache:
        with validated_csv.open("w", encoding="utf-8", newline="") as f:
            headers = ["filename","json_path","cluster_id","proposed_category","final_category"]
            w = csv.DictWriter(f, fieldnames=headers)
            w.writeheader()
            for r in cache.values():
                w.writerow(r)

    return jsonify(ok=True, updated=ok_count, csv=f"/download/{job_id}/work/clusters/{validated_csv.name}")

@app.get("/api/catalog/summary")
def catalog_summary():
    job_id = request.args.get("job_id")
    if not job_id:
        return jsonify(ok=False, error="job_id requis"), 400
    path, raw = _catalog_file_for(job_id)
    norm = _normalize_catalog(raw)
    categories = [{"name": k, "fields": v} for k, v in sorted(norm.items())]
    return jsonify(ok=True, path=str(path), categories=categories)

@app.post("/api/catalog/update")
def catalog_update():
    data = request.get_json(force=True)
    job_id = data.get("job_id")
    cats = data.get("categories") or []
    if not job_id:
        return jsonify(ok=False, error="job_id requis"), 400

    # normaliser entrée -> dict {cat:[fields]}
    edited: dict[str, list[str]] = {}
    for c in cats:
        name = (c.get("name") or "").strip()
        if not name:
            continue
        fields = [str(x).strip() for x in (c.get("fields") or []) if str(x).strip()]
        edited[name] = fields

    path, raw = _catalog_file_for(job_id)
    out = _write_catalog_preserving_shape(path, raw, edited)

    # copie de sûreté à côté des JSON source (souvent attendu par l'extracteur)
    root, work, content_json, *_ = _job_paths(job_id)
    try:
        mirror = content_json / path.name  # ex: fields_catalog.json
        mirror.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

    # stats + URL de download
    def count_fields(_out):
        if isinstance(_out, dict) and "categories" in _out and isinstance(_out["categories"], list):
            return sum(len(it.get("fields") or []) for it in _out["categories"] if isinstance(it, dict))
        if isinstance(_out, dict):
            total = 0
            for v in _out.values():
                if isinstance(v, dict): total += len(v.get("fields") or [])
                elif isinstance(v, list): total += len(v)
            return total
        if isinstance(_out, list):
            return sum(len(it.get("fields") or []) for it in _out if isinstance(it, dict))
        return 0

    total = count_fields(out)
    saved_rel = path.relative_to(root)
    mirror_rel = (mirror.relative_to(root) if (content_json / path.name).exists() else None)
    return jsonify(
        ok=True,
        saved_path=str(path),
        saved_url=f"/download/{job_id}/{saved_rel.as_posix()}",
        mirror_path=(str(mirror) if mirror_rel else None),
        mirror_url=(f"/download/{job_id}/{mirror_rel.as_posix()}" if mirror_rel else None),
        total_fields=total
    )




def _norm_date(s):
    if not s:
        return None
    s = str(s).strip()
    # ISO YYYY-MM-DD... -> tronqué à 10
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        return s[:10]
    return s

def _extract_item_from_dict(d: dict, p: Path) -> dict | None:
    """
    Essaie d'en faire un 'item frise' si on reconnaît une structure de doc extrait.
    """
    # candidates pour les champs extraits
    fields = (d.get("fields") or d.get("extracted") or d.get("data") or
              (isinstance(d.get("result"), dict) and (d["result"].get("fields") or d["result"].get("extracted") or d["result"].get("data"))))
    if not isinstance(fields, dict):
        return None  # ce JSON ne ressemble pas à un résultat d'extraction par document

    # la date d'organisation = date_doc (prioritaire)
    date_doc = (fields.get("date_doc")
                or d.get("date_doc")
                or d.get("doc_date")
                or (isinstance(d.get("metadata"), dict) and d["metadata"].get("date_doc")))
    if isinstance(date_doc, dict):
        date_doc = date_doc.get("value") or date_doc.get("text")

    return {
        "filename": d.get("filename") or d.get("source_file") or p.name,
        "category": d.get("category") or (isinstance(d.get("meta"), dict) and d["meta"].get("category")),
        "date_doc": _norm_date(date_doc),
        "fields": fields,
        "json_path": str(p),
    }

@app.get("/api/timeline")
def api_timeline():
    job_id = request.args.get("job_id", "").strip()
    if not job_id:
        return jsonify(ok=False, error="missing job_id"), 400

    base = Path(f"runs/{job_id}")
    if not base.exists():
        return jsonify(ok=True, items=[])

    # Idéalement on ne scanne que les sorties d'extraction; mais si on ne sait pas le sous-dossier, on filtre par contenu.
    json_paths = list(base.rglob("*.json"))

    items = []
    for p in json_paths:
        try:
            js = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue

        if isinstance(js, dict):
            it = _extract_item_from_dict(js, p)
            if it: items.append(it)

        elif isinstance(js, list):
            # Certains fichiers sont des listes; on ne garde que les éléments qui ressemblent à des docs extraits
            for elem in js:
                if isinstance(elem, dict):
                    it = _extract_item_from_dict(elem, p)
                    if it: items.append(it)

        # Autres types ignorés

    # tri par date_doc puis nom
    items.sort(key=lambda x: ((x["date_doc"] or ""), x["filename"] or ""))
    return jsonify(ok=True, items=items)

@app.get("/api/timeline/fields")
def api_timeline_fields():
    """
    Retourne les champs 'live' pour un JSON précis (pour le tooltip).
    """
    job_id = request.args.get("job_id", "").strip()
    json_path = request.args.get("json_path", "").strip()
    if not job_id or not json_path:
        return jsonify(ok=False, error="missing job_id or json_path"), 400

    p = Path(json_path)
    if not p.exists():
        p2 = Path(f"runs/{job_id}") / json_path
        if p2.exists():
            p = p2
        else:
            return jsonify(ok=False, error="json not found"), 404

    try:
        js = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return jsonify(ok=False, error="invalid json"), 400

    fields = {}
    date_doc = None

    def take_from(d: dict):
        nonlocal fields, date_doc
        cand = (d.get("fields") or d.get("extracted") or d.get("data") or
                (isinstance(d.get("result"), dict) and (d["result"].get("fields") or d["result"].get("extracted") or d["result"].get("data"))))
        if isinstance(cand, dict):
            fields = cand
            dd = (fields.get("date_doc") or d.get("date_doc") or
                  (isinstance(d.get("metadata"), dict) and d["metadata"].get("date_doc")))
            if isinstance(dd, dict):
                dd = dd.get("value") or dd.get("text")
            date_doc = dd

    if isinstance(js, dict):
        take_from(js)
    elif isinstance(js, list):
        for it in js:
            if isinstance(it, dict):
                take_from(it)
                if fields:
                    break

    return jsonify(ok=True, fields=fields or {}, date_doc=_norm_date(date_doc))




RUNS = pathlib.Path("runs").absolute()
BASE_DIR = pathlib.Path(__file__).parent

def _job_dir(job_id: str) -> pathlib.Path:
    d = RUNS / job_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "work").mkdir(exist_ok=True)
    return d

def _latest_excel(job_id: str) -> str | None:
    base = _job_dir(job_id)
    candidates = glob.glob(str(base / "**/*.xlsx"), recursive=True)
    return max(candidates, key=os.path.getmtime) if candidates else None

@app.post("/api/ask_excel")
def ask_excel():
    js = request.get_json(force=True) or {}
    job_id   = (js.get("job_id") or "").strip()
    question = (js.get("question") or "").strip()
    excel    = (js.get("excel_path") or "").strip()

    shared   = ((js.get("opts") or {}).get("shared") or {}) or {}
    model    = (shared.get("model") or "mistral-medium-2508").strip()
    api_key  = (shared.get("api_key") or os.getenv("MISTRAL_API_KEY", "")).strip()
    max_methods   = int(shared.get("max_methods") or 1)
    depth_retries = int(shared.get("depth_retries") or 3)

    if not job_id:
        return jsonify(ok=False, error="job_id manquant")
    if not question:
        return jsonify(ok=False, error="question manquante")

    if not excel:
        excel = _latest_excel(job_id)
        if not excel:
            return jsonify(ok=False, error="Aucun Excel trouvé pour ce job (exécute d’abord l’étape 5).")

    # 1) SQL direct si la question commence par SELECT
    if question.lstrip().lower().startswith("select"):
        try:
            import pandas as pd, duckdb
            con = duckdb.connect()
            xls = pd.ExcelFile(excel)
            for sheet in xls.sheet_names:
                df = xls.parse(sheet_name=sheet)
                tbl = "sheet_" + "".join(ch if ch.isalnum() else "_" for ch in sheet.lower())
                con.register(tbl, df)
            df = con.execute(question).fetchdf()
            preview = df.to_csv(index=False)
            log = f"Tableur: {excel}\nSQL: {question}"
            return jsonify(ok=True, response=preview, log=log)
        except Exception as e:
            return jsonify(ok=False, error="Erreur SQL", log=str(e))

    # 2) Sinon : appeler ExtractionAutoSave5.py et LUI passer la question sur stdin
    env = os.environ.copy()
    if api_key:
        env["MISTRAL_API_KEY"] = api_key

    cmd = [
        sys.executable, "ExtractionAutoSave5.py",
        "--excel", excel,
        "--model", model,
        "--max-methods", str(max_methods),
        "--depth-retries", str(depth_retries),
    ]
    if api_key:
        cmd += ["--api-key", api_key]

    try:
        p = subprocess.run(
            cmd,
            input=question + "\n",   # <<< IMPORTANT : la question passe par stdin
            text=True,
            capture_output=True,
            cwd=str(BASE_DIR),
            env=env,
            check=False,
        )
        out = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")

        # Extraire la réponse entre les marqueurs imprimés par le script
        def between(s, a, b):
            i = s.find(a)
            if i == -1: return ""
            j = s.find(b, i + len(a))
            return s[i+len(a):j].strip() if j != -1 else ""

        final = between(out, "<<FINAL_RESPONSE_START>>", "<<FINAL_RESPONSE_END>>")
        ids   = between(out, "<<ID_START>>", "<<ID_END>>")
        if not final:
            final = (p.stdout or "").strip() or (p.stderr or "").strip()

        return jsonify(
            ok=(p.returncode == 0),
            response=final,
            ids=ids,
            log="Tableur: {}\nCMD: {}\n\n{}".format(excel, " ".join(cmd), out),
        )
    except FileNotFoundError:
        return jsonify(ok=False, error="ExtractionAutoSave5.py introuvable à côté de app.py")
    except Exception as e:
        return jsonify(ok=False, error="Exception pendant l'appel ExtractionAutoSave5.py", log=str(e))



def _resolve_job_dir(job_id: str) -> str:
    # ← prends ta fonction existante; sinon adapte
    return os.path.join(DATA_DIR, job_id)

def _find_patient_excel(job_id: str, patient_id: str | int | None) -> str | None:
    """
    Essaie d'abord runs/<job_id>/work/export_<patient_id>.xlsx,
    puis n'importe quel .xlsx du work/ qui contient l'id patient dans son nom.
    """
    if not patient_id:
        return None
    work = _job_root(str(job_id)) / "work"

    # 1) nom exact
    cand = work / f"export_{patient_id}.xlsx"
    if cand.exists():
        return str(cand)

    # 2) fallback: tout .xlsx contenant l'id
    hits = sorted(
        glob.glob(str(work / f"*{patient_id}*.xlsx")),
        key=os.path.getmtime,
        reverse=True
    )
    return hits[0] if hits else None

import os, glob

def _find_patient_excel(job_id: str, patient_id: str | None) -> str | None:
    """
    1) runs/<job_id>/work/exports_by_id/export_<patient_id>.xlsx (prioritaire)
    2) runs/<job_id>/work/exports_by_id/*<patient_id>*.xlsx
    3) runs/<job_id>/work/*<patient_id>*.xlsx (fallback)
    4) runs/<job_id>/work/**/*<patient_id>*.xlsx (fallback récursif)
    """
    if not patient_id:
        return None

    work = _job_root(str(job_id)) / "work"

    # 1) exact dans exports_by_id/
    exact = work / "exports_by_id" / f"export_{patient_id}.xlsx"
    if exact.exists():
        return str(exact)

    candidates = []
    # 2) exports_by_id/ contenant l'id
    candidates += glob.glob(str(work / "exports_by_id" / f"*{patient_id}*.xlsx"))
    # 3) racine de work/
    candidates += glob.glob(str(work / f"*{patient_id}*.xlsx"))
    # 4) récursif sous work/ (au cas où l’export serait rangé autrement)
    candidates += glob.glob(str(work / f"**/*{patient_id}*.xlsx"), recursive=True)

    if not candidates:
        return None

    # prend le plus récent
    return max(candidates, key=os.path.getmtime)

@app.post("/api/indicators")
def api_indicators():
    try:
        data = request.get_json(force=True) or {}
        job_id = data.get("job_id")
        patient_id = data.get("patient_id")          # ← NEW (reçu du front)
        first_contact_date = data.get("first_contact_date")

        if not job_id:
            return jsonify({"ok": False, "error": "job_id manquant"}), 400

        # priorité : tableur du patient → sinon export global
        xlsx = _find_patient_excel(job_id, patient_id) or _find_latest_excel_for_job(job_id)

        # (facultatif) log de debug
        try:
            print(f"[indicators] job={job_id} pid={patient_id} -> {os.path.basename(xlsx) if xlsx else 'None'}")
        except Exception:
            pass

        lang = getattr(g, "lang", "fr") if 'g' in globals() else "fr"
        res = compute_indicators_from_excel(xlsx, first_contact_date, lang=lang)

        return jsonify({
            "ok": True,
            "job_id": job_id,
            "items": res["items"],
            "points": res["points"],
            "log": res.get("log", "")
        })
    except FileNotFoundError as e:
        return jsonify({"ok": False, "error": str(e)})
    except Exception as e:
        return jsonify({"ok": False, "error": f"Erreur indicateurs: {e}"})


@app.get("/api/fiche")
def api_fiche_get():
    job_id = request.args.get("job_id","").strip()
    if not job_id:
        return jsonify(ok=False, error="job_id requis")
    pdf_path, pdf_url = _fiche_paths(job_id)
    if not os.path.exists(pdf_path):
        xlsx = _get_last_excel(job_id)
        if xlsx:
            try:
                app.logger.info("Auto-generate fiche on GET for job %s from %s", job_id, xlsx)
                generate_for_job(job_id, xlsx)  # crée le PDF si possible
            except Exception as e:
                # On répond quand même, mais on expose l'erreur
                return jsonify(ok=True, exists=False, url=None, error=str(e))
    exists = os.path.exists(pdf_path)
    return jsonify(ok=True, exists=exists, url=(pdf_url if exists else None))


@app.post("/api/fiche/generate")
def api_fiche_generate():
    data = request.get_json(force=True) or {}
    job_id = data.get("job_id") or ""
    xlsx_path = data.get("excel") or _get_last_excel(job_id)
    if not job_id: return jsonify(ok=False, error="job_id requis")
    if not xlsx_path or not os.path.exists(xlsx_path):
        return jsonify(ok=False, error="Excel introuvable pour ce job")
    try:
        out_pdf = generate_for_job(job_id, xlsx_path)
        _, pdf_url = _fiche_paths(job_id)
        return jsonify(ok=True, path=out_pdf, url=pdf_url, generated_at=time.time())
    except Exception as e:
        return jsonify(ok=False, error=f"Echec génération fiche: {e}")

@app.get("/api/whoami")
def whoami():
    lang = getattr(g, 'lang', 'fr')
    app.logger.info("whoami: lang=%s", lang)  # ← s’affiche dans le terminal
    return jsonify(ok=True, lang=lang)


@app.get("/health")
def health():
    return {"status": "ok"}, 200

# app.py
from flask import render_template

@app.get("/dashboard")
def dashboard():
    return render_template("index2.html")

# --- helpers communs ---

def _norm_key(s: str) -> str:
    return "".join(ch for ch in str(s).lower().strip() if ch.isalnum())

def _take_val(v):
    if v is None: return ""
    if isinstance(v, (str, int, float)): return str(v)
    if isinstance(v, list) and v: return _take_val(v[0])
    if isinstance(v, dict):
        for k in ("value","text","val","id","name"):
            if k in v: 
                vv = _take_val(v[k])
                if vv: return vv
    return ""

def _pick_id_from_fields(fields: dict | list) -> str:
    # candidats fréquents
    candidates = {"id","patient_id","id_patient","ipp","num_patient","patientid"}
    if isinstance(fields, dict):
        for k, v in fields.items():
            if _norm_key(k) in {_norm_key(c) for c in candidates}:
                val = _take_val(v).strip()
                if val: return val
        # parfois {name: "...", value: "..."}
        for v in fields.values():
            if isinstance(v, dict) and _norm_key(v.get("name","")) in {_norm_key(c) for c in candidates}:
                val = _take_val(v).strip()
                if val: return val
    elif isinstance(fields, list):
        for it in fields:
            if isinstance(it, dict) and _norm_key(it.get("name","")) in {_norm_key(c) for c in candidates}:
                val = _take_val(it).strip()
                if val: return val
    return ""

def _timeline_items_for_job(job_id: str):
    base = Path(f"runs/{job_id}")
    if not base.exists(): 
        return []
    items = []
    for p in base.rglob("*.json"):
        try:
            js = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue

        # Reprend la même logique que api_timeline pour fabriquer l'item
        def _extract_one(d: dict):
            it = _extract_item_from_dict(d, p)  # ta fonction existante
            if not it: 
                return None
            # on tente de deviner l'id à partir des fields
            fields = (d.get("fields") or d.get("extracted") or d.get("data") or
                      (isinstance(d.get("result"), dict) and (d["result"].get("fields") or d["result"].get("extracted") or d["result"].get("data"))))
            idv = _pick_id_from_fields(fields) if fields else ""
            it["id"] = idv or "unknown"
            return it

        if isinstance(js, dict):
            it = _extract_one(js)
            if it: items.append(it)
        elif isinstance(js, list):
            for elem in js:
                if isinstance(elem, dict):
                    it = _extract_one(elem)
                    if it: items.append(it)

    # ordre par id puis date
    items.sort(key=lambda x: (str(x.get("id") or ""), str(x.get("date_doc") or ""), x.get("filename") or ""))
    return items

@app.get("/api/timeline_dashboard")
def api_timeline_dashboard():
    job_id = request.args.get("job_id","").strip()
    if not job_id:
        return jsonify(ok=False, error="missing job_id"), 400
    items = _timeline_items_for_job(job_id)
    # groupe par id
    buckets: dict[str, list] = {}
    for it in items:
        bid = str(it.get("id") or "unknown")
        buckets.setdefault(bid, []).append(it)
    groups = [{"id": k, "items": v} for k, v in buckets.items()]
    groups = merge_timeline_groups_with_base(groups, job_id)
    return jsonify(ok=True, groups=groups)


# === [ADD] Helpers Append Excel ===
def get_job_dir(job_id: str) -> Path:
    d = JOBS_ROOT / secure_filename(job_id or "default")
    d.mkdir(parents=True, exist_ok=True)
    return d

def base_items_cache_path(job_id: str) -> Path:
    return get_job_dir(job_id) / "base_items_by_id.json"

def write_base_items_cache(job_id: str, data: dict) -> None:
    p = base_items_cache_path(job_id)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def read_base_items_cache(job_id: str) -> dict:
    p = base_items_cache_path(job_id)
    if p.exists():
        with p.open(encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_uploaded_excel(file_storage, job_id: str) -> Path:
    job_dir = get_job_dir(job_id)
    dest = job_dir / "base.xlsx"
    file_storage.save(dest)
    return dest

def find_base_excel_for_job(job_id: str) -> Path | None:
    p = get_job_dir(job_id) / "base.xlsx"
    return p if p.exists() else None

def _normalize_iso_date(val) -> str:
    try:
        dt = pd.to_datetime(val, errors="coerce", utc=False)
        if pd.isna(dt):
            return ""
        # Uniformiser en ISO (YYYY-MM-DD si sans heure, sinon complet)
        if getattr(dt, "time", None) and (dt.time() == pd.Timestamp("1970-01-01").time()):
            return str(dt.date())
        return dt.isoformat()
    except Exception:
        return str(val) or ""
    
    
def excel_to_items_by_id(xlsx_path: Path) -> dict:
    df = pd.read_excel(xlsx_path)

    lower_map = {c.lower().strip(): c for c in df.columns}
    def col(*cands):
        return next((lower_map[c] for c in cands if c in lower_map), None)

    id_col      = col("id", "patient_id", "pid")
    date_col    = col("date_doc", "date", "date_document", "date_evenement")
    title_col   = col("title", "filename", "document", "nom")
    cat_col     = col("category", "categorie", "type")
    content_col = col("content", "texte", "resume", "contenu", "abstract")

    if not id_col or not date_col:
        raise ValueError("L'Excel doit contenir les colonnes 'id' et 'date_doc' (ou 'date').")

    def clean(x):
        import pandas as pd
        return "" if (x is None or (isinstance(x, float) and pd.isna(x))) else str(x)

    recs: dict[str, list] = {}
    for _, row in df.iterrows():
        pid = clean(row[id_col]).strip()
        if not pid:
            continue
        item = {
            "date_doc": _normalize_iso_date(row[date_col]),
            "title":    clean(row[title_col])   if title_col   else "",
            "category": clean(row[cat_col])     if cat_col     else "Excel",
            "content":  clean(row[content_col]) if content_col else "",  # ← IMPORTANT
            "json_path": None,
            "source":   "excel",
        }
        recs.setdefault(pid, []).append(item)
    return recs




def merge_timeline_groups_with_base(groups: list[dict], job_id: str) -> list[dict]:
    """Fusionne les items Excel de base avec les items extraits déjà présents dans `groups`."""
    base = read_base_items_cache(job_id)
    by_id = {str(g.get("id")): g for g in groups}
    # Ajout/merge
    for pid, items in base.items():
        if pid in by_id:
            by_id[pid].setdefault("items", []).extend(items)
        else:
            groups.append({"id": pid, "items": list(items)})
            by_id[pid] = groups[-1]
    # Tri par date_doc
    def _key(it):
        try:
            return pd.to_datetime(it.get("date_doc") or "", errors="coerce")
        except Exception:
            return pd.NaT
    for g in groups:
        g.setdefault("items", []).sort(key=_key)
    return groups
def combine_existing_export_with_base(job_id: str, export_path: Path, base_excel_path: Path | None) -> Path:
    """
    Relit export.xlsx produit par le pipeline, le fusionne avec base.xlsx s'il existe,
    impose l'ordre de colonnes (les 4/6 premières selon l'ordre du tableur de base),
    puis écrit l'export global + les exports par patient.
    """
    if not export_path.exists() or not base_excel_path or not base_excel_path.exists():
        # Toujours produire des exports par patient si possible
        if export_path.exists():
            try:
                df_all = pd.read_excel(export_path)
                _write_exports_by_patient(export_path.parent, df_all)
            except Exception:
                pass
        return export_path

    df_new  = pd.read_excel(export_path)
    df_base = pd.read_excel(base_excel_path)

    # Union des colonnes
    all_cols = list({*df_new.columns, *df_base.columns})
    df_new  = df_new.reindex(columns=all_cols)
    df_base = df_base.reindex(columns=all_cols)

    # Concat + dédup
    df_all = pd.concat([df_base, df_new], ignore_index=True)

    # Déduplication "safe" si on a les colonnes clés
    subset = [c for c in ["id", "date_doc", "filename", "category"] if c in df_all.columns]
    if subset:
        df_all = df_all.drop_duplicates(subset=subset, keep="first")

    # === Nouvel ordre de colonnes ==========================================
    # 1) On prend l'ordre du tableur de base pour les premières colonnes.
    #    -> si le base a >= 6 colonnes : on aligne les 6 premières
    #    -> sinon, on aligne ce qu'on peut (au moins 4 si présentes)
    base_cols = list(df_base.columns)
    desired_front = [c for c in ["id_dossier", "title", "content", "category", "id", "date_doc"] if c in df_all.columns]

    # 2) Le reste des colonnes suit dans l’ordre actuel
    tail = [c for c in df_all.columns if c not in desired_front]
    df_all = df_all[desired_front + tail]
    # =======================================================================

    # Écrit export global fusionné (on écrase)
    df_all.to_excel(export_path, index=False)

    # Exports par patient (conservent l'ordre de df_all)
    _write_exports_by_patient(export_path.parent, df_all)
    return export_path




def _write_exports_by_patient(out_dir: Path, df_all: pd.DataFrame) -> None:
    byp = out_dir / "by_patient"
    byp.mkdir(exist_ok=True)
    if "id" in df_all.columns:
        for pid, g in df_all.groupby("id"):
            g.to_excel(byp / f"export_{pid}.xlsx", index=False)


def _combine_export_inplace_with_base(export_path: Path, base_excel_path: str | None) -> None:
    """
    Ouvre export_path (ex: runs/<job>/work/export.xlsx) et le concatène
    avec base_excel_path (ex: runs/<job>/base.xlsx), puis réécrit *export_path*.
    Déduplique sur un sous-ensemble de colonnes si possible.
    """
    if not base_excel_path:
        return
    bpath = Path(base_excel_path)
    if not (export_path and export_path.exists() and bpath.exists()):
        return

    df_new  = pd.read_excel(export_path)
    df_base = pd.read_excel(bpath)

    # Union des colonnes
    all_cols = list({*df_new.columns, *df_base.columns})
    df_new  = df_new.reindex(columns=all_cols)
    df_base = df_base.reindex(columns=all_cols)

    # Concat
    df_all = pd.concat([df_base, df_new], ignore_index=True)



    # Réécrit *au même endroit* : runs/<job>/work/export.xlsx
    df_all.to_excel(export_path, index=False)
    
    
# === [ADD] POST /api/base/upload_excel ===
@app.post("/api/base/upload_excel")
def api_upload_base_excel():
    try:
        job_id = (request.form.get("job_id") or "").strip()
        file   = request.files.get("excel")
        if not job_id:
            return jsonify(ok=False, error="Missing job_id"), 400
        if not file or not file.filename:
            return jsonify(ok=False, error="No file"), 400

        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_XLSX:
            return jsonify(ok=False, error="Only .xlsx files are supported"), 400

        dest = save_uploaded_excel(file, job_id)
        items_by_id = excel_to_items_by_id(dest)
        write_base_items_cache(job_id, items_by_id)

        return jsonify(
            ok=True,
            xlsx_path=str(dest),
            filename=Path(file.filename).name,
            rows=sum(len(v) for v in items_by_id.values()),
            patient_ids=list(items_by_id.keys()),
        )
    except Exception as e:
        app.logger.exception("upload_excel failed")
        return jsonify(ok=False, error=str(e)), 500


# -------- PAGE MODE PERSONNALISÉ --------
@app.get("/custom")
def custom_page():
    # même dossier de templates que "/" → index3.html est au même niveau que index.html
    return render_template("index3.html")

# -------- RUN CATALOG --------
@app.post("/api/run_catalog")
def run_catalog():
    """
    Exécute: extract_content -> cluster (noms renforcés avec catégories du catalogue) ->
             extract_fields (avec ce catalogue) -> json_to_excel
    Body JSON:
      {
        "job_id": str,
        "src_path": str,  # dossier uploadé (ou local)
        "catalog": object | null,        # JSON du catalogue (option 1)
        "catalog_path": str | null,      # chemin d'un fichier existant (option 2)
        "opts": {
          "shared": {"model": "...", "api_key": "...", "seed": 42},
          "steps": { ... }               # overrides optionnels par étape
        }
      }
    """
    data = request.get_json(force=True) or {}
    job_id = (data.get("job_id") or "").strip()
    src_path = Path(data.get("src_path") or "")
    if not job_id or not src_path:
        return jsonify(ok=False, error="job_id et src_path requis"), 400

    opts_all = data.get("opts") or {}
    shared = (opts_all.get("shared") or {}).copy()
    steps_opts = (opts_all.get("steps") or {}).copy()

    # fallback API key depuis env (comme run_all)
    if not (shared.get("api_key") or "").strip():
        env_key = os.getenv("MISTRAL_API_KEY", "").strip()
        if env_key:
            shared["api_key"] = env_key
        else:
            shared.pop("api_key", None)

    root = _job_root(job_id)
    work = root / "work"
    logs_dir = root / "logs"
    logs_dir.mkdir(exist_ok=True)
    work.mkdir(exist_ok=True)

    # Dossiers de travail
    content_json = work / "json_from_files"
    cluster_out  = work / "clusters"
    catalog_dir  = work / "catalog"
    extracted    = work / "json_extracted"
    excel_path   = work / "export.xlsx"

    def run_and_log(stepname: str, cmd: list[str]) -> tuple[int, str]:
        code, out = _capture(cmd, env=os.environ.copy())
        (logs_dir / f"{stepname}.log").write_text(out, encoding="utf-8")
        return code, out

    # --- 1) Sauvegarder / récupérer le catalogue dans runs/<job>/work/catalog/fields_catalog.json
    catalog_dir.mkdir(parents=True, exist_ok=True)
    dst_cat = catalog_dir / "fields_catalog.json"

    src_catalog_json = data.get("catalog")
    src_catalog_path = data.get("catalog_path")
    try:
        if src_catalog_json:
            with dst_cat.open("w", encoding="utf-8") as f:
                json.dump(src_catalog_json, f, ensure_ascii=False, indent=2)
            src_cat_for_scripts = dst_cat
            catalog_raw = src_catalog_json
        elif src_catalog_path:
            p = Path(src_catalog_path)
            shutil.copy2(p, dst_cat)
            src_cat_for_scripts = p  # on passe ce chemin aux scripts (comme en mode médical)
            with p.open("r", encoding="utf-8") as f:
                catalog_raw = json.load(f)
        else:
            return jsonify(ok=False, error="Catalogue manquant (catalog ou catalog_path)"), 400
    except Exception as e:
        return jsonify(ok=False, error=f"Impossible d'enregistrer le catalogue: {e}"), 400

    # --- 2) Construire la liste des catégories autorisées depuis 'catalog_raw'
    def _allowed_from_catalog(raw):
        try:
            # Formes supportées (mêmes shapes que _normalize_catalog / is_catalog_shape)
            if isinstance(raw, dict):
                if isinstance(raw.get("categories"), list):
                    return [str((it.get("name") or it.get("category") or "")).strip()
                            for it in raw["categories"] if isinstance(it, dict)]
                if isinstance(raw.get("fields_catalog"), dict):
                    return [str(k).strip() for k in raw["fields_catalog"].keys()]
                # dict direct: { "Imagerie": [...], "RCP": [...] }
                return [str(k).strip() for k in raw.keys()]
            if isinstance(raw, list):
                # liste d'objets [{name/category, fields:[...]}]
                return [str((it.get("name") or it.get("category") or "")).strip()
                        for it in raw if isinstance(it, dict)]
        except Exception:
            pass
        return []

    allowed = [x for x in _allowed_from_catalog(catalog_raw) if x]
    lang = getattr(g, "lang", "fr")
    name_extra = None
    if allowed:
        # même mécanisme que le mode médical, mais avec TES catégories
        # (cf. NAME_EXTRA_TPL + passage 'name_extra' à build_cmd_cluster) :contentReference[oaicite:2]{index=2}
        name_extra = NAME_EXTRA_TPL.get(lang, NAME_EXTRA_TPL["fr"]).format(list=", ".join(allowed))

    # --- 3) extract_content (commun)
    code, out = run_and_log(
        "extract_content",
        build_cmd_extract_content(src_path, content_json, steps_opts.get("extract_content", {}) | shared)
    )
    if code != 0:
        return jsonify(ok=False, where="extract_content", log=out, job_id=job_id), 500

    # --- 4) cluster (avec name_extra et write_back=True)
    cluster_opts = (steps_opts.get("cluster", {}) | shared) | {
        "name_extra": name_extra or "",
        "write_back": True,
    }
    code, out = run_and_log("cluster", build_cmd_cluster(content_json, cluster_out, cluster_opts))
    if code != 0:
        return jsonify(ok=False, where="cluster", log=out, job_id=job_id), 500

    # --- 5) extract_fields (avec --catalog pointant sur le fichier source que tu as fourni)
    ef_opts = (steps_opts.get("extract_fields", {}) | shared) | {"catalog": str(src_cat_for_scripts)}
    code, out = run_and_log("extract_fields", build_cmd_extract_fields(content_json, extracted, ef_opts))
    if code != 0:
        return jsonify(ok=False, where="extract_fields", log=out, job_id=job_id), 500

    # --- 6) json_to_excel
    code, out = run_and_log("json_to_excel", build_cmd_json_to_excel(extracted, excel_path, steps_opts.get("json_to_excel", {}) | shared))
    if code != 0:
        return jsonify(ok=False, where="json_to_excel", log=out, job_id=job_id), 500

    # (comme run_all) fusion éventuelle avec base.xlsx + mémorisation dernier excel
    try:
        base_excel_path = opts_all.get("base_excel_path") or str((_job_root(job_id) / "base.xlsx"))
        _combine_export_inplace_with_base(excel_path, base_excel_path)
    except Exception:
        pass
    _set_last_excel(job_id, str(excel_path))

    rel = excel_path.relative_to(root).as_posix()
    return jsonify(ok=True, excel=f"/download/{job_id}/{rel}", job_id=job_id)


# --- REVIEW MODE ---
from flask import send_file, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os, io, json, time, shutil, pathlib
import pandas as pd

try:
    import docx  # python-docx
except Exception:
    docx = None
    
def _norm(s: str) -> str:
    s = str(s or '').strip()
    s = re.sub(r'\s+', ' ', s)
    return s.lower()

def _norm_file_basename(relpath: str) -> str:
    b = os.path.basename(relpath or '')
    b = re.sub(r'\.[^.]+$', '', b)  # retire l’extension
    return _norm(b)

def _ts(): return time.strftime('%Y%m%d-%H%M%S')
def _mk_jobdir(job_id):
    base = os.path.join('runs', job_id, 'work')
    os.makedirs(base, exist_ok=True)
    return base
import datetime as dt
import numpy as np
import pandas as pd

def _json_safe(x):
    # None / NaN
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    # pandas NaT / NaN / numpy scalars
    if pd.isna(x):
        return None
    if isinstance(x, np.generic):      # int64, float64, bool_, etc.
        return x.item()
    # dates
    if isinstance(x, (pd.Timestamp, dt.datetime, dt.date)):
        return x.isoformat()
    # conteneurs
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    # tout le reste (str, int/float natifs, bool, …)
    return x

@app.route('/review')
def review_page():
    # Render index4.html
    return render_template('index4.html')

@app.post('/api/review/upload')
def api_review_upload():
    excel_file = request.files.get('excel')
    folder_files = request.files.getlist('folder')

    if not excel_file or not folder_files:
        return jsonify(ok=False, error='Excel ou dossier manquant'), 400

    job_id = request.args.get('job_id') or f"{_ts()}-review"
    work = _mk_jobdir(job_id)
    src_dir = os.path.join(work, 'src')
    os.makedirs(src_dir, exist_ok=True)

    # 1) Enregistrer l'Excel
    ext = os.path.splitext(excel_file.filename)[1].lower()
    xl_path = os.path.join(work, 'review.xlsx')
    if ext in ('.csv',):
        df = pd.read_csv(excel_file)
        df.to_excel(xl_path, index=False)
    else:
        excel_file.save(xl_path)
        df = pd.read_excel(xl_path)

    # Colonne filename (essaye de détecter)
    cols = list(df.columns)
    cand = [c for c in cols if c.lower() in ('title','file','fichier','path','chemin')]
    if cand:
        fname_col = cand[0]
    else:
        # rien trouvé : on crée une colonne 'filename' vide qu'on remplira si possible
        fname_col = 'title'
        if 'title' not in df.columns:
            df['title'] = ''

    # 2) Enregistrer les fichiers du dossier
    rel_paths = []
    for f in folder_files:
        rel = f.filename  # contient webkitRelativePath normalement
        if not rel or rel == os.path.basename(rel):
            # fallback: si pas de chemin relatif, on force dans src_dir
            rel = secure_filename(os.path.basename(f.filename))
        dst = os.path.join(src_dir, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        f.save(dst)
        rel_paths.append(rel)

    # 3) Order par tri alpha du chemin relatif
    order = sorted(rel_paths, key=lambda p: p.lower())

    # 4) Tenter de remplir filename si vide par nom de fichier "proche"
    #    (ici: on met juste le basename si vide)
    def base(p): return os.path.basename(p)
    if df[fname_col].isna().all() or (df[fname_col]== '').all():
        # on met simplement le basename des documents dans la colonne vide
        # (si df a plus de lignes que de docs, on ne touche qu’aux lignes vides)
        empties = df[fname_col].isna() | (df[fname_col]=='')
        # longueur commune
        n = min(empties.sum(), len(order))
        df.loc[empties[empties].index[:n], fname_col] = [base(p) for p in order[:n]]
        df.to_excel(xl_path, index=False)

    # 5) Déterminer colonnes éditables (tout sauf filename & index probables)
    reserved = set([fname_col, 'id', 'ID', 'index', 'Index'])
    columns = [c for c in df.columns if c not in reserved]

    # 6) État JSON
    state = {
        'job_id': job_id,
        'excel': xl_path,
        'src_dir': src_dir,
        'fname_col': fname_col,
        'columns': columns,
        'order': order,
    }
    with open(os.path.join(work, 'review_state.json'), 'w') as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

    return jsonify(ok=True, job_id=job_id, order=order, columns=columns, n_docs=len(order))

def _load_state(job_id):
    work = _mk_jobdir(job_id)
    p = os.path.join(work, 'review_state.json')
    if not os.path.exists(p): return None
    with open(p, 'r') as f:
        return json.load(f)

@app.get('/api/review/preview')
def api_review_preview():
    job_id = request.args.get('job_id'); file_rel = request.args.get('file')
    st = _load_state(job_id)
    if not st or not file_rel: return jsonify(ok=False, error='state or file missing'), 400
    path = os.path.join(st['src_dir'], file_rel)
    if not os.path.exists(path): return jsonify(ok=False, error='file not found'), 404

    ext = os.path.splitext(path)[1].lower()
    if ext in ('.pdf', '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'):
        # servir via une URL publique
        url = f"/api/review/file?job_id={job_id}&file={file_rel}"
        return jsonify(ok=True, type='embed', url=url)
    elif ext in ('.txt',):
        with open(path, 'r', errors='ignore') as f:
            txt = f.read()
        return jsonify(ok=True, type='text', text=txt[:300000])  # limite
    elif ext in ('.docx',) and docx is not None:
        try:
            d = docx.Document(path)
            txt = '\n'.join(p.text for p in d.paragraphs)
            return jsonify(ok=True, type='text', text=txt[:300000])
        except Exception as e:
            return jsonify(ok=True, type='text', text=f"[docx lecture partielle]\n{e}\nFichier: {os.path.basename(path)}")
    else:
        # pas d’aperçu -> lien
        return jsonify(ok=True, type='none', msg='Aperçu indisponible', url=f"/api/review/file?job_id={job_id}&file={file_rel}")

@app.get('/api/review/file')
def api_review_file():
    job_id = request.args.get('job_id'); file_rel = request.args.get('file')
    st = _load_state(job_id)
    if not st or not file_rel: return 'Missing', 400
    path = os.path.join(st['src_dir'], file_rel)
    if not os.path.exists(path): return 'Not found', 404
    return send_file(path, as_attachment=False)

@app.get('/api/review/fields')
def api_review_fields():
    import os, datetime as dt
    import numpy as np
    job_id = request.args.get('job_id'); file_rel = request.args.get('file')
    st = _load_state(job_id)
    if not st or not file_rel: 
        return jsonify(ok=False, error='state or file missing'), 400

    xl = st['excel']; fname_col = st['fname_col']; columns_all = st['columns']
    df = pd.read_excel(xl)

    base = os.path.basename(file_rel)
    base_noext = os.path.splitext(base)[0].strip().lower()

    # 1) match par filename (basenane)
    m = df[df[fname_col].astype(str).str.strip().str.lower() == base.lower()]

    # 2) fallback: match par colonne "title" si présente
    if m.empty:
        title_col = next((c for c in df.columns if c.lower() == 'title'), None)
        if title_col:
            m = df[df[title_col].astype(str).str.strip().str.lower() == base_noext]

    def _filled(v):
        if pd.isna(v): return False
        if isinstance(v, str): return v.strip() != ''
        return True  # garde 0, False, dates, etc.

    def _jsonable(v):
        if pd.isna(v): return None
        if isinstance(v, (np.integer, np.floating)): return v.item()
        if isinstance(v, (pd.Timestamp, dt.datetime, dt.date)): return str(v)
        if isinstance(v, (bool, int, float, str)): return v
        return str(v)

    values = {}
    if not m.empty:
        row = m.iloc[0]
        # 3) ne renvoyer que les colonnes "remplies" pour CE document
        columns = [c for c in columns_all if _filled(row.get(c))]
        for c in columns:
            values[c] = _jsonable(row.get(c))
    else:
        columns = []  # aucun champ si aucun match

    return jsonify(ok=True, columns=columns, values=values)




@app.post('/api/review/save')
def api_review_save():
    js = request.get_json(force=True, silent=True) or {}
    job_id = js.get('job_id'); file_rel = js.get('file'); values = js.get('values') or {}
    st = _load_state(job_id)
    if not st or not file_rel: return jsonify(ok=False), 400

    xl = st['excel']; columns = st['columns']
    title_col = st.get('title_col'); fname_col = st.get('fname_col')
    match_col = st.get('match_col') or title_col or fname_col

    df = pd.read_excel(xl)

    key_from_file = _norm_file_basename(file_rel)
    series = df[match_col].astype(str).map(_norm)
    if str(match_col).lower().strip() == 'title':
        series = series.map(lambda s: re.sub(r'\.[^.]+$', '', s))

    idxs = df.index[series == key_from_file].tolist()
    if not idxs:
        # crée la ligne
        new_row = {col: '' for col in df.columns}
        # remplis les colonnes clés
        if title_col and title_col in df.columns:
            new_row[title_col] = key_from_file
        if fname_col and fname_col in df.columns:
            import os
            new_row[fname_col] = os.path.basename(file_rel)
        # applique les valeurs éditées
        for c in columns:
            if c in values: new_row[c] = values[c]
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        i = idxs[0]
        for c in columns:
            if c in values:
                df.at[i, c] = values[c]

    df.to_excel(xl, index=False)
    return jsonify(ok=True)


@app.get('/api/review/download')
def api_review_download():
    job_id = request.args.get('job_id')
    st = _load_state(job_id)
    if not st: return 'Missing', 400
    return send_file(st['excel'], as_attachment=True, download_name='review_updated.xlsx')

# ---------------------- MAIN ----------------------

if __name__ == "__main__":
    # Flask simple (tu peux activer debug si besoin)
    app.run(host="0.0.0.0", port=3000, debug=True)
