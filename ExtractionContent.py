#!/usr/bin/env python3
"""
Extraction de contenus (titre + content) depuis tous les fichiers d'un dossier racine,
avec exploration récursive des sous-dossiers, et écriture d'une fiche JSON par fichier.

Entrées :
  --src  : dossier source (ex: "/Users/ethanrouzier/Documents/Accès précoces")
  --dst  : dossier de sortie (ex: "/Users/ethanrouzier/Documents/Fiches acces precoce")

Types pris en charge : .pdf, .png, .doc, .docx
- PDF : extraction via pypdf si possible ; OCR fallback si pdf2image + pytesseract dispo
- PNG : OCR via pytesseract si dispo
- DOCX : via python-docx si dispo, sinon 'textutil' (macOS) ou textract si dispo
- DOC (legacy) : 'textutil' (macOS) conseillé ; sinon textract si dispo

Dépendances facultatives utiles :
  pip install pypdf python-docx pillow pytesseract pdf2image textract

Sur macOS, 'textutil' prend en charge .doc/.docx -> .txt (pré-installé).
Pour l'OCR (png/pdf scannés), Tesseract doit être installé (ex: via Homebrew: brew install tesseract).

Sortie : pour chaque fichier traité, un JSON dans --dst avec :
  {
    "title": str,   # titre du document ou nom de fichier
    "content": str  # texte brut extrait
  }

Si l'extraction échoue, une fiche est quand même créée avec content="".
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Chargements paresseux (import si disponible)
try:
    from pypdf import PdfReader  # type: ignore
except Exception:
    PdfReader = None  # type: ignore

try:
    from PIL import Image  # type: ignore
except Exception:
    Image = None  # type: ignore

try:
    import pytesseract  # type: ignore
except Exception:
    pytesseract = None  # type: ignore

try:
    import docx as docx_lib  # python-docx
except Exception:
    docx_lib = None  # type: ignore

try:
    import textract  # type: ignore
except Exception:
    textract = None  # type: ignore

try:
    from pdf2image import convert_from_path  # type: ignore
except Exception:
    convert_from_path = None  # type: ignore


@dataclass
class ExtractionResult:
    title: str
    content: str


def is_macos() -> bool:
    return platform.system() == "Darwin"


def call_textutil_to_txt(path: Path) -> Optional[str]:
    """Utilise 'textutil' (macOS) pour convertir DOC/DOCX en texte brut.
    Retourne le texte ou None si indisponible/échec.
    """
    if not is_macos() or shutil.which("textutil") is None:
        return None
    try:
        # -convert txt -stdout permet de récupérer directement le texte
        proc = subprocess.run(
            ["textutil", "-convert", "txt", str(path), "-stdout"],
            check=True,
            capture_output=True,
        )
        return proc.stdout.decode("utf-8", errors="ignore")
    except Exception:
        return None


def read_docx(path: Path) -> Optional[str]:
    """Lit un .docx avec python-docx si dispo, sinon tente textutil puis textract."""
    # 1) python-docx
    if docx_lib is not None:
        try:
            doc = docx_lib.Document(str(path))
            paragraphs = [p.text for p in doc.paragraphs]
            return "\n".join([p for p in paragraphs if p is not None])
        except Exception:
            pass
    # 2) textutil (macOS)
    txt = call_textutil_to_txt(path)
    if txt is not None:
        return txt
    # 3) textract
    if textract is not None:
        try:
            raw = textract.process(str(path))
            return raw.decode("utf-8", errors="ignore")
        except Exception:
            pass
    return None


def read_doc(path: Path) -> Optional[str]:
    """Lit un .doc (legacy). Préfère textutil (macOS), sinon textract."""
    # 1) textutil (macOS)
    txt = call_textutil_to_txt(path)
    if txt is not None:
        return txt
    # 2) textract
    if textract is not None:
        try:
            raw = textract.process(str(path))
            return raw.decode("utf-8", errors="ignore")
        except Exception:
            pass
    return None


def read_pdf(path: Path) -> Optional[str]:
    """Lit un PDF. Essaie d'abord pypdf (texte natif), sinon OCR via pdf2image+pytesseract si dispo."""
    text = None
    # 1) pypdf (texte sélectionnable)
    if PdfReader is not None:
        try:
            reader = PdfReader(str(path))
            chunks = []
            for page in reader.pages:
                try:
                    chunks.append(page.extract_text() or "")
                except Exception:
                    chunks.append("")
            text = "\n".join(chunks).strip()
            # Nettoyage rudimentaire
            if text:
                # Parfois, pypdf renvoie beaucoup de sauts de ligne ; on compacte un peu
                text = "\n".join(line.strip() for line in text.splitlines())
        except Exception:
            text = None
    # 2) OCR si vide ou impossible
    if (not text) and convert_from_path is not None and pytesseract is not None and Image is not None:
        try:
            # Pour éviter des OCR trop lourds, on traite jusqu'à 25 pages par défaut
            images = convert_from_path(str(path), fmt="png", dpi=300, first_page=1, last_page=None)
            ocr_pages = []
            max_pages = 25
            for i, img in enumerate(images):
                if i >= max_pages:
                    break
                ocr_pages.append(pytesseract.image_to_string(img))
            text = "\n".join(ocr_pages).strip()
        except Exception:
            pass
    return text


def read_png(path: Path) -> Optional[str]:
    """OCR sur PNG si pytesseract + PIL dispo."""
    if pytesseract is None or Image is None:
        return None
    try:
        img = Image.open(str(path))
        return pytesseract.image_to_string(img)
    except Exception:
        return None

# --- AJOUT : lecture de fichiers .txt ---
from typing import Optional  # (déjà importé chez toi normalement)

def read_txt(path: Path) -> Optional[str]:
    """Lit un .txt en essayant plusieurs encodages courants."""
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            with open(path, "r", encoding=enc) as f:
                text = f.read()
            # normalisation des fins de lignes
            return text.replace("\r\n", "\n").replace("\r", "\n")
        except Exception:
            continue
    # Dernier recours : on tente un decode en ignorant les erreurs
    try:
        with open(path, "rb") as f:
            raw = f.read()
        return raw.decode("utf-8", errors="ignore")
    except Exception:
        return None
    
def guess_title_from_pdf(path: Path) -> Optional[str]:
    if PdfReader is None:
        return None
    try:
        reader = PdfReader(str(path))
        meta = getattr(reader, "metadata", None)
        if meta:
            title = getattr(meta, "title", None)
            if title and isinstance(title, str):
                t = title.strip()
                if t:
                    return t
    except Exception:
        return None
    return None


def extract_one(path: Path, root: Path) -> ExtractionResult:
    suffix = path.suffix.lower()
    content: Optional[str] = None
    title: str = path.stem  # <- titre par défaut = nom de fichier sans extension

    if suffix == ".pdf":
        meta_title = guess_title_from_pdf(path)
        if meta_title:
            title = meta_title
        content = read_pdf(path)
    elif suffix == ".png":
        content = read_png(path)
    elif suffix == ".docx":
        content = read_docx(path)
    elif suffix == ".doc":
        content = read_doc(path)
    elif suffix == ".txt":
        # ICI: on prend toujours le nom de fichier sans extension
        content = read_txt(path)

    if not content:
        content = ""

    return ExtractionResult(title=title, content=content)



def safe_json_name(src_file: Path, root: Path) -> str:
    """Génère un nom de fiche unique à partir du nom de fichier + hash du chemin relatif."""
    rel = src_file.relative_to(root).as_posix()
    h = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:8]
    base = f"{src_file.stem}__{h}.json"
    # Remplacer caractères peu pratiques
    base = base.replace(os.sep, "_")
    return base


def extract_to_json(src_dir: Path, dst_dir: Path) -> None:
    supported = {".pdf", ".png", ".doc", ".docx", ".txt"}
    files = [p for p in src_dir.rglob("*") if p.is_file() and p.suffix.lower() in supported]

    if not files:
        print(f"Aucun fichier pris en charge trouvé dans {src_dir}")
        return

    dst_dir.mkdir(parents=True, exist_ok=True)

    ok, failed = 0, 0
    for f in files:
        try:
            res = extract_one(f, src_dir)
            out_name = safe_json_name(f, src_dir)
            out_path = dst_dir / out_name
            data = {"title": res.title, "content": res.content}
            with open(out_path, "w", encoding="utf-8") as g:
                json.dump(data, g, ensure_ascii=False, indent=2)
            print(f"✅ {f} -> {out_path}")
            ok += 1
        except Exception as e:
            print(f"⚠️  Échec extraction pour {f}: {e}")
            # On écrit quand même une fiche vide, pour respecter "une fiche par fichier"
            try:
                out_name = safe_json_name(f, src_dir)
                out_path = dst_dir / out_name
                data = {"title": f.stem, "content": ""}
                with open(out_path, "w", encoding="utf-8") as g:
                    json.dump(data, g, ensure_ascii=False, indent=2)
                print(f"→ Fiche vide créée: {out_path}")
                failed += 1
            except Exception as e2:
                print(f"❌ Impossible d'écrire la fiche pour {f}: {e2}")
                failed += 1

    print("\nRésumé:")
    print(f"  Réussites : {ok}")
    print(f"  Échecs    : {failed}")
    print(f"  Sortie    : {dst_dir}")


def main():
    parser = argparse.ArgumentParser(description="Extraction titre+contenu en JSON, une fiche par fichier.")
    parser.add_argument("--src", required=True, type=Path, help="Dossier source à explorer récursivement")
    parser.add_argument("--dst", required=True, type=Path, help="Dossier de sortie pour les JSON")
    args = parser.parse_args()

    extract_to_json(args.src, args.dst)


if __name__ == "__main__":
    main()


#python3 ExtractionContent.py \
#  --src "/Users/ethanrouzier/Documents/Accès précoces" \
#  --dst "/Users/ethanrouzier/Documents/Fiches acces precoce NP"