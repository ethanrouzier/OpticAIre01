#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import json
import shutil
from typing import Any, Dict, List, Tuple, Optional
# --- Shim pour résoudre __main__.SBERTEncoder au chargement du .joblib ---
try:
    from sklearn.base import BaseEstimator, TransformerMixin
    import numpy as np
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        SentenceTransformer = None

    class SBERTEncoder(BaseEstimator, TransformerMixin):
        """
        Shim minimal pour des pipelines picklés qui référencent __main__.SBERTEncoder.
        Il encode des textes en embeddings SBERT via sentence-transformers.
        """
        def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2",
                     normalize=True, batch_size=256, device=None):
            self.model_name = model_name
            self.normalize = normalize
            self.batch_size = batch_size
            self.device = device
            self._model = None

        def _ensure_model(self):
            if self._model is None:
                if SentenceTransformer is None:
                    raise RuntimeError(
                        "Le package 'sentence-transformers' est requis. "
                        "Installe: pip install sentence-transformers"
                    )
                self._model = SentenceTransformer(self.model_name, device=self.device)

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            self._ensure_model()
            texts = [x if isinstance(x, str) else ("" if x is None else str(x)) for x in X]
            embs = self._model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                normalize_embeddings=self.normalize,
                convert_to_numpy=True
            )
            return embs

        # scikit ≥1.0 tolère l'absence ; on laisse None
        def get_feature_names_out(self, input_features=None):
            return None

except Exception as _shim_err:
    # On ne bloque pas ici; si le joblib n'a pas besoin de SBERTEncoder, ce try/except évite d'échouer.
    pass

import joblib

def iter_json_files(root_dir: str, recursive: bool = True) -> List[str]:
    files = []
    if recursive:
        for dirpath, _, filenames in os.walk(root_dir):
            for fn in filenames:
                if fn.lower().endswith(".json"):
                    files.append(os.path.join(dirpath, fn))
    else:
        for fn in os.listdir(root_dir):
            p = os.path.join(root_dir, fn)
            if os.path.isfile(p) and fn.lower().endswith(".json"):
                files.append(p)
    files.sort()
    return files

def get_parent_and_key_for_path(obj: Any, path: List[str]) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Retourne (parent_obj, last_key) pour une clé pointée par path dans un dict imbriqué.
    - Si le root est une liste, on ne gère pas ici ; on gère le cas liste à part.
    - Ne crée rien ; si le chemin n'existe pas, renvoie (None, None).
    """
    cur = obj
    for k in path[:-1]:
        if not isinstance(cur, dict) or k not in cur:
            return None, None
        cur = cur[k]
    if isinstance(cur, dict) and path[-1] in cur:
        return cur, path[-1]
    return None, None

def extract_targets_from_obj(
    obj: Any,
    content_path: List[str],
    category_key: str
) -> List[Tuple[Dict, str, str]]:
    """
    Retourne une liste d'items à prédire sous forme de tuples:
    (parent_dict, category_key, text_content)
    - Si obj est un dict: on cherche content_path, et on écrira category_key au même niveau.
    - Si obj est une liste: on applique la même logique à chaque élément dict.
    """
    targets: List[Tuple[Dict, str, str]] = []

    def as_text(x: Any) -> str:
        if x is None:
            return ""
        if isinstance(x, (str, int, float, bool)):
            return str(x)
        return json.dumps(x, ensure_ascii=False)

    if isinstance(obj, dict):
        parent, last_key = get_parent_and_key_for_path(obj, content_path)
        if parent is not None and last_key is not None:
            txt = as_text(parent[last_key])
            targets.append((parent, category_key, txt))
    elif isinstance(obj, list):
        for el in obj:
            if isinstance(el, dict):
                parent, last_key = get_parent_and_key_for_path(el, content_path)
                if parent is not None and last_key is not None:
                    txt = as_text(parent[last_key])
                    targets.append((parent, category_key, txt))
    # autres types (str, etc.) -> rien à faire
    return targets

def main():
    parser = argparse.ArgumentParser(
        description="Parcourt un dossier de JSON, lit 'content', prédit 'category' avec un pipeline .joblib et écrit la catégorie dans chaque JSON."
    )
    parser.add_argument("model_path", type=str, help="Chemin vers le pipeline .joblib entraîné (SBERT + SVC)")
    parser.add_argument("input_dir", type=str, help="Dossier contenant les fichiers .json")
    parser.add_argument("--content-key", type=str, default="content",
                        help="Clé (chemin pointé par des points) pour le texte. Ex: 'content' ou 'payload.content'")
    parser.add_argument("--category-key", type=str, default="category",
                        help="Nom du champ à écrire pour la prédiction")
    parser.add_argument("--recursive", action="store_true", help="Parcourir le dossier en récursif")
    parser.add_argument("--batch-size", type=int, default=256, help="Taille de batch pour la prédiction")
    parser.add_argument("--backup", action="store_true", help="Créer une sauvegarde .bak avant écriture")
    parser.add_argument("--dry-run", action="store_true", help="Ne rien écrire, juste afficher un récapitulatif")
    args = parser.parse_args()

    # Chargement du pipeline entraîné (doit être celui produit par train_doc_type_svc.py)
    try:
        pipe = joblib.load(args.model_path)
    except Exception as e:
        print(f"[ERREUR] Impossible de charger le modèle '{args.model_path}': {e}", file=sys.stderr)
        sys.exit(1)

    files = iter_json_files(args.input_dir, recursive=args.recursive)
    if not files:
        print("[INFO] Aucun fichier .json trouvé.")
        sys.exit(0)

    content_path = [p for p in args.content_key.split(".") if p]
    cat_key = args.category_key

    # Collecte des cibles à prédire
    all_entries: List[Tuple[str, Dict, str]] = []  # (file_path, parent_dict_ref, text)
    per_file_docs: Dict[str, Any] = {}  # stocke le JSON chargé pour réécriture
    missing_count = 0

    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARN] Lecture échouée pour {fp}: {e}")
            continue

        per_file_docs[fp] = data
        targets = extract_targets_from_obj(data, content_path, cat_key)
        if not targets:
            missing_count += 1
            continue

        for (parent, category_key, text) in targets:
            all_entries.append((fp, parent, text))

    if not all_entries:
        print("[INFO] Aucune entrée 'content' trouvée dans les JSON fournis.")
        if missing_count:
            print(f"[INFO] Fichiers sans champ content détectés: {missing_count}")
        sys.exit(0)

    # Prédictions en batch
    texts = [t[2] for t in all_entries]
    preds: List[str] = []
    bs = max(1, args.batch_size)
    for i in range(0, len(texts), bs):
        batch = texts[i:i+bs]
        try:
            preds.extend(pipe.predict(batch).tolist())
        except Exception as e:
            print(f"[ERREUR] Échec de prédiction sur le batch {i}:{i+bs}: {e}", file=sys.stderr)
            sys.exit(1)

    # Application des prédictions dans les objets en mémoire
    for (entry, pred) in zip(all_entries, preds):
        _, parent, _ = entry
        parent[cat_key] = pred

    # Écriture
    written, errors = 0, 0
    if args.dry_run:
        print(f"[DRY-RUN] {len(all_entries)} éléments auraient reçu un champ '{cat_key}'.")
        print(f"[DRY-RUN] Fichiers sans 'content': {missing_count}")
        sys.exit(0)

    for fp, data in per_file_docs.items():
        try:
            if args.backup:
                try:
                    shutil.copy2(fp, fp + ".bak")
                except Exception as e:
                    print(f"[WARN] Impossible de créer la sauvegarde pour {fp}: {e}")
            with open(fp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            written += 1
        except Exception as e:
            print(f"[WARN] Écriture échouée pour {fp}: {e}")
            errors += 1

    print(f"✅ Fichiers écrits: {written} | ⚠️ erreurs: {errors} | 🕳️ sans 'content': {missing_count}")
    print(f"Champ prédiction utilisé: '{cat_key}' | Clé source: '{args.content_key}'")
    print("Terminé.")

if __name__ == "__main__":
    main()

#python predict_json_dir.py svc_sbert_mMiniLM.joblib /chemin/vers/mon_dossier_json --recursive --backup
