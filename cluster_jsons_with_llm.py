
"""
Clustering de fiches JSON pseudonymisées + naming des clusters via Mistral,
avec envoi des **5 contenus les plus informatifs par cluster** au LLM pour améliorer le nommage,
et écriture **directe** du **nom de catégorie** dans chaque JSON en sortie (champ "category").

Entrée : un dossier contenant des .json (une fiche par fichier) au format :
{
  "title": str,
  "content": str,
  ... (autres champs ignorés)
}

Sorties dans --out :
- cluster_assignments.csv : fichier, title, cluster_id, cluster_name
- cluster_names.json      : mapping {cluster_id: {name, keywords, size, rationale}}
- clusters_pca.png        : scatter 2D PCA (couleurs par cluster)
- (effet de bord) ajout d'un champ "category" (nom du LLM) dans chaque JSON source

Dépendances :
  pip install sentence-transformers scikit-learn pandas numpy matplotlib mistralai tqdm

Clé API Mistral :
  export MISTRAL_API_KEY=...  # ou --api-key

Exemple :
  python3 cluster_jsons_with_llm.py \
    --src "/Users/ethanrouzier/Documents/Fiches acces precoce" \
    --out "/Users/ethanrouzier/Documents/Fiches_acces_precoce_clusters" \
    --model mistral-large-latest \
    --name-extra "Contexte: documents français d'accès précoce, oncologie, style administratif." \
    --seed 42

Option : fixer k
  --k 8  (sinon auto-sélection par silhouette sur [--min-k, --max-k])
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
# en haut des imports
from collections import Counter
import re

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

# Embeddings
from sentence_transformers import SentenceTransformer

# LLM (Mistral)
try:
    from mistralai import Mistral
except Exception:
    Mistral = None  # type: ignore

# Paramètre cluster TUNING
import os
NCLUSTER = int(os.getenv("NCLUSTER", "200"))

# ------------------------------ Utilitaires IO --------------------------------
def _norm_label(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s[:80]

def _labels_from_doc(d: dict) -> list[str]:
    out = []
    for k in ("category", "category_name"):
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            out.append(_norm_label(v))
    fields = d.get("fields") or d.get("Fields")
    if isinstance(fields, dict):
        for k in ("category", "category_name"):
            v = fields.get(k)
            if isinstance(v, str) and v.strip():
                out.append(_norm_label(v))
    return out

def collect_existing_labels(src_dir: Path, max_items: int = 50) -> list[str]:
    """
    Parcourt les JSON source et remonte les labels existants les plus fréquents.
    Retourne une liste unique triée par fréquence (max max_items).
    """
    cnt = Counter()
    for p in src_dir.rglob("*.json"):
        try:
            js = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(js, dict):
            for lab in _labels_from_doc(js):
                cnt[lab] += 1
        elif isinstance(js, list):
            for it in js:
                if isinstance(it, dict):
                    for lab in _labels_from_doc(it):
                        cnt[lab] += 1
    return [lab for lab, _ in cnt.most_common(max_items)]

def load_json_fiches(src_dir: Path) -> pd.DataFrame:
    files = sorted([p for p in src_dir.rglob('*.json') if p.is_file()])
    rows = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding='utf-8'))
            if isinstance(data, dict):
                title = str(data.get('title', f.stem) or f.stem)
                content = str(data.get('content', '') or '')
                text = (title + "" + content).strip()
                rows.append({
                    'file': str(f),
                    'filename': f.name,
                    'title': title,
                    'content': content,
                    'text': text,
                })
            elif isinstance(data, list):
                # on garde seulement les dicts avec title/content
                for i, item in enumerate(data):
                    if isinstance(item, dict):
                        title = str(item.get('title', f"{f.stem}#{i}"))
                        content = str(item.get('content', ''))
                        text = (title + "" + content).strip()
                        rows.append({
                            'file': f"{f}::{i}",
                            'filename': f"{f.name}::{i}",
                            'title': title,
                            'content': content,
                            'text': text,
                        })
        except Exception as e:
            print(f"⚠️  JSON invalide ignoré : {f} ({e})")
    if not rows:
        raise RuntimeError(f"Aucune fiche JSON trouvée dans {src_dir}")
    return pd.DataFrame(rows)


# ------------------------------ Embeddings ------------------------------------

def build_embeddings(texts: List[str], model_name: str, batch_size: int = 32) -> np.ndarray:
    st = SentenceTransformer(model_name)
    embs = st.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    return embs


# ------------------------------ Clustering ------------------------------------

def pick_best_k(X: np.ndarray, k: int | None, min_k: int, max_k: int, seed: int) -> Tuple[int, List[Tuple[int, float]]]:
    n = X.shape[0]
    if k is not None:
        return k, []
    ks = [kk for kk in range(min_k, max_k + 1) if kk < n]
    if not ks:
        return min(2, n), []
    scores = []
    for kk in ks:
        km = KMeans(n_clusters=kk, random_state=seed, n_init=10)
        labels = km.fit_predict(X)
        if len(set(labels)) == 1:
            sc = -1.0
        else:
            sc = float(silhouette_score(X, labels))
        scores.append((kk, sc))
    best_k = max(scores, key=lambda t: t[1])[0]
    return best_k, scores


def cluster_x(X: np.ndarray, k: int, seed: int) -> Tuple[np.ndarray, KMeans]:
    km = KMeans(n_clusters=k, random_state=seed, n_init=20)
    labels = km.fit_predict(X)
    return labels, km


# ------------------------------ Stopwords FR (robuste) ------------------------

def french_stopwords_or_none():
    """Essaie de fournir une liste de stopwords FR. Retourne None si indisponible."""
    try:
        # Tentative via NLTK
        import nltk  # type: ignore
        from nltk.corpus import stopwords  # type: ignore
        try:
            sw = stopwords.words('french')
        except LookupError:
            try:
                nltk.download('stopwords')
                sw = stopwords.words('french')
            except Exception:
                return None
        return sw
    except Exception:
        return None


# ------------------------------ Keywords --------------------------------------

def keywords_per_cluster(texts: List[str], labels: np.ndarray, topk: int = 30) -> Dict[int, List[str]]:
    # TF-IDF sur tout, puis on agrège par cluster
    sw = french_stopwords_or_none()
    tf = TfidfVectorizer(max_features=20000, ngram_range=(1,2), stop_words=sw)
    M = tf.fit_transform(texts)
    vocab = np.array(tf.get_feature_names_out())

    keywords = {}
    for cid in sorted(set(labels)):
        idx = np.where(labels == cid)[0]
        if len(idx) == 0:
            keywords[cid] = []
            continue
        mean_tfidf = np.asarray(M[idx].mean(axis=0)).ravel()
        top_idx = mean_tfidf.argsort()[::-1][:topk]
        keywords[cid] = [str(vocab[i]) for i in top_idx]
    return keywords


# ------------------------------ LLM naming ------------------------------------

def mistral_client(api_key: str | None) -> Any:
    if Mistral is None:
        raise RuntimeError("Le paquet 'mistralai' n'est pas installé. Faites: pip install mistralai")
    key = api_key or os.environ.get('MISTRAL_API_KEY')
    if not key:
        raise RuntimeError("Fournissez une clé API Mistral via $MISTRAL_API_KEY ou --api-key")
    return Mistral(api_key=key)


import os

LANG_MAP = {
    "fr": "français",
    "en": "anglais",
    "es": "espagnol",
    "it": "italien",
    "de": "allemand",
}

lang_code = (os.getenv("APP_LANG", "fr") or "fr").lower()
lang_name = LANG_MAP.get(lang_code, "français")

# TUNING PROMPT

def llm_name_cluster(
    client: Any,
    model: str,
    cid: int,
    samples: List[Dict[str, str]],  # [{title, content}], 5 entrées
    kw: List[str],
    extra: str | None,
    suggestions: List[str] | None = None,   # ← nouveau
) -> Dict[str, str]:
    # Construit un prompt concis et demande un JSON {name, rationale}
    sys = (
    f"N’écris que en {lang_name}. Si tu utilises une autre langue, corrige-toi immédiatement."
    "Tu es data scientist francophone. On te donne des exemples d’un même cluster."
    " Donne UNIQUEMENT le NOM DU CLUSTER, 1–4 mots clairs."
    " Interdits: JSON, Markdown, guillemets, explications."
)
    # Tronquer les contenus pour rester compacts (≈1200 chars chacun)
    def trunc(s: str, n: int = 1200) -> str:
        s = (s or '').replace('', ' ').strip()
        return s[:n] + ('…' if len(s) > n else '')

    user = {
        "cluster_id": int(cid),
        "samples": [{"title": str(t.get('title','')), "content": trunc(str(t.get('content','')))} for t in samples[:5]],
    }
    if extra:
        user["context"] = str(extra)
    if suggestions:
        short = ", ".join(suggestions[:40])  # borne dure pour garder un prompt léger
        sys += (
          " Si l'un des labels suivants correspond, réutilise EXACTEMENT celui-ci (préférable pour la cohérence) : "
          + short +
          ". Sinon, propose un nouveau label concis. Choisis des labels précis."
        )
    try:
        payload = json.dumps(user, ensure_ascii=False)
        resp = client.chat.complete(
            model=model,
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": payload}
            ],
            temperature=0.2,
            max_tokens=200,
        )
        text = resp.choices[0].message.content.strip()

        # --- parsing ultra-léger (plus de JSON obligatoire) ---
        raw = text.splitlines()[0].strip().strip(" #*—:-")
        # évite les justifications du style 'Oncologie — essais précoces'
        name = raw.split("—")[0].split(" - ")[0][:80] or f"Cluster {cid}"

        # garde une mini 'rationale' vide (pour compat CSV éventuelle)
        return {"name": name, "rationale": ""}
    except Exception as e:
        return {"name": f"Cluster {cid}", "rationale": f"(nommage LLM indisponible: {e})"}


# ------------------------------ Écriture dans les JSON ------------------------

def write_back_category(json_path: Path, category: str, list_index: int | None = None) -> None:
    """Ajoute/écrase le champ 'category' (nom du LLM) dans le JSON donné.
       - Si le JSON est un dict : ajoute 'category'
       - Si c'est une liste et list_index est fourni : ajoute 'category' sur l'élément dict visé
       - Si c'est une liste sans index : ajoute 'category' à tous les dicts
    """
    try:
        data = json.loads(json_path.read_text(encoding='utf-8'))
    except Exception:
        return

    changed = False
    if isinstance(data, dict):
        data['category'] = str(category)
        changed = True
    elif isinstance(data, list):
        if list_index is not None and 0 <= list_index < len(data) and isinstance(data[list_index], dict):
            data[list_index]['category'] = str(category)
            changed = True
        else:
            for i, it in enumerate(data):
                if isinstance(it, dict):
                    it['category'] = str(category)
                    changed = True
    if changed:
        try:
            json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        except Exception:
            pass


# ------------------------------ Pipeline principal ----------------------------

def run_pipeline(
    src: Path,
    out: Path,
    embedder: str,
    k: int | None,
    min_k: int,
    max_k: int,
    seed: int,
    model: str,
    api_key: str | None,
    name_extra: str | None,
    max_samples_per_cluster: int,
    naming_sample_size: int,
    write_back: bool,
):
    out.mkdir(parents=True, exist_ok=True)
    # 1) Charger les fiches
    df = load_json_fiches(src)
    n = len(df)

    # 2) Embeddings + 3) Choix de k + clustering
    if n < NCLUSTER:
        # ⚠️ Pas d'embeddings, pas de K-means
        X = None
        best_k = n
        scores = [(n, float("nan"))]
        labels = np.arange(n, dtype=int)   # un cluster par doc
        km = None
    else:
        # Embeddings
        X = build_embeddings(df['text'].tolist(), embedder)
        # Choix de k + clustering
        
        best_k, scores = pick_best_k(X, k, min_k, 50, seed)
        labels, km = cluster_x(X, best_k, seed)

    df['cluster_id'] = labels


    # 4) Mots-clés
    kw_map = keywords_per_cluster(df['text'].tolist(), labels)

    # 5) Nommage via LLM (5 contenus complets par cluster)
    client = mistral_client(api_key)

    # AVANT la boucle
    existing_labels = collect_existing_labels(src)      # scan 1 seule fois
    suggestions_pool = { _norm_label(x) for x in existing_labels }

    names: Dict[int, Dict[str, Any]] = {}
    for cid in sorted(set(labels)):
        sub = df[df.cluster_id == cid]

        # top N exemples les plus longs
        n = min(naming_sample_size, len(sub))
        sub_s = (sub.assign(_clen=sub['content'].fillna('').map(len))
                    .sort_values('_clen', ascending=False)
                    .head(n))
        samples = [{"title": str(r.title), "content": str(r.content)}
                for _, r in sub_s[['title','content']].iterrows()]

        # ⚠️ envoie VRAIMENT 5 exemples, pas [:1]
        res = llm_name_cluster(
            client, model, int(cid),
            samples, kw_map.get(cid, []), name_extra,
            suggestions=sorted(suggestions_pool)    # ← suggestions actuelles
        )

        names[int(cid)] = {"name": res['name'], "rationale": res['rationale'],
                        "keywords": kw_map.get(cid, []), "size": int(len(sub))}

        # ← ajoute tout de suite le nom choisi pour les prochains clusters
        suggestions_pool.add(_norm_label(res['name']))
    # 6) Sorties tabulaires
    name_map = {cid: v['name'] for cid, v in names.items()}
    df['cluster_name'] = df['cluster_id'].map(name_map)

    df_out = df[['file', 'filename', 'title', 'cluster_id', 'cluster_name']].copy()
    df_out.to_csv(out / 'cluster_assignments.csv', index=False)

    with open(out / 'cluster_names.json', 'w', encoding='utf-8') as g:
        json.dump({int(k): v for k, v in names.items()}, g, ensure_ascii=False, indent=2)

    # 7) PCA 2D pour visualisation (UNIQUEMENT si n >= 10)
    if X is not None and len(df) >= NCLUSTER:
        try:
            pca = PCA(n_components=2, random_state=seed)
            X2 = pca.fit_transform(X)
            plt.figure(figsize=(8,6))
            for cid in sorted(set(labels)):
                idx = np.where(labels == cid)[0]
                plt.scatter(X2[idx,0], X2[idx,1], s=10, label=f"{int(cid)}: {name_map[int(cid)]}")
            plt.xlabel('PC1'); plt.ylabel('PC2'); plt.title('Clustering fiches (PCA 2D)')
            plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', fontsize='small')
            plt.tight_layout()
            plt.savefig(out / 'clusters_pca.png', dpi=160)
        except Exception as e:
            print(f"[PCA] ignorée (erreur): {e}")
    else:
        print("[PCA] ignorée: pas assez de documents.")


    if scores:
        with open(out / 'silhouette_scores.json', 'w', encoding='utf-8') as g:
            json.dump([{ 'k': int(k), 'silhouette': float(s)} for k, s in scores], g, indent=2)

    # 8) Écriture de la catégorie (nom du LLM) dans les JSON source
    if write_back:
        # mapping fichier → cluster_id
        assignments = dict(zip(df['file'], df['cluster_id']))
        # mapping cluster_id → nom
        cluster_names = name_map

        for key, cid in assignments.items():
            p = _resolve_json_path(src, key)
            if p is None:
                # on tente aussi avec le basename
                p = _resolve_json_path(src, os.path.basename(str(key)))
            if p is None:
                continue
            write_category_into_json(p, cid, names_map=cluster_names)


    print(f"✅ Terminé. k={best_k} | Résultats dans: {out} | Catégories (noms LLM) écrites: {write_back}")

def _resolve_json_path(src_dir: str | Path, path_like: str | Path) -> Path | None:
    """Résout un chemin absolu/relatif, ou recherche par basename dans src_dir."""
    p = Path(path_like)
    if p.exists():
        return p
    # recherche par basename dans tout src_dir
    try:
        base = p.name
    except Exception:
        base = str(p)
    for cand in Path(src_dir).rglob("*.json"):
        if cand.name == base:
            return cand
    return None


def _ensure_category_fields(doc: dict, category_value: str) -> dict:
    """Écrit la category en top-level et dans fields (si dict). N'écrase pas l'existant."""
    if isinstance(category_value, str) and category_value.strip():
        if not doc.get("category"):
            doc["category"] = category_value.strip()
        if "category_name" not in doc:
            doc["category_name"] = doc["category"]
        # refléter aussi dans fields/Fields si présent
        fields = doc.get("fields") or doc.get("Fields")
        if isinstance(fields, dict):
            fields.setdefault("category", doc["category"])
            fields.setdefault("category_name", doc["category_name"])
            # standardiser sur "fields" (optionnel)
            doc["fields"] = fields
    return doc


def write_category_into_json(json_path: Path, cluster_id: int | str, names_map: dict | None):
    """Ouvre le JSON, calcule le nom lisible du cluster, et écrit category + category_name."""
    # récupère le nom du cluster
    name = None
    if names_map:
        name = names_map.get(str(cluster_id)) or names_map.get(cluster_id)
    if not name:
        name = f"Cluster {cluster_id}"

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except Exception:
        return  # on ignore silencieusement les fichiers illisibles

    doc = _ensure_category_fields(doc, name)

    # sauvegarde
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# ------------------------------ CLI -------------------------------------------

def main():
    ap = argparse.ArgumentParser(description='Clustering de fiches JSON + naming via Mistral (5 contenus/cluster) + écriture catégorie (nom LLM)')
    ap.add_argument('--src', required=True, type=Path, help='Dossier avec JSON pseudonymisés')
    ap.add_argument('--out', required=True, type=Path, help='Dossier de sortie')
    ap.add_argument('--embedder', default='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', help='Modèle SentenceTransformer')
    ap.add_argument('--k', type=int, default=None, help='Nombre de clusters (sinon auto)')
    ap.add_argument('--min-k', type=int, default=4, help='k min pour auto-sélection')
    ap.add_argument('--max-k', type=int, default=14, help='k max pour auto-sélection')
    ap.add_argument('--seed', type=int, default=42)

    ap.add_argument('--model', default='mistral-large-latest', help='Modèle LLM pour le naming')
    ap.add_argument('--api-key', default=None, help='Clé API Mistral (sinon $MISTRAL_API_KEY)')
    ap.add_argument('--name-extra', default=None, help='Informations supplémentaires à inclure dans le prompt de naming')
    ap.add_argument('--max-samples-per-cluster', type=int, default=15, help='Nb max d’exemples pour features (conservé pour compat)')
    ap.add_argument('--naming-sample-size', type=int, default=5, help='Nb de contenus complets envoyés au LLM (par cluster)')
    ap.add_argument('--no-write-back', action='store_true', help="Ne pas écrire la catégorie dans les JSON")

    args = ap.parse_args()
    run_pipeline(
        src=args.src,
        out=args.out,
        embedder=args.embedder,
        k=args.k,
        min_k=args.min_k,
        max_k=args.max_k,
        seed=args.seed,
        model=args.model,
        api_key=args.api_key,
        name_extra=args.name_extra,
        max_samples_per_cluster=args.max_samples_per_cluster,
        naming_sample_size=args.naming_sample_size,
        write_back=(not args.no_write_back),
    )




if __name__ == '__main__':
    main()

#python3 cluster_jsons_with_llm.py \
#  --src "/Users/ethanrouzier/Documents/Fiches acces precoce" \
#  --out "/Users/ethanrouzier/Documents/Fiches_acces_precoce_clusters" \
#  --model mistral-large-latest \
#  --name-extra "Contexte: documents français d'accès précoce, oncologie, style administratif." \
#  --seed 42