import os
import re
import argparse
import logging
from typing import Any, List, Optional, Tuple, Dict
import pandas as pd
import duckdb
from mistralai import Mistral
from dataclasses import dataclass, asdict
import json


# --- Constants: Default configuration ---
MODEL_NAME = 'mistral-large-latest'
EXCEL_PATH = '/Users/ethanrouzier/Documents/Pseudonymisation/resultats_pseudonymes.xlsx'
MAX_METHODS_DEFAULT = 1
DEPTH_RETRIES_DEFAULT = 1
MAX_AGENT_CALLS = 5  # Nombre maximum d'appels à l'agent secondaire


# --- Configuration and Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

@dataclass
class Config:
    model_name: str
    api_key: str
    excel_path: str
    max_methods: int
    depth_retries: int

def load_config() -> Config:
    parser = argparse.ArgumentParser(
        description="Pipeline Excel-QA via SQL avec moteur Mistral et agent secondaire"
    )
    parser.add_argument(
        "--excel", default=EXCEL_PATH,
        help=f"Chemin du fichier Excel à analyser (défaut: {EXCEL_PATH})"
    )
    parser.add_argument(
        "--model", default=MODEL_NAME,
        help=f"Nom du modèle Mistral à utiliser (défaut: {MODEL_NAME})",
    )
    parser.add_argument(
        "--api-key", dest="api_key", default=None,
    help="Clé API (priorité) ; sinon MISTRAL_API_KEY/OPENAI_API_KEY dans l'environnement"
    )
    parser.add_argument(
        "--max-methods", type=int, default=MAX_METHODS_DEFAULT,
        help=f"Nombre de méthodes différentes à tester (défaut: {MAX_METHODS_DEFAULT})"
    )
    parser.add_argument(
        "--depth-retries", type=int, default=DEPTH_RETRIES_DEFAULT,
        help=f"Nombre d'affinements par méthode (défaut: {DEPTH_RETRIES_DEFAULT})"
    )
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("MISTRAL_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        parser.error("API key manquante : utilisez --api-key ou définissez MISTRAL_API_KEY/OPENAI_API_KEY.")

    if not api_key:
        logging.error("La variable d'environnement MISTRAL_API_KEY n'est pas définie.")
        parser.error("MISTRAL_API_KEY manquant dans l'environnement.")

    return Config(
        excel_path=args.excel,
        model_name=args.model,
        api_key=api_key,
        max_methods=args.max_methods,
        depth_retries=args.depth_retries,
    )

# ------------------ LLM helpers ------------------
def call_mistral(
    client: Mistral,
    prompt: str,
    model: str,
    max_tokens: int = 1024
) -> Optional[str]:
    logging.debug("Prompt envoyé à Mistral:\n%s", prompt)
    try:
        resp = client.chat.complete(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        text = resp.choices[0].message.content.strip()
        logging.debug("Réponse reçue de Mistral.")
        return text
    except Exception as e:
        logging.error("Erreur lors de l'appel Mistral: %s", e)
        return None

# Stratégies explicites par "méthode" (le LLM les voit et on les journalise)
METHOD_STRATEGIES = [
    "Filtrage mots-clés: tokens >=3 sur colonnes texte (content, title, Conclusion, etc.) avec AND de groupes OR LIKE/ILIKE.",
    "Regex ciblées + normalisation: REGEXP_MATCHES sur content et champs cliniques, variantes et mots composés.",
    "Filtrage métadonnées + texte: restreindre par colonnes (category, type, organe, date) puis recherche texte.",
    "Agrégation par id_dossier: compter motifs multiples, ANY/COUNT, GROUP BY puis JOIN content.",
]

@dataclass
class Attempt:
    method: int
    depth: int
    strategy: str
    prompt: str
    sql: str | None
    error: str | None
    row_count: int | None
    ids: list[str]
    sample: list[Dict[str, str]]     # mini-aperçus (id + snippet content)
    supervisor_hint: str | None      # recommandations de l’agent-critique
    agent_summaries: list[str]       # résumés par l’agent "enrichissement"

# ------------------ Utils ------------------
def _preview_rows(df, max_rows=5, snip=220):
    out = []
    if df is None or df.empty: 
        return out
    cols_lower = {c.lower(): c for c in df.columns}
    id_col = cols_lower.get("id_dossier", next(iter(df.columns)))
    content_col = cols_lower.get("content")
    for r in df.head(max_rows).itertuples(index=False):
        d = getattr(r, id_col, "")
        snippet = ""
        if content_col:
            snippet = str(getattr(r, content_col, "")).replace("\n", " ")[:snip]
        out.append({"id_dossier": str(d), "snippet": snippet})
    return out

def review_sql_attempt(
    client: Mistral,
    question: str,
    columns: list[str],
    last_sql: str,
    outcome: str,               # ex: "Erreur: <msg>" ou "0 lignes" ou "Résultat: 12 lignes"
    tested_cols: list[str],
    strategy: str,
    model: str,
) -> str:
    """
    Retourne 3–6 recommandations concises pour FORCER un vrai changement (colonnes, opérateurs, regex, agrégations, JOIN, fenêtres...).
    """
    prompt = (
        "Tu es 'SQLSupervisor'. Objectif: améliorer la requête pour obtenir un échantillon exploitable.\n"
        f"Question: {question}\n"
        f"Colonnes disponibles: {columns}\n"
        f"Stratégie en cours: {strategy}\n"
        f"Dernière requête:\n{last_sql}\n"
        f"Résultat dernier essai: {outcome}\n"
        f"Colonnes déjà testées/rencontrées: {sorted(set(map(str.lower, tested_cols)))}\n\n"
        "Donne 3 à 6 puces d'actions CONCRÈTES ET DIFFÉRENTES à appliquer au prochain essai.\n"
        "Pas de SQL complet; uniquement des recommandations opérationnelles."
    )
    tips = call_mistral(client, prompt, model, max_tokens=300)
    return tips or ""

def _strip_sql_markdown(sql: str) -> str:
    if not sql:
        return sql
    s = sql.strip()
    s = re.sub(r"```(?:sql)?\s*", "", s, flags=re.I)
    s = re.sub(r"\s*```", "", s)
    s = re.sub(r"^\s*sql\s*", "", s, flags=re.I)
    return s.strip()

def _normalize_from_table(sql: str, table: str = "docs") -> str:
    # Remplace le premier FROM <qqch> par FROM docs (sans toucher aux sous-requêtes)
    return re.sub(r"(?is)\bfrom\s+['\"]?([A-Za-z0-9_]+)['\"]?", f"FROM {table}", sql, count=1)

def _ensure_single_limit(sql: str, limit: int) -> str:
    s = str(sql).strip()
    s = re.sub(r';+\s*$', '', s)
    s = re.sub(r'(?is)(?:\s+limit\s+\d+\s*)+$', '', s)  # retire tous les LIMIT finaux
    s = re.sub(r';+\s*$', '', s)
    return f"{s} LIMIT {limit}"

def clean_sql(raw_sql: str) -> str:
    m2 = re.search(r"```sql\s*(.*?)```", raw_sql, flags=re.IGNORECASE | re.DOTALL)
    sql = m2.group(1) if m2 else raw_sql
    sql = _strip_sql_markdown(sql)
    m = re.search(r"\b(SELECT|WITH)\b", sql, flags=re.IGNORECASE)
    sql = (sql[m.start():].strip() if m else sql.strip().strip("`"))
    sql = _normalize_from_table(sql, "docs")
    return sql

def extract_columns_from_sql(sql: str, all_columns: List[str]) -> List[str]:
    lower = sql.lower()
    return [col for col in all_columns if col.lower() in lower]

def anti_join_exclude_known(sql: str, known_ids: set[str], limit: int = 100) -> str:
    """Empêche de ressortir des dossiers déjà trouvés en enveloppant la requête."""
    if not known_ids:
        return _ensure_single_limit(sql, limit)
    # enlève un LIMIT éventuel en fin de requête pour éviter un double LIMIT
    sql_no_limit = re.sub(r"(?is)\blimit\s+\d+\s*;?\s*$", "", sql).strip()
    ids_csv = ", ".join("'" + i.replace("'", "''") + "'" for i in sorted(known_ids))
    wrapped = f"""
    WITH base AS (
        {sql_no_limit}
    )
    SELECT *
    FROM base
    WHERE id_dossier NOT IN ({ids_csv})
    LIMIT {limit}
    """
    return wrapped

def ensure_id_and_limit(sql: str, limit: int = 50) -> str:
    """Force la présence d'id_dossier dans le SELECT et ajoute LIMIT unique en fin."""
    has_id = re.search(r"(?is)\bid_dossier\b", sql) is not None
    if not has_id:
        if re.search(r"(?is)\bselect\s+distinct\b", sql):
            sql = re.sub(r"(?is)\bselect\s+distinct\s+", "SELECT DISTINCT id_dossier, ", sql, count=1)
        else:
            sql = re.sub(r"(?is)\bselect\s+", "SELECT id_dossier, ", sql, count=1)
    # LIMIT unique
    sql = _ensure_single_limit(sql, limit)
    return sql

def execute_with_content(con: duckdb.DuckDBPyConnection, sql: str) -> tuple[str, pd.DataFrame]:
    """
    Exécute la requête. Si 'content' n'est pas présent mais 'id_dossier' l'est,
    wrap la requête et JOIN docs USING(id_dossier) pour récupérer content.
    Retourne (sql_effectivement_exécuté, df).
    """
    df = con.execute(sql).fetchdf()
    cols_lower = {c.lower(): c for c in df.columns}
    if "content" in cols_lower:
        return sql, df

    if "id_dossier" in cols_lower:
        wrapped = f"""
        WITH base AS (
            {sql}
        )
        SELECT base.*, docs.content
        FROM base
        JOIN docs USING (id_dossier)
        """
        df2 = con.execute(wrapped).fetchdf()
        return wrapped, df2

    return sql, df

def _text_columns(df: pd.DataFrame) -> List[str]:
    # Cols textuelles, ordonnées par pertinence clinique si présentes
    prefer = [
        "Conclusion",
        "Proposition thérapeutique",
        "Titre de l'intervention",
        "Type d'imagerie",
        "Type histologique",
        "Motif de consultation",
        "Observations",
        "Résultat",
        "Evolution",
        "category",
        "title",
        "content",
    ]
    cols = [c for c in prefer if c in df.columns]
    if len(cols) < 12:
        other = [str(c) for c in df.columns if c not in cols and (pd.api.types.is_string_dtype(df[c]) or df[c].dtype == "object")]
        cols += other[:max(0, 12 - len(cols))]
    return cols[:12] if cols else [str(df.columns[0])]

def _tokens_from_question(question: str) -> List[str]:
    toks = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+", question.lower())
    toks = [t for t in toks if len(t) >= 3]
    seen = set()
    uniq = []
    for t in toks:
        if t not in seen:
            uniq.append(t)
            seen.add(t)
    return uniq[:6]

def build_like_where_from_question(question: str, df: pd.DataFrame) -> Tuple[str, List[str]]:
    """Transforme la question en tokens (>=3) et génère WHERE (AND de groupes OR) sur colonnes texte."""
    tokens = _tokens_from_question(question)
    cols = _text_columns(df)
    if not tokens:
        return "", cols
    groups = []
    for t in tokens:
        kw = t.replace("%", "%%").replace("_", "\\_")
        ors = [f"LOWER(\"{c}\") LIKE '%{kw.lower()}%'" for c in cols]
        groups.append("(" + " OR ".join(ors) + ")")
    where = " AND ".join(groups)
    return where, cols

def heuristic_sql(table: str, df: pd.DataFrame, question: str, limit: int, drop: int = 0, top1: bool = False) -> str:
    """Heuristique locale (sans LLM) : AND de mots-clés, OR entre colonnes texte, tri par date_doc desc."""
    where, cols = build_like_where_from_question(question, df)
    if drop and where:
        # enlève les derniers groupes AND pour élargir
        parts = re.split(r"\)\s+AND\s+\(", where.strip("()"))
        parts = ["(" + p + ")" for p in parts]
        keep = max(1, len(parts) - drop)
        where = " AND ".join(parts[:keep])
    base = f"SELECT id_dossier, * FROM {table}"
    if where:
        base += f" WHERE {where}"
    order = ' ORDER BY TRY_CAST("date_doc" AS TIMESTAMP) DESC NULLS LAST' if 'date_doc' in df.columns else ""
    lim = " LIMIT 1" if top1 else f" LIMIT {limit}"
    return base + order + lim

def enrich_with_agent(
    client: Mistral,
    content: str,
    question: str,
    model: str
) -> Optional[str]:
    snippet = "\n".join(content.splitlines()[:150])
    prompt = (
        f"Voici un extrait de la colonne 'content' (max 15 lignes) :\n{snippet}\n"
        f"À partir de cet extrait, extrais les informations pertinentes pour répondre à la question : \"{question}\". "
        "Réponds uniquement par ces informations. Justifie un minimum."
    )
    return call_mistral(client, prompt, model="mistral-large-2411")

def run_sql_cycles(
    con: duckdb.DuckDBPyConnection,
    columns: List[str],
    question: str,
    client: Mistral,
    cfg: Config,
    context: str,
) -> tuple[list[tuple[str, List[tuple[Any, ...]], List[str]]], list[Attempt]]:
    """
    results: [(sql, rows(list of tuples), summaries)], pour rétro-compat.
    attempts: log structuré de *tous* les essais (succès/échec), profondeur comprise.
    """
    results: list[tuple[str, List[tuple[Any, ...]], List[str]]] = []
    attempts: list[Attempt] = []
    known_ids: set[str] = set()
    PER_QUERY_LIMIT = 100
    TARGET_TOTAL = 100

    for method in range(1, cfg.max_methods + 1):
        strategy = METHOD_STRATEGIES[(method - 1) % len(METHOD_STRATEGIES)]
        logging.info("==== MÉTHODE %d — %s ====", method, strategy)
        seen_sql = set()
        tested_cols: set[str] = set()
        prev_sql = prev_error = None
        prev_rows_df: Optional[pd.DataFrame] = None
        supervisor_hint = ""

        for depth in range(1, cfg.depth_retries + 1):
            print("<<MAIN_START>>")
            # 1) PROMPT SQL
            if depth == 1:
                prompt = (
                    f"{context}"
                    f"Table DuckDB 'docs' avec colonnes: {columns}.\n"
                    f"Méthode {method} (stratégie: {strategy}), essai {depth}.\n"
                    f"Écris une requête SQL STRICTEMENT pour répondre à: «{question}».\n"
                    "Contraintes: commence par SELECT/WITH, inclure id_dossier dans le SELECT, LIMIT <= 100.\n"
                    "Réponds uniquement par la requête."
                )
            else:
                last_used = extract_columns_from_sql(prev_sql or "", columns)
                tested_cols.update(last_used)
                outcome = f"Erreur: {prev_error}" if prev_error else (
                    f"0 lignes" if prev_rows_df is not None and prev_rows_df.empty else f"{len(prev_rows_df)} lignes"
                )
                supervisor_hint = review_sql_attempt(
                    client, question, columns, prev_sql or "", outcome,
                    list(tested_cols), strategy, cfg.model_name
                )
                known_ids_list = list(known_ids)
                known_ids_str = ", ".join(known_ids_list[:30])
                prompt = (
                    f"{context}"
                    f"Méthode {method} (stratégie: {strategy}), essai {depth}.\n"
                    f"Dossiers déjà trouvés (exclus): {len(known_ids)}  (extrait: {known_ids_str})\n"
                    f"Requête précédente:\n{prev_sql}\n"
                    f"Recommandations du SQLSupervisor:\n{supervisor_hint}\n"
                    "Produis UNE NOUVELLE requête SQL *différente* "
                    "qui explore d'autres colonnes/conditions/regex/join et évite les IDs déjà trouvés.\n"
                    "Réponds uniquement par la requête. N'oublie pas id_dossier et LIMIT."
                )

            raw_sql = call_mistral(client, prompt, cfg.model_name)
            print("<<MAIN_END>>")
            # 1bis) si pas de SQL LLM → heuristique directe
            candidate_sqls: List[Tuple[str, str]] = []  # (label, sql)
            if raw_sql:
                sql = clean_sql(raw_sql)
                sql = ensure_id_and_limit(sql, limit=PER_QUERY_LIMIT)
                sql = anti_join_exclude_known(sql, known_ids, limit=PER_QUERY_LIMIT)
                candidate_sqls.append(("llm", sql))
            else:
                logging.warning("LLM n'a pas renvoyé de SQL, on va tenter l'heuristique.")

            # Heuristiques (de la plus stricte à la plus large)
            # On échantillonne le DF pour récupérer la signature (colonnes)
            df_sig = con.execute("SELECT * FROM docs LIMIT 1").fetchdf()
            candidate_sqls.append(("heuristic_top1", heuristic_sql("docs", df_sig, question, PER_QUERY_LIMIT, drop=0, top1=True)))
            candidate_sqls.append(("heuristic", heuristic_sql("docs", df_sig, question, PER_QUERY_LIMIT, drop=0, top1=False)))
            candidate_sqls.append(("heuristic_relax1", heuristic_sql("docs", df_sig, question, PER_QUERY_LIMIT, drop=1, top1=False)))
            candidate_sqls.append(("fallback_all", f"SELECT id_dossier, * FROM docs ORDER BY TRY_CAST(\"date_doc\" AS TIMESTAMP) DESC NULLS LAST LIMIT {PER_QUERY_LIMIT}"))

            # Exécute chaque candidate jusqu'à succès (>=1 ligne)
            tried_this_depth = False
            for label, sql in candidate_sqls:
                print("<<SQL_START>>")
                if sql in seen_sql:
                    print("<<SQL_END>>")
                    continue
                seen_sql.add(sql)
                logging.info("Exécution SQL (%s) (m%d, essai %d): %s", label, method, depth, sql)
                try:
                    exec_sql, res_df = execute_with_content(con, sql)
                    prev_error = None
                except Exception as e:
                    prev_error = str(e)[:400]
                    attempts.append(Attempt(
                        method, depth, f"{strategy} [{label}]", prompt, sql, prev_error, None, [], [], supervisor_hint, []
                    ))
                    prev_sql = sql
                    prev_rows_df = None
                    print("<<SQL_END>>")
                    continue

                prev_sql = exec_sql
                prev_rows_df = res_df

                ids = []
                cols_lower = {c.lower(): c for c in (res_df.columns if res_df is not None else [])}
                if res_df is not None and "id_dossier" in cols_lower:
                    ids = res_df[cols_lower["id_dossier"]].astype(str).tolist()

                row_count = int(len(res_df)) if res_df is not None else 0
                new_ids = [i for i in ids if i not in known_ids]
                known_ids.update(new_ids)

                # Enrichissement "agent secondaire" SUR CONTENT (échantillon limité)
                summaries: list[str] = []
                if res_df is not None and "content" in cols_lower and row_count > 0:
                    content_col = cols_lower["content"]
                    agent_calls = 0
                    for content in res_df[content_col].astype(str):
                        if agent_calls >= MAX_AGENT_CALLS:
                            break
                        if content and content.strip():
                            print("<<AGENT_START>>")
                            s = enrich_with_agent(client, content, question, cfg.model_name)
                            agent_calls += 1
                            if s:
                                summaries.append(s)
                            print("<<AGENT_END>>")

                attempts.append(Attempt(
                    method=method,
                    depth=depth,
                    strategy=f"{strategy} [{label}]",
                    prompt=prompt,
                    sql=prev_sql,
                    error=None if row_count > 0 else "0 lignes",
                    row_count=row_count,
                    ids=ids[:100],
                    sample=_preview_rows(prev_rows_df),
                    supervisor_hint=supervisor_hint,
                    agent_summaries=summaries,
                ))
                print("<<SQL_END>>")

                tried_this_depth = True
                if row_count > 0:
                    # Succès → on garde ce résultat pour cette méthode et on passe à la suivante
                    results.append((prev_sql, res_df.itertuples(index=False), summaries))
                    break  # sort de candidate_sqls

            if not tried_this_depth:
                # Rien n'a tourné (cas extrême), on passe à la profondeur suivante
                continue

            if prev_rows_df is not None and len(prev_rows_df) > 0:
                # succès à cette profondeur → on passe à la méthode suivante
                break

    return results, attempts

def format_answer_with_agent(answer: str, question: str, client: Mistral, model: str) -> Optional[str]:
    prompt = (
        f"Formate la réponse suivante pour répondre clairement à la question “{question}”:\n{answer}\n"
        "Réponds de manière concise, structurée et lisible."
        "Précise bien systématiquement les id_dossier des lignes qui ont permis de répondre."
    )
    return call_mistral(client, prompt, model)

def get_id_with_agent(answer: str, question: str, client: Mistral, model: str) -> Optional[str]:
    prompt = (
        f"Trouve toutes les ID_dossier de cette réponse:\n{answer}\n"
        "Réponds avec la liste des ID_dossier de la réponse uniquement, sans texte, que des nombres"
    )
    return call_mistral(client, prompt, model)

def format_from_intermediates(
    client: Mistral,
    question: str,
    attempts: list[Attempt],
    model: str,
) -> Optional[str]:
    """
    Envoie un JSON compact de *tous* les essais au formateur final.
    On coupe pour rester token-safe: on ne garde que les essais avec lignes > 0,
    on limite ids/snippets.
    """
    useful = [a for a in attempts if (a.row_count or 0) > 0]
    if not useful:
        useful = attempts[-3:]  # derniers essais à titre d'info

    payload = []
    for a in useful[:8]:  # limite le nombre d'essais transférés
        payload.append({
            "method": a.method,
            "depth": a.depth,
            "strategy": a.strategy,
            "sql": a.sql,
            "row_count": a.row_count,
            "ids": a.ids[:50],
            "snippets": a.sample[:5],
            "summaries": a.agent_summaries[:5],
        })

    json_blob = json.dumps(payload, ensure_ascii=False)
    # TUNING PROMPT
    prompt = (
        "Tu es 'FormattingAgent'. Compose la MEILLEURE réponse finale à partir d'essais intermédiaires.\n"
        "EXIGENCES:\n"
        "1) Appuie-toi uniquement sur les données/kernels fournis.\n"
        "2) Structure en listes/bullets si pertinent.\n"
        "3) Cite explicitement les id_dossier utilisés.\n"
        "4) Termine par une ligne unique au format exact:  IDs: id1, id2, ...\n\n"
        f"Question: {question}\n\n"
        "Essais intermédiaires (JSON):\n"
        f"```json\n{json_blob}\n```\n"
        "Donne uniquement la réponse finale formatée (pas d'explications techniques)."
        f"Trouve la langue de {question} et traduit IMPÉRATIVEMENT la réponse dans cette langue"
    )
    return call_mistral(client, prompt, model="mistral-large-2411", max_tokens=1200)

def main(question: str, cfg: Config) -> Any:
    logging.info("Démarrage du pipeline Excel-QA Agent via SQL")
    client = Mistral(api_key=cfg.api_key)

    # Chargement et nettoyage du fichier Excel
    df = pd.read_excel(cfg.excel_path)
    # Normalise les colonnes pour éviter des erreurs (espaces, etc. gardés mais trim)
    df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]
    if "content" in df.columns:
        df["content"] = df["content"].astype(str).str.strip()
    for col in df.columns[df.columns.str.contains('date', case=False)]:
        df[col] = df[col].replace('', None)

    # --- Génération du contexte : schéma + statistiques descriptives ---
    schema_lines = [f"- {col}: type={dtype}" for col, dtype in zip(df.columns, df.dtypes)]
    schema = "\n".join(schema_lines)

    stats_lines = []
    for col in df.columns:
        null_pct = df[col].isna().mean()
        dtype = df[col].dtype
        line = f"- {col}: type={dtype}, nulls={null_pct:.1%}"
        if pd.api.types.is_numeric_dtype(df[col]):
            mean = df[col].mean()
            median = df[col].median()
            line += f", mean={mean:.2f}, median={median:.2f}"
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            min_date = df[col].min()
            max_date = df[col].max()
            line += f", range={min_date} to {max_date}"
        else:
            top_vals = df[col].value_counts(dropna=True).head(5).to_dict()
            line += f", top5={top_vals}"
        stats_lines.append(line)
    stats = "\n".join(stats_lines)

    context = f"Schéma de la table 'docs':\n{schema}\n\nStatistiques descriptives:\n{stats}\n\n"

    # --- Exécution des cycles SQL ---
    with duckdb.connect(database=":memory:") as con:
        con.register("docs", df)
        columns = df.columns.tolist()
        results, attempts = run_sql_cycles(con, columns, question, client, cfg, context)

    # 1) Réponse brute rétro-compat (optionnel)
    if results:
        raw_blocks = []
        for sql, rows, summaries in results:
            rows_list = list(rows)
            raw_blocks.append(f"SQL: {sql}\nRésultats: {rows_list}\nSummaries: {summaries}")
        raw_answer = "\n---\n".join(raw_blocks)
    else:
        raw_answer = "Aucune réponse exploitable trouvée."

    # 2) Réponse finale via tous les intermédiaires
    print("<<AGENT_START>>")
    formatted = format_from_intermediates(client, question, attempts, cfg.model_name)
    print("<<AGENT_END>>")
    return formatted or raw_answer

if __name__ == "__main__":
    cfg = load_config()
    client = Mistral(api_key=cfg.api_key)
    question = input("Entrez votre question : ")
    final_text = main(question, cfg)
    id_list = get_id_with_agent(final_text, question, client, cfg.model_name)
    print("<<FINAL_RESPONSE_START>>")
    print(final_text)
    print("<<FINAL_RESPONSE_END>>")
    print("<<ID_START>>")
    print(id_list if id_list else final_text)
    print("<<ID_END>>")