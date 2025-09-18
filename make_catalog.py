# make_catalog.py
# -*- coding: utf-8 -*-
import json
import re
from pathlib import Path

raw_text = """Biologie

Date
Hémoglobine
Leucocytes
Polynucléaires neutrophiles
Plaquettes
D.F.G. selon la formule de CKD-EPI
Sodium
Potassium
Transaminases ASAT (SGOT)
Transaminases ALAT (SGPT)
Phosphatases alcalines (PAL)
Bilirubine Totale
 
 
RCP

Date
Localisation
Circonstances de la découverte
Histologie
Facteurs histo-pronostiques
OMS
Avis de la RCP
 
 
Ordonnance
Date
Prescripteur
Médicaments
 
Hospitalisation
Date
Type : conventionnel, HDJ
Service : chirurgie, médecine, soins de support
Numéro de la cure
Protocole
Évènements intercurrents
OMS
Chimiothérapie prescrite
Autres médicaments
 
Imagerie
Date
Type d’imagerie : scanner, IRM, PET-scan, échographie
Localisation (abdomen, thorax, cérébral, sein, thyroïde, autre)
Résultat :normal, suspicion de métastase, métastase avérée
Evolution: stable, régression, progression

Histologie
Date
Observations
Conclusion

Séance de radiothérapie
Date
Localisation
 
Endoscopie
Date
Titre de l’intervention
Chirurgien
 
Télésurveillance
Date
Observations
Conclusions

Consultation
Date
Motif de consultation
Conclusion
"""

# Catégories attendues (telles qu'elles apparaissent dans le texte)
CATEGORIES = {
    "Biologie",
    "RCP",
    "Ordonnance",
    "Hospitalisation",
    "Imagerie",
    "Histologie",
    "Séance de radiothérapie",
    "Endoscopie",
    "Télésurveillance",
    "Consultation",
}

def clean_field_label(label: str) -> str:
    """Garde uniquement le nom du champ (avant ':' et sans contenu entre parenthèses)."""
    # normaliser espaces (y compris les NBSP)
    s = label.replace("\u00a0", " ").strip()
    # couper ce qui suit un ':' (ex: 'Type : ...' -> 'Type')
    if ":" in s:
        s = s.split(":", 1)[0].strip()
    # retirer un éventuel bloc final entre parenthèses (ex: 'Localisation (abdomen, ...)' -> 'Localisation')
    s = re.sub(r"\s*\(.*?\)\s*$", "", s)
    # compacter les espaces multiples
    s = re.sub(r"\s{2,}", " ", s)
    return s

def build_catalog(text: str) -> dict:
    catalog = {}
    current_cat = None

    for raw_line in text.splitlines():
        line = raw_line.replace("\u00a0", " ").strip()
        if not line:
            continue  # ignorer les lignes vides

        # nouvelle catégorie ?
        if line in CATEGORIES:
            current_cat = line
            catalog[current_cat] = []
            continue

        # si on est dans une catégorie, enregistrer le champ
        if current_cat is not None:
            field = clean_field_label(line)
            if field:  # éviter d'ajouter des vides
                # Format EXACT souhaité: une chaîne "{'name': '...'}"
                catalog[current_cat].append(f"{{'name': '{field}'}}")

    return catalog

if __name__ == "__main__":
    catalog = build_catalog(raw_text)

    out_path = Path("fields_catalog_from_text.json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)

    print(f"✅ Fichier écrit: {out_path.resolve()}")
