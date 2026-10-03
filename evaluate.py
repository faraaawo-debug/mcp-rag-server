"""Évalue le pipeline sur un jeu de questions annotées (eval_set.json).

Mesures :
- taux de recherche réussie : le bon document est-il dans les extraits récupérés ?
- exactitude : la réponse correspond-elle à la réponse attendue ? (LLM juge)
- fidélité : chaque affirmation est-elle justifiée par les extraits ? (LLM juge)
- refus corrects : le système dit-il "je ne sais pas" quand l'information n'existe pas ?
- temps de génération moyen
Le script propose aussi un seuil pour l'indicateur de fiabilité du serveur.
"""
import json
import statistics

import ollama

import config
from rag_utils import rechercher, repondre

JUGE_EXACTITUDE = """Compare la réponse proposée à la réponse attendue.
Question : {question}
Réponse attendue : {attendue}
Réponse proposée : {proposee}
La réponse proposée contient-elle l'information essentielle de la réponse attendue, sans la contredire ?
Réponds uniquement par OUI ou NON."""

JUGE_FIDELITE = """Voici des extraits de documents et une réponse.
Extraits :
{contexte}
Réponse : {reponse}
Chaque affirmation de la réponse est-elle justifiée par les extraits ?
Réponds uniquement par OUI ou NON."""


def juger(prompt):
    resultat = ollama.chat(
        model=config.LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},
    )
    return resultat["message"]["content"].strip().upper().startswith("OUI")


def charger_jeu():
    jeu = json.loads(config.EVAL_SET_PATH.read_text(encoding="utf-8"))
    for i, item in enumerate(jeu):
        if "A_COMPLETER" in json.dumps(item, ensure_ascii=False):
            raise ValueError(f"Question {i + 1} incomplète dans eval_set.json : remplacez les A_COMPLETER.")
    return jeu


def proposer_seuil(lignes):
    """Cherche le seuil qui sépare le mieux les bonnes réponses des mauvaises."""
    candidates = [l for l in lignes if l["score_global"] is not None]
    if len(candidates) < 4:
        return None
    meilleur = None
    for s in [x / 100 for x in range(50, 96)]:
        bien_classees = sum(
            (l["score_global"] >= s) == (l["exacte"] and l["fidele"]) for l in candidates
        )
        taux = bien_classees / len(candidates)
        if meilleur is None or taux > meilleur[1]:
            meilleur = (s, taux)
    return meilleur


def main():
    jeu = charger_jeu()
    lignes = []

    for item in jeu:
        passages = rechercher(item["question"])
        resultat = repondre(item["question"], passages)
        sources = {p["source"] for p in passages}
        contexte = "\n\n".join(p["texte"] for p in passages)

        ligne = {
            "question": item["question"],
            "repondable": item["repondable"],
            "reponse": resultat["reponse"],
            "statut": resultat["statut"],
            "score_global": resultat["score_global"],
            "latence_s": resultat["latence_s"],
        }
        if item["repondable"]:
            ligne["recherche_reussie"] = item["source_attendue"] in sources
            ligne["exacte"] = juger(JUGE_EXACTITUDE.format(
                question=item["question"], attendue=item["reponse_attendue"], proposee=resultat["reponse"]))
            ligne["fidele"] = resultat["statut"] != "refus" and juger(
                JUGE_FIDELITE.format(contexte=contexte, reponse=resultat["reponse"]))
        else:
            ligne["refus_correct"] = resultat["statut"] == "refus"
        lignes.append(ligne)
        print(f"- {item['question'][:70]} -> {resultat['statut']}")

    rep = [l for l in lignes if l["repondable"]]
    non_rep = [l for l in lignes if not l["repondable"]]
    pct = lambda vals: f"{100 * sum(vals) / len(vals):.0f} %" if vals else "n/a"

    resume = {
        "questions": len(lignes),
        "recherche_reussie": pct([l["recherche_reussie"] for l in rep]),
        "exactitude": pct([l["exacte"] for l in rep]),
        "fidelite": pct([l["fidele"] for l in rep]),
        "refus_corrects": pct([l["refus_correct"] for l in non_rep]),
        "latence_moyenne_s": round(statistics.mean(l["latence_s"] for l in lignes), 2),
    }
    seuil = proposer_seuil(rep)
    if seuil:
        resume["seuil_propose"] = seuil[0]
        resume["precision_indicateur"] = f"{100 * seuil[1]:.0f} %"

    config.RESULTATS_PATH.write_text(
        json.dumps({"resume": resume, "details": lignes}, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n| Mesure | Résultat |\n|---|---|")
    for cle, valeur in resume.items():
        print(f"| {cle} | {valeur} |")
    print(f"\nDétails enregistrés dans {config.RESULTATS_PATH.name}")


if __name__ == "__main__":
    main()
