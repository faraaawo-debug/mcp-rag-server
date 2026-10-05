"""Évalue le pipeline sur un jeu de questions annotées (eval_set.json).

Usage :
  python evaluate.py --etiquette reference          # évaluation complète
  python evaluate.py --etiquette essai --recherche-seule   # recherche uniquement (~1 min, sans LLM)
  python evaluate.py --etiquette reference --rejuger resultats/reference.json
      # rejuge des réponses déjà générées, sans les regénérer

Mesures :
- recherche (k fixes, indépendants du pipeline) : bon fichier dans les k premiers extraits,
  phrase clé dans les k premiers extraits, MRR de la phrase clé
- routeur : taux d'intentions correctement détectées (n/a tant qu'il n'y a pas de routeur)
- réponses : exactitude et fidélité (LLM juge), refus corrects, temps de génération moyen
Le script propose aussi un seuil pour l'indicateur de fiabilité du serveur.
"""
import argparse
import json
import re
import statistics

import ollama

import config
from rag_utils import est_un_refus, rechercher, repondre

# Prompts en anglais (langue des documents et des questions) et sortie JSON imposée :
# un verdict unique, lisible sans ambiguïté.
JUGE_EXACTITUDE = """You are grading an answer against a reference answer.
Question: {question}
Reference answer: {attendue}
Proposed answer: {proposee}

Does the proposed answer contain the essential information of the reference answer, without contradicting it?
An answer that says the information is not available, or that misses the essential point, is NOT correct.
Reply with JSON only, exactly one of: {{"verdict": "YES"}} or {{"verdict": "NO"}}"""

JUGE_FIDELITE = """You are checking whether an answer is supported by document excerpts.
Excerpts:
{contexte}

Answer: {reponse}

Is EVERY claim in the answer supported by the excerpts? If at least one claim is not supported, the verdict is NO.
Reply with JSON only, exactly one of: {{"verdict": "YES"}} or {{"verdict": "NO"}}"""


def juger(prompt):
    """Pose une question fermée au modèle juge.
    Retourne (verdict, texte brut) ; verdict vaut None si la sortie est illisible."""
    resultat = ollama.chat(
        model=config.JUGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        format="json",
        options={"temperature": 0},
    )
    brut = resultat["message"]["content"].strip()
    try:
        verdict = str(json.loads(brut).get("verdict", "")).strip().upper()
    except (json.JSONDecodeError, AttributeError):
        return None, brut
    return {"YES": True, "NO": False}.get(verdict), brut


def normaliser(texte):
    """Minuscules, sans espaces ni ponctuation : la phrase clé est retrouvée même si
    l'extraction du PDF a collé ou séparé des mots ("toshare" / "to share")."""
    return re.sub(r"[\W_]+", "", texte.lower())


def charger_jeu():
    return json.loads(config.EVAL_SET_PATH.read_text(encoding="utf-8"))


def mesurer_recherche(item):
    """Rang (1, 2, ...) du bon fichier et du premier extrait contenant la phrase clé,
    parmi les max(K_EVAL) premiers extraits. None si absent."""
    passages = rechercher(item["question"], k=max(config.K_EVAL))
    cle = normaliser(item["phrase_cle"])
    rang_fichier = next(
        (i + 1 for i, p in enumerate(passages) if p["source"] == item["source_attendue"]), None)
    rang_cle = next(
        (i + 1 for i, p in enumerate(passages)
         if p["source"] == item["source_attendue"] and cle in normaliser(p["texte"])), None)
    return rang_fichier, rang_cle


def juger_ligne(ligne, item):
    """Ajoute à une ligne de résultat les verdicts d'exactitude et de fidélité."""
    if not item["repondable"]:
        return
    ligne["exacte"], ligne["juge_exactitude"] = juger(JUGE_EXACTITUDE.format(
        question=item["question"], attendue=item["reponse_attendue"], proposee=ligne["reponse"]))
    if ligne["statut"] == "refus":
        ligne["fidele"], ligne["juge_fidelite"] = None, None  # un refus n'affirme rien
    else:
        contexte = "\n\n".join(p["texte"] for p in ligne["passages"])
        ligne["fidele"], ligne["juge_fidelite"] = juger(
            JUGE_FIDELITE.format(contexte=contexte, reponse=ligne["reponse"]))


def proposer_seuil(lignes):
    """Cherche le seuil qui sépare le mieux les bonnes réponses des mauvaises."""
    candidates = [l for l in lignes if l["score_global"] is not None]
    if len(candidates) < 4:
        return None
    meilleur = None
    for s in [x / 100 for x in range(50, 96)]:
        bien_classees = sum(
            (l["score_global"] >= s) == (l["exacte"] is True and l["fidele"] is True) for l in candidates
        )
        taux = bien_classees / len(candidates)
        if meilleur is None or taux > meilleur[1]:
            meilleur = (s, taux)
    return meilleur


def pct(valeurs):
    valeurs = list(valeurs)
    return f"{100 * sum(valeurs) / len(valeurs):.0f} % ({sum(valeurs)}/{len(valeurs)})" if valeurs else "n/a"


def resumer(etiquette, lignes, recherche_seule=False):
    rep = [l for l in lignes if l["repondable"]]
    resume = {"version": etiquette, "questions": len(lignes)}
    for k in config.K_EVAL:
        resume[f"fichier_trouve@{k}"] = pct(l["rang_fichier"] is not None and l["rang_fichier"] <= k for l in rep)
        resume[f"phrase_cle@{k}"] = pct(l["rang_cle"] is not None and l["rang_cle"] <= k for l in rep)
    resume[f"mrr_phrase_cle@{max(config.K_EVAL)}"] = round(
        statistics.mean(1 / l["rang_cle"] if l["rang_cle"] else 0 for l in rep), 3)
    if recherche_seule:
        return resume

    non_rep = [l for l in lignes if not l["repondable"]]
    avec_intention = [l for l in lignes if l["intention_detectee"] is not None]
    resume["intentions_correctes"] = pct(
        l["intention_detectee"] == l["intention_attendue"] for l in avec_intention)
    resume["exactitude"] = pct(l["exacte"] is True for l in rep)
    resume["fidelite_reponses_donnees"] = pct(l["fidele"] is True for l in rep if l["statut"] != "refus")
    resume["refus_corrects"] = pct(l["refus_correct"] for l in non_rep)
    resume["latence_moyenne_s"] = round(statistics.mean(l["latence_s"] for l in lignes), 2)
    resume["modele_juge"] = config.JUGE_MODEL
    # Un refus n'a volontairement pas de verdict de fidélité : il n'est pas compté ici
    resume["verdicts_illisibles"] = sum(l["exacte"] is None for l in rep) + sum(
        l["fidele"] is None for l in rep if l["statut"] != "refus")
    seuil = proposer_seuil([l for l in rep if l["statut"] != "refus"])
    if seuil:
        resume["seuil_propose"] = seuil[0]
        resume["precision_indicateur"] = f"{100 * seuil[1]:.0f} %"
    return resume


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--etiquette", default="dernier", help="nom de la version évaluée (ex : reference)")
    parser.add_argument("--recherche-seule", action="store_true",
                        help="mesure uniquement la recherche, sans appel au LLM")
    parser.add_argument("--rejuger", metavar="FICHIER",
                        help="rejuge les réponses d'un fichier de résultats, sans les regénérer")
    args = parser.parse_args()

    jeu = charger_jeu()
    if args.rejuger:
        anciennes = json.loads(open(args.rejuger, encoding="utf-8").read())["details"]
        if [l["question"] for l in anciennes] != [item["question"] for item in jeu]:
            raise ValueError("Le fichier à rejuger ne correspond pas au jeu d'évaluation actuel.")
        if any("passages" not in l for l in anciennes if l["repondable"]):
            raise ValueError("Ce fichier ne contient pas les extraits utilisés : impossible de le rejuger.")

    lignes = []
    for n, item in enumerate(jeu, 1):
        ligne = {
            "question": item["question"],
            "repondable": item["repondable"],
            "intention_attendue": item["intention_attendue"],
        }
        if item["repondable"]:
            ligne["rang_fichier"], ligne["rang_cle"] = mesurer_recherche(item)

        if not args.recherche_seule:
            if args.rejuger:
                ancienne = anciennes[n - 1]
                for cle in ("intention_detectee", "reponse", "statut", "score_global",
                            "latence_s", "sources", "passages"):
                    ligne[cle] = ancienne[cle]
            else:
                resultat = repondre(item["question"])
                ligne.update({
                    "intention_detectee": resultat.get("intention"),  # absente tant qu'il n'y a pas de routeur
                    "reponse": resultat["reponse"],
                    "statut": resultat["statut"],
                    "score_global": resultat["score_global"],
                    "latence_s": resultat["latence_s"],
                    "sources": resultat["sources"],
                    # extraits réellement donnés au LLM : permettent de rejuger sans regénérer
                    "passages": [{k: p[k] for k in ("source", "chunk", "similarite", "texte")}
                                 for p in resultat["passages"]],
                })
            if item["repondable"]:
                juger_ligne(ligne, item)
            else:
                ligne["refus_correct"] = est_un_refus(ligne["reponse"])
            print(f"[{n}/{len(jeu)}] {item['question'][:60]} -> {ligne['statut']}", flush=True)
        lignes.append(ligne)

    resume = resumer(args.etiquette, lignes, args.recherche_seule)
    config.RESULTATS_DIR.mkdir(exist_ok=True)
    chemin = config.RESULTATS_DIR / f"{args.etiquette}.json"
    chemin.write_text(json.dumps({"resume": resume, "details": lignes}, ensure_ascii=False, indent=2),
                      encoding="utf-8")

    print("\n| Mesure | Résultat |\n|---|---|")
    for cle, valeur in resume.items():
        print(f"| {cle} | {valeur} |")
    print(f"\nDétails enregistrés dans {chemin.relative_to(config.BASE_DIR)}")


if __name__ == "__main__":
    main()
