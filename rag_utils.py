"""Briques communes : embeddings, recherche, génération et indicateur de fiabilité."""
import time

import chromadb
import numpy as np
import ollama
from sentence_transformers import SentenceTransformer

import config

_model = None


def get_model():
    """Charge le modèle d'embeddings une seule fois."""
    global _model
    if _model is None:
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def embed_passages(textes):
    return get_model().encode([f"passage: {t}" for t in textes], normalize_embeddings=True)


def embed_queries(textes):
    # Préfixe "query: " aussi pour comparer une réponse à un texte (usage symétrique)
    return get_model().encode([f"query: {t}" for t in textes], normalize_embeddings=True)


def get_collection():
    client = chromadb.PersistentClient(path=str(config.CHROMA_PATH))
    return client.get_or_create_collection(
        name=config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )


def rechercher(question, k=config.TOP_K):
    """Retourne les k extraits les plus proches de la question, avec leur source."""
    collection = get_collection()
    vecteur = embed_queries([question])[0].tolist()
    res = collection.query(
        query_embeddings=[vecteur],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    passages = []
    for texte, meta, distance in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        passages.append({
            "texte": texte,
            "source": meta["source"],
            "chunk": meta["chunk"],
            "similarite": round(1 - distance, 3),  # distance cosinus -> similarité
        })
    return passages


PROMPT = """Tu es un assistant qui répond uniquement à partir des extraits fournis.
Règles :
- Utilise seulement les informations présentes dans les extraits.
- Si la réponse ne s'y trouve pas, réponds exactement : "Je ne sais pas."
- Réponds dans la langue de la question.
- Cite entre crochets le fichier source de chaque information, par exemple [rapport.pdf].

Extraits :
{contexte}

Question : {question}
"""

REFUS = ("je ne sais pas", "i don't know", "i do not know")


def est_un_refus(reponse):
    return any(r in reponse.lower() for r in REFUS)


def generer_reponse(question, passages):
    """Appelle le LLM local et mesure le temps de réponse."""
    contexte = "\n\n".join(f"[{p['source']}] {p['texte']}" for p in passages)
    debut = time.perf_counter()
    resultat = ollama.chat(
        model=config.LLM_MODEL,
        messages=[{"role": "user", "content": PROMPT.format(contexte=contexte, question=question)}],
        options={"temperature": 0},  # réponses reproductibles
    )
    latence = time.perf_counter() - debut
    return resultat["message"]["content"].strip(), latence


def scores_similarite(question, reponse, passages):
    """Trois scores de similarité cosinus (entre 0 et 1) qui servent d'indicateur rapide.
    Ce ne sont pas des mesures de vérité : la vraie évaluation est dans evaluate.py."""
    v_question, v_reponse = embed_queries([question, reponse])
    v_passages = embed_queries([p["texte"] for p in passages])
    ancrage = float(np.max(v_passages @ v_reponse))           # réponse proche d'au moins un extrait ?
    contexte = float(np.mean([p["similarite"] for p in passages]))  # extraits proches de la question ?
    pertinence = float(v_question @ v_reponse)                # réponse proche de la question ?
    return {
        "ancrage_documents": round(ancrage, 3),
        "pertinence_extraits": round(contexte, 3),
        "pertinence_reponse": round(pertinence, 3),
    }


def repondre(question, passages=None):
    """Pipeline complet : recherche, génération, indicateur de fiabilité."""
    if passages is None:
        passages = rechercher(question)
    reponse, latence = generer_reponse(question, passages)

    if est_un_refus(reponse):
        # Un refus est un comportement correct, pas une réponse hors documents
        statut, scores, score_global = "refus", None, None
    else:
        scores = scores_similarite(question, reponse, passages)
        score_global = round(sum(scores.values()) / 3, 3)
        statut = "fiable" if score_global >= config.SEUIL_FIABILITE else "a_verifier"

    return {
        "reponse": reponse,
        "sources": sorted({p["source"] for p in passages}),
        "passages": passages,
        "scores": scores,
        "score_global": score_global,
        "statut": statut,
        "latence_s": round(latence, 2),
    }
