"""Serveur MCP qui permet à un assistant IA d'interroger une base de documents."""
import asyncio
import logging

from mcp.server import MCPServer

import config
from rag_utils import get_model, rechercher, repondre

# Les logs vont dans un fichier et sur stderr : stdout est réservé au protocole MCP
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(config.LOG_PATH, encoding="utf-8"), logging.StreamHandler()],
)
logging.getLogger("httpx").setLevel(logging.WARNING)  # masque chaque requête HTTP vers les API
logger = logging.getLogger(__name__)

mcp = MCPServer("mcp-rag-server")

STATUTS = {
    "fiable": "FIABLE : la réponse s'appuie sur les documents",
    "a_verifier": "À VÉRIFIER : la réponse s'éloigne peut-être des documents",
    "refus": "PAS DE RÉPONSE : l'information n'a pas été trouvée dans les documents",
}


def formater_resultat(resultat):
    lignes = [resultat["reponse"], "", "---"]
    lignes.append("Sources : " + ", ".join(resultat["sources"]))
    statut = STATUTS[resultat["statut"]]
    if resultat["score_global"] is not None:
        statut += f" (score {resultat['score_global']:.2f})"
    lignes.append("Fiabilité : " + statut)
    lignes.append(f"Temps de génération : {resultat['latence_s']} s")
    return "\n".join(lignes)


@mcp.tool()
async def rechercher_documents(query: str) -> str:
    """Retourne les extraits de documents les plus pertinents pour une question, avec leur source.

    Args:
        query: La question ou le sujet à rechercher
    """
    logger.info(f"Recherche : {query}")
    # Les embeddings sont bloquants : on les exécute hors de la boucle asyncio
    passages = await asyncio.to_thread(rechercher, query)
    return "\n\n---\n\n".join(
        f"[{p['source']}, extrait {p['chunk']}, similarité {p['similarite']:.2f}]\n{p['texte']}"
        for p in passages
    )


@mcp.tool()
async def poser_question(query: str) -> str:
    """Répond à une question à partir des documents, en citant les sources
    et en indiquant un niveau de fiabilité.

    Args:
        query: La question ou le sujet à rechercher
    """
    logger.info(f"Question : {query}")
    # Le LLM et les embeddings sont bloquants : on les exécute hors de la boucle asyncio
    resultat = await asyncio.to_thread(repondre, query)
    logger.info(
        f"Statut : {resultat['statut']} | score : {resultat['score_global']} | "
        f"scores : {resultat['scores']} | latence : {resultat['latence_s']} s"
    )
    return formater_resultat(resultat)


if __name__ == "__main__":
    logger.info("Chargement du modèle d'embeddings...")
    get_model()  # chargé au démarrage pour que la première question ne soit pas lente
    logger.info(f"Démarrage du serveur MCP (LLM : {config.LLM_PROVIDER}/{config.LLM_MODEL})")
    mcp.run()  # transport stdio : Claude Desktop lance ce script et dialogue par stdin/stdout
