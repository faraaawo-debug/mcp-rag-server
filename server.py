"""Serveur MCP qui permet à un assistant IA d'interroger une base de documents."""
import asyncio
import logging

from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server

import config
from rag_utils import get_model, rechercher, repondre

# Les logs vont dans un fichier et sur stderr : stdout est réservé au protocole MCP
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(config.LOG_PATH), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

server = Server("mcp-rag-server")

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


@server.list_tools()
async def list_tools():
    schema = {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "La question ou le sujet à rechercher"}},
        "required": ["query"],
    }
    return [
        types.Tool(
            name="rechercher_documents",
            description="Retourne les extraits de documents les plus pertinents pour une question, avec leur source.",
            inputSchema=schema,
        ),
        types.Tool(
            name="poser_question",
            description=(
                "Répond à une question à partir des documents, en citant les sources "
                "et en indiquant un niveau de fiabilité."
            ),
            inputSchema=schema,
        ),
    ]


@server.call_tool()
async def call_tool(nom_outil, arguments):
    question = arguments["query"]

    if nom_outil == "rechercher_documents":
        logger.info(f"Recherche : {question}")
        passages = await asyncio.to_thread(rechercher, question)
        texte = "\n\n---\n\n".join(
            f"[{p['source']}, extrait {p['chunk']}, similarité {p['similarite']:.2f}]\n{p['texte']}"
            for p in passages
        )
        return [types.TextContent(type="text", text=texte)]

    if nom_outil == "poser_question":
        logger.info(f"Question : {question}")
        # Le LLM et les embeddings sont bloquants : on les exécute hors de la boucle asyncio
        resultat = await asyncio.to_thread(repondre, question)
        logger.info(
            f"Statut : {resultat['statut']} | score : {resultat['score_global']} | "
            f"scores : {resultat['scores']} | latence : {resultat['latence_s']} s"
        )
        return [types.TextContent(type="text", text=formater_resultat(resultat))]

    logger.warning(f"Outil inconnu appelé : {nom_outil}")
    return [types.TextContent(type="text", text=f"Outil inconnu : {nom_outil}")]


async def main():
    logger.info("Chargement du modèle d'embeddings...")
    get_model()  # chargé au démarrage pour que la première question ne soit pas lente
    logger.info("Démarrage du serveur MCP")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
