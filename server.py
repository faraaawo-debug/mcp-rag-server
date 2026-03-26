import asyncio
import logging
import chromadb
from sentence_transformers import SentenceTransformer
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types
import numpy as np
import ollama


model = SentenceTransformer("BAAI/bge-small-en-v1.5")
chroma_client= chromadb.PersistentClient(path="./chroma_db")
collection= chroma_client.get_or_create_collection(name="docs")
server= Server("mcp-rag-server")


# Configuration du logging 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("server.log"),
        logging.StreamHandler()
    ]
)
logger= logging.getLogger(__name__)


# Mesure si la réponse est fidèle aux chunks récupérés. Score entre 0 et 1. Plus le score est élevé, plus la réponse 
# est ancrée dans les documents
def score_fidelite(reponse, chunks) :
    reponse_embedding= model.encode(reponse)
    chunk_embeddings= model.encode(chunks)
    scores= []
    for chq in chunk_embeddings:
        # calcul de la similarité cosinus entre la réponse et le chunk correspondant
        score= np.dot(reponse_embedding, chq)/(np.linalg.norm(reponse_embedding)* np.linalg.norm(chq))
        scores.append(score)
    return float(max(scores)) # on retourne le score le plus élevé

# Mesure si les chunks récupérés sont pertinents par rapport à la query
def context_score(query, chunks):
    query_embedding= model.encode(query)
    chunk_embeddings= model.encode(chunks)
    scores= []
    for chq in chunk_embeddings:
        score= np.dot(query_embedding, chq)/(np.linalg.norm(query_embedding)* np.linalg.norm(chq))
        scores.append(score)
    return float(np.mean(scores))  

#Mesure si la réponse est pertinente par rapport à la question posée.
def rep_score(query, reponse):
    query_embedding= model.encode(query)
    reponse_embedding= model.encode(reponse)
    score= np.dot(query_embedding, reponse_embedding)/(np.linalg.norm(query_embedding)* np.linalg.norm(reponse_embedding))
    return float(score)

@server.list_tools()
async def list_tools() :
    return [
        types.Tool(
            name="rechercher_documents",
            description="Recherche les passages les plus pertinents dans les documents",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "La question ou le sujet à rechercher dans les documents"
                    }
                },
                "required": ["query"]
            }
        ),
        types.Tool(
            name="poser_question",
            description="Pose une question et obtient une réponse basée sur les documents",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "La question à poser dans les documents"
                    }
                },
                "required": ["query"]
            }
        )
    ]

@server.call_tool()
async def call_tool(nom_outil, arguments) : # fonction pour exécuter l'outil demandé
    if nom_outil == "rechercher_documents":
        query= arguments["query"]
        logger.info(f"Recherche pour la requête : {query}") # Trace la requete dans le log
        query_embedding=model.encode(query).tolist() # vectorise la requête
        resultats= collection.query(query_embeddings=[query_embedding],n_results=3)
        passages= resultats["documents"][0]
        logger.info(f"{len(passages)} passages trouvés")
        reponse="\n\n---\n\n".join(passages)
        return [types.TextContent(type="text", text=reponse)]
    
    elif nom_outil == "poser_question":
        query =arguments["query"]
        logger.info(f"Question reçue : {query}")
        query_embedding=model.encode(query).tolist() 
        resultats= collection.query(query_embeddings=[query_embedding],n_results=3)
        passages= resultats["documents"][0]
        logger.info(f"{len(passages)} chunks récupérés")
        context = "\n\n".join(passages)
        
        # Ici on va construire le prompt pour la réponse de l'assistant
        prompt= f"""Tu es un assistant qui répond aux questions en te basant uniquement sur le contexte fourni.Répond uniquement avec les informations présentes dans le contexte.
Si l'information n'est pas dans le contexte, dis "Je ne sais pas".

Contexte :
{context}

Question : {query}

"""

        response = ollama.chat(model="mistral",messages=[{"role": "user", "content": prompt}] ) # on utilise ollama pour générer la réponse
        rep_llm= response["message"]["content"] # on récupère la réponse de l'assistant
        fidelite= score_fidelite(rep_llm, passages) # on calcule le score de fidélité
        pertinence_ctx = context_score(query, passages)
        pertinence_rep = rep_score(query, rep_llm)
        score_global = (fidelite + pertinence_ctx + pertinence_rep) /3

        logger.info(f"Faithfulness : {fidelite:.2f}")
        logger.info(f"Context relevance : {pertinence_ctx:.2f}")
        logger.info(f"Answer relevance : {pertinence_rep:.2f}")
        logger.info(f"Score global : {score_global:.2f}")

        if score_global < 0.5:
            logger.warning(f"Réponse potentiellement hors document, score : {score_global:.2f}")
            flag= "ATTENTION : Réponse potentiellement hors document."
        else:
            logger.info(f"Réponse fidèle au document, score : {score_global:.2f}")
            flag= "SUPER : Réponse fidèle au document."
        
        rep_llm = f"""{rep_llm}"""
        return [types.TextContent(type="text", text=rep_llm)]
    # outils inexistants
    else:
        logger.warning(f"Outil inconnu appelé : {nom_outil}")
        return [types.TextContent(type="text", text=f"Outil inconnu : {nom_outil}")]

async def main():
    logger.info("Démarrage du serveur MCP...")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())