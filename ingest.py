"""Indexe les documents PDF et TXT de docs/<matiere>/<type>/ dans ChromaDB."""
import logging

import chromadb
from pypdf import PdfReader

import config
from rag_utils import embed_passages

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)  # masque les requêtes de téléchargement du modèle
logger = logging.getLogger(__name__)

EXTENSIONS = {".pdf", ".txt"}


def lire_texte(chemin):
    if chemin.suffix.lower() == ".pdf":
        lecteur = PdfReader(chemin)
        return " ".join((page.extract_text() or "") for page in lecteur.pages)
    return chemin.read_text(encoding="utf-8")


def classer(chemin):
    """Déduit la matière et le type du document à partir de docs/<matiere>/<type>/<fichier>.
    Un fichier rangé ailleurs est quand même indexé, avec la valeur "unknown"."""
    dossiers = chemin.relative_to(config.DOCS_DIR).parts[:-1]
    if len(dossiers) == 2 and dossiers[1] in config.DOC_TYPES:
        return dossiers[0], dossiers[1]
    logger.warning(
        f"{chemin.relative_to(config.DOCS_DIR)} : emplacement non conforme à "
        f"docs/<matiere>/<type>/ (types : {', '.join(config.DOC_TYPES)}), "
        f"matière et type notés \"{config.INCONNU}\""
    )
    return config.INCONNU, config.INCONNU


def charger_documents():
    """Lit les fichiers PDF et TXT de docs/ et de tous ses sous-dossiers."""
    documents = []
    for chemin in sorted(config.DOCS_DIR.rglob("*")):
        if not chemin.is_file() or chemin.suffix.lower() not in EXTENSIONS:
            continue
        texte = lire_texte(chemin)
        if not texte.strip():
            logger.warning(f"{chemin.name} : aucun texte extrait, fichier ignoré")
            continue
        matiere, type_doc = classer(chemin)
        documents.append({
            "source": chemin.name,
            "chemin": chemin.relative_to(config.DOCS_DIR).as_posix(),
            "matiere": matiere,
            "type_doc": type_doc,
            "texte": texte,
        })
    return documents


def decouper_en_mots(texte, taille=config.CHUNK_WORDS, chevauchement=config.OVERLAP_WORDS):
    """Découpe le texte en extraits de `taille` mots, avec `chevauchement` mots en commun
    entre deux extraits consécutifs (aucun mot n'est coupé en deux)."""
    if chevauchement >= taille:
        raise ValueError("le chevauchement doit être plus petit que la taille")
    mots = texte.split()
    pas = taille - chevauchement
    extraits = []
    for debut in range(0, len(mots), pas):
        extraits.append(" ".join(mots[debut:debut + taille]))
        if debut + taille >= len(mots):
            break
    return extraits


def main():
    documents = charger_documents()
    if not documents:
        print(f"Aucun document trouvé dans {config.DOCS_DIR}")
        return

    # On repart d'une collection vide : relancer l'ingestion ne crée pas de doublons
    client = chromadb.PersistentClient(path=str(config.CHROMA_PATH))
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    total = 0
    for doc in documents:
        extraits = decouper_en_mots(doc["texte"])
        collection.add(
            # Le chemin relatif rend l'identifiant unique même si deux dossiers
            # contiennent un fichier du même nom
            ids=[f"{doc['chemin']}-{i}" for i in range(len(extraits))],
            documents=extraits,
            embeddings=embed_passages(extraits).tolist(),
            metadatas=[
                {
                    "source": doc["source"],
                    "chemin": doc["chemin"],
                    "matiere": doc["matiere"],
                    "type_doc": doc["type_doc"],
                    "chunk": i,
                }
                for i in range(len(extraits))
            ],
        )
        total += len(extraits)
        print(f"{doc['chemin']} [{doc['matiere']} / {doc['type_doc']}] : {len(extraits)} extraits indexés")

    print(f"Terminé : {total} extraits indexés pour {len(documents)} documents.")


if __name__ == "__main__":
    main()
