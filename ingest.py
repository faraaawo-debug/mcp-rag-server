"""Indexe les documents PDF et TXT du dossier docs/ dans ChromaDB."""
import chromadb
from pypdf import PdfReader

import config
from rag_utils import embed_passages


def charger_documents():
    """Lit les fichiers PDF et TXT du dossier docs/."""
    documents = []
    for chemin in sorted(config.DOCS_DIR.iterdir()):
        if chemin.suffix.lower() == ".pdf":
            lecteur = PdfReader(chemin)
            texte = " ".join((page.extract_text() or "") for page in lecteur.pages)
        elif chemin.suffix.lower() == ".txt":
            texte = chemin.read_text(encoding="utf-8")
        else:
            continue
        if texte.strip():
            documents.append({"source": chemin.name, "texte": texte})
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
            ids=[f"{doc['source']}-{i}" for i in range(len(extraits))],
            documents=extraits,
            embeddings=embed_passages(extraits).tolist(),
            metadatas=[{"source": doc["source"], "chunk": i} for i in range(len(extraits))],
        )
        total += len(extraits)
        print(f"{doc['source']} : {len(extraits)} extraits indexés")

    print(f"Terminé : {total} extraits indexés pour {len(documents)} documents.")


if __name__ == "__main__":
    main()
