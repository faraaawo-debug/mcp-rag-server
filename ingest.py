"""Indexes the PDF and TXT documents of docs/<subject>/<type>/ into ChromaDB."""
import logging

import chromadb
import pymupdf

import config
from rag_utils import embed_passages

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)  # hides the model download requests
logger = logging.getLogger(__name__)

EXTENSIONS = {".pdf", ".txt"}


def read_text(path):
    if path.suffix.lower() == ".pdf":
        # PyMuPDF rather than pypdf: on this corpus, pypdf glued words together
        # ("thematter", "efficiently.Despite"), PyMuPDF did not
        with pymupdf.open(path) as document:
            return "\n".join(page.get_text() for page in document)
    return path.read_text(encoding="utf-8")


def classify(path):
    """Reads the subject and document type from docs/<subject>/<type>/<file>.
    A file stored elsewhere is still indexed, with the value "unknown"."""
    folders = path.relative_to(config.DOCS_DIR).parts[:-1]
    if len(folders) == 2 and folders[1] in config.DOC_TYPES:
        return folders[0], folders[1]
    logger.warning(
        f"{path.relative_to(config.DOCS_DIR)}: location does not match "
        f"docs/<subject>/<type>/ (types: {', '.join(config.DOC_TYPES)}), "
        f"subject and type set to \"{config.UNKNOWN}\""
    )
    return config.UNKNOWN, config.UNKNOWN


def load_documents():
    """Reads the PDF and TXT files of docs/ and all its subfolders."""
    documents = []
    for path in sorted(config.DOCS_DIR.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue
        text = read_text(path)
        if not text.strip():
            logger.warning(f"{path.name}: no text extracted, file skipped")
            continue
        subject, doc_type = classify(path)
        documents.append({
            "source": path.name,
            "path": path.relative_to(config.DOCS_DIR).as_posix(),
            "subject": subject,
            "doc_type": doc_type,
            "text": text,
        })
    return documents


def chunk_by_words(text, size=config.CHUNK_WORDS, overlap=config.OVERLAP_WORDS):
    """Splits the text into excerpts of `size` words, with `overlap` words shared by two
    consecutive excerpts (no word is cut in half)."""
    if overlap >= size:
        raise ValueError("the overlap must be smaller than the size")
    words = text.split()
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return chunks


def main():
    documents = load_documents()
    if not documents:
        print(f"No document found in {config.DOCS_DIR}")
        return

    # Start from an empty collection: re-running the ingestion creates no duplicates
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
        chunks = chunk_by_words(doc["text"])
        collection.add(
            # The relative path makes the id unique even if two folders
            # contain a file with the same name
            ids=[f"{doc['path']}-{i}" for i in range(len(chunks))],
            documents=chunks,
            embeddings=embed_passages(chunks).tolist(),
            metadatas=[
                {
                    "source": doc["source"],
                    "path": doc["path"],
                    "subject": doc["subject"],
                    "doc_type": doc["doc_type"],
                    "chunk": i,
                }
                for i in range(len(chunks))
            ],
        )
        total += len(chunks)
        print(f"{doc['path']} [{doc['subject']} / {doc['doc_type']}]: {len(chunks)} excerpts indexed")

    print(f"Done: {total} excerpts indexed for {len(documents)} documents.")


if __name__ == "__main__":
    main()
