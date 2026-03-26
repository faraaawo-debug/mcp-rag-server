import os
from pypdf import PdfReader
import chromadb
from sentence_transformers import SentenceTransformer


model = SentenceTransformer("BAAI/bge-small-en-v1.5") # On charge le modèle d'embeddings
client=chromadb.PersistentClient(path="./chroma_db") # Initialisation de ChromaDB
collection=client.get_or_create_collection(name="docs") # On crée une collection pour les documents

#Lit les fichiers PDF et TXT du dossier docs/
def charger_doc(doc) :
    documents = []
    for filename in os.listdir(doc):
        filepath = os.path.join(doc, filename)
        if filename.endswith(".pdf"):
            reader = PdfReader(filepath)
            text = " ".join(page.extract_text() for page in reader.pages)
            documents.append({"filename": filename, "text": text})
        elif filename.endswith(".txt"):
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            documents.append({"filename": filename, "text": text})
    return documents

#Découpe le texte en chunks avec chevauchement de 50 mots 
def chunk_texte(texte, taille_chunk=500, step=50):
    chunks=[]
    deb=0
    if step>= taille_chunk:
        raise ValueError("step doit être plus petit que taille_chunk")
    while deb< len(texte):
        fin= deb + taille_chunk
        chunk= texte[deb:fin]
        if chunk.strip():
            chunks.append(chunk)
        deb += taille_chunk-step
    return chunks

def charger_documents():
    print("Chargement des documents...")
    documents=charger_doc("./docs")
    
    if not documents:
        print("Aucun document trouvé dans le dossier docs/")
        return
    for doc in documents:
        print(f"Traitement de {doc['filename']}...")
        chunks= chunk_texte(doc["text"])
        embeddings= model.encode(chunks)

        collection.add(documents= chunks, embeddings=embeddings,
            ids=[f"{doc['filename']}-chunk-{i}" for i in range(len(chunks))])
        print(f"{len(chunks)} chunks indexés pour {doc['filename']}")
    
    print("Documents chargés et indexés avec succès !")

if __name__ == "__main__":
    charger_documents()