"""Paramètres partagés par l'ingestion, le serveur et l'évaluation."""
from pathlib import Path

# Chemins absolus : le serveur fonctionne quel que soit le dossier depuis lequel
# il est lancé (indispensable quand c'est Claude Desktop qui le démarre).
BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
CHROMA_PATH = BASE_DIR / "chroma_db"
EVAL_SET_PATH = BASE_DIR / "eval_set.json"
LOG_PATH = BASE_DIR / "server.log"

COLLECTION_NAME = "docs"

# Organisation attendue : docs/<matiere>/<type>/<fichier>
# La matière et le type de chaque extrait sont déduits de ce chemin.
DOC_TYPES = ("lectures", "tutorials", "exams", "assignments")
INCONNU = "unknown"

# Modèle d'embeddings multilingue (français + anglais), 512 tokens maximum.
# Les modèles E5 attendent les préfixes "query: " et "passage: ".
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

# LLM local servi par Ollama
LLM_MODEL = "mistral"

# Découpage : 200 mots par extrait, 40 mots de chevauchement
CHUNK_WORDS = 200
OVERLAP_WORDS = 40

# Nombre d'extraits récupérés par question
TOP_K = 3

# Évaluation : modèle juge différent du modèle qui génère (limite l'auto-préférence)
JUGE_MODEL = "llama3"
# La recherche est toujours mesurée sur ces k, quel que soit le k utilisé par le pipeline,
# pour que les mesures restent comparables d'une version à l'autre
K_EVAL = (1, 5)
RESULTATS_DIR = BASE_DIR / "resultats"

# Seuil de l'indicateur de fiabilité : à recalibrer avec evaluate.py
SEUIL_FIABILITE = 0.80
