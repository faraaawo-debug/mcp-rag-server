# MCP RAG Server — Assistant documentaire agentique

Serveur MCP (Model Context Protocol) en Python exposant un pipeline RAG complet.
Permet à n'importe quel client MCP compatible d'interroger une base documentaire
via des outils standardisés.

## Architecture
```
Document (PDF/TXT)
      ↓
Ingestion + vectorisation (bge-small-en-v1.5)
      ↓
Stockage ChromaDB
      ↓
Serveur MCP expose 2 outils
      ↓
rechercher_documents → retrieval sémantique
poser_question       → réponse Mistral + évaluation RAG
```

## Outils exposés

- `rechercher_documents(query)` — retourne les passages les plus pertinents
- `poser_question(query)` — génère une réponse via Mistral + évalue la qualité

## Évaluation des réponses

Chaque réponse est évaluée avec 3 métriques :
- **Fidélité** : La réponse générée par le LLM est-elle fidèle aux chunks ?
- **Pertinence du contexte** : les chunks récupérés sont-ils pertinents ?
- **Pertinence des réponses** : la réponse répond-elle à la question ?

## Stack technique

- Python 3.11
- MCP SDK (Model Context Protocol)
- ChromaDB — base vectorielle locale
- Sentence Transformers (BAAI/bge-small-en-v1.5) — embeddings
- Ollama + Mistral — LLM local

## Installation
```bash
# Cloner le repo
git clone https://github.com/faraaawo-debug/mcp-rag-server.git
cd mcp-rag-server

# Créer l'environnement virtuel
python3.11 -m venv venv
source venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt

# Lancer Ollama avec Mistral
ollama pull mistral
```

## Utilisation
```bash
# 1. Ajouter des documents dans le dossier docs/
# 2. Ingérer les documents
python3 ingest.py

# 3. Lancer le client de test
python3 test_client.py
```

## Limites identifiées

- Performances meilleures sur les questions précises que sur les questions larges (A retravailler)