# MCP RAG Server : interroger ses documents avec un LLM local

Serveur MCP (Model Context Protocol) en Python qui permet à un assistant IA, comme Claude Desktop,
d'interroger une base de documents PDF ou texte. Le serveur retrouve les extraits utiles, fait générer
une réponse par un LLM open source exécuté en local (Mistral via Ollama), cite les sources et
indique un niveau de fiabilité.

## Architecture

```
Documents (PDF / TXT)
      ↓  ingest.py : découpage en extraits de 200 mots (40 mots de chevauchement)
Embeddings multilingues (intfloat/multilingual-e5-small)
      ↓
Base vectorielle ChromaDB (similarité cosinus)
      ↓  server.py : 2 outils MCP
rechercher_documents → extraits les plus proches + source + similarité
poser_question       → réponse de Mistral (local) + sources + indicateur de fiabilité
```

## Documents

Les documents fournis sont des supports de cours que j'ai suivis à Asia Pacific University of
Technology and Innovation (APU). Ils sont en anglais et restent la propriété de leurs auteurs
(mentions de copyright dans les fichiers).

| Fichier | Matière | Type | Contenu |
|---|---|---|---|
| `research_methods_part1.pdf` | artificial_intelligence | lectures | Slides : qu'est-ce que la recherche et ses composantes |
| `research_methods_part2.pdf` | artificial_intelligence | lectures | Slides : choisir un problème de recherche, problem statement, objectifs |
| `ai_assignment_2026.pdf` | artificial_intelligence | assignments | Sujet du devoir individuel (Master AI 2026) |
| `bis_assignment_guidelines.pdf` | business_intelligence_systems | assignments | Consignes du devoir de Business Intelligence Systems |

Pour utiliser vos propres documents (PDF ou TXT), rangez-les selon l'arborescence
`docs/<matiere>/<type>/<fichier>` :

```
docs/
├── artificial_intelligence/
│   ├── lectures/
│   └── assignments/
└── business_intelligence_systems/
    └── assignments/
```

- `<matiere>` : nom libre, en minuscules et sans espaces (il sert de filtre exact).
- `<type>` : `lectures`, `tutorials`, `exams` ou `assignments`.

La matière et le type de chaque extrait sont déduits de ce chemin et enregistrés dans ses métadonnées.
Un fichier rangé ailleurs est quand même indexé, avec la matière et le type `unknown`, et un
avertissement est affiché.

## Outils exposés

- `rechercher_documents(query)` : retourne les 3 extraits les plus pertinents, avec le fichier source et le score de similarité.
- `poser_question(query)` : génère une réponse à partir de ces extraits uniquement. La réponse cite ses sources, et le LLM répond « Je ne sais pas » si l'information est absente.

## Indicateur de fiabilité

Chaque réponse est accompagnée d'un statut renvoyé à l'utilisateur :

- **FIABLE** ou **À VÉRIFIER**, selon un score calculé à partir de trois similarités cosinus (réponse / extraits, extraits / question, réponse / question). Le seuil est calibré par `evaluate.py`.
- **PAS DE RÉPONSE** quand le modèle indique que l'information n'est pas dans les documents.

Ce score est un indicateur rapide, pas une preuve. La qualité réelle est mesurée séparément (section suivante).

## Évaluation

`evaluate.py` teste le système sur un jeu de questions annotées à la main (`eval_set.json`),
qui contient des questions dont la réponse est dans les documents et d'autres dont elle n'y est pas.
Un LLM sert de juge pour l'exactitude et la fidélité.

| Mesure | Résultat |
|---|---|
| Questions | À COMPLÉTER |
| Recherche réussie (bon document dans les 3 extraits) | À COMPLÉTER |
| Exactitude des réponses | À COMPLÉTER |
| Fidélité aux documents | À COMPLÉTER |
| Refus corrects (information absente) | À COMPLÉTER |
| Temps de génération moyen | À COMPLÉTER |

## Stack technique

Python 3.11, MCP SDK, ChromaDB, Sentence Transformers (multilingual-e5-small), Ollama + Mistral.

## Installation

```bash
git clone https://github.com/faraaawo-debug/mcp-rag-server.git
cd mcp-rag-server
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
ollama pull mistral
```

## Utilisation

```bash
# 1. Placer les documents dans docs/<matiere>/<type>/ (voir la section Documents)
# 2. Indexer les documents (relançable sans créer de doublons)
python ingest.py
# 3. Tester les outils
python test_client.py
# 4. Évaluer le système
python evaluate.py
```

## Utilisation avec Claude Desktop

1. Ouvrir le fichier de configuration de Claude Desktop :
   - macOS : `~/Library/Application Support/Claude/claude_desktop_config.json`
   - Windows : `%APPDATA%\Claude\claude_desktop_config.json`
2. Y ajouter le contenu de `claude_desktop_config.example.json`, en remplaçant les chemins par les chemins absolus de votre machine.
3. Redémarrer Claude Desktop : les deux outils apparaissent dans la liste des outils disponibles.

## Limites

- Le corpus est petit : 4 documents (environ 4 800 mots, 73 pages). Les mesures d'évaluation portent donc sur peu de données et doivent être lues comme des tendances, pas comme des résultats généralisables. Avec seulement 4 fichiers, retrouver le bon fichier est facile : la phrase clé retrouvée et le MRR sont plus parlants.

- L'indicateur de fiabilité repose sur des similarités d'embeddings : il signale les réponses éloignées des documents, mais ne détecte pas toutes les erreurs.
- Les questions larges, dont la réponse est répartie dans plusieurs parties d'un document, sont moins bien traitées que les questions précises.
