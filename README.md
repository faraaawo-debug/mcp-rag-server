# MCP RAG Server : interroger ses documents de cours depuis Claude Desktop

Serveur MCP (Model Context Protocol) en Python qui permet à un assistant IA, comme Claude Desktop,
d'interroger une base de documents PDF ou texte. La recherche des extraits utiles se fait en local ;
la réponse est rédigée par un LLM via l'API Mistral (`ministral-14b-2512`), ou en local via Ollama
en option. Chaque réponse cite ses sources et indique un niveau de fiabilité.

## Architecture

```
Documents (PDF / TXT)
      ↓  ingest.py : découpage en extraits de 200 mots (40 mots de chevauchement)
Embeddings multilingues (intfloat/multilingual-e5-small)
      ↓
Base vectorielle ChromaDB (similarité cosinus)
      ↓  server.py : 2 outils MCP
rechercher_documents → extraits les plus proches + source + similarité
poser_question       → réponse du LLM (API Mistral, ou Ollama en local) + sources + indicateur de fiabilité
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
- `poser_question(query)` : génère une réponse à partir de ces extraits uniquement. La réponse cite ses sources, et le LLM répond « I don't know. » si l'information est absente.

## Indicateur de fiabilité

Chaque réponse est accompagnée d'un statut renvoyé à l'utilisateur :

- **FIABLE** ou **À VÉRIFIER**, selon un score calculé à partir de trois similarités cosinus (réponse / extraits, extraits / question, réponse / question). Le seuil est calibré par `evaluate.py`.
- **PAS DE RÉPONSE** quand le modèle indique que l'information n'est pas dans les documents.

Ce score est un indicateur rapide, pas une preuve. La qualité réelle est mesurée séparément (section suivante).

## Évaluation

`evaluate.py` teste le système sur 23 questions annotées à la main (`eval_set.json`) :
15 dont la réponse est dans les documents, 5 dans le thème des cours mais absentes des documents,
et 3 hors sujet. Pour chaque question répondable, j'ai noté le fichier source, la réponse attendue
et une phrase clé recopiée du bon passage.

- **Recherche**, mesurée sans LLM, toujours sur les 5 premiers extraits quel que soit le réglage
  du pipeline : bon fichier et phrase clé retrouvés à k=1 et k=5, et MRR (moyenne de 1/rang du
  premier extrait contenant la phrase clé). La phrase clé est comparée sans espaces ni ponctuation,
  pour ne pas dépendre de la façon dont le PDF a été extrait.
- **Réponses**, notées par un LLM juge (`openai/gpt-oss-120b` via l'API Groq), d'une autre famille
  que le modèle qui génère pour limiter le biais d'auto-préférence : exactitude par rapport à la
  réponse attendue, et fidélité aux extraits (calculée sur les réponses données, un refus
  n'affirmant rien). Le juge répond en JSON strict ; sa fiabilité reste à vérifier (étape 9).
- **Refus corrects** sur les 8 questions sans réponse, et **temps de génération** moyen.

Chaque amélioration est mesurée séparément ; les résultats détaillés sont dans `resultats/`.
Avec 15 questions répondables, une question vaut 6,7 points : les écarts se lisent comme des
tendances.

### Progression

| Version | Phrase clé @1 | Phrase clé @5 | MRR | Exactitude | Fidélité | Refus corrects | Temps moyen |
|---|---|---|---|---|---|---|---|
| Référence : pipeline initial + `ministral-14b-2512` | 53 % (8/15) | 93 % (14/15) | 0,678 | 73 % (11/15) | 77 % (10/13) | 100 % (8/8) | 1,9 s |

Le bon fichier est retrouvé en 1re position dans 80 % des cas et dans les 5 premiers dans 100 % des
cas : avec 4 documents, cette mesure distingue peu les versions.

Une première mesure entièrement locale (mistral 7B via Ollama, juge llama3) est conservée comme
trace dans `resultats/trace_local_*` mais n'est pas comparée : sur un processeur de portable sans
GPU, elle prenait environ 1 h 45 et son juge s'est révélé peu fiable.

## Stack technique

Python 3.12, MCP SDK 2, ChromaDB, Sentence Transformers (multilingual-e5-small), API Mistral
(`ministral-14b-2512`), API Groq pour le juge de l'évaluation, Ollama en option.

## Installation

```bash
git clone https://github.com/faraaawo-debug/mcp-rag-server.git
cd mcp-rag-server
python -m venv venv
venv\Scripts\activate          # Windows (macOS / Linux : source venv/bin/activate)
pip install -r requirements.txt
```

Clés API, à créer dans les consoles Mistral et Groq puis à enregistrer comme variables
d'environnement utilisateur (jamais dans le code ni dans Git) :

- `MISTRAL_API_KEY` : génération des réponses (offre gratuite suffisante).
- `GROQ_API_KEY` : juge de l'évaluation uniquement.

Sous Windows : « Modifier les variables d'environnement pour votre compte » → Variables utilisateur
→ Nouvelle. Le serveur relit aussi ces variables utilisateur directement, car un client MCP ne
transmet qu'une liste restreinte de variables au serveur qu'il lance.

Pour fonctionner hors ligne, mettre `LLM_PROVIDER = "ollama"` dans `config.py` et installer le
modèle local (`ollama pull mistral`).

## Utilisation

```bash
# 1. Placer les documents dans docs/<matiere>/<type>/ (voir la section Documents)
# 2. Indexer les documents (relançable sans créer de doublons)
python ingest.py
# 3. Tester les outils
python test_client.py
# 4. Évaluer le système
python evaluate.py --etiquette nom_de_version
# Mesure rapide de la recherche seule, sans LLM
python evaluate.py --etiquette essai --recherche-seule
```

## Utilisation avec Claude Desktop

1. Ouvrir le fichier de configuration de Claude Desktop :
   - macOS : `~/Library/Application Support/Claude/claude_desktop_config.json`
   - Windows : `%APPDATA%\Claude\claude_desktop_config.json`
2. Y ajouter le contenu de `claude_desktop_config.example.json`, en remplaçant les chemins par les chemins absolus de votre machine.
3. Redémarrer Claude Desktop : les deux outils apparaissent dans la liste des outils disponibles.

## Limites

- Le corpus est petit : 4 documents (environ 4 800 mots, 73 pages). Les mesures d'évaluation portent donc sur peu de données et doivent être lues comme des tendances, pas comme des résultats généralisables. Avec seulement 4 fichiers, retrouver le bon fichier est facile : la phrase clé retrouvée et le MRR sont plus parlants.

- L'indicateur de fiabilité repose sur des similarités d'embeddings. Dans la version actuelle, il ne sépare pas les bonnes réponses des mauvaises : toutes les réponses données obtiennent un score entre 0,83 et 0,92 (calibration prévue).
- Quand le bon passage n'est pas parmi les 3 extraits transmis au LLM, celui-ci répond « I don't know. » plutôt que d'inventer : c'est le cas de 2 des 15 questions répondables dans la référence.
- Le texte barré et le texte contenu dans les images des slides ne sont pas extraits des PDF.
- Les questions larges, dont la réponse est répartie dans plusieurs parties d'un document, sont moins bien traitées que les questions précises.
