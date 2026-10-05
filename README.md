# MCP RAG Server: ask questions about your course documents from Claude Desktop

[![tests](https://github.com/faraaawo-debug/mcp-rag-server/actions/workflows/tests.yml/badge.svg)](https://github.com/faraaawo-debug/mcp-rag-server/actions/workflows/tests.yml)

An MCP (Model Context Protocol) server in Python that lets an AI assistant such as Claude Desktop
query a set of PDF or text course documents. Retrieval runs locally; the answer is written by an
LLM through the Mistral API (`ministral-14b-2512`), or locally through Ollama as an option. Every
answer cites its sources and comes with a reliability status.

## What it does

The server exposes two MCP tools:

- `search_documents(query)`: returns the 3 most relevant excerpts, with
  their source file and their cosine similarity to the question.
- `ask_question(query)`: first classifies the question with an intent router, then writes an
  answer using only the retrieved excerpts, cites the source files, and returns the detected
  intent and a reliability status: **RELIABLE**, **TO VERIFY** or **NO ANSWER**. When the information is not in the excerpts, the LLM is
  instructed to reply "I don't know.", which counts as a correct behaviour, not as an error.

## Architecture

```
Documents (PDF / TXT) in docs/<subject>/<type>/
      ↓  ingest.py: text extraction (PyMuPDF), fixed-size chunks of 200 words (40-word overlap),
      ↓             subject and document type stored as metadata (read from the file path)
Local embeddings (intfloat/multilingual-e5-small)
      ↓
ChromaDB vector store (cosine similarity)
      ↓  server.py: 2 MCP tools
search_documents → closest excerpts + source + similarity (no LLM call)
ask_question     → intent router (LLM) → retrieval adapted to the intent
                   → LLM answer (Mistral API, or Ollama locally) + intent + sources + reliability status
```

### Intent router (the agent part)

The router is the only part of the project that behaves like an agent: before retrieval, an LLM
call (`router.py`, temperature 0, JSON output, closed list of labels) decides how the question is
handled.

| The router decides | Effect |
|---|---|
| The intent: `definition`, `exercise`, `summary`, `comparison` or `off_topic` | Number of excerpts retrieved (3, or 5 for summaries and comparisons, whose answer is spread over several passages) |
| `off_topic` | Direct refusal: no retrieval and no answer generation |
| The document type (`lectures`, `assignments`...), only when the question names one | The search is limited to that type of document |

Safety rules: an unknown or unreadable label falls back to the default strategy (3 excerpts, no
filter), and a document-type filter is only kept if the question actually contains the name of
that type. Without that rule, the LLM restricted the search to assignments for a question about
an exercise found in a lecture, which would have hidden the right passage.

Implementation details kept on purpose: temperature 0, generation time measured, re-runnable
ingestion without duplicates, absolute paths (the server is started by Claude Desktop from any
folder), logs written to a file and stderr (stdout is reserved for the MCP protocol), and blocking
calls (embeddings, LLM) run outside the asyncio event loop.

## Design choices

| Component | Choice | Why |
|---|---|---|
| PDF extraction | PyMuPDF | pypdf glued about ten words together on this corpus ("thematter", "efficiently.Despite"); PyMuPDF glued none. |
| Embeddings | `intfloat/multilingual-e5-small`, run locally | `BAAI/bge-base-en-v1.5` was tested (row 3b below): it always placed the right passage among the 3 excerpts read by the LLM, but final answers scored lower on the test set, so e5-small was kept. |
| Answer LLM | Mistral API, `ministral-14b-2512` | A local 7B model on a laptop CPU without GPU took ~2 min per answer (~1 h 45 per evaluation) and invented answers ("3 marks"). `ministral-14b-2512` answers in ~2 s, is included in Mistral's free tier (on this account, `mistral-small` and larger models were not), and the dated version keeps measurements reproducible. |
| Offline option | `LLM_PROVIDER = "ollama"` in `config.py` | For offline or confidential use; not used for the measurements below. |
| Evaluation judge | `openai/gpt-oss-120b` via the Groq API, strict JSON output | A different model family from the generator, to limit self-preference bias. A local llama3 8B judge proved unreliable (too lenient, then too strict). |
| Prompt language | English | A French prompt made the model answer English questions in French. |
| API keys | User environment variables | Never in the code or in Git (see Setup). |

## Evaluation

`evaluate.py` runs the system on 23 hand-annotated questions (`eval_set.json`): 15 answerable from
the documents, 5 on the course topics but not answered by the documents (they check that the LLM
refuses instead of inventing), and 3 off-topic. For each answerable question, I recorded the source
file, the expected answer and a key sentence copied from the right passage.

- **Retrieval**, measured without any LLM and always on the top 5 excerpts whatever the pipeline
  settings: right file and key sentence found at k = 1, 3 (the number of excerpts the LLM reads)
  and 5, and MRR (mean of 1/rank of the first excerpt containing the key sentence). The key sentence
  is compared without spaces or punctuation, so the result does not depend on how the PDF text was
  extracted.
- **Answers**, graded by the LLM judge: correctness against the expected answer, and faithfulness
  to the excerpts (computed on the answers actually given, since a refusal makes no claim).
- **Correct refusals** on the 8 questions without an answer, and mean **generation time** (time
  spent in LLM calls: router and answer).
- **Router**: share of the 23 questions whose detected intent matches the annotated intent.

Each change was measured separately; detailed results are in `results/`.

### Results

| Version | Key sentence @1 | @3 | @5 | MRR | Correct intent | Correctness | Faithfulness | Correct refusals | Mean time |
|---|---|---|---|---|---|---|---|---|---|
| Baseline: initial pipeline + `ministral-14b-2512` | 53 % (8/15) | 80 % (12/15) | 93 % (14/15) | 0.678 | n/a | 73 % (11/15) | 77 % (10/13) | 100 % (8/8) | 1.9 s |
| 3a: PyMuPDF instead of pypdf | 67 % (10/15) | 87 % (13/15) | 93 % (14/15) | 0.761 | n/a | 80 % (12/15) | 93 % (13/14) | 100 % (8/8) | 2.0 s |
| 3b: + BGE-base embeddings (tested, not kept) | 47 % (7/15) | 100 % (15/15) | 100 % (15/15) | 0.700 | n/a | 73 % (11/15) | 87 % (13/15) | 100 % (8/8) | 3.2 s |
| 5: 3a + intent router (**current version**) | 67 % (10/15) | 87 % (13/15) | 93 % (14/15) | 0.761 | 74 % (17/23) | 80 % (12/15) | 100 % (14/14) | 100 % (8/8) | 2.8 s |

How to read these numbers:

- **The test set is small.** With 15 answerable questions, one question is worth 6.7 points.
  Re-grading the same baseline answers changed 1 judge verdict out of 28, and the generator's
  answers also vary when the order of the excerpts changes. A difference of one or two questions
  between versions is therefore not significant; the retrieval metrics, which are deterministic,
  are the most reliable.
- **3a.** Part of the gain probably comes from a side effect: with fixed 200-word chunks, any change
  in the extracted text shifts every chunk boundary, and therefore which chunk contains the key
  sentence.
- **3b.** BGE put the right passage first less often but always within the 3 excerpts read by the
  LLM; two questions moved from "I don't know." to a correct answer. Correctness did not improve,
  because verdicts on long answers (comparisons, summaries) changed for reasons unrelated to
  retrieval. e5-small was kept.
- **5.** The retrieval columns are measured at fixed k without the router, so they do not change.
  The router does not improve the answers on this test set (one more faithful answer is within the
  noise). What it adds is control: off-topic questions are refused without retrieval or
  generation, summaries and comparisons get more context, and the detected intent is shown to the
  user. It costs one extra LLM call (about 0.9 s). Its mistakes: two course questions that the
  documents do not answer were labelled `off_topic` (harmless here, since the right behaviour was
  to refuse, but a real course question could be refused the same way), and some confusions
  between `definition`, `exercise` and `comparison`. Its detected intents also varied by one
  question between two runs, despite temperature 0.
- **The LLM switch mattered most for refusals**: the local 7B model invented answers to questions
  the documents do not answer; `ministral-14b-2512` correctly refuses all 8.
- The right file is ranked first in 80 to 100 % of cases depending on the version: with only
  4 documents, this metric hardly separates versions.

A first, fully local run (mistral 7B via Ollama, llama3 judge) is kept as a trace in
`results/trace_local_*` but is not compared: it took about 1 h 45 on a laptop CPU and its judge
proved unreliable.

## Documents

The provided documents are course materials from courses I attended at Asia Pacific University of
Technology and Innovation (APU), Malaysia. They are in English and remain the property of their
authors (copyright notices in the files).

| File | Subject | Type | Content |
|---|---|---|---|
| `research_methods_part1.pdf` | artificial_intelligence | lectures | Slides: what research is and its components |
| `research_methods_part2.pdf` | artificial_intelligence | lectures | Slides: choosing a research problem, problem statement, objectives, abstracts |
| `ai_assignment_2026.pdf` | artificial_intelligence | assignments | Individual assignment brief (Master in AI, 2026) |
| `bis_assignment_guidelines.pdf` | business_intelligence_systems | assignments | Business Intelligence Systems assignment guidelines |

To use your own documents (PDF or TXT), store them as `docs/<subject>/<type>/<file>`:

```
docs/
├── artificial_intelligence/
│   ├── lectures/
│   └── assignments/
└── business_intelligence_systems/
    └── assignments/
```

- `<subject>`: any name, lowercase without spaces (it is used as an exact filter).
- `<type>`: `lectures`, `tutorials`, `exams` or `assignments`.

The subject and type of each excerpt are read from this path and stored in its metadata. A file
stored elsewhere is still indexed, with subject and type `unknown`, and a warning is printed.

## Setup

```bash
git clone https://github.com/faraaawo-debug/mcp-rag-server.git
cd mcp-rag-server
python -m venv venv
venv\Scripts\activate          # Windows (macOS / Linux: source venv/bin/activate)
pip install -r requirements.txt
```

API keys, created in the Mistral and Groq consoles and stored as user environment variables
(never in the code or in Git):

- `MISTRAL_API_KEY`: answer generation (the free tier is enough).
- `GROQ_API_KEY`: evaluation judge only.

On Windows: "Edit environment variables for your account" → User variables → New. The server also
reads these user variables directly from Windows, because an MCP client only passes a restricted
list of environment variables to the server it launches.

To run fully offline, set `LLM_PROVIDER = "ollama"` in `config.py` and install the local model
(`ollama pull mistral`).

## Usage

```bash
# 1. Put the documents in docs/<subject>/<type>/ (see Documents)
# 2. Index them (can be re-run without creating duplicates)
python ingest.py
# 3. Try both tools through an MCP client
python test_client.py
# Unit tests (no API key needed, no model download)
pip install -r requirements-dev.txt
pytest
# 4. Evaluate (results saved in results/<label>.json)
python evaluate.py --label my_version
# Quick retrieval-only evaluation, no LLM call (~1 min)
python evaluate.py --label my_version --retrieval-only
# Re-grade saved answers without regenerating them
python evaluate.py --label my_version --regrade results/baseline.json
```

## Use with Claude Desktop

1. Open the Claude Desktop configuration file:
   - Windows: `%APPDATA%\Claude\claude_desktop_config.json`
   - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
2. Add the content of `claude_desktop_config.example.json`, replacing the paths with the absolute
   paths on your machine (on Windows, the Python executable is `venv\Scripts\python.exe`).
3. Restart Claude Desktop: both tools appear in the list of available tools.

## Limitations

- **Small corpus**: 4 documents (about 4,800 words, 73 pages).
- **Small test set**: 23 questions, of which 15 are answerable; results are trends, not
  statistically solid measurements (see Evaluation).
- **Struck-through text and text inside images are not extracted** from the PDFs. For example, one
  slide strikes out "Results" and "Conclusion" from the parts of a descriptive abstract; the
  extracted text lists them as if they were included.
- **The reliability status does not yet separate good answers from bad ones**: it is computed from
  cosine similarities between the question, the excerpts and the answer, and every answer given
  scores between 0.83 and 0.92. It is a quick signal, not a proof.
- When the right passage is not among the 3 excerpts sent to the LLM, it answers "I don't know."
  rather than inventing an answer (2 of the 15 answerable questions in the baseline).
- The LLM judge itself has not yet been validated against human annotations.
- The intent router can label a genuine course question as `off_topic` and refuse it; its
  accuracy (74 % on 23 questions) was measured on the same small test set.
- PyMuPDF is licensed under AGPL-3.0: fine for an open-source project, but closed commercial use
  would require a commercial licence or switching back to pypdf (BSD licence).

## Next steps

- Grow the test set to at least 40 answerable questions, including harder ones (exact details,
  paraphrases, answers spread over several slides or several documents, trick questions), and run
  each evaluation several times to average out the noise.
- Replace fixed-size chunks with recursive chunking that follows the document structure (titles,
  paragraphs, sentences) and stores the section in the metadata.
- Calibrate the reliability status on the evaluation results.
- Validate the LLM judge against my own annotations and report the agreement rate.

## Tech stack

Python 3.12, MCP Python SDK 2, PyMuPDF, ChromaDB, Sentence Transformers
(`intfloat/multilingual-e5-small`), Mistral API (`ministral-14b-2512`), Groq API (evaluation judge),
Ollama (optional), pytest and GitHub Actions for the unit tests.
