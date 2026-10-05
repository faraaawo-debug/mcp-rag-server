"""Settings shared by ingestion, the server and the evaluation."""
from pathlib import Path

# Absolute paths: the server works whatever folder it is started from
# (required when Claude Desktop starts it).
BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
CHROMA_PATH = BASE_DIR / "chroma_db"
EVAL_SET_PATH = BASE_DIR / "eval_set.json"
LOG_PATH = BASE_DIR / "server.log"

COLLECTION_NAME = "docs"

# Expected layout: docs/<subject>/<type>/<file>
# The subject and type of each excerpt are read from this path.
DOC_TYPES = ("lectures", "tutorials", "exams", "assignments")
UNKNOWN = "unknown"

# Multilingual embedding model, 384 dimensions, 512 tokens max, run locally.
# BAAI/bge-base-en-v1.5 was tested (step 3b): better retrieval at k=3, but lower final
# answer scores on the test set; e5-small was kept.
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
# E5 models expect the "query: " prefix before a question and "passage: " before an indexed
# passage; "query: " is also used to compare two texts with each other.
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "
COMPARISON_PREFIX = "query: "

# LLM that writes the answers
# "mistral" (default): Mistral API, key in the MISTRAL_API_KEY environment variable.
#   ministral-14b-2512: included in the free tier (mistral-small is blocked there), and the
#   dated version keeps measurements reproducible.
# "ollama": local model, usable offline without sending any data (~2 min per answer on a
#   laptop CPU without GPU).
LLM_PROVIDER = "mistral"
LLM_MODELS = {"mistral": "ministral-14b-2512", "ollama": "mistral"}
LLM_MODEL = LLM_MODELS[LLM_PROVIDER]
# Number of attempts when an API answers "too many requests" or is unavailable
LLM_MAX_ATTEMPTS = 6

# Chunking: 200 words per excerpt, 40 words of overlap
CHUNK_WORDS = 200
OVERLAP_WORDS = 40

# Number of excerpts retrieved per question (default, and when the router fails)
TOP_K = 3

# Intent router (router.py): one LLM call before retrieval
USE_ROUTER = True
# Excerpts retrieved per detected intent: summaries and comparisons need more context
TOP_K_BY_INTENT = {"definition": 3, "exercise": 3, "summary": 5, "comparison": 5}
OFF_TOPIC_ANSWER = "I don't know. This question does not seem related to the course documents."

# Evaluation: judge from a different model family than the generator (limits self-preference
# bias), key in the GROQ_API_KEY environment variable
JUDGE_PROVIDER = "groq"
JUDGE_MODEL = "openai/gpt-oss-120b"
# Retrieval is always measured at these k, whatever k the pipeline uses,
# so that measurements stay comparable from one version to the next
K_EVAL = (1, 3, 5)  # 3 = number of excerpts the LLM reads (TOP_K)
RESULTS_DIR = BASE_DIR / "results"

# Threshold of the reliability status: to be recalibrated with evaluate.py
RELIABILITY_THRESHOLD = 0.80
