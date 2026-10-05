"""Intent router: before retrieval, an LLM call classifies the question and the server adapts
its retrieval strategy. This is the only "agent" part of the project: the LLM decides
(1) the intent of the question, which sets how many excerpts are retrieved and whether the
question is refused straight away (off-topic), and (2) the document type to search in, only
when the question names one explicitly."""
import json
import logging

import config
import llm

logger = logging.getLogger(__name__)

INTENTS = ("definition", "exercise", "summary", "comparison", "off_topic")

PROMPT = """You route questions asked by a student about their course documents.
The documents cover these subjects: {subjects}. Document types: {doc_types}.

Classify the question with exactly one intent:
- "definition": asks what something is, means, or a precise fact or explanation
- "exercise": asks for help doing an exercise or an assignment task (how to do, how to phrase, is X appropriate)
- "summary": asks to summarize or list the main points of a topic or document
- "comparison": asks for the differences or similarities between two or more things
- "off_topic": unrelated to studying these subjects (general knowledge, cooking, sport, news...)
A question about the courses is NOT off_topic, even if the documents might not contain the answer.

Also give "doc_type": one of the document types ONLY if the question explicitly names that kind of
document (for example "the lecture" -> "lectures", "the assignment" -> "assignments"); otherwise null.

Question: {question}

Reply with JSON only: {{"intent": "...", "doc_type": "..." or null}}"""


def _subjects():
    """Subjects are the folder names in docs/ (e.g. artificial_intelligence)."""
    if not config.DOCS_DIR.exists():
        return "unknown"
    return ", ".join(sorted(p.name for p in config.DOCS_DIR.iterdir() if p.is_dir())) or "unknown"


def named_in(doc_type, question):
    """True if the question contains the name of the document type ("lecture", "assignment"...).
    Safety rule: the LLM once restricted the search to assignments for a question about a
    lecture exercise, so a filter is only kept when the question really names the type."""
    return doc_type.rstrip("s") in question.lower()


def parse(raw):
    """Reads the router's JSON output. An unknown or unreadable intent gives None, so that the
    pipeline falls back to its default strategy instead of failing."""
    try:
        data = json.loads(raw)
        intent, doc_type = data.get("intent"), data.get("doc_type")
    except (json.JSONDecodeError, AttributeError):
        return None, None
    intent = intent if intent in INTENTS else None
    doc_type = doc_type if doc_type in config.DOC_TYPES else None
    return intent, doc_type


def route(question):
    """Returns (intent, doc_type) for a question, temperature 0, JSON output."""
    raw = llm.chat(
        [{"role": "user", "content": PROMPT.format(
            subjects=_subjects(), doc_types=", ".join(config.DOC_TYPES), question=question)}],
        provider=config.LLM_PROVIDER, model=config.LLM_MODEL, json_mode=True,
    )
    intent, doc_type = parse(raw)
    if doc_type and not named_in(doc_type, question):
        doc_type = None
    if intent is None:
        logger.warning(f"Router output not understood, default strategy used: {raw!r}")
    return intent, doc_type


def top_k(intent):
    """Number of excerpts to retrieve: more for summaries and comparisons, whose answer is
    spread over several passages."""
    return config.TOP_K_BY_INTENT.get(intent, config.TOP_K)
