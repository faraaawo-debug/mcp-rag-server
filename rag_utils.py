"""Shared building blocks: embeddings, retrieval, generation and reliability status."""
import time

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

import config
import llm
import router

_model = None


def get_model():
    """Loads the embedding model only once."""
    global _model
    if _model is None:
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def _encode(texts, prefix):
    # Normalized vectors: the dot product is then the cosine similarity
    return get_model().encode([prefix + t for t in texts], normalize_embeddings=True)


def embed_passages(texts):
    return _encode(texts, config.PASSAGE_PREFIX)


def embed_queries(texts):
    return _encode(texts, config.QUERY_PREFIX)


def embed_for_comparison(texts):
    """To compare texts with each other (answer, excerpts, question)."""
    return _encode(texts, config.COMPARISON_PREFIX)


def get_collection():
    client = chromadb.PersistentClient(path=str(config.CHROMA_PATH))
    return client.get_or_create_collection(
        name=config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )


def search(question, k=config.TOP_K, doc_type=None):
    """Returns the k excerpts closest to the question, with their source.
    With doc_type, only excerpts of that document type are searched; if there is none,
    the search falls back to all documents."""
    collection = get_collection()
    vector = embed_queries([question])[0].tolist()
    query = {"query_embeddings": [vector], "n_results": k,
             "include": ["documents", "metadatas", "distances"]}
    res = collection.query(**query, where={"doc_type": doc_type}) if doc_type else None
    if res is None or not res["documents"][0]:
        res = collection.query(**query)
    passages = []
    for text, meta, distance in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        passages.append({
            "text": text,
            "source": meta["source"],
            "chunk": meta["chunk"],
            "similarity": round(1 - distance, 3),  # cosine distance -> similarity
        })
    return passages


# Prompt in English, the language of the documents: a French prompt pushed the model
# to answer in French to questions asked in English.
PROMPT = """You are an assistant that answers questions using ONLY the excerpts below.
Rules:
- Use only information stated in the excerpts. Do not use your general knowledge.
- If the answer is not in the excerpts, reply exactly: "I don't know."
- Answer in the language of the question.
- Cite the source file of each piece of information in square brackets, e.g. [lecture.pdf].

Excerpts:
{context}

Question: {question}
"""

# Refusal phrases (the LLM answers in the language of the question)
REFUSALS = ("i don't know", "i do not know", "je ne sais pas")


def is_refusal(answer):
    return any(r in answer.lower() for r in REFUSALS)


def generate_answer(question, passages):
    """Calls the configured LLM (temperature 0) and measures the response time,
    including any wait caused by API rate limits."""
    context = "\n\n".join(f"[{p['source']}] {p['text']}" for p in passages)
    start = time.perf_counter()
    answer = llm.chat(
        [{"role": "user", "content": PROMPT.format(context=context, question=question)}],
        provider=config.LLM_PROVIDER, model=config.LLM_MODEL,
    )
    latency = time.perf_counter() - start
    return answer, latency


def similarity_scores(question, answer, passages):
    """Three cosine similarity scores (between 0 and 1) used as a quick indicator.
    They are not measures of truth: the real evaluation is in evaluate.py."""
    v_question, v_answer = embed_for_comparison([question, answer])
    v_passages = embed_for_comparison([p["text"] for p in passages])
    grounding = float(np.max(v_passages @ v_answer))                  # answer close to at least one excerpt?
    context = float(np.mean([p["similarity"] for p in passages]))    # excerpts close to the question?
    relevance = float(v_question @ v_answer)                          # answer close to the question?
    return {
        "grounding": round(grounding, 3),
        "excerpt_relevance": round(context, 3),
        "answer_relevance": round(relevance, 3),
    }


def answer_question(question, passages=None):
    """Full pipeline: routing, retrieval, generation, reliability status.
    latency_s is the time spent in LLM calls (router + answer)."""
    intent, doc_type, routing_time = None, None, 0.0
    if passages is None:
        if config.USE_ROUTER:
            start = time.perf_counter()
            intent, doc_type = router.route(question)
            routing_time = time.perf_counter() - start
            if intent == "off_topic":
                # Direct refusal: no retrieval and no generation
                return {"answer": config.OFF_TOPIC_ANSWER, "sources": [], "passages": [],
                        "scores": None, "overall_score": None, "status": "refusal",
                        "latency_s": round(routing_time, 2), "intent": intent, "doc_type": None}
        passages = search(question, k=router.top_k(intent), doc_type=doc_type)
    answer, latency = generate_answer(question, passages)
    latency += routing_time

    if is_refusal(answer):
        # A refusal is correct behaviour, not an answer outside the documents
        status, scores, overall_score = "refusal", None, None
    else:
        scores = similarity_scores(question, answer, passages)
        overall_score = round(sum(scores.values()) / 3, 3)
        status = "reliable" if overall_score >= config.RELIABILITY_THRESHOLD else "to_verify"

    return {
        "answer": answer,
        "sources": sorted({p["source"] for p in passages}),
        "passages": passages,
        "scores": scores,
        "overall_score": overall_score,
        "status": status,
        "latency_s": round(latency, 2),
        "intent": intent,
        "doc_type": doc_type,
    }
