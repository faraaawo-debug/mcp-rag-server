"""Refusal detection and reliability status (the LLM and the embeddings are replaced)."""
import pytest

import config
import rag_utils

PASSAGES = [{"text": "An abstract is a brief summary.", "source": "a.pdf", "chunk": 0, "similarity": 0.9}]


@pytest.mark.parametrize("answer, expected", [
    ("I don't know.", True),
    ("I do not know the answer.", True),
    ("Je ne sais pas.", True),
    ("An abstract is a brief summary [a.pdf].", False),
])
def test_refusal_detection(answer, expected):
    assert rag_utils.is_refusal(answer) is expected


def test_a_refusal_gets_the_refusal_status_without_score(monkeypatch):
    monkeypatch.setattr(rag_utils.llm, "chat", lambda *a, **k: "I don't know.")
    result = rag_utils.answer_question("What is the capital of Australia?", passages=PASSAGES)
    assert result["status"] == "refusal"
    assert result["overall_score"] is None
    assert result["sources"] == ["a.pdf"]


@pytest.mark.parametrize("score, status", [(0.95, "reliable"), (0.50, "to_verify")])
def test_status_depends_on_the_threshold(monkeypatch, score, status):
    monkeypatch.setattr(rag_utils.llm, "chat", lambda *a, **k: "A brief summary [a.pdf].")
    monkeypatch.setattr(rag_utils, "similarity_scores",
                        lambda *a: {"grounding": score, "excerpt_relevance": score, "answer_relevance": score})
    result = rag_utils.answer_question("What is an abstract?", passages=PASSAGES)
    assert result["status"] == status
    assert result["overall_score"] == pytest.approx(score)
    assert (score >= config.RELIABILITY_THRESHOLD) == (status == "reliable")
