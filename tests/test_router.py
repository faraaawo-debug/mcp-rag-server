"""Intent router: output parsing, safety rule on filters, retrieval strategy (no API call)."""
import pytest

import config
import rag_utils
import router


@pytest.mark.parametrize("raw, expected", [
    ('{"intent": "summary", "doc_type": "lectures"}', ("summary", "lectures")),
    ('{"intent": "comparison", "doc_type": null}', ("comparison", None)),
    ('{"intent": "joke", "doc_type": "slides"}', (None, None)),   # labels outside the closed lists
    ("not json", (None, None)),
])
def test_router_output_parsing(raw, expected):
    assert router.parse(raw) == expected


def test_a_filter_is_kept_only_if_the_question_names_the_document_type(monkeypatch):
    monkeypatch.setattr(router.llm, "chat", lambda *a, **k: '{"intent": "exercise", "doc_type": "assignments"}')
    assert router.route("For my BIS assignment, is this appropriate?") == ("exercise", "assignments")
    assert router.route("I have to do the One Page Research exercise.") == ("exercise", None)


def test_more_excerpts_for_summaries_and_comparisons():
    assert router.top_k("summary") > router.top_k("definition")
    assert router.top_k(None) == config.TOP_K   # router failure: default strategy


def test_off_topic_questions_are_refused_without_retrieval_or_generation(monkeypatch):
    monkeypatch.setattr(config, "USE_ROUTER", True)
    monkeypatch.setattr(router, "route", lambda q: ("off_topic", None))
    monkeypatch.setattr(rag_utils, "search", lambda *a, **k: pytest.fail("no retrieval expected"))
    monkeypatch.setattr(rag_utils.llm, "chat", lambda *a, **k: pytest.fail("no generation expected"))
    result = rag_utils.answer_question("What is the capital of Australia?")
    assert result["status"] == "refusal"
    assert result["intent"] == "off_topic"
    assert rag_utils.is_refusal(result["answer"])


def test_the_detected_intent_sets_the_number_of_excerpts(monkeypatch):
    seen = {}
    monkeypatch.setattr(config, "USE_ROUTER", True)
    monkeypatch.setattr(router, "route", lambda q: ("summary", "lectures"))
    monkeypatch.setattr(rag_utils, "search", lambda q, k, doc_type: seen.update(k=k, doc_type=doc_type) or [])
    monkeypatch.setattr(rag_utils.llm, "chat", lambda *a, **k: "I don't know.")
    result = rag_utils.answer_question("Summarize the lecture.")
    assert seen == {"k": config.TOP_K_BY_INTENT["summary"], "doc_type": "lectures"}
    assert result["intent"] == "summary"
