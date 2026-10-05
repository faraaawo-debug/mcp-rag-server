"""Evaluation helpers: key sentence matching, judge output parsing, metrics."""
import pytest

import evaluate


def test_key_sentence_matching_ignores_spaces_case_and_punctuation():
    assert evaluate.normalize("Two guys have one apple to share.") == evaluate.normalize("two guys haveone apple toshare")


@pytest.mark.parametrize("raw, verdict", [
    ('{"verdict": "YES"}', True),
    ('{"verdict": "no"}', False),
    ('{"verdict": "MAYBE"}', None),
    ("YES, the answer is correct", None),
])
def test_judge_verdict_parsing(monkeypatch, raw, verdict):
    monkeypatch.setattr(evaluate.llm, "chat", lambda *a, **k: raw)
    assert evaluate.judge("prompt")[0] is verdict


def test_retrieval_metrics():
    rows = [
        {"answerable": True, "file_rank": 1, "key_rank": 1},
        {"answerable": True, "file_rank": 1, "key_rank": 2},
        {"answerable": True, "file_rank": 3, "key_rank": None},
        {"answerable": False},
    ]
    summary = evaluate.summarize("test", rows, retrieval_only=True)
    assert summary["questions"] == 4
    assert summary["key_sentence@1"] == "33 % (1/3)"
    assert summary["key_sentence@3"] == "67 % (2/3)"
    assert summary["file_found@3"] == "100 % (3/3)"
    assert summary["mrr_key_sentence@5"] == pytest.approx((1 + 0.5 + 0) / 3, abs=1e-3)
