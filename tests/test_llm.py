"""Retries on API rate limits (the API is replaced by a fake)."""
import pytest

import llm


class ApiError(Exception):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


def test_retries_after_a_rate_limit_then_succeeds(monkeypatch):
    calls, waits = [], []

    def fake_call(*args):
        calls.append(1)
        if len(calls) < 3:
            raise ApiError(429)
        return "  answer  "

    monkeypatch.setattr(llm, "_call", fake_call)
    monkeypatch.setattr(llm.time, "sleep", waits.append)
    assert llm.chat([], provider="mistral", model="m") == "answer"
    assert len(calls) == 3
    assert waits == [2, 4]


def test_a_permanent_error_is_not_retried(monkeypatch):
    calls = []

    def fake_call(*args):
        calls.append(1)
        raise ApiError(400)

    monkeypatch.setattr(llm, "_call", fake_call)
    with pytest.raises(ApiError):
        llm.chat([], provider="mistral", model="m")
    assert len(calls) == 1


def test_gives_up_after_the_maximum_number_of_attempts(monkeypatch):
    monkeypatch.setattr(llm, "_call", lambda *a: (_ for _ in ()).throw(ApiError(503)))
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    with pytest.raises(ApiError):
        llm.chat([], provider="groq", model="m")
