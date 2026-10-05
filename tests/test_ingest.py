"""Chunking and subject/type classification (no model, no API)."""
import pytest

import config
from ingest import chunk_by_words, classify


def words(n):
    return " ".join(f"w{i}" for i in range(n))


def test_chunks_have_the_right_size_and_overlap():
    chunks = chunk_by_words(words(500), size=200, overlap=40)
    assert [len(c.split()) for c in chunks] == [200, 200, 180]
    # the last 40 words of a chunk start the next one
    assert chunks[0].split()[-40:] == chunks[1].split()[:40]


def test_short_text_gives_a_single_chunk():
    assert chunk_by_words(words(50), size=200, overlap=40) == [words(50)]


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_by_words(words(10), size=40, overlap=40)


def test_subject_and_type_are_read_from_the_path():
    path = config.DOCS_DIR / "artificial_intelligence" / "lectures" / "slides.pdf"
    assert classify(path) == ("artificial_intelligence", "lectures")


def test_misplaced_file_is_classified_as_unknown():
    assert classify(config.DOCS_DIR / "slides.pdf") == (config.UNKNOWN, config.UNKNOWN)
    assert classify(config.DOCS_DIR / "ai" / "notes" / "f.pdf") == (config.UNKNOWN, config.UNKNOWN)
