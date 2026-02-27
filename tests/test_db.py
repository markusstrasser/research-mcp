import pytest
from pathlib import Path
from research_mcp.db import PaperDB


@pytest.fixture
def db(tmp_path):
    return PaperDB(tmp_path / "test.db")


def test_init_creates_tables(db):
    tables = db.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    table_names = {t[0] for t in tables}
    assert "papers" in table_names
    assert "cache" in table_names


def test_upsert_and_get(db):
    db.upsert_paper({
        "paper_id": "abc123",
        "title": "Test Paper",
        "abstract": "This is a test.",
        "authors": ["Alice", "Bob"],
        "year": 2024,
        "citation_count": 42,
    })
    paper = db.get_paper("abc123")
    assert paper is not None
    assert paper["title"] == "Test Paper"
    assert paper["authors"] == ["Alice", "Bob"]
    assert paper["citation_count"] == 42


def test_upsert_updates_existing(db):
    db.upsert_paper({"paper_id": "abc", "title": "V1", "abstract": "Old"})
    db.upsert_paper({"paper_id": "abc", "title": "V2", "abstract": "New"})
    paper = db.get_paper("abc")
    assert paper["title"] == "V2"
    assert paper["abstract"] == "New"


def test_list_papers(db):
    for i in range(3):
        db.upsert_paper({
            "paper_id": f"paper_{i}",
            "title": f"Paper {i}",
            "abstract": f"Abstract {i}",
        })
    papers = db.list_papers()
    assert len(papers) == 3


def test_export_for_selve(db):
    db.upsert_paper({
        "paper_id": "abc123",
        "doi": "10.1234/test",
        "title": "Test Paper",
        "abstract": "This is a test abstract.",
        "authors": ["Alice"],
        "year": 2024,
        "venue": "Nature",
    })
    entries = db.export_for_selve()
    assert len(entries) == 1
    e = entries[0]
    assert e["id"] == "s2_abc123"
    assert e["source"] == "papers"
    assert e["title"] == "Test Paper"
    assert "This is a test abstract" in e["text"]
    assert "Alice" in e["text"]
    assert "Nature" in e["text"]
    assert e["date"] == "2024-01-01"


def test_cache_hit(db):
    db.set_cache("key1", {"data": "value"})
    assert db.get_cache("key1") == {"data": "value"}


def test_cache_miss(db):
    assert db.get_cache("nonexistent") is None
