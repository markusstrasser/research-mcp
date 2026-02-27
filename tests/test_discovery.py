import pytest
import respx
import httpx
from pathlib import Path
from research_mcp.db import PaperDB
from research_mcp.discovery import SemanticScholar, S2_BASE

FAKE_SEARCH_RESPONSE = {
    "total": 1,
    "data": [{
        "paperId": "abc123",
        "title": "G6PD Deficiency Review",
        "abstract": "Glucose-6-phosphate dehydrogenase deficiency is...",
        "year": 2023,
        "authors": [{"name": "Alice Smith"}, {"name": "Bob Jones"}],
        "citationCount": 150,
        "journal": {"name": "Nature Reviews"},
        "externalIds": {"DOI": "10.1038/test", "ArXiv": None, "PubMed": "12345"},
        "openAccessPdf": {"url": "https://example.com/paper.pdf"},
    }],
}

FAKE_PAPER_RESPONSE = FAKE_SEARCH_RESPONSE["data"][0]


@pytest.fixture
def db(tmp_path):
    return PaperDB(tmp_path / "test.db")


@pytest.fixture
def s2(db):
    return SemanticScholar(db)


@respx.mock
def test_search_papers(s2):
    respx.get(f"{S2_BASE}/paper/search").mock(
        return_value=httpx.Response(200, json=FAKE_SEARCH_RESPONSE)
    )
    results = s2.search("G6PD deficiency")
    assert len(results) == 1
    assert results[0]["paper_id"] == "abc123"
    assert results[0]["title"] == "G6PD Deficiency Review"
    assert results[0]["authors"] == ["Alice Smith", "Bob Jones"]
    assert results[0]["doi"] == "10.1038/test"
    assert results[0]["open_access_url"] == "https://example.com/paper.pdf"


@respx.mock
def test_search_caches_response(s2):
    route = respx.get(f"{S2_BASE}/paper/search").mock(
        return_value=httpx.Response(200, json=FAKE_SEARCH_RESPONSE)
    )
    s2.search("G6PD deficiency")
    s2.search("G6PD deficiency")  # should hit cache
    assert route.call_count == 1


@respx.mock
def test_get_paper(s2):
    respx.get(f"{S2_BASE}/paper/abc123").mock(
        return_value=httpx.Response(200, json=FAKE_PAPER_RESPONSE)
    )
    paper = s2.get_paper("abc123")
    assert paper["paper_id"] == "abc123"
    assert paper["citation_count"] == 150


@respx.mock
def test_get_paper_not_found(s2):
    respx.get(f"{S2_BASE}/paper/missing").mock(
        return_value=httpx.Response(404)
    )
    assert s2.get_paper("missing") is None


@respx.mock
def test_search_empty(s2):
    respx.get(f"{S2_BASE}/paper/search").mock(
        return_value=httpx.Response(200, json={"total": 0, "data": []})
    )
    assert s2.search("nonexistent topic xyz") == []
