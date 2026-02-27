import json
import pytest
import respx
import httpx
from fastmcp import Client
from research_mcp.server import create_mcp
from research_mcp.discovery import S2_BASE

FAKE_SEARCH = {
    "total": 1,
    "data": [{
        "paperId": "abc123",
        "title": "Test Paper",
        "abstract": "An abstract.",
        "year": 2024,
        "authors": [{"name": "Alice"}],
        "citationCount": 10,
        "journal": {"name": "Science"},
        "externalIds": {"DOI": "10.1234/test"},
        "openAccessPdf": None,
    }],
}

FAKE_PAPER = FAKE_SEARCH["data"][0]


@pytest.fixture
def data_dir(tmp_path):
    return tmp_path / "data"


@pytest.fixture
def selve_root(tmp_path):
    interpreted = tmp_path / "selve" / "interpreted"
    interpreted.mkdir(parents=True)
    return tmp_path / "selve"


@pytest.fixture
def mcp(data_dir, selve_root):
    return create_mcp(data_dir=data_dir, selve_root=selve_root)


@pytest.mark.anyio
@respx.mock
async def test_search_papers(mcp):
    respx.get(f"{S2_BASE}/paper/search").mock(
        return_value=httpx.Response(200, json=FAKE_SEARCH)
    )
    async with Client(mcp) as client:
        result = await client.call_tool("search_papers", {"query": "test"})
        data = json.loads(result.content[0].text)
        assert len(data) == 1
        assert data[0]["title"] == "Test Paper"


@pytest.mark.anyio
@respx.mock
async def test_save_and_get_paper(mcp):
    respx.get(f"{S2_BASE}/paper/abc123").mock(
        return_value=httpx.Response(200, json=FAKE_PAPER)
    )
    async with Client(mcp) as client:
        save_result = await client.call_tool("save_paper", {"paper_id": "abc123"})
        save_data = json.loads(save_result.content[0].text)
        assert save_data["saved"] == "Test Paper"

        get_result = await client.call_tool("get_paper", {"paper_id": "abc123"})
        get_data = json.loads(get_result.content[0].text)
        assert get_data["title"] == "Test Paper"


@pytest.mark.anyio
@respx.mock
async def test_list_corpus(mcp):
    respx.get(f"{S2_BASE}/paper/abc123").mock(
        return_value=httpx.Response(200, json=FAKE_PAPER)
    )
    async with Client(mcp) as client:
        await client.call_tool("save_paper", {"paper_id": "abc123"})
        result = await client.call_tool("list_corpus", {})
        data = json.loads(result.content[0].text)
        assert len(data) == 1
        assert data[0]["title"] == "Test Paper"


@pytest.mark.anyio
@respx.mock
async def test_export_for_selve(mcp, selve_root):
    respx.get(f"{S2_BASE}/paper/abc123").mock(
        return_value=httpx.Response(200, json=FAKE_PAPER)
    )
    async with Client(mcp) as client:
        await client.call_tool("save_paper", {"paper_id": "abc123"})
        result = await client.call_tool("export_for_selve", {})
        data = json.loads(result.content[0].text)
        assert data["exported"] == 1

    export_path = selve_root / "interpreted" / "research_papers_export.json"
    assert export_path.exists()
    exported = json.loads(export_path.read_text())
    assert len(exported["entries"]) == 1
    assert exported["entries"][0]["source"] == "papers"
