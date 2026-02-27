"""Semantic Scholar API client with caching."""

import hashlib
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from research_mcp.db import PaperDB

S2_BASE = "https://api.semanticscholar.org/graph/v1"
S2_FIELDS = "paperId,title,abstract,year,authors,citationCount,journal,externalIds,openAccessPdf"


class SemanticScholar:
    def __init__(self, db: PaperDB, api_key: str | None = None):
        self.db = db
        headers = {"x-api-key": api_key} if api_key else {}
        self.client = httpx.Client(base_url=S2_BASE, headers=headers, timeout=30)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=10),
        retry=retry_if_exception_type(httpx.HTTPStatusError),
    )
    def search(self, query: str, limit: int = 10) -> list[dict]:
        cache_key = f"search:{hashlib.md5(f'{query}:{limit}'.encode()).hexdigest()}"
        cached = self.db.get_cache(cache_key)
        if cached is not None:
            return cached
        resp = self.client.get(
            "/paper/search",
            params={"query": query, "limit": limit, "fields": S2_FIELDS},
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        results = [self._normalize(p) for p in data]
        self.db.set_cache(cache_key, results)
        return results

    def get_paper(self, paper_id: str) -> dict | None:
        cache_key = f"paper:{paper_id}"
        cached = self.db.get_cache(cache_key)
        if cached is not None:
            return cached
        resp = self.client.get(
            f"/paper/{paper_id}", params={"fields": S2_FIELDS}
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        result = self._normalize(resp.json())
        self.db.set_cache(cache_key, result)
        return result

    def _normalize(self, raw: dict) -> dict:
        authors = [a.get("name", "") for a in raw.get("authors", [])]
        ext_ids = raw.get("externalIds") or {}
        return {
            "paper_id": raw["paperId"],
            "doi": ext_ids.get("DOI"),
            "title": raw.get("title", ""),
            "abstract": raw.get("abstract"),
            "authors": authors,
            "year": raw.get("year"),
            "venue": (raw.get("journal") or {}).get("name"),
            "citation_count": raw.get("citationCount", 0),
            "external_ids": ext_ids,
            "open_access_url": (raw.get("openAccessPdf") or {}).get("url"),
        }
