"""bioRxiv / medRxiv preprint search with date filtering and keyword matching."""

import logging
from datetime import datetime, timedelta

import httpx

log = logging.getLogger(__name__)

BIORXIV_API = "https://api.biorxiv.org/details"
# Valid servers: "biorxiv", "medrxiv"
# Endpoint: GET /details/{server}/{start_date}/{end_date}/{cursor}
# Optional query param: ?category=genomics


def search_preprints(
    query: str,
    *,
    server: str = "biorxiv",
    days: int = 7,
    category: str | None = None,
    max_results: int = 20,
    client: httpx.Client | None = None,
) -> list[dict]:
    """Search bioRxiv/medRxiv for recent preprints matching keywords.

    The API only supports date range + category filtering, so keyword matching
    is done client-side against title and abstract.

    Args:
        query: Keywords to match in title/abstract. Space-separated terms are
               ANDed together. Empty string returns all papers in the date range.
        server: "biorxiv" or "medrxiv".
        days: Number of days to look back (default 7).
        category: bioRxiv category filter (e.g. "genomics", "bioinformatics",
                  "genetics"). See https://api.biorxiv.org for valid categories.
        max_results: Maximum papers to return after keyword filtering.
        client: Optional httpx.Client to reuse.
    """
    if server not in ("biorxiv", "medrxiv"):
        raise ValueError(f"server must be 'biorxiv' or 'medrxiv', got '{server}'")

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    # Build keyword patterns (case-insensitive, all terms must match)
    terms = query.lower().split() if query.strip() else []

    own_client = client is None
    if own_client:
        client = httpx.Client(timeout=30)

    try:
        return _fetch_and_filter(
            client, server, start_date, end_date, category, terms, max_results
        )
    finally:
        if own_client:
            client.close()


def _fetch_and_filter(
    client: httpx.Client,
    server: str,
    start_date: str,
    end_date: str,
    category: str | None,
    terms: list[str],
    max_results: int,
) -> list[dict]:
    matched = []
    cursor = 0
    max_pages = 10  # safety limit — 1000 papers max scanned

    for _ in range(max_pages):
        url = f"{BIORXIV_API}/{server}/{start_date}/{end_date}/{cursor}"
        params = {}
        if category:
            params["category"] = category.lower().replace(" ", "_")

        try:
            resp = client.get(url, params=params)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            log.warning("Preprint API request failed: %s", e)
            break

        data = resp.json()
        collection = data.get("collection", [])
        if not collection:
            break

        for item in collection:
            title = item.get("title", "")
            abstract = item.get("abstract", "")
            searchable = f"{title} {abstract}".lower()

            if terms and not all(t in searchable for t in terms):
                continue

            matched.append(_normalize(item, server))
            if len(matched) >= max_results:
                return matched

        # bioRxiv API pages in batches of 100
        if len(collection) < 100:
            break
        cursor += 100

    return matched


def _normalize(item: dict, server: str) -> dict:
    """Convert bioRxiv/medRxiv API response to our standard format."""
    doi = item.get("doi", "")
    version = item.get("version", "1")
    base_url = f"https://www.{server}.org/content"

    return {
        "doi": doi,
        "title": item.get("title", ""),
        "authors": [a.strip() for a in item.get("authors", "").split(";") if a.strip()],
        "abstract": item.get("abstract", ""),
        "date": item.get("date", ""),
        "category": item.get("category", ""),
        "url": f"{base_url}/{doi}v{version}",
        "pdf_url": f"{base_url}/{doi}v{version}.full.pdf",
        "source": server,
        "version": version,
    }
