"""Research MCP server — paper discovery, full-text RAG, and corpus management."""

import hashlib
import json
import logging
import math
import os
from pathlib import Path
from contextlib import asynccontextmanager
from urllib.parse import urlparse

from fastmcp import FastMCP, Context
from mcp.types import ToolAnnotations
from research_mcp.middleware import TelemetryMiddleware
from tenacity import RetryError

from research_mcp.db import PaperDB
from research_mcp.discovery import SemanticScholar
from research_mcp.openalex import OpenAlex
from research_mcp.papers import download_paper, download_url, extract_text
from research_mcp.cag import ask_corpus, ask_corpus_rcs
from research_mcp.rcs import prepare_evidence_async
from research_mcp.extraction import extract_table_async, COLUMN_PRESETS
from research_mcp.exa_verify import get_exa_client, exa_verify_claim
from research_mcp.preprints import search_preprints as _search_preprints

log = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent / "data"
DEFAULT_SELVE_ROOT = Path.home() / "Projects" / "selve"

# -- Annotation presets --
_RO = ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=True)
_RO_LOCAL = ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False)
_WRITE = ToolAnnotations(readOnlyHint=False, idempotentHint=True, openWorldHint=False)


def create_mcp(
    data_dir: Path | None = None,
    selve_root: Path | None = None,
) -> FastMCP:
    data_dir = data_dir or Path(os.environ.get("RESEARCH_MCP_DATA", DEFAULT_DATA_DIR))
    selve_root = selve_root or Path(os.environ.get("SELVE_ROOT", DEFAULT_SELVE_ROOT))
    pdf_dir = data_dir / "pdfs"

    @asynccontextmanager
    async def lifespan(server):
        pdf_dir.mkdir(parents=True, exist_ok=True)
        db = PaperDB(data_dir / "papers.db", check_same_thread=False)
        s2 = SemanticScholar(db, api_key=os.environ.get("S2_API_KEY"))
        oa = OpenAlex(
            db,
            api_key=os.environ.get("OPENALEX_API_KEY"),
            email=os.environ.get("OPENALEX_EMAIL"),
        )
        exa = get_exa_client()
        if exa:
            log.info("Exa client initialized for claim verification")
        else:
            log.info("No EXA_API_KEY — verify_claim will return insufficient")
        yield {"db": db, "s2": s2, "oa": oa, "exa": exa, "selve_root": selve_root, "pdf_dir": pdf_dir}

    mcp = FastMCP(
        "research",
        middleware=[TelemetryMiddleware()],
        instructions=(
            "Research paper discovery and corpus management via Semantic Scholar.\n\n"
            "Workflow:\n"
            "1. search_papers — live S2 search for papers by topic\n"
            "2. save_paper — save a discovered paper to local corpus\n"
            "3. fetch_paper — download PDF and extract full text (Sci-Hub + OA)\n"
            "4. prepare_evidence — score paper chunks for relevance (RCS). Use before ask_papers for better synthesis.\n"
            "5. ask_papers — ask questions against full-text papers (Gemini 1M context). Set use_rcs=True for scored evidence.\n"
            "6. traverse_citations — discover related papers via S2 citation graph (one hop)\n"
            "7. extract_table — Elicit-style structured extraction across papers\n"
            "8. list_corpus / get_paper — browse saved papers\n"
            "9. export_for_selve — export for ./selve update to embed into unified index\n\n"
            "Web source archiving:\n"
            "- save_source — archive a web page (blog post, docs, news) with its content\n"
            "- get_source — retrieve an archived web source by URL\n"
            "- list_sources — browse archived web sources, optionally filter by domain\n\n"
            "Preprint surveillance:\n"
            "- search_preprints — search bioRxiv/medRxiv by date range + keywords (free API, no S2 needed)\n\n"
            "Claim verification:\n"
            "- verify_claim — verify a factual claim against web sources (Exa /answer, cached 7 days)"
        ),
        lifespan=lifespan,
    )

    # ── Discovery ────────────────────────────────────────────────

    @mcp.tool(annotations=_RO, tags={"discovery"})
    def search_papers(
        ctx: Context,
        query: str,
        limit: int = 10,
        backend: str | None = None,
    ) -> list[dict]:
        """Search for papers. Returns titles, abstracts, citation counts.

        Tries Semantic Scholar first, falls back to OpenAlex if S2 is rate-limited.
        Use this to discover papers on a topic. Save interesting ones with save_paper.

        Args:
            query: Search query.
            limit: Max results (capped at 50).
            backend: Force a backend: "s2" or "openalex". If None, tries S2 then falls back.
        """
        s2 = ctx.lifespan_context["s2"]
        oa = ctx.lifespan_context["oa"]
        capped = min(limit, 50)
        results = None
        used_backend = None

        if backend == "openalex":
            try:
                results = oa.search(query, limit=capped)
                used_backend = "openalex"
            except RetryError as e:
                cause = e.last_attempt.exception() if e.last_attempt else e
                log.warning("OpenAlex search failed: %s", cause)
                return {"error": f"OpenAlex unavailable. ({cause})"}
        else:
            # Try S2 first
            try:
                results = s2.search(query, limit=capped)
                used_backend = "s2"
            except RetryError as e:
                if backend == "s2":
                    cause = e.last_attempt.exception() if e.last_attempt else e
                    log.warning("S2 search failed (no fallback): %s", cause)
                    return {"error": f"Semantic Scholar rate-limited or unavailable. ({cause})"}
                # Fall back to OpenAlex
                log.info("S2 search failed, falling back to OpenAlex")
                try:
                    results = oa.search(query, limit=capped)
                    used_backend = "openalex"
                except RetryError as e2:
                    cause = e2.last_attempt.exception() if e2.last_attempt else e2
                    log.warning("Both S2 and OpenAlex failed: %s", cause)
                    return {"error": f"Both Semantic Scholar and OpenAlex unavailable. ({cause})"}

        for r in results:
            if r.get("abstract") and len(r["abstract"]) > 300:
                r["abstract"] = r["abstract"][:300] + "..."
        log.debug("search_papers used backend=%s, %d results", used_backend, len(results))
        return results

    @mcp.tool(annotations=_RO, tags={"discovery"})
    def search_preprints(
        ctx: Context,
        query: str,
        server: str = "biorxiv",
        days: int = 7,
        category: str | None = None,
        max_results: int = 20,
    ) -> list[dict]:
        """Search bioRxiv/medRxiv for recent preprints matching keywords.

        The bioRxiv API supports date-range browsing. Keywords are matched
        client-side against title and abstract (all terms must appear).

        Use this for preprint surveillance — finding new papers in the last
        N days on a topic. For comprehensive literature search, use search_papers
        (Semantic Scholar) instead.

        Args:
            query: Keywords to match in title/abstract. Space-separated terms
                   are ANDed. Empty string returns all papers in the date range.
            server: "biorxiv" or "medrxiv".
            days: Days to look back (default 7).
            category: Optional bioRxiv/medRxiv category filter (e.g. "genomics",
                      "bioinformatics", "genetics", "genetic and genomic medicine").
            max_results: Max papers to return (default 20).
        """
        results = _search_preprints(
            query,
            server=server,
            days=days,
            category=category,
            max_results=max_results,
        )
        for r in results:
            if r.get("abstract") and len(r["abstract"]) > 300:
                r["abstract"] = r["abstract"][:300] + "..."
        return results

    @mcp.tool(annotations=_RO, tags={"discovery"})
    def traverse_citations(
        ctx: Context,
        paper_ids: list[str],
        direction: str = "both",
        auto_save: bool = True,
        limit: int = 20,
    ) -> dict:
        """Discover related papers via S2 citation graph (one hop).

        Finds papers that cite or are cited by the given seed papers.
        For multiple seeds, applies overlap filtering — papers appearing
        in multiple citation lists are ranked higher.

        Args:
            paper_ids: Seed paper IDs to traverse from.
            direction: "references" (cited by seeds), "citations" (citing seeds), or "both".
            auto_save: Auto-save discovered papers to corpus. Default True.
            limit: Max papers to return.
        """
        s2 = ctx.lifespan_context["s2"]
        db = ctx.lifespan_context["db"]

        all_papers = {}
        paper_seeds: dict[str, set] = {}

        for seed_id in paper_ids:
            refs = []
            if direction in ("references", "both"):
                try:
                    refs.extend(s2.get_references(seed_id) or [])
                except RetryError as e:
                    log.warning("get_references failed for %s: %s", seed_id, e)
            if direction in ("citations", "both"):
                try:
                    refs.extend(s2.get_citations(seed_id) or [])
                except RetryError as e:
                    log.warning("get_citations failed for %s: %s", seed_id, e)

            for paper in refs:
                pid = paper["paper_id"]
                if pid not in all_papers:
                    all_papers[pid] = paper
                    paper_seeds[pid] = set()
                paper_seeds[pid].add(seed_id)

        # Overlap filtering for multi-seed (PaperQA2: alpha=0.34)
        min_overlap = math.ceil(0.34 * len(paper_ids))

        filtered = [
            (pid, paper) for pid, paper in all_papers.items()
            if len(paper_seeds[pid]) >= min_overlap
            and pid not in paper_ids  # exclude seeds
        ]

        # Sort by citation count (descending) and limit
        filtered.sort(key=lambda x: x[1].get("citation_count", 0), reverse=True)
        filtered = filtered[:limit]

        # Auto-save
        saved = 0
        if auto_save:
            for pid, paper in filtered:
                db.upsert_paper(paper)
                saved += 1

        return {
            "discovered": len(filtered),
            "saved": saved,
            "direction": direction,
            "seeds": paper_ids,
            "overlap_threshold": min_overlap,
            "papers": [
                {
                    "paper_id": pid,
                    "title": paper.get("title", ""),
                    "year": paper.get("year"),
                    "citations": paper.get("citation_count", 0),
                    "found_via": list(paper_seeds[pid]),
                }
                for pid, paper in filtered
            ],
        }

    # ── Corpus management ────────────────────────────────────────

    @mcp.tool(annotations=_WRITE, tags={"corpus"})
    def save_paper(ctx: Context, paper_id: str) -> dict:
        """Save a paper to the local corpus by its Semantic Scholar paper ID.

        Use after search_papers to persist interesting results. Fetches full
        metadata from S2 and stores locally.
        """
        s2 = ctx.lifespan_context["s2"]
        db = ctx.lifespan_context["db"]
        try:
            paper = s2.get_paper(paper_id)
        except RetryError as e:
            cause = e.last_attempt.exception() if e.last_attempt else e
            log.warning("S2 get_paper failed after retries: %s", cause)
            return {"error": f"Semantic Scholar rate-limited or unavailable. ({cause})"}
        if paper is None:
            return {"error": f"Paper {paper_id} not found on Semantic Scholar"}
        db.upsert_paper(paper)
        return {"saved": paper["title"], "paper_id": paper["paper_id"]}

    @mcp.tool(annotations=_WRITE, tags={"corpus"})
    def fetch_paper(
        ctx: Context,
        paper_id: str | None = None,
        doi: str | None = None,
        url: str | None = None,
    ) -> dict:
        """Download a paper's PDF and extract full text.

        Tries Sci-Hub first (most reliable for paywalled papers), then OA.
        The paper must be saved to the corpus first (via save_paper), OR
        provide a DOI/URL directly.

        Args:
            paper_id: Semantic Scholar paper ID (must be in corpus already).
            doi: DOI to download directly (will also save to corpus).
            url: Direct PDF URL to download.
        """
        db = ctx.lifespan_context["db"]
        pdir = ctx.lifespan_context["pdf_dir"]
        s2 = ctx.lifespan_context["s2"]

        # Resolve DOI
        resolved_doi = doi
        target_paper_id = paper_id

        if paper_id and not doi:
            paper = db.get_paper(paper_id)
            if paper is None:
                return {"error": f"Paper {paper_id} not in corpus. Use save_paper first."}
            resolved_doi = paper.get("doi")
            if not resolved_doi and not url:
                oa_url = paper.get("open_access_url")
                if oa_url:
                    url = oa_url
                else:
                    return {"error": f"Paper {paper_id} has no DOI or OA URL."}

        if doi and not paper_id:
            # Search S2 for the DOI, save it
            results = s2.search(doi, limit=1)
            if results:
                target_paper_id = results[0]["paper_id"]
                db.upsert_paper(results[0])
            else:
                target_paper_id = doi.replace("/", "_")
                db.upsert_paper({"paper_id": target_paper_id, "doi": doi, "title": f"DOI: {doi}"})

        # Download PDF
        pdf_path = None
        if resolved_doi:
            pdf_path = download_paper(resolved_doi, pdir)
        if not pdf_path and url:
            pdf_path = download_url(url, pdir)

        if not pdf_path:
            return {"error": f"Could not download PDF for doi={resolved_doi} url={url}"}

        # Extract text
        full_text = extract_text(pdf_path)
        if not full_text.strip():
            return {"error": f"PDF downloaded but no text extractable: {pdf_path.name}"}

        # Store in DB
        if target_paper_id:
            db.update_paper_pdf(target_paper_id, str(pdf_path), full_text)

        chars = len(full_text)
        est_tokens = chars // 4
        return {
            "paper_id": target_paper_id,
            "pdf": pdf_path.name,
            "size_mb": round(pdf_path.stat().st_size / 1_048_576, 1),
            "text_chars": chars,
            "est_tokens": est_tokens,
            "preview": full_text[:500] + "..." if chars > 500 else full_text,
        }

    @mcp.tool(annotations=_RO_LOCAL, tags={"corpus"})
    def read_paper(ctx: Context, paper_id: str) -> dict:
        """Get full extracted text of a paper. Must have been fetched first."""
        db = ctx.lifespan_context["db"]
        paper = db.get_paper(paper_id)
        if paper is None:
            return {"error": f"Paper {paper_id} not in corpus"}
        if not paper.get("full_text"):
            return {"error": f"Paper {paper_id} has no full text. Use fetch_paper first."}
        return {
            "paper_id": paper["paper_id"],
            "title": paper["title"],
            "text": paper["full_text"],
            "chars": len(paper["full_text"]),
        }

    @mcp.tool(annotations=_RO_LOCAL, tags={"corpus"})
    def get_paper(ctx: Context, paper_id: str) -> dict:
        """Get full details of a saved paper from the local corpus."""
        db = ctx.lifespan_context["db"]
        paper = db.get_paper(paper_id)
        if paper is None:
            return {"error": f"Paper {paper_id} not in local corpus"}
        return paper

    @mcp.tool(annotations=_RO_LOCAL, tags={"corpus"})
    def list_corpus(ctx: Context, limit: int = 50) -> list[dict]:
        """List papers saved in the local corpus, newest-saved first."""
        db = ctx.lifespan_context["db"]
        papers = db.list_papers(limit=limit)
        return [
            {
                "paper_id": p["paper_id"],
                "title": p["title"],
                "year": p.get("year"),
                "citations": p.get("citation_count"),
            }
            for p in papers
        ]

    @mcp.tool(annotations=_WRITE, tags={"corpus"})
    def export_for_selve(ctx: Context) -> dict:
        """Export corpus to selve-compatible JSON for embedding.

        After calling this, run ./selve update to embed papers into the unified index.
        Then search with: ./selve search "query" -s papers
        """
        db = ctx.lifespan_context["db"]
        sr = ctx.lifespan_context["selve_root"]
        entries = db.export_for_selve()
        out = sr / "interpreted" / "research_papers_export.json"
        out.write_text(json.dumps({"entries": entries}, indent=2))
        return {"exported": len(entries), "path": str(out)}

    # ── Synthesis ────────────────────────────────────────────────

    @mcp.tool(annotations=_RO_LOCAL, tags={"synthesis"})
    async def ask_papers(
        ctx: Context,
        question: str,
        paper_ids: list[str] | None = None,
        model: str | None = None,
        use_rcs: bool = False,
    ) -> dict:
        """Ask a question against full-text papers using Gemini's 1M context.

        Two modes:
        - Default (use_rcs=False): stuffs full paper texts into context (CAG).
        - RCS (use_rcs=True): scores chunks for relevance first, then synthesizes
          only the relevant evidence. Higher quality for focused questions.

        Args:
            question: Research question. Be specific for best results.
            paper_ids: Optional list of paper IDs to query. If None, uses all papers with text.
            model: Override model (e.g. 'gemini-3-flash-preview').
            use_rcs: If True, score chunks for relevance before synthesis (slower but more focused).
        """
        db = ctx.lifespan_context["db"]
        papers = db.get_papers_with_text(paper_ids)
        if not papers:
            return {"error": "No papers with full text. Use fetch_paper to download PDFs first."}

        if use_rcs:
            evidence = await prepare_evidence_async(question, papers)
            if not evidence:
                return {"error": "No relevant evidence found after RCS scoring."}
            return ask_corpus_rcs(question, evidence, model=model)

        return ask_corpus(question, papers, model=model)

    @mcp.tool(annotations=_RO_LOCAL, tags={"synthesis"})
    async def prepare_evidence(
        ctx: Context,
        query: str,
        paper_ids: list[str] | None = None,
        min_score: float = 3.0,
    ) -> dict:
        """Score paper text chunks for relevance to a research question (RCS).

        Chunks each paper's full text, scores via Gemini Flash, returns sorted
        summaries with relevance scores. PaperQA2 ablation showed removing this
        step drops accuracy (p<0.001). Use before ask_papers for better synthesis.

        Args:
            query: Research question to score relevance against.
            paper_ids: Papers to process. If None, uses all papers with text.
            min_score: Minimum relevance score (0-10) to include. Default 3.
        """
        db = ctx.lifespan_context["db"]
        papers = db.get_papers_with_text(paper_ids)
        if not papers:
            return {"error": "No papers with full text. Use fetch_paper first."}
        evidence = await prepare_evidence_async(query, papers, min_score=min_score)
        return {
            "query": query,
            "papers_processed": len(papers),
            "evidence_chunks": len(evidence),
            "evidence": evidence,
        }

    @mcp.tool(annotations=_RO_LOCAL, tags={"synthesis"})
    async def extract_table(
        ctx: Context,
        paper_ids: list[str],
        columns: list[dict] | None = None,
        preset: str | None = None,
    ) -> dict:
        """Extract structured data from papers (Elicit-style comparison table).

        Extracts specified columns from each paper in parallel via Gemini Flash.
        Returns a table where each row is a paper and columns are the requested fields.

        Args:
            paper_ids: Papers to extract from.
            columns: Column definitions, e.g. [{"name": "sample_size", "prompt": "Total sample size (N)"}].
            preset: Use a preset column set instead: "clinical" (sample_size, study_design, population, main_finding, effect_size).
        """
        db = ctx.lifespan_context["db"]
        papers = db.get_papers_with_text(paper_ids)
        if not papers:
            return {"error": "No papers with full text. Use fetch_paper first."}

        if preset and preset in COLUMN_PRESETS:
            cols = COLUMN_PRESETS[preset]
        elif columns:
            cols = columns
        else:
            return {"error": "Provide columns or a preset ('clinical')."}

        rows = await extract_table_async(papers, cols)
        return {
            "papers_processed": len(papers),
            "columns": [c["name"] for c in cols],
            "rows": rows,
        }

    # ── Web sources ──────────────────────────────────────────────

    @mcp.tool(annotations=_WRITE, tags={"sources"})
    def save_source(ctx: Context, url: str, title: str, content: str) -> dict:
        """Archive a web source (blog post, docs, news article) with its content.

        Use after fetching a URL via WebFetch/Exa to persist it for later retrieval.
        Automatically extracts domain and computes content hash.

        Args:
            url: The source URL.
            title: Page title.
            content: The fetched content (markdown or plain text).
        """
        db = ctx.lifespan_context["db"]
        domain = urlparse(url).netloc
        content_hash = hashlib.md5(content.encode()).hexdigest()
        db.save_source(url, title, domain, content, content_hash)
        return {"url": url, "title": title, "domain": domain, "chars": len(content)}

    @mcp.tool(annotations=_RO_LOCAL, tags={"sources"})
    def get_source(ctx: Context, url: str) -> dict:
        """Retrieve an archived web source by URL."""
        db = ctx.lifespan_context["db"]
        source = db.get_source(url)
        if source is None:
            return {"error": f"Source not archived: {url}"}
        return source

    @mcp.tool(annotations=_RO_LOCAL, tags={"sources"})
    def list_sources(ctx: Context, limit: int = 50, domain: str | None = None) -> list[dict]:
        """List archived web sources, newest first.

        Args:
            limit: Max results (default 50).
            domain: Optional domain filter (e.g. "arxiv.org").
        """
        db = ctx.lifespan_context["db"]
        return db.list_sources(limit=limit, domain=domain)

    # ── Verification ─────────────────────────────────────────────

    @mcp.tool(annotations=_RO, tags={"verification"})
    def verify_claim(ctx: Context, claim: str) -> dict:
        """Verify a factual claim against web sources via Exa /answer.

        Returns structured verdict with evidence and citations. Cached 7 days.
        Uses Exa's web-grounded LLM — not a cross-model adversarial check.

        Args:
            claim: A specific factual claim to verify (e.g. "SpaceX was valued at $350B in Dec 2024").
        """
        exa = ctx.lifespan_context.get("exa")
        db = ctx.lifespan_context["db"]

        if exa is None:
            return {
                "verdict": "insufficient",
                "evidence_summary": "Exa not configured — set EXA_API_KEY environment variable",
                "confidence": 0.0,
                "citations": [],
                "cost_dollars": None,
                "cached": False,
                "error": "no_api_key",
            }

        return exa_verify_claim(claim, exa, db=db)

    return mcp


def main():
    mcp = create_mcp()
    mcp.run()
