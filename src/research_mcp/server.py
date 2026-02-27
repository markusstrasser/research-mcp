"""Research MCP server — paper discovery, full-text RAG, and corpus management."""

import json
import os
from pathlib import Path
from contextlib import asynccontextmanager

from fastmcp import FastMCP, Context

from research_mcp.db import PaperDB
from research_mcp.discovery import SemanticScholar
from research_mcp.papers import download_paper, download_url, extract_text
from research_mcp.cag import ask_corpus

DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent / "data"
DEFAULT_SELVE_ROOT = Path.home() / "Projects" / "selve"


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
        yield {"db": db, "s2": s2, "selve_root": selve_root, "pdf_dir": pdf_dir}

    mcp = FastMCP(
        "research",
        instructions=(
            "Research paper discovery and corpus management via Semantic Scholar.\n\n"
            "Workflow:\n"
            "1. search_papers — live S2 search for papers by topic\n"
            "2. save_paper — save a discovered paper to local corpus\n"
            "3. fetch_paper — download PDF and extract full text (Sci-Hub + OA)\n"
            "4. ask_corpus — ask questions against full-text papers (Gemini 1M context)\n"
            "5. list_corpus / get_paper — browse saved papers\n"
            "6. export_for_selve — export for ./selve update to embed into unified index"
        ),
        lifespan=lifespan,
    )

    @mcp.tool()
    def search_papers(ctx: Context, query: str, limit: int = 10) -> list[dict]:
        """Search Semantic Scholar for papers. Returns titles, abstracts, citation counts.

        Use this to discover papers on a topic. Save interesting ones with save_paper.
        """
        s2 = ctx.lifespan_context["s2"]
        results = s2.search(query, limit=min(limit, 50))
        for r in results:
            if r.get("abstract") and len(r["abstract"]) > 300:
                r["abstract"] = r["abstract"][:300] + "..."
        return results

    @mcp.tool()
    def save_paper(ctx: Context, paper_id: str) -> dict:
        """Save a paper to the local corpus by its Semantic Scholar paper ID.

        Use after search_papers to persist interesting results. Fetches full
        metadata from S2 and stores locally.
        """
        s2 = ctx.lifespan_context["s2"]
        db = ctx.lifespan_context["db"]
        paper = s2.get_paper(paper_id)
        if paper is None:
            return {"error": f"Paper {paper_id} not found on Semantic Scholar"}
        db.upsert_paper(paper)
        return {"saved": paper["title"], "paper_id": paper["paper_id"]}

    @mcp.tool()
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

    @mcp.tool()
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

    @mcp.tool()
    def ask_papers(
        ctx: Context,
        question: str,
        paper_ids: list[str] | None = None,
        model: str | None = None,
    ) -> dict:
        """Ask a question against full-text papers using Gemini's 1M context.

        Stuffs all paper texts into Gemini's context window (CAG — no chunking,
        no retrieval, just the full papers). Automatically selects model tier:
        - gemini-2.5-flash-lite for large corpus (>30 papers, cheap)
        - gemini-2.5-flash for focused queries (<=30 papers, more capable)

        Args:
            question: Research question. Be specific for best results.
            paper_ids: Optional list of paper IDs to query. If None, uses all papers with text.
            model: Override model (e.g. 'gemini-2.5-flash', 'gemini-2.5-flash-lite').
        """
        db = ctx.lifespan_context["db"]
        papers = db.get_papers_with_text(paper_ids)
        if not papers:
            return {"error": "No papers with full text. Use fetch_paper to download PDFs first."}
        return ask_corpus(question, papers, model=model)

    @mcp.tool()
    def get_paper(ctx: Context, paper_id: str) -> dict:
        """Get full details of a saved paper from the local corpus."""
        db = ctx.lifespan_context["db"]
        paper = db.get_paper(paper_id)
        if paper is None:
            return {"error": f"Paper {paper_id} not in local corpus"}
        return paper

    @mcp.tool()
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

    @mcp.tool()
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

    return mcp


def main():
    mcp = create_mcp()
    mcp.run()
