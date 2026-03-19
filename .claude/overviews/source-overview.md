```
# INDEX
# -----------------------------------------------------------------------------
# MCP SERVER:
#   - src/research_mcp/server.py: Main application entrypoint, defines MCP tools.
#
# EXTERNAL SERVICE CLIENTS:
#   - src/research_mcp/discovery.py: Semantic Scholar API client.
#   - src/research_mcp/openalex.py: OpenAlex API client (fallback).
#   - src/research_mcp/exa_verify.py: Exa API client for claim verification.
#
# CORE INFRASTRUCTURE:
#   - src/research_mcp/papers.py: PDF download and full-text extraction pipeline.
#   - src/research_mcp/cag.py: Cache-Augmented Generation (large-context RAG).
#
# DATA LAYER:
#   - src/research_mcp/db.py: SQLite database for paper metadata and caching.
#
# TESTING & CONFIGURATION:
#   - tests/*: Unit and integration tests for server and clients.
#   - pyproject.toml: Project dependencies and configuration.
# -----------------------------------------------------------------------------

# =============================================================================
# MCP SERVER
# =============================================================================

# src/research_mcp/server.py
# Defines the FastMCP server, its tools, and application lifecycle.
# Key Functions: create_mcp(), search_papers(), save_paper(), fetch_paper(), ask_papers(), verify_claim()
# Dependencies: fastmcp, db, discovery, openalex, papers, cag, exa_verify

# =============================================================================
# EXTERNAL SERVICE CLIENTS (HTTP, Caching, Retries)
# =============================================================================

# src/research_mcp/discovery.py
# Implements a robust client for the Semantic Scholar API.
# Key Classes/Functions: SemanticScholar, search(), get_paper()
# Infrastructure:
#   - HTTP Client: httpx.Client for API requests.
#   - Caching: Uses PaperDB for caching API responses (search, get_paper).
#   - Retries: Uses tenacity for exponential backoff on 429/5xx errors.
# Dependencies: httpx, tenacity, db.PaperDB

# src/research_mcp/openalex.py
# Implements a client for the OpenAlex API, used as a fallback for Semantic Scholar.
# Key Classes/Functions: OpenAlex, search(), get_paper(), _reconstruct_abstract()
# Infrastructure:
#   - HTTP Client: httpx.Client for API requests.
#   - Caching: Uses PaperDB to cache API responses.
#   - Retries: Uses tenacity for exponential backoff.
# Dependencies: httpx, tenacity, db.PaperDB

# src/research_mcp/exa_verify.py
# Provides claim verification using the Exa /answer API with structured output.
# Key Functions: get_exa_client(), exa_verify_claim()
# Infrastructure:
#   - HTTP Client: exa-py library.
#   - Caching: Uses PaperDB to cache verification results for 7 days.
#   - Structured Output: Defines and requests a JSON schema from the Exa API.
# Dependencies: exa-py, db.PaperDB

# =============================================================================
# CORE INFRASTRUCTURE (PDF Processing, RAG)
# =============================================================================

# src/research_mcp/papers.py
# Handles downloading paper PDFs and extracting their full text.
# Key Functions: download_paper(), download_url(), extract_text()
# Infrastructure:
#   - PDF Download: Multi-source strategy (Sci-Hub mirrors, then direct OA URL).
#   - PDF Extraction: Two-tiered pipeline:
#     1. Primary: `google-genai` (Gemini Flash-Lite) for structured markdown extraction.
#     2. Fallback: `pymupdf` for raw text extraction on API failure.
# Dependencies: google-genai, pymupdf

# src/research_mcp/cag.py
# Implements Cache-Augmented Generation (CAG) for RAG over full papers.
# Key Functions: ask_corpus()
# Infrastructure:
#   - Large-Context RAG: Stuffs full paper texts into Gemini's 1M+ context window.
#   - Model Tiering: Auto-selects between cheaper/faster and more capable models.
# Dependencies: google-genai

# =============================================================================
# DATA LAYER
# =============================================================================

# src/research_mcp/db.py
# Provides a persistent SQLite database for papers, web sources, and API caches.
# Key Classes/Functions: PaperDB, upsert_paper(), get_paper(), get_cache(), set_cache(), save_source()
# Infrastructure:
#   - Persistence: SQLite for storing paper metadata and extracted text.
#   - Caching: Implements a generic key-value store with TTL logic, used by API clients.
# Dependencies: sqlite3

# =============================================================================
# TESTING & CONFIGURATION
# =============================================================================

# tests/test_server.py, tests/test_db.py, tests/test_discovery.py, tests/test_openalex.py
# Unit and integration tests for the application components.
# Infrastructure:
#   - Test Runner: pytest.
#   - HTTP Mocking: `respx` library to mock API calls for client tests.
# Dependencies: pytest, respx, httpx

# pyproject.toml
# Defines project metadata, dependencies, and entry points.
# Key Sections: [project], [project.scripts]
# Infrastructure:
#   - Defines the `papers-mcp` console script.
# Dependencies: fastmcp, httpx, tenacity, google-genai, pymupdf, exa-py
```
