"""Exa /answer-based claim verification.

Uses Exa's answer endpoint to verify factual claims against web sources.
Returns structured verdicts with evidence and citations. Results cached 7 days.

Note: Uses Exa's internal LLM (model family unverified) — not a cross-model
adversarial check. Provides web-grounded verification, not independent reasoning.
"""

import hashlib
import json
import logging
import os
import time
from typing import Any

log = logging.getLogger(__name__)

VERIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["supported", "contradicted", "insufficient"],
            "description": "Whether web evidence supports the claim",
        },
        "evidence_summary": {
            "type": "string",
            "description": "Brief summary of the evidence found",
        },
        "confidence": {
            "type": "number",
            "description": "Confidence in the verdict from 0 to 1",
        },
    },
    "required": ["verdict", "evidence_summary", "confidence"],
}


def get_exa_client():
    """Get an Exa client from environment. Returns None if no API key."""
    api_key = os.environ.get("EXA_API_KEY", "")
    if not api_key:
        # Try to extract from .mcp.json (same pattern as SAFE-lite)
        from pathlib import Path

        for mcp_path in [
            Path.home() / "Projects" / "meta" / ".mcp.json",
            Path.home() / ".mcp.json",
        ]:
            if mcp_path.exists():
                try:
                    with open(mcp_path) as f:
                        config = json.load(f)
                    for server in config.get("mcpServers", {}).values():
                        url = server.get("url", "")
                        if "exaApiKey=" in url:
                            api_key = url.split("exaApiKey=")[1].split("&")[0]
                            break
                except (json.JSONDecodeError, OSError):
                    continue
            if api_key:
                break

    if not api_key:
        return None

    from exa_py import Exa

    return Exa(api_key=api_key)


def _cache_key(claim: str) -> str:
    """Deterministic cache key for a claim."""
    normalized = claim.strip().lower()
    return "exa_verify:" + hashlib.sha256(normalized.encode()).hexdigest()[:16]


def exa_verify_claim(
    claim: str,
    exa,
    db=None,
    *,
    no_cache: bool = False,
) -> dict[str, Any]:
    """Verify a factual claim via Exa /answer with structured output.

    Args:
        claim: The factual claim to verify.
        exa: An Exa client instance.
        db: Optional PaperDB for caching (7-day TTL).
        no_cache: Skip cache read (still writes to cache).

    Returns:
        Dict with verdict, evidence_summary, confidence, citations, cost_dollars.
    """
    # Check cache
    if db and not no_cache:
        key = _cache_key(claim)
        cached = db.get_cache(key, max_age_days=7)
        if cached:
            cached["cached"] = True
            return cached

    start = time.monotonic()
    try:
        response = exa.answer(
            f"Is this claim true or false? Evaluate the evidence: {claim}",
            text=True,
            output_schema=VERIFICATION_SCHEMA,
        )
    except Exception as e:
        log.warning("Exa /answer failed for claim: %s", e)
        return {
            "verdict": "insufficient",
            "evidence_summary": f"Exa /answer error: {e}",
            "confidence": 0.0,
            "citations": [],
            "cost_dollars": None,
            "cached": False,
            "latency_ms": round((time.monotonic() - start) * 1000),
            "error": str(e),
        }
    latency_ms = round((time.monotonic() - start) * 1000)

    # Parse answer — may be dict (schema worked) or str (fallback)
    answer = response.answer
    if isinstance(answer, dict):
        verdict = answer.get("verdict", "insufficient")
        evidence_summary = answer.get("evidence_summary", "")
        confidence = answer.get("confidence", 0.5)
        schema_valid = True
    elif isinstance(answer, str):
        # Schema fallback — raw string answer
        verdict = "insufficient"
        evidence_summary = answer
        confidence = 0.0
        schema_valid = False
        log.info("Exa returned string instead of schema dict — falling back")
    else:
        verdict = "insufficient"
        evidence_summary = str(answer)
        confidence = 0.0
        schema_valid = False

    # Normalize verdict
    if verdict not in ("supported", "contradicted", "insufficient"):
        verdict = "insufficient"

    # Build citations from response
    citations = []
    key_source = None
    for c in response.citations or []:
        cite = {
            "url": c.url,
            "title": c.title,
            "published_date": c.published_date,
        }
        citations.append(cite)
        if key_source is None:
            key_source = c.url

    # Extract cost
    cost_dollars = None
    if response.cost_dollars:
        cost_dollars = response.cost_dollars.total

    result = {
        "verdict": verdict,
        "evidence_summary": evidence_summary,
        "confidence": confidence,
        "key_source": key_source,
        "citations": citations,
        "cost_dollars": cost_dollars,
        "cached": False,
        "schema_valid": schema_valid,
        "latency_ms": latency_ms,
    }

    # Cache the result
    if db:
        db.set_cache(_cache_key(claim), result)

    return result
