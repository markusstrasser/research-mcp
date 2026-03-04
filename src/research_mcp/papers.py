"""Paper PDF download (Sci-Hub + OA) and full-text extraction.

Extraction pipeline:
  1. Gemini Flash-Lite: upload PDF → structured markdown (tables, figures, sections)
  2. Fallback: pymupdf raw text extraction (offline / API failure)
"""

import logging
import re
import shutil
import urllib.request
from pathlib import Path

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

SCIHUB_MIRRORS = [
    "https://sci-hub.ru",
    "https://sci-hub.ee",
    "https://sci-hub.st",
]

_UA = {"User-Agent": "Mozilla/5.0"}


def download_paper(doi: str, pdf_dir: Path) -> Path | None:
    """Download PDF by DOI. Tries Sci-Hub mirrors, then DOI redirect (OA)."""
    safe_name = re.sub(r"[^\w\-.]", "_", doi) + ".pdf"
    dest = pdf_dir / safe_name

    if dest.exists() and dest.stat().st_size > 1000:
        return dest

    # Try Sci-Hub mirrors
    for base_url in SCIHUB_MIRRORS:
        path = _try_scihub(doi, base_url, dest)
        if path:
            return path

    # Fallback: DOI redirect (works for open-access)
    path = _download_url(f"https://doi.org/{doi}", dest)
    if path:
        return path

    return None


def download_url(url: str, pdf_dir: Path, name: str | None = None) -> Path | None:
    """Download PDF from a direct URL."""
    safe_name = name or re.sub(r"[^\w\-.]", "_", url.split("/")[-1])
    if not safe_name.endswith(".pdf"):
        safe_name += ".pdf"
    dest = pdf_dir / safe_name

    if dest.exists() and dest.stat().st_size > 1000:
        return dest

    return _download_url(url, dest)


_EXTRACT_PROMPT = """\
Convert this PDF to clean, structured markdown. Preserve:
- Section headings (##, ###)
- Tables (as markdown tables)
- Figure/table captions (as **Figure N:** / **Table N:**)
- Equations (as LaTeX in $...$ or $$...$$)
- Reference list at the end
- All text content faithfully

Do NOT summarize. Output the FULL text as markdown."""

EXTRACT_MODEL = "gemini-3-flash-preview"


def extract_text(pdf_path: Path) -> str:
    """Extract structured markdown from PDF.

    Primary: Gemini Flash-Lite (sees rendered pages → structured markdown).
    Fallback: pymupdf raw text (offline / API failure).
    """
    try:
        return _extract_with_gemini(pdf_path)
    except Exception as e:
        logger.warning(f"Gemini extraction failed, falling back to pymupdf: {e}")
        return _extract_with_pymupdf(pdf_path)


def _extract_with_gemini(pdf_path: Path) -> str:
    """Upload PDF to Gemini Flash-Lite → structured markdown."""
    client = genai.Client()

    # Upload the PDF file
    uploaded = client.files.upload(file=pdf_path, config={"mime_type": "application/pdf"})

    response = client.models.generate_content(
        model=EXTRACT_MODEL,
        contents=[
            types.Content(
                parts=[
                    types.Part.from_uri(file_uri=uploaded.uri, mime_type="application/pdf"),
                    types.Part.from_text(text=_EXTRACT_PROMPT),
                ],
            ),
        ],
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=65536,
        ),
    )

    text = response.text or ""
    if len(text) < 100:
        raise ValueError(f"Gemini returned too little text ({len(text)} chars)")
    return text


def _extract_with_pymupdf(pdf_path: Path) -> str:
    """Fallback: raw text extraction via pymupdf."""
    import fitz

    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        text = page.get_text()
        if text.strip():
            pages.append(text)
    doc.close()
    return "\n\n".join(pages)


def _try_scihub(doi: str, base_url: str, dest: Path) -> Path | None:
    """Try downloading from a single Sci-Hub mirror."""
    try:
        url = f"{base_url}/{doi}"
        req = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(req, timeout=30) as resp:
            html = resp.read().decode("utf-8", errors="ignore")

        # Extract PDF URL — try citation_pdf_url meta tag first
        match = re.search(r'citation_pdf_url"\s+content="([^"]+)"', html)
        if not match:
            match = re.search(
                r'(?:src|href|data)\s*=\s*["\']?(/(?:download|storage)/[^"\'>\s]+\.pdf[^"\'>\s]*)',
                html,
            )
        if not match:
            match = re.search(
                r'(?:src|href)\s*=\s*["\']?((?:https?:)?//[^"\'>\s]+\.pdf[^"\'>\s]*)',
                html,
            )
        if not match:
            return None

        pdf_url = match.group(1)
        if pdf_url.startswith("/"):
            pdf_url = base_url + pdf_url
        elif pdf_url.startswith("//"):
            pdf_url = "https:" + pdf_url

        req = urllib.request.Request(pdf_url, headers=_UA)
        with urllib.request.urlopen(req, timeout=60) as resp:
            with open(dest, "wb") as f:
                shutil.copyfileobj(resp, f)

        if dest.stat().st_size > 1000:
            return dest
        dest.unlink(missing_ok=True)

    except Exception as e:
        logger.warning(f"Sci-Hub {base_url} failed for {doi}: {e}")

    return None


def _download_url(url: str, dest: Path) -> Path | None:
    """Download a URL to dest. Returns dest if successful."""
    try:
        req = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(req, timeout=60) as resp:
            content_type = resp.headers.get("Content-Type", "")
            if "pdf" not in content_type and "octet" not in content_type:
                return None
            with open(dest, "wb") as f:
                shutil.copyfileobj(resp, f)
        if dest.stat().st_size > 1000:
            return dest
        dest.unlink(missing_ok=True)
    except Exception as e:
        logger.warning(f"Download failed for {url}: {e}")
    return None
