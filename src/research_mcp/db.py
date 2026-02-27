"""SQLite store for paper metadata and response cache."""

import json
import sqlite3
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    paper_id TEXT PRIMARY KEY,
    doi TEXT,
    title TEXT NOT NULL,
    abstract TEXT,
    authors TEXT,
    year INTEGER,
    venue TEXT,
    citation_count INTEGER,
    external_ids TEXT,
    open_access_url TEXT,
    pdf_path TEXT,
    full_text TEXT,
    saved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS cache (
    cache_key TEXT PRIMARY KEY,
    response TEXT NOT NULL,
    cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

MIGRATIONS = [
    "ALTER TABLE papers ADD COLUMN pdf_path TEXT",
    "ALTER TABLE papers ADD COLUMN full_text TEXT",
]


class PaperDB:
    def __init__(self, db_path: Path, check_same_thread: bool = True):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=check_same_thread)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self.conn.executescript(SCHEMA)
        self._migrate()

    def _migrate(self):
        for sql in MIGRATIONS:
            try:
                self.conn.execute(sql)
            except sqlite3.OperationalError:
                pass  # column already exists

    def execute(self, sql: str, params=()) -> sqlite3.Cursor:
        return self.conn.execute(sql, params)

    def upsert_paper(self, paper: dict[str, Any]) -> None:
        authors = json.dumps(paper.get("authors", []))
        external_ids = json.dumps(paper.get("external_ids", {}))
        self.conn.execute(
            """INSERT INTO papers
                   (paper_id, doi, title, abstract, authors, year, venue,
                    citation_count, external_ids, open_access_url)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(paper_id) DO UPDATE SET
                   doi=excluded.doi, title=excluded.title,
                   abstract=excluded.abstract, authors=excluded.authors,
                   year=excluded.year, venue=excluded.venue,
                   citation_count=excluded.citation_count,
                   external_ids=excluded.external_ids,
                   open_access_url=excluded.open_access_url""",
            (paper["paper_id"], paper.get("doi"), paper["title"],
             paper.get("abstract"), authors, paper.get("year"),
             paper.get("venue"), paper.get("citation_count"),
             external_ids, paper.get("open_access_url")),
        )
        self.conn.commit()

    def get_paper(self, paper_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM papers WHERE paper_id = ?", (paper_id,)
        ).fetchone()
        return self._row_to_dict(row) if row else None

    def list_papers(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM papers ORDER BY saved_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def export_for_selve(self) -> list[dict[str, Any]]:
        """Export papers in selve-compatible format for embedding."""
        rows = self.conn.execute("SELECT * FROM papers").fetchall()
        entries = []
        for row in rows:
            p = self._row_to_dict(row)
            parts = [p["title"]]
            if p.get("abstract"):
                parts.append(p["abstract"])
            if p.get("authors"):
                parts.append("Authors: " + ", ".join(p["authors"]))
            if p.get("venue"):
                parts.append(f"Venue: {p['venue']}")
            entries.append({
                "id": f"s2_{p['paper_id']}",
                "source": "papers",
                "title": p["title"],
                "date": f"{p['year']}-01-01" if p.get("year") else "",
                "text": "\n\n".join(parts),
                "metadata": {
                    "doi": p.get("doi"),
                    "citation_count": p.get("citation_count"),
                    "open_access_url": p.get("open_access_url"),
                    "external_ids": p.get("external_ids", {}),
                },
            })
        return entries

    def _row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        for field in ("authors", "external_ids"):
            if d.get(field):
                d[field] = json.loads(d[field])
        return d

    def update_paper_pdf(self, paper_id: str, pdf_path: str, full_text: str) -> None:
        self.conn.execute(
            "UPDATE papers SET pdf_path = ?, full_text = ? WHERE paper_id = ?",
            (pdf_path, full_text, paper_id),
        )
        self.conn.commit()

    def get_papers_with_text(self, paper_ids: list[str] | None = None) -> list[dict[str, Any]]:
        """Get papers that have full_text extracted."""
        if paper_ids:
            placeholders = ",".join("?" * len(paper_ids))
            rows = self.conn.execute(
                f"SELECT * FROM papers WHERE full_text IS NOT NULL AND paper_id IN ({placeholders})",
                paper_ids,
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM papers WHERE full_text IS NOT NULL ORDER BY saved_at DESC"
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_cache(self, key: str, max_age_days: int = 7) -> dict | None:
        row = self.conn.execute(
            """SELECT response FROM cache
               WHERE cache_key = ? AND cached_at > datetime('now', ?)""",
            (key, f"-{max_age_days} days"),
        ).fetchone()
        return json.loads(row[0]) if row else None

    def set_cache(self, key: str, response: dict) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO cache (cache_key, response) VALUES (?, ?)",
            (key, json.dumps(response)),
        )
        self.conn.commit()
