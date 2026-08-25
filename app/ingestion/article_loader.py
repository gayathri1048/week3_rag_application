"""Read articles out of `documents/help_articles/` — Markdown with YAML front matter.

Expected file format
--------------------
---
article_id: BM-001
product_area: Billing
last_updated: 2025-11-01
tags: billing, migration, error-codes
---
# Article Title

... body with prose and Markdown tables ...

The four mandatory Week-3 metadata fields (source_file, article_id,
product_area, last_updated) are extracted here and passed through to the
chunkers so every chunk carries them.
"""

from __future__ import annotations

import logging
import re
from datetime import date
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ..schemas import Article

logger = logging.getLogger(__name__)

FRONT_MATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


def load_articles(articles_dir: Path) -> tuple[list[Article], int]:
    """Load every .md file in `articles_dir` as an Article.

    Returns (articles, files_read).
    Malformed files are logged and skipped — one bad file must not block the rest.
    """
    if not articles_dir.exists():
        raise FileNotFoundError(f"articles directory not found: {articles_dir}")

    articles: list[Article] = []
    seen_ids: set[str] = set()
    files_read = 0

    for path in sorted(articles_dir.glob("*.md")):
        if path.name.startswith((".", "_")):
            continue
        files_read += 1
        article = _parse_article(path)
        if article is None:
            continue
        if article.article_id in seen_ids:
            logger.warning("duplicate article_id %s in %s — skipping", article.article_id, path.name)
            continue
        seen_ids.add(article.article_id)
        articles.append(article)

    logger.info("loaded %d articles from %d files", len(articles), files_read)
    return articles, files_read


def _parse_article(path: Path) -> Article | None:
    """Parse one Markdown article file into an Article model."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.error("could not read %s: %s", path, exc)
        return None

    front: dict[str, Any] = {}
    if m := FRONT_MATTER_RE.match(text):
        front = _parse_front_matter(m.group(1))
        text = text[m.end():]

    # Extract title from first H1 heading
    title = front.get("title", "")
    if heading := re.search(r"^#\s+(.+)$", text, re.MULTILINE):
        title = title or heading.group(1).strip()
        text = text[heading.end():]

    title = title or path.stem.replace("-", " ").replace("_", " ").title()
    body = text.strip()

    # Parse tags
    raw_tags = front.get("tags", "")
    tags = [t.strip() for t in raw_tags.split(",") if t.strip()] if isinstance(raw_tags, str) else []

    # Parse last_updated
    raw_date = front.get("last_updated", "")
    try:
        last_updated = date.fromisoformat(str(raw_date))
    except (ValueError, TypeError):
        logger.warning("%s: invalid last_updated %r — using today", path.name, raw_date)
        last_updated = date.today()

    raw: dict[str, Any] = {
        "article_id": front.get("article_id", path.stem.upper()),
        "title": title,
        "body": body,
        "source_file": path.name,          # mandatory Week-3 field
        "product_area": front.get("product_area", "General"),
        "last_updated": last_updated,
        "tags": tags,
    }

    try:
        return Article(**raw)
    except ValidationError as exc:
        logger.warning("article %s failed validation: %s", path.name, exc)
        return None


def _parse_front_matter(block: str) -> dict[str, Any]:
    """Parse simple key: value YAML front matter (no nested structures)."""
    result: dict[str, Any] = {}
    for line in block.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            result[key.strip()] = value.strip()
    return result
