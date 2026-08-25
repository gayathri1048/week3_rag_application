"""Strategy 2 — Structure-aware table chunker.

Core rule: a table row is NEVER separated from its header row.

Algorithm:
  1. Walk the article body line-by-line, tracking whether we are inside a
     Markdown table or in prose.
  2. A "table block" is: the header row + separator row + all data rows that
     follow without a blank line break.
  3. Each table block is kept intact as one unit.  If a table block alone
     exceeds chunk_size it is hard-split AT THE ROW BOUNDARY (never mid-row),
     carrying the header row into every sub-chunk so a retrieved chunk is
     always self-explanatory.
  4. Prose blocks between tables are packed with the same paragraph-overlap
     logic as Strategy 1.

Week-3 required metadata: source_file, article_id, product_area, last_updated
are stamped on every chunk (same as Strategy 1).  The extra key
  chunker = "table_aware"
distinguishes these chunks from Strategy-1 chunks inside Chroma.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Matches a Markdown table row: starts with optional whitespace then |
_TABLE_ROW = re.compile(r"^\s*\|")
# Matches the separator row:  | --- | --- |  (dashes, colons, spaces inside pipes)
_TABLE_SEP = re.compile(r"^\s*\|[\s\-:|]+\|")


@dataclass
class Chunk:
    """Embeddable unit (shared shape with Strategy-1 Chunk)."""

    chunk_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ================================================================ public API =

def chunk_article_table_aware(
    article: Any,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> list[Chunk]:
    """Split one Article using the structure-aware strategy (Strategy 2).

    Table rows are never separated from their header row.
    """
    blocks = _parse_blocks(article.body.strip())
    pieces = _pack_blocks(blocks, chunk_size, chunk_overlap)
    header = f"Article {article.article_id} — {article.title}"

    return [
        Chunk(
            chunk_id=f"{article.article_id}::t{index}",
            text=f"{header}\n\n{piece}",
            metadata=_article_metadata(article, index, len(pieces)),
        )
        for index, piece in enumerate(pieces)
    ]


# ============================================================= block parsing =

@dataclass
class _Block:
    """A contiguous region of the article body, either prose or a table."""

    kind: str           # "prose" | "table"
    lines: list[str]
    header_lines: list[str] = field(default_factory=list)  # for table blocks only

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    def __len__(self) -> int:
        return len(self.text)


def _parse_blocks(body: str) -> list[_Block]:
    """Walk lines and emit alternating prose / table blocks."""
    blocks: list[_Block] = []
    prose_lines: list[str] = []
    table_lines: list[str] = []
    header_lines: list[str] = []
    in_table = False

    def _flush_prose() -> None:
        text = "\n".join(prose_lines).strip()
        if text:
            blocks.append(_Block(kind="prose", lines=text.splitlines()))
        prose_lines.clear()

    def _flush_table() -> None:
        if table_lines:
            blocks.append(
                _Block(kind="table", lines=list(table_lines), header_lines=list(header_lines))
            )
        table_lines.clear()
        header_lines.clear()

    for line in body.splitlines():
        if _TABLE_ROW.match(line):
            if not in_table:
                # entering a table — flush any pending prose
                _flush_prose()
                in_table = True
            table_lines.append(line)
            # capture header (first row) and separator (second row)
            if len(table_lines) <= 2:
                header_lines.append(line)
        else:
            if in_table:
                # leaving a table
                _flush_table()
                in_table = False
            prose_lines.append(line)

    # flush whatever is left
    if in_table:
        _flush_table()
    else:
        _flush_prose()

    return blocks


# ============================================================= block packing =

def _pack_blocks(blocks: list[_Block], chunk_size: int, chunk_overlap: int) -> list[str]:
    """Pack blocks into final chunk strings, respecting the table-intact rule."""
    pieces: list[str] = []
    current = ""

    for block in blocks:
        if block.kind == "table":
            # Each table block is kept whole (or row-split if it's enormous)
            table_pieces = _split_table_block(block, chunk_size)
            for tp in table_pieces:
                candidate = f"{current}\n\n{tp}".strip() if current else tp
                if len(candidate) <= chunk_size:
                    current = candidate
                else:
                    if current:
                        pieces.append(current)
                    current = tp
        else:
            # prose: paragraph-pack with overlap
            for para in block.text.split("\n\n"):
                para = para.strip()
                if not para:
                    continue
                sub_paras = [para] if len(para) <= chunk_size else _hard_split(para, chunk_size)
                for sp in sub_paras:
                    candidate = f"{current}\n\n{sp}".strip() if current else sp
                    if len(candidate) <= chunk_size:
                        current = candidate
                    else:
                        if current:
                            pieces.append(current)
                            tail = current[-chunk_overlap:] if chunk_overlap else ""
                            current = f"{tail}\n\n{sp}".strip() if tail else sp
                        else:
                            current = sp

    if current:
        pieces.append(current)
    return pieces or [""]


def _split_table_block(block: _Block, chunk_size: int) -> list[str]:
    """Split an oversized table AT row boundaries, prepending the header to each sub-chunk."""
    header = "\n".join(block.header_lines)
    data_rows = block.lines[len(block.header_lines):]

    if len(block.text) <= chunk_size:
        return [block.text]

    pieces: list[str] = []
    current_rows: list[str] = list(block.header_lines)

    for row in data_rows:
        candidate = "\n".join(current_rows + [row])
        if len(candidate) <= chunk_size:
            current_rows.append(row)
        else:
            if len(current_rows) > len(block.header_lines):
                pieces.append("\n".join(current_rows))
            # start a new sub-chunk with the header prepended
            current_rows = block.header_lines + [row]

    if current_rows:
        pieces.append("\n".join(current_rows))

    return pieces if pieces else [block.text]


# ============================================================= metadata ======

def _article_metadata(article: Any, index: int, total: int) -> dict[str, Any]:
    """Week-3 required metadata fields stamped on every article chunk."""
    meta: dict[str, Any] = {
        "doc_type": "article",
        # --- Week-3 required fields ---
        "source_file": article.source_file,
        "article_id": article.article_id,
        "product_area": article.product_area,
        "last_updated": str(article.last_updated),
        # --- housekeeping ---
        "subject": article.title,
        "chunk_index": index,
        "chunk_total": total,
        "chunker": "table_aware",         # strategy tag — different from Strategy 1
    }
    if article.tags:
        meta["tags"] = ", ".join(article.tags)
    return meta


# ============================================================= shared util ===

def _hard_split(para: str, chunk_size: int) -> list[str]:
    """Break an oversized prose paragraph on sentence boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", para)
    out: list[str] = []
    current = ""
    for sentence in sentences:
        while len(sentence) > chunk_size:
            out.append(sentence[:chunk_size])
            sentence = sentence[chunk_size:]
        candidate = f"{current} {sentence}".strip()
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                out.append(current)
            current = sentence
    if current:
        out.append(current)
    return out
