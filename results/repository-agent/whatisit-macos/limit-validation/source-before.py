"""Search explicitly captured local manuals; no command execution or network access."""

from contextlib import closing
import hashlib
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Manual:
    tool: str
    source: str
    tool_path: str
    macos_version: str
    captured_at: str
    text: str


def build_index(path: Path, manuals: list[Manual]) -> None:
    """Create a new FTS5 index; refuse to overwrite an existing capture."""
    if not manuals:
        raise ValueError('At least one manual is required.')
    with path.open('xb'):
        pass
    try:
        with closing(sqlite3.connect(path)) as connection, connection:
            connection.execute('CREATE VIRTUAL TABLE docs USING fts5(tool, text, '
                               'source UNINDEXED, tool_path UNINDEXED, macos_version UNINDEXED, '
                               'captured_at UNINDEXED, sha256 UNINDEXED, chunk UNINDEXED)')
            for manual in manuals:
                if not all((manual.tool, manual.source, manual.tool_path, manual.macos_version,
                            manual.captured_at, manual.text.strip())):
                    raise ValueError('Manual text and provenance must be nonempty.')
                digest = hashlib.sha256(manual.text.encode()).hexdigest()
                # shortcut: fixed line windows may split sections; use section parsing if retrieval needs it.
                lines = manual.text.splitlines()
                for offset in range(0, len(lines), 30):
                    text = '\n'.join(lines[offset:offset + 35])
                    connection.execute('INSERT INTO docs VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                                       (manual.tool, text, manual.source, manual.tool_path,
                                        manual.macos_version, manual.captured_at, digest, offset // 30))
    except Exception:
        path.unlink()
        raise


def search(path: Path, request: str, limit: int = 3) -> list[dict]:
    """Return lexical evidence, never a command or a claim of task correctness."""
    if not 1 <= limit <= 10:
        raise ValueError('limit must be between 1 and 10')
    stopwords = {'a', 'all', 'an', 'and', 'by', 'command', 'displays', 'for', 'how', 'i',
                 'in', 'is', 'it', 'mac', 'macos', 'me', 'my', 'of', 'on', 'or', 'show',
                 'specific', 'the', 'to', 'what', 'which', 'with', 'your'}
    tokens = sorted(set(re.findall(r'[a-z0-9_]+', request.casefold())) - stopwords)
    if not tokens:
        return []
    expression = ' OR '.join('"' + token + '"' for token in tokens)
    with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute('PRAGMA query_only = ON')
        rows = connection.execute('SELECT tool, text, source, tool_path, macos_version, '
                                  'captured_at, sha256, chunk FROM docs WHERE docs MATCH ? '
                                  'ORDER BY bm25(docs, 3.0, 1.0), rowid LIMIT ?',
                                  (expression, limit)).fetchall()
        return [dict(row) for row in rows]
