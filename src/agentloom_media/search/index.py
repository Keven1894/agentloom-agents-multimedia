"""Hybrid search index: one SQLite file, no server.

FTS5 (built in, BM25) for lexical matching and `sqlite-vec` for vector matching, fused with
Reciprocal Rank Fusion. A vector DB or Meilisearch would add a service to operate for no
capability needed at personal-agent scale, and `data/medialoom.db` stays trivially
regenerable from the canonical transcripts.

Two implementation notes that matter for correctness:

- FTS5's default `unicode61` tokenizer splits on whitespace, so a Chinese transcript collapses
  into a handful of enormous tokens and BM25 stops working. The index uses `tokenize='trigram'`,
  which is language-agnostic and needs no extra dependency.
- The embedding model id and vector width are recorded once per index. Mixing embedders would
  put vectors from different spaces in one table, which fails silently rather than loudly, so
  a mismatch is refused.

See plan §3.1-3.3.
"""

from __future__ import annotations

import re
import sqlite3
import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from agentloom_media.search.embed import Embedder, EmbeddingUnavailable, embeddings_enabled
from agentloom_media.search.windows import Window, build_windows

SCHEMA_VERSION = 1

DEFAULT_DB_NAME = "medialoom.db"

# RRF constant. 60 is the value from the original paper and the usual default; it damps the
# influence of any single ranker's top hit.
RRF_K = 60

# How deep to read each ranker before fusing.
CANDIDATE_DEPTH = 50

# FTS5's trigram tokenizer indexes 3-character sequences and cannot match anything shorter.
TRIGRAM_MIN_CHARS = 3

# Cap on OR clauses sent to FTS5, so a long question cannot build a pathological query.
MAX_FTS_CLAUSES = 32

# `\w+` keeps CJK runs together and drops punctuation.
_TERM_PATTERN = re.compile(r"\w+", re.UNICODE)


class IndexError_(RuntimeError):
    """Raised for index-level misuse, such as mixing embedding models."""


@dataclass
class SearchHit:
    """One retrieved time range, ready to render as a deep link."""

    media_id: str
    title: str
    channel: Optional[str]
    url: Optional[str]
    t0: float
    t1: float
    text: str
    score: float
    lexical_rank: Optional[int]
    vector_rank: Optional[int]

    @property
    def anchor_url(self) -> Optional[str]:
        if not self.url:
            return None
        joiner = "&" if "?" in self.url else "?"
        return f"{self.url}{joiner}t={int(self.t0)}s"

    def matched_quote(self, limit: int = 160) -> str:
        text = " ".join(self.text.split())
        return text if len(text) <= limit else text[: limit - 1] + "…"


def default_db_path(repo_root: Path) -> Path:
    return repo_root / "data" / DEFAULT_DB_NAME


def _serialize_vector(vector: Sequence[float]) -> bytes:
    """sqlite-vec accepts float32 blobs."""
    return struct.pack(f"{len(vector)}f", *vector)


def _quote_fts_token(token: str) -> str:
    """Quote one token so FTS5 treats it as a literal, not as syntax.

    FTS5 reads `-`, `*`, `"`, `:` and parentheses as query operators, so any user text must
    be quoted before it reaches MATCH.
    """
    return '"' + token.replace('"', '""') + '"'


def extract_terms(query: str) -> List[str]:
    """Split a query into searchable terms.

    `\\w+` keeps CJK runs intact as single terms, which is the right unit to start from:
    Chinese is not space-delimited, so there is nothing finer to split on without a
    segmenter.
    """
    return [t for t in _TERM_PATTERN.findall(query or "") if t]


def build_fts_query(query: str, max_clauses: int = MAX_FTS_CLAUSES) -> Optional[str]:
    """Build an OR query for the trigram index.

    Phrase-searching the whole query only ever matches verbatim strings, which makes BM25
    contribute nothing to a natural-language question. Instead each term becomes its own
    clause, and long CJK runs are additionally expanded into overlapping 3-grams so a partial
    match can be found — n-gram matching is the usual way to do lexical retrieval on text
    that has no word boundaries.

    Returns None when nothing in the query is long enough for the trigram index; the caller
    then falls back to substring matching.
    """
    clauses: List[str] = []
    seen = set()

    def add(token: str) -> None:
        if len(token) < TRIGRAM_MIN_CHARS or token in seen:
            return
        seen.add(token)
        clauses.append(_quote_fts_token(token))

    for term in extract_terms(query):
        add(term)
        # Expand CJK runs: "还要不要拆" also matches on "还要不", "要不要", "不要拆".
        if len(term) > TRIGRAM_MIN_CHARS and _is_cjk(term):
            for start in range(len(term) - TRIGRAM_MIN_CHARS + 1):
                add(term[start : start + TRIGRAM_MIN_CHARS])

    if not clauses:
        return None
    return " OR ".join(clauses[:max_clauses])


def _is_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


class SearchIndex:
    """Read/write access to the hybrid index."""

    def __init__(self, db_path: Path, embedder: Optional[Embedder] = None):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self._vec_loaded = False
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._load_vec_extension()
        self._create_core_schema()

    # ---- setup ---------------------------------------------------------------------

    def _load_vec_extension(self) -> None:
        """Load sqlite-vec, tolerating environments that forbid extensions.

        Without it the index still works as a lexical index; callers surface that rather than
        pretending the vector half of the ranking ran.
        """
        try:
            import sqlite_vec

            self.conn.enable_load_extension(True)
            sqlite_vec.load(self.conn)
            self.conn.enable_load_extension(False)
            self._vec_loaded = True
        except Exception:
            self._vec_loaded = False

    @property
    def vector_available(self) -> bool:
        return self._vec_loaded

    def _create_core_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS index_meta (
                key TEXT PRIMARY KEY,
                value TEXT
            );

            CREATE TABLE IF NOT EXISTS media (
                media_id TEXT PRIMARY KEY,
                title TEXT,
                channel TEXT,
                url TEXT,
                duration REAL,
                source TEXT,
                engine TEXT,
                model TEXT,
                language TEXT,
                timing_granularity TEXT,
                indexed_at TEXT
            );

            CREATE TABLE IF NOT EXISTS windows (
                window_id INTEGER PRIMARY KEY AUTOINCREMENT,
                media_id TEXT NOT NULL REFERENCES media(media_id) ON DELETE CASCADE,
                t0 REAL NOT NULL,
                t1 REAL NOT NULL,
                text TEXT NOT NULL,
                utterance_ids TEXT,
                embedding_model TEXT
            );

            CREATE INDEX IF NOT EXISTS windows_media_idx ON windows(media_id);

            CREATE VIRTUAL TABLE IF NOT EXISTS windows_fts USING fts5(
                text,
                content='windows',
                content_rowid='window_id',
                tokenize='trigram'
            );
            """
        )
        self._set_meta("schema_version", str(SCHEMA_VERSION))
        self.conn.commit()

    def _ensure_vector_table(self, dimensions: int) -> None:
        """Create the vec0 table, whose width is fixed at creation time."""
        if not self._vec_loaded:
            return

        recorded = self._get_meta("embedding_dimensions")
        if recorded and int(recorded) != dimensions:
            raise IndexError_(
                f"This index holds {recorded}-dimensional vectors but the current embedder "
                f"produces {dimensions}. Vectors from different models are not comparable; "
                "re-index from scratch (delete the database) to switch embedders."
            )

        self.conn.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS windows_vec USING vec0(
                window_id INTEGER PRIMARY KEY,
                embedding FLOAT[{dimensions}]
            )
            """
        )
        self._set_meta("embedding_dimensions", str(dimensions))
        self.conn.commit()

    def _has_vector_table(self) -> bool:
        if not self._vec_loaded:
            return False
        row = self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='windows_vec'"
        ).fetchone()
        return row is not None

    # ---- metadata ------------------------------------------------------------------

    def _get_meta(self, key: str) -> Optional[str]:
        row = self.conn.execute(
            "SELECT value FROM index_meta WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None

    def _set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO index_meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    @property
    def embedding_model(self) -> Optional[str]:
        return self._get_meta("embedding_model")

    def stats(self) -> Dict[str, Any]:
        media_count = self.conn.execute("SELECT COUNT(*) AS n FROM media").fetchone()["n"]
        window_count = self.conn.execute("SELECT COUNT(*) AS n FROM windows").fetchone()["n"]
        vector_count = 0
        if self._has_vector_table():
            vector_count = self.conn.execute(
                "SELECT COUNT(*) AS n FROM windows_vec"
            ).fetchone()["n"]
        return {
            "db_path": str(self.db_path),
            "media": media_count,
            "windows": window_count,
            "vectors": vector_count,
            "embedding_model": self.embedding_model,
            "vector_available": self._has_vector_table(),
        }

    # ---- writing -------------------------------------------------------------------

    def index_media(
        self,
        transcript: Any,
        meta: Dict[str, Any],
        provenance: Optional[Dict[str, Any]] = None,
        window_seconds: Optional[float] = None,
        overlap_seconds: Optional[float] = None,
        embed: bool = True,
    ) -> Dict[str, Any]:
        """Index (or re-index) one media item.

        Re-indexing replaces the item's windows, so the index cannot accumulate stale
        windows from an earlier transcript of the same video.
        """
        provenance = provenance or {}
        media_id = transcript.media_id

        kwargs: Dict[str, Any] = {}
        if window_seconds is not None:
            kwargs["window_seconds"] = window_seconds
        if overlap_seconds is not None:
            kwargs["overlap_seconds"] = overlap_seconds
        windows = build_windows(transcript, **kwargs)

        if not windows:
            return {"media_id": media_id, "windows": 0, "vectors": 0, "embedded": False}

        vectors: List[List[float]] = []
        embedded = False
        embedding_model: Optional[str] = None

        if embed and self._vec_loaded and embeddings_enabled():
            embedder = self.embedder or Embedder()
            existing = self.embedding_model
            if existing and existing != embedder.model:
                raise IndexError_(
                    f"This index was built with embedder '{existing}' but '{embedder.model}' "
                    "was requested. Vectors from different models are not comparable; "
                    "re-index from scratch to switch embedders."
                )
            try:
                vectors = embedder.embed([w.text for w in windows])
                embedding_model = embedder.model
                embedded = bool(vectors)
            except EmbeddingUnavailable:
                vectors = []
                embedded = False

        if embedded and vectors:
            self._ensure_vector_table(len(vectors[0]))
            self._set_meta("embedding_model", embedding_model or "")

        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            """
            INSERT INTO media(media_id, title, channel, url, duration, source, engine,
                              model, language, timing_granularity, indexed_at)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(media_id) DO UPDATE SET
                title = excluded.title, channel = excluded.channel, url = excluded.url,
                duration = excluded.duration, source = excluded.source,
                engine = excluded.engine, model = excluded.model,
                language = excluded.language,
                timing_granularity = excluded.timing_granularity,
                indexed_at = excluded.indexed_at
            """,
            (
                media_id,
                meta.get("title"),
                meta.get("channel"),
                meta.get("url"),
                float(meta.get("duration") or transcript.duration or 0.0),
                provenance.get("source"),
                provenance.get("engine"),
                provenance.get("model"),
                transcript.language,
                transcript.timing_granularity,
                now,
            ),
        )

        self._delete_windows(media_id)

        for position, window in enumerate(windows):
            row = window.to_row()
            cursor = self.conn.execute(
                """
                INSERT INTO windows(media_id, t0, t1, text, utterance_ids, embedding_model)
                VALUES(?, ?, ?, ?, ?, ?)
                """,
                (
                    media_id,
                    row["t0"],
                    row["t1"],
                    row["text"],
                    row["utterance_ids"],
                    embedding_model,
                ),
            )
            window_id = cursor.lastrowid
            self.conn.execute(
                "INSERT INTO windows_fts(rowid, text) VALUES(?, ?)",
                (window_id, row["text"]),
            )
            if embedded and position < len(vectors):
                self.conn.execute(
                    "INSERT INTO windows_vec(window_id, embedding) VALUES(?, ?)",
                    (window_id, _serialize_vector(vectors[position])),
                )

        self.conn.commit()

        return {
            "media_id": media_id,
            "windows": len(windows),
            "vectors": len(vectors) if embedded else 0,
            "embedded": embedded,
            "embedding_model": embedding_model,
        }

    def _delete_windows(self, media_id: str) -> None:
        """Drop an item's windows from all three tables."""
        rows = self.conn.execute(
            "SELECT window_id FROM windows WHERE media_id = ?", (media_id,)
        ).fetchall()
        for row in rows:
            window_id = row["window_id"]
            self.conn.execute(
                "INSERT INTO windows_fts(windows_fts, rowid, text) VALUES('delete', ?, "
                "(SELECT text FROM windows WHERE window_id = ?))",
                (window_id, window_id),
            )
            if self._has_vector_table():
                self.conn.execute(
                    "DELETE FROM windows_vec WHERE window_id = ?", (window_id,)
                )
        self.conn.execute("DELETE FROM windows WHERE media_id = ?", (media_id,))

    # ---- reading -------------------------------------------------------------------

    def _lexical_candidates(
        self, query: str, depth: int, media_id: Optional[str]
    ) -> List[int]:
        """BM25 over trigram terms, with a substring fallback.

        Two things the trigram tokenizer cannot do on its own, both of which bite on Chinese:
        it cannot match a query shorter than 3 characters (and 2-character words like "广告"
        or "结构" are everywhere), and a whole-query phrase search only matches verbatim
        strings. `build_fts_query` handles the second by turning the query into OR'd terms and
        n-grams; anything still unmatched falls back to a substring scan, which is exact but
        unranked and O(n) — acceptable at personal-corpus scale, and the honest alternative to
        silently returning nothing.
        """
        fts_query = build_fts_query(query)
        if fts_query:
            sql = (
                "SELECT f.rowid AS window_id FROM windows_fts f "
                "JOIN windows w ON w.window_id = f.rowid "
                "WHERE windows_fts MATCH ?"
            )
            params: List[Any] = [fts_query]
            if media_id:
                sql += " AND w.media_id = ?"
                params.append(media_id)
            sql += " ORDER BY bm25(windows_fts) LIMIT ?"
            params.append(depth)

            try:
                rows = self.conn.execute(sql, params).fetchall()
            except sqlite3.OperationalError:
                rows = []
            if rows:
                return [row["window_id"] for row in rows]

        return self._substring_candidates(query, depth, media_id)

    def _substring_candidates(
        self, query: str, depth: int, media_id: Optional[str]
    ) -> List[int]:
        """Exact substring match on the query and its terms.

        Windows are ranked by how many distinct terms they contain, so a window matching both
        "广告" and "结构" outranks one matching only "广告".
        """
        terms = [query.strip()] + [
            t for t in extract_terms(query) if t != query.strip()
        ]
        terms = [t for t in terms if t][:MAX_FTS_CLAUSES]
        if not terms:
            return []

        hit_counts: Dict[int, int] = {}
        order: Dict[int, Tuple[str, float]] = {}

        for term in terms:
            sql = "SELECT window_id, media_id, t0 FROM windows WHERE text LIKE ? ESCAPE '\\'"
            pattern = (
                "%"
                + term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                + "%"
            )
            params: List[Any] = [pattern]
            if media_id:
                sql += " AND media_id = ?"
                params.append(media_id)
            sql += " LIMIT ?"
            params.append(depth)

            for row in self.conn.execute(sql, params).fetchall():
                window_id = row["window_id"]
                hit_counts[window_id] = hit_counts.get(window_id, 0) + 1
                order.setdefault(window_id, (row["media_id"], row["t0"]))

        ranked = sorted(
            hit_counts.items(),
            key=lambda item: (-item[1], order.get(item[0], ("", 0.0))),
        )
        return [window_id for window_id, _ in ranked[:depth]]

    def _vector_candidates(
        self, query: str, depth: int, media_id: Optional[str]
    ) -> List[int]:
        if not self._has_vector_table():
            return []
        if not embeddings_enabled():
            return []

        embedder = self.embedder or Embedder(self.embedding_model)
        if self.embedding_model and embedder.model != self.embedding_model:
            # Querying with a different embedder than the index was built with would compare
            # vectors across spaces. Skip the vector half rather than return nonsense.
            return []
        try:
            vector = embedder.embed_one(query)
        except EmbeddingUnavailable:
            return []

        rows = self.conn.execute(
            """
            SELECT window_id FROM windows_vec
            WHERE embedding MATCH ? AND k = ?
            ORDER BY distance
            """,
            (_serialize_vector(vector), depth),
        ).fetchall()
        candidates = [row["window_id"] for row in rows]

        if media_id and candidates:
            placeholders = ",".join("?" * len(candidates))
            kept = {
                row["window_id"]
                for row in self.conn.execute(
                    f"SELECT window_id FROM windows WHERE media_id = ? "
                    f"AND window_id IN ({placeholders})",
                    [media_id, *candidates],
                ).fetchall()
            }
            candidates = [c for c in candidates if c in kept]

        return candidates

    def search(
        self,
        query: str,
        limit: int = 10,
        media_id: Optional[str] = None,
        depth: int = CANDIDATE_DEPTH,
    ) -> Tuple[List[SearchHit], Dict[str, Any]]:
        """Hybrid search over time windows.

        Returns `(hits, info)`. `info` states which rankers actually contributed, so a result
        list produced by lexical matching alone is not mistaken for a hybrid one.
        """
        query = (query or "").strip()
        if not query:
            return [], {"lexical": 0, "vector": 0, "fused": 0, "vector_used": False}

        lexical = self._lexical_candidates(query, depth, media_id)
        vector = self._vector_candidates(query, depth, media_id)

        # Reciprocal Rank Fusion: rank position matters, raw scores from the two rankers are
        # not comparable and must not be added.
        scores: Dict[int, float] = {}
        lexical_ranks: Dict[int, int] = {}
        vector_ranks: Dict[int, int] = {}

        for rank, window_id in enumerate(lexical, start=1):
            lexical_ranks[window_id] = rank
            scores[window_id] = scores.get(window_id, 0.0) + 1.0 / (RRF_K + rank)
        for rank, window_id in enumerate(vector, start=1):
            vector_ranks[window_id] = rank
            scores[window_id] = scores.get(window_id, 0.0) + 1.0 / (RRF_K + rank)

        if not scores:
            return [], {
                "lexical": 0,
                "vector": 0,
                "fused": 0,
                "vector_used": bool(vector),
            }

        ordered = sorted(
            scores.items(),
            key=lambda item: (-item[1], lexical_ranks.get(item[0], 10**6)),
        )[:limit]

        window_ids = [window_id for window_id, _ in ordered]
        placeholders = ",".join("?" * len(window_ids))
        rows = {
            row["window_id"]: row
            for row in self.conn.execute(
                f"""
                SELECT w.window_id, w.media_id, w.t0, w.t1, w.text,
                       m.title, m.channel, m.url
                FROM windows w JOIN media m ON m.media_id = w.media_id
                WHERE w.window_id IN ({placeholders})
                """,
                window_ids,
            ).fetchall()
        }

        hits = []
        for window_id, score in ordered:
            row = rows.get(window_id)
            if row is None:
                continue
            hits.append(
                SearchHit(
                    media_id=row["media_id"],
                    title=row["title"] or row["media_id"],
                    channel=row["channel"],
                    url=row["url"],
                    t0=row["t0"],
                    t1=row["t1"],
                    text=row["text"],
                    score=score,
                    lexical_rank=lexical_ranks.get(window_id),
                    vector_rank=vector_ranks.get(window_id),
                )
            )

        return hits, {
            "lexical": len(lexical),
            "vector": len(vector),
            "fused": len(scores),
            "vector_used": bool(vector),
        }

    def list_media(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT m.*, COUNT(w.window_id) AS window_count
            FROM media m LEFT JOIN windows w ON w.media_id = m.media_id
            GROUP BY m.media_id ORDER BY m.indexed_at DESC
            """
        ).fetchall()
        return [dict(row) for row in rows]

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "SearchIndex":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
