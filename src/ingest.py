"""
src/ingest.py — Policy document ingestion pipeline.

Responsibilities:
- Load markdown files from data/policies/
- Split documents into overlapping chunks, preserving §section metadata
- Embed chunks using a local sentence-transformers model (no API key)
- Upsert chunks into a persistent ChromaDB collection (idempotent)

Run directly:
    python -m src.ingest
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import List

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

POLICIES_DIR       = Path(os.getenv("POLICIES_DIR",       "data/policies"))
CHROMA_PERSIST_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", "data/chroma_db"))
COLLECTION_NAME    = "compliance_policies"

CHUNK_SIZE    = 512   # characters per chunk
CHUNK_OVERLAP = 64    # overlap between consecutive chunks

# Chunks shorter than this after stripping are almost always header-only
# fragments (e.g. "## 3. Retention Periods by Record Type") that carry no
# policy content, produce §unknown citations, and hurt retrieval precision.
MIN_CHUNK_CONTENT_LENGTH = 60  # characters

# Local embedding model — runs entirely on-device, no API key needed.
# all-MiniLM-L6-v2: 22 MB, 384-dim, good quality/speed trade-off.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# ChromaDB client  (module-level singleton, lazy-initialised)
# ---------------------------------------------------------------------------

_client: chromadb.PersistentClient | None = None
_collection = None


def get_chroma_client() -> chromadb.PersistentClient:
    """Return (or create) the persistent ChromaDB client."""
    global _client
    if _client is None:
        CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    return _client


def get_or_create_collection(client: chromadb.PersistentClient):
    """
    Return (or create) the compliance_policies ChromaDB collection.

    Uses SentenceTransformerEmbeddingFunction so that both ingest and query
    time use the same local embeddings — no embedding API needed.
    """
    global _collection
    if _collection is None:
        ef = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},   # cosine similarity
        )
    return _collection


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------

def load_policy_files(policies_dir: Path = POLICIES_DIR) -> List[dict]:
    """
    Recursively find all *.md files under `policies_dir`.

    Returns a list of dicts:
        {
            "source":  "<relative path from project root, e.g. data/policies/aml.md>",
            "content": "<raw markdown text>",
        }
    """
    docs = []
    for md_path in sorted(policies_dir.rglob("*.md")):
        try:
            content = md_path.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"  [WARN] Cannot read {md_path}: {exc}")
            continue
        # Store relative path so section IDs are portable
        try:
            relative = md_path.relative_to(Path.cwd())
        except ValueError:
            relative = md_path
        docs.append({"source": str(relative), "content": content})
    return docs


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _extract_section_id(text: str, chunk_index: int) -> str:
    """
    Derive a §section label from the first markdown header found in `text`.

    Falls back to §chunk-N if no header is present.

    Examples:
        "## 3. Data Retention\n…"  →  "§3-data-retention"
        (no header)                →  "§chunk-4"
    """
    m = re.search(r"^#{1,6}\s+(.+)$", text, re.MULTILINE)
    if m:
        slug = re.sub(r"[^\w\s-]", "", m.group(1)).strip()
        slug = re.sub(r"[\s]+", "-", slug).lower()[:40]
        return f"§{slug}"
    return f"§chunk-{chunk_index}"


def chunk_document(doc: dict) -> List[dict]:
    """
    Split a single policy document into overlapping text chunks.

    Each returned chunk dict has:
        {
            "id":          "<sha256-based stable ID>",
            "text":        "<chunk text>",
            "source":      "<relative file path>",
            "chunk_index": <int>,
            "section_id":  "<§section-slug or §chunk-N>",
        }

    Strategy: RecursiveCharacterTextSplitter with markdown-aware separators.
    Header boundaries are tried first so chunks respect document structure.
    """
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " "],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        keep_separator=True,
    )

    raw_chunks = splitter.split_text(doc["content"])
    chunks = []
    for idx, text in enumerate(raw_chunks):
        text = text.strip()
        if not text:
            continue
        # Drop header-only fragments — they carry no policy content, produce
        # unhelpful §unknown citations, and degrade retrieval precision.
        if len(text) < MIN_CHUNK_CONTENT_LENGTH:
            continue
        section_id = _extract_section_id(text, idx)
        chunks.append({
            "id":          _chunk_id(doc["source"], idx),
            "text":        text,
            "source":      doc["source"],
            "chunk_index": idx,
            "section_id":  section_id,
        })
    return chunks


def _chunk_id(source: str, chunk_index: int) -> str:
    """Stable, deterministic 32-char hex ID for a chunk."""
    raw = f"{source}::{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


# ---------------------------------------------------------------------------
# Upsert into ChromaDB
# ---------------------------------------------------------------------------

def upsert_chunks(collection, chunks: List[dict]) -> int:
    """
    Upsert `chunks` into the ChromaDB collection.

    Returns the number of chunks upserted.
    Idempotent — re-running with the same source files is safe.
    """
    if not chunks:
        return 0

    collection.upsert(
        ids       =[c["id"]          for c in chunks],
        documents =[c["text"]        for c in chunks],
        metadatas =[{
            "source":      c["source"],
            "chunk_index": c["chunk_index"],
            "section_id":  c["section_id"],
        } for c in chunks],
    )
    return len(chunks)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def ingest_all() -> dict:
    """
    End-to-end ingestion: load → chunk → upsert.

    Returns a summary dict: {"files": N, "chunks": M}.
    """
    client     = get_chroma_client()
    collection = get_or_create_collection(client)
    docs       = load_policy_files()

    if not docs:
        print(f"[ingest] No .md files found in {POLICIES_DIR}. Nothing to do.")
        return {"files": 0, "chunks": 0}

    total_chunks = 0
    for doc in docs:
        chunks = chunk_document(doc)
        n = upsert_chunks(collection, chunks)
        total_chunks += n
        print(f"  [ingest] {doc['source']}  →  {n} chunks")

    print(f"\n[ingest] Done. {len(docs)} file(s), {total_chunks} chunk(s) upserted.")
    return {"files": len(docs), "chunks": total_chunks}


if __name__ == "__main__":
    ingest_all()
