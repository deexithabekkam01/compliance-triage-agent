"""
src/retrieve.py — Retrieval-augmented answer generation.

Public API
----------
retrieve(query, top_k=3) -> list[ChunkMatch]
    Semantic search over ChromaDB. Returns matched chunks with source,
    §section_id, text, and cosine similarity score (0–1, higher = better).

answer(query, top_k=3) -> AnswerResult
    1. Calls retrieve().
    2. If best score < CONFIDENCE_THRESHOLD  →  no-match result.
    3. Otherwise builds a grounded prompt and calls Ollama (llama3.2).
       The LLM must cite §section IDs for every claim. If the chunks don't
       contain the answer it must say so — it is NOT allowed to use
       general knowledge.
    4. Returns AnswerResult with answer text, §section citations, and score.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Optional

import requests

# Reuse ChromaDB helpers from ingest to guarantee the same client / collection
# / embedding function is used for both write and read paths.
from src.ingest import get_chroma_client, get_or_create_collection

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL       = os.getenv("OLLAMA_BASE_URL",   "http://localhost:11434")
OLLAMA_MODEL          = os.getenv("OLLAMA_MODEL",      "llama3.2")
OLLAMA_TIMEOUT        = int(os.getenv("OLLAMA_TIMEOUT", "120"))

# Cosine similarity threshold (0–1).  ChromaDB returns *distance* in [0, 2]
# for cosine space; we convert: similarity = 1 - (distance / 2).
# A threshold of 0.35 means we require at least 35 % similarity.
CONFIDENCE_THRESHOLD  = float(os.getenv("CONFIDENCE_THRESHOLD", "0.35"))

# System prompt — strict grounding, §citation enforcement, explicit no-hallucinate rule.
ANSWER_SYSTEM_PROMPT = """\
You are a compliance advisory assistant. Answer questions using ONLY the numbered
policy excerpts provided below. Do not use any external knowledge.

RULES:
1. Base every factual claim strictly on the provided excerpts.
2. Cite the §section ID in square brackets after each claim, e.g. [§data-retention].
3. If the excerpts contain a clear answer — even a conditional one ("yes, but only
   if X") — give that answer with citations. Do NOT refuse a answerable question.
4. ONLY respond with the single word CANNOT_ANSWER (nothing else) if the excerpts
   contain absolutely no information relevant to the question.
5. After your answer, add a "Citations:" line listing every §section ID used,
   comma-separated.
"""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ChunkMatch:
    """A single retrieved chunk with its similarity score."""
    chunk_id:   str
    text:       str
    source:     str          # e.g. "data/policies/aml_policy.md"
    section_id: str          # e.g. "§data-retention"
    score:      float        # cosine similarity 0–1 (higher = more relevant)


@dataclass
class AnswerResult:
    """Result returned by answer()."""
    query:      str
    answer:     Optional[str]        # None when no_match is True
    citations:  list[str] = field(default_factory=list)   # §section IDs
    confidence: float = 0.0          # best chunk similarity score
    no_match:   bool  = False
    reason:     Optional[str] = None # "not_in_corpus" | "cannot_answer" | None
    chunks:     list[ChunkMatch] = field(default_factory=list)  # raw matches


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def _distance_to_similarity(distance: float) -> float:
    """
    Convert ChromaDB cosine distance [0, 2] to similarity [0, 1].
    ChromaDB cosine distance = 1 − cosine_similarity, so distance 0 → sim 1.
    """
    return max(0.0, min(1.0, 1.0 - distance))


def retrieve(query: str, top_k: int = 3) -> list[ChunkMatch]:
    """
    Embed `query` with the same sentence-transformers model used at ingest time
    and return the `top_k` most similar chunks from ChromaDB.

    Chunks are sorted descending by similarity score.
    Only chunks above CONFIDENCE_THRESHOLD are returned; if none pass,
    the list is empty (caller should treat this as no-match).

    Args:
        query:  The user's compliance question.
        top_k:  Maximum number of chunks to return (default 3).

    Returns:
        List of ChunkMatch objects, best match first.
    """
    client     = get_chroma_client()
    collection = get_or_create_collection(client)

    # Guard: empty collection returns an error from ChromaDB
    count = collection.count()
    if count == 0:
        return []

    results = collection.query(
        query_texts=[query],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )

    matches: list[ChunkMatch] = []
    ids        = results["ids"][0]
    documents  = results["documents"][0]
    metadatas  = results["metadatas"][0]
    distances  = results["distances"][0]

    for chunk_id, text, meta, distance in zip(ids, documents, metadatas, distances):
        score = _distance_to_similarity(distance)
        matches.append(ChunkMatch(
            chunk_id   = chunk_id,
            text       = text,
            source     = meta.get("source", "unknown"),
            section_id = meta.get("section_id", "§unknown"),
            score      = round(score, 4),
        ))

    # Sort best-first (highest similarity)
    matches.sort(key=lambda c: c.score, reverse=True)
    return matches


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------

def _build_context_block(chunks: list[ChunkMatch]) -> str:
    """
    Format retrieved chunks into a numbered context block for the LLM prompt.

    Example output:
        [1] §data-retention  |  data/policies/gdpr_policy.md
        Personal data must not be retained beyond the period necessary…

        [2] §third-party-sharing  |  data/policies/gdpr_policy.md
        Before sharing data with a third party, a data processing agreement…
    """
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        lines.append(
            f"[{i}] {chunk.section_id}  |  {chunk.source}\n{chunk.text}"
        )
    return "\n\n".join(lines)


def _parse_citations(answer_text: str) -> list[str]:
    """
    Extract §section IDs from the 'Citations:' line at the end of the answer.

    Handles formats:
        Citations: §data-retention, §third-party-sharing
        Citations: [§data-retention, §third-party-sharing]

    Also extracts inline §citations from the answer body as a fallback.
    Returns a deduplicated list preserving first-seen order.
    """
    seen: dict[str, None] = {}

    # 1. Explicit Citations: line
    citations_match = re.search(
        r"Citations\s*:\s*(.+)$", answer_text, re.IGNORECASE | re.MULTILINE
    )
    if citations_match:
        raw = citations_match.group(1)
        for item in re.findall(r"§[\w-]+", raw):
            seen[item] = None

    # 2. Fallback: inline §references anywhere in the answer body
    for item in re.findall(r"§[\w-]+", answer_text):
        seen[item] = None

    return list(seen.keys())


def _call_ollama(prompt: str) -> str:
    """POST to Ollama /api/generate and return the response text."""
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }
    resp = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT)
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def _generate_answer(
    query: str,
    chunks: list[ChunkMatch],
) -> tuple[str | None, list[str], str | None]:
    """
    Call the LLM with the grounded prompt.

    Returns:
        (answer_text, citations, reason)
        answer_text is None and reason is "cannot_answer" if the LLM says
        it cannot answer from the provided excerpts.
    """
    context_block = _build_context_block(chunks)
    prompt = (
        f"{ANSWER_SYSTEM_PROMPT}\n\n"
        f"--- POLICY EXCERPTS ---\n{context_block}\n\n"
        f"--- QUESTION ---\n{query}\n\n"
        f"Answer (cite §section IDs inline and list them in Citations:):"
    )

    try:
        raw = _call_ollama(prompt)
    except requests.ConnectionError:
        return None, [], "ollama_unavailable"
    except (requests.Timeout, requests.HTTPError) as exc:
        return None, [], f"ollama_error:{exc}"

    # Treat CANNOT_ANSWER as no-match only when the model genuinely cannot
    # answer — i.e. CANNOT_ANSWER appears at the very start of the response
    # (possibly after whitespace).  If the model produces a substantive answer
    # and then hedges with CANNOT_ANSWER somewhere in the body, that is a
    # partial-answer artefact and should NOT suppress the real answer text.
    if raw.strip().upper().startswith("CANNOT_ANSWER"):
        return None, [], "cannot_answer"

    citations = _parse_citations(raw)

    # Strip the Citations: line from the answer body for cleaner display,
    # then scrub any residual CANNOT_ANSWER token that the model may have
    # appended as a hedge after a substantive answer.  This prevents the
    # token from appearing in the user-facing answer or the audit log.
    answer_body = re.sub(
        r"\n*Citations\s*:.*$", "", raw, flags=re.IGNORECASE | re.DOTALL
    ).strip()
    answer_body = re.sub(
        r"\s*CANNOT_ANSWER\s*", " ", answer_body, flags=re.IGNORECASE
    ).strip()

    return answer_body, citations, None


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def answer(query: str, top_k: int = 5) -> AnswerResult:
    """
    Full retrieval-augmented answer pipeline.

    Steps:
        1. retrieve() → ranked ChunkMatch list
        2. If empty or best score < CONFIDENCE_THRESHOLD → no-match result
        3. _generate_answer() → grounded LLM answer with §citations
        4. Return AnswerResult

    Args:
        query:  The compliance question.
        top_k:  Number of chunks to retrieve (default 5; gives the LLM
                enough context when the direct answer sits in a lower-ranked
                chunk behind header/purpose chunks).

    Returns:
        AnswerResult — inspect .answer, .citations, .confidence, .no_match.
    """
    chunks = retrieve(query, top_k=top_k)

    # No chunks at all (empty collection or all below threshold)
    if not chunks or chunks[0].score < CONFIDENCE_THRESHOLD:
        best_score = chunks[0].score if chunks else 0.0
        return AnswerResult(
            query      = query,
            answer     = None,
            confidence = best_score,
            no_match   = True,
            reason     = "not_in_corpus",
            chunks     = chunks,
        )

    answer_text, citations, reason = _generate_answer(query, chunks)
    no_match = answer_text is None

    return AnswerResult(
        query      = query,
        answer     = answer_text,
        citations  = citations,
        confidence = chunks[0].score,
        no_match   = no_match,
        reason     = reason,
        chunks     = chunks,
    )
