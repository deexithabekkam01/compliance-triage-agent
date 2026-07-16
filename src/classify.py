"""
src/classify.py — Dual classifier: topic + stakes.

Responsibilities:
- Topic classification: assign one of {DPO, AML, Legal, Other}
- Stakes classification: assign one of {low, medium, high}

Both classifications are performed in a single Ollama (llama3.2) call using
the JSON-constrained output mode so the response is always parseable.

Topic definitions:
    DPO   — Data protection, privacy, GDPR/CCPA, PII handling, customer data,
             data retention periods, third-party data sharing, data subject rights,
             consent management, data breach notification
    AML   — Anti-money laundering, sanctions screening, transaction monitoring,
             SAR/CTR filing, KYC/CDD, financial crime, correspondent banking risk
    Legal — All other regulatory compliance: employment law, contract obligations,
             licensing, competition/antitrust, consumer protection, regulatory reporting
    Other — Does not fit DPO, AML, or Legal

Stakes definitions:
    low    — Informational query, no regulatory breach risk, no personal data involved
    medium — Moderate risk: could affect individuals or trigger a minor regulatory breach
    high   — Severe risk: potential regulatory breach, legal liability, involves
              sensitive personal data, large financial exposure, or criminal penalties
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Literal

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_TIMEOUT  = int(os.getenv("OLLAMA_TIMEOUT", "60"))   # seconds

TopicLabel  = Literal["DPO", "AML", "Legal", "Other"]
StakesLabel = Literal["low", "medium", "high"]

VALID_TOPICS: frozenset[str] = frozenset({"DPO", "AML", "Legal", "Other"})
VALID_STAKES: frozenset[str] = frozenset({"low", "medium", "high"})
FALLBACK: dict = {"topic": "Other", "stakes": "high"}

# The full classification instruction is baked into the prompt (not a system
# field) because Ollama's /api/generate endpoint uses a single "prompt" string.
CLASSIFICATION_PROMPT_TEMPLATE = """\
You are a compliance classification engine.

Given the query (and optional answer excerpt) below, respond with a JSON object
containing exactly two keys:
  "topic":  one of ["DPO", "AML", "Legal", "Other"]
  "stakes": one of ["low", "medium", "high"]

--- TOPIC DEFINITIONS ---
DPO   — Data protection, privacy, and personal information. Use DPO for:
          • GDPR / CCPA / data protection law questions
          • Personally identifiable information (PII) handling or storage
          • Customer data sharing with third parties or vendors
          • Data retention periods and deletion obligations
          • Data subject rights (access, erasure, portability, rectification)
          • Consent management and lawful basis for processing
          • Data breach identification, containment, and notification
          • Privacy impact assessments (DPIA / PIA)

AML   — Anti-money laundering, financial crime, and sanctions. Use AML for:
          • Suspicious activity reports (SAR) and currency transaction reports (CTR)
          • Transaction monitoring, screening, and red-flag analysis
          • Sanctions lists (OFAC, EU, UN) and politically exposed persons (PEP)
          • Know Your Customer (KYC) and Customer Due Diligence (CDD / EDD)
          • Money laundering, terrorist financing, or fraud typologies
          • Correspondent banking risk and de-risking
          • Beneficial ownership and ultimate beneficial owner (UBO) rules

Legal — All other regulatory and legal compliance not covered by DPO or AML:
          • Employment law, workplace obligations, disciplinary procedures,
            redundancy, notice periods, working time rules, discrimination
          • Contract interpretation, obligations, and breach
          • Licensing, regulatory registration, and authorisation requirements
          • Competition / antitrust rules and market abuse
          • Consumer protection and fair dealing obligations
          • General corporate governance and regulatory reporting
          • Health & safety obligations

Other — Use ONLY if the query is not a compliance or regulatory question at all
        (e.g. "how do I reset my VPN password", purely operational logistics).

--- STAKES DEFINITIONS ---
low    — Informational; no regulatory breach risk; no sensitive personal data
medium — Moderate risk; could affect individuals or trigger a minor regulatory breach
high   — Severe risk; potential regulatory breach, legal liability, criminal
         penalties, sensitive personal data, or large financial exposure

--- FEW-SHOT EXAMPLES ---
Example 1
Query: What is the maximum retention period for customer KYC records?
Answer excerpt: (none)
Output: {{"topic": "AML", "stakes": "low"}}

Example 2
Query: Can we share customer PII with our third-party marketing vendor without
       explicit consent?
Answer excerpt: (none)
Output: {{"topic": "DPO", "stakes": "high"}}

Example 3
Query: Do we need to file a SAR for this transaction where a customer deposited
       $12,000 in cash across two branches on the same day?
Answer excerpt: Structuring transactions to avoid CTR thresholds is a red flag.
Output: {{"topic": "AML", "stakes": "high"}}

Example 4
Query: What notice period are we required to give employees before redundancy?
Answer excerpt: (none)
Output: {{"topic": "Legal", "stakes": "medium"}}

Example 5
Query: Can we dismiss an employee who has been on sick leave for six months?
Answer excerpt: (none)
Output: {{"topic": "Legal", "stakes": "high"}}

Example 6
Query: Do we need a DPIA before launching this new customer analytics feature?
Answer excerpt: (none)
Output: {{"topic": "DPO", "stakes": "low"}}

Example 7
Query: What is the retention period for customer records?
Answer excerpt: Customer KYC records must be retained for five (5) years.
Output: {{"topic": "DPO", "stakes": "low"}}

Example 8
Query: Can we just ignore GDPR requirements for this one strategic client?
Answer excerpt: GDPR obligations may not be waived for an individual client.
Output: {{"topic": "DPO", "stakes": "high"}}

Example 9
Query: Our CEO says we should skip AML screening for this transaction — is that okay?
Answer excerpt: (none)
Output: {{"topic": "AML", "stakes": "high"}}

--- BEFORE YOU ANSWER, CHECK ---
STEP 1 — BYPASS/WAIVER OVERRIDE (check this first, before anything else):
  If the query asks to bypass, waive, ignore, suspend, or make an exception to
  ANY compliance, legal, or regulatory obligation — even if the correct answer
  is "no" or "that is not permitted" — the stakes MUST be "high".
  The stakes rating reflects the risk of the request itself, not the risk of
  the (refusing) answer. A request to circumvent compliance is always high-stakes
  regardless of how the system responds to it.
  Phrases that trigger this rule: "ignore", "waive", "bypass", "skip",
  "exception for", "just this once", "for this client", "override", "suspend".

STEP 2 — INFORMATIONAL LOOKUP (check before assigning medium or high):
  If the query is ONLY asking what a policy says, what a period is, what a
  deadline is, or what a procedure requires — with no indication of intent to
  violate, bypass, or bend the rule — the stakes MUST be "low".
  Phrases that signal an informational lookup: "what is", "what are",
  "how long", "what's the", "retention period", "required to", "do we need to".

STEP 3 — TOPIC:
- Mentions PII, personal data, GDPR, CCPA, DPIA, data subject rights, retention,
  consent, data breach, or third-party data sharing? -> topic = "DPO"
- Mentions sanctions, SAR, CTR, AML, KYC, transaction screening, money laundering,
  structuring, or terrorist financing? -> topic = "AML"
- Mentions employment, redundancy, dismissal, notice period, discipline,
  discrimination, licensing, contracts, regulatory reporting, or health & safety?
  -> topic = "Legal"
- Use topic = "Other" ONLY for non-compliance questions (e.g. IT passwords,
  canteen menus). If the query is about any law, regulation, or compliance
  obligation, it MUST be DPO, AML, or Legal — never Other.

Example of Other (non-compliance):
Query: How do I reset my VPN password?
Output: {{"topic": "Other", "stakes": "low"}}

--- INPUT ---
Query: {query}
Answer excerpt (may be empty): {answer_excerpt}

Respond with ONLY the JSON object. No explanation, no markdown, no extra keys.
"""


# ---------------------------------------------------------------------------
# Keyword pre-filter
# Handles unambiguous queries before sending to the LLM.
# Keys are (topic, stakes); values are lists of keyword sets — a query matches
# if it contains ALL keywords in any one set (case-insensitive).
# ---------------------------------------------------------------------------

_DPO_KEYWORDS: list[set[str]] = [
    {"gdpr"}, {"ccpa"}, {"pii"}, {"personal data"}, {"personal information"},
    {"data subject"}, {"data retention"}, {"retention period"}, {"data breach"},
    {"dpia"}, {"privacy impact"}, {"right to erasure"}, {"right of access"},
    {"data portability"}, {"lawful basis"}, {"consent"}, {"data protection"},
    {"third-party", "data"}, {"third party", "data"}, {"vendor", "data"},
    {"customer data"}, {"customer pii"}, {"customer records"},
]

_AML_KEYWORDS: list[set[str]] = [
    {"sar"}, {"ctr"}, {"aml"}, {"kyc"}, {"cdd"}, {"edd"},
    {"money laundering"}, {"sanctions"}, {"ofac"}, {"structuring"},
    {"transaction monitoring"}, {"transaction screening"}, {"suspicious activity"},
    {"terrorist financing"}, {"politically exposed"}, {"pep"},
    {"beneficial owner"}, {"ubo"}, {"correspondent banking"},
]

_LEGAL_KEYWORDS: list[set[str]] = [
    {"redundancy"}, {"notice period"}, {"dismissal"}, {"disciplinary"},
    {"discrimination"}, {"employment"}, {"labour law"}, {"labor law"},
    {"unfair dismissal"}, {"sick leave", "dismiss"}, {"sick leave", "terminate"},
    {"contract"}, {"licensing"}, {"competition law"}, {"antitrust"},
    {"health and safety"}, {"health & safety"}, {"regulatory reporting"},
]


def _keyword_prefilter(text: str) -> str | None:
    """
    Return a topic string if `text` unambiguously matches a keyword set,
    or None if the LLM should decide.

    Checks DPO → AML → Legal in priority order.
    """
    lowered = text.lower()

    for kw_set in _DPO_KEYWORDS:
        if all(kw in lowered for kw in kw_set):
            return "DPO"

    for kw_set in _AML_KEYWORDS:
        if all(kw in lowered for kw in kw_set):
            return "AML"

    for kw_set in _LEGAL_KEYWORDS:
        if all(kw in lowered for kw in kw_set):
            return "Legal"

    return None

@dataclass
class ClassificationResult:
    """Output of a classify() call."""
    topic: TopicLabel
    stakes: StakesLabel
    raw_response: str = field(default="", repr=False)  # raw model output


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_classification_prompt(query: str, answer_excerpt: str) -> str:
    """
    Render the classification prompt template with the given inputs.

    Args:
        query:          The original user compliance question.
        answer_excerpt: First ~500 characters of the generated answer, or "".

    Returns:
        A fully-rendered prompt string ready to send to Ollama.
    """
    return CLASSIFICATION_PROMPT_TEMPLATE.format(
        query=query.strip(),
        answer_excerpt=answer_excerpt.strip() or "(none)",
    )


# ---------------------------------------------------------------------------
# Ollama call
# ---------------------------------------------------------------------------

def call_ollama(prompt: str) -> str:
    """
    Send a prompt to the Ollama /api/generate endpoint and return the
    raw response text.

    Uses ``format="json"`` to instruct Ollama to constrain its output to
    valid JSON — this significantly reduces parse failures with llama3.1.

    Raises:
        requests.HTTPError:   If the Ollama server returns a non-2xx status.
        requests.Timeout:     If the request exceeds OLLAMA_TIMEOUT seconds.
        requests.ConnectionError: If Ollama is not running.
    """
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "format": "json",   # constrain output to JSON
        "stream": False,    # return full response at once
    }
    response = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT)
    response.raise_for_status()
    data = response.json()
    return data.get("response", "")


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_classification_response(raw: str) -> dict:
    """
    Parse the model's JSON output into a validated ``{"topic": ..., "stakes": ...}`` dict.

    Validation rules:
    - ``topic``  must be one of VALID_TOPICS
    - ``stakes`` must be one of VALID_STAKES

    Falls back to ``{"topic": "Other", "stakes": "high"}`` on any parse or
    validation failure — fail-safe because unknown output = potentially high risk.

    Args:
        raw: Raw string returned by the model (should be valid JSON).

    Returns:
        Validated dict with keys "topic" and "stakes".
    """
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return dict(FALLBACK)

    topic  = parsed.get("topic",  "")
    stakes = parsed.get("stakes", "")

    # Normalise case variations.
    # stakes: always lowercase  ("High" → "high")
    # topic:  match case-insensitively against the canonical set and return
    #         the canonical form.  Using .upper() broke "Legal" → "LEGAL"
    #         which is absent from VALID_TOPICS, causing every Legal response
    #         to silently fall back to {"topic": "Other", "stakes": "high"}.
    stakes_normalised = stakes.lower() if isinstance(stakes, str) else ""

    topic_normalised = ""
    if isinstance(topic, str):
        for canonical in VALID_TOPICS:
            if topic.strip().lower() == canonical.lower():
                topic_normalised = canonical
                break

    if topic_normalised not in VALID_TOPICS or stakes_normalised not in VALID_STAKES:
        return dict(FALLBACK)

    return {"topic": topic_normalised, "stakes": stakes_normalised}


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def classify(query: str, answer_excerpt: str = "") -> ClassificationResult:
    """
    Classify a compliance query by topic and stakes using llama3.2 via Ollama.

    Pipeline:
        1. Run a Python keyword pre-filter on (query + answer_excerpt).
           If it confidently identifies the topic, use that result directly
           and still call the LLM for stakes.
        2. If the pre-filter is inconclusive, send the full prompt to the LLM
           and parse topic + stakes from its response.

    The pre-filter handles short, unambiguous queries (e.g. "What does GDPR
    stand for?") that a 3B model may misclassify. The LLM handles nuanced or
    cross-domain queries where keyword matching would be unreliable.

    Args:
        query:          The original user query.
        answer_excerpt: First ~500 chars of the generated answer (optional).
                        Providing the answer improves stakes accuracy.

    Returns:
        ClassificationResult with .topic and .stakes populated.

    Example:
        >>> result = classify("What is our data retention policy?")
        >>> result.topic   # "DPO"
        >>> result.stakes  # "low"
    """
    combined = f"{query} {answer_excerpt}"
    prefilter_topic = _keyword_prefilter(combined)

    prompt = build_classification_prompt(query, answer_excerpt)

    try:
        raw = call_ollama(prompt)
    except requests.ConnectionError:
        return ClassificationResult(
            topic=prefilter_topic or "Other",
            stakes="high",
            raw_response="ERROR: Ollama not reachable at " + OLLAMA_BASE_URL,
        )
    except requests.Timeout:
        return ClassificationResult(
            topic=prefilter_topic or "Other",
            stakes="high",
            raw_response="ERROR: Ollama request timed out",
        )
    except requests.HTTPError as exc:
        return ClassificationResult(
            topic=prefilter_topic or "Other",
            stakes="high",
            raw_response=f"ERROR: Ollama HTTP {exc.response.status_code}",
        )

    validated = parse_classification_response(raw)

    # If the pre-filter fired, trust it for topic but take stakes from the LLM.
    # This corrects the 3B model's tendency to collapse short queries to Other.
    if prefilter_topic is not None:
        validated["topic"] = prefilter_topic

    return ClassificationResult(
        topic=validated["topic"],     # type: ignore[arg-type]
        stakes=validated["stakes"],   # type: ignore[arg-type]
        raw_response=raw,
    )
