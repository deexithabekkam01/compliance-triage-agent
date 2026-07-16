"""
app.py — Streamlit UI for the Compliance Advisory & Triage Agent.

Tabs:
    1. 💬 Chat           — submit queries, view answers with citations and routing badge
    2. 🕐 Pending Review — table of escalated queries awaiting human review
    3. 📋 Audit Log      — full append-only audit trail with filters

Run:
    streamlit run app.py
"""

from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.graph import run_query
from src.audit import read_all_records, read_pending_review

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

load_dotenv()  # load ANTHROPIC_API_KEY and CHROMA_PERSIST_DIR from .env

st.set_page_config(
    page_title="Compliance Advisory & Triage Agent",
    page_icon="⚖️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Routing badge config
# ---------------------------------------------------------------------------

_BADGE: dict[str, tuple[str, str]] = {
    # routing value → (label, hex colour)
    "answered":              ("✅ Answered",              "#1a7f3c"),
    "escalated_with_answer": ("🟠 Escalated with answer", "#b45309"),
    "escalated":             ("🔴 Escalated",             "#b91c1c"),
    "refused":               ("⬜ Refused",               "#6b7280"),
}

def _badge_html(routing: str) -> str:
    label, colour = _BADGE.get(routing, (routing, "#6b7280"))
    return (
        f'<span style="background:{colour};color:#fff;padding:2px 10px;'
        f'border-radius:12px;font-size:0.8rem;font-weight:600;">{label}</span>'
    )


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def init_session_state() -> None:
    """Initialise Streamlit session state keys on first load."""
    if "chat_history" not in st.session_state:
        # Each entry: {"role": "user"|"assistant", "content": str, "meta": dict|None}
        st.session_state.chat_history = []


# ---------------------------------------------------------------------------
# Tab 1: Chat
# ---------------------------------------------------------------------------

def render_chat_tab() -> None:
    """
    Render the conversational chat interface.

    - Scrollable message history (user / assistant bubbles)
    - st.chat_input pinned to the bottom
    - On submit: calls run_query(), displays answer + citations + routing badge
    - Routing badge colours:
        answered              → green
        escalated_with_answer → orange
        escalated             → red
        refused               → grey
    """
    st.header("💬 Compliance Chat")
    st.caption(
        "Ask a compliance question. Answers are grounded exclusively in "
        "ingested policy documents — no general knowledge."
    )

    # ── Render existing history ──────────────────────────────────────────────
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)
            if msg["role"] == "assistant" and msg.get("meta"):
                _render_answer_meta(msg["meta"])

    # ── Input ────────────────────────────────────────────────────────────────
    query = st.chat_input("Ask a compliance question…")

    if query:
        # Display the user message immediately
        with st.chat_message("user"):
            st.markdown(query)
        st.session_state.chat_history.append(
            {"role": "user", "content": query, "meta": None}
        )

        # Run the pipeline
        with st.chat_message("assistant"):
            with st.spinner("Searching policies…"):
                try:
                    state = run_query(query)
                except Exception as exc:
                    st.error(f"Pipeline error: {exc}")
                    return

            final_answer = state["final_answer"]
            routing      = state["routing"]
            citations    = state["retrieval"].citations
            topic        = state["classification"].topic
            stakes       = state["classification"].stakes
            audit_id     = state.get("audit_id", "")

            st.markdown(final_answer)
            meta = {
                "routing":   routing,
                "citations": citations,
                "topic":     topic,
                "stakes":    stakes,
                "audit_id":  audit_id,
            }
            _render_answer_meta(meta)

        st.session_state.chat_history.append(
            {"role": "assistant", "content": final_answer, "meta": meta}
        )


def _render_answer_meta(meta: dict) -> None:
    """Render the routing badge, citations, and audit ID below an answer."""
    routing   = meta.get("routing", "")
    citations = meta.get("citations", [])
    topic     = meta.get("topic", "")
    stakes    = meta.get("stakes", "")
    audit_id  = meta.get("audit_id", "")

    # Routing badge + topic/stakes chips
    badge = _badge_html(routing)
    topic_chip  = (
        f'<span style="background:#1e40af;color:#fff;padding:2px 8px;'
        f'border-radius:12px;font-size:0.75rem;margin-left:6px;">{topic}</span>'
    )
    stakes_chip = (
        f'<span style="background:#374151;color:#fff;padding:2px 8px;'
        f'border-radius:12px;font-size:0.75rem;margin-left:4px;">stakes: {stakes}</span>'
    )
    st.markdown(
        badge + topic_chip + stakes_chip,
        unsafe_allow_html=True,
    )

    # Citations — only shown when the answer is actually returned to the user.
    # For routing=escalated and routing=refused, withhold citations entirely:
    # section titles alone (e.g. "Storing EU Customer Data in the US Region")
    # can reveal policy conclusions before human review has taken place.
    if routing in ("answered", "escalated_with_answer"):
        if citations:
            st.markdown(
                "**Citations:** " + " · ".join(f"`{c}`" for c in citations)
            )
        else:
            st.markdown("**Citations:** —")

    # Audit ID (small, muted)
    if audit_id:
        st.caption(f"Audit record: `{audit_id}`")


# ---------------------------------------------------------------------------
# Tab 2: Pending Human Review
# ---------------------------------------------------------------------------

_PENDING_COLUMNS = ["timestamp", "topic", "stakes", "routing", "query", "id"]

def render_pending_review_tab() -> None:
    """
    Render a table of queries currently pending human review
    (routing = escalated or escalated_with_answer).
    """
    st.header("🕐 Pending Human Review")
    st.caption(
        "Queries with **stakes=high** are escalated without an answer. "
        "Queries with **stakes=medium** are escalated with a draft answer pending review."
    )

    records = read_pending_review()

    if not records:
        st.info("No queries currently pending review.")
        return

    df = pd.DataFrame(records)

    # Truncate query for the table view
    df["query"] = df["query"].str[:120] + "…"

    # Reorder / select columns that exist
    cols = [c for c in _PENDING_COLUMNS if c in df.columns]
    st.dataframe(
        df[cols],
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.subheader("Record detail")

    for record in records:
        ts    = record.get("timestamp", "")[:19].replace("T", " ")
        label = f"{ts} · {record.get('topic')} · {record.get('routing')} · {record.get('query', '')[:80]}"
        with st.expander(label):
            col_l, col_r = st.columns([2, 1])
            with col_l:
                st.markdown(f"**Query:** {record.get('query', '')}")
                answer = record.get("answer", "")
                if answer:
                    st.markdown(f"**Draft answer:**\n\n{answer}")
                else:
                    st.markdown("*Answer withheld — high-stakes escalation.*")
            with col_r:
                st.markdown(f"**Topic:** `{record.get('topic')}`")
                st.markdown(f"**Stakes:** `{record.get('stakes')}`")
                routing = record.get("routing", "")
                st.markdown(
                    f"**Routing:** " + _badge_html(routing),
                    unsafe_allow_html=True,
                )
                cites = record.get("citations", [])
                if cites:
                    st.markdown("**Citations:** " + ", ".join(f"`{c}`" for c in cites))
                reason = record.get("escalation_reason")
                if reason:
                    st.markdown(f"**Escalation reason:** {reason}")
                st.caption(f"Audit ID: `{record.get('id', '')}`")


# ---------------------------------------------------------------------------
# Tab 3: Audit Log
# ---------------------------------------------------------------------------

_AUDIT_COLUMNS = ["timestamp", "id", "topic", "stakes", "routing", "query"]

def render_audit_log_tab() -> None:
    """
    Render the full audit log with filter controls.

    Controls:
    - Topic filter (multiselect)
    - Stakes filter (multiselect)
    - Routing filter (multiselect)
    - Free-text search over the query field
    """
    st.header("📋 Audit Log")

    records = read_all_records()

    if not records:
        st.info("No audit records yet. Submit a query in the Chat tab to get started.")
        return

    df = pd.DataFrame(records)
    total = len(df)

    # ── Filter controls ──────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns([2, 2, 2, 3])

    with col1:
        sel_topic = st.multiselect(
            "Topic", sorted(df["topic"].unique().tolist()), key="filter_topic"
        )
    with col2:
        sel_stakes = st.multiselect(
            "Stakes", ["low", "medium", "high"], key="filter_stakes"
        )
    with col3:
        sel_routing = st.multiselect(
            "Routing",
            ["answered", "refused", "escalated", "escalated_with_answer"],
            key="filter_routing",
        )
    with col4:
        search = st.text_input("Search queries", placeholder="e.g. GDPR, SAR, retention…")

    # ── Apply filters ────────────────────────────────────────────────────────
    filtered = df.copy()
    if sel_topic:
        filtered = filtered[filtered["topic"].isin(sel_topic)]
    if sel_stakes:
        filtered = filtered[filtered["stakes"].isin(sel_stakes)]
    if sel_routing:
        filtered = filtered[filtered["routing"].isin(sel_routing)]
    if search:
        filtered = filtered[
            filtered["query"].str.contains(search, case=False, na=False)
        ]

    st.caption(f"Showing {len(filtered)} of {total} records")

    # ── Summary metrics ──────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total records",  total)
    m2.metric("Answered",       int((df["routing"] == "answered").sum()))
    m3.metric("Escalated",      int(df["routing"].str.startswith("escalated").sum()))
    m4.metric("Refused",        int((df["routing"] == "refused").sum()))

    st.divider()

    # ── Table ────────────────────────────────────────────────────────────────
    display = filtered.copy()
    display["query"] = display["query"].str[:100] + "…"
    cols = [c for c in _AUDIT_COLUMNS if c in display.columns]
    st.dataframe(display[cols], use_container_width=True, hide_index=True)

    # ── Expandable record detail ─────────────────────────────────────────────
    st.divider()
    st.subheader("Record detail")

    # Show most recent first
    for _, row in filtered.sort_values("timestamp", ascending=False).head(50).iterrows():
        ts    = str(row.get("timestamp", ""))[:19].replace("T", " ")
        label = f"{ts} · {row.get('topic')} · {row.get('routing')} · {str(row.get('query', ''))[:80]}"
        with st.expander(label):
            col_l, col_r = st.columns([2, 1])
            with col_l:
                st.markdown(f"**Query:** {row.get('query', '')}")
                answer = row.get("answer", "")
                if answer:
                    st.markdown(f"**Answer:**\n\n{answer}")
                else:
                    st.markdown("*No answer — refused or high-stakes escalation.*")
            with col_r:
                routing = row.get("routing", "")
                st.markdown(
                    "**Routing:** " + _badge_html(routing),
                    unsafe_allow_html=True,
                )
                st.markdown(f"**Topic:** `{row.get('topic')}`")
                st.markdown(f"**Stakes:** `{row.get('stakes')}`")
                cites = row.get("citations", [])
                if cites:
                    st.markdown("**Citations:** " + ", ".join(f"`{c}`" for c in cites))
                reason = row.get("escalation_reason")
                if reason:
                    st.markdown(f"**Escalation reason:** {reason}")
                st.caption(f"Audit ID: `{row.get('id', '')}`")
                st.caption(f"Timestamp: `{row.get('timestamp', '')}`")


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point — renders tab layout."""
    init_session_state()

    st.title("⚖️ Compliance Advisory & Triage Agent")
    st.caption(
        "Grounded in your internal policy documents · "
        "Powered by llama3.2 · "
        "Audit-logged · Human-in-the-loop escalation"
    )

    tab_chat, tab_review, tab_audit = st.tabs(
        ["💬 Chat", "🕐 Pending Review", "📋 Audit Log"]
    )

    with tab_chat:
        render_chat_tab()

    with tab_review:
        render_pending_review_tab()

    with tab_audit:
        render_audit_log_tab()


if __name__ == "__main__":
    main()
