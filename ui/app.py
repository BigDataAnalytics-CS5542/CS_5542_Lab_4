import os
import streamlit as st
import streamlit.components.v1 as components
import requests
import re
import html as html_module

DEFAULT_API_URL = os.environ.get("API_URL", "http://127.0.0.1:3001")

# ── Page config (must be first Streamlit call) ──────────────────────────
st.set_page_config(
    page_title="CS 5542 RAG Chatbot",
    layout="wide",
)

# ── Custom CSS ──────────────────────────────────────────────────────────
CSS = """
<style>
    .answer-card {
        background: #1e1e2e;
        border: 1px solid #383850;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        line-height: 1.75;
        font-size: 1.05rem;
        color: #e0e0e0;
    }
    button[kind="primary"] {
        background-color: #16a34a !important;
        border-color: #16a34a !important;
    }
    button[kind="primary"]:hover {
        background-color: #15803d !important;
        border-color: #15803d !important;
    }
    input[type="text"] {
        outline: none !important;
    }
    input[type="text"]:focus {
        border-color: #16a34a !important;
        box-shadow: none !important;
        outline: none !important;
    }
    div[data-baseweb="base-input"] {
        border-color: #dee2e6 !important;
        box-shadow: none !important;
    }
    div[data-baseweb="base-input"]:focus-within {
        border-color: #16a34a !important;
        box-shadow: 0 0 0 1px #16a34a !important;
    }
    .warning-banner {
        background: #fef3c7;
        border: 1px solid #f59e0b;
        border-radius: 6px;
        padding: 0.7rem 1rem;
        color: #92400e;
        font-weight: 500;
        margin-bottom: 1rem;
    }
    /* Hide the built-in "Press Enter to Apply" tooltip */
    div[data-testid="InputInstructions"] {
        display: none !important;
    }
    /* Remove the form border */
    div[data-testid="stForm"] {
        border: none !important;
        padding: 0 !important;
    }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ── Helper functions ────────────────────────────────────────────────────

def call_query_api(api_url: str, query: str, top_k: int, retrieval_mode: str, alpha: float, user_id: str) -> dict:
    """POST to /query with query parameters (not JSON body)."""
    r = requests.post(
        f"{api_url}/query",
        params={"query": query, "top_k": top_k, "retrieval_mode": retrieval_mode, "alpha": alpha, "userID": user_id},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def fetch_history(api_url: str, user_id: str) -> list:
    """GET /history for a given user."""
    r = requests.get(
        f"{api_url}/history",
        params={"userID": user_id},
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


def build_answer_inner_html(answer_text: str) -> str:
    """Parse [N] citation markers and wrap text regions + badges in HTML.

    Text immediately preceding a [N] marker is treated as the cited region
    for source N. Returns raw HTML (no outer wrapper).
    """
    marker_pattern = r"(\[\d+\])"
    parts = re.split(marker_pattern, answer_text)

    # No markers found — return escaped plain text
    if len(parts) == 1:
        return html_module.escape(answer_text)

    html_parts: list[str] = []
    i = 0
    while i < len(parts):
        segment = parts[i]
        # Check if the next part is a citation marker
        if i + 1 < len(parts):
            marker_match = re.match(r"\[(\d+)\]", parts[i + 1])
            if marker_match:
                n = marker_match.group(1)
                escaped = html_module.escape(segment)
                html_parts.append(
                    f'<span class="cite-region cite-region-{n}">{escaped}</span>'
                    f'<span class="cite-badge" data-cite="{n}">[{n}]</span>'
                )
                i += 2
                continue
        # Plain text with no following marker
        html_parts.append(html_module.escape(segment))
        i += 1

    return "".join(html_parts)


def render_answer_component(answer_text: str) -> None:
    """Render the answer as a self-contained HTML component with hover JS."""
    inner_html = build_answer_inner_html(answer_text)
    has_citations = bool(re.search(r"\[\d+\]", answer_text))

    height = max(120, min(500, 60 + len(answer_text) // 4))

    component_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Source Sans Pro', -apple-system, BlinkMacSystemFont, sans-serif;
            color: #e0e0e0;
            padding: 0;
        }}
        .answer-card {{
            background: #1e1e2e;
            border: 1px solid #383850;
            border-radius: 8px;
            padding: 1.2rem 1.5rem;
            line-height: 1.75;
            font-size: 1.05rem;
        }}
        .cite-badge {{
            display: inline-block;
            background: #2563eb;
            color: #ffffff;
            font-size: 0.72rem;
            font-weight: 600;
            padding: 1px 6px;
            border-radius: 4px;
            margin: 0 2px;
            cursor: pointer;
            vertical-align: super;
            line-height: 1;
            transition: background 0.15s;
        }}
        .cite-badge:hover {{
            background: #1d4ed8;
        }}
        .cite-region {{
            transition: background 0.2s ease;
            border-radius: 2px;
            padding: 1px 0;
        }}
        .cite-region.highlight {{
            background: #854d0e;
        }}
    </style>
    </head>
    <body>
        <div class="answer-card">{inner_html}</div>
        {"" if not has_citations else '''
        <script>
            document.querySelectorAll('.cite-badge').forEach(function(badge) {
                var n = badge.getAttribute('data-cite');
                badge.addEventListener('mouseenter', function() {
                    document.querySelectorAll('.cite-region-' + n).forEach(function(el) {
                        el.classList.add('highlight');
                    });
                });
                badge.addEventListener('mouseleave', function() {
                    document.querySelectorAll('.cite-region-' + n).forEach(function(el) {
                        el.classList.remove('highlight');
                    });
                });
            });
        </script>
        '''}
    </body>
    </html>
    """
    components.html(component_html, height=height, scrolling=True)


# ── Session state defaults ──────────────────────────────────────────────
if "current_result" not in st.session_state:
    st.session_state.current_result = None
if "history" not in st.session_state:
    st.session_state.history = []

# ── Sidebar ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    with st.expander("Configuration", expanded=False):
        api_url = st.text_input("API URL", value=DEFAULT_API_URL)
        top_k = st.slider("top_k (evidence count)", min_value=1, max_value=20, value=5)
        retrieval_mode = st.selectbox("Retrieval Mode", options=["Sparse", "Dense", "Hybrid"], index=2)
        alpha = st.slider("alpha (dense weight)", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
        user_id = st.text_input("User ID", value="default_user")

    st.divider()
    st.subheader("Query History")

    if st.button("Refresh History"):
        try:
            st.session_state.history = fetch_history(api_url, user_id)
        except Exception as e:
            st.error(f"Failed to load history: {e}")

    for i, item in enumerate(reversed(st.session_state.history)):
        question_preview = (item.get("question") or "Unknown")[:60]
        if st.button(question_preview, key=f"hist_{i}", use_container_width=True):
            st.session_state.current_result = item

# ── Main area ───────────────────────────────────────────────────────────
st.title("CS 5542 \u2014 RAG Chatbot")
st.caption("Ask a question and get an evidence-grounded answer from the document collection.")

with st.form("query_form", clear_on_submit=False):
    col_input, col_button = st.columns([5, 1])
    with col_input:
        query_text = st.text_input(
            "Your question",
            placeholder="e.g. Explain BM25 length normalization...",
            label_visibility="collapsed",
        )
    with col_button:
        ask_clicked = st.form_submit_button("Ask", type="primary", use_container_width=True)

# ── Handle query submission ─────────────────────────────────────────────
if ask_clicked and query_text.strip():
    with st.spinner("Searching and generating answer..."):
        try:
            result = call_query_api(api_url, query_text.strip(), top_k, retrieval_mode, alpha, user_id)
            st.session_state.current_result = result
            # Refresh history so the sidebar updates immediately
            try:
                st.session_state.history = fetch_history(api_url, user_id)
            except Exception:
                pass
            st.rerun()
        except requests.exceptions.ConnectionError:
            st.error(
                f"Cannot connect to the API at {api_url}. "
                "Make sure the FastAPI backend is running: "
                "`uvicorn backend.main:app --reload --port 3001`"
            )
        except requests.exceptions.HTTPError as e:
            st.error(f"API error: {e.response.status_code} \u2014 {e.response.text}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

# ── Display current result ──────────────────────────────────────────────
result = st.session_state.current_result
if result is not None:
    # Metadata chips
    meta_cols = st.columns(4)
    with meta_cols[0]:
        st.metric("Latency", f"{result.get('latency_ms', 0):.0f} ms")
    with meta_cols[1]:
        st.metric("top_k", result.get("top_k", "?"))
    with meta_cols[2]:
        st.metric("alpha", f"{result.get('alpha', '?')}")
    with meta_cols[3]:
        st.metric("Evidence", len(result.get("evidence", [])))

    # Missing evidence warning
    if result.get("missing_evidence_behavior"):
        st.markdown(
            '<div class="warning-banner">'
            "Evidence was insufficient for this query. "
            "The retrieved documents may not contain relevant information."
            "</div>",
            unsafe_allow_html=True,
        )

    # Answer panel
    st.subheader("Answer")
    answer_text = result.get("answer", "")
    has_citations = bool(re.search(r"\[\d+\]", answer_text))

    if has_citations:
        render_answer_component(answer_text)
    else:
        st.markdown(
            f'<div class="answer-card">{html_module.escape(answer_text)}</div>',
            unsafe_allow_html=True,
        )
        if not result.get("missing_evidence_behavior"):
            st.caption(
                "Inline citation markers [1], [2], etc. will appear "
                "once the LLM answer generator is connected."
            )

    # Sources panel
    st.subheader("Sources")
    evidence_list = result.get("evidence", [])
    for idx, ev in enumerate(evidence_list, start=1):
        with st.expander(
            f"[{idx}] {ev.get('source_file', 'unknown')} \u2014 "
            f"p.{ev.get('page', '?')}  |  "
            f"score: {ev.get('hybrid_score', 0):.3f}"
        ):
            score_cols = st.columns(3)
            with score_cols[0]:
                st.caption("Hybrid Score")
                st.progress(min(ev.get("hybrid_score", 0), 1.0))
            with score_cols[1]:
                st.caption("BM25 (normalized)")
                st.progress(min(ev.get("bm25_norm", 0), 1.0))
            with score_cols[2]:
                st.caption("Dense (normalized)")
                st.progress(min(ev.get("dense_norm", 0), 1.0))

            st.markdown("---")
            st.markdown(f"**Citation:** `{ev.get('citation', '')}`")
            st.markdown(f"**Evidence ID:** `{ev.get('evidence_id', '')}`")
            st.text_area(
                "Chunk text",
                value=ev.get("text", ""),
                height=150,
                disabled=True,
                key=f"source_text_{idx}",
            )
