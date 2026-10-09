import sys
import os

# Ensure the root directory is in Python's import search path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Now import your local packages
from src.indexing import build_or_load_vectorstores
from src.engines import run_v1, run_v2, run_v3, run_v4

import os
import streamlit as st
from flashrank import Ranker
from src.indexing import build_or_load_vectorstores
from src.engines import run_v1, run_v2, run_v3, run_v4

st.set_page_config(
    page_title="FIDE Arbiter Copilot",
    page_icon="♟️",
    layout="wide"
)

st.title("♟️ FIDE Arbiter Copilot")
st.caption("Regulatory decision-support system evaluating V1 (Naive) through V4 (Two-Stage Reranked RAG).")

# ------------------------------------------------------------------------------
# SIDEBAR SETUP & API KEYS
# ------------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    default_key = os.environ.get("GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY", ""))
    api_key = st.text_input("Gemini API Key", value=default_key, type="password")

    st.markdown("---")
    st.subheader("📖 Indexed Rulebooks")
    st.markdown("""
    - **Arbiters' Manual 2025** *(General Laws, Articles 1–12)*
    - **ACC Regulations** *(Anti-Cheating Complaints)*
    - **C02 Standards** *(Tie-Breaks & Virtual Opponents)*
    """)

# Cache FlashRank cross-encoder model
@st.cache_resource
def get_ranker():
    return Ranker(model_name="ms-marco-TinyBERT-L-2-v2", cache_dir="./flashrank_cache")

with st.spinner("Loading vectorstores and neural models..."):
    v1_store, v2_store, v3_chroma = build_or_load_vectorstores()
    ranker = get_ranker()

# ------------------------------------------------------------------------------
# QUERY INPUT INTERFACE
# ------------------------------------------------------------------------------
sample_queries = [
    "A player suspects an opponent has a hidden electronic device in their clothing or ears. What inspection steps can the arbiter perform?",
    "A player touches a piece, realizes it has no legal moves, and moves another piece. What penalty applies?",
    "Black's flag falls, but White only has a bare King. Is the result a win for White or a draw?",
    "How is an unplayed game (forfeit win) scored for Buchholz Cut 1 under the Virtual Opponent system?"
]

selected_query = st.selectbox("Select a sample dispute scenario (or type custom below):", ["Custom Query"] + sample_queries)
if selected_query == "Custom Query":
    user_query = st.text_area("Dispute Scenario:", value=sample_queries[0], height=100)
else:
    user_query = st.text_area("Dispute Scenario:", value=selected_query, height=100)

tab_prod, tab_benchmark = st.tabs(["🚀 Production Ruling (V4)", "📊 Comparative Benchmark (V1 – V4)"])

# ------------------------------------------------------------------------------
# TAB 1: PRODUCTION V4
# ------------------------------------------------------------------------------
with tab_prod:
    if st.button("Generate Official Ruling", type="primary", use_container_width=True):
        if not api_key:
            st.error("Please enter a valid Gemini API key in the sidebar.")
        else:
            with st.spinner("Reranking candidate chunks and generating ruling..."):
                try:
                    answer, docs, t_ret, t_total = run_v4(user_query, v3_chroma, ranker, api_key)
                    
                    st.success(f"Ruling generated in **{t_total:.2f}s** (Retrieval & Rerank: {t_ret:.3f}s)")
                    st.markdown("### Official Ruling & Legal Grounding")
                    st.markdown(answer)

                    st.markdown("---")
                    st.subheader("📚 Verified Source Documents")
                    for i, doc in enumerate(docs, 1):
                        score = doc.metadata.get("score", 0.0)
                        fname = doc.metadata.get("filename", "Unknown")
                        page = doc.metadata.get("page", 0)
                        with st.expander(f"Rank #{i} (Score: {score:.4f}) | {fname} — Page {page}"):
                            st.write(doc.page_content)
                except Exception as e:
                    st.error(f"Generation error: {str(e)}")

# ------------------------------------------------------------------------------
# TAB 2: MULTI-VERSION COMPARISON (V1 vs V2 vs V3 vs V4)
# ------------------------------------------------------------------------------
with tab_benchmark:
    st.info("Demonstrates how chunk boundaries, metadata filters, and cross-encoder reranking affect the final ruling.")
    if st.button("Run Multi-Version Comparison", use_container_width=True):
        if not api_key:
            st.error("Please enter a valid Gemini API key in the sidebar.")
        else:
            c1, c2, c3, c4 = st.columns(4)

            # V1 Column
            with c1:
                st.markdown("#### V1: Naive FAISS")
                with st.spinner("V1 processing..."):
                    try:
                        ans1, docs1, tr1, tt1 = run_v1(user_query, v1_store, api_key)
                        st.caption(f"⏱️ Retrieval: {tr1:.3f}s | Total: {tt1:.2f}s")
                        st.write(ans1[:400] + "...")
                        with st.expander("V1 Sources"):
                            for d in docs1:
                                st.write(f"- {d.metadata.get('filename')} (P.{d.metadata.get('page', 0)+1})")
                    except Exception as e:
                        st.error(f"V1: {e}")

            # V2 Column
            with c2:
                st.markdown("#### V2: Article FAISS")
                with st.spinner("V2 processing..."):
                    try:
                        ans2, docs2, tr2, tt2 = run_v2(user_query, v2_store, api_key)
                        st.caption(f"⏱️ Retrieval: {tr2:.3f}s | Total: {tt2:.2f}s")
                        st.write(ans2[:400] + "...")
                        with st.expander("V2 Sources"):
                            for d in docs2:
                                st.write(f"- {d.metadata.get('filename')} (P.{d.metadata.get('page', 0)+1})")
                    except Exception as e:
                        st.error(f"V2: {e}")

            # V3 Column
            with c3:
                st.markdown("#### V3: ChromaDB Filter")
                with st.spinner("V3 processing..."):
                    try:
                        ans3, docs3, tr3, tt3 = run_v3(user_query, v3_chroma, api_key, doc_filter="anti_cheating")
                        st.caption(f"⏱️ Retrieval: {tr3:.3f}s | Total: {tt3:.2f}s")
                        st.write(ans3[:400] + "...")
                        with st.expander("V3 Sources"):
                            for d in docs3:
                                st.write(f"- {d.metadata.get('filename')} (P.{d.metadata.get('page', 0)+1})")
                    except Exception as e:
                        st.error(f"V3: {e}")

            # V4 Column
            with c4:
                st.markdown("#### V4: Two-Stage Hybrid")
                with st.spinner("V4 processing..."):
                    try:
                        ans4, docs4, tr4, tt4 = run_v4(user_query, v3_chroma, ranker, api_key)
                        st.caption(f"⏱️ Retrieval: {tr4:.3f}s | Total: {tt4:.2f}s")
                        st.write(ans4[:400] + "...")
                        with st.expander("V4 Sources (Reranked)"):
                            for d in docs4:
                                st.write(f"- {d.metadata.get('filename')} (P.{d.metadata.get('page')}) [{d.metadata.get('score', 0):.3f}]")
                    except Exception as e:
                        st.error(f"V4: {e}")