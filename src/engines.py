import time
from flashrank import Ranker, RerankRequest
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

PROMPT_TEMPLATE = """
You are the official FIDE Arbiter Copilot.
Answer the chess dispute question STRICTLY based on the retrieved context below.

Retrieved Context:
{context}

Dispute Question:
{question}

Structure your response into:
1. Official Ruling & Legal Grounding
2. Arbiter Action Checklist (Exact procedural steps)
3. Prescribed Sanctions / Technical Specifications
4. Verifiable Citations (Document name, Article/Chapter, and Page number)
"""

def get_llm(api_key: str):
    """Initializes Gemini with active fallback endpoints to prevent 404/429 failures."""
    candidate_models = ["gemini-flash-latest", "gemini-2.5-flash", "gemini-3.8-flash"]
    for model_name in candidate_models:
        try:
            return ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=0,
                max_retries=1
            )
        except Exception:
            continue
    # Default fallback
    return ChatGoogleGenerativeAI(
        model="gemini-flash-latest",
        google_api_key=api_key,
        temperature=0
    )

def _generate_answer(query: str, context_str: str, api_key: str):
    llm = get_llm(api_key)
    chain = ChatPromptTemplate.from_template(PROMPT_TEMPLATE) | llm | StrOutputParser()
    return chain.invoke({"context": context_str, "question": query})

# ------------------------------------------------------------------------------
# V1: NAIVE RAG (Fixed-size chunks + FAISS)
# ------------------------------------------------------------------------------
def run_v1(query: str, naive_faiss, api_key: str, k: int = 3):
    t0 = time.perf_counter()
    docs = naive_faiss.similarity_search(query, k=k)
    t_ret = time.perf_counter() - t0

    context = "\n\n---\n\n".join([f"[{d.metadata.get('filename')} P.{d.metadata.get('page', 0)+1}]\n{d.page_content}" for d in docs])
    answer = _generate_answer(query, context, api_key)
    t_total = time.perf_counter() - t0

    return answer, docs, t_ret, t_total

# ------------------------------------------------------------------------------
# V2: SEMANTIC ARTICLE-AWARE RAG (FAISS)
# ------------------------------------------------------------------------------
def run_v2(query: str, faiss_store, api_key: str, k: int = 3):
    t0 = time.perf_counter()
    docs = faiss_store.similarity_search(query, k=k)
    t_ret = time.perf_counter() - t0

    context = "\n\n---\n\n".join([f"[{d.metadata.get('filename')} P.{d.metadata.get('page', 0)+1}]\n{d.page_content}" for d in docs])
    answer = _generate_answer(query, context, api_key)
    t_total = time.perf_counter() - t0

    return answer, docs, t_ret, t_total

# ------------------------------------------------------------------------------
# V3: CHROMADB WITH METADATA FILTERING
# ------------------------------------------------------------------------------
def run_v3(query: str, chroma_store, api_key: str, doc_filter: str = "all", k: int = 3):
    t0 = time.perf_counter()
    kwargs = {"k": k}
    if doc_filter and doc_filter != "all":
        kwargs["filter"] = {"doc_type": doc_filter}
        
    docs = chroma_store.similarity_search(query, **kwargs)
    t_ret = time.perf_counter() - t0

    context = "\n\n---\n\n".join([f"[{d.metadata.get('filename')} P.{d.metadata.get('page', 0)+1}]\n{d.page_content}" for d in docs])
    answer = _generate_answer(query, context, api_key)
    t_total = time.perf_counter() - t0

    return answer, docs, t_ret, t_total

# ------------------------------------------------------------------------------
# V4: TWO-STAGE HYBRID (ChromaDB Recall + FlashRank Cross-Encoder)
# ------------------------------------------------------------------------------
def run_v4(query: str, chroma_store, ranker: Ranker, api_key: str, top_k_candidates: int = 15, final_n: int = 3):
    t0 = time.perf_counter()
    
    # Stage 1: Coarse candidate retrieval across all rulebooks
    candidates = chroma_store.similarity_search(query, k=top_k_candidates)
    
    # Stage 2: Token-level cross-attention reranking
    passages = [
        {
            "id": idx,
            "text": doc.page_content,
            "meta": {
                "filename": doc.metadata.get("filename", "Unknown"),
                "page": doc.metadata.get("page", 0) + 1,
                "doc_type": doc.metadata.get("doc_type", "Unknown")
            }
        }
        for idx, doc in enumerate(candidates)
    ]
    rerank_request = RerankRequest(query=query, passages=passages)
    reranked = ranker.rerank(rerank_request)[:final_n]
    t_ret = time.perf_counter() - t0

    context_blocks = []
    top_docs = []
    for item in reranked:
        meta = item["meta"]
        score = item["score"]
        header = f"[{meta['filename']} (P. {meta['page']}) | Cross-Encoder Score: {score:.4f}]"
        context_blocks.append(f"{header}\n{item['text']}")
        top_docs.append(Document(page_content=item["text"], metadata={**meta, "score": score}))

    context = "\n\n---\n\n".join(context_blocks)
    answer = _generate_answer(query, context, api_key)
    t_total = time.perf_counter() - t0

    return answer, top_docs, t_ret, t_total