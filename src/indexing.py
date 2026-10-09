import os
import streamlit as st

# Suppress OpenTelemetry warnings & telemetry overhead
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["OTEL_SDK_DISABLED"] = "true"

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

DOC_CATALOG = {
    "Arbiters_Manual_2025.pdf": "arbiter_manual",
    "ACCRegulations.pdf": "anti_cheating",
    "C02Standards.pdf": "tie_breaks"
}

@st.cache_resource
def load_embeddings():
    """Caches sentence-transformers model to prevent repeated downloads."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def build_or_load_vectorstores(data_dir="data", persist_dir="./fide_chroma_v3"):
    """
    Builds or connects to ChromaDB and constructs in-memory FAISS indices
    for V1 (Naive) and V2 (Article-aware) benchmarking.
    """
    embeddings = load_embeddings()
    
    # Check if persistent ChromaDB already exists
    if not os.path.exists(persist_dir) or len(os.listdir(persist_dir)) == 0:
        tagged_docs = []
        for filename, doc_type in DOC_CATALOG.items():
            filepath = os.path.join(data_dir, filename)
            if os.path.exists(filepath):
                loader = PyPDFLoader(filepath)
                pages = loader.load()
                for p in pages:
                    p.metadata["doc_type"] = doc_type
                    p.metadata["filename"] = filename
                tagged_docs.extend(pages)
            else:
                st.warning(f"File not found: {filepath}. Please place all 3 PDFs in the 'data/' folder.")

        # V2-V4: Regex Article-Aware chunking
        regex_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1800,
            chunk_overlap=200,
            separators=[
                r"\n(?=Article\s+\d+)",
                r"\n(?=Appendix\s+[A-Z])",
                r"\n(?=[A-Z]\.\d+)",
                r"\n(?=\d+\.\d+\s+)",
                "\n\n", "\n", " "
            ],
            is_separator_regex=True
        )
        article_chunks = regex_splitter.split_documents(tagged_docs)

        chroma_store = Chroma.from_documents(
            documents=article_chunks,
            embedding=embeddings,
            persist_directory=persist_dir
        )
    else:
        chroma_store = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )

    # Instantiate V2: In-memory FAISS using article-aware chunks from Chroma
    chroma_records = chroma_store.get()
    article_docs = [
        Document(page_content=txt, metadata=meta)
        for txt, meta in zip(chroma_records["documents"], chroma_records["metadatas"])
    ]
    faiss_v2_store = FAISS.from_documents(article_docs, embeddings)

    # Instantiate V1: Naive fixed 1000-char chunks for baseline comparison
    naive_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    naive_docs = naive_splitter.split_documents(article_docs)
    faiss_v1_store = FAISS.from_documents(naive_docs, embeddings)

    return faiss_v1_store, faiss_v2_store, chroma_store