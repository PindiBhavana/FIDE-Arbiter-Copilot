"""
FIDE Arbiter Copilot Source Package.
Contains vector indexing and multi-version RAG inference engines.
"""
from .indexing import load_embeddings, build_or_load_vectorstores
from .engines import run_v1, run_v2, run_v3, run_v4

__all__ = [
    "load_embeddings",
    "build_or_load_vectorstores",
    "run_v1",
    "run_v2",
    "run_v3",
    "run_v4"
]