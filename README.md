# ♟️ FIDE Arbiter Copilot: An Intelligent Regulatory & Anti-Cheating RAG Assistant

## 📌 Project Overview

**FIDE Arbiter Copilot** is a Retrieval-Augmented Generation (RAG) application designed to assist chess arbiters in accessing official FIDE regulations and resolving tournament-related queries. It combines semantic search, article-aware document chunking, metadata-based retrieval, and cross-encoder reranking to retrieve relevant regulatory passages and generate structured, citation-grounded responses.

The system aims to reduce the time spent manually searching lengthy regulatory documents while helping arbiters make informed decisions based on retrieved official rules.

## 🎯 Problem Statement

Chess arbiters often need to consult multiple official regulatory documents when handling tournament disputes. Manually locating relevant articles can be time-consuming, while general-purpose AI systems may generate unsupported answers or inaccurate rule references.

This project addresses these challenges by building a RAG-based assistant that retrieves relevant regulatory passages and uses them as context for generating responses.

## ✨ Key Features

- **Article-Aware Chunking:** Splits regulatory documents around articles, appendices, and sub-clauses to preserve legal context.
- **Semantic Retrieval:** Uses MiniLM embeddings to represent document passages and user queries for similarity-based search.
- **Metadata-Enhanced Search:** Uses ChromaDB to store document chunks with metadata such as document type, source file, and page information.
- **Two-Stage Retrieval:** Retrieves the top 15 candidate passages and applies FlashRank cross-encoder reranking to select the top 3 most relevant passages.
- **Citation-Grounded Responses:** Uses Gemini to generate structured answers based on retrieved regulatory context, including relevant citations and decision checklists.
- **Model Fallback:** Implements a Gemini model fallback strategy to handle API availability issues and rate limits.
- **Pipeline Observability:** Integrates LangSmith tracing to monitor pipeline execution, retrieval latency, token usage, and qualitative arbiter feedback.
- **Benchmark Evaluation:** Evaluates retrieval performance using a 20-question benchmark.

## 🏗️ Technical Architecture

1. **Document Ingestion:** Extracts text from official FIDE regulatory PDFs.
2. **Article-Aware Chunking:** Divides documents into meaningful regulatory sections while preserving article and sub-clause boundaries.
3. **Embedding Generation:** Converts text chunks into vector representations using MiniLM.
4. **Vector Storage:** Stores embeddings and associated metadata in ChromaDB.
5. **Candidate Retrieval:** Retrieves the top 15 candidate passages relevant to the user's query.
6. **Reranking:** Uses FlashRank to rerank the retrieved passages and select the top 3.
7. **Response Generation:** Passes the query and selected regulatory context to Gemini to generate a structured response.
8. **Tracing and Evaluation:** Uses LangSmith to trace pipeline execution and assess retrieval performance.

## 🛠️ Tech Stack

- **Programming Language:** Python
- **RAG Framework:** LangChain
- **Embedding Model:** MiniLM
- **Vector Database:** ChromaDB
- **Reranking:** FlashRank
- **LLM:** Gemini
- **Observability:** LangSmith
- **Document Processing:** PDF text extraction and article-aware chunking

## 📊 Results

- **762** article-aware document chunks created from 3 FIDE PDF documents spanning 395 pages.
- **90% retrieval hit rate** across a 20-question benchmark.
- **15 → 3** candidate selection using FlashRank reranking.
- **0.27 seconds average retrieval and reranking time** for the V4 hybrid pipeline in the reported evaluation.


## ⚖️ Example Use Cases

The assistant is designed to support queries involving:

- Illegal moves and related rulings
- Chess clock malfunctions
- Electronic device infractions
- Tie-break disputes

It serves as a decision-support tool for locating relevant regulations. Final decisions remain the responsibility of the tournament arbiter.

## 🚀 Future Enhancements

- Offline or edge-based inference
- Multimodal incident analysis using scoresheets, chess clocks, and board images
- Automated auditing of tie-break calculations and Swiss pairing procedures
- Voice-based query interaction
- Expanded benchmarking and continuous evaluation

## 💡 Key Learning Outcomes

- Designed and compared four RAG pipeline versions, from baseline retrieval to hybrid retrieval with reranking.
- Improved regulatory context preservation through article-aware chunking.
- Combined vector similarity search with cross-encoder reranking.
- Implemented metadata-based retrieval and citation-grounded response generation.
- Applied LangSmith tracing to improve pipeline observability and evaluation.

## 👩‍💻 Project Information

- **Project:** FIDE Arbiter Copilot
- **Domain:** Sports Governance
- **Approach:** Retrieval-Augmented Generation (RAG)
- **Primary Goal:** Provide fast, context-grounded access to official chess regulations.
  
---

## Title: FIDE Arbiter Copilot

emoji: ♟️

colorFrom: blue

colorTo: indigo

sdk: streamlit

sdk_version: 1.38.0

app_file: app.py

pinned: false

---

## FIDE Arbiter Copilot RAG

An AI legal assistant for tournament chess arbiters covering:
- FIDE Arbiter's Manual 2025
- Anti-Cheating Regulations (ACC)
- FIDE Standards C02 (Tie-breaks)
