# Telegram RAG Search Bot (FAISS + BM25)

A hybrid Retrieval-Augmented search bot for Telegram datasets.
Combines semantic vector search (FAISS) with keyword ranking (BM25),
and surfaces relevant message threads (smart replies).

## Features
- Hybrid ranking: FAISS semantic search + BM25 keyword search
- Advanced filters: date/range, replies-only, include/exclude keywords
- Smart replies: thread traversal to extract the best related replies
- Inline search support for Telegram
- Exports: JSON / CSV / HTML
- Modular architecture with async loading and concurrency support
- Reliability checks: index/rows validation and graceful fallbacks

## Tech Stack
- Python
- FAISS
- SentenceTransformers (embeddings)
- BM25 ranking
- Telegram Bot API (PTB/handlers-based UI)

## Architecture (High-level)
- DataRepository: loads/normalizes messages + thread relations
- FaissIndex: loads FAISS index + sets tuning params (e.g., nprobe)
- EmbeddingModel: encodes queries (with caching/batching)
- SearchEngine: hybrid scoring + filters + optional reranking
- UI/Handlers: Telegram UI pages, pagination, exports

## Use Cases
- Search inside large Telegram groups/channels archives
- University support bots (FAQs + thread-based answers)
- Knowledge base search over chat logs

## Status
Production-ready / Actively iterated
