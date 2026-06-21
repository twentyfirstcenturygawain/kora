# Kora — AI Knowledge Agent

A production-ready RAG (Retrieval-Augmented Generation) pipeline. Load any company's documentation and instantly get an AI agent that answers questions grounded in that content — zero hallucination.

![Python](https://img.shields.io/badge/Python-3.11+-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green) ![Claude](https://img.shields.io/badge/Claude-Sonnet-orange)

## How it works

1. **Ingest** — paste any FAQ, docs, or support content
2. **Chunk** — splits into overlapping 300-word chunks
3. **Embed** — TF-IDF vectors computed for every chunk
4. **Retrieve** — cosine similarity finds the most relevant chunks for your question
5. **Answer** — Claude generates a grounded answer using only retrieved context

## Stack

- **Backend** — Python, FastAPI
- **AI** — Anthropic Claude API (`claude-sonnet-4-6`)
- **Retrieval** — TF-IDF embeddings + cosine similarity (NumPy)
- **Frontend** — Vanilla HTML/CSS/JS

## Run locally

```bash
git clone https://github.com/twentyfirstcenturygawain/kora.git
cd kora
pip install -r requirements.txt
cp .env.example .env  # add your Anthropic API key
python -m uvicorn main:app --reload
```

Open `http://localhost:8000`

## API

```bash
# Ask a question
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What plans do you offer?"}'

# Load custom docs
curl -X POST http://localhost:8000/api/load-docs \
  -H "Content-Type: application/json" \
  -d '{"content": "Your docs here...", "source_name": "My Docs"}'
```
