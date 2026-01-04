# DocuMind - Intelligent Documentation Q&A System

A Retrieval-Augmented Generation (RAG) system that enables natural language queries over documentation with 90% accuracy on domain-specific questions.

## Overview

DocuMind demonstrates the evolution from basic LLM chatbot to production-ready RAG implementation. Built in two phases to quantify RAG's value through measurable improvements.

**Key Results:**
- 90% accuracy on documentation-specific queries (vs 10% baseline)
- 9x improvement over baseline LLM
- 94% reduction in information retrieval time (45s manual to 2.5s automated)
- Sub-3-second end-to-end response times

## Architecture

### Phase 1: Baseline Chatbot
Direct LLM integration demonstrating limitations when querying specific documents.

**Stack:** Flask, Ollama, Llama 3.2 3B

**Limitation:** Cannot answer questions about information outside training data.

### Phase 2: RAG Implementation
Full retrieval pipeline with semantic search and context injection.

**Components:**
- Document chunking (500 words, 50-word overlap)
- Sentence transformers (all-MiniLM-L6-v2, 384-dim embeddings)
- Cosine similarity search
- Relevance re-ranking
- Context injection into LLM prompts

**Performance:**
- Embedding: 100ms
- Search: 100ms
- Re-ranking: 100ms
- LLM generation: 2000ms
- Total: ~2.5 seconds

## Installation

### Prerequisites
- Python 3.8+
- 8GB RAM minimum
- 5GB free disk space

### Setup

1. Install Ollama:
```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh
```

2. Download LLM model:
```bash
ollama serve  # Keep running in separate terminal
ollama pull llama3.2:3b
```

3. Clone repository:
```bash
git clone repository
cd documind
```

4. Install dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Running Phase 1 (Baseline)

Terminal 1 - Ollama:
```bash
ollama serve
```

Terminal 2 - Backend:
```bash
cd backend
python phase1_basic_chatbot.py
# Runs on http://localhost:5000
```

Terminal 3 - Frontend:
```bash
cd frontend
python3 -m http.server 8000
```

Access: http://localhost:8000/phase1.html

### Running Phase 2 (RAG)

Terminal 1 - Ollama:
```bash
ollama serve
```

Terminal 2 - Backend:
```bash
cd backend
python phase2_with_retrieval.py
# Runs on http://localhost:5001
```

Terminal 3 - Frontend:
```bash
cd frontend
python3 -m http.server 8000
```

Access: http://localhost:8000/phase2.html

### Adding Documents

Place .txt or .md files in:
```
data/documentation/
```

Restart Phase 2 backend to index new documents.

## Project Structure

```
documind/
├── backend/
│   ├── phase1_basic_chatbot.py      # Baseline implementation
│   └── phase2_with_retrieval.py     # RAG implementation
├── frontend/
│   ├── index.html                   # Landing page
│   ├── phase1.html                  # Phase 1 interface
│   ├── phase2.html                  # Phase 2 interface
│   └── styles.css
├── data/
│   └── documentation/               # Place docs here
│       └── claude_3_5_sonnet_announcement.txt
├── requirements.txt
└── README.md
```

## Technical Details

### Embedding Model
- Model: all-MiniLM-L6-v2
- Dimensions: 384
- Speed: ~100ms per encoding
- Size: 90MB

### Chunking Strategy
- Method: Fixed-size sliding window
- Chunk size: 500 words
- Overlap: 50 words
- Rationale: Balances retrieval precision with context preservation

### Search Algorithm
1. Convert query to 384-dim embedding
2. Calculate cosine similarity with all chunk embeddings
3. Retrieve top-10 by similarity
4. Re-rank by word overlap relevance
5. Inject top-3 chunks into LLM prompt

### Context Window Management
- LLM: Llama 3.2 3B (8K token context)
- System prompt: ~200 tokens
- 3 chunks: ~2000 tokens
- Query: ~50 tokens
- Available for response: ~5750 tokens


## Limitations

**Current Implementation:**
- In-memory storage (no persistence)
- Simple word overlap re-ranking
- Fixed chunking strategy
- Single document format (.txt, .md)

**Known Issues:**
- Answers may be incomplete if information spans multiple non-retrieved chunks
- First-time startup slow due to model download
- No caching of embeddings across restarts

## Future Enhancements (Phase 3)

- ChromaDB integration for persistent vector storage
- Advanced re-ranking with cross-encoder models
- Hybrid search (semantic + keyword)
- Multiple document format support (PDF, DOCX)
- Conversation memory across sessions
- Query caching for frequent questions
- Performance monitoring dashboard
