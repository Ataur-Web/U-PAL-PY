# U-Pal, a bilingual UWTSD student assistant

U-Pal is a retrieval-augmented chatbot that answers student questions in
Welsh and English. It was built as a Level 6 BSc Applied Computing
dissertation project at the University of Wales Trinity Saint David (UWTSD).

**Live demo:** https://u-pal-py.vercel.app

The repo holds both halves of the system:

- `app/` + `scripts/` + `run.py`, a FastAPI Python backend that does the
  retrieval, Welsh detection and LLM call.
- `frontend/`, a Next.js 14 frontend that ships to Vercel.

---

## Stack

| Layer | Tech |
| ----- | ---- |
| Frontend | Next.js 14, React 18, hosted on Vercel |
| Backend | FastAPI + Uvicorn (Python 3.11 / 3.12) |
| LLM (primary) | Anthropic Claude Haiku 4.5 (`claude-haiku-4-5-20251001`) via `langchain-anthropic` |
| LLM (fallback) | Ollama Llama 3.1 8B (local, via ngrok static domain) |
| Retrieval | Hybrid, ChromaDB dense + BM25, merged with LangChain `EnsembleRetriever` |
| Embeddings | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Welsh detection | Multi-signal detector trained on BydTermCymru terminology |
| Feedback storage | MongoDB Atlas, with a local JSON fallback |
| Tunnel | ngrok static domains |

References the project leans on:

- Lewis et al. 2020, *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, https://arxiv.org/abs/2005.11401
- Reimers & Gurevych 2019, *Sentence-BERT*, https://arxiv.org/abs/1908.10084
- Robertson & Zaragoza 2009, *The Probabilistic Relevance Framework, BM25 and Beyond*
- Greshake et al. 2023, *Not what you've signed up for* (indirect prompt injection), https://arxiv.org/abs/2302.12173

---

## Quickstart, clone and run locally

The repo is public, anyone can clone and run their own copy for testing.

### 1. Prerequisites

- Node.js 18+
- Python 3.11 or 3.12
- Git
- An Anthropic API key (free tier is enough for a demo), from
  https://console.anthropic.com/settings/keys
- *Optional*, a MongoDB Atlas free M0 cluster for feedback storage.
  Without it the frontend writes feedback to a local JSON file.
- *Optional*, [Ollama](https://ollama.com) + [ngrok](https://ngrok.com)
  if you want a local LLM fallback.

### 2. Clone

```bash
git clone https://github.com/Ataur-Web/U-PAL-PY.git
cd U-PAL-PY
```

### 3. Backend (FastAPI)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set `ANTHROPIC_API_KEY`. Everything else has a sensible
default.

```bash
# first-time only, build the Chroma vector store from the JSON corpus
# (downloads the ~450 MB embedding model on first run)
python -m scripts.ingest --reset

# start the backend (Windows-safe entrypoint that avoids a uvicorn CLI
# segfault inside sentence-transformers model load)
python run.py
```

Health check at http://localhost:3001/api/health.

### 4. Frontend (Next.js)

Open a second terminal:

```bash
cd frontend
npm install
```

Create `frontend/.env.local`:

```
ADMIN_PASSWORD=your-secret-password
MONGODB_URI=                    # optional, omit to use local JSON
CHAT_BACKEND_URL=http://localhost:3001
```

Then:

```bash
npm run dev
```

Open http://localhost:3000 and ask U-Pal a question. The admin dashboard
is at http://localhost:3000/admin (any username + your `ADMIN_PASSWORD`).

---

## Project layout

```
U-PAL-PY/
  app/
    main.py               FastAPI entrypoint
    config.py             .env-backed settings
    routes/
      chat.py             POST /api/chat
      feedback.py         (handled by frontend, kept for reference)
      health.py           GET  /api/health
      llm_config.py       GET/POST /api/llm-config, runtime LLM switch
      translate.py        POST /api/translate
    services/
      rag.py              ChromaDB retriever + hybrid ranking
      llm.py              LangChain wrapper around Claude / Ollama
      welsh.py            Language detection, bilingual query rewrite
      intent.py           Knowledge-base intent classifier
      emotion.py          Emotional state detector (5 states, EN + CY)
      conversation.py     History, student profile, anchor resolution
    prompts/
      system_en.txt       English system prompt
      system_cy.txt       Welsh system prompt
    data/                 Knowledge base + corpus JSON files
  scripts/
    ingest.py                    Build the Chroma vector index
    fetch_openorca.py            Optional, OpenOrca instructional Q&A
    fetch_naturalquestions.py    Optional, Google Natural Questions general knowledge
    fetch_welsh_chat.py          Optional, nemotron-chat-welsh for Welsh fluency
    fetch_termcymru.py           Refresh BydTermCymru bilingual map
  tests/
    test_welsh.py
    test_health.py
  frontend/
    pages/                Next.js pages (index, admin, API routes)
    lib/                  Embed + rerank + safety + DB helpers
    styles/               CSS
    demo/                 Stand-alone integration demos
    *.json                Corpus, knowledge base, Welsh maps
  run.py                  Windows-safe backend launcher
  requirements.txt
  .env.example
  README.md
```

---

## Deploy the frontend to Vercel

1. Fork this repo on GitHub.
2. In Vercel, click **Add New Project** and import your fork.
3. Set **Root Directory** to `frontend`.
4. Under **Settings → Environment Variables** add:
   - `ADMIN_PASSWORD`
   - `MONGODB_URI` (optional)
   - `CHAT_BACKEND_URL` pointing at your public backend (ngrok static
     domain works well during testing)
5. Deploy. Every push to `main` triggers a rebuild.

The backend needs to be reachable from the Vercel frontend. For a
zero-cost demo, run the backend locally and expose it with an ngrok
static domain, then point `CHAT_BACKEND_URL` at that URL.

---

## Enriching the knowledge base (optional)

The default Chroma index covers the curated UWTSD JSON corpus. Two helper
scripts can broaden coverage so U-Pal can hold a more natural conversation
on general academic topics:

```bash
# instructional Q&A from OpenOrca (filtered to education topics)
fetch-openorca.bat            # Windows one-click
# or
python -m scripts.fetch_openorca --sample 20000

# general-knowledge Q&A from Google Natural Questions
fetch-naturalquestions.bat    # Windows one-click
# or
python -m scripts.fetch_naturalquestions --sample 30000

# Welsh chat pairs from nemotron-chat-welsh, improves Welsh fluency
fetch-welsh-chat.bat          # Windows one-click
# or
python -m scripts.fetch_welsh_chat --sample 15000
```

All three scripts stream the dataset, filter for student-relevant or
Welsh-quality rows, and ingest into Chroma so the next chat turn can
retrieve them.

References for the datasets:

- OpenOrca, https://huggingface.co/datasets/Open-Orca/OpenOrca
- Natural Questions, Kwiatkowski et al. 2019, *Natural Questions: a Benchmark for Question Answering Research*, https://ai.google.com/research/NaturalQuestions
- Nemotron Chat Welsh, https://huggingface.co/datasets/locailabs/nemotron-chat-welsh
- BydTermCymru, https://termau.cymru

---

## LLM configuration

Default provider is Anthropic Claude Haiku 4.5 (`claude-haiku-4-5-20251001`).
The runtime provider and model can be swapped without a restart via:

```bash
curl -X POST http://localhost:3001/api/llm-config \
  -H "Content-Type: application/json" \
  -d '{"provider": "ollama", "model": "llama3.1:8b-instruct-q5_K_M"}'
```

Supported providers, `anthropic` and `ollama`.

---

## Testing

```bash
pytest
```

Unit tests cover the Welsh detector and the health endpoint. Integration
testing of the retrieval + LLM path is manual, the live demo at
https://u-pal-py.vercel.app is the easiest way to exercise it.

---

## Licence & research use

This is a research prototype built for a UWTSD Level 6 dissertation.
Anonymised interaction data (satisfaction ratings, free-text feedback)
from the live demo may be used in the academic write-up. No
personally-identifying information is collected.

Not a substitute for official university support, wellbeing services or
professional advice.
