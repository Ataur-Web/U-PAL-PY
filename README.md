# U-Pal, a bilingual UWTSD student assistant

U-Pal is a retrieval-augmented chatbot that answers student questions in
Welsh and English. It was built as a Level 6 BSc Applied Computing
dissertation project at the University of Wales Trinity Saint David (UWTSD).

**Live demo:** https://u-pal-py.vercel.app

The system detects the language of each query without the user declaring it,
retrieves grounding passages from a UWTSD knowledge base using hybrid dense and
sparse retrieval, and answers in the language the student wrote in, showing the
institutional sources it used.

The repo holds both halves of the system:

- `app/` + `scripts/` + `run.py`, a FastAPI Python backend that does the
  retrieval, Welsh detection and LLM call.
- `frontend/`, a Next.js 14 frontend that ships to Vercel.

---

## Evaluation summary

Full method and discussion are in Chapter 5 of the dissertation; the raw data
for every figure below is in `benchmark_results/`, indexed by that folder's own
README.

| Requirement | Measure | Result |
|---|---|---|
| F5 hybrid retrieval | Hit@5 / MRR, 20 queries, relaxed rule | Hybrid 0.90 / 0.66, ahead of dense 0.75 / 0.51 and sparse 0.75 / 0.53 |
| F2 language detection | 100 queries, no language declared | 99.0% accuracy; Welsh recall 98.3%, English 100% |
| F8 source attribution | 100 queries, 256 citations | 256/256 traced to an indexed document and resolving to a live UWTSD page |
| F7 prompt injection | 130 cases (70 attack / 60 benign) | 100% attack recall, 5.0% false-positive rate, 97.5% balanced accuracy |
| N1 latency | 30 requests per provider | Claude Haiku 4.5 mean 2.81 s; local Llama 3.1 8B mean 6.05 s |
| Usability | System Usability Scale, n = 50 | Mean 71.80 (SD 11.10) |

Two findings shaped the final system. Welsh retrieval initially scored an MRR
of 0.20 against 0.71 for English; the cause was the composition of the Welsh
index rather than the multilingual encoder, and excluding glossary entries and
conversational intents from the dense pass raised Welsh hybrid MRR from 0.40 to
0.65. The local fallback provider meets the latency target overall but not for
Welsh queries alone, which average 8.51 s.

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
    map_source_urls.py           Map curated facts to institutional URLs
    organize_corpus.py           Corpus preparation
    process_scraped_corpus.py    Convert crawl output into the corpus format
    complete_scrape_workflow.py  End-to-end crawl helper
    fetch_termcymru.py           Refresh the BydTermCymru bilingual map
    fetch_openorca.py            Optional, OpenOrca instructional Q&A
    fetch_naturalquestions.py    Optional, Google Natural Questions
    fetch_welsh_chat.py          Optional, Welsh-language chat data
    benchmark_retrieval.py       Dense vs sparse vs hybrid retrieval (F5)
    benchmark_latency.py         End-to-end response latency (N1)
    make_langdetect_sample.py    Draw the language-detection test sample
    benchmark_langdetect.py      Language detector, component level only
    benchmark_f2_endtoend.py     Language handling through the API (F2)
    benchmark_sources.py         Source attribution (F8)
    injection_prompts.py         Prompt-injection test set (130 cases)
    benchmark_injection.py       Prompt-injection resistance (F7)
    analyse_sus.py               System Usability Scale analysis
    legacy/                      Superseded scripts, kept for provenance
  tests/                  pytest suite, 75 tests across ten modules
  frontend/
    pages/                Next.js pages (index, admin, API routes)
    lib/                  Embed + rerank + safety + DB helpers
    styles/               CSS
    demo/                 Stand-alone integration demos
    *.json                Corpus, knowledge base, Welsh maps
  uwtsd_scraper/          Scrapy spider for the institutional corpus
  benchmark_results/      Evaluation evidence, one folder per report section
                          (see benchmark_results/README.md)
  docs/
    PROJECT_REFERENCE.md
    SCRAPY_GUIDE.md
    report_drafts/        Dissertation chapter drafts
  tools/                  Optional dataset-fetch launchers (Windows)
  run.py                  Windows-safe backend launcher
  start.bat               Start the backend
  start-everything.bat    Backend + frontend + tunnel
  ingest.bat              Rebuild the Chroma index
  requirements.txt
  .env.example
  README.md
```

---

## Reproducing the evaluation

Results land in `benchmark_results/`, one folder per report section. Retrieval
needs no running server; the rest post to a live backend, so start it first
with `python run.py`.

```bash
# F5, retrieval configurations - no server needed
python scripts/benchmark_retrieval.py

# N1, latency. --tag ollama writes the fallback run to separate files
python scripts/benchmark_latency.py
python scripts/benchmark_latency.py --tag ollama

# F2, language detection. The sample is drawn once, labelled by hand,
# then reused by both the detection and source-attribution runs
python scripts/make_langdetect_sample.py
python scripts/benchmark_f2_endtoend.py

# F8, source attribution
python scripts/benchmark_sources.py

# F7, prompt-injection resistance
python scripts/benchmark_injection.py

# Usability, reads the questionnaire export
python scripts/analyse_sus.py
```

Two of these need a step that cannot be automated. `make_langdetect_sample.py`
writes a sheet whose ground-truth column is deliberately left blank, because
labelling it with a rule that keys on Welsh function words would replicate the
logic of the detector under test. `benchmark_sources.py` writes a relevance
rating sheet for the same reason: relevance is a judgement, not a property of
the system.

`benchmark_injection.py` classifies outcomes automatically as a first pass and
flags every response for review. Layer-1 blocks and verbatim prompt leaks are
detected exactly; the remaining labels are heuristic and were confirmed by hand
before being reported.

Run the test suite with:

```bash
python -m pytest tests/ -q
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
tools\fetch-openorca.bat            # Windows one-click
# or
python -m scripts.fetch_openorca --sample 20000

# general-knowledge Q&A from Google Natural Questions
tools\fetch-naturalquestions.bat    # Windows one-click
# or
python -m scripts.fetch_naturalquestions --sample 30000

# Welsh chat pairs from nemotron-chat-welsh, improves Welsh fluency
tools\fetch-welsh-chat.bat          # Windows one-click
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
