# U-Pal frontend

This folder is the Next.js 14 frontend that gets deployed to Vercel. It
talks to the FastAPI Python backend (in `../app/`) over a public URL
that the operator sets in `CHAT_BACKEND_URL`.

The chat UI, the admin dashboard and the docs page all live here. Most
of the heavy work (retrieval, language detection, LLM call) happens on
the Python side, this folder is mostly presentation plus a thin
fallback that takes over if the Python backend is unreachable.

## What lives where

```
frontend/
  pages/
    index.js          chat UI, consent modal, theming, docs page
    admin.js          feedback dashboard, basic-auth gated
    api/
      chat.js         POST /api/chat, proxies to the Python backend
                      and falls back to the legacy node pipeline
                      if the backend is down
      feedback.js     POST saves a rating, GET returns all (admin)
      health.js       GET, checks backend connectivity
      llm-config.js   GET/POST runtime LLM provider switch
      logout.js       GET, clears the basic-auth credential cache
      translate.js    POST, lazy translate of a bot reply
  lib/
    nlp.js            legacy node TF-IDF + Bayes pipeline, used as a
                      fallback only when the Python backend is offline
    embed.js          embedding helper for the legacy fallback
    safety.js         PII scrub + injection filter for the fallback
    rerank.js         reranker for the fallback
    db.js             MongoDB singleton with a local file fallback
  styles/
    globals.css       design tokens, layout, components, responsive
  scripts/            optional build helpers for the legacy fallback
                      (audit intents, dedupe patterns, etc)

  knowledge.json
  uwtsd-corpus.json
  uwtsd-facts.json
  welsh-bilingual-map.json
  welsh-terms.json    JSON data files used by the legacy node fallback
```

## Two backends, one frontend

The chat endpoint picks where to send each request:

1. If `CHAT_BACKEND_URL` is set, the request is forwarded to the
   Python backend. The Python backend runs the proper hybrid retrieval
   plus Claude / Ollama path.
2. If the Python backend returns nothing or the env var is missing,
   the request falls through to the legacy node pipeline in `lib/nlp.js`.
   That path uses TF-IDF + Naive Bayes against the JSON files in this
   folder. It exists so the live demo never goes down even if the
   tunnel breaks.

For the dissertation work the Python path is the primary system. The
node fallback is kept around for resilience.

## Running the frontend locally

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

Open http://localhost:3000.

## Environment variables

| Variable | Required | What it does |
|---|---|---|
| `ADMIN_PASSWORD` | for /admin | basic-auth password for the dashboard |
| `MONGODB_URI` | optional | Atlas connection, falls back to local JSON |
| `CHAT_BACKEND_URL` | yes for the Python path | public URL of the FastAPI backend |

## Deployment

The frontend deploys to Vercel automatically on push to `main`. Set
the env vars under Settings, Environment Variables. The backend has
to be reachable from Vercel, the simplest way is to expose it through
an ngrok static domain and put that URL in `CHAT_BACKEND_URL`.
