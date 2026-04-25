# U-Pal — Bilingual Student Assistant Chatbot

U-Pal is a bilingual Welsh and English conversational chatbot built for UWTSD (University of Wales Trinity Saint David). It answers common student queries about admissions, courses, fees, accommodation, IT support, campus locations, wellbeing, and more.

The project was developed as part of a dissertation investigating how a university chatbot can be designed to support both Welsh and English-speaking students without relying on a cloud-based AI API. The NLP pipeline is built entirely on open-source libraries, with optional local LLM enhancement via Ollama.

---

## Architecture

The system follows a retrieval-augmented generation (RAG) pattern. Responses are grounded in three curated tiers, in this order of preference, before any generative step runs:

1. **Curated facts** — `uwtsd-facts.json`: 25 hand-written, qualifier-tagged facts (e.g. `fees-international-postgraduate`, `graduation`, `wellbeing-crisis`, `financial-hardship`) with bilingual questions/answers. Highest authority.
2. **Intent library** — `knowledge.json`: 55 bilingual intents (34,000+ trigger phrases) covering broad student topics, scored with TF-IDF + Naive Bayes.
3. **UWTSD corpus** — 699 passages pre-extracted from the UWTSD website and indexed offline (via Morphik during the build). Hybrid retrieval: **TF-IDF + dense embeddings (nomic-embed-text)** fused with Reciprocal Rank Fusion (RRF), then reranked by term-overlap (or optionally an LLM-as-judge).

The LLM (Ollama/Groq) is never asked to invent facts — it rephrases passages retrieved from tier 1, 2, or 3 so the answer sounds conversational while staying grounded. A post-generation grounding check measures token-overlap between the answer and the retrieved context and down-weights the response confidence when it drops too low.

```
User message
    │
    ▼
Language detection          Welsh word-set matching (19,602-word vocab from TermCymru)
    │
    ▼
Self-disclosed context      extractStudentContext() — name, level, year, campus, type
    │                       (fed into the LLM system prompt as STUDENT CONTEXT block)
    ▼
Curated fact lookup         Score against facts.questions + keywords + qualifiers
    │                       Bilingual FACT_STOPWORDS filter applied to query & facts
    │                       Match found → use fact answer (+ optional LLM rephrase)
    ▼
Intent classifier           TF-IDF + Naive Bayes over 55 intents
    │                       score ≥ 0.18 → matched intent response
    │                       0.05 ≤ score < 0.18 → clarification prompt
    │                       score < 0.05 → escalate to RAG
    ▼
Hybrid RAG retrieval        Lexical TF-IDF   ┐
    │                                        ├─→ RRF fusion (k=60) → top-k candidates
    │                       Dense cosine sim ┘    │
    │                       Ollama /api/embeddings    │
    │                       nomic-embed-text (768d)   ▼
    │                                            Rerank (term-overlap IDF
    │                                              + optional LLM judge)
    ▼
Safety sanitation           PII scrubber (NI, card, phone, email, student-id)
    │                       Injection filter drops chunks with "ignore previous…"
    ▼
LLM generation              Ollama or Groq, adaptive temperature
    │                       (0.15 factual · 0.30 neutral · 0.55 empathetic)
    ▼
Grounding check             Token-overlap between answer and retrieved passages.
    │                       Low ratio (<0.15) → confidence clamped to 0.45.
    ▼
Welsh safety-net            hasRepetition() + looksWelsh() checks. On fail, re-route
                            through English and translate via MyMemory (chunked)
```

Both Welsh and English responses are returned for every answer. A single recognised Welsh word in the user's message is enough to flip the response language — no manual toggle is needed.

### Hybrid retrieval (TF-IDF ⊕ dense embeddings)

The offline UWTSD corpus used to be indexed with pure TF-IDF, which fails on paraphrase (e.g. "money for uni" vs "tuition fees"). The corpus is now indexed twice:

- **Lexical (natural.js TF-IDF)** — built at module load over preprocessed tokens; nails exact-match queries, postcodes, course codes.
- **Dense (nomic-embed-text, 768-dim)** — pre-computed offline by `scripts/build-embeddings.js` and stored in `uwtsd-corpus-embeddings.json`; captures paraphrase and cross-lingual similarity.

At query time both retrievers return a ranked list over the same 699 passages; the lists are combined with **Reciprocal Rank Fusion** (Cormack et al., SIGIR 2009), `score = Σ 1/(60 + rank)`, and the top-10 fused candidates are re-ranked by a cheap term-overlap × IDF signal. Set `LLM_RERANK=1` to additionally run a Groq-based LLM judge over the top-8 (adds ~400 ms).

If the embedding sidecar is missing **or** Ollama is unreachable at query time, the pipeline degrades gracefully to lexical-only — no new dependency is ever required for the fallback to work.

---

## Tech stack

| Layer | Technology |
|---|---|
| Framework | Next.js 14 (pages router), React 18 |
| NLP | natural.js — TfIdf, BayesClassifier, PorterStemmer, WordTokenizer |
| Intent library | knowledge.json — 55 bilingual intents, 34k+ trigger patterns |
| Curated facts | uwtsd-facts.json — 25 qualifier-tagged bilingual facts |
| Corpus retrieval | **Hybrid TF-IDF + dense embeddings** over 699 UWTSD passages, fused with RRF, reranked by term-overlap (optional LLM-as-judge) |
| Embeddings | Ollama `/api/embeddings` + nomic-embed-text (768-d) — pre-computed offline, compatible with Morphik's semantic space |
| Safety layer | PII scrubber (NI, card, phone, email, student-ID) · injection filter · off-topic gate · grounding check |
| Chunker | Sentence-boundary windows (~1200 chars) with 250-char overlap, written into `scripts/harvest-morphik-corpus.py` |
| Language detection | Custom WELSH_WORDS/WELSH_STRONG + 43k-term TermCymru vocab |
| NER and sentiment | Stanford CoreNLP 4.5 (optional, local, via ngrok tunnel) |
| LLM enhancement | Ollama (Llama 3.1 8B) or Groq API — **adaptive temperature (0.15 factual · 0.55 empathetic)**, repeat-penalty tuned |
| Translation | MyMemory API — chunked for long replies to avoid truncation |
| Safety-net | hasRepetition() loop-detector + looksWelsh() guard with pivot-through-English fallback |
| Feedback storage | MongoDB Atlas (falls back to a local JSON file if not configured) |
| Deployment | Vercel (serverless functions) |

---

## Project structure

```
pages/
  index.js          Main chatbot UI — consent modal, chat, how it works, docs tabs
  admin.js          Admin feedback dashboard — login, stats, paginated table, CSV export
  api/
    chat.js         POST /api/chat — runs the NLP pipeline
    feedback.js     POST saves a rating; GET returns all (admin only)
    health.js       GET /api/health — checks Ollama and CoreNLP connectivity
    logout.js       GET /api/logout — clears browser Basic Auth cache
lib/
  nlp.js            NLP pipeline — language detection, TF-IDF, Bayes, facts, RAG, LLM, safety-net
  embed.js          Ollama embedding client + in-memory vector store (cosineTopK)
  safety.js         PII scrubber, injection filter, off-topic gate, grounding check
  rerank.js         Local term-overlap reranker + optional Groq LLM-as-judge
  db.js             MongoDB singleton with file fallback for serverless
scripts/
  analyze-feedback.js      Aggregate ratings into failure-mode insights
  audit-intents.js         Lint knowledge.json + uwtsd-facts.json against R1–R5
  dedupe-patterns.js       Remove duplicate patterns with curated specificity rules
  diversify-patterns.js    Add typos/slang/Welsh variants to top intents
  build-welsh-terms.js     Pull TermCymru CSV → welsh-terms.json
  build-embeddings.js      Pre-compute dense vectors for the corpus via Ollama
  harvest-morphik-corpus.py  Sentence-boundary chunker + Morphik seed harvester
styles/
  globals.css       Design system — CSS custom properties, layout, components, responsive
knowledge.json                55 bilingual intents (34k+ trigger patterns)
uwtsd-facts.json              25 curated, qualifier-tagged bilingual facts (highest-authority tier)
uwtsd-corpus.json             699 sentence-boundary chunks harvested from UWTSD pages
uwtsd-corpus-embeddings.json  Sidecar: 699 × 768-d dense vectors (built by scripts/build-embeddings.js)
welsh-terms.json              Additional Welsh vocabulary from TermCymru
demo-setup.bat      One-click local demo launcher (Ollama + CoreNLP + ngrok tunnels)
```

---

## Local setup

### Requirements

- Node.js 18 or later
- A MongoDB Atlas cluster (optional — feedback is saved to a local file without it)
- Java 11 or later (optional — for Stanford CoreNLP)
- Ollama installed (optional — for LLM rephrasing)

### Install and run

```bash
git clone https://github.com/Ataur-Web/U-Pal-RAG.git
cd U-Pal-RAG
npm install
cp .env.example .env
npm run dev
```

Open http://localhost:3000.

### Environment variables

Copy `.env.example` to `.env` and fill in the values you need.

| Variable | Required | Description |
|---|---|---|
| `ADMIN_PASSWORD` | Recommended | Password for the admin dashboard at `/admin` |
| `MONGODB_URI` | Optional | MongoDB Atlas connection string. Without this, feedback is saved to `feedback.json` |
| `OLLAMA_URL` | Optional | Public URL of your Ollama tunnel (e.g. from ngrok). Enables LLM rephrasing |
| `OLLAMA_MODEL` | Optional | Model name to use. Defaults to `llama3.1:8b` |
| `CORENLP_URL` | Optional | Public URL of your Stanford CoreNLP tunnel. Enables NER and sentiment scoring |

---

## Running the full demo locally (Ollama + CoreNLP)

`demo-setup.bat` is a one-click Windows launcher that starts all four services:

1. Ollama (model server on port 11434)
2. Stanford CoreNLP (Java server on port 9000)
3. ngrok tunnel for Ollama
4. ngrok tunnel for CoreNLP (using a free static domain so the URL never changes)

Before running it, open the file and fill in your ngrok authtoken and your free static ngrok domain. Then copy the Ollama tunnel URL from the ngrok dashboard and set it as `OLLAMA_URL` in your Vercel environment variables.

Download Stanford CoreNLP from https://stanfordnlp.github.io/CoreNLP/ and extract it to `%USERPROFILE%\Downloads\stanford-corenlp-4.5.10\stanford-corenlp-4.5.10`.

---

## Adding intents

Edit `knowledge.json`. Each intent needs a unique tag, a set of training patterns (mix of English and Welsh phrases), and bilingual response arrays.

```json
{
  "tag": "example_intent",
  "patterns": [
    "English example question",
    "Another English phrase",
    "Welsh example phrase"
  ],
  "responses": {
    "en": ["English answer one", "English answer two"],
    "cy": ["Welsh answer one", "Welsh answer two"]
  }
}
```

Aim for at least 10 to 14 patterns per intent. After editing, redeploy to Vercel — the TF-IDF index and Naive Bayes classifier retrain automatically on startup.

---

## Data curation workflow

The chatbot's answer quality depends on keeping the three curated tiers clean: no duplicate patterns confusing the classifier, enough Welsh coverage for bilingual parity, and enough real-user phrasing variants so the Naive Bayes classifier generalises. The `scripts/` directory contains four tools that automate this loop:

```
feedback.json ──▶ analyze-feedback.js ──▶ identify weak intents / failure keywords
                                              │
                                              ▼
                                      edit knowledge.json / uwtsd-facts.json
                                              │
                                              ▼
                                       audit-intents.js  (R1–R5 lint)
                                              │
                                              ├─▶ fix R3 / R4 violations by hand
                                              ▼
                                       dedupe-patterns.js  (resolve R5 with KEEP_IN rules)
                                              │
                                              ▼
                                    diversify-patterns.js  (add typos, slang, Welsh variants)
                                              │
                                              ▼
                                       redeploy → classifier retrains
```

### `scripts/analyze-feedback.js`
Aggregates the 5-point rating stream into: overall satisfaction, helpfulness %, Welsh-vs-English quality split, keyword-clustered low-rating comments (failure modes), and daily volume. Usage: `node scripts/analyze-feedback.js [--mongo | --json path] [--since N_DAYS]`.

### `scripts/audit-intents.js`
Enforces five design rules adapted from Dialogflow/Rasa intent-design guidance:

| Rule | Check |
|---|---|
| R1 | ≥ 3 training patterns per intent, ≥ 2 responses per language |
| R2 | ≤ 40 patterns per intent (otherwise split into sub-intents + entities) |
| R3 | Every fact has a Welsh question and a Welsh answer |
| R4 | Qualifier-tagged fees facts declare valid qualifiers from the whitelist |
| R5 | No duplicate trigger pattern across two intents |

Exits non-zero on any violation — wire it into CI before deploys.

### `scripts/dedupe-patterns.js`
Resolves R5 violations. Duplicate patterns across intents flatten the Naive Bayes posterior and blur the decision boundary. A hand-curated `KEEP_IN` map picks the more-specific intent per ambiguous pair (e.g. "feeling hopeless" stays in `wellbeing_crisis`, not `wellbeing_general`); anything unmapped falls back to first-seen order. Run with `--dry` to preview.

### `scripts/diversify-patterns.js`
Adds realistic variants — typos (`libary`, `aplly`), txt-speak (`hw much is uni`, `reset pword`), casual phrasing (`hiya how do i apply`), and Welsh alternatives — to the highest-traffic intents. Curated rather than auto-paraphrased so quality stays high.

### The intent vs entity principle (R2)

Large flat intents (> 40 patterns) almost always mix several distinct user problems. When R2 fires, the right fix is usually to promote the varying noun into an entity rather than splitting into many `X_for_Y` intents. For example, one `fees_tuition` intent with a `studentType` entity (`home`/`international`/`postgraduate`) beats three `fees_home`, `fees_international`, `fees_postgraduate` intents — the qualifier routing in `uwtsd-facts.json` already demonstrates this pattern.

---

## RAG pipeline operations

### Re-chunk & re-embed the corpus

When UWTSD publishes new content or the Morphik index is rebuilt, refresh the offline tier in two steps:

```bash
# 1. Harvest chunks (runs on the VM where Morphik is reachable)
python3 scripts/harvest-morphik-corpus.py
#    → writes uwtsd-corpus.json (sentence-boundary windows + metadata)

# 2. Re-embed changed passages (runs anywhere Ollama is available)
OLLAMA_URL=http://localhost:11434 node scripts/build-embeddings.js
#    → writes uwtsd-corpus-embeddings.json (768-d sidecar)
#    content-hash cache means unchanged passages are NOT re-embedded
```

The chunker targets ~1200-char windows with 250-char overlap. Set `CHUNK_MODE=raw` to fall back to the legacy single-window behaviour for backwards-compatibility with older sidecars.

### Runtime diagnostics

The new RAG surfaces expose diagnostic env vars so you can observe each layer without code changes:

| Variable | Effect |
|---|---|
| `DEBUG_EMBED=1`    | Log embedding call failures (timeouts, Ollama 5xx)         |
| `DEBUG_SAFETY=1`   | Log PII scrubs / dropped injection chunks                   |
| `DEBUG_GROUNDING=1`| Log answer-vs-context token-overlap ratio                   |
| `DEBUG_RERANK=1`   | Log local + LLM-judge rerank decisions                      |
| `LLM_RERANK=1`     | Enable the Groq LLM-as-judge reranker (off by default)      |

The corpus/vector diagnostics are also exposed programmatically via `nlp.embedStatus()` for `/api/health` integration.

### Why this stack, not OpenAI / Voyage / transformers.js

- **Ollama + `nomic-embed-text`** was chosen because it is already part of the LLM tier and — crucially — it's the same model Morphik uses, so our offline vectors sit in the same semantic space as the live Morphik index. This keeps the three tiers coherent.
- **`natural.js` TF-IDF** is kept for lexical retrieval so the system has a zero-dependency fallback when Ollama is down. Hybrid retrieval degrades to lexical-only automatically.
- **RRF** was chosen over learned-to-rank fusion because it needs no calibration across scoring systems and has a single well-understood knob (`k=60`).
- **No transformers.js in the runtime** — a 25 MB model download on every cold-start in Vercel's serverless environment was too costly for the latency budget. Pre-computed vectors + cheap cosine math wins on both dimensions.

---

## Admin dashboard

The admin dashboard is available at `/admin`. It requires the password set in `ADMIN_PASSWORD`. Enter any username and the correct password. The dashboard shows summary statistics, a paginated feedback table, and a CSV export button.

---

## Deployment

The project deploys to Vercel automatically on push to `main`. Set environment variables in Vercel → Project → Settings → Environment Variables.

---

## Academic references

This project is informed by the following works:

- Inuwa-Dutse, I. (2023). *Simplifying Student Queries: A Dialogflow-Based Conversational Chatbot for University Websites.* The paper describes a university chatbot architecture using intent classification and a curated knowledge base — the design this project adapts for a bilingual Welsh/English context.

- Manning, C. D., Surdeanu, M., Bauer, J., Finkel, J., Bethard, S. J., & McClosky, D. (2014). *The Stanford CoreNLP Natural Language Processing Toolkit.* Proceedings of ACL 2014 System Demonstrations, pp. 55–60. Used for named entity recognition, part-of-speech tagging, and sentiment scoring.

- Natural.js library: https://naturalnode.github.io/natural/ — provides the TfIdf vectoriser, BayesClassifier, PorterStemmer, and WordTokenizer used in the NLP pipeline.

- Ollama: https://ollama.com — local LLM inference engine used to run Llama 3.1 8B for optional response rephrasing.

- MyMemory Translation API: https://mymemory.translated.net — used to generate Welsh alt-responses when the Ollama path is active for English queries.

- BydTermCymru / TermCymru: Welsh technical terminology dataset (cwd24/termcymru, Crown copyright). The `scripts/build-welsh-terms.js` script pulls this CSV and extracts Welsh word forms into `welsh-terms.json`, which is merged into the language-detection vocabulary set at startup to improve Welsh detection accuracy.

---

## Licence

This project was built as part of an academic dissertation at UWTSD. It is shared openly for reference and learning purposes.
