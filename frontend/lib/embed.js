'use strict';
// embed.js. thin embedding client + in-process vector store for the legacy
// Node NLP pipeline. the live pipeline uses ChromaDB on the Python backend,
// this file is only reached when the backend is unreachable and the Node
// fallback takes over.
//
// pipeline layout:
//   1. build-time  scripts/build-embeddings.js reads uwtsd-corpus.json,
//                  POSTs each passage to {OLLAMA_URL}/api/embeddings, and
//                  writes vectors to uwtsd-corpus-embeddings.json
//   2. run-time    embedQuery() hits the same endpoint for the user's
//                  message. returns null if Ollama is unreachable so the
//                  caller can degrade to lexical-only retrieval
//   3. ranking     cosineTopK() does plain cosine similarity in memory.
//                  the corpus is ~700 passages × 768 dims (around 2 MB) so
//                  we skip an ANN index and just do a linear scan
//
// why nomic-embed-text via Ollama? Ollama is already in the stack so no
// new dependency, and the model matches what the offline corpus sidecar
// was built with so vectors live in the same semantic space.
// ref: https://ollama.com/library/nomic-embed-text
// ref: https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings

const fs   = require('fs');
const path = require('path');

const OLLAMA_URL    = (process.env.OLLAMA_URL || '').replace(/\/$/, '');
const EMBED_MODEL   = process.env.EMBED_MODEL || 'nomic-embed-text';
const EMBED_TIMEOUT = Number(process.env.EMBED_TIMEOUT_MS || 5000);
const EMBED_DIM     = Number(process.env.EMBED_DIM || 768);

// module-level state. everything is lazy-loaded once then shared across
// requests (Next.js keeps the module alive between warm invocations on
// the same Lambda instance).
let CORPUS_VECTORS = null;           // Array<{ id, lang, vec: Float32Array }>
let VECTORS_LOADED_FROM = null;      // diagnostic string
let EMBED_ONLINE = null;             // tri-state: null=unknown, true, false

const EMBED_PATH = path.join(
  process.cwd(),
  'uwtsd-corpus-embeddings.json',
);

// sidecar file shape:
//   { model, dim, count, ids: [...], vectors: [[...768 floats...], ...] }
// we keep vectors as Float32Array so cosine math is cheap.
// ref: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Float32Array
function loadCorpusVectors() {
  if (CORPUS_VECTORS !== null) return CORPUS_VECTORS;
  try {
    if (!fs.existsSync(EMBED_PATH)) {
      CORPUS_VECTORS = [];
      return CORPUS_VECTORS;
    }
    const raw = JSON.parse(fs.readFileSync(EMBED_PATH, 'utf8'));
    const ids   = Array.isArray(raw.ids) ? raw.ids : [];
    const vs    = Array.isArray(raw.vectors) ? raw.vectors : [];
    const langs = Array.isArray(raw.langs) ? raw.langs : [];
    const out = [];
    for (let i = 0; i < ids.length; i++) {
      const v = vs[i];
      if (!v || !v.length) continue;
      out.push({
        id:   ids[i],
        lang: langs[i] || 'en',
        vec:  Float32Array.from(v),
      });
    }
    CORPUS_VECTORS = out;
    VECTORS_LOADED_FROM = `${EMBED_PATH} (${out.length} vectors, dim ${raw.dim || EMBED_DIM}, model ${raw.model || EMBED_MODEL})`;
    if (out.length) console.log(`[embed] Loaded ${out.length} corpus vectors from sidecar`);
    return CORPUS_VECTORS;
  } catch (e) {
    console.warn(`[embed] Failed to load ${EMBED_PATH}: ${e.message}`);
    CORPUS_VECTORS = [];
    return CORPUS_VECTORS;
  }
}

// query-time embedding. returns a plain number[] (768 floats) on success,
// null on any failure so callers can skip dense retrieval cleanly.
// ref: https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
async function embedQuery(text) {
  if (!OLLAMA_URL) { EMBED_ONLINE = false; return null; }
  if (!text || !text.trim()) return null;

  try {
    // AbortController enforces a timeout so a hung Ollama container cannot
    // block the API route.
    // ref: https://developer.mozilla.org/en-US/docs/Web/API/AbortController
    const ctrl = (typeof AbortController !== 'undefined') ? new AbortController() : null;
    const to   = ctrl ? setTimeout(() => ctrl.abort(), EMBED_TIMEOUT) : null;

    const res = await fetch(`${OLLAMA_URL}/api/embeddings`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ model: EMBED_MODEL, prompt: text }),
      signal:  ctrl ? ctrl.signal : undefined,
    });
    if (to) clearTimeout(to);

    if (!res.ok) {
      if (process.env.DEBUG_EMBED) console.warn(`[embed] HTTP ${res.status} from Ollama`);
      EMBED_ONLINE = false;
      return null;
    }
    const json = await res.json();
    const v = json && json.embedding;
    if (!Array.isArray(v) || !v.length) {
      EMBED_ONLINE = false;
      return null;
    }
    EMBED_ONLINE = true;
    return v;
  } catch (e) {
    if (process.env.DEBUG_EMBED) console.warn(`[embed] embedQuery failed: ${e.message}`);
    EMBED_ONLINE = false;
    return null;
  }
}

// cosine similarity. inputs are iterables of numbers (Array or Float32Array).
// returns a scalar in [-1, 1]. zero-norm edge cases fall through to 0.
// ref: https://en.wikipedia.org/wiki/Cosine_similarity
function cosine(a, b) {
  const n = Math.min(a.length, b.length);
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < n; i++) {
    const x = a[i], y = b[i];
    dot += x * y;
    na  += x * x;
    nb  += y * y;
  }
  if (!na || !nb) return 0;
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

// top-k corpus passage IDs by cosine similarity to queryVec. langFilter
// ('en' or 'cy') restricts candidates when we know the query language.
function cosineTopK(queryVec, k = 10, langFilter = null) {
  if (!queryVec || !queryVec.length) return [];
  const vectors = loadCorpusVectors();
  if (!vectors.length) return [];

  const scored = [];
  for (const entry of vectors) {
    if (langFilter && entry.lang !== langFilter) continue;
    const s = cosine(queryVec, entry.vec);
    if (s > 0) scored.push({ id: entry.id, score: s, lang: entry.lang });
  }
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, k);
}

// diagnostic status, surfaced via /api/health for debugging
function embedStatus() {
  const vectors = loadCorpusVectors();
  return {
    enabled:        Boolean(OLLAMA_URL),
    online:         EMBED_ONLINE,
    model:          EMBED_MODEL,
    dim:            EMBED_DIM,
    corpus_vectors: vectors.length,
    sidecar_path:   fs.existsSync(EMBED_PATH) ? EMBED_PATH : null,
    loaded_from:    VECTORS_LOADED_FROM,
  };
}

module.exports = {
  embedQuery,
  cosineTopK,
  cosine,
  loadCorpusVectors,
  embedStatus,
};
