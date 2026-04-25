'use strict';
// rerank.js. re-scores the fused retrieval candidates produced by RRF so
// the top passages going into the LLM are the ones that actually share
// content words with the query.
//
// we have two rerankers:
//   1. localRerank  cheap term-overlap weighted by IDF. runs every time
//   2. llmRerank    optional Groq call using a small judge model. only
//                   active when LLM_RERANK=1 (adds around 400 ms)
//
// a proper cross-encoder (ms-marco-MiniLM) would be better but that needs
// transformers.js or a Python service, so we approximate with these two
// signals instead.
// ref: Nogueira & Cho 2019, "Passage Re-ranking with BERT",
//      https://arxiv.org/abs/1901.04085
// ref: Robertson & Zaragoza 2009, "The Probabilistic Relevance Framework:
//      BM25 and Beyond", https://doi.org/10.1561/1500000019

// -----------------------------------------------------------------------
// 1. cheap local reranker
// -----------------------------------------------------------------------
const STOP = new Set([
  'the','a','an','is','are','was','were','be','been','have','has','had','do','does','did',
  'to','of','in','on','at','for','with','by','from','and','or','but','so','if',
  'i','you','he','she','it','we','they','me','my','your','his','her','our','their',
  'what','which','who','when','where','why','how','this','that','these','those',
  'can','could','would','should','may','might','will','shall','must','about',
  'am','pm','mae','oes','sut','beth','ble','pryd','pam','faint','ydy','yw','dw','rwy',
]);

function tok(s) {
  return String(s || '')
    .toLowerCase()
    .replace(/[^a-zA-ZÀ-ÿŵŷ0-9]+/g, ' ')
    .split(/\s+/)
    .filter(t => t.length >= 3 && !STOP.has(t));
}

// rerank candidates by weighted term overlap combined multiplicatively
// with the fused RRF score. passages that scored well in RRF AND share
// real content words with the query float to the top.
function localRerank(query, passages, k = 5) {
  if (!passages || !passages.length) return [];
  const qTerms = new Set(tok(query));
  if (!qTerms.size) return passages.slice(0, k);

  // compute document frequency across just the candidate set. small-N IDF
  // is enough to down-weight common terms like "university" that appear
  // in every passage. ref: Spärck Jones 1972, "A Statistical Interpretation
  // of Term Specificity and its Application in Retrieval"
  const df = new Map();
  const passageTokens = passages.map(p => {
    const ts = tok(p.content);
    const seen = new Set();
    for (const t of ts) {
      if (!seen.has(t)) { df.set(t, (df.get(t) || 0) + 1); seen.add(t); }
    }
    return ts;
  });

  const N = passages.length;
  const idf = t => Math.log(1 + N / (1 + (df.get(t) || 0)));

  const scored = passages.map((p, i) => {
    const ts = passageTokens[i];
    // for each unique query term present in the passage, add its IDF
    const presentQTerms = new Set();
    for (const t of ts) if (qTerms.has(t) && !presentQTerms.has(t)) presentQTerms.add(t);
    let overlap = 0;
    for (const t of presentQTerms) overlap += idf(t);
    // coverage: fraction of query terms the passage actually hits
    const coverage = presentQTerms.size / qTerms.size;
    const fused    = p.score || 0;
    // length-norm dampens very long passages that accidentally contain
    // the query terms without being specific about them. ref: Singhal et
    // al. 1996, "Pivoted Document Length Normalization"
    const lengthNorm = 1 / (1 + Math.log(1 + ts.length));
    const rerank_score = (1 + 2 * fused) * (1 + coverage) * overlap * lengthNorm;
    return { ...p, rerank_score, rerank_coverage: coverage };
  });

  scored.sort((a, b) => b.rerank_score - a.rerank_score);
  return scored.slice(0, k);
}

// -----------------------------------------------------------------------
// 2. LLM-as-judge (optional)
// -----------------------------------------------------------------------
// asks Groq to rate each passage on a [0,1] relevance scale. we only
// judge the top 8 from the local reranker to keep the token cost down.
// ref: Zheng et al. 2023, "Judging LLM-as-a-Judge",
//      https://arxiv.org/abs/2306.05685
const GROQ_URL      = 'https://api.groq.com/openai/v1/chat/completions';
const GROQ_API_KEY  = process.env.GROQ_API_KEY  || '';
const GROQ_MODEL    = process.env.GROQ_MODEL    || 'llama-3.1-8b-instant';
const LLM_RERANK_ON = process.env.LLM_RERANK === '1' && Boolean(GROQ_API_KEY);

async function llmRerank(query, passages, k = 5) {
  const local = localRerank(query, passages, Math.max(k, 8));
  if (!LLM_RERANK_ON || local.length <= 1) return local.slice(0, k);

  const candidates = local.slice(0, 8);
  const payload = candidates.map((p, i) => ({
    idx:     i,
    excerpt: String(p.content || '').slice(0, 400),
  }));

  // strict JSON schema in the system prompt so the response parses cleanly
  const sys = 'You rate how well each passage answers the user question. ' +
              'Return ONLY valid JSON of the form {"scores": [{"idx": n, "score": 0.0-1.0}, ...]}, one entry per passage, same length as input.';
  const user = `Question: ${String(query).slice(0, 400)}\n\nPassages:\n` +
               payload.map(p => `[${p.idx}] ${p.excerpt}`).join('\n\n');

  try {
    // 4s timeout. Groq is usually sub-second but we do not want to
    // block the API route on a slow judge call.
    // ref: https://developer.mozilla.org/en-US/docs/Web/API/AbortController
    const ctrl = new AbortController();
    const to   = setTimeout(() => ctrl.abort(), 4000);
    const res = await fetch(GROQ_URL, {
      method:  'POST',
      headers: {
        'Authorization': `Bearer ${GROQ_API_KEY}`,
        'Content-Type':  'application/json',
      },
      body: JSON.stringify({
        model:           GROQ_MODEL,
        temperature:     0,
        max_tokens:      220,
        // Groq supports OpenAI's JSON mode, which guarantees parseable output
        // ref: https://console.groq.com/docs/text-chat#json-mode-object-object
        response_format: { type: 'json_object' },
        messages: [
          { role: 'system', content: sys },
          { role: 'user',   content: user },
        ],
      }),
      signal: ctrl.signal,
    });
    clearTimeout(to);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const j = await res.json();
    const txt = j && j.choices && j.choices[0] && j.choices[0].message && j.choices[0].message.content;
    const parsed = JSON.parse(txt);
    const byIdx = new Map((parsed.scores || []).map(s => [s.idx, Number(s.score) || 0]));

    // blend the judge score with the local rerank score so a passage that
    // both judge and local agree on wins cleanly
    const rescored = candidates.map((p, i) => ({
      ...p,
      rerank_score:  (byIdx.get(i) ?? 0) * (1 + (p.rerank_score || 0) / 10),
      llm_score:     byIdx.get(i) ?? null,
    }));
    rescored.sort((a, b) => b.rerank_score - a.rerank_score);
    return rescored.slice(0, k);
  } catch (e) {
    if (process.env.DEBUG_RERANK) console.warn(`[rerank] LLM judge failed: ${e.message}, using local rerank`);
    return local.slice(0, k);
  }
}

module.exports = { localRerank, llmRerank };
