#!/usr/bin/env node
'use strict';
/**
 * build-embeddings.js — pre-compute dense embeddings for the UWTSD corpus.
 *
 * Reads  uwtsd-corpus.json                 (699 passages, ~1 MB)
 * Writes uwtsd-corpus-embeddings.json      (vectors + ids, ~4 MB)
 *
 * Uses Ollama's /api/embeddings with nomic-embed-text (768-d) — same model
 * Morphik uses so our offline vectors live in the same semantic space.
 *
 * Hash-caching: each passage has a content hash. If the sidecar exists and
 * the content hasn't changed, we reuse the old vector instead of re-embedding.
 * Lets you re-run cheaply after scripts/harvest-morphik-corpus.py adds new
 * passages — only the new ones pay embedding cost.
 *
 * Usage:
 *   OLLAMA_URL=http://localhost:11434 node scripts/build-embeddings.js
 *
 * Env vars:
 *   OLLAMA_URL    required
 *   EMBED_MODEL   default nomic-embed-text
 *   EMBED_BATCH   default 1 (Ollama /api/embeddings is single-prompt; we
 *                 keep concurrency low to not flood the VM)
 */

const fs      = require('fs');
const path    = require('path');
const crypto  = require('crypto');

const ROOT        = path.resolve(__dirname, '..');
const CORPUS_IN   = path.join(ROOT, 'uwtsd-corpus.json');
const EMBED_OUT   = path.join(ROOT, 'uwtsd-corpus-embeddings.json');

const OLLAMA_URL  = (process.env.OLLAMA_URL || '').replace(/\/$/, '');
const EMBED_MODEL = process.env.EMBED_MODEL || 'nomic-embed-text';
const CONCURRENCY = Math.max(1, Number(process.env.EMBED_BATCH || 1));
const TIMEOUT_MS  = Number(process.env.EMBED_TIMEOUT_MS || 30000);

if (!OLLAMA_URL) {
  console.error('ERROR: OLLAMA_URL is required');
  console.error('  export OLLAMA_URL=http://localhost:11434');
  process.exit(1);
}

function hashContent(s) {
  return crypto.createHash('sha1').update(s).digest('hex').slice(0, 16);
}

async function embedOne(text) {
  const ctrl = new AbortController();
  const to   = setTimeout(() => ctrl.abort(), TIMEOUT_MS);
  try {
    const res = await fetch(`${OLLAMA_URL}/api/embeddings`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ model: EMBED_MODEL, prompt: text }),
      signal:  ctrl.signal,
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const json = await res.json();
    if (!Array.isArray(json.embedding)) throw new Error('no embedding in response');
    return json.embedding;
  } finally {
    clearTimeout(to);
  }
}

async function runPool(items, worker, concurrency) {
  const out = new Array(items.length);
  let next = 0, active = 0, done = 0;
  return new Promise((resolve, reject) => {
    const tick = () => {
      while (active < concurrency && next < items.length) {
        const idx = next++;
        active++;
        worker(items[idx], idx)
          .then(r => { out[idx] = r; })
          .catch(e => { out[idx] = { __error: e.message }; })
          .finally(() => {
            active--; done++;
            if (done % 25 === 0 || done === items.length) {
              process.stdout.write(`\r  embedded ${done}/${items.length}`);
            }
            if (done === items.length) { process.stdout.write('\n'); resolve(out); }
            else tick();
          });
      }
    };
    tick();
  });
}

async function main() {
  console.log('='.repeat(60));
  console.log('  UWTSD corpus → dense embeddings');
  console.log('='.repeat(60));
  console.log(`  Ollama:  ${OLLAMA_URL}`);
  console.log(`  Model:   ${EMBED_MODEL}`);
  console.log(`  Input:   ${CORPUS_IN}`);
  console.log(`  Output:  ${EMBED_OUT}`);
  console.log();

  const corpus = JSON.parse(fs.readFileSync(CORPUS_IN, 'utf8'));
  console.log(`  Passages: ${corpus.length}`);

  // Load existing sidecar for cache reuse.
  let cache = new Map(); // id → { hash, vec }
  if (fs.existsSync(EMBED_OUT)) {
    try {
      const prev = JSON.parse(fs.readFileSync(EMBED_OUT, 'utf8'));
      const ids    = prev.ids    || [];
      const hashes = prev.hashes || [];
      const vecs   = prev.vectors || [];
      for (let i = 0; i < ids.length; i++) {
        if (vecs[i]) cache.set(ids[i], { hash: hashes[i], vec: vecs[i] });
      }
      console.log(`  Cache:    ${cache.size} existing vectors (will reuse where content unchanged)`);
    } catch (e) {
      console.warn(`  Cache:    failed to load (${e.message}) — rebuilding all`);
    }
  }
  console.log();

  // Probe: confirm Ollama is alive with a small embed call.
  try {
    await embedOne('ping');
  } catch (e) {
    console.error(`FATAL: Ollama embed probe failed: ${e.message}`);
    console.error('  Make sure `ollama pull nomic-embed-text` has been run on the VM.');
    process.exit(1);
  }

  // Build work list: (passage, needsEmbed)
  const work = corpus.map(p => {
    const txt  = [
      (Array.isArray(p.topics) && p.topics.join(' ')) || '',
      p.content || '',
    ].join('\n').trim();
    const hash = hashContent(txt);
    const cached = cache.get(p.id);
    return { passage: p, text: txt, hash, cached };
  });

  const toEmbed = work.filter(w => !w.cached || w.cached.hash !== w.hash);
  console.log(`  Embedding ${toEmbed.length} passages (${work.length - toEmbed.length} cached).\n`);

  const startTs = Date.now();
  await runPool(toEmbed, async (w) => {
    const vec = await embedOne(w.text);
    w.vec = vec;
    return vec;
  }, CONCURRENCY);
  const elapsed = ((Date.now() - startTs) / 1000).toFixed(1);
  console.log(`  Completed in ${elapsed}s.\n`);

  // Assemble final sidecar in passage order.
  const ids = [], langs = [], vectors = [], hashes = [];
  let errors = 0, dim = 0;
  for (const w of work) {
    let vec = w.vec;
    if (!vec && w.cached && w.cached.hash === w.hash) vec = w.cached.vec;
    if (!Array.isArray(vec) || !vec.length) { errors++; continue; }
    dim = vec.length;
    ids.push(w.passage.id);
    langs.push(w.passage.lang === 'cy' ? 'cy' : 'en');
    hashes.push(w.hash);
    vectors.push(vec);
  }

  const sidecar = {
    model:    EMBED_MODEL,
    dim,
    count:    ids.length,
    built_at: new Date().toISOString(),
    ids,
    langs,
    hashes,
    vectors,
  };
  fs.writeFileSync(EMBED_OUT, JSON.stringify(sidecar));

  console.log('='.repeat(60));
  console.log(`  Wrote ${ids.length} vectors (dim ${dim})`);
  if (errors) console.log(`  WARNING: ${errors} passages failed to embed`);
  console.log(`  File:  ${EMBED_OUT}`);
  console.log('='.repeat(60));
}

main().catch(e => {
  console.error('FATAL:', e.stack || e.message);
  process.exit(1);
});
