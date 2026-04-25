'use strict';
// GET /api/health. live connectivity check used by the CONNECTION card.
//
// primary path: proxy to the Python backend via CHAT_BACKEND_URL. that
// backend knows the real state of Claude, Ollama, Chroma, and the Welsh
// detector. if the tunnel is down we return a degraded response so the
// frontend still renders something.
// ref: https://nextjs.org/docs/pages/api-reference/functions/next-request

const BACKEND_URL = process.env.CHAT_BACKEND_URL;

// ngrok interstitial skip + a friendly user-agent for our own server logs
const CF_HEADERS = {
  'User-Agent':                 'UPal-UWTSD-Chatbot/1.0',
  'ngrok-skip-browser-warning': 'true',
};

async function fetchBackendHealth() {
  if (!BACKEND_URL) return null;
  try {
    // AbortSignal.timeout gives us a built-in timeout without wiring a
    // full AbortController. 8s is enough for the Python backend's three
    // parallel probes but short enough to not hang the Vercel function.
    // ref: https://developer.mozilla.org/en-US/docs/Web/API/AbortSignal/timeout_static
    const r = await fetch(`${BACKEND_URL.replace(/\/$/, '')}/api/health`, {
      method:  'GET',
      signal:  AbortSignal.timeout(8000),
      headers: CF_HEADERS,
    });
    if (!r.ok) return null;
    return await r.json();
  } catch {
    return null;
  }
}

export default async function handler(req, res) {
  if (req.method !== 'GET') return res.status(405).json({ error: 'Method not allowed' });

  const backend = await fetchBackendHealth();

  if (backend) {
    // pass the backend's report through so the UI sees the real state.
    // we whitelist the fields instead of spreading the whole object so
    // a future backend version can't leak internal fields accidentally.
    return res.status(200).json({
      status:          'OK',
      provider:        backend.provider        || 'anthropic',
      anthropic:       backend.anthropic       ?? 'not_configured',
      anthropicModel:  backend.anthropicModel  || null,
      ollama:          backend.ollama          ?? 'not_configured',
      ollamaModel:     backend.ollamaModel     || null,
      chroma:          backend.chroma          ?? 'empty',
      chromaDocs:      backend.chromaDocs      ?? 0,
      welsh:           backend.welsh           || 'active',
      bilingualTerms:  backend.bilingualTerms  ?? 0,
      welshVocab:      backend.welshVocab      ?? 0,
    });
  }

  // backend unreachable. we still return 200 so the CONNECTION card can
  // render (just with everything shown as offline) instead of showing
  // a raw fetch error in the UI.
  return res.status(200).json({
    status:          'DEGRADED',
    provider:        null,
    anthropic:       'offline',
    anthropicModel:  null,
    ollama:          'offline',
    ollamaModel:     null,
    chroma:          'offline',
    chromaDocs:      0,
    welsh:           'active',
    bilingualTerms:  0,
    welshVocab:      0,
  });
}
