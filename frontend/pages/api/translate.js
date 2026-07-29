'use strict';
// POST /api/translate. thin proxy to the Python backend's translate endpoint.
// the chat UI lazily fetches the opposite-language version of a reply when
// the user clicks the translate button, so we only pay for translation when
// someone actually asks for it.
// ref: https://nextjs.org/docs/pages/building-your-application/routing/api-routes

const BACKEND_URL = process.env.CHAT_BACKEND_URL;

export default async function handler(req, res) {
  // only POST, everything else is a 405
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const { text, from_lang, to_lang } = req.body || {};

  // basic shape validation. the backend also validates but failing fast
  // here saves a round trip.
  if (!text || typeof text !== 'string') {
    return res.status(400).json({ error: 'text is required' });
  }
  if (!from_lang || !to_lang) {
    return res.status(400).json({ error: 'from_lang and to_lang are required' });
  }

  // translation needs the LLM so we need the Python backend URL
  if (!BACKEND_URL) {
    return res.status(503).json({ error: 'Translation unavailable (Python backend not configured)' });
  }

  // 30s timeout via AbortController. Anthropic usually finishes in under
  // 3s but Ollama fallback on a cold container can take longer.
  // ref: https://developer.mozilla.org/en-US/docs/Web/API/AbortController
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), 30000);

  try {
    const r = await fetch(`${BACKEND_URL.replace(/\/$/, '')}/api/translate`, {
      method:  'POST',
      signal:  ctrl.signal,
      headers: {
        'Content-Type':               'application/json',
        // ngrok free tier interstitial skip, same as in /api/chat
        'ngrok-skip-browser-warning': 'true',
      },
      // cap at 8k chars so a rogue payload can't balloon the LLM prompt
      body: JSON.stringify({ text: text.slice(0, 8000), from_lang, to_lang }),
    });

    if (!r.ok) {
      console.warn('[translate] Python backend returned', r.status);
      return res.status(503).json({ error: `Backend returned ${r.status}` });
    }

    const data = await r.json();
    return res.status(200).json(data);
  } catch (e) {
    console.warn('[translate] Python backend unreachable:', e.message);
    return res.status(503).json({ error: 'Translation backend unreachable' });
  } finally {
    clearTimeout(t);
  }
}
