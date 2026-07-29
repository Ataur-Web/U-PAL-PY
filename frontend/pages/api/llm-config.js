'use strict';
// /api/llm-config. thin proxy to the Python backend so the operator can
// flip between Claude and Ollama (or change the model) without restarting.
// state is in-memory on the Python side and resets on backend restart.

const BACKEND_URL = process.env.CHAT_BACKEND_URL;

const CF_HEADERS = {
  'User-Agent':                 'UPal-UWTSD-Chatbot/1.0',
  'ngrok-skip-browser-warning': 'true',
};

export default async function handler(req, res) {
  // we only accept GET (read current config) and POST (update it)
  if (req.method !== 'GET' && req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }
  if (!BACKEND_URL) {
    return res.status(503).json({ error: 'Python backend not configured (CHAT_BACKEND_URL missing)' });
  }

  // 10s timeout. the backend usually responds instantly but the ngrok
  // tunnel can add latency.
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), 10000);

  try {
    const init = {
      method:  req.method,
      signal:  ctrl.signal,
      headers: { ...CF_HEADERS },
    };
    if (req.method === 'POST') {
      init.headers['Content-Type'] = 'application/json';
      init.body = JSON.stringify(req.body || {});
    }

    // forward the request and mirror the status code back. this keeps
    // the frontend error handling simple, if the backend 400s we 400 too.
    const r = await fetch(`${BACKEND_URL.replace(/\/$/, '')}/api/llm-config`, init);
    // .catch(() => ({})) stops a bad JSON body from crashing the route
    const data = await r.json().catch(() => ({}));
    return res.status(r.status).json(data);
  } catch (e) {
    console.warn('[llm-config] backend unreachable:', e.message);
    return res.status(503).json({ error: 'Backend unreachable' });
  } finally {
    clearTimeout(t);
  }
}
