'use strict';
// POST /api/chat. proxies a student message to the Python backend and
// reshapes the response for the frontend.
//
// if CHAT_BACKEND_URL is set we forward to the FastAPI backend (U-PAL-PY)
// over the ngrok static tunnel. if it's down we fall back to the legacy
// in-process Node NLP pipeline so the deploy still answers.
// ref: https://nextjs.org/docs/pages/building-your-application/routing/api-routes

const { processChat } = require('../../lib/nlp');

const BACKEND_URL = process.env.CHAT_BACKEND_URL;

// the Python backend returns { reply, intent, lang, emotion, sources }.
// the frontend expects { response, altResponse, tag, ... } from the old
// Node version, so we re-shape here to keep the contract stable.
function normalisePythonResponse(py, lang) {
  if (!py || typeof py !== 'object') return null;
  return {
    response:    py.reply || '',
    altResponse: '',
    tag:         py.intent || 'python_backend',
    lang:        py.lang || lang,
    confidence:  1,
    source:      'python',
    emotion:     py.emotion || 'neutral',
    sources:     Array.isArray(py.sources) ? py.sources : [],
  };
}

async function callPythonBackend({ message, runningLang, history }) {
  // we use AbortController to enforce a 30s timeout. without it a dead
  // tunnel would block the request forever and Vercel would 504.
  // ref: https://developer.mozilla.org/en-US/docs/Web/API/AbortController
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), 30000);
  try {
    const r = await fetch(`${BACKEND_URL.replace(/\/$/, '')}/api/chat`, {
      method:  'POST',
      signal:  ctrl.signal,
      headers: {
        'Content-Type':               'application/json',
        // ngrok free tier shows an interstitial HTML page unless we
        // send this header, which would break the JSON parse.
        'ngrok-skip-browser-warning': 'true',
      },
      body: JSON.stringify({
        message,
        runningLang,
        // only send the last 6 turns to keep the LLM prompt small
        history: Array.isArray(history) ? history.slice(-6) : [],
      }),
    });
    if (!r.ok) {
      console.warn('[chat] Python backend returned', r.status);
      return null;
    }
    const data = await r.json();
    return normalisePythonResponse(data, runningLang);
  } catch (e) {
    console.warn('[chat] Python backend unreachable:', e.message);
    return null;
  } finally {
    clearTimeout(t);
  }
}

export default async function handler(req, res) {
  // only POST is allowed, anything else gets a 405.
  // ref: https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const { message, runningLang, history } = req.body || {};

  if (!message || typeof message !== 'string') {
    return res.status(400).json({ error: 'message is required' });
  }

  // cap at 1000 chars so a malicious user can't flood the LLM prompt
  const trimmedMsg = message.slice(0, 1000);

  // prefer the Python backend when we have a URL for it
  if (BACKEND_URL) {
    const py = await callPythonBackend({
      message:     trimmedMsg,
      runningLang: runningLang || 'en',
      history,
    });
    if (py) return res.status(200).json(py);
    // on failure we fall through to the Node pipeline so the bot still replies
  }

  try {
    const result = await processChat(
      trimmedMsg,
      runningLang || 'en',
      Array.isArray(history) ? history.slice(-6) : []
    );
    return res.status(200).json(result);
  } catch (err) {
    console.error('[/api/chat]', err.message, err.stack);
    // bilingual error fallback, shown if both backends failed
    const lang  = runningLang || 'en';
    const errMsg = {
      en: "I'm sorry, something didn't work as expected on my end. Some features are still being improved. In the meantime, please try rephrasing your question, or contact UWTSD directly at enquiries@uwtsd.ac.uk or 01792 481 111.",
      cy: "Mae'n ddrwg gen i, aeth rhywbeth o'i le ar fy ochr i. Mae rhai nodweddion yn dal i gael eu gwella. Yn y cyfamser, rhowch gynnig ar aileirio eich cwestiwn, neu cysylltwch â PCYDDS yn uniongyrchol ar enquiries@uwtsd.ac.uk neu 01792 481 111."
    };
    return res.status(200).json({
      response:    errMsg[lang] || errMsg.en,
      altResponse: errMsg[lang === 'cy' ? 'en' : 'cy'],
      tag:         'error_fallback',
      lang,
      confidence:  0,
      source:      'error'
    });
  }
}
