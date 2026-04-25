'use strict';
// /api/feedback. POST saves a rating from the feedback modal. GET returns
// all rows for the admin dashboard (Basic Auth protected).

const { saveFeedback, readFeedback } = require('../../lib/db');

function requireAdmin(req, res) {
  // HTTP Basic Auth check. we read the password from ADMIN_PASSWORD env
  // var so it's never in the code. any username is accepted.
  // ref: https://developer.mozilla.org/en-US/docs/Web/HTTP/Authentication
  const pw = process.env.ADMIN_PASSWORD;
  if (!pw) {
    res.status(503).json({ error: 'Admin access not configured.' });
    return false;
  }
  const auth = req.headers['authorization'] || '';
  if (!auth.startsWith('Basic ')) {
    res.setHeader('WWW-Authenticate', 'Basic realm="U-Pal Admin"');
    res.status(401).json({ error: 'Authentication required.' });
    return false;
  }
  // the Basic Auth header is base64("user:pass"). we split on the first
  // ':' so passwords that contain colons still work.
  const decoded  = Buffer.from(auth.split(' ')[1], 'base64').toString();
  const password = decoded.split(':').slice(1).join(':');
  if (password !== pw) {
    res.setHeader('WWW-Authenticate', 'Basic realm="U-Pal Admin"');
    res.status(401).json({ error: 'Incorrect password.' });
    return false;
  }
  return true;
}

export default async function handler(req, res) {
  if (req.method === 'POST') {
    // public POST, anyone can submit feedback. we clamp and sanitise so
    // a bad client can't stuff the DB with rubbish.
    const { satisfaction, correctLanguage, helpfulAnswer, comments } = req.body || {};
    if (!satisfaction) {
      return res.status(400).json({ error: 'satisfaction is required' });
    }
    const entry = {
      timestamp:       new Date().toISOString(),
      // Math.min/max clamps the star rating into the 1-5 range
      satisfaction:    Math.min(5, Math.max(1, Number(satisfaction))),
      helpfulAnswer:   helpfulAnswer === true || helpfulAnswer === false ? helpfulAnswer : null,
      correctLanguage: correctLanguage === true || correctLanguage === false ? correctLanguage : null,
      // cap comment length so a large POST can't fill the DB
      comments:        comments ? String(comments).slice(0, 300) : ''
    };
    try {
      await saveFeedback(entry);
      return res.status(201).json({ success: true });
    } catch (err) {
      console.error('Feedback save error:', err.message);
      return res.status(500).json({ error: 'Could not save feedback' });
    }
  }

  if (req.method === 'GET') {
    // admin-only. requireAdmin sends the 401 for us if auth fails.
    if (!requireAdmin(req, res)) return;
    try {
      const data = await readFeedback();
      return res.status(200).json(data);
    } catch (err) {
      console.error('Feedback read error:', err.message);
      return res.status(500).json({ error: 'Could not read feedback' });
    }
  }

  return res.status(405).json({ error: 'Method not allowed' });
}
