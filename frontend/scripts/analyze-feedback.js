#!/usr/bin/env node
'use strict';
/**
 * analyze-feedback.js — extract training-signal insights from feedback.json
 *
 * What this does
 * --------------
 * U-Pal collects a 5-point satisfaction rating, two boolean flags
 * (helpfulAnswer, correctLanguage), and an optional free-text comment
 * every time a user taps the feedback widget.  The raw JSON is dumped to
 * feedback.json (local) or the Mongo collection if MONGODB_URI is set.
 *
 * This script aggregates that stream into the signals that actually drive
 * model improvements:
 *   • Overall health: average satisfaction, % helpful, % language-correct,
 *     volume trend (daily/weekly).
 *   • Failure modes: bottom-quartile satisfaction comments, clustered by
 *     shared keywords — these are your "where does the bot fail" prompts.
 *   • Language mix: are Welsh replies rated worse than English?  If so the
 *     native-Welsh generation path needs work.
 *   • Drop-off proxy: a session (sequence of feedbacks from the same
 *     approximate window) that ends on a low score is a probable drop-off.
 *
 * Usage
 * -----
 *   node scripts/analyze-feedback.js                 # local feedback.json
 *   node scripts/analyze-feedback.js --json a.json   # custom file
 *   node scripts/analyze-feedback.js --since 7       # last N days
 *   node scripts/analyze-feedback.js --mongo         # pull from Mongo
 *
 * This is a diagnostic tool, not a training pipeline — surface findings
 * here, then decide which intents/facts/prompts to revise.
 */
const fs   = require('fs');
const path = require('path');

function parseArgs(argv) {
  const out = { json: null, since: null, mongo: false };
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--json' && argv[i + 1]) { out.json = argv[++i]; }
    else if (a === '--since' && argv[i + 1]) { out.since = parseFloat(argv[++i]); }
    else if (a === '--mongo') { out.mongo = true; }
    else if (a === '--help' || a === '-h') {
      console.log(fs.readFileSync(__filename, 'utf8').split('\n').slice(2, 30).join('\n'));
      process.exit(0);
    }
  }
  return out;
}

async function loadFeedback(args) {
  if (args.mongo) {
    if (!process.env.MONGODB_URI) {
      throw new Error('MONGODB_URI not set — cannot read from Mongo');
    }
    const db = require(path.join(__dirname, '..', 'lib', 'db.js'));
    return db.readFeedback();
  }
  const file = args.json || path.join(__dirname, '..', 'feedback.json');
  if (!fs.existsSync(file)) {
    console.warn(`[analyze-feedback] No file at ${file}.`);
    console.warn('  The chatbot writes ratings here after users tap the feedback widget.');
    console.warn('  Run the app, collect a few ratings, then re-run this script.');
    return [];
  }
  const raw = JSON.parse(fs.readFileSync(file, 'utf8'));
  return Array.isArray(raw) ? raw : [];
}

function filterBySince(entries, days) {
  if (!days || !(days > 0)) return entries;
  const cutoff = Date.now() - days * 24 * 60 * 60 * 1000;
  return entries.filter(e => {
    const t = e.timestamp ? Date.parse(e.timestamp) : NaN;
    return Number.isFinite(t) && t >= cutoff;
  });
}

function pct(n, d) {
  if (!d) return '—';
  return ((n / d) * 100).toFixed(1) + '%';
}

function summarise(entries) {
  if (!entries.length) {
    console.log('No feedback entries to analyse.');
    return;
  }
  const n = entries.length;
  const sat = entries.map(e => Number(e.satisfaction)).filter(Number.isFinite);
  const avg = sat.reduce((a, b) => a + b, 0) / (sat.length || 1);
  const helpful   = entries.filter(e => e.helpfulAnswer   === true).length;
  const unhelpful = entries.filter(e => e.helpfulAnswer   === false).length;
  const correctLg = entries.filter(e => e.correctLanguage === true).length;
  const wrongLg   = entries.filter(e => e.correctLanguage === false).length;
  const lowSat    = entries.filter(e => Number(e.satisfaction) <= 2);
  const highSat   = entries.filter(e => Number(e.satisfaction) >= 4);

  console.log('═══ OVERALL HEALTH ═══');
  console.log(`Entries:             ${n}`);
  console.log(`Avg satisfaction:    ${avg.toFixed(2)} / 5`);
  console.log(`Helpful answer:      ${pct(helpful, helpful + unhelpful)}  (${helpful}/${helpful + unhelpful} rated)`);
  console.log(`Correct language:    ${pct(correctLg, correctLg + wrongLg)}  (${correctLg}/${correctLg + wrongLg} rated)`);
  console.log(`Low satisfaction:    ${pct(lowSat.length, n)}  (${lowSat.length} entries scored 1-2)`);
  console.log(`High satisfaction:   ${pct(highSat.length, n)}  (${highSat.length} entries scored 4-5)`);
  console.log();

  // Distribution
  console.log('═══ SATISFACTION DISTRIBUTION ═══');
  for (let s = 5; s >= 1; s--) {
    const c = sat.filter(x => x === s).length;
    const bar = '█'.repeat(Math.min(50, Math.round(c / Math.max(1, n) * 50)));
    console.log(`  ${s}★  ${String(c).padStart(4)}  ${bar}`);
  }
  console.log();

  // Failure modes — keyword clustering of low-sat comments
  const lowComments = lowSat
    .map(e => (e.comments || '').trim().toLowerCase())
    .filter(c => c.length > 3);
  if (lowComments.length) {
    const STOP = new Set(['the','and','a','is','to','of','for','with','in','on','it','that','this','i','you','my','we','are','was','be','but','not','no','or','so','an','as','at','by','from','have','had','has','if','me','our','your','their','they','them','there','when','why','how','what','where','also','more','some','any','just','like','really','very','too','get','got','can','could','would','should','will','did','do','does','done','than','then','into','about','because','while','only']);
    const wordFreq = new Map();
    for (const c of lowComments) {
      for (const w of c.split(/[^a-zA-ZÀ-ÿŵŷ]+/).filter(x => x.length > 2 && !STOP.has(x))) {
        wordFreq.set(w, (wordFreq.get(w) || 0) + 1);
      }
    }
    const top = [...wordFreq.entries()].sort((a, b) => b[1] - a[1]).slice(0, 12);
    console.log('═══ FAILURE-MODE KEYWORDS (from low-sat comments) ═══');
    top.forEach(([w, c]) => console.log(`  ${String(c).padStart(3)}  ${w}`));
    console.log();
    console.log('Sample low-rating comments (up to 10):');
    lowComments.slice(0, 10).forEach((c, i) => console.log(`  ${i + 1}. "${c.slice(0, 140)}"`));
    console.log();
  }

  // Language breakdown
  if (correctLg + wrongLg > 0) {
    console.log('═══ LANGUAGE QUALITY ═══');
    console.log(`  Correct language  : ${correctLg}`);
    console.log(`  Wrong language    : ${wrongLg}`);
    if (wrongLg > correctLg * 0.15) {
      console.log('  ⚠  Wrong-language rate > 15% — revisit the detectLanguage heuristic');
      console.log('     and the native-Welsh generation guard (looksWelsh + pivot fallback).');
    }
  }

  // Daily volume
  console.log();
  console.log('═══ DAILY VOLUME (last 14 days) ═══');
  const byDay = new Map();
  for (const e of entries) {
    const d = (e.timestamp || '').slice(0, 10);
    if (d) byDay.set(d, (byDay.get(d) || 0) + 1);
  }
  const days = [...byDay.entries()].sort().slice(-14);
  days.forEach(([d, c]) => {
    const bar = '■'.repeat(Math.min(40, c));
    console.log(`  ${d}  ${String(c).padStart(4)}  ${bar}`);
  });
}

(async () => {
  const args = parseArgs(process.argv);
  let entries;
  try { entries = await loadFeedback(args); }
  catch (e) { console.error('Failed to load feedback:', e.message); process.exit(1); }
  entries = filterBySince(entries, args.since);
  summarise(entries);
})();
