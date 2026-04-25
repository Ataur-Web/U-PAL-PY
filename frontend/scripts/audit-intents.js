#!/usr/bin/env node
'use strict';
/**
 * audit-intents.js — apply the intent-design checklist to our data files.
 *
 * Rules we enforce (adapted from common chatbot-design guidance — e.g.
 * Dialogflow/Rasa "keep intents focused, 3+ examples each"):
 *
 *   1. Each intent in knowledge.json needs ≥ 3 trigger patterns (so the
 *      classifier has real signal) and ≥ 2 response variants per language
 *      (so repeated hits don't feel canned).
 *   2. Intents shouldn't sprawl — > 40 patterns is usually a sign the
 *      intent is doing too many jobs and should be split, with entities
 *      carrying the variation.
 *   3. Every curated fact in uwtsd-facts.json should have at least one
 *      Welsh question and a Welsh answer (bilingual parity).
 *   4. Qualifier-tagged facts (e.g. "fees-international-postgraduate")
 *      should declare at least one qualifier and have a clean match in
 *      QUALIFIER_PATTERNS.
 *   5. Duplicate pattern strings across intents cause classifier
 *      confusion — flag them.
 *
 * Run in CI or on-demand: `node scripts/audit-intents.js`.
 * Exits with code 1 if any rule is violated, 0 otherwise.
 */
const fs   = require('fs');
const path = require('path');

const ROOT = path.join(__dirname, '..');
const read = p => JSON.parse(fs.readFileSync(path.join(ROOT, p), 'utf8'));

const kb        = read('knowledge.json');
const facts     = read('uwtsd-facts.json');
// Qualifier list hard-copied from lib/nlp.js to avoid booting the whole app.
const QUALIFIERS = new Set([
  'international','postgraduate','undergraduate','home','online','parttime',
  'phd','pgce',
]);

const problems = [];
const warn = (rule, tag, msg) => problems.push({ rule, tag, msg });

// ── knowledge.json ────────────────────────────────────────────────────────
const seenPatterns = new Map();
for (const intent of kb) {
  const tag = intent.tag || '(no tag)';
  const patterns = Array.isArray(intent.patterns) ? intent.patterns : [];
  const r = intent.responses || {};
  const enR = Array.isArray(r) ? r : (r.en || []);
  const cyR = Array.isArray(r) ? [] : (r.cy || []);

  if (patterns.length < 3) warn('R1', tag, `only ${patterns.length} training patterns — add at least 3`);
  if (enR.length < 2)       warn('R1', tag, `only ${enR.length} English responses — add at least 2 for variety`);
  if (cyR.length < 2)       warn('R1', tag, `only ${cyR.length} Welsh responses — add at least 2 for variety`);
  if (patterns.length > 40) warn('R2', tag, `${patterns.length} patterns — consider splitting into sub-intents with entities`);

  for (const p of patterns) {
    const key = String(p).toLowerCase().trim();
    if (!key) continue;
    if (seenPatterns.has(key) && seenPatterns.get(key) !== tag) {
      warn('R5', tag, `duplicate pattern "${p}" also in intent "${seenPatterns.get(key)}"`);
    } else {
      seenPatterns.set(key, tag);
    }
  }
}

// ── uwtsd-facts.json ──────────────────────────────────────────────────────
for (const fact of facts) {
  const id = fact.id || '(no id)';
  const hasCyQuestion = (fact.questions || []).some(
    q => /[ŵŷâêîôûẁỳẃýÿï]/.test(q) || /\b(sut|beth|ble|pryd|pam|faint|ydy|yw)\b/i.test(q)
  );
  if (!fact.answer_cy)   warn('R3', id, 'missing Welsh answer (answer_cy)');
  if (!hasCyQuestion)    warn('R3', id, 'no Welsh question in questions[] — Welsh users may not match');

  const quals = fact.qualifiers || [];
  if (id.includes('fees-') && id !== 'fees-general') {
    if (!quals.length) warn('R4', id, 'fees fact without qualifiers — will not be routed correctly');
    for (const q of quals) {
      if (!QUALIFIERS.has(q)) {
        warn('R4', id, `qualifier "${q}" not recognised — must be one of: ${[...QUALIFIERS].join(', ')}`);
      }
    }
  }
}

// ── Report ────────────────────────────────────────────────────────────────
console.log(`Audited ${kb.length} intents and ${facts.length} facts.`);
if (!problems.length) {
  console.log('✓  No issues found.');
  process.exit(0);
}
const byRule = new Map();
for (const p of problems) {
  if (!byRule.has(p.rule)) byRule.set(p.rule, []);
  byRule.get(p.rule).push(p);
}
const ruleDescs = {
  R1: 'R1 — min 3 patterns + 2 responses per language',
  R2: 'R2 — avoid > 40 patterns per intent',
  R3: 'R3 — bilingual parity (Welsh questions + answers)',
  R4: 'R4 — fees facts must declare valid qualifiers',
  R5: 'R5 — no duplicate patterns across intents',
};
for (const [rule, items] of [...byRule.entries()].sort()) {
  console.log();
  console.log(`── ${ruleDescs[rule] || rule} — ${items.length} issue(s) ──`);
  items.slice(0, 30).forEach(p => console.log(`  [${p.tag}]  ${p.msg}`));
  if (items.length > 30) console.log(`  … and ${items.length - 30} more`);
}
console.log();
console.log(`Total: ${problems.length} issue(s).`);
process.exit(problems.length ? 1 : 0);
