#!/usr/bin/env node
'use strict';
/**
 * dedupe-patterns.js — remove duplicate training patterns across intents.
 *
 * Duplicate patterns across intents confuse the Naive Bayes classifier:
 * the same string is presented as evidence for two different labels,
 * which flattens the posterior and makes the intent boundary fuzzy.
 *
 * Strategy: for each pair, keep the pattern in the more-specific intent.
 * Specificity is coded below as KEEP_IN — the intent name mapped to wins.
 * If a pattern isn't in the map, we default to removing from whichever
 * intent appears later in the file (the "also" in the audit report),
 * which empirically is the less-authoritative home.
 *
 * Run:  node scripts/dedupe-patterns.js [--dry]
 */
const fs   = require('fs');
const path = require('path');

const DRY = process.argv.includes('--dry');
const KB_PATH = path.join(__dirname, '..', 'knowledge.json');
const kb = JSON.parse(fs.readFileSync(KB_PATH, 'utf8'));

// Hand-curated rules for ambiguous pairs where specificity isn't obvious
// from file order.  Keys are the normalised pattern (lowercased, trimmed).
// Value is the intent to KEEP the pattern in — it'll be removed from any
// other intent.
const KEEP_IN = {
  'feeling hopeless':    'wellbeing_crisis',        // crisis > general
  'money for university':'fees_scholarships',       // specific > generic tuition
  'professional development': 'career_services',    // career > PG courses
  'pgce':                'courses_education',       // domain intent > PG catalog
  'teacher training':    'courses_education',
  'undergraduate courses': 'undergraduate_courses',
  'postgraduate courses':  'postgraduate_courses',
  'mature student entry':  'admissions_requirements',
  // campus_locations is the umbrella listing; specific campus intents win
  'carmarthen campus':   'campuses_carmarthen',
  'swansea campus':      'campuses_swansea',
  'lampeter campus':     'campuses_lampeter',
  'how many campuses':   'campuses_general',
  // enrolment-specific wording goes to enrolment_tasks
  'how to enrol':        'enrolment_tasks',
  'enrolment process':   'enrolment_tasks',
  // human hand-off goes to human_agent
  'speak to someone':    'human_agent',
  'talk to a human':     'human_agent',
  // accommodation cost wording -> cost intent
  'how much is accommodation': 'accommodation_cost',
  'accommodation fees':        'accommodation_cost',
  'cost of halls':             'accommodation_cost',
  'cost llety':                'accommodation_cost',
  // hardship/financial wording -> financial_support
  'hardship fund':     'financial_support',
  'emergency fund':    'financial_support',
  'help ariannol':     'financial_support',
  // "how do I apply" family -> admissions_apply (has richer response set)
  'how do i apply':       'admissions_apply',
  'application process':  'admissions_apply',
  'apply for a course':   'admissions_apply',
  'where do i apply':     'admissions_apply',
  'application deadline': 'admissions_deadline',
};

// Build a first-seen map so we know where each pattern currently lives.
const firstSeen = new Map();
const pairs = []; // list of { pattern, intents[] }
for (const intent of kb) {
  for (const p of intent.patterns || []) {
    const k = String(p).toLowerCase().trim();
    if (!k) continue;
    if (firstSeen.has(k)) {
      // find or create the pair record
      let rec = pairs.find(r => r.pattern === k);
      if (!rec) { rec = { pattern: k, intents: [firstSeen.get(k)] }; pairs.push(rec); }
      rec.intents.push(intent.tag);
    } else {
      firstSeen.set(k, intent.tag);
    }
  }
}

if (!pairs.length) {
  console.log('No duplicate patterns found. Nothing to do.');
  process.exit(0);
}

let removed = 0;
for (const { pattern, intents } of pairs) {
  const keeper = KEEP_IN[pattern] || intents[0];  // default: keep in first-seen
  const losers = intents.filter(t => t !== keeper);
  if (!intents.includes(keeper)) {
    // KEEP_IN references an intent that doesn't actually contain this pattern —
    // fall back to first-seen to avoid deleting everywhere.
    console.warn(`[warn] KEEP_IN["${pattern}"] = "${keeper}" but pattern is only in [${intents.join(', ')}] — keeping first-seen instead`);
    continue;
  }
  for (const loserTag of losers) {
    const intent = kb.find(k => k.tag === loserTag);
    if (!intent || !intent.patterns) continue;
    const before = intent.patterns.length;
    intent.patterns = intent.patterns.filter(p => p.toLowerCase().trim() !== pattern);
    const after = intent.patterns.length;
    if (after < before) {
      removed += (before - after);
      console.log(`  - ${loserTag}: removed "${pattern}"  (keeps in ${keeper})`);
    }
  }
}

console.log();
console.log(`Removed ${removed} duplicate pattern instance(s) across ${pairs.length} pattern groups.`);

if (DRY) {
  console.log('[dry run] — no file changes written.');
} else {
  fs.writeFileSync(KB_PATH, JSON.stringify(kb, null, 2) + '\n', 'utf8');
  console.log(`Wrote cleaned knowledge.json (${kb.length} intents).`);
}
