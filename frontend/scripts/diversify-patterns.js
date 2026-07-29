#!/usr/bin/env node
'use strict';
/**
 * diversify-patterns.js — add diverse, natural-sounding trigger phrases
 * (typos, contractions, slang, long/short variants) to the highest-traffic
 * intents.  Driven by a curated additions map rather than auto-generated
 * paraphrases so we keep quality high.
 *
 * Why this matters: Naive Bayes + TF-IDF classifiers generalise from the
 * shape of the patterns they see.  If every pattern is a polished, fully
 * punctuated question ("How do I apply to UWTSD?"), the classifier
 * struggles with the real user ("hi how do i aply pls" / "any idea how 2
 * apply???").  Adding realistic variants broadens the margin.
 *
 * Run:  node scripts/diversify-patterns.js [--dry]
 */
const fs   = require('fs');
const path = require('path');

const DRY     = process.argv.includes('--dry');
const KB_PATH = path.join(__dirname, '..', 'knowledge.json');
const kb      = JSON.parse(fs.readFileSync(KB_PATH, 'utf8'));

// Curated additions per tag.  Mix of: typos, missing-apostrophes,
// all-lowercase casual, txt-speak, chatty/long variants, Welsh variants,
// and slang.  We avoid adding patterns that already conflict with other
// intents (see scripts/audit-intents.js for the dedupe rules).
const ADDITIONS = {
  graduation: [
    'im graduating this year help',
    'graduating soon whats next',
    'finished my degree what now',
    'just done my finals how do I book graduation',
    'graduation gown hire uwtsd',
    'can my family come to graduation',
    'how many guests at grad',
    'grad day tickets',
    'i finish uni this summer info please',
    'capping ceremony',
    'congrats ceremony',
    'rwyf wedi gorffen fy ngradd beth nesaf',
    'sut dw i\'n cael gwn graddio',
  ],
  admissions_apply: [
    'hiya how do i apply',
    'how to aply to uwtsd',   // typo
    'how to aplly',           // typo
    'wheres the application form',
    'whats the application link',
    'can u send me the apply link',
    'i wanna apply how',
    'how do i get on a course',
    'start my application',
    'applying as an international student',
    'apply via ucas or direct',
    'apply direct to uwtsd',
    'sign up for a degree',
    'sut mae gwneud cais i PCYDDS',
    'cais i pcydds',
  ],
  fees_tuition: [
    'hw much is uni',         // txt-speak
    'how much it cost',
    'whats the fee',
    'what r the fees',
    'full time fees uwtsd',
    'part time fees per credit',
    'fees for 2025 26',
    'fees for next year',
    'cost of doing a degree',
    'uni prices',
    'how expensive is uwtsd',
    'can i afford uwtsd',
    'faint mae uwtsd yn costio',
  ],
  library: [
    'libary opening hours',    // typo
    'libary times',
    'lib hours',
    'when does the library close',
    'when does the library open',
    'is the library open today',
    'open 24 7 library',
    'can i study in the library overnight',
    'borrow a laptop from library',
    'library print credit',
    'renew my books',
    'when do i return books',
    'oriau llyfrgell heddiw',
  ],
  it_helpdesk: [
    'cant login to moodle',
    'cant access my uni email',
    'forgot my uwtsd password',
    'reset my password pls',
    'reset pword',
    'help with wifi',
    'laptop broken who to contact',
    'it help pls',
    'computer not working',
    'login issue',
    'multi factor authentication problem',
    'mfa not working',
    'help gyda mewngofnodi',
    'dw i ddim yn gallu cael mynediad i moodle',
  ],
  accommodation_general: [
    'halls of residence info',
    'is accommodation guaranteed for first years',
    'can i pick my room',
    'how do i get a room at uwtsd',
    'student housing options',
    'student digs swansea',
    'off campus housing',
    'privately rented near uwtsd',
    'can i bring my pet to halls',
    'is there parking at halls',
    'halls of residence cost',
    'shared vs ensuite rooms',
    'am i getting a single room',
    'llety pcydds i fyfyrwyr israddedig',
    'neuaddau preswyl abertawe',
  ],
  wellbeing_general: [
    'im struggling mentally',
    'need someone to talk to',
    'feeling really down lately',
    'uni is getting too much',
    'cant cope with assignments',
    'homesick and sad',
    'anxious about exams',
    'stressed and cant sleep',
    'who do i talk to at wellbeing',
    'mental health support uni',
    'teimlo\'n isel yn yr brifysgol',
    'dw i\'n cael trafferth yn feddyliol',
  ],
};

let totalAdded = 0;
for (const [tag, add] of Object.entries(ADDITIONS)) {
  const intent = kb.find(k => k.tag === tag);
  if (!intent) { console.warn(`[skip] intent "${tag}" not found`); continue; }
  intent.patterns = intent.patterns || [];
  const have = new Set(intent.patterns.map(p => String(p).toLowerCase().trim()));
  let added = 0;
  for (const p of add) {
    const k = p.toLowerCase().trim();
    if (!have.has(k)) {
      intent.patterns.push(p);
      have.add(k);
      added++;
    }
  }
  totalAdded += added;
  console.log(`  ${tag.padEnd(25)} +${added} patterns  (now ${intent.patterns.length})`);
}

console.log();
console.log(`Added ${totalAdded} new patterns across ${Object.keys(ADDITIONS).length} intents.`);
if (DRY) {
  console.log('[dry run] — no file changes written.');
} else {
  fs.writeFileSync(KB_PATH, JSON.stringify(kb, null, 2) + '\n', 'utf8');
  console.log('Wrote knowledge.json.');
}
