'use strict';
// safety.js. guardrails that sit between retrieval and the LLM in the
// legacy Node NLP pipeline. the hard-coded crisis keyword gate (Samaritans
// short-circuit) lives upstream in nlp.js, this module handles everything
// else a RAG layer needs.
//
// four guardrails are implemented here:
//   1. scrubPII          strip student IDs, phones, emails, NI numbers,
//                        and card numbers from retrieved chunks
//   2. detectInjection   drop chunks that contain indirect-prompt-injection
//                        phrasing. ref: Greshake et al. 2023,
//                        "Not what you've signed up for",
//                        https://arxiv.org/abs/2302.12173
//   3. isOffTopic        refuse politely when every retrieval signal is weak
//   4. checkGrounding    word-overlap ratio between LLM answer and the
//                        retrieved passages. low ratio hints at fabrication
//
// all four are pure functions so they are safe to call from anywhere.

// -----------------------------------------------------------------------
// 1. PII scrubber
// -----------------------------------------------------------------------
// we replace matches with a short placeholder so the LLM never sees them
// and cannot echo them back.

// UWTSD student IDs are 7-9 digit numbers that appear on portal pages.
// rare in the public corpus but conservative to strip.
const RE_STUDENT_ID = /\b(?:student\s*id[:\s]*)?(\d{7,9})\b/gi;

// UK phone numbers. we keep the public UWTSD switchboard numbers so the
// bot can still tell a student to ring enquiries.
// ref: https://en.wikipedia.org/wiki/Telephone_numbers_in_the_United_Kingdom
const UWTSD_PHONE_ALLOWLIST = /^(?:\+?44\s?)?(?:0?1792\s?481000|0?300\s?500|0?1267\s?676|0?1570\s?422|0?300\s?323)\b/;
const RE_PHONE = /\b(?:\+?44\s?)?(?:0\s?)?(?:\d[\s\-]?){10,11}\b/g;

// personal emails get masked, public @uwtsd addresses are kept
const RE_EMAIL = /\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b/gi;

// UK postcodes. campus postcodes are public so we default to keeping them.
// callers who want strict mode pass { postcodes: true }.
// ref: https://www.gov.uk/government/publications/bulk-data-transfer-for-sponsors-xml-schema
const RE_POSTCODE = /\b[A-Z]{1,2}\d{1,2}[A-Z]?\s*\d[A-Z]{2}\b/gi;

// National Insurance numbers are never legitimate inside a UWTSD answer.
// ref: https://design.tax.service.gov.uk/hmrc-design-patterns/national-insurance-number/
const RE_NI = /\b[A-CEGHJ-PR-TW-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D]\b/gi;

// payment cards: 13-19 digits, possibly grouped. we run Luhn on any hit
// to avoid stripping a coincidental long number.
// ref: https://en.wikipedia.org/wiki/Luhn_algorithm
const RE_CARD = /\b(?:\d[ -]?){13,19}\b/g;
function luhn(s) {
  const digits = String(s).replace(/\D/g, '');
  if (digits.length < 13) return false;
  let sum = 0, alt = false;
  for (let i = digits.length - 1; i >= 0; i--) {
    let d = +digits[i];
    if (alt) { d *= 2; if (d > 9) d -= 9; }
    sum += d;
    alt = !alt;
  }
  return sum % 10 === 0;
}

function isPublicPhone(s) {
  return UWTSD_PHONE_ALLOWLIST.test(String(s).trim());
}
function isPublicEmail(s) {
  const lc = String(s).toLowerCase();
  return /@uwtsd\.ac\.uk$|@uwtsd\.com$|@tsdsu\.co\.uk$/.test(lc);
}

// scrub a retrieved passage. returns the cleaned string.
function scrubPII(text, opts = {}) {
  if (typeof text !== 'string' || !text) return text;
  let out = text;

  out = out.replace(RE_NI, '[NI-NUMBER]');
  out = out.replace(RE_CARD, m => luhn(m) ? '[CARD-NUMBER]' : m);
  out = out.replace(RE_EMAIL, m => isPublicEmail(m) ? m : '[EMAIL]');
  out = out.replace(RE_PHONE, m => isPublicPhone(m) ? m : '[PHONE]');
  out = out.replace(RE_STUDENT_ID, m => /student\s*id/i.test(m) ? '[STUDENT-ID]' : m);
  if (opts.postcodes) out = out.replace(RE_POSTCODE, '[POSTCODE]');

  return out;
}

// -----------------------------------------------------------------------
// 2. Injection filter
// -----------------------------------------------------------------------
// the list is kept deliberately tight. the goal is to drop chunks where
// someone has clearly written an instruction aimed at the LLM, not every
// chunk that contains the word "ignore".
// ref: Greshake et al. 2023, https://arxiv.org/abs/2302.12173
const INJECTION_PATTERNS = [
  /ignore\s+(all\s+|the\s+|any\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|rules?|messages?|directives?)/i,
  /disregard\s+(all\s+|the\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)/i,
  /you\s+are\s+now\s+(a\s+|an\s+)?[a-z\s]{2,40}(instead|not)/i,
  /forget\s+(everything|all)\s+(above|before|prior|previous)/i,
  /new\s+instructions?\s*:\s*/i,
  /(system|admin|developer)\s*(prompt|message|override)\s*[:=]/i,
  /\[\s*(system|admin|override|jailbreak)\s*\]/i,
  /output\s+only\s+(the\s+following|exactly|this)/i,
  /respond\s+only\s+with\s+["'`]/i,
  /print\s+your\s+(system\s+)?prompt/i,
  /repeat\s+the\s+(above|following)\s+text/i,
];

function detectInjection(text) {
  if (typeof text !== 'string' || !text) return { injected: false };
  for (const re of INJECTION_PATTERNS) {
    const m = text.match(re);
    if (m) return { injected: true, pattern: re.source, match: m[0].slice(0, 80) };
  }
  return { injected: false };
}

// filter retrieved passages, dropping anything that looks injected and
// scrubbing PII from the rest. `dropped` is returned for diagnostics.
function sanitisePassages(passages) {
  const clean = [];
  const dropped = [];
  for (const p of passages || []) {
    const det = detectInjection(p.content || '');
    if (det.injected) {
      dropped.push({ id: p.id, reason: 'injection', pattern: det.pattern, match: det.match });
      continue;
    }
    clean.push({ ...p, content: scrubPII(p.content) });
  }
  return { clean, dropped };
}

// -----------------------------------------------------------------------
// 3. Off-topic gate
// -----------------------------------------------------------------------
// decides whether to refuse rather than let the LLM hallucinate. inputs
// come from the caller's retrieval signals, we do not hit the corpus again.
function isOffTopic({
  topIntentConfidence = 0,
  hasFactMatch        = false,
  topLexicalScore     = 0,
  topDenseScore       = 0,
  hasMorphikContext   = false,
  hasCorpusHits       = false,
} = {}) {
  // any single strong positive signal means on-topic
  if (topIntentConfidence >= 0.40) return false;
  if (hasFactMatch)                return false;
  if (hasMorphikContext)           return false;
  if (topDenseScore   >= 0.55)     return false;
  if (topLexicalScore >= 0.30)     return false;

  // off-topic only when every signal is near-zero
  const nothing = !hasCorpusHits &&
                  topIntentConfidence < 0.15 &&
                  topDenseScore   < 0.35 &&
                  topLexicalScore < 0.10;
  return nothing;
}

// -----------------------------------------------------------------------
// 4. Grounding check
// -----------------------------------------------------------------------
// lightweight word-overlap: what fraction of answer tokens (4+ chars, not
// stopwords) appear verbatim in the retrieved passages. very low ratio
// (< 0.15) hints at fabrication. ref: Es et al. 2023, "RAGAS",
// https://arxiv.org/abs/2309.15217
const STOP = new Set([
  'the','a','an','is','are','was','were','be','been','being','have','has','had',
  'do','does','did','will','would','shall','should','may','might','must','can','could',
  'to','of','in','on','at','for','with','by','from','about','into','through',
  'and','but','or','so','if','that','this','these','those','there','here','not',
  'you','your','they','their','them','we','our','us','he','his','she','her','i','me','my',
  'what','which','who','when','where','why','how','also','just','some','any','all',
  'more','most','many','much','over','such','than','then','only','own','same','each',
]);

function tokens(s) {
  return String(s || '')
    .toLowerCase()
    .replace(/[^a-zA-ZÀ-ÿŵŷ0-9]+/g, ' ')
    .split(/\s+/)
    .filter(t => t.length >= 4 && !STOP.has(t));
}

function checkGrounding(answer, passages) {
  const ansToks = tokens(answer);
  if (!ansToks.length) return { ratio: 0, supported: 0, total: 0 };
  const passageToks = new Set();
  for (const p of passages || []) tokens(p.content || p).forEach(t => passageToks.add(t));
  let supported = 0;
  for (const t of ansToks) if (passageToks.has(t)) supported++;
  return {
    ratio:     supported / ansToks.length,
    supported,
    total:     ansToks.length,
  };
}

module.exports = {
  scrubPII,
  detectInjection,
  sanitisePassages,
  isOffTopic,
  checkGrounding,
};
