'use strict';
/**
 * build-naivebayes-training.js
 *
 * Reads naive-bayes-training.json (produced by train_from_openorca.py) and
 * emits a pre-trained natural.js BayesClassifier sidecar file so the chatbot
 * can load a richer classifier at startup instead of training from the slim
 * knowledge.json patterns alone.
 *
 * Usage:
 *   node scripts/build-naivebayes-training.js
 *   node scripts/build-naivebayes-training.js --input naive-bayes-training.json \
 *                                              --output lib/bayes-classifier.json
 *
 * The output JSON is loaded by lib/nlp.js at startup if present:
 *   try {
 *     bayesClassifier = natural.BayesClassifier.restore(
 *       JSON.parse(fs.readFileSync('lib/bayes-classifier.json', 'utf8'))
 *     );
 *   } catch(_) { // train inline
 *   }
 */

const path    = require('path');
const fs      = require('fs');
const natural = require('natural');

// ─── CLI args ──────────────────────────────────────────────────────────────
const argv = process.argv.slice(2);
const getArg = (flag, def) => {
  const i = argv.indexOf(flag);
  return i >= 0 && argv[i + 1] ? argv[i + 1] : def;
};
const INPUT_PATH    = getArg('--input',    path.join(process.cwd(), 'naive-bayes-training.json'));
const OUTPUT_PATH   = getArg('--output',   path.join(process.cwd(), 'lib', 'bayes-classifier.json'));
const KNOWLEDGE_PATH = getArg('--knowledge', path.join(process.cwd(), 'knowledge.json'));

// ─── Load training data ────────────────────────────────────────────────────
console.log(`[BayesTrainer] Loading training data from ${INPUT_PATH}...`);
let trainingData;
try {
  trainingData = JSON.parse(fs.readFileSync(INPUT_PATH, 'utf8'));
} catch (e) {
  console.error('[BayesTrainer] ERROR: could not read training file:', e.message);
  console.error('  Run: python3 scripts/train_from_openorca.py  first');
  process.exit(1);
}

// ─── Build pre-processor (same as lib/nlp.js) ─────────────────────────────
const tokenizer = new natural.WordTokenizer();
const EN_STOP = new Set([
  'the','a','an','is','are','was','were','be','been','being','have','has','had',
  'do','does','did','will','would','shall','should','may','might','must','can','could',
  'to','of','in','on','at','for','with','by','from','about','into','through',
  'i','me','my','we','our','you','your','he','his','she','her','it','its','they','their',
  'what','which','who','this','that','these','those','am','not','no','so','if','or',
  'and','but','how','when','where','why','please','want','need','help','tell','know',
  'get','like','just','also','more','some','any','all','very','really','actually',
  'im','ive','id','ill','cant','dont','doesnt','isnt','arent','wasnt','wont','havent'
]);

function preprocess(text) {
  const norm    = text.toLowerCase().replace(/[^\w\u00C0-\u024F\s]/g, ' ').replace(/\s+/g, ' ').trim();
  const tokens  = tokenizer.tokenize(norm) || norm.split(/\s+/);
  const filtered = tokens.filter(t => t.length > 1 && !EN_STOP.has(t));
  return filtered.map(t => natural.PorterStemmer.stem(t)).join(' ');
}

// ─── Also load knowledge.json patterns (includes Welsh additions) ──────────
let knowledgeData = [];
try {
  knowledgeData = JSON.parse(fs.readFileSync(KNOWLEDGE_PATH, 'utf8'));
  console.log(`[BayesTrainer] Also loading ${knowledgeData.length} intents from ${KNOWLEDGE_PATH}...`);
} catch (e) {
  console.warn('[BayesTrainer] WARN: could not read knowledge.json:', e.message);
}

// Merge knowledge.json patterns into trainingData (knowledge.json wins for new tags)
for (const intent of knowledgeData) {
  if (!intent.tag || !Array.isArray(intent.patterns)) continue;
  if (!trainingData[intent.tag]) {
    trainingData[intent.tag] = [];
  }
  const existing = new Set(trainingData[intent.tag]);
  for (const p of intent.patterns) {
    if (!existing.has(p)) {
      trainingData[intent.tag].push(p);
      existing.add(p);
    }
  }
}

// ─── Train classifier ──────────────────────────────────────────────────────
console.log('[BayesTrainer] Training NaiveBayes classifier...');
const classifier = new natural.BayesClassifier();
let totalPatterns = 0;
let intentCount   = 0;

for (const [tag, patterns] of Object.entries(trainingData)) {
  if (!Array.isArray(patterns) || patterns.length === 0) continue;
  intentCount++;
  for (const pattern of patterns) {
    const processed = preprocess(pattern);
    if (processed.trim()) {
      classifier.addDocument(processed, tag);
      totalPatterns++;
    }
  }
}

console.log(`[BayesTrainer] Training on ${totalPatterns.toLocaleString()} patterns across ${intentCount} intents...`);
classifier.train();
console.log('[BayesTrainer] Training complete.');

// ─── Serialize to JSON ─────────────────────────────────────────────────────
const serialized = JSON.stringify(classifier, null, 2);
fs.mkdirSync(path.dirname(OUTPUT_PATH), { recursive: true });
fs.writeFileSync(OUTPUT_PATH, serialized, 'utf8');
const fileSizeKB = (fs.statSync(OUTPUT_PATH).size / 1024).toFixed(0);
console.log(`[BayesTrainer] Written to ${OUTPUT_PATH} (${fileSizeKB} KB)`);

// ─── Quick sanity check ────────────────────────────────────────────────────
console.log('\n[BayesTrainer] Quick sanity check (top classification per test query):');
const TEST_QUERIES = [
  'How much are the tuition fees?',
  'I am feeling really anxious about my exams',
  'Can I talk to someone about my dissertation?',
  'I need to submit my assignment but the portal is down',
  'Where is the Carmarthen campus?',
  'Can I apply for student accommodation?',
];

// Reload from disk to test the serialized version
const restoredClassifier = natural.BayesClassifier.restore(
  JSON.parse(fs.readFileSync(OUTPUT_PATH, 'utf8'))
);

for (const q of TEST_QUERIES) {
  const processed = preprocess(q);
  const tag       = restoredClassifier.classify(processed);
  console.log(`  "${q.slice(0, 55)}"`);
  console.log(`    → ${tag}`);
}

console.log(`
✓ Classifier saved to ${OUTPUT_PATH}

Next steps:
  1. lib/nlp.js will auto-load this file if it exists at startup.
     (The code to load it is in the 'Load BayesClassifier sidecar' block.)
  2. To rebuild after adding more patterns, re-run:
       node scripts/build-naivebayes-training.js
  3. Deploy to Vercel — the lib/bayes-classifier.json file is included in
     the deployment bundle because it's in the lib/ directory.
`);
