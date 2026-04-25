/**
 * build-welsh-terms.js
 * Downloads the BydTermCymru / TermCymru dataset from the cwd24/termcymru
 * GitHub repository, extracts every Welsh word, and writes a compact JSON
 * file used by server.js to extend Welsh language detection.
 *
 * Run once locally before deploying:
 *   node scripts/build-welsh-terms.js
 */

const https = require('https');
const fs    = require('fs');
const path  = require('path');

const CSV_URL = 'https://raw.githubusercontent.com/cwd24/termcymru/master/20150522-termcymru.csv';
const OUT     = path.join(__dirname, '..', 'welsh-terms.json');

// Common English words that might slip through — exclude them
const ENGLISH_STOP = new Set([
  'the','and','or','of','in','to','a','an','is','are','was','were',
  'for','with','by','at','on','from','this','that','have','has','not',
  'be','as','it','its','but','if','do','did','so','up','out','can',
  'all','one','two','three','four','five','six','had','they','them',
  'their','our','we','he','she','you','my','me','him','her','us',
  'per','may','use','act','set','new','age','end','day','men','man',
  'yes','no','non','sub','pre','pro','via','see','any','how',
]);

// Welsh characters — their presence is a strong Welsh signal
const WELSH_CHAR = /[âêîôûŵŷäëïöüàèìòùáéíóúãõñçÂÊÎÔÛŴŶ]/;

function fetch(url) {
  return new Promise((resolve, reject) => {
    https.get(url, res => {
      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => resolve(data));
      res.on('error', reject);
    }).on('error', reject);
  });
}

async function main() {
  console.log('Fetching TermCymru CSV…');
  const csv = await fetch(CSV_URL);
  const lines = csv.split('\n');
  console.log(`Downloaded ${lines.length} lines`);

  const wordSet = new Set();

  for (let i = 1; i < lines.length; i++) {           // skip header
    const cols = lines[i].split('\t');
    const welshTerm = (cols[1] || '').trim();
    if (!welshTerm) continue;

    // Split multi-word terms into individual tokens
    const tokens = welshTerm
      .toLowerCase()
      .replace(/[^\wŵŷâêîôûäëïöüàèìòùáéíóúãõñç'\-]/gi, ' ')
      .split(/[\s\-]+/)
      .filter(Boolean);

    for (const tok of tokens) {
      // Keep words ≥ 3 chars, skip pure numbers, skip English stop words
      if (tok.length < 3)               continue;
      if (tok.length > 25)              continue;
      if (/^\d+$/.test(tok))            continue;
      if (/^["'`]/.test(tok))           continue;  // skip quoted/messy tokens
      if (/[^a-zâêîôûŵŷäëïöüàèìòùáéíóúãõñç'\-]/i.test(tok)) continue;
      if (ENGLISH_STOP.has(tok))        continue;
      wordSet.add(tok);
    }
  }

  // Also pull whole phrases ≤ 3 words (useful for pattern matching)
  for (let i = 1; i < lines.length; i++) {
    const cols = lines[i].split('\t');
    const welshTerm = (cols[1] || '').trim();
    if (!welshTerm || welshTerm.startsWith('"')) continue;
    const lower = welshTerm.toLowerCase();
    const wordCount = lower.split(/\s+/).length;
    if (wordCount >= 2 && wordCount <= 3) {
      wordSet.add(lower);
    }
  }

  const words = [...wordSet].sort();
  fs.writeFileSync(OUT, JSON.stringify(words, null, 2), 'utf8');
  console.log(`Written ${words.length} Welsh terms to welsh-terms.json`);
}

main().catch(err => { console.error(err); process.exit(1); });
