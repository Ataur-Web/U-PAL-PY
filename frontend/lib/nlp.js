'use strict';
// NLP pipeline for U-Pal — bilingual TF-IDF + Naive Bayes classifier
// References:
//   natural.js: https://naturalnode.github.io/natural/
//   Stanford CoreNLP: Manning et al. (2014), https://stanfordnlp.github.io/CoreNLP/
//   Ollama: https://ollama.com
//   Morphik Core: https://github.com/morphik-org/morphik-core
//     Document RAG backend — pgvector retrieval, nomic-embed-text embeddings
//   BydTermCymru / TermCymru: Welsh technical terminology dataset (cwd24/termcymru)
//     used to extend the Welsh vocabulary set for language detection
//   Pipeline design based on: Inuwa-Dutse (2023), "Simplifying Student Queries:
//     A Dialogflow-Based Conversational Chatbot for University Websites"

const path    = require('path');
const fs      = require('fs');
const natural = require('natural');
const embed   = require('./embed');
const safety  = require('./safety');
const rerank  = require('./rerank');

// Load knowledge base and Welsh term list once at module level
let TERMCYMRU_WORDS = [];
try {
  TERMCYMRU_WORDS = JSON.parse(
    fs.readFileSync(path.join(process.cwd(), 'welsh-terms.json'), 'utf8')
  );
  console.log(`[TermCymru] Loaded ${TERMCYMRU_WORDS.length} Welsh terms`);
} catch (e) {
  console.warn('[TermCymru] welsh-terms.json not found — using built-in vocab only');
}

// BydTermCymru bilingual map: Welsh term → English term
// Used to augment Welsh Morphik queries so they match English-language scraped content.
// Build/refresh: python3 scripts/build-welsh-map.py
let WELSH_EN_MAP = {};
try {
  const raw = JSON.parse(
    fs.readFileSync(path.join(process.cwd(), 'welsh-bilingual-map.json'), 'utf8')
  );
  // Strip metadata comment
  Object.entries(raw).forEach(([k, v]) => {
    if (!k.startsWith('__') && typeof v === 'string') WELSH_EN_MAP[k] = v;
  });
  console.log(`[BydTermCymru] Loaded ${Object.keys(WELSH_EN_MAP).length} Welsh↔English pairs`);
} catch (e) {
  console.warn('[BydTermCymru] welsh-bilingual-map.json not found — bilingual augmentation disabled');
}

const knowledge = JSON.parse(
  fs.readFileSync(path.join(process.cwd(), 'knowledge.json'), 'utf8')
);

// UWTSD Morphik corpus — harvested from the live Morphik RAG backend by
// scripts/harvest-morphik-corpus.py.  Each entry is a real passage retrieved
// from the 400 ingested UWTSD pages.  This file ships with the deployment so
// the chatbot has a local, always-available knowledge snapshot even when the
// live Morphik tunnel is unreachable from Vercel.
let UWTSD_CORPUS = [];
try {
  const raw = JSON.parse(
    fs.readFileSync(path.join(process.cwd(), 'uwtsd-corpus.json'), 'utf8')
  );
  if (Array.isArray(raw)) UWTSD_CORPUS = raw.filter(
    r => r && typeof r.content === 'string' && r.content.length > 40
  );
  console.log(`[UWTSD corpus] Loaded ${UWTSD_CORPUS.length} passages from Morphik`);
} catch (e) {
  console.warn('[UWTSD corpus] uwtsd-corpus.json not found — run scripts/harvest-morphik-corpus.py on the VM');
}

// NLP tokenizer and stop-word lists (English and Welsh)
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

const CY_STOP = new Set([
  'y','yr','a','ac','ar','at','i','o','am','dan','dros','drwy','heb','tan','wrth',
  'yn','ym','yng','eu','ein','ei','fy','dy','eich','yw','ydy','mae','oedd','roedd',
  'bydd','fydd','bod','oes','sydd','sy','hefyd','dim','nid','os','neu','ond','fel',
  'pan','nad','rwy','rwyf','rydw','dw','dwi','chi','ni','nhw','fe','hi','ef'
]);

function preprocess(text, lang) {
  const norm     = text.toLowerCase().replace(/[^\w\u00C0-\u024F\s]/g,' ').replace(/\s+/g,' ').trim();
  const tokens   = tokenizer.tokenize(norm) || norm.split(/\s+/);
  const stopSet  = lang === 'cy' ? CY_STOP : EN_STOP;
  const filtered = tokens.filter(t => t.length > 1 && !stopSet.has(t));
  return (lang === 'en' ? filtered.map(t => natural.PorterStemmer.stem(t)) : filtered).join(' ');
}

// Welsh vocabulary sets used for language detection
const WELSH_WORDS = new Set([
  'sut','beth','ble','pryd','pam','pwy','faint','pa','sydd',
  'mae','oes','ydy','yw','wyt','bydd','fydd','gall','gallaf','gallwch',
  'hoffwn','hoffech','allaf','allech','allwch','alla','allet','allai',
  'ydych','ydw','caf','cei','ceir',
  'gweld','deall','gwybod','hoffi','gallu','cael','mynd','dod',
  'helpu','gofyn','ateb','siarad','ysgrifennu','darllen',
  'dysgu','mwynhau','cofio','dechrau','gorffen','cysylltu',
  'rhoi','agor','parhau','dewis','cymryd','dweud','gwneud',
  'ddod','daeth','daw','ddaw','doi','deuaf',
  'wnaf','wnaiff','wnawn','wnewch',
  'ddod','ddweud','ddarllen','ddysgu','ddechrau','ddeall',
  'ngweld','ngwybod','ngwneud',
  'gyda','gydag','drwy','trwy','dros','rhwng','oherwydd','achos',
  'wrth','gan','dan','tan','heb','mewn','tu','oddi','uwch','islaw','gerllaw','ger','agos',
  'nawr','hefyd','ond','neu','fel','pan','oni','tra',
  'felly','wedyn','eto','rwan','serch','eithr',
  'mwy','llai','mawr','bach','newydd','gwael',
  'hawdd','anodd','cyflym','araf','hir','byr','uchel','isel',
  'pwysig','diddorol','prysur','rhad','drud','agored','caeedig',
  'fi','mi','ti','chi','nhw','fe','ni','ef',
  'fy','dy','ei','ein','eich','eu',
  'hwn','hon','hyn','hynny','yr','rhai','peth',
  'dyna','cyfan','pob','dim','nid','rhaid','digon',
  'popeth','rhywbeth','unrhyw','nôl','ymlaen',
  'holl','oll','lan','lawr','yno','yma',
  'croeso','ardderchog','gwych','bendigedig','perffaith',
  'iawn','ocê','siŵr',
  'shwmae','diolch','hwyl','cymraeg','cymru','pcydds',
  'hyd','gael','goll',
  'myfyriwr','myfyrwyr','prifysgol','cwrs','cyrsiau','llety','llyfrgell',
  'gofynion','mynediad','ffioedd','cymorth','lles','anabledd',
  'graddio','canlyniadau','amserlen','argraffu','gwasanaethau',
  'eisiau','isio','moyn','angen','wneud',
  'cyfeiriad','cyswllt','ymgeisio','derbyniadau','benthyciad',
  'ysgoloriaeth','bwrsari','neuaddau','preswyl','campysau',
  'graddau','marciau','arholiad','aseiniad','tymor','modiwl',
  'darlith','darlithydd','tiwtorial','adborth','cofrestru',
  'e-bost','ebost','neges','galwad','ffonio','anfon',
  'tiwtoriaid','tiwtor','athro','athrawes','darlithwyr',
  'adran','adrannau','adrannu',
]);

const WELSH_STRONG = new Set([
  'shwmae','shwmai','shw','helo','helô',
  'diolch','diolchwch','diolchus',
  'hwyl','hwylfawr',
  'bore','prynhawn','noswaith','nos',
  'dwi','rwyf','rwy','dwin','ydw','ydy','yw',
  "dwi'n","rwy'n","ti'n","wedi'n","mae'n","oes'n",
  "hynny'n","sydd'n",
  'cymraeg','cymru','pcydds',
  'caerfyrddin','nghaerfyrddin','gaerfyrddin','abertawe','llambed','caerdydd',
  'myfyriwr','myfyrwyr','prifysgol','llyfrgell','llety',
  'ffioedd','bwrsari','bwrsariaeth','neuaddau','preswyl',
  'cwrs','cyrsiau','modiwl','darlith','darlithydd','tiwtorial',
  'aseiniad','arholiad','graddio','cofrestru',
  'anabledd','dyslecsia','lles','cymorth',
  'rhaglen','astudio','ymgeisio','derbyniadau',
  'ddefnyddio','ddefnyddiol','ddod','ddweud','ddarllen',
  'ddeall','ddysgu','ddechrau','ngweld','ngwybod',
  'angenrheidiol','gwybodaeth',
  'hyn','hynny','hwn','hon','honno','yna','yma',
  'fwrdd','ddim','fawr',
  'llongyfarchiadau','llongyfarch','llongyfarchaf',
  'dyna','cyfan','croeso','ardderchog','gwych','bendigedig',
  'hefyd','wrth','oherwydd','newydd','rhaid',
  'gweld','gwybod','hoffi','gallu','deall',
  'popeth','rhywbeth','unrhyw','digon','felly',
  'nôl','ymlaen','parhau','cysylltu',
  'e-bost','ebost','neges',
  'ymchwil','cyflogaeth','gwirfoddoli',
  'derbyn','dewis','gorffen','parhau',
  'tiwtoriaid','tiwtor','darlithwyr','adran',
  // Question words (no English collision; "pa" is safe because pure "pa"
  // doesn't appear in everyday English queries).
  'sut','pam','ble','pryd','pwy','pa','faint','beth','paham','sawl',
  // Common verb forms that are uniquely Welsh (gallu, cael, mynd, dod,
  // gweld, ...).  Inflected forms only — base forms like "gallu" are
  // already above.
  'allaf','alla','gallaf','galla','gawn','cawn','gaf','caf',
  'wnaf','wna','wnân','wnaeth','wnaethoch','wnewch',
  'oedd','oeddwn','oeddet','oeddech','oeddem','oedden',
  'fydd','bydd','fyddai','byddai','fyddan','byddan','fyddwn','byddwn',
  'fyddech','byddech','fydden','bydden','fyddi','byddi','fyddaf','byddaf',
  'euthum','aethost','aeth','aethom','aethoch','aethant',
  'ddaeth','ddoi','ddoist','ddown','ddewch','ddônt',
  'welais','welodd','welodd','welwch','welant','welwn',
  // Common Welsh prepositions + conjunctions with no English collision.
  // (Bare "a", "i", "o" overlap with English so stay OUT.)
  'rhwng','gyda','gyd','hebddo','hebddi','heb','drosof','arnaf','ataf',
  'oddi','drosodd','trosodd','ynddo','ynddi','ohoni','ohono',
  // Welsh-specific content verbs / nouns that show up in student queries.
  'teithio','deithio','deithiais','deithiodd','fyw','byw','chwilio',
  'weithio','gweithio','dalu','dalais','dalodd','talu','prynu',
  'gyrraedd','cyrraedd','agor','cau','ffurflen','ffurflenni',
  // Campus-related Welsh spellings
  'campws','gampws','ngampws','nghampws',
]);

const ENGLISH_HINTS = new Set([
  'how','where','when','why','what','which','who','whose',
  'you','your','yours','my','mine','our','ours','their',
  'this','that','these','those','there','here',
  'me','he','she','it','we','they','us','him','her','them',
  'can','could','would','should','will','shall','may','might','must',
  'do','does','did','doing','done',
  'have','has','had','having',
  'is','are','was','were','been','being',
  'get','got','getting','make','makes','made','take','took',
  'go','goes','going','went','gone','come','comes','coming','came',
  'see','saw','seen','look','looked','looking','looks',
  'use','used','using','uses','try','tried','trying','tries',
  'ask','asked','asking','tell','told','telling','know','knew','known',
  'think','thought','thinking','feel','felt','feeling','work','works','worked',
  'speak','spoke','spoken','speaking','say','says','said','saying',
  'flip','flipped','flipping','switch','switched','switching',
  'back','just','still','already','now','then','too','also','again',
  'actually','really','very','quite','pretty','rather','instead',
  'ok','okay','yeah','yes','no','yep','nope','sure','alright',
  'the','a','an','and','or','but','if','so','nor','yet',
  'with','from','about','into','onto','over','under','after','before',
  'because','though','although','while','unless','until','since',
  'between','through','during','without','within','along','across',
  'in','on','at','by','up','off','out','down','for','of','to',
  'help','helps','helping','please','thanks','thank','hello','hi','hey',
  'need','needs','want','wants','looking','find','tell','show',
  'know','knows','say','says','said',
  'thing','things','something','anything','nothing','everything',
  'time','day','week','month','year','today','tomorrow','yesterday',
  'way','ways','place','part','man','woman','people','person',
  'good','bad','new','old','big','small','long','short','right','wrong',
  'first','last','next','any','some','more','most','other','another',
  'same','different','better','best','worse','worst',
  'able','unable','ready','sure','clear','hard','easy',
  'let','lets','put','set','keep','kept','start','started','stop','stopped',
  'give','gave','given','take','took','taken','send','sent','sending',
  'check','checked','checking','open','opened','close','closed',
  'apply','application','applicant','hardship','bursary','accommodation',
  'library','financial','english','welsh','explain','question','answer',
  'student','students','university','campus','course','courses',
  'email','password','login','account','system','portal','website',
  'fee','fees','support','service','services','staff','tutor',
  'module','assignment','result','grade','exam','timetable','semester',
]);

// Merge BydTermCymru single-word terms (welsh-terms.json)
for (const term of TERMCYMRU_WORDS) {
  if (!term.includes(' ') && term.length <= 20 && !ENGLISH_HINTS.has(term)) {
    WELSH_WORDS.add(term);
  }
}

// Merge Welsh keys from the bilingual map (welsh-bilingual-map.json)
// This expands Welsh language detection to cover all university/education vocabulary.
for (const wTerm of Object.keys(WELSH_EN_MAP)) {
  const tokens = wTerm.split(/\s+/);
  // Single-word terms go into the detection vocabulary
  if (tokens.length === 1 && tokens[0].length >= 2 && tokens[0].length <= 25
      && !ENGLISH_HINTS.has(tokens[0])) {
    WELSH_WORDS.add(tokens[0]);
  }
}
console.log(`[Welsh detection] Vocabulary: ${WELSH_WORDS.size} words (includes BydTermCymru)`);

function detectLanguage(text, runningLang) {
  const lower  = text.toLowerCase();
  const words  = lower.replace(/[^a-z\u00C0-\u024F\s']/g,' ').split(/\s+/).filter(Boolean);
  let welshTotal = 0, welshStrong = 0, englishHits = 0;
  // Track which tokens we couldn't classify — only THOSE are candidates
  // for the English-morphology regex.  This stops Welsh place names and
  // verbs that happen to end in -ed, -ing, -ly etc. (e.g. "Llambed",
  // "teithio" → n/a but "cofrestru" → n/a; "dosbarth-ed" variants) from
  // being mis-counted as English past-tense forms.
  const unclassified = [];
  for (const w of words) {
    if (WELSH_STRONG.has(w))  { welshStrong++; welshTotal++; continue; }
    if (ENGLISH_HINTS.has(w)) { englishHits++; continue; }
    if (WELSH_WORDS.has(w))   { welshTotal++; continue; }
    unclassified.push(w);
  }
  const hasWelshDiacritic = /[ŵŷâêîôûẁỳẃýÿï]/i.test(text);
  const englishMorph = unclassified.filter(w =>
    w.length >= 4 && /(ing|tion|tions|ings|ness|ment|ments|ed|ly)$/.test(w)
  ).length;
  const englishSignal = englishHits + englishMorph;
  const isPureAscii   = !/[\u00C0-\u024F]/.test(text);
  const isShortQuery  = words.length <= 5;

  // ── Decision ladder ──────────────────────────────────────────────────────
  // Strong Welsh markers take priority.  Two strong markers → Welsh, full
  // stop.  One strong marker beats or ties the English signal → Welsh.
  if (welshStrong >= 2) return 'cy';
  if (welshStrong >= 1 && welshStrong >= englishSignal) return 'cy';

  // Diacritic anywhere → Welsh unless the English signal is overwhelming.
  if (hasWelshDiacritic && englishSignal <= 1) return 'cy';

  // Welsh vocabulary wins clearly.
  if (welshTotal >= 3 && welshTotal > englishSignal) return 'cy';
  if (welshTotal >= 2 && englishSignal === 0) return 'cy';

  // Short pure-ASCII queries ("sut mae?", "beth yw?") — any Welsh hit and
  // no English hits → Welsh.  A genuine English query of ≤5 words almost
  // always contains at least one ENGLISH_HINT (is/what/the/...).
  if (isShortQuery && welshTotal >= 1 && englishSignal === 0) return 'cy';

  // Clear English signals.
  if (englishSignal >= 2 && welshTotal === 0) return 'en';
  if (englishSignal > welshTotal) return 'en';

  // Running-language tie-breaker: respect the conversation's existing
  // language if the current turn is ambiguous (common for one-word
  // follow-ups like "ie" / "na").
  if (runningLang === 'cy' && welshTotal >= 1) return 'cy';
  if (runningLang === 'en' && englishSignal >= 1) return 'en';

  // Final tie-breaker: prefer Welsh when we saw ANY Welsh evidence that
  // at least matches the English evidence.  Previously this bucket
  // returned 'en' by default, which misrouted pure-ASCII Welsh queries.
  if (welshTotal >= 1 && welshTotal >= englishSignal) return 'cy';
  return 'en';
}

// Quick sanity check: given an LLM reply, does it actually look Welsh?
// Used as a guard after native-Welsh generation — if llama3.1 slipped back
// into English (common when the system prompt is long and the context is
// all English), we fall back to the pivot-through-English+MyMemory path.
// Heuristic: count Welsh function words, Welsh-specific digraphs (ll/dd/
// ff/ch/rh/ng), and diacritics.  A real Welsh reply of 2-4 sentences
// should hit several of these; an English reply will hit almost none.
function looksWelsh(text) {
  if (!text || text.length < 20) return false;
  const lower = text.toLowerCase();
  const hasDiacritic = /[ŵŷâêîôûẁỳẃýÿï]/.test(lower);
  const functionWords = (lower.match(/\b(yw|ydy|ydw|mae|y|yr|yn|ar|i|o|a'r|neu|sy|sydd|eich|ein|dy|fy|gallaf|gallwch|gellir|hoffech|dydw|ddim|gan|gyda|am|drwy|hyn|hwn|hon|dyna|dyma|ac|ond|os|pob|bob|hefyd|iawn|rhyngwladol|cartref|myfyrwyr|ffioedd|llety|llyfrgell|cymorth|cysylltwch|ewch|defnyddia)\b/g) || []).length;
  const welshDigraphs = (lower.match(/\b\w*(ll|dd|ff|ch|rh|ng|wy|ys)\w*\b/g) || []).length;
  const welshScore = functionWords * 2 + welshDigraphs + (hasDiacritic ? 3 : 0);
  // English-only tells: "the", "and", "of", "is", "are" etc. without Welsh
  // function words nearby means the model ignored the Welsh prompt.
  const englishHits = (lower.match(/\b(the|and|of|is|are|for|with|to|from|you|your|can|please|please contact|students|university)\b/g) || []).length;
  return welshScore >= 6 && welshScore >= englishHits;
}

// Detect rambling / looping LLM output.  llama3.1:8b occasionally generates
// Welsh text that repeats the same sentence 3-4 times with tiny variations —
// classic small-model failure mode when temperature is low and the Welsh
// system prompt strains its token distribution.  We look for:
//   (a) the same long n-gram (>= 6 tokens) appearing twice or more, or
//   (b) two whole sentences being >= 80% identical.
// If either fires we treat the reply as broken and trigger the pivot
// fallback (English LLM + MyMemory translation) which is more reliable.
function hasRepetition(text) {
  if (!text || text.length < 100) return false;
  const clean = text.toLowerCase().replace(/\s+/g, ' ').trim();

  // (a) repeated 6-token window
  const tokens = clean.split(' ').filter(Boolean);
  if (tokens.length >= 12) {
    const seen = new Map();
    for (let i = 0; i <= tokens.length - 6; i++) {
      const gram = tokens.slice(i, i + 6).join(' ');
      const count = (seen.get(gram) || 0) + 1;
      seen.set(gram, count);
      if (count >= 2) return true;
    }
  }

  // (b) two sentences >= 80% token-overlap
  const sentences = clean.match(/[^.!?…\n]+[.!?…]?/g) || [];
  const longSents = sentences.map(s => s.trim()).filter(s => s.split(' ').length >= 5);
  for (let i = 0; i < longSents.length; i++) {
    const aTokens = new Set(longSents[i].split(' '));
    for (let j = i + 1; j < longSents.length; j++) {
      const bTokens = longSents[j].split(' ');
      const overlap = bTokens.filter(t => aTokens.has(t)).length;
      const ratio   = overlap / Math.max(aTokens.size, bTokens.length);
      if (ratio >= 0.80) return true;
    }
  }
  return false;
}

// Train TF-IDF index and Naive Bayes classifier on all knowledge-base patterns
const tfidf     = new natural.TfIdf();
const intentMap = [];

knowledge.forEach(intent => {
  intent.patterns.forEach(pattern => {
    tfidf.addDocument(preprocess(pattern, detectLanguage(pattern)));
    intentMap.push(intent.tag);
  });
});

// Load BayesClassifier sidecar (pre-trained from OpenOrca augmented patterns).
// If lib/bayes-classifier.json exists (built by scripts/build-naivebayes-training.js
// after running scripts/train_from_openorca.py), restore it — it has hundreds of
// real-world student question phrasings per intent instead of the 10–40 hand-written
// examples in knowledge.json, so intent classification is significantly more robust.
// Falls back gracefully to training inline from knowledge.json if the file is absent.
let bayesClassifier;
try {
  const sidecarPath = path.join(process.cwd(), 'lib', 'bayes-classifier.json');
  const sidecarRaw  = fs.readFileSync(sidecarPath, 'utf8');
  bayesClassifier   = natural.BayesClassifier.restore(JSON.parse(sidecarRaw));
  console.log('[Bayes] Loaded pre-trained classifier from lib/bayes-classifier.json');
} catch (_) {
  // Sidecar not present — train from knowledge.json patterns (default path)
  bayesClassifier = new natural.BayesClassifier();
  knowledge.forEach(intent => {
    intent.patterns.forEach(pattern => {
      bayesClassifier.addDocument(preprocess(pattern, detectLanguage(pattern)), intent.tag);
    });
  });
  bayesClassifier.train();
  console.log('[Bayes] Trained inline classifier from knowledge.json');
}

const THRESHOLD_FALLBACK = 0.07;
const THRESHOLD_CLARIFY  = 0.20;

function getTopIntents(msg, lang, n = 5, history) {
  let retrievalText = msg;
  if (Array.isArray(history) && history.length > 0) {
    const userTurns = history.filter(t => t.role === 'user').slice(-2);
    retrievalText = [...userTurns.map(t => t.text), msg].join(' ');
  }
  const processed = preprocess(retrievalText, lang);
  const scores = [];
  tfidf.tfidfs(processed, (i, score) => scores.push({ i, score }));
  scores.sort((a, b) => b.score - a.score);
  const seen = new Set(), results = [];
  for (const { i, score } of scores) {
    const tag = intentMap[i];
    if (!seen.has(tag) && score > 0.05) {
      seen.add(tag);
      const intent = knowledge.find(k => k.tag === tag);
      if (intent) results.push({ tag, score, intent });
    }
    if (results.length >= n) break;
  }
  return results;
}

function getResponse(tag, lang) {
  const intent = knowledge.find(i => i.tag === tag);
  if (!intent) return null;
  const pool = intent.responses[lang] || intent.responses['en'];
  return pool[Math.floor(Math.random() * pool.length)];
}

// Fallback and clarification response strings (bilingual)
const FALLBACK = {
  en: "Hey! I'm U-Pal, your UWTSD student assistant — I didn't quite catch what you needed there. What can I help you with? Whether it's fees, accommodation, courses, IT, campus info, or wellbeing, just ask!",
  cy: "Hei! U-Pal ydw i, eich cynorthwyydd myfyrwyr PCYDDS — doeddwn i ddim yn siŵr beth oeddet ti angen yn fanno. Beth alla i dy helpu di ag ef? Gofynna unrhyw beth am ffioedd, llety, cyrsiau, TG, campysau, neu les!"
};

const CLARIFICATION = {
  en: "Sounds like something's on your mind — what's going on? I'm here to help with anything UWTSD-related 😊",
  cy: "Mae'n swnio fel bod rhywbeth ar dy feddwl — beth sy'n digwydd? Rwy'n yma i helpu gyda phopeth PCYDDS 😊"
};

const CRISIS_KEYWORDS = [
  'going to die','want to die','wanna die','want to kill myself','kill myself',
  'end my life','end it all','take my life','take my own life',
  'dont want to live',"don't want to live","don't want to be alive",
  'no reason to live','not worth living','life not worth',
  'thinking about suicide','thinking of suicide','suicidal','suicide',
  'self harm','self-harm','selfharm','hurt myself','harm myself',
  'overdose','cutting myself','in crisis','mental health crisis',
  'having a breakdown','cant cope anymore',"can't cope anymore",
  'breaking down','losing my mind',
  'eisiau marw','eisiau lladd fy hun','lladd fy hun','diwedd fy mywyd',
  'meddyliau hunanladdol','hunan-niweidio','hunan niweidio',
  'argyfwng','alla i ddim ymdopi'
];
function isCrisis(text) {
  const lower = text.toLowerCase();
  return CRISIS_KEYWORDS.some(kw => lower.includes(kw));
}

// Stanford CoreNLP integration — NER, lemmatisation, and sentiment scoring
const CORENLP_URL = (process.env.CORENLP_URL || '').replace(/\/$/, '');

const NER_INTENT_BOOST = {
  swansea:    ['campuses_swansea','library_swansea','accommodation_general','accommodation_cost'],
  abertawe:   ['campuses_swansea','library_swansea','accommodation_general','accommodation_cost'],
  carmarthen: ['campuses_carmarthen','library_carmarthen','accommodation_general','accommodation_cost'],
  caerfyrddin:['campuses_carmarthen','library_carmarthen','accommodation_general','accommodation_cost'],
  lampeter:   ['campuses_lampeter','library_lampeter','accommodation_general','accommodation_cost'],
  llambed:    ['campuses_lampeter','library_lampeter','accommodation_general','accommodation_cost'],
  accommodation:['accommodation_general','accommodation_cost'],
  llety:      ['accommodation_general','accommodation_cost'],
  neuaddau:   ['accommodation_general','accommodation_cost'],
  library:    ['library','library_swansea','library_carmarthen','library_lampeter'],
  llyfrgell:  ['library','library_swansea','library_carmarthen','library_lampeter'],
  wifi:       ['it_wifi'], moodle:['it_moodle'], mytsd:['it_portal'],
  wellbeing:  ['wellbeing_general'], lles:['wellbeing_general'],
  disability: ['wellbeing_disability'], anabledd:['wellbeing_disability'],
  graduation: ['graduation'], graddio:['graduation'],
  fees:       ['fees_tuition','fees_student_loan','fees_scholarships'],
  ffioedd:    ['fees_tuition','fees_student_loan','fees_scholarships'],
  bursary:    ['fees_scholarships','financial_support'],
  bwrsari:    ['fees_scholarships','financial_support'],
  timetable:  ['timetable'], amserlen:['timetable'],
  printing:   ['printing'], argraffu:['printing'],
  union:      ['students_union'],
};

const SENTIMENT_CONCERN_THRESHOLD = 1.2;

async function enrichWithCoreNLP(text, lang) {
  if (!CORENLP_URL) return null;
  const annotators = lang === 'en' ? 'tokenize,ssplit,pos,lemma,ner,sentiment' : 'tokenize,ssplit';
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 12000);
    const res = await fetch(
      `${CORENLP_URL}?annotators=${encodeURIComponent(annotators)}&outputFormat=json&properties=${encodeURIComponent(JSON.stringify({ pipelineLanguage: 'en' }))}`,
      { method: 'POST', body: text, signal: controller.signal,
        headers: { 'Content-Type': 'text/plain; charset=UTF-8' } }
    );
    clearTimeout(timeout);
    if (!res.ok) return null;
    const data = await res.json();
    const sentences = data.sentences || [];
    const entities = [];
    const lemmas = [];
    let sentimentSum = 0, sentimentCount = 0;
    for (const sent of sentences) {
      if (sent.sentimentValue !== undefined) {
        sentimentSum += Number(sent.sentimentValue);
        sentimentCount++;
      }
      for (const token of (sent.tokens || [])) {
        if (token.lemma) lemmas.push(token.lemma.toLowerCase());
      }
      if (sent.entitymentions) {
        for (const e of sent.entitymentions) {
          if (!['O','DURATION','NUMBER','ORDINAL','MONEY','PERCENT'].includes(e.ner)) {
            entities.push({ text: e.text, type: e.ner, norm: e.normalizedNER || e.text });
          }
        }
      }
    }
    return {
      entities,
      lemmas,
      sentimentScore: sentimentCount > 0 ? sentimentSum / sentimentCount : 2,
    };
  } catch (e) {
    console.warn('[CoreNLP] unavailable:', e.message);
    return null;
  }
}

function welshNER(text) {
  const lower   = text.toLowerCase();
  const words   = lower.replace(/[^\w\u00C0-\u024F\s']/g, ' ').split(/\s+/).filter(Boolean);
  const entities = [];
  const WELSH_NER_PATTERNS = [
    { pattern: /\b(abertawe|swansea)\b/,                                          type: 'LOCATION', norm: 'swansea'     },
    { pattern: /\b(caerfyrddin|caerfyrdd|nghaerfyrddin|gaerfyrddin|carmarthen)\b/, type: 'LOCATION', norm: 'carmarthen'  },
    { pattern: /\b(llambed|lampeter|llanbed)\b/,                                  type: 'LOCATION', norm: 'lampeter'    },
    { pattern: /\b(aberystwyth)\b/,                                                type: 'LOCATION', norm: 'aberystwyth' },
    { pattern: /\b(moodle|mwdl)\b/,       type: 'SYSTEM',   norm: 'moodle'     },
    { pattern: /\b(mytsd|my tsd)\b/,      type: 'SYSTEM',   norm: 'mytsd'      },
    { pattern: /\b(wifi|wi-fi|rhyngrwyd)\b/, type: 'SYSTEM', norm: 'wifi'      },
    { pattern: /\b(llyfrgell|library)\b/, type: 'FACILITY', norm: 'library'    },
    { pattern: /\b(llety|neuaddau|halls)\b/, type: 'FACILITY', norm: 'accommodation' },
    { pattern: /\b(ffioedd|fees?)\b/,     type: 'TOPIC',    norm: 'fees'       },
    { pattern: /\b(lles|wellbeing)\b/,    type: 'TOPIC',    norm: 'wellbeing'  },
    { pattern: /\b(graddio|graduation)\b/,type: 'TOPIC',    norm: 'graduation' },
    { pattern: /\b(bwrsari|bursary)\b/,   type: 'TOPIC',    norm: 'bursary'    },
    { pattern: /\b(anabledd|disability)\b/,type: 'TOPIC',   norm: 'disability' },
    { pattern: /\b(argraffu|printing)\b/, type: 'TOPIC',    norm: 'printing'   },
    { pattern: /\b(amserlen|timetable)\b/,type: 'TOPIC',    norm: 'timetable'  },
    { pattern: /\b(undeb|union)\b/,       type: 'TOPIC',    norm: 'union'      },
  ];
  for (const { pattern, type, norm } of WELSH_NER_PATTERNS) {
    const m = lower.match(pattern);
    if (m) entities.push({ text: m[0], type, norm });
  }
  const negWords = ['ofnadwy','methu','ffaelu','pryder','straen','anhapus','pryderus','trist','ofni','poeni'];
  const posWords = ['gwych','ardderchog','da','hapus','diolch','perffaith','iawn','bendigedig','croeso'];
  const neg = words.filter(w => negWords.includes(w)).length;
  const pos = words.filter(w => posWords.includes(w)).length;
  const sentimentScore = Math.max(0, Math.min(4, 2 - neg + pos));
  return { entities, lemmas: words, sentimentScore };
}

function applyNERBoost(topIntents, entities = []) {
  if (!entities || entities.length === 0) return topIntents;
  const boosted = new Set();
  for (const entity of entities) {
    const norm = (entity.norm || entity.text).toLowerCase();
    for (const [key, tags] of Object.entries(NER_INTENT_BOOST)) {
      if (norm.includes(key) || key.includes(norm)) tags.forEach(t => boosted.add(t));
    }
  }
  return topIntents.map(intent => ({
    ...intent,
    score: boosted.has(intent.tag) ? intent.score * 1.4 : intent.score
  })).sort((a, b) => b.score - a.score);
}

function rescoreWithLemmas(topIntents, lemmas, lang) {
  if (lang !== 'en' || !lemmas || lemmas.length === 0) return topIntents;
  const lemmaQuery = lemmas.join(' ');
  const lemmaScores = {};
  tfidf.tfidfs(lemmaQuery, (i, score) => {
    const tag = intentMap[i];
    if (score > (lemmaScores[tag] || 0)) lemmaScores[tag] = score;
  });
  return topIntents.map(intent => ({
    ...intent,
    score: Math.max(intent.score, lemmaScores[intent.tag] || 0),
  }));
}

function buildNLPContext(enrichment, lang) {
  if (!enrichment) return '';
  const lines = [];
  if (enrichment.entities.length > 0) {
    lines.push(`Entities detected: ${enrichment.entities.map(e => `${e.text} [${e.type}]`).join(', ')}`);
  }
  if (lang === 'en' && enrichment.sentimentScore !== undefined) {
    const labels = ['very negative','negative','neutral','positive','very positive'];
    const label = labels[Math.round(Math.min(4, Math.max(0, enrichment.sentimentScore)))];
    lines.push(`Student tone: ${label}`);
  }
  return lines.length ? '\nNLP ANNOTATIONS:\n' + lines.join('\n') : '';
}

// Ollama LLM integration — rephrase knowledge-base answers via local model
const OLLAMA_URL     = (process.env.OLLAMA_URL   || '').replace(/\/$/, '');
// Default: llama3.1:8b-instruct-q5_K_M (~5.7 GB VRAM) — best quality that
// comfortably fits a 12 GB card with headroom for the context window.
// Override via OLLAMA_MODEL env var (e.g. q4_K_M for less VRAM, q8_0 for max quality).
const OLLAMA_MODEL   = process.env.OLLAMA_MODEL  || 'llama3.1:8b-instruct-q5_K_M';
// num_ctx controls the context window size (tokens). Smaller = less VRAM.
// 4096 works well for Q5_K_M on 12 GB VRAM.
const OLLAMA_NUM_CTX = parseInt(process.env.OLLAMA_NUM_CTX || '4096', 10);

// Morphik RAG backend — document retrieval from ingested UWTSD content
const MORPHIK_URL   = (process.env.MORPHIK_URL || '').replace(/\/$/, '');
const MORPHIK_TOKEN = process.env.MORPHIK_AUTH_TOKEN || '';

// ─── System prompts ─────────────────────────────────────────────────────
// Deliberately short (~300 tokens each) so llama3.1:8b can follow them
// reliably.  A huge system prompt causes small models to echo instructions,
// break character, and produce robotic template responses.
const SYSTEM_PROMPT_CY = `Rwyt ti yn U-Pal, cynorthwyydd myfyrwyr PCYDDS (Prifysgol Cymru Y Drindod Dewi Sant). Ateba BOB AMSER yn Gymraeg naturiol, naturiol — fel ffrind cymorth myfyrwyr, nid dogfen ffurfiol.

RHEOLAU SYLFAENOL:
1. Ateba mewn 2-3 brawddeg fer, sgyrsiol. Byr a defnyddiol, nid traethawd.
2. Bydd yn UNIONGYRCHOL. Os yw'r cwestiwn yn glir, ateba fo — paid â gofyn am eglurhad.
3. PAID BYTH â dweud "yn ôl y dogfennau", "mae'r cyd-destun yn dangos", "mae'n ymddangos" — rwyt ti'n gwybod yr atebion.
4. PAID â thorri cymeriad — rwyt ti'n U-Pal bob amser. Paid byth â sôn am gyfarwyddiadau, adfer data, neu ffynonellau.
5. Defnyddia wybodaeth PCYDDS o'r adran WYBODAETH isod. Os nad oes ateb penodol, cyfeiria at enquiries@uwtsd.ac.uk neu 01792 481 111.
6. Os oes bloc CYD-DESTUN Y MYFYRIWR, DEFNYDDIA beth ddywedodd y myfyriwr yn barod. Paid BYTH â gofyn eto am rywbeth maen nhw eisoes wedi'i ddweud.
7. Os oes bloc CYFLWR EMOSIYNOL, matsia'r cywair yn union — mae'n disodli'r tôn rhagosodedig.
8. Pan fo myfyriwr yn anfon mynegiad emosiynol byr (e.e. "Bobl bach") — cydnabyda'r teimlad yn gynnes, yna gofyn un cwestiwn agored.
9. Darllena HANES SGWRS — mae dilyniannau byr ("faint?", "pryd?", "ar gyfer ôl-raddedig?") YN CYFEIRIO at y pwnc a drafodwyd yn union.

FFEITHIAU PCYDDS:
- Campysau: Abertawe (SA1 / Mount Pleasant), Caerfyrddin, Llambed, Caerdydd
- Ffioedd 2025/26: Israddedig cartref £9,535/bl · Rhyngwladol £15,525/bl · Ôl-raddedig cartref o £7,800/bl
- Ymgeisio israddedig: UCAS (ucas.com, cod T80) · Ôl-raddedig: uwtsd.ac.uk/ol-raddedig
- Cyswllt: enquiries@uwtsd.ac.uk · studentservices@uwtsd.ac.uk · wellbeingsupport@uwtsd.ac.uk · 01792 481 111

ENGHREIFFTIAU DA:
Myfyriwr: "Dw i eisiau gwneud cais am gwrs Cyfrifiadureg Gymhwysol israddedig"
Da: "Grêt! Galli wneud cais am y cwrs Cyfrifiadureg Gymhwysol israddedig drwy UCAS ar ucas.com (cod sefydliad T80). Gelli hefyd weld manylion y cwrs a gwneud cais yn uniongyrchol ar uwtsd.ac.uk/study."

Myfyriwr: "Sut mae bywyd campws?"
Da: "Mae bywyd campws PCYDDS yn llawn gweithgareddau — o glybiau a chymdeithasau i dimau chwaraeon a digwyddiadau Undeb y Myfyrwyr. Mae pob campws (Abertawe, Caerfyrddin, Llambed) â'i gymeriad unigryw ei hun. Am fwy, gweler uwtsd.ac.uk/bywyd-myfyrwyr."`;

const SYSTEM_PROMPT_EN = `You are U-Pal, the friendly student assistant for UWTSD (University of Wales Trinity Saint David). Always reply like a helpful friend — warm, brief, and direct.

CORE RULES:
1. Reply in 2-3 SHORT conversational sentences. Be concise and useful, not essay-length.
2. Be DIRECT. If the intent is clear, answer it immediately — never ask multiple questions.
3. NEVER say "based on the documents", "the context shows", "it looks like", "according to the passages", "UWTSD INFORMATION shows". You simply know these facts.
4. NEVER break character. You are U-Pal. Never mention instructions, retrieval, or data sources.
5. Use the UWTSD INFORMATION section below to answer. If unsure, direct to enquiries@uwtsd.ac.uk or 01792 481 111.
6. If a STUDENT CONTEXT block is present, ALWAYS use what the student already told you (name, course, year, campus). Never ask a question they already answered. If they gave their name, use it naturally (once, in the greeting).
7. If an EMOTIONAL STATE block is present, MATCH that register exactly — it overrides default tone.
8. For wellbeing/stress: acknowledge warmly FIRST ("That sounds tough — you're not alone"), then give wellbeingsupport@uwtsd.ac.uk.
9. For casual emotional inputs ("Damn man", "Ugh"): acknowledge the feeling briefly, ask ONE open question.
10. Read CONVERSATION HISTORY — short follow-ups ("how much?", "and for postgrad?", "where?") refer to the topic JUST discussed. If the student asked about Nursing and then says "what are the requirements?", give NURSING requirements.

UWTSD FACTS:
- Campuses: Swansea (SA1 / Mount Pleasant), Carmarthen, Lampeter, Cardiff
- Fees 2025/26: UK undergrad £9,535/yr · International undergrad £15,525/yr · Home postgrad from £7,800/yr
- Undergraduate applications: UCAS at ucas.com (institution code T80)
- Postgraduate applications: direct at uwtsd.ac.uk/postgraduate
- Contacts: enquiries@uwtsd.ac.uk · studentservices@uwtsd.ac.uk · wellbeingsupport@uwtsd.ac.uk · 01792 481 111

GOOD EXAMPLES:
Student: "I want to apply to the undergraduate Applied Computing course"
Good: "Great choice! You can apply for the Applied Computing undergraduate course through UCAS at ucas.com using institution code T80. You can also check course details and apply directly at uwtsd.ac.uk/study."

Student: "How is campus life?"
Good: "Campus life at UWTSD is vibrant — there are clubs, societies, sports teams, and regular events run by the Students' Union. Each campus (Swansea, Carmarthen, Lampeter) has its own character. Find out more at uwtsd.ac.uk/student-life."

Student: "Damn man"
Good: "Sounds like things might be a bit rough — what's going on? Happy to help 😊"

Student: "I'm really stressed about my dissertation"
Good: "Dissertation stress is completely normal, especially near deadlines — you're not alone. Your supervisor and Personal Tutor are there to support you academically, and UWTSD's Wellbeing team is always available at wellbeingsupport@uwtsd.ac.uk or 01792 481 111."`;

// Returns the appropriate system prompt for the user's language.
function getSystemPrompt(lang) {
  return lang === 'cy' ? SYSTEM_PROMPT_CY : SYSTEM_PROMPT_EN;
}

// Pick a sampling temperature based on what kind of question this is:
// factual questions (fees, dates, emails, URLs, procedures) reward low
// temperature — we want the LLM to repeat the retrieved facts verbatim.
// Empathetic / open-ended questions (wellbeing, advice, "what's it like
// to live in Swansea") reward a warmer sampler so the reply sounds human
// and not like a spec sheet.
function pickTemperature(text) {
  const t = (text || '').toLowerCase();
  if (/\b(fee|fees|cost|price|tuition|deadline|apply|application|when|date|open|close|hours?|phone|email|address|postcode|how many|how much|list)\b/.test(t)) {
    return 0.15;
  }
  if (/\b(feel|worry|anxious|stressed|lonely|sad|homesick|help me|advice|what should|scared|overwhelmed|scared)\b/.test(t)) {
    return 0.55;
  }
  return 0.30;
}

async function askOllama(userMessage, context, lang, history, opts = {}) {
  if (!OLLAMA_URL) return null;
  const isWelsh = lang === 'cy';
  const system  = getSystemPrompt(lang);
  const temp    = typeof opts.temperature === 'number' ? opts.temperature : 0.35;

  let historyBlock = '';
  if (Array.isArray(history) && history.length > 0) {
    historyBlock = 'CONVERSATION HISTORY:\n' + history.slice(-6).map(t =>
      `${t.role === 'user' ? 'Student' : 'U-Pal'}: ${t.text}`
    ).join('\n') + '\n\n';
  }
  const prompt = `${historyBlock}---\n${context}\n---\n\nStudent: ${userMessage}\n\n${isWelsh ? 'U-Pal (Cymraeg yn unig):' : 'U-Pal:'}`;

  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 25000);
    const res = await fetch(`${OLLAMA_URL}/api/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'ngrok-skip-browser-warning': 'true' },
      signal: controller.signal,
      body: JSON.stringify({
        model: OLLAMA_MODEL,
        prompt,
        system,
        stream: false,
        options: {
          temperature:    temp,
          num_predict:    350,  // reduces tail where the model tends to loop
          repeat_penalty: 1.25, // discourages looping Welsh sentences (llama3.1:8b)
          top_p:          0.9,
          num_ctx:        OLLAMA_NUM_CTX, // context window — set lower to save VRAM
        },
      })
    });
    clearTimeout(timeout);
    if (!res.ok) return null;
    const data = await res.json();
    return (data.response || '').trim() || null;
  } catch (e) {
    console.warn('[Ollama] unavailable:', e.message);
    return null;
  }
}

// ─── Unified LLM dispatcher ─────────────────────────────────────────────
// Uses Ollama (local) as the sole LLM backend.
// Returns { reply, backend } or null if Ollama is unavailable.
async function askLLM(userMessage, context, lang, history, opts = {}) {
  // Adaptive temperature based on query type (factual vs empathetic),
  // unless the caller pinned a value.
  const temperature = typeof opts.temperature === 'number'
    ? opts.temperature
    : pickTemperature(userMessage);
  const llmOpts = { ...opts, temperature };

  if (OLLAMA_URL) {
    const reply = await askOllama(userMessage, context, lang, history, llmOpts);
    if (reply) return { reply, backend: 'ollama', temperature };
  }
  return null;
}

// MyMemory has a ~500-char/request limit and silently truncates longer
// payloads.  For a 4-sentence chatbot reply (often 400-800 chars) this means
// the English alt-response would drop the second half of the Welsh primary
// (or vice-versa).  We chunk by sentence so each request stays under the
// limit, then rejoin with spaces.
async function translateOllamaOnce(text, sourceLang, targetLang, placeholders) {
  const url = `https://api.mymemory.translated.net/get?q=${encodeURIComponent(text)}&langpair=${sourceLang}|${targetLang}`;
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 10000);
    const res = await fetch(url, { signal: controller.signal });
    clearTimeout(timeout);
    if (!res.ok) return null;
    const data = await res.json();
    if (data.responseStatus !== 200 || !data.responseData?.translatedText) return null;
    let translated = data.responseData.translatedText;
    if (placeholders) {
      for (const [k, v] of Object.entries(placeholders)) {
        translated = translated.replace(new RegExp(k, 'g'), v);
      }
    }
    return translated || null;
  } catch (e) {
    return null;
  }
}

function chunkForTranslation(text, maxLen = 450) {
  // Split into sentences (keeping punctuation), then greedily pack chunks
  // up to maxLen chars so each MyMemory request stays under the 500-char
  // cap.  Sentences longer than maxLen are cut at the nearest space.
  const sentences = text.match(/[^.!?…\n]+[.!?…]?/g) || [text];
  const chunks = [];
  let buf = '';
  for (const s of sentences) {
    const piece = s.trim();
    if (!piece) continue;
    if (piece.length > maxLen) {
      if (buf) { chunks.push(buf); buf = ''; }
      let rest = piece;
      while (rest.length > maxLen) {
        let cut = rest.lastIndexOf(' ', maxLen);
        if (cut < 100) cut = maxLen;
        chunks.push(rest.slice(0, cut).trim());
        rest = rest.slice(cut).trim();
      }
      if (rest) buf = rest;
      continue;
    }
    if ((buf + ' ' + piece).trim().length <= maxLen) {
      buf = buf ? buf + ' ' + piece : piece;
    } else {
      chunks.push(buf);
      buf = piece;
    }
  }
  if (buf) chunks.push(buf);
  return chunks;
}

async function translateOllama(text, targetLang) {
  if (!text) return null;
  const placeholders = {};
  let i = 0;
  const protected_ = text
    .replace(/[\w._%+\-]+@[\w.\-]+\.[a-z]{2,}/gi, m => { const k=`__EMAIL${i++}__`; placeholders[k]=m; return k; })
    .replace(/https?:\/\/[^\s]+/gi,                m => { const k=`__URL${i++}__`;   placeholders[k]=m; return k; })
    .replace(/(?<!\d)\d{5,}(?!\d)/g,               m => { const k=`__PHONE${i++}__`; placeholders[k]=m; return k; });
  const sourceLang = targetLang === 'cy' ? 'en' : 'cy';

  // Short enough for a single request — keep the old single-call path.
  if (protected_.length <= 450) {
    return translateOllamaOnce(protected_, sourceLang, targetLang, placeholders);
  }

  // Long text → split, translate in parallel, rejoin.  If any chunk fails
  // we still stitch the rest together so the alt response is mostly
  // translated rather than silently null.
  const chunks = chunkForTranslation(protected_, 450);
  const results = await Promise.all(
    chunks.map(c => translateOllamaOnce(c, sourceLang, targetLang, placeholders))
  );
  const ok = results.filter(Boolean);
  if (!ok.length) return null;
  // Fill in any null chunks with the originals so the user sees *something*
  // rather than a mid-sentence drop.
  const stitched = results.map((r, idx) => r || chunks[idx]).join(' ');
  return stitched || null;
}

// BydTermCymru query augmentation —————————————————————————————————————————
// When a Welsh query arrives, translate key Welsh tokens to English using the
// bilingual map and append them to the query string.  This significantly
// improves Morphik retrieval because most ingested UWTSD content is English:
// a query like "Beth yw'r ffioedd dysgu?" becomes
// "Beth yw'r ffioedd dysgu? fees learning education" before embedding.
function augmentWelshQuery(text) {
  if (!Object.keys(WELSH_EN_MAP).length) return text;
  const normalised  = text.toLowerCase().replace(/[^\w\u00C0-\u024F\s']/g, ' ');
  const tokens      = normalised.split(/\s+/).filter(t => t.length > 1);
  const englishHits = new Set();

  // First pass: multi-word compound lookups (longest match first)
  const usedIndices = new Set();
  for (let len = 3; len >= 2; len--) {
    for (let i = 0; i <= tokens.length - len; i++) {
      if ([...Array(len).keys()].some(j => usedIndices.has(i + j))) continue;
      const compound = tokens.slice(i, i + len).join(' ');
      if (WELSH_EN_MAP[compound]) {
        englishHits.add(WELSH_EN_MAP[compound]);
        for (let j = 0; j < len; j++) usedIndices.add(i + j);
      }
    }
  }
  // Second pass: single-token lookups for remaining tokens
  tokens.forEach((t, i) => {
    if (usedIndices.has(i)) return;
    if (WELSH_EN_MAP[t]) englishHits.add(WELSH_EN_MAP[t]);
  });

  if (!englishHits.size) return text;
  return `${text} ${[...englishHits].join(' ')}`;
}

// Morphik RAG retrieval — fetches relevant document chunks from ingested UWTSD content.
// Returns a formatted string of passages, or '' if Morphik is unavailable / unconfigured.
async function queryMorphik(message, k = 7) {
  if (!MORPHIK_URL) return '';
  try {
    const controller = new AbortController();
    const timeout    = setTimeout(() => controller.abort(), 12000);
    const headers    = {
      'Content-Type':               'application/json',
      'ngrok-skip-browser-warning': 'true',
      'User-Agent':                 'UPal-UWTSD-Chatbot/1.0',
    };
    if (MORPHIK_TOKEN) headers['Authorization'] = `Bearer ${MORPHIK_TOKEN}`;
    const res = await fetch(`${MORPHIK_URL}/retrieve/chunks`, {
      method:  'POST',
      headers,
      signal:  controller.signal,
      body:    JSON.stringify({ query: message, k }),
    });
    clearTimeout(timeout);
    if (!res.ok) return '';
    const data   = await res.json();
    const chunks = Array.isArray(data) ? data : (data.results || data.chunks || []);
    if (!chunks.length) return '';
    // Keep the top 2 chunks, each capped so the LLM gets focused context
    // instead of drowning in a 5-passage dump.  Remove the "[Doc N]" label
    // so the model doesn't echo it back ("According to Doc 1…").
    return chunks
      .slice(0, 2)
      .map(c => (c.content || c.text || '').trim().slice(0, 450))
      .filter(s => s.length > 20)
      .join('\n\n');
  } catch (e) {
    console.warn('[Morphik] unavailable:', e.message);
    return '';
  }
}

function buildContext(topIntents, lang) {
  return topIntents.map(({ intent }) => {
    const responses = intent.responses[lang] || intent.responses['en'] || [];
    return responses.join(' ');
  }).join('\n\n');
}

// Local chit-chat detection — greetings, thanks, farewells handled without
// contacting any external services.  Returns a { en, cy, tag } object or null.
//
// Why: these are the top frustration path for a chatbot — if "hello" or
// "shwmae" returns a generic fallback, the user loses confidence immediately.
// We short-circuit them here so the bot always feels responsive.
const SMALLTALK_PATTERNS = [
  {
    tag: 'greeting',
    match: /^\s*(hi|hiya|hey|hello|heya|hola|yo|sup|howdy|good\s+(morning|afternoon|evening)|morning|evening|shw+mae|shwmai|helô?|haia|bore\s+da|prynhawn\s+da|noswaith\s+dda|henffych)(\s+(there|all|everyone|u-?pal|bot))?[\s!.?,]*$/i,
    en: "Hi there! I'm U-Pal, your UWTSD student assistant. I can help with admissions, courses, fees and funding, accommodation, campus locations, wellbeing, the library, IT support and more. What would you like to know?",
    cy: "Shwmae! U-Pal ydw i, cynorthwyydd myfyrwyr PCYDDS. Gallaf helpu gyda derbyniadau, cyrsiau, ffioedd ac ariannu, llety, lleoliadau campws, lles, y llyfrgell, cymorth TG a mwy. Beth hoffech chi wybod?"
  },
  {
    tag: 'how_are_you',
    match: /^\s*(how\s+are\s+you|how'?s\s+it\s+going|how\s+are\s+things|you\s+good|sut\s+(w?yt|mae)\s+(ti|chi)|sut\s+wyt|sut\s+mae)[\s!.?]*$/i,
    en: "I'm doing great, thanks for asking! I'm here and ready to help with anything UWTSD-related — courses, fees, accommodation, support services, you name it. What's on your mind?",
    cy: "Dw i'n iawn, diolch am ofyn! Dwi yma i helpu gydag unrhyw beth sy'n ymwneud â PCYDDS — cyrsiau, ffioedd, llety, gwasanaethau cymorth, beth bynnag. Beth sydd ar eich meddwl?"
  },
  {
    tag: 'thanks',
    match: /^\s*(thanks|thank\s+you|thx|ty|cheers|much\s+appreciated|appreciate\s+it|diolch(\s+yn\s+fawr)?|diolch\s+byth)[\s!.?]*$/i,
    en: "You're very welcome! If anything else comes up about your studies, funding, accommodation or campus life, just ask.",
    cy: "Croeso mawr! Os oes unrhyw beth arall yn codi am eich astudiaethau, ariannu, llety neu fywyd campws, gofynnwch."
  },
  {
    tag: 'goodbye',
    match: /^\s*(bye|goodbye|see\s+ya|see\s+you|cya|later|hwyl(\s+fawr)?|da\s+bo\s+ti|ta\s+ra)[\s!.?]*$/i,
    en: "Take care! Come back any time you need a hand with UWTSD matters.",
    cy: "Cymerwch ofal! Dewch 'nôl unrhyw bryd y bydd arnoch angen help gyda materion PCYDDS."
  },
  {
    tag: 'who_are_you',
    match: /^\s*(who\s+are\s+you|what\s+are\s+you|what'?s\s+your\s+name|who'?s\s+this|pwy\s+(w?yt|ydych)\s+(ti|chi)|pwy\s+wyt\s+ti|beth\s+yw'?r?\s+enw)[\s!.?]*$/i,
    en: "I'm U-Pal — a bilingual student support chatbot for UWTSD (University of Wales Trinity Saint David). I use the university's public information to answer questions about applying, studying, funding, living, and getting support here. How can I help?",
    cy: "U-Pal ydw i — chatbot dwyieithog i gefnogi myfyrwyr PCYDDS (Prifysgol Cymru Y Drindod Dewi Sant). Dwi'n defnyddio gwybodaeth gyhoeddus y brifysgol i ateb cwestiynau am ymgeisio, astudio, ariannu, byw, a chael cymorth yma. Sut alla i helpu?"
  },
  {
    tag: 'capabilities',
    match: /^\s*(what\s+can\s+you\s+do|what\s+do\s+you\s+know|help(\s+me)?|help\s+please|beth\s+(allet|allwch|wyt|ydych)\s+.{0,20}(helpu|gwneud)|helpu(\s+fi)?)[\s!.?]*$/i,
    en: "I can answer questions about UWTSD — admissions and applying, open days, courses and modules, tuition fees, scholarships and bursaries, student loans, accommodation, campus life in Swansea, Carmarthen and Lampeter, library services, IT and Wi-Fi, wellbeing, disability support, graduation, and the Students' Union. Ask me anything specific!",
    cy: "Gallaf ateb cwestiynau am PCYDDS — derbyniadau ac ymgeisio, diwrnodau agored, cyrsiau a modiwlau, ffioedd dysgu, ysgoloriaethau a bwrsariaethau, benthyciadau myfyrwyr, llety, bywyd campws yn Abertawe, Caerfyrddin a Llambed, gwasanaethau llyfrgell, TG a Wi-Fi, lles, cymorth anabledd, graddio, ac Undeb y Myfyrwyr. Gofynnwch unrhyw beth penodol!"
  },
];

function detectSmallTalk(text) {
  for (const s of SMALLTALK_PATTERNS) {
    if (s.match.test(text)) return s;
  }
  return null;
}

// ─── UWTSD corpus retrieval ──────────────────────────────────────────────
// Build a TF-IDF index over the harvested Morphik passages at module load.
// This gives us a local, always-available retrieval source that looks and
// behaves like Morphik but works offline — vital when the ngrok tunnel to the
// VM is slow, or when Vercel is deployed somewhere that can't reach the VM.
const corpusTfidf = new natural.TfIdf();
// Parallel array: CORPUS_ID_BY_POS[i] === UWTSD_CORPUS[i].id  (O(1) pos ↔ id lookups
// when merging lexical and dense results by ID in retrieveFromCorpus).
const CORPUS_ID_BY_POS  = new Array(UWTSD_CORPUS.length);
const CORPUS_POS_BY_ID  = new Map();
UWTSD_CORPUS.forEach((passage, i) => {
  const lang = passage.lang === 'cy' ? 'cy' : 'en';
  const topicsText = Array.isArray(passage.topics) ? passage.topics.join(' ') : '';
  // Index on topics + content so queries match either the canonical question
  // that was used to retrieve the chunk OR the chunk body itself.
  corpusTfidf.addDocument(preprocess(`${topicsText}\n${passage.content}`, lang));
  CORPUS_ID_BY_POS[i] = passage.id;
  CORPUS_POS_BY_ID.set(passage.id, i);
});
if (UWTSD_CORPUS.length) {
  console.log(`[UWTSD corpus] Indexed ${UWTSD_CORPUS.length} passages for offline retrieval`);
}

// ─── Hybrid retrieval: TF-IDF ⊕ dense embeddings, fused via RRF ──────────
// Reciprocal Rank Fusion (Cormack et al., SIGIR 2009) combines two or more
// ranked lists without needing to calibrate scores across systems — the
// score of a doc is Σ 1/(k + rank_in_each_list).  k=60 is the textbook
// default and works well in practice.
//
// Why fuse instead of picking one retriever?
//   - Lexical TF-IDF nails exact-match queries ("PGCE", "£9,535") but
//     misses paraphrase ("money for uni" vs "tuition cost").
//   - Dense cosine nails paraphrase but can miss rare tokens (course
//     codes, postcodes, obscure acronyms).
//   - RRF takes the best of both with a single well-understood knob.
//
// If the embedding sidecar is missing OR Ollama is unreachable at query
// time, we fall back to the pure-TF-IDF path — the caller never sees a
// degradation, just slightly weaker semantic matches.
const RRF_K = 60;

function rrfFuse(rankedLists, k = RRF_K) {
  const fused = new Map();   // id -> { score, hits: [{src, rank, raw}, ...] }
  for (const { name, ranked } of rankedLists) {
    ranked.forEach((entry, idx) => {
      const id    = entry.id;
      const rrf   = 1 / (k + idx + 1);
      const prev  = fused.get(id) || { score: 0, hits: [] };
      prev.score += rrf;
      prev.hits.push({ src: name, rank: idx + 1, raw: entry.score });
      fused.set(id, prev);
    });
  }
  return [...fused.entries()]
    .map(([id, v]) => ({ id, score: v.score, hits: v.hits }))
    .sort((a, b) => b.score - a.score);
}

// Lexical candidates from natural.TfIdf.  Returns [{ id, score }, ...] sorted desc.
function lexicalTopK(processed, k) {
  if (!processed) return [];
  const scored = [];
  corpusTfidf.tfidfs(processed, (i, score) => {
    if (score > 0.05) scored.push({ pos: i, score });
  });
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, k).map(({ pos, score }) => ({
    id: CORPUS_ID_BY_POS[pos],
    score,
  }));
}

// Retrieve top-k UWTSD passages using hybrid lexical + dense retrieval fused
// with RRF.  Returns { content, score, source, lang, topics, hits[] } — where
// `hits` explains which retriever(s) surfaced the passage (useful for the
// rerank stage and for debugging).  Async because we embed the query with
// Ollama; the function degrades to lexical-only if embedding fails.
async function retrieveFromCorpus(message, lang, k = 5) {
  if (!UWTSD_CORPUS.length) return [];

  const expanded  = lang === 'cy' ? augmentWelshQuery(message) : message;
  const processed = preprocess(expanded, lang);

  // Lexical pool (over-retrieve so RRF has room to re-rank).
  const POOL = Math.max(k * 4, 20);
  const lexRanked = lexicalTopK(processed, POOL);

  // Dense pool.  Ollama embedding is async; `null` signals Ollama-down.
  let denseRanked = [];
  try {
    const qVec = await embed.embedQuery(expanded || message);
    if (qVec && qVec.length) {
      // Don't hard-filter by language for dense: nomic-embed-text is
      // reasonably cross-lingual and we sometimes want CY queries to find
      // EN passages (the Morphik corpus is overwhelmingly English).
      denseRanked = embed.cosineTopK(qVec, POOL).map(r => ({
        id:    r.id,
        score: r.score,
      }));
    }
  } catch (e) {
    if (process.env.DEBUG_EMBED) console.warn(`[retrieveFromCorpus] dense failed: ${e.message}`);
  }

  // Fuse.  If dense was unavailable we fuse a single list (RRF degenerates
  // to a rank-preserving score — same order as lexical alone).
  const rankedLists = [{ name: 'lexical', ranked: lexRanked }];
  if (denseRanked.length) rankedLists.push({ name: 'dense', ranked: denseRanked });
  const fused = rrfFuse(rankedLists);

  // Materialise the top-(k*2) fused candidates, then rerank down to k.
  // The rerank budget is small (~8) but meaningful: RRF gets ordering
  // mostly right; rerank fixes the last-mile ties using term-overlap and
  // optionally an LLM judge.
  const prelim = [];
  for (const { id, score, hits } of fused.slice(0, Math.max(k * 2, 10))) {
    const pos = CORPUS_POS_BY_ID.get(id);
    if (pos === undefined) continue;
    const p = UWTSD_CORPUS[pos];
    if (!p) continue;
    prelim.push({
      id,
      content: p.content,
      score,
      source:  p.source || 'morphik',
      lang:    p.lang   || 'en',
      topics:  p.topics || [],
      hits,
    });
  }

  // Reranker.  When LLM_RERANK=1 we call Groq; otherwise local-only.
  // We rerank against the ORIGINAL (non-augmented) query so that the
  // user's actual wording drives the final ordering — Welsh expansion
  // can over-weight whatever English synonym was picked.
  try {
    return await rerank.llmRerank(message, prelim, k);
  } catch (e) {
    if (process.env.DEBUG_RERANK) console.warn(`[retrieveFromCorpus] rerank failed: ${e.message}`);
    return prelim.slice(0, k);
  }
}

// Format corpus passages as a block suitable for pasting into the Ollama
// prompt.  We truncate to avoid blowing the token budget.
// Format the top passages as LLM context. Smaller models (llama3.1:8b)
// lose focus when handed 3+ long passages — they try to summarise all of
// them instead of answering the question.  We keep this tight:
//   • Only top 2 passages (rerank order caller-provided)
//   • Each capped at 500 chars (≈1 paragraph — enough for a specific answer)
//   • Total ≤ 1200 chars
// Caller can override if they've already reranked and want a wider pool.
function formatCorpusForPrompt(passages, maxChars = 1200, maxPassages = 2, perChars = 500) {
  if (!passages.length) return '';
  const lines = [];
  let total = 0;
  for (let i = 0; i < Math.min(passages.length, maxPassages); i++) {
    const snippet = passages[i].content.slice(0, perChars);
    // Use a simple label the LLM won't echo back.  "Passage 1" etc. is
    // tidy but some models re-emit "According to passage 1…" — avoid it.
    const block   = snippet;
    if (total + block.length > maxChars) break;
    lines.push(block);
    total += block.length;
  }
  return lines.join('\n\n');
}

// Compose a natural, human-sounding answer from the top corpus passages when
// ── Question-type detection ──────────────────────────────────────────────
// Classify the English query into a semantic question type so we know
// *what kind of information* the student is asking for.  This drives
// sentence-level scoring in composeCorpusAnswer below.
function detectQuestionType(text) {
  const t = (text || '').toLowerCase().trim();
  if (/^(where\b|ble mae|ble yw|location of|address of|how (do i |can i )?(get|find|reach|travel|go) to)/.test(t))
    return 'LOCATION';
  if (/^(how much|how many|what (is|are) the (cost|fee|price|charge)|faint (yw|ydy|mae)|what does it cost)/.test(t))
    return 'QUANTITY';
  if (/^(when|pryd|what (date|time|day|are the hours?|are (the )?opening)|opening hours?|what time does)/.test(t))
    return 'DATE';
  if (/^(how (do i|can i|should i|would i|to)|sut (allaf|mae|ydw|i)|what (is|are) the (process|steps?|procedure)|how (do i|can i) apply)/.test(t))
    return 'PROCESS';
  if (/^(who|pwy|who (is|are|do i|should i|can i)|who (is my|are my))/.test(t))
    return 'PERSON';
  if (/^(is there|are there|oes (yna|mae)|do you have|does uwtsd (have|offer|provide))/.test(t))
    return 'YESNO';
  if (/^(what (is|are|does|do)|what'?s|tell me (about|what)|explain|describe|beth (yw|ydy|mae))/.test(t))
    return 'DEFINITION';
  return 'GENERAL';
}

// Per question-type: a function that adds a bonus score to a sentence
// based on how directly it answers that type of question.
// LOCATION → postcodes, building names, directions
// QUANTITY  → £ amounts, fee figures
// DATE      → day names, month names, times
// PROCESS   → emails, URLs, imperative verbs
// PERSON    → roles, contact details
// YESNO     → "yes"/"no" statements, availability language
// DEFINITION→ definitional copulas
const QTYPE_BOOST = {
  LOCATION:   s => {
    let b = 0;
    if (/[A-Z]{1,2}\d{1,2}\s*\d[A-Z]{2}/i.test(s))                                b += 6; // postcode
    if (/\b(road|street|avenue|way|lane|waterfront|building|house)\b/i.test(s))    b += 3;
    if (/\b(located|situated|find us|address|directions?|junction|campus)\b/i.test(s)) b += 2;
    if (/\b(city centre|train station|walking distance|bus stop|park)\b/i.test(s)) b += 1;
    return b;
  },
  QUANTITY:   s => {
    let b = 0;
    if (/£[\d,]+/.test(s))                                                          b += 6;
    if (/\b(per year|annually|per semester|per credit|per annum)\b/i.test(s))       b += 3;
    if (/\b(fee|fees|cost|costs|price|charge|tuition|bursary|loan)\b/i.test(s))    b += 2;
    return b;
  },
  DATE:       s => {
    let b = 0;
    if (/\d{1,2}(am|pm|:\d{2})/i.test(s))                                          b += 5; // time
    if (/\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b/i.test(s)) b += 4;
    if (/\b(january|february|march|april|may|june|july|august|september|october|november|december)\b/i.test(s)) b += 3;
    if (/\b(deadline|open|close|close[ds]?|until|by|before|from)\b/i.test(s))      b += 2;
    return b;
  },
  PROCESS:    s => {
    let b = 0;
    if (/@[\w.]+\.ac\.uk/.test(s))                                                  b += 5; // email
    if (/https?:\/\/|uwtsd\.ac\.uk/i.test(s))                                      b += 4; // URL
    if (/\b(apply|contact|visit|complete|submit|email|call|register|log in|sign up|fill in|book)\b/i.test(s)) b += 3;
    if (/\b(step|first|then|next|finally|once|after|before)\b/i.test(s))           b += 2;
    return b;
  },
  PERSON:     s => {
    let b = 0;
    if (/@[\w.]+\.ac\.uk/.test(s))                                                  b += 4;
    if (/\b(tutor|director|coordinator|manager|officer|adviser|team|staff|contact)\b/i.test(s)) b += 3;
    if (/01792|0300/.test(s))                                                       b += 2;
    return b;
  },
  YESNO:      s => {
    let b = 0;
    if (/\b(available|provided|offered|accessible|free|support|service|yes|there is|there are)\b/i.test(s)) b += 3;
    return b;
  },
  DEFINITION: s => {
    let b = 0;
    if (/\b(is a|are a|provides?|offers?|supports?|allows?|enables?|gives?|means?)\b/i.test(s)) b += 3;
    return b;
  },
  GENERAL:    () => 0,
};

// Ollama is unreachable.  This is NOT a fallback apology — it's a direct
// answer drawn from the same UWTSD data Morphik uses.
function composeCorpusAnswer(passages, lang, query = '') {
  if (!passages.length) return null;

  // Detect what *type* of question this is so we can score sentences on
  // whether they actually answer it — not just whether they share keywords.
  const qType  = detectQuestionType(query);
  const qtBoost = QTYPE_BOOST[qType] || QTYPE_BOOST.GENERAL;

  // Detect student-type qualifiers ("international", "postgraduate", etc.)
  // so sentences about the right category are surfaced first.
  const queryQualifiers = detectQueryQualifiers(query);

  // Extract content keywords for overlap scoring (strip stopwords).
  const stopSet = lang === 'cy' ? CY_STOP : EN_STOP;
  const queryTokens = query
    .toLowerCase()
    .replace(/[^\w\s]/g, ' ')
    .split(/\s+/)
    .filter(t => t.length > 2 && !stopSet.has(t));

  const scoreSentence = (sentence) => {
    const sl = sentence.toLowerCase();
    // 1. Keyword overlap — shared content words with the query.
    //    Short tokens like "sa1" are given double weight (highly specific).
    const keywordScore = queryTokens.reduce((sum, t) => {
      if (!sl.includes(t)) return sum;
      return sum + (t.length <= 3 ? 2 : 1);
    }, 0);
    // 2. Question-type boost — does this sentence contain the *kind* of
    //    information the question is looking for?
    //    "Where is SA1?" needs a postcode/address sentence, not an event date.
    const typeScore = qtBoost(sentence);
    // 3. Qualifier boost — does this sentence discuss the student category
    //    the user asked about?  E.g. for "international fees" we want the
    //    sentence containing "£15,525" not the one containing "£9,535".
    let qualifierScore = 0;
    if (queryQualifiers.size > 0) {
      for (const qual of queryQualifiers) {
        if (sl.includes(qual)) qualifierScore += 4;
      }
      // Penalise sentences that are clearly about a *different* student group
      // than the one asked about, to avoid mixing tiers in the answer.
      if (queryQualifiers.has('international') &&
          /\b(uk|home|domestic)\b/.test(sl) && !sl.includes('international')) {
        qualifierScore -= 4;
      }
      if ((queryQualifiers.has('home') || queryQualifiers.has('uk')) &&
          sl.includes('international') && !/\b(home|uk|domestic)\b/.test(sl)) {
        qualifierScore -= 4;
      }
      if (queryQualifiers.has('postgraduate') &&
          /\bundergraduate\b/.test(sl) && !/\bpostgraduate\b|\bmaster/.test(sl)) {
        qualifierScore -= 2;
      }
      if (queryQualifiers.has('undergraduate') &&
          /\bpostgraduate\b|\bmaster/.test(sl) && !/\bundergraduate\b/.test(sl)) {
        qualifierScore -= 2;
      }
    }
    return keywordScore + typeScore + qualifierScore;
  };

  // Collect and score sentences across all top passages.  Morphik chunks
  // often repeat the same paragraph twice (ingestion artifact) — the seen-
  // sentence set deduplicates them.
  const seenSentences = new Set();
  const norm = (s) => s.toLowerCase().replace(/\s+/g, ' ').replace(/[^\w\s]/g, '').trim();
  const candidates = [];

  for (let pi = 0; pi < Math.min(passages.length, 5); pi++) {
    const p = passages[pi];
    const sentences = (p.content || '')
      .replace(/\s+/g, ' ')
      .split(/(?<=[.!?])\s+/)
      .filter(Boolean);

    for (const s of sentences) {
      const key = norm(s);
      if (!key || key.length < 20) continue;
      if (seenSentences.has(key)) continue;
      seenSentences.add(key);
      candidates.push({
        text:      s.trim(),
        relevance: scoreSentence(s),
        pScore:    p.score || 0,
        pi,
      });
    }
  }

  if (!candidates.length) return null;

  // Sort: most query-relevant first, then by passage retrieval score.
  // This surfaces "IQ Building at SA1 Waterfront, SA1 8EW" over a UKCISA
  // blurb even when the UKCISA passage ranked higher overall.
  candidates.sort((a, b) =>
    b.relevance !== a.relevance
      ? b.relevance - a.relevance
      : b.pScore - a.pScore
  );

  // Take the top sentences (cap each at 250 chars to avoid wall-of-text)
  const chosen = candidates
    .slice(0, 4)
    .map(c => c.text.length > 250 ? c.text.slice(0, 247) + '…' : c.text);

  const body = [chosen[0]];
  if (chosen[1] && chosen[1] !== chosen[0]) body.push(chosen[1]);
  if (chosen[2] && !body.some(b => b.includes(chosen[2].slice(0, 30)))) body.push(chosen[2]);

  const primary = body.join(' ');
  if (!primary) return null;

  const footer = lang === 'cy'
    ? '\n\nAm ragor o fanylion: enquiries@uwtsd.ac.uk | 01792 481 111 | uwtsd.ac.uk/cy'
    : '\n\nFor more details: enquiries@uwtsd.ac.uk | 01792 481 111 | uwtsd.ac.uk';

  return `${primary}${footer}`;
}

// ─── NLU topic classifier ────────────────────────────────────────────────
// Loads the 600+ topic database produced by scripts/build-nlu-topics.py and
// builds a TF-IDF index over every trigger phrase (EN + CY).  At query time:
//   • classifyTopic(text) returns the best-matching topic + score
//   • If the topic ships a fast-path reply (e.g. greetings, crisis), we use
//     it directly; no LLM round-trip.
//   • Otherwise we use the topic's morphik_hint to augment the Morphik
//     query, which dramatically improves retrieval for vague phrasings.
let NLU_TOPICS = [];
try {
  const raw = JSON.parse(fs.readFileSync(path.join(process.cwd(), 'nlu-topics.json'), 'utf8'));
  NLU_TOPICS = Array.isArray(raw.topics) ? raw.topics : [];
  if (NLU_TOPICS.length) {
    console.log(`[NLU] Loaded ${NLU_TOPICS.length} topics with ` +
      `${NLU_TOPICS.reduce((a,t)=>a+(t.phrases_en||[]).length+(t.phrases_cy||[]).length,0)} trigger phrases`);
  }
} catch (e) { /* optional file */ }

// ── Curated UWTSD facts (uwtsd-facts.json) ───────────────────────────────
// Hand-written polished Q&A pairs that fire BEFORE corpus fallback and
// work 100% offline.  Each entry has:
//   id        – unique slug
//   questions – array of example questions (used for TF-IDF matching)
//   keywords  – extra matching terms (weighted ×2)
//   answer_en – clean English answer shown directly without LLM synthesis
//   answer_cy – Welsh equivalent
//
// To add more: edit uwtsd-facts.json in the project root, no code change
// needed.  Threshold is 0.25 (adjust FACTS_THRESHOLD if too strict/loose).
let UWTSD_FACTS = [];
const factsTfidf = new natural.TfIdf();
const FACTS_THRESHOLD = 0.28;
// Bilingual stopwords — stripped at index-build AND query time so very common
// filler words don't create false matches when Welsh queries hit the shared
// index (the Welsh→English translator can flake, and without it the raw query
// is tokenised directly against the combined EN+CY fact document).
const FACT_STOPWORDS = new Set([
  // English
  'a','an','the','is','are','was','were','be','been','am','do','does','did',
  'i','you','he','she','it','we','they','me','my','your','his','her','our','their',
  'to','of','in','on','at','for','with','from','by','as','and','or','but','so',
  'this','that','these','those','what','when','where','why','how','which','who','whom',
  'not','no','yes','can','could','would','should','will','may','might',
  'about','there','into','please','pls','plz','ok','okay','uh','um','hi','hello','hey',
  // High-frequency emotional/intensifier words that appear in BOTH crisis and
  // non-crisis queries and therefore carry no topic-discriminating signal.
  // Without stripping these, "i am feeling really anxious about my exams" scores
  // against wellbeing-crisis because the crisis fact contains "feeling hopeless".
  'feeling','feel','feels','felt',
  'really','very','quite','so','just','bit','little','bit',
  'need','want','get','got','have','having','has','going','trying','hard',
  'im','ive','id','ill','dont','cant','wont','isnt','arent','wasnt',
  // 'help' appears in crisis keywords ("urgent help") AND in countless innocent
  // queries ("I need help with my dissertation") — strip it so it never tips the
  // balance toward crisis.  Specific topic words (crisis, suicidal, breakdown)
  // still provide all the discriminating signal we need.
  'help','helps','helping',
  // 'right' in casual English is a sentence-final confirmation tag ("done my
  // project right?") — in crisis questions it appears as "right now".  Stripping
  // it prevents "project right?" from scoring against "mental health crisis right now".
  'right','now','ok','okay','fine','sure','good','bad','like','talk','see',
  // Generic navigation / intent words that appear in almost every question pattern
  // ("how do I FIND my tutor", "where can I FIND activities", "LEARN about fees").
  // These carry zero topic signal and cause false fact matches when the query
  // contains an unrelated topic word alongside the navigation verb.
  'find','found','look','looking','learn','know','tell','show','get','go',
  'information','info','details','more','something','anything','everything',
  // Welsh — very common fillers
  'i','y','yr','yn','ar','o','am','fy','dy','ei','eu','ein','eich','a','ac','ond',
  'dw','rwy','rwyf','ydw','ydy','yw','ond','mae','oes','beth','ble','pryd','pam','sut','faint',
  'ti','chi','fo','fe','hi','nhw','chdi','eich',
  'yr','hwn','hon','hwnna','hwnnw','honno',
  'plis','helo','helô','shwmae','shwdi',
  // Welsh intensifiers / filler verbs (mirrors EN additions above)
  'teimlo','teimla','teimlaf','teimlol',
  'wir','iawn','hefyd','nawr','rwan',
  'eisiau','isie','isio','moyn','angen','cael',
]);
function stripStopwords(text) {
  return text.toLowerCase()
    .split(/[^a-zA-ZÀ-ÿŵŷ0-9]+/)
    .filter(tok => tok && !FACT_STOPWORDS.has(tok))
    .join(' ');
}
try {
  UWTSD_FACTS = JSON.parse(fs.readFileSync(path.join(process.cwd(), 'uwtsd-facts.json'), 'utf8'));
  UWTSD_FACTS.forEach(f => {
    const raw = [
      ...(f.questions || []),
      ...((f.keywords || []).flatMap(k => [k, k])),  // double-weight keywords
    ].join(' ');
    factsTfidf.addDocument(stripStopwords(raw));
  });
  console.log(`[Facts] Loaded ${UWTSD_FACTS.length} curated UWTSD fact entries`);
} catch (e) { /* optional file — graceful if absent */ }

// ─── Query qualifier extraction ─────────────────────────────────────────
// Detects student-type modifiers so qualifier-specific fact entries are
// preferred over generic catch-all entries.  Returns a Set of qualifier keys.
const QUALIFIER_PATTERNS = new Map([
  // English + Welsh equivalents
  ['international', /\b(international|overseas|non[-\s]?uk|foreign\s+students?|rhyngwladol)\b/i],
  ['postgraduate',  /\b(postgrad(uate)?|masters?'?|msc|mba|pg\b|pg\s+taught|ôl.raddedig|ol.raddedig)\b/i],
  ['undergraduate', /\b(undergrads?(uate)?s?|bachelors?'?|bsc|b\.?a\.?\b|foundation\s+year|ug\b|first\s+degree|israddedig)\b/i],
  ['home',          /\b(home\s+students?|uk\s+students?|domestic\s+students?|home\/uk|uk\/home|home\s+fees?|uk\s+fees?|for\s+home\b|cartref\s+myfyrwyr?|myfyrwyr\s+y\s+du)\b/i],
  ['online',        /\b(online|distance\s+learning|remote\s+study|blended\s+learning|ar.lein)\b/i],
  ['parttime',      /\b(part[-\s]time|parttime|rhan.amser)\b/i],
  // Course-specific qualifiers — let the fees-pgce / fees-phd entries win when
  // the course name is in the query, and lose to fees-general when it isn't.
  ['phd',           /\b(phd|doctorate|doctoral|dphil|mphil|research\s+degree|doethuriaeth)\b/i],
  ['pgce',          /\b(pgce|tar|teacher\s+training|postgraduate\s+certificate\s+in\s+education|tystysgrif\s+ôl.raddedig\s+mewn\s+addysg|hyfforddi\s+athrawon)\b/i],
]);

function detectQueryQualifiers(query) {
  const q = (query || '').toLowerCase();
  const found = new Set();
  for (const [qualifier, pattern] of QUALIFIER_PATTERNS) {
    if (pattern.test(q)) found.add(qualifier);
  }
  return found;
}

// options.minScore  — absolute score floor (default FACTS_THRESHOLD)
// options.minRatio  — require top/second ratio ≥ this (default 0 = off).
//                      Generic queries like "tell me about UWTSD" cluster at
//                      similar scores across many facts; a ratio gate prunes
//                      those away on the fast-path without needing a tuned
//                      absolute threshold.
function lookupFact(query, optsOrMin) {
  if (!UWTSD_FACTS.length || !query) return null;
  const q = query.toLowerCase();
  // Qualifier detection runs against the original query (needs the full regex
  // context, e.g. "first-year" or "ôl-raddedig").
  const qualifiers = detectQueryQualifiers(q);
  // Strip bilingual stopwords before scoring so common filler words ("i", "yn",
  // "the", "please") don't push queries toward whichever fact happens to
  // contain more of them.  This keeps scoring driven by topic vocabulary.
  const qScored = stripStopwords(q);
  const opts = typeof optsOrMin === 'number'
    ? { minScore: optsOrMin }
    : (optsOrMin || {});
  const threshold = typeof opts.minScore === 'number' ? opts.minScore : FACTS_THRESHOLD;
  const minRatio  = typeof opts.minRatio === 'number' ? opts.minRatio  : 0;

  // Collect raw TF-IDF scores for every fact entry.
  const rawScores = new Array(UWTSD_FACTS.length).fill(0);
  factsTfidf.tfidfs(qScored, (i, score) => { rawScores[i] = score; });

  // Apply qualifier boost: when the query contains qualifiers that the fact
  // declares (e.g. fact has ["international","undergraduate"] and query says
  // "international students"), multiply that fact's score heavily so it beats
  // the generic all-tiers entry.  Generic facts (qualifiers:[]) are penalised
  // slightly when a qualifier was detected.
  let best = { idx: -1, score: 0 };
  let second = 0;
  for (let i = 0; i < UWTSD_FACTS.length; i++) {
    let s = rawScores[i];
    if (s <= 0) continue;

    const fact = UWTSD_FACTS[i];
    const factQuals = Array.isArray(fact.qualifiers) ? fact.qualifiers : [];

    if (qualifiers.size > 0) {
      const matches = factQuals.filter(fq => qualifiers.has(fq)).length;
      if (matches > 0 && matches === factQuals.length) {
        // Perfect qualifier match: ALL of the fact's declared qualifiers were
        // found in the query.  Give the biggest boost — "international
        // undergraduate" beats both "international-only" and plain undergraduate
        // entries for a query that says both.
        s *= (1 + matches * 3.0);
      } else if (matches > 0) {
        // Partial qualifier match: some (not all) of the fact's qualifiers
        // appear in the query.  Moderate boost — better than generic or wrong.
        s *= (1 + matches * 1.2);
      } else if (factQuals.length === 0) {
        // Generic (all-tiers) entry — penalise when query is qualifier-specific.
        s *= 0.55;
      } else {
        // Wrong qualifier type (e.g. UK entry when user asked about international)
        // — penalise hard so it doesn't surface.
        s *= 0.4;
      }
    } else {
      // No qualifier detected in query → strongly prefer generic entries over
      // qualifier-specific ones so "how much are the fees?" returns all tiers.
      if (factQuals.length > 0) s *= 0.30;
    }

    if (s > best.score) {
      second = best.score;
      best = { idx: i, score: s };
    } else if (s > second) {
      second = s;
    }
  }

  if (process.env.DEBUG_FACTS) {
    // Opt-in trace: run with DEBUG_FACTS=1 when tuning thresholds or
    // investigating mis-routes — shows top candidate, absolute score,
    // ratio vs runner-up, and which gate (if any) rejected the match.
    const _dbgId = UWTSD_FACTS[best.idx]?.id || 'none';
    const _dbgRatio = second > 0 ? (best.score / second).toFixed(2) : 'inf';
    const _outcome = best.score < threshold
      ? 'NULL_score'
      : (minRatio > 0 && second > 0 && best.score < second * minRatio)
        ? 'NULL_ratio'
        : _dbgId;
    console.log(`[lookupFact] q="${query.slice(0,55)}" top=${_dbgId} score=${best.score.toFixed(2)} ratio=${_dbgRatio} threshold=${threshold} minRatio=${minRatio} → ${_outcome}`);
  }
  if (best.score < threshold || best.idx < 0) return null;
  // Ratio gate: the winning fact must be meaningfully ahead of the runner-
  // up.  When several facts score the same (e.g. the query only contains a
  // generic token like "UWTSD"), we're not confident enough for a canned
  // reply — let the caller fall through to retrieval+LLM.
  if (minRatio > 0 && second > 0 && best.score < second * minRatio) return null;
  return UWTSD_FACTS[best.idx] || null;
}

const topicTfidf = new natural.TfIdf();
NLU_TOPICS.forEach(t => {
  const doc = [
    ...(t.phrases_en || []),
    ...(t.phrases_cy || []),
    ...(t.keywords   || []),
    t.title || '',
    t.title_cy || '',
  ].join(' ').toLowerCase();
  topicTfidf.addDocument(doc);
});

// classifyTopic — returns { topic, score } of the best-matching topic, or
// null if no topic crosses the confidence floor.
function classifyTopic(text, floor = 0.12) {
  if (!NLU_TOPICS.length || !text) return null;
  const q = String(text).toLowerCase();
  let best = { idx: -1, score: 0 };
  topicTfidf.tfidfs(q, (i, score) => {
    // Priority-weighted score: high-priority topics (crisis, greetings)
    // win ties more easily without drowning out precise matches
    const weight = 1 + (NLU_TOPICS[i].priority || 0) * 0.02;
    const s = score * weight;
    if (s > best.score) best = { idx: i, score: s };
  });
  if (best.idx < 0 || best.score < floor) return null;
  return { topic: NLU_TOPICS[best.idx], score: best.score };
}

// Convenience for processChat — builds the augmented retrieval query when a
// topic match is confident enough to sharpen Morphik's vector search.
function augmentWithTopicHint(query, topic) {
  if (!topic || !topic.morphik_hint) return query;
  return `${query}\n[topic: ${topic.id} — ${topic.morphik_hint}]`;
}

// ─── Follow-up / coreference rewriter ─────────────────────────────────────
// Real chat is elliptical.  Users say "how much?", "what about postgrad?",
// "do I need to pay for it?" — none of which retrieve sensibly without the
// prior turn's topic.  This helper:
//   • detects elliptical/pronoun-only/connector-led follow-ups
//   • pulls content words from the last user + assistant turns
//   • returns an expanded retrieval query that keeps the original question
//     but grounds it in the ongoing topic chain.
// It is ONLY used to enrich the Morphik/corpus retrieval query.  The prompt
// shown to Ollama still contains the raw user message + full history block.
const FOLLOWUP_TRIGGERS = [
  /^(what|how)\s+about\b/i,
  /^(and|but|also)\b/i,
  /^(for|in|at|to|with|about)\s+(the\s+)?[a-z]/i,   // "for postgrad", "in Carmarthen"
  /^(it|that|they|them|these|those|this|he|she|this one|that one)\b/i,
  /^(is|are|does|do|can|could|should|would|will|was|were)\s+(it|they|them|that|this)\b/i,
  /^(is|are)\s+there\b/i,                           // "is there a student bus?"
  /^(do|does)\s+(you|they)\s+(have|offer|provide)\b/i,
  /^(when|where|how|why|who)\s+(is|are|do|does|can)\s+(it|they|them|that|this)\b/i,
  /^(really|why|when|where|how|who|what)\??\s*$/i,
  /^(yes|yeah|ok|okay|sure|no|nope)\b/i,
  /\b(it|them|that|those|these|this one|the same|one of them)\b/i,
  /^(tell me more|more|more info|more details|go on|continue|elaborate)\b/i,
  /^(what else|anything else|any other)\b/i,
  /^(when is|where is|how much|how long|how many)\s+(it|that|this|the)\b/i,
];
const STOPWORDS = new Set([
  'a','an','the','i','me','my','we','you','your','is','am','are','was','were','be','been','being',
  'do','does','did','to','of','in','on','at','for','by','with','from','that','this','these','those',
  'and','or','but','so','as','if','it','its','they','them','their','there','then','than','about',
  'have','has','had','how','what','where','when','why','who','which','can','could','should','would',
  'will','just','also','like','please','thanks','thank','help','need','want','know','tell','say',
  'really','still','yet','now','here','some','any','all','each','every','no','not','too','very',
  'get','got','go','going','come','make','made','take','took','give','gave','find','found','okay','ok','yes',
]);
function extractKeywords(text, max = 6) {
  if (!text) return [];
  return String(text)
    .toLowerCase()
    .replace(/[^\w\s£$€@.:\/-]/g, ' ')
    .split(/\s+/)
    .filter(w => w.length > 2 && !STOPWORDS.has(w))
    .slice(0, max);
}
function isFollowUp(raw) {
  const text = String(raw || '').trim();
  if (!text) return false;
  const wordCount = text.split(/\s+/).filter(Boolean).length;
  if (wordCount <= 6) return true;                            // elliptical or terse
  return FOLLOWUP_TRIGGERS.some(rx => rx.test(text));         // connector/pronoun
}
// Branded / domain-specific terms we want to preserve as subject anchors.
// Matching is case-insensitive; the returned value keeps the canonical casing
// from this list so the LLM sees a clean brand term when we substitute for a
// pronoun ("how do I access it?" -> "how do I access HallPad?").
//
// Split into two lists:
//   • NAMED_ENTITIES  — specific, unambiguous nouns. Safe to carry across
//     *both* user AND assistant turns in the retrieval lead, because their
//     presence in an assistant reply means the assistant is talking about
//     that exact topic (and the student implicitly adopted it).
//   • TOPIC_QUALIFIERS — generic modifiers ("postgraduate", "accommodation",
//     "Swansea"). We only pull them in from USER turns, because a qualifier
//     mentioned incidentally in an assistant reply can drag retrieval
//     sideways (e.g. "accommodation" in a reply about scholarships).
const NAMED_ENTITIES = [
  'HallPad','Moodle','MyTSD','PCYDDS','Wellbeing','IT Services',
  'Student Finance Wales','Student Finance','Student Services','HEFCW',
  'SA1 Waterfront','Dynevor','ALSS','UCAS','DSA','VLE','Fforwm','Y Fforwm',
  'Mount Pleasant','Townhill','Clearing','HallPad portal','Founders Library',
  'Swansea College of Art','Additional Learning Support','Careers',
  'Disabled Students Allowance','Programme Manager','personal tutor',
  'Myunijourney',
];
const TOPIC_QUALIFIERS = [
  // Institutions & campuses
  'UWTSD','SA1','Carmarthen','Lampeter','Swansea','Cardiff','Birmingham',
  'London','Trinity Saint David',
  // Common courses / subject areas (lowercase + titlecase so both match)
  'Nursing','nursing','Engineering','engineering','Business','business',
  'Psychology','psychology','Art and Design','art and design',
  'Applied Computing','Computing','computing','Education','education',
  'Law','law','Architecture','architecture','Film','film','Acting','acting',
  'Performing Arts','Games Design','Games Art','Animation','animation',
  'Cyber Security','Computer Science','Sport','sport','Tourism','tourism',
  'Marketing','marketing','Accounting','accounting','Finance','finance',
  'Health','health','Social Work','social work','Counselling','counselling',
  'Teacher Training','PGCE','MBA','PhD','masters','Masters',
  // Life / admin topics
  'accommodation','postgraduate','undergraduate','tuition','scholarship',
  'bursary','halls','fees','dissertation','assignment','coursework',
  'deadline','deadlines','timetable','timetables','library','libraries',
  'wifi','printing','parking','bus','transport','shuttle','enrolment',
  'enrollment','graduation','open day','clearing','visa','placement',
  'internship','exam','exams','revision','results','transcript','diploma',
  'references','reference','application','apply','offer','interview',
];
const UWTSD_BRAND_TERMS = [...NAMED_ENTITIES, ...TOPIC_QUALIFIERS];
function extractSubjectAnchor(text) {
  if (!text) return null;
  const t = String(text);
  // 1) Prefer UWTSD/branded nouns (case-insensitive match, canonical return)
  for (const term of UWTSD_BRAND_TERMS) {
    const rx = new RegExp('\\b' + term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '\\b', 'i');
    if (rx.test(t)) return term;
  }
  // 2) Any TitleCase phrase (2-3 words) looks like a proper noun
  const m = t.match(/\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})\b/);
  if (m) return m[1];
  // 3) Last resort: last content word before the question mark —
  //    usually the topic noun ("is there a student bus?" -> "bus").
  const kw = extractKeywords(t, 12);
  return kw[kw.length - 1] || null;
}
// Scan recent history for a concrete UWTSD topic the student has raised.
// Used by the casual-vent detector to decide whether "ugh" / "damn" is a
// fresh vent (→ return warm catch-all) or a reaction to the previous topic
// (→ let the LLM respond with empathy grounded in the actual subject).
// Returns the branded/topic term found, or null.
function findPriorTopicAnchor(history) {
  if (!Array.isArray(history)) return null;
  const userTurns = history.filter(t => t && t.role === 'user' && t.text).slice(-3);
  for (const t of userTurns.reverse()) {
    for (const term of UWTSD_BRAND_TERMS) {
      const rx = new RegExp('\\b' + term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '\\b', 'i');
      if (rx.test(t.text)) return term;
    }
  }
  return null;
}

// Extract self-disclosed student context from the current message + recent
// user turns.  This surfaces facts the student has volunteered ("my name is
// Ataur", "I'm an international postgrad", "I live in Swansea", "I'm graduating
// this year") so we can inject them into the LLM prompt as an explicit
// STUDENT CONTEXT block.  The LLM can then tailor its answer instead of
// asking questions the student already answered.
//
// Kept intentionally conservative: only extracts signals with very high
// precision (named self-intro patterns, explicit student-type phrases) so
// we don't hallucinate context from ambiguous prose.
function extractStudentContext(currentText, history) {
  const ctx = {};
  const sources = [currentText || ''];
  if (Array.isArray(history)) {
    for (const t of history.slice(-6)) {
      if (t && t.role === 'user' && t.text) sources.push(t.text);
    }
  }
  const combined = sources.join(' \n ');
  const lower    = combined.toLowerCase();

  // Name: "my name is X", "i'm X", "i am X", "this is X", "call me X"
  //       "fy enw yw X", "fi yw X"
  // Guard: only accept if the captured name is 1-3 TitleCase tokens and isn't
  // a known demonym/course word.  Many false positives look like "I am a
  // student" → captured "a Student" — we filter those below.
  const BAD_NAMES = new Set([
    'a','an','the','student','international','home','uk','welsh','english','not',
    'sure','confused','graduating','applying','looking','trying','studying',
    'interested','hoping','here','there','new','old','final','first','second','third',
    'fourth','year','myfyriwr','rhyngwladol','cartref','newydd',
  ]);
  // Case-insensitive prefix match, but we verify the captured name starts
  // with an uppercase letter afterwards (so "I am a student" is rejected
  // but "I am Ataur" is accepted).  The captured group keeps original case.
  const namePatterns = [
    /\bmy\s+name\s+is\s+([A-Za-z'’-]{2,20}(?:\s+[A-Za-z'’-]{2,20}){0,2})/i,
    /\bi['’]?m\s+([A-Za-z'’-]{2,20}(?:\s+[A-Za-z'’-]{2,20}){0,2})\b/i,
    /\bi\s+am\s+([A-Za-z'’-]{2,20}(?:\s+[A-Za-z'’-]{2,20}){0,2})\b/i,
    /\bthis\s+is\s+([A-Za-z'’-]{2,20}(?:\s+[A-Za-z'’-]{2,20}){0,2})\b/i,
    /\bcall\s+me\s+([A-Za-z'’-]{2,20})/i,
    /\bfy\s+enw\s+(?:i\s+)?yw\s+([A-Za-z'’-]{2,20}(?:\s+[A-Za-z'’-]{2,20}){0,2})/i,
  ];
  for (const rx of namePatterns) {
    const m = combined.match(rx);
    if (m) {
      const first = m[1].split(/\s+/)[0];
      // Require an uppercase first letter (TitleCase) so common-word false
      // positives like "I'm a student" / "I am confused" get rejected.
      if (first && !BAD_NAMES.has(first.toLowerCase()) && /^[A-Z][a-z]/.test(first)) {
        ctx.name = first;
        break;
      }
    }
  }

  // Student type
  if (/\b(international|overseas|non[-\s]?uk)\s+(student|applicant|postgrad|undergrad)/i.test(combined) ||
      /\bi['\u2019]?m\s+(an?\s+)?international\b/i.test(combined) ||
      /\bmyfyriwr\s+rhyngwladol\b/i.test(combined)) {
    ctx.studentType = 'international';
  } else if (/\b(uk|home|welsh|domestic)\s+(student|applicant)/i.test(combined) ||
             /\bi['\u2019]?m\s+(a\s+)?(uk|home|welsh)\s+(student|applicant)/i.test(combined) ||
             /\bmyfyriwr\s+cartref\b/i.test(combined)) {
    ctx.studentType = 'home';
  }

  // Level of study
  if (/\b(post.?grad(uate)?|masters?|phd|doctoral|doctorate|mphil|pgce)\b/i.test(combined)) {
    ctx.level = 'postgraduate';
  } else if (/\b(undergrad(uate)?|bachelors?|first\s+degree)\b/i.test(combined)) {
    ctx.level = 'undergraduate';
  }

  // Year / graduation status.  Accepts verb-ish ("graduating this year") and
  // the common mis-phrasing students use ("I am graduation this year").
  if (/\b(graduating|graduation|graduate)\s+(this\s+year|this\s+summer|soon|in\s+(july|december)|eleni)\b/i.test(combined) ||
      /\b(i['\u2019]?m|i\s+am)\s+(graduating|graduation)\b/i.test(lower) ||
      /\b(rwy['\u2019]?n|dw\s+i['\u2019]?n)\s+graddio\s+eleni\b/i.test(lower)) {
    ctx.year = 'graduating';
  } else if (/\b(final\s+year|last\s+year)\b/i.test(lower)) {
    ctx.year = 'final-year';
  } else if (/\b(first\s+year|fresher|just\s+started)\b/i.test(lower)) {
    ctx.year = 'first-year';
  } else if (/\b(second\s+year)\b/i.test(lower))  ctx.year = 'second-year';
  else if   (/\b(third\s+year)\b/i.test(lower))   ctx.year = 'third-year';

  // Campus
  const campusMatch = combined.match(/\b(?:at|on|in|based\s+(?:at|in))\s+(Swansea|Carmarthen|Lampeter|Cardiff|Birmingham|Llambed|Caerfyrddin|Abertawe|Caerdydd)\b/i);
  if (campusMatch) {
    const raw = campusMatch[1].toLowerCase();
    const map = { 'llambed':'Lampeter','caerfyrddin':'Carmarthen','abertawe':'Swansea','caerdydd':'Cardiff' };
    ctx.campus = map[raw] || (raw.charAt(0).toUpperCase() + raw.slice(1));
  }

  // Course of interest — the single most useful signal for follow-up
  // questions.  "I want to apply for Nursing" → every subsequent "what
  // are the entry requirements" stays nursing-specific.  We match a known
  // list of UWTSD course families; free-form capture is too noisy.
  const COURSE_PATTERNS = [
    { rx: /\b(applied\s+computing|computer\s+science|computing)\b/i,     label: 'Computing' },
    { rx: /\b(nursing|midwifery|healthcare)\b/i,                           label: 'Nursing / Healthcare' },
    { rx: /\b(business|management|mba|marketing|accounting|finance)\b/i,   label: 'Business' },
    { rx: /\b(psychology|counselling|social\s+work)\b/i,                   label: 'Psychology / Counselling' },
    { rx: /\b(engineering|mechanical|electrical|civil|automotive)\b/i,     label: 'Engineering' },
    { rx: /\b(art(?:s)?\s+(and|&)\s+design|fine\s+art|graphic\s+design)\b/i, label: 'Art & Design' },
    { rx: /\b(games\s+(art|design)|animation)\b/i,                         label: 'Games / Animation' },
    { rx: /\b(film|acting|performing\s+arts|drama|theatre)\b/i,            label: 'Film / Performing Arts' },
    { rx: /\b(education|teacher\s+training|pgce|teaching)\b/i,             label: 'Education / Teaching' },
    { rx: /\b(law|criminology)\b/i,                                        label: 'Law' },
    { rx: /\b(architecture|built\s+environment|construction)\b/i,          label: 'Architecture / Construction' },
    { rx: /\b(sport|fitness|coaching)\b/i,                                 label: 'Sport' },
    { rx: /\b(tourism|hospitality|event(s)?\s+management)\b/i,             label: 'Tourism / Hospitality' },
  ];
  for (const { rx, label } of COURSE_PATTERNS) {
    if (rx.test(combined)) { ctx.course = label; break; }
  }

  // Topic stack — ordered list (oldest → newest) of distinct UWTSD topics
  // the student has raised so far.  Helps the LLM see the journey, not
  // just the latest turn.  We only extract from USER turns (the student's
  // own words), newest last, dedup while preserving order.
  const TOPIC_SIGNALS = [
    { rx: /\b(accommodation|halls|residence|hallpad)\b/i,                      label: 'accommodation' },
    { rx: /\b(fee|tuition|cost|price|bursary|scholarship|funding|loan)\b/i,    label: 'fees/funding' },
    { rx: /\b(wellbeing|stress|anxious|mental|crisis|lonely|homesick)\b/i,     label: 'wellbeing' },
    { rx: /\b(library|study\s+space|books?|founders)\b/i,                      label: 'library' },
    { rx: /\b(wifi|moodle|mytsd|password|login|portal|printing|print)\b/i,     label: 'IT services' },
    { rx: /\b(timetable|class(es)?|lecture|seminar|attendance)\b/i,            label: 'timetable' },
    { rx: /\b(dissertation|thesis|research\s+project)\b/i,                     label: 'dissertation' },
    { rx: /\b(assignment|coursework|submission|turnitin|deadline)\b/i,         label: 'assignments' },
    { rx: /\b(apply|application|ucas|offer|interview|entry\s+requirement|clearing)\b/i, label: 'applying' },
    { rx: /\b(visa|cas|biometric|international)\b/i,                           label: 'international/visa' },
    { rx: /\b(campus|carmarthen|lampeter|swansea|sa1|mount\s+pleasant|cardiff)\b/i, label: 'campus info' },
    { rx: /\b(club|society|union|activit(y|ies)|social|sport)\b/i,             label: 'student life' },
    { rx: /\b(parking|bus|shuttle|travel|transport)\b/i,                       label: 'transport' },
    { rx: /\b(graduation|graduating|ceremony)\b/i,                             label: 'graduation' },
  ];
  const topicOrder = [];
  const userSources = Array.isArray(history)
    ? history.filter(t => t && t.role === 'user' && t.text).map(t => t.text)
    : [];
  userSources.push(currentText || '');
  for (const text of userSources) {
    for (const { rx, label } of TOPIC_SIGNALS) {
      if (rx.test(text) && !topicOrder.includes(label)) topicOrder.push(label);
    }
  }
  if (topicOrder.length) ctx.topicStack = topicOrder.slice(-5);

  return ctx;
}

// Format the extracted student context as a short, explicit section for the
// LLM prompt.  Returns null if nothing useful was extracted.
function formatStudentContext(ctx, lang) {
  if (!ctx || Object.keys(ctx).length === 0) return null;
  const bits = [];
  if (ctx.name)        bits.push(lang === 'cy' ? `Enw: ${ctx.name}` : `Name: ${ctx.name}`);
  if (ctx.studentType) bits.push(lang === 'cy'
                          ? `Math o fyfyriwr: ${ctx.studentType === 'international' ? 'rhyngwladol' : 'cartref/DU'}`
                          : `Student type: ${ctx.studentType}`);
  if (ctx.level)       bits.push(lang === 'cy'
                          ? `Lefel: ${ctx.level === 'postgraduate' ? 'ôl-raddedig' : 'israddedig'}`
                          : `Level: ${ctx.level}`);
  if (ctx.year)        bits.push(lang === 'cy' ? `Blwyddyn: ${ctx.year}` : `Year/status: ${ctx.year}`);
  if (ctx.campus)      bits.push(lang === 'cy' ? `Campws: ${ctx.campus}` : `Campus: ${ctx.campus}`);
  if (ctx.course)      bits.push(lang === 'cy' ? `Maes diddordeb: ${ctx.course}` : `Course of interest: ${ctx.course}`);
  if (ctx.topicStack && ctx.topicStack.length) {
    bits.push(lang === 'cy'
      ? `Pynciau a drafodwyd: ${ctx.topicStack.join(' → ')}`
      : `Topics discussed so far: ${ctx.topicStack.join(' → ')}`);
  }
  if (!bits.length) return null;
  const header = lang === 'cy'
    ? 'CYD-DESTUN Y MYFYRIWR (mae\'r myfyriwr eisoes wedi dweud hyn — defnyddia ef, paid â gofyn eto):'
    : 'STUDENT CONTEXT (the student has already told you this — USE it, do not ask again):';
  return `${header}\n- ${bits.join('\n- ')}`;
}

// ─── Emotional state detection ──────────────────────────────────────────
// Classify the student's emotional register so the LLM can match it in tone.
// We read the current message PLUS the last 2 user turns — emotion builds
// across the conversation ("I'm confused" + "I don't understand" → frustrated).
// Returns one of:
//   'distressed'  — crisis / despair language → warm, slow down, signpost
//                   support before any facts
//   'stressed'    — exam / dissertation / deadline panic → acknowledge first
//   'frustrated' — bot not helping, asked same thing twice → apologise, retry
//   'confused'    — "I don't get it", "what do you mean"   → simplify
//   'excited'     — "can't wait", "so happy" → match the energy
//   'neutral'     — no signal
//
// Kept deliberately keyword-based and cheap — the LLM does the real work
// of tone-matching; this just flags what register to use.
const EMOTION_KEYWORDS = {
  distressed: [
    /\b(want\s+to\s+(?:die|quit|drop\s*out|give\s*up))\b/i,
    /\b(can(?:'|)?t\s+cope|cant\s+cope|breaking\s+down|falling\s+apart|can(?:'|)?t\s+do\s+this(?:\s+any\s*more)?)\b/i,
    /\b(suicidal|hopeless|no\s+point|rock\s+bottom|crisis|desperate)\b/i,
    /\b(wedi\s+cael\s+digon|methu\s+ymdopi|methu\s+dal\s+ati)\b/i, // Welsh
  ],
  stressed: [
    /\b(stressed|stressful|anxious|anxiety|worried|worries|worrying|panic|panicking|panick?ed|overwhelmed|overwhelm|burn(?:t|ed)?\s*out|exhausted|drained|scared|terrified)\b/i,
    /\b(too\s+much|can(?:'|)?t\s+sleep|losing\s+sleep|pressure|deadline\s+is\s+killing)\b/i,
    /\b(wedi\s+blino|ofnus|pryderus|dan\s+bwysau|straen)\b/i, // Welsh
  ],
  frustrated: [
    /\b(frustrated|frustrating|annoyed|annoying|angry|mad|furious|pissed|fed\s*up|sick\s+of|tired\s+of|ridiculous|useless|waste\s+of\s+time|rubbish|terrible|awful)\b/i,
    /\b(you('|\s*a)re?\s+(?:not\s+)?(?:understanding|helping|wrong)|asked\s+(?:this|that)\s+already|already\s+asked)\b/i,
    /\b(wedi\s+laru|dig|blin|dwp)\b/i, // Welsh
  ],
  confused: [
    /\b(confused|confusing|don'?t\s+(?:get|understand)|doesn'?t\s+make\s+sense|makes?\s+no\s+sense|lost|unclear|not\s+sure\s+what)\b/i,
    /\b(what\s+(?:do\s+you\s+mean|does\s+that\s+mean)|can\s+you\s+explain)\b/i,
    /\b(dydw\s+i\s+ddim\s+yn\s+deall|methu\s+deall|ar\s+goll)\b/i, // Welsh
  ],
  excited: [
    /\b(excited|can(?:'|)?t\s+wait|so\s+happy|thrilled|pumped|love\s+it|amazing|awesome|brilliant|fantastic|yay)\b/i,
    /\b(cyffrous|methu\s+aros|mor\s+hapus|gwych|bendigedig)\b/i, // Welsh
  ],
};
function detectEmotionalState(currentText, history) {
  const sources = [currentText || ''];
  if (Array.isArray(history)) {
    for (const t of history.filter(t => t && t.role === 'user' && t.text).slice(-2)) {
      sources.push(t.text);
    }
  }
  const blob = sources.join(' \n ');
  // Priority order — distressed first (safety-critical), then stressed,
  // then negative states, finally positive.  First match wins.
  for (const state of ['distressed','stressed','frustrated','confused','excited']) {
    for (const rx of EMOTION_KEYWORDS[state]) {
      if (rx.test(blob)) return state;
    }
  }
  return 'neutral';
}

// Short, prompt-ready guidance for each emotional state.  Placed into the
// LLM context so it matches the student's register.  We keep the
// instructions compact — llama3.1:8b follows 1-2 sentence rules reliably
// but ignores paragraph-long instructions.
function formatEmotionalGuidance(state, lang) {
  if (!state || state === 'neutral') return null;
  const cy = {
    distressed:  "CYFLWR EMOSIYNOL: TRALLODUS. Arafa, cydnabya'r teimlad yn gynnes yn gyntaf, yna cyfeiria at wellbeingsupport@uwtsd.ac.uk neu 01792 481 111. PAID â thaflu ffeithiau cwrs ato/ati.",
    stressed:    "CYFLWR EMOSIYNOL: DAN STRAEN. Dechrau gyda chydnabyddiaeth ('Mae hynny'n swnio'n anodd'), yna rho gam un ymarferol. Paid ag ychwanegu mwy o fanylion na sy'n ofynnol.",
    frustrated:  "CYFLWR EMOSIYNOL: RHWYSTREDIG. Ymddiheura'n fyr os aeth rhywbeth o'i le, yna ateba'r cwestiwn go iawn yn uniongyrchol — paid â gofyn am eglurhad eto.",
    confused:    "CYFLWR EMOSIYNOL: DRYSLYD. Eglura mewn geiriau syml, un cam ar y tro. Dim jargon. Gall fod yn briodol dechrau gyda 'Wrth gwrs'.",
    excited:     "CYFLWR EMOSIYNOL: CYFFROUS. Matsia'r egni, byr ac optimistaidd. Paid â mynd yn rhy ffurfiol.",
  };
  const en = {
    distressed:  "EMOTIONAL STATE: DISTRESSED. Slow down, warmly acknowledge the feeling FIRST, then signpost wellbeingsupport@uwtsd.ac.uk or 01792 481 111. Do NOT dump course/admin facts.",
    stressed:    "EMOTIONAL STATE: STRESSED. Open with acknowledgement ('That sounds tough — you're not alone'), then give ONE practical next step. Keep it short.",
    frustrated:  "EMOTIONAL STATE: FRUSTRATED. Briefly apologise if the bot was unhelpful, then answer the real question directly — never ask for clarification again.",
    confused:    "EMOTIONAL STATE: CONFUSED. Explain in simple words, one step at a time. No jargon. It's fine to start with 'Of course' or 'No worries'.",
    excited:     "EMOTIONAL STATE: EXCITED. Match the energy — short, upbeat, enthusiastic. Don't slip into formal corporate tone.",
  };
  return (lang === 'cy' ? cy : en)[state] || null;
}

function rewriteWithContext(raw, history) {
  if (!Array.isArray(history) || history.length === 0) return raw;
  if (!isFollowUp(raw)) return raw;

  // Pull keywords from the last two user turns + last assistant turn.
  // User turns carry the strongest topic signal; assistant turns fill in
  // specific nouns (HallPad, Wellbeing, clearing, etc.).
  const recent        = history.slice(-5);
  const userTurns     = recent.filter(t => t.role === 'user'      && t.text).slice(-2);
  const lastAssist    = [...recent].reverse().find(t => t.role === 'assistant' && t.text);
  const lastUser      = userTurns[userTurns.length - 1];
  const userKw        = userTurns.flatMap(t => extractKeywords(t.text, 5));
  const assistKw      = extractKeywords(lastAssist?.text, 4);
  const rawKw         = new Set(extractKeywords(raw, 10));
  // User keywords carry the student's actual intent.  Assistant keywords
  // can drag retrieval sideways if the assistant rambled into a tangent —
  // so we use them only for the [context:] tag guiding Ollama, NOT for the
  // TF-IDF lead.  Branded/domain terms from the assistant reply are an
  // exception: they are *named topics* the conversation established, and
  // we do want them in the lead so retrieval keeps the topic locked in
  // (e.g. assistant said "HallPad accommodation" -> next turn should keep
  // those terms even if the student's short follow-up doesn't repeat them).
  const userAnchor    = [...new Set(userKw)].filter(w => !rawKw.has(w)).slice(0, 8);
  const fullCtx       = [...new Set([...userKw, ...assistKw])]
    .filter(w => !rawKw.has(w))
    .slice(0, 10);
  if (!userAnchor.length && !fullCtx.length) return raw;

  // Find a concrete subject noun for pronoun resolution.
  //
  // Priority (most specific → most generic):
  //   1. NAMED_ENTITY in the last user turn  (e.g. "I cannot access Moodle")
  //   2. NAMED_ENTITY in the last assistant reply
  //      (e.g. "I need accommodation" → reply names "HallPad" → we adopt it)
  //   3. TOPIC_QUALIFIER or any TitleCase/content word from the last user turn
  //   4. Same fallback from the last assistant reply
  //
  // Without this priority, "in Carmarthen" (a TOPIC_QUALIFIER) would win on
  // step 1 and block HallPad (NAMED_ENTITY) from the assistant reply on step
  // 2, so the retrieval lead for "when is the deadline?" ends up dominated
  // by generic qualifiers and drifts to the UCAS-deadline corpus chunk
  // instead of the accommodation-deadline one.
  const findNamedEntity = (text) => {
    if (!text) return null;
    for (const term of NAMED_ENTITIES) {
      const rx = new RegExp('\\b' + term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '\\b', 'i');
      if (rx.test(text)) return term;
    }
    return null;
  };
  const subjectAnchor =
       findNamedEntity(lastUser?.text)
    || findNamedEntity(lastAssist?.text)
    || extractSubjectAnchor(lastUser?.text)
    || extractSubjectAnchor(lastAssist?.text);

  // Collect branded / domain terms from history to anchor retrieval.
  // We split this by term class:
  //   • NAMED_ENTITIES (HallPad, ALSS, Wellbeing…) are specific and safe
  //     from BOTH user and assistant turns — if the assistant named one,
  //     the student implicitly adopted that topic.
  //   • TOPIC_QUALIFIERS (postgraduate, accommodation, Swansea…) are
  //     generic modifiers, so we pull them only from USER turns. A
  //     qualifier the assistant mentioned in passing can otherwise drag
  //     retrieval sideways (e.g. "accommodation" in a scholarship reply).
  // We scan the full history (not just the last 5 turns) so that e.g.
  // "I need accommodation" at turn 0 still anchors "when is the
  // deadline?" at turn 3.
  const historyTopics = new Set();
  for (const t of history) {
    if (!t.text) continue;
    for (const term of NAMED_ENTITIES) {
      const rx = new RegExp('\\b' + term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '\\b', 'i');
      if (rx.test(t.text)) historyTopics.add(term);
    }
    if (t.role === 'user') {
      for (const term of TOPIC_QUALIFIERS) {
        const rx = new RegExp('\\b' + term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '\\b', 'i');
        if (rx.test(t.text)) historyTopics.add(term);
      }
    }
  }

  // Pronoun rewrite: "how do I access it?" -> "how do I access HallPad?"
  let rewritten = raw;
  if (subjectAnchor && /\b(it|that|this|they|them|these|those|one)\b/i.test(raw)) {
    rewritten = raw.replace(
      /\b(it|that|this|they|them|these|those|one)\b/gi,
      subjectAnchor
    );
  }

  // Lead = repeated subject anchor (TF-IDF boost) + recent branded topics +
  // user-turn keywords.  The rewritten question follows.  [context:] tag
  // holds the fuller user+assistant keyword mix for Ollama's coreference.
  // We only *double* the anchor when it's a NAMED_ENTITY (HallPad, ALSS…);
  // doubling a TOPIC_QUALIFIER like "postgraduate" over-weights the lead
  // toward generic-qualifier passages (alumni discounts, etc.) instead of
  // the actual topic the student is asking about.
  const leadParts = [];
  if (subjectAnchor) {
    const isNamedEntity = NAMED_ENTITIES.some(
      term => term.toLowerCase() === String(subjectAnchor).toLowerCase()
    );
    if (isNamedEntity) leadParts.push(subjectAnchor, subjectAnchor);
    else leadParts.push(subjectAnchor);
  }
  leadParts.push(...[...historyTopics].slice(0, 4));
  leadParts.push(...userAnchor.slice(0, 5));
  const lead = leadParts.join(' ').trim() || userAnchor.slice(0, 5).join(' ');
  return `${lead} ${rewritten} [context: ${fullCtx.join(' ')}]`;
}

// Knowledge-base fallback — uses the pre-trained TF-IDF + Bayes classifier to
// match the query against the 55 UWTSD intents in knowledge.json.  This is the
// offline safety net: it runs entirely locally with zero network calls, so we
// can always produce *some* relevant answer even if every external service is
// down.
// Boost intent scores based on continuity with recent history.
// Two tiers:
//   1. Specific-term matching — if the student mentioned "Computing" and a
//      candidate intent's name or knowledge entry mentions "computing", 2x
//      boost (strong signal).  Stops "what are the entry requirements?"
//      after a Computing question from routing to courses_nursing.
//   2. Family matching — broader category (accommodation, fees, wellbeing)
//      gets 1.3x boost.  Keeps follow-ups in the same topic area without
//      overriding strong signals in the current message.
function applyHistoryIntentBoost(top, history) {
  if (!Array.isArray(history) || history.length === 0) return top;
  const recentText = history.slice(-4)
    .filter(t => t && t.text)
    .map(t => t.text.toLowerCase())
    .join(' ');
  if (!recentText) return top;

  // Tier 1: specific course / subject terms.  Each entry = [term that must
  // appear in recent text, list of tag name fragments that identify the
  // matching intents in knowledge.json].
  const SPECIFIC_TERMS = [
    { term: /\b(applied\s+computing|computer\s+science|computing|cyber|games)\b/i, tags: ['comput','cyber','games'] },
    { term: /\b(nursing|midwifery|healthcare)\b/i,                                   tags: ['nursing','health'] },
    { term: /\b(business|management|mba|marketing|finance|accounting)\b/i,           tags: ['business','mba'] },
    { term: /\b(psychology|counselling|social\s+work)\b/i,                           tags: ['psychology','counsell','social_work'] },
    { term: /\b(engineering|mechanical|electrical|civil|automotive)\b/i,             tags: ['engineer'] },
    { term: /\b(art|design|graphic|fine\s+art|fashion)\b/i,                          tags: ['art','design'] },
    { term: /\b(film|acting|performing\s+arts|drama|theatre)\b/i,                    tags: ['film','acting','performing'] },
    { term: /\b(education|teacher|pgce|teaching)\b/i,                                tags: ['education','pgce','teach'] },
    { term: /\b(law|criminology)\b/i,                                                tags: ['law','crimin'] },
    { term: /\b(architecture|construction|built\s+environment)\b/i,                  tags: ['architect','construct'] },
    { term: /\b(sport|fitness|coaching)\b/i,                                         tags: ['sport','fitness'] },
  ];
  const activeSpecific = SPECIFIC_TERMS
    .filter(({ term }) => term.test(recentText))
    .flatMap(({ tags }) => tags);

  // Tier 2: family hints (broader — used when no specific course matched).
  const FAMILY_HINTS = {
    accommodation: /\b(accommodation|halls|residence|hallpad|neuadd|llety)\b/i,
    fees:          /\b(fee|tuition|cost|price|ffioedd|scholarship|bursary)\b/i,
    wellbeing:     /\b(wellbeing|stress|anxious|mental|lles|support|crisis)\b/i,
    it:            /\b(wifi|moodle|mytsd|password|login|portal|it\s+services)\b/i,
    library:       /\b(library|llyfrgell|book|study\s+space|founders)\b/i,
    campus:        /\b(campus|carmarthen|lampeter|swansea|sa1|mount\s+pleasant|townhill|cardiff)\b/i,
    apply:         /\b(apply|application|ucas|offer|entry\s+requirement|clearing)\b/i,
    graduation:    /\b(graduation|graduating|ceremony|graddio)\b/i,
  };
  const activeFamilies = Object.entries(FAMILY_HINTS)
    .filter(([, rx]) => rx.test(recentText))
    .map(([fam]) => fam);

  return top.map(item => {
    const tagLower = (item.tag || '').toLowerCase();
    // Tier 1: specific term match (strong boost — must beat generic
    // admissions_* entries that tie on surface keywords).
    if (activeSpecific.length && activeSpecific.some(t => tagLower.includes(t))) {
      return { ...item, score: item.score * 2.5 };
    }
    // Tier 1 penalty: if a specific course was mentioned but THIS tag
    // belongs to a different course family (e.g. nursing when user said
    // "Computing"), cut score sharply so the right course wins.
    const isOtherCourse = activeSpecific.length && tagLower.startsWith('courses_') &&
      !activeSpecific.some(t => tagLower.includes(t));
    if (isOtherCourse) {
      return { ...item, score: item.score * 0.3 };
    }
    // Tier 2: family match (softer boost — enough to beat generic
    // catch-alls like postgraduate_courses when a category was recent).
    if (activeFamilies.some(fam => tagLower.includes(fam))) {
      return { ...item, score: item.score * 1.5 };
    }
    return item;
  });
}

function knowledgeFallback(message, lang, history) {
  let top = getTopIntents(message, lang, 5, history);
  if (!top.length) return null;

  // Use whichever NER we have available to boost the right intents
  const ner = welshNER(message);
  top = applyNERBoost(top, ner.entities);
  // History continuity boost — keeps follow-ups in the same topic family
  top = applyHistoryIntentBoost(top, history);
  top.sort((a, b) => b.score - a.score);

  const best = top[0];
  if (!best || best.score < THRESHOLD_FALLBACK) return null;

  const response = getResponse(best.tag, lang);
  if (!response) return null;

  return {
    tag:        best.tag,
    score:      best.score,
    response,
    altResponse: getResponse(best.tag, lang === 'cy' ? 'en' : 'cy'),
    clarify:    best.score < THRESHOLD_CLARIFY,
  };
}

// Main chat handler — resilient RAG+LLM pipeline.
//
// Every real question goes through the same RAG flow: we retrieve relevant
// UWTSD passages from BOTH the live Morphik backend AND the local
// harvested corpus, merge them, and ask Ollama to synthesise a natural answer.
// Because the corpus ships with the deployment, retrieval works even when the
// live Morphik tunnel or Ollama is unreachable — the system just picks the
// best available tier:
//
//   • Crisis keywords     → hardcoded safe response
//   • Smalltalk / greet   → local warm reply (no network)
//   • RAG + Ollama        → live Morphik + local corpus + CoreNLP + Ollama
//   • RAG only            → top corpus passage composed into a natural answer
//   • Knowledge intents   → 55 canonical UWTSD intents (TF-IDF match)
//   • Last-resort prompt  → CLARIFICATION + FALLBACK text
async function processChat(message, runningLang, history) {
  const raw = (message || '').trim();
  if (!raw) {
    return {
      response:    FALLBACK.en,
      altResponse: FALLBACK.cy,
      tag: 'empty', lang: 'en', confidence: 0, source: 'empty',
    };
  }

  const lang = detectLanguage(raw, runningLang);
  const alt  = lang === 'cy' ? 'en' : 'cy';

  // ── Crisis override ──────────────────────────────────────────────────────
  // Always checked first, never bypassed by any other tier.
  if (isCrisis(raw)) {
    return {
      response:    getResponse('wellbeing_crisis', lang),
      altResponse: getResponse('wellbeing_crisis', alt),
      tag: 'wellbeing_crisis', lang, confidence: 1.0, source: 'safety',
    };
  }

  // ── Smalltalk ────────────────────────────────────────────────────────────
  // Short greetings / thanks handled instantly without hitting Ollama.
  const chit = detectSmallTalk(raw);
  if (chit) {
    return {
      response:    chit[lang],
      altResponse: chit[alt],
      tag:         chit.tag,
      lang,
      confidence:  1.0,
      source:      'smalltalk',
    };
  }

  // Sanitise conversation history for downstream use (moved up so casual-vent
  // and every other check can read it).
  let safeHistory = [];
  if (Array.isArray(history)) {
    safeHistory = history
      .filter(t => t && (t.role === 'user' || t.role === 'assistant') && typeof t.text === 'string')
      .slice(-6).map(t => ({ role: t.role, text: t.text.slice(0, 600) }));
  }

  // ── Casual/emotional venting ─────────────────────────────────────────────
  // Catch short frustrated or emotional expressions that aren't questions and
  // don't match any specific intent.  When the LLM is available it handles
  // these naturally (the system prompt has CASUAL AND EMOTIONAL INPUTS rules).
  // When there's no LLM, return the short warm CLARIFICATION immediately so
  // the student gets "Sounds like something's on your mind…" rather than a
  // random knowledge-base answer or a topic-dump FALLBACK.
  //
  // History-aware: if the previous turns established a concrete topic, a
  // follow-up like "ugh" is probably frustration ABOUT that topic, not a
  // fresh vent.  In that case we skip the short-circuit and let the LLM
  // respond with an empathetic reply GROUNDED in the actual topic.
  const CASUAL_EXPR_RE = /^(damn|dang|omg|wtf|ugh+|argh+|oh\s*no|oh\s*man|oh\s*wow|blimey|crikey|jeez|geez|yikes|shoot|crap|phew|hmm+|lol|haha+|lmao|sigh|oof|noo+|help!*|bobl\s*bach|duw\s*annwyl|o\s*diar|wedi\s*laru|mawredd)(\s*(man|bro|mate|dude|fam|lads|tbh))?[!?.]*$/i;
  const rawWords    = raw.trim().split(/\s+/);
  const hasTopic    = /\b(fee|cost|where|when|how|what|who|which|campus|course|apply|accommodation|loan|library|wifi|moodle|timetable|submit|deadline|dissertation|assignment|tutor|wellbeing|crisis|graduat|enrol|activit|club|sport|parking|bus|travel|print|card)\b/i.test(raw);
  // A "prior topic" = there's at least one recent user turn with a concrete
  // anchor we can hang this vent onto.  If so, treat it as a continuation,
  // not a fresh vent.
  const priorAnchor = safeHistory.length
    ? findPriorTopicAnchor(safeHistory)
    : null;
  const isCasualVent = !hasTopic && !priorAnchor && (
    CASUAL_EXPR_RE.test(raw.trim()) ||
    (rawWords.length <= 3 && !/[?]/.test(raw))   // ultra-short non-question
  );

  // When there's no LLM at all, short-circuit here with a warm reply.
  if (isCasualVent && !OLLAMA_URL) {
    const casualEn = lang === 'en'
      ? "Sounds like something's on your mind — what's going on? I'm here to help with anything UWTSD-related 😊"
      : "Mae'n swnio fel bod rhywbeth ar dy feddwl — beth sy'n digwydd? Rwy'n yma i helpu gyda phopeth PCYDDS 😊";
    const casualAlt = lang === 'cy'
      ? "Sounds like something's on your mind — what's going on? I'm here to help with anything UWTSD-related 😊"
      : "Mae'n swnio fel bod rhywbeth ar dy feddwl — beth sy'n digwydd? Rwy'n yma i helpu gyda phopeth PCYDDS 😊";
    return {
      response:    casualEn,
      altResponse: casualAlt,
      tag: 'casual_vent', lang, confidence: 0.9, source: 'casual_handler',
    };
  }
  // When LLM IS available, fall through — the CASUAL AND EMOTIONAL INPUTS
  // rules in the system prompt ensure the model handles it warmly.

  // ── Cross-lingual NLU (Welsh → English pivot) ────────────────────────────
  // When the student writes in Welsh, we translate the query + recent
  // history to English once and run ALL downstream NLU (topic match,
  // context rewrite, pronoun anchor, Morphik/corpus retrieval, CoreNLP,
  // and Ollama generation) in English. Rationale:
  //   1. The UWTSD corpus (Morphik + local passages) is overwhelmingly
  //      English — we lack Welsh coverage for most university content.
  //   2. NAMED_ENTITIES / TOPIC_QUALIFIERS lists and CoreNLP are English.
  //   3. llama3.1:8b understands English much better than Welsh.
  // The answer is translated back to Welsh at the very end so the
  // student still sees a Welsh reply. If any individual translation
  // fails we fall back transparently to the original text — every
  // downstream step tolerates that.
  let rawEn = raw;
  let safeHistoryEn = safeHistory;
  if (lang === 'cy') {
    const recent = safeHistory.slice(-4);
    try {
      const [curEn, ...histEn] = await Promise.all([
        translateOllama(raw, 'en'),
        ...recent.map(t => translateOllama(t.text, 'en')),
      ]);
      if (curEn) rawEn = curEn;
      const translatedRecent = recent.map((t, i) => ({
        role: t.role,
        text: histEn[i] || t.text,
      }));
      safeHistoryEn = recent.length < safeHistory.length
        ? [...safeHistory.slice(0, safeHistory.length - recent.length), ...translatedRecent]
        : translatedRecent;
    } catch (_) {
      // Translator hiccup — stay on the original Welsh text; the
      // BydTermCymru augmenter still provides term-level English hints.
    }
  }

  // ── NLU topic classification ─────────────────────────────────────────────
  // 602 bilingual topics (built by scripts/build-nlu-topics.py).  If a topic
  // ships a polished canned reply (greetings, thanks etc.) AND the query is
  // short enough that a canned reply is appropriate, return it without the
  // LLM.  Longer real questions always go through RAG.
  //
  // IMPORTANT — do NOT fire the fast-path for follow-up turns.  A short
  // message like "for postgrad?" or "do I pay for it?" can only be
  // answered in light of the prior turn, and a canned topic reply will
  // always ignore that.  Let those fall through to full RAG.
  const isCtxFollowUp = safeHistoryEn.length > 0 && isFollowUp(rawEn);
  // Topic classification runs on the English-translated query so the
  // English-keyword topic index matches reliably (it used to miss when
  // Welsh tokens fell outside the bilingual keyword set).
  const topicMatch = classifyTopic(rawEn);
  const shortEnoughForCanned = raw.split(/\s+/).length <= 5;
  if (!isCtxFollowUp && topicMatch && topicMatch.topic.reply_en && shortEnoughForCanned && topicMatch.score >= 3.0) {
    const t = topicMatch.topic;
    return {
      response:    t[lang === 'cy' ? 'reply_cy' : 'reply_en'] || t.reply_en,
      altResponse: t[alt  === 'cy' ? 'reply_cy' : 'reply_en'] || t.reply_cy || t.reply_en,
      tag:         `nlu:${t.id}`,
      lang,
      confidence:  Math.min(1, topicMatch.score),
      source:      'nlu_topic',
    };
  }

  // ── Curated facts (fast-path, bilingual) ────────────────────────────────
  // Check the curated uwtsd-facts.json file BEFORE calling the LLM.  These
  // entries already have polished bilingual answers (answer_en + answer_cy),
  // so returning them directly gives a guaranteed-Welsh reply for Welsh users
  // instead of an English LLM response that then has to round-trip through
  // MyMemory (which can rate-limit or fail on long text).  Skips context-
  // dependent follow-ups ("how much?", "and for postgrad?") so they still
  // flow through retrieval + LLM where pronoun resolution lives.
  //
  // Uses a STRICT threshold (0.55) for the fast-path so only high-confidence
  // matches skip retrieval.  Generic queries like "tell me about UWTSD" or
  // "what courses are available" score weakly against several facts and
  // should flow through the LLM instead of getting a misrouted canned reply.
  // The fallback fact lookup later in this function keeps the normal 0.28
  // threshold as a last-resort answer when the LLM is unavailable.
  if (!isCtxFollowUp) {
    // Tuning notes:
    //   minScore 1.0   — filter out generic single-token matches ("UWTSD"
    //                    alone scores ~5 across many facts; a real fact
    //                    hit (specific question wording) scores 10-60).
    //   minRatio 1.25  — require the top match to be at least 25% ahead
    //                    of the runner-up.  "How do I apply to UWTSD?"
    //                    scores ~60 on how-to-apply vs ~15 on the fees
    //                    entries (ratio ~4) — clear hit.  Cluttered
    //                    queries like "tell me about UWTSD" score ~6
    //                    everywhere (ratio ~1) — pass through to LLM.
    // minScore 3.0 — a single-token query like "information please" scores
    // ~3 on whichever fact happens to contain that word; real question
    // matches score 10-60.  Setting the floor at 3.0 lets those drifters
    // fall through to the NLU/retrieval path instead of anchoring a canned
    // reply on one accidental token.
    const factHit = lookupFact(rawEn, { minScore: 3.0, minRatio: 1.15 });
    if (factHit) {
      // Wellbeing and academic-support facts need a warm, personalised LLM response
      // rather than a generic canned contact-list. We pass the curated fact as
      // grounding context so the LLM stays accurate but the tone is empathetic and
      // conversational — it acknowledges the student's specific situation and, for
      // academic queries, asks ONE focused clarifying question to understand their need.
      const isWellbeingFact  = factHit.id && factHit.id.startsWith('wellbeing');
      const isAcademicFact   = factHit.id === 'academic-support';
      const needsLLMResponse = (isWellbeingFact || isAcademicFact) && OLLAMA_URL;
      if (needsLLMResponse) {
        let factsContext;
        if (isAcademicFact) {
          factsContext =
            `UWTSD ACADEMIC SUPPORT INFORMATION:\n${factHit.answer_en}\n\n` +
            `Use the above information to respond to the student. ` +
            `STEP 1 — Acknowledge what they said warmly (1 sentence). ` +
            `STEP 2 — Ask ONE focused clarifying question to understand exactly what they need ` +
            `(e.g. "Is this about getting feedback on the content itself, or more about the ` +
            `deadline/submission process?" OR "Would you like to speak to your dissertation ` +
            `supervisor, or is it more about how you're feeling about the workload?"). ` +
            `STEP 3 — Give a brief pointer to the right next step. ` +
            `Keep it concise (3 sentences total). Do NOT just list contacts without context.`;
        } else {
          factsContext =
            `UWTSD WELLBEING INFORMATION:\n${factHit.answer_en}\n\n` +
            `Use the above information to help the student. ` +
            `Acknowledge their specific situation warmly and personally. ` +
            `Do NOT just list the contact details — first empathise with what they said, ` +
            `then gently guide them to the right support. Keep it concise (3–4 sentences).`;
        }
        // Welsh pivot: small models (llama3.1:8b, Groq) struggle to generate
        // natural Welsh when the context is in English.  Instead we run the LLM
        // in English (which is reliable) then translate the reply to Welsh with
        // MyMemory — identical to what the main corpus path does.  This prevents
        // the model from starting with "Hwyl!" / "Goodbye!" and producing broken
        // mixed-language output.
        const llmLang = lang === 'cy' ? 'en' : lang;
        const llmMsg  = lang === 'cy' ? rawEn : raw;
        const llmHist = lang === 'cy' ? safeHistoryEn : safeHistory;

        const llmResult = await askLLM(llmMsg, factsContext, llmLang, llmHist);
        if (llmResult && llmResult.reply) {
          let primaryResp, altResp;
          if (lang === 'cy') {
            // Translate English LLM reply → Welsh for primary; English is alt.
            const welshReply = await translateOllama(llmResult.reply, 'cy');
            // If translation produced something that looks Welsh, use it.
            // Otherwise fall back to the canned Welsh fact answer so the user
            // always sees something in Welsh rather than English.
            primaryResp = (welshReply && looksWelsh(welshReply))
              ? welshReply
              : (factHit.answer_cy || factHit.answer_en);
            altResp = llmResult.reply;
          } else {
            primaryResp = llmResult.reply;
            altResp = factHit.answer_cy
              || await translateOllama(llmResult.reply, 'cy')
              || '';
          }
          return {
            response:    primaryResp,
            altResponse: altResp,
            tag:         `fact:${factHit.id}`,
            lang,
            confidence:  0.92,
            source:      `uwtsd_facts+${llmResult.backend}`,
          };
        }
        // LLM unavailable — fall through to the canned fact answer below
      }

      let primary = lang === 'cy' ? (factHit.answer_cy || factHit.answer_en) : factHit.answer_en;
      let altAns  = lang === 'cy' ? factHit.answer_en : (factHit.answer_cy || await translateOllama(factHit.answer_en, 'cy'));

      // If the student just introduced themselves by name on *this* turn,
      // prepend a short, warm greeting so the fast-path canned answer still
      // feels conversational.  We only do this for fresh introductions
      // (checking current text, not history) to avoid over-using the name
      // across long sessions.
      const freshCtx = extractStudentContext(rawEn, []);
      if (freshCtx.name) {
        const greetEn = `Hi ${freshCtx.name} — `;
        const greetCy = `Helô ${freshCtx.name} — `;
        if (lang === 'cy' && !primary.includes(freshCtx.name)) {
          primary = greetCy + primary.charAt(0).toLowerCase() + primary.slice(1);
        } else if (lang !== 'cy' && !primary.includes(freshCtx.name)) {
          primary = greetEn + primary.charAt(0).toLowerCase() + primary.slice(1);
        }
        if (altAns && typeof altAns === 'string' && !altAns.includes(freshCtx.name)) {
          const altGreet = lang === 'cy' ? greetEn : greetCy;
          altAns = altGreet + altAns.charAt(0).toLowerCase() + altAns.slice(1);
        }
      }

      return {
        response:    primary,
        altResponse: altAns,
        tag:         `fact:${factHit.id}`,
        lang,
        confidence:  0.92,
        source:      'uwtsd_facts',
      };
    }
  }

  // Build the retrieval query from the English form (either the native
  // English input, or the CY→EN translation for Welsh users). If the
  // translator was down we fall back to BydTermCymru term-level
  // augmentation so retrieval still has some English hooks to match on.
  const langBase       = (lang === 'cy' && rawEn === raw) ? augmentWelshQuery(raw) : rawEn;
  const contextBase    = rewriteWithContext(langBase, safeHistoryEn);
  const retrievalQuery = topicMatch
    ? augmentWithTopicHint(contextBase, topicMatch.topic)
    : contextBase;

  // Derive a subject anchor to hand to Ollama alongside the raw message.
  // This lets the LLM resolve pronouns consistently even if the retrieval
  // context happens to include a tangential topic.  We also fire for short
  // topic-less follow-ups like "how much?", "when is it?", "and for postgrad?"
  // which humans treat as implicit references even without a pronoun.
  const PRONOUN_RE    = /\b(it|that|this|they|them|these|those|one|there)\b/i;
  const SHORT_FOLLOWUP = (() => {
    const words = rawEn.trim().split(/\s+/);
    if (words.length > 6) return false;
    // Short elliptical queries that clearly need context from earlier.
    return /^(how much|what about|and|how long|when|where|why|who)\b/i.test(rawEn) ||
           /^(postgraduate|undergraduate|international|home|part-time|full-time|online)\b/i.test(rawEn);
  })();
  // Scan backwards through recent history for a concrete anchor — skip turns
  // that are too short or are pure chit-chat.  Prefer user turns (what THEY
  // said about their own topic) over assistant turns.
  function findAnchorInHistory(history) {
    const userTurns = history.filter(t => t.role === 'user').reverse();
    for (const t of userTurns.slice(0, 3)) {
      const a = extractSubjectAnchor(t.text);
      if (a && a.length > 2) return a;
    }
    const assistantTurns = history.filter(t => t.role === 'assistant').reverse();
    for (const t of assistantTurns.slice(0, 2)) {
      const a = extractSubjectAnchor(t.text);
      if (a && a.length > 2) return a;
    }
    return null;
  }
  const ollamaAnchor = (isCtxFollowUp && (PRONOUN_RE.test(rawEn) || SHORT_FOLLOWUP))
    ? findAnchorInHistory(safeHistoryEn)
    : null;

  // ── Retrieval (parallel) ─────────────────────────────────────────────────
  // Pull from live Morphik, the local UWTSD corpus, and NLP enrichment all at
  // once.  Every branch is tolerant of failure: if Morphik times out or the
  // corpus is empty, the others carry on.  For Welsh users we send the
  // English-translated query to every retriever so the (overwhelmingly
  // English) corpus and Morphik index match properly — this is the core
  // of the cross-lingual NLU pivot.
  const [enrichment, morphikContext] = await Promise.all([
    enrichWithCoreNLP(rawEn, 'en'),
    queryMorphik(retrievalQuery),
  ]);

  // Local corpus retrieval also benefits from the context-rewritten query so
  // that follow-ups like "when is the deadline?" don't drift to a generic
  // deadline page.  We pass lang='en' for Welsh users (when CY→EN worked)
  // so TF-IDF preprocesses the query with English stemming, matching the
  // English-indexed UWTSD passages.
  const corpusLang     = (lang === 'cy' && rawEn !== raw) ? 'en' : lang;
  const corpusQuery    = rewriteWithContext(rawEn, safeHistoryEn);
  const corpusHitsRaw  = await retrieveFromCorpus(corpusQuery, corpusLang, 6);
  // Safety sanitation BEFORE the LLM sees retrieved text: strip PII, drop
  // chunks with injection attempts.  The sanitised set is what we pass to
  // the LLM and to composeCorpusAnswer; the raw hits are kept only if we
  // need to audit a safety decision.
  const { clean: corpusHits, dropped: corpusDropped } = safety.sanitisePassages(corpusHitsRaw);
  if (corpusDropped.length && process.env.DEBUG_SAFETY) {
    console.log(`[safety] Dropped ${corpusDropped.length} chunk(s): ${corpusDropped.map(d => `${d.id}(${d.reason})`).join(', ')}`);
  }
  const usedMorphik    = Boolean(morphikContext);
  const usedCorpus     = corpusHits.length > 0;
  const nlpAnnotations = buildNLPContext(enrichment, lang);
  const corpusBlock    = formatCorpusForPrompt(corpusHits);

  // ── RAG + LLM (primary) ─────────────────────────────────────────────────
  // Uses Ollama as the sole LLM backend — if offline, falls through to
  // curated facts / corpus.
  if (OLLAMA_URL) {
    const sections = [];
    // Merge the two retrieval sources into a single UWTSD INFORMATION block
    // — duplicate headers confuse the model and bloat the prompt.  Morphik
    // (live) wins over cached corpus when both have content; if only one is
    // available, use that.
    const mergedInfo = [morphikContext, corpusBlock].filter(Boolean).join('\n\n').slice(0, 1600);
    if (mergedInfo) {
      sections.push(`UWTSD INFORMATION:\n${mergedInfo}`);
    }
    if (nlpAnnotations) {
      sections.push(nlpAnnotations.trim());
    }
    // Final layer: a couple of hand-curated intent responses as supporting
    // context for edge cases the scraped pages don't cover directly.
    // Score against the English form so the English keyword list in
    // getTopIntents matches reliably; context block stays in the user's
    // display language so the LLM doesn't code-switch.
    const intentLang = (lang === 'cy' && rawEn !== raw) ? 'en' : lang;
    const top = getTopIntents(rawEn, intentLang, 3, safeHistoryEn);
    if (top.length) {
      const kbText = buildContext(top.slice(0, 2), 'en').slice(0, 900);
      if (kbText) sections.push(`CURATED INTENT NOTES:\n${kbText}`);
    }

    // Question-type hint: tell Ollama exactly what kind of answer is needed
    // so it filters the passages correctly rather than summarising everything.
    // "Where is X?" → give a specific address/directions.
    // "How much?" → give a specific number or price.
    // "How do I?" → give step-by-step instructions or a contact/URL.
    const questionType = detectQuestionType(rawEn);
    const QTYPE_INSTRUCTION = {
      LOCATION:   'The student is asking for a LOCATION or ADDRESS. Extract and state the specific address, postcode, building name, or directions. Do NOT describe the university in general.',
      QUANTITY:   'The student is asking for a SPECIFIC AMOUNT or FIGURE (e.g. fees, cost). State the exact number or range. Do NOT give vague descriptions.',
      DATE:       'The student is asking for a SPECIFIC DATE, TIME, or DEADLINE. State it directly. Do NOT describe processes unless asked.',
      PROCESS:    'The student is asking HOW TO DO something. Give clear steps, and always end with the most relevant contact (email/phone/URL).',
      PERSON:     'The student is asking WHO to contact. Name the role and give their contact details (email/phone). If you cannot name a specific person, direct them to the right team.',
      YESNO:      'The student is asking YES or NO. Start your answer with "Yes," or "No," then explain briefly.',
      DEFINITION: 'The student is asking WHAT something is. Give a clear one-sentence definition, then add the most relevant detail.',
      GENERAL:    null,
    };
    const qtInstruction = QTYPE_INSTRUCTION[questionType];
    if (qtInstruction) {
      sections.push(`QUESTION TYPE — ${questionType}:\n${qtInstruction}`);
    }

    // Pronoun-resolution hint: if the student used "it/that/this/they/...",
    // tell the LLM explicitly what the pronoun refers to. This sits in the
    // context block (not the system prompt) so it's tied to this specific
    // turn and doesn't linger across conversation.
    if (ollamaAnchor) {
      sections.push(
        `PRONOUN RESOLUTION:\nIn the student's next message, any pronoun (it / that / this / they / them) refers to: "${ollamaAnchor}". Answer about "${ollamaAnchor}", NOT about any other topic that happens to appear in the passages above.`
      );
    }

    // Student-disclosed context: name, student type, level, year, campus.
    // When the student tells us "my name is Ataur and I'm graduating this
    // year", we surface those facts as a dedicated section so the LLM uses
    // them naturally (greeting by name, answering the implied topic) rather
    // than ignoring them as chat noise.
    const studentCtx = extractStudentContext(rawEn, safeHistoryEn);
    const studentCtxBlock = formatStudentContext(studentCtx, lang);
    if (studentCtxBlock) {
      sections.push(studentCtxBlock);
    }

    // Emotional-state guidance: tells the LLM what register to match.  Runs
    // on the native-language text so Welsh emotion keywords ("dan straen",
    // "ar goll") are picked up even when we pivot retrieval through English.
    const emotionalState   = detectEmotionalState(raw, safeHistory);
    const emotionalBlock   = formatEmotionalGuidance(emotionalState, lang);
    if (emotionalBlock) {
      sections.push(emotionalBlock);
    }

    const context = sections.length
      ? sections.join('\n\n---\n\n')
      : 'No external context available — answer from general UWTSD knowledge.';

    // ── Native-language generation ───────────────────────────────────────
    // Pass the user's actual language to the LLM so the Welsh system prompt
    // kicks in and the model replies in Welsh directly.  Retrieval context
    // stays in English (corpus/Morphik is indexed in English) — the Welsh
    // system prompt tells the LLM explicitly to read English context and
    // answer in natural idiomatic Welsh.  This avoids the MyMemory round-
    // trip entirely in the happy path.  If the LLM slips back into English
    // (detectable via heuristic below) we fall back to the pivot strategy.
    const userMsgForLLM    = lang === 'cy' ? raw : rawEn;
    const historyForLLM    = lang === 'cy' ? safeHistory : safeHistoryEn;
    let llmResult = await askLLM(userMsgForLLM, context, lang, historyForLLM);
    let llmReply  = llmResult ? llmResult.reply : null;
    let backend   = llmResult ? llmResult.backend : null;

    // Reject lazy clarification responses when the user asked a specific
    // question (>= 4 words and not purely interjection).  Ollama sometimes
    // bounces concrete queries ("where can I park?", "what are opening hours?")
    // with "could you be more specific" — useless.  Retry once with an explicit
    // instruction to answer directly.
    const wordCount = rawEn.trim().split(/\s+/).length;
    const clarifyPatterns = /(could you be (a bit )?more specific|can you (please )?clarify|allech chi fod (ychydig )?yn fwy penodol|allwch chi egluro)/i;
    const looksLikeClarification = llmReply && (
      clarifyPatterns.test(llmReply) ||
      /^(sorry[ ,—-]+)?(could|can) you/i.test(llmReply) ||
      /^(mae'?n ddrwg gen i[ ,—-]+)?(allech|allwch)/i.test(llmReply)
    );
    if (looksLikeClarification && wordCount >= 4) {
      const retryContext = context +
        (lang === 'cy'
          ? '\n\nGORCHYMYN GORFODOL: NID YW\'R CWESTIWN YN AMWYS. Rho ateb uniongyrchol, defnyddiol nawr gan ddefnyddio\'r cyd-destun neu wybodaeth gyffredinol am PCYDDS (enquiries@uwtsd.ac.uk, 01792 481 111, uwtsd.ac.uk). PAID â gofyn am eglurhad.'
          : '\n\nMANDATORY OVERRIDE: THE QUESTION IS NOT AMBIGUOUS. Give a direct, helpful answer now using the context or general UWTSD knowledge (enquiries@uwtsd.ac.uk, 01792 481 111, uwtsd.ac.uk). DO NOT ask for clarification.');
      const retryResult = await askLLM(userMsgForLLM, retryContext, lang, historyForLLM);
      const retry = retryResult ? retryResult.reply : null;
      if (retry && !clarifyPatterns.test(retry)) {
        llmReply = retry;
        backend  = retryResult.backend;
      }
    }

    // Safety net: if the user asked in Welsh but the LLM replied in English
    // (older llama3.1 sometimes ignores the Welsh system prompt), OR it
    // produced a looping / rambling Welsh response (common llama3.1:8b
    // failure mode on long Welsh prompts), fall back to the pivot-through-
    // English strategy — generate in English, then translate to Welsh via
    // MyMemory.  English generation is more reliable on small models and
    // MyMemory doesn't loop.
    if (lang === 'cy' && llmReply && (!looksWelsh(llmReply) || hasRepetition(llmReply))) {
      const reason = !looksWelsh(llmReply) ? 'not_welsh' : 'repetition';
      console.warn(`[Welsh safety-net] LLM reply rejected (${reason}) — pivoting through English`);
      const enResult = await askLLM(rawEn, context, 'en', safeHistoryEn);
      if (enResult && enResult.reply && !hasRepetition(enResult.reply)) {
        const welsh = await translateOllama(enResult.reply, 'cy');
        if (welsh && looksWelsh(welsh)) {
          llmReply = welsh;
          backend  = `${enResult.backend}+translate`;
        }
      }
    }

    if (llmReply) {
      // Grounding check: what fraction of the LLM's content tokens appear
      // in the retrieved passages?  Very low overlap suggests the model
      // ignored our context and is making things up.  We don't block on
      // this — we surface it to callers as a confidence signal and log
      // it when DEBUG_GROUNDING=1 for offline tuning.
      const groundingPool = [
        morphikContext || '',
        ...corpusHits.map(h => h.content),
      ];
      const grounding = safety.checkGrounding(llmReply, groundingPool.map(c => ({ content: c })));
      if (process.env.DEBUG_GROUNDING) {
        console.log(`[grounding] ratio=${grounding.ratio.toFixed(2)} supported=${grounding.supported}/${grounding.total}`);
      }

      // Compose primary/alt based on the user's language.  The LLM now
      // replies in the user's language directly; the alt side needs a
      // translation round-trip (cheap, and non-critical for UX).
      let primary, altResp;
      if (lang === 'cy') {
        primary = llmReply;                                   // native Welsh
        altResp = await translateOllama(llmReply, 'en') || llmReply;
      } else {
        primary = llmReply;                                   // native English
        altResp = await translateOllama(llmReply, 'cy');
      }
      const source = [
        usedMorphik ? 'morphik'     : null,
        usedCorpus  ? 'uwtsd_cache' : null,
        CORENLP_URL ? 'corenlp'     : null,
        backend,                      // 'ollama' / 'ollama+translate'
        lang === 'cy' ? 'cy_native' : null,
      ].filter(Boolean).join('+');
      // Confidence is nudged down when grounding is poor — callers can
      // use this to decide whether to show the answer alongside a
      // disclaimer, or fall through to the curated-facts tier.
      let confidence = usedMorphik ? 0.95 : (usedCorpus ? 0.85 : 0.55);
      if (grounding.total >= 10 && grounding.ratio < 0.15) {
        confidence = Math.min(confidence, 0.45);
      }
      return {
        response:    primary,
        altResponse: altResp,
        tag:         usedMorphik || usedCorpus ? 'morphik_rag' : backend,
        lang,
        confidence,
        grounding:   { ratio: Number(grounding.ratio.toFixed(3)), supported: grounding.supported, total: grounding.total },
        source,
      };
    }
  }

  // ── Curated facts lookup (offline-first) ────────────────────────────────
  // Before falling back to raw corpus sentences, check the hand-curated
  // uwtsd-facts.json file.  If the query matches a fact entry above the
  // confidence threshold, return the polished pre-written answer directly.
  // These answers are always clean and synthesised — no data-dump risk.
  //
  // The minRatio gate mirrors the fast-path (line 2151): a short ambiguous
  // query like "gwybodaeth plis" produces a near-tie across several facts;
  // insisting the winner be 1.25× ahead of runner-up lets those cases fall
  // through to the retrieval+LLM path instead of crowning an arbitrary one.
  const factHit = lookupFact(rawEn, { minRatio: 1.15 });
  if (factHit) {
    const primary = lang === 'cy' ? (factHit.answer_cy || factHit.answer_en) : factHit.answer_en;
    const altAns  = lang === 'cy' ? factHit.answer_en : (factHit.answer_cy || await translateOllama(factHit.answer_en, 'cy'));
    return {
      response:    primary,
      altResponse: altAns,
      tag:         `fact:${factHit.id}`,
      lang,
      confidence:  0.92,
      source:      'uwtsd_facts',
    };
  }

  // ── Curated intent match (before raw corpus dump) ────────────────────────
  // Check knowledge.json FIRST when no LLM is available.  This prevents
  // the Morphik raw-dump path below from winning on generic queries like
  // "time management strategies" where Morphik returns semantically wrong
  // course-listing passages (matching "management" as a subject keyword).
  // A curated student_stress / study-skills answer is far better than a
  // raw page dump for queries that have nothing to do with course listings.
  const kb = knowledgeFallback(raw, lang, safeHistory);
  if (kb && !kb.clarify && kb.score >= 0.35) {
    return {
      response:    kb.response,
      altResponse: kb.altResponse,
      tag:         kb.tag,
      lang,
      confidence:  Math.min(0.7, 0.3 + kb.score),
      source:      'knowledge_base',
    };
  }

  // ── RAG without the LLM ──────────────────────────────────────────────────
  // Ollama is down (or not configured), but we still have Morphik data.
  // Compose a natural answer directly from the top retrieved passages — the
  // user asked a real question, so we give them a real answer from the
  // UWTSD pages, not a generic apology.
  if (usedCorpus && corpusHits[0].score > 0.15) {
    const directAnswer = composeCorpusAnswer(corpusHits, lang, rawEn);
    if (directAnswer) {
      const altAnswer = await translateOllama(directAnswer, alt);
      return {
        response:    directAnswer,
        altResponse: altAnswer,
        tag:         'morphik_rag',
        lang,
        confidence:  Math.min(0.8, 0.4 + corpusHits[0].score),
        source:      usedMorphik ? 'morphik+uwtsd_cache' : 'uwtsd_cache',
      };
    }
  }

  // If live Morphik did return chunks, surface them — but only if the top
  // passage is actually relevant (guard against course-listing false hits).
  // We check the first chunk's char-level relevance heuristic: if Morphik
  // returned content but it's all course catalogue boilerplate without any
  // words from the query, prefer the knowledge-base answer below.
  if (usedMorphik) {
    const queryWords = rawEn.toLowerCase().split(/\s+/).filter(w => w.length > 3);
    const firstChunk = (morphikContext || '').toLowerCase();
    const morphikRelevant = queryWords.some(w => firstChunk.includes(w));
    if (morphikRelevant) {
      const lead = lang === 'cy'
        ? 'Dyma beth sydd gan wefan PCYDDS ar hyn o bryd:'
        : "Here's what the UWTSD website currently says about that:";
      const body = morphikContext.split('\n\n').slice(0, 2).join('\n\n');
      const directAnswer = `${lead}\n\n${body}`;
      const altAnswer    = await translateOllama(directAnswer, alt);
      return {
        response:    directAnswer,
        altResponse: altAnswer,
        tag:         'morphik_rag',
        lang,
        confidence:  0.75,
        source:      'morphik',
      };
    }
  }

  // ── Curated intent match (low-score fallback) ─────────────────────────────
  // If the high-score check above was not met (score < 0.35), still try the
  // knowledge base as a last resort rather than returning nothing.
  if (kb && !kb.clarify) {
    return {
      response:    kb.response,
      altResponse: kb.altResponse,
      tag:         kb.tag,
      lang,
      confidence:  Math.min(0.5, 0.25 + kb.score),
      source:      'knowledge_base',
    };
  }

  // ── Clarification (low confidence) ───────────────────────────────────────
  if (kb && kb.clarify) {
    return {
      response:    `${kb.response}\n\n${CLARIFICATION[lang]}`,
      altResponse: `${kb.altResponse || ''}\n\n${CLARIFICATION[alt]}`.trim(),
      tag:         'clarification',
      lang,
      confidence:  0.25,
      source:      'knowledge_base+clarify',
    };
  }

  // ── Absolute last resort ─────────────────────────────────────────────────
  return {
    response:    CLARIFICATION[lang] + '\n\n' + FALLBACK[lang],
    altResponse: CLARIFICATION[alt]  + '\n\n' + FALLBACK[alt],
    tag:         'fallback',
    lang,
    confidence:  0,
    source:      'fallback',
  };
}

module.exports = {
  processChat,
  detectLanguage,
  detectQueryQualifiers,
  augmentWelshQuery,
  retrieveFromCorpus,
  classifyTopic,
  lookupFact,
  looksWelsh,
  hasRepetition,
  extractStudentContext,
  formatStudentContext,
  getKnowledgeCount:    () => knowledge.length,
  getBilingualMapSize:  () => Object.keys(WELSH_EN_MAP).length,
  getCorpusSize:        () => UWTSD_CORPUS.length,
  getNLUTopicCount:     () => NLU_TOPICS.length,
  OLLAMA_URL:   () => OLLAMA_URL,
  OLLAMA_MODEL: () => OLLAMA_MODEL,
  CORENLP_URL:  () => CORENLP_URL,
  MORPHIK_URL:  () => MORPHIK_URL,
  // RAG upgrade surfaces — diagnostics for /api/health and test harnesses.
  embedStatus:  () => embed.embedStatus(),
  pickTemperature,
  retrieveFromCorpus,   // exported so regression tests can hit the hybrid path
};
