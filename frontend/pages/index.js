import Head from 'next/head';
import { useState, useRef, useEffect, useCallback } from 'react';

// Constants

const ACCENTS = [
  { name: 'Rust',   hue: 28  },
  { name: 'Olive',  hue: 120 },
  { name: 'Teal',   hue: 190 },
  { name: 'Indigo', hue: 260 },
  { name: 'Plum',   hue: 330 },
];

const TABS = [
  { id: 'demo', num: '01', label: 'Live demo' },
  { id: 'how',  num: '02', label: 'How it works' },
  { id: 'docs', num: '03', label: 'Docs' },
];

// Microsoft Forms URL, paste your form link here
const FORMS_URL = 'https://forms.cloud.microsoft/e/N78VN5GYC4';

const CHIPS = [
  { num: '01', text: 'How do I apply to UWTSD?' },
  { num: '02', text: 'What courses are available?' },
  { num: '03', text: 'Sut mae cofrestru am gwrs?' },
  { num: '04', text: 'Tell me about student support' },
];

const STEPS = [
  {
    n: '01', total: '/ 5',
    title: 'Bilingual language detection',
    desc: 'Every message runs through a multi-signal detector: Unicode characters, Welsh orthographic patterns, function-word matches, and vocabulary coverage from a BydTermCymru-trained map. A single decisive signal flips the response to Welsh.',
    tags: ['function words', 'BydTermCymru', 'auto-detect'],
  },
  {
    n: '02', total: '/ 5',
    title: 'Welsh ↔ English query augmentation',
    desc: 'Welsh terms are mapped to English equivalents before retrieval, so a Welsh query still hits the English-indexed corpus. Bilingual acronyms (UWTSD/PCYDDS) are normalised so both spellings match the same passages.',
    tags: ['bilingual map', 'augment_query()', 'cy→en'],
  },
  {
    n: '03', total: '/ 5',
    title: 'Hybrid semantic retrieval',
    desc: 'Retrieval combines two signals: dense multilingual embeddings (ChromaDB) for semantic meaning, and BM25 lexical matching for exact keyword hits. An ensemble retriever merges both so Welsh exact-match terms and English paraphrases both land the right passage.',
    tags: ['Chroma', 'BM25', 'EnsembleRetriever', 'hybrid'],
  },
  {
    n: '04', total: '/ 5',
    title: 'LLM generation (Claude Haiku 4.5)',
    desc: 'The top passages are passed to Anthropic Claude Haiku 4.5, a fast, inexpensive model that handles Welsh and English fluently. A single fixed model is used so every user gets an identical, fair experience. Ollama Llama 3.1 8B is available as a local fallback for offline demos.',
    tags: ['Claude Haiku 4.5', 'Ollama 3.1 8B', 'fixed model'],
  },
  {
    n: '05', total: '/ 5',
    title: 'Response post-processing',
    desc: 'Output is monolingual-scrubbed, parenthetical self-translations are stripped so Welsh replies never carry English glosses and vice versa. A live translate button lazily calls the same LLM to produce the opposite-language version on demand.',
    tags: ['strip parentheticals', 'live translate', 'monolingual'],
  },
];

const FEATURES = [
  {
    iconKey: 'bilingual',
    title: 'Bilingual by default',
    desc: 'Welsh and English detected per message via a multi-signal detector trained on BydTermCymru terminology. Responses are generated 100% in the detected language, no parenthetical glosses.',
    meta: 'cy + en · BydTermCymru',
  },
  {
    iconKey: 'nollm',
    title: 'Fixed LLM (Claude Haiku 4.5)',
    desc: 'Anthropic Claude Haiku 4.5 powers every reply, fast, inexpensive, and fluent in both Welsh and English. One fixed model so every student gets the same answer quality. Ollama Llama 3.1 8B is available as a local fallback for offline demos.',
    meta: 'Claude Haiku 4.5 · fixed model',
  },
  {
    iconKey: 'edge',
    title: 'Hybrid semantic search',
    desc: 'ChromaDB dense embeddings combine with BM25 keyword matching via LangChain EnsembleRetriever. Catches both semantic paraphrases and exact Welsh vocabulary hits in the same query.',
    meta: 'Chroma · BM25 · Ensemble',
  },
  {
    iconKey: 'feedback',
    title: 'Feedback loop',
    desc: 'Built-in satisfaction ratings stored in MongoDB Atlas (or local JSON). Admin dashboard with CSV export for dissertation analysis.',
    meta: 'MongoDB Atlas · CSV',
  },
];

function FeatIcon({ k }) {
  if (k === 'bilingual') return (
    <svg viewBox="0 0 24 24">
      <path d="M5 7h6M5 12h4M5 17h6" strokeLinecap="round"/>
      <path d="M15 8h4M15 12h4M15 16h4" strokeLinecap="round"/>
      <line x1="12" y1="4" x2="12" y2="20"/>
    </svg>
  );
  if (k === 'nollm') return (
    <svg viewBox="0 0 24 24">
      <rect x="3" y="3" width="7" height="7" rx="1"/>
      <rect x="14" y="3" width="7" height="7" rx="1"/>
      <rect x="3" y="14" width="7" height="7" rx="1"/>
      <path d="M17.5 14v7M14 17.5h7" strokeLinecap="round"/>
    </svg>
  );
  if (k === 'edge') return (
    <svg viewBox="0 0 24 24">
      <polygon points="12 2 22 20 2 20"/>
    </svg>
  );
  return (
    <svg viewBox="0 0 24 24">
      <path d="M12 20h9"/>
      <path d="M16.5 3.5a2.121 2.121 0 013 3L7 19l-4 1 1-4L16.5 3.5z"/>
    </svg>
  );
}

// Doc content

const DOC_GROUPS = [
  {
    title: 'Getting started',
    items: [
      { id: 'introduction', label: 'Introduction' },
      { id: 'quickstart',   label: 'Quick start', pill: 'new' },
    ],
  },
  {
    title: 'Guides',
    items: [
      { id: 'intents',        label: 'Intents & patterns' },
      { id: 'welsh',          label: 'Welsh language' },
      { id: 'feedback-guide', label: 'Feedback system' },
    ],
  },
  {
    title: 'Reference',
    items: [
      { id: 'api-chat',     label: 'POST /api/chat' },
      { id: 'api-feedback', label: 'POST /api/feedback' },
      { id: 'env',          label: 'Environment vars' },
    ],
  },
];

const DOC_ORDER = [
  'introduction', 'quickstart', 'intents', 'welsh',
  'feedback-guide', 'api-chat', 'api-feedback', 'env',
];

function docLabel(id) {
  for (const g of DOC_GROUPS) {
    for (const item of g.items) {
      if (item.id === id) return item.label;
    }
  }
  return id;
}
function docGroup(id) {
  for (const g of DOC_GROUPS) {
    if (g.items.find(i => i.id === id)) return g.title;
  }
  return '';
}

function CodeBlock({ filename, children }) {
  const [copied, setCopied] = useState(false);
  function copy() {
    const text = typeof children === 'string' ? children : '';
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    }).catch(() => {});
  }
  return (
    <div className="code-block">
      <div className="code-head">
        {filename && <span className="filename">{filename}</span>}
        <span className="spacer" />
        <button className="code-copy" onClick={copy}>{copied ? 'copied ✓' : 'copy'}</button>
      </div>
      <div className="code-body">{children}</div>
    </div>
  );
}

function Callout({ label = 'NOTE', children }) {
  return (
    <div className="callout">
      <span className="ico">{label}</span>
      <p>{children}</p>
    </div>
  );
}

function DocTable({ headers, rows }) {
  return (
    <table className="doc-table">
      <thead>
        <tr>{headers.map(h => <th key={h}>{h}</th>)}</tr>
      </thead>
      <tbody>
        {rows.map((row, i) => (
          <tr key={i}>{row.map((cell, j) => <td key={j}>{cell}</td>)}</tr>
        ))}
      </tbody>
    </table>
  );
}

/* Individual doc page content */

function IntroDoc() {
  return (
    <>
      <h1 className="doc-title">Introduction</h1>
      <p className="doc-lede">U-Pal is a bilingual Welsh and English student assistant for UWTSD. It pairs hybrid retrieval (ChromaDB dense embeddings + BM25 sparse keyword scores, fused via Reciprocal Rank Fusion) with Claude Haiku 4.5 for fluent grounded replies.</p>

      <h2 className="doc-h2"><span className="num">01</span>What is U-Pal?</h2>
      <p>U-Pal answers student questions using a FastAPI Python backend. Every answer is grounded in retrieved passages from the UWTSD corpus and the public datasets we enrich it with, the LLM rephrases rather than invents. A bilingual detector routes Welsh and English queries to the correct language, and a live translate button lets users see either version on demand.</p>

      <h2 className="doc-h2"><span className="num">02</span>How a message flows through the backend</h2>
      <p>Every user turn passes through this pipeline:</p>
      <ol>
        <li><strong>Language detection</strong>, multi-signal Welsh detector using Unicode characters, orthographic clusters, function words, and BydTermCymru vocab. The detector overrides any stale frontend hint.</li>
        <li><strong>History filtering</strong>, prior turns in the other language are stripped so they cannot bias the LLM into code-mixing.</li>
        <li><strong>Query augmentation</strong>, Welsh terms get English glosses appended so the English-heavy corpus still matches.</li>
        <li><strong>Hybrid retrieval</strong>, Chroma dense vectors + BM25 sparse scores fused with Reciprocal Rank Fusion (Cormack 2009). English queries exclude cy-tagged passages, Welsh queries restrict to them.</li>
        <li><strong>LLM generation</strong>, Claude Haiku 4.5 by default, with a per-turn language lock wrapped at the head and tail of the prompt. Ollama Llama 3.1 8B is an optional self-hosted fallback.</li>
        <li><strong>Output validation</strong>, the reply is detected and, if the language drifted, translated back via the same translate prompt the public endpoint uses.</li>
      </ol>
      <Callout label="NOTE">Retrieval and language detection run on the operator's own backend. The LLM call is the only external request.</Callout>

      <h2 className="doc-h2"><span className="num">03</span>Tech stack</h2>
      <DocTable
        headers={['Layer', 'Technology']}
        rows={[
          ['Frontend', 'Next.js 14 + React 18 on Vercel'],
          ['Backend', 'FastAPI + Uvicorn (Python 3.11 / 3.12)'],
          ['LLM (primary)', 'Anthropic Claude Haiku 4.5 via langchain-anthropic'],
          ['LLM (fallback)', 'Ollama Llama 3.1 8B, self-hosted, operator-only'],
          ['Dense retrieval', 'ChromaDB with paraphrase-multilingual-MiniLM-L12-v2 embeddings'],
          ['Sparse retrieval', 'rank-bm25 in-memory index, refreshed on each ingest'],
          ['Fusion', 'Reciprocal Rank Fusion (k=60) over dense + sparse rankings'],
          ['Welsh detector', 'Multi-signal detector with English-stopword filter and word-boundary orthographic regex'],
          ['Feedback storage', 'MongoDB Atlas with a local JSON fallback'],
          ['Tunnel', 'ngrok static domains, one per service, configured by the operator'],
        ]}
      />
    </>
  );
}

function QuickstartDoc() {
  return (
    <>
      <h1 className="doc-title">Quick <em>start</em></h1>
      <p className="doc-lede">Clone the public repo and run U-Pal on your own machine in under five minutes. Everything you need, backend, frontend, corpus and scripts, ships with the repo.</p>

      <h2 className="doc-h2"><span className="num">01</span>Prerequisites</h2>
      <ul>
        <li><strong>Node.js 18+</strong> for the Next.js frontend</li>
        <li><strong>Python 3.11 or 3.12</strong> for the FastAPI backend</li>
        <li><strong>Git</strong> to clone the repo</li>
        <li>An <strong>Anthropic API key</strong> (free tier is enough for a demo), from <a href="https://console.anthropic.com/settings/keys" target="_blank" rel="noopener noreferrer">console.anthropic.com</a></li>
        <li><em>Optional</em>, a <strong>MongoDB Atlas</strong> cluster (free M0 tier) for feedback storage. Without it U-Pal writes feedback to a local JSON file.</li>
        <li><em>Optional</em>, <strong>Ollama</strong> + <strong>ngrok</strong> for an offline LLM fallback.</li>
      </ul>

      <h2 className="doc-h2"><span className="num">02</span>Clone the repo</h2>
      <p>The project is public. Anyone can clone and test it:</p>
      <CodeBlock filename="terminal">{`git clone https://github.com/Ataur-Web/U-PAL-PY.git
cd U-PAL-PY`}</CodeBlock>

      <h2 className="doc-h2"><span className="num">03</span>Backend setup (FastAPI)</h2>
      <CodeBlock filename="terminal">{`python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env`}</CodeBlock>
      <p>Edit <code>.env</code> and set <code>ANTHROPIC_API_KEY</code>. Everything else has a sensible default.</p>
      <CodeBlock filename="terminal">{`# first-time only, build the Chroma vector store from the corpus
# (downloads the ~450 MB embedding model on first run)
python -m scripts.ingest --reset

# start the backend (Windows-safe entrypoint)
python run.py`}</CodeBlock>
      <p>The backend is now live at <code>http://localhost:3001</code>. Hit <code>http://localhost:3001/api/health</code> to confirm.</p>

      <h2 className="doc-h2"><span className="num">04</span>Frontend setup (Next.js)</h2>
      <p>In a second terminal:</p>
      <CodeBlock filename="terminal">{`cd frontend
npm install
cp .env.example .env.local   # or create it manually`}</CodeBlock>
      <CodeBlock filename="frontend/.env.local">{`ADMIN_PASSWORD=your-secret-password
MONGODB_URI=                    # optional, omit to use local JSON fallback
CHAT_BACKEND_URL=http://localhost:3001`}</CodeBlock>
      <CodeBlock filename="terminal">{`npm run dev`}</CodeBlock>
      <p>Open <code>http://localhost:3000</code>. Ask a question in English or Welsh and U-Pal will reply.</p>
      <Callout label="TIP">The admin dashboard is at <code>/admin</code>. Use any username and your <code>ADMIN_PASSWORD</code>.</Callout>

      <h2 className="doc-h2"><span className="num">05</span>Deploy the frontend to Vercel</h2>
      <p>Fork the repo, connect it in Vercel, and set <strong>Root Directory</strong> to <code>frontend</code>. Add the environment variables under <strong>Settings → Environment Variables</strong>, then deploy. Every push to <code>main</code> triggers a rebuild.</p>
      <p>For the backend to be reachable from the Vercel frontend you need a public HTTPS URL. The easiest path is <a href="https://ngrok.com" target="_blank" rel="noopener noreferrer">ngrok</a> with a free static domain, point <code>CHAT_BACKEND_URL</code> at that.</p>

      <h2 className="doc-h2"><span className="num">06</span>Optional, local Ollama fallback</h2>
      <p>Install <a href="https://ollama.com" target="_blank" rel="noopener noreferrer">Ollama</a> and pull the default model:</p>
      <CodeBlock filename="terminal">{`ollama pull llama3.1:8b-instruct-q5_K_M
ollama serve`}</CodeBlock>
      <p>Add <code>OLLAMA_URL=http://localhost:11434</code> to your backend <code>.env</code> and the UI will show Ollama as an ACTIVE provider alongside Claude.</p>

      <h2 className="doc-h2"><span className="num">07</span>Optional, broaden the knowledge base</h2>
      <p>The default Chroma index covers the curated UWTSD JSON corpus. Two helper scripts can broaden coverage so U-Pal can chat naturally about general academic topics, not just university-specific ones:</p>
      <CodeBlock filename="terminal">{`# instructional Q&A from OpenOrca (filtered to education topics)
fetch-openorca.bat
# or
python -m scripts.fetch_openorca --sample 20000

# general-knowledge Q&A from Google's Natural Questions
fetch-naturalquestions.bat
# or
python -m scripts.fetch_naturalquestions --sample 30000

# Welsh chat pairs (nemotron-chat-welsh) for natural Welsh phrasing
fetch-welsh-chat.bat
# or
python -m scripts.fetch_welsh_chat --sample 15000`}</CodeBlock>
      <p>All three scripts stream the dataset, filter for student-relevant or Welsh-quality rows, and ingest into Chroma so the next chat turn can retrieve them. Restart the backend afterwards to pick up the enriched index.</p>
      <Callout label="REF">Natural Questions is a Google Research benchmark, see <a href="https://ai.google.com/research/NaturalQuestions" target="_blank" rel="noopener noreferrer">ai.google.com/research/NaturalQuestions</a>. The Welsh chat dataset is published by locailabs at <a href="https://huggingface.co/datasets/locailabs/nemotron-chat-welsh" target="_blank" rel="noopener noreferrer">huggingface.co/datasets/locailabs/nemotron-chat-welsh</a>.</Callout>
    </>
  );
}

function IntentsDoc() {
  return (
    <>
      <h1 className="doc-title">Intents & <em>patterns</em></h1>
      <p className="doc-lede">U-Pal&apos;s curated knowledge lives in <code>app/data/knowledge.json</code>, a flat array of intent objects. Each intent has bilingual patterns (used by the classifier) and bilingual responses (returned to the user).</p>

      <h2 className="doc-h2"><span className="num">01</span>Intent structure</h2>
      <CodeBlock filename="app/data/knowledge.json">{`{
  "tag": "apply",
  "patterns": [
    "How do I apply to UWTSD?",
    "Sut mae gwneud cais?",
    "application process",
    "how to enrol",
    "admission requirements",
    "sut i gofrestru"
  ],
  "responses": {
    "en": ["You can apply via UCAS or directly on the UWTSD website..."],
    "cy": ["Gallwch wneud cais drwy UCAS neu'n uniongyrchol ar wefan PCYDDS..."]
  }
}`}</CodeBlock>

      <h2 className="doc-h2"><span className="num">02</span>Adding a new intent</h2>
      <p>Append a new object to the array. Include at least 10 to 14 patterns per intent, mixing English and Welsh phrasing so the classifier has enough signal in both languages. Then re-run the ingest script:</p>
      <CodeBlock filename="terminal">{`python -m scripts.ingest --reset`}</CodeBlock>
      <p>The ingest reads <code>knowledge.json</code>, splits each response into Chroma documents tagged with their language, and rebuilds the BM25 index in memory.</p>

      <h2 className="doc-h2"><span className="num">03</span>How the classifier scores intents</h2>
      <p>An incoming query is scored against every intent&apos;s patterns using the same hybrid retrieval the chat pipeline uses. The top-scoring intent passes its tag to the LLM as a hint, the LLM is free to ignore it if the retrieved passages contradict.</p>
      <DocTable
        headers={['Confidence band', 'Behaviour']}
        rows={[
          ['Low', 'No tag passed, the LLM answers from retrieval alone'],
          ['Medium', 'Tag passed as a hint, retrieval still drives the answer'],
          ['High', 'Tag passed and the matching response is offered as a baseline'],
        ]}
      />
    </>
  );
}

function WelshDoc() {
  return (
    <>
      <h1 className="doc-title">Welsh <em>language</em></h1>
      <p className="doc-lede">U-Pal detects Welsh using a multi-signal detector backed by BydTermCymru terminology. Several weak signals are combined rather than relying on a single language model, which is more reliable for the short messages students actually type.</p>

      <h2 className="doc-h2"><span className="num">01</span>Detection signals</h2>
      <ul>
        <li><strong>Unicode characters</strong>, any of <code>â ê î ô û ŵ ŷ</code> immediately decides Welsh.</li>
        <li><strong>Orthographic clusters</strong>, word-initial <em>ll</em>, <em>dd</em>, <em>rh</em>, <em>ff</em>, <em>ngh</em>, <em>mh</em>, <em>nh</em> are anchored to word boundaries to avoid matching English words like &quot;hello&quot; or &quot;entry&quot;.</li>
        <li><strong>Function words</strong>, a curated set of unambiguously Welsh words (<em>yr, mae, dw, shwmae, beth, prifysgol, ffioedd</em>). English homographs like <em>hi</em>, <em>fe</em>, <em>na</em>, <em>no</em> are deliberately excluded.</li>
        <li><strong>Vocabulary coverage</strong>, hits in the bilingual map count toward Welsh, but a 90-word English-stopword filter subtracts hits on words like <em>what / help / fees / course</em> so that English questions are not misclassified.</li>
      </ul>

      <h2 className="doc-h2"><span className="num">02</span>Language enforcement during generation</h2>
      <p>The detector runs on every turn and overrides any stale frontend hint. The chat route then:</p>
      <ol>
        <li>Filters conversation history to turns matching the current language so prior turns can&apos;t bias the LLM into code-mixing.</li>
        <li>Wraps the system prompt with a per-turn language lock at both ends.</li>
        <li>Wraps the user message with the same language tag at both ends, transformer attention weighs head and tail tokens heaviest.</li>
        <li>Validates the LLM&apos;s output language and, if it drifted, rewrites it through the translate prompt.</li>
      </ol>

      <h2 className="doc-h2"><span className="num">03</span>Query augmentation and retrieval</h2>
      <p>Welsh queries are augmented with English glosses of known terms before the vector store hit, so a Welsh student asking <em>Beth yw&apos;r ffioedd dysgu?</em> still matches passages indexed against <em>tuition fees</em>. The retriever then runs a cy-only filtered pass on Chroma and tops up with general matches if the cy slice is thin on a topic.</p>

      <h2 className="doc-h2"><span className="num">04</span>Live translation</h2>
      <p>Every bot reply has a translate button that lazily calls <code>POST /api/translate</code>. The same LLM produces an opposite-language version on demand, so the chat endpoint never has to pre-translate.</p>
    </>
  );
}

function FeedbackGuideDoc() {
  return (
    <>
      <h1 className="doc-title">Feedback <em>system</em></h1>
      <p className="doc-lede">U-Pal captures satisfaction ratings, stores them in MongoDB Atlas, and exposes them through a protected admin dashboard.</p>
      <h2 className="doc-h2"><span className="num">01</span>What is collected</h2>
      <DocTable
        headers={['Field', 'Type', 'Description']}
        rows={[
          ['satisfaction', '1–5', 'Star rating from the modal'],
          ['helpfulAnswer', 'boolean | null', 'Was the answer helpful?'],
          ['correctLanguage', 'boolean | null', 'Was the language detected correctly?'],
          ['comments', 'string', 'Free-text comment (optional)'],
          ['timestamp', 'Date', 'Server-side timestamp, added automatically'],
        ]}
      />
      <h2 className="doc-h2"><span className="num">02</span>Storage</h2>
      <p>Feedback is written to MongoDB Atlas via <code>lib/db.js</code>. If <code>MONGODB_URI</code> is not set, the module falls back to a local <code>feedback.json</code> file. This means the app runs fully offline without any cloud dependencies.</p>
      <h2 className="doc-h2"><span className="num">03</span>Admin dashboard</h2>
      <p>Visit <code>/admin</code>. The browser prompts for HTTP Basic Auth, use any username and your <code>ADMIN_PASSWORD</code>. The dashboard shows:</p>
      <ul>
        <li>Average satisfaction score</li>
        <li>Helpful answer percentage</li>
        <li>Correct language percentage</li>
        <li>Paginated feedback table</li>
        <li>CSV export button</li>
      </ul>
    </>
  );
}

function ApiChatDoc() {
  return (
    <>
      <h1 className="doc-title"><em>POST</em> /api/chat</h1>
      <p className="doc-lede">The main chat endpoint. Proxies to the Python backend, which runs language detection → hybrid retrieval → LLM generation and returns a monolingual reply.</p>
      <h2 className="doc-h2"><span className="num">01</span>Request</h2>
      <CodeBlock filename="POST /api/chat">{`Content-Type: application/json

{
  "message":     "How do I apply?",
  "runningLang": "en",
  "history": [
    { "role": "user",      "text": "Tell me about Computing" },
    { "role": "assistant", "text": "UWTSD offers BSc Computing..." }
  ]
}`}</CodeBlock>
      <h2 className="doc-h2"><span className="num">02</span>Response</h2>
      <CodeBlock filename="200 OK">{`{
  "response":    "You can apply via UCAS or the UWTSD website...",
  "altResponse": "",
  "tag":         "admissions",
  "lang":        "en",
  "confidence":  1,
  "source":      "python",
  "emotion":     "neutral",
  "sources":     [ { "title": "corpus", "chars": 480 } ]
}`}</CodeBlock>
      <h2 className="doc-h2"><span className="num">03</span>Response fields</h2>
      <DocTable
        headers={['Field', 'Type', 'Description']}
        rows={[
          ['response', 'string', 'LLM-generated answer in the detected language'],
          ['altResponse', 'string', 'Always "", use POST /api/translate for opposite-language text'],
          ['tag', 'string', 'Matched intent tag from the knowledge base (if any)'],
          ['lang', '"en" | "cy"', 'Detected language of the input'],
          ['emotion', 'string', 'Detected emotional state (neutral, stressed, distressed, …)'],
          ['sources', 'array', 'Retrieved passages, { title, chars }'],
        ]}
      />
      <h2 className="doc-h2"><span className="num">04</span>Errors</h2>
      <DocTable
        headers={['Status', 'Condition']}
        rows={[
          ['400', 'Missing or empty message'],
          ['405', 'Method not POST'],
          ['200 + error_fallback tag', 'Python backend unreachable, Node returns a graceful fallback reply'],
        ]}
      />
    </>
  );
}

function ApiFeedbackDoc() {
  return (
    <>
      <h1 className="doc-title"><em>POST</em> /api/feedback</h1>
      <p className="doc-lede">Save a feedback record (public). GET retrieves records for the admin dashboard (Basic Auth protected).</p>
      <h2 className="doc-h2"><span className="num">01</span>POST: save feedback</h2>
      <CodeBlock filename="POST /api/feedback">{`Content-Type: application/json

{
  "satisfaction":    4,
  "helpfulAnswer":   true,
  "correctLanguage": true,
  "comments":        "Very fast responses!"
}`}</CodeBlock>
      <p>Returns <code>201 Created</code> on success. All fields except <code>satisfaction</code> are optional.</p>
      <h2 className="doc-h2"><span className="num">02</span>GET: retrieve feedback (admin)</h2>
      <CodeBlock filename="GET /api/feedback">{`GET /api/feedback?page=1&limit=20
Authorization: Basic <base64(admin:ADMIN_PASSWORD)>`}</CodeBlock>
      <CodeBlock filename="200 OK">{`{
  "feedback": [ ... ],
  "total": 128,
  "page": 1,
  "pages": 7,
  "stats": {
    "avgSatisfaction": 4.2,
    "helpfulPct": 87,
    "langCorrectPct": 94
  }
}`}</CodeBlock>
      <Callout label="NOTE">The GET endpoint returns <code>401</code> if the <code>Authorization</code> header is missing or the password is wrong. This is also how <code>/api/logout</code> clears the browser Basic Auth cache.</Callout>
    </>
  );
}

function EnvDoc() {
  return (
    <>
      <h1 className="doc-title">Environment <em>variables</em></h1>
      <p className="doc-lede">U-Pal has two environment surfaces: <strong>Vercel</strong> (Node frontend) and the <strong>Python backend</strong> (local FastAPI). The Python backend owns the LLM config; Vercel only needs the tunnel URL + admin secrets.</p>

      <h2 className="doc-h2"><span className="num">01</span>Vercel (frontend)</h2>
      <DocTable
        headers={['Variable', 'Required', 'Description']}
        rows={[
          ['ADMIN_PASSWORD', 'Yes for /admin', 'HTTP Basic Auth password for /admin and GET /api/feedback. Any username is accepted.'],
          ['MONGODB_URI', 'No', 'MongoDB Atlas connection string. Omit to use local file fallback for feedback storage.'],
          ['CHAT_BACKEND_URL', 'Yes', 'Public URL of the Python backend, reached via an ngrok static tunnel. Format: https://<your-backend-domain>.ngrok.app'],
        ]}
      />
      <Callout label="WARN">Never commit <code>.env</code> to git. The <code>.gitignore</code> already excludes it. Treat <code>ADMIN_PASSWORD</code> and <code>ANTHROPIC_API_KEY</code> as secrets.</Callout>

      <h2 className="doc-h2"><span className="num">02</span>Python backend (U-PAL-PY)</h2>
      <DocTable
        headers={['Variable', 'Required', 'Description']}
        rows={[
          ['LLM_PROVIDER', 'No', 'Either "anthropic" (default, cloud) or "ollama" (local). Runtime-switchable via /api/llm-config.'],
          ['ANTHROPIC_API_KEY', 'If using Claude', 'Anthropic API key from console.anthropic.com. Claude Haiku 4.5 is cheap and fast.'],
          ['ANTHROPIC_MODEL', 'No', 'Model ID. Defaults to claude-haiku-4-5-20251001. Other options: claude-3-5-sonnet-latest, claude-sonnet-4-5.'],
          ['OLLAMA_URL', 'If using Ollama', 'Local Ollama URL. For LAN-only use http://localhost:11434. For remote access expose it via your own ngrok static domain.'],
          ['OLLAMA_MODEL', 'No', 'Ollama model tag. Defaults to llama3.1:8b-instruct-q5_K_M.'],
        ]}
      />

      <h2 className="doc-h2"><span className="num">03</span>Running Claude</h2>
      <p>Create an API key at <a href="https://console.anthropic.com/settings/keys" target="_blank" rel="noopener noreferrer">console.anthropic.com/settings/keys</a> and set a monthly spend cap in the Billing tab. Put <code>ANTHROPIC_API_KEY</code> in the Python backend's <code>.env</code>. The frontend never sees it.</p>

      <h2 className="doc-h2"><span className="num">04</span>Running Ollama locally</h2>
      <p>Install <a href="https://ollama.com" target="_blank" rel="noopener noreferrer">Ollama</a> and pull the default model. If you only need Ollama on the same machine as the backend, use it via <code>http://localhost:11434</code> and skip the tunnel. If the frontend is hosted on Vercel, expose Ollama via your own ngrok static domain so the deployed frontend can reach it.</p>
      <CodeBlock filename="terminal">{`ollama pull llama3.1:8b-instruct-q5_K_M
ollama serve
# Optional, expose to a remote frontend via your own ngrok domain:
ngrok http --domain=<your-ollama-domain>.ngrok.app 11434`}</CodeBlock>
      <Callout label="TIP">Use <code>start-everything.bat</code> (Windows) to launch Ollama, the FastAPI backend, and the ngrok tunnels in one click. The script reads your tunnel domains from <code>.env.local</code> so they never get committed to git.</Callout>
    </>
  );
}

function DocPage({ id, onNav }) {
  const idx = DOC_ORDER.indexOf(id);
  const prevId = idx > 0 ? DOC_ORDER[idx - 1] : null;
  const nextId = idx < DOC_ORDER.length - 1 ? DOC_ORDER[idx + 1] : null;

  return (
    <article className="doc page-fade">
      <div className="doc-breadcrumb">
        {docGroup(id)}<span>/</span>{docLabel(id)}
      </div>
      {id === 'introduction'    && <IntroDoc />}
      {id === 'quickstart'      && <QuickstartDoc />}
      {id === 'intents'         && <IntentsDoc />}
      {id === 'welsh'           && <WelshDoc />}
      {id === 'feedback-guide'  && <FeedbackGuideDoc />}
      {id === 'api-chat'        && <ApiChatDoc />}
      {id === 'api-feedback'    && <ApiFeedbackDoc />}
      {id === 'env'             && <EnvDoc />}
      <div className="doc-nav-footer">
        {prevId ? (
          <button className="doc-nav-card" onClick={() => onNav(prevId)}>
            <span className="dir">← Previous</span>
            <span className="nav-label">{docLabel(prevId)}</span>
          </button>
        ) : <div />}
        {nextId ? (
          <button className="doc-nav-card next" onClick={() => onNav(nextId)}>
            <span className="dir">Next →</span>
            <span className="nav-label">{docLabel(nextId)}</span>
          </button>
        ) : <div />}
      </div>
    </article>
  );
}

// Consent modal, shown on first visit
function ConsentModal({ onConsent }) {
  const [checked, setChecked] = useState(false);

  return (
    <div className="consent-overlay">
      <div className="consent-card">
        <div className="consent-logo">
          <div className="consent-brand-mark">U</div>
          <div className="consent-brand-word">U‑Pal <em>assistant</em></div>
        </div>

        <div className="consent-head">
          <h2>Before we begin <span className="consent-cy">/ Cyn dechrau</span></h2>
          <p className="consent-sub">Participant Information, Taflen Gwybodaeth Cyfranogwyr</p>
        </div>

        <ul className="consent-points">
          <li>This is a research prototype built by <strong>Ataur Rahman</strong> as part of a BSc dissertation at the University of Wales Trinity Saint David (UWTSD).</li>
          <li>U-Pal provides bilingual Welsh and English student support using hybrid retrieval (ChromaDB + BM25) with Claude Haiku 4.5 as the LLM.</li>
          <li>Anonymised interaction data, including satisfaction ratings and feedback comments, may be used in academic research.</li>
          <li>No personally identifying information is collected or stored during your session.</li>
          <li>Participation is entirely voluntary, you may stop at any time without consequence.</li>
          <li>This tool is not a substitute for official university support, wellbeing services, or professional advice.</li>
        </ul>

        <label className="consent-check-row">
          <input
            type="checkbox"
            checked={checked}
            onChange={e => setChecked(e.target.checked)}
          />
          <span>
            Rwy'n deall ac yn cytuno i gymryd rhan &mdash; <strong>I understand and agree to participate in this research</strong>
          </span>
        </label>

        <button
          className={`consent-btn${checked ? ' active' : ''}`}
          disabled={!checked}
          onClick={() => {
            if (!checked) return;
            try { sessionStorage.setItem('upal_consented', '1'); } catch (_) {}
            onConsent();
          }}
        >
          Start chatting / Dechrau sgwrsio
        </button>
      </div>
    </div>
  );
}

// Linkify, convert plain-text URLs in bot responses to <a> tags
const URL_RE = /(https?:\/\/[^\s<>"')\]]+[^\s<>"')\].,!?])/g;

function linkify(text) {
  if (!text) return '';
  return text.replace(
    URL_RE,
    '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>',
  );
}

// Reactive status card
/**
 * Derives a CSS modifier from a raw status string.
 *   'connected' | 'ready' | 'active'  → 'ok'
 *   'offline'   | 'error' | http_*    → 'down'
 *   'not_configured'                  → '' (neutral/dim)
 *   null (still fetching)             → 'checking'
 */
function rowClass(val) {
  if (val === null || val === undefined) return 'checking';
  if (val === 'not_configured')          return '';
  if (val === 'connected' || val === 'ready' || val === 'active') return 'ok';
  return 'down';   // offline | timeout | http_XXX | error
}

function rowDetail(val) {
  if (val === null || val === undefined) return '…';
  if (val === 'not_configured') return 'not configured';
  if (val === 'connected')      return 'connected';
  if (val === 'ready')          return 'ready';
  if (val === 'active')         return 'active';
  if (val === 'offline')        return 'offline';
  if (val === 'timeout')        return 'timed out';
  if (val === 'error')          return 'error';
  if (val === 'http_503') return 'server down';
  if (val === 'http_401') return 'auth required';
  if (val === 'http_403') return 'forbidden';
  if (typeof val === 'string' && val.startsWith('http_')) return `HTTP ${val.slice(5)}`;
  return val;
}

// LLM switcher, currently unused on the public demo page (removed so every
// visitor sees the same fixed model).  Kept here commented out in case the
// operator wants to re-enable it for internal demos; the backend endpoint
// /api/llm-config still works and can be POSTed to via curl.
// eslint-disable-next-line no-unused-vars
function _LLMSwitcher({ onChange }) {
  const [cfg,     setCfg]     = useState(null);
  const [saving,  setSaving]  = useState(false);
  const [err,     setErr]     = useState(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const r = await fetch('/api/llm-config');
        const data = await r.json();
        if (!cancelled && r.ok) setCfg(data);
        else if (!cancelled) setErr(data?.error || `HTTP ${r.status}`);
      } catch (e) {
        if (!cancelled) setErr('Backend unreachable');
      }
    })();
    return () => { cancelled = true; };
  }, []);

  async function update(patch) {
    setSaving(true);
    setErr(null);
    try {
      const r = await fetch('/api/llm-config', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(patch),
      });
      const data = await r.json();
      if (!r.ok) {
        setErr(data?.detail || data?.error || `HTTP ${r.status}`);
      } else {
        setCfg(data);
        onChange?.();
      }
    } catch (e) {
      setErr('Backend unreachable');
    }
    setSaving(false);
  }

  if (err && !cfg) {
    return (
      <div className="llm-switcher llm-switcher-err">
        <div className="llm-switcher-head">LLM unavailable</div>
        <div className="llm-switcher-detail">{err}</div>
      </div>
    );
  }
  if (!cfg) {
    return (
      <div className="llm-switcher">
        <div className="llm-switcher-head">LLM loading…</div>
      </div>
    );
  }

  // map each provider to its available model list and the one currently picked
  const modelsByProvider = {
    anthropic: cfg.anthropic_models || [],
    ollama:    cfg.ollama_models    || [],
  };
  const currentModelByProvider = {
    anthropic: cfg.anthropic_model,
    ollama:    cfg.ollama_model,
  };
  const models       = modelsByProvider[cfg.provider] || [];
  const currentModel = currentModelByProvider[cfg.provider] || '';

  return (
    <div className="llm-switcher">
      <div className="llm-switcher-head">
        <span>LLM</span>
        {saving && <span className="llm-switcher-saving">updating…</span>}
      </div>

      <div className="llm-switcher-provider">
        <button
          className={`llm-pill${cfg.provider === 'anthropic' ? ' on' : ''}${!cfg.anthropic_available ? ' disabled' : ''}`}
          disabled={saving || !cfg.anthropic_available}
          onClick={() => update({ provider: 'anthropic' })}
          title={cfg.anthropic_available ? 'Anthropic Claude (cloud)' : 'ANTHROPIC_API_KEY not set'}
        >
          Claude
        </button>
        <button
          className={`llm-pill${cfg.provider === 'ollama' ? ' on' : ''}${!cfg.ollama_available ? ' disabled' : ''}`}
          disabled={saving || !cfg.ollama_available}
          onClick={() => update({ provider: 'ollama' })}
          title={cfg.ollama_available ? 'Local Ollama via tunnel' : 'OLLAMA_URL not set'}
        >
          Ollama
        </button>
      </div>

      <select
        className="llm-switcher-select"
        value={currentModel}
        disabled={saving}
        onChange={(e) => update({ model: e.target.value })}
      >
        {/* Allow unknown current value as a disabled entry so the dropdown still renders */}
        {!models.some(m => m.id === currentModel) && (
          <option value={currentModel}>{currentModel} (custom)</option>
        )}
        {models.map(m => (
          <option key={m.id} value={m.id}>{m.label || m.id}</option>
        ))}
      </select>

      {err && <div className="llm-switcher-err-inline">{err}</div>}
    </div>
  );
}

function StatusCard({ health }) {
  // null means the /api/health request hasn't come back yet, so we show a
  // 'checking' state instead of red dots.
  const checking = health === null;

  // each row maps a raw status string to a css class (ok / down / checking)
  // ref: https://developer.mozilla.org/en-US/docs/Web/API/Element/classList
  const anthropicClass = rowClass(health?.anthropic ?? null);
  const ollamaClass    = rowClass(health?.ollama    ?? null);
  const chromaClass    = rowClass(health?.chroma    ?? null);
  const welshClass     = rowClass(health?.welsh     ?? null);

  // 'anthropic' | 'ollama' | null. used to show the ACTIVE badge on the right row.
  const activeProvider = health?.provider || null;

  return (
    <div className="status-card">
      <div className="status-head">
        <span className="status-head-label">CONNECTION</span>
        <span className="status-head-sub">
          {checking ? 'CHECKING…' : health?.status === 'OK' ? 'LIVE' : 'DEGRADED'}
        </span>
      </div>

      {/* claude (cloud llm, primary) */}
      <div className={`status-row ${anthropicClass}`}>
        <span className="status-dot" />
        <span className="status-label">
          Claude
          {activeProvider === 'anthropic' && <span className="status-badge">ACTIVE</span>}
        </span>
        <span className="status-detail">
          {health?.anthropicModel
            ? health.anthropicModel
            : rowDetail(health?.anthropic ?? null)}
        </span>
      </div>

      {/* ollama (local fallback, only used offline) */}
      <div className={`status-row ${ollamaClass}`}>
        <span className="status-dot" />
        <span className="status-label">
          Ollama
          {activeProvider === 'ollama' && <span className="status-badge">ACTIVE</span>}
        </span>
        <span className="status-detail">
          {health?.ollamaModel
            ? health.ollamaModel
            : rowDetail(health?.ollama ?? null)}
        </span>
      </div>

      {/* Chroma, hybrid RAG vector store */}
      <div className={`status-row ${chromaClass}`}>
        <span className="status-dot" />
        <span className="status-label">Chroma RAG</span>
        <span className="status-detail">
          {typeof health?.chromaDocs === 'number' && health.chromaDocs > 0
            ? `${health.chromaDocs.toLocaleString()} docs`
            : rowDetail(health?.chroma ?? null)}
        </span>
      </div>

      {/* Welsh detection */}
      <div className={`status-row ${welshClass}`}>
        <span className="status-dot" />
        <span className="status-label">Welsh detection</span>
        <span className="status-detail">
          {typeof health?.bilingualTerms === 'number' && health.bilingualTerms > 0
            ? `${health.bilingualTerms.toLocaleString()} terms`
            : rowDetail(health?.welsh ?? null)}
        </span>
      </div>
    </div>
  );
}

// Typing dots
function TypingDots() {
  return (
    <div className="typing">
      <span /><span /><span />
    </div>
  );
}

// Feedback modal
function FeedbackModal({ lang, onClose }) {
  const [stars,      setStars]      = useState(0);
  const [hovered,    setHovered]    = useState(0);
  const [helpful,    setHelpful]    = useState(null);
  const [langOk,     setLangOk]     = useState(null);
  const [comments,   setComments]   = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [done,       setDone]       = useState(false);
  const [closing,    setClosing]    = useState(false);
  const starRowRef = useRef(null);

  const isCy = lang === 'cy';

  const close = useCallback(() => {
    setClosing(true);
    setTimeout(() => onClose(), 240);
  }, [onClose]);

  useEffect(() => {
    const handler = (e) => { if (e.key === 'Escape') close(); };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [close]);

  async function submit() {
    if (!stars) {
      const row = starRowRef.current;
      if (row) {
        row.classList.remove('shake');
        void row.offsetWidth;
        row.classList.add('shake');
        setTimeout(() => row.classList.remove('shake'), 400);
      }
      return;
    }
    setSubmitting(true);
    try {
      await fetch('/api/feedback', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ satisfaction: stars, helpfulAnswer: helpful, correctLanguage: langOk, comments }),
      });
    } catch (_) {}
    setDone(true);
    setSubmitting(false);
    window.dispatchEvent(new CustomEvent('upal:toast', {
      detail: { msg: isCy ? 'Diolch am eich adborth!' : 'Thanks for your feedback!' },
    }));
    setTimeout(() => close(), 1800);
  }

  return (
    <div
      className={`modal-overlay${closing ? ' closing' : ''}`}
      onMouseDown={(e) => { if (e.target === e.currentTarget) close(); }}
    >
      <div className={`modal${closing ? ' closing' : ''}`} onMouseDown={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <div className="modal-title">
            {isCy ? 'Sut oedd eich profiad?' : 'How was your experience?'}
          </div>
          <div className="modal-sub">
            {isCy ? 'Mae eich adborth yn ein helpu i wella.' : 'Your feedback helps us improve.'}
          </div>
        </div>

        {done ? (
          <div className="modal-done">
            {isCy ? 'Diolch yn fawr ✓' : 'Thank you ✓'}
          </div>
        ) : (
          <div className="modal-body">
            <div>
              <div className="modal-field-label">{isCy ? 'Sgôr cyffredinol' : 'Overall rating'}</div>
              <div className="modal-stars" ref={starRowRef}>
                {[1, 2, 3, 4, 5].map(n => (
                  <button
                    key={n}
                    className={`star-btn${(hovered || stars) >= n ? ' on' : ''}`}
                    onMouseEnter={() => setHovered(n)}
                    onMouseLeave={() => setHovered(0)}
                    onClick={() => setStars(n)}
                  >★</button>
                ))}
              </div>
            </div>

            <div>
              <div className="modal-field-label">{isCy ? 'Oedd yr ateb yn ddefnyddiol?' : 'Was the answer helpful?'}</div>
              <div className="modal-toggles">
                <button className={`modal-toggle-btn${helpful === true ? ' on' : ''}`} onClick={() => setHelpful(helpful === true ? null : true)}>{isCy ? 'Oedd' : 'Yes'}</button>
                <button className={`modal-toggle-btn${helpful === false ? ' on' : ''}`} onClick={() => setHelpful(helpful === false ? null : false)}>{isCy ? 'Nac oedd' : 'No'}</button>
              </div>
            </div>

            <div>
              <div className="modal-field-label">{isCy ? 'Iaith gywir?' : 'Correct language detected?'}</div>
              <div className="modal-toggles">
                <button className={`modal-toggle-btn${langOk === true ? ' on' : ''}`} onClick={() => setLangOk(langOk === true ? null : true)}>{isCy ? 'Oedd' : 'Yes'}</button>
                <button className={`modal-toggle-btn${langOk === false ? ' on' : ''}`} onClick={() => setLangOk(langOk === false ? null : false)}>{isCy ? 'Nac oedd' : 'No'}</button>
              </div>
            </div>

            <div>
              <div className="modal-field-label">{isCy ? 'Sylwadau (dewisol)' : 'Comments (optional)'}</div>
              <textarea
                className="modal-textarea"
                rows={3}
                value={comments}
                onChange={(e) => setComments(e.target.value)}
                placeholder={isCy ? 'Ychwanegu sylwadau…' : 'Add any comments…'}
              />
            </div>

            <div className="modal-footer">
              <button className="modal-cancel-btn" onClick={close}>{isCy ? 'Canslo' : 'Cancel'}</button>
              <button className="modal-submit-btn" onClick={submit} disabled={submitting}>
                {submitting ? '…' : (isCy ? 'Cyflwyno' : 'Submit')}
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// Main app
export default function Home() {
  // Consent, sessionStorage so every new tab/session shows the modal
  // (localStorage was persisting consent across all future visits, wrong for a study)
  const [consented, setConsented] = useState(false);
  useEffect(() => {
    try {
      if (sessionStorage.getItem('upal_consented') === '1') setConsented(true);
      // else: leave as false so the modal shows
    } catch (_) {
      // sessionStorage unavailable (e.g. private browsing restrictions), show the modal
    }
  }, []);

  // Theming
  const [theme,      setTheme]      = useState('light');
  const [font,       setFont]       = useState('editorial');
  const [accent,     setAccent]     = useState(28);
  const [tweaksOpen, setTweaksOpen] = useState(false);

  // Docs search
  const [docSearch, setDocSearch] = useState('');

  // Navigation
  const [tab,           setTab]           = useState('demo');
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const [docPage,       setDocPage]       = useState('introduction');

  // Chat
  const [msgs,           setMsgs]           = useState([]);
  const [text,           setText]           = useState('');
  const [busy,           setBusy]           = useState(false);
  const [lang,           setLang]           = useState('en');
  const [chipsDismissed, setChipsDismissed] = useState(false);

  // UI
  const [fbOpen,        setFbOpen]        = useState(false);
  const [toasts,        setToasts]        = useState([]);
  const [healthStatus,  setHealthStatus]  = useState(null); // null = checking

  const msgsEndRef    = useRef(null);
  const inputRef      = useRef(null);
  const msgCountRef   = useRef(0);   // tracks last count so translate doesn't trigger scroll

  /* Live health / connectivity check, runs on mount and every 60 s */
  useEffect(() => {
    async function checkHealth() {
      try {
        const res  = await fetch('/api/health');
        const data = await res.json();
        setHealthStatus(data);
      } catch {
        setHealthStatus({ status: 'error', ollama: 'offline', corenlp: 'offline', nlp: 'error', welsh: 'unknown', intents: 0 });
      }
    }
    checkHealth();
    const id = setInterval(checkHealth, 60_000);
    return () => clearInterval(id);
  }, []);

  /* One-shot restore of theme / font / accent from localStorage on mount.
     the same keys are used on the /admin page so the two stay in sync.
     ref: https://developer.mozilla.org/en-US/docs/Web/API/Window/localStorage */
  useEffect(() => {
    if (typeof window === 'undefined') return;
    try {
      const savedTheme  = localStorage.getItem('upal.theme');
      const savedFont   = localStorage.getItem('upal.font');
      const savedAccent = localStorage.getItem('upal.accent');
      if (savedTheme === 'light' || savedTheme === 'dark') setTheme(savedTheme);
      if (savedFont)  setFont(savedFont);
      if (savedAccent && !Number.isNaN(Number(savedAccent))) setAccent(Number(savedAccent));
    } catch (_) { /* Safari private mode throws on localStorage access */ }
  }, []);

  /* Apply theme / font / accent to <html> and persist them so the next
     visit restores the same look. */
  useEffect(() => {
    if (typeof window === 'undefined') return;
    document.documentElement.dataset.theme = theme;
    document.documentElement.dataset.font  = font;
    document.documentElement.style.setProperty('--accent-h', String(accent));
    try {
      localStorage.setItem('upal.theme',  theme);
      localStorage.setItem('upal.font',   font);
      localStorage.setItem('upal.accent', String(accent));
    } catch (_) { /* ignore, theme still applies for this session */ }
  }, [theme, font, accent]);

  /* Toast listener */
  useEffect(() => {
    const handler = (e) => {
      const id = Date.now();
      setToasts(t => [...t, { id, msg: e.detail.msg }]);
      setTimeout(() => setToasts(t => t.filter(x => x.id !== id)), 4000);
    };
    window.addEventListener('upal:toast', handler);
    return () => window.removeEventListener('upal:toast', handler);
  }, []);

  /* Scroll to bottom ONLY when a new message is added, not when translate toggles */
  useEffect(() => {
    if (msgs.length > msgCountRef.current) {
      msgsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
    msgCountRef.current = msgs.length;
  }, [msgs]);

  function switchTab(id) {
    setTab(id);
    setMobileNavOpen(false);
  }

  function addToast(msg) {
    const id = Date.now();
    setToasts(t => [...t, { id, msg }]);
    setTimeout(() => setToasts(t => t.filter(x => x.id !== id)), 4000);
  }

  async function dispatchQuery(query) {
    const q = (query !== undefined ? query : text).trim();
    if (!q || busy) return;
    if (q.length > 1000) { addToast('Message too long (max 1000 chars)'); return; }

    setChipsDismissed(true);
    const now = new Date().toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' });
    // Snapshot history BEFORE we push the new user turn so the backend sees
    // only prior exchanges (the new message arrives as `message`).  Send last
    // 6 turns, enough for topic continuity without overloading small-model
    // context windows.  Use `altText` fallback so Welsh turns carry English
    // equivalents where available (helps downstream NLU).
    const historyForAPI = msgs.slice(-6).map(m => ({
      role: m.role === 'user' ? 'user' : 'assistant',
      text: m.text || m.altText || '',
    })).filter(t => t.text);
    setMsgs(m => [...m, { id: Date.now(), role: 'user', text: q, time: now }]);
    setText('');
    setBusy(true);

    try {
      const res  = await fetch('/api/chat', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({
          message:     q,
          runningLang: lang,
          history:     historyForAPI,
        }),
      });
      const data = await res.json();
      const botLang = data.lang || 'en';
      setLang(botLang);
      const botNow = new Date().toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' });
      setMsgs(m => [...m, {
        id:       Date.now(),
        role:     'bot',
        text:     data.response,
        altText:  data.altResponse,
        tag:      data.tag,
        lang:     botLang,
        time:     botNow,
        showAlt:  false,
      }]);
    } catch {
      const errNow = new Date().toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' });
      setMsgs(m => [...m, {
        id:   Date.now(),
        role: 'bot',
        text: 'Sorry, something went wrong. Please try again.',
        time: errNow,
        lang: 'en',
      }]);
    }
    setBusy(false);
  }

  async function toggleAlt(id) {
    const msg = msgs.find(m => m.id === id);
    if (!msg) return;

    // If already toggled on, just toggle back off, no fetch needed
    if (msg.showAlt) {
      setMsgs(m => m.map(x => x.id === id ? { ...x, showAlt: false } : x));
      return;
    }

    // If altText is already loaded, just show it
    if (msg.altText) {
      setMsgs(m => m.map(x => x.id === id ? { ...x, showAlt: true } : x));
      return;
    }

    // Otherwise fetch translation lazily, show loading state first
    setMsgs(m => m.map(x => x.id === id ? { ...x, translating: true } : x));

    const to_lang = msg.lang === 'cy' ? 'en' : 'cy';
    try {
      const r = await fetch('/api/translate', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ text: msg.text, from_lang: msg.lang, to_lang }),
      });
      if (r.ok) {
        const data = await r.json();
        setMsgs(m => m.map(x =>
          x.id === id
            ? { ...x, altText: data.translation, showAlt: true, translating: false }
            : x
        ));
      } else {
        setMsgs(m => m.map(x => x.id === id ? { ...x, translating: false } : x));
        addToast('Translation unavailable, try again shortly');
      }
    } catch {
      setMsgs(m => m.map(x => x.id === id ? { ...x, translating: false } : x));
      addToast('Translation unavailable, try again shortly');
    }
  }

  function handleKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); dispatchQuery(); }
  }

  const isCy = lang === 'cy';

  return (
    <>
      <Head>
        <title>U-Pal: UWTSD Student Assistant</title>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>

      <div className="app">

        {/* Consent modal */}
        {!consented && <ConsentModal onConsent={() => setConsented(true)} />}

        {/* Topbar */}
        <header className="topbar">
          <div className="topbar-inner">

            {/* Brand mark */}
            <div className="brand">
              <div className="brand-mark">
                U
                <span className="brand-mark-bar" />
                <span className="brand-mark-dot" />
              </div>
              <div className="brand-wordmark">U‑Pal <em>assistant</em></div>
            </div>

            {/* Tabs */}
            <div className="topbar-tabs">
              {TABS.map(t => (
                <button
                  key={t.id}
                  className={`tab${tab === t.id ? ' active' : ''}`}
                  onClick={() => switchTab(t.id)}
                >
                  <span className="tab-num">{t.num}</span>
                  {t.label}
                </button>
              ))}
            </div>

            <div className="topbar-spacer" />

            {/* Actions */}
            <div className="topbar-actions">
              <div className="lang-pill">
                <span className="dot" />
                <span className="lit">{isCy ? 'cy' : 'en'}</span>
                <span>{isCy ? 'Cymraeg' : 'English'}</span>
              </div>
              <a
                className="topbar-btn secondary"
                href={FORMS_URL}
                target="_blank"
                rel="noopener noreferrer"
              >
                Detailed feedback
              </a>
              <button className="topbar-btn primary" onClick={() => setFbOpen(true)}>
                Rate U-Pal
              </button>
              {/* Themes button, shown only on mobile in topbar */}
              <button
                className="topbar-themes-btn"
                onClick={() => setTweaksOpen(v => !v)}
                title="Themes"
              >
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" width="18" height="18">
                  <line x1="4" y1="6" x2="20" y2="6"/><circle cx="8" cy="6" r="2" fill="currentColor" stroke="none"/>
                  <line x1="4" y1="12" x2="20" y2="12"/><circle cx="16" cy="12" r="2" fill="currentColor" stroke="none"/>
                  <line x1="4" y1="18" x2="20" y2="18"/><circle cx="10" cy="18" r="2" fill="currentColor" stroke="none"/>
                </svg>
              </button>
              <button
                className={`hamburger-btn${mobileNavOpen ? ' open' : ''}`}
                onClick={() => setMobileNavOpen(v => !v)}
                aria-label="Open navigation"
              >
                <span className="bar" />
                <span className="bar" />
                <span className="bar" />
              </button>
            </div>
          </div>
        </header>

        {/* Mobile nav */}
        {mobileNavOpen && (
          <>
            <div className="mobile-nav-backdrop" onClick={() => setMobileNavOpen(false)} />
            <nav className="mobile-nav">
              <div className="mobile-nav-header">
                <span className="mobile-nav-title">Navigation</span>
                <button className="mobile-nav-close" onClick={() => setMobileNavOpen(false)}>
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" width="18" height="18">
                    <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
                  </svg>
                </button>
              </div>
              {TABS.map(t => (
                <button
                  key={t.id}
                  className={`mobile-nav-item${tab === t.id ? ' active' : ''}`}
                  onClick={() => switchTab(t.id)}
                >
                  <span className="mobile-nav-num">{t.num}</span>
                  {t.label}
                </button>
              ))}
              <div className="mobile-nav-divider" />
              <div className="mobile-nav-section-label">Feedback</div>
              <a
                className="mobile-nav-item mobile-nav-survey"
                href={FORMS_URL}
                target="_blank"
                rel="noopener noreferrer"
                onClick={() => setMobileNavOpen(false)}
              >
                <span className="mobile-nav-num">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" width="14" height="14">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/>
                  </svg>
                </span>
                Take our survey
              </a>
            </nav>
          </>
        )}

        {/* Main */}
        <main className="main">
          <div className="page-shell">

            {/* DEMO tab */}
            {tab === 'demo' && (
              <div className="demo page-fade">

                {/* Hero */}
                <div className="demo-hero">
                  <div className="demo-hero-inner">
                    <div className="demo-hero-text">
                      <div className="eyebrow">Live demo · v0.4.2</div>
                      <h1 className="demo-title">Ask <em>anything</em> about UWTSD</h1>
                      <p className="demo-sub">
                        A bilingual student assistant built for UWTSD, supporting academic needs in Welsh and English. Powered by Claude Haiku 4.5, hybrid retrieval, and a Welsh-aware detector.
                      </p>
                    </div>
                    <div className="demo-meta">
                      <StatusCard health={healthStatus} />
                    </div>
                  </div>
                </div>

                {/* Chat body */}
                <div className="demo-body">
                  <div className="messages">
                    <div className="messages-inner">
                      {msgs.map(msg => (
                        <div key={msg.id} className={`msg ${msg.role}`}>
                          <div className="msg-gutter">
                            <span className="msg-gutter-label">
                              {msg.role === 'user' ? 'You' : 'U-Pal'}
                            </span>
                            <span className="msg-gutter-time">{msg.time}</span>
                          </div>
                          <div className="msg-body">
                            {msg.role === 'bot' && msg.lang && (
                              <div className="msg-lang">
                                <span className="dot" />
                                {msg.lang === 'cy' ? 'Cymraeg' : 'English'}
                              </div>
                            )}
                            <div
                              className="msg-content"
                              dangerouslySetInnerHTML={{ __html: linkify(msg.showAlt ? msg.altText : msg.text) }}
                            />
                            {msg.role === 'bot' && msg.text && (
                              <div className="msg-actions">
                                <button
                                  className={`translate-btn${msg.translating ? ' translating' : ''}`}
                                  onClick={() => toggleAlt(msg.id)}
                                  disabled={!!msg.translating}
                                >
                                  {msg.translating
                                    ? '…'
                                    : msg.showAlt
                                      ? (msg.lang === 'cy' ? '← Cymraeg' : '← English')
                                      : (msg.lang === 'cy' ? 'Show English →' : 'Dangos Cymraeg →')}
                                </button>
                              </div>
                            )}
                          </div>
                        </div>
                      ))}

                      {/* Typing indicator */}
                      {busy && (
                        <div className="msg bot">
                          <div className="msg-gutter">
                            <span className="msg-gutter-label">U-Pal</span>
                          </div>
                          <div className="msg-body">
                            <TypingDots />
                          </div>
                        </div>
                      )}

                      {/* Quick-start chips */}
                      {!chipsDismissed && msgs.length === 0 && (
                        <div className="chips-block">
                          <div className="chips-label">Try asking</div>
                          <div className="chips">
                            {CHIPS.map(c => (
                              <button
                                key={c.num}
                                className="chip"
                                onClick={() => dispatchQuery(c.text)}
                              >
                                <span className="chip-num">{c.num}</span>
                                {c.text}
                                <span className="chip-arrow">→</span>
                              </button>
                            ))}
                          </div>
                        </div>
                      )}

                      <div ref={msgsEndRef} />
                    </div>
                  </div>

                  {/* Input */}
                  <div className="input-dock">
                    <div className="input-dock-inner">
                      <div className="input-row">
                        <input
                          ref={inputRef}
                          type="text"
                          value={text}
                          onChange={(e) => setText(e.target.value)}
                          onKeyDown={handleKey}
                          placeholder={isCy ? 'Gofynnwch gwestiwn…' : 'Ask a question about UWTSD…'}
                          disabled={busy}
                          maxLength={1000}
                          autoComplete="off"
                        />
                        <button
                          className="send-btn"
                          onClick={() => dispatchQuery()}
                          disabled={busy || !text.trim()}
                        >
                          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <line x1="22" y1="2" x2="11" y2="13" />
                            <polygon points="22 2 15 22 11 13 2 9 22 2" />
                          </svg>
                          <span className="send-label">Send</span>
                        </button>
                      </div>
                      <div className="input-footnote">
                        <span>© U-Pal 2026 · All rights reserved</span>
                        <kbd>↵ Enter</kbd>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* HOW IT WORKS tab */}
            {tab === 'how' && (
              <div className="how page-fade">
                <div className="how-hero">
                  <div className="eyebrow">Architecture</div>
                  <h1 className="how-title">How U-Pal <em>thinks</em></h1>
                  <p className="how-lede">
                    Five stages from raw text to bilingual response: detect the
                    language, expand the query, retrieve grounded passages,
                    generate a reply with Claude, and clean up the output.
                  </p>
                </div>

                <div className="steps">
                  {STEPS.map((step) => (
                    <div key={step.n} className="step">
                      <div className="step-num">
                        <span className="n">{step.n}</span>
                        <span className="total">{step.total}</span>
                      </div>
                      <div className="step-body">
                        <h3>{step.title}</h3>
                        <p>{step.desc}</p>
                        <div className="step-tags">
                          {step.tags.map((tag, i) => (
                            <span key={i} className={`step-tag${i === 0 ? ' accent' : ''}`}>{tag}</span>
                          ))}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>

                <div className="features-section">
                  <div className="section-label">
                    <span className="label">FEATURES</span>
                    <h2>What makes U-Pal <em>different</em></h2>
                  </div>
                  <div className="cards-grid">
                    {FEATURES.map((feat, i) => (
                      <div key={i} className="feat">
                        <div className="feat-icon">
                          <FeatIcon k={feat.iconKey} />
                        </div>
                        <h4>{feat.title}</h4>
                        <p>{feat.desc}</p>
                        <div className="feat-meta">{feat.meta}</div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* DOCS tab */}
            {tab === 'docs' && (
              <div className="docs page-fade">

                {/* Mobile-only doc nav, horizontal pill strip shown when sidebar is hidden */}
                <div className="doc-mobile-nav">
                  {DOC_ORDER.map(id => (
                    <button
                      key={id}
                      className={`doc-mobile-pill${docPage === id ? ' active' : ''}`}
                      onClick={() => setDocPage(id)}
                    >
                      {docLabel(id)}
                    </button>
                  ))}
                </div>

                <aside className="doc-sidebar">
                  <div className="doc-search-wrap">
                    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <circle cx="11" cy="11" r="8" />
                      <line x1="21" y1="21" x2="16.65" y2="16.65" />
                    </svg>
                    <input
                      className="doc-search-input"
                      type="text"
                      placeholder="Search docs…"
                      value={docSearch}
                      onChange={e => setDocSearch(e.target.value)}
                    />
                    {docSearch && (
                      <button className="doc-search-clear" onClick={() => setDocSearch('')}>×</button>
                    )}
                  </div>
                  {(() => {
                    const q = docSearch.trim().toLowerCase();
                    const filtered = DOC_GROUPS.map(g => ({
                      ...g,
                      items: g.items.filter(item =>
                        !q ||
                        item.label.toLowerCase().includes(q) ||
                        g.title.toLowerCase().includes(q)
                      ),
                    })).filter(g => g.items.length > 0);
                    if (q && filtered.length === 0) {
                      return <div className="doc-no-results">No results for "{docSearch}"</div>;
                    }
                    return filtered.map(g => (
                      <div key={g.title} className="doc-group">
                        <div className="doc-group-title">{g.title}</div>
                        {g.items.map(item => (
                          <button
                            key={item.id}
                            className={`doc-link${docPage === item.id ? ' active' : ''}`}
                            onClick={() => { setDocPage(item.id); setDocSearch(''); }}
                          >
                            <span className="dot" />
                            {item.label}
                            {item.pill && <span className="pill">{item.pill}</span>}
                          </button>
                        ))}
                      </div>
                    ));
                  })()}
                </aside>
                <div className="doc-main">
                  <div className="doc-inner">
                    <DocPage key={docPage} id={docPage} onNav={setDocPage} />
                  </div>
                </div>
              </div>
            )}

          </div>
        </main>

        {/* Themes panel */}
        {tweaksOpen ? (
          <div className="tweaks-panel">
            <div className="tweaks-head">
              <span className="dot" />
              Themes
              <button className="close-btn" onClick={() => setTweaksOpen(false)} aria-label="Close themes">×</button>
            </div>
            <div className="tweaks-body">
              <div className="tweak-row">
                <div className="tweak-label">Accent colour</div>
                <div className="tweak-swatches">
                  {ACCENTS.map(a => (
                    <button
                      key={a.hue}
                      className={`tweak-sw${accent === a.hue ? ' on' : ''}`}
                      title={a.name}
                      style={{ background: `oklch(0.62 0.15 ${a.hue})` }}
                      onClick={() => setAccent(a.hue)}
                    />
                  ))}
                </div>
              </div>
              <div className="tweak-row">
                <div className="tweak-label">Font pairing</div>
                <div className="tweak-fonts">
                  {['editorial', 'grotesk', 'plex'].map(f => (
                    <button
                      key={f}
                      className={`tweak-btn${font === f ? ' on' : ''}`}
                      onClick={() => setFont(f)}
                    >
                      {f === 'editorial' ? 'Editorial' : f === 'grotesk' ? 'Grotesk' : 'Plex'}
                    </button>
                  ))}
                </div>
              </div>
              <div className="tweak-row">
                <div className="tweak-label">Theme</div>
                <div className="tweak-toggle">
                  {['light', 'dark'].map(t => (
                    <button
                      key={t}
                      className={`tweak-btn${theme === t ? ' on' : ''}`}
                      onClick={() => setTheme(t)}
                    >
                      {t === 'light' ? 'Light' : 'Dark'}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>
        ) : (
          <button className="tweaks-fab" onClick={() => setTweaksOpen(true)} title="Open themes">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" width="18" height="18">
              <line x1="4" y1="6" x2="20" y2="6"/><circle cx="8" cy="6" r="2" fill="currentColor" stroke="none"/>
              <line x1="4" y1="12" x2="20" y2="12"/><circle cx="16" cy="12" r="2" fill="currentColor" stroke="none"/>
              <line x1="4" y1="18" x2="20" y2="18"/><circle cx="10" cy="18" r="2" fill="currentColor" stroke="none"/>
            </svg>
          </button>
        )}

        {/* Toast stack */}
        <div className="toast-stack" aria-live="polite">
          {toasts.map(t => (
            <div key={t.id} className="toast">{t.msg}</div>
          ))}
        </div>

        {/* Feedback modal */}
        {fbOpen && <FeedbackModal lang={lang} onClose={() => setFbOpen(false)} />}

      </div>
    </>
  );
}
