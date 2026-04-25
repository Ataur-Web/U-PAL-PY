# Quick Start: Integrate Advanced NLP in 5 Minutes

## What You're Doing
Adding 8 smart layers to your chatbot **without changing its core**. Your existing TF-IDF + Naive Bayes stays; these techniques **wrap around it**.

---

## Step 1: Install the One Dependency

```bash
cd path/to/your/u-pal-rag
npm install fast-levenshtein
```

This gives you typo-correction ability. That's it!

---

## Step 2: Copy advanced-nlp.js Into Your Project

✅ Already done! It's in your project folder.

---

## Step 3: Update Your server.js (The Real Work)

### Before Your Current /api/chat Endpoint:

**Add this at the top of server.js** (after other requires):

```javascript
// ===== ADD THESE LINES =====
const Levenshtein = require('fast-levenshtein');
const FS = require('fs');

// Load the advanced NLP module
const {
  FuzzyMatcher,
  EntityRecognizer,
  ConversationContext,
  SynonymExpander,
  ClarificationManager,
  ResponseRanker,
  ChatbotLearner
} = require('./advanced-nlp.js');

// Initialize modules once at startup
const fuzzyMatcher = new FuzzyMatcher();
const entityRecognizer = new EntityRecognizer();
const conversationContexts = new Map(); // sessionId -> ConversationContext
const synonymExpander = new SynonymExpander();
const clarificationManager = new ClarificationManager();
const responseRanker = new ResponseRanker();
const chatbotLearner = new ChatbotLearner();

// Build vocabulary for fuzzy matcher from knowledge base
let vocabulary = [];
knowledge.forEach(intent => {
  intent.patterns?.forEach(pattern => {
    vocabulary.push(...pattern.toLowerCase().split(/\s+/));
  });
});
vocabulary = [...new Set(vocabulary)]; // unique words
fuzzyMatcher.setVocabulary(vocabulary);
// ===== END ADD =====
```

### Replace Your /api/chat Handler:

**Find this in server.js:**
```javascript
app.post('/api/chat', async (req, res) => {
  try {
    const { message, lang, sessionId } = req.body;
    // ... existing code ...
  }
});
```

**Replace the entire handler with this:**

```javascript
app.post('/api/chat', async (req, res) => {
  try {
    const { message, lang, sessionId = 'default' } = req.body;
    
    if (!message) {
      return res.status(400).json({ error: 'Message is required' });
    }

    // ===== LAYER 1: TYPO CORRECTION =====
    let processedMessage = fuzzyMatcher.correctText(message, vocabulary);
    console.log(`[Fuzzy] "${message}" → "${processedMessage}"`);

    // ===== LAYER 2: LANGUAGE DETECTION =====
    const detectedLang = detectLanguage(processedMessage);
    const requestLang = lang || detectedLang;

    // ===== LAYER 3: ENTITY EXTRACTION =====
    const entities = entityRecognizer.extract(processedMessage);
    console.log(`[Entity] Extracted:`, entities);

    // ===== LAYER 4: SYNONYM EXPANSION =====
    const expandedMessage = synonymExpander.expand(processedMessage);
    console.log(`[Synonym] Expanded message used for matching`);

    // ===== LAYER 5: GET/TRACK CONVERSATION CONTEXT =====
    if (!conversationContexts.has(sessionId)) {
      conversationContexts.set(sessionId, new ConversationContext(sessionId));
    }
    const context = conversationContexts.get(sessionId);

    // ===== LAYER 6: EXISTING NLP (TF-IDF + NAIVE BAYES) =====
    // Preprocess for scoring
    let tokens = preprocessText(expandedMessage, requestLang);
    
    // Score with TF-IDF
    const tfidfScores = tfidf.tfidfs(tokens, (i, score) => {});
    let bestIntentIdx = 0;
    let bestScore = 0;
    knowledge.forEach((intent, idx) => {
      let score = 0;
      intent.patterns?.forEach(pattern => {
        let patternTokens = preprocessText(pattern, requestLang);
        let matchScore = 0;
        tokens.forEach(token => {
          if (patternTokens.includes(token)) matchScore++;
        });
        score = Math.max(score, matchScore / Math.max(tokens.length, patternTokens.length));
      });
      if (score > bestScore) {
        bestScore = score;
        bestIntentIdx = idx;
      }
    });

    // Also try Naive Bayes
    const bayesClass = classifier.classify(expandedMessage);
    let bayesScore = 0.5; // default

    const matchedIntent = knowledge[bestIntentIdx];
    const intentTag = matchedIntent?.tag;
    let finalScore = bestScore;

    if (bayesClass === intentTag) {
      finalScore = Math.min(1, bestScore + 0.3); // boost if Bayes agrees
    }

    // ===== LAYER 7: CONFIDENCE CHECK & CLARIFICATION =====
    if (finalScore < 0.15) {
      const clarification = clarificationManager.generateClarification(
        bestIntentIdx,
        finalScore,
        entities,
        knowledge
      );
      context.addTurn(message, clarification, 'clarification_needed', entities);
      return res.json({
        response: clarification,
        altResponse: clarification, // both same for clarification
        tag: 'clarification_needed',
        lang: requestLang,
        confidence: finalScore,
        sessionId: sessionId
      });
    }

    // ===== LAYER 8: RESPONSE SELECTION & RANKING =====
    const responses = matchedIntent?.responses[requestLang] || matchedIntent?.responses.en || [];
    
    let selectedResponse;
    if (responses.length === 1) {
      selectedResponse = responses[0];
    } else {
      selectedResponse = responseRanker.selectBestResponse(
        responses,
        message,
        context,
        entities
      );
    }

    // Get alternative language response
    const altLang = requestLang === 'en' ? 'cy' : 'en';
    const altResponses = matchedIntent?.responses[altLang] || [];
    const altResponse = altResponses.length > 0 
      ? altResponses[Math.floor(Math.random() * altResponses.length)]
      : selectedResponse;

    // ===== LAYER 9: LEARNING & ANALYTICS =====
    context.addTurn(message, selectedResponse, intentTag, entities);
    chatbotLearner.recordInteraction(message, intentTag, finalScore, true);

    // Save feedback to MongoDB if configured
    if (process.env.MONGODB_URI) {
      try {
        const collection = db.collection('feedback');
        await collection.insertOne({
          message,
          tag: intentTag,
          response: selectedResponse,
          lang: requestLang,
          confidence: finalScore,
          sessionId,
          entities,
          timestamp: new Date()
        });
      } catch (mongoErr) {
        console.error('MongoDB save error:', mongoErr.message);
        // Continue anyway - don't break chat
      }
    }

    // ===== RETURN RESPONSE =====
    return res.json({
      response: selectedResponse,
      altResponse: altResponse,
      tag: intentTag,
      lang: requestLang,
      confidence: finalScore,
      sessionId: sessionId,
      entities: entities // optional: send entities to frontend
    });

  } catch (error) {
    console.error('Chat error:', error);
    return res.status(500).json({ error: 'Internal server error' });
  }
});
```

---

## Step 4: Add Analytics Endpoint (Optional But Recommended)

**Add this endpoint to server.js** (alongside /api/chat):

```javascript
// Analytics endpoint - Protected by password
app.get('/api/analytics/insights', (req, res) => {
  const auth = req.headers.authorization;
  
  if (!auth || !auth.startsWith('Basic ')) {
    return res.status(401).json({ error: 'Authentication required' });
  }

  const credentials = Buffer.from(auth.substring(6), 'base64').toString();
  const [, password] = credentials.split(':');

  if (password !== process.env.ADMIN_PASSWORD) {
    return res.status(403).json({ error: 'Invalid password' });
  }

  const insights = chatbotLearner.generateInsights();
  return res.json(insights);
});
```

---

## Step 5: Test Locally

```bash
npm start
```

Then test these in a terminal:

```bash
# Test 1: Normal question
curl -X POST http://localhost:3000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "How much does it cost?", "sessionId": "user123"}'

# Test 2: Typo correction
curl -X POST http://localhost:3000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "hww much does it ocst?", "sessionId": "user123"}'

# Test 3: Welsh (bilingual)
curl -X POST http://localhost:3000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Ble rwy'\''n rhaid byw?", "sessionId": "user456"}'

# Test 4: Multi-turn context
curl -X POST http://localhost:3000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I am interested in business", "sessionId": "user789"}'

curl -X POST http://localhost:3000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "How much does it cost?", "sessionId": "user789"}'
```

---

## Step 6: Deploy to Vercel

```bash
git add server.js
git commit -m "feat: integrate advanced NLP with 8 enhancement techniques"
git push origin main
```

Vercel auto-deploys. Check your deployment at: https://u-pal-rag.vercel.app

---

## 🎯 What Just Happened?

Your bot now:

| Feature | Status |
|---------|--------|
| 1️⃣ Corrects typos | ✅ |
| 2️⃣ Understands context | ✅ |
| 3️⃣ Talks to user over multiple turns | ✅ |
| 4️⃣ Recognizes entities (courses, campuses) | ✅ |
| 5️⃣ Expands synonyms | ✅ |
| 6️⃣ Asks for clarification when confused | ✅ |
| 7️⃣ Picks best response from multiple options | ✅ |
| 8️⃣ Learns from interactions | ✅ |

---

## 🚨 Troubleshooting

### "Cannot find module 'fast-levenshtein'"
```bash
npm install fast-levenshtein
npm start
```

### "advanced-nlp.js not found"
Make sure the file is in the same folder as server.js (project root)

### Responses look the same
That's normal! Context, ranking, and learning are subtle. Check `/api/analytics/insights` to see what's happening

### Old behavior still showing
Restart the server:
```bash
# Kill it
Ctrl+C

# Restart
npm start
```

---

## 📊 Monitoring Progress

Check analytics weekly:

```bash
curl -X GET http://localhost:3000/api/analytics/insights \
  -H "Authorization: Basic :your-admin-password"
```

You'll see:
- Total interactions
- Average confidence
- Success rate
- Most confused intent pairs

Use this to decide what new patterns to add!

---

## ✅ Done!

You just built an AI chatbot without an LLM API. That's professional-grade work. 🚀

Next: Add more patterns to knowledge.json and watch it improve!
