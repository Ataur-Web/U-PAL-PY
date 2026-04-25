/**
 * HOW TO INTEGRATE ADVANCED NLP INTO U-PAL
 * 
 * This file shows exactly where and how to add the advanced features
 * to your existing server.js without breaking anything.
 */

// ═══════════════════════════════════════════════════════════════════════════
// STEP 1: INSTALL DEPENDENCY (one-time)
// ═══════════════════════════════════════════════════════════════════════════
/*
  npm install fast-levenshtein
  
  This gives us fuzzy matching for typo correction.
*/

// ═══════════════════════════════════════════════════════════════════════════
// STEP 2: UPDATE IMPORTS IN server.js (at the top)
// ═══════════════════════════════════════════════════════════════════════════
/*
  const {
    FuzzyMatcher,
    EntityRecognizer,
    ConversationContext,
    SynonymExpander,
    ClarificationManager,
    ResponseRanker,
    ChatbotLearner
  } = require('./advanced-nlp');
*/

// ═══════════════════════════════════════════════════════════════════════════
// STEP 3: INITIALIZE IN server.js (after knowledge base loads)
// ═══════════════════════════════════════════════════════════════════════════
/*
  // Build vocabulary from knowledge base patterns
  const allWords = new Set();
  knowledge.forEach(intent => {
    intent.patterns.forEach(pattern => {
      pattern.split(/\s+/).forEach(word => allWords.add(word));
    });
  });
  const vocabulary = Array.from(allWords);

  // Initialize advanced NLP tools
  const fuzzyMatcher = new FuzzyMatcher(vocabulary, 0.85);
  const entityRecognizer = new EntityRecognizer();
  const synonymExpander = new SynonymExpander();
  const clarificationManager = new ClarificationManager();
  const responseRanker = new ResponseRanker();
  const chatbotLearner = new ChatbotLearner();

  // In-memory conversation contexts (in production, use Redis or MongoDB)
  const conversationContexts = new Map();
*/

// ═══════════════════════════════════════════════════════════════════════════
// STEP 4: UPDATE THE /api/chat ENDPOINT
// ═══════════════════════════════════════════════════════════════════════════
/*
  app.post('/api/chat', async (req, res) => {
    try {
      let message = req.body.message?.trim();
      if (!message) return res.json({ response: 'Please type something.' });

      const sessionId = req.body.sessionId || crypto.randomUUID();

      // ─────────────────────────────────────────────────────────────────
      // ENHANCEMENT 1: Fuzzy Matching (correct typos)
      // ─────────────────────────────────────────────────────────────────
      const correctedMessage = fuzzyMatcher.correctText(message, vocabulary);
      console.log(`Original: "${message}" → Corrected: "${correctedMessage}"`);

      // ─────────────────────────────────────────────────────────────────
      // ENHANCEMENT 2: Entity Recognition (extract key info)
      // ─────────────────────────────────────────────────────────────────
      const entities = entityRecognizer.extract(message);
      console.log('Extracted entities:', entities);

      // ─────────────────────────────────────────────────────────────────
      // ENHANCEMENT 3: Synonym Expansion (understand variations)
      // ─────────────────────────────────────────────────────────────────
      const expandedMessage = synonymExpander.expand(correctedMessage);

      // ─────────────────────────────────────────────────────────────────
      // ENHANCEMENT 4: Context Awareness (track conversation)
      // ─────────────────────────────────────────────────────────────────
      if (!conversationContexts.has(sessionId)) {
        conversationContexts.set(sessionId, new ConversationContext(sessionId));
      }
      const context = conversationContexts.get(sessionId);

      // ─────────────────────────────────────────────────────────────────
      // Existing similarity scoring + NLP preprocessing
      // ─────────────────────────────────────────────────────────────────
      const lang = detectLanguage(message);
      const processedMessage = preprocess(expandedMessage, lang);

      let bestScore = 0;
      let bestIntent = null;
      let matchedPattern = null;

      for (const intent of knowledge) {
        for (const pattern of intent.patterns) {
          const processedPattern = preprocess(pattern, lang);
          const score = getTfIdfScore(processedMessage, processedPattern, knowledge, lang);
          const bayesScore = getBayesScore(processedMessage, intent.tag);

          const combined = (score * 0.6) + (bayesScore * 0.4);

          if (combined > bestScore) {
            bestScore = combined;
            bestIntent = intent;
            matchedPattern = pattern;
          }
        }
      }

      // ─────────────────────────────────────────────────────────────────
      // ENHANCEMENT 5: Confidence-based clarification (when uncertain)
      // ─────────────────────────────────────────────────────────────────
      if (bestScore < 0.15) {
        const clarification = clarificationManager.generateClarification(
          bestIntent?.tag,
          bestScore,
          entities
        );
        return res.json({
          response: clarification,
          tag: 'clarification_needed',
          confidence: bestScore,
          sessionId,
          lang
        });
      }

      // ─────────────────────────────────────────────────────────────────
      // ENHANCEMENT 6: Response Ranking (pick the best response)
      // ─────────────────────────────────────────────────────────────────
      const responses = lang === 'cy' ? bestIntent.responses.cy : bestIntent.responses.en;
      const selectedResponse = responseRanker.rank(
        bestIntent.tag,
        responses,
        message,
        context.getContext()
      );

      // ─────────────────────────────────────────────────────────────────
      // ENHANCEMENT 7: Learning System (track for improvement)
      // ─────────────────────────────────────────────────────────────────
      const success = bestScore > 0.3;
      chatbotLearner.recordInteraction(message, bestIntent.tag, bestScore, success);

      // ─────────────────────────────────────────────────────────────────
      // ENHANCEMENT 8: Update Context (for multi-turn conversations)
      // ─────────────────────────────────────────────────────────────────
      context.addTurn(message, selectedResponse, bestIntent.tag, entities);

      // ─────────────────────────────────────────────────────────────────
      // SAVE FEEDBACK & RETURN RESPONSE
      // ─────────────────────────────────────────────────────────────────
      if (process.env.MONGODB_URI) {
        try {
          const client = new MongoClient(process.env.MONGODB_URI);
          await client.connect();
          const feedback = client.db('upal').collection('feedback');
          await feedback.insertOne({
            timestamp: new Date(),
            userMessage: message,
            correctedMessage,
            entities,
            intent: bestIntent.tag,
            confidence: bestScore,
            responseGiven: selectedResponse,
            language: lang,
            sessionId
          });
          await client.close();
        } catch (err) {
          console.error('Feedback insert error:', err.message);
        }
      }

      res.json({
        response: selectedResponse,
        tag: bestIntent.tag,
        confidence: bestScore,
        entities,
        clarifiedText: correctedMessage,
        contextAwareness: {
          sessionLength: context.history.length,
          userProfile: context.userData
        },
        sessionId,
        lang
      });

    } catch (err) {
      console.error('Chat error:', err);
      res.status(500).json({ error: err.message });
    }
  });
*/

// ═══════════════════════════════════════════════════════════════════════════
// STEP 5: ADD ANALYTICS ENDPOINT (optional but powerful)
// ═══════════════════════════════════════════════════════════════════════════
/*
  app.get('/api/analytics/insights', (req, res) => {
    const password = req.headers['x-admin-password'];
    if (password !== process.env.ADMIN_PASSWORD) {
      return res.status(403).json({ error: 'Forbidden' });
    }

    const insights = chatbotLearner.generateInsights();
    res.json({
      timestamp: new Date(),
      insights,
      activeContexts: conversationContexts.size,
      topConfusions: chatbotLearner.getCommonConfusions()
    });
  });
*/

// ═══════════════════════════════════════════════════════════════════════════
// STEP 6: ONGOING TRAINING - Update knowledge base as you learn
// ═══════════════════════════════════════════════════════════════════════════
/*
  // The chatbotLearner tracks what confuses the bot.
  // Periodically review commonConfusions and add new patterns:
  
  const confusions = chatbotLearner.getCommonConfusions();
  // If you see "fees_tuition -> fees_and_finance" often,
  // it means you should add more patterns to fees_and_finance
  // that are similar to tuition questions.
  
  // Add these to your knowledge.json:
  {
    "tag": "fees_and_finance",
    "patterns": [
      "what is the tuition cost",
      "how much tuition fee",
      "tuition payment options",
      ... more tuition-related patterns
    ],
    "responses": { ... }
  }
*/

// ═══════════════════════════════════════════════════════════════════════════
// ARCHITECTURE SUMMARY
// ═══════════════════════════════════════════════════════════════════════════

/*
  INPUT
    │
    ├─→ [1] FUZZY MATCHING ────────→ Correct typos
    │
    ├─→ [2] ENTITY RECOGNITION ────→ Extract: COURSE, CAMPUS, SUPPORT_TYPE
    │
    ├─→ [3] SYNONYM EXPANSION ────→ Add similar words for matching
    │
    ├─→ [4] TF-IDF SCORING ────────→ Vectorize & score against patterns
    │
    ├─→ [5] NAIVE BAYES ────────────→ Probability-based intent ranking
    │
    ├─→ [6] CONFIDENCE CHECK ──────→ If < 0.15, ask for clarification
    │
    ├─→ [7] CONTEXT MANAGER ───────→ Track conversation history
    │
    ├─→ [8] RESPONSE RANKING ──────→ Pick best response from options
    │
    └─→ [9] LEARNING SYSTEM ───────→ Track success/failure patterns
         
         OUTPUT + FEEDBACK TO MONGODB
*/

// ═══════════════════════════════════════════════════════════════════════════
// TRAINING YOUR CHATBOT (WITHOUT LLMs)
// ═══════════════════════════════════════════════════════════════════════════

/*
  1. START: Crawl UWTSD site → Extract info → Build knowledge.json
     (You already did this with Firecrawl!)

  2. ADD PATTERNS: For each intent, add 15-20 question variations
     - Think like a student: "How much is x?", "What's the cost?", etc.
     - Add both English & Welsh

  3. TEST & ITERATE: Run conversations, collect failures
     - Review chatbotLearner.getCommonConfusions()
     - Add patterns to fix confused intents

  4. OPTIMIZE RESPONSES: Rank responses by usefulness
     - Include URLs, contact info, actionable steps
     - Keep responses 150-250 words

  5. CONTINUOUS LEARNING:
     - Monitor analytics endpoint
     - Update patterns weekly based on real user queries
     - Retrain Naive Bayes classifier when you add patterns

  This is how LLMs work at a basic level:
  - They're trained on massive amounts of text
  - You're training this bot on YOUR specific domain (UWTSD)
  - Your knowledge is MUCH more reliable than generic LLMs
*/

module.exports = `
Integration complete! Your U-Pal chatbot now has:
✓ Typo correction (Fuzzy matching)
✓ Entity extraction (What's the user asking about?)
✓ Context awareness (Remember the conversation)
✓ Synonym understanding (Recognize word variations)
✓ Confidence-based clarification (Ask when unsure)
✓ Smart response selection (Pick the best answer)
✓ Learning system (Improve from mistakes)
✓ Analytics (Track what's working/failing)
`;
