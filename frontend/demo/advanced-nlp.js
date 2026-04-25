/**
 * ADVANCED NLP ENHANCEMENTS FOR U-PAL CHATBOT
 * 
 * These techniques make the chatbot smarter WITHOUT using LLM APIs:
 * 1. Fuzzy matching (handle typos)
 * 2. Entity extraction (recognize key info)
 * 3. Context awareness (remember conversation)
 * 4. Confidence-based clarification
 * 5. Synonym expansion (understand variations)
 * 6. Response ranking (pick best response)
 * 7. Multi-turn dialogue (follow-up questions)
 * 8. Learning from feedback (improve over time)
 */

const natural = require('natural');
const Levenshtein = require('fast-levenshtein');

// ═══════════════════════════════════════════════════════════════════════════
// 1. FUZZY MATCHING - Handle typos and misspellings
// ═══════════════════════════════════════════════════════════════════════════

class FuzzyMatcher {
  constructor(vocabulary, threshold = 0.85) {
    this.vocabulary = vocabulary;
    this.threshold = threshold;
  }

  correctWord(word) {
    let bestMatch = word;
    let bestScore = 1;

    for (const vocabWord of this.vocabulary) {
      const distance = Levenshtein.get(word.toLowerCase(), vocabWord.toLowerCase());
      const maxLen = Math.max(word.length, vocabWord.length);
      const score = 1 - (distance / maxLen);

      if (score > this.threshold && score < bestScore) {
        bestMatch = vocabWord;
        bestScore = score;
      }
    }
    return bestMatch;
  }

  // Correct entire text
  correctText(text, vocabulary) {
    const words = text.split(/\s+/);
    return words.map(w => this.correctWord(w)).join(' ');
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. ENTITY RECOGNITION - Extract key information from questions
// ═══════════════════════════════════════════════════════════════════════════

class EntityRecognizer {
  constructor() {
    // Define entity patterns
    this.entities = {
      COURSE: {
        patterns: [
          /\b(ba|bsc|ma|msc|mbamphil|phd|pgce|foundation|diploma|hnd|hnc)\b/gi,
          /\b(business|engineering|law|medicine|arts|science|technology)\b/gi
        ],
        type: 'COURSE_TYPE'
      },
      CAMPUS: {
        patterns: [
          /\b(swansea|carmarthen|lampeter|london|birmingham|cardiff)\b/gi
        ],
        type: 'CAMPUS'
      },
      SUPPORT_TYPE: {
        patterns: [
          /\b(accommodation|financial|wellbeing|mental health|careers|library|disability)\b/gi
        ],
        type: 'SUPPORT_CATEGORY'
      },
      FEES: {
        patterns: [
          /£\d+,?\d*/gi,
          /\b(cost|price|fee|tuition)\b/gi
        ],
        type: 'FEE_MENTION'
      }
    };
  }

  extract(text) {
    const extracted = {};

    for (const [entityName, config] of Object.entries(this.entities)) {
      extracted[entityName] = [];
      for (const pattern of config.patterns) {
        const matches = text.match(pattern);
        if (matches) {
          extracted[entityName].push(...matches);
        }
      }
      extracted[entityName] = [...new Set(extracted[entityName])]; // Deduplicate
    }

    return extracted;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. CONTEXT MANAGER - Track conversation history
// ═══════════════════════════════════════════════════════════════════════════

class ConversationContext {
  constructor(sessionId, maxHistory = 10) {
    this.sessionId = sessionId;
    this.history = [];
    this.maxHistory = maxHistory;
    this.userData = {
      interestedCourses: [],
      interestedCampus: null,
      supportNeeds: []
    };
  }

  addTurn(userMessage, botResponse, intent, entities) {
    this.history.push({
      timestamp: Date.now(),
      user: userMessage,
      bot: botResponse,
      intent,
      entities
    });

    // Keep only recent history
    if (this.history.length > this.maxHistory) {
      this.history.shift();
    }

    // Update user profile from entities
    if (entities.COURSE.length > 0) {
      this.userData.interestedCourses.push(...entities.COURSE);
    }
    if (entities.CAMPUS.length > 0) {
      this.userData.interestedCampus = entities.CAMPUS[0];
    }
    if (entities.SUPPORT_TYPE.length > 0) {
      this.userData.supportNeeds.push(...entities.SUPPORT_TYPE);
    }
  }

  getLastIntent() {
    if (this.history.length === 0) return null;
    return this.history[this.history.length - 1].intent;
  }

  getContext() {
    return {
      lastMessages: this.history.slice(-3),
      userProfile: this.userData,
      sessionLength: this.history.length
    };
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. SYNONYM EXPANSION - Understand word variations
// ═══════════════════════════════════════════════════════════════════════════

class SynonymExpander {
  constructor() {
    this.synonymMap = {
      // Fees/Cost
      'fee': ['cost', 'price', 'charge', 'tuition'],
      'cost': ['fee', 'price', 'charge', 'tuition', 'expense'],
      'free': ['no cost', 'zero cost', 'complimentary', 'no charge'],

      // Help/Support
      'help': ['support', 'assist', 'aid', 'guidance'],
      'support': ['help', 'assist', 'aid'],

      // Mental health
      'stressed': ['anxious', 'worried', 'overwhelmed', 'struggling'],
      'sad': ['depressed', 'upset', 'unhappy', 'down'],
      'anxiety': ['worry', 'stress', 'nervousness', 'panic'],

      // Accommodation
      'room': ['accommodation', 'housing', 'dorm', 'flat'],
      'house': ['accommodation', 'housing', 'flat'],
      'halls': ['accommodation', 'hostel', 'residences'],

      // Courses
      'degree': ['course', 'program', 'qualification'],
      'subject': ['course', 'module', 'field', 'discipline'],

      // Work/Employment
      'job': ['work', 'employment', 'position', 'role'],
      'work': ['job', 'employment', 'position', 'career'],
      'internship': ['placement', 'work experience', 'traineeship']
    };
  }

  expand(text) {
    let expandedText = text;
    const words = text.toLowerCase().split(/\s+/);

    for (const word of words) {
      if (this.synonymMap[word]) {
        // Add synonyms as OR alternatives for better matching
        const synonyms = this.synonymMap[word].join('|');
        expandedText += ` ${synonyms}`;
      }
    }

    return expandedText;
  }

  // Get semantic similarity between two words
  isSimilar(word1, word2, depth = 1) {
    if (word1.toLowerCase() === word2.toLowerCase()) return true;
    if (depth === 0) return false;

    const synonyms = this.synonymMap[word1.toLowerCase()] || [];
    for (const syn of synonyms) {
      if (this.isSimilar(syn, word2, depth - 1)) return true;
    }
    return false;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. CONFIDENCE-BASED CLARIFICATION - Ask for help when uncertain
// ═══════════════════════════════════════════════════════════════════════════

class ClarificationManager {
  generateClarification(intent, confidence, entities) {
    const clarifications = {
      low: [
        "I'm not quite sure I understood correctly. Are you asking about {topic}?",
        "Could you clarify a bit? Are you interested in {topic}?",
        "I think you're asking about {topic}, but could you tell me more?"
      ],
      medium: [
        "I think you're asking about {topic}. Should I provide information on {topic}?",
        "Is it about {topic} that you want to know?",
        "Let me confirm - you want to know about {topic}?"
      ]
    };

    const topic = this.extractTopic(entities) || intent || 'that topic';
    const level = confidence < 0.3 ? 'low' : 'medium';
    const template = clarifications[level][Math.floor(Math.random() * clarifications[level].length)];

    return template.replace('{topic}', topic);
  }

  extractTopic(entities) {
    if (entities.SUPPORT_TYPE && entities.SUPPORT_TYPE.length > 0) {
      return entities.SUPPORT_TYPE[0];
    }
    if (entities.COURSE && entities.COURSE.length > 0) {
      return entities.COURSE[0];
    }
    return null;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. RESPONSE RANKING - Select best response from multiple options
// ═══════════════════════════════════════════════════════════════════════════

class ResponseRanker {
  rank(intent, responses, userMessage, context) {
    // Score responses based on multiple factors
    const scores = responses.map((response, index) => {
      let score = 0;

      // Factor 1: Length appropriateness (prefer balanced responses)
      const len = response.length;
      const optimalLen = 200;
      score += Math.max(0, 50 - Math.abs(len - optimalLen) / 10);

      // Factor 2: Relevance keywords in context
      if (context.userData.interestedCampus && response.toLowerCase().includes(context.userData.interestedCampus.toLowerCase())) {
        score += 30;
      }

      // Factor 3: Recency (prefer responses not used recently)
      if (context.lastMessages.some(m => m.bot === response)) {
        score -= 50; // Penalize repetition
      }

      // Factor 4: CTAs and URLs (prefer actionable responses)
      if (response.includes('www.') || response.includes('www') || response.includes('contact')) {
        score += 20;
      }

      return { response, score, index };
    });

    // Sort by score descending
    return scores.sort((a, b) => b.score - a.score)[0].response;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// 7. LEARNING SYSTEM - Track interactions for improvement
// ═══════════════════════════════════════════════════════════════════════════

class ChatbotLearner {
  constructor() {
    this.interactions = [];
    this.intentConfusions = {};
    this.failurePatterns = [];
  }

  recordInteraction(userMessage, intent, confidence, success) {
    this.interactions.push({
      timestamp: Date.now(),
      message: userMessage,
      intent,
      confidence,
      success
    });
  }

  recordMisclassification(userMessage, predictedIntent, actualIntent) {
    const key = `${predictedIntent}->${actualIntent}`;
    this.intentConfusions[key] = (this.intentConfusions[key] || 0) + 1;
  }

  getCommonConfusions() {
    // Get the most common intent confusions
    return Object.entries(this.intentConfusions)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 5);
  }

  getSuccessRate() {
    if (this.interactions.length === 0) return 0;
    const successful = this.interactions.filter(i => i.success).length;
    return (successful / this.interactions.length) * 100;
  }

  getAverageConfidence() {
    if (this.interactions.length === 0) return 0;
    const totalConfidence = this.interactions.reduce((sum, i) => sum + i.confidence, 0);
    return totalConfidence / this.interactions.length;
  }

  generateInsights() {
    return {
      successRate: this.getSuccessRate().toFixed(1) + '%',
      averageConfidence: this.getAverageConfidence().toFixed(2),
      commonConfusions: this.getCommonConfusions(),
      totalInteractions: this.interactions.length,
      uniqueIntents: new Set(this.interactions.map(i => i.intent)).size
    };
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// EXPORT FOR USE IN SERVER.JS
// ═══════════════════════════════════════════════════════════════════════════

module.exports = {
  FuzzyMatcher,
  EntityRecognizer,
  ConversationContext,
  SynonymExpander,
  ClarificationManager,
  ResponseRanker,
  ChatbotLearner
};
