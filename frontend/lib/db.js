'use strict';
// feedback persistence. two backends: MongoDB Atlas for the deployed
// Vercel build, and a local JSON file for dev.
//
// on Vercel every API invocation gets a fresh Node process, so we cache
// the MongoClient in module scope. Node keeps the module alive between
// warm invocations, which means we skip the TCP+TLS handshake on repeat
// calls. ref: https://vercel.com/guides/how-can-i-use-mongodb-atlas-with-vercel

const { MongoClient } = require('mongodb');
const path            = require('path');
const fs              = require('fs');

const MONGODB_URI = process.env.MONGODB_URI;

// on Vercel the only writable path is /tmp, so we put the local fallback
// there in production. in dev we use the cwd so the file survives restarts.
// ref: https://vercel.com/docs/functions/runtimes/node-js#filesystem
const FEEDBACK_FILE = path.join(process.env.VERCEL ? '/tmp' : process.cwd(), 'feedback.json');

let cachedClient = null;

async function getCollection() {
  // lazy connect so we don't pay the handshake cost unless someone actually
  // submits or reads feedback
  if (!cachedClient) {
    cachedClient = new MongoClient(MONGODB_URI);
    await cachedClient.connect();
  }
  return cachedClient.db('upal-rag').collection('feedback');
}

async function saveFeedback(entry) {
  if (MONGODB_URI) {
    const col = await getCollection();
    await col.insertOne(entry);
    return;
  }
  // JSON fallback. we read-modify-write the whole file because the volume
  // is tiny (a dissertation demo, not production) and it keeps the code
  // dependency-free. for anything larger we would use an append-only log.
  let existing = [];
  try {
    if (fs.existsSync(FEEDBACK_FILE))
      existing = JSON.parse(fs.readFileSync(FEEDBACK_FILE, 'utf8'));
  } catch (_) {
    // corrupt file, start fresh rather than crashing the API route
  }
  existing.push(entry);
  fs.writeFileSync(FEEDBACK_FILE, JSON.stringify(existing, null, 2), 'utf8');
}

async function readFeedback() {
  if (MONGODB_URI) {
    const col = await getCollection();
    // projection strips the internal _id so the admin page JSON is clean.
    // sort newest-first for the dashboard.
    // ref: https://www.mongodb.com/docs/manual/reference/method/db.collection.find/
    return col.find({}, { projection: { _id: 0 } }).sort({ timestamp: -1 }).toArray();
  }
  if (!fs.existsSync(FEEDBACK_FILE)) return [];
  try {
    // .reverse() so the newest entry is first, matching the Mongo branch
    return JSON.parse(fs.readFileSync(FEEDBACK_FILE, 'utf8')).reverse();
  } catch (_) {
    return [];
  }
}

module.exports = { saveFeedback, readFeedback };
