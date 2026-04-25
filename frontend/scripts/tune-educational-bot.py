#!/usr/bin/env python3
"""
tune-educational-bot.py — RAG-based tuning for the UWTSD educational chatbot
============================================================================

Goal:
    Teach the LLM how an educational student-support chatbot should *sound*
    when users send simple human interactions — acknowledgements ("ok"),
    clarifications ("what?" "huh?"), continuations ("tell me more"),
    yes/no answers, confusion ("I'm lost"), greetings, and off-topic chat.

    Because we can't fine-tune Ollama on the free Oracle VM, we do the next
    best thing: we generate a corpus of *exemplar* behaviour chunks and
    ingest them into Morphik.  At chat time, the RAG retriever surfaces the
    closest exemplar alongside the factual UWTSD content, and the LLM
    imitates the style.  Effectively the educational-bot personality is
    embedded in the knowledge base itself — no JS code changes needed.

Pipeline:
    1. Scrape real human conversation patterns from the HuggingFace
       DailyDialog dataset (13,118 dialogues, labelled with dialogue acts)
    2. Classify short utterances into four categories: clarify,
       acknowledge, ask_more, yes_no — these are the phrasings students
       actually use that the bot must handle gracefully
    3. Pair each mined category with a hand-crafted *educational chatbot
       response template* (English + Welsh) that models the ideal reply
    4. Generate bilingual Q/A exemplar chunks that teach the LLM:
           "When a student says <short utterance>, an educational UWTSD
            chatbot replies with <friendly, redirecting question>."
    5. Bundle each chunk with rich metadata (category, language, utterance)
       and POST them to Morphik /ingest/text so they live alongside the
       real UWTSD factual passages.

Usage:
    # Run against a local Morphik (recommended — on the Oracle VM):
    MORPHIK_URL=http://localhost:8000 python3 scripts/tune-educational-bot.py

    # Or skip ingestion and just generate the exemplar file:
    DRY_RUN=1 python3 scripts/tune-educational-bot.py

    # Use a different dataset or language slice:
    HF_DATASET=daily_dialog MAX_ROWS=13000 python3 scripts/tune-educational-bot.py

Outputs:
    ./educational-exemplars.json — the generated exemplar chunks (for
                                   review / reproducibility)
    Morphik                       — ingested chunks tagged
                                   {source:"educational-bot-tuning", ...}

Env vars:
    MORPHIK_URL   Default http://localhost:8000 (use the VM URL otherwise)
    HF_DATASET    Default daily_dialog
    HF_CONFIG     Default default
    HF_SPLIT      Default train
    MAX_ROWS      Default 13000
    BATCH         Default 100
    DELAY         Default 0.3 (seconds between HF requests)
    DRY_RUN       If set, don't POST to Morphik — just write local JSON
    OUT_PATH      Default ./educational-exemplars.json

Why RAG-tuning instead of JS regex:
    Regex patterns are brittle — they miss phrasings like "could you repeat
    that please" or "I'm not quite following you."  Vector search over
    embedded exemplars handles paraphrases naturally.  Crucially, the LLM
    sees the exemplar in its context window and *imitates the style*,
    which generalises to phrasings not in the mined set.
"""

import os
import sys
import json
import time
import re
import hashlib
import urllib.request
import urllib.parse
import urllib.error

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HF_API      = "https://datasets-server.huggingface.co/rows"
DATASET     = os.environ.get("HF_DATASET", "daily_dialog")
CONFIG      = os.environ.get("HF_CONFIG",  "default")
SPLIT       = os.environ.get("HF_SPLIT",   "train")
BATCH       = int(os.environ.get("BATCH", "100"))
MAX_ROWS    = int(os.environ.get("MAX_ROWS", "13000"))
DELAY       = float(os.environ.get("DELAY", "0.3"))

MORPHIK_URL = os.environ.get("MORPHIK_URL", "http://localhost:8000").rstrip("/")
DRY_RUN     = bool(os.environ.get("DRY_RUN"))

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUT_PATH     = os.environ.get(
    "OUT_PATH",
    os.path.join(PROJECT_ROOT, "educational-exemplars.json"),
)

# ---------------------------------------------------------------------------
# Classification patterns — narrow regexes to tag short utterances.
# High precision, low recall.  Anything that matches becomes a training
# exemplar, so we err on the side of catching only unambiguous phrasings.
# ---------------------------------------------------------------------------
CLARIFY_PATTERNS = [
    r"^what\??$", r"^huh\??$", r"^pardon( me)?\??$", r"^sorry\??$",
    r"^(i )?beg your pardon\??$", r"^come again\??$",
    r"^(could|can|would) you repeat( that| it)?( please)?\??$",
    r"^(could|can) you (say|repeat) that again\??$",
    r"^(i )?don'?t (quite )?understand\.?$",
    r"^(i'?m )?not sure (i )?(understand|follow)\.?$",
    r"^what do you mean\??$",
    r"^what are you (saying|talking about)\??$",
    r"^(i'?m )?confused\.?$",
    r"^say (that )?again( please)?\??$",
    r"^didn'?t (quite )?catch (that|you)\.?$",
    r"^eh\??$", r"^mm\??$", r"^hmm\??$",
]

ACKNOWLEDGE_PATTERNS = [
    r"^ok(ay)?\.?!?$", r"^alright\.?!?$", r"^cool\.?!?$", r"^nice\.?!?$",
    r"^(sounds )?good\.?!?$", r"^great\.?!?$", r"^fine\.?!?$",
    r"^got it\.?!?$", r"^gotcha\.?!?$", r"^understood\.?!?$", r"^noted\.?!?$",
    r"^i see\.?!?$", r"^makes sense\.?!?$", r"^that makes sense\.?!?$",
    r"^fair enough\.?!?$", r"^right\.?!?$", r"^(all )?right then\.?!?$",
    r"^sure(,? thanks)?\.?!?$", r"^perfect\.?!?$", r"^awesome\.?!?$",
]

ASK_MORE_PATTERNS = [
    r"^(please )?(tell|say) me more( about (it|that))?\.?!?\??$",
    r"^more (please|details|info|information)?\.?!?\??$",
    r"^(go on|keep going|continue)\.?!?\??$",
    r"^(and|so|well|then)\??$",
    r"^anything else\??$",
    r"^what else\??$",
    r"^(can you )?elaborate( on (this|that))?\??$",
    r"^expand on (this|that)\??$",
    r"^(please )?explain( (more|further|it))?\??$",
]

YES_NO_PATTERNS = [
    r"^yes\.?!?$", r"^yes please\.?!?$", r"^yeah\.?!?$", r"^yep\.?!?$", r"^yup\.?!?$",
    r"^(yes,? )?(of course|certainly|definitely|absolutely)\.?!?$",
    r"^sure( thing)?\.?!?$",
    r"^no\.?!?$", r"^no thanks\.?!?$", r"^nope\.?!?$", r"^nah\.?!?$",
    r"^(not )?really\.?!?$", r"^i don'?t think so\.?!?$",
]

CATEGORIES = {
    "clarify":     [re.compile(p, re.I) for p in CLARIFY_PATTERNS],
    "acknowledge": [re.compile(p, re.I) for p in ACKNOWLEDGE_PATTERNS],
    "ask_more":    [re.compile(p, re.I) for p in ASK_MORE_PATTERNS],
    "yes_no":      [re.compile(p, re.I) for p in YES_NO_PATTERNS],
}

# Hand-curated seed utterances — always present even if the HF API is down.
# Includes Welsh (cy) so the bilingual bot gets coverage for both languages.
SEED_PHRASES = {
    "clarify": [
        "what", "huh", "sorry", "pardon", "pardon me", "what do you mean",
        "i dont understand", "come again", "say that again",
        "repeat that", "repeat please", "could you repeat",
        "didnt catch that", "im confused", "not sure i follow",
        # Welsh
        "beth", "sori", "mae'n ddrwg gennyf", "beth ydych chin ei feddwl",
        "ailadroddwch", "dydw i ddim yn deall", "dwin drysu",
    ],
    "acknowledge": [
        "ok", "okay", "alright", "cool", "nice", "sounds good",
        "got it", "gotcha", "understood", "noted", "i see",
        "makes sense", "fair enough", "right", "perfect", "awesome",
        # Welsh
        "iawn", "ocê", "wrth gwrs", "deallaf", "da iawn",
    ],
    "ask_more": [
        "tell me more", "more", "more please", "more info",
        "go on", "keep going", "continue", "anything else",
        "what else", "elaborate", "explain",
        # Welsh
        "mwy", "mwy os gwelwch yn dda", "parhau", "ymhelaethu",
        "dywedwch fwy", "beth arall",
    ],
    "yes_no": [
        "yes", "yes please", "yeah", "yep", "yup", "of course",
        "certainly", "definitely", "absolutely", "sure",
        "no", "no thanks", "nope", "nah", "not really",
        # Welsh
        "ydw", "ie", "wrth gwrs", "na", "nac ydw", "dim diolch",
    ],
}

# ---------------------------------------------------------------------------
# Educational-chatbot response templates
#
# For each category of simple human interaction, we provide an ideal
# response that:
#   - Is warm and conversational (not corporate)
#   - Stays in the educational-support domain (UWTSD-focused)
#   - Offers concrete next-step topics so the conversation progresses
#   - Does NOT attempt to answer a question that wasn't asked
#   - Does NOT hallucinate UWTSD facts
#
# The LLM reads these exemplars via RAG and imitates the style when it
# encounters similar user inputs.
# ---------------------------------------------------------------------------
EXEMPLAR_TEMPLATES = {
    "clarify": [
        {
            "student_en": "what?",
            "student_cy": "beth?",
            "reply_en": "Sorry if that wasn't clear! Could you tell me what part of UWTSD life you'd like help with? I can chat about applying, courses, fees, funding, accommodation, campus services, or wellbeing.",
            "reply_cy": "Mae'n ddrwg gen i os nad oedd hynny'n glir! Allech chi ddweud wrthyf pa ran o fywyd PCYDDS yr hoffech gael help gyda hi? Gallaf sgwrsio am ymgeisio, cyrsiau, ffioedd, ariannu, llety, gwasanaethau campws, neu les.",
        },
        {
            "student_en": "I don't understand.",
            "student_cy": "Dydw i ddim yn deall.",
            "reply_en": "No problem — let's take it step by step. Which bit is unclear? Or, if you'd like a fresh start, just tell me the topic you're interested in — admissions, a specific course, accommodation, student funding, anything like that.",
            "reply_cy": "Dim problem — cymerwn ni hi gam wrth gam. Pa ran sy'n aneglur? Neu, os hoffech ddechrau o'r newydd, dywedwch wrthyf y pwnc sydd o ddiddordeb ichi — derbyniadau, cwrs penodol, llety, ariannu myfyrwyr, rhywbeth felly.",
        },
        {
            "student_en": "Could you repeat that please?",
            "student_cy": "Allech chi ailadrodd hynny os gwelwch yn dda?",
            "reply_en": "Of course! It'd help me give you the clearest answer if you tell me the specific topic — for example, 'tuition fees for international students' or 'how to apply for accommodation'.",
            "reply_cy": "Wrth gwrs! Byddai'n fy helpu i roi'r ateb cliriaf ichi os dywedwch wrthyf y pwnc penodol — er enghraifft, 'ffioedd dysgu ar gyfer myfyrwyr rhyngwladol' neu 'sut i ymgeisio am lety'.",
        },
        {
            "student_en": "What do you mean?",
            "student_cy": "Beth ydych chi'n ei feddwl?",
            "reply_en": "Happy to rephrase — which part would you like me to clarify? Or give me the keyword you're stuck on and I'll explain it in plain terms.",
            "reply_cy": "Yn falch o aralleirio — pa ran hoffech chi imi egluro? Neu rhowch y gair allweddol rydych chi'n cael trafferth gydag ef a byddaf yn ei egluro mewn termau syml.",
        },
        {
            "student_en": "I'm confused.",
            "student_cy": "Dwi'n drysu.",
            "reply_en": "That's OK — university life has a lot of moving parts. Let's narrow it down. Are you asking about applying, studying, funding, living at UWTSD, or getting support?",
            "reply_cy": "Mae hynny'n iawn — mae gan fywyd prifysgol lawer o elfennau. Gadewch i ni gulhau'r cwestiwn. Ydych chi'n gofyn am ymgeisio, astudio, ariannu, byw ym PCYDDS, neu gael cymorth?",
        },
    ],
    "acknowledge": [
        {
            "student_en": "ok",
            "student_cy": "iawn",
            "reply_en": "Great — anything else I can help you figure out? I can cover admissions, courses, fees, scholarships, accommodation, campus services, IT and Wi-Fi, wellbeing, or anything else student-related.",
            "reply_cy": "Gwych — unrhyw beth arall y gallaf eich helpu i'w ddeall? Gallaf drafod derbyniadau, cyrsiau, ffioedd, ysgoloriaethau, llety, gwasanaethau campws, TG a Wi-Fi, lles, neu unrhyw beth arall sy'n ymwneud â myfyrwyr.",
        },
        {
            "student_en": "Got it.",
            "student_cy": "Deallaf.",
            "reply_en": "Brilliant. Want to go deeper on that topic, or move on to something else — timetables, student finance, the Students' Union, campus locations?",
            "reply_cy": "Ardderchog. Hoffech chi fynd yn ddyfnach i'r pwnc hwnnw, neu symud ymlaen i rywbeth arall — amserlenni, cyllid myfyrwyr, Undeb y Myfyrwyr, lleoliadau campws?",
        },
        {
            "student_en": "Makes sense.",
            "student_cy": "Mae'n gwneud synnwyr.",
            "reply_en": "Glad that helped. Is there a related area you'd like to explore next, or a different UWTSD question on your mind?",
            "reply_cy": "Yn falch bod hynny wedi helpu. A oes maes cysylltiedig yr hoffech ei archwilio nesaf, neu gwestiwn PCYDDS gwahanol ar eich meddwl?",
        },
        {
            "student_en": "Cool.",
            "student_cy": "Cŵl.",
            "reply_en": "Nice one! Shall I tell you about something else — maybe scholarships, accommodation in Swansea or Carmarthen, or the application timeline?",
            "reply_cy": "Da iawn! A ddylwn i ddweud wrthych am rywbeth arall — efallai ysgoloriaethau, llety yn Abertawe neu Gaerfyrddin, neu amserlen ymgeisio?",
        },
    ],
    "ask_more": [
        {
            "student_en": "Tell me more.",
            "student_cy": "Dywedwch fwy wrthyf.",
            "reply_en": "Happy to go further — which angle interests you most? I can zoom in on the application process, entry requirements, fees, funding options, course content, campus life, or student support.",
            "reply_cy": "Yn falch o fynd ymhellach — pa ongl sydd o'r diddordeb mwyaf ichi? Gallaf ganolbwyntio ar y broses ymgeisio, gofynion mynediad, ffioedd, opsiynau ariannu, cynnwys cwrs, bywyd campws, neu gefnogaeth myfyrwyr.",
        },
        {
            "student_en": "Go on.",
            "student_cy": "Ewch ymlaen.",
            "reply_en": "Sure — what would you like me to expand on? Give me a keyword (like 'deadlines', 'halls of residence', 'Erasmus', 'scholarships') and I'll dig deeper.",
            "reply_cy": "Wrth gwrs — ar beth hoffech chi imi ehangu? Rhowch air allweddol imi (fel 'terfynau amser', 'neuaddau preswyl', 'Erasmus', 'ysgoloriaethau') a byddaf yn archwilio ymhellach.",
        },
        {
            "student_en": "What else?",
            "student_cy": "Beth arall?",
            "reply_en": "Plenty to explore! Popular topics: open days, virtual tours, BTEC/A-level entry, postgraduate routes, placements, Welsh-medium study, disability support, and student wellbeing. Which fits?",
            "reply_cy": "Digonedd i'w archwilio! Pynciau poblogaidd: diwrnodau agored, teithiau rhithwir, mynediad BTEC/Safon Uwch, llwybrau ôl-raddedig, lleoliadau, astudio trwy gyfrwng y Gymraeg, cymorth anabledd, a lles myfyrwyr. Pa un sy'n addas?",
        },
        {
            "student_en": "Elaborate.",
            "student_cy": "Ymhelaethwch.",
            "reply_en": "Of course — which point should I expand on? I'll give you a clear, practical answer if you name the specific thing.",
            "reply_cy": "Wrth gwrs — ar ba bwynt ddylwn i ehangu? Byddaf yn rhoi ateb clir, ymarferol ichi os enwch y peth penodol.",
        },
    ],
    "yes_no": [
        {
            "student_en": "Yes please.",
            "student_cy": "Ie os gwelwch yn dda.",
            "reply_en": "Great — what specifically would you like me to help with next? Tell me the topic or keyword and I'll pull together the relevant UWTSD info.",
            "reply_cy": "Gwych — beth yn benodol hoffech chi imi helpu â hi nesaf? Dywedwch y pwnc neu'r gair allweddol wrthyf a byddaf yn casglu'r wybodaeth berthnasol am PCYDDS.",
        },
        {
            "student_en": "No thanks.",
            "student_cy": "Dim diolch.",
            "reply_en": "No problem! I'm here whenever you need something. If anything else crosses your mind about UWTSD, just ask.",
            "reply_cy": "Dim problem! Rydw i yma pryd bynnag y bydd angen rhywbeth arnoch. Os daw unrhyw beth arall i'ch meddwl am PCYDDS, gofynnwch.",
        },
        {
            "student_en": "Yeah.",
            "student_cy": "Ydw.",
            "reply_en": "Cool — which direction? Happy to cover admissions, courses, fees, accommodation, wellbeing, or anything in between.",
            "reply_cy": "Cŵl — pa gyfeiriad? Yn falch o drafod derbyniadau, cyrsiau, ffioedd, llety, lles, neu unrhyw beth yn y canol.",
        },
    ],
}

# ---------------------------------------------------------------------------
# HuggingFace fetch helpers
# ---------------------------------------------------------------------------
def fetch_batch(offset):
    params = urllib.parse.urlencode({
        "dataset": DATASET,
        "config":  CONFIG,
        "split":   SPLIT,
        "offset":  offset,
        "length":  BATCH,
    })
    req = urllib.request.Request(
        f"{HF_API}?{params}",
        headers={"User-Agent": "UPal-EduTuner/1.0"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8")).get("rows", [])


def extract_utterances(row):
    d = row.get("row", {})
    dialog = d.get("dialog") or d.get("utterances") or []
    if not isinstance(dialog, list):
        return []
    return [str(u) for u in dialog if u]


def classify(utterance):
    raw = utterance.strip()
    if not raw or len(raw) > 60 or len(raw.split()) > 6:
        return None
    for cat, patterns in CATEGORIES.items():
        for p in patterns:
            if p.fullmatch(raw):
                return cat
    return None


# ---------------------------------------------------------------------------
# Exemplar chunk builders
# ---------------------------------------------------------------------------
def build_exemplar_chunk(category, student_en, student_cy, reply_en, reply_cy,
                         related_utterances=None):
    """
    Build one RAG-ready passage that teaches the LLM how to respond to a
    specific flavour of simple human interaction.  The chunk is phrased as a
    narrated example so the LLM sees the Q and A together and infers the
    style to imitate.
    """
    related = ""
    if related_utterances:
        shown = sorted(set(related_utterances))[:20]
        related = (
            "\n\nReal students also phrase this as: "
            + ", ".join(f'"{u}"' for u in shown)
            + "."
        )
    title = {
        "clarify":     "UWTSD chatbot — how to reply to a student asking for clarification",
        "acknowledge": "UWTSD chatbot — how to reply to a short acknowledgement",
        "ask_more":    "UWTSD chatbot — how to reply when a student wants more detail",
        "yes_no":      "UWTSD chatbot — how to reply to a bare yes/no answer",
    }[category]

    body = (
        f"{title}.\n\n"
        f"When a UWTSD student types a short message like \"{student_en}\" "
        f"(or in Welsh, \"{student_cy}\"), the educational-support chatbot "
        f"should NOT try to invent a factual UWTSD answer.  Instead it "
        f"should respond warmly and steer the conversation back to a "
        f"concrete topic the student can pick from.\n\n"
        f"Exemplar English reply:\n{reply_en}\n\n"
        f"Exemplar Welsh (Cymraeg) reply:\n{reply_cy}"
        f"{related}"
    )
    return {
        "category": category,
        "lang":     "bilingual",
        "title":    title,
        "content":  body,
        "metadata": {
            "source":            "educational-bot-tuning",
            "tuning_category":   category,
            "student_prompt_en": student_en,
            "student_prompt_cy": student_cy,
            "chunk_kind":        "dialogue-act-exemplar",
        },
    }


def build_persona_chunk():
    """
    One overarching passage that defines the bot's personality.  Sits at the
    top of the exemplar corpus so any RAG query that even loosely matches
    "chatbot / behaviour / persona" pulls it in.
    """
    body = (
        "UWTSD U-Pal educational chatbot — persona and conversation style.\n\n"
        "U-Pal is a bilingual (English + Welsh) student-support chatbot for "
        "the University of Wales Trinity Saint David (Prifysgol Cymru Y "
        "Drindod Dewi Sant — PCYDDS).  It speaks to prospective and current "
        "students in a warm, concise, practical voice.\n\n"
        "Core rules U-Pal follows:\n"
        "1. Answer only from verified UWTSD information.  If unsure, say so "
        "   honestly and point the student at staff contacts.\n"
        "2. Never describe the retrieval process ('the documents show...', "
        "   'based on your context...').  Speak to the student directly.\n"
        "3. Match the student's language — English replies in English, "
        "   Welsh queries in Welsh.\n"
        "4. For one-word messages (what? ok. huh? more.) DO NOT hallucinate "
        "   a UWTSD fact — instead ask a gentle clarifying question with "
        "   concrete topic options.\n"
        "5. Be brief.  Prefer 2–4 short sentences over a wall of text.\n"
        "6. For emotional or wellbeing content, lead with empathy and flag "
        "   the UWTSD Wellbeing Service and Samaritans (116 123).\n"
        "7. Refer to the institution as 'UWTSD' in English and 'PCYDDS' in "
        "   Welsh.  Never invent course names or staff names."
    )
    return {
        "category": "persona",
        "lang":     "bilingual",
        "title":    "UWTSD U-Pal chatbot persona and style guide",
        "content":  body,
        "metadata": {
            "source":         "educational-bot-tuning",
            "tuning_category": "persona",
            "chunk_kind":     "persona-charter",
        },
    }


# ---------------------------------------------------------------------------
# Morphik ingestion
# ---------------------------------------------------------------------------
def ingest_to_morphik(chunk):
    """POST a single chunk to Morphik /ingest/text."""
    payload = {
        "content":  chunk["content"],
        "metadata": chunk["metadata"],
        # No auth rules — this is a public UWTSD knowledge base
    }
    req = urllib.request.Request(
        f"{MORPHIK_URL}/ingest/text",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type":   "application/json",
            "Accept":         "application/json",
            "User-Agent":     "UPal-EduTuner/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read()[:200]
    except urllib.error.HTTPError as e:
        return e.code, e.read()[:200]
    except Exception as e:
        return None, str(e)[:200]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 64)
    print("  UWTSD educational-bot RAG tuner")
    print("=" * 64)
    print(f"  Dataset    : {DATASET} / {CONFIG} / {SPLIT}")
    print(f"  Max rows   : {MAX_ROWS:,}")
    print(f"  Morphik    : {MORPHIK_URL} {'(dry run — not ingesting)' if DRY_RUN else ''}")
    print(f"  Output file: {OUT_PATH}")
    print()

    # ─── 1. Harvest real human phrasings from DailyDialog ────────────────
    mined = {cat: set() for cat in CATEGORIES}
    mined_any = False
    offset = 0
    scanned = 0
    print("  Step 1/3  Harvesting phrasings from HuggingFace DailyDialog...")
    while offset < MAX_ROWS:
        try:
            rows = fetch_batch(offset)
        except Exception as e:
            print(f"  WARN: HF batch @ {offset} failed ({e}) — continuing with seeds only.")
            break
        if not rows:
            break
        mined_any = True
        for row in rows:
            for utt in extract_utterances(row):
                scanned += 1
                cat = classify(utt)
                if cat:
                    mined[cat].add(utt.strip().lower())
        offset += BATCH
        if (offset // BATCH) % 10 == 0:
            totals = {c: len(v) for c, v in mined.items()}
            print(f"    rows={offset:,} utterances_seen={scanned:,} mined={totals}")
        time.sleep(DELAY)

    # Fall back to seeds if network was unavailable
    for cat, phrases in SEED_PHRASES.items():
        for p in phrases:
            mined[cat].add(p.lower())

    totals = {c: len(v) for c, v in mined.items()}
    print(f"  Done — unique mined phrasings: {totals}")
    print()

    # ─── 2. Build exemplar chunks ────────────────────────────────────────
    print("  Step 2/3  Building educational exemplar chunks...")
    chunks = [build_persona_chunk()]
    for cat, templates in EXEMPLAR_TEMPLATES.items():
        related = list(mined.get(cat, set()))
        for tpl in templates:
            chunks.append(build_exemplar_chunk(
                category=cat,
                student_en=tpl["student_en"],
                student_cy=tpl["student_cy"],
                reply_en=tpl["reply_en"],
                reply_cy=tpl["reply_cy"],
                related_utterances=related,
            ))
    print(f"  Built {len(chunks)} chunks ({len(chunks)-1} exemplars + 1 persona)")
    print()

    # ─── 3. Write local JSON + (optionally) ingest to Morphik ────────────
    print(f"  Step 3/3  Writing {OUT_PATH}...")
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "generated_from": DATASET,
            "scanned_utterances": scanned,
            "mined_counts": totals,
            "exemplar_count": len(chunks) - 1,
            "chunks": chunks,
        }, f, ensure_ascii=False, indent=2)

    if DRY_RUN:
        print("  DRY_RUN set — skipping Morphik ingestion.")
    else:
        print(f"  Ingesting to Morphik @ {MORPHIK_URL}...")
        ok, fail = 0, 0
        for i, chunk in enumerate(chunks, 1):
            status, body = ingest_to_morphik(chunk)
            if status and 200 <= status < 300:
                ok += 1
            else:
                fail += 1
                print(f"    [{i}/{len(chunks)}] FAIL status={status} body={body!r}")
            if i % 5 == 0:
                print(f"    [{i}/{len(chunks)}] ok={ok} fail={fail}")
            time.sleep(0.1)
        print(f"  Ingestion complete: ok={ok} fail={fail}")

    print()
    print("=" * 64)
    print("  Tuning complete")
    print(f"  Exemplar file: {OUT_PATH}")
    if not DRY_RUN:
        print(f"  Morphik now contains {len(chunks)} educational-bot-tuning chunks.")
        print(f"  The chatbot will imitate these examples via RAG at query time —")
        print(f"  no lib/nlp.js changes required.")
    print("=" * 64)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
