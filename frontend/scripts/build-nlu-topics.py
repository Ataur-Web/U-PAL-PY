#!/usr/bin/env python3
"""
build-nlu-topics.py — 600+ topic UWTSD NLU database
============================================================================

Goal:
    Give the chatbot a proper NLU layer on top of Morphik RAG.  Modelled on
    the Microsoft Copilot Studio trigger-phrase approach: every topic has
    a short list of phrases students actually use, in English AND Welsh.

    The bot's lib/nlp.js matches the user's message against this database,
    picks the most likely topic, and:
        1. Augments the Morphik query with a topic-specific retrieval hint
           so RAG fetches the right chunks even from vague phrasings.
        2. Optionally serves a fast-path canned reply for high-priority
           topics (greetings, crisis support, capabilities).
        3. Tags the conversation for analytics.

Scale:
    600+ topics, ~15–20 phrases each → roughly 10,000 trigger phrases,
    bilingual (EN + CY), generated from a structured taxonomy so we don't
    have to hand-write everything.

    Topics are produced by cross-producting five taxonomies:
        • Core hand-curated topics         (~55)
        • UWTSD subject catalogue × 3      (~200)  — applying/modules/careers
        • Campus × service                 (~80)
        • Student-type × concern           (~70)
        • How-to process topics            (~120)
        • Wellbeing / policy / funding     (~80)

    Adding new topics = extending one of the taxonomy lists — no new code.

Usage:
    python3 scripts/build-nlu-topics.py
    INGEST_MORPHIK=1 MORPHIK_URL=http://localhost:8000 python3 scripts/build-nlu-topics.py

Output:
    ./nlu-topics.json — the full trigger-phrase database
"""

import os
import sys
import json
import time
import urllib.request
import urllib.error

MORPHIK_URL    = os.environ.get("MORPHIK_URL", "http://localhost:8000").rstrip("/")
INGEST_MORPHIK = bool(os.environ.get("INGEST_MORPHIK"))

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUT_PATH     = os.environ.get(
    "OUT_PATH", os.path.join(PROJECT_ROOT, "nlu-topics.json")
)

# ---------------------------------------------------------------------------
# Paraphrase templates
# ---------------------------------------------------------------------------
EN_TMPL = [
    "how do I {X}", "how can I {X}", "where do I {X}", "where can I {X}",
    "what is {X}", "what's {X}", "tell me about {X}", "info on {X}",
    "information about {X}", "help with {X}", "i need help with {X}",
    "what about {X}", "any info on {X}", "explain {X}",
    "can you tell me about {X}", "show me {X}",
    "i want to know about {X}", "{X} please", "{X}?",
]
CY_TMPL = [
    "sut wyf i'n {X}", "sut alla i {X}", "ble ydw i'n {X}",
    "beth yw {X}", "beth am {X}", "dywedwch wrtha i am {X}",
    "gwybodaeth am {X}", "help gyda {X}", "eglurwch {X}",
    "{X} os gwelwch yn dda", "{X}?",
]


def expand(seeds, templates, entities):
    out = list(seeds)
    seen = set(s.lower() for s in seeds)
    for e in entities:
        for t in templates:
            p = t.format(X=e).strip()
            low = p.lower()
            if low not in seen and len(p) > 1:
                seen.add(low)
                out.append(p)
    return out


# ---------------------------------------------------------------------------
# 1. Core hand-curated topics (high-priority, polished canned replies)
# ---------------------------------------------------------------------------
CORE = [
    {
        "id": "crisis", "title": "Crisis / urgent wellbeing",
        "title_cy": "Argyfwng / lles brys",
        "seeds_en": ["i want to hurt myself","i want to end my life",
                     "suicidal thoughts","i can't go on","i need urgent help",
                     "emergency mental health","self harm"],
        "seeds_cy": ["dwi eisiau brifo fy hun","dwi eisiau gorffen fy mywyd"],
        "entities_en": [], "entities_cy": [],
        "keywords": ["suicidal","hurt myself","self-harm","crisis","emergency"],
        "morphik_hint": "",
        "reply_en": "I'm really sorry you're feeling this way. Please reach out to someone who can help right now — call Samaritans on 116 123 (free, 24/7), or text SHOUT to 85258. In immediate danger call 999. UWTSD's Wellbeing Service is also here for you: https://www.uwtsd.ac.uk/student-services/wellbeing/",
        "reply_cy": "Mae'n ddrwg iawn gennyf. Cysylltwch â rhywun sy'n gallu helpu nawr — ffoniwch Samaritans 116 123 (am ddim, 24/7), neu testun SHOUT i 85258. Mewn perygl uniongyrchol ffoniwch 999.",
        "priority": 100,
    },
    {
        "id": "smalltalk-greeting","title": "Greeting","title_cy": "Cyfarchiad",
        "seeds_en": ["hi","hello","hey","good morning","good afternoon",
                     "good evening","hiya","hey there"],
        "seeds_cy": ["helo","shwmae","bore da","prynhawn da","noswaith dda"],
        "entities_en": [], "entities_cy": [],
        "keywords": ["hi","hello","hey","shwmae","helo"],
        "morphik_hint": "",
        "reply_en": "Hi! I'm U-Pal — your UWTSD student-support chatbot. Ask me about applying, courses, fees, accommodation, campus life, wellbeing, anything UWTSD-related.",
        "reply_cy": "Helo! U-Pal ydw i — eich chatbot cymorth myfyrwyr PCYDDS. Gofynnwch am ymgeisio, cyrsiau, ffioedd, llety, bywyd campws, lles, unrhyw beth sy'n ymwneud â PCYDDS.",
        "priority": 10,
    },
    {
        "id": "smalltalk-capabilities","title": "What can you do",
        "title_cy": "Beth allwch chi ei wneud",
        "seeds_en": ["what can you do","help me","what do you know",
                     "what topics can you cover","what are you for"],
        "seeds_cy": ["beth allwch chi ei wneud","helpu fi"],
        "entities_en": [], "entities_cy": [],
        "keywords": ["help","capabilities","topics"],
        "morphik_hint": "",
        "reply_en": "I can answer anything about UWTSD — admissions (UG, PG, international), open days, courses and modules, fees and funding (scholarships, bursaries, loans), accommodation (all campuses), campus life, library, IT, wellbeing, mental health, disability support, careers, placements, Welsh-medium study, graduation — just ask!",
        "reply_cy": "Gallaf ateb unrhyw beth am PCYDDS — derbyniadau, diwrnodau agored, cyrsiau, ffioedd, ysgoloriaethau, benthyciadau, llety, bywyd campws, llyfrgell, TG, lles, iechyd meddwl, anabledd, gyrfaoedd, lleoliadau, astudio trwy'r Gymraeg, graddio — gofynnwch!",
        "priority": 8,
    },
    {
        "id": "smalltalk-thanks","title": "Thanks","title_cy": "Diolch",
        "seeds_en": ["thanks","thank you","cheers","ta","appreciate it","much appreciated"],
        "seeds_cy": ["diolch","diolch yn fawr"],
        "entities_en": [], "entities_cy": [],
        "keywords": ["thanks","thank you","diolch"],
        "morphik_hint": "",
        "reply_en": "You're very welcome! Anything else I can help with?",
        "reply_cy": "Croeso! Unrhyw beth arall y gallaf helpu â hi?",
        "priority": 8,
    },
    {
        "id": "smalltalk-goodbye","title": "Goodbye","title_cy": "Ffarwel",
        "seeds_en": ["bye","goodbye","see you","see ya","cya","later","take care"],
        "seeds_cy": ["hwyl","hwyl fawr","wela i di","da bo ti"],
        "entities_en": [], "entities_cy": [],
        "keywords": ["bye","goodbye","hwyl"],
        "morphik_hint": "",
        "reply_en": "Take care! Come back any time you need a hand with UWTSD matters.",
        "reply_cy": "Cymerwch ofal! Dewch 'nôl unrhyw bryd.",
        "priority": 8,
    },
    {
        "id": "smalltalk-who-are-you","title": "Who are you",
        "title_cy": "Pwy ydych chi",
        "seeds_en": ["who are you","what are you","what's your name","who is this","are you a bot"],
        "seeds_cy": ["pwy ydych chi","beth yw eich enw"],
        "entities_en": [], "entities_cy": [],
        "keywords": ["who","you","bot","name"],
        "morphik_hint": "",
        "reply_en": "I'm U-Pal — a bilingual student-support chatbot for UWTSD. I use the university's public information to help with applying, studying, funding, living, and getting support here.",
        "reply_cy": "U-Pal ydw i — chatbot dwyieithog i gefnogi myfyrwyr PCYDDS.",
        "priority": 7,
    },
    # ── Admissions & applying
    {
        "id": "admissions-undergraduate","title": "Undergraduate admissions",
        "title_cy": "Derbyniadau israddedig",
        "seeds_en": ["how do I apply for undergrad","apply to UWTSD",
                     "UCAS application UWTSD","when is the application deadline"],
        "seeds_cy": ["sut ydw i'n ymgeisio am gwrs israddedig",
                     "ymgeisio i PCYDDS"],
        "entities_en": ["apply","application","admission","UCAS",
                        "register for a course","enrol"],
        "entities_cy": ["ymgeisio","UCAS","cofrestru"],
        "keywords": ["apply","admission","ucas","undergrad","register"],
        "morphik_hint": "UWTSD undergraduate admissions application UCAS process deadlines entry",
        "priority": 10,
    },
    {
        "id": "admissions-postgraduate","title": "Postgraduate admissions",
        "title_cy": "Derbyniadau ôl-raddedig",
        "seeds_en": ["how do I apply for a master's","postgrad application UWTSD",
                     "MA entry process","PhD UWTSD"],
        "seeds_cy": ["sut i ymgeisio am feistr","ymgeisio ôl-raddedig"],
        "entities_en": ["master's","MA","MSc","PhD","postgraduate","doctoral study"],
        "entities_cy": ["meistr","MA","ôl-raddedig","PhD"],
        "keywords": ["postgrad","masters","phd","doctoral","msc","ma"],
        "morphik_hint": "UWTSD postgraduate admissions master's PhD application taught research",
        "priority": 10,
    },
    {
        "id": "admissions-international","title": "International admissions",
        "title_cy": "Derbyniadau rhyngwladol",
        "seeds_en": ["can international students apply","visa requirements for UWTSD",
                     "student visa UK","CAS letter","IELTS score UWTSD"],
        "seeds_cy": ["ymgeisio fel myfyriwr rhyngwladol","gofynion fisa"],
        "entities_en": ["international students","visa","overseas students",
                        "CAS","IELTS","English proficiency"],
        "entities_cy": ["myfyrwyr rhyngwladol","fisa"],
        "keywords": ["international","visa","overseas","ielts","cas"],
        "morphik_hint": "UWTSD international students admissions visa CAS IELTS English language",
        "priority": 10,
    },
    {
        "id": "entry-requirements","title": "Entry requirements",
        "title_cy": "Gofynion mynediad",
        "seeds_en": ["what grades do I need","A-level requirements",
                     "BTEC entry requirements","UCAS points needed"],
        "seeds_cy": ["pa raddau sydd eu hangen","gofynion Safon Uwch"],
        "entities_en": ["entry requirements","A-levels","BTEC","UCAS tariff","grades"],
        "entities_cy": ["gofynion mynediad","Safon Uwch"],
        "keywords": ["grades","a-level","btec","tariff","requirements"],
        "morphik_hint": "UWTSD entry requirements A-level BTEC UCAS tariff points grades",
        "priority": 10,
    },
    {
        "id": "open-days","title": "Open days","title_cy": "Diwrnodau agored",
        "seeds_en": ["when are the open days","book an open day","next open day UWTSD","can I visit the campus"],
        "seeds_cy": ["pryd mae'r diwrnodau agored","archebu diwrnod agored"],
        "entities_en": ["open day","visit campus","campus tour","virtual tour","applicant day"],
        "entities_cy": ["diwrnod agored","ymweliad campws"],
        "keywords": ["open","day","visit","tour","virtual"],
        "morphik_hint": "UWTSD open days campus visit virtual tour booking dates",
        "priority": 9,
    },
    # ── Fees & funding
    {
        "id": "tuition-fees-uk","title": "Tuition fees (UK / home)",
        "title_cy": "Ffioedd dysgu (DU)",
        "seeds_en": ["how much is tuition for home students","UK student fees",
                     "home fees","what does it cost to study here"],
        "seeds_cy": ["faint yw'r ffioedd dysgu","ffioedd myfyrwyr y DU"],
        "entities_en": ["tuition fees","home fees","UK student fees","cost of study","course fee"],
        "entities_cy": ["ffioedd dysgu","cost astudio"],
        "keywords": ["fee","tuition","cost","home","uk"],
        "morphik_hint": "UWTSD tuition fees UK home students undergraduate cost per year",
        "priority": 10,
    },
    {
        "id": "tuition-fees-international","title": "Tuition fees (international)",
        "title_cy": "Ffioedd dysgu (rhyngwladol)",
        "seeds_en": ["international student fees","overseas tuition cost","non-UK fees UWTSD"],
        "seeds_cy": ["ffioedd myfyrwyr rhyngwladol"],
        "entities_en": ["international student fees","overseas fees","non-UK tuition"],
        "entities_cy": ["ffioedd rhyngwladol"],
        "keywords": ["international","overseas","non-uk","fees"],
        "morphik_hint": "UWTSD international overseas tuition fees non-UK students annual cost",
        "priority": 10,
    },
    {
        "id": "scholarships","title": "Scholarships","title_cy": "Ysgoloriaethau",
        "seeds_en": ["what scholarships are available","merit scholarships","international scholarships UWTSD"],
        "seeds_cy": ["pa ysgoloriaethau sydd ar gael"],
        "entities_en": ["scholarships","merit scholarships","international scholarships"],
        "entities_cy": ["ysgoloriaethau"],
        "keywords": ["scholarship","merit","award"],
        "morphik_hint": "UWTSD scholarships merit international apply eligibility amount",
        "priority": 9,
    },
    {
        "id": "bursaries","title": "Bursaries","title_cy": "Bwrsariaethau",
        "seeds_en": ["am I eligible for a bursary","hardship fund UWTSD","care leaver bursary","WGIL grant"],
        "seeds_cy": ["bwrsariaeth UWTSD"],
        "entities_en": ["bursary","bursaries","hardship fund","care leaver bursary","financial support"],
        "entities_cy": ["bwrsariaethau","cefnogaeth ariannol"],
        "keywords": ["bursary","hardship","grant","care leaver"],
        "morphik_hint": "UWTSD bursary hardship fund financial support grant care leaver",
        "priority": 9,
    },
    {
        "id": "student-loans","title": "Student loans","title_cy": "Benthyciadau myfyrwyr",
        "seeds_en": ["how do I get a student loan","student finance Wales","maintenance loan UWTSD","tuition fee loan"],
        "seeds_cy": ["benthyciad myfyriwr","cyllid myfyrwyr Cymru"],
        "entities_en": ["student loan","maintenance loan","tuition fee loan","Student Finance Wales","SFW"],
        "entities_cy": ["benthyciad myfyriwr"],
        "keywords": ["loan","finance","sfw","maintenance","repayment"],
        "morphik_hint": "UWTSD student loan maintenance tuition Student Finance Wales SFW application",
        "priority": 10,
    },
    # ── Accommodation
    {
        "id": "accommodation-overview","title": "Accommodation overview",
        "title_cy": "Trosolwg llety",
        "seeds_en": ["where do students live","UWTSD accommodation options","student halls","book a room"],
        "seeds_cy": ["llety myfyrwyr","ymgeisio am lety"],
        "entities_en": ["accommodation","halls of residence","student housing","on-campus halls","dorm"],
        "entities_cy": ["llety","neuaddau preswyl"],
        "keywords": ["accommodation","halls","housing","dorm","residence"],
        "morphik_hint": "UWTSD student accommodation halls of residence apply book room",
        "priority": 10,
    },
    # ── Wellbeing & support
    {
        "id": "wellbeing","title": "Student wellbeing","title_cy": "Lles myfyrwyr",
        "seeds_en": ["wellbeing support UWTSD","I'm feeling stressed","student support services"],
        "seeds_cy": ["cymorth lles"],
        "entities_en": ["wellbeing service","student support","wellbeing advisor"],
        "entities_cy": ["gwasanaeth lles","cymorth myfyrwyr"],
        "keywords": ["wellbeing","welfare","support","stress"],
        "morphik_hint": "UWTSD wellbeing service support student advisor stress welfare",
        "priority": 10,
    },
    {
        "id": "mental-health","title": "Mental health support",
        "title_cy": "Cymorth iechyd meddwl",
        "seeds_en": ["mental health support UWTSD","counselling service","I need to talk to someone"],
        "seeds_cy": ["cymorth iechyd meddwl","gwasanaeth cwnsela"],
        "entities_en": ["mental health","counselling","therapist","emotional support"],
        "entities_cy": ["iechyd meddwl","cwnsela"],
        "keywords": ["mental","depression","anxiety","counselling","therapy"],
        "morphik_hint": "UWTSD mental health counselling service support therapy emotional",
        "priority": 10,
    },
    {
        "id": "disability-support","title": "Disability support",
        "title_cy": "Cymorth anabledd",
        "seeds_en": ["disability support UWTSD","inclusion and accessibility","DSA application"],
        "seeds_cy": ["cymorth anabledd"],
        "entities_en": ["disability service","accessibility","DSA","disabled students allowance"],
        "entities_cy": ["cymorth anabledd"],
        "keywords": ["disability","accessibility","dsa","inclusion"],
        "morphik_hint": "UWTSD disability support service DSA accessibility inclusion adjustments",
        "priority": 9,
    },
    # ── Library, IT
    {
        "id": "library-services","title": "Library services",
        "title_cy": "Gwasanaethau llyfrgell",
        "seeds_en": ["library opening hours","book a study room","borrow a book UWTSD","e-books and journals"],
        "seeds_cy": ["oriau agor y llyfrgell"],
        "entities_en": ["library","library hours","study room booking","e-books","journals"],
        "entities_cy": ["llyfrgell","ystafell astudio"],
        "keywords": ["library","book","journal","e-book","catalogue"],
        "morphik_hint": "UWTSD library services opening hours borrow books catalogue e-books",
        "priority": 8,
    },
    {
        "id": "it-support","title": "IT support","title_cy": "Cymorth TG",
        "seeds_en": ["reset my password","I can't log in","IT helpdesk UWTSD","student email not working"],
        "seeds_cy": ["ailosod cyfrinair","cymorth TG"],
        "entities_en": ["IT support","helpdesk","password reset","login issues","VPN"],
        "entities_cy": ["cymorth TG","ailosod cyfrinair"],
        "keywords": ["it","password","login","helpdesk","reset"],
        "morphik_hint": "UWTSD IT helpdesk password reset login issues VPN support",
        "priority": 9,
    },
    {
        "id": "wifi-eduroam","title": "Wi-Fi / eduroam","title_cy": "Wi-Fi",
        "seeds_en": ["how do I connect to Wi-Fi","eduroam setup UWTSD","campus internet"],
        "seeds_cy": ["sut i gysylltu â Wi-Fi"],
        "entities_en": ["Wi-Fi","eduroam","campus Wi-Fi","internet"],
        "entities_cy": ["Wi-Fi","eduroam"],
        "keywords": ["wifi","wi-fi","eduroam","internet"],
        "morphik_hint": "UWTSD Wi-Fi eduroam setup connect campus internet",
        "priority": 7,
    },
    {
        "id": "moodle","title": "Moodle / VLE","title_cy": "Moodle",
        "seeds_en": ["how do I access Moodle","Moodle login","submit assignment on Moodle"],
        "seeds_cy": ["mynediad Moodle"],
        "entities_en": ["Moodle","VLE","virtual learning environment"],
        "entities_cy": ["Moodle"],
        "keywords": ["moodle","vle"],
        "morphik_hint": "UWTSD Moodle VLE virtual learning environment login assignment",
        "priority": 8,
    },
    # ── Welsh-medium + specialist
    {
        "id": "welsh-medium-study","title": "Welsh-medium study",
        "title_cy": "Astudio trwy gyfrwng y Gymraeg",
        "seeds_en": ["can I study in Welsh","Welsh-medium courses UWTSD","Coleg Cymraeg Cenedlaethol"],
        "seeds_cy": ["astudio trwy'r Gymraeg","cyrsiau cyfrwng Cymraeg","Coleg Cymraeg Cenedlaethol"],
        "entities_en": ["Welsh-medium study","courses in Welsh","Coleg Cymraeg","Welsh language scholarship"],
        "entities_cy": ["astudio trwy'r Gymraeg","Coleg Cymraeg"],
        "keywords": ["welsh","cymraeg","coleg cymraeg","bilingual"],
        "morphik_hint": "UWTSD Welsh medium study Coleg Cymraeg Cenedlaethol bilingual scholarship",
        "priority": 9,
    },
    {
        "id": "erasmus-study-abroad","title": "Study abroad / Turing",
        "title_cy": "Astudio dramor",
        "seeds_en": ["can I study abroad","Erasmus UWTSD","Turing scheme","year abroad"],
        "seeds_cy": ["astudio dramor"],
        "entities_en": ["study abroad","Erasmus+","Turing scheme","year abroad","international exchange"],
        "entities_cy": ["astudio dramor"],
        "keywords": ["abroad","erasmus","turing","exchange"],
        "morphik_hint": "UWTSD study abroad Erasmus Turing scheme year abroad exchange international",
        "priority": 7,
    },
    {
        "id": "careers","title": "Careers & employability",
        "title_cy": "Gyrfaoedd",
        "seeds_en": ["career support UWTSD","CV help","find a job after graduation","careers advisor"],
        "seeds_cy": ["cymorth gyrfa"],
        "entities_en": ["career service","careers advisor","CV support","job search"],
        "entities_cy": ["gwasanaeth gyrfa"],
        "keywords": ["career","cv","job","employment"],
        "morphik_hint": "UWTSD career service CV advice job search graduate employment advisor",
        "priority": 9,
    },
    {
        "id": "students-union","title": "Students' Union","title_cy": "Undeb y Myfyrwyr",
        "seeds_en": ["what does the Students' Union do","SU events UWTSD","freshers fair"],
        "seeds_cy": ["Undeb y Myfyrwyr","ffair y glas"],
        "entities_en": ["Students' Union","SU","freshers fair","student activities"],
        "entities_cy": ["Undeb y Myfyrwyr"],
        "keywords": ["union","su","freshers"],
        "morphik_hint": "UWTSD Students Union SU events freshers fair activities",
        "priority": 8,
    },
    {
        "id": "contact-general","title": "General contact","title_cy": "Cyswllt",
        "seeds_en": ["how can I contact UWTSD","phone number UWTSD","email address for admissions","who do I speak to"],
        "seeds_cy": ["sut i gysylltu â PCYDDS"],
        "entities_en": ["contact","phone number","email address","enquiry","admissions contact"],
        "entities_cy": ["cysylltu","rhif ffôn"],
        "keywords": ["contact","phone","email","enquiry"],
        "morphik_hint": "UWTSD contact phone email address enquiry admissions general",
        "priority": 9,
    },
]

# ---------------------------------------------------------------------------
# 2. Taxonomies for generation
# ---------------------------------------------------------------------------
# ~70 UWTSD subject areas
SUBJECTS = [
    ("archaeology","Archaeology","Archaeoleg"),
    ("ancient-history","Ancient History","Hanes yr Henfyd"),
    ("anthropology","Anthropology","Anthropoleg"),
    ("art","Art","Celf"),
    ("fine-art","Fine Art","Celfyddyd Gain"),
    ("graphic-design","Graphic Design","Dylunio Graffig"),
    ("illustration","Illustration","Darlunio"),
    ("photography","Photography","Ffotograffiaeth"),
    ("fashion-design","Fashion Design","Dylunio Ffasiwn"),
    ("textiles","Textiles","Tecstilau"),
    ("animation","Animation","Animeiddio"),
    ("film-production","Film Production","Cynhyrchu Ffilm"),
    ("film-studies","Film Studies","Astudiaethau Ffilm"),
    ("games-design","Computer Games Design","Dylunio Gemau Cyfrifiadurol"),
    ("computing","Computing","Cyfrifiadura"),
    ("software-engineering","Software Engineering","Peirianneg Meddalwedd"),
    ("cybersecurity","Cybersecurity","Seiberddiogelwch"),
    ("ai","Artificial Intelligence","Deallusrwydd Artiffisial"),
    ("data-science","Data Science","Gwyddor Data"),
    ("business-management","Business Management","Rheoli Busnes"),
    ("marketing","Marketing","Marchnata"),
    ("accounting","Accounting","Cyfrifeg"),
    ("finance","Finance","Cyllid"),
    ("human-resources","Human Resources","Adnoddau Dynol"),
    ("logistics","Logistics","Logisteg"),
    ("tourism-management","Tourism Management","Rheoli Twristiaeth"),
    ("hospitality","Hospitality","Lletygarwch"),
    ("events-management","Events Management","Rheoli Digwyddiadau"),
    ("culinary-arts","Culinary Arts","Celfyddydau Coginio"),
    ("automotive-engineering","Automotive Engineering","Peirianneg Fodurol"),
    ("motorsport","Motorsport","Chwaraeon Modur"),
    ("civil-engineering","Civil Engineering","Peirianneg Sifil"),
    ("electronic-engineering","Electronic Engineering","Peirianneg Electronig"),
    ("mechanical-engineering","Mechanical Engineering","Peirianneg Fecanyddol"),
    ("architectural-technology","Architectural Technology","Technoleg Bensaernïol"),
    ("built-environment","Built Environment","Yr Amgylchedd Adeiledig"),
    ("construction","Construction","Adeiladu"),
    ("early-childhood","Early Childhood Studies","Astudiaethau Plentyndod Cynnar"),
    ("primary-education","Primary Education","Addysg Gynradd"),
    ("pgce","PGCE","TAR"),
    ("secondary-education","Secondary Education","Addysg Uwchradd"),
    ("welsh-education","Welsh Education","Addysg Gymraeg"),
    ("outdoor-education","Outdoor Education","Addysg Awyr Agored"),
    ("youth-work","Youth and Community Work","Gwaith Ieuenctid"),
    ("social-work","Social Work","Gwaith Cymdeithasol"),
    ("counselling","Counselling","Cwnsela"),
    ("psychology","Psychology","Seicoleg"),
    ("criminology","Criminology","Troseddeg"),
    ("law","Law","Y Gyfraith"),
    ("english-literature","English Literature","Llenyddiaeth Saesneg"),
    ("creative-writing","Creative Writing","Ysgrifennu Creadigol"),
    ("history","History","Hanes"),
    ("welsh-history","Welsh History","Hanes Cymru"),
    ("theology","Theology","Diwinyddiaeth"),
    ("religious-studies","Religious Studies","Astudiaethau Crefyddol"),
    ("islamic-studies","Islamic Studies","Astudiaethau Islamaidd"),
    ("chinese-studies","Chinese Studies","Astudiaethau Tsieina"),
    ("celtic-studies","Celtic Studies","Astudiaethau Celtaidd"),
    ("music","Music","Cerddoriaeth"),
    ("sound-production","Sound Production","Cynhyrchu Sain"),
    ("performing-arts","Performing Arts","Celfyddydau Perfformio"),
    ("acting","Acting","Actio"),
    ("sports-coaching","Sports Coaching","Hyfforddi Chwaraeon"),
    ("sports-therapy","Sports Therapy","Therapi Chwaraeon"),
    ("exercise-science","Exercise Science","Gwyddor Ymarfer"),
    ("nutrition","Nutrition","Maeth"),
    ("nursing","Nursing","Nyrsio"),
    ("health-wellbeing","Health and Wellbeing","Iechyd a Lles"),
    ("public-services","Public Services","Gwasanaethau Cyhoeddus"),
    ("policing","Policing","Plismona"),
    ("paramedic","Paramedic Practice","Ymarfer Parafeddygol"),
]

# Campuses
CAMPUSES = [
    ("swansea","Swansea","Abertawe"),
    ("carmarthen","Carmarthen","Caerfyrddin"),
    ("lampeter","Lampeter","Llambed"),
    ("london","London","Llundain"),
    ("birmingham","Birmingham","Birmingham"),
]

# Services offered across campuses
SERVICES = [
    ("library","library","llyfrgell"),
    ("gym","gym","campfa"),
    ("accommodation","accommodation","llety"),
    ("food","food and dining","bwyd a bwyta"),
    ("transport","transport and parking","trafnidiaeth a pharcio"),
    ("chaplaincy","chaplaincy","caplaniaeth"),
    ("medical","medical centre","canolfan feddygol"),
    ("bookshop","bookshop","siop lyfrau"),
    ("nursery","nursery","meithrinfa"),
    ("prayer","prayer / multifaith room","ystafell weddi"),
    ("shop","campus shop","siop campws"),
    ("postroom","post room","ystafell bost"),
    ("cash-machine","cash machine / ATM","peiriant arian"),
    ("printing","printing service","gwasanaeth argraffu"),
    ("locker","locker rental","rhentu loceri"),
    ("study-space","study space","man astudio"),
]

# Student types
STUDENT_TYPES = [
    ("home-student","home student","myfyriwr y DU"),
    ("international-student","international student","myfyriwr rhyngwladol"),
    ("eu-student","EU student","myfyriwr yr UE"),
    ("mature-student","mature student","myfyriwr hŷn"),
    ("part-time-student","part-time student","myfyriwr rhan-amser"),
    ("distance-student","distance learning student","myfyriwr dysgu o bell"),
    ("disabled-student","disabled student","myfyriwr ag anabledd"),
    ("care-leaver","care leaver","ymadawyr gofal"),
    ("estranged-student","estranged student","myfyriwr dieithrwydd"),
    ("student-parent","student parent","myfyriwr sy'n rhiant"),
    ("carer","student carer","myfyriwr sy'n gofalu"),
    ("veteran","student veteran","myfyriwr cyn-filwr"),
    ("lgbtqia","LGBTQIA+ student","myfyriwr LHDTCRhA+"),
    ("first-in-family","first in family student","myfyriwr cyntaf yn y teulu"),
]

# Common student concerns
CONCERNS = [
    ("funding","funding","ariannu"),
    ("accommodation","accommodation","llety"),
    ("wellbeing","wellbeing","lles"),
    ("transport","transport","trafnidiaeth"),
    ("visa","visa","fisa"),
    ("childcare","childcare","gofal plant"),
    ("community","community and social life","cymuned"),
]

# How-to process topics (each becomes a discrete topic)
HOW_TO = [
    ("how-to-apply-ucas","apply through UCAS","ymgeisio trwy UCAS"),
    ("how-to-apply-direct","apply directly to UWTSD","ymgeisio'n uniongyrchol"),
    ("how-to-track-application","track my application","olrhain fy nghais"),
    ("how-to-accept-offer","accept an offer","derbyn cynnig"),
    ("how-to-decline-offer","decline an offer","gwrthod cynnig"),
    ("how-to-defer-entry","defer entry to next year","gohirio mynediad"),
    ("how-to-change-course","change my course","newid fy nghwrs"),
    ("how-to-withdraw","withdraw from my course","ymddiswyddo o'm cwrs"),
    ("how-to-appeal-grade","appeal a grade","apelio am raddau"),
    ("how-to-request-extension","request an extension","gofyn am estyniad"),
    ("how-to-submit-mitigating","submit mitigating circumstances","cyflwyno amgylchiadau lliniarol"),
    ("how-to-register-gp","register with a GP","cofrestru gyda meddyg"),
    ("how-to-council-tax-exempt","get council tax exemption","cael eithriad treth gyngor"),
    ("how-to-pay-tuition","pay tuition fees","talu ffioedd dysgu"),
    ("how-to-pay-instalments","pay in instalments","talu mewn rhandaliadau"),
    ("how-to-refund","request a fee refund","gofyn am ad-daliad ffioedd"),
    ("how-to-letter-visa","request a visa letter","gofyn am lythyr fisa"),
    ("how-to-letter-bank","request a bank letter","gofyn am lythyr banc"),
    ("how-to-transcript","request an academic transcript","gofyn am drawsgrifiad"),
    ("how-to-confirm-enrolment","confirm my enrolment","cadarnhau fy nghofrestriad"),
    ("how-to-id-card","get a student ID card","cael cerdyn adnabod myfyriwr"),
    ("how-to-replace-id","replace a lost ID card","cael cerdyn newydd"),
    ("how-to-book-library-room","book a library study room","archebu ystafell astudio"),
    ("how-to-use-printers","use the campus printers","defnyddio'r argraffwyr"),
    ("how-to-submit-assignment","submit an assignment online","cyflwyno aseiniad"),
    ("how-to-check-turnitin","check a Turnitin report","gwirio adroddiad Turnitin"),
    ("how-to-access-moodle","access Moodle","cyrchu Moodle"),
    ("how-to-email-tutor","email my personal tutor","anfon e-bost at diwtor"),
    ("how-to-book-tutor","book a tutor meeting","trefnu cyfarfod tiwtor"),
    ("how-to-language-support","get academic English support","cael cymorth Saesneg academaidd"),
    ("how-to-dissertation","choose a dissertation topic","dewis pwnc traethawd hir"),
    ("how-to-find-supervisor","find a project supervisor","dod o hyd i oruchwyliwr"),
    ("how-to-library-card","get a library card","cael cerdyn llyfrgell"),
    ("how-to-renew-books","renew borrowed books","adnewyddu llyfrau"),
    ("how-to-interlibrary","request an interlibrary loan","gofyn am fenthyciad rhyng-lyfrgell"),
    ("how-to-report-harassment","report harassment","riportio aflonyddu"),
    ("how-to-report-bullying","report bullying","riportio bwlio"),
    ("how-to-report-discrimination","report discrimination","riportio gwahaniaethu"),
    ("how-to-counselling","book a counselling session","archebu sesiwn gwnsela"),
    ("how-to-wellbeing-appointment","book a wellbeing appointment","archebu apwyntiad lles"),
    ("how-to-register-disability","register a disability","cofrestru anabledd"),
    ("how-to-dsa-apply","apply for DSA","ymgeisio am DSA"),
    ("how-to-dyslexia-assessment","book a dyslexia assessment","archebu asesiad dyslecsia"),
    ("how-to-join-su","join the Students' Union","ymuno ag Undeb y Myfyrwyr"),
    ("how-to-start-society","start a new society","dechrau cymdeithas newydd"),
    ("how-to-join-club","join a sports club","ymuno â chlwb chwaraeon"),
    ("how-to-book-gym","book a gym session","archebu sesiwn campfa"),
    ("how-to-vote-elections","vote in the SU elections","pleidleisio mewn etholiadau"),
    ("how-to-rep-course","become a course rep","bod yn gynrychiolydd"),
    ("how-to-use-wifi","connect to eduroam Wi-Fi","cysylltu â Wi-Fi eduroam"),
    ("how-to-vpn","set up the UWTSD VPN","sefydlu VPN"),
    ("how-to-office365","access Office 365","cyrchu Office 365"),
    ("how-to-onedrive","use OneDrive for assignments","defnyddio OneDrive"),
    ("how-to-teams","use Microsoft Teams","defnyddio Microsoft Teams"),
    ("how-to-panopto","watch Panopto lectures","gwylio darlithoedd Panopto"),
    ("how-to-read-receipts","view assessment feedback","gweld adborth asesu"),
    ("how-to-grade-appeal-process","understand the grade appeal process","deall y broses apêl"),
    ("how-to-change-personal","update my personal details","diweddaru fy manylion"),
    ("how-to-change-address","change my registered address","newid fy nghyfeiriad"),
    ("how-to-placement-find","find a work placement","dod o hyd i leoliad"),
    ("how-to-placement-register","register a placement","cofrestru lleoliad"),
    ("how-to-cv","write a student CV","ysgrifennu CV"),
    ("how-to-linkedin","set up student LinkedIn","sefydlu LinkedIn"),
    ("how-to-interview-prep","prepare for an interview","paratoi ar gyfer cyfweliad"),
    ("how-to-graduate","graduate from UWTSD","graddio o PCYDDS"),
    ("how-to-book-graduation","book my graduation ceremony","archebu fy seremoni raddio"),
    ("how-to-gown","hire a graduation gown","llogi gwn graddio"),
    ("how-to-graduation-guests","invite graduation guests","gwahodd gwesteion graddio"),
    ("how-to-alumni","join the alumni network","ymuno â'r rhwydwaith alumni"),
    ("how-to-reference-old-student","get a reference after graduation","cael geirda ar ôl graddio"),
    ("how-to-reapply","reapply after withdrawal","ailymgeisio"),
    ("how-to-find-lost-item","find a lost item","dod o hyd i eitem goll"),
    ("how-to-report-lost-card","report a lost student card","riportio cerdyn coll"),
    ("how-to-parking-permit","get a parking permit","cael trwydded barcio"),
    ("how-to-cycle-storage","use cycle storage","defnyddio storfa beiciau"),
    ("how-to-print-credit","top up print credit","ychwanegu credyd argraffu"),
    ("how-to-food-voucher","get a hardship food voucher","cael taleb fwyd"),
    ("how-to-emergency-loan","apply for an emergency loan","ymgeisio am fenthyciad brys"),
    ("how-to-safezone","use the SafeZone app","defnyddio ap SafeZone"),
    ("how-to-report-concern","report a safeguarding concern","adrodd pryder diogelu"),
    ("how-to-feedback","give feedback about a module","rhoi adborth modiwl"),
    ("how-to-complaint","submit a formal complaint","cyflwyno cwyn ffurfiol"),
    ("how-to-sports-team","try out for a sports team","gwneud prawf tîm chwaraeon"),
    ("how-to-mentoring","get a student mentor","cael mentor myfyriwr"),
    ("how-to-be-mentor","become a student mentor","bod yn fentor myfyriwr"),
    ("how-to-job-on-campus","find an on-campus job","dod o hyd i swydd ar y campws"),
    ("how-to-fx-account","open a UK bank account","agor cyfrif banc yn y DU"),
    ("how-to-gp-overseas","register with a GP as an international student","cofrestru gyda meddyg fel myfyriwr rhyngwladol"),
    ("how-to-police-registration","complete police registration","cwblhau cofrestru heddlu"),
    ("how-to-brp-collect","collect my BRP","casglu fy BRP"),
    ("how-to-extend-visa","extend my student visa","ymestyn fy fisa myfyriwr"),
    ("how-to-work-visa","work on my student visa","gweithio ar fy fisa"),
    ("how-to-graduate-visa","apply for the graduate visa","ymgeisio am fisa graddedig"),
    ("how-to-nus-extra","get a NUS Totum card","cael cerdyn NUS Totum"),
    ("how-to-discount","access student discounts","cael gostyngiadau myfyriwr"),
    ("how-to-bus-pass","get a student bus pass","cael tocyn bws"),
    ("how-to-railcard","buy a 16-25 railcard","prynu cerdyn 16-25"),
    ("how-to-exam-timetable","find my exam timetable","dod o hyd i'r amserlen arholiad"),
    ("how-to-special-exam","request exam adjustments","gofyn am addasiadau arholiad"),
    ("how-to-resit","resit a failed module","ailsefyll modiwl"),
    ("how-to-compensate","compensate a failed credit","digolledu credyd"),
    ("how-to-change-language","change module language of delivery","newid iaith y modiwl"),
    ("how-to-welsh-scholarship","apply for a Welsh-medium scholarship","ymgeisio am ysgoloriaeth Cymraeg"),
    ("how-to-erasmus","apply for Erasmus / Turing","ymgeisio am Erasmus / Turing"),
    ("how-to-summer-school","join a summer school","ymuno ag ysgol haf"),
    ("how-to-insurance","get contents insurance in halls","cael yswiriant cynnwys"),
    ("how-to-reset-locker","reset a locker PIN","ailosod PIN locer"),
    ("how-to-flat-swap","swap halls rooms","newid ystafelloedd neuaddau"),
    ("how-to-guest-stay","have a guest stay in halls","cael gwestai yn y neuaddau"),
    ("how-to-interfaith","attend interfaith events","mynychu digwyddiadau rhyng-ffydd"),
    ("how-to-prayer-room","find a prayer room","dod o hyd i ystafell weddi"),
    ("how-to-wellbeing-dog","book a wellbeing dog session","archebu sesiwn ci lles"),
    ("how-to-peer-mentor","find a peer mentor","dod o hyd i fentor cymar"),
    ("how-to-language-partner","find a language exchange partner","dod o hyd i bartner iaith"),
    ("how-to-welsh-lessons","take free Welsh lessons","cymryd gwersi Cymraeg am ddim"),
    ("how-to-skills-workshop","join a study-skills workshop","ymuno â gweithdy sgiliau"),
    ("how-to-booksale","access the second-hand book sale","siop lyfrau ail-law"),
    ("how-to-nightbus","catch the student night bus","dal y bws nos"),
    ("how-to-graduation-photos","order graduation photos","archebu lluniau graddio"),
    ("how-to-cap-gown-return","return my cap and gown","dychwelyd fy nghap a gwn"),
]

# Policies / formal documents
POLICIES = [
    ("policy-attendance","attendance policy","polisi presenoldeb"),
    ("policy-plagiarism","plagiarism / academic misconduct","twyll academaidd"),
    ("policy-extensions","extensions policy","polisi estyniadau"),
    ("policy-resit","resit and referral policy","polisi ailsefyll"),
    ("policy-grading","grading and classification","graddio a dosbarthu"),
    ("policy-feedback","feedback turnaround times","amseroedd troi adborth"),
    ("policy-bullying","anti-bullying policy","polisi gwrth-fwlio"),
    ("policy-harassment","anti-harassment policy","polisi gwrth-aflonyddu"),
    ("policy-discrimination","equality and diversity policy","polisi cydraddoldeb"),
    ("policy-data-protection","data protection and GDPR","diogelu data"),
    ("policy-ip","intellectual property policy","polisi eiddo deallusol"),
    ("policy-alcohol","alcohol and drugs policy","polisi alcohol a chyffuriau"),
    ("policy-smoking","smoking and vaping policy","polisi ysmygu"),
    ("policy-safeguarding","safeguarding policy","polisi diogelu"),
    ("policy-prevent","Prevent duty","dyletswydd Prevent"),
    ("policy-fitness-to-study","fitness to study policy","polisi ffitrwydd i astudio"),
    ("policy-fitness-to-practice","fitness to practice","ffitrwydd i ymarfer"),
    ("policy-dress-code","dress code policy","polisi cod gwisg"),
    ("policy-placement","placement expectations","disgwyliadau lleoliad"),
    ("policy-social-media","social media policy","polisi cyfryngau cymdeithasol"),
    ("policy-refund","fee refund policy","polisi ad-dalu ffioedd"),
    ("policy-deposit","accommodation deposit policy","polisi blaendal llety"),
    ("policy-halls-rules","halls of residence rules","rheolau neuaddau preswyl"),
    ("policy-guest","halls guest policy","polisi gwesteion"),
    ("policy-environment","sustainability policy","polisi cynaliadwyedd"),
    ("policy-travel","student travel policy","polisi teithio"),
    ("policy-pet","pet policy in halls","polisi anifeiliaid anwes"),
    ("policy-noise","noise policy in halls","polisi sŵn"),
    ("policy-fire","fire safety rules","rheolau diogelwch tân"),
    ("policy-internet","acceptable use policy","polisi defnydd derbyniol"),
]

# Wellbeing sub-topics
WELLBEING_SUB = [
    ("wellbeing-anxiety","anxiety support","cymorth pryder"),
    ("wellbeing-depression","depression support","cymorth iselder"),
    ("wellbeing-stress","stress management","rheoli straen"),
    ("wellbeing-homesick","homesickness","hiraeth am gartref"),
    ("wellbeing-loneliness","loneliness","unigrwydd"),
    ("wellbeing-sleep","sleep problems","problemau cysgu"),
    ("wellbeing-eating","eating disorders","anhwylderau bwyta"),
    ("wellbeing-self-harm","self-harm support","cymorth hunan-niweidio"),
    ("wellbeing-bereavement","bereavement","profedigaeth"),
    ("wellbeing-substance","drug and alcohol concerns","pryderon cyffuriau"),
    ("wellbeing-sexual-health","sexual health","iechyd rhywiol"),
    ("wellbeing-domestic-abuse","domestic abuse support","cymorth cam-drin domestig"),
    ("wellbeing-assault","sexual assault support","cymorth ymosodiad rhywiol"),
    ("wellbeing-panic","panic attacks","trawiadau panig"),
    ("wellbeing-phobia","phobias","ffobiâu"),
    ("wellbeing-ocd","OCD support","cymorth OCD"),
    ("wellbeing-ptsd","PTSD support","cymorth PTSD"),
    ("wellbeing-trans","trans student support","cymorth myfyrwyr traws"),
    ("wellbeing-faith","faith and chaplaincy support","cymorth ffydd"),
]

# Funding sub-topics beyond core loans/bursaries
FUNDING_SUB = [
    ("funding-part-time-job","part-time jobs for students","swyddi rhan-amser"),
    ("funding-budget","budgeting advice","cyngor cyllidebu"),
    ("funding-food-bank","student food bank","banc bwyd myfyrwyr"),
    ("funding-travel-grant","travel grant","grant teithio"),
    ("funding-placement-grant","placement year grant","grant lleoliad"),
    ("funding-care-leaver","care leaver funding","ariannu ymadawyr gofal"),
    ("funding-estranged","estranged student funding","ariannu dieithrwydd"),
    ("funding-parent","student parent funding","ariannu rhiant myfyriwr"),
    ("funding-disabled","disabled student allowance","lwfans myfyriwr anabl"),
    ("funding-nhs","NHS bursary","bwrsariaeth GIG"),
    ("funding-teacher","teacher training bursary","bwrsariaeth hyfforddi athrawon"),
    ("funding-postgraduate","postgraduate funding","ariannu ôl-raddedig"),
    ("funding-phd","PhD studentships","ysgoloriaethau PhD"),
]

# ---------------------------------------------------------------------------
# 3. Generators
# ---------------------------------------------------------------------------
def subject_topics():
    """3 sub-topics per subject: course / career / entry-requirements."""
    out = []
    for sid, en, cy in SUBJECTS:
        # Course overview
        out.append({
            "id": f"subject-{sid}",
            "title": f"{en} at UWTSD",
            "title_cy": f"{cy} ym PCYDDS",
            "seeds_en": [f"{en} at UWTSD", f"study {en}", f"{en} degree",
                         f"{en} BA", f"{en} BSc"],
            "seeds_cy": [f"astudio {cy}", f"gradd {cy}", f"{cy} ym PCYDDS"],
            "entities_en": [f"{en}", f"{en} degree", f"{en} course",
                            f"BA {en}", f"BSc {en}"],
            "entities_cy": [f"{cy}", f"gradd {cy}"],
            "keywords": [sid, en.lower()] + en.lower().split(),
            "morphik_hint": f"UWTSD {en} undergraduate postgraduate course degree programme modules",
            "priority": 6,
        })
        # Careers outcome
        out.append({
            "id": f"subject-{sid}-careers",
            "title": f"{en} careers / employability",
            "title_cy": f"Gyrfaoedd {cy}",
            "seeds_en": [f"what jobs can I get with a {en} degree",
                         f"{en} graduate careers",
                         f"is {en} a good degree for jobs"],
            "seeds_cy": [f"gyrfaoedd {cy}", f"swyddi {cy}"],
            "entities_en": [f"{en} careers", f"{en} jobs", f"{en} graduate"],
            "entities_cy": [f"gyrfaoedd {cy}"],
            "keywords": [sid, "careers", "jobs", en.lower()],
            "morphik_hint": f"UWTSD {en} graduate careers employability jobs industry outcomes",
            "priority": 5,
        })
        # Entry requirements per subject
        out.append({
            "id": f"subject-{sid}-entry",
            "title": f"{en} entry requirements",
            "title_cy": f"Gofynion mynediad {cy}",
            "seeds_en": [f"what grades for {en}",
                         f"{en} entry requirements",
                         f"A-levels needed for {en}"],
            "seeds_cy": [f"gofynion mynediad {cy}"],
            "entities_en": [f"{en} entry requirements",
                            f"{en} A-level requirements"],
            "entities_cy": [f"gofynion {cy}"],
            "keywords": [sid, "entry", "grades", en.lower()],
            "morphik_hint": f"UWTSD {en} entry requirements A-level BTEC UCAS tariff grades needed",
            "priority": 5,
        })
    return out


def campus_service_topics():
    """Campus × service grid → per-campus service topics."""
    out = []
    for cid, cen, ccy in CAMPUSES:
        for sid, sen, scy in SERVICES:
            out.append({
                "id": f"campus-{cid}-{sid}",
                "title": f"{cen} {sen}",
                "title_cy": f"{ccy} {scy}",
                "seeds_en": [f"{cen} {sen}",
                             f"{sen} at {cen} campus",
                             f"where is the {sen} in {cen}",
                             f"{cen} campus {sen} location"],
                "seeds_cy": [f"{scy} {ccy}"],
                "entities_en": [f"{cen} {sen}", f"{sen} {cen}"],
                "entities_cy": [f"{scy} {ccy}"],
                "keywords": [cid, sid, cen.lower(), sen.lower()],
                "morphik_hint": f"UWTSD {cen} campus {sen} location opening hours contact",
                "priority": 4,
            })
    return out


def student_type_topics():
    """Student-type × concern grid."""
    out = []
    for tid, ten, tcy in STUDENT_TYPES:
        for cid, cen, ccy in CONCERNS:
            out.append({
                "id": f"student-{tid}-{cid}",
                "title": f"{ten} — {cen}",
                "title_cy": f"{tcy} — {ccy}",
                "seeds_en": [f"{cen} for {ten}s",
                             f"{ten} {cen}",
                             f"as a {ten}, {cen}"],
                "seeds_cy": [f"{ccy} ar gyfer {tcy}"],
                "entities_en": [f"{ten} {cen}"],
                "entities_cy": [f"{tcy} {ccy}"],
                "keywords": [tid, cid, ten.lower(), cen.lower()],
                "morphik_hint": f"UWTSD {ten} {cen} support eligibility process",
                "priority": 4,
            })
    return out


def how_to_topics():
    out = []
    for hid, hen, hcy in HOW_TO:
        out.append({
            "id": hid,
            "title": f"How to {hen}",
            "title_cy": f"Sut i {hcy}",
            "seeds_en": [f"how do I {hen}", f"how to {hen}",
                         f"{hen} UWTSD", f"I want to {hen}"],
            "seeds_cy": [f"sut i {hcy}", f"{hcy} ym PCYDDS"],
            "entities_en": [hen],
            "entities_cy": [hcy],
            "keywords": hid.split("-") + hen.lower().split(),
            "morphik_hint": f"UWTSD how to {hen} process step by step instructions",
            "priority": 6,
        })
    return out


def policy_topics():
    out = []
    for pid, pen, pcy in POLICIES:
        out.append({
            "id": pid,
            "title": pen.capitalize(),
            "title_cy": pcy,
            "seeds_en": [pen, f"UWTSD {pen}", f"what is the {pen}"],
            "seeds_cy": [pcy],
            "entities_en": [pen],
            "entities_cy": [pcy],
            "keywords": pid.split("-") + pen.lower().split(),
            "morphik_hint": f"UWTSD {pen} policy rules student",
            "priority": 5,
        })
    return out


def wellbeing_sub_topics():
    out = []
    for wid, wen, wcy in WELLBEING_SUB:
        out.append({
            "id": wid,
            "title": wen.capitalize(),
            "title_cy": wcy,
            "seeds_en": [wen, f"UWTSD {wen}", f"help with {wen}"],
            "seeds_cy": [wcy, f"cymorth {wcy}"],
            "entities_en": [wen],
            "entities_cy": [wcy],
            "keywords": wid.split("-") + wen.lower().split(),
            "morphik_hint": f"UWTSD {wen} wellbeing support counselling service",
            "priority": 7,
        })
    return out


def funding_sub_topics():
    out = []
    for fid, fen, fcy in FUNDING_SUB:
        out.append({
            "id": fid,
            "title": fen.capitalize(),
            "title_cy": fcy,
            "seeds_en": [fen, f"UWTSD {fen}", f"how do I get {fen}"],
            "seeds_cy": [fcy],
            "entities_en": [fen],
            "entities_cy": [fcy],
            "keywords": fid.split("-") + fen.lower().split(),
            "morphik_hint": f"UWTSD {fen} funding finance support eligibility",
            "priority": 6,
        })
    return out


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build_topic(t):
    return {
        "id": t["id"],
        "title": t["title"],
        "title_cy": t["title_cy"],
        "phrases_en": expand(t.get("seeds_en", []), EN_TMPL,
                             t.get("entities_en", [])),
        "phrases_cy": expand(t.get("seeds_cy", []), CY_TMPL,
                             t.get("entities_cy", [])),
        "keywords": t.get("keywords", []),
        "morphik_hint": t.get("morphik_hint", ""),
        "reply_en": t.get("reply_en"),
        "reply_cy": t.get("reply_cy"),
        "priority": t.get("priority", 5),
    }


def collect_all():
    raw = []
    raw.extend(CORE)
    raw.extend(subject_topics())
    raw.extend(campus_service_topics())
    raw.extend(student_type_topics())
    raw.extend(how_to_topics())
    raw.extend(policy_topics())
    raw.extend(wellbeing_sub_topics())
    raw.extend(funding_sub_topics())
    # De-dupe by id (later wins only if same id — shouldn't happen but safe)
    seen = {}
    for t in raw:
        seen[t["id"]] = t
    return [build_topic(t) for t in seen.values()]


def ingest_topic(topic):
    body = (
        f"UWTSD topic: {topic['title']} ({topic['title_cy']}).\n\n"
        f"Common student phrasings: "
        f"{'; '.join(topic['phrases_en'][:10])}.\n\n"
        f"Welsh: {'; '.join(topic['phrases_cy'][:6])}.\n\n"
        f"Retrieval context: {topic['morphik_hint']}"
    )
    payload = {
        "content": body,
        "metadata": {
            "source": "nlu-topic-database",
            "topic_id": topic["id"],
            "topic_kind": "nlu-intent",
        },
    }
    req = urllib.request.Request(
        f"{MORPHIK_URL}/ingest/text",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json",
                 "User-Agent": "UPal-NLU/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:
        return None


def main():
    print("=" * 64)
    print("  UWTSD NLU topic database builder (600+ topics)")
    print("=" * 64)

    topics = collect_all()
    total_en = sum(len(t["phrases_en"]) for t in topics)
    total_cy = sum(len(t["phrases_cy"]) for t in topics)

    print(f"  Topics       : {len(topics)}")
    print(f"  Phrases EN   : {total_en:,}")
    print(f"  Phrases CY   : {total_cy:,}")
    print(f"  Output       : {OUT_PATH}")
    print(f"  Morphik ingest: {'YES ('+MORPHIK_URL+')' if INGEST_MORPHIK else 'NO'}")
    print()

    doc = {
        "version":       2,
        "generated_by":  "build-nlu-topics.py",
        "topic_count":   len(topics),
        "phrase_counts": {"en": total_en, "cy": total_cy},
        "topics":        topics,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    print(f"  Wrote {OUT_PATH}")

    if INGEST_MORPHIK:
        print()
        print(f"  Ingesting {len(topics)} topic summaries into Morphik...")
        ok, fail = 0, 0
        for i, t in enumerate(topics, 1):
            s = ingest_topic(t)
            if s and 200 <= s < 300:
                ok += 1
            else:
                fail += 1
            if i % 50 == 0:
                print(f"    {i}/{len(topics)} done (ok={ok} fail={fail})")
            time.sleep(0.03)
        print(f"  Ingestion: ok={ok} fail={fail}")

    print()
    print(f"  Done. {len(topics)} topics, "
          f"{total_en + total_cy:,} total trigger phrases.")
    print("=" * 64)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
