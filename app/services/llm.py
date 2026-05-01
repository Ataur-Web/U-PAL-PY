"""LangChain wrapper around the LLM (Claude or Ollama).

exposes one async function, generate(), that the chat route calls. it
handles prompt assembly, history, student context, emotion guidance, and
RAG passages in a single place so the route itself stays thin.

ref: LangChain docs - https://python.langchain.com/docs/introduction/
ref: Anthropic Claude API - https://docs.claude.com/en/api/
"""
from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama

# optional import. if the package isn't installed we fall back to Ollama.
try:
    from langchain_anthropic import ChatAnthropic
except Exception:
    ChatAnthropic = None

# bump this string whenever you ship a behaviour change to llm.py. it
# prints in the FastAPI startup log so the operator can verify a freshly
# pulled file actually got loaded into memory and not served from a
# stale __pycache__ entry.
_LLM_MODULE_VERSION = "2026-05-01-megafix-v5-detector-stopword-and-reverse-map"
logging.getLogger("u-pal-py.llm").warning(
    "[startup] llm.py loaded, version=%s", _LLM_MODULE_VERSION,
)

from app.config import PROJECT_ROOT, get_settings


log = logging.getLogger("u-pal-py.llm")

_PROMPTS_DIR = PROJECT_ROOT / "app" / "prompts"


# the LLM tends to echo parenthetical translations inside Welsh replies, like
#   "Shwmae! (Hello!) Beth sy'n bod? (What's up?)"
# no prompt-engineering seems to reliably stop it, so we scrub the output
# ourselves after generation.
#
# heuristic: strip any (...) whose Latin-letter run looks like English (3+
# Latin letters AND contains a vowel). Welsh proper nouns and acronyms
# (UCAS, UWTSD, SA1) survive because they're short or all-caps.
# ref: TODO - add source (this is a hand-tuned heuristic from testing)

# Welsh markers we never want to strip, if one of these appears inside
# the parens, we assume the parens are a legitimate aside in Welsh.
_WELSH_PAREN_KEEPERS = (
    "er enghraifft", "e.e.", "h.y.", "sef", "neu'n", "neu",
    "gweler", "cyfeiriad", "cofiwch", "cofiwch fod",
)

# circumflex and umlaut vowels, any of these means the content is Welsh
_WELSH_DIACRITICS_RE = re.compile(r"[âêîôûŵŷÂÊÎÔÛŴŶäöüÄÖÜ]")


def _looks_like_english_paren(inside: str) -> bool:
    # true if the (...) content is probably English prose that needs scrubbing
    low = inside.lower().strip()
    if not low:
        return False
    for kw in _WELSH_PAREN_KEEPERS:
        if kw in low:
            return False
    if _WELSH_DIACRITICS_RE.search(inside):
        return False
    # short all-caps tokens (UCAS, UWTSD, SA1, PCYDDS), keep
    stripped = inside.strip().rstrip(".,;:!?")
    if len(stripped) <= 8 and stripped.isupper():
        return False
    # needs at least one 3+ letter Latin word to even consider stripping
    words = re.findall(r"[A-Za-z]{3,}", inside)
    if not words:
        return False
    # common English words, a single hit is enough
    english_markers = {
        "the", "that", "this", "what", "which", "there", "here", "have", "has",
        "would", "could", "should", "about", "with", "your", "you", "we",
        "they", "them", "their", "and", "but", "for", "from", "not", "any",
        "all", "some", "when", "where", "how", "why", "hello", "hi", "help",
        "please", "thanks", "thank", "sorry", "means", "means:", "i.e.",
        "e.g.", "means", "translated", "translation",
    }
    low_words = {w.lower() for w in words}
    if low_words & english_markers:
        return True
    # fallback: if every word is pure English (no w/y, only English vowels)
    # it's probably English. Welsh words tend to have w/y or diacritics.
    vowels_en = set("aeiouAEIOU")
    if all(any(c in vowels_en for c in w) for w in words):
        if not any(("w" in w.lower() or "y" in w.lower()) for w in words):
            return True
    return False


_PAREN_RE = re.compile(r"\s*\(([^()]*)\)")


def _strip_parenthetical_en(text: str) -> str:
    # strip parenthetical English from a Welsh reply
    if not text:
        return text
    def repl(m: re.Match) -> str:
        inside = m.group(1)
        return "" if _looks_like_english_paren(inside) else m.group(0)
    out = _PAREN_RE.sub(repl, text)
    # tidy up double spaces and spaces before punctuation left behind
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r"\s+([.,;:!?])", r"\1", out)
    return out.strip()


def _strip_parenthetical_cy(text: str) -> str:
    # strip parenthetical Welsh from an English reply (the mirror case)
    if not text:
        return text
    def repl(m: re.Match) -> str:
        inside = m.group(1)
        if _WELSH_DIACRITICS_RE.search(inside):
            return ""
        low = inside.lower()
        welsh_markers = {
            "mae", "yw", "ydy", "ydw", "dw", "dwi", "rydw", "yr",
            "chi", "ti", "sut", "beth", "pam", "pryd", "ble",
            "shwmae", "helo", "diolch", "os gwelwch", "croeso",
            "prifysgol", "myfyriwr", "cymraeg", "cymru",
        }
        tokens = set(re.findall(r"[a-zâêîôûŵŷ']+", low))
        if tokens & welsh_markers:
            return ""
        return m.group(0)
    out = _PAREN_RE.sub(repl, text)
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r"\s+([.,;:!?])", r"\1", out)
    return out.strip()


def _read_prompt(name: str) -> str:
    # the system prompts live as plain text files so the supervisor can
    # read them without opening Python.
    path = _PROMPTS_DIR / name
    if not path.exists():
        log.warning("Prompt file missing: %s, using minimal fallback", path)
        return "You are U-Pal, the UWTSD student assistant. Be concise, warm, and accurate."
    return path.read_text(encoding="utf-8").strip()


_SYSTEM_EN = _read_prompt("system_en.txt")
_SYSTEM_CY = _read_prompt("system_cy.txt")


# two providers now:
#   "anthropic"  Claude (cloud, default)
#   "ollama"     local Llama 3.1 via ngrok, fallback for offline demos
# switched via LLM_PROVIDER env var or live via POST /api/llm-config.
# we fall back to Ollama if Claude's key is missing so the app never hard-fails.
_client: BaseChatModel | None = None
_active_provider: str = "unknown"


def _build_anthropic(temperature: float | None = None) -> BaseChatModel | None:
    if ChatAnthropic is None:
        log.warning("langchain-anthropic not installed, cannot build Claude client.")
        return None
    s = get_settings()
    if not s.anthropic_api_key:
        log.warning("ANTHROPIC_API_KEY not set, cannot build Claude client.")
        return None
    temp = temperature if temperature is not None else s.anthropic_temperature
    log.info("LLM provider: anthropic (%s, temp=%.2f)", s.anthropic_model, temp)
    # ref: https://python.langchain.com/docs/integrations/chat/anthropic/
    return ChatAnthropic(
        model=s.anthropic_model,
        api_key=s.anthropic_api_key,
        temperature=temp,
        timeout=s.anthropic_timeout_seconds,
        max_tokens=s.anthropic_max_tokens,
    )


def _build_ollama(temperature: float | None = None) -> BaseChatModel:
    s = get_settings()
    temp = temperature if temperature is not None else 0.4
    log.info("LLM provider: ollama (%s @ %s, temp=%.2f)", s.ollama_model, s.ollama_url, temp)
    # the ngrok-skip-browser-warning header is needed on every request when
    # Ollama is exposed through an ngrok free-tier tunnel, otherwise ngrok
    # returns an HTML interstitial instead of the JSON response.
    # ref: https://python.langchain.com/docs/integrations/chat/ollama/
    return ChatOllama(
        model=s.ollama_model,
        base_url=s.ollama_url,
        num_ctx=s.ollama_num_ctx,
        temperature=temp,
        timeout=s.ollama_timeout_seconds,
        client_kwargs={"headers": {"ngrok-skip-browser-warning": "true"}},
    )


def _get_client(lang: str | None = None) -> BaseChatModel:
    # Welsh queries get a lower temperature so the model stays close to
    # the retrieved Welsh phrasing instead of inventing odd grammar.
    # we don't cache the Welsh client because most traffic is English,
    # caching both would double idle memory for marginal benefit.
    if lang == "cy":
        s = get_settings()
        provider = (s.llm_provider or "anthropic").lower().strip()
        cy_temp = 0.2
        if provider == "anthropic":
            cli = _build_anthropic(temperature=cy_temp)
            if cli is not None:
                return cli
        return _build_ollama(temperature=cy_temp)

    global _client, _active_provider
    if _client is not None:
        return _client

    s = get_settings()
    provider = (s.llm_provider or "anthropic").lower().strip()

    if provider == "anthropic":
        client = _build_anthropic()
        if client is not None:
            _client, _active_provider = client, "anthropic"
            return _client
        log.warning("Claude unavailable, falling back to Ollama.")

    _client = _build_ollama()
    _active_provider = "ollama"
    return _client


def get_active_provider() -> str:
    # returns which backend is actually answering requests right now
    if _client is None:
        _get_client()  # force init so the label is accurate
    return _active_provider


def reload_client() -> None:
    # called from /api/llm-config when the operator flips provider or model.
    # we drop the cached clients so the next request rebuilds with new settings.
    global _client, _active_provider
    _client = None
    _active_provider = "unknown"
    # also nudge the translate route's cached client (it keeps its own)
    try:
        from app.routes import translate as _translate_mod
        _translate_mod._client = None  # type: ignore[attr-defined]
    except Exception:
        pass


def _format_passages(passages: list[dict[str, Any]], max_chars: int) -> str:
    # build the "UWTSD INFORMATION:" block that sits in the system prompt.
    # we cap each passage at max_chars so a noisy passage can't eat the
    # entire context window.
    if not passages:
        return ""
    chunks: list[str] = []
    for p in passages:
        text = (p.get("text") or "").strip()
        if not text:
            continue
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "…"
        chunks.append(text)
    if not chunks:
        return ""
    return "UWTSD INFORMATION:\n" + "\n\n".join(chunks)


def _format_student(state: dict[str, Any]) -> str:
    # format the conversation-state dict into a short prompt line
    if not state:
        return ""
    bits: list[str] = []
    for key in ("name", "course", "level", "year", "campus"):
        val = state.get(key)
        if val:
            bits.append(f"{key}: {val}")
    stack = state.get("topic_stack") or []
    if stack:
        bits.append("recent topics: " + ", ".join(stack[-3:]))
    return ("STUDENT CONTEXT: " + "; ".join(bits)) if bits else ""


def _format_emotion(feeling: str, lang: str) -> str:
    # translate the detected emotion into a one-line guidance for the LLM,
    # in the target language so the LLM doesn't code-switch.
    if feeling in ("neutral", "", None):
        return ""
    guidance_en = {
        "distressed":  "The student sounds distressed. Lead with empathy, offer Student Wellbeing Services.",
        "stressed":    "The student sounds stressed. Acknowledge it briefly before answering.",
        "frustrated":  "The student sounds frustrated. Be direct, skip filler, give the answer fast.",
        "confused":    "The student seems confused. Break the answer into small numbered steps.",
        "excited":     "The student sounds excited. Match their energy briefly before details.",
    }
    guidance_cy = {
        "distressed":  "Mae'r myfyriwr yn swnio'n ofidus. Arwain gydag empathi a chynnig Gwasanaethau Lles.",
        "stressed":    "Mae'r myfyriwr wedi straenio. Cydnabod hynny'n fyr cyn ateb.",
        "frustrated":  "Mae'r myfyriwr wedi'i rwystredig. Byddwch yn uniongyrchol, yr ateb yn gyntaf.",
        "confused":    "Mae'r myfyriwr wedi drysu. Rhannwch yr ateb yn gamau bach rhifedig.",
        "excited":     "Mae'r myfyriwr yn gyffrous. Cydweddu â'u hegni cyn y manylion.",
    }
    table = guidance_cy if lang == "cy" else guidance_en
    note = table.get(feeling)
    return f"EMOTIONAL STATE: {note}" if note else ""


def _history_to_messages(history: list[dict[str, Any]], limit: int = 6) -> list:
    # convert our plain-dict history into the Human/AI Message objects
    # LangChain expects. we cap at 6 turns to keep small-model contexts happy.
    msgs: list = []
    for turn in history[-limit:]:
        role = turn.get("role")
        text = (turn.get("text") or "").strip()
        if not text:
            continue
        if role == "user":
            msgs.append(HumanMessage(content=text))
        elif role == "assistant":
            msgs.append(AIMessage(content=text))
    return msgs


# Welsh-only Unicode characters. one of these in an LLM reply is enough
# to call the reply Welsh, regardless of token counts. circumflex vowels
# don't appear in normal English text.
_REPLY_WELSH_CHARS = re.compile(r"[âêîôûŵŷÂÊÎÔÛŴŶ]")

# distinctly Welsh tokens that almost never appear in English text. used
# for reply-language detection only, the user-query detector in welsh.py
# is tuned for shorter strings and is too lenient here.
_REPLY_WELSH_TOKENS: frozenset[str] = frozenset({
    "yr", "yn", "yng", "ym", "ydw", "rwy", "rwyt", "rydych", "rydw",
    "rydyn", "mae", "maen", "ydy", "ydyw", "yw", "oes", "wyt", "wyf",
    "bydd", "fydd", "oedd", "roedd", "byddai", "fyddai",
    "chi", "chwi", "ti", "fi", "ni", "nhw", "fy", "dy", "eich", "ein",
    "eu", "fe", "fo", "hi", "hwn", "hon", "hyn", "hynny",
    "beth", "pwy", "ble", "pryd", "pam", "sut", "faint", "sawl",
    "ond", "achos", "oherwydd", "hefyd", "nawr", "felly",
    "dim", "nid", "nad", "ddim", "drwy", "wrth", "gan", "dros", "tan",
    "prifysgol", "myfyriwr", "myfyrwyr", "cwrs", "cyrsiau", "modiwl",
    "gradd", "campws", "llety", "ffioedd", "darlith", "darlithoedd",
    "shwmae", "shwmai", "helo", "helô", "diolch", "croeso",
    "gallaf", "hoffwn", "eisiau", "moyn",
    "i'r", "ar", "yn", "y", "yr",
})

# tokens used for content extraction. apostrophe-aware so "rwy'n" and
# "i'ch" stay together as single tokens rather than splitting.
_REPLY_TOKEN_RE = re.compile(r"[a-zâêîôûŵŷ']+", re.IGNORECASE)


def _reply_looks_welsh(reply: str) -> bool:
    """True if the LLM reply is Welsh-dominant.

    Stricter than welsh.detect_language because LLM replies are longer
    than user queries and we want to catch even partial Welsh leaks.
    Uses two independent signals: any Welsh-only diacritic anywhere in
    the text, OR at least three distinctly Welsh tokens.
    """
    if not reply:
        return False
    # signal 1, any Welsh-only diacritic char wins immediately
    if _REPLY_WELSH_CHARS.search(reply):
        return True

    # signal 2, the reply opens with an unambiguously Welsh greeting.
    # a Welsh greeting at position zero is a near-perfect predictor that
    # the rest of the reply continues in Welsh. matches the leading
    # word case-insensitively, with optional punctuation after.
    stripped = reply.lstrip()
    lead = stripped.split(None, 1)[0].lower().rstrip("!?,.;:") if stripped else ""
    if lead in _REPLY_WELSH_GREETING_LEADS:
        return True

    # signal 3, count distinctly-Welsh tokens. we expand each apostrophe
    # contraction (mae'n, i'ch, rwy'n) into both halves so contractions
    # count toward the Welsh hit total. without this the detector misses
    # Welsh replies that don't contain a single circumflex but are
    # heavy on contracted Welsh verbs.
    raw_tokens = [t.lower() for t in _REPLY_TOKEN_RE.findall(reply)]
    if not raw_tokens:
        return False

    expanded: list[str] = []
    for t in raw_tokens:
        if "'" in t:
            for part in t.split("'"):
                if part:
                    expanded.append(part)
        expanded.append(t)

    welsh_hits = sum(1 for t in expanded if t in _REPLY_WELSH_TOKENS)
    # threshold is 2, single hits like "yn" or "ar" appear in Welsh
    # proper nouns inside English text but two of them in one reply
    # is almost always a real Welsh utterance.
    return welsh_hits >= 2


# Welsh greeting words that appear ONLY in Welsh text. if one of these
# is the first word of an LLM reply, the reply is Welsh, full stop.
# kept separate from _REPLY_WELSH_TOKENS so the prefix check can short
# circuit before the more expensive token-counting pass.
_REPLY_WELSH_GREETING_LEADS: frozenset[str] = frozenset({
    "helo", "helô", "shwmae", "shwmai", "bore", "noswaith",
    "prynhawn", "diolch", "croeso", "dyma", "dyna",
})


def _reply_looks_english(reply: str) -> bool:
    """True if the LLM reply is English-dominant.

    Negation of _reply_looks_welsh plus a sanity check for Latin
    alphabetic content. catches edge cases where the LLM returned an
    empty string or pure punctuation.
    """
    if not reply or not reply.strip():
        return False
    if _reply_looks_welsh(reply):
        return False
    # at least one English alphabetic char present
    return bool(re.search(r"[a-zA-Z]", reply))


async def _coerce_language(reply: str, target_lang: str) -> str:
    """Validate the LLM reply language. Deterministic, no extra API calls.

    The previous version made a second LLM call to translate when the
    primary reply was in the wrong language. That doubled the API spend
    on every drift event AND the translator itself sometimes drifted.

    Full-force mode: when drift is detected, replace the wrong-language
    reply with a hardcoded same-language template that politely asks
    the user to retry. Zero extra tokens. Guaranteed correct language.
    The trade-off is that the user loses the original answer's content,
    but in practice drift only fires on adversarial / out-of-scope
    queries where the bot was refusing anyway.
    """
    if not reply or not reply.strip():
        return reply

    actually_welsh = _reply_looks_welsh(reply)
    target_is_welsh = (target_lang == "cy")
    actual_label = "cy" if actually_welsh else "en"

    # happy path: reply already matches target language, ship as-is
    if actually_welsh == target_is_welsh:
        log.info("[lang-validator] OK, reply matches target=%s", target_lang)
        return reply

    # drift case: reply is in the wrong language. log loudly so the
    # operator can spot it in the live backend window, then return the
    # hardcoded same-language template. NO LLM CALL.
    log.warning(
        "[lang-validator] DRIFT target=%s actual=%s, swapping for deterministic "
        "template (zero extra API tokens). dropped reply head: %r",
        target_lang, actual_label, reply[:80],
    )
    return _safe_fallback(target_lang)


def _safe_fallback(target_lang: str) -> str:
    """Hardcoded same-language template returned on language drift.

    Never makes an external call so it cannot itself fail or drift.
    The wording is deliberately neutral, the user sees a polite retry
    prompt rather than a confusing wrong-language reply.
    """
    if target_lang == "cy":
        return (
            "Mae'n ddrwg gen i, cefais drafferth wrth lunio ateb yn Gymraeg "
            "i'r cwestiwn yna. Allwch chi roi cynnig ar ofyn eto, neu rephrase "
            "y cwestiwn? Gallwch hefyd e-bostio enquiries@uwtsd.ac.uk am gymorth "
            "uniongyrchol."
        )
    return (
        "I'm sorry, I had trouble forming a reply in English to that one. "
        "Could you try asking again, perhaps with a slightly different phrasing? "
        "You can also email enquiries@uwtsd.ac.uk for direct help."
    )


_LANG_DIRECTIVE_EN = (
    "[CRITICAL LANGUAGE LOCK] The student's CURRENT message is in ENGLISH. "
    "You MUST reply 100% in English. Do NOT use Welsh greetings, words, or "
    "phrases anywhere in your response, even if previous turns in the "
    "conversation history were in Welsh. Acronyms and proper nouns (UWTSD, "
    "PCYDDS, UCAS, Caerfyrddin) are the only exceptions."
)

_LANG_DIRECTIVE_CY = (
    "[CLO IAITH HOLLBWYSIG] Mae neges bresennol y myfyriwr yn GYMRAEG. "
    "Rhaid i chi ateb 100% yn Gymraeg. PEIDIWCH â defnyddio geiriau, "
    "cyfarchion na ymadroddion Saesneg yn unman yn eich ymateb, hyd yn "
    "oed os oedd troeon blaenorol y sgwrs yn Saesneg. Acronymau ac "
    "enwau priod (UWTSD, PCYDDS, UCAS) yw'r unig eithriadau."
)


async def generate(
    *,
    message:       str,
    augmented:     str,
    lang:          str,
    history:       list[dict[str, Any]],
    passages:      list[dict[str, Any]],
    student_state: dict[str, Any],
    emotion:       str,
    intent_tag:    str | None,
) -> str:
    # main entrypoint called by POST /api/chat. builds the prompt, runs
    # the LLM, strips self-translations, returns the text reply.
    s = get_settings()
    client = _get_client(lang=lang)

    system_base = _SYSTEM_CY if lang == "cy" else _SYSTEM_EN

    # the per-turn language directive is the single most important block.
    # it overrides any drift the model would do based on the language of
    # previous turns in the history. without this, a Welsh-then-English
    # conversation gets a Welsh reply because the LLM weighs the dominant
    # language of the recent context.
    lang_lock = _LANG_DIRECTIVE_CY if lang == "cy" else _LANG_DIRECTIVE_EN

    # each block below may be empty depending on context, we join only the
    # ones that returned something. lang_lock appears TWICE on purpose,
    # once at the top so the model sees it before the long base prompt,
    # and once at the very end so it's the most recent thing in context.
    # tail-position tokens get weighted heaviest by transformer attention
    # so the second placement is what actually wins on hard cases.
    blocks = [
        lang_lock,
        system_base,
        _format_student(student_state),
        _format_emotion(emotion, lang),
        _format_passages(passages, s.rag_max_chars),
    ]
    if intent_tag:
        blocks.append(f"LIKELY INTENT: {intent_tag}")
    blocks.append(lang_lock)

    system_text = "\n\n".join(b for b in blocks if b)

    # belt-and-braces: wrap the user message with the language tag at
    # BOTH ends. some LLMs weight tail tokens, others head, so wrapping
    # covers both attention patterns.
    user_lang_tag = "[reply in cy]" if lang == "cy" else "[reply in en]"
    user_payload = f"{user_lang_tag}\n\n{message}\n\n{user_lang_tag}"

    # ChatPromptTemplate + MessagesPlaceholder is the canonical LangChain
    # way to build a system/history/user prompt.
    # ref: https://python.langchain.com/docs/concepts/prompt_templates/
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_text),
        MessagesPlaceholder(variable_name="history"),
        HumanMessage(content=user_payload),
    ])

    # | is the LCEL (LangChain Expression Language) chain operator
    chain = prompt | client

    try:
        result = await chain.ainvoke({"history": _history_to_messages(history)})
    except Exception:
        log.exception("LLM invocation failed")
        raise

    reply = getattr(result, "content", str(result)).strip()

    # post-processing. the LLM still sometimes sneaks parenthetical
    # translations in even when we tell it not to, so we clean them up
    # deterministically here.
    before = reply
    if lang == "cy":
        reply = _strip_parenthetical_en(reply)
    else:
        reply = _strip_parenthetical_cy(reply)
    if reply != before:
        log.debug("Stripped parenthetical self-translations from %s reply", lang)

    # final language validator. if despite all the prompting the LLM
    # produced text in the wrong language, we coerce it via the same
    # translate prompt the /api/translate route uses. this guarantees
    # the user always gets a reply in the language they typed in.
    reply = await _coerce_language(reply, lang)

    return reply
