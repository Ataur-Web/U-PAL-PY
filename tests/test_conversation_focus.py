"""Tests for conversational focus decay in build_state.

build_state previously concatenated every past user turn with the current
message and ran the topic regexes over the whole string. A topic mentioned
once was therefore never released: after "what courses are available?", every
later turn - including "thanks", "wow" and even "where is the SA1 campus?" -
still carried topic=courses, and the prompt steered the model back to courses
each time.

The distinction the code now draws is between durable profile signals (name,
course of study, level, year, campus), which are read from the whole history,
and transient conversational focus (topic_stack, anchor), which is read from
the current message and inherited only when the message genuinely refers back.
"""
from __future__ import annotations

from app.services.conversation import build_state


COURSE_HISTORY = [
    {"role": "user", "text": "What courses are available?"},
    {"role": "assistant", "text": "We offer Art and Design, Business, Computing."},
]


def test_conversational_turn_carries_no_topic():
    """The reported bug. These must not inherit the previous topic."""
    for msg in ("Thanks", "Wow", "ok cool", "Diolch"):
        state = build_state(msg, COURSE_HISTORY)
        assert state["topic_stack"] == [], f"{msg!r} inherited {state['topic_stack']}"
        assert state["anchor"] is None, f"{msg!r} inherited anchor {state['anchor']}"


def test_new_topic_replaces_previous_one():
    state = build_state("Where is the SA1 campus?", COURSE_HISTORY)
    assert state["topic_stack"] == ["campus"]
    assert state["anchor"] == "campus"


def test_anaphoric_follow_up_still_inherits():
    """Reference resolution must keep working - this is why the history was
    being consulted in the first place."""
    for msg in ("Is it expensive?", "tell me more about it", "are those full time?"):
        state = build_state(msg, COURSE_HISTORY)
        assert state["anchor"] == "courses", f"{msg!r} failed to resolve"


def test_topic_does_not_survive_beyond_the_window():
    """Once the topic has scrolled out of the recent window it is closed, even
    for an anaphoric message."""
    long_history = COURSE_HISTORY + [
        {"role": "user", "text": "Where is the library?"},
        {"role": "assistant", "text": "The library is on the second floor."},
        {"role": "user", "text": "When does graduation happen?"},
        {"role": "assistant", "text": "Graduation is in July."},
    ]
    state = build_state("Is it expensive?", long_history)
    assert state["anchor"] != "courses"


def test_profile_signals_remain_durable():
    """Course of study, level and year describe the student, not the current
    topic, so they must persist across turns."""
    history = [
        {"role": "user", "text": "I'm a second year undergraduate computing student"},
        {"role": "assistant", "text": "Thanks, noted."},
    ]
    state = build_state("Thanks", history)
    assert state["course"] == "computing"
    assert state["level"] == "undergraduate"
    # _YEAR_RE captures the whole phrase, not just the ordinal
    assert state["year"] == "second year"
    # but the transient focus is still cleared
    assert state["topic_stack"] == []


def test_courses_is_a_recognised_topic():
    """It had no signal at all, so course questions produced an empty stack."""
    assert build_state("What courses are available?", [])["topic_stack"] == ["courses"]
    # \b must stop this matching "coursework", which is assessment
    assert build_state("When is my coursework due?", [])["topic_stack"] == ["assessment"]


def test_welsh_soft_mutation_is_recognised():
    """Welsh initial consonants mutate in context: cyrsiau -> gyrsiau after
    'pa'. Matching only the radical form misses the natural phrasing."""
    assert build_state("Pa gyrsiau sydd ar gael?", [])["topic_stack"] == ["courses"]
    assert build_state("Beth yw'r cwrs?", [])["topic_stack"] == ["courses"]
