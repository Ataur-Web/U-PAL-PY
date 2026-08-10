#!/usr/bin/env python3
"""
source url mapping - attaches verified institutional urls to curated knowledge

the scraped corpus already carries the url each passage came from. the two
curated sources (knowledge.json intents and uwtsd-facts.json entries) were
written by hand and had no url, so any source shown to a student from those
layers could not be clicked through to an official page.

this script assigns each curated entry the uwtsd page its content derives
from. every assigned url is validated against the set of urls actually
present in uwtsd-corpus.json, so a url can only be assigned if the crawler
successfully fetched that page. entries with no crawled equivalent are left
with a null url rather than being given a guessed one - an unverifiable
source is worse than an absent one.

usage:
    python scripts/map_source_urls.py          # apply mapping
    python scripts/map_source_urls.py --check  # validate only, write nothing
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "app" / "data"
BASE = "https://www.uwtsd.ac.uk"

# BydTermCymru is the Welsh Government terminology source the bilingual map
# derives from. it is not a uwtsd page, so it is tracked separately from the
# institutional urls and reported separately in the evaluation.
BYDTERMCYMRU_URL = "https://termau.cymru/"

# intent tag -> uwtsd path. conversational intents (greeting, thanks) and
# intents describing systems with no public page (moodle, wifi, printing)
# are mapped to None deliberately.
KNOWLEDGE_URLS: dict[str, str | None] = {
    # conversational - no institutional source exists
    "greeting":                 None,
    "capabilities":             None,
    "goodbye":                  None,
    "thanks":                   None,
    "how_are_you":              None,
    "human_agent":              "/contact-us",

    # admissions
    "admissions_apply":         "/study/how-apply",
    "admissions_requirements":  "/apply/undergraduate/advice-applicants",
    "admissions_deadline":      "/study/how-apply",
    "admissions_international": "/study/how-apply/international-applications",
    "admissions_foundation":    "/apply/undergraduate/applying-foundation-degree",
    "how_to_apply":             "/study/how-apply",
    "undergraduate_courses":    "/apply/undergraduate",
    "postgraduate_courses":     "/apply/postgraduate",

    # courses
    "courses_general":          "/study",
    "courses_computing":        "/stem",
    "courses_nursing":          "/study",
    "courses_education":        "/study",

    # fees and finance
    "fees_tuition":             "/study/fees-and-finance/tuition-fee-schedule",
    "fees_student_loan":        "/study/fees-and-finance/undergraduate-fees/undergraduate-funding-support",
    "fees_scholarships":        "/study/fees-and-finance/undergraduate-fees/undergraduate-funding-support/undergraduate-bursaries",
    "fees_and_finance":         "/study/fees-and-finance",
    "financial_support":        "/study/fees-and-finance/undergraduate-fees/undergraduate-funding-support",

    # it systems - uwtsd serves these behind authentication, not on the
    # public site, so the crawl contains no page for them
    "it_portal":                None,
    "it_moodle":                None,
    "it_helpdesk":              None,
    "it_wifi":                  None,
    "uwtsd_digital_systems":    None,
    "printing":                 None,
    "timetable":                None,
    "enrolment_tasks":          None,
    "results_grades":           "/about/governance-and-management/academic-office",

    # accommodation
    "accommodation_general":    "/study/experience-and-facilities/accommodation",
    "accommodation_cost":       "/study/experience-facilities/accommodation/frequently-asked-questions",

    # campuses
    "campuses_general":         "/campuses",
    "campuses_swansea":         "/swansea",
    "campuses_carmarthen":      "/carmarthen",
    "campuses_lampeter":        "/campuses",
    "campus_locations":         "/campuses",

    # library
    "library":                  "/library",
    "library_swansea":          "/library",
    "library_carmarthen":       "/library",
    "library_lampeter":         "/library",

    # wellbeing and support
    "wellbeing_general":        "/study/experience-and-facilities/student-support-and-wellbeing",
    "wellbeing_crisis":         "/study/experience-and-facilities/student-support-and-wellbeing",
    "wellbeing_disability":     "/study/experience-facilities/student-support-wellbeing/disability-support",
    "student_stress":           "/study/experience-and-facilities/student-support-and-wellbeing",
    "student_services":         "/experience-facilities/student-support-wellbeing/student-hwb",
    "academic_support":         "/study/experience-and-facilities/student-support-and-wellbeing",
    "dissertation_support":     "/study/experience-and-facilities/student-support-and-wellbeing",
    "career_services":          "/study/experience-and-facilities/student-support-and-wellbeing/careers-service",
    "english_language_support": "/study/how-apply/international-applications/english-language-requirements",

    # student life
    "students_union":           "/experience-facilities/students-union",
    "graduation":               "/graduation",
    "welsh_language":           "/study/experience-and-facilities/welsh-opportunities-uwtsd",

    # institutional
    "uwtsd_about":              "/about",
    "contact_general":          "/contact-us",
    "staff_contact_query":      "/contact-us",
}

# fact id -> uwtsd path
FACT_URLS: dict[str, str | None] = {
    "campus-sa1-location":            "/swansea",
    "campus-mount-pleasant":          "/swansea",
    "campus-carmarthen-location":     "/carmarthen",
    "campus-lampeter-location":       "/campuses",
    "campus-cardiff-location":        "/cardiff",
    "fees-general":                   "/study/fees-and-finance",
    "fees-international-undergraduate": "/study/fees-and-finance/tuition-fee-schedule",
    "fees-international-postgraduate": "/study/fees-and-finance/tuition-fee-schedule",
    "fees-uk-undergraduate":          "/study/fees-and-finance/tuition-fee-schedule",
    "fees-postgraduate-home":         "/study/fees-and-finance/tuition-fee-schedule",
    "fees-pgce":                      "/study/fees-and-finance/tuition-fee-schedule",
    "fees-phd":                       "/study/fees-and-finance/tuition-fee-schedule",
    "accommodation-apply":            "/study/experience-and-facilities/accommodation",
    "library-hours":                  "/library",
    "how-to-apply":                   "/study/how-apply",
    "student-finance":                "/study/fees-and-finance/undergraduate-fees/undergraduate-funding-support",
    "financial-hardship":             "/study/fees-and-finance/undergraduate-fees/undergraduate-funding-support",
    "wellbeing-support":              "/study/experience-and-facilities/student-support-and-wellbeing",
    "academic-support":               "/study/experience-and-facilities/student-support-and-wellbeing",
    "wellbeing-crisis":               "/study/experience-and-facilities/student-support-and-wellbeing",
    "students-union-activities":      "/experience-facilities/students-union",
    "it-support":                     None,
    "inter-campus-travel":            "/campuses",
    "personal-tutor":                 "/study/experience-and-facilities/student-support-and-wellbeing",
    "graduation":                     "/graduation",
    "extenuating-circumstances":      "/about/governance-and-management/academic-office",
    "open-day":                       "/open-days",
}


def crawled_urls() -> set[str]:
    """URLs the scraper actually fetched. the validation whitelist."""
    raw = json.loads((DATA / "uwtsd-corpus.json").read_text(encoding="utf-8"))
    return {
        (e.get("url") or e.get("source") or "").rstrip("/")
        for e in raw
        if isinstance(e, dict)
    }


def validate(mapping: dict[str, str | None], allowed: set[str], label: str) -> list[str]:
    """every non-null url must appear in the crawl. returns error strings."""
    errors = []
    for key, path in mapping.items():
        if path is None:
            continue
        full = (BASE + path).rstrip("/")
        if full not in allowed:
            errors.append(f"  {label}[{key}] -> {path}  NOT IN CRAWL")
    return errors


def apply_knowledge(check_only: bool) -> tuple[int, int]:
    path = DATA / "knowledge.json"
    entries = json.loads(path.read_text(encoding="utf-8"))
    mapped = unmapped = 0
    for e in entries:
        tag = e.get("tag")
        if tag not in KNOWLEDGE_URLS:
            print(f"  WARNING: intent '{tag}' has no mapping entry", file=sys.stderr)
            e["source_url"] = None
            unmapped += 1
            continue
        p = KNOWLEDGE_URLS[tag]
        e["source_url"] = (BASE + p) if p else None
        if p:
            mapped += 1
        else:
            unmapped += 1
    if not check_only:
        path.write_text(
            json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return mapped, unmapped


def apply_facts(check_only: bool) -> tuple[int, int]:
    path = DATA / "uwtsd-facts.json"
    entries = json.loads(path.read_text(encoding="utf-8"))
    mapped = unmapped = 0
    for e in entries:
        fid = e.get("id")
        p = FACT_URLS.get(fid)
        e["source_url"] = (BASE + p) if p else None
        if p:
            mapped += 1
        else:
            unmapped += 1
    if not check_only:
        path.write_text(
            json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return mapped, unmapped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="validate only, write nothing")
    args = ap.parse_args()

    allowed = crawled_urls()
    print(f"crawled url whitelist: {len(allowed)} pages\n")

    errors = validate(KNOWLEDGE_URLS, allowed, "knowledge")
    errors += validate(FACT_URLS, allowed, "facts")
    if errors:
        print("VALIDATION FAILED - these urls are not in the crawl:", file=sys.stderr)
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("validation passed: every mapped url exists in the crawl\n")

    km, ku = apply_knowledge(args.check)
    fm, fu = apply_facts(args.check)

    print(f"knowledge intents : {km:3d} mapped, {ku:3d} left null")
    print(f"curated facts     : {fm:3d} mapped, {fu:3d} left null")
    print(f"welsh bootstrap   : all entries -> {BYDTERMCYMRU_URL}")
    if args.check:
        print("\n--check set, no files written")
    else:
        print("\nwrote knowledge.json and uwtsd-facts.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
