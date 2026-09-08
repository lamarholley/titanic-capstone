"""
Natural-language -> model-feature parsing for the Titanic survival
interface.

Two parsers are provided behind the same interface:

- ``parse_with_llm``: asks the configured LLM (Nebius AI Studio / OpenAI)
  to extract structured fields from free text, returned as JSON. This is
  the primary path used by the app whenever an API key is configured.
- ``parse_with_rules``: a deterministic, dependency-free regex parser
  used whenever no LLM key is configured, and used directly by the unit
  tests so they never require network access or a paid API key.

Both return a ``ParsedQuery`` with the same shape, so the rest of the
pipeline (validation, defaulting, prediction) doesn't care which one
produced it.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

REQUIRED_FIELDS = ["pclass", "sex", "age"]
OPTIONAL_FIELDS = ["sibsp", "parch", "fare", "embarked"]
ALL_PARSED_FIELDS = REQUIRED_FIELDS + OPTIONAL_FIELDS

VALID_SEX = {"male", "female"}
VALID_PCLASS = {1, 2, 3}
VALID_EMBARKED = {"C", "Q", "S"}

SYSTEM_PROMPT = """You extract structured passenger information from a natural language \
question about whether someone would have survived the Titanic disaster.

Return ONLY a JSON object with these keys:
- "pclass": integer 1, 2, or 3 (ticket/passenger class; 1=upper/first, 2=middle, 3=lower/third/steerage), or null if not stated or not inferable
- "sex": "male" or "female", or null if not stated
- "age": number (years), or null if not stated
- "sibsp": integer, number of siblings/spouses aboard (default 0 if travelling info suggests alone, null if genuinely unclear)
- "parch": integer, number of parents/children aboard (default 0 if travelling info suggests alone, null if genuinely unclear)
- "fare": number (ticket price in 1912 British pounds), or null if not stated
- "embarked": "C" (Cherbourg), "Q" (Queenstown), or "S" (Southampton), or null if not stated
- "out_of_scope": true if the user's question is NOT about predicting a Titanic passenger's survival (e.g. general chit-chat, unrelated questions), else false
- "ambiguous_fields": array of field names above that were mentioned but are contradictory or too vague to resolve confidently (do NOT include fields that were simply never mentioned -- those are just null)

Only extract what is explicitly stated or very directly implied (e.g. "boy" implies male and young). Do not invent values. Respond with JSON only, no other text."""


@dataclass
class ParsedQuery:
    pclass: Optional[int] = None
    sex: Optional[str] = None
    age: Optional[float] = None
    sibsp: Optional[int] = None
    parch: Optional[int] = None
    fare: Optional[float] = None
    embarked: Optional[str] = None
    out_of_scope: bool = False
    ambiguous_fields: list = field(default_factory=list)
    source: str = "rules"  # "rules" or "llm"

    def as_dict(self) -> dict:
        return {
            "pclass": self.pclass,
            "sex": self.sex,
            "age": self.age,
            "sibsp": self.sibsp,
            "parch": self.parch,
            "fare": self.fare,
            "embarked": self.embarked,
            "out_of_scope": self.out_of_scope,
            "ambiguous_fields": list(self.ambiguous_fields),
            "source": self.source,
        }


def parse_with_llm(text: str, client, model: str) -> ParsedQuery:
    """Extract fields via an LLM chat completion in JSON mode."""
    from src.llm_client import chat_json

    raw = chat_json(client, model, SYSTEM_PROMPT, text)
    data = json.loads(raw)
    return _from_llm_json(data)


def _from_llm_json(data: dict) -> ParsedQuery:
    def clean_int(v):
        try:
            return int(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    def clean_float(v):
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    sex = data.get("sex")
    sex = sex.lower() if isinstance(sex, str) and sex.lower() in VALID_SEX else None

    embarked = data.get("embarked")
    embarked = (
        embarked.upper()
        if isinstance(embarked, str) and embarked.upper() in VALID_EMBARKED
        else None
    )

    pclass = clean_int(data.get("pclass"))
    if pclass not in VALID_PCLASS:
        pclass = None

    return ParsedQuery(
        pclass=pclass,
        sex=sex,
        age=clean_float(data.get("age")),
        sibsp=clean_int(data.get("sibsp")),
        parch=clean_int(data.get("parch")),
        fare=clean_float(data.get("fare")),
        embarked=embarked,
        out_of_scope=bool(data.get("out_of_scope", False)),
        ambiguous_fields=list(data.get("ambiguous_fields") or []),
        source="llm",
    )


# ---------------------------------------------------------------------------
# Rule-based fallback parser (no external dependencies, no network calls)
# ---------------------------------------------------------------------------

_OUT_OF_SCOPE_HINTS = [
    "weather", "capital of", "recipe", "stock price", "joke", "poem",
    "translate", "who won", "what time", "president", "define ",
]

_TITANIC_HINTS = [
    "titanic", "survive", "survival", "passenger", "sink", "sank",
    "lifeboat", "iceberg", "board", "aboard", "class", "fare", "embark",
]

_PORT_MAP = {
    "southampton": "S",
    "cherbourg": "C",
    "queenstown": "Q",
    "cobh": "Q",
}


def _detect_out_of_scope(text_lower: str) -> bool:
    has_titanic_context = any(h in text_lower for h in _TITANIC_HINTS)
    has_out_of_scope_hint = any(h in text_lower for h in _OUT_OF_SCOPE_HINTS)
    if has_out_of_scope_hint and not has_titanic_context:
        return True
    # A message with none of our domain vocabulary AND no extractable
    # passenger attributes at all is very likely off-topic small talk.
    return False


# "Strong" terms directly describe the subject. Relational words like
# husband/wife describe a companion, not the subject, so they are
# deliberately excluded -- "a woman travelling with her husband" must
# resolve to female, not ambiguous. Pronouns are a weaker fallback signal
# used only when no strong term is present.
_STRONG_FEMALE = re.compile(r"\b(female|woman|women|girl|lady|mrs\.?|miss)\b")
_STRONG_MALE = re.compile(r"\b(male|man|men|boy|gentleman|mr\.?)\b")
_WEAK_FEMALE = re.compile(r"\b(she|her|hers)\b")
_WEAK_MALE = re.compile(r"\b(he|his|him)\b")


def _extract_sex(text_lower: str) -> Optional[str]:
    # Word-boundary regex (not substring checks) so "woman" doesn't
    # falsely match a bare "man" term, etc.
    strong_female = bool(_STRONG_FEMALE.search(text_lower))
    strong_male = bool(_STRONG_MALE.search(text_lower))
    if strong_female and not strong_male:
        return "female"
    if strong_male and not strong_female:
        return "male"
    if strong_female and strong_male:
        return None  # genuinely contradictory strong terms

    weak_female = bool(_WEAK_FEMALE.search(text_lower))
    weak_male = bool(_WEAK_MALE.search(text_lower))
    if weak_female and not weak_male:
        return "female"
    if weak_male and not weak_female:
        return "male"
    return None


def _extract_age(text_lower: str) -> Optional[float]:
    m = re.search(r"(\d{1,3})\s*[- ]?\s*(?:years?[- ]old|yo\b|y/o|years? of age)", text_lower)
    if m:
        return float(m.group(1))
    m = re.search(r"\bage[d]?\D{0,4}(\d{1,3})\b", text_lower)
    if m:
        return float(m.group(1))
    m = re.search(r"\b(?:is|was)\s+(\d{1,3})\b", text_lower)
    if m:
        return float(m.group(1))
    if re.search(r"\binfant\b|\bbaby\b", text_lower):
        return 1.0
    return None


def _extract_pclass(text_lower: str) -> Optional[int]:
    if re.search(r"\bfirst[- ]class\b|\b1st[- ]class\b", text_lower):
        return 1
    if re.search(r"\bsecond[- ]class\b|\b2nd[- ]class\b", text_lower):
        return 2
    if re.search(r"\bthird[- ]class\b|\b3rd[- ]class\b|\bsteerage\b", text_lower):
        return 3
    m = re.search(r"\bclass\s*(\d)\b", text_lower)
    if m and int(m.group(1)) in VALID_PCLASS:
        return int(m.group(1))
    return None


def _extract_fare(text_lower: str) -> Optional[float]:
    m = re.search(r"(?:fare of|paid|ticket (?:cost|price) of)?\s*[£$]\s*(\d+(?:\.\d+)?)", text_lower)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:pounds|gbp)\b", text_lower)
    if m:
        return float(m.group(1))
    return None


def _extract_embarked(text_lower: str) -> Optional[str]:
    for name, code in _PORT_MAP.items():
        if name in text_lower:
            return code
    m = re.search(r"embarked (?:at|from)?\s*([cqs])\b", text_lower)
    if m:
        return m.group(1).upper()
    return None


def _extract_family(text_lower: str) -> tuple:
    sibsp = None
    parch = None

    if re.search(r"\balone\b|\bno family\b|\bsolo\b|\bunaccompanied\b", text_lower):
        sibsp, parch = 0, 0

    m = re.search(r"(\d+)\s*sibling", text_lower)
    if m:
        sibsp = int(m.group(1))
    elif re.search(r"\b(?:with|had)\s+(?:a|his|her)?\s*spouse\b|\bwith (?:his|her) wife\b|\bwith (?:his|her) husband\b", text_lower):
        sibsp = (sibsp or 0) + 1

    m = re.search(r"(\d+)\s*(?:children|child|kids?)\b", text_lower)
    if m:
        parch = int(m.group(1))
    m2 = re.search(r"(\d+)\s*parents?\b", text_lower)
    if m2:
        parch = (parch or 0) + int(m2.group(1))
    if re.search(r"\btravell?ing with (?:his|her) (?:parents|mother|father|mom|dad)\b", text_lower):
        parch = (parch or 0) + 1

    return sibsp, parch


def parse_with_rules(text: str) -> ParsedQuery:
    """Deterministic regex/keyword-based extraction. No network access,
    no LLM -- used as the fallback path and directly by unit tests."""
    text_lower = text.lower()

    sibsp, parch = _extract_family(text_lower)
    parsed = ParsedQuery(
        pclass=_extract_pclass(text_lower),
        sex=_extract_sex(text_lower),
        age=_extract_age(text_lower),
        sibsp=sibsp,
        parch=parch,
        fare=_extract_fare(text_lower),
        embarked=_extract_embarked(text_lower),
        out_of_scope=_detect_out_of_scope(text_lower),
        ambiguous_fields=[],
        source="rules",
    )

    # Nothing at all extracted and no domain vocabulary -> treat as
    # out-of-scope rather than silently asking for every field.
    if (
        not parsed.out_of_scope
        and parsed.pclass is None
        and parsed.sex is None
        and parsed.age is None
        and not any(h in text_lower for h in _TITANIC_HINTS)
    ):
        parsed.out_of_scope = True

    return parsed


def parse_query(text: str, client=None, model: Optional[str] = None) -> ParsedQuery:
    """Parse a natural language query, preferring the LLM when a client is
    configured and falling back to rules on any error."""
    if client is not None and model is not None:
        try:
            return parse_with_llm(text, client, model)
        except Exception:
            # Fail open to the deterministic parser rather than crashing
            # the interface on a transient API error or malformed JSON.
            pass
    return parse_with_rules(text)
