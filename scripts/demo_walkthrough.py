"""
Generates demo_transcript.md by actually running the application
(src/interface.py, against the real exported model) on a fixed script of
queries covering: a complete natural-language query, the system
parsing it and invoking the model, and an edge case (incomplete input).

This stands in for a screen-recorded demo in an environment with no
display. Every line of output below is real, not hand-written -- run this
script yourself to reproduce it.

Usage:
    python scripts/demo_walkthrough.py
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.interface import TitanicSurvivalAssistant  # noqa: E402

SCRIPT = [
    (
        "1. A complete natural-language query",
        "A 29-year-old woman travelling in first class with her husband, paid a $150 fare.",
    ),
    (
        "2. An out-of-scope query (edge case)",
        "What's the weather like in Denver today?",
    ),
    (
        "3. An incomplete query -- missing required fields (edge case)",
        "A passenger who paid $50.",
    ),
    (
        "4. Following up with the missing details",
        "He was a 40-year-old male in third class.",
    ),
]


def render_result(result) -> str:
    lines = [f"**Status:** `{result.status}`", "", f"**Response:** {result.message}"]
    if result.parsed:
        lines += ["", "**Parsed fields:**", "```json", json.dumps(result.parsed, indent=2), "```"]
    if result.status == "prediction":
        lines += [
            "",
            f"**Prediction:** {result.prediction}  |  **Survival probability:** {result.probability:.1%}",
            "",
            "**Model input features:**",
            "```json",
            json.dumps(result.features, indent=2),
            "```",
        ]
    return "\n".join(lines)


def main():
    assistant = TitanicSurvivalAssistant()
    mode = "LLM (Nebius/OpenAI)" if assistant.client else "rule-based fallback (no LLM API key configured)"

    out = [
        "# Demo walkthrough",
        "",
        f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by "
        "`scripts/demo_walkthrough.py`, running the real application code against the "
        "trained production model.",
        "",
        f"**Parsing/response mode for this run:** {mode}",
        "",
    ]

    for title, query in SCRIPT:
        result = assistant.ask(query)
        out += [f"## {title}", "", f"**User:** {query}", "", render_result(result), ""]

    text = "\n".join(out)
    Path("demo_transcript.md").write_text(text)
    print(text)
    print("\n\nSaved to demo_transcript.md")


if __name__ == "__main__":
    main()
