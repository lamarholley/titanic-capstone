"""
Interactive command-line demo of the Titanic Survival Assistant --
useful for a quick terminal walkthrough without starting Streamlit.

Usage:
    python scripts/cli_demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.interface import TitanicSurvivalAssistant  # noqa: E402


def main():
    print("Loading model...")
    assistant = TitanicSurvivalAssistant()
    mode = "LLM (Nebius/OpenAI)" if assistant.client else "rule-based fallback (no API key set)"
    print(f"Ready. Parsing mode: {mode}")
    print("Describe a Titanic passenger, or type 'quit' to exit.\n")

    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text or text.lower() in {"quit", "exit"}:
            break

        result = assistant.ask(text)
        print(f"[{result.status}] {result.message}\n")


if __name__ == "__main__":
    main()
