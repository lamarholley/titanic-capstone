"""
Streamlit delivery for the Titanic survival LLM interface.

Run with:
    streamlit run src/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Streamlit executes this file directly (not via `python -m`), which only
# puts src/ on sys.path, not the project root -- so `from src.interface
# import ...` below can't resolve without this. (scripts/cli_demo.py and
# scripts/demo_walkthrough.py already do this; it was missing here.)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src.interface import TitanicSurvivalAssistant

st.set_page_config(page_title="Titanic Survival Assistant", page_icon="🚢", layout="centered")

st.title("🚢 Titanic Survival Assistant")
st.caption(
    "Describe a Titanic passenger in plain English and get a survival-likelihood "
    "estimate from a trained ML model, explained by an LLM."
)


@st.cache_resource
def get_assistant() -> TitanicSurvivalAssistant:
    return TitanicSurvivalAssistant()


assistant = get_assistant()

if assistant.client is None:
    st.info(
        "No LLM API key detected (NEBIUS_API_KEY or OPENAI_API_KEY). "
        "Running with the deterministic rule-based parser and template responses. "
        "Set an API key to enable full natural-language understanding.",
        icon="ℹ️",
    )

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.write(turn["content"])

example_cols = st.columns(3)
examples = [
    "A 29-year-old woman in first class travelling with her husband.",
    "A 22-year-old third-class man travelling alone.",
    "A 5-year-old boy in third class with his parents, embarked at Southampton.",
]
for col, example in zip(example_cols, examples):
    if col.button(example, use_container_width=True):
        st.session_state.pending_query = example

query = st.chat_input("Describe a passenger...")
if "pending_query" in st.session_state:
    query = st.session_state.pop("pending_query")

if query:
    st.session_state.history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    result = assistant.ask(query)

    with st.chat_message("assistant"):
        st.write(result.message)
        if result.status == "prediction":
            c1, c2 = st.columns(2)
            c1.metric("Prediction", result.prediction)
            c2.metric("Survival probability", f"{result.probability:.1%}")
            with st.expander("Parsed features used for this prediction"):
                st.json(result.features)
        elif result.status == "needs_clarification":
            with st.expander("What I understood so far"):
                st.json(result.parsed)

    st.session_state.history.append({"role": "assistant", "content": result.message})

with st.sidebar:
    st.header("About")
    st.write(
        "This app pairs a trained classifier (selected from several MLflow-tracked "
        "experiment runs) with an LLM that parses free-text passenger descriptions "
        "into model features and explains the resulting prediction."
    )
    st.write("**LLM provider:**", "Nebius/OpenAI (live)" if assistant.client else "Rule-based fallback")
    if st.button("Clear conversation"):
        st.session_state.history = []
        st.rerun()
