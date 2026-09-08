"""
The LLM-powered natural language interface described in the project spec:

    Input Parsing -> Model Invocation -> Response Generation -> Edge Cases

This module is UI-agnostic on purpose: `answer_query()` is the single
entry point both the Streamlit app (src/app.py) and the CLI demo
(scripts/cli_demo.py) call, and it's what the interface tests exercise
directly.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import mlflow.sklearn
import pandas as pd

from src.llm_client import chat_text, get_llm_client
from src.nlp_parsing import ALL_PARSED_FIELDS, REQUIRED_FIELDS, ParsedQuery, parse_query
from src.preprocessing import engineer_features, select_model_columns

MODEL_DIR = "models/production"
STATS_PATH = "models/reference_stats.json"


@dataclass
class QueryResult:
    status: str  # "prediction" | "needs_clarification" | "out_of_scope"
    message: str
    parsed: dict = field(default_factory=dict)
    features: Optional[dict] = None
    prediction: Optional[str] = None
    probability: Optional[float] = None


def load_reference_stats(path: str = STATS_PATH) -> dict:
    if not Path(path).exists():
        # Sane fallback so the app doesn't hard-crash if run before
        # src/select_best_model.py has been executed.
        return {"fare_by_pclass": {1: 84.15, 2: 20.66, 3: 13.68}, "embarked_mode": "S"}
    with open(path) as f:
        data = json.load(f)
    data["fare_by_pclass"] = {int(k): v for k, v in data["fare_by_pclass"].items()}
    return data


def load_production_model(path: str = MODEL_DIR):
    return mlflow.sklearn.load_model(path)


def _default_title(sex: str, age: Optional[float]) -> str:
    if sex == "male":
        return "Master" if age is not None and age < 12 else "Mr"
    return "Mrs" if age is not None and age >= 18 else "Miss"


def fill_defaults(parsed: ParsedQuery, stats: dict) -> dict:
    """Turn a ParsedQuery (which may have optional fields missing) into a
    complete feature dict ready for the model pipeline, applying
    documented, non-random defaults for anything optional the user
    didn't mention. Required fields must already be present -- callers
    check that via `missing_required_fields` first.
    """
    sibsp = parsed.sibsp if parsed.sibsp is not None else 0
    parch = parsed.parch if parsed.parch is not None else 0
    embarked = parsed.embarked or stats["embarked_mode"]
    fare = parsed.fare if parsed.fare is not None else stats["fare_by_pclass"][parsed.pclass]
    title = _default_title(parsed.sex, parsed.age)

    # Route through the SAME feature-engineering function used at
    # training time, by building a one-row raw-schema frame, so there is
    # exactly one place that defines FamilySize/IsAlone/Deck logic.
    raw_row = pd.DataFrame(
        [
            {
                "Name": f"Query, {title}. Passenger",
                "Sex": parsed.sex,
                "Age": parsed.age,
                "SibSp": sibsp,
                "Parch": parch,
                "Fare": fare,
                "Pclass": parsed.pclass,
                "Embarked": embarked,
                "Cabin": None,
            }
        ]
    )
    engineered = engineer_features(raw_row)
    features_df = select_model_columns(engineered)
    return features_df


def missing_required_fields(parsed: ParsedQuery) -> list:
    return [f for f in REQUIRED_FIELDS if getattr(parsed, f) is None]


CLARIFICATION_PROMPTS = {
    "pclass": "which class they were travelling in (1st, 2nd, or 3rd)",
    "sex": "their sex (male or female)",
    "age": "their age",
}


def build_clarification_message(missing: list, ambiguous: list) -> str:
    parts = []
    if missing:
        needed = "; ".join(CLARIFICATION_PROMPTS.get(f, f) for f in missing)
        parts.append(
            f"I need a bit more information to make a prediction: {needed}."
        )
    if ambiguous:
        parts.append(
            f"Also, {', '.join(ambiguous)} looked contradictory or unclear -- "
            "could you clarify?"
        )
    parts.append(
        "For example: \"a 29-year-old woman in first class travelling with her husband\"."
    )
    return " ".join(parts)


OUT_OF_SCOPE_MESSAGE = (
    "I'm a focused tool for one thing: estimating whether a Titanic "
    "passenger, described in a sentence, would have survived the sinking. "
    "I can't help with unrelated questions -- try describing a passenger "
    "instead, e.g. \"a 22-year-old third-class male passenger travelling alone\"."
)

RESPONSE_SYSTEM_PROMPT = """You explain the output of a machine learning model that predicts \
whether a Titanic passenger survived, to a general audience. You will be given the \
passenger description, the model's prediction, and its predicted survival probability. \
Write 2-4 sentences: state the prediction and probability, mention which factors from \
the historical data most plausibly drove it (e.g. sex, class, age, fare), and note this \
is a probabilistic estimate from a historical dataset, not a certainty. Do not invent \
facts not given to you. Be warm but factual."""


def generate_response_text(features_row: dict, prediction_label: str, probability: float, client, model) -> str:
    if client is not None and model is not None:
        try:
            user_prompt = (
                f"Passenger description: {json.dumps(features_row)}\n"
                f"Model prediction: {prediction_label}\n"
                f"Predicted survival probability: {probability:.1%}"
            )
            return chat_text(client, model, RESPONSE_SYSTEM_PROMPT, user_prompt).strip()
        except Exception:
            pass  # fall through to the template below

    # Deterministic template fallback (no LLM configured, or the call failed).
    verb = "would likely have survived" if prediction_label == "Survived" else "would likely NOT have survived"
    return (
        f"Based on the historical data, this passenger {verb} "
        f"(estimated survival probability: {probability:.1%}). This reflects patterns "
        f"such as sex, passenger class, age, and fare in the training data -- it's a "
        f"statistical estimate from a century-old dataset, not a certainty."
    )


class TitanicSurvivalAssistant:
    """Stateful convenience wrapper that loads the model once and answers
    many queries. `answer_query` (module-level) is the stateless core;
    this class just avoids reloading the model/LLM client per call.
    """

    def __init__(self, model_dir: str = MODEL_DIR, stats_path: str = STATS_PATH):
        self.model = load_production_model(model_dir)
        self.stats = load_reference_stats(stats_path)
        self.client, self.llm_model = get_llm_client()

    def ask(self, text: str) -> QueryResult:
        return answer_query(
            text,
            model=self.model,
            stats=self.stats,
            client=self.client,
            llm_model=self.llm_model,
        )


def answer_query(text: str, model, stats: dict, client=None, llm_model=None) -> QueryResult:
    """Full pipeline: parse -> validate/edge-cases -> invoke model ->
    generate response. Pure function of its arguments (no global state),
    so it's easy to unit test with a stub model/client.
    """
    parsed = parse_query(text, client=client, model=llm_model)

    if parsed.out_of_scope:
        return QueryResult(status="out_of_scope", message=OUT_OF_SCOPE_MESSAGE, parsed=parsed.as_dict())

    missing = missing_required_fields(parsed)
    if missing or parsed.ambiguous_fields:
        return QueryResult(
            status="needs_clarification",
            message=build_clarification_message(missing, parsed.ambiguous_fields),
            parsed=parsed.as_dict(),
        )

    features_df = fill_defaults(parsed, stats)
    prediction = model.predict(features_df)[0]
    probability = (
        model.predict_proba(features_df)[0][1] if hasattr(model, "predict_proba") else float(prediction)
    )
    label = "Survived" if int(prediction) == 1 else "Did not survive"

    features_row = features_df.iloc[0].to_dict()
    response_text = generate_response_text(features_row, label, float(probability), client, llm_model)

    return QueryResult(
        status="prediction",
        message=response_text,
        parsed=parsed.as_dict(),
        features=features_row,
        prediction=label,
        probability=float(probability),
    )
