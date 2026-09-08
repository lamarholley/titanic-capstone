"""Interface tests: natural-language parsing accuracy and edge-case
handling (missing fields, ambiguity, out-of-scope queries).

These tests exercise the deterministic rule-based parser
(`parse_with_rules`) rather than calling a live LLM, so they run offline
and without an API key -- `answer_query` falls back to this same parser
automatically whenever no LLM client is configured, so it is exactly the
code path exercised in an environment with no Nebius/OpenAI key set.
"""
import numpy as np
import pandas as pd
import pytest

from src.interface import answer_query, fill_defaults, missing_required_fields
from src.nlp_parsing import parse_with_rules


class StubModel:
    """A fake trained model standing in for the full production Pipeline
    (preprocessing + classifier). The real Pipeline's `.predict` /
    `.predict_proba` take the raw engineered feature frame (Age, Fare,
    Pclass, Sex, ...) directly -- this stub does the same, so it's a
    faithful substitute. It predicts survived (1) iff the passenger is
    female, mimicking "women and children first", without needing a real
    fitted classifier. Lets interface tests be fast and fully
    deterministic.
    """

    def predict(self, X: pd.DataFrame):
        return np.where(X["Sex"] == "female", 1, 0)

    def predict_proba(self, X: pd.DataFrame):
        p1 = np.where(X["Sex"] == "female", 0.9, 0.1)
        return np.column_stack([1 - p1, p1])


@pytest.fixture
def stub_model():
    return StubModel()


@pytest.fixture
def stats():
    return {"fare_by_pclass": {1: 84.15, 2: 20.66, 3: 13.68}, "embarked_mode": "S"}


# ---------------------------------------------------------------------------
# Input parsing accuracy
# ---------------------------------------------------------------------------


def test_parses_full_description_correctly():
    parsed = parse_with_rules(
        "A 22-year-old third-class male passenger travelling alone."
    )
    assert parsed.pclass == 3
    assert parsed.sex == "male"
    assert parsed.age == 22.0
    assert parsed.sibsp == 0
    assert parsed.parch == 0
    assert parsed.out_of_scope is False


def test_parses_family_and_port_details():
    parsed = parse_with_rules(
        "A 5-year-old boy in third class travelling with his parents, embarked at Southampton."
    )
    assert parsed.sex == "male"
    assert parsed.pclass == 3
    assert parsed.age == 5.0
    assert parsed.embarked == "S"
    assert parsed.parch == 1


def test_relational_words_do_not_override_subjects_sex():
    """'woman ... with her husband' must resolve to female -- the
    husband is a companion, not the subject."""
    parsed = parse_with_rules(
        "A 29-year-old woman travelling in first class with her husband, paid a $150 fare."
    )
    assert parsed.sex == "female"
    assert parsed.pclass == 1
    assert parsed.fare == 150.0
    assert parsed.sibsp == 1


# ---------------------------------------------------------------------------
# Edge cases: missing / ambiguous / out-of-scope
# ---------------------------------------------------------------------------


def test_missing_required_fields_are_detected():
    parsed = parse_with_rules("A passenger who paid $50.")
    missing = missing_required_fields(parsed)
    assert set(missing) == {"pclass", "sex", "age"}


def test_answer_query_asks_for_clarification_when_incomplete(stub_model, stats):
    result = answer_query("A passenger who paid $50.", model=stub_model, stats=stats)
    assert result.status == "needs_clarification"
    assert "class" in result.message.lower() or "sex" in result.message.lower()


def test_answer_query_flags_out_of_scope_questions(stub_model, stats):
    result = answer_query("What is the weather like today?", model=stub_model, stats=stats)
    assert result.status == "out_of_scope"
    assert "titanic" in result.message.lower() or "survival" in result.message.lower()


def test_answer_query_produces_prediction_for_complete_input(stub_model, stats):
    result = answer_query(
        "A 30-year-old female passenger in first class travelling alone.",
        model=stub_model,
        stats=stats,
    )
    assert result.status == "prediction"
    assert result.prediction == "Survived"
    assert result.probability == pytest.approx(0.9)
    assert result.features["Pclass"] == 1


def test_fill_defaults_uses_class_median_fare_when_unstated(stats):
    parsed = parse_with_rules("A 40-year-old male passenger in third class.")
    features = fill_defaults(parsed, stats)
    assert features.iloc[0]["Fare"] == stats["fare_by_pclass"][3]
    assert features.iloc[0]["Embarked"] == stats["embarked_mode"]
