"""
Data loading, feature engineering, and preprocessing for the Titanic
survival prediction project.

Design notes
------------
- All imputation / scaling / encoding statistics are learned with
  ``ColumnTransformer.fit`` on the TRAINING split only, then applied with
  ``.transform`` to the test split. This is enforced by always building
  the preprocessor as part of an sklearn ``Pipeline`` together with the
  estimator, so a fresh call to ``.fit`` never sees test rows
  (see ``src/train.py``). This avoids data leakage.
- Feature engineering functions here are pure functions of a single
  dataframe (no fitting), so they are safe to apply identically to train
  and test data before the split.
"""
from __future__ import annotations

import re

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "Survived"

# Titles that are common enough to keep as their own category; everything
# else is bucketed into "Rare" so the one-hot encoder doesn't explode into
# dozens of near-empty columns.
_COMMON_TITLES = {"Mr", "Mrs", "Miss", "Master"}

_TITLE_NORMALIZATION = {
    "Mlle": "Miss",
    "Ms": "Miss",
    "Mme": "Mrs",
}

NUMERIC_FEATURES = ["Age", "Fare", "FamilySize"]
CATEGORICAL_FEATURES = ["Pclass", "Sex", "Embarked", "Title", "IsAlone", "Deck"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_raw_data(path: str) -> pd.DataFrame:
    """Load the raw Titanic CSV from disk."""
    return pd.read_csv(path)


def extract_title(name: str) -> str:
    """Extract an honorific title (Mr, Mrs, Miss, Master, Rare) from a
    passenger's full name string, e.g. ``"Braund, Mr. Owen Harris"`` ->
    ``"Mr"``.

    Returns ``"Unknown"`` if no title-like token can be found, so this
    function never raises on malformed input.
    """
    if not isinstance(name, str):
        return "Unknown"
    match = re.search(r",\s*([^.]*)\.", name)
    if not match:
        return "Unknown"
    title = match.group(1).strip()
    title = _TITLE_NORMALIZATION.get(title, title)
    if title not in _COMMON_TITLES:
        return "Rare"
    return title


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered columns and return a NEW dataframe (does not mutate
    the input). Safe to call on train and test/inference data alike since
    it fits nothing -- it only derives deterministic columns from the raw
    fields.

    Adds:
      - Title: honorific parsed from Name
      - FamilySize: SibSp + Parch + 1 (the passenger themself)
      - IsAlone: 1 if FamilySize == 1 else 0
      - Deck: first letter of Cabin, or "Unknown" if Cabin is missing
    """
    out = df.copy()

    if "Name" in out.columns:
        out["Title"] = out["Name"].apply(extract_title)
    else:
        out["Title"] = "Unknown"

    sibsp = out["SibSp"] if "SibSp" in out.columns else 0
    parch = out["Parch"] if "Parch" in out.columns else 0
    out["FamilySize"] = sibsp + parch + 1
    out["IsAlone"] = (out["FamilySize"] == 1).astype(int)

    if "Cabin" in out.columns:
        out["Deck"] = out["Cabin"].apply(
            lambda c: c[0] if isinstance(c, str) and len(c) > 0 else "Unknown"
        )
    else:
        out["Deck"] = "Unknown"

    return out


def select_model_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the columns the model pipeline consumes, in a fixed
    order. Raises KeyError with a clear message if engineering wasn't run.
    """
    missing = [c for c in ALL_FEATURES if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing expected feature columns {missing}. "
            "Did you call engineer_features() first?"
        )
    return df[ALL_FEATURES]


def build_preprocessor() -> ColumnTransformer:
    """Build (but do not fit) the ColumnTransformer used ahead of every
    estimator. Median-imputes + scales numeric columns; most-frequent
    imputes + one-hot encodes categorical columns. Fitting happens later,
    only on the training split, inside a Pipeline (see src/train.py).
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def load_and_prepare(path: str):
    """Convenience wrapper: load raw CSV, engineer features, and return
    (X, y) ready to be split. No fitting/leakage-prone steps happen here.
    """
    df = load_raw_data(path)
    df = engineer_features(df)
    X = select_model_columns(df)
    y = df[TARGET_COLUMN]
    return X, y


def reference_statistics(raw_path: str) -> dict:
    """Compute small lookup tables from the full raw dataset (not fitted
    on any particular split) that the LLM interface uses to fill in
    optional fields a user didn't mention, e.g. a typical fare for a
    given class. These are descriptive dataset facts, not
    label-derived leakage -- the same numbers a domain expert would
    already know from studying the manifest.
    """
    df = load_raw_data(raw_path)
    fare_by_pclass = (
        df.groupby("Pclass")["Fare"].median().round(2).to_dict()
    )
    embarked_mode = df["Embarked"].mode(dropna=True).iloc[0]
    return {
        "fare_by_pclass": {int(k): float(v) for k, v in fare_by_pclass.items()},
        "embarked_mode": str(embarked_mode),
    }


def train_test_split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    """Stratified train/test split so the survival rate is preserved in
    both splits."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
