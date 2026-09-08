"""Preprocessing unit tests: missing values, encoding, scaling, and
immutability of the input dataframe."""
import numpy as np
import pandas as pd

from src.preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_preprocessor,
    engineer_features,
    extract_title,
    select_model_columns,
    train_test_split_data,
)


def test_missing_values_are_imputed_not_dropped(sample_passengers_df):
    """A row with a missing Age/Cabin/Embarked must survive preprocessing
    with no NaNs in the transformed output (imputation, not row-dropping)."""
    engineered = engineer_features(sample_passengers_df)
    X = select_model_columns(engineered)

    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(X)
    transformed = np.asarray(transformed.todense()) if hasattr(transformed, "todense") else transformed

    assert transformed.shape[0] == len(sample_passengers_df), "no rows should be dropped"
    assert not np.isnan(transformed).any(), "no NaNs should remain after imputation"


def test_categorical_columns_are_one_hot_encoded(sample_passengers_df):
    """Categorical features must come out as multiple binary (0/1)
    columns, not as raw strings or a single label-encoded integer."""
    engineered = engineer_features(sample_passengers_df)
    X = select_model_columns(engineered)

    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(X)
    transformed = np.asarray(transformed.todense()) if hasattr(transformed, "todense") else transformed

    n_numeric = len(NUMERIC_FEATURES)
    cat_block = transformed[:, n_numeric:]
    # One-hot columns are strictly 0/1.
    assert set(np.unique(cat_block.round(6))).issubset({0.0, 1.0})
    # More output columns than raw categorical columns confirms one-hot
    # expansion actually happened (not a passthrough / label encoding).
    assert transformed.shape[1] > len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)


def test_numeric_features_are_scaled(raw_df):
    """After StandardScaler, numeric columns should have ~0 mean and
    ~unit variance on the data they were fit on."""
    engineered = engineer_features(raw_df)
    X = select_model_columns(engineered)

    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(X)
    transformed = np.asarray(transformed.todense()) if hasattr(transformed, "todense") else transformed

    numeric_block = transformed[:, : len(NUMERIC_FEATURES)]
    assert np.allclose(numeric_block.mean(axis=0), 0, atol=1e-6)
    assert np.allclose(numeric_block.std(axis=0), 1, atol=1e-6)


def test_engineer_features_does_not_mutate_input(sample_passengers_df):
    """engineer_features must return a new dataframe and leave the
    caller's original untouched (important since the same raw frame is
    reused for both train and test splits)."""
    original = sample_passengers_df.copy(deep=True)
    _ = engineer_features(sample_passengers_df)
    pd.testing.assert_frame_equal(sample_passengers_df, original)


def test_extract_title_handles_common_and_rare_titles():
    assert extract_title("Braund, Mr. Owen Harris") == "Mr"
    assert extract_title("Cumings, Mrs. John Bradley (Florence Briggs Thayer)") == "Mrs"
    assert extract_title("Heikkinen, Miss. Laina") == "Miss"
    assert extract_title("Palsson, Master. Gosta Leonard") == "Master"
    assert extract_title("Some, Countess. Weird Title") == "Rare"
    assert extract_title(None) == "Unknown"


def test_train_test_split_is_stratified(raw_df):
    """The survival rate in each split should closely match the overall
    rate -- this is what stratify=y guarantees and guards against a
    future refactor silently dropping it (which would risk data leakage
    style bias in evaluation)."""
    engineered = engineer_features(raw_df)
    X = select_model_columns(engineered)
    y = engineered["Survived"]

    X_train, X_test, y_train, y_test = train_test_split_data(X, y, test_size=0.2, random_state=42)

    overall_rate = y.mean()
    assert abs(y_train.mean() - overall_rate) < 0.03
    assert abs(y_test.mean() - overall_rate) < 0.05
    assert len(X_train) + len(X_test) == len(X)


def test_unseen_categorical_value_at_inference_does_not_crash(sample_passengers_df):
    """OneHotEncoder must be configured with handle_unknown='ignore' so a
    category never seen in training (e.g. a Deck letter or Title that
    didn't appear in the fit data) doesn't raise at inference time."""
    engineered = engineer_features(sample_passengers_df)
    X = select_model_columns(engineered)

    preprocessor = build_preprocessor()
    preprocessor.fit(X)

    novel_row = X.iloc[[0]].copy()
    novel_row["Deck"] = "Z"  # deck letter that does not exist in the fit data
    novel_row["Title"] = "TotallyNovelTitle"

    # Should not raise.
    transformed = preprocessor.transform(novel_row)
    assert transformed.shape[0] == 1
