import pandas as pd
import pytest

RAW_DATA_PATH = "data/raw/titanic.csv"


@pytest.fixture(scope="session")
def raw_df():
    return pd.read_csv(RAW_DATA_PATH)


@pytest.fixture
def sample_passengers_df():
    """A tiny, hand-crafted frame covering: a normal row, a row with a
    missing Age, a missing Embarked, and a missing Cabin -- exactly the
    columns real-world Titanic data is missing values in."""
    return pd.DataFrame(
        [
            {
                "PassengerId": 1,
                "Survived": 1,
                "Pclass": 1,
                "Name": "Cumings, Mrs. John Bradley (Florence Briggs Thayer)",
                "Sex": "female",
                "Age": 38.0,
                "SibSp": 1,
                "Parch": 0,
                "Ticket": "PC 17599",
                "Fare": 71.28,
                "Cabin": "C85",
                "Embarked": "C",
            },
            {
                "PassengerId": 2,
                "Survived": 0,
                "Pclass": 3,
                "Name": "Braund, Mr. Owen Harris",
                "Sex": "male",
                "Age": None,  # missing age
                "SibSp": 0,
                "Parch": 0,
                "Ticket": "A/5 21171",
                "Fare": 7.25,
                "Cabin": None,  # missing cabin
                "Embarked": "S",
            },
            {
                "PassengerId": 3,
                "Survived": 1,
                "Pclass": 2,
                "Name": "Master. Timmy",
                "Sex": "male",
                "Age": 4.0,
                "SibSp": 1,
                "Parch": 1,
                "Ticket": "11753",
                "Fare": 26.0,
                "Cabin": None,
                "Embarked": None,  # missing embarked
            },
        ]
    )
