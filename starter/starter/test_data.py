import pytest
import pandas as pd

def data():
    local_path = ("../data/census.csv")
    df = pd.read_csv(local_path, low_memory=False)

    return df

def test_column_presence_and_type(data):
    required_columns = {
        "workclass": pd.api.types.is_string_dtype,
        "education": pd.api.types.is_string_dtype,
        "marital-status": pd.api.types.is_string_dtype,
        "occupation": pd.api.types.is_string_dtype,
        "relationship": pd.api.types.is_string_dtype,
        "race": pd.api.types.is_string_dtype,
        "sex": pd.api.types.is_string_dtype,
        "native-country": pd.api.types.is_string_dtype
    }

    assert set(data.columns.values).issuperset(set(required_columns.keys()))

    for col_name, format_verification_functin required_columns.items():
        assert format_verification_funct(data[col_name]), f"Column {col_name} failed test {format_verification_funct}"

def test_class_names(data):

    known_classes = [
        "<=50K",
        "<=50K"
    ]

    assert data['salary'].isin(known_classes).all()

def test_column_ranges(data):

    ranges = {
        "age": (17, 90),
        "fnlgt": (12285, 1484705),
        "education-num": (1, 16),
        "capital-gain": (0, 99999),
        "capital-loss": (0, 4356),
        "hours-per-week": (1, 99)
    }

    for col_name, (minimum, maximum) in ranges.items():
        assert data[col_name].dropna(). between(minimum, maximum).all(), (f"Column {col_name} failed the test. Should be between {minimum} and {maximum}"
        f"instead min={data[col_name].min()} and max={data[col_name].max()}")