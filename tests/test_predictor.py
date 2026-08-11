"""Tests for the per-model raw-input predictors.

The contract these lock down: a pickled :class:`DivePredictor` accepts data in
the same shape as the *source file*, not the encoded matrix the estimator was
actually fitted on, and it survives a pickle round-trip without needing the
originating ``Dive`` object.
"""

from __future__ import annotations

import pickle

import pandas as pd
import pytest

from dive import Dive
from dive.predictor import DivePredictor, SchemaMismatch


@pytest.fixture(scope="module")
def trained(multiclass_df_module):
    dive = Dive(target="species", mode="fast", time_budget=60, verbose=False)
    dive.fit(multiclass_df_module)
    return dive


@pytest.fixture(scope="module")
def multiclass_df_module():
    import numpy as np

    rng = np.random.default_rng(11)
    n = 180
    petal = np.concatenate([rng.normal(loc, 0.4, n // 3) for loc in (1.5, 4.3, 5.6)])
    sepal = np.concatenate([rng.normal(loc, 0.4, n // 3) for loc in (5.0, 5.9, 6.6)])
    return pd.DataFrame(
        {
            "petal length (cm)": petal,
            "sepal length (cm)": sepal,
            "noise": rng.normal(size=n),
            "species": np.repeat(["setosa", "versicolor", "virginica"], n // 3),
        }
    )


@pytest.fixture(scope="module")
def predictors(trained):
    return trained.build_predictors(dataset_name="iris sample")


def test_one_predictor_per_trained_model(trained, predictors):
    assert set(predictors) == set(trained.leaderboard()["Model"])
    assert all(isinstance(p, DivePredictor) for p in predictors.values())


def test_predicts_from_raw_dataframe(predictors, multiclass_df_module):
    """Raw frames go in; original string labels come out."""
    raw = multiclass_df_module.drop(columns=["species"])
    for name, predictor in predictors.items():
        out = predictor.predict(raw)
        assert len(out) == len(raw), name
        assert set(out) <= {"setosa", "versicolor", "virginica"}, name


def test_accepts_single_dict_and_list_of_dicts(predictors, multiclass_df_module):
    row = multiclass_df_module.drop(columns=["species"]).iloc[0].to_dict()
    predictor = predictors[next(iter(predictors))]
    assert len(predictor.predict(row)) == 1
    assert len(predictor.predict([row, row])) == 2
    assert len(predictor(row)) == 1


def test_target_column_may_be_present_or_absent(predictors, multiclass_df_module):
    """Scoring a labelled file shouldn't require dropping the target first."""
    predictor = predictors[next(iter(predictors))]
    with_target = predictor.predict(multiclass_df_module)
    without = predictor.predict(multiclass_df_module.drop(columns=["species"]))
    assert list(with_target) == list(without)


def test_column_order_does_not_matter(predictors, multiclass_df_module):
    predictor = predictors[next(iter(predictors))]
    raw = multiclass_df_module.drop(columns=["species"])
    shuffled = raw[list(reversed(raw.columns))]
    assert list(predictor.predict(shuffled)) == list(predictor.predict(raw))


def test_missing_column_raises_actionable_error(predictors, multiclass_df_module):
    predictor = predictors[next(iter(predictors))]
    raw = multiclass_df_module.drop(columns=["species", "noise"])
    with pytest.raises(SchemaMismatch) as excinfo:
        predictor.predict(raw)
    message = str(excinfo.value)
    assert "noise" in message
    assert "describe_input" in message


def test_predict_proba_columns_are_class_names(predictors, multiclass_df_module):
    raw = multiclass_df_module.drop(columns=["species"]).head(5)
    for name, predictor in predictors.items():
        if not predictor.has_proba:
            continue
        proba = predictor.predict_proba(raw)
        assert list(proba.columns) == ["setosa", "versicolor", "virginica"], name
        assert proba.sum(axis=1).round(5).eq(1.0).all(), name


def test_describe_input_lists_every_required_column(predictors):
    text = predictors[next(iter(predictors))].describe_input()
    for column in ("petal length (cm)", "sepal length (cm)", "noise"):
        assert column in text
    assert "setosa" in text


def test_survives_pickle_round_trip(predictors, multiclass_df_module, tmp_path):
    """The predictor must stand alone - no live Dive object in the pickle."""
    predictor = predictors[next(iter(predictors))]
    raw = multiclass_df_module.drop(columns=["species"]).head(10)
    expected = list(predictor.predict(raw))

    path = tmp_path / "p.pkl"
    path.write_bytes(pickle.dumps(predictor))
    revived = pickle.loads(path.read_bytes())

    assert list(revived.predict(raw)) == expected
    assert revived.model_name == predictor.model_name


def test_categorical_and_missing_values_are_encoded(tmp_path):
    """Raw strings, unseen categories and NaNs all pass through the pipeline."""
    import numpy as np

    rng = np.random.default_rng(3)
    n = 200
    frame = pd.DataFrame(
        {
            "age": rng.integers(18, 80, n),
            "city": rng.choice(["Delhi", "Mumbai", "Pune"], n),
            "spend": rng.normal(100, 20, n),
        }
    )
    frame.loc[::13, "spend"] = np.nan
    frame["churn"] = np.where(frame["age"] < 40, "yes", "no")

    dive = Dive(target="churn", mode="fast", time_budget=60, verbose=False)
    dive.fit(frame)
    predictor = next(iter(dive.build_predictors(dataset_name="churn").values()))

    assert predictor.predict({"age": 25, "city": "Delhi", "spend": 40.0})[0] in {
        "yes",
        "no",
    }
    # Unseen category and an explicit null must not raise.
    assert len(predictor.predict({"age": 25, "city": "Tokyo", "spend": None})) == 1
    assert len(predictor.predict(frame)) == n


def test_integer_labels_round_trip_as_integers(binary_df):
    """A model trained on 0/1 must predict 0/1, not '0'/'1'.

    Labels are stringified before encoding to tolerate mixed-type targets, so
    without an explicit lookup the decoded values come back as strings and any
    downstream comparison against the original column silently fails.
    """
    dive = Dive(target="target", mode="fast", time_budget=60, verbose=False)
    dive.fit(binary_df)
    predictor = next(iter(dive.build_predictors(dataset_name="binary").values()))

    raw = binary_df.drop(columns=["target"])
    predictions = predictor.predict(raw)
    assert set(predictions) <= set(binary_df["target"].unique())
    assert pd.api.types.is_integer_dtype(pd.Series(predictions))
    # The full-run artifact must agree with the exported predictor.
    assert set(dive.predict(raw)) <= set(binary_df["target"].unique())


def test_regression_predictor_returns_numbers(regression_df):
    dive = Dive(target="target", mode="fast", time_budget=60, verbose=False)
    dive.fit(regression_df)
    predictor = next(iter(dive.build_predictors(dataset_name="reg").values()))

    out = predictor.predict(regression_df.drop(columns=["target"]))
    assert predictor.problem_type == "regression"
    assert predictor.class_names is None
    assert pd.api.types.is_numeric_dtype(pd.Series(out))
    with pytest.raises(SchemaMismatch):
        predictor.predict_proba(regression_df.drop(columns=["target"]))
