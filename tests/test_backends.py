from types import SimpleNamespace
import importlib
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from recursive_selection import (CatBoostAdapter, LightGBMAdapter, MacroQualityPolicy,
                              RecursiveGroupSelector, XGBoostAdapter)
from recursive_selection.backends._base import BaseModelAdapter, normalized_shap


def test_importing_core_does_not_import_optional_libraries():
    command = [sys.executable, "-c", "import sys,recursive_selection; assert not {'catboost','xgboost','lightgbm','shap'}.intersection(sys.modules)"]
    subprocess.run(command, check=True, env=os.environ.copy(), capture_output=True, text=True)


def test_selector_requires_an_explicit_adapter():
    with pytest.raises(TypeError, match="model_adapter"):
        RecursiveGroupSelector()
    with pytest.raises(ValueError, match="model_adapter is required"):
        RecursiveGroupSelector(model_adapter=None)
    with pytest.raises(TypeError, match="estimator"):
        RecursiveGroupSelector(estimator=object())
    import recursive_selection as package
    assert not hasattr(package, 'register_backend')


def test_adapter_params_are_copied_for_each_fresh_model():
    class Model:
        def __init__(self, **params):
            self.params = params

    class Adapter(BaseModelAdapter):
        @staticmethod
        def model_class():
            return Model

    params = {'depth': 3, 'nested': {'values': [1]}}
    adapter = Adapter(params=params)
    params['nested']['values'].append(2)
    first, second = adapter.new_model(), adapter.new_model()
    first.params['nested']['values'].append(3)
    assert first is not second
    assert second.params == adapter.params == {'depth': 3, 'nested': {'values': [1]}}


def test_native_shap_normalization():
    binary = np.arange(8).reshape(2, 4)
    np.testing.assert_array_equal(normalized_shap(binary, 2, 3), binary[:, None, :-1])
    multi = np.arange(24).reshape(2, 3, 4)
    np.testing.assert_array_equal(normalized_shap(multi, 2, 3), multi[:, :, :-1])
    with pytest.raises(ValueError):
        normalized_shap(np.ones((2, 2)), 2, 3)


def test_xgboost_shap_uses_best_iteration(monkeypatch):
    calls = {}

    class Booster:
        feature_names = None

        def predict(self, matrix, **options):
            calls.update(options)
            return np.ones((2, 3, 5))

    booster = Booster()
    model = SimpleNamespace(best_iteration=6, get_booster=lambda: booster)
    monkeypatch.setitem(sys.modules, "xgboost", SimpleNamespace(DMatrix=lambda X, **kw: X))
    values = XGBoostAdapter(params={}).feature_shap(model, np.ones((2, 4)), [0, 1])
    assert calls["iteration_range"] == (0, 7)
    assert calls["strict_shape"] is True
    assert values.shape == (2, 3, 4)


def test_lightgbm_multiclass_layout_and_best_iteration():
    class Booster:
        best_iteration = 9

        def predict(self, X, **options):
            assert options == {"pred_contrib": True, "num_iteration": 9}
            return np.arange(len(X) * 15).reshape(len(X), 15)

    values = LightGBMAdapter(params={}).feature_shap(Booster(), np.ones((2, 4)), [0, 1])
    assert values.shape == (2, 3, 4)
    np.testing.assert_array_equal(values[0, 1], [5, 6, 7, 8])


def test_adapter_rejects_warm_start_and_unmapped_feature_settings():
    with pytest.raises(ValueError, match="feature remapping"):
        CatBoostAdapter(params={"cat_features": [0]})
    with pytest.raises(ValueError, match="warm-start"):
        XGBoostAdapter(fit_params={"xgb_model": "old.json"})


@pytest.mark.parametrize("adapter_type", [CatBoostAdapter, XGBoostAdapter, LightGBMAdapter])
def test_adapters_accept_keyword_configuration_only(adapter_type):
    with pytest.raises(TypeError, match='estimator'):
        adapter_type(estimator=object())
    with pytest.raises(TypeError):
        adapter_type(object())
    assert adapter_type(params={'seed': 42}).params == {'seed': 42}


class RecordingModel:
    def __init__(self):
        self.params = {}
        self.calls = []

    def set_params(self, **params):
        self.params.update(params)
        return self

    def fit(self, X, y, **options):
        self.calls.append((X, y, options))


def test_xgboost_fit_converts_validation_and_configures_stopping_on_model():
    model = RecordingModel()
    X, y = np.ones((4, 2)), np.array([0, 1, 0, 1])
    adapter = XGBoostAdapter(params={"early_stopping_rounds": 7}, early_stopping_metric="configured")
    adapter.fit(model, X, y, (X, y))
    assert model.params["early_stopping_rounds"] == 7
    assert model.calls[0][2]["eval_set"] == [(X, y)]
    assert "early_stopping_rounds" not in model.calls[0][2]
    assert model.calls[0][2]["verbose"] is False


def test_catboost_fit_keeps_validation_tuple_and_best_model():
    model = RecordingModel()
    X, y = np.ones((4, 2)), np.array([0, 1, 0, 1])
    adapter = CatBoostAdapter(params={}, fit_params={"early_stopping_rounds": 8})
    adapter.fit(model, X, y, (X, y))
    assert model.calls[0][2]["eval_set"] == (X, y)
    assert model.calls[0][2]["early_stopping_rounds"] == 8
    assert model.calls[0][2]["use_best_model"] is True


def test_lightgbm_creates_fresh_early_stopping_callbacks(monkeypatch):
    callbacks = []

    def early_stopping(rounds, verbose):
        callback = {"rounds": rounds, "verbose": verbose}
        callbacks.append(callback)
        return callback

    monkeypatch.setitem(sys.modules, "lightgbm", SimpleNamespace(early_stopping=early_stopping))
    X, y = np.ones((4, 2)), np.array([0, 1, 0, 1])
    adapter = LightGBMAdapter(params={}, early_stopping_rounds=9, early_stopping_metric="configured")
    for _ in range(2):
        model = RecordingModel()
        adapter.fit(model, X, y, (X, y))
        assert model.calls[0][2]["eval_set"] == [(X, y)]
        assert model.calls[0][2]["callbacks"][0]["rounds"] == 9
    assert callbacks[0] is not callbacks[1]


def sample_data(n_classes=3):
    rng = np.random.default_rng(2026)
    X = rng.normal(size=(120, 6)).astype(np.float32)
    signal = X[:, 0] + 0.3 * X[:, 2]
    y = np.digitize(signal, np.quantile(signal, [1 / 3, 2 / 3])) if n_classes == 3 else (signal > 0).astype(int)
    return X[:84], y[:84], X[84:], y[84:]


@pytest.mark.integration
@pytest.mark.parametrize("backend", ["catboost", "xgboost", "lightgbm"])
@pytest.mark.parametrize("n_classes", [2, 3])
def test_real_backend_selection_and_native_save_load(backend, n_classes, tmp_path):
    module = pytest.importorskip(backend)
    if backend == "catboost":
        params = dict(iterations=12, depth=3, learning_rate=0.1,
                      random_seed=42, thread_count=1, verbose=False, allow_writing_files=False)
        adapter = CatBoostAdapter(params=params)
    elif backend == "xgboost":
        params = dict(n_estimators=12, max_depth=3, learning_rate=0.1,
                      random_state=42, n_jobs=1, tree_method="hist")
        adapter = XGBoostAdapter(params=params)
    else:
        params = dict(n_estimators=12, max_depth=3, learning_rate=0.1,
                      random_state=42, n_jobs=1, verbosity=-1, min_child_samples=3)
        adapter = LightGBMAdapter(params=params)
    X, y, X_val, y_val = sample_data(n_classes)
    selector = RecursiveGroupSelector(model_adapter=adapter,
        quality_policy=MacroQualityPolicy(0, 0, 0, 1), min_groups=1)
    groups = {1: [0, 1], 2: [2, 3], 3: [4, 5]}
    selector.fit(X, y, eval_set=(X_val, y_val), feature_groups=groups,
                 feature_names=[f"f{i}" for i in range(6)])
    assert len(selector.selected_group_ids_) == 1
    assert len(selector.history_) == 2
    assert len(selector.shap_history_) == 3
    assert selector.adapter_.feature_count(selector.estimator_) == 2
    assert selector.predict(X_val).shape == y_val.shape
    assert adapter.params == params
    assert not hasattr(selector, 'estimator')
    assert selector.full_model_ is not selector.estimator_
    saved = selector.save(tmp_path / "selection")
    loaded = RecursiveGroupSelector.load(saved)
    np.testing.assert_array_equal(loaded.predict(X_val), selector.predict(X_val))
    assert loaded.selected_columns_ == selector.selected_columns_
    assert loaded.history_records() == selector.history_records()
    assert loaded.full_model_ is None
    with pytest.raises(FileExistsError):
        selector.save(saved)
    with pytest.raises(ValueError, match="original input columns"):
        loaded.predict(X_val[:, selector.selected_columns_])


@pytest.mark.integration
def test_supplied_catboost_baseline_is_not_retrained(tmp_path):
    module = pytest.importorskip("catboost")
    X, y, X_val, y_val = sample_data()
    params = dict(iterations=10, depth=2, learning_rate=0.1, random_seed=42,
                  thread_count=1, verbose=False, allow_writing_files=False)
    full = module.CatBoostClassifier(**params)
    full.fit(X, y, eval_set=(X_val, y_val), early_stopping_rounds=5, use_best_model=True)
    before = full.predict(X_val).copy()
    selector = RecursiveGroupSelector(model_adapter=CatBoostAdapter(params=params, early_stopping_metric='configured'),
                 quality_policy=MacroQualityPolicy(0, 0, 0, 1))
    selector.fit(X, y, eval_set=(X_val, y_val), feature_groups={1: [0, 1], 2: [2, 3], 3: [4, 5]}, full_model=full)
    assert selector.timings_["full_training_seconds"] == 0
    np.testing.assert_array_equal(full.predict(X_val), before)
    saved = selector.save(tmp_path / "selection")
    model_path = saved / "model.cbm"
    with model_path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="hash"):
        RecursiveGroupSelector.load(saved)
