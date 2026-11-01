"""External stopping, multiclass attribution, CV and trusted GBC persistence."""
import json

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, train_test_split

from recursive_selection import (SklearnGBCAdapter, RecursiveFeatureSelector,
    RecursiveGroupSelector, MacroQualityPolicy, CVF1ReferenceQualityPolicy,
    BatchRemovalOptions, CVReferenceQualityPolicy, SelectionSpeedOptions)
from recursive_selection.evaluation import audit_selection
from recursive_selection.results import MacroScores


def data(n_classes=3):
    X, y = make_classification(n_samples=120, n_features=6, n_informative=4,
        n_redundant=0, n_classes=n_classes, random_state=17)
    return train_test_split(X, y, test_size=.3, stratify=y, random_state=42)


@pytest.mark.parametrize('n_classes', [2, 3])
@pytest.mark.parametrize('subsample', [1., .7])
def test_external_checkpoint_matches_independent_staged_loss(n_classes, subsample):
    rng = np.random.default_rng(11)
    X, X_val = rng.normal(size=(100, 5)), rng.normal(size=(80, 5))
    y, y_val = np.arange(100) % n_classes, np.arange(80) % n_classes
    params = dict(n_estimators=60, learning_rate=.3, max_depth=2,
                  random_state=42, subsample=subsample)
    full = GradientBoostingClassifier(**params).fit(X, y)
    losses = [log_loss(y_val, p, labels=full.classes_) for p in full.staged_predict_proba(X_val)]
    best_loss, best, evaluated = np.inf, 0, 0
    for iteration, loss in enumerate(losses, 1):
        evaluated = iteration
        if loss < best_loss:
            best_loss, best = loss, iteration
        if iteration - best >= 3:
            break
    assert evaluated < params['n_estimators']
    adapter = SklearnGBCAdapter(params=params, early_stopping_rounds=3, early_stopping_metric="configured")
    model = adapter.new_model()
    adapter.fit(model, X, y, (X_val, y_val))
    assert adapter.best_rounds(model) == best == model.n_estimators_
    assert model.best_iteration_ == best - 1
    assert model.n_iter_no_change is None
    assert model.n_estimators == params['n_estimators']
    np.testing.assert_allclose(model.validation_log_loss_, losses[:evaluated], rtol=0, atol=1e-14)
    expected = list(full.staged_predict_proba(X_val))[best-1]
    np.testing.assert_array_equal(model.predict_proba(X_val), expected)
    np.testing.assert_array_equal(model.train_score_, full.train_score_[:best])
    if subsample < 1:
        np.testing.assert_array_equal(model.oob_scores_, full.oob_scores_[:best])
        assert model.oob_score_ == model.oob_scores_[-1]


def test_monitor_advances_one_generator_and_cap_still_restores_best(monkeypatch):
    X, X_val, y, y_val = data()
    adapter = SklearnGBCAdapter(params=dict(n_estimators=6, random_state=42), early_stopping_rounds=20, early_stopping_metric="configured")
    model = adapter.new_model()
    original = model.staged_predict_proba
    calls = []
    def stages(values):
        calls.append(len(values))
        yield from original(values)
    monkeypatch.setattr(model, 'staged_predict_proba', stages)
    adapter.fit(model, X, y, (X_val, y_val))
    assert calls == [len(X_val)]
    assert model.n_estimators_ == np.argmin(model.validation_log_loss_) + 1
    assert len(model.validation_log_loss_) == 6


@pytest.mark.parametrize('n_classes', [2, 3])
@pytest.mark.parametrize('init', [None, 'zero'])
def test_signed_tree_shap_sums_to_raw_margin_differences(n_classes, init):
    shap = pytest.importorskip('shap')
    X, X_val, y, y_val = data(n_classes)
    adapter = SklearnGBCAdapter(params=dict(n_estimators=7, max_depth=2,
        learning_rate=.07, init=init, random_state=42), early_stopping_rounds=None)
    model = adapter.new_model()
    adapter.fit(model, X, y, (X_val, y_val))
    values = adapter.feature_shap(model, X_val, y_val)
    outputs = 1 if n_classes == 2 else n_classes
    assert values.shape == (len(X_val), outputs, X.shape[1])
    margin = np.asarray(model.decision_function(X_val)).reshape(len(X_val), outputs)
    np.testing.assert_allclose(values.sum(axis=-1)-values[0].sum(axis=-1),
                               margin-margin[0], atol=1e-9, rtol=1e-8)
    if n_classes == 2 and init is None:
        direct = shap.TreeExplainer(model, feature_perturbation='tree_path_dependent').shap_values(X_val)
        np.testing.assert_allclose(values[:,0,:], direct, atol=1e-9, rtol=1e-8)


def test_refit_uses_exact_budget_without_validation_or_stopping():
    X, X_val, y, y_val = data()
    adapter = SklearnGBCAdapter(params=dict(n_estimators=50, random_state=42),
        fit_params=dict(early_stopping_rounds=2, sample_weight=np.ones(len(y))))
    model = adapter.refit(X, y, 9)
    assert model.n_estimators == model.n_estimators_ == 9
    assert model.n_iter_no_change is None
    assert not hasattr(model, 'validation_log_loss_')
    assert adapter.params['n_estimators'] == 50
    for invalid in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match='positive integer'):
            adapter.refit(X, y, invalid)


@pytest.mark.parametrize('params,match', [
    ({'n_iter_no_change': 5}, 'internal split'),
    ({'warm_start': True}, 'Warm-start'),
    ({'init': GradientBoostingClassifier()}, 'constant init'),
])
def test_reject_hidden_splits_warm_start_and_nonconstant_init(params, match):
    with pytest.raises(ValueError, match=match):
        SklearnGBCAdapter(params=params)


def test_reject_monitor_and_class_incomplete_validation():
    with pytest.raises(ValueError, match='monitor is managed'):
        SklearnGBCAdapter(fit_params={'monitor': object()})
    X, X_val, y, y_val = data()
    adapter = SklearnGBCAdapter()
    with pytest.raises(ValueError, match='same classes'):
        adapter.fit(adapter.new_model(), X, y, (X_val[y_val == 0], y_val[y_val == 0]))


@pytest.mark.integration
@pytest.mark.parametrize('mode', ['feature', 'group'])
@pytest.mark.parametrize('n_classes', [2, 3])
def test_selection_and_trusted_model_roundtrip(tmp_path, mode, n_classes):
    pytest.importorskip('shap')
    X, X_val, y, y_val = data(n_classes)
    adapter = SklearnGBCAdapter(params=dict(n_estimators=8, max_depth=2, random_state=42))
    cls = RecursiveFeatureSelector if mode == 'feature' else RecursiveGroupSelector
    selector = cls(model_adapter=adapter, quality_policy=MacroQualityPolicy(0,0,0,1))
    options = {} if mode == 'feature' else dict(feature_groups={1:[0,1],2:[2,3],3:[4,5]})
    selector.fit(X, y, eval_set=(X_val,y_val), **options)
    saved = selector.save(tmp_path/'model')
    with pytest.raises(ValueError, match='trusted=True'):
        cls.load(saved)
    loaded = cls.load(saved, trusted=True)
    np.testing.assert_array_equal(loaded.predict(X_val), selector.predict(X_val))
    assert loaded.adapter_.specification() == adapter.specification()
    assert loaded.history_records() == selector.history_records()
    model_file = saved/'model.joblib'
    model_file.write_bytes(model_file.read_bytes()+b'tampered')
    with pytest.raises(ValueError, match='hash'):
        cls.load(saved, trusted=True)


@pytest.mark.integration
@pytest.mark.parametrize('mode', ['feature', 'group'])
def test_fixed_cv_scores_median_refit_audit_and_persistence(tmp_path, mode):
    pytest.importorskip('shap')
    X, y = make_classification(n_samples=120, n_features=6, n_informative=4,
        n_redundant=0, n_classes=3, random_state=17)
    splits = list(StratifiedKFold(4, shuffle=True, random_state=42).split(X,y))
    params = dict(n_estimators=10, max_depth=2, random_state=42)
    adapter = SklearnGBCAdapter(params=params, early_stopping_metric="configured")
    scores, best = [], []
    # Independent sklearn fits and public staged predictions, not adapter.fit.
    for train, val in splits:
        model = GradientBoostingClassifier(**params).fit(X[train], y[train])
        probabilities = list(model.staged_predict_proba(X[val]))
        losses = [log_loss(y[val], p, labels=model.classes_) for p in probabilities]
        best_loss, rounds = np.inf, 0
        for i, loss in enumerate(losses,1):
            if loss < best_loss:best_loss, rounds = loss, i
            if i-rounds >= 5:break
        pred = model.classes_[probabilities[rounds-1].argmax(axis=1)]
        scores.append(MacroScores.from_predictions(y[val],pred,np.unique(y)))
        best.append(rounds)
    cls = RecursiveFeatureSelector if mode == 'feature' else RecursiveGroupSelector
    options = {'min_features':5} if mode == 'feature' else {'min_groups':2}
    selector = cls(model_adapter=adapter, quality_policy=CVF1ReferenceQualityPolicy(),
        speed_options=BatchRemovalOptions(.1), **options)
    options = {} if mode == 'feature' else dict(feature_groups={1:[0,1],2:[2,3],3:[4,5]})
    selector.fit_cv(X,y,cv_splits=splits,**options)
    assert selector.cv_evaluation_.baseline.fold_scores == tuple(scores)
    assert selector.cv_evaluation_.best_rounds[0] == tuple(best)
    assert selector.full_model_.n_estimators_ == int(np.ceil(np.median(best)))
    assert selector.estimator_.n_estimators_ == selector.refit_rounds_
    assert not hasattr(selector.estimator_,'validation_log_loss_')
    audit_selection(selector.result_, feature_groups=selector.feature_groups_.as_dict(),
                    quality_policy=selector.quality_policy, min_groups=selector.min_groups)
    selector.save(tmp_path/'cv')
    restored = cls.load(tmp_path/'cv', trusted=True)
    assert restored.cv_evaluation_ == selector.cv_evaluation_
    np.testing.assert_array_equal(restored.predict(X), selector.predict(X))
    metadata = json.loads((tmp_path/'cv/selection.json').read_text())
    assert metadata['backend']['name'] == 'sklearn_gbc'


@pytest.mark.integration
def test_train_only_cv_speed_options_work_without_n_jobs_parameter():
    pytest.importorskip('shap')
    X, X_val, y, y_val = data()
    selector = RecursiveFeatureSelector(
        model_adapter=SklearnGBCAdapter(params=dict(n_estimators=5,max_depth=1,random_state=42)),
        quality_policy=CVReferenceQualityPolicy(n_splits=2), min_features=5,
        speed_options=SelectionSpeedOptions(cv_jobs=2,removal_fraction=.1))
    selector.fit(X,y,eval_set=(X_val,y_val))
    assert selector.cv_evaluation_ is not None
    assert selector.predict(X_val).shape == y_val.shape
