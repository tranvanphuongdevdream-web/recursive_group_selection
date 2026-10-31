"""Baseline-compatible external folds, fixed F1, batch retries and median refits."""

from dataclasses import replace
import json

import numpy as np
import pytest

from recursive_selection import (BatchRemovalOptions, CatBoostAdapter, CVF1ReferenceQualityPolicy,
    F1ReferenceQualityPolicy, LightGBMAdapter, RecursiveFeatureSelector, RecursiveGroupSelector,
    XGBoostAdapter)
from recursive_selection.cross_validation import summarize_scores
from recursive_selection.evaluation import audit_selection
from recursive_selection.fixed_cv import median_rounds, validate_external_folds, validate_fixed_cv_trace
from recursive_selection.reporting import SelectionReport
from recursive_selection.results import MacroScores
from test_selection import ScriptedAdapter, TieRanker


class CVAdapter(ScriptedAdapter):
    def __init__(self):
        super().__init__()
        self.refit_calls = []

    def fit(self, model, X, y, eval_set):
        super().fit(model, X, y, eval_set)
        model.rounds = (4, 5, 6, 10)[(len(self.fit_calls)-1) % 4] + 3*(4-X.shape[1])

    def best_rounds(self, model):
        return model.rounds

    def refit(self, X, y, rounds):
        model = self.new_model()
        model.groups = tuple(map(int, X[0]))
        model.classes_, model.n_features_in_, model.rounds = np.unique(y), X.shape[1], rounds
        self.refit_calls.append((model.groups, len(X), rounds))
        return model


@pytest.fixture
def development():
    X = np.tile(np.arange(1, 5, dtype=float), (24, 1))
    y = np.arange(24) % 2
    splits = [(np.flatnonzero(np.arange(24)//6 != fold), np.arange(6*fold, 6*(fold+1)))
              for fold in range(4)]
    return X, y, splits, np.repeat(np.arange(12), 2)


def fit_scripted(development, fraction=.1, selector_class=RecursiveFeatureSelector):
    X, y, splits, groups = development
    adapter, ranker = CVAdapter(), TieRanker()
    selector = selector_class(model_adapter=adapter, ranker=ranker,
        quality_policy=CVF1ReferenceQualityPolicy(), speed_options=BatchRemovalOptions(fraction))
    options = {'feature_groups': {i+1: [i] for i in range(4)}} if selector_class is RecursiveGroupSelector else {}
    selector.fit_cv(X, y, cv_splits=splits, cv_groups=groups, **options)
    return selector, adapter, ranker


def test_gate_is_only_mean_f1_with_fixed_full_reference():
    policy = CVF1ReferenceQualityPolicy(reference='full')
    summary = lambda fs: summarize_scores([MacroScores(.1, .2, f) for f in fs])
    full = summary([.7, .9, .7, .9])
    floors = policy.thresholds(full)
    assert floors.precision_macro == floors.recall_macro == 0
    assert floors.f1_macro == pytest.approx(.8)
    assert policy.evaluate(summary([.6, 1, .8, .8]).mean, floors).passed
    assert not policy.evaluate(summary([.79]*4).mean, floors).passed
    # A previous accepted improvement must not raise the original reference.
    assert policy.evaluate(summary([.9]*4).mean, floors).passed
    assert policy.evaluate(summary([.81]*4).mean, floors).passed
    with pytest.raises(TypeError, match='CVScoreSummary'):
        policy.thresholds(full.mean)


@pytest.mark.parametrize('fraction', [0, .1, .5, 1])
@pytest.mark.parametrize('selector_class', [RecursiveFeatureSelector, RecursiveGroupSelector])
def test_external_cv_preserves_batch_backoff_retries_and_only_refits_accepted(
        development, fraction, selector_class):
    X, y, _, _ = development
    original = X.copy()
    selector, adapter, ranker = fit_scripted(development, fraction, selector_class)
    # Compare the complete removal/decision path with the existing single-Validation planner.
    old = selector_class(model_adapter=ScriptedAdapter(), ranker=TieRanker(),
        quality_policy=F1ReferenceQualityPolicy(), speed_options=BatchRemovalOptions(fraction))
    options = {'feature_groups': {i+1: [i] for i in range(4)}} if selector_class is RecursiveGroupSelector else {}
    old.fit(X, y, eval_set=(X, y), **options)
    assert [(t.removed_group_ids, t.accepted, t.configuration_id) for t in selector.history_] == [
        (t.removed_group_ids, t.accepted, t.configuration_id) for t in old.history_]
    assert selector.selected_group_ids_ == (1, 2)
    assert len(adapter.fit_calls) == 4*(1+len(selector.history_))
    assert len(adapter.refit_calls) == 1+sum(t.accepted for t in selector.history_)
    assert len({id(model) for model in adapter.models}) == len(adapter.models)
    expected_refits = ([((1, 2, 3, 4), 24, 6), ((1, 2), 24, 12)] if fraction == .5 else
                       [((1, 2, 3, 4), 24, 6), ((1, 2, 4), 24, 9), ((1, 2), 24, 12)])
    assert adapter.refit_calls == expected_refits
    assert ranker.states == [entry[0] for entry in expected_refits]
    assert selector.refit_rounds_ == 12
    assert all(len(summary.fold_scores) == 4 for summary in selector.cv_evaluation_.trials)
    np.testing.assert_array_equal(selector.predict(X), y)
    np.testing.assert_array_equal(X, original)
    audit_selection(selector.result_, feature_groups=selector.feature_groups_.as_dict(),
                    quality_policy=selector.quality_policy)
    report = SelectionReport(selector.result_, feature_groups=selector.feature_groups_.as_dict())
    assert report.score_source == 'Validation CV mean'
    assert report.gate_constraints.Metric.tolist() == ['F1 macro']
    assert len(report.cv_scores) == 4*(len(selector.history_)+1)
    assert 'Best boosting rounds' in report.cv_scores
    assert 'Development row' in report.cv_membership
    assert report.metadata()['score_source'] == 'Validation CV mean'


@pytest.mark.parametrize('bad', ['overlap', 'missing', 'negative', 'duplicate', 'float', 'bool', 'group_overlap', 'class_missing'])
def test_reject_bad_folds_before_any_model_fit(development, bad):
    X, y, splits, groups = development
    splits = [(a.copy(), b.copy()) for a, b in splits]
    if bad == 'overlap': splits[0] = (np.r_[splits[0][0], 0], splits[0][1])
    elif bad == 'missing': splits = splits[:3]
    elif bad == 'negative': splits[0][1][0] = -1
    elif bad == 'duplicate': splits[0][1][0] = splits[0][1][1]
    elif bad == 'float': splits[0] = (splits[0][0].astype(float)+.5, splits[0][1])
    elif bad == 'bool': splits[0] = (splits[0][0].astype(bool), splits[0][1])
    elif bad == 'group_overlap': groups = np.zeros(len(y), dtype=int)
    elif bad == 'class_missing': splits[0] = (np.flatnonzero(y == 1), np.flatnonzero(y == 0))
    adapter = CVAdapter()
    selector = RecursiveFeatureSelector(model_adapter=adapter, quality_policy=CVF1ReferenceQualityPolicy())
    with pytest.raises(ValueError): selector.fit_cv(X, y, cv_splits=splits, cv_groups=groups)
    assert not adapter.fit_calls and not adapter.refit_calls and not selector._is_fitted


def test_no_removal_minimum_and_protocol_guard(development):
    X, y, splits, groups = development
    adapter = CVAdapter()
    selector = RecursiveFeatureSelector(model_adapter=adapter, ranker=TieRanker(),
        quality_policy=CVF1ReferenceQualityPolicy(), min_features=4)
    with pytest.raises(ValueError, match='requires fit_cv'):
        selector.fit(X, y, eval_set=(X, y))
    selector.fit_cv(X, y, cv_splits=iter(splits), cv_groups=groups)
    assert not selector.history_ and len(adapter.fit_calls) == 4
    assert selector.refit_rounds_ == 6
    validate_fixed_cv_trace(selector.result_, selector.quality_policy)
    with pytest.raises(ValueError, match='requires CVF1'):
        RecursiveFeatureSelector(model_adapter=adapter).fit_cv(X, y, cv_splits=splits)


def test_fold_mean_is_not_pooled_out_of_fold_f1(development):
    X, y, _, groups = development
    class UnequalScoreAdapter(CVAdapter):
        def predict(self, model, matrix):
            return np.arange(len(matrix)) % 2 if len(matrix) == 4 else np.zeros(len(matrix), dtype=int)
    validations = [np.arange(0, 4), np.arange(4, 8), np.arange(8, 16), np.arange(16, 24)]
    splits = [(np.setdiff1d(np.arange(24), val), val) for val in validations]
    selector = RecursiveFeatureSelector(model_adapter=UnequalScoreAdapter(), ranker=TieRanker(),
        quality_policy=CVF1ReferenceQualityPolicy(), min_features=4)
    selector.fit_cv(X, y, cv_splits=splits, cv_groups=groups)
    assert selector.baseline_metrics_.f1_macro == pytest.approx((1+1+1/3+1/3)/4)
    pooled = MacroScores.from_predictions(y, np.r_[y[:8], np.zeros(16, dtype=int)], [0, 1])
    assert selector.baseline_metrics_.f1_macro != pytest.approx(pooled.f1_macro)


@pytest.mark.parametrize('reference', ['full', 'best'])
def test_selector_reference_after_a_better_accepted_trial(development, reference):
    X, y, splits, groups = development
    class ImprovedIntermediateAdapter(CVAdapter):
        def predict(self, model, matrix):
            if model.groups not in {(1, 2, 3, 4), (1, 2, 4), (1, 2)}:
                return np.zeros(len(matrix), dtype=int)
            predictions = np.arange(len(matrix)) % 2
            if model.groups != (1, 2, 4):
                predictions[0] = 1  # Full and final score below the intermediate perfect model.
            return predictions
    selector = RecursiveFeatureSelector(model_adapter=ImprovedIntermediateAdapter(),
        ranker=TieRanker(), quality_policy=CVF1ReferenceQualityPolicy(reference=reference))
    selector.fit_cv(X, y, cv_splits=splits, cv_groups=groups)
    assert selector.selected_group_ids_ == ((1, 2) if reference=='full' else (1, 2, 4))
    assert selector.accepted_history_[1].metrics.f1_macro == 1
    assert selector.final_metrics_.f1_macro == (selector.baseline_metrics_.f1_macro if reference=='full' else 1)
    assert selector.effective_floors_.f1_macro == selector.baseline_metrics_.f1_macro


@pytest.mark.parametrize('bad', ['mean', 'budget', 'best_rounds', 'fit_count', 'folds', 'final', 'gate'])
def test_audit_rejects_tampered_cv_evidence(development, bad):
    selector, _, _ = fit_scripted(development)
    result, trace = selector.result_, selector.cv_evaluation_
    if bad == 'mean': trace = replace(trace, baseline=replace(trace.baseline, mean=MacroScores(.2, .2, .2)))
    elif bad == 'budget': trace = replace(trace, refit_rounds=(1, *trace.refit_rounds[1:]))
    elif bad == 'best_rounds': trace = replace(trace, best_rounds=((1, 2), *trace.best_rounds[1:]))
    elif bad == 'fit_count': result = replace(result, timings={**result.timings, 'cv_fit_count': 0})
    elif bad == 'folds': trace = replace(trace, folds=trace.folds[:3])
    elif bad == 'final': trace = replace(trace, final=replace(trace.final, mean=MacroScores(0, 0, 0)))
    elif bad == 'gate': result = replace(result, gate_metrics=('precision_macro',))
    result = replace(result, cv_evaluation=trace)
    with pytest.raises(ValueError):
        audit_selection(result, feature_groups=selector.feature_groups_.as_dict(), quality_policy=selector.quality_policy)


@pytest.mark.parametrize('rounds, expected', [((80, 101, 120, 200), 111), ((1, 1), 1), ((2, 3, 4), 3)])
def test_median_uses_one_based_rounds_and_rounds_up(rounds, expected):
    assert median_rounds(rounds) == expected


@pytest.mark.integration
@pytest.mark.parametrize('backend', ['catboost', 'lightgbm', 'xgboost'])
@pytest.mark.parametrize('mode', ['feature', 'group'])
def test_native_external_cv_matches_manual_baseline_and_roundtrips(tmp_path, backend, mode):
    pytest.importorskip(backend)
    from sklearn.datasets import make_classification
    from sklearn.model_selection import StratifiedGroupKFold
    X, y = make_classification(n_samples=180, n_features=6, n_informative=4,
        n_redundant=0, n_classes=3, random_state=17)
    groups = np.repeat(np.arange(60), 3)
    splits = list(StratifiedGroupKFold(4, shuffle=True, random_state=42).split(X, y, groups))
    if backend == 'catboost':
        adapter = CatBoostAdapter(params=dict(iterations=16, depth=2, learning_rate=.1,
            loss_function='MultiClass', eval_metric='MultiClass', random_seed=42,
            thread_count=1, allow_writing_files=False, verbose=False))
    elif backend == 'lightgbm':
        adapter = LightGBMAdapter(params=dict(n_estimators=16, max_depth=2, num_leaves=4,
            min_child_samples=2, learning_rate=.1, metric='multi_logloss',
            random_state=42, n_jobs=1, verbosity=-1))
    else:
        adapter = XGBoostAdapter(params=dict(n_estimators=16, max_depth=2, learning_rate=.1,
            eval_metric='mlogloss', random_state=42, n_jobs=1))
    manual_scores, best = [], []
    for train, val in splits:
        model = adapter.new_model()
        adapter.fit(model, X[train], y[train], (X[val], y[val]))
        manual_scores.append(MacroScores.from_predictions(y[val], adapter.predict(model, X[val]), np.unique(y)))
        expected_rounds = (model.get_best_iteration()+1 if backend == 'catboost' else
                           model.best_iteration+1 if backend == 'xgboost' else model.best_iteration_)
        assert adapter.best_rounds(model) == expected_rounds
        best.append(expected_rounds)
    specification = adapter.specification()
    selector_class = RecursiveFeatureSelector if mode == 'feature' else RecursiveGroupSelector
    minimum = 5 if mode == 'feature' else 2
    selector = selector_class(model_adapter=adapter, quality_policy=CVF1ReferenceQualityPolicy(),
        speed_options=BatchRemovalOptions(.1), **({'min_features': minimum} if mode == 'feature' else {'min_groups': minimum}))
    selector.fit_cv(X, y, cv_splits=splits, cv_groups=groups,
                    feature_names=[f'f{i}' for i in range(6)],
                    **({'feature_groups': {5: [0, 2], 7: [1, 4], 9: [3, 5]}} if mode == 'group' else {}))
    assert selector.cv_evaluation_.baseline.fold_scores == tuple(manual_scores)
    assert selector.cv_evaluation_.best_rounds[0] == tuple(best)
    assert selector.cv_evaluation_.refit_rounds[0] == median_rounds(best)
    assert adapter.specification() == specification
    assert selector.estimator_.get_params().get('iterations' if backend == 'catboost' else 'n_estimators') == selector.refit_rounds_
    if backend == 'catboost':
        assert selector.estimator_.get_params()['use_best_model'] is False
    elif backend == 'xgboost':
        assert selector.estimator_.get_params()['early_stopping_rounds'] is None
        with pytest.raises(AttributeError):
            _ = selector.estimator_.best_iteration
    else:
        assert selector.estimator_.best_iteration_ == 0
    audit_selection(selector.result_, feature_groups=selector.feature_groups_.as_dict(),
                    quality_policy=selector.quality_policy, min_groups=minimum)
    report = SelectionReport(selector.result_, feature_groups=selector.feature_groups_.as_dict())
    report.save(tmp_path/'report')
    assert (tmp_path/'report/selection_cv_summary.csv').is_file()
    selector.save(tmp_path/'model')
    metadata = json.loads((tmp_path/'model/selection.json').read_text())
    assert metadata['format_version'] == 8 and metadata['policy_type'] == 'fixed_cv_f1'
    restored = selector_class.load(tmp_path/'model')
    assert restored.refit_rounds_ == selector.refit_rounds_
    assert restored.cv_evaluation_ == selector.cv_evaluation_
    np.testing.assert_array_equal(restored.predict(X), selector.predict(X))
    audit_selection(restored.result_, feature_groups=restored.feature_groups_.as_dict(),
                    quality_policy=restored.quality_policy, min_groups=minimum)
    # Persistence must validate refit-budget evidence, not merely model bytes.
    metadata['cv_evaluation']['refit_rounds'][0] += 1
    (tmp_path/'model/selection.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='Refit budget'):
        selector_class.load(tmp_path/'model')
