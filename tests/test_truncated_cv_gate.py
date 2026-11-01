"""Fixed Full F1 buckets, complete-fold means, audits and native persistence."""
from dataclasses import replace
import json

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold

from recursive_selection import (BatchRemovalOptions, CatBoostAdapter,
    CVF1ReferenceQualityPolicy, RecursiveFeatureSelector)
from recursive_selection.cross_validation import summarize_scores
from recursive_selection.evaluation import audit_selection
from recursive_selection.fixed_cv import validate_fixed_cv_trace
from recursive_selection.reporting import SelectionReport
from recursive_selection.results import MacroScores
from test_fixed_cv import CVAdapter
from test_selection import TieRanker


def summary(values):
    return summarize_scores([MacroScores(.1, .2, value) for value in values])


def test_fixed_full_gate_accepts_same_bucket_and_never_tracks_improvements():
    policy = CVF1ReferenceQualityPolicy(reference='full', truncate_decimals=2)
    full = summary([.689989]*4)
    floor = policy.thresholds(full)
    assert floor.f1_macro == .68
    for value, passed in [(.708418, True), (.688421, True), (.680000, True),
                          (.679999, False), (.700000, True)]:
        trial = summary([value]*4)
        assert policy.evaluate(trial.mean, floor).passed is passed
        if passed:
            floor = policy.advance(floor, trial.mean)
        assert floor.f1_macro == .68
    assert full.mean.f1_macro == .689989
    # Truncation compares buckets strictly, even if a raw-score tolerance is set.
    loose = replace(policy, tolerance=.1)
    assert not loose.evaluate(summary([.679999]*4).mean, floor).passed


def test_truncate_after_mean_and_decimal_boundaries():
    policy = CVF1ReferenceQualityPolicy(reference='full', truncate_decimals=2)
    trial = summary([.6899, .6899, .6999, .6999])
    floor = policy.thresholds(summary([.6999]*4))
    assert policy.evaluate(trial.mean, floor).passed
    assert policy.gate_value(trial.mean.f1_macro) == .69
    assert policy.gate_value(np.mean([policy.gate_value(s.f1_macro) for s in trial.fold_scores])) == .68
    assert [policy.gate_value(x) for x in (0., .58, .68, .70, 1.)] == [0., .58, .68, .70, 1.]
    assert policy.gate_value(.688421) == .68
    assert CVF1ReferenceQualityPolicy().gate_value(.688421) == .688421
    best = replace(policy, reference='best')
    assert best.advance(floor, summary([.708418]*4).mean).f1_macro == .70


@pytest.mark.parametrize('value', [-1, 16, True, 2.0, '2'])
def test_invalid_precision(value):
    with pytest.raises(ValueError, match='truncate_decimals'):
        CVF1ReferenceQualityPolicy(truncate_decimals=value)


@pytest.mark.parametrize('value', [-.01, 1.01, np.nan, np.inf, True])
def test_invalid_gate_score(value):
    with pytest.raises(ValueError, match='F1'):
        CVF1ReferenceQualityPolicy(truncate_decimals=2).gate_value(value)


def test_final_raw_f1_can_fall_below_full_and_best_with_valid_audit():
    class BucketAdapter(CVAdapter):
        def predict(self, model, matrix):
            error_rate = {(1,2,3,4): .312, (1,2,4): .292, (1,2): .318}.get(model.groups, .4)
            predictions = np.arange(len(matrix)) % 2
            predictions[:round(error_rate*len(matrix))] ^= 1
            return predictions

    X = np.tile(np.arange(1,5,dtype=float), (4000,1))
    y = np.arange(len(X)) % 2
    folds = [(np.flatnonzero(np.arange(len(X))//1000 != f), np.arange(1000*f,1000*(f+1)))
             for f in range(4)]
    policy = CVF1ReferenceQualityPolicy(reference='full', truncate_decimals=2)
    adapter, ranker = BucketAdapter(), TieRanker()
    selector = RecursiveFeatureSelector(model_adapter=adapter, ranker=ranker,
        quality_policy=policy, speed_options=BatchRemovalOptions(.1))
    selector.fit_cv(X,y,cv_splits=folds)
    assert selector.selected_group_ids_ == (1,2)
    assert selector.baseline_metrics_.f1_macro == pytest.approx(.688)
    assert selector.best_f1_ == pytest.approx(.708)
    assert selector.final_metrics_.f1_macro == pytest.approx(.682)
    assert len(adapter.refit_calls) == len(ranker.states) == 3
    assert len(adapter.fit_calls) == 4*(1+len(selector.history_))
    assert all(t.f1_reference_before == t.f1_reference_after == .68 for t in selector.history_)
    audit_selection(selector.result_,feature_groups=selector.feature_groups_.as_dict(),quality_policy=policy)
    report = SelectionReport(selector.result_,feature_groups=selector.feature_groups_.as_dict())
    assert set(report.f1_reference_history['Full F1 gate']) == {.68}
    assert report.cv_summary.loc[report.cv_summary.Trial.eq(0),'F1 reference before'].item() == .68
    assert report.gate_constraints['Selected CV mean'].item() == pytest.approx(.682)
    assert report.gate_constraints['Selected F1 gate'].item() == .68
    assert report.gate_constraints.Margin.item() == 0
    assert report.metadata()['truncate_decimals'] == 2
    changed = replace(selector.result_,cv_evaluation=replace(selector.cv_evaluation_,truncate_decimals=1))
    with pytest.raises(ValueError, match='precision'):
        validate_fixed_cv_trace(changed,policy)


@pytest.mark.integration
def test_catboost_truncated_gate_roundtrip_and_precision_provenance(tmp_path):
    pytest.importorskip('catboost')
    X,y = make_classification(n_samples=120,n_features=6,n_informative=4,
        n_redundant=0,n_classes=3,random_state=17)
    folds = list(StratifiedKFold(4,shuffle=True,random_state=42).split(X,y))
    policy = CVF1ReferenceQualityPolicy(reference='full',truncate_decimals=2)
    selector = RecursiveFeatureSelector(model_adapter=CatBoostAdapter(params=dict(
        iterations=10,depth=2,random_seed=42,thread_count=1,verbose=False,allow_writing_files=False)),
        quality_policy=policy,min_features=5)
    selector.fit_cv(X,y,cv_splits=folds)
    selector.save(tmp_path/'bundle')
    path = tmp_path/'bundle/selection.json'
    metadata = json.loads(path.read_text())
    assert metadata['format_version'] == 10
    assert metadata['policy']['truncate_decimals'] == metadata['cv_evaluation']['truncate_decimals'] == 2
    restored = RecursiveFeatureSelector.load(tmp_path/'bundle')
    assert restored.quality_policy == policy
    assert restored.cv_evaluation_ == selector.cv_evaluation_
    np.testing.assert_array_equal(restored.predict(X),selector.predict(X))
    audit_selection(restored.result_,feature_groups=restored.feature_groups_.as_dict(),quality_policy=policy,min_groups=5)
    report = SelectionReport(restored.result_,feature_groups=restored.feature_groups_.as_dict())
    report.save(tmp_path/'report')
    for location in ('policy','cv_evaluation'):
        bad = json.loads(json.dumps(metadata))
        bad[location].pop('truncate_decimals')
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError,match='precision'):
            RecursiveFeatureSelector.load(tmp_path/'bundle')
    bad = json.loads(json.dumps(metadata))
    bad['cv_evaluation']['truncate_decimals'] = 1
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='precision'):
        RecursiveFeatureSelector.load(tmp_path/'bundle')
