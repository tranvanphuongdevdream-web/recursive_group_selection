from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from recursive_selection.benchmarking import ProcessingBenchmark, benchmark_prediction, collect_environment
from recursive_selection.evaluation import audit_selection, compare_evaluations, evaluate_classifier
from recursive_selection.reporting import BootstrapReport, SelectionReport, save_reports
from recursive_selection.evaluation.results import BootstrapComparison
from test_selection import make_selector


def selected(min_groups=1):
    X = np.tile(np.arange(1, 5, dtype=float), (12, 1))
    y = np.arange(12) % 2
    groups = {i: [i - 1] for i in range(1, 5)}
    selector = make_selector(min_groups=min_groups).fit(X, y, eval_set=(X, y), feature_groups=groups)
    return selector, groups


@pytest.mark.parametrize("copy_on_write", [False, True])
def test_reports_replay_retry_and_roundtrip(tmp_path, request, copy_on_write):
    # Pandas options are descriptors, not normal attributes. option_context also
    # restores correctly when pandas 3 deprecates the copy_on_write attribute.
    context = pd.option_context('mode.copy_on_write', copy_on_write)
    context.__enter__()
    request.addfinalizer(lambda: context.__exit__(None, None, None))
    selector, groups = selected()
    report = SelectionReport(selector.result_, feature_groups=groups, unit_name="sensor")
    assert report.trials["Decision"].tolist() == ["Rejected - try another", "Accepted", "Accepted",
                                                 "Rejected - try another", "Rejected - try another"]
    assert report.retry_group_ids == (4,)
    assert report.importance_matrix.shape == (4, 3)
    assert report.importance_history.groupby("Step").size().tolist() == [4, 3, 2]
    audit = audit_selection(selector.result_, feature_groups=groups, quality_policy=selector.quality_policy)
    assert len(audit.final_deletions) == 2
    assert audit.retry_configurations["First failed trial"].tolist() == [1]
    assert audit.retry_configurations["Later accepted trial"].tolist() == [3]
    files = report.save(tmp_path, prefix="demo")
    pd.testing.assert_frame_equal(pd.read_csv(files['demo_selection_history.csv'], keep_default_na=False),
                                  report.trials, check_dtype=False)
    assert json.loads(files['demo_selection_metadata.json'].read_text())['selected_group_ids'] == [1, 2]
    with pytest.raises(FileExistsError):
        report.save(tmp_path, prefix="demo")
    report.save(tmp_path, prefix="demo", overwrite=True)
    with pytest.raises(ValueError, match="override"):
        report.save(tmp_path, metadata={"selected_group_ids": [99]})


@pytest.mark.parametrize("minimum", [3, 4])
def test_audit_minimum_including_no_trials(minimum):
    selector, groups = selected(minimum)
    report = SelectionReport(selector.result_, feature_groups=groups)
    audit = audit_selection(selector.result_, feature_groups=groups, quality_policy=selector.quality_policy,
                            min_groups=minimum)
    assert audit.final_deletions.empty
    if minimum == 4:
        assert report.trials.empty and report.retry_evidence.empty


@pytest.mark.parametrize("field,value", [("group_id", 1), ("accepted", True),
    ("importance", 0.5), ("worst_margin", 0.2), ("change_from_full", None)])
def test_audit_detects_tampered_trials(field, value):
    selector, groups = selected()
    first = selector.history_[0]
    if field == "change_from_full":
        value = replace(first.change_from_full, f1_macro=0.5)
    bad = replace(selector.result_, history=(replace(first, **{field: value}), *selector.history_[1:]))
    with pytest.raises(ValueError, match="audit failed"):
        audit_selection(bad, feature_groups=groups, quality_policy=selector.quality_policy)


def test_unequal_noncontiguous_group_counts_and_no_mutation():
    selector, _ = selected()
    # Re-encode a real recorded run into unequal, noncontiguous input groups.
    groups = {1: [0, 5], 2: [1], 3: [2, 4], 4: [3]}
    result = replace(selector.result_, selected_columns=(0, 1, 5))
    report = SelectionReport(result, feature_groups=groups)
    assert report.trials['Features before'].tolist() == [6, 6, 4, 3, 3]
    assert report.trials['Features tried'].tolist() == [5, 4, 3, 2, 1]
    audit_selection(result, feature_groups=groups, quality_policy=selector.quality_policy)
    report.trials.loc[0, 'F1 macro'] = 123
    assert selector.history_[0].metrics.f1_macro != 123


class Predictor:
    classes_ = np.array(["z", "a"])

    def __init__(self):
        self.predict_calls = self.probability_calls = 0

    def predict(self, X):
        self.predict_calls += 1
        return np.array(["z", "a", "a", "z"])

    def predict_proba(self, X):
        self.probability_calls += 1
        return np.array([[.8, .2], [.1, .9], [.4, .6], [.7, .3]])


def evaluated(model=None, y=None):
    model = model or Predictor()
    return evaluate_classifier(model=model, splits={"Test": (np.ones((4, 2)),
        np.array(["z", "a", "a", "z"]) if y is None else y)})


def test_evaluation_unsorted_string_classes_and_reuse():
    model = Predictor()
    evaluation = evaluated(model)
    assert model.predict_calls == model.probability_calls == 1
    assert evaluation['Test'].metrics['Log loss'] == pytest.approx(-np.log([.8, .9, .6, .7]).mean())
    assert evaluation['Test'].classes == ('z', 'a')
    assert evaluation['Test'].metrics['F1 macro'] == 1
    assert not evaluation['Test'].predictions.flags.writeable
    comparison = compare_evaluations(reference=evaluation, candidate=evaluation)
    assert (comparison.metrics['Change'] == 0).all()
    assert comparison.class_metrics('Test')['Class'].tolist() == ['z', 'a']
    assert model.predict_calls == model.probability_calls == 1


def test_evaluation_without_probability_capability_and_missing_class():
    class LabelsOnly:
        classes_ = np.array([0, 1])
        def predict(self, X):
            return np.zeros(len(X), dtype=int)
    with pytest.raises(ValueError, match="predict_proba"):
        evaluate_classifier(model=LabelsOnly(), splits={'Test': (np.ones((4, 2)), np.zeros(4))})
    evaluation = evaluate_classifier(model=LabelsOnly(), include_probabilities=False,
        splits={'Test': (np.ones((4, 2)), np.zeros(4))})
    assert evaluation['Test'].metrics['F1 macro'] == .5
    assert np.isnan(evaluation['Test'].metrics['Log loss'])


@pytest.mark.parametrize("probabilities", [np.ones((4, 2)), np.full((4, 2), np.nan),
    np.ones((4, 1)), np.tile([-.1, 1.1], (4, 1))])
def test_invalid_probabilities_rejected(probabilities):
    class Bad(Predictor):
        def predict_proba(self, X):
            return probabilities
    with pytest.raises(ValueError, match="Probabilities"):
        evaluated(Bad())


def test_comparison_rejects_different_ordered_labels():
    with pytest.raises(ValueError, match="ordered labels"):
        compare_evaluations(reference=evaluated(), candidate=evaluated(y=np.array(['a', 'z', 'z', 'a'])))


def test_prediction_timing_excludes_probabilities(monkeypatch):
    import recursive_selection.evaluation.classification as module
    monkeypatch.setattr(module, 'perf_counter', iter([1., 1.25]).__next__)
    assert evaluated()['Test'].inference_seconds == .25


def test_processing_uses_recorded_times_without_predicting(tmp_path):
    model = Predictor()
    evaluation = evaluated(model)
    benchmark = ProcessingBenchmark.from_evaluations(reference=evaluation, candidate=evaluation,
        reference_feature_times={'Test': 2.}, candidate_feature_times={'Test': .5},
        reference_groups=2, candidate_groups=1)
    assert model.predict_calls == 1
    np.testing.assert_allclose(benchmark.metrics['Total processing (s)'],
        benchmark.metrics['Feature extraction (s)'] + benchmark.metrics['Model inference (s)'])
    assert benchmark.method['repeats'] == 1 and benchmark.method['warmup'] == 0
    benchmark.save(tmp_path)
    with pytest.raises(ValueError, match='exactly'):
        ProcessingBenchmark.from_evaluations(reference=evaluation, candidate=evaluation,
            reference_feature_times={}, candidate_feature_times={'Test': 0}, reference_groups=1, candidate_groups=1)


def test_explicit_repeat_benchmark():
    calls = []
    result = benchmark_prediction(lambda X: calls.append(len(X)), np.ones((4, 2)), repeats=3, warmup=2)
    assert calls == [4] * 5 and len(result.seconds) == 3 and result.warmup == 2
    with pytest.raises(ValueError):
        benchmark_prediction(lambda X: X, np.ones((4, 2)), repeats=True)


def test_export_collision_preflight_and_filename_validation(tmp_path):
    existing = tmp_path/'b.json'
    existing.write_text('original')
    with pytest.raises(FileExistsError):
        save_reports(tmp_path, documents={'a.json': {}, 'b.json': {}}, prefix='demo')
    assert not (tmp_path/'a.json').exists()
    assert existing.read_text() == 'original'
    with pytest.raises(ValueError):
        save_reports(tmp_path, documents={'../escape.json': {}})
    with pytest.raises(ValueError):
        save_reports(tmp_path, documents={'invalid.json': {'bad': float('nan')}})
    assert not (tmp_path/'invalid.json').exists()


def test_environment_fallback_when_machine_query_fails(monkeypatch):
    import recursive_selection.benchmarking.environment as module
    monkeypatch.setattr(module.subprocess, 'run', lambda *a, **k: (_ for _ in ()).throw(OSError()))
    environment = collect_environment(packages=('a-nonexistent-package-name',))
    assert environment.information['Package: a-nonexistent-package-name'] == 'Not installed'
    assert 'CPU:' in environment.caption
    assert 'Python' in environment.table['Information'].tolist()


def test_bootstrap_report_validates_scores_and_exports_explicit_margin(tmp_path):
    evaluation = evaluated()
    result = BootstrapComparison(reference_score=1., candidate_score=1.,
        bootstrap_differences=np.zeros(3), lower_bound=0., confidence_level=.95,
        n_resamples=3, random_state=42, batch_size=1000, n_samples=4, n_clusters=4,
        clusters_per_stratum=(2, 2), classes=('a', 'z'), stratified=True)
    report = BootstrapReport(result, reference=evaluation, candidate=evaluation, margin=.04)
    assert report.summary['one_sided_95_lower_pp'] == 0
    assert report.decision.passed
    files = report.save(tmp_path)
    assert json.loads(files['noninferiority_decision.json'].read_text())['margin'] == .04
    with pytest.raises(ValueError, match='scores/rows'):
        BootstrapReport(replace(result, candidate_score=.5), reference=evaluation, candidate=evaluation)
