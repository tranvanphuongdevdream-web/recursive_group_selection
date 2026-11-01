"""External Train/Validation folds, best-so-far mean-F1 gate, and median refits.

This protocol intentionally uses each Validation fold both for early stopping
and scoring, matching caller-supplied baselines. It is distinct from the existing
Train-only fit/stopping/held-out protocol in cross_validation.py.
"""

from numbers import Integral
from time import perf_counter
from copy import copy, deepcopy
from concurrent.futures import ThreadPoolExecutor

from threadpoolctl import threadpool_limits

import numpy as np

from .acceleration import BatchRemovalOptions, RemovalPlanner
from .cross_validation import summarize_scores
from .groups import FeatureGroups, labels, numeric_matrix
from .policies import CVF1ReferenceQualityPolicy
from .results import (AcceptedState, CVScoreSummary, FixedCVEvaluationTrace,
                      MacroScores, SelectionResult, ShapSnapshot, TrialRecord,
                      ValidationFold)


def validate_external_folds(cv_splits, y, groups=None, *, fold_kind='partition'):
    """Validate supplied partitions or explicit repeated holdouts, without reshuffling."""
    if fold_kind not in ('partition', 'repeated'):
        raise ValueError('fold_kind must be partition or repeated.')
    if groups is not None:
        groups = np.asarray(groups)
        if groups.ndim != 1 or len(groups) != len(y):
            raise ValueError('cv_groups needs one group ID per development row.')
        if any(value is None or (isinstance(value, (float, np.floating))
                                and not np.isfinite(value)) for value in groups):
            raise ValueError('cv_groups must not contain missing/nonfinite IDs.')
        try:
            set(groups.tolist())
        except TypeError as error:
            raise ValueError('cv_groups must contain hashable scalar IDs.') from error
    folds, counts, classes = [], np.zeros(len(y), dtype=int), np.unique(y)
    for split in cv_splits:
        if isinstance(split, ValidationFold):
            split = (split.train_rows, split.validation_rows)
        if not isinstance(split, (tuple, list)) or len(split) != 2:
            raise ValueError('Each cv_split must be a (train_indices, validation_indices) pair.')
        parts = []
        for rows in split:
            rows = np.asarray(rows)
            if (rows.ndim != 1 or rows.dtype.kind not in 'iu' or not len(rows)
                    or len(np.unique(rows)) != len(rows)
                    or np.any(rows < 0) or np.any(rows >= len(y))):
                raise ValueError('Fold indices must be nonempty, unique, in-range integers.')
            if not np.array_equal(np.unique(y[rows]), classes):
                raise ValueError('Every fold Train/Validation partition needs all classes.')
            parts.append(rows.astype(int))
        train, validation = parts
        if not np.array_equal(np.sort(np.r_[train, validation]), np.arange(len(y))):
            raise ValueError('Each fold must partition development into disjoint Train/Validation rows.')
        if groups is not None and set(groups[train]) & set(groups[validation]):
            raise ValueError('Groups overlap between fold Train and Validation.')
        counts[validation] += 1
        folds.append(ValidationFold(tuple(map(int, train)), tuple(map(int, validation))))
    if len(folds) < 2:
        raise ValueError('Use at least two folds.')
    if fold_kind == 'partition' and not np.all(counts == 1):
        raise ValueError('Use at least two folds; every development row must be Validation exactly once.')
    if fold_kind == 'repeated' and len({tuple(sorted(f.validation_rows)) for f in folds}) != len(folds):
        raise ValueError('Repeated holdouts must have distinct Validation memberships.')
    return tuple(folds), groups


def median_rounds(rounds):
    if not rounds or any(isinstance(value, bool) or not isinstance(value, Integral)
                         or value < 1 for value in rounds):
        raise ValueError('Best boosting rounds must be positive one-based integers.')
    return int(np.ceil(np.median(rounds)))


def evaluate_external_folds(selector, X, y, folds, classes, trial_id=0, forbidden=()):
    adapter = selector.model_adapter
    scores, rounds, models = [], [], []
    training_seconds = prediction_seconds = 0.
    options = selector.speed_options
    jobs, threads = options.resources(len(folds), adapter) if options else (1, None)
    # A default sequential fit retains the adapter's original thread settings.
    managed = options is not None and (options.cv_jobs > 1 or options.cpu_budget is not None)

    def run_fold(model, fold, worker):
        train, val = np.asarray(fold.train_rows), np.asarray(fold.validation_rows)
        tick = perf_counter()
        worker.fit(model, X[train], y[train], (X[val], y[val]))
        training = perf_counter()-tick
        selector._check_model(worker, model, X.shape[1], classes)
        best = worker.best_rounds(model)
        median_rounds((best,))
        tick = perf_counter()
        score = MacroScores.from_predictions(y[val], worker.predict(model, X[val]), classes)
        return score, int(best), training, perf_counter()-tick

    # Callbacks/events stay on the caller thread; collection is fold-ordered.
    # Each fit has its own adapter configuration, model and stopping state.
    with threadpool_limits(limits=threads if managed else None), ThreadPoolExecutor(max_workers=jobs) as pool:
        pending = []
        for number, fold in enumerate(folds, 1):
            worker = adapter
            if managed:
                worker = copy(adapter)
                if hasattr(adapter, 'params'):
                    worker.params = deepcopy(adapter.params)
                if hasattr(adapter, 'fit_params'):
                    worker.fit_params = deepcopy(adapter.fit_params)
                if getattr(worker, 'backend_name', '') == 'catboost':
                    for config in (worker.params, worker.fit_params):
                        for key in ('verbose', 'verbose_eval', 'logging_level', 'silent'):
                            config.pop(key, None)
                    worker.params['verbose'] = worker.fit_params['verbose'] = False
            model = worker.new_model()
            if any(model is previous for previous in (*models, *forbidden)):
                raise ValueError('CV backend must return a fresh model for every fold.')
            models.append(model)
            if managed and getattr(worker, 'backend_name', '') in ('catboost', 'xgboost', 'lightgbm'):
                parameter_name = 'thread_count' if worker.backend_name == 'catboost' else 'n_jobs'
                thread_settings = {parameter_name: threads}
                model.set_params(**thread_settings)
            selector._emit('cv_fold_started', trial_id=trial_id, fold=number, folds=len(folds))
            # Sequential mode completes each fold before starting the next, retaining
            # historical event ordering and support for scripted adapters.
            future = pool.submit(run_fold, model, fold, worker) if managed else None
            pending.append((number, future, None if managed else run_fold(model, fold, worker)))
            if managed:
                continue
            score, best, training, predicting = pending[-1][2]
            training_seconds += training
            prediction_seconds += predicting
            rounds.append(int(best))
            scores.append(score)
            selector._emit('cv_fold_completed', trial_id=trial_id, fold=number,
                           folds=len(folds), scores=score.as_dict(), best_rounds=int(best))
        if managed:
            for number, future, _ in pending:
                score, best, training, predicting = future.result()
                training_seconds += training
                prediction_seconds += predicting
                scores.append(score)
                rounds.append(best)
                selector._emit('cv_fold_completed', trial_id=trial_id, fold=number,
                    folds=len(folds), scores=score.as_dict(), best_rounds=best)
    return summarize_scores(scores, training_seconds=training_seconds,
                            prediction_seconds=prediction_seconds), tuple(rounds)


def fit_fixed_cv(selector, X_dev, y_dev, *, cv_splits, cv_groups=None,
                 feature_groups=None, feature_names=None, fold_kind='partition'):
    selector._is_fitted = False
    started = perf_counter()
    if not isinstance(selector.quality_policy, CVF1ReferenceQualityPolicy):
        raise ValueError('fit_cv() requires CVF1ReferenceQualityPolicy.')
    if selector.speed_options is not None and not isinstance(selector.speed_options, BatchRemovalOptions):
        raise ValueError('fit_cv() supports BatchRemovalOptions; every trial completes all supplied folds.')
    adapter = selector.model_adapter
    if not all(callable(getattr(adapter, name, None)) for name in ('best_rounds', 'refit')):
        raise TypeError('fit_cv() adapters must implement best_rounds(model) and refit(X, y, rounds).')
    if feature_names is None and hasattr(X_dev, 'columns'):
        feature_names = list(X_dev.columns)
    names = tuple(feature_names) if feature_names is not None else None
    X = numeric_matrix(X_dev, name='X_dev', feature_names=names)
    if names is not None and (len(names) != X.shape[1]
            or any(not isinstance(name, str) for name in names) or len(set(names)) != len(names)):
        raise ValueError('feature_names must be unique strings matching the original columns.')
    y = labels(y_dev, name='y_dev', n_samples=len(X))
    classes = np.unique(y)
    if len(classes) < 2:
        raise ValueError('Development needs at least two classes.')
    if feature_groups is None:
        if selector._selection_mode == 'group':
            raise ValueError('Supply feature_groups for RecursiveGroupSelector.fit_cv().')
        feature_groups = {column+1: [column] for column in range(X.shape[1])}
    elif selector._selection_mode == 'feature':
        raise ValueError('RecursiveFeatureSelector uses singleton features; omit feature_groups.')
    groups = FeatureGroups(feature_groups, X.shape[1])
    if selector.min_groups > len(groups.group_ids):
        raise ValueError('Minimum selection count exceeds the supplied units.')
    folds, cv_groups = validate_external_folds(cv_splits, y, cv_groups, fold_kind=fold_kind)
    timings = dict(full_training_seconds=0., baseline_prediction_seconds=0., initial_shap_seconds=0.,
                   trial_training_seconds=0., validation_prediction_seconds=0., shap_update_seconds=0.,
                   trial_model_fit_count=0, cv_fit_count=0, refit_fit_count=0,
                   cv_baseline_training_seconds=0., cv_baseline_prediction_seconds=0.,
                   cv_trial_training_seconds=0., cv_trial_prediction_seconds=0.)
    selector._emit('run_started', group_ids=groups.group_ids)
    baseline, full_rounds = evaluate_external_folds(selector, X, y, folds, classes)
    timings.update(cv_fit_count=len(folds), cv_baseline_training_seconds=baseline.training_seconds,
                   cv_baseline_prediction_seconds=baseline.prediction_seconds)
    floors = selector.quality_policy.thresholds(baseline)
    current_floors = floors
    if selector.speed_options:
        jobs, threads = selector.speed_options.resources(len(folds), adapter)
        managed = selector.speed_options.cv_jobs > 1 or selector.speed_options.cpu_budget is not None
        parameter = 'thread_count' if getattr(adapter, 'backend_name', '') == 'catboost' else 'n_jobs'
        actual_threads = threads if managed else getattr(adapter, 'params', {}).get(parameter, 1)
        timings.update(cv_workers=jobs, cv_threads_per_model=actual_threads)

    def refit(matrix, best, forbidden=()):
        budget = median_rounds(best)
        tick = perf_counter()
        model = adapter.refit(matrix, y, budget)
        seconds = perf_counter()-tick
        if any(model is previous for previous in forbidden):
            raise ValueError('Development refit must return a fresh model.')
        selector._check_model(adapter, model, matrix.shape[1], classes)
        return model, budget, seconds

    full_model, budget, seconds = refit(X, full_rounds)
    timings.update(full_training_seconds=seconds, refit_fit_count=1)
    refit_budgets = [budget]
    current_model, current_groups, current_summary = full_model, groups.group_ids, baseline
    planner = RemovalPlanner(selector.min_groups,
                             selector.speed_options.removal_fraction if selector.speed_options else 0.)
    tick = perf_counter()
    ranking = selector._rank(adapter, current_model, X, y, groups, current_groups)
    timings['initial_shap_seconds'] = perf_counter()-tick
    initial_ranking = dict(ranking)
    accepted = [AcceptedState(0, 0, current_groups, baseline.mean)]
    snapshots = [ShapSnapshot(0, 0, current_groups, tuple(ranking.items()))]
    history, summaries, best_rounds, attempted = [], [], [full_rounds], set()
    selector._emit('initialized', baseline=baseline.mean.as_dict(), floors=floors.as_dict(),
                   ranking=ranking, score_source='validation_cv')
    loop_started, configuration = perf_counter(), 0
    while True:
        if len(current_groups) <= selector.min_groups:
            stop_reason = 'min_groups_reached'
            break
        removed = planner.choose(current_groups, ranking)
        if not removed:
            stop_reason = 'all_candidates_rejected'
            break
        proposed = tuple(group for group in current_groups if group not in removed)
        matrix = X[:, groups.columns(proposed)]
        trial_id = len(history)+1
        selector._emit('trial_started', trial_id=trial_id, configuration_id=configuration,
                       group_id=removed[0], removed_group_ids=removed,
                       groups_before=current_groups, proposed_groups=proposed)
        summary, rounds = evaluate_external_folds(selector, matrix, y, folds, classes,
                                                  trial_id, (full_model, current_model))
        summaries.append(summary)
        best_rounds.append(rounds)
        timings['cv_fit_count'] += len(folds)
        timings['cv_trial_training_seconds'] += summary.training_seconds
        timings['cv_trial_prediction_seconds'] += summary.prediction_seconds
        before_f1 = current_floors.f1_macro
        decision = selector.quality_policy.evaluate(summary.mean, current_floors)
        model, refit_seconds = None, 0.
        if decision.passed:
            model, budget, refit_seconds = refit(matrix, rounds, (full_model, current_model))
            refit_budgets.append(budget)
            timings['trial_training_seconds'] += refit_seconds
            timings['trial_model_fit_count'] += 1
            timings['refit_fit_count'] += 1
            current_floors = selector.quality_policy.advance(current_floors, summary.mean)
        change = MacroScores(*(getattr(summary.mean, key)-getattr(baseline.mean, key)
                               for key in baseline.mean.as_dict()))
        trial = TrialRecord(trial_id, configuration, removed[0], current_groups, proposed,
                            float(np.mean([ranking[group] for group in removed])), summary.mean,
                            change, decision.passed, decision.worst_margin, decision.failed_metrics,
                            summary.training_seconds+refit_seconds, summary.prediction_seconds,
                            removed if selector.speed_options else (), 'mean',
                            before_f1, current_floors.f1_macro)
        history.append(trial)
        attempted.update(removed)
        selector._emit('trial_completed', **trial.as_dict())
        planner.observe(removed, decision.passed)
        if not decision.passed:
            continue
        current_model, current_groups, current_summary = model, proposed, summary
        configuration += 1
        accepted.append(AcceptedState(configuration, trial_id, current_groups, summary.mean))
        tick = perf_counter()
        ranking = selector._rank(adapter, current_model, matrix, y, groups, current_groups)
        timings['shap_update_seconds'] += perf_counter()-tick
        snapshots.append(ShapSnapshot(configuration, trial_id, current_groups, tuple(ranking.items())))
        selector._emit('configuration_accepted', configuration_id=configuration, trial_id=trial_id,
                       group_ids=current_groups, ranking=ranking, f1_reference=current_floors.f1_macro)
    timings['elimination_loop_seconds'] = perf_counter()-loop_started
    timings['total_fit_seconds'] = perf_counter()-started
    columns = groups.columns(current_groups)
    trace = FixedCVEvaluationTrace(folds, cv_groups is not None, baseline, tuple(summaries),
        current_summary, tuple(best_rounds), tuple(refit_budgets), tuple(y.tolist()),
        tuple(cv_groups.tolist()) if cv_groups is not None else None, selector.quality_policy.reference,
        fold_kind, getattr(adapter, 'early_stopping_metric', 'custom'))
    result = SelectionResult(current_model, current_groups, columns,
        tuple(names[column] for column in columns) if names is not None else None,
        baseline.mean, floors, current_summary.mean, tuple(history), tuple(accepted),
        tuple(snapshots), dict(timings), stop_reason, trace, selector.speed_options,
        selector.quality_policy.gate_metrics)
    selector.adapter_, selector.full_model_, selector.classes_ = adapter, full_model, classes.copy()
    selector.n_features_in_, selector.feature_names_in_, selector.feature_groups_ = X.shape[1], names, groups
    selector.initial_ranking_, selector.final_ranking_ = initial_ranking, dict(ranking)
    selector.attempted_group_ids_ = tuple(sorted(attempted))
    selector.rejected_group_ids_ = tuple(sorted(planner.failed))
    selector._emit('run_finished', group_ids=current_groups, stop_reason=stop_reason,
                   trial_count=len(history), shap_calls=len(snapshots))
    selector._set_result(result)
    return selector


def validate_fixed_cv_trace(result, policy):
    """Replay all-fold means, fixed thresholds, refit budgets and fit counts."""
    trace = result.cv_evaluation
    if not isinstance(policy, CVF1ReferenceQualityPolicy) or not isinstance(trace, FixedCVEvaluationTrace):
        raise ValueError('External Validation CV requires its own policy and trace.')
    if trace.grouped != (trace.group_ids is not None):
        raise ValueError('CV grouping metadata is inconsistent.')
    if trace.reference != policy.reference:
        raise ValueError('CV reference mode differs from the quality policy.')
    validate_external_folds(trace.folds, np.asarray(trace.labels),
                            np.asarray(trace.group_ids) if trace.grouped else None,
                            fold_kind=trace.fold_kind)
    if trace.checkpoint_metric not in ('configured', 'accuracy', 'custom'):
        raise ValueError('Unknown CV checkpoint metric.')
    if result.recording_split is not None:
        split = result.recording_split.validate()
        if (trace.fold_kind != 'repeated' or split.folds != trace.folds
                or tuple(np.asarray(split.model_labels)[list(split.development_rows)]) != trace.labels
                or split.development_groups != trace.group_ids):
            raise ValueError('Recording split differs from the fitted development folds/labels/groups.')
    if len(trace.trials) != len(result.history) or len(trace.best_rounds) != len(result.history)+1:
        raise ValueError('CV trial or best-round history length differs.')
    def same(a, b):
        return np.allclose(list(a.as_dict().values()), list(b.as_dict().values()), atol=1e-12, rtol=0)
    for summary, rounds in zip((trace.baseline, *trace.trials), trace.best_rounds):
        if not isinstance(summary, CVScoreSummary) or len(summary.fold_scores) != len(trace.folds):
            raise ValueError('Every candidate must complete all external folds.')
        expected = summarize_scores(summary.fold_scores)
        if any(not same(getattr(summary, key), getattr(expected, key))
               for key in ('mean', 'standard_deviation', 'standard_error')):
            raise ValueError('CV mean/SD/SE differs from recorded fold scores.')
        if len(rounds) != len(trace.folds):
            raise ValueError('CV best-round count differs from fold membership.')
        median_rounds(rounds)
        if any(not np.isfinite(value) or value < 0 for value in
               (summary.training_seconds, summary.prediction_seconds)):
            raise ValueError('Invalid CV timing.')
    if not same(result.baseline_metrics, trace.baseline.mean) or not same(
            result.effective_floors, policy.thresholds(trace.baseline)):
        raise ValueError('CV reference or initial Full F1 threshold differs.')
    selected, budgets = trace.baseline, [median_rounds(trace.best_rounds[0])]
    floors = policy.thresholds(trace.baseline)
    for trial, summary, rounds in zip(result.history, trace.trials, trace.best_rounds[1:]):
        if not same(trial.metrics, summary.mean) or trial.metric_kind != 'mean':
            raise ValueError('Trial metrics must be the complete fold mean.')
        decision = policy.evaluate(summary.mean, floors)
        if (trial.accepted != decision.passed or trial.failed_metrics != decision.failed_metrics
                or not np.isclose(trial.worst_margin, decision.worst_margin, rtol=0, atol=1e-12)):
            raise ValueError('Trial decision differs from the running F1 reference.')
        before = floors.f1_macro
        if trial.accepted:
            floors = policy.advance(floors, summary.mean)
        if trial.f1_reference_before is None or trial.f1_reference_after is None:
            if policy.reference != 'full':
                raise ValueError('Missing best-so-far F1 reference evidence.')
        elif not np.allclose([trial.f1_reference_before, trial.f1_reference_after],
                             [before, floors.f1_macro], rtol=0, atol=1e-12):
            raise ValueError('Recorded F1 reference transition differs.')
        if trial.accepted:
            selected = summary
            budgets.append(median_rounds(rounds))
    if selected != trace.final or not same(result.final_metrics, trace.final.mean):
        raise ValueError('Final CV summary differs from the last accepted trial.')
    if tuple(budgets) != trace.refit_rounds:
        raise ValueError('Refit budget differs from ceiling median of fold best rounds.')
    expected_fits = len(trace.folds)*(1+len(result.history))
    if (result.timings.get('cv_fit_count') != expected_fits
            or result.timings.get('refit_fit_count') != len(budgets)
            or result.timings.get('trial_model_fit_count') != len(budgets)-1):
        raise ValueError('Recorded CV/refit fit counts differ from history.')
