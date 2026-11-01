"""Shared recursive elimination engine with group and individual-feature APIs."""

from copy import deepcopy
from numbers import Integral
from time import perf_counter

import numpy as np

from .groups import FeatureGroups, labels, numeric_matrix
from .acceleration import SelectionSpeedOptions, BatchRemovalOptions, RemovalPlanner
from .importance import MeanAbsoluteShapRanker
from .policies import MacroQualityPolicy, CVReferenceQualityPolicy, CVF1ReferenceQualityPolicy
from .cross_validation import make_folds, evaluate_folds
from .results import (
    AcceptedState,
    MacroScores,
    SelectionResult,
    ShapSnapshot,
    TrialRecord,
    CVEvaluationTrace,
    CVPrunedEvaluation,
)


class _RecursiveSelectorBase:
    """Coordinate refits and configuration-specific retries for selection units."""

    _selection_mode = "group"

    def __init__(
        self,
        *,
        model_adapter,
        ranker=None,
        quality_policy=None,
        min_groups=1,
        recorder=None,
        speed_options=None,
    ):
        if model_adapter is None:
            raise ValueError("model_adapter is required.")
        if (
            isinstance(min_groups, bool)
            or not isinstance(min_groups, Integral)
            or min_groups < 1
        ):
            raise ValueError("min_groups must be a positive integer.")
        self.model_adapter = model_adapter
        self.ranker = ranker if ranker is not None else MeanAbsoluteShapRanker()
        self.quality_policy = (
            quality_policy if quality_policy is not None else MacroQualityPolicy()
        )
        self.min_groups = int(min_groups)
        if speed_options is not None and not isinstance(
            speed_options, (SelectionSpeedOptions, BatchRemovalOptions)
        ):
            raise TypeError(
                "speed_options must be SelectionSpeedOptions, BatchRemovalOptions or None."
            )
        self.speed_options = speed_options
        self.recorder = recorder
        self._is_fitted = False

    def _emit(self, kind, **payload):
        if self.recorder is not None:
            self.recorder.record(deepcopy({"event": kind, **payload}))

    def _rank(self, adapter, model, X, y, feature_groups, group_ids):
        scores = dict(
            self.ranker.rank(adapter, model, X, y, feature_groups.relative(group_ids))
        )
        if set(scores) != set(group_ids):
            raise ValueError("Ranker must return exactly the retained group IDs.")
        scores = {group: float(scores[group]) for group in group_ids}
        if any(not np.isfinite(value) or value < 0 for value in scores.values()):
            raise ValueError("Ranker must return finite nonnegative group importance.")
        return scores

    @staticmethod
    def _check_model(adapter, model, n_features, expected_classes=None):
        if adapter.feature_count(model) != n_features:
            raise ValueError(
                "Fitted model feature count differs from the candidate schema."
            )
        classes = np.asarray(adapter.classes(model)).reshape(-1)
        if len(classes) < 2 or len(np.unique(classes)) != len(classes):
            raise ValueError(
                "A classification model with at least two distinct classes is required."
            )
        if expected_classes is not None and not np.array_equal(
            classes, expected_classes
        ):
            raise ValueError(
                "Candidate model classes/order differ from the full model."
            )
        return classes

    def _fit(
        self,
        X_train,
        y_train,
        *,
        eval_set,
        feature_groups,
        feature_names=None,
        full_model=None,
        cv_groups=None,
    ):
        """Fit the entire search. Existing estimators/inputs are never retrained in place.

        feature_names is optional for arrays and inferred for named DataFrames.
        cv_groups optionally supplies Train recording IDs for the CV gate only.
        full_model must already be fitted with the same input columns, labels,
        model parameters and training protocol. The caller owns its provenance.
        """
        self._is_fitted = False
        if isinstance(self.quality_policy, CVF1ReferenceQualityPolicy):
            raise ValueError('CVF1ReferenceQualityPolicy requires fit_cv() with external folds.')
        started = perf_counter()
        if feature_names is None and hasattr(X_train, "columns"):
            feature_names = list(X_train.columns)
        names = tuple(feature_names) if feature_names is not None else None
        X = numeric_matrix(X_train, name="X_train", feature_names=names)
        if names is not None:
            if (
                len(names) != X.shape[1]
                or any(not isinstance(n, str) for n in names)
                or len(set(names)) != len(names)
            ):
                raise ValueError(
                    "feature_names must be unique strings, one per original column."
                )
        y = labels(y_train, name="y_train", n_samples=len(X))
        if not isinstance(eval_set, (tuple, list)) or len(eval_set) != 2:
            raise ValueError("eval_set must be the pair (X_val, y_val).")
        X_val = numeric_matrix(
            eval_set[0], name="X_val", n_features=X.shape[1], feature_names=names
        )
        y_val = labels(eval_set[1], name="y_val", n_samples=len(X_val))
        if feature_groups is None:
            feature_groups = {column + 1: [column] for column in range(X.shape[1])}
        groups = FeatureGroups(feature_groups, X.shape[1])
        if self.min_groups > len(groups.group_ids):
            if self._selection_mode == "feature":
                raise ValueError("min_features exceeds the number of input features.")
            raise ValueError(
                "min_groups exceeds the number of supplied feature groups."
            )
        train_classes = np.unique(y)
        if len(train_classes) < 2 or not np.isin(y_val, train_classes).all():
            raise ValueError(
                "Training needs at least two classes; validation labels must be known."
            )
        adapter = self.model_adapter
        cv_mode = isinstance(self.quality_policy, CVReferenceQualityPolicy)
        speed = self.speed_options
        cv_first = cv_mode and speed is not None
        if isinstance(speed, BatchRemovalOptions) and cv_mode:
            raise ValueError(
                "BatchRemovalOptions requires a Validation quality policy."
            )
        if isinstance(speed, SelectionSpeedOptions) and not cv_mode:
            raise ValueError("SelectionSpeedOptions requires CVReferenceQualityPolicy.")
        if cv_groups is not None and not cv_mode:
            raise ValueError("cv_groups requires CVReferenceQualityPolicy.")
        cv_folds, validated_cv_groups = (
            make_folds(X, y, self.quality_policy, cv_groups) if cv_mode else ((), None)
        )
        cv_trials, validation_trials = [], []
        timings = {
            "full_training_seconds": 0.0,
            "baseline_prediction_seconds": 0.0,
            "initial_shap_seconds": 0.0,
            "trial_training_seconds": 0.0,
            "validation_prediction_seconds": 0.0,
            "shap_update_seconds": 0.0,
            "trial_model_fit_count": 0,
            "cv_fit_count": 0,
        }
        self._emit("run_started", group_ids=groups.group_ids)
        if full_model is None:
            full_model = adapter.new_model()
            tick = perf_counter()
            adapter.fit(full_model, X, y, (X_val, y_val))
            timings["full_training_seconds"] = perf_counter() - tick
        else:
            adapter.validate_full_model(full_model)
        classes = self._check_model(adapter, full_model, X.shape[1])
        if set(classes.tolist()) != set(train_classes.tolist()):
            raise ValueError(
                "full_model class labels differ from the supplied training labels."
            )
        tick = perf_counter()
        baseline = MacroScores.from_predictions(
            y_val, adapter.predict(full_model, X_val), classes
        )
        timings["baseline_prediction_seconds"] = perf_counter() - tick
        validation_baseline = baseline
        validation_final = baseline
        cv_baseline = cv_current = None
        if cv_mode:
            cv_baseline = evaluate_folds(
                adapter,
                X,
                y,
                cv_folds,
                classes,
                check_model=self._check_model,
                forbidden_models=(full_model,),
                emit=self._emit,
                speed_options=speed,
            )
            cv_current = cv_baseline
            baseline = cv_baseline.mean
            timings.update(
                cv_baseline_training_seconds=cv_baseline.training_seconds,
                cv_baseline_prediction_seconds=cv_baseline.prediction_seconds,
                cv_trial_training_seconds=0.0,
                cv_trial_prediction_seconds=0.0,
                cv_fit_count=len(cv_folds),
            )
        floors = self.quality_policy.thresholds(cv_baseline if cv_mode else baseline)
        if not self.quality_policy.evaluate(baseline, floors).passed:
            raise ValueError(
                "Full model does not meet the fixed validation floors; selection did not start."
            )
        current_model = full_model
        current_groups = groups.group_ids
        current_scores = baseline
        planner = RemovalPlanner(
            self.min_groups, speed.removal_fraction if speed else 0.0
        )
        failed = planner.failed
        attempted = set()
        tick = perf_counter()
        ranking = self._rank(adapter, current_model, X, y, groups, current_groups)
        timings["initial_shap_seconds"] = perf_counter() - tick
        initial_ranking = dict(ranking)
        history = []
        accepted = [AcceptedState(0, 0, current_groups, baseline)]
        snapshots = [ShapSnapshot(0, 0, current_groups, tuple(ranking.items()))]
        self._emit(
            "initialized",
            baseline=baseline.as_dict(),
            floors=floors.as_dict(),
            ranking=ranking,
            score_source="train_cv" if cv_mode else "validation",
        )
        loop_started = perf_counter()
        configuration_id = 0
        if cv_first:
            jobs, threads = speed.resources(len(cv_folds), adapter)
            timings.update(
                cv_workers=jobs, cv_threads_per_worker=threads, cv_pruned_trials=0
            )

        def fit_candidate(X_candidate, X_val_candidate):
            model = adapter.new_model()
            if model is current_model or model is full_model:
                raise ValueError("Backend must return a fresh model for every trial.")
            tick = perf_counter()
            adapter.fit(model, X_candidate, y, (X_val_candidate, y_val))
            training = perf_counter() - tick
            timings["trial_training_seconds"] += training
            timings["trial_model_fit_count"] += 1
            self._check_model(adapter, model, X_candidate.shape[1], classes)
            tick = perf_counter()
            predictions = adapter.predict(model, X_val_candidate)
            predicting = perf_counter() - tick
            timings["validation_prediction_seconds"] += predicting
            return (
                model,
                MacroScores.from_predictions(y_val, predictions, classes),
                training,
                predicting,
            )

        while True:
            if len(current_groups) <= self.min_groups:
                stop_reason = "min_groups_reached"
                break
            removed = planner.choose(current_groups, ranking)
            if not removed:
                stop_reason = "all_candidates_rejected"
                break
            group = removed[0]
            proposed = tuple(s for s in current_groups if s not in removed)
            columns = groups.columns(proposed)
            X_candidate = X[:, columns]
            X_val_candidate = X_val[:, columns]
            trial_id = len(history) + 1
            self._emit(
                "trial_started",
                trial_id=trial_id,
                configuration_id=configuration_id,
                group_id=group,
                removed_group_ids=removed,
                groups_before=current_groups,
                proposed_groups=proposed,
            )
            trial_model, validation_scores = None, None
            train_seconds = predict_seconds = 0.0
            if not cv_first:
                trial_model, validation_scores, train_seconds, predict_seconds = (
                    fit_candidate(X_candidate, X_val_candidate)
                )
            scores = validation_scores
            metric_kind = "mean"
            if cv_mode:
                cv_trial = evaluate_folds(
                    adapter,
                    X_candidate,
                    y,
                    cv_folds,
                    classes,
                    check_model=self._check_model,
                    forbidden_models=(full_model, current_model, trial_model),
                    emit=self._emit,
                    trial_id=trial_id,
                    speed_options=speed,
                    floors=floors,
                    tolerance=self.quality_policy.tolerance,
                )
                timings["cv_trial_training_seconds"] += cv_trial.training_seconds
                timings["cv_trial_prediction_seconds"] += cv_trial.prediction_seconds
                timings["cv_fit_count"] += len(cv_trial.fold_scores)
                if isinstance(cv_trial, CVPrunedEvaluation):
                    scores, metric_kind = cv_trial.upper_bounds, "cv_upper_bound"
                    timings["cv_pruned_trials"] += 1
                else:
                    scores = cv_trial.mean
                cv_trials.append(cv_trial)
            decision = self.quality_policy.evaluate(scores, floors)
            if cv_first and decision.passed:
                trial_model, validation_scores, train_seconds, predict_seconds = (
                    fit_candidate(X_candidate, X_val_candidate)
                )
            if cv_mode:
                validation_trials.append(validation_scores)
            changes = MacroScores(
                *(
                    scores.as_dict()[name] - baseline.as_dict()[name]
                    for name in baseline.as_dict()
                )
            )
            trial = TrialRecord(
                trial_id,
                configuration_id,
                group,
                current_groups,
                proposed,
                float(np.mean([ranking[item] for item in removed])),
                scores,
                changes,
                decision.passed,
                decision.worst_margin,
                decision.failed_metrics,
                train_seconds,
                predict_seconds,
                removed if speed else (),
                metric_kind,
            )
            history.append(trial)
            attempted.update(removed)
            self._emit("trial_completed", **trial.as_dict())
            planner.observe(removed, decision.passed)
            if not decision.passed:
                continue
            current_groups = proposed
            current_model = trial_model
            current_scores = scores
            validation_final = validation_scores
            if cv_mode:
                cv_current = cv_trial
            failed.clear()
            configuration_id += 1
            accepted.append(
                AcceptedState(configuration_id, trial_id, current_groups, scores)
            )
            tick = perf_counter()
            ranking = self._rank(
                adapter, current_model, X_candidate, y, groups, current_groups
            )
            timings["shap_update_seconds"] += perf_counter() - tick
            snapshots.append(
                ShapSnapshot(
                    configuration_id, trial_id, current_groups, tuple(ranking.items())
                )
            )
            self._emit(
                "configuration_accepted",
                configuration_id=configuration_id,
                trial_id=trial_id,
                group_ids=current_groups,
                ranking=ranking,
            )
        timings["elimination_loop_seconds"] = perf_counter() - loop_started
        timings["total_fit_seconds"] = perf_counter() - started
        columns = groups.columns(current_groups)
        selected_names = tuple(names[c] for c in columns) if names is not None else None
        cv_trace = (
            CVEvaluationTrace(
                cv_folds,
                validated_cv_groups is not None,
                cv_baseline,
                tuple(cv_trials),
                cv_current,
                validation_baseline,
                tuple(validation_trials),
                validation_final,
                tuple(y.tolist()),
                (
                    tuple(validated_cv_groups.tolist())
                    if validated_cv_groups is not None
                    else None
                ),
            )
            if cv_mode
            else None
        )
        result = SelectionResult(
            current_model,
            current_groups,
            columns,
            selected_names,
            baseline,
            floors,
            current_scores,
            tuple(history),
            tuple(accepted),
            tuple(snapshots),
            dict(timings),
            stop_reason,
            cv_trace,
            speed,
            getattr(self.quality_policy, "gate_metrics", tuple(baseline.as_dict())),
        )
        self._emit(
            "run_finished",
            group_ids=current_groups,
            stop_reason=stop_reason,
            trial_count=len(history),
            shap_calls=len(snapshots),
        )
        self.adapter_ = adapter
        self.full_model_ = full_model
        self.classes_ = classes.copy()
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = names
        self.feature_groups_ = groups
        self.initial_ranking_ = initial_ranking
        self.final_ranking_ = dict(ranking)
        self.attempted_group_ids_ = tuple(sorted(attempted))
        self.rejected_group_ids_ = tuple(sorted(failed))
        self._set_result(result)
        return self

    def _set_result(self, result):
        self.result_ = result
        self.split_ = result.recording_split
        self.test_indices_ = self.split_.test_rows if self.split_ is not None else None
        self.development_indices_ = self.split_.development_rows if self.split_ is not None else None
        self.cv_evaluation_ = result.cv_evaluation
        budgets = getattr(result.cv_evaluation, 'refit_rounds', ())
        self.refit_rounds_ = budgets[-1] if budgets else None
        self.estimator_ = result.estimator
        self.selected_group_ids_ = result.selected_group_ids
        self.selected_columns_ = result.selected_columns
        self.selected_feature_names_ = result.selected_feature_names
        self.baseline_metrics_ = result.baseline_metrics
        self.effective_floors_ = result.effective_floors
        self.final_metrics_ = result.final_metrics
        self.best_f1_ = max(state.metrics.f1_macro for state in result.accepted_history)
        self.history_ = result.history
        self.accepted_history_ = result.accepted_history
        self.shap_history_ = result.shap_history
        self.timings_ = dict(result.timings)
        self.stop_reason_ = result.stop_reason
        self._is_fitted = True

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError(
                "Call fit() successfully or load() a saved selector first."
            )

    def fit_cv(self, X_dev, y_dev, *, cv_splits, cv_groups=None,
               feature_groups=None, feature_names=None, fold_kind='partition'):
        """Select using supplied development Train/Validation folds and mean F1.

        Requires CVF1ReferenceQualityPolicy. Every fold trains a fresh model;
        Validation both monitors early stopping and supplies the fold score.
        Full and accepted candidates are refitted without Validation using the
        ceiling of median best boosting rounds. Test must never be supplied.
        BatchRemovalOptions retains the existing batch/backoff/retry semantics.
        """
        from .fixed_cv import fit_fixed_cv

        return fit_fixed_cv(self, X_dev, y_dev, cv_splits=cv_splits,
                            cv_groups=cv_groups, feature_groups=feature_groups,
                            feature_names=feature_names, fold_kind=fold_kind)

    def fit_recording_cv(self, X, y, *, metadata=None, split_policy=None,
                         parent_metadata=None, label_mapping=None, split=None,
                         feature_groups=None, feature_names=None):
        """Hold out Test, then select on four recording-disjoint development splits.

        X is an already extracted feature matrix; metadata rows align with X/y.
        Supply a SetupRecordingSplit policy or a validated RecordingSplit. Parent
        metadata allocates source conditions before a class subset is retained.
        No Test rows reach model fits, SHAP ranking or feature-selection scores.
        """
        from dataclasses import replace
        from .recording_split import SetupRecordingSplit, RecordingSplit
        self._is_fitted = False
        if feature_names is None and hasattr(X, 'columns'):
            feature_names = list(X.columns)
        values = numeric_matrix(X, name='X', feature_names=feature_names)
        target = labels(y, name='y', n_samples=len(values))
        if split is None:
            if metadata is None or not isinstance(split_policy, SetupRecordingSplit):
                raise ValueError('Supply aligned metadata and a SetupRecordingSplit policy.')
            split = split_policy.build(metadata, target, parent_metadata=parent_metadata,
                                       label_mapping=label_mapping)
        else:
            if not isinstance(split, RecordingSplit):
                raise TypeError('split must be a RecordingSplit.')
            if any(item is not None for item in (metadata, split_policy, parent_metadata, label_mapping)):
                raise ValueError('Use a saved split or metadata/policy, not both.')
            split.validate()
        if len(values) != len(split.sample_ids) or tuple(target.tolist()) != split.model_labels:
            raise ValueError('X/y differ from the recording split row count/labels.')
        dev = list(split.development_rows)
        self.fit_cv(values[dev], target[dev], cv_splits=split.folds,
                    cv_groups=split.development_groups, fold_kind='repeated',
                    feature_groups=feature_groups, feature_names=feature_names)
        self._set_result(replace(self.result_, recording_split=split))
        return self

    def transform(self, X):
        """Select retained columns from the complete original feature schema."""
        self._check_fitted()
        matrix = numeric_matrix(
            X,
            name="X",
            n_features=self.n_features_in_,
            feature_names=self.feature_names_in_,
        )
        return matrix[:, self.selected_columns_]

    def predict(self, X):
        """Predict from the full original input schema, selecting columns internally."""
        selected = self.transform(X)
        predictions = np.asarray(
            self.adapter_.predict(self.estimator_, selected)
        ).reshape(-1)
        if (
            len(predictions) != len(selected)
            or not np.isin(predictions, self.classes_).all()
        ):
            raise ValueError("Backend returned invalid prediction rows or classes.")
        return predictions

    def history_records(self):
        self._check_fitted()
        return [record.as_dict() for record in self.history_]

    def save(self, directory):
        """Save model + explicit schema/history (GBC uses a joblib model file)."""
        from .persistence import save_selector

        self._check_fitted()
        return save_selector(self, directory)

    @classmethod
    def load(cls, directory, *, trusted=False):
        """Restore a selection; GBC joblib bundles require trusted=True."""
        from .persistence import load_selector

        selector = load_selector(directory, trusted=trusted)
        if selector._selection_mode != cls._selection_mode:
            raise ValueError(
                f"Saved selection uses {selector._selection_mode} mode; load with {type(selector).__name__}."
            )
        return selector


class RecursiveGroupSelector(_RecursiveSelectorBase):
    """Select complete feature groups with fresh refits and reject/retry.

    Supply a model_adapter explicitly and a partition of columns to fit().
    min_groups counts groups, even when they contain different numbers of columns.
    Test data must not be passed as eval_set.
    """

    def fit(
        self,
        X_train,
        y_train,
        *,
        eval_set,
        feature_groups,
        feature_names=None,
        full_model=None,
        cv_groups=None,
    ):
        """Select groups; preserve the original column order within each refit.

        feature_names is inferred for DataFrames. An optional full_model must be
        fitted using the same input schema, parameters and training protocol.
        """
        if feature_groups is None:
            self._is_fitted = False
            raise ValueError(
                "Supply feature_groups, or use RecursiveFeatureSelector for individual columns."
            )
        return self._fit(
            X_train,
            y_train,
            eval_set=eval_set,
            feature_groups=feature_groups,
            feature_names=feature_names,
            full_model=full_model,
            cv_groups=cv_groups,
        )


class RecursiveFeatureSelector(_RecursiveSelectorBase):
    """Select individual columns with the same refit and reject/retry algorithm.

    Each original column is a singleton group internally: group ID = column + 1.
    selected_columns_ contains zero-based original indices. Group-based history,
    reports, plotting and audits retain their existing schema. This convenience
    API changes selection granularity; it does not implement Probatus ShapRFECV.
    """

    _selection_mode = "feature"

    def __init__(
        self,
        *,
        model_adapter,
        ranker=None,
        quality_policy=None,
        min_features=1,
        recorder=None,
        speed_options=None,
    ):
        if (
            isinstance(min_features, bool)
            or not isinstance(min_features, Integral)
            or min_features < 1
        ):
            raise ValueError("min_features must be a positive integer.")
        super().__init__(
            model_adapter=model_adapter,
            ranker=ranker,
            quality_policy=quality_policy,
            min_groups=min_features,
            recorder=recorder,
            speed_options=speed_options,
        )

    @property
    def min_features(self):
        """Minimum column count; identical to the shared singleton-group minimum."""
        return self.min_groups

    def fit(
        self,
        X_train,
        y_train,
        *,
        eval_set,
        feature_names=None,
        full_model=None,
        cv_groups=None,
    ):
        """Select individual features without requiring a feature_groups mapping.

        feature_names is inferred for DataFrames. full_model has the same reuse
        requirements as in RecursiveGroupSelector.fit().
        """
        return self._fit(
            X_train,
            y_train,
            eval_set=eval_set,
            feature_groups=None,
            feature_names=feature_names,
            full_model=full_model,
            cv_groups=cv_groups,
        )
