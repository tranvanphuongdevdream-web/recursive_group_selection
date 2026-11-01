"""Serializable records, independent of any boosting library."""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.metrics import precision_recall_fscore_support


@dataclass(frozen=True)
class MacroScores:
    precision_macro: float
    recall_macro: float
    f1_macro: float

    @classmethod
    def from_predictions(cls, y, predictions, classes):
        predicted = np.asarray(predictions).reshape(-1)
        if len(predicted) != len(y) or not np.isin(predicted, classes).all():
            raise ValueError(
                "Predictions must match validation rows and known classes."
            )
        precision, recall, f1, _ = precision_recall_fscore_support(
            y,
            predicted,
            labels=classes,
            average=None,
            zero_division=0,
        )
        return cls(float(precision.mean()), float(recall.mean()), float(f1.mean()))

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class QualityDecision:
    passed: bool
    worst_margin: float
    failed_metrics: tuple[str, ...]


@dataclass(frozen=True)
class TrialRecord:
    trial_id: int
    configuration_id: int
    group_id: int
    groups_before: tuple[int, ...]
    proposed_groups: tuple[int, ...]
    importance: float
    metrics: MacroScores
    change_from_full: MacroScores
    accepted: bool
    worst_margin: float
    failed_metrics: tuple[str, ...]
    train_seconds: float
    validation_predict_seconds: float
    removed_group_ids: tuple[int, ...] = ()
    metric_kind: str = "mean"
    f1_reference_before: float | None = None
    f1_reference_after: float | None = None

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class AcceptedState:
    configuration_id: int
    trial_id: int
    group_ids: tuple[int, ...]
    metrics: MacroScores


@dataclass(frozen=True)
class ShapSnapshot:
    configuration_id: int
    trial_id: int
    group_ids: tuple[int, ...]
    # Immutable pairs preserve original input-group order.
    importance: tuple[tuple[int, float], ...]


@dataclass(frozen=True)
class SelectionResult:
    estimator: Any
    selected_group_ids: tuple[int, ...]
    selected_columns: tuple[int, ...]
    selected_feature_names: tuple[str, ...] | None
    baseline_metrics: MacroScores
    effective_floors: MacroScores
    final_metrics: MacroScores
    history: tuple[TrialRecord, ...]
    accepted_history: tuple[AcceptedState, ...]
    shap_history: tuple[ShapSnapshot, ...]
    timings: dict[str, float]
    stop_reason: str
    cv_evaluation: "CVEvaluationTrace | FixedCVEvaluationTrace | None" = None
    speed_options: Any = None
    gate_metrics: tuple[str, ...] = ("precision_macro", "recall_macro", "f1_macro")
    recording_split: Any = None


@dataclass(frozen=True)
class CVFold:
    """Train-local row indices; stopping rows never overlap scored held-out rows."""

    train_rows: tuple[int, ...]
    stopping_rows: tuple[int, ...]
    heldout_rows: tuple[int, ...]


@dataclass(frozen=True)
class CVScoreSummary:
    fold_scores: tuple[MacroScores, ...]
    mean: MacroScores
    standard_deviation: MacroScores
    standard_error: MacroScores
    training_seconds: float = 0.0
    prediction_seconds: float = 0.0


@dataclass(frozen=True)
class CVPrunedEvaluation:
    """Observed folds and rigorous all-fold upper bounds; no full CV mean."""

    fold_scores: tuple[MacroScores, ...]
    total_folds: int
    upper_bounds: MacroScores
    training_seconds: float = 0.0
    prediction_seconds: float = 0.0


@dataclass(frozen=True)
class CVEvaluationTrace:
    folds: tuple[CVFold, ...]
    grouped: bool
    baseline: CVScoreSummary
    trials: tuple[CVScoreSummary | CVPrunedEvaluation, ...]
    final: CVScoreSummary
    validation_baseline: MacroScores
    validation_trials: tuple[MacroScores | None, ...]
    validation_final: MacroScores
    labels: tuple = ()
    group_ids: tuple | None = None


@dataclass(frozen=True)
class ValidationFold:
    """Development-local indices; Validation selects checkpoints and scores F1."""

    train_rows: tuple[int, ...]
    validation_rows: tuple[int, ...]


@dataclass(frozen=True)
class FixedCVEvaluationTrace:
    """External two-part folds, complete CV scores, and accepted refit budgets.

    best_rounds[0] describes Full; subsequent entries correspond to every trial.
    refit_rounds corresponds to accepted_history, including the initial Full.
    """

    folds: tuple[ValidationFold, ...]
    grouped: bool
    baseline: CVScoreSummary
    trials: tuple[CVScoreSummary, ...]
    final: CVScoreSummary
    best_rounds: tuple[tuple[int, ...], ...]
    refit_rounds: tuple[int, ...]
    labels: tuple
    group_ids: tuple | None = None
    reference: str = 'full'
    fold_kind: str = 'partition'
    checkpoint_metric: str = 'configured'
