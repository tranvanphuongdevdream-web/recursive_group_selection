"""Validation acceptance rules. They never change group or model state."""

from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_DOWN
from math import isfinite
from numbers import Integral, Real

from .results import MacroScores, QualityDecision


@dataclass(frozen=True)
class MacroQualityPolicy:
    min_precision_macro: float = 0.87
    min_recall_macro: float = 0.87
    min_f1_macro: float = 0.87
    max_drop_from_full: float = 0.04
    tolerance: float = 1e-12

    def __post_init__(self):
        for name in (
            "min_precision_macro",
            "min_recall_macro",
            "min_f1_macro",
            "max_drop_from_full",
        ):
            value = getattr(self, name)
            if not isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and between 0 and 1.")
        if not isfinite(self.tolerance) or self.tolerance < 0:
            raise ValueError("tolerance must be finite and nonnegative.")

    def thresholds(self, baseline):
        return MacroScores(
            max(
                self.min_precision_macro,
                baseline.precision_macro - self.max_drop_from_full,
            ),
            max(self.min_recall_macro, baseline.recall_macro - self.max_drop_from_full),
            max(self.min_f1_macro, baseline.f1_macro - self.max_drop_from_full),
        )

    def evaluate(self, scores, thresholds):
        margins = {
            name: score - thresholds.as_dict()[name]
            for name, score in scores.as_dict().items()
        }
        if not all(isfinite(value) for value in margins.values()):
            raise ValueError("Quality scores and thresholds must be finite.")
        failed = tuple(
            name for name, margin in margins.items() if margin < -self.tolerance
        )
        return QualityDecision(not failed, min(margins.values()), failed)

    def to_config(self):
        return asdict(self)


@dataclass(frozen=True)
class CVReferenceQualityPolicy:
    """Fixed full-reference CV mean minus one fold-based standard error.

    This is a baseline-anchored heuristic, not Probatus's best-path subset rule,
    a confidence interval, or a formal non-inferiority test. Each trial is scored
    on the same Train-only folds. There is no absolute score floor or fixed drop.
    """

    n_splits: int = 3
    random_state: int = 42
    se_multiplier: float = 1.0
    tolerance: float = 1e-12

    def __post_init__(self):
        from numbers import Integral

        if (
            isinstance(self.n_splits, bool)
            or not isinstance(self.n_splits, Integral)
            or self.n_splits < 2
        ):
            raise ValueError("n_splits must be an integer >= 2.")
        if (
            isinstance(self.random_state, bool)
            or not isinstance(self.random_state, Integral)
            or self.random_state < 0
        ):
            raise ValueError("random_state must be a nonnegative integer.")
        for name in ("se_multiplier", "tolerance"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")

    def thresholds(self, baseline):
        from .results import CVScoreSummary

        if not isinstance(baseline, CVScoreSummary):
            raise TypeError(
                "CVReferenceQualityPolicy needs a CVScoreSummary from Train folds, not Validation scores."
            )
        for key in baseline.mean.as_dict():
            mean, error = getattr(baseline.mean, key), getattr(
                baseline.standard_error, key
            )
            if (
                not isfinite(mean)
                or not 0 <= mean <= 1
                or not isfinite(error)
                or error < 0
            ):
                raise ValueError(
                    "CV reference means/standard errors must be finite and valid."
                )
        return MacroScores(
            *(
                max(
                    0.0,
                    getattr(baseline.mean, key)
                    - self.se_multiplier * getattr(baseline.standard_error, key),
                )
                for key in baseline.mean.as_dict()
            )
        )

    def evaluate(self, scores, thresholds):
        margins = {
            key: score - getattr(thresholds, key)
            for key, score in scores.as_dict().items()
        }
        if not all(isfinite(value) for value in margins.values()):
            raise ValueError("Quality scores and thresholds must be finite.")
        failed = tuple(
            key for key, margin in margins.items() if margin < -self.tolerance
        )
        return QualityDecision(not failed, min(margins.values()), failed)

    def to_config(self):
        return asdict(self)


@dataclass(frozen=True)
class F1ReferenceQualityPolicy:
    """Accept when Validation macro F1 matches the original full-model reference.

    The reference stays fixed throughout selection. Precision and recall remain
    diagnostics. Reusing Validation for search does not guarantee Test quality.
    """

    tolerance: float = 1e-12
    gate_metrics = ("f1_macro",)

    def __post_init__(self):
        if (
            isinstance(self.tolerance, bool)
            or not isfinite(self.tolerance)
            or self.tolerance < 0
        ):
            raise ValueError("tolerance must be finite and nonnegative.")

    def thresholds(self, baseline):
        self._validate(baseline)
        return MacroScores(0.0, 0.0, baseline.f1_macro)

    @staticmethod
    def _validate(scores):
        if not isinstance(scores, MacroScores) or not all(
            isfinite(value) and 0 <= value <= 1 for value in scores.as_dict().values()
        ):
            raise ValueError(
                "Quality scores and thresholds must be finite and between 0 and 1."
            )

    def evaluate(self, scores, thresholds):
        self._validate(scores)
        self._validate(thresholds)
        margin = scores.f1_macro - thresholds.f1_macro
        failed = ("f1_macro",) if margin < -self.tolerance else ()
        return QualityDecision(not failed, margin, failed)

    def to_config(self):
        return asdict(self)


@dataclass(frozen=True)
class CVF1ReferenceQualityPolicy(F1ReferenceQualityPolicy):
    """Compare complete external-fold mean F1 with a full or running reference.

    No SD/SE allowance is subtracted. Other macro metrics remain diagnostics.
    Optional decimal truncation applies after averaging folds, to both sides of
    the gate. Truncated comparisons are exact; tolerance only applies when
    truncate_decimals is None. Raw scores remain unchanged in the result.
    Use with fit_cv(); the single-Validation and Train-only CV modes are unchanged.
    """

    reference: str = 'best'
    truncate_decimals: int | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.reference not in ('best', 'full'):
            raise ValueError("reference must be 'best' or 'full'.")
        if self.truncate_decimals is not None and (
            isinstance(self.truncate_decimals, bool)
            or not isinstance(self.truncate_decimals, Integral)
            or not 0 <= self.truncate_decimals <= 15
        ):
            raise ValueError('truncate_decimals must be None or an integer from 0 to 15.')

    def gate_value(self, f1):
        """Return the comparison value without modifying the stored raw F1."""
        if isinstance(f1, bool) or not isinstance(f1, Real) or not isfinite(f1) or not 0 <= f1 <= 1:
            raise ValueError('F1 must be finite and between 0 and 1.')
        if self.truncate_decimals is None:
            return float(f1)
        # Decimal text avoids turning values such as .58 into .57 through
        # binary floating-point multiplication before truncation.
        quantum = Decimal(1).scaleb(-int(self.truncate_decimals))
        return float(Decimal(str(f1)).quantize(quantum, rounding=ROUND_DOWN))

    def evaluate(self, scores, thresholds):
        if self.truncate_decimals is None:
            return super().evaluate(scores, thresholds)
        self._validate(scores)
        self._validate(thresholds)
        score, threshold = self.gate_value(scores.f1_macro), self.gate_value(thresholds.f1_macro)
        failed = ('f1_macro',) if score < threshold else ()
        return QualityDecision(not failed, score-threshold, failed)

    def advance(self, floors, scores):
        """Return the next threshold after acceptance; never lower the reference."""
        self._validate(floors)
        self._validate(scores)
        return MacroScores(0., 0., max(floors.f1_macro, self.gate_value(scores.f1_macro))) if self.reference == 'best' else floors

    def thresholds(self, baseline):
        from .results import CVScoreSummary

        if not isinstance(baseline, CVScoreSummary):
            raise TypeError('CVF1ReferenceQualityPolicy requires a full CVScoreSummary.')
        floors = super().thresholds(baseline.mean)
        return MacroScores(0., 0., self.gate_value(floors.f1_macro))
