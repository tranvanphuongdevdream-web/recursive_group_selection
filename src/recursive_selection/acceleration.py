"""Opt-in CV-first evaluation and deterministic batch/backoff scheduling."""

from dataclasses import asdict, dataclass
from math import ceil, isfinite
from numbers import Integral
from joblib import cpu_count


@dataclass(frozen=True)
class SelectionSpeedOptions:
    """CV first, exact rejection bounds, bounded fold workers, and batch backoff.

    A first fold is evaluated alone before parallel waves. Only accepted CV
    candidates receive a deployment fit. removal_fraction=0 uses single removals
    for an apples-to-apples comparison; batch removal changes the search path.
    """

    cv_jobs: int = 2
    cpu_budget: int | None = None
    removal_fraction: float = 0.1
    early_reject: bool = True

    def __post_init__(self):
        for key in ("cv_jobs", "cpu_budget"):
            value = getattr(self, key)
            if value is None and key == "cpu_budget":
                continue
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{key} must be a positive integer.")
        if (
            isinstance(self.removal_fraction, bool)
            or not isfinite(self.removal_fraction)
            or not 0 <= self.removal_fraction <= 1
        ):
            raise ValueError("removal_fraction must be between 0 and 1.")
        if not isinstance(self.early_reject, bool):
            raise ValueError("early_reject must be a bool.")

    def resources(self, n_folds, adapter):
        # joblib respects affinity and container CPU quotas, not just host cores.
        available = max(1, cpu_count())
        budget = min(available, self.cpu_budget or available)
        jobs = min(self.cv_jobs, n_folds, budget)
        params = getattr(adapter, "params", {})
        device = str(params.get('device', params.get('device_type', 'cpu'))).lower()
        if str(params.get("task_type", "CPU")).upper() == "GPU" or device.startswith(('gpu', 'cuda')):
            jobs = 1
        threads = max(1, budget // jobs)
        requested = params.get(
            "thread_count"
            if getattr(adapter, "backend_name", "") == "catboost"
            else "n_jobs"
        )
        if isinstance(requested, Integral) and requested > 0:
            threads = min(threads, requested)
        return jobs, threads

    def to_config(self):
        return asdict(self)


@dataclass(frozen=True)
class BatchRemovalOptions:
    """Batch/backoff search; optionally parallelize folds in external fit_cv().

    Zero uses singleton removals. Batches can change the resulting search path.
    cv_jobs/cpu_budget affect external CV only; trials remain sequential.
    """

    removal_fraction: float = 0.1
    cv_jobs: int = 1
    cpu_budget: int | None = None

    def __post_init__(self):
        for name in ('cv_jobs', 'cpu_budget'):
            value = getattr(self, name)
            if value is None and name == 'cpu_budget':
                continue
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f'{name} must be a positive integer.')
        if (
            isinstance(self.removal_fraction, bool)
            or not isfinite(self.removal_fraction)
            or not 0 <= self.removal_fraction <= 1
        ):
            raise ValueError("removal_fraction must be between 0 and 1.")

    def to_config(self):
        return asdict(self)

    def resources(self, n_folds, adapter):
        return SelectionSpeedOptions(cv_jobs=self.cv_jobs, cpu_budget=self.cpu_budget).resources(n_folds, adapter)


class RemovalPlanner:
    """Ascending SHAP; failed batches shrink, failed singletons become ineligible."""

    def __init__(self, minimum, fraction=0.0):
        self.minimum, self.fraction = minimum, fraction
        self.failed = set()
        self.batch_limit = None

    def choose(self, current, ranking):
        eligible = sorted(
            (g for g in current if g not in self.failed), key=lambda g: (ranking[g], -g)
        )
        if not eligible or len(current) <= self.minimum:
            return ()
        count = max(1, ceil(len(current) * self.fraction))
        count = min(count, len(current) - self.minimum, len(eligible))
        if self.batch_limit is not None:
            count = min(count, self.batch_limit)
        return tuple(eligible[:count])

    def observe(self, removed, accepted):
        if accepted:
            self.failed.clear()
            self.batch_limit = None
        elif len(removed) > 1:
            self.batch_limit = max(1, len(removed) // 2)
        else:
            self.failed.add(removed[0])


def removed_ids(trial):
    return trial.removed_group_ids or (trial.group_id,)
