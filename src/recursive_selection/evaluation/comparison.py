"""Pure comparisons of already evaluated, row-aligned classifications."""

import numpy as np

from ..reporting.selection import METRIC_NAMES
from ..reporting.export import save_reports


class EvaluationComparison:
    def __init__(self, reference, candidate):
        import pandas as pd

        if tuple(reference.splits) != tuple(candidate.splits):
            raise ValueError("Compared evaluations must have identical splits/order.")
        self.reference, self.candidate = reference, candidate
        rows = []
        for name in reference.splits:
            full, selected = reference[name], candidate[name]
            if full.classes != selected.classes or not np.array_equal(
                full.y_true, selected.y_true
            ):
                raise ValueError(
                    "Compared evaluations must have the same classes and ordered labels."
                )
            for metric in ("Accuracy", *METRIC_NAMES.values(), "Log loss"):
                a, b = full.metrics[metric], selected.metrics[metric]
                rows.append(
                    {
                        "Split": name,
                        "Metric": metric,
                        "Full": a,
                        "Selected": b,
                        "Change (pp)": (
                            100 * (b - a) if metric != "Log loss" else np.nan
                        ),
                        "Change": b - a,
                    }
                )
        self.metrics = pd.DataFrame(rows)

    def class_metrics(self, split):
        import pandas as pd

        full, selected = (
            self.reference[split].per_class,
            self.candidate[split].per_class,
        )
        rows = {"Class": full["Class"], "Samples": full["Samples"]}
        for metric in ("Precision", "Recall", "F1"):
            rows[metric + " full"] = full[metric]
            rows[metric + " selected"] = selected[metric]
            rows[metric + " change (pp)"] = 100 * (selected[metric] - full[metric])
        return pd.DataFrame(rows)

    def validation_constraints(self, *, policy, baseline, split="Validation"):
        import pandas as pd
        from ..policies import CVReferenceQualityPolicy, CVF1ReferenceQualityPolicy

        if isinstance(policy, (CVReferenceQualityPolicy, CVF1ReferenceQualityPolicy)):
            source = 'Validation CV' if isinstance(policy, CVF1ReferenceQualityPolicy) else 'Train CV'
            raise ValueError(
                f"CV gate constraints use {source} scores; use SelectionReport.cv_constraints, not outer Validation metrics."
            )
        floors = policy.thresholds(baseline)
        actual = self.reference[split].metrics
        if not np.allclose(
            [actual[label] for label in METRIC_NAMES.values()],
            list(baseline.as_dict().values()),
            atol=1e-12,
            rtol=0,
        ):
            raise ValueError(
                "Evaluation reference does not match the fixed selection baseline."
            )
        rows = []
        for key in getattr(policy, "gate_metrics", tuple(METRIC_NAMES)):
            label = METRIC_NAMES[key]
            full, selected = (
                getattr(baseline, key),
                self.candidate[split].metrics[label],
            )
            extra = (
                {
                    "Absolute minimum": getattr(policy, "min_" + key),
                    "Full minus allowed drop": full - policy.max_drop_from_full,
                }
                if hasattr(policy, "max_drop_from_full")
                else {}
            )
            rows.append(
                {
                    "Metric": label,
                    "Full": full,
                    "Selected": selected,
                    **extra,
                    "Minimum": getattr(floors, key),
                    "Change from full (pp)": 100 * (selected - full),
                    "Passed": selected >= getattr(floors, key) - policy.tolerance,
                }
            )
        return pd.DataFrame(rows)

    def feasibility_comparison(
        self, *, floors, validation_split="Validation", splits=("Validation", "Test")
    ):
        rows = self.metrics.loc[
            self.metrics["Split"].isin(splits)
            & self.metrics["Metric"].isin(METRIC_NAMES.values())
        ].copy()
        rows = rows.drop(columns="Change")
        thresholds = {
            label: getattr(floors, key) for key, label in METRIC_NAMES.items()
        }
        additional = {
            "Validation minimum": [
                thresholds[metric] if split == validation_split else np.nan
                for split, metric in zip(rows["Split"], rows["Metric"])
            ],
            "Scope": np.where(
                rows["Split"] == validation_split,
                "Selection constraint",
                "Final report; no test guarantee",
            ),
        }
        return rows.assign(**additional).reset_index(drop=True)

    def save(self, directory, *, prefix="comparison", tables=None, overwrite=False):
        from ..reporting.export import safe_name

        safe_name(prefix)
        return save_reports(
            directory,
            tables={f"{prefix}_metric_comparison.csv": self.metrics, **(tables or {})},
            prefix=prefix,
            overwrite=overwrite,
        )


def compare_evaluations(*, reference, candidate):
    """Rows must already be aligned by the caller; equal labels cannot prove row identity."""
    return EvaluationComparison(reference, candidate)
