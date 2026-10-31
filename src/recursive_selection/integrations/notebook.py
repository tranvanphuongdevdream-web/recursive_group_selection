"""Notebook progress recorder; callbacks only observe detached selector events."""

from ..reporting.selection import METRIC_NAMES, ranking_table


class NotebookSelectionRecorder:
    def __init__(
        self,
        *,
        unit_name="group",
        initial_title=None,
        group_labels=None,
        verbose=True,
        top_n=None,
    ):
        from numbers import Integral
        from IPython.display import display

        if top_n is not None and (
            isinstance(top_n, bool) or not isinstance(top_n, Integral) or top_n < 1
        ):
            raise ValueError("top_n must be a positive integer or None.")
        self.unit_name = unit_name
        self.initial_title = initial_title
        self.group_labels = group_labels
        self.verbose = verbose
        self.top_n = top_n
        self.table_display = display(
            "Run selection below to populate the initial importance ranking.",
            display_id=True,
        )
        self.chart_display = display(
            "The initial importance chart will appear when selection initializes.",
            display_id=True,
        )

    def record(self, event):
        if event["event"] == "initialized":
            self._score_label = (
                'Validation CV mean' if event.get('score_source') == 'validation_cv' else
                "Train CV" if event.get("score_source") == "train_cv" else "Validation"
            )
            from IPython.display import display
            from matplotlib import pyplot as plt
            from ..plotting import plot_group_importance

            table = ranking_table(
                event["ranking"],
                unit_name=self.unit_name,
                group_labels=self.group_labels,
            )
            styled = table.style.format(
                {
                    "importance": "{:.6f}",
                    "SHAP share": "{:.2%}",
                    "Cumulative SHAP": "{:.2%}",
                }
            )
            if self.table_display is None:
                display(styled)
            else:
                self.table_display.update(styled)
            fig, _ = plot_group_importance(
                event["ranking"],
                title=self.initial_title,
                group_labels=self.group_labels,
                total_groups=len(event["ranking"]),
                unit_name=self.unit_name,
                value_label="Mean |SHAP|",
                top_n=self.top_n,
            )
            if self.chart_display is None:
                display(fig)
            else:
                self.chart_display.update(fig)
            plt.close(fig)
        elif self.verbose and event["event"] == "trial_started":
            print(
                f"\n{'=' * 30}\n{self.unit_name.upper()} ELIMINATION - ROUND "
                f"{event['configuration_id'] + 1}, TRIAL {event['trial_id']}\n{'=' * 30}"
            )
            print(
                f"Try removing {self.unit_name}:",
                [
                    (self.group_labels or {}).get(group, group)
                    for group in event.get("removed_group_ids", (event["group_id"],))
                ],
            )
            print(f"{self.unit_name.title()}s before:", len(event["groups_before"]))
            print(f"{self.unit_name.title()}s tried:", len(event["proposed_groups"]))
        elif self.verbose and event["event"] == "trial_completed":
            for key, label in METRIC_NAMES.items():
                print(
                    f"{getattr(self, '_score_label', 'Validation')} {label}"
                    f"{' upper bound' if event.get('metric_kind') == 'cv_upper_bound' else ''}: {event['metrics'][key]:.2%}"
                )
            print(f"Worst gate margin (pp): {100 * event['worst_margin']:+.4f}")
            print(
                "Decision:",
                (
                    "ACCEPTED"
                    if event["accepted"]
                    else f"REJECTED - TRY ANOTHER {self.unit_name.upper()}"
                ),
            )
            if not event["accepted"]:
                print(
                    "Failed metrics:",
                    ", ".join(
                        METRIC_NAMES.get(key, key) for key in event["failed_metrics"]
                    ),
                )
