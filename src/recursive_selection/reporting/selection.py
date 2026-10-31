"""Convert recorded results to tables without fitting, prediction or SHAP."""

from dataclasses import asdict

import numpy as np

from ..acceleration import removed_ids
from ..results import CVPrunedEvaluation, FixedCVEvaluationTrace
from ..groups import FeatureGroups
from ..plotting.importance import importance_evolution_data
from ..plotting.retries import retry_history_data
from .export import safe_name, save_reports

METRIC_NAMES = {
    "precision_macro": "Precision macro",
    "recall_macro": "Recall macro",
    "f1_macro": "F1 macro",
}


def ranking_table(importance, rejected_ids=(), *, unit_name="group", group_labels=None):
    import pandas as pd

    safe_name(unit_name)
    table = (
        pd.DataFrame(
            [
                {
                    f"{unit_name}_id": group,
                    unit_name: (group_labels or {}).get(
                        group, f"{unit_name.title()} {group}"
                    ),
                    "importance": value,
                }
                for group, value in importance.items()
            ]
        )
        .sort_values("importance", ascending=False, kind="stable")
        .reset_index(drop=True)
    )
    total = float(table["importance"].sum())
    shares = table["importance"] / total if total > 0 else table["importance"] * 0.0
    additional = {
        "SHAP share": shares,
        "Cumulative SHAP": shares.cumsum(),
        "Group": np.where(
            table[f"{unit_name}_id"].isin(rejected_ids),
            "Rejected in current configuration",
            "Eligible",
        ),
    }
    return table.assign(**additional)


class SelectionReport:
    """Pandas views of a SelectionResult. No estimator methods are called.

    unit_name controls presentation only. Group IDs and input-column order remain
    unchanged, including unequal-size and noncontiguous feature groups.
    """

    def __init__(self, result, *, feature_groups, unit_name="group", group_labels=None):
        import pandas as pd

        safe_name(unit_name)
        self.result, self.unit_name = result, unit_name
        n_features = sum(len(columns) for columns in feature_groups.values())
        self.groups = FeatureGroups(feature_groups, n_features)
        self.group_labels = dict(group_labels or {})
        if (
            not result.shap_history
            or tuple(result.shap_history[0].group_ids) != self.groups.group_ids
        ):
            raise ValueError(
                "Report groups/order differ from the initial recorded schema."
            )
        if result.selected_columns != self.groups.columns(result.selected_group_ids):
            raise ValueError(
                "Selected columns differ from the supplied feature groups."
            )
        plural, singular = unit_name.title() + "s", unit_name.title()
        self.trial_columns = [
            "Step",
            "Round",
            f"{singular} tried",
            "Importance",
            f"{plural} before",
            f"{plural} tried",
            "Features before",
            "Features tried",
            *METRIC_NAMES.values(),
            *[name + " change (pp)" for name in METRIC_NAMES.values()],
            "Train time (s)",
            "Validation prediction (s)",
            "Decision",
            "Failed metrics",
        ]
        # In feature mode the unit counts and physical feature counts coincide.
        self.trial_columns = list(dict.fromkeys(self.trial_columns))
        rows = []
        for trial in result.history:
            row = {
                "Step": trial.trial_id,
                "Round": trial.configuration_id + 1,
                f"{singular} tried": ", ".join(
                    self.group_labels.get(group, f"{singular} {group}")
                    for group in removed_ids(trial)
                ),
                "Importance": trial.importance,
                f"{plural} before": len(trial.groups_before),
                f"{plural} tried": len(trial.proposed_groups),
                "Features before": len(self.groups.columns(trial.groups_before)),
                "Features tried": len(self.groups.columns(trial.proposed_groups)),
                "Train time (s)": trial.train_seconds,
                "Validation prediction (s)": trial.validation_predict_seconds,
                "Decision": "Accepted" if trial.accepted else "Rejected - try another",
                "Failed metrics": ", ".join(
                    METRIC_NAMES.get(key, key) for key in trial.failed_metrics
                )
                or "None",
            }
            for key, name in METRIC_NAMES.items():
                row[name] = getattr(trial.metrics, key)
                row[name + " change (pp)"] = 100 * getattr(trial.change_from_full, key)
            rows.append(row)
        self.trials = pd.DataFrame(rows, columns=self.trial_columns)
        self.accepted_states = pd.DataFrame(
            [
                {
                    "Step": state.trial_id,
                    plural: len(state.group_ids),
                    **{
                        name: getattr(state.metrics, key)
                        for key, name in METRIC_NAMES.items()
                    },
                }
                for state in result.accepted_history
            ],
            columns=["Step", plural, *METRIC_NAMES.values()],
        )
        failed = {
            t.group_id
            for t in result.history
            if t.configuration_id == result.accepted_history[-1].configuration_id
            and not t.accepted
            and len(removed_ids(t)) == 1
        }
        self.rejected_group_ids = tuple(
            group for group in self.groups.group_ids if group in failed
        )
        self.attempted_group_ids = tuple(
            sorted({g for t in result.history for g in removed_ids(t)})
        )
        self.initial_ranking = ranking_table(
            dict(result.shap_history[0].importance),
            unit_name=unit_name,
            group_labels=self.group_labels,
        )
        self.final_ranking = ranking_table(
            dict(result.shap_history[-1].importance),
            failed,
            unit_name=unit_name,
            group_labels=self.group_labels,
        )
        self.importance_history = pd.DataFrame(
            [
                {
                    f"{unit_name}_id": group,
                    "importance": value,
                    "Step": snapshot.trial_id,
                    plural: len(snapshot.group_ids),
                }
                for snapshot in result.shap_history
                for group, value in snapshot.importance
            ],
            columns=[f"{unit_name}_id", "importance", "Step", plural],
        )
        plot_data = importance_evolution_data(result)
        self.importance_matrix = pd.DataFrame(
            plot_data["importance"],
            index=plot_data["group_ids"],
            columns=plot_data["trial_ids"],
        )
        self.importance_matrix.index.name = f"{unit_name}_id"
        self.importance_matrix.columns.name = "Step"
        self.retry_group_ids = tuple(
            sorted({row["group_id"] for row in retry_history_data(result)})
        )
        self.retry_evidence = pd.DataFrame(
            [
                {
                    f"{singular} ID": row["group_id"],
                    "Step": row["trial_id"],
                    "Round": row["configuration_id"] + 1,
                    f"{plural} before": row["groups_before"],
                    f"{plural} tried": row["proposed_groups"],
                    **{name: row["metrics"][key] for key, name in METRIC_NAMES.items()},
                    "Worst margin (pp)": row["worst_margin_pp"],
                    "Accepted": row["accepted"],
                    "Score kind": row["metric_kind"],
                    "Decision": (
                        "Accepted" if row["accepted"] else "Rejected - try another"
                    ),
                }
                for row in retry_history_data(result)
            ],
            columns=[
                f"{singular} ID",
                "Step",
                "Round",
                f"{plural} before",
                f"{plural} tried",
                *METRIC_NAMES.values(),
                "Worst margin (pp)",
                "Accepted",
                "Score kind",
                "Decision",
            ],
        )
        required_columns = {
            "Required " + name: getattr(result.effective_floors, key)
            for key, name in METRIC_NAMES.items()
            if key in result.gate_metrics
        }
        self.retry_evidence = self.retry_evidence.assign(**required_columns)
        self.group_summary = pd.DataFrame(
            [
                {
                    singular: self.group_labels.get(group, f"{singular} {group}"),
                    "Retained features": (
                        len(columns) if group in result.selected_group_ids else 0
                    ),
                    "Status": (
                        "Retained" if group in result.selected_group_ids else "Removed"
                    ),
                }
                for group, columns in self.groups.as_dict().items()
            ]
        )
        times = result.timings
        self.selection_costs = pd.DataFrame(
            [
                {
                    "Stage": "Initial SHAP calculation",
                    "Calls": 1,
                    "Time (s)": times["initial_shap_seconds"],
                },
                {
                    "Stage": "Trial training",
                    "Calls": times.get("trial_model_fit_count", len(result.history)),
                    "Time (s)": times["trial_training_seconds"],
                },
                {
                    "Stage": "Validation prediction",
                    "Calls": times.get("trial_model_fit_count", len(result.history)),
                    "Time (s)": times["validation_prediction_seconds"],
                },
                {
                    "Stage": "SHAP updates",
                    "Calls": len(result.shap_history) - 1,
                    "Time (s)": times["shap_update_seconds"],
                },
                {
                    "Stage": "Elimination loop (including recorder)",
                    "Calls": 1,
                    "Time (s)": times["elimination_loop_seconds"],
                },
                {
                    "Stage": "Initial SHAP + elimination loop",
                    "Calls": 1,
                    "Time (s)": times["initial_shap_seconds"]
                    + times["elimination_loop_seconds"],
                },
                {
                    "Stage": "Library fit (including initialization and displays)",
                    "Calls": 1,
                    "Time (s)": times["total_fit_seconds"],
                },
            ]
        )
        self.summary = pd.DataFrame(
            [
                {
                    "Property": "Selected " + unit_name + " IDs",
                    "Value": list(result.selected_group_ids),
                },
                {
                    "Property": "Selected features",
                    "Value": len(result.selected_columns),
                },
                {
                    "Property": "Removed " + plural.lower(),
                    "Value": len(self.groups.group_ids)
                    - len(result.selected_group_ids),
                },
                {
                    "Property": "Trial models trained",
                    "Value": times.get("trial_model_fit_count", len(result.history)),
                },
                {"Property": "SHAP calls", "Value": len(result.shap_history)},
                {
                    "Property": "Final rejected " + plural.lower(),
                    "Value": list(self.rejected_group_ids),
                },
                {"Property": "Stop reason", "Value": result.stop_reason},
            ]
        )

        external_cv = isinstance(result.cv_evaluation, FixedCVEvaluationTrace)
        self.score_source = ('Validation CV mean' if external_cv else
            'Train CV mean' if result.cv_evaluation is not None else 'Validation')
        if result.cv_evaluation is not None:
            trace = result.cv_evaluation
            self.trials["Score source"] = [
                (
                    "Train CV rejection upper bound"
                    if t.metric_kind == "cv_upper_bound"
                    else self.score_source
                )
                for t in result.history
            ]
            self.trials["Removed IDs"] = [list(removed_ids(t)) for t in result.history]
            self.trials["Completed folds"] = [
                len(item.fold_scores) for item in trace.trials
            ]
            self.trials["Total folds"] = len(trace.folds)
            self.accepted_states["Score source"] = self.score_source
            self.cv_membership = pd.DataFrame(
                [
                    {
                        "Fold": number,
                        "Role": role,
                        ('Development row' if external_cv else 'Train row'): int(row),
                        "Class": trace.labels[row],
                        **({"Group": trace.group_ids[row]} if trace.grouped else {}),
                    }
                    for number, fold in enumerate(trace.folds, 1)
                    for role, rows in ((('Train', fold.train_rows),
                                        ('Validation (stopping and scoring)', fold.validation_rows)) if external_cv else (
                        ("Fit", fold.train_rows),
                        ("Early stopping", fold.stopping_rows),
                        ("Held out", fold.heldout_rows),
                    ))
                    for row in rows
                ]
            )
            self.cv_scores = pd.DataFrame(
                [
                    {
                        "Trial": trial,
                        "Fold": fold,
                        "Evaluation": (
                            "Pruned: no full CV mean"
                            if isinstance(summary, CVPrunedEvaluation)
                            else "Complete"
                        ),
                        **({'Best boosting rounds': trace.best_rounds[trial][fold-1]} if external_cv else {}),
                        **{
                            label: getattr(scores, key)
                            for key, label in METRIC_NAMES.items()
                        },
                    }
                    for trial, summary in enumerate((trace.baseline, *trace.trials))
                    for fold, scores in enumerate(summary.fold_scores, 1)
                ]
            )
            self.cv_constraints = pd.DataFrame(
                [
                    {
                        "Metric": label,
                        "Full CV mean": getattr(trace.baseline.mean, key),
                        "Full CV SD": getattr(trace.baseline.standard_deviation, key),
                        "Full CV SE": getattr(trace.baseline.standard_error, key),
                        "Minimum": getattr(result.effective_floors, key),
                        "Selected CV mean": getattr(trace.final.mean, key),
                        "Margin": getattr(trace.final.mean, key)
                        - getattr(result.effective_floors, key),
                    }
                    for key, label in METRIC_NAMES.items() if key in result.gate_metrics
                ]
            )
            extra = [
                {
                    "Stage": "CV baseline training",
                    "Calls": len(trace.folds),
                    "Time (s)": times["cv_baseline_training_seconds"],
                },
                {
                    "Stage": "CV trial training",
                    "Calls": sum(len(item.fold_scores) for item in trace.trials),
                    "Time (s)": times["cv_trial_training_seconds"],
                },
                {
                    "Stage": "CV held-out prediction",
                    "Calls": times["cv_fit_count"],
                    "Time (s)": times["cv_baseline_prediction_seconds"]
                    + times["cv_trial_prediction_seconds"],
                },
            ]
            self.selection_costs = pd.concat(
                [self.selection_costs, pd.DataFrame(extra)], ignore_index=True
            )
            self.summary = pd.concat(
                [
                    self.summary,
                    pd.DataFrame(
                        [
                            {
                                "Property": "Gate score source",
                                "Value": self.score_source,
                            },
                            {
                                "Property": "CV fold model fits",
                                "Value": int(times["cv_fit_count"]),
                            },
                        ]
                    ),
                ],
                ignore_index=True,
            )
            if external_cv:
                accepted_budgets = {state.trial_id: budget for state, budget in
                                    zip(result.accepted_history, trace.refit_rounds)}
                self.cv_summary = pd.DataFrame([
                    {'Trial': trial, 'Accepted': trial in accepted_budgets,
                     'Refit boosting rounds': accepted_budgets.get(trial),
                     **{f'{label} {stat}': getattr(getattr(summary, attr), key)
                        for key, label in METRIC_NAMES.items()
                        for stat, attr in [('mean', 'mean'), ('SD', 'standard_deviation'),
                                           ('SE', 'standard_error')]}}
                    for trial, summary in enumerate((trace.baseline, *trace.trials))])
                gate_rows = []
                best = result.baseline_metrics.f1_macro
                for trial in result.history:
                    before = trial.f1_reference_before
                    after = trial.f1_reference_after
                    before = result.effective_floors.f1_macro if before is None else before
                    after = before if after is None else after
                    if trial.accepted:
                        best = max(best, trial.metrics.f1_macro)
                    gate_rows.append({'Step': trial.trial_id, 'Reference': trace.reference,
                        'Full F1': result.baseline_metrics.f1_macro,
                        'F1 reference before': before, 'Trial mean F1': trial.metrics.f1_macro,
                        'F1 reference after': after, 'Best F1 so far': best,
                        'Accepted': trial.accepted, 'Margin (pp)': 100*trial.worst_margin})
                self.f1_reference_history = pd.DataFrame(gate_rows, columns=[
                    'Step','Reference','Full F1','F1 reference before','Trial mean F1',
                    'F1 reference after','Best F1 so far','Accepted','Margin (pp)'])
                for name in ('Reference','Full F1','F1 reference before','Trial mean F1',
                             'F1 reference after','Best F1 so far','Margin (pp)'):
                    self.trials[name] = self.f1_reference_history[name].to_numpy()
                self.cv_summary = self.cv_summary.merge(self.f1_reference_history.drop(
                    columns=['Accepted']).rename(columns={'Step':'Trial'}), on='Trial',how='left')
                initial = self.cv_summary.Trial.eq(0)
                self.cv_summary.loc[initial, 'Reference'] = trace.reference
                for name in ('Full F1','F1 reference before','Trial mean F1',
                             'F1 reference after','Best F1 so far'):
                    self.cv_summary.loc[initial, name] = result.baseline_metrics.f1_macro
                self.cv_summary.loc[initial, 'Margin (pp)'] = 0.
                self.cv_constraints['Reference'] = trace.reference
                self.cv_constraints['Initial minimum'] = self.cv_constraints['Minimum']
                if trace.reference == 'best':
                    self.cv_constraints['Minimum'] = best
                    self.cv_constraints['Margin'] = self.cv_constraints['Selected CV mean']-best
                # Refits have no Validation prediction; all scoring is counted in CV.
                self.selection_costs = self.selection_costs.loc[
                    self.selection_costs.Stage != 'Validation prediction'].copy()
                self.selection_costs.loc[self.selection_costs.Stage == 'Trial training', 'Stage'] = 'Accepted development refits'
                self.selection_costs = pd.concat([self.selection_costs, pd.DataFrame([
                    {'Stage': 'Full development refit', 'Calls': 1,
                     'Time (s)': times['full_training_seconds']}])], ignore_index=True)

        self.gate_constraints = (
            self.cv_constraints.copy()
            if result.cv_evaluation is not None
            else pd.DataFrame(
                [
                    {
                        "Metric": METRIC_NAMES[key],
                        "Full validation": getattr(result.baseline_metrics, key),
                        "Minimum": getattr(result.effective_floors, key),
                        "Selected validation": getattr(result.final_metrics, key),
                        "Margin": getattr(result.final_metrics, key)
                        - getattr(result.effective_floors, key),
                    }
                    for key in result.gate_metrics
                ]
            )
        )

    def style(self, name):
        table = getattr(self, name)
        formats = {
            column: "{:.2%}"
            for column in table
            if column in METRIC_NAMES.values() or "SHAP" in column
        }
        formats.update({column: "{:+.2f}" for column in table if "(pp)" in column})
        formats.update({column: "{:.4f}" for column in table if "(s)" in column})
        formats["importance"] = formats["Importance"] = "{:.6f}"
        return table.style.format(
            {key: value for key, value in formats.items() if key in table}
        )

    def metadata(self):
        r, unit = self.result, self.unit_name
        return {
            "format_version": 1,
            "selected_group_ids": list(r.selected_group_ids),
            f"selected_{unit}_ids": list(r.selected_group_ids),
            "selected_feature_indices": list(r.selected_columns),
            "selected_feature_names": r.selected_feature_names,
            "feature_groups": self.groups.as_dict(),
            "trial_models_trained": r.timings.get(
                "trial_model_fit_count", len(r.history)
            ),
            "trial_candidates_evaluated": len(r.history),
            "speed_options": r.speed_options.to_config() if r.speed_options else None,
            "retry_evidence_scope": "Singleton rejections and later singleton acceptance; failed batches do not imply singleton failure",
            "timing_scope": (
                "CV training/prediction times sum fold durations, including concurrent fits; library fit is elapsed wall time"
                if r.cv_evaluation is not None
                else "One full-Train fit and Validation prediction per trial; library fit is elapsed wall time"
            ),
            "shap_calls": len(r.shap_history),
            "library_stop_reason": r.stop_reason,
            "library_timings_seconds": dict(r.timings),
            "gate_metrics": list(r.gate_metrics),
            "baseline_metrics": asdict(r.baseline_metrics),
            "effective_floors": asdict(r.effective_floors),
            "final_metrics": asdict(r.final_metrics),
            f"attempted_{unit}_ids": list(self.attempted_group_ids),
            f"rejected_{unit}_ids_in_final_configuration": list(
                self.rejected_group_ids
            ),
            "initial_shap_time_seconds": r.timings["initial_shap_seconds"],
            "elimination_loop_time_seconds": r.timings["elimination_loop_seconds"],
            "score_source": self.score_source,
            "scope": (
                f'External-Validation-CV mean-F1-constrained greedy selection; {r.cv_evaluation.reference} reference; no test guarantee'
                if isinstance(r.cv_evaluation, FixedCVEvaluationTrace) else
                "Train-CV-constrained greedy selection; baseline mean minus SE is a heuristic, not a test guarantee"
                if r.cv_evaluation is not None
                else "Validation-constrained greedy selection; no global minimum or test guarantee"
            ),
            **(
                {"cv_evaluation": asdict(r.cv_evaluation)}
                if r.cv_evaluation is not None
                else {}
            ),
        }

    def save(self, directory, *, prefix="selection", metadata=None, overwrite=False):
        safe_name(prefix)
        u = self.unit_name
        tables = {
            f"{prefix}_gate_constraints.csv": self.gate_constraints,
            f"{prefix}_initial_{u}_ranking.csv": self.initial_ranking,
            f"{prefix}_final_{u}_ranking.csv": self.final_ranking,
            f"{prefix}_selection_history.csv": self.trials,
            f"{prefix}_accepted_history.csv": self.accepted_states,
            f"{prefix}_{u}_summary.csv": self.group_summary,
            f"{prefix}_selection_costs.csv": self.selection_costs,
            f"{prefix}_{u}_shap_evolution.csv": self.importance_history,
            f"{prefix}_{u}_shap_evolution_matrix.csv": self.importance_matrix,
            f"{prefix}_{u}_retry_evidence.csv": self.retry_evidence,
        }
        if self.result.cv_evaluation is not None:
            tables.update(
                {
                    f"{prefix}_cv_membership.csv": self.cv_membership,
                    f"{prefix}_cv_fold_scores.csv": self.cv_scores,
                    f"{prefix}_cv_gate_constraints.csv": self.cv_constraints,
                }
            )
            if isinstance(self.result.cv_evaluation, FixedCVEvaluationTrace):
                tables[f'{prefix}_cv_summary.csv'] = self.cv_summary
                tables[f'{prefix}_f1_reference_history.csv'] = self.f1_reference_history
        generated = self.metadata()
        if set(metadata or {}).intersection(generated):
            raise ValueError(
                "Application metadata cannot override recorded selection fields."
            )
        return save_reports(
            directory,
            tables=tables,
            documents={
                f"{prefix}_selection_metadata.json": {**generated, **(metadata or {})}
            },
            prefix=prefix + "_selection",
            overwrite=overwrite,
            indexed=(f"{prefix}_{u}_shap_evolution_matrix.csv",),
        )
