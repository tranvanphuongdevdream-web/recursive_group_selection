"""Replay recorded selection state without calling an estimator or a ranker."""

from numbers import Integral

import numpy as np

from ..acceleration import RemovalPlanner, removed_ids
from ..groups import FeatureGroups
from ..reporting.export import save_reports
from ..reporting.selection import METRIC_NAMES, SelectionReport


class SelectionAudit:
    def __init__(self, checks, final_deletions, retry_configurations):
        self.checks = checks
        self.final_deletions = final_deletions
        self.retry_configurations = retry_configurations

    def save(self, directory, *, prefix="selection", overwrite=False):
        from ..reporting.export import safe_name

        safe_name(prefix)
        return save_reports(
            directory,
            tables={
                f"{prefix}_selection_feasibility_audit.csv": self.checks,
                f"{prefix}_final_deletion_margins.csv": self.final_deletions,
                f"{prefix}_retry_configuration_audit.csv": self.retry_configurations,
            },
            prefix=prefix + "_audit",
            overwrite=overwrite,
        )


def audit_selection(
    result,
    *,
    feature_groups,
    quality_policy,
    min_groups=1,
    unit_name="group",
    group_labels=None,
):
    """Raise ValueError on inconsistent records; return tables after a full replay.

    This verifies the implemented greedy rules, including the larger-ID tie rule.
    It cannot verify raw data provenance or prove a global minimum.
    """
    import pandas as pd

    def require(condition, message):
        if not bool(condition):
            raise ValueError("Selection audit failed: " + message)

    def close(a, b):
        return np.allclose(
            list(a.as_dict().values()), list(b.as_dict().values()), atol=1e-12, rtol=0
        )

    groups = FeatureGroups(feature_groups, sum(len(c) for c in feature_groups.values()))
    require(
        not isinstance(min_groups, bool)
        and isinstance(min_groups, Integral)
        and 1 <= min_groups <= len(groups.group_ids),
        "invalid minimum group count",
    )
    from ..policies import CVReferenceQualityPolicy, CVF1ReferenceQualityPolicy

    require(
        result.gate_metrics
        == getattr(quality_policy, "gate_metrics", tuple(METRIC_NAMES)),
        "gate metrics differ from the quality policy",
    )
    external_cv = isinstance(quality_policy, CVF1ReferenceQualityPolicy)
    cv_mode = external_cv or isinstance(quality_policy, CVReferenceQualityPolicy)
    if cv_mode:
        if external_cv:
            from ..fixed_cv import validate_fixed_cv_trace
            validate_fixed_cv_trace(result, quality_policy)
        else:
            from ..cross_validation import validate_cv_trace
            validate_cv_trace(result, quality_policy)
    elif result.cv_evaluation is not None:
        raise ValueError("CV results require CVReferenceQualityPolicy for audit.")
    reference = result.cv_evaluation.baseline if cv_mode else result.baseline_metrics
    require(
        close(quality_policy.thresholds(reference), result.effective_floors),
        "thresholds differ from the fixed full-model baseline",
    )
    require(
        quality_policy.evaluate(
            result.baseline_metrics, result.effective_floors
        ).passed,
        "initial full model is infeasible",
    )
    require(
        len(result.accepted_history) == len(result.shap_history) > 0,
        "accepted states and SHAP snapshot counts differ",
    )
    current = groups.group_ids
    planner = RemovalPlanner(
        min_groups,
        result.speed_options.removal_fraction if result.speed_options else 0.0,
    )
    failed = planner.failed
    configuration, snapshot_index = 0, 0
    current_metrics, current_trial = result.baseline_metrics, 0
    before, final_trial_ids = {}, []

    def check_snapshot():
        state = result.accepted_history[snapshot_index]
        snapshot = result.shap_history[snapshot_index]
        require(
            state.configuration_id == snapshot.configuration_id == configuration,
            "configuration IDs do not match accepted state",
        )
        require(
            state.trial_id == snapshot.trial_id == current_trial,
            "snapshot trial differs from acceptance",
        )
        require(
            state.group_ids == snapshot.group_ids == current,
            "accepted group schema/order differs",
        )
        require(
            close(state.metrics, current_metrics), "accepted metric differs from trial"
        )
        pairs = snapshot.importance
        require(
            len(pairs) == len(current) and tuple(g for g, _ in pairs) == current,
            "importance must contain each retained group in input order",
        )
        require(all(np.isfinite(v) and v >= 0 for _, v in pairs), "invalid importance")
        require(
            quality_policy.evaluate(state.metrics, result.effective_floors).passed,
            "accepted state violates quality thresholds",
        )
        return dict(pairs)

    ranking = check_snapshot()
    for expected_id, trial in enumerate(result.history, 1):
        require(len(current) > min_groups, "trial continued after minimum group count")
        require(
            trial.trial_id == expected_id and trial.configuration_id == configuration,
            "trial sequence/configuration is inconsistent",
        )
        removed = planner.choose(current, ranking)
        require(bool(removed), "trial continued after all candidates failed")
        chosen = removed[0]
        require(
            trial.group_id == chosen and removed_ids(trial) == removed,
            "importance order, batch backoff or tie rule differs",
        )
        proposed = tuple(g for g in current if g not in removed)
        require(
            trial.groups_before == current and trial.proposed_groups == proposed,
            "trial group schema/order differs",
        )
        require(
            np.isclose(
                trial.importance,
                np.mean([ranking[g] for g in removed]),
                atol=1e-12,
                rtol=0,
            ),
            "trial importance differs",
        )
        for key, score in trial.metrics.as_dict().items():
            require(np.isfinite(score) and 0 <= score <= 1, "invalid metric")
            require(
                np.isclose(
                    getattr(trial.change_from_full, key),
                    score - getattr(result.baseline_metrics, key),
                    atol=1e-12,
                    rtol=0,
                ),
                "metric change differs from full baseline",
            )
        decision = quality_policy.evaluate(trial.metrics, result.effective_floors)
        require(
            trial.accepted == decision.passed
            and trial.failed_metrics == decision.failed_metrics,
            "acceptance or failed metrics contradict the policy",
        )
        require(
            np.isclose(trial.worst_margin, decision.worst_margin, atol=1e-12, rtol=0),
            "worst margin differs",
        )
        require(
            all(
                np.isfinite(v) and v >= 0
                for v in (trial.train_seconds, trial.validation_predict_seconds)
            ),
            "invalid recorded time",
        )
        before[trial.trial_id] = current
        if len(removed) == 1:
            final_trial_ids.append(trial.trial_id)
        planner.observe(removed, trial.accepted)
        if trial.accepted:
            current, current_metrics, current_trial = (
                proposed,
                trial.metrics,
                trial.trial_id,
            )
            configuration += 1
            failed.clear()
            final_trial_ids.clear()
            snapshot_index += 1
            require(
                snapshot_index < len(result.shap_history), "missing accepted snapshot"
            )
            ranking = check_snapshot()
    require(snapshot_index + 1 == len(result.shap_history), "extra accepted snapshots")
    require(
        current == result.selected_group_ids
        and groups.columns(current) == result.selected_columns,
        "final retained groups or input columns differ",
    )
    require(close(current_metrics, result.final_metrics), "final metrics differ")
    if len(current) == min_groups:
        require(
            result.stop_reason == "min_groups_reached",
            "minimum-count stop reason differs",
        )
    else:
        require(
            result.stop_reason == "all_candidates_rejected" and failed == set(current),
            "final deletion attempts are incomplete",
        )
    trial_bound = (
        len(groups.group_ids) * (len(groups.group_ids) + 1)
        - min_groups * (min_groups + 1)
    ) // 2
    if result.speed_options is not None:
        # Each accepted configuration can try a shrinking batch chain, followed
        # by at most k distinct failed singleton removals before acceptance/stop.
        from math import ceil

        trial_bound = sum(
            k + max(1, ceil(k * result.speed_options.removal_fraction)).bit_length()
            for k in range(min_groups + 1, len(groups.group_ids) + 1)
        )
    require(len(result.history) <= trial_bound, "finite trial bound exceeded")
    require(
        len(result.shap_history) == 1 + sum(trial.accepted for trial in result.history),
        "SHAP call count differs",
    )
    report = SelectionReport(
        result,
        feature_groups=feature_groups,
        unit_name=unit_name,
        group_labels=group_labels,
    )
    final = report.trials.loc[report.trials["Step"].isin(final_trial_ids)].copy()
    singular = unit_name.title()
    final_columns = {
        f"{singular} ID": [
            t.group_id for t in result.history if t.trial_id in final_trial_ids
        ],
        "Worst margin (pp)": [
            100 * t.worst_margin
            for t in result.history
            if t.trial_id in final_trial_ids
        ],
    }
    final = final.assign(**final_columns)
    retry_rows = []
    for group in report.retry_group_ids:
        attempts = [
            t
            for t in result.history
            if len(removed_ids(t)) == 1 and t.group_id == group
        ]
        accepted = next(t for t in attempts if t.accepted)
        rejected = next(
            t for t in attempts if not t.accepted and t.trial_id < accepted.trial_id
        )
        require(
            before[rejected.trial_id] != before[accepted.trial_id],
            "retry occurred in the same configuration",
        )
        retry_rows.append(
            {
                singular: (group_labels or {}).get(group, f"{singular} {group}"),
                "First failed trial": rejected.trial_id,
                "Later accepted trial": accepted.trial_id,
                f"{unit_name.title()}s before fail": len(rejected.groups_before),
                f"{unit_name.title()}s before pass": len(accepted.groups_before),
                "Failed margin (pp)": 100 * rejected.worst_margin,
                "Accepted margin (pp)": 100 * accepted.worst_margin,
                "SHAP before fail": rejected.importance,
                "SHAP before pass": accepted.importance,
            }
        )
    retry = pd.DataFrame(
        retry_rows,
        columns=[
            singular,
            "First failed trial",
            "Later accepted trial",
            f"{singular}s before fail",
            f"{singular}s before pass",
            "Failed margin (pp)",
            "Accepted margin (pp)",
            "SHAP before fail",
            "SHAP before pass",
        ],
    )
    min_margin = (
        min(
            quality_policy.evaluate(s.metrics, result.effective_floors).worst_margin
            for s in result.accepted_history
        )
        * 100
    )
    observations = [
        ("Feasible full-model initialization", "Passed", "Verified"),
        (
            "SHAP priority, ties and rejection reset",
            f"All {len(result.history)} trials consistent",
            "Verified",
        ),
        (
            "All accepted states feasible",
            f"{len(result.accepted_history)} states; minimum margin {min_margin:.4f} pp",
            "Verified",
        ),
        ("Finite trial bound", f"{len(result.history)} <= {trial_bound}", "Verified"),
        (
            "SHAP calls = 1 + accepted removals",
            str(len(result.shap_history)),
            "Verified",
        ),
        (
            "Final single-deletion condition",
            (
                f"All {len(current)} deletions rejected"
                if len(current) > min_groups
                else f"Minimum group count reached ({min_groups})"
            ),
            "Verified",
        ),
        (
            "Group reduction",
            f"{len(groups.group_ids)} -> {len(current)} ({100 * (1 - len(current) / len(groups.group_ids)):.2f}% removed)",
            "Observed",
        ),
        ("Groups rejected before later acceptance", str(len(retry)), "Observed"),
    ]
    if external_cv:
        observations.extend([
            ('Gate score source', 'Fixed Full mean macro F1 on supplied Validation folds', 'Verified'),
            ('Train/Validation separation',
             f'{len(result.cv_evaluation.folds)} external folds; Validation monitors loss and scores F1', 'Verified'),
            ('Development refit budget', 'Ceiling median of each accepted candidate\'s best fold rounds; no eval_set', 'Verified'),
        ])
    elif cv_mode:
        observations.extend(
            [
                (
                    "Gate score source",
                    "Fixed Train-only CV; Validation scores are diagnostics",
                    "Verified",
                ),
                (
                    "Fit/stopping/held-out separation",
                    f"{len(result.cv_evaluation.folds)} fixed folds; held-out rows excluded from early stopping",
                    "Verified",
                ),
                (
                    "CV calibration and trial summaries",
                    "Full-fold means/SD/SE and early-rejection bounds replayed from observed folds",
                    "Verified",
                ),
            ]
        )
    return SelectionAudit(
        pd.DataFrame(observations, columns=["Property", "Observed", "Status"]),
        final,
        retry,
    )
