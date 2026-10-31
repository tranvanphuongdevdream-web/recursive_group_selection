"""Versioned JSON + model files. Joblib GBC loading requires explicit trust."""

from dataclasses import asdict
from hashlib import sha256
import json
import math
from pathlib import Path
import platform

import numpy as np

from .backends import adapter_from_specification
from .acceleration import SelectionSpeedOptions, BatchRemovalOptions
from .importance import MeanAbsoluteShapRanker
from .policies import (
    MacroQualityPolicy,
    CVReferenceQualityPolicy,
    F1ReferenceQualityPolicy,
    CVF1ReferenceQualityPolicy,
)
from .results import (
    AcceptedState,
    MacroScores,
    SelectionResult,
    ShapSnapshot,
    TrialRecord,
    CVFold,
    CVScoreSummary,
    CVEvaluationTrace,
    CVPrunedEvaluation,
    ValidationFold,
    FixedCVEvaluationTrace,
)


def _encode(value):
    if isinstance(value, np.generic):
        return _encode(value.item())
    if isinstance(value, np.ndarray):
        return _encode(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return {
            "__float__": (
                "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")
            )
        }
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (tuple, list)):
        return [_encode(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _encode(item) for key, item in value.items()}
    raise TypeError(
        f"Cannot persist {type(value).__name__}; use JSON-compatible backend settings."
    )


def _decode(value):
    if isinstance(value, dict):
        if set(value) == {"__float__"}:
            return float(value["__float__"])
        return {key: _decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item) for item in value]
    return value


def _hash(path):
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_selector(selector, directory):
    from . import __version__

    if (
        type(selector.quality_policy)
        not in (MacroQualityPolicy, CVReferenceQualityPolicy, F1ReferenceQualityPolicy, CVF1ReferenceQualityPolicy)
        or type(selector.ranker) is not MeanAbsoluteShapRanker
    ):
        raise ValueError(
            "Native save/load supports built-in quality policies and the standard SHAP ranker only."
        )
    if not hasattr(selector.adapter_, "specification"):
        raise ValueError(
            "Custom backend persistence must be implemented by the application."
        )
    specification = selector.adapter_.specification()
    if specification["name"] not in {"catboost", "xgboost", "lightgbm", "sklearn_gbc"}:
        raise ValueError("Native save/load supports built-in backends only.")
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(
            "Choose an empty/new directory; saved selections are not overwritten."
        )
    metadata = {
        "format_version": (
            8 if isinstance(selector.quality_policy, CVF1ReferenceQualityPolicy) else 6
            if isinstance(selector.quality_policy, F1ReferenceQualityPolicy)
            or isinstance(selector.speed_options, BatchRemovalOptions)
            else (
                5
                if selector.speed_options
                else (4 if selector.cv_evaluation_ is not None else 3)
            )
        ),
        "package_version": __version__,
        "python_version": platform.python_version(),
        "selection_mode": selector._selection_mode,
        "backend": specification,
        "policy": selector.quality_policy.to_config(),
        "speed_options": (
            selector.speed_options.to_config() if selector.speed_options else None
        ),
        "speed_options_type": (
            "batch" if isinstance(selector.speed_options, BatchRemovalOptions) else "cv"
        ),
        "gate_metrics": selector.result_.gate_metrics,
        "policy_type": (
            "f1_reference"
            if isinstance(selector.quality_policy, F1ReferenceQualityPolicy)
            else "macro"
        ),
        "min_groups": selector.min_groups,
        "classes": selector.classes_.tolist(),
        "n_features_in": selector.n_features_in_,
        "feature_names": selector.feature_names_in_,
        "feature_groups": selector.feature_groups_.as_dict(),
        "selected_group_ids": selector.selected_group_ids_,
        "selected_columns": selector.selected_columns_,
        "selected_feature_names": selector.selected_feature_names_,
        "baseline_metrics": asdict(selector.baseline_metrics_),
        "effective_floors": asdict(selector.effective_floors_),
        "final_metrics": asdict(selector.final_metrics_),
        "history": selector.history_records(),
        "accepted_history": [asdict(x) for x in selector.accepted_history_],
        "shap_history": [asdict(x) for x in selector.shap_history_],
        "timings": selector.timings_,
        "stop_reason": selector.stop_reason_,
        "attempted_group_ids": selector.attempted_group_ids_,
        "rejected_group_ids": selector.rejected_group_ids_,
        "model_file": selector.adapter_.model_filename,
    }
    if selector.cv_evaluation_ is not None:
        if isinstance(selector.quality_policy, CVF1ReferenceQualityPolicy):
            from .fixed_cv import validate_fixed_cv_trace
            validate_fixed_cv_trace(selector.result_, selector.quality_policy)
            policy_type = 'fixed_cv_f1'
        else:
            from .cross_validation import validate_cv_trace
            validate_cv_trace(selector.result_, selector.quality_policy)
            policy_type = 'cv_reference'
        metadata.update(policy_type=policy_type, cv_evaluation=asdict(selector.cv_evaluation_))
    # Validate serialization before writing any files.
    encoded = _encode(metadata)
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / metadata["model_file"]
    selector.adapter_.save_model(selector.estimator_, model_path)
    encoded["model_sha256"] = _hash(model_path)
    (directory / "selection.json").write_text(
        json.dumps(encoded, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return directory


def load_selector(directory, *, trusted=False):
    from .groups import FeatureGroups
    from .selector import RecursiveFeatureSelector, RecursiveGroupSelector

    directory = Path(directory)
    data = _decode(
        json.loads((directory / "selection.json").read_text(encoding="utf-8"))
    )
    data = _migrate_metadata(data)
    if data["format_version"] not in (2, 3, 4, 5, 6, 7, 8):
        raise ValueError("Unsupported selection metadata version.")
    mode = (
        data.get("selection_mode", "group")
        if data["format_version"] == 2
        else data.get("selection_mode")
    )
    if mode not in {"group", "feature"}:
        raise ValueError("Unsupported saved selection mode.")
    adapter = adapter_from_specification(data["backend"])
    # Ignore arbitrary filesystem paths in metadata; use the backend's fixed filename.
    if data["model_file"] != adapter.model_filename:
        raise ValueError("Model filename does not match backend metadata.")
    model_path = directory / adapter.model_filename
    if _hash(model_path) != data["model_sha256"]:
        raise ValueError("Saved model hash differs from selection metadata.")
    if adapter.backend_name == 'sklearn_gbc' and trusted is not True:
        raise ValueError('GBC bundles use joblib: load only a trusted source with trusted=True.')
    model = adapter.load_model(model_path, data["classes"])
    groups = FeatureGroups(
        {int(k): v for k, v in data["feature_groups"].items()}, data["n_features_in"]
    )
    if mode == "feature" and groups.as_dict() != {
        column + 1: [column] for column in range(data["n_features_in"])
    }:
        raise ValueError(
            "Saved feature selection must contain one singleton group per original column."
        )
    group_ids = tuple(data["selected_group_ids"])
    if (
        len(group_ids) != len(set(group_ids))
        or not set(group_ids).issubset(groups.group_ids)
        or not group_ids
        or groups.columns(group_ids) != tuple(data["selected_columns"])
    ):
        raise ValueError("Saved group IDs do not match the selected column schema.")
    RecursiveGroupSelector._check_model(
        adapter, model, len(data["selected_columns"]), np.asarray(data["classes"])
    )
    history = []
    for item in data["history"]:
        item = dict(item)
        for name in ("groups_before", "proposed_groups", "failed_metrics"):
            item[name] = tuple(item[name])
        if "removed_group_ids" in item:
            item["removed_group_ids"] = tuple(item["removed_group_ids"])
        item["metrics"] = MacroScores(**item["metrics"])
        item["change_from_full"] = MacroScores(**item["change_from_full"])
        history.append(TrialRecord(**item))
    accepted = tuple(
        AcceptedState(
            x["configuration_id"],
            x["trial_id"],
            tuple(x["group_ids"]),
            MacroScores(**x["metrics"]),
        )
        for x in data["accepted_history"]
    )
    snapshots = tuple(
        ShapSnapshot(
            x["configuration_id"],
            x["trial_id"],
            tuple(x["group_ids"]),
            tuple((int(k), float(v)) for k, v in x["importance"]),
        )
        for x in data["shap_history"]
    )
    names = tuple(data["feature_names"]) if data["feature_names"] is not None else None
    selected_names = (
        tuple(data["selected_feature_names"])
        if data["selected_feature_names"] is not None
        else None
    )
    if names is not None and (
        len(names) != data["n_features_in"]
        or tuple(names[c] for c in data["selected_columns"]) != selected_names
    ):
        raise ValueError("Saved feature names do not match the selected column schema.")
    policy_name = data.get("policy_type", "macro")
    if data["format_version"] in (4, 5) and policy_name != "cv_reference":
        raise ValueError("CV formats require CV policy metadata.")
    if (data['format_version'] in (7, 8)) != (policy_name == 'fixed_cv_f1'):
        raise ValueError('External-fold F1 CV requires format 7/8 and matching policy metadata.')
    if policy_name not in {"macro", "cv_reference", "f1_reference", 'fixed_cv_f1'}:
        raise ValueError("Unknown saved quality policy.")
    policy_config = dict(data['policy'])
    if policy_name == 'fixed_cv_f1':
        if data['format_version'] == 7:
            if policy_config.get('reference', 'full') != 'full':
                raise ValueError('Legacy format 7 requires the fixed Full reference.')
            policy_config['reference'] = 'full'
        elif 'reference' not in policy_config or 'reference' not in data['cv_evaluation']:
            raise ValueError('Format 8 requires explicit F1 reference provenance.')
    policy = {
        "macro": MacroQualityPolicy,
        "cv_reference": CVReferenceQualityPolicy,
        "f1_reference": F1ReferenceQualityPolicy,
        'fixed_cv_f1': CVF1ReferenceQualityPolicy,
    }[policy_name](**policy_config)
    cv_trace = (
        _restore_cv_trace(data["cv_evaluation"])
        if policy_name == "cv_reference"
        else None
    )
    if policy_name == 'fixed_cv_f1':
        cv_trace = _restore_fixed_cv_trace(data['cv_evaluation'])
    speed_type = data.get("speed_options_type", "cv")
    if speed_type not in {"batch", "cv"}:
        raise ValueError("Unknown saved speed options type.")
    if data.get("speed_options") is not None and (
        (speed_type == "cv") != (policy_name == "cv_reference")
    ):
        raise ValueError("Saved speed options do not match the quality policy.")
    speed = (
        (BatchRemovalOptions if speed_type == "batch" else SelectionSpeedOptions)(
            **data["speed_options"]
        )
        if data.get("speed_options") is not None
        else None
    )
    if data["format_version"] == 5 and speed is None:
        raise ValueError("Accelerated format requires speed options.")
    result = SelectionResult(
        model,
        group_ids,
        tuple(data["selected_columns"]),
        selected_names,
        MacroScores(**data["baseline_metrics"]),
        MacroScores(**data["effective_floors"]),
        MacroScores(**data["final_metrics"]),
        tuple(history),
        accepted,
        snapshots,
        data["timings"],
        data["stop_reason"],
        cv_trace,
        speed,
        tuple(
            data.get("gate_metrics", ("precision_macro", "recall_macro", "f1_macro"))
        ),
    )
    if cv_trace is not None:
        if policy_name == 'fixed_cv_f1':
            from .fixed_cv import validate_fixed_cv_trace
            validate_fixed_cv_trace(result, policy)
        else:
            from .cross_validation import validate_cv_trace
            validate_cv_trace(result, policy)
    selector_class = (
        RecursiveFeatureSelector if mode == "feature" else RecursiveGroupSelector
    )
    minimum = {
        "min_features" if mode == "feature" else "min_groups": data["min_groups"]
    }
    selector = selector_class(
        model_adapter=adapter, quality_policy=policy, speed_options=speed, **minimum
    )
    selector.adapter_ = adapter
    # Only the final model is persisted. The original full model is not restored.
    selector.full_model_ = None
    selector.classes_ = np.asarray(data["classes"])
    selector.n_features_in_ = data["n_features_in"]
    selector.feature_names_in_ = names
    selector.feature_groups_ = groups
    selector.initial_ranking_ = dict(snapshots[0].importance) if snapshots else {}
    selector.final_ranking_ = dict(snapshots[-1].importance) if snapshots else {}
    selector.attempted_group_ids_ = tuple(data["attempted_group_ids"])
    selector.rejected_group_ids_ = tuple(data["rejected_group_ids"])
    selector._set_result(result)
    if data["format_version"] in (6, 7, 8):
        from .evaluation import audit_selection

        audit_selection(
            result,
            feature_groups=groups.as_dict(),
            quality_policy=policy,
            min_groups=selector.min_groups,
        )
    return selector


def _migrate_metadata(data):
    """Read the version-1 sensor schema without importing the former package."""
    if data.get("format_version") != 1:
        return data
    names = {
        "min_sensors": "min_groups",
        "sensor_groups": "feature_groups",
        "selected_sensor_ids": "selected_group_ids",
        "sensor_id": "group_id",
        "sensor_ids": "group_ids",
        "sensors_before": "groups_before",
        "proposed_sensors": "proposed_groups",
        "attempted_sensor_ids": "attempted_group_ids",
        "rejected_sensor_ids": "rejected_group_ids",
    }

    def convert(value):
        if isinstance(value, dict):
            return {names.get(key, key): convert(item) for key, item in value.items()}
        if isinstance(value, list):
            return [convert(item) for item in value]
        return value

    migrated = convert(data)
    migrated["format_version"] = 2
    if migrated.get("stop_reason") == "min_sensors_reached":
        migrated["stop_reason"] = "min_groups_reached"
    return migrated


def _restore_cv_trace(data):
    def summary(value):
        if "upper_bounds" in value:
            return CVPrunedEvaluation(
                tuple(MacroScores(**score) for score in value["fold_scores"]),
                value["total_folds"],
                MacroScores(**value["upper_bounds"]),
                value["training_seconds"],
                value["prediction_seconds"],
            )
        return CVScoreSummary(
            tuple(MacroScores(**score) for score in value["fold_scores"]),
            MacroScores(**value["mean"]),
            MacroScores(**value["standard_deviation"]),
            MacroScores(**value["standard_error"]),
            value["training_seconds"],
            value["prediction_seconds"],
        )

    return CVEvaluationTrace(
        tuple(
            CVFold(
                tuple(fold["train_rows"]),
                tuple(fold["stopping_rows"]),
                tuple(fold["heldout_rows"]),
            )
            for fold in data["folds"]
        ),
        data["grouped"],
        summary(data["baseline"]),
        tuple(summary(item) for item in data["trials"]),
        summary(data["final"]),
        MacroScores(**data["validation_baseline"]),
        tuple(
            MacroScores(**item) if item is not None else None
            for item in data["validation_trials"]
        ),
        MacroScores(**data["validation_final"]),
        tuple(data["labels"]),
        tuple(data["group_ids"]) if data["group_ids"] is not None else None,
    )


def _restore_fixed_cv_trace(data):
    def summary(value):
        return CVScoreSummary(tuple(MacroScores(**score) for score in value['fold_scores']),
            MacroScores(**value['mean']), MacroScores(**value['standard_deviation']),
            MacroScores(**value['standard_error']), value['training_seconds'], value['prediction_seconds'])
    return FixedCVEvaluationTrace(
        tuple(ValidationFold(tuple(fold['train_rows']), tuple(fold['validation_rows']))
              for fold in data['folds']), data['grouped'], summary(data['baseline']),
        tuple(summary(item) for item in data['trials']), summary(data['final']),
        tuple(tuple(rounds) for rounds in data['best_rounds']), tuple(data['refit_rounds']),
        tuple(data['labels']), tuple(data['group_ids']) if data['group_ids'] is not None else None,
        data.get('reference', 'full'))
