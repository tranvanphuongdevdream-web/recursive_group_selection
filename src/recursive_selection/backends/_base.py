"""Shared implementation helpers; no optional library is imported here."""
from copy import deepcopy
from numbers import Integral

import numpy as np


class BaseModelAdapter:
    backend_name = ""
    model_filename = ""

    def __init__(self, *, params=None, fit_params=None, early_stopping_rounds=5):
        if (early_stopping_rounds is not None and
                (isinstance(early_stopping_rounds, bool)
                 or not isinstance(early_stopping_rounds, Integral)
                 or early_stopping_rounds < 1)):
            raise ValueError("early_stopping_rounds must be a positive integer or None.")
        self.params = deepcopy(params or {})
        self.fit_params = deepcopy(fit_params or {})
        self.early_stopping_rounds = early_stopping_rounds
        forbidden = {"eval_set", "X", "y", "init_model", "xgb_model", "cat_features",
                     "text_features", "embedding_features", "categorical_feature",
                     "feature_name", "feature_names", "feature_weights"}
        if forbidden.intersection(self.fit_params):
            raise ValueError("fit_params cannot override data, feature schema or warm-start models.")
        if any(self.params.get(key) not in (None, False) for key in ("warm_start",)):
            raise ValueError("Warm-start training is not supported; every trial must train afresh.")
        for key in ("cat_features", "text_features", "embedding_features", "ignored_features",
                    "monotone_constraints", "interaction_constraints", "feature_weights",
                    "first_feature_use_penalties", "per_object_feature_penalties"):
            value = self.params.get(key)
            if value is not None and value is not False and repr(value) not in ("[]", "{}", "()", "''"):
                raise ValueError(f"{key} requires feature remapping and is not supported in this release.")

    def new_model(self):
        return self.model_class()(**deepcopy(self.params))

    def refit(self, X, y, rounds):
        """Fresh fixed-budget development fit, without validation/early stopping.

        Fold-only callbacks are intentionally not reused. LightGBM may still stop
        intrinsically when no valid split remains; rounds is the requested budget.
        """
        if isinstance(rounds, bool) or not isinstance(rounds, Integral) or rounds < 1:
            raise ValueError('Refit rounds must be a positive integer.')
        params, options = deepcopy(self.params), deepcopy(self.fit_params)
        for key in ('early_stopping_rounds', 'early_stopping_round', 'early_stopping', 'callbacks',
                    'use_best_model', 'od_type', 'od_wait', 'od_pval'):
            params.pop(key, None)
            options.pop(key, None)
        options.pop('eval_metric', None)
        if self.backend_name == 'catboost':
            for key in ('iterations', 'n_estimators', 'num_boost_round', 'num_trees'):
                params.pop(key, None)
            params.update(iterations=int(rounds), use_best_model=False)
        elif self.backend_name == 'lightgbm':
            for key in ('num_iterations', 'num_iteration', 'n_iter', 'num_tree', 'num_trees',
                        'num_round', 'num_rounds', 'nrounds', 'num_boost_round',
                        'n_estimators', 'max_iter'):
                params.pop(key, None)
            params['n_estimators'] = int(rounds)
        elif self.backend_name == 'xgboost':
            params['n_estimators'] = int(rounds)
        else:
            raise NotImplementedError('Custom CV adapters must implement refit().')
        model = self.model_class()(**params)
        model.fit(X, y, **options)
        return model

    def stopping_rounds(self, options):
        if "early_stopping_rounds" in options:
            return options.pop("early_stopping_rounds")
        for name in ("early_stopping_rounds", "early_stopping_round"):
            if self.params.get(name) is not None:
                return self.params[name]
        return self.early_stopping_rounds

    def classes(self, model):
        return np.asarray(model.classes_)

    def feature_count(self, model):
        return int(model.n_features_in_)

    def predict(self, model, X):
        return np.asarray(model.predict(X)).reshape(-1)

    def predict_proba(self, model, X):
        """Optional classification evaluation capability, in model class order."""
        return np.asarray(model.predict_proba(X), dtype=float)

    def validate_full_model(self, model):
        if not isinstance(model, self.model_class()):
            raise ValueError("full_model is incompatible with the selected backend.")
        actual = model.get_params(deep=False)
        different = []
        for key, value in self.params.items():
            candidate = actual.get(key)
            try:
                equal = np.array_equal(value, candidate, equal_nan=True)
            except (TypeError, ValueError):
                try:
                    equal = bool(value == candidate)
                except (TypeError, ValueError):
                    equal = False
            if not equal:
                different.append(key)
        if different:
            raise ValueError("full_model parameters differ from adapter params: " + ", ".join(different))

    def specification(self):
        return {
            "name": self.backend_name,
            "params": deepcopy(self.params),
            "fit_params": deepcopy(self.fit_params),
            "early_stopping_rounds": self.early_stopping_rounds,
        }


def normalized_shap(values, n_samples, n_features):
    """Remove the expected-value column; retain a class/output axis."""
    array = np.asarray(values)
    if array.ndim == 2 and array.shape == (n_samples, n_features + 1):
        return array[:, None, :-1]
    if array.ndim == 3 and array.shape[0] == n_samples and array.shape[2] == n_features + 1:
        return array[:, :, :-1]
    raise ValueError(f"Unexpected native SHAP shape {array.shape} for {n_features} features.")
