"""LightGBM native TreeSHAP and portable native-model persistence."""
from copy import deepcopy

import numpy as np

from ._base import BaseModelAdapter, normalized_shap


class LightGBMAdapter(BaseModelAdapter):
    backend_name = "lightgbm"
    model_filename = "model.txt"

    def best_rounds(self, model):
        return int(getattr(model, 'best_iteration_', None) or self._booster(model).current_iteration())

    @staticmethod
    def model_class():
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ImportError('Install recursive-selection[lightgbm] to use this backend.') from exc
        return LGBMClassifier

    def fit(self, model, X, y, eval_set):
        from lightgbm import early_stopping
        options = deepcopy(self.fit_params)
        rounds = self.stopping_rounds(options)
        callbacks = list(options.pop("callbacks", []))
        if rounds is not None:
            callbacks.append(early_stopping(rounds, verbose=False))
        model.fit(X, y, eval_set=[eval_set], callbacks=callbacks, **options)

    @staticmethod
    def _booster(model):
        return getattr(model, "booster_", model)

    def feature_shap(self, model, X, y):
        booster = self._booster(model)
        best = getattr(model, "best_iteration_", None) or booster.best_iteration
        values = np.asarray(booster.predict(X, pred_contrib=True, num_iteration=best or None))
        # Native multiclass output is sample-major, then class-major blocks.
        if values.ndim == 2 and values.shape[1] != X.shape[1] + 1:
            width = X.shape[1] + 1
            if values.shape[1] % width:
                raise ValueError("Unexpected multiclass LightGBM SHAP width.")
            values = values.reshape(len(X), -1, width)
        return normalized_shap(values, len(X), X.shape[1])

    def classes(self, model):
        if hasattr(model, "classes_"):
            return np.asarray(model.classes_)
        return self._loaded_classes.copy()

    def feature_count(self, model):
        return int(self._booster(model).num_feature())

    def predict(self, model, X):
        if hasattr(model, "classes_"):
            return super().predict(model, X)
        probabilities = np.asarray(model.predict(X))
        codes = (probabilities > 0.5).astype(int) if probabilities.ndim == 1 else probabilities.argmax(axis=1)
        return self._loaded_classes[codes]

    def save_model(self, model, path):
        self._booster(model).save_model(str(path))

    def load_model(self, path, classes):
        from lightgbm import Booster
        self._loaded_classes = np.asarray(classes)
        return Booster(model_file=str(path))
