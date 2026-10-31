"""CatBoost support. The dependency is loaded only on use."""
from copy import deepcopy

from ._base import BaseModelAdapter, normalized_shap


class CatBoostAdapter(BaseModelAdapter):
    backend_name = "catboost"
    model_filename = "model.cbm"

    def best_rounds(self, model):
        best = model.get_best_iteration()
        return int(best)+1 if best is not None and best >= 0 else int(model.tree_count_)

    @staticmethod
    def model_class():
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:
            raise ImportError('Install recursive-selection[catboost] to use this backend.') from exc
        return CatBoostClassifier

    def fit(self, model, X, y, eval_set):
        options = deepcopy(self.fit_params)
        options["early_stopping_rounds"] = self.stopping_rounds(options)
        options.setdefault("use_best_model", True)
        if options["use_best_model"] is not True:
            raise ValueError("CatBoost must retain the best validation model.")
        model.fit(X, y, eval_set=eval_set, **options)

    def feature_shap(self, model, X, y):
        from catboost import Pool
        values = model.get_feature_importance(Pool(X, label=y), type="ShapValues")
        return normalized_shap(values, len(X), X.shape[1])

    def feature_count(self, model):
        # CatBoost can expose n_features_in_=0 after native load_model().
        # Its persisted ordered feature names retain the complete input schema.
        return len(model.feature_names_)

    def save_model(self, model, path):
        model.save_model(str(path))

    def load_model(self, path, classes):
        model = self.model_class()()
        model.load_model(str(path))
        return model
