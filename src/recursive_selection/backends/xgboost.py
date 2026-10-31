"""XGBoost native TreeSHAP, using exactly the trees used for prediction."""
from copy import deepcopy

from ._base import BaseModelAdapter, normalized_shap


class XGBoostAdapter(BaseModelAdapter):
    backend_name = "xgboost"
    model_filename = "model.json"

    def best_rounds(self, model):
        try:
            return int(model.best_iteration)+1
        except AttributeError:
            return int(model.get_booster().num_boosted_rounds())

    @staticmethod
    def model_class():
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError('Install recursive-selection[xgboost] to use this backend.') from exc
        return XGBClassifier

    def fit(self, model, X, y, eval_set):
        options = deepcopy(self.fit_params)
        # XGBoost >=2.1 configures early stopping on the estimator, not fit().
        rounds = self.stopping_rounds(options)
        model.set_params(early_stopping_rounds=rounds)
        options.setdefault("verbose", False)
        model.fit(X, y, eval_set=[eval_set], **options)

    def feature_shap(self, model, X, y):
        from xgboost import DMatrix
        booster = model.get_booster()
        options = {"pred_contribs": True, "strict_shape": True}
        try:
            best = model.best_iteration
        except AttributeError:
            best = None
        if best is not None:
            options["iteration_range"] = (0, int(best) + 1)
        matrix = DMatrix(X, feature_names=booster.feature_names)
        values = booster.predict(matrix, **options)
        return normalized_shap(values, len(X), X.shape[1])

    def save_model(self, model, path):
        model.save_model(str(path))

    def load_model(self, path, classes):
        model = self.model_class()()
        model.load_model(str(path))
        return model
