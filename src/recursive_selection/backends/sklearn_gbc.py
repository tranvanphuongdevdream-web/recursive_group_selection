"""Classic sklearn GBC with external Accuracy stopping and additive raw TreeSHAP."""
from copy import deepcopy
from numbers import Integral

import numpy as np
from sklearn.metrics import log_loss, accuracy_score

from ._base import BaseModelAdapter


class _ValidationMonitor:
    """Advance one persistent public staged-prediction iterator per fitted stage."""

    def __init__(self, X, y, patience, metric):
        self.X, self.y, self.patience = X, y, patience
        self.predictions = None
        self.classes = None
        self.metric = metric
        self.losses = []
        self.accuracies = []
        self.best_loss = np.inf
        self.best_accuracy = -np.inf
        self.best_rounds = 0

    def __call__(self, iteration, model, state):
        if self.predictions is None:
            self.predictions = model.staged_predict_proba(self.X)
            self.classes = model.staged_predict(self.X)
        probability = next(self.predictions)
        accuracy = float(accuracy_score(self.y, next(self.classes)))
        loss = float(log_loss(self.y, probability, labels=model.classes_))
        if not np.isfinite(loss):
            raise ValueError('Nonfinite external Validation log loss.')
        self.losses.append(loss)
        self.accuracies.append(accuracy)
        improved = accuracy > self.best_accuracy if self.metric == 'accuracy' else loss < self.best_loss
        if improved:
            self.best_loss, self.best_accuracy, self.best_rounds = loss, accuracy, iteration + 1
        return iteration + 1 - self.best_rounds >= self.patience


class SklearnGBCAdapter(BaseModelAdapter):
    """GradientBoostingClassifier; no hidden validation split or warm-start fits.

    External stopping uses strict global Validation Accuracy improvements
    (min_delta=0) and keeps the best stage, not the patience tail. The estimator's
    tol/validation_fraction do not control this external rule.
    """

    backend_name = 'sklearn_gbc'
    model_filename = 'model.joblib'

    def __init__(self, *, params=None, fit_params=None, early_stopping_rounds=5,
                 early_stopping_metric='accuracy'):
        super().__init__(params=params, fit_params=fit_params,
                         early_stopping_rounds=early_stopping_rounds,
                         early_stopping_metric=early_stopping_metric)
        self._validate_parameters(self.params)
        if 'monitor' in self.fit_params:
            raise ValueError('monitor is managed by SklearnGBCAdapter.')

    @staticmethod
    def _validate_parameters(params):
        if params.get('n_iter_no_change') is not None:
            raise ValueError('Use external early_stopping_rounds; n_iter_no_change creates an internal split.')
        if params.get('warm_start', False):
            raise ValueError('Warm-start training is not supported.')
        init = params.get('init')
        if init is not None and not (isinstance(init, str) and init == 'zero'):
            raise ValueError("TreeSHAP requires the constant init=None or init='zero'.")

    @staticmethod
    def model_class():
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier

    def fit(self, model, X, y, eval_set):
        self._validate_parameters(model.get_params(deep=False))
        options = deepcopy(self.fit_params)
        patience = self.stopping_rounds(options)
        if patience is None:
            model.fit(X, y, **options)
            return
        if isinstance(patience, bool) or not isinstance(patience, Integral) or patience < 1:
            raise ValueError('early_stopping_rounds must be a positive integer or None.')
        X_val, y_val = eval_set
        if not np.array_equal(np.unique(y), np.unique(y_val)):
            raise ValueError('Train and external Validation must contain the same classes.')
        monitor = _ValidationMonitor(X_val, y_val, int(patience), self.early_stopping_metric)
        model.fit(X, y, monitor=monitor, **options)
        best = monitor.best_rounds
        # predict/proba/SHAP must all use the same best-stage ensemble. Diagnostic
        # arrays are trimmed too; validation_log_loss_ retains the stopping trace.
        model.estimators_ = model.estimators_[:best].copy()
        model.n_estimators_ = best
        for name in ('train_score_', 'oob_improvement_', 'oob_scores_'):
            if hasattr(model, name):
                setattr(model, name, getattr(model, name)[:best].copy())
        if hasattr(model, 'oob_scores_'):
            model.oob_score_ = model.oob_scores_[-1]
        model.best_iteration_ = best - 1
        model.best_validation_loss_ = monitor.best_loss
        model.best_validation_accuracy_ = monitor.best_accuracy
        model.early_stopping_metric_ = self.early_stopping_metric
        model.validation_log_loss_ = np.asarray(monitor.losses)
        model.validation_accuracy_ = np.asarray(monitor.accuracies)

    def best_rounds(self, model):
        return int(model.n_estimators_)

    def refit(self, X, y, rounds):
        if isinstance(rounds, bool) or not isinstance(rounds, Integral) or rounds < 1:
            raise ValueError('Refit rounds must be a positive integer.')
        params, options = deepcopy(self.params), deepcopy(self.fit_params)
        for name in ('early_stopping_rounds', 'early_stopping_round'):
            params.pop(name, None)
            options.pop(name, None)
        params.update(n_estimators=int(rounds), n_iter_no_change=None)
        model = self.model_class()(**params)
        model.fit(X, y, **options)
        return model

    def feature_shap(self, model, X, y):
        """Sum signed per-tree SHAP first, then let the ranker take absolute values.

        GBC multiclass is an additive ensemble of regression trees per class.
        TreeExplainer on the full multiclass GBC is unsupported, so explain its
        public estimators_ directly. Constant initialization has zero feature
        attribution. Outputs explain decision_function margins, not probabilities.
        """
        self._validate_parameters(model.get_params(deep=False))
        try:
            from shap import TreeExplainer
        except ImportError as exc:
            raise ImportError('Install recursive-selection[gbc] for sklearn GBC TreeSHAP.') from exc
        result = np.zeros((len(X), model.estimators_.shape[1], X.shape[1]), dtype=float)
        for stage in model.estimators_:
            for output, tree in enumerate(stage):
                explainer = TreeExplainer(tree, model_output='raw',
                                          feature_perturbation='tree_path_dependent')
                values = np.asarray(explainer.shap_values(X, check_additivity=False))
                if values.shape == (len(X), X.shape[1], 1):
                    values = values[:, :, 0]
                if values.shape != (len(X), X.shape[1]):
                    raise ValueError('Unexpected regression-tree SHAP shape.')
                result[:, output, :] += model.learning_rate * values
        return result

    def save_model(self, model, path):
        import joblib
        joblib.dump(model, path, compress=3)

    def load_model(self, path, classes):
        """Load a trusted joblib model; selector.load enforces an explicit opt-in."""
        import joblib
        model = joblib.load(path)
        if not isinstance(model, self.model_class()):
            raise ValueError('Saved model is not a GradientBoostingClassifier.')
        if not np.array_equal(model.classes_, classes):
            raise ValueError('Saved GBC classes differ from the selection schema.')
        return model
