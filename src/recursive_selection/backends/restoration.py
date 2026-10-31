"""Restore a built-in adapter from a saved bundle; no automatic model discovery."""
from .catboost import CatBoostAdapter
from .lightgbm import LightGBMAdapter
from .xgboost import XGBoostAdapter
from .sklearn_gbc import SklearnGBCAdapter


def adapter_from_specification(spec):
    adapters = {"catboost": CatBoostAdapter, "xgboost": XGBoostAdapter,
                "lightgbm": LightGBMAdapter, "sklearn_gbc": SklearnGBCAdapter}
    name = spec["name"]
    if name not in adapters:
        raise ValueError("Persistence supports built-in backends only.")
    return adapters[name](params=spec["params"], fit_params=spec["fit_params"],
                          early_stopping_rounds=spec["early_stopping_rounds"])
