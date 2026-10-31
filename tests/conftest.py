import importlib
import importlib.machinery
import importlib.util

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--require-compiled",
        action="store_true",
        help="Reject source imports when testing a wheel.",
    )
    parser.addoption(
        "--require-all-backends",
        action="store_true",
        help="Require all backend integration dependencies, including GBC TreeSHAP.",
    )


def pytest_sessionstart(session):
    if session.config.getoption("--require-all-backends"):
        missing = [
            name
            for name in ("catboost", "xgboost", "lightgbm", "shap")
            if importlib.util.find_spec(name) is None
        ]
        if missing:
            raise pytest.UsageError(
                "Missing required integration backends: " + ", ".join(missing)
            )
    if session.config.getoption("--require-compiled"):
        modules = (
            "selector",
            "contracts",
            "groups",
            "policies",
            "results",
            "persistence",
            "cross_validation",
            "fixed_cv",
            "acceleration",
            "backends._base",
            "backends.restoration",
            "backends.catboost",
            "backends.xgboost",
            "backends.lightgbm",
            "backends.sklearn_gbc",
            "importance.shap",
            "evaluation.bootstrap",
            "evaluation.results",
            "evaluation.noninferiority",
            "plotting.importance",
            "plotting.retries",
            "plotting.charts",
            "reporting.selection",
            "reporting.classification",
            "reporting.bootstrap",
            "reporting.export",
            "evaluation.classification",
            "evaluation.comparison",
            "evaluation.audit",
            "benchmarking.timing",
            "benchmarking.environment",
            "integrations.notebook",
        )
        for name in modules:
            module = importlib.import_module("recursive_selection." + name)
            if not any(
                str(module.__file__).endswith(suffix)
                for suffix in importlib.machinery.EXTENSION_SUFFIXES
            ):
                raise pytest.UsageError(
                    f"Wheel test imported Python source instead of extension: {module.__file__}"
                )
