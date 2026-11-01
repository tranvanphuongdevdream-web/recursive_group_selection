# Recursive Selection

A Python library for selecting **individual features or complete feature groups**
through SHAP-guided recursive elimination, fresh model refits, fixed validation
thresholds, and configuration-specific retries. Use it to select whole sensors,
channels, or other groups of related columns, or to select individual columns.
It supports numeric classification data with CatBoost, XGBoost, LightGBM,
scikit-learn GBC, or a custom model adapter.

| Selection mode | Public class | Example |
| --- | --- | --- |
| Group selection | `RecursiveGroupSelector` | Remove a sensor and all its extracted statistics together |
| Feature selection | `RecursiveFeatureSelector` | Remove one statistic while retaining other features from that sensor |

The GitHub repository is
[`recursive-selection`](https://github.com/tranvanphuongdevdream-web/recursive-selection).
The installable distribution is `recursive-selection` and the Python import is
`recursive_selection`. Version 1.0.0 introduces these names; applications using
0.4.x or 0.5.x must install the new distribution and update their imports.
The library is independent of the Z24 and QUGS research notebooks and is not
published on PyPI.

## Install from GitHub Releases

CI builds **12 compiled wheels** for CPython 3.11, 3.12, and 3.13 on Windows x64,
Linux x86_64, macOS Intel, and macOS Apple Silicon. Install the renamed package
from **v1.0.0 or later** on [GitHub Releases](https://github.com/tranvanphuongdevdream-web/recursive-selection/releases).
A version is downloadable after all build and test jobs finish and CI publishes
its Release.

For **Windows x64 and CPython 3.13**, install the following into the Python
environment used by your application or notebook:

```powershell
$libraryPython = "python"
$release = Invoke-RestMethod -Uri "https://api.github.com/repos/tranvanphuongdevdream-web/recursive-selection/releases/latest" -ErrorAction Stop
$wheel = @($release.assets | Where-Object { $_.name -match '^recursive_selection-.+-cp313-cp313-win_amd64\.whl$' })
if ($wheel.Count -ne 1) { throw "The latest Release has no unique recursive-selection wheel for CPython 3.13 / Windows x64. Wait for CI publication or check GitHub Releases." }
& $libraryPython -m pip install --upgrade --no-compile "recursive-selection[catboost,plot,report] @ $($wheel[0].browser_download_url)"
if ($LASTEXITCODE -ne 0) { throw "Installing the latest library wheel failed." }
```

The command resolves the latest published GitHub Release on every invocation,
selects its CPython 3.13 / Windows x64 wheel, and passes that exact URL to
`pip install --upgrade`. It stops if the latest Release has no matching wheel.
The old 0.4.x/0.5.x assets do not match the renamed distribution. Rerun the
same command to upgrade after future Releases; restart any notebook kernel
that has loaded the binary. GitHub API availability/rate limits apply. For a
reproducible experiment, record the installed version or use a specific
Release's wheel URL instead.

This installs the compiled library, its core dependencies, and the optional
CatBoost, plotting, and reporting dependencies. No C++ compiler or Git is
required to install this wheel. For another supported Python version or platform,
copy the matching wheel's download link from the Release and pass it to
`python -m pip install --upgrade "recursive-selection[catboost,plot,report] @ WHEEL_URL"`.
Select the wheel by all three properties:

| Property | Wheel filename tag |
| --- | --- |
| CPython 3.11 / 3.12 / 3.13 | `cp311` / `cp312` / `cp313` |
| Windows x64 | `win_amd64` |
| Linux x86_64 | `manylinux...x86_64` |
| macOS Intel | `macosx...x86_64` |
| macOS Apple Silicon | `macosx...arm64` |

The Release also includes `SHA256SUMS`. Historical releases
[v0.4.0](https://github.com/tranvanphuongdevdream-web/recursive-selection/releases/tag/v0.4.0)
and [v0.5.0](https://github.com/tranvanphuongdevdream-web/recursive-selection/releases/tag/v0.5.0)
remain available under the former distribution `recursive-group-selection` and
import namespace `recursive_group_selection`. Those wheels do not provide
`recursive_selection`.

```python
from recursive_selection import (
    RecursiveGroupSelector, RecursiveFeatureSelector, CatBoostAdapter,
)
```

Repository names, distribution names, and import names are separate: clone
`recursive-selection`, install `recursive-selection`, and import
`recursive_selection`.

## Design

Both selectors use composition rather than inheriting from CatBoost. They expose
two APIs over one shared search engine; the feature API does not duplicate the loop.

| API | Removal unit | Minimum | Group mapping |
| --- | --- | --- | --- |
| `RecursiveGroupSelector` | One complete feature group | `min_groups` | Required in `fit()` |
| `RecursiveFeatureSelector` | One original column | `min_features` | Generated internally |

| Component | Responsibility |
| --- | --- |
| Selector | Coordinate the search loop and accepted state |
| Model adapter | Create fresh models, fit, predict, and validate model schemas |
| SHAP provider | Normalize backend attributions to `(samples, outputs, features)` |
| Group ranker | Compute mean absolute attribution for each feature group |
| Quality policy | Define fixed thresholds and make pass/fail decisions |
| Recorder | Receive copied events for logging, without mutable model/state access |

Four backends are available: `CatBoostClassifier`, `XGBClassifier`,
`LGBMClassifier`, and scikit-learn's classic `GradientBoostingClassifier` via
`SklearnGBCAdapter`. Optional libraries are imported when needed and are not
bundled inside this wheel. CatBoost/XGBoost/LightGBM use native TreeSHAP and do
not need `shap`. GBC needs the `gbc` extra for TreeSHAP. XGBoost and LightGBM use
their respective best iterations when computing SHAP.

This is a greedy search: the final configuration satisfies the thresholds, but
the algorithm does not guarantee the globally smallest group set. When the stop
reason is `all_candidates_rejected`, every attempt to remove **one** group from
the final configuration has been rejected in that same configuration.
`min_groups_reached` means that the user-defined minimum unit count was reached.
This shared history value also applies when `min_features` is reached in feature mode.

## Usage

Both APIs require an explicit `model_adapter`. Configure the model through
adapter `params` and training through `fit_params`; there is no estimator input,
automatic backend detection, or backend registration API.

```python
from recursive_selection import CatBoostAdapter, MacroQualityPolicy, RecursiveGroupSelector

selector = RecursiveGroupSelector(
    model_adapter=CatBoostAdapter(
        params={
            "iterations": 1000, "depth": 5, "learning_rate": 0.03,
            "loss_function": "MultiClass", "eval_metric": "Accuracy",
            "random_seed": 42, "l2_leaf_reg": 5, "random_strength": 1,
            "verbose": 20, "allow_writing_files": False,
        },
        fit_params={"early_stopping_rounds": 5, "use_best_model": True},
    ),
    quality_policy=MacroQualityPolicy(
        min_precision_macro=0.87, min_recall_macro=0.87, min_f1_macro=0.87,
        max_drop_from_full=0.04, tolerance=1e-12,
    ),
    min_groups=1,
)
feature_groups = {
    group + 1: list(range(10 * group, 10 * group + 10))
    for group in range(X_train.shape[1] // 10)
}
selector.fit(X_train, y_train, eval_set=(X_val, y_val),
             feature_groups=feature_groups, feature_names=feature_names)
model = selector.estimator_  # Fitted output model, expecting selected columns.
predictions = selector.predict(X_test)  # Selector accepts the original full schema.
```

For individual-column selection, use the same adapter and policy settings with
`RecursiveFeatureSelector`. Do not supply `feature_groups`:

```python
from recursive_selection import RecursiveFeatureSelector

feature_selector = RecursiveFeatureSelector(
    model_adapter=selector.model_adapter,
    quality_policy=selector.quality_policy,
    min_features=1,
)
feature_selector.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    feature_names=feature_names,
    # full_model=full_model,  # Optional existing full-schema fitted model.
)
indices = feature_selector.selected_columns_       # Zero-based original columns.
names = feature_selector.selected_feature_names_  # Names, or None for unnamed arrays.
model = feature_selector.estimator_
predictions = feature_selector.predict(X_test)
```

Feature mode uses `{1: [0], 2: [1], ...}` internally. Its history IDs are one-based;
`selected_columns_` is zero-based. `min_features` counts columns; `min_groups` in
group mode counts groups even when their sizes differ. A feature can be removed
while other features of the same sensor remain; use group mode to remove sensors
as complete units. Neither minimum parameter guarantees the global optimum.

Both APIs retain configuration-specific reject/retry, fixed validation floors,
fresh model refits and SHAP updates after acceptance. A rejected removal keeps
the current model and ranking; after another removal is accepted, previously
rejected candidates may be tried again in the new configuration. Feature mode
uses this same search and does not add cross-validation. Both selectors can use
CatBoost, XGBoost, LightGBM or a custom model adapter.

Reports, charts and audits consume the same result schema in either mode:

```python
from recursive_selection.reporting import SelectionReport
from recursive_selection.plotting import plot_importance_evolution

report = SelectionReport(
    feature_selector.result_,
    feature_groups=feature_selector.feature_groups_.as_dict(),
    unit_name="feature",
)
fig, ax = plot_importance_evolution(feature_selector.result_, unit_name="feature")
```

Bootstrap comparison remains independent of selection granularity: pass the
reference and selected models' aligned predictions to `PairedClusterBootstrap`.

Choose `XGBoostAdapter(params=...)`, `LightGBMAdapter(params=...)`, or
`SklearnGBCAdapter(params=...)` explicitly for other supported models. All adapters
accept keyword configuration only. Set seeds,
model parameters and the early-stopping metric for the chosen backend. Different
backends can produce different models, rankings and selection paths. Do not pass
CatBoost-specific fit options to another backend. All built-in adapters default
to `early_stopping_metric="accuracy"`, patience 5.
CatBoost maximizes Validation Accuracy; LightGBM scopes its callback to external
Validation Accuracy only; XGBoost minimizes `merror` (multiclass) or `error`
(binary) and saves the best ensemble; GBC maximizes external Validation Accuracy.
Training objectives remain their configured losses. Accuracy ties keep the first
maximum; log loss does not break those ties. Each fit receives fresh stopping state.
Use `early_stopping_metric="configured"` to retain native caller-configured metrics
(or the former external log-loss monitor for GBC). Old saved adapter specifications
without this field restore that former behavior.

If a full model is already fitted, configure `params=full_model.get_params()` in
the adapter and pass `full_model=full_model` to `fit()` to reuse it. The library
checks backend, parameters, feature count and classes; the caller owns data,
column-order and training-protocol provenance. No input model is retrained in place.

`estimator_` (with a trailing underscore) remains the fitted output model; it is
unrelated to the removed constructor parameter. Custom backends remain supported
by passing a `model_adapter` implementing the protocols in `contracts.py`.

## Rules and supported data

- `X_train` and `X_val` must be finite, dense, two-dimensional numeric matrices
  with the same column order. Extract features and clean NaN values outside the
  library. Raw time-series tensors, sparse matrices, categorical/text/embedding
  inputs, and feature weights are not currently supported.
- `y` contains one label per row. XGBoost requires consecutive integer classes
  starting at zero. Encode labels consistently before calling the library if
  you want to switch backends easily.
- `feature_groups` maps positive integer group IDs to zero-based column indices.
  Each column must belong to exactly one group. Groups need not be contiguous or
  have equal sizes. Retained columns always follow the original matrix order.
- DataFrame column names and order are checked. For NumPy inputs, the caller is
  responsible for preserving the physical column order.
- Each threshold is `max(absolute_floor, original_full_score - max_drop_from_full)`.
  `0.04` means four percentage points, not a relative reduction of four percent.
- The full model must satisfy the thresholds before selection starts. Macro
  scores use the fixed class set from the full model.
- No group is initially protected. The group with the lowest SHAP importance is
  tried first; ties are resolved by trying the higher group ID first.
- Every trial trains a fresh model. A failed trial retains the accepted model and
  ranking. An accepted trial clears the rejection set and recomputes SHAP.
  Training/prediction errors propagate as errors, not metric-based rejections.
- Test, holdout, and bootstrap data do not participate in selection. Validation
  data may be reused for early stopping and selection; use an independent
  evaluation protocol appropriate for the problem.
- Attributions are aggregated over features; they are not group Shapley values or
  causal effects. Native binary SHAP may have one raw-output axis; multiclass
  SHAP has one output axis per class.

## Results and logging

`fit()` returns `self`. The following attributes are available after fitting:

```text
estimator_ / full_model_
selected_group_ids_ / selected_columns_ / selected_feature_names_
baseline_metrics_ / effective_floors_ / final_metrics_
history_ / accepted_history_ / shap_history_
initial_ranking_ / final_ranking_
attempted_group_ids_ / rejected_group_ids_
timings_ / stop_reason_ / result_
```

History and snapshots are immutable dataclass records. `history_records()` returns
a list of dictionaries suitable for pandas. `trial_id` starts at one;
`configuration_id` starts at zero and increases only after an accepted removal.
Scores, margins, and differences use the 0-to-1 scale; multiply by 100 to express
them in percentage points. The loop timing excludes full-model training and
initial SHAP. `total_fit_seconds` also includes preparation, validation, and
initialization. Identical timings, models, or group sets across operating
systems are not guaranteed.

An optional recorder must implement `record(event)`. Each event is a deep-copied
dictionary. Events are `run_started`, `initialized`, `trial_started`,
`trial_completed`, `configuration_accepted`, and `run_finished`.
Callback errors cause fitting to fail; they are not hidden as trial rejections.

## Saving and loading

```python
selector.save("my_selection")  # Requires a new or empty directory.
restored = RecursiveGroupSelector.load("my_selection")
predictions = restored.predict(X_test)

feature_selector.save("my_feature_selection")
restored_features = RecursiveFeatureSelector.load("my_feature_selection")
```

The native model (`.cbm`/`.json`/`.txt`) is saved alongside `selection.json`, which
contains the schema, policy, backend configuration, classes, history, timings, and
model SHA-256 hash. These three backends do not use pickle. GBC uses
`model.joblib` and requires `load(path, trusted=True)`; only load bundles from a
trusted source, using matching scikit-learn/joblib versions. The hash detects
modified bytes but does not authenticate a bundle's origin. Only the final model is saved;
`full_model_` is `None` after loading.

Version 0.5.0 writes metadata format 3 with the selection mode and restores the
matching selector type. Load with the same public class that saved the selection.
Existing metadata formats 1 and 2 remain readable as group selections. Older
library versions cannot read format 3.

After loading, LightGBM uses a native `Booster` in `estimator_`. Use
`restored.predict(X_full)` to obtain class labels using the saved class mapping.
The backend runtime version must be able to read the corresponding native model.
Custom policies, rankers, backends, and callbacks that are not JSON-compatible
require application-specific persistence. They are not automatically
deserialized or executed from metadata.

## Scikit-learn GradientBoostingClassifier

Install `recursive-selection[gbc]` to include the optional `shap` runtime:

```python
from recursive_selection import (
    SklearnGBCAdapter, RecursiveFeatureSelector,
    CVF1ReferenceQualityPolicy, BatchRemovalOptions,
)

selector = RecursiveFeatureSelector(
    model_adapter=SklearnGBCAdapter(
        params=dict(n_estimators=1000, learning_rate=.03, max_depth=3,
                    loss="log_loss", random_state=42),
        early_stopping_rounds=5,
    ),
    quality_policy=CVF1ReferenceQualityPolicy(),
    speed_options=BatchRemovalOptions(removal_fraction=.1),
)
selector.fit_cv(X_dev, y_dev, cv_splits=cv_splits, cv_groups=recording_groups)
selector.save("gbc_selection")
restored = RecursiveFeatureSelector.load("gbc_selection", trusted=True)
```

This adapter supports both `fit()` and `fit_cv()`, for feature or group selection.
It uses sklearn's public `fit(monitor=...)` and `staged_predict_proba()` APIs to
evaluate the supplied external Validation after each boosting stage. Persistent
staged probability/class iterators advance once per stage; they do not repeatedly predict all prior
stages. A strictly larger Validation Accuracy updates the checkpoint
(`min_delta=0`); configured legacy mode instead minimizes log loss. After `early_stopping_rounds` stages without improvement, the
fit stops. The model is trimmed to the best stage, including when it reaches
`n_estimators`. Set `early_stopping_rounds=None` to train the entire budget.
Checkpoint diagnostics are `best_iteration_` (zero-based),
`best_validation_loss_` (loss at the chosen checkpoint), `best_validation_accuracy_`,
`validation_log_loss_` and `validation_accuracy_` (including the patience tail).
`adapter.best_rounds(model)` returns the one-based number of retained stages.

Keep `n_iter_no_change=None`: sklearn's built-in stopping would make a separate
internal split and is rejected by this adapter. `tol` and `validation_fraction`
do not control external stopping. Warm starts, custom monitors and feature-dependent
`init` estimators are rejected. `init=None` and `init="zero"` are supported.
GBC does not have an `n_jobs` or GPU setting; a model fit is single-threaded.
The existing selection gate, batching, retries and fixed-fold membership apply
unchanged. Full and accepted `fit_cv()` configurations receive fresh development
refits using their own ceil-median best-stage budgets, without a Validation split
or stopping monitor.

TreeSHAP explains each regression tree in `estimators_`, then adds **signed**
contributions across stages, scaled by `learning_rate`, before the ranker takes
absolute values. This supports multiclass without relying on TreeExplainer's
unsupported direct multiclass GBC path. Output shape is `(samples, 1, features)`
for binary or `(samples, classes, features)` for multiclass. These attributions
explain raw `decision_function` margins, not probabilities. Constant initialization
has no feature attribution. Computation visits each retained tree and can be slower
than the other backends' native ensemble SHAP calls.

Unlike the other three backends, GBC models use joblib serialization.
`load(..., trusted=True)` is required for a trusted GBC bundle; never use that
flag for an unknown source. Keep compatible sklearn/joblib versions for loading.
CatBoost, XGBoost and LightGBM retain their existing native loading behavior.

API references: [sklearn GBC](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html),
[SHAP TreeExplainer](https://shap.readthedocs.io/en/stable/generated/shap.TreeExplainer.html).

## Extending backends

Small protocols are defined in `contracts.py`. Supply a custom `model_adapter`
and, when needed, a custom `ranker`. No registration step is required. The default ranker
also requires the adapter to provide `feature_shap(model, X, y)` with shape
`(samples, outputs, features)`, excluding the base-value column.
If a model does not support SHAP, supply a different ranker.
The search loop calls these contracts without branching on library names.

## Building locally on Windows

You can read, edit, and test the source without building it. To build binaries,
use 64-bit CPython 3.11, 3.12, or 3.13 and MSVC C++ Build Tools compatible with
that Python version. Clone the repository, then run from its root directory:

```powershell
git clone https://github.com/tranvanphuongdevdream-web/recursive-selection.git
Set-Location -LiteralPath ".\recursive-selection"
py -3.13 -m venv .venv
$libraryPython = ".\.venv\Scripts\python.exe"
& $libraryPython -m pip install "build>=1.2,<2"
& $libraryPython -m build --wheel
```

`python -m build` creates an isolated build environment and installs the build
dependencies declared in `pyproject.toml`. It runs Cython on the `.py` modules,
compiles them into `.pyd` files, and creates a wheel in `dist/`. Small
`__init__.py` files remain as public facades; duplicate Python implementation
modules are not included alongside their binaries in the wheel.
Do not use `python setup.py install`.

After building, replace the wheel filename below with the actual output filename:

```powershell
& $libraryPython -m pip install ".\dist\recursive_selection-1.0.0-cp313-cp313-win_amd64.whl[catboost,plot,report]"
```

Available extras are `catboost`, `xgboost`, `lightgbm`, `gbc`, `all`, `plot`, `report`,
`notebook`, `test`, and `dev`.
You do not need to install all optional model libraries. A `.pyd` binary cannot be
shared across all three operating systems. Compiling the orchestration code is
not expected to substantially reduce training or SHAP computation costs.

## Testing source without building or installing the package

From the package directory, use an environment with NumPy, scikit-learn, pytest,
pandas, matplotlib, and the backend you want to test. The commands below assume
the local `.venv` created above; an existing compatible environment also works:

```powershell
$libraryPython = ".\.venv\Scripts\python.exe"
$env:PYTHONPATH = (Resolve-Path ".\src").Path
$env:PYTHONDONTWRITEBYTECODE = "1"
& $libraryPython -m pytest tests -p no:cacheprovider
```

Tests for missing backends are skipped. Use `--require-all-backends` to require
all model libraries plus `shap` for GBC. Testing source does not invoke setup,
Cython, or a compiler.
To test an installed wheel, open a new terminal or remove the source directory
from `PYTHONPATH`, run outside the checkout, and use:

```powershell
python -m pytest <path-to-tests> --require-compiled --require-all-backends
```

`--require-compiled` verifies that every implementation module is imported from
a `.pyd` or `.so` file. The `__init__.py` facades may remain pure Python.

## Automatic versions, binary wheels and GitHub Releases

The workflow `.github/workflows/build_group_selection.yml` runs on pushes to
`master` and can also be started with **Run workflow**. It plans versions using
Python Semantic Release 10.7.0, builds native wheels, tests their installed
copies, and publishes all platforms together to a GitHub Release. It does not
publish to PyPI.

Use Conventional Commit messages to describe the impact of changes:

| Commit | Next version from 1.0.0 |
| --- | --- |
| `fix: correct a calculation` | 1.0.1 |
| `perf: reduce prediction overhead` | 1.0.1 |
| `feat: add a ranking method` | 1.1.0 |
| `feat!: change the selector API` or a `BREAKING CHANGE:` footer | 2.0.0 |
| `docs: update examples`, `chore: ...`, `ci: ...` | No new release |

The highest-impact change since the last version tag determines the bump. A
commit such as `update code` cannot communicate that impact and does not trigger
a release. Non-release changes still build and test wheels. You do not manually
edit version numbers or create tags for normal releases.

The one-time bootstrap published source version **0.5.0** after the existing
`v0.4.0` tag, including the previously untagged feature-selection API. CI updates
`project.version` in `pyproject.toml` and `__version__` in the
public package together, generates `CHANGELOG.md`, and pushes a release commit
and tag. Release commits contain `[skip ci]` to prevent duplicate builds. Both
version preparation and wheel publication happen in the same workflow: a tag
created using `GITHUB_TOKEN` is not relied on to trigger another workflow.

```text
Conventional commit + push to master
  -> plan/stamp version, changelog, release commit and tag
  -> build/test the exact release commit on four native runners
  -> verify all 12 wheels, CPython ABIs, metadata and compiled modules
  -> upload wheels + SHA256SUMS to a draft release
  -> publish only when every platform and upload succeeded
```

| Target | CPython | Wheel |
| --- | --- | --- |
| Windows x64 | 3.11, 3.12, 3.13 | win_amd64 |
| Linux x86_64 | 3.11, 3.12, 3.13 | manylinux_2_28_x86_64 |
| macOS Intel | 3.11, 3.12, 3.13 | macosx x86_64 |
| macOS Apple Silicon | 3.11, 3.12, 3.13 | macosx arm64 |

`cibuildwheel` requires all four model backends (including GBC TreeSHAP) in its integration tests and
checks that implementation modules come from `.pyd`/`.so`, not source. macOS
backend tests install the OpenMP runtime `libomp` when needed. Backend users may
also need this runtime according to their model library's instructions.

CI serializes releases, skips superseded queued pushes, and pins every build
job to the prepared commit SHA. The version commit/tag is prepared before the
platform builds, so a failed build can leave an unpublished version tag. Use
**Re-run failed jobs** on the original run to retry those jobs with its prepared
commit. **Run workflow** on `master` can also retry if `master` still points to
that tagged release commit. Preparation recognizes the tag and reuses the same
version. Artifact uploads get a second attempt for
transient service failures. Incomplete uploads remain drafts. A complete public
release is left unchanged on reruns; an incomplete public release fails rather
than silently replacing published files. Existing releases are retained.

The workflow uses GitHub's automatic `GITHUB_TOKEN`, with `contents: write`
limited to version preparation and publication. No personal token is required
for an ordinary unprotected `master` branch. Repository rules must permit the
release bot to push its version commit/tag if branch/tag protection is enabled.

CI pushes version/changelog commits back to `master`. Before starting the next
change locally, fetch that commit:

```powershell
git pull --ff-only
# Edit and test the library.
git add <files-you-changed>
git commit -m "fix: correct selection history"
git push origin master
```

Wheels also remain downloadable as Actions artifacts. Choose a wheel matching
the Python version, OS and CPU architecture, then install it with optional
backend/report extras. The version tag and wheel filename must agree.

For a local read-only version preview, install the release tooling in a developer
environment and run:

```powershell
python -m pip install "python-semantic-release==10.7.0" packaging
python .github/scripts/release.py plan
```

`plan` prints JSON without changing files, creating commits/tags, or publishing.
The automation's tests run with `python -m pytest .github/tests`. Source tests
and the Windows local build commands above remain available independently of CI.

## Paired cluster bootstrap of fixed-model predictions

```python
from recursive_selection.evaluation import PairedClusterBootstrap

bootstrap = PairedClusterBootstrap(
    n_resamples=50_000, confidence_level=0.90, random_state=42, batch_size=1_000,
)
comparison = bootstrap.compare(
    y_true=y_test,
    reference_predictions=full_predictions,
    candidate_predictions=selected_predictions,
    cluster_ids=recording_ids,
    strata=y_test,
)
print(comparison.observed_difference_pp)
print(comparison.lower_bound_pp)
comparison.save("evaluation")
decision = comparison.noninferiority(margin=0.04)
print(decision.passed)
```

The comparison metric is F1 macro with a fixed class set. Differences are
always candidate minus reference. Score units are 0 to 1; properties ending in
`_pp` use percentage points. The confidence bound is the empirical lower
quantile at `1 - confidence_level`, using linear interpolation. This percentile
bound is approximate; it is not a bootstrap-tail p-value. With a 90% one-sided
bound, the associated nominal significance level is 10%.

Every cluster is resampled as a whole with shared weights for the two prediction
vectors. Cluster IDs may be strings or numbers. If strata are supplied, a cluster
must belong to exactly one stratum and each stratum must contain at least two
clusters. `strata=y_test` therefore requires class-pure clusters. Omitting strata
uses an unstratified cluster bootstrap, including mixed-class clusters.
The number of clusters drawn is fixed within each stratum; the number of rows
in a replicate can vary with cluster size. Resampling uses sorted cluster and
stratum order. Reproduce the seed, batch size, input order and software versions
when comparing exact replicate arrays.

For non-inferiority, define the margin before inspecting evaluation results.
`margin=0.04` means four percentage points. The decision rejects
`H0: delta <= -margin` only when the one-sided lower bound is strictly greater
than `-margin`. Failure to reject does not prove inferiority. Interpretation
depends on the cluster/stratum definitions and independence of the evaluation
protocol. No evaluation result feeds back into selection or refits either model.

`save()` writes `bootstrap_comparison.json` and `bootstrap_differences.csv`.
It overwrites those report filenames when called again in the same directory.

## Optional plots of the recorded search

Install the `[plot]` extra for Matplotlib. Importing the selector or the plotting
namespace does not load Matplotlib; drawing a chart does.

```python
from recursive_selection.plotting import (
    plot_importance_evolution, plot_retry_history,
)

fig, ax = plot_importance_evolution(
    selector.result_, title="Feature-group importance through accepted configurations",
    group_labels={1: "Channel A", 2: "Channel B"},
)
fig.savefig("importance.png", dpi=300, bbox_inches="tight")

fig, axes = plot_retry_history(selector.result_)
fig.savefig("retries.png", dpi=300, bbox_inches="tight")
```

Both functions return Figure/Axes without calling `show()` or saving files.
They read recorded history only, without training or recomputing explanations.
The importance heatmap shares one scale across accepted configurations; removed
groups are NaN, not zero, and an x marks the first configuration after removal.
Use `value_label` to identify SHAP or a custom ranker's importance units.
The retry chart includes groups rejected before a later accepted removal; an
empty retry history produces an informative empty chart. Its margins come from
the recorded quality-policy decisions, in percentage points.

All report charts are available from the same optional plotting namespace:

| Function | Input |
| --- | --- |
| `plot_group_importance` | Recorded group-ID-to-importance mapping |
| `plot_accepted_scores` | Selection result with accepted configurations and fixed floors |
| `plot_confusion_matrix` | True labels and existing predictions |
| `plot_processing_times` | Existing feature-extraction and inference times in seconds |
| `plot_metric_comparison` | Two sets of scores on the 0-to-1 scale and metric labels |
| `plot_selection_mask` | Caller-defined rectangular group-by-feature boolean mask |
| `plot_importance_evolution` | Recorded importance snapshots from a selection result |
| `plot_retry_history` | Recorded trial decisions from a selection result |
| `plot_bootstrap_distribution` | Existing `BootstrapComparison` result |

Every function returns Figure/Axes and leaves display and file saving to the
caller. None calls a model, recomputes attributions, reruns selection, or draws
additional bootstrap samples. Timing charts keep inference segments at their
actual measured heights even when they are much smaller than extraction times.
The bootstrap histogram handles constant replicate values and labels tied values
as at/below the bound rather than claiming that exactly ten percent lie below it.

```python
from recursive_selection.plotting import plot_bootstrap_distribution

fig, ax = plot_bootstrap_distribution(
    comparison,
    title="Bootstrap distribution of F1 macro change",
    difference_label="F1 macro difference: candidate minus reference (percentage points)",
)
fig.savefig("bootstrap.png", dpi=300, bbox_inches="tight")
```

## Migration from 0.4.x / 0.5.x

Starting with 1.0.0, the distribution name is `recursive-selection` and the
import namespace is `recursive_selection`. Install the new wheel and replace
`recursive_group_selection` with `recursive_selection` in application imports:

```python
from recursive_selection import (
    RecursiveGroupSelector, RecursiveFeatureSelector,
    CatBoostAdapter, MeanAbsoluteShapRanker, MacroQualityPolicy,
)
from recursive_selection.evaluation import PairedClusterBootstrap
from recursive_selection.reporting import SelectionReport
```

The selector class names and arguments are unchanged. Existing 0.4.x / 0.5.x
installations continue to expose only their original namespace; installing the
new distribution does not rewrite notebooks or replace the old distribution.
This package does not provide an alias for the former namespace. Native
model/JSON bundles keep the same format and can be loaded through the matching
selector class in the new namespace. The historical wheels and Releases retain
their original filenames.

## Migration from 0.1.0

Version 0.4.0 introduced the distribution `recursive-group-selection`, import
namespace `recursive_group_selection`, and class `RecursiveGroupSelector`. These
package names were subsequently replaced in 1.0.0 as described above. The former
`shm-sensor-selection` distribution and `sensor_selection` import are not aliases
of the new package. Existing installations can coexist because the namespaces differ.
Use `feature_groups`, `min_groups`, and `selected_group_ids_` in new code.
The selection loader accepts version-1 native model/JSON exports and translates
their sensor field names to group field names. Version 0.4.0 writes metadata
format 2; version 0.5.0 writes format 3 for both selection modes.
The legacy project directory has been removed. The maintained library source is
this standalone project root; saved version-1 model bundles remain loadable
without installing the retired package.

## References

- [Cython compilation](https://docs.cython.org/en/latest/src/userguide/source_files_and_compilation.html)
- [Python wheel formats](https://packaging.python.org/en/latest/discussions/package-formats/)
- [cibuildwheel](https://cibuildwheel.pypa.io/en/stable/)
- [XGBoost Python API](https://xgboost.readthedocs.io/en/stable/python/python_api.html)
- [LightGBM Python API](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMClassifier.html)
- [SciPy bootstrap conventions and one-sided confidence intervals](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)


## Reports, fixed-model evaluation and benchmarks (0.4.0)

These APIs consume records or already fitted models; they never refit a model or
change selection decisions. Install `[report]` for pandas tables, `[plot]` for
Matplotlib, or `[notebook]` for pandas, plotting and the optional IPython recorder.
Core selector/bootstrap do not require pandas, Matplotlib or IPython. Scikit-learn
may import pandas transitively when it is already installed; report construction
is the only part of this library that directly requests pandas.

```python
from recursive_selection.reporting import SelectionReport, style_metrics
from recursive_selection.evaluation import (
    evaluate_classifier, compare_evaluations, audit_selection,
)
from recursive_selection.benchmarking import ProcessingBenchmark, collect_environment

report = SelectionReport(selector.result_, feature_groups=feature_groups)
display(report.style("trials"))
display(report.summary)
report.save("reports/selection", prefix="example")

full_eval = evaluate_classifier(
    model=selector.full_model_, adapter=selector.adapter_,
    splits={"Validation": (X_val, y_val), "Test": (X_test, y_test)},
)
selected_eval = evaluate_classifier(
    model=selector.estimator_, adapter=selector.adapter_,
    splits={"Validation": (selector.transform(X_val), y_val),
            "Test": (selector.transform(X_test), y_test)},
)
comparison = compare_evaluations(reference=full_eval, candidate=selected_eval)
display(style_metrics(comparison.class_metrics("Validation")))
display(style_metrics(comparison.validation_constraints(
    policy=selector.quality_policy, baseline=selector.baseline_metrics_,
)))
audit = audit_selection(selector.result_, feature_groups=feature_groups,
                        quality_policy=selector.quality_policy, min_groups=selector.min_groups)
display(audit.checks)
audit.save("reports/audit", prefix="example")
```

Evaluation caches label predictions, probabilities, per-class metrics and text
reports. All model classes contribute to macro averages (`zero_division=0`),
including classes absent from a split. Probability columns must follow the
adapter's class order; log loss is aligned even when those labels are unsorted.
Custom adapters need `predict_proba(model, X)` only when probabilities are requested.
Use `include_probabilities=False` for label-only classifiers (log loss is unavailable).
The separate `ClassificationPredictor` protocol does not expand selector requirements.
Comparisons require identical split order, classes and ordered true labels. The
caller must ensure actual row identity; equal labels alone do not prove alignment.

```python
benchmark = ProcessingBenchmark.from_evaluations(
    reference=full_eval, candidate=selected_eval,
    reference_feature_times={"Validation": full_val_feature_seconds, "Test": full_test_feature_seconds},
    candidate_feature_times={"Validation": selected_val_feature_seconds, "Test": selected_test_feature_seconds},
    reference_groups=len(feature_groups), candidate_groups=len(selector.selected_group_ids_),
)
environment = collect_environment(packages=("catboost", "numpy", "recursive-selection"),
                                  inference_device="CPU", inference_threads="Model default")
benchmark.save("reports/benchmark", prefix="example", environment=environment)
```

Aggregation makes **no extra predictions**. Its timings describe one recorded
label-prediction call per split, with zero warmup. Probability prediction, scoring,
rendering and environment collection are outside the inference timer. Feature
extraction/preprocessing durations are supplied by the application; the library
does not know how raw time series become features. Batch averages are not request
latency. For explicit repeated measurements use
`benchmark_prediction(predict_callable, X, repeats=10, warmup=2)`; this is separate
from single-run aggregation and reports every measurement plus the median.

Reports support unequal-size/noncontiguous groups. `unit_name="sensor"` and optional
`group_labels` affect presentation, not IDs or model input order. `SelectionReport`
contains `trials`, `accepted_states`, `initial_ranking`, `final_ranking`,
`importance_history`, `importance_matrix`, `retry_evidence`, `group_summary`,
`selection_costs` and `summary`. All tables are built from detached records.
Report saves default to refusing collisions. Opt into `overwrite=True` for reruns;
unrelated files are retained. Per-report JSON manifests list filenames and SHA256
hashes. The report exporter does not provide a multi-file transaction.
`selector.save(new_empty_directory)` remains the native model/schema persistence
API; `SelectionReport.save` exports tables and metadata, not a loadable estimator.

`NotebookSelectionRecorder` is available under `integrations.notebook`. Construct
it in the initial-importance section and pass it as `recorder` to selection below;
its display handles keep the initial ranking/chart in the original notebook position.
It observes copied events and never fits, ranks or makes acceptance decisions.

`BootstrapReport` under `reporting` validates bootstrap scores against cached
full/selected evaluations and writes optional compatibility CSV/summary and an
explicit non-inferiority decision. Pass `margin` in score units, e.g. `0.04`.
The existing `BootstrapComparison.save` and plotting APIs remain unchanged.
Audits replay the recorded greedy rules, larger-ID importance ties and rejection
reset, including minimum-group stopping and no-removal runs. They do not establish
data provenance, independence or a global minimum.

## Automatic recording-disjoint development/Test splits

`fit_recording_cv()` accepts the entire **already extracted** two-dimensional
feature matrix, labels and aligned per-sample metadata. It creates a fixed Test
pool and four repeated Train/Validation holdouts, then selects on development only.
It does not extract raw time-series features or search model hyperparameters.
Install `recursive-selection[recording]` (pandas) to use metadata policies.

```python
from recursive_selection import (
    RecursiveFeatureSelector, CatBoostAdapter, CVF1ReferenceQualityPolicy,
    BatchRemovalOptions, SetupRecordingSplit, RecordingSplit,
)

selector = RecursiveFeatureSelector(
    model_adapter=CatBoostAdapter(
        params={"iterations": 1000, "depth": 4, "learning_rate": 0.03,
                "loss_function": "MultiClass", "random_seed": 42,
                "thread_count": 1, "verbose": False, "allow_writing_files": False},
        early_stopping_rounds=5, early_stopping_metric="accuracy",
    ),
    quality_policy=CVF1ReferenceQualityPolicy(reference="best"),
    speed_options=BatchRemovalOptions(removal_fraction=0.1, cv_jobs=4, cpu_budget=6),
)
selector.fit_recording_cv(
    X, y, metadata=recording_metadata,
    split_policy=SetupRecordingSplit.z24(seed=42),
    feature_names=feature_names,
)
selector.split_.save("new_split_directory")
# Only after selection/refit has finished; Test was never supplied to fit or SHAP:
rows = list(selector.test_indices_)
predictions = selector.predict(X[rows])  # use X.iloc[rows] for a DataFrame
```

Z24 metadata columns are `sample_id`, `condition_id`, `recording_id`, `setup_id`
and `segment_id`; `verified`, when present, must be true. Metadata rows must match
X/y row-for-row. Source Setup IDs 0..8 become groups 1..9. All ten segments of a
recording stay together. The policy uses one outer RNG and a separate CV RNG,
both seed 42, over sorted source conditions. Choose one Test from groups 3..7;
choose four distinct Validation pairs from groups 2..8 excluding Test, minimum
pair distance 2. Six eligible groups have Validation appearances [1,1,1,1,2,2];
groups 1 and 9 remain Train. Each split has six Train / two Validation / one Test
groups. This is repeated holdout evaluation, not exhaustive out-of-fold CV.
Numerical group distance does not establish elapsed time or physical separation.
Defaults match the baseline's canonical pair enumeration and RNG draw order.
No label count or feature width is fixed in the library. Column names, index
base, seeds, candidate groups, minimum distance and segments per recording are
policy settings. This balanced protocol currently requires nine groups and four
holdouts. Split SD/SE are descriptive; overlapping holdouts do not justify an
independence assumption or inferential confidence interval.

For OpenBridgeLAB, `SetupRecordingSplit.openbridgelab()` consumes `global_sample_id`,
`label` (original source label), `recording_key`, `segment_id`, `source_datetime`.
It orders whole recordings by source time per condition and uses `array_split`
to form nine contiguous near-equal recording-count groups. Recording intervals
must not overlap; timestamps must identify an unambiguous order. A group may
contain many recordings, so six/two/one groups need not imply exact sample ratios.

To retain the Damage 1/2 baseline's **parent** allocation, pass metadata for all
three source conditions, without any Reference features or signals:

```python
active = parent_metadata.loc[parent_metadata.label.isin([1, 2])].reset_index(drop=True)
# X_damage rows align with active; y_damage has already been mapped to 0/1.
selector.fit_recording_cv(
    X_damage, y_damage, metadata=active,
    parent_metadata=parent_metadata, label_mapping={1: 0, 2: 1},
    split_policy=SetupRecordingSplit.openbridgelab(seed=42),
    feature_names=feature_names,
)
```

Use a binary model objective (e.g. CatBoost `Logloss`) for that binary baseline.
Allocation visits all parent source conditions before retaining active conditions;
it never restarts RNG on the two model labels. Active conditions must be complete.
Original source sample IDs/labels and model labels are kept separately. For the
current baseline this preserves Test T6/T6, development 2030 and Test 250 rows;
Z24 has development 1360 and Test 170 rows. Source/model IDs are not inferred
from feature values or guessed from row counts.

`split_.membership_table()` exposes input row, original sample/condition/recording,
group, model label, pool, four roles and Validation count. `split_.folds` indexes
**development locally**; `development_rows`/`test_rows` index the original input.
`split_.save()` exports immutable recipe JSON, membership CSV and index NPZ.
Reuse a saved recipe with `RecordingSplit.load(...)` and
`selector.fit_recording_cv(X, y, split=saved_split)`. For existing baseline row
indices use `RecordingSplit.from_indices(metadata, y, development_rows=...,
test_rows=..., cv_splits=...)`; its fold indices are development-local. Predefined
splits are validated for class/recording separation but do not claim the automatic
policy's time/group constraints. Reuse requires the same input row order and labels.

Full and every trial share folds. Only development rows reach fits, refits and
SHAP. Acceptance compares the complete unweighted mean Validation macro F1 to
the running best F1, retaining batch/backoff/retry and bounded parallel folds.
Full and accepted subsets refit on all development with their own ceiling-median
best-round budget and no eval_set or stopping callback. No fold model is deployed.
Test indices are returned for an explicit final evaluation; `fit_recording_cv()`
does not automatically score Test or let Test influence selection.

Format 9 persists the split recipe and validates it against saved CV membership,
labels, recording groups and source metadata on load. Old bundles remain readable.
The final selected model is persisted; the in-memory `full_model_` is not restored.
Selection reports additionally export recording membership and Validation coverage.

## External-fold mean F1 selection matching a baseline

Use `fit_cv()` when the baseline already has fixed development Train/Validation
folds. This is an opt-in protocol, separate from single-Validation `fit()` and
the Train-only CV gate described below. Neither existing protocol changes.

```python
from recursive_selection import (
    RecursiveFeatureSelector, CatBoostAdapter,
    CVF1ReferenceQualityPolicy, BatchRemovalOptions,
)

selector = RecursiveFeatureSelector(
    model_adapter=CatBoostAdapter(
        params={"iterations": 1000, "depth": 4, "learning_rate": 0.03,
                "loss_function": "MultiClass", "eval_metric": "Accuracy",
                "random_seed": 42, "thread_count": 2,
                "verbose": False, "allow_writing_files": False},
        early_stopping_rounds=5,
    ),
    quality_policy=CVF1ReferenceQualityPolicy(reference="best"),
    speed_options=BatchRemovalOptions(removal_fraction=0.1, cv_jobs=4, cpu_budget=6),
)
selector.fit_cv(
    X_dev, y_dev,
    cv_splits=baseline_folds,       # [(train_indices, validation_indices), ...]
    cv_groups=development_groups,   # original recording or time-block IDs
    feature_names=feature_names,
)
selected_model = selector.estimator_
print("Final fixed refit budget:", selector.refit_rounds_)
# Only after selection is frozen:
test_predictions = selector.predict(X_test)
```

`cv_splits` indexes **X_dev locally**, not the original complete dataset. At least
two folds are required; four is a caller choice rather than a library constant.
Each fold must partition development into disjoint, nonempty, class-complete
Train/Validation sets. By default (`fold_kind="partition"`), every development row
must appear in Validation exactly once. Explicit `fold_kind="repeated"` allows
unused/repeated Validation rows and requires distinct holdouts. It still enforces
complete per-fold development coverage, class support and group separation.
Supplied `cv_groups` stay disjoint within every fold. For recording/time-block
experiments, supply the actual composite IDs used by the baseline (including the
condition when IDs are only unique within a condition). The library does not
construct or reshuffle these folds and never accepts a separate Test set.
`RecursiveGroupSelector.fit_cv()` also requires its usual `feature_groups` mapping.

The caller supplies the same fixed model hyperparameters, training seed and checkpoint
monitor as the baseline; there is no per-trial hyperparameter search. Full and
every trial train a **fresh model for every fold**. Validation selects the
early-stopping checkpoint and supplies the fold score, intentionally matching
the baseline protocol. New adapters monitor Accuracy by default; set their patience
and training objectives to match the baseline. Complete fold macro F1 scores are averaged **without weighting by
fold size**; this is not macro F1 recomputed from pooled out-of-fold predictions.

The default acceptance rule uses the highest complete CV mean F1 found so far:

```text
best_F1 = original_Full_mean_macro_F1
PASS = trial_mean_macro_F1 + numerical_tolerance >= best_F1
if PASS:
    best_F1 = max(best_F1, trial_mean_macro_F1)
```

For example, Full 78% → trial 80% PASS raises the threshold to 80%; a subsequent
79% FAILS even though it exceeds Full. An equal 80% can PASS while reducing
features. Numerical tolerance does not lower the running maximum.
`CVF1ReferenceQualityPolicy(reference="full")` explicitly retains the former fixed
Full rule. This setting changes external `fit_cv()` only; the single-Validation
`F1ReferenceQualityPolicy` and Train-only CV policies retain their existing rules.

To compare against the fixed Full mean after truncating both sides to two decimal
places, use:

```python
quality_policy = CVF1ReferenceQualityPolicy(reference="full", truncate_decimals=2)
```

Compute the complete unweighted fold mean first, then truncate toward zero
(all F1 scores are nonnegative). Full `0.689989` gives a fixed gate of `0.68`;
a trial mean of `0.688421` also gives `0.68` and passes, while `0.679999` gives
`0.67` and fails. An accepted `0.708418` does not raise the fixed gate. This is
truncation, rather than display rounding or an allowed fixed drop of `0.01`.
Truncated gate values are compared exactly; `tolerance` applies only when
`truncate_decimals=None`, the default. Raw fold scores, mean/SD/SE and the best
observed F1 retain their original precision. The final model is the last accepted
subset, so its raw F1 can be below Full or an earlier accepted subset.
Reports include raw means, truncated gate values and precision. Format 10 bundles
persist the precision in both the policy and CV trace and validate it on load;
older untruncated bundles retain their original behavior.

Precision, recall, sample SD and SE remain diagnostics. There is no mean-minus-SE
allowance or requirement to win every individual fold. `best_F1` compares complete
unweighted fold means, never the highest individual fold or a Test score.

All trials complete all supplied folds. `BatchRemovalOptions(cv_jobs=4,
cpu_budget=6)` runs up to four fold workers with a bounded CPU budget. For four
CPU folds on a six-core budget, each model uses at most one thread. Available CPU
quotas, fold count and a smaller budget can reduce worker count. GPU adapters use
one worker. GBC has no `n_jobs` setting. Folds receive separate fresh models,
adapter settings and stopping state; events are emitted on the caller thread and
results are collected in fold order. Trials remain sequential: the next trial
needs the accepted feature subset, updated SHAP and current F1 reference.
One failed worker aborts the run without accepting a partial-fold score.
`cv_jobs=1` is the default. No pruning or Train-only `SelectionSpeedOptions` is
used in this protocol. Single-Validation `fit()` ignores these fold-worker options.

For sequential/parallel comparisons, keep per-model thread counts and seeds the
same. Changing a native learner's thread count can change floating-point results;
parallelism is not a guarantee of identical checkpoints under different thread
settings or of a fourfold speedup. Fold durations are summed in cost tables;
`total_fit_seconds` is elapsed wall time. Development refits retain the original
adapter's thread configuration after the fold pool has finished.

The existing ascending-SHAP/batch/backoff/retry search is preserved. With 320
features and fraction 0.1, a rejected batch shrinks 32 → 16 → 8 → 4 → 2 → 1.
A failed batch does not mark its members individually failed. A rejected
singleton becomes temporarily ineligible. An accepted removal resets failed
singletons and batch size, so previously rejected features can be retried in the
new configuration. Zero fraction requests single-feature removal.

Full and **accepted** candidates receive a new fit on all development, without
an eval_set or early stopping. Its requested boosting-round budget is
`ceil(median(best_rounds_per_fold))`, using one-based rounds (including the +1
conversion for zero-based CatBoost/XGBoost best iterations). Each candidate uses
its own CV rounds, not Full's budget. Native learners can still stop intrinsically
if no valid split remains, especially LightGBM. Fold-only callbacks are not reused
during refit. SHAP ranking is calculated on development using this refitted model;
it is not out-of-fold SHAP. Rejected candidates receive no development refit and
leave the accepted model/ranking unchanged. The last accepted refit is the final
estimator; no fold model is deployed and no additional final fit is necessary.

Custom adapters additionally implement `best_rounds(model)` and
`refit(X_dev, y_dev, rounds)` (see `CVRefitAdapter`). All four built-in adapters
provide these operations. Existing adapters/settings are not
mutated. Full reuse from an unrelated fitted estimator is intentionally absent
from `fit_cv()` because its fold provenance would not be established.

`selector.cv_evaluation_` records immutable external fold membership, complete
fold summaries, per-trial best-round counts and accepted refit budgets.
`SelectionReport` labels the score source **Validation CV mean**, exports fold
scores/best rounds, membership, F1 gate constraints and an additional
`*_cv_summary.csv` containing means/SD/SE/refit budgets. An additional
`*_f1_reference_history.csv` records Full F1, the gate before each trial, trial
mean F1, the gate after it, running best F1, decision and margin. Accepted-score
charts show the running reference for best mode. `selector.best_f1_` reports the
highest accepted F1, including Full; `effective_floors_` retains the **initial**
Full threshold. Audit replays each threshold transition, fold coverage, gate
decisions, batch retries and median budgets.
Partition external-CV bundles use format 8 with explicit reference provenance.
Repeated/automatic recording holdouts use format 9, including fold kind, checkpoint
metric and (when present) the complete metadata split recipe.
Legacy format 7 bundles load as `reference="full"`, preserving their original
decisions; formats 1–6 remain readable. Older library versions cannot load format 8.

Repeated adaptive use of these Validation folds is selection, not an independent
performance estimate. Keep Test outside selection, SHAP, stopping and tuning.
The F1-constrained greedy search does not guarantee a global minimum feature set or
unseen-data performance.

## Automatic full-reference CV quality gate

`CVReferenceQualityPolicy` is an opt-in alternative to manual absolute floors and
maximum drops. `MacroQualityPolicy` and its existing defaults remain supported.

```python
from recursive_selection import (
    RecursiveFeatureSelector, CatBoostAdapter, CVReferenceQualityPolicy,
)

selector = RecursiveFeatureSelector(
    model_adapter=CatBoostAdapter(
        params={"iterations": 1000, "depth": 5, "learning_rate": 0.03,
                "loss_function": "MultiClass", "random_seed": 42,
                "verbose": False, "allow_writing_files": False},
    ),
    quality_policy=CVReferenceQualityPolicy(n_splits=3, random_state=42),
)
selector.fit(
    X_train, y_train, eval_set=(X_val, y_val),
    feature_names=feature_names,
    cv_groups=train_recording_ids,  # omit only when recording/group IDs are unavailable
    # full_model=already_fitted_reference,  # same schema/parameters/protocol required
)
```

For each macro metric independently, the selector fits a full-feature CV reference
and fixes `max(0, full_cv_mean - se_multiplier * full_cv_se)`. The default multiplier
is 1.0. SE is the sample standard deviation (`ddof=1`) of the fold scores divided
by the square root of the number of folds. Every proposed removal uses the same
fold membership and must pass all three fixed thresholds. Reject/retry, SHAP
updates after acceptance and original-column ordering are unchanged. Thresholds
are never recomputed from an accepted candidate.

CV sees **Train rows only**. The outer Validation set still controls early
stopping for the full reference and each deployment candidate, but its scores do
not determine CV-gate acceptance. Within each CV fold, the fitting pool is split
again into fit and stopping rows. Scored held-out rows never enter early stopping.
The sample split reserves approximately 25% of the fitting pool for stopping.
Grouped CV uses a class-complete grouped stopping split with up to four partitions
(fewer for small group counts), preserving group disjointness among fit, stopping
and held-out sets. All three sets must contain every training class; insufficient
samples/groups raise an error instead of silently falling back to sample splits.

`cv_groups` refers to **original recording/group IDs for Train rows**, not feature
groups or invented IDs. Supplied groups are kept disjoint in both levels of CV.
Without them, the design is stratified sample-level CV and cannot establish
recording independence. CV groups require the CV policy; the original macro
policy does not silently ignore them.

The CV reference and every trial add `n_splits` fresh model fits, in addition to
the original full/candidate fits. CV models are discarded, and native SHAP ranking
continues to explain the accepted deployment model. Fold membership, per-fold
scores, sample SD/SE, timing and outer Validation diagnostics are available through
`selector.cv_evaluation_` and exported by `SelectionReport` as CV membership,
fold-score and gate-constraint tables. In this mode `baseline_metrics_`,
`final_metrics_`, trial metrics and acceptance plots mean **Train CV mean**, not
outer Validation performance. Outer Validation scores remain separate diagnostics.
`SelectionReport.cv_constraints` displays the appropriate gate comparison;
`EvaluationComparison.validation_constraints` refuses to mislabel outer Validation
scores as CV gate scores.

Built-in native save/load supports the CV policy with format 4, including fixed
membership and all CV evidence. Existing macro-policy formats 1–3 remain readable.
`audit_selection` replays fixed thresholds, fold separation, CV statistics and the
same greedy/retry decisions without training again. Recorder events include
`cv_fold_started`/`cv_fold_completed` with trial/fold numbers for live monitoring.

This baseline-anchored mean-minus-SE gate is a **selection heuristic**. It is not
Probatus's best-path parsimonious subset rule, a confidence interval, a formal
non-inferiority test or an absolute deployment-quality guarantee. Correlated CV
fits limit the interpretation of the fold-based SE; a variable reference yields
more permissive floors. The adaptive search and repeated use of CV still require
independent outer evaluation of the complete selection procedure. Keep Test and
external datasets outside calibration, acceptance and model/seed tuning.

### Opt-in accelerated CV selection

```python
from recursive_selection import SelectionSpeedOptions

selector = RecursiveFeatureSelector(
    model_adapter=adapter,
    quality_policy=CVReferenceQualityPolicy(),
    speed_options=SelectionSpeedOptions(
        cv_jobs=2,                 # bounded by available CPU quota/affinity
        removal_fraction=0.10,     # 0: singleton search for comparison
        # cpu_budget=4,            # optional total CPU budget for CV
    ),
)
```

The original singleton implementation remains the default (`speed_options=None`).
Acceleration requires `CVReferenceQualityPolicy` and combines four changes:

1. Evaluate candidate CV **before** fitting the deployment model. Rejected
   candidates receive no deployment fit; accepted candidates receive a fresh fit
   for SHAP and subsequent deployment. The initial full model is still fitted or
   reused, as before.
2. For each metric after `m` observed folds out of `K`, calculate the maximum
   possible full-CV mean `(sum(observed_scores) + K - m) / K`. If any upper bound
   is below its fixed floor minus numerical tolerance, reject without fitting the
   remaining folds. Bounds are rounded outward. A low individual fold alone is
   not a rejection rule. Accepted candidates always complete all folds.
3. With early rejection enabled, score one pilot fold, then evaluate the remaining
   folds in bounded parallel waves. Bounds are checked between waves; already
   running fits finish and are all recorded. Baseline calibration always completes
   every fold. Model CPU threads share the configured budget, respecting quotas
   and affinity; GPU adapters use one fold worker. CatBoost CV workers run quietly
   so they do not flood notebook logs. Recorder callbacks remain on the caller
   thread. Fold order and membership stay fixed.
4. Try the lowest ascending-SHAP fraction of eligible units as a batch, capped to
   preserve the minimum count. A failed batch is halved (rounding down, minimum
   one) until singleton attempts are reached; batch failure does **not** mark its
   members individually failed. Singleton failures remain ineligible only until
   the next successful removal. Success recomputes SHAP, resets failed singleton
   attempts and restores the fractional batch size. Selection stops at the minimum
   count or after every remaining singleton removal fails in the current state.

Batch removal changes the search path and can select different features from
singleton elimination. It does not remove the quadratic worst case. The same
fixed quality floors apply to every accepted batch. For a controlled comparison,
use `removal_fraction=0.0`; `early_reject=False` disables only the bound optimization.

Partial CV is represented by `CVPrunedEvaluation` with observed fold scores and
`upper_bounds`, **without a full-CV mean/SD/SE**. Its trial `metric_kind` is
`cv_upper_bound`; reports label those gate values as rejection upper bounds.
Rejected CV-first trials have no outer Validation diagnostics. Batch trials record
all `removed_group_ids`; legacy `group_id` identifies the first proposed unit.
Retry charts/counts describe singleton failures and later singleton successes,
not failed batches. Reports count actual CV/deployment fits and export the speed
configuration. CV training/prediction times sum individual fold durations, which
can overlap; total library fit time is elapsed wall time. Native bundles use
format 5 and restore speed settings, bounds, batches and fold evidence. Formats
1–4 remain readable. Audits replay batch backoff, singleton retry, upper-bound
certificates and the final singleton stopping condition without fitting models.

## Fixed Validation F1 reference and batch removal

`F1ReferenceQualityPolicy` accepts a candidate when its Validation macro F1 is at
least the original full-feature model's Validation macro F1 (within numerical
`tolerance=1e-12`). The reference stays fixed even if an intermediate subset
scores higher. Precision and Recall are recorded as diagnostics, not constraints.

```python
from recursive_selection import (
    RecursiveFeatureSelector, CatBoostAdapter,
    F1ReferenceQualityPolicy, BatchRemovalOptions,
)

selector = RecursiveFeatureSelector(
    model_adapter=CatBoostAdapter(
        params={"iterations": 1000, "depth": 5, "random_seed": 42,
                "thread_count": -1, "verbose": False},
        fit_params={"early_stopping_rounds": 5, "use_best_model": True},
    ),
    quality_policy=F1ReferenceQualityPolicy(),
    speed_options=BatchRemovalOptions(removal_fraction=0.1),
    min_features=1,
)
selector.fit(X_train, y_train, eval_set=(X_val, y_val))
```

Each trial fits one fresh model on the entire Train set, predicts Validation,
and applies the fixed F1 gate. This mode trains no CV models. An optional
`full_model` is reused under the existing same-data/parameters/protocol contract.
Test must stay outside selection. Repeated Validation use can overfit that set;
this policy does not solve validation adaptation or guarantee Test performance.

Batch removal starts with `ceil(current_count * removal_fraction)` lowest-SHAP
units, capped by eligible units and the minimum retained count. Ties use larger
original IDs first. Failed batches are halved (floor, at least one), without
marking members individually failed. A rejected singleton is skipped until the
next accepted removal. Acceptance keeps that model, recalculates exact SHAP on
Train and resets retry eligibility and batch size. Rejections do not recalculate
SHAP. Search stops at the requested minimum or after all current singleton
removals fail. Batches change the search path and do not remove the quadratic
worst case. Use fraction zero or omit speed options for singleton search.

`BatchRemovalOptions` supports Validation policies; the existing
`SelectionSpeedOptions` remains specific to the CV gate. Native metadata format 6
persists the F1 policy, batch settings and active `gate_metrics`. Older formats
remain readable. `SelectionReport.gate_constraints` reports only active
constraints; all three macro metrics remain in trial histories. Score plots draw
only active threshold lines, and `audit_selection` replays the gate and scheduling.
