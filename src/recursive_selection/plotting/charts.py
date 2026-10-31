"""Optional report charts from existing scores, predictions and bootstrap results."""

import numpy as np

from .importance import _pyplot


def _axes(ax, figsize):
    if ax is not None:
        return ax.figure, ax
    return _pyplot().subplots(figsize=figsize)


def _values(values, name, n=None, score=False):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or (n is not None and len(values) != n):
        raise ValueError(f"{name} must be a nonempty vector with one value per label.")
    if (
        not np.isfinite(values).all()
        or (values < 0).any()
        or (score and (values > 1).any())
    ):
        raise ValueError(
            f"{name} must be finite and nonnegative"
            + (" on the 0-to-1 scale." if score else ".")
        )
    return values


def plot_group_importance(
    importance,
    *,
    title=None,
    group_labels=None,
    total_groups=None,
    unit_name="group",
    value_label="Mean absolute importance",
    figsize=(16, 6),
    ax=None,
    top_n=None,
):
    """Plot a recorded ranking supplied as an ID-to-importance mapping."""
    from matplotlib.patches import Patch

    items = sorted(importance.items(), key=lambda item: -item[1])
    if not items:
        raise ValueError("An importance mapping with at least one group is required.")
    values = _values([value for _, value in items], "importance")
    retained_count = len(items)
    if top_n is not None:
        if (
            isinstance(top_n, bool)
            or not isinstance(top_n, (int, np.integer))
            or top_n < 1
        ):
            raise ValueError("top_n must be a positive integer or None.")
        items, values = items[:top_n], values[:top_n]
    labels = group_labels or {}
    fig, ax = _axes(ax, figsize)
    bars = ax.bar(
        [labels.get(group, str(group)) for group, _ in items], values, color="#2563eb"
    )
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    suffix = (
        f"{retained_count}/{total_groups} {unit_name}s retained"
        if total_groups is not None
        else f"{retained_count} {unit_name}s retained"
    )
    if len(items) < retained_count:
        suffix += f" | Showing top {len(items)}"
    ax.set_title((title or "Group Importance") + "\n" + suffix)
    ax.set_xlabel(unit_name.capitalize())
    ax.set_ylabel(value_label)
    ax.tick_params(axis="x", labelrotation=90)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.margins(y=0.20)
    ax.legend(handles=[Patch(facecolor="#2563eb", label=f"Retained {unit_name}s")])
    fig.tight_layout()
    return fig, ax


def plot_accepted_scores(
    result, *, title=None, unit_name="group", figsize=(10, 5), ax=None
):
    """Plot accepted macro scores and the configured fixed or best-so-far gate."""
    if not result.accepted_history:
        raise ValueError("At least one accepted configuration is required.")
    fig, ax = _axes(ax, figsize)
    counts = [len(state.group_ids) for state in result.accepted_history]
    for key, label, color in (
        ("precision_macro", "Precision macro", "#2563eb"),
        ("recall_macro", "Recall macro", "#16a34a"),
        ("f1_macro", "F1 macro", "#f59e0b"),
    ):
        minimum = getattr(result.effective_floors, key)
        ax.plot(
            counts,
            [100 * getattr(state.metrics, key) for state in result.accepted_history],
            marker="o",
            color=color,
            label=label,
        )
        if key in result.gate_metrics:
            if key == 'f1_macro' and getattr(result.cv_evaluation, 'reference', None) == 'best':
                reference = np.maximum.accumulate([state.metrics.f1_macro for state in result.accepted_history])
                ax.step(counts, 100*reference, where='post', color=color,
                        linestyle='--', alpha=.6, label='Best F1 so far')
                continue
            ax.axhline(
                100 * minimum,
                color=color,
                linestyle="--",
                alpha=0.45,
                label=f"{label} minimum: {minimum:.2%}",
            )
    ax.set_xticks(counts)
    ax.invert_xaxis()
    ax.set_ylim(0, 100)
    ax.set_title(
        title
        or (
            'Accepted Validation CV Macro Scores'
            if hasattr(result.cv_evaluation, 'refit_rounds') else
            "Accepted Train CV Macro Scores"
            if result.cv_evaluation is not None
            else "Accepted Validation Macro Scores"
        )
    )
    ax.set_xlabel(f"Number of retained {unit_name}s")
    ax.set_ylabel("Score (%)")
    ax.grid(alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig, ax


def plot_confusion_matrix(
    y_true, predictions, *, labels=None, title=None, figsize=(10, 10), ax=None
):
    """Plot supplied labels/predictions, without invoking an estimator."""
    _pyplot()
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    if len(y_true) != len(predictions):
        raise ValueError("Predictions and true labels must have equal lengths.")
    matrix = confusion_matrix(y_true, predictions, labels=labels)
    if labels is None:
        labels = np.unique(
            np.concatenate([np.asarray(y_true), np.asarray(predictions)])
        )
    fig, ax = _axes(ax, figsize)
    ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels).plot(
        ax=ax,
        xticks_rotation=90,
        values_format="d",
    )
    ax.set_title(title or "Confusion Matrix")
    fig.tight_layout()
    return fig, ax


def plot_processing_times(
    feature_times,
    inference_times,
    model_labels,
    *,
    title=None,
    caption=None,
    xlabel="Configuration",
    figsize=(12, 7),
    ax=None,
):
    """Stack existing stage timings in seconds; annotate inference in milliseconds."""
    labels = list(model_labels)
    features = _values(feature_times, "feature_times", len(labels))
    inference = _values(inference_times, "inference_times", len(labels))
    fig, ax = _axes(ax, figsize)
    feature_bars = ax.bar(
        labels, features, width=0.55, color="#2563eb", label="Feature extraction"
    )
    bars = ax.bar(
        labels,
        inference,
        bottom=features,
        width=0.55,
        color="#facc15",
        edgecolor="#facc15",
        linewidth=2.5,
        antialiased=False,
        zorder=3,
        label="Model inference",
    )
    ax.bar_label(
        bars,
        labels=[f"Total: {value:.4f} s" for value in features + inference],
        padding=5,
        fontsize=10,
    )
    ax.bar_label(
        feature_bars,
        labels=[f"Feature extraction\n{value:.4f} s" for value in features],
        label_type="center",
        color="white",
        fontsize=10,
    )
    for index, bar in enumerate(bars):
        right = index % 2 == 0
        ax.annotate(
            f"Model inference\n{inference[index] * 1000:.4f} ms",
            xy=(
                bar.get_x() + bar.get_width() / 2,
                features[index] + inference[index] / 2,
            ),
            xytext=(75 if right else -75, 35),
            textcoords="offset points",
            ha="left" if right else "right",
            va="bottom",
            fontsize=10,
            color="#854d0e",
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "edgecolor": "#facc15",
            },
            arrowprops={"arrowstyle": "->", "color": "#facc15", "linewidth": 1.2},
        )
    ax.set_title(title or "Processing Time - Stacked Bar")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Processing time (s)")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.margins(x=0.20, y=0.35)
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    if caption:
        fig.text(0.02, 0.02, caption, fontsize=8, va="bottom", ha="left")
    fig.tight_layout(rect=(0, 0.20 if caption else 0, 1, 1))
    return fig, ax


def plot_metric_comparison(
    reference_scores,
    candidate_scores,
    metric_names,
    *,
    reference_label="Reference",
    candidate_label="Candidate",
    title=None,
    figsize=(9, 5.5),
    ax=None,
):
    """Compare scores supplied on the 0-to-1 scale, displayed as percentages."""
    from matplotlib.ticker import PercentFormatter

    names = list(metric_names)
    reference = 100 * _values(
        reference_scores, "reference_scores", len(names), score=True
    )
    candidate = 100 * _values(
        candidate_scores, "candidate_scores", len(names), score=True
    )
    fig, ax = _axes(ax, figsize)
    positions = np.arange(len(names))
    width = 0.35
    for scores, shift, color, label in (
        (reference, -width / 2, "#2563eb", reference_label),
        (candidate, width / 2, "#f59e0b", candidate_label),
    ):
        bars = ax.bar(positions + shift, scores, width=width, color=color, label=label)
        ax.bar_label(
            bars, labels=[f"{score:.2f}%" for score in scores], padding=4, fontsize=10
        )
    ax.set_title(title or "Model Metric Comparison")
    ax.set_xticks(positions, names)
    ax.set_xlabel("Metric")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    fig.tight_layout()
    return fig, ax


def plot_selection_mask(
    mask,
    feature_labels,
    *,
    group_labels=None,
    unit_name="group",
    title=None,
    figsize=(12, 12),
    ax=None,
):
    """Plot a caller-defined rectangular group-by-feature boolean selection mask."""
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch

    mask = np.asarray(mask)
    if mask.ndim != 2 or not mask.size or not np.isin(mask, [False, True]).all():
        raise ValueError("mask must be a nonempty 2D boolean or 0/1 selection matrix.")
    if len(feature_labels) != mask.shape[1] or (
        group_labels is not None and len(group_labels) != len(mask)
    ):
        raise ValueError("Feature and group labels must match the mask dimensions.")
    fig, ax = _axes(ax, figsize)
    removed, retained = "#dc2626", "#2563eb"
    ax.imshow(
        mask.astype(int),
        cmap=ListedColormap([removed, retained]),
        norm=BoundaryNorm([-0.5, 0.5, 1.5], 2),
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_xticks(range(mask.shape[1]), feature_labels, rotation=45, ha="right")
    ax.set_yticks(
        range(len(mask)),
        (
            group_labels
            if group_labels is not None
            else [str(i + 1) for i in range(len(mask))]
        ),
    )
    ax.set_xlabel("Extracted feature")
    ax.set_ylabel(unit_name.capitalize())
    ax.set_title(
        (title or "Final Group × Feature Selection")
        + f"\nRetained: {int(mask.sum())}/{mask.size} pairs | "
        f"{int(mask.any(axis=1).sum())}/{len(mask)} {unit_name}s"
    )
    ax.set_xticks(np.arange(-0.5, mask.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(mask), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.legend(
        handles=[
            Patch(facecolor=retained, label="Retained pairs"),
            Patch(facecolor=removed, label="Removed pairs"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=False,
    )
    fig.tight_layout()
    return fig, ax


def plot_bootstrap_distribution(
    result,
    *,
    title=None,
    subtitle=None,
    scope_note=None,
    difference_label=None,
    bins=80,
    figsize=(12, 6.5),
    ax=None,
):
    """Draw existing bootstrap differences and their one-sided percentile bound."""
    plt = _pyplot()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    values = np.asarray(result.bootstrap_differences_pp)
    confidence = result.confidence_level
    percentile = 100 * (1 - confidence)
    lower = result.lower_bound_pp
    if isinstance(bins, bool) or not isinstance(bins, (int, np.integer)) or bins < 1:
        raise ValueError("bins must be a positive integer.")
    left, right = float(values.min()), float(values.max())
    if left == right:
        left, right = left - 0.5, right + 0.5
    edges = np.linspace(left, right, bins + 1)
    tail = values <= lower
    with plt.rc_context({"font.size": 10, "figure.facecolor": "white"}):
        fig, ax = _axes(ax, figsize)
        fig.subplots_adjust(left=0.09, right=0.98, bottom=0.29, top=0.83)
        ax.hist(
            [values[tail], values[~tail]],
            bins=edges,
            stacked=True,
            color=["#f59e0b", "#93c5fd"],
            edgecolor="white",
            linewidth=0.4,
            zorder=2,
        )
        ax.axvline(lower, color="#7c3aed", linewidth=2.3, zorder=3)
        ax.annotate(
            f"{100 * confidence:g}% one-sided lower: {lower:.2f} pp",
            xy=(lower, 0.96),
            xycoords=("data", "axes fraction"),
            xytext=(8, 0),
            textcoords="offset points",
            ha="left",
            va="top",
            color="#7c3aed",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 3},
        )
        ax.set_xlabel(
            difference_label
            or "F1 macro difference: candidate minus reference (percentage points)"
        )
        ax.set_ylabel("Bootstrap replicates (count)")
        ax.set_ylim(0, max(1, np.histogram(values, edges)[0].max()) * 1.18)
        ax.set_xlim(edges[0] - 0.3, edges[-1] + 0.3)
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle(
            title or "Bootstrap Distribution of F1 Macro Change",
            y=0.96,
            fontsize=15,
            weight="bold",
        )
        fig.text(
            0.5,
            0.89,
            subtitle
            or (
                f"{result.n_resamples:,} paired cluster replicates | {result.n_samples} samples | {result.n_clusters} clusters"
            ),
            ha="center",
            fontsize=10,
            color="#475569",
        )
        ax.legend(
            handles=[
                Patch(
                    facecolor="#f59e0b", label=f"At/below {percentile:g}th percentile"
                ),
                Patch(facecolor="#93c5fd", label="Above the lower bound"),
                Line2D(
                    [],
                    [],
                    color="#7c3aed",
                    linewidth=2.3,
                    label=f"{100 * confidence:g}% one-sided lower bound ({percentile:g}th percentile)",
                ),
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.16),
            ncol=2,
            frameon=False,
            fontsize=9,
        )
        note = scope_note or (
            f"Fixed predictions; {result.n_resamples:,} resamples reuse the evaluation data.\n"
            "Resamples are not additional independent clusters."
        )
        fig.text(0.5, 0.035, note, ha="center", fontsize=9, color="#475569")
    return fig, ax
