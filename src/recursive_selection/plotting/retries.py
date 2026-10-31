"""Configuration-specific rejection and later successful removal."""

from .importance import _pyplot


def retry_history_data(result):
    """Return attempts for groups rejected before their first accepted removal."""
    grouped = {}
    for trial in sorted(result.history, key=lambda item: item.trial_id):
        # A failed batch does not establish that any singleton removal failed.
        if len(trial.removed_group_ids or (trial.group_id,)) != 1:
            continue
        grouped.setdefault(trial.group_id, []).append(trial)
    selected = []
    for group_id in sorted(grouped):
        attempts = grouped[group_id]
        passed = [trial for trial in attempts if trial.accepted]
        if passed and any(
            not trial.accepted and trial.trial_id < passed[0].trial_id
            for trial in attempts
        ):
            selected.append(group_id)
    return tuple(
        {
            "group_id": trial.group_id,
            "trial_id": trial.trial_id,
            "configuration_id": trial.configuration_id,
            "groups_before": len(trial.groups_before),
            "proposed_groups": len(trial.proposed_groups),
            "metrics": trial.metrics.as_dict(),
            "worst_margin_pp": 100 * trial.worst_margin,
            "accepted": trial.accepted,
            "metric_kind": trial.metric_kind,
        }
        for group_id in selected
        for trial in grouped[group_id]
    )


def plot_retry_history(
    result,
    *,
    title=None,
    group_labels=None,
    unit_name="group",
    margin_label=None,
    figsize=None,
    axes=None,
):
    """Return (Figure, two Axes), including an informative empty-history figure."""
    plt = _pyplot()
    import numpy as np
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MultipleLocator

    rows = retry_history_data(result)
    ids = tuple(sorted({row["group_id"] for row in rows}))
    labels = group_labels or {}
    if axes is None:
        fig, axes = plt.subplots(
            1,
            2,
            figsize=figsize or (18, max(6.5, 0.85 * len(ids) + 2)),
            sharey=True,
            gridspec_kw={"width_ratios": [1.35, 1]},
            layout="constrained",
        )
    else:
        if len(axes) != 2 or axes[0].figure is not axes[1].figure:
            raise ValueError("Provide two axes belonging to the same figure.")
        fig = axes[0].figure
    timeline_ax, margin_ax = axes
    if not ids:
        for ax in axes:
            ax.set_axis_off()
        timeline_ax.text(
            0.5,
            0.5,
            "No rejected removal later passed in this run.",
            ha="center",
            va="center",
            transform=timeline_ax.transAxes,
        )
        fig.suptitle(title or "Removal Retry History")
        return fig, axes
    fail_color, pass_color = "#b91c1c", "#15803d"
    for position, group_id in enumerate(ids):
        attempts = [row for row in rows if row["group_id"] == group_id]
        trial_ids = [row["trial_id"] for row in attempts]
        margins = [row["worst_margin_pp"] for row in attempts]
        ys = position + np.linspace(-0.20, 0.20, len(attempts))
        timeline_ax.plot(
            trial_ids,
            np.full(len(attempts), position),
            "--",
            linewidth=1.2,
            color="#9ca3af",
            zorder=1,
        )
        margin_ax.plot(margins, ys, "--", linewidth=1.1, color="#9ca3af", zorder=1)
        for index, row in enumerate(attempts):
            passed = row["accepted"]
            color, marker = (pass_color, "o") if passed else (fail_color, "x")
            trial, margin = row["trial_id"], row["worst_margin_pp"]
            timeline_ax.scatter(
                trial,
                position,
                marker=marker,
                s=85,
                color=color,
                linewidths=2,
                zorder=3,
            )
            timeline_ax.annotate(
                f"T{trial}\n{row['groups_before']} → {row['proposed_groups']}",
                (trial, position),
                xytext=(0, 11),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color=color,
            )
            margin_ax.scatter(
                margin,
                ys[index],
                marker=marker,
                s=75,
                color=color,
                linewidths=2,
                zorder=3,
            )
            margin_ax.annotate(
                f"T{trial}: {margin:+.2f}",
                (margin, ys[index]),
                xytext=(7 if passed else -7, 0),
                textcoords="offset points",
                ha="left" if passed else "right",
                va="center",
                fontsize=8.5,
                color=color,
            )
    timeline_ax.set_yticks(range(len(ids)), [labels.get(g, str(g)) for g in ids])
    timeline_ax.set_ylim(len(ids) - 0.45, -0.75)
    timeline_ax.set_xlim(0, max(row["trial_id"] for row in rows) + 3)
    timeline_ax.xaxis.set_major_locator(MultipleLocator(5))
    timeline_ax.xaxis.set_minor_locator(MultipleLocator(1))
    timeline_ax.set_xlabel(
        f"Trial number | labels: {unit_name}s before → proposed remaining {unit_name}s"
    )
    timeline_ax.set_ylabel(f"{unit_name.capitalize()} whose removal was attempted")
    timeline_ax.set_title(
        "A. Removal attempts across changing configurations", loc="left", fontsize=12
    )
    minimum = min(-0.1, min(row["worst_margin_pp"] for row in rows))
    maximum = max(0.1, max(row["worst_margin_pp"] for row in rows))
    pad = max(1.8, 0.2 * (maximum - minimum))
    margin_ax.set_xlim(minimum - pad, maximum + pad)
    margin_ax.axvspan(minimum - pad, 0, color=fail_color, alpha=0.04)
    margin_ax.axvspan(0, maximum + pad, color=pass_color, alpha=0.04)
    margin_ax.axvline(0, color="#374151", linewidth=1.2)
    cv_mode = result.cv_evaluation is not None
    margin_ax.set_xlabel(
        (
            'Validation CV F1 margin' if hasattr(result.cv_evaluation, 'refit_rounds') else
            "Worst CV gate margin / rejection upper bound"
            if cv_mode
            else "Worst validation margin"
        )
        + ", g (percentage points)\nNegative: fail | nonnegative: pass (within numerical tolerance)"
    )
    margin_ax.set_title("B. Gate margin for the same attempts", loc="left", fontsize=12)
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    timeline_ax.legend(
        handles=[
            Line2D(
                [],
                [],
                color=fail_color,
                marker="x",
                linestyle="None",
                markersize=8,
                markeredgewidth=2,
                label=f"Removal rejected; {unit_name} kept",
            ),
            Line2D(
                [],
                [],
                color=pass_color,
                marker="o",
                linestyle="None",
                markersize=7,
                label=f"Removal accepted; {unit_name} removed",
            ),
        ],
        loc="upper left",
        fontsize=9,
        frameon=True,
    )
    fig.suptitle(
        (title or f"{len(ids)} Groups Rejected Before Successful Removal")
        + "\nRemoval feasibility changes with the accepted configuration",
        fontsize=15,
    )
    fig.supxlabel(
        margin_label
        or "g = 100 × minimum gate margin; early rejection uses an upper bound",
        fontsize=10,
    )
    return fig, axes
