from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .model import FitObject


@dataclass
class AssignmentResult:
    dataframe: pd.DataFrame
    figure: plt.Figure | None


def _resolve_test_labels(fit_object: FitObject, combo: str | None) -> tuple[str, str]:
    if fit_object.raw_pvals is None or fit_object.bh_fdrs is None:
        raise ValueError("fit_edgepython must be run before plotting.")

    if combo is None:
        if "beta_cis" in fit_object.raw_pvals.columns and "beta_trans" in fit_object.raw_pvals.columns:
            return "beta_cis", "beta_trans"
        return "null: no cis", "null: no trans"

    cis_label = f"{combo} null: no cis"
    trans_label = f"{combo} null: no trans"
    if cis_label in fit_object.raw_pvals.columns and trans_label in fit_object.raw_pvals.columns:
        return cis_label, trans_label

    alt_cis = f"beta_cis*{combo}"
    alt_trans = f"beta_trans*{combo}"
    if alt_cis in fit_object.raw_pvals.columns and alt_trans in fit_object.raw_pvals.columns:
        return alt_cis, alt_trans

    raise KeyError(f"Could not resolve test labels for combo '{combo}'.")


def plot_pval_histograms(fit_object: FitObject, combo: str | None = None) -> plt.Figure:
    if fit_object.raw_pvals is None or fit_object.bh_fdrs is None:
        raise ValueError("fit_edgepython must be run before plotting.")

    cis_label, trans_label = _resolve_test_labels(fit_object, combo)
    raw_cis = fit_object.raw_pvals[cis_label]
    raw_trans = fit_object.raw_pvals[trans_label]
    fdr_cis = fit_object.bh_fdrs[cis_label]
    fdr_trans = fit_object.bh_fdrs[trans_label]

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    axes[0, 0].hist(raw_cis, bins=30, color="steelblue")
    axes[0, 0].set_title("Raw P Values")
    axes[0, 0].set_xlabel(cis_label)
    axes[0, 0].set_ylabel("Count")

    axes[0, 1].hist(raw_trans, bins=30, color="steelblue")
    axes[0, 1].set_xlabel(trans_label)
    axes[0, 1].set_ylabel("Count")

    axes[1, 0].hist(fdr_cis, bins=30, color="darkred")
    axes[1, 0].set_title("Benjamini-Hochberg corrected FDRs")
    axes[1, 0].set_xlabel(cis_label)
    axes[1, 0].set_ylabel("Count")

    axes[1, 1].hist(fdr_trans, bins=30, color="darkred")
    axes[1, 1].set_xlabel(trans_label)
    axes[1, 1].set_ylabel("Count")

    fig.tight_layout()
    return fig


def _combo_projection_matrix(
    fit_object: FitObject,
    combo: str | None,
    interaction_designator: str,
) -> tuple[np.ndarray, pd.Series, pd.Series]:
    if fit_object.bh_fdrs is None:
        raise ValueError("fit_edgepython must be run before assignment.")

    weight_names = fit_object.design_matrix_full.columns.to_list()
    combo_x = np.zeros((4, len(weight_names)), dtype=float)
    cis_label, trans_label = _resolve_test_labels(fit_object, combo)
    fdrs_no_cis = fit_object.bh_fdrs[cis_label]
    fdrs_no_trans = fit_object.bh_fdrs[trans_label]

    if combo is None:
        combo_x[:, 0] = 1.0
        combo_x[1, 1] = 1.0
        combo_x[3, 1] = 1.0

        if fit_object.trans_model == "log_additive":
            combo_x[0, 2] = 1.0
            combo_x[2, 2] = 0.5
            combo_x[3, 2] = 0.5
        else:
            combo_x[0, 2] = 1.0
            combo_x[2, 2] = 1.0
            combo_x[3, 2] = 1.0
        return combo_x, fdrs_no_cis, fdrs_no_trans

    fields = fit_object.fields_to_test or []
    unique_categories = {
        field_name: [str(v) for v in pd.unique(fit_object.metadata[field_name])]
        for field_name in fields
    }
    all_fields = [item for values in unique_categories.values() for item in values]
    combo_split = set(combo.split(interaction_designator))
    bad_fields = [field for field in all_fields if field not in combo_split]

    for index, weight in enumerate(weight_names):
        keep = not any(bad_field in weight for bad_field in bad_fields)
        cis = keep and "beta_cis" in weight
        trans = keep and "beta_trans" in weight

        if cis:
            combo_x[1, index] = 1.0
            combo_x[3, index] = 1.0

        if trans:
            combo_x[0, index] = 1.0
            if fit_object.trans_model == "log_additive":
                combo_x[2, index] = 0.5
                combo_x[3, index] = max(combo_x[3, index], 0.5)
            else:
                combo_x[2, index] = 1.0
                combo_x[3, index] = 1.0

        if keep and not cis and not trans:
            combo_x[:, index] = 1.0

    return combo_x, fdrs_no_cis, fdrs_no_trans


def get_assignments_and_plot(
    fit_object: FitObject,
    combo: str | None = None,
    make_plot: bool = True,
    cell_size: float = 10000.0,
    alpha: float = 0.05,
    interaction_designator: str = "*",
) -> AssignmentResult:
    if fit_object.weights is None or fit_object.bh_fdrs is None:
        raise ValueError("fit_edgepython must be run before assignment.")

    combo_x, fdrs_no_cis, fdrs_no_trans = _combo_projection_matrix(
        fit_object,
        combo=combo,
        interaction_designator=interaction_designator,
    )

    pred_log = fit_object.weights.to_numpy(dtype=float) @ combo_x.T + np.log(cell_size)
    pred_counts = np.exp(pred_log)

    p1 = pred_counts[:, 0]
    p2 = pred_counts[:, 1]
    h1 = pred_counts[:, 2]
    h2 = pred_counts[:, 3]

    df = pd.DataFrame(
        {
            "gene": fit_object.counts.index.to_numpy(),
            "P1": p1,
            "P2": p2,
            "H1": h1,
            "H2": h2,
            "Parlog2FC": np.log2(p1 / p2),
            "Hyblog2FC": np.log2(h1 / h2),
            "fdr_cis": fdrs_no_cis.to_numpy(dtype=float),
            "fdr_trans": fdrs_no_trans.to_numpy(dtype=float),
        },
        index=fit_object.counts.index,
    )

    p_values = df["Parlog2FC"].to_numpy(dtype=float)
    h_values = df["Hyblog2FC"].to_numpy(dtype=float)
    delta = p_values - h_values

    cis_index = (df["fdr_cis"] < alpha) & (df["fdr_trans"] > alpha)
    trans_index = (df["fdr_cis"] > alpha) & (df["fdr_trans"] < alpha)
    cis_plus_trans_index = (
        (df["fdr_cis"] < alpha)
        & (df["fdr_trans"] < alpha)
        & (((delta > 0) & (h_values > 0)) | ((delta < 0) & (h_values < 0)))
    )
    cis_x_trans_index = (
        (df["fdr_cis"] < alpha)
        & (df["fdr_trans"] < alpha)
        & (((delta <= 0) & (h_values >= 0)) | ((delta >= 0) & (h_values <= 0)))
    )

    df["colors"] = "lightgray"
    df.loc[cis_index, "colors"] = "orangered"
    df.loc[trans_index, "colors"] = "royalblue"
    df.loc[cis_plus_trans_index, "colors"] = "skyblue"
    df.loc[cis_x_trans_index, "colors"] = "forestgreen"

    df["reg_assignment"] = "conserved"
    df.loc[cis_index, "reg_assignment"] = "cis"
    df.loc[trans_index, "reg_assignment"] = "trans"
    df.loc[cis_plus_trans_index, "reg_assignment"] = "cis+trans"
    df.loc[cis_x_trans_index, "reg_assignment"] = "cisxtrans"

    theta_scaled = (2.0 / np.pi) * np.arctan2(h_values, delta)
    radius = np.sqrt((delta ** 2) + (h_values ** 2))
    cis_prop_reordered = theta_scaled - 0.5
    cis_prop_reordered = np.where(cis_prop_reordered > 1.0, cis_prop_reordered - 2.0, cis_prop_reordered)
    cis_prop_reordered = np.where(cis_prop_reordered <= -1.0, cis_prop_reordered + 2.0, cis_prop_reordered)

    df["R"] = radius
    df["theta_scaled"] = theta_scaled
    df["cis_prop"] = np.abs(theta_scaled)
    df["cis_prop_reordered"] = cis_prop_reordered

    if not make_plot:
        return AssignmentResult(dataframe=df, figure=None)

    fig, axes = plt.subplots(3, 1, figsize=(8, 14))

    axes[0].scatter(df["Parlog2FC"], df["Hyblog2FC"], s=8, alpha=0.7, c=df["colors"])
    axes[0].axline((0, 0), slope=1.0, color="orangered")
    axes[0].axline((0, 0), slope=0.5, color="skyblue")
    axes[0].axline((0, 0), slope=-5.0, color="forestgreen")
    axes[0].axvline(0.0, color="black")
    axes[0].axhline(0.0, color="darkblue")
    axes[0].set_xlabel("Parental log2 fold change")
    axes[0].set_ylabel("Hybrid log2 fold change")

    axes[1].scatter(delta, h_values, s=8, alpha=0.7, c=df["colors"])
    axes[1].axvline(0.0, color="orangered")
    axes[1].axhline(0.0, color="darkblue")
    axes[1].axline((0, 0), slope=1.0, color="skyblue")
    axes[1].axline((0, 0), slope=-1.0, color="forestgreen")
    axes[1].set_xlabel("R_P - R_H")
    axes[1].set_ylabel("R_H")

    axes[2].scatter(df["cis_prop_reordered"], p_values, s=8, alpha=0.7, c=df["colors"])
    for x_value, color in [(-1.0, "forestgreen"), (-0.5, "darkblue"), (0.0, "skyblue"), (0.5, "orangered"), (1.0, "forestgreen")]:
        axes[2].axvline(x_value, color=color)
    axes[2].axhline(0.0, color="black")
    axes[2].set_xlabel("Proportion cis")
    axes[2].set_ylabel("R_P")
    axes[2].set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0], [0.5, 0.0, 0.5, 1.0, 0.5])

    if combo:
        fig.suptitle(combo)
    fig.tight_layout()
    return AssignmentResult(dataframe=df, figure=fig)


def plot_regulatory_histogram(
    df: pd.DataFrame,
    title: str = "Regulatory Categories Histogram",
) -> plt.Figure:
    categories = ["cis", "trans", "cisxtrans", "cis+trans", "conserved"]
    colors = ["orangered", "darkblue", "forestgreen", "skyblue", "gray"]
    counts = df["reg_assignment"].value_counts().reindex(categories, fill_value=0)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(categories)), counts.to_numpy(), color=colors)
    for index, value in enumerate(counts.to_numpy()):
        ax.text(index, value, str(int(value)), ha="center", va="bottom")
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels([])
    ax.set_xlabel("Regulatory assignment")
    ax.set_ylabel("Number of genes")
    ax.set_title(title)
    return fig
